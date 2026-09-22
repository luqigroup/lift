"""Multi-target 1-D log-concave ICNN-EBM gallery.

Trains a log-concave ICNN-EBM on each of six 1-D targets (Gumbel, Laplace,
Gamma, Beta, half-normal, skew-normal) under one FKL objective and training
recipe. The driver is a loop over ``HypernetVsDirect1D``: each iteration sets
``args.target_kind`` and ``args.experiment_name`` and runs the existing training
and evaluation code, with the landscape slice skipped.

Each target writes into ``<out_dir>/<target>/`` the log-density grids, a
``metrics.json`` and the per-seed checkpoints ``seed<S>_<backend>.pt``. The
companion renderer ``scripts/render_lcmm_1d_logconcave.py`` glues them into the
panel gallery and the paired-TV bar chart.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D  # noqa


TARGETS = [
    "gumbel",
    "laplace",
    "gamma",
    "beta",
    "halfnormal",
    "skewnormal",
]


def _make_args(*, target: str, base: argparse.Namespace) -> argparse.Namespace:
    """Per-target args clone, with target-specific output directory."""
    args = copy.deepcopy(base)
    args.target_kind = target
    args.experiment_name = f"experiments_lcmm_1d_logconcave_{target}"
    # Write straight into out_dir/target/, so the renderer finds checkpoints
    # without the per-config hash routing.
    args.checkpoints_dir = os.path.join(args.out_dir, target)
    args.plots_dir = os.path.join(args.out_dir, target, "plots")
    args.logs_dir = os.path.join(args.out_dir, target, "logs")
    args.experiment = args.experiment_name
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    return args


def _patch_projorg_dirs(args):
    """Point ``projorg.checkpointsdir`` and ``plotsdir`` at the per-target dirs.

    The lib resolves its output root through ``projorg.checkpointsdir``, so the
    module-level functions are patched to return the dirs set on ``args``.
    """
    import projorg  # type: ignore

    args._stash_target_dir = args.checkpoints_dir

    def _cd(_name):
        return args.checkpoints_dir

    def _pd(_name):
        return args.plots_dir

    projorg.checkpointsdir = _cd  # type: ignore[attr-defined]
    projorg.plotsdir = _pd  # type: ignore[attr-defined]
    # Also patch the names the lib already imported.
    import _experiments_hypernet_vs_direct_1d_lib as _lib
    _lib.checkpointsdir = _cd
    _lib.plotsdir = _pd


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--out_dir", default="data/checkpoints/lcmm_1d_logconcave")
    p.add_argument("--n_iters", type=int, default=3000,
                   help="iters per (target, backend, seed); v1-stable default")
    p.add_argument("--seed", type=int, default=0,
                   help="single-seed convenience flag; superseded by --seeds if given")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed list, e.g. '0,1,2,...,29' for the 30-seed sweep")
    p.add_argument("--hyper_hidden_sizes", default="64,64,96",
                   help="v1-stable hypernet body; v2 wide [192,192,256] diverged earlier")
    p.add_argument("--n_batch_hyper", type=int, default=256,
                   help="v1-stable DeepSets batch; v2 B_cond=1 diverged earlier")
    # Emitter architecture switches. Every default reproduces the shipped
    # construction exactly, and only non-default values reach HyperNetwork.
    p.add_argument("--hyper_pool_norm", type=int, default=1,
                   help="standardize the pooled summary. 1 is the recipe the paper's "
                        "one-dimensional results are reported at; 0 is the unstandardized "
                        "pooling and fits these targets an order of magnitude worse")
    p.add_argument("--hyper_pool_scale", type=float, default=0.0,
                   help=">0: pin the pooled summary's norm to this value")
    p.add_argument("--hyper_outer_net", type=int, default=0,
                   help="1: three-layer MLP on the pooled summary before the readouts")
    p.add_argument("--hyper_head_bias", type=float, default=-2.0,
                   help="readout-bias init of the positivity-tagged heads (the slack init)")
    p.add_argument("--hyper_body_layernorm", type=int, default=0,
                   help="1: LayerNorm after every hidden layer of the per-sample encoder")
    p.add_argument("--hyper_noise_std", type=float, default=0.0,
                   help=">0: Gaussian noise on the pooled summary, annealed to zero at half training")
    p.add_argument("--init_mode", default="ours",
                   choices=["ours", "hk", "cpflow", "torch"],
                   help="constrained-path initialization: ours (the folded "
                        "normal the code ships), hk (Hoedt-Klambauer 2023, "
                        "log-normal mean 1.855/n_in with bias +0.74), cpflow "
                        "(the zero-centred latent of the CP-Flow recipe, "
                        "without its 1/fan_in gain), torch (plain "
                        "nn.Linear.reset_parameters everywhere)")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument(
        "--lr_ebm", type=float, default=1e-3,
        help="energy-model step size; the paper's gallery value is 1e-3. "
             "Raising it is the step-size robustness sweep "
             "(slurm/lowdim_stepsize_sweep.slurm); give each value its "
             "own --out_dir, which is the only thing separating cells "
             "in this projorg-free driver",
    )
    p.add_argument(
        "--n_batch_cond", type=int, default=0,
        help="conditioning-batch size n the emitter's mean-pool sees; 0 "
             "(default) means the whole gradient batch, which is what "
             "every paper run does",
    )
    p.add_argument(
        "--cond_source", default="independent",
        choices=["independent", "subbatch"],
        help="which n samples, when --n_batch_cond is set. 'subbatch' "
             "takes them from the gradient batch, so the emission stays "
             "correlated with the gradient and both channels of the "
             "diffusion excess are present; 'independent' draws a fresh "
             "batch, which zeroes the coupling and leaves the jitter. "
             "Run at a matched n these two differ in the coupling "
             "channel and nothing else",
    )
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument(
        "--resume", type=int, default=0,
        help="skip any (seed, backend) pair whose checkpoint is already in "
             "out_dir/<target>/ and read its metrics back from disk; the "
             "way to relaunch a sweep that ran out of wall clock",
    )
    p.add_argument(
        "--backends", default="hypernet,direct",
        help="comma-separated subset of {hypernet, direct, pgd}",
    )
    return p.parse_args()


def main():
    cli = _parse()
    targets = [t.strip() for t in cli.targets.split(",") if t.strip()]
    if cli.smoke:
        cli.n_iters = 100

    if cli.seeds is not None:
        seed_list = [int(s) for s in cli.seeds.split(",") if s.strip() != ""]
    else:
        seed_list = [int(cli.seed)]
    print(f"[driver] running targets={targets} seeds={seed_list}")

    # Build the "rich" args namespace that ``HypernetVsDirect1D`` expects.
    base = argparse.Namespace(
        target_kind="gumbel",  # overwritten per loop iteration
        objective="fkl-direct",
        # This Namespace lists every field explicitly, so a flag added to the
        # parser but not added here is silently dropped and the lib's getattr
        # default wins. Any new CLI flag must be threaded through here too.
        init_mode=str(cli.init_mode),
        n_iters=int(cli.n_iters),
        lr_ebm=float(cli.lr_ebm),
        lr_sampler=5e-3,
        hidden_dim=128,
        nlayers=3,
        hyper_hidden_sizes=[int(s) for s in cli.hyper_hidden_sizes.split(",")],
        sampler_hyper_hidden_sizes=[64, 64, 96],
        strong_convexity=0.02,
        n_batch_hyper=int(cli.n_batch_hyper),
        n_batch_val=4096,
        eval_every=200,
        grad_clip=1.0,
        tv_grid_n=8000,
        flow_D_aug=4,
        flow_n_hidden=64,
        flow_n_layers=4,
        flow_n_mlp_layers=3,
        T_sampler=1,
        T_sampler_max=30,
        n_sampler_warmup_iters=500,
        sampler_kl_threshold=0.1,
        M_kl_check=256,
        eta_svgd=1.0,
        M_particles=1024,
        M_logZ=1024,
        M_logZ_eval=8192,
        cond_dim=8,
        sampler_objective="rkl",
        landscape_grid_n=1,
        skip_landscape=1,
        landscape_radius=0.0,
        landscape_margin=1.35,
        direction_mode="trajectory",
        seeds=seed_list,
        seed=seed_list[0],
        hyper_extra_kwargs={
            k: v for k, v in dict(
                pool_norm=bool(cli.hyper_pool_norm),
                pool_scale=float(cli.hyper_pool_scale),
                use_outer_net=bool(cli.hyper_outer_net),
                head_init_pos_bias=float(cli.hyper_head_bias),
                body_layernorm=bool(cli.hyper_body_layernorm),
            ).items()
            if v != dict(pool_norm=False, pool_scale=0.0,
                         use_outer_net=False, head_init_pos_bias=-2.0,
                         body_layernorm=False)[k]
        },
        hyper_noise_std=float(cli.hyper_noise_std),
        gpu_id=int(cli.gpu_id),
        phase="train",
        # The lib saves ``seed<S>_<backend>.pt`` as each run finishes and,
        # under ``resume``, reads a finished run's metrics back from it
        # instead of retraining.
        resume=int(cli.resume),
        # None means the emitter reads the whole gradient batch; see
        # --cond_source for what the two sources mean.
        n_batch_cond=(int(cli.n_batch_cond)
                      if int(cli.n_batch_cond) > 0 else None),
        cond_source=str(cli.cond_source),
        K=1,
        backends=[b.strip() for b in cli.backends.split(",") if b.strip()],
        out_dir=cli.out_dir,
        # Per-target defaults (override in the lib via target-specific HPs):
        laplace_mu=0.0,
        laplace_b=1.0,
        gamma_alpha=2.0,
        gamma_beta=1.0,
        beta_alpha=2.0,
        beta_beta=5.0,
        halfnormal_sigma=1.0,
        skewnormal_xi=0.0,
        skewnormal_omega=1.0,
        skewnormal_alpha=4.0,
        experiment_name="experiments_lcmm_1d_logconcave",
        experiment="experiments_lcmm_1d_logconcave",
    )

    os.makedirs(cli.out_dir, exist_ok=True)
    aggregate = {"per_target": {}}
    t_start = time.time()
    for tgt in targets:
        print(f"\n##################### TARGET = {tgt} ####################")
        args = _make_args(target=tgt, base=base)
        _patch_projorg_dirs(args)
        exp = HypernetVsDirect1D(args, target_kind=tgt)
        exp.train()
        # The lib writes metrics.json into args.checkpoints_dir.
        mfile = os.path.join(args.checkpoints_dir, "metrics.json")
        with open(mfile) as f:
            m = json.load(f)
        aggregate["per_target"][tgt] = m
        elapsed = time.time() - t_start
        print(f"[{tgt}] done; cumulative wall={elapsed:.1f}s")
    with open(os.path.join(cli.out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {os.path.join(cli.out_dir, 'aggregate.json')}")


if __name__ == "__main__":
    main()
