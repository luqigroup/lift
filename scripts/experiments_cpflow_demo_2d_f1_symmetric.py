"""CP-Flow 2-D toy sweep with the initialization matched across recipes.

A wrapper around ``experiments_cpflow_demo_2d_f1.py`` that perturbs each
recipe's own pre-positivity parameter by the same per-element
``N(0, sigma_bh^2)``, so the recipes differ only in the parameterization.

Usage:

    python scripts/experiments_cpflow_demo_2d_f1_symmetric.py \
        --start_seed 0 --n_seeds 100 \
        --targets eight_gaussians two_spirals \
        --backends direct hypernet

``--smoke`` runs one seed for 200 iterations.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from projorg import checkpointsdir

# Targets, CP-Flow math, DirectICNNParam, build_icnn_template, the
# hypernet bias randomizer and the per-run training loop are reused from
# the original driver.
from experiments_cpflow_demo_2d_f1 import (
    DirectICNNParam,
    HyperNetwork,
    build_icnn_template,
    icnn_pos_param_names,
    model_log_prob,
    randomise_hypernet_pos_bias,
    sample_target,
)
from lift.models.icnn import ICNN


# ------------------------------------------------- the PGD parameterization
class PGDICNNParam(torch.nn.Module):
    """ICNN weights held raw and projected onto the non-negative cone.

    Projected gradient descent (Amos et al., 2017): the constrained weights
    are ordinary parameters, the optimizer steps on them unconstrained, and
    every step is followed by ``W <- max(W, 0)``.
    ``ICNN(pos_constraint_mode="clamp")`` returns the raw weights and leaves
    that projection to the caller, which is :meth:`project`.

    The initialization matches direct softplus by construction: both modes
    draw the same folded-normal positive weights and the softplus mode
    stores ``softplus^-1`` of them, so the two start from
    identically-distributed effective weights.

    ``forward`` ignores its argument and mirrors the dict signature of
    ``DirectICNNParam`` and ``HyperNetwork``, so the CP-Flow code path
    downstream is the same for all three recipes.
    """

    def __init__(
        self,
        D: int,
        hidden_dim: int,
        nlayers: int,
        strong_convexity: float,
    ) -> None:
        super().__init__()
        self._inner = ICNN(
            input_size=D,
            hidden_dim=hidden_dim,
            nlayers=nlayers,
            activation="softplus",
            strong_convexity=strong_convexity,
            pos_constraint_mode="clamp",
        )
        self._pos_names = icnn_pos_param_names(nlayers, prefix="")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: p for name, p in self._inner.named_parameters()}

    def project(self) -> None:
        """Project onto the non-negative cone. Call after ``opt.step()``."""
        self._inner.clamp_weights()


def randomise_pgd_pos_bias(
    pgd_module: "PGDICNNParam",
    sigma_bh: float,
    generator: torch.Generator,
) -> None:
    """The PGD counterpart of :func:`randomise_direct_pos_bias`.

    Each recipe is perturbed in its own raw parameter space by the same
    per-element ``N(0, sigma_bh^2)``: the lift perturbs the readout bias and
    direct softplus perturbs ``W_raw``, both arguments of a softplus. PGD
    has no softplus, so its perturbation lands in weight space and is
    followed by the projection it applies after every step. The three
    perturbations are matched in each recipe's own coordinates and not in
    effective-weight space, where the two softplus recipes compress theirs
    and PGD does not.
    """
    inner = pgd_module._inner
    pos_names = set(pgd_module._pos_names)
    with torch.no_grad():
        for name, p in inner.named_parameters():
            if name not in pos_names:
                continue
            noise = torch.empty_like(p)
            noise.normal_(mean=0.0, std=float(sigma_bh), generator=generator)
            p.add_(noise)
    pgd_module.project()


# ------------ symmetric-init helper: the direct-softplus counterpart
def randomise_direct_pos_bias(
    direct_module: "DirectICNNParam",
    sigma_bh: float,
    generator: torch.Generator,
) -> None:
    """Add a per-element ``N(0, sigma_bh^2)`` offset to the positivity-tagged
    raw weights of direct softplus, the counterpart of
    :func:`randomise_hypernet_pos_bias`.

    ``DirectICNNParam`` wraps an :class:`ICNN` in
    ``pos_constraint_mode="softplus"``, so each positivity-tagged weight is
    ``W = softplus(W_raw)`` and ``W_raw`` plays the role of the hypernet
    readout head's pre-softplus argument. The perturbation is added on top
    of the folded-normal initialization rather than replacing it, mirroring
    the hypernet path, where the randomizer touches only the bias term and
    the pre-softplus argument keeps its initialization-scale mean.
    """
    inner = direct_module._inner
    pos_names = set(direct_module._pos_names)
    with torch.no_grad():
        for name, p in inner.named_parameters():
            if name not in pos_names:
                continue
            noise = torch.empty_like(p)
            noise.normal_(mean=0.0, std=float(sigma_bh), generator=generator)
            p.add_(noise)


# ------------------------------------- per-run trainer (symmetric init)
def train_one_symmetric(
    target: str,
    backend: str,
    seed: int,
    *,
    D: int = 2,
    hidden_dim: int = 64,
    nlayers: int = 3,
    strong_convexity: float = 0.05,
    hyper_hidden: Tuple[int, ...] = (192, 192, 256),
    sigma_bh: float = 0.5,
    n_iters: int = 6000,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_test: int = 1024,
    eval_every: int = 200,
    device: torch.device = torch.device("cpu"),
    log_every: int = 1000,
    save_train_history: bool = True,
    snap_every: int = 0,
) -> dict:
    """As ``experiments_cpflow_demo_2d_f1.train_one``, except that every
    backend receives the ``N(0, sigma_bh^2)`` perturbation of its own
    pre-positivity parameter.

    All three backends seed the perturbation generator with the same offset
    ``+13_579``, so they draw from identically seeded streams.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    template = build_icnn_template(
        D=D, hidden_dim=hidden_dim, nlayers=nlayers,
        strong_convexity=strong_convexity,
    ).to(device)
    pos_names = icnn_pos_param_names(nlayers, prefix="")

    if backend == "direct":
        param_module: torch.nn.Module = DirectICNNParam(
            D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
        ).to(device)
        # The same N(0, sigma_bh^2) perturbation the hypernet gets, from a
        # generator with the same seed offset.
        g_init = torch.Generator(device=device)
        g_init.manual_seed(int(seed) + 13_579)
        randomise_direct_pos_bias(param_module, sigma_bh, g_init)
    elif backend == "hypernet":
        param_module = HyperNetwork(
            input_size=D,
            hidden_sizes=list(hyper_hidden),
            downstream_network=template,
            use_outer_net=False,
            pos_param_names=pos_names,
            pos_constraint_mode="softplus",
        ).to(device)
        g_init = torch.Generator(device=device)
        g_init.manual_seed(int(seed) + 13_579)
        randomise_hypernet_pos_bias(param_module, sigma_bh, g_init)
    elif backend == "pgd":
        param_module = PGDICNNParam(
            D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
        ).to(device)
        g_init = torch.Generator(device=device)
        g_init.manual_seed(int(seed) + 13_579)
        randomise_pgd_pos_bias(param_module, sigma_bh, g_init)
    else:
        raise ValueError(f"unknown backend: {backend}")

    opt = torch.optim.Adam(param_module.parameters(), lr=lr)

    test_seed_g = torch.Generator(device=device)
    test_seed_g.manual_seed(int(seed) + 9999)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 9999)
    with torch.no_grad():
        x_test = sample_target(target, n_test, device).detach()

    history: Dict[str, list] = {
        "iter": [], "train_nll": [],
        "eval_iter": [], "test_nll": [],
    }

    # Parameter snapshots for the landscape trajectory overlays;
    # ``snap_every <= 0`` disables them. The ``direct`` backend saves the
    # effective PICNN parameters (softplus applied to the positivity-tagged
    # weights); ``hypernet`` saves both the parameters emitted at a fixed
    # reference batch ``x_ref`` and the raw hypernet state dict.
    # ``render_landscape_cpflow_2d.py`` reads ``picnn_snaps`` for the
    # constrained-theta panel and ``hyper_snaps`` for the lifted panel.
    picnn_snaps: list = []
    hyper_snaps: list = []
    snap_iters: list = []
    x_ref = None
    if snap_every > 0:
        with torch.no_grad():
            x_ref = sample_target(target, batch_size, device).detach()

    t0 = time.time()
    diverged_at = None
    for it in range(1, n_iters + 1):
        x_batch = sample_target(target, batch_size, device)
        params = param_module(x_batch)
        logp = model_log_prob(template, params, x_batch)
        loss = -logp.mean()
        # A step-size sweep drives every recipe past its stable range on
        # purpose, so a non-finite loss is a measurement rather than a
        # crash: record the iteration it happened at and stop the run,
        # which would otherwise finish on NaN with no record of when it
        # broke.
        if not torch.isfinite(loss):
            diverged_at = int(it)
            print(
                f"[{target}|{backend}|seed{seed}] DIVERGED at it={it} "
                f"(non-finite training loss); stopping this arm",
                flush=True,
            )
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_module.parameters(), max_norm=10.0)
        opt.step()
        if backend == "pgd":
            param_module.project()

        if save_train_history:
            history["iter"].append(int(it))
            history["train_nll"].append(float(loss.item()))

        # Parameter snapshots; empty when snap_every <= 0.
        if snap_every > 0 and (
            it == 1 or it % snap_every == 0 or it == n_iters
        ):
            with torch.no_grad():
                if backend == "hypernet":
                    # Emit at the fixed reference batch so trajectory
                    # PICNN params live in a common coordinate system.
                    emitted = {
                        k: v.detach().cpu().clone()
                        for k, v in param_module(x_ref).items()
                    }
                    picnn_snaps.append(emitted)
                    hyper_snaps.append({
                        k: v.detach().cpu().clone()
                        for k, v in param_module.state_dict().items()
                    })
                else:
                    # For direct softplus the call returns the effective
                    # ICNN parameters, softplus applied to the
                    # positivity-tagged weights, whatever the input; for
                    # PGD it returns the projected weights.
                    emitted = {
                        k: v.detach().cpu().clone()
                        for k, v in param_module(
                            x_ref if x_ref is not None
                            else x_batch.detach()
                        ).items()
                    }
                    picnn_snaps.append(emitted)
                snap_iters.append(int(it))

        if (it % eval_every == 0) or (it == 1):
            param_module.eval()
            params_eval = {k: v.detach() for k, v in
                           param_module(x_test).items()}
            x_test_g = x_test.clone().requires_grad_(True)
            logp_test = model_log_prob(template, params_eval, x_test_g)
            test_nll = float(-logp_test.detach().mean().item())
            history["eval_iter"].append(int(it))
            history["test_nll"].append(test_nll)
            param_module.train()

        if (it % log_every == 0) or (it == 1):
            print(
                f"[{target}|{backend}|seed{seed}] it={it:5d} "
                f"train_nll={loss.item():.4f} "
                f"test_nll={history['test_nll'][-1]:.4f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    param_module.eval()
    with torch.no_grad():
        params_final = {k: v.detach() for k, v in
                        param_module(x_test).items()}
    x_test_g = x_test.clone().requires_grad_(True)
    logp_test = model_log_prob(template, params_final, x_test_g)
    final_test_nll = float(-logp_test.detach().mean().item())
    per_sample_nll = (-logp_test.detach()).cpu().numpy().astype(np.float32)
    import math
    se_mean = float(per_sample_nll.std(ddof=1) / math.sqrt(len(per_sample_nll)))

    return {
        "history": history,
        "test_nll": final_test_nll,
        "test_nll_se": se_mean,
        # None when the run reached n_iters; otherwise the iteration the
        # training loss first went non-finite.
        "diverged_at_iter": diverged_at,
        # State dicts so a renderer can replay a per-seed run. The
        # ``template`` is the frozen clamp-mode ICNN whose Brenier map
        # ``T = grad phi`` defines the CP-Flow used by ``model_log_prob``.
        "param_module_state": {k: v.detach().cpu()
                               for k, v in param_module.state_dict().items()},
        "template_state": {k: v.detach().cpu()
                           for k, v in template.state_dict().items()},
        # ``picnn_snaps`` is the constrained-theta trajectory of either
        # backend, ``hyper_snaps`` the lifted trajectory of the hypernet,
        # and ``x_ref`` the frozen reference batch the emission was
        # evaluated at.
        "picnn_snaps": picnn_snaps,
        "hyper_snaps": hyper_snaps,
        "snap_iters": snap_iters,
        "snap_every": int(snap_every),
        "x_ref": (x_ref.detach().cpu() if x_ref is not None else None),
        "config": dict(
            target=target, backend=backend, seed=seed,
            D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
            hyper_hidden=list(hyper_hidden),
            sigma_bh=float(sigma_bh),
            n_iters=n_iters, batch_size=batch_size, lr=lr, n_test=n_test,
            symmetric_init=True,
        ),
        "wall_seconds": float(time.time() - t0),
    }


# ----------------------------------------------------------------- driver
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment", default="experiments_cpflow_demo_2d_f1_symmetric",
    )
    p.add_argument("--n_iters", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--strong_convexity", type=float, default=0.05)
    p.add_argument("--hyper_hidden", type=int, nargs="+",
                   default=[192, 192, 256])
    p.add_argument("--sigma_bh", type=float, default=0.5)
    p.add_argument("--n_test", type=int, default=1024)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--targets", nargs="+",
                   default=["eight_gaussians", "two_spirals"])
    p.add_argument("--backends", nargs="+", default=["direct", "hypernet"])
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--no_train_history", action="store_true")
    p.add_argument("--snap_every", type=int, default=0,
                   help="If >0, snapshot param state every K iters into "
                        "picnn_snaps (+ hyper_snaps for hypernet backend) "
                        "for the Fig-16 loss-landscape renderer.")
    p.add_argument("--auto_snap_target", type=int, default=0,
                   help="If >0 and --snap_every == 0, auto-set snap_every "
                        "to max(1, n_iters // auto_snap_target).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip (target, backend, seed) triples whose "
                        "test_nll.json already exists.")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n_iters = 200
        args.batch_size = 64
        args.start_seed = 0
        args.n_seeds = 1
        args.eval_every = 50
        args.log_every = 50

    if args.snap_every <= 0 and args.auto_snap_target > 0:
        args.snap_every = max(1, int(args.n_iters) // int(args.auto_snap_target))

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))

    if args.device == "auto":
        device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[cpflow-demo-f1-symmetric] device={device}", flush=True)

    ckpt_dir = checkpointsdir(args.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[cpflow-demo-f1-symmetric] ckpt_dir={ckpt_dir}", flush=True)
    print(f"[cpflow-demo-f1-symmetric] SYMMETRIC INIT: both arms get "
          f"N(0,{args.sigma_bh}^2) readout-bias randomisation", flush=True)
    print(f"[cpflow-demo-f1-symmetric] seeds=[{seeds[0]}..{seeds[-1]}] "
          f"({len(seeds)} seeds), targets={args.targets}, "
          f"backends={args.backends}", flush=True)

    summary: Dict[str, Dict[str, Dict[str, dict]]] = {}
    n_total = len(args.targets) * len(args.backends) * len(seeds)
    n_done = 0
    n_skipped = 0
    t0_all = time.time()
    for target in args.targets:
        summary[target] = {}
        for backend in args.backends:
            summary[target][backend] = {}
            for s in seeds:
                tag = f"{target}_seed{s}_{backend}"
                nll_path = os.path.join(ckpt_dir, f"{tag}_test_nll.json")
                if args.skip_existing and os.path.exists(nll_path):
                    with open(nll_path) as f:
                        rec = json.load(f)
                    summary[target][backend][str(s)] = {
                        "test_nll": rec["test_nll"],
                        "test_nll_se": rec["test_nll_se"],
                        "diverged_at_iter": rec.get("diverged_at_iter"),
                        "wall_seconds": rec.get("wall_seconds", 0.0),
                    }
                    n_skipped += 1
                    n_done += 1
                    continue

                n_done += 1
                print(
                    f"\n==== [{n_done}/{n_total}] target={target} "
                    f"backend={backend} seed={s} (elapsed_all="
                    f"{(time.time()-t0_all)/60.0:.1f} min) ====",
                    flush=True,
                )
                out = train_one_symmetric(
                    target=target, backend=backend, seed=s,
                    hidden_dim=args.hidden_dim, nlayers=args.nlayers,
                    strong_convexity=args.strong_convexity,
                    hyper_hidden=tuple(args.hyper_hidden),
                    sigma_bh=args.sigma_bh,
                    n_iters=args.n_iters, batch_size=args.batch_size,
                    lr=args.lr, n_test=args.n_test,
                    eval_every=args.eval_every, log_every=args.log_every,
                    device=device,
                    save_train_history=not args.no_train_history,
                    snap_every=int(args.snap_every),
                )

                payload = {
                    "history": out["history"],
                    "test_nll": out["test_nll"],
                    "test_nll_se": out["test_nll_se"],
                    "param_module_state": out["param_module_state"],
                    "template_state": out["template_state"],
                    "picnn_snaps": out.get("picnn_snaps", []),
                    "hyper_snaps": out.get("hyper_snaps", []),
                    "snap_iters": out.get("snap_iters", []),
                    "snap_every": out.get("snap_every", 0),
                    "x_ref": out.get("x_ref", None),
                    "config": out["config"],
                    "wall_seconds": out["wall_seconds"],
                }
                torch.save(payload, os.path.join(ckpt_dir, f"{tag}.pt"))
                if int(out.get("snap_every", 0)) > 0:
                    print(f"[{tag}] saved {len(out.get('snap_iters', []))} "
                          f"snapshots (snap_every={out.get('snap_every')})",
                          flush=True)
                with open(nll_path, "w") as f:
                    json.dump(
                        {
                            "target": target, "backend": backend, "seed": s,
                            "test_nll": out["test_nll"],
                            "test_nll_se": out["test_nll_se"],
                            "diverged_at_iter": out.get("diverged_at_iter"),
                            "wall_seconds": out["wall_seconds"],
                            "config": out["config"],
                        },
                        f, indent=2,
                    )
                summary[target][backend][str(s)] = {
                    "test_nll": out["test_nll"],
                    "test_nll_se": out["test_nll_se"],
                    "diverged_at_iter": out.get("diverged_at_iter"),
                    "wall_seconds": out["wall_seconds"],
                }
                print(
                    f"[{tag}] FINAL test_nll={out['test_nll']:.4f} "
                    f"+/- SE {out['test_nll_se']:.4f} "
                    f"wall={out['wall_seconds']:.1f}s",
                    flush=True,
                )

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target, backends in summary.items():
        agg[target] = {}
        for backend, seeds_d in backends.items():
            arr = np.asarray([v["test_nll"] for v in seeds_d.values()])
            agg[target][backend] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)) if len(arr) else 0.0,
                "q75": float(np.quantile(arr, 0.75)) if len(arr) else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_seeds": int(len(arr)),
                "per_seed": seeds_d,
            }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump({"per_seed": summary, "agg": agg,
                   "symmetric_init": True}, f, indent=2)
    print(f"\n[cpflow-demo-f1-symmetric] skipped {n_skipped} pre-existing "
          f"runs.", flush=True)
    print("==== SUMMARY (symmetric init; mean / median / std) ====",
          flush=True)
    for target, backends in agg.items():
        print(f"[{target}]")
        for backend, stats in backends.items():
            print(
                f"  {backend}: test_nll mean={stats['mean']:.4f} "
                f"median={stats['median']:.4f} std={stats['std']:.4f} "
                f"IQR=[{stats['q25']:.4f},{stats['q75']:.4f}] "
                f"min={stats['min']:.4f} max={stats['max']:.4f} "
                f"n={stats['n_seeds']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
