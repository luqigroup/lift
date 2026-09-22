"""Re-evaluate a trained lifted EBM with its emission pinned on all training data.

The driver reads the lift with its weights emitted from one conditioning
batch of ``n_batch_hyper`` training samples (``_eval_density`` in
``_experiments_hypernet_vs_direct_1d_lib.py``). Here the DeepSets summary,
a mean of per-sample encodings, is accumulated in chunks over every row and
mapped through ``HyperNetwork.emit_from_pooled`` -- the same code path
training uses after its own pooling (norm pinning, standardization,
readouts), so nothing can drift. The scored object is then one density
chosen before any held-out point is seen, and the metrics are the lib's: the
TV / Hellinger / KL grid of ``grid_metrics_1d`` in one dimension, the
held-out test NLL with the flow-IS normalizer above it.

Both readings are recorded per seed -- ``pinned`` (all training data) and
``batch`` (the as-trained conditioning batch, resampled with a fixed seed).
For one-dimensional archives the hypernet snapshots stored in each
checkpoint are re-read the same way, giving the TV-along-training curve with
the pinned emission at every stored iterate.

Post-hoc tool: reads the archive's ``seed{s}_hypernet.pt`` files directly and
does not go through ``projorg.setup_environment``, since minting a fresh
config hash would point at an empty directory. Architecture comes from the
driver's config JSON plus ``key-value`` tokens parsed from a projorg
directory name, overridable with ``--set``; a wrong architecture fails loudly
at ``load_state_dict``.

Usage::

    python scripts/reeval_ebm_pinned.py \\
        --archive data/checkpoints/lcmm_1d_logconcave_30seed_poolnorm/gumbel \\
        --target_kind gumbel --config configs/experiments_lcmm_1d_logconcave.json \\
        --pool_norm 1 --snapshots 1 --out data/analysis/pinned/lcmm_1d_poolnorm_gumbel
    python scripts/reeval_ebm_pinned.py \\
        --archive 'data/_fetch_tmp/uci_hepmass_10seed_ramp_poolnorm_*' \\
        --target_kind uci_hepmass --config configs/experiments_pgd_uci_hepmass.json \\
        --pool_norm 1 --seeds 0 1 2 --out data/analysis/pinned/uci_hepmass_lift
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from _experiments_hypernet_vs_direct_1d_lib import (  # noqa: E402
    HypernetVsDirect1D, grid_metrics_1d,
)
from lift.models.conditional_sampler_flow import ConditionalSamplerFlow  # noqa: E402
from lift.objectives.builders import build_hypernet_ebm  # noqa: E402
from lift.objectives.eval import build_eval_grid, hypernet_ebm_logp  # noqa: E402
from lift.objectives.partition import log_Z_via_grid_1d  # noqa: E402

# projorg encodes every config field as ``key-value`` in the directory name;
# these are the ones the rebuild depends on.
_NAME_KEYS = {
    "hidden_dim": int, "nlayers": int, "strong_convexity": float,
    "flow_D_aug": int, "flow_n_hidden": int, "flow_n_layers": int,
    "n_batch_hyper": int, "M_logZ_eval": int, "tv_grid_n": int,
    "hyper_pool_norm": int,
}


class PinnedEmitter(nn.Module):
    """A stand-in for ``hyper_E`` whose forward returns fixed emitted weights.

    ``hypernet_ebm_logp`` calls ``hyper_E(target_sample)`` once and reads the
    parameter dtype off ``hyper_E.parameters()``; this module satisfies both
    with the pinned emission, so the evaluator's own code path scores it.
    """

    def __init__(self, params: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._keys = list(params)
        for i, k in enumerate(self._keys):
            self.register_buffer(f"p{i}", params[k].detach().clone())
        self._dtype_anchor = nn.Parameter(
            torch.zeros((), dtype=next(iter(params.values())).dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: ARG002
        return {k: getattr(self, f"p{i}") for i, k in enumerate(self._keys)}


def _args_from(config: str, archive_dir: str, sets: List[str]) -> SimpleNamespace:
    with open(config) as f:
        cfg = json.load(f)
    name = os.path.basename(archive_dir.rstrip("/"))
    for key, typ in _NAME_KEYS.items():
        # Greedy number, ended by the next ``_key`` or by the end of the
        # name: a lazy match captures "0" out of "strong_convexity-0.02".
        m = re.search(rf"(?:^|_){re.escape(key)}-(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)(?=_[A-Za-z]|$)", name)
        if m:
            cfg[key] = typ(float(m.group(1))) if typ is int else typ(m.group(1))
    for s in sets:
        k, v = s.split("=", 1)
        cfg[k] = type(cfg[k])(v) if k in cfg and not isinstance(cfg[k], str) else v
    for k in ("hyper_hidden_sizes", "sampler_hyper_hidden_sizes"):
        if isinstance(cfg.get(k), str):
            cfg[k] = [int(t) for t in cfg[k].split(",") if t]
    cfg.setdefault("gpu_id", 0)
    return SimpleNamespace(**cfg)


def _pooled_params(hyper_E, x_all: torch.Tensor, chunk: int) -> Dict[str, torch.Tensor]:
    """The emission from the mean per-sample encoding over all rows of ``x_all``."""
    hyper_E.eval()
    with torch.no_grad():
        acc, n = None, 0
        for k in range(0, x_all.shape[0], chunk):
            z = hyper_E.inner_net(x_all[k:k + chunk].reshape(-1, hyper_E.input_size))
            s = z.sum(dim=0, keepdim=True)
            acc = s if acc is None else acc + s
            n += int(z.shape[0])
        return {k: v.detach() for k, v in hyper_E.emit_from_pooled(acc / n).items()}


def _score(ebm, emitter, flow, target, args, device, D: int) -> Dict[str, float]:
    """The lib's final metrics for a fixed emission (mirrors ``_eval_density``)."""
    K = 1
    D_aug = int(args.flow_D_aug)
    n_grid = int(getattr(args, "tv_grid_n", 0))   # unused above one dimension
    dummy = torch.zeros(1, D, device=device)
    with torch.no_grad():
        log_Z = None
        if D == 1:
            xs, _, dvol = build_eval_grid(target, n_grid, device)
            log_Z = log_Z_via_grid_1d(ebm, emitter(dummy), K=K, grid=xs, dvol=float(dvol))
    M_logZ_eval = int(getattr(args, "M_logZ_eval", None) or args.M_logZ)
    logp_fn = hypernet_ebm_logp(ebm, emitter, flow, K, D, D_aug,
                                target_sample=dummy, M_logZ=M_logZ_eval, device=device,
                                log_Z_override=log_Z)
    if D == 1:
        m = grid_metrics_1d(target.log_prob_np, logp_fn, *target.xlim, n=n_grid)
        xs_np = np.linspace(target.xlim[0], target.xlim[1], n_grid)
        m["log_p_grid"] = np.asarray(logp_fn(xs_np), dtype=np.float64)
        return m
    test = np.asarray(target.test_data, dtype=np.float64)
    return {"test_nll": float(-np.asarray(logp_fn(test)).mean())}


def _run_ensemble(a, device) -> None:
    """Pinned re-read of the per-digit MNIST ensemble, one lifted EBM per class."""
    from lift.dataset.mnist_latent import MNISTLatentClassConditional
    d = sorted(x for x in glob.glob(a.archive) if os.path.isdir(x))[0]
    args = _args_from(a.config, d, a.set)
    ae_path = getattr(args, "ae_path", "") or None
    out_json: Dict[str, object] = {"archive": d, "ensemble": True, "classes": {}}
    for c in range(10):
        path = os.path.join(d, f"lcmm_class{c}.pth")
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        D = int(args.latent_dim)
        ebm, hyper_E = build_hypernet_ebm(K=1, D=D, hidden_dim=int(args.hidden_dim), nlayers=int(args.nlayers),
                                          strong_convexity=float(args.strong_convexity),
                                          hyper_hidden_sizes=[int(h) for h in args.hyper_hidden_sizes])
        hyper_E.load_state_dict(state["hyper_E"])
        flow = ConditionalSamplerFlow(K=1, D=D, D_aug=int(args.flow_D_aug), n_hidden=int(args.flow_n_hidden),
                                      n_flow_layers=int(args.flow_n_layers), n_mlp_layers=int(getattr(args, "flow_n_mlp_layers", 3)),
                                      cond_dim=int(getattr(args, "cond_dim", 8)))
        flow.load_state_dict(state["flow"])
        ebm, hyper_E, flow = ebm.to(device).eval(), hyper_E.to(device).eval(), flow.to(device).eval()
        target = MNISTLatentClassConditional(digit_class=c, ae_path=ae_path, latent_dim=D,
                                             hidden=int(args.ae_hidden), device=device)
        x_all = torch.from_numpy(np.asarray(target.train_data, dtype=np.float32)).to(device)
        pinned = PinnedEmitter(_pooled_params(hyper_E, x_all, a.chunk)).to(device)
        with torch.no_grad():
            torch.manual_seed(c)
            batch = PinnedEmitter(dict(hyper_E(target.sample(int(args.n_batch_hyper), device=device)))).to(device)
        m_pin = _score(ebm, pinned, flow, target, args, device, D)
        m_bat = _score(ebm, batch, flow, target, args, device, D)
        out_json["classes"][str(c)] = {"pinned": m_pin, "batch": m_bat, "n_train_rows": int(x_all.shape[0])}
        print(f"  class {c}  test_nll pinned {m_pin['test_nll']:.4f}  batch {m_bat['test_nll']:.4f}  (pinned on {x_all.shape[0]} latents)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"wrote {a.out}.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", required=True, help="directory (or glob of directories) holding seed*_hypernet.pt")
    p.add_argument("--target_kind", required=True)
    p.add_argument("--config", required=True, help="the driver's config JSON")
    p.add_argument("--set", action="append", default=[], help="key=value override of a config field")
    p.add_argument("--pool_norm", type=int, default=-1,
                   help="1/0: the lift standardises its pooled summary; -1 infers it from a "
                        "'poolnorm' token in the archive path. Parameter-free, so "
                        "load_state_dict cannot check it: state it for any archive whose "
                        "name does not carry it.")
    p.add_argument("--hyper_pool_scale", type=float, default=0.0,
                   help="the driver's --hyper_pool_scale of the archive (parameter-free; 0 in every shipped run)")
    p.add_argument("--hyper_noise_std", type=float, default=0.0,
                   help="the driver's --hyper_noise_std of the archive (parameter-free; 0 in every shipped run)")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--n_pin", type=int, default=1 << 17, help="rows pooled over for an analytic (sampled) target; a finite training split is pooled over entirely")
    p.add_argument("--chunk", type=int, default=1 << 16)
    p.add_argument("--snapshots", type=int, default=0, help="1: also re-read every stored hypernet snapshot (TV along training, one-dimensional targets)")
    p.add_argument("--out", required=True, help="output stem; writes <out>.json and <out>.npz")
    p.add_argument("--ensemble", type=int, default=0, help="1: --archive is a per-class MNIST ensemble dir (lcmm_class{c}.pth, hyper_E + flow state); one class-conditional latent target per digit, as experiments_mnist_latent_per_class_ensemble._load_class rebuilds it")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if a.ensemble:
        _run_ensemble(a, device); return
    dirs = sorted(d for d in glob.glob(a.archive) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no archive matches {a.archive!r}")
    pts = sorted(pt for d in dirs for pt in glob.glob(os.path.join(d, "seed*_hypernet.pt")))
    if a.seeds is not None:
        pts = [pt for pt in pts if int(re.search(r"seed(\d+)_", os.path.basename(pt)).group(1)) in set(a.seeds)]
    if not pts:
        raise SystemExit("no seed*_hypernet.pt found")
    args = _args_from(a.config, dirs[0], a.set)
    pool_norm = bool(a.pool_norm) if a.pool_norm >= 0 else ("poolnorm" in os.path.basename(dirs[0].rstrip("/")) or "poolnorm" in os.path.dirname(dirs[0]))
    exp = HypernetVsDirect1D(args, target_kind=a.target_kind)
    exp.device = device
    target = exp._build_target()
    D = int(getattr(target, "D", 1))
    print(f"archive(s): {len(dirs)}  seeds: {len(pts)}  target: {a.target_kind} (D={D})  pool_norm={pool_norm}  "
          f"pool_scale={a.hyper_pool_scale}  noise_std={a.hyper_noise_std}  strong_convexity={args.strong_convexity}  "
          f"arch: {args.hidden_dim}x{args.nlayers} body {args.hyper_hidden_sizes}  D_aug={args.flow_D_aug}", flush=True)

    # The full training data, once: a finite split entirely, an analytic target by a fixed large sample.
    if hasattr(target, "train_data"):
        x_all = torch.from_numpy(np.asarray(target.train_data, dtype=np.float32)).to(device)
        pin_desc = f"the full training split ({x_all.shape[0]} rows)"
    else:
        torch.manual_seed(2026)
        x_all = target.sample(int(a.n_pin), device=device).detach().float()
        pin_desc = f"{x_all.shape[0]} target samples (seed 2026)"
    print(f"pinned on {pin_desc}", flush=True)

    out_json: Dict[str, object] = {"archive": a.archive, "target_kind": a.target_kind, "pinned_on": pin_desc,
                                   "pool_norm": pool_norm, "seeds": {}}
    out_npz: Dict[str, np.ndarray] = {}
    for pt in pts:
        seed = int(re.search(r"seed(\d+)_", os.path.basename(pt)).group(1))
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        ebm, hyper_E = build_hypernet_ebm(
            K=1, D=D, hidden_dim=int(args.hidden_dim), nlayers=int(args.nlayers),
            strong_convexity=float(args.strong_convexity),
            hyper_hidden_sizes=[int(h) for h in args.hyper_hidden_sizes],
            hyper_summary_kind="mlp",
            hyper_extra_kwargs={"pool_norm": bool(pool_norm), "pool_scale": float(a.hyper_pool_scale)},
            hyper_activation_noise_std=float(a.hyper_noise_std),
        )
        # Archives written before the energy model was renamed store its state as ``lcmm_state``.
        ebm.load_state_dict(ck["ebm_state"] if "ebm_state" in ck else ck["lcmm_state"]); hyper_E.load_state_dict(ck["hyper_E_state"])
        ebm, hyper_E = ebm.to(device).eval(), hyper_E.to(device).eval()
        flow = ConditionalSamplerFlow(K=1, D=D, D_aug=int(args.flow_D_aug), n_hidden=int(args.flow_n_hidden),
                                      n_flow_layers=int(args.flow_n_layers), n_mlp_layers=int(getattr(args, "flow_n_mlp_layers", 3)),
                                      cond_dim=int(getattr(args, "cond_dim", 8))).to(device)
        flow.load_state_dict(ck["flow_state"]); flow.eval()

        pinned = PinnedEmitter(_pooled_params(hyper_E, x_all, a.chunk)).to(device)
        with torch.no_grad():
            torch.manual_seed(seed)
            x_h = target.sample(int(args.n_batch_hyper), device=device).detach()
            batch = PinnedEmitter({k: v for k, v in hyper_E(x_h).items()}).to(device)
        m_pin = _score(ebm, pinned, flow, target, args, device, D)
        m_bat = _score(ebm, batch, flow, target, args, device, D)
        rec = {"pinned": {k: float(v) for k, v in m_pin.items() if np.ndim(v) == 0},
               "batch": {k: float(v) for k, v in m_bat.items() if np.ndim(v) == 0},
               "recorded": {k: float(v) for k, v in ck.get("metrics", {}).items() if np.ndim(v) == 0}}
        if "log_p_grid" in m_pin:
            out_npz[f"seed{seed}/log_p_grid_pinned"] = m_pin["log_p_grid"]
            out_npz[f"seed{seed}/log_p_grid_batch"] = m_bat["log_p_grid"]
        if a.snapshots and ck.get("hyper_E_snaps"):
            its, tvs = [], []
            for it, sd in zip(ck["snap_iters"], ck["hyper_E_snaps"]):
                hyper_E.load_state_dict(sd)
                em = PinnedEmitter(_pooled_params(hyper_E, x_all, a.chunk)).to(device)
                tvs.append(float(_score(ebm, em, flow, target, args, device, D)["tv"])); its.append(int(it))
            hyper_E.load_state_dict(ck["hyper_E_state"])
            out_npz[f"seed{seed}/tv_hist_iters"] = np.asarray(its); out_npz[f"seed{seed}/tv_hist_pinned"] = np.asarray(tvs)
            rec["tv_hist_pinned_last"] = tvs[-1]
        out_json["seeds"][str(seed)] = rec
        key = "tv" if D == 1 else "test_nll"
        print(f"  seed {seed:2d}  {key} pinned {rec['pinned'][key]:.4f}  batch {rec['batch'][key]:.4f}  "
              f"recorded {rec['recorded'].get(key, float('nan')):.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(out_json, f, indent=2)
    np.savez_compressed(a.out + ".npz", **out_npz)
    print(f"wrote {a.out}.json and {a.out}.npz")


if __name__ == "__main__":
    main()
