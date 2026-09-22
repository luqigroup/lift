r"""Recompute a hybrid (Delta_fin + PCA-orth) loss-landscape NPZ.

Re-evaluates the plane of a saved hypernet-vs-direct(-vs-PGD) checkpoint at an
arbitrary grid resolution. The directions mirror the lib's
``_trajectory_pca_dirs``:

- ``d1 = direct_final - hypernet_final``, so direct's final iterate lands at
  alpha = 1;
- ``d2 = top SVD loading direction across the union of the snapshots``, after
  Gram-Schmidt removal of ``d1``, rescaled so the largest projection is 1.

The loss closure is the EBM NLL of the lib's ``_build_landscape_1d``: Riemann
log-Z at ``D <= 2``, and at ``D >= 3`` a frozen flow-IS proposal drawn once from
the converged lift's sampler flow.

The radius is the lib's auto-recipe, ``max-abs(traj_proj) * margin``, locked by
default to the alpha range of the existing ``seed{S}_landscape.npz`` so the new
grid covers the same domain and the figure window does not move.

Usage::

    python scripts/recompute_landscape_hybrid.py \
        --run_dir checkpoints/<config-hash-dir> \
        --config configs/experiments_hypernet_vs_direct_2d_k1.json \
        --target_kind gamma_mode_2d \
        --backends hypernet,direct,pgd \
        --seed 0 \
        --grid_n 121 \
        --out_suffix _121
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
from projorg import gitdir

# ``lift.scripts`` is not a package; import the lib via sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = gitdir()
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)


def _flatten(d: Dict[str, torch.Tensor], keys) -> torch.Tensor:
    return torch.cat([d[k].reshape(-1) for k in keys])


def _diff_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]):
    return {k: a[k] - b[k] for k in a}


def _dot_dict(a, b):
    return sum((a[k] * b[k]).sum().item() for k in a)


def _scale_dict(d, scale):
    # Multiplication, as in the lib: after rescaling d2 by
    # scale = max|projs|, the projection coordinates lie in [-1, 1].
    s = float(scale)
    return {k: v * s for k, v in d.items()}


def _gs_orthogonalise(d, basis_list):
    """Gram-Schmidt remove ``basis_list`` from ``d``."""
    out = {k: v.clone() for k, v in d.items()}
    for b in basis_list:
        nb = _dot_dict(b, b)
        if nb <= 1e-20:
            continue
        coef = _dot_dict(out, b) / nb
        for k in out:
            out[k] = out[k] - coef * b[k]
    return out


def _trajectory_pca_dirs_multi(
    snaps_by_backend: Dict[str, List[Dict[str, torch.Tensor]]],
    center: Dict[str, torch.Tensor],
):
    """Hybrid directions: ``d1 = Delta_fin``, ``d2 = PCA-orth``.

    Mirrors ``HypernetVsDirect1D._trajectory_pca_dirs`` but accepts an arbitrary
    set of backend snapshot stacks, so the PGD trajectory enters the PCA when
    present. ``d1`` is always the ``direct_final - hypernet_final`` delta.
    """
    keys = list(center.keys())
    if "direct" not in snaps_by_backend or "hypernet" not in snaps_by_backend:
        raise SystemExit(
            "trajectory PCA requires both 'hypernet' and 'direct' snapshots"
        )
    d_final = snaps_by_backend["direct"][-1]
    d1 = _diff_dict(d_final, center)

    all_snaps: List[Dict[str, torch.Tensor]] = []
    for b in ("hypernet", "direct", "pgd"):
        if b in snaps_by_backend:
            all_snaps += snaps_by_backend[b]
    diffs = [_diff_dict(t, center) for t in all_snaps]
    diffs_perp = [_gs_orthogonalise(d, [d1]) for d in diffs]
    flat = torch.stack([_flatten(d, keys) for d in diffs_perp])
    _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
    v_top = Vh[0]
    d2: Dict[str, torch.Tensor] = {}
    o = 0
    for k in keys:
        n = center[k].numel()
        d2[k] = v_top[o : o + n].view_as(center[k])
        o += n
    # rescale d2 so the max projection magnitude across snaps equals 1
    projs = [
        _dot_dict(d, d2) / max(_dot_dict(d2, d2), 1e-20)
        for d in diffs_perp
    ]
    scale = max(abs(min(projs)), abs(max(projs)), 1e-12)
    d2 = _scale_dict(d2, scale)
    return d1, d2


def _project_traj(
    snaps: List[Dict[str, torch.Tensor]],
    center: Dict[str, torch.Tensor],
    d1: Dict[str, torch.Tensor],
    d2: Dict[str, torch.Tensor],
) -> np.ndarray:
    keys = list(center.keys())
    v_star = _flatten(center, keys)
    v_d1 = _flatten(d1, keys)
    v_d2 = _flatten(d2, keys)
    D_mat = torch.stack([v_d1, v_d2], dim=1)
    A = D_mat.T @ D_mat
    coords = []
    for theta in snaps:
        v = _flatten({k: theta.get(k, center[k]) for k in keys}, keys)
        rhs = D_mat.T @ (v - v_star)
        ab = torch.linalg.solve(A, rhs)
        coords.append(ab.detach().cpu().numpy())
    return np.stack(coords) if coords else np.zeros((0, 2), dtype=np.float64)


def _coerce_args(config: dict) -> argparse.Namespace:
    """Mirror ``projorg.setup_environment`` argparse coercion for the keys
    the lib's ``_build_landscape_1d`` reads."""
    ns = argparse.Namespace()
    for k, v in config.items():
        if isinstance(v, str) and "," in v and all(
            s.strip().lstrip("-").isdigit() for s in v.split(",")
        ):
            v = [int(s) for s in v.split(",")]
        setattr(ns, k, v)
    return ns


def _build_ebm_template(args: argparse.Namespace, D: int, K: int, device):
    from lift.objectives.builders import build_hypernet_ebm
    ebm_template, _ = build_hypernet_ebm(
        K=K, D=D,
        hidden_dim=int(args.hidden_dim),
        nlayers=int(args.nlayers),
        strong_convexity=float(args.strong_convexity),
        device=device,
    )
    for p in ebm_template.parameters():
        p.requires_grad_(False)
    return ebm_template


def _build_flow_template(args: argparse.Namespace, D: int, K: int, device):
    from lift.models.conditional_sampler_flow import ConditionalSamplerFlow
    flow = ConditionalSamplerFlow(
        K=K, D=D,
        D_aug=int(args.flow_D_aug),
        n_hidden=int(args.flow_n_hidden),
        n_flow_layers=int(args.flow_n_layers),
        n_mlp_layers=int(args.flow_n_mlp_layers),
        cond_dim=int(getattr(args, "cond_dim", 8)),
    ).to(device)
    for p in flow.parameters():
        p.requires_grad_(False)
    return flow


def _free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eval_grid_loss(
    ebm_template,
    center_theta,
    d1, d2,
    alphas, betas,
    x_val,
    D: int,
    K: int,
    *,
    grid=None, dvol=None,
    x_prop_aug=None, log_q_prop=None, D_aug=None,
    pos_names: Optional[set] = None,
    verbose_every: int = 8,
) -> np.ndarray:
    """Sequential grid eval. Uses Riemann log-Z at D<=2 and flow-IS at D>=3."""
    from lift.objectives._batched import aug_energy_batched
    from lift.objectives._param_utils import detach_dict, stack_per_k_params
    from lift.objectives.partition import log_Z_via_grid_1d

    pos_names = pos_names or set()
    Z = np.zeros((len(alphas), len(betas)), dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                theta = {}
                for k, v_star in center_theta.items():
                    t = v_star + float(a) * d1[k] + float(b) * d2[k]
                    if k in pos_names:
                        t = t.clamp(min=0.0)
                    theta[k] = t
                ebm_stacked = detach_dict(
                    stack_per_k_params(theta, K=K, prefix="components.")
                )
                if D <= 2:
                    log_Z = log_Z_via_grid_1d(
                        ebm_template, theta, K=K, grid=grid, dvol=float(dvol),
                    )
                else:
                    E_prop = aug_energy_batched(
                        ebm_template.components[0], ebm_stacked, x_prop_aug,
                        K=K, D=D, D_aug=int(D_aug),
                    )
                    M = int(x_prop_aug.shape[0])
                    log_Z = (
                        torch.logsumexp(-E_prop - log_q_prop[None, :], dim=1)
                        - math.log(M)
                    )
                log_pi = torch.log_softmax(theta["log_pi"], dim=-1)
                E_val = aug_energy_batched(
                    ebm_template.components[0], ebm_stacked, x_val,
                    K=K, D=D, D_aug=D,
                )
                log_q = torch.logsumexp(
                    log_pi[:, None] - E_val - log_Z[:, None], dim=0,
                )
                Z[i, j] = float((-log_q).mean().item())
            if (i + 1) % verbose_every == 0 or i == len(alphas) - 1:
                elapsed = time.time() - t0
                done = (i + 1) * len(betas)
                total = len(alphas) * len(betas)
                eta = elapsed / max(done, 1) * (total - done)
                print(
                    f"  [grid] row {i + 1}/{len(alphas)} "
                    f"({done}/{total}, {elapsed:.1f}s, "
                    f"ETA {eta:.1f}s) "
                    f"Z range so far [{Z[:i+1, :].min():.4g}, "
                    f"{Z[:i+1, :].max():.4g}]",
                    flush=True,
                )
                _free_memory()
    return Z


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--target_kind", required=True, type=str)
    p.add_argument(
        "--backends", type=str, default="hypernet,direct",
        help="Subset of {hypernet, direct, pgd}.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_suffix", type=str, default="_121",
                   help="Saved NPZ basename: seed{S}_landscape{out_suffix}.npz")
    p.add_argument("--grid_n", type=int, default=121)
    p.add_argument(
        "--lock_to_existing_alpha_range", type=int, default=1,
        help="If 1, take alpha/beta endpoints from the existing "
             "seed{S}_landscape.npz (or --reference_npz_basename). "
             "If 0, recompute radius via lib auto-recipe.",
    )
    p.add_argument("--reference_npz_basename", type=str, default="")
    p.add_argument(
        "--M_logZ_eval", type=int, default=0,
        help="Override IS proposal sample count (0 -> use config value, "
             "default 4096). Set to 2048 if memory pressure forces it.",
    )
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    backends_req = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "hypernet" not in backends_req or "direct" not in backends_req:
        raise SystemExit("--backends must contain both hypernet and direct")

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith("cuda") else "cpu")
    print(f"[hybrid] device={device}", flush=True)
    print(f"[hybrid] run_dir={args.run_dir}", flush=True)
    print(f"[hybrid] config={args.config}", flush=True)
    print(f"[hybrid] target_kind={args.target_kind}", flush=True)
    print(f"[hybrid] backends={backends_req}, seed={args.seed}, "
          f"grid_n={args.grid_n}", flush=True)

    with open(args.config) as f:
        config = json.load(f)
    ns_args = _coerce_args(config)
    if args.M_logZ_eval > 0:
        setattr(ns_args, "M_logZ_eval", int(args.M_logZ_eval))

    # ----- load all requested backends' checkpoints -----
    seed = int(args.seed)
    snaps_by_backend: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    snap_iters_per_backend: Dict[str, np.ndarray] = {}
    flow_state = None
    for backend in backends_req:
        path = os.path.join(args.run_dir, f"seed{seed}_{backend}.pt")
        if not os.path.isfile(path):
            if backend == "pgd":
                print(f"[hybrid] missing PGD ckpt {path}; "
                      f"continuing without PGD")
                continue
            raise SystemExit(f"missing checkpoint {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ts = ck.get("theta_snaps", [])
        snaps_by_backend[backend] = [
            {k: v.detach().to(device).float() for k, v in s.items()}
            for s in ts
        ]
        snap_iters_per_backend[backend] = np.asarray(
            ck.get("snap_iters", []), dtype=np.int64,
        )
        if backend == "hypernet":
            flow_state = ck.get("flow_state", None)

    center_theta = {
        k: v.clone() for k, v in snaps_by_backend["hypernet"][-1].items()
    }

    # ----- target build / data dim -----
    try:
        from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D
    except Exception:
        from scripts._experiments_hypernet_vs_direct_1d_lib import (
            HypernetVsDirect1D,
        )
    setattr(ns_args, "target_kind", args.target_kind)
    setattr(ns_args, "seeds", [seed])
    setattr(ns_args, "experiment", os.path.basename(args.run_dir))
    setattr(ns_args, "phase", "visualization")
    expt = HypernetVsDirect1D(ns_args, target_kind=args.target_kind)
    expt.device = device
    target = expt._build_target()
    D = int(getattr(target, "D", 1))
    K = int(getattr(ns_args, "K", 1))
    print(f"[hybrid] D={D}, K={K}", flush=True)

    # ----- log-Z eval grid -----
    grid = None
    dvol = None
    x_prop_aug = None
    log_q_prop = None
    flow = None
    if D <= 2:
        from lift.objectives.eval import build_eval_grid
        n_grid_arg = int(getattr(ns_args, "tv_grid_n", 4096))
        if D == 1:
            xs, _, _dvol = build_eval_grid(target, n_grid_arg, device)
            grid, dvol = xs, _dvol
        else:
            n_axis = max(int(np.sqrt(n_grid_arg)), 96)
            (lo_x, hi_x), (lo_y, hi_y) = target.xlim
            xs_x = torch.linspace(lo_x, hi_x, n_axis, device=device)
            xs_y = torch.linspace(lo_y, hi_y, n_axis, device=device)
            gx, gy = torch.meshgrid(xs_x, xs_y, indexing="xy")
            grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
            dvol = float((hi_x - lo_x) * (hi_y - lo_y) / (n_axis * n_axis))
        print(f"[hybrid] Riemann log-Z grid: D={D}, n_grid={grid.shape}",
              flush=True)
    else:
        from lift.objectives._batched import conditional_flow_sample
        if flow_state is None:
            raise SystemExit(
                "D>=3 path requires hypernet ckpt with 'flow_state'"
            )
        flow = _build_flow_template(ns_args, D, K, device)
        flow.load_state_dict(flow_state, strict=False)
        flow = flow.to(device)
        for p_ in flow.parameters():
            p_.requires_grad_(False)
        D_aug = int(ns_args.flow_D_aug)
        M_prop = int(getattr(ns_args, "M_logZ_eval", 4096) or 4096)
        torch.manual_seed(int(seed) + 31)
        with torch.no_grad():
            eps = torch.randn(1, M_prop, D_aug, device=device)
            x_aug, log_q_aug = conditional_flow_sample(
                flow, eps, K=1, with_logprob=True,
            )
        x_prop_aug = x_aug[0].detach()
        log_q_prop = log_q_aug[0].detach()
        print(f"[hybrid] flow IS proposal M={M_prop}, D_aug={D_aug}",
              flush=True)

    # ----- positivity mask -----
    from lift.models import icnn_pos_param_names
    pos_names = set()
    for kk in range(K):
        pos_names |= icnn_pos_param_names(
            int(ns_args.nlayers), prefix=f"components.{kk}.",
        )

    # ----- ebm template -----
    ebm_template = _build_ebm_template(ns_args, D, K, device)

    # ----- x_val for the landscape (reuse the on-disk one) -----
    x_val_np = np.load(
        os.path.join(args.run_dir, f"seed{seed}_x_val_landscape.npy")
    )
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)

    # ----- hybrid (Delta_fin + PCA-orth) directions -----
    d1, d2 = _trajectory_pca_dirs_multi(snaps_by_backend, center_theta)

    # ----- radius: lock to existing alpha range when available -----
    ref_npz_path = ""
    if args.lock_to_existing_alpha_range:
        if args.reference_npz_basename:
            ref_npz_path = os.path.join(args.run_dir, args.reference_npz_basename)
        else:
            ref_npz_path = os.path.join(args.run_dir, f"seed{seed}_landscape.npz")
    if ref_npz_path and os.path.isfile(ref_npz_path):
        ref = np.load(ref_npz_path, allow_pickle=True)
        radius = float(np.max(np.abs(ref["alphas"])))
        print(f"[hybrid] locked radius={radius:.4f} from {ref_npz_path}",
              flush=True)
    else:
        # Lib auto-recipe.
        pre_proj_max = 0.0
        for b in backends_req:
            if b in snaps_by_backend:
                proj = _project_traj(snaps_by_backend[b], center_theta, d1, d2)
                if proj.size:
                    pre_proj_max = max(pre_proj_max, float(np.max(np.abs(proj))))
        if pre_proj_max <= 0.0:
            pre_proj_max = 1.0
        margin = float(getattr(ns_args, "landscape_margin", 1.4))
        radius_cfg = float(getattr(ns_args, "landscape_radius", 0.0) or 0.0)
        radius = pre_proj_max * margin if radius_cfg <= 0.0 else radius_cfg
        print(f"[hybrid] auto radius={radius:.4f} "
              f"(max_abs={pre_proj_max:.4f}, margin={margin})", flush=True)

    gn = int(args.grid_n)
    alphas = np.linspace(-radius, radius, gn)
    betas = np.linspace(-radius, radius, gn)

    # ----- grid eval -----
    Z = _eval_grid_loss(
        ebm_template, center_theta, d1, d2, alphas, betas, x_val,
        D=D, K=K,
        grid=grid, dvol=dvol,
        x_prop_aug=x_prop_aug, log_q_prop=log_q_prop,
        D_aug=getattr(ns_args, "flow_D_aug", None),
        pos_names=pos_names,
    )
    print(f"[hybrid] Z range [{Z.min():.4g}, {Z.max():.4g}]", flush=True)

    # ----- per-backend traj coords -----
    out: Dict[str, np.ndarray] = {
        "alphas": alphas, "betas": betas, "Z": Z,
        "method": np.array("hybrid_trajectory_pca"),
        "radius": np.float64(radius),
    }
    for backend, snaps in snaps_by_backend.items():
        out[f"coords_{backend}"] = _project_traj(
            snaps, center_theta, d1, d2,
        )
        out[f"iters_{backend}"] = snap_iters_per_backend.get(
            backend, np.zeros(0, dtype=np.int64),
        )

    # rename 'coords_hypernet' -> 'coords_hyper' to match the lib's
    # canonical key naming (renderers also accept 'coords_hypernet').
    if "coords_hypernet" in out:
        out["coords_hyper"] = out.pop("coords_hypernet")
        out["iters_hyper"] = out.pop("iters_hypernet")

    out_path = os.path.join(
        args.run_dir, f"seed{seed}_landscape{args.out_suffix}.npz"
    )
    np.savez(out_path, **out)
    print(f"[hybrid] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
