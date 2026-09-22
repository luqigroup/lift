r"""Loss landscape of the HEPMASS run in the lifted space.

The companion of the constrained-:math:`\theta` panel: the same FKL-direct
test-NLL closure is evaluated on a 2-D slice of the hypernetwork's own
parameter space rather than of the ICNN constrained :math:`\theta` space. Only
the lift appears in the panel, since PGD and direct softplus have no embedding
into hypernetwork space.

Method. Center on the last ``hyper_E_snaps`` entry, take the top two
trajectory-PCA directions of the lifted snapshots, filter-normalize them per
Li 2018 against the center's per-row Frobenius norms, then for each
``(alpha, beta)`` on the grid load ``center + alpha * d1 + beta * d2`` into a
fresh ``HyperNetwork``, forward it on the saved ``snap_x``, and evaluate the
NLL of the emitted :math:`\theta`. The NPZ carries ``alphas, betas, Z,
coords_hyper, iters_hyper, method``.

``--target_kind uci_miniboone`` runs the same code on MiniBooNE (D = 43). The
checkpoint is memory-mapped and only the ``--max_snaps`` kept snapshots are
materialized, which roughly halves the host memory a 10 GB pickle would need.

Usage::

    conda run -n sips python scripts/compute_lifted_landscape_hepmass.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
from projorg import gitdir

# lift.scripts is not a package; import the lib via sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = gitdir()
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)


_DEFAULT_RUN = os.path.join(
    _REPO, "data", "checkpoints",
    "experiments_pgd_uci_hepmass_target_kind-uci_hepmass_K-1_objective-fkl-direct_data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_sampler-0.005_hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_sampler_hyper.b0929293841311db47eff4ffed48c9fb67bf0bca",
)
_DEFAULT_CONFIG = os.path.join(
    _REPO, "configs", "experiments_pgd_uci_hepmass.json",
)


# ----- helpers ------------------------------------------------------------


def _flatten(d: Dict[str, torch.Tensor], keys) -> torch.Tensor:
    return torch.cat([d[k].reshape(-1) for k in keys])


def _unflatten(
    flat: torch.Tensor,
    shapes: Dict[str, torch.Size],
    keys,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = OrderedDict()
    off = 0
    for k in keys:
        n = int(np.prod(shapes[k]))
        out[k] = flat[off : off + n].view(shapes[k]).clone()
        off += n
    return out


def _filter_normalise(
    direction: Dict[str, torch.Tensor],
    center: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Li-2018 filter normalization, eq. 5.

    Rescales each row of each direction tensor to the Frobenius norm of
    the matching row of the center. For 1-D tensors, biases, it matches
    the tensor norm instead.
    """
    out: Dict[str, torch.Tensor] = OrderedDict()
    for name, r in direction.items():
        w = center[name]
        if r.ndim >= 2 and r.size(0) > 1:
            n = r.size(0)
            rf = r.reshape(n, -1)
            wf = w.detach().reshape(n, -1)
            rn = rf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            wn = wf.norm(dim=-1, keepdim=True)
            rf = rf * (wn / rn)
            out[name] = rf.reshape_as(r).detach()
        else:
            rn = r.norm().clamp(min=1e-12)
            out[name] = (r * (w.detach().norm() / rn)).detach()
    return out


def _project_traj_lifted(
    snaps: List[Dict[str, torch.Tensor]],
    center: Dict[str, torch.Tensor],
    d1: Dict[str, torch.Tensor],
    d2: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Project each hyper_E snapshot onto the (d1, d2) frame at ``center``.

    Solves the 2x2 normal equations, since d1 and d2 are not orthonormal
    after filter normalization. Tensors are coerced to CPU double to keep
    GPU memory free.
    """
    keys = list(center.keys())
    v_star = _flatten(
        {k: center[k].detach().cpu().double() for k in keys}, keys,
    )
    v_d1 = _flatten(
        {k: d1[k].detach().cpu().double() for k in keys}, keys,
    )
    v_d2 = _flatten(
        {k: d2[k].detach().cpu().double() for k in keys}, keys,
    )
    D_mat = torch.stack([v_d1, v_d2], dim=1)
    A = D_mat.T @ D_mat
    coords = []
    for snap in snaps:
        v = _flatten(
            {k: snap.get(k, center[k]).detach().cpu().double() for k in keys},
            keys,
        )
        rhs = D_mat.T @ (v - v_star)
        ab = torch.linalg.solve(A, rhs)
        coords.append(ab.detach().cpu().numpy())
    return np.stack(coords) if coords else np.zeros((0, 2), dtype=np.float64)


def _coerce_args(config: dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in config.items():
        if isinstance(v, str) and "," in v and all(
            s.strip().lstrip("-").isdigit() for s in v.split(",")
        ):
            v = [int(s) for s in v.split(",")]
        setattr(ns, k, v)
    return ns


# ----- main ---------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=str, default=_DEFAULT_RUN)
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target_kind", type=str, default="uci_hepmass",
        help="the UCI target of the run in --run_dir. The lifted "
             "landscape is target-agnostic -- it evaluates the emitter's "
             "own loss on the run's own validation batch -- so the same "
             "code serves MiniBooNE (uci_miniboone, D = 43) once the "
             "target is not hard-coded",
    )
    p.add_argument("--out_suffix", type=str, default="_lifted")
    p.add_argument(
        "--out_path", type=str, default="",
        help="where to write the NPZ; default "
             "<run_dir>/seed<S>_landscape<out_suffix>.npz",
    )
    p.add_argument("--grid_n", type=int, default=31)
    p.add_argument("--margin", type=float, default=1.6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max_snaps", type=int, default=0,
        help="evenly subsample the lifted trajectory to at most this many "
             "snapshots before the PCA (0 = keep all). At HEPMASS scale "
             "each snapshot is ~0.43 GB, so 12 (the two-spaces lifted-EBM "
             "cap) is what fits a 32 GB host.")
    args = p.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu",
    )
    print(f"[lifted-landscape] device={device}")
    print(f"[lifted-landscape] run_dir={args.run_dir}")

    with open(args.config) as f:
        config = json.load(f)
    ns_args = _coerce_args(config)
    setattr(ns_args, "phase", "visualization")
    setattr(ns_args, "experiment", os.path.basename(args.run_dir))

    # ---- load hypernet ckpt ----
    seed = int(args.seed)
    ck_path = os.path.join(args.run_dir, f"seed{seed}_hypernet.pt")
    try:
        # Memory-map the pickle: only the kept snapshots are materialized,
        # by the clone below. MAP_PRIVATE is torch's default, so the file is
        # never written to.
        ck = torch.load(ck_path, map_location="cpu", weights_only=False,
                        mmap=True)
    except (TypeError, RuntimeError) as exc:   # legacy (non-zip) pickle
        print(f"[lifted-landscape] mmap load unavailable ({exc}); "
              "falling back to a full load")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    snaps_E_raw = ck.get("hyper_E_snaps", None)
    if snaps_E_raw is None or len(snaps_E_raw) == 0:
        raise SystemExit(
            f"missing 'hyper_E_snaps' in {args.run_dir}/seed{seed}_hypernet.pt; "
            "training run must have persisted lifted-space snapshots.",
        )
    # Snapshots stay on CPU; only the center and the two directions go to the
    # GPU. A lifted snapshot runs to hundreds of megabytes, so ``--max_snaps``
    # subsamples the trajectory evenly, keeping the first and the last: the PCA
    # needs the directions of motion, not every recorded step.
    keep_idx = list(range(len(snaps_E_raw)))
    if int(args.max_snaps) > 0 and len(keep_idx) > int(args.max_snaps):
        sel = np.linspace(0, len(keep_idx) - 1, int(args.max_snaps))
        keep_idx = sorted({int(round(v)) for v in sel})
        print(f"[lifted-landscape] subsampling trajectory "
              f"{len(snaps_E_raw)} -> {len(keep_idx)} snapshots "
              f"(--max_snaps {args.max_snaps})")
    snaps_E = [
        OrderedDict(
            (k, v.detach().cpu().float().clone())
            for k, v in snaps_E_raw[i].items()
        )
        for i in keep_idx
    ]
    snap_iters_all = np.asarray(ck.get("snap_iters", []), dtype=np.int64)
    snap_iters = (
        snap_iters_all[keep_idx] if snap_iters_all.size == len(snaps_E_raw)
        else snap_iters_all
    )
    print(f"[lifted-landscape] hyper_E_snaps: len={len(snaps_E)} keys={len(snaps_E[0])}")
    snap_x = ck["snap_x"].detach().clone().to(device).float()
    print(f"[lifted-landscape] snap_x: {tuple(snap_x.shape)}")
    flow_state = ck.get("flow_state", None)
    if flow_state is not None:
        flow_state = {
            k: v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k, v in flow_state.items()
        }
    # Drop the pickle before the PCA allocates; everything still needed has
    # been copied out above.
    import gc
    del snaps_E_raw, ck
    gc.collect()

    center_cpu = OrderedDict((k, v.clone()) for k, v in snaps_E[-1].items())
    keys = list(center_cpu.keys())
    shapes = {k: v.shape for k, v in center_cpu.items()}

    # ---- top-2 trajectory PCA directions (CPU) ----
    # Thin SVD through the (T, T) Gram matrix: accumulate
    # centered @ centered.T, then eigh, then recover the right singular
    # vectors as
    # centered.T @ U[:, :k] / S[:k].
    T = len(snaps_E)
    P = int(sum(int(np.prod(shapes[k])) for k in keys))
    print(f"[lifted-landscape] T={T} snapshots, P={P} lifted params total")

    mean = OrderedDict((k, torch.zeros_like(center_cpu[k])) for k in keys)
    for s in snaps_E:
        for k in keys:
            mean[k] += s[k]
    for k in keys:
        mean[k] /= float(T)

    # Center the snapshots in place, then build the (T, T) Gram matrix tensor
    # by tensor: flattening T float64 copies costs T * P * 8 bytes and exhausts
    # the host, while accumulating <s_i, s_j> in float64 gives the same number
    # with no P-sized temporary. From here to the restore below, ``snaps_E``
    # holds centered snapshots; ``center_cpu`` was cloned, so the converged
    # state is unaffected.
    for s in snaps_E:
        for k in keys:
            s[k] -= mean[k]
    G = torch.zeros((T, T), dtype=torch.float64)
    for i in range(T):
        for j in range(i, T):
            acc = 0.0
            for k in keys:
                acc += float(torch.dot(
                    snaps_E[i][k].reshape(-1).double(),
                    snaps_E[j][k].reshape(-1).double(),
                ))
            G[i, j] = acc
            G[j, i] = acc
    # Eigendecomp of Gram = U S^2 U^T -> singular values are sqrt(eigvals).
    eigvals, eigvecs = torch.linalg.eigh(G)
    # eigh returns ascending; reverse for descending.
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx].clamp(min=0.0)
    eigvecs = eigvecs[:, idx]
    sing = eigvals.sqrt()
    print(f"[lifted-landscape] singular values (top 5): {sing[:5].tolist()}")
    # Right singular vectors V[:, k] = centered.T @ U[:, k] / sing[k],
    # accumulated per parameter tensor for the same memory reason as the Gram.
    u1 = eigvecs[:, 0]
    u2 = eigvecs[:, 1]
    s1 = max(float(sing[0].item()), 1e-30)
    s2 = max(float(sing[1].item()), 1e-30)
    d1_raw: Dict[str, torch.Tensor] = OrderedDict(
        (k, torch.zeros(shapes[k], dtype=torch.float32)) for k in keys
    )
    d2_raw: Dict[str, torch.Tensor] = OrderedDict(
        (k, torch.zeros(shapes[k], dtype=torch.float32)) for k in keys
    )
    for i in range(T):
        c1 = float(u1[i].item()) / s1
        c2 = float(u2[i].item()) / s2
        for k in keys:
            d1_raw[k] += c1 * snaps_E[i][k]
            d2_raw[k] += c2 * snaps_E[i][k]
    # Restore ``snaps_E`` to uncentered values: the trajectory projection below
    # reads them as raw lifted states.
    for s in snaps_E:
        for k in keys:
            s[k] += mean[k]
    del G, mean
    gc.collect()
    center = OrderedDict((k, v.to(device)) for k, v in center_cpu.items())
    d1_raw = OrderedDict((k, v.to(device)) for k, v in d1_raw.items())
    d2_raw = OrderedDict((k, v.to(device)) for k, v in d2_raw.items())
    d1 = _filter_normalise(d1_raw, center)
    d2 = _filter_normalise(d2_raw, center)

    # ---- radius autosize via pre-projection of the snapshot trajectory ----
    pre_coords = _project_traj_lifted(snaps_E, center, d1, d2)
    max_abs = float(np.max(np.abs(pre_coords))) if pre_coords.size else 1.0
    if max_abs <= 0.0:
        max_abs = 1.0
    radius = max_abs * float(args.margin)
    gn = int(args.grid_n)
    alphas = np.linspace(-radius, radius, gn)
    betas = np.linspace(-radius, radius, gn)
    print(
        f"[lifted-landscape] radius={radius:.4g} "
        f"(max_abs={max_abs:.4g}, margin={args.margin}), grid_n={gn}",
    )

    # ---- build target + ebm template + flow template ----
    setattr(ns_args, "target_kind", str(args.target_kind))
    setattr(ns_args, "seeds", [seed])

    # Scripts are not a package, hence the sys.path import above.
    from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D
    expt = HypernetVsDirect1D(ns_args, target_kind=str(args.target_kind))
    expt.device = device
    target = expt._build_target()
    D = int(getattr(target, "D", 21))
    K = int(getattr(ns_args, "K", 1))
    print(f"[lifted-landscape] D={D}, K={K}")

    from lift.objectives.builders import (
        build_hypernet_ebm,
    )
    ebm_template, hyper_template = build_hypernet_ebm(
        K=K, D=D,
        hidden_dim=int(ns_args.hidden_dim),
        nlayers=int(ns_args.nlayers),
        strong_convexity=float(ns_args.strong_convexity),
        hyper_hidden_sizes=list(map(int, str(ns_args.hyper_hidden_sizes).split(",")))
            if isinstance(ns_args.hyper_hidden_sizes, str)
            else list(ns_args.hyper_hidden_sizes),
        device=device,
    )
    ebm_template.requires_grad_(False)
    hyper_template.requires_grad_(False)
    hyper_template.eval()
    ebm_template.eval()

    # ---- positivity mask on the constrained theta (used by NLL closure) ----
    from lift.models import icnn_pos_param_names
    pos_names = set()
    for kk in range(K):
        pos_names |= icnn_pos_param_names(
            int(ns_args.nlayers), prefix=f"components.{kk}.",
        )

    # ---- flow-IS log-Z proposal (D>=3 path; matches strict-Li-2018 script) ----
    from lift.models.conditional_sampler_flow import ConditionalSamplerFlow
    flow = ConditionalSamplerFlow(
        K=K, D=D,
        D_aug=int(ns_args.flow_D_aug),
        n_hidden=int(ns_args.flow_n_hidden),
        n_flow_layers=int(ns_args.flow_n_layers),
        n_mlp_layers=int(ns_args.flow_n_mlp_layers),
        cond_dim=int(getattr(ns_args, "cond_dim", 8)),
    ).to(device)
    if flow_state is None:
        raise SystemExit("D>=3 needs flow_state in ckpt for log-Z IS proposal")
    flow.load_state_dict(flow_state, strict=False)
    for p_ in flow.parameters():
        p_.requires_grad_(False)
    flow.eval()
    from lift.objectives._batched import (
        aug_energy_batched,
        conditional_flow_sample,
    )
    from lift.objectives._param_utils import detach_dict, stack_per_k_params
    D_aug = int(ns_args.flow_D_aug)
    M_prop = int(getattr(ns_args, "M_logZ_eval", 4096) or 4096)
    torch.manual_seed(0)
    with torch.no_grad():
        eps = torch.randn(1, M_prop, D_aug, device=device)
        x_aug, log_q_aug = conditional_flow_sample(
            flow, eps, K=1, with_logprob=True,
        )
    x_prop_aug = x_aug[0].detach()
    log_q_prop = log_q_aug[0].detach()
    print(f"[lifted-landscape] flow IS proposal: M={M_prop}, D_aug={D_aug}")

    # ---- x_val (held-out, same as constrained-theta NPZ) ----
    x_val_np = np.load(
        os.path.join(args.run_dir, f"seed{seed}_x_val_landscape.npy"),
    )
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)
    print(f"[lifted-landscape] x_val: {tuple(x_val.shape)}")

    # ---- grid eval over the lifted slice ----
    Z = np.zeros((len(alphas), len(betas)), dtype=np.float64)
    sd_template = hyper_template.state_dict()

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                pert_state = OrderedDict()
                for k in keys:
                    pert_state[k] = center[k] + float(a) * d1[k] + float(b) * d2[k]
                missing = [kk for kk in sd_template if kk not in pert_state]
                if missing:
                    # A key absent from the snapshot falls back to the
                    # template's own value rather than raising.
                    for kk in missing:
                        pert_state[kk] = sd_template[kk].detach().to(device)
                hyper_template.load_state_dict(pert_state, strict=False)
                theta = hyper_template(snap_x)
                # The positivity readout has already been applied inside the
                # hypernetwork, so theta is feasible and must not be clamped
                # again here.
                ebm_stacked = detach_dict(
                    stack_per_k_params(theta, K=K, prefix="components."),
                )
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
            if (i + 1) % max(1, gn // 5) == 0:
                print(
                    f"[lifted-landscape] row {i + 1}/{gn} "
                    f"min={Z[: i + 1].min():.3g} max={Z[: i + 1].max():.3g}",
                )

    print(
        f"[lifted-landscape] Z range "
        f"[{Z.min():.4g}, {Z.max():.4g}], "
        f"argmin at idx={np.unravel_index(int(Z.argmin()), Z.shape)}",
    )

    # ---- project the saved hyper trajectory onto (d1, d2) ----
    coords_hyper = _project_traj_lifted(snaps_E, center, d1, d2)
    print(
        f"[lifted-landscape] coords_hyper shape: {coords_hyper.shape}, "
        f"alpha range [{coords_hyper[:,0].min():.3f}, "
        f"{coords_hyper[:,0].max():.3f}], "
        f"beta range [{coords_hyper[:,1].min():.3f}, "
        f"{coords_hyper[:,1].max():.3f}]",
    )

    # ---- save ----
    out_path = args.out_path or os.path.join(
        args.run_dir, f"seed{seed}_landscape{args.out_suffix}.npz",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        method=np.array("lifted_pca_filter_norm"),
        alphas=alphas, betas=betas, Z=Z,
        coords_hyper=coords_hyper,
        iters_hyper=snap_iters,
        source=np.array(os.path.basename(os.path.normpath(args.run_dir))),
        target_kind=np.array(str(args.target_kind)),
    )
    print(f"[lifted-landscape] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
