r"""Render the two-space landscape figure: rows are problems, columns are spaces.

Rows are the 1-D Gumbel and 2-D gamma-mode K=1 ICNN-EBM problems; columns
are parameter spaces, the constrained ICNN theta on the left and the lifted
hypernetwork (Theta_E, b_h) on the right. Orange is the lift trajectory,
red is direct softplus, the gold star is the converged lift and the red
plus the converged direct model.

The constrained-space landscapes reuse the ``seed{S}_landscape.npz`` arrays
the training pipeline already saved, so nothing is re-evaluated. The lifted
landscapes are computed here by composing the hypernetwork with the
FKL-direct loss closure and passing it to
:func:`lift.utils.landscape.render_generic.compute_landscape`.

``_compute_pcpmap_landscapes`` builds the same pair of panels for the
PCP-Map Lotka-Volterra problem, which this figure no longer draws.

Usage::

    python scripts/render_landscape_two_spaces.py

The output PDF lands at
``figures/landscape_two_spaces_three_problems.pdf``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _free_memory(note: str = "") -> None:
    """Aggressively release CPU + GPU memory between phases.

    The lifted-landscape compute holds the whole ``seed{S}_hypernet.pt``
    pickle in CPU RAM, which can reach several GB, and a 25x25 grid of
    ``functional_call`` evaluations leaves CUDA cache fragments. Without
    this hook, sequential problem rows stack their pickles and exhaust
    host memory.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    if note:
        try:
            import resource
            rss_kb = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            print(f"  [mem after {note}] max RSS = {rss_kb / 1024:.0f} MB",
                  flush=True)
        except Exception:
            pass


def _subsample_traj(
    snaps: List, max_keep: int = 12,
) -> Tuple[List, np.ndarray]:
    """Subsample a snapshot list to at most ``max_keep`` evenly-spaced entries.

    Returns ``(subsampled_snaps, kept_indices)``. The first and last
    snapshots are always retained (training start + converged state)
    so the trajectory PCA's principal directions still span the
    full descent.
    """
    T = len(snaps)
    if T <= max_keep:
        return list(snaps), np.arange(T, dtype=np.int64)
    idx = np.linspace(0, T - 1, max_keep).round().astype(np.int64)
    idx = np.unique(idx)
    return [snaps[int(i)] for i in idx], idx

# Repo root on sys.path so ``lift.*`` imports resolve when invoked directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The scripts directory on sys.path, so the experiment-driver libs (e.g.
# ``_experiments_hypernet_vs_direct_1d_lib``) import directly; they are not
# part of the lift package.
_SCRIPTS_DIR = _THIS_DIR
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from lift.utils.landscape.render_generic import (  # noqa: E402
    compute_landscape,
    _filter_norm_direction,
    _project_trajectory,
    _evaluate_loss_grid,
)
from lift.utils.loss_sparkline import add_loss_sparkline  # noqa: E402
from lift.utils.paper_style import apply_paper_style  # noqa: E402


def _evaluate_loss_grid_with_grad(
    loss_fn, params_dict, d1, d2, alphas, betas,
):
    """Variant of :func:`_evaluate_loss_grid` that keeps autograd active.

    PCP-Map's NLL closure calls ``torch.autograd.grad(..., create_graph=True)``
    along the data argument, so the surrounding evaluation must not wrap the
    forward in ``torch.no_grad()``. The offset parameter tensors are detached
    instead, which keeps them outside the autograd graph at no extra memory
    cost while leaving the closure's internal x-gradient chain intact.

    The per-grid-point CUDA cache and GC flush keeps the evaluation loop from
    accumulating create-graph allocations: the graph is GB-scale at the larger
    hypernetwork sizes, and one stale reference per grid point multiplies by
    the number of points.
    """
    from collections import OrderedDict
    import torch
    keys = list(params_dict.keys())
    Z = np.empty((len(alphas), len(betas)), dtype=np.float64)
    flush_every = max(1, int(os.environ.get("LANDSCAPE_FLUSH_EVERY", 32)))
    step = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            offset = OrderedDict()
            for k in keys:
                offset[k] = (
                    params_dict[k] + float(a) * d1[k] + float(b) * d2[k]
                ).detach()
            # Off-converged PICNN parameters can have ill-conditioned
            # Hessians that crash eigvalsh; treat those points as very bad so
            # the contour rendering caps them through the log-axis tail.
            try:
                val = loss_fn(offset)
                Z[i, j] = float(val.detach() if hasattr(val, "detach") else val)
                del val
            except (torch._C._LinAlgError, RuntimeError) as exc:
                if "ill-conditioned" not in str(exc) and "linalg" not in str(exc).lower():
                    raise
                Z[i, j] = float("inf")
            del offset
            step += 1
            if step % flush_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return Z


# --------------------------------------------------------------------- paths

_CKPT_ROOT = os.path.join(_REPO_ROOT, "data", "checkpoints")
# Extra checkpoint roots to search by name, if any.
_CKPT_ROOTS_FALLBACK: list[str] = []
_OUT_PDF = os.path.join(
    _REPO_ROOT,
    "figures", "figures",
    "landscape_two_spaces_three_problems.pdf",
)

# Palette source of truth: ``lift/utils/paper_style.PALETTE``.
# Hypernet = orange (#ff7f0e), direct softplus = red (#d62728). Markers /
# line styles per the paper-wide landscape convention so the legend reads
# unambiguously even when reproduced in greyscale: hypernet = solid line
# + filled squares, direct softplus = dashed line + open circles.
_HYP_COLOR = "#ff7f0e"
_DIRECT_COLOR = "#d62728"
_HYP_LS = "-"
_DIRECT_LS = "--"
_HYP_MARKER = "s"
_DIRECT_MARKER = "o"
_DIRECT_MFC = "none"


def _load_ebm_loss_history(ckpt_dir: str, seed: int) -> Dict[str, np.ndarray]:
    """Return ``{'hyper': ..., 'direct': ...}`` per-iter losses from .pt ckpts.

    Used for the 1-D and 2-D ICNN-EBM rows; the matching ``history``
    dict has key ``loss_E_per_iter``.
    """
    import torch
    out: Dict[str, np.ndarray] = {
        "hyper": np.zeros(0, dtype=np.float64),
        "direct": np.zeros(0, dtype=np.float64),
    }
    for backend, key in (("hyper", "hypernet"), ("direct", "direct")):
        path = os.path.join(ckpt_dir, f"seed{seed}_{key}.pt")
        if not os.path.isfile(path):
            continue
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # pragma: no cover - corrupt ckpt
            continue
        hist = d.get("history") if isinstance(d, dict) else None
        if not isinstance(hist, dict):
            continue
        losses = hist.get("loss_E_per_iter")
        if losses is None:
            continue
        out[backend] = np.asarray(losses, dtype=np.float64)
    return out


def _find_ckpt_dir(prefix: str, hash_substr: Optional[str] = None) -> str:
    """Return the most-recently-modified ckpt dir matching ``prefix``.

    Searches the lift ``data/checkpoints`` dir first, then any
    fallback roots (see ``_CKPT_ROOTS_FALLBACK``). If ``hash_substr`` is
    given, the dir name must contain it.
    """
    candidates: List[Tuple[float, str]] = []
    roots_searched: List[str] = []
    for root in [_CKPT_ROOT, *_CKPT_ROOTS_FALLBACK]:
        if not os.path.isdir(root):
            continue
        roots_searched.append(root)
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith(prefix):
                continue
            if hash_substr is not None and hash_substr not in name:
                continue
            candidates.append((os.path.getmtime(full), full))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint dir under {roots_searched} with "
            f"prefix={prefix!r} and hash_substr={hash_substr!r}"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


# --------------------------------------------------------- per-problem loaders

def _load_1d2d_seed(ckpt_dir: str, seed: int) -> Dict:
    """Load a seed-S {hypernet,direct}.pt pair and shared landscape npz."""
    import torch
    npz_path = os.path.join(ckpt_dir, f"seed{seed}_landscape.npz")
    npz = None
    if os.path.isfile(npz_path):
        with np.load(npz_path) as nf:
            npz = {k: np.asarray(nf[k]) for k in nf.files}
    hyper_pt = torch.load(
        os.path.join(ckpt_dir, f"seed{seed}_hypernet.pt"),
        map_location="cpu", weights_only=False,
    )
    direct_pt = torch.load(
        os.path.join(ckpt_dir, f"seed{seed}_direct.pt"),
        map_location="cpu", weights_only=False,
    )
    return {"npz": npz, "hyper": hyper_pt, "direct": direct_pt}


def _build_hypernet_for_ebm(args_dict: Dict, D: int, device) -> "torch.nn.Module":
    """Rebuild the lift hypernet template matching the training config."""
    from lift.objectives.builders import build_hypernet_ebm
    _, hyper_E = build_hypernet_ebm(
        K=1, D=D,
        hidden_dim=int(args_dict["hidden_dim"]),
        nlayers=int(args_dict["nlayers"]),
        strong_convexity=float(args_dict["strong_convexity"]),
    )
    hyper_E = hyper_E.to(device)
    for p in hyper_E.parameters():
        p.requires_grad_(False)
    return hyper_E


def _fkl_loss_closure_ebm(
    hyper_E,
    ebm_template,
    flow,
    snap_x,
    x_val,
    *,
    D: int,
    D_aug: int,
    M_logZ_eval: int,
    device,
):
    """Placeholder: the closure is inlined in ``_compute_lifted_landscape_ebm``."""
    raise NotImplementedError("inlined in _compute_lifted_landscape_ebm")


def _compute_lifted_landscape_ebm(
    ckpt_dir: str,
    seed: int,
    args_dict: Dict,
    D: int,
    device,
    grid_n: int = 25,
    grid_radius: float = 1.0,
    direction_seed: int = 0,
) -> Dict:
    """Compute the (Theta_E, b_h) lifted landscape for the 1-D / 2-D rows.

    Uses a coarser grid than the constrained-space NPZ because the
    hypernetwork parameter space is about ten times larger and the per-point
    cost is dominated by a fresh hypernetwork forward.
    """
    import torch
    from torch.func import functional_call
    from lift.objectives.builders import build_hypernet_ebm
    from lift.objectives._batched import aug_energy_batched
    from lift.objectives._param_utils import detach_dict, stack_per_k_params
    from lift.objectives.partition import log_Z_via_grid_1d
    from lift.objectives.eval import build_eval_grid
    from lift.models import icnn_pos_param_names

    hyper_pt = torch.load(
        os.path.join(ckpt_dir, f"seed{seed}_hypernet.pt"),
        map_location="cpu", weights_only=False,
    )
    if "hyper_E_snaps" not in hyper_pt:
        raise RuntimeError(
            f"{ckpt_dir}/seed{seed}_hypernet.pt missing 'hyper_E_snaps' --- "
            "re-run training with the patched _snap callback."
        )

    # Rebuild the hypernet + EBM template.
    ebm_template, hyper_E = build_hypernet_ebm(
        K=1, D=D,
        hidden_dim=int(args_dict["hidden_dim"]),
        nlayers=int(args_dict["nlayers"]),
        strong_convexity=float(args_dict["strong_convexity"]),
    )
    hyper_E = hyper_E.to(device)
    ebm_template = ebm_template.to(device)
    for p in hyper_E.parameters():
        p.requires_grad_(False)
    for p in ebm_template.parameters():
        p.requires_grad_(False)
    param_keys = {k for k, _ in hyper_E.named_parameters()}

    # The full ``seed{S}_hypernet.pt`` also carries ``theta_snaps``,
    # ``flow_state``, ``history``, ``log_p_grid``, ``metrics`` and
    # ``ebm_state``, which can be GB-scale. Move only what is needed out,
    # then delete the pickle dict so the collector can reclaim the rest
    # before the per-grid-point forward pass allocates.
    phi_star_src = hyper_pt["hyper_E_state"]
    snaps_src = list(hyper_pt["hyper_E_snaps"])  # shallow copy of list
    snap_x_src = hyper_pt["snap_x"]
    snap_iters_src = list(hyper_pt.get("snap_iters", []) or [])
    del hyper_pt
    _free_memory("pickle drop (lifted-EBM)")

    phi_star: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in phi_star_src.items():
        if k in param_keys:
            phi_star[k] = v.detach().to(device).clone()
    del phi_star_src

    # Subsample the trajectory: the PCA needs only the directions of motion,
    # and holding every snapshot in device memory at once is the dominant
    # lifted-landscape allocation.
    snaps_kept, kept_idx = _subsample_traj(snaps_src, max_keep=12)
    snap_iters_kept = (
        [int(snap_iters_src[i]) for i in kept_idx]
        if snap_iters_src else []
    )
    traj = []
    for snap in snaps_kept:
        d = OrderedDict(
            (k, v.detach().to(device).clone())
            for k, v in snap.items()
            if k in param_keys
        )
        traj.append(d)
    del snaps_src, snaps_kept
    _free_memory("traj load (lifted-EBM)")

    # snap_x and x_val for the FKL closure.
    snap_x = snap_x_src.detach().to(device)
    del snap_x_src
    x_val_np = np.load(
        os.path.join(ckpt_dir, f"seed{seed}_x_val_landscape.npy")
    )
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)
    del x_val_np

    # Rebuild the target to build the eval grid for log Z.
    from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D
    import argparse
    ns = argparse.Namespace(**args_dict)
    expt = HypernetVsDirect1D(ns, target_kind=str(args_dict["target_kind"]))
    expt.device = device
    target = expt._build_target()

    n_grid_arg = int(args_dict.get("tv_grid_n", 8000))
    if D == 1:
        xs, _, dvol = build_eval_grid(target, n_grid_arg, device)
    elif D == 2:
        # Replicate _build_2d_grid from the lift lib (Riemann grid).
        import torch as _t
        n_axis = max(int(math.sqrt(n_grid_arg)), 96)
        (lo_x, hi_x), (lo_y, hi_y) = target.xlim
        xs_x = _t.linspace(lo_x, hi_x, n_axis, device=device)
        xs_y = _t.linspace(lo_y, hi_y, n_axis, device=device)
        gx, gy = _t.meshgrid(xs_x, xs_y, indexing="xy")
        xs = _t.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        dvol = float((hi_x - lo_x) * (hi_y - lo_y) / (n_axis * n_axis))
    else:
        raise NotImplementedError(f"D={D} not supported for lifted landscape")

    K = 1
    pos_names = icnn_pos_param_names(
        int(args_dict["nlayers"]), prefix="components.0.",
    )

    def loss_fn(phi_dict):
        theta = functional_call(hyper_E, dict(phi_dict), (snap_x,))
        theta = {
            k: (v.clamp(min=0.0) if k in pos_names else v)
            for k, v in theta.items()
        }
        ebm_stacked = detach_dict(
            stack_per_k_params(theta, K=K, prefix="components.")
        )
        log_Z = log_Z_via_grid_1d(
            ebm_template, theta, K=K, grid=xs, dvol=float(dvol),
        )
        log_pi = torch.log_softmax(theta["log_pi"], dim=-1)
        E_val = aug_energy_batched(
            ebm_template.components[0], ebm_stacked, x_val,
            K=K, D=D, D_aug=D,
        )
        log_q = torch.logsumexp(
            log_pi[:, None] - E_val - log_Z[:, None], dim=0
        )
        return (-log_q).mean()

    # Directions by PCA over the trajectory in lifted (Theta_E, b_h) space:
    # the top-2 right singular vectors of (snap_t - phi_star) span the
    # largest motion of training, so the rendered slice contains the
    # trajectory's geometry. Two filter-norm random directions in a space
    # this large would be nearly orthogonal to the trajectory and collapse
    # it to the origin.
    keys_phi = list(phi_star.keys())

    def _flatten_phi(d):
        return torch.cat([d[k].reshape(-1) for k in keys_phi])

    v_star = _flatten_phi(phi_star)
    diff_rows = torch.stack(
        [_flatten_phi(snap) - v_star for snap in traj], dim=0,
    )
    # SVD on CPU: with ``full_matrices=False`` the right-singular matrix is
    # (T, P) with P up to about 1e6, too large to materialize on the GPU but
    # comfortable on the host. Only the top 2 right singular vectors are
    # needed, so the work is bounded by T^2 * P.
    print(f"  [PCA] diff shape={tuple(diff_rows.shape)} (CPU SVD)")
    diff_rows_cpu = diff_rows.cpu()
    del diff_rows
    Uh, S, Vh = torch.linalg.svd(diff_rows_cpu, full_matrices=False)
    Vh = Vh.to(device)
    del Uh, S, diff_rows_cpu, v_star
    _free_memory("EBM lifted SVD done")
    # Rescale d1, d2 so the trajectory spans roughly [-1, 1].
    d1: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    d2: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    o = 0
    for k in keys_phi:
        n = phi_star[k].numel()
        d1[k] = Vh[0, o:o + n].view_as(phi_star[k])
        d2[k] = Vh[1, o:o + n].view_as(phi_star[k])
        o += n
    # Project, then rescale so the trajectory range is O(1).
    coords_pre = _project_trajectory(traj, phi_star, d1, d2)
    scale = float(max(np.max(np.abs(coords_pre)), 1e-12))
    for k in keys_phi:
        d1[k] = d1[k] * scale
        d2[k] = d2[k] * scale
    coords_pre = _project_trajectory(traj, phi_star, d1, d2)
    if coords_pre.shape[0] >= 1:
        max_abs = float(max(np.max(np.abs(coords_pre)), 1.0))
    else:
        max_abs = 1.0
    radius = max_abs * 1.35
    alphas = np.linspace(-radius, radius, grid_n)
    betas = np.linspace(-radius, radius, grid_n)
    Z = _evaluate_loss_grid(loss_fn, phi_star, d1, d2, alphas, betas)
    coords = _project_trajectory(traj, phi_star, d1, d2)
    out = {
        "alphas": alphas, "betas": betas, "Z": Z,
        "traj_coords": coords,
        "snap_iters": np.asarray(snap_iters_kept, dtype=np.int64),
    }
    del traj, phi_star, d1, d2, hyper_E, ebm_template
    _free_memory("EBM lifted landscape return")
    return out


# ---------------------------------------------------------- PCP-Map row helpers

def _load_pcpmap_lv_ckpt(ckpt_dir: str, seed: int = 0, backend: str = "hypernet") -> Dict:
    import torch
    sub = os.path.join(ckpt_dir, f"seed{seed}_{backend}")
    d = torch.load(
        os.path.join(sub, "ckpt.pt"), map_location="cpu", weights_only=False,
    )
    with open(os.path.join(sub, "history.json")) as f:
        hist = json.load(f)
    return {"ckpt": d, "history": hist, "dir": sub}


def _compute_pcpmap_landscapes(
    ckpt_dir: str,
    args_dict: Dict,
    device,
    grid_n_constrained: int = 21,
    grid_n_lifted: int = 21,
    grid_radius: float = 1.0,
    direction_seed: int = 0,
    seed: int = 2,
) -> Dict:
    """Build the panel-(c) (shared PICNN-theta) and panel-(d) (lifted) arrays.

    Both panels evaluate the same PCP-Map NLL closure, panel (c) at the
    converged hypernetwork's PICNN parameters and panel (d) at the converged
    hypernetwork state dict. The direct model's converged PICNN is projected
    onto the panel-(c) (alpha, beta) plane by the same 2x2 normal equations
    the pipeline uses for trajectory projection.
    """
    import torch
    from torch.func import functional_call
    from lift.baselines.picnn import PICNN
    from lift.baselines.pcpmap import PCPMap
    from lift.baselines.picnn_hypernet import PicnnHypernet
    from lift.dataset.lotka_volterra import LotkaVolterraDataset
    from torch import distributions

    # 1) Load the converged lift and direct checkpoints at the given seed.
    hyp_pkg = _load_pcpmap_lv_ckpt(ckpt_dir, seed=seed, backend="hypernet")
    dir_pkg = _load_pcpmap_lv_ckpt(ckpt_dir, seed=seed, backend="direct")
    _free_memory("PCP-Map ckpts loaded")

    # 2) Rebuild dataset and a val batch.
    # The PCP-Map training loss is -E_{(theta, y) ~ p_data}[ log p_base(T(theta;y))
    # + log|det H_theta Phi(theta;y)| ]. The closure below must average over the
    # same heterogeneous (theta_i, y_i) pairs the network was trained on: the
    # converged map only learned to push those pairs onto N(0, I), so replacing
    # per-sample y_i with one shared y_ref sends T(theta_i; y_ref) far outside
    # the Gaussian's bulk and the landscape is wrong. Pass the real per-sample
    # y from the validation batch.
    ds = LotkaVolterraDataset()
    D_x_full = int(ds.D_x)
    D_y = int(ds.D_y)
    val = ds.val_data.to(device)
    # A large fixed batch, so the per-grid-point loss estimate has low
    # variance and the contours are smooth. The same batch is reused at every
    # grid point, with no resampling, per the Li-2018 landscape recipe.
    n_landscape_batch = min(int(os.environ.get("PCPMAP_LANDSCAPE_BATCH", 2048)),
                            val.shape[0])
    x_val_b = val[:n_landscape_batch, :D_x_full].contiguous()
    y_val_b = val[:n_landscape_batch, D_x_full:].contiguous()

    # 3) Rebuild model objects matching the training config.
    pos_mode = str(args_dict.get("pos_constraint_mode", "cone"))
    feat_dim = int(args_dict["feature_dim"])
    feat_y_dim = int(args_dict["feature_y_dim"])
    num_layers = int(args_dict["num_layers_pi"])
    hyper_hidden = str(args_dict.get("hyper_hidden_sizes", "128,128,192"))
    cond_dim = int(args_dict.get("cond_dim", 8))

    picnn = PICNN(
        input_x_dim=D_x_full, input_y_dim=D_y,
        feature_dim=feat_dim, feature_y_dim=feat_y_dim,
        out_dim=1, num_layers=num_layers,
        pos_constraint_mode=pos_mode,
    ).to(device)
    prior = distributions.MultivariateNormal(
        torch.zeros(D_x_full, device=device),
        torch.eye(D_x_full, device=device),
    )
    # Hypernet object (override path)
    hyp = PicnnHypernet(
        picnn.lw_weight_shapes, cond_dim=cond_dim,
        hidden_sizes=hyper_hidden, pos_constraint_mode=pos_mode,
    ).to(device)
    for p in hyp.parameters():
        p.requires_grad_(False)
    pcp_hyp = PCPMap(prior, picnn, pos_weights_supplier=hyp.emit).to(device)
    pcp_hyp.load_state_dict(hyp_pkg["ckpt"]["pcp_state"], strict=False)
    hyp.load_state_dict(hyp_pkg["ckpt"]["hyp_state"], strict=True)
    for p in pcp_hyp.parameters():
        p.requires_grad_(False)

    # ------------- panel (c): shared PICNN-theta landscape ----------------
    # Center on the converged hypernet's *emitted* Lw weights (override dict).
    with torch.no_grad():
        emitted = hyp.emit()  # dict[int -> tensor]
    # Build params_dict over the override slots (keyed by layer-idx string).
    picnn_params_star: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in emitted.items():
        picnn_params_star[str(int(k))] = v.detach().to(device).clone()

    # The PICNN forward takes every other weight from pcp_hyp.picnn; only the
    # Lw slots are overridden.
    def _nll_at_picnn_override(override_dict: Dict[str, "torch.Tensor"]):
        # Reassemble the int-keyed override dict expected by PICNN.
        int_dict = {int(k): v for k, v in override_dict.items()}
        # Bypass pcp_hyp.emit by temporarily swapping in a supplier that
        # returns this override.
        saved_sup = pcp_hyp.pos_weights_supplier
        try:
            pcp_hyp.pos_weights_supplier = lambda: int_dict
            x = x_val_b.detach().clone().requires_grad_(True)
            ll = pcp_hyp.loglik_picnn(x, y_val_b)
            return -ll.mean()
        finally:
            pcp_hyp.pos_weights_supplier = saved_sup

    # The direct model's converged PICNN, projected onto the same (d1, d2)
    # plane. It stores the raw Lw weights inside pcp_dir.picnn.Lw[k].
    picnn_dir = PICNN(
        input_x_dim=D_x_full, input_y_dim=D_y,
        feature_dim=feat_dim, feature_y_dim=feat_y_dim,
        out_dim=1, num_layers=num_layers,
        pos_constraint_mode=pos_mode,
    ).to(device)
    pcp_dir = PCPMap(prior, picnn_dir, pos_weights_supplier=None).to(device)
    pcp_dir.load_state_dict(dir_pkg["ckpt"]["pcp_state"], strict=False)
    # Extract direct's Lw[k].weight in the same (positive-projected) form.
    direct_override: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    with torch.no_grad():
        for k, lw in enumerate(pcp_dir.picnn.Lw):
            # The same positivity convention (cone => relu), so the numbers
            # live in the same submanifold as the emitted override.
            w = lw.weight.detach().to(device)
            if pos_mode == "cone":
                w = w.clamp(min=0.0)
            direct_override[str(k)] = w

    # Directions chosen so the converged direct PICNN lands at a visible
    # non-origin spot on the (alpha, beta) plane: d1 points from the lift's
    # converged PICNN to the direct one, filter-normalized per layer so the
    # trajectory unit matches each layer's converged Frobenius norm, and d2
    # is a filter-norm random direction Gram-Schmidt orthogonalized against
    # d1. Filter normalization per Li 2018.
    keys_c = list(picnn_params_star.keys())
    diff_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        diff_dict[k] = (direct_override[k] - picnn_params_star[k]).detach()
    # Rescale per layer so each layer's difference is L2-matched to the
    # converged center's per-layer norm, which puts d1 on the Li-2018
    # filter-normalized manifold. A layer with zero difference gets a zero
    # direction.
    d1_c: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        w = picnn_params_star[k]; r = diff_dict[k]
        if w.ndim >= 2 and w.size(0) > 1:
            n = w.size(0)
            rf = r.view(n, -1); wf = w.view(n, -1)
            rn = rf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            wn = wf.norm(dim=-1, keepdim=True)
            d1_c[k] = (rf * (wn / rn)).view_as(w)
        else:
            rn = r.norm().clamp(min=1e-12)
            d1_c[k] = r * (w.norm() / rn)
    # Random filter-norm direction; Gram-Schmidt out d1.
    g = torch.Generator(device=device); g.manual_seed(int(direction_seed))
    d2_c_raw = _filter_norm_direction(picnn_params_star, generator=g)
    # Flatten + GS
    def _flat(dct):
        return torch.cat([dct[k].reshape(-1) for k in keys_c])
    v_d1 = _flat(d1_c); v_d2 = _flat(d2_c_raw)
    denom = max(float((v_d1 * v_d1).sum()), 1e-20)
    coef = float((v_d2 * v_d1).sum()) / denom
    d2_c: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        d2_c[k] = (d2_c_raw[k] - coef * d1_c[k]).detach()
    # Project direct onto (d1_c, d2_c) -- by construction this is near (1, 0).
    direct_proj_c = _project_trajectory(
        [direct_override], picnn_params_star, d1_c, d2_c,
    )

    # ----- per-iter trajectory projections (panel c) -----
    # The lift checkpoint stores hyper_snaps (a full hypernetwork state dict
    # per snapshot) and the direct one stores picnn_snaps. Project both onto
    # the (d1_c, d2_c) plane under the same positivity clamp the painted
    # override uses, so the numbers live in the same submanifold as the
    # converged center.
    hyper_snaps = hyp_pkg["ckpt"].get("hyper_snaps", []) or []
    picnn_snaps_direct = dir_pkg["ckpt"].get("picnn_snaps", []) or []
    snap_iters_h = hyp_pkg["ckpt"].get("snap_iters", []) or []
    snap_iters_d = dir_pkg["ckpt"].get("snap_iters", []) or []
    # Subsample the snapshot lists to bound the PCA matrix size and the
    # device-cloned trajectory footprint in panel (d).
    if hyper_snaps:
        hyper_snaps, _hk_idx = _subsample_traj(hyper_snaps, max_keep=12)
        if snap_iters_h:
            snap_iters_h = [int(snap_iters_h[i]) for i in _hk_idx]
    if picnn_snaps_direct:
        picnn_snaps_direct, _dk_idx = _subsample_traj(
            picnn_snaps_direct, max_keep=12,
        )
        if snap_iters_d:
            snap_iters_d = [int(snap_iters_d[i]) for i in _dk_idx]

    # Emit an override at each snapshot, then project.
    hyper_overrides: List["OrderedDict[str, torch.Tensor]"] = []
    if hyper_snaps:
        # Strip non-parameter buffers (``cond`` is a buffer in some
        # variants) by intersecting with the actual parameter names.
        hyp_param_names = {k for k, _ in hyp.named_parameters()}
        saved_state = {
            k: v.detach().clone() for k, v in hyp.state_dict().items()
        }
        try:
            for snap in hyper_snaps:
                # Load the snapshot, emit, and convert to a
                # picnn_params_star-shaped override (string-keyed, clamped).
                # A snapshot may lack some buffers, hence strict=False.
                load_dict = {
                    k: v.detach().to(device) for k, v in snap.items()
                }
                hyp.load_state_dict(load_dict, strict=False)
                with torch.no_grad():
                    emitted_t = hyp.emit()
                ov: "OrderedDict[str, torch.Tensor]" = OrderedDict()
                for k, v in emitted_t.items():
                    w = v.detach().to(device)
                    if pos_mode == "cone":
                        w = w.clamp(min=0.0)
                    ov[str(int(k))] = w
                hyper_overrides.append(ov)
        finally:
            hyp.load_state_dict(saved_state, strict=False)

    # Direct snaps -> extract Lw[k].weight, clamp, str-key.
    direct_overrides_traj: List["OrderedDict[str, torch.Tensor]"] = []
    if picnn_snaps_direct:
        # The PICNN names its Lw layers "Lw.0.weight", "Lw.1.weight", and so
        # on; take the indices present in the converged direct_override.
        for snap in picnn_snaps_direct:
            ov: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k in keys_c:
                key = f"Lw.{int(k)}.weight"
                if key not in snap:
                    # Layer absent from the snapshot: use the center, which
                    # projects to 0.
                    ov[k] = picnn_params_star[k].clone()
                    continue
                w = snap[key].detach().to(device)
                if pos_mode == "cone":
                    w = w.clamp(min=0.0)
                ov[k] = w
            direct_overrides_traj.append(ov)

    # Project the trajectories onto (d1_c, d2_c).
    hyper_traj_c = (
        _project_trajectory(hyper_overrides, picnn_params_star, d1_c, d2_c)
        if hyper_overrides else np.zeros((0, 2), dtype=np.float64)
    )
    direct_traj_c = (
        _project_trajectory(direct_overrides_traj, picnn_params_star,
                            d1_c, d2_c)
        if direct_overrides_traj else direct_proj_c
    )

    # Choose grid radius so all projected coords are within frame.
    max_abs_c = 1.0
    for arr in (direct_proj_c, hyper_traj_c, direct_traj_c):
        if arr.size > 0:
            max_abs_c = max(max_abs_c, float(np.nanmax(np.abs(arr))))
    radius_c = max_abs_c * 1.4
    alphas_c = np.linspace(-radius_c, radius_c, grid_n_constrained)
    betas_c = np.linspace(-radius_c, radius_c, grid_n_constrained)
    Z_c = _evaluate_loss_grid_with_grad(
        _nll_at_picnn_override, picnn_params_star, d1_c, d2_c, alphas_c, betas_c,
    )

    # ------------- panel (d): lifted (Theta_E, b_h) landscape -------------
    # Center on converged hypernet state_dict (parameter-only keys).
    phi_param_keys = {k for k, _ in hyp.named_parameters()}
    phi_star: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in hyp.state_dict().items():
        if k in phi_param_keys:
            phi_star[k] = v.detach().to(device).clone()

    def _nll_at_phi(phi_dict):
        emitted_local = functional_call(hyp, dict(phi_dict), ())
        return None  # placeholder; ``loss_fn_phi`` below is what runs

    # PicnnHypernet defines emit() and no forward(), which functional_call
    # requires, so wrap it in a module whose forward() is emit().
    class _HypForwardWrap(torch.nn.Module):
        def __init__(self, host: PicnnHypernet) -> None:
            super().__init__()
            self.host = host

        def forward(self):  # noqa: D401
            return self.host.emit()

    hyp_wrap = _HypForwardWrap(hyp).to(device)
    for p in hyp_wrap.parameters():
        p.requires_grad_(False)

    # functional_call expects parameter names rooted at the wrapper, so
    # phi_star is re-keyed with the "host." prefix the wrapper adds.
    phi_star_wrap: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in phi_star.items():
        phi_star_wrap[f"host.{k}"] = v

    def loss_fn_phi(phi_dict):
        emitted_local = functional_call(hyp_wrap, dict(phi_dict), ())
        int_dict = {int(k): v for k, v in emitted_local.items()}
        saved_sup = pcp_hyp.pos_weights_supplier
        try:
            pcp_hyp.pos_weights_supplier = lambda: int_dict
            x = x_val_b.detach().clone().requires_grad_(True)
            ll = pcp_hyp.loglik_picnn(x, y_val_b)
            return -ll.mean()
        finally:
            pcp_hyp.pos_weights_supplier = saved_sup

    # Directions by PCA on the snapshot trajectory, the same recipe as the
    # 1-D and 2-D lifted panels, falling back to filter-norm random
    # directions when no snapshots were stored.
    hyper_traj_d = np.zeros((0, 2), dtype=np.float64)
    if hyper_snaps:
        # Rekey each snap with the "host." prefix that wraps phi_star_wrap.
        snap_param_names = set(phi_star.keys())
        traj_phi_wrap: List["OrderedDict[str, torch.Tensor]"] = []
        for snap in hyper_snaps:
            d = OrderedDict()
            for k, v in snap.items():
                if k in snap_param_names:
                    d[f"host.{k}"] = v.detach().to(device).clone()
            # Backfill any missing parameter keys with phi_star_wrap.
            for k in phi_star_wrap.keys():
                if k not in d:
                    d[k] = phi_star_wrap[k].detach().clone()
            traj_phi_wrap.append(d)
        # PCA on (snap - phi_star_wrap) with a CPU-side SVD: the top-2 right
        # singular vectors span the largest training motion in lifted
        # (Theta_E, b_h) space.
        keys_phi = list(phi_star_wrap.keys())
        def _flat_phi(dd):
            return torch.cat([dd[k].reshape(-1) for k in keys_phi])
        v_star = _flat_phi(phi_star_wrap)
        diff_rows = torch.stack(
            [_flat_phi(s) - v_star for s in traj_phi_wrap], dim=0,
        ).cpu()
        print(f"  [LV-lifted PCA] diff shape={tuple(diff_rows.shape)}")
        Uh, S, Vh = torch.linalg.svd(diff_rows, full_matrices=False)
        Vh = Vh.to(device)
        del Uh, S, diff_rows, v_star
        _free_memory("PCP-Map lifted SVD done")
        d1_d: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        d2_d: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        o = 0
        for k in keys_phi:
            n = phi_star_wrap[k].numel()
            d1_d[k] = Vh[0, o:o + n].view_as(phi_star_wrap[k])
            d2_d[k] = Vh[1, o:o + n].view_as(phi_star_wrap[k])
            o += n
        coords_pre = _project_trajectory(
            traj_phi_wrap, phi_star_wrap, d1_d, d2_d,
        )
        scale = float(max(np.max(np.abs(coords_pre)), 1e-12))
        for k in keys_phi:
            d1_d[k] = d1_d[k] * scale
            d2_d[k] = d2_d[k] * scale
        hyper_traj_d = _project_trajectory(
            traj_phi_wrap, phi_star_wrap, d1_d, d2_d,
        )
        max_abs_d = float(max(np.max(np.abs(hyper_traj_d)), 1.0))
        radius_d = max_abs_d * 1.35
    else:
        g2 = torch.Generator(device=device); g2.manual_seed(int(direction_seed))
        d1_d = _filter_norm_direction(phi_star_wrap, generator=g2)
        d2_d = _filter_norm_direction(phi_star_wrap, generator=g2)
        radius_d = float(grid_radius)
    alphas_d = np.linspace(-radius_d, radius_d, grid_n_lifted)
    betas_d = np.linspace(-radius_d, radius_d, grid_n_lifted)
    Z_d = _evaluate_loss_grid_with_grad(
        loss_fn_phi, phi_star_wrap, d1_d, d2_d, alphas_d, betas_d,
    )

    out = {
        "panel_c": {
            "alphas": alphas_c, "betas": betas_c, "Z": Z_c,
            "direct_proj": direct_proj_c,
            "hyper_traj": hyper_traj_c,
            "direct_traj": direct_traj_c,
            "snap_iters_h": np.asarray(snap_iters_h, dtype=np.int64),
            "snap_iters_d": np.asarray(snap_iters_d, dtype=np.int64),
        },
        "panel_d": {
            "alphas": alphas_d, "betas": betas_d, "Z": Z_d,
            "hyper_traj": hyper_traj_d,
        },
        "hyp_history": hyp_pkg["history"],
        "dir_history": dir_pkg["history"],
    }
    # Drop the large pickle dicts and the per-snapshot device clones, so the
    # caller's next problem row does not stack host memory.
    del hyp_pkg, dir_pkg, hyper_snaps, picnn_snaps_direct
    try:
        del hyper_overrides, direct_overrides_traj  # may be unset on early returns
    except NameError:
        pass
    try:
        del traj_phi_wrap
    except NameError:
        pass
    _free_memory("PCP-Map landscape return")
    return out


# ------------------------------------------------------------- panel painters

def _paint_landscape_panel(
    ax,
    *,
    alphas, betas, Z,
    hyper_coords=None,
    direct_coords=None,
    hyper_final_star: bool = True,
    direct_final_plus: bool = True,
    show_legend: bool = False,
    title: str = "",
    xlabel: str = r"$\alpha$ (filter-norm units)",
    ylabel: str = r"$\beta$ (filter-norm units)",
    hyper_loss=None,
    direct_loss=None,
):
    """Render one panel: viridis filled contours + per-panel colorbar.

    The colormap range is capped at min(75th percentile of log_Z, log10=3),
    so corner outliers, such as the lifted panels where the NLL explodes far
    from the converged lift, do not drown out the basin structure the figure
    exists to show.
    """
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    Z_safe = Z - np.nanmin(Z) + 1e-8
    log_Z = np.log10(Z_safe)
    finite = log_Z[np.isfinite(log_Z)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    # The cap is the 75th percentile with a hard ceiling at log10 = 3, so a
    # divergent lifted panel still shows basin structure; constrained panels
    # sit well below the ceiling and are unaffected. vmin is then clamped to
    # vmax - 3.5 so the basin and any local-minimum trap both read as dark
    # depressions against the wall.
    if finite.size:
        vmax = float(min(np.percentile(finite, 75.0), 3.0))
    else:
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    vmin = max(vmin, vmax - 3.5)
    levels = np.linspace(vmin, vmax, 25)
    cs = ax.contourf(
        A, B, log_Z, levels=levels, cmap="viridis", extend="both",
        vmin=vmin, vmax=vmax,
    )
    cb = ax.figure.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(
        r"$\log_{10}(\mathrm{loss}-\min\mathrm{loss})$", fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)

    if hyper_coords is not None and len(hyper_coords) >= 1:
        ax.plot(
            hyper_coords[:, 0], hyper_coords[:, 1],
            linestyle=_HYP_LS, marker=_HYP_MARKER,
            color=_HYP_COLOR, lw=1.4, ms=2.8,
            mec="black", mew=0.25,
            label="hypernet (orange)" if show_legend else None,
            zorder=4, alpha=0.95,
        )
    if direct_coords is not None and len(direct_coords) >= 1:
        ax.plot(
            direct_coords[:, 0], direct_coords[:, 1],
            linestyle=_DIRECT_LS, marker=_DIRECT_MARKER,
            color=_DIRECT_COLOR, lw=1.4, ms=3.2,
            mec=_DIRECT_COLOR, mew=0.8, mfc=_DIRECT_MFC,
            label="direct softplus (red)" if show_legend else None,
            zorder=4, alpha=0.95,
        )
    if hyper_final_star:
        ax.plot(
            [0], [0], "*",
            color="gold", ms=16, mec="black", mew=0.6, zorder=6,
            label=r"converged hypernet $\boldsymbol{\theta}^{\star}$" if show_legend else None,
        )
    if direct_final_plus and direct_coords is not None and len(direct_coords) >= 1:
        ax.plot(
            direct_coords[-1, 0], direct_coords[-1, 1], "P",
            color=_DIRECT_COLOR, ms=11, mec="black", mew=0.6, zorder=6,
            label="converged direct" if show_legend else None,
        )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)


def _add_loss_sparkline_legacy(ax, train_loss, *, log_y: bool = True) -> None:
    """Wrapper around :func:`lift.utils.loss_sparkline.add_loss_sparkline`,
    which carries the paper-wide sparkline style.
    """
    add_loss_sparkline(ax, [(np.asarray(train_loss), _HYP_COLOR)],
                       log_y=log_y)


# -------------------------------------------------------------- main driver

def main() -> None:
    import argparse
    import matplotlib.pyplot as plt
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-lifted", action="store_true",
        help="Skip the (Θ_E, b_h) lifted-landscape compute on every "
             "row and render the right column from cached NPZs if "
             "present, blank otherwise. Use this when the lifted "
             "compute risks OOM on the host (the full hypernet pickle + "
             "per-grid-point functional_call create-graph evaluations "
             "can climb to multi-GB CPU RAM; a partial-OOM kills the "
             "controlling shell, including any Claude session that "
             "launched the render).",
    )
    parser.add_argument(
        "--grid-n-lifted-1d", type=int, default=25,
        help="Override the 1-D row's lifted-grid size (default 25).",
    )
    parser.add_argument(
        "--grid-n-lifted-2d", type=int, default=21,
        help="Override the 2-D row's lifted-grid size (default 21).",
    )
    parser.add_argument(
        "--npz-basename", type=str, default="seed{seed}_landscape_121.npz",
        help="Basename pattern for the constrained-space NPZ to read on "
             "rows 1 + 2. ``{seed}`` is replaced with the seed index. "
             "Defaults to the 121x121 hybrid NPZ; pass "
             "'seed{seed}_landscape.npz' to fall back to the legacy 41x41.",
    )
    cli_args = parser.parse_args()

    apply_paper_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[landscape-figure] device={device}", flush=True)
    _free_memory("start")

    # ------- 1-D Gumbel: find the checkpoint that has hyper_E_snaps -------
    print("[1-D] locating re-run ckpt...")
    ckpt_1d_prefix = (
        "experiments_hypernet_vs_direct_1d_target_kind-gumbel_"
        "objective-fkl-direct_n_iters-3000"
    )
    ckpt_1d = _find_ckpt_dir(ckpt_1d_prefix)
    print(f"[1-D] using {os.path.basename(ckpt_1d)}")

    # Recover the config from the json the driver was run with.
    with open(os.path.join(_REPO_ROOT, "configs",
                           "experiments_hypernet_vs_direct_1d.json")) as f:
        args_1d = json.load(f)

    seed1d = 0
    npz_1d = None
    # The hybrid (delta-fin + PCA-orth) NPZ written by the training
    # pipeline. The default basename is the 121x121 NPZ, with a fallback to
    # the 41x41 ``seed{seed}_landscape.npz``.
    npz_path_1d = os.path.join(
        ckpt_1d, cli_args.npz_basename.format(seed=seed1d),
    )
    if not os.path.isfile(npz_path_1d):
        legacy = os.path.join(ckpt_1d, f"seed{seed1d}_landscape.npz")
        print(f"[1-D] requested NPZ {os.path.basename(npz_path_1d)} missing; "
              f"falling back to legacy {os.path.basename(legacy)}")
        npz_path_1d = legacy
    if os.path.isfile(npz_path_1d):
        print(f"[1-D] loading {os.path.basename(npz_path_1d)}")
        with np.load(npz_path_1d) as nf:
            npz_1d = {k: np.asarray(nf[k]) for k in nf.files}
    hist_1d = _load_ebm_loss_history(ckpt_1d, seed=seed1d)
    print(f"[1-D] loss histories hyper={len(hist_1d['hyper'])} "
          f"direct={len(hist_1d['direct'])}")

    if cli_args.skip_lifted:
        print("[1-D] --skip-lifted: skipping lifted-landscape compute", flush=True)
        lifted_1d = None
    else:
        print("[1-D] computing lifted-space (Theta_E, b_h) landscape...", flush=True)
        lifted_1d = _compute_lifted_landscape_ebm(
            ckpt_1d, seed=seed1d, args_dict=args_1d, D=1, device=device,
            grid_n=int(cli_args.grid_n_lifted_1d),
        )
        print(f"[1-D] lifted Z range "
              f"[{lifted_1d['Z'].min():.4g}, {lifted_1d['Z'].max():.4g}]; "
              f"traj {lifted_1d['traj_coords'].shape}", flush=True)
        _free_memory("after 1-D lifted")

    # ------- 2-D gamma-mode -------
    print("[2-D] locating re-run ckpt...")
    ckpt_2d_prefix = (
        "experiments_hypernet_vs_direct_2d_k1_target_kind-gamma_mode_2d_"
        "objective-fkl-direct_K-1_"
    )
    # Prefer the checkpoint that already carries the hybrid delta-fin +
    # PCA-orth NPZ (``f036aa7d``); the newer re-run does not have it.
    try:
        ckpt_2d = _find_ckpt_dir(ckpt_2d_prefix, hash_substr="f036aa7d")
    except FileNotFoundError:
        ckpt_2d = _find_ckpt_dir(ckpt_2d_prefix)
    print(f"[2-D] using {os.path.basename(ckpt_2d)}")
    with open(os.path.join(_REPO_ROOT, "configs",
                           "experiments_hypernet_vs_direct_2d_k1.json")) as f:
        args_2d = json.load(f)
    seed2d = 0
    npz_2d = None
    # The hybrid (delta-fin + PCA-orth) NPZ, with the same fallback as the
    # 1-D row.
    npz_path_2d = os.path.join(
        ckpt_2d, cli_args.npz_basename.format(seed=seed2d),
    )
    if not os.path.isfile(npz_path_2d):
        legacy = os.path.join(ckpt_2d, f"seed{seed2d}_landscape.npz")
        print(f"[2-D] requested NPZ {os.path.basename(npz_path_2d)} missing; "
              f"falling back to legacy {os.path.basename(legacy)}")
        npz_path_2d = legacy
    if os.path.isfile(npz_path_2d):
        print(f"[2-D] loading {os.path.basename(npz_path_2d)}")
        with np.load(npz_path_2d) as nf:
            npz_2d = {k: np.asarray(nf[k]) for k in nf.files}
    hist_2d = _load_ebm_loss_history(ckpt_2d, seed=seed2d)
    print(f"[2-D] loss histories hyper={len(hist_2d['hyper'])} "
          f"direct={len(hist_2d['direct'])}")

    if cli_args.skip_lifted:
        print("[2-D] --skip-lifted: skipping lifted-landscape compute", flush=True)
        lifted_2d = None
    else:
        print("[2-D] computing lifted-space (Theta_E, b_h) landscape...", flush=True)
        lifted_2d = _compute_lifted_landscape_ebm(
            ckpt_2d, seed=seed2d, args_dict=args_2d, D=2, device=device,
            grid_n=int(cli_args.grid_n_lifted_2d),  # 2-D D-grid is expensive
        )
        print(f"[2-D] lifted Z range "
              f"[{lifted_2d['Z'].min():.4g}, {lifted_2d['Z'].max():.4g}]; "
              f"traj {lifted_2d['traj_coords'].shape}", flush=True)
        _free_memory("after 2-D lifted")


    # ----------------------- assemble the 2x2 figure --------------------------
    print("[fig] assembling 2x2 layout...")
    fig = plt.figure(figsize=(11.6, 8.8))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.42, wspace=0.30,
        left=0.075, right=0.97, top=0.93, bottom=0.07,
    )
    row_labels = [
        "1-D Gumbel\n(ICNN-EBM)",
        "2-D gamma-mode\n(ICNN-EBM)",
    ]
    col_titles = [
        "Constrained problem (shared ICNN/PICNN $\\boldsymbol{\\theta}$)",
        "Lifted problem (hypernetwork $\\boldsymbol{\\phi},\\,b$)",
    ]

    def _coords_h(npz):
        # The NPZ names this key either ``coords_hyper`` or
        # ``coords_hypernet``.
        for k in ("coords_hyper", "coords_hypernet"):
            if k in npz:
                return np.asarray(npz[k])
        return None

    # Row 1 left: shared theta from the cached NPZ, both trajectories.
    ax00 = fig.add_subplot(gs[0, 0])
    if npz_1d is not None:
        _paint_landscape_panel(
            ax00,
            alphas=npz_1d["alphas"], betas=npz_1d["betas"], Z=npz_1d["Z"],
            hyper_coords=_coords_h(npz_1d),
            direct_coords=np.asarray(npz_1d["coords_direct"]),
            show_legend=True,
            title="shared $\\boldsymbol{\\theta}$ space",
            hyper_loss=hist_1d["hyper"],
            direct_loss=hist_1d["direct"],
        )
    # Row 1 right: lifted, the lift trajectory only.
    ax01 = fig.add_subplot(gs[0, 1])
    if lifted_1d is not None:
        _paint_landscape_panel(
            ax01,
            alphas=lifted_1d["alphas"], betas=lifted_1d["betas"], Z=lifted_1d["Z"],
            hyper_coords=lifted_1d["traj_coords"],
            direct_coords=None,
            direct_final_plus=False,
            title="lifted $(\\boldsymbol{\\phi},\\,b)$ space",
            hyper_loss=hist_1d["hyper"],
        )
    else:
        ax01.text(0.5, 0.5, "lifted panel skipped\n(--skip-lifted)",
                  ha="center", va="center", fontsize=11, color="0.4",
                  transform=ax01.transAxes)
        ax01.set_xticks([]); ax01.set_yticks([])
        ax01.set_title("lifted $(\\boldsymbol{\\phi},\\,b)$ space", fontsize=11)

    # Row 2
    ax10 = fig.add_subplot(gs[1, 0])
    if npz_2d is not None:
        _paint_landscape_panel(
            ax10,
            alphas=npz_2d["alphas"], betas=npz_2d["betas"], Z=npz_2d["Z"],
            hyper_coords=_coords_h(npz_2d),
            direct_coords=np.asarray(npz_2d["coords_direct"]),
            title="shared $\\boldsymbol{\\theta}$ space",
            hyper_loss=hist_2d["hyper"],
            direct_loss=hist_2d["direct"],
        )
    ax11 = fig.add_subplot(gs[1, 1])
    if lifted_2d is not None:
        _paint_landscape_panel(
            ax11,
            alphas=lifted_2d["alphas"], betas=lifted_2d["betas"], Z=lifted_2d["Z"],
            hyper_coords=lifted_2d["traj_coords"],
            direct_coords=None,
            direct_final_plus=False,
            title="lifted $(\\boldsymbol{\\phi},\\,b)$ space",
            hyper_loss=hist_2d["hyper"],
        )
    else:
        ax11.text(0.5, 0.5, "lifted panel skipped\n(--skip-lifted)",
                  ha="center", va="center", fontsize=11, color="0.4",
                  transform=ax11.transAxes)
        ax11.set_xticks([]); ax11.set_yticks([])
        ax11.set_title("lifted $(\\boldsymbol{\\phi},\\,b)$ space", fontsize=11)


    # Column titles (top-of-figure)
    for ax, t in [(ax00, col_titles[0]), (ax01, col_titles[1])]:
        ax.annotate(
            t, xy=(0.5, 1.18), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=12,
            annotation_clip=False,
        )
    # Row labels on the left (outside the y-axis)
    for ax, lbl in [(ax00, row_labels[0]),
                    (ax10, row_labels[1])]:
        ax.annotate(
            lbl, xy=(-0.27, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90, fontsize=11,
            annotation_clip=False,
        )

    fig.savefig(_OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {_OUT_PDF}")


if __name__ == "__main__":
    main()
