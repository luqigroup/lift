r"""Two-space loss-landscape figure for the convex potential flow (CPFlow)
on the 2-D 8-Gaussians target.

The plotted loss is the CPFlow change-of-variables NLL

    -log p(x) = -log N(T(x); 0, I) - log det H_phi(x),

with T(x) = grad phi(x) and phi the ICNN convex potential. The left panel
is constrained ICNN-theta space, centered on the converged emitted PICNN
parameters: d1 is the end-difference direction, direct minus the lift,
filter-normalized per layer after Li 2018, and d2 is the top
trajectory-PCA component Gram-Schmidt-orthogonalized against d1. The
right panel is the lifted (Theta_E, b_h) space, centered on the converged
hypernetwork state and spanned by the top-2 PCA axes of its trajectory.
Two random filter-norm directions miss the narrow channel between the
methods at these dimensions and collapse either panel to a featureless
bowl, which is why neither panel uses them.

Requires checkpoints written with ``--snap_every > 0``, which hold the
per-iteration ``picnn_snaps`` and ``hyper_snaps``. The figure is written
to ``figures/landscape_cpflow_2d.pdf``.
"""
from __future__ import annotations

import gc
import math
import os
import sys
from collections import OrderedDict


def _free_memory(note: str = "") -> None:
    """Release CPU and GPU memory between lifted-landscape phases."""
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


def _subsample_traj(snaps, max_keep: int = 12):
    import numpy as _np
    T = len(snaps)
    if T <= max_keep:
        return list(snaps), _np.arange(T, dtype=_np.int64)
    idx = _np.linspace(0, T - 1, max_keep).round().astype(_np.int64)
    idx = _np.unique(idx)
    return [snaps[int(i)] for i in idx], idx
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The experiment drivers are not part of the ``lift`` package, so
# scripts/ goes on sys.path to import the CPFlow training-loss helpers.
_SCRIPTS_DIR = _THIS_DIR
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from lift.utils.landscape.render_generic import (  # noqa: E402
    _filter_norm_direction,
    _project_trajectory,
)
from lift.utils.loss_sparkline import add_loss_sparkline  # noqa: E402
from lift.utils.paper_style import apply_paper_style  # noqa: E402


# Colors come from ``lift/utils/paper_style.PALETTE``: the lift is orange
# and direct softplus is red. The line styles and markers keep the two
# readable in grayscale.
_HYP_COLOR = "#ff7f0e"
_DIRECT_COLOR = "#d62728"
_HYP_LS = "-"
_DIRECT_LS = "--"
_HYP_MARKER = "s"
_DIRECT_MARKER = "o"
_DIRECT_MFC = "none"

# Default checkpoint directory; --ckpt_dir overrides it.
_CKPT_DIR_DEFAULT = os.path.join(
    _REPO_ROOT, "data", "checkpoints",
    "experiments_cpflow_demo_2d_landscape",
)
_OUT_PDF = os.path.join(
    _REPO_ROOT, "figures", "figures",
    "landscape_cpflow_2d.pdf",
)


# ---------------------------------------------------------- loss closure


def _build_cpflow_loss_closure(
    *,
    template,
    x_val,
):
    """Return a callable ``params -> scalar NLL``.

    The closure calls the training driver's own ``model_log_prob`` rather
    than reimplementing it, so the landscape evaluates the same loss the
    training loop minimized.
    """
    from experiments_cpflow_demo_2d import model_log_prob  # noqa: E402

    def loss_fn(params_dict):
        device = next(iter(template.parameters())).device
        params = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in params_dict.items()
        }
        x = x_val.detach().clone()
        logp = model_log_prob(template, params, x)
        return -logp.mean()

    return loss_fn


def _evaluate_loss_grid_with_grad(
    loss_fn,
    params_dict,
    d1,
    d2,
    alphas,
    betas,
):
    """Evaluate the loss on a 2-D grid while keeping autograd active.

    The CPFlow NLL closure takes a Hessian through the data argument with
    ``create_graph=True``, so the forward must not be wrapped in
    ``torch.no_grad()``. Only the offset parameter tensors are detached,
    which keeps them out of the graph while leaving the closure's internal
    x-gradient chain intact. Off-converged grid points can have
    ill-conditioned Hessians whose log-determinant saturates and produces
    NaN gradients; such a cell is set to +inf and the contour rendering
    caps it in the log-axis tail.
    """
    import torch

    keys = list(params_dict.keys())
    Z = np.empty((len(alphas), len(betas)), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            offset = OrderedDict()
            for k in keys:
                offset[k] = (
                    params_dict[k] + float(a) * d1[k] + float(b) * d2[k]
                ).detach()
            try:
                val = loss_fn(offset)
                v = float(val.detach() if hasattr(val, "detach") else val)
                if not math.isfinite(v):
                    v = float("inf")
                Z[i, j] = v
            except (torch._C._LinAlgError, RuntimeError) as exc:
                msg = str(exc).lower()
                if "ill-conditioned" not in msg and "linalg" not in msg and "singular" not in msg:
                    raise
                Z[i, j] = float("inf")
    return Z


# ---------------------------------------------------------- loaders


def _load_ckpt(ckpt_dir: str, target: str, seed: int, backend: str):
    import torch
    # Multi-target checkpoint directories prefix the basename with the
    # target; single-target ones use the bare ``seed{S}_{backend}.pt``.
    candidates = [
        os.path.join(ckpt_dir, f"{target}_seed{seed}_{backend}.pt"),
        os.path.join(ckpt_dir, f"seed{seed}_{backend}.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return torch.load(path, map_location="cpu", weights_only=False), path
    raise FileNotFoundError(
        f"no ckpt for backend={backend!r} seed={seed} target={target!r} "
        f"in {ckpt_dir}: tried {candidates}"
    )


def _build_template(config: Dict, device):
    from experiments_cpflow_demo_2d import build_icnn_template  # noqa: E402
    template = build_icnn_template(
        D=int(config["D"]),
        hidden_dim=int(config["hidden_dim"]),
        nlayers=int(config["nlayers"]),
        strong_convexity=float(config["strong_convexity"]),
    ).to(device)
    return template


def _build_hypernet(config: Dict, template, device):
    from lift.models import HyperNetwork, icnn_pos_param_names  # noqa: E402
    pos_names = icnn_pos_param_names(int(config["nlayers"]), prefix="")
    hnet = HyperNetwork(
        input_size=int(config["D"]),
        hidden_sizes=list(config["hyper_hidden"]),
        downstream_network=template,
        use_outer_net=False,
        pos_param_names=pos_names,
        pos_constraint_mode="softplus",
    ).to(device)
    return hnet


# ---------------------------------------------------------- landscape compute


def compute_cpflow_landscapes(
    *,
    ckpt_dir: str,
    target: str = "eight_gaussians",
    seed: int = 0,
    device=None,
    grid_n_constrained: int = 21,
    grid_n_lifted: int = 21,
    direction_seed: int = 0,
    n_val_landscape: int = 1024,
) -> Dict:
    """Build panel (a) constrained-theta and panel (b) lifted-(Theta_E, b_h)
    landscape arrays for the CPFlow demo at one (target, seed).
    """
    import torch
    from torch.func import functional_call
    from experiments_cpflow_demo_2d import (  # noqa: E402
        sample_target,
        model_log_prob,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyp_pkg, hyp_path = _load_ckpt(ckpt_dir, target, seed, "hypernet")
    dir_pkg, dir_path = _load_ckpt(ckpt_dir, target, seed, "direct")

    for pkg, path in ((hyp_pkg, hyp_path), (dir_pkg, dir_path)):
        if not pkg.get("snap_iters"):
            raise RuntimeError(
                f"{path} has no per-iter snapshots; re-run training "
                "with --snap_every > 0."
            )

    config = hyp_pkg["config"]

    # ---------- fixed validation batch (same across all grid points) ----------
    # Deterministic resample; Li 2018 uses a fixed batch of at least 1024.
    torch.manual_seed(int(seed) + 7777)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 7777)
    x_val = sample_target(target, int(n_val_landscape), device).detach()

    # ---------- rebuild template (shared between landscape closures) --------
    template = _build_template(config, device)
    # Frozen template; closure mutates only via functional_call params.
    for p in template.parameters():
        p.requires_grad_(False)

    # ---------------------------------------------- panel (a): constrained ----
    # The center is the lift's converged PICNN parameters emitted at the
    # reference batch x_ref stored with the checkpoint. Every ``picnn_snaps``
    # entry lives in that same emit-at-x_ref coordinate system, so the
    # projections are well defined. The center is the last snapshot rather
    # than ``params_final``, which was emitted at x_test, so that the
    # trajectory ends exactly at the origin of the panel.
    picnn_star: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    last_snap = hyp_pkg["picnn_snaps"][-1]
    for k, v in last_snap.items():
        picnn_star[k] = v.detach().to(device).clone()
    keys_c = list(picnn_star.keys())

    # Direct softplus at convergence, from the last ``picnn_snaps`` entry.
    direct_final: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        v = dir_pkg["picnn_snaps"][-1][k].detach().to(device)
        direct_final[k] = v

    # d1 is the end difference, direct minus the lift, filter-normalized
    # per layer after Li 2018.
    diff_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        diff_dict[k] = (direct_final[k] - picnn_star[k]).detach()
    d1_c: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        w = picnn_star[k]
        r = diff_dict[k]
        if w.ndim >= 2 and w.size(0) > 1:
            n = w.size(0)
            rf = r.view(n, -1)
            wf = w.view(n, -1)
            rn = rf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            wn = wf.norm(dim=-1, keepdim=True)
            d1_c[k] = (rf * (wn / rn)).view_as(w).detach()
        else:
            rn = r.norm().clamp(min=1e-12)
            d1_c[k] = (r * (w.norm() / rn)).detach()
    # d2: filter-norm random direction, Gram-Schmidt against d1.
    g = torch.Generator(device=device)
    g.manual_seed(int(direction_seed))
    d2_c_raw = _filter_norm_direction(picnn_star, generator=g)

    def _flat(dd: Mapping[str, "torch.Tensor"]):
        return torch.cat([dd[k].reshape(-1) for k in keys_c])

    v_d1 = _flat(d1_c)
    v_d2 = _flat(d2_c_raw)
    denom = max(float((v_d1 * v_d1).sum()), 1e-20)
    coef = float((v_d2 * v_d1).sum()) / denom
    d2_c: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys_c:
        d2_c[k] = (d2_c_raw[k] - coef * d1_c[k]).detach()

    loss_fn_c = _build_cpflow_loss_closure(template=template, x_val=x_val)

    # ---------- closure-equivalence sanity check (panel a) ------------------
    # At the origin the closure must reproduce ``model_log_prob`` on the
    # same parameters and the same batch; a mismatch means the landscape is
    # not showing the training loss.
    with torch.no_grad():
        pass
    loss_landscape = float(loss_fn_c(picnn_star).detach())
    x_val_canon = x_val.detach().clone()
    logp_canon = model_log_prob(template, picnn_star, x_val_canon)
    loss_canonical = float((-logp_canon).mean().detach())
    diff = abs(loss_landscape - loss_canonical)
    print(
        f"  [closure-equivalence] panel(a): "
        f"landscape={loss_landscape:.6f}  canonical={loss_canonical:.6f}  "
        f"|diff|={diff:.2e}",
        flush=True,
    )
    assert diff < 1e-4, (
        f"CPFlow landscape closure mismatch at converged params: "
        f"{loss_landscape} vs {loss_canonical} (|diff|={diff})."
    )

    # ---------- project trajectories onto (d1_c, d2_c) ----------------------
    hyper_snaps_a: List["OrderedDict[str, torch.Tensor]"] = []
    for snap in hyp_pkg["picnn_snaps"]:
        d = OrderedDict()
        for k in keys_c:
            d[k] = snap[k].detach().to(device).clone()
        hyper_snaps_a.append(d)
    direct_snaps_a: List["OrderedDict[str, torch.Tensor]"] = []
    for snap in dir_pkg["picnn_snaps"]:
        d = OrderedDict()
        for k in keys_c:
            d[k] = snap[k].detach().to(device).clone()
        direct_snaps_a.append(d)

    hyper_traj_a = _project_trajectory(hyper_snaps_a, picnn_star, d1_c, d2_c)
    direct_traj_a = _project_trajectory(direct_snaps_a, picnn_star, d1_c, d2_c)

    # ---------- choose grid radius (panel a) ------------------------------
    max_abs_a = 1.0
    for arr in (hyper_traj_a, direct_traj_a):
        if arr.size > 0:
            max_abs_a = max(max_abs_a, float(np.nanmax(np.abs(arr))))
    radius_a = max_abs_a * 1.35
    alphas_a = np.linspace(-radius_a, radius_a, grid_n_constrained)
    betas_a = np.linspace(-radius_a, radius_a, grid_n_constrained)

    print(f"  [panel-a] grid {grid_n_constrained}^2; radius={radius_a:.3f}",
          flush=True)
    Z_a = _evaluate_loss_grid_with_grad(
        loss_fn_c, picnn_star, d1_c, d2_c, alphas_a, betas_a,
    )
    print(f"  [panel-a] Z range "
          f"[{np.nanmin(Z_a):.3f}, {np.nanmax(Z_a[np.isfinite(Z_a)]):.3f}]; "
          f"#inf={int(np.sum(~np.isfinite(Z_a)))}",
          flush=True)

    # ---------------------------------------------- panel (b): lifted ---------
    # Center on the converged hypernet's state_dict (raw Theta_E, b_h).
    hnet = _build_hypernet(config, template, device)
    hnet.load_state_dict({
        k: v.detach().to(device) for k, v in hyp_pkg["param_module_state"].items()
    })
    for p in hnet.parameters():
        p.requires_grad_(False)
    hnet.eval()

    phi_param_keys = {k for k, _ in hnet.named_parameters()}
    phi_star: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in hnet.state_dict().items():
        if k in phi_param_keys:
            phi_star[k] = v.detach().to(device).clone()

    # ``HyperNetwork.forward(x)`` returns the dict of effective ICNN
    # parameters, so ``functional_call`` routes candidate lifted states
    # through the same emit pathway training used.
    x_ref = hyp_pkg.get("x_ref")
    if x_ref is None:
        # Fallback: use first n=64 samples of x_val.
        x_ref_b = x_val[:64].detach()
    else:
        x_ref_b = x_ref.detach().to(device)

    def loss_fn_b(phi_dict):
        # Emit ICNN params from lifted phi, then evaluate CPFlow NLL.
        emitted = functional_call(hnet, dict(phi_dict), (x_ref_b,))
        x = x_val.detach().clone()
        logp = model_log_prob(template, emitted, x)
        return -logp.mean()

    # ---------- panel (b) sanity check ------------------------------------
    loss_b0 = float(loss_fn_b(phi_star).detach())
    with torch.no_grad():
        canonical_emit = hnet(x_ref_b)
    canonical_emit_detached = {k: v.detach() for k, v in canonical_emit.items()}
    loss_b_canon = float(
        (-model_log_prob(template, canonical_emit_detached, x_val.detach().clone()))
        .mean().detach()
    )
    diff_b = abs(loss_b0 - loss_b_canon)
    print(
        f"  [closure-equivalence] panel(b): "
        f"landscape={loss_b0:.6f}  canonical={loss_b_canon:.6f}  "
        f"|diff|={diff_b:.2e}",
        flush=True,
    )
    assert diff_b < 1e-4, (
        f"CPFlow lifted landscape closure mismatch: "
        f"{loss_b0} vs {loss_b_canon} (|diff|={diff_b})."
    )

    # ---------- directions for panel (b): top-2 PCA of hypernet trajectory ---
    # Spanning the slice by the top-2 trajectory-PCA axes puts the actual
    # training motion inside the panel. At most 12 evenly spaced snapshots
    # enter the SVD, which bounds its cost and keeps the cloned trajectory
    # from doubling the GPU footprint.
    snaps_src = list(hyp_pkg["hyper_snaps"])
    snaps_kept, _ = _subsample_traj(snaps_src, max_keep=12)
    del snaps_src
    hyper_snaps_b: List["OrderedDict[str, torch.Tensor]"] = []
    for snap in snaps_kept:
        d = OrderedDict()
        for k in phi_param_keys:
            if k in snap:
                d[k] = snap[k].detach().to(device).clone()
            else:
                d[k] = phi_star[k].detach().clone()
        hyper_snaps_b.append(d)
    del snaps_kept
    _free_memory("panel-b traj load")

    keys_b = list(phi_star.keys())

    def _flat_b(dd):
        return torch.cat([dd[k].reshape(-1) for k in keys_b])

    v_star_b = _flat_b(phi_star)
    diff_rows = torch.stack(
        [_flat_b(s) - v_star_b for s in hyper_snaps_b], dim=0,
    ).cpu()
    print(f"  [panel-b PCA] diff shape={tuple(diff_rows.shape)} (CPU SVD)",
          flush=True)
    _U, _S, Vh = torch.linalg.svd(diff_rows, full_matrices=False)
    Vh = Vh.to(device)
    del _U, _S, diff_rows, v_star_b
    _free_memory("panel-b SVD done")
    d1_b: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    d2_b: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    o = 0
    for k in keys_b:
        n = phi_star[k].numel()
        d1_b[k] = Vh[0, o:o + n].view_as(phi_star[k])
        d2_b[k] = Vh[1, o:o + n].view_as(phi_star[k])
        o += n

    coords_pre = _project_trajectory(hyper_snaps_b, phi_star, d1_b, d2_b)
    scale = float(max(np.max(np.abs(coords_pre)), 1e-12))
    for k in keys_b:
        d1_b[k] = d1_b[k] * scale
        d2_b[k] = d2_b[k] * scale
    hyper_traj_b = _project_trajectory(hyper_snaps_b, phi_star, d1_b, d2_b)
    max_abs_b = float(max(np.max(np.abs(hyper_traj_b)), 1.0))
    radius_b = max_abs_b * 1.35
    alphas_b = np.linspace(-radius_b, radius_b, grid_n_lifted)
    betas_b = np.linspace(-radius_b, radius_b, grid_n_lifted)

    print(f"  [panel-b] grid {grid_n_lifted}^2; radius={radius_b:.3f}", flush=True)
    Z_b = _evaluate_loss_grid_with_grad(
        loss_fn_b, phi_star, d1_b, d2_b, alphas_b, betas_b,
    )
    print(f"  [panel-b] Z range "
          f"[{np.nanmin(Z_b):.3f}, {np.nanmax(Z_b[np.isfinite(Z_b)]):.3f}]; "
          f"#inf={int(np.sum(~np.isfinite(Z_b)))}", flush=True)

    return {
        "panel_a": {
            "alphas": alphas_a, "betas": betas_a, "Z": Z_a,
            "hyper_traj": hyper_traj_a,
            "direct_traj": direct_traj_a,
            "loss_landscape_at_star": loss_landscape,
            "loss_canonical_at_star": loss_canonical,
        },
        "panel_b": {
            "alphas": alphas_b, "betas": betas_b, "Z": Z_b,
            "hyper_traj": hyper_traj_b,
            "loss_landscape_at_star": loss_b0,
            "loss_canonical_at_star": loss_b_canon,
        },
        "hyp_history": hyp_pkg["history"],
        "dir_history": dir_pkg["history"],
        "config": config,
        "target": target,
        "seed": seed,
    }


# ---------------------------------------------------------------- panel painter


def _paint(
    ax,
    *,
    alphas, betas, Z,
    hyper_coords=None,
    direct_coords=None,
    title: str = "",
    show_legend: bool = False,
    hyper_loss=None,
    direct_loss=None,
    direct_final_plus: bool = True,
    xlim=None,
    ylim=None,
):
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    Z_finite = Z[np.isfinite(Z)]
    z_min = float(np.nanmin(Z_finite)) if Z_finite.size else 0.0
    Z_safe = np.where(np.isfinite(Z), Z, np.nanmax(Z_finite) if Z_finite.size else 1.0)
    Z_safe = Z_safe - z_min + 1e-8
    log_Z = np.log10(Z_safe)
    # The plotted domain is cropped to the structured low-loss region and
    # the colormap is scaled to that window alone. Each panel's basin fills
    # a small part of the full slice; the rest is featureless high-loss
    # wall that would otherwise flatten the colormap and hide the
    # structure, so ``vmin``/``vmax`` come from the windowed loss range.
    xl = tuple(xlim) if xlim is not None else (float(alphas.min()), float(alphas.max()))
    yl = tuple(ylim) if ylim is not None else (float(betas.min()), float(betas.max()))
    in_win = (
        (A >= xl[0]) & (A <= xl[1]) & (B >= yl[0]) & (B <= yl[1])
        & np.isfinite(log_Z)
    )
    win_vals = log_Z[in_win]
    if win_vals.size:
        vmin = float(np.min(win_vals))
        vmax = float(np.percentile(win_vals, 97.0))
    else:
        finite = log_Z[np.isfinite(log_Z)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    levels = np.linspace(vmin, vmax, 25)
    cs = ax.contourf(A, B, log_Z, levels=levels, cmap="viridis",
                     extend="both", vmin=vmin, vmax=vmax)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    cb = ax.figure.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$\log_{10}(\mathrm{loss}-\min\mathrm{loss})$", fontsize=8)
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
    # Gold star at the converged hypernet (always at origin by construction).
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

    ax.set_xlabel(r"$\alpha$ (filter-norm units)", fontsize=9)
    ax.set_ylabel(r"$\beta$ (filter-norm units)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)


# ---------------------------------------------------------------- driver


def main() -> None:
    import argparse
    import matplotlib.pyplot as plt
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", default=_CKPT_DIR_DEFAULT,
        help="Directory of the CPFlow demo ckpts with snap_iters/picnn_snaps.",
    )
    parser.add_argument("--target", default="eight_gaussians",
                        choices=["eight_gaussians", "two_spirals"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_n_constrained", type=int, default=121,
                        help="Grid points per axis for panel (a). "
                             "Pass 21 to "
                             "recreate the legacy figure.")
    parser.add_argument("--grid_n_lifted", type=int, default=121,
                        help="Grid points per axis for panel (b). "
                             "")
    parser.add_argument("--n_val_landscape", type=int, default=1024)
    parser.add_argument("--direction_seed", type=int, default=0)
    parser.add_argument("--out_pdf", default=_OUT_PDF)
    parser.add_argument(
        "--cache_npz", action="store_true",
        help="Save the computed (alphas, betas, Z) grids alongside the "
             "ckpts (seed{S}_landscape_constrained_{grid_n}.npz and "
             "..._lifted_{grid_n}.npz) so re-renders can be picked up "
             "via --load_npz without re-computing.",
    )
    parser.add_argument(
        "--load_npz", action="store_true",
        help="If set, attempt to load (alphas, betas, Z) panels from "
             "the cached NPZs first; only compute on miss.",
    )
    args = parser.parse_args()

    apply_paper_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cpflow-landscape] device={device}", flush=True)

    cache_a = os.path.join(
        args.ckpt_dir,
        f"seed{args.seed}_landscape_constrained_{args.grid_n_constrained}.npz",
    )
    cache_b = os.path.join(
        args.ckpt_dir,
        f"seed{args.seed}_landscape_lifted_{args.grid_n_lifted}.npz",
    )
    if args.load_npz and os.path.isfile(cache_a) and os.path.isfile(cache_b):
        print(f"[cpflow-landscape] loading cached NPZs:", flush=True)
        print(f"  {cache_a}")
        print(f"  {cache_b}")
        import torch as _torch
        cache_pkg_a = dict(np.load(cache_a, allow_pickle=True))
        cache_pkg_b = dict(np.load(cache_b, allow_pickle=True))
        # The histories are small, so they come straight from the ckpts.
        hyp_pkg, _ = _load_ckpt(args.ckpt_dir, args.target, args.seed, "hypernet")
        dir_pkg, _ = _load_ckpt(args.ckpt_dir, args.target, args.seed, "direct")
        data = {
            "panel_a": {
                "alphas": cache_pkg_a["alphas"],
                "betas": cache_pkg_a["betas"],
                "Z": cache_pkg_a["Z"],
                "hyper_traj": cache_pkg_a["hyper_traj"],
                "direct_traj": cache_pkg_a["direct_traj"],
            },
            "panel_b": {
                "alphas": cache_pkg_b["alphas"],
                "betas": cache_pkg_b["betas"],
                "Z": cache_pkg_b["Z"],
                "hyper_traj": cache_pkg_b["hyper_traj"],
            },
            "hyp_history": hyp_pkg["history"],
            "dir_history": dir_pkg["history"],
        }
    else:
        data = compute_cpflow_landscapes(
            ckpt_dir=args.ckpt_dir,
            target=args.target, seed=args.seed,
            device=device,
            grid_n_constrained=int(args.grid_n_constrained),
            grid_n_lifted=int(args.grid_n_lifted),
            direction_seed=int(args.direction_seed),
            n_val_landscape=int(args.n_val_landscape),
        )
        if args.cache_npz:
            np.savez(
                cache_a,
                alphas=data["panel_a"]["alphas"],
                betas=data["panel_a"]["betas"],
                Z=data["panel_a"]["Z"],
                hyper_traj=data["panel_a"]["hyper_traj"],
                direct_traj=data["panel_a"]["direct_traj"],
            )
            np.savez(
                cache_b,
                alphas=data["panel_b"]["alphas"],
                betas=data["panel_b"]["betas"],
                Z=data["panel_b"]["Z"],
                hyper_traj=data["panel_b"]["hyper_traj"],
            )
            print(f"[cpflow-landscape] cached NPZs:", flush=True)
            print(f"  {cache_a}")
            print(f"  {cache_b}")

    hyp_loss = np.asarray(data["hyp_history"].get("train_nll", []),
                          dtype=np.float64)
    dir_loss = np.asarray(data["dir_history"].get("train_nll", []),
                          dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    # Each panel is cropped to the sub-window its structure occupies: a
    # wide, thin horizontal lens in constrained-theta space, and the lower
    # band in lifted space, where only the high-loss top is cut away.
    pa = data["panel_a"]
    _paint(
        axes[0],
        alphas=pa["alphas"], betas=pa["betas"], Z=pa["Z"],
        hyper_coords=pa["hyper_traj"],
        direct_coords=pa["direct_traj"],
        title=r"constrained ICNN $\boldsymbol{\theta}$ space",
        show_legend=True,
        hyper_loss=hyp_loss, direct_loss=dir_loss,
        direct_final_plus=True,
        xlim=(-0.7, 1.4), ylim=(-0.62, 0.62),
    )
    pb = data["panel_b"]
    _paint(
        axes[1],
        alphas=pb["alphas"], betas=pb["betas"], Z=pb["Z"],
        hyper_coords=pb["hyper_traj"],
        direct_coords=None,
        title=r"lifted $(\boldsymbol{\phi},\,b)$ space",
        show_legend=False,
        hyper_loss=hyp_loss,
        direct_final_plus=False,
        xlim=(-1.35, 1.35), ylim=(-1.35, 0.42),
    )

    fig.suptitle(
        f"CPFlow loss-landscape on {args.target.replace('_', '-')} "
        f"(seed {args.seed})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[cpflow-landscape] wrote {args.out_pdf}", flush=True)


if __name__ == "__main__":
    main()
