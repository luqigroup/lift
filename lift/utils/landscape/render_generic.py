"""Loss-landscape renderer for any ``(loss_fn, params, trajectory)`` triple.

The method is Li et al. 2018, "Visualizing the Loss Landscape of Deep
Nets" (NeurIPS): pick two directions in parameter space, evaluate the loss
on a grid of ``params* + alpha * d1 + beta * d2``, and project the
trajectory onto that plane by the 2x2 normal equations
``D^T D @ ab = D^T (theta_t - theta*)`` with ``D = [d1 | d2]`` flattened.
Three direction schemes are available: ``filter_norm_random_pair`` samples
a Gaussian pair and rescales each layer to that layer's own filter norms,
never across layers; ``pca_trajectory`` takes the top-2 right singular
vectors of ``trajectory - converged``; ``delta_fin_pca_orth`` puts an
anchor solution at alpha = 1 and the orthogonalized top trajectory-PCA
loading on beta.
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from lift.utils.paper_style import PALETTE, apply_paper_style


# --------------------------------------------------------------------- types

ParamsDict = "OrderedDict[str, object]"  # OrderedDict[str, torch.Tensor]
LossFn = Callable[[ParamsDict], float]


# ---------------------------------------------------------- direction helpers


def _filter_norm_direction(
    params_dict: Mapping[str, "torch.Tensor"],
    *,
    generator: "torch.Generator | None" = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Sample a Li-2018 filter-normalized random direction.

    A tensor ``W`` with ``W.ndim >= 2`` and more than one output filter is
    normalized per row, so each row of the direction carries the norm of
    the matching row of ``W``; every other tensor, such as a bias or a
    scalar, is normalized as one block. A zero-norm ``W`` yields a
    zero-norm direction component, which gives that axis no influence.
    """
    import torch  # local import: keep the module importable without torch

    direction: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, w in params_dict.items():
        # Passing a ``size=`` tuple covers scalars, vectors and tensors
        # alike; ``torch.randn(*shape)`` fails on 0-d shapes.
        r = torch.randn(
            tuple(w.shape),
            generator=generator,
            device=w.device, dtype=w.dtype,
        )
        if w.ndim >= 2 and w.size(0) > 1:
            n = w.size(0)
            rf = r.view(n, -1)
            wf = w.detach().view(n, -1)
            rn = rf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            wn = wf.norm(dim=-1, keepdim=True)
            rf = rf * (wn / rn)
            direction[name] = rf.view_as(w).detach()
        else:
            rn = r.norm().clamp(min=1e-12)
            direction[name] = (r * (w.detach().norm() / rn)).detach()
    return direction


def _flatten(d: Mapping[str, "torch.Tensor"], keys: Sequence[str]):
    import torch

    return torch.cat([d[k].reshape(-1) for k in keys])


def _pca_trajectory_directions(
    params_dict: Mapping[str, "torch.Tensor"],
    trajectory_params_list: Sequence[Mapping[str, "torch.Tensor"]],
) -> Tuple["OrderedDict[str, torch.Tensor]", "OrderedDict[str, torch.Tensor]"]:
    """Top-2 PCA directions of ``(traj - params*)`` in flat param space."""
    import torch

    keys = list(params_dict.keys())
    v_star = _flatten(params_dict, keys)
    rows = []
    for snap in trajectory_params_list:
        v = _flatten(snap, keys)
        rows.append((v - v_star).reshape(-1))
    if not rows:
        raise ValueError(
            "pca_trajectory requires a non-empty trajectory_params_list."
        )
    X = torch.stack(rows, dim=0)  # (T, P)
    # The top right-singular vectors are the loading directions.
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    if Vh.shape[0] < 2:
        # Degenerate trajectory: fall back to a filter-norm second axis.
        v_top = Vh[0]
        d1_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        o = 0
        for k in keys:
            n = params_dict[k].numel()
            d1_dict[k] = v_top[o:o + n].view_as(params_dict[k])
            o += n
        d2_dict = _filter_norm_direction(params_dict)
        return d1_dict, d2_dict

    out: Tuple[
        "OrderedDict[str, torch.Tensor]", "OrderedDict[str, torch.Tensor]"
    ] = (OrderedDict(), OrderedDict())
    for idx, axis in enumerate((Vh[0], Vh[1])):
        o = 0
        for k in keys:
            n = params_dict[k].numel()
            out[idx][k] = axis[o:o + n].view_as(params_dict[k])
            o += n
    return out


def _project_trajectory(
    trajectory_params_list: Sequence[Mapping[str, "torch.Tensor"]],
    params_dict: Mapping[str, "torch.Tensor"],
    d1: Mapping[str, "torch.Tensor"],
    d2: Mapping[str, "torch.Tensor"],
) -> np.ndarray:
    """Solve the 2x2 normal equations to project ``traj`` onto (d1, d2)."""
    import torch

    keys = list(params_dict.keys())
    v_star = _flatten(params_dict, keys)
    v_d1 = _flatten(d1, keys)
    v_d2 = _flatten(d2, keys)
    D = torch.stack([v_d1, v_d2], dim=1)
    A = D.T @ D
    coords: List[np.ndarray] = []
    for snap in trajectory_params_list:
        v = _flatten(snap, keys)
        rhs = D.T @ (v - v_star)
        ab = torch.linalg.solve(A, rhs)
        coords.append(ab.detach().cpu().numpy())
    if not coords:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(coords, axis=0)


# ----------------------------------------------------- grid evaluation core


def _evaluate_loss_grid(
    loss_fn: LossFn,
    params_dict: Mapping[str, "torch.Tensor"],
    d1: Mapping[str, "torch.Tensor"],
    d2: Mapping[str, "torch.Tensor"],
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Evaluate ``loss_fn`` at every grid point sequentially.

    Returns ``Z`` of shape ``(len(alphas), len(betas))`` with
    ``Z[i, j] = loss_fn(params* + alphas[i]*d1 + betas[j]*d2)``.

    The periodic ``gc.collect`` and CUDA cache flush keep the steady-state
    footprint flat over the grid_n^2 iterations: reference cycles inside a
    per-call ``functional_call`` hold stale tensors alive until a full GC
    cycle, which costs a lifted-space caller several GB without them.
    """
    import gc as _gc
    import os as _os
    import torch

    keys = list(params_dict.keys())
    Z = np.empty((len(alphas), len(betas)), dtype=np.float64)
    flush_every = max(1, int(_os.environ.get("LANDSCAPE_FLUSH_EVERY", 32)))
    step = 0
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                offset: "OrderedDict[str, torch.Tensor]" = OrderedDict()
                for k in keys:
                    offset[k] = (
                        params_dict[k]
                        + float(a) * d1[k]
                        + float(b) * d2[k]
                    )
                Z[i, j] = float(loss_fn(offset))
                del offset
                step += 1
                if step % flush_every == 0:
                    _gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return Z


# --------------------------------------------------------- vmap grid eval


def _evaluate_loss_grid_vmap(
    loss_fn: LossFn,
    params_dict: Mapping[str, "torch.Tensor"],
    d1: Mapping[str, "torch.Tensor"],
    d2: Mapping[str, "torch.Tensor"],
    alphas: np.ndarray,
    betas: np.ndarray,
    *,
    grid_chunk: int = 64,
) -> np.ndarray:
    """Batched grid eval via :func:`torch.vmap` over flat ``(alpha, beta)``.

    The caller's ``loss_fn`` must be written functionally, through
    ``torch.func.functional_call``, and free of stateful side effects;
    otherwise vmap raises, which :func:`compute_landscape` catches in
    ``'auto'`` mode and demotes to the serial path. Each parameter tensor
    gains a leading batch dimension of ``min(grid_chunk, G - s)``, and the
    flat grid is walked in chunks so peak GPU memory stays bounded.
    """
    import torch

    keys = list(params_dict.keys())
    device = next(iter(params_dict.values())).device
    dtype = next(iter(params_dict.values())).dtype

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    a_flat = torch.as_tensor(A.reshape(-1), dtype=dtype, device=device)
    b_flat = torch.as_tensor(B.reshape(-1), dtype=dtype, device=device)
    G = int(a_flat.numel())

    # ``in_dims`` is pytree-matched against the argument structure, and
    # dict and OrderedDict are not interchangeable there, so the container
    # type must match the one the caller passes.
    in_dims_dict = OrderedDict((k, 0) for k in keys)
    vmapped = torch.vmap(loss_fn, in_dims=(in_dims_dict,))

    out = torch.empty(G, dtype=dtype, device=device)
    chunk = max(1, int(grid_chunk))
    with torch.no_grad():
        for s in range(0, G, chunk):
            e = min(s + chunk, G)
            a_c = a_flat[s:e]
            b_c = b_flat[s:e]
            stacked: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k in keys:
                v_star = params_dict[k]
                a_shape = (-1, *([1] * v_star.ndim))
                stacked[k] = (
                    v_star.unsqueeze(0)
                    + a_c.view(*a_shape) * d1[k].unsqueeze(0)
                    + b_c.view(*a_shape) * d2[k].unsqueeze(0)
                )
            out[s:e] = vmapped(stacked).reshape(-1)

    Z = out.detach().cpu().numpy().astype(np.float64).reshape(
        len(alphas), len(betas),
    )
    return Z


def _vmap_probe(
    loss_fn: LossFn,
    params_dict: Mapping[str, "torch.Tensor"],
    d1: Mapping[str, "torch.Tensor"],
    d2: Mapping[str, "torch.Tensor"],
) -> Tuple[bool, Optional[Exception]]:
    """Run a one-point ``torch.vmap`` probe; return ``(ok, exc_or_None)``.

    ``vmap='auto'`` uses this to decide whether to take the batched path. A
    single-element vmap call exercises the same batched-tensor regime as a
    full chunked run.
    """
    import torch

    keys = list(params_dict.keys())
    device = next(iter(params_dict.values())).device
    dtype = next(iter(params_dict.values())).dtype
    try:
        in_dims_dict = OrderedDict((k, 0) for k in keys)
        vmapped = torch.vmap(loss_fn, in_dims=(in_dims_dict,))
        a = torch.zeros(1, dtype=dtype, device=device)
        b = torch.zeros(1, dtype=dtype, device=device)
        stacked: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for k in keys:
            v_star = params_dict[k]
            a_shape = (-1, *([1] * v_star.ndim))
            stacked[k] = (
                v_star.unsqueeze(0)
                + a.view(*a_shape) * d1[k].unsqueeze(0)
                + b.view(*a_shape) * d2[k].unsqueeze(0)
            )
        with torch.no_grad():
            _ = vmapped(stacked)
        return True, None
    except Exception as exc:  # noqa: BLE001 -- any failure demotes to serial
        return False, exc


# ---------------------------------------------------------- top-level API


def compute_landscape(
    loss_fn: LossFn,
    params_dict: "OrderedDict[str, torch.Tensor]",
    trajectory_params_list: Sequence[Mapping[str, "torch.Tensor"]],
    *,
    grid_n: int = 41,
    grid_radius: float = 1.0,
    direction_method: str = "filter_norm_random_pair",
    seed: int = 0,
    vmap: "str | bool" = "auto",
    grid_chunk: int = 64,
    anchor_params=None,
    alpha_range=None,
    beta_range=None,
    directions=None,
) -> dict:
    """Compute the Li-2018 filter-norm landscape arrays.

    ``alpha_range`` / ``beta_range`` take an explicit ``(lo, hi)`` window
    instead of the symmetric ``[-grid_radius, grid_radius]``, and
    ``directions`` takes a ``(d1, d2)`` pair computed by an earlier call.
    Together they produce a zoomed inset that is guaranteed to lie in the
    *same* plane as its parent panel: pass the parent's returned ``d1`` /
    ``d2`` and a narrow window, and raise ``grid_n`` to resolve structure
    the parent's mesh is too coarse to show.

    Parameters
    ----------
    vmap
        Controls grid-evaluation batching:

        - ``'auto'`` (default): probe ``torch.vmap`` once on a 1-point
          stacked input; if the probe succeeds, evaluate the whole
          grid in chunks of ``grid_chunk`` via vmap. If the probe
          raises (loss is stateful, uses Python control flow on
          tensor values, calls ``.item()``, etc.), fall back to the
          serial path with a one-line :func:`warnings.warn`.
        - ``True``: force the vmap path; let exceptions propagate.
          The caller asserts the loss is vmap-compatible (typically
          structured via :func:`torch.func.functional_call`).
        - ``False``: force the serial path.
    grid_chunk
        Number of grid points evaluated per ``torch.vmap`` call when
        the batched path is taken; default 64. Bounded above by GPU
        memory, below by the cost of the chunk-loop overhead.

    Returns a dict with keys ``alphas, betas, Z, traj_coords, d1, d2,
    method``.  ``Z`` has shape ``(grid_n, grid_n)``; ``traj_coords``
    has shape ``(T, 2)``.
    """
    import torch

    device = next(iter(params_dict.values())).device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    if directions is not None:
        d1, d2 = directions
    elif direction_method == "filter_norm_random_pair":
        d1 = _filter_norm_direction(params_dict, generator=g)
        d2 = _filter_norm_direction(params_dict, generator=g)
    elif direction_method == "pca_trajectory":
        d1, d2 = _pca_trajectory_directions(
            params_dict, trajectory_params_list,
        )
    elif direction_method == "delta_fin_pca_orth":
        # The house frame, and the one to reach for: a strict random pair
        # collapses these surfaces to a featureless bowl.
        #
        #   d1 = anchor - center        (the anchor lands at alpha = 1)
        #   d2 = top PCA loading of (trajectory - center) after
        #        Gram-Schmidt removal of d1, rescaled so the trajectory
        #        spans about one unit in beta.
        #
        # ``anchor_params`` is the solution that sits at (1, 0) and
        # ``params_dict`` is the origin; for the tabular figures those are
        # converged direct softplus and the converged lift.
        if anchor_params is None:
            raise ValueError(
                "direction_method='delta_fin_pca_orth' needs "
                "anchor_params (the solution placed at alpha = 1)."
            )
        keys = list(params_dict.keys())
        v_c = _flatten(params_dict, keys)
        d1_flat = _flatten(anchor_params, keys) - v_c
        diffs = torch.stack([
            _flatten(t, keys) - v_c for t in trajectory_params_list
        ])
        u1 = d1_flat / d1_flat.norm().clamp_min(1e-20)
        perp = diffs - (diffs @ u1).unsqueeze(1) * u1.unsqueeze(0)
        _, _, Vh = torch.linalg.svd(perp, full_matrices=False)
        d2_flat = Vh[0]
        span = (perp @ d2_flat).abs().max().clamp_min(1e-12)
        d2_flat = d2_flat * span
        d1, d2, off = OrderedDict(), OrderedDict(), 0
        for k in keys:
            n = params_dict[k].numel()
            d1[k] = d1_flat[off:off + n].view_as(params_dict[k]).clone()
            d2[k] = d2_flat[off:off + n].view_as(params_dict[k]).clone()
            off += n
    else:
        raise ValueError(
            f"unknown direction_method={direction_method!r}; expected "
            "'filter_norm_random_pair', 'pca_trajectory' or "
            "'delta_fin_pca_orth'"
        )

    a_lo, a_hi = alpha_range if alpha_range is not None else (
        -grid_radius, grid_radius)
    b_lo, b_hi = beta_range if beta_range is not None else (
        -grid_radius, grid_radius)
    alphas = np.linspace(a_lo, a_hi, grid_n)
    betas = np.linspace(b_lo, b_hi, grid_n)

    # ------------------------------------------------ vmap dispatch
    if vmap is True:
        use_vmap = True
    elif vmap is False:
        use_vmap = False
    elif vmap == "auto":
        ok, exc = _vmap_probe(loss_fn, params_dict, d1, d2)
        if ok:
            use_vmap = True
        else:
            warnings.warn(
                "compute_landscape: torch.vmap probe failed "
                f"({type(exc).__name__}: {exc}); "
                "falling back to serial grid evaluation.",
                stacklevel=2,
            )
            use_vmap = False
    else:
        raise ValueError(
            f"unknown vmap={vmap!r}; expected 'auto', True, or False."
        )

    if use_vmap:
        Z = _evaluate_loss_grid_vmap(
            loss_fn, params_dict, d1, d2, alphas, betas,
            grid_chunk=grid_chunk,
        )
    else:
        Z = _evaluate_loss_grid(loss_fn, params_dict, d1, d2, alphas, betas)

    if trajectory_params_list:
        traj_coords = _project_trajectory(
            trajectory_params_list, params_dict, d1, d2,
        )
    else:
        traj_coords = np.zeros((0, 2), dtype=np.float64)

    return {
        "alphas": alphas,
        "betas": betas,
        "Z": Z,
        "traj_coords": traj_coords,
        "d1": d1,
        "d2": d2,
        "method": direction_method,
    }


def plot_landscape_generic(
    ax,
    *,
    alphas: np.ndarray,
    betas: np.ndarray,
    Z: np.ndarray,
    traj_coords: Optional[np.ndarray] = None,
    title: str = "",
    show_legend: bool = False,
    traj_color: str = PALETTE["hypernet"],
    loss_label: str = r"$\log_{10}(\mathrm{loss}-\min\mathrm{loss})$",
) -> None:
    """Draw one landscape panel onto ``ax``.

    One trajectory, with a gold star at the origin, which is the converged
    parameter point.
    """
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    Z_safe = Z - Z.min() + 1e-8
    cs = ax.contourf(A, B, np.log10(Z_safe), levels=25, cmap="viridis")
    ax.contour(
        A, B, np.log10(Z_safe),
        levels=15, colors="white", linewidths=0.4, alpha=0.5,
    )
    cb = ax.figure.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(loss_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if traj_coords is not None and traj_coords.shape[0] >= 1:
        ax.plot(
            traj_coords[:, 0], traj_coords[:, 1], "-o",
            color=traj_color, lw=1.4, ms=2.6,
            mec="black", mew=0.25,
            label="trajectory", zorder=4, alpha=0.95,
        )
        ax.plot(
            traj_coords[0, 0], traj_coords[0, 1], "X",
            color=traj_color, ms=11, mec="black", mew=0.7, zorder=5,
            label="init" if show_legend else None,
        )
    ax.plot(
        [0], [0], "*",
        color="gold", ms=18, mec="black", mew=0.7, zorder=6,
        label=r"converged $\theta^\star$" if show_legend else None,
    )

    ax.set_xlabel(r"$\alpha$ (filter-norm units)", fontsize=9)
    ax.set_ylabel(r"$\beta$ (filter-norm units)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)


def render_landscape_generic(
    loss_fn: LossFn,
    params_dict: "OrderedDict[str, torch.Tensor]",
    trajectory_params_list: Sequence[Mapping[str, "torch.Tensor"]],
    *,
    grid_n: int = 41,
    grid_radius: float = 1.0,
    direction_method: str = "filter_norm_random_pair",
    seed: int = 0,
    out_path: str = "landscape.pdf",
    title: str = "",
    show_legend: bool = True,
    save_npz_path: Optional[str] = None,
    vmap: "str | bool" = "auto",
    grid_chunk: int = 64,
    anchor_params=None,
) -> dict:
    """Compute the filter-norm landscape and save a PDF.

    Parameters
    ----------
    loss_fn
        Callable mapping an ``OrderedDict[str, torch.Tensor]`` of
        parameters to a scalar loss: the model's full loss closure, a
        forward pass with no training step.
    params_dict
        Ordered dict of the trainable parameters at the converged seed,
        with no buffers and no optimizer state.
    trajectory_params_list
        Parameter snapshots in chronological order, overlaid as the
        trajectory. May be empty.
    grid_n
        Number of grid points per axis.
    grid_radius
        Scale of the grid in filter-norm units, where a unit step matches
        the converged parameters' per-layer Frobenius norms.
    direction_method
        ``"filter_norm_random_pair"``, ``"pca_trajectory"`` or
        ``"delta_fin_pca_orth"``.
    seed
        RNG seed for the random direction sampler.
    out_path
        Where to write the PDF. Parent directories are created on demand.
    title
        Title for the rendered panel.
    show_legend
        Whether to overlay the legend on the panel.
    save_npz_path
        Optional path for the raw ``alphas, betas, Z, traj_coords`` arrays,
        so a replot does not re-evaluate ``loss_fn``.

    Returns
    -------
    dict
        The output of :func:`compute_landscape`.
    """
    import matplotlib.pyplot as plt

    apply_paper_style()
    result = compute_landscape(
        loss_fn,
        params_dict,
        trajectory_params_list,
        grid_n=grid_n,
        grid_radius=grid_radius,
        direction_method=direction_method,
        seed=seed,
        vmap=vmap,
        grid_chunk=grid_chunk,
        anchor_params=anchor_params,
    )

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.6))
    plot_landscape_generic(
        ax,
        alphas=result["alphas"],
        betas=result["betas"],
        Z=result["Z"],
        traj_coords=result["traj_coords"],
        title=title,
        show_legend=show_legend,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if save_npz_path is not None:
        os.makedirs(os.path.dirname(save_npz_path) or ".", exist_ok=True)
        np.savez(
            save_npz_path,
            alphas=result["alphas"],
            betas=result["betas"],
            Z=result["Z"],
            traj_coords=result["traj_coords"],
            method=np.array(result["method"]),
        )

    return result


__all__ = [
    "compute_landscape",
    "plot_landscape_generic",
    "render_landscape_generic",
]
