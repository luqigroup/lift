"""Replot the 1-D shared-theta-space NLL landscape.

Renders the saved ``seed{S}_landscape.npz`` arrays, whose keys are
``alphas, betas, Z, coords_hyper, coords_direct, iters_hyper,
iters_direct``; nothing here recomputes a loss. Each panel draws
log-scaled NLL contours, both trajectories with start crosses, a gold star
at the origin for the lift's final point and a plus for direct softplus's.
Colors come from :data:`lift.utils.paper_style.PALETTE`.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lift.utils.loss_sparkline import add_loss_sparkline
from lift.utils.paper_style import PALETTE, apply_paper_style


def _load_landscape_npz(ckpt_dir: str, seed: int) -> Optional[dict]:
    """Return the .npz arrays for ``seed`` as a plain dict, or None."""
    path = os.path.join(ckpt_dir, f"seed{seed}_landscape.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def _discover_seeds(ckpt_dir: str, max_seed: int = 8) -> List[int]:
    """Return the sorted list of seed indices that have landscape npz."""
    out: List[int] = []
    for s in range(max_seed):
        if os.path.isfile(os.path.join(ckpt_dir, f"seed{s}_landscape.npz")):
            out.append(s)
    return out


def plot_landscape_panel(
    ax,
    npz_data: dict,
    *,
    title: str = "",
    show_full_legend: bool = False,
    hypernet_color: str = PALETTE["hypernet"],
    direct_color: str = PALETTE["direct_softplus"],
    hyper_loss: Optional[np.ndarray] = None,
    direct_loss: Optional[np.ndarray] = None,
    draw_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Draw one (alpha, beta) landscape panel onto ``ax``.

    Returns the ``QuadContourSet``, so a caller building a multi-panel
    figure with one shared colorbar can pass it to ``fig.colorbar``.
    """
    alphas = np.asarray(npz_data["alphas"])
    betas = np.asarray(npz_data["betas"])
    Z = np.asarray(npz_data["Z"])
    cd = np.asarray(npz_data["coords_direct"])
    ch = np.asarray(npz_data["coords_hyper"])

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    Z_safe = Z - Z.min() + 1e-8
    logZ = np.log10(Z_safe)
    if vmin is None:
        vmin = float(logZ.min())
    if vmax is None:
        vmax = float(logZ.max())
    levels = np.linspace(vmin, vmax, 25)
    cs = ax.contourf(A, B, logZ, levels=levels, cmap="viridis",
                     vmin=vmin, vmax=vmax, extend="both")
    ax.contour(
        A, B, logZ,
        levels=np.linspace(vmin, vmax, 15),
        colors="white", linewidths=0.4, alpha=0.5,
    )
    if draw_colorbar:
        cb = ax.figure.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(r"$\log_{10}(\mathrm{NLL}-\min\mathrm{NLL})$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    ax.plot(
        ch[:, 0], ch[:, 1], "-s",
        color=hypernet_color, lw=1.4, ms=2.6,
        mec="black", mew=0.25,
        label="hypernet traj", zorder=4, alpha=0.95,
    )
    ax.plot(
        cd[:, 0], cd[:, 1], "-o",
        color=direct_color, lw=1.4, ms=2.6,
        mec="black", mew=0.25,
        label="direct traj", zorder=4, alpha=0.95,
    )
    ax.plot(
        ch[0, 0], ch[0, 1], "X",
        color=hypernet_color, ms=11, mec="black", mew=0.7, zorder=5,
        label="hyper init" if show_full_legend else None,
    )
    ax.plot(
        cd[0, 0], cd[0, 1], "X",
        color=direct_color, ms=11, mec="black", mew=0.7, zorder=5,
        label="direct init" if show_full_legend else None,
    )
    ax.plot(
        [0], [0], "*",
        color="gold", ms=18, mec="black", mew=0.7, zorder=6,
        label=r"hyper final $\theta^\star$" if show_full_legend else None,
    )
    ax.plot(
        cd[-1, 0], cd[-1, 1], "P",
        color="#ff7b00", ms=13, mec="black", mew=0.7, zorder=6,
        label=r"direct final $\theta$" if show_full_legend else None,
    )

    ax.set_xlabel(
        r"$\alpha$ along $\theta_{\rm direct,fin}-\theta^\star$",
        fontsize=9,
    )
    ax.set_ylabel(r"$\beta$ orth (traj PC)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    if show_full_legend:
        ax.legend(loc="lower left", fontsize=7, framealpha=0.85,
                  borderpad=0.3, handlelength=1.2, ncol=2,
                  columnspacing=0.8)

    # Loss-versus-iteration sparkline inset, in the shared
    # ``lift.utils.loss_sparkline`` style.
    curves: List[Tuple[np.ndarray, str]] = []
    if hyper_loss is not None and len(hyper_loss) > 0:
        curves.append((np.asarray(hyper_loss), hypernet_color))
    if direct_loss is not None and len(direct_loss) > 0:
        curves.append((np.asarray(direct_loss), direct_color))
    if curves:
        add_loss_sparkline(ax, curves)

    return cs


def _load_loss_history(ckpt_dir: str, seed: int) -> dict:
    """Return ``{'hyper': np.array, 'direct': np.array}`` of per-iteration loss.

    Reads ``seed{S}_{hypernet,direct}.pt`` from ``ckpt_dir``. A missing
    checkpoint yields an empty array, which the renderer draws as no inset.
    """
    import torch  # local import: rendering still works without torch
    out: dict = {"hyper": np.zeros(0), "direct": np.zeros(0)}
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


def render_landscape_multiseed(
    ckpt_dir: str,
    out_path: str,
    *,
    seeds: Optional[Sequence[int]] = None,
    row_label: Optional[str] = None,
    loss_history_dir: Optional[str] = None,
) -> None:
    """Render a 1xN landscape figure, one panel per seed, to ``out_path``.

    ``loss_history_dir`` is the directory holding the
    ``seed{S}_{hypernet,direct}.pt`` checkpoints whose ``history`` carries
    the per-iteration training losses. It defaults to ``ckpt_dir``, and is
    passed explicitly when the npz files are mirrored somewhere without
    the .pt files.
    """
    import matplotlib.pyplot as plt

    if seeds is None:
        seeds = _discover_seeds(ckpt_dir)
    npz_per_seed: List[Tuple[int, dict]] = []
    for s in seeds:
        d = _load_landscape_npz(ckpt_dir, s)
        if d is not None:
            npz_per_seed.append((s, d))
    if not npz_per_seed:
        raise FileNotFoundError(
            f"No seed*_landscape.npz files under {ckpt_dir!r}"
        )

    hist_dir = loss_history_dir or ckpt_dir

    n = len(npz_per_seed)
    # One vmin/vmax across panels, so they can share a colorbar.
    vmin = np.inf
    vmax = -np.inf
    for _, d in npz_per_seed:
        Z = np.asarray(d["Z"])
        Z_safe = Z - Z.min() + 1e-8
        logZ = np.log10(Z_safe)
        vmin = min(vmin, float(logZ.min()))
        vmax = max(vmax, float(logZ.max()))

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.5, 3.3), squeeze=False)
    last_cs = None
    for idx, (s, d) in enumerate(npz_per_seed):
        ax = axes[0, idx]
        title = f"seed {s}"
        if row_label is not None and idx == 0:
            title = f"{row_label} --- seed {s}"
        hist = _load_loss_history(hist_dir, s)
        last_cs = plot_landscape_panel(
            ax, d, title=title, show_full_legend=(idx == 0),
            hyper_loss=hist.get("hyper"),
            direct_loss=hist.get("direct"),
            draw_colorbar=False, vmin=vmin, vmax=vmax,
        )
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    cbar_ax = fig.add_axes([0.955, 0.18, 0.012, 0.66])
    cb = fig.colorbar(last_cs, cax=cbar_ax)
    cb.set_label(r"$\log_{10}(\mathrm{NLL}-\min\mathrm{NLL})$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_landscape_2x3(
    fkl_dir: str,
    rkl_dir: str,
    out_path: str,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    fkl_history_dir: Optional[str] = None,
    rkl_history_dir: Optional[str] = None,
) -> None:
    """Render a 2x3 figure: rows are (FKL, RKL), columns are seeds.

    ``fkl_history_dir`` and ``rkl_history_dir`` hold the
    ``seed{S}_{hypernet,direct}.pt`` checkpoints, and default to
    ``fkl_dir`` and ``rkl_dir``, where the npz files usually sit beside
    them.
    """
    import matplotlib.pyplot as plt

    rows = (
        ("FKL-direct", fkl_dir, fkl_history_dir or fkl_dir),
        ("RKL (centred baseline)", rkl_dir, rkl_history_dir or rkl_dir),
    )
    n = len(seeds)
    # One vmin/vmax across all 2*n panels.
    vmin = np.inf
    vmax = -np.inf
    panels: list = []
    for label, ddir, hist_dir in rows:
        for s in seeds:
            d = _load_landscape_npz(ddir, s)
            if d is None:
                raise FileNotFoundError(
                    f"seed{s}_landscape.npz missing under {ddir!r}"
                )
            Z = np.asarray(d["Z"])
            Z_safe = Z - Z.min() + 1e-8
            logZ = np.log10(Z_safe)
            vmin = min(vmin, float(logZ.min()))
            vmax = max(vmax, float(logZ.max()))
            panels.append((label, d, hist_dir, s))

    fig, axes = plt.subplots(2, n, figsize=(3.6 * n + 0.5, 6.4),
                             squeeze=False)
    last_cs = None
    for k, (label, d, hist_dir, s) in enumerate(panels):
        r, c = divmod(k, n)
        ax = axes[r, c]
        title = f"{label} --- seed {s}" if c == 0 else f"seed {s}"
        hist = _load_loss_history(hist_dir, s)
        last_cs = plot_landscape_panel(
            ax, d, title=title,
            show_full_legend=(r == 0 and c == 0),
            hyper_loss=hist.get("hyper"),
            direct_loss=hist.get("direct"),
            draw_colorbar=False, vmin=vmin, vmax=vmax,
        )
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    cbar_ax = fig.add_axes([0.955, 0.10, 0.012, 0.80])
    cb = fig.colorbar(last_cs, cax=cbar_ax)
    cb.set_label(r"$\log_{10}(\mathrm{NLL}-\min\mathrm{NLL})$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
