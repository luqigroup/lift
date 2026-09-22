"""Shared loss-sparkline inset for landscape figures.

Every loss-landscape panel carries the same small inset axis in the
lower-right corner, showing the per-iteration training loss on a log y scale.
Each curve is shifted by ``-min(loss) + 1e-6`` before the log, so it stays
positive even when the underlying loss is a log-likelihood that has gone
negative, and the iteration axis is downsampled to about ``max_points``
evenly spaced indices. Any iterable of ``(losses, color)`` pairs is overlaid.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# Locked bbox so every panel in the paper agrees on the inset position.
INSET_BBOX: Tuple[float, float, float, float] = (0.72, 0.05, 0.25, 0.20)


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 200):
    """Return ``(x, y)`` downsampled to at most ``max_points`` indices."""
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def add_loss_sparkline(
    ax,
    curves: Sequence[Tuple[np.ndarray, str]],
    *,
    bbox: Tuple[float, float, float, float] = INSET_BBOX,
    max_points: int = 200,
    log_y: bool = True,
    bg_alpha: float = 0.85,
    border_lw: float = 0.4,
    font_size: int = 6,
    curve_lw: float = 0.9,
) -> "matplotlib.axes.Axes":  # type: ignore[name-defined]
    """Add a small loss-versus-iteration inset axis to ``ax``.

    Parameters
    ----------
    ax
        Parent axes, a landscape panel.
    curves
        Iterable of ``(loss_array, color)`` pairs, one per trajectory the
        panel draws, each ``loss_array`` a 1-D per-iteration training loss
        and each color matching that trajectory.
    bbox
        ``(x0, y0, w, h)`` in axes fraction.
    max_points
        Downsample target; longer curves are subsampled with
        ``np.linspace``.
    log_y
        Log-scaled y, with each curve shifted so its minimum is ``1e-6``
        and the log is defined even for a negative loss.
    bg_alpha
        Inset face alpha; 0.85 keeps the contour visible underneath.
    border_lw
        Spine line width.
    font_size
        Tick font size in points.
    curve_lw
        Loss-curve line width in points.

    Returns
    -------
    matplotlib.axes.Axes
        The inset axes, so the caller can adjust it further.
    """
    iax = ax.inset_axes(list(bbox))

    cleaned: list = []
    for losses, color in curves:
        y_arr = np.asarray(losses, dtype=np.float64)
        if y_arr.size == 0:
            continue
        cleaned.append((y_arr, color))
    if not cleaned:
        return iax

    # The log shift is per curve, so each backend stays readable when the
    # curves live on very different scales.
    plotted_y: list = []
    final_x = 0
    for y_arr, color in cleaned:
        x_arr = np.arange(len(y_arr), dtype=np.float64)
        x_ds, y_ds = _downsample(x_arr, y_arr, max_points=max_points)
        if log_y:
            local_min = float(np.nanmin(y_ds))
            y_plot = y_ds - local_min + 1e-6
            iax.semilogy(x_ds, y_plot, color=color, lw=curve_lw,
                         solid_joinstyle="round", solid_capstyle="round")
        else:
            iax.plot(x_ds, y_ds, color=color, lw=curve_lw,
                     solid_joinstyle="round", solid_capstyle="round")
            y_plot = y_ds
        plotted_y.append(y_plot)
        final_x = max(final_x, int(x_arr[-1]))

    iax.set_facecolor("white")
    iax.patch.set_alpha(bg_alpha)
    for spine in iax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(border_lw)

    # Minimal ticks: one x-tick at the final iteration, two y-ticks
    # bracketing the visible log range.
    iax.set_xticks([final_x])
    iax.set_xticklabels([f"{final_x:d}"])
    if plotted_y:
        all_y = np.concatenate([np.asarray(y) for y in plotted_y])
        all_y = all_y[np.isfinite(all_y)]
        if all_y.size > 0:
            ymin = float(np.nanmin(all_y))
            ymax = float(np.nanmax(all_y))
            if log_y and ymin > 0 and ymax > ymin:
                iax.set_yticks([ymin, ymax])
                iax.set_yticklabels(
                    [_fmt_log_tick(ymin), _fmt_log_tick(ymax)]
                )
            elif ymax > ymin:
                iax.set_yticks([ymin, ymax])
                iax.set_yticklabels([f"{ymin:.2g}", f"{ymax:.2g}"])

    iax.tick_params(
        axis="both", which="both",
        labelsize=font_size, length=1.5, pad=1, color="black",
        labelcolor="black",
    )
    # The paper style turns the grid on globally; the sparkline frame is
    # kept uncluttered.
    iax.grid(False)
    return iax


def _fmt_log_tick(v: float) -> str:
    """Compact log-tick label (no exponent if 0.01 <= v <= 99)."""
    if v <= 0:
        return f"{v:.2g}"
    av = abs(v)
    if 0.01 <= av < 100:
        return f"{v:.2g}"
    return f"{v:.1e}"


__all__ = ["add_loss_sparkline", "INSET_BBOX"]
