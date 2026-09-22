"""Li-2018 filter-norm loss-landscape rendering primitives.

Renders the shared-theta-space NLL landscape arrays with numpy and
matplotlib. The generic ``compute_landscape`` / ``plot_landscape_generic``
/ ``render_landscape_generic`` triple handles any ``(loss_fn, params,
trajectory)`` triple (ICNN-EBM, CP-Flow, PCP-Map).

A saved ``seed{S}_landscape.npz`` carries everything needed to redraw the
figure: ``alphas, betas, Z, coords_hyper, coords_direct, iters_hyper,
iters_direct``.
"""

from .render import render_landscape_multiseed, plot_landscape_panel
from .render_generic import (
    compute_landscape,
    plot_landscape_generic,
    render_landscape_generic,
)

__all__ = [
    "render_landscape_multiseed",
    "plot_landscape_panel",
    "compute_landscape",
    "plot_landscape_generic",
    "render_landscape_generic",
]
