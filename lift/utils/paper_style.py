"""Single source of truth for paper-quality matplotlib styling.

``apply_paper_style()`` installs serif fonts, Type-42 font embedding in
PDFs, and savefig defaults. ``PALETTE`` fixes the hex color of each
role so a construction keeps one color across every figure.
"""

from __future__ import annotations

from typing import Dict

import matplotlib

PALETTE: Dict[str, str] = {
    "target": "#000000",
    "hypernet": "#ff7f0e",
    "direct_softplus": "#d62728",
    "gmm": "#7f7f7f",
    "rkl": "#2ca02c",
    "id": "#1f77b4",
    "ood": "#d62728",
    "certificate": "#9467bd",
    "loss_E": "#ff7f0e",
    "loss_S": "#000000",
    "val_nll": "#d62728",
}


_RC_PAPER = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
}


def apply_paper_style() -> None:
    """Install the paper-quality matplotlib rcParams in-place."""
    for k, v in _RC_PAPER.items():
        try:
            matplotlib.rcParams[k] = v
        except (KeyError, ValueError):
            pass


def palette(role: str) -> str:
    """Return the canonical hex color for ``role``.

    An unknown role falls back to ``"#333333"`` rather than raising, so
    a typo shows up as a gray mark instead of killing a plot pipeline.
    """
    return PALETTE.get(role, "#333333")
