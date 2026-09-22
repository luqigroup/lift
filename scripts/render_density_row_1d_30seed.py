"""The four 1-D fits over all 30 paired seeds, drawn as a row of panels.

Each construction is the pointwise median over the seeds; ``--spread`` selects
the signed-discrepancy fill, a 10-90 percentile band, or the individual curves.
Densities come from the ``log_p_grid`` stored in every checkpoint, on the shared
``x_grid.npy`` of the target directory, so nothing is retrained; the grids are
cached under ``data/analysis/`` on first run.

Output: ``<out_dir>/logconcave_density_row_30seed.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

TARGET_LABEL = {"gumbel": "Gumbel", "laplace": "Laplace",
                "gamma": r"Gamma$(\alpha{=}2)$", "beta": r"Beta$(2,5)$"}
ORDER = ["gumbel", "laplace", "gamma", "beta"]
ARMS = [("lift", "lcmm_1d_logconcave_30seed_poolnorm", "hypernet"),
        ("direct", "lcmm_1d_logconcave_30seed", "direct"),
        ("pgd", "lcmm_1d_logconcave_30seed_pgd", "pgd")]


def _build_cache(ckpt_root: str, n_seeds: int, cache: str, pinned: bool = True) -> dict:
    import torch
    if pinned:
        import _pinned_lib
    out = {}
    for tgt in ORDER:
        tdir = os.path.join(ckpt_root, "lcmm_1d_logconcave_30seed", tgt)
        out[f"{tgt}/x"] = np.load(os.path.join(tdir, "x_grid.npy")).ravel()
        out[f"{tgt}/target"] = np.load(
            os.path.join(tdir, "target_lp_grid.npy")).ravel()
        for arm, archive, kind in ARMS:
            rows = []
            for s in range(n_seeds):
                if pinned and arm == "lift":
                    rows.append(_pinned_lib.one_d_log_p_grid(tgt, s)); continue
                p = os.path.join(ckpt_root, archive, tgt, f"seed{s}_{kind}.pt")
                ck = torch.load(p, map_location="cpu", weights_only=False)
                rows.append(np.asarray(ck["log_p_grid"]).ravel())
                del ck
            out[f"{tgt}/{arm}"] = np.stack(rows)
            print(f"cached {tgt:8s} {arm:7s} {out[f'{tgt}/{arm}'].shape}", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_root", default=os.path.join(_REPO, "data", "checkpoints"))
    p.add_argument("--cache", default=None, help="default: data/analysis/density_grids_1d[_pinned].npz by --pinned")
    p.add_argument("--pinned", type=int, default=1, help="1: the lift's grids from the full-training-data re-evaluation (data/analysis/pinned); 0: the archives' batch-conditioned records")
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--spread", choices=("fill", "band", "curves"),
                   default="fill",
                   help="fill: opaque signed-discrepancy lobes (default); "
                        "band: 10-90 seed band; curves: all 30 curves")
    p.add_argument("--tint", type=float, default=0.45)
    p.add_argument("--out_dir", default=os.path.join(_REPO, "figures", "figures"))
    p.add_argument("--mirror_to",
                   default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--panel_h", type=float, default=1.15)
    p.add_argument("--band", type=float, default=10.0, help="percentile for the band")
    a = p.parse_args()

    if a.cache is None:
        a.cache = os.path.join(_REPO, "data", "analysis", "density_grids_1d_pinned.npz" if a.pinned else "density_grids_1d.npz")
    d = (dict(np.load(a.cache)) if os.path.isfile(a.cache)
         else _build_cache(a.ckpt_root, a.n_seeds, a.cache, pinned=bool(a.pinned)))

    apply_paper_style()
    matplotlib.rcParams["savefig.bbox"] = "standard"
    matplotlib.rcParams["axes.grid"] = False
    col = {"lift": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"],
           "pgd": "#2ca02c"}
    dash = {"lift": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
    # The lift is the widest line and sits under the two baselines, so a curve
    # that coincides with it still leaves a visible fringe rather than erasing it.
    zline = {"lift": 3.0, "direct": 3.6, "pgd": 4.0}
    lw_line = {"lift": 1.3, "direct": 1.1, "pgd": 1.1}
    # The error lobes are opaque tints, painted worst first, so the three fills
    # cannot blend together.
    tint = {k: tuple(1.0 - a.tint * (1.0 - c) for c in mcolors.to_rgb(v))
            for k, v in col.items()}
    zfill = {"direct": 2.0, "pgd": 2.1, "lift": 2.2}

    W = a.fig_width
    left_in, right_in, gap_in = 0.44, 0.10, 0.24
    n = len(ORDER)
    pw = (W - left_in - right_in - (n - 1) * gap_in) / n
    top_in, bot_in, leg_in = 0.17, 0.30, 0.21
    H = top_in + a.panel_h + bot_in + leg_in
    fig = plt.figure(figsize=(W, H))

    for i, tgt in enumerate(ORDER):
        ax = fig.add_axes([(left_in + i * (pw + gap_in)) / W,
                           (bot_in + leg_in) / H, pw / W, a.panel_h / H])
        x = d[f"{tgt}/x"]
        tgt_dens = np.exp(d[f"{tgt}/target"])
        ax.fill_between(x, 0.0, tgt_dens, color="0.93", linewidth=0, zorder=1)
        for arm, _, _ in ARMS:
            dens = np.exp(d[f"{tgt}/{arm}"])
            med = np.median(dens, axis=0)
            if a.spread == "curves":
                for r in dens:
                    ax.plot(x, r, color=col[arm], lw=0.35, alpha=0.16, zorder=2)
            elif a.spread == "band":
                lo = np.percentile(dens, a.band, axis=0)
                hi = np.percentile(dens, 100 - a.band, axis=0)
                ax.fill_between(x, lo, hi, color=col[arm], alpha=0.20,
                                linewidth=0, zorder=2)
            else:
                # Signed discrepancy to the target: a patch inside the grey
                # is target mass missed, one outside it is mass invented.
                ax.fill_between(x, tgt_dens, med, color=tint[arm], linewidth=0,
                                zorder=zfill[arm])
            ax.plot(x, med, color=col[arm], lw=lw_line[arm],
                    linestyle=dash[arm], zorder=zline[arm],
                    solid_capstyle="round")
        ax.plot(x, tgt_dens, color="0.66", lw=2.3, zorder=2.9,
                solid_capstyle="round")
        ax.set_title(TARGET_LABEL[tgt], fontsize=9, pad=2.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
        ax.set_xlabel(r"$x$", fontsize=8.5, labelpad=0.5)
        if i == 0:
            ax.set_ylabel("density", fontsize=8.5, labelpad=2)
        ax.set_ylim(bottom=0.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [Patch(facecolor="0.93", edgecolor="0.66", lw=1.2, label="target"),
               Line2D([], [], color=col["lift"], lw=lw_line["lift"], label="lift"),
               Line2D([], [], color=col["direct"], lw=lw_line["direct"],
                      linestyle=dash["direct"], label="direct softplus"),
               Line2D([], [], color=col["pgd"], lw=lw_line["pgd"],
                      linestyle=dash["pgd"], label="PGD")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.4, handletextpad=0.5)

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, "logconcave_density_row_30seed")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to):
        shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf")


if __name__ == "__main__":
    main()
