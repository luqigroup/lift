"""Convergence of the three constructions on the 1-D gallery, 30 paired seeds.

Total-variation distance to the target against iteration, read from the
``history`` stored in every checkpoint; nothing is retrained. One curve per
construction: the median over the 30 seeds inside a 10-90 percentile band.

Grids are cached to ``data/analysis/convergence_1d.npz`` on first run.
Output: ``<out_dir>/logconcave_convergence.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

TARGET_LABEL = {"gumbel": "Gumbel", "laplace": "Laplace",
                "gamma": r"Gamma$(\alpha{=}2)$", "beta": r"Beta$(2,5)$"}
ORDER = ["gumbel", "laplace", "gamma", "beta"]
ARMS = [("lift", "lcmm_1d_logconcave_30seed_poolnorm", "hypernet", "lift"),
        ("pgd", "lcmm_1d_logconcave_30seed_pgd", "pgd", "PGD"),
        ("direct", "lcmm_1d_logconcave_30seed", "direct", "direct softplus")]
# The validation row is off by default: the energy model's validation loss is
# dominated by the normalizer estimate, so all three constructions sit on one
# flat line. Add it back with --rows val,tv.
ALL_ROWS = {"val": "validation loss", "tv": "TV to target"}


def _cache(ckpt_root: str, n_seeds: int, path: str, pinned: bool = True) -> dict:
    import torch
    if pinned:
        import _pinned_lib
    out = {}
    for tgt in ORDER:
        for arm, archive, kind, _ in ARMS:
            tv, val, it = [], [], None
            for s in range(n_seeds):
                c = torch.load(os.path.join(ckpt_root, archive, tgt,
                                            f"seed{s}_{kind}.pt"),
                               map_location="cpu", weights_only=False)
                h = c["history"]
                if pinned and arm == "lift":
                    it, tv_s = _pinned_lib.one_d_tv_history(tgt, s)   # the stored snapshots, pinned
                    tv.append(tv_s)
                else:
                    tv.append(np.asarray(h["tv_per_eval"], dtype=float))
                    it = np.asarray(h["eval_iters"], dtype=float)
                val.append(np.asarray(h["val_loss_per_eval"], dtype=float))
                del c
            out[f"{tgt}/{arm}/tv"] = np.stack(tv)
            out[f"{tgt}/{arm}/val"] = np.stack(val)
            out[f"{tgt}/{arm}/it"] = it
            print(f"cached {tgt:8s} {arm:7s}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def _train_cache(ckpt_root: str, n_seeds: int, path: str) -> dict:
    """``{tgt}/{arm}/train``: every seed's ``loss_E_per_iter`` stacked, the
    objective each run minimized, at every iteration. Seeds are truncated to
    the shortest history so the stack is rectangular."""
    import torch
    out = {}
    for tgt in ORDER:
        for arm, archive, kind, _ in ARMS:
            tr = []
            for s in range(n_seeds):
                c = torch.load(os.path.join(ckpt_root, archive, tgt, f"seed{s}_{kind}.pt"),
                               map_location="cpu", weights_only=False)
                tr.append(np.asarray(c["history"]["loss_E_per_iter"], dtype=float))
                del c
            L = min(len(t) for t in tr)
            out[f"{tgt}/{arm}/train"] = np.stack([t[:L] for t in tr])
            print(f"cached train {tgt:8s} {arm:7s} {L} iterations", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_root", default=os.path.join(_REPO, "data", "checkpoints"))
    p.add_argument("--cache", default=None, help="default: data/analysis/convergence_1d[_pinned].npz by --pinned")
    p.add_argument("--pinned", type=int, default=1, help="1: the lift's TV along training from the pinned re-read of its stored snapshots; 0: the archives' batch-conditioned history")
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--out_dir", default=os.path.join(_REPO, "figures", "figures"))
    p.add_argument("--mirror_to",
                   default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--row_h", type=float, default=1.12)
    p.add_argument("--band", type=float, default=10.0)
    p.add_argument("--rows", default="tv",
                   help="comma-separated: tv, val (see ALL_ROWS)")
    a = p.parse_args()

    if a.cache is None:
        a.cache = os.path.join(_REPO, "data", "analysis", "convergence_1d_pinned.npz" if a.pinned else "convergence_1d.npz")
    d = (dict(np.load(a.cache)) if os.path.isfile(a.cache)
         else _cache(a.ckpt_root, a.n_seeds, a.cache, pinned=bool(a.pinned)))

    apply_paper_style()
    matplotlib.rcParams["savefig.bbox"] = "standard"
    matplotlib.rcParams["axes.grid"] = False
    col = {"lift": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"],
           "pgd": "#2ca02c"}
    dash = {"lift": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}

    rows = [(k, ALL_ROWS[k]) for k in a.rows.split(",") if k in ALL_ROWS]
    W = a.fig_width
    left_in, right_in, gap_in, vgap_in = 0.64, 0.10, 0.22, 0.34
    n = len(ORDER)
    pw = (W - left_in - right_in - (n - 1) * gap_in) / n
    top_in, bot_in, leg_in = 0.20, 0.40, 0.32
    H = (top_in + len(rows) * a.row_h + (len(rows) - 1) * vgap_in
         + bot_in + leg_in)
    fig = plt.figure(figsize=(W, H))

    for r, (key, ylab) in enumerate(rows):
        y0 = (leg_in + bot_in
              + (len(rows) - 1 - r) * (a.row_h + vgap_in)) / H
        for i, tgt in enumerate(ORDER):
            ax = fig.add_axes([(left_in + i * (pw + gap_in)) / W, y0,
                               pw / W, a.row_h / H])
            it = d[f"{tgt}/{arm}/it"]
            for arm, _, _, _ in ARMS:
                M = d[f"{tgt}/{arm}/{key}"]
                ax.fill_between(it, np.percentile(M, a.band, axis=0),
                                np.percentile(M, 100 - a.band, axis=0),
                                color=col[arm], alpha=0.13, linewidth=0, zorder=2)
                ax.plot(it, np.median(M, axis=0), color=col[arm], lw=1.2,
                        linestyle=dash[arm], zorder=3)
            # Log scale: the lift's first-evaluation transient sits orders
            # of magnitude above everything after it, and on a linear axis
            # would compress the whole informative range onto the baseline.
            ax.set_yscale("log")
            ax.set_xlim(0, float(it.max()))
            ax.set_xticks([0, 1500, 3000])
            ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
            if r == 0:
                ax.set_title(TARGET_LABEL[tgt], fontsize=9, pad=2.5)
            if r < len(rows) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("iteration", fontsize=8.5, labelpad=0.5)
            if i == 0:
                ax.set_ylabel(ylab, fontsize=8.5, labelpad=2)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col[arm], lw=1.2, linestyle=dash[arm],
                      label=lab) for arm, _, _, lab in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
               columnspacing=1.5, handletextpad=0.5)

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, "logconcave_convergence")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to):
        shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
