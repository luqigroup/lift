#!/usr/bin/env python3
"""The CIFAR-10 classification task of Sivaprasad et al. (2021): curves and test accuracy.

Render only; nothing here trains. Reads the collector's cache
``data/analysis/sivaprasad_cifar.json`` (written by
``scripts/collect_sivaprasad_cifar.py``) and draws, for the lift, direct
softplus, PGD and the rule of Sivaprasad et al. (after each step a negative
constrained weight w becomes exp(w - 5)), each at its own selected learning
rate:

* the training loss, the validation loss, the validation accuracy and the
  test accuracy per epoch, epoch 0 included; one thin line per seed and the
  pointwise median bold, over the epochs every seed of that method reached;
* the test accuracy at the epoch of best validation accuracy, one point per
  seed and the median of each method as a bar.

The scale-matched control and the non-convex reference are in the cache and
are left out of the figure.

    python scripts/render_sivaprasad_cifar.py

Output: ``figures/sivaprasad_cifar.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from projorg import gitdir  # noqa: E402

_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

METHODS = ["direct", "hypernet", "pgd", "exppost"]          # legend order
KEY = {"hypernet": "lift", "direct": "direct", "pgd": "pgd", "exppost": "exppost"}
LABEL = {"hypernet": "lift", "direct": "direct softplus", "pgd": "PGD",
         "exppost": "Sivaprasad et al."}
COL = {"hypernet": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c",
       "exppost": "#17becf"}
DASH = {"hypernet": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2)),
        "exppost": (0, (1, 1.1))}
Z = {"hypernet": 3.0, "exppost": 3.3, "direct": 3.6, "pgd": 4.0}
CURVES = [("train_loss_levels", "training loss", "log"),
          ("val_loss_levels", "validation loss", "linear"),
          ("val_acc_epochs", "validation accuracy (%)", "acc"),
          ("test_acc_epochs", "test accuracy (%)", "acc")]
DOT_ORDER = METHODS


def _series(per_seed: dict, field: str) -> dict[int, np.ndarray]:
    out = {}
    for s, r in per_seed.items():
        v = np.asarray(r[field], dtype=float)
        out[int(s)] = 100.0 * v if field.endswith("acc_epochs") else v
    return out


def _median_common(series: dict[int, np.ndarray]) -> np.ndarray:
    n = min(len(v) for v in series.values())
    return np.median(np.stack([v[:n] for v in series.values()]), axis=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", default=os.path.join(_REPO, "data", "analysis", "sivaprasad_cifar.json"))
    p.add_argument("--out_dir", default=os.path.join(_REPO, "figures"))
    p.add_argument("--fig_width", type=float, default=5.5)
    a = p.parse_args(argv)

    with open(a.cache) as f:
        table = json.load(f)["table"]
    per = {m: table[KEY[m]]["per_seed"] for m in METHODS}

    apply_paper_style()
    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rcParams["text.usetex"] = False

    W = a.fig_width
    fig = plt.figure(figsize=(W, 3.0))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.995, bottom=0.12, top=0.845,
                          wspace=0.36, hspace=0.68)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0])]
    ax_dot = fig.add_subplot(gs[1, 1:])

    xmax = max(len(v) for m in METHODS for v in _series(per[m], "train_loss_levels").values()) - 1
    for ax, (field, title, kind) in zip(axes, CURVES):
        lo, hi = np.inf, -np.inf
        for m in METHODS:
            ser = _series(per[m], field)
            if kind == "acc":
                for f2 in ("val_acc_epochs", "test_acc_epochs"):
                    for v2 in _series(per[m], f2).values():
                        hi = max(hi, np.nanmax(v2[1:]))
            for s, v in ser.items():
                ep = np.arange(len(v))
                ax.plot(ep, v, color=COL[m], lw=0.55, linestyle=DASH[m], alpha=0.55, zorder=Z[m])
                tail = v[1:] if kind == "acc" else v
                lo, hi = min(lo, np.nanmin(tail)), max(hi, np.nanmax(tail))
            med = _median_common(ser)
            ax.plot(np.arange(len(med)), med, color=COL[m], lw=1.5, linestyle=DASH[m], zorder=Z[m] + 0.5)
        if kind == "log":
            ax.set_yscale("log")
            ax.set_ylim(lo / 1.08, hi * 1.08)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.4, 0.6, 1.0, 2.0]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        elif kind == "acc":
            ax.set_ylim(40.0, hi + 0.6)
            ax.set_yticks([40, 45, 50, 55])
        else:
            pad = 0.04 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("epoch", fontsize=8.5, labelpad=0.5)
        ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    xs = {m: i for i, m in enumerate(DOT_ORDER)}
    for m in DOT_ORDER:
        y = np.array([100.0 * r["test_acc_at_best_val_acc"] for r in per[m].values()])
        ax_dot.scatter(np.full_like(y, xs[m]), y, s=14, color=COL[m], edgecolors="white",
                       linewidths=0.4, zorder=3)
        med = float(np.median(y))
        ax_dot.plot([xs[m] - 0.22, xs[m] + 0.22], [med, med], color=COL[m], lw=1.5, zorder=2,
                    solid_capstyle="butt")
    ax_dot.set_xlim(-0.45, len(DOT_ORDER) - 0.55)
    ax_dot.set_xticks(range(len(DOT_ORDER)))
    ax_dot.set_xticklabels([LABEL[m] for m in DOT_ORDER], fontsize=7.5)
    ax_dot.set_title("test accuracy at the best validation epoch (%)", fontsize=9, pad=4)
    ax_dot.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    for sp in ("top", "right"):
        ax_dot.spines[sp].set_visible(False)

    handles = [Line2D([], [], color=COL[m], lw=1.5, linestyle=DASH[m]) for m in METHODS]
    fig.legend(handles, [LABEL[m] for m in METHODS], loc="upper center", ncol=4, frameon=False,
               fontsize=8, handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.53, 1.0))

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, "sivaprasad_cifar")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    for m in METHODS:
        y = [100.0 * r["test_acc_at_best_val_acc"] for r in per[m].values()]
        print(f"  {LABEL[m]:18s} lr {table[KEY[m]]['lr']:<7g} seeds {sorted(int(s) for s in per[m])} "
              f"median test at best val {np.median(y):.2f}%")
    print("wrote", stem + ".pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
