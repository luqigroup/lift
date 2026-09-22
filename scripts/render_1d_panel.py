"""The 1-D log-concave study as one 2x4 panel.

Columns are the four targets, so a reader reads a target vertically:

  * top row -- the fitted density of each construction at its median-TV seed
    against the target. Lines only: no fills, no bands, no hatching.
  * bottom row -- total-variation distance to the target against iteration,
    median over the 30 paired seeds with a 10-90 percentile band, on a
    logarithmic axis shared across the four columns so the targets are
    directly comparable.

Loaders come from ``render_density_row_1d_30seed.py`` and
``render_convergence_1d.py`` rather than being restated here.

Output: ``<out_dir>/logconcave_1d_panel.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402
import render_density_row_1d_30seed as _dens  # noqa: E402
import render_convergence_1d as _conv  # noqa: E402
import render_tabular_val_loss_multiseed as _tab  # noqa: E402

TARGET_LABEL = _dens.TARGET_LABEL
ORDER = _dens.ORDER
DENS_ARMS = _dens.ARMS      # (method, archive, kind)
CONV_ARMS = _conv.ARMS      # (method, archive, kind, label)

# ---------------------------------------------------------------------------
# Type sizes in points; names ending _IN are inches. They are set for the
# printed 5.5 in column width, not for the screen.
# ---------------------------------------------------------------------------
TITLE_FS, TICK_FS, LABEL_FS, LEG_FS = 9.0, 7.5, 8.5, 8.0

# archive and metrics.json key per construction, taken from the density
# row's own table so the two stay in step
ARCHIVE = {arm: arch for arm, arch, _ in DENS_ARMS}
TVKEY = {arm: kind for arm, _, kind in DENS_ARMS}

# Every held-out read is taken at the iterate of lowest held-out loss, not at
# the last one. Iterates before this one are excluded: the flow-IS normalizer
# has not settled there, and without the guard the rule can select iteration
# zero on an unsettled normalizer.
HOLDOUT_MIN_ITER = 200


def tv_at_best_val(c, tgt: str, arm: str) -> float:
    """Median over seeds of the TV to the target at each seed's best held-out loss.

    ``c`` is the convergence cache: ``tv`` and ``val`` are (seeds, evals) and
    ``it`` carries the evaluation iterates.  The lift's ``tv`` has one column
    more than its ``val`` (the final iterate is stored twice), so the columns
    are aligned on ``val``'s width.
    """
    tv = np.asarray(c[f"{tgt}/{arm}/tv"], float)
    val = np.asarray(c[f"{tgt}/{arm}/val"], float)
    it = np.asarray(c[f"{tgt}/{arm}/it"], float)[: val.shape[1]]
    ok = it >= HOLDOUT_MIN_ITER
    picked = []
    for r in range(val.shape[0]):
        v = np.where(np.isfinite(val[r]) & ok, val[r], np.inf)
        picked.append(tv[r, int(np.argmin(v))])
    return float(np.median(picked))


def _text_w(fig, s: str, fs: float) -> float:
    """Width of ``s`` in inches at font size ``fs``, measured not guessed."""
    t = fig.text(0.0, 0.0, s, fontsize=fs)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def _text_box_rot(fig, s: str, fs: float) -> tuple[float, float]:
    """(thickness, length) of ``s`` in inches when set at ``fs`` and rotated
    90 degrees: the horizontal and vertical footprint of a legend-strip
    label, measured not guessed."""
    t = fig.text(0.0, 0.0, s, fontsize=fs, rotation=90)
    b = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return b.width / fig.dpi, b.height / fig.dpi


def _audit(fig, axes_list, strip=None) -> int:
    """Print every text artist that leaves the canvas or lands on a panel.

    Two tests:

      * CLIPPED -- the artist's bounding box crosses one of the four canvas
        edges, so the PDF loses it. ``savefig.bbox`` is "standard", which
        does not grow the canvas to rescue an overhanging label.
      * INTRUDES -- the artist's bounding box lands inside a different
        panel's data rectangle, which is what a y tick label does when the
        inter-column gap is too narrow to hold it.

    ``strip`` is the right-hand legend strip as ``(axis, entries)`` with
    ``entries`` a list of ``(key, artist)``: its handles and rotated labels
    take the same tests as any tick label, and its rectangle joins the panels
    as something a neighboring tick label must not enter. Entries sharing a
    ``key`` (a handle and its own label) are not tested against each other,
    just as a panel's own ticks are not.

    Returns the number of offenders; zero is the publication bar.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    wpx, hpx = fig.get_size_inches() * fig.dpi
    boxes = [(ax, ax.get_window_extent(r)) for ax in axes_list]
    if strip is not None:
        boxes.append((strip[0], strip[0].get_window_extent(r)))

    def _edges(b, what):
        n = 0
        for side, bad in (("left", b.x0 < -0.5), ("right", b.x1 > wpx + 0.5),
                          ("bottom", b.y0 < -0.5), ("top", b.y1 > hpx + 0.5)):
            if bad:
                print(f"  CLIPPED on the {side} edge: {what}")
                n += 1
        return n

    def _hits(b, what, own=None):
        n = 0
        for other, ob in boxes:
            if other is own:
                continue
            if (b.x0 < ob.x1 - 0.5 and b.x1 > ob.x0 + 0.5
                    and b.y0 < ob.y1 - 0.5 and b.y1 > ob.y0 + 0.5):
                print(f"  INTRUDES into a neighbouring panel: {what}")
                n += 1
        return n

    def _live_tick_labels(axis):
        """Tick labels actually inside the view. A locator also manufactures
        ticks beyond the limits, and those carry text artists parked off the
        canvas that would otherwise be reported as clipped."""
        v0, v1 = sorted(axis.get_view_interval())
        out = []
        for tk in axis.get_major_ticks():
            if v0 - 1e-9 <= tk.get_loc() <= v1 + 1e-9:
                out.append(tk.label1)
        return out

    bad, texts = 0, []
    for ax in axes_list:
        items = ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + _live_tick_labels(ax.xaxis) + _live_tick_labels(ax.yaxis))
        for t in items:
            if not t.get_text() or not t.get_visible():
                continue
            b = t.get_window_extent(r)
            what = repr(t.get_text())
            bad += _edges(b, what) + _hits(b, what, own=ax)
            texts.append((ax, b, what))
    for t in fig.texts:
        b = t.get_window_extent(r)
        what = f"figure text {t.get_text()!r}"
        bad += _edges(b, what) + _hits(b, what)
        texts.append((None, b, what))
    for lg in fig.legends:
        b = lg.get_window_extent(r)
        bad += _edges(b, "the legend") + _hits(b, "the legend")
        texts.append((None, b, "the legend"))
    if strip is not None:
        for key, art in strip[1]:
            b = art.get_window_extent(r)
            what = f"legend strip {key!r}"
            bad += _edges(b, what) + _hits(b, what, own=strip[0])
            texts.append((("strip", key), b, what))

    # Text against text. A tick label at the right end of one column and one
    # at the left end of the next touch neither panel's data rectangle, so the
    # two tests above miss them although they read as a collision on the page.
    # The clearance is 0.05 in, the smallest gap that still parses as a gap at
    # the printed width.
    clear = 0.05 * fig.dpi
    for i in range(len(texts)):
        ax_i, bi, wi = texts[i]
        for j in range(i + 1, len(texts)):
            ax_j, bj, wj = texts[j]
            if ax_i is not None and ax_i == ax_j:
                continue            # a panel's own ticks never collide
            dx = max(bj.x0 - bi.x1, bi.x0 - bj.x1)
            dy = max(bj.y0 - bi.y1, bi.y0 - bj.y1)
            if dx < clear and dy < clear:
                print(f"  CROWDED  {wi} and {wj} are "
                      f"{max(max(dx, dy), 0.0) / fig.dpi:.3f} in apart")
                bad += 1
    # A panel's own x tick labels can also crowd each other, and the pair
    # loop above skips same-axis pairs because an axis label legitimately sits
    # a hair from its own ticks.
    for ax in axes_list:
        lb = [t for t in _live_tick_labels(ax.xaxis)
              if t.get_text() and t.get_visible()]
        bs = [(t.get_window_extent(r), t.get_text()) for t in lb]
        bs.sort(key=lambda e: e[0].x0)
        for (b0, t0), (b1, t1) in zip(bs, bs[1:]):
            if b1.x0 - b0.x1 < clear:
                print(f"  CROWDED  own x ticks {t0!r} and {t1!r} are "
                      f"{(b1.x0 - b0.x1) / fig.dpi:.3f} in apart")
                bad += 1
    print(f"  edge/overlap audit: {bad} offender(s)")
    return bad


# The drawn curve is the median-TV seed, an actual run, not the pointwise
# median across seeds: a pointwise median cancels seed-specific, sign-varying
# error while leaving systematic error intact, and the constructions differ in
# how much of each they carry. With the median-TV seed the drawn curve's error
# equals the number the row below reports for that construction.
def median_tv_seed(ckpt_root, archive, key, tgt, n_seeds=30):
    with open(os.path.join(ckpt_root, archive, tgt, "metrics.json")) as fh:
        ps = json.load(fh)["per_seed"]
    tvs = np.array([float(ps[str(s)][key]["tv"]) for s in range(n_seeds)])
    return int(np.argsort(tvs)[len(tvs) // 2]), tvs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_root", default=os.path.join(_REPO, "data", "checkpoints"))
    p.add_argument("--cache", default=None, help="default: data/analysis/density_grids_1d[_pinned].npz by --pinned")
    p.add_argument("--conv_cache", default=None, help="default: data/analysis/convergence_1d[_pinned].npz by --pinned")
    p.add_argument("--pinned", type=int, default=1, help="1 (default): the lift read with its emission pinned on the full training data; 0: the batch-conditioned records")
    p.add_argument("--curve", choices=("train", "tv"), default="train",
                   help="bottom row: the training objective (default) or the TV to the target along training")
    p.add_argument("--train_cache", default=os.path.join(_REPO, "data", "analysis", "convergence_1d_train.npz"))
    p.add_argument("--skip_frac", type=float, default=0.1, help="--curve train: the y-range is taken over the run after this fraction of it")
    p.add_argument("--holdout", choices=("block", "table"), default="table",
                   help="--curve train: how the median TV to the target at the best held-out loss is shown. block: colored numbers on one line in the "
                        "headroom (default); table: a framed legend-like table in the headroom, dash handle beside each number "
                        "")
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--top", choices=("lines", "fill", "band", "both"),
                   default="lines",
                   help="top row: discrepancy fill, 10-90 seed band, or both")
    p.add_argument("--tint", type=float, default=0.45,
                   help="weight of the construction color in its opaque error lobe")
    p.add_argument("--target_edge", default="black")
    p.add_argument("--target_lw", type=float, default=0.9)
    p.add_argument("--lift_lw", type=float, default=1.3)
    p.add_argument("--base_lw", type=float, default=1.1)
    p.add_argument("--band", type=float, default=10.0)
    p.add_argument("--band_alpha", type=float, default=0.15)
    p.add_argument("--out_dir", default=os.path.join(_REPO, "figures", "figures"))
    p.add_argument("--stem", default=None, help="default: logconcave_1d_panel (--curve tv) or logconcave_1d_panel_train")
    p.add_argument("--mirror_to",
                   default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--dens_h", type=float, default=0.86)
    p.add_argument("--conv_h", type=float, default=1.04)   # with --curve train, room for the held-out table above the bands
    p.add_argument("--trim", type=float, default=0.004,
                   help="crop x to where any curve exceeds this fraction of "
                        "the column's peak density (0 disables)")
    a = p.parse_args()

    sfx = "_pinned" if a.pinned else ""
    if a.cache is None:
        a.cache = os.path.join(_REPO, "data", "analysis", f"density_grids_1d{sfx}.npz")
    if a.conv_cache is None:
        a.conv_cache = os.path.join(_REPO, "data", "analysis", f"convergence_1d{sfx}.npz")
    if a.stem is None:
        a.stem = "logconcave_1d_panel" + ("_train" if a.curve == "train" else "")
    d = (dict(np.load(a.cache)) if os.path.isfile(a.cache)
         else _dens._build_cache(a.ckpt_root, a.n_seeds, a.cache, pinned=bool(a.pinned)))
    c = (dict(np.load(a.conv_cache)) if os.path.isfile(a.conv_cache)
         else _conv._cache(a.ckpt_root, a.n_seeds, a.conv_cache, pinned=bool(a.pinned)))
    if a.curve == "train":
        ct = (dict(np.load(a.train_cache)) if os.path.isfile(a.train_cache)
              else _conv._train_cache(a.ckpt_root, a.n_seeds, a.train_cache))
        # A running median over 1% of the run per seed: the per-iteration
        # objective is a 512-sample estimate whose noise would hide the band.
        def _smooth(M):
            w = max(1, M.shape[1] // 100)
            return np.stack([[np.nanmedian(r[max(0, i - w):i + 1]) for i in range(len(r))] for r in M])
        ct = {k: _smooth(v) for k, v in ct.items()}

    apply_paper_style()
    matplotlib.rcParams["savefig.bbox"] = "standard"
    matplotlib.rcParams["hatch.linewidth"] = 0.35
    matplotlib.rcParams["axes.grid"] = False
    col = {"lift": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"],
           "pgd": "#2ca02c"}
    dash = {"lift": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
    # The lift is the widest line and sits under the baselines, so where a
    # baseline lands on it the coincidence still reads as one. The same order
    # holds in both rows.
    zline = {"lift": 3.0, "direct": 3.6, "pgd": 4.0}
    lw_line = {"lift": a.lift_lw, "direct": a.base_lw, "pgd": a.base_lw}
    # Error lobes are opaque tints, not alpha washes: three translucent fills
    # over a gray target turn to mud. Off by default (--top lines).
    tint = {k: tuple(1.0 - a.tint * (1.0 - ch) for ch in mcolors.to_rgb(v))
            for k, v in col.items()}
    zfill = {"direct": 2.0, "pgd": 2.1, "lift": 2.2}
    hatch = {"direct": "//////", "pgd": "\\\\\\\\\\\\", "lift": None}

    W = a.fig_width
    # The right margin holds the legend strip: four entries stacked down the
    # right edge, outside every panel, each a short vertical line sample in
    # the construction's color and dash with its label rotated 90 degrees
    # beside it. The strip costs width only, taken out of the panel widths
    # rather than the 5.5 in canvas. It is built outward from the last panel:
    # clearance for row 1's last tick label (row 2's end ticks are tucked
    # inside the panel), the handle column, its pad, the rotated label's
    # measured thickness, then edge slack. savefig.bbox is "standard", which
    # does not grow the canvas to rescue anything that pokes out.
    gap_in = 0.25 if a.curve != "train" else 0.34   # the training objective has its own y ticks per column
    strip_gap_in, strip_handle_in, strip_pad_in = 0.10, 0.30, 0.045
    strip_lw_in = 0.04                  # the widest handle is the 2.3 pt target
    STRIP_FS = TICK_FS                  # the strip sits in the tick-label tier
    strip_entries = [                   # bottom to top: the head-turned reading order
        ("target", a.target_edge, a.target_lw, (0, ())),
        ("lift", col["lift"], lw_line["lift"], dash["lift"]),
        ("PGD", col["pgd"], lw_line["pgd"], dash["pgd"]),
        ("direct softplus", col["direct"], lw_line["direct"], dash["direct"]),
    ]
    # Every band below is the measured footprint of what sits in it plus
    # 0.05 in of clearance. A 7.5 pt tick label is 0.12 in tall and hangs
    # 0.035 in of tick plus 0.021 in of pad below its spine, so 0.22 in.
    top_in = 0.17                       # column titles
    dens_x_in = 0.22                    # row-1 tick labels
    xlab_in = 0.145                     # the shared, centered row-2 x label
    xlab1_in = 0.0                      # row 1 has no x label
    mid_in = 0.04                       # rows as close as the row-1 tick labels allow
    conv_x_in = 0.22                    # row-2 tick labels
    leg_in = 0.0                        # the legend is the right-hand strip: no band of its own
    H = (top_in + a.dens_h + dens_x_in + xlab1_in + mid_in
         + a.conv_h + conv_x_in + xlab_in + leg_in)
    fig = plt.figure(figsize=(W, H))

    # Left margin, built outward from the canvas edge so nothing can fall off
    # it: edge slack, the rotated y label, its gap, the widest tick label any
    # row can produce, then the tick and its pad. Both rotated labels are
    # pinned to one x; left to matplotlib they sit at a fixed pad from their
    # own ticks, which puts the two rows' labels at different depths.
    lab_h = LABEL_FS / 72.0 * 1.35      # a rotated label's footprint
    tick_w = max(_text_w(fig, "0.30", TICK_FS),
                 _text_w(fig, r"$10^{-2}$", TICK_FS))
    edge_in, ylab_gap_in = 0.06, 0.075
    ylab_x_in = edge_in + lab_h / 2
    left_in = edge_in + lab_h + ylab_gap_in + tick_w + 0.056
    # A rotated label's thickness, measured on the tallest glyphs it carries.
    strip_text_in = max(_text_box_rot(fig, s, STRIP_FS)[0]
                        for s, _, _, _ in strip_entries)
    right_in = (strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in
                + edge_in)
    pw = (W - left_in - right_in - 3 * gap_in) / 4
    y_conv = (leg_in + xlab_in + conv_x_in) / H
    y_dens = y_conv + (a.conv_h + mid_in + xlab1_in + dens_x_in) / H

    # ---- row 2's logarithmic TV axis is shared across the four columns ----
    # Each target's own 10-90 envelope already spans about two decades, so the
    # union costs almost no range. It buys tick labels on column 0 only, so no
    # exponent lands on a neighboring panel's curves, and it makes the four
    # columns directly comparable.
    lo = min(np.percentile(c[f"{t}/{arm}/tv"], a.band, axis=0).min()
             for t in ORDER for arm, _, _, _ in CONV_ARMS)
    hi = max(np.percentile(c[f"{t}/{arm}/tv"], 100 - a.band, axis=0).max()
             for t in ORDER for arm, _, _, _ in CONV_ARMS)
    ylim_tv = (lo / 1.25, hi * 1.12)

    axes_all = []
    for i, tgt in enumerate(ORDER):
        x0 = (left_in + i * (pw + gap_in)) / W

        # ------------------------- density panel -------------------------
        ax = fig.add_axes([x0, y_dens, pw / W, a.dens_h / H])
        axes_all.append(ax)
        x = d[f"{tgt}/x"]
        tdens = np.exp(d[f"{tgt}/target"])
        ax.plot(x, tdens, color=a.target_edge, lw=a.target_lw, zorder=9.0,
                solid_capstyle="round")
        peak = tdens.max()
        for arm, _, _ in DENS_ARMS:
            dens = np.exp(d[f"{tgt}/{arm}"])
            k, _ = median_tv_seed(a.ckpt_root, ARCHIVE[arm], TVKEY[arm], tgt)
            med = dens[k]
            peak = max(peak, med.max())
            if a.top in ("band", "both"):
                ax.fill_between(x, np.percentile(dens, a.band, axis=0),
                                np.percentile(dens, 100 - a.band, axis=0),
                                color=col[arm], alpha=0.12, linewidth=0, zorder=2)
            if a.top in ("fill", "both"):
                ax.fill_between(x, tdens, med, color=tint[arm], linewidth=0,
                                zorder=zfill[arm])
            ax.plot(x, med, color=col[arm], lw=lw_line[arm],
                    linestyle=dash[arm], zorder=zline[arm])
        if a.top in ("fill", "both"):
            for arm, _, _ in DENS_ARMS:
                if hatch[arm] is None:
                    continue
                k, _ = median_tv_seed(a.ckpt_root, ARCHIVE[arm], TVKEY[arm], tgt)
                med = np.exp(d[f"{tgt}/{arm}"])[k]
                ax.fill_between(x, tdens, med, facecolor="none",
                                hatch=hatch[arm], edgecolor=col[arm],
                                linewidth=0.0, zorder=2.8)

        if a.trim > 0:                  # crop the dead tails, not the error
            hi_m = np.exp(d[f"{tgt}/target"])
            for arm, _, _ in DENS_ARMS:
                k, _ = median_tv_seed(a.ckpt_root, ARCHIVE[arm], TVKEY[arm], tgt)
                hi_m = np.maximum(hi_m, np.exp(d[f"{tgt}/{arm}"])[k])
            keep = np.where(hi_m > a.trim * peak)[0]
            ax.set_xlim(x[keep[0]], x[keep[-1]])
        ax.set_title(TARGET_LABEL[tgt], fontsize=TITLE_FS, pad=2.5)
        # Three ticks a side, and `steps` forces round ones: "0.4" is a third
        # narrower than "0.30", which is the difference between a y tick label
        # that fits the inter-column gap and one that sits on its neighbor.
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(labelsize=TICK_FS, length=2.5, width=0.6, pad=1.5)
        if i == 0:
            ax.set_ylabel("density", fontsize=LABEL_FS)
            ax.yaxis.set_label_coords((ylab_x_in - left_in) / pw, 0.5)
        # A uniform 8% of headroom, not autoscale's 5%: it keeps the top tick
        # from falling just outside the view on one column and inside it on
        # the next, and stops the tallest curve grazing the title.
        ax.set_ylim(0.0, peak * 1.08)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # --------------------- TV against iteration ----------------------
        ax = fig.add_axes([x0, y_conv, pw / W, a.conv_h / H])
        axes_all.append(ax)
        for arm, _, _, _ in CONV_ARMS:
            it = c[f"{tgt}/{arm}/it"]   # each construction on its own grid: the pinned lift curve sits on its stored snapshots
            M = c[f"{tgt}/{arm}/tv"]
            if a.curve == "train":
                M = ct[f"{tgt}/{arm}/train"]; it = np.arange(1, M.shape[1] + 1, dtype=float)
            ax.fill_between(it, np.percentile(M, a.band, axis=0),
                            np.percentile(M, 100 - a.band, axis=0),
                            color=col[arm], alpha=a.band_alpha, linewidth=0,
                            zorder=zline[arm] - 2.0)
            ax.plot(it, np.median(M, axis=0), color=col[arm], lw=lw_line[arm],
                    linestyle=dash[arm], zorder=zline[arm],
                    solid_capstyle="round")
        if a.curve == "train":
            # Linear, the range set by the run after its first tenth: the
            # objective's opening transient is the same on every construction
            # and would compress the descent to a line.
            k = int(round(a.skip_frac * M.shape[1]))
            lo = min(np.percentile(ct[f"{tgt}/{arm}/train"][:, k:], a.band, axis=0).min() for arm, _, _, _ in CONV_ARMS)
            hi = max(np.percentile(ct[f"{tgt}/{arm}/train"][:, k:], 100 - a.band, axis=0).max() for arm, _, _, _ in CONV_ARMS)
            head = 0.50 if a.holdout == "block" else 0.82          # the table is three rows and a title, so it needs taller headroom
            pad = 0.06 * (hi - lo); ax.set_ylim(lo - pad, hi + pad + head * (hi - lo))   # headroom for the held-out read
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            tv_end = _tab.holdout_order([(tv_at_best_val(c, tgt, arm), arm) for arm, _, _, _ in CONV_ARMS])
            if a.holdout == "block":
                # The held-out read as a block in the headroom: a header, then
                # the median over seeds of the TV to the target at each seed's
                # best held-out loss, on one line in the strip's colors.
                ax.text(0.97, 0.96, "TV to target", transform=ax.transAxes, ha="right", va="top", fontsize=TICK_FS - 0.5, color="0.35")
                for kk, (arm, _, _, _) in enumerate(reversed(CONV_ARMS)):
                    ax.text(0.97 - 0.31 * kk, 0.80, f"{tv_at_best_val(c, tgt, arm):.3f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=TICK_FS, color=col[arm])
            else:
                # A legend whose entries are the values: the construction's
                # dash as the handle and its median TV at the best held-out
                # loss as the label, in the strip's order, framed, in the upper
                # right of the headroom where the transient does not sit.
                handles = [Line2D([], [], color=col[arm], lw=1.2, linestyle=dash[arm]) for _, arm in tv_end]
                leg = ax.legend(handles, [f"{v:.3f}" for v, _ in tv_end], title="TV to target", loc="upper right",
                                bbox_to_anchor=(1.0, 1.0), frameon=True, fancybox=False, fontsize=TICK_FS - 1.0,
                                title_fontsize=TICK_FS - 1.5, handlelength=1.1, handletextpad=0.45, borderpad=0.3,
                                labelspacing=0.22, borderaxespad=0.1)
                leg.get_frame().set_edgecolor("0.65"); leg.get_frame().set_linewidth(0.5); leg.get_title().set_color("0.35")
                for t, (_, arm) in zip(leg.get_texts(), tv_end):
                    t.set_color(col[arm])
        else:
            # Logarithmic: on a linear axis the first-evaluation transient,
            # two orders of magnitude above everything that follows, compresses
            # the whole informative range onto the axis floor.
            ax.set_yscale("log")
            ax.set_ylim(*ylim_tv)
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(0.0, 3000.0)
        ax.set_xticks([0, 1500, 3000])
        ax.tick_params(labelsize=TICK_FS, length=2.5, width=0.6, pad=1.5)
        # The first and last ticks sit on the spines, so a centered label
        # hangs half its width into the neighboring column, and off the canvas
        # in the last column. Tucking them inward buys back the inter-column
        # clearance and the right margin at once.
        lab = ax.get_xticklabels()
        lab[0].set_horizontalalignment("left")
        lab[-1].set_horizontalalignment("right")
        ax.tick_params(axis="y", which="minor", length=1.3, width=0.5)
        if i == 0 or a.curve == "train":     # the training objective has its own scale per target
            ax.set_ylabel("training loss" if a.curve == "train" else "TV to target", fontsize=LABEL_FS)
            ax.yaxis.set_label_coords((ylab_x_in - left_in) / pw, 0.5)
        if i > 0 and a.curve == "train":
            ax.set_ylabel("")
        if i > 0 and a.curve != "train":
            ax.set_yticklabels([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # One x label a row, centered under the four columns, since the four
    # panels of a row carry the same variable.
    cx = (left_in + (4 * pw + 3 * gap_in) / 2) / W
    fig.text(cx, (leg_in + xlab_in) / H, "iteration",
             ha="center", va="top", fontsize=LABEL_FS)

    # ------------------------- the legend strip -------------------------
    # One narrow axis, axis off, spanning both rows down the right edge. Its
    # data coordinates are inches from its lower-left corner, so the layout
    # below is the same arithmetic as the margins above. It costs neither
    # panel area, as an in-panel legend does, nor height, as a legend band
    # below the figure does.
    sx0_in = left_in + 4 * pw + 3 * gap_in + strip_gap_in
    sw_in = W - edge_in - sx0_in
    sy0_in = leg_in + xlab_in + conv_x_in            # bottom of row 2
    sh_in = (a.conv_h + mid_in + xlab1_in + dens_x_in + a.dens_h)  # to top of row 1
    ax_s = fig.add_axes([sx0_in / W, sy0_in / H, sw_in / W, sh_in / H])
    ax_s.set_axis_off()
    ax_s.set_xlim(0.0, sw_in)
    ax_s.set_ylim(0.0, sh_in)
    # Each entry's footprint is the longer of its handle and its rotated
    # label; the slack is shared out as equal gaps, half a gap at each end.
    foot = [max(strip_handle_in, _text_box_rot(fig, s, STRIP_FS)[1])
            for s, _, _, _ in strip_entries]
    sgap = (sh_in - sum(foot)) / len(foot)
    xh = strip_lw_in / 2                             # handle column centre
    xt = strip_lw_in + strip_pad_in + strip_text_in / 2  # label column centre
    strip_arts, yc = [], sgap / 2
    for (label, colr, lw, ls), f in zip(strip_entries, foot):
        yc += f / 2
        h = Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2],
                   color=colr, lw=lw, linestyle=ls, solid_capstyle="butt",
                   dash_capstyle="butt", clip_on=False)
        ax_s.add_line(h)
        t = ax_s.text(xt, yc, label, rotation=90, ha="center", va="center",
                      fontsize=STRIP_FS, clip_on=False)
        strip_arts += [(label, h), (label, t)]
        yc += f / 2 + sgap

    _audit(fig, axes_all, strip=(ax_s, strip_arts))
    print("  " + ", ".join(f"{k}={matplotlib.rcParams[k]!r}" for k in
                           ("pdf.fonttype", "ps.fonttype",
                            "axes.unicode_minus", "savefig.bbox")))

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, a.stem)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to):
        shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in), "
          f"left={left_in:.3f} in, panel={pw:.3f} in")


if __name__ == "__main__":
    main()
