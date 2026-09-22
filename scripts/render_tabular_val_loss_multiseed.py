"""Loss along training for the tabular datasets, every seed.

Three columns, POWER / HEPMASS / MiniBooNE, log y, the three constructions in
the paper's colors and dash patterns; one thin line per seed, the pointwise
median bold, and the median held-out test NLL over the seeds that finished.

Failed seeds are drawn rather than dropped. A run the trainer aborted gets a
dotted vertical rule at its abort iteration with a cross on the top spine; a
run that finished with its loss outside the panel gets a downward triangle at
its last in-panel evaluation. Both carry the count of such seeds.

The first evaluation is at initialization, where the lift's
importance-sampled normalizer is meaningless, so the x axis starts at the
next evaluation and each panel's y range comes from the post-transient part
of the run.

Reads ``data/analysis/tabular_multiseed.npz``; writes
``<out_dir>/tabular_val_loss.{pdf,png}``.
"""
from __future__ import annotations

import argparse, os, shutil, sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

COLS = [("POWER (6 dim.)", "power"), ("HEPMASS (21 dim.)", "hepmass"), ("MiniBooNE (43 dim.)", "miniboone")]
ARMS = [("pgd", "PGD"), ("direct", "direct softplus"), ("hypernet", "lift")]
# y range: top = this multiple of the largest value at or after ``Y_FROM``
Y_FROM = {"power": 200.0, "hepmass": 300.0, "miniboone": 1200.0}  # MiniBooNE: the sampler ramp runs to 1,000, so its transient lasts longer
Y_TOP_PAD, Y_BOT_PAD = 1.6, 0.8


def _text_box_rot(fig, s, fs):
    """(thickness, length) in inches of ``s`` at ``fs`` rotated 90 degrees."""
    t = fig.text(0.0, 0.0, s, fontsize=fs, rotation=90)
    b = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return b.width / fig.dpi, b.height / fig.dpi


def _median_curve(d, name, arm, seeds, series=None):
    if series is not None:   # pointwise median of the plotted (possibly smoothed) series
        rows = [series(name, arm, s) for s in seeds]; L = min(len(r[1]) for r in rows)
        return rows[0][0][:L], np.nanmedian(np.stack([r[1][:L] for r in rows]), axis=0)
    """Pointwise median over the seeds that still have a finite value."""
    grids = [d[f"{name}/{arm}/s{s}/it"] for s in seeds]
    it = max(grids, key=len)
    M = np.full((len(seeds), len(it)), np.nan)
    for r, s in enumerate(seeds):
        g, v = d[f"{name}/{arm}/s{s}/it"], d[f"{name}/{arm}/s{s}/val"]
        idx = np.searchsorted(it, g)
        ok = (idx < len(it)) & np.isin(g, it)
        M[r, idx[ok]] = v[ok]
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(M), M, np.nan), axis=0)
    return it, med


HOLDOUT_ORDER = ("direct", "pgd", "hypernet", "lift")   # the strip's order top to bottom, the same in every table


def holdout_order(ends):
    """[(value, key)] in the fixed construction order of the legend strip; a
    lowest-first table would read as a ranking and mislead on ties."""
    return sorted(ends, key=lambda e: HOLDOUT_ORDER.index(e[1]))


# The held-out read of a run is its lowest held-out loss, not its last one.
# The opening evaluations are excluded: these problems normalize by flow
# importance sampling, whose estimate is orders of magnitude off over the
# first few evaluations, so an unguarded minimum selects one of them rather
# than a property of the model.
HOLDOUT_MIN_ITER = 1000


def best_val(curve, iters, min_iter: float = HOLDOUT_MIN_ITER) -> float:
    """Lowest value of one run's held-out curve, outside the opening guard."""
    v = np.asarray(curve, float); it = np.asarray(iters, float)[: len(curve)]
    m = np.isfinite(v) & (it >= min_iter)
    return float(np.min(v[m])) if m.any() else float("nan")


def _draw_holdout(fig, ax, mode, ends, col, dash, fmt, rect_in, fig_in, inside=False, anchor_x=1.0):
    """The held-out read of one panel, kept visibly apart from the curve.

    ``ends`` is [(value, key)] in construction order, ``rect_in`` the panel's
    (x, y, w, h) in inches and ``fig_in`` the figure's (W, H). Each mode gives
    the number an object of its own -- a legend frame, an axis, or a row -- so
    it cannot be mistaken for an end label.
    """
    x_in, y_in, w_in, h_in = rect_in; W, H = fig_in
    if mode == "none":
        return
    if mode == "table":
        # A legend whose entries are the values: a dash handle in the
        # construction's style, the median held-out test NLL as its label.
        # It sits outside the right edge, top-aligned.
        handles = [Line2D([], [], color=col[arm], lw=1.2, linestyle=dash[arm]) for _, arm in ends]
        leg = ax.legend(handles, [fmt(v) for v, _ in ends], title="held-out", loc="upper right" if inside else "upper left",
                        bbox_to_anchor=(anchor_x, 1.0) if inside else (anchor_x, 1.02), frameon=True, fancybox=False, fontsize=6.8, title_fontsize=6.0,
                        handlelength=1.1, handletextpad=0.45, borderpad=0.3, labelspacing=0.22, borderaxespad=0.12)
        leg.get_frame().set_edgecolor("0.65"); leg.get_frame().set_linewidth(0.5)
        leg.get_title().set_color("0.35")
        for t, (_, arm) in zip(leg.get_texts(), ends):
            t.set_color(col[arm])
        return
    if mode == "axis":
        # A companion axis: the three values as dots on their own log scale
        # and spine, the number beside each dot. It is shorter than the panel
        # and top-aligned so its scale is not read against the panel's.
        cw_in, ch_in, gap = 0.09, 0.62 * h_in, 0.07
        cax = fig.add_axes([(x_in + w_in + gap) / W, (y_in + h_in - ch_in) / H, cw_in / W, ch_in / H])
        vals = np.array([v for v, _ in ends]); lo, hi = np.log10(vals.min()), np.log10(vals.max())
        span = max(hi - lo, 0.3); lo, hi = lo - 0.22 * span, hi + 0.22 * span
        cax.set_yscale("log"); cax.set_ylim(10 ** lo, 10 ** hi); cax.set_xlim(-1, 1)
        cax.set_xticks([]); cax.set_yticks([]); cax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for sp in ("top", "right", "bottom"): cax.spines[sp].set_visible(False)
        cax.spines["left"].set_color("0.35"); cax.spines["left"].set_linewidth(0.6)
        cax.text(-0.9, -0.06, "held-out\nNLL", transform=cax.transAxes, ha="left", va="top", fontsize=6.0, color="0.35",
                 linespacing=0.95)
        # label positions dodged in log space so three 7-pt numbers clear each other
        step = 0.105 * 1.18 / ch_in * (hi - lo)     # ~0.11 in, whatever the companion's height
        ys = [np.log10(v) for v, _ in ends]
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < step: ys[k] = ys[k - 1] + step
        for (v, arm), y in zip(ends, ys):
            cax.plot([0.0], [v], marker="o", ms=3.6, color=col[arm], mec="white", mew=0.4, clip_on=False, zorder=5)
            cax.annotate(fmt(v), xy=(0.0, v), xytext=(1.25, 10 ** y), textcoords=("data", "data"),
                         fontsize=6.8, color=col[arm], ha="left", va="center", clip_on=False)
        return
    if mode == "strip":
        # A row below the x label: a gray row label, then dot and number per
        # construction, lowest first.
        sax = fig.add_axes([x_in / W, (y_in - 0.36) / H, w_in / W, 0.12 / H]); sax.set_axis_off()
        sax.set_xlim(0, w_in); sax.set_ylim(0, 1)
        sax.text(0.0, 0.5, "held-out", ha="left", va="center", fontsize=6.0, color="0.35")
        xs = 0.36; cell = (w_in - xs) / len(ends)
        for k, (v, arm) in enumerate(ends):
            x = xs + k * cell
            sax.plot([x + 0.035], [0.5], marker="o", ms=3.4, color=col[arm], clip_on=False, zorder=5)
            sax.text(x + 0.085, 0.5, fmt(v), ha="left", va="center", fontsize=6.8, color=col[arm], clip_on=False)
        return
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=f"{_REPO}/data/analysis/tabular_multiseed.npz")
    p.add_argument("--out_dir", default=f"{_REPO}/figures")
    p.add_argument("--mirror_to", default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--panel_h", type=float, default=1.18)   # nine seeds per panel
    p.add_argument("--curve", choices=["val", "train"], default="train",
                   help="train (default): "
                        "the objective each run minimized, per iteration, thin per seed with a running median over one "
                        "percent of the run, the pointwise median bold; val: the held-out validation loss along training, "
                        "the appendix companion")
    p.add_argument("--holdout", choices=["block", "ends", "table", "axis", "strip", "none"], default=None,
                   help="how the median held-out test NLL of each construction is shown ("
                        "reads as the curve's last value). block: colored numbers outside the right edge (default with "
                        "--curve train); ends: labels at the curve ends (default with --curve val); table: a framed "
                        "legend-like table outside the right edge, dash handle beside each number; axis: a narrow "
                        "companion axis right of the panel with the values as dots on their own scale; strip: a row of "
                        "dot + number below the x label; none: prose only")
    a = p.parse_args()
    d = dict(np.load(a.cache, allow_pickle=True))
    if a.holdout is None:
        a.holdout = "table" if a.curve == "train" else "ends"
    def series(name, arm, s):
        """(iterations, values) of the plotted curve for one seed."""
        if a.curve == "val":
            return d[f"{name}/{arm}/s{s}/it"], d[f"{name}/{arm}/s{s}/val"]
        it, v = d[f"{name}/{arm}/s{s}/train_it"], d[f"{name}/{arm}/s{s}/train"]
        w = max(1, len(v) // 100)
        return it, np.array([np.nanmedian(v[max(0, i - w):i + 1]) for i in range(len(v))])

    apply_paper_style()
    matplotlib.rcParams["savefig.bbox"] = "standard"
    matplotlib.rcParams["axes.grid"] = False
    col = {"hypernet": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c"}
    dash = {"hypernet": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
    z = {"hypernet": 3.0, "direct": 3.6, "pgd": 4.0}

    W = a.fig_width
    n = len(COLS)
    strip_entries = [("lift", col["hypernet"], 1.2, dash["hypernet"]),
                     ("PGD", col["pgd"], 1.2, dash["pgd"]),
                     ("direct softplus", col["direct"], 1.2, dash["direct"])]
    STRIP_FS = 7.0
    strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.44, 0.30, 0.045, 0.04, 0.06
    fig = plt.figure(figsize=(W, 2.0))
    strip_text_in = max(_text_box_rot(fig, t, STRIP_FS)[0] for t, _, _, _ in strip_entries)
    plt.close(fig)
    left_in, gap_in = 0.62, {"table": 0.72, "axis": 0.66}.get(a.holdout, 0.62)   # the framed table / companion axis live in the gap
    tail_in = {"table": 0.42, "axis": 0.34}.get(a.holdout, 0.0)                    # and the last column needs the same room before the strip
    right_in = tail_in + strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in
    pw = (W - left_in - right_in - (n - 1) * gap_in) / n
    top_in, bot_in, leg_in = 0.29, (0.51 if a.holdout == "strip" else 0.33), 0.0   # the strip row sits below the x label
    H = top_in + a.panel_h + bot_in + leg_in
    fig = plt.figure(figsize=(W, H))
    for i, (title, name) in enumerate(COLS):
        ax = fig.add_axes([(left_in + i * (pw + gap_in)) / W, (bot_in + leg_in) / H, pw / W, a.panel_h / H])
        # ---- y range from the post-transient part of every seed ----
        lo, hi = np.inf, -np.inf
        for arm, _ in ARMS:
            for s in d[f"{name}/{arm}/seeds"]:
                it, v = series(name, arm, s)
                m = (it >= Y_FROM[name]) & np.isfinite(v) & (v > 0)
                if m.any():
                    lo, hi = min(lo, v[m].min()), max(hi, v[m].max())
        ax.set_yscale("log"); ax.set_ylim(lo * Y_BOT_PAD, hi * Y_TOP_PAD)
        ylo, yhi = ax.get_ylim()
        notes, marks = [], []
        for arm, _ in ARMS:
            seeds = list(d[f"{name}/{arm}/seeds"])
            n_div = n_gone = 0
            for s in seeds:
                it, v = series(name, arm, s)
                m = (it > 0) & np.isfinite(v) & (v > 0)
                ax.plot(it[m], v[m], color=col[arm], lw=0.55, linestyle=dash[arm], alpha=0.55, zorder=z[arm])
                div = float(d[f"{name}/{arm}/s{s}/div"])
                if np.isfinite(div):                       # the trainer aborted this run
                    n_div += 1
                    ax.axvline(div, color=col[arm], lw=0.7, linestyle=(0, (1, 1.4)), alpha=0.9, zorder=z[arm] + 1)
                    ax.plot([div], [yhi], marker="x", ms=3.4, mew=0.9, color=col[arm], clip_on=False, zorder=6)
                    marks.append(("x", div, yhi, arm))
                elif m.any() and not m[-1]:                # finished, but the loss left the panel
                    n_gone += 1
                    k = np.where(m)[0][-1]
                    ax.plot([it[k]], [v[k]], marker="v", ms=3.6, mfc="none", mew=0.9, color=col[arm],
                            clip_on=False, zorder=6)
                    marks.append(("o", it[k], v[k], arm))
            it, med = _median_curve(d, name, arm, seeds, series)
            mm = np.isfinite(med) & (med > 0) & (it > 0)
            ax.plot(it[mm], med[mm], color=col[arm], lw=1.5, linestyle=dash[arm], zorder=z[arm] + 0.5)
            if n_div:
                notes.append(("x", f"{n_div}/{len(seeds)} diverged", col[arm]))
            if n_gone:
                notes.append(("v", f"{n_gone}/{len(seeds)} non-physical", col[arm]))
        ax.set_xlim(0, 4000); ax.set_xticks([0, 2000, 4000])
        # ---- end labels: the median best held-out loss over finished seeds ----
        ends = []
        for arm, _ in ARMS:
            t = [best_val(d[f"{name}/{arm}/s{s}/val"], d[f"{name}/{arm}/s{s}/it"])
                 for s in d[f"{name}/{arm}/seeds"]]
            # The median is taken over every finished seed. Dropping the
            # non-physical ones first would move it onto the survivors alone,
            # and the panel already marks those seeds.
            t = [x for x in t if np.isfinite(x)]
            if t:
                ends.append((float(np.median(t)), arm))
        fmt = lambda v: f"{v:.1f}" if v >= 100 else f"{v:.2f}" if v < 10 else f"{v:.1f}"
        if a.holdout == "block":
            # The median best held-out loss of each construction, outside the
            # axes on the right, stacked from the top, lowest first.
            ax.text(1.03, 1.0, "held-out", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color="0.35", clip_on=False)
            for k, (v, arm) in enumerate(sorted(ends)):
                ax.text(1.03, 1.0 - 0.13 * (k + 1), fmt(v), transform=ax.transAxes, ha="left", va="top",
                        fontsize=7.0, color=col[arm], clip_on=False, zorder=7)
            ends = []
        elif a.holdout != "ends":
            rect_in = (left_in + i * (pw + gap_in), bot_in + leg_in, pw, a.panel_h)
            _draw_holdout(fig, ax, a.holdout, holdout_order(ends), col, dash, fmt, rect_in, (W, H))
            ends = []
        ends = sorted((np.log10(max(v, ylo)), arm, v) for v, arm in ends)
        span = np.log10(yhi) - np.log10(ylo); step = 0.105 * span
        ys = [e[0] for e in ends]
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < step: ys[k] = ys[k - 1] + step
        for (lv, arm, val), y in zip(ends, ys):
            ax.annotate(f"{val:.1f}" if val >= 100 else f"{val:.2f}" if val < 10 else f"{val:.1f}",
                        xy=(4000, 10 ** lv), xytext=(1.03, 10 ** y), textcoords=("axes fraction", "data"),
                        xycoords="data", fontsize=7.5, color=col[arm], ha="left", va="center", clip_on=False,
                        arrowprops=dict(arrowstyle="-", color=col[arm], lw=0.5, alpha=0.6, shrinkA=0, shrinkB=1))
        # ---- red subtitle: direct softplus's shoulder share at the end, median seed ----
        sh = np.median([float(d[f"{name}/direct/s{s}/shoulder"]) for s in d[f"{name}/direct/seeds"]
                        if f"{name}/direct/s{s}/shoulder" in d])
        ax.text(0.5, 1.015, f"{sh:.2f} on shoulder", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.0, color=col["direct"])
        # ---- failure counts, beside the marker that carries them ----
        for j, (kind, txt, c) in enumerate(notes):
            mk = [m for m in marks if m[0] == kind]
            if not mk:
                continue
            _, mx, my, _ = mk[0]
            off, ha, va = ((5, -6), "left", "top") if kind == "x" else ((-5, 6), "right", "bottom")
            ax.annotate(txt, xy=(mx, my), xytext=off, textcoords="offset points",
                        fontsize=5.8, color=c, ha=ha, va=va, clip_on=False, zorder=7,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.75))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=9, pad=9.5)
        ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
        ax.set_xlabel("iteration", fontsize=8.5, labelpad=0.5)
        if i == 0:
            ax.set_ylabel("training loss" if a.curve == "train" else "validation loss", fontsize=8.5, labelpad=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # ------------------------- the legend strip -------------------------
    sx0_in = left_in + n * pw + (n - 1) * gap_in + strip_gap_in
    sw_in = W - edge_in - sx0_in
    sy0_in, sh_in = edge_in, H - 2 * edge_in
    ax_s = fig.add_axes([sx0_in / W, sy0_in / H, sw_in / W, sh_in / H]); ax_s.set_axis_off()
    ax_s.set_xlim(0.0, sw_in); ax_s.set_ylim(0.0, sh_in)
    foot = [max(strip_handle_in, _text_box_rot(fig, t, STRIP_FS)[1]) for t, _, _, _ in strip_entries]
    sgap = (sh_in - sum(foot)) / len(foot)
    xh = strip_lw_in / 2; xt = strip_lw_in + strip_pad_in + strip_text_in / 2
    yc = sgap / 2
    for (label, colr, lw, ls), f in zip(strip_entries, foot):
        yc += f / 2
        ax_s.add_line(Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2], color=colr, lw=lw,
                             linestyle=ls, solid_capstyle="butt", dash_capstyle="butt", clip_on=False))
        ax_s.text(xt, yc, label, rotation=90, ha="center", va="center", fontsize=STRIP_FS, clip_on=False)
        yc += f / 2 + sgap

    # The curve is the validation loss and the end label the median final
    # test NLL, so print the gap between the two quantities.
    for _, name in COLS:
        for arm, _ in ARMS:
            g = [abs(float(d[f"{name}/{arm}/s{s}/test"]) - float(d[f"{name}/{arm}/s{s}/val"][-1]))
                 for s in d[f"{name}/{arm}/seeds"]
                 if np.isfinite(float(d[f"{name}/{arm}/s{s}/test"])) and abs(float(d[f"{name}/{arm}/s{s}/test"])) < 1e6]
            if g:
                print(f"  val-vs-test gap  {name:10s} {arm:9s} max {max(g):.3g} nats over {len(g)} seeds")

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"{a.out_dir}/" + ("tabular_train_loss" if a.curve == "train" else "tabular_val_loss")
    if a.holdout not in ("table", "ends"):   # the two default modes
        stem += f"_holdout_{a.holdout}"          # so no mode overwrites another
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to):
        shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
