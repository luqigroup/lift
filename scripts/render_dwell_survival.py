r"""``dwell_survival`` -- the dwell of a visit to the stuck set.

Two panels. Panel (a) is the exit-rate dot plot of ``render_dwell_texture``,
drawn by that module's own ``draw_panel_a`` so the two figures cannot drift.
Panel (b) is the Kaplan-Meier estimate of how long a visit to the stuck set
lasts, one curve per construction, on HEPMASS seed 0.

Panel (b) plots ``1 - S(k)``, the fraction of visits ended by ``k``, on
logarithmic axes; ``S(k)`` itself puts direct softplus and the lift on nearly
the same line, and ``--plot survival`` draws it for comparison. On log axes the
slope is the tail shape: a constant hazard, the geometric dwell of a projected
coordinate, gives ``1 - S`` growing like ``k`` at small values, and a
decreasing hazard bends it below that line.

A visit still running at the last snapshot is right-censored rather than
counted as an exit, which is why Kaplan-Meier is used; censoring dominates
direct softplus, so its level sits near the resolution of the instrument.
Episodes already in the set at the first snapshot are left-truncated, their
start never seen, and are dropped by default; ``--include_initial`` keeps them.

Everything read here is cached and nothing is trained.

Usage::

    python scripts/render_dwell_survival.py
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np, matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
from lift.utils.paper_style import apply_paper_style  # noqa: E402
from render_dwell_texture import (  # noqa: E402
    ARMS, COL, DASH, LABEL, RASTERS, RASTER_SET, _cells, _masks, _text_box_rot,
    draw_panel_a, panel_a_data,
)

ITERS_PER_INTERVAL = 200          # HEPMASS: the snapshot grid steps by 200


def durations(M: np.ndarray, include_initial: bool = False) -> tuple:
    """Episode lengths in one construction's stuck set.

    Returns the observed episode lengths and the censored ones, separately.

    An episode is a maximal run of consecutive snapshots one coordinate spends
    in the set. Time is counted in snapshot intervals from the episode's first
    snapshot, so a coordinate in the set at ``s`` and out of it at ``s+1`` ends
    at ``k = 1``. An episode still running at the last snapshot is censored at
    ``L - 1``: it was seen to survive that many intervals and the next one was
    never observed.
    """
    T = M.shape[1]
    P = np.zeros((M.shape[0], T + 2), dtype=bool); P[:, 1:T + 1] = M
    s = np.argwhere(P[:, 1:] & ~P[:, :-1])[:, 1]
    e = np.argwhere(P[:, :-1] & ~P[:, 1:])[:, 1] - 1
    L = e - s + 1
    cens = e == T - 1
    if not include_initial:                 # drop the left-truncated episodes
        k = s > 0
        L, cens = L[k], cens[k]
    return L[~cens].astype(np.int64), (L[cens] - 1).astype(np.int64)


def kaplan_meier(t_event: np.ndarray, t_censor: np.ndarray, kmax: int) -> tuple:
    """``S(k)`` for ``k = 0..kmax``, plus the events and censorings per step.

    At risk at ``k`` is every episode whose event time or censoring time is at
    least ``k``. An episode censored at 0 was seen for no complete interval and
    is at risk nowhere, so it is removed before the first step.
    """
    ev = np.bincount(t_event, minlength=kmax + 2).astype(np.int64)
    ce = np.bincount(np.clip(t_censor, 0, None), minlength=kmax + 2).astype(np.int64)
    n = int(ev.sum() + ce.sum()) - int(ce[0])
    S, out, risk = 1.0, [1.0], []
    for k in range(1, kmax + 1):
        risk.append(n)
        if n > 0:
            S *= 1.0 - ev[k] / n
        out.append(S)
        n -= int(ev[k]) + int(ce[k])
    return np.asarray(out), ev, ce, np.asarray(risk)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default=os.path.join(_REPO, "docs/analysis/dwell_time.json"))
    p.add_argument("--out_dir", default=os.path.join(_REPO, "figures"))
    p.add_argument("--mask_cache", default=os.path.join(_REPO, "data/analysis/dwell_texture_masks.npz"))
    p.add_argument("--refresh_masks", action="store_true")
    p.add_argument("--mirror_to", default="")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--fig_height", type=float, default=0.0,
                   help="0 = the layout's own default: 1.58 in, or 1.70 in with --legacy_layout")
    p.add_argument("--legacy_layout", action="store_true",
                   help="the earlier layout: rotated right-margin legend strip, numeric "
                        "end labels, the inset sentence, the dagger and the diverged-seed rings")
    p.add_argument("--inset_numbers", action="store_true",
                   help="keep the exits-observed sentence inside panel (b) without the old layout")
    p.add_argument("--plot", choices=["ended", "survival"], default="ended",
                   help="'ended' plots 1 - S(k), which separates the three; "
                        "'survival' plots S(k), on which direct and the lift coincide")
    p.add_argument("--include_initial", action="store_true",
                   help="keep episodes already in the set at the first snapshot (left-truncated)")
    p.add_argument("--keep_duplicate_last", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    apply_paper_style()
    matplotlib.rcParams["savefig.bbox"] = "standard"; matplotlib.rcParams["axes.grid"] = False

    # ---- panel (a), the shared dot plot -------------------------------------
    blob = json.load(open(a.json))
    print(f"panel (a) source: {a.json}")
    plot = panel_a_data(_cells(blob))

    # ---- panel (b), Kaplan-Meier on HEPMASS seed 0 --------------------------
    mk = _masks(a)
    iters = np.asarray(mk["iters"], int)
    keep = iters.size
    if not a.keep_duplicate_last and keep >= 2 and all(
            bool((mk[arm][:, -1] == mk[arm][:, -2]).all()) for arm in RASTERS):
        keep -= 1
        print(f"\n  snapshot {iters.size - 1} (iteration {iters[-1]}) is bit-identical to "
              f"snapshot {iters.size - 2} (iteration {iters[-2]}) in all three arms and is "
              f"dropped; {keep} snapshots, {keep - 1} intervals")
    iters = iters[:keep]
    K = keep - 1
    S, info = {}, {}
    print(f"\npanel (b) source: HEPMASS seed 0, {keep} snapshots, "
          f"{K} intervals of {ITERS_PER_INTERVAL} iterations, "
          f"left-truncated episodes {'kept' if a.include_initial else 'dropped'}")
    for arm in RASTERS:
        M = np.asarray(mk[arm])[:, :keep]
        te, tc = durations(M, a.include_initial)
        S[arm], ev, ce, risk = kaplan_meier(te, tc, K)
        n_ep = int(te.size + tc.size)
        haz = [1.0 - S[arm][k] / S[arm][k - 1] if S[arm][k - 1] > 0 else float("nan")
               for k in range(1, K + 1)]
        info[arm] = dict(n_ep=n_ep, n_ev=int(te.size), cens=tc.size / n_ep, haz=haz)
        print(f"  {LABEL[arm]:15s} ({RASTER_SET[arm]}) episodes {n_ep:>9,d}  "
              f"observed exits {int(te.size):>9,d}  censored {tc.size / n_ep:.5f}  "
              f"at risk at k=1 {risk[0]:>9,d}")
        print(f"    S(k)   " + " ".join(f"k{k}={S[arm][k]:.5f}" for k in (1, 2, 3, 5, 10, K)))
        print(f"    1-S(k) " + " ".join(f"k{k}={1 - S[arm][k]:.4e}" for k in (1, 2, 3, 5, 10, K)))
        print(f"    hazard per interval, k=1..8: "
              + " ".join(f"{v:.2e}" for v in haz[:8]))
        # A memoryless visit has a flat hazard; a decreasing hazard means the
        # exits concentrate just after entry, and the average rate is then not
        # what the visit does later. The log-log slope is quoted only where 1-S
        # is far from saturation, since a bounded quantity flattens for reasons
        # unrelated to the tail.
        h = np.asarray(haz, float)
        early = h[0]; late = np.nanmean(h[K // 2:])
        info[arm].update(h1=early, h_late=late,
                         ratio=(early / late) if late > 0 else float("inf"))
        y = np.asarray([1 - S[arm][j] for j in range(1, K + 1)])
        m = (y > 0) & (y < 0.2)
        if m.sum() >= 3:
            sl = np.polyfit(np.log(np.arange(1, K + 1)[m]), np.log(y[m]), 1)[0]
            info[arm]["slope"] = sl
            print(f"    log-log slope of 1-S where 1-S < 0.2 ({int(m.sum())} points): "
                  f"{sl:.3f}; a memoryless visit gives 1.0")
        # Below a few dozen observed exits the shape is not estimable.
        if int(te.size) < 30:
            verdict = f"NOT ESTIMABLE, only {int(te.size)} observed exits"
        elif info[arm]["ratio"] < 1.6:
            verdict = "FLAT hazard, the visit is memoryless"
        else:
            verdict = "DECREASING hazard, the exits sit just after entry"
        print(f"    hazard interval 1 {early:.3e}, mean over k>{K // 2} {late:.3e}, "
              f"ratio {info[arm]['ratio']:.1f} -> {verdict}")

    # ---- layout -------------------------------------------------------------
    W = a.fig_width
    H = a.fig_height if a.fig_height > 0 else (1.70 if a.legacy_layout else 1.58)
    strip_entries = [(LABEL["lift"], COL["lift"], DASH["lift"]),
                     (LABEL["direct"], COL["direct"], DASH["direct"]),
                     (LABEL["pgd"], COL["pgd"], DASH["pgd"])]
    STRIP_FS = 6.6; strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.34, 0.28, 0.045, 0.04, 0.05
    LAB_FS, TICK_FS, AX_FS, TAG_FS = 6.6, 6.8, 7.6, 7.0
    probe = plt.figure(figsize=(W, H))
    strip_text_in = max(_text_box_rot(probe, t, STRIP_FS)[0] for t, _, _ in strip_entries)
    from render_dwell_texture import ROWS, DAGGER, _text_w
    lab_w = max(_text_w(probe, lab + (DAGGER if (dag and a.legacy_layout) else ""), LAB_FS)
                for lab, _, _, dag in ROWS)
    plt.close(probe)
    if a.legacy_layout:
        right_in = strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in
        left_in, pa_w, gap_ab = lab_w + 0.05, 1.16, 0.52
        bot_in, top_in = 0.34, 0.13
    else:
        right_in = edge_in + 0.03
        left_in, pa_w, gap_ab = lab_w + 0.05, 1.78, 0.52
        bot_in, top_in = 0.50, 0.30      # hint row under (a); legend band on top
    sv_w = W - right_in - left_in - pa_w - gap_ab

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left_in / W, bot_in / H, pa_w / W, (H - bot_in - top_in) / H])
    if a.legacy_layout:
        draw_panel_a(ax, plot, LAB_FS, TICK_FS, AX_FS, TAG_FS)
    else:
        draw_panel_a(ax, plot, LAB_FS, TICK_FS, AX_FS, TAG_FS, dagger=False,
                     show_diverged=False, range_lw=1.3, range_alpha=0.45, ms=3.4,
                     xlabel="exit rate per visit",
                     hint=("rarely leaves", "leaves at once"))

    x0 = left_in + pa_w + gap_ab
    axb = fig.add_axes([x0 / W, bot_in / H, sv_w / W, (H - bot_in - top_in) / H])
    k = np.arange(1, K + 1)
    ended = a.plot == "ended"
    lo = 1e-5
    for arm in RASTERS:
        y = np.asarray([(1 - S[arm][j]) if ended else S[arm][j] for j in k])
        axb.plot(k, np.clip(y, lo * 0.6, None), color=COL[arm], lw=1.2, linestyle=DASH[arm],
                 drawstyle="steps-post", zorder={"lift": 3.0, "direct": 3.6, "pgd": 4.0}[arm])
        if a.legacy_layout:
            axb.annotate(f"{y[-1]:.1e}" if y[-1] < 0.01 else f"{y[-1]:.3f}",
                         xy=(k[-1], y[-1]), xytext=(1.02, y[-1]),
                         textcoords=("axes fraction", "data"), xycoords="data",
                         fontsize=6.6, color=COL[arm], ha="left", va="center", clip_on=False)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlim(0.9, K * 1.12); axb.set_ylim(lo, 1.8 if a.legacy_layout else 2.2)
    axb.set_xticks([1, 2, 5, 10, 20]); axb.set_xticklabels(["1", "2", "5", "10", "20"], fontsize=TICK_FS)
    if a.legacy_layout:
        axb.set_yticks([1e-5, 1e-3, 1e-1])
        axb.set_yticklabels([r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"], fontsize=TICK_FS)
    else:
        # The tick at 1 is what shows that PGD ends every visit.
        axb.set_yticks([1e-5, 1e-3, 1e-1, 1e0])
        axb.set_yticklabels([r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$", r"$1$"], fontsize=TICK_FS)
    axb.tick_params(length=2.2, pad=1.4); axb.minorticks_off()
    axb.set_xlabel(f"snapshot intervals of {ITERS_PER_INTERVAL} iterations" if a.legacy_layout
                   else f"snapshot intervals since a visit began\n({ITERS_PER_INTERVAL} iterations each)",
                   fontsize=AX_FS, labelpad=1.0)
    axb.set_ylabel(("fraction of visits ended" if ended else "fraction still in the set")
                   if a.legacy_layout else
                   ("fraction of visits ended" if ended else "fraction on the shoulder"),
                   fontsize=AX_FS, labelpad=2.0)
    for s_ in ("top", "right"):
        axb.spines[s_].set_visible(False)
    axb.text(0.0, 1.008, "(b)", transform=axb.transAxes, ha="left", va="bottom",
             fontsize=TAG_FS, color="0.35")
    if a.legacy_layout or a.inset_numbers:
        axb.text(0.03, 0.34,
                 f"HEPMASS seed 0; {info['direct']['n_ev']} exits observed\n"
                 f"of {info['direct']['n_ep']:,} direct-softplus visits",
                 transform=axb.transAxes, ha="left", va="center", fontsize=5.6, color="0.35")

    if a.legacy_layout:
        sx0 = W - right_in + strip_gap_in; sw = W - edge_in - sx0
        ax_s = fig.add_axes([sx0 / W, bot_in / H, sw / W, (H - bot_in - edge_in) / H])
        ax_s.set_axis_off(); ax_s.set_xlim(0, sw); ax_s.set_ylim(0, H - bot_in - edge_in)
        foot = [max(strip_handle_in, _text_box_rot(fig, t, STRIP_FS)[1]) for t, _, _ in strip_entries]
        sgap = (H - bot_in - edge_in - sum(foot)) / len(foot)
        xh = strip_lw_in / 2; xt = strip_lw_in + strip_pad_in + strip_text_in / 2; yc = sgap / 2
        for (label, c, ls), f in zip(strip_entries, foot):
            yc += f / 2
            ax_s.add_line(Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2],
                                 color=c, lw=1.2, linestyle=ls, clip_on=False))
            ax_s.add_line(Line2D([xh], [yc], color=c, marker="o", markersize=2.9, ls="none",
                                 mfc=c, mec=c, clip_on=False))
            ax_s.text(xt, yc, label, rotation=90, ha="center", va="center", fontsize=STRIP_FS, clip_on=False)
            yc += f / 2 + sgap
    else:
        # One horizontal row, ordered as the constructions stand on panel (a)'s
        # axis. Each handle is the dot of (a) on the line of (b), so the two
        # panels read as the same three things.
        hl = [Line2D([], [], color=COL[arm], lw=1.2, linestyle=DASH[arm], marker="o",
                     ms=3.4, mfc=COL[arm], mec=COL[arm], label=LABEL[arm])
              for arm in ("direct", "lift", "pgd")]
        hl.append(Line2D([], [], color="0.45", lw=0, marker="o", ms=3.4, mfc="none",
                         mec="0.45", mew=0.8, label="open dot: the set is almost never entered"))
        leg = fig.legend(handles=hl, loc="upper left",
                         bbox_to_anchor=(left_in / W, 1.0 - 0.012), frameon=False, ncol=4,
                         fontsize=STRIP_FS, handlelength=1.9, handletextpad=0.4,
                         columnspacing=1.1, borderaxespad=0.0, borderpad=0.0)
        for t_, hnd in zip(leg.get_texts(), ("direct", "lift", "pgd", None)):
            if hnd is not None:
                t_.set_color(COL[hnd])
            else:
                t_.set_color("0.45")

    os.makedirs(a.out_dir, exist_ok=True); stem = os.path.join(a.out_dir, "dwell_survival")
    fig.savefig(stem + ".pdf", dpi=a.dpi); fig.savefig(stem + ".png", dpi=a.dpi); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to) and not os.path.samefile(a.mirror_to, a.out_dir):
        shutil.copy2(stem + ".pdf", a.mirror_to)
    print(f"\nwrote {stem}.pdf and .png ({W:.2f} x {H:.2f} in at {a.dpi} dpi)")


if __name__ == "__main__":
    main()
