"""Two panels on the FFHQ image regularizer, across step sizes and seeds.

The template is the paper's: grayscale inpainting at 256 px, 32 filters,
20 epochs (2,500 Adam steps), mean-matched negatives.

(a) Test PSNR at the reported (lowest-validation-loss) iterate against the
    step that iterate was reached at. One marker shape per construction:
    circle direct softplus, triangle PGD, square the lift. The published
    step size, 5e-4, carries the three large filled markers, each labeled
    in its own color, with the ten-to-ninety band of its seeds in PSNR.
    Every other step size on disk is a small open marker of the same shape,
    and only direct softplus at four times the published step is labeled.
    A cell that lost a seed to the constant-output solution is drawn as a
    cross. Markers are per-cell medians of PSNR and of step, taken
    separately over the seeds that did not collapse.

(b) The median constrained weight along training at the published step size,
    with a star at each seed's reported iterate, the shoulder edge
    (effective weight 0.0513, latent weight -2.944) dashed, and direct
    softplus at a larger step size as the dashed median.

Everything is read from the per-run ``metrics.json`` the driver writes
(``test_psnr_bestval_mean``, ``test_psnr_mean``, ``bestval_step``,
``history.w_median``, ``history.w_frac_zero``); no training, no model
loading. Cells are selected by the driver args, fan-in readout cells
excluded.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys
import numpy as np, matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

LR_PUB = 5e-4
COL = {"hypernet": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c"}
LABEL = {"hypernet": "lift", "direct": "direct softplus", "pgd": "PGD"}
SHOULDER_EDGE = 0.05131  # softplus(-2.944), the paper's screen threshold in effective-weight units
CONST_DB = 12.0          # anything at or below this is the constant-output solution (11.17 dB)
BAND_MIN_SEEDS = 5       # above this a cell is drawn as a median inside a 10-90 seed band
REF_HH, REF_CC = [64], [32, 64, 128]


def _arms():
    """Every completed arm at the paper's template, with its cell arguments."""
    out = []
    for path in sorted(glob.glob(f"{_REPO}/data/checkpoints/*meanmatch*/metrics.json")):
        try: rec = json.load(open(path))
        except (OSError, json.JSONDecodeError): continue
        a = rec.get("args") or {}
        if (a.get("n_filters"), a.get("n_epochs"), a.get("n_channels")) != (32, 20, 1): continue
        if int(a.get("mean_match_negatives", 0) or 0) != 1: continue
        if int(a.get("readout_fanin_scale", 0) or 0) != 0: continue
        d = os.path.dirname(path); h = os.path.basename(d)[-8:]
        for ap in sorted(glob.glob(d + "/*_seed*/metrics.json")):
            m = json.load(open(ap))
            if not m.get("history"): continue
            out.append(dict(hash=h, backend=m["backend"], seed=int(m["seed"]), lr=float(a["lr"]),
                            init=a.get("init_mode"),
                            ps=a.get("pool_scale"), ncond=int(a.get("n_cond", 0) or 0), bias=a.get("head_init_pos_bias"),
                            hh=a.get("hyper_hidden_sizes"), cc=a.get("conv_channels"), decay=float(a.get("emission_decay_frac", 0) or 0),
                            best=float(m["test_psnr_bestval_mean"]), final=float(m["test_psnr_mean"]), bstep=int(m["bestval_step"]),
                            hist=m["history"]))
    return out


# head_init_pos_bias is read only by HypernetBackend (see the lib's
# apply_ffhq_init docstring), so a baseline cell at -5.3 and one at -2.0 are
# the same experiment: both take the published U[0, 0.01] draw. The
# initialization-axis cells at "own" therefore carry 27 direct and 29 PGD
# seeds of the run the three-seed step-size sweep also holds, and excluding
# them left the baselines at three seeds against the lift's thirty. They are
# pooled here, preferring the larger family so one cell supplies the whole
# curve family. Every other differing argument in those cells (pool_scale,
# n_cond, readout_fanin_scale, save_emitter, backends, seeds) is either
# lift-only or run control.
_BASELINE_BIAS = (-5.3, -2.0)


def _select(arms, backend, lr=None, ps=None, pinned=None):
    """Arms of one construction; lift arms filtered to the paper's body (d 64, encoder 32/64/128, bias -5.3, n_cond 0)."""
    sel = []
    for r in arms:
        if r["backend"] != backend: continue
        if lr is not None and r["lr"] != lr: continue
        if backend == "hypernet":
            if r["hh"] != REF_HH or r["cc"] != REF_CC or r["ncond"] != 0 or r["decay"] != 0: continue
            p = float(r["ps"] or 0.0)
            if pinned is True and (p <= 0 or (ps is not None and p != ps)): continue
            if pinned is False and p > 0: continue
            if pinned is None and ps is not None and p != ps: continue
        elif r["bias"] not in _BASELINE_BIAS: continue
        if r.get("init") not in (None, "own"): continue
        sel.append(r)
    # one arm per seed per cell. For a baseline the same seed can sit in two
    # cells that differ only in inert arguments; -5.3 sorts first so the whole
    # family comes from the initialization-axis cells.
    seen, out = set(), []
    for r in sorted(sel, key=lambda r: (r["lr"], _BASELINE_BIAS.index(r["bias"]) if backend != "hypernet" and r["bias"] in _BASELINE_BIAS else 0, r["hash"], r["seed"])):
        k = (r["lr"], r.get("ps") if backend == "hypernet" else None, r["seed"])
        if k in seen: continue
        seen.add(k); out.append(r)
    return out


def _text_box_rot(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs, rotation=90); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def _rel(lr):
    """Step size relative to the published one, as a compact label."""
    f = lr / LR_PUB
    return rf"{f:g}$\times$"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=f"{_REPO}/figures")
    p.add_argument("--out_name", default="ffhq_resilience")
    p.add_argument("--mirror_to", default="", help="copy the PDF here as well (the paper repo's figures dir); off by default")
    p.add_argument("--fig_width", type=float, default=5.5); p.add_argument("--panel_h", type=float, default=1.02)   # the height the paper ships
    p.add_argument("--unpinned", action="store_true", help="add the unpinned lift's collapsed cells to (a)")
    p.add_argument("--ymin", type=float, default=28.40)
    p.add_argument("--direct_fast_lr", type=float, default=2e-3, help="draw direct softplus at this step size dashed in (b); 0 to omit")
    a = p.parse_args()
    apply_paper_style(); matplotlib.rcParams["savefig.bbox"] = "standard"; matplotlib.rcParams["axes.grid"] = False

    arms = _arms()
    series = {  # panel (a): the cells of each construction, grouped by step size below
        "direct": _select(arms, "direct"), "pgd": _select(arms, "pgd"),
        "lift1": [r for r in _select(arms, "hypernet", ps=1.0, pinned=True) if r["lr"] != LR_PUB], "lift0.5": _select(arms, "hypernet", ps=0.5, pinned=True),
    }
    lift1_pub = _select(arms, "hypernet", lr=LR_PUB, ps=1.0, pinned=True)   # one seed: reported in the table, not drawn as a median
    # the unpinned lift at both inits: bias -5.3 (the paper's) in its own dirs, bias -2 beside direct / PGD in theirs
    unpinned = _select(arms, "hypernet", pinned=False) if a.unpinned else []

    def by_lr(rs):
        g = {}
        for r in rs: g.setdefault(r["lr"], []).append(r)
        return dict(sorted(g.items()))

    # ---- report every plotted number ------------------------------------------------------------
    print("panel (a): test PSNR at the reported iterate @ its step (hash s<seed>); collapsed = final iterate at the constant output")
    for key, rs in list(series.items()) + [("lift1 (1 seed)", lift1_pub)] + ([("lift, no pin", unpinned)] if unpinned else []):
        for lr, cell in by_lr(rs).items():
            cell = sorted(cell, key=lambda r: r["seed"]); ok = [r for r in cell if r["best"] > CONST_DB]
            med_p, med_s = (float(np.median([r["best"] for r in ok])), float(np.median([r["bstep"] for r in ok]))) if ok else (float("nan"),) * 2
            allb = [r["best"] for r in cell]
            print(f"  {key:14s} lr={lr:<8g} ({lr / LR_PUB:g}x) n={len(cell)} plotted median (surviving seeds) {med_p:.2f} dB @ {med_s:.0f}  mean over all seeds {np.mean(allb):.2f} +- {np.std(allb, ddof=1) if len(cell) > 1 else 0:.2f}"
                  f"  seeds: " + "; ".join(f"{r['hash']} s{r['seed']} {r['best']:.2f}@{r['bstep']} (final {r['final']:.2f}{', collapsed' if r['final'] <= CONST_DB else ''}"
                                          + (f", zero frac end {r['hist']['w_frac_zero'][-1]:.3f} max {max(r['hist']['w_frac_zero']):.3f}" if r["backend"] == "pgd" else "") + ")" for r in cell))
    pinned_all = _select(arms, "hypernet", pinned=True) + [r for r in arms if r["backend"] == "hypernet" and (r["ps"] or 0) > 0 and r["ncond"] > 0 and r["hh"] == REF_HH and r["cc"] == REF_CC]
    print(f"  pinned lift arms on disk at this template (all pins, n_cond 0-4): {len(pinned_all)}, collapsed at the final iterate: {sum(r['final'] <= CONST_DB for r in pinned_all)}, at the reported iterate: {sum(r['best'] <= CONST_DB for r in pinned_all)}")

    # ---- layout -----------------------------------------------------------------------------------
    W = a.fig_width; STRIP_FS = 6.6
    strip_entries = [("lift", COL["hypernet"]), ("direct softplus", COL["direct"]), ("PGD", COL["pgd"])]
    strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.30, 0.30, 0.045, 0.04, 0.06
    fig = plt.figure(figsize=(W, 2.0)); strip_text_in = max(_text_box_rot(fig, t, STRIP_FS)[0] for t, _ in strip_entries); plt.close(fig)
    right_in = strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in
    left_in, gap_in, top_in, bot_in = 0.40, 0.60, 0.16, 0.30
    pw = (W - left_in - right_in - gap_in) / 2
    H = top_in + a.panel_h + bot_in
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left_in / W, bot_in / H, pw / W, a.panel_h / H])                   # (a)
    axb = fig.add_axes([(left_in + pw + gap_in) / W, bot_in / H, pw / W, a.panel_h / H])  # (b)
    for ax_ in (ax, axb):
        ax_.tick_params(labelsize=7.0, length=2.5, pad=1.5)
        for s_ in ("top", "right"): ax_.spines[s_].set_visible(False)

    # ---- (a) --------------------------------------------------------------------------------------
    # Two tiers and one shape per construction. The published step size is the
    # paper's comparison and carries the three big markers; every other step
    # size is context, drawn small and open, and is named only where it is part
    # of the argument. Color, never shape, separates the constructions.
    ymax = 30.05
    ax.set_xscale("log"); ax.set_xlim(170, 3400); ax.set_ylim(a.ymin, ymax)
    ax.axvline(2500, color="0.6", lw=0.6, linestyle=(0, (1.2, 1.6)), zorder=0.5)
    ax.text(2400, ymax - 0.015 * (ymax - a.ymin), "budget", fontsize=6.4, color="0.45", ha="right", va="top")
    SHAPE = {"direct": "o", "pgd": "^", "hypernet": "s"}

    def _stat(cell):
        ok = [r for r in cell if r["best"] > CONST_DB]
        if not ok: return None
        pv = np.array([r["best"] for r in ok], float); sv = np.array([r["bstep"] for r in ok], float)
        return dict(n=len(cell), ncoll=sum(r["final"] <= CONST_DB for r in cell), p=float(np.median(pv)), s=float(np.median(sv)),
                    plo=float(np.percentile(pv, 10)), phi=float(np.percentile(pv, 90)),
                    slo=float(np.percentile(sv, 10)), shi=float(np.percentile(sv, 90)))

    cells = {}
    for key, rs in series.items():
        con = "hypernet" if key.startswith("lift") else key
        for lr, cell in by_lr(rs).items():
            st = _stat(cell)
            if st is not None: cells[(con, lr)] = st
    print("  panel (a) drawn cells: " + "; ".join(
        f"{c} {lr / LR_PUB:g}x n={s['n']} {s['p']:.2f} dB @ {s['s']:.0f} [step 10-90 {s['slo']:.0f}-{s['shi']:.0f}, dB 10-90 {s['plo']:.2f}-{s['phi']:.2f}]"
        + (f" COLLAPSED {s['ncoll']}" if s["ncoll"] else "") for (c, lr), s in cells.items()))

    # step-size sweep
    for (con, lr), st in sorted(cells.items()):
        if lr == LR_PUB: continue
        c = COL[con]
        if st["ncoll"]:   # a seed of this cell ended at the constant output
            ax.plot([st["s"]], [st["p"]], linestyle="none", marker="x", ms=4.4, mew=1.1, color=c, zorder=4.5)
        else:
            ax.plot([st["s"]], [st["p"]], linestyle="none", marker=SHAPE[con], ms=3.5, mfc="white", mec=c, mew=0.9, zorder=4.5)

    # the published step size: big marker, the seed spread in PSNR, a label in
    # its own color so the eye never travels to the margin strip
    ANCH = {"hypernet": ("lift", (8, 3), "left", "center"),
            "direct": ("direct softplus", (-6, 0), "right", "center"),
            "pgd": ("PGD", (-7, 0), "right", "center")}
    for con in ("pgd", "direct", "hypernet"):
        st = cells[(con, LR_PUB)]; c = COL[con]
        ax.errorbar([st["s"]], [st["p"]], yerr=[[st["p"] - st["plo"]], [st["phi"] - st["p"]]], fmt="none",
                    ecolor=c, elinewidth=0.6, capsize=1.9, capthick=0.7, alpha=0.55, zorder=3.2)
        ax.plot([st["s"]], [st["p"]], linestyle="none", marker=SHAPE[con], ms=6.0, color=c, mec="white", mew=0.5, zorder=4)
        lab, off, ha, va = ANCH[con]
        ax.annotate(lab, xy=(st["s"], st["p"]), xytext=off, textcoords="offset points", fontsize=7.0, color=c,
                    ha=ha, va=va, zorder=5)

    # the one other step size the argument turns on, and the two cells that collapsed
    d4 = cells[("direct", 4 * LR_PUB)]
    ax.annotate(_rel(4 * LR_PUB) + " step", xy=(d4["s"], d4["p"]), xytext=(-4.5, 0), textcoords="offset points",
                fontsize=6.8, color=COL["direct"], ha="right", va="center", zorder=5)
    ax.text(3380, 28.72, "a seed collapsed to a constant" "\n" r"output, at 10$\times$ and 20$\times$ step",
            fontsize=6.6, color=COL["pgd"], ha="right", va="center", linespacing=0.95, zorder=5)
    if unpinned:   # --unpinned: the lift with no norm pin does collapse, and is drawn the same way
        for lr, cell in by_lr(unpinned).items():
            st = _stat(cell)
            if st is None or not st["ncoll"]: continue
            ax.plot([st["s"]], [st["p"]], linestyle="none", marker="x", ms=4.4, mew=1.1, color=COL["hypernet"], alpha=0.8, zorder=4.5)
        ax.text(182, 29.06, "no pinned lift cell collapsed," "\n" r"at half to twice the step",
                fontsize=6.6, color=COL["hypernet"], ha="left", va="center", linespacing=0.95, zorder=5)
    else:
        ax.text(182, 29.06, "no lift cell collapsed," "\n" r"at half to twice the step",
                fontsize=6.6, color=COL["hypernet"], ha="left", va="center", linespacing=0.95, zorder=5)

    ax.set_ylabel("test PSNR (dB)", fontsize=7.4, labelpad=2)
    ax.set_yticks([28.5, 29.0, 29.5, 30.0]); ax.set_yticklabels(["28.5", "29", "29.5", "30"])
    ax.set_xticks([200, 500, 1000, 2500]); ax.set_xticklabels(["200", "500", "1k", "2.5k"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("step of the reported iterate", fontsize=7.4, labelpad=1)

    # ---- (b) --------------------------------------------------------------------------------------
    print("panel (b): median constrained weight along training at lr 5e-4 (hash s<seed>): value at step 100, at the reported iterate, at 2500; latent-median slope over the first 500 steps")
    for key in ("direct", "pgd", "hypernet"):
        cell = _select(arms, key, lr=LR_PUB, ps=0.5 if key == "hypernet" else None, pinned=True if key == "hypernet" else None)
        c = COL[key]
        # With three seeds a line apiece reads; with thirty it does not, so a
        # cell above BAND_MIN_SEEDS is drawn as the house median-inside-band.
        band = len(cell) > BAND_MIN_SEEDS
        if band:
            x = np.asarray(cell[0]["hist"]["eval_steps"], float)
            Y = np.vstack([np.asarray(r["hist"]["w_median"], float) for r in cell])
            axb.fill_between(x, np.percentile(Y, 10, axis=0), np.percentile(Y, 90, axis=0),
                             color=c, alpha=0.16, lw=0, zorder=2.5)
            axb.plot(x, np.median(Y, axis=0), color=c, lw=1.3, alpha=0.95, zorder=3.2)
            bi = int(np.argmin(np.abs(x - float(np.median([r["bstep"] for r in cell])))))
            axb.plot([x[bi]], [np.median(Y, axis=0)[bi]], marker="*", ms=6.0, color=c,
                     mec="black", mew=0.35, linestyle="none", zorder=5)
        for r in sorted(cell, key=lambda r: r["seed"]):
            h = r["hist"]; x = np.asarray(h["eval_steps"], float); y = np.asarray(h["w_median"], float)
            if not band:
                axb.plot(x, y, color=c, lw=0.9, alpha=0.85, zorder=3)
                i = int(np.argmin(np.abs(x - r["bstep"])))
                axb.plot([x[i]], [y[i]], marker="*", ms=6.0, color=c, mec="black", mew=0.35, linestyle="none", zorder=5)
            i = int(np.argmin(np.abs(x - r["bstep"])))
            pre = np.asarray(h["w_median_pre_softplus_pos"], float)
            slope = (pre[4] - pre[0]) / (x[4] - x[0]) if key != "pgd" else float("nan")
            fz = np.asarray(h["w_frac_zero"], float)
            print(f"  {key:9s} {r['hash']} s{r['seed']}  w_median {y[0]:.4f} -> {y[i]:.4f} @ {int(x[i])} -> {y[-1]:.4f}  latent median {pre[0]:.2f} -> {pre[i]:.2f} -> {pre[-1]:.2f}"
                  f"  slope(100-500) {slope:+.2e}/step = {slope / LR_PUB:+.2f} lr  frac_zero end {fz[-1]:.3f} [{fz.min():.3f}, {fz.max():.3f}]  shoulder share end {h['w_frac_shoulder'][-1]:.3f}")
    axb.set_yscale("log"); axb.set_ylim(0.0042, 0.11); axb.set_xlim(0, 2500)
    axb.axhline(SHOULDER_EDGE, color="0.3", lw=0.7, linestyle=(0, (4, 2)), zorder=1)
    axb.text(60, SHOULDER_EDGE / 1.06, "shoulder edge", fontsize=7.0, color="0.3", ha="left", va="top")
    axb.set_xticks([0, 1000, 2000, 2500]); axb.set_xticklabels(["0", "1k", "2k", "2.5k"])
    axb.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1]); axb.set_yticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"])
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axb.set_xlabel("training step (published step size)", fontsize=7.4, labelpad=1)
    axb.set_ylabel("median constrained weight", fontsize=7.4, labelpad=2)
    axb.plot([0.035], [0.955], transform=axb.transAxes, marker="*", ms=6.0, color="0.55", mec="black", mew=0.35, linestyle="none", clip_on=False)
    axb.text(0.075, 0.955, "reported iterate", transform=axb.transAxes, fontsize=7.0, color="0.25", ha="left", va="center")
    if a.direct_fast_lr > 0:   # direct softplus at the step size where it climbs as fast as the lift: median over seeds, dashed
        cell = _select(arms, "direct", lr=a.direct_fast_lr)
        if cell:
            x = np.asarray(cell[0]["hist"]["eval_steps"], float); Y = np.median([np.asarray(r["hist"]["w_median"], float) for r in cell], axis=0)
            axb.plot(x, Y, color=COL["direct"], lw=0.8, linestyle=(0, (3, 1.5)), alpha=0.8, zorder=2.8)
            axb.text(x[-1] - 30, Y[-1] / 1.09, rf"direct, {a.direct_fast_lr / LR_PUB:g}$\times$ step", fontsize=7.0, color=COL["direct"], ha="right", va="top")
            pre = np.median([np.asarray(r["hist"]["w_median_pre_softplus_pos"], float) for r in cell], axis=0)
            print(f"  direct at {a.direct_fast_lr:g} (dashed, median of {len(cell)} seeds): latent median {pre[0]:.2f} -> {pre[-1]:.2f}, slope(100-500) {(pre[4] - pre[0]) / (x[4] - x[0]):+.2e}/step = {(pre[4] - pre[0]) / (x[4] - x[0]) / a.direct_fast_lr:+.2f} of its own lr")
    # (b) sits left of its own y-axis label, not on it: panel (b)'s rotated
    # label reaches the top of its axes, which (a)'s does not because (a) is
    # the taller pair of axes.
    for ax_, lab, dx in ((ax, "(a)", -0.19), (axb, "(b)", -0.285)):
        ax_.text(dx, 1.0, lab, transform=ax_.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")

    # ---- legend strip (house convention: rotated, right margin) -------------------------------------
    sx0 = W - right_in + strip_gap_in; sw = W - edge_in - sx0
    ax_s = fig.add_axes([sx0 / W, edge_in / H, sw / W, (H - 2 * edge_in) / H]); ax_s.set_axis_off(); ax_s.set_xlim(0, sw); ax_s.set_ylim(0, H - 2 * edge_in)
    foot = [max(strip_handle_in, _text_box_rot(fig, t, STRIP_FS)[1]) for t, _ in strip_entries]; sgap = (H - 2 * edge_in - sum(foot)) / len(foot)
    xh = strip_lw_in / 2; xt = strip_lw_in + strip_pad_in + strip_text_in / 2; yc = sgap / 2
    for (label, c), f in zip(strip_entries, foot):
        yc += f / 2
        ax_s.add_line(Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2], color=c, lw=1.2, clip_on=False))
        ax_s.text(xt, yc, label, rotation=90, ha="center", va="center", fontsize=STRIP_FS, clip_on=False); yc += f / 2 + sgap

    os.makedirs(a.out_dir, exist_ok=True); stem = f"{a.out_dir}/{a.out_name}"
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to): shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
