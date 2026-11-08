"""Panels on the FFHQ image regularizer, across step sizes and seeds.

The template is the paper's: grayscale inpainting at 256 px, 32 filters,
20 epochs (2,500 Adam steps), mean-matched negatives.

(a) With ``--panel_a seeds`` (the default): the iteration of the reported
    (lowest-validation-loss) model at the published step size, 5e-4, as a
    box over the seeds with whiskers at the tenth and ninetieth percentiles
    and the median as a bar. With ``--panel_a sweep``: test PSNR at the
    reported iterate against the step that iterate was reached at. One
    marker shape per construction: circle direct softplus, triangle PGD,
    square the lift. The published step size carries the three large filled
    markers, each labeled in its own color, with the ten-to-ninety band of
    its seeds in PSNR. Every other step size on disk is a small open marker
    of the same shape, and only direct softplus at four times the published
    step is labeled. A cell that lost a seed to the constant-output solution
    is drawn as a cross. Markers are per-cell medians of PSNR and of step,
    taken separately over the seeds that did not collapse.

(b) The median constrained weight along training at the published step size,
    the shoulder edge (effective weight 0.0513, latent weight -2.944) dashed,
    and, with ``--direct_fast_lr``, direct softplus at that step size as the
    dashed median.

(c) With ``--panel_c``: the training objective along training at the
    published step size, the median over seeds inside the ten-to-ninety band,
    direct softplus's median end share on the shoulder above, and the median
    test PSNR at the reported iterate over the same seeds, framed.

``--sweep_only`` writes the step-size sweep of (a) alone, as
``ffhq_stepsizes``.

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
    p.add_argument("--panel_a", choices=["seeds", "sweep"], default="seeds",
                   help="seeds: the iteration of the reported model over the seeds at the published step size, "
                        "median as a bar; sweep: test PSNR against that iteration over step sizes")
    p.add_argument("--sweep_only", action="store_true",
                   help="write the step-size sweep alone, one panel (out_name ffhq_stepsizes)")
    p.add_argument("--direct_fast_lr", type=float, default=0.0, help="draw direct softplus at this step size dashed in (b); 0 to omit")
    p.add_argument("--panel_c", action="store_true",
                   help="add (c), the training objective along training, the median over seeds inside the band of its "
                        "seeds, with direct softplus's end share on the shoulder and the median test PSNR framed")
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

    def draw_sweep(ax):
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

        for (con, lr), st in sorted(cells.items()):
            if lr == LR_PUB: continue
            c = COL[con]
            if st["ncoll"]:
                ax.plot([st["s"]], [st["p"]], linestyle="none", marker="x", ms=4.4, mew=1.1, color=c, zorder=4.5)
            else:
                ax.plot([st["s"]], [st["p"]], linestyle="none", marker=SHAPE[con], ms=3.5, mfc="white", mec=c, mew=0.9, zorder=4.5)

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

        d4 = cells[("direct", 4 * LR_PUB)]
        ax.annotate(_rel(4 * LR_PUB) + " step", xy=(d4["s"], d4["p"]), xytext=(-4.5, 0), textcoords="offset points",
                    fontsize=6.8, color=COL["direct"], ha="right", va="center", zorder=5)
        ax.text(3380, 28.72, "a seed collapsed to a constant" "\n" r"output, at 10$\times$ and 20$\times$ step",
                fontsize=6.6, color=COL["pgd"], ha="right", va="center", linespacing=0.95, zorder=5)
        if unpinned:
            for lr, cell in by_lr(unpinned).items():
                st = _stat(cell)
                if st is None or not st["ncoll"]: continue
                ax.plot([st["s"]], [st["p"]], linestyle="none", marker="x", ms=4.4, mew=1.1, color=COL["hypernet"], alpha=0.8, zorder=4.5)
            ax.text(182, 29.06, "no pinned lift run collapsed," "\n" r"at half to twice the step",
                    fontsize=6.6, color=COL["hypernet"], ha="left", va="center", linespacing=0.95, zorder=5)
        else:
            ax.text(182, 29.06, "no lift run collapsed," "\n" r"at half to twice the step",
                    fontsize=6.6, color=COL["hypernet"], ha="left", va="center", linespacing=0.95, zorder=5)

        ax.set_ylabel("test PSNR (dB)", fontsize=7.4, labelpad=2)
        ax.set_yticks([28.5, 29.0, 29.5, 30.0]); ax.set_yticklabels(["28.5", "29", "29.5", "30"])
        ax.set_xticks([200, 500, 1000, 2500]); ax.set_xticklabels(["200", "500", "1k", "2.5k"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("iteration of best validation PSNR", fontsize=7.4, labelpad=1)

    def draw_seeds(ax):
        rows = [("hypernet", 2.0), ("direct", 1.0), ("pgd", 0.0)]
        for con, y0 in rows:
            cell = _select(arms, con, lr=LR_PUB, ps=0.5 if con == "hypernet" else None,
                           pinned=True if con == "hypernet" else None)
            ok = sorted([r for r in cell if r["best"] > CONST_DB], key=lambda r: r["seed"])
            xs = np.array([r["bstep"] for r in ok], float)
            med = float(np.median(xs))
            bp = ax.boxplot([xs], positions=[y0], vert=False, widths=0.5, whis=(10, 90), showfliers=False,
                            patch_artist=True, manage_ticks=False, zorder=3)
            for box in bp["boxes"]: box.set(facecolor=COL[con], alpha=0.25, edgecolor=COL[con], linewidth=0.9)
            for el in bp["whiskers"] + bp["caps"]: el.set(color=COL[con], linewidth=0.9)
            for el in bp["medians"]: el.set(color=COL[con], linewidth=2.0)
            for el in bp["boxes"] + bp["whiskers"] + bp["caps"] + bp["medians"]: el.set_clip_on(False)
            print(f"  panel (a) seeds: {con:9s} n={len(xs)} (of {len(cell)}) median iteration {med:.0f}, range {xs.min():.0f}-{xs.max():.0f}")
        ax.set_xlim(0, 2500); ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([2.0, 1.0, 0.0]); ax.set_yticklabels(["lift", "direct\nsoftplus", "PGD"])
        for lab, con in zip(ax.get_yticklabels(), ("hypernet", "direct", "pgd")): lab.set_color(COL[con])
        ax.tick_params(axis="y", length=0, labelsize=7.0)
        ax.set_xticks([0, 1000, 2000, 2500]); ax.set_xticklabels(["0", "1k", "2k", "2.5k"])
        ax.set_xlabel("iteration of best validation PSNR", fontsize=7.4, labelpad=1)

    if a.sweep_only:
        W = 3.3; H = 0.16 + a.panel_h + 0.30
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.45 / W, 0.30 / H, (W - 0.55) / W, a.panel_h / H])
        ax.tick_params(labelsize=7.0, length=2.5, pad=1.5)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        draw_sweep(ax)
        os.makedirs(a.out_dir, exist_ok=True); stem = f"{a.out_dir}/{a.out_name if a.out_name != 'ffhq_resilience' else 'ffhq_stepsizes'}"
        fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
        print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in)")
        return

    # ---- layout -----------------------------------------------------------------------------------
    W = a.fig_width; STRIP_FS = 6.6
    strip_entries = [("lift", COL["hypernet"]), ("direct softplus", COL["direct"]), ("PGD", COL["pgd"])]
    strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.30, 0.30, 0.045, 0.04, 0.06
    fig = plt.figure(figsize=(W, 2.0)); strip_text_in = max(_text_box_rot(fig, t, STRIP_FS)[0] for t, _ in strip_entries); plt.close(fig)
    right_in = (strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in) if a.panel_a == "sweep" else 0.10
    left_in, gap_in, top_in, bot_in = (0.62 if a.panel_a == "seeds" else 0.40), 0.60, 0.16, 0.30
    n_pan = 3 if a.panel_c else 2
    if a.panel_c: gap_in = 0.55
    pw = (W - left_in - right_in - (n_pan - 1) * gap_in) / n_pan
    H = top_in + a.panel_h + bot_in
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left_in / W, bot_in / H, pw / W, a.panel_h / H])                   # (a)
    axb = fig.add_axes([(left_in + pw + gap_in) / W, bot_in / H, pw / W, a.panel_h / H])  # (b)
    axc = fig.add_axes([(left_in + 2 * (pw + gap_in)) / W, bot_in / H, pw / W, a.panel_h / H]) if a.panel_c else None
    for ax_ in (ax, axb) + ((axc,) if axc is not None else ()):
        ax_.tick_params(labelsize=7.0, length=2.5, pad=1.5)
        for s_ in ("top", "right"): ax_.spines[s_].set_visible(False)

    # ---- (a) --------------------------------------------------------------------------------------
    (draw_seeds if a.panel_a == "seeds" else draw_sweep)(ax)

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
        for r in sorted(cell, key=lambda r: r["seed"]):
            h = r["hist"]; x = np.asarray(h["eval_steps"], float); y = np.asarray(h["w_median"], float)
            if not band:
                axb.plot(x, y, color=c, lw=0.9, alpha=0.85, zorder=3)
            i = int(np.argmin(np.abs(x - r["bstep"])))
            pre = np.asarray(h["w_median_pre_softplus_pos"], float)
            slope = (pre[4] - pre[0]) / (x[4] - x[0]) if key != "pgd" else float("nan")
            fz = np.asarray(h["w_frac_zero"], float)
            print(f"  {key:9s} {r['hash']} s{r['seed']}  w_median {y[0]:.4f} -> {y[i]:.4f} @ {int(x[i])} -> {y[-1]:.4f}  latent median {pre[0]:.2f} -> {pre[i]:.2f} -> {pre[-1]:.2f}"
                  f"  slope(100-500) {slope:+.2e}/step = {slope / LR_PUB:+.2f} lr  frac_zero end {fz[-1]:.3f} [{fz.min():.3f}, {fz.max():.3f}]  shoulder share end {h['w_frac_shoulder'][-1]:.3f}")
    axb.set_yscale("log"); axb.set_ylim(0.0042, 0.11); axb.set_xlim(0, 2500)
    axb.axhline(SHOULDER_EDGE, color="0.3", lw=0.7, linestyle=(0, (4, 2)), zorder=1)
    axb.text(60, SHOULDER_EDGE / 1.06, "shoulder edge", fontsize=7.0, color="0.1", ha="left", va="top")
    if a.panel_a == "seeds":
        for key, xv, yv, txt in (("pgd", 2450, 0.074, "PGD"), ("hypernet", 1700, 0.0262, "lift"),
                                 ("direct", 2450, 0.0185 if axc is None else 0.0056, "direct softplus")):
            axb.text(xv, yv, txt, fontsize=7.0, color=COL[key], ha="right", va="bottom")
    axb.set_xticks([0, 1000, 2000, 2500]); axb.set_xticklabels(["0", "1k", "2k", "2.5k"])
    axb.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1]); axb.set_yticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"])
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axb.set_xlabel("iteration", fontsize=7.4, labelpad=1)
    axb.set_ylabel("median constrained weight", fontsize=7.4, labelpad=2)

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
    if axc is not None:
        sys.path.insert(0, _HERE)
        from render_tabular_val_loss_multiseed import _draw_holdout, holdout_order
        DASH = {"hypernet": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
        ZO = {"hypernet": 3.0, "direct": 3.6, "pgd": 4.0}
        ends = []
        for key in ("direct", "pgd", "hypernet"):
            cell = _select(arms, key, lr=LR_PUB, ps=0.5 if key == "hypernet" else None, pinned=True if key == "hypernet" else None)
            x = np.asarray(cell[0]["hist"]["steps"], float)
            Y = np.vstack([np.asarray(r["hist"]["diff_loss"], float) for r in cell])
            axc.fill_between(x, np.percentile(Y, 10, axis=0), np.percentile(Y, 90, axis=0), color=COL[key], alpha=0.14, lw=0, zorder=ZO[key] - 0.5)
            axc.plot(x, np.median(Y, axis=0), color=COL[key], lw=1.2, linestyle=DASH[key], zorder=ZO[key])
            m_, n_ = float(np.median([r["best"] for r in cell])), len(cell); ends.append((m_, key))
            print(f"  panel (c): {key:9s} n={len(cell)} objective median {np.median(Y[:, 0]):.1f} -> {np.median(Y[:, -1]):.1f}; framed test PSNR median {m_:.2f} over {n_} seeds")
        sh = float(np.median([r["hist"]["w_frac_shoulder"][-1] for r in _select(arms, "direct", lr=LR_PUB)]))
        axc.text(0.5, 1.015, f"{sh:.2f} on shoulder", transform=axc.transAxes, ha="center", va="bottom", fontsize=7.0, color=COL["direct"])
        axc.set_xlim(0, 2500); axc.set_xticks([0, 1000, 2000, 2500]); axc.set_xticklabels(["0", "1k", "2k", "2.5k"])
        lo, hi = axc.get_ylim(); axc.set_ylim(lo, hi + 0.85 * (hi - lo))
        axc.set_yticks([t for t in axc.get_yticks() if lo <= t <= hi])
        axc.set_xlabel("iteration", fontsize=7.4, labelpad=1); axc.set_ylabel("training objective", fontsize=7.4, labelpad=2)
        x_c = left_in + 2 * (pw + gap_in)
        _draw_holdout(fig, axc, "table", holdout_order(ends), COL, DASH, lambda v: f"{v:.1f} dB",
                      (x_c, bot_in, pw, a.panel_h), (W, H), inside=True)
        axc.get_legend().set_title("test PSNR (median)"); axc.get_legend().get_title().set_color("0.15")
    for ax_, lab, dx in ((ax, "(a)", -0.40 / pw), (axb, "(b)", -0.60 / pw)) + (((axc, "(c)", -0.55 / pw),) if axc is not None else ()):
        ax_.text(dx, 1.0, lab, transform=ax_.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")

    if a.panel_a == "sweep":
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
