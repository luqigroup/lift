r"""``dwell_texture`` -- same level, different texture.

Panel (a) is the exit rate per coordinate-visit, every experiment and every
construction, on a log axis. Panels (b), (c) and (d) draw the stuck set as a
mask over training for one experiment, so the textures behind comparable
occupancy levels are visible.

The stuck set differs by construction. For direct softplus and the lift it is
the shoulder, emitted weight ``theta <= softplus(logit(0.05)) = 0.0512933``;
for PGD it is the projection's fixed set ``theta <= 0``, which is exactly zero
once the weight is clamped after every step. The two never merge: a projected
weight cannot satisfy ``sigmoid(w) <= 0.05`` and a softplus weight is never
exactly zero. The exit rate, transitions out of the set divided by
coordinate-visits in it, is defined the same way for both, which is why it is
the statistic panel (a) compares.

Everything is read from caches and nothing is trained here:
``docs/analysis/dwell_time.json`` for panel (a), and the HEPMASS energy-model
archive's ``theta_snaps`` for the rasters. Per-coordinate snapshots exist only
on the 1-D gallery and the three tabular energy-model lines, so the remaining
experiments are absent from this figure rather than proxied into it.

Usage::

    python scripts/render_dwell_texture.py
"""
from __future__ import annotations
import argparse, json, math, os, shutil, sys
import numpy as np, matplotlib, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

COL = {"lift": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c"}
DASH = {"lift": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
LABEL = {"lift": "lift", "direct": "direct softplus", "pgd": "PGD"}
ARMS = ["direct", "lift", "pgd"]          # top-to-bottom order inside one row of panel (a)
RASTERS = ["direct", "lift", "pgd"]       # and top to bottom down panels (b), (c), (d)
CKPT_TAG = {"direct": "direct", "lift": "hypernet", "pgd": "pgd"}
# The stuck set of each raster. The two softplus constructions share the
# shoulder threshold; PGD's is the projection's exact-zero set, a different
# set, which is never compared as a level with the other two.
RASTER_SET = {"direct": "shoulder", "lift": "shoulder", "pgd": "exactly zero"}
XLO, XHI = 1e-6, 2.0      # panel (a) exit-rate axis, six decades
MEMBER_ALPHA = 0.16       # three levels stay distinct at print size: white
# (out of the set), pale color (in it), saturated color (in it and leaving
# before the next snapshot).
DAGGER = r"$^{\dagger}$"                   # cmr10 has no dagger glyph; mathtext supplies it

# Panel (a) rows, top to bottom: the four 1-D gallery targets at 30 seeds
# each, then the three tabular energy-model lines. ``dag`` marks POWER, whose
# only PGD archive is at 512 x 5 against its own direct and lift at 256 x 3.
# The exit rate is per visit and is not normalized by width, so the row is
# kept and the mismatch is marked.
ROWS = [("Gumbel", "gallery", "gumbel", False), ("Laplace", "gallery", "laplace", False),
        ("Gamma", "gallery", "gamma", False), ("Beta", "gallery", "beta", False),
        ("HEPMASS", "tabular", "HEPMASS-EBM", False), ("MiniBooNE", "tabular", "MiniBooNE-EBM", False),
        ("POWER", "tabular", "POWER-EBM", True)]

HEPMASS_DIR = os.path.join(
    _REPO, "data", "checkpoints",
    "experiments_pgd_uci_hepmass_target_kind-uci_hepmass_K-1_objective-fkl-direct_data_path-"
    "_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_sampler-0.005_hidden_dim-512_nlayers-5"
    "_hyper_hidden_sizes-64-64-96_sampler_hyper.b0929293841311db47eff4ffed48c9fb67bf0bca")


def _diverged(rec: dict) -> bool:
    """A run whose reported test NLL is negative or non-finite.

    A diverged run has a frozen stuck set, so it is drawn as an open marker
    rather than folded into the median.
    """
    v = rec.get("test_nll")
    return v is not None and (not math.isfinite(float(v)) or float(v) < 0.0)


def _cells(blob: dict) -> dict:
    """(row label, method) -> the per-seed rows of ``dwell_time.json`` for that cell."""
    out: dict = {}
    for label, sec, key, _ in ROWS:
        for arm in ARMS:
            if sec == "gallery":
                rs = [r for r in blob["gallery"] if r["target"] == key and r["arm"] == arm]
            else:                       # POWER's PGD lives under its own width-annotated dataset name
                rs = [r for r in blob["tabular"]
                      if r["arm"] == arm and r["dataset"].startswith(key.split("-")[0])]
            out[(label, arm)] = sorted(rs, key=lambda r: r["seed"])
    return out


def panel_a_data(cells: dict, verbose: bool = True) -> dict:
    """Panel (a), cell by cell: median exit rate, per-seed range, and the flags.

    Shared with ``render_dwell_survival``, so the left-hand panel is one
    implementation and cannot drift between the two figures. Zero exit rates
    cannot be drawn on a log axis: a zero means no exit was observed among that
    run's coordinate-visits, so it is censored at ``1 / n_coord_visits_in``,
    the smallest rate the instrument can resolve.
    """
    plot: dict = {}

    plot: dict = {}
    for label, sec, key, dag in ROWS:
        for arm in ARMS:
            rs = cells[(label, arm)]
            if not rs:
                continue
            ok = [r for r in rs if not _diverged(r)]
            bad = [r for r in rs if _diverged(r)]
            e = np.asarray([r["exit_rate"] for r in ok], float)
            nz = np.asarray([1.0 / r["n_coord_visits_in"] if r["n_coord_visits_in"] else XLO for r in ok], float)
            e_draw = np.where(e > 0, e, nz)              # censored at the resolution limit
            med = float(np.median(e_draw))
            degenerate = float(np.median([r["ever_frac"] for r in ok])) < 0.05
            plot[(label, arm)] = dict(
                med=med, lo=float(e_draw.min()), hi=float(e_draw.max()), n=len(ok),
                n_zero=int((e == 0).sum()), degenerate=degenerate,
                diverged=[(r["seed"], float(r["exit_rate"]) or
                           (1.0 / r["n_coord_visits_in"] if r["n_coord_visits_in"] else XLO),
                           r["exit_rate"] == 0.0) for r in bad],
                seeds=[r["seed"] for r in ok],
                occ_last=float(np.median([r["occ_last"] for r in ok])),
                ever=float(np.median([r["ever_frac"] for r in ok])),
                ep=float(np.median([r["n_episodes"] / (r["ever_frac"] * r["n_constrained"])
                                    for r in ok if r["ever_frac"] > 0])) if any(r["ever_frac"] > 0 for r in ok) else float("nan"),
                mono=int(sum(bool(r["monotone_nondecreasing"]) for r in ok)),
                src=os.path.relpath(rs[0]["source"], os.path.expanduser("~")))
            d = plot[(label, arm)]
            if verbose: print(f"(a) {label:10s}{'(dagger)' if dag else '        '} {LABEL[arm]:15s} n={d['n']:2d} seeds {d['seeds'][0]}-{d['seeds'][-1]}"
                  f"  exit median {d['med']:.3e}  range {d['lo']:.3e}-{d['hi']:.3e}"
                  f"  zeros {d['n_zero']}  monotone {d['mono']}/{d['n']}"
                  f"  occ_last {d['occ_last']:.4f}  ever {d['ever']:.4f}  episodes/coord {d['ep']:.3f}"
                  + ("  [DEGENERATE: the set is essentially empty]" if d["degenerate"] else ""))
            if verbose: print(f"    checkpoints ~/{d['src']}")
            for s_, v, z in d["diverged"]:
                if verbose: print(f"    seed {s_} DIVERGED (test NLL negative), exit rate "
                      f"{'0 (no exit observed), drawn censored at ' if z else ''}{v:.3e}: "
                      f"drawn open, excluded from the median")

    return plot



def draw_panel_a(ax, plot, lab_fs, tick_fs, ax_fs, tag_fs, tag="(a)", *,
                 dagger=True, show_diverged=True, range_lw=0.8, range_alpha=1.0, ms=2.9,
                 xlabel="exit rate per visit", hint=None):
    """Draw panel (a) into ``ax``. One implementation, both figures.

    The defaults are what ``render_dwell_texture`` draws;
    ``render_dwell_survival`` overrides them.

    * ``dagger=False`` drops the POWER row's dagger, a mark whose meaning a
      reader cannot recover from the figure; the caption carries it instead.
    * ``show_diverged=False`` drops the small open markers of the diverged
      seeds, which are already excluded from the median.
    * ``hint=(left, right)`` writes two gray words under the ends of the axis,
      so its direction is readable without arithmetic.
    """
    ax.set_xscale("log"); ax.set_xlim(XLO, XHI); ax.set_ylim(len(ROWS) - 0.5, -0.5)
    dy = {"direct": -0.26, "lift": 0.0, "pgd": 0.26}
    for i, (label, sec, key, dag) in enumerate(ROWS):
        ax.axhspan(i - 0.5, i + 0.5, color="0.0", alpha=0.045 if i % 2 else 0.0, lw=0)
        for arm in ARMS:
            d = plot.get((label, arm))
            if d is None:
                continue
            y = i + dy[arm]; c = COL[arm]
            ax.plot([max(d["lo"], XLO), min(d["hi"], XHI)], [y, y], color=c, lw=range_lw,
                    alpha=range_alpha, solid_capstyle="butt", zorder=3)
            ax.plot([d["med"]], [y], marker="o", ms=ms, mfc=("none" if d["degenerate"] else c),
                    mec=c, mew=0.8, ls="none", zorder=4)
            if show_diverged:
                for _, v, _z in d["diverged"]:
                    ax.plot([max(v, XLO)], [y], marker="o", ms=2.4, mfc="none", mec=c, mew=0.6,
                            ls="none", zorder=4)
        ax.text(-0.035, i, label + (DAGGER if (dag and dagger) else ""),
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=lab_fs)
    ax.set_yticks([])
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1e0])
    ax.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$1$"], fontsize=tick_fs)
    ax.tick_params(axis="x", length=2.2, pad=1.4)
    ax.set_xlabel(xlabel, fontsize=ax_fs, labelpad=(1.0 if hint is None else 11.0))
    if hint is not None:
        for frac, s_txt, ha in ((0.0, hint[0], "left"), (1.0, hint[1], "right")):
            ax.annotate(s_txt, xy=(frac, 0.0), xycoords="axes fraction",
                        xytext=(0, -12.5), textcoords="offset points",
                        ha=ha, va="top", fontsize=ax_fs - 1.0, color="0.45",
                        annotation_clip=False)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.text(0.0, 1.008, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=tag_fs, color="0.35")


def _masks(args) -> dict:
    """Per-coordinate stuck-set masks of the three HEPMASS seed-0 runs.

    ``M[i, t]`` is True when constrained coordinate ``i`` is in that run's
    stuck set at snapshot ``t``. Read once from the archived ``theta_snaps``
    through ``measure_dwell_time``'s own ``_constrained``, so the key set,
    order and thresholds match that measurement, then cached.
    """
    if os.path.exists(args.mask_cache) and not args.refresh_masks:
        z = np.load(args.mask_cache, allow_pickle=True)
        if all(k in z.files for k in RASTERS):        # an older two-entry cache is re-read
            return {k: z[k] for k in z.files} | {"iters": z["iters"], "cached": True}
        print(f"  cache {os.path.basename(args.mask_cache)} predates the lift raster; re-reading")
    import torch
    from measure_dwell_time import _constrained, THR_EMITTED
    out: dict = {}
    for arm, thr in (("direct", THR_EMITTED), ("lift", THR_EMITTED), ("pgd", 0.0)):
        path = os.path.join(HEPMASS_DIR, f"seed0_{CKPT_TAG[arm]}.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        snaps, iters = ck["theta_snaps"], ck.get("snap_iters")
        out[arm] = np.stack([(_constrained(th).numpy() <= thr) for th in snaps], axis=1)
        out["iters"] = np.asarray(iters, dtype=int)
        out[f"src_{arm}"] = np.asarray(path)
        print(f"  read {os.path.basename(path):16s} {out[arm].shape[0]} constrained x "
              f"{out[arm].shape[1]} snapshots  thr {thr:.7f}", flush=True)
        del ck, snaps
    os.makedirs(os.path.dirname(args.mask_cache), exist_ok=True)
    np.savez_compressed(args.mask_cache, **out)
    out["cached"] = False
    return out


def _raster(M: np.ndarray, n_coords: int, seed: int) -> tuple:
    """A fixed random sample of coordinates, rows ordered by exit count.

    Returns ``(member, exit_, n_exit, idx)``. ``member[i, t]`` is True when the
    sampled coordinate is in the set at snapshot ``t``; ``exit_[i, t]`` is True
    when it is in the set at ``t`` and out of it at ``t+1``, so a marked cell is
    the LAST snapshot of an episode and always sits on a membership cell.

    The sample is drawn once, uniformly over all constrained coordinates and
    not only over the ones that participate, so the pale field of a raster has
    the density of that occupancy and the strips above stay readable against
    it.

    Rows are ordered by the number of exits, most mobile at the top, stably, so
    the random order is kept inside one count and no second structure is
    imposed.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(M.shape[0], size=min(n_coords, M.shape[0]), replace=False))
    S = M[idx]
    E = np.zeros_like(S)
    E[:, :-1] = S[:, :-1] & ~S[:, 1:]
    n_exit = E.sum(1)
    order = np.argsort(-n_exit, kind="stable")
    return S[order], E[order], n_exit[order], idx


def _text_box_rot(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs, rotation=90); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def _text_w(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default=os.path.join(_REPO, "docs/analysis/dwell_time.json"))
    p.add_argument("--out_dir", default=os.path.join(_REPO, "figures"))
    p.add_argument("--mask_cache", default=os.path.join(_REPO, "data/analysis/dwell_texture_masks.npz"))
    p.add_argument("--refresh_masks", action="store_true", help="re-read the two HEPMASS checkpoints instead of the cache")
    p.add_argument("--mirror_to", default="", help="empty disables the copy; off here")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--fig_height", type=float, default=2.75)
    p.add_argument("--n_coords", type=int, default=0,
                   help="constrained coordinates sampled for the rasters; 0 (default) means one "
                        "per output pixel row, so no drawn cell is lost to image resampling")
    p.add_argument("--sample_seed", type=int, default=0, help="RNG seed of that sample; the sample is the only randomness in the figure")
    p.add_argument("--keep_duplicate_last", action="store_true",
                   help="keep snapshot 21 (iteration 4,000), bit-identical to snapshot 20 (iteration 3,999)")
    p.add_argument("--dpi", type=int, default=1200,
                   help="also sets the raster row count when --n_coords is 0")
    a = p.parse_args()

    apply_paper_style()
    matplotlib.rcParams["savefig.bbox"] = "standard"; matplotlib.rcParams["axes.grid"] = False

    blob = json.load(open(a.json))
    cells = _cells(blob)
    print(f"panel (a) source: {a.json}\n")

    plot = panel_a_data(cells)

    # ---- layout -------------------------------------------------------------
    W, H = a.fig_width, a.fig_height
    strip_entries = [(LABEL["lift"], COL["lift"], DASH["lift"]),
                     (LABEL["direct"], COL["direct"], DASH["direct"]),
                     (LABEL["pgd"], COL["pgd"], DASH["pgd"])]
    STRIP_FS = 6.6; strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.34, 0.28, 0.045, 0.04, 0.05
    LAB_FS, TICK_FS, AX_FS, TAG_FS = 6.6, 6.8, 7.6, 7.0
    probe = plt.figure(figsize=(W, H))
    strip_text_in = max(_text_box_rot(probe, t, STRIP_FS)[0] for t, _, _ in strip_entries)
    lab_w = max(_text_w(probe, lab + (DAGGER if dag else ""), LAB_FS) for lab, _, _, dag in ROWS)
    plt.close(probe)
    right_in = strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in

    left_in = lab_w + 0.05
    pa_w = 1.16                      # panel (a) axes: six decades, 1e-6 to 2
    gap_ab = 0.46                    # room for the two rotated labels beside the rasters
    ras_w = W - right_in - left_in - pa_w - gap_ab
    bot_in, top_in = 0.28, 0.13      # x tick labels + axis label; panel tags
    strip_h, strip_gap, mid_in = 0.115, 0.016, 0.095
    n_ras = len(RASTERS)
    ras_h = (H - bot_in - top_in - n_ras * (strip_h + strip_gap)
             - (n_ras - 1) * mid_in) / n_ras

    # ---- panels (b, c, d): the three HEPMASS seed-0 masks --------------------
    print(f"\npanels (b, c, d) source: {HEPMASS_DIR}")
    mk = _masks(a)
    iters = np.asarray(mk["iters"], int)
    keep = iters.size
    if not a.keep_duplicate_last and keep >= 2 and all(
            bool((mk[arm][:, -1] == mk[arm][:, -2]).all()) for arm in RASTERS):
        keep -= 1
        print(f"  snapshot {iters.size - 1} (iteration {iters[-1]}) is bit-identical to snapshot "
              f"{iters.size - 2} (iteration {iters[-2]}) in all three arms and is dropped; "
              f"{keep} columns drawn")
    iters = iters[:keep]

    # One sampled coordinate per output pixel row, so every drawn cell is a
    # real coordinate at a real snapshot and nothing is lost to image
    # resampling. Decimating the rows into the strip height would hide the
    # sparse exits and leave the dense ones, a rendering artifact that would
    # read as a finding.
    n_rows = a.n_coords if a.n_coords > 0 else int(round(ras_h * a.dpi))
    R, X, occ, sample, stats = {}, {}, {}, {}, {}
    for arm in RASTERS:
        F = np.asarray(mk[arm])                      # the full record, every snapshot
        M = F[:, :keep]                              # what is drawn
        R[arm], X[arm], n_ex, sample[arm] = _raster(M, n_rows, a.sample_seed)
        occ[arm] = M.mean(0)
        # Every statistic printed on the figure is computed on the full
        # record, so the tag line carries the numbers of panel (a). Dropping
        # the duplicate last snapshot changes what is drawn, never what is
        # quoted.
        starts = int(F[:, 0].sum() + (F[:, 1:] & ~F[:, :-1]).sum())
        ever = float(F.any(1).mean())
        n_in = int(F[:, :-1].sum()); n_exit = int((F[:, :-1] & ~F[:, 1:]).sum())
        stats[arm] = dict(ever=ever, ep=starts / (ever * F.shape[0]),
                          exit=n_exit / n_in if n_in else float("nan"))
        # rows of the sample carrying at least one exit: the marked rows
        holed = int((n_ex > 0).sum()); stats[arm]["marked"] = holed
        print(f"  {LABEL[arm]:15s} seed 0  {M.shape[0]} constrained ({RASTER_SET[arm]}) x "
              f"{keep} snapshots (iterations {iters[0]}-{iters[-1]})")
        print(f"    occupancy {' '.join(f'{v:.3f}' for v in occ[arm])}")
        print(f"    occupancy {occ[arm][0]:.4f} -> {occ[arm][-1]:.4f}, mean {occ[arm].mean():.4f}, "
              f"monotone {bool(np.all(np.diff(occ[arm]) >= -1e-12))}, ever {ever:.4f}, "
              f"episodes/participating coordinate {stats[arm]['ep']:.3f}, "
              f"exit rate {stats[arm]['exit']:.3e}")
        print(f"    raster: {R[arm].shape[0]} coordinates sampled uniformly (rng seed "
              f"{a.sample_seed}), rows sorted by exit count; "
              f"{int((~R[arm].any(1)).sum())} never enter; "
              f"{holed} of {R[arm].shape[0]} rows carry at least one exit and are MARKED "
              f"({int(X[arm].sum())} marked cells, max {int(n_ex.max())} per row); "
              f"sampled fill {R[arm].mean():.4f} against the full-run {M.mean():.4f}")

    fig = plt.figure(figsize=(W, H))

    # ---- panel (a) ----------------------------------------------------------
    ax = fig.add_axes([left_in / W, bot_in / H, pa_w / W, (H - bot_in - top_in) / H])
    draw_panel_a(ax, plot, LAB_FS, TICK_FS, AX_FS, TAG_FS)

    # ---- panels (b, c, d) ---------------------------------------------------
    x0 = left_in + pa_w + gap_ab
    for j, (arm, tag) in enumerate(zip(RASTERS, ("(b)", "(c)", "(d)"))):
        c = COL[arm]
        y_ras = (bot_in + (n_ras - 1 - j) * (ras_h + strip_h + strip_gap + mid_in)) / H
        y_str = y_ras + (ras_h + strip_gap) / H
        axr = fig.add_axes([x0 / W, y_ras, ras_w / W, ras_h / H])
        axr.set_facecolor("white")
        ext = [-0.5, keep - 0.5, R[arm].shape[0] - 0.5, -0.5]
        # Two layers: membership is a pale ground in the method's color and an
        # exit is a saturated cell in the same color, so the eye reads the
        # marks rather than the gaps between them.
        axr.imshow(R[arm], aspect="auto", origin="upper", interpolation="nearest",
                   cmap=ListedColormap([(0, 0, 0, 0), to_rgba(c, MEMBER_ALPHA)]),
                   vmin=0, vmax=1, extent=ext, zorder=2)
        axr.imshow(X[arm], aspect="auto", origin="upper", interpolation="nearest",
                   cmap=ListedColormap([(0, 0, 0, 0), to_rgba(c, 1.0)]),
                   vmin=0, vmax=1, extent=ext, zorder=3)
        axr.set_xlim(ext[0], ext[1]); axr.set_ylim(ext[2], ext[3])
        axr.set_yticks([])
        for s_ in axr.spines.values():
            s_.set_edgecolor("0.55"); s_.set_linewidth(0.5)
        axr.set_ylabel(LABEL[arm], fontsize=LAB_FS, color=c, labelpad=2.0)
        if j == 0:      # one shared row-axis label, outside the three names
            fig.text((x0 - 0.27) / W, (bot_in + (H - bot_in - top_in) / 2) / H,
                     f"{R[arm].shape[0]:,} coordinates, sorted by exit count",
                     rotation=90, ha="center", va="center", fontsize=5.9, color="0.35")
        if j == n_ras - 1:
            axr.set_xticks([0, 5, 10, 15, 20][: keep])
            axr.set_xticklabels([f"{iters[t] / 1000:.0f}k" if iters[t] else "0"
                                 for t in [0, 5, 10, 15, 20][: keep]], fontsize=TICK_FS)
            axr.tick_params(axis="x", length=2.2, pad=1.4)
            axr.set_xlabel("iteration", fontsize=AX_FS, labelpad=1.0)
        else:
            axr.set_xticks([])
        axs = fig.add_axes([x0 / W, y_str, ras_w / W, strip_h / H])
        axs.fill_between(np.arange(keep), 0.0, occ[arm], color=c, alpha=0.30, lw=0, step="mid")
        axs.plot(np.arange(keep), occ[arm], color=c, lw=0.8, drawstyle="steps-mid")
        axs.set_xlim(-0.5, keep - 0.5); axs.set_ylim(0, 1); axs.set_xticks([]); axs.set_yticks([])
        for s_ in axs.spines.values():
            s_.set_edgecolor("0.55"); s_.set_linewidth(0.5)
        axs.text(0.0, 1.05, tag, transform=axs.transAxes, ha="left", va="bottom",
                 fontsize=TAG_FS, color="0.35")
        # The level and the mobility in numbers, on the tag line, so the strip
        # itself carries only the shape of the series. PGD's set is named, so
        # its level is not read against the two shoulder levels above it.
        axs.text(0.055, 1.05,
                 rf"{RASTER_SET[arm]}, occupancy {occ[arm][0]:.2f}$\,\to\,${occ[arm][-1]:.2f}, "
                 rf"exit {stats[arm]['exit']:.1e} on {stats[arm]['marked']} of "
                 rf"{R[arm].shape[0]} rows",
                 transform=axs.transAxes, ha="left", va="bottom", fontsize=5.8, color=c)

    # ---- rotated legend strip ----------------------------------------------
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

    os.makedirs(a.out_dir, exist_ok=True); stem = os.path.join(a.out_dir, "dwell_texture")
    fig.savefig(stem + ".pdf", dpi=a.dpi); fig.savefig(stem + ".png", dpi=a.dpi); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to):
        shutil.copy2(stem + ".pdf", a.mirror_to)
    print(f"\nwrote {stem}.pdf and .png ({W:.2f} x {H:.2f} in at {a.dpi} dpi)")
    print(f"mask cache {a.mask_cache}")


if __name__ == "__main__":
    main()
