"""Captioned figures for the additional experiments, one PDF each.

Nothing is trained or re-evaluated: every figure is drawn from the records the
experiments write, and every number drawn is printed with the file it was read
from. The medians drawn are checked against the reported values below, and a
figure stops on any disagreement beyond rounding. The PDF metadata is empty.

Usage::

    python scripts/render_additional_figures.py --figure all

Output: ``figures/additional/<name>.pdf``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from projorg import gitdir

_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style

OUT_DIR = os.path.join(_REPO, "figures", "additional")
ANALYSIS = os.path.join(_REPO, "data", "analysis")
CK = os.path.join(_REPO, "data", "checkpoints")
SERIES_CACHE = os.path.join(ANALYSIS, "occupancy_series_1d.json")
RATES_CACHE = os.path.join(ANALYSIS, "occupancy_rates_after_first_1d.json")
EXP_MAP = os.path.join(ANALYSIS, "exp_map_1d.json")
EXP_MAP_PINNED = os.path.join(ANALYSIS, "pinned", "exp_map_1d")

TARGETS = ("gumbel", "laplace", "gamma", "beta")
TARGET_LABEL = {"gumbel": "Gumbel", "laplace": "Laplace", "gamma": "Gamma", "beta": "Beta"}
COL = {"lift": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c",
       "exppost": "#17becf", "direct_exp": PALETTE["direct_softplus"], "lift_exp": PALETTE["hypernet"]}
LABEL = {"lift": "lift", "direct": "direct softplus", "pgd": "PGD",
         "exppost": "Sivaprasad et al.", "direct_exp": "direct, exp. map", "lift_exp": "lift, exp. map"}
DASH = {"lift": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2)),
        "exppost": (0, (1, 1.1)), "direct_exp": (0, (2, 1.0)), "lift_exp": (0, (2, 1.0))}
PDF_META = {k: None for k in ("Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate")}
GUARD_1D = 200.0
FS = 8.0
LINESP = 1.3
STEP_TEX = {"5e-4": r"$5{\cdot}10^{-4}$", "1e-3": r"$10^{-3}$", "2e-3": r"$2{\cdot}10^{-3}$"}

MEDIAN_TV_BY_METHOD = {"lift": (0.016, 0.027, 0.012, 0.012), "direct": (0.137, 0.140, 0.125, 0.073),
                       "pgd": (0.130, 0.139, 0.038, 0.072), "exppost": (0.110, 0.144, 0.051, 0.071),
                       "direct_exp": (0.136, 0.142, 0.129, 0.079)}
SEEDS_BY_METHOD = {"lift": 30, "direct": 30, "pgd": 30, "exppost": 3, "direct_exp": 3}
SHARE_ENTRY_EXIT = {("gumbel", "direct"): (0.36, 0.39, 3.8e-3, 9.8e-6),
                    ("gumbel", "lift"): (0.06, 0.97, 0.64, 1.4e-3),
                    ("beta", "direct"): (0.36, 0.31, 3.8e-4, 1.1e-2),
                    ("beta", "lift"): (0.05, 0.76, 0.17, 9.3e-3)}
GUMBEL_END_SHARE = {"direct": (0.16, 0.39, 0.71), "lift": (0.75, 0.97, 1.00)}
CIFAR_TEST_ACCURACY = {"lift": 53.4, "direct": 53.2, "pgd": 52.9, "exppost": 52.5}
REFERENCE_SET = {"pinned": (0.022, 0.030, 0.033, 0.023), "batch": (0.031, 0.033, 0.035, 0.023),
                 "direct": (0.145, 0.151, 0.175, 0.113), "darcy_max_mean_shift": 0.0002, "darcy_max_sd": 0.0004}
DIRECT_SHARE = {"1d_start": 0.36, "1d_end": (0.31, 0.40), "power": (3, 0.49, 0.54), "hepmass": (3, 0.65, 0.94),
                "miniboone": (3, 0.65, 0.96), "ffhq": (28, 1.00, 1.00), "flows": (5, 0.00, 0.00),
                "darcy": (10, 0.00, 0.00)}
MEDIAN_TV_EXP_MAP = {"direct_exp": (0.139, 0.141, 0.116, 0.074), "lift_exp": (0.014, 0.022, 0.052, 0.014),
                     "lift": (0.015, 0.023, 0.020, 0.023)}
STEP_EXP_MAP = {"direct_exp": ("1e-3", "1e-3", "2e-3", "5e-4"), "lift_exp": ("1e-3",) * 4, "lift": ("1e-3",) * 4}
SHARE_EXIT_BY_THRESHOLD = {
    "gumbel": {"direct": ((0.16, 0.39, 0.71), (0.0, 9.8e-6, 5.5e-6)),
               "lift": ((0.75, 0.97, 1.00), (7.4e-3, 1.4e-3, 3.1e-4))},
    "laplace": {"direct": ((0.16, 0.40, 0.71), (5.9e-6, 9.8e-6, 5.5e-6)),
                "lift": ((0.80, 0.97, 1.00), (4.4e-3, 1.2e-3, 2.3e-4))},
    "gamma": {"direct": ((0.16, 0.40, 0.71), (1.2e-5, 7.4e-6, 8.1e-6)),
              "lift": ((0.49, 0.84, 0.96), (2.0e-2, 6.9e-3, 3.0e-3))},
    "beta": {"direct": ((0.12, 0.31, 0.58), (1.2e-2, 1.1e-2, 9.1e-3)),
             "lift": ((0.38, 0.76, 0.92), (1.5e-2, 9.3e-3, 5.3e-3))},
}
CITE_URL = {"Sivaprasad et al. (2021)": "https://doi.org/10.1007/978-3-030-86486-6_45"}
LINK_COLOR = "#1f4e9c"
PSI_KEYS = ("psi0.02", "psi0.05", "psi0.10")
PSI_LABEL = ("0.02", "0.05", "0.10")

CAPTIONS = {
    "positivity_methods_1d":
        "On every target the lift is closest to the target at the median; the rule of Sivaprasad et al. (2021) "
        "sits near PGD and the exponential map near direct softplus (one point per seed, medians as bars).",
    "exp_map_tuned_1d":
        "Under the exponential map the lift stays well ahead of direct training on every seed and close to the "
        "paper's lift except on Gamma: distance at each run's best held-out loss (top), held-out loss (bottom).",
    "shoulder_thresholds_1d":
        "The comparison does not depend on the threshold: at each one the lift ends with more weights on the "
        "shoulder (top) and its visits end far sooner (bottom), except on Beta, where the rates are close. "
        "Medians over seeds; the paper's threshold in gray.",
    "shoulder_entry_exit":
        "On Gumbel a weight of direct softplus on the shoulder essentially never leaves, while the lift's weights "
        "enter and leave far more often; on Beta both leave about as often and the lift's weights enter far more "
        "often (medians; rates after the first interval).",
    "reference_set_1d":
        "The lift reading one batch instead of the fixed sample (each seed's two readings joined) stays far "
        "below direct softplus; on the transport map the held-out loss moves by under a thousandth of a nat.",
    "direct_share_by_setting":
        "Direct softplus reaches the shoulder on the one-dimensional targets and the tabular benchmarks and starts "
        "on it on the image regularizer, but barely reaches it on the flows and not on the transport map.",
    "cifar10_sivaprasad":
        "On the CIFAR-10 task of Sivaprasad et al. (2021) the lift has the highest median test accuracy at the "
        "best validation epoch (bottom); training and validation curves above, seeds thin and medians bold.",
}


def _rel(path: str) -> str:
    return os.path.relpath(path, _REPO) if os.path.isabs(path) else path


def _say(fig: str, what: str, values, source: str) -> None:
    """Print a drawn number or array with the file it came from."""
    arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    txt = f"{arr[0]:.4g}" if arr.size == 1 else "[" + " ".join(f"{x:.4g}" for x in arr) + "]"
    print(f"[{fig}] {what}: {txt}  <- {_rel(source)}")


class Check:
    """Prints a drawn median beside its reported value; stops the figure on a disagreement."""

    def __init__(self, fig: str) -> None:
        self.fig, self.bad = fig, []

    def __call__(self, label: str, mine: float, theirs: float, decimals: int | None = None,
                 rel: float | None = None) -> None:
        if decimals is not None:
            ok = abs(mine - theirs) <= 0.5 * 10.0 ** (-decimals) + 1e-12
            s = f"{mine:.{decimals + 2}f} vs {theirs:.{decimals}f}"
        else:
            ok = (mine == 0 and theirs == 0) or (theirs != 0 and abs(mine - theirs) / abs(theirs) <= rel)
            s = f"{mine:.3g} vs {theirs:.3g}"
        print(f"  [check {self.fig}] {label:52s} {s:>24s}  {'ok' if ok else 'DISAGREES'}")
        if not ok:
            self.bad.append(label)

    def stop_if_bad(self) -> None:
        if self.bad:
            raise SystemExit(f"[{self.fig}] {len(self.bad)} number(s) disagree with the reported values: {self.bad}")


def _text_width_in(s: str, fontsize: float) -> float:
    if not s:
        return 0.0
    tp = TextPath((0, 0), s, size=fontsize, prop=FontProperties(family="serif"))
    return tp.get_extents().width / 72.0


def _wrap(text: str, width_in: float, fontsize: float = FS) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        cand = (cur + " " + w).strip()
        if cur and _text_width_in(cand, fontsize) > width_in:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _caption_block(name: str, width_in: float) -> tuple[list[str], float]:
    text = CAPTIONS[name]
    for cite in CITE_URL:
        text = text.replace(cite, cite.replace(" ", "\u00a0"))
    lines = [ln.replace("\u00a0", " ") for ln in _wrap(text, width_in)]
    if len(lines) > 2:
        raise SystemExit(f"[{name}] caption wraps to {len(lines)} lines at {width_in:.2f} in; two at most")
    return lines, len(lines) * FS * LINESP / 72.0 + 0.12


def _axes_grid(fig, nrows: int, ncols: int, *, left: float, right: float, top: float, bottom: float,
               wsp, hsp: float, widths: Sequence[float] | None = None):
    """Axes placed in inches; ``wsp`` is one gap or one gap per column boundary."""
    W, H = fig.get_size_inches()
    widths = list(widths) if widths else [1.0] * ncols
    gaps = list(wsp) if isinstance(wsp, (list, tuple)) else [float(wsp)] * max(ncols - 1, 0)
    pw = (W - left - right - sum(gaps)) / sum(widths)
    ph = (H - top - bottom - hsp * (nrows - 1)) / nrows
    out = []
    y = H - top
    for _ in range(nrows):
        y -= ph
        x, row = left, []
        for c in range(ncols):
            ww = pw * widths[c]
            row.append(fig.add_axes([x / W, y / H, ww / W, ph / H]))
            x += ww + (gaps[c] if c < len(gaps) else 0.0)
        out.append(row)
        y -= hsp
    return out


def _despine(ax, fs: float = FS - 0.5) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=fs, length=2.5, pad=1.5)


def _axis_break(ax, y: float, size_pt: float = 3.0) -> None:
    """Two short slashes across the left spine at data height ``y``."""
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    shift = matplotlib.transforms.ScaledTranslation
    ax.plot([0], [y], transform=trans, marker="s", ms=0.9 * size_pt, mew=0, color="white", clip_on=False,
            zorder=6.5)
    for dy in (-0.45 * size_pt, 0.45 * size_pt):
        ax.plot([0], [y], transform=trans + shift(0, dy / 72.0, ax.figure.dpi_scale_trans),
                marker=[(-1.0, -0.55), (1.0, 0.55)], ms=2 * size_pt, mew=0.8, color="black", clip_on=False,
                zorder=7)


def _caption_line(fig, line: str, x: float, y: float) -> None:
    """One caption line, with every citation in ``CITE_URL`` drawn as a link."""
    renderer = fig.canvas.get_renderer()
    parts, rest = [], line
    while rest:
        hits = [(rest.find(c), c) for c in CITE_URL if c in rest]
        if not hits:
            parts.append((rest, None))
            break
        k, c = min(hits)
        if k:
            parts.append((rest[:k], None))
        parts.append((c, CITE_URL[c]))
        rest = rest[k + len(c):]
    space = (_text_width_in("a a", FS) - _text_width_in("aa", FS)) / fig.get_size_inches()[0]
    for seg, url in parts:
        if seg.startswith(" "):
            x += space
        t = fig.text(x, y, seg.strip(" "), ha="left", va="bottom", fontsize=FS,
                     color=LINK_COLOR if url else "black", url=url)
        x += t.get_window_extent(renderer).width / fig.bbox.width
        if seg.endswith(" "):
            x += space


def _finish(fig, name: str, lines: list[str], left_in: float, out_dir: str) -> None:
    W, H = fig.get_size_inches()
    step = FS * LINESP / 72.0
    for k, line in enumerate(lines):
        _caption_line(fig, line, left_in / W, (0.05 + (len(lines) - 1 - k) * step) / H)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".pdf")
    fig.savefig(path, metadata=PDF_META)
    plt.close(fig)
    print(f"[{name}] wrote {_rel(path)}  ({len(lines)} caption lines)")


def _strip(ax, x: float, vals, color: str, rng, *, filled: bool = True, ms: float = 3.0,
           jitter: float = 0.14, med_w: float = 0.28, med_lw: float = 1.6, alpha: float = 1.0,
           horizontal: bool = False, z: float = 3.0, points_above: bool = False) -> float:
    """One strip of seeds at category ``x`` with its median as a bar; returns the median."""
    v = np.asarray(vals, dtype=float)
    off = rng.uniform(-jitter, jitter, size=v.size) if v.size > 1 else np.zeros(1)
    xs = np.full(v.size, x) + off
    a, b = (v, xs) if horizontal else (xs, v)
    ax.scatter(a, b, s=ms ** 2, facecolors=color if filled else "white", edgecolors=color,
               linewidths=0.6, alpha=alpha, zorder=z + 2 if points_above else z)
    m = float(np.median(v))
    if horizontal:
        ax.plot([m, m], [x - med_w, x + med_w], color=color, lw=med_lw, zorder=z + 1, solid_capstyle="butt")
    else:
        ax.plot([x - med_w, x + med_w], [m, m], color=color, lw=med_lw, zorder=z + 1, solid_capstyle="butt")
    return m


def _thresholds_cache() -> dict:
    with open(os.path.join(ANALYSIS, "occupancy_thresholds.json")) as f:
        return json.load(f)


def _gallery_rows(cache: dict, target: str, arm: str) -> list[dict]:
    rows = sorted((r for r in cache["gallery"] if r["target"] == target and r["arm"] == arm),
                  key=lambda r: r["seed"])
    if len(rows) != 30:
        raise RuntimeError(f"{target}/{arm}: {len(rows)} seeds in occupancy_thresholds.json, expected 30")
    return rows


def _occupancy_series(cache: dict, targets: Sequence[str]) -> dict:
    """Per-seed share on the shoulder at every snapshot, cached and checked against the thresholds cache."""
    series: dict = {}
    if os.path.exists(SERIES_CACHE):
        with open(SERIES_CACHE) as f:
            series = json.load(f)
    todo = [(t, a) for t in targets for a in ("direct", "lift")
            if not (t in series and a in series[t] and len(series[t][a]) == 30)]
    if todo:
        import measure_occupancy_thresholds as MO
        for t, a in todo:
            series.setdefault(t, {})[a] = {}
            for r in _gallery_rows(cache, t, a):
                loaded = MO.load_snapshots(r["source"])
                if loaded is None:
                    raise RuntimeError(f"{r['source']}: no theta_snaps")
                W, iters = loaded
                st = MO.dwell_at(W, MO.THR_EMITTED)
                series[t][a][str(r["seed"])] = {"iters": iters, "occ": st["occ_series"],
                                                "exit_rate": st["exit_rate"], "entry_rate": st["entry_rate"],
                                                "source": r["source"]}
                print(f"  [series] {t} {a} seed {r['seed']:2d} occ {st['occ_series'][0]:.3f} -> "
                      f"{st['occ_series'][-1]:.3f}  <- {_rel(r['source'])}", flush=True)
        os.makedirs(os.path.dirname(SERIES_CACHE), exist_ok=True)
        with open(SERIES_CACHE, "w") as f:
            json.dump(series, f)
        print(f"  [series] wrote {_rel(SERIES_CACHE)}")
    n_checked = 0
    for t in targets:
        for a in ("direct", "lift"):
            for r in _gallery_rows(cache, t, a):
                s = series[t][a][str(r["seed"])]
                ref = r["at"]["psi0.05"]
                for mine, theirs, what in ((s["occ"][0], ref["occ_first"], "occ_first"),
                                           (s["occ"][-1], ref["occ_last"], "occ_last"),
                                           (s["exit_rate"], ref["exit_rate"], "exit_rate"),
                                           (s["entry_rate"], ref["entry_rate"], "entry_rate")):
                    if abs(mine - theirs) > 1e-9:
                        raise SystemExit(f"series cache disagrees with occupancy_thresholds.json: "
                                         f"{t} {a} seed {r['seed']} {what} {mine} vs {theirs}")
                n_checked += 1
    print(f"  [series] {n_checked} seeds reproduce occupancy_thresholds.json at the paper's threshold")
    return series


def _rates_after_first(cache: dict, target: str, arm: str) -> list[tuple]:
    """Per seed (entry rate, exit rate) at the paper's threshold over the snapshot intervals after the first."""
    store: dict = {}
    if os.path.exists(RATES_CACHE):
        with open(RATES_CACHE) as f:
            store = json.load(f)
    key = f"{target}/{arm}"
    if key not in store or len(store[key]) != 30:
        import measure_occupancy_thresholds as MO
        store[key] = {}
        for r in _gallery_rows(cache, target, arm):
            W, iters = MO.load_snapshots(r["source"])
            full = MO.dwell_at(W, MO.THR_EMITTED)
            if abs(full["exit_rate"] - r["at"]["psi0.05"]["exit_rate"]) > 1e-9:
                raise SystemExit(f"{key} seed {r['seed']}: snapshots do not reproduce the cached exit rate")
            st = MO.dwell_at(W[1:], MO.THR_EMITTED)
            store[key][str(r["seed"])] = (float(st["entry_rate"]), float(st["exit_rate"]))
            print(f"  [rates] {key} seed {r['seed']:2d} after the first interval: entry {st['entry_rate']:.3g} "
                  f"exit {st['exit_rate']:.3g}  <- {_rel(r['source'])}", flush=True)
        os.makedirs(os.path.dirname(RATES_CACHE), exist_ok=True)
        with open(RATES_CACHE, "w") as f:
            json.dump(store, f)
    return [tuple(v) for v in store[key].values()]


def fig_positivity_methods_1d(out_dir: str) -> None:
    name = "positivity_methods_1d"
    chk, rng = Check(name), np.random.default_rng(0)
    import collect_positivity_maps_1d as CPM
    from rederive import rederive_bestval as RB
    zpath = os.path.join(ANALYSIS, "convergence_1d_pinned.npz")
    z = np.load(zpath, allow_pickle=True)
    methods = ["lift", "direct", "pgd", "exppost", "direct_exp"]
    data: dict[str, dict[str, tuple]] = {t: {} for t in TARGETS}
    for t in TARGETS:
        for m in ("lift", "direct", "pgd"):
            data[t][m] = (RB._sel_1d(z, t, m), zpath)
    for m, root, be in (("exppost", "lcmm_1d_exppost", "direct_exppost"),
                        ("direct_exp", "lcmm_1d_exp", "direct_exp")):
        res = CPM.collect(os.path.join(CK, root), [be], list(TARGETS))[be]
        for t in TARGETS:
            data[t][m] = (np.array(res[t]["best"]), os.path.join(CK, root, t))

    W, margin = 7.0, 0.1
    lines, cap_h = _caption_block(name, W - 2 * margin)
    panel_h = 1.55
    fig = plt.figure(figsize=(W, panel_h + 0.28 + 0.42 + cap_h))
    axes = _axes_grid(fig, 1, 4, left=1.0, right=0.1, top=0.28, bottom=cap_h + 0.42, wsp=0.22, hsp=0)[0]
    ys = list(range(len(methods)))[::-1]
    xmax = 0.0
    for ax, t in zip(axes, TARGETS):
        for y, m in zip(ys, methods):
            v, src = data[t][m]
            if len(v) != SEEDS_BY_METHOD[m]:
                raise RuntimeError(f"{t} {m}: {len(v)} seeds, expected {SEEDS_BY_METHOD[m]}")
            few = len(v) < 10
            med = _strip(ax, y, v, COL[m], rng, horizontal=True, jitter=0.18, filled=(m != "direct_exp"), ms=2.45,
                         alpha=1.0 if few else 0.45, med_w=0.34, med_lw=2.2, points_above=few)
            xmax = max(xmax, float(np.max(v)))
            _say(name, f"{t} {LABEL[m]} TV at best held-out loss, {len(v)} seeds", v, src)
            chk(f"{t} {LABEL[m]} median", med, MEDIAN_TV_BY_METHOD[m][TARGETS.index(t)], decimals=3)
        ax.set_title(TARGET_LABEL[t], fontsize=FS + 1, pad=4)
        ax.set_yticks(ys)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        _despine(ax)
        ax.grid(axis="x", alpha=0.25, ls="--", lw=0.5)
    for ax in axes:
        ax.set_xlim(0, 1.08 * xmax)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
    Wf, Hf = fig.get_size_inches()
    fig.text((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2, (cap_h + 0.09) / Hf,
             "TV distance to the target (lower is better)", ha="center", va="bottom", fontsize=FS)
    axes[0].set_yticklabels([f"{LABEL[m]} ({SEEDS_BY_METHOD[m]})" for m in methods], fontsize=FS - 0.5)
    for lab in axes[0].get_yticklabels():
        if lab.get_text().startswith("Sivaprasad"):
            lab.set_url(CITE_URL["Sivaprasad et al. (2021)"])
            lab.set_color(LINK_COLOR)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)


def fig_exp_map_tuned_1d(out_dir: str) -> None:
    name = "exp_map_tuned_1d"
    chk, rng = Check(name), np.random.default_rng(1)
    with open(EXP_MAP) as f:
        rep = json.load(f)
    methods = ["direct_exp", "lift_exp", "lift"]
    rec_key = {"direct_exp": "direct_exp", "lift_exp": "hypernet_exp", "lift": "hypernet"}
    seeds = [1, 2, 3, 4, 5]

    W, margin = 7.0, 0.1
    lines, cap_h = _caption_block(name, W - 2 * margin)
    fig = plt.figure(figsize=(W, 1.55 + 1.45 + 0.52 + 0.74 + 0.55 + cap_h))
    rows = _axes_grid(fig, 2, 4, left=0.55, right=0.08, top=0.52, bottom=cap_h + 0.4, wsp=0.5, hsp=0.62)
    short = {"direct_exp": "direct,\nexp. map", "lift_exp": "lift,\nexp. map", "lift": "lift"}
    for j, t in enumerate(TARGETS):
        ax_top, ax_bot = rows[0][j], rows[1][j]
        best_direct = []
        for i, m in enumerate(methods):
            k = rec_key[m]
            npz_path = os.path.join(EXP_MAP_PINNED, f"B_{k}_{t}.npz")
            npz = np.load(npz_path)
            step = rep["selected"][k][t]["lr"]
            if step != STEP_EXP_MAP[m][j]:
                raise RuntimeError(f"{t} {m}: selected step {step}, expected {STEP_EXP_MAP[m][j]}")
            tvs, curves, grid = [], [], None
            for s in seeds:
                run = rep["runs"][t][k][str(s)]
                if run["lost"]:
                    raise RuntimeError(f"{t} {k} seed {s}: run is {run}")
                its = np.asarray(npz[f"seed{s}/tv_hist_iters"], dtype=float)
                tvh = np.asarray(npz[f"seed{s}/tv_hist_pinned"], dtype=float)
                nll = np.asarray(npz[f"seed{s}/nll_hist_pinned"], dtype=float)
                hit = np.flatnonzero(its == float(run["iter"]))
                if hit.size != 1:
                    raise RuntimeError(f"{t} {k} seed {s}: best iterate {run['iter']} not stored once")
                tv = float(tvh[hit[0]])
                if abs(tv - run["tv"]) > 1e-5:
                    raise RuntimeError(f"{t} {k} seed {s}: pinned TV {tv} differs from the report's {run['tv']}")
                tvs.append(tv)
                keep = (its >= GUARD_1D) & (np.mod(its, 200) == 0)
                if grid is None:
                    grid = its[keep]
                elif not np.array_equal(grid, its[keep]):
                    raise RuntimeError(f"{t} {k} seed {s}: evaluation iterations differ across seeds")
                curves.append(nll[keep])
                _say(name, f"{t} {LABEL[m]} seed {s} held-out loss at iterations {its[keep].astype(int).tolist()}",
                     nll[keep], npz_path)
                if m == "direct_exp":
                    best_direct.append(float(np.min(nll[keep])))
            stack = np.stack(curves)
            med_curve = np.median(stack, axis=0)
            lo_curve, hi_curve = np.percentile(stack, 10, axis=0), np.percentile(stack, 90, axis=0)
            ax_bot.fill_between(grid, lo_curve, hi_curve, color=COL[m], alpha=0.18, lw=0, zorder=2 - i * 0.1)
            ax_bot.plot(grid, med_curve, color=COL[m], lw=1.4, ls=DASH[m], zorder=4 - i * 0.1)
            _say(name, f"{t} {LABEL[m]} held-out loss, median over seeds {seeds}", med_curve, npz_path)
            _say(name, f"{t} {LABEL[m]} held-out loss, 10th percentile", lo_curve, npz_path)
            _say(name, f"{t} {LABEL[m]} held-out loss, 90th percentile", hi_curve, npz_path)
            med = _strip(ax_top, i, tvs, COL[m], rng, filled=(m == "lift"), med_w=0.24)
            _say(name, f"{t} {LABEL[m]} step {step} TV at best held-out loss, seeds {seeds}", tvs, npz_path)
            chk(f"{t} {LABEL[m]} median", med, MEDIAN_TV_EXP_MAP[m][j], decimals=3)
        level = min(best_direct)
        _say(name, f"{t} lowest held-out loss of any direct run", level, EXP_MAP)
        crossings = []
        lift_exp_path = os.path.join(EXP_MAP_PINNED, f"B_hypernet_exp_{t}.npz")
        for s in seeds:
            npz = np.load(lift_exp_path)
            its = np.asarray(npz[f"seed{s}/tv_hist_iters"], dtype=float)
            nll = np.asarray(npz[f"seed{s}/nll_hist_pinned"], dtype=float)
            hit = np.flatnonzero((its >= GUARD_1D) & (nll < level))
            crossings.append(int(its[hit[0]]) if hit.size else -1)
        _say(name, f"{t} lift, exp. map: first iteration below the best direct run, seeds {seeds}", crossings,
             lift_exp_path)
        if min(crossings) < 0 or max(crossings) > 600:
            chk.bad.append(f"{t}: a lift run is not below the best direct run by iteration 600 ({crossings})")
        ax_top.set_title(TARGET_LABEL[t], fontsize=FS + 1, pad=4)
        ax_top.set_xticks(range(3))
        ax_top.set_xticklabels([f"{short[m]}\n{STEP_TEX[STEP_EXP_MAP[m][j]]}" for m in methods], fontsize=FS - 1)
        ax_top.set_xlim(-0.3, 2.3)
        ax_top.set_ylim(0, 0.17)
        ax_top.set_yticks([0, 0.05, 0.10, 0.15])
        ax_bot.set_xlim(GUARD_1D, 3000)
        ax_bot.set_xticks([200, 1000, 2000, 3000])
        ax_bot.set_xlabel("iteration", fontsize=FS, labelpad=1)
        for ax in (ax_top, ax_bot):
            _despine(ax)
    rows[0][0].set_ylabel("TV distance\n(lower is better)", fontsize=FS)
    rows[1][0].set_ylabel("held-out loss\n(lower is better)", fontsize=FS)
    handles = [Line2D([], [], color=COL[m], lw=1.2, ls=DASH[m], marker="o", ms=3.5, mew=0.8,
                      mfc=COL[m] if m == "lift" else "white", mec=COL[m]) for m in methods]
    fig.legend(handles, [LABEL[m] for m in methods],
               fontsize=FS - 0.5, frameon=False, loc="upper center", ncol=3, handlelength=2.6,
               columnspacing=1.6, bbox_to_anchor=(0.53, 1.0))
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)


def fig_shoulder_thresholds_1d(out_dir: str) -> None:
    name = "shoulder_thresholds_1d"
    chk = Check(name)
    cache_path = os.path.join(ANALYSIS, "occupancy_thresholds.json")
    cache = _thresholds_cache()
    floor = 1e-7
    brk = np.sqrt(floor * 1e-5)
    W, margin = 7.0, 0.1
    lines, cap_h = _caption_block(name, W - 2 * margin)
    fig = plt.figure(figsize=(W, 1.35 + 1.35 + 0.48 + 0.4 + 0.42 + cap_h))
    rows = _axes_grid(fig, 2, 4, left=0.55, right=0.08, top=0.48, bottom=cap_h + 0.42, wsp=0.3, hsp=0.4)
    x = np.arange(3)
    for j, t in enumerate(TARGETS):
        for arm in ("direct", "lift"):
            rws = _gallery_rows(cache, t, arm)
            occ = np.array([[r["at"][k]["occ_last"] for k in PSI_KEYS] for r in rws])
            ext = np.array([[r["at"][k]["exit_rate"] for k in PSI_KEYS] for r in rws])
            for ax, arr, what, stat in ((rows[0][j], occ, "share of weights on the shoulder at the end",
                                         "occ_last"),
                                        (rows[1][j], ext, "rate at which a visit ends", "exit_rate")):
                med, lo, hi = np.median(arr, 0), np.percentile(arr, 10, 0), np.percentile(arr, 90, 0)
                _say(name, f"{t} {LABEL[arm]} {what} at thresholds {PSI_LABEL}, median", med, cache_path)
                _say(name, f"{t} {LABEL[arm]} {what}, 10th percentile", lo, cache_path)
                _say(name, f"{t} {LABEL[arm]} {what}, 90th percentile", hi, cache_path)
                if stat == "exit_rate":
                    med_d = np.maximum(med, floor)
                    lo_band, hi_band = np.maximum(lo, brk), np.maximum(hi, brk)
                else:
                    med_d, lo_band, hi_band = med, lo, hi
                ax.scatter(x, med_d, s=14, color=COL[arm], zorder=5, clip_on=False)
                above = med_d > floor if stat == "exit_rate" else np.ones(x.size, dtype=bool)
                runs = np.split(np.arange(x.size), np.flatnonzero(np.diff(above.astype(int))) + 1)
                for r in runs:
                    if above[r[0]] and r.size > 1:
                        ax.plot(x[r], med_d[r], color=COL[arm], lw=1.4, ls=DASH[arm], zorder=4)
                ax.fill_between(x, lo_band, hi_band, color=COL[arm], alpha=0.18, lw=0, zorder=2)
                tab_i = 0 if stat == "occ_last" else 1
                for k_i, lab in enumerate(PSI_LABEL):
                    theirs = SHARE_EXIT_BY_THRESHOLD[t][arm][tab_i][k_i]
                    if stat == "occ_last":
                        chk(f"{t} {LABEL[arm]} share at {lab}", float(med[k_i]), theirs, decimals=2)
                    else:
                        chk(f"{t} {LABEL[arm]} exit rate at {lab}", float(med[k_i]), theirs, rel=0.06)
            if t == "gumbel":
                for k_i, lab in enumerate(PSI_LABEL):
                    chk(f"Gumbel {LABEL[arm]} end share at {lab}", float(np.median(occ[:, k_i])),
                        GUMBEL_END_SHARE[arm][k_i], decimals=2)
        rows[0][j].set_title(TARGET_LABEL[t], fontsize=FS + 1, pad=4)
        rows[0][j].set_ylim(0, 1.05)
        rows[0][j].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        rows[1][j].set_yscale("log")
        rows[1][j].set_ylim(floor / 1.5, 0.3)
        rows[1][j].set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
        rows[1][j].set_yticklabels(["0", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"])
        rows[1][j].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        _axis_break(rows[1][j], brk)
        for ax in (rows[0][j], rows[1][j]):
            ax.axvline(1, color="0.82", lw=0.9, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels(PSI_LABEL)
            ax.set_xlim(-0.3, 2.3)
            _despine(ax)
    Wf, Hf = fig.get_size_inches()
    fig.text((rows[1][0].get_position().x0 + rows[1][3].get_position().x1) / 2, (cap_h + 0.14) / Hf,
             "threshold on the softplus derivative", ha="center", va="bottom", fontsize=FS)
    rows[0][0].set_ylabel("share of weights on the\nshoulder at the end", fontsize=FS)
    rows[1][0].set_ylabel("rate a visit ends,\nper snapshot interval", fontsize=FS)
    handles = [Line2D([], [], color=COL[a], lw=1.4, ls=DASH[a], marker="o", ms=3.5) for a in ("direct", "lift")]
    x_mid = (rows[0][0].get_position().x0 + rows[0][3].get_position().x1) / 2
    fig.legend(handles, [LABEL["direct"], LABEL["lift"]], fontsize=FS - 0.5, frameon=False, loc="upper center",
               ncol=2, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(x_mid, 1.0))
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)


def fig_shoulder_entry_exit(out_dir: str) -> None:
    name = "shoulder_entry_exit"
    chk = Check(name)
    cache_path = os.path.join(ANALYSIS, "occupancy_thresholds.json")
    cache = _thresholds_cache()
    targets = ("gumbel", "beta")
    series = _occupancy_series(cache, targets)
    W, margin = 7.0, 0.1
    lines, cap_h = _caption_block(name, W - 2 * margin)
    fig = plt.figure(figsize=(W, 1.6 + 0.56 + 0.45 + cap_h))
    axes = _axes_grid(fig, 1, 4, left=0.5, right=0.08, top=0.56, bottom=cap_h + 0.45, wsp=(0.55, 0.5, 0.55),
                      hsp=0, widths=[1.4, 1.0, 1.4, 1.0])[0]
    floor = 1e-6
    rates_drawn: dict[tuple, tuple] = {}
    for j, t in enumerate(targets):
        ax_s, ax_r = axes[2 * j], axes[2 * j + 1]
        for arm in ("direct", "lift"):
            per_seed = series[t][arm]
            iters = np.asarray(per_seed["0"]["iters"], dtype=float)
            occ = np.array([per_seed[str(s)]["occ"] for s in range(30)])
            med, lo, hi = np.median(occ, 0), np.percentile(occ, 10, 0), np.percentile(occ, 90, 0)
            ax_s.plot(iters, med, color=COL[arm], lw=1.4, ls=DASH[arm], zorder=4)
            ax_s.fill_between(iters, lo, hi, color=COL[arm], alpha=0.18, lw=0, zorder=2)
            _say(name, f"{t} {LABEL[arm]} share on the shoulder at iterations {iters.astype(int).tolist()}, median",
                 med, SERIES_CACHE)
            _say(name, f"{t} {LABEL[arm]} share, 10th percentile", lo, SERIES_CACHE)
            _say(name, f"{t} {LABEL[arm]} share, 90th percentile", hi, SERIES_CACHE)
            rws = _gallery_rows(cache, t, arm)
            entry_all = float(np.median([r["at"]["psi0.05"]["entry_rate"] for r in rws]))
            leave_all = float(np.median([r["at"]["psi0.05"]["exit_rate"] for r in rws]))
            _say(name, f"{t} {LABEL[arm]} entering and leaving rate over all intervals, median",
                 [entry_all, leave_all], cache_path)
            after = _rates_after_first(cache, t, arm)
            entry = float(np.median([a[0] for a in after]))
            leave = float(np.median([a[1] for a in after]))
            _say(name, f"{t} {LABEL[arm]} entering and leaving rate after the first interval, median over "
                       f"{len(after)} seeds (drawn)", [entry, leave], RATES_CACHE)
            rates_drawn[(t, arm)] = (entry, leave)
            off = -0.19 if arm == "direct" else 0.19
            ax_r.scatter(np.arange(2) + off, [max(entry, floor), max(leave, floor)], s=24, color=COL[arm],
                         zorder=4, clip_on=False)
            g = SHARE_ENTRY_EXIT[(t, arm)]
            chk(f"{t} {LABEL[arm]} share at the start", float(med[0]), g[0], decimals=2)
            chk(f"{t} {LABEL[arm]} share at the end", float(med[-1]), g[1], decimals=2)
            chk(f"{t} {LABEL[arm]} entering, all intervals", entry_all, g[2], rel=0.06)
            chk(f"{t} {LABEL[arm]} leaving, all intervals", leave_all, g[3], rel=0.06)
        ax_s.set_xlim(0, 3000)
        ax_s.set_xticks([0, 1000, 2000, 3000])
        ax_s.set_ylim(0, 1.03)
        ax_s.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_s.set_xlabel("iteration", fontsize=FS, labelpad=1)
        ax_r.set_yscale("log")
        ax_r.set_ylim(floor, 2.0)
        ax_r.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
        ax_r.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax_r.set_xticks([0, 1])
        ax_r.set_xticklabels(["entering", "leaving"])
        ax_r.set_xlim(-0.5, 1.5)
        ax_r.grid(axis="y", alpha=0.25, ls="--", lw=0.5)
        for ax in (ax_s, ax_r):
            _despine(ax)
    for t in targets:
        (ed, ld), (el, ll) = rates_drawn[(t, "direct")], rates_drawn[(t, "lift")]
        print(f"  [{name}] {t}: after the first interval the lift enters {el / ed:.0f} times as often as direct "
              f"softplus and leaves {ll / ld:.2f} times as often")
        if not el > ed:
            chk.bad.append(f"{t}: the entering order flips after the first interval")
    axes[0].set_ylabel("share of weights\non the shoulder", fontsize=FS)
    axes[1].set_ylabel("rate per snapshot interval", fontsize=FS)
    axes[3].set_ylabel("rate per snapshot interval", fontsize=FS)
    Wf, Hf = fig.get_size_inches()
    for j, t in enumerate(targets):
        x0, x1 = axes[2 * j].get_position().x0, axes[2 * j + 1].get_position().x1
        y_rule = axes[0].get_position().y1 + 0.1 / Hf
        fig.add_artist(Line2D([x0, x1], [y_rule, y_rule], transform=fig.transFigure, lw=0.6, color="0.3"))
        fig.text((x0 + x1) / 2, y_rule + 0.03 / Hf, TARGET_LABEL[t], ha="center", va="bottom", fontsize=FS + 1)
    handles = [Line2D([], [], color=COL[a], lw=1.4, ls=DASH[a], marker="o", ms=4) for a in ("direct", "lift")]
    x_mid = (axes[0].get_position().x0 + axes[3].get_position().x1) / 2
    fig.legend(handles, [LABEL["direct"], LABEL["lift"]], fontsize=FS - 0.5, frameon=False, loc="upper center",
               ncol=2, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(x_mid, 1.0))
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)


def _darcy_anchor_spread() -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    """Per seed: the shift of the held-out NLL under resampled batches, and its spread."""
    dirs = []
    for tag in ("6641e617", "770fbc2a"):
        hits = sorted(glob.glob(os.path.join(CK, f"darcy_improve_*{tag}*", "results.json")))
        if len(hits) != 1:
            raise FileNotFoundError(f"darcy_improve_*{tag}*/results.json: {len(hits)} matches")
        dirs.append(hits[0])
    seeds, shift, sd = [], [], []
    for p in dirs:
        with open(p) as f:
            recs = json.load(f)
        for r in sorted((r for r in recs if r["backend"] == "hypernet"), key=lambda r: r["seed"]):
            a = r["anchor_spread"]
            seeds.append(int(r["seed"]))
            shift.append(a["resampled_mean"] - a["full_split"])
            sd.append(a["resampled_std"])
    if sorted(seeds) != list(range(10)):
        raise RuntimeError(f"Darcy lift seeds {seeds}, expected 0 to 9")
    return seeds, np.array(shift), np.array(sd), dirs


def fig_reference_set_1d(out_dir: str) -> None:
    name = "reference_set_1d"
    chk, rng = Check(name), np.random.default_rng(5)
    zpath = os.path.join(ANALYSIS, "convergence_1d_pinned.npz")
    z = np.load(zpath, allow_pickle=True)
    W, margin = 7.0, 0.1
    lines, cap_h = _caption_block(name, W - 2 * margin)
    ph, gap, left, bottom = 1.25, 0.18, 0.62, cap_h + 0.55
    fig = plt.figure(figsize=(W, ph + 0.62 + bottom))
    H = fig.get_size_inches()[1]
    pw = 1.05
    axes = [fig.add_axes([(left + j * (pw + gap)) / W, bottom / H, pw / W, ph / H]) for j in range(4)]
    x_last = left + 4 * (pw + gap) + 0.72
    axes.append(fig.add_axes([x_last / W, bottom / H, (W - x_last - 0.08) / W, ph / H]))
    for j, t in enumerate(TARGETS):
        ax = axes[j]
        jpath = os.path.join(ANALYSIS, "pinned", f"lcmm_1d_poolnorm_{t}.json")
        with open(jpath) as f:
            seeds = json.load(f)["seeds"]
        if len(seeds) != 30:
            raise RuntimeError(f"{t}: {len(seeds)} pinned seeds, expected 30")
        order = sorted(seeds, key=int)
        pin = np.array([seeds[s]["pinned"]["tv"] for s in order])
        bat = np.array([seeds[s]["batch"]["tv"] for s in order])
        direct_final = np.asarray(z[f"{t}/direct/tv"], dtype=float)[:, -1]
        it_last = float(np.asarray(z[f"{t}/direct/it"], dtype=float)[-1])
        _say(name, f"{t} lift final TV reading the fixed sample, seeds {order}", pin, jpath)
        _say(name, f"{t} lift final TV reading one batch of 256, same seeds", bat, jpath)
        _say(name, f"{t} direct softplus TV at the final evaluation (iteration {it_last:.0f}), 30 seeds",
             direct_final, zpath)
        for a, b in zip(pin, bat):
            ax.plot([0.12, 0.88], [a, b], color=COL["lift"], lw=0.4, alpha=0.35, zorder=2)
        m_pin = _strip(ax, 0, pin, COL["lift"], rng, filled=True, jitter=0.1, med_w=0.24)
        m_bat = _strip(ax, 1, bat, COL["lift"], rng, filled=False, jitter=0.1, med_w=0.24)
        dm = _strip(ax, 2, direct_final, COL["direct"], rng, filled=True, jitter=0.1, med_w=0.24)
        chk(f"{t} lift, fixed sample, median", m_pin, REFERENCE_SET["pinned"][j], decimals=3)
        chk(f"{t} lift, one batch, median", m_bat, REFERENCE_SET["batch"][j], decimals=3)
        chk(f"{t} direct softplus, median", dm, REFERENCE_SET["direct"][j], decimals=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["lift,\nfixed\nsample", "lift,\none\nbatch", "direct\nsoftplus"], fontsize=FS - 1)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 0.32)
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        if float(np.max(direct_final)) > 0.32 or float(np.max(np.r_[pin, bat])) > 0.32:
            chk.bad.append(f"{t}: a seed falls above the axis")
        ax.set_title(TARGET_LABEL[t], fontsize=FS + 1, pad=4)
        _despine(ax)
        if j:
            ax.set_yticklabels([])
    axes[0].set_ylabel("TV distance to the target\nat the final iterate", fontsize=FS)

    ax = axes[4]
    seeds, shift, sd, dirs = _darcy_anchor_spread()
    for p in dirs:
        _say(name, "transport map, held-out NLL shift (batch mean minus whole split) per seed", shift, p)
        _say(name, "transport map, standard deviation across the resampled batches per seed", sd, p)
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.errorbar(seeds, shift, yerr=sd, fmt="o", ms=3.2, color=COL["lift"], ecolor=COL["lift"], elinewidth=0.8,
                capsize=1.5, zorder=4)
    worst_shift, worst_sd = float(np.max(np.abs(shift))), float(np.max(sd))
    ok1 = worst_shift <= REFERENCE_SET["darcy_max_mean_shift"]
    ok2 = worst_sd <= REFERENCE_SET["darcy_max_sd"]
    print(f"  [check {name}] {'transport map, largest |mean shift| over seeds':52s} "
          f"{worst_shift:.5f} <= {REFERENCE_SET['darcy_max_mean_shift']}  {'ok' if ok1 else 'DISAGREES'}")
    print(f"  [check {name}] {'transport map, largest sd over seeds':52s} "
          f"{worst_sd:.5f} <= {REFERENCE_SET['darcy_max_sd']}  {'ok' if ok2 else 'DISAGREES'}")
    if not (ok1 and ok2):
        chk.bad.append("transport map")
    ax.set_xticks([0, 3, 6, 9])
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(-1.2e-3, 1.2e-3)
    ax.set_yticks([-1e-3, -5e-4, 0, 5e-4, 1e-3])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
    ax.set_title("transport map", fontsize=FS + 1, pad=4)
    ax.set_xlabel("seed\n(mean and spread\nof seven batches)", fontsize=FS - 1, labelpad=1)
    ax.set_ylabel("held-out loss, one batch\nminus full training set\n(thousandths of a nat)", fontsize=FS - 1)
    _despine(ax)
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)

def _toy_flow_shares(target: str) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    """Direct softplus on one toy target, seeds 0 to 4, preferring the archive with more seeds."""
    pat = os.path.join(CK, f"experiments_cpflow_toy_published_backend-published_recipe-B_target-{target}_*")
    dirs = [d for d in sorted(glob.glob(pat)) if os.path.exists(os.path.join(d, "metrics.json"))]
    blobs = {}
    for d in dirs:
        with open(os.path.join(d, "metrics.json")) as f:
            blobs[d] = json.load(f)
    dirs.sort(key=lambda d: -len(blobs[d]["per_seed"]))
    picked: dict[int, tuple] = {}
    for d in dirs:
        for s, rec in blobs[d]["per_seed"].items():
            picked.setdefault(int(s), (d, rec))
    seeds = [s for s in sorted(picked) if s < 5]
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"{target}: direct seeds {seeds}, expected 0 to 4")
    start = np.array([picked[s][1]["softplus_shoulder_at_init"]["fraction"] for s in seeds])
    end = np.array([picked[s][1]["softplus_shoulder_at_end"]["fraction"] for s in seeds])
    return seeds, start, end, sorted({os.path.join(picked[s][0], "metrics.json") for s in seeds})


def fig_direct_share_by_setting(out_dir: str) -> None:
    name = "direct_share_by_setting"
    chk, rng = Check(name), np.random.default_rng(2)
    cache_path = os.path.join(ANALYSIS, "occupancy_thresholds.json")
    cache = _thresholds_cache()
    entries: list[dict] = []
    for t in TARGETS:
        rws = _gallery_rows(cache, t, "direct")
        entries.append(dict(label=TARGET_LABEL[t], group="one-dimensional targets",
                            start=np.array([r["at"]["psi0.05"]["occ_first"] for r in rws]),
                            end=np.array([r["at"]["psi0.05"]["occ_last"] for r in rws]), src=cache_path,
                            check=("1d", t)))
    tpath = os.path.join(ANALYSIS, "tabular_multiseed.npz")
    tab = np.load(tpath, allow_pickle=True)
    for ds, lab in (("power", "POWER"), ("hepmass", "HEPMASS"), ("miniboone", "MiniBooNE")):
        seeds = [int(s) for s in np.atleast_1d(tab[f"{ds}/direct/seeds"])]
        test_nll = np.array([float(tab[f"{ds}/direct/s{s}/test"]) for s in seeds])
        bad = [k for k, v in enumerate(test_nll) if not np.isfinite(v) or v < -1e3]
        _say(name, f"{lab} direct softplus test NLL, seeds {seeds} (crossed where non-physical)", test_nll, tpath)
        entries.append(dict(label=lab, group="tabular energy-based models",
                            start=np.array([float(tab[f"{ds}/direct/s{s}/shoulder0"]) for s in seeds]),
                            end=np.array([float(tab[f"{ds}/direct/s{s}/shoulder"]) for s in seeds]), src=tpath,
                            check=(ds,), crossed=bad))
    import render_ffhq_resilience as RF
    sel = RF._select(RF._arms(), "direct", lr=RF.LR_PUB)
    entries.append(dict(label="image\nregularizer", group="image",
                        start=np.array([r["hist"]["w_frac_shoulder"][0] for r in sel]),
                        end=np.array([r["hist"]["w_frac_shoulder"][-1] for r in sel]),
                        src=f"data/checkpoints/*meanmatch*/direct_seed*/metrics.json (lr {RF.LR_PUB:g}, "
                            f"{len(sel)} seeds, render_ffhq_resilience._select)", check=("ffhq",)))
    for tgt, lab in (("eight_gaussians", "eight\nGaussians"), ("two_spirals", "two\nspirals"),
                     ("one_moon", "one\nmoon")):
        _, start, end, srcs = _toy_flow_shares(tgt)
        entries.append(dict(label=lab, group="convex potential flows", start=start, end=end,
                            src="; ".join(_rel(s) for s in srcs), check=("flows", tgt)))
    hits = sorted(glob.glob(os.path.join(CK, "experiments_pcpmap_darcy_*785a2e5d*", "results.json")))
    if len(hits) != 1:
        raise FileNotFoundError(f"experiments_pcpmap_darcy_*785a2e5d*/results.json: {len(hits)} matches")
    with open(hits[0]) as f:
        recs = sorted((r for r in json.load(f) if r["backend"] == "direct"), key=lambda r: r["seed"])
    entries.append(dict(label="transport\nmap", group="transport map",
                        start=np.array([float(r["shoulder_occupancy_init"]) for r in recs]),
                        end=np.array([float(r["shoulder_occupancy"]) for r in recs]), src=hits[0],
                        check=("darcy",)))

    W, margin = 7.0, 0.1
    lines, cap_h = _caption_block(name, W - 2 * margin)
    fig = plt.figure(figsize=(W, 2.0 + 0.66 + 0.62 + cap_h))
    ax = _axes_grid(fig, 1, 1, left=0.55, right=0.08, top=0.66, bottom=cap_h + 0.62, wsp=0, hsp=0)[0][0]
    xs, group_span, group_n = [], {}, {}
    x = 0.0
    prev_group = None
    for e in entries:
        slot = 1.35 if e["group"] == "tabular energy-based models" else 1.0
        if prev_group is not None and e["group"] != prev_group:
            x += 0.5
        x += (slot - 1.0) / 2
        xs.append(x)
        group_span.setdefault(e["group"], [x, x])[1] = x
        group_n[e["group"]] = group_n.get(e["group"], 0) + 1
        prev_group = e["group"]
        x += 1.0 + (slot - 1.0) / 2
    for x, e in zip(xs, entries):
        n = len(e["start"])
        _say(name, f"{e['label'].replace(chr(10), ' ')} direct softplus share on the shoulder at the start, {n} seeds",
             e["start"], e["src"])
        _say(name, f"{e['label'].replace(chr(10), ' ')} direct softplus share on the shoulder at the end, {n} seeds",
             e["end"], e["src"])
        ms = _strip(ax, x - 0.2, e["start"], COL["direct"], rng, filled=False, jitter=0.1, med_w=0.17)
        crossed = e.get("crossed", [])
        keep = np.array([k not in crossed for k in range(len(e["end"]))])
        me = float(np.median(e["end"]))
        _strip(ax, x + 0.2, e["end"][keep], COL["direct"], rng, filled=True, jitter=0.1, med_w=0.0)
        ax.plot([x + 0.2 - 0.17, x + 0.2 + 0.17], [me, me], color=COL["direct"], lw=1.6, zorder=4,
                solid_capstyle="butt")
        if crossed:
            ax.scatter(np.full(len(crossed), x + 0.2), e["end"][~keep], marker="x", s=16, color=COL["direct"],
                       linewidths=0.9, zorder=5)
        c = e["check"]
        lab = e["label"].replace(chr(10), " ")
        if c[0] == "1d":
            chk(f"{lab} start", ms, DIRECT_SHARE["1d_start"], decimals=2)
            lo_end, hi_end = DIRECT_SHARE["1d_end"]
            inside = lo_end - 0.005 <= me <= hi_end + 0.005
            print(f"  [check {name}] {lab + ' end within the reported range':52s} "
                  f"{me:.4f} in [{lo_end}, {hi_end}]  {'ok' if inside else 'DISAGREES'}")
            if not inside:
                chk.bad.append(f"{lab} end")
        elif c[0] == "flows":
            chk(f"{lab} start", ms, DIRECT_SHARE["flows"][1], decimals=2)
            chk(f"{lab} end", me, DIRECT_SHARE["flows"][2], decimals=2)
            if n != DIRECT_SHARE["flows"][0]:
                chk.bad.append(f"{lab} seeds {n}")
            if float(np.max(e["end"])) > 0.01:
                chk.bad.append(f"{lab}: a seed ends above 0.01")
        else:
            n_rep, s_rep, e_rep = DIRECT_SHARE[c[0]]
            chk(f"{lab} start", ms, s_rep, decimals=2)
            chk(f"{lab} end", me, e_rep, decimals=2)
            if n != n_rep:
                print(f"  [check {name}] {lab + ' seed count':52s} {n} vs {n_rep}  DISAGREES")
                chk.bad.append(f"{lab} seeds")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e['label']}\n({len(e['start'])})" for e in entries], fontsize=FS - 1)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of weights on the shoulder", fontsize=FS)
    ax.set_title("direct softplus", fontsize=FS + 1, pad=30)
    for g, (a, b) in group_span.items():
        if group_n[g] < 2:
            continue
        ax.annotate("", xy=(a - 0.35, 1.1), xytext=(b + 0.35, 1.1), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3", shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        ax.text((a + b) / 2, 1.12, g, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=FS - 0.5, color="0.2")
    ax.legend([Line2D([], [], marker="o", ls="", mfc="white", mec=COL["direct"], ms=3.5),
               Line2D([], [], marker="o", ls="", color=COL["direct"], ms=3.5),
               Line2D([], [], marker="x", ls="", color=COL["direct"], ms=4.0, mew=0.9)],
              ["start of training", "end of training", "likelihood turns non-physical"], fontsize=FS - 1,
              frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55), handlelength=1.2)
    _despine(ax)
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)


def fig_cifar10_sivaprasad(out_dir: str) -> None:
    """The figure of ``render_sivaprasad_cifar``, kept open instead of saved, with the caption added."""
    name = "cifar10_sivaprasad"
    chk = Check(name)
    import render_sivaprasad_cifar as RS
    cache = os.path.join(ANALYSIS, "sivaprasad_cifar.json")
    with open(cache) as f:
        table = json.load(f)["table"]
    for m in ("lift", "direct", "pgd", "exppost"):
        per = table[m]["per_seed"]
        acc = np.array([100.0 * r["test_acc_at_best_val_acc"] for r in per.values()])
        _say(name, f"{m} test accuracy at the best validation epoch (%), seeds {sorted(int(s) for s in per)}, "
                   f"step {table[m]['lr']:g}", acc, cache)
        chk(f"{LABEL.get(m, m)} median test accuracy", float(np.median(acc)), CIFAR_TEST_ACCURACY[m], decimals=1)
    import tempfile
    saved_savefig, saved_close = matplotlib.figure.Figure.savefig, plt.close
    matplotlib.figure.Figure.savefig = lambda self, *a, **k: None
    plt.close = lambda *a, **k: None
    try:
        with tempfile.TemporaryDirectory(prefix="cifar_render_") as scratch:
            RS.main(["--cache", cache, "--out_dir", scratch])
    finally:
        matplotlib.figure.Figure.savefig, plt.close = saved_savefig, saved_close
    nums = plt.get_fignums()
    if len(nums) != 1:
        raise RuntimeError(f"expected one open figure from the CIFAR renderer, found {len(nums)}")
    fig = plt.figure(nums[0])
    by_title = {ax.get_title(): ax for ax in fig.axes}
    per_epoch_test, summary = by_title["test accuracy (%)"], by_title["test accuracy at the best validation epoch (%)"]
    x0 = per_epoch_test.get_position().x0
    fig.delaxes(per_epoch_test)
    p = summary.get_position()
    summary.set_position([x0, p.y0, p.x1 - x0, p.height])
    W, H0 = fig.get_size_inches()
    margin = 0.05
    lines, cap_h = _caption_block(name, W - 2 * margin)
    H = H0 + cap_h + 0.1
    fig.set_size_inches(W, H, forward=True)
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position([p.x0, (p.y0 * H0 + cap_h + 0.1) / H, p.width, p.height * H0 / H])
    for lg in fig.legends:
        bb = lg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        lg.set_bbox_to_anchor((bb.x0, (bb.y0 * H0 + cap_h + 0.1) / H), transform=fig.transFigure)
    for lg in list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend()]:
        for txt in lg.get_texts():
            if txt.get_text().startswith("Sivaprasad"):
                txt.set_url(CITE_URL["Sivaprasad et al. (2021)"])
                txt.set_color(LINK_COLOR)
    chk.stop_if_bad()
    _finish(fig, name, lines, margin, out_dir)


FIGURES = {
    "positivity_methods_1d": fig_positivity_methods_1d,
    "exp_map_tuned_1d": fig_exp_map_tuned_1d,
    "shoulder_thresholds_1d": fig_shoulder_thresholds_1d,
    "shoulder_entry_exit": fig_shoulder_entry_exit,
    "reference_set_1d": fig_reference_set_1d,
    "direct_share_by_setting": fig_direct_share_by_setting,
    "cifar10_sivaprasad": fig_cifar10_sivaprasad,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--figure", default="all", help="one of %s, or all" % ", ".join(FIGURES))
    p.add_argument("--out_dir", default=OUT_DIR)
    a = p.parse_args(argv)
    apply_paper_style()
    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rcParams["text.usetex"] = False
    names = list(FIGURES) if a.figure == "all" else [a.figure]
    for n in names:
        if n not in FIGURES:
            raise SystemExit(f"unknown figure {n}; choose from {', '.join(FIGURES)}")
        print(f"\n=== {n}")
        FIGURES[n](a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
