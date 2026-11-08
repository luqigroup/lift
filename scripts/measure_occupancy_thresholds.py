r"""Shoulder share and exit rate at several thresholds, from stored snapshots.

The paper defines the shoulder once, by the gain ``psi'(theta~) =
sigmoid(theta~) <= 0.05``, which on the emitted weight is ``theta <=
softplus(logit(0.05)) = -log(1 - 0.05) = 0.0512933``. This script recomputes
the statistics of ``scripts/measure_dwell_time.py`` on a grid of thresholds,
from the same per-seed snapshot archives, with no training.

Thresholds
----------
* On the gain: ``sigma`` in {0.02, 0.05, 0.10}, i.e. ``theta <= -log(1 -
  sigma)``. For the two softplus methods, whose snapshots store
  ``theta = softplus(theta~)``, ``theta~ <= logit(sigma)`` and
  ``theta <= -log(1 - sigma)`` are the same set.
* On the constrained weight: ``theta <= e`` for ``e`` in {0.01, 0.0513, 0.1,
  0.2}. PGD has no latent and no gain, so a band on the projected weight is
  the only statistic its snapshots support; the same band is evaluated on all
  three methods. For a softplus method a weight edge ``e`` is the gain
  threshold ``sigma = 1 - exp(-e)``.
* ``theta <= 0``: PGD's exact-zero set, as in ``measure_dwell_time``.

Statistics, per run and threshold
---------------------------------
The definitions are those of ``measure_dwell_time.dwell``, and the output of
that script is reproduced at the paper's threshold before the cache is
written. ``occ_first`` / ``occ_last`` / ``mean_occ`` are the fraction of
constrained coordinates in the band at the first and last snapshot and on
average; ``exit_rate`` is the number of (coordinate, interval) pairs in the
band at the earlier snapshot and out at the later one, divided by the number
in at the earlier snapshot; ``entry_rate`` is the same for the complement;
``median_dwell_snaps`` is the median episode length in snapshots, with
episodes still running at the last snapshot counted at their observed
length.

Every statistic is on the emitted, post-positivity weight ``theta_snaps[t] =
hyper_E(snap_x)`` the 1-D lib stores for every method; the constrained set
is every ``z_layers.*.weight`` plus ``output_layer.weight``.

Usage::

    python scripts/measure_occupancy_thresholds.py
    python scripts/measure_occupancy_thresholds.py --tabular 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

from projorg import gitdir

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = gitdir()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from measure_dwell_time import (  # noqa: E402
    _CK, ARMNAME, GALLERY, MODE, TABULAR, TARGETS, THR_EMITTED, _constrained,
)

PSI_THRESHOLDS: tuple[float, ...] = (0.02, 0.05, 0.10)
THETA_EDGES: tuple[float, ...] = (0.01, 0.0513, 0.1, 0.2)
ARMS: tuple[str, ...] = ("direct", "lift", "pgd")
METHOD_LABEL = {"direct": "direct", "lift": "lift", "pgd": "PGD"}


def edge_from_psi(sigma: float) -> float:
    """``softplus(logit(sigma)) = -log(1 - sigma)``: the weight at gain sigma."""
    return -math.log1p(-sigma)


def psi_from_edge(edge: float) -> float:
    """``sigmoid(softplus^{-1}(e)) = 1 - exp(-e)``: the gain at weight e."""
    return -math.expm1(-edge)


def thresholds() -> list[dict]:
    """The threshold grid, each with its weight edge and gain value."""
    rows: list[dict] = []
    for s in PSI_THRESHOLDS:
        # The float32 edge measure_dwell_time uses, so its output is
        # reproduced to the last digit.
        edge = THR_EMITTED if s == 0.05 else edge_from_psi(s)
        rows.append({
            "key": f"psi{s:.2f}", "kind": "gain", "psi_prime": s,
            "latent": math.log(s / (1.0 - s)), "edge_theta": edge,
        })
    for e in THETA_EDGES:
        rows.append({
            "key": f"theta{e:g}", "kind": "weight", "psi_prime": psi_from_edge(e),
            "latent": math.log(math.expm1(e)), "edge_theta": float(e),
        })
    rows.append({"key": "zero", "kind": "exact_zero", "psi_prime": 0.0,
                 "latent": None, "edge_theta": 0.0})
    return rows


def dwell_at(W: np.ndarray, thr: float) -> dict[str, object]:
    """Dwell statistics of the band ``W[t] <= thr`` over the snapshot axis.

    ``W`` is ``[T, n]``: ``T`` snapshots of ``n`` constrained coordinates.
    The counting mirrors ``measure_dwell_time.dwell`` step for step.
    """
    T = W.shape[0]
    inside_all = W <= thr
    occ = inside_all.mean(axis=1)
    n_in = n_exit = n_out = n_enter = 0
    run_len = np.zeros(W.shape[1], dtype=np.int32)
    hist: dict[int, int] = {}
    prev: np.ndarray | None = None
    for t in range(T):
        inside = inside_all[t]
        if prev is not None:
            was = prev
            n_in += int(was.sum())
            n_exit += int((was & ~inside).sum())
            n_out += int((~was).sum())
            n_enter += int((~was & inside).sum())
            ended = was & ~inside
            if ended.any():
                for L, c in zip(*np.unique(run_len[ended], return_counts=True)):
                    hist[int(L)] = hist.get(int(L), 0) + int(c)
            run_len[ended] = 0
        run_len[inside] += 1
        run_len[~inside] = 0
        prev = inside
    cens: dict[int, int] = {}
    live = run_len > 0
    if live.any():
        for L, c in zip(*np.unique(run_len[live], return_counts=True)):
            cens[int(L)] = cens.get(int(L), 0) + int(c)

    def _median(h: dict[int, int]) -> float:
        n = sum(h.values())
        if not n:
            return float("nan")
        half_n = (n + 1) / 2.0
        acc = 0
        for L in sorted(h):
            acc += h[L]
            if acc >= half_n:
                return float(L)
        return float(max(h))

    allh = dict(hist)
    for L, c in cens.items():
        allh[L] = allh.get(L, 0) + c
    n_ep = sum(allh.values())
    return {
        "occ_first": float(occ[0]),
        "occ_last": float(occ[-1]),
        "mean_occ": float(occ.mean()),
        "occ_series": [float(x) for x in occ],
        "monotone_nondecreasing": bool(np.all(np.diff(occ) >= -1e-12)),
        "exit_rate": (n_exit / n_in) if n_in else float("nan"),
        "entry_rate": (n_enter / n_out) if n_out else float("nan"),
        "n_coord_visits_in": n_in,
        "n_exits": n_exit,
        "median_dwell_snaps": _median(allh),
        "n_episodes": n_ep,
        "censored_frac": (sum(cens.values()) / n_ep) if n_ep else float("nan"),
    }


def load_snapshots(path: str) -> tuple[np.ndarray, list[int]] | None:
    """Stack one archive's constrained emitted weights as ``[T, n]`` double."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    snaps = ck.get("theta_snaps")
    if not snaps:
        return None
    iters = [int(i) for i in (ck.get("snap_iters") or range(len(snaps)))]
    W = np.stack([_constrained(th).numpy() for th in snaps], axis=0)
    del ck
    return W, iters


def scan(path: str, grid: list[dict]) -> dict | None:
    """Every threshold's statistics for one archive."""
    loaded = load_snapshots(path)
    if loaded is None:
        return None
    W, iters = loaded
    rec: dict = {
        "source": path, "n_snapshots": int(W.shape[0]),
        "n_constrained": int(W.shape[1]),
        "iter_first": iters[0], "iter_last": iters[-1], "at": {},
    }
    for th in grid:
        rec["at"][th["key"]] = dwell_at(W, th["edge_theta"])
    return rec


def _nanmedian(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=float)
    return float(np.nanmedian(a)) if a.size else float("nan")


def summarize(rows: list[dict], grid: list[dict], group: str,
              ) -> tuple[dict, dict]:
    """Per-group medians over seeds and the direct-versus-lift ordering.

    ``group`` is the row key that names the problem (``target`` for the
    gallery, ``dataset`` for the tabular lines). The ordering compares the
    median over seeds and also counts seed-matched pairs, since the gallery's
    thirty seeds are paired across methods.
    """
    summary: dict = {}
    ordering: dict = {}
    groups = sorted({r[group] for r in rows}, key=lambda g: (
        TARGETS.index(g) if g in TARGETS else len(TARGETS), g))
    for g in groups:
        summary[g] = {}
        ordering[g] = {}
        by_arm = {a: [r for r in rows if r[group] == g and r["arm"] == a]
                  for a in ARMS}
        for a in ARMS:
            if not by_arm[a]:
                continue
            summary[g][a] = {}
            for th in grid:
                k = th["key"]
                stats = [r["at"][k] for r in by_arm[a]]
                summary[g][a][k] = {
                    "n_seeds": len(stats),
                    "occ_first_median": _nanmedian([s["occ_first"] for s in stats]),
                    "occ_last_median": _nanmedian([s["occ_last"] for s in stats]),
                    "mean_occ_median": _nanmedian([s["mean_occ"] for s in stats]),
                    "exit_rate_median": _nanmedian([s["exit_rate"] for s in stats]),
                    "median_dwell_snaps_median": _nanmedian(
                        [s["median_dwell_snaps"] for s in stats]),
                    "n_monotone": int(sum(s["monotone_nondecreasing"]
                                          for s in stats)),
                }
        if not (by_arm["direct"] and by_arm["lift"]):
            continue
        d_by_seed = {r["seed"]: r for r in by_arm["direct"]}
        l_by_seed = {r["seed"]: r for r in by_arm["lift"]}
        paired = sorted(set(d_by_seed) & set(l_by_seed))
        for th in grid:
            k = th["key"]
            sd, sl = summary[g]["direct"][k], summary[g]["lift"][k]
            d_occ = sl["occ_last_median"] - sd["occ_last_median"]
            d_exit = sl["exit_rate_median"] - sd["exit_rate_median"]
            n_lift_deeper = sum(
                l_by_seed[s]["at"][k]["occ_last"]
                > d_by_seed[s]["at"][k]["occ_last"] for s in paired)
            n_lift_faster = sum(
                l_by_seed[s]["at"][k]["exit_rate"]
                > d_by_seed[s]["at"][k]["exit_rate"] for s in paired)
            ordering[g][k] = {
                "ends_deeper": ("lift" if d_occ > 0 else
                                "direct" if d_occ < 0 else "tie"),
                "leaves_faster": ("lift" if d_exit > 0 else
                                  "direct" if d_exit < 0 else "tie"),
                "occ_last_median_lift_minus_direct": d_occ,
                "exit_rate_median_lift_over_direct": (
                    sl["exit_rate_median"] / sd["exit_rate_median"]
                    if sd["exit_rate_median"] > 0 else float("inf")),
                "n_pairs": len(paired),
                "n_pairs_lift_ends_deeper": int(n_lift_deeper),
                "n_pairs_lift_leaves_faster": int(n_lift_faster),
            }
    return summary, ordering


def ordering_holds(ordering: dict, grid: list[dict]) -> dict:
    """Whether each ordering is the same at every gain and weight threshold."""
    keys = [th["key"] for th in grid if th["kind"] != "exact_zero"]
    out: dict = {}
    for g, per in ordering.items():
        if not per:
            continue
        deeper = {per[k]["ends_deeper"] for k in keys}
        faster = {per[k]["leaves_faster"] for k in keys}
        out[g] = {
            "ends_deeper_same_at_every_threshold": len(deeper) == 1,
            "ends_deeper": sorted(deeper),
            "leaves_faster_same_at_every_threshold": len(faster) == 1,
            "leaves_faster": sorted(faster),
        }
    return out


def check_against_dwell_time(rows: list[dict], section: str,
                             path: str) -> dict:
    """Reproduce ``measure_dwell_time``'s output at the paper's threshold.

    The softplus methods are matched at ``psi0.05`` (edge 0.0512933), PGD at
    ``zero``; ``occ_first``, ``occ_last``, ``exit_rate`` and
    ``median_dwell_snaps`` must agree to 1e-9.
    """
    if not os.path.exists(path):
        return {"checked": 0, "note": f"{path} not found"}
    with open(path) as fh:
        ref = json.load(fh).get(section, [])
    idx = {(r["source"], r["arm"], r["seed"]): r for r in rows}
    fields = ("occ_first", "occ_last", "exit_rate", "median_dwell_snaps")
    n_checked = n_bad = 0
    worst = 0.0
    bad: list[str] = []
    for r in ref:
        mine = idx.get((r["source"], r["arm"], r["seed"]))
        if mine is None:
            continue
        key = "zero" if r["arm"] == "pgd" else "psi0.05"
        s = mine["at"][key]
        n_checked += 1
        for f in fields:
            a, b = float(r[f]), float(s[f])
            if math.isnan(a) and math.isnan(b):
                continue
            diff = abs(a - b)
            worst = max(worst, diff)
            if diff > 1e-9:
                n_bad += 1
                bad.append(f"{r.get('target', r.get('dataset'))}/{r['arm']}/"
                           f"seed{r['seed']}/{f}: {a} vs {b}")
    return {"checked": n_checked, "n_mismatch": n_bad,
            "max_abs_diff": worst, "mismatches": bad[:20]}


def print_tables(summary: dict, ordering: dict, grid: list[dict]) -> None:
    keys = [th["key"] for th in grid]
    head = "".join(f"{k:>12s}" for k in keys)
    w = max([10] + [len(g) for g in summary]) + 1
    for stat, fmt in (("occ_last_median", "{:12.3f}"),
                      ("exit_rate_median", "{:12.1e}"),
                      ("median_dwell_snaps_median", "{:12.1f}")):
        print(f"\n{stat} (median over seeds)")
        print(f"{'problem':{w}s}{'method':8s}{head}")
        for g, per_arm in summary.items():
            for a in ARMS:
                if a not in per_arm:
                    continue
                cells = "".join(fmt.format(per_arm[a][k][stat]) for k in keys)
                print(f"{g:{w}s}{METHOD_LABEL[a]:8s}{cells}")
    print("\nordering, lift versus direct (median over seeds; pairs = seeds "
          "where the lift is deeper / leaves faster)")
    print(f"{'problem':{w}s}{'':8s}{head}")
    for g, per in ordering.items():
        if not per:
            continue
        for label, who, n in (("deeper", "ends_deeper",
                               "n_pairs_lift_ends_deeper"),
                              ("faster", "leaves_faster",
                               "n_pairs_lift_leaves_faster")):
            cells = "".join(
                f"{per[k][who] + ' ' + str(per[k][n]) + '/' + str(per[k]['n_pairs']):>12s}"
                for k in keys)
            print(f"{g:{w}s}{label:8s}{cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json_out", default=os.path.join(
        _REPO, "data", "analysis", "occupancy_thresholds.json"))
    p.add_argument("--dwell_time_json", default=os.path.join(
        _REPO, "docs", "analysis", "dwell_time.json"))
    p.add_argument("--tabular", type=int, default=0,
                   help="also scan the tabular EBM snapshot archives "
                        "(HEPMASS, MiniBooNE, POWER) of measure_dwell_time")
    p.add_argument("--keep_series", type=int, default=0,
                   help="keep the per-snapshot occupancy series in the cache")
    args = p.parse_args()
    grid = thresholds()
    for th in grid:
        print(f"[threshold] {th['key']:10s} kind={th['kind']:10s} "
              f"theta <= {th['edge_theta']:.7f}  psi' = {th['psi_prime']:.4f}"
              f"  latent <= {th['latent']}")
    assert abs(edge_from_psi(0.05) - THR_EMITTED) < 1e-7

    out: dict = {
        "thresholds": grid,
        "definitions": {
            "band": "theta <= edge_theta on the emitted constrained weight",
            "constrained_set": "z_layers.*.weight and output_layer.weight",
            "occ": "fraction of constrained coordinates in the band",
            "exit_rate": "(in at t, out at t+1) pairs / (in at t) pairs, "
                         "over consecutive snapshots",
            "median_dwell_snaps": "median episode length in snapshots, "
                                  "censored episodes at observed length",
            "ordering": "lift versus direct on the median over seeds; "
                        "n_pairs_* count seed-matched pairs",
        },
        "gallery": [],
    }

    missing: list[str] = []
    for arm, root, tag, _mode in GALLERY:
        for target in TARGETS:
            paths = sorted(glob.glob(
                os.path.join(_CK, root, target, f"seed*_{tag}.pt")))
            if len(paths) != 30:
                missing.append(f"{root}/{target}: {len(paths)} of 30 files")
            for path in paths:
                r = scan(path, grid)
                if r is None:
                    missing.append(f"{path}: no theta_snaps")
                    continue
                seed = int(os.path.basename(path).split("_")[0][4:])
                r.update(arm=arm, target=target, dataset="1D-gallery", seed=seed)
                if not args.keep_series:
                    for s in r["at"].values():
                        s.pop("occ_series", None)
                out["gallery"].append(r)
                a05, a02, a10 = (r["at"]["psi0.05"], r["at"]["psi0.02"],
                                 r["at"]["psi0.10"])
                print(f"[gallery] {target:8s} {arm:7s} seed{seed:<3d}"
                      f" occ_last psi.02/.05/.10 {a02['occ_last']:.3f}/"
                      f"{a05['occ_last']:.3f}/{a10['occ_last']:.3f}"
                      f" exit {a02['exit_rate']:.1e}/{a05['exit_rate']:.1e}/"
                      f"{a10['exit_rate']:.1e}", flush=True)

    out["summary"], out["ordering"] = summarize(out["gallery"], grid, "target")
    out["ordering_holds_at_every_threshold"] = ordering_holds(
        out["ordering"], grid)
    out["reproduces_dwell_time_json"] = {
        "gallery": check_against_dwell_time(
            out["gallery"], "gallery", args.dwell_time_json)}

    if args.tabular:
        out["tabular"] = []
        for name, D, run_dir in TABULAR:
            for path in sorted(glob.glob(os.path.join(run_dir, "seed*_*.pt"))):
                base = os.path.basename(path)[:-3]
                seed_s, tag = base.split("_", 1)
                if tag not in MODE:
                    continue
                r = scan(path, grid)
                if r is None:
                    missing.append(f"{path}: no theta_snaps")
                    continue
                r.update(dataset=name, D=D, arm=ARMNAME[tag],
                         seed=int(seed_s[4:]))
                if not args.keep_series:
                    for s in r["at"].values():
                        s.pop("occ_series", None)
                out["tabular"].append(r)
                a05 = r["at"]["psi0.05"]
                print(f"[tabular] {name[:14]:15s} {r['arm']:7s} "
                      f"seed{r['seed']:<3d} occ_last(psi.05) "
                      f"{a05['occ_last']:.4f} exit {a05['exit_rate']:.2e}",
                      flush=True)
        out["tabular_summary"], out["tabular_ordering"] = summarize(
            out["tabular"], grid, "dataset")
        out["tabular_ordering_holds_at_every_threshold"] = ordering_holds(
            out["tabular_ordering"], grid)
        out["reproduces_dwell_time_json"]["tabular"] = check_against_dwell_time(
            out["tabular"], "tabular", args.dwell_time_json)

    out["missing"] = missing

    print_tables(out["summary"], out["ordering"], grid)
    if args.tabular:
        print("\n=== tabular ===")
        print_tables(out["tabular_summary"], out["tabular_ordering"], grid)
    print("\nordering holds at every gain and weight threshold:")
    for g, h in out["ordering_holds_at_every_threshold"].items():
        print(f"  {g:10s} ends_deeper {h['ends_deeper']} "
              f"same={h['ends_deeper_same_at_every_threshold']}  "
              f"leaves_faster {h['leaves_faster']} "
              f"same={h['leaves_faster_same_at_every_threshold']}")
    print("\nreproduction of dwell_time.json at the paper's threshold:",
          json.dumps(out["reproduces_dwell_time_json"]))
    if missing:
        print("\nmissing or unreadable:", *missing, sep="\n  ")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
