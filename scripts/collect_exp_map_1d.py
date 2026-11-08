"""The exponential map on the one-dimensional targets, with the step size selected.

Three methods: direct training under the exponential map (``direct_exp``), the
lift under it (``hypernet_exp``) and the lift under softplus (``hypernet``).
Reads the checkpoints of ``experiments_lcmm_1d_logconcave.py`` and the pinned
records of ``reeval_ebm_pinned.py``; nothing is retrained.

Selection, on seed 0 (``<checkpoints>/A_lr<step>/<target>/``): each method
takes the step with the lowest best held-out loss, over the evaluations from
iteration 200 on. A difference below 0.002 nats goes to 1e-3, a step at which
any method's run is not finite is not eligible, and a selection at 5e-4 or
2e-3 is labeled as an end of the range.

Comparison, seeds 1 to 5 at the selected steps
(``<checkpoints>/B_<method>_lr<step>/<target>/``): the total-variation distance
at the best held-out iterate, read from the pinned record
(``<pinned>/B_<method>_<target>``) for the lifts and as recorded for direct
training. A run that is not finite loses its pair, with its distance read at
its last finite evaluation or set to 1. For each run of the exponential-map
lift, the first evaluation at which its held-out NLL on the fixed sample of the
pinned records is at or below the lowest best NLL of the direct runs there.

Usage::

    python scripts/collect_exp_map_1d.py --select_only
    python scripts/collect_exp_map_1d.py
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from projorg import gitdir

STEPS = ("5e-4", "1e-3", "2e-3")
DEFAULT_STEP = "1e-3"
EDGE = {"5e-4": "lower", "2e-3": "upper"}
TIE_NATS = 0.002
GUARD = 200
SEEDS = (1, 2, 3, 4, 5)
TARGETS = ("gumbel", "laplace", "gamma", "beta")
DIRECT, LIFT, SOFTPLUS_LIFT = "direct_exp", "hypernet_exp", "hypernet"
METHODS = (DIRECT, LIFT, SOFTPLUS_LIFT)


def run_summary(hist: dict) -> dict:
    """Whether a run stayed finite, and its best held-out iterate from iteration 200 on."""
    loss = np.asarray(hist["loss_E_per_iter"], float)
    it = np.asarray(hist["eval_iters"], int)
    val = np.asarray(hist["val_loss_per_eval"], float)
    finite = bool(np.isfinite(loss).all() and hist.get("aborted_at_iter") is None
                  and it.size and np.isfinite(val).all())
    ok = np.flatnonzero(np.isfinite(val) & (it >= GUARD))
    j = int(ok[np.argmin(val[ok])]) if ok.size else None
    return {"finite": finite, "eval_iters": it.tolist(),
            "best_iter": None if j is None else int(it[j]),
            "best_val": None if j is None else float(val[j])}


def select(runs: dict) -> dict:
    """The step of each method on one target, from ``{method: {step: run_summary}}`` of seed 0."""
    eligible = {s: all(runs[m][s]["finite"] for m in runs) for s in STEPS}
    out = {}
    for m, by_step in runs.items():
        cand = {s: by_step[s]["best_val"] for s in STEPS
                if eligible[s] and by_step[s]["best_val"] is not None}
        lr = min(cand, key=cand.get) if cand else None
        if lr is not None and DEFAULT_STEP in cand and cand[DEFAULT_STEP] - cand[lr] < TIE_NATS:
            lr = DEFAULT_STEP
        out[m] = {"lr": lr, "edge": EDGE.get(lr), "eligible": eligible,
                  "best_val": {s: by_step[s]["best_val"] for s in STEPS}}
    return out


def value_at(iters, values, it: int) -> float:
    """``values`` at iteration ``it``, matched by iteration, not by position."""
    hit = np.flatnonzero(np.asarray(iters, int) == int(it))
    if hit.size != 1:
        raise ValueError(f"iteration {it} occurs {hit.size} times in {list(iters)}")
    return float(np.asarray(values, float)[hit[0]])


def tv_at_best(s: dict, iters, tvs) -> dict:
    """TV at the best held-out iterate; a run that is not finite there is lost."""
    if s["finite"] and s["best_iter"] is not None and math.isfinite(value_at(iters, tvs, s["best_iter"])):
        return {"tv": value_at(iters, tvs, s["best_iter"]), "iter": s["best_iter"], "lost": False}
    have = set(np.asarray(iters, int).tolist())
    last = [i for i in s["eval_iters"] if i in have and math.isfinite(value_at(iters, tvs, i))]
    return {"tv": value_at(iters, tvs, last[-1]) if last else 1.0,
            "iter": last[-1] if last else None, "lost": True}


def history(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"missing run: {path}")
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)["history"]


def _tail(path: str) -> list[str]:
    return os.path.normpath(path).split(os.sep)[-2:]


def pinned(stem: str, method: str, folder: str, seeds) -> tuple[dict, dict]:
    """A pinned record, refused unless it re-read ``folder`` as ``method`` and its guard passed."""
    with open(stem + ".json") as f:
        js = json.load(f)
    if js.get("backend") != method or _tail(js.get("archive", "")) != _tail(folder):
        raise SystemExit(f"{stem}.json re-read {js.get('backend')} in {js.get('archive')}, "
                         f"not {method} in {folder}")
    bad = [s for s in seeds if js["seeds"][str(s)]["guard"]["status"] != "pass"]
    if bad:
        raise SystemExit(f"{stem}.json: the replay guard did not pass on seeds {bad}")
    with np.load(stem + ".npz") as z:
        return js, {k: z[k] for k in z.files}


def selection(root: str) -> dict:
    sel = {m: {} for m in METHODS}
    for t in TARGETS:
        runs = {m: {s: run_summary(history(os.path.join(root, f"A_lr{s}", t, f"seed0_{m}.pt")))
                    for s in STEPS} for m in METHODS}
        for m, r in select(runs).items():
            sel[m][t] = r
    return sel


def report(root: str, pin: str, sel: dict) -> dict:
    runs, nll = {t: {} for t in TARGETS}, {t: {} for t in TARGETS}
    for t in TARGETS:
        for m in METHODS:
            lr = sel[m][t]["lr"]
            if lr is None:
                raise SystemExit(f"no eligible step for {m} on {t}")
            folder = os.path.join(root, f"B_{m}_lr{lr}", t)
            _, npz = pinned(os.path.join(pin, f"B_{m}_{t}"), m, folder, SEEDS)
            runs[t][m], nll[t][m] = {}, {}
            for s in SEEDS:
                hist = history(os.path.join(folder, f"seed{s}_{m}.pt"))
                summ = run_summary(hist)
                its = npz[f"seed{s}/tv_hist_iters"]
                if m == DIRECT:
                    r = tv_at_best(summ, hist["eval_iters"], hist["tv_per_eval"])
                else:
                    r = tv_at_best(summ, its, npz[f"seed{s}/tv_hist_pinned"])
                runs[t][m][str(s)] = {"lr": lr, **r}
                nll[t][m][s] = [(i, value_at(its, npz[f"seed{s}/nll_hist_pinned"], i))
                                for i in summ["eval_iters"] if i >= GUARD]
    out = {"median_tv": {}, "exp_lift_below_direct": {}, "every_exp_lift_run_below_every_direct_run": {},
           "first_below_best_direct": {}}
    for t in TARGETS:
        r = runs[t]
        out["median_tv"][t] = {m: float(np.median([v["tv"] for v in r[m].values()])) for m in METHODS}
        out["exp_lift_below_direct"][t] = sum(
            not r[LIFT][str(s)]["lost"] and (r[DIRECT][str(s)]["lost"] or r[LIFT][str(s)]["tv"] < r[DIRECT][str(s)]["tv"])
            for s in SEEDS)
        out["every_exp_lift_run_below_every_direct_run"][t] = bool(
            max(v["tv"] for v in r[LIFT].values()) < min(v["tv"] for v in r[DIRECT].values()))
        level = min(min((v for _, v in c if math.isfinite(v)), default=math.inf) for c in nll[t][DIRECT].values())
        out["first_below_best_direct"][t] = {
            "best_direct_nll": level,
            "iter": {str(s): next((i for i, v in c if v <= level), None) for s, c in nll[t][LIFT].items()}}
    out["pairs_exp_lift_below_direct"] = sum(out["exp_lift_below_direct"].values())
    out["pairs"] = len(TARGETS) * len(SEEDS)
    out["runs"] = runs
    return out


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")


def main() -> None:
    root = gitdir()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoints", default=os.path.join(root, "data", "checkpoints", "exp_map_1d"),
                   help="holds A_lr<step>/ and B_<method>_lr<step>/, the driver's --out_dir")
    p.add_argument("--pinned", default=os.path.join(root, "data", "analysis", "pinned", "exp_map_1d"),
                   help="holds B_<method>_<target>.json and .npz, the --out of reeval_ebm_pinned.py")
    p.add_argument("--selection", default=os.path.join(root, "data", "analysis", "exp_map_1d_selection.json"))
    p.add_argument("--out", default=os.path.join(root, "data", "analysis", "exp_map_1d.json"))
    p.add_argument("--select_only", action="store_true", help="write the selection and stop")
    a = p.parse_args()

    sel = selection(a.checkpoints)
    print(f"{'step size':14s} " + " ".join(f"{t:>12s}" for t in TARGETS))
    for m in METHODS:
        print(f"{m:14s} " + " ".join(f"{(sel[m][t]['lr'] or '-') + (' (end)' if sel[m][t]['edge'] else ''):>12s}"
                                     for t in TARGETS))
    if a.select_only:
        _write(a.selection, sel)
        return

    rep = {"selected": sel, **report(a.checkpoints, a.pinned, sel)}
    print(f"\n{'median TV':14s} " + " ".join(f"{t:>12s}" for t in TARGETS))
    for m in METHODS:
        print(f"{m:14s} " + " ".join(f"{rep['median_tv'][t][m]:12.4f}" for t in TARGETS))
    print(f"\nexp lift below direct_exp on {rep['pairs_exp_lift_below_direct']} of {rep['pairs']} pairs")
    for t in TARGETS:
        f = rep["first_below_best_direct"][t]
        print(f"{t:8s} every exp-lift run below every direct run: "
              f"{rep['every_exp_lift_run_below_every_direct_run'][t]}; first iteration at or below the "
              f"best direct held-out NLL: {' '.join(str(i) for i in f['iter'].values())}")
    _write(a.out, rep)


if __name__ == "__main__":
    main()
