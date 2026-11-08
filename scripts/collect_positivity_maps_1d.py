"""Total-variation distance at the best held-out loss, per backend and target.

Reads the per-seed checkpoints ``seed<S>_<backend>.pt`` that
``experiments_lcmm_1d_logconcave.py`` writes under ``<out_dir>/<target>/`` and
prints, for each backend and target, the median over seeds of the
total-variation distance to the target at the evaluation of lowest held-out
loss, with iteration 0 excluded as in ``scripts/rederive/rederive_bestval.py``,
and at the final iterate. Nothing is retrained.

Usage::

    python scripts/collect_positivity_maps_1d.py \\
        --out_dir data/checkpoints/lcmm_1d_exppost --backends direct_exppost
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

GUARD = 200.0
TARGETS = ("gumbel", "laplace", "gamma", "beta")


def tv_at_best_val(history: dict, guard: float = GUARD) -> tuple[float, float]:
    """``(TV at the lowest held-out loss past ``guard``, TV at the last evaluation)``."""
    tv = np.asarray(history["tv_per_eval"], float)
    val = np.asarray(history["val_loss_per_eval"], float)
    it = np.asarray(history["eval_iters"], float)[: val.size]
    ok = np.isfinite(val) & (it >= guard)
    if not ok.any():
        return float("nan"), float(tv[-1]) if tv.size else float("nan")
    return float(tv[int(np.argmin(np.where(ok, val, np.inf)))]), float(tv[-1])


def _median(xs: list[float]) -> float:
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def collect(out_dir: str, backends: list[str], targets: list[str]) -> dict:
    """``{backend: {target: {"seeds", "best", "final"}}}`` read from the checkpoints."""
    out: dict = {}
    for b in backends:
        out[b] = {}
        for t in targets:
            rows = []
            for p in sorted(glob.glob(os.path.join(out_dir, t, f"seed*_{b}.pt"))):
                seed = int(re.search(r"seed(\d+)_", os.path.basename(p)).group(1))
                ck = torch.load(p, map_location="cpu", weights_only=False)
                best, final = tv_at_best_val(ck["history"])
                rows.append((seed, best, final))
                del ck
            out[b][t] = {"seeds": [r[0] for r in rows], "best": [r[1] for r in rows],
                         "final": [r[2] for r in rows]}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", required=True, help="the driver's --out_dir")
    p.add_argument("--backends", default="direct_exp",
                   help="comma-separated backends, as passed to the driver")
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--json_out", default="", help="also write the per-seed values here")
    a = p.parse_args()
    backends = [b for b in a.backends.split(",") if b]
    targets = [t for t in a.targets.split(",") if t]
    res = collect(a.out_dir, backends, targets)
    print(f"{'backend':16s} {'target':8s} {'seeds':>5s} {'TV best val':>12s} {'TV final':>9s}")
    for b in backends:
        for t in targets:
            r = res[b][t]
            if not r["seeds"]:
                print(f"{b:16s} {t:8s} {0:5d} {'--':>12s} {'--':>9s}")
                continue
            print(f"{b:16s} {t:8s} {len(r['seeds']):5d} {_median(r['best']):12.3f} "
                  f"{_median(r['final']):9.3f}")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
