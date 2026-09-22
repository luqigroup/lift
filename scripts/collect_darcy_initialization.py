"""Summary table for the Darcy initialization probe.

Reads the ``data/checkpoints/darcy_init_*/results.json`` archives -- one per
initialization setting, since ``init_mode`` is outside the run identity and the
settings are told apart by their ``--experiment_name`` prefix -- and prints, per
setting, construction and seed: the held-out conditional NLL at the final
iterate (Darcy has a single held-out split and no checkpoint selection),
shoulder occupancy at initialization and at the end, the deepest latent weight
at the end, the trainable parameter count, the anchor spread and wall time. Then
the paired comparisons and the ordering of the three constructions.

For softplus the occupancy column is the fraction of constrained latent weights
below ``logit(0.05) = -2.9444``; for PGD it is the projection's active set,
which is a different statistic and is labeled as one.

Usage::

    python scripts/collect_darcy_initialization.py
    python scripts/collect_darcy_initialization.py --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
from projorg import gitdir

_REPO = gitdir()
_CKPT = os.path.join(_REPO, "data", "checkpoints")
_LOGS = os.path.join(_REPO, "logs")

THRESHOLD = float(np.log(0.05 / 0.95))
ORDER = ("pgd", "direct", "hypernet")
LABEL = {"pgd": "PGD", "direct": "direct softplus", "hypernet": "lift"}
SETTINGS = ("published", "pytorch", "harmonized")


def _load_args(archive_dir: str) -> dict:
    p = os.path.join(
        _LOGS, os.path.basename(os.path.normpath(archive_dir)),
        "experiment_info.json",
    )
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)["arguments"]


def _rows(setting: str) -> List[dict]:
    pattern = os.path.join(_CKPT, f"darcy_init_{setting}_*", "results.json")
    out: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        args = _load_args(os.path.dirname(path))
        with open(path) as f:
            recs = json.load(f)
        for rec in recs:
            tail = rec["history"].get("theta_tail") or [[float("nan")] * 5]
            spread = rec.get("anchor_spread") or {}
            out.append({
                "setting": rec.get("init_mode", args.get("init_mode", setting)),
                "archive": os.path.basename(os.path.dirname(path)),
                "backend": rec["backend"],
                "seed": int(rec["seed"]),
                "nll": float(rec["final_eval_nll"]),
                "occ_init": float(rec["shoulder_occupancy_init"]),
                "occ_end": float(rec["shoulder_occupancy"]),
                "theta_min_end": float(tail[-1][0]),
                "theta_p50_end": float(tail[-1][3]),
                "n_params": int(rec["n_params_trainable"]),
                "anchor_sd": spread.get("resampled_std"),
                "wall": float(rec.get("wall_seconds_train", float("nan"))),
                "n_train": rec.get("n_train"),
            })
    return out


def _paired(rows: List[dict], a: str, b: str) -> dict:
    """``a - b`` per seed, over the seeds both methods ran."""
    va = {r["seed"]: r["nll"] for r in rows if r["backend"] == a}
    vb = {r["seed"]: r["nll"] for r in rows if r["backend"] == b}
    seeds = sorted(set(va) & set(vb))
    d = np.array([va[s] - vb[s] for s in seeds])
    if not len(d):
        return {}
    return {
        "n": len(d),
        "wins": int((d < 0).sum()),
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "max_abs": float(np.abs(d).max()),
        "per_seed": {int(s): float(x) for s, x in zip(seeds, d)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    payload: Dict[str, dict] = {"threshold": THRESHOLD, "settings": {}}
    print(f"shoulder threshold logit(0.05) = {THRESHOLD:.4f}\n")
    for setting in SETTINGS:
        rows = _rows(setting)
        if not rows:
            print(f"### {setting}: no archive yet\n")
            continue
        print(f"### {setting}")
        header = (
            f"{'arm':16s} {'seed':>4s} {'NLL':>9s} {'occ init':>9s} "
            f"{'occ end':>8s} {'min latent':>11s} {'p50 latent':>11s} "
            f"{'params':>10s} {'anchor sd':>10s} {'wall s':>7s}"
        )
        print(header)
        for backend in ORDER:
            for r in sorted(
                [x for x in rows if x["backend"] == backend],
                key=lambda x: x["seed"],
            ):
                sd = "--" if r["anchor_sd"] is None else f"{r['anchor_sd']:.4f}"
                print(
                    f"{LABEL[backend]:16s} {r['seed']:4d} {r['nll']:9.3f} "
                    f"{r['occ_init']:9.4f} {r['occ_end']:8.4f} "
                    f"{r['theta_min_end']:11.3f} {r['theta_p50_end']:11.3f} "
                    f"{r['n_params']:10d} {sd:>10s} {r['wall']:7.0f}"
                )
        stats = {}
        for backend in ORDER:
            v = np.array([r["nll"] for r in rows if r["backend"] == backend])
            if not len(v):
                continue
            stats[backend] = {
                "n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=0)),
                "median": float(np.median(v)),
                "occ_init": float(np.mean(
                    [r["occ_init"] for r in rows if r["backend"] == backend])),
                "occ_end": float(np.mean(
                    [r["occ_end"] for r in rows if r["backend"] == backend])),
                "min_latent_end": float(np.min(
                    [r["theta_min_end"] for r in rows if r["backend"] == backend])),
            }
            print(
                f"  {LABEL[backend]:16s} mean {v.mean():.3f} +- {v.std(ddof=0):.3f} "
                f"(median {np.median(v):.3f}, n={len(v)})"
            )
        pairs = {
            "lift-direct": _paired(rows, "hypernet", "direct"),
            "lift-pgd": _paired(rows, "hypernet", "pgd"),
            "direct-pgd": _paired(rows, "direct", "pgd"),
        }
        for name, p in pairs.items():
            if p:
                print(
                    f"  {name:12s} below on {p['wins']}/{p['n']}, median "
                    f"{p['median']:+.3f}, mean {p['mean']:+.3f}, "
                    f"largest |diff| {p['max_abs']:.3f}"
                )
        ranked = sorted(stats, key=lambda b: stats[b]["mean"])
        print("  ordering by mean NLL (best first): "
              + " < ".join(LABEL[b] for b in ranked))
        print()
        payload["settings"][setting] = {
            "rows": rows, "stats": stats, "paired": pairs,
            "ordering": [LABEL[b] for b in ranked],
        }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved to {args.json}")


if __name__ == "__main__":
    main()
