#!/usr/bin/env python3
"""Read the CIFAR-10 cells back: the learning-rate selection, the table, the curves.

Collect and render only; nothing here trains. The cells are written by
``scripts/experiments_sivaprasad_cifar.py``, one JSON per cell under
``data/checkpoints/<experiment_name>_<fields>/cells/``, one directory per
learning rate. This script reads every directory under the prefix, drops
smoke directories (``max_steps_per_epoch > 0``) and skipped cells, and writes:

* the selected learning rate per construction: the rate with the best median
  validation accuracy over the selection seeds (ties to the smaller rate; the
  test set is never read), to ``data/analysis/sivaprasad_cifar_selection.json``;
* the table: per construction at its selected rate and per seed, test
  accuracy at the epoch of best validation accuracy, final test accuracy,
  best validation accuracy, the selected and stopping epochs, with medians;
  the paired lift-minus-X differences with sign counts, and no p-value below
  five paired seeds; to ``data/analysis/sivaprasad_cifar.json``, which
  ``scripts/render_sivaprasad_cifar.py`` reads;
* training-curve and dwell summaries, and NaN-padded per-epoch series (early
  stopping makes them unequal in length);
* the curves and the occupancy trajectories as PNG.

Losses are compared as levels per epoch with epoch 0 shown, never as a drop
from the step-zero cross entropy. Dwell and occupancy are read at
``tau = 1e-4``, ``tau = 0`` and the relative threshold (a tenth of the cell's
own initial median), because at fan-in 800 under ``hk`` nearly the whole
initial draw lies below the shoulder threshold.

    python scripts/collect_sivaprasad_cifar.py                 # all of it
    python scripts/collect_sivaprasad_cifar.py --select_only   # the selection
    python scripts/collect_sivaprasad_cifar.py --no_figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from projorg import gitdir

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO = gitdir()
ORDER = ("lift", "exppost", "exppost_matched", "direct", "pgd", "nonconvex")
MIN_SEEDS_FOR_P = 5
TAU_KEYS = ("0.0001", "0")
LEVEL_EPOCHS = (0, 1, 2, 5, 10, 20, 40)


def _records(checkpoints: str, prefix: str, dataset: str,
             include_smoke: bool = False) -> tuple[dict, dict]:
    """``{(construction, lr, seed): record}`` and ``{lr: config}``."""
    cells, configs = {}, {}
    for d in sorted(glob.glob(os.path.join(checkpoints, prefix + "_*"))):
        manifest = os.path.join(d, "results.json")
        if not os.path.exists(manifest):
            continue
        with open(manifest) as f:
            cfg = json.load(f).get("config", {})
        if (int(cfg.get("max_steps_per_epoch", 0) or 0) > 0
                and not include_smoke):
            continue                                 # a smoke directory
        lr = float(cfg["lr"])
        configs[lr] = cfg
        for path in glob.glob(os.path.join(d, "cells", "*.json")):
            with open(path) as f:
                r = json.load(f)
            if r.get("dataset") != dataset or "skipped_reason" in r:
                continue
            if "test_acc_at_best_val_acc" not in r:
                print(f"  NOTE: {os.path.basename(path)} at lr {lr:g} has no "
                      "test record; dropped")
                continue
            r["_lr"] = lr
            cells[(r["construction"], lr, int(r["seed"]))] = r
    return cells, configs


def _median(v) -> float:
    a = np.asarray([x for x in v if x is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def select_lr(cells: dict, select_seeds: list[int]) -> dict:
    """Best median validation accuracy over ``select_seeds``; ties to the
    smaller rate. The test columns are never read here."""
    by = defaultdict(lambda: defaultdict(list))
    for (con, lr, seed), r in cells.items():
        if seed in select_seeds:
            by[con][lr].append(float(r["best_val_acc"]))
    out = {}
    for con, lrs in by.items():
        scored = sorted(
            ((_median(v), -lr, lr, len(v)) for lr, v in lrs.items()),
            reverse=True,
        )
        med, _, lr, n = scored[0]
        out[con] = {
            "lr": lr, "median_best_val_acc": med, "n_seeds": n,
            "per_lr": {f"{k:g}": {"median_best_val_acc": _median(v),
                                  "n_seeds": len(v)} for k, v in lrs.items()},
        }
    return out


def _paired(a: dict, b: dict, field: str) -> dict:
    shared = sorted(set(a) & set(b))
    d = np.asarray([a[s][field] - b[s][field] for s in shared], dtype=float)
    keep = np.isfinite(d)
    d = d[keep]
    out = {
        "n_paired_seeds": int(d.size),
        "seeds": [int(s) for s, k in zip(shared, keep) if k],
        "per_seed_diff": [float(x) for x in d],
        "median_diff": float(np.median(d)) if d.size else float("nan"),
        "n_positive": int(np.sum(d > 0)), "n_negative": int(np.sum(d < 0)),
        "n_zero": int(np.sum(d == 0)),
        "wilcoxon_two_sided_p": None,
    }
    if d.size >= MIN_SEEDS_FOR_P and np.any(d != 0):
        from scipy import stats
        out["wilcoxon_two_sided_p"] = float(
            stats.wilcoxon(d, alternative="two-sided").pvalue)
    return out


def _series(r: dict, field: str) -> list[float]:
    return [float(e[field]) for e in r["val_curve"]
            if e.get("full_split") and field in e]


def _pad(rows: list[list[float]]) -> list[list]:
    n = max((len(x) for x in rows), default=0)
    return [[float(v) for v in x] + [None] * (n - len(x)) for x in rows]


def _dwell_at(r: dict, tau_key: str) -> dict:
    """The tracker's numbers at one threshold (tracked subset) and, where the
    exact profiles carry that threshold, the exact occupancies."""
    hz = r.get("shoulder", {}).get("hazard_curve", {}).get(tau_key, {})
    out = {k: hz.get(k) for k in (
        "tau", "exit_hazard", "entry_hazard", "hazard_dwell_steps",
        "mean_observed_episode_steps", "censored_frac",
        "frac_coords_ever_inside", "occupancy_init", "occupancy_final",
        "occupancy_mean", "monotone_nondecreasing")}
    out["occ_init_exact"] = r["step0_profile"]["occupancy"].get(tau_key)
    out["occ_final_exact"] = r["stepT_profile"]["occupancy"].get(tau_key)
    late = r.get("shoulder_late", {}).get("hazard_curve", {}).get(tau_key, {})
    out["exit_hazard_late"] = late.get("exit_hazard")
    return out


def _curve_summary(r: dict) -> dict:
    tl = [float(v) for v in r["epoch_train_loss"]]
    tl0 = [float(r["train_loss_epoch0"])] + tl
    vl = _series(r, "val_loss")
    va = _series(r, "val_acc")
    ta = _series(r, "test_acc")
    rel_tau = r.get("shoulder_rel_tau")
    rel_key = f"{float(rel_tau):.6g}" if rel_tau else None
    dwell = {f"tau_{k}": _dwell_at(r, k) for k in TAU_KEYS}
    dwell["tau_rel"] = _dwell_at(r, rel_key) if rel_key else None
    dwell["rel_tau"] = rel_tau
    dwell["downward_at_primary"] = r.get("shoulder_downward")
    dwell["late_window_started"] = r.get("shoulder_late_started")
    return {
        "train_loss_epoch0": r.get("train_loss_epoch0"),
        "train_loss_epochs": tl,
        "train_loss_levels": tl0,
        "val_loss_levels": vl,
        "val_acc_epochs": va, "test_acc_epochs": ta,
        "train_loss_level_at": {str(k): (tl0[k] if len(tl0) > k else None)
                                for k in LEVEL_EPOCHS},
        "val_loss_level_at": {str(k): (vl[k] if len(vl) > k else None)
                              for k in LEVEL_EPOCHS},
        "best_val_acc": r["best_val_acc"],
        "best_val_acc_epoch": r["best_val_acc_epoch"],
        "stopped_epoch": r["stopped_epoch"],
        "early_stopped": r["early_stopped"],
        "test_acc_at_best_val_acc": r["test_acc_at_best_val_acc"],
        "final_test_acc": r["final_test_acc"],
        "test_acc_cond_spread_at_best_val_acc": r.get(
            "test_acc_cond_spread_at_best_val_acc"),
        "ce_step0": r.get("ce_step0"),
        "ce_step0_direct_ref": r.get("ce_step0_direct_ref"),
        "ce_step0_rel_gap": r.get("ce_step0_rel_gap"),
        "first_layer_pre_var_ratio": r.get("first_layer_pre_var_ratio"),
        "draw_match": r.get("draw_match"),
        "exp_eps": r.get("exp_eps"),
        "dwell": dwell,
        "n_trainable": r["n_trainable"], "wall_s": r["wall_s"],
        "peak_gpu_mib": r.get("peak_gpu_mib"),
        "peak_gpu_reserved_mib": r.get("peak_gpu_reserved_mib"),
        "device_name": r.get("device_name"),
        "resumed_from_epoch": r.get("resumed_from_epoch"),
    }


def _figures(table: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _experiments_sivaprasad_cifar_lib import CONSTRUCTION_COLOR

    os.makedirs(out_dir, exist_ok=True)
    panels = (("train_loss_levels", "train loss (epoch 0 shown)", True),
              ("val_loss_levels", "validation loss (epoch 0 shown)", True),
              ("val_acc_epochs", "validation accuracy", False),
              ("test_acc_epochs", "test accuracy", False))
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.0))
    for ax, (field, label, log) in zip(axes, panels):
        for con in ORDER:
            if con not in table:
                continue
            rows = [c[field] for c in table[con]["per_seed"].values()
                    if c[field]]
            if not rows:
                continue
            col = CONSTRUCTION_COLOR[con]
            n = min(len(x) for x in rows)
            med = np.median(np.stack([np.asarray(x[:n]) for x in rows]), 0)
            ax.plot(np.arange(n), med, color=col, lw=1.5,
                    label=f"{con} (lr {table[con]['lr']:g})")
            for x in rows:
                ax.plot(np.arange(len(x)), x, color=col, lw=0.6,
                        ls="--", alpha=0.45)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    path = os.path.join(out_dir, "sivaprasad_cifar_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {path}")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    for ax, (key, label) in zip(axes, (
            ("occupancy_1e-4_series", "occupancy at tau = 1e-4"),
            ("occupancy_0_series", "occupancy at tau = 0 (exact zeros)"),
            ("occupancy_rel_series", "occupancy at the relative threshold"))):
        for con in ORDER:
            if con not in table:
                continue
            col = CONSTRUCTION_COLOR[con]
            for c in table[con]["per_seed"].values():
                s = c.get(key)
                if not s:
                    continue
                t = np.arange(len(s)) * c["occupancy_stride"] / c["steps_per_epoch"]
                ax.plot(t, s, color=col, lw=0.8, alpha=0.7, label=con)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    axes[0].legend(seen.values(), seen.keys(), frameon=False, fontsize=7)
    path = os.path.join(out_dir, "sivaprasad_cifar_occupancy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints",
                    default=os.path.join(_REPO, "data", "checkpoints"))
    ap.add_argument("--prefix", default="experiments_sivaprasad_cifar")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--select_seeds", default="0",
                    help="seeds the learning-rate selection is made on "
                         "(comma list)")
    ap.add_argument("--selection",
                    default=os.path.join(_REPO, "data", "analysis",
                                         "sivaprasad_cifar_selection.json"))
    ap.add_argument("--out", default=os.path.join(_REPO, "data", "analysis",
                                                  "sivaprasad_cifar.json"))
    ap.add_argument("--plots", default=os.path.join(_REPO, "plots",
                                                    "sivaprasad_cifar"))
    ap.add_argument("--select_only", action="store_true",
                    help="write the selection json and stop")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--include_smoke", action="store_true",
                    help="also read directories with max_steps_per_epoch > 0")
    a = ap.parse_args(argv)
    select_seeds = [int(s) for s in a.select_seeds.split(",") if s.strip()]

    cells, configs = _records(a.checkpoints, a.prefix, a.dataset,
                              include_smoke=a.include_smoke)
    if not cells:
        print(f"no cells under {a.checkpoints}/{a.prefix}_*; nothing to do")
        return 1
    print(f"{len(cells)} cells at learning rates "
          f"{sorted({lr for _, lr, _ in cells})}")

    selection = select_lr(cells, select_seeds)
    os.makedirs(os.path.dirname(a.selection), exist_ok=True)
    with open(a.selection, "w") as f:
        json.dump({"select_seeds": select_seeds, "rule":
                   "best median validation accuracy over select_seeds; "
                   "ties to the smaller rate; test set never read",
                   "selected": selection}, f, indent=2)
    print(f"Saved to {a.selection}")
    for con in ORDER:
        if con in selection:
            s = selection[con]
            print(f"  {con:10s} lr {s['lr']:g}  median best val acc "
                  f"{100 * s['median_best_val_acc']:.2f}%  over "
                  f"{s['n_seeds']} seed(s)  "
                  + "  ".join(f"[{k}: {100 * v['median_best_val_acc']:.2f}%]"
                              for k, v in sorted(s["per_lr"].items(),
                                                 key=lambda kv: float(kv[0]))))
    if a.select_only:
        return 0

    table = {}
    for con in ORDER:
        if con not in selection:
            continue
        lr = selection[con]["lr"]
        per_seed = {seed: _curve_summary(r)
                    for (c, l, seed), r in sorted(cells.items())
                    if c == con and l == lr}
        for (c, l, seed), r in cells.items():
            if c == con and l == lr:
                hc = r.get("shoulder", {}).get("hazard_curve", {})
                rel = r.get("shoulder_rel_tau")
                rel_key = f"{float(rel):.6g}" if rel else ""
                per_seed[seed]["occupancy_1e-4_series"] = hc.get(
                    "0.0001", {}).get("series")
                per_seed[seed]["occupancy_0_series"] = hc.get(
                    "0", {}).get("series")
                per_seed[seed]["occupancy_rel_series"] = hc.get(
                    rel_key, {}).get("series")
                per_seed[seed]["occupancy_stride"] = r.get(
                    "occupancy_series_stride", 1)
                per_seed[seed]["steps_per_epoch"] = r["steps_per_epoch"]
        table[con] = {
            "lr": lr, "n_seeds": len(per_seed),
            "seeds": sorted(per_seed),
            "median_test_acc_at_best_val_acc": _median(
                [c["test_acc_at_best_val_acc"] for c in per_seed.values()]),
            "median_final_test_acc": _median(
                [c["final_test_acc"] for c in per_seed.values()]),
            "median_best_val_acc": _median(
                [c["best_val_acc"] for c in per_seed.values()]),
            "median_best_val_acc_epoch": _median(
                [c["best_val_acc_epoch"] for c in per_seed.values()]),
            "median_stopped_epoch": _median(
                [c["stopped_epoch"] for c in per_seed.values()]),
            "n_early_stopped": int(sum(c["early_stopped"]
                                       for c in per_seed.values())),
            "median_exit_hazard_1e-4": _median(
                [c["dwell"]["tau_0.0001"]["exit_hazard"]
                 for c in per_seed.values()]),
            "median_hazard_dwell_steps_1e-4": _median(
                [c["dwell"]["tau_0.0001"]["hazard_dwell_steps"]
                 for c in per_seed.values()]),
            "median_exit_hazard_0": _median(
                [c["dwell"]["tau_0"]["exit_hazard"]
                 for c in per_seed.values()]),
            "median_occ_final_1e-4_exact": _median(
                [c["dwell"]["tau_0.0001"]["occ_final_exact"]
                 for c in per_seed.values()]),
            "median_occ_final_0_exact": _median(
                [c["dwell"]["tau_0"]["occ_final_exact"]
                 for c in per_seed.values()]),
            "median_train_loss_level_at": {
                str(k): _median([c["train_loss_level_at"][str(k)]
                                 for c in per_seed.values()])
                for k in LEVEL_EPOCHS},
            "median_val_loss_level_at": {
                str(k): _median([c["val_loss_level_at"][str(k)]
                                 for c in per_seed.values()])
                for k in LEVEL_EPOCHS},
            "padded": {
                k: _pad([c[k] for c in per_seed.values()])
                for k in ("train_loss_levels", "val_loss_levels",
                          "val_acc_epochs", "test_acc_epochs")
            },
            "per_seed": {str(s): c for s, c in per_seed.items()},
        }

    paired = {}
    if "lift" in table:
        lift = {int(s): c for s, c in table["lift"]["per_seed"].items()}
        for con in ("exppost", "exppost_matched", "direct", "pgd", "nonconvex"):
            if con not in table:
                continue
            other = {int(s): c for s, c in table[con]["per_seed"].items()}
            paired[f"lift_minus_{con}"] = {
                "test_acc_at_best_val_acc": _paired(
                    lift, other, "test_acc_at_best_val_acc"),
                "final_test_acc": _paired(lift, other, "final_test_acc"),
                "best_val_acc": _paired(lift, other, "best_val_acc"),
            }

    out = {
        "dataset": a.dataset, "prefix": a.prefix,
        "configs_by_lr": {f"{lr:g}": {k: cfg[k] for k in (
            "epochs", "patience", "batch_size", "loss_reduction",
            "hidden_dim", "train_split", "hyper_hidden_sizes",
            "lift_init_match", "ce0_match_rtol") if k in cfg}
            for lr, cfg in configs.items()},
        "selection": selection, "table": table, "paired": paired,
        "min_seeds_for_p": MIN_SEEDS_FOR_P,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved to {a.out}")

    print(f"\n{'construction':16s} {'lr':>7s} {'n':>2s} "
          f"{'test@bestval':>12s} {'final test':>10s} {'best val':>9s} "
          f"{'sel ep':>6s} {'stop ep':>7s} {'exit hz 1e-4':>12s} "
          f"{'val loss ep0/1/5':>18s}")
    for con in ORDER:
        if con not in table:
            continue
        t = table[con]
        lv = t["median_val_loss_level_at"]
        print(f"{con:16s} {t['lr']:7g} {t['n_seeds']:2d} "
              f"{100 * t['median_test_acc_at_best_val_acc']:11.2f}% "
              f"{100 * t['median_final_test_acc']:9.2f}% "
              f"{100 * t['median_best_val_acc']:8.2f}% "
              f"{t['median_best_val_acc_epoch']:6.0f} "
              f"{t['median_stopped_epoch']:7.0f} "
              f"{t['median_exit_hazard_1e-4']:12.3g} "
              f"{lv['0']:6.3f}/{lv['1']:5.3f}/{lv['5']:5.3f}")
    for name, p in paired.items():
        q = p["test_acc_at_best_val_acc"]
        print(f"  {name:22s} test@bestval median diff "
              f"{100 * q['median_diff']:+.2f} pts  "
              f"+{q['n_positive']} / -{q['n_negative']} / ={q['n_zero']} "
              f"of {q['n_paired_seeds']} seeds"
              + (f"  p={q['wilcoxon_two_sided_p']:.3g}"
                 if q["wilcoxon_two_sided_p"] is not None
                 else "  (no p below five seeds)"))
    if not a.no_figures:
        _figures(table, a.plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
