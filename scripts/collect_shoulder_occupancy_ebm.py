r"""Shoulder occupancy of the ICNN-EBM lines, recomputed from stored weights.

The statistic is the fraction of constrained latent weights whose
positivity-map derivative has collapsed, ``psi'(w) = sigmoid(w) <= 0.05``,
that is ``w <= logit(0.05) = -2.9444``.

The EBM drivers store ``theta_snaps`` as the emitted, post-positivity
weights rather than the latent ones. Under a softplus readout
``theta = softplus(w)``, so the shoulder condition is applied here as
``theta <= softplus(-2.9444) = 0.0512933``.

The constrained set is ``z_layers[i].weight`` for every hidden layer plus
``output_layer.weight`` (see ``lift/models/icnn.py``); the input skip
connections and all biases are unconstrained and are excluded.

PGD is a different statistic and is labeled as such. In ``clamp`` mode there
is no softplus, so ``theta`` is the clamped weight itself and ``sigmoid`` is
not the local gain. The column reported for PGD is the fraction of
constrained weights at exactly zero, and it is never merged into the
softplus occupancy column.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from projorg import gitdir

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = gitdir()
_CK = os.path.join(_REPO, "data", "checkpoints")

THR_LATENT = -2.9444389791664403          # logit(0.05)
THR_EMITTED = float(F.softplus(torch.tensor(THR_LATENT)))   # 0.0512933


def _constrained(theta: dict) -> torch.Tensor:
    """Flatten the constrained (non-negative) weights of one theta snapshot."""
    keys = [
        k for k in theta
        if k.endswith(".weight")
        and (".z_layers." in k or k.endswith("output_layer.weight"))
    ]
    if not keys:
        raise KeyError(f"no constrained weights among {sorted(theta)[:6]}…")
    return torch.cat([theta[k].reshape(-1) for k in sorted(keys)]).double()


def occupancy(theta: dict, mode: str) -> Dict[str, float]:
    w = _constrained(theta)
    out = {"n_constrained": int(w.numel()), "median": float(w.median())}
    if mode == "softplus":
        out["occupancy"] = float((w <= THR_EMITTED).double().mean())
    else:  # pgd / clamp
        out["frac_exact_zero"] = float((w <= 0).double().mean())
        out["frac_below_shoulder_magnitude"] = float(
            (w <= THR_EMITTED).double().mean(),
        )
    return out


def _load_snaps(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return ck.get("theta_snaps"), ck.get("snap_iters")


def reversibility(snaps: List[dict]) -> Dict[str, float]:
    """Whether the shoulder is absorbing for this run.

    ``monotone_nondecreasing`` says the occupancy series never decreases
    between snapshots. ``leave_rate`` is, over all (coordinate, consecutive
    snapshot) pairs whose coordinate was on the shoulder at the earlier
    snapshot, the fraction off it at the later one; ``dwell`` is its
    complement. An absorbing shoulder has leave rate 0 and dwell 1. This,
    not the occupancy level, is the mean-exit-time quantity.
    """
    series, prev_in, n_in, n_left = [], None, 0, 0
    for th in snaps:
        w = _constrained(th)
        inside = w <= THR_EMITTED
        series.append(float(inside.double().mean()))
        if prev_in is not None:
            was = prev_in
            n_in += int(was.sum())
            n_left += int((was & ~inside).sum())
        prev_in = inside
    s = np.asarray(series)
    return {
        "monotone_nondecreasing": bool(np.all(np.diff(s) >= -1e-12)),
        "leave_rate": (n_left / n_in) if n_in else float("nan"),
        "dwell": (1.0 - n_left / n_in) if n_in else float("nan"),
        "series": [float(x) for x in s],
    }


def scan(path: str, mode: str) -> Optional[dict]:
    snaps, iters = _load_snaps(path)
    if not snaps:
        return None
    first, last = occupancy(snaps[0], mode), occupancy(snaps[-1], mode)
    rec = {
        "source": os.path.relpath(path, _REPO),
        "iter_init": int(iters[0]) if iters else 0,
        "iter_final": int(iters[-1]) if iters else -1,
        "init": first,
        "final": last,
    }
    if mode == "softplus":
        rec["reversibility"] = reversibility(snaps)
    return rec


# --------------------------------------------------------------------------
GALLERY = [
    ("direct", "lcmm_1d_logconcave_30seed", "direct", "softplus"),
    ("lift", "lcmm_1d_logconcave_30seed_poolnorm", "hypernet", "softplus"),
    ("pgd", "lcmm_1d_logconcave_30seed_pgd", "pgd", "pgd"),
]
TARGETS = ["gumbel", "laplace", "gamma", "beta"]

TABULAR = [
    ("HEPMASS", 21,
     "experiments_pgd_uci_hepmass_target_kind-uci_hepmass_K-1_objective-fkl-"
     "direct_data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_sampler-"
     "0.005_hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_sampler_"
     "hyper.b0929293841311db47eff4ffed48c9fb67bf0bca"),
    ("MiniBooNE", 43,
     "pgd_uci_miniboone_target_kind-uci_miniboone_K-1_objective-fkl-direct_"
     "data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_sampler-0.005_"
     "hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_sampler_hyper_"
     "hidden_.0abf822e2e99cce7f95d618abe3b7e09e642a5e4"),
]
MODE = {"direct": "softplus", "hypernet": "softplus", "pgd": "pgd"}

# The POWER runs live outside this checkout. Its PGD run is at a different
# width, 512 x 5 against 256 x 3, so it is NOT architecture-matched to the
# other two and its row must be read with that in mind.
_LCMM = ""   # a second checkpoint root, for archives outside the project
EXTERNAL = [
    ("POWER", 6, os.path.join(
        _LCMM,
        "experiments_hypernet_vs_direct_uci_power_k1_target_kind-uci_power_"
        "K-1_objective-fkl-direct_data_path-_max_train_rows-0_n_iters-4000_"
        "lr_ebm-0.001_lr_sampler-0.005_hidden_dim-256_nlayers-3_hyper_hidden_"
        "sizes-96-96-12.3e445ef759a707a5c7a107b16bb1bcd252cdab73")),
    ("POWER (PGD, 512x5 — NOT width-matched)", 6, os.path.join(
        _LCMM,
        "experiments_pgd_uci_power_target_kind-uci_power_K-1_objective-fkl-"
        "direct_data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_"
        "sampler-0.005_hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_"
        "sampler_hyper_hid.bf7e6401ba075aacd7768a299fbd3cc7df32cf83")),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()
    payload: Dict[str, list] = {"gallery": [], "tabular": []}

    print("## 1-D log-concave gallery (D = 1), 30 seeds per cell")
    print("\n| target | construction | occupancy @ init | occupancy @ final "
          "| seeds | statistic |")
    print("|---|---|---|---|---|---|")
    for arm, root, tag, mode in GALLERY:
        for target in TARGETS:
            recs = []
            for path in sorted(glob.glob(
                    os.path.join(_CK, root, target, f"seed*_{tag}.pt"))):
                r = scan(path, mode)
                if r is None:
                    continue
                r.update(arm=arm, target=target,
                         seed=int(os.path.basename(path).split("_")[0][4:]))
                recs.append(r)
            if not recs:
                continue
            payload["gallery"].extend(recs)
            key = "occupancy" if mode == "softplus" else "frac_exact_zero"
            lab = ("softplus occupancy" if mode == "softplus"
                   else "fraction at exactly zero (NOT the shoulder statistic)")
            a = np.array([r["init"][key] for r in recs])
            b = np.array([r["final"][key] for r in recs])
            extra = ""
            if mode == "softplus":
                mono = sum(r["reversibility"]["monotone_nondecreasing"]
                           for r in recs)
                lr_ = np.array([r["reversibility"]["leave_rate"]
                                for r in recs])
                extra = (f" monotone {mono}/{len(recs)}, "
                         f"median leave rate {np.median(lr_):.4f}")
            print(f"| {target} | {arm} | {a.mean():.4f} "
                  f"({a.min():.4f}–{a.max():.4f}) | {b.mean():.4f} "
                  f"({b.min():.4f}–{b.max():.4f}) | {len(recs)} | {lab}"
                  f"{extra} |")

    print("\n## Tabular ICNN-EBMs")
    print("\n| dataset | D | construction | seed | occupancy @ init "
          "| occupancy @ final | constrained coords | statistic |")
    print("|---|---|---|---|---|---|---|---|")
    runs = [(n, D, os.path.join(_CK, d)) for n, D, d in TABULAR] + EXTERNAL
    for name, D, run_dir in runs:
        for path in sorted(glob.glob(os.path.join(run_dir, "seed*_*.pt"))):
            base = os.path.basename(path)[:-3]
            if "_" not in base:
                continue
            seed_s, tag = base.split("_", 1)
            if tag not in MODE:
                continue
            r = scan(path, MODE[tag])
            if r is None:
                continue
            r.update(dataset=name, D=D, arm=tag, seed=int(seed_s[4:]))
            payload["tabular"].append(r)
            mode = MODE[tag]
            key = "occupancy" if mode == "softplus" else "frac_exact_zero"
            lab = ("softplus occupancy" if mode == "softplus"
                   else "fraction at exactly zero (NOT the shoulder statistic)")
            arm = {"direct": "direct", "hypernet": "lift", "pgd": "pgd"}[tag]
            extra = ""
            if mode == "softplus":
                rv = r["reversibility"]
                extra = (f"; monotone={rv['monotone_nondecreasing']}, "
                         f"leave rate {rv['leave_rate']:.4f}")
            print(f"| {name} | {D} | {arm} | {r['seed']} "
                  f"| {r['init'][key]:.4f} | {r['final'][key]:.4f} "
                  f"| {r['final']['n_constrained']:,} | {lab}{extra} |")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
