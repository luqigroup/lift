r"""Dwell statistics of the stuck set, all three constructions, per seed.

Measures how long a constrained coordinate stays stuck, rather than how many
are stuck at any instant, wherever an archive stores per-coordinate weight
snapshots along training.

The stuck set
-------------
* softplus (direct and the lift) -- the shoulder: the emitted weight
  ``theta <= softplus(logit(0.05)) = 0.0512933``, i.e. the latent weight has
  ``sigmoid(w) <= 0.05`` and its update is attenuated by 20x or more.
* PGD -- the projection's fixed set: ``theta <= 0``, which for weights
  clamped after every step means exactly zero. PGD has no shoulder (no
  softplus, so ``sigmoid`` is not a local gain) and the softplus
  constructions have no exact zeros, so the two columns are never merged.

What is reported, per run
-------------------------
``occ_first`` / ``occ_last``  fraction of constrained coordinates in the
    stuck set at the first and last snapshot.
``exit_rate``   transitions out of the stuck set divided by the number of
    coordinate-visits in it: over consecutive snapshot pairs, the count of
    (coordinate, interval) pairs that were in at the earlier snapshot and out
    at the later one, divided by the count that were in at the earlier
    snapshot. This is the per-visit escape probability -- the mean-exit-time
    object, not a level.
``entry_rate``  the same for the complement: out-then-in over out-visits.
``median_dwell_frac``  median over episodes of (episode length in snapshots)
    / (number of snapshots). An episode is a maximal run of consecutive
    snapshots a coordinate spends in the set; episodes still running at the
    last snapshot are censored and counted at their observed length
    (``censored_frac`` says how many).
``monotone``    the occupancy series never decreases.
``ever_frac``   fraction of coordinates in the stuck set at some snapshot.
``most_frac``   fraction in it at more than half of the snapshots.
``always_2h``   fraction in it at every snapshot of the second half.

Usage::

    python scripts/measure_dwell_time.py --json_out docs/analysis/dwell_time.json
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
_LCMM = ""   # a second checkpoint root, if one is kept outside the project

THR_LATENT = -2.9444389791664403                  # logit(0.05)
THR_EMITTED = float(F.softplus(torch.tensor(THR_LATENT)))   # 0.0512933


def _constrained(theta: dict) -> torch.Tensor:
    keys = [
        k for k in theta
        if k.endswith(".weight")
        and (".z_layers." in k or k.endswith("output_layer.weight"))
    ]
    if not keys:
        raise KeyError(f"no constrained weights among {sorted(theta)[:6]}")
    return torch.cat([theta[k].reshape(-1) for k in sorted(keys)]).double()


def dwell(snaps: List[dict], mode: str) -> Dict[str, object]:
    """Dwell statistics of one run's stuck set, streamed over snapshots."""
    thr = 0.0 if mode == "pgd" else THR_EMITTED
    T = len(snaps)
    occ: List[float] = []
    prev: Optional[np.ndarray] = None
    n_in = n_exit = n_out = n_enter = 0
    run_len: Optional[np.ndarray] = None     # current episode length
    hist: Dict[int, int] = {}                # completed episode lengths
    count_in: Optional[np.ndarray] = None    # snapshots spent in the set
    always_2h: Optional[np.ndarray] = None
    half = T // 2
    for t, th in enumerate(snaps):
        w = _constrained(th).numpy()
        inside = w <= thr
        occ.append(float(inside.mean()))
        if count_in is None:
            count_in = np.zeros(inside.size, dtype=np.int32)
            run_len = np.zeros(inside.size, dtype=np.int32)
        count_in += inside
        if prev is not None:
            was, now = prev, inside
            n_in += int(was.sum())
            n_exit += int((was & ~now).sum())
            n_out += int((~was).sum())
            n_enter += int((~was & now).sum())
            ended = was & ~now
            if ended.any():
                for L, c in zip(*np.unique(run_len[ended],
                                           return_counts=True)):
                    hist[int(L)] = hist.get(int(L), 0) + int(c)
            run_len[ended] = 0
        run_len[inside] += 1
        run_len[~inside] = 0
        if t >= half:
            always_2h = inside if always_2h is None else (always_2h & inside)
        prev = inside
    # Episodes still running at the last snapshot are right-censored.
    cens: Dict[int, int] = {}
    live = run_len > 0
    if live.any():
        for L, c in zip(*np.unique(run_len[live], return_counts=True)):
            cens[int(L)] = cens.get(int(L), 0) + int(c)

    def _median(h: Dict[int, int]) -> float:
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
    s = np.asarray(occ)
    return {
        "n_snapshots": T,
        "n_constrained": int(prev.size),
        "occ_first": occ[0],
        "occ_last": occ[-1],
        # Time-average occupancy: the share of constrained coordinates
        # contributing nothing at a typical instant.
        "mean_occ": float(np.mean(occ)),
        "occ_series": [float(x) for x in occ],
        "monotone_nondecreasing": bool(np.all(np.diff(s) >= -1e-12)),
        "exit_rate": (n_exit / n_in) if n_in else float("nan"),
        "entry_rate": (n_enter / n_out) if n_out else float("nan"),
        "n_coord_visits_in": n_in,
        "n_exits": n_exit,
        "median_dwell_snaps": _median(allh),
        "median_dwell_frac": _median(allh) / T if n_ep else float("nan"),
        "median_dwell_frac_completed": (
            _median(hist) / T if sum(hist.values()) else float("nan")),
        "n_episodes": n_ep,
        "censored_frac": (sum(cens.values()) / n_ep) if n_ep else float("nan"),
        "ever_frac": float((count_in > 0).mean()),
        "most_frac": float((count_in > T / 2).mean()),
        "always_2h_frac": (float(always_2h.mean())
                           if always_2h is not None else float("nan")),
    }


def scan(path: str, mode: str) -> Optional[dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    snaps = ck.get("theta_snaps")
    if not snaps:
        return None
    iters = ck.get("snap_iters") or list(range(len(snaps)))
    rec = {
        "source": path,
        "iter_first": int(iters[0]),
        "iter_last": int(iters[-1]),
        "statistic": ("exactly zero" if mode == "pgd"
                      else f"theta <= {THR_EMITTED:.7f} (shoulder)"),
        **dwell(snaps, mode),
    }
    m = ck.get("metrics") or {}
    for k in ("test_nll", "tv", "tv_distance"):
        if k in m:
            rec[k] = m[k]
    del ck
    return rec


GALLERY = [
    ("direct", "lcmm_1d_logconcave_30seed", "direct", "softplus"),
    ("lift", "lcmm_1d_logconcave_30seed_poolnorm", "hypernet", "softplus"),
    ("pgd", "lcmm_1d_logconcave_30seed_pgd", "pgd", "pgd"),
]
TARGETS = ["gumbel", "laplace", "gamma", "beta"]
MODE = {"direct": "softplus", "hypernet": "softplus", "pgd": "pgd"}
ARMNAME = {"direct": "direct", "hypernet": "lift", "pgd": "pgd"}

TABULAR = [
    ("HEPMASS-EBM", 21, os.path.join(_CK,
     "experiments_pgd_uci_hepmass_target_kind-uci_hepmass_K-1_objective-fkl-"
     "direct_data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_sampler-"
     "0.005_hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_sampler_"
     "hyper.b0929293841311db47eff4ffed48c9fb67bf0bca")),
    ("MiniBooNE-EBM", 43, os.path.join(_CK,
     "pgd_uci_miniboone_target_kind-uci_miniboone_K-1_objective-fkl-direct_"
     "data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_sampler-0.005_"
     "hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_sampler_hyper_"
     "hidden_.0abf822e2e99cce7f95d618abe3b7e09e642a5e4")),
    ("POWER-EBM", 6, os.path.join(_LCMM,
     "experiments_hypernet_vs_direct_uci_power_k1_target_kind-uci_power_"
     "K-1_objective-fkl-direct_data_path-_max_train_rows-0_n_iters-4000_"
     "lr_ebm-0.001_lr_sampler-0.005_hidden_dim-256_nlayers-3_hyper_hidden_"
     "sizes-96-96-12.3e445ef759a707a5c7a107b16bb1bcd252cdab73")),
    ("POWER-EBM (PGD, 512x5, NOT width-matched)", 6, os.path.join(_LCMM,
     "experiments_pgd_uci_power_target_kind-uci_power_K-1_objective-fkl-"
     "direct_data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_lr_"
     "sampler-0.005_hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_"
     "sampler_hyper_hid.bf7e6401ba075aacd7768a299fbd3cc7df32cf83")),
]


# ---------------------------------------------------------------------------
# Families with no per-coordinate snapshots on disk. Only the endpoints (and,
# where a scalar occupancy trace was logged, monotonicity) are recoverable; an
# exit rate needs coordinate identity across snapshots and is reported as
# unavailable rather than guessed.
# ---------------------------------------------------------------------------


def endpoints_darcy() -> List[dict]:
    """Darcy PCP-Map: one final ``.pth`` per method and seed, no snapshots.

    The stored ``record`` carries each method's own occupancy at init and at
    the end. For PGD that is the projection's active set (``w <= 0`` and the
    gradient pointing out of the cone), a different statistic from the
    softplus shoulder, read on a fresh batch.
    """
    rows = []
    # The ten-seed Darcy archive: the only one carrying all three
    # constructions at ten seeds.
    for d in sorted(glob.glob(os.path.join(_CK, "*darcy*fa11ec69"))):
        for path in sorted(glob.glob(os.path.join(d, "*_seed*.pth"))):
            base = os.path.basename(path)
            arm = base.split("_seed")[0]
            ck = torch.load(path, map_location="cpu", weights_only=False)
            rec = ck.get("record", {}) or {}
            # PGD's own stuck set, recomputed: the projected weights at
            # exactly zero at the final iterate. The record's number is the
            # gradient-qualified active set on a fresh batch.
            frac_zero = None
            if arm == "pgd":
                model = ck.get("model", {})
                names = sorted(k for k in model if k.startswith("picnn.Lw.")
                               and k.endswith(".weight"))
                if names:
                    w = torch.cat([model[n].reshape(-1) for n in names])
                    frac_zero = float((w <= 0).double().mean())
            del ck
            rows.append({
                "frac_exact_zero_final": frac_zero,
                "dataset": "Darcy-PCPMap",
                "arm": {"hypernet": "lift"}.get(arm, arm),
                "seed": int(base.split("_seed")[1].split(".")[0]),
                "occ_first": rec.get("shoulder_occupancy_init"),
                "occ_last": rec.get("shoulder_occupancy"),
                "statistic": ("PGD active set (w<=0 and grad out of cone)"
                              if arm == "pgd" else "shoulder"),
                "final_eval_nll": rec.get("final_eval_nll"),
                "dwell": "not recoverable: no weight snapshots on disk",
                "source": path,
            })
    return rows


def endpoints_ffhq(cells: Dict[str, List[str]]) -> List[dict]:
    """FFHQ ICNN regularizer: per-eval scalar traces, no weight snapshots.

    ``w_frac_zero`` (exactly zero) and ``w_frac_shoulder`` (below the
    shoulder edge in magnitude) are logged at each evaluation, so the level
    and its monotonicity are recoverable and the dwell is not.
    """
    rows = []
    for arm, dirs in cells.items():
        for arm_dir in dirs:
            mp = os.path.join(arm_dir, "metrics.json")
            if not os.path.exists(mp):
                continue
            with open(mp) as fh:
                m = json.load(fh)
            h = m.get("history", {})
            z = np.asarray(h.get("w_frac_zero") or [], dtype=float)
            sh = np.asarray(h.get("w_frac_shoulder") or [], dtype=float)
            key = z if arm == "pgd" else sh
            rows.append({
                "dataset": "FFHQ-ICNN",
                "arm": arm,
                "seed": int(os.path.basename(arm_dir).split("_seed")[1]),
                "n_evals": int(key.size),
                "occ_first": float(key[0]) if key.size else None,
                "occ_last": float(key[-1]) if key.size else None,
                "monotone_nondecreasing": (
                    bool(np.all(np.diff(key) >= -1e-12)) if key.size else None),
                "statistic": ("exactly zero" if arm == "pgd" else "shoulder"),
                "frac_zero_first": float(z[0]) if z.size else None,
                "frac_zero_last": float(z[-1]) if z.size else None,
                "below_shoulder_magnitude_first": (
                    float(sh[0]) if sh.size else None),
                "below_shoulder_magnitude_last": (
                    float(sh[-1]) if sh.size else None),
                "test_psnr_mean": m.get("test_psnr_mean"),
                "dwell": "not recoverable: scalar trace only, no per-coordinate snapshots",
                "source": arm_dir,
            })
    return rows


def endpoints_toy_pgd() -> List[dict]:
    """2-D CP-Flow toys, PGD: fraction at exactly zero at the end.

    The drivers logged ``softplus_shoulder_fraction`` for every construction,
    which is ``sigmoid(w) <= 0.05`` -- unsatisfiable for a projected
    (non-negative) weight, so the PGD column in the toy metrics is a
    structural 0.0 and says nothing. This recomputes the statistic the
    projection can produce. Final iterate only: the toy archives keep no
    snapshots.
    """
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from lift.baselines.cpflow_toy import (
        build_toy_cpflow, softplus_shoulder_fraction,
    )
    rows = []
    for d in sorted(glob.glob(os.path.join(
            _CK, "experiments_cpflow_toy_published_backend-pgd*"))):
        if not os.listdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "seed*_flow.pt"))):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            r = ck["resolved"]
            m, lift = r["model"], r["lift"]
            flow = build_toy_cpflow(
                dim=2, dimh=int(m["dimh"]), depth=int(m["depth"]),
                nblocks=int(m["nblocks"]),
                symm_act_first=bool(m["symm_act_first"]),
                softplus_type=str(m["softplus_type"]),
                zero_softplus=bool(m["zero_softplus"]),
                bias_w1=float(m["bias_w1"]),
                trainable_w0=bool(m["trainable_w0"]),
                logdet_mode=str(m["logdet_mode"]),
                m1=int(m["cpflow_config"]["m1"]),
                atol=float(m["cpflow_config"]["atol"]),
                rtol=float(m["cpflow_config"]["rtol"]),
                backend=str(lift["backend"]),
            ).double()
            flow.load_state_dict(ck["state_dict"], strict=True)
            st = softplus_shoulder_fraction(flow)
            rows.append({
                "dataset": "2D-toy",
                "target": r["data"]["target"],
                "arm": "pgd",
                "seed": int(os.path.basename(path)[4:].split("_")[0]),
                "occ_first": None,
                "occ_last": st["frac_exact_zero"],
                "n_constrained": st["n_constrained_weights"],
                "statistic": "exactly zero (recomputed; the logged shoulder "
                             "column is structurally 0 for a projected arm)",
                "dwell": "not recoverable: final flow only, no snapshots",
                "source": path,
            })
            del ck, flow
    return rows


def endpoints_hepmass_flow(n_cond: int = 1024, cond_seed: int = 0,
                           ) -> List[dict]:
    """HEPMASS convex potential flow: occupancy at the final iterate only.

    ``snap_every`` was 0 in every archived run, so nothing along training is
    on disk, and PGD has no weight file at all.
    """
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import numpy as _np
    from _experiments_cpflow_uci_lib import (
        CPFlowConfig, build_cpflow_direct, build_cpflow_hypernet, load_hepmass,
    )
    from lift.baselines.cpflow_toy import softplus_shoulder_fraction
    ds = load_hepmass()
    X = torch.as_tensor(_np.asarray(ds.train_data), dtype=torch.float32)
    g = torch.Generator().manual_seed(int(cond_seed))
    x_cond = X[torch.randperm(X.shape[0], generator=g)[:int(n_cond)]]
    rows = []
    cells = (
        [("direct", os.path.join(_CK, f"cpflow_hepmass_rerun_direct_seed{s}"),
          s) for s in range(5)]
        + [("lift", os.path.join(
            _REPO, f"checkpoints/cpflow_hepmass_200k_ps1.0_seed{s}_schema3"),
            s) for s in range(3)]
    )
    for arm, d, seed in cells:
        rp = os.path.join(d, "results.json")
        if not os.path.exists(rp):
            continue
        with open(rp) as fh:
            a = json.load(fh)["args"]
        tag = "direct" if arm == "direct" else "hypernet"
        ck_path = os.path.join(d, f"{tag}_seed{seed}_final.pt")
        if not os.path.exists(ck_path):
            rows.append({"dataset": "HEPMASS-flow", "arm": arm, "seed": seed,
                         "occ_last": None,
                         "note": "no weight file on disk", "source": d})
            continue
        cfg = CPFlowConfig(D=21, dimh=a["dimh"], nhidden=a["nhidden"],
                           nblocks=a["nblocks"])
        if arm == "direct":
            flow = build_cpflow_direct(cfg, sigma_bh=a["sigma_bh"], seed=seed)
            xc = None
        else:
            flow = build_cpflow_hypernet(
                cfg, hyper_hidden=a["hyper_hidden"], sigma_bh=a["sigma_bh"],
                seed=seed, condition_on=a["condition_on"],
                pool_norm=bool(a["pool_norm"]), pool_scale=a["pool_scale"],
                gain_cap=a["gain_cap"],
                readout_fanin_scale=bool(a.get("readout_fanin_scale", 0)))
            xc = x_cond
        sd = torch.load(ck_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        flow.load_state_dict(sd, strict=False)
        st = softplus_shoulder_fraction(flow, x_cond=xc)
        rows.append({
            "dataset": "HEPMASS-flow", "arm": arm, "seed": seed,
            "occ_first": None, "occ_last": st["fraction"],
            "n_constrained": st["n_constrained_weights"],
            "statistic": "shoulder",
            "dwell": "not recoverable: snap_every=0 in every archived run",
            "source": ck_path,
        })
        del flow, sd
    return rows


# The reported FFHQ cells, resolved by archive hash: direct and PGD from one
# three-seed archive, the lift from its own runs, one seed per archive.
def _ffhq_cells() -> Dict[str, List[str]]:
    def one(h: str, sub: str) -> List[str]:
        ds = glob.glob(os.path.join(_CK, f"*{h}", sub))
        return sorted(d for d in ds if os.path.isdir(d))
    return {
        "direct": one("3761cdab38", "direct_seed*"),
        "pgd": one("3761cdab38", "pgd_seed*"),
        "lift": (one("a039510c22", "hypernet_seed0")
                 + one("f0a57dff5e", "hypernet_seed1")
                 + one("d6a423ccce", "hypernet_seed2")),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json_out",
                   default=os.path.join(_REPO, "docs/analysis/dwell_time.json"))
    p.add_argument("--only", default=None, help="gallery|tabular")
    args = p.parse_args()
    out: Dict[str, list] = {"gallery": [], "tabular": []}

    if args.only in (None, "gallery"):
        for arm, root, tag, mode in GALLERY:
            for target in TARGETS:
                for path in sorted(glob.glob(
                        os.path.join(_CK, root, target, f"seed*_{tag}.pt"))):
                    r = scan(path, mode)
                    if r is None:
                        continue
                    r.update(arm=arm, target=target, dataset="1D-gallery",
                             seed=int(os.path.basename(path).split("_")[0][4:]))
                    r.pop("occ_series", None)
                    out["gallery"].append(r)
                    print(f"[gallery] {target:8s} {arm:7s} seed{r['seed']:<3d}"
                          f" occ {r['occ_first']:.4f}->{r['occ_last']:.4f}"
                          f" exit {r['exit_rate']:.2e}"
                          f" entry {r['entry_rate']:.2e}"
                          f" dwell {r['median_dwell_frac']:.3f}"
                          f" mono {int(r['monotone_nondecreasing'])}",
                          flush=True)

    if args.only in (None, "tabular"):
        for name, D, run_dir in TABULAR:
            for path in sorted(glob.glob(os.path.join(run_dir, "seed*_*.pt"))):
                base = os.path.basename(path)[:-3]
                seed_s, tag = base.split("_", 1)
                if tag not in MODE:
                    continue
                r = scan(path, MODE[tag])
                if r is None:
                    continue
                r.update(dataset=name, D=D, arm=ARMNAME[tag],
                         seed=int(seed_s[4:]))
                out["tabular"].append(r)
                print(f"[tabular] {name[:14]:15s} {r['arm']:7s} "
                      f"seed{r['seed']:<3d} occ {r['occ_first']:.4f}->"
                      f"{r['occ_last']:.4f} exit {r['exit_rate']:.2e}"
                      f" entry {r['entry_rate']:.2e}"
                      f" dwell {r['median_dwell_frac']:.3f}"
                      f" ever {r['ever_frac']:.4f} most {r['most_frac']:.4f}"
                      f" mono {int(r['monotone_nondecreasing'])}", flush=True)

    if args.only in (None, "endpoints"):
        out["endpoints"] = (endpoints_darcy() + endpoints_ffhq(_ffhq_cells())
                            + endpoints_toy_pgd() + endpoints_hepmass_flow())
        for r in out["endpoints"]:
            print(f"[endpoint] {r['dataset']:14s} {r.get('target','') or '':12s}"
                  f" {r['arm']:7s} seed{r['seed'] if 'seed' in r else '?'}"
                  f" {r.get('statistic','')[:28]:30s}"
                  f" first={r.get('occ_first')} last={r.get('occ_last')}"
                  f" mono={r.get('monotone_nondecreasing')}", flush=True)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
