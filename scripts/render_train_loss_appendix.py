r"""The training objective, or the held-out read, along training on every problem.

One panel per problem. Each curve is the pointwise median over seeds of the
objective the run itself minimized, smoothed by a running median over one
percent of the run, and the y range starts after the first tenth of training
so the first-iteration transient does not flatten the panel. ``--curve val``
draws instead the held-out read each driver logged: the pinned TV to the
target in one dimension, the validation loss on the tabular EBMs and MNIST,
the test NLL of the two-dimensional flows, the held-out NLL of the Darcy map,
and the validation PSNR of the FFHQ regularizer.

What each panel plots, from its own driver's recorded series:

* energy-based models: ``loss_E_per_iter``, the forward-KL energy loss with
  the flow importance-sampling normalizer, that is the training negative
  log-likelihood up to the estimator;
* convex potential flows in two dimensions: ``train_nll``, with the exact
  log-determinant;
* the HEPMASS flow: the logged series is a stochastic log-determinant
  surrogate whose bias drifts along training, so the panel is drawn from an
  exact re-evaluation of the stored snapshots on fixed training rows;
* the Darcy map: ``train_loss`` of the published-initialization matrix;
* the FFHQ regularizer: the critic objective, which is adversarial, so only
  its descent speed and plateau compare.

Reads the archives and trains nothing. Usage::

    python scripts/render_train_loss_appendix.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

ROOTS = [f"{_REPO}/data/_fetch_tmp", f"{_REPO}/data/checkpoints"]
MNIST = False
COL = {"lift": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c"}
DASH = {"lift": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
LABEL = {"lift": "lift", "direct": "direct softplus", "pgd": "PGD"}
Curve = Tuple[np.ndarray, np.ndarray, int]   # x, median y, n seeds
CURVE = "train"                                # set from --curve before the readers run


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if len(y) < 2 * w:
        return y
    return np.array([np.nanmedian(y[max(0, i - w):i + 1]) for i in range(len(y))])


def _median(curves: List[Tuple[np.ndarray, np.ndarray]], w: int) -> Curve:
    L = min(len(c[1]) for c in curves)
    return curves[0][0][:L], _smooth(np.nanmedian(np.stack([c[1][:L] for c in curves]), axis=0), w), len(curves)


def _ebm(files: List[str]) -> Optional[Curve]:
    curves = []
    for f in files:
        h = torch.load(f, map_location="cpu", weights_only=False)["history"]
        if CURVE == "val":
            curves.append((np.asarray(h["eval_iters"], float), np.asarray(h["val_loss_per_eval"], float)))
        else:
            y = np.asarray(h["loss_E_per_iter"], float)
            curves.append((np.arange(1, len(y) + 1, dtype=float), y))
    return _median(curves, len(curves[0][1]) // 100 if CURVE == "train" else 0) if curves else None


def _one_d_val(t: str, arm: str) -> Optional[Curve]:
    """The TV to the target along training, from the pinned convergence cache."""
    p = f"{_REPO}/data/analysis/convergence_1d_pinned.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p); M = d[f"{t}/{arm}/tv"]
    return d[f"{t}/{arm}/it"], np.nanmedian(M, axis=0), M.shape[0]


def _tab_val(ds: str, arm: str) -> Optional[Curve]:
    """The validation loss along training, from the multi-seed tabular cache."""
    p = f"{_REPO}/data/analysis/tabular_multiseed.npz"
    if not os.path.exists(p):
        return None
    d = dict(np.load(p, allow_pickle=True)); key = {"lift": "hypernet"}.get(arm, arm); ds = ds.lower()
    curves = [(d[f"{ds}/{key}/s{s}/it"], d[f"{ds}/{key}/s{s}/val"]) for s in (0, 1, 2) if f"{ds}/{key}/s{s}/val" in d]
    return _median(curves, 0) if curves else None


def _find(prefix: str, seed: int, kind: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    for root in ROOTS:
        for d in sorted(glob.glob(f"{root}/{prefix}*")):
            if any(x in os.path.basename(d) for x in exclude):
                continue
            p = f"{d}/seed{seed}_{kind}.pt"
            if os.path.exists(p):
                return p
    return None


def panels_ebm() -> List[Tuple[str, Dict[str, Curve]]]:
    out = []
    for t, lab in (("gumbel", "Gumbel"), ("laplace", "Laplace"), ("gamma", "Gamma"), ("beta", "Beta")):
        arms = {}
        for arm, arch, kind in (("lift", "lcmm_1d_logconcave_30seed_poolnorm", "hypernet"),
                                ("direct", "lcmm_1d_logconcave_30seed", "direct"),
                                ("pgd", "lcmm_1d_logconcave_30seed_pgd", "pgd")):
            c = _ebm(sorted(glob.glob(f"{_REPO}/data/checkpoints/{arch}/{t}/seed*_{kind}.pt"))) if CURVE == "train" else _one_d_val(t, arm)
            if c: arms[arm] = c
        out.append((lab, arms))
    # the tabular cells, seeds 0-2
    spec = {"POWER": {"lift": ("experiments_hypernet_vs_direct_uci_power_k1", "hypernet", ()),
                      "direct": ("experiments_hypernet_vs_direct_uci_power_k1", "direct", ()),
                      "pgd": ("uci_power_10seed_ramp", "pgd", ())},
            "HEPMASS": {"lift": ("uci_hepmass_10seed_ramp_poolnorm", "hypernet", ()),
                        "direct": ("uci_hepmass_10seed_target", "direct", ("ramp",)),
                        "pgd": ("uci_hepmass_10seed_target", "pgd", ("ramp",))},
            "MiniBooNE": {"lift": ("uci_miniboone_10seed_ramp_slr_poolnorm", "hypernet", ()),
                          "direct": ("pgd_uci_miniboone", "direct", ()),
                          "pgd": ("uci_miniboone_10seed_ramp_slr_target", "pgd", ("poolnorm",))}}
    for ds, cells in spec.items():
        arms = {}
        for arm, (prefix, kind, excl) in cells.items():
            fs = [p for p in (_find(prefix, s, kind, excl) for s in (0, 1, 2)) if p]
            c = _ebm(fs) if CURVE == "train" else _tab_val(ds, arm)
            if c: arms[arm] = c
        out.append((ds, arms))
    # The MNIST-latent panel is off by default; --with_mnist adds it.
    if MNIST:
        arms = {}
        for arm, kind in (("lift", "hypernet"), ("direct", "direct")):
            c = _ebm(sorted(glob.glob(f"{_REPO}/data/checkpoints/experiments_hypernet_vs_direct_mnist_latent_k1_*/datasets/mnist_n_iters-4000_*e4297a48*/seed*_{kind}.pt")))
            if c: arms[arm] = c
        out.append(("MNIST latents", arms))
    return out


def panels_cpflow_2d() -> List[Tuple[str, Dict[str, Curve]]]:
    out = []
    for t, lab in (("eight_gaussians", "8-Gaussians"), ("two_spirals", "2-spirals"), ("one_moon", "one-moon")):
        arms = {}
        for arm, backend in (("lift", "lifted"), ("direct", "published")):   # no PGD here: CP-Flow ships with the positivity map
            curves, seen = [], set()
            dirs = [d for d in glob.glob(f"{_REPO}/data/checkpoints/experiments_cpflow_toy_published_backend-{backend}_recipe-B_target-{t}_*") if os.path.exists(f"{d}/metrics.json")]
            for d in sorted(dirs, key=lambda d: -len(json.load(open(f"{d}/metrics.json"))["per_seed"])):
                m = json.load(open(f"{d}/metrics.json"))
                lift_cfg = m["resolved_config"].get("lift", {})
                if backend == "lifted" and not (lift_cfg.get("readout_fanin_scale") is True and float(lift_cfg.get("readout_lr_mult", 1.0)) == 1.0):
                    continue
                for s, rec in m["per_seed"].items():
                    h = rec.get("history", {})
                    if int(s) < 5 and int(s) not in seen and "train_nll" in h:
                        seen.add(int(s))
                        curves.append((np.asarray(h["iter"], float), np.asarray(h["train_nll"], float)) if CURVE == "train"
                                      else (np.asarray(h["eval_iter"], float), np.asarray(h["test_nll"], float)))
            if curves: arms[arm] = _median(curves, 3 if CURVE == "train" else 0)
        out.append((lab, arms))
    return out


def panel_hepmass_flow() -> Tuple[str, Dict[str, Curve]]:
    arms: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for f in sorted(glob.glob(f"{_REPO}/checkpoints/cpflow_hepmass_100k_*/snapshot_exact_nll.json")):
        recs = json.load(open(f))["records"]
        by = {}
        for r in recs:
            by.setdefault((r["backend"], r["seed"]), []).append((r["iter"], r["train_self"]))
        for (backend, _), pts in by.items():
            pts = sorted(pts); arm = {"hypernet": "lift", "direct": "direct", "pgd": "pgd"}[backend]
            arms.setdefault(arm, []).append((np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)))
    return ("HEPMASS flow", {a: _median(c, 1) for a, c in arms.items()})


def panel_darcy() -> Tuple[str, Dict[str, Curve]]:
    arms: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    seen = set()   # one record per (backend, seed)
    # Direct and PGD come from the published-initialization matrix cell and
    # the lift from its own cell, the same archives the Darcy panel reads.
    files = [(sorted(glob.glob(f"{_REPO}/data/checkpoints/experiments_pcpmap_darcy_*785a2e5d*/results.json")) or [""])[-1],
             (sorted(glob.glob(f"{_REPO}/data/checkpoints/darcy_improve_*6641e617*/results.json")) or [""])[-1]]
    for f in files:
        if not f:
            continue
        for rec in json.load(open(f)):
            arm = {"hypernet": "lift", "direct": "direct", "pgd": "pgd"}.get(rec["backend"]); h = rec.get("history", {})
            if arm == "lift" and "6641e617" not in f:
                continue
            if int(rec.get("seed", -1)) >= 5:   # the lift cell has seeds 0-4; keep the baselines paired with it
                continue
            if (arm, int(rec.get("seed", -1))) in seen:
                continue
            seen.add((arm, int(rec.get("seed", -1))))
            if arm and "train_loss" in h:
                arms.setdefault(arm, []).append((np.asarray(h["iters"], float), np.asarray(h["train_loss" if CURVE == "train" else "eval_nll"], float)))
    return ("Darcy map", {a: _median(c, 5 if CURVE == "train" else 0) for a, c in arms.items()})


def panel_ffhq() -> Tuple[str, Dict[str, Curve]]:
    arms: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for d in sorted(glob.glob(f"{_REPO}/data/checkpoints/*meanmatch*")):
        top = f"{d}/metrics.json"
        if not os.path.exists(top):
            continue
        try:
            a = json.load(open(top)).get("args", {}) or {}
        except (OSError, json.JSONDecodeError):
            continue
        if (a.get("n_filters"), a.get("n_epochs"), a.get("n_channels")) != (32, 20, 1) or a.get("lr") != 5e-4:
            continue
        # Every construction at the attenuated start
        # (head_init_pos_bias -5.3), the lift with pool_scale 0.5 and
        # n_cond 0, and no fan-in readout.
        if "initmatch" not in str(a.get("experiment_name", "")) or int(a.get("readout_fanin_scale", 0) or 0) != 0:
            continue
        base_cell = a.get("head_init_pos_bias") == -5.3
        lift_cell = base_cell and a.get("pool_scale") == 0.5 and int(a.get("n_cond", 0) or 0) == 0
        for f in glob.glob(f"{d}/*_seed*/metrics.json"):
            r = json.load(open(f)); arm = {"hypernet": "lift", "direct": "direct", "pgd": "pgd"}.get(r.get("backend")); h = r.get("history", {})
            if arm and not (lift_cell if arm == "lift" else base_cell):
                continue
            if arm and "loss" in h:
                arms.setdefault(arm, []).append((np.asarray(h["steps"], float), np.asarray(h["loss"], float)) if CURVE == "train"
                                                else (np.asarray(h["eval_steps"], float), np.asarray(h["val_psnr"], float)))
    return ("FFHQ regularizer" if CURVE == "train" else "FFHQ (PSNR)", {a: _median(c, 2 if CURVE == "train" else 0) for a, c in arms.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=f"{_REPO}/figures")
    p.add_argument("--mirror_to", default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--panel_h", type=float, default=0.92)
    p.add_argument("--with_mnist", action="store_true", help="add the MNIST-latent panel, which no section of the paper reports")
    p.add_argument("--skip_frac", type=float, default=0.1, help="the y-range is taken over the training after this fraction of the run")
    p.add_argument("--curve", choices=["train", "val"], default="train", help="the training objective, or the held-out read each driver logged (see the module docstring)")
    a = p.parse_args()
    global CURVE, MNIST; CURVE = a.curve; MNIST = bool(a.with_mnist)
    apply_paper_style(); matplotlib.rcParams["axes.grid"] = False

    panels = panels_ebm() + panels_cpflow_2d() + ([panel_hepmass_flow()] if CURVE == "train" else []) + [panel_darcy(), panel_ffhq()]
    panels = [(t, arms) for t, arms in panels if arms]
    # Twelve panels tile exactly four by three. Any other count falls back to
    # the widest grid that leaves no more than one gap.
    ncol = 4 if len(panels) % 4 == 0 else (5 if len(panels) % 5 <= 1 else 4)
    nrow = int(np.ceil(len(panels) / ncol))
    W = a.fig_width; H = nrow * a.panel_h + 0.42
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, H)); axes = np.atleast_1d(axes).ravel()
    for ax, (title, arms) in zip(axes, panels):
        for arm in ("direct", "pgd", "lift"):
            if arm not in arms:
                continue
            x, y, _ = arms[arm]
            ax.plot(x, y, color=COL[arm], lw=1.0 if arm != "lift" else 1.2, linestyle=DASH[arm], zorder={"lift": 3.0, "direct": 3.6, "pgd": 4.0}[arm])
        k = a.skip_frac
        lo = min(np.nanmin(v[1][int(len(v[1]) * k):]) for v in arms.values()); hi = max(np.nanmax(v[1][int(len(v[1]) * k):]) for v in arms.values())
        if lo > 0 and hi / lo > 30:
            ax.set_yscale("log"); ax.set_ylim(lo / 1.3, hi * 1.3)
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=5)); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10.0))
        else:
            pad = 0.06 * (hi - lo + 1e-9); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        if ax.get_yscale() != "log":
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.remove()
    for ax in axes[max(0, len(panels) - ncol):len(panels)]:
        ax.set_xlabel("iteration", fontsize=6.5, labelpad=1)
    for i in range(0, len(panels), ncol):
        axes[i].set_ylabel("training loss" if CURVE == "train" else "held-out", fontsize=6.5, labelpad=1)
    handles = [plt.Line2D([], [], color=COL[k], linestyle=DASH[k], lw=1.2, label=LABEL[k]) for k in ("direct", "lift", "pgd")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.5, w_pad=0.5)
    os.makedirs(a.out_dir, exist_ok=True); stem = os.path.join(a.out_dir, "train_loss_all" if CURVE == "train" else "val_loss_all")
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
    print(f"wrote {stem}.pdf ({W:.2f} x {H:.2f} in), panels: " + ", ".join(t for t, _ in panels))
    for t, arms in panels:
        print("  " + f"{t:14s} " + "  ".join(f"{k}: n={v[2]}" for k, v in arms.items()))
    if a.mirror_to and os.path.isdir(a.mirror_to):
        import shutil; shutil.copy(stem + ".pdf", a.mirror_to); print(f"mirrored to {a.mirror_to}")


if __name__ == "__main__":
    main()
