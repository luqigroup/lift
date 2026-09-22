"""Re-derive every held-out number of the results under the best-iterate rule.

Each run is read at the iterate of its lowest held-out loss rather than at its
last one. The opening evaluations are excluded, because the held-out loss they
report is not a property of the model: in one dimension the normalizer is
quadrature and only iteration 0 misbehaves, so ``GUARD_1D`` is 200, while the
tabular problems normalize by flow importance sampling, whose estimate takes
several evaluations to settle, so ``GUARD_TAB`` is 1000.

Every number is printed from the cache the corresponding figure draws, so the
figure, the body and the appendix table cannot disagree. Run from the
repository root.
"""
from __future__ import annotations

import numpy as np

GUARD_1D = 200.0
GUARD_TAB = 1000.0
TARGETS = ("gumbel", "laplace", "gamma", "beta")
ARMS1D = ("lift", "direct", "pgd")


def _sel_1d(z, tgt: str, arm: str) -> np.ndarray:
    """TV at each seed's best held-out loss, from the density figure's cache."""
    tv = np.asarray(z[f"{tgt}/{arm}/tv"], float)
    val = np.asarray(z[f"{tgt}/{arm}/val"], float)
    it = np.asarray(z[f"{tgt}/{arm}/it"], float)[: val.shape[1]]
    ok = it >= GUARD_1D
    return np.array([tv[r, int(np.argmin(np.where(np.isfinite(val[r]) & ok, val[r], np.inf)))]
                     for r in range(val.shape[0])])


def one_d() -> None:
    z = np.load("data/analysis/convergence_1d_pinned.npz", allow_pickle=True)
    print("=" * 74)
    print(f"1-D GALLERY -- TV at the best held-out loss, guard {GUARD_1D:.0f}. Lower better.")
    print("The lift is the pinned read (emission pooled over the whole training split).")
    print("=" * 74)
    print(f"{'target':8s} {'n':>3s} {'lift':>7s} {'direct':>7s} {'PGD':>7s}   {'L<D':>6s} {'L<P':>6s}")
    tot = {"d": [0, 0], "p": [0, 0]}
    for t in TARGETS:
        L, D, P = (_sel_1d(z, t, a) for a in ARMS1D)
        tot["d"][0] += int((L < D).sum()); tot["d"][1] += len(L)
        tot["p"][0] += int((L < P).sum()); tot["p"][1] += len(L)
        print(f"{t:8s} {len(L):3d} {np.median(L):7.4f} {np.median(D):7.4f} {np.median(P):7.4f} "
              f"  {int((L<D).sum()):2d}/{len(L):<3d} {int((L<P).sum()):2d}/{len(L):<3d}")
    print(f"{'ALL':8s}     paired runs: lift < direct {tot['d'][0]}/{tot['d'][1]}, "
          f"lift < PGD {tot['p'][0]}/{tot['p'][1]}")
    print()
    print("SEED-WISE DOMINATION: is the lift's WORST seed below the baseline's BEST?")
    for t in TARGETS:
        L, D, P = (_sel_1d(z, t, a) for a in ARMS1D)
        print(f"  {t:8s} max lift {L.max():.4f} | min direct {D.min():.4f} -> "
              f"{'YES' if L.max()<D.min() else 'no '} | min PGD {P.min():.4f} -> "
              f"{'YES' if L.max()<P.min() else 'no '}")
    print()
    print("APPENDIX TABLE tab:gallery -- median [10th, 90th] over 30 seeds")
    for t in TARGETS:
        cells = []
        for a in ARMS1D:
            v = _sel_1d(z, t, a)
            cells.append(f"{np.median(v):.3f} [{np.percentile(v,10):.3f}, {np.percentile(v,90):.3f}]")
        print(f"  {t:8s} " + " & ".join(cells))
    print()


def tabular() -> None:
    d = np.load("data/analysis/tabular_multiseed.npz", allow_pickle=True)
    print("=" * 74)
    print(f"TABULAR EBM -- best held-out loss, guard {GUARD_TAB:.0f}, nats. Lower better.")
    print("The lift cannot be re-scored at an earlier iterate: its checkpoints store")
    print("one emitter snapshot against 42 iterates, so this is the RECORDED curve.")
    print("=" * 74)
    for ds in ("power", "hepmass", "miniboone"):
        print(f"\n{ds.upper()}")
        for arm, lab in (("hypernet", "lift"), ("direct", "direct"), ("pgd", "PGD")):
            per = []
            for s in np.atleast_1d(d[f"{ds}/{arm}/seeds"]):
                s = int(s)
                v = np.asarray(d[f"{ds}/{arm}/s{s}/val"], float)
                it = np.asarray(d[f"{ds}/{arm}/s{s}/it"], float)
                m = np.isfinite(v) & (it >= GUARD_TAB)
                per.append(float(np.min(v[m])) if m.any() else float("nan"))
            fin = [float(d[f"{ds}/{arm}/s{s}/test"]) for s in np.atleast_1d(d[f"{ds}/{arm}/seeds"])]
            print(f"  {lab:7s} best  " + "  ".join(f"{x:.4g}" for x in per) +
                  f"   median {np.nanmedian(per):.4g}")
            print(f"  {'':7s} final " + "  ".join(f"{x:.4g}" for x in fin) +
                  f"   median {np.nanmedian(fin):.4g}")
    print()


if __name__ == "__main__":
    one_d()
    tabular()
    print("Darcy: run scripts/render_darcy_panel.py and read its PROVENANCE block;")
    print("the framed medians are printed there over the ten seeds of each archive.")
