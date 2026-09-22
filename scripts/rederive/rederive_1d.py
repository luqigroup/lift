"""1-D gallery, re-derived from its own records.

lift   data/checkpoints/lcmm_1d_logconcave_30seed_poolnorm, key 'hypernet',
       re-evaluated with the whole training split pooled (data/analysis/pinned/)
direct lcmm_1d_logconcave_30seed, key 'direct'
PGD    lcmm_1d_logconcave_30seed_pgd, key 'pgd'
The metric is TV to the target at the final iterate. Lower is better.
"""
from __future__ import annotations
import json, os
import numpy as np

TARGETS = ["gumbel", "laplace", "gamma", "beta"]
ARM = {"direct": ("data/checkpoints/lcmm_1d_logconcave_30seed", "direct"),
       "pgd": ("data/checkpoints/lcmm_1d_logconcave_30seed_pgd", "pgd")}


def base(arm, t):
    arch, key = ARM[arm]
    ps = json.load(open(os.path.join(arch, t, "metrics.json")))["per_seed"]
    return {int(s): v[key]["tv"] for s, v in ps.items() if key in v}


def lift(t):
    d = json.load(open(f"data/analysis/pinned/lcmm_1d_poolnorm_{t}.json"))
    return {int(k): v["pinned"]["tv"] for k, v in d["seeds"].items()}


print("=" * 74)
print("1-D GALLERY -- TV to the target, final iterate, paired seeds. Lower better.")
print("=" * 74)
hdr = f"{'target':9s} {'n':>3s} {'lift':>7s} {'direct':>7s} {'PGD':>7s}  {'L<D':>6s} {'L<P':>6s}"
print(hdr)
for t in TARGETS:
    L, D, P = lift(t), base("direct", t), base("pgd", t)
    c = sorted(set(L) & set(D) & set(P))
    l = np.array([L[s] for s in c]); d = np.array([D[s] for s in c]); p = np.array([P[s] for s in c])
    print(f"{t:9s} {len(c):3d} {np.median(l):7.4f} {np.median(d):7.4f} {np.median(p):7.4f}"
          f"  {int((l<d).sum()):3d}/{len(c):<2d} {int((l<p).sum()):3d}/{len(c):<2d}")
print("\nBANDS (10th-90th percentile over seeds): does the spread cover the gap?")
for t in TARGETS:
    L, D, P = lift(t), base("direct", t), base("pgd", t)
    c = sorted(set(L) & set(D) & set(P))
    l = np.array([L[s] for s in c]); d = np.array([D[s] for s in c]); p = np.array([P[s] for s in c])
    def band(x): return f"[{np.percentile(x,10):.4f},{np.percentile(x,90):.4f}]"
    ld = "DISJOINT" if np.percentile(l, 90) < np.percentile(d, 10) else "overlap"
    lp = "DISJOINT" if np.percentile(l, 90) < np.percentile(p, 10) else "overlap"
    print(f"  {t:9s} lift {band(l)}  direct {band(d)} -> {ld:8s} |  PGD {band(p)} -> {lp}")
