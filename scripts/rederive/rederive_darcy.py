"""Darcy initialization axis, re-derived from each cell's own results.json.

Reports held-out negative log-likelihood in nats at the final iterate,
and shoulder occupancy, the fraction of constrained weights with
sigmoid(latent) <= 0.05.
"""
from __future__ import annotations
import json, glob, os, re
import numpy as np

rows = []
for d in sorted(glob.glob('data/checkpoints/darcy_init_*')):
    b = os.path.basename(d)
    m = re.match(r'darcy_init_(\w+?)_data_tag-50k_backends-(\w+)_n_iters-(\d+)', b)
    if not m or int(m.group(3)) != 4000:
        continue
    r = json.load(open(os.path.join(d, 'results.json')))
    recs = r if isinstance(r, list) else r.get('per_run', [r])
    for rec in recs:
        rows.append(dict(setting=m.group(1), backend=m.group(2), rec=rec))

print("record fields:", sorted(rows[0]['rec'].keys()))
print()


def get(rec, *names):
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return None


print("=" * 84)
print("DARCY, 3 initializations x 3 constructions x 5 seeds, 4,000 iterations")
print("held-out NLL in nats, lower is better; occupancy = shoulder fraction")
print("=" * 84)
print(f"{'setting':12s} {'backend':9s} {'n':>2s} {'med NLL':>9s} {'occ init':>9s} {'occ end':>9s}")
cellmed = {}
for s in ['published', 'pytorch', 'harmonized']:
    for b in ['direct', 'pgd', 'hypernet']:
        v = [r['rec'] for r in rows if r['setting'] == s and r['backend'] == b]
        if not v:
            continue
        nll = np.array([get(x, 'nll', 'final_eval_nll', 'test_nll', 'eval_nll') for x in v], dtype=float)
        oi = np.array([get(x, 'shoulder_occupancy_init', 'occ_init') for x in v], dtype=float)
        oe = np.array([get(x, 'shoulder_occupancy', 'occ_end') for x in v], dtype=float)
        cellmed[(s, b)] = np.median(nll)
        print(f"{s:12s} {b:9s} {len(v):2d} {np.median(nll):9.4f} {np.nanmedian(oi):9.4f} {np.nanmedian(oe):9.4f}")

mm = list(cellmed.values())
print(f"\nnine cell medians span {min(mm):.4f} to {max(mm):.4f}  =  {max(mm)-min(mm):.4f} nats")

print("\nPAIRED differences within a setting (lift minus baseline; negative = lift ahead):")
allpair = []
for s in ['published', 'pytorch', 'harmonized']:
    def by_seed(b):
        return {get(r['rec'], 'seed'): float(get(r['rec'], 'nll', 'final_eval_nll', 'test_nll', 'eval_nll'))
                for r in rows if r['setting'] == s and r['backend'] == b}
    L, D, P = by_seed('hypernet'), by_seed('direct'), by_seed('pgd')
    cd = sorted(set(L) & set(D)); cp = sorted(set(L) & set(P)); dp = sorted(set(D) & set(P))
    ld = np.array([L[k] - D[k] for k in cd]); lp = np.array([L[k] - P[k] for k in cp])
    print(f"  {s:12s} lift-direct median {np.median(ld):+.4f} (n={len(cd)}), "
          f"lift-PGD median {np.median(lp):+.4f} (n={len(cp)})")
    allpair += [abs(x) for x in ld] + [abs(x) for x in lp] + [abs(D[k] - P[k]) for k in dp]
print(f"  LARGEST paired |difference| anywhere in the matrix: {max(allpair):.4f} nats")

print("\nDIRECT occupancy, every one of its runs:")
dd = [r['rec'] for r in rows if r['backend'] == 'direct']
print("  init:", sorted({round(float(get(x, 'shoulder_occupancy_init', 'occ_init')), 6) for x in dd}),
      " end:", sorted({round(float(get(x, 'shoulder_occupancy', 'occ_end')), 6) for x in dd}),
      f" over {len(dd)} runs")
print("PGD occupancy by setting (median init -> median end):")
for s in ['published', 'pytorch', 'harmonized']:
    v = [r['rec'] for r in rows if r['setting'] == s and r['backend'] == 'pgd']
    print(f"  {s:12s} {np.median([float(get(x,'occ_init','shoulder_init','shoulder0')) for x in v]):.4f}"
          f" -> {np.median([float(get(x,'occ_end','shoulder_end','shoulder')) for x in v]):.4f}")
