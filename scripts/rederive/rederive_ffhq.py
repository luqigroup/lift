"""FFHQ initialization axis, re-derived from each run's own metrics.json.

Runs are split by init_mode, since pooling modes mixes different starts, and
only the published setting (20 epochs, step size 5e-4) is read. PSNR is the test
mean at the best-validation iterate, in dB, higher is better; a run at or below
20 dB is counted untrained. The paired comparison uses the matched start, 'own'.
"""
from __future__ import annotations
import json, glob, re, collections
import numpy as np

rows = {}
for p in glob.glob('data/checkpoints/*ffhq*initaxis*/metrics.json'):
    m = json.load(open(p))
    a = m.get('args', {})
    if str(a.get('n_epochs')) != '20' or abs(float(a.get('lr', 0)) - 5e-4) > 1e-12:
        continue
    im = a.get('init_mode')
    fis = str(a.get('readout_fanin_scale', 0))
    for key, arm in m.get('per_arm', {}).items():
        mm = re.match(r'(.+)_seed(\d+)$', key)
        if not mm:
            continue
        b, s = mm.group(1), int(mm.group(2))
        if fis not in ('0', 'None', '0.0'):
            b += '+fanin'
        rows.setdefault((im, b, s), dict(
            init=im, backend=b, seed=s, step=arm.get('bestval_step'),
            psnr=arm.get('test_psnr_bestval_mean')))

DEPART = 20.0
print("=" * 88)
print("FFHQ INPAINTING -- test PSNR (dB, higher better) at the reported iterate,")
print(f"published step size, 2,500 steps.  'trained' = above {DEPART:.0f} dB.")
print("=" * 88)
g = collections.defaultdict(list)
for r in rows.values():
    g[(r['init'], r['backend'])].append(r)
print(f"{'init_mode':13s} {'construction':16s} {'n':>3s} {'trained':>9s} "
      f"{'med step':>9s} {'med PSNR':>9s} {'spread':>7s}")
for (im, b) in sorted(g):
    v = g[(im, b)]
    ok = [x for x in v if x['psnr'] is not None and x['psnr'] > DEPART]
    st = [x['step'] for x in ok if x['step'] is not None]
    ps = [x['psnr'] for x in ok]
    print(f"{str(im):13s} {b:16s} {len(v):3d} {len(ok):4d}/{len(v):<4d} "
          f"{(np.median(st) if st else float('nan')):9.0f} "
          f"{(np.median(ps) if ps else float('nan')):9.2f} "
          f"{((max(ps)-min(ps)) if ps else float('nan')):7.2f}")

print("\n--- the paper's matched start ('own'), paired by seed ---")
L = {r['seed']: r for r in g.get(('own', 'hypernet'), [])}
D = {r['seed']: r for r in g.get(('own', 'direct'), [])}
P = {r['seed']: r for r in g.get(('own', 'pgd'), [])}
c = sorted(set(L) & set(D))
print(f"paired seeds: {len(c)}")
both = [s for s in c if L[s]['psnr'] > DEPART and D[s]['psnr'] > DEPART]
print(f"both trained on {len(both)} of {len(c)} seeds")
if both:
    r = [D[s]['step'] / L[s]['step'] for s in both]
    print(f"  direct's reported iterate over the lift's, on those seeds: "
          f"median {np.median(r):.2f}x, range {min(r):.2f}-{max(r):.2f}x")
    dp = [L[s]['psnr'] - D[s]['psnr'] for s in both]
    print(f"  lift minus direct PSNR there: median {np.median(dp):+.2f} dB, "
          f"lift ahead on {sum(1 for x in dp if x > 0)} of {len(both)}")
print(f"lift trained on {sum(1 for s in L if L[s]['psnr']>DEPART)} of {len(L)}; "
      f"direct on {sum(1 for s in D if D[s]['psnr']>DEPART)} of {len(D)}; "
      f"PGD on {sum(1 for s in P if P[s]['psnr']>DEPART)} of {len(P)}")
