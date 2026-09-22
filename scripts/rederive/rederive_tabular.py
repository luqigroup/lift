"""Tabular energy-based models, re-derived from their own records.

Held-out test negative log-likelihood at the final iterate, in nats, lower is
better. The lift is read with its emission pooled over the whole training split;
direct softplus and PGD read no batch. Shoulder occupancy is the fraction of
direct's constrained weights at or below softplus(logit(0.05)) = 0.0513.
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, 'scripts')

z = np.load('data/analysis/tabular_multiseed.npz', allow_pickle=True)
K = set(z.keys())
PIN = {'power': 'uci_power_lift', 'hepmass': 'uci_hepmass_lift',
       'miniboone': 'uci_miniboone_lift'}

print("=" * 72)
print("TABULAR ENERGY-BASED MODELS -- held-out test NLL, final iterate, nats")
print("=" * 72)
for ds in ['power', 'hepmass', 'miniboone']:
    print(f"\n{ds.upper()}   schedules: " + ", ".join(
        f"{a}={z[f'{ds}/{a}/schedule']}" for a in ['hypernet', 'direct', 'pgd']
        if f'{ds}/{a}/schedule' in K))
    pin = json.load(open(f'data/analysis/pinned/{PIN[ds]}.json'))['seeds']
    for arm in ['hypernet', 'direct', 'pgd']:
        vals, seeds = [], []
        for s in range(3):
            if arm == 'hypernet':
                if str(s) in pin:
                    vals.append(pin[str(s)]['pinned']['test_nll']); seeds.append(s)
            else:
                k = f'{ds}/{arm}/s{s}/test'
                if k in K and np.isfinite(float(z[k])):
                    vals.append(float(z[k])); seeds.append(s)
        if vals:
            lab = {'hypernet': 'lift', 'direct': 'direct', 'pgd': 'PGD'}[arm]
            print(f"  {lab:7s} seeds {seeds} -> " +
                  "  ".join(f"{v:.3f}" for v in vals) +
                  f"   median {np.median(vals):.3f}")
    sh = [f'{ds}/direct/s{s}/shoulder' for s in range(3)]
    sh0 = [f'{ds}/direct/s{s}/shoulder0' for s in range(3)]
    have = [(float(z[a]), float(z[b])) for a, b in zip(sh0, sh) if a in K and b in K]
    if have:
        print(f"  direct shoulder occupancy, init -> end: " +
              ", ".join(f"{a:.2f}->{b:.2f}" for a, b in have))
