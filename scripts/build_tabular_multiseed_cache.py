"""Build ``data/analysis/tabular_multiseed.npz``, the tabular figure's cache.

Reads the training history out of the ten-seed tabular ICNN-EBM checkpoints and
keeps only what the figure needs, per seed: the evaluation iterations, the
validation loss at each, the final held-out test NLL, the abort iteration if the
run was aborted, and, for direct softplus, the shoulder share at the first and
last emitted snapshot.

``SEL`` names the archive each (dataset, construction) pair is read from, and
metrics come from each archive's ``metrics.json``; where an archive is not
local, its ``metrics.json`` is read from the mirror named by ``META`` and its
``*.pt`` files are expected under ``FETCH``.

The output is consumed by ``scripts/render_tabular_val_loss_multiseed.py``.
"""
import glob, json, math, os, sys
import numpy as np
import torch
from projorg import gitdir

# projorg locates the project; nothing here hard-codes a path.
REPO = gitdir()
LCMM = ""   # an optional second checkpoint root outside the project
FETCH = f"{REPO}/data/_fetch_tmp"
OUT = f"{REPO}/data/analysis/tabular_multiseed.npz"

# (dataset, construction) -> (schedule label, exact experiment-name prefixes,
# seeds).
#
# The prefixes are matched against ``dirname.split("_target_kind")[0]`` for
# EQUALITY, never by glob. A glob is unsafe here: "uci_miniboone_10seed_ramp_slr_*"
# also matches "uci_miniboone_10seed_ramp_slr_poolnorm_...", a different run with
# the same seed numbers, and whichever sorted first would silently win.
ROOTS = [f"{FETCH}", f"{REPO}/data/checkpoints", LCMM]
# PINNED=0 in the environment restores the archives' batch-conditioned lift values.
PINNED = os.environ.get("PINNED", "1") == "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pinned_lib  # noqa: E402  the lift's final test NLL with its emission pinned on the full training split
SEL = {
    ("power", "hypernet"):     ("no ramp",  ["experiments_hypernet_vs_direct_uci_power_k1"], [0, 1, 2]),
    ("power", "direct"):       ("no ramp",  ["experiments_hypernet_vs_direct_uci_power_k1"], [0, 1, 2]),
    ("power", "pgd"):          ("ramp",     ["uci_power_10seed_ramp"], [0, 1, 2]),
    ("hepmass", "hypernet"):   ("ramp+std", ["uci_hepmass_10seed_ramp_poolnorm"], [0, 1, 2]),
    ("hepmass", "direct"):     ("no ramp",  ["uci_hepmass_10seed"], [0, 1, 2]),
    ("hepmass", "pgd"):        ("no ramp",  ["uci_hepmass_10seed"], [0, 1, 2]),
    ("miniboone", "hypernet"): ("ramp+slr+std", ["uci_miniboone_10seed_ramp_slr_poolnorm"], [0, 1, 2]),
    # Pinned to the hash: two archives carry the name ``pgd_uci_miniboone`` and
    # both hold a seed0_direct.pt. 0abf822e is the three-seed one with a
    # metrics.json; eb8c9cb9 is a two-seed fragment.
    ("miniboone", "direct"):   ("no ramp",  ["pgd_uci_miniboone@0abf822e"], [0, 1, 2]),
    ("miniboone", "pgd"):      ("ramp+slr", ["uci_miniboone_10seed_ramp_slr"], [0, 1, 2]),
}

# final test NLL and divergence flag, keyed by the archive hash (metrics.json mirror)
META = {}
MIRROR = os.environ.get("ICNNLIFT_DBX_METRICS_MIRROR", f"{REPO}/data/analysis/dbx_metrics")
for p in glob.glob(f"{MIRROR}/*/metrics.json") + glob.glob(f"{FETCH}/*/metrics.json") + \
         glob.glob(f"{REPO}/data/checkpoints/*/metrics.json") + glob.glob(f"{LCMM}/*/metrics.json"):
    h = os.path.basename(os.path.dirname(p)).rsplit(".", 1)[-1][:8]
    try:
        META[h] = json.load(open(p))
    except Exception:
        pass

THR_W = float(np.log1p(np.exp(np.log(0.05 / 0.95))))
POS = ("components.0.output_layer.weight", "components.0.z_layers.0.weight", "components.0.z_layers.1.weight",
       "components.0.z_layers.2.weight", "components.0.z_layers.3.weight")


def _dirs_named(prefix):
    """Every archive whose experiment name is EXACTLY ``prefix`` (not a prefix
    match: ``uci_miniboone_10seed_ramp_slr`` must not pick up ``..._poolnorm``).

    ``name@hashprefix`` additionally pins the archive hash, for the names that
    more than one archive carries.
    """
    prefix, _, want_hash = prefix.partition("@")
    out = []
    for root in ROOTS:
        for d in sorted(glob.glob(f"{root}/*_target_kind-*")):
            b = os.path.basename(d)
            if b.split("_target_kind")[0] != prefix:
                continue
            if want_hash and not b.rsplit(".", 1)[-1].startswith(want_hash):
                continue
            out.append(d)
    return out


def find_file(prefixes, seed, arm):
    hits = [f"{d}/seed{seed}_{arm}.pt" for pre in prefixes for d in _dirs_named(pre)]
    hits = [p for p in hits if os.path.isfile(p)]
    if len(hits) > 1:
        raise SystemExit(f"ambiguous: seed{seed}_{arm}.pt in {len(hits)} archives of {prefixes}:\n  "
                         + "\n  ".join(hits))
    return hits[0] if hits else None


def final_test(path, seed, arm):
    d = os.path.dirname(path)
    bv = f"{d}/seed{seed}_test_nll_at_bestval.json"
    if os.path.isfile(bv):
        e = json.load(open(bv)).get("per_backend", {}).get(arm, {})
        if "test_nll_at_final" in e:
            return float(e["test_nll_at_final"]), float("nan")
    h = os.path.basename(d).rsplit(".", 1)[-1][:8]
    m = META.get(h, {}).get("per_seed", {}).get(str(seed), {}).get(arm, {})
    return m.get("test_nll", float("nan")), m.get("diverged_at_iter", float("nan"))


def main():
    out = {}
    for (ds, arm), (sched, prefixes, seeds) in SEL.items():
        out[f"{ds}/{arm}/schedule"] = np.array(sched)
        got = []
        for s in seeds:
            p = find_file(prefixes, s, arm)
            if p is None:
                print(f"MISSING {ds} {arm} seed{s}", flush=True)
                continue
            try:
                ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
            except Exception:
                ck = torch.load(p, map_location="cpu", weights_only=False)
            h = ck["history"]
            it = np.asarray(h["eval_iters"], float)
            val = np.asarray(h["val_loss_per_eval"], float)
            tn, div = final_test(p, s, arm)
            out[f"{ds}/{arm}/s{s}/it"] = it
            out[f"{ds}/{arm}/s{s}/val"] = val
            # the objective the run minimized, per iteration
            tr = np.asarray(h["loss_E_per_iter"], float)
            out[f"{ds}/{arm}/s{s}/train"] = tr
            out[f"{ds}/{arm}/s{s}/train_it"] = np.arange(1, len(tr) + 1, dtype=float)
            if PINNED and arm == "hypernet":
                tn_p = _pinned_lib.tabular_test_nll(ds, s)
                if tn_p is None:
                    raise SystemExit(f"no pinned re-evaluation for {ds} lift seed {s}")
                out[f"{ds}/{arm}/s{s}/test_batch"] = np.array(float(tn) if tn is not None else np.nan)
                tn = tn_p
            out[f"{ds}/{arm}/s{s}/test"] = np.array(float(tn) if tn is not None else np.nan)
            out[f"{ds}/{arm}/s{s}/div"] = np.array(float(div) if div is not None else np.nan)
            if arm == "direct" and ck.get("theta_snaps"):
                for tag, sn in (("shoulder0", ck["theta_snaps"][0]), ("shoulder", ck["theta_snaps"][-1])):
                    w = torch.cat([sn[k].reshape(-1).double() for k in POS if k in sn])
                    out[f"{ds}/direct/s{s}/{tag}"] = np.array(float((w <= THR_W).double().mean()))
            got.append(s)
            print(f"{ds:10s} {arm:9s} seed{s:<2d} evals={len(it):3d} last_it={it[-1]:6.0f} "
                  f"val_end={val[-1]:12.4g} test={tn if tn is None else float(tn):12.4g} div={div}", flush=True)
            del ck
        out[f"{ds}/{arm}/seeds"] = np.array(got, dtype=int)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, os.path.getsize(OUT) / 1e3, "kB")


if __name__ == "__main__":
    main()
