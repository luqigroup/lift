"""Re-render the FFHQ reconstruction examples at the best-validation iterate.

The driver writes ``recon_examples.npz`` from the final iterate, while every
reported PSNR is the test PSNR at the best-validation iterate, so the stored
images and the reported numbers come from different points of the run. This
script reloads ``icnn_best.pt``, reruns the published primal-dual solver over
the test images, and writes ``recon_examples_bestval.npz`` beside the
original. Nothing is trained and nothing existing is overwritten.

    python scripts/render_ffhq_bestval_recons.py                 # all cells found
    python scripts/render_ffhq_bestval_recons.py --seeds 2       # one seed

The renderer prefers the best-val file when it exists.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, types
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
from _experiments_ffhq_icnn_regularizer_lib import FFHQICNNRegularizer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=f"{_REPO}/data/checkpoints/*ps0.5_initmatch_*/metrics.json")
    p.add_argument("--seeds", default="", help="comma-separated; default every seed found")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    want = {int(s) for s in a.seeds.split(",") if s.strip()} if a.seeds.strip() else None

    for top in sorted(glob.glob(a.glob)):
        rec = json.load(open(top)); args_d = dict(rec.get("args") or {})
        d = os.path.dirname(top)
        for arm_dir in sorted(glob.glob(d + "/*_seed*")):
            backend, seed = os.path.basename(arm_dir).rsplit("_seed", 1); seed = int(seed)
            if want is not None and seed not in want: continue
            best_path = os.path.join(arm_dir, "icnn_best.pt")
            out = os.path.join(arm_dir, "recon_examples_bestval.npz")
            if not os.path.isfile(best_path): print(f"skip {backend} s{seed}: no icnn_best.pt"); continue
            if os.path.isfile(out) and not a.force: print(f"have {backend} s{seed}"); continue

            ns = types.SimpleNamespace(**args_d)
            ns.gpu_id = a.gpu_id; ns.backends = [backend]; ns.seeds = [seed]; ns.phase = "eval"
            exp = FFHQICNNRegularizer(ns)
            ck = torch.load(best_path, map_location="cpu", weights_only=True)
            prior = exp._scratch_prior(); prior.load_state_dict(ck["prior_state"])
            test = exp._test_pairs()
            max_iter = (int(ns.eval_pdhg_max_iter) if exp.task == "inpaint" else int(ns.snp_eval_max_iter))
            n_ex = min(int(ns.n_recon_examples), len(test))
            psnrs, recons = exp._solve_psnr(prior, test.clean, test.noisy, test.mask, max_iter, return_recons=True)
            np.savez(out, clean=test.clean[:n_ex].numpy(), noisy=test.noisy[:n_ex].numpy(),
                     recon=torch.cat(recons[:n_ex], dim=0).numpy(),
                     step=np.array(int(ck["step"])), test_psnr_mean=np.array(float(np.mean(psnrs))))
            tag = rec["per_arm"].get(f"{backend}_seed{seed}", {})
            print(f"{backend:9s} s{seed}  step {int(ck['step']):5d}  test PSNR {np.mean(psnrs):.2f}"
                  f"  (metrics.json bestval {tag.get('test_psnr_bestval_mean', float('nan')):.2f})  -> {os.path.relpath(out, _REPO)}")


if __name__ == "__main__":
    main()
