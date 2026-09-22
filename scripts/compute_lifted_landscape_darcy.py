r"""Lifted-:math:`(\phi, b)` loss-landscape NPZ for the Darcy PCP-Map lift.

The Darcy analogue of ``compute_lifted_landscape_hepmass.py``, for the lifted
row of the landscape composite. It needs a run whose snapshots carry the
emitter's own state dict (``snap_*.pth`` key ``hypernet``, written by
``experiments_pcpmap_darcy.py --snap_every N --save_hyper_snaps 1``); an
archive holding only the emitted weights per snapshot cannot give this panel.

Method, the lifted convention shared with HEPMASS and MiniBooNE:

1. center = the converged hypernetwork state ``hypernet_seed<S>.pth``;
2. trajectory = the per-snapshot hypernetwork states, all of them, since at
   12.0 M parameters there is nothing to subsample;
3. directions = the top-2 right singular vectors of the mean-centered
   trajectory matrix, Li-2018 filter-normalized against the center;
4. radius = ``margin`` times the trajectory's largest absolute coordinate;
5. loss at each grid point = the PCP-Map held-out NLL ``-mean log p(x | y)``
   (``PCPMapDarcy._loglik``) on the same fixed batch the constrained Darcy
   panel used, ``target.x_eval[:eval_n]`` at ``eval_n = 256``, with the
   emitted weights frozen on the whole train split exactly as in training
   (``_freeze_weights`` -> ``prepare`` -> ``emit``).

``--check_only`` evaluates the closure at the center of any run and prints it
next to the run's recorded ``final_eval_nll``.

Output keys match the HEPMASS lifted NPZ the composite reads: ``alphas,
betas, Z, coords_hyper, iters_hyper, method``, plus provenance.

Usage::

    python scripts/compute_lifted_landscape_darcy.py \
        --out_path checkpoints/landscape_composite/darcy_seed0_landscape_lifted.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
for p in (_HERE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from _darcy_render_common import (  # noqa: E402
    build_experiment, find_data, run_arm_flags,
)
from compute_lifted_landscape_hepmass import (  # noqa: E402
    _filter_normalise, _project_traj_lifted,
)


def find_lift_run(run_dir: str = "") -> str:
    """Newest ``landscape_darcy_lift_*`` archive unless one is given."""
    if run_dir:
        return run_dir
    # Matches both ``landscape_darcy_fanin_lift_*`` and
    # ``landscape_darcy_lift_*``; the newest wins, and the composite passes
    # the one it wants explicitly.
    hits = glob.glob(os.path.join(
        _REPO, "data", "checkpoints", "landscape_darcy_*lift_*"))
    if not hits:
        raise FileNotFoundError(
            "no data/checkpoints/landscape_darcy_*lift_* archive; run the "
            "seed-0 lift retrain first")
    return max(hits, key=os.path.getmtime)


def load_lift_arm(data_path: str, run_dir: str, seed: int, gpu_id: int):
    """The trained lift, model and hypernetwork, of ``run_dir``."""
    # The emitter flags (fan-in readout, standardized summary) come from the
    # run's own projorg log, so the checkpoint is loaded into the architecture
    # that produced it.
    exp = build_experiment(data_path, backends=["hypernet"], gpu_id=gpu_id,
                           **run_arm_flags(run_dir))
    arm = exp._build("hypernet", seed)
    ck = torch.load(os.path.join(run_dir, f"hypernet_seed{seed}.pth"),
                    map_location=exp.device, weights_only=False)
    arm["model"].load_state_dict(ck["model"])
    arm["hypernet"].load_state_dict(ck["hypernet"])
    return exp, arm, ck


def make_loss(exp, arm, x, y):
    """Closure: hypernetwork parameter dict -> held-out NLL on (x, y)."""
    hyp = arm["hypernet"]
    names = [k for k, _ in hyp.named_parameters()]

    def loss_at(state) -> float:
        with torch.no_grad():
            for k, p in hyp.named_parameters():
                p.copy_(state[k])
        exp._freeze_weights(arm)          # prepare on the whole train split
        return float(-exp._loglik(arm, x, y).mean().item())

    return loss_at, names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default="")
    p.add_argument("--data_path", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--eval_n", type=int, default=256,
                   help="held-out points; 256 = the constrained panel's batch")
    p.add_argument("--grid_n", type=int, default=121)
    p.add_argument("--margin", type=float, default=1.6)
    p.add_argument("--out_path", default="")
    p.add_argument("--check_only", action="store_true",
                   help="print the closure at the centre and exit")
    a = p.parse_args()

    data_path = find_data(a.data_path)
    run_dir = find_lift_run(a.run_dir)
    print(f"[darcy-lifted] data: {data_path}\n[darcy-lifted] run : {run_dir}")
    exp, arm, ck = load_lift_arm(data_path, run_dir, a.seed, a.gpu_id)
    x = exp.target.x_eval[:a.eval_n]
    y = exp.target.y_eval[:a.eval_n]
    loss_at, names = make_loss(exp, arm, x, y)
    center = OrderedDict(
        (k, ck["hypernet"][k].detach().to(exp.device).clone()) for k in names)
    P = int(sum(v.numel() for v in center.values()))
    z0 = loss_at(center)
    print(f"[darcy-lifted] P={P} hypernetwork parameters; loss at centre "
          f"(eval_n={a.eval_n}) = {z0:.4f}; recorded final_eval_nll "
          f"(full split) = {ck['record']['final_eval_nll']:.4f}")
    if a.check_only:
        print(f"[darcy-lifted] full-split NLL from the closure = "
              f"{exp._eval_nll(arm):.4f}")
        return 0

    # ---- trajectory: hypernetwork states along training ----
    snaps = []
    for f in sorted(glob.glob(os.path.join(
            run_dir, f"snaps_hypernet_seed{a.seed}", "snap_*.pth"))):
        s = torch.load(f, map_location="cpu", weights_only=False)
        if "hypernet" not in s:
            raise SystemExit(
                f"{f} has no 'hypernet' key -- the run was not trained with "
                "--save_hyper_snaps 1; this archive cannot give a lifted panel")
        snaps.append((int(s["iter"]), OrderedDict(
            (k, s["hypernet"][k].detach().float().cpu()) for k in names)))
    snaps.sort(key=lambda t: t[0])
    if not snaps:
        raise SystemExit("no snapshots found")
    n_iters = int(snaps[-1][0])
    center_cpu = OrderedDict((k, v.cpu()) for k, v in center.items())
    last_is_final = all(
        torch.equal(snaps[-1][1][k], center_cpu[k]) for k in names)
    if not last_is_final:
        snaps.append((n_iters + 1, center_cpu))   # converged state, appended
    iters = np.asarray([t for t, _ in snaps], dtype=np.int64)
    traj = [d for _, d in snaps]
    T = len(traj)
    print(f"[darcy-lifted] T={T} snapshots (iters {iters[0]}..{iters[-1]}; "
          f"last snapshot {'is' if last_is_final else 'is NOT'} the final state)")

    # ---- top-2 PCA of the mean-centered trajectory (CPU, T x P) ----
    def flat(d):
        return torch.cat([d[k].reshape(-1) for k in names])
    M = torch.stack([flat(d) for d in traj], dim=0).double()   # (T, P)
    M -= M.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    print(f"[darcy-lifted] singular values (top 5): "
          f"{[round(float(v), 3) for v in S[:5]]}")
    del M, U

    def unflat(vec):
        out, o = OrderedDict(), 0
        for k in names:
            n = center[k].numel()
            out[k] = vec[o:o + n].view_as(center[k]).float().to(exp.device)
            o += n
        return out
    d1 = _filter_normalise(unflat(Vh[0]), center)
    d2 = _filter_normalise(unflat(Vh[1]), center)
    del Vh

    coords = _project_traj_lifted(traj, center, d1, d2)
    max_abs = float(np.max(np.abs(coords))) if coords.size else 1.0
    radius = max(max_abs, 1e-12) * float(a.margin)
    alphas = np.linspace(-radius, radius, int(a.grid_n))
    betas = np.linspace(-radius, radius, int(a.grid_n))
    print(f"[darcy-lifted] radius={radius:.4g} (max_abs={max_abs:.4g}, "
          f"margin={a.margin}), grid_n={a.grid_n}")

    # ---- grid ----
    Z = np.empty((len(alphas), len(betas)), dtype=np.float64)
    t0 = time.time()
    for i, al in enumerate(alphas):
        for j, be in enumerate(betas):
            state = OrderedDict(
                (k, center[k] + float(al) * d1[k] + float(be) * d2[k])
                for k in names)
            try:
                Z[i, j] = loss_at(state)
            except (torch._C._LinAlgError, RuntimeError) as exc:
                if "linalg" not in str(exc).lower() and \
                        "ill-conditioned" not in str(exc):
                    raise
                Z[i, j] = float("inf")
        if (i + 1) % max(1, len(alphas) // 10) == 0 or i == 0:
            fin = Z[:i + 1][np.isfinite(Z[:i + 1])]
            print(f"[darcy-lifted] row {i + 1}/{len(alphas)} "
                  f"min={fin.min():.4g} max={fin.max():.4g} "
                  f"({time.time() - t0:.0f} s)")
    loss_at(center)   # leave the model at the converged state

    ia = int(np.argmin(np.abs(alphas))); ib = int(np.argmin(np.abs(betas)))
    print(f"[darcy-lifted] Z(origin)={Z[ia, ib]:.4f} Zmin={np.nanmin(Z):.4f} "
          f"at idx={np.unravel_index(int(np.nanargmin(Z)), Z.shape)}; "
          f"traj start {np.round(coords[0], 4)} final {np.round(coords[-1], 4)}")

    out_path = a.out_path or os.path.join(
        _REPO, "checkpoints", "landscape_composite",
        f"darcy_seed{a.seed}_landscape_lifted.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        method=np.array("lifted_pca_filter_norm"),
        alphas=alphas, betas=betas, Z=Z,
        coords_hyper=coords, iters_hyper=iters,
        singular_values=S[:5].numpy(),
        loss_at_center=np.float64(z0), eval_n=np.int64(a.eval_n),
        source=np.array(os.path.basename(os.path.normpath(run_dir))),
        arm_flags=np.array(repr(run_arm_flags(run_dir))),
    )
    print(f"[darcy-lifted] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
