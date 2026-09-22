"""Loss-landscape slice for the Darcy PCP-Map models.

Two steps, both in this file:

1. Effective weights. The three constructions store their constrained
   weights in different representations: the lift emits them (its
   ``picnn.Lw`` is dead weight left over from construction), direct softplus
   stores the pre-softplus raw parameter, and PGD stores the
   already-projected value. Interpolating those raw state dicts compares
   objects that are not the trained models and reproduces none of the
   reported losses. Each construction is therefore first reduced to the
   post-readout weight ``W`` its forward pass actually uses, and the whole
   plane is evaluated in a cone-mode PICNN whose positivity readout is the
   identity, so every point on the plane means the same thing.

2. The slice. Rendered by ``lift.utils.landscape.render_generic`` with the
   hybrid frame ``direction_method="delta_fin_pca_orth"``: the converged
   lift sits at the origin, the converged direct model is the anchor at
   :math:`\\alpha = 1`, and the second axis is the trajectory spread
   orthogonalized against the first. Strict random directions collapse these
   surfaces to a featureless bowl.

Usage::

    python scripts/render_landscape_pcpmap_darcy.py \\
        --out_npz plots/darcy_landscape.npz --out_pdf plots/darcy_landscape.pdf
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _darcy_render_common import (  # noqa: E402
    ARMS, build_experiment, find_data, find_run, run_arm_flags,
)

from projorg import gitdir
sys.path.insert(0, gitdir())
from lift.utils.landscape.render_generic import (  # noqa: E402
    compute_landscape, render_landscape_generic,
)


def _apply_effective(model_sd, W):
    """Overwrite the convex-path weights with their post-readout values."""
    sd = {k: v.clone() for k, v in model_sd.items()}
    for k_i, w in W.items():
        sd[f"picnn.Lw.{k_i}.weight"] = w
    return sd


def snapshot_states(run: str, arm_name: str, final_sd, mode: str,
                    device: str = "cpu"):
    """Effective-weight states along one optimization trajectory.

    Snapshots store the 45 ``picnn.*`` tensors plus, for the lift, the
    emitted pre-readout weights. They do not store the three scalars
    ``w1_picnn`` / ``w2_picnn`` / ``w3_picnn`` of the potential, which do
    move during training, so those are pinned at their final values for
    every trajectory point: the drawn path is the true path in the
    45-tensor subspace with the three scalars held fixed, not the exact
    iterate. They are three numbers against several thousand, so the
    projected path is dominated by the weight matrices either way.
    """
    out = []
    for p in sorted(glob.glob(os.path.join(run, f"snaps_{arm_name}_seed0",
                                           "snap_*.pth"))):
        snap = torch.load(p, map_location="cpu", weights_only=False)
        sd = {f"picnn.{k}": v for k, v in snap["picnn"].items()}
        for k in final_sd:
            if not k.startswith("picnn."):          # w1/w2/w3, pinned
                sd[k] = final_sd[k].detach().cpu().clone()
        if "emitted" in snap:                       # the lift: W is emitted
            W = {k: F.softplus(v) for k, v in snap["emitted"].items()}
        else:                                       # direct / PGD
            W, i = {}, 0
            while f"picnn.Lw.{i}.weight" in sd:
                lw = sd[f"picnn.Lw.{i}.weight"]
                W[i] = F.softplus(lw) if mode == "softplus_reparam" else lw
                i += 1
        eff_sd = _apply_effective(sd, W)
        out.append((int(snap["iter"]),
                    {k: v.to(device) for k, v in eff_sd.items()}))
    out.sort(key=lambda t: t[0])
    return [s for _, s in out]


def arm_run(run: str, lift_run: str, arm_name: str) -> str:
    """The archive one construction is read from.

    The lift may live in its own run while direct and PGD stay in another,
    so a single slice can draw from two directories.
    """
    return (lift_run or run) if arm_name == "hypernet" else run


def effective_states(data_path: str, run: str, gpu_id: int, verbose=True,
                     lift_run: str = ""):
    """Post-readout weights and positivity modes, keyed by the ``ARMS`` names."""
    eff, modes = {}, {}
    for arm_name in ARMS:
        src = arm_run(run, lift_run, arm_name)
        flags = run_arm_flags(src) if arm_name == "hypernet" else {}
        exp = build_experiment(data_path, backends=[arm_name], gpu_id=gpu_id,
                               **flags)
        armd = exp._build(arm_name, 0)
        ck = torch.load(os.path.join(src, f"{arm_name}_seed0.pth"),
                        map_location=exp.device, weights_only=False)
        armd["model"].load_state_dict(ck["model"])
        if armd["hypernet"] is not None and ck.get("hypernet") is not None:
            armd["hypernet"].load_state_dict(ck["hypernet"])
            exp._freeze_weights(armd)

        picnn, W = armd["picnn"], {}
        with torch.no_grad():
            if armd["hypernet"] is not None:
                for k, v in armd["hypernet"].emit().items():
                    W[k] = F.softplus(v).clone()
            else:
                soft = picnn.pos_constraint_mode == "softplus_reparam"
                for k_i, lw in enumerate(picnn.Lw):
                    W[k_i] = (F.softplus(lw.weight) if soft
                              else lw.weight.clone())

        eff[arm_name] = _apply_effective(ck["model"], W)
        modes[arm_name] = picnn.pos_constraint_mode
        if verbose:
            print("  %-9s reported final_eval_nll %.3f"
                  % (arm_name, ck["record"]["final_eval_nll"]))
    return eff, modes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="")
    p.add_argument("--run_dir", default="")
    p.add_argument("--lift_run_dir", default="",
                   help="archive holding the lift (hypernet_seed0.pth + "
                        "snaps_hypernet_seed0/); default = --run_dir. Its "
                        "emitter flags are read from its projorg log.")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--grid_n", type=int, default=41)
    p.add_argument("--grid_radius", type=float, default=1.4)
    p.add_argument("--eval_n", type=int, default=256,
                   help="held-out points the NLL surface is averaged over")
    p.add_argument("--grid_chunk", type=int, default=8)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_pdf", default="",
                   help="defaults to the npz path with a .pdf suffix")
    p.add_argument("--inset_pad", type=float, default=0.014,
                   help="half-width of the zoom window around the direct / "
                        "PGD cluster; 0 disables the inset")
    p.add_argument("--inset_grid_n", type=int, default=31)
    a = p.parse_args()
    if not a.out_pdf:
        a.out_pdf = os.path.splitext(a.out_npz)[0] + ".pdf"

    data_path, run = find_data(a.data_path), find_run(a.run_dir)
    lift_run = a.lift_run_dir
    print("data:", data_path, "\nrun :", run,
          "\nlift:", lift_run or "(same run)")
    eff, modes = effective_states(data_path, run, a.gpu_id, lift_run=lift_run)
    exp0 = build_experiment(data_path, backends=["pgd"], gpu_id=a.gpu_id)

    # Optimization trajectories, present when the run was trained with
    # --snap_every > 0. Without them the trajectory degenerates to the three
    # converged solutions and the panel shows no path.
    traj = {k: snapshot_states(arm_run(run, lift_run, k), k, eff[k],
                               modes[k], exp0.device)
            for k in ARMS}
    n_snap = {k: len(v) for k, v in traj.items()}
    print("\n snapshots per arm:", n_snap)
    if not any(n_snap.values()):
        print(" WARNING: no snap_*.pth found -- retrain with --snap_every 50 "
              "or the panel will have no trajectory.")

    # Cone mode gives an identity positivity readout, so the effective
    # weights are consumed verbatim.
    exp = build_experiment(data_path, backends=["pgd"], gpu_id=a.gpu_id)
    arm = exp._build("pgd", 0)
    x, y = exp.target.x_eval[:a.eval_n], exp.target.y_eval[:a.eval_n]

    print("\n effective states, re-evaluated in the cone-mode arm:")
    for k in ARMS:
        arm["model"].load_state_dict(eff[k])
        print("  %-9s %.3f" % (k, float(-exp._loglik(arm, x, y).mean())))

    def loss_fn(params):
        arm["model"].load_state_dict(dict(params))
        return float(-exp._loglik(arm, x, y).mean().item())

    # One flat list for the renderer: each path followed by its converged
    # solution, so the PCA that fixes the second axis is taken over the
    # optimization trajectories rather than over three isolated endpoints.
    flat, span = [], {}
    for k in ARMS:
        lo = len(flat)
        flat.extend(OrderedDict(s) for s in traj[k])
        flat.append(OrderedDict(eff[k]))
        span[k] = (lo, len(flat))                    # final is span[k][1]-1

    out = render_landscape_generic(
        loss_fn,
        OrderedDict(eff["hypernet"]),                    # origin: the lift
        flat,
        grid_n=a.grid_n, grid_radius=a.grid_radius,
        direction_method="delta_fin_pca_orth",
        anchor_params=OrderedDict(eff["direct"]),        # direct at alpha=1
        out_path=a.out_pdf, save_npz_path=a.out_npz,
        title="Darcy PCP-Map: direct-anchored slice",
        vmap=False, grid_chunk=a.grid_chunk,
    )
    Z = out.get("Z") if isinstance(out, dict) else None
    if Z is not None:
        print("\n Z range [%.2f, %.3g]" % (float(np.min(Z)), float(np.max(Z))))

    tc = np.asarray(out["traj_coords"])
    store = {"Z": Z, "alphas": out["alphas"], "betas": out["betas"]}
    print(" converged solution coords:")
    for k in ARMS:
        lo, hi = span[k]
        store[f"traj_{k}"] = tc[lo:hi - 1]
        store[f"final_{k}"] = tc[hi - 1]
        print("   %-9s final (%.3f, %.3f)  path %d pts"
              % (k, tc[hi - 1][0], tc[hi - 1][1], hi - 1 - lo))

    # Zoomed inset. Direct and PGD converge on top of each other near
    # (1, 0), inside a window the parent mesh cannot resolve: at
    # grid_radius 1.4 over 41 points the spacing is 0.07 and the cluster
    # spans about 0.02. Re-evaluating the same loss on a fine window, with
    # the parent's d1 and d2 so it is the same plane, separates them.
    if a.inset_pad > 0:
        pts = np.concatenate(
            [store[f"traj_{k}"] for k in ("direct", "pgd")]
            + [np.atleast_2d(store[f"final_{k}"]) for k in ("direct", "pgd")]
        )
        ca, cb = pts[:, 0].mean(), pts[:, 1].mean()
        win_a = (ca - a.inset_pad, ca + a.inset_pad)
        win_b = (cb - a.inset_pad, cb + a.inset_pad)
        print("\n inset window alpha [%.4f, %.4f] beta [%.4f, %.4f] "
              "at %dx%d" % (*win_a, *win_b, a.inset_grid_n, a.inset_grid_n))
        ins = compute_landscape(
            loss_fn, OrderedDict(eff["hypernet"]), flat,
            grid_n=a.inset_grid_n,
            direction_method="delta_fin_pca_orth",
            anchor_params=OrderedDict(eff["direct"]),
            directions=(out["d1"], out["d2"]),      # same plane, exactly
            alpha_range=win_a, beta_range=win_b,
            vmap=False, grid_chunk=a.grid_chunk,
        )
        store.update(inset_Z=ins["Z"], inset_alphas=ins["alphas"],
                     inset_betas=ins["betas"])
        print(" inset Z range [%.3f, %.3f]"
              % (float(np.min(ins["Z"])), float(np.max(ins["Z"]))))

    store["source_run"] = np.array(os.path.basename(os.path.normpath(run)))
    store["source_lift_run"] = np.array(
        os.path.basename(os.path.normpath(lift_run or run)))
    store["eval_n"] = np.int64(a.eval_n)
    np.savez_compressed(a.out_npz, **store)
    print(" wrote", a.out_npz)


if __name__ == "__main__":
    main()
