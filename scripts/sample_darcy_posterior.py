"""Cache amortized posterior samples from the trained Darcy PCP-Maps.

PCP-Map is an amortized conditional sampler: one trained map serves any
observation, so a posterior sample is :math:`z \\sim N(0, I)` followed by
solving :math:`\\nabla_x F(x, y) = z`. :math:`F` is strongly convex in
:math:`x` (the quadratic floor lower-bounds its Hessian), so that solve is
the unique minimizer of the convex-conjugate objective
:math:`F(x, y) - \\langle z, x \\rangle` and LBFGS on it is well posed --
the same inverse-Brenier route the CP-Flow renderers use.

The solve needs gradients only, never the :math:`D \\times D` Hessian the
density path builds, so rows are chunked at ``--row_chunk`` and peak memory
is bounded however many observations times samples are asked for.

Writes one npz holding the KL coefficients for every method plus the ground
truth and the observations; the renderer never re-runs the inversion. Each
method reports a round-trip residual
:math:`\\|\\nabla_x F(x^\\star, y) - z\\| / \\|z\\|` so a silently
unconverged solve cannot pass as a posterior sample.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _darcy_render_common import (  # noqa: E402
    ARMS, build_experiment, find_data, find_run, run_arm_flags,
)


def sample_arm(model, y, n_samples, D, device, row_chunk, iters):
    """(n_obs, n_samples, D) posterior samples, chunked over rows."""
    n_obs = y.shape[0]
    y_rep = y.repeat_interleave(n_samples, dim=0)
    z = torch.randn(n_obs * n_samples, D, device=device)
    out = torch.empty(n_obs * n_samples, D, device="cpu")
    for lo in range(0, n_obs * n_samples, row_chunk):
        hi = min(lo + row_chunk, n_obs * n_samples)
        yb, zb = y_rep[lo:hi], z[lo:hi]
        x = torch.zeros(hi - lo, D, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([x], max_iter=iters,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            obj = (model.get_picnn(x, yb).squeeze(-1) - (zb * x).sum(-1)).sum()
            obj.backward()
            return obj

        opt.step(closure)
        out[lo:hi] = x.detach().cpu()
        del x, opt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out.view(n_obs, n_samples, D).numpy(), z.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="")
    p.add_argument("--run_dir", default="")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--n_obs", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--row_chunk", type=int, default=256)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma list of constructions to sample (default all three)")
    p.add_argument("--lift_run_dir", default="",
                   help="archive holding the lift; default = --run_dir. "
                        "Its emitter flags (fan-in readout, pool_norm) are "
                        "read from its projorg log.")
    p.add_argument("--base_cache", default="",
                   help="an earlier npz from this script; constructions not in --arms "
                        "(and xi_true / y_obs) are copied from it, so the "
                        "lift can be re-sampled from a new run while direct "
                        "and PGD keep their paper-archive draws")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    data_path, run = find_data(a.data_path), find_run(a.run_dir)
    lift_run = a.lift_run_dir or run
    arms = [x for x in a.arms.split(",") if x]
    exp = build_experiment(data_path, gpu_id=a.gpu_id, eval_chunk=250,
                           **run_arm_flags(lift_run))
    print("run :", run, "\nlift:", lift_run, "\narms:", arms)

    y = exp.target.y_eval[:a.n_obs]
    store = {"xi_true": exp.target.x_eval[:a.n_obs].cpu().numpy(),
             "y_obs": y.cpu().numpy()}
    if a.base_cache:
        base = np.load(a.base_cache)
        for k in ("xi_true", "y_obs"):
            if not np.allclose(base[k][:a.n_obs], store[k]):
                raise SystemExit(f"--base_cache {k} differs from this eval "
                                 "split; refusing to mix")
        for arm in ARMS:
            if arm not in arms and f"xi_{arm}" in base.files:
                store[f"xi_{arm}"] = base[f"xi_{arm}"]
                print(f"  {arm:9s} copied from {a.base_cache}")
    for arm in arms:
        src = lift_run if arm == "hypernet" else run
        armd = exp._build(arm, 0)
        ck = torch.load(os.path.join(src, f"{arm}_seed0.pth"),
                        map_location=exp.device, weights_only=False)
        armd["model"].load_state_dict(ck["model"])
        if armd["hypernet"] is not None and ck.get("hypernet") is not None:
            armd["hypernet"].load_state_dict(ck["hypernet"])
            exp._freeze_weights(armd)
        xs, z = sample_arm(armd["model"], y, a.n_samples, exp.target.D,
                           exp.device, a.row_chunk, a.iters)
        store[f"xi_{arm}"] = xs

        # Round-trip check on the first sample of each observation: the
        # solution must reproduce the z it was solved for.
        xf = torch.tensor(xs[:, 0, :], device=exp.device,
                          dtype=torch.float32).requires_grad_(True)
        zz = armd["model"].gxinv(xf, y).detach().cpu().numpy()
        z0 = z.reshape(a.n_obs, a.n_samples, -1)[:, 0, :]
        rel = float(np.linalg.norm(zz - z0) / np.linalg.norm(z0))
        print(f"  {arm:9s} samples {xs.shape}  round-trip rel err {rel:.3e}")
        del armd
        if exp.device.startswith("cuda"):
            torch.cuda.empty_cache()

    store["source_run"] = np.array(os.path.basename(os.path.normpath(run)))
    store["source_lift_run"] = np.array(
        os.path.basename(os.path.normpath(lift_run)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **store)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
