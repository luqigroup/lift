"""CPFlow demonstration on 2-D 8-Gaussians and 2-spirals.

A convex potential flow (Huang et al. 2021) parameterizes an invertible
map ``T(z) = grad phi_theta(z)`` with ``phi_theta : R^D -> R`` strictly
convex. Maximum likelihood through the forward map would require
inverting ``T`` at every training point, so this demonstration optimizes
the solver-free reverse form instead: ``T`` maps data to latent, and

    log p_X(x) = log N(T(x); 0, I) + log det Hess phi(x).

See Huang et al. 2021, eq. (4). In 2-D the Hessian is 2x2, so its log
determinant is computed exactly rather than by a stochastic trace
estimator.

Two backends supply the ICNN's positivity-tagged weights:

* ``direct``: the tagged weights are free parameters pushed through
  ``softplus`` at every forward pass.
* ``hypernet``: a DeepSets hypernetwork emits the same tagged weights,
  conditioned on the current batch.

Outputs per target, backend and seed, in the projorg checkpoint directory:
``{target}_seed{S}_{backend}.pt`` (model state and training curves) and
``{target}_seed{S}_{backend}_test_nll.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from projorg import checkpointsdir, plotsdir
from torch import nn
from torch.func import functional_call

from lift.models import HyperNetwork, ICNN, icnn_pos_param_names


# ---------------------------------------------------------------------- targets
def sample_eight_gaussians(
    n: int, device: torch.device, radius: float = 2.0, std: float = 0.1,
) -> torch.Tensor:
    """8-Gaussians on a circle of radius ``radius`` with isotropic std."""
    centres = torch.stack(
        [
            torch.tensor(
                [radius * math.cos(2 * math.pi * k / 8),
                 radius * math.sin(2 * math.pi * k / 8)],
                device=device,
            )
            for k in range(8)
        ]
    )
    idx = torch.randint(0, 8, (n,), device=device)
    c = centres[idx]
    return c + std * torch.randn(n, 2, device=device)


def sample_two_spirals(n: int, device: torch.device) -> torch.Tensor:
    """The standard 2-spirals benchmark."""
    n_per = (n + 1) // 2
    t = torch.rand(n_per, device=device).sqrt() * 540.0 * math.pi / 180.0
    r = t + math.pi
    spiral1 = torch.stack([-r * torch.cos(t), r * torch.sin(t)], dim=-1)
    spiral2 = -spiral1
    x = torch.cat([spiral1, spiral2], dim=0)[:n]
    x = x + 0.1 * torch.randn_like(x)
    # Scale to a reasonable range for 2-D CPFlow demos.
    return x / 3.0


def sample_target(
    target: str, n: int, device: torch.device,
) -> torch.Tensor:
    if target == "eight_gaussians":
        return sample_eight_gaussians(n, device)
    if target == "two_spirals":
        return sample_two_spirals(n, device)
    raise ValueError(f"unknown target: {target}")


# ---------------------------------------------------------------------- backends
class DirectICNNParam(nn.Module):
    """ICNN parameters by direct softplus reparameterization.

    Wraps one ``ICNN`` in ``pos_constraint_mode='softplus'`` so the
    positivity-tagged weights live as unconstrained ``nn.Parameter`` s and
    become non-negative effective weights through softplus at retrieval.
    Returns a dict matching ``ICNN.named_parameters()``, mirroring
    ``HyperNetwork.forward()``'s output signature so the downstream
    ``functional_call`` path is identical for the two backends.
    """

    def __init__(
        self,
        D: int,
        hidden_dim: int,
        nlayers: int,
        strong_convexity: float,
    ) -> None:
        super().__init__()
        self._inner = ICNN(
            input_size=D,
            hidden_dim=hidden_dim,
            nlayers=nlayers,
            activation="softplus",
            strong_convexity=strong_convexity,
            pos_constraint_mode="softplus",
        )
        self._pos_names = icnn_pos_param_names(nlayers, prefix="")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in self._inner.named_parameters():
            if name in self._pos_names:
                out[name] = F.softplus(p)
            else:
                out[name] = p
        return out


def build_icnn_template(
    D: int, hidden_dim: int, nlayers: int, strong_convexity: float,
) -> ICNN:
    """A clamp-mode ICNN whose parameters are overridden by functional_call.

    The template must be in clamp mode: both backends hand it weights that
    are already non-negative, and a softplus-mode template would apply the
    readout a second time.
    """
    icnn = ICNN(
        input_size=D,
        hidden_dim=hidden_dim,
        nlayers=nlayers,
        activation="softplus",
        strong_convexity=strong_convexity,
        pos_constraint_mode="clamp",
    )
    for p in icnn.parameters():
        p.requires_grad_(False)
    return icnn


# ---------------------------------------------------------------- CPFlow math
def potential_and_grad(
    icnn_template: ICNN,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute phi(x) and grad_x phi(x) under the provided ICNN params.

    Uses ``torch.func.functional_call`` to evaluate the template ICNN
    with the (backend-emitted) ``params`` dict.
    """
    x = x.requires_grad_(True)
    phi = functional_call(icnn_template, params, (x,)).squeeze(-1)  # (B,)
    grad_phi = torch.autograd.grad(
        phi.sum(), x, create_graph=True, retain_graph=True,
    )[0]
    return phi, grad_phi


def logdet_hess_2d(
    icnn_template: ICNN,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Exact log det of the 2x2 Hessian of ``phi`` at ``x``.

    For 2-D, computing the Hessian element by element is cheap and
    avoids the stochastic-trace error of Hutchinson's estimator.
    """
    assert x.shape[-1] == 2
    x = x.requires_grad_(True)
    phi = functional_call(icnn_template, params, (x,)).squeeze(-1)
    grad_phi = torch.autograd.grad(
        phi.sum(), x, create_graph=True, retain_graph=True,
    )[0]  # (B, 2)
    g0 = grad_phi[:, 0]
    g1 = grad_phi[:, 1]
    H00 = torch.autograd.grad(
        g0.sum(), x, create_graph=True, retain_graph=True,
    )[0][:, 0]
    H01 = torch.autograd.grad(
        g0.sum(), x, create_graph=True, retain_graph=True,
    )[0][:, 1]
    H11 = torch.autograd.grad(
        g1.sum(), x, create_graph=True, retain_graph=True,
    )[0][:, 1]
    det = H00 * H11 - H01 * H01
    # A convex potential has a positive semi-definite Hessian, so the
    # determinant should be >= 0; the clamp guards against a near-zero or
    # barely negative one.
    return torch.log(det.clamp(min=1e-12))


def model_log_prob(
    icnn_template: ICNN,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """log p_model(x) under the reverse CPFlow ``T = grad phi``.

    log p_X(x) = log N(T(x); 0, I) + log det Hess phi(x).
    """
    _, T_x = potential_and_grad(icnn_template, params, x)
    log_pz = -0.5 * (T_x ** 2).sum(-1) - x.shape[-1] * 0.5 * math.log(2 * math.pi)
    log_det = logdet_hess_2d(icnn_template, params, x)
    return log_pz + log_det


# ---------------------------------------------------------------- train one run
def train_one(
    target: str,
    backend: str,
    seed: int,
    *,
    D: int = 2,
    hidden_dim: int = 64,
    nlayers: int = 3,
    strong_convexity: float = 0.05,
    hyper_hidden: Tuple[int, ...] = (96, 96, 128),
    n_iters: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_test: int = 1024,
    eval_every: int = 100,
    device: torch.device = torch.device("cpu"),
    log_every: int = 500,
    snap_every: int = 0,
) -> dict:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    template = build_icnn_template(
        D=D, hidden_dim=hidden_dim, nlayers=nlayers,
        strong_convexity=strong_convexity,
    ).to(device)
    pos_names = icnn_pos_param_names(nlayers, prefix="")

    if backend == "direct":
        param_module: nn.Module = DirectICNNParam(
            D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
        ).to(device)
    elif backend == "hypernet":
        param_module = HyperNetwork(
            input_size=D,
            hidden_sizes=list(hyper_hidden),
            downstream_network=template,
            use_outer_net=False,
            pos_param_names=pos_names,
            pos_constraint_mode="softplus",
        ).to(device)
    else:
        raise ValueError(f"unknown backend: {backend}")

    opt = torch.optim.Adam(param_module.parameters(), lr=lr)

    # Fixed held-out test split per (target, seed).
    test_seed_g = torch.Generator(device=device)
    test_seed_g.manual_seed(int(seed) + 9999)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 9999)
    with torch.no_grad():
        x_test = sample_target(target, n_test, device).detach()

    history = {
        "iter": [], "train_nll": [],
        "eval_iter": [], "test_nll": [],
    }

    # Per-iteration parameter snapshots for loss-landscape trajectory
    # overlays; snap_every <= 0 disables them. The direct backend stores the
    # effective ICNN state, the hypernet backend stores both the parameters
    # emitted at a fixed reference batch and the raw hypernetwork state dict.
    picnn_snaps: List[Dict[str, torch.Tensor]] = []
    hyper_snaps: List[Dict[str, torch.Tensor]] = []
    snap_iters: List[int] = []
    # One frozen reference batch, so the parameters emitted along the
    # trajectory are directly comparable.
    x_ref = None
    if snap_every > 0:
        with torch.no_grad():
            x_ref = sample_target(target, batch_size, device).detach()

    t0 = time.time()
    for it in range(1, n_iters + 1):
        x_batch = sample_target(target, batch_size, device)
        params = param_module(x_batch)
        logp = model_log_prob(template, params, x_batch)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_module.parameters(), max_norm=10.0)
        opt.step()

        history["iter"].append(int(it))
        history["train_nll"].append(float(loss.item()))

        if snap_every > 0 and (
            it == 1 or it % snap_every == 0 or it == n_iters
        ):
            with torch.no_grad():
                if backend == "hypernet":
                    # Emit at the reference batch so the trajectory lives in
                    # one coordinate system.
                    emitted = {
                        k: v.detach().cpu().clone()
                        for k, v in param_module(x_ref).items()
                    }
                    picnn_snaps.append(emitted)
                    hyper_snaps.append({
                        k: v.detach().cpu().clone()
                        for k, v in param_module.state_dict().items()
                    })
                else:
                    # Direct: the effective ICNN parameters are the module's
                    # output for any input, since softplus is applied to the
                    # positivity-tagged weights regardless of the batch.
                    emitted = {
                        k: v.detach().cpu().clone()
                        for k, v in param_module(
                            x_ref if x_ref is not None
                            else x_batch.detach()
                        ).items()
                    }
                    picnn_snaps.append(emitted)
                snap_iters.append(int(it))

        if (it % eval_every == 0) or (it == 1):
            with torch.no_grad():
                pass
            # Eval pass: needs grad-through-x for the Hessian; keep
            # parameters frozen by detaching the emitted dict.
            param_module.eval()
            params_eval = {k: v.detach() for k, v in
                           param_module(x_test).items()}
            x_test_g = x_test.clone().requires_grad_(True)
            logp_test = model_log_prob(template, params_eval, x_test_g)
            test_nll = float(-logp_test.detach().mean().item())
            history["eval_iter"].append(int(it))
            history["test_nll"].append(test_nll)
            param_module.train()

        if (it % log_every == 0) or (it == 1):
            print(
                f"[{target}|{backend}|seed{seed}] it={it:5d} "
                f"train_nll={loss.item():.4f} "
                f"test_nll={history['test_nll'][-1]:.4f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    # Final test eval (gradient enabled for Hessian).
    param_module.eval()
    with torch.no_grad():
        params_final = {k: v.detach() for k, v in
                        param_module(x_test).items()}
    x_test_g = x_test.clone().requires_grad_(True)
    logp_test = model_log_prob(template, params_final, x_test_g)
    final_test_nll = float(-logp_test.detach().mean().item())

    # Per-sample NLLs (for SE-of-mean), evaluated on held-out test split.
    per_sample_nll = (-logp_test.detach()).cpu().numpy().astype(np.float32)
    se_mean = float(per_sample_nll.std(ddof=1) / math.sqrt(len(per_sample_nll)))

    return {
        "history": history,
        "test_nll": final_test_nll,
        "test_nll_se": se_mean,
        "x_test": x_test.detach().cpu().numpy(),
        "params_final": {k: v.detach().cpu() for k, v in params_final.items()},
        "param_module_state": param_module.state_dict(),
        "template_state": template.state_dict(),
        "config": dict(
            target=target, backend=backend, seed=seed,
            D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
            hyper_hidden=list(hyper_hidden),
            n_iters=n_iters, batch_size=batch_size, lr=lr, n_test=n_test,
            snap_every=int(snap_every),
        ),
        # Empty when snap_every <= 0.
        "picnn_snaps": picnn_snaps,
        "hyper_snaps": hyper_snaps,
        "snap_iters": snap_iters,
        "snap_every": int(snap_every),
        "x_ref": (x_ref.detach().cpu() if x_ref is not None else None),
        "wall_seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------- driver
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="experiments_cpflow_demo_2d")
    p.add_argument("--n_iters", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--strong_convexity", type=float, default=0.05)
    p.add_argument(
        "--hyper_hidden", type=int, nargs="+", default=[96, 96, 128],
    )
    p.add_argument("--n_test", type=int, default=1024)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--targets", nargs="+",
        default=["eight_gaussians", "two_spirals"],
    )
    p.add_argument("--backends", nargs="+", default=["direct", "hypernet"])
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--snap_every", type=int, default=0,
                   help="If >0, snapshot param state every K iters into "
                        "picnn_snaps (+ hyper_snaps for hypernet backend). "
                        "Set 0 to aim for ~80 snaps; see --auto_snap_target.")
    p.add_argument("--auto_snap_target", type=int, default=0,
                   help="If >0 and --snap_every == 0, auto-set snap_every "
                        "to max(1, n_iters // auto_snap_target).")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n_iters = 200
        args.batch_size = 64
        args.seeds = [0]
        args.eval_every = 50
        args.log_every = 50

    if args.snap_every <= 0 and args.auto_snap_target > 0:
        args.snap_every = max(1, int(args.n_iters) // int(args.auto_snap_target))

    if args.device == "auto":
        device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[cpflow-demo] device={device}", flush=True)

    ckpt_dir = checkpointsdir(args.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[cpflow-demo] ckpt_dir={ckpt_dir}", flush=True)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target in args.targets:
        summary[target] = {}
        for backend in args.backends:
            summary[target][backend] = {}
            for s in args.seeds:
                print(
                    f"\n==== target={target} backend={backend} seed={s} ====",
                    flush=True,
                )
                out = train_one(
                    target=target, backend=backend, seed=s,
                    hidden_dim=args.hidden_dim, nlayers=args.nlayers,
                    strong_convexity=args.strong_convexity,
                    hyper_hidden=tuple(args.hyper_hidden),
                    n_iters=args.n_iters, batch_size=args.batch_size,
                    lr=args.lr, n_test=args.n_test,
                    eval_every=args.eval_every, log_every=args.log_every,
                    device=device,
                    snap_every=int(args.snap_every),
                )
                tag = f"{target}_seed{s}_{backend}"
                payload = {
                    "history": out["history"],
                    "test_nll": out["test_nll"],
                    "test_nll_se": out["test_nll_se"],
                    "x_test": out["x_test"],
                    "params_final": out["params_final"],
                    "param_module_state": out["param_module_state"],
                    "template_state": out["template_state"],
                    "config": out["config"],
                    "wall_seconds": out["wall_seconds"],
                    # Empty when snap_every <= 0.
                    "picnn_snaps": out.get("picnn_snaps", []),
                    "hyper_snaps": out.get("hyper_snaps", []),
                    "snap_iters": out.get("snap_iters", []),
                    "snap_every": out.get("snap_every", 0),
                    "x_ref": out.get("x_ref", None),
                }
                torch.save(payload, os.path.join(ckpt_dir, f"{tag}.pt"))
                if int(out.get("snap_every", 0)) > 0:
                    print(f"[{tag}] saved {len(out.get('snap_iters', []))} "
                          f"snapshots (snap_every={out.get('snap_every')})",
                          flush=True)
                with open(
                    os.path.join(ckpt_dir, f"{tag}_test_nll.json"), "w",
                ) as f:
                    json.dump(
                        {
                            "target": target, "backend": backend, "seed": s,
                            "test_nll": out["test_nll"],
                            "test_nll_se": out["test_nll_se"],
                            "wall_seconds": out["wall_seconds"],
                            "config": out["config"],
                        },
                        f, indent=2,
                    )
                summary[target][backend][str(s)] = {
                    "test_nll": out["test_nll"],
                    "test_nll_se": out["test_nll_se"],
                    "wall_seconds": out["wall_seconds"],
                }
                print(
                    f"[{tag}] FINAL test_nll={out['test_nll']:.4f} "
                    f"+/- SE {out['test_nll_se']:.4f}",
                    flush=True,
                )

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target, backends in summary.items():
        agg[target] = {}
        for backend, seeds in backends.items():
            arr = np.asarray([v["test_nll"] for v in seeds.values()])
            agg[target][backend] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_seeds": int(len(arr)),
                "per_seed": seeds,
            }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump({"per_seed": summary, "agg": agg}, f, indent=2)
    print("\n==== SUMMARY (mean +/- std across seeds) ====", flush=True)
    for target, backends in agg.items():
        print(f"[{target}]")
        for backend, stats in backends.items():
            print(
                f"  {backend}: test_nll = {stats['mean']:.4f} "
                f"+/- {stats['std']:.4f} (min={stats['min']:.4f}, "
                f"max={stats['max']:.4f}, n={stats['n_seeds']})",
                flush=True,
            )


if __name__ == "__main__":
    main()
