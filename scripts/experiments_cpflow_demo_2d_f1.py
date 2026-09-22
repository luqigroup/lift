"""CPFlow on the 2-D 8-Gaussians and 2-spirals targets.

Trains the convex potential flow over a sweep of seeds with either the lift
or direct softplus as the parameter source, and writes one checkpoint and
one test-NLL record per (target, method, seed). ``--save_trajectory_every K``
stores the full ``param_module`` state every K iterations for the
filter-norm landscape renderer; it is off by default because the snapshots
are large. ``--skip_existing`` resumes a chunked sweep by skipping any
``{tag}_test_nll.json`` already on disk.
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
from projorg import checkpointsdir
from torch import nn
from torch.func import functional_call

from lift.models import HyperNetwork, ICNN, icnn_pos_param_names


# ---------------------------------------------------------------------- targets
def sample_eight_gaussians(
    n: int, device: torch.device, radius: float = 2.0, std: float = 0.1,
) -> torch.Tensor:
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
    n_per = (n + 1) // 2
    t = torch.rand(n_per, device=device).sqrt() * 540.0 * math.pi / 180.0
    r = t + math.pi
    spiral1 = torch.stack([-r * torch.cos(t), r * torch.sin(t)], dim=-1)
    spiral2 = -spiral1
    x = torch.cat([spiral1, spiral2], dim=0)[:n]
    x = x + 0.1 * torch.randn_like(x)
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
    x = x.requires_grad_(True)
    phi = functional_call(icnn_template, params, (x,)).squeeze(-1)
    grad_phi = torch.autograd.grad(
        phi.sum(), x, create_graph=True, retain_graph=True,
    )[0]
    return phi, grad_phi


def logdet_hess_2d(
    icnn_template: ICNN,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    assert x.shape[-1] == 2
    x = x.requires_grad_(True)
    phi = functional_call(icnn_template, params, (x,)).squeeze(-1)
    grad_phi = torch.autograd.grad(
        phi.sum(), x, create_graph=True, retain_graph=True,
    )[0]
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
    return torch.log(det.clamp(min=1e-12))


def model_log_prob(
    icnn_template: ICNN,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    _, T_x = potential_and_grad(icnn_template, params, x)
    log_pz = -0.5 * (T_x ** 2).sum(-1) - x.shape[-1] * 0.5 * math.log(2 * math.pi)
    log_det = logdet_hess_2d(icnn_template, params, x)
    return log_pz + log_det


# ---------------------------------------- positivity bias-init helper
def randomise_hypernet_pos_bias(
    hypernet: "HyperNetwork", sigma_bh: float, generator: torch.Generator,
) -> None:
    """Replace each positivity-tagged readout head's bias with N(0, sigma_bh^2).

    The default initialization puts one constant bias on every softplus-mode
    positivity head, which starts every readout at the same small value. A
    per-element Gaussian of width ``sigma_bh`` breaks that uniformity both
    across heads and within a head.
    """
    with torch.no_grad():
        for name, layer in zip(hypernet._param_names, hypernet.weight_predictors):
            if name not in hypernet.pos_param_names:
                continue
            layer.bias.normal_(mean=0.0, std=float(sigma_bh), generator=generator)


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
    hyper_hidden: Tuple[int, ...] = (192, 192, 256),
    sigma_bh: float = 0.5,
    n_iters: int = 6000,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_test: int = 1024,
    eval_every: int = 200,
    device: torch.device = torch.device("cpu"),
    log_every: int = 1000,
    save_trajectory_every: int = 0,
    save_train_history: bool = True,
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
        # Replace the constant positivity-head bias init with a Gaussian.
        g_init = torch.Generator(device=device)
        g_init.manual_seed(int(seed) + 13_579)
        randomise_hypernet_pos_bias(param_module, sigma_bh, g_init)
    else:
        raise ValueError(f"unknown backend: {backend}")

    opt = torch.optim.Adam(param_module.parameters(), lr=lr)

    test_seed_g = torch.Generator(device=device)
    test_seed_g.manual_seed(int(seed) + 9999)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 9999)
    with torch.no_grad():
        x_test = sample_target(target, n_test, device).detach()

    history: Dict[str, list] = {
        "iter": [], "train_nll": [],
        "eval_iter": [], "test_nll": [],
    }
    trajectory: List[Dict[str, torch.Tensor]] = []

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

        if save_train_history:
            history["iter"].append(int(it))
            history["train_nll"].append(float(loss.item()))

        if save_trajectory_every > 0 and (
            it == 1 or it % save_trajectory_every == 0 or it == n_iters
        ):
            trajectory.append(
                {k: v.detach().cpu().clone()
                 for k, v in param_module.state_dict().items()}
            )

        if (it % eval_every == 0) or (it == 1):
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

    param_module.eval()
    with torch.no_grad():
        params_final = {k: v.detach() for k, v in
                        param_module(x_test).items()}
    x_test_g = x_test.clone().requires_grad_(True)
    logp_test = model_log_prob(template, params_final, x_test_g)
    final_test_nll = float(-logp_test.detach().mean().item())
    per_sample_nll = (-logp_test.detach()).cpu().numpy().astype(np.float32)
    se_mean = float(per_sample_nll.std(ddof=1) / math.sqrt(len(per_sample_nll)))

    return {
        "history": history,
        "trajectory": trajectory,
        "test_nll": final_test_nll,
        "test_nll_se": se_mean,
        "x_test": x_test.detach().cpu().numpy(),
        "params_final": {k: v.detach().cpu() for k, v in params_final.items()},
        "param_module_state": {k: v.detach().cpu()
                               for k, v in param_module.state_dict().items()},
        "template_state": {k: v.detach().cpu()
                           for k, v in template.state_dict().items()},
        "config": dict(
            target=target, backend=backend, seed=seed,
            D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
            hyper_hidden=list(hyper_hidden),
            sigma_bh=float(sigma_bh),
            n_iters=n_iters, batch_size=batch_size, lr=lr, n_test=n_test,
            save_trajectory_every=int(save_trajectory_every),
        ),
        "wall_seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------- driver
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="experiments_cpflow_demo_2d_f1")
    p.add_argument("--n_iters", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--strong_convexity", type=float, default=0.05)
    p.add_argument(
        "--hyper_hidden", type=int, nargs="+", default=[192, 192, 256],
    )
    p.add_argument("--sigma_bh", type=float, default=0.5)
    p.add_argument("--n_test", type=int, default=1024)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument(
        "--targets", nargs="+",
        default=["eight_gaussians", "two_spirals"],
    )
    p.add_argument("--backends", nargs="+", default=["direct", "hypernet"])
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--save_trajectory_every", type=int, default=0,
                   help="If >0, save the param_module state every K iters.")
    p.add_argument("--no_train_history", action="store_true",
                   help="Skip per-iter train-nll history to save disk.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip (target, backend, seed) triples whose "
                        "test_nll.json already exists.")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n_iters = 200
        args.batch_size = 64
        args.start_seed = 0
        args.n_seeds = 1
        args.eval_every = 50
        args.log_every = 50

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))

    if args.device == "auto":
        device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[cpflow-demo-f1] device={device}", flush=True)

    ckpt_dir = checkpointsdir(args.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[cpflow-demo-f1] ckpt_dir={ckpt_dir}", flush=True)
    print(f"[cpflow-demo-f1] seeds=[{seeds[0]}..{seeds[-1]}] "
          f"({len(seeds)} seeds), targets={args.targets}, "
          f"backends={args.backends}", flush=True)

    summary: Dict[str, Dict[str, Dict[str, dict]]] = {}
    n_total = len(args.targets) * len(args.backends) * len(seeds)
    n_done = 0
    n_skipped = 0
    t0_all = time.time()
    for target in args.targets:
        summary[target] = {}
        for backend in args.backends:
            summary[target][backend] = {}
            for s in seeds:
                tag = f"{target}_seed{s}_{backend}"
                nll_path = os.path.join(ckpt_dir, f"{tag}_test_nll.json")
                if args.skip_existing and os.path.exists(nll_path):
                    with open(nll_path) as f:
                        rec = json.load(f)
                    summary[target][backend][str(s)] = {
                        "test_nll": rec["test_nll"],
                        "test_nll_se": rec["test_nll_se"],
                        "wall_seconds": rec.get("wall_seconds", 0.0),
                    }
                    n_skipped += 1
                    n_done += 1
                    continue

                n_done += 1
                print(
                    f"\n==== [{n_done}/{n_total}] target={target} "
                    f"backend={backend} seed={s} (elapsed_all="
                    f"{(time.time()-t0_all)/60.0:.1f} min) ====",
                    flush=True,
                )
                out = train_one(
                    target=target, backend=backend, seed=s,
                    hidden_dim=args.hidden_dim, nlayers=args.nlayers,
                    strong_convexity=args.strong_convexity,
                    hyper_hidden=tuple(args.hyper_hidden),
                    sigma_bh=args.sigma_bh,
                    n_iters=args.n_iters, batch_size=args.batch_size,
                    lr=args.lr, n_test=args.n_test,
                    eval_every=args.eval_every, log_every=args.log_every,
                    device=device,
                    save_trajectory_every=args.save_trajectory_every,
                    save_train_history=not args.no_train_history,
                )

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
                }
                if out["trajectory"]:
                    payload["trajectory"] = out["trajectory"]
                torch.save(payload, os.path.join(ckpt_dir, f"{tag}.pt"))
                with open(nll_path, "w") as f:
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
                    f"+/- SE {out['test_nll_se']:.4f} "
                    f"wall={out['wall_seconds']:.1f}s",
                    flush=True,
                )

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target, backends in summary.items():
        agg[target] = {}
        for backend, seeds_d in backends.items():
            arr = np.asarray([v["test_nll"] for v in seeds_d.values()])
            agg[target][backend] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_seeds": int(len(arr)),
                "per_seed": seeds_d,
            }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump({"per_seed": summary, "agg": agg}, f, indent=2)
    print(f"\n[cpflow-demo-f1] skipped {n_skipped} pre-existing runs.",
          flush=True)
    print("==== SUMMARY (mean / median / std across seeds) ====", flush=True)
    for target, backends in agg.items():
        print(f"[{target}]")
        for backend, stats in backends.items():
            print(
                f"  {backend}: test_nll mean={stats['mean']:.4f} "
                f"median={stats['median']:.4f} std={stats['std']:.4f} "
                f"min={stats['min']:.4f} max={stats['max']:.4f} "
                f"n={stats['n_seeds']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
