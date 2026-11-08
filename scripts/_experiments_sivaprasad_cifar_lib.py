r"""The input-convex classifier of Sivaprasad et al. (2021) on CIFAR-10.

Sivaprasad, Singh, Manwani and Gandhi, "The Curious Case of Convex Neural
Networks" (ECML-PKDD 2021, arXiv:2006.05103), keep the inter-layer weights of
an input-output-convex classifier non-negative with a post-update rule: after
every optimizer step each negative constrained weight ``w`` becomes
``exp(w - 5)`` (their Algorithm 1). This lib trains their CIFAR-10
architecture, three hidden layers of 800 units, on our ICNN under six
positivity constructions on one pipeline:

* ``exppost``: their rule, ``eps = 5``;
* ``exppost_matched``: the same rule with the return point at the median of
  the ``hk`` constrained draw at fan-in 800 (``eps = 8.80``);
* ``direct``: direct softplus;
* ``pgd``: projected gradient descent;
* ``lift``: the lift, emitting the weights from the training batch;
* ``nonconvex``: no positivity, the reference.

Every construction uses the ``hk`` initialization, batch 64, mean cross
entropy and Adam, with early stopping on validation accuracy. Test accuracy
is read at the epoch of best validation accuracy.

The evaluation template is built in ``pos_constraint_mode="clamp"`` so that
the tensor a construction emits is used as the weight, and
:func:`check_single_positivity` measures that softplus is not applied twice.
The projection and the post-update rule are applied once at construction, so
step zero is already in the orthant. :func:`switch_preflight` builds every
cell, runs it for two steps and checks that the constructions differ, that
every one moves and that a rerun at the same seed is bit-identical.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import os
import time
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from projorg import checkpointsdir, gitdir, plotsdir
from torch import nn
from torch.func import functional_call

from lift.dataset.flat_images import (
    FlatImageSplit, get_spec, image_split, image_test,
)
from lift.models import HyperNetwork, icnn_pos_param_names
from lift.models.icnn import ICNN
from lift.objectives.shoulder import (
    THR_EMITTED, ShoulderTracker, constrained_names,
)
from lift.utils.paper_style import PALETTE

from _experiments_hypernet_vs_direct_1d_lib import (
    INIT_MODES, _LECUN_FIRST_LAYER_MODES, DirectParamModule, apply_init_mode,
)

CONSTRUCTIONS = ("pgd", "direct", "lift", "exppost", "exppost_matched",
                 "nonconvex")

EXPPOST_EPS = 5.0

#: The scale-matched control. Under ``hk`` the constrained weight is
#: log-normal with ``mu_w = sqrt(6 pi / (N D))`` and ``var_w = 1 / N``
#: (Hoedt and Klambauer 2023, Eq. 11-12 at rho* = 1/2, with
#: ``D = 6(pi - 1) + (N - 1)(3 sqrt 3 + 2 pi - 6)``), so its median is
#: ``mu_w^2 / sqrt(var_w + mu_w^2) = 1.515e-4`` at ``N = 800`` and
#: ``exp(-8.80) = 1.507e-4``.
EXPPOST_EPS_MATCHED = 8.80

#: ``nonconvex`` stores and emits the raw weight as ``pgd`` does and is never
#: projected.
_DIRECT_MODES = {
    "pgd": "pgd", "nonconvex": "pgd", "exppost": "exp_post",
    "exppost_matched": "exp_post", "direct": "softplus",
}

_PROJECTED = ("pgd", "exppost", "exppost_matched")

_LIFTS = ("lift",)


def exppost_eps(construction: str) -> Optional[float]:
    """The ``eps`` of the exponentiation rule, or ``None`` for the rest."""
    if construction == "exppost":
        return EXPPOST_EPS
    if construction == "exppost_matched":
        return EXPPOST_EPS_MATCHED
    return None


#: Every file whose contents can change a recorded number.
_MEASUREMENT_FILES = (
    "lift/objectives/shoulder.py",
    "lift/models/icnn.py",
    "lift/models/hypernet.py",
    "lift/dataset/flat_images.py",
    "scripts/_experiments_hypernet_vs_direct_1d_lib.py",
    "scripts/_experiments_sivaprasad_cifar_lib.py",
    "configs/experiments_sivaprasad_cifar.json",
)


@functools.lru_cache(maxsize=1)
def measurement_provenance() -> dict:
    """``{path: sha256}`` for every file the measurements depend on."""
    root = gitdir()
    out = {}
    for rel in _MEASUREMENT_FILES:
        try:
            with open(os.path.join(root, rel), "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()[:16]
        except OSError:
            out[rel] = "MISSING"
    out["torch"] = torch.__version__
    return out


TAUS: tuple[float, ...] = (
    0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, THR_EMITTED, 0.1,
)

CONSTRUCTION_COLOR = {
    "pgd": "#2ca02c",
    "direct": PALETTE["direct_softplus"],
    "lift": PALETTE["hypernet"],
    "nonconvex": "0.45",
    "exppost": "#17becf",
    "exppost_matched": "#0b6b70",
}

INIT_COLOR = {
    "hk": "#1f77b4", "he": "#d62728", "ours": "#ff7f0e",
    "cpflow": "#2ca02c", "torch": "#8c564b",
}

LIFT_INIT_MATCH = ("elementwise", "mean", "off")


def build_template(spec, hidden_dim: int, nlayers: int,
                   activation: str = "relu") -> ICNN:
    """The frozen network every construction is scored through.

    ``pos_constraint_mode="clamp"`` makes the template use the emitted
    tensor unchanged as the weight.
    """
    return ICNN(
        input_size=spec.width,
        hidden_dim=int(hidden_dim),
        nlayers=int(nlayers),
        activation=str(activation),
        strong_convexity=0.0,
        pos_constraint_mode="clamp",
        output_size=int(spec.num_classes),
        use_skip=False,
        output_bias=True,
    )


def _activation_name(module: nn.Module) -> str:
    """The ``ICNN`` activation keyword that rebuilds ``module``."""
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.ELU):
        return "elu"
    if isinstance(module, nn.Softplus):
        return "softplus"
    raise ValueError(f"no ICNN activation keyword for {type(module).__name__}")


@torch.no_grad()
def check_single_positivity(template: ICNN, theta: dict,
                            x: torch.Tensor) -> float:
    """The relative logit shift a second softplus would cause.

    Runs the emitted parameters through a ``softplus``-mode copy of the
    template with the same activation; the preflight requires the shift to
    be large, i.e. that the template in use does not apply softplus.
    """
    twice = ICNN(
        input_size=template.input_size, hidden_dim=template.first_layer.
        out_features, nlayers=template.nlayers,
        activation=_activation_name(template.activation),
        strong_convexity=0.0, pos_constraint_mode="softplus",
        output_size=template.output_size, use_skip=False,
        output_bias=template.output_layer.bias is not None,
    ).to(x.device)
    a = functional_call(template, theta, (x,)).double()
    b = functional_call(twice, theta, (x,)).double()
    denom = float(a.abs().mean()) + 1e-12
    return float((a - b).abs().mean()) / denom


def _readouts_by_param(hyper: HyperNetwork):
    """``(param_name, readout Linear)`` pairs, dense or shared."""
    if getattr(hyper, "share_readouts_by_shape", False):
        for name, key in zip(hyper._param_names, hyper._readout_key):
            yield name, hyper._shared_readouts[key]
    else:
        for name, layer in zip(hyper._param_names, hyper.weight_predictors):
            yield name, layer


def match_lift_init(hyper: HyperNetwork, probe: nn.Module, pos_names: set,
                    how: str) -> dict:
    """Start the lift where ``apply_init_mode`` starts a stored weight.

    ``elementwise`` fills each readout bias, in emit order, with the latent
    value the direct initializer draws for that entry, so the emitted tensor
    at step zero is the direct construction's own draw up to the
    ``Theta_E h`` perturbation. It needs dense readouts. ``mean`` fills each
    constrained readout bias with the softplus inverse of the mean emitted
    weight. ``off`` leaves the emitter at its default bias.
    """
    if how not in LIFT_INIT_MATCH:
        raise ValueError(
            f"lift_init_match must be one of {LIFT_INIT_MATCH}; got {how!r}",
        )
    if how == "off":
        return {}
    if hyper.pos_constraint_mode != "softplus":
        raise ValueError(
            "lift init matching is defined for the softplus readout only; "
            f"got {hyper.pos_constraint_mode!r}",
        )
    shared = bool(getattr(hyper, "share_readouts_by_shape", False))
    if how == "elementwise" and shared:
        raise ValueError(
            "lift_init_match='elementwise' needs one bias per emitted "
            "entry and so needs dense readouts; set hyper_share_readouts=0 "
            "or use lift_init_match='mean'",
        )
    src = dict(probe.named_parameters())
    info: dict[str, float] = {}
    with torch.no_grad():
        for name, readout in _readouts_by_param(hyper):
            p = src[name].detach().reshape(-1)
            if how == "elementwise":
                readout.bias.copy_(p.to(readout.bias.device))
                continue
            if name in pos_names:
                mean = float(F.softplus(p).mean())
                info[name] = mean
                readout.bias.fill_(
                    float(math.log(math.expm1(max(mean, 1e-8)))),
                )
            else:
                readout.bias.fill_(float(p.mean()))
    return info


def build_arm(
    construction: str,
    spec,
    hidden_dim: int,
    nlayers: int,
    init_mode: str,
    hyper_hidden_sizes: list[int],
    share_readouts: bool,
    readout_fanin_scale: bool,
    lift_init_match: str,
    activation: str,
    device: torch.device,
    lecun_first_layer: bool = True,
    init_seed: Optional[int] = None,
    max_lift_params: float = 4.0e8,
) -> tuple[nn.Module, ICNN, set[str]]:
    """Build one cell: the trainable module, its frozen template, its pos names.

    ``DirectParamModule`` carries softplus, the projection and the
    post-update rule; :class:`HyperNetwork` carries the lift.

    ``init_seed`` reseeds the global generator immediately before the stored
    (or probe) network is drawn, so the projection, softplus and the lift's
    probe draw the same numbers at one seed.

    ``lecun_first_layer`` LeCun-initializes the unconstrained layers for every
    initialization, which ``apply_init_mode`` does only for ``hk`` and ``he``.
    """
    if construction not in CONSTRUCTIONS:
        raise ValueError(
            f"unknown construction {construction!r}; expected one of "
            f"{CONSTRUCTIONS}",
        )
    if init_mode not in INIT_MODES:
        raise ValueError(
            f"unknown init_mode {init_mode!r}; expected one of {INIT_MODES}",
        )
    template = build_template(spec, hidden_dim, nlayers, activation).to(device)
    template.requires_grad_(False)
    pos_names = icnn_pos_param_names(int(nlayers))
    golden = constrained_names(
        [n for n, _ in template.named_parameters()],
    )
    if set(golden) != set(pos_names):
        raise RuntimeError(
            "the shoulder instrument and the positivity tags disagree on "
            f"which weights are constrained: {sorted(golden)} vs "
            f"{sorted(pos_names)}",
        )
    names = {n for n, _ in template.named_parameters()}
    missing_bias = sorted(
        n for n in pos_names
        if n[: -len("weight")] + "bias" not in names
    )
    if missing_bias:
        raise RuntimeError(
            "constrained layers without a bias would silently lose the "
            f"Hoedt-Klambauer shift: {missing_bias}",
        )

    def _lecun(module: nn.Module) -> None:
        if not lecun_first_layer or init_mode in _LECUN_FIRST_LAYER_MODES:
            return
        with torch.no_grad():
            for name, p in module.named_parameters():
                if name in pos_names or not name.endswith(".weight"):
                    continue
                nn.init.normal_(p, 0.0, math.sqrt(1.0 / int(p.shape[-1])))
                sibling = name[: -len("weight")] + "bias"
                if sibling in dict(module.named_parameters()):
                    module.get_parameter(sibling).zero_()

    def _storage(mode: str) -> ICNN:
        return ICNN(
            input_size=spec.width, hidden_dim=int(hidden_dim),
            nlayers=int(nlayers), activation=str(activation),
            strong_convexity=0.0, pos_constraint_mode=mode,
            output_size=int(spec.num_classes), use_skip=False,
            output_bias=True,
        )

    if construction in _DIRECT_MODES:
        raw = _DIRECT_MODES[construction] != "softplus"
        if init_seed is not None:
            torch.manual_seed(int(init_seed))
        storage = _storage("clamp" if raw else "softplus")
        arm = DirectParamModule(
            K=1, D=spec.width, hidden_dim=int(hidden_dim),
            nlayers=int(nlayers), strong_convexity=0.0,
            activation=str(activation),
            pos_constraint_mode=_DIRECT_MODES[construction],
            exp_eps=exppost_eps(construction) or EXPPOST_EPS,
            init_mode=str(init_mode),
            inner_module=storage,
            pos_names=pos_names,
        ).to(device)
        _lecun(storage)
        if construction in _PROJECTED:
            # Mapped once at construction, so step zero is in the orthant.
            arm.project_positive()
    else:
        if readout_fanin_scale and share_readouts:
            raise ValueError(
                "hyper_readout_fanin_scale is implemented for dense "
                "readouts only; set hyper_share_readouts=0",
            )
        n_target = sum(
            int(np.prod(p.shape)) for p in template.parameters()
        )
        d_h = int(hyper_hidden_sizes[-1])
        if share_readouts:
            n_readout, detail = None, ""
        else:
            n_readout = n_target * (d_h + 1)
            detail = f"{n_target:,d} emitted entries x (d_h={d_h} + 1)"
        if n_readout is not None and n_readout > int(max_lift_params):
            raise ValueError(
                f"the lift's dense readouts would hold {n_readout:,d} "
                f"parameters ({detail}), above "
                f"max_lift_params={int(max_lift_params):,d}: "
                f"{4 * n_readout / 2 ** 30:.1f} GiB of weights and about "
                f"{16 * n_readout / 2 ** 30:.1f} GiB once Adam carries "
                "two moments. Lower hyper_hidden_sizes, set "
                "hyper_share_readouts=1 with lift_init_match='mean', or "
                "lower hidden_dim.",
            )
        arm = HyperNetwork(
            input_size=spec.width,
            hidden_sizes=[int(h) for h in hyper_hidden_sizes],
            downstream_network=template,
            pos_param_names=pos_names,
            pos_constraint_mode="softplus",
            share_readouts_by_shape=bool(share_readouts),
            readout_fanin_scale=bool(readout_fanin_scale),
        ).to(device)
        if lift_init_match != "off":
            if init_seed is not None:
                torch.manual_seed(int(init_seed))
            probe = _storage("softplus").to(device)
            apply_init_mode(probe, pos_names, str(init_mode), latent=True)
            _lecun(probe)
            match_lift_init(arm, probe, pos_names, lift_init_match)
            del probe
    return arm, template, pos_names


def projects(construction: str) -> bool:
    """Whether a construction runs its post-step map after every step."""
    return construction in _PROJECTED


def emit(arm: nn.Module, x: torch.Tensor) -> dict:
    """The emitted parameter dict. ``x`` conditions the lift and only it."""
    return arm(x)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha1(
        np.ascontiguousarray(t.detach().float().cpu().numpy()).tobytes()
    ).hexdigest()[:16]


def _emitted_constrained(theta: dict) -> torch.Tensor:
    keys = constrained_names(theta.keys())
    return torch.cat([theta[k].detach().reshape(-1) for k in keys])


@torch.no_grad()
def signal_propagation(template: ICNN, theta: dict,
                       x: torch.Tensor) -> list[dict]:
    """Per layer: pre-activation mean, variance and mean feature correlation.

    The quantities the ``hk`` initialization is derived to fix. A unit with
    zero batch variance has an undefined correlation; those pairs are dropped
    and counted.
    """
    out = []
    z = x
    n_hidden = len(template.z_layers)
    for i in range(n_hidden + 2):
        if i == 0:
            pre = F.linear(z, theta["first_layer.weight"],
                           theta.get("first_layer.bias"))
        elif i <= n_hidden:
            pre = F.linear(z, theta[f"z_layers.{i - 1}.weight"],
                           theta.get(f"z_layers.{i - 1}.bias"))
        else:
            pre = F.linear(z, theta["output_layer.weight"],
                           theta.get("output_layer.bias"))
        f = pre.double()
        n_dead = int((f.var(dim=0) <= 0).sum())
        if f.shape[1] > 1:
            c = torch.corrcoef(f.T)
            m = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
            vals = c[m]
            finite = vals[~torch.isnan(vals)]
            rho = float(finite.mean()) if finite.numel() else float("nan")
            n_undef = int(vals.numel() - finite.numel())
        else:
            rho, n_undef = float("nan"), 0
        out.append({
            "layer": i,
            "pre_mean": float(f.mean()),
            "pre_mean_sq_per_unit": float((f.mean(dim=0) ** 2).mean()),
            "pre_var": float(f.var(dim=0).mean()),
            "pre_corr_offdiag": rho,
            "n_units_zero_var": n_dead,
            "n_corr_pairs_undefined": n_undef,
        })
        z = template.activation(pre)
    return out


@torch.no_grad()
def emission_probe(arm, index: dict, x_batches: list, floor: float = 1e-12,
                   ) -> dict:
    """How much the lift's pre-positivity emission moves between batches.

    Returns, on the tracked coordinate subset, the standard deviation of the
    pre-positivity emission across the conditioning batches
    (``across_batch_std``), its spread across entries at one batch
    (``entrywise_std``), and the fraction of coordinates that do not move
    with the batch (``dead_frac``). Empty for a construction that does not
    emit from a batch.
    """
    if not isinstance(arm, HyperNetwork):
        return {}
    was_training = arm.training
    arm.eval()
    try:
        with arm.frozen_pooled_ema():
            rows, slack = [], {}
            for xb in x_batches:
                with arm.capture_pre_positivity() as cap:
                    arm(xb)
                rows.append(torch.cat([
                    (cap[k].reshape(-1) if idx is None
                     else cap[k].reshape(-1)[idx]).double()
                    for k, idx in index.items()
                ]))
            for k, idx in index.items():
                b = dict(arm.positivity_readout_heads(names=[k]))
                if k in b:
                    v = b[k].detach().reshape(-1)
                    slack[k] = v if idx is None else v[idx]
    finally:
        if was_training:
            arm.train()
    M = torch.stack(rows, 0)
    sd = M.std(dim=0)
    out = {
        "across_batch_std_mean": float(sd.mean()),
        "across_batch_std_median": float(sd.median()),
        "dead_frac": float((sd <= floor).double().mean()),
        "n_probe_batches": int(M.shape[0]),
        "n_probe_coords": int(M.shape[1]),
    }
    if len(slack) == len(index):
        b = torch.cat([slack[k].double() for k in index])
        out["entrywise_std"] = float((M[0] - b).std())
    return out


def _occ_profile(w: torch.Tensor) -> dict:
    d = w.double()
    return {
        "mean": float(d.mean()), "median": float(d.median()),
        "std": float(d.std()),
        "occupancy": {f"{t:.6g}": float((d <= t).double().mean())
                      for t in TAUS},
    }


def _run_cell_steps(arm, template, x, y, n_steps: int, construction: str,
                    lr: float = 1e-3):
    opt = torch.optim.Adam(arm.parameters(), lr=float(lr))
    for _ in range(int(n_steps)):
        theta = emit(arm, x)
        loss = F.cross_entropy(
            functional_call(template, theta, (x,)), y, reduction="sum",
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if projects(construction):
            arm.project_positive()
    return opt


def switch_preflight(
    spec,
    constructions: Iterable[str],
    init_modes: Iterable[str],
    hidden_dim: int,
    nlayers: int,
    hyper_hidden_sizes: list[int],
    share_readouts: bool,
    readout_fanin_scale: bool,
    lift_init_match: str,
    activation: str,
    device: torch.device,
    seed: int = 0,
    n_steps: int = 2,
    batch: int = 64,
    probe_batch: int = 1024,
    lecun_first_layer: bool = True,
    lr: float = 1e-3,
    max_lift_params: float = 4.0e8,
    x_real: Optional[torch.Tensor] = None,
    y_real: Optional[torch.Tensor] = None,
) -> dict:
    """Check that every setting takes effect before any real compute.

    Each (construction, initialization) cell is built, run for ``n_steps``
    optimizer steps on a fixed batch, and fingerprinted at step zero and
    after the steps. Enforced: initializations differ at step zero within a
    construction; constructions differ after the steps within an
    initialization; every construction moves its emitted weight; the lift's
    emission moves between conditioning batches; the template does not apply
    softplus a second time; and a rerun at the same seed is bit-identical.
    Per cell it returns the step-zero occupancy profile, the layerwise signal
    propagation and the step-zero cross entropy.

    Raises ``RuntimeError`` naming the failing pair.
    """
    constructions, init_modes = list(constructions), list(init_modes)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7717)
    if x_real is not None:
        x = x_real[:batch].to(device)
        y = y_real[:batch].to(device)
    else:
        x = torch.randn(batch, spec.width, generator=g).to(device)
        y = torch.randint(0, spec.num_classes, (batch,),
                          generator=g).to(device)
    if x_real is not None and x_real.shape[0] >= 2 * batch:
        x_alt = x_real[batch:2 * batch].to(device)
    else:
        x_alt = torch.randn(batch, spec.width, generator=g).to(device)
    x_probe = (x_real[:probe_batch].to(device) if x_real is not None
               and x_real.shape[0] >= probe_batch else None)

    def _one(con: str, mode: str) -> dict:
        torch.manual_seed(int(seed))
        arm, template, _ = build_arm(
            con, spec, hidden_dim, nlayers, mode, hyper_hidden_sizes,
            share_readouts, readout_fanin_scale, lift_init_match,
            activation, device, lecun_first_layer, init_seed=int(seed),
            max_lift_params=max_lift_params,
        )
        with torch.no_grad():
            theta0 = emit(arm, x)
            w0 = _emitted_constrained(theta0)
            if projects(con) and float(w0.min()) < 0.0:
                raise RuntimeError(
                    f"{con}/{mode}: the projected construction emitted a "
                    f"negative weight ({float(w0.min()):.3e}) at step zero",
                )
            logits0 = functional_call(template, theta0, (x,))
            ce0 = float(F.cross_entropy(logits0, y, reduction="mean"))
            xp = x if x_probe is None else x_probe
            sigprop = signal_propagation(template, theta0, xp)
            double_shift = check_single_positivity(template, theta0, x)
            idx = _shoulder_index(theta0, 65536, int(seed))
            probe = emission_probe(arm, idx, [x, x_alt])
        _run_cell_steps(arm, template, x, y, n_steps, con, lr=lr)
        with torch.no_grad():
            wT = _emitted_constrained(emit(arm, x))
        out = {
            "sha_init": _sha(w0), "sha_after": _sha(wT),
            "step0": _occ_profile(w0),
            "ce_step0": ce0,
            "logit_var_step0": float(logits0.double().var()),
            "finite_step0": bool(np.isfinite(ce0)),
            "signal_propagation": sigprop,
            "signal_propagation_n_obs": int(
                (x if x_probe is None else x_probe).shape[0]),
            "double_softplus_rel_shift": double_shift,
            "mean_after": float(wT.double().mean()),
            "moved": float((wT - w0).abs().double().mean()),
            "n_trainable": int(sum(p.numel() for p in arm.parameters()
                                   if p.requires_grad)),
            "n_emitted": int(w0.numel()),
            "emission_probe": probe,
            "emitter_head_kinds": list(
                getattr(arm, "emitter_head_kinds", []),
            ),
        }
        del arm, template
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    prints = {(c, m): _one(c, m)
              for c, m in itertools.product(constructions, init_modes)}

    failures: list[str] = []
    for con in constructions:
        seen: dict[str, str] = {}
        for mode in init_modes:
            s = prints[(con, mode)]["sha_init"]
            if s in seen:
                failures.append(
                    f"init_mode {mode!r} and {seen[s]!r} give an identical "
                    f"step-zero emitted weight under construction {con!r}",
                )
            seen[s] = mode
    for mode in init_modes:
        seen = {}
        for con in constructions:
            v = prints[(con, mode)]
            if not (np.isfinite(v["mean_after"]) and v["finite_step0"]):
                continue
            s = v["sha_after"]
            if s in seen:
                failures.append(
                    f"construction {con!r} and {seen[s]!r} are identical "
                    f"after {n_steps} steps under init_mode {mode!r}",
                )
            seen[s] = con
            if v["moved"] == 0.0:
                failures.append(
                    f"construction {con!r} under init_mode {mode!r} did not "
                    f"move its emitted weight in {n_steps} steps",
                )
            pr = v.get("emission_probe") or {}
            if con in _LIFTS and pr:
                if not (pr.get("across_batch_std_mean", 0.0) > 0.0):
                    failures.append(
                        f"construction {con!r} under init_mode {mode!r}: "
                        "the emission does not move between conditioning "
                        "batches",
                    )
                if pr.get("dead_frac", 0.0) > 0.5:
                    failures.append(
                        f"construction {con!r} under init_mode {mode!r}: "
                        f"{100 * pr['dead_frac']:.1f}% of tracked emitted "
                        "coordinates do not move with the conditioning "
                        "batch",
                    )
            if v["double_softplus_rel_shift"] < 1e-6:
                failures.append(
                    f"construction {con!r} under init_mode {mode!r}: a "
                    "second application of softplus leaves the logits "
                    "unchanged, so the template applies the positivity map "
                    "twice",
                )

    mode0 = init_modes[0]
    for con0 in constructions:
        again = _one(con0, mode0)
        for k in ("sha_init", "sha_after"):
            if again[k] != prints[(con0, mode0)][k]:
                failures.append(
                    f"non-deterministic: {con0}/{mode0} gave a different {k} "
                    "on a rerun at the same seed",
                )
    if failures:
        raise RuntimeError(
            "switch preflight failed:\n  " + "\n  ".join(failures),
        )
    return {f"{c}|{m}": v for (c, m), v in prints.items()}


def split_independence_check(dataset: str, seed: int,
                             train_split: float) -> dict:
    """Check that the split depends on the seed and on nothing else."""
    a, _, _ = image_split(dataset, seed=seed, train_split=train_split)
    torch.manual_seed(seed + 12345)
    torch.randn(1000)                       # disturb the global stream
    b, _, _ = image_split(dataset, seed=seed, train_split=train_split)
    if not torch.equal(a.y, b.y):
        raise RuntimeError(
            f"{dataset}: the train/valid split at seed {seed} changed when "
            "the global RNG stream changed",
        )
    c, _, _ = image_split(dataset, seed=seed + 1, train_split=train_split)
    if torch.equal(a.y, c.y):
        raise RuntimeError(
            f"{dataset}: the split is identical at seeds {seed} and "
            f"{seed + 1}",
        )
    return {"paired": True, "seed_sensitive": True,
            "n_train": len(a), "label_sha": _sha(a.y[:4096].float())}


def assert_cells_differ(records: list[dict]) -> dict:
    """Two finished cells differing in one grid axis must differ in loss."""
    axes = ("dataset", "construction", "init_mode", "nlayers", "activation",
            "seed")
    by_key = {tuple(r[a] for a in axes): r for r in records}

    def _unreadable(r: dict) -> bool:
        return ("skipped_reason" in r or r.get("diverged")
                or not r.get("epoch_train_loss")
                or not all(np.isfinite(v) for v in r["epoch_train_loss"]))

    failures, checked, unresolved = [], 0, 0
    for k1, r1 in by_key.items():
        for k2, r2 in by_key.items():
            if k1 >= k2:
                continue
            diff = [a for a, v1, v2 in zip(axes, k1, k2) if v1 != v2]
            if len(diff) != 1 or diff[0] == "seed":
                continue
            if _unreadable(r1) or _unreadable(r2):
                unresolved += 1
                continue
            checked += 1
            if r1["traj_sha"] == r2["traj_sha"]:
                failures.append(
                    f"cells differing only in {diff[0]} ({k1} vs {k2}) have "
                    "identical loss trajectories",
                )
    if failures:
        raise RuntimeError(
            "differential check failed:\n  " + "\n  ".join(failures),
        )
    if unresolved:
        print(
            f"  NOTE: {unresolved} pairs left unresolved because a cell "
            "was skipped or diverged",
        )
    return {"pairs_checked": checked, "pairs_unresolved": unresolved}


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return float(total ** 0.5)


def _shoulder_index(theta: dict, max_coords: int, seed: int) -> dict:
    """A fixed random subset of the constrained coordinates, per tensor.

    Stratified across the constrained tensors in proportion to their size
    and drawn from the shapes and the seed alone, so every construction at a
    seed follows the same coordinates. ``None`` means the whole tensor.
    """
    keys = constrained_names(theta.keys())
    sizes = {k: int(theta[k].numel()) for k in keys}
    total = sum(sizes.values())
    if total <= int(max_coords):
        return {k: None for k in keys}
    rng = np.random.default_rng(int(seed) + 991)
    out = {}
    for k in keys:
        take = max(1, int(round(max_coords * sizes[k] / total)))
        idx = rng.choice(sizes[k], size=min(take, sizes[k]), replace=False)
        out[k] = torch.from_numpy(np.sort(idx)).long().to(theta[k].device)
    return out


def _sub(theta: dict, index: dict) -> dict:
    out = {}
    for k, idx in index.items():
        flat = theta[k].detach().reshape(-1)
        out[k] = flat if idx is None else flat[idx]
    return out


def _thin(series, max_points: int = 1500) -> tuple[list, int]:
    series = list(series)
    if len(series) <= max_points:
        return [float(v) for v in series], 1
    stride = int(np.ceil(len(series) / max_points))
    return [float(v) for v in series[::stride]], stride


def _downward(series) -> dict:
    """Total downward variation and downward steps per thousand steps."""
    a = np.asarray(list(series), dtype=np.float64)
    if a.size < 2:
        return {"downward_variation": 0.0, "downward_steps_per_1k": 0.0,
                "max_single_drop": 0.0}
    d = np.diff(a)
    neg = d[d < 0]
    return {
        "downward_variation": float(-neg.sum()) if neg.size else 0.0,
        "downward_steps_per_1k": float(1000.0 * neg.size / d.size),
        "max_single_drop": float(-neg.min()) if neg.size else 0.0,
    }


def _compact(summary: dict) -> dict:
    """Thin every stored series; keep every scalar exact."""
    out = dict(summary)
    for key in ("median_series", "disp_median_series",
                "disp_on_shoulder_series", "frac_zero_disp_series",
                "frac_exact_zero_series", "occupancy_series"):
        if key in out:
            out[key], _ = _thin(out[key])
    for curve in out.get("hazard_curve", {}).values():
        curve["series"], _ = _thin(curve.get("series", []))
    return out


@torch.no_grad()
def evaluate(arm, template, split: FlatImageSplit, device, chunk: int,
             cond: Optional[torch.Tensor]):
    """Cross entropy and accuracy on a split.

    ``cond`` is the lift's fixed conditioning batch, of the training batch
    size, so the score does not depend on how the split is chunked; it is
    ``None`` and ignored for the other constructions. The loss is summed in
    float64 so the chunk size cannot change it.
    """
    was_training = arm.training
    arm.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(split), chunk):
            idx = torch.arange(start, min(start + chunk, len(split)))
            xb, yb = split.batch(idx, device)
            theta = emit(arm, xb if cond is None else cond)
            logits = functional_call(template, theta, (xb,))
            total += float(F.cross_entropy(logits.double(), yb,
                                           reduction="sum"))
            correct += int((logits.argmax(dim=1) == yb).sum())
            n += int(yb.numel())
    if was_training:
        arm.train()
    return total / n, correct / n


def _probe_batches(split, device, batch: int, seed: int,
                   n: int = 8) -> list:
    """A fixed set of conditioning batches, from a dedicated generator."""
    g = torch.Generator().manual_seed(int(seed) + 5501)
    out = []
    for _ in range(int(n)):
        idx = torch.randperm(len(split), generator=g)[:int(batch)]
        out.append(split.batch(idx, device)[0])
    return out


@torch.no_grad()
def _time_emission(arm, x: torch.Tensor, n: int = 20) -> float:
    """Milliseconds per emission, on CUDA events; NaN off the accelerator."""
    if not isinstance(arm, HyperNetwork) or x.device.type != "cuda":
        return float("nan")
    was_training = arm.training
    arm.eval()
    try:
        with arm.frozen_pooled_ema():
            for _ in range(3):
                arm(x)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(int(n)):
                arm(x)
            end.record()
            torch.cuda.synchronize()
    finally:
        if was_training:
            arm.train()
    return float(start.elapsed_time(end)) / float(n)


@functools.lru_cache(maxsize=4)
def _cached_split(dataset: str, seed: int, train_split: float):
    """One decode per ``(dataset, seed)`` and process."""
    return image_split(dataset, seed=seed, train_split=train_split)


@functools.lru_cache(maxsize=2)
def _cached_test(dataset: str) -> FlatImageSplit:
    """The official test split, decoded once per process."""
    return image_test(dataset)[0]


def cell_key(dataset: str, construction: str, init_mode: str, nlayers: int,
             activation: str, seed: int) -> str:
    """The cell's name on disk: its six grid-axis values, ``|``-joined."""
    return (f"{dataset}|{construction}|{init_mode}|{int(nlayers)}|"
            f"{activation}|{int(seed)}")


def select_at_epoch(val_curve: list[dict], epoch: int, field: str) -> float:
    """``field`` of the full-split entry of ``val_curve`` at ``epoch``.

    Test accuracy at the epoch of best validation accuracy is read off the
    per-epoch record. Epoch 0 is the untrained network; the probe entries
    carry ``full_split=False`` and are never selected.
    """
    hits = [e for e in val_curve
            if e.get("full_split") and int(e["epoch"]) == int(epoch)]
    if len(hits) != 1:
        raise KeyError(
            f"expected exactly one full-split entry at epoch {epoch}, "
            f"found {len(hits)}",
        )
    v = hits[0].get(field)
    return float("nan") if v is None else float(v)


def _direct_reference(spec, hidden_dim: int, nlayers: int, init_mode: str,
                      activation: str, device: torch.device,
                      lecun_first_layer: bool, seed: int,
                      xb: torch.Tensor, yb: torch.Tensor) -> dict:
    """The direct construction's step zero at this seed and batch.

    Read for the lift's step-zero gates: the cross entropy, the constrained
    emitted profile and the first-layer pre-activation variance. The caller
    reseeds afterwards, so the lift's own construction is unaffected.
    """
    arm, template, _ = build_arm(
        "direct", spec, hidden_dim, nlayers, init_mode, [1], False, False,
        "off", activation, device, lecun_first_layer, init_seed=int(seed),
    )
    with torch.no_grad():
        theta = emit(arm, xb)
        ce = float(F.cross_entropy(
            functional_call(template, theta, (xb,)), yb, reduction="mean",
        ))
        profile = _occ_profile(_emitted_constrained(theta))
        sigprop = signal_propagation(template, theta, xb)
    del arm, template, theta
    return {"ce0": ce, "profile": profile,
            "first_layer_pre_var": float(sigprop[0]["pre_var"])}


def check_matched_draw(lift: dict, direct: dict, median_rtol: float,
                       occ_atol: float) -> dict:
    """Gate: the lift's step-zero constrained draw is the direct one's.

    Compares two ``_occ_profile`` dicts on the median (relative) and on the
    occupancy at every threshold (absolute), and raises naming the worst
    entry. A 15% scale mismatch fails both at the default tolerances.
    """
    med_d = float(direct["median"])
    med_gap = abs(float(lift["median"]) - med_d) / max(abs(med_d), 1e-30)
    gaps = {t: abs(float(lift["occupancy"][t]) - float(direct["occupancy"][t]))
            for t in direct["occupancy"]}
    worst_tau = max(gaps, key=gaps.get)
    out = {"median_rel_gap": med_gap, "occupancy_abs_gap": gaps,
           "worst_tau": worst_tau, "worst_occupancy_abs_gap": gaps[worst_tau],
           "median_rtol": float(median_rtol), "occ_atol": float(occ_atol)}
    if not med_gap <= float(median_rtol):
        raise RuntimeError(
            f"step-zero draw mismatch: the lift's constrained median "
            f"{float(lift['median']):.4e} is {100 * med_gap:.2f}% from the "
            f"direct method's {med_d:.4e} (tolerance "
            f"{100 * float(median_rtol):.1f}%)",
        )
    if not gaps[worst_tau] <= float(occ_atol):
        raise RuntimeError(
            f"step-zero draw mismatch: occupancy at tau={worst_tau} differs "
            f"by {gaps[worst_tau]:.4f} between the lift "
            f"({float(lift['occupancy'][worst_tau]):.4f}) and the direct "
            f"method ({float(direct['occupancy'][worst_tau]):.4f}); "
            f"tolerance {float(occ_atol):.4f}",
        )
    return out


def _weights_path(args, key: str, which: str) -> Optional[str]:
    """``cells/<key>_<best|final>.pt``, or ``None`` when ``save_weights`` is off."""
    if not int(getattr(args, "save_weights", 0)):
        return None
    return os.path.join(checkpointsdir(args.experiment), "cells",
                        f"{key}_{which}.pt")


def _save_weights(path: str, arm: nn.Module, epoch: int, val_acc: float,
                  x_eval: Optional[torch.Tensor]) -> None:
    """The state dict and, for the lift, the ICNN it emits at ``x_eval``."""
    state = {"epoch": int(epoch), "val_acc": float(val_acc),
             "arm": arm.state_dict()}
    if x_eval is not None:
        was_training = arm.training
        arm.eval()
        with torch.no_grad():
            state["theta_eval"] = {
                k: v.detach().clone() for k, v in emit(arm, x_eval).items()
            }
        if was_training:
            arm.train()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


_TRACKER_LISTS = ("median", "disp_median", "disp_on_shoulder",
                  "frac_zero_disp")
_THRESHOLD_SLOTS = ("tau", "series", "prev", "counts", "episodes",
                    "inside_steps")


def _tracker_state(tracker: ShoulderTracker) -> dict:
    """Every tensor and list the tracker holds, so a resume is exact."""
    thr = []
    for st in tracker._states:
        thr.append({k: getattr(st, k) for k in _THRESHOLD_SLOTS})
    return {
        "primary_tau": tracker.primary_tau,
        "n_constrained": int(tracker.n_constrained),
        "_tau_vec": tracker._tau_vec,
        "_prev_w": tracker._prev_w,
        "lists": {k: list(getattr(tracker, k)) for k in _TRACKER_LISTS},
        "thresholds": thr,
    }


def _tracker_restore(tracker: ShoulderTracker, state: dict) -> None:
    if len(state["thresholds"]) != len(tracker._states):
        raise RuntimeError(
            "resume checkpoint has a different threshold grid "
            f"({len(state['thresholds'])} vs {len(tracker._states)})",
        )
    for st, saved in zip(tracker._states, state["thresholds"]):
        if abs(float(saved["tau"]) - st.tau) > 1e-12:
            raise RuntimeError(
                f"resume checkpoint threshold {saved['tau']} != {st.tau}",
            )
        for k in _THRESHOLD_SLOTS:
            setattr(st, k, saved[k])
    tracker.primary_tau = float(state["primary_tau"])
    tracker.n_constrained = int(state["n_constrained"])
    tracker._tau_vec = state["_tau_vec"]
    tracker._prev_w = state["_prev_w"]
    for k in _TRACKER_LISTS:
        setattr(tracker, k, list(state["lists"][k]))


def _resume_path(args, key: str) -> Optional[str]:
    """``resume/<key>.pt``, or ``None`` when per-epoch checkpoints are off.

    Removed when the cell completes.
    """
    if int(getattr(args, "checkpoint_every_epochs", 0)) <= 0:
        return None
    return os.path.join(checkpointsdir(args.experiment), "resume",
                        f"{key}.pt")


def _save_resume(path: str, state: dict) -> None:
    """Atomic: a kill mid-write leaves the previous checkpoint intact."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_resume(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def _cudnn_version() -> int:
    """The cuDNN build number, or 0 when the runtime cannot be queried."""
    try:
        return int(torch.backends.cudnn.version() or 0)
    except RuntimeError:
        return 0


def train_cell(dataset: str, construction: str, init_mode: str, nlayers: int,
               activation: str, seed: int, args,
               device: torch.device) -> dict:
    """Train one cell and return its record.

    Settings read through ``getattr(args, ..., default)`` are off at their
    default: the test split (``eval_test``), early stopping on validation
    accuracy (``patience``), the lift's step-zero gates (``ce0_match_rtol``),
    the lift's test accuracy over several conditioning batches
    (``lift_cond_spread_batches``), per-epoch checkpoints and resume
    (``checkpoint_every_epochs``), saved weights (``save_weights``) and the
    debug step cap (``max_steps_per_epoch``).
    """
    t0 = time.time()
    key = cell_key(dataset, construction, init_mode, nlayers, activation,
                   seed)
    torch.manual_seed(int(seed))
    _t_decode = time.time()
    train, valid, spec = _cached_split(
        dataset, int(seed), float(args.train_split),
    )
    decode_s = time.time() - _t_decode
    if int(args.data_on_device):
        train, valid = train.to(device), valid.to(device)
    val_probe = valid.fixed_subset(int(args.n_val_probe), seed=0)

    hidden_dim = int(args.hidden_dim) or spec.width
    batch_size = int(args.batch_size)
    patience = int(getattr(args, "patience", 0))
    eval_test = bool(int(getattr(args, "eval_test", 0)))
    ce0_rtol = float(getattr(args, "ce0_match_rtol", 0.0))
    n_spread = (int(getattr(args, "lift_cond_spread_batches", 0))
                if construction in _LIFTS and eval_test else 0)
    step_cap = int(getattr(args, "max_steps_per_epoch", 0))
    resume_path = _resume_path(args, key)

    test: Optional[FlatImageSplit] = None
    if eval_test:
        test = _cached_test(dataset)
        if int(args.data_on_device):
            test = test.to(device)

    const_idx = torch.arange(min(batch_size, len(train)))
    best_path = _weights_path(args, key, "best")
    final_path = _weights_path(args, key, "final")
    if final_path is not None and os.path.exists(final_path):
        raise RuntimeError(
            f"{key}: finished weights exist at {final_path}; refusing to "
            "overwrite a finished cell. Delete them to retrain.",
        )
    # The direct construction's step zero for the lift's gates, built before
    # the lift and followed by a reseed, so the lift is the same object with
    # the gates on or off.
    ref: Optional[dict] = None
    if construction in _LIFTS and ce0_rtol > 0.0:
        xb_ref, yb_ref = train.batch(const_idx, device)
        ref = _direct_reference(
            spec, hidden_dim, int(nlayers), init_mode, str(activation),
            device, bool(int(args.lecun_first_layer)), int(seed),
            xb_ref, yb_ref,
        )
        del xb_ref, yb_ref
        torch.manual_seed(int(seed))
    ce0_ref = None if ref is None else float(ref["ce0"])
    arm, template, _ = build_arm(
        construction, spec, hidden_dim, int(nlayers), init_mode,
        [int(h) for h in args.hyper_hidden_sizes],
        bool(int(args.hyper_share_readouts)),
        bool(int(args.hyper_readout_fanin_scale)),
        str(args.lift_init_match), str(activation), device,
        bool(int(args.lecun_first_layer)), init_seed=int(seed),
        max_lift_params=float(args.max_lift_params),
    )
    opt = torch.optim.Adam(arm.parameters(), lr=float(args.lr))
    do_project = projects(construction)
    reduction = str(args.loss_reduction)

    x_fixed = train.batch(const_idx, device)[0]
    x_eval = x_fixed if construction in _LIFTS else None

    g_epoch = torch.Generator().manual_seed(int(seed) + 31337)

    with torch.no_grad():
        xb0, yb0 = train.batch(const_idx, device)
        theta0 = emit(arm, xb0)
        w0 = _emitted_constrained(theta0)
        if do_project and float(w0.min()) < 0.0:
            raise RuntimeError(
                f"{construction}: negative emitted weight at step zero",
            )
        step0 = _occ_profile(w0)
        sha_init = _sha(w0)
        sigprop = signal_propagation(template, theta0, xb0)
        draw_match: Optional[dict] = None
        first_layer_ratio = float("nan")
        if ref is not None:
            draw_match = check_matched_draw(
                step0, ref["profile"],
                float(getattr(args, "draw_match_median_rtol", 0.05)),
                float(getattr(args, "draw_match_occ_atol", 0.01)),
            )
            first_layer_ratio = (float(sigprop[0]["pre_var"])
                                 / max(ref["first_layer_pre_var"], 1e-30))
        ce0 = float(F.cross_entropy(
            functional_call(template, theta0, (xb0,)), yb0, reduction="mean",
        ))
        ce0_gap = float("nan")
        if ce0_ref is not None:
            ce0_gap = abs(ce0 - ce0_ref) / max(abs(ce0_ref), 1e-12)
            if not ce0_gap <= ce0_rtol:
                raise RuntimeError(
                    f"{key}: step-zero pairing FAILED -- the lift's CE0 "
                    f"{ce0:.4f} is {100 * ce0_gap:.1f}% from the direct "
                    f"method's {ce0_ref:.4f} at the same seed (tolerance "
                    f"{100 * ce0_rtol:.1f}%). Not training this cell.",
                )
        ce_cap = float(getattr(args, "ce_skip_factor", 1e7)) * float(
            np.log(spec.num_classes))
        if (not np.isfinite(ce0) or ce0 > ce_cap) and int(
                getattr(args, "skip_nonfinite_cells", 1)):
            return {
                "dataset": dataset, "construction": construction,
                "init_mode": init_mode, "nlayers": int(nlayers),
                "activation": str(activation), "seed": int(seed),
                "skipped_reason": (
                    f"step-zero cross entropy {ce0:.4g} exceeds "
                    f"{ce_cap:.4g} (= ce_skip_factor x log C)"),
                "code_sha": measurement_provenance(),
                "emitter_head_kinds": list(
                    getattr(arm, "emitter_head_kinds", []),
                ),
                "ce_step0": ce0, "step0_profile": step0,
                "signal_propagation": sigprop, "sha_init": sha_init,
                "traj_sha": f"skip:{sha_init}", "diverged": True,
                "epoch_train_loss": [], "wall_s": time.time() - t0,
            }
        _p = w0.double()
        _p = _p[_p > 0]
        rel_base = float(_p.median()) if _p.numel() else 0.0
        rel_tau = float(args.shoulder_rel_frac) * rel_base
    probe_idx = _shoulder_index(theta0, 65536, int(seed))
    probe_batches = _probe_batches(train, device, int(args.batch_size),
                                   int(seed))
    probe0 = emission_probe(arm, probe_idx, probe_batches)
    emit_ms = _time_emission(arm, xb0)
    taus = TAUS
    if rel_tau > 0 and f"{rel_tau:.6g}" not in {f"{t:.6g}" for t in TAUS}:
        taus = tuple(sorted(set(TAUS) | {rel_tau}))
    tracker = ShoulderTracker(taus=taus, primary_tau=THR_EMITTED)
    tracker_late: Optional[ShoulderTracker] = None
    shoulder_idx = _shoulder_index(
        theta0, int(args.shoulder_max_coords), int(seed),
    )
    n_tracked = sum(int(theta0[k].numel()) if v is None else int(v.numel())
                    for k, v in shoulder_idx.items())
    del theta0, w0

    steps_per_epoch = int(np.ceil(len(train) / batch_size))
    if step_cap > 0:
        steps_per_epoch = min(steps_per_epoch, step_cap)
    total_steps = steps_per_epoch * int(args.epochs)
    half = total_steps // 2
    ckpt_every = int(getattr(args, "checkpoint_every_epochs", 0))

    epoch_train_loss: list[float] = []
    val_curve: list[dict] = []
    step_loss: list[float] = []
    ema, step, gn, diverged = None, 0, float("nan"), False
    start_epoch, wall_before, resumed_from = 1, 0.0, 0
    early_stopped = False

    def _test_fields() -> dict:
        """Test loss and accuracy at the deployed conditioning batch, and
        the lift's test accuracy over the first ``n_spread`` fixed probe
        batches."""
        if test is None:
            return {}
        tl, ta = evaluate(arm, template, test, device, int(args.eval_chunk),
                          x_eval)
        out = {"test_loss": tl, "test_acc": ta}
        if n_spread > 0:
            accs = [
                evaluate(arm, template, test, device, int(args.eval_chunk),
                         xb)[1]
                for xb in probe_batches[:n_spread]
            ]
            a = np.asarray(accs, dtype=np.float64)
            out["test_acc_cond_spread"] = {
                "n": int(a.size), "mean": float(a.mean()),
                "std": float(a.std()), "min": float(a.min()),
                "max": float(a.max()),
            }
        return out

    resumed = None
    if resume_path is not None and os.path.exists(resume_path):
        resumed = _load_resume(resume_path, device)
    if resumed is None:
        tr0, _ = evaluate(arm, template, train, device,
                          int(args.eval_chunk), x_eval)
        v0, a0 = evaluate(arm, template, valid, device,
                          int(args.eval_chunk), x_eval)
        val_curve.append({
            "step": 0, "epoch": 0, "val_loss": v0, "val_acc": a0,
            "train_ema": float("nan"), "grad_norm": float("nan"),
            "lr": float(args.lr), "full_split": True, **_test_fields(),
        })
        best_val, best_epoch = v0, 0
        best_val_acc, best_val_acc_epoch = a0, 0
        stopped_epoch = 0
        if best_path is not None:
            _save_weights(best_path, arm, 0, a0, x_eval)
    else:
        arm.load_state_dict(resumed["arm"])
        opt.load_state_dict(resumed["opt"])
        g_epoch.set_state(resumed["g_epoch"])
        torch.set_rng_state(resumed["torch_rng"])
        if device.type == "cuda" and resumed.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(resumed["cuda_rng"], device)
        tr0 = float(resumed["train_loss_epoch0"])
        epoch_train_loss = list(resumed["epoch_train_loss"])
        val_curve = list(resumed["val_curve"])
        step_loss = list(resumed["step_loss"])
        ema, step = resumed["ema"], int(resumed["step"])
        gn, diverged = float(resumed["gn"]), bool(resumed["diverged"])
        best_val, best_epoch = resumed["best_val"], int(resumed["best_epoch"])
        best_val_acc = resumed["best_val_acc"]
        best_val_acc_epoch = int(resumed["best_val_acc_epoch"])
        _tracker_restore(tracker, resumed["tracker"])
        if resumed.get("tracker_late") is not None:
            tracker_late = ShoulderTracker(taus=taus, primary_tau=THR_EMITTED)
            _tracker_restore(tracker_late, resumed["tracker_late"])
        start_epoch = int(resumed["epoch"]) + 1
        wall_before = float(resumed["wall_s"])
        resumed_from = int(resumed["epoch"])
        stopped_epoch = resumed_from
        if resumed.get("code_sha") != measurement_provenance():
            print(
                f"  WARNING: {key} resumes across a code change; the "
                "record's code_sha is the resuming process's",
            )
        print(f"  resumed {key} from epoch {resumed_from}")
        del resumed
        # The checkpoint is written before the stopping test, so a process
        # killed between the two resumes into a finished cell.
        if patience > 0 and (start_epoch - 1) - best_val_acc_epoch >= patience:
            early_stopped = True
            start_epoch = int(args.epochs) + 1

    arm.train()
    for epoch in range(start_epoch, int(args.epochs) + 1):
        run_total, run_n = 0.0, 0
        for b_i, idx in enumerate(train.epoch_batches(batch_size, g_epoch)):
            if step_cap > 0 and b_i >= step_cap:
                break
            xb, yb = train.batch(idx, device)
            theta = emit(arm, xb)
            # The tracked iterate is the one before this update; for the
            # lift it is emitted from the current batch.
            with torch.no_grad():
                sub = _sub(theta, shoulder_idx)
                tracker.update(sub)
                if tracker_late is not None:
                    tracker_late.update(sub)
            logits = functional_call(template, theta, (xb,))
            loss = F.cross_entropy(logits, yb, reduction=reduction)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step % int(args.log_every) == 0:
                gn = _grad_norm(arm.parameters())
            opt.step()
            if do_project:
                arm.project_positive()

            scale = 1.0 if reduction == "mean" else 1.0 / int(yb.numel())
            per_sample = float(loss.detach()) * scale
            run_total += per_sample * int(yb.numel())
            run_n += int(yb.numel())
            ema = per_sample if ema is None else 0.99 * ema + 0.01 * per_sample
            step += 1
            if not np.isfinite(per_sample):
                diverged = True
            if step % int(args.log_every) == 0:
                step_loss.append(per_sample)
            if step == half:
                tracker_late = ShoulderTracker(taus=taus,
                                               primary_tau=THR_EMITTED)
            if int(args.val_every) and step % int(args.val_every) == 0:
                vl, va = evaluate(arm, template, val_probe, device,
                                  int(args.eval_chunk), x_eval)
                val_curve.append({
                    "step": step, "epoch": epoch, "val_loss": vl,
                    "val_acc": va, "train_ema": float(ema),
                    "grad_norm": float(gn), "lr": float(args.lr),
                    "full_split": False,
                })
                arm.train()

        epoch_train_loss.append(run_total / run_n)
        vl, va = evaluate(arm, template, valid, device, int(args.eval_chunk),
                          x_eval)
        tf = _test_fields()
        arm.train()
        val_curve.append({
            "step": step, "epoch": epoch, "val_loss": vl, "val_acc": va,
            "train_ema": float(ema), "grad_norm": float(gn),
            "lr": float(args.lr), "full_split": True, **tf,
        })
        if np.isfinite(vl) and vl < best_val:
            best_val, best_epoch = vl, epoch
        if np.isfinite(va) and va > best_val_acc:
            best_val_acc, best_val_acc_epoch = va, epoch
            if best_path is not None:
                _save_weights(best_path, arm, epoch, va, x_eval)
        stopped_epoch = epoch
        if resume_path is not None and epoch % ckpt_every == 0:
            _save_resume(resume_path, {
                "key": key, "epoch": epoch, "step": step,
                "arm": arm.state_dict(), "opt": opt.state_dict(),
                "g_epoch": g_epoch.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state(device)
                             if device.type == "cuda" else None),
                "train_loss_epoch0": float(tr0),
                "epoch_train_loss": list(epoch_train_loss),
                "val_curve": list(val_curve), "step_loss": list(step_loss),
                "ema": ema, "gn": gn, "diverged": diverged,
                "best_val": best_val, "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_acc_epoch": best_val_acc_epoch,
                "tracker": _tracker_state(tracker),
                "tracker_late": (_tracker_state(tracker_late)
                                 if tracker_late is not None else None),
                "wall_s": wall_before + (time.time() - t0),
                "code_sha": measurement_provenance(),
            })
        if patience > 0 and epoch - best_val_acc_epoch >= patience:
            early_stopped = True
            break

    with torch.no_grad():
        xbT, _ = train.batch(const_idx, device)
        thetaT = emit(arm, xbT)
        subT = _sub(thetaT, shoulder_idx)
        tracker.update(subT)
        if tracker_late is not None:
            tracker_late.update(subT)
        stepT = _occ_profile(_emitted_constrained(thetaT))
    if final_path is not None:
        _save_weights(final_path, arm, stopped_epoch,
                      float(val_curve[-1]["val_acc"]), x_eval)
    probeT = emission_probe(arm, probe_idx, probe_batches)

    raw_summary = tracker.summary()
    raw_occ = list(raw_summary.get("occupancy_series", []))
    occ_series, occ_stride = _thin(raw_occ)
    downward = _downward(raw_occ)
    sh = _compact(raw_summary)
    sh_late = _compact(tracker_late.summary()) if tracker_late else {}

    peak = (torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda" else float("nan"))
    peak_reserved = (torch.cuda.max_memory_reserved() / 2 ** 20
                     if device.type == "cuda" else float("nan"))
    dev_name = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else "cpu")
    record = {
        "dataset": dataset, "construction": construction,
        "init_mode": init_mode, "nlayers": int(nlayers),
        "activation": str(activation), "seed": int(seed),
        "width": int(spec.width), "hidden_dim": int(hidden_dim),
        "num_classes": int(spec.num_classes),
        "d_h": int(args.hyper_hidden_sizes[-1]),
        "share_readouts": int(args.hyper_share_readouts),
        "readout_fanin_scale": int(args.hyper_readout_fanin_scale),
        "lift_init_match": str(args.lift_init_match),
        "loss_reduction": reduction,
        "n_trainable": int(sum(p.numel() for p in arm.parameters()
                               if p.requires_grad)),
        "n_emitted_constrained": int(sum(
            int(np.prod(template.get_parameter(n).shape))
            for n in constrained_names(
                [k for k, _ in template.named_parameters()]))),
        "train_loss_epoch0": float(tr0),
        "ce_step0": float(ce0),
        "signal_propagation": sigprop,
        "shoulder_rel_tau": float(rel_tau),
        "shoulder_rel_base_median_positive": float(rel_base),
        "epoch_train_loss": [float(v) for v in epoch_train_loss],
        "step_loss": [float(v) for v in step_loss],
        "step_loss_stride": int(args.log_every),
        "steps_per_epoch": steps_per_epoch,
        "val_curve": val_curve,
        "best_val_loss": float(best_val), "best_val_epoch": int(best_epoch),
        "final_val_loss": float(val_curve[-1]["val_loss"]),
        "final_val_acc": float(val_curve[-1]["val_acc"]),
        "diverged": bool(diverged),
        # Exact, on the full constrained vector, at the first and last
        # iterate; the hazards are on the tracked subset.
        "step0_profile": step0, "stepT_profile": stepT,
        "shoulder": sh, "shoulder_late": sh_late,
        "shoulder_downward": downward,
        "frac_zero_disp_late": sh.get("frac_zero_disp_late", float("nan")),
        "occupancy_series": occ_series,
        "occupancy_series_stride": occ_stride,
        "shoulder_n_tracked": int(n_tracked),
        "sha_init": sha_init,
        "traj_sha": hashlib.sha1(
            json.dumps([round(v, 10) if np.isfinite(v) else "nan"
                        for v in epoch_train_loss]).encode()
        ).hexdigest()[:16],
        "code_sha": measurement_provenance(),
        "wall_s": wall_before + (time.time() - t0),
        "peak_gpu_mib": float(peak),
        "peak_gpu_reserved_mib": float(peak_reserved),
        "device_name": dev_name,
        "torch_cuda": str(torch.version.cuda),
        "cudnn": _cudnn_version(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "n_steps": int(step),
        "emitter_head_kinds": list(
            getattr(arm, "emitter_head_kinds", []),
        ),
        "emitter_readout_params": int(sum(
            p.numel() for p in arm.weight_predictors.parameters()
        )) if hasattr(arm, "weight_predictors") else 0,
        "emission_probe_init": probe0,
        "emission_probe_final": probeT,
        "emission_wall_ms_per_step": float(emit_ms),
        "split_decode_s": float(decode_s),
        "patience": int(patience),
        "early_stopped": bool(early_stopped),
        "stopped_epoch": int(stopped_epoch),
        "epochs_budget": int(args.epochs),
        "best_val_acc": float(best_val_acc),
        "best_val_acc_epoch": int(best_val_acc_epoch),
        "shoulder_late_started": tracker_late is not None,
        "max_steps_per_epoch": int(step_cap),
        "resumed_from_epoch": int(resumed_from),
        "exp_eps": exppost_eps(construction),
        "ce_step0_direct_ref": ce0_ref,
        "ce_step0_rel_gap": float(ce0_gap),
        "ce0_match_rtol": float(ce0_rtol),
        "draw_match": draw_match,
        "first_layer_pre_var": float(sigprop[0]["pre_var"]),
        "first_layer_pre_var_direct_ref": (
            None if ref is None else float(ref["first_layer_pre_var"])),
        "first_layer_pre_var_ratio": float(first_layer_ratio),
        "save_weights": int(best_path is not None),
        "weights_best": best_path, "weights_final": final_path,
        "eval_test": int(test is not None),
    }
    if test is not None:
        at_best = next(
            e for e in val_curve
            if e.get("full_split") and int(e["epoch"]) == best_val_acc_epoch
        )
        record.update({
            "test_acc_at_best_val_acc": select_at_epoch(
                val_curve, best_val_acc_epoch, "test_acc"),
            "test_loss_at_best_val_acc": select_at_epoch(
                val_curve, best_val_acc_epoch, "test_loss"),
            "final_test_acc": float(val_curve[-1]["test_acc"]),
            "final_test_loss": float(val_curve[-1]["test_loss"]),
            "test_acc_cond_spread_at_best_val_acc": at_best.get(
                "test_acc_cond_spread"),
            "final_test_acc_cond_spread": val_curve[-1].get(
                "test_acc_cond_spread"),
        })
    if resume_path is not None and os.path.exists(resume_path):
        os.remove(resume_path)
    del arm, template, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


class SivaprasadCIFAR:
    """The six constructions on the classification task, cell by cell."""

    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available()
            and int(args.gpu_id) >= 0 else "cpu",
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            frac = float(getattr(args, "gpu_memory_fraction", 0.0))
            if frac > 0.0:
                torch.cuda.set_per_process_memory_fraction(
                    min(frac, 1.0), self.device,
                )
                print(
                    f"GPU memory for this process capped at "
                    f"{100 * min(frac, 1.0):.0f}% of the device",
                )
        self.datasets = [str(d) for d in args.datasets]
        self.constructions = [str(c) for c in args.constructions]
        self.init_modes = [str(m) for m in args.init_modes]
        self.depths = [int(d) for d in args.depths]
        self.activations = [str(a) for a in args.activations]
        self.seeds = [int(s) for s in args.seeds]
        for c in self.constructions:
            if c not in CONSTRUCTIONS:
                raise ValueError(
                    f"unknown construction {c!r}; expected a subset of "
                    f"{CONSTRUCTIONS}",
                )
        for m in self.init_modes:
            if m not in INIT_MODES:
                raise ValueError(
                    f"unknown init_mode {m!r}; expected a subset of "
                    f"{INIT_MODES}",
                )
        if str(args.lift_init_match) not in LIFT_INIT_MATCH:
            raise ValueError(
                f"lift_init_match must be one of {LIFT_INIT_MATCH}; got "
                f"{args.lift_init_match!r}",
            )
        self.records: list[dict] = []
        self.preflight: dict = {}

    @property
    def _ckpt(self) -> str:
        return os.path.join(
            checkpointsdir(self.args.experiment), "results.json",
        )

    @property
    def _preflight_dir(self) -> str:
        """One preflight file per process, merged on load."""
        return os.path.join(
            checkpointsdir(self.args.experiment), "preflight",
        )

    @property
    def _cells_dir(self) -> str:
        """One record file per cell."""
        return os.path.join(checkpointsdir(self.args.experiment), "cells")

    @staticmethod
    def _atomic_dump(obj, path: str) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    def _read_preflight(self) -> dict:
        """Every process's preflight, merged. Later files win a key."""
        import glob
        out: dict = {}
        for path in sorted(glob.glob(os.path.join(self._preflight_dir,
                                                  "*.json"))):
            try:
                with open(path) as f:
                    out.update(json.load(f))
            except json.JSONDecodeError:
                print(f"  skipping unreadable preflight file {path}")
        return out

    def _save_preflight(self) -> None:
        os.makedirs(self._preflight_dir, exist_ok=True)
        tag = "_".join([
            ",".join(self.datasets), ",".join(self.constructions),
            ",".join(self.init_modes),
        ])
        name = hashlib.sha1(tag.encode()).hexdigest()[:16]
        self._atomic_dump(
            self.preflight,
            os.path.join(self._preflight_dir, f"{name}.json"),
        )

    def _read_cells(self) -> dict:
        import glob
        out = {}
        for path in sorted(glob.glob(os.path.join(self._cells_dir,
                                                  "*.json"))):
            try:
                with open(path) as f:
                    r = json.load(f)
            except json.JSONDecodeError:
                print(f"  skipping unreadable cell file {path}")
                continue
            out[self._key(r)] = r
        return out

    def train(self) -> None:
        os.makedirs(self._cells_dir, exist_ok=True)
        self.preflight = self._read_preflight()
        done: dict[str, dict] = {}
        if int(getattr(self.args, "resume", 0)):
            done = self._read_cells()
            if done:
                print(f"resuming: {len(done)} cells already on disk")

        for dataset in self.datasets:
            spec = get_spec(dataset)
            hidden_dim = int(self.args.hidden_dim) or spec.width
            key = f"preflight|{dataset}"
            if key in self.preflight:
                continue
            print(f"[preflight] {dataset}: checking every setting takes effect")
            self.preflight[f"split|{dataset}"] = split_independence_check(
                dataset, self.seeds[0], float(self.args.train_split),
            )
            tr, _, _ = _cached_split(
                dataset, self.seeds[0], float(self.args.train_split),
            )
            xb, yb = tr.batch(torch.arange(64), self.device)
            if len(self.constructions) * len(self.init_modes) < 2:
                print(
                    "  WARNING: this process carries one construction and "
                    "one initialization, so its preflight can only check "
                    "determinism and movement. Run the full-grid preflight "
                    "once before the sweep.",
                )
            self.preflight[key] = switch_preflight(
                spec, self.constructions, self.init_modes, hidden_dim,
                max(self.depths),
                [int(h) for h in self.args.hyper_hidden_sizes],
                bool(int(self.args.hyper_share_readouts)),
                bool(int(self.args.hyper_readout_fanin_scale)),
                str(self.args.lift_init_match), self.activations[0],
                self.device, seed=self.seeds[0],
                max_lift_params=float(self.args.max_lift_params),
                lecun_first_layer=bool(int(self.args.lecun_first_layer)),
                lr=float(self.args.lr), x_real=xb, y_real=yb,
            )
            del tr, xb, yb
            self._save_preflight()
            for cell, v in self.preflight[key].items():
                occ = v["step0"]["occupancy"]
                sp = v["signal_propagation"][-1]
                print(
                    f"    {cell:22s} params {v['n_trainable']:>11,d}  "
                    f"w0 mean {v['step0']['mean']:.3e} med "
                    f"{v['step0']['median']:.3e}  occ(1e-4) "
                    f"{occ['0.0001']:.4f} occ(tau*) "
                    f"{occ[f'{THR_EMITTED:.6g}']:.4f}  CE0 {v['ce_step0']:.4g}"
                    f"  logit var {v['logit_var_step0']:.3g}"
                    f"  rho(L{sp['layer']}) {sp['pre_corr_offdiag']:.3f}",
                )

        if str(getattr(self.args, "phase", "train")) == "preflight":
            print("preflight only; no cell trained")
            return

        cells = list(itertools.product(
            self.datasets, self.constructions, self.init_modes,
            self.depths, self.activations, self.seeds,
        ))
        print(f"{len(cells)} cells on {self.device}")
        for i, (dataset, con, mode, depth, act, seed) in enumerate(cells, 1):
            k = f"{dataset}|{con}|{mode}|{depth}|{act}|{seed}"
            if k in done:
                print(f"[{i}/{len(cells)}] {k} cached")
                continue
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rec = train_cell(dataset, con, mode, depth, act, seed,
                             self.args, self.device)
            done[k] = rec
            if "skipped_reason" in rec:
                print(
                    f"[{i}/{len(cells)}] {k}  SKIPPED: "
                    f"{rec['skipped_reason']} (CE0 {rec['ce_step0']})",
                )
                self._save_cell(rec)
                self._save_manifest(done)
                continue
            test_part = (
                f"test {100 * rec['final_test_acc']:.2f}% "
                f"(at best val acc ep{rec['best_val_acc_epoch']}: "
                f"{100 * rec['test_acc_at_best_val_acc']:.2f}%)  "
                if "final_test_acc" in rec else ""
            )
            stop_part = (
                f"stopped ep{rec['stopped_epoch']}  "
                if rec.get("early_stopped") else ""
            )
            print(
                f"[{i}/{len(cells)}] {k}  "
                f"ep0 {rec['train_loss_epoch0']:.4g}  "
                f"train {rec['epoch_train_loss'][-1]:.4f}  "
                f"val {rec['final_val_loss']:.4f}  "
                f"acc {100 * rec['final_val_acc']:.2f}%  "
                + test_part + stop_part +
                f"occ(1e-4) {rec['step0_profile']['occupancy']['0.0001']:.3f}"
                f"->{rec['stepT_profile']['occupancy']['0.0001']:.3f}  "
                f"exit {rec['shoulder'].get('exit_hazard', float('nan')):.4g}"
                f"  dead {rec['frac_zero_disp_late']:.3f}  "
                f"{rec['wall_s']:.0f}s  {rec['peak_gpu_mib']:.0f}MiB",
            )
            self._save_cell(rec)
            self._save_manifest(done)
        self.records = list(done.values())
        self._save_manifest(done)

    def _save_cell(self, rec: dict) -> None:
        os.makedirs(self._cells_dir, exist_ok=True)
        self._atomic_dump(
            rec, os.path.join(self._cells_dir, f"{self._key(rec)}.json"),
        )

    def _save_manifest(self, done: dict) -> None:
        """The preflight, the config and the cell keys. No records."""
        os.makedirs(checkpointsdir(self.args.experiment), exist_ok=True)
        self._atomic_dump(
            {"cells": sorted(done), "preflight": self.preflight,
             "config": {k: str(v) for k, v in vars(self.args).items()}},
            self._ckpt,
        )

    @staticmethod
    def _key(r: dict) -> str:
        return (f"{r['dataset']}|{r['construction']}|{r['init_mode']}|"
                f"{r['nlayers']}|{r.get('activation', 'relu')}|{r['seed']}")

    def load_checkpoint(self) -> None:
        self.preflight = self._read_preflight()
        self.records = list(self._read_cells().values())
        if not self.records:
            raise FileNotFoundError(
                f"no cell records under {self._cells_dir}",
            )
        print(f"loaded {len(self.records)} cells from {self._cells_dir}")
        print("differential check:", assert_cells_differ(self.records))

    def visualize(self) -> None:
        out = plotsdir(self.args.experiment)
        self._plot_training_curves(out)
        self._plot_occupancy(out)
        self._write_mechanism(out)

    def _select(self, **kw) -> list[dict]:
        """Skipped cells are recorded and reported, never plotted."""
        return [r for r in self.records
                if "skipped_reason" not in r
                and all(r.get(k) == v for k, v in kw.items())]

    @staticmethod
    def _band(recs: list[dict], field: str):
        arrs = [np.asarray(r[field], dtype=float) for r in recs]
        n = min(len(a) for a in arrs)
        m = np.stack([a[:n] for a in arrs])
        return (np.median(m, axis=0), np.percentile(m, 25, axis=0),
                np.percentile(m, 75, axis=0))

    def _plot_training_curves(self, out: str) -> None:
        """Training loss per epoch, one panel per construction."""
        for depth, act in itertools.product(self.depths, self.activations):
            fig, axes = plt.subplots(
                len(self.constructions), len(self.datasets),
                figsize=(3.9 * len(self.datasets),
                         2.9 * len(self.constructions)), squeeze=False,
            )
            for i, con in enumerate(self.constructions):
                for j, ds in enumerate(self.datasets):
                    ax = axes[i][j]
                    for mode in self.init_modes:
                        recs = self._select(dataset=ds, construction=con,
                                            init_mode=mode, nlayers=depth,
                                            activation=act)
                        if not recs:
                            continue
                        col = INIT_COLOR.get(mode, "0.3")
                        med, lo, hi = self._band(recs, "epoch_train_loss")
                        ep = np.arange(1, len(med) + 1)
                        ax.plot(ep, med, label=mode, lw=1.4, color=col)
                        ax.fill_between(ep, lo, hi, alpha=0.18, lw=0,
                                        color=col)
                        for r in recs:
                            ax.plot(np.arange(1, len(r["epoch_train_loss"])
                                              + 1), r["epoch_train_loss"],
                                    lw=0.4, alpha=0.35, color=col)
                    ax.set_yscale("log")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    if i == 0:
                        ax.set_title(ds)
                    if j == 0:
                        ax.set_ylabel(f"{con}\ntrain loss")
                    if i == len(self.constructions) - 1:
                        ax.set_xlabel("epoch")
            if axes[0][0].get_legend_handles_labels()[0]:
                axes[0][0].legend(frameon=False, fontsize=7)
            path = os.path.join(out, f"train_loss_L{depth}_{act}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved to {path}")

    def _plot_occupancy(self, out: str) -> None:
        """Occupancy trajectories at the shoulder threshold, by epoch."""
        for depth, act in itertools.product(self.depths, self.activations):
            fig, axes = plt.subplots(
                len(self.init_modes), len(self.datasets),
                figsize=(3.9 * len(self.datasets),
                         2.5 * len(self.init_modes)), squeeze=False,
            )
            for i, mode in enumerate(self.init_modes):
                for j, ds in enumerate(self.datasets):
                    ax = axes[i][j]
                    for con in self.constructions:
                        recs = self._select(dataset=ds, construction=con,
                                            init_mode=mode, nlayers=depth,
                                            activation=act)
                        if not recs:
                            continue
                        med, lo, hi = self._band(recs, "occupancy_series")
                        spe = recs[0]["steps_per_epoch"]
                        t = (np.arange(len(med))
                             * recs[0]["occupancy_series_stride"] / spe)
                        col = CONSTRUCTION_COLOR[con]
                        ax.plot(t, med, color=col, label=con, lw=1.3)
                        ax.fill_between(t, lo, hi, color=col, alpha=0.15,
                                        lw=0)
                    ax.set_ylim(-0.02, 1.02)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    if i == 0:
                        ax.set_title(ds)
                    if j == 0:
                        ax.set_ylabel(f"{mode}\noccupancy")
                    if i == len(self.init_modes) - 1:
                        ax.set_xlabel("epoch")
            if axes[0][0].get_legend_handles_labels()[0]:
                axes[0][0].legend(frameon=False, fontsize=7)
            path = os.path.join(out, f"occupancy_L{depth}_{act}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved to {path}")

    def _write_mechanism(self, out: str) -> None:
        """Every dwell number per cell, with its threshold."""
        rows = []
        for r in self.records:
            if "skipped_reason" in r:
                rows.append({k: r[k] for k in
                             ("dataset", "construction", "init_mode",
                              "nlayers", "activation", "seed",
                              "skipped_reason", "ce_step0")})
                continue
            sh, late = r["shoulder"], r.get("shoulder_late", {})
            rows.append({
                k: r[k] for k in
                ("dataset", "construction", "init_mode", "nlayers",
                 "activation", "seed", "width", "d_h", "n_trainable")
            } | {
                "n_tracked": r["shoulder_n_tracked"],
                # Occupancies are exact on the full constrained vector at
                # tau = 1e-4; hazards and dwell are on the tracked subset at
                # the primary tau.
                "primary_tau": r["shoulder"].get("primary_tau"),
                "rel_tau": r.get("shoulder_rel_tau"),
                "occ_init_1e-4": r["step0_profile"]["occupancy"]["0.0001"],
                "occ_final_1e-4": r["stepT_profile"]["occupancy"]["0.0001"],
                "exit_hazard_at_primary_tau": sh.get("exit_hazard"),
                "entry_hazard_at_primary_tau": sh.get("entry_hazard"),
                "hazard_dwell_steps": sh.get("hazard_dwell_steps"),
                "mean_observed_episode_steps": sh.get(
                    "mean_observed_episode_steps"),
                "censored_frac": sh.get("censored_frac"),
                "exit_hazard_late": late.get("exit_hazard"),
                "hazard_dwell_steps_late": late.get("hazard_dwell_steps"),
                "frac_coords_ever_inside": sh.get("frac_coords_ever_inside"),
                "frac_zero_disp_late": r["frac_zero_disp_late"],
                "renewal_residual": sh.get("renewal_residual"),
                "downward": r["shoulder_downward"],
                "diverged": r["diverged"],
            })
        path = os.path.join(out, "mechanism_table.json")
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved to {path}")
