r"""CP-Flow on tabular UCI: the shared library behind the HEPMASS runs.

Wraps Huang et al. ICLR 2021 Convex Potential Flows training machinery
(vendored at ``lift.baselines.cpflow_vendor``) and adds the lift, in which the
ICNN's parameters are emitted by a hypernet rather than stored as
``nn.Parameter``. Every backend shares one loss (negative log-likelihood under
the convex potential flow), training loop, log-det estimator, evaluation
protocol and data path, so the only knob is the parameter-emission backend.

Backends:

* ``direct`` -- vanilla ICNN2 weights as ``nn.Parameter``, identical to
  ``train_tabular.py`` in CP-Flow.
* ``hypernet`` -- ICNN2's parameters come from
  :class:`lift.models.HyperNetwork`. ``PosLinear``'s internal softplus reparam
  stays in place, so the hypernet bias readout is composed with the target's
  softplus.

CP-Flow knobs, with defaults matching ``train_tabular.py``:

* ``nblocks=1`` (one convex potential block), ``dimh=64``, ``nhidden=10``,
* batch 1024, Adam ``lr=1e-3``, weight decay ``1e-6``, betas ``(0.9, 0.99)``,
* stochastic CG log-det estimator during training (``m1=10`` CG iterations),
* bruteforce evaluation log-det (full Hessian and slogdet) on test and
  validation, so the reported NLL is comparable to the literature.

A 500-iteration smoke pass is exposed; full HEPMASS training uses about 300k
iterations.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from lift.baselines.cpflow_vendor.icnn import PosLinear
from lift.baselines.cpflow_vendor import (
    ActNorm,
    DeepConvexFlow,
    ICNN2,
    SequentialFlow,
)
from lift.dataset.uci import HEPMASS
from lift.models import HyperNetwork
from lift.objectives.diagnostics import (
    CrossCovAccumulator,
    CrossCovDegenerateInputs,
)
# The measurement stack itself lives in the package; the underscored aliases
# keep the names this module exports, so callers and tests that reach for
# ``lib._cross_cov_taps`` keep resolving.
from lift.objectives.cross_cov_taps import (  # noqa: F401
    BiasChannelTap,
    cross_cov_taps as _cross_cov_taps,
    flat_bias_dim as _flat_bias_dim,
    hypernets_of as _hypernets,
    make_probe_rng,
    record_cross_cov_snapshot,
    tap_iterate_source as _tap_iterate_source,
)


# ---------------------------------------------------------------------------
# Hypernet glue: emit ICNN2 weights through functional_call.
# ---------------------------------------------------------------------------
class GainCappedPosLinear(PosLinear):
    """``PosLinear`` with the per-layer gain cap the emitted ICNN uses.

    A layer's gain is ``mean(softplus(W))``, since ``PosLinear`` divides by
    fan-in. Direct softplus sits well below any cap in practice, so the cap is
    expected to be a no-op there, but it must be applied rather than assumed, or
    the comparison silently gives one method a constraint the other lacks.
    ``gain_binds`` records whether it ever bound.
    """

    gain_cap: float = 0.0
    gain_binds: int = 0

    def forward(self, x):
        w = F.softplus(self.weight)
        if self.gain_cap > 0:
            m = w.mean()
            if float(m) > self.gain_cap:
                self.gain_binds += 1
            w = w / torch.clamp(m / self.gain_cap, min=1.0)
        return F.linear(x, w, self.bias) * (1 / x.size(1))


def apply_gain_cap_to_direct(flow, gain_cap: float) -> int:
    """Swap every PosLinear in ``flow`` for the capped variant."""
    if gain_cap <= 0:
        return 0
    n = 0
    for mod in flow.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, PosLinear) and not isinstance(
                    child, GainCappedPosLinear):
                child.__class__ = GainCappedPosLinear
                child.gain_cap = float(gain_cap)
                child.gain_binds = 0
                n += 1
    return n


class HypernetParametrisedICNN(nn.Module):
    r"""Wrap an ICNN2 template and a HyperNetwork for an unconditional flow.

    An unconditional CP-Flow needs one fixed parameter set for all samples, so
    that the convex potential is a deterministic function of ``x``. The hypernet
    is therefore driven by a learnable code of shape ``(1, code_dim)``,
    registered as a parameter, and the resulting weight dict is computed once
    per forward and applied to every sample through ``functional_call``.

    The code is mean-zero Gaussian at initialization, so the effective weights
    are determined by the hypernet's MLP. The wrapper exposes the standard ICNN
    signature ``x -> (B, 1)``.
    """

    def __init__(
        self,
        D: int,
        dimh: int,
        nhidden: int,
        hyper_hidden: List[int],
        code_dim: int = 16,
        sigma_bh: float = 0.5,
        softplus_type: str = "softplus",
        zero_softplus: bool = False,
        seed: int = 0,
        condition_on: str = "code",
        gain_cap: float = 0.0,
        pool_norm: bool = False,
        pool_scale: float = 0.0,
        batch_scale: float = 1.0,
        emission_decay_frac: float = 0.0,
        emission_decay_end: float = 1.0,
        emission_decay_floor: float = 0.0,
        emission_decay_ema: float = 0.99,
        readout_fanin_scale: bool = False,
    ) -> None:
        super().__init__()
        if condition_on not in ("code", "batch", "code+batch"):
            raise ValueError(f"condition_on must be 'code', 'batch' or "
                             f"'code+batch', got {condition_on!r}")
        # ``batch``: the emission is a DeepSets summary of the input batch
        # itself, so the emitted weights fluctuate with the batch and
        # sigma_Jac^2 is live. ``code``: the deterministic learnable-code
        # variant, where sigma_Jac is structurally zero.
        self.condition_on = condition_on
        self.gain_cap = float(gain_cap)
        self.icnn_template = ICNN2(
            dim=int(D),
            dimh=int(dimh),
            num_hidden_layers=int(nhidden),
            softplus_type=softplus_type,
            zero_softplus=zero_softplus,
        )
        for p in self.icnn_template.parameters():
            p.requires_grad_(False)
        for m in self.icnn_template.modules():
            if hasattr(m, "initialized"):
                m.initialized = True

        self.code_dim = int(code_dim)
        # Learnable input code feeding the hypernet: the first-layer Jacobian
        # ``Theta_E`` acts on this code, and the readout biases are ``b_h``.
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + 24681)
        code = torch.randn(1, self.code_dim, generator=g) * 0.01
        self.code = nn.Parameter(code)

        # Names of the emitted tensors that carry the positivity constraint,
        # the ones whose mean(softplus(.)) is the layer gain.
        self._pos_emitted = set()
        self.hypernet = HyperNetwork(
            pool_norm=bool(pool_norm),
            pool_scale=float(pool_scale),
            input_size=(int(D) if condition_on == "batch"
                        else self.code_dim),  # code+batch keeps code width
            hidden_sizes=list(hyper_hidden),
            downstream_network=self.icnn_template,
            use_outer_net=False,
            pos_param_names=None,
            pos_constraint_mode="softplus",
            emission_decay_frac=float(emission_decay_frac),
            emission_decay_end=float(emission_decay_end),
            emission_decay_floor=float(emission_decay_floor),
            emission_decay_ema=float(emission_decay_ema),
            readout_fanin_scale=bool(readout_fanin_scale),
        )
        # The positivity lives in ICNN2's PosLinear.forward, as
        # softplus(weight)/fan_in, and not in the hypernet's readout, so the
        # tensors whose gain must be capped are exactly the template's
        # PosLinear weights.
        self._pos_emitted = {
            "%s.weight" % name
            for name, mod in self.icnn_template.named_modules()
            if isinstance(mod, PosLinear)
        }
        # ``code+batch`` keeps the learned code as the slack term and adds a
        # permutation-invariant batch summary, giving
        # theta-tilde = b + h_phi(X) with both terms present. The ``code``
        # variant has only the constant term, and ``batch`` only the data term,
        # which is unstable at this depth.
        self.batch_scale = float(batch_scale)
        if condition_on == "code+batch":
            self.batch_proj = nn.Linear(int(D), self.code_dim)
            # Small random init rather than zero: the summary is normalized
            # in ``_cond_input``, so a zero projection would have no direction
            # to normalize. Its magnitude is set by batch_scale alone.
            nn.init.normal_(self.batch_proj.weight, std=0.01)
            nn.init.zeros_(self.batch_proj.bias)
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(int(seed) + 13579)
        with torch.no_grad():
            for layer in self.hypernet.weight_predictors:
                layer.bias.normal_(mean=0.0, std=float(sigma_bh), generator=g2)

    def emitted_constrained_weights(self, x_cond=None):
        """The emitted latent constrained weights, that is what PosLinear sees.

        The positivity map lives in ``PosLinear.forward``, as
        ``softplus(w)/fan_in``, so these tensors are in the same space as direct
        softplus's raw ``PosLinear.weight`` and the shoulder statistic means the
        same thing for both. Read by
        ``lift.baselines.cpflow_toy.constrained_latent_weights``, which would
        otherwise report the frozen template's weights, and those never train.
        """
        with torch.no_grad():
            params = self.hypernet(self._cond_input(x_cond))
            if self.gain_cap > 0:
                params = self._cap_gain(params)
        return {k: v.detach() for k, v in params.items()
                if k in self._pos_emitted}

    def _cond_input(self, x: "torch.Tensor | None" = None) -> torch.Tensor:
        """The hypernet's input under this body's conditioning mode."""
        if self.condition_on == "batch":
            if x is None:
                raise ValueError(
                    "condition_on='batch' needs a conditioning batch")
            return x.detach()
        if self.condition_on == "code+batch":
            if x is None:
                raise ValueError(
                    "condition_on='code+batch' needs a conditioning batch")
            # Mean-pooling keeps the summary permutation-invariant, as the
            # DeepSets emission requires. The summary is then normalized and
            # scaled, so the direction of the batch dependence is free to learn
            # while its magnitude is pinned at batch_scale; without this the
            # projection's weights grow and the batch contribution is
            # unbounded.
            s = self.batch_proj(x.detach()).mean(dim=0, keepdim=True)
            if self.batch_scale > 0:
                s = self.batch_scale * s / (s.norm() + 1e-8)
            else:
                s = s * 0.0
            return self.code + s
        return self.code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The conditioning input is built by ``_cond_input``, so the diagnostic
        # accessor above and the training forward can never read different
        # emissions. In batch mode that input is the batch as data, detached, so
        # the potential's x-derivatives (grad f and the log-det Hessian) treat
        # the emitted weights as constants.
        cond = self._cond_input(x)
        params = self.hypernet(cond)
        if self.gain_cap > 0:
            params = self._cap_gain(params)
        return functional_call(self.icnn_template, params, (x,))

    def _cap_gain(self, params):
        """Keep each emitted layer non-expansive.

        ``PosLinear`` computes ``linear(x, softplus(W)) / fan_in``, so a layer's
        gain is ``mean(softplus(W))`` and a block chaining ``nhidden`` of them
        amplifies as ``mean(softplus(W)) ** nhidden``. The emitted ICNN sits
        near a gain of one, where any excursion above it is amplified
        exponentially by the chain. Rescaling ``softplus(W)`` by
        ``max(1, mean/cap)`` bounds the gain without touching the emission
        where it is already contractive, and is differentiable, so the
        optimizer feels the constraint rather than having it imposed behind its
        back.
        """
        out = {}
        for k, v in params.items():
            if v.dim() >= 2 and k in self._pos_emitted:
                sp = F.softplus(v)
                m = sp.mean()
                scale = torch.clamp(m / self.gain_cap, min=1.0)
                out[k] = torch.log(torch.expm1(sp / scale).clamp_min(1e-12))
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Build a CPFlow (direct or hypernet).
# ---------------------------------------------------------------------------
@dataclass
class CPFlowConfig:
    D: int = 21
    dimh: int = 64
    nhidden: int = 10
    nblocks: int = 1
    softplus_type: str = "softplus"
    zero_softplus: bool = False
    m1: int = 10            # CG iters during training log-det estimator.
    atol: float = 1e-3
    rtol: float = 0.0
    trainable_w0: bool = True


def build_cpflow_direct(cfg: CPFlowConfig, sigma_bh: float = 0.0,
                        seed: int = 0,
                        gain_cap: float = 0.0) -> SequentialFlow:
    """Build the direct (vanilla CP-Flow) tabular model.

    ``sigma_bh > 0`` applies the symmetric initialization of the 2-D
    sweep: the same per-element N(0, sigma_bh^2) pre-softplus jitter
    the hypernet's readout biases carry, added to every PosLinear raw
    weight, so initialization is no longer a confound.
    """
    icnns = [
        ICNN2(
            dim=cfg.D,
            dimh=cfg.dimh,
            num_hidden_layers=cfg.nhidden,
            softplus_type=cfg.softplus_type,
            zero_softplus=cfg.zero_softplus,
        )
        for _ in range(cfg.nblocks)
    ]
    if float(sigma_bh) > 0:
        from lift.baselines.cpflow_vendor.icnn import PosLinear
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + 97531)
        with torch.no_grad():
            for icnn in icnns:
                for m in icnn.modules():
                    if isinstance(m, PosLinear):
                        m.weight.add_(torch.randn(
                            m.weight.shape, generator=g,
                        ) * float(sigma_bh))
    layers: List[nn.Module] = [None] * (2 * cfg.nblocks + 1)
    layers[0::2] = [ActNorm(cfg.D) for _ in range(cfg.nblocks + 1)]
    layers[1::2] = [
        DeepConvexFlow(
            icnn,
            cfg.D,
            unbiased=False,
            m1=cfg.m1,
            m2=cfg.D,
            atol=cfg.atol,
            rtol=cfg.rtol,
            trainable_w0=cfg.trainable_w0,
        )
        for icnn in icnns
    ]
    flow = SequentialFlow(layers)
    n_capped = apply_gain_cap_to_direct(flow, gain_cap)
    if n_capped:
        print(f"[direct] gain cap {gain_cap} applied to "
              f"{n_capped} PosLinear layers (expected to be a "
              f"no-op; binds are counted)", flush=True)
    return flow


def build_cpflow_pgd(cfg: CPFlowConfig, sigma_bh: float = 0.0, seed: int = 0,
                     gain_cap: float = 0.0) -> SequentialFlow:
    """The PGD baseline on the tabular CP-Flow.

    Builds the direct model, then replaces every ``PosLinear`` of every
    ``ICNN2`` by :class:`lift.baselines.cpflow_toy.PGDPosLinear`: the raw
    constrained weight starts at ``softplus(W_direct)``, the same effective
    weight direct softplus has from the same seed, the forward uses it with
    ``PosLinear``'s ``1/fan_in`` gain and no map, and :func:`train_cpflow`
    clamps it to ``>= 0`` after every optimizer step.
    """
    from lift.baselines.cpflow_vendor.icnn import ICNN2 as _ICNN2, PosLinear as _PL
    from lift.baselines.cpflow_toy import PGDPosLinear
    flow = build_cpflow_direct(cfg, sigma_bh=sigma_bh, seed=seed, gain_cap=gain_cap)
    for icnn in [m for m in flow.modules() if isinstance(m, _ICNN2)]:
        for i, m in enumerate(icnn.Wzs):
            if isinstance(m, _PL) and not isinstance(m, PGDPosLinear):
                p = PGDPosLinear(m.in_features, m.out_features, bias=m.bias is not None)
                p = p.to(m.weight.device, m.weight.dtype)
                with torch.no_grad():
                    p.weight.copy_(F.softplus(m.weight))
                    if m.bias is not None:
                        p.bias.copy_(m.bias)
                icnn.Wzs[i] = p
    return flow


def build_cpflow_hypernet(
    cfg: CPFlowConfig,
    hyper_hidden: List[int],
    sigma_bh: float = 0.5,
    seed: int = 0,
    condition_on: str = "code",
    pool_scale: float = 0.0,
    gain_cap: float = 0.0,
    pool_norm: bool = False,
    batch_scale: float = 1.0,
    emission_decay_frac: float = 0.0,
    emission_decay_end: float = 1.0,
    emission_decay_floor: float = 0.0,
    emission_decay_ema: float = 0.99,
    readout_fanin_scale: bool = False,
) -> SequentialFlow:
    """Build the hypernet-lifted CP-Flow.

    Each ICNN2 inside a DeepConvexFlow is wrapped by a
    :class:`HypernetParametrisedICNN`; the DeepConvexFlow's
    ``self.icnn`` is then the hypernet wrapper, so the entire
    log-det path (CG / brute-force) sees the hypernet output as the
    ICNN's effective weights.
    """
    hns = [
        HypernetParametrisedICNN(
            D=cfg.D,
            dimh=cfg.dimh,
            nhidden=cfg.nhidden,
            hyper_hidden=list(hyper_hidden),
            sigma_bh=float(sigma_bh),
            softplus_type=cfg.softplus_type,
            zero_softplus=cfg.zero_softplus,
            seed=int(seed) + k,
            condition_on=condition_on,
            pool_scale=float(pool_scale),
            gain_cap=float(gain_cap),
            pool_norm=bool(pool_norm),
            batch_scale=float(batch_scale),
            emission_decay_frac=float(emission_decay_frac),
            emission_decay_end=float(emission_decay_end),
            emission_decay_floor=float(emission_decay_floor),
            emission_decay_ema=float(emission_decay_ema),
            readout_fanin_scale=bool(readout_fanin_scale),
        )
        for k in range(cfg.nblocks)
    ]
    layers: List[nn.Module] = [None] * (2 * cfg.nblocks + 1)
    layers[0::2] = [ActNorm(cfg.D) for _ in range(cfg.nblocks + 1)]
    layers[1::2] = [
        DeepConvexFlow(
            hn,
            cfg.D,
            unbiased=False,
            m1=cfg.m1,
            m2=cfg.D,
            atol=cfg.atol,
            rtol=cfg.rtol,
            trainable_w0=cfg.trainable_w0,
        )
        for hn in hns
    ]
    return SequentialFlow(layers)


# ---------------------------------------------------------------------------
# Data dependent init pass (must run before the first parameter update).
# ---------------------------------------------------------------------------
def actnorm_init(flow: SequentialFlow, x_init: torch.Tensor) -> None:
    """Run a single forward over ``x_init`` to data-init every ActNorm."""
    with torch.no_grad():
        for f in flow.flows:
            if isinstance(f, ActNorm):
                if not f.initialized:
                    f.forward_transform(x_init.unsqueeze(-1) if x_init.ndim == 2 else x_init)
                    break


def init_all_actnorms(flow: SequentialFlow, x_init: torch.Tensor) -> None:
    """Walk forward through the flow, initializing every ActNorm in order.

    Each ActNorm is initialized on the cascaded input it actually sees.
    ``DeepConvexFlow``'s forward is ``grad_x F(x)``, evaluated under a local
    ``enable_grad`` context so the next ActNorm's data-dependent initialization
    sees the post-flow input.
    """
    flow.eval()
    x = x_init.clone()
    logdet = 0
    for f in flow.flows:
        if isinstance(f, ActNorm):
            with torch.no_grad():
                x, logdet = f.forward_transform(x, logdet)
        elif isinstance(f, DeepConvexFlow):
            with torch.enable_grad():
                x_req = x.clone().detach().requires_grad_(True)
                Fval = f.get_potential(x_req)
                grad_x = torch.autograd.grad(Fval.sum(), x_req)[0]
                x = grad_x.detach()
    flow.train()


# ---------------------------------------------------------------------------
# Training / evaluation loop.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    n_iters: int = 300_000
    batch_size: int = 1024
    lr: float = 1e-3
    wd: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.99
    clip_grad: float = 0.0       # 0 -> no clipping (CP-Flow default).
    exact_logdet_train: bool = False  # diagnostic: exact Hessian log-det
    max_skips: int = 2000
    eval_every: int = 2000
    save_bestval: bool = False   # keep one checkpoint at the best validation NLL
    log_every: int = 200
    snap_every: int = 0          # 0 -> off; otherwise save state_dict.
    val_batch_size: int = 1024
    seed: int = 0
    # ---- Cross-covariance filmstrip hook -----------------------------------
    # Under ``with_cross_cov`` the trainer attaches a
    # :class:`CrossCovAccumulator` keyed off the snapshot iteration list and
    # persists one NPZ per snapshot alongside the snap_*.pt files. At each
    # snapshot it draws ``cross_cov_n_probes`` fresh batches, the sample axis
    # being probe batches at a fixed iterate rather than training steps, and
    # records per probe the emitted pre-positivity iterate and the readout-bias
    # gradient, from which the accumulator forms E[d(theta-tilde) d(g)^T].
    # Every backend runs that same code, including the unlifted ones, whose
    # iterate does not move across probes and whose matrix is therefore zero by
    # measurement.
    # ``cross_cov_jl_dim`` projects the flattened bias channel down to a
    # manageable d x d heatmap; -1 keeps the full d_bias. With
    # ``with_cross_cov`` off the accumulator is never constructed.
    with_cross_cov: bool = False
    cross_cov_n_probes: int = 64
    cross_cov_jl_dim: int = 21
    # ``raise`` aborts on an aliased or symmetric estimate; ``warn`` records it
    # with a RuntimeWarning. The guard is the only thing standing between a
    # silent auto-covariance and the archive, so relaxing it has to be a
    # deliberate, recorded act rather than an edit to the accumulator.
    cross_cov_on_degenerate: str = "raise"
    # Permutation null on the probe pairing, per snapshot. Each shuffle is
    # O(B^2) off the Gram matrices the estimator already forms; 0 turns it off.
    cross_cov_n_permutations: int = 200
    # Run the probe forwards with the exact (bruteforce) log-det instead of the
    # stochastic CG estimator. The estimator's noise is independent of the probe
    # batch, so it does not bias Sigma_slack, but it does inflate the variance;
    # whichever ran is recorded in the archive's ``probe_protocol``.
    cross_cov_exact_logdet: bool = False


def _batch_iter(x: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    n = x.shape[0]
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for k in range(0, n, batch_size):
        yield torch.from_numpy(x[idx[k:k + batch_size]]).float()


class _FixedEmission(nn.Module):
    """Stands in for a body's ``hypernet`` while its emission is pinned.

    ``HypernetParametrisedICNN.forward`` calls ``self.hypernet(cond)`` and
    hands the result to the template; returning a fixed dict there pins the
    block's weights without touching ``_cond_input`` or the gain cap.
    """

    def __init__(self, params: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._params = {k: v.detach() for k, v in params.items()}

    def forward(self, cond: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: ARG002
        return dict(self._params)


@contextlib.contextmanager
def pinned_emission(
    flow: SequentialFlow,
    x_cond: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
    n_passes: int = 2,
):
    """Pin every lifted block's emission on ``x_cond`` for the duration.

    A CP-Flow block's body conditions on the block's INPUT, i.e. the data
    transformed by the preceding blocks, so the conditioning rows are pushed
    through the flow and each body's pooled summary is accumulated in its
    forward pre-hook: the mean per-sample encoding (``inner_net``) for a
    batch-conditioned body, the mean ``batch_proj`` for a code+batch body.
    The summary is then mapped through the body's own
    ``HyperNetwork.emit_from_pooled`` (or ``code + s`` and the hypernet, for
    code+batch), and the body's ``hypernet`` is replaced by a
    :class:`_FixedEmission` holding that one weight set. Memory is
    ``O(batch_size)`` whatever the size of ``x_cond``, so the entire training
    split is the intended argument.

    With ``n_passes=2`` the recording pass is repeated with the pins from the
    first pass installed, so block k's conditioning rows are those produced by
    the pinned blocks before it, which is the fixed point to one iteration.
    Feeding the raw rows to every block instead is a different model. With the
    emission pinned the scored object is one density, chosen before the test
    rows are seen, so the number is a held-out likelihood comparable to direct
    softplus's.
    """
    bodies = [m for m in flow.modules() if hasattr(m, "_cond_input")]
    if any(isinstance(b.hypernet, _FixedEmission) for b in bodies):
        raise RuntimeError("pinned_emission: the flow is already pinned (nested use is not supported)")
    orig = {id(b): b.hypernet for b in bodies}
    convex = [f for f in flow.flows if isinstance(f, DeepConvexFlow)]
    was = [f.no_bruteforce for f in convex]

    def pooled_params() -> Dict[int, Optional[Dict[str, torch.Tensor]]]:
        acc: Dict[int, Optional[torch.Tensor]] = {id(b): None for b in bodies}
        n: Dict[int, int] = {id(b): 0 for b in bodies}

        def hook(module, inp, _b):
            x = inp[0].detach()
            hn = orig[id(_b)]
            with torch.no_grad():
                if _b.condition_on == "batch":
                    z = hn.inner_net(x.reshape(-1, hn.input_size))
                elif _b.condition_on == "code+batch":
                    z = _b.batch_proj(x)
                else:
                    return   # code-conditioned: reads no batch
                s = z.sum(dim=0, keepdim=True)
                acc[id(_b)] = s if acc[id(_b)] is None else acc[id(_b)] + s
                n[id(_b)] += int(z.shape[0])

        hooks = [b.register_forward_pre_hook(lambda m, inp, _b=b: hook(m, inp, _b)) for b in bodies]
        try:
            with torch.no_grad():
                for batch in _batch_iter(x_cond, batch_size, shuffle=False, rng=np.random.default_rng(0)):
                    flow.logp(batch.to(device))
        finally:
            for h in hooks:
                h.remove()
        out: Dict[int, Optional[Dict[str, torch.Tensor]]] = {}
        with torch.no_grad():
            for b in bodies:
                hn = orig[id(b)]
                if acc[id(b)] is None:
                    out[id(b)] = None
                    continue
                s = acc[id(b)] / n[id(b)]
                if b.condition_on == "batch":
                    out[id(b)] = hn.emit_from_pooled(s)
                else:   # code+batch: the pooled projection is normalized and added to the code (see _cond_input)
                    s = b.batch_scale * s / (s.norm() + 1e-8) if b.batch_scale > 0 else s * 0.0
                    out[id(b)] = hn(b.code + s)
        return out

    for f in convex:
        f.no_bruteforce = True   # block inputs do not depend on the log-det estimator
    flow.eval()
    try:
        for _ in range(max(1, int(n_passes))):
            params = pooled_params()
            for b in bodies:
                if params[id(b)] is not None:
                    b.hypernet = _FixedEmission(params[id(b)])
        yield
    finally:
        for b in bodies:
            b.hypernet = orig[id(b)]
        for f, w in zip(convex, was):
            f.no_bruteforce = w


def evaluate_nll(
    flow: SequentialFlow,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
    bruteforce: bool = True,
    x_cond: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return (mean NLL in nats, std-error of mean) on ``x``.

    ``x_cond`` pins the lifted blocks' emission on those rows (see
    :func:`pinned_emission`); ``None`` lets each scored chunk condition
    the weights that score it, which is the training-time semantics.
    """
    if x_cond is not None:
        with pinned_emission(flow, x_cond, device=device):
            return evaluate_nll(flow, x, device=device, batch_size=batch_size,
                                bruteforce=bruteforce, x_cond=None)
    flow.eval()
    # Switch DeepConvexFlow to bruteforce log-det.
    for f in flow.flows:
        if isinstance(f, DeepConvexFlow):
            f.no_bruteforce = not bool(bruteforce)
    nll_chunks: List[np.ndarray] = []
    rng = np.random.default_rng(0)
    for batch in _batch_iter(x, batch_size, shuffle=False, rng=rng):
        batch = batch.to(device)
        logp = flow.logp(batch)
        nll_chunks.append((-logp.detach().cpu().numpy()).astype(np.float64))
    arr = np.concatenate(nll_chunks)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    flow.train()
    # Restore stochastic estimator for training.
    for f in flow.flows:
        if isinstance(f, DeepConvexFlow):
            f.no_bruteforce = True
    return mean, se


def train_cpflow(
    flow: SequentialFlow,
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Optional[str] = None,
    backend_tag: str = "direct",
    val_bruteforce: bool = True,
) -> Dict[str, list]:
    """Train a CP-Flow with CP-Flow's training recipe.

    Returns a history dict ``{'iter': [...], 'train_nll': [...],
    'eval_iter': [...], 'val_nll': [...]}``.
    """
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    flow.to(device)
    optim = torch.optim.Adam(
        flow.parameters(),
        lr=float(cfg.lr),
        betas=(float(cfg.beta1), float(cfg.beta2)),
        weight_decay=float(cfg.wd),
    )

    # Data-dependent init using a fresh batch.
    init_batch = torch.from_numpy(
        train_data[rng.choice(train_data.shape[0], size=min(cfg.batch_size, 4096), replace=False)]
    ).float().to(device)
    init_all_actnorms(flow, init_batch)

    history = {
        "iter": [], "train_nll": [],
        "eval_iter": [], "val_nll": [], "val_nll_se": [],
        "snap_iter": [], "snap_paths": [],
        "wall_seconds": [],
    }

    snap_dir = None
    if cfg.snap_every > 0 and out_dir is not None:
        snap_dir = os.path.join(out_dir, f"snapshots_{backend_tag}")
        os.makedirs(snap_dir, exist_ok=True)

    # ---- Cross-cov accumulator setup ---------------------------------------
    # The accumulator is built only under ``cfg.with_cross_cov``, so with the
    # flag off there are no extra forwards and no extra autograd graphs.
    cross_cov_acc: Optional[CrossCovAccumulator] = None
    cross_cov_taps: List[BiasChannelTap] = []
    cross_cov_snap_iters: List[int] = []
    # Provenance only: written to the NPZ and used by no arithmetic. Every
    # backend runs the identical measurement (see BiasChannelTap).
    cross_cov_is_pgd = backend_tag.startswith("pgd")
    if cfg.with_cross_cov:
        if cfg.snap_every <= 0:
            raise ValueError(
                "with_cross_cov=True requires snap_every>0 to define the "
                "snapshot iter schedule for the cross-cov accumulator."
            )
        if int(cfg.cross_cov_n_probes) < 2:
            raise ValueError(
                "cross_cov_n_probes must be >= 2: the sample axis of "
                "the cross-covariance is the probe batches, and one "
                "probe has no fluctuation to measure. Got "
                f"{cfg.cross_cov_n_probes}."
            )
        cross_cov_snap_iters = (
            [1]
            + list(range(cfg.snap_every, cfg.n_iters + 1, cfg.snap_every))
        )
        if cross_cov_snap_iters[-1] != cfg.n_iters:
            cross_cov_snap_iters.append(int(cfg.n_iters))
        cross_cov_taps = _cross_cov_taps(flow)
        d_bias = _flat_bias_dim(cross_cov_taps)
        if not cross_cov_taps:
            # This flow exposes no positivity channel, so there is nothing to
            # measure. Nothing is persisted rather than a zero sentinel: an
            # absent measurement and a measured zero are different claims and
            # must not share a representation.
            print(
                f"[{backend_tag}] cross-cov: no positivity channel found "
                "on this flow (no hypernet readout heads and no "
                "PosLinear weights). No accumulator attached; no NPZ "
                "will be written.",
                flush=True,
            )
            history["cross_cov_status"] = "no_tappable_channel"
        else:
            jl_dim = (
                int(cfg.cross_cov_jl_dim)
                if cfg.cross_cov_jl_dim > 0 and cfg.cross_cov_jl_dim < d_bias
                else None
            )
            iterate_source = _tap_iterate_source(cross_cov_taps)
            print(
                f"[{backend_tag}] cross-cov: {len(cross_cov_taps)} taps, "
                f"d_bias={d_bias}, jl_dim={jl_dim}, "
                f"iterate_source={iterate_source}",
                flush=True,
            )
            history["cross_cov_status"] = "measured"
            history["cross_cov_iterate_source"] = iterate_source
            cross_cov_acc = CrossCovAccumulator(
                d_bias=d_bias,
                snap_iters=cross_cov_snap_iters,
                is_pgd=cross_cov_is_pgd,
                jl_proj_dim=jl_dim,
                device=device,
                iterate_source=iterate_source,
                on_degenerate=str(cfg.cross_cov_on_degenerate),
                n_permutations=int(cfg.cross_cov_n_permutations),
            )
            # The observer's own generator, built from cfg.seed and never
            # touched by training. Sharing ``rng`` would advance the training
            # batch stream and make the run diverge from its control.
            cross_cov_rng = make_probe_rng(int(cfg.seed))

    flow.train()
    if bool(getattr(cfg, "exact_logdet_train", False)):
        n_forced = 0
        for f in flow.flows:
            if isinstance(f, DeepConvexFlow):
                f.force_bruteforce = True
                n_forced += 1
        print(f"[{backend_tag}] EXACT log-det during training on "
              f"{n_forced} convex blocks (diagnostic mode)", flush=True)
    # Emission-decay schedule. Telling every emitter how far training has run
    # lets it anneal the batch-dependent component toward its running mean
    # late. The call is a no-op unless the run sets ``--emission_decay_frac``,
    # and the list is empty for the direct backend.
    decay_hypernets = _hypernets(flow)
    t0 = time.time()
    it = 0
    n_skipped = 0
    train_iter = _make_inf_batch_iter(train_data, cfg.batch_size, rng)
    while it < cfg.n_iters:
        it += 1
        for hn in decay_hypernets:
            hn.set_train_progress(float(it) / float(cfg.n_iters))
        x = next(train_iter).to(device)
        optim.zero_grad()
        loss = -flow.logp(x).mean()
        # The stochastic Lanczos/CG log-det estimator returns a non-finite
        # value on occasional batches, and backpropagating one such loss
        # poisons every weight permanently, which reads like divergence in the
        # log. The step is skipped and counted instead; the guard is
        # method-agnostic and the skip rate is reported.
        if not torch.isfinite(loss):
            n_skipped += 1
            history.setdefault("skipped_iters", []).append(int(it))
            optim.zero_grad(set_to_none=True)
            if n_skipped in (1, 10, 100) or n_skipped % 500 == 0:
                print(f"[{backend_tag}] iter {it}: non-finite loss, batch skipped "
                      f"(skip {n_skipped})", flush=True)
            if n_skipped > int(getattr(cfg, "max_skips", 2000)):
                print(f"[{backend_tag}] ABORT: {n_skipped} skipped batches",
                      flush=True)
                break
            continue
        loss.backward()
        if cfg.clip_grad and cfg.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.clip_grad)
        optim.step()
        if str(backend_tag).startswith("pgd"):
            from lift.baselines.cpflow_toy import project_positive_all
            project_positive_all(flow)          # the PGD step's projection

        history["iter"].append(int(it))
        history["train_nll"].append(float(loss.item()))

        if cfg.snap_every and (it == 1 or it % cfg.snap_every == 0 or it == cfg.n_iters):
            if snap_dir is not None:
                path = os.path.join(snap_dir, f"snap_{it:07d}.pt")
                torch.save({"iter": it, "state_dict": flow.state_dict()}, path)
                history["snap_iter"].append(int(it))
                history["snap_paths"].append(path)
            if cross_cov_acc is not None and it in cross_cov_snap_iters:
                # A diagnostic must never abort the run it is observing: a
                # failed snapshot is recorded and training carries on, so the
                # archive keeps whatever snapshots did land.
                try:
                    stats = record_cross_cov_snapshot(
                        flow,
                        accumulator=cross_cov_acc,
                        taps=cross_cov_taps,
                        train_data=train_data,
                        batch_size=int(cfg.batch_size),
                        n_probes=int(cfg.cross_cov_n_probes),
                        iter_idx=int(it),
                        device=device,
                        probe_rng=cross_cov_rng,
                        seed=int(cfg.seed),
                        exact_logdet=bool(cfg.cross_cov_exact_logdet),
                    )
                    if stats is not None:
                        print(
                            f"[{backend_tag}] cross-cov it={it}: "
                            f"tr={stats['sigma_Jac_sq']:+.4e} "
                            f"frob={stats['frob']:.4e} "
                            f"asym={stats['asymmetry']:.3f} "
                            f"rel_var(iter)={stats['iterate_rel_var']:.2e} "
                            f"perm_p={stats['perm_p_trace']:.3f}",
                            flush=True,
                        )
                except CrossCovDegenerateInputs as exc:
                    # The one failure not swallowed: a degenerate estimate
                    # means the instrument is measuring the wrong thing.
                    history["cross_cov_status"] = f"degenerate: {exc}"
                    raise
                except Exception as exc:   # noqa: BLE001
                    # The whole run is not stamped "failed", since later
                    # snapshots may succeed and the status is what a downstream
                    # reader gates on. The failure is recorded per iteration
                    # and the status downgraded once.
                    history["cross_cov_status"] = "measured_with_failures"
                    history["cross_cov_last_failure"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    history.setdefault(
                        "cross_cov_failed_iters", [],
                    ).append(int(it))
                    print(
                        f"[{backend_tag}] cross-cov it={it}: snapshot "
                        f"FAILED ({type(exc).__name__}: {exc}); training "
                        "continues.",
                        flush=True,
                    )
                # Persist the per-snapshot NPZ immediately, so the renderer
                # can read partway through training and a crash still leaves
                # the captured snapshots on disk.
                if out_dir is not None and cross_cov_acc.matrices:
                    npz_path = os.path.join(
                        out_dir,
                        f"{backend_tag}_cross_cov_T{it:07d}.npz",
                    )
                    cross_cov_acc.to_npz(npz_path)

        if it == 1 or it % cfg.eval_every == 0 or it == cfg.n_iters:
            val_nll, val_se = evaluate_nll(
                flow, val_data, device=device,
                batch_size=int(cfg.val_batch_size),
                bruteforce=bool(val_bruteforce),
            )
            history["eval_iter"].append(int(it))
            history["val_nll"].append(float(val_nll))
            history["val_nll_se"].append(float(val_se))
            history["wall_seconds"].append(float(time.time() - t0))
            # Early-stopping checkpoint: one file per run, overwritten
            # whenever the validation likelihood improves, so a reported
            # iterate can be scored on the test split without the cost of a
            # snapshot cadence. The selection rule is the same for every
            # construction and every seed.
            if bool(getattr(cfg, "save_bestval", False)) and out_dir is not None:
                if float(val_nll) < float(history.get("bestval_nll", [float("inf")])[-1]
                                          if history.get("bestval_nll") else float("inf")):
                    history.setdefault("bestval_nll", []).append(float(val_nll))
                    history["bestval_iter"] = int(it)
                    torch.save({"model": flow.state_dict(), "iter": int(it),
                                "val_nll": float(val_nll)},
                               os.path.join(out_dir, f"{backend_tag}_bestval.pt"))
            if decay_hypernets and decay_hypernets[0].emission_decay_frac > 0:
                # Written only by a decay run, so other curves.json files keep
                # exactly the keys they have.
                history.setdefault("emission_gain", []).append(
                    float(decay_hypernets[0].emission_gain),
                )
            flow.train()

        if it == 1 or it % cfg.log_every == 0 or it == cfg.n_iters:
            last_val = history["val_nll"][-1] if history["val_nll"] else float("nan")
            print(
                f"[{backend_tag}|seed{cfg.seed}] it={it:6d} "
                f"train_nll={loss.item():.4f} val_nll={last_val:.4f} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    # Final consolidated NPZ: one archive per backend run, with every
    # snapshot's (d_eff x d_eff) matrix stacked. The filmstrip renderer reads
    # this file.
    if cross_cov_acc is not None and out_dir is not None:
        consolidated_path = os.path.join(
            out_dir, f"{backend_tag}_cross_cov.npz",
        )
        cross_cov_acc.to_npz(consolidated_path)
        history["cross_cov_npz"] = consolidated_path
        history["cross_cov_snap_iters"] = list(cross_cov_acc.recorded_iters)

    return history


def _make_inf_batch_iter(x: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = x.shape[0]
    while True:
        idx = rng.permutation(n)
        for k in range(0, n - batch_size + 1, batch_size):
            yield torch.from_numpy(x[idx[k:k + batch_size]]).float()


# ---------------------------------------------------------------------------
# High-level driver entrypoint.
# ---------------------------------------------------------------------------
def load_hepmass() -> HEPMASS:
    return HEPMASS()
