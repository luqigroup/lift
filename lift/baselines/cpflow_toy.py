r"""Published CP-Flow toy-density recipe (Huang+ ICLR 2021, ``train_toy.py``).

Primitives the upstream toy driver needs that the vendored subset at
:mod:`lift.baselines.cpflow_vendor` does not provide. Every constant here
is upstream's, from the mirror at https://github.com/CW-Huang/CP-Flow:
the driver is ``train_toy.py``, ``ICNN3`` comes from ``lib/icnn.py``, the
flows from ``lib/flows/``, the toy generators from ``data/toy_data.py`` and
the prior from ``lib/distributions.py``.

What this file adds beyond the vendored subset:

1. ``ICNN3``, the input-augmented ICNN the toy driver uses, transcribed
   from upstream; the vendored copy carries only ``ICNN2``. Its
   dependencies are imported from the vendored copy rather than re-typed.
2. ``log_normal``, upstream's general Gaussian log-density, since the toy
   prior is ``N(0, exp(plogv) I)`` and only the standard-normal special
   case is vendored.
3. :class:`ExactLogDet2DConvexFlow`, a closed-form 2x2 Hessian log-det
   path, which the vendored ``DeepConvexFlow`` has no analogue for.
4. The four toy generators, transcribed constant for constant, in the
   upstream convention and (for the two that have one) in this repo's own
   convention; see :data:`TARGET_CONVENTIONS`.

It also carries the lifted version of the recipe
(:class:`HypernetICNN3`): the identical ICNN3, with the constrained
weights emitted by :class:`lift.models.HyperNetwork` instead of stored as
free raw parameters. The layer list, the ActNorm placement and priming,
the potential head, the prior, the optimizer and the log-det path are
shared code, so the two constructions differ in one expression.

The tabular builder cannot be reused for the toy recipe: it hard-wires
``ICNN2`` and always emits ``nblocks + 1`` log-det-carrying ``ActNorm``
layers, where ``train_toy.py`` emits exactly one when ``nblocks == 1``.
:func:`build_toy_cpflow` mirrors that layer list instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call

from lift.baselines.cpflow_vendor.flows import (
    ActNorm,
    ActNormNoLogdet,
    DeepConvexFlow,
    SequentialFlow,
)
from lift.baselines.cpflow_vendor.icnn import (
    PosLinear,
    Softplus,
    symm_softplus,
)
from lift.models import HyperNetwork


Log2PI = float(np.log(2 * np.pi))


# ---------------------------------------------------------------------------
# Prior, transcribed from upstream ``lib/distributions.py``; the vendored
# copy keeps only the standard-normal special case.
# ---------------------------------------------------------------------------
def log_normal(x: Tensor, mean: Tensor, log_var: Tensor, eps: float = 0.00001) -> Tensor:
    """Upstream ``log_normal``, including its ``eps`` in the denominator.

    The ``eps`` is upstream's own, not a numerical patch added here.
    """
    z = -0.5 * Log2PI
    return -(x - mean) ** 2 / (2.0 * torch.exp(log_var) + eps) - log_var / 2.0 + z


# ---------------------------------------------------------------------------
# ICNN3: the input-augmented ICNN. Verbatim from upstream ``lib/icnn.py``.
# ---------------------------------------------------------------------------
class ICNN3(nn.Module):
    r"""Input-augmented ICNN (upstream ``lib/icnn.py::ICNN3``).

    Half of every hidden layer's units are a direct (augmented) branch
    of the input, so the potential's gradient carries a skip
    connection. Structure, verbatim:

    * ``Wzs[0] = Linear(dim, dimh)``; ``Wzs[1:-1] = PosLinear(dimh,
      dimh // 2, bias=True)``; ``Wzs[-1] = PosLinear(dimh, 1,
      bias=False)``.
    * ``Wxs``/``Wx2s`` are unconstrained ``Linear(dim, dimh // 2)``
      (the last ``Wxs`` is ``Linear(dim, 1, bias=False)``).
    * ``ActNormNoLogdet`` before every activation except the first
      (upstream's ICNN2 actnorms the first pre-activation, ICNN3 does
      not); the final ``ActNormNoLogdet(1)`` has its bias ``b``
      frozen -- an additive constant on a potential is a no-op on its
      gradient, so leaving it trainable would only add a null
      direction.
    * ``symm_softplus`` (``s(x) - x/2``) on the first activation and on
      the augmented branch when ``symm_act_first``; plain ``s`` inside.

    The vendored ``PosLinear`` body is
    ``F.linear(x, softplus(W), b) * (1 / fan_in)``. The ``1 / fan_in`` gain
    is inseparable from the init: the weights come from
    ``nn.Linear.reset_parameters``, zero-centered uniform, so
    ``softplus(W)`` is about 0.69 per entry and the gain is what keeps the
    output scale from growing with width. The bias is the ordinary learned
    ``nn.Linear`` bias, neither zeroed nor constrained.
    """

    def __init__(
        self,
        dim: int = 2,
        dimh: int = 16,
        num_hidden_layers: int = 2,
        symm_act_first: bool = False,
        softplus_type: str = "softplus",
        zero_softplus: bool = False,
    ) -> None:
        super().__init__()
        # with data dependent init
        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs: List[nn.Module] = [nn.Linear(dim, dimh)]
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh // 2, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = nn.ModuleList(Wzs)

        Wxs: List[nn.Module] = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = nn.ModuleList(Wxs)

        Wx2s: List[nn.Module] = []
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = nn.ModuleList(Wx2s)

        actnorms: List[nn.Module] = []
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        actnorms[-1].b.requires_grad_(False)
        self.actnorms = nn.ModuleList(actnorms)

    def forward(self, x: Tensor) -> Tensor:
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        for Wz, Wx, Wx2, actnorm in zip(
            self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:], self.actnorms[:-1],
        ):
            z = self.act(actnorm(Wz(z) + Wx(x)))
            aug = Wx2(x)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))


# ---------------------------------------------------------------------------
# Exact 2x2 log-det block.
# ---------------------------------------------------------------------------
class ExactLogDet2DConvexFlow(DeepConvexFlow):
    r"""``DeepConvexFlow`` with the closed-form 2x2 Hessian log-det.

    Used for both training and evaluation, which is what the CP-Flow paper's
    appendix states for the toy densities. The upstream driver leaves
    ``no_bruteforce=True`` instead, so training uses the conjugate-gradient
    log-det gradient surrogate and evaluation uses stochastic Lanczos
    quadrature. At ``d = 2`` the Krylov parts of both are exact, since two
    iterations span R^2, but both still contract the Hessian with one
    Rademacher probe per sample; the closed form here is the zero-variance
    limit of that estimator. ``logdet_mode="upstream_stochastic"`` in
    :func:`build_toy_cpflow` selects the upstream behavior.

    The determinant is ``h11 h22 - h12 h21`` over the two Hessian rows, each
    from one extra ``autograd.grad`` on ``f = grad_x F``. ``F`` is strictly
    convex, so in exact arithmetic the determinant is positive; the argument
    holds for the lifted construction too, whose emitted weights are a
    function of a detached conditioning batch and are therefore constants in
    the x-derivatives. Monotonicity is nevertheless audited rather than
    assumed: every determinant passes through :meth:`_audit`, so a
    non-positive one is counted and reported. The clamp at ``det_floor``
    still fires, because a NaN in the loss would destroy the run that was
    about to report the violation.

    The audit counters are on-device tensors updated without ``.item()``, so
    the training loop takes no host synchronization per iteration.
    """

    det_floor: float = 1e-300

    def forward_transform(
        self,
        x: Tensor,
        logdet=0,
        context=None,
        extra=None,
        icnn_override=None,
    ):
        bsz = x.shape[0]
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Fval = self.get_potential(x, context, icnn_override=icnn_override)
            f = torch.autograd.grad(Fval.sum(), x, create_graph=True)[0]
            f_flat = f.reshape(bsz, -1)
            if f_flat.shape[1] != 2:
                raise ValueError(
                    "ExactLogDet2DConvexFlow is the closed-form 2x2 path; "
                    f"got dim={f_flat.shape[1]}",
                )
            rows = []
            for i in range(2):
                rows.append(
                    torch.autograd.grad(
                        f_flat[:, i].sum(),
                        x,
                        create_graph=self.training,
                        retain_graph=True,
                    )[0].reshape(bsz, -1)
                )
        h11, h12 = rows[0][:, 0], rows[0][:, 1]
        h21, h22 = rows[1][:, 0], rows[1][:, 1]
        det = h11 * h22 - h12 * h21
        self._audit(det)
        return f, logdet + torch.log(det.clamp(min=self.det_floor))

    # -- determinant audit ------------------------------------------------
    def _audit(self, det: Tensor) -> None:
        d = det.detach()
        if getattr(self, "_audit_n_seen", None) is None:
            self.reset_det_audit()
        self._audit_n_seen = self._audit_n_seen + d.numel()
        self._audit_n_nonpos = self._audit_n_nonpos + (d <= 0).sum()
        self._audit_min = torch.minimum(
            self._audit_min.to(d.device, d.dtype), d.min(),
        )

    def reset_det_audit(self) -> None:
        self._audit_n_seen = 0
        self._audit_n_nonpos = torch.zeros((), dtype=torch.long)
        self._audit_min = torch.full((), float("inf"))

    def read_det_audit(self) -> Dict[str, float]:
        if getattr(self, "_audit_n_seen", None) is None:
            self.reset_det_audit()
        return {
            "n_evaluated": int(self._audit_n_seen),
            "n_nonpositive": int(self._audit_n_nonpos.item()),
            "min_det": float(self._audit_min.item()),
        }


# ---------------------------------------------------------------------------
# The lift: same ICNN3, constrained weights emitted instead of stored.
# ---------------------------------------------------------------------------
class _PosWeightView(nn.Module):
    """Name-preserving view of an :class:`ICNN3`'s constrained weights.

    :class:`~lift.models.HyperNetwork` builds one readout head per entry of
    ``downstream_network.named_parameters()`` and keeps only the sizes, never
    the module, so a throwaway module exposing exactly the tensors to be
    emitted -- under their ICNN3 dotted names, so ``functional_call`` can
    substitute them -- is enough to restrict the emission.

    Only the constrained weights are lifted. The unconstrained ``Wxs``,
    ``Wx2s`` and ``Wzs[0]`` linears, the ``PosLinear`` biases, which sit
    outside the positivity map, and every in-ICNN ``ActNormNoLogdet`` stay
    ordinary parameters, trained and data-initialized as in the published
    recipe; emitting them would break the data-dependent ActNorm init that
    the two constructions must share.

    All of ICNN3's ``PosLinear`` layers live in ``Wzs``. The constructor
    asserts it, so an upstream change that moves one elsewhere fails loudly
    instead of leaving it unlifted.
    """

    def __init__(self, icnn: ICNN3) -> None:
        super().__init__()
        holders: List[nn.Module] = []
        for m in icnn.Wzs:
            h = nn.Module()
            if isinstance(m, PosLinear):
                h.weight = nn.Parameter(
                    torch.empty_like(m.weight), requires_grad=False,
                )
            holders.append(h)
        self.Wzs = nn.ModuleList(holders)
        n_here = sum(1 for m in icnn.Wzs if isinstance(m, PosLinear))
        n_all = sum(1 for m in icnn.modules() if isinstance(m, PosLinear))
        if n_here != n_all:
            raise AssertionError(
                f"{n_all - n_here} PosLinear layer(s) of the ICNN3 live "
                "outside Wzs; the emitted-weight view would miss them",
            )


class PGDPosLinear(PosLinear):
    """``PosLinear`` with positivity by projection instead of softplus.

    The forward pass uses the raw weight with ``PosLinear``'s ``1/fan_in``
    gain, and after every optimizer step :meth:`project_positive` clamps that
    weight to ``>= 0``. Built through :func:`pgd_from_published`, the raw
    weight starts at ``softplus(W_published)``, so PGD and the published
    recipe start from the same effective weight at the same seed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = 1 / x.size(1)
        return F.linear(x, self.weight, self.bias) * gain

    @torch.no_grad()
    def project_positive(self) -> None:
        self.weight.clamp_(min=0.0)


def pgd_from_published(icnn: ICNN3) -> ICNN3:
    """Replace every ``PosLinear`` of an ICNN3 by a :class:`PGDPosLinear` at the
    same effective initial weight (``softplus`` of the published raw weight)."""
    for i, m in enumerate(icnn.Wzs):
        if isinstance(m, PosLinear) and not isinstance(m, PGDPosLinear):
            p = PGDPosLinear(m.in_features, m.out_features, bias=m.bias is not None)
            p = p.to(m.weight.device, m.weight.dtype)
            with torch.no_grad():
                p.weight.copy_(F.softplus(m.weight))
                if m.bias is not None:
                    p.bias.copy_(m.bias)
            icnn.Wzs[i] = p
    return icnn


def pgd_layers(flow: nn.Module) -> List[PGDPosLinear]:
    return [m for m in flow.modules() if isinstance(m, PGDPosLinear)]


def project_positive_all(flow: nn.Module) -> None:
    """The PGD step's projection: clamp every raw constrained weight to ``>= 0``."""
    for m in pgd_layers(flow):
        m.project_positive()


class HypernetICNN3(nn.Module):
    r"""ICNN3 whose constrained weights are emitted rather than stored.

    In the published recipe a constrained weight is a free raw parameter
    ``W~`` that ``PosLinear`` maps to ``softplus(W~) / fan_in``. Here

    ``W~ = Theta_E h_phi(X) + b_h``,

    where ``h_phi`` is a permutation-invariant DeepSets summary of the
    conditioning batch ``X`` (a per-sample MLP followed by a mean pool,
    standardized when ``pool_norm``), ``Theta_E`` is the unconstrained
    linear readout and ``b_h`` is that readout's bias, a free
    full-dimensional tensor per constrained weight, which is what lets the
    emitted weight reach any value the raw parameter could. The positivity
    map and the ``1 / fan_in`` gain are then applied to ``W~`` by the
    unmodified ``PosLinear.forward``: the same expression on a different
    argument, and the entire difference between the two constructions.

    The emitter is :class:`lift.models.HyperNetwork`, unmodified and with
    ``pos_param_names`` empty, so its own readout-level positivity is inert
    and positivity stays inside ``PosLinear`` where the published recipe puts
    it. The tabular driver wires the same emitter onto an ``ICNN2``, defaults
    to a learnable code rather than the batch, jitters the readout biases,
    and marks the template's ActNorms initialized, which would suppress the
    data-dependent priming both constructions must share; this class is the
    wiring for those differences, not a second emitter.

    The conditioning batch is the batch as it enters this block, which for
    ``nblocks > 1`` is the previous block's output rather than the raw data,
    and it is detached. Detaching is what keeps the emitted weights constant
    with respect to ``x``, so the potential stays convex in ``x`` and the
    Hessian log-det means what it says.

    ``readout_bias_init`` defaults to zero rather than a negative bias, so
    that the emitted latent weight starts zero-centered like the published
    one, at maximum softplus gradient and with no weight on the shoulder.
    The two inits still differ in spread, since the emitted weight inherits
    the readout's scale; ``readout_weight_init_scale`` can equalize them and
    defaults to ``1.0``, the emitter's own init.
    """

    def __init__(
        self,
        *,
        dim: int = 2,
        dimh: int = 32,
        depth: int = 20,
        symm_act_first: bool = True,
        softplus_type: str = "gaussian_softplus",
        zero_softplus: bool = True,
        hyper_hidden: Sequence[int] = (64, 64, 96),
        pool_norm: bool = True,
        pool_scale: float = 0.0,
        use_outer_net: bool = False,
        readout_bias_init: float = 0.0,
        readout_weight_init_scale: float = 1.0,
        emission_decay_frac: float = 0.0,
        emission_decay_end: float = 1.0,
        emission_decay_floor: float = 0.0,
        emission_decay_ema: float = 0.99,
        body_layernorm: bool = False,
        readout_fanin_scale: bool = False,
    ) -> None:
        super().__init__()
        # The template is built FIRST, so it draws from the global RNG in
        # the same order the published recipe does: at one seed the shared
        # parameters (Wxs, Wx2s, Wzs[0], the ActNorms and the PosLinear
        # biases) are then bit-identical across the two constructions.
        # ``emission_decay_*`` and ``body_layernorm`` pass to the emitter
        # unchanged and draw nothing from the RNG.
        self.icnn_template = ICNN3(
            dim,
            dimh,
            depth,
            symm_act_first=symm_act_first,
            softplus_type=softplus_type,
            zero_softplus=zero_softplus,
        )
        view = _PosWeightView(self.icnn_template)
        self.emitted_names: List[str] = [
            n for n, _ in view.named_parameters()
        ]
        # The raw constrained weights are replaced on every forward, so
        # freeze them: they must neither collect gradients nor count as
        # trainable parameters here.
        for name, p in self.icnn_template.named_parameters():
            if name in set(self.emitted_names):
                p.requires_grad_(False)
        self.hypernet = HyperNetwork(
            input_size=int(dim),
            hidden_sizes=list(hyper_hidden),
            downstream_network=view,
            use_outer_net=bool(use_outer_net),
            pool_norm=bool(pool_norm),
            pool_scale=float(pool_scale),
            pos_param_names=None,   # -> empty: positivity is PosLinear's
            pos_constraint_mode="softplus",
            emission_decay_frac=float(emission_decay_frac),
            emission_decay_end=float(emission_decay_end),
            emission_decay_floor=float(emission_decay_floor),
            emission_decay_ema=float(emission_decay_ema),
            body_layernorm=bool(body_layernorm),
            readout_fanin_scale=bool(readout_fanin_scale),
        )
        # ``head_init_pos_bias`` is inert here, being gated on a name in
        # ``pos_param_names``, which is empty, so the slack init is set
        # explicitly rather than left at ``nn.Linear``'s U(+/-1/sqrt(h)).
        self.readout_bias_init = float(readout_bias_init)
        self.readout_weight_init_scale = float(readout_weight_init_scale)
        with torch.no_grad():
            for layer in self.hypernet.weight_predictors:
                layer.bias.fill_(self.readout_bias_init)
                if self.readout_weight_init_scale != 1.0:
                    layer.weight.mul_(self.readout_weight_init_scale)
        self._pinned: Optional[Dict[str, Tensor]] = None
        self._use_pinned: bool = False
        self._capture: bool = False

    # -- emission ---------------------------------------------------------
    def emit(self, x: Tensor) -> Dict[str, Tensor]:
        """Emitted latent weights for conditioning batch ``x``."""
        return self.hypernet(x.detach())

    def forward(self, x: Tensor) -> Tensor:
        if self._use_pinned and self._pinned is not None:
            params = self._pinned
        else:
            params = self.emit(x)
            if self._capture:
                self._pinned = {k: v.detach() for k, v in params.items()}
        # Partial substitution: only the constrained weights are replaced,
        # so every other parameter and buffer is the template's own and the
        # in-ICNN ActNorm data-dependent init fires as it does in the
        # published recipe.
        return functional_call(self.icnn_template, params, (x,))


def is_lifted_body(icnn: nn.Module) -> bool:
    """Whether this ICNN body emits its constrained weights or stores them.

    Duck-typed on purpose: :class:`HypernetICNN3` is the toy body, but the
    tabular driver wires the same emitter onto an ``ICNN2`` in a different
    class. An ``isinstance`` test would classify that body as published and
    read its frozen template weights, which are never updated, so every
    weight statistic taken from it would be meaningless. Any body carrying a
    ``hypernet`` counts as lifted, and one that cannot hand back its
    emission raises in :func:`constrained_latent_weights` rather than
    falling through to the template.
    """
    return isinstance(icnn, HypernetICNN3) or hasattr(icnn, "hypernet")


def lifted_bodies(flow: SequentialFlow) -> List[nn.Module]:
    """The lifted ICNN bodies of ``flow``; empty for the published recipe."""
    return [
        f.icnn for f in flow.flows
        if isinstance(f, DeepConvexFlow) and is_lifted_body(f.icnn)
    ]


@contextmanager
def pinned_emission(
    flow: SequentialFlow, x_cond: Optional[Tensor],
) -> Iterator[None]:
    """Freeze every block's emission at the weights ``x_cond`` induces.

    Inside the context the flow is a fixed density: the emitted weights no
    longer depend on the batch being scored, so a likelihood computed here
    is the normalized log-density of one model. Outside it, or with
    ``x_cond=None``, each batch conditions the weights that score it, which
    is not a normalized likelihood.

    A no-op on the published recipe, which has nothing to emit.
    """
    # Only a body whose emission depends on the batch has anything to pin. A
    # code-conditioned body emits the same weights for every batch and is
    # left alone rather than handed attributes it does not own.
    bodies = [b for b in lifted_bodies(flow) if hasattr(b, "_capture")]
    if not bodies or x_cond is None:
        yield
        return
    for b in bodies:
        b._capture = True
        b._use_pinned = False
    # One forward pass fills every block's cache with the emission induced
    # by the conditioning batch as that block sees it, each block
    # conditioning on its own input exactly as in training. Under
    # ``no_grad``, since the conditioning set may be the whole training set
    # and nothing here is differentiated; the convex block opens its own
    # ``enable_grad`` for the x-derivatives it needs.
    with torch.no_grad():
        flow.forward_transform(x_cond)
    for b in bodies:
        b._capture = False
        b._use_pinned = True
    try:
        yield
    finally:
        for b in bodies:
            b._use_pinned = False
            b._pinned = None


@contextmanager
def _null_ctx() -> Iterator[None]:
    yield


def projection_arm(flow: SequentialFlow) -> bool:
    """Whether ``flow``'s positivity is imposed by projection, not softplus.

    The shoulder statistic below is ``sigmoid(w) <= threshold``, that is
    ``w <= logit(threshold) < 0``. A projected weight is clamped to
    ``w >= 0``, so that condition is unsatisfiable and the fraction would be
    a meaningless 0.0 for every PGD run. Callers get this flag and the
    projection's own statistic instead.
    """
    return any(isinstance(m, PGDPosLinear) for m in flow.modules())


def constrained_latent_weights(
    flow: SequentialFlow, x_cond: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """The latent, pre-positivity constrained weights of ``flow``.

    For the published recipe these are the raw ``PosLinear.weight``
    parameters; for the lift they are the emitted
    ``Theta_E h_phi(X) + b_h`` at the conditioning batch ``x_cond``. It is
    the same tensor in the same place in the same expression, which is what
    makes the shoulder diagnostic comparable across the two.
    """
    bodies = lifted_bodies(flow)
    out: Dict[str, Tensor] = {}
    # A lifted body that keeps its own emission accessor, as the tabular one
    # does, is read through it: it is not a ``HypernetICNN3`` and has no
    # pinning machinery.
    other = [b for b in bodies if not hasattr(b, "_pinned")]
    if other:
        for b in other:
            if not hasattr(b, "emitted_constrained_weights"):
                raise TypeError(
                    f"{type(b).__name__} emits its constrained weights but "
                    "exposes no way to read them; add "
                    "emitted_constrained_weights(x_cond) rather than letting "
                    "the diagnostic read a frozen template",
                )
        # Each block must be read at the emission its own input induces,
        # which past the first block is not the raw conditioning batch. One
        # forward pass with a pre-hook records that input per block; a
        # code-conditioned body ignores it.
        seen: Dict[int, Tensor] = {}
        if x_cond is not None:
            def _rec(mod, inp, _b=None):
                seen[id(mod)] = inp[0].detach()
            handles = [b.register_forward_pre_hook(_rec) for b in other]
            try:
                with torch.no_grad():
                    flow.forward_transform(x_cond)
            finally:
                for h in handles:
                    h.remove()
        for i, b in enumerate(other):
            emitted = b.emitted_constrained_weights(seen.get(id(b), x_cond))
            for k, v in emitted.items():
                out[f"block{i}.{k}"] = v.detach()
        return out
    if bodies:
        already = all(
            b._use_pinned and b._pinned is not None for b in bodies
        )
        if not already and x_cond is None:
            raise ValueError(
                "the lifted arm's latent weights are a function of a "
                "conditioning batch; pass x_cond (or call inside "
                "pinned_emission)",
            )
        # Each block must be read at the emission its own input induces,
        # which past the first block is not the raw conditioning batch.
        # Pinning walks the stack once and caches exactly that.
        ctx = _null_ctx() if already else pinned_emission(flow, x_cond)
        with ctx:
            for i, b in enumerate(bodies):
                for k in b.emitted_names:
                    out[f"block{i}.{k}"] = b._pinned[k].detach()
        return out
    for i, f in enumerate(flow.flows):
        if not isinstance(f, DeepConvexFlow):
            continue
        for name, mod in f.icnn.named_modules():
            if isinstance(mod, PosLinear):
                out[f"block{i}.{name}.weight"] = mod.weight.detach()
    return out


# ---------------------------------------------------------------------------
# Determinant audit helpers, for every construction.
# ---------------------------------------------------------------------------
def reset_det_audit(flow: SequentialFlow) -> None:
    """Zero the Hessian-determinant counters on every convex block."""
    for f in flow.flows:
        if isinstance(f, ExactLogDet2DConvexFlow):
            f.reset_det_audit()


def read_det_audit(flow: SequentialFlow) -> Dict[str, float]:
    """Aggregate the Hessian-determinant audit over the flow's blocks.

    ``n_nonpositive > 0`` means the map was not monotone somewhere, so the
    log-det is meaningless there and the reported NLL is not a likelihood.
    """
    n_seen = 0
    n_bad = 0
    min_det = float("inf")
    per_block = []
    for f in flow.flows:
        if isinstance(f, ExactLogDet2DConvexFlow):
            a = f.read_det_audit()
            per_block.append(a)
            n_seen += a["n_evaluated"]
            n_bad += a["n_nonpositive"]
            min_det = min(min_det, a["min_det"])
    return {
        "n_evaluated": int(n_seen),
        "n_nonpositive": int(n_bad),
        "min_det": float(min_det),
        "per_block": per_block,
    }


# ---------------------------------------------------------------------------
# Model builder: mirrors ``train_toy.py``'s layer list.
# ---------------------------------------------------------------------------
#: The CP-Flow appendix table "Architectural details for toy density
#: estimation" (nblocks / hidden layers / hidden units), all three rows as
#: the table has them. :func:`resolve_recipe` looks the row up by target
#: name, so this is the one place the table is written down.
PAPER_APPENDIX_TABLE: Dict[str, Dict[str, int]] = {
    "one_moon": {"nblocks": 5, "depth": 3, "dimh": 32},
    "eight_gaussians": {"nblocks": 5, "depth": 3, "dimh": 32},
    "rings": {"nblocks": 5, "depth": 5, "dimh": 256},
}

#: The appendix table has no 2-spirals row, so recipe B on that target
#: borrows the Eight-Gaussians row. The substitution is not the paper's and
#: travels with the run as ``recipe_note``.
RECIPE_B_ROW_FOR_TARGET: Dict[str, str] = {
    "two_spirals": "eight_gaussians",
}

#: The two published configurations, both selectable.
#:
#: A -- ``train_toy.py``'s argparse defaults, which are also its README
#:      command. Upstream applies them to whatever ``--data`` is passed, so
#:      A is target-independent, and being a repo default rather than a row
#:      of the appendix table it is not the published configuration for any
#:      particular target.
#: B -- the appendix table, one row per target, at 50 epochs. The entry
#:      below is the Eight-Gaussians row; the row actually used is resolved
#:      per target by :func:`resolve_recipe`.
PUBLISHED_RECIPES: Dict[str, Dict[str, int]] = {
    "A": {"nblocks": 1, "depth": 20, "dimh": 32, "num_epochs": 10},
    "B": {**PAPER_APPENDIX_TABLE["eight_gaussians"], "num_epochs": 50},
}

_RECIPE_A_NOTE = (
    "recipe A is train_toy.py's argparse default (== the upstream README "
    "command), which upstream applies to whatever --data is passed; the "
    "paper's appendix table gives a different architecture per toy target, "
    "so A is a repo default and NOT the published configuration for this "
    "target"
)


def resolve_recipe(recipe: str, target: str) -> Dict[str, object]:
    """Architecture and epoch count for ``(recipe, target)``.

    Recipe A is one configuration for every target, upstream's argparse
    defaults. Recipe B is the appendix table, so its row depends on the
    target: One moon and Eight Gaussians at 5 / 3 / 32, Rings at 5 / 5 /
    256. The row is looked up in :data:`PAPER_APPENDIX_TABLE`, never copied.

    Returns ``nblocks``, ``depth``, ``dimh`` and ``num_epochs``, plus the
    ``recipe_note`` and ``arch_source`` the experiment records verbatim in
    ``metrics.json``. ``"custom"`` resolves to A's numbers, so that a config
    overriding some fields still has a fallback for those left at ``-1``.
    """
    r = str(recipe).upper()
    if target not in TARGETS:
        raise ValueError(
            f"unknown target {target!r}; expected one of {TARGETS}",
        )
    if r in ("A", "CUSTOM"):
        out: Dict[str, object] = dict(PUBLISHED_RECIPES["A"])
        out["recipe_note"] = (
            _RECIPE_A_NOTE if r == "A"
            else "custom: unset (-1) fields fall back to recipe A's values"
        )
        out["arch_source"] = (
            "CP-Flow train_toy.py argparse defaults (target-independent)"
        )
        return out
    if r == "B":
        row_target = RECIPE_B_ROW_FOR_TARGET.get(target, target)
        row = PAPER_APPENDIX_TABLE[row_target]
        out = dict(row)
        out["num_epochs"] = int(PUBLISHED_RECIPES["B"]["num_epochs"])
        out["arch_source"] = (
            "CP-Flow appendix 'Architectural details for toy density "
            f"estimation', row '{row_target}'"
        )
        out["recipe_note"] = (
            "" if row_target == target else
            "the paper's appendix table has no 2-spirals row (its toy "
            "figure is One moon / Eight Gaussians / Rings), so recipe B on "
            "2-spirals borrows the Eight-Gaussians row 5/3/32; that "
            "substitution is ours, not the paper's"
        )
        return out
    raise ValueError(
        f"recipe must be 'A', 'B' or 'custom'; got {recipe!r}",
    )


def build_toy_cpflow(
    *,
    dim: int = 2,
    dimh: int = 32,
    depth: int = 20,
    nblocks: int = 1,
    symm_act_first: bool = True,
    softplus_type: str = "gaussian_softplus",
    zero_softplus: bool = True,
    bias_w1: float = -0.0,
    trainable_w0: Optional[bool] = None,
    logdet_mode: str = "exact2x2",
    m1: int = 10,
    atol: float = 1e-3,
    rtol: float = 0.0,
    backend: str = "published",
    hyper_hidden: Sequence[int] = (64, 64, 96),
    pool_norm: bool = True,
    pool_scale: float = 0.0,
    use_outer_net: bool = False,
    readout_bias_init: float = 0.0,
    readout_weight_init_scale: float = 1.0,
    emission_decay_frac: float = 0.0,
    emission_decay_end: float = 1.0,
    emission_decay_floor: float = 0.0,
    emission_decay_ema: float = 0.99,
    body_layernorm: bool = False,
    readout_fanin_scale: bool = False,
) -> SequentialFlow:
    """Build ``train_toy.py``'s CP-Flow stack.

    Layer list, verbatim from upstream:

    * ``nblocks == 1`` gives ``[ActNorm(dim), DeepConvexFlow(...)]``, that
      is exactly one log-det-carrying ActNorm, with ``w0`` trainable.
    * ``nblocks > 1`` gives ``nblocks + 1`` ActNorms interleaved with
      ``nblocks`` DeepConvexFlows, and ``trainable_w0=False``.

    ``trainable_w0=None`` resolves to that rule; pass a bool to override it.

    ``backend`` selects the positivity construction and nothing else:
    ``"published"`` gives each block a plain :class:`ICNN3` whose
    constrained weights are free raw parameters, ``"pgd"`` the same ICNN3
    with those weights projected instead, and ``"lifted"`` a
    :class:`HypernetICNN3` that emits them. Every other argument, the layer
    list, the ActNorm placement, the flow class, ``w0`` and ``w1`` and the
    log-det path are shared.
    """
    if trainable_w0 is None:
        trainable_w0 = nblocks == 1
    if logdet_mode == "exact2x2":
        flow_cls = ExactLogDet2DConvexFlow
    elif logdet_mode == "upstream_stochastic":
        flow_cls = DeepConvexFlow
    else:
        raise ValueError(f"unknown logdet_mode {logdet_mode!r}")
    if backend not in ("published", "lifted", "pgd"):
        raise ValueError(f"unknown backend {backend!r}")

    def _body() -> nn.Module:
        if backend in ("published", "pgd"):
            icnn = ICNN3(
                dim,
                dimh,
                depth,
                symm_act_first=symm_act_first,
                softplus_type=softplus_type,
                zero_softplus=zero_softplus,
            )
            # "pgd": same ICNN3 and same seed-drawn init, with the
            # constrained weights turned into raw, projected parameters at
            # the same effective value.
            return pgd_from_published(icnn) if backend == "pgd" else icnn
        return HypernetICNN3(
            dim=dim,
            dimh=dimh,
            depth=depth,
            symm_act_first=symm_act_first,
            softplus_type=softplus_type,
            zero_softplus=zero_softplus,
            hyper_hidden=hyper_hidden,
            pool_norm=pool_norm,
            pool_scale=pool_scale,
            use_outer_net=use_outer_net,
            readout_bias_init=readout_bias_init,
            readout_weight_init_scale=readout_weight_init_scale,
            emission_decay_frac=emission_decay_frac,
            emission_decay_end=emission_decay_end,
            emission_decay_floor=emission_decay_floor,
            emission_decay_ema=emission_decay_ema,
            body_layernorm=body_layernorm,
            readout_fanin_scale=readout_fanin_scale,
        )

    icnns = [_body() for _ in range(nblocks)]

    def _block(icnn: nn.Module) -> nn.Module:
        return flow_cls(
            icnn,
            dim,
            unbiased=False,
            bias_w1=bias_w1,
            trainable_w0=bool(trainable_w0),
            m1=m1,
            m2=dim,
            atol=atol,
            rtol=rtol,
        )

    layers: List[Optional[nn.Module]] = []
    if nblocks == 1:
        layers = [None] * (nblocks + 1)
        layers[0] = ActNorm(dim)
        layers[1:] = [_block(icnn) for icnn in icnns]
    else:
        layers = [None] * (2 * nblocks + 1)
        layers[0::2] = [ActNorm(dim) for _ in range(nblocks + 1)]
        layers[1::2] = [_block(icnn) for icnn in icnns]
    return SequentialFlow(layers)


def toy_logp(flow: SequentialFlow, x: Tensor, plogv: float) -> Tensor:
    """``train_toy.py``'s ``logp``: ``log N(z; 0, e^plogv I) + logdet``.

    Upstream's ``plogv`` is ``2`` when ``nblocks == 1`` and ``0``
    otherwise, an ``N(0, e^2 I)`` prior for the single-block model and a
    standard normal for the deep one. This is not ``SequentialFlow.logp``,
    which hard-codes the standard normal.
    """
    z, logdet = flow.forward_transform(x, context=None)
    mean = torch.zeros_like(x)
    log_var = torch.zeros_like(x) + plogv
    return log_normal(z, mean, log_var).sum(-1) + logdet


def resolve_plogv(nblocks: int) -> float:
    """``plogv = 2 if nblocks == 1 else 0`` (upstream ``train_toy.py``)."""
    return 2.0 if nblocks == 1 else 0.0


def mark_actnorms_initialized(flow: nn.Module, initialized: bool = True) -> int:
    """Set ``.initialized`` on every ActNorm in ``flow``.

    ``ActNorm.initialized`` is a plain Python attribute, not a buffer, so it
    does not survive ``state_dict`` / ``load_state_dict``. A reloaded flow
    would therefore re-run the data-dependent init on its first forward pass
    and overwrite the trained ``b`` and ``logs``. Every reload path must
    call this.
    """
    n = 0
    for mod in flow.modules():
        if isinstance(mod, ActNorm):
            mod.initialized = bool(initialized)
            n += 1
    return n


def prime_actnorms(flow: SequentialFlow, x_init: Tensor) -> None:
    """The single priming forward pass upstream runs before training.

    ``train_toy.py`` calls ``flow.logp(x.double()).mean()`` on the first
    training batch and drops the result. One pass is enough because
    ``SequentialFlow.forward_transform`` walks the stack in order, so every
    ``ActNorm`` and every in-ICNN ``ActNormNoLogdet`` sees its own
    data-dependent input. The same entry point is used here, which applies
    the standard-normal prior rather than :func:`toy_logp`'s ``plogv`` one;
    that is immaterial, since no optimizer step follows and only the ActNorm
    statistics are being set.
    """
    flow.logp(x_init).mean()


# ---------------------------------------------------------------------------
# Constrained-weight shoulder diagnostic.
# ---------------------------------------------------------------------------
def softplus_shoulder_fraction(
    flow: SequentialFlow,
    threshold: float = 0.05,
    x_cond: Optional[Tensor] = None,
) -> Dict[str, object]:
    """Fraction of constrained weights with ``softplus'(w) <= threshold``.

    ``PosLinear`` maps a latent weight ``w`` to ``softplus(w)``, so the
    local gain of the positivity reparametrization is
    ``softplus'(w) = sigmoid(w)``. Entries whose sigmoid has collapsed below
    ``threshold`` sit on the flat shoulder, their gradient attenuated by
    that factor.

    The latent weight is the raw parameter under the published recipe and
    the emitted ``Theta_E h_phi(X) + b_h`` under the lift, so the number
    means the same thing in both. Summary statistics of ``w`` travel with
    it: ``mean_softplus_w`` says whether the two started from a different
    effective weight, in which case the comparison is reading the init
    rather than the construction.
    """
    weights = constrained_latent_weights(flow, x_cond=x_cond)
    is_pgd = projection_arm(flow)
    n_zero = 0
    per_layer: List[float] = []
    n_low = 0
    n_tot = 0
    sum_w = 0.0
    sum_w2 = 0.0
    sum_sig = 0.0
    sum_sp = 0.0
    with torch.no_grad():
        for name in sorted(weights):
            w = weights[name].detach().double()
            sig = torch.sigmoid(w)
            low = int((sig <= threshold).sum().item())
            tot = int(sig.numel())
            per_layer.append(low / tot if tot else float("nan"))
            n_low += low
            n_tot += tot
            sum_w += float(w.sum().item())
            sum_w2 += float((w ** 2).sum().item())
            sum_sig += float(sig.sum().item())
            sum_sp += float(nn.functional.softplus(w).sum().item())
            n_zero += int((w <= 0).sum().item())
    mean_w = sum_w / n_tot if n_tot else float("nan")
    return {
        "source": (
            "emitted latent weight Theta_E h_phi(X) + b_h"
            if lifted_bodies(flow) else
            "raw PGDPosLinear.weight (PROJECTED, already non-negative)"
            if is_pgd else "raw PosLinear.weight"
        ),
        # A projected weight has no shoulder: ``fraction`` below is
        # structurally 0 and must never be quoted as one. Its analogue is
        # ``frac_exact_zero``, the set the projection actually pins.
        "is_projection_arm": bool(is_pgd),
        # Only meaningful under projection, where the weight is the
        # effective one and can be pinned exactly at zero. Under softplus
        # ``w <= 0`` is a latent-space count, at effective weight 0.693, so
        # it is not reported.
        "frac_exact_zero": ((n_zero / n_tot) if n_tot else float("nan"))
        if is_pgd else None,
        "shoulder_defined": not is_pgd,
        "threshold": float(threshold),
        "fraction": (n_low / n_tot) if n_tot else float("nan"),
        "n_low": n_low,
        "n_constrained_weights": n_tot,
        "per_layer_fraction": per_layer,
        "mean_w": mean_w,
        "std_w": (
            float(np.sqrt(max(sum_w2 / n_tot - mean_w ** 2, 0.0)))
            if n_tot else float("nan")
        ),
        "mean_sigmoid_w": sum_sig / n_tot if n_tot else float("nan"),
        "mean_softplus_w": sum_sp / n_tot if n_tot else float("nan"),
    }


def parameter_counts(flow: SequentialFlow) -> Dict[str, int]:
    """Total and trainable parameter counts, split by role.

    The emitter is far larger than the body it emits into, so the counts are
    reported rather than assumed comparable across constructions.
    """
    def _n(ps: Iterable[nn.Parameter], trainable_only: bool = False) -> int:
        return int(sum(
            p.numel() for p in ps if (p.requires_grad or not trainable_only)
        ))

    hyper = [
        p for b in lifted_bodies(flow) for p in b.hypernet.parameters()
    ]
    hyper_ids = {id(p) for p in hyper}
    rest = [p for p in flow.parameters() if id(p) not in hyper_ids]
    return {
        "total": _n(flow.parameters()),
        "trainable": _n(flow.parameters(), True),
        "trainable_emitter": _n(hyper, True),
        "trainable_icnn_and_flow": _n(rest, True),
        "frozen": _n(flow.parameters()) - _n(flow.parameters(), True),
    }


# ---------------------------------------------------------------------------
# Targets. Two conventions; absolute nats are not comparable across them, so
# every metrics file names the one in use.
# ---------------------------------------------------------------------------
#: ``cpflow``  -- ``data/toy_data.py`` in the upstream mirror.
#: ``lift_f1`` -- ``scripts/experiments_cpflow_demo_2d_f1.py`` in this repo,
#:                at radius 2.0 and per-mode sigma 0.1.
CONVENTIONS: Tuple[str, ...] = ("cpflow", "lift_f1")

#: The four upstream toy generators this module carries, from
#: ``data/toy_data.py``: ``EightGaussian``, ``TwoSpirals``, ``Rings`` and
#: ``MAFMoon``.
TARGETS: Tuple[str, ...] = (
    "eight_gaussians",
    "two_spirals",
    "rings",
    "one_moon",
)

#: Which conventions each target is defined in.
#:
#: ``rings`` and ``one_moon`` exist in the CP-Flow convention only. The
#: ``lift_f1`` convention is not a rescaling rule for an arbitrary
#: generator: it is the particular radius and width pair that this repo's
#: 2-D demo picked for its own two generators, and it has no ring and no
#: moon to map these onto.
TARGET_CONVENTIONS: Dict[str, Tuple[str, ...]] = {
    "eight_gaussians": ("cpflow", "lift_f1"),
    "two_spirals": ("cpflow", "lift_f1"),
    "rings": ("cpflow",),
    "one_moon": ("cpflow",),
}


def check_target_convention(target: str, convention: str) -> None:
    """Raise unless ``convention`` is a convention ``target`` has."""
    if target not in TARGETS:
        raise ValueError(
            f"unknown target {target!r}; expected one of {TARGETS}",
        )
    if convention not in CONVENTIONS:
        raise ValueError(
            f"unknown convention {convention!r}; expected one of "
            f"{CONVENTIONS}",
        )
    allowed = TARGET_CONVENTIONS[target]
    if convention not in allowed:
        raise ValueError(
            f"target {target!r} exists only in convention(s) {allowed}: it "
            "is upstream's own generator, and this repo's 'lift_f1' "
            "convention has no ring / moon analogue to map it onto. "
            "Re-run with --convention cpflow.",
        )


def _eight_gaussian_modes(convention: str) -> Tuple[np.ndarray, float]:
    """Mode centers and per-mode sigma of the 8-Gaussians target."""
    if convention == "cpflow":
        # data/toy_data.py::EightGaussian: scale 4 circle, per-component
        # sigma 0.5, the whole sample divided by 1.414 at the end.
        s = 1.0 / np.sqrt(2.0)
        centres = np.array(
            [(1, 0), (-1, 0), (0, 1), (0, -1), (s, s), (s, -s), (-s, s), (-s, -s)],
            dtype=np.float64,
        ) * 4.0 / 1.414
        return centres, 0.5 / 1.414
    if convention == "lift_f1":
        # experiments_cpflow_demo_2d_f1.py::sample_eight_gaussians.
        k = np.arange(8)
        centres = np.stack(
            [2.0 * np.cos(2 * np.pi * k / 8), 2.0 * np.sin(2 * np.pi * k / 8)], axis=-1,
        )
        return centres.astype(np.float64), 0.1
    raise ValueError(f"unknown convention {convention!r}")


def _rings_centres(n: int) -> np.ndarray:
    """``data/toy_data.py::Rings`` minus its additive noise, ``(n, 2)``.

    Four concentric circles of radii ``3 * (1, 0.75, 0.5, 0.25)``, with the
    nodes on an even angular grid at ``endpoint=False``; the Gaussian noise
    of scale ``0.08`` is added afterwards by :func:`sample_target`.

    Transcribed as written, including two lines that look like slips and are
    upstream's: ``circ3_x`` uses ``linspace4`` rather than ``linspace3``,
    which changes nothing because both are built from the same
    ``batch_size // 4``; and the three outer rings take ``batch_size // 4``
    nodes each while the innermost absorbs the remainder, so at
    ``n % 4 != 0`` the ring weights are not exactly 1/4. The reference
    statistics read the weights off this same enumeration rather than
    assuming quarters.

    Unlike the other three targets these centers are enumerated rather than
    sampled, one point per grid node, which is what makes this target's
    density exact rather than Monte Carlo. No RNG is consulted here;
    :func:`draw_centres` applies the shuffle.
    """
    n = int(n)
    if n < 4:
        raise ValueError(
            f"the Rings generator needs at least 4 samples (one per ring); "
            f"got n={n}",
        )
    n_samples4 = n_samples3 = n_samples2 = n // 4
    n_samples1 = n - n_samples4 - n_samples3 - n_samples2

    # so as not to have the first point = last point, endpoint=False
    linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
    linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
    linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
    linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

    circ4_x = np.cos(linspace4)
    circ4_y = np.sin(linspace4)
    circ3_x = np.cos(linspace4) * 0.75      # upstream's linspace4, sic
    circ3_y = np.sin(linspace3) * 0.75
    circ2_x = np.cos(linspace2) * 0.5
    circ2_y = np.sin(linspace2) * 0.5
    circ1_x = np.cos(linspace1) * 0.25
    circ1_y = np.sin(linspace1) * 0.25

    return np.vstack([
        np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
        np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
    ]).T * 3.0


#: Differential entropy of the One-moon (``MAFMoon``) target, in nats, in
#: closed form: the map is a triangular invertible transform of ``N(0, I_2)``
#: with constant Jacobian determinant ``1/2``, so
#: ``H(x) = H(z) - log 2 = log(2 pi e) - log 2``.
ONE_MOON_ENTROPY_NATS: float = 1.0 + Log2PI - float(np.log(2.0))


def _sample_one_moon(n: int, rng: np.random.Generator) -> np.ndarray:
    """``data/toy_data.py::MAFMoon``, ``(n, 2)`` float64.

    Upstream, verbatim::

        x = torch.randn(batch_size, 2)
        x[:, 0] += x[:, 1] ** 2
        x[:, 0] /= 2
        x[:, 0] -= 2

    that is ``z ~ N(0, I_2)``, ``x1 = z1``, ``x0 = (z0 + z1^2) / 2 - 2``.
    The arithmetic is transcribed statement for statement; the draw comes
    from the passed ``Generator`` rather than the global RNG, which is this
    module's convention for every target.

    This target does not decompose into center plus isotropic noise: it is a
    deterministic map of a Gaussian, not a mixture, so :func:`draw_centres`
    and :func:`target_noise_sigma` refuse it and its density is computed in
    closed form.
    """
    x = rng.standard_normal((int(n), 2))
    x[:, 0] += x[:, 1] ** 2
    x[:, 0] /= 2
    x[:, 0] -= 2
    return x


def _one_moon_log_prob(x: np.ndarray) -> np.ndarray:
    """Exact log-density of the One-moon target at ``x``, ``(N,)`` nats.

    The generator is triangular and invertible: ``z1 = x1``,
    ``z0 = 2 (x0 + 2) - x1^2``, with ``|det dz/dx| = 2`` everywhere, so
    ``log p(x) = log N(z; 0, I) + log 2``.
    """
    z1 = x[:, 1]
    z0 = 2.0 * (x[:, 0] + 2.0) - z1 ** 2
    return -0.5 * (z0 ** 2 + z1 ** 2) - Log2PI + np.log(2.0)


def target_noise_sigma(target: str, convention: str) -> float:
    """Isotropic Gaussian sd added to the noiseless center of a sample.

    Defined for the three mixture-shaped targets. ``one_moon`` is a
    pushforward of a Gaussian rather than a convolution with one, so it has
    no such parameter and raises.
    """
    if target == "eight_gaussians":
        return _eight_gaussian_modes(convention)[1]
    if target == "two_spirals":
        # cpflow: TwoSpirals adds randn * 0.1 after the /3 rescale.
        # lift_f1: the noise is added before the /3, so the effective sd in
        # final units is 0.1 / 3.
        return 0.1 if convention == "cpflow" else 0.1 / 3.0
    if target == "rings":
        # data/toy_data.py::Rings adds np.random.normal(scale=0.08) to the
        # already-scaled (x3) ring grid.
        check_target_convention(target, convention)
        return 0.08
    if target == "one_moon":
        raise ValueError(
            "one_moon (upstream MAFMoon) is not a centre + isotropic-noise "
            "generator: it is a triangular map of N(0, I), so it has no "
            "additive noise sd. Its density is closed form -- see "
            "target_log_prob_np.",
        )
    raise ValueError(f"unknown target {target!r}")


def draw_centres(
    target: str, convention: str, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Draw the noiseless part of ``n`` samples, ``(n, 2)`` float64.

    Splitting a generator into center plus isotropic Gaussian noise is what
    lets the reference statistics reuse the sampler itself: the target
    density is exactly ``E_centre[N(x; centre, sigma^2 I)]``, so the entropy
    floor is computed with the same code that produced the data.

    Three of the four targets decompose this way; ``one_moon`` does not and
    raises, its density being closed form.
    """
    check_target_convention(target, convention)
    if target == "eight_gaussians":
        centres, _ = _eight_gaussian_modes(convention)
        idx = rng.integers(0, 8, size=n)
        return centres[idx]
    if target == "rings":
        # The grid is deterministic; upstream then shuffles so the rows are
        # not ring-ordered. Same permutation here, from the Generator this
        # module passes everywhere. Row order cannot move a mixture density,
        # so target_log_prob_np reads the unshuffled grid directly.
        centres = _rings_centres(n)
        return centres[rng.permutation(centres.shape[0])]
    if target == "one_moon":
        raise ValueError(
            "one_moon (upstream MAFMoon) has no centre / noise "
            "decomposition: it is a deterministic map of N(0, I). Use "
            "sample_target, whose one_moon branch draws it directly.",
        )
    if target == "two_spirals":
        if convention == "cpflow":
            # data/toy_data.py::TwoSpirals, from BNAF's generate2d.py:
            # angle ~ sqrt(U) * 540 deg, a Uniform(0, 0.5) offset per
            # coordinate, the mirrored branch, then /3.
            half = (n + 1) // 2
            ang = np.sqrt(rng.random((half, 1))) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(ang) * ang + rng.random((half, 1)) * 0.5
            d1y = np.sin(ang) * ang + rng.random((half, 1)) * 0.5
            x = np.vstack(
                [np.hstack([d1x, d1y]), np.hstack([-d1x, -d1y])],
            ) / 3.0
            return x[:n]
        if convention == "lift_f1":
            # experiments_cpflow_demo_2d_f1.py::sample_two_spirals.
            n_per = (n + 1) // 2
            t = np.sqrt(rng.random(n_per)) * 540.0 * np.pi / 180.0
            r = t + np.pi
            s1 = np.stack([-r * np.cos(t), r * np.sin(t)], axis=-1)
            x = np.concatenate([s1, -s1], axis=0)[:n]
            return x / 3.0
        raise ValueError(f"unknown convention {convention!r}")
    raise ValueError(f"unknown target {target!r}")


def sample_target(
    target: str, convention: str, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` samples, ``(n, 2)`` float64.

    Under ``convention="cpflow"`` the sample is round-tripped through
    float32, because every upstream generator produces float32 and
    ``train_toy.py`` then casts back to double. The quantization is about
    1e-7 relative.
    """
    check_target_convention(target, convention)
    if target == "one_moon":
        x = _sample_one_moon(n, rng)
    else:
        sigma = target_noise_sigma(target, convention)
        x = draw_centres(target, convention, n, rng) + sigma * (
            rng.standard_normal((n, 2))
        )
    if convention == "cpflow":
        x = x.astype(np.float32).astype(np.float64)
    return x


def target_log_prob_np(
    target: str,
    convention: str,
    x: np.ndarray,
    n_latents: int = 50_000,
    rng: Optional[np.random.Generator] = None,
    chunk: int = 512,
) -> np.ndarray:
    """Log-density of the target at ``x`` (``(N, 2)``), in nats.

    Three of the four are exact and one is an estimate:

    * 8-Gaussians. The center takes 8 equiprobable values, so the density
      is the closed-form 8-component isotropic mixture.
    * Rings. The generator enumerates its centers on a deterministic
      angular grid rather than sampling them, so the equal-weight mixture
      over that grid is the density rather than an estimate of it. The grid
      mixture is the periodic trapezoid rule for the four ring integrals,
      whose error decays exponentially in the node count and at this
      spacing is below double precision, so the value does not depend on
      the batch size the grid was enumerated for.
    * One moon. A triangular invertible map of ``N(0, I_2)`` with constant
      Jacobian determinant; see :func:`_one_moon_log_prob`.
    * 2-spirals. No closed form, so the density is
      ``(1/M) sum_j N(x; c_j, sigma^2 I)`` over ``M = n_latents`` centers
      from :func:`draw_centres`, a consistent estimator of
      ``E_centre[N(x; centre, sigma^2 I)]``. Since ``E[log p_hat] <= log p``
      the entropy floor derived from it is biased slightly high. The
      estimator, ``M`` and the number of evaluation points are recorded in
      the metrics file.
    """
    check_target_convention(target, convention)
    if target == "one_moon":
        return _one_moon_log_prob(x)
    sigma = target_noise_sigma(target, convention)
    if target == "eight_gaussians":
        centres, _ = _eight_gaussian_modes(convention)
    elif target == "rings":
        centres = _rings_centres(int(n_latents))
    else:
        if rng is None:
            rng = np.random.default_rng(20210420)
        centres = draw_centres(target, convention, int(n_latents), rng)
    log_w = -np.log(float(centres.shape[0]))
    norm = -Log2PI - 2.0 * np.log(sigma)          # log 1/(2 pi sigma^2)
    out = np.empty(x.shape[0], dtype=np.float64)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        d2 = ((xb[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        lp = norm - 0.5 * d2 / (sigma ** 2) + log_w
        m = lp.max(axis=1, keepdims=True)
        out[i:i + chunk] = (m[:, 0] + np.log(np.exp(lp - m).sum(axis=1)))
    return out


def gaussian_moment_matched_nll(
    x_train: np.ndarray, x_eval: np.ndarray,
) -> float:
    """Mean NLL in nats per 2-D sample of the moment-matched Gaussian.

    Full-covariance ``N(mu, Sigma)`` fitted on the train set and scored on
    the evaluation set. A run reporting this number has learned the first
    two moments and nothing else.
    """
    mu = x_train.mean(axis=0)
    cov = np.cov(x_train, rowvar=False, bias=True)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("nan")
    prec = np.linalg.inv(cov)
    d = x_eval - mu
    quad = np.einsum("ni,ij,nj->n", d, prec, d)
    ll = -0.5 * (quad + logdet + 2 * Log2PI)
    return float(-ll.mean())


#: How each target's density, and therefore its entropy floor, is obtained.
#: ``exact`` means the log-density is closed form, or comes from the
#: generator's own deterministic enumeration, so the only randomness in the
#: floor is the average over evaluation points, which is unbiased and
#: carries a reported SEM. ``mc_latents`` means the density itself is a
#: Monte-Carlo estimate, which by Jensen biases the floor high.
ENTROPY_FLOOR_ESTIMATOR: Dict[str, str] = {
    "eight_gaussians": "exact",
    "rings": "exact",
    "one_moon": "exact",
    "two_spirals": "mc_latents",
}

_FLOOR_NOTE: Dict[str, str] = {
    "eight_gaussians": (
        "exact 8-component isotropic mixture density, MC average over "
        "test points (unbiased)"
    ),
    "rings": (
        "exact equal-weight Gaussian mixture over the generator's OWN "
        "deterministic centre grid (the generator enumerates its centres "
        "instead of sampling them, so the density is not estimated); MC "
        "average over test points (unbiased)"
    ),
    "one_moon": (
        "exact closed-form density -- triangular invertible map of "
        "N(0, I_2) with constant Jacobian determinant 2; MC average over "
        "test points (unbiased). The closed-form entropy log(2 pi e) - "
        "log 2 is reported alongside as entropy_floor_closed_form_nats"
    ),
    "two_spirals": (
        "MC over latent centres drawn from the generator -- an ESTIMATE, "
        "biased slightly high (see target_log_prob_np)"
    ),
}


def target_reference_stats(
    target: str,
    convention: str,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    n_latents: int = 50_000,
    n_eval: int = 5_000,
    seed: int = 20210420,
) -> Dict[str, object]:
    """Entropy floor and single-Gaussian NLL of the target in use.

    The entropy floor is ``-E_p[log p]``, estimated on a prefix of the fixed
    test set, which is p-distributed by construction, and is the smallest
    mean test NLL any density model can attain. It is returned alongside the
    moment-matched Gaussian NLL, so that a run can be placed on the interval
    between the Gaussian and the floor.

    ``entropy_floor_estimator`` records which case the number is in: for
    8-Gaussians, Rings and One moon the density is exact and only the
    average over test points is Monte Carlo, with its SEM reported; for
    2-spirals the density itself is a Monte-Carlo average over sampled
    centers, so the floor is biased slightly high.
    """
    check_target_convention(target, convention)
    xe = x_eval[: int(n_eval)]
    rng = np.random.default_rng(int(seed))
    lp = target_log_prob_np(
        target, convention, xe, n_latents=int(n_latents), rng=rng,
    )
    n_lat = {
        "eight_gaussians": 8,
        "rings": int(n_latents),        # deterministic grid nodes
        "one_moon": 0,                  # closed form, no latent set
    }.get(target, int(n_latents))
    out: Dict[str, object] = {
        "entropy_floor_nats": float(-lp.mean()),
        "entropy_floor_sem": float(lp.std(ddof=1) / np.sqrt(len(lp))),
        "entropy_floor_estimator": _FLOOR_NOTE[target],
        "entropy_floor_density_is_exact": (
            ENTROPY_FLOOR_ESTIMATOR[target] == "exact"
        ),
        "entropy_floor_n_latents": n_lat,
        "entropy_floor_n_eval_points": int(len(xe)),
        "gaussian_moment_matched_nll_nats": gaussian_moment_matched_nll(
            x_train, xe,
        ),
        "target_noise_sigma": (
            None if target == "one_moon"
            else float(target_noise_sigma(target, convention))
        ),
    }
    if target == "one_moon":
        out["entropy_floor_closed_form_nats"] = float(ONE_MOON_ENTROPY_NATS)
        out["target_noise_sigma_note"] = (
            "n/a: MAFMoon is a pushforward of N(0, I_2), not a "
            "convolution with isotropic noise"
        )
    return out


__all__ = [
    "CONVENTIONS",
    "ENTROPY_FLOOR_ESTIMATOR",
    "ExactLogDet2DConvexFlow",
    "HypernetICNN3",
    "ICNN3",
    "ONE_MOON_ENTROPY_NATS",
    "PAPER_APPENDIX_TABLE",
    "PUBLISHED_RECIPES",
    "RECIPE_B_ROW_FOR_TARGET",
    "TARGETS",
    "TARGET_CONVENTIONS",
    "build_toy_cpflow",
    "check_target_convention",
    "constrained_latent_weights",
    "draw_centres",
    "gaussian_moment_matched_nll",
    "lifted_bodies",
    "is_lifted_body",
    "projection_arm",
    "log_normal",
    "mark_actnorms_initialized",
    "parameter_counts",
    "pinned_emission",
    "prime_actnorms",
    "read_det_audit",
    "reset_det_audit",
    "resolve_plogv",
    "resolve_recipe",
    "sample_target",
    "softplus_shoulder_fraction",
    "target_log_prob_np",
    "target_noise_sigma",
    "target_reference_stats",
    "toy_logp",
]
