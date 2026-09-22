r"""Reading the slack channel off a live CP-Flow.

:mod:`lift.objectives.diagnostics` holds the estimator
(:class:`~lift.objectives.diagnostics.CrossCovAccumulator`); this module holds
the instrument that fills it: the taps that name one positivity channel apiece,
and the snapshot recorder that drives a fixed iterate with fresh probe batches
and reads both halves of the pair off each probe's single forward pass.

The recorder is a pure observer of the run it measures. It draws from its own
``numpy`` generator, forks the global torch RNG for the duration of the probe
loop, restores the flow's train/eval mode, and restores the emission-decay EMA
after every probe. A ``--with_cross_cov`` run is therefore bit-identical to the
same run without the flag, so the flag is not part of a run's identity and two
such runs may share an output directory.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from lift.baselines.cpflow_vendor import DeepConvexFlow, SequentialFlow
from lift.baselines.cpflow_vendor.icnn import PosLinear
from lift.models import HyperNetwork
from lift.objectives.diagnostics import CrossCovAccumulator


@dataclass(frozen=True, eq=False)
class BiasChannelTap:
    r"""One positivity channel, paired with how to read its iterate.

    The measured object is ``Sigma_slack = E[d(theta-tilde) d(g)^T]``: the left
    factor is the fluctuation of the pre-positivity iterate ``theta-tilde``, the
    right factor the fluctuation of the loss gradient on the parameter that
    carries that channel. A tap holds both halves, so the two rows are built
    from one object and cannot drift out of alignment.

    ``kind`` says where the iterate comes from:

    * ``"emitted"`` (the lift) --- the iterate is the emitted pre-positivity
      tensor ``theta-tilde = Theta_E h_phi(X) + b_h``, captured from the probe
      batch's own forward pass. The sample axis is probe batches at a fixed
      iterate, so the bias parameter ``b_h`` is constant and contributes
      nothing; what moves is the emitted term, because the hypernet is
      conditioned on the batch. Reading ``b_h`` itself would give a constant
      sequence and hence an exactly zero matrix, which ``iterate_rel_var``
      catches.
    * ``"parameter"`` (direct softplus) --- there is no emission: the iterate is
      the stored pre-softplus weight ``W-tilde``, which at a fixed iterate does
      not depend on the probe batch. Its fluctuation is therefore zero to
      float32 round-off, and ``Sigma_slack`` comes out at that floor by
      measurement.

    ``eq=False`` because the dataclass holds tensors, and a generated ``__eq__``
    over them raises an ambiguous-truth-value error.

    Attributes:
        hypernet: emitting hypernet, or ``None`` for a ``"parameter"`` tap.
        emitted_name: for ``"emitted"``, the hypernet target-tensor name
            (e.g. ``Wzs.3.weight``) whose captured emission is this channel's
            iterate; for ``"parameter"``, the dotted module path of the weight.
        grad_param: the parameter the loss gradient is taken on --- the readout
            bias ``b_h`` for ``"emitted"``, the pre-softplus weight for
            ``"parameter"``.
        kind: ``"emitted"`` or ``"parameter"``.
    """

    hypernet: Optional[HyperNetwork]
    emitted_name: str
    grad_param: torch.nn.Parameter
    kind: str

    @property
    def width(self) -> int:
        return int(self.grad_param.numel())

    def iterate(
        self, stores: Dict[int, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """This channel's flat iterate for the probe just evaluated."""
        if self.kind == "parameter":
            return self.grad_param.detach().reshape(-1)
        store = stores[id(self.hypernet)]
        if self.emitted_name not in store:
            raise RuntimeError(
                f"pre-positivity emission for {self.emitted_name!r} was "
                "not captured; the probe forward did not run this "
                "hypernet."
            )
        return store[self.emitted_name].reshape(-1)


def _gain_cap_active(icnn: torch.nn.Module) -> bool:
    """True if any gain cap is installed on this block, in either construction.

    The lift carries the cap on the wrapper
    (``HypernetParametrisedICNN.gain_cap``, applied by ``_cap_gain`` after the
    readout); direct softplus carries it on each ``GainCappedPosLinear`` child.
    ``icnn.modules()`` includes ``icnn`` itself, so one test covers both.
    """
    return any(
        float(getattr(m, "gain_cap", 0.0)) > 0.0 for m in icnn.modules()
    )


def cross_cov_taps(flow: SequentialFlow) -> List[BiasChannelTap]:
    r"""Enumerate the positivity channels of whichever construction ``flow`` is.

    For the lift: one tap per hypernet readout head feeding a ``PosLinear``
    weight, carrying that head's readout bias ``b_h`` as the gradient parameter
    and its emitted pre-positivity tensor as the iterate. The head set comes
    from the wrapper's own ``_pos_emitted`` when it is available, so the
    selection is self-checking rather than a ``Wzs.{k>=1}.weight`` naming
    convention that happens to coincide for CP-Flow's ICNN2 and would silently
    tap the wrong channel for a different ICNN class.

    For direct softplus: one tap per ``PosLinear.weight``, the same channel
    without the lift, namely the raw ``W-tilde`` that ``PosLinear.forward``
    pushes through softplus.

    Both are enumerated in block order and, within a block, in the module order
    of the target, so the concatenated rows line up across constructions as well
    as within one.
    """
    taps: List[BiasChannelTap] = []
    capped_blocks = 0
    for f in flow.flows:
        if not isinstance(f, DeepConvexFlow):
            continue
        icnn = getattr(f, "icnn", None)
        if icnn is None:
            continue
        if _gain_cap_active(icnn):
            capped_blocks += 1
        hn = getattr(icnn, "hypernet", None)
        if isinstance(hn, HyperNetwork):
            if len(hn.weight_predictors) == 0:
                raise NotImplementedError(
                    "cross-cov taps are not implemented for "
                    "share_readouts_by_shape=True: the readout bias is "
                    "shared across every target tensor of a given "
                    "width, so there is no per-head bias channel to "
                    "align the emission against."
                )
            # ``_pos_emitted`` is the wrapper's own record of which emitted
            # tensors carry the positivity constraint; the ICNN2 name pattern
            # is the fallback when the wrapper does not publish one.
            declared = getattr(icnn, "_pos_emitted", None)
            if declared:
                wanted = set(declared)
            else:
                wanted = {
                    n for n in hn._param_names
                    if n.startswith("Wzs.")
                    and n.endswith(".weight")
                    and not n.endswith("Wzs.0.weight")
                }
            missing = wanted - set(hn._param_names)
            if missing:
                raise RuntimeError(
                    "the block declares positivity-constrained emitted "
                    f"tensors the hypernet does not emit: {sorted(missing)}"
                )
            for name, bias in hn.positivity_readout_heads(names=wanted):
                taps.append(BiasChannelTap(
                    hypernet=hn,
                    emitted_name=name,
                    grad_param=bias,
                    kind="emitted",
                ))
            continue
        for name, mod in icnn.named_modules():
            if isinstance(mod, PosLinear):
                taps.append(BiasChannelTap(
                    hypernet=None,
                    emitted_name=f"{name}.weight",
                    grad_param=mod.weight,
                    kind="parameter",
                ))
    if capped_blocks:
        print(
            f"[cross-cov] WARNING: a gain cap is active on "
            f"{capped_blocks} block(s). The cap rescales the tensor "
            "AFTER the point this tap reads, so the recorded iterate is "
            "the raw pre-cap value. The shipped HEPMASS runs use "
            "gain_cap=0, where this is vacuous.",
            flush=True,
        )
    frozen = [t.emitted_name for t in taps if not t.grad_param.requires_grad]
    if frozen:
        # A frozen channel cannot supply the gradient half of the pair; the
        # usual cause is tapping a hypernet's frozen ICNN template instead of
        # its emitter.
        raise RuntimeError(
            "cross-cov taps landed on parameters that do not require "
            f"grad: {frozen[:4]}{' ...' if len(frozen) > 4 else ''}"
        )
    return taps


def tap_iterate_source(taps: List[BiasChannelTap]) -> str:
    """Provenance string naming what filled the accumulator's left slot."""
    kinds = sorted({t.kind for t in taps})
    label = {
        "emitted": "hypernet_emitted_pre_positivity",
        "parameter": "direct_pos_weight_parameter",
    }
    return "+".join(label.get(k, k) for k in kinds) if kinds else "none"


def flat_bias_dim(taps: List[BiasChannelTap]) -> int:
    return int(sum(t.width for t in taps))


def hypernets_of(flow: SequentialFlow) -> List[HyperNetwork]:
    """Every :class:`HyperNetwork` inside a lifted flow, in block order.

    Empty for the ``direct`` backend, so the trainer's emission-decay schedule
    call is a no-op there.
    """
    out: List[HyperNetwork] = []
    for f in flow.flows:
        if not isinstance(f, DeepConvexFlow):
            continue
        hn = getattr(getattr(f, "icnn", None), "hypernet", None)
        if isinstance(hn, HyperNetwork):
            out.append(hn)
    return out


@contextmanager
def _frozen_emission_ema(hypernets: List[HyperNetwork]) -> Iterator[None]:
    """Run a block without letting it move the emission-decay EMA.

    Delegates to each hypernet's :meth:`frozen_pooled_ema` hook. It must be
    entered per probe, not once around the loop: the EMA updates on every
    forward, so a single wrapper would let probe ``b``'s emission depend on
    probes ``1..b-1`` and the probes would stop being independent. A no-op
    unless ``emission_decay_frac > 0``.
    """
    with ExitStack() as stack:
        for hn in hypernets:
            stack.enter_context(hn.frozen_pooled_ema())
        yield


@contextmanager
def _observer_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Fork the global torch RNG so the probes do not perturb training.

    ``forward_transform_stochastic`` draws Rademacher vectors from the global
    torch generator on every ``logp`` call, so without the fork a probe forward
    would consume the same stream the training loop draws from and would change
    the trajectory being measured.
    """
    devices: List[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [
            device.index if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        yield


@contextmanager
def _restored_mode(flow: torch.nn.Module, train: bool) -> Iterator[None]:
    """Put ``flow`` in a mode for a block and put it back afterwards."""
    was_training = bool(flow.training)
    flow.train(train)
    try:
        yield
    finally:
        flow.train(was_training)


@contextmanager
def _probe_logdet(flow: SequentialFlow, exact: bool) -> Iterator[str]:
    """Optionally force the exact log-det for the probe forwards.

    The stochastic Lanczos/CG estimator's noise is independent of the probe
    batch, so it does not bias ``Sigma_slack``, but it does inflate the
    estimate's variance. Which one ran is recorded in ``probe_protocol``.
    """
    blocks = [f for f in flow.flows if isinstance(f, DeepConvexFlow)]
    if not exact:
        yield "stochastic_logdet"
        return
    saved = [getattr(f, "force_bruteforce", False) for f in blocks]
    for f in blocks:
        f.force_bruteforce = True
    try:
        yield "exact_logdet"
    finally:
        for f, was in zip(blocks, saved):
            f.force_bruteforce = was


def probe_seed(seed: int, iter_idx: int) -> int:
    """Deterministic per-snapshot seed, independent of the train stream."""
    return int((int(seed) * 1_000_003 + int(iter_idx) * 7_919) % (2 ** 31))


def make_probe_rng(seed: int) -> np.random.Generator:
    """The observer's own numpy generator.

    Constructed once per run from ``cfg.seed`` and never shared with
    ``_make_inf_batch_iter``: drawing probe batches from the training generator
    would advance it and change the training batches.
    """
    return np.random.default_rng([0xC05C0, int(seed)])


def record_cross_cov_snapshot(
    flow: SequentialFlow,
    *,
    accumulator: CrossCovAccumulator,
    taps: List[BiasChannelTap],
    train_data: np.ndarray,
    batch_size: int,
    n_probes: int,
    iter_idx: int,
    device: torch.device,
    probe_rng: np.random.Generator,
    seed: int = 0,
    exact_logdet: bool = False,
    max_redraws: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    r"""Capture one cross-covariance snapshot at a fixed iterate.

    Draws ``n_probes`` fresh batches from ``train_data``. For each batch it runs
    one forward and reads both halves of the pair off that single pass:

    * the iterate row --- each tap's pre-positivity value, that is the emitted
      ``Theta_E h_phi(X) + b_h`` captured through
      :meth:`HyperNetwork.capture_pre_positivity`, or the stored ``W-tilde``
      for direct softplus;
    * the gradient row --- ``d(NLL)/d(tap.grad_param)``.

    Both are flattened and concatenated over taps in the same order, so entry
    ``j`` of one row names the same scalar channel as entry ``j`` of the other,
    and the per-tap widths are asserted rather than assumed. The rows fill two
    preallocated ``(n_probes, d_bias)`` blocks and are handed to the
    accumulator, iterate first, which centers them and forms the
    cross-covariance.

    Every construction runs this same code path, with no sentinel branch: one
    whose iterate does not depend on the probe batch reports a floor-level
    matrix because that is what was measured.

    Non-finite probes are redrawn. The stochastic log-det estimator returns a
    non-finite value on occasional batches, and such a probe must not reach the
    accumulator, since NaN fails both ``> tol`` and ``<= tol`` and would trip
    the wrong guard. It is discarded and another drawn, up to ``max_redraws``
    (default ``2 * n_probes``).

    Returns the accumulator's scalar dict for this snapshot, augmented with
    ``n_probes_used`` and ``n_probes_discarded``, or ``None`` if ``iter_idx`` is
    not on the accumulator's schedule.

    The probe is a pure observer: its batches come from ``probe_rng`` and never
    the training generator, the global torch RNG is forked for the duration, the
    flow's train/eval mode is restored, ``torch.autograd.grad`` does not touch
    ``.grad``, no optimizer step is taken, and the emission-decay EMA is
    restored after every probe.
    """
    if not taps:
        raise ValueError(
            "record_cross_cov_snapshot called with no taps; the caller "
            "must not construct an accumulator for an arm that has no "
            "positivity channel (see train_cpflow)."
        )
    n_probes = int(n_probes)
    if n_probes < 2:
        raise ValueError(
            "the sample axis of the cross-covariance is the probe "
            f"batches; {n_probes} of them has no fluctuation to measure."
        )
    d_bias = flat_bias_dim(taps)
    budget = int(2 * n_probes if max_redraws is None else max_redraws)

    grad_params = [t.grad_param for t in taps]
    hypernets = [t.hypernet for t in taps if t.hypernet is not None]
    uniq_hypernets = list({id(h): h for h in hypernets}.values())

    # Preallocated rather than a list of rows plus torch.stack, which would
    # copy both blocks again: at HEPMASS geometry that is about 95 MB of
    # avoidable transient per snapshot.
    iterate_block = torch.empty((n_probes, d_bias), device=device)
    grad_block = torch.empty((n_probes, d_bias), device=device)

    filled = 0
    discarded = 0
    with _restored_mode(flow, train=True), \
            _observer_rng(device, probe_seed(seed, iter_idx)), \
            _probe_logdet(flow, exact_logdet) as protocol:
        while filled < n_probes:
            idx = probe_rng.choice(
                train_data.shape[0], size=int(batch_size), replace=False,
            )
            x = torch.from_numpy(train_data[idx]).float().to(device)
            with ExitStack() as stack:
                stack.enter_context(_frozen_emission_ema(uniq_hypernets))
                stores = {
                    id(hn): stack.enter_context(hn.capture_pre_positivity())
                    for hn in uniq_hypernets
                }
                nll = -flow.logp(x).mean()
                for hn in uniq_hypernets:
                    n_emit = int(hn._pre_positivity_emissions)
                    if n_emit != 1:
                        raise RuntimeError(
                            "expected exactly one emission per hypernet "
                            f"per probe forward; saw {n_emit}. The "
                            "captured iterate would not be the one whose "
                            "loss is differentiated below."
                        )
                if not torch.isfinite(nll):
                    discarded += 1
                    if discarded > budget:
                        break
                    continue
                iterate_parts = [t.iterate(stores) for t in taps]
            grads = torch.autograd.grad(
                nll, grad_params,
                retain_graph=False, create_graph=False,
                allow_unused=False,
            )
            grad_parts = [g.reshape(-1).detach() for g in grads]
            # Index alignment is the load-bearing assumption of the whole
            # measurement: entry j of the iterate row and entry j of the
            # gradient row must be the same scalar channel.
            for tap, jp, gp in zip(taps, iterate_parts, grad_parts):
                if jp.numel() != gp.numel():
                    raise RuntimeError(
                        f"tap {tap.emitted_name!r} ({tap.kind}) is "
                        f"misaligned: iterate has {jp.numel()} entries, "
                        f"gradient has {gp.numel()}."
                    )
            row_j = torch.cat(iterate_parts).to(device)
            row_g = torch.cat(grad_parts).to(device)
            if not (torch.isfinite(row_j).all() and
                    torch.isfinite(row_g).all()):
                # A finite loss can still carry a non-finite gradient.
                discarded += 1
                if discarded > budget:
                    break
                continue
            iterate_block[filled].copy_(row_j)
            grad_block[filled].copy_(row_g)
            filled += 1

    if filled < 2:
        raise RuntimeError(
            f"cross-cov snapshot at iter {iter_idx}: only {filled} of "
            f"{n_probes} probes were finite after {discarded} redraws. "
            "Two probes is the minimum for a fluctuation; the estimator "
            "will not be handed a single row."
        )
    if discarded:
        print(
            f"[cross-cov] iter {iter_idx}: {discarded} non-finite probe(s) "
            f"discarded; recorded with {filled}/{n_probes}.",
            flush=True,
        )

    accumulator.probe_protocol = (
        f"{protocol}; probe_batch={int(batch_size)}; "
        f"axis=probe_batches_at_fixed_iterate"
    )
    out = accumulator.record(
        iter_idx=iter_idx,
        iterate_batch=iterate_block[:filled],
        grad_batch=grad_block[:filled],
    )
    if out is not None:
        out = dict(out)
        out["n_probes_used"] = float(filled)
        out["n_probes_discarded"] = float(discarded)
    return out


__all__ = [
    "BiasChannelTap",
    "cross_cov_taps",
    "flat_bias_dim",
    "hypernets_of",
    "make_probe_rng",
    "probe_seed",
    "record_cross_cov_snapshot",
    "tap_iterate_source",
]
