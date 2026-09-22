"""Hypernetwork that predicts the weights of a target network.

Vendored and lightly extended from torchhyper (Luqi group, Rice 2024).
The permutation-invariant variant maps a batch of input samples to a flat
representation through a per-sample MLP and a mean-pool (DeepSets), then
through a stack of independent linear readouts, one per target named
parameter. The same architecture appears in PLE (Mayer, Luzi, Siahkoohi,
Johnson, Baraniuk, *Improving Fairness and Mitigating MADness in Generative
Models*, arXiv:2405.13977), which reparameterizes GMM training without an
explicit EM step.

Extensions on top of vanilla torchhyper:

* ``pos_param_names``: readouts whose target name is in this set pass through
  a positivity reparameterization before the reshape, so the predicted tensor
  is non-negative element-wise. ``pos_constraint_mode`` selects the map:

    - ``'softplus'`` (default): ``W = softplus(u)``. Smooth, monotone,
      Lipschitz; the bias init at ``-2`` matches the folded-normal scale at
      fan-in 64. Recipe from PLE and Amos 2017.
    - ``'square'``: ``W = u**2``. Smooth, with ``W'(0) = 0``, so escape from
      zero is slow and ``W'(u)`` is unbounded on either side.
    - ``'qform'``: ``W = sum_r u_r**2`` over a width-``r`` virtual readout, so
      each entry of ``W`` is the squared norm of an ``r``-vector and
      positivity is structural. Default ``r = 4``.
    - ``'cone'``: ``W = clamp(u, min=0)``. Hard cone projection; the
      subgradient on the negative half-space is exactly zero.

* ``activation_noise_std``: Gaussian noise on the pooled hidden activations
  during training, scaled by ``set_noise_scale(scale)``. The trainer anneals
  the scale from 1 to 0 over the first half of training, which makes the early
  phase Langevin-like in phi-space.

* ``summary_kind='conv'``: replaces the per-sample MLP with the
  :class:`ConvSummaryEncoder` stack so the conditioning batch can be
  image-shaped ``(n, C, H, W)``. The mean-pool still follows the per-sample
  encoder, so permutation invariance over the conditioning batch is preserved
  by construction, and everything from the pooled summary onward is shared
  with the MLP path.

* ``emitter_kind='conv'``: replaces the per-tensor dense readout with a
  :class:`lift.models.conv_emitter.ConvTransposeEmitter`, a transposed
  convolution stack that upsamples the pooled summary to the target tensor's
  shape and carries the same per-entry slack. A dense readout stores
  ``d_h + 1`` parameters per emitted entry, which is prohibitive at image
  width. Everything upstream of the readout and everything downstream of it is
  shared with the dense path. Off by default.

* ``emission_decay_*``: a schedule that anneals the batch-dependent component
  of the emission toward its running mean late in training. The emission
  ``W = psi(Theta_E h_phi(X) + b_h)`` fluctuates from step to step because
  ``X`` is a fresh minibatch; that fluctuation is the ``sigma_Jac`` channel,
  and nothing otherwise quiets it, so the iterate never settles. Writing
  ``zbar`` for a running mean of the pooled summary, the schedule replaces
  ``z`` by ``zbar + gamma(t) (z - zbar)`` with ``gamma`` falling from ``1`` to
  ``emission_decay_floor`` over a configurable window. At the floor the
  emission is a deterministic function of ``phi`` alone, ``sigma_Jac -> 0``,
  and the model is the code-conditioned limit with the code pinned at the
  training-average summary. Off by default (``emission_decay_frac = 0``), and
  the forward pass is then the pre-schedule one exactly.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lift.models.conv_emitter import (
    DEFAULT_BASE_CHANNELS, DEFAULT_MIN_CHANNELS, DEFAULT_SEED_DIM_MAX,
    DEFAULT_SEED_MAX, DEFAULT_STRIDE, ConvTransposeEmitter,
    emitter_param_count, grid_for_shape,
)
from lift.models.conv_summary import ConvSummaryEncoder


_POS_MODES = ("softplus", "square", "qform", "cone")
_SUMMARY_KINDS = ("mlp", "conv", "moments")
_EMITTER_KINDS = ("dense", "conv")


def infer_pos_param_names(module: nn.Module) -> Set[str]:
    """Auto-detect positivity-required parameter names on ``module``.

    Walks ``module.named_parameters()`` and returns the dotted names of
    every :class:`torch.nn.Parameter` carrying the ``_pos_required``
    attribute flag (set to a truthy value). The canonical
    :class:`lift.models.icnn.ICNN` and
    :class:`lift.models.conv_icnn.ConvICNN` tag their constrained
    weights at construction so this walker is a drop-in replacement for
    the manual :func:`icnn_pos_param_names` / :func:`conv_icnn_pos_param_names`
    helpers when ``module`` is one of those (or contains them, e.g. as
    components of a :class:`LogConcaveEBM`).

    For untagged modules, such as CP-Flow's vendored ``ICNN2`` whose
    ``PosLinear`` handles positivity internally, the walker returns an
    empty set.
    """
    names: Set[str] = set()
    for name, p in module.named_parameters():
        if getattr(p, "_pos_required", False):
            names.add(name)
    return names


class MomentFeatures(nn.Module):
    """Per-sample features ``[x, x * x]`` for the minimal lift.

    Mean-pooled over the conditioning batch these are the batch's first
    two raw moments, so the emitted latent weight is
    ``b + A [m1(X), m2(X)]``: a learnable slack plus a learnable linear
    map of the batch statistics, with no body. It keeps the four things
    the coupling needs, a slack, batch dependence, the loss batch, and
    entry before the positivity map, and drops everything else.
    Parameter-free by construction.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.input_size)
        return torch.cat([x, x * x], dim=-1)


class HyperNetwork(nn.Module):
    """Permutation-invariant hypernet predicting weights of ``downstream``.

    Args:
        input_size: per-sample input dimension. The hypernet input is
            treated as a set of points in ``R^{input_size}``.
        hidden_sizes: hidden sizes of the per-sample encoder MLP. Last
            entry is the dimension of the aggregated representation
            fed to the readouts.
        downstream_network: target network whose ``named_parameters()``
            shape and order define the readout heads.
        use_outer_net: if True, run an additional MLP on the
            mean-pooled representation before the readout heads
            (matches PLE Eq. 6's ``h_phi^{(2)}`` as a pre-readout MLP).
        pos_param_names: target-parameter names whose predicted tensor
            must be non-negative element-wise. The positivity recipe
            is selected by ``pos_constraint_mode``.
        pos_constraint_mode: one of ``softplus / square / qform /
            cone``. See module docstring for the math behind each.
        head_init_pos_bias: when a parameter is in ``pos_param_names``
            **and** ``pos_constraint_mode == 'softplus'``, initialize
            the corresponding linear-readout bias to this value so
            that ``softplus(bias)`` matches the magnitude of a
            folded-normal init for the target weights at ``t=0``. The
            default ``-2.0`` gives ``softplus(-2.0) ~ 0.13``, close to
            the folded-normal mean for fan-in 64. For ``square`` and
            ``cone`` modes, the bias init is overridden to ``0`` so
            that the post-positivity output starts near zero.
        qform_rank: virtual readout width for ``pos_constraint_mode ==
            'qform'``. Each output entry of a tagged readout is
            ``sum_{r=1..qform_rank} u_r**2``. Larger rank -> easier to
            represent large positive weights, but the readout has
            ``qform_rank`` times more parameters.
        activation_noise_std: when ``> 0`` and the module is in
            training mode, Gaussian noise of std
            ``activation_noise_std * noise_scale`` is added to the
            pooled hidden representation before the readouts.
            ``noise_scale`` is set externally via
            :meth:`set_noise_scale` (default ``1``); the trainer
            anneals it linearly 1 -> 0 over the first half of training.
        summary_kind: ``'mlp'`` (default) is the per-sample MLP encoder;
            ``'moments'`` is the parameter-free ``[x, x^2]`` map of
            :class:`MomentFeatures`, the minimal lift; ``'conv'`` is a
            per-sample :class:`ConvSummaryEncoder` for image-shaped
            conditioning batches and requires ``image_shape``. In conv
            mode only the last entry of ``hidden_sizes``, the summary
            dimension the readouts consume, is used.
        image_shape: required iff ``summary_kind == 'conv'``: the
            per-sample conditioning image shape ``(C, H, W)`` with
            ``prod(image_shape) == input_size``. A flat
            ``(n, input_size)`` conditioning batch is the C-contiguous
            raveling of ``(C, H, W)`` and reshapes to images
            internally; an image-shaped ``(n, C, H, W)`` batch passes
            through at any spatial extent (the encoder's global
            average pool absorbs ``H, W``), so only the channel count
            is pinned. Ignored for ``'mlp'``.
        conv_channels: stride-2 stage widths of the conv encoder;
            ``None`` -> the :class:`ConvSummaryEncoder` default.
            Ignored for ``'mlp'``.
        emission_decay_frac: length of the emission-decay window as a
            fraction of training. ``0`` (default) switches the whole
            mechanism off: no blend, no running mean, no state.
        emission_decay_end: where the window closes, as a fraction of
            training. The window is ``[end - frac, end]``; before it the
            gain is exactly ``1``, after it the gain is held at the
            floor. Default ``1.0`` anneals over the last
            ``emission_decay_frac`` of training.
        emission_decay_floor: the gain at and after the close. ``0``
            (default) is the deterministic limit exactly: the summary is
            frozen at its running mean, the emission stops depending on
            the conditioning batch, and only the readouts
            ``Theta_E, b_h`` keep training. A small positive floor keeps
            the summary encoder in the autograd graph.
        emitter_kind: ``'dense'`` (default) is one ``Linear(d_h, n_out)``
            per target tensor. ``'conv'`` is a
            :class:`~lift.models.conv_emitter.ConvTransposeEmitter` per
            target tensor of at least two dimensions, used wherever that
            head is cheaper than the dense readout it would replace;
            one-dimensional targets, the biases, and any target where the
            conv head would cost more keep the dense readout.
            :attr:`emitter_head_kinds` records what was chosen.
        emitter_stride, emitter_seed_max, emitter_base_channels,
            emitter_min_channels: the upsampling plan; see
            :func:`~lift.models.conv_emitter.plan_stages`. At the
            defaults a ``(3072, 3072)`` target is a ``(3, 3)`` seed and
            five stages, a ``(784, 784)`` target a ``(4, 4)`` seed and
            four.
        emitter_out_scale: fixed multiplier on the conv stack's output,
            applied before the slack. At the default ``1.0`` the
            emitter's initialization matches the dense readout's emission
            on both the entrywise and the across-batch spread.
        emitter_permute: the ``lift_conv_perm`` control. Applies a
            frozen random permutation of the output indices between the
            conv stack and the slack, which leaves the parameter count,
            the arithmetic and the emitted-value multiset identical and
            destroys only the correspondence between the stack's spatial
            neighborhoods and the weight matrix's. Requires
            ``emitter_permute_seed``.
        emitter_permute_seed: seed for that permutation, so the control
            is a function of the cell seed.
        emission_decay_ema: momentum of the running mean that stands in
            for the deterministic-limit summary. It is updated on every
            training forward while the window is open, and before it
            opens, so the mean is warm when the blend starts; it is
            initialized at the first pooled summary it sees, so it
            carries no zero-init bias; and it is pinned once the window
            closes, so the emission at the floor is exactly
            deterministic.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        downstream_network: nn.Module,
        use_outer_net: bool = False,
        pool_norm: bool = False,
        pool_scale: float = 0.0,
        pos_param_names: Optional[Iterable[str]] = None,
        pos_constraint_mode: str = "softplus",
        head_init_pos_bias: float = -2.0,
        qform_rank: int = 4,
        share_readouts_by_shape: bool = False,
        activation_noise_std: float = 0.0,
        summary_kind: str = "mlp",
        image_shape: Optional[tuple[int, int, int]] = None,
        conv_channels: Optional[List[int]] = None,
        emission_decay_frac: float = 0.0,
        emission_decay_end: float = 1.0,
        emission_decay_floor: float = 0.0,
        emission_decay_ema: float = 0.99,
        body_layernorm: bool = False,
        readout_fanin_scale: bool = False,
        emitter_kind: str = "dense",
        # Defaults come from the emitter module's own constants, not from
        # literals: a second copy would drift, and the guard would then price
        # one architecture while the constructor built another.
        emitter_stride: int = DEFAULT_STRIDE,
        emitter_seed_max: int = DEFAULT_SEED_MAX,
        emitter_base_channels: int = DEFAULT_BASE_CHANNELS,
        emitter_min_channels: int = DEFAULT_MIN_CHANNELS,
        emitter_seed_dim_max: int = DEFAULT_SEED_DIM_MAX,
        emitter_out_scale: float = 1.0,
        emitter_permute: bool = False,
        emitter_permute_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # ``readout_fanin_scale``: the muP output-layer rule for the readouts.
        # The latent weight theta~ = Theta_E h + b is a sum over the d_h pooled
        # features, so under Adam one step moves theta~ by about d_h * lr while
        # a directly parameterized latent weight moves by lr. With the flag the
        # emission is (Theta_E h) / d_h + b with Theta_E initialized N(0, 1),
        # so each Adam step moves theta~ at the shared step size and the
        # initial emission has spread about 1/sqrt(d_h). Off by default.
        self.readout_fanin_scale = bool(readout_fanin_scale)
        # ``body_layernorm``: a LayerNorm after every hidden Linear of the
        # per-sample MLP encoder, before its ReLU. Off by default.
        self.body_layernorm = bool(body_layernorm)
        if pos_constraint_mode not in _POS_MODES:
            raise ValueError(
                f"pos_constraint_mode {pos_constraint_mode!r} must be "
                f"one of {_POS_MODES}",
            )
        if int(qform_rank) < 1:
            raise ValueError(
                f"qform_rank must be >= 1, got {qform_rank}",
            )
        if summary_kind not in _SUMMARY_KINDS:
            raise ValueError(
                f"summary_kind {summary_kind!r} must be one of "
                f"{_SUMMARY_KINDS}",
            )
        if emitter_kind not in _EMITTER_KINDS:
            raise ValueError(
                f"emitter_kind {emitter_kind!r} must be one of "
                f"{_EMITTER_KINDS}",
            )
        self.emitter_kind = str(emitter_kind)
        if self.emitter_kind == "conv":
            if share_readouts_by_shape:
                raise ValueError(
                    "emitter_kind='conv' and share_readouts_by_shape are "
                    "two different answers to the same cost problem; pick "
                    "one. The conv emitter keeps a per-entry slack, which "
                    "shared readouts do not, so it is the one that "
                    "supports lift_init_match='elementwise'.",
                )
            if readout_fanin_scale:
                raise ValueError(
                    "readout_fanin_scale is implemented for dense readouts "
                    "only. Its factor is d_h because a dense readout is ONE "
                    "linear map whose output entry is a sum over the d_h "
                    "pooled features; a conv emitter's output entry is a "
                    "sum over c_in*(k_r/s_r)*(k_c/s_c) entries of the last "
                    "stage's input, which is itself a nonlinear function of "
                    "the summary through several stages, and its parameters "
                    "are shared across every emitted entry. The correct "
                    "normalisation is a per-stage derivation, not the "
                    "factor d_h, and it has not been done -- so the "
                    "combination is refused rather than silently scaled by "
                    "a number with nothing behind it.",
                )
        self.emitter_out_scale = float(emitter_out_scale)
        self.emitter_permute = bool(emitter_permute)
        self._emitter_plan_kwargs = dict(
            stride=int(emitter_stride), seed_max=int(emitter_seed_max),
            base_channels=int(emitter_base_channels),
            min_channels=int(emitter_min_channels),
            seed_dim_max=int(emitter_seed_dim_max),
        )
        self._emitter_permute_seed = emitter_permute_seed
        self.summary_kind = str(summary_kind)
        if self.summary_kind == "conv":
            if image_shape is None:
                raise ValueError(
                    "summary_kind='conv' requires image_shape "
                    "(e.g. (1, 28, 28))",
                )
            self.image_shape: Optional[tuple[int, int, int]] = tuple(
                int(v) for v in image_shape
            )
            if int(np.prod(self.image_shape)) != int(input_size):
                raise ValueError(
                    f"prod(image_shape={self.image_shape})="
                    f"{int(np.prod(self.image_shape))} must equal "
                    f"input_size={input_size}",
                )
        else:
            self.image_shape = None
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.use_outer_net = use_outer_net
        self.pos_constraint_mode = str(pos_constraint_mode)
        self.qform_rank = int(qform_rank)
        self.activation_noise_std = float(activation_noise_std)
        # Schedulable scale, set by trainer via set_noise_scale().
        self.register_buffer(
            "_noise_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        self.pool_norm = bool(pool_norm)
        # ``pool_scale`` pins the pooled summary's norm while leaving its
        # direction free to learn, and changes nothing else.
        self.pool_scale = float(pool_scale)
        # ---- emission-decay schedule (see the module docstring) -------
        # Validated here so a mistyped fraction cannot silently produce a run
        # with no decay, or with a decay that never ends.
        if not 0.0 <= float(emission_decay_frac) <= 1.0:
            raise ValueError(
                f"emission_decay_frac must be in [0, 1], got "
                f"{emission_decay_frac!r}",
            )
        if not 0.0 <= float(emission_decay_end) <= 1.0:
            raise ValueError(
                f"emission_decay_end must be in [0, 1], got "
                f"{emission_decay_end!r}",
            )
        if not 0.0 <= float(emission_decay_floor) <= 1.0:
            raise ValueError(
                f"emission_decay_floor must be in [0, 1], got "
                f"{emission_decay_floor!r}",
            )
        if not 0.0 < float(emission_decay_ema) < 1.0:
            raise ValueError(
                f"emission_decay_ema must be in (0, 1), got "
                f"{emission_decay_ema!r}",
            )
        if float(emission_decay_frac) > float(emission_decay_end):
            raise ValueError(
                f"emission_decay window [end - frac, end] must start at or "
                f"after t=0: frac={emission_decay_frac!r} > "
                f"end={emission_decay_end!r}",
            )
        self.emission_decay_frac = float(emission_decay_frac)
        self.emission_decay_end = float(emission_decay_end)
        self.emission_decay_floor = float(emission_decay_floor)
        self.emission_decay_ema = float(emission_decay_ema)
        # State: the gain gamma(t) and the running mean it anneals toward.
        # With the mechanism off these are plain attributes, so no extra buffer
        # is registered and the state dict stays identical to the pre-schedule
        # one. With it on both are persistent buffers, because a checkpoint
        # reloaded outside the trainer would otherwise evaluate at full
        # emission strength rather than at the annealed gain it was trained at.
        # Reading the gain back in the forward pass costs one device sync per
        # emission, which is why it sits behind the knob.
        if self.emission_decay_frac > 0.0:
            self.register_buffer(
                "_emission_gain", torch.tensor(1.0, dtype=torch.float32),
            )
            self.register_buffer(
                "_pooled_ema",
                torch.zeros(1, int(self.hidden_sizes[-1])),
            )
            self.register_buffer(
                "_pooled_ema_ready", torch.tensor(0.0, dtype=torch.float32),
            )
        else:
            self._emission_gain = 1.0
            self._pooled_ema = None
        # When the caller omits ``pos_param_names`` the tagged names are
        # auto-detected from ``downstream_network``. Modules that do not tag
        # their parameters, such as CP-Flow's vendored ``ICNN2`` whose
        # ``PosLinear`` reparameterization is internal, yield an empty set.
        if pos_param_names is None:
            self.pos_param_names: Set[str] = infer_pos_param_names(
                downstream_network,
            )
            self._pos_auto_inferred = True
        else:
            self.pos_param_names = set(pos_param_names)
            self._pos_auto_inferred = False
        self.target_named_param_sizes: Dict[str, torch.Size] = {
            k: v.size() for k, v in downstream_network.named_parameters()
        }

        missing = self.pos_param_names - set(self.target_named_param_sizes)
        if missing:
            raise ValueError(
                f"pos_param_names not in downstream named_parameters: {missing}"
            )

        if self.summary_kind == "conv":
            # Per-sample conv branch; everything after the mean-pool is shared
            # with the MLP path, which consumes hidden_sizes[-1] either way.
            self.inner_net: nn.Module = ConvSummaryEncoder(
                image_shape=self.image_shape,
                out_dim=self.hidden_sizes[-1],
                channels=conv_channels,
            )
        elif self.summary_kind == "moments":
            # The minimal lift: no per-sample network at all, so the code width
            # is pinned to the feature width.
            if int(self.hidden_sizes[-1]) != 2 * int(input_size):
                raise ValueError(
                    "summary_kind='moments' needs hidden_sizes[-1] == "
                    f"2 * input_size = {2 * int(input_size)}; got "
                    f"{self.hidden_sizes[-1]}",
                )
            self.inner_net = MomentFeatures(input_size)
        else:
            layers: List[nn.Module] = [
                nn.Linear(input_size, self.hidden_sizes[0]),
            ]
            if self.body_layernorm:
                layers.append(nn.LayerNorm(self.hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(self.hidden_sizes) - 1):
                layers.append(
                    nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
                )
                if self.body_layernorm:
                    layers.append(nn.LayerNorm(self.hidden_sizes[i + 1]))
                layers.append(nn.ReLU())
            self.inner_net = nn.Sequential(*layers)

        if use_outer_net:
            outer: List[nn.Module] = [
                nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
                nn.ReLU(),
            ]
            for _ in range(2):
                outer.append(
                    nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1])
                )
                outer.append(nn.ReLU())
            self.outer_net: Optional[nn.Sequential] = nn.Sequential(*outer)
        else:
            self.outer_net = None

        # Sorted alias for the resolved positivity-tagged names.
        self._pos_names: List[str] = sorted(self.pos_param_names)

        # Diagnostic tap; see :meth:`capture_pre_positivity`. ``None`` whenever
        # no measurement is in flight, and the emission arithmetic is the same
        # either way.
        self._pre_positivity_capture: Optional[Dict[str, torch.Tensor]] = None
        self._pre_positivity_emissions: int = 0

        # Order matters: weight_predictors aligns with target_named_param_sizes.
        self._param_names: List[str] = list(self.target_named_param_sizes.keys())
        # ``share_readouts_by_shape``: one dense readout per distinct output
        # width, plus a per-tensor square code matrix. A dense
        # ``Linear(h, n_out)`` stores ``n_out * h`` parameters for a map of
        # rank at most ``h``, so per-tensor copies of one shape buy memory,
        # not expressiveness; sharing leaves the emission rank unchanged.
        self.share_readouts_by_shape = bool(share_readouts_by_shape)
        h_last = self.hidden_sizes[-1]

        def _n_out_for(name: str) -> int:
            n = int(np.prod(self.target_named_param_sizes[name]))
            if (
                name in self.pos_param_names
                and self.pos_constraint_mode == "qform"
            ):
                n = n * self.qform_rank
            return n

        self.weight_predictors = nn.ModuleList()
        #: Which head kind was built for each target tensor.
        self.emitter_head_kinds: List[str] = []
        if self.emitter_kind == "conv":
            # One conv head per target tensor of at least two dimensions,
            # wherever it is cheaper than the dense readout it would replace.
            # A one-dimensional target has no second axis to upsample along and
            # keeps the dense readout. The rule is a comparison rather than a
            # threshold, so it cannot misfire on an unseen shape.
            for name in self._param_names:
                shape = tuple(self.target_named_param_sizes[name])
                n_out = _n_out_for(name)
                rank = (
                    self.qform_rank
                    if (name in self.pos_param_names
                        and self.pos_constraint_mode == "qform")
                    else 1
                )
                use_conv = len(shape) >= 2
                if use_conv:
                    rows, cols = grid_for_shape(shape, rank)
                    n_conv = emitter_param_count(
                        h_last, rows, cols, n_out,
                        **self._emitter_plan_kwargs,
                    )
                    use_conv = n_conv < n_out * (h_last + 1)
                if use_conv:
                    self.weight_predictors.append(
                        ConvTransposeEmitter(
                            h_last, shape, rank=rank,
                            out_scale=self.emitter_out_scale,
                            permute=self.emitter_permute,
                            permute_seed=(
                                None if not self.emitter_permute
                                else int(self._emitter_permute_seed)
                                + len(self.emitter_head_kinds)
                            ),
                            **self._emitter_plan_kwargs,
                        ),
                    )
                    self.emitter_head_kinds.append("conv")
                else:
                    self.weight_predictors.append(nn.Linear(h_last, n_out))
                    self.emitter_head_kinds.append("dense")
        elif not self.share_readouts_by_shape:
            for name in self._param_names:
                self.weight_predictors.append(
                    nn.Linear(h_last, _n_out_for(name)),
                )
            self.emitter_head_kinds = ["dense"] * len(self._param_names)
            if self.readout_fanin_scale:
                with torch.no_grad():
                    for layer in self.weight_predictors:
                        nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        elif self.readout_fanin_scale:
            raise ValueError(
                "readout_fanin_scale is implemented for dense readouts only",
            )
        else:
            self._shared_readouts = nn.ModuleDict()
            self._readout_key: List[str] = []
            self.tensor_codes = nn.ModuleList()
            for name in self._param_names:
                n_out = _n_out_for(name)
                key = "%d_%s" % (
                    n_out, "p" if name in self.pos_param_names else "u",
                )
                if key not in self._shared_readouts:
                    self._shared_readouts[key] = nn.Linear(h_last, n_out)
                self._readout_key.append(key)
                # Per-tensor code, which keeps tensors distinguishable under
                # the shared readout. Identity-initialized, so at step zero the
                # shared design matches the dense one in distribution.
                code = nn.Linear(h_last, h_last, bias=False)
                with torch.no_grad():
                    code.weight.copy_(torch.eye(h_last))
                self.tensor_codes.append(code)

        # Bias init for positivity-tagged readouts. In shared mode the bias
        # lives on the shared readout; keys carry the positivity tag, so a
        # shared readout is always all-tagged or all-untagged.
        with torch.no_grad():
            if self.share_readouts_by_shape:
                _pairs = [
                    (n, self._shared_readouts[k])
                    for n, k in zip(self._param_names, self._readout_key)
                ]
            else:
                _pairs = list(zip(self._param_names, self.weight_predictors))
            for name, layer in _pairs:
                if name not in self.pos_param_names:
                    continue
                if self.pos_constraint_mode == "softplus":
                    layer.bias.fill_(float(head_init_pos_bias))
                else:
                    # square, qform, cone: a zero bias keeps the
                    # post-positivity output near zero at init.
                    layer.bias.zero_()
                    if self.pos_constraint_mode in ("square", "qform"):
                        # W = u^2 has E[W] ~ Var(u), so halving the default
                        # Xavier weight std puts the initial post-positivity
                        # outputs on the scale of the softplus map. A conv head
                        # has no single ``weight``; halving its final stage's
                        # kernel halves the whole emission, which is the same.
                        if isinstance(layer, ConvTransposeEmitter):
                            layer.halve_emission_scale()
                        else:
                            layer.weight.data.mul_(0.5)

    # ------------------------------------------------------------------ noise
    def set_noise_scale(self, scale: float) -> None:
        """Set the per-iter scale on ``activation_noise_std``.

        The trainer calls this every outer iteration with
        ``scale = max(0, 1 - 2 * t_frac)``, a linear anneal to zero at
        half-training. The call is a no-op when
        ``activation_noise_std == 0``, and the noise is skipped outside
        training mode.
        """
        self._noise_scale.fill_(float(scale))

    # ------------------------------------------------------- emission decay
    def set_train_progress(self, t_frac: float) -> None:
        """Tell the module how far training has run, ``t_frac`` in [0, 1].

        Drives the emission-decay gain ``gamma``: exactly ``1`` until
        ``t_frac`` reaches ``emission_decay_end - emission_decay_frac``,
        then linear to ``emission_decay_floor`` at
        ``emission_decay_end``, then held at the floor. A no-op when
        ``emission_decay_frac == 0`` (the default), so every driver may
        call it unconditionally.
        """
        if self.emission_decay_frac <= 0.0:
            return
        t = min(max(float(t_frac), 0.0), 1.0)
        start = self.emission_decay_end - self.emission_decay_frac
        u = (t - start) / self.emission_decay_frac
        u = min(max(u, 0.0), 1.0)
        self._emission_gain.fill_(
            1.0 - (1.0 - self.emission_decay_floor) * u,
        )

    @property
    def emission_gain(self) -> float:
        """Current emission-decay gain ``gamma`` (``1`` = full strength)."""
        return float(self._emission_gain)

    def _decay_pooled(self, z: torch.Tensor) -> torch.Tensor:
        """Blend the pooled summary toward its running mean.

        The running mean is a detached statistic and is pinned once the
        window closes, so at gain ``0`` the pooled summary is constant:
        no gradient reaches the DeepSets encoder and the emission stops
        depending on the conditioning batch, the code-conditioned limit,
        with the readouts still training.
        """
        gain = float(self._emission_gain)
        ready = float(self._pooled_ema_ready) > 0.0
        if self.training and gain > self.emission_decay_floor:
            # The mean tracks the summary while the window is open and is
            # pinned at the close. Without the pin the emission at the floor
            # keeps a ``1 - emission_decay_ema`` dependence on the current
            # batch, a residual sigma_Jac that never goes away.
            with torch.no_grad():
                zd = z.detach()
                if not ready:
                    # Seed the mean at the summary rather than at zero, so the
                    # blend carries no warm-up bias.
                    self._pooled_ema.copy_(zd)
                    self._pooled_ema_ready.fill_(1.0)
                    ready = True
                else:
                    m = self.emission_decay_ema
                    self._pooled_ema.mul_(m).add_(zd, alpha=1.0 - m)
        if gain >= 1.0 or not ready:
            # The window has not opened, or nothing has been pooled yet, as on
            # an eval forward before the first training step. Returning ``z``
            # untouched keeps a decay run identical to a no-decay run until the
            # window opens.
            return z
        zbar = self._pooled_ema
        z = zbar + gain * (z - zbar)
        if self.pool_scale > 0:
            # ``pool_scale`` pins the summary's norm and leaves its direction
            # free; the blend moves the direction, so the norm is re-pinned
            # rather than allowed to shrink with the decay.
            z = self.pool_scale * z / (z.norm() + 1e-8)
        return z

    # ------------------------------------------------------------------ pos
    def _apply_positivity(
        self, name: str, flat: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured positivity reparameterization."""
        if name not in self.pos_param_names:
            return flat
        if self.pos_constraint_mode == "softplus":
            return F.softplus(flat)
        if self.pos_constraint_mode == "square":
            return flat.pow(2)
        if self.pos_constraint_mode == "qform":
            # flat: (1, n_out * r) -> (1, n_out, r) -> sum_r u^2 -> (1, n_out)
            n_out = int(np.prod(self.target_named_param_sizes[name]))
            r = int(self.qform_rank)
            return flat.view(flat.shape[0], n_out, r).pow(2).sum(dim=-1)
        if self.pos_constraint_mode == "cone":
            return flat.clamp(min=0.0)
        raise RuntimeError(
            f"unhandled pos_constraint_mode: {self.pos_constraint_mode}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict ``downstream`` weights from a batch of inputs ``x``.

        ``x`` is treated as a set: rows are pooled by mean. Returns a
        ``{name: tensor}`` dict matching the target's
        ``named_parameters()``.
        """
        if self.summary_kind == "conv":
            # An image-shaped batch passes through unreshaped at any H, W,
            # since the encoder's global average pool absorbs the spatial
            # extent and the encoder checks the channel count. A flat
            # ``(n, input_size)`` batch is the C-contiguous raveling of
            # ``(C, H, W)`` and is reshaped to images. Anything else raises,
            # rather than silently re-slicing the conditioning set into a
            # different number of pseudo-images, which would corrupt the
            # emission and inflate the conditioning-batch size ``n``.
            if x.dim() == 4:
                z = self.inner_net(x)
            elif x.dim() == 2 and int(x.shape[1]) == int(self.input_size):
                z = self.inner_net(x.reshape(-1, *self.image_shape))
            else:
                raise ValueError(
                    f"conv summary expects (n, C, H, W) or "
                    f"(n, {self.input_size}); got {tuple(x.shape)}",
                )
        else:
            z = self.inner_net(x.reshape(-1, self.input_size))
        return self.emit_from_pooled(z.mean(dim=0, keepdim=True))

    def emit_from_pooled(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map a pooled summary ``(1, hidden_sizes[-1])`` to the weights.

        Everything downstream of the mean-pool lives here: norm pinning,
        the emission-decay blend, standardization, the outer net,
        activation noise, and the readouts with their positivity map. A
        caller that pools over something other than a single conditioning
        batch therefore shares one code path with :meth:`forward` instead
        of restating it.
        """
        if self.pool_scale > 0:
            z = self.pool_scale * z / (z.norm() + 1e-8)
        if self.emission_decay_frac > 0.0:
            z = self._decay_pooled(z)
        if self.pool_norm:
            # Standardize the pooled summary before the readouts. Otherwise
            # the readouts' input scale, and so how far one optimizer step
            # moves the emitted weights, inherits the scale of whatever
            # conditions the emission. Standardizing does not change the model
            # class: the emission is still a function of the conditioning set.
            # Off by default.
            z = (z - z.mean()) / (z.std() + 1e-5)
        if self.outer_net is not None:
            z = self.outer_net(z)
        # Optional Gaussian noise on the pooled activation, scaled by
        # ``_noise_scale`` from the trainer's anneal schedule.
        if (
            self.training
            and self.activation_noise_std > 0.0
            and float(self._noise_scale.item()) > 0.0
        ):
            z = z + torch.randn_like(z) * (
                self.activation_noise_std
                * float(self._noise_scale.item())
            )
        capture = self._pre_positivity_capture
        if capture is not None:
            self._pre_positivity_emissions += 1
        out: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self._param_names):
            flat = self._emit_flat(i, name, z)
            if capture is not None:
                # ``flat`` is exactly ``Theta_E h_phi(X) + b_h``, the emitted
                # latent weight before the positivity map ``psi``: softplus at
                # the readout for this package's ICNN, softplus inside
                # CP-Flow's ``PosLinear`` when ``pos_param_names`` is empty.
                # Its per-batch fluctuation is the left factor of
                # ``Sigma_slack``.
                capture[name] = flat.detach()
            flat = self._apply_positivity(name, flat)
            out[name] = flat.view(self.target_named_param_sizes[name])
        return out

    @contextmanager
    def capture_pre_positivity(self) -> Iterator[Dict[str, torch.Tensor]]:
        r"""Record each head's pre-positivity emission during a forward.

        Yields a dict that is filled, as a side effect of any
        :meth:`forward` / :meth:`emit_from_pooled` executed inside the
        ``with`` block, with ``{target_param_name: flat_emission}`` where
        ``flat_emission`` is the detached ``Theta_E h_phi(X) + b_h`` vector
        for that readout head, the tensor about to pass through the
        positivity map.

        The tap captures the emission of the same forward pass whose loss
        is then differentiated, so the recorded iterate and the recorded
        gradient belong to one probe batch by construction. The emission
        itself is unchanged: the captured tensor is a detached alias and
        the tap is not part of ``state_dict``.

        ``self._pre_positivity_emissions`` counts the emissions seen inside
        the block, so a caller that expects exactly one forward per
        hypernetwork can assert it instead of reading whichever emission
        landed last.
        """
        store: Dict[str, torch.Tensor] = {}
        prev = self._pre_positivity_capture
        prev_n = self._pre_positivity_emissions
        self._pre_positivity_capture = store
        self._pre_positivity_emissions = 0
        try:
            yield store
        finally:
            self._pre_positivity_capture = prev
            # Restore the count as well, or a nested capture zeroes the outer
            # block's tally and its assertion reads the inner block's count.
            self._pre_positivity_emissions = prev_n

    @contextmanager
    def frozen_pooled_ema(self) -> Iterator[None]:
        """Run a block without letting it advance the emission-decay EMA.

        :meth:`_decay_pooled` updates the running mean on every training
        forward, so a diagnostic that runs extra training forwards the
        optimizer never sees would otherwise advance the decay schedule.
        This is the hook the trainer calls instead of reaching into
        ``_pooled_ema`` and ``_pooled_ema_ready`` itself.

        A no-op unless ``emission_decay_frac > 0``. The restore is
        unconditional, so the block is safe to nest and safe under an
        exception.
        """
        if getattr(self, "_pooled_ema", None) is None:
            yield
            return
        ema = self._pooled_ema.clone()
        ready = self._pooled_ema_ready.clone()
        try:
            yield
        finally:
            self._pooled_ema.copy_(ema)
            self._pooled_ema_ready.copy_(ready)

    def positivity_readout_heads(
        self, names: Optional[Iterable[str]] = None,
    ) -> List[tuple]:
        """``[(target_name, readout_bias_parameter), ...]``, in emit order.

        The readout bias ``b_h`` of head ``i`` is the parameter whose loss
        gradient is the right factor of ``Sigma_slack``, and
        ``self._param_names[i]`` names the emitted tensor whose
        pre-positivity value is the left factor, so the two are index
        aligned here rather than by a naming convention at the call site.

        ``names`` restricts the result, with the order still following
        ``_param_names``; the default is every positivity-tagged target
        that has a bias, and heads without a bias are omitted.

        Raises under ``share_readouts_by_shape``, where the bias lives on a
        readout shared across every target of a given width and there is no
        per-head channel to align an emission against.
        """
        if self.share_readouts_by_shape:
            raise NotImplementedError(
                "positivity_readout_heads is not defined for "
                "share_readouts_by_shape=True: the readout bias is "
                "shared across every target tensor of a given width, "
                "so there is no per-head bias channel to align the "
                "emission against."
            )
        wanted = None if names is None else set(names)
        out: List[tuple] = []
        for name, layer in zip(self._param_names, self.weight_predictors):
            if wanted is None:
                if name not in self.pos_param_names:
                    continue
            elif name not in wanted:
                continue
            if getattr(layer, "bias", None) is None:
                # A head the caller named must produce a tap; skipping it would
                # let a cross-covariance measurement cover a subset of the
                # requested tensors and be reported as if it covered all.
                if wanted is not None:
                    raise RuntimeError(
                        f"readout head for {name!r} has no bias, so the "
                        "slack channel it was asked for does not exist; "
                        f"head type is {type(layer).__name__}",
                    )
                continue
            out.append((name, layer.bias))
        return out

    def _emit_flat(self, i: int, name: str, z: torch.Tensor) -> torch.Tensor:
        """Readout for target tensor ``i``, dense or shape-shared."""
        if not self.share_readouts_by_shape:
            layer = self.weight_predictors[i]
            if self.readout_fanin_scale:
                # (Theta_E h) / d_h + b; the slack b is not scaled.
                return F.linear(z, layer.weight, None) / float(z.shape[-1]) + layer.bias
            return layer(z)
        # Per-tensor code first, which keeps tensors distinguishable, then the
        # readout shared by every tensor of this output width.
        return self._shared_readouts[self._readout_key[i]](
            self.tensor_codes[i](z)
        )

    def icnn_pos_param_names(self, prefix: str = "") -> Set[str]:
        """Helper: returns the canonical pos-name set for a single ICNN.

        ``prefix`` lets the caller scope the names when the ICNN is
        nested (e.g. ``components.0.`` inside a ``LogConcaveEBM``).
        """
        raise NotImplementedError(
            "icnn_pos_param_names is a free helper; use "
            "lift.models.hypernet.icnn_pos_param_names(nlayers, prefix)"
        )


def icnn_pos_param_names(nlayers: int, prefix: str = "") -> Set[str]:
    """Canonical positivity-tagged names for a single ICNN target.

    Following Amos 2017, only the hidden-to-hidden weights and the
    positive output head must be non-negative; the biases, the input
    layer, and every input-skip layer are free.
    """
    names = {f"{prefix}z_layers.{i}.weight" for i in range(nlayers - 1)}
    names.add(f"{prefix}output_layer.weight")
    return names
