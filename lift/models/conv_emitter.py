r"""Convolutional weight emitter: the lift's readout at CIFAR width.

The lift emits a target's weights as
:math:`\mathbf{W} = \psi(\Theta_E h_\phi(\mathbf{X}) + \mathbf{b}_h)`.
A dense readout realizes :math:`\Theta_E` as one ``Linear(d_h, n_out)``
per target tensor, which costs ``d_h + 1`` parameters per emitted entry:
at the Hoedt-Klambauer CIFAR width that is 3.09e9 parameters, about 34 GiB
once Adam carries two moments. The body producing :math:`h_\phi` is about
1e4 parameters and does not scale with the target, so the cost is entirely
in the readout.

This module replaces :math:`\Theta_E h_\phi(\mathbf{X})` for one target
tensor by a convolution stack that upsamples the pooled summary to the
tensor's own shape, and keeps the slack :math:`\mathbf{b}_h` as a full
per-entry parameter:

    ``flat = out_scale * upsample(z)[perm] + bias``

The readout cost falls from ``n_out (d_h + 1)`` to
``n_out + O(channels^2 kernel^2 depth)``, a factor of 60.7 at CIFAR width
and 34.2 at MNIST width.

Three properties are load-bearing. The slack is per-entry, so
:math:`\partial\tilde\theta/\partial\mathbf{b}` is the identity and the
added curvature lives on all of :math:`\mathbb{R}^d`. Nothing inside the
stack carries a bias, so the slack is the only batch-independent additive
channel, which is the two-channel decomposition the cross-covariance is
defined on. The upsample is sub-pixel rather than a strided transposed
convolution: the same operator (Shi et al., CVPR 2016), evaluated in the
order that gives each output phase its own filter, so there is no uneven
kernel overlap and no residue-class artifact.

The initialization is derived rather than fitted. ``out_scale`` defaults to
``1.0``, the seed carries the dense readout's own law
``U(-1/sqrt(d_h), 1/sqrt(d_h))``, and every stage is variance-preserving
for its effective fan-in ``c_in (k_r/s_r)(k_c/s_c)``, with the ReLU gain on
the stages that have one.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


#: Defaults. ``stride`` and ``seed_max`` together fix the stage count:
#: five stages from a 3x3 seed at width 3072, four from a 4x4 seed at
#: width 784. See ``plan_stages``.
DEFAULT_STRIDE = 4
DEFAULT_SEED_MAX = 4
DEFAULT_BASE_CHANNELS = 64
DEFAULT_MIN_CHANNELS = 8

#: An axis no longer than this is NOT upsampled: its seed extent is the
#: axis itself. Without the cap a ``(100, 3072)`` output head would
#: generate 100 classes from 2 latent rows upsampled 64x, so classes
#: sharing a latent row would share their emission up to one kernel. The
#: cost is a larger seed map, ``d_h * c_0 * rows * w_0``.
DEFAULT_SEED_DIM_MAX = 128


class StageSpec(tuple):
    """``(c_in, c_out, stride_r, stride_c, k_r, k_c, pad_r, pad_c)``."""

    __slots__ = ()

    @property
    def c_in(self) -> int:
        return self[0]

    @property
    def c_out(self) -> int:
        return self[1]

    @property
    def fan_in_eff(self) -> float:
        """Input entries summed into one output entry.

        In the sub-pixel form every output entry is one output channel of a
        plain ``Conv2d``, so the fan-in is the ordinary ``c_in * k_r * k_c``:
        each output phase has its own filter and no output entry is a sum
        over several overlapping kernels. This is the denominator of the
        variance-preserving initialization.
        """
        return float(self[0]) * float(self[4]) * float(self[5])

    @property
    def upscale(self) -> Tuple[int, int]:
        return int(self[2]), int(self[3])


def _stages_for_axis(dim: int, stride: int, seed_max: int,
                    seed_dim_max: int) -> Tuple[int, int]:
    """``(n_upsampling_stages, seed_extent)`` for one axis.

    A short axis (``dim <= seed_dim_max``) is carried at full extent in
    the seed and never upsampled, so every index along it keeps its own
    latent row. See :data:`DEFAULT_SEED_DIM_MAX`.
    """
    if int(dim) <= int(seed_dim_max):
        return 0, int(dim)
    a = 0
    while math.ceil(dim / stride ** a) > seed_max:
        a += 1
    return a, int(math.ceil(dim / stride ** a))


def plan_stages(
    rows: int,
    cols: int,
    stride: int = DEFAULT_STRIDE,
    seed_max: int = DEFAULT_SEED_MAX,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    min_channels: int = DEFAULT_MIN_CHANNELS,
    seed_dim_max: int = DEFAULT_SEED_DIM_MAX,
) -> Tuple[int, int, List[StageSpec]]:
    """Plan the upsampling stack for a ``(rows, cols)`` target grid.

    The stage count is per axis: the smallest ``a`` with
    ``ceil(dim / stride**a) <= seed_max``. The stack runs
    ``max(a_rows, a_cols)`` stages, and an axis is upsampled in the LAST
    ``a`` of them, so the short axis of an anisotropic target (an
    ``(10, 3072)`` output head, say) stays short for as long as
    possible and the activations stay small.

    Returns ``(seed_h, seed_w, stages)``. The stack's output is
    ``(seed_h * stride**a_rows, seed_w * stride**a_cols)``, which is
    ``>= (rows, cols)``; the caller crops.
    """
    if int(rows) < 1 or int(cols) < 1:
        raise ValueError(f"rows and cols must be >= 1; got {rows}, {cols}")
    if int(stride) < 2 or int(stride) % 2 != 0:
        raise ValueError(
            f"stride must be an even integer >= 2 so that kernel = 2*stride "
            f"has padding stride//2 and even overlap; got {stride}",
        )
    if int(seed_max) < 1:
        raise ValueError(f"seed_max must be >= 1; got {seed_max}")
    if int(base_channels) < 1 or int(min_channels) < 1:
        raise ValueError(
            f"channel counts must be >= 1; got base={base_channels}, "
            f"min={min_channels}",
        )
    stride, seed_max = int(stride), int(seed_max)
    a_r, h0 = _stages_for_axis(int(rows), stride, seed_max, seed_dim_max)
    a_c, w0 = _stages_for_axis(int(cols), stride, seed_max, seed_dim_max)
    n_stages = max(a_r, a_c, 1)

    # The final stage's input channel count sets the rank of the emitted
    # perturbation inside one output tile: a stride-s last stage writes each
    # s-by-s tile as a combination of c_in fixed atoms, so c_in caps the tile
    # rank at min(c_in, s^2). Hence the channel floor is a parameter.
    chans = [int(base_channels)]
    for i in range(n_stages - 1):
        chans.append(max(int(base_channels) >> (i + 1), int(min_channels)))
    chans.append(1)

    stages: List[StageSpec] = []
    for i in range(n_stages):
        s_r = stride if i >= n_stages - a_r else 1
        s_c = stride if i >= n_stages - a_c else 1
        # Sub-pixel: a same-padded 3x3 convolution at the stage's input
        # resolution emitting ``c_out * s_r * s_c`` channels, rearranged into
        # an ``s_r`` by ``s_c`` block per input position. The kernel is the
        # convolution's own, not a function of the stride.
        stages.append(
            StageSpec((chans[i], chans[i + 1], s_r, s_c, 3, 3, 1, 1)),
        )
    return h0, w0, stages


def emitter_param_count(
    in_features: int,
    rows: int,
    cols: int,
    n_out: int,
    **plan_kwargs,
) -> int:
    """Parameters of one conv head, including its per-entry slack.

    Pure arithmetic on the shapes, so a builder can compare it against the
    dense readout it would replace, and against ``max_lift_params``, without
    constructing anything.
    """
    h0, w0, stages = plan_stages(rows, cols, **plan_kwargs)
    n = int(in_features) * stages[0].c_in * h0 * w0        # seed, bias-free
    for st in stages:
        s_r, s_c = st.upscale
        n += st.c_in * (st.c_out * s_r * s_c) * st[4] * st[5]
    return int(n + int(n_out))                             # + the slack


def grid_for_shape(
    shape: Sequence[int], rank: int = 1,
) -> Tuple[int, int]:
    """The ``(rows, cols)`` grid a target tensor is emitted on.

    ``rows`` is the leading dimension and ``cols`` the product of the
    rest, times ``rank`` for the ``qform`` positivity mode. The
    ``qform`` factor goes on the COLUMN axis because
    ``flat.view(n_out, rank)`` is row-major: flat index
    ``((i * cols + j) * rank + k)`` is exactly entry ``(i, j * rank + k)``
    of a ``(rows, cols * rank)`` grid, so no reordering is needed and
    the layout ``_apply_positivity`` expects is the one produced.
    """
    shape = tuple(int(v) for v in shape)
    if len(shape) >= 2:
        rows = shape[0]
        cols = 1
        for v in shape[1:]:
            cols *= v
    else:
        rows = 1
        cols = shape[0] if shape else 1
    return int(rows), int(cols * int(rank))


def _subpixel(y: torch.Tensor, c_out: int, s_r: int, s_c: int,
              ) -> torch.Tensor:
    """``(n, c_out*s_r*s_c, H, W) -> (n, c_out, H*s_r, W*s_c)``.

    ``nn.PixelShuffle`` takes one scalar factor and so cannot express an
    anisotropic stage, which every stage of an ``(10, 3072)`` output head is,
    since the class axis is not upsampled. This is the same rearrangement
    with a factor per axis.
    """
    if s_r == 1 and s_c == 1:
        return y
    n, _, h, w = y.shape
    y = y.view(n, c_out, s_r, s_c, h, w)
    y = y.permute(0, 1, 4, 2, 5, 3)
    return y.reshape(n, c_out, h * s_r, w * s_c)


class ConvTransposeEmitter(nn.Module):
    """One readout head: pooled summary ``(1, d_h)`` -> ``(1, n_out)``.

    The name refers to the operator, not the kernel call: the stages are
    transposed (upsampling) convolutions realized in their sub-pixel form.

    A drop-in for the ``nn.Linear(d_h, n_out)`` the dense path uses. It
    exposes the same ``bias`` parameter of shape ``(n_out,)``, which is the
    lift's slack and the channel every diagnostic attaches to, and it is
    held in the same ``weight_predictors`` ModuleList, so
    ``positivity_readout_heads``, the cross-covariance taps and
    ``match_lift_init`` reach it unchanged.

    It deliberately does not expose ``weight``. Code that reaches into a
    readout's ``weight`` (``component_reinit``, the CP-Flow rescalers) is
    undefined for this head, and an ``AttributeError`` naming the attribute
    is the right failure.

    Args:
        in_features: ``d_h``, the pooled summary dimension.
        shape: the target tensor's own shape, e.g. ``(3072, 3072)``.
        rank: ``qform_rank`` when the head is qform-tagged, else 1.
        stride, seed_max, base_channels, min_channels: the stack plan;
            see :func:`plan_stages`.
        out_scale: a fixed multiplier on the conv output, applied before
            the slack is added so the slack's Jacobian is the identity
            whatever it is set to. Defaults to ``1.0``.
        permute: when True, a frozen random permutation of the output
            indices is applied between the cropped stack output and the
            slack. This is the ``lift_conv_perm`` control: no extra
            parameters and the same emitted-value multiset at every step,
            with the correspondence between the stack's spatial
            neighborhoods and the weight matrix's ``(row, col)``
            neighborhoods destroyed.
        permute_seed: seed for that permutation. Required when ``permute``
            is True, so the control is a function of the cell seed rather
            than of the global generator.
    """

    def __init__(
        self,
        in_features: int,
        shape: Sequence[int],
        rank: int = 1,
        stride: int = DEFAULT_STRIDE,
        seed_max: int = DEFAULT_SEED_MAX,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        min_channels: int = DEFAULT_MIN_CHANNELS,
        seed_dim_max: int = DEFAULT_SEED_DIM_MAX,
        out_scale: float = 1.0,
        permute: bool = False,
        permute_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.target_shape = tuple(int(v) for v in shape)
        self.rank = int(rank)
        self.rows, self.cols = grid_for_shape(self.target_shape, self.rank)
        n_out = self.rows * self.cols
        self.out_features = int(n_out)
        # A persistent buffer rather than a Python float: the scale
        # multiplies the batch-dependent channel, so a snapshot reloaded
        # outside the trainer would otherwise silently emit at the default.
        self.register_buffer(
            "out_scale", torch.tensor(float(out_scale), dtype=torch.float32),
        )

        h0, w0, stages = plan_stages(
            self.rows, self.cols, stride=stride, seed_max=seed_max,
            base_channels=base_channels, min_channels=min_channels,
            seed_dim_max=seed_dim_max,
        )
        self.seed_h, self.seed_w = int(h0), int(w0)
        self.seed_channels = int(stages[0].c_in)
        self.stage_specs = list(stages)

        # Seed: the dense readout's own map, into a small image.
        self.seed = nn.Linear(
            self.in_features,
            self.seed_channels * self.seed_h * self.seed_w,
            bias=False,
        )
        with torch.no_grad():
            bound = 1.0 / math.sqrt(max(self.in_features, 1))
            self.seed.weight.uniform_(-bound, bound)

        stack: List[nn.Module] = []
        for i, st in enumerate(stages):
            s_r, s_c = st.upscale
            conv = nn.Conv2d(
                st.c_in, st.c_out * s_r * s_c, (st[4], st[5]),
                padding=(st[6], st[7]), bias=False,
            )
            # Variance-preserving for this stage's fan-in, with the ReLU
            # gain on every stage that is followed by one.
            gain_sq = 2.0 if i < len(stages) - 1 else 1.0
            with torch.no_grad():
                conv.weight.normal_(0.0, math.sqrt(gain_sq / st.fan_in_eff))
            stack.append(conv)
        self.stages = nn.ModuleList(stack)

        # The slack. Zero-initialized here; the caller's positivity-bias
        # convention (``head_init_pos_bias`` / ``match_lift_init``)
        # overwrites it exactly as it overwrites a Linear's bias.
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        self.permute = bool(permute)
        if self.permute:
            if permute_seed is None:
                raise ValueError(
                    "permute=True requires permute_seed so the control is a "
                    "function of the cell seed, not of the global generator",
                )
            g = torch.Generator(device="cpu").manual_seed(int(permute_seed))
            self.register_buffer(
                "perm", torch.randperm(self.out_features, generator=g),
            )
        else:
            self.perm = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features="
            f"{self.out_features}, target_shape={self.target_shape}, "
            f"seed=({self.seed_channels},{self.seed_h},{self.seed_w}), "
            f"stages={len(self.stages)}, out_scale={float(self.out_scale)}, "
            f"permute={self.permute}"
        )

    @torch.no_grad()
    def halve_emission_scale(self) -> None:
        """Halve the emitted perturbation, leaving the slack alone.

        The dense readout's ``square`` / ``qform`` initialization halves the
        readout weight so the post-positivity output starts at the magnitude
        the softplus reparameterization starts at. The analogue here is
        halving the final stage's kernel, which scales the whole emission by
        one half and touches nothing else.
        """
        self.stages[-1].weight.mul_(0.5)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(n, d_h) -> (n, n_out)``, the pre-positivity emission."""
        if z.dim() != 2 or int(z.shape[1]) != self.in_features:
            raise ValueError(
                f"conv emitter expects (n, {self.in_features}); "
                f"got {tuple(z.shape)}",
            )
        y = self.seed(z).view(
            z.shape[0], self.seed_channels, self.seed_h, self.seed_w,
        )
        last = len(self.stages) - 1
        for i, (conv, st) in enumerate(zip(self.stages, self.stage_specs)):
            y = _subpixel(conv(y), st.c_out, *st.upscale)
            if i < last:
                # No activation on the final stage: the emission must span
                # negatives, or psi would be applied on a cone.
                y = torch.relu(y)
        flat = y[:, 0, :self.rows, :self.cols].reshape(
            z.shape[0], self.out_features,
        )
        if float(self.out_scale) != 1.0:
            flat = flat * self.out_scale
        if self.perm is not None:
            flat = flat[:, self.perm]
        # The slack is added last and unscaled, so d theta~ / d bias is the
        # identity whatever out_scale and perm are.
        return flat + self.bias
