"""Construction helpers for the two hypernetwork stacks.

:func:`build_hypernet_ebm` pairs a frozen log-concave EBM template with a
permutation-invariant DeepSets hypernetwork :math:`H_E` that predicts
every ICNN weight, tagging the constrained parameters for a positivity
readout. :func:`build_hypernet_sampler_flow` pairs a frozen
:class:`MultiSamplerFlow` with a sampler hypernetwork :math:`H_S`; its
coupling parameters are unconstrained, so it carries no positivity tags.
Both hypernetworks share one DeepSets shape and are meant to be fed the
same batch, so they consume the same summary :math:`\\bm{u}`.
"""

from __future__ import annotations

from typing import Optional

import torch

from lift.models import (
    HyperNetwork,
    LogConcaveEBM,
    MultiSamplerFlow,
    conv_icnn_pos_param_names,
    icnn_pos_param_names,
)


# Hypernet hidden sizes, held fixed across experiments.
HYPER_HIDDEN_SIZES = [64, 64, 96]


def build_hypernet_ebm(
    K: int, D: int = 1,
    hidden_dim: int = 128, nlayers: int = 3,
    strong_convexity: float = 0.02,
    activation: str = "softplus",
    pos_constraint_mode: str = "clamp",
    hyper_hidden_sizes: Optional[list[int]] = None,
    device: Optional[torch.device] = None,
    component_kind: str = "dense",
    image_shape: Optional[tuple[int, int, int]] = None,
    conv_hidden_channels: int = 32,
    conv_kernel_size: int = 3,
    hyper_pos_constraint_mode: str = "softplus",
    hyper_qform_rank: int = 4,
    hyper_activation_noise_std: float = 0.0,
    hyper_summary_kind: str = "mlp",
    hyper_conv_channels: Optional[list[int]] = None,
    hyper_extra_kwargs: Optional[dict] = None,
) -> tuple[LogConcaveEBM, HyperNetwork]:
    """Build the EBM stack ``H_E -> LogConcaveEBM``.

    The EBM template is frozen after construction, so only the
    hypernetwork is trained. The ICNN inter-layer and output-head weights
    get positivity-tagged readouts, which makes convexity structural
    rather than enforced after the fact.

    Args:
        component_kind: ``"dense"`` for the MLP ICNN, or ``"conv"`` for
            the ConvICNN on image-shaped inputs. Conv mode requires
            ``image_shape`` and switches the positivity tags to the
            Conv-ICNN parameter names.
        image_shape, conv_hidden_channels, conv_kernel_size: Conv-ICNN
            knobs, ignored for ``component_kind='dense'``.
        hyper_summary_kind: ``"mlp"`` for the DeepSets per-sample MLP
            summary; ``"moments"`` for the parameter-free ``[x, x^2]``
            summary, which requires ``hyper_hidden_sizes[-1] == 2 * D``;
            ``"conv"`` for a per-sample conv encoder
            (:class:`lift.models.ConvSummaryEncoder`) that conditions the
            emission on image batches ``(n, C, H, W)`` and requires
            ``image_shape``. This is independent of ``component_kind``, so
            an image-conditioned hypernet can drive either ICNN body.
        hyper_conv_channels: conv-encoder stage widths; ``None`` takes the
            encoder default. Ignored for ``hyper_summary_kind="mlp"``.
        hyper_extra_kwargs: further :class:`HyperNetwork` keyword
            arguments, such as ``pool_norm``, ``pool_scale``,
            ``use_outer_net``, ``head_init_pos_bias`` or
            ``body_layernorm``. ``None`` adds nothing.
    """
    hyper_hidden_sizes = hyper_hidden_sizes or HYPER_HIDDEN_SIZES
    ebm = LogConcaveEBM(
        K=K, D=D,
        hidden_dim=hidden_dim, nlayers=nlayers,
        activation=activation,
        strong_convexity=strong_convexity,
        pos_constraint_mode=pos_constraint_mode,
        component_kind=component_kind,
        image_shape=image_shape,
        conv_hidden_channels=conv_hidden_channels,
        conv_kernel_size=conv_kernel_size,
    )
    if device is not None:
        ebm = ebm.to(device)
    ebm.requires_grad_(False)

    pos_names: set[str] = set()
    pos_helper = (
        conv_icnn_pos_param_names if component_kind == "conv"
        else icnn_pos_param_names
    )
    for k in range(K):
        pos_names |= pos_helper(nlayers, prefix=f"components.{k}.")

    hyper = HyperNetwork(
        input_size=D,
        hidden_sizes=hyper_hidden_sizes,
        downstream_network=ebm,
        pos_param_names=pos_names,
        pos_constraint_mode=str(hyper_pos_constraint_mode),
        qform_rank=int(hyper_qform_rank),
        activation_noise_std=float(hyper_activation_noise_std),
        summary_kind=str(hyper_summary_kind),
        image_shape=image_shape,
        conv_channels=hyper_conv_channels,
        **(dict(hyper_extra_kwargs) if hyper_extra_kwargs else {}),
    )
    if device is not None:
        hyper = hyper.to(device)
    return ebm, hyper


def build_hypernet_sampler_flow(
    K: int, D: int = 1,
    D_aug: Optional[int] = None,
    n_hidden: int = 64,
    n_flow_layers: int = 4,
    depth: Optional[int] = None,
    n_mlp_layers: int = 3,
    hyper_input_size: Optional[int] = None,
    hyper_hidden_sizes: Optional[list[int]] = None,
    device: Optional[torch.device] = None,
) -> tuple[MultiSamplerFlow, HyperNetwork]:
    """Build the sampler stack ``H_S -> MultiSamplerFlow``.

    The flow template runs on :math:`\\mathbb{R}^{D_{\\text{aug}}}`, and
    HINT requires :math:`D_{\\text{aug}} \\ge 2`. When ``D_aug > D``, the
    auxiliary-Gaussian lift in ``augmentation.py`` is what keeps
    :math:`Z_{\\text{aug}} = Z`.
    """
    hyper_hidden_sizes = hyper_hidden_sizes or HYPER_HIDDEN_SIZES
    multi = MultiSamplerFlow(
        K=K, D=D, D_aug=D_aug, n_hidden=n_hidden,
        n_flow_layers=n_flow_layers, depth=depth,
        n_mlp_layers=n_mlp_layers,
    )
    if device is not None:
        multi = multi.to(device)
    multi.requires_grad_(False)

    hyper = HyperNetwork(
        input_size=int(hyper_input_size if hyper_input_size is not None else D),
        hidden_sizes=hyper_hidden_sizes,
        downstream_network=multi,
        pos_param_names=None,
    )
    if device is not None:
        hyper = hyper.to(device)
    return multi, hyper
