"""Log-concave energy-based model.

The density is ``q(x) = exp(-E(x)) / Z`` with ``E`` an ICNN, so ``q`` is
log-concave by construction. The location is encoded in the ICNN's input-
and output-skip linear maps, so no separate center parameter is needed.

On the lift path this module is only a template: its parameters stay frozen
and the hypernetwork emits every weight tensor each forward pass through
:func:`torch.func.functional_call` against ``components`` and ``log_pi``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from lift.models.conv_icnn import ConvICNN
from lift.models.icnn import ICNN


class LogConcaveEBM(nn.Module):

    def __init__(
        self,
        K: int,
        D: int,
        hidden_dim: int = 64,
        nlayers: int = 3,
        activation: str = "softplus",
        strong_convexity: float = 0.1,
        pos_constraint_mode: str = "clamp",
        component_kind: str = "dense",
        image_shape: Optional[tuple[int, int, int]] = None,
        conv_hidden_channels: int = 32,
        conv_kernel_size: int = 3,
    ) -> None:
        """
        Args:
            component_kind: ``"dense"`` (default) -> :class:`ICNN`,
                ``"conv"`` -> :class:`ConvICNN`. The conv variant
                expects ``image_shape`` (e.g. ``(1, 28, 28)``) so the
                input can be reshaped internally; ``D`` must equal
                ``prod(image_shape)``.
            image_shape: required iff ``component_kind == 'conv'``.
            conv_hidden_channels, conv_kernel_size: ConvICNN knobs.
        """
        super().__init__()
        self.K = K
        self.D = D
        self.pos_constraint_mode = pos_constraint_mode
        if component_kind not in ("dense", "conv"):
            raise ValueError(
                f"unknown component_kind {component_kind!r}; "
                "expected 'dense' or 'conv'",
            )
        self.component_kind = component_kind
        self.image_shape = image_shape

        if component_kind == "conv":
            if image_shape is None:
                raise ValueError(
                    "component_kind='conv' requires image_shape "
                    "(e.g. (1, 28, 28))",
                )
            from math import prod
            if int(prod(image_shape)) != int(D):
                raise ValueError(
                    f"prod(image_shape={image_shape})="
                    f"{int(prod(image_shape))} must equal D={D}",
                )
            self.components = nn.ModuleList([
                ConvICNN(
                    input_size=D,
                    image_shape=tuple(int(v) for v in image_shape),
                    hidden_channels=conv_hidden_channels,
                    nlayers=nlayers,
                    kernel_size=conv_kernel_size,
                    strong_convexity=strong_convexity,
                    pos_constraint_mode=pos_constraint_mode,
                )
                for _ in range(K)
            ])
        else:
            self.components = nn.ModuleList([
                ICNN(input_size=D, hidden_dim=hidden_dim, nlayers=nlayers,
                     activation=activation, strong_convexity=strong_convexity,
                     pos_constraint_mode=pos_constraint_mode)
                for _ in range(K)
            ])
        self.log_pi = nn.Parameter(torch.zeros(K))
