"""Convolutional input-convex neural network for image-shaped inputs.

Drop-in replacement for :class:`lift.models.icnn.ICNN` when the input is
image-shaped (1-channel ``(28, 28)`` for MNIST). The energy ``E(x)`` is
convex in the flattened input by the same Amos recipe as the dense ICNN:
hidden-to-hidden conv kernels are non-negative, input-skip convs are
unconstrained, and a strong-convexity quadratic ``(c/2) ||x||^2`` is added.

The interface matches :class:`ICNN` so the rest of the EBM stack does not
change: ``forward(x)`` accepts a flat ``(B, input_size)`` tensor and returns
``(B, 1)`` energies, ``input_size`` is exposed as an attribute, the
positivity-constrained parameters are named so the hypernetwork can target
them (see :func:`conv_icnn_pos_param_names`), and ``clamp_weights()``
projects them onto the non-negative orthant under
``pos_constraint_mode='clamp'``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _folded_normal_pos_init(shape: tuple[int, ...]) -> torch.Tensor:
    fan_in = 1
    for s in shape[1:]:
        fan_in *= int(s)
    sigma = (math.pi / (2.0 * fan_in)) ** 0.5
    return torch.abs(torch.randn(*shape) * sigma)


class ConvICNN(nn.Module):
    """Conv ICNN with the standard Amos non-negativity recipe.

    Architecture (defaults; image_shape=(1, 28, 28), hidden=32, nlayers=3):

      x_flat ----+--------------+--------------+----- skip_lin (linear, free)
                 |              |              |
                 v              v              v
        first_conv          x_skip_conv     x_skip_conv
        (free) -> ReLU
                 |              |
                 v              v
            +-----------------------+
            | z_conv (positive) +    |
            | x_skip_conv (free) ->  |
            | ReLU                   |
            +-----------------------+
                  ... (nlayers-1 such blocks) ...
                              |
                              v
                          GAP -> Linear (positive) -> scalar
                                  +
                          (c/2) ||x_flat||^2 (strong convexity)
    """

    def __init__(
        self,
        input_size: int,
        *,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        hidden_channels: int = 32,
        nlayers: int = 3,
        kernel_size: int = 3,
        strong_convexity: float = 0.1,
        pos_constraint_mode: str = "clamp",
    ) -> None:
        super().__init__()
        if pos_constraint_mode not in ("clamp", "softplus"):
            raise ValueError(
                f"unknown pos_constraint_mode: {pos_constraint_mode}",
            )
        if int(input_size) != int(math.prod(image_shape)):
            raise ValueError(
                f"input_size={input_size} must equal "
                f"prod(image_shape={image_shape})="
                f"{int(math.prod(image_shape))}",
            )
        self.input_size = int(input_size)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.hidden_channels = int(hidden_channels)
        self.nlayers = int(nlayers)
        self.kernel_size = int(kernel_size)
        self.strong_convexity = float(strong_convexity)
        self.pos_constraint_mode = pos_constraint_mode

        in_channels = int(image_shape[0])
        pad = int(kernel_size // 2)

        # First layer: free.
        self.first_conv = nn.Conv2d(
            in_channels, hidden_channels, kernel_size, padding=pad,
        )

        # Hidden-to-hidden (positive) and input-skip (free) convs.
        self.z_convs = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad)
            for _ in range(self.nlayers - 1)
        ])
        self.x_skip_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad,
                      bias=False)
            for _ in range(self.nlayers - 1)
        ])

        # Output head: positive linear + free linear skip from flat input.
        self.output_layer = nn.Linear(hidden_channels, 1, bias=False)
        self.output_x_layer = nn.Linear(self.input_size, 1, bias=False)

        # Tag positivity-required parameters for ``HyperNetwork``'s drop-in
        # auto-detect. The tagged set mirrors
        # :func:`conv_icnn_pos_param_names`.
        for layer in self.z_convs:
            layer.weight._pos_required = True
        self.output_layer.weight._pos_required = True

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.z_convs:
            shape = tuple(layer.weight.shape)
            if self.pos_constraint_mode == "softplus":
                pos = _folded_normal_pos_init(shape)
                layer.weight.data = torch.log(
                    torch.expm1(pos.clamp(min=1e-6)),
                )
            else:
                layer.weight.data = _folded_normal_pos_init(shape)
            layer.bias.data.zero_()

        if self.pos_constraint_mode == "softplus":
            shape = tuple(self.output_layer.weight.shape)
            pos = _folded_normal_pos_init(shape)
            self.output_layer.weight.data = torch.log(
                torch.expm1(pos.clamp(min=1e-6)),
            )
        else:
            self.output_layer.weight.data = _folded_normal_pos_init(
                tuple(self.output_layer.weight.shape),
            )

        nn.init.xavier_uniform_(self.first_conv.weight)
        self.first_conv.bias.data.zero_()
        for layer in self.x_skip_convs:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_x_layer.weight)

    def _z_weight(self, i: int) -> torch.Tensor:
        raw = self.z_convs[i].weight
        if self.pos_constraint_mode == "clamp":
            return raw
        return F.softplus(raw)

    def _out_weight(self) -> torch.Tensor:
        raw = self.output_layer.weight
        if self.pos_constraint_mode == "clamp":
            return raw
        return F.softplus(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x
        if x.dim() == 2:
            img = x.view(-1, *self.image_shape)
        elif x.dim() == 4:
            img = x
            x_flat = img.flatten(start_dim=1)
        else:
            raise ValueError(f"x must be 2D or 4D; got {tuple(x.shape)}")
        z = F.relu(self.first_conv(img))
        for i, x_skip in enumerate(self.x_skip_convs):
            W = self._z_weight(i)
            z_conv = F.conv2d(
                z, W, self.z_convs[i].bias,
                stride=1,
                padding=int(self.kernel_size // 2),
            )
            z = F.relu(z_conv + x_skip(img))
        # Global average pool -> (B, hidden_channels)
        z_pool = z.mean(dim=(-2, -1))
        Wout = self._out_weight()
        energy = F.linear(z_pool, Wout, None) + self.output_x_layer(x_flat)
        if self.strong_convexity > 0.0:
            energy = energy + 0.5 * self.strong_convexity * x_flat.pow(2).sum(
                dim=-1, keepdim=True,
            )
        return energy

    def clamp_weights(self) -> None:
        if self.pos_constraint_mode != "clamp":
            return
        for layer in self.z_convs:
            layer.weight.data.clamp_(min=0.0)
        self.output_layer.weight.data.clamp_(min=0.0)


def conv_icnn_pos_param_names(nlayers: int, prefix: str = "") -> set[str]:
    """Canonical positivity-tagged names for a single :class:`ConvICNN`.

    Mirrors :func:`lift.models.hypernet.icnn_pos_param_names` but for
    the conv variant: the constrained tensors are
    ``z_convs.{i}.weight`` and ``output_layer.weight``. Biases,
    ``first_conv.weight``, ``x_skip_convs``, and ``output_x_layer``
    are unconstrained.
    """
    names = {f"{prefix}z_convs.{i}.weight" for i in range(int(nlayers) - 1)}
    names.add(f"{prefix}output_layer.weight")
    return names
