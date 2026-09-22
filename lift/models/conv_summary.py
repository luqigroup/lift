"""Per-sample convolutional encoder for image-conditioned hypernetworks.

Swaps the per-sample MLP branch of the DeepSets summary
:math:`h_\\phi` (see :class:`lift.models.hypernet.HyperNetwork`)
for a small conv stack, so the lift's emission
:math:`\\mathbf{W} = \\psi(\\Theta_E h_\\phi(\\mathbf{x}) +
\\mathbf{b}_h)` can condition on batches of images ``(n, C, H, W)``
instead of flat vectors. The mean-pool over the conditioning batch
stays OUTSIDE this module (in ``HyperNetwork.forward``), so
permutation invariance of the summary
:math:`\\bm{u} = n^{-1} \\sum_i h_\\phi(\\bm{x}_i)` is preserved by
construction -- only the per-sample branch changes.

Architecture: one stride-2 conv and ReLU stage per entry of ``channels``,
each halving the spatial extent, then a global average pool over the
remaining spatial dimensions, then a linear map to ``out_dim`` and a ReLU.
The pool makes the encoder resolution-agnostic, so the same widths run on
8x8 test images and 256x256 crops, with the length of ``channels``
controlling how much spatial extent survives to the pool.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


_DEFAULT_CHANNELS = (32, 64, 128)


class ConvSummaryEncoder(nn.Module):
    """Per-sample conv encoder: ``(n, C, H, W) -> (n, out_dim)``.

    Args:
        image_shape: per-sample conditioning image shape ``(C, H, W)``.
            Only the channel count is enforced at forward time; the
            global average pool absorbs any spatial extent.
        out_dim: output feature dimension. ``HyperNetwork`` passes its
            ``hidden_sizes[-1]`` here so everything downstream of the
            pool (outer net, noise, readouts) is untouched.
        channels: output channels of the stride-2 conv stages; one
            stage per entry, each halving the spatial extent.
            ``None`` -> ``(32, 64, 128)``.
        kernel_size: conv kernel size (odd, so each stage is
            same-padded).
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        out_dim: int,
        channels: tuple[int, ...] | list[int] | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = _DEFAULT_CHANNELS
        if len(image_shape) != 3:
            raise ValueError(
                f"image_shape must be (C, H, W); got {image_shape!r}",
            )
        if len(channels) < 1:
            raise ValueError(
                f"channels must be non-empty; got {channels!r}",
            )
        if int(kernel_size) % 2 != 1:
            raise ValueError(
                f"kernel_size must be odd for same-padding; "
                f"got {kernel_size}",
            )
        self.image_shape = tuple(int(v) for v in image_shape)
        self.out_dim = int(out_dim)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        pad = int(kernel_size // 2)

        layers: list[nn.Module] = []
        c_in = self.image_shape[0]
        for c_out in self.channels:
            layers.append(
                nn.Conv2d(c_in, c_out, kernel_size, stride=2, padding=pad),
            )
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(self.channels[-1], self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or int(x.shape[1]) != self.image_shape[0]:
            raise ValueError(
                f"expected (n, {self.image_shape[0]}, H, W); "
                f"got {tuple(x.shape)}",
            )
        z = self.conv(x)
        # Global average pool -> (n, channels[-1]); resolution-agnostic.
        z = z.mean(dim=(-2, -1))
        # Final ReLU for consistency with the MLP summary branch, which also
        # ends in ReLU; nothing downstream relies on the sign.
        return F.relu(self.head(z))
