"""Inpainting physics, L2 fidelity, and PSNR for the Ehrhardt FFHQ line.

These are the only pieces of ``deepinv`` (pinned at 0.0.1) and
``scikit-image`` that the FFHQ pipeline of ``hsw43/icnn_primal_dual``
(commit ``bb0d0bd``, Zenodo DOI 10.5281/zenodo.17426033) touches, rewritten
here rather than pinned:

* :class:`InpaintingPhysics` for ``deepinv.physics.Inpainting``: the
  forward operator is an element-wise mask multiply, and since the mask is
  a 0/1 diagonal, the adjoint and pseudo-inverse are the same multiply.
  ``PDHG_ICNN`` reads ``physics.mask.data`` and calls ``A_adjoint``.
* :class:`L2Fidelity` for ``deepinv.optim.L2`` at ``sigma=1``:
  ``0.5 * ||A(x) - y||^2`` reduced per sample, with gradient
  ``A_adjoint(A(x) - y)``.
* :func:`compare_psnr` for
  ``skimage.metrics.peak_signal_noise_ratio`` called with no
  ``data_range``, as the upstream evaluation scripts call it. On
  floating-point input skimage infers ``data_range`` from the dtype range
  and the sign of the true image: 1.0 when ``true.min() >= 0``, which is
  the case for clean FFHQ images in [0, 1], 2.0 otherwise, and it raises
  if the true image leaves [-1, 1]. That inference is reproduced exactly,
  with the MSE in float64, because the published numbers are defined by
  that call.
"""

from __future__ import annotations

import numpy as np
import torch


class InpaintingPhysics:
    """Masking forward operator ``A(x) = mask * x`` (0/1 mask).

    Args:
        mask: broadcastable 0/1 tensor, e.g. ``(1, 1, H, W)`` -- the
            distributed ``data/mask.npy`` layout. Stored as-is so that
            ``physics.mask.data`` (the access pattern inside
            ``PDHG_ICNN``) sees exactly this tensor.
    """

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * x

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.mask * y

    def A_dagger(self, y: torch.Tensor) -> torch.Tensor:
        # Pseudo-inverse of a 0/1 diagonal operator == the adjoint.
        return self.mask * y


class L2Fidelity:
    """``0.5 * ||A(x) - y||_2^2`` per sample (deepinv ``L2``, sigma=1)."""

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, physics: InpaintingPhysics,
    ) -> torch.Tensor:
        r = physics.A(x) - y
        return 0.5 * r.reshape(r.shape[0], -1).pow(2).sum(dim=-1)

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: InpaintingPhysics,
    ) -> torch.Tensor:
        return physics.A_adjoint(physics.A(x) - y)


def compare_psnr(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """PSNR replicating their exact skimage call (no ``data_range``).

    ``skimage.metrics.peak_signal_noise_ratio(true, test)`` on float
    input infers ``data_range`` from the float dtype range (-1, 1) and
    the true image's sign: 1.0 for non-negative images, 2.0 otherwise.
    MSE is computed in float64 over the whole array.
    """
    image_true = np.asarray(image_true, dtype=np.float64)
    image_test = np.asarray(image_test, dtype=np.float64)
    if image_true.shape != image_test.shape:
        raise ValueError(
            f"shape mismatch: {image_true.shape} vs {image_test.shape}",
        )
    true_min, true_max = float(image_true.min()), float(image_true.max())
    if true_max > 1.0 or true_min < -1.0:
        raise ValueError(
            "image_true outside the float dtype range [-1, 1]; skimage "
            "would raise here too. Pass images scaled to [0, 1].",
        )
    data_range = 1.0 if true_min >= 0.0 else 2.0
    err = float(np.mean((image_true - image_test) ** 2))
    return float(10.0 * np.log10((data_range ** 2) / err))
