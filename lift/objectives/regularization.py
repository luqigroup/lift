"""Entropy regularizer that hooks on top of the EBM-side losses.

A Hutchinson-trace estimate of the per-component Hessian trace,
generalized from the single-ICNN FlatVI version (Ghosh et al. 2025,
arXiv 2506.12903) to ``K`` components.

The per-component differential entropy of :math:`q_{\\theta_k}` is related
to :math:`-\\log\\det \\nabla_x^2 E_{\\theta_k}` by a Laplace argument, and
the :math:`O(D^3)` log-det is replaced by the AM-GM-bounding trace
:math:`R(\\theta_k) = \\E_{x \\sim \\mu_k}[\\mathrm{tr}(\\nabla_x^2
E_{\\theta_k}(x))]`. Components are aggregated as the responsibility-
weighted average :math:`\\sum_k \\bar\\gamma_k R(\\theta_k)`, which matches
the EM M-step structure of HSM; at :math:`\\bar\\gamma_k = 1/K` this is the
unweighted mean. Adding ``-beta * R`` to the EBM-side loss pushes the
per-component Hessian eigenvalues down, giving wider components.

The Hutchinson Laplacian primitive is the one HSM already uses, so the
cost is one extra Hessian-vector product per component per iteration.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from lift.objectives._param_utils import stack_per_k_params


def hess_trace_hutchinson(
    icnn,
    samples: torch.Tensor,
    n_probes: int = 1,
    hutchinson_dist: str = "rademacher",
) -> torch.Tensor:
    """Single-ICNN trace estimator (FlatVI-compatible signature).

    Returns a scalar tensor, differentiable in ``icnn``'s parameters, whose
    value is an unbiased Hutchinson estimate of the mean trace of
    :math:`\\nabla_x^2 E`.

    Args:
        icnn: callable mapping ``(B, D)`` to ``(B, 1)`` or ``(B,)``.
        samples: detached query points, typically MALA or flow samples.
        n_probes: number of probes averaged.
        hutchinson_dist: ``"rademacher"`` (default; lower variance on
            diagonal-dominant Hessians) or ``"gaussian"``.
    """
    if hutchinson_dist not in ("rademacher", "gaussian"):
        raise ValueError(
            f"unknown hutchinson_dist {hutchinson_dist!r}; "
            "expected 'rademacher' or 'gaussian'",
        )
    x = samples.detach().requires_grad_(True)
    E = icnn(x)
    if E.dim() > 1:
        E = E.squeeze(-1)
    grad_E = torch.autograd.grad(
        E.sum(), x, create_graph=True, retain_graph=True,
    )[0]

    acc = torch.zeros((), device=x.device, dtype=x.dtype)
    for _ in range(int(n_probes)):
        if hutchinson_dist == "rademacher":
            z = (
                torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype)
                * 2.0 - 1.0
            )
        else:
            z = torch.randn_like(x)
        Hz = torch.autograd.grad(
            (grad_E * z).sum(), x,
            create_graph=True, retain_graph=True,
        )[0]
        acc = acc + (z * Hz).sum(dim=-1).mean()
    return acc / float(int(n_probes))


def ebm_hess_trace_regulariser(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    samples_per_k: torch.Tensor,
    *,
    K: int,
    D: int,
    bar_gamma: Optional[torch.Tensor] = None,
    n_probes: int = 1,
    hutchinson_dist: str = "rademacher",
) -> torch.Tensor:
    """K-component responsibility-weighted Hessian-trace regularizer.

    For each component k, computes a Hutchinson estimate of
    :math:`\\E_{x \\sim \\mu_k}[\\mathrm{tr}(\\nabla_x^2 E_{\\theta_k}(x))]`
    using ``samples_per_k[k]`` as the integration measure, typically flow
    samples :math:`q_{\\phi_k}` or noise-perturbed data.

    Args:
        ebm: :class:`LogConcaveEBM` template.
        ebm_params: emitted parameters, autograd-attached; the
            regularizer's gradient flows back through these.
        samples_per_k: ``(K, M, D)`` detached per-component query points.
            Autograd is enabled internally on the Hutchinson probe path.
        K, D: mixture and dimension sizes.
        bar_gamma: optional ``(K,)`` weights summing to 1, e.g. the HSM/SM
            M-step ``bar_gamma``. ``None`` means uniform ``1/K``.
        n_probes: Hutchinson probes per component per particle.
        hutchinson_dist: ``"rademacher"`` (default) or ``"gaussian"``.

    Returns:
        Scalar tensor, differentiable in ``ebm_params``, for the caller to
        add to the EBM loss with a coefficient.
    """
    if hutchinson_dist not in ("rademacher", "gaussian"):
        raise ValueError(
            f"unknown hutchinson_dist {hutchinson_dist!r}; "
            "expected 'rademacher' or 'gaussian'",
        )
    if samples_per_k.dim() != 3 or int(samples_per_k.shape[0]) != int(K):
        raise ValueError(
            f"samples_per_k must be (K, M, D); got "
            f"{tuple(samples_per_k.shape)}, K={K}",
        )

    from lift.objectives._batched import aug_energy_hutchinson_lap_batched

    ebm_stacked = stack_per_k_params(
        ebm_params, K=int(K), prefix="components.",
    )
    M = int(samples_per_k.shape[1])
    x = samples_per_k.detach()
    device = x.device
    dtype = x.dtype

    if bar_gamma is None:
        bg = torch.full((int(K),), 1.0 / float(int(K)),
                        device=device, dtype=dtype)
    else:
        if bar_gamma.dim() != 1 or int(bar_gamma.shape[0]) != int(K):
            raise ValueError(
                f"bar_gamma must be 1-D of length K={K}; got "
                f"{tuple(bar_gamma.shape)}",
            )
        bg = bar_gamma.detach().to(device=device, dtype=dtype)

    laps_acc = torch.zeros(int(K), M, device=device, dtype=dtype)
    for _ in range(int(n_probes)):
        if hutchinson_dist == "rademacher":
            v = (torch.randint(
                0, 2, (int(K), M, int(D)), device=device,
            ).to(dtype) * 2.0 - 1.0)
        else:
            v = torch.randn(int(K), M, int(D), device=device, dtype=dtype)
        # (K, M) per-particle Laplacian estimates, autograd-attached.
        laps_acc = laps_acc + aug_energy_hutchinson_lap_batched(
            ebm.components[0], ebm_stacked, x, v,
            K=int(K), D=int(D), D_aug=int(D),
        )
    laps = laps_acc / float(int(n_probes))                      # (K, M)
    per_k_trace = laps.mean(dim=1)                              # (K,)
    return (bg * per_k_trace).sum()
