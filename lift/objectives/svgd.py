"""Building blocks for amortized Stein variational gradient descent.

Implements the three pieces consumed by the amortized-SVGD trainer:

  * :func:`rbf_median_bandwidth_sq` -- bandwidth :math:`h^2` from the
    median pairwise squared distance, :math:`h^2 = \\mathrm{med} / \\log M`
    (Liu & Wang, 2016).
  * :func:`svgd_direction` -- the SVGD update at every particle,
    :math:`\\Delta x_i = (1/M) \\sum_j [k(x_j, x_i) \\nabla_{x_j} \\log q(x_j)
    + \\nabla_{x_j} k(x_j, x_i)]`, with the RBF kernel and gradient
    :math:`\\nabla_{x_j} k(x_j, x_i) = k_{ji} (x_i - x_j) / h^2`.
  * :func:`distillation_loss` -- the squared-error regression of
    :math:`g_\\phi(\\epsilon)` onto the detached SVGD step
    :math:`x^{old} + \\eta \\Delta x^{old}`.

Reference:
    Wang and Liu, ``Learning to Draw Samples with Amortized Stein
    Variational Gradient Descent``, arXiv:1611.01722, 2016.
"""

from __future__ import annotations

import math

import torch


def rbf_median_bandwidth_sq(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Squared RBF bandwidth from the median pairwise squared distance.

    For ``M >= 2`` particles ``x`` of shape ``(M, D)``,
    :math:`h^2 = \\mathrm{median}_{i \\ne j}\\|x_i - x_j\\|^2 / \\log M`.
    Falls back to ``1.0`` for ``M < 2``. Floors at ``eps`` to avoid
    division-by-zero when particles collapse.
    """
    M = int(x.shape[0])
    if M < 2:
        return torch.ones((), device=x.device, dtype=x.dtype)
    sq = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(-1)  # (M, M)
    mask = ~torch.eye(M, dtype=torch.bool, device=x.device)
    med_sq = torch.median(sq[mask])
    h_sq = med_sq / max(math.log(M), 1.0)
    return h_sq.clamp(min=eps)


def rbf_median_bandwidth_sq_batched(
    x: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    """K-batched squared RBF bandwidth.

    For ``x`` of shape ``(K, M, D)``, returns ``(K,)`` per-component
    bandwidths. The off-diagonal median is computed via ``kthvalue``
    on the flattened pairwise-distance matrix with the diagonal
    replaced by ``+inf`` (so the diagonal zeros do not enter the
    median selection).
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (K, M, D); got shape {tuple(x.shape)}")
    K = int(x.shape[0])
    M = int(x.shape[1])
    if M < 2:
        return torch.ones(K, device=x.device, dtype=x.dtype)
    sq = ((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(-1)  # (K, M, M)
    # Send the diagonal to +inf so it sorts behind every off-diagonal entry
    # in ``kthvalue``; ``torch.where`` avoids the 0 * inf = NaN a multiply
    # would produce.
    eye_mask = torch.eye(M, device=x.device, dtype=torch.bool)
    sq = torch.where(
        eye_mask, torch.full_like(sq, float("inf")), sq,
    )
    flat = sq.reshape(K, M * M)
    # Match ``torch.median``, which on an even count returns the lower
    # middle, i.e. the ``N // 2``-th smallest (1-based) of the N = M*(M-1)
    # off-diagonal entries that now occupy the first sorted positions.
    k = max(1, M * (M - 1) // 2)
    med_sq, _ = flat.kthvalue(k, dim=-1)                    # (K,)
    h_sq = med_sq / max(math.log(M), 1.0)
    return h_sq.clamp(min=eps)


def svgd_direction_batched(
    x: torch.Tensor,
    score: torch.Tensor,
    h_sq: torch.Tensor | None = None,
) -> torch.Tensor:
    """K-batched SVGD update direction.

    Args:
        x: ``(K, M, D)`` per-component particles.
        score: ``(K, M, D)`` per-component scores
            (``-grad_x E_aug`` for an EBM).
        h_sq: ``(K,)`` per-component squared bandwidths; if ``None``,
            computed by :func:`rbf_median_bandwidth_sq_batched`.

    Returns:
        ``(K, M, D)`` per-component SVGD step.
    """
    if x.dim() != 3 or score.dim() != 3:
        raise ValueError(
            f"x / score must be (K, M, D); got {tuple(x.shape)} / "
            f"{tuple(score.shape)}",
        )
    K = int(x.shape[0])
    M = int(x.shape[1])
    if h_sq is None:
        h_sq = rbf_median_bandwidth_sq_batched(x)
    if h_sq.shape != (K,):
        raise ValueError(
            f"h_sq must be (K={K},); got {tuple(h_sq.shape)}",
        )

    sq = ((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(-1)   # (K, M, M)
    K_ker = torch.exp(-sq / (2.0 * h_sq.view(-1, 1, 1)))    # (K, M, M)

    attractive = (K_ker @ score) / float(M)                 # (K, M, D)
    K_sum = K_ker.sum(dim=1)                                # (K, M)
    repulsive = (
        K_sum.unsqueeze(-1) * x - K_ker @ x
    ) / (float(M) * h_sq.view(-1, 1, 1))
    return attractive + repulsive


def svgd_direction(
    x: torch.Tensor,
    score: torch.Tensor,
    h_sq: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """SVGD update direction at every particle.

    Args:
        x: particles, shape ``(M, D)``.
        score: ``grad_x log q(x)`` at every particle, shape ``(M, D)``.
            For an EBM, ``score = -grad_x E_theta(x)``.
        h_sq: squared bandwidth. If ``None``, computed by the median
            heuristic on ``x``.

    Returns:
        ``Delta x`` of shape ``(M, D)``: each row is the SVGD step at
        the corresponding particle.
    """
    M = int(x.shape[0])
    if h_sq is None:
        h_sq = rbf_median_bandwidth_sq(x)
    if not torch.is_tensor(h_sq):
        h_sq = torch.tensor(float(h_sq), device=x.device, dtype=x.dtype)

    sq = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(-1)   # (M, M)
    K = torch.exp(-sq / (2.0 * h_sq))                       # symmetric

    # Attractive term: (1/M) sum_j K_{ji} score(x_j) = (1/M) (K @ score)[i].
    attractive = (K @ score) / M

    # Repulsive term: (1/M) sum_j K_{ji} (x_i - x_j)/h^2
    # = (1/M) (x_i * K_sum[i] - (K @ x)[i]) / h^2
    K_sum = K.sum(dim=0)                                    # (M,)
    repulsive = (K_sum.unsqueeze(-1) * x - K @ x) / (M * h_sq)

    return attractive + repulsive


def distillation_loss(
    g_phi_eps: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between sampler output and detached SVGD step.

    :math:`L_S = (1/M) \\sum_i \\|g_\\phi(\\epsilon_i) - t_i\\|^2`,
    where ``target`` is the detached
    :math:`t_i = x_i^{old} + \\eta \\Delta x_i^{old}`. Per-particle norms
    are summed across the data dimension and averaged over the batch.
    """
    return ((g_phi_eps - target) ** 2).sum(-1).mean()


def synthetic_distillation_loss(
    g_phi_eps: torch.Tensor,
    delta: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """Synthetic-gradient distillation loss (Lop / Wang--Liu Algorithm 1).

    Implements the Theano ``T.Lop`` formulation of the SteinVAE
    reference (lewisKit/Amortized_SVGD/SteinVAE/steinvae.py:220-223):
    a scalar whose autograd w.r.t. the sampler parameters yields the
    Jacobian-vector product :math:`-\\eta\\,J_g^\\top \\Delta`, which an
    Adam step interprets as moving :math:`g_\\phi(\\epsilon)` in the
    SVGD direction.

    Equivalent in expectation to the squared-error distillation above, but
    uses one forward through the sampler instead of two.

    Args:
        g_phi_eps: sampler output, shape ``(M, D)`` -- WITH gradient.
        delta: SVGD direction at the same particles, shape ``(M, D)``,
            **detached**.
        eta: optional multiplicative knob; the Adam learning rate
            already scales the step, so this is usually left at 1.

    Returns:
        Scalar ``-(eta * sum_i <g_phi(eps_i), delta_i>) / M`` whose
        autograd is the synthetic gradient.
    """
    return -float(eta) * (g_phi_eps * delta.detach()).sum() / float(g_phi_eps.shape[0])
