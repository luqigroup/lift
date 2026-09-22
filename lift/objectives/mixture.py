"""Log-density and frozen responsibilities on :math:`\\mathbb{R}^D`.

The user-facing density on :math:`\\mathbb{R}^D` is
:math:`q_{\\boldsymbol\\theta}(\\boldsymbol x) = \\sum_k \\pi_k\\,
q_{\\boldsymbol\\theta_k}(\\boldsymbol x)`, with
:math:`\\log q_{\\boldsymbol\\theta_k}(\\boldsymbol x) = -E_{\\boldsymbol\\theta_k}
(\\boldsymbol x) - \\log Z_{\\boldsymbol\\theta_k}`. Numerically:

    .. math::

        \\log q_{\\boldsymbol\\theta}(\\boldsymbol x)
        \\;=\\; \\mathrm{logsumexp}_k\\bigl(\\log\\pi_k - E_{\\boldsymbol\\theta_k}
        (\\boldsymbol x) - \\log Z_{\\boldsymbol\\theta_k}\\bigr).

The frozen responsibility is

    .. math::

        \\hat\\gamma_k(\\boldsymbol x) \\;=\\;
        \\mathrm{softmax}_k\\bigl(\\log\\pi_k - E_{\\boldsymbol\\theta_k}(\\boldsymbol x)
        - \\log Z_{\\boldsymbol\\theta_k}\\bigr),

i.e.\\ the per-component log-density terms of the same logsumexp,
softmaxed and detached. Both are computed on the user-facing space
:math:`\\mathbb{R}^D` (no augmentation): :math:`\\log Z_k =
\\log\\widetilde Z_k` by construction, so the augmented Gaussian factor
cancels and the original :math:`E_{\\boldsymbol\\theta_k}` is evaluated
here.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from lift.objectives._batched import aug_energy_batched
from lift.objectives._param_utils import detach_dict, stack_per_k_params


def _per_component_log_q(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    log_Z: torch.Tensor,
    K: int,
    x: torch.Tensor,
    *,
    detach_params: bool,
) -> torch.Tensor:
    """``(M, K)`` matrix of :math:`-E_k(x) - \\log Z_k` values.

    Uses :func:`aug_energy_batched` to evaluate all K components in
    parallel via vmap. ``x`` is the user-facing :math:`\\R^D` query
    point; the augmented flag ``D_aug == D`` makes ``aug_energy_batched``
    a no-op augmentation (just the ICNN forward).

    If ``detach_params`` is True, the stacked ICNN parameters are
    detached so no gradient flows back into ``ebm_params`` -- used by
    the responsibility-softmax computation (frozen at the current
    iterate).
    """
    stacked = stack_per_k_params(ebm_params, K=K, prefix="components.")
    if detach_params:
        stacked = detach_dict(stacked)
    D = int(x.shape[-1])
    # (K, M)
    E_k = aug_energy_batched(
        ebm.components[0], stacked, x,
        K=K, D=D, D_aug=D,
    )
    # (M, K) with sign flip and per-K log Z subtraction.
    return (-E_k - log_Z.view(-1, 1)).transpose(0, 1).contiguous()


def mixture_log_density(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    *,
    K: int,
    x: torch.Tensor,
    detach_params: bool = True,
) -> torch.Tensor:
    """``log q_{theta}(x)`` on :math:`\\mathbb{R}^D`.

    Args:
        ebm: :class:`LogConcaveEBM` template.
        ebm_params: hypernet-emitted parameters for ``ebm``.
        log_pi: ``(K,)`` log-mixing-weights.
        log_Z: ``(K,)`` log-partition values (detached).
        K: number of components.
        x: ``(M, D)`` query points.
        detach_params: whether to detach the EBM parameters before
            evaluating. The two callers do different things:
            ``frozen_log_responsibilities`` always detaches; eval-time
            log-density evaluation may also detach.

    Returns:
        ``(M,)`` tensor of log-density values.
    """
    log_q_per_k = _per_component_log_q(
        ebm, ebm_params, log_Z, K, x, detach_params=detach_params,
    )
    return torch.logsumexp(log_pi.unsqueeze(0) + log_q_per_k, dim=-1)


def frozen_log_responsibilities(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    *,
    K: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """``log hat_gamma_k(x)`` for every k, at points ``x``.

    Returns a detached ``(M, K)`` tensor (every column is the log of
    the k-th responsibility at the M query points). The detach
    reflects the EM-step convention: responsibilities are frozen at
    the current iterate and do not contribute to the EBM gradient.
    """
    log_q_per_k = _per_component_log_q(
        ebm, ebm_params, log_Z, K, x, detach_params=True,
    )
    log_q_full = torch.logsumexp(
        log_pi.unsqueeze(0) + log_q_per_k, dim=-1,
    )
    log_g = log_pi.unsqueeze(0) + log_q_per_k - log_q_full.unsqueeze(-1)
    return log_g.detach()
