"""Partition-function estimator: importance sampling with the flow as proposal.

    .. math::

        \\log\\widehat Z_{\\boldsymbol\\theta_k}
        \\;=\\; \\mathrm{logsumexp}_i
        \\bigl[-\\widetilde E_{\\boldsymbol\\theta_k}(\\boldsymbol{x}_i)
        - \\log q_{\\boldsymbol\\phi_k}(\\boldsymbol{x}_i)\\bigr]
        - \\log M,
        \\qquad \\boldsymbol{x}_i = g_{\\boldsymbol\\phi_k}(\\boldsymbol\\epsilon_i),
        \\;\\boldsymbol\\epsilon_i \\sim \\mathcal{N}(\\boldsymbol 0, \\boldsymbol I_{D_{\\text{aug}}}).

The proposal is the flow itself, and ``log q_phi`` is the exact
change-of-variables density it supplies. The estimator is unbiased for
:math:`Z_k`, with small variance whenever :math:`q_\\phi` tracks
:math:`q_{\\boldsymbol\\theta_k}`, which the sampler-side loss enforces.

The result is detached: :math:`\\log Z` reaches the EBM gradient only
through the responsibility softmax, which is frozen at every iterate.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from lift.objectives._batched import aug_energy_batched
from lift.objectives._param_utils import detach_dict, stack_per_k_params


def log_Z_via_grid_1d(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    *,
    K: int,
    grid: torch.Tensor,
    dvol: float,
) -> torch.Tensor:
    """Per-component :math:`\\log Z_k` by Riemann quadrature at D=1.

    On a 1-D target with bounded ``xlim``, integrating the per-component
    energy over a fine grid is essentially exact and essentially free. The
    flow-IS estimator (:func:`log_Z_via_flow_is`) is instead biased low by
    Jensen's inequality, by an amount that scales with the variance of
    :math:`q_\\phi/q_\\theta`; above D=1 it is the only practical option
    and that bias is the price.

    Args:
        ebm: :class:`LogConcaveEBM` template.
        ebm_params: hypernet-emitted parameters.
        K: number of components.
        grid: ``(N, 1)`` 1D evaluation grid (e.g. from
            :func:`lift.objectives.eval.build_eval_grid`).
        dvol: per-cell volume (cell width in 1D).

    Returns:
        Detached ``(K,)`` tensor of :math:`\\log\\widehat Z_k` values,
        accurate up to grid resolution and the trapezoidal rule's
        :math:`O(dvol^2)` error on smooth integrands.
    """
    ebm_stacked = detach_dict(stack_per_k_params(
        ebm_params, K=K, prefix="components.",
    ))
    with torch.no_grad():
        # (K, N) per-component energy on the grid.
        E = aug_energy_batched(
            ebm.components[0], ebm_stacked, grid,
            K=int(K), D=int(grid.shape[-1]), D_aug=int(grid.shape[-1]),
        )
        # log_Z_k = log(sum_x exp(-E_k(x)) * dvol)
        #        = logsumexp(-E_k) + log(dvol)
        log_Z = torch.logsumexp(-E, dim=1) + math.log(float(dvol))
    return log_Z.detach()


def log_Z_via_grid_md(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    *,
    K: int,
    grid: torch.Tensor,
    dvol: float,
) -> torch.Tensor:
    """Multi-dimensional Riemann quadrature ``log Z_k``.

    An alias of :func:`log_Z_via_grid_1d`, which is already generic in
    ``grid.shape[-1]``. Usable for 2-D and 3-D diagnostics; above that the
    grid becomes impractical and :func:`log_Z_via_flow_is` is the estimator.
    """
    return log_Z_via_grid_1d(
        ebm, ebm_params, K=K, grid=grid, dvol=dvol,
    )


def _parse_base_dist(base_dist: str) -> tuple[str, float]:
    """Resolve ``base_dist`` to (kind, df).

    Accepts ``'normal'`` or ``'t<df>'`` for any positive integer ``<df>``
    (e.g. ``'t4'``, ``'t8'``). Returns ``('normal', float('inf'))`` or
    ``('t', df)``.
    """
    s = str(base_dist).lower().strip()
    if s in ("normal", "n", "gauss", "gaussian"):
        return "normal", float("inf")
    if s.startswith("t"):
        tail = s[1:]
        try:
            df = float(tail)
        except ValueError as e:
            raise ValueError(
                f"unrecognised base_dist {base_dist!r}; expected "
                "'normal' or 't<df>' (e.g. 't4').",
            ) from e
        if df <= 0.0:
            raise ValueError(
                f"t-distribution df must be > 0; got {df}",
            )
        return "t", df
    raise ValueError(
        f"unrecognised base_dist {base_dist!r}; expected "
        "'normal' or 't<df>'.",
    )


def _sample_eps_with_logp_correction(
    base_dist: str,
    *,
    K: int,
    M: int,
    D_aug: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw IS-proposal eps and return the log-density correction.

    For ``base_dist='normal'``: returns ``(eps, 0)`` where ``eps ~
    N(0, I_{D_aug})`` (the flow's native base).

    For ``base_dist='t<df>'``: draws ``eps`` from a per-coordinate
    Student-t with ``df`` degrees of freedom, and returns the
    correction term ``c = log p_t(eps) - log p_normal(eps)`` (shape
    ``(K, M)``). Adding ``c`` to the flow's emitted ``log q`` recovers
    the IS proposal density under the heavier-tail base.

    A Student-t is ``eps = z / sqrt(u/df)`` with ``z ~ N(0, I)`` and
    ``u ~ Chi^2(df)``. Per coordinate,
    ``log p_t(e) = const(df) - 0.5 * (df+1) * log(1 + e^2/df)`` and
    ``log p_N(e) = -0.5 * (e^2 + log 2pi)``.
    """
    kind, df = _parse_base_dist(base_dist)
    if kind == "normal":
        eps = torch.randn(
            int(K), int(M), int(D_aug), device=device, dtype=dtype,
        )
        return eps, torch.zeros(int(K), int(M), device=device, dtype=dtype)

    # t-base: draw eps ~ t_df coordinate-wise.
    z = torch.randn(int(K), int(M), int(D_aug), device=device, dtype=dtype)
    chi2 = torch.distributions.Chi2(df=df).sample(
        (int(K), int(M), 1),
    ).to(device=device, dtype=dtype)
    eps = z / torch.sqrt(chi2 / df)

    # log p_t(eps) summed across D_aug coordinates.
    # const = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi)
    log_const = (
        math.lgamma(0.5 * (df + 1.0))
        - math.lgamma(0.5 * df)
        - 0.5 * math.log(df * math.pi)
    )
    log_p_t = (
        log_const
        - 0.5 * (df + 1.0) * torch.log1p(eps.pow(2) / df)
    ).sum(dim=-1)                                            # (K, M)
    # log p_N(eps) summed across D_aug coordinates.
    log_p_n = (
        -0.5 * eps.pow(2).sum(dim=-1)
        - 0.5 * float(D_aug) * math.log(2.0 * math.pi)
    )                                                         # (K, M)
    return eps, (log_p_t - log_p_n)


def log_Z_via_flow_is(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    flow: nn.Module,
    *,
    K: int,
    D: int,
    D_aug: int,
    M_logZ: int,
    device: torch.device,
    dtype: torch.dtype,
    return_ess: bool = False,
    base_dist: str = "normal",
):
    """Per-component flow-IS estimator of :math:`\\log Z_k`.

    The :class:`ConditionalSamplerFlow` is the proposal: one batched call
    draws K-conditioned samples and their ``log q_phi(x | k)``, the K
    augmented energies come from :func:`aug_energy_batched`, and the
    estimate is the per-k reduction
    ``logsumexp(-E_aug - log q_phi) - log M``.

    Args:
        base_dist: base distribution of the proposal; ``'normal'``, which
            matches the flow's training-time base, or ``'t<df>'`` such as
            ``'t4'`` for a coordinate-wise Student-t. Heavier tails reduce
            the Jensen bias on sharp targets at the cost of ESS. The flow's
            ``log q`` assumes an ``N(0, I)`` base, so the closed-form
            ``log p_t - log p_N`` correction is added and the weight
            ``log w = -E - log q`` uses the heavier-tail density.

    Returns a detached ``(K,)`` tensor of :math:`\\log\\widehat Z_k`.
    """
    from lift.objectives._batched import conditional_flow_sample

    ebm_params_stacked = detach_dict(
        stack_per_k_params(ebm_params, K=K, prefix="components."),
    )

    with torch.no_grad():
        eps, log_p_correction = _sample_eps_with_logp_correction(
            base_dist,
            K=int(K), M=int(M_logZ), D_aug=int(D_aug),
            device=device, dtype=dtype,
        )
        x_aug, log_q = conditional_flow_sample(
            flow, eps, K=K, with_logprob=True,
        )
        # (K, M)
        E_aug = aug_energy_batched(
            ebm.components[0], ebm_params_stacked, x_aug,
            K=K, D=D, D_aug=D_aug,
        )
        # log_q under the requested base = log_q (under N) + correction.
        log_q_corrected = log_q + log_p_correction
        log_w = -E_aug - log_q_corrected
        log_Z = torch.logsumexp(log_w, dim=1) - math.log(int(M_logZ))
        if return_ess:
            log_ess = (
                2.0 * torch.logsumexp(log_w, dim=1)
                - torch.logsumexp(2.0 * log_w, dim=1)
            )
            ess = log_ess.exp()
    if return_ess:
        return log_Z.detach(), ess.detach()
    return log_Z.detach()
