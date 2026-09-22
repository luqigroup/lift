"""Vmap'd K-component primitives for the EBM and flow forwards.

The hot paths in :mod:`lift.objectives` (mixture log-density, log-Z
estimator, EBM losses, sampler step) each run K independent ICNN
forwards, flow forwards or inner gradients. ``torch.func.vmap`` issues
those K kernels as one batched kernel.

Every primitive here takes a *stacked* parameter dict (see
:func:`lift.objectives._param_utils.stack_per_k_params`) and returns
outputs whose leading dimension is ``K``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.func import functional_call, grad, vmap


def _check_stacked_dict(stacked: Dict[str, torch.Tensor], K: int) -> None:
    """Assert every value has leading dim ``K``."""
    for name, t in stacked.items():
        if int(t.shape[0]) != int(K):
            raise ValueError(
                f"stacked param {name!r} has leading dim {t.shape[0]}, "
                f"expected K={K}",
            )


def aug_energy_batched(
    component_template: nn.Module,
    stacked_params: Dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
) -> torch.Tensor:
    """Augmented EBM energy across K components in parallel.

    Args:
        component_template: any of the per-component ICNN modules
            (e.g. ``lift.components[0]``); used as the template for
            ``functional_call``. Must have the architecture matching
            the stacked parameters.
        stacked_params: ``{name: (K, *param_shape)}`` from
            :func:`stack_per_k_params`.
        x: ``(M, D_aug)`` query points (shared across K -- e.g. data
            for the data-side phase of FKL or HSM responsibilities)
            **or** ``(K, M, D_aug)`` if each component has its own
            particles (e.g. flow draws in RKL / log-Z).
        K, D, D_aug: mixture / dimension knobs.

    Returns:
        ``(K, M)`` tensor of augmented energies.
    """
    _check_stacked_dict(stacked_params, K)

    def _single(p: Dict[str, torch.Tensor], xk: torch.Tensor) -> torch.Tensor:
        # xk is (M, D_aug); the ICNN sees only the first D coordinates.
        E = functional_call(component_template, p, xk[:, :D]).squeeze(-1)
        if D_aug == D:
            return E
        aux = 0.5 * xk[:, D:].pow(2).sum(dim=-1)
        const = 0.5 * (D_aug - D) * math.log(2.0 * math.pi)
        return E + aux + const

    # A shared (M, D_aug) x broadcasts across K; a per-K (K, M, D_aug) x is
    # mapped along dim 0 alongside the parameters.
    if x.dim() == 2:
        return vmap(_single, in_dims=(0, None))(stacked_params, x)
    if x.dim() == 3:
        if int(x.shape[0]) != int(K):
            raise ValueError(
                f"x has leading dim {x.shape[0]}, expected K={K}",
            )
        return vmap(_single, in_dims=(0, 0))(stacked_params, x)
    raise ValueError(f"x must be 2D or 3D, got shape {tuple(x.shape)}")


def conditional_flow_sample(
    flow: nn.Module,
    eps: torch.Tensor,
    *,
    K: int,
    with_logprob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Sample / log-prob from a :class:`ConditionalSamplerFlow`.

    Mirrors :func:`flow_sample_batched`'s signature, but builds the
    per-component index ``k_idx = arange(K)`` internally.

    Args:
        flow: a :class:`~lift.models.conditional_sampler_flow.ConditionalSamplerFlow`
            instance with ``flow.K == K``.
        eps: ``(K, M, D_aug)`` per-component standard-normal latents.
        K: number of components.
        with_logprob: return ``(x_aug, log_q)`` if ``True``, else
            just ``x_aug``.
    """
    if eps.dim() != 3 or int(eps.shape[0]) != int(K):
        raise ValueError(
            f"eps must be (K, M, D_aug); got {tuple(eps.shape)}, K={K}",
        )
    k_idx = torch.arange(int(K), device=eps.device, dtype=torch.long)
    if with_logprob:
        return flow(eps, k_idx, mode="sample_with_logprob")
    return flow(eps, k_idx, mode="sample")


def flow_sample_batched(
    flow_template: nn.Module,
    stacked_params: Dict[str, torch.Tensor],
    stacked_buffers: Dict[str, torch.Tensor],
    eps: torch.Tensor,
    *,
    K: int,
    with_logprob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Sample from K per-component flows in parallel.

    Args:
        flow_template: any per-component flow (e.g.
            ``multi.samplers[0]``); used as the template for
            ``functional_call``.
        stacked_params: ``{name: (K, *)}`` flow parameters.
        stacked_buffers: ``{name: (K, *)}`` flow buffers (``perms`` /
            ``inv_perms`` for HINT).
        eps: ``(K, M, D_aug)`` per-component standard-normal latents.
        K: mixture size (first dim of ``stacked_params`` /
            ``stacked_buffers`` / ``eps``).
        with_logprob: when ``True`` return ``(x_aug, log_q)``; else
            just ``x_aug``.

    Returns:
        ``x_aug`` of shape ``(K, M, D_aug)``, plus ``log_q`` of shape
        ``(K, M)`` when ``with_logprob`` is set.
    """
    _check_stacked_dict(stacked_params, K)
    if stacked_buffers:
        _check_stacked_dict(stacked_buffers, K)
    if eps.dim() != 3 or int(eps.shape[0]) != int(K):
        raise ValueError(
            f"eps must be (K, M, D_aug); got {tuple(eps.shape)}, K={K}",
        )

    if with_logprob:
        def _single(p, b, e):
            x_aug, log_q = functional_call(
                flow_template, (p, b), e,
                kwargs={"mode": "sample_with_logprob"},
            )
            return x_aug, log_q
        return vmap(_single, in_dims=(0, 0, 0))(
            stacked_params, stacked_buffers, eps,
        )
    else:
        def _single(p, b, e):
            return functional_call(flow_template, (p, b), e)
        return vmap(_single, in_dims=(0, 0, 0))(
            stacked_params, stacked_buffers, eps,
        )


def _aug_E_sum_factory(component_template, D, D_aug):
    """Build the single-component scalar :math:`\\sum_i E_{\\text{aug}}(x_i)`.

    The returned function ``_aug_E_sum(p, x)`` accepts per-K parameters
    ``p`` (a flat dict matching the template) and ``x`` of shape
    ``(M, D_aug)``, and returns a scalar (the sum). Differentiating
    this scalar w.r.t. ``x`` via :func:`torch.func.grad` yields the
    per-particle :math:`\\nabla_x E_{\\text{aug}}` of shape
    ``(M, D_aug)`` (the cross-particle Hessian is zero because each
    ``E_{\\text{aug}}(x_i)`` only depends on ``x_i``).
    """
    log_2pi = math.log(2.0 * math.pi)

    def _aug_E_sum(p: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        E = functional_call(component_template, p, x[:, :D]).squeeze(-1)
        if D_aug == D:
            return E.sum()
        aux = 0.5 * x[:, D:].pow(2).sum(dim=-1)
        const = 0.5 * (D_aug - D) * log_2pi
        return (E + aux + const).sum()

    return _aug_E_sum


def aug_energy_grad_batched(
    component_template: nn.Module,
    stacked_params: Dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
) -> torch.Tensor:
    """K-fold per-particle gradient :math:`\\nabla_x E_{\\text{aug},k}`.

    Args:
        component_template: per-component ICNN module (template for
            ``functional_call``).
        stacked_params: ``{name: (K, *)}`` parameters.
        x: ``(M, D_aug)`` shared particles or ``(K, M, D_aug)`` per-K
            particles.

    Returns:
        ``(K, M, D_aug)`` tensor of per-particle gradients. The output
        carries an autograd graph back to ``stacked_params``, so the
        outer ``loss.backward()`` propagates through to the EBM
        hypernetwork's parameters even when this is called with
        autograd-attached ``stacked_params``.
    """
    _check_stacked_dict(stacked_params, K)
    _aug_E_sum = _aug_E_sum_factory(component_template, D, D_aug)
    grad_x = grad(_aug_E_sum, argnums=1)

    if x.dim() == 2:
        return vmap(grad_x, in_dims=(0, None))(stacked_params, x)
    if x.dim() == 3:
        if int(x.shape[0]) != int(K):
            raise ValueError(
                f"x has leading dim {x.shape[0]}, expected K={K}",
            )
        return vmap(grad_x, in_dims=(0, 0))(stacked_params, x)
    raise ValueError(f"x must be 2D or 3D, got shape {tuple(x.shape)}")


def aug_energy_hutchinson_lap_batched(
    component_template: nn.Module,
    stacked_params: Dict[str, torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
) -> torch.Tensor:
    """K-fold Hutchinson estimate of :math:`\\Delta_x E_{\\text{aug},k}`.

    Computes the per-particle quantity
    :math:`v^\\top \\nabla_x(v^\\top \\nabla_x E_{\\text{aug},k}(x))`,
    which has expectation :math:`\\Delta_x E_{\\text{aug},k}(x)` over
    random ``v`` with :math:`\\E[vv^\\top] = I`.

    Args:
        stacked_params: ``{name: (K, *)}`` EBM params.
        x: ``(M, D_aug)`` shared particles or ``(K, M, D_aug)`` per-K.
        v: same shape as ``x``. Must be detached -- the caller is
            responsible for sampling.

    Returns:
        ``(K, M)`` per-particle Laplacian estimates with autograd graph
        back to ``stacked_params``.
    """
    _check_stacked_dict(stacked_params, K)
    _aug_E_sum = _aug_E_sum_factory(component_template, D, D_aug)

    def _kv_lap(p, xk, vk):
        # vk is detached upstream and is never differentiated through.
        inner_grad = grad(_aug_E_sum, argnums=1)
        # f_v(z) = (v . grad_x E_sum(z)).sum() -- a scalar in z, with v fixed.
        def _f_v(z):
            return (vk * inner_grad(p, z)).sum()
        Hv = grad(_f_v, argnums=0)(xk)             # (M, D_aug)
        return (vk * Hv).sum(dim=-1)               # (M,)

    if x.dim() == 2 and v.dim() == 2:
        return vmap(_kv_lap, in_dims=(0, None, None))(
            stacked_params, x, v,
        )
    if x.dim() == 3 and v.dim() == 3:
        if int(x.shape[0]) != int(K) or int(v.shape[0]) != int(K):
            raise ValueError(
                f"x / v leading dim {x.shape[0]} / {v.shape[0]} "
                f"!= K={K}",
            )
        return vmap(_kv_lap, in_dims=(0, 0, 0))(
            stacked_params, x, v,
        )
    raise ValueError(
        f"x and v must have matching dim (2 or 3); got "
        f"{tuple(x.shape)} and {tuple(v.shape)}",
    )


def stack_module_buffers(
    modules,
) -> Dict[str, torch.Tensor]:
    """Stack the buffers of a list of identical modules along a new K dim.

    Mirrors the ``buffers`` half of :func:`torch.func.stack_module_state`,
    but takes a Python list. Used by the trainer to lift the K flow-template
    buffers (``perms``, ``inv_perms``) into the form
    :func:`flow_sample_batched` consumes.
    """
    out: Dict[str, torch.Tensor] = {}
    for mod in modules:
        for name, buf in mod.named_buffers():
            out.setdefault(name, []).append(buf)
    return {name: torch.stack(bs, dim=0) for name, bs in out.items()}
