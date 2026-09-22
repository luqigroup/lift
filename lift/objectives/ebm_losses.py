"""EBM-step scalar autograd losses for the training objectives.

Every loss here forms a scalar whose autograd in the EBM hypernetwork's
parameters equals the EM-frozen per-component gradient of one objective:

  * :func:`rkl_em_loss` -- centered-baseline reverse KL at detached flow
    particles. Density-known regime, using ``target.log_prob``.
  * :func:`fkl_direct_em_loss` -- direct MLE, with ``log Z`` re-estimated
    each step by self-normalized importance sampling. Samples-known regime.
  * :func:`fkl_em_loss` -- responsibility-weighted contrastive form: a
    data-side positive phase against a model-side negative phase at
    detached flow particles. Samples-known regime.
  * :func:`sm_em_loss` -- Fisher divergence under the target in Hyvarinen
    form, estimated by self-normalized importance sampling against the
    flow. Density-known regime.
  * :func:`hsm_em_loss` -- Hyvarinen score matching in variance-
    decomposition form, with a Hutchinson estimator for the per-component
    Laplacian. Samples-known regime.
  * :func:`dsm_em_loss` -- denoising score matching (Vincent, 2011) at
    noise-perturbed data. No Laplacian and one ICNN gradient per component
    per noise sample, at the cost of an ``O(sigma^2)`` smoothing bias.

The gradient identities require the sample points to be fixed, so flow
particles, data points, Hutchinson probes, target scores, the frozen
responsibilities and the centering constants are all detached. The energy
and its x-gradient are the only quantities carrying autograd.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from lift.objectives._batched import (
    aug_energy_batched,
    conditional_flow_sample,
)
from lift.objectives._param_utils import (
    detach_dict,
    stack_per_k_params,
)
from lift.objectives.mixture import frozen_log_responsibilities


def _draw_detached_flow_particles_batched(
    flow: nn.Module,
    *,
    K: int,
    M_particles: int,
    D_aug: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw ``(K, M, D_aug)`` detached samples from the conditional flow.

    Each component gets its own ``M_particles`` particles, through the
    ``(K, M, D_aug)`` noise tensor.
    """
    with torch.no_grad():
        eps = torch.randn(int(K), int(M_particles), int(D_aug),
                          device=device, dtype=dtype)
        x_aug = conditional_flow_sample(
            flow, eps, K=K, with_logprob=False,
        ).detach()
    return x_aug


def rkl_em_loss(
    target,
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    flow: nn.Module,
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
    M_particles: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reverse-KL EBM-step loss.

    For each component k, draws :math:`M` detached flow particles, projects
    them to :math:`\\mathbb{R}^D`, forms the personalized target
    :math:`\\hat U_k(x) = U(x) - \\log \\hat\\gamma_k(x)`, and accumulates
    :math:`\\pi_k\\,\\hat\\E_M[(E_{\\theta_k} - \\hat U_k - \\bar c_k)\\,
    E_{\\theta_k}]`, whose autograd is the frozen-responsibility reverse-KL
    gradient. The K ICNN forwards are vmap'd by :func:`aug_energy_batched`.
    """
    pi = torch.exp(log_pi)
    x_aug_per_k = _draw_detached_flow_particles_batched(
        flow,
        K=K, M_particles=int(M_particles), D_aug=D_aug,
        device=device, dtype=dtype,
    )                                                       # (K, M, D_aug)
    x_per_k = x_aug_per_k[:, :, :D]                         # (K, M, D)
    M = int(x_per_k.shape[1])

    # Per-component frozen responsibilities. Each component's particles
    # need their own responsibility evaluation, so K*M is flattened into a
    # single (K*M, D) batch, evaluated, and reshaped back.
    flat = x_per_k.reshape(int(K) * M, int(D))
    log_g_flat = frozen_log_responsibilities(
        ebm, ebm_params, log_pi, log_Z, K=K, x=flat,
    )                                                       # (K*M, K)
    log_g = log_g_flat.reshape(int(K), M, int(K))           # (K, M, K)

    # Diagonal slice: hat_gamma_k(x_i^{(k)}).
    log_hg = log_g[torch.arange(int(K)), :, torch.arange(int(K))]  # (K, M)

    # U(x) at the per-component particles.
    U_per_k = -target.log_prob(flat).reshape(int(K), M)     # (K, M)
    hat_U = (U_per_k - log_hg).detach()                     # (K, M)

    # E_k at its own particles, with autograd.
    ebm_stacked = stack_per_k_params(
        ebm_params, K=K, prefix="components.",
    )
    E_K = aug_energy_batched(
        ebm.components[0], ebm_stacked, x_per_k,
        K=K, D=D, D_aug=D,
    )                                                       # (K, M)
    E_K_det = E_K.detach()

    # The centering constant absorbs the constant log Z.
    bar_c = (E_K_det - hat_U).mean(dim=1, keepdim=True).detach()  # (K, 1)
    coeff = (E_K_det - hat_U - bar_c).detach()              # (K, M)

    loss = (pi.view(-1, 1) * coeff * E_K).mean(dim=1).sum()

    # Mixing-logit M-step. b_k = E_{q_theta_k}[hat_U_k - E_k] - log_Z_k,
    # estimated at the already-detached flow particles; the surrogate is
    # sum_k pi_k * b_k + sum_k pi_k * log_pi_k.
    log_pi_grad = torch.log_softmax(ebm_params["log_pi"], dim=-1)
    pi_grad = torch.exp(log_pi_grad)
    b_hat = (hat_U - E_K_det).mean(dim=1) - log_Z.detach()  # (K,)
    b_hat = b_hat.detach()
    loss = loss + (pi_grad * b_hat).sum() + (pi_grad * log_pi_grad).sum()
    return loss


def fkl_direct_em_loss(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    flow: nn.Module,
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    x_data: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
    M_particles: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Direct-MLE form of the forward-KL EBM step.

    Computes :math:`-\\E_p[\\log \\sum_k \\pi_k \\exp(-E_k(x)) /
    \\widehat Z_k(\\theta)]` with autograd active through both the data-side
    energies and :math:`\\log \\widehat Z_k`, which is re-estimated each step
    by self-normalized importance sampling. The flow enters only as the
    proposal, so the bias is :math:`O(1/M)` in the particle count, where the
    contrastive :func:`fkl_em_loss` instead treats flow particles as exact
    :math:`q_\\theta` samples and carries a bias that does not vanish with
    :math:`M`. Autograd produces the responsibility-weighted contrastive
    split on its own, since :math:`\\nabla_\\theta \\log \\widehat Z_k` is
    minus the importance-weighted average of the model-side energy
    gradients.

    The pre-computed ``log_Z`` argument is unused; it is kept for signature
    parity with :func:`fkl_em_loss`.
    """
    import math as _math

    ebm_stacked = stack_per_k_params(
        ebm_params, K=int(K), prefix="components.",
    )

    # 1) Draw flow particles. The flow is the proposal, and no autograd
    # passes through it at this step.
    with torch.no_grad():
        eps = torch.randn(int(K), int(M_particles), int(D_aug),
                          device=device, dtype=dtype)
        x_aug, log_q_phi = conditional_flow_sample(
            flow, eps, K=int(K), with_logprob=True,
        )
        x_aug = x_aug.detach()                                  # (K, M, D_aug)
        log_q_phi = log_q_phi.detach()                          # (K, M)

    # 2) Fresh autograd-active log Z_k by SNIS in augmented space. The
    # augmented log Z equals the user-facing one, because the auxiliary
    # N(0, I) block is normalized and contributes zero.
    E_aug = aug_energy_batched(
        ebm.components[0], ebm_stacked, x_aug,
        K=int(K), D=int(D), D_aug=int(D_aug),
    )                                                           # (K, M)
    log_Z_fresh = (
        torch.logsumexp(-E_aug - log_q_phi, dim=1)
        - _math.log(int(M_particles))
    )                                                           # (K,)

    # 3) Data-side per-component energy E_k(x_i).
    E_data = aug_energy_batched(
        ebm.components[0], ebm_stacked, x_data,
        K=int(K), D=int(D), D_aug=int(D),
    )                                                           # (K, n)

    # 4) log q_theta(x_i) = logsumexp_k(log pi_k - E_k(x_i) - log Z_k).
    log_pi_grad = torch.log_softmax(ebm_params["log_pi"], dim=-1)  # (K,)
    log_q_at_data = torch.logsumexp(
        log_pi_grad.view(int(K), 1)
        - E_data
        - log_Z_fresh.view(int(K), 1),
        dim=0,
    )                                                           # (n,)

    return -log_q_at_data.mean()


def fkl_em_loss(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    flow: nn.Module,
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    x_data: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
    M_particles: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Contrastive forward-KL EBM-step loss.

    Three terms whose sum's autograd reproduces the responsibility-
    weighted contrastive-divergence gradient:

        (1) data-side positive phase
            :math:`(1/n)\\sum_i \\hat\\gamma_k(\\boldsymbol x_i)\\,
            E_{\\boldsymbol\\theta_k}(\\boldsymbol x_i)`,
        (2) model-side negative phase
            :math:`-\\bar\\gamma_k\\,\\hat\\E_M[E_{\\boldsymbol\\theta_k}
            (\\boldsymbol x_m)]` at flow particles,
        (3) mixing-logit cross-entropy
            :math:`-\\bar\\gamma_k\\,\\log\\pi_k`.

    Both the data-side and the model-side K ICNN forwards are vmap'd by
    :func:`aug_energy_batched`.
    """
    # Frozen responsibilities at the data points (n, K).
    log_g_data = frozen_log_responsibilities(
        ebm, ebm_params, log_pi, log_Z, K=K, x=x_data,
    )
    gamma_data = torch.exp(log_g_data)                      # (n, K)
    bar_gamma = gamma_data.mean(dim=0).detach()             # (K,)

    # Stacked EBM params for the batched forwards.
    ebm_stacked = stack_per_k_params(
        ebm_params, K=K, prefix="components.",
    )

    # (1) data-side positive phase: E_k(x_i) for all (i, k). x_data is
    # shared across K, so aug_energy_batched broadcasts (n, D) to (K, n).
    E_K_data = aug_energy_batched(
        ebm.components[0], ebm_stacked, x_data,
        K=K, D=D, D_aug=D,
    )                                                       # (K, n)
    coeff_data = gamma_data.t().detach()                    # (K, n)
    loss_data = (coeff_data * E_K_data).mean(dim=1).sum()

    # (2) model-side negative phase at flow particles.
    x_aug_per_k = _draw_detached_flow_particles_batched(
        flow,
        K=K, M_particles=int(M_particles), D_aug=D_aug,
        device=device, dtype=dtype,
    )                                                       # (K, M, D_aug)
    E_K_model = aug_energy_batched(
        ebm.components[0], ebm_stacked, x_aug_per_k[:, :, :D],
        K=K, D=D, D_aug=D,
    )                                                       # (K, M)
    loss_model = -(bar_gamma * E_K_model.mean(dim=1)).sum()

    # (3) mixing-logit cross-entropy.
    log_pi_grad = torch.log_softmax(ebm_params["log_pi"], dim=-1)
    loss_pi = -(bar_gamma * log_pi_grad).sum()

    return loss_data + loss_model + loss_pi


def sm_em_loss(
    target,
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    flow: nn.Module,
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    *,
    K: int,
    D: int,
    D_aug: int,
    M_particles: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Score-matching EBM-step loss: Hyvarinen form under :math:`p`.

    Integration by parts makes the Fisher divergence under :math:`p` equal,
    up to a constant in :math:`\\btheta`, to
    :math:`\\E_{\\bx \\sim p}[\\|\\nabla\\log q_{\\btheta}\\|^2 +
    2\\Delta\\log q_{\\btheta}]`. The Laplacian term carries the
    mode-curvature signal that the plain squared residual lacks, since at
    the modes of :math:`p` both scores are near zero.

    The expectation is estimated by self-normalized importance sampling
    against the per-component flow, which adapts to the current model and
    so keeps the variance bounded as :math:`q_{\\btheta}` sharpens; the
    jointly trained SVGD distillation step is what keeps it a usable
    proposal. In augmented space the target log-density is
    :math:`\\log p(\\tilde\\bx_{:D}) + \\log\\N(\\tilde\\bx_{D:};\\bm 0, \\bI)`,
    the auxiliary block being standard normal by construction, and the
    weights are a softmax of :math:`\\log p_{\\rm aug} - \\log q_{\\bphi}`
    over the flat :math:`(K \\cdot M)` axis.

    The frozen-responsibility variance decomposition extends unchanged to
    weighted samples, so the body is :func:`hsm_em_loss` called with
    ``weights``.
    """
    import math as _math

    # Per-component flow particles with their own log-density, from the
    # conditional flow's k-conditioned forward.
    with torch.no_grad():
        eps = torch.randn(int(K), int(M_particles), int(D_aug),
                          device=device, dtype=dtype)
        x_aug_stack, log_q_phi_at_own = conditional_flow_sample(
            flow, eps, K=int(K), with_logprob=True,
        )
        x_aug_stack = x_aug_stack.detach()                  # (K, M, D_aug)
        log_q_phi_at_own = log_q_phi_at_own.detach()        # (K, M)

    M = int(x_aug_stack.shape[1])
    x_per_k = x_aug_stack[:, :, :D]                         # (K, M, D)
    flat = x_per_k.reshape(int(K) * M, int(D))              # (K*M, D)

    # Augmented-space target log-density: log p(x_:D) + log N(x_D:; 0, I),
    # the auxiliary block being unit-variance Gaussian by construction.
    with torch.no_grad():
        log_p_user = target.log_prob(flat).reshape(int(K), M)   # (K, M)
        log_p_aux = (
            -0.5 * x_aug_stack[:, :, D:].pow(2).sum(dim=-1)
            - 0.5 * (int(D_aug) - int(D)) * _math.log(2.0 * _math.pi)
        )                                                       # (K, M)
        log_p_aug = log_p_user + log_p_aux                      # (K, M)
        # SNIS weights: softmax over the flat (K*M) axis.
        log_w = (log_p_aug - log_q_phi_at_own).reshape(int(K) * M)
        tilde_w = torch.softmax(log_w, dim=0).detach()          # (K*M,)

    # Re-derive log_pi from the autograd-active EBM logits so the pi-step
    # inside hsm_em_loss receives a gradient.
    log_pi_for_resp = torch.log_softmax(
        ebm_params["log_pi"], dim=-1,
    )
    return hsm_em_loss(
        ebm, ebm_params, log_pi_for_resp, log_Z, flat.detach(),
        K=int(K), D=int(D), device=device, dtype=dtype,
        weights=tilde_w,
    )


def hsm_em_loss(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    x_data: torch.Tensor,
    *,
    K: int,
    D: int,
    device: torch.device,
    dtype: torch.dtype,
    n_hutchinson: int = 1,
    hutchinson_dist: str = "rademacher",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hyvarinen score-matching EBM-step loss.

    Variance-decomposition form: responsibility-weighted single-component
    score-matching losses plus the component-spread term.

    The K per-component score evaluations and Hutchinson Laplacian
    estimates are vmap'd over the K axis; multi-probe averaging iterates in
    Python, one call per probe.

    Args:
        n_hutchinson: probes per data point. Larger values reduce the
            Laplacian's variance at proportional extra backward cost.
        hutchinson_dist: ``"rademacher"`` (default) or ``"gaussian"``.
            Rademacher probes satisfy :math:`v_i^2 = 1`, so the diagonal of
            :math:`H` enters :math:`v^\\top H v` exactly and only the
            off-diagonal entries carry noise; the estimate is then exact for
            a diagonal Hessian, such as at :math:`D = 1`. Gaussian probes
            add variance from the diagonal as well, so they are strictly
            noisier on diagonal-dominant Hessians.
        weights: optional per-point weights summing to one, used by
            :func:`sm_em_loss` to pass importance weights.
    """
    import lift.objectives._batched as _b

    # Frozen responsibilities at the data points: (n, K).
    log_g = frozen_log_responsibilities(
        ebm, ebm_params, log_pi, log_Z, K=K, x=x_data,
    )
    gamma = torch.exp(log_g).detach()                       # (n, K)

    ebm_stacked = stack_per_k_params(
        ebm_params, K=K, prefix="components.",
    )
    x = x_data.detach()                                     # (n, D)

    # K parallel gradients: (K, n, D).
    grads_K = _b.aug_energy_grad_batched(
        ebm.components[0], ebm_stacked, x,
        K=int(K), D=int(D), D_aug=int(D),
    )

    # Hutchinson Laplacian -- V samples, averaged.
    n = int(x.shape[0])
    laps_acc = torch.zeros(int(K), n, device=device, dtype=dtype)
    for _ in range(int(n_hutchinson)):
        if hutchinson_dist == "rademacher":
            v = (torch.randint(
                0, 2, (n, int(D)), device=device,
            ).to(dtype) * 2.0 - 1.0)                        # +/- 1 i.i.d.
        elif hutchinson_dist == "gaussian":
            v = torch.randn(n, int(D), device=device, dtype=dtype)
        else:
            raise ValueError(
                f"unknown hutchinson_dist: {hutchinson_dist!r} "
                f"(expected 'rademacher' or 'gaussian')",
            )
        # K parallel Laplacians at fixed v: (K, n).
        laps_acc = laps_acc + _b.aug_energy_hutchinson_lap_batched(
            ebm.components[0], ebm_stacked, x, v,
            K=int(K), D=int(D), D_aug=int(D),
        )
    laps_K = laps_acc / float(int(n_hutchinson))            # (K, n)

    # Reshape + assemble in (n, K) frame to match gamma.
    grads_nDK = grads_K.permute(1, 2, 0).contiguous()       # (n, D, K)
    laps_nK = laps_K.t().contiguous()                       # (n, K)
    norm_grads = grads_nDK.pow(2).sum(dim=1)                # (n, K)

    # bar_s(x) = sum_k gamma_k(x) * grad_E_k(x).
    bar_s = (gamma.unsqueeze(1) * grads_nDK).sum(dim=-1)    # (n, D)

    # Spread term: sum_k gamma_k * ||grad_E_k - bar_s||^2.
    spreads = (grads_nDK - bar_s.unsqueeze(-1)).pow(2).sum(dim=1)

    term1 = (gamma * (norm_grads - 2.0 * laps_nK)).sum(-1)  # (n,)
    term2 = (gamma * spreads).sum(-1)                       # (n,)
    integrand = term1 + term2                               # (n,)

    if weights is None:
        loss = integrand.mean()
        bar_gamma = gamma.mean(dim=0).detach()              # (K,)
    else:
        # Weighted estimator, e.g. importance weights from a proposal. The
        # loss and the mixing-weight estimate share the same average.
        if weights.dim() != 1 or int(weights.shape[0]) != n:
            raise ValueError(
                f"weights must be 1-D of length n={n}; got "
                f"{tuple(weights.shape)}",
            )
        loss = (weights * integrand).sum()
        bar_gamma = (weights.unsqueeze(-1) * gamma).sum(dim=0).detach()

    # Mixing-logit M-step. The integrand above has no explicit log_pi
    # dependence, since log_pi only enters through the detached
    # responsibilities, so without this term the mixing weights would stay
    # at their 1/K initialization for every K >= 2. The cross-entropy
    # surrogate has gradient softmax(logits) - bar_gamma in the raw logits,
    # which is the EM M-step direction toward the mean responsibility.
    log_pi_grad = torch.log_softmax(ebm_params["log_pi"], dim=-1)
    loss = loss - (bar_gamma * log_pi_grad).sum()
    return loss


def dsm_em_loss(
    ebm: nn.Module,
    ebm_params: Dict[str, torch.Tensor],
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    x_data: torch.Tensor,
    *,
    K: int,
    D: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float | list[float] | torch.Tensor = 0.05,
) -> torch.Tensor:
    """Denoising-score-matching EBM-step loss (Vincent, 2011).

    For Gaussian noise :math:`\\bm\\varepsilon \\sim \\N(\\bm 0, \\sigma^2
    \\bI)` the conditional score of the perturbation kernel is
    :math:`-\\bm\\varepsilon/\\sigma^2`, so up to a constant in
    :math:`\\btheta` the Fisher divergence against the smoothed target
    :math:`p_\\sigma = p \\ast \\N(\\bm 0, \\sigma^2 \\bI)` equals

    .. math::

        \\E_{\\bx \\sim p,\\,\\bm\\varepsilon}\\bigl[\\,
        \\bigl\\|\\nabla_{\\tilde\\bx}\\log q_{\\btheta}(\\tilde\\bx)
        + \\bm\\varepsilon/\\sigma^2\\bigr\\|^2\\,\\bigr],
        \\qquad \\tilde\\bx = \\bx + \\bm\\varepsilon.

    With frozen responsibilities :math:`\\nabla\\log q_{\\btheta} =
    -\\sum_k \\hat\\gamma_k \\nabla E_{\\btheta_k}`, so the cost is one ICNN
    gradient per component per noise sample and a single backward through
    the squared norm: no Hessian and no Hutchinson probe. The price is that
    the model is fitted to :math:`p_\\sigma` rather than :math:`p`, an
    :math:`O(\\sigma^2)` smoothing bias.

    Given a multi-scale schedule, one :math:`\\sigma_i` is drawn uniformly
    per call and the unweighted per-scale loss is returned, so across
    iterations the optimizer sees an unbiased estimate of the
    schedule-averaged loss at the natural gradient scale of each
    :math:`\\sigma_i`. A closed-form :math:`\\lambda(\\sigma) = \\sigma^2`
    weighting is not applied: it presumes a score network trained to predict
    :math:`\\sigma s`, and on an unscaled ICNN gradient it would overweight
    the largest scale by :math:`\\sigma_{\\max}^2/\\sigma_{\\min}^2`. On a
    multimodal target the largest :math:`\\sigma` should be of order the
    mode separation, so that :math:`p_{\\sigma_{\\max}}` is unimodal.

    Args:
        sigma: noise scale, either a positive scalar (default ``0.05``) or
            a 1-D sequence or tensor of positive scalars, from which one
            scale is drawn per call.
    """
    import lift.objectives._batched as _b

    sigmas = _normalise_sigma_schedule(sigma)
    if sigmas.numel() == 0:
        raise ValueError("sigma schedule is empty")

    ebm_stacked = stack_per_k_params(
        ebm_params, K=K, prefix="components.",
    )
    log_pi_grad = torch.log_softmax(ebm_params["log_pi"], dim=-1)

    # One sigma per call, drawn uniformly; a single-element schedule
    # collapses to the constant case.
    L = int(sigmas.numel())
    if L == 1:
        sigma_i = float(sigmas[0].item())
    else:
        idx = int(torch.randint(0, L, (1,)).item())
        sigma_i = float(sigmas[idx].item())

    # 1) Perturb the data with i.i.d. Gaussian noise.
    eps = torch.randn_like(x_data) * sigma_i                    # (n, D)
    x_tilde = (x_data + eps).detach()                           # (n, D)

    # 2) Frozen responsibilities at the perturbed points.
    log_g = frozen_log_responsibilities(
        ebm, ebm_params, log_pi, log_Z, K=K, x=x_tilde,
    )                                                           # (n, K)
    gamma = torch.exp(log_g).detach()                           # (n, K)

    # 3) K parallel per-particle gradients of E_k at the perturbed points,
    # autograd-attached so the outer backward reaches the emitter.
    grads_K = _b.aug_energy_grad_batched(
        ebm.components[0], ebm_stacked, x_tilde,
        K=int(K), D=int(D), D_aug=int(D),
    )                                                           # (K, n, D)
    grads_nDK = grads_K.permute(1, 2, 0).contiguous()           # (n, D, K)

    # 4) Mixture energy gradient bar_g(x) = sum_k gamma_k grad_E_k, and the
    # unweighted residual ||bar_g - eps/sigma^2||^2.
    bar_g = (gamma.unsqueeze(1) * grads_nDK).sum(dim=-1)        # (n, D)
    target_score = (eps / (sigma_i * sigma_i)).detach()
    residual = bar_g - target_score                              # (n, D)
    loss = residual.pow(2).sum(dim=-1).mean()

    # 5) Mixing-logit M-step: bar_gamma at the chosen sigma.
    bar_gamma = gamma.mean(dim=0).detach()                       # (K,)
    loss = loss - (bar_gamma * log_pi_grad).sum()
    return loss


def _normalise_sigma_schedule(
    sigma: float | list[float] | torch.Tensor,
) -> torch.Tensor:
    """Coerce ``sigma`` to a 1-D float tensor and check it is positive."""
    if isinstance(sigma, torch.Tensor):
        s = sigma.flatten().to(dtype=torch.float64)
    elif isinstance(sigma, (list, tuple)):
        s = torch.tensor(list(sigma), dtype=torch.float64)
    else:
        s = torch.tensor([float(sigma)], dtype=torch.float64)
    if s.numel() == 0 or not bool((s > 0.0).all()):
        raise ValueError(
            f"sigma must be a positive scalar or a sequence of "
            f"positive scalars; got {sigma!r}",
        )
    return s
