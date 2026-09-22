"""Joint training of the EBM hypernetwork and the conditional sampler flow.

One outer loop interleaves an EBM step on :math:`H_E` with :math:`T_S`
sampler steps on the flow. The :math:`\\log Z_k` estimator and the
responsibility softmax are shared across every EBM-side objective.
:func:`train_ebm` returns the templates, the trained networks and a
per-iteration history dict.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

from lift.models import HyperNetwork, LogConcaveEBM
from lift.models.conditional_sampler_flow import ConditionalSamplerFlow
from lift.objectives._param_utils import detach_dict, stack_per_k_params
from lift.objectives.builders import (
    HYPER_HIDDEN_SIZES,
    build_hypernet_ebm,
)
from lift.objectives.ebm_losses import (
    dsm_em_loss,
    fkl_direct_em_loss,
    fkl_em_loss,
    hsm_em_loss,
    rkl_em_loss,
    sm_em_loss,
)
from lift.objectives.eval import build_eval_grid
from lift.objectives.mixture import mixture_log_density
from lift.objectives.partition import log_Z_via_flow_is, log_Z_via_grid_1d


_OBJECTIVES = ("rkl", "fkl", "fkl-direct", "sm", "hsm", "dsm")


# ---------------------------------------------------------------------- helpers

def _ebm_step(
    *,
    objective: str,
    target,
    ebm,
    ebm_params: Dict[str, torch.Tensor],
    flow,
    log_pi: torch.Tensor,
    log_Z: torch.Tensor,
    x_data: torch.Tensor,
    K: int, D: int, D_aug: int, M_particles: int,
    device: torch.device, dtype: torch.dtype,
    n_hutchinson: int = 1,
    hutchinson_dist: str = "rademacher",
    dsm_sigma: float | list[float] = 0.05,
) -> torch.Tensor:
    """Dispatch on the objective and return the scalar autograd loss."""
    if objective == "rkl":
        return rkl_em_loss(
            target, ebm, ebm_params, flow,
            log_pi, log_Z,
            K=K, D=D, D_aug=D_aug, M_particles=M_particles,
            device=device, dtype=dtype,
        )
    if objective == "fkl":
        return fkl_em_loss(
            ebm, ebm_params, flow,
            log_pi, log_Z, x_data,
            K=K, D=D, D_aug=D_aug, M_particles=M_particles,
            device=device, dtype=dtype,
        )
    if objective == "fkl-direct":
        return fkl_direct_em_loss(
            ebm, ebm_params, flow,
            log_pi, log_Z, x_data,
            K=K, D=D, D_aug=D_aug, M_particles=M_particles,
            device=device, dtype=dtype,
        )
    if objective == "sm":
        return sm_em_loss(
            target, ebm, ebm_params, flow,
            log_pi, log_Z,
            K=K, D=D, D_aug=D_aug, M_particles=M_particles,
            device=device, dtype=dtype,
        )
    if objective == "hsm":
        return hsm_em_loss(
            ebm, ebm_params, log_pi, log_Z, x_data,
            K=K, D=D, device=device, dtype=dtype,
            n_hutchinson=int(n_hutchinson),
            hutchinson_dist=str(hutchinson_dist),
        )
    if objective == "dsm":
        # A list of sigmas selects the multi-scale, sigma^2-weighted form.
        return dsm_em_loss(
            ebm, ebm_params, log_pi, log_Z, x_data,
            K=K, D=D, device=device, dtype=dtype,
            sigma=dsm_sigma,
        )
    raise ValueError(
        f"unknown objective: {objective!r} (expected one of {_OBJECTIVES})",
    )


def _sampler_step_svgd(
    *,
    ebm,
    ebm_params_det: Dict[str, torch.Tensor],
    flow,
    K: int, D: int, D_aug: int, M_particles: int,
    eta_svgd: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """K-component SVGD distillation loss, averaged across K."""
    from lift.objectives._batched import (
        aug_energy_grad_batched,
        conditional_flow_sample,
    )
    from lift.objectives.svgd import svgd_direction_batched

    ebm_stacked_det = detach_dict(stack_per_k_params(
        ebm_params_det, K=K, prefix="components.",
    ))

    eps = torch.randn(int(K), int(M_particles), int(D_aug),
                      device=device, dtype=dtype)
    # Autograd flows through ``flow``'s parameters here, and only here.
    x = conditional_flow_sample(flow, eps, K=K, with_logprob=False)
    # Score on a detached copy, so no autograd reaches theta.
    x_det = x.detach()
    score = -aug_energy_grad_batched(
        ebm.components[0], ebm_stacked_det, x_det,
        K=K, D=D, D_aug=D_aug,
    ).detach()                                              # (K, M, D_aug)
    delta = svgd_direction_batched(x_det, score).detach()   # (K, M, D_aug)
    # Synthetic-gradient distillation loss.
    M = float(M_particles)
    return -float(eta_svgd) * (x * delta).sum(dim=(-2, -1)).mean() / M


def _sampler_step_rkl(
    *,
    ebm,
    ebm_params_det: Dict[str, torch.Tensor],
    flow,
    K: int, D: int, D_aug: int, M_particles: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Per-component reverse-KL training of the flow against the EBM.

    For each component :math:`k`, minimize
    :math:`\\mathrm{KL}(q_{\\bphi_k}\\,\\|\\,q_{\\btheta_k})`. Expanding
    against :math:`q_{\\btheta_k}(\\bx) \\propto \\exp(-E_{\\btheta_k}(\\bx))`:

    .. math::

        \\mathrm{KL}(q_{\\bphi_k}\\|q_{\\btheta_k}) =
        \\E_{\\bx \\sim q_{\\bphi_k}}\\!\\Bigl[\\log q_{\\bphi_k}(\\bx)
        + E_{\\btheta_k}(\\bx)\\Bigr] + \\log Z_{\\btheta_k}.

    The :math:`\\log Z_{\\btheta_k}` is constant in :math:`\\bphi`, so the
    autograd surrogate is the bracketed term only, averaged over K. One set
    of flow parameters serves every k, specialized by ``flow.cond_embed``.
    """
    from lift.objectives._batched import (
        aug_energy_batched,
        conditional_flow_sample,
    )

    ebm_stacked_det = detach_dict(stack_per_k_params(
        ebm_params_det, K=K, prefix="components.",
    ))

    eps = torch.randn(int(K), int(M_particles), int(D_aug),
                      device=device, dtype=dtype)
    # (K, M, D_aug) with autograd through ``flow``'s parameters.
    x_aug, log_q_phi = conditional_flow_sample(
        flow, eps, K=K, with_logprob=True,
    )
    # E_aug = E(x_:D) + 0.5||x_aux||^2 + const, with the EBM frozen.
    E_aug = aug_energy_batched(
        ebm.components[0], ebm_stacked_det, x_aug,
        K=K, D=D, D_aug=D_aug,
    )                                                       # (K, M)
    # KL(q_phi_k || q_theta_k) up to the phi-independent constant log Z_k.
    return (log_q_phi + E_aug).mean()


def _sampler_step(
    *,
    sampler_objective: str = "rkl",
    ebm,
    ebm_params_det: Dict[str, torch.Tensor],
    flow,
    K: int, D: int, D_aug: int, M_particles: int,
    eta_svgd: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Dispatch to the chosen sampler-side training objective."""
    if sampler_objective == "rkl":
        return _sampler_step_rkl(
            ebm=ebm, ebm_params_det=ebm_params_det, flow=flow,
            K=K, D=D, D_aug=D_aug, M_particles=M_particles,
            device=device, dtype=dtype,
        )
    if sampler_objective == "svgd":
        return _sampler_step_svgd(
            ebm=ebm, ebm_params_det=ebm_params_det, flow=flow,
            K=K, D=D, D_aug=D_aug, M_particles=M_particles,
            eta_svgd=eta_svgd,
            device=device, dtype=dtype,
        )
    raise ValueError(
        f"unknown sampler_objective: {sampler_objective!r} "
        "(expected 'rkl' or 'svgd')",
    )


def _sampler_kl_estimate(
    *,
    ebm,
    ebm_params_det: Dict[str, torch.Tensor],
    flow,
    log_Z: torch.Tensor,
    K: int, D: int, D_aug: int, M_kl: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Per-component reverse KL of the flow against the augmented EBM target.

    .. math::

        \\KL(q_{\\bphi_k} \\,\\|\\, q_{\\btheta_k})
        \\approx \\frac{1}{M}\\sum_i \\bigl[\\log q_{\\bphi_k}(x_i)
        + E_{\\text{aug}, k}(x_i)\\bigr]
        + \\log Z_{\\btheta_k}.

    Returns a detached ``(K,)`` tensor of KL estimates in nats.
    """
    from lift.objectives._batched import (
        aug_energy_batched,
        conditional_flow_sample,
    )

    ebm_stacked = detach_dict(stack_per_k_params(
        ebm_params_det, K=K, prefix="components.",
    ))

    with torch.no_grad():
        eps = torch.randn(int(K), int(M_kl), int(D_aug),
                          device=device, dtype=dtype)
        x_aug, log_q = conditional_flow_sample(
            flow, eps, K=K, with_logprob=True,
        )                                                   # (K, M), (K, M)
        E_aug = aug_energy_batched(
            ebm.components[0], ebm_stacked, x_aug,
            K=K, D=D, D_aug=D_aug,
        )                                                   # (K, M)
        kl_per_k = (log_q + E_aug).mean(dim=1) + log_Z      # (K,)
    return kl_per_k.detach()


def _evaluate(
    *,
    target,
    ebm, hyper_E, flow,
    K: int, D: int, D_aug: int, M_logZ: int,
    n_batch_hyper: int,
    xs: Optional[torch.Tensor],
    log_p_grid: Optional[torch.Tensor],
    dvol: Optional[float],
    x_val: Optional[torch.Tensor],
    device: torch.device, dtype: torch.dtype,
    base_dist: str = "normal",
) -> Tuple[float, float, float]:
    """One eval pass: returns ``(tv, kl_grid, val_nll)`` (NaN if not configured).

    * ``tv`` and ``kl_grid`` come from the 1D-only TV-grid quadrature
      and are NaN when ``xs is None`` (i.e., :math:`D \\ge 2`).
    * ``val_nll = -E_{x \\sim p}[\\log q_\\theta(x)]`` is the held-out
      negative log-likelihood; this is :math:`\\KL(p\\,\\|\\,q_\\theta) +
      H(p)`, the headline metric that scales to any ``D``.
    """
    with torch.no_grad():
        x_h = target.sample(int(n_batch_hyper), device=device).detach()
        params_E = hyper_E(x_h)
        log_pi = torch.log_softmax(params_E["log_pi"], dim=-1)
        # Grid quadrature is essentially exact at D=1; flow-IS otherwise.
        if xs is not None:
            log_Z = log_Z_via_grid_1d(
                ebm, params_E, K=K, grid=xs, dvol=float(dvol),
            )
        else:
            log_Z = log_Z_via_flow_is(
                ebm, params_E, flow,
                K=K, D=D, D_aug=D_aug, M_logZ=int(M_logZ),
                device=device, dtype=dtype,
                base_dist=str(base_dist),
            )
        if xs is not None:
            log_q = mixture_log_density(
                ebm, params_E, log_pi, log_Z, K=K, x=xs,
            )
            p = torch.exp(log_p_grid); q = torch.exp(log_q)
            p = p / (p.sum() * dvol); q = q / (q.sum() * dvol)
            tv = float(0.5 * (p - q).abs().sum() * dvol)
            mask = (p > 1e-12) & (q > 1e-12)
            kl_grid = float(
                (p[mask] * (p[mask].log() - q[mask].log())).sum() * dvol,
            )
        else:
            tv = float("nan")
            kl_grid = float("nan")
        if x_val is not None:
            log_q_val = mixture_log_density(
                ebm, params_E, log_pi, log_Z, K=K, x=x_val,
            )
            val_nll = float(-log_q_val.mean().item())
        else:
            val_nll = float("nan")
    return tv, kl_grid, val_nll


# ---------------------------------------------------------------------- public

def _conditioning_batch(
    x_data, target, n_batch_cond, cond_source: str, device,
):
    """The samples the emission is conditioned on, given the gradient batch.

    ``n_batch_cond`` unset or non-positive returns the whole gradient batch
    ``x_data``, the default everywhere. Otherwise ``cond_source`` selects n
    samples: ``"independent"`` draws a fresh batch from the target, so the
    emission carries no information about the batch the gradient is taken
    on, while ``"subbatch"`` takes the first n rows of ``x_data``, so the
    two share those samples. ``x_data`` is a fresh i.i.d. draw at every
    iteration, so its first n rows are exchangeable with any other n.
    """
    if n_batch_cond is None or int(n_batch_cond) <= 0:
        return x_data
    n = int(n_batch_cond)
    src = str(cond_source)
    if src == "subbatch":
        if n > int(x_data.shape[0]):
            raise ValueError(
                f"cond_source='subbatch' needs n_batch_cond ({n}) <= the "
                f"gradient batch ({int(x_data.shape[0])}); a sub-batch "
                "cannot be larger than the batch it is drawn from",
            )
        return x_data[:n].detach()
    if src != "independent":
        raise ValueError(
            f"unknown cond_source {src!r}; expected 'independent' or "
            "'subbatch'",
        )
    return target.sample(n, device=device).detach()


def train_ebm(
    target,
    K: int,
    *,
    objective: str,
    n_iters: int,
    lr_ebm: float,
    lr_sampler: Optional[float] = None,
    n_batch_hyper: int,
    eval_every: int,
    grad_clip: float,
    tv_grid_n: int = 0,
    n_batch_val: int = 0,
    n_batch_cond: Optional[int] = None,
    cond_source: str = "independent",
    hidden_dim: int = 128,
    nlayers: int = 3,
    strong_convexity: float = 0.02,
    strong_convexity_schedule: str = "constant",
    strong_convexity_final: Optional[float] = None,
    hyper_hidden_sizes: Optional[list[int]] = None,
    sampler_hyper_hidden_sizes: Optional[list[int]] = None,
    flow_D_aug: Optional[int] = None,
    flow_n_hidden: int = 64,
    flow_n_layers: int = 4,
    flow_n_mlp_layers: int = 3,
    cond_dim: int = 8,
    T_sampler: int = 1,
    T_sampler_max: Optional[int] = None,
    sampler_kl_threshold: Optional[float] = None,
    M_kl_check: int = 256,
    eta_svgd: float = 1.0,
    sampler_objective: str = "rkl",
    n_sampler_warmup_iters: int = 0,
    n_ebm_warmup_iters: int = 0,
    abort_loss_abs: float = 0.0,
    abort_patience: int = 50,
    abort_after: int = 200,
    sampler_lr_ramp: int = 0,
    ramp_sampler_T_max: int = 0,
    pre_train_callback=None,
    eval_callback=None,
    M_particles: int = 512,
    M_logZ: int = 512,
    M_logZ_eval: Optional[int] = None,
    n_hutchinson: int = 1,
    hutchinson_dist: str = "rademacher",
    dsm_sigma: float | list[float] = 0.05,
    component_kind: str = "dense",
    image_shape: Optional[tuple[int, int, int]] = None,
    conv_hidden_channels: int = 32,
    conv_kernel_size: int = 3,
    entropy_reg_weight: float = 0.0,
    entropy_reg_n_probes: int = 1,
    restore_best_val: bool = False,
    n_cooldown_iters: int = 0,
    # Parameter-space repulsive prior between components.
    repulsive_lambda: float = 0.0,
    repulsive_sigma: float = 1.0,
    repulsive_param_key: str = "components.first_layer.bias",
    # ESS-floor watchdog with re-anchor on degenerate components.
    ess_reinit: bool = False,
    ess_floor: int = 50,
    ess_floor_patience: int = 3,
    ess_ema_tau: int = 50,
    ess_below_frac: float = 0.20,
    ess_check_every: int = 100,
    # Heavier-tail base distribution for the IS proposal.
    flow_base_dist: str = "normal",
    # Gaussian noise on hypernet hidden activations (Langevin in phi).
    hyper_activation_noise_std: float = 0.0,
    # Positivity reparameterization for the hypernet readouts on the
    # ICNN-positive parameters: softplus (default), square, qform or cone.
    hyper_pos_constraint_mode: str = "softplus",
    hyper_qform_rank: int = 4,
    # Extra HyperNetwork kwargs; None builds the default architecture.
    hyper_extra_kwargs: Optional[dict] = None,
    ebm_and_hyper_E: Optional[Tuple["LogConcaveEBM", "torch.nn.Module"]] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> Tuple[
    LogConcaveEBM, HyperNetwork, ConditionalSamplerFlow, dict,
]:
    """Joint training of the EBM hypernetwork and the sampler flow.

    Args:
        target: dataset / target with ``.sample(n)``, ``.log_prob(x)``,
            ``.D``, and (for the in-loop TV diagnostic at ``D=1``)
            ``.xlim``.
        K: number of mixture components.
        objective: which EBM-side surrogate to use; one of
            ``{'rkl', 'fkl', 'fkl-direct', 'sm', 'hsm', 'dsm'}``.
            ``'rkl'`` is the centered-baseline reverse KL, ``'fkl'`` the
            responsibility-weighted forward-KL/MLE surrogate, ``'sm'``
            Fisher-divergence score matching under
            :math:`\\mu = q_{\\bphi}`, ``'hsm'`` Hyvärinen score
            matching with the Hutchinson trace estimator, and ``'dsm'``
            Vincent's denoising score matching against the noise-smoothed
            target :math:`p_\\sigma` (takes ``dsm_sigma`` as the noise
            scale). All share the same sampler step and the same flow-IS
            log Z estimator.
        n_iters, lr_ebm, lr_sampler, n_batch_hyper, eval_every,
        grad_clip: training-loop knobs. ``lr_sampler`` defaults to
            ``lr_ebm``.
        tv_grid_n: in-loop TV grid size (1D-only diagnostic; pass 0
            to disable).
        n_batch_val: held-out validation batch size for ``-log q``
            (any D; pass 0 to disable).
        hidden_dim, nlayers, strong_convexity: per-component ICNN
            knobs.
        hyper_hidden_sizes, sampler_hyper_hidden_sizes: DeepSets
            encoder widths for ``H_E`` / ``H_S``. Default
            ``[64, 64, 96]`` for both.
        flow_D_aug: flow working dimension (the auxiliary-Gaussian
            lift of ``augmentation.py`` is applied iff
            ``flow_D_aug > D``). Defaults to ``max(2, D)``: HINT
            requires :math:`D_{\\text{aug}} \\ge 2`.
        flow_n_hidden, flow_n_layers, flow_n_mlp_layers: HINT
            architecture knobs.
        T_sampler: number of sampler-side inner Adam steps per outer
            iteration. When ``sampler_kl_threshold`` is set, this is
            interpreted as the *minimum* number of inner steps; the
            loop runs until ``max_k KL(q_phi_k || q_theta_k)`` falls
            below the threshold or until ``T_sampler_max`` is hit
            (whichever comes first).
        T_sampler_max: cap on the number of inner sampler steps when
            adaptive iteration is active. Defaults to
            ``max(T_sampler, 5)``. Ignored when
            ``sampler_kl_threshold is None``.
        sampler_kl_threshold: per-component KL threshold (in nats) for
            the adaptive-sampler-iteration early-exit. ``None``
            disables adaptation and the loop runs exactly
            ``T_sampler`` inner steps every outer iteration. A loose,
            non-aggressive starting value is ``0.1`` nats.
        M_kl_check: flow-particle budget for the per-iter KL diagnostic
            ``KL(q_phi_k || q_theta_k)``. Used both for the adaptive
            early-exit and for the ``kl_max_per_iter`` history channel.
        eta_svgd: SVGD step in the synthetic-gradient distillation
            loss (multiplicative knob; usually 1).
        M_particles: flow-particle budget for the EBM step and the
            sampler step.
        M_logZ: flow-particle budget for the log Z IS estimator
            during training (responsibility softmax).
        M_logZ_eval: flow-particle budget for the eval-time log Z IS
            estimator. The flow-IS log Z is biased low by Jensen's
            inequality (:math:`\\E[\\log\\widehat Z] \\le \\log Z`) with
            bias proportional to :math:`1/M`, so a larger eval-time budget
            buys cleaner densities and val-NLL numbers outside the
            per-iteration hot path. Defaults to ``M_logZ``; ignored at
            :math:`D = 1`, where the grid quadrature estimator
            (:func:`lift.objectives.partition.log_Z_via_grid_1d`) is used.
        n_hutchinson: number of Hutchinson samples per particle for
            the HSM Laplacian estimator. Used only when
            ``objective == 'hsm'``; ignored for other objectives.
            Larger values reduce the Laplacian variance at a
            proportional extra backward-pass cost.
        dsm_sigma: noise scale for denoising score matching. Used
            only when ``objective == 'dsm'``; ignored otherwise. The
            model is trained against
            :math:`p_\\sigma = p \\ast \\N(\\bm 0,\\sigma^2\\bI)`, so
            sigma sets an :math:`O(\\sigma^2)` smoothing bias.
        entropy_reg_weight: coefficient on the FlatVI-style
            Hutchinson Hessian-trace entropy regularizer
            (Ghosh et al. 2025, arXiv 2506.12903). Adds ``beta * R`` to
            the EBM loss with
            ``R = sum_k bar_gamma_k * E_{q_phi_k}[tr(Hess E_k)]``
            (``bar_gamma = 1/K``, since the regularizer does not need
            responsibilities). Default ``0.0`` (off); non-zero values
            widen the per-component modes.
        entropy_reg_n_probes: Hutchinson probes per particle for the
            entropy regularizer. ``1`` is enough at moderate D.
        restore_best_val: when ``True``, the EBM hypernetwork and the
            sampler flow are checkpointed at every eval step that
            improves ``val_loss_per_eval`` (held-out NLL on
            ``n_batch_val`` target samples), and the best state is
            restored before returning. Held-out NLL equals
            ``KL(p||q) + H(p)`` with ``H(p)`` constant in theta, so its
            argmin is a sharp selection criterion even where the training
            loss has flattened. Requires ``n_batch_val > 0``.
        progress: when ``True`` (default), shows a tqdm progress bar
            with live ``loss_E``, ``loss_S``, ``NLL``, ``TV``, and
            sampler-loop length in the postfix. Set ``False`` for
            quieter logs (tests).

    Returns:
        ``(ebm, hyper_E, flow, history)``: the frozen EBM template, the
        trained EBM hypernetwork ``H_E``, the trained component-conditional
        sampler flow, and a dict of per-iteration training metrics.
    """
    device = device or torch.device("cpu")
    D = int(getattr(target, "D", 1))
    D_aug = int(flow_D_aug if flow_D_aug is not None else max(2, D))
    if lr_sampler is None:
        lr_sampler = float(lr_ebm)
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"unknown objective: {objective!r} (expected one of {_OBJECTIVES})",
        )

    # The caller may inject a pre-built (frozen template, hypernet-shaped
    # Module) pair via ``ebm_and_hyper_E``; this is how the comparison
    # supplies a ``DirectParamModule`` in place of the DeepSets hypernet.
    if ebm_and_hyper_E is None:
        ebm, hyper_E = build_hypernet_ebm(
            K=K, D=D, hidden_dim=hidden_dim, nlayers=nlayers,
            strong_convexity=strong_convexity,
            hyper_hidden_sizes=hyper_hidden_sizes,
            device=device,
            component_kind=str(component_kind),
            image_shape=image_shape,
            conv_hidden_channels=int(conv_hidden_channels),
            conv_kernel_size=int(conv_kernel_size),
            hyper_pos_constraint_mode=str(hyper_pos_constraint_mode),
            hyper_qform_rank=int(hyper_qform_rank),
            hyper_activation_noise_std=float(hyper_activation_noise_std),
            hyper_extra_kwargs=hyper_extra_kwargs,
        )
    else:
        ebm, hyper_E = ebm_and_hyper_E
        if device is not None:
            ebm = ebm.to(device)
            hyper_E = hyper_E.to(device)
    flow = ConditionalSamplerFlow(
        K=K, D=D, D_aug=D_aug,
        n_hidden=flow_n_hidden,
        n_flow_layers=flow_n_layers,
        n_mlp_layers=flow_n_mlp_layers,
        cond_dim=int(cond_dim),
    ).to(device)
    if pre_train_callback is not None:
        # Optional EBM pretraining hook, e.g. a single-Gaussian warm start.
        pre_train_callback(
            target=target, K=K, D=D, D_aug=D_aug,
            ebm=ebm, hyper_E=hyper_E, flow=flow, device=device,
        )

    # A module may expose ``param_groups(lr)`` to give one subset of its
    # parameters a different Adam step. Without the method, every parameter
    # lands in a single group at ``lr_ebm``.
    if hasattr(hyper_E, "param_groups"):
        opt_E = torch.optim.Adam(hyper_E.param_groups(float(lr_ebm)))
    else:
        opt_E = torch.optim.Adam(hyper_E.parameters(), lr=float(lr_ebm))
    opt_S = torch.optim.Adam(flow.parameters(), lr=float(lr_sampler))

    # Eval scaffolding.
    if D == 1 and int(tv_grid_n) > 0:
        xs, log_p_grid, dvol = build_eval_grid(target, int(tv_grid_n), device)
    else:
        xs = log_p_grid = dvol = None
    if n_batch_val > 0:
        with torch.no_grad():
            x_val = target.sample(int(n_batch_val), device=device).detach()
    else:
        x_val = None

    dtype = next(hyper_E.parameters()).dtype

    # Resolve the inner-sampler-loop schedule.
    adaptive = sampler_kl_threshold is not None
    if adaptive:
        T_max = int(T_sampler_max) if T_sampler_max is not None \
            else max(int(T_sampler), 5)
        T_min = int(T_sampler) if T_sampler else 1
    else:
        T_max = int(T_sampler)
        T_min = int(T_sampler)

    history = {
        "loss_E_per_iter": [], "loss_S_per_iter": [],
        "n_inner_steps_per_iter": [], "kl_max_per_iter": [],
        "tv_per_eval": [], "kl_per_eval": [],
        "val_loss_per_eval": [],
        "eval_iters": [],
        "ess_per_k_per_iter": [],
        "log_Z_per_k_per_iter": [],
        "c_per_iter": [],
        "loss_repulsive_per_iter": [],
        "reinit_events": [],
    }

    # ----- ESS-watchdog state -----
    # The per-iteration SNIS ``ess_per_k`` jitters by about two orders of
    # magnitude even for a genuinely collapsing component, so the tracked
    # statistic is an EMA of the indicator ``ess_per_k < ess_floor`` -- the
    # long-run rate of below-floor events -- not an EMA of the noisy value
    # itself, which never crosses a useful threshold. Reinit fires when that
    # rate exceeds ``ess_below_frac`` for ``ess_floor_patience`` consecutive
    # checks.
    _watchdog_consec = torch.zeros(int(K), dtype=torch.long, device=device)
    _ess_below_ema = torch.zeros(
        int(K), dtype=dtype, device=device,
    )
    _ess_ema_alpha = 1.0 / max(1.0, float(ess_ema_tau))

    # The hypernet may expose ``set_noise_scale(scale)``, which scales its
    # internal ``activation_noise_std``. The hypernet injects the noise; the
    # trainer only drives the schedule.
    _has_noise_setter = (
        float(hyper_activation_noise_std) > 0.0
        and hasattr(hyper_E, "set_noise_scale")
    )

    # The strong-convexity floor is an additive ``(c_t/2)||x||^2`` in each
    # component's energy, with c_t interpolating from ``strong_convexity``
    # to ``strong_convexity_final``. Annealing it trades an early SNIS
    # variance bound for late per-component sharpness.
    _sched = str(strong_convexity_schedule).lower()
    _SCHEDULES = ("constant", "linear", "exponential", "cosine")
    if _sched not in _SCHEDULES:
        raise ValueError(
            f"strong_convexity_schedule {_sched!r} must be one of "
            f"{_SCHEDULES}",
        )
    _c_init = float(strong_convexity)
    _c_final = (
        float(strong_convexity_final)
        if strong_convexity_final is not None else _c_init
    )
    # A non-constant schedule crosses every value in (_c_final, _c_init],
    # and c <= 0 breaks log-concave-tail integrability.
    if _sched != "constant" and _c_final <= 0.0:
        raise ValueError(
            f"strong_convexity_final must be positive (got {_c_final}); "
            "c=0 breaks log-concavity-tail integrability.",
        )
    if _sched == "exponential" and _c_init <= 0.0:
        raise ValueError(
            "exponential schedule requires strong_convexity > 0 "
            f"(got {_c_init}); ratio (c_final/c_init)**t is ill-defined.",
        )

    last_tv = float("nan")
    last_nll = float("nan")
    last_n_inner = T_min

    # Best-by-validation-NLL bookkeeping (off unless restore_best_val).
    if bool(restore_best_val) and int(n_batch_val) <= 0:
        raise ValueError(
            "restore_best_val=True requires n_batch_val > 0; got "
            f"n_batch_val={n_batch_val}.",
        )
    best_val_nll: float = float("inf")
    best_iter: int = -1
    best_hyper_E_state: Optional[dict] = None
    best_flow_state: Optional[dict] = None

    # ------------- Sampler warmup -------------
    # Train the flow against the frozen initial EBM before the joint loop.
    # At initialization the flow is approximately the identity over
    # N(0, I) while the EBM is a random ICNN plus the floor (c/2)||x||^2,
    # whose scale alone is 1/sqrt(c); the two can differ by orders of
    # magnitude, and without warmup the first joint iteration takes its
    # EBM-step gradient against a flow that has never seen the EBM.
    if int(n_sampler_warmup_iters) > 0:
        warmup_bar = tqdm(
            range(int(n_sampler_warmup_iters)),
            unit="iter", colour="#A9D8F2",
            dynamic_ncols=True, desc=f"sampler_warmup({objective})",
            disable=not progress,
        )
        for _w in warmup_bar:
            x_data_w = target.sample(
                int(n_batch_hyper), device=device,
            ).detach()
            # Honor the conditioning-batch setting here too, so the
            # warmup sampler sees the fluctuation the main loop sees.
            x_cond_w = _conditioning_batch(
                x_data_w, target, n_batch_cond, cond_source, device,
            )
            with torch.no_grad():
                params_E_w = hyper_E(x_cond_w)
                ebm_params_det_w = detach_dict(params_E_w)
            # ``sampler_lr_ramp`` starts this phase from a small step,
            # lr_S = lr_sampler * (w + 1) / n_sampler_warmup_iters: a steep
            # initial energy can otherwise inflate the flow to a non-finite
            # loss within a few tens of full-step updates, after which every
            # later step is skipped. Off by default.
            if int(sampler_lr_ramp):
                _ws = min(1.0, (_w + 1) / float(int(n_sampler_warmup_iters)))
                for _g in opt_S.param_groups:
                    _g["lr"] = float(lr_sampler) * _ws
            loss_S_w = _sampler_step(
                sampler_objective=str(sampler_objective),
                ebm=ebm, ebm_params_det=ebm_params_det_w, flow=flow,
                K=K, D=D, D_aug=D_aug,
                M_particles=int(M_particles),
                eta_svgd=float(eta_svgd),
                device=device, dtype=dtype,
            )
            if not torch.isfinite(loss_S_w):
                opt_S.zero_grad()
                continue
            opt_S.zero_grad()
            loss_S_w.backward()
            torch.nn.utils.clip_grad_norm_(
                flow.parameters(), float(grad_clip),
            )
            opt_S.step()
            warmup_bar.set_postfix({"loss_S": f"{float(loss_S_w.item()):.3f}"})

    bar = tqdm(
        range(int(n_iters)),
        unit="iter", colour="#B5F2A9",
        dynamic_ncols=True, desc=f"train_ebm({objective})",
        disable=not progress,
    )
    _abort_streak = 0          # dead-run abort
    for it in bar:
        # ------------- (S-1) EBM learning-rate warmup -------------
        # The joint phase begins with the proposal far from the energy: on
        # a heavy-tailed target the sampler can leave its own warmup with
        # an effective sample size of one, and if the energy moves at full
        # step size while the partition estimate is still noise, the pair
        # locks into a degenerate state within a few tens of iterations and
        # never recovers. Ramping the energy's step size from zero over
        # ``n_ebm_warmup_iters`` holds it still while the proposal catches
        # up. Zero, the default, disables the ramp.
        _T_max_it = T_max
        if int(n_ebm_warmup_iters) > 0:
            _scale = min(1.0, (it + 1) / float(int(n_ebm_warmup_iters)))
            for _g in opt_E.param_groups:
                _g["lr"] = float(lr_ebm) * _scale
            # ------------- (S-1.b) Sampler under the ramp -------------
            # Both knobs default off. ``sampler_lr_ramp`` makes the
            # sampler's step size follow the energy's ramp factor, which
            # keeps long inner bursts at full ``lr_sampler`` from inflating
            # the flow while the energy is still far from the data.
            # ``ramp_sampler_T_max`` caps the inner loop while the ramp is
            # active: the adaptive loop exists to track energy motion, and
            # there is little to track while the energy step is scaled down.
            if int(sampler_lr_ramp):
                for _g in opt_S.param_groups:
                    _g["lr"] = float(lr_sampler) * _scale
            if int(ramp_sampler_T_max) > 0 and it < int(n_ebm_warmup_iters):
                _T_max_it = min(T_max, int(ramp_sampler_T_max))

        # ------------- (S0) Strong-convexity schedule -------------
        # c_t runs from ``_c_init`` to ``_c_final`` and propagates into
        # each component's strong-convexity floor.
        t_frac = it / max(1, int(n_iters) - 1)
        if _sched == "linear":
            c_t = _c_init + (_c_final - _c_init) * t_frac
        elif _sched == "exponential":
            # Geometric interpolation in log-c (slow late, fast early).
            c_t = _c_init * (_c_final / _c_init) ** t_frac
        elif _sched == "cosine":
            # Half-cosine: slow start, fast middle, slow end.
            c_t = _c_final + 0.5 * (_c_init - _c_final) * (
                1.0 + math.cos(math.pi * t_frac)
            )
        else:  # constant
            c_t = _c_init
        if _sched != "constant":
            for _comp in ebm.components:
                _comp.strong_convexity = float(c_t)
        history["c_per_iter"].append(float(c_t))

        # ------------- (S0.b) Hypernet activation-noise schedule -------
        # The scale anneals linearly from 1 to 0 over the first half of
        # training; the hypernet multiplies its base std by it.
        if _has_noise_setter:
            noise_scale = max(0.0, 1.0 - 2.0 * t_frac)
            hyper_E.set_noise_scale(float(noise_scale))

        # ------------- Hypernet forward (EBM only) -------------
        # ``n_batch_hyper`` sizes the loss-side batch; ``n_batch_cond``, if
        # set, sizes the conditioning batch the hypernet pools over, so the
        # conditioning size moves without changing the loss-side
        # stochasticity. ``_conditioning_batch`` documents ``cond_source``.
        x_data = target.sample(int(n_batch_hyper), device=device).detach()
        x_cond = _conditioning_batch(
            x_data, target, n_batch_cond, cond_source, device,
        )
        params_E = hyper_E(x_cond)

        # ------------- (L) Partition function -------------
        log_pi = torch.log_softmax(params_E["log_pi"], dim=-1)
        log_Z, ess_per_k = log_Z_via_flow_is(
            ebm, params_E, flow,
            K=K, D=D, D_aug=D_aug, M_logZ=int(M_logZ),
            device=device, dtype=dtype,
            return_ess=True,
            base_dist=str(flow_base_dist),
        )
        history["ess_per_k_per_iter"].append(
            ess_per_k.detach().cpu().tolist(),
        )
        history["log_Z_per_k_per_iter"].append(
            log_Z.detach().cpu().tolist(),
        )

        # ------------- (L.b) ESS-floor watchdog -------------
        # A component whose below-floor rate stays high across
        # ``ess_floor_patience`` consecutive checks is re-anchored at a
        # random training point. Off by default.
        with torch.no_grad():
            _below_indicator = (
                (ess_per_k.detach() < float(ess_floor)).to(dtype)
            )
            _ess_below_ema = (
                (1.0 - _ess_ema_alpha) * _ess_below_ema
                + _ess_ema_alpha * _below_indicator
            )
        if bool(ess_reinit) and (it + 1) % int(ess_check_every) == 0:
            with torch.no_grad():
                _below = (_ess_below_ema > float(ess_below_frac)).long()
                _watchdog_consec = _watchdog_consec * _below + _below
                _dead = (
                    _watchdog_consec >= int(ess_floor_patience)
                ).nonzero(as_tuple=True)[0].tolist()
            if _dead:
                from lift.objectives.component_reinit import (
                    reinit_component_k,
                )
                for k_dead in _dead:
                    reinit_component_k(
                        hyper_E=hyper_E, opt_E=opt_E,
                        K=int(K), D=int(D), k=int(k_dead),
                        x_data=x_data,
                        device=device, dtype=dtype,
                    )
                    history["reinit_events"].append(
                        {"iter": int(it), "k": int(k_dead),
                         "ess": float(ess_per_k[int(k_dead)].item())},
                    )
                    _watchdog_consec[int(k_dead)] = 0

        # ------------- (A) EBM step on H_E -------------
        loss_E = _ebm_step(
            objective=objective,
            target=target,
            ebm=ebm, ebm_params=params_E,
            flow=flow,
            log_pi=log_pi.detach(), log_Z=log_Z,
            x_data=x_data,
            K=K, D=D, D_aug=D_aug, M_particles=int(M_particles),
            device=device, dtype=dtype,
            n_hutchinson=int(n_hutchinson),
            hutchinson_dist=str(hutchinson_dist),
            dsm_sigma=dsm_sigma,
        )

        # ------------- (A.1) Optional Hessian-trace entropy reg -------------
        # Following FlatVI (Ghosh et al. 2025), ``R = E_{q_phi}[tr(Hess E)]``
        # is an AM upper bound on the negative differential entropy, so
        # minimizing ``loss + beta * R`` pushes the per-component Hessian
        # eigenvalues down and widens the components.
        if float(entropy_reg_weight) > 0.0 and torch.isfinite(loss_E):
            from lift.objectives.regularization import (
                ebm_hess_trace_regulariser,
            )
            from lift.objectives._batched import conditional_flow_sample

            with torch.no_grad():
                eps_reg = torch.randn(
                    int(K), int(M_particles), int(D_aug),
                    device=device, dtype=dtype,
                )
                x_reg_aug = conditional_flow_sample(
                    flow, eps_reg, K=int(K), with_logprob=False,
                ).detach()
                x_reg = x_reg_aug[:, :, :D].contiguous()
            reg = ebm_hess_trace_regulariser(
                ebm, params_E, x_reg,
                K=int(K), D=int(D),
                bar_gamma=None,
                n_probes=int(entropy_reg_n_probes),
                hutchinson_dist=str(hutchinson_dist),
            )
            loss_E = loss_E + float(entropy_reg_weight) * reg

        # ------------- (A.2) Component-repulsive prior --------
        # Adds ``lambda * 0.5 * sum_{k!=k'} exp(-||v_k - v_k'||^2 /
        # 2 sigma_rep^2)`` to ``loss_E``, where ``v_k`` is the
        # ``repulsive_param_key`` slice of component k after
        # ``stack_per_k_params``. Pushes degenerate components apart in
        # parameter space.
        loss_rep_value = float("nan")
        if (
            float(repulsive_lambda) > 0.0
            and torch.isfinite(loss_E)
            and int(K) > 1
        ):
            stacked_for_rep = stack_per_k_params(
                params_E, K=int(K), prefix="components.",
            )
            rep_key = str(repulsive_param_key).replace(
                "components.", "", 1,
            )
            if rep_key not in stacked_for_rep:
                raise KeyError(
                    f"repulsive_param_key {repulsive_param_key!r} not "
                    f"in hypernet outputs; available keys (post-strip): "
                    f"{sorted(stacked_for_rep.keys())}",
                )
            v = stacked_for_rep[rep_key].reshape(int(K), -1)  # (K, P)
            d2 = (
                (v.unsqueeze(0) - v.unsqueeze(1)).pow(2).sum(dim=-1)
            )                                                # (K, K)
            mask = 1.0 - torch.eye(
                int(K), device=device, dtype=v.dtype,
            )
            rep = 0.5 * (
                torch.exp(-d2 / (2.0 * float(repulsive_sigma) ** 2))
                * mask
            ).sum()
            loss_E = loss_E + float(repulsive_lambda) * rep
            loss_rep_value = float(rep.item())
        history["loss_repulsive_per_iter"].append(loss_rep_value)

        # ------------- Dead-run abort -------------
        # A run whose energy loss is non-finite or absurd for
        # ``abort_patience`` consecutive iterations after ``abort_after``
        # does not recover: an early kick from a garbage log Z sends the
        # emitted weights far out, the non-finite guard then freezes them,
        # and the loss stays there for the rest of the budget. The caller
        # reads ``history["aborted_at_iter"]``; ``abort_loss_abs = 0``, the
        # default, disables the check.
        if float(abort_loss_abs) > 0.0 and it >= int(abort_after):
            _lv = float(loss_E.item())
            _bad_now = (not math.isfinite(_lv)) or abs(_lv) > float(abort_loss_abs)
            _abort_streak = _abort_streak + 1 if _bad_now else 0
            if _abort_streak >= int(abort_patience):
                history["aborted_at_iter"] = int(it + 1)
                print(
                    f"[train_ebm] ABORT at iter {it + 1}: |loss_E| > "
                    f"{float(abort_loss_abs):g} or non-finite for "
                    f"{int(abort_patience)} consecutive iterations -- dead run",
                    flush=True,
                )
                break
        if torch.isfinite(loss_E):
            opt_E.zero_grad()
            loss_E.backward()
            torch.nn.utils.clip_grad_norm_(
                hyper_E.parameters(), float(grad_clip),
            )
            opt_E.step()
            # Post-step positivity projection for PGD. The lift emits
            # non-negative weights through its readout reparameterization
            # and needs none; ``DirectParamModule`` in PGD mode exposes
            # ``project_positive()``, which clamps the positivity-tagged
            # raw weights in place. A plain DeepSets hypernet does not
            # define the method, so the call is skipped.
            _proj = getattr(hyper_E, "project_positive", None)
            if callable(_proj):
                _proj()
            history["loss_E_per_iter"].append(float(loss_E.item()))
        else:
            opt_E.zero_grad()
            history["loss_E_per_iter"].append(float("nan"))
            history["loss_S_per_iter"].append(float("nan"))
            history["n_inner_steps_per_iter"].append(0)
            history["kl_max_per_iter"].append(float("nan"))
            continue

        # ------------- (B) Sampler step(s) on H_S -------------
        # ``params_E`` predates the EBM step (one-step-stale theta), which
        # is accurate enough here and saves one H_E forward.
        ebm_params_det = detach_dict(params_E)
        loss_S_last = float("nan")
        kl_max_last = float("nan")
        n_inner_done = 0
        for _t in range(_T_max_it):
            loss_S = _sampler_step(
                sampler_objective=str(sampler_objective),
                ebm=ebm, ebm_params_det=ebm_params_det, flow=flow,
                K=K, D=D, D_aug=D_aug,
                M_particles=int(M_particles),
                eta_svgd=float(eta_svgd),
                device=device, dtype=dtype,
            )
            if not torch.isfinite(loss_S):
                opt_S.zero_grad()
                continue
            opt_S.zero_grad()
            loss_S.backward()
            torch.nn.utils.clip_grad_norm_(
                flow.parameters(), float(grad_clip),
            )
            opt_S.step()
            loss_S_last = float(loss_S.item())
            n_inner_done += 1

            # Adaptive early-exit: cheap IS-based KL estimate at the
            # current flow parameters, exit when below threshold.
            if adaptive and n_inner_done >= T_min:
                kl_per_k = _sampler_kl_estimate(
                    ebm=ebm, ebm_params_det=ebm_params_det, flow=flow,
                    log_Z=log_Z,
                    K=K, D=D, D_aug=D_aug, M_kl=int(M_kl_check),
                    device=device, dtype=dtype,
                )
                kl_max_last = float(kl_per_k.max().item())
                if math.isfinite(kl_max_last) \
                        and kl_max_last < float(sampler_kl_threshold):
                    break

        if adaptive and not math.isfinite(kl_max_last) \
                and n_inner_done >= 1:
            with torch.no_grad():
                kl_per_k = _sampler_kl_estimate(
                    ebm=ebm, ebm_params_det=ebm_params_det, flow=flow,
                    log_Z=log_Z,
                    K=K, D=D, D_aug=D_aug, M_kl=int(M_kl_check),
                    device=device, dtype=dtype,
                )
                kl_max_last = float(kl_per_k.max().item())

        history["loss_S_per_iter"].append(loss_S_last)
        history["n_inner_steps_per_iter"].append(int(n_inner_done))
        history["kl_max_per_iter"].append(kl_max_last)
        last_n_inner = int(n_inner_done)

        # ------------- Eval -------------
        if it % int(eval_every) == 0 or it == int(n_iters) - 1:
            M_logZ_eval_eff = (
                int(M_logZ_eval) if M_logZ_eval is not None
                else int(M_logZ)
            )
            tv, kl_grid, val_nll = _evaluate(
                target=target,
                ebm=ebm, hyper_E=hyper_E, flow=flow,
                K=K, D=D, D_aug=D_aug, M_logZ=M_logZ_eval_eff,
                n_batch_hyper=int(n_batch_hyper),
                xs=xs, log_p_grid=log_p_grid, dvol=dvol, x_val=x_val,
                device=device, dtype=dtype,
                base_dist=str(flow_base_dist),
            )
            history["tv_per_eval"].append(tv)
            history["kl_per_eval"].append(kl_grid)
            history["val_loss_per_eval"].append(val_nll)
            history["eval_iters"].append(it)
            last_tv = tv
            last_nll = val_nll
            if bool(restore_best_val) and math.isfinite(val_nll) \
                    and val_nll < best_val_nll:
                best_val_nll = float(val_nll)
                best_iter = int(it)
                best_hyper_E_state = {
                    k: v.detach().clone()
                    for k, v in hyper_E.state_dict().items()
                }
                best_flow_state = {
                    k: v.detach().clone()
                    for k, v in flow.state_dict().items()
                }
            if eval_callback is not None:
                eval_callback(
                    it=int(it), ebm=ebm, hyper_E=hyper_E, flow=flow,
                    K=int(K), D=int(D), D_aug=int(D_aug),
                    history=history, device=device,
                )

        bar.set_postfix({
            "loss_E": f"{history['loss_E_per_iter'][-1]:.3f}",
            "loss_S": f"{history['loss_S_per_iter'][-1]:.3f}",
            "NLL": (f"{last_nll:.3f}"
                    if math.isfinite(last_nll) else "n/a"),
            "TV": (f"{last_tv:.3f}"
                   if math.isfinite(last_tv) else "n/a"),
            "T_S": f"{last_n_inner:d}",
        })

    if bool(restore_best_val) and best_hyper_E_state is not None:
        hyper_E.load_state_dict(best_hyper_E_state)
        flow.load_state_dict(best_flow_state)
        history["best_val_iter"] = int(best_iter)
        history["best_val_nll"] = float(best_val_nll)

    # ---------------- Sampler cool-down (post-training) ----------------
    # With the EBM frozen, extra sampler-only iterations let the flow track
    # the now-stationary q_theta more tightly, which sharpens the flow-IS
    # log Z and downstream likelihoods without moving the EBM.
    if int(n_cooldown_iters) > 0 and "aborted_at_iter" not in history:
        cooldown_bar = tqdm(
            range(int(n_cooldown_iters)),
            unit="iter", colour="#FFD27A",
            dynamic_ncols=True, desc=f"sampler_cooldown({objective})",
            disable=not progress,
        )
        history.setdefault("loss_S_cooldown", [])
        for _c in cooldown_bar:
            x_data_c = target.sample(
                int(n_batch_hyper), device=device,
            ).detach()
            x_cond_c = _conditioning_batch(
                x_data_c, target, n_batch_cond, cond_source, device,
            )
            with torch.no_grad():
                params_E_c = hyper_E(x_cond_c)
                ebm_params_det_c = detach_dict(params_E_c)
            loss_S_c = _sampler_step(
                sampler_objective=str(sampler_objective),
                ebm=ebm, ebm_params_det=ebm_params_det_c, flow=flow,
                K=K, D=D, D_aug=D_aug,
                M_particles=int(M_particles),
                eta_svgd=float(eta_svgd),
                device=device, dtype=dtype,
            )
            if not torch.isfinite(loss_S_c):
                opt_S.zero_grad()
                continue
            opt_S.zero_grad()
            loss_S_c.backward()
            torch.nn.utils.clip_grad_norm_(
                flow.parameters(), float(grad_clip),
            )
            opt_S.step()
            history["loss_S_cooldown"].append(float(loss_S_c.item()))
            cooldown_bar.set_postfix(
                {"loss_S": f"{float(loss_S_c.item()):.3f}"},
            )

    return ebm, hyper_E, flow, history
