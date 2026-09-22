r"""Single-Gaussian warm-start for the EBM energy hypernetwork.

Pretrains the energy hypernetwork :math:`H_E` so the augmented energy
:math:`\widetilde E_{\boldsymbol\theta}` matches a maximum-likelihood
Gaussian fit of the target up to the unknown :math:`\log Z`. The fit is
an MSE regression against :math:`-\log \N(\bx)`; per-component mean
centering removes the constant offset, and the auxiliary-dimension
quadratic is identical on both sides and drops out.

The result is plumbed into :func:`lift.objectives.trainer.train_ebm`
through its ``pre_train_callback`` argument.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from lift.objectives._batched import aug_energy_batched
from lift.objectives._param_utils import stack_per_k_params


__all__ = [
    "make_single_gaussian_init_callback",
    "pretrain_ebm_to_gaussian",
]


def _fit_single_gaussian(samples: np.ndarray, K: int, seed: int):
    """Return ``(mu (K,D), sigma (K,D), pi (K,))`` from a sklearn fit.

    ``sigma`` is the per-dim standard deviation. The fit is diagonal
    because the regression target is per-coordinate additive in the
    augmented space.
    """
    from sklearn.mixture import GaussianMixture
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    fit = GaussianMixture(
        n_components=K, covariance_type="diag",
        random_state=seed, max_iter=300, reg_covar=1e-6,
    )
    fit.fit(samples)
    mu = fit.means_.astype(np.float32)            # (K, D)
    sigma = np.sqrt(fit.covariances_).astype(np.float32)  # (K, D)
    pi = fit.weights_.astype(np.float32)          # (K,)
    return mu, sigma, pi


def _gaussian_neg_log_prob(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """``-log N(x; mu, diag(sigma**2))``, shape ``(K, M)`` from
    ``x: (M, D)``, ``mu: (K, D)``, ``sigma: (K, D)``.
    """
    diff = x[None, :, :] - mu[:, None, :]
    inv = (diff / sigma[:, None, :]).pow(2)
    log_norm = torch.log(sigma).sum(dim=-1) + 0.5 * mu.shape[-1] * math.log(2.0 * math.pi)
    return 0.5 * inv.sum(dim=-1) + log_norm[:, None]   # (K, M)


def pretrain_ebm_to_gaussian(
    *,
    target,
    K: int,
    D: int,
    D_aug: int,
    ebm,
    hyper_E,
    flow=None,
    device: torch.device,
    n_iters: int = 2000,
    lr: float = 1e-3,
    M: int = 512,
    n_data: int = 10000,
    seed: int = 0,
    grad_clip: float = 1.0,
) -> dict:
    """Regress :math:`\\widetilde E_{\\boldsymbol\\theta}` onto the
    per-component negative log-density of a maximum-likelihood Gaussian
    fit of the target.

    Returns a history dict with ``loss_per_iter`` and ``final_loss``.
    """
    dtype = next(hyper_E.parameters()).dtype

    # Maximum-likelihood Gaussian fit on target samples.
    samples_np = target.sample(int(n_data), device="cpu").detach().cpu().numpy()
    if samples_np.ndim == 1:
        samples_np = samples_np.reshape(-1, 1)
    mu_np, sigma_np, _pi_np = _fit_single_gaussian(samples_np, K=K, seed=int(seed))
    mu = torch.from_numpy(mu_np).to(device=device, dtype=dtype)        # (K, D)
    sigma = torch.from_numpy(sigma_np).to(device=device, dtype=dtype)  # (K, D)

    # Hypernet input: a target-sample batch for the DeepSets summary.
    samples_t = torch.from_numpy(samples_np.astype(np.float32)).to(
        device=device, dtype=dtype,
    )
    n_h_max = int(min(samples_t.shape[0], 1024))
    x_h = samples_t[:n_h_max]

    rng = torch.Generator(device=device).manual_seed(int(seed))

    opt = torch.optim.Adam(hyper_E.parameters(), lr=float(lr))
    loss_history: list[float] = []
    for it in range(int(n_iters)):
        opt.zero_grad()

        # Pool per component over [mu_k - 4 sigma_k, mu_k + 4 sigma_k], with
        # i.i.d. Gaussian auxiliary dims, shaped (K, M, D_aug) for the 3D mode
        # of aug_energy_batched.
        x_data_per_k = []
        for k in range(K):
            pts = mu[k] + sigma[k] * 4.0 * (
                2.0 * torch.rand(M, D, generator=rng, device=device, dtype=dtype) - 1.0
            )                                                           # (M, D)
            x_data_per_k.append(pts)
        x_data = torch.stack(x_data_per_k, dim=0)                       # (K, M, D)
        if D_aug > D:
            aux = torch.randn(
                K, M, D_aug - D, generator=rng, device=device, dtype=dtype,
            )
            x_aug = torch.cat([x_data, aux], dim=-1)                    # (K, M, D_aug)
        else:
            x_aug = x_data

        params_E = hyper_E(x_h)
        ebm_stacked = stack_per_k_params(params_E, K=K, prefix="components.")
        E_aug_K = aug_energy_batched(
            ebm.components[0], ebm_stacked, x_aug,
            K=K, D=D, D_aug=D_aug,
        )                                                                # (K, M)

        # Target is -log N(x[:, :D]; mu_k, sigma_k) per k. The auxiliary
        # half-quadratic and the (D_aug - D)/2 * log 2pi constant are the same
        # on both sides and cancel under per-k centering.
        target_per_k_list = []
        for k in range(K):
            t_k = _gaussian_neg_log_prob(
                x_data[k], mu[k:k + 1], sigma[k:k + 1],
            ).squeeze(0)                                                  # (M,)
            target_per_k_list.append(t_k)
        target_K = torch.stack(target_per_k_list, dim=0)                  # (K, M)

        # Per-k centering removes the unknown log Z_k.
        E_centred = E_aug_K - E_aug_K.mean(dim=-1, keepdim=True)
        T_centred = target_K - target_K.mean(dim=-1, keepdim=True)
        loss = (E_centred - T_centred).pow(2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(hyper_E.parameters(), float(grad_clip))
        opt.step()
        # Mirror the trainer's post-step projection so the warm start ends
        # on the feasible cone.
        _proj = getattr(hyper_E, "project_positive", None)
        if callable(_proj):
            _proj()

        loss_history.append(float(loss.detach().cpu().item()))
        if not math.isfinite(loss_history[-1]):
            raise RuntimeError(
                f"warm-start pretraining diverged at iter {it}: "
                f"loss={loss_history[-1]}",
            )

    return {"loss_per_iter": loss_history, "final_loss": loss_history[-1]}


def make_single_gaussian_init_callback(
    *, n_iters: int = 2000, lr: float = 1e-3, M: int = 512,
    n_data: int = 10000, seed: int = 0,
) -> Callable:
    """Build a ``pre_train_callback`` for :func:`train_ebm`.

    The returned callback runs the Gaussian fit and the MSE pretraining
    from its trainer-supplied arguments.
    """
    def _cb(*, target, K, D, D_aug, ebm, hyper_E, flow, device):
        pretrain_ebm_to_gaussian(
            target=target, K=K, D=D, D_aug=D_aug,
            ebm=ebm, hyper_E=hyper_E, flow=flow, device=device,
            n_iters=n_iters, lr=lr, M=M, n_data=n_data, seed=seed,
        )
    return _cb
