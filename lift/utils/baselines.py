"""GMM baseline + grid metric helpers used by the experiment scripts.

Provides:

* :func:`gmm_k_sweep`, :func:`gmm_fit_logp` -- 1-D sklearn GMM fits.
* :func:`gmm_fit_logp_md` -- multivariate sklearn GMM fit (full or
  diagonal covariance) returning a numpy ``log p(x)`` callable on
  ``(N, D)`` arrays. Used as the high-D baseline for tabular and
  image-latent experiments.
* :func:`grid_metrics_1d`, :func:`grid_metrics_2d` -- TV / Hellinger /
  KL between two log-densities on uniform grids.
"""

from __future__ import annotations

import math as _math
from typing import Callable

import numpy as np
from sklearn.mixture import GaussianMixture


# ----------------------------- GMM K-sweep ---------------------------------

class _GMMWrapper:
    """Light wrapper giving a numpy ``log_prob(x)`` with input shape handling."""

    def __init__(self, gmm: GaussianMixture):
        self.gmm = gmm

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out_shape = x.shape
        xf = x.ravel().reshape(-1, 1)
        return self.gmm.score_samples(xf).reshape(out_shape)


def gmm_k_sweep(
    samples: np.ndarray,
    K_list: list[int],
    seed: int = 0,
    n_init: int = 5,
    max_iter: int = 200,
    reg_covar: float = 1e-6,
) -> dict[int, _GMMWrapper]:
    """Fit a sklearn GMM (full covariance) for each ``K`` in ``K_list``."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1, 1)
    out: dict[int, _GMMWrapper] = {}
    for K in K_list:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
            random_state=seed,
        )
        gmm.fit(x)
        out[K] = _GMMWrapper(gmm)
    return out


def gmm_fit_logp(samples: np.ndarray, K: int, seed: int = 0,
                 max_iter: int = 300, reg_covar: float = 1e-3,
                 n_init: int = 5):
    """Fit a single sklearn ``GaussianMixture`` and return ``log p(x)``.

    The defaults are a regularized EM (``reg_covar=1e-3``, ``n_init=5``)
    rather than sklearn's ``1e-6`` and ``1``: at large ``K`` on data drawn
    from a much simpler distribution, weakly regularized EM produces
    degenerate components, of vanishing variance or a single sample, whose
    per-component log-density spikes inflate the held-out NLL and distort
    the TV and Hellinger metrics downstream. Regularizing the covariance
    toward isotropic ``reg_covar * I`` is the Bayesian-prior reading that
    BIC-based GMM practice relies on.
    """
    gmm = GaussianMixture(
        n_components=K, covariance_type="full",
        random_state=seed, max_iter=max_iter, reg_covar=reg_covar,
        n_init=int(n_init),
    )
    gmm.fit(np.asarray(samples).reshape(-1, 1))

    def logp(x):
        x_in = np.asarray(x).reshape(-1, 1)
        return gmm.score_samples(x_in).reshape(np.asarray(x).shape)

    return logp


# -------------------------- TV / metric utilities --------------------------

def _normalised(log_q: np.ndarray, dx: float) -> np.ndarray:
    msh = float(np.max(log_q))
    q = np.exp(log_q - msh)
    return q / (q.sum() * dx)


def tv_1d_from_log_probs(
    log_q: np.ndarray, log_p: np.ndarray, dx: float,
) -> float:
    """Total-variation distance from two 1D log-density grids."""
    p = _normalised(log_p, dx)
    q = _normalised(log_q, dx)
    return 0.5 * float(np.abs(p - q).sum() * dx)


def grid_metrics_1d(p_logp, q_logp, lo: float, hi: float, n: int) -> dict:
    """Compute TV, Hellinger, KL(p||q) of two 1D log-densities on a grid."""
    x = np.linspace(lo, hi, n)
    dx = (hi - lo) / (n - 1)
    # Subtract the max before exponentiating. The normalization that
    # follows cancels it exactly, so healthy densities are unchanged, but it
    # keeps a diverged model from overflowing to inf or underflowing to zero
    # across the whole grid, which reads as a TV above 1 and a negative KL.
    p = _normalised(np.asarray(p_logp(x), dtype=np.float64), dx)
    q = _normalised(np.asarray(q_logp(x), dtype=np.float64), dx)
    tv = 0.5 * float(np.abs(p - q).sum() * dx)
    hel = _math.sqrt(0.5 * float(((np.sqrt(p) - np.sqrt(q)) ** 2).sum() * dx))
    mask = (p > 1e-12) & (q > 1e-12)
    kl = float((p[mask] * np.log(p[mask] / q[mask])).sum() * dx)
    # How much of q sits in the outermost 1% of the window. TV against a
    # fixed grid is censored once a model's mass leaves that window: it
    # reads 1 whether the model is just outside or catastrophically
    # outside. A large value here means the TV above is a ceiling, not a
    # measurement.
    edge = max(1, int(0.01 * n))
    q_edge = float((q[:edge].sum() + q[-edge:].sum()) * dx)
    return {"tv": tv, "hel": hel, "kl": kl, "q_edge_mass": q_edge}


def grid_metrics_2d(
    p_logp: Callable[[np.ndarray], np.ndarray],
    q_logp: Callable[[np.ndarray], np.ndarray],
    xlim: tuple[tuple[float, float], tuple[float, float]],
    n_per_axis: int,
) -> dict:
    """Compute TV, Hellinger, KL(p||q) on a uniform 2-D grid.

    Args:
        p_logp, q_logp: callables accepting ``(N, 2)`` numpy arrays
            and returning ``(N,)`` log-densities.
        xlim: ``((lo_x, hi_x), (lo_y, hi_y))`` rectangle.
        n_per_axis: grid resolution per axis (so ``n_per_axis**2``
            evaluations of each density).

    Densities are renormalized on the rectangle before comparison, so any
    constant offset such as an unknown log Z cancels.
    """
    (lo_x, hi_x), (lo_y, hi_y) = xlim
    xs = np.linspace(lo_x, hi_x, int(n_per_axis))
    ys = np.linspace(lo_y, hi_y, int(n_per_axis))
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    log_p = np.asarray(p_logp(pts)).reshape(grid_x.shape)
    log_q = np.asarray(q_logp(pts)).reshape(grid_x.shape)
    dx = (hi_x - lo_x) / (int(n_per_axis) - 1)
    dy = (hi_y - lo_y) / (int(n_per_axis) - 1)
    dvol = dx * dy
    p = np.exp(log_p - float(np.max(log_p)))
    p = p / (p.sum() * dvol)
    q = np.exp(log_q - float(np.max(log_q)))
    q = q / (q.sum() * dvol)
    tv = 0.5 * float(np.abs(p - q).sum() * dvol)
    hel = _math.sqrt(0.5 * float(((np.sqrt(p) - np.sqrt(q)) ** 2).sum() * dvol))
    mask = (p > 1e-12) & (q > 1e-12)
    kl = float((p[mask] * np.log(p[mask] / q[mask])).sum() * dvol)
    return {"tv": tv, "hel": hel, "kl": kl}


def gmm_fit_logp_md(
    samples: np.ndarray,
    K: int,
    seed: int = 0,
    covariance_type: str = "full",
    max_iter: int = 300,
    reg_covar: float = 1e-6,
    n_init: int = 1,
):
    """Fit a multivariate sklearn ``GaussianMixture`` (any ``D``).

    Returns a numpy ``log p(x)`` callable accepting ``(N, D)`` arrays.
    For ``D >= 50``, prefer ``covariance_type='diag'`` to avoid
    ``D x D`` covariance storage and inversion cost.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"samples must be (N, D) for gmm_fit_logp_md; got shape {x.shape}",
        )
    gmm = GaussianMixture(
        n_components=int(K),
        covariance_type=str(covariance_type),
        random_state=int(seed),
        max_iter=int(max_iter),
        reg_covar=float(reg_covar),
        n_init=int(n_init),
    )
    gmm.fit(x)

    def logp(x_in):
        x_arr = np.asarray(x_in, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"input has dim {x_arr.shape[-1]}, expected {x.shape[-1]}",
            )
        flat = x_arr.reshape(-1, x_arr.shape[-1])
        return gmm.score_samples(flat).reshape(x_arr.shape[:-1])

    return logp
