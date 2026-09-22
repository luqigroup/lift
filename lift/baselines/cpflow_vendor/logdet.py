"""Stochastic log-det estimators vendored from CP-Flow.

One change over upstream ``lib/logdet_estimators.py``:
``torch.symeig(..., eigenvectors=True)`` becomes ``torch.linalg.eigh(...)``.
Every estimator upstream's ``DeepConvexFlow`` can select is kept, including
the ones this repo never selects.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-7
CG_ITERS_TRACER: list = []


def _eigh(M: torch.Tensor):
    """Symmetric-eigh that matches the old ``torch.symeig`` interface."""
    return torch.linalg.eigh(M)


def gram_schmidt_ortho(Q: torch.Tensor, v: torch.Tensor, tol: float = 1e-5):
    *shape, m, d = Q.shape
    Q = Q.reshape(-1, m, d)
    v = v.reshape(-1, d)
    inner_qv = torch.einsum("bmd,bd->bm", Q, v)
    proj_v = torch.einsum("bm,bmd->bd", inner_qv, Q)
    v = v - proj_v
    v_norm = torch.norm(v, dim=-1, keepdim=True).detach() + EPS
    inner_qv = torch.einsum("bmd,bd->bm", Q, v / v_norm)
    retries = 0
    while (torch.abs(inner_qv) > tol).any():
        proj_v = torch.einsum("bm,bmd->bd", inner_qv * v_norm, Q)
        v = v - proj_v
        inner_qv = torch.einsum("bmd,bd->bm", Q, v)
        retries += 1
        if retries >= 10:
            break
    return v.reshape(*shape, d)


def lanczos_tridiagonalization(hvp_fun, m: int, v: torch.Tensor) -> torch.Tensor:
    bsz, d = v.shape
    vecs = [v]
    Q = torch.stack(vecs, dim=1)

    w = hvp_fun(v)
    alpha = torch.einsum("bi,bi->b", w, v)
    w = gram_schmidt_ortho(Q, w)

    alphas = [alpha]
    betas: list = []

    for _ in range(2, m + 1):
        beta = div = torch.norm(w, dim=-1)
        while (div < EPS).any():
            idx = beta < EPS
            w_new = F.normalize(torch.randn_like(w), dim=-1)
            w_new = gram_schmidt_ortho(Q, w_new)
            w = w * ~idx.unsqueeze(-1) + w_new * idx.unsqueeze(-1)
            div = torch.norm(w, dim=-1)
        v = F.normalize(w, dim=-1)
        vecs.append(v)
        Q = torch.stack(vecs, dim=1)
        w = hvp_fun(v)
        alpha = torch.einsum("bi,bi->b", w, v)
        w = gram_schmidt_ortho(Q, w)
        alphas.append(alpha)
        betas.append(beta)

    alphas_t = torch.stack(alphas, dim=-1)
    betas_t = torch.stack(betas, dim=-1)
    T = (
        torch.diag_embed(betas_t, offset=-1)
        + torch.diag_embed(alphas_t, offset=0)
        + torch.diag_embed(betas_t, offset=1)
    )
    return T


def stochastic_quadrature(T: torch.Tensor, dim: int, func=torch.log) -> torch.Tensor:
    eigvals, eigvecs = _eigh(T)
    clamped = eigvals.clamp(min=0).detach() + (eigvals - eigvals.detach())
    tau = eigvecs[..., 0, :]
    return torch.sum(tau * tau * func(clamped + 1e-8), dim=-1) * dim


def stochastic_lanczos_quadrature(hvp_fun, v: torch.Tensor, m: int, func=torch.log) -> torch.Tensor:
    bsz, dim = v.shape
    T = lanczos_tridiagonalization(hvp_fun, m, v)
    return stochastic_quadrature(T, dim, func=func)


def batch_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze(2)


def conjugate_gradient(hvp, b, m: int = 10, rtol: float = 0.0, atol: float = 1e-3):
    x = b.clone().detach()
    r = b - hvp(x)
    tol = atol + rtol * torch.abs(x)
    if (torch.abs(r) < tol).all():
        CG_ITERS_TRACER.append(0)
        return x
    p = r
    r2 = batch_dot_product(r, r)
    k = 0
    while k < m:
        k += 1
        Ap = hvp(p)
        a = r2 / (batch_dot_product(p, Ap) + 1e-8)
        x = x + a * p
        r = r - a * Ap
        tol = atol + rtol * torch.abs(x)
        if (torch.abs(r) < tol).all():
            break
        r2_new = batch_dot_product(r, r)
        beta = r2_new / r2
        r2 = r2_new
        p = r + beta * p
    CG_ITERS_TRACER.append(k)
    return x


def stochastic_logdet_gradient_estimator(hvp_fun, v, m, rtol: float = 0.0, atol: float = 1e-3):
    with torch.no_grad():
        v_Hinv = conjugate_gradient(hvp_fun, v, m, rtol=rtol, atol=atol)
    surrog_logdet = torch.sum(hvp_fun(v_Hinv) * v, dim=1)
    return surrog_logdet


def unbiased_logdet(hvp_fun, v, p: float = 0.1, n_exact_terms: int = 4):
    bsz, dim = v.shape
    m = int(np.random.geometric(p)) + n_exact_terms

    def coeff_fn(kk):
        if kk <= n_exact_terms:
            return 1.0
        return 1.0 / ((1 - p) ** max(kk - n_exact_terms - 1, 0))

    T = lanczos_tridiagonalization(hvp_fun, m, v)
    estimate = 0.0
    prev = 0.0
    for k in range(n_exact_terms, m + 1):
        cur = stochastic_quadrature(T[:, :k, :k], dim)
        estimate = estimate + coeff_fn(k) * (cur - prev)
        prev = cur
    return estimate
