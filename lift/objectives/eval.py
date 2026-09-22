"""Eval-time helpers: 1D TV grid and a numpy-friendly ``log p`` callable.

These are diagnostics, not part of the training algorithm proper.
The TV grid is a 1D-only convenience; for :math:`D \\ge 2` the
held-out negative log-likelihood (returned by the trainer's
``val_loss_per_eval`` history channel) is the relevant metric and
scales to any dimension.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch

from lift.models import HyperNetwork, LogConcaveEBM
from lift.models.conditional_sampler_flow import ConditionalSamplerFlow
from lift.objectives.mixture import mixture_log_density
from lift.objectives.partition import log_Z_via_flow_is


def build_eval_grid(
    target,
    n_axis: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Build a 1D evaluation grid for the in-loop TV diagnostic.

    Returns ``(grid_pts, log_p_grid, dvol)`` where ``grid_pts`` is a
    ``(N, 1)`` tensor of evaluation points, ``log_p_grid`` is the
    target log-density at those points, and ``dvol`` is the per-cell
    volume (cell width). Raises for :math:`D \\ge 2`: the in-loop TV
    grid is a 1D-only convenience.
    """
    D = int(getattr(target, "D", 1))
    if D != 1:
        raise NotImplementedError(
            f"in-loop TV grid is 1D-only; got D={D}. "
            "Use held-out -log q (n_batch_val > 0) for D >= 2."
        )
    lo, hi = target.xlim
    xs = torch.linspace(lo, hi, int(n_axis), device=device).unsqueeze(-1)
    dvol = float((hi - lo) / (int(n_axis) - 1))
    log_p = target.log_prob(xs)
    return xs, log_p, dvol


def hypernet_ebm_logp(
    ebm: LogConcaveEBM,
    hyper_E: HyperNetwork,
    flow: ConditionalSamplerFlow,
    K: int,
    D: int,
    D_aug: int,
    *,
    target_sample: torch.Tensor,
    M_logZ: int,
    device: torch.device,
    log_Z_override: Optional[torch.Tensor] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Numpy-friendly ``log p(x)`` callable for the trained model.

    ``log Z`` is computed once via the flow IS estimator and cached.
    The returned callable evaluates the log-density at arbitrary query
    points (any ``D``, no quadrature, no grid).
    """
    with torch.no_grad():
        params_E = hyper_E(target_sample.to(device))
    log_pi = torch.log_softmax(params_E["log_pi"], dim=-1).detach()
    if log_Z_override is not None:
        log_Z = log_Z_override.to(
            device=device, dtype=next(hyper_E.parameters()).dtype,
        ).detach()
    else:
        log_Z = log_Z_via_flow_is(
            ebm, params_E, flow,
            K=K, D=D, D_aug=D_aug, M_logZ=int(M_logZ),
            device=device,
            dtype=next(hyper_E.parameters()).dtype,
        )

    def _logp(x_np, *, chunk=8192):
        x_in = np.asarray(x_np, dtype=np.float64)
        # Output shape: drop the last dim if it carries D coordinates,
        # else match input (1-D-input grid case).
        if x_in.ndim > 1:
            out_shape = x_in.shape[:-1]
            x_flat = x_in.reshape(-1, D)
        else:
            out_shape = (x_in.size,)
            x_flat = x_in.reshape(-1, 1)
        N = x_flat.shape[0]
        out = np.empty(N, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x_t = torch.from_numpy(x_flat[s:e]).to(
                    device=device, dtype=log_pi.dtype,
                )
                out[s:e] = mixture_log_density(
                    ebm, params_E, log_pi, log_Z, K=K, x=x_t,
                ).cpu().numpy()
        return out.reshape(out_shape)

    return _logp
