"""SequentialFlow / ActNorm / DeepConvexFlow vendored from CP-Flow.

The implementation mirrors ``lib/flows/flows.py`` and
``lib/flows/cpflows.py`` from https://github.com/CW-Huang/CP-Flow,
restricted to the modules used in the tabular density-estimation
experiment (no image / coupling / NAF / SylvesterFlow).

Modifications from upstream:

* ``torch.symeig`` (deprecated) -> ``torch.linalg.eigh`` (handled inside
  ``lift.baselines.cpflow_vendor.logdet``).
* ``log_standard_normal`` imported from the sibling distribution
  module rather than ``lib.distributions``.
* ``DeepConvexFlow.get_potential`` accepts an optional
  ``icnn_override`` argument, a callable ``x -> ICNN(x)``, so the lift
  can supply functionally parameterized weights without subclassing.
"""

from __future__ import annotations

import gc
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lift.baselines.cpflow_vendor.distributions import log_standard_normal
from lift.baselines.cpflow_vendor.logdet import (
    stochastic_lanczos_quadrature,
    stochastic_logdet_gradient_estimator,
    unbiased_logdet,
)


_scaling_min = 0.001
#: Diagnostic sink for per-HVP Hessian-norm ratios. The list is
#: append-only and nothing clears it, so enabling the trace on a long
#: run grows it without bound. Off by default.
HESS_NORM_TRACE_ENABLED: bool = False
HESS_NORM_TRACER: list = []


# ---------------------------------------------------------------------------
# ActNorm. CP-Flow carries an extra (C, *, *) tensor shape; tabular data
# here is (B, num_features).
# ---------------------------------------------------------------------------
class ActNorm(nn.Module):
    """ActNorm with data-dependent init, faithful to CP-Flow."""

    def __init__(
        self,
        num_features: int,
        logscale_factor: float = 1.0,
        scale: float = 1.0,
        learn_scale: bool = True,
    ) -> None:
        super().__init__()
        self.initialized = False
        self.num_features = num_features
        self.register_parameter(
            "b", nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
        )
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter(
                "logs",
                nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
            )

    def forward_transform(self, x: torch.Tensor, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            def unsqueeze(t):
                return t.unsqueeze(0).unsqueeze(-1).detach()

            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(
                    torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size,
                )
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b
        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)
            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        scale = torch.exp(logs) + _scaling_min
        x = y / scale - self.b
        return x.view(input_shape)

    def extra_repr(self) -> str:
        return f"{self.num_features}"


class ActNormNoLogdet(ActNorm):
    """ActNorm without the log-det contribution (used inside ICNNs)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward_transform(x)[0]


# ---------------------------------------------------------------------------
# DeepConvexFlow (CP-Flow's main block).
# ---------------------------------------------------------------------------
class DeepConvexFlow(nn.Module):
    """One convex-potential block ``x -> grad_x F(x)``.

    ``F(x) = softplus(w1) * ICNN(x) + softplus(w0) * ||x||^2 / 2``.

    On training (or when ``no_bruteforce=True``), the log-det is
    estimated via stochastic CG / Lanczos; on eval with
    ``no_bruteforce=False`` an exact log-det is computed by building
    the full Hessian.
    """

    def __init__(
        self,
        icnn: nn.Module,
        dim: int,
        unbiased: bool = False,
        no_bruteforce: bool = True,
        m1: int = 10,
        m2: Optional[int] = None,
        rtol: float = 0.0,
        atol: float = 1e-3,
        bias_w1: float = 0.0,
        trainable_w0: bool = True,
    ) -> None:
        super().__init__()
        if m2 is None:
            m2 = dim
        self.icnn = icnn
        self.no_bruteforce = no_bruteforce
        self.rtol = rtol
        self.atol = atol
        self.w0 = nn.Parameter(
            torch.log(torch.exp(torch.ones(1)) - 1), requires_grad=trainable_w0,
        )
        self.w1 = nn.Parameter(torch.zeros(1) + bias_w1)
        self.bias_w1 = bias_w1
        self.m1, self.m2 = m1, m2
        self.stochastic_estimate_fn = (
            unbiased_logdet
            if unbiased
            else partial(stochastic_lanczos_quadrature, m=min(m1, dim))
        )
        self.stochastic_grad_estimate_fn = partial(
            stochastic_logdet_gradient_estimator,
            m=min(m2, dim),
            rtol=self.rtol,
            atol=self.atol,
        )

    def _icnn_eval(self, x: torch.Tensor, icnn_override: Optional[Callable] = None) -> torch.Tensor:
        if icnn_override is not None:
            return icnn_override(x)
        return self.icnn(x)

    def get_potential(
        self,
        x: torch.Tensor,
        context=None,
        icnn_override: Optional[Callable] = None,
    ) -> torch.Tensor:
        n = x.size(0)
        icnn_out = self._icnn_eval(x, icnn_override=icnn_override)
        return (
            F.softplus(self.w1) * icnn_out
            + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2
        )

    def reverse(
        self,
        y: torch.Tensor,
        max_iter: int = 1000000,
        lr: float = 1.0,
        tol: float = 1e-12,
        x: Optional[torch.Tensor] = None,
        context=None,
        icnn_override: Optional[Callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        if x is None:
            x = y.clone().detach().requires_grad_(True)

        def closure():
            Fval = self.get_potential(x, context, icnn_override=icnn_override)
            loss = torch.sum(Fval) - torch.sum(x * y)
            x.grad = torch.autograd.grad(loss, x)[0].detach()
            return loss

        optimizer = torch.optim.LBFGS(
            [x],
            lr=lr,
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_grad=tol,
            tolerance_change=tol,
        )
        optimizer.step(closure)
        return x

    def forward(
        self, x: torch.Tensor, context=None, icnn_override: Optional[Callable] = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Fval = self.get_potential(x, context, icnn_override=icnn_override)
            f = torch.autograd.grad(Fval.sum(), x, create_graph=True)[0]
        return f

    def forward_transform(
        self,
        x: torch.Tensor,
        logdet=0,
        context=None,
        extra=None,
        icnn_override: Optional[Callable] = None,
    ):
        # ``force_bruteforce``, absent by default, overrides the train-time
        # preference for the stochastic estimator, which separates a failure
        # of the model from a failure of the log-det estimator. Without the
        # attribute this branch is exactly as released.
        if (self.training or self.no_bruteforce) and not getattr(
            self, "force_bruteforce", False
        ):
            return self.forward_transform_stochastic(
                x, logdet, context=context, extra=extra, icnn_override=icnn_override,
            )
        return self.forward_transform_bruteforce(
            x, logdet, context=context, icnn_override=icnn_override,
        )

    def forward_transform_stochastic(
        self,
        x: torch.Tensor,
        logdet=0,
        context=None,
        extra=None,
        icnn_override: Optional[Callable] = None,
    ):
        bsz, *dims = x.shape
        dim = int(np.prod(dims))
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Fval = self.get_potential(x, context, icnn_override=icnn_override)
            f = torch.autograd.grad(Fval.sum(), x, create_graph=True)[0]

            def hvp_fun(v):
                v = v.reshape(bsz, *dims)
                hvp = torch.autograd.grad(
                    f, x, v, create_graph=self.training, retain_graph=True,
                )[0]
                if HESS_NORM_TRACE_ENABLED:
                    HESS_NORM_TRACER.append(
                        (torch.norm(hvp)
                         / (torch.norm(v) + 1e-12)).detach().cpu(),
                    )
                hvp = hvp.reshape(bsz, dim)
                return hvp

        if self.training:
            v1 = _sample_rademacher(bsz, dim).to(x)
            est1 = self.stochastic_grad_estimate_fn(hvp_fun, v1)
        else:
            est1 = 0

        if not self.training or (extra is not None and len(extra) > 0):
            try:
                v2 = F.normalize(_sample_rademacher(bsz, dim), dim=-1).to(f)
                est2 = self.stochastic_estimate_fn(hvp_fun, v2)
            except Exception:
                est2 = torch.zeros_like(logdet).fill_(float("nan")) if isinstance(logdet, torch.Tensor) else torch.zeros(bsz).fill_(float("nan"))
            if extra is not None and len(extra) > 0:
                extra[0] = extra[0] + est2.detach()
        else:
            est2 = 0

        return f, logdet + est1 if self.training else logdet + est2

    def forward_transform_bruteforce(
        self,
        x: torch.Tensor,
        logdet=0,
        context=None,
        icnn_override: Optional[Callable] = None,
    ):
        bsz = x.shape[0]
        input_shape = x.shape[1:]
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Fval = self.get_potential(x, context, icnn_override=icnn_override)
            f = torch.autograd.grad(Fval.sum(), x, create_graph=True)[0]
            f_flat = f.reshape(bsz, -1)
            H_rows = []
            for i in range(f_flat.shape[1]):
                retain = self.training or (i < f_flat.shape[1] - 1)
                H_rows.append(
                    torch.autograd.grad(
                        f_flat[:, i].sum(),
                        x,
                        create_graph=self.training,
                        retain_graph=retain,
                    )[0].reshape(bsz, -1)
                )
            H = torch.stack(H_rows, dim=1)  # (B, D, D)
            # The Hessian is symmetric in exact arithmetic; the autograd
            # output can be slightly non-symmetric.
            H = 0.5 * (H + H.transpose(-2, -1))
            sign, log_abs = torch.linalg.slogdet(H)
            log_det = log_abs
        return f, logdet + log_det


# ---------------------------------------------------------------------------
# SequentialFlow (composition of flows; logp = log_N(z) + sum_blocks logdet).
# ---------------------------------------------------------------------------
class SequentialFlow(nn.Module):
    def __init__(self, flows: List[nn.Module]) -> None:
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward_transform(self, x: torch.Tensor, logdet=0, context=None, extra=None):
        for flow in self.flows:
            if isinstance(flow, DeepConvexFlow):
                x, logdet = flow.forward_transform(
                    x, logdet, context=context, extra=extra,
                )
            else:
                prev_logdet = logdet
                x, logdet = flow.forward_transform(x, logdet)
                if extra is not None and len(extra) > 0:
                    extra[0] = extra[0] + (logdet - prev_logdet).detach()
        return x, logdet

    def reverse(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for flow in self.flows[::-1]:
            x = flow.reverse(x, **kwargs)
        return x

    def logp(self, x: torch.Tensor, context=None, extra=None) -> torch.Tensor:
        z, logdet = self.forward_transform(x, context=context, extra=extra)
        logp0 = log_standard_normal(z).sum(-1)
        if extra is not None and len(extra) > 0:
            extra[0] = extra[0] + logp0.detach()
        return logp0 + logdet


def _sample_rademacher(*shape) -> torch.Tensor:
    return (torch.randint(0, 2, shape, dtype=torch.float32) * 2 - 1)
