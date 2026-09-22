"""1-D log-concave targets for the multi-target ICNN-EBM gallery.

Four target classes, each with concave :math:`\\log p`, each exposing the
target interface used throughout ``lift``: ``log_prob`` (torch),
``log_prob_np`` (numpy), ``sample(n, device)`` (torch), and an ``xlim``
rectangle for grid-based metric integration. ``D = 1`` for all.

* :class:`Laplace1D` -- :math:`p(x) \\propto \\exp(-|x - \\mu| / b)`.
* :class:`Gamma1D` -- shape :math:`\\alpha \\ge 1` (log-concave iff
  :math:`\\alpha \\ge 1`), rate :math:`\\beta`. Support :math:`x > 0`.
* :class:`Beta1D` -- :math:`B(\\alpha, \\beta)` on :math:`(0, 1)`,
  log-concave iff :math:`\\alpha, \\beta \\ge 1`.
* :class:`HalfNormal1D` -- :math:`|\\mathcal N(0, \\sigma^2)|` on
  :math:`x \\ge 0`.
"""

from __future__ import annotations

import math

import numpy as np
import torch


_LOG_2PI = math.log(2.0 * math.pi)


class Laplace1D:
    """Laplace(:math:`\\mu, b`); log-concave for every :math:`b > 0`."""

    name = "laplace"
    D = 1

    def __init__(
        self,
        mu: float = 0.0,
        b: float = 1.0,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if not b > 0:
            raise ValueError(f"b must be > 0; got {b!r}")
        self.mu = float(mu)
        self.b = float(b)
        self.xlim = xlim or (self.mu - 8.0 * self.b, self.mu + 8.0 * self.b)
        self._log_norm = -math.log(2.0 * self.b)

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self._log_norm - np.abs(x - self.mu) / self.b

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        return self._log_norm - torch.abs(x - self.mu) / self.b

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        u = torch.rand(int(n), device=device) - 0.5
        x = self.mu - self.b * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))
        return x.unsqueeze(-1)


class Gamma1D:
    """Gamma(shape :math:`\\alpha`, rate :math:`\\beta`); log-concave iff :math:`\\alpha \\ge 1`."""

    name = "gamma"
    D = 1

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 1.0,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if not alpha >= 1.0:
            raise ValueError(
                f"alpha must be >= 1 for log-concavity; got {alpha!r}",
            )
        if not beta > 0:
            raise ValueError(f"beta must be > 0; got {beta!r}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        mean = self.alpha / self.beta
        std = math.sqrt(self.alpha) / self.beta
        self.xlim = xlim or (0.0, float(mean + 8.0 * std))
        self._log_norm = (
            self.alpha * math.log(self.beta) - math.lgamma(self.alpha)
        )

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            lp = (
                self._log_norm
                + (self.alpha - 1.0) * np.log(np.clip(x, 1e-300, None))
                - self.beta * x
            )
        return np.where(x > 0.0, lp, -np.inf)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        safe = torch.clamp(x, min=1e-30)
        lp = (
            self._log_norm
            + (self.alpha - 1.0) * torch.log(safe)
            - self.beta * x
        )
        return torch.where(
            x > 0.0,
            lp,
            torch.full_like(lp, float("-inf")),
        )

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        dist = torch.distributions.Gamma(
            concentration=torch.tensor(self.alpha, device=device),
            rate=torch.tensor(self.beta, device=device),
        )
        return dist.sample((int(n),)).unsqueeze(-1)


class Beta1D:
    """Beta(:math:`\\alpha, \\beta`) on :math:`(0, 1)`; log-concave iff both shape
    parameters are :math:`\\ge 1`."""

    name = "beta"
    D = 1

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 5.0,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if not (alpha >= 1.0 and beta >= 1.0):
            raise ValueError(
                f"alpha, beta must be >= 1 for log-concavity; got "
                f"alpha={alpha!r}, beta={beta!r}",
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.xlim = xlim or (0.0, 1.0)
        self._log_norm = (
            math.lgamma(self.alpha + self.beta)
            - math.lgamma(self.alpha)
            - math.lgamma(self.beta)
        )

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            lp = (
                self._log_norm
                + (self.alpha - 1.0) * np.log(np.clip(x, 1e-300, None))
                + (self.beta - 1.0) * np.log(np.clip(1.0 - x, 1e-300, None))
            )
        return np.where((x > 0.0) & (x < 1.0), lp, -np.inf)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        safe_x = torch.clamp(x, min=1e-30, max=1.0 - 1e-30)
        lp = (
            self._log_norm
            + (self.alpha - 1.0) * torch.log(safe_x)
            + (self.beta - 1.0) * torch.log1p(-safe_x)
        )
        return torch.where(
            (x > 0.0) & (x < 1.0),
            lp,
            torch.full_like(lp, float("-inf")),
        )

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        dist = torch.distributions.Beta(
            concentration1=torch.tensor(self.alpha, device=device),
            concentration0=torch.tensor(self.beta, device=device),
        )
        return dist.sample((int(n),)).unsqueeze(-1)


class HalfNormal1D:
    """Half-normal :math:`|\\mathcal N(0, \\sigma^2)|` on :math:`x \\ge 0`.

    Log-concave for every :math:`\\sigma > 0`.
    """

    name = "halfnormal"
    D = 1

    def __init__(
        self,
        sigma: float = 1.0,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if not sigma > 0:
            raise ValueError(f"sigma must be > 0; got {sigma!r}")
        self.sigma = float(sigma)
        self.xlim = xlim or (0.0, float(6.0 * self.sigma))
        # log p(x) = log(2 / sqrt(2 pi sigma^2)) - x^2 / (2 sigma^2)
        self._log_norm = (
            math.log(2.0) - 0.5 * (_LOG_2PI + 2.0 * math.log(self.sigma))
        )

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        lp = self._log_norm - 0.5 * (x / self.sigma) ** 2
        return np.where(x >= 0.0, lp, -np.inf)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        lp = self._log_norm - 0.5 * (x / self.sigma) ** 2
        return torch.where(
            x >= 0.0,
            lp,
            torch.full_like(lp, float("-inf")),
        )

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        z = torch.randn(int(n), device=device).abs() * self.sigma
        return z.unsqueeze(-1)
