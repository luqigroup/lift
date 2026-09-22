"""1-D Gumbel and bimodal-Gumbel targets for the paper experiments.

Both classes expose the standard target interface used throughout
this package: ``log_prob`` (torch), ``log_prob_np`` (numpy),
``sample(n, device)`` (torch), and an ``xlim`` rectangle for grid-based
metric integration. Gumbel is smooth, log-concave, and modestly
asymmetric (left tail decays as ``exp(x)``, right as ``exp(-x)``); the
bimodal version places two such modes at ``+/- centre``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class Gumbel1D:
    """1-D Gumbel: ``p(x) ∝ exp(-(x - mu) - exp(-(x - mu)))``."""

    name = "gumbel"
    D = 1

    def __init__(self, mu: float = 0.0,
                 xlim: tuple[float, float] = (-5.0, 12.0)) -> None:
        self.mu = float(mu)
        self.xlim = xlim

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64) - self.mu
        return -x - np.exp(-x)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        x = x - self.mu
        return -x - torch.exp(-x)

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        u = torch.rand(n, device=device).clamp(min=1e-9, max=1.0 - 1e-9)
        return (self.mu - torch.log(-torch.log(u))).unsqueeze(-1)


class BimodalGumbel:
    """Mixture of two Gumbels at :math:`\\pm` ``centre``.

    ``weight`` is the mass of the left component (at :math:`-c`); the
    right component (at :math:`+c`) carries ``1 - weight``.
    """

    name = "bimodal-gumbel"
    D = 1

    def __init__(
        self,
        centre: float = 3.0,
        xlim: tuple[float, float] | None = None,
        weight: float = 0.5,
    ) -> None:
        self.centre = float(centre)
        self.weight = float(weight)
        if not 0.0 < self.weight < 1.0:
            raise ValueError(
                f"weight must lie in (0, 1); got {self.weight!r}",
            )
        self.xlim = xlim or (-(centre + 5.0), centre + 5.0)

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        l = -(x + self.centre) - np.exp(-(x + self.centre))
        r = -(x - self.centre) - np.exp(-(x - self.centre))
        return np.logaddexp(
            l + math.log(self.weight),
            r + math.log(1.0 - self.weight),
        )

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        l = -(x + self.centre) - torch.exp(-(x + self.centre))
        r = -(x - self.centre) - torch.exp(-(x - self.centre))
        return torch.logaddexp(
            l + math.log(self.weight),
            r + math.log(1.0 - self.weight),
        )

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        m = torch.rand(n, device=device) < self.weight
        u = torch.rand(n, device=device).clamp(min=1e-9, max=1.0 - 1e-9)
        g = -torch.log(-torch.log(u))
        x = torch.where(m, g - self.centre, g + self.centre)
        return x.unsqueeze(-1)


class GumbelMD:
    """Product of ``D`` independent 1-D Gumbels with shared offset ``mu``.

    log p(x) = sum_d (-z_d - exp(-z_d)) with z_d = x_d - mu; sampling is
    a D-fold inverse-CDF transform of independent uniforms.
    """

    name = "gumbel-md"

    def __init__(
        self,
        D: int,
        mu: float = 0.0,
        xlim: tuple[float, float] = (-5.0, 12.0),
    ) -> None:
        self.D = int(D)
        self.mu = float(mu)
        self.xlim = xlim                  # per-dim bounds, used for variance estimates

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        z = X - self.mu                                     # (..., D)
        return (-z - torch.exp(-z)).sum(dim=-1)             # (...,)

    def sample(
        self, n: int, device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        u = torch.rand(int(n), self.D, device=device).clamp(
            min=1e-9, max=1.0 - 1e-9,
        )
        return self.mu - torch.log(-torch.log(u))           # (n, D)


class BimodalGumbelMD:
    """Mixture of two product-Gumbel components at :math:`\\pm c\\,\\bm{1}`.

    Each component is a product of ``D`` independent 1-D Gumbels with
    per-dim offset :math:`-c` (left) or :math:`+c` (right); ``weight``
    is the mass on the left component.
    """

    name = "bimodal-gumbel-md"

    def __init__(
        self,
        D: int,
        centre: float = 4.0,
        xlim: tuple[float, float] | None = None,
        weight: float = 0.5,
    ) -> None:
        self.D = int(D)
        self.centre = float(centre)
        self.weight = float(weight)
        if not 0.0 < self.weight < 1.0:
            raise ValueError(
                f"weight must lie in (0, 1); got {self.weight!r}",
            )
        self.xlim = xlim or (-(centre + 5.0), centre + 5.0)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        zl = X + self.centre
        zr = X - self.centre
        log_l = (-zl - torch.exp(-zl)).sum(dim=-1)          # (...,)
        log_r = (-zr - torch.exp(-zr)).sum(dim=-1)          # (...,)
        return torch.logaddexp(
            log_l + math.log(self.weight),
            log_r + math.log(1.0 - self.weight),
        )

    def sample(
        self, n: int, device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        m = torch.rand(int(n), device=device) < self.weight  # (n,)
        u = torch.rand(int(n), self.D, device=device).clamp(
            min=1e-9, max=1.0 - 1e-9,
        )
        g = -torch.log(-torch.log(u))                        # (n, D)
        offsets = torch.where(
            m.unsqueeze(-1),
            torch.full_like(g, -self.centre),
            torch.full_like(g, self.centre),
        )
        return g + offsets                                   # (n, D)
