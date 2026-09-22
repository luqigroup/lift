"""Two 1-D asymmetric log-concave targets.

:class:`SkewNormal1D` is the skew-normal ``SN(xi, omega, alpha)``, with
density ``p(x) = (2 / omega) phi(z) Phi(alpha z)``, ``z = (x - xi) / omega``
(Azzalini, 1985). It is a product of two log-concave functions, hence
log-concave for every ``alpha``, smooth, and standard normal at
``alpha = 0``; ``alpha`` controls a cliff-like perturbation near ``xi``.

:class:`AsymmetricGaussian1D` is piecewise quadratic:
``p(x) propto exp(-c_- x^2 / 2)`` for ``x < 0`` and ``exp(-c_+ x^2 / 2)``
for ``x >= 0``, with curvature ratio ``delta = c_- / c_+ >= 1``. Its
asymmetry is the ratio of the directional second derivatives of
``-log p`` at the mode, which the skew-normal does not control.

Both follow the target interface of :mod:`lift.dataset.gumbel`:
``log_prob`` (torch), ``log_prob_np`` (numpy), ``sample(n, device)``, and
an ``xlim`` window for grid-based metric integration.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm as _scipy_norm


_LOG_2PI = math.log(2.0 * math.pi)


class SkewNormal1D:
    """1-D skew-normal ``SN(xi, omega, alpha)``.

    Parameters
    ----------
    xi : float
        Location parameter; the mode location at ``alpha = 0``.
    omega : float
        Scale parameter, positive.
    alpha : float
        Shape parameter. ``alpha = 0`` is standard normal, and larger
        ``|alpha|`` skews the density further.
    xlim : tuple of float, optional
        Plotting and quadrature window. ``None`` selects an asymmetric
        default covering the body of the distribution at this ``alpha``.
    """

    name = "skew-normal"
    D = 1

    def __init__(
        self,
        xi: float = 0.0,
        omega: float = 1.0,
        alpha: float = 0.0,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if not omega > 0.0:
            raise ValueError(f"omega must be positive; got {omega!r}")
        self.xi = float(xi)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.xlim = xlim if xlim is not None else self._default_xlim()

    def _default_xlim(self) -> tuple[float, float]:
        # Body of SN at scale omega extends ~5 stddevs each side at
        # alpha=0; the skewed tail extends further as |alpha| grows.
        lo_pad = 5.0 + max(0.0, -0.6 * self.alpha)
        hi_pad = 5.0 + max(0.0, 0.6 * self.alpha)
        return (
            self.xi - lo_pad * self.omega,
            self.xi + hi_pad * self.omega,
        )

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.xi) / self.omega
        log_phi = -0.5 * _LOG_2PI - 0.5 * z * z
        log_Phi = _scipy_norm.logcdf(self.alpha * z)
        return math.log(2.0) - math.log(self.omega) + log_phi + log_Phi

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        z = (x - self.xi) / self.omega
        log_phi = -0.5 * _LOG_2PI - 0.5 * z * z
        log_Phi = torch.special.log_ndtr(self.alpha * z)
        return math.log(2.0) - math.log(self.omega) + log_phi + log_Phi

    def sample(
        self, n: int, device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        # Azzalini's stochastic representation:
        # X = xi + omega * (delta * |Z0| + sqrt(1 - delta^2) * Z1)
        delta = self.alpha / math.sqrt(1.0 + self.alpha * self.alpha)
        z0 = torch.randn(int(n), device=device).abs()
        z1 = torch.randn(int(n), device=device)
        x = self.xi + self.omega * (
            delta * z0 + math.sqrt(max(0.0, 1.0 - delta * delta)) * z1
        )
        return x.unsqueeze(-1)


class AsymmetricGaussian1D:
    """Piecewise-quadratic log-concave density.

    .. math::

        p(x) \\;\\propto\\;
        \\begin{cases}
            \\exp(-c_- \\, x^2 / 2), & x < 0 \\\\
            \\exp(-c_+ \\, x^2 / 2), & x \\ge 0,
        \\end{cases}

    with normalizing constant
    :math:`Z = \\sqrt{\\pi/2}\\,\\bigl(1/\\sqrt{c_+} + 1/\\sqrt{c_-}\\bigr)`.

    The density and the score are continuous at ``x = 0``; the
    discontinuity is in the second derivative, which is :math:`c_-` to the
    left of the mode and :math:`c_+` to its right.

    Parameters
    ----------
    delta : float
        Asymmetry ratio :math:`\\delta = c_- / c_+ \\ge 1`.
    c_plus : float
        Right-side curvature :math:`c_+`. At the default ``1.0`` the right
        half is the standard half-normal.
    xlim : tuple of float, optional
        Plotting and quadrature window, by default ``5/sqrt(c)`` per side.
    """

    name = "asym-gaussian"
    D = 1

    def __init__(
        self,
        delta: float = 1.0,
        c_plus: float = 1.0,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if not float(delta) >= 1.0:
            raise ValueError(
                f"delta must be >= 1 (no asymmetry inversion); got "
                f"{delta!r}",
            )
        if not float(c_plus) > 0.0:
            raise ValueError(f"c_plus must be > 0; got {c_plus!r}")
        self.delta = float(delta)
        self.c_plus = float(c_plus)
        self.c_minus = float(delta) * float(c_plus)
        # log Z = log(sqrt(pi/2) (1/sqrt(c+) + 1/sqrt(c-)))
        z = (
            (1.0 / math.sqrt(self.c_plus))
            + (1.0 / math.sqrt(self.c_minus))
        )
        self.log_norm = -math.log(z) - 0.5 * math.log(math.pi / 2.0)
        # Auto xlim: 5 standard deviations on each side, where the
        # left "std" is 1/sqrt(c_-) and the right is 1/sqrt(c_+).
        lo = -5.0 / math.sqrt(self.c_minus)
        hi = 5.0 / math.sqrt(self.c_plus)
        self.xlim = xlim or (lo, hi)

    def log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.where(
            x < 0.0,
            -0.5 * self.c_minus * x * x,
            -0.5 * self.c_plus * x * x,
        )
        return out + self.log_norm

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1) if X.dim() == 2 else X
        out = torch.where(
            x < 0.0,
            -0.5 * self.c_minus * x.pow(2),
            -0.5 * self.c_plus * x.pow(2),
        )
        return out + self.log_norm

    def sample(
        self, n: int, device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        # Mixture of two half-normals with mass proportional to 1/sqrt(c).
        n = int(n)
        m_plus = 1.0 / math.sqrt(self.c_plus)
        m_minus = 1.0 / math.sqrt(self.c_minus)
        p_plus = m_plus / (m_plus + m_minus)
        plus_mask = torch.rand(n, device=device) < p_plus
        z = torch.randn(n, device=device).abs()
        x = torch.where(
            plus_mask,
            z / math.sqrt(self.c_plus),
            -z / math.sqrt(self.c_minus),
        )
        return x.unsqueeze(-1)
