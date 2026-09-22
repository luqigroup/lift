"""Eight two-dimensional benchmark targets, registered by name.

Two-Moons, Pinwheel and Checkerboard are sample-only: their ``log_prob``
is a Gaussian-kernel KDE on a cached fine sample. The other five --
Eight-Gaussians, SkewedMode, GumbelMixture2D, GammaMode2D and
AsymmetricLaplace2D -- have an exact log-density, so they can also be
trained against a density rather than samples.

Each class follows the target interface of :mod:`lift.dataset.gumbel`:
class attributes ``D = 2`` and ``name`` (the registry key), an instance
attribute ``xlim`` holding ``((lo_x, hi_x), (lo_y, hi_y))`` for grid-based
metric integration, ``log_prob(X)`` mapping ``(..., 2)`` to ``(...,)``, and
``sample(n, device)`` returning float32 ``(n, 2)``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


_LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------- KDE

class _KDESurrogate:
    """Gaussian-kernel KDE used as a sample-only ``log_prob`` surrogate.

    Samples are stored on CPU and moved to ``x.device`` on each call. The
    bandwidth is fixed; the caller chooses it.
    """

    def __init__(self, samples: torch.Tensor, bandwidth: float = 0.20) -> None:
        if samples.dim() != 2 or samples.shape[-1] != 2:
            raise ValueError(
                f"_KDESurrogate expects (M, 2) samples; got {tuple(samples.shape)}",
            )
        self.samples = samples.detach().cpu()
        self.h = float(bandwidth)
        self.D = int(samples.shape[-1])
        self.log_norm = (
            -0.5 * self.D * _LOG_2PI - self.D * math.log(self.h)
            - math.log(int(samples.shape[0]))
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diffs = x.unsqueeze(-2) - self.samples.to(x.device).unsqueeze(0)
        log_kern = -0.5 * (diffs.pow(2).sum(dim=-1)) / (self.h * self.h)
        return self.log_norm + torch.logsumexp(log_kern, dim=-1)


def _kde_log_prob(target, x: torch.Tensor) -> torch.Tensor:
    """Lazily cache a KDE surrogate on the target instance."""
    cache = getattr(target, "_kde", None)
    if cache is None:
        s = target.sample(8192, device=x.device)
        cache = _KDESurrogate(s, bandwidth=getattr(target, "kde_bandwidth", 0.20))
        target._kde = cache
    return cache.log_prob(x)


# ------------------------------------------------------------------- targets

class TwoMoons:
    """Two-moons crescents -- the canonical NF-paper benchmark."""

    name = "two-moons"
    D = 2

    def __init__(
        self,
        noise: float = 0.05,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
        kde_bandwidth: float = 0.15,
    ) -> None:
        self.noise = float(noise)
        self.xlim = xlim or ((-2.5, 2.5), (-2.5, 2.5))
        self.kde_bandwidth = float(kde_bandwidth)

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n_per = int(n) // 2
        u1 = torch.rand(n_per) * math.pi
        c1 = torch.stack([torch.cos(u1), torch.sin(u1)], dim=-1)
        u2 = torch.rand(int(n) - n_per) * math.pi
        c2 = torch.stack([1.0 - torch.cos(u2), 0.5 - torch.sin(u2)], dim=-1)
        x = torch.cat([c1, c2], dim=0)
        x = (x - x.mean(0)) / x.std(0)
        x = x + self.noise * torch.randn_like(x)
        return x.to(device).float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return _kde_log_prob(self, x)


class Pinwheel:
    """Pinwheel of warped Gaussians (Goyal et al., NeurIPS 2017)."""

    name = "pinwheel"
    D = 2

    def __init__(
        self,
        n_classes: int = 5,
        radial_std: float = 0.3,
        tangential_std: float = 0.1,
        rate: float = 0.25,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
        kde_bandwidth: float = 0.15,
    ) -> None:
        self.n_classes = int(n_classes)
        self.radial_std = float(radial_std)
        self.tangential_std = float(tangential_std)
        self.rate = float(rate)
        self.xlim = xlim or ((-2.5, 2.5), (-2.5, 2.5))
        self.kde_bandwidth = float(kde_bandwidth)

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        rng = np.random.default_rng()
        n = int(n)
        n_per_class = [n // self.n_classes] * self.n_classes
        n_per_class[-1] += n - sum(n_per_class)
        rads = np.linspace(0, 2 * np.pi, self.n_classes, endpoint=False)
        feats = np.concatenate([
            rng.normal(loc=[1, 0],
                       scale=[self.radial_std, self.tangential_std],
                       size=(npc, 2))
            for npc in n_per_class
        ], axis=0)
        labels = np.concatenate(
            [np.full(npc, i) for i, npc in enumerate(n_per_class)],
        )
        angles = rads[labels] + self.rate * np.exp(feats[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles),
             np.sin(angles), np.cos(angles)],
            axis=-1,
        ).reshape(-1, 2, 2)
        x = np.einsum("bij,bj->bi", rotations, feats)
        x = (x - x.mean(0)) / x.std(0)
        return torch.from_numpy(x).float().to(device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return _kde_log_prob(self, x)


class Checkerboard:
    """Checkerboard of disjoint axis-aligned squares."""

    name = "checkerboard"
    D = 2

    def __init__(
        self,
        scale: float = 0.7,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
        kde_bandwidth: float = 0.10,
    ) -> None:
        self.scale = float(scale)
        self.xlim = xlim or ((-2.5, 2.5), (-2.5, 2.5))
        self.kde_bandwidth = float(kde_bandwidth)

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n = int(n)
        x1 = torch.rand(n) * 4 - 2
        x2_ = torch.rand(n) - torch.randint(0, 2, (n,)).float() * 2
        x2 = x2_ + (torch.floor(x1) % 2)
        x = torch.stack([x1, x2], dim=-1) * self.scale
        return x.to(device).float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return _kde_log_prob(self, x)


class EightGaussians:
    """Eight isotropic Gaussian components on a circle. Exact log_prob."""

    name = "8-gaussians"
    D = 2

    def __init__(
        self,
        scale: float = 2.0,
        std: float = 0.2,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.scale = float(scale)
        self.std = float(std)
        self.centres = torch.tensor(
            [[math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)]
             for i in range(8)],
            dtype=torch.float32,
        ) * self.scale
        pad = 4.0 * self.std
        rng = self.scale + pad
        self.xlim = xlim or ((-rng, rng), (-rng, rng))

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n = int(n)
        idx = torch.randint(0, 8, (n,))
        means = self.centres[idx]
        x = means + self.std * torch.randn(n, 2)
        return x.to(device).float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        centres = self.centres.to(device)
        diffs = x.unsqueeze(-2) - centres                       # (..., 8, 2)
        log_comp = -0.5 * diffs.pow(2).sum(dim=-1) / (self.std ** 2)
        log_comp = log_comp - 0.5 * 2 * math.log(2 * math.pi * self.std ** 2)
        log_pi = -math.log(8.0)
        return torch.logsumexp(log_pi + log_comp, dim=-1)


class SkewedMode:
    """Two skew-Gaussian modes with asymmetric log-densities.

    Each component's log-density is a sum of two log-concave pieces, a
    Gaussian and ``log_ndtr``, so every mode is log-concave while its body
    is sharply asymmetric: a single Gaussian must trade the left cliff off
    against the right tail.
    """

    name = "skewed"
    D = 2

    def __init__(
        self,
        sigma: float = 0.6,
        separation: float = 3.0,
        alpha_pair: tuple[float, float] = (4.0, -4.0),
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.sigma = float(sigma)
        s = float(separation) / 2.0
        self.centres = torch.tensor(
            [[-s, 0.0], [s, 0.0]], dtype=torch.float32,
        )
        self.alpha = torch.tensor(list(alpha_pair), dtype=torch.float32)
        pad_x = 5.0 * self.sigma + s
        pad_y = 5.0 * self.sigma
        self.xlim = xlim or ((-pad_x, pad_x), (-pad_y, pad_y))

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n = int(n)
        idx = torch.randint(0, 2, (n,))
        delta = self.alpha / torch.sqrt(1.0 + self.alpha.pow(2))
        d = delta[idx]
        z1 = torch.randn(n).abs()
        z2 = torch.randn(n)
        x_axis = self.centres[idx, 0] + self.sigma * (
            d * z1 + torch.sqrt((1.0 - d.pow(2)).clamp(min=0.0)) * z2
        )
        y_axis = self.centres[idx, 1] + self.sigma * torch.randn(n)
        return torch.stack([x_axis, y_axis], dim=-1).to(device).float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        centres = self.centres.to(device)
        alpha = self.alpha.to(device)
        out = []
        for k in range(2):
            xi = (x[..., 0] - centres[k, 0]) / self.sigma
            yi = (x[..., 1] - centres[k, 1]) / self.sigma
            log_phi_x = -0.5 * xi.pow(2) - 0.5 * _LOG_2PI
            log_phi_y = -0.5 * yi.pow(2) - 0.5 * _LOG_2PI
            log_Phi = torch.special.log_ndtr(alpha[k] * xi)
            out.append(
                log_phi_x + log_phi_y + log_Phi
                + math.log(2.0) - 2.0 * math.log(self.sigma),
            )
        log_comp = torch.stack(out, dim=-1)
        return torch.logsumexp(log_comp - math.log(2.0), dim=-1)


class GumbelMixture2D:
    """Two-mode Gumbel-Gaussian product mixture, with exact log_prob.

    Each mode has a Gumbel marginal on x, sharply asymmetric and
    log-concave, and a Gaussian marginal on y. One log-concave component
    absorbs such a mode, while a Gaussian mixture needs a component count
    growing with the accuracy demanded.
    """

    name = "gumbel-mix-2d"
    D = 2

    def __init__(
        self,
        separation: float = 3.0,
        sigma_y: float = 0.4,
        weight: float = 0.5,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        if not 0.0 < weight < 1.0:
            raise ValueError(f"weight must be in (0,1); got {weight!r}")
        self.sigma_y = float(sigma_y)
        self.weight = float(weight)
        s = float(separation) / 2.0
        self.centres = torch.tensor([[-s, 0.0], [s, 0.0]], dtype=torch.float32)
        # Gumbel body: roughly mu-2 to mu+8.
        self.xlim = xlim or ((-(s + 5.0), s + 8.0),
                             (-5.0 * self.sigma_y, 5.0 * self.sigma_y))

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n = int(n)
        # m=0 (left, centres[0]) with prob `weight`; m=1 (right) otherwise.
        # Convention matches :class:`BimodalGumbel`: ``weight`` is left mass.
        m = (torch.rand(n) >= self.weight).long()
        u = torch.rand(n).clamp(min=1e-9, max=1.0 - 1e-9)
        gumb = -torch.log(-torch.log(u))                        # (n,)
        x = self.centres[m, 0] + gumb
        y = self.centres[m, 1] + self.sigma_y * torch.randn(n)
        return torch.stack([x, y], dim=-1).to(device).float()

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        centres = self.centres.to(device)
        out = []
        for k in range(2):
            xc = X[..., 0] - centres[k, 0]
            yc = X[..., 1] - centres[k, 1]
            log_px = -xc - torch.exp(-xc)                       # (...,)
            log_py = (
                -0.5 * (yc / self.sigma_y).pow(2)
                - 0.5 * math.log(2.0 * math.pi * self.sigma_y ** 2)
            )
            out.append(log_px + log_py)
        log_comp = torch.stack(out, dim=-1)
        log_pi = torch.tensor(
            [math.log(self.weight), math.log(1.0 - self.weight)],
            dtype=log_comp.dtype, device=device,
        )
        return torch.logsumexp(log_comp + log_pi, dim=-1)


class GammaMode2D:
    """Single Gamma-Gaussian mode, with exact log_prob.

    The x marginal is a shifted Gamma(``shape``, ``rate``), log-concave for
    ``shape >= 1``; the y marginal is Gaussian. ``origin_shift`` moves the
    support cutoff well outside the high-density region, so the boundary is
    benign for sampling.
    """

    name = "gamma-mode"
    D = 2

    def __init__(
        self,
        shape: float = 2.0,
        rate: float = 1.0,
        origin_shift: float = -3.0,
        sigma_y: float = 0.5,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        if not float(shape) >= 1.0:
            raise ValueError(
                f"shape must be >= 1 for log-concavity; got {shape!r}",
            )
        self.shape = float(shape)
        self.rate = float(rate)
        self.origin_shift = float(origin_shift)
        self.sigma_y = float(sigma_y)
        # Body: roughly (shift, shift + (shape-1)/rate + 6/rate).
        body = max(6.0 / self.rate, 1.0)
        self.xlim = xlim or (
            (self.origin_shift - 1.0, self.origin_shift + body + 4.0),
            (-4.0 * self.sigma_y, 4.0 * self.sigma_y),
        )

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        gamma = torch.distributions.Gamma(self.shape, self.rate).sample((int(n),))
        x = self.origin_shift + gamma
        y = self.sigma_y * torch.randn(int(n))
        return torch.stack([x, y], dim=-1).to(device).float()

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0] - self.origin_shift
        y = X[..., 1]
        log_const = self.shape * math.log(self.rate) - math.lgamma(self.shape)
        log_px = torch.where(
            x > 0,
            (self.shape - 1.0) * x.clamp(min=1e-12).log()
            - self.rate * x + log_const,
            torch.full_like(x, -1e10),
        )
        log_py = (
            -0.5 * (y / self.sigma_y).pow(2)
            - 0.5 * math.log(2.0 * math.pi * self.sigma_y ** 2)
        )
        return log_px + log_py


class AsymmetricLaplace2D:
    """Two-mode mixture of asymmetric-Laplace x marginals + Gaussian y."""

    name = "asymmetric-laplace-2d"
    D = 2

    def __init__(
        self,
        alpha: float = 4.0,
        beta: float = 1.0,
        separation: float = 3.0,
        sigma_y: float = 0.4,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        if not (alpha > 0.0 and beta > 0.0):
            raise ValueError(
                f"alpha, beta must be positive; got {alpha!r}, {beta!r}",
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma_y = float(sigma_y)
        s = float(separation) / 2.0
        self.centres = torch.tensor([[-s, 0.0], [s, 0.0]], dtype=torch.float32)
        # Per-mode normalizer: the unnormalized pdf integrates to
        # 1/alpha + 1/beta.
        self.log_norm = -math.log(1.0 / self.alpha + 1.0 / self.beta)
        pad_x = max(8.0 / self.alpha, 8.0 / self.beta) + s
        self.xlim = xlim or (
            (-pad_x, pad_x), (-5.0 * self.sigma_y, 5.0 * self.sigma_y),
        )

    def _sample_one_mode(self, n: int) -> torch.Tensor:
        p_neg = (1.0 / self.beta) / (1.0 / self.alpha + 1.0 / self.beta)
        u = torch.rand(int(n))
        x = torch.empty(int(n))
        neg = u < p_neg
        x[neg] = (1.0 / self.beta) * torch.log(
            (u[neg] / p_neg).clamp(min=1e-12),
        )
        x[~neg] = -(1.0 / self.alpha) * torch.log(
            (1.0 - (u[~neg] - p_neg) / (1.0 - p_neg)).clamp(min=1e-12),
        )
        return x

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n = int(n)
        idx = torch.randint(0, 2, (n,))
        x_axis = self.centres[idx, 0] + self._sample_one_mode(n)
        y_axis = self.centres[idx, 1] + self.sigma_y * torch.randn(n)
        return torch.stack([x_axis, y_axis], dim=-1).to(device).float()

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        centres = self.centres.to(device)
        out = []
        for k in range(2):
            xc = X[..., 0] - centres[k, 0]
            yc = X[..., 1] - centres[k, 1]
            log_px = (
                torch.where(xc >= 0, -self.alpha * xc, self.beta * xc)
                + self.log_norm
            )
            log_py = (
                -0.5 * (yc / self.sigma_y).pow(2)
                - 0.5 * math.log(2.0 * math.pi * self.sigma_y ** 2)
            )
            out.append(log_px + log_py)
        log_comp = torch.stack(out, dim=-1)
        return torch.logsumexp(log_comp - math.log(2.0), dim=-1)


# Registry --------------------------------------------------------------------

TARGETS_2D: dict[str, type] = {
    TwoMoons.name: TwoMoons,
    Pinwheel.name: Pinwheel,
    Checkerboard.name: Checkerboard,
    EightGaussians.name: EightGaussians,
    SkewedMode.name: SkewedMode,
    GumbelMixture2D.name: GumbelMixture2D,
    GammaMode2D.name: GammaMode2D,
    AsymmetricLaplace2D.name: AsymmetricLaplace2D,
}


def get_target_2d(name: str, **kwargs):
    """Look up a 2-D target class by ``name`` and instantiate it.

    Raises ``ValueError`` if ``name`` is not registered. Keyword
    arguments are forwarded to the target's constructor.
    """
    if name not in TARGETS_2D:
        raise ValueError(
            f"unknown 2D target: {name!r}; pick from {sorted(TARGETS_2D)}",
        )
    return TARGETS_2D[name](**kwargs)


def has_exact_log_prob(name: str) -> bool:
    """Whether the target's ``log_prob`` is exact rather than a KDE.

    Only an exact log-density supports the density-known regime (the SM
    objective); a KDE-only target belongs in the samples-known regime
    (HSM, FKL, RKL).
    """
    if name not in TARGETS_2D:
        raise ValueError(f"unknown 2D target: {name!r}")
    return name not in ("two-moons", "pinwheel", "checkerboard")
