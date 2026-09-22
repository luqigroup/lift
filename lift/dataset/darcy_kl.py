"""Darcy-flow Bayesian inverse problem in Karhunen-Loeve coordinates.

Paired samples for an amortized conditional density ``p(xi | y)`` over a
nonlinear groundwater-flow inverse problem. The unknown ``xi`` is a
truncated Karhunen-Loeve coefficient vector in ``R^100`` under a standard
normal prior; the observation ``y`` is 33 sensor readings through the
nonlinear Darcy forward map, corrupted by noise of standard deviation
``sigma_y``. Coefficients rather than the 40x40 field, because the field is
a deterministic linear image of the 100 coefficients and so has no density
with respect to Lebesgue measure on ``R^1600``.

The archive also ships an orthogonal decomposition of coefficient space
into the linearized operator's leading ``resolved`` directions and the
``blind`` complement, both ordered by singular value. The split is a
truncation of a smoothly decaying spectrum, not an intrinsic rank:
directions just inside the blind block still carry small but nonzero
information, and past the last nonzero singular value any orthonormal
basis is equally valid.

``xi`` arrives whitened and is passed through untouched. ``y`` is centered
per sensor and divided by a single global scale. The centering is
necessary because sensor means span a wide range while their fluctuations
are two orders of magnitude smaller, which leaves the conditioning path
numerically invisible against O(1) weight-init scales. The scale is global
on purpose: the weakest sensor's spread equals ``sigma_y``, so it carries
pure noise, and per-sensor standardization would raise that channel to the
magnitude of the most informative one. Conditioning on the transformed
observation is the same event as conditioning on the raw one and the
modeled density is in ``x``, so no Jacobian enters.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from projorg import gitdir

_DATA_DIR = os.path.join(
    gitdir(),
    "data", "datasets", "darcy",
)

_SOURCE_REPO = "https://github.com/alisiahkoohi/priorlaundermat"

_SPLIT_TRAIN = 0
_SPLIT_EVAL = 1


class GlobalScaleNormalizer:
    """Per-coordinate centering with a single shared scale.

    Follows the mean/std/eps contract of ``priorlaundermat``'s
    ``Normalizer``, but collapses the per-coordinate standard deviation to
    one global number, so relative coordinate magnitudes -- here the
    per-sensor signal-to-noise ordering -- survive the transform.
    """

    def __init__(self, dataset: torch.Tensor, eps: float = 1e-5) -> None:
        self.mean = dataset.mean(dim=0)
        self.scale = dataset.std(dim=0).mean()
        self.eps = float(eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.scale + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.scale + self.eps) + self.mean


class DarcyKL:
    """Paired ``(xi, y)`` samples for amortized Darcy posterior transport.

    Attributes
    ----------
    D, D_y :
        Coefficient and observation dimension (100 and 33).
    x_train, y_train, x_eval, y_eval :
        Float32 tensors on ``device``; ``y`` is centered and globally
        scaled.
    resolved, blind :
        ``(D, K)`` and ``(D, D - K)`` orthonormal bases in coefficient
        coordinates, ordered by singular value.
    sv, sigma_y :
        Singular values of the linearized operator and the observation
        noise level, from which the conditional reference is built.
    """

    name = "darcy_kl"

    def __init__(self, data_path: str = "", device: str = "cpu") -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover -- env-dependent
            raise ImportError(
                "h5py is required to read the Darcy archive. "
                "Install with `pip install h5py`.",
            ) from exc

        path = data_path or os.path.join(_DATA_DIR, "darcy_laundering.h5")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Darcy archive not found at {path!r}. Generate it with the "
                f"groundwater pipeline in {_SOURCE_REPO} and either copy "
                f"darcy_laundering.h5 into {_DATA_DIR} or pass --data_path.",
            )

        with h5py.File(path, "r") as h:
            for key in ("xi_true", "y_obs", "xi_map", "split",
                        "resolved", "blind", "sv"):
                if key not in h:
                    raise KeyError(
                        f"Darcy archive {path!r} is missing dataset {key!r}; "
                        f"found {sorted(h.keys())!r}.",
                    )
            xi = np.asarray(h["xi_true"][:], dtype=np.float32)
            y = np.asarray(h["y_obs"][:], dtype=np.float32)
            xi_map = np.asarray(h["xi_map"][:], dtype=np.float32)
            split = np.asarray(h["split"][:])
            resolved = np.asarray(h["resolved"][:], dtype=np.float32)
            blind = np.asarray(h["blind"][:], dtype=np.float32)
            sv = np.asarray(h["sv"][:], dtype=np.float32)
            if "sigma_y" not in h.attrs:
                raise KeyError(
                    f"Darcy archive {path!r} carries no 'sigma_y' attribute; "
                    "the conditional reference curve cannot be built.",
                )
            sigma_y = float(h.attrs["sigma_y"])
            rank = int(h.attrs.get("rank", resolved.shape[1]))

        n = len(xi)
        for name, arr in (("y_obs", y), ("xi_map", xi_map), ("split", split)):
            if len(arr) != n:
                raise ValueError(
                    f"row-count mismatch: xi_true has {n}, {name} has "
                    f"{len(arr)}.",
                )
        if resolved.shape[0] != xi.shape[1] or blind.shape[0] != xi.shape[1]:
            raise ValueError(
                f"basis dimension mismatch: xi has {xi.shape[1]} coefficients, "
                f"resolved is {resolved.shape}, blind is {blind.shape}.",
            )

        is_train = split == _SPLIT_TRAIN
        is_eval = split == _SPLIT_EVAL
        if not is_train.any() or not is_eval.any():
            raise ValueError(
                f"expected split values {{{_SPLIT_TRAIN}, {_SPLIT_EVAL}}}; "
                f"got {np.unique(split).tolist()!r}",
            )

        self.D = int(xi.shape[1])
        self.D_y = int(y.shape[1])
        self.rank = rank
        self.sigma_y = sigma_y
        self.device = device

        y_train_raw = torch.from_numpy(y[is_train])
        y_eval_raw = torch.from_numpy(y[is_eval])
        # The eval split never informs the normalization statistics.
        self.y_normalizer = GlobalScaleNormalizer(y_train_raw)

        self.x_train = torch.from_numpy(xi[is_train]).to(device)
        self.x_eval = torch.from_numpy(xi[is_eval]).to(device)
        self.y_train = self.y_normalizer.normalize(y_train_raw).to(device)
        self.y_eval = self.y_normalizer.normalize(y_eval_raw).to(device)

        self.resolved = torch.from_numpy(resolved).to(device)
        self.blind = torch.from_numpy(blind).to(device)
        self.sv = torch.from_numpy(sv).to(device)

        residual = torch.from_numpy(xi[is_eval] - xi_map[is_eval]).to(device)
        self.residual_std_resolved = (residual @ self.resolved).std(dim=0)
        self.residual_std_blind = (residual @ self.blind).std(dim=0)

    def __repr__(self) -> str:  # pragma: no cover -- cosmetic
        return (
            f"DarcyKL(D={self.D}, D_y={self.D_y}, rank={self.rank}, "
            f"n_train={len(self.x_train)}, n_eval={len(self.x_eval)}, "
            f"sigma_y={self.sigma_y:g})"
        )

    # ------------------------------------------------------------------
    # Direction bases and their reference curves
    # ------------------------------------------------------------------

    def probe_directions(self, n_blind: int) -> torch.Tensor:
        """Resolved block followed by the leading ``n_blind`` blind columns."""
        n_blind = min(int(n_blind), self.blind.shape[1])
        return torch.cat([self.resolved, self.blind[:, :n_blind]], dim=1)

    def conditional_precision_reference(self, n_dirs: int) -> torch.Tensor:
        """Linearized-Gaussian conditional precision per direction.

        For a linear-Gaussian model with a standard normal prior and
        observation noise ``sigma_y``, the posterior precision along the
        ``j``-th right singular direction is ``1 + (sv_j / sigma_y)^2``.
        Directions past the operator's last nonzero singular value are
        uninformed and sit at the prior precision of one.

        This is the estimand a directional curvature of the learned
        log-density estimates, and it carries no bias from a point estimate.
        """
        prec = torch.ones(int(n_dirs), device=self.sv.device)
        m = min(int(n_dirs), len(self.sv))
        prec[:m] = 1.0 + (self.sv[:m] / self.sigma_y) ** 2
        return prec

    def residual_precision_reference(self, n_blind: int) -> torch.Tensor:
        """Empirical marginal precision from the MAP residual.

        Marginal rather than conditional, so by Jensen it sits below
        ``conditional_precision_reference``, and on a nonlinear problem it
        also conflates posterior spread with MAP error. It is a secondary
        curve, not the estimand.
        """
        n_blind = min(int(n_blind), self.blind.shape[1])
        std = torch.cat([
            self.residual_std_resolved,
            self.residual_std_blind[:n_blind],
        ])
        return 1.0 / std ** 2

    def n_informed_directions(self) -> int:
        """Count of directions with a nonzero singular value."""
        return int((self.sv > 1e-8).sum().item())

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def train_batch(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` paired rows uniformly from the train split."""
        idx = torch.randint(
            0, len(self.x_train), (n,),
            generator=generator, device=self.x_train.device,
        )
        return self.x_train[idx], self.y_train[idx]

    def anchor_batches(
        self,
        n_anchor: int,
        n_batches: int,
        generator: Optional[torch.Generator] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Reference batches for freezing an emitted readout.

        The first entry is the whole train split, the minimum-variance
        anchor for a mean-pooled summary; the rest are resampled batches of
        size ``n_anchor``, which report how sensitive the frozen score is to
        the anchor choice.
        """
        out = [(self.x_train, self.y_train)]
        for _ in range(max(0, int(n_batches) - 1)):
            idx = torch.randperm(
                len(self.x_train), generator=generator,
                device=self.x_train.device,
            )[:int(n_anchor)]
            out.append((self.x_train[idx], self.y_train[idx]))
        return out
