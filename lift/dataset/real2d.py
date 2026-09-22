"""Real 2-D datasets for clustering and density-estimation experiments.

Two loaders matching the 2-D target interface of
:mod:`lift.dataset.toy2d`:

  * :class:`OldFaithful` -- 272 (eruption duration, waiting time) pairs;
    bimodal and slightly skewed.
  * :class:`Gaia2D` -- a stellar Hertzsprung--Russell color-magnitude
    diagram (BP-RP color, absolute G magnitude); banana-shaped main
    sequence plus a thin red-giant branch.

Both expect their data under ``data/datasets/<name>/`` and download
nothing; a missing file raises ``FileNotFoundError`` naming the upstream
source.

``sample`` bootstraps the empirical points with i.i.d. Gaussian jitter, so
the target has finite-data support with smoothed shoulders. ``log_prob``
is a Gaussian KDE surrogate over the same points, used for plotting.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch

from lift.dataset.toy2d import _KDESurrogate
from projorg import gitdir


_DATASETS_DIR = os.path.join(
    gitdir(),
    "data", "datasets",
)


class _BootstrapWithJitter2D:
    """Base class: bootstrap the empirical points with isotropic jitter.

    Subclasses set ``data`` (an ``(N, 2)`` numpy array) and a default
    ``jitter`` and ``kde_bandwidth``; the rest of the target interface
    (``D``, ``log_prob``, ``sample``, ``xlim``) is inherited.
    """

    D = 2
    jitter: float = 0.05
    kde_bandwidth: float = 0.10

    def __init__(
        self,
        data: np.ndarray,
        *,
        jitter: Optional[float] = None,
        kde_bandwidth: Optional[float] = None,
        xlim: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        if data.ndim != 2 or data.shape[-1] != 2:
            raise ValueError(
                f"data must be (N, 2); got shape {data.shape}",
            )
        self.data = np.asarray(data, dtype=np.float32)
        if jitter is not None:
            self.jitter = float(jitter)
        if kde_bandwidth is not None:
            self.kde_bandwidth = float(kde_bandwidth)
        # Pad the empirical bbox by 5x jitter for plotting.
        lo = self.data.min(axis=0) - 5.0 * self.jitter
        hi = self.data.max(axis=0) + 5.0 * self.jitter
        self.xlim = xlim or ((float(lo[0]), float(hi[0])),
                             (float(lo[1]), float(hi[1])))

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        """Bootstrap ``n`` points from the empirical data + Gaussian jitter."""
        n = int(n)
        idx = torch.randint(0, int(self.data.shape[0]), (n,))
        base = torch.from_numpy(self.data[idx.numpy()]).float()
        return (base + self.jitter * torch.randn_like(base)).to(device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian-KDE surrogate evaluated at ``x``.

        The KDE is built lazily on first call, with the empirical samples
        as kernel centers and bandwidth ``kde_bandwidth``.
        """
        cache = getattr(self, "_kde", None)
        if cache is None:
            samples = torch.from_numpy(self.data).float()
            cache = _KDESurrogate(samples, bandwidth=self.kde_bandwidth)
            self._kde = cache
        return cache.log_prob(x)


def _load_csv_or_explain(path: str, source_url: str, *,
                          expected_cols: int = 2) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            f"Place the dataset there. Source: {source_url}\n"
            "The repository deliberately does not ship this file; "
            "drop it in once and the loader will pick it up.",
        )
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, expected_cols)
    if arr.ndim != 2 or arr.shape[-1] != expected_cols:
        raise ValueError(
            f"loaded {path} with shape {arr.shape}; "
            f"expected (N, {expected_cols}). "
            "Check delimiter / header line.",
        )
    return arr.astype(np.float32)


class OldFaithful(_BootstrapWithJitter2D):
    """Old Faithful geyser: (eruption duration, waiting time).

    The R `faithful` dataset, 272 rows, 2 columns. Canonical
    Gaussian-mixture clustering example (Bishop's PRML §9.2).
    Bimodal, slightly skewed.

    Data file: ``data/datasets/old_faithful/faithful.csv`` --
    two-column comma-separated, no header, columns
    ``(eruptions, waiting)`` in minutes. Public-domain R dataset; place
    the file once, then the loader picks it up. Source:
    https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html
    """

    name = "old-faithful"

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        standardise: bool = True,
        jitter: Optional[float] = None,
        kde_bandwidth: Optional[float] = None,
    ) -> None:
        path = path or os.path.join(
            _DATASETS_DIR, "old_faithful", "faithful.csv",
        )
        data = _load_csv_or_explain(
            path,
            source_url="https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html",
            expected_cols=2,
        )
        if standardise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            data = (data - mean) / np.maximum(std, 1e-6)
        super().__init__(
            data,
            jitter=jitter if jitter is not None else 0.05,
            kde_bandwidth=kde_bandwidth if kde_bandwidth is not None else 0.10,
        )


class Gaia2D(_BootstrapWithJitter2D):
    """Gaia DR3 stellar HR diagram: (BP-RP color, absolute G magnitude).

    Data file: ``data/datasets/gaia/gaia_dr3_hr.csv`` -- two-column
    comma-separated, no header, columns ``(bp_rp, M_G)``. The intent
    is the Gaia DR3 100-pc sample (~330k rows) restricted to stars
    with finite parallax + photometric quality cuts. Source query
    available at https://gea.esac.esa.int/archive/.

    Pre-processing (parallax cuts, dust de-reddening, magnitude limits) is
    assumed already applied. Standardization happens here so the training
    stack stays in :math:`O(1)`-scale coordinates.
    """

    name = "gaia-hr"

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        standardise: bool = True,
        jitter: Optional[float] = None,
        kde_bandwidth: Optional[float] = None,
        max_rows: Optional[int] = 50_000,
    ) -> None:
        path = path or os.path.join(_DATASETS_DIR, "gaia", "gaia_dr3_hr.csv")
        data = _load_csv_or_explain(
            path,
            source_url="https://gea.esac.esa.int/archive/",
            expected_cols=2,
        )
        if max_rows is not None and int(max_rows) > 0 and \
                data.shape[0] > int(max_rows):
            rng = np.random.default_rng(0)
            idx = rng.choice(data.shape[0], int(max_rows), replace=False)
            data = data[idx]
        if standardise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            data = (data - mean) / np.maximum(std, 1e-6)
        super().__init__(
            data,
            jitter=jitter if jitter is not None else 0.02,
            kde_bandwidth=kde_bandwidth if kde_bandwidth is not None else 0.05,
        )


REAL_TARGETS_2D: dict[str, type] = {
    OldFaithful.name: OldFaithful,
    Gaia2D.name: Gaia2D,
}


def get_real_target_2d(name: str, **kwargs):
    if name not in REAL_TARGETS_2D:
        raise ValueError(
            f"unknown real 2D target: {name!r}; "
            f"pick from {sorted(REAL_TARGETS_2D)}",
        )
    return REAL_TARGETS_2D[name](**kwargs)
