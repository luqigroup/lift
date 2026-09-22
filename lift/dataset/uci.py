"""UCI / BSDS300 tabular density-estimation datasets (MAF benchmark suite).

Loaders for the tabular density-estimation benchmark of MAF (Papamakarios et
al., 2017): POWER, GAS, HEPMASS, MINIBOONE and BSDS300.

The splits are neither downloaded nor preprocessed here. Each loader expects
``data/datasets/uci/<name>/{train,val,test}.npy``, each ``(N, D)`` float32
and whitened per the MAF pipeline, and raises ``FileNotFoundError`` pointing
at the MAF preprocessing scripts when they are missing.

Each loader exposes the target interface the EBM trainer expects: the class
attributes ``D`` and ``name``; ``sample(n, device)``, which bootstraps ``n``
rows with replacement from the train split; ``log_prob(x)``, a Gaussian-KDE
surrogate over the train split meant for diagnostics rather than in-loop
evaluation; and ``train_data`` / ``val_data`` / ``test_data`` for direct NLL
evaluation.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
from projorg import gitdir


_DATASETS_DIR = os.path.join(
    gitdir(),
    "data", "datasets", "uci",
)

_MAF_REPO_URL = "https://github.com/gpapamak/maf"


def _load_split(name: str, split: str) -> np.ndarray:
    path = os.path.join(_DATASETS_DIR, name, f"{split}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required UCI split not found: {path}\n"
            f"Run 'python scripts/prepare_uci_{name}.py', which downloads the "
            f"benchmark data of {_MAF_REPO_URL} and applies its published "
            f"preprocessing. Preprocessed splits are not vendored here.",
        )
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(
            f"loaded {path} with shape {arr.shape}; expected (N, D).",
        )
    return arr.astype(np.float32)


class _UCIBase:
    """Base class for tabular UCI datasets with the EBM target interface."""

    name: str = "uci"
    expected_D: Optional[int] = None
    kde_bandwidth: float = 0.10  # for plotting only; tune via subclass

    @staticmethod
    def _load_local(path: str, split: str) -> np.ndarray:
        full = os.path.join(path, f"{split}.npy")
        if not os.path.isfile(full):
            raise FileNotFoundError(
                f"{full} not found under explicit path {path!r}.",
            )
        arr = np.load(full)
        if arr.ndim != 2:
            raise ValueError(
                f"loaded {full} with shape {arr.shape}; expected (N, D).",
            )
        return arr.astype(np.float32)

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        max_train_rows: Optional[int] = None,
    ) -> None:
        if path is None:
            self.train_data = _load_split(self.name, "train")
            self.val_data = _load_split(self.name, "val")
            self.test_data = _load_split(self.name, "test")
        else:
            self.train_data = self._load_local(path, "train")
            self.val_data = self._load_local(path, "val")
            self.test_data = self._load_local(path, "test")
        if (
            self.expected_D is not None
            and self.train_data.shape[-1] != self.expected_D
        ):
            raise ValueError(
                f"{self.name}: train_data has D={self.train_data.shape[-1]}, "
                f"expected D={self.expected_D}. Check the preprocessing.",
            )
        self.D = int(self.train_data.shape[-1])
        if (
            max_train_rows is not None
            and int(max_train_rows) > 0
            and self.train_data.shape[0] > int(max_train_rows)
        ):
            rng = np.random.default_rng(0)
            idx = rng.choice(
                self.train_data.shape[0], int(max_train_rows), replace=False,
            )
            self.train_data = self.train_data[idx]

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        n = int(n)
        idx = torch.randint(0, int(self.train_data.shape[0]), (n,))
        s = torch.from_numpy(self.train_data[idx.numpy()]).float()
        return s.to(device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Lazily-built Gaussian-KDE surrogate over the train split.

        For high D this is expensive (``O(N_train * batch * D)``);
        intended for diagnostics only, not for in-loop evaluation.
        """
        kde = getattr(self, "_kde", None)
        if kde is None:
            kde = _BatchedKDE(self.train_data, bandwidth=self.kde_bandwidth)
            self._kde = kde
        return kde.log_prob(x)


class _BatchedKDE:
    """Minimal high-D Gaussian KDE with chunked evaluation."""

    def __init__(self, samples: np.ndarray, bandwidth: float = 0.10):
        self.samples = torch.from_numpy(samples).float()
        self.h = float(bandwidth)
        self.D = int(samples.shape[-1])
        self.log_norm = (
            -0.5 * self.D * math.log(2.0 * math.pi)
            - self.D * math.log(self.h)
            - math.log(int(samples.shape[0]))
        )

    # Query- and reference-side chunk sizes. The product bounds the
    # largest intermediate, which is (q_chunk, n_chunk) after ``cdist``.
    q_chunk: int = 1024
    n_chunk: int = 65536

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Kernel density estimate, chunked on both axes.

        Forming ``x[:, None] - s[None]`` of shape ``(b, N_train, D)`` would
        allocate the training set times the query batch, hundreds of
        gigabytes on the larger tabular targets, and the merely large cases
        are the dangerous ones: they do not fail fast, they spill into swap
        and the process is killed. ``cdist`` gives the same squared
        distances without materializing the difference tensor, and the
        log-sum-exp is accumulated across reference chunks through
        ``logaddexp`` rather than by concatenating an ``(b, N_train)``
        intermediate, so peak memory is ``q_chunk * n_chunk``, independent
        of the training-set size.
        """
        s = self.samples.to(x.device)
        inv_two_h2 = 0.5 / (self.h * self.h)
        out = []
        for chunk in x.split(self.q_chunk, dim=0):
            running: Optional[torch.Tensor] = None
            for ref in s.split(self.n_chunk, dim=0):
                log_kern = -inv_two_h2 * torch.cdist(chunk, ref).pow(2)
                part = torch.logsumexp(log_kern, dim=-1)
                running = part if running is None else torch.logaddexp(
                    running, part,
                )
            out.append(self.log_norm + running)
        return torch.cat(out, dim=0)


class POWER(_UCIBase):
    """Individual household electric power consumption (D=6)."""
    name = "power"
    expected_D = 6


class GAS(_UCIBase):
    """Gas-sensor array drift (D=10).

    MAF's published preprocessing yields D=8 after iterative correlation
    pruning; ``scripts/prepare_uci_gas.py`` yields D=10 because the
    iteration order of the 0.98 correlation filter is implementation
    specific, leaving two extra weakly correlated columns.
    """
    name = "gas"
    expected_D = 10


class HEPMASS(_UCIBase):
    """High-energy physics, particle vs.\\ background (D=21).

    The 21-D representation is the canonical MAF / CP-Flow tabular
    benchmark (Papamakarios et al., 2017; Huang et al., 2021).
    ``scripts/prepare_uci_hepmass.py`` follows the MAF recipe
    (positive-class restriction, dedup, drop of the 7 discrete "mass"
    columns with unique-value fraction below 5\\%) and writes a
    ``(N, 21)`` whitened split. Relaxing that script's
    ``unique_frac_threshold`` flag gives the unpruned 28-D split.
    """
    name = "hepmass"
    expected_D = 21


class MINIBOONE(_UCIBase):
    """MiniBooNE neutrino oscillation (D=43), MAF's published dimension.

    ``scripts/prepare_uci_miniboone.py`` reproduces MAF's example count of
    36,488 exactly (36,499 signal rows less the 11 carrying the -999
    sentinel) and drops the seven most quantized feature columns to reach
    D=43.
    """
    name = "miniboone"
    expected_D = 43


class BSDS300(_UCIBase):
    """BSDS300 image patches, 8x8 minus the center pixel (D=63)."""
    name = "bsds300"
    expected_D = 63


UCI_TARGETS: dict[str, type] = {
    POWER.name: POWER,
    GAS.name: GAS,
    HEPMASS.name: HEPMASS,
    MINIBOONE.name: MINIBOONE,
    BSDS300.name: BSDS300,
}


def get_uci_target(name: str, **kwargs) -> _UCIBase:
    if name not in UCI_TARGETS:
        raise ValueError(
            f"unknown UCI target: {name!r}; pick from {sorted(UCI_TARGETS)}",
        )
    return UCI_TARGETS[name](**kwargs)
