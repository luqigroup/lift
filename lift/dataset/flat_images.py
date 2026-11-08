"""Flattened, normalized image classification splits (MNIST, CIFAR-10).

Loads a training split through ``torchvision``, applies ``ToTensor`` and the
per-channel ``Normalize(mean, std)`` constants below, flattens each image to a
vector and splits ``train_split`` / ``1 - train_split`` under a generator
seeded by the run seed, so every construction run at one seed sees the same
update and evaluation sets. The official test split is read with the training
constants and is never split. No augmentation and no whitening.

The normalized split is materialized once into a contiguous ``float32``
tensor, and batching is an index shuffle over it under a per-epoch generator,
which is what ``DataLoader(shuffle=True)`` does. ``device`` decides whether
that tensor lives on the host or on the accelerator (0.55 GB for a CIFAR-10
training split).

Data live under ``projorg.datadir("datasets")`` and are fetched from their
publisher on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from projorg import datadir


@dataclass(frozen=True)
class ImageBenchmarkSpec:
    """One benchmark: its shape, class count and normalization constants.

    ``width`` is the flattened input dimension.
    """

    name: str
    torchvision_name: str
    subdir: str
    shape: tuple[int, int, int]
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @property
    def width(self) -> int:
        return int(np.prod(self.shape))


BENCHMARKS: dict[str, ImageBenchmarkSpec] = {
    "mnist": ImageBenchmarkSpec(
        name="mnist",
        torchvision_name="MNIST",
        subdir="mnist",
        shape=(1, 28, 28),
        num_classes=10,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "cifar10": ImageBenchmarkSpec(
        name="cifar10",
        torchvision_name="CIFAR10",
        subdir="cifar10",
        shape=(3, 32, 32),
        num_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
}


def get_spec(name: str) -> ImageBenchmarkSpec:
    key = str(name).strip().lower()
    if key not in BENCHMARKS:
        raise ValueError(
            f"unknown image benchmark {name!r}; expected one of "
            f"{sorted(BENCHMARKS)}",
        )
    return BENCHMARKS[key]


def default_root(mkdir: bool = True) -> str:
    """The parent of the per-dataset directories."""
    return datadir("datasets", mkdir=mkdir)


class FlatImageSplit:
    """A materialized, normalized, flattened split and its index shuffler.

    Holds ``x`` of shape ``(N, width)`` and integer labels ``y`` of shape
    ``(N,)``. :meth:`epoch_batches` yields index tensors rather than slices,
    so a split kept on the host and one kept on the accelerator run the same
    loop.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"expected x (N, D) and y (N,), got {tuple(x.shape)} and "
                f"{tuple(y.shape)}",
            )
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def width(self) -> int:
        return int(self.x.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.x.numel() * self.x.element_size()
                   + self.y.numel() * self.y.element_size())

    def to(self, device: torch.device | str) -> "FlatImageSplit":
        return FlatImageSplit(self.x.to(device), self.y.to(device))

    def batch(self, idx: torch.Tensor, device: torch.device | str):
        """One batch, moved to ``device`` (a no-op if already there)."""
        xb = self.x[idx]
        yb = self.y[idx]
        if xb.device != torch.device(device):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
        return xb, yb

    def epoch_batches(
        self,
        batch_size: int,
        generator: torch.Generator,
        drop_last: bool = False,
    ):
        """Index tensors for one shuffled epoch (``DataLoader`` order)."""
        n = len(self)
        perm = torch.randperm(n, generator=generator)
        stop = (n // batch_size) * batch_size if drop_last else n
        for start in range(0, stop, batch_size):
            yield perm[start:start + batch_size]

    def fixed_subset(self, n: int, seed: int = 0) -> "FlatImageSplit":
        """A deterministic subset, for the cheap validation cadence."""
        n = min(int(n), len(self))
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(len(self), generator=g)[:n]
        return FlatImageSplit(self.x[idx].clone(), self.y[idx].clone())


def _materialize(dataset, spec: ImageBenchmarkSpec) -> tuple:
    """``ToTensor`` + ``Normalize`` + flatten, for the whole split at once."""
    raw = dataset.data
    if isinstance(raw, torch.Tensor):                     # MNIST: uint8 (N,H,W)
        arr = raw.numpy()
    else:                                                 # CIFAR: (N,H,W,C)
        arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()                # ToTensor's (N,C,H,W)
    mean = torch.tensor(spec.mean).view(1, -1, 1, 1)
    std = torch.tensor(spec.std).view(1, -1, 1, 1)
    x = (x - mean) / std
    y = torch.as_tensor(dataset.targets, dtype=torch.long)
    return x.reshape(x.shape[0], -1).contiguous(), y


def _load(spec: ImageBenchmarkSpec, train: bool, root: str | None,
          download: bool):
    from torchvision import datasets as tv_datasets

    root = root or default_root()
    ctor = getattr(tv_datasets, spec.torchvision_name)
    data = ctor(os.path.join(root, spec.subdir), train=bool(train),
                download=bool(download))
    x, y = _materialize(data, spec)
    if x.shape[1] != spec.width:
        raise ValueError(
            f"{spec.name}: flattened width {x.shape[1]} != spec width "
            f"{spec.width}",
        )
    return x, y


def image_split(
    name: str,
    seed: int,
    root: str | None = None,
    train_split: float = 0.9,
    download: bool = True,
) -> tuple[FlatImageSplit, FlatImageSplit, ImageBenchmarkSpec]:
    """The training split, normalized, flattened and split in two.

    Args:
        name: ``mnist`` or ``cifar10``.
        seed: seeds the permutation that defines the split.
        root: parent of the per-dataset directories; defaults to
            ``projorg.datadir("datasets")``.
        train_split: the fraction kept for updates.
        download: fetch the dataset from its publisher if it is missing.

    Returns:
        ``(train, valid, spec)``.
    """
    spec = get_spec(name)
    x, y = _load(spec, True, root, download)
    n = int(x.shape[0])
    n_train = int(float(train_split) * n)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    tr, va = perm[:n_train], perm[n_train:]
    return (
        FlatImageSplit(x[tr].contiguous(), y[tr].contiguous()),
        FlatImageSplit(x[va].contiguous(), y[va].contiguous()),
        spec,
    )


def image_test(
    name: str,
    root: str | None = None,
    download: bool = True,
) -> tuple[FlatImageSplit, ImageBenchmarkSpec]:
    """The official test split, normalized with the training constants.

    Returns:
        ``(test, spec)``.
    """
    spec = get_spec(name)
    x, y = _load(spec, False, root, download)
    return FlatImageSplit(x.contiguous(), y.contiguous()), spec
