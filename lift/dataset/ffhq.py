"""FFHQ degradation pairs for the ICNN regularizer (Ehrhardt Section 5 setup).

Loads the FFHQ images shipped inside the Ehrhardt, Mukherjee and Wong release
(``hsw43/icnn_primal_dual`` commit ``bb0d0bd``, Zenodo DOI
10.5281/zenodo.17426033): ``training_data/`` holds 1,000 images and
``test_data/`` holds 20, all 256x256 8-bit grayscale PNG, expected unpacked
under ``data/third_party/ehrhardt_icnn_primal_dual/``. The FFHQ imagery carries
NVIDIA's CC BY-NC-SA 4.0 dataset license and is never redistributed here.

Degradation pairs follow their ``dataset.py::inpaint_dataset``: the corrupted
copy is built once at construction as ``mask * img + 0.03 * randn`` with the
shipped ``data/mask.npy`` for ``task="inpaint"``, or as salt-and-pepper at
probability 0.25 for ``task="snp"``. Two deviations from their loader: the
noise realization comes from an explicit per-seed ``torch.Generator`` rather
than from global torch state, and the tensors stay in host memory for the
trainer to move.

Training negatives. The adversarial-regularizer loss takes the degraded copies
as its negative class. Their loader hands the critic the zero-filled
``mask * img + noise``, whose mean intensity is below the clean image's by the
masked fraction of the image mean, so the two classes are separable by
brightness alone and the learned regularizer acquires a DC component. Their
eq. 1 specifies the mean-preserving pseudo-inverse ``A^dagger y`` instead, and
``mean_match_negatives=True`` shifts each negative by a per-image constant onto
the clean mean, then clamps to [0, 1]. The ``noisy`` measurement is never
shifted: it is the ``y`` of the validation and test solves.

Pixel decoding: they read each PNG with ``cv2.imread`` and mix with
``skimage.color.rgb2gray``, which on equal channels reduces to the uint8 value
over 255 in float64. Loading through PIL mode "L" and dividing by 255 in
float64 before the float32 cast reproduces that to float32 resolution without
those dependencies.

:class:`FFHQColorPairs` carries the same recipe over the prepared color FFHQ
memmap under ``data/datasets/ffhq_color`` (built by
``scripts/prepare_ffhq_color.py``; FFHQ carries NVIDIA's CC BY-NC-SA 4.0
license plus per-image Flickr licenses, research use only, never redistributed
here), with the shipped mask broadcast across the three channels and
nearest-neighbor downsampled when the prepared resolution is below 256.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from projorg import gitdir


_ZENODO_DOI = "10.5281/zenodo.17426033"

_DEFAULT_ROOT = os.path.join(
    gitdir(),
    "data", "third_party", "ehrhardt_icnn_primal_dual",
)

_TASKS = ("inpaint", "snp")


def resolve_archive_dir(data_root: str | None = None) -> str:
    """Locate the unpacked ``icnn_primal_dual`` archive directory.

    ``data_root`` may point at the archive directory itself (containing
    ``training_data/``) or at a parent holding the unzipped archive one
    level down (the layout ``unzip`` produces under the default root).
    ``None`` / ``""`` falls back to the default
    ``data/third_party/ehrhardt_icnn_primal_dual``.
    """
    root = data_root if data_root else _DEFAULT_ROOT
    if os.path.isdir(os.path.join(root, "training_data")):
        return root
    hits = sorted(glob.glob(os.path.join(root, "*", "training_data")))
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError(
        f"Ehrhardt FFHQ archive not found under {root!r} (no "
        f"training_data/ directory). Download the release archive from "
        f"Zenodo DOI {_ZENODO_DOI} (GitHub archive hsw43/icnn_primal_dual, "
        f"commit bb0d0bd) and unzip it under "
        f"data/third_party/ehrhardt_icnn_primal_dual/. "
        f"slurm/cluster_setup.sh automates this."
    )


def _load_grayscale_png(path: str) -> torch.Tensor:
    """One PNG -> (1, 1, H, W) float32 in [0, 1] (see module docstring)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    t = torch.from_numpy(arr).float()
    return t.view(1, 1, *t.shape)


def dc_matched_negatives(clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    """Per-image DC-matched negatives; see the module docstring.

    ``noisy`` is clamped to [0, 1] first, so the shift is computed on what the
    critic would otherwise see, then shifted by ``mean(clean_i) -
    mean(noisy_i)`` per image and clamped again. Shapes ``(N, C, H, W)``, with
    the mean over ``C, H, W``.
    """
    base = noisy.clamp(0, 1)
    dims = tuple(range(1, base.dim()))
    shift = clean.mean(dim=dims, keepdim=True) - base.mean(dim=dims, keepdim=True)
    return (base + shift).clamp(0, 1)


class FFHQPairs:
    """Clean / degraded FFHQ pairs with their fixed-realization recipe.

    Args:
        split: ``"train"`` (1,000 images) or ``"test"`` (20 images).
        task: ``"inpaint"`` (shipped mask + sigma-0.03 Gaussian noise)
            or ``"snp"`` (salt-and-pepper at ``snp_prob``).
        degradation_seed: seed of the generator that draws the single
            fixed corruption realization at construction.
        data_root: archive location override (see
            :func:`resolve_archive_dir`); ``None`` -> repository default.
        noise_sigma: Gaussian noise level for ``task="inpaint"``
            (theirs: 0.03).
        snp_prob: salt-and-pepper corruption probability for
            ``task="snp"`` (theirs: 0.25).
        max_images: optional truncation of the sorted file list, a test
            hook the experiment drivers do not use.
        mean_match_negatives: build the training ``negatives`` with
            :func:`dc_matched_negatives`, removing the DC shortcut, instead
            of aliasing ``noisy`` as their released loader does.

    Attributes:
        clean, noisy: ``(N, 1, H, W)`` float32 HOST tensors.
        negatives: ``(N, 1, H, W)`` critic negatives; ``noisy`` itself
            unless ``mean_match_negatives`` is set.
        mask: the shipped inpainting mask ``(1, 1, 256, 256)`` (loaded
            for both tasks; the evaluation harness builds the
            inpainting physics from it).
    """

    def __init__(
        self,
        split: str = "train",
        task: str = "inpaint",
        degradation_seed: int = 0,
        data_root: str | None = None,
        noise_sigma: float = 0.03,
        snp_prob: float = 0.25,
        max_images: int | None = None,
        mean_match_negatives: bool = False,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        self.split = str(split)
        self.task = str(task)
        self.degradation_seed = int(degradation_seed)
        self.noise_sigma = float(noise_sigma)
        self.snp_prob = float(snp_prob)
        self.mean_match_negatives = bool(mean_match_negatives)

        archive = resolve_archive_dir(data_root)
        img_dir = os.path.join(
            archive, "training_data" if split == "train" else "test_data",
        )
        paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not paths:
            raise FileNotFoundError(
                f"no PNG files under {img_dir!r}; the archive at "
                f"{archive!r} looks incomplete (Zenodo DOI {_ZENODO_DOI}).",
            )
        if max_images is not None and int(max_images) > 0:
            paths = paths[: int(max_images)]
        self.paths = paths

        mask_path = os.path.join(archive, "data", "mask.npy")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"shipped inpainting mask missing at {mask_path!r} "
                f"(Zenodo DOI {_ZENODO_DOI}).",
            )
        # A torch-saved float tensor of shape (1, 1, 256, 256) despite
        # the .npy suffix -- their dataset.py loads it with torch.load.
        self.mask = torch.load(mask_path, map_location="cpu")

        self.clean = torch.cat(
            [_load_grayscale_png(p) for p in paths], dim=0,
        )
        gen = torch.Generator().manual_seed(self.degradation_seed)
        if task == "inpaint":
            noise = torch.randn(self.clean.shape, generator=gen)
            self.noisy = self.mask * self.clean + self.noise_sigma * noise
        else:
            probs = torch.rand(self.clean.shape, generator=gen)
            noisy = self.clean.clone()
            noisy[probs < self.snp_prob / 2] = 0.0
            noisy[probs > 1 - self.snp_prob / 2] = 1.0
            self.noisy = noisy
        # The critic's negative class. ``noisy`` stays the measurement.
        if self.mean_match_negatives:
            self.negatives = dc_matched_negatives(self.clean, self.noisy)
        else:
            self.negatives = self.noisy

        self.img_size = int(self.clean.shape[-1])
        self.n_channels = 1

    def __len__(self) -> int:
        return int(self.clean.shape[0])


# ---------------------------------------------------------------------------
# Color / full-FFHQ extension
# ---------------------------------------------------------------------------

_DEFAULT_COLOR_ROOT = os.path.join(
    gitdir(),
    "data", "datasets", "ffhq_color",
)

# The color test split: FFHQ images 0-19, the same 20 identities as the
# Ehrhardt archive's test_data/00000.png-00019.png, whose files are grayscale
# reductions of these canonical FFHQ indices. Everything else trains.
N_COLOR_TEST = 20

_IMAGES_NPY = "images_u8.npy"
_META_JSON = "meta.json"

# Per-image noise-stream stride: the corruption generator for memmap row
# ``r`` under ``degradation_seed`` ``s`` is seeded ``s * stride + r``,
# giving disjoint streams for every (seed, row) with rows < stride.
_NOISE_SEED_STRIDE = 100_000_019


class _LazyImageView:
    """Sliceable lazy stand-in for the eager ``clean`` / ``noisy`` tensors.

    Holding full FFHQ in color as float copies would cost tens of gigabytes,
    so this view materializes float32 batches on access from the uint8 memmap
    and leaves the caching to the OS page cache. It supports exactly the
    access grammar the FFHQ trainer uses on the eager tensors:
    ``view[tensor_idx]``, ``view[list_idx]`` and ``view[slice]``, each a
    ``(B, C, H, W)`` float32 tensor, ``view[int]`` giving ``(C, H, W)``,
    ``len``, ``.shape`` and ``.split(chunk)``.

    The degraded view applies the fixed corruption recipe on the fly, from a
    fresh per-image generator, so the realization is identical across
    accesses, epochs and ``max_images`` truncations.
    """

    def __init__(self, pairs: "FFHQColorPairs", degraded: bool,
                 mean_match: bool = False) -> None:
        self._pairs = pairs
        self._degraded = bool(degraded)
        self._mean_match = bool(mean_match)

    @property
    def shape(self) -> torch.Size:
        p = self._pairs
        return torch.Size((p._n, 3, p.img_size, p.img_size))

    def __len__(self) -> int:
        return int(self._pairs._n)

    def __getitem__(self, idx) -> torch.Tensor:
        single = isinstance(idx, int) or (
            isinstance(idx, torch.Tensor) and idx.dim() == 0
        )
        if single:
            idx = [int(idx)]
        if isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self))))
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().to(torch.int64).tolist()
        idx = [int(i) + len(self) if int(i) < 0 else int(i) for i in idx]
        for i in idx:
            if not 0 <= i < len(self):
                raise IndexError(
                    f"index {i} out of range for split of {len(self)}",
                )
        out = self._pairs._materialize(
            idx, degraded=self._degraded, mean_match=self._mean_match,
        )
        return out[0] if single else out

    def split(self, chunk: int):
        """Yield consecutive ``(<=chunk, C, H, W)`` batches (torch.split
        semantics on dim 0), lazily."""
        for start in range(0, len(self), int(chunk)):
            yield self[start:start + int(chunk)]


class FFHQColorPairs:
    """Clean / degraded pairs over the prepared color FFHQ memmap.

    The color sibling of :class:`FFHQPairs`.
    ``scripts/prepare_ffhq_color.py`` materializes ``data_root`` as a single
    uint8 memmap ``images_u8.npy`` of shape ``(N, 3, S, S)``, where row ``i``
    is canonical FFHQ image ``i``, plus ``meta.json``. The test split is rows
    ``0..19``, the same 20 identities as the Ehrhardt grayscale test set, and
    the train split is every remaining row. Pixel decoding matches the
    grayscale loader: uint8 over 255 in float64, then the float32 cast.

    The degradation recipe is identical to :class:`FFHQPairs` per element.
    ``task="inpaint"`` is ``mask * img + noise_sigma * randn`` with the same
    shipped Ehrhardt ``mask.npy``, single-channel ``(1, 1, 256, 256)``, which
    broadcasts across the three channels so the same pixels are removed in
    every channel and is nearest-neighbor downsampled first if the prepared
    resolution is below 256. ``task="snp"`` sets elements below ``snp_prob/2``
    to 0 and above ``1 - snp_prob/2`` to 1, independently per channel. Lazy
    loading forces one mechanical deviation: instead of one generator drawing
    the whole ``(N, C, H, W)`` realization at construction, each image draws
    from its own generator seeded by
    ``degradation_seed * 100_000_019 + memmap_row``. The realization is still
    fixed across accesses and epochs, reproducible per seed and independent
    of ``max_images``.

    Args:
        split: ``"train"`` (rows 20..N-1) or ``"test"`` (rows 0..19).
        task, degradation_seed, noise_sigma, snp_prob: as in
            :class:`FFHQPairs`.
        data_root: prepared color root (``images_u8.npy`` +
            ``meta.json``); ``None`` / ``""`` -> the repository default
            ``data/datasets/ffhq_color``.
        max_images: optional truncation of the split, a test hook.
        mean_match_negatives: as in :class:`FFHQPairs`, applied on the fly
            inside the lazy ``negatives`` view.

    Attributes:
        clean, noisy: lazy ``(N_split, 3, S, S)`` float32 views
            (:class:`_LazyImageView`).
        negatives: lazy view of the critic negatives; ``noisy`` itself
            unless ``mean_match_negatives`` is set.
        mask: ``(1, 1, S, S)`` float tensor (shipped mask, resized if
            needed).
    """

    def __init__(
        self,
        split: str = "train",
        task: str = "inpaint",
        degradation_seed: int = 0,
        data_root: str | None = None,
        noise_sigma: float = 0.03,
        snp_prob: float = 0.25,
        max_images: int | None = None,
        mean_match_negatives: bool = False,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        self.split = str(split)
        self.task = str(task)
        self.degradation_seed = int(degradation_seed)
        self.noise_sigma = float(noise_sigma)
        self.snp_prob = float(snp_prob)
        self.mean_match_negatives = bool(mean_match_negatives)

        root = str(data_root) if data_root else _DEFAULT_COLOR_ROOT
        self.data_root = root
        npy_path = os.path.join(root, _IMAGES_NPY)
        meta_path = os.path.join(root, _META_JSON)
        if not (os.path.isfile(npy_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError(
                f"prepared color FFHQ not found under {root!r} (need "
                f"{_IMAGES_NPY} + {_META_JSON}); run "
                f"scripts/prepare_ffhq_color.py first.",
            )
        with open(meta_path) as f:
            meta = json.load(f)
        self.meta = meta
        self._images = np.load(npy_path, mmap_mode="r")
        if self._images.ndim != 4 or self._images.shape[1] != 3 \
                or self._images.dtype != np.uint8:
            raise ValueError(
                f"{npy_path!r} must be uint8 of shape (N, 3, S, S); got "
                f"{self._images.dtype} {self._images.shape}",
            )
        count = int(meta.get("count", self._images.shape[0]))
        if count != int(self._images.shape[0]):
            raise ValueError(
                f"meta.json count={count} disagrees with memmap rows "
                f"{int(self._images.shape[0])} under {root!r}; re-run "
                f"scripts/prepare_ffhq_color.py.",
            )
        if count <= N_COLOR_TEST:
            raise ValueError(
                f"prepared root holds {count} images; need more than "
                f"{N_COLOR_TEST} (rows 0..{N_COLOR_TEST - 1} are the "
                f"test split).",
            )
        self.img_size = int(self._images.shape[-1])
        self.n_channels = 3

        if split == "test":
            row0, n = 0, N_COLOR_TEST
        else:
            row0, n = N_COLOR_TEST, count - N_COLOR_TEST
        if max_images is not None and int(max_images) > 0:
            n = min(n, int(max_images))
        self._row0, self._n = int(row0), int(n)

        self.mask = self._load_mask(root)
        self.clean = _LazyImageView(self, degraded=False)
        self.noisy = _LazyImageView(self, degraded=True)
        if self.mean_match_negatives:
            self.negatives = _LazyImageView(self, degraded=True,
                                            mean_match=True)
        else:
            self.negatives = self.noisy

    def _load_mask(self, root: str) -> torch.Tensor:
        """Shipped Ehrhardt mask, from ``<root>/mask.npy`` if the prepare
        script dropped a copy there and otherwise from the unpacked archive,
        nearest-neighbor downsampled when the resolution is below 256."""
        local = os.path.join(root, "mask.npy")
        if os.path.isfile(local):
            mask = torch.load(local, map_location="cpu")
        else:
            archive = resolve_archive_dir(None)
            mask_path = os.path.join(archive, "data", "mask.npy")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(
                    f"shipped inpainting mask missing at {mask_path!r} "
                    f"(Zenodo DOI {_ZENODO_DOI}) and no {local!r}.",
                )
            mask = torch.load(mask_path, map_location="cpu")
        if int(mask.shape[-1]) != self.img_size:
            # Nearest-neighbor keeps the mask exactly 0/1; an averaging
            # kernel would blur the removed-pixel pattern.
            mask = torch.nn.functional.interpolate(
                mask, size=(self.img_size, self.img_size), mode="nearest",
            )
        return mask

    def _materialize(self, idx: list[int], degraded: bool,
                     mean_match: bool = False) -> torch.Tensor:
        rows = [self._row0 + i for i in idx]
        u8 = np.asarray(self._images[rows])
        # uint8 over 255 in float64 before the float32 cast: the grayscale
        # loader's decode semantics, bit for bit.
        clean = torch.from_numpy(
            (u8.astype(np.float64) / 255.0).astype(np.float32),
        )
        if not degraded:
            return clean
        out = torch.empty_like(clean)
        for k, row in enumerate(rows):
            gen = torch.Generator().manual_seed(
                self.degradation_seed * _NOISE_SEED_STRIDE + int(row),
            )
            img = clean[k]
            if self.task == "inpaint":
                noise = torch.randn(img.shape, generator=gen)
                out[k] = self.mask[0] * img + self.noise_sigma * noise
            else:
                probs = torch.rand(img.shape, generator=gen)
                noisy = img.clone()
                noisy[probs < self.snp_prob / 2] = 0.0
                noisy[probs > 1 - self.snp_prob / 2] = 1.0
                out[k] = noisy
        if mean_match:
            out = dc_matched_negatives(clean, out)
        return out

    def __len__(self) -> int:
        return int(self._n)
