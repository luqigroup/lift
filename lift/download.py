"""Automatic download of the public data the experiments train on.

Every artifact an experiment reads is public and fetched on first use. Each
preparation script calls :func:`ensure` on what it needs; the call is a no-op
once the file is on disk. Destinations are resolved by :func:`projorg.datadir`,
so a registry key is a location under the project's data directory, which
``projorg`` turns into an absolute path and creates.

Two tiers, one per artifact family:

``uci``     the tabular density-estimation benchmark of Papamakarios et al.
            (2017), as one archive holding POWER, HEPMASS, MiniBooNE and
            BSDS300. The ``prepare_uci_*`` scripts turn it into the whitened
            splits the loaders read.
``ffhq``    the release of Ehrhardt et al. (2026), which carries the grayscale
            FFHQ images and the inpainting mask their convex regularizer is
            trained on.

The groundwater archive is in no tier because it is generated rather than
downloaded; its generator needs ``priorlaundermat`` (see the README).

The FFHQ imagery inside the Ehrhardt archive is NVIDIA's, under CC BY-NC-SA 4.0.
It is fetched from its own publisher and never redistributed here.
"""

from __future__ import annotations

import os
import sys
import urllib.request

from projorg import datadir

# key under the project's data directory -> (tier, direct-download URL, source).
REGISTRY: dict[str, tuple[str, str, str]] = {
    "datasets/uci/_maf_cache/data.tar.gz": (
        "uci",
        "https://zenodo.org/records/1161203/files/data.tar.gz",
        "Papamakarios et al. (2017) benchmark data, Zenodo record 1161203 (857 MB)",
    ),
    "third_party/ehrhardt_icnn_primal_dual.zip": (
        "ffhq",
        "https://zenodo.org/api/records/17426033/files/"
        "hsw43/icnn_primal_dual-icnn_primal_dual.zip/content",
        "Ehrhardt et al. (2026) release, DOI 10.5281/zenodo.17426033 (52 MB)",
    ),
    "datasets/darcy/darcy_kl_50k.h5": (
        "darcy",
        "https://www.dropbox.com/scl/fi/un9jaamucwj9x6i4d7b46/darcy_kl_50k.h5"
        "?rlkey=oqx70u88tfmr900owfx2lsfdw&dl=1",
        "the generated groundwater archive, fifty thousand training pairs and "
        "two thousand evaluation pairs (46 MB)",
    ),
}


def path_of(key: str) -> str:
    """The absolute path ``key`` resolves to, creating its directory.

    The directory comes from :func:`projorg.datadir`, which locates the project
    root itself, so nothing here depends on where this file sits.
    """
    subdir, name = os.path.split(key)
    return os.path.join(datadir(subdir), name)


def _hook(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100.0, 100.0 * count * block / total)
    sys.stdout.write(f"\r  {pct:5.1f}% of {total / 1e6:.0f} MB")
    sys.stdout.flush()
    if pct >= 100.0:
        sys.stdout.write("\n")


def ensure(*keys: str) -> "str | tuple[str, ...]":
    """Return absolute paths for ``keys``, downloading any that are missing.

    Args:
        *keys: registry keys, e.g. ``"datasets/uci/_maf_cache/data.tar.gz"``.

    Returns:
        The absolute path if one argument was given, else a tuple of them.

    Raises:
        RuntimeError: the key is not a registered public artifact, or the
            download returned a web page instead of the file.
    """
    out = tuple(_ensure_one(k) for k in keys)
    return out[0] if len(out) == 1 else out


def _ensure_one(key: str) -> str:
    entry = REGISTRY.get(key)
    if entry is None:
        raise RuntimeError(
            f"{key} is not a public artifact this repository knows how to fetch. "
            f"The README's data table names the script that produces it.")
    _, url, source = entry

    path = path_of(key)
    if os.path.exists(path):
        return path

    print(f"[lift] {key} not found; fetching once from {source}.")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, _hook)
        # .tar.gz and .zip carry their own magic; an expired or redirected link
        # returns HTML, which would otherwise be unpacked as a corrupt archive.
        with open(tmp, "rb") as f:
            head = f.read(4)
        if head[:1] == b"<":
            raise RuntimeError(
                f"the link for {key} returned a web page, not the file; "
                f"see the data section of the README.")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    print(f"[lift] saved {key} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def ensure_tier(tier: str) -> None:
    """Download every artifact in one tier ahead of time (see the module docstring)."""
    keys = [k for k, (t, _, _) in REGISTRY.items() if t == tier]
    if not keys:
        raise RuntimeError(f"unknown tier {tier!r}; known tiers: "
                           f"{sorted({t for t, _, _ in REGISTRY.values()})}")
    for k in sorted(keys):
        _ensure_one(k)


def sources() -> str:
    """A printable table of every artifact, its tier and where it comes from."""
    rows = [(k, tier, source) for k, (tier, _, source) in sorted(REGISTRY.items())]
    w = max(len(r[0]) for r in rows)
    return "\n".join(f"{k:<{w}}  {tier:<6}  {source}" for k, tier, source in rows)
