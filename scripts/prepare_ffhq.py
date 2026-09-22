#!/usr/bin/env python
"""Fetch and unpack the FFHQ inpainting data the image regularizer trains on.

Downloads the release of Ehrhardt et al. (2026) from Zenodo and unzips it where
``lift.dataset.ffhq`` looks for it. The archive holds the grayscale FFHQ images
and the inpainting mask of their published setup.

The FFHQ imagery is NVIDIA's, under CC BY-NC-SA 4.0. It is fetched from its
publisher here and is not redistributed by this repository.

Usage::

    python scripts/prepare_ffhq.py
"""
from __future__ import annotations

import argparse
import os
import zipfile

from projorg import datadir

from lift.dataset.ffhq import resolve_archive_dir
from lift.download import ensure

ARCHIVE_KEY = "third_party/ehrhardt_icnn_primal_dual.zip"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=None,
                   help="destination; default is the directory the loader reads")
    a = p.parse_args()

    out = a.out_dir or datadir("third_party/ehrhardt_icnn_primal_dual")
    try:
        found = resolve_archive_dir(out)
        print(f"already unpacked at {found}")
        return
    except FileNotFoundError:
        pass

    archive = ensure(ARCHIVE_KEY)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)
    print(f"unpacked into {out}")
    print(f"loader resolves it to {resolve_archive_dir(out)}")


if __name__ == "__main__":
    main()
