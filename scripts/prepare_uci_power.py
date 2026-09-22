#!/usr/bin/env python
"""Prepare the UCI POWER density-estimation split (MAF recipe).

Fetches the benchmark archive of Papamakarios et al. (2017) if it is not
already on disk, applies their published preprocessing for POWER
(https://github.com/gpapamak/maf/blob/master/datasets/power.py), and writes
``train/val/test.npy`` to the POWER directory the loader reads.

The recipe, in order:

1. Shuffle the 2,049,280 raw rows with ``RandomState(42)``.
2. Drop columns 3 then 1, leaving D = 6.
3. Add the published noise: uniform on [0, 0.001) to the gap column, on
   [0, 0.01) to the voltage column, on [0, 1) to the three sub-metering
   columns, and none to the time column.
4. Hold out the last 10 percent as test, then the last 10 percent of the
   remainder as validation.
5. Whiten all three splits on the train-plus-validation mean and standard
   deviation.

Usage::

    python scripts/prepare_uci_power.py
"""
from __future__ import annotations

import argparse
import os
import tarfile

import numpy as np
from projorg import datadir

from lift.download import ensure, path_of

ARCHIVE_KEY = "datasets/uci/_maf_cache/data.tar.gz"
MEMBER = "data/power/data.npy"


def raw_array() -> np.ndarray:
    """The raw POWER matrix, extracting it from the archive on first use."""
    cache = os.path.join(datadir("datasets/uci/_maf_cache"), "data", "power", "data.npy")
    if not os.path.exists(cache):
        archive = ensure(ARCHIVE_KEY)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extract(tar.getmember(MEMBER), path=path_of(ARCHIVE_KEY).rsplit(os.sep, 1)[0])
    return np.load(cache)


def splits(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, validation and test, whitened, following MAF's power.py."""
    rng = np.random.RandomState(42)
    rng.shuffle(data)
    n = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)

    voltage_noise = 0.01 * rng.rand(n, 1)
    gap_noise = 0.001 * rng.rand(n, 1)
    sm_noise = rng.rand(n, 3)
    time_noise = np.zeros((n, 1))
    data = data + np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))

    n_test = int(0.1 * data.shape[0])
    test, data = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * data.shape[0])
    val, train = data[-n_val:], data[:-n_val]

    both = np.vstack((train, val))
    mu, sd = both.mean(axis=0), both.std(axis=0)
    return ((train - mu) / sd, (val - mu) / sd, (test - mu) / sd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=None,
                   help="destination; default is the POWER directory the loader reads")
    a = p.parse_args()

    out = a.out_dir or datadir("datasets/uci/power")
    train, val, test = splits(raw_array())
    for name, arr in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(out, f"{name}.npy")
        np.save(path, arr.astype(np.float32))
        print(f"wrote {path}  {arr.shape}")


if __name__ == "__main__":
    main()
