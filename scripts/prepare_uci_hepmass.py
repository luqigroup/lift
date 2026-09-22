r"""Prepare the UCI HEPMASS density-estimation split (MAF / CP-Flow recipe).

Downloads the HEPMASS CSVs (Baldi et al. 2016, UCI ML repository dataset 347),
applies the MAF (Papamakarios et al., 2017) preprocessing pipeline published at
https://github.com/gpapamak/maf/blob/master/datasets/hepmass.py, and writes
``train/val/test.npy`` under ``data/datasets/uci/hepmass/``.

MAF recipe, in order:

1. Drop the label column (column 0).
2. Restrict to the positive (``# label`` == 1) class.
3. Drop the last column of the test split: the released test CSV has an extra
   orphan column that the train CSV does not.
4. Whiten train and test on the train mean and standard deviation.
5. Drop the quantized feature columns, ranked by the occurrence count of each
   column's most common value; see :func:`_maf_discrete_columns`, which keeps
   MAF's ``D = 21``.
6. Hold out the last 10 percent of the train slice as ``val``.

The output schema matches ``lift/dataset/uci.py``: three ``.npy`` files
holding ``(N, 21)`` float32 arrays.

Usage::

    python scripts/prepare_uci_hepmass.py \
        [--raw_dir data/datasets/uci/hepmass/raw] \
        [--out_dir data/datasets/uci/hepmass]
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import urllib.request
from collections import Counter
from typing import Tuple

import numpy as np


_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00347"
_TRAIN_GZ = "all_train.csv.gz"
_TEST_GZ = "all_test.csv.gz"


def _download(url: str, dst: str) -> None:
    """Stream a single URL to disk with a coarse progress print."""
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  skip download (already present): {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  GET {url}  ->  {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        chunk = 1 << 20  # 1 MB
        total = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            total += len(buf)
            if total % (32 << 20) < chunk:
                print(f"    ... {total >> 20} MB")
    print(f"  done {dst} ({os.path.getsize(dst) >> 20} MB)")


def _load_csv_gz(path: str) -> np.ndarray:
    """Load a HEPMASS CSV.gz into an ``(N, D+1)`` float64 array.

    The first column is the label (``# label``) and the rest are the 28
    features. The MAF reference reads with the pandas default, float64, so the
    Counter-based discrete-column detector below sees exact repeats on the raw
    float values. The data stays float64 here and is cast to float32 only at
    the end.
    """
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip().split(",")
        data = np.loadtxt(f, delimiter=",", dtype=np.float64)
    if data.shape[1] != len(header):
        raise ValueError(
            f"shape/header mismatch on {path}: data has "
            f"{data.shape[1]} columns, header has {len(header)}.",
        )
    return data


def _maf_discrete_columns(data: np.ndarray, n_drop: int = 7):
    """Drop the ``n_drop`` most quantized columns.

    MAF drops the columns whose max-value-count exceeds 5 on the whitened
    pandas float64 representation, which yields D = 21 on HEPMASS. That
    literal threshold is brittle to numpy against pandas float-precision drift
    at large N, so the columns with the highest max-count are dropped instead,
    with ``n_drop = 7`` reproducing MAF's D = 21.

    Returns the kept column indices, the dropped ones, and the per-column
    max-value-counts.
    """
    max_counts = []
    for j in range(data.shape[1]):
        c = Counter(data[:, j].tolist())
        max_counts.append(c.most_common(1)[0][1])
    max_counts = np.asarray(max_counts)
    drop = sorted(np.argsort(max_counts)[-int(n_drop):].tolist())
    keep = np.array(
        [j for j in range(data.shape[1]) if j not in drop], dtype=int,
    )
    return keep, drop, max_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_root = os.path.join(here, "data", "datasets", "uci", "hepmass")
    parser.add_argument(
        "--raw_dir", type=str, default=os.path.join(default_root, "raw"),
    )
    parser.add_argument(
        "--out_dir", type=str, default=default_root,
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.10,
        help="Fraction of train held out as val (MAF default 0.10).",
    )
    parser.add_argument(
        "--n_drop", type=int, default=7,
        help="Number of most-quantised columns to drop (MAF reports 7, D=21).",
    )
    parser.add_argument(
        "--keep_raw", action="store_true",
        help="Don't delete the raw CSV.gz after preparation.",
    )
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    train_gz = os.path.join(args.raw_dir, _TRAIN_GZ)
    test_gz = os.path.join(args.raw_dir, _TEST_GZ)
    _download(f"{_UCI_URL}/{_TRAIN_GZ}", train_gz)
    _download(f"{_UCI_URL}/{_TEST_GZ}", test_gz)

    print("loading raw CSVs ...")
    raw_train = _load_csv_gz(train_gz)   # (N, 29) -- label + 28 features
    raw_test = _load_csv_gz(test_gz)     # (N_test, 30) -- one extra trailing col

    print(f"  raw train: {raw_train.shape}")
    print(f"  raw test:  {raw_test.shape}")

    # Step 3 (per MAF): drop test's trailing orphan column.
    if raw_test.shape[1] == raw_train.shape[1] + 1:
        raw_test = raw_test[:, :-1]
        print(f"  dropped trailing test column -> {raw_test.shape}")

    # Step 1+2: drop label col, restrict to positive class.
    train_pos = raw_train[raw_train[:, 0] == 1.0, 1:]
    test_pos = raw_test[raw_test[:, 0] == 1.0, 1:]
    print(f"  positive train: {train_pos.shape}")
    print(f"  positive test:  {test_pos.shape}")

    # MAF computes mu and sd, whitens, and only then runs the Counter-based
    # discrete-column scan; the same order is followed here, in float64.
    mu = train_pos.mean(axis=0, keepdims=True)
    sd = train_pos.std(axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    train_w = (train_pos - mu) / sd
    test_w = (test_pos - mu) / sd

    # Step 5: drop the ``n_drop`` columns with the highest max-value-count,
    # which on HEPMASS are the six mass columns plus one weakly discrete
    # column.
    keep_idx, drop_idx, max_counts = _maf_discrete_columns(
        train_w, n_drop=int(args.n_drop),
    )
    print(f"  kept {len(keep_idx)}/{train_w.shape[1]} cols  "
          f"(D={len(keep_idx)}); dropped: {drop_idx} "
          f"(max-counts at the drop boundary: "
          f"{sorted(max_counts.tolist(), reverse=True)[:int(args.n_drop)+2]})")

    train_w = train_w[:, keep_idx]
    test_w = test_w[:, keep_idx]
    mu = mu[:, keep_idx]
    sd = sd[:, keep_idx]

    # Step 6: hold out the last val_frac of train as val.
    n_train = train_w.shape[0]
    n_val = int(round(args.val_frac * n_train))
    val_arr = train_w[-n_val:]
    train_arr = train_w[:-n_val] if n_val > 0 else train_w

    train_arr = train_arr.astype(np.float32)
    val_arr = val_arr.astype(np.float32)
    test_arr = test_w.astype(np.float32)

    np.save(os.path.join(args.out_dir, "train.npy"), train_arr)
    np.save(os.path.join(args.out_dir, "val.npy"), val_arr)
    np.save(os.path.join(args.out_dir, "test.npy"), test_arr)
    np.savez(
        os.path.join(args.out_dir, "norm.npz"),
        mu=mu, sd=sd, keep_idx=keep_idx, drop_idx=np.asarray(drop_idx, dtype=int),
    )
    print(f"  wrote {args.out_dir}/{{train,val,test}}.npy "
          f"with D={train_arr.shape[1]}, "
          f"N_train={train_arr.shape[0]}, N_val={val_arr.shape[0]}, "
          f"N_test={test_arr.shape[0]}.")

    if not args.keep_raw:
        try:
            os.remove(train_gz)
            os.remove(test_gz)
            print(f"  removed raw CSV.gz files in {args.raw_dir}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
