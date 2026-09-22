r"""Prepare the UCI MINIBOONE density-estimation split (MAF recipe).

Downloads the MiniBooNE particle-identification file (Roe et al., UCI ML
repo dataset 199), applies the MAF (Papamakarios et al., 2017)
preprocessing pipeline as used by MAF, RealNVP and CP-Flow, and writes
``train/val/test.npy`` under ``data/datasets/uci/miniboone/``.

MAF recipe (in order):

1. Read the header line, which gives the signal and background counts, and
   keep the signal block only (the first ``n_signal`` rows).
2. Drop every row containing the missing-value sentinel ``-999``, which on the
   released file removes 11 rows and leaves MAF's published 36,488.
3. Drop the most quantized feature columns, leaving ``D = 43``.
4. Hold out the last 10 percent as test, then the last 10 percent of the
   remainder as validation.
5. Whiten all three splits on the train-plus-validation mean and standard
   deviation.

MAF publishes its own cleaned array, and when it is present at
``data/datasets/uci/_maf_cache/data/miniboone/data.npy`` this script uses it
verbatim and applies only steps 4 and 5. That path is what the literature
reports against, and it is the one to use whenever the archive has been
fetched (https://zenodo.org/records/1161203, ``data.tar.gz``).

The raw-file path is the fallback for an absent cache. It reproduces MAF's row
cleaning exactly, 36,499 signal rows less the 11 carrying the sentinel giving
MAF's published 36,488, and recovers the right number of columns but not
necessarily the same ones: the seven columns its max-repeat heuristic drops are
not the seven MAF dropped. Step 3 carries the same deviation as
``scripts/prepare_uci_hepmass.py``, because MAF's literal rule, discarding
every column whose most-common value repeats more than five times, drops 18
columns here and yields ``D = 32``. Fallback output is approximate and cannot
support a literature comparison.

The output schema matches ``lift/dataset/uci.py``: three ``.npy`` files of
``(N, 43)`` float32.

Usage::

    python scripts/prepare_uci_miniboone.py
    python scripts/prepare_uci_miniboone.py --n_drop 7 --out_dir <dir>
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from collections import Counter

import numpy as np
from projorg import gitdir

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/"
    "MiniBooNE_PID.txt"
)
_MISSING = -999.0
_REPO = gitdir()
_DEFAULT_DIR = os.path.join(_REPO, "data", "datasets", "uci", "miniboone")
_MAF_CACHE = os.path.join(
    _REPO, "data", "datasets", "uci", "_maf_cache", "data", "miniboone", "data.npy",
)


def _download(url: str, dst: str) -> None:
    """Stream a single URL to disk, skipping a non-empty existing file."""
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  skip download (already present): {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  GET {url}  ->  {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    print(f"  done {dst} ({os.path.getsize(dst) >> 20} MB)")


def _load_signal(path: str) -> np.ndarray:
    """Return the signal block as ``(n_signal, 50)`` float64."""
    with open(path) as f:
        counts = f.readline().split()
        if len(counts) != 2:
            raise ValueError(
                f"expected a two-integer header in {path!r}, got {counts!r}.",
            )
        n_signal = int(counts[0])
        data = np.loadtxt(f, dtype=np.float64)
    if len(data) < n_signal:
        raise ValueError(
            f"{path!r} declares {n_signal} signal rows but holds {len(data)}.",
        )
    return data[:n_signal]


def _drop_quantised(data: np.ndarray, n_drop: int) -> np.ndarray:
    """Drop the ``n_drop`` columns whose most-common value repeats most."""
    if not 0 <= n_drop < data.shape[1]:
        raise ValueError(
            f"n_drop must lie in [0, {data.shape[1]}); got {n_drop}.",
        )
    max_counts = np.asarray([
        Counter(data[:, j].tolist()).most_common(1)[0][1]
        for j in range(data.shape[1])
    ])
    drop = set(np.argsort(max_counts)[-n_drop:].tolist()) if n_drop else set()
    keep = np.array([j for j in range(data.shape[1]) if j not in drop])
    print(
        f"  dropping {n_drop} quantised columns "
        f"{sorted(drop)} with max-repeat counts "
        f"{sorted(max_counts[list(drop)].tolist(), reverse=True)}",
    )
    return data[:, keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.join(_DEFAULT_DIR, "raw"))
    ap.add_argument("--out_dir", default=_DEFAULT_DIR)
    ap.add_argument("--n_drop", type=int, default=7)
    ap.add_argument("--maf_cache", default=_MAF_CACHE)
    ap.add_argument(
        "--force_raw", type=int, default=0,
        help="ignore the MAF cache and rebuild from the raw UCI file.",
    )
    args = ap.parse_args()

    if os.path.isfile(args.maf_cache) and not int(args.force_raw):
        data = np.load(args.maf_cache).astype(np.float64)
        print(f"  using MAF's cached array {args.maf_cache}  {data.shape}")
    else:
        print("  MAF cache absent; rebuilding from the raw UCI file. "
              "Column selection is approximate -- see the module docstring.")
        raw = os.path.join(args.raw_dir, "MiniBooNE_PID.txt")
        _download(_UCI_URL, raw)

        data = _load_signal(raw)
        print(f"  signal block {data.shape}")
        keep_rows = ~(data == _MISSING).any(axis=1)
        data = data[keep_rows]
        print(f"  dropped {int((~keep_rows).sum())} rows containing "
              f"{_MISSING:g}  ->  {data.shape}")

        data = _drop_quantised(data, args.n_drop)

    n_test = int(0.1 * len(data))
    test, rest = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * len(rest))
    val, train = rest[-n_val:], rest[:-n_val]

    # Whitening statistics come from train plus validation, never test.
    ref = np.vstack([train, val])
    mu, sd = ref.mean(axis=0), ref.std(axis=0)
    if not np.all(sd > 0):
        raise ValueError(
            f"{int((sd <= 0).sum())} column(s) are constant after "
            "preprocessing; whitening would divide by zero.",
        )

    os.makedirs(args.out_dir, exist_ok=True)
    for name, arr in (("train", train), ("val", val), ("test", test)):
        out = ((arr - mu) / sd).astype(np.float32)
        path = os.path.join(args.out_dir, f"{name}.npy")
        np.save(path, out)
        print(f"Saved to {path}  {out.shape}")


if __name__ == "__main__":
    main()
