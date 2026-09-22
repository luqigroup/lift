"""Readers for the full-training-data re-evaluation of the lift.

``reeval_ebm_pinned.py`` regenerates every lift result with the entire
training data passed to the hypernetwork at evaluation, writing one file
pair per archive under ``data/analysis/pinned/``. Figure and statistics
scripts read the lift's numbers from here under ``--pinned 1``, their
default, and from the archives' own batch-conditioned records otherwise.
Direct softplus and PGD read no batch, so their records are unchanged. A
missing pinned file raises instead of falling back, because one figure must
not mix the two rules.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
from projorg import gitdir

_REPO = gitdir()
PINNED_DIR = os.path.join(_REPO, "data", "analysis", "pinned")

# archive stem per experiment line, as reeval_ebm_pinned.py wrote them
_ONE_D = "lcmm_1d_poolnorm_{target}"
_TABULAR = {"power": "uci_power_lift", "hepmass": "uci_hepmass_lift", "miniboone": "uci_miniboone_lift"}
_MNIST_ENSEMBLE = "mnist_per_class_ensemble_lift"


def _path(stem: str, ext: str) -> str:
    p = os.path.join(PINNED_DIR, stem + ext)
    if not os.path.exists(p):
        raise FileNotFoundError(f"pinned re-evaluation missing: {p} (run scripts/reeval_ebm_pinned.py)")
    return p


def one_d_json(target: str) -> Dict[str, dict]:
    """``seed -> {pinned, batch, recorded}`` metric dicts of a 1-D target."""
    with open(_path(_ONE_D.format(target=target), ".json")) as f:
        return json.load(f)["seeds"]


def one_d_tv(target: str) -> Dict[int, float]:
    """Pinned final TV per seed of a 1-D target."""
    return {int(s): float(v["pinned"]["tv"]) for s, v in one_d_json(target).items()}


def one_d_log_p_grid(target: str, seed: int) -> np.ndarray:
    """Pinned log-density on the archive's evaluation grid, one seed."""
    with np.load(_path(_ONE_D.format(target=target), ".npz")) as z:
        return np.asarray(z[f"seed{seed}/log_p_grid_pinned"], dtype=np.float64)


def one_d_tv_history(target: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(iterations, TV)`` of the pinned emission at every stored snapshot, one seed."""
    with np.load(_path(_ONE_D.format(target=target), ".npz")) as z:
        return (np.asarray(z[f"seed{seed}/tv_hist_iters"], dtype=float),
                np.asarray(z[f"seed{seed}/tv_hist_pinned"], dtype=float))


def tabular_test_nll(dataset: str, seed: int) -> Optional[float]:
    """Pinned final test NLL of the tabular lift; ``None`` if that seed was not re-evaluated."""
    with open(_path(_TABULAR[dataset], ".json")) as f:
        rec = json.load(f)["seeds"].get(str(seed))
    return None if rec is None else float(rec["pinned"]["test_nll"])


def mnist_class_test_nll(c: int) -> float:
    """Pinned test NLL of the per-class MNIST ensemble's lift, class ``c``."""
    with open(_path(_MNIST_ENSEMBLE, ".json")) as f:
        return float(json.load(f)["classes"][str(c)]["pinned"]["test_nll"])
