"""Thin wrapper around :func:`train_ebm` with ``objective='fkl'``.

The training implementation lives in :mod:`lift.objectives.trainer`.
"""

from __future__ import annotations

from lift.objectives.trainer import train_ebm


def train_hypernet_fkl_amortized(target, K, **kwargs):
    """FKL/MLE EBM step plus amortized-SVGD sampler step.

    Equivalent to :func:`lift.objectives.trainer.train_ebm` with
    ``objective='fkl'``.
    """
    return train_ebm(target, K, objective="fkl", **kwargs)
