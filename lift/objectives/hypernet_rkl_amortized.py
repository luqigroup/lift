"""Thin wrapper around :func:`train_ebm` with ``objective='rkl'``.

The training implementation lives in :mod:`lift.objectives.trainer`.
"""

from __future__ import annotations

from lift.objectives.eval import hypernet_ebm_logp as _hypernet_ebm_logp
from lift.objectives.trainer import train_ebm


def train_hypernet_rkl_amortized(target, K, **kwargs):
    """Equivalent to :func:`lift.objectives.trainer.train_ebm` with
    ``objective='rkl'``.
    """
    return train_ebm(target, K, objective="rkl", **kwargs)


def hypernet_ebm_logp(
    ebm, hyper_E, flow, K, D, D_aug,
    target_sample, M_logZ, device,
):
    """Numpy-friendly ``log p(x)`` callable for the trained model.

    Re-exported from :mod:`lift.objectives.eval`.
    """
    return _hypernet_ebm_logp(
        ebm, hyper_E, flow,
        K=K, D=D, D_aug=D_aug,
        target_sample=target_sample, M_logZ=M_logZ, device=device,
    )
