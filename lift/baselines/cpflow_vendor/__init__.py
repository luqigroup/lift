"""Vendored subset of CP-Flow (Huang, Chen, Tsirigotis, Courville, ICLR 2021).

Upstream: https://github.com/CW-Huang/CP-Flow

Carries the tabular density-estimation training math needed for a faithful
HEPMASS replication, vendored rather than reimplemented so the experiment
stays comparable to the published table: ``icnn`` (``ICNN2``), ``logdet``
(the stochastic Lanczos and conjugate-gradient log-determinant estimators),
``flows`` (``ActNorm``, ``ActNormNoLogdet``, ``SequentialFlow``,
``DeepConvexFlow``) and ``distributions`` (``log_standard_normal``).

Patched against upstream for modern PyTorch: ``torch.symeig(M,
eigenvectors=True)`` became ``torch.linalg.eigh(M)``, and the ``tqdm`` and
command-line-only imports were removed.
"""

from lift.baselines.cpflow_vendor.icnn import ICNN2, PosLinear, Softplus
from lift.baselines.cpflow_vendor.flows import (
    ActNorm,
    ActNormNoLogdet,
    DeepConvexFlow,
    SequentialFlow,
)
from lift.baselines.cpflow_vendor.distributions import log_standard_normal

__all__ = [
    "ICNN2",
    "PosLinear",
    "Softplus",
    "ActNorm",
    "ActNormNoLogdet",
    "DeepConvexFlow",
    "SequentialFlow",
    "log_standard_normal",
]
