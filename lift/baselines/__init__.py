"""Baseline stacks: PICNN / PCP-Map transport maps, vendored CPFlow
primitives, and the single-Gaussian warm-start.
"""

from lift.baselines.single_gaussian_init import (
    make_single_gaussian_init_callback,
    pretrain_ebm_to_gaussian,
)

# Alias so ``lift.baselines.icnn`` resolves; the canonical module is
# :mod:`lift.models.icnn`.
from lift.models import icnn as _icnn_module  # noqa: F401
import sys as _sys
_sys.modules[__name__ + ".icnn"] = _icnn_module
del _sys

__all__ = [
    "make_single_gaussian_init_callback",
    "pretrain_ebm_to_gaussian",
]
