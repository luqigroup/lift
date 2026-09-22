"""Vendored subset of the Ehrhardt/Mukherjee/Wong ICNN primal-dual code.

Upstream: ``hsw43/icnn_primal_dual``, commit ``bb0d0bd``, archived at Zenodo
DOI 10.5281/zenodo.17426033 (MIT license, copyright 2025 "hsw43").

Covers the FFHQ ICNN-regularizer surface: the ``simple_ICNN`` architecture
with its clamp-to-zero positivity, the PDHG inpainting solver with the
smoothed epigraph projection, and the L1 primal-dual loop for salt-and-pepper
denoising. Per-file PROVENANCE headers carry the patch lists; in summary:

* ``models`` : ``simple_ICNN`` / ``deeper_ICNN`` + the ``*Prior`` grad
  wrappers; the ``deepinv.optim.Prior`` base replaced by ``nn.Module``.
* ``solver`` : ``PDHG_ICNN``, ``proj_epi``, ``cubic``, plus ``power`` /
  ``proj_Lrelu`` extracted from their ``utils.py`` (severing its top-level
  ``odl``/CT import), ``soft_thresh`` rewritten from the L1 prox, and
  ``PDHG_ICNN_L1`` lifted from ``example_denoise.py``.
* ``physics`` : the deepinv/skimage surface their FFHQ line touches
  (inpainting operator, L2 fidelity, their exact PSNR call), rewritten to
  drop both dependencies.
"""

from lift.baselines.ehrhardt_vendor.models import (
    deeper_ICNN,
    deeper_ICNNPrior,
    simple_ICNN,
    simple_ICNNPrior,
)
from lift.baselines.ehrhardt_vendor.physics import (
    InpaintingPhysics,
    L2Fidelity,
    compare_psnr,
)
from lift.baselines.ehrhardt_vendor.solver import (
    PDHG_ICNN,
    PDHG_ICNN_L1,
    cubic,
    power,
    proj_Lrelu,
    proj_epi,
    soft_thresh,
)

__all__ = [
    "simple_ICNN",
    "simple_ICNNPrior",
    "deeper_ICNN",
    "deeper_ICNNPrior",
    "InpaintingPhysics",
    "L2Fidelity",
    "compare_psnr",
    "PDHG_ICNN",
    "PDHG_ICNN_L1",
    "cubic",
    "power",
    "proj_Lrelu",
    "proj_epi",
    "soft_thresh",
]
