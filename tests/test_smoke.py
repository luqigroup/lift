"""Smoke-import tests for the lift package.

Every paper-relevant module is imported once at collection time. Any
ImportError (e.g. a renamed symbol) surfaces immediately. The test is
intended to be fast (<5 s on CPU) and so does NOT exercise end-to-end
training; per-script smoke runs (with ``--n_iters 1 --seeds 0``) are
left to the caller.
"""

from __future__ import annotations


def test_package_importable():
    import lift

    assert lift.__version__ == "1.0.0"


def test_models_importable():
    from lift.models.icnn import ICNN  # noqa: F401
    from lift.models.hypernet import HyperNetwork  # noqa: F401
    from lift.models.log_concave_ebm import LogConcaveEBM  # noqa: F401
    from lift.models.conv_icnn import ConvICNN  # noqa: F401
    from lift.models.conv_summary import ConvSummaryEncoder  # noqa: F401
    from lift.models.sampler_flow import SamplerFlow  # noqa: F401
    from lift.models.multi_sampler_flow import MultiSamplerFlow  # noqa: F401


def test_baselines_importable():
    from lift.baselines.picnn import PICNN  # noqa: F401
    from lift.baselines.picnn_hypernet import PicnnHypernet  # noqa: F401
    from lift.baselines.pcpmap import PCPMap  # noqa: F401
    from lift.baselines.single_gaussian_init import (  # noqa: F401
        pretrain_ebm_to_gaussian,
    )
    from lift.baselines.ehrhardt_vendor import (  # noqa: F401
        PDHG_ICNN,
        simple_ICNN,
        simple_ICNNPrior,
        soft_thresh,
    )


def test_objectives_importable():
    from lift.objectives.builders import (  # noqa: F401
        build_hypernet_ebm,
        build_hypernet_sampler_flow,
    )
    from lift.objectives.trainer import train_ebm  # noqa: F401
    from lift.objectives.eval import (  # noqa: F401
        hypernet_ebm_logp,
        build_eval_grid,
    )


def test_datasets_importable():
    from lift.dataset.gumbel import Gumbel1D  # noqa: F401
    from lift.dataset.real2d import get_real_target_2d  # noqa: F401
    from lift.dataset.uci import (  # noqa: F401
        get_uci_target,
        HEPMASS,
        MINIBOONE,
        POWER,
    )
    from lift.dataset.ffhq import FFHQPairs  # noqa: F401
    from lift.dataset.darcy_kl import DarcyKL  # noqa: F401
    from lift.dataset.logconcave_1d import Beta1D, Gamma1D, Laplace1D  # noqa: F401
    from lift.dataset.toy2d import EightGaussians  # noqa: F401
