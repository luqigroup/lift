"""Code for "A lift for input-convex neural net training".

The lift emits the constrained weights of an input-convex network from a
permutation-invariant summary of the training batch, instead of storing them as
free latent weights behind a positivity map. :class:`Lift` applies it to any
module in one line.
"""

from lift.lift import Lift, tag_positive_
from lift.models import (
    ConvICNN,
    HyperNetwork,
    ICNN,
    LogConcaveEBM,
    conv_icnn_pos_param_names,
    icnn_pos_param_names,
    infer_pos_param_names,
)

__version__ = "1.0.0"

__all__ = [
    "Lift",
    "tag_positive_",
    "ICNN",
    "ConvICNN",
    "HyperNetwork",
    "LogConcaveEBM",
    "icnn_pos_param_names",
    "conv_icnn_pos_param_names",
    "infer_pos_param_names",
]
