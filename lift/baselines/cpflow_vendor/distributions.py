"""Log-density helpers vendored from CP-Flow ``lib/distributions.py``."""

from __future__ import annotations

import numpy as np
import torch

Log2PI = float(np.log(2 * np.pi))


def log_standard_normal(x: torch.Tensor) -> torch.Tensor:
    z = -0.5 * Log2PI
    return -x ** 2 / 2 + z
