"""ICNN architectures vendored from CP-Flow ``lib/icnn.py``.

Only ``ICNN2`` (the default tabular architecture) and its dependencies
are kept. Vendored verbatim except for the import path of
``ActNormNoLogdet`` (which now sits in
``lift.baselines.cpflow_vendor.flows``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Forward-declared to break the import cycle between icnn.py and flows.py;
# ``cpflow_vendor/__init__`` populates it once flows.py has imported.
ActNormNoLogdet = None


def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (
        z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x ** 2 / 2) + z * x
    ) / (2 * z)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))
    return shifted_activation


def get_softplus(softplus_type: str = "softplus", zero_softplus: bool = False):
    if softplus_type == "softplus":
        act = nn.functional.softplus
    elif softplus_type == "gaussian_softplus":
        act = gaussian_softplus
    else:
        raise NotImplementedError(f"softplus type {softplus_type} not supported.")
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type: str = "softplus", zero_softplus: bool = False):
        super().__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class PosLinear(torch.nn.Linear):
    """Linear with ``softplus``-reparameterized non-negative weights."""

    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return F.linear(x, F.softplus(self.weight), self.bias) * gain


class ICNN2(torch.nn.Module):
    r"""ICNN2: input-convex MLP from CP-Flow's tabular experiments.

    Convex in ``x`` by construction (positive Wz, free Wx, monotone
    activation). ActNormNoLogdet inside each hidden layer is also
    permitted (it is affine in x with positive scale, preserving
    convexity).

    Parameters mirror CP-Flow's defaults: ``dim`` is the data
    dimensionality, ``dimh`` the hidden width, ``num_hidden_layers``
    the depth.
    """

    def __init__(
        self,
        dim: int = 2,
        dimh: int = 16,
        num_hidden_layers: int = 2,
        symm_act_first: bool = False,
        softplus_type: str = "softplus",
        zero_softplus: bool = False,
    ) -> None:
        super().__init__()
        from lift.baselines.cpflow_vendor.flows import ActNormNoLogdet as _AN
        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = [nn.Linear(dim, dimh)]
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        self.actnorm0 = _AN(dimh)
        actnorms = []
        for _ in range(num_hidden_layers - 1):
            actnorms.append(_AN(dimh))
        actnorms.append(_AN(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x: Tensor) -> Tensor:
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        for Wz, Wx, actnorm in zip(
            self.Wzs[1:-1], self.Wxs[:-1], self.actnorms[:-1]
        ):
            z = self.act(actnorm(Wz(z) + Wx(x)))
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))
