"""PCP-Map loss wrapper around a PICNN.

In-repo port of PCP-Map's ``src/pcpmap.py::PCPMap`` (Wang, Baptista, Marzouk,
Ruthotto, Verma, SISC 2024), https://github.com/EmoryMLIP/PCP-Map. The loss is
max log-likelihood::

    log p(x|y) = log p_prior(grad_x F(x,y)) + log det H_F(x,y)

where ``F(x,y) = softplus(w1) * PICNN(x,y) + (relu(w2)+softplus(w3)) * ||x||^2/2``
is the partially convex potential. The quadratic floor lower-bounds the smallest
eigenvalue of ``H``, so it is positive definite. The lift supplies the
convex-path weights through the ``pos_weights_override`` kwarg of
``PICNN.forward``; the wrapper here is the same for both variants.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCPMap(nn.Module):
    """PCP-Map: PICNN-parameterized partially convex potential map.

    Parameters
    ----------
    prior :
        A ``torch.distributions.MultivariateNormal`` (or any distribution
        with ``.log_prob(samples) -> (B,)``) on the state space.
    picnn :
        A ``PICNN`` instance from ``lift.baselines.picnn``.
    pos_weights_supplier :
        Optional callable that takes nothing and returns a
        ``dict[int, Tensor]`` mapping convex-path layer indices to
        externally-supplied weight tensors. Used by the lift to plug
        into ``PICNN.forward``.
    """

    def __init__(
        self,
        prior,
        picnn,
        pos_weights_supplier: Optional[Callable[[], Dict[int, torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.picnn = picnn
        self.pos_weights_supplier = pos_weights_supplier
        # Trainable scalars for the convex envelope.
        self.w1_picnn = nn.Parameter(torch.zeros(1))
        self.w2_picnn = nn.Parameter(torch.tensor(0.1))
        self.w3_picnn = nn.Parameter(torch.tensor(0.1))

    def _pos_weights(self) -> Optional[Dict[int, torch.Tensor]]:
        if self.pos_weights_supplier is None:
            return None
        return self.pos_weights_supplier()

    def get_picnn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the scalar potential F(x, y) per sample, shape (B, 1)."""
        quad = (x * x) / 2.0
        picnn_out = self.picnn(x, y, pos_weights_override=self._pos_weights())
        floor = (F.relu(self.w2_picnn) + F.softplus(self.w3_picnn)) * quad
        return (F.softplus(self.w1_picnn) * picnn_out
                + floor.sum(dim=-1, keepdim=True))

    def compute_sum(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.get_picnn(x, y).sum()

    def gxinv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the inverse transport map z = grad_x F(x, y).

        Requires ``x.requires_grad_(True)`` on entry; uses
        ``create_graph=True`` so the Hessian below is reachable.
        """
        out = self.get_picnn(x, y)
        (zx,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        return zx

    def gxinv_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-sample Hessian H = grad_x^2 F(x,y); shape (B, D_x, D_x).

        Uses ``torch.func.vmap`` over the batch dim plus
        ``torch.func.hessian`` along the x argument.
        """
        # The closure must return a sum, so that the gradient is well defined
        # on a single (x, y) sample.
        def single(x_one, y_one):
            x_one = x_one.unsqueeze(0)
            y_one = y_one.unsqueeze(0)
            return self.get_picnn(x_one, y_one).sum()

        hess_fn = torch.func.hessian(single, argnums=0)
        return torch.func.vmap(hess_fn, in_dims=(0, 0))(x, y)

    def loglik_picnn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log p(x|y) per sample, shape (B,).

        ``log p(x|y) = log p_prior(grad_x F(x,y)) + log det H_F(x,y)``.
        """
        zx = self.gxinv(x, y)
        logprob = self.prior.log_prob(zx)
        if x.shape[-1] > 1:
            hess = self.gxinv_grad(x, y)
            # H is positive definite by construction, so a Cholesky
            # factorization gives the log-determinant and is much cheaper than
            # an eigendecomposition. ``cholesky_ex`` reports failure through
            # ``info`` instead of raising, so a sample that is not numerically
            # positive definite falls back to the eigenvalue path.
            chol, info = torch.linalg.cholesky_ex(hess)
            if bool((info != 0).any()):
                eig = torch.linalg.eigvalsh(hess)
                eig = torch.clamp(eig, min=1e-12)
                logdet = torch.sum(torch.log(eig), dim=-1)
            else:
                logdet = 2.0 * torch.sum(
                    torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1,
                )
        else:
            # 1-D case: Hessian is a (B,) tensor of scalars.
            hess = self.gxinv_grad(x, y).view(-1)
            logdet = torch.log(torch.clamp(hess, min=1e-12))
        return logprob + logdet

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood, the quantity training minimizes."""
        return -self.loglik_picnn(x, y).mean()
