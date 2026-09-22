"""Partially Input Convex Neural Network (PICNN).

Re-implementation of the PICNN used by PCP-Map (Wang, Baptista, Marzouk,
Ruthotto, Verma, SISC 2024, arXiv 2310.16975), itself a re-implementation
of Amos and Kolter (ICML 2017), following ``src/icnn.py::PICNN`` from
https://github.com/EmoryMLIP/PCP-Map.

Only the convex path ``Lw`` carries positive weights. Under
``pos_constraint_mode="cone"``, PCP-Map's published default, they are
stored raw and clamped by ``project_positive`` after each optimizer step;
under ``"softplus_reparam"`` they are stored raw and passed through
softplus inside the forward. The y-context path ``Lv`` and the skip paths
``Lvw``, ``Lwv``, ``Lxv`` and ``Lx`` are unconstrained.

``forward`` accepts ``pos_weights_override``, a dict from convex-path
layer index to an externally supplied weight tensor that replaces
``self.Lw[k].weight`` for that call. That is where the lift emits the
convex weights and the PICNN consumes them.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_POS_MODES = ("cone", "softplus_reparam")


class PICNN(nn.Module):
    """Partially Input Convex Neural Network: convex in x, free in y.

    Forward signature::

        F(x, y, pos_weights_override=None) -> shape (B, out_dim)

    Convex weight count is ``len(Lw)`` tensors with shapes
    ``[ (feat, in_x), (feat, feat), ..., (out_dim, feat) ]``.
    """

    def __init__(
        self,
        input_x_dim: int,
        input_y_dim: int,
        feature_dim: int,
        feature_y_dim: int,
        out_dim: int,
        num_layers: int,
        *,
        act: nn.Module = nn.Softplus(),
        act_v: nn.Module = nn.ELU(),
        pos_constraint_mode: str = "cone",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}.")
        if pos_constraint_mode not in _POS_MODES:
            raise ValueError(
                f"pos_constraint_mode must be one of {_POS_MODES}, "
                f"got {pos_constraint_mode!r}.",
            )

        self.input_x_dim = input_x_dim
        self.input_y_dim = input_y_dim
        self.feature_dim = feature_dim
        self.feature_y_dim = feature_y_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.pos_constraint_mode = pos_constraint_mode

        # ---- y-path: Lv (unconstrained linear, ELU between)
        Lv = [nn.Linear(input_y_dim, feature_y_dim, bias=True)]
        for _ in range(num_layers - 1):
            Lv.append(nn.Linear(feature_y_dim, feature_y_dim, bias=True))
        self.Lv = nn.ModuleList(Lv)

        # ---- y-into-w bias path: Lvw (unconstrained, no bias)
        Lvw = [nn.Linear(input_y_dim, feature_dim, bias=False)]
        for _ in range(num_layers - 1):
            Lvw.append(nn.Linear(feature_y_dim, feature_dim, bias=False))
        Lvw.append(nn.Linear(feature_y_dim, out_dim, bias=False))
        self.Lvw = nn.ModuleList(Lvw)

        # ---- convex path: Lw (POSITIVE linear weights)
        Lw: List[nn.Linear] = []
        Lw0 = nn.Linear(input_x_dim, feature_dim, bias=True)
        with torch.no_grad():
            Lw0.weight.data = F.relu(Lw0.weight)
        Lw.append(Lw0)
        for _ in range(num_layers - 1):
            Lwk = nn.Linear(feature_dim, feature_dim, bias=True)
            with torch.no_grad():
                Lwk.weight.data = F.relu(Lwk.weight)
            Lw.append(Lwk)
        LwK = nn.Linear(feature_dim, out_dim, bias=True)
        with torch.no_grad():
            LwK.weight.data = F.relu(LwK.weight)
        Lw.append(LwK)
        self.Lw = nn.ModuleList(Lw)

        # ---- y-modulates-w-prod: Lwv (positivity via in-forward relu)
        Lwv = [nn.Linear(input_y_dim, input_x_dim, bias=True)]
        for _ in range(num_layers):
            Lwv.append(nn.Linear(feature_y_dim, feature_dim, bias=True))
        self.Lwv = nn.ModuleList(Lwv)

        # ---- y-modulates-x-skip: Lxv (unconstrained)
        Lxv = [nn.Linear(feature_y_dim, input_x_dim, bias=True)
               for _ in range(num_layers)]
        self.Lxv = nn.ModuleList(Lxv)

        # ---- x-skip into the convex path: Lx (unconstrained, no bias)
        Lx = [nn.Linear(input_x_dim, feature_dim, bias=False)
              for _ in range(num_layers - 1)]
        Lx.append(nn.Linear(input_x_dim, out_dim, bias=False))
        self.Lx = nn.ModuleList(Lx)

        self.act = act
        self.act_v = act_v

    # ------------------------------------------------------------------
    # Public API for the hypernet variant
    # ------------------------------------------------------------------

    @property
    def lw_weight_shapes(self) -> List[torch.Size]:
        """Shapes of the convex-path weight tensors (matches Lw[k].weight)."""
        return [lw.weight.shape for lw in self.Lw]

    def project_positive(self) -> None:
        """Clamp the convex-path weights into the non-negative cone.

        Called after each optimizer step, and a no-op outside ``cone`` mode.
        """
        if self.pos_constraint_mode != "cone":
            return
        with torch.no_grad():
            for lw in self.Lw:
                lw.weight.data = F.relu(lw.weight.data)

    def _positive_weight(self, k: int,
                         override: Optional[Dict[int, torch.Tensor]]
                         ) -> torch.Tensor:
        """Resolve the positive weight tensor for convex-path layer k.

        An override goes through the positivity reparameterization again,
        so an emitted weight satisfies the constraint the same way a stored
        one does. Otherwise the stored ``Lw[k].weight`` is used: it is
        already non-negative in ``cone`` mode, courtesy of
        ``project_positive``, and passes through softplus here in
        ``softplus_reparam`` mode.
        """
        if override is not None and k in override:
            w = override[k]
            if self.pos_constraint_mode == "cone":
                return F.relu(w)
            return F.softplus(w)
        w = self.Lw[k].weight
        if self.pos_constraint_mode == "softplus_reparam":
            return F.softplus(w)
        return w

    # ------------------------------------------------------------------
    # Forward pass — matches PCP-Map's `src/icnn.py::PICNN.forward`.
    # ------------------------------------------------------------------

    def forward(
        self,
        in_x: torch.Tensor,
        in_y: torch.Tensor,
        *,
        pos_weights_override: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # ---- first layer
        v = in_y
        w0_prod = in_x * F.relu(self.Lwv[0](v))
        Lw0_pos = self._positive_weight(0, pos_weights_override)
        w = self.act(
            F.linear(w0_prod, Lw0_pos, self.Lw[0].bias) + self.Lvw[0](v)
        )

        # ---- intermediate layers
        for k in range(1, len(self.Lw) - 1):
            down = 1.0 / w.shape[-1]
            v = self.act_v(self.Lv[k - 1](v))
            wk_prod = w * F.relu(self.Lwv[k](v))
            xk_prod = in_x * self.Lxv[k - 1](v)
            Lwk_pos = self._positive_weight(k, pos_weights_override)
            w = self.act(
                F.linear(wk_prod, Lwk_pos, self.Lw[k].bias) * down
                + self.Lx[k - 1](xk_prod)
                + self.Lvw[k](v)
            )

        # ---- last layer
        down = 1.0 / w.shape[-1]
        vK = self.act_v(self.Lv[-1](v))
        wK_prod = w * F.relu(self.Lwv[-1](vK))
        xK_prod = in_x * self.Lxv[-1](vK)
        K = len(self.Lw) - 1
        LwK_pos = self._positive_weight(K, pos_weights_override)
        w = (
            F.linear(wK_prod, LwK_pos, self.Lw[K].bias) * down
            + self.Lx[-1](xK_prod)
            + self.Lvw[-1](vK)
        )
        return w
