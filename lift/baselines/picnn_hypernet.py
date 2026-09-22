"""Hypernetworks that emit PICNN's convex-path weights.

For the with-and-without-hypernetwork comparison on PCP-Map's ICNN, only the
positive convex-path weights ``PICNN.Lw[k].weight`` are emitted, through a
small DeepSets-style MLP over a conditioning vector; every other PICNN weight
(Lv, Lvw, Lwv, Lxv, Lx and the Lw biases) stays a direct ``nn.Linear`` in both
variants. The emitted flat vector of length
``sum(prod(s) for s in picnn.lw_weight_shapes)`` is split and reshaped back
into per-layer tensors, which ``PICNN.forward`` consumes through its
``pos_weights_override`` argument and passes through the positivity map of its
own ``pos_constraint_mode``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import torch.nn as nn


def _parse_hidden_sizes(s) -> List[int]:
    if isinstance(s, str):
        return [int(t) for t in s.split(",") if t.strip()]
    return list(s)


class PicnnHypernet(nn.Module):
    """DeepSets-style hypernet emitting PICNN convex-path weights.

    Parameters
    ----------
    lw_weight_shapes :
        List of ``torch.Size`` objects (the shapes of the PICNN's
        Lw[k].weight tensors, in order). Use
        ``picnn.lw_weight_shapes``.
    cond_dim :
        Dimensionality of the trainable conditioning embedding.
        Default 8.
    hidden_sizes :
        Sequence of hidden widths for the readout MLP.
    pos_constraint_mode :
        Must match the host PICNN's ``pos_constraint_mode``. The emitted
        override is passed through that positivity map by the PICNN's own
        ``_positive_weight``, so the two settings have to agree; it only
        selects the bias initialization here.
    """

    def __init__(
        self,
        lw_weight_shapes: Sequence[torch.Size],
        *,
        cond_dim: int = 8,
        hidden_sizes="128,128,192",
        pos_constraint_mode: str = "cone",
        readout_fanin_scale: bool = False,
        pool_norm: bool = False,
        pool_scale: float = 0.0,
    ) -> None:
        super().__init__()
        # ``pool_norm`` and ``pool_scale`` act on the readout's input; see
        # ``_readout``. Both are off by default.
        self.pool_norm = bool(pool_norm)
        self.pool_scale = float(pool_scale)
        self._last_readout_input: Optional[torch.Tensor] = None
        self.lw_weight_shapes = [tuple(s) for s in lw_weight_shapes]
        self.cond_dim = int(cond_dim)
        self.pos_constraint_mode = pos_constraint_mode

        self._slot_sizes = [int(torch.tensor(s).prod()) for s in self.lw_weight_shapes]
        self._total_out = int(sum(self._slot_sizes))

        # Trainable K=1 conditioning embedding (shape (cond_dim,)).
        self.cond = nn.Parameter(torch.randn(self.cond_dim) * 0.1)

        # Readout MLP.
        hidden = _parse_hidden_sizes(hidden_sizes)
        layers: List[nn.Module] = []
        in_dim = self.cond_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, self._total_out))
        self.mlp = nn.Sequential(*layers)
        self.readout_fanin_scale = bool(readout_fanin_scale)
        if self.readout_fanin_scale:
            with torch.no_grad():
                nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1.0)

        # Bias initialization targeting a small magnitude in the final
        # positive outputs: zero under relu, the softplus inverse of the
        # target under softplus.
        with torch.no_grad():
            final = self.mlp[-1]
            if pos_constraint_mode == "cone":
                final.bias.fill_(0.0)
            elif pos_constraint_mode == "softplus_reparam":
                # softplus(-2) ~ 0.13
                final.bias.fill_(-2.0)

    def _readout(self, u: torch.Tensor) -> torch.Tensor:
        """The readout MLP.

        With ``readout_fanin_scale`` the last layer is ``(W h)/fan_in + b``
        with ``W ~ N(0, 1)``, the muP output rule, so one Adam step moves the
        emitted weights at the shared step size.

        ``pool_scale`` and ``pool_norm`` act on ``h``, the trunk's output and
        the readout's input. ``pool_scale > 0`` pins ``||h||`` to that value
        and leaves its direction free; ``pool_norm`` standardizes ``h`` to
        zero mean and unit variance over its coordinates, a LayerNorm without
        affine, after which ``(W h)/d_h`` has spread ``1/sqrt(d_h)`` at
        initialization and one Adam step moves it by about ``lr``. Both are
        off by default, leaving ``self.mlp(u)`` unchanged operation for
        operation.
        """
        h = self.mlp[:-1](u) if len(self.mlp) > 1 else u
        if getattr(self, "pool_scale", 0.0) > 0:
            h = self.pool_scale * h / (h.norm() + 1e-8)
        if getattr(self, "pool_norm", False):
            h = (h - h.mean()) / (h.std() + 1e-5)
        # Diagnostics only: detached, never saved, and no device sync here.
        self._last_readout_input = h.detach()
        final = self.mlp[-1]
        if getattr(self, "readout_fanin_scale", False):
            return F.linear(h, final.weight, None) / float(h.shape[-1]) + final.bias
        return final(h)

    @torch.no_grad()
    def num_emitted_params(self) -> int:
        return self._total_out

    def emit(self) -> Dict[int, torch.Tensor]:
        """Return ``{layer_idx -> weight_tensor}`` for the PICNN forward's
        ``pos_weights_override`` argument.

        Shapes match ``lw_weight_shapes`` exactly. Positivity is not applied
        here: ``PICNN._positive_weight`` applies the relu or softplus to the
        override, so the constraint is enforced at a single point.
        """
        flat = self._readout(self.cond)
        out: Dict[int, torch.Tensor] = {}
        offset = 0
        for k, (shape, sz) in enumerate(zip(self.lw_weight_shapes, self._slot_sizes)):
            chunk = flat[offset:offset + sz].view(*shape)
            out[k] = chunk
            offset += sz
        return out


# ----------------------------------------------------------------------
# y-conditioned (and theta-batch-conditioned) variant
# ----------------------------------------------------------------------


class PicnnHypernetYConditional(nn.Module):
    """DeepSets-style hypernetwork emitting PICNN convex-path weights
    conditioned on a permutation-invariant summary of the current
    minibatch's ``(theta, y)`` pairs.

    A per-sample MLP ``phi_theta``, respectively ``phi_y``, maps each row of
    the minibatch's theta, respectively y, into an embedding space; the rows
    are mean-aggregated into one set summary; and a readout MLP ``rho_theta``,
    respectively ``rho_y``, maps that summary to a fixed-width conditioning
    vector. The trainable ``cond`` vector of :class:`PicnnHypernet` is
    concatenated as a y-independent baseline channel, and the combined
    ``u = [cond; u_theta; u_y]`` feeds the readout MLP that emits the flat
    positive-weight vector.

    Stateful API
    ------------
    ``prepare(theta_batch, y_batch)`` caches the emitted weight dict for the
    current minibatch and ``emit()`` returns that cache. A per-sample Hessian
    through ``torch.func.vmap`` re-enters the supplier once per sample, so the
    cache both makes that O(1) and gives every sample the same emitted
    weights, that is one convex body for the whole batch. The training loop
    must call ``prepare`` before each ``loss(...)`` call.

    Parameters
    ----------
    lw_weight_shapes : list of ``torch.Size``
    input_x_dim : int -- dim of theta (per row).
    input_y_dim : int -- dim of y (per row).
    cond_dim : int -- width of the trainable y-independent baseline
        embedding (matches ``PicnnHypernet.cond_dim``).
    summary_dim : int -- width of u_theta and u_y after rho.
    phi_hidden : int -- hidden width of phi_theta / phi_y MLPs.
    hidden_sizes : sequence of int -- readout MLP widths.
    pos_constraint_mode : str -- must match host PICNN.
    """

    def __init__(
        self,
        lw_weight_shapes: Sequence[torch.Size],
        *,
        input_x_dim: int,
        input_y_dim: int,
        cond_dim: int = 8,
        summary_dim: int = 16,
        phi_hidden: int = 64,
        phi_out: int = 32,
        rho_hidden: int = 16,
        hidden_sizes="128,128,192",
        pos_constraint_mode: str = "cone",
        readout_fanin_scale: bool = False,
        pool_norm: bool = False,
        pool_scale: float = 0.0,
    ) -> None:
        super().__init__()
        # ``pool_norm`` and ``pool_scale`` act on the readout's input; see
        # ``_readout``. Both are off by default.
        self.pool_norm = bool(pool_norm)
        self.pool_scale = float(pool_scale)
        self._last_readout_input: Optional[torch.Tensor] = None
        self.lw_weight_shapes = [tuple(s) for s in lw_weight_shapes]
        self.cond_dim = int(cond_dim)
        self.summary_dim = int(summary_dim)
        self.input_x_dim = int(input_x_dim)
        self.input_y_dim = int(input_y_dim)
        self.pos_constraint_mode = pos_constraint_mode

        self._slot_sizes = [int(torch.tensor(s).prod()) for s in self.lw_weight_shapes]
        self._total_out = int(sum(self._slot_sizes))

        # Trainable y-independent baseline embedding, matching
        # PicnnHypernet's ``cond`` channel.
        self.cond = nn.Parameter(torch.randn(self.cond_dim) * 0.1)

        # DeepSets phi/rho for theta.
        self.phi_theta = nn.Sequential(
            nn.Linear(self.input_x_dim, phi_hidden),
            nn.SiLU(),
            nn.Linear(phi_hidden, phi_out),
        )
        self.rho_theta = nn.Sequential(
            nn.Linear(phi_out, rho_hidden),
            nn.SiLU(),
            nn.Linear(rho_hidden, self.summary_dim),
        )
        # DeepSets phi/rho for y.
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_y_dim, phi_hidden),
            nn.SiLU(),
            nn.Linear(phi_hidden, phi_out),
        )
        self.rho_y = nn.Sequential(
            nn.Linear(phi_out, rho_hidden),
            nn.SiLU(),
            nn.Linear(rho_hidden, self.summary_dim),
        )

        # Readout MLP from concatenated [cond; u_theta; u_y] to a flat
        # positive-weight vector.
        hidden = _parse_hidden_sizes(hidden_sizes)
        in_dim = self.cond_dim + 2 * self.summary_dim
        layers: List[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, self._total_out))
        self.mlp = nn.Sequential(*layers)
        self.readout_fanin_scale = bool(readout_fanin_scale)
        if self.readout_fanin_scale:
            with torch.no_grad():
                nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1.0)

        with torch.no_grad():
            final = self.mlp[-1]
            if pos_constraint_mode == "cone":
                final.bias.fill_(0.0)
            elif pos_constraint_mode == "softplus_reparam":
                final.bias.fill_(-2.0)

        # Cache filled by ``prepare(theta_batch, y_batch)``.
        self._cached_weights: Dict[int, torch.Tensor] = {}

    def _readout(self, u: torch.Tensor) -> torch.Tensor:
        """The readout MLP.

        With ``readout_fanin_scale`` the last layer is ``(W h)/fan_in + b``
        with ``W ~ N(0, 1)``, the muP output rule, so one Adam step moves the
        emitted weights at the shared step size.

        ``pool_scale`` and ``pool_norm`` act on ``h``, the trunk's output and
        the readout's input. ``pool_scale > 0`` pins ``||h||`` to that value
        and leaves its direction free; ``pool_norm`` standardizes ``h`` to
        zero mean and unit variance over its coordinates, a LayerNorm without
        affine, after which ``(W h)/d_h`` has spread ``1/sqrt(d_h)`` at
        initialization and one Adam step moves it by about ``lr``. Both are
        off by default, leaving ``self.mlp(u)`` unchanged operation for
        operation.
        """
        h = self.mlp[:-1](u) if len(self.mlp) > 1 else u
        if getattr(self, "pool_scale", 0.0) > 0:
            h = self.pool_scale * h / (h.norm() + 1e-8)
        if getattr(self, "pool_norm", False):
            h = (h - h.mean()) / (h.std() + 1e-5)
        # Diagnostics only: detached, never saved, and no device sync here.
        self._last_readout_input = h.detach()
        final = self.mlp[-1]
        if getattr(self, "readout_fanin_scale", False):
            return F.linear(h, final.weight, None) / float(h.shape[-1]) + final.bias
        return final(h)

    @torch.no_grad()
    def num_emitted_params(self) -> int:
        return self._total_out

    def _summarize(self, theta_batch: torch.Tensor,
                   y_batch: torch.Tensor,
                   chunk: Optional[int] = None) -> torch.Tensor:
        # phi -> mean over the batch -> rho. The mean, rather than a sum,
        # keeps the summary scale roughly batch-size invariant, which matters
        # because PCP-Map's last minibatch can be smaller.
        if chunk is None:
            u_theta = self.rho_theta(self.phi_theta(theta_batch).mean(dim=0))
            u_y = self.rho_y(self.phi_y(y_batch).mean(dim=0))
            return torch.cat([self.cond, u_theta, u_y], dim=0)
        # Chunked path, opt-in, with ``None`` leaving the computation
        # unchanged. The pool is a mean over rows, so summing per-chunk
        # embeddings and dividing once at the end gives the same number,
        # while the conditioning set never has to be resident all at once.
        # Chunks are moved to the parameters' device here, so the caller may
        # hold the set in host memory.
        dev = self.cond.device
        n = int(theta_batch.shape[0])
        acc_t = acc_y = None
        for i in range(0, n, int(chunk)):
            tb = theta_batch[i:i + chunk].to(dev, non_blocking=True)
            yb = y_batch[i:i + chunk].to(dev, non_blocking=True)
            st = self.phi_theta(tb).sum(dim=0)
            sy = self.phi_y(yb).sum(dim=0)
            acc_t = st if acc_t is None else acc_t + st
            acc_y = sy if acc_y is None else acc_y + sy
        u_theta = self.rho_theta(acc_t / n)
        u_y = self.rho_y(acc_y / n)
        return torch.cat([self.cond, u_theta, u_y], dim=0)

    def prepare(self, theta_batch: torch.Tensor,
                y_batch: torch.Tensor,
                chunk: Optional[int] = None) -> None:
        """Compute and cache the emitted weight dict for this batch.

        Must be called before ``PCPMap.loss(...)`` so that any
        subsequent ``emit()`` (including inside ``torch.func.vmap``)
        returns the same cached tensors, preserving gradient flow into
        the hypernet's parameters.

        ``chunk`` walks the conditioning set in slices instead of embedding
        it in one pass; see ``_summarize``. It changes no number and defaults
        to off.
        """
        u = self._summarize(theta_batch, y_batch, chunk)
        flat = self._readout(u)
        out: Dict[int, torch.Tensor] = {}
        offset = 0
        for k, (shape, sz) in enumerate(
            zip(self.lw_weight_shapes, self._slot_sizes)
        ):
            chunk = flat[offset:offset + sz].view(*shape)
            out[k] = chunk
            offset += sz
        self._cached_weights = out

    def emit(self) -> Dict[int, torch.Tensor]:
        if not self._cached_weights:
            raise RuntimeError(
                "PicnnHypernetYConditional.emit() called before prepare(). "
                "The training/eval loop must call prepare(theta, y) before "
                "PCPMap.loss(x, y) / .loglik_picnn(x, y)."
            )
        return self._cached_weights
