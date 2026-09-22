"""HINT normalizing flow conditioned on a per-component index ``k``.

One set of trainable flow parameters takes the component index
``k \\in \\{0, \\ldots, K-1\\}`` as an additional input: the per-k learned
embedding is concatenated to ``z_upper`` inside every coupling MLP, so the
component specialization is threaded through the whole flow stack.

Forward signatures:

  * ``flow(eps, k_idx, mode='sample')`` -- ``eps`` of shape
    ``(K, M, D_aug)``, ``k_idx`` of shape ``(K,)`` of ints
    ``[0, ..., K-1]``. Returns ``x_aug`` of shape ``(K, M, D_aug)``.
  * ``flow(eps, k_idx, mode='sample_with_logprob')`` -- same input,
    returns ``(x_aug, log_q_phi)`` with ``log_q_phi`` of shape ``(K, M)``.
  * ``flow.log_prob(x, k_idx)`` -- ``x`` of shape ``(K, M, D_aug)``,
    ``k_idx`` of shape ``(K,)``; returns ``(K, M)`` log-densities.

The architecture matches :class:`~lift.models.sampler_flow.SamplerFlow`
except that every coupling MLP takes ``Linear(upper_dim + cond_dim,
n_hidden)`` as the first layer, and the auxiliary-block log-density (when
``D_aug > D``) follows the augmentation lemma's ``q(x_aux | x_user) =
N(0, I)`` constraint at init through the same coupling-final-layer
``std=0.01`` init.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class _ConditionalNFTree(nn.Module):
    """HINT binary-tree coupling, conditioned on a vector ``cond``.

    Symmetric to :class:`~lift.models.sampler_flow._UnconditionalNFTree`
    but each coupling MLP's input is ``cat([z_upper, cond], -1)``.
    """

    def __init__(
        self, n_in: int, n_hidden: int, depth: int,
        n_mlp_layers: int = 3, cond_dim: int = 8,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.n_in = int(n_in)
        self.depth = int(depth)
        self.cond_dim = int(cond_dim)
        if depth <= 0 or n_in <= 1:
            self.is_leaf = True
            return
        self.is_leaf = False
        self.split_idx = n_in // 2
        upper_dim = self.split_idx
        lower_dim = n_in - self.split_idx
        layers: list[nn.Module] = [
            nn.Linear(upper_dim + self.cond_dim, n_hidden), activation,
        ]
        for _ in range(n_mlp_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation)
        final = nn.Linear(n_hidden, 2 * lower_dim)
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.coupling_mlp = nn.Sequential(*layers)
        self.upper_tree = _ConditionalNFTree(
            upper_dim, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers, cond_dim=cond_dim,
            activation=activation,
        )
        self.lower_tree = _ConditionalNFTree(
            lower_dim, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers, cond_dim=cond_dim,
            activation=activation,
        )

    def _coupling(
        self, z_upper: torch.Tensor, cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_upper: (B, upper_dim).  cond: (B, cond_dim).  B = K * M flat.
        inp = torch.cat([z_upper, cond], dim=-1)
        out = self.coupling_mlp(inp)
        lower_dim = self.n_in - self.split_idx
        log_S = torch.clamp(out[:, :lower_dim], -5.0, 5.0)
        T = out[:, lower_dim:]
        return torch.exp(log_S), T, log_S

    def decode_with_logdet(
        self, z: torch.Tensor, cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode ``z -> x`` accumulating ``log|det dx/dz|`` per sample."""
        if self.is_leaf:
            return z, torch.zeros(
                z.shape[0], device=z.device, dtype=z.dtype,
            )
        z_upper = z[:, :self.split_idx]
        z_lower = z[:, self.split_idx:]
        x_upper, ld_upper = self.upper_tree.decode_with_logdet(z_upper, cond)
        S, T, log_S = self._coupling(z_upper, cond)
        z_lower_inner, ld_lower = self.lower_tree.decode_with_logdet(z_lower, cond)
        x_lower = (z_lower_inner - T) / S
        log_det = ld_upper + ld_lower - log_S.sum(dim=1)
        return torch.cat([x_upper, x_lower], dim=1), log_det

    def encode_with_logdet(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`decode_with_logdet`."""
        if self.is_leaf:
            return x, torch.zeros(
                x.shape[0], device=x.device, dtype=x.dtype,
            )
        x_upper = x[:, :self.split_idx]
        x_lower = x[:, self.split_idx:]
        z_upper, ld_upper = self.upper_tree.encode_with_logdet(x_upper, cond)
        S, T, log_S = self._coupling(z_upper, cond)
        z_lower_inner = S * x_lower + T
        z_lower, ld_lower = self.lower_tree.encode_with_logdet(z_lower_inner, cond)
        log_det = ld_upper + ld_lower + log_S.sum(dim=1)
        return torch.cat([z_upper, z_lower], dim=1), log_det


class ConditionalSamplerFlow(nn.Module):
    """Single set of flow parameters, conditioned on component index ``k``.

    ``forward(eps, k_idx, mode)``: ``eps`` shape ``(K, M, D_aug)``,
    ``k_idx`` shape ``(K,)`` of ints in ``[0, K)``. The component embedding
    ``cond_embed(k_idx)`` of shape ``(K, cond_dim)`` is broadcast over ``M``
    and concatenated to ``z_upper`` inside every coupling MLP.
    """

    def __init__(
        self,
        K: int,
        D: int,
        D_aug: Optional[int] = None,
        n_hidden: int = 64,
        n_flow_layers: int = 4,
        depth: Optional[int] = None,
        n_mlp_layers: int = 3,
        cond_dim: int = 8,
    ):
        super().__init__()
        self.K = int(K)
        self.D = int(D)
        self.D_aug = int(D_aug if D_aug is not None else max(2, self.D))
        if self.D_aug < 2:
            raise ValueError(
                f"D_aug must be >= 2 (HINT requires it); got {self.D_aug}",
            )
        if self.D_aug < self.D:
            raise ValueError(
                f"D_aug ({self.D_aug}) must be >= D ({self.D})",
            )
        if depth is None:
            depth = max(1, int(math.ceil(math.log2(self.D_aug))))
        self.depth = int(depth)
        self.n_flow_layers = int(n_flow_layers)
        self.cond_dim = int(cond_dim)
        self.cond_embed = nn.Embedding(self.K, self.cond_dim)
        # Small init so the conditioning signal does not dominate the
        # coupling MLP at the start of training.
        nn.init.normal_(self.cond_embed.weight, std=0.1)

        self.trees = nn.ModuleList([
            _ConditionalNFTree(
                self.D_aug, n_hidden, self.depth,
                n_mlp_layers=n_mlp_layers, cond_dim=self.cond_dim,
            )
            for _ in range(self.n_flow_layers)
        ])
        perms = []
        inv_perms = []
        for _ in range(self.n_flow_layers):
            p = torch.randperm(self.D_aug)
            perms.append(p)
            inv_perms.append(torch.argsort(p))
        self.register_buffer("perms", torch.stack(perms))
        self.register_buffer("inv_perms", torch.stack(inv_perms))

    # ----------------------------------------------------------------- helpers

    def _expand_cond(self, k_idx: torch.Tensor, M: int) -> torch.Tensor:
        """``(K,)`` -> ``(K * M, cond_dim)`` via embed + broadcast.

        Flat-batched output for the trees, which run on a flat
        ``(K * M, D_aug)`` particle pool.
        """
        cond_K = self.cond_embed(k_idx)                         # (K, cond_dim)
        cond_KM = cond_K.unsqueeze(1).expand(-1, M, -1)         # (K, M, cond_dim)
        return cond_KM.reshape(-1, self.cond_dim).contiguous()

    def _decode_with_logdet(
        self, eps: torch.Tensor, k_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K_in, M, D_aug = eps.shape
        if K_in != self.K:
            raise ValueError(
                f"eps has K={K_in}; flow was built for K={self.K}",
            )
        if D_aug != self.D_aug:
            raise ValueError(
                f"eps has D_aug={D_aug}; flow has D_aug={self.D_aug}",
            )
        cond_flat = self._expand_cond(k_idx, M)                 # (K*M, cond_dim)
        z = eps.reshape(-1, self.D_aug)                         # (K*M, D_aug)
        log_det = torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype,
        )
        for k_layer in reversed(range(self.n_flow_layers)):
            z, ld = self.trees[k_layer].decode_with_logdet(z, cond_flat)
            log_det = log_det + ld
            z = z[:, self.inv_perms[k_layer]]
        x_aug = z.reshape(K_in, M, self.D_aug)
        log_det = log_det.reshape(K_in, M)
        return x_aug, log_det

    def _encode_with_logdet(
        self, x: torch.Tensor, k_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K_in, M, D_aug = x.shape
        cond_flat = self._expand_cond(k_idx, M)
        z = x.reshape(-1, self.D_aug)
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for k_layer in range(self.n_flow_layers):
            z = z[:, self.perms[k_layer]]
            z, ld = self.trees[k_layer].encode_with_logdet(z, cond_flat)
            log_det = log_det + ld
        eps = z.reshape(K_in, M, self.D_aug)
        log_det = log_det.reshape(K_in, M)
        return eps, log_det

    # ----------------------------------------------------------------- public

    def log_prob(
        self, x: torch.Tensor, k_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Exact ``log q_phi(x_aug | k)`` via encode + change-of-variables.

        ``x``: ``(K, M, D_aug)``. ``k_idx``: ``(K,)``. Returns ``(K, M)``.
        """
        eps, log_det_enc = self._encode_with_logdet(x, k_idx)
        log_pz = (
            -0.5 * eps.pow(2).sum(dim=-1)
            - 0.5 * self.D_aug * math.log(2.0 * math.pi)
        )
        return log_pz + log_det_enc

    def forward(
        self, eps: torch.Tensor, k_idx: torch.Tensor,
        mode: str = "sample",
    ):
        if mode == "sample":
            x, _ = self._decode_with_logdet(eps, k_idx)
            return x
        if mode == "sample_with_logprob":
            x, log_det = self._decode_with_logdet(eps, k_idx)
            log_pz = (
                -0.5 * eps.pow(2).sum(dim=-1)
                - 0.5 * self.D_aug * math.log(2.0 * math.pi)
            )
            log_q = log_pz - log_det
            return x, log_q
        if mode == "log_prob":
            return self.log_prob(eps, k_idx)
        raise ValueError(f"unknown mode: {mode!r}")
