"""Per-component HINT normalizing-flow sampler with tractable density.

The sampler works on an augmented latent space of dimension
``D_aug >= max(D, 2)``, since HINT is trivial below two dimensions and
``D_aug > D`` lets the 1-D experiments use HINT while keeping the density
tractable. ``forward(eps)`` maps ``eps ~ N(0, I_{D_aug})`` to a sample
``x_aug``: its first ``D`` coordinates are the marginal sample of ``x``, and the
remaining ``D_aug - D`` are auxiliary coordinates that the trainer pins to
``N(0, I)`` through the augmented target

    E_aug(x_aug) = E(x_aug[:D]) + ||x_aug[D:]||^2 / 2
                   + (D_aug - D) log(2 pi) / 2.

The auxiliary Gaussian factor integrates to one, so ``Z_aug = Z``, and the
flow's exact density on the augmented space gives ``log Z_k`` by importance
sampling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class _UnconditionalNFTree(nn.Module):
    """HINT binary-tree coupling, unconditional. Requires ``n_in >= 2``."""

    def __init__(
        self, n_in: int, n_hidden: int, depth: int,
        n_mlp_layers: int = 3, activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.n_in = n_in
        self.depth = depth
        if depth <= 0 or n_in <= 1:
            self.is_leaf = True
            return
        self.is_leaf = False
        self.split_idx = n_in // 2
        upper_dim = self.split_idx
        lower_dim = n_in - self.split_idx
        layers: list[nn.Module] = [nn.Linear(upper_dim, n_hidden), activation]
        for _ in range(n_mlp_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation)
        final = nn.Linear(n_hidden, 2 * lower_dim)
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.coupling_mlp = nn.Sequential(*layers)
        self.upper_tree = _UnconditionalNFTree(
            upper_dim, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers, activation=activation,
        )
        self.lower_tree = _UnconditionalNFTree(
            lower_dim, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers, activation=activation,
        )

    def _coupling(self, z_upper: torch.Tensor):
        out = self.coupling_mlp(z_upper)
        lower_dim = self.n_in - self.split_idx
        log_S = torch.clamp(out[:, :lower_dim], -5.0, 5.0)
        T = out[:, lower_dim:]
        return torch.exp(log_S), T, log_S

    def decode_with_logdet(
        self, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode ``z -> x`` and accumulate ``log|det dx/dz|``.

        For the affine coupling :math:`x_{\\ell} = (z_{\\ell} - T) / S`
        with scaling vector ``S = exp(log_S)``, the Jacobian's log-det
        contribution is ``-sum log_S`` per sample.
        """
        if self.is_leaf:
            return z, torch.zeros(
                z.shape[0], device=z.device, dtype=z.dtype,
            )
        z_upper = z[:, :self.split_idx]
        z_lower = z[:, self.split_idx:]
        x_upper, ld_upper = self.upper_tree.decode_with_logdet(z_upper)
        S, T, log_S = self._coupling(z_upper)
        z_lower_inner, ld_lower = self.lower_tree.decode_with_logdet(z_lower)
        x_lower = (z_lower_inner - T) / S
        log_det = ld_upper + ld_lower - log_S.sum(dim=1)
        return torch.cat([x_upper, x_lower], dim=1), log_det

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.decode_with_logdet(z)
        return x

    def encode_with_logdet(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`decode_with_logdet`: return ``(z, log|det dz/dx|)``.

        Per-coupling: ``z_lower = S(z_upper) * x_lower + T(z_upper)`` where
        ``z_upper = upper_tree.encode(x_upper)``. The leaf is identity.
        Per-coupling log-det of the encode direction is :math:`+\\sum
        \\log S = +\\sum \\log_S`.
        """
        if self.is_leaf:
            return x, torch.zeros(
                x.shape[0], device=x.device, dtype=x.dtype,
            )
        x_upper = x[:, :self.split_idx]
        x_lower = x[:, self.split_idx:]
        z_upper, ld_upper = self.upper_tree.encode_with_logdet(x_upper)
        S, T, log_S = self._coupling(z_upper)
        z_lower_inner = S * x_lower + T
        z_lower, ld_lower = self.lower_tree.encode_with_logdet(z_lower_inner)
        log_det = ld_upper + ld_lower + log_S.sum(dim=1)
        return torch.cat([z_upper, z_lower], dim=1), log_det


class SamplerFlow(nn.Module):
    """Per-component HINT sampler on an augmented latent space."""

    def __init__(
        self,
        D: int,
        D_aug: Optional[int] = None,
        n_hidden: int = 64,
        n_flow_layers: int = 4,
        depth: Optional[int] = None,
        n_mlp_layers: int = 3,
    ):
        super().__init__()
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
        self.trees = nn.ModuleList([
            _UnconditionalNFTree(
                self.D_aug, n_hidden, self.depth,
                n_mlp_layers=n_mlp_layers,
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

    def _decode_with_logdet(
        self, eps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the HINT decode stack and accumulate the per-sample log-det.

        ``eps`` shape ``(M, D_aug)``; returns ``(x_aug, log_det)`` where
        ``log_det`` has shape ``(M,)`` and is :math:`\\log|\\det\\,dx/d\\epsilon|`.
        """
        z = eps
        log_det = torch.zeros(eps.shape[0], device=eps.device, dtype=eps.dtype)
        for k in reversed(range(self.n_flow_layers)):
            z, ld = self.trees[k].decode_with_logdet(z)
            log_det = log_det + ld
            z = z[:, self.inv_perms[k]]
        return z, log_det

    def _encode_with_logdet(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`_decode_with_logdet`: ``x_aug -> eps``.

        Returns ``(eps, log_det)`` with ``log_det = log|det d eps / d x_aug|``.
        """
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for k in range(self.n_flow_layers):
            z = z[:, self.perms[k]]
            z, ld = self.trees[k].encode_with_logdet(z)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact ``log q_phi(x_aug)`` by encode plus change of variables.

        ``x`` has shape ``(M, D_aug)``; returns the ``(M,)`` per-sample
        log-density with respect to Lebesgue measure.
        """
        eps, log_det_enc = self._encode_with_logdet(x)
        log_pz = (
            -0.5 * eps.pow(2).sum(dim=1)
            - 0.5 * self.D_aug * math.log(2.0 * math.pi)
        )
        return log_pz + log_det_enc

    def forward(
        self, eps: torch.Tensor, mode: str = "sample",
    ):
        """Polymorphic forward dispatchable via ``functional_call``.

        Args:
            eps: ``(M, D_aug)`` standard-normal latents.
            mode:
              * ``"sample"``: returns ``x_aug`` of shape ``(M, D_aug)``.
              * ``"sample_with_logprob"``: returns ``(x_aug, log_q_phi)``
                where ``log_q_phi`` has shape ``(M,)``.
        """
        if mode == "sample":
            x, _ = self._decode_with_logdet(eps)
            return x
        if mode == "sample_with_logprob":
            x, log_det = self._decode_with_logdet(eps)
            log_pz = (
                -0.5 * eps.pow(2).sum(dim=1)
                - 0.5 * self.D_aug * math.log(2.0 * math.pi)
            )
            log_q = log_pz - log_det
            return x, log_q
        if mode == "log_prob":
            # In this mode ``eps`` is the point at which to evaluate the
            # density, not a latent.
            return self.log_prob(eps)
        raise ValueError(f"unknown mode: {mode!r}")
