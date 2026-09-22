"""Plain convolutional autoencoder for the MNIST latent-density experiments.

Non-variational by design: a VAE's KL-to-prior term pulls the latent
toward a standard Gaussian, which would favor the Gaussian baseline in
the latent-density comparison. Encoder ``(B, 1, 28, 28) -> (B, latent_dim)``
through two strided convolutions; decoder mirrors it and ends in a
sigmoid for reconstruction against images in [0, 1].
"""

from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """Encoder: ``(B, 1, 28, 28) -> (B, latent_dim)``."""

    def __init__(self, latent_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(hidden * 7 * 7, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Decoder: ``(B, latent_dim) -> (B, 1, 28, 28)`` in [0, 1]."""

    def __init__(self, latent_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.fc = nn.Linear(latent_dim, hidden * 7 * 7)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, hidden // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden // 2, 1,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.hidden, 7, 7)
        return self.net(h)


class ConvAutoencoder(nn.Module):
    """Plain Conv-AE composed of :class:`ConvEncoder` + :class:`ConvDecoder`."""

    def __init__(self, latent_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, hidden)
        self.decoder = ConvDecoder(latent_dim, hidden)
        self.latent_dim = int(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return torch.nn.functional.binary_cross_entropy(x_hat, x,
                                                       reduction="mean")
