"""Container of K per-component normalizing-flow samplers."""

from __future__ import annotations

from typing import Optional

from torch import nn

from lift.models.sampler_flow import SamplerFlow


class MultiSamplerFlow(nn.Module):

    def __init__(
        self,
        K: int,
        D: int,
        D_aug: Optional[int] = None,
        n_hidden: int = 64,
        n_flow_layers: int = 4,
        depth: Optional[int] = None,
        n_mlp_layers: int = 3,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.D = int(D)
        self.samplers = nn.ModuleList([
            SamplerFlow(
                D=D, D_aug=D_aug, n_hidden=n_hidden,
                n_flow_layers=n_flow_layers, depth=depth,
                n_mlp_layers=n_mlp_layers,
            )
            for _ in range(K)
        ])
