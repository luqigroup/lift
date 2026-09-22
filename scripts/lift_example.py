#!/usr/bin/env python
"""Worked example of the lift (lift.Lift).

Applies the lift to an input-convex neural net and shows what it does in one
forward pass: the body reads a batch, the readout emits every weight of the
target, and the constrained ones come out positive, so the target is convex in
its input at each step. The gradient reaches the body and the slack; the target
itself stays frozen.

Run:  python scripts/lift_example.py      # CPU, no downloads, no training
"""
from __future__ import annotations

import torch
from torch import nn

from lift import ICNN, Lift, tag_positive_


def main() -> None:
    torch.manual_seed(0)

    icnn = ICNN(input_size=2, hidden_dim=64, nlayers=3)
    model = Lift(icnn, cond_size=2)

    x = torch.randn(256, 2)
    weights = model.weights(x)
    energy = model(x)

    print("constrained weights, emitted from the batch:")
    for name in sorted(model.pos_param_names):
        w = weights[name]
        print(f"  {name:20s} shape {tuple(w.shape)}  min {w.min().item():.4f}")
    print(f"energy {tuple(energy.shape)}")

    energy.mean().backward()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters {trainable} (the body and the slack)")
    print(f"target parameters frozen: "
          f"{all(not p.requires_grad for p in icnn.parameters())}")

    # Convexity is the point of the constraint: at one emission the target sits
    # at or below every chord.
    a, b = torch.randn(64, 2), torch.randn(64, 2)
    f = lambda z: torch.func.functional_call(model.template, weights, (z,))
    gap = (0.5 * (f(a) + f(b)) - f(0.5 * (a + b))).min().item()
    print(f"smallest chord minus midpoint {gap:.4f} (non-negative iff convex)")

    # A module that does not mark its own constrained weights.
    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 1)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.lin(z)

    try:
        Lift(Plain(), cond_size=2)
    except ValueError as exc:
        print(f"unmarked module refused: {exc}")

    lifted = Lift(tag_positive_(Plain(), ["lin.weight"]), cond_size=2)
    print(f"after marking: {sorted(lifted.pos_param_names)}")


if __name__ == "__main__":
    main()
