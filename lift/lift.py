"""The lift, as a one-line wrapper around any module with constrained weights.

Equation (1) of the paper: a constrained weight is emitted as
``psi(b + h_phi(X))`` from a permutation-invariant summary of the training
batch ``X``, rather than stored as a free latent weight. :class:`Lift` joins a
frozen template module to a :class:`~lift.models.HyperNetwork` that emits its
weights, so the trainable parameters are exactly ``phi`` and ``b``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn
from torch.func import functional_call

from lift.models.hypernet import HyperNetwork, infer_pos_param_names

DEFAULT_BODY = (64, 64, 96)


def tag_positive_(module: nn.Module, names: Iterable[str]) -> nn.Module:
    """Mark parameters of ``module`` as the ones positivity applies to.

    :class:`Lift` finds constrained weights by looking for this mark, which the
    ICNNs in this package set themselves. Use this to lift a module that does
    not.

    Args:
        module: the module to mark, in place.
        names: parameter names as ``module.named_parameters()`` reports them.

    Returns:
        ``module``, so the call can be inlined.

    Raises:
        ValueError: a name is not a parameter of ``module``.
    """
    params = dict(module.named_parameters())
    for name in names:
        if name not in params:
            raise ValueError(
                f"{name!r} is not a parameter of {type(module).__name__}")
        params[name]._pos_required = True
    return module


class Lift(nn.Module):
    """A frozen module whose weights are emitted from the batch.

    Args:
        module: the target, typically an ICNN. It is frozen: its own parameters
            never receive a gradient, and its constrained weights are replaced
            at every forward by emitted ones.
        cond_size: width of the conditioning samples the body reads.
        hidden_sizes: widths of the body.
        pos_param_names: which parameters to apply positivity to. ``None``
            detects them from the marks :func:`tag_positive_` sets. Pass an
            explicit empty set for a target that handles positivity itself.
        **hypernet_kwargs: forwarded to :class:`~lift.models.HyperNetwork`.

    Raises:
        ValueError: ``pos_param_names`` was left to detection and the target
            carries no marks, which would emit weights through no positivity
            map at all and leave the target non-convex.
    """

    def __init__(
        self,
        module: nn.Module,
        cond_size: int,
        hidden_sizes: Sequence[int] = DEFAULT_BODY,
        pos_param_names: Optional[Iterable[str]] = None,
        **hypernet_kwargs,
    ) -> None:
        super().__init__()
        if pos_param_names is None:
            pos_param_names = infer_pos_param_names(module)
            if not pos_param_names:
                raise ValueError(
                    f"no constrained parameters found on {type(module).__name__}. "
                    "Mark them with tag_positive_(module, [...]), or pass "
                    "pos_param_names explicitly; an explicit empty set means the "
                    "target applies positivity itself.")
        self.template = module.requires_grad_(False)
        self.hypernet = HyperNetwork(
            input_size=int(cond_size),
            hidden_sizes=list(hidden_sizes),
            downstream_network=module,
            pos_param_names=set(pos_param_names),
            **hypernet_kwargs,
        )

    @property
    def pos_param_names(self) -> set:
        """The parameters positivity is applied to."""
        return self.hypernet.pos_param_names

    def weights(self, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Emit one weight tensor per parameter of the target from ``cond``."""
        return self.hypernet(cond.detach())

    def forward(self, x: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the target on ``x`` with weights emitted from ``cond``.

        ``cond`` defaults to ``x``, which is the coupling the paper studies: the
        weights are emitted from the same batch that forms the gradient.
        """
        return functional_call(
            self.template, self.weights(x if cond is None else cond), (x,))
