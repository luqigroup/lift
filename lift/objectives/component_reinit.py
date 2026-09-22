"""Component re-anchor kernel for the ESS-floor watchdog.

The classical mixture-EM rescue for a mixing weight that has collapsed is
to re-initialize that component's mean and covariance at a random training
point and a typical local scale. The analogue on the emitted parameters of
component ``k`` is:

  1. pick a random anchor ``x*`` from the current batch;
  2. for every readout head whose target name starts with
     ``components.<k>.``, reset the head's bias to its origin value and
     scale the head's weight matrix down by ``shrink``, so the emitted
     parameters revert to a near-fresh state;
  3. drop the Adam moment buffers for those heads, so the optimizer does
     not steer the component back to its collapsed state;
  4. shift ``components.<k>.first_layer.bias`` by an offset proportional
     to ``x*``, so the component's energy bowl starts pulled toward ``x*``.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _readout_heads_for_component_k(
    hyper_E: nn.Module, k: int,
) -> list[tuple[str, nn.Linear, str]]:
    """Yield ``(target_name, readout_module, post_components_name)``.

    ``weight_predictors`` is an ``nn.ModuleList`` aligned with
    ``hyper_E._param_names``, the downstream's ``named_parameters()``
    order. Only the heads whose target name starts with
    ``components.<k>.`` are returned.
    """
    prefix = f"components.{int(k)}."
    out: list[tuple[str, nn.Linear, str]] = []
    for name, layer in zip(hyper_E._param_names, hyper_E.weight_predictors):
        if not name.startswith(prefix):
            continue
        post = name[len(prefix):]
        out.append((name, layer, post))
    return out


def _zero_adam_state_for(opt: torch.optim.Optimizer, params: Iterable[nn.Parameter]) -> None:
    """Drop Adam, or any other first/second-moment optimizer, state slices."""
    state = opt.state
    for p in params:
        if p in state:
            state.pop(p)


def reinit_component_k(
    *,
    hyper_E: nn.Module,
    opt_E: torch.optim.Optimizer,
    K: int,
    D: int,
    k: int,
    x_data: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    shrink: float = 0.1,
    pos_bias_target: float = -2.0,
) -> torch.Tensor:
    """Re-anchor component ``k`` of the EBM hypernet at a random data point.

    Args:
        hyper_E: the EBM hypernet (a
            :class:`lift.models.hypernet.HyperNetwork`). Must have a
            ``_param_names`` list and a ``weight_predictors`` module list
            aligned with the downstream's ``named_parameters()``.
        opt_E: optimizer whose state is reset for the modified parameters.
        K: number of components.
        D: data dimensionality (used to select the input-bias shift).
        k: zero-based index of the component to re-anchor.
        x_data: ``(M, D)`` current training-batch tensor; one row is
            picked uniformly at random as the anchor ``x*``.
        device, dtype: target tensor placement / precision.
        shrink: multiplicative factor on the weight matrix of each
            re-anchored head. The default is a soft reset: it keeps the
            input-conditioned shape and collapses the output magnitude.
        pos_bias_target: bias init for positivity-tagged readouts under
            ``pos_constraint_mode='softplus'``, matching the
            :class:`HyperNetwork` constructor default. The ``square``,
            ``qform`` and ``cone`` modes use zero instead, as they do
            there.

    Returns:
        The chosen anchor ``x*`` (shape ``(D,)``), useful for logging.
    """
    if int(k) < 0 or int(k) >= int(K):
        raise ValueError(f"k={k} out of range for K={K}")
    n = int(x_data.shape[0])
    idx = int(torch.randint(0, n, (1,), device=device).item())
    x_star = x_data[idx, :int(D)].detach().to(device=device, dtype=dtype)

    pos_param_names = hyper_E.pos_param_names
    pos_mode = getattr(hyper_E, "pos_constraint_mode", "softplus")

    heads = _readout_heads_for_component_k(hyper_E, int(k))
    if not heads:
        raise RuntimeError(
            f"no readout heads found for component {k}; nothing to reinit",
        )

    touched_params: list[nn.Parameter] = []
    with torch.no_grad():
        for name, layer, post in heads:
            # Shrinking the readout weight keeps its orientation and
            # collapses its magnitude; the bias takes the init that
            # belongs to the positivity mode.
            layer.weight.data.mul_(float(shrink))
            if name in pos_param_names:
                if pos_mode == "softplus":
                    layer.bias.data.fill_(float(pos_bias_target))
                else:
                    # Under square, qform and cone a zero bias keeps the
                    # post-readout output near zero.
                    layer.bias.data.zero_()
            else:
                layer.bias.data.zero_()
            touched_params.extend([layer.weight, layer.bias])

            # An ICNN whose ``first_layer.bias = -W_first_layer @ x*``
            # has its energy minimum at x*. Since ``first_layer.weight``
            # is emitted too, a small shift on ``first_layer.bias``
            # approximates that and pulls the bowl toward x*.
            if post == "first_layer.bias":
                hidden_dim = int(layer.bias.numel())
                shift = (
                    torch.linspace(-1.0, 1.0, hidden_dim,
                                   device=device, dtype=dtype)
                    * float(x_star.norm())
                    * 0.1
                )
                layer.bias.data.add_(shift)

    # Dropping the Adam moments for the touched heads keeps the next
    # step from undoing the re-anchor.
    _zero_adam_state_for(opt_E, touched_params)

    return x_star.detach()
