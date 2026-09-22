"""Helpers for slicing and detaching the flat parameter dicts the
hypernetwork emits.

The hypernetwork returns ``{name: tensor}`` mirroring the downstream
module's ``named_parameters()``; per-component access requires a prefix
slice (``components.{k}.`` for the EBM, ``samplers.{k}.`` for
:class:`MultiSamplerFlow`).
"""

from __future__ import annotations

from typing import Dict

import torch


def slice_per_k(params: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Strip ``prefix`` from every key starting with it. Other keys dropped."""
    return {key[len(prefix):]: val
            for key, val in params.items()
            if key.startswith(prefix)}


def detach_dict(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Element-wise ``.detach()`` on every tensor in ``d``."""
    return {k: v.detach() for k, v in d.items()}


def stack_per_k_params(
    params: Dict[str, torch.Tensor],
    K: int,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Stack ``K`` copies of a per-component parameter dict along a leading dim.

    Given the hypernet output ``{f'{prefix}{k}.{name}': tensor}`` for
    ``k = 0,...,K-1``, returns ``{name: tensor of shape (K, *original)}``
    by stacking the per-K tensors along dim 0. The result tensors share
    autograd graph with the input tensors, so the outer
    ``loss.backward()`` reaches the original parameters.

    Used as the boundary between the per-K hypernetwork output and the
    vmap'd primitives in :mod:`lift.objectives._batched`.
    """
    per_name: Dict[str, list[torch.Tensor]] = {}
    for k in range(int(K)):
        sub = slice_per_k(params, f"{prefix}{k}.")
        for name, t in sub.items():
            per_name.setdefault(name, [None] * int(K))[k] = t
    stacked: Dict[str, torch.Tensor] = {}
    for name, lst in per_name.items():
        if any(t is None for t in lst):
            missing = [k for k, t in enumerate(lst) if t is None]
            raise KeyError(
                f"missing per-K entries for {name!r} at indices {missing} "
                f"(prefix={prefix!r})",
            )
        stacked[name] = torch.stack(lst, dim=0)
    return stacked
