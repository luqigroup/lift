"""The exponential positivity map ``W = exp(W_raw)`` and the post-update rule.

Direct: ``DirectParamModule(pos_constraint_mode='exp')`` emits the
exponential of its latent, with the latent initialized as
``log(softplus(w_0))`` so that the constrained weight at step zero is the
softplus recipe's. Lift: ``HyperNetwork`` with ``pos_constraint_mode='exp'``
emits ``exp(.)`` on the positivity-tagged readouts, and at a zero readout
pre-activation the emitted weight is the softplus readout's. Post-update:
``pos_constraint_mode='exp_post'`` maps every negative constrained
coordinate to ``exp(w - eps)`` after a step and leaves the rest alone.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts"),
)
from _experiments_hypernet_vs_direct_1d_lib import (  # noqa: E402
    DirectParamModule, _exppost_eps,
)

from lift.objectives.builders import build_hypernet_ebm  # noqa: E402


_KW = dict(K=1, D=1, hidden_dim=32, nlayers=3, strong_convexity=0.02)


def _direct_pair() -> tuple[DirectParamModule, DirectParamModule]:
    torch.manual_seed(0)
    softplus = DirectParamModule(pos_constraint_mode="softplus", **_KW)
    torch.manual_seed(0)
    exp = DirectParamModule(pos_constraint_mode="exp", **_KW)
    return softplus, exp


def test_direct_exp_emits_exp_of_the_latent_and_is_strictly_positive():
    _, arm = _direct_pair()
    out = arm(torch.randn(3, 1))
    latent = dict(arm._inner.named_parameters())
    assert arm._pos_names
    for name in arm._pos_names:
        assert torch.equal(out[name], torch.exp(latent[name]))
        assert bool((out[name] > 0).all())
    for name in set(out) - arm._pos_names:
        assert out[name] is latent[name]


def test_direct_exp_starts_at_the_softplus_recipe_weight():
    softplus, exp = _direct_pair()
    x = torch.randn(3, 1)
    w_sp, w_ex = softplus(x), exp(x)
    for name in exp._pos_names:
        assert torch.allclose(w_ex[name], w_sp[name], rtol=1e-6, atol=1e-7)


def test_direct_exp_chain_rule_factor_is_the_weight_itself():
    _, arm = _direct_pair()
    out = arm(torch.randn(3, 1))
    sum(out[n].sum() for n in arm._pos_names).backward()
    for name, p in arm._inner.named_parameters():
        if name in arm._pos_names:
            assert torch.allclose(p.grad, torch.exp(p.detach()))


def test_direct_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="pos_constraint_mode"):
        DirectParamModule(pos_constraint_mode="log", **_KW)


def test_exp_post_maps_negative_coordinates_and_leaves_the_rest():
    torch.manual_seed(0)
    arm = DirectParamModule(pos_constraint_mode="exp_post", **_KW)
    params = dict(arm._inner.named_parameters())
    with torch.no_grad():
        for name in arm._pos_names:
            params[name].normal_()
    before = {n: params[n].detach().clone() for n in arm._pos_names}
    arm.project_positive()
    for name in arm._pos_names:
        w0, w1 = before[name], params[name].detach()
        neg = w0 < 0
        assert bool(neg.any()) and bool((~neg).any())
        assert torch.equal(w1[~neg], w0[~neg])
        assert torch.allclose(w1[neg], torch.exp(w0[neg] - 5.0))
        assert bool((w1[neg] > 0).all())
        assert float(w1[neg].max()) <= math.exp(-5.0)


def test_exppost_backend_names_carry_the_constant():
    assert _exppost_eps("direct_exppost") == 5.0
    assert _exppost_eps("direct_exppost8.8") == 8.8
    assert _exppost_eps("direct") is None and _exppost_eps("pgd") is None


def _lift(mode: str):
    torch.manual_seed(0)
    _, hyper = build_hypernet_ebm(
        hyper_hidden_sizes=[8, 8], hyper_pos_constraint_mode=mode, **_KW,
    )
    with torch.no_grad():
        for name, layer in zip(hyper._param_names, hyper.weight_predictors):
            if name in hyper.pos_param_names:
                layer.weight.zero_()
    return hyper


def test_lift_exp_readout_is_positive_and_matches_softplus_at_zero_preactivation():
    x = torch.randn(16, 1)
    w_sp, w_ex = _lift("softplus")(x), _lift("exp")(x)
    assert set(w_ex) == set(w_sp)
    tagged = _lift("exp").pos_param_names
    assert tagged
    for name in tagged:
        assert bool((w_ex[name] > 0).all())
        assert torch.allclose(w_ex[name], w_sp[name], rtol=1e-5, atol=1e-7)
