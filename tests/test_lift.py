"""The public lift API: emission, positivity, freezing, and detection."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from lift.lift import DEFAULT_BODY, Lift, tag_positive_
from lift.models import ICNN


def _icnn(d=2, h=32, n=3):
    torch.manual_seed(0)
    return ICNN(input_size=d, hidden_dim=h, nlayers=n)


def test_emitted_constrained_weights_are_positive():
    lf = Lift(_icnn(), cond_size=2)
    w = lf.weights(torch.randn(64, 2))
    assert lf.pos_param_names
    for name in lf.pos_param_names:
        assert w[name].min().item() > 0.0


def test_forward_conditions_on_the_same_batch_by_default():
    lf = Lift(_icnn(), cond_size=2)
    x = torch.randn(64, 2)
    torch.testing.assert_close(lf(x), lf(x, cond=x))


def test_emission_varies_with_the_batch():
    """The lift is the batch dependence; a constant emission is direct softplus."""
    lf = Lift(_icnn(), cond_size=2)
    a = lf.weights(torch.randn(64, 2))
    b = lf.weights(torch.randn(64, 2) + 5.0)
    name = sorted(lf.pos_param_names)[0]
    assert not torch.allclose(a[name], b[name])


def test_summary_is_permutation_invariant():
    lf = Lift(_icnn(), cond_size=2)
    x = torch.randn(64, 2)
    w = lf.weights(x)
    wp = lf.weights(x[torch.randperm(x.shape[0])])
    for name in w:
        torch.testing.assert_close(w[name], wp[name], rtol=1e-4, atol=1e-6)


def test_template_is_frozen_and_gradients_reach_the_lift():
    icnn = _icnn()
    lf = Lift(icnn, cond_size=2)
    lf(torch.randn(64, 2)).mean().backward()
    assert all(not p.requires_grad for p in icnn.parameters())
    assert all(p.grad is None for p in icnn.parameters())
    assert any(p.grad is not None for p in lf.hypernet.parameters())


def test_trainable_parameters_are_exactly_the_lift():
    icnn = _icnn()
    lf = Lift(icnn, cond_size=2)
    trainable = {id(p) for p in lf.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in lf.hypernet.parameters() if p.requires_grad}
    assert not (trainable & {id(p) for p in icnn.parameters()})


def test_untagged_module_is_refused_rather_than_silently_unconstrained():
    """Detection returns the empty set for an unmarked module, which would
    leave the target non-convex. The wrapper refuses instead."""
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 1)

        def forward(self, x):
            return self.lin(x)

    with pytest.raises(ValueError, match="no constrained parameters"):
        Lift(Plain(), cond_size=2)


def test_tag_positive_makes_a_foreign_module_liftable():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 1)

        def forward(self, x):
            return self.lin(x)

    lf = Lift(tag_positive_(Plain(), ["lin.weight"]), cond_size=2)
    assert lf.pos_param_names == {"lin.weight"}
    assert lf.weights(torch.randn(32, 2))["lin.weight"].min().item() > 0.0


def test_tag_positive_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="is not a parameter"):
        tag_positive_(nn.Linear(2, 1), ["nope"])


def test_explicit_empty_set_is_honored():
    """A target that applies positivity itself passes an explicit empty set."""
    lf = Lift(_icnn(), cond_size=2, pos_param_names=set())
    assert lf.pos_param_names == set()


def test_default_body_matches_the_shipped_recipe():
    assert tuple(DEFAULT_BODY) == (64, 64, 96)


def test_lifted_energy_is_convex_in_its_input():
    """The point of the constraint: with the weights held at one emission the
    target is convex, so the midpoint sits at or below the chord."""
    lf = Lift(_icnn(d=2, h=64, n=4), cond_size=2)
    cond = torch.randn(128, 2)
    w = lf.weights(cond)
    a, b = torch.randn(64, 2), torch.randn(64, 2)
    f = lambda z: torch.func.functional_call(lf.template, w, (z,))
    mid = f(0.5 * (a + b))
    chord = 0.5 * (f(a) + f(b))
    assert (mid <= chord + 1e-4).all()
