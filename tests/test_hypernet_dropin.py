"""Drop-in API tests for :class:`lift.models.HyperNetwork`.

These tests pin the auto-detection contract that makes
``HyperNetwork(input_size, hidden_sizes, downstream_network=icnn)`` a
one-line drop-in:

  * Constructing a canonical :class:`ICNN` and passing it to the
    hypernet with ``pos_param_names=None`` must auto-infer exactly the
    same name set as the manual :func:`icnn_pos_param_names` helper.
  * Same for the canonical :class:`ConvICNN` vs
    :func:`conv_icnn_pos_param_names`.
  * CP-Flow's vendored ``ICNN2`` (no ``_pos_required`` tags;
    positivity is internal to ``PosLinear``) must continue to yield
    the empty set, matching the historical ``None``-fallback. This
    pins the no-behaviour-change guarantee for the CP-Flow HEPMASS
    driver.
"""

from __future__ import annotations


def test_hypernet_dropin_infers_icnn_pos_names():
    from lift.models import (
        HyperNetwork,
        ICNN,
        icnn_pos_param_names,
    )

    nlayers = 5
    icnn = ICNN(input_size=21, hidden_dim=64, nlayers=nlayers)
    hyper = HyperNetwork(
        input_size=21,
        hidden_sizes=[64, 64],
        downstream_network=icnn,
        pos_param_names=None,
    )
    expected = icnn_pos_param_names(nlayers)
    assert hyper.pos_param_names == expected, (
        f"auto-inferred {sorted(hyper.pos_param_names)} != "
        f"icnn_pos_param_names(nlayers={nlayers})={sorted(expected)}"
    )
    assert hyper._pos_names == sorted(expected)
    assert hyper._pos_auto_inferred is True


def test_hypernet_dropin_explicit_overrides_inference():
    """Explicit ``pos_param_names`` must NOT be overwritten by the walker."""
    from lift.models import HyperNetwork, ICNN

    icnn = ICNN(input_size=4, hidden_dim=8, nlayers=3)
    # Override with empty set: must take precedence over auto-detect.
    hyper = HyperNetwork(
        input_size=4,
        hidden_sizes=[8],
        downstream_network=icnn,
        pos_param_names=set(),
    )
    assert hyper.pos_param_names == set()
    assert hyper._pos_auto_inferred is False


def test_hypernet_dropin_infers_conv_icnn_pos_names():
    from lift.models import (
        ConvICNN,
        HyperNetwork,
        conv_icnn_pos_param_names,
    )

    nlayers = 3
    cicnn = ConvICNN(
        input_size=28 * 28,
        image_shape=(1, 28, 28),
        hidden_channels=8,
        nlayers=nlayers,
    )
    hyper = HyperNetwork(
        input_size=4,
        hidden_sizes=[8],
        downstream_network=cicnn,
        pos_param_names=None,
    )
    expected = conv_icnn_pos_param_names(nlayers)
    assert hyper.pos_param_names == expected


def test_hypernet_dropin_untagged_module_yields_empty_set():
    """CP-Flow's ``ICNN2`` does not tag params (PosLinear handles
    positivity internally). The walker must therefore return the empty
    set, exactly matching the pre-refactor ``pos_param_names=None``
    behaviour the HEPMASS / 2-D CPFlow drivers rely on."""
    from lift.baselines.cpflow_vendor import ICNN2
    from lift.models import HyperNetwork

    icnn2 = ICNN2(dim=4, dimh=8, num_hidden_layers=2)
    hyper = HyperNetwork(
        input_size=4,
        hidden_sizes=[8],
        downstream_network=icnn2,
        pos_param_names=None,
    )
    assert hyper.pos_param_names == set()
