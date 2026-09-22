"""Conv-summary tests for :class:`lift.models.HyperNetwork`.

These tests pin the image-conditioned summary contract
(``summary_kind='conv'``): the per-sample conv encoder replaces the
per-sample MLP branch, mean-pooling still happens AFTER the
per-sample encoder (so the emission is permutation-invariant over the
conditioning batch by construction), and the emitted-weights dict is
unchanged in keys, shapes, and positivity -- the same drop-in
contract ``build_hypernet_ebm`` / ``DirectParamModule`` rely on.
CPU-fast: 4 fake images of 1x8x8, tiny widths.
"""

from __future__ import annotations

import pytest
import torch


def _tiny_conv_pair(seed: int = 0):
    """Tiny ConvICNN EBM + conv-summary hypernet through the builder."""
    from lift.objectives.builders import build_hypernet_ebm

    torch.manual_seed(seed)
    return build_hypernet_ebm(
        K=1, D=64, nlayers=3,
        component_kind="conv",
        image_shape=(1, 8, 8),
        conv_hidden_channels=4,
        hyper_hidden_sizes=[8, 8],
        hyper_summary_kind="conv",
        hyper_conv_channels=[4, 8],
    )


def test_conv_summary_emits_target_shapes():
    ebm, hyper = _tiny_conv_pair()
    x = torch.randn(4, 1, 8, 8)
    out = hyper(x)
    expected = {k: v.size() for k, v in ebm.named_parameters()}
    assert set(out) == set(expected)
    for name, size in expected.items():
        assert out[name].shape == size, (
            f"{name}: emitted {tuple(out[name].shape)} != "
            f"target {tuple(size)}"
        )
    # Positivity-tagged tensors must be non-negative element-wise
    # (default softplus emission).
    for name in hyper.pos_param_names:
        assert (out[name] >= 0).all(), f"{name} has negative entries"


def test_conv_summary_permutation_invariance():
    ebm, hyper = _tiny_conv_pair()
    x = torch.randn(4, 1, 8, 8)
    perm = torch.randperm(4, generator=torch.Generator().manual_seed(1))
    # Train mode included: with activation noise at its default 0 the
    # forward is deterministic, so invariance must hold there too. (A
    # batch-statistics bug would be permutation-symmetric and slip
    # past this test either way; the per-sample independence test
    # below pins that class.)
    for train in (True, False):
        hyper.train(train)
        with torch.no_grad():
            out_a = hyper(x)
            out_b = hyper(x[perm])
        for name in out_a:
            assert torch.allclose(out_a[name], out_b[name], atol=1e-6), (
                f"{name} changed under conditioning-batch permutation "
                f"(train={train})"
            )


def test_conv_summary_per_sample_independence():
    """DeepSets sum-decomposition: encoding sample ``i`` alone equals
    row ``i`` of encoding the batch, so the pooled summary is the mean
    of independently encoded samples. Permutation invariance cannot
    catch a batch-statistics (BatchNorm-class) coupling -- batch
    statistics are permutation-symmetric -- so this is the test that
    pins the per-sample branch."""
    ebm, hyper = _tiny_conv_pair()
    hyper.train()
    x = torch.randn(4, 1, 8, 8)
    with torch.no_grad():
        z_batch = hyper.inner_net(x)
        z_rows = torch.cat(
            [hyper.inner_net(x[i:i + 1]) for i in range(4)]
        )
    assert torch.allclose(z_batch, z_rows, atol=1e-6), (
        "per-sample encoding depends on the rest of the batch"
    )


def test_conv_summary_gradients_reach_conv_encoder():
    ebm, hyper = _tiny_conv_pair()
    x = torch.randn(4, 1, 8, 8)
    out = hyper(x)
    loss = sum(v.pow(2).sum() for v in out.values())
    loss.backward()
    grads = [p.grad for p in hyper.inner_net.parameters()]
    assert all(g is not None for g in grads)
    total = sum(float(g.norm()) for g in grads)
    assert total > 0.0, "no gradient reached the conv encoder"


def test_conv_summary_flat_and_image_conditioning_agree():
    """A flat ``(n, D)`` batch must emit the same weights as its
    image-shaped ``(n, C, H, W)`` view (the internal reshape mirrors
    ConvICNN's forward contract)."""
    ebm, hyper = _tiny_conv_pair()
    hyper.eval()
    x = torch.randn(4, 1, 8, 8)
    with torch.no_grad():
        out_img = hyper(x)
        out_flat = hyper(x.reshape(4, 64))
    for name in out_img:
        assert torch.allclose(out_img[name], out_flat[name])


def test_conv_summary_functional_call_energies():
    """Emitted weights drive the frozen ConvICNN via functional_call --
    the training path's consumption pattern."""
    ebm, hyper = _tiny_conv_pair()
    x = torch.randn(4, 1, 8, 8)
    out = hyper(x)
    prefix = "components.0."
    comp_params = {
        k[len(prefix):]: v for k, v in out.items() if k.startswith(prefix)
    }
    E = torch.func.functional_call(
        ebm.components[0], comp_params, x.reshape(4, 64),
    )
    assert E.shape == (4, 1)
    assert torch.isfinite(E).all()


def test_conv_summary_with_shared_readouts():
    """Conv mode composes with ``share_readouts_by_shape`` (the
    FFHQ-scale memory regime): construction, forward, and backward all
    hold the same emitted-dict contract."""
    from lift.models import ConvICNN, HyperNetwork

    torch.manual_seed(0)
    target = ConvICNN(
        input_size=64, image_shape=(1, 8, 8), hidden_channels=4, nlayers=3,
    )
    hyper = HyperNetwork(
        input_size=64,
        hidden_sizes=[8, 8],
        downstream_network=target,
        summary_kind="conv",
        image_shape=(1, 8, 8),
        conv_channels=[4, 8],
        share_readouts_by_shape=True,
    )
    x = torch.randn(4, 1, 8, 8)
    out = hyper(x)
    expected = {k: v.size() for k, v in target.named_parameters()}
    assert set(out) == set(expected)
    for name, size in expected.items():
        assert out[name].shape == size
    loss = sum(v.pow(2).sum() for v in out.values())
    loss.backward()
    grads = [p.grad for p in hyper.parameters()]
    assert all(g is not None for g in grads)
    enc_total = sum(float(p.grad.norm()) for p in hyper.inner_net.parameters())
    assert enc_total > 0.0, "no gradient reached the conv encoder"


def test_conv_summary_forward_error_contract():
    """Image-shaped batches pass through unreshaped (any spatial
    extent); mis-shaped batches raise instead of being silently
    re-sliced into a different number of pseudo-images."""
    ebm, hyper = _tiny_conv_pair()
    hyper.eval()
    # Wrong channel count: rejected, not re-sliced.
    with pytest.raises(ValueError):
        hyper(torch.randn(3, 2, 8, 8))
    # Wrong flat width: rejected.
    with pytest.raises(ValueError):
        hyper(torch.randn(2, 32))
    # Wrong rank: rejected.
    with pytest.raises(ValueError):
        hyper(torch.randn(2, 8, 8))
    # A 4D batch at a non-registered resolution is legitimate: the
    # encoder's global average pool absorbs the spatial extent.
    with torch.no_grad():
        out = hyper(torch.randn(3, 1, 16, 16))
    expected = {k: v.size() for k, v in ebm.named_parameters()}
    assert set(out) == set(expected)
    for name, size in expected.items():
        assert out[name].shape == size


def test_conv_summary_encoder_resolution_agnostic():
    """The standalone encoder runs any spatial extent (global average
    pool), enforces its channel count, and rejects even kernels."""
    from lift.models import ConvSummaryEncoder

    torch.manual_seed(0)
    enc = ConvSummaryEncoder(
        image_shape=(1, 8, 8), out_dim=8, channels=[4, 8],
    )
    for hw in (8, 32):
        z = enc(torch.randn(2, 1, hw, hw))
        assert z.shape == (2, 8)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 3, 8, 8))
    with pytest.raises(ValueError):
        ConvSummaryEncoder(
            image_shape=(1, 8, 8), out_dim=8, kernel_size=4,
        )


def test_conv_summary_validation():
    from lift.models import ConvICNN, HyperNetwork

    target = ConvICNN(
        input_size=64, image_shape=(1, 8, 8), hidden_channels=4, nlayers=3,
    )
    with pytest.raises(ValueError):
        HyperNetwork(
            input_size=64, hidden_sizes=[8], downstream_network=target,
            summary_kind="conv",  # missing image_shape
        )
    with pytest.raises(ValueError):
        HyperNetwork(
            input_size=64, hidden_sizes=[8], downstream_network=target,
            summary_kind="conv", image_shape=(1, 4, 4),  # prod != input_size
        )
    with pytest.raises(ValueError):
        HyperNetwork(
            input_size=64, hidden_sizes=[8], downstream_network=target,
            summary_kind="attention",
        )
    # Default construction is untouched: MLP summary, no image shape.
    hyper = HyperNetwork(
        input_size=64, hidden_sizes=[8], downstream_network=target,
    )
    assert hyper.summary_kind == "mlp"
    assert hyper.image_shape is None
