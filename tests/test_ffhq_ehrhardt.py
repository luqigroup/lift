"""Contract tests for the FFHQ ICNN-regularizer line (Ehrhardt pipeline).

CPU-fast: the template runs at img_size=32 (the smallest extent the
vendored 16x16/stride-16 average pool supports beyond 16 itself, and
divisible by 16 as the architecture requires), with tiny fake data.
The one dataset test that needs the shipped FFHQ archive is skipped
when ``data/third_party/ehrhardt_icnn_primal_dual`` is absent.
"""

from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "scripts"),
)

from lift.baselines.ehrhardt_vendor import (  # noqa: E402
    InpaintingPhysics,
    L2Fidelity,
    PDHG_ICNN,
    PDHG_ICNN_L1,
    compare_psnr,
    simple_ICNN,
    simple_ICNNPrior,
    soft_thresh,
)

_IMG = 32
_DEVICE = "cpu"


def _template_args():
    # (n_channels, n_filters, kernel_size, img_size, smoothed, device)
    return (1, 4, 5, _IMG, True, _DEVICE)


def _fake_pairs(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    clean = torch.rand(n, 1, _IMG, _IMG, generator=g)
    mask = (torch.rand(1, 1, _IMG, _IMG, generator=g) > 0.3).float()
    noisy = mask * clean + 0.03 * torch.randn(
        n, 1, _IMG, _IMG, generator=g,
    )
    return clean, noisy, mask


# ------------------------------------------------------------- vendor


def test_constrained_weight_count_matches_plan():
    """simple_ICNN(1, 32, 5, 256) has 2,097,408 constrained weights."""
    m = simple_ICNN(1, 32, 5, 256, True, "cpu")
    n_constrained = m.fc1.weight.numel() + m.fc2.weight.numel()
    assert n_constrained == 2_097_408
    assert sum(p.numel() for p in m.parameters()) == 2_098_497


def test_their_init_starts_inside_shoulder():
    torch.manual_seed(0)
    prior = simple_ICNNPrior(*_template_args())
    w = torch.cat([prior.icnn.fc1.weight.flatten(),
                   prior.icnn.fc2.weight.flatten()])
    assert w.min() >= 0.0
    assert w.max() <= 0.01
    # 100 percent shoulder occupancy at init (edge = softplus(-2.944)).
    assert (w <= 0.0513).float().mean() == 1.0


def test_soft_thresh_is_l1_prox():
    torch.manual_seed(1)
    x = torch.randn(64)
    y = torch.randn(64)
    tau = 0.3
    out = soft_thresh(x, y, tau)
    d = x - y
    expected = y + torch.sign(d) * torch.clamp(d.abs() - tau, min=0.0)
    assert torch.allclose(out, expected)
    # Fixed point: within tau of y, the prox returns y exactly.
    near = y + 0.1 * torch.sign(torch.randn(64)) * tau
    assert torch.allclose(soft_thresh(near, y, tau), y)


def test_compare_psnr_matches_skimage_semantics():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, size=(1, 1, 8, 8))
    b = np.clip(a + rng.normal(0, 0.05, size=a.shape), 0, 1)
    mse = np.mean((a - b) ** 2)
    # Non-negative float true image -> data_range inferred as 1.0.
    assert compare_psnr(a, b) == pytest.approx(10 * np.log10(1.0 / mse))


def test_pdhg_inpaint_smoke():
    torch.manual_seed(2)
    prior = simple_ICNNPrior(*_template_args())
    clean, noisy, mask = _fake_pairs(n=1, seed=2)
    physics = InpaintingPhysics(mask)
    fid = L2Fidelity()
    y = noisy[:1]
    x0 = physics.A_adjoint(y)
    x, iters, res = PDHG_ICNN(
        x0, y, clean[:1], physics, fid, prior,
        0.1, 0.01, 0.001, 1.0, 1.0,
        max_iter=3, tol=1e-12, device=_DEVICE,
    )
    assert x.shape == y.shape
    assert torch.isfinite(x).all()
    assert iters == 3


def test_pdhg_l1_smoke():
    torch.manual_seed(3)
    prior = simple_ICNNPrior(*_template_args())
    clean, _, _ = _fake_pairs(n=1, seed=3)
    y = clean[:1].clone()
    y[torch.rand_like(y) < 0.1] = 0.0
    x = PDHG_ICNN_L1(y, prior, 0.02, 0.1, 5e-5, max_iter=3,
                     device=_DEVICE)
    assert x.shape == y.shape
    assert torch.isfinite(x).all()


# ------------------------------------------------------------ backends


def _one_step(adapter, clean, noisy):
    from _experiments_ffhq_icnn_regularizer_lib import _gradient_penalty

    trainable = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=5e-4, betas=(0.5, 0.99))
    adapter.begin_step(clean)
    diff = adapter.g(clean).mean() - adapter.g(noisy).mean()
    gp = _gradient_penalty(adapter.g, clean.detach(), noisy.detach())
    loss = diff + 5.0 * gp
    opt.zero_grad()
    loss.backward()
    opt.step()
    adapter.post_step()
    return float(loss.item())


def test_pgd_backend_one_step_stays_nonnegative():
    from _experiments_ffhq_icnn_regularizer_lib import PGDBackend

    torch.manual_seed(4)
    adapter = PGDBackend(*_template_args())
    clean, noisy, _ = _fake_pairs(seed=4)
    loss = _one_step(adapter, clean.clamp(0, 1), noisy.clamp(0, 1))
    assert np.isfinite(loss)
    eff = adapter.effective_param_dict()
    assert eff["fc1.weight"].min() >= 0.0
    assert eff["fc2.weight"].min() >= 0.0
    # The input conv wx is UNCONSTRAINED in their recipe: it must keep
    # negative entries after the step + post-step clamp (an extra clamp
    # on the wrong tensor would silently pass without this).
    assert (adapter.prior.icnn.wx.weight < 0).any()


def test_gradient_penalty_matches_analytic_linear_critic():
    """For g = a * sum(x), grad_x g = a * ones, so the one-sided GP is
    exactly relu(a * sqrt(D) - 1)^2 regardless of the interpolates --
    catches a sign flip or a two-sided variant."""
    from _experiments_ffhq_icnn_regularizer_lib import _gradient_penalty

    torch.manual_seed(8)
    D = 1 * _IMG * _IMG
    real = torch.rand(4, 1, _IMG, _IMG)
    fake = torch.rand(4, 1, _IMG, _IMG)

    a = 0.1  # a * sqrt(D) = 3.2 > 1: penalty active
    gp = _gradient_penalty(lambda x: a * x.flatten(1).sum(dim=1),
                           real, fake)
    expected = max(a * math.sqrt(D) - 1.0, 0.0) ** 2
    assert float(gp) == pytest.approx(expected, abs=1e-3)

    a2 = 0.01  # a2 * sqrt(D) = 0.32 < 1: one-sided penalty exactly 0
    gp2 = _gradient_penalty(lambda x: a2 * x.flatten(1).sum(dim=1),
                            real, fake)
    assert float(gp2) == 0.0


def test_hypernet_emission_identity_no_double_softplus():
    """Emitted tensors equal softplus(readout(mean(encoder(cond))))
    computed manually -- and the untagged tensors get NO softplus. A
    double softplus anywhere in the hypernet path breaks equality."""
    from _experiments_ffhq_icnn_regularizer_lib import (
        CONSTRAINED_NAMES,
        HypernetBackend,
    )

    torch.manual_seed(9)
    adapter = HypernetBackend(
        *_template_args(),
        hyper_hidden_sizes=[16],
        conv_channels=[8, 16],
        head_init_pos_bias=-2.0,
    )
    cond = torch.rand(4, 1, _IMG, _IMG).clamp(0, 1)
    emitted = adapter.effective_param_dict(cond=cond)
    h = adapter.hyper
    with torch.no_grad():
        z = h.inner_net(cond).mean(dim=0, keepdim=True)
        for i, name in enumerate(h._param_names):
            flat = h.weight_predictors[i](z)
            if name in CONSTRAINED_NAMES:
                flat = F.softplus(flat)
            manual = flat.view(h.target_named_param_sizes[name])
            assert torch.equal(emitted[name], manual), name


def test_direct_backend_one_step_softplus_reparam():
    from _experiments_ffhq_icnn_regularizer_lib import (
        DirectBackend,
        softplus_inv,
    )

    torch.manual_seed(5)
    adapter = DirectBackend(*_template_args())
    # Init: V = softplus-inverse of the U[0, 0.01] draw, deep inside
    # the shoulder (median pre-readout ~ -5.3).
    v1 = adapter.raw["fc1__weight"].detach()
    assert v1.median() < -4.0
    eff0 = adapter.effective_param_dict()
    assert eff0["fc1.weight"].max() <= 0.01 + 1e-6
    clean, noisy, _ = _fake_pairs(seed=5)
    before = adapter.raw["fc1__weight"].detach().clone()
    loss = _one_step(adapter, clean.clamp(0, 1), noisy.clamp(0, 1))
    assert np.isfinite(loss)
    assert not torch.equal(before, adapter.raw["fc1__weight"].detach())
    eff = adapter.effective_param_dict()
    assert eff["fc1.weight"].min() > 0.0  # softplus is strictly positive
    # softplus_inv inverts softplus on the effective weights (float32
    # round-trip at pre-readouts near -5 resolves to about 5e-4).
    assert torch.allclose(
        softplus_inv(eff0["fc1.weight"]).clamp(min=-30), v1.clamp(min=-30),
        atol=5e-3,
    )


def test_hypernet_backend_emits_template_shapes_and_trains():
    from _experiments_ffhq_icnn_regularizer_lib import HypernetBackend

    torch.manual_seed(6)
    adapter = HypernetBackend(
        *_template_args(),
        hyper_hidden_sizes=[16],
        conv_channels=[8, 16],
        head_init_pos_bias=-2.0,
    )
    clean, noisy, _ = _fake_pairs(seed=6)
    clean, noisy = clean.clamp(0, 1), noisy.clamp(0, 1)

    # Emitted tensors match the vendored template's named parameters.
    adapter.begin_step(clean)
    template_shapes = {
        n: tuple(p.shape) for n, p in adapter.template.named_parameters()
    }
    emitted_shapes = {
        n: tuple(t.shape) for n, t in adapter._params.items()
    }
    assert emitted_shapes == template_shapes
    # The constrained tensors come out of the softplus readout.
    assert adapter._params["fc1.weight"].min() > 0.0
    assert adapter._params["fc2.weight"].min() > 0.0
    adapter.post_step()

    before = adapter.effective_param_dict(cond=clean)
    loss = _one_step(adapter, clean, noisy)
    assert np.isfinite(loss)
    after = adapter.effective_param_dict(cond=clean)
    assert not torch.equal(before["fc1.weight"], after["fc1.weight"])

    # Deployment freeze over a fake "training set" pools every image.
    class _Pairs:
        pass

    pairs = _Pairs()
    pairs.clean = clean
    frozen = adapter.export_param_dict(pairs, _DEVICE, chunk=2)
    assert {n: tuple(t.shape) for n, t in frozen.items()} == template_shapes
    # Full-set pooling over exactly the conditioning batch must agree
    # with a direct emission on that batch (same DeepSets mean).
    direct_emit = adapter.effective_param_dict(cond=clean)
    for name in frozen:
        assert torch.allclose(
            frozen[name], direct_emit[name], atol=1e-5,
        ), name


def test_effective_weights_load_into_plain_prior():
    from _experiments_ffhq_icnn_regularizer_lib import (
        FFHQICNNRegularizer,
        HypernetBackend,
    )

    torch.manual_seed(7)
    adapter = HypernetBackend(
        *_template_args(),
        hyper_hidden_sizes=[16],
        conv_channels=[8],
        head_init_pos_bias=-2.0,
    )
    clean, _, _ = _fake_pairs(seed=7)
    eff = adapter.effective_param_dict(cond=clean.clamp(0, 1))
    prior = simple_ICNNPrior(*_template_args())
    FFHQICNNRegularizer._load_effective(prior, eff)
    for name, t in eff.items():
        assert torch.equal(prior.icnn.get_parameter(name).data, t)
    # The loaded prior evaluates the same energy as the functional path.
    x = torch.rand(2, 1, _IMG, _IMG)
    adapter.begin_step(clean.clamp(0, 1))
    with torch.no_grad():
        g_fun = adapter.g(x)
        g_prior = prior.g(x)
    assert torch.allclose(g_fun, g_prior, atol=1e-6)


def test_constrained_stats_screen_fields():
    from _experiments_ffhq_icnn_regularizer_lib import (
        SHOULDER_EDGE,
        _constrained_stats,
    )

    assert SHOULDER_EDGE == pytest.approx(0.0513, abs=1e-3)
    w = torch.tensor([0.0, 0.0, 0.005, 0.05, 0.2, 1.0])
    s = _constrained_stats(w)
    assert s["frac_zero"] == pytest.approx(2 / 6)
    assert s["frac_shoulder"] == pytest.approx(4 / 6)
    assert np.isfinite(s["median_pre_softplus_pos"])


# ---------------------------------------- best-val checkpoint selection


class _FakeFFHQPairs:
    """Duck-typed stand-in for FFHQPairs: clean / noisy / mask / len."""

    def __init__(self, n, seed, mean_match=False):
        from lift.dataset.ffhq import dc_matched_negatives

        clean, noisy, mask = _fake_pairs(n=n, seed=seed)
        self.clean = clean.clamp(0, 1)
        self.noisy = noisy
        self.negatives = (dc_matched_negatives(self.clean, noisy)
                          if mean_match else noisy)
        self.mask = mask

    def __len__(self):
        return int(self.clean.shape[0])


def _make_args(**over):
    """Args namespace mirroring configs/experiments_ffhq_icnn_regularizer
    .json, shrunk to CPU-fast extents (2 training steps, 2-iter PDHG)."""
    base = dict(
        task="inpaint", data_root="", img_size=_IMG, n_channels=1,
        n_filters=4, kernel_size=5, smoothed=1, noise_sigma=0.03,
        snp_prob=0.25, n_epochs=1, batch_size=3, lr=5e-4,
        adam_beta1=0.5, adam_beta2=0.99, lambda_gp=5.0,
        hyper_hidden_sizes=[8], conv_channels=[4, 8],
        head_init_pos_bias=-2.0, log_every=1, eval_every=1,
        checkpoint_every=0, best_val_ckpt=1, val_n_images=2,
        val_pdhg_max_iter=2, eval_pdhg_max_iter=2, pdhg_tol=0.1,
        pdhg_lambda=0.1, pdhg_c1=0.01, pdhg_c2=0.001,
        snp_reg_param=0.02, snp_c1=0.1, snp_c2=5e-5,
        snp_eval_max_iter=2, eval_noise_seed=0, n_recon_examples=1,
        seeds=[0], backends=["pgd"], seed=0, gpu_id=0, phase="train", experiment="test_ffhq_icnn_regularizer",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _run_arm(root, backend, best_val_ckpt, seed=0, **over):
    """Run one (backend, seed) arm on fake data; return its arm dir."""
    from _experiments_ffhq_icnn_regularizer_lib import FFHQICNNRegularizer

    exp = FFHQICNNRegularizer(_make_args(best_val_ckpt=best_val_ckpt, **over))
    exp.device = torch.device("cpu")  # never touch a busy local GPU
    exp._pairs_cache[("train", seed)] = _FakeFFHQPairs(n=6, seed=10)
    exp._pairs_cache[("test", 0)] = _FakeFFHQPairs(n=2, seed=20)
    arm_dir = os.path.join(str(root), f"{backend}_seed{seed}")
    os.makedirs(arm_dir, exist_ok=True)
    exp._train_one_arm(backend, seed, arm_dir)
    return arm_dir


_BESTVAL_KEYS = ("test_psnr_bestval_mean", "test_psnr_bestval_std",
                 "bestval_step", "bestval_val_psnr")


def test_best_val_ckpt_written_and_reported(tmp_path):
    """Knob on: icnn_best.pt exists (val improved at least once -- the
    first finite eval always beats -inf), carries CPU weights + step +
    val PSNR, and the argmax of the recorded val curve is what got
    selected and reported."""
    arm_dir = _run_arm(tmp_path, "pgd", best_val_ckpt=1)

    best_path = os.path.join(arm_dir, "icnn_best.pt")
    assert os.path.isfile(best_path)
    ck = torch.load(best_path, map_location="cpu", weights_only=True)
    assert set(ck) >= {"step", "val_psnr", "prior_state"}
    assert all(v.device.type == "cpu" for v in ck["prior_state"].values())
    # Same tensor set as icnn_final.pt (a plain prior state dict).
    fin = torch.load(os.path.join(arm_dir, "icnn_final.pt"),
                     map_location="cpu", weights_only=True)
    assert set(ck["prior_state"]) == set(fin)

    with open(os.path.join(arm_dir, "metrics.json")) as f:
        m = json.load(f)
    for k in _BESTVAL_KEYS:
        assert k in m, k
    vals, steps = m["history"]["val_psnr"], m["history"]["eval_steps"]
    assert len(vals) >= 2  # the run actually had improvement chances
    i_best = int(np.argmax(vals))  # first max == strict-> semantics
    assert m["bestval_val_psnr"] == pytest.approx(max(vals))
    assert m["bestval_step"] == steps[i_best]
    assert ck["step"] == steps[i_best]
    assert ck["val_psnr"] == pytest.approx(max(vals))
    assert len(m["test_psnr_bestval_per_image"]) == 2
    # Last-iterate keys unchanged alongside.
    assert "test_psnr_mean" in m and "test_psnr_std" in m

    with open(os.path.join(arm_dir, "RESULTS_COMPLETE.json")) as f:
        marker = json.load(f)
    for k in _BESTVAL_KEYS:
        assert k in marker, k


def test_best_val_hypernet_goes_through_emission_export(
        tmp_path, monkeypatch):
    """The hypernet arm's best-val capture uses the SAME chunked
    full-training-set emission path as its final export (>= 1 best
    capture during training + 1 final export), never the
    diagnostic-batch emission."""
    import _experiments_ffhq_icnn_regularizer_lib as lib

    calls = []
    orig = lib.HypernetBackend.export_param_dict

    def spy(self, pairs, device, chunk=64):
        calls.append(pairs)
        return orig(self, pairs, device, chunk)

    monkeypatch.setattr(lib.HypernetBackend, "export_param_dict", spy)
    arm_dir = _run_arm(tmp_path, "hypernet", best_val_ckpt=1)

    assert len(calls) >= 2
    assert all(p is not None and hasattr(p, "clean") for p in calls)
    ck = torch.load(os.path.join(arm_dir, "icnn_best.pt"),
                    map_location="cpu", weights_only=True)
    # Softplus emission: constrained tensors strictly positive.
    assert ck["prior_state"]["icnn.fc1.weight"].min() > 0.0
    assert ck["prior_state"]["icnn.fc2.weight"].min() > 0.0
    with open(os.path.join(arm_dir, "metrics.json")) as f:
        m = json.load(f)
    for k in _BESTVAL_KEYS:
        assert k in m, k


def test_best_val_ckpt_disabled_restores_old_outputs(tmp_path):
    """Knob off: no icnn_best.pt, no new keys anywhere, and the exact
    same file set and training trajectory as before the knob existed
    (the knob-on run must not perturb RNG or results either)."""
    d_on = _run_arm(tmp_path / "on", "direct", best_val_ckpt=1)
    d_off = _run_arm(tmp_path / "off", "direct", best_val_ckpt=0)

    old_files = {"icnn_final.pt", "metrics.json",
                 "RESULTS_COMPLETE.json", "recon_examples.npz"}
    assert set(os.listdir(d_off)) == old_files
    assert set(os.listdir(d_on)) == old_files | {"icnn_best.pt"}

    new_keys = set(_BESTVAL_KEYS) | {"test_psnr_bestval_per_image"}
    with open(os.path.join(d_off, "metrics.json")) as f:
        m_off = json.load(f)
    with open(os.path.join(d_on, "metrics.json")) as f:
        m_on = json.load(f)
    assert not (new_keys & set(m_off))
    with open(os.path.join(d_off, "RESULTS_COMPLETE.json")) as f:
        marker_off = json.load(f)
    assert not (new_keys & set(marker_off))
    # Identical trajectory and last-iterate results at equal seed: the
    # best-val machinery consumes no RNG and touches no live weights.
    assert m_on["history"] == m_off["history"]
    assert m_on["test_psnr_per_image"] == m_off["test_psnr_per_image"]
    assert m_on["test_psnr_mean"] == m_off["test_psnr_mean"]
    assert m_on["test_psnr_std"] == m_off["test_psnr_std"]


def test_save_emitter_zero_omits_emitter_file(tmp_path):
    """``save_emitter=0`` skips only ``emitter_final.pt`` (the sweep's
    ~0.5-1 GB provenance file); 1 (the default) keeps it; every
    deployment artifact is written either way."""
    d_on = _run_arm(tmp_path / "on", "hypernet", best_val_ckpt=0,
                    save_emitter=1)
    d_off = _run_arm(tmp_path / "off", "hypernet", best_val_ckpt=0,
                     save_emitter=0)

    assert os.path.isfile(os.path.join(d_on, "emitter_final.pt"))
    assert not os.path.exists(os.path.join(d_off, "emitter_final.pt"))
    for d in (d_on, d_off):
        for f in ("icnn_final.pt", "metrics.json",
                  "RESULTS_COMPLETE.json", "recon_examples.npz"):
            assert os.path.isfile(os.path.join(d, f)), (d, f)
    # The knob touches nothing computed: identical trajectory and
    # last-iterate results at equal seed.
    with open(os.path.join(d_on, "metrics.json")) as f:
        m_on = json.load(f)
    with open(os.path.join(d_off, "metrics.json")) as f:
        m_off = json.load(f)
    assert m_on["history"] == m_off["history"]
    assert m_on["test_psnr_mean"] == m_off["test_psnr_mean"]


def test_run_root_summary_records_args(tmp_path, monkeypatch):
    """The run-root metrics.json carries the resolved run arguments
    under "args" (JSON-serializable), which is how the sweep
    aggregator identifies a hash dir's cell."""
    import _experiments_ffhq_icnn_regularizer_lib as lib

    monkeypatch.setattr(lib, "checkpointsdir", lambda name: str(tmp_path))
    exp = lib.FFHQICNNRegularizer(_make_args(best_val_ckpt=0))
    exp.device = torch.device("cpu")
    exp._pairs_cache[("train", 0)] = _FakeFFHQPairs(n=6, seed=10)
    exp._pairs_cache[("test", 0)] = _FakeFFHQPairs(n=2, seed=20)
    arm_dir = os.path.join(str(tmp_path), "pgd_seed0")
    os.makedirs(arm_dir, exist_ok=True)
    exp._train_one_arm("pgd", 0, arm_dir)
    exp._write_summary()

    with open(os.path.join(tmp_path, "metrics.json")) as f:
        summary = json.load(f)  # implicitly: valid JSON end to end
    a = summary["args"]
    assert a["backends"] == ["pgd"]
    assert a["seeds"] == [0]
    assert a["lr"] == pytest.approx(5e-4)
    assert a["conv_channels"] == [4, 8]
    assert a["hyper_hidden_sizes"] == [8]
    # Non-serializable values are stringified, never dropped.
    exp.args.device_probe = torch.device("cpu")
    exp._write_summary()
    with open(os.path.join(tmp_path, "metrics.json")) as f:
        summary2 = json.load(f)
    assert summary2["args"]["device_probe"] == "cpu"


def test_failed_arm_is_recorded_and_sweep_continues(tmp_path, monkeypatch):
    """A push-sweep cell expects some recipe to diverge at a large step:
    the failing arm gets RESULTS_FAILED.json (and no completion marker,
    so a resubmission retries it) and the remaining arms still run."""
    import _experiments_ffhq_icnn_regularizer_lib as lib

    monkeypatch.setattr(lib, "checkpointsdir", lambda name: str(tmp_path))
    exp = lib.FFHQICNNRegularizer(
        _make_args(best_val_ckpt=0, backends=["direct", "pgd"], seeds=[0]))
    exp.device = torch.device("cpu")
    for s in (0,):
        exp._pairs_cache[("train", s)] = _FakeFFHQPairs(n=6, seed=10)
    exp._pairs_cache[("test", 0)] = _FakeFFHQPairs(n=2, seed=20)

    real = exp._train_one_arm

    def flaky(backend, seed, arm_dir):
        if backend == "direct":
            raise RuntimeError("non-finite loss at step 1/2 (test)")
        return real(backend, seed, arm_dir)

    monkeypatch.setattr(exp, "_train_one_arm", flaky)
    exp.train()

    direct_dir = os.path.join(str(tmp_path), "direct_seed0")
    pgd_dir = os.path.join(str(tmp_path), "pgd_seed0")
    with open(os.path.join(direct_dir, "RESULTS_FAILED.json")) as f:
        rec = json.load(f)
    assert rec["backend"] == "direct" and "non-finite" in rec["error"]
    assert not os.path.isfile(os.path.join(direct_dir, lib._MARKER_NAME))
    assert os.path.isfile(os.path.join(pgd_dir, lib._MARKER_NAME))
    with open(os.path.join(tmp_path, "metrics.json")) as f:
        summary = json.load(f)
    assert set(summary["per_arm"]) == {"pgd_seed0"}


def test_non_finite_loss_aborts_the_arm(tmp_path, monkeypatch):
    """An lr that blows the loss up stops the arm at that step instead of
    training on NaN weights to the end of the budget."""
    import _experiments_ffhq_icnn_regularizer_lib as lib

    monkeypatch.setattr(
        lib, "_gradient_penalty",
        lambda *a, **k: torch.tensor(float("nan")))
    exp = lib.FFHQICNNRegularizer(_make_args(best_val_ckpt=0))
    exp.device = torch.device("cpu")
    exp._pairs_cache[("train", 0)] = _FakeFFHQPairs(n=6, seed=10)
    exp._pairs_cache[("test", 0)] = _FakeFFHQPairs(n=2, seed=20)
    arm_dir = os.path.join(str(tmp_path), "pgd_seed0")
    os.makedirs(arm_dir, exist_ok=True)
    with pytest.raises(RuntimeError, match="non-finite loss at step 1/"):
        exp._train_one_arm("pgd", 0, arm_dir)


# ------------------------------------------------------------- dataset

_ARCHIVE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "third_party", "ehrhardt_icnn_primal_dual",
)


@pytest.mark.skipif(
    not os.path.isdir(_ARCHIVE),
    reason="Ehrhardt FFHQ archive not downloaded (Zenodo "
           "10.5281/zenodo.17426033)",
)
def test_ffhq_pairs_shapes_and_determinism():
    from lift.dataset.ffhq import FFHQPairs

    a = FFHQPairs(split="test", task="inpaint", degradation_seed=0,
                  max_images=2)
    assert a.clean.shape == (2, 1, 256, 256)
    assert a.noisy.shape == (2, 1, 256, 256)
    assert a.mask.shape == (1, 1, 256, 256)
    assert a.clean.min() >= 0.0 and a.clean.max() <= 1.0
    assert a.clean.device.type == "cpu"  # host-resident, not GPU preload
    # Their construction: mask*img + 0.03*noise, one fixed realization.
    resid = a.noisy - a.mask * a.clean
    assert resid.std() == pytest.approx(0.03, rel=0.05)
    b = FFHQPairs(split="test", task="inpaint", degradation_seed=0,
                  max_images=2)
    assert torch.equal(a.noisy, b.noisy)  # deterministic per seed
    c = FFHQPairs(split="test", task="inpaint", degradation_seed=1,
                  max_images=2)
    assert not torch.equal(a.noisy, c.noisy)
    d = FFHQPairs(split="test", task="snp", degradation_seed=0,
                  max_images=2)
    corrupted = (d.noisy != d.clean).float().mean()
    assert corrupted == pytest.approx(0.25, abs=0.02)


def test_dc_matched_negatives_match_clean_mean():
    """The DC-shortcut fix: each negative's mean equals its clean
    image's (up to the [0, 1] clamp), the mask/noise structure is kept,
    and the measurement passed alongside is not touched."""
    from lift.dataset.ffhq import dc_matched_negatives

    clean, noisy, mask = _fake_pairs(n=3, seed=11)
    clean = clean.clamp(0, 1)
    noisy_before = noisy.clone()
    neg = dc_matched_negatives(clean, noisy)
    assert neg.shape == noisy.shape
    assert torch.equal(noisy, noisy_before)
    assert neg.min() >= 0.0 and neg.max() <= 1.0
    # Zero-filled negatives sit below the clean mean; the fix removes
    # that gap to within the clamp's truncation.
    gap_before = (clean.mean(dim=(1, 2, 3)) - noisy.clamp(0, 1).mean(dim=(1, 2, 3)))
    gap_after = (clean.mean(dim=(1, 2, 3)) - neg.mean(dim=(1, 2, 3)))
    assert (gap_before > 0).all()
    assert (gap_after.abs() < 0.1 * gap_before).all()
    # A per-image constant shift: the pixel-to-pixel structure survives.
    diff = neg - noisy.clamp(0, 1)
    interior = ((noisy > 0.05) & (noisy < 0.95) & (neg > 0) & (neg < 1))
    for i in range(3):
        d = diff[i][interior[i]]
        assert d.std() < 1e-5 and d.mean() > 0


def test_trajectory_snapshots_written(tmp_path):
    """``snap_every > 0`` stores non-overwriting effective-weight
    trajectory snapshots (the landscape renderers' input: the frame's
    second direction is the top SVD loading across the snapshot union);
    the tensor set matches ``icnn_final.pt`` so the landscape reduction
    starts from the same prior state, and the atomic writes leave no
    tmp litter."""
    import glob

    from _experiments_ffhq_icnn_regularizer_lib import FFHQICNNRegularizer

    exp = FFHQICNNRegularizer(_make_args(best_val_ckpt=0, snap_every=1))
    exp.device = torch.device("cpu")
    exp._pairs_cache[("train", 0)] = _FakeFFHQPairs(n=6, seed=10)
    exp._pairs_cache[("test", 0)] = _FakeFFHQPairs(n=2, seed=20)
    arm_dir = os.path.join(str(tmp_path), "pgd_seed0")
    os.makedirs(arm_dir, exist_ok=True)
    exp._train_one_arm("pgd", 0, arm_dir)

    snaps = sorted(glob.glob(os.path.join(arm_dir, "snapshots", "snap_*.pt")))
    assert [os.path.basename(s) for s in snaps] == \
        ["snap_00001.pt", "snap_00002.pt"]
    fin = torch.load(os.path.join(arm_dir, "icnn_final.pt"),
                     map_location="cpu", weights_only=True)
    for s in snaps:
        ck = torch.load(s, map_location="cpu", weights_only=True)
        assert set(ck["prior_state"]) == set(fin)
        assert all(v.device.type == "cpu"
                   for v in ck["prior_state"].values())
    assert not glob.glob(os.path.join(arm_dir, "snapshots", "*.tmp"))
