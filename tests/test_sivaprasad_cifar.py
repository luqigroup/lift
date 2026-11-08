"""The CIFAR-10 classifier comparison with the rule of Sivaprasad et al. (2021).

Their post-update rule as a construction (``exppost``: ``w <- exp(w - 5)`` on
the negative constrained coordinates, identity elsewhere, bit-equal to
``pgd`` off the crossed set); ELU as an ICNN activation that keeps the
network convex; the official test split under the training constants when
it is on disk; the selection-epoch indexing; the lift's step-zero gates;
exact per-epoch resume; and the saved best and final weights. The cell tests
train on a small synthetic split of MNIST width, so they need no download.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from projorg import checkpointsdir, gitdir                    # noqa: E402

from lift.dataset.flat_images import (                       # noqa: E402
    FlatImageSplit, default_root, get_spec, image_test,
)
from lift.models import icnn_pos_param_names                 # noqa: E402
from lift.models.icnn import ICNN                            # noqa: E402

import _experiments_sivaprasad_cifar_lib as lib              # noqa: E402
from _experiments_sivaprasad_cifar_lib import (              # noqa: E402
    CONSTRUCTIONS, EXPPOST_EPS, EXPPOST_EPS_MATCHED, build_arm,
    check_matched_draw, emit, _emitted_constrained, _occ_profile,
    exppost_eps, projects, select_at_epoch, train_cell,
)

CPU = torch.device("cpu")
_ROOT = default_root(mkdir=False)
_HAS_MNIST_TEST = os.path.isdir(os.path.join(_ROOT, "mnist", "MNIST", "raw"))
_HAS_CIFAR_TEST = os.path.isfile(os.path.join(
    _ROOT, "cifar10", "cifar-10-batches-py", "test_batch"))


def _small(con: str, seed: int = 0, activation: str = "relu",
           init_mode: str = "hk", width: int = 48):
    spec = get_spec("mnist")
    return build_arm(con, spec, width, 3, init_mode, [16, 16], False, False,
                     "elementwise", activation, CPU, True, init_seed=seed)


def test_exppost_is_a_construction_that_projects():
    assert "exppost" in CONSTRUCTIONS and "exppost_matched" in CONSTRUCTIONS
    assert projects("exppost") and projects("exppost_matched") and projects("pgd")
    assert not projects("direct") and not projects("nonconvex")
    assert EXPPOST_EPS == 5.0 and EXPPOST_EPS_MATCHED == 8.80
    assert exppost_eps("exppost") == 5.0
    assert exppost_eps("exppost_matched") == 8.80
    assert exppost_eps("pgd") is None and exppost_eps("lift") is None


def test_matched_eps_puts_the_return_point_at_the_hk_median_at_fan_in_800():
    """``exp(-8.80)`` against the closed-form median of the ``hk`` draw at
    N = 800 and against the realized draw."""
    n = 800
    d = 6.0 * (math.pi - 1.0) + (n - 1) * (3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0)
    mu_w, var_w = math.sqrt(6.0 * math.pi / (n * d)), 1.0 / n
    median = mu_w * mu_w / math.sqrt(var_w + mu_w * mu_w)
    assert median == pytest.approx(1.515e-4, rel=1e-2)
    assert math.exp(-EXPPOST_EPS_MATCHED) == pytest.approx(median, rel=1e-2)
    assert math.exp(-EXPPOST_EPS) > 40 * median
    torch.manual_seed(0)
    net = ICNN(3072, hidden_dim=800, nlayers=3, activation="elu",
               strong_convexity=0.0, pos_constraint_mode="clamp",
               output_size=10, use_skip=False, output_bias=True)
    from _experiments_hypernet_vs_direct_1d_lib import apply_init_mode
    apply_init_mode(net, icnn_pos_param_names(3), "hk", latent=False)
    w = net.z_layers[0].weight.detach().double()
    assert float(w.median()) == pytest.approx(median, rel=2e-2)


def test_exppost_matched_differs_from_exppost_only_in_the_return_point():
    """Same crossed set, ``exp(w - 8.80)`` instead of ``exp(w - 5)``."""
    spec = get_spec("mnist")
    g = torch.Generator().manual_seed(5)
    x = torch.randn(64, spec.width, generator=g)
    y = torch.randint(0, 10, (64,), generator=g)
    out = {}
    for con in ("exppost", "exppost_matched"):
        arm, tpl, _ = _small(con)
        assert arm.exp_eps == exppost_eps(con)
        opt = torch.optim.Adam(arm.parameters(), lr=1e-3)
        loss = torch.nn.functional.cross_entropy(
            torch.func.functional_call(tpl, emit(arm, x), (x,)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pre = _emitted_constrained(emit(arm, x)).clone()
        arm.project_positive()
        out[con] = (pre, _emitted_constrained(emit(arm, x)).clone())
    pre_a, w_a = out["exppost"]
    pre_b, w_b = out["exppost_matched"]
    assert torch.equal(pre_a, pre_b)
    crossed = pre_a < 0
    assert int(crossed.sum()) > 0
    assert torch.allclose(w_b[crossed], torch.exp(pre_b[crossed] - 8.80))
    assert torch.allclose(w_a[crossed], torch.exp(pre_a[crossed] - 5.0))
    assert torch.equal(w_a[~crossed], w_b[~crossed])


def test_matched_draw_gate_fails_a_15_percent_mismatch_and_passes_the_lift():
    """A 15% scale on the direct ``hk`` draw at 3 x 800 fails on both the
    median and an occupancy; the lift at a small width passes."""
    arm, _, _ = build_arm("direct", get_spec("cifar10"), 800, 3, "hk", [8],
                          False, False, "off", "elu", CPU, True, init_seed=0)
    w = _emitted_constrained(emit(arm, torch.zeros(1, 3072)))
    ref = _occ_profile(w)
    assert check_matched_draw(ref, ref, 0.05, 0.01)["median_rel_gap"] == 0.0
    for scale in (1.15, 0.85):
        bad = _occ_profile(w * scale)
        with pytest.raises(RuntimeError, match="median"):
            check_matched_draw(bad, ref, 0.05, 0.01)
        with pytest.raises(RuntimeError, match="occupancy"):
            check_matched_draw(bad, ref, 1.0, 0.01)
        assert max(abs(bad["occupancy"][t] - ref["occupancy"][t])
                   for t in ref["occupancy"]) > 0.02
    x = torch.randn(64, 784)
    direct, _, _ = _small("direct")
    lift, _, _ = _small("lift")
    info = check_matched_draw(
        _occ_profile(_emitted_constrained(emit(lift, x))),
        _occ_profile(_emitted_constrained(emit(direct, x))), 0.05, 0.01)
    assert info["median_rel_gap"] < 0.05
    assert info["worst_occupancy_abs_gap"] < 0.01


def test_exppost_maps_negative_constrained_weights_to_exp_w_minus_eps():
    """``exp(w - 5)`` where ``w < 0``, identity on ``w >= 0`` and on every
    unconstrained tensor."""
    arm, _, pos = _small("exppost")
    params = dict(arm._inner.named_parameters())
    with torch.no_grad():
        for name in pos:
            p = params[name]
            mask = torch.rand_like(p) < 0.3
            p[mask] = -torch.rand(int(mask.sum())) * 3.0
            p[~mask] = p[~mask].abs() + 1e-6
    before = {n: p.detach().clone() for n, p in params.items()}
    arm.project_positive()
    for name, p in params.items():
        if name in pos:
            neg = before[name] < 0
            assert torch.allclose(p[neg], torch.exp(before[name][neg] - 5.0))
            assert torch.equal(p[~neg], before[name][~neg])
            assert float(p.min()) > 0.0
            assert float(p[neg].max()) <= math.exp(-5.0) + 1e-9
        else:
            assert torch.equal(p, before[name])


def test_exppost_and_pgd_differ_exactly_on_the_crossed_coordinates():
    """One Adam step from the same ``hk`` draw: identical before the map,
    then ``pgd`` at exact zero and ``exppost`` in ``(0, e^-5]`` on every
    coordinate that went negative, bit-equal on every other one."""
    spec = get_spec("mnist")
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, spec.width, generator=g)
    y = torch.randint(0, 10, (64,), generator=g)
    out = {}
    for con in ("pgd", "exppost"):
        arm, tpl, _ = _small(con)
        opt = torch.optim.Adam(arm.parameters(), lr=1e-3)
        loss = torch.nn.functional.cross_entropy(
            torch.func.functional_call(tpl, emit(arm, x), (x,)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pre = _emitted_constrained(emit(arm, x)).clone()
        arm.project_positive()
        out[con] = (pre, _emitted_constrained(emit(arm, x)).clone())
    pre_p, w_p = out["pgd"]
    pre_e, w_e = out["exppost"]
    assert torch.equal(pre_p, pre_e)
    crossed = pre_e < 0
    assert int(crossed.sum()) > 0, "no coordinate crossed; the test has no power"
    assert torch.equal(w_p[crossed], torch.zeros_like(w_p[crossed]))
    assert float(w_e[crossed].min()) > 0.0
    assert float(w_e[crossed].max()) <= math.exp(-5.0) + 1e-9
    assert torch.equal(w_p[~crossed], w_e[~crossed])


def test_exppost_is_mapped_once_at_construction_under_a_signed_init():
    """``he`` draws signed raw weights; the construction is mapped once
    before its first forward, so step zero is already in the orthant."""
    arm, _, pos = _small("exppost", init_mode="he")
    theta = emit(arm, torch.randn(4, 784))
    for name in pos:
        assert float(theta[name].detach().min()) > 0.0
    # Under ``hk`` the draw is positive and the map is a no-op.
    a, _, _ = _small("exppost")
    b, _, _ = _small("pgd")
    assert torch.equal(_emitted_constrained(emit(a, torch.zeros(1, 784))),
                       _emitted_constrained(emit(b, torch.zeros(1, 784))))


def test_elu_classifier_is_convex_in_its_input():
    """Midpoint convexity of every logit under ELU."""
    torch.manual_seed(0)
    net = ICNN(32, hidden_dim=32, nlayers=4, activation="elu",
               strong_convexity=0.0, pos_constraint_mode="clamp",
               output_size=5, use_skip=False, output_bias=True)
    net.clamp_weights()
    assert isinstance(net.activation, torch.nn.ELU)
    a, b = torch.randn(512, 32), torch.randn(512, 32)
    mid, ends = net(0.5 * a + 0.5 * b), 0.5 * net(a) + 0.5 * net(b)
    assert bool((mid <= ends + 1e-4).all())
    t = torch.linspace(-6, 6, 2001)
    f = torch.nn.functional.elu(t)
    d = torch.diff(f)
    assert bool((d >= 0).all())                # non-decreasing
    assert bool((torch.diff(d) >= -1e-6).all())  # convex


def test_double_softplus_check_rebuilds_the_same_activation():
    from torch import nn
    assert lib._activation_name(nn.ELU()) == "elu"
    assert lib._activation_name(nn.ReLU()) == "relu"
    assert lib._activation_name(nn.Softplus()) == "softplus"
    with pytest.raises(ValueError):
        lib._activation_name(nn.Tanh())


@pytest.mark.skipif(not _HAS_MNIST_TEST, reason="MNIST not on disk")
def test_mnist_test_split_has_the_official_shape_and_the_training_constants():
    test, spec = image_test("mnist", download=False)
    assert tuple(test.x.shape) == (10000, 784) and tuple(test.y.shape) == (10000,)
    assert int(test.y.min()) == 0 and int(test.y.max()) == 9
    from torchvision import datasets as tv
    raw = tv.MNIST(os.path.join(_ROOT, "mnist"), train=False, download=False)
    v = float(raw.data[0].reshape(-1)[200]) / 255.0
    assert float(test.x[0, 200]) == pytest.approx(
        (v - spec.mean[0]) / spec.std[0], abs=1e-6)
    assert int(test.y[0]) == int(raw.targets[0])


@pytest.mark.skipif(not _HAS_CIFAR_TEST, reason="CIFAR-10 test batch not on disk")
def test_cifar10_test_split_has_the_official_shape_and_the_training_constants():
    test, spec = image_test("cifar10", download=False)
    assert tuple(test.x.shape) == (10000, 3072) and tuple(test.y.shape) == (10000,)
    assert int(test.y.min()) == 0 and int(test.y.max()) == 9
    from torchvision import datasets as tv
    raw = tv.CIFAR10(os.path.join(_ROOT, "cifar10"), train=False,
                     download=False)
    # Channel-first raveling: pixel (0, 0) of channel 1 sits at index 1024.
    v = float(raw.data[0, 0, 0, 1]) / 255.0
    assert float(test.x[0, 1024]) == pytest.approx(
        (v - spec.mean[1]) / spec.std[1], abs=1e-6)
    assert int(test.y[0]) == int(raw.targets[0])


def test_select_at_epoch_reads_the_full_split_entry_the_best_epoch_names():
    curve = [
        {"epoch": 0, "full_split": True, "val_acc": 0.10, "test_acc": 0.11},
        {"epoch": 1, "full_split": False, "val_acc": 0.30, "test_acc": 0.99},
        {"epoch": 1, "full_split": True, "val_acc": 0.40, "test_acc": 0.41},
        {"epoch": 2, "full_split": True, "val_acc": 0.35, "test_acc": 0.90},
    ]
    best = max((e for e in curve if e["full_split"]), key=lambda e: e["val_acc"])
    assert best["epoch"] == 1
    assert select_at_epoch(curve, best["epoch"], "test_acc") == 0.41
    assert select_at_epoch(curve, 0, "test_acc") == 0.11
    with pytest.raises(KeyError):
        select_at_epoch(curve, 7, "test_acc")
    with pytest.raises(KeyError):
        select_at_epoch(curve + [curve[-1]], 2, "test_acc")
    assert math.isnan(select_at_epoch(curve, 2, "absent"))


def _synthetic(n: int, seed: int) -> FlatImageSplit:
    """``n`` points of MNIST width whose labels a linear map decides."""
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(n, 784, generator=g)
    w = torch.randn(784, 10, generator=torch.Generator().manual_seed(99))
    return FlatImageSplit(x, (x @ w).argmax(dim=1))


@pytest.fixture
def synthetic_data(monkeypatch):
    """Replace the image loaders the lib caches with a small synthetic set."""
    spec = get_spec("mnist")

    def _split(dataset, seed, train_split):
        return _synthetic(640, 1000 + int(seed)), _synthetic(256, 2000), spec

    monkeypatch.setattr(lib, "_cached_split", _split)
    monkeypatch.setattr(lib, "_cached_test", lambda dataset: _synthetic(256, 3000))


def _args(**over) -> SimpleNamespace:
    with open(os.path.join(gitdir(), "configs",
                           "experiments_sivaprasad_cifar.json")) as f:
        cfg = json.load(f)
    cfg.update(dict(
        eval_test=0, patience=0, ce0_match_rtol=0.0, max_steps_per_epoch=0,
        lift_cond_spread_batches=0, checkpoint_every_epochs=0, save_weights=0,
        hidden_dim=32, lr=1e-3, data_on_device=0, n_val_probe=256,
        val_every=0, log_every=5, eval_chunk=4096, hyper_hidden_sizes=[16, 16],
        experiment=f"pytest_sivaprasad_{os.getpid()}",
    ))
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _cleanup(args) -> None:
    d = checkpointsdir(args.experiment, mkdir=False)
    if os.path.isdir(d):
        shutil.rmtree(d)


def test_train_cell_with_every_knob_off_records_no_test_columns(synthetic_data):
    args = _args(epochs=2, max_steps_per_epoch=3)
    rec = train_cell("mnist", "direct", "hk", 3, "relu", 0, args, CPU)
    assert rec["eval_test"] == 0 and "final_test_acc" not in rec
    assert rec["patience"] == 0 and rec["early_stopped"] is False
    assert rec["stopped_epoch"] == 2 and len(rec["epoch_train_loss"]) == 2
    assert all("test_acc" not in e for e in rec["val_curve"])
    assert rec["exp_eps"] is None and rec["ce_step0_direct_ref"] is None
    assert rec["draw_match"] is None and rec["save_weights"] == 0
    assert math.isnan(rec["first_layer_pre_var_ratio"])


def test_train_cell_records_test_columns_and_stops_on_validation_accuracy(
        synthetic_data):
    args = _args(epochs=6, patience=1, eval_test=1, max_steps_per_epoch=2,
                 lr=0.3)                      # a large rate so accuracy wobbles
    rec = train_cell("mnist", "exppost", "hk", 3, "relu", 0, args, CPU)
    assert rec["eval_test"] == 1 and rec["exp_eps"] == 5.0
    full = [e for e in rec["val_curve"] if e["full_split"]]
    assert [e["epoch"] for e in full] == list(range(rec["stopped_epoch"] + 1))
    assert all("test_acc" in e and "test_loss" in e for e in full)
    best = max(full, key=lambda e: e["val_acc"])
    first_best = min(e["epoch"] for e in full if e["val_acc"] == best["val_acc"])
    assert rec["best_val_acc_epoch"] == first_best           # ties: earliest
    assert rec["test_acc_at_best_val_acc"] == select_at_epoch(
        rec["val_curve"], rec["best_val_acc_epoch"], "test_acc")
    assert rec["final_test_acc"] == full[-1]["test_acc"]
    if rec["early_stopped"]:
        assert rec["stopped_epoch"] - rec["best_val_acc_epoch"] == 1
        assert rec["stopped_epoch"] < 6
    else:
        assert rec["stopped_epoch"] == 6
    assert rec["shoulder_late"] == {} or rec["shoulder_late_started"]


def test_lift_step_zero_gate_fires_and_leaves_the_lift_unchanged(synthetic_data):
    base = dict(epochs=1, eval_test=1, max_steps_per_epoch=1,
                lift_cond_spread_batches=2)
    strict = _args(ce0_match_rtol=1e-6, **base)
    with pytest.raises(RuntimeError, match="step-zero pairing FAILED"):
        train_cell("mnist", "lift", "hk", 3, "relu", 0, strict, CPU)
    loose = _args(ce0_match_rtol=1.0, **base)
    rec = train_cell("mnist", "lift", "hk", 3, "relu", 0, loose, CPU)
    off = _args(**base)
    ref = train_cell("mnist", "lift", "hk", 3, "relu", 0, off, CPU)
    # The reference build reseeds; the lift's own draw must not move.
    assert rec["sha_init"] == ref["sha_init"]
    assert rec["traj_sha"] == ref["traj_sha"]
    assert rec["ce_step0_direct_ref"] is not None
    assert rec["ce_step0_rel_gap"] == pytest.approx(
        abs(rec["ce_step0"] - rec["ce_step0_direct_ref"])
        / rec["ce_step0_direct_ref"])
    d = train_cell("mnist", "direct", "hk", 3, "relu", 0, off, CPU)
    assert rec["ce_step0_direct_ref"] == pytest.approx(d["ce_step0"], rel=1e-6)
    sp = rec["final_test_acc_cond_spread"]
    assert sp["n"] == 2 and sp["min"] <= sp["mean"] <= sp["max"]
    dm = rec["draw_match"]
    assert dm["median_rel_gap"] < 0.05 and dm["worst_occupancy_abs_gap"] < 0.01
    assert dm["median_rel_gap"] == pytest.approx(
        abs(rec["step0_profile"]["median"] - d["step0_profile"]["median"])
        / d["step0_profile"]["median"], rel=1e-6)
    assert rec["first_layer_pre_var_direct_ref"] == pytest.approx(
        d["signal_propagation"][0]["pre_var"], rel=1e-6)
    assert rec["first_layer_pre_var_ratio"] == pytest.approx(
        rec["first_layer_pre_var"] / d["first_layer_pre_var"], rel=1e-6)
    assert rec["first_layer_pre_var_ratio"] > 1.0
    # The tight gate fires on a draw that is not the direct one.
    bad = _args(ce0_match_rtol=1.0, draw_match_median_rtol=1e-9,
                draw_match_occ_atol=1e-9, **base)
    with pytest.raises(RuntimeError, match="step-zero draw mismatch"):
        train_cell("mnist", "lift", "hk", 3, "relu", 0, bad, CPU)


def test_per_epoch_checkpoint_resumes_exactly(synthetic_data, monkeypatch):
    """Kill after the first epoch's checkpoint; the resumed cell reproduces an
    uninterrupted one, number for number."""
    common = dict(epochs=2, eval_test=1, max_steps_per_epoch=4,
                  checkpoint_every_epochs=1, patience=0)
    straight = _args(**common)
    interrupted = _args(**common)
    interrupted.experiment += "_resume"
    try:
        ref = train_cell("mnist", "pgd", "hk", 3, "relu", 0, straight, CPU)
        real_save = lib._save_resume

        def _save_then_die(path, state):
            real_save(path, state)
            if state["epoch"] == 1:
                raise KeyboardInterrupt("simulated preemption")

        monkeypatch.setattr(lib, "_save_resume", _save_then_die)
        with pytest.raises(KeyboardInterrupt):
            train_cell("mnist", "pgd", "hk", 3, "relu", 0, interrupted, CPU)
        path = lib._resume_path(interrupted, lib.cell_key(
            "mnist", "pgd", "hk", 3, "relu", 0))
        assert os.path.exists(path)
        monkeypatch.setattr(lib, "_save_resume", real_save)
        rec = train_cell("mnist", "pgd", "hk", 3, "relu", 0, interrupted, CPU)
        assert not os.path.exists(path)              # removed on completion
        assert rec["resumed_from_epoch"] == 1 and ref["resumed_from_epoch"] == 0
        skip = {"wall_s", "split_decode_s", "emission_wall_ms_per_step",
                "peak_gpu_mib", "peak_gpu_reserved_mib", "resumed_from_epoch"}
        for k in ref:
            if k in skip:
                continue
            assert _same(ref[k], rec[k]), k
    finally:
        _cleanup(straight)
        _cleanup(interrupted)


def test_resume_after_the_stopping_checkpoint_keeps_the_stopped_epoch(
        synthetic_data, monkeypatch):
    """A kill between the stopping epoch's checkpoint and the record resumes
    into a finished cell with the same stopped epoch."""
    common = dict(epochs=8, eval_test=1, max_steps_per_epoch=2, patience=1,
                  checkpoint_every_epochs=1, lr=0.3)
    straight = _args(**common)
    interrupted = _args(**common)
    interrupted.experiment += "_stopresume"
    try:
        ref = train_cell("mnist", "exppost", "hk", 3, "relu", 0, straight, CPU)
        assert ref["early_stopped"] and ref["stopped_epoch"] < 8
        stop = ref["stopped_epoch"]
        real_save = lib._save_resume

        def _save_then_die(path, state):
            real_save(path, state)
            if state["epoch"] == stop:
                raise KeyboardInterrupt("simulated kill after the checkpoint")

        monkeypatch.setattr(lib, "_save_resume", _save_then_die)
        with pytest.raises(KeyboardInterrupt):
            train_cell("mnist", "exppost", "hk", 3, "relu", 0, interrupted, CPU)
        monkeypatch.setattr(lib, "_save_resume", real_save)
        rec = train_cell("mnist", "exppost", "hk", 3, "relu", 0, interrupted, CPU)
        assert rec["resumed_from_epoch"] == stop
        assert rec["stopped_epoch"] == stop and rec["early_stopped"]
        assert rec["best_val_acc_epoch"] == ref["best_val_acc_epoch"]
        assert rec["test_acc_at_best_val_acc"] == ref["test_acc_at_best_val_acc"]
        assert len(rec["epoch_train_loss"]) == len(ref["epoch_train_loss"])
    finally:
        _cleanup(straight)
        _cleanup(interrupted)


def test_best_and_final_weights_are_saved_and_a_finished_cell_is_not_overwritten(
        synthetic_data):
    args = _args(epochs=3, eval_test=1, max_steps_per_epoch=2, patience=0,
                 save_weights=1, lift_cond_spread_batches=1)
    try:
        rec = train_cell("mnist", "lift", "hk", 3, "relu", 0, args, CPU)
        assert rec["save_weights"] == 1
        best = torch.load(rec["weights_best"], map_location="cpu",
                          weights_only=False)
        final = torch.load(rec["weights_final"], map_location="cpu",
                           weights_only=False)
        assert best["epoch"] == rec["best_val_acc_epoch"]
        assert best["val_acc"] == pytest.approx(rec["best_val_acc"])
        assert final["epoch"] == rec["stopped_epoch"]
        assert final["val_acc"] == pytest.approx(rec["final_val_acc"])
        # The deployed ICNN of the lift is the emission at ``x_eval``.
        assert set(final["theta_eval"]) == {
            n for n, _ in ICNN(784, 32, 3, activation="relu",
                               strong_convexity=0.0, output_size=10,
                               use_skip=False, output_bias=True,
                               ).named_parameters()}
        assert set(best["arm"]) == set(final["arm"])
        with pytest.raises(RuntimeError, match="finished weights exist"):
            train_cell("mnist", "lift", "hk", 3, "relu", 0, args, CPU)
    finally:
        _cleanup(args)


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(p, q) for p, q in zip(a, b))
    return a == b
