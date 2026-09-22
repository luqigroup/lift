"""Emission-decay schedule tests for :class:`lift.models.HyperNetwork`.

The lift's emission ``W = psi(Theta_E h_phi(X) + b_h)`` fluctuates from
step to step because ``X`` is a fresh minibatch; that fluctuation buys
the escape from the attenuated region, and nothing ever quiets it, so
the iterate never settles (CP-Flow HEPMASS: a validation band through
the whole second half of training, last-iterate test NLL 17.35 / 16.10 /
21.28 against 16.42 / 16.10 / 16.94 under best-val selection; color
FFHQ: validation peaking inside the first tenth of the epoch and then
degrading 23.9 dB -> 11.7). ``emission_decay_*`` anneals the
batch-dependent component toward its running mean late in training.

These tests pin the three things that make the knob safe to ship:

  * the schedule holds at ``1``, ramps over the configured window, and
    reaches its floor exactly at the close -- with the window placeable
    anywhere in training, not only at its end;
  * the default is a BITWISE no-op: at equal seed a default-constructed
    emitter and a fully wired one emit identical tensors, register no
    extra state, and write an identical state dict (the conv-summary
    drop-in protocol);
  * as the schedule closes, the annealed emission tends to the
    deterministic-limit emission -- the emitted weights stop depending
    on the conditioning batch at all, which is the code-conditioned
    variant the paper reports as a limit.

CPU-fast: tiny widths, a handful of forwards.
"""

from __future__ import annotations

import pytest
import torch


def _emitter(seed: int = 0, **decay):
    """Tiny ICNN + hypernet pair; ``decay`` overrides the schedule knobs."""
    from lift.models import HyperNetwork, ICNN

    torch.manual_seed(seed)
    icnn = ICNN(input_size=6, hidden_dim=8, nlayers=3)
    torch.manual_seed(seed)
    return HyperNetwork(
        input_size=6,
        hidden_sizes=[8, 8],
        downstream_network=icnn,
        pos_param_names=None,
        **decay,
    )


def _gap(h, xa, xb) -> float:
    """Largest emitted-weight disagreement between two conditioning sets."""
    with torch.no_grad():
        a, b = h(xa), h(xb)
    return max(float((a[k] - b[k]).abs().max()) for k in a)


# ------------------------------------------------------------ the schedule
def test_schedule_reaches_its_floor_on_time():
    """Gain is 1 before the window, linear inside it, floor at the close."""
    h = _emitter(emission_decay_frac=0.4, emission_decay_floor=0.1)
    # Window is [end - frac, end] = [0.6, 1.0].
    for t in (0.0, 0.3, 0.59, 0.6):
        h.set_train_progress(t)
        assert h.emission_gain == pytest.approx(1.0), t
    h.set_train_progress(0.8)  # halfway through the window
    assert h.emission_gain == pytest.approx(0.55)
    h.set_train_progress(1.0)
    assert h.emission_gain == pytest.approx(0.1)
    # Past the close (a driver that overshoots its own step count) the
    # gain is HELD at the floor, never driven below it.
    h.set_train_progress(1.7)
    assert h.emission_gain == pytest.approx(0.1)


def test_schedule_window_is_placeable():
    """``emission_decay_end`` moves the window off the end of training.

    The color FFHQ trace peaks at step 500-750 of 8,748, so a window
    anchored at the end of the run would open long after the damage;
    the placement knob is what lets the close land near the peak.
    """
    h = _emitter(emission_decay_frac=0.4, emission_decay_end=0.6)
    h.set_train_progress(0.2)          # window opens at 0.2
    assert h.emission_gain == pytest.approx(1.0)
    h.set_train_progress(0.4)
    assert h.emission_gain == pytest.approx(0.5)
    h.set_train_progress(0.6)
    assert h.emission_gain == pytest.approx(0.0)
    h.set_train_progress(0.95)         # held at the floor to the end
    assert h.emission_gain == pytest.approx(0.0)


def test_schedule_knobs_validate_at_the_boundary():
    for bad in ({"emission_decay_frac": 1.4},
                {"emission_decay_end": -0.1},
                {"emission_decay_floor": 2.0},
                {"emission_decay_ema": 1.0},
                # window would have to open before training starts
                {"emission_decay_frac": 0.8, "emission_decay_end": 0.5}):
        with pytest.raises(ValueError):
            _emitter(**bad)


# ------------------------------------------------------- the no-op default
def test_default_is_bitwise_the_pre_schedule_emitter():
    """Default knobs emit BIT-IDENTICAL tensors to a bare construction.

    The comparison is against an emitter built without mentioning the
    schedule at all, at equal seed, over several training-mode forwards
    (so any running-mean update or blend would show up), then in eval.
    """
    bare = _emitter(seed=3)
    knobbed = _emitter(seed=3, emission_decay_frac=0.0,
                       emission_decay_end=1.0, emission_decay_floor=0.0,
                       emission_decay_ema=0.99)
    g = torch.Generator().manual_seed(11)
    for step in range(5):
        x = torch.randn(4, 6, generator=g)
        bare.set_train_progress(step / 5.0)
        knobbed.set_train_progress(step / 5.0)
        a, b = bare(x), knobbed(x)
        assert set(a) == set(b)
        for name in a:
            assert torch.equal(a[name], b[name]), (
                f"step {step}, {name}: default emission is not bitwise "
                f"unchanged (max |diff| "
                f"{(a[name] - b[name]).abs().max().item():.3e})"
            )
    bare.eval()
    knobbed.eval()
    x = torch.randn(4, 6, generator=g)
    for name, t in bare(x).items():
        assert torch.equal(t, knobbed(x)[name])


def test_default_registers_no_extra_state():
    """Off = no buffer, no state-dict key: old checkpoints load strict."""
    bare = _emitter(seed=3)
    knobbed = _emitter(seed=3, emission_decay_frac=0.0)
    assert set(bare.state_dict()) == set(knobbed.state_dict())
    assert not any("emission" in k or "pooled_ema" in k
                   for k in knobbed.state_dict())
    assert not any(n == "_pooled_ema" for n, _ in knobbed.named_buffers())
    # And a decay run DOES carry its schedule state, so a snapshot
    # reloaded outside the trainer evaluates the annealed function
    # rather than silently reverting to full emission strength.
    on = _emitter(seed=3, emission_decay_frac=0.5)
    keys = set(on.state_dict()) - set(bare.state_dict())
    assert keys == {"_emission_gain", "_pooled_ema", "_pooled_ema_ready"}
    on.set_train_progress(1.0)
    on(torch.randn(4, 6))
    reloaded = _emitter(seed=3, emission_decay_frac=0.5)
    reloaded.load_state_dict(on.state_dict())
    assert reloaded.emission_gain == pytest.approx(on.emission_gain)
    # A pre-schedule state dict still loads strict into a default emitter.
    _emitter(seed=3).load_state_dict(bare.state_dict())


def test_run_is_bitwise_unchanged_until_the_window_opens():
    """Gain 1 means UNTOUCHED, not "multiplied by one".

    So a decay run and its baseline are the same run up to the step the
    window opens, which is what makes the comparison an attribution.
    """
    off = _emitter(seed=5)
    on = _emitter(seed=5, emission_decay_frac=0.3, emission_decay_end=1.0)
    g = torch.Generator().manual_seed(7)
    for step in range(6):                       # t_frac 0.0 .. 0.5 < 0.7
        x = torch.randn(4, 6, generator=g)
        t = step / 12.0
        off.set_train_progress(t)
        on.set_train_progress(t)
        a, b = off(x), on(x)
        for name in a:
            assert torch.equal(a[name], b[name]), (step, name)
    # ... and the running mean was warming the whole time, so the blend
    # has something to close onto the moment the window opens.
    assert float(on._pooled_ema_ready) == 1.0


# ------------------------------------------------- the deterministic limit
def test_closed_schedule_tends_to_the_deterministic_emission():
    """At the floor the emission stops depending on the batch.

    Two different conditioning batches must emit the SAME weights once
    the schedule has closed (they emit different ones while it is open),
    and no gradient reaches the DeepSets summary encoder -- the
    code-conditioned limit, with the readouts still training.
    """
    h = _emitter(seed=1, emission_decay_frac=0.5, emission_decay_floor=0.0)
    g = torch.Generator().manual_seed(2)
    # Warm the running mean over the open half of the schedule.
    for step in range(20):
        h.set_train_progress(step / 40.0)
        h(torch.randn(8, 6, generator=g))

    xa = torch.randn(8, 6, generator=g)
    xb = torch.randn(8, 6, generator=g) + 3.0

    h.set_train_progress(0.5)                    # window just opening
    assert _gap(h, xa, xb) > 0.0, "emission is batch-independent while open"

    h.set_train_progress(1.0)                    # closed
    assert h.emission_gain == pytest.approx(0.0)
    closed = _gap(h, xa, xb)
    assert closed == 0.0, (
        f"emission still depends on the conditioning batch at the floor "
        f"(max |diff| {closed:.3e})"
    )
    # The summary encoder is out of the graph; the readouts are not.
    h.zero_grad(set_to_none=True)
    sum(v.pow(2).sum() for v in h(xa).values()).backward()
    enc = [p.grad for p in h.inner_net.parameters()]
    assert all(gr is None or float(gr.abs().max()) == 0.0 for gr in enc)
    head = h.weight_predictors[0]
    assert head.weight.grad is not None
    assert float(head.weight.grad.abs().max()) > 0.0


def test_annealed_emission_approaches_the_deterministic_one():
    """The approach is CONTINUOUS: the gap shrinks as the gain falls."""
    h = _emitter(seed=4, emission_decay_frac=0.5, emission_decay_floor=0.0)
    g = torch.Generator().manual_seed(9)
    for step in range(20):
        h.set_train_progress(step / 40.0)
        h(torch.randn(8, 6, generator=g))
    xa = torch.randn(8, 6, generator=g)
    xb = torch.randn(8, 6, generator=g) + 3.0

    h.eval()  # freeze the running mean so only the gain moves
    gaps = []
    for t in (0.5, 0.625, 0.75, 0.875, 1.0):
        h.set_train_progress(t)
        gaps.append(_gap(h, xa, xb))
    assert all(b <= a + 1e-9 for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] == 0.0 < gaps[0]


def test_pool_scale_norm_survives_the_blend():
    """With ``pool_scale`` on, the decayed summary keeps its pinned norm.

    ``pool_scale=1.0`` is how the batch-conditioned HEPMASS runs are
    stabilised; a decay that let the summary's magnitude collapse toward
    the mean would give up that stabilisation while claiming only to
    quiet the fluctuation.
    """
    h = _emitter(seed=6, pool_scale=1.0, emission_decay_frac=1.0,
                 emission_decay_floor=0.0)
    g = torch.Generator().manual_seed(3)
    for step in range(10):
        h.set_train_progress(step / 10.0)
        h(torch.randn(8, 6, generator=g))
    # Walk the same path the readouts see: mean-pool, pin, then blend.
    with torch.no_grad():
        for t in (0.4, 0.8, 1.0):
            h.set_train_progress(t)
            z = h.inner_net(
                torch.randn(8, 6, generator=g),
            ).mean(0, keepdim=True)
            z = h.pool_scale * z / (z.norm() + 1e-8)
            assert float(h._decay_pooled(z).norm()) == pytest.approx(
                1.0, abs=1e-5,
            ), t


# --------------------------------------------------------- driver wiring
def test_ffhq_backends_accept_the_progress_call():
    """All three FFHQ adapters expose ``set_train_progress``.

    The trainer calls it unconditionally every step, so a missing method
    on pgd or direct would crash those arms rather than no-op.
    """
    import os
    import sys

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts"),
    )
    ffhq = pytest.importorskip("_experiments_ffhq_icnn_regularizer_lib")
    for cls in (ffhq.PGDBackend, ffhq.DirectBackend, ffhq.HypernetBackend):
        assert hasattr(cls, "set_train_progress"), cls.__name__


def test_ffhq_arm_trains_through_the_schedule(tmp_path):
    """End-to-end on a two-step FFHQ arm: the driver closes the window.

    Exercises the wiring the color run depends on -- the config key
    reaching :class:`HyperNetwork`, the per-step
    ``adapter.set_train_progress``, the gain landing in ``metrics.json``
    -- and pins that the decay-off arm's metrics keep exactly the keys
    they had before the knob existed.
    """
    import json
    import os

    from tests.test_ffhq_ehrhardt import _run_arm

    on = _run_arm(tmp_path / "on", "hypernet", best_val_ckpt=0,
                  save_emitter=0, emission_decay_frac=1.0,
                  emission_decay_end=1.0, emission_decay_floor=0.0,
                  emission_decay_ema=0.9)
    off = _run_arm(tmp_path / "off", "hypernet", best_val_ckpt=0,
                   save_emitter=0)
    with open(os.path.join(on, "metrics.json")) as f:
        m_on = json.load(f)
    with open(os.path.join(off, "metrics.json")) as f:
        m_off = json.load(f)

    gains = m_on["history"]["emission_gain"]
    assert gains, "the decay run recorded no gain trace"
    assert gains[-1] == pytest.approx(0.0), gains
    assert gains == sorted(gains, reverse=True), gains
    assert "emission_gain" not in m_off["history"]
    # The arm still produces every deployment artifact.
    for f_name in ("icnn_final.pt", "metrics.json", "RESULTS_COMPLETE.json"):
        assert os.path.isfile(os.path.join(on, f_name)), f_name


def test_cpflow_hypernet_walker_finds_every_block():
    """``_hypernets`` reaches the emitter in every lifted CP-Flow block.

    The HEPMASS trainer drives the schedule through this walker, so a
    block it misses is a block whose emission never decays.
    """
    import os
    import sys

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts"),
    )
    lib = pytest.importorskip("_experiments_cpflow_uci_lib")
    from lift.models import HyperNetwork

    cfg = lib.CPFlowConfig(D=4, dimh=4, nhidden=2, nblocks=3, m1=2)
    flow = lib.build_cpflow_hypernet(
        cfg, hyper_hidden=[8, 8], seed=0, condition_on="batch",
        pool_scale=1.0, emission_decay_frac=0.25,
    )
    hns = lib._hypernets(flow)
    assert len(hns) == cfg.nblocks
    assert all(isinstance(h, HyperNetwork) for h in hns)
    for h in hns:
        h.set_train_progress(1.0)
        assert h.emission_gain == pytest.approx(0.0)
    # And the direct backend has none, so the trainer's loop is empty.
    assert lib._hypernets(lib.build_cpflow_direct(cfg, seed=0)) == []
