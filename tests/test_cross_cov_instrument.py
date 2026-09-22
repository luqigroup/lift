r"""Regression tests for the cross-covariance instrument.

The paper's certificate reports

.. math::

    \Sigma_{\rm slack} = \mathbb{E}\bigl[\,
        \delta\widetilde\theta\;\delta g^\top \bigr],

the covariance BETWEEN the emitted pre-positivity iterate
``theta-tilde = Theta_E h_phi(X) + b_h`` and the loss gradient on the
readout bias ``b_h``. The shipped instrument measured something else:
the CP-Flow snapshot recorder passed the mean-removed bias GRADIENT as
its second argument, so the accumulator formed ``Cov(g)`` --- a
symmetric PSD AUTO-covariance whose trace is a sum of variances and is
therefore positive by construction. Every persisted CP-Flow HEPMASS
snapshot is symmetric to float32 roundoff for exactly that reason.

The contrast arm was instrumental in the same way: the recorder
short-circuited on ``is_pgd or is_direct`` and wrote a literal zero
matrix, so "the unlifted arm has no slack" was a branch, not a reading.

These tests pin the corrected instrument. They are written to fail on
MUTANTS, not merely on the shipped line -- each of the following was
introduced into a scratch copy and each is now caught:

===================================================  =================
mutant                                               caught by
===================================================  =================
alias guard deleted                                  the guard tests
slots swapped (``gc.T @ jc``)                        ``..._slot_order``
iterate := the bias PARAMETER ``b_h``                ``..._is_caught``
hard-coded zero for the unlifted arm                 ``..._every_arm``
``_frozen_emission_ema`` neutered                    ``..._ema_frozen``
one-emission-per-probe assertion removed             ``..._one_emission``
scalars read off the JL projection                   ``..._exact_not_projected``
divisor ``1/B`` instead of ``1/(B-1)``               ``..._normalisation``
probe RNG shared with training                       ``..._pure_observer``
===================================================  =================

CPU-fast: tiny widths, a handful of forwards.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"),
)

from lift.objectives.cross_cov_taps import (  # noqa: E402
    cross_cov_taps,
    flat_bias_dim,
    hypernets_of,
    make_probe_rng,
    record_cross_cov_snapshot,
    tap_iterate_source,
)
from lift.objectives.diagnostics import (  # noqa: E402
    ASYMMETRY_FLOOR,
    CROSS_COV_SCHEMA_VERSION,
    CrossCovAccumulator,
    CrossCovDegenerateInputs,
    CrossCovNonFinite,
    CrossCovSymmetricEstimate,
    MoreauProbe,
)


# --------------------------------------------------- the accumulator itself
def _known_pair():
    """Iterate / gradient sequences whose cross-covariance is exact.

    Two probes, two channels, chosen so the centred blocks are
    rank-one and the answer can be written down:

        jc = [[+1, +2], [-1, -2]],  gc = [[+3, -4], [-3, +4]],

    so with the (B-1) = 1 normalisation

        Sigma = jc^T gc = [[6, -8], [12, -16]],

    which is NOT symmetric (the off-diagonals are -8 and +12) and has
    trace 6 - 16 = -10 --- negative, which an auto-covariance can never
    be.
    """
    j = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    g = torch.tensor([[3.0, -4.0], [-3.0, 4.0]])
    expected = torch.tensor([[6.0, -8.0], [12.0, -16.0]])
    return j, g, expected


def test_accumulator_recovers_a_known_nonsymmetric_cross_covariance():
    """The persisted matrix is E[d(iterate) d(grad)^T], not an auto-cov."""
    j, g, expected = _known_pair()
    acc = CrossCovAccumulator(d_bias=2, snap_iters=[7], n_permutations=0)
    acc.record(iter_idx=7, iterate_batch=j, grad_batch=g)

    assert len(acc.matrices) == 1
    got = acc.matrices[0]
    np.testing.assert_allclose(got, expected.numpy(), rtol=0, atol=1e-6)

    # The three properties a genuine cross-covariance has and an
    # auto-covariance cannot:
    assert not np.allclose(got, got.T), "cross-cov must not be symmetric"
    assert acc.sigma_Jac_sq[0] == pytest.approx(-10.0, abs=1e-6)
    assert acc.asymmetry[0] > 0.5
    # ...and the eigenvalues are not all positive, unlike a PSD matrix.
    assert float(np.linalg.eigvalsh((got + got.T) / 2).min()) < 0.0


def test_left_slot_is_the_iterate_and_right_slot_is_the_gradient():
    """Slot order, pinned against the HAND-COMPUTED entries.

    ``A(j, g) == A(g, j).T`` -- which is what this test used to assert
    -- holds identically under BOTH ``jc.T @ gc`` and ``gc.T @ jc``, so
    it pinned nothing. The off-diagonal entries are what distinguish
    ``Sigma_slack`` from ``Sigma_slack^T``:

        Sigma[0, 1] = Cov(theta-tilde_0, g_1) = 1*(-4) + (-1)*4 = -8
        Sigma[1, 0] = Cov(theta-tilde_1, g_0) = 2*(+3) + (-2)*(-3) = +12
    """
    j, g, _ = _known_pair()
    acc = CrossCovAccumulator(d_bias=2, snap_iters=[0], n_permutations=0)
    acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)
    mat = acc.matrices[0]
    assert mat[0, 1] == pytest.approx(-8.0, abs=1e-6)
    assert mat[1, 0] == pytest.approx(+12.0, abs=1e-6)
    # ...and swapping the slots really is a different object.
    swapped = CrossCovAccumulator(d_bias=2, snap_iters=[0], n_permutations=0)
    swapped.record(iter_idx=0, iterate_batch=g, grad_batch=j)
    assert swapped.matrices[0][0, 1] == pytest.approx(+12.0, abs=1e-6)
    assert not np.allclose(mat, swapped.matrices[0])


def test_the_normalisation_is_the_unbiased_one():
    """Divisor is ``B - 1``, not ``B``.

    Both factors are centred at their own sample mean, so only
    ``1/(B-1)`` is unbiased -- which is what the theorem statement
    claims. ``_known_pair`` has B = 2, where the two divisors coincide,
    so it pins nothing; this uses B = 3.
    """
    j = torch.tensor([[1.0], [0.0], [-1.0]])
    g = torch.tensor([[2.0], [0.0], [-2.0]])
    acc = CrossCovAccumulator(d_bias=1, snap_iters=[0], n_permutations=0)
    acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)
    # sum_b jc_b gc_b = 1*2 + 0 + (-1)(-2) = 4;  4/(B-1) = 2, 4/B = 4/3.
    assert float(acc.matrices[0][0, 0]) == pytest.approx(2.0, abs=1e-6)
    assert acc.sigma_Jac_sq[0] == pytest.approx(2.0, abs=1e-6)


# ------------------------------------------------------------- the guard
def test_feeding_the_same_sequence_twice_is_refused():
    """The defect class, stated directly.

    The EXACT type matters. Both guards fire on an exact alias, so
    ``pytest.raises(CrossCovDegenerateInputs)`` alone passes even with
    the literal alias check deleted (``CrossCovSymmetricEstimate`` is a
    subclass and catches the same input downstream). Pin the cheap,
    precise guard: it runs before the O(B^2 d) Gram products and it is
    the one whose message names the slot contract.
    """
    g = torch.randn(16, 4)
    acc = CrossCovAccumulator(d_bias=4, snap_iters=[0])
    with pytest.raises(CrossCovDegenerateInputs) as ei:
        acc.record(iter_idx=0, iterate_batch=g, grad_batch=g)
    assert type(ei.value) is CrossCovDegenerateInputs, (
        "an exact alias must be caught by the alias guard, not only by "
        "the downstream symmetry check"
    )
    assert "SAME sequence" in str(ei.value)
    assert acc.matrices == []


def test_the_exact_shipped_recipe_is_refused():
    """The literal line that shipped: body := grad - grad.mean(0).

    ``scripts/_experiments_cpflow_uci_lib.py`` built its second
    argument as the mean-removed copy of its first. Centring inside the
    accumulator then made the two arguments identical, so the result
    was ``Cov(g)``. Reproduce that call and require it to raise.
    """
    torch.manual_seed(0)
    g_b = torch.randn(32, 5)
    g_body = g_b - g_b.mean(dim=0, keepdim=True)   # the shipped line
    acc = CrossCovAccumulator(d_bias=5, snap_iters=[100])
    with pytest.raises(CrossCovDegenerateInputs) as ei:
        acc.record(iter_idx=100, iterate_batch=g_b, grad_batch=g_body)
    assert type(ei.value) is CrossCovDegenerateInputs


@pytest.mark.parametrize("recipe", ["scaled", "near"])
def test_a_proportional_pair_is_refused_too(recipe):
    """The near-misses the literal alias guard cannot see.

    ``j = 3g`` and ``j = g + 1e-3 * noise`` both slip past
    ``||jc - gc|| / scale <= 1e-6`` and both produce a symmetric PSD
    auto-covariance with a tautologically positive trace. The
    ASYMMETRY of the resulting estimate is what catches them.
    """
    torch.manual_seed(3)
    g = torch.randn(16, 5)
    j = 3.0 * g if recipe == "scaled" else g + 1e-3 * torch.randn(16, 5)
    acc = CrossCovAccumulator(d_bias=5, snap_iters=[0], n_permutations=0)
    with pytest.raises(CrossCovSymmetricEstimate):
        acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)
    assert acc.matrices == []


def test_the_shipped_recipe_would_have_produced_a_psd_auto_covariance():
    """Why it matters: the old output was positive by construction.

    Computed here directly (not through the accumulator, which now
    refuses it) to show what the persisted snapshots contain.
    """
    torch.manual_seed(0)
    g_b = torch.randn(64, 6)
    gc = g_b - g_b.mean(dim=0, keepdim=True)
    shipped = (gc.T @ gc / (g_b.shape[0] - 1)).numpy()
    assert np.allclose(shipped, shipped.T, atol=1e-6)
    assert float(np.linalg.eigvalsh(shipped).min()) > 0.0
    assert float(np.trace(shipped)) > 0.0     # tautologically


def test_a_structurally_constant_pair_is_not_mistaken_for_the_defect():
    """Both channels flat is a legitimate (if empty) reading, not an alias."""
    z = torch.zeros(8, 3)
    acc = CrossCovAccumulator(d_bias=3, snap_iters=[0], n_permutations=0)
    acc.record(iter_idx=0, iterate_batch=z, grad_batch=z)
    assert np.allclose(acc.matrices[0], 0.0)
    # ``asymmetry`` is UNDEFINED when the estimate is exactly zero. It
    # must not be reported as 0.0, which is the auto-covariance
    # signature this very field exists to expose.
    assert math.isnan(acc.asymmetry[0])


def test_warn_mode_records_instead_of_raising():
    g = torch.randn(8, 3)
    acc = CrossCovAccumulator(
        d_bias=3, snap_iters=[0], on_degenerate="warn", n_permutations=0,
    )
    with pytest.warns(RuntimeWarning):
        acc.record(iter_idx=0, iterate_batch=g, grad_batch=g)
    assert len(acc.matrices) == 1


def test_a_single_probe_is_refused():
    """B = 1 has no fluctuation; the old code divided by max(B-1, 1)."""
    acc = CrossCovAccumulator(d_bias=2, snap_iters=[0])
    with pytest.raises(ValueError, match="B >= 2"):
        acc.record(
            iter_idx=0,
            iterate_batch=torch.randn(1, 2),
            grad_batch=torch.randn(1, 2),
        )


def test_a_width_mismatch_is_refused():
    """A block of the wrong width would silently mis-size every matrix."""
    acc = CrossCovAccumulator(d_bias=5, snap_iters=[0])
    with pytest.raises(ValueError, match="d_bias"):
        acc.record(
            iter_idx=0,
            iterate_batch=torch.randn(4, 3),
            grad_batch=torch.randn(4, 3),
        )


@pytest.mark.parametrize("slot", ["iterate", "grad"])
def test_a_non_finite_probe_raises_its_own_error_not_the_alias_one(slot):
    """A NaN must not be diagnosed as an alias.

    ``nan > 1e-6`` and ``nan <= 1e-6`` are BOTH False, so the pre-fix
    guard returned NaN and the caller read it as "aliased" -- aborting
    a 22-hour run with the wrong diagnosis, on an event the trainer
    itself budgets 2000 occurrences for.
    """
    torch.manual_seed(1)
    j, g = torch.randn(6, 4), torch.randn(6, 4)
    (j if slot == "iterate" else g)[2, 1] = float("nan")
    acc = CrossCovAccumulator(d_bias=4, snap_iters=[0])
    with pytest.raises(CrossCovNonFinite):
        acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)
    assert acc.matrices == []


def test_an_unprojected_matrix_that_would_not_fit_is_refused_at_setup():
    """Failing at construction beats failing 10k iters into a 17h run."""
    with pytest.raises(ValueError, match="max_eff_d"):
        CrossCovAccumulator(d_bias=200_000, snap_iters=[0], jl_proj_dim=None)


def test_is_pgd_is_provenance_only():
    """The sentinel branch is gone: is_pgd changes no arithmetic."""
    j, g, expected = _known_pair()
    acc = CrossCovAccumulator(
        d_bias=2, snap_iters=[0], is_pgd=True, n_permutations=0,
    )
    acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)
    np.testing.assert_allclose(
        acc.matrices[0], expected.numpy(), rtol=0, atol=1e-6,
    )


# ------------------------------------------- exactness of the scalars (D1)
def test_scalars_are_exact_not_read_off_the_jl_projection():
    """The projected trace is unbiased but its noise swamps the signal.

    ``Var[tr(P^T Sigma P)] = (2/k) ||Sigma_sym||_F^2``, so at k << d the
    projected trace can carry the WRONG SIGN. The persisted
    ``sigma_Jac_sq`` / ``frob`` / ``asymmetry`` must therefore be
    computed on the unprojected blocks; ``*_proj`` keeps the projected
    values for comparison.
    """
    torch.manual_seed(11)
    d, k, B = 600, 8, 12
    j = torch.randn(B, d)
    g = torch.randn(B, d)
    acc = CrossCovAccumulator(
        d_bias=d, snap_iters=[0], jl_proj_dim=k, n_permutations=0,
    )
    acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)

    jc = (j - j.mean(0, keepdim=True)).double()
    gc = (g - g.mean(0, keepdim=True)).double()
    sigma = jc.T @ gc / (B - 1)
    assert acc.sigma_Jac_sq[0] == pytest.approx(
        float(sigma.trace()), rel=1e-4,
    )
    assert acc.frob[0] == pytest.approx(float(sigma.norm()), rel=1e-4)
    assert acc.asymmetry[0] == pytest.approx(
        float((sigma - sigma.T).norm() / sigma.norm()), rel=1e-4,
    )
    # ...and the projected reading really is a different number, so
    # this test would fail if the scalars went back to reading `mat`.
    assert acc.frob_proj[0] != pytest.approx(acc.frob[0], rel=1e-2)
    # The per-channel variances divide by the TRUE width, not k.
    assert acc.iterate_var[0] == pytest.approx(
        float(jc.pow(2).sum()) / ((B - 1) * d), rel=1e-4,
    )


def test_the_permutation_null_separates_coupled_from_independent():
    """Shuffling the probe pairing is an EXACT test of case (iii).

    Under conditional independence of the two channels the pairing is
    exchangeable, so the permutation distribution needs no asymptotics.
    This is the only handle in the repo on the condition
    ``04_mechanism.tex`` says "is isolated by no architecture here".
    """
    torch.manual_seed(5)
    B, d = 40, 24
    j = torch.randn(B, d)
    coupled = 0.6 * j + 0.8 * torch.randn(B, d)
    independent = torch.randn(B, d)

    def _p(g):
        acc = CrossCovAccumulator(
            d_bias=d, snap_iters=[0], n_permutations=300,
        )
        acc.record(iter_idx=0, iterate_batch=j, grad_batch=g)
        return (acc.perm_p_trace[0], acc.frob[0],
                acc.frob_null_median[0], acc.frob_excess[0])

    p_c, fro_c, null_c, exc_c = _p(coupled)
    p_i, fro_i, null_i, exc_i = _p(independent)
    assert p_c < 0.02, f"a coupled pair must be detected; got p={p_c}"
    assert p_i > 0.05, f"an independent pair must not be; got p={p_i}"
    # THE POINT OF THE NULL: the RAW Frobenius norm barely separates the
    # two (4.54 vs 3.71 here) because at finite B most of the plotted
    # mass is sampling noise. The null-corrected excess separates them
    # by ~6x. A figure that plots ||Sigma||_F without its null is
    # plotting mostly noise.
    assert fro_c < 1.5 * fro_i, "the raw norm is a weak discriminator"
    assert exc_c > 4.0 * exc_i, "the null-corrected excess is a strong one"
    assert fro_i == pytest.approx(null_i, rel=0.3)


def test_archive_carries_the_provenance_a_reader_needs(tmp_path):
    j, g, _ = _known_pair()
    acc = CrossCovAccumulator(
        d_bias=2, snap_iters=[3], iterate_source="unit_test",
        n_permutations=16,
    )
    acc.record(iter_idx=3, iterate_batch=j, grad_batch=g)
    path = str(tmp_path / "cc.npz")
    acc.to_npz(path)

    z = np.load(path)
    # Keys the existing renderers read must survive untouched.
    for key in ("matrices", "iters", "sigma_Jac_sq", "frob", "is_pgd",
                "d_bias", "jl_proj_dim"):
        assert key in z.files, key
    assert int(z["schema_version"]) == CROSS_COV_SCHEMA_VERSION
    assert str(z["left_slot"]) == "iterate_fluctuation"
    assert str(z["right_slot"]) == "grad_fluctuation"
    assert str(z["iterate_source"]) == "unit_test"
    assert "d(theta_tilde) d(g)" in str(z["quantity"])
    assert "B-1" in str(z["estimator_normalisation"])
    assert int(z["scalars_are_exact"]) == 1
    assert float(z["asymmetry"][0]) > 0.5
    assert int(z["n_samples"][0]) == 2
    # The projection defect is measurable from the archive alone.
    for key in ("sigma_Jac_sq_proj", "frob_proj", "iterate_rel_var",
                "grad_rel_var", "perm_p_trace", "frob_excess"):
        assert key in z.files, key


# ------------------------------------------------------- the hypernet tap
def _emitter(seed: int = 0):
    from lift.models import HyperNetwork, ICNN

    torch.manual_seed(seed)
    icnn = ICNN(input_size=4, hidden_dim=6, nlayers=3)
    torch.manual_seed(seed)
    return icnn, HyperNetwork(
        input_size=4,
        hidden_sizes=[8, 8],
        downstream_network=icnn,
        pos_param_names=None,
    )


def test_capture_returns_the_pre_positivity_emission():
    """The tap is ``Theta_E h_phi(X) + b_h``, before psi is applied."""
    import torch.nn.functional as F

    _, hyper = _emitter()
    assert hyper.pos_param_names, "expected a positivity-tagged ICNN target"
    x = torch.randn(5, 4)
    with hyper.capture_pre_positivity() as store:
        out = hyper(x)
    assert set(store) == set(hyper._param_names)
    for name, tensor in out.items():
        flat = store[name].view(tensor.shape)
        if name in hyper.pos_param_names:
            assert torch.allclose(F.softplus(flat), tensor, atol=1e-6)
            # ...and the raw emission is genuinely not the emitted W.
            assert not torch.allclose(flat, tensor, atol=1e-4)
        else:
            assert torch.allclose(flat, tensor, atol=1e-6)


def test_capture_is_off_by_default_and_changes_nothing():
    _, a = _emitter(seed=1)
    _, b = _emitter(seed=1)
    x = torch.randn(5, 4)
    assert a._pre_positivity_capture is None
    with torch.no_grad():
        plain = a(x)
        with b.capture_pre_positivity():
            tapped = b(x)
        after = b(x)
    for k in plain:
        assert torch.equal(plain[k], tapped[k])
        assert torch.equal(plain[k], after[k])
    assert b._pre_positivity_capture is None
    # The tap is not model state.
    assert not any("pre_positivity" in k for k in b.state_dict())


def test_a_nested_capture_does_not_clobber_the_outer_emission_count():
    """The count is what a caller asserts "exactly one probe" against."""
    _, hyper = _emitter(seed=4)
    x = torch.randn(3, 4)
    with hyper.capture_pre_positivity():
        hyper(x)
        assert hyper._pre_positivity_emissions == 1
        with hyper.capture_pre_positivity():
            hyper(x)
            assert hyper._pre_positivity_emissions == 1
        assert hyper._pre_positivity_emissions == 1, (
            "the inner block reset the outer tally"
        )


def test_emitted_iterate_and_bias_grads_are_index_aligned():
    """Head by head, the two rows describe the same scalar channels.

    This is the alignment ``record_cross_cov_snapshot`` asserts: the
    per-head width of the captured emission equals the per-head width
    of the readout bias whose gradient fills the other slot, and the
    concatenation order is one shared iteration over
    ``_param_names`` / ``weight_predictors``.
    """
    _, hyper = _emitter(seed=2)
    x = torch.randn(6, 4)
    with hyper.capture_pre_positivity() as store:
        out = hyper(x)
    loss = sum(v.sum() for v in out.values())

    heads = hyper.positivity_readout_heads()
    assert heads, "no positivity-tagged head with a bias to tap"
    names = [n for n, _ in heads]
    biases = [b for _, b in heads]

    grads = torch.autograd.grad(loss, biases, allow_unused=False)
    for name, bias, grad in zip(names, biases, grads):
        assert store[name].reshape(-1).numel() == bias.numel()
        assert grad.reshape(-1).numel() == bias.numel()
    j_vec = torch.cat([store[n].reshape(-1) for n in names])
    g_vec = torch.cat([g.reshape(-1) for g in grads])
    assert j_vec.shape == g_vec.shape
    # Aligned but genuinely different sequences.
    assert not torch.allclose(j_vec, g_vec)


# -------------------------------------------------- the CP-Flow recorder
def _tiny_cpflow(backend: str, **kw):
    lib = pytest.importorskip("_experiments_cpflow_uci_lib")
    cfg = lib.CPFlowConfig(D=2, dimh=4, nhidden=2, nblocks=1, m1=2)
    if backend == "hypernet":
        flow = lib.build_cpflow_hypernet(
            cfg, hyper_hidden=[8, 8], seed=0, condition_on="batch", **kw,
        )
    else:
        flow = lib.build_cpflow_direct(cfg, seed=0, **kw)
    return lib, flow


def _run_recorder(flow, *, n_probes=6, jl=4, seed=0, iter_idx=1,
                  n_perm=0, data_seed=0):
    taps = cross_cov_taps(flow)
    acc = CrossCovAccumulator(
        d_bias=flat_bias_dim(taps), snap_iters=[iter_idx], jl_proj_dim=jl,
        iterate_source=tap_iterate_source(taps), n_permutations=n_perm,
    )
    rng = np.random.default_rng(data_seed)
    data = rng.standard_normal((256, 2)).astype(np.float32)
    stats = record_cross_cov_snapshot(
        flow, accumulator=acc, taps=taps, train_data=data,
        batch_size=16, n_probes=n_probes, iter_idx=iter_idx,
        device=torch.device("cpu"), probe_rng=make_probe_rng(seed),
        seed=seed,
    )
    return taps, acc, stats


@pytest.mark.parametrize("backend", ["hypernet", "direct"])
def test_recorder_runs_the_same_code_on_every_arm(backend):
    """No sentinel branch: both arms are measured, and say so.

    The lifted arm's emission moves with the probe batch, so its
    iterate fluctuation is live. The unlifted arm's iterate is a stored
    parameter that does not move at a fixed iterate, so its
    fluctuation --- and hence Sigma_slack --- is at the float32
    round-off floor BY MEASUREMENT. That is Theorem 1 case (ii)
    observed, not asserted.
    """
    lib, flow = _tiny_cpflow(backend)
    taps, acc, _ = _run_recorder(flow)
    assert taps, f"{backend}: expected a tappable positivity channel"
    kinds = {t.kind for t in taps}
    assert kinds == ({"emitted"} if backend == "hypernet" else {"parameter"})

    assert len(acc.matrices) == 1
    mat = acc.matrices[0]
    assert acc.grad_var[0] > 0.0, "the gradient fluctuates on either arm"
    if backend == "hypernet":
        # RELATIVE, not absolute. An absolute ``iterate_var > 0`` is
        # cleared by float32 round-off on an exactly-constant sequence
        # (measured 1.8e-14), so it does not distinguish a live emission
        # from a bias-parameter tap. The relative fluctuation does:
        # ~2e-3 live against ~5e-15 constant.
        assert acc.iterate_rel_var[0] > 1e-8, (
            "the emission must move with the probe batch"
        )
        assert acc.iterate_var[0] > 0.0
        assert float(np.abs(mat).max()) > 0.0
        assert acc.asymmetry[0] > ASYMMETRY_FLOOR, (
            "a genuine cross-cov is not symmetric"
        )
    else:
        # The residual is round-off in the ITERATE's own units, so it is
        # compared to the ITERATE's own scale. Comparing it to
        # ``grad_var`` (which is what shipped) is dimensionally
        # inconsistent and fails by 8 orders of magnitude at paper
        # geometry, where the two channels' scales differ.
        assert acc.iterate_rel_var[0] < 1e-10, (
            "a stored parameter cannot move across probes at a fixed "
            "iterate"
        )
        assert acc.frob[0] < 1e-6 * acc.frob_null_median[0] + 1e-9 or \
            acc.frob[0] < 1e-9, "the unlifted arm sits at the round-off floor"


def test_tapping_the_bias_parameter_is_caught():
    """The exact degeneracy the brief warned against.

    Recording ``b_h`` per probe instead of the EMISSION gives an
    exactly-constant sequence, hence an exactly-zero object -- and the
    lifted arm would then report the unlifted arm's reading while
    looking alive, because float32 mean subtraction leaves round-off
    that clears any ABSOLUTE variance threshold and, being noise, makes
    the resulting matrix non-symmetric too. Only the RELATIVE
    fluctuation separates them.
    """
    lib, flow = _tiny_cpflow("hypernet")
    good_taps = cross_cov_taps(flow)
    # The mutant: kind="parameter" on the readout bias, i.e. read b_h.
    from lift.objectives.cross_cov_taps import BiasChannelTap
    bad_taps = [
        BiasChannelTap(
            hypernet=None, emitted_name=t.emitted_name,
            grad_param=t.grad_param, kind="parameter",
        )
        for t in good_taps
    ]

    def _rel(taps):
        acc = CrossCovAccumulator(
            d_bias=flat_bias_dim(taps), snap_iters=[1], jl_proj_dim=4,
            n_permutations=0,
        )
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 2)).astype(np.float32)
        record_cross_cov_snapshot(
            flow, accumulator=acc, taps=taps, train_data=data,
            batch_size=16, n_probes=6, iter_idx=1,
            device=torch.device("cpu"), probe_rng=make_probe_rng(0),
        )
        return acc

    live, dead = _rel(good_taps), _rel(bad_taps)
    assert live.iterate_rel_var[0] > 1e-8
    assert dead.iterate_rel_var[0] < 1e-10
    # The trap: the ABSOLUTE variance does NOT separate them reliably,
    # which is why the threshold has to be the relative one.
    assert dead.iterate_rel_var[0] < 1e-6 * live.iterate_rel_var[0]


def test_recorder_asserts_one_emission_per_probe(monkeypatch):
    """Two forwards inside one capture means the wrong iterate.

    The captured emission must come from the SAME forward whose loss is
    differentiated. Nothing else in the recorder checks that, so the
    counter is the only guard -- and it must be a HARD FAILURE reached
    through the recorder, not merely a number a test reads.
    """
    lib, flow = _tiny_cpflow("hypernet")
    taps = cross_cov_taps(flow)
    hyper = hypernets_of(flow)[0]
    real_logp = flow.logp

    def double_emitting(x, *a, **kw):
        hyper(x.detach())          # a second, unrelated emission
        return real_logp(x, *a, **kw)

    monkeypatch.setattr(flow, "logp", double_emitting)
    acc = CrossCovAccumulator(
        d_bias=flat_bias_dim(taps), snap_iters=[1], jl_proj_dim=4,
        n_permutations=0,
    )
    data = np.random.default_rng(0).standard_normal(
        (256, 2)).astype(np.float32)
    with pytest.raises(RuntimeError, match="exactly one emission"):
        record_cross_cov_snapshot(
            flow, accumulator=acc, taps=taps, train_data=data,
            batch_size=16, n_probes=4, iter_idx=1,
            device=torch.device("cpu"), probe_rng=make_probe_rng(0),
        )
    assert acc.matrices == []


def test_the_emission_ema_is_frozen_around_every_probe():
    """A measurement must not advance the schedule it measures.

    ``_decay_pooled`` updates the running mean on every TRAINING
    forward. Freezing once around the whole probe loop (which is what
    shipped) still lets probe b's emission depend on probes 1..b-1, so
    the probes stop being i.i.d. -- the one property the sample axis
    needs. The EMA must be identical before and after the snapshot.
    """
    lib, flow = _tiny_cpflow(
        "hypernet", emission_decay_frac=0.9, emission_decay_ema=0.5,
    )
    hyper = hypernets_of(flow)[0]
    assert hyper._pooled_ema is not None
    hyper.set_train_progress(0.5)
    # Prime the EMA with a real training forward, then snapshot it.
    flow.train()
    hyper(torch.randn(8, 2))
    before = hyper._pooled_ema.clone()
    ready_before = hyper._pooled_ema_ready.clone()
    assert float(ready_before) > 0.0, "EMA must be armed for this to bite"

    _run_recorder(flow, n_probes=5)
    assert torch.equal(hyper._pooled_ema, before), (
        "the probe loop advanced the emission-decay EMA"
    )
    assert torch.equal(hyper._pooled_ema_ready, ready_before)


def test_the_probe_is_a_pure_observer_of_the_training_stream():
    """``--with_cross_cov`` must not change the run it measures.

    The recorder used to draw its probe batches from the TRAINING numpy
    generator and to run ``flow.logp`` (hence Rademacher draws) on the
    global torch RNG, so a ``--with_cross_cov`` run diverged from its
    control at the first iteration after a snapshot -- on BOTH arms,
    and newly so on the direct arm, whose short-circuit used to return
    before drawing anything. Two runs differing only in the flag then
    wrote different models into the same directory.
    """
    lib, flow = _tiny_cpflow("hypernet")
    data = np.random.default_rng(0).standard_normal(
        (512, 2)).astype(np.float32)

    def _stream(with_probe: bool):
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        it = lib._make_inf_batch_iter(data, 16, rng)
        out = [next(it).clone()]
        if with_probe:
            _run_recorder(flow, n_probes=4, data_seed=1)
        out.append(next(it).clone())
        out.append(torch.randn(3))          # global torch RNG probe
        return out

    a = _stream(False)
    b = _stream(True)
    for x, y in zip(a, b):
        assert torch.equal(x, y), (
            "the diagnostic perturbed the stream it observes"
        )


def test_recorder_leaves_no_gradient_residue_on_the_tapped_params():
    """The probe must not perturb the run it is measuring."""
    lib, flow = _tiny_cpflow("hypernet")
    taps = cross_cov_taps(flow)
    before = [t.grad_param.detach().clone() for t in taps]
    was_training = flow.training
    flow.eval()
    _run_recorder(flow, n_probes=4, data_seed=1)
    for tap, was in zip(taps, before):
        assert torch.equal(tap.grad_param.detach(), was)
        assert tap.grad_param.grad is None or torch.count_nonzero(
            tap.grad_param.grad,
        ) == 0
    for hyper in hypernets_of(flow):
        assert hyper._pre_positivity_capture is None
    assert not flow.training, "the recorder must restore the flow's mode"
    flow.train(was_training)


def test_a_non_finite_probe_is_redrawn_not_fatal(monkeypatch):
    """One bad probe must not kill the snapshot, let alone the run."""
    lib, flow = _tiny_cpflow("hypernet")
    taps = cross_cov_taps(flow)
    real_logp = flow.logp
    state = {"n": 0}

    def flaky(x, *a, **kw):
        state["n"] += 1
        out = real_logp(x, *a, **kw)
        if state["n"] in (2, 5):
            return out * float("nan")
        return out

    monkeypatch.setattr(flow, "logp", flaky)
    acc = CrossCovAccumulator(
        d_bias=flat_bias_dim(taps), snap_iters=[1], jl_proj_dim=4,
        n_permutations=0,
    )
    rng = np.random.default_rng(0)
    data = rng.standard_normal((256, 2)).astype(np.float32)
    stats = record_cross_cov_snapshot(
        flow, accumulator=acc, taps=taps, train_data=data,
        batch_size=16, n_probes=5, iter_idx=1,
        device=torch.device("cpu"), probe_rng=make_probe_rng(0),
    )
    assert stats["n_probes_discarded"] == 2.0
    assert stats["n_probes_used"] == 5.0
    assert len(acc.matrices) == 1


# ------------------------------------------------------------ MoreauProbe
def test_moreau_probe_refuses_a_single_probe_and_carries_the_1_over_d():
    """The two defects it inherited from the accumulator, both closed.

    ``B = max(B - 1, 1)`` accepted one probe and returned ``0.0`` --- an
    absent measurement rendered as a measured zero. And ``eq:mu-eff``
    is ``sigma_Jac^2 / (d kappa^2)``; the ``1/d`` was missing.
    """
    p = MoreauProbe()
    with pytest.raises(ValueError, match="B >= 2"):
        p.record(iter_idx=0, iterate_batch=torch.randn(1, 3),
                 grad_batch=torch.randn(1, 3), kappa=1.0)

    j, g, _ = _known_pair()
    out = p.record(iter_idx=1, iterate_batch=j, grad_batch=g, kappa=2.0)
    assert out["sigma_Jac_sq"] == pytest.approx(-10.0, abs=1e-6)
    assert out["mu_eff_hat"] == pytest.approx(-10.0 / (2 * 2.0 ** 2), abs=1e-6)


def test_moreau_probe_refuses_an_aliased_pair():
    p = MoreauProbe()
    g = torch.randn(8, 3)
    with pytest.raises(CrossCovDegenerateInputs):
        p.record(iter_idx=0, iterate_batch=g, grad_batch=g, kappa=1.0)


def test_kappa_cannot_silently_collapse_to_1e_minus_12():
    """The lifted arm's readout scale is not in ``state_dict`` at all.

    Falling back to ``1e-12`` inflates ``mu_eff_hat`` by 10^24. Say so
    instead, and offer the reading that does exist: the captured
    emission.
    """
    with pytest.raises(KeyError, match="kappa_from_emission"):
        MoreauProbe._kappa_from_state_dict(
            {"hypernet.trunk.0.weight": torch.randn(3, 3)},
            pos_param_names={"z_layers.0.weight"},
        )
    assert MoreauProbe.kappa_from_emission(
        torch.tensor([0.5, -2.5, 1.0]),
    ) == pytest.approx(2.5)
