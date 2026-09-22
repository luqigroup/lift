r"""Training-time diagnostics for the ICNN-lift mechanism.

Two probes, both batch-resolved and both free when their accumulator is
disabled. They are wired into the ``train_ebm`` outer loop
(:mod:`lift.objectives.trainer`) and into the PICNN trainer
(:mod:`lift.baselines.picnn_hypernet`) behind config flags.

The CP-Flow side of the instrument, the taps that read the emitted iterate off
a live flow and the snapshot recorder that drives them, lives in
:mod:`lift.objectives.cross_cov_taps`; this module holds the estimator.

Cross-covariance accumulator
----------------------------
:class:`CrossCovAccumulator` records the empirical batch-level
cross-covariance

.. math::

    \Sigma_{\rm slack}
    \;=\;
    \mathbb{E}\bigl[\,
        \delta\widetilde{\theta}\;\delta g^{\top}
    \bigr],

between the fluctuation of the emitted pre-positivity iterate
``theta-tilde = Theta_E h_phi(X) + b_h``, the left factor, and the
fluctuation of the loss gradient on the readout bias ``b_h``, the right
factor. Those are two different sequences: the matrix is neither symmetric
nor PSD, and its trace ``sigma_Jac^2`` carries a sign.

The sample axis is probe batches at a fixed iterate, not training steps. The
fluctuation is defined with ``(phi, Theta_E, b)`` held fixed, so the estimator
draws several batches at one iterate and centers over that axis; probe batches
at a fixed iterate are i.i.d., which the training-step axis is not.

.. warning::

    Feeding the same sequence into both slots, for example the bias gradient
    passed as the left argument with its mean removed, turns the estimate into
    ``Cov(g)``: a symmetric PSD auto-covariance whose trace is a sum of
    variances and so positive by construction. It is refused at runtime by
    :meth:`CrossCovAccumulator.record`, and every archive carries
    ``asymmetry`` per snapshot so a reader can tell the two apart from the
    artifact alone.

.. warning::

    ``sigma_Jac_sq``, ``frob``, ``asymmetry``, ``iterate_var`` and
    ``grad_var`` are computed exactly, on the unprojected ``(B, d)`` blocks,
    through three ``B x B`` Gram matrices. Reading them off the
    Johnson-Lindenstrauss projection instead is unbiased but has standard
    deviation ``sqrt(2/k) ||Sigma_sym||_F``, which at ``k = 21`` and HEPMASS
    geometry is about 150 times the trace being estimated, so the projected
    trace gets the sign wrong about half the time. Schema 3 holds the exact
    values and keeps the projected ones alongside under ``*_proj`` keys.

The ``(k, k)`` projected matrix is persisted per snapshot so the filmstrip
figure can be drawn as a heatmap. It is a picture of the estimate; every
number quoted from the archive comes from the exact keys.

Moreau-envelope probe
---------------------
:class:`MoreauProbe` measures the two scalars whose ratio is the effective
Moreau smoothing parameter ``mu_eff``:

* ``sigma_Jac^2(t)``, the trace of the slack-channel cross-covariance;
* ``kappa(t) = ||W_tilde||_inf``, the typical readout scale.

It returns ``mu_eff_hat = sigma_Jac^2 / (d kappa^2)`` directly, with the
``1/d`` part of the definition.

Both probes take the model's ``state_dict``-shaped parameter dict at call time
and return detached tensors or floats; neither mutates the model.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------- cross-covariance

#: Relative tolerance below which the two centered inputs handed to
#: :meth:`CrossCovAccumulator.record` are judged to be the same sequence. A
#: genuine (iterate, gradient) pair carries different units and cannot land
#: here; a pair that does is the aliasing defect of the module docstring.
ALIAS_RTOL = 1e-6

#: Relative asymmetry ``||M - M^T||_F / ||M||_F`` below which a nonzero
#: estimate is judged to be an auto-covariance in disguise. The alias guard
#: above catches only ``j == g``; ``j = 3g`` and ``j = g + 1e-3 * noise`` slip
#: past it and still give a symmetric PSD matrix with a positive trace by
#: construction. At ``B << d`` a genuine sample cross-covariance is
#: rank-deficient and its asymmetry saturates near ``sqrt(2)``.
ASYMMETRY_FLOOR = 1e-3

#: Hard cap on the side length of the persisted matrix. ``jl_proj_dim <= 0``
#: keeps the full ``d_bias``, which at HEPMASS width is a 184640^2 float32
#: matrix, so the cap fails at construction rather than deep into a run.
MAX_EFF_D = 4096

#: Bumped whenever the persisted NPZ layout or the measured quantity changes.
#:
#: * ``1`` (implicit, no field): the auto-covariance ``Cov(g)``, read off the
#:   JL projection. Nothing in such an archive is a cross-covariance.
#: * ``2``: ``Sigma_slack`` with the correct slots, scalars still read off the
#:   JL projection.
#: * ``3``: ``Sigma_slack``, scalars computed exactly on the unprojected
#:   blocks, projected values kept alongside, and an optional permutation null
#:   on the probe pairing.
CROSS_COV_SCHEMA_VERSION = 3


class CrossCovDegenerateInputs(ValueError):
    """The two sequences handed to the accumulator are the same sequence.

    Raised by :meth:`CrossCovAccumulator.record`. Measuring
    ``E[d a d a^T]`` instead of ``E[d a d b^T]`` yields a symmetric PSD
    auto-covariance whose trace is a sum of variances, positive by
    construction and therefore evidence of nothing.
    """


class CrossCovSymmetricEstimate(CrossCovDegenerateInputs):
    """The estimate came out symmetric, i.e. an auto-covariance.

    A subclass, so ``except CrossCovDegenerateInputs`` catches the whole
    defect class. It fires on the near misses the alias guard cannot see,
    such as ``j = c * g`` or ``j = g + small noise``.
    """


class CrossCovNonFinite(ValueError):
    """A probe block carried a NaN or an inf.

    Distinct from the aliasing errors: with a non-finite entry every
    downstream comparison evaluates ``False``, so the alias guard would
    report the wrong diagnosis. The stochastic Lanczos/CG log-det
    estimator that supplies the probe loss returns a non-finite value on
    occasional batches, so this is an expected event, and the CP-Flow
    recorder redraws such a probe rather than letting it reach here.
    """


def _alias_defect_message(rel: float) -> str:
    return (
        "the SAME sequence was handed to both cross-covariance slots "
        f"(relative difference {rel:.3e} <= {ALIAS_RTOL:g}). The result "
        "would be a symmetric PSD auto-covariance whose trace is a sum "
        "of variances, i.e. positive by construction. The left slot "
        "must be the ITERATE fluctuation (the emitted pre-positivity "
        "theta-tilde) and the right slot the GRADIENT fluctuation on "
        "the readout bias."
    )


def _symmetric_estimate_message(asym: float) -> str:
    return (
        f"the estimate is symmetric to {asym:.3e} < {ASYMMETRY_FLOOR:g} "
        "relative Frobenius mass, i.e. it is an AUTO-covariance. The "
        "literal alias guard passed, so the two slots are not equal --- "
        "but they are proportional or near-proportional, which produces "
        "the same PSD, tautologically-positive-trace object. At "
        "B << d_bias a genuine sample cross-covariance is rank-deficient "
        "and its asymmetry sits near sqrt(2); the only way to land here "
        "legitimately is a population Sigma = Cov(dtheta) H with "
        "Cov(dtheta) isotropic AND B >= d_bias, which no configuration "
        "in this repo has. Pass on_degenerate='warn' if you are that "
        "exception."
    )


def _alias_defect_rel(
    left_c: torch.Tensor, right_c: torch.Tensor,
) -> Optional[float]:
    """Relative gap between two centered blocks, or ``None`` if fine.

    Returns ``None`` when the pair is acceptable and the relative distance
    otherwise. Cost is two norms over a ``(B, d)`` block.

    An all-zero pair is acceptable: both channels constant across probes
    is an uninformative but legitimate reading, and not the aliasing
    defect, which always shows a live gradient fluctuation mirrored into
    the iterate slot.

    A non-finite entry makes both ``scale`` and ``rel`` NaN, and every
    comparison against them is ``False``, which would read as "aliased".
    Non-finiteness is therefore its own condition, raised by
    :func:`_check_finite_or_raise` before this runs.
    """
    scale = float(torch.maximum(left_c.norm(), right_c.norm()))
    if not math.isfinite(scale) or scale == 0.0:
        return None
    rel = float((left_c - right_c).norm() / scale)
    if not math.isfinite(rel):
        return None
    return None if rel > ALIAS_RTOL else rel


def _check_finite_or_raise(
    left: torch.Tensor, right: torch.Tensor, *, where: str = "record",
) -> None:
    bad_l = int((~torch.isfinite(left)).sum())
    bad_r = int((~torch.isfinite(right)).sum())
    if bad_l or bad_r:
        raise CrossCovNonFinite(
            f"{where}: {bad_l} non-finite entries in the iterate block "
            f"and {bad_r} in the gradient block. A cross-covariance "
            "cannot be formed from these, and every subsequent guard "
            "would read them as a different defect."
        )


def _check_distinct_or_raise(
    left_c: torch.Tensor, right_c: torch.Tensor,
) -> None:
    rel = _alias_defect_rel(left_c, right_c)
    if rel is not None:
        raise CrossCovDegenerateInputs(_alias_defect_message(rel))


def _gram_triple(
    jc: torch.Tensor, gc: torch.Tensor, *, chunk: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""The three ``B x B`` Gram matrices every exact scalar reduces to.

    Returns ``(A, Bm, K)`` in float64 where

    * ``A   = jc jc^T``
    * ``Bm  = gc gc^T``
    * ``K   = jc gc^T``

    With ``Sigma = jc^T gc / (n - 1)`` the identities are

    * ``tr Sigma          = tr(K) / (n - 1)``
    * ``||Sigma||_F^2     = tr(A Bm) / (n - 1)^2 = sum(A * Bm) / (n-1)^2``
    * ``tr(Sigma^2)       = tr(K^2) / (n - 1)^2 = sum(K * K^T) / (n-1)^2``
    * ``||Sigma - Sigma^T||_F^2 = 2||Sigma||_F^2 - 2 tr(Sigma^2)``

    so the ``d x d`` matrix never has to be formed to read its trace,
    its Frobenius norm or its asymmetry exactly.

    All four scalars are transpose invariant, so building ``K`` the other
    way round, ``gc jc^T``, changes none of them. Slot order is observable
    only in the persisted ``matrices`` heatmap, which is formed separately
    from ``jp.T @ gp``: do not derive the heatmap from ``K`` without also
    pinning its orientation.

    Cost is ``O(B^2 d)``, once per snapshot. The accumulation is float64
    and chunked over the ``d`` axis, so exactness costs a ``(B, chunk)``
    float64 slice rather than a float64 copy of the whole block.
    """
    n, d = jc.shape
    A = torch.zeros((n, n), dtype=torch.float64, device=jc.device)
    Bm = torch.zeros((n, n), dtype=torch.float64, device=jc.device)
    K = torch.zeros((n, n), dtype=torch.float64, device=jc.device)
    for s in range(0, d, int(chunk)):
        js = jc[:, s:s + int(chunk)].double()
        gs = gc[:, s:s + int(chunk)].double()
        A += js @ js.T
        Bm += gs @ gs.T
        K += js @ gs.T
    return A, Bm, K


def _permutation_null(
    A: torch.Tensor,
    Bm: torch.Tensor,
    K: torch.Tensor,
    *,
    n_perm: int,
    nm1: float,
    seed: int,
) -> Dict[str, float]:
    r"""Exchangeability test on the PROBE PAIRING.

    Shuffling which gradient row is paired with which iterate row destroys
    any within-probe coupling and leaves both marginals untouched. Under
    conditional independence of the two channels the pairing is
    exchangeable, so the permutation distribution is exact: no asymptotics
    and no normality assumption.

    Every null draw is ``O(B^2)`` because the permuted statistics read
    off the same Gram matrices:
    ``tr Sigma_pi = sum_b K[b, pi(b)] / (n-1)`` and
    ``||Sigma_pi||_F^2 = sum(A * Bm[pi][:, pi]) / (n-1)^2``.
    """
    n = A.shape[0]
    A_np = A.detach().cpu().numpy()
    B_np = Bm.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()
    tr_obs = float(np.trace(K_np)) / nm1
    fro_obs = math.sqrt(max(float((A_np * B_np).sum()), 0.0)) / nm1

    rng = np.random.default_rng([0xC05C0, int(seed)])
    rows = np.arange(n)
    tr_null = np.empty(int(n_perm), dtype=np.float64)
    fro_null = np.empty(int(n_perm), dtype=np.float64)
    for i in range(int(n_perm)):
        pi = rng.permutation(n)
        tr_null[i] = float(K_np[rows, pi].sum()) / nm1
        fro_null[i] = math.sqrt(
            max(float((A_np * B_np[np.ix_(pi, pi)]).sum()), 0.0)
        ) / nm1

    # (1 + hits) / (1 + n_perm): the finite-sample p-value, which never
    # returns an impossible 0.
    p_two = (1.0 + float((np.abs(tr_null) >= abs(tr_obs)).sum())) / (
        1.0 + float(n_perm)
    )
    p_gt = (1.0 + float((tr_null >= tr_obs).sum())) / (1.0 + float(n_perm))
    fro_med = float(np.median(fro_null))
    return {
        "perm_p_trace": p_two,
        "perm_p_trace_greater": p_gt,
        "frob_null_median": fro_med,
        "frob_excess": math.sqrt(max(fro_obs ** 2 - fro_med ** 2, 0.0)),
    }


class CrossCovAccumulator:
    r"""Per-snapshot cross-covariance of iterate and gradient.

    The accumulator stores one projected matrix per snapshot in
    chronological order; the caller (``train_ebm`` or the CP-Flow
    trainer) decides the cadence. The estimate at iter ``t`` is

    .. math::

        \widehat{\Sigma}_{\mathrm{slack}}(t)
        \;\equiv\;
        \frac{1}{B - 1}\sum_{b=1}^{B}\bigl(
            \widetilde{\theta}_b - \overline{\widetilde{\theta}}
        \bigr)\bigl(
            \partial_{\mathbf{b}_h}\ell_b
            - \overline{\partial_{\mathbf{b}_h}\ell}
        \bigr)^{\!\top}\!,

    where the sample index ``b`` runs over the probe batches drawn at the
    fixed iterate ``t``, ``theta-tilde_b`` is the pre-positivity emission
    that probe batch elicits from the hypernetwork, and ``d/d b_h ell_b``
    is the loss gradient on the readout bias for the same probe batch. The
    iterate is the left factor and the gradient the right one, so the
    product is not symmetric, which is the point.

    The divisor is ``B - 1``, not ``B``: both factors are centered at their
    own sample mean, so ``1/B`` is biased low by ``(B-1)/B`` and only
    ``1/(B-1)`` is unbiased.

    The two sequences must be index-aligned, with entry ``j`` of the
    iterate row and entry ``j`` of the gradient row referring to the same
    scalar readout channel. Callers build both by concatenating the same
    heads in the same order and assert that the per-head widths match; see
    :mod:`lift.objectives.cross_cov_taps`.

    ``is_pgd`` is provenance only: it is written to the archive and changes
    no arithmetic. A construction with no batch-conditioned emission,
    direct softplus or PGD, runs the identical measurement code and returns
    a zero matrix by measurement, because its iterate is a parameter that
    does not move across probe batches at a fixed iterate. That zero is a
    measured result, not a hard-coded sentinel.

    That zero is a float32 zero rather than a symbolic one: the mean of
    ``B`` identical float32 values is exact only for special ``B``, so the
    centered block still holds about one ulp of round-off. Do not report it
    as exactly zero; ``iterate_rel_var`` is the field that says how dead
    the channel is.
    """

    def __init__(
        self,
        *,
        d_bias: int,
        snap_iters: List[int],
        is_pgd: bool = False,
        jl_proj_dim: Optional[int] = None,
        device: torch.device | str = "cpu",
        iterate_source: str = "unspecified",
        on_degenerate: str = "raise",
        n_permutations: int = 200,
        probe_protocol: str = "unspecified",
        max_eff_d: int = MAX_EFF_D,
    ) -> None:
        if on_degenerate not in ("raise", "warn"):
            raise ValueError(
                f"on_degenerate must be 'raise' or 'warn'; got "
                f"{on_degenerate!r}"
            )
        self.d_bias = int(d_bias)
        self.snap_iters = list(snap_iters)
        self.is_pgd = bool(is_pgd)
        self.device = torch.device(device)
        self.jl_proj_dim = int(jl_proj_dim) if jl_proj_dim else None
        # Free-text label for what the left slot was filled from, such as
        # ``hypernet_emitted_pre_positivity`` or
        # ``direct_pos_weight_parameter``. Persisted so a reader of the NPZ
        # alone can tell which quantity was measured.
        self.iterate_source = str(iterate_source)
        # Free-text label for how the probe loss was computed, such as
        # ``stochastic_logdet_m1=10`` against ``exact_logdet``. Estimator noise
        # does not bias Sigma but does inflate its variance, so which one ran
        # is recorded rather than inferred.
        self.probe_protocol = str(probe_protocol)
        self.on_degenerate = str(on_degenerate)
        self.n_permutations = max(int(n_permutations), 0)
        self._warned_degenerate = False

        if self.jl_proj_dim is not None and self.jl_proj_dim < self.d_bias:
            g = torch.Generator(device="cpu").manual_seed(0)
            self.jl = torch.randn(
                self.d_bias, self.jl_proj_dim, generator=g,
            ) / float(self.jl_proj_dim) ** 0.5
            self.jl = self.jl.to(self.device)
            self._eff_d = self.jl_proj_dim
        else:
            self.jl = None
            self._eff_d = self.d_bias
        if self._eff_d > int(max_eff_d):
            raise ValueError(
                f"the persisted matrix would be {self._eff_d} x "
                f"{self._eff_d} float32 "
                f"({4 * self._eff_d ** 2 / 2 ** 30:.1f} GB per snapshot). "
                f"Set jl_proj_dim <= {max_eff_d}, or raise max_eff_d "
                "deliberately. Failing at construction so a 17-hour run "
                "does not fail 10k iterations in."
            )

        self.matrices: List[np.ndarray] = []
        self.recorded_iters: List[int] = []
        # ---- Exact scalars, computed on the unprojected (B, d) blocks:
        # ``sigma_Jac^2 = tr(Sigma_slack)`` and ``||Sigma_slack||_F``. Reading
        # these off the JL projection is unbiased but far noisier than the
        # trace itself, so the projection only draws the picture.
        self.sigma_Jac_sq: List[float] = []
        self.frob: List[float] = []
        # ``asymmetry`` is ``||M - M^T||_F / ||M||_F``, and it is not a signal
        # check: at ``B << d_bias`` the sample matrix has rank at most B-1, so
        # ``tr(Sigma^2)`` is near zero and the asymmetry saturates at sqrt(2)
        # whether or not anything was measured. What it discriminates is an
        # auto-covariance, which reads near 0. Whether anything was measured is
        # answered by ``iterate_rel_var``, and whether the coupling is real by
        # ``perm_p_trace``. It is NaN when the estimate is exactly zero, since
        # 0.0 there would collide with the auto-covariance signature this field
        # exists to expose.
        self.asymmetry: List[float] = []
        # Per-channel mean variance ``tr(Cov)/d_bias``, over the true width
        # rather than the projected one.
        self.iterate_var: List[float] = []
        self.grad_var: List[float] = []
        # Dimensionless: the share of each channel's second moment that is
        # probe-to-probe fluctuation. This separates a tap that moved from a
        # constant read through a float32 mean subtraction, which the absolute
        # variances cannot, since round-off on a constant sequence clears any
        # absolute threshold.
        self.iterate_rel_var: List[float] = []
        self.grad_rel_var: List[float] = []
        self.n_samples: List[int] = []
        # What the JL projection alone would have reported, kept so the size
        # of the projection error is visible in the artifact.
        self.sigma_Jac_sq_proj: List[float] = []
        self.frob_proj: List[float] = []
        # Permutation null on the probe pairing; empty when
        # ``n_permutations == 0``.
        self.perm_p_trace: List[float] = []
        self.perm_p_trace_greater: List[float] = []
        self.frob_null_median: List[float] = []
        self.frob_excess: List[float] = []

    # ------------------------------------------------------------- guards
    def _flag(self, exc: CrossCovDegenerateInputs) -> None:
        if self.on_degenerate == "raise":
            raise exc
        if not self._warned_degenerate:
            self._warned_degenerate = True
            warnings.warn(str(exc), RuntimeWarning, stacklevel=3)

    def _check_distinct(
        self, left_c: torch.Tensor, right_c: torch.Tensor,
    ) -> None:
        """Refuse an alias, or flag it under ``on_degenerate='warn'``.

        Runs only on snapshot iterations, and costs two norms over the
        ``(B, d_bias)`` block.
        """
        rel = _alias_defect_rel(left_c, right_c)
        if rel is None:
            return
        self._flag(CrossCovDegenerateInputs(
            "CrossCovAccumulator.record: " + _alias_defect_message(rel)
        ))

    def _check_asymmetric(self, frob: float, asym: float) -> None:
        """Refuse a nonzero estimate that came out symmetric.

        Exempt when ``d_bias < 2``: a 1x1 cross-covariance equals its own
        transpose for arithmetic reasons, and a single-channel readout is a
        real configuration, as on the 1-D log-concave targets.
        """
        if self.d_bias < 2:
            return
        if not (frob > 0.0) or not math.isfinite(asym):
            return
        if asym >= ASYMMETRY_FLOOR:
            return
        self._flag(CrossCovSymmetricEstimate(
            "CrossCovAccumulator.record: " + _symmetric_estimate_message(asym)
        ))

    def record(
        self,
        *,
        iter_idx: int,
        iterate_batch: torch.Tensor,
        grad_batch: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        """Persist a snapshot; return its scalars (``None`` if skipped).

        Args:
            iter_idx: training iteration this snapshot belongs to. A
                value outside ``snap_iters`` is a no-op.
            iterate_batch: ``(B, d_bias)``, the left factor. Row ``b`` is
                the pre-positivity emitted iterate
                ``theta-tilde = Theta_E h_phi(X_b) + b_h`` elicited by
                probe batch ``b``, flattened and concatenated over the
                positivity-tagged heads.
            grad_batch: ``(B, d_bias)``, the right factor. Row ``b`` is the
                gradient of probe batch ``b``'s loss with respect to the
                readout biases ``b_h``, flattened and concatenated over the
                same heads in the same order.

        Both are centered here. The persisted matrix is the JL projection
        of ``iterate_c^T @ grad_c / (B - 1)``, so entry ``(i, j)`` is
        ``Cov(theta-tilde_i, g_j)``: the row index is the iterate, the
        column index the gradient. Every persisted scalar is computed
        exactly, without the projection.

        Raises:
            CrossCovNonFinite: if either block carries a NaN or inf.
            CrossCovDegenerateInputs: if the two arguments are the same
                sequence, or if the resulting estimate is symmetric,
                unless the accumulator was built with
                ``on_degenerate='warn'``.
        """
        if iter_idx not in self.snap_iters:
            return None

        if iterate_batch.shape != grad_batch.shape:
            raise ValueError(
                "iterate_batch and grad_batch must be index-aligned and "
                f"share a shape; got {tuple(iterate_batch.shape)} vs "
                f"{tuple(grad_batch.shape)}"
            )
        if iterate_batch.dim() != 2 or int(iterate_batch.shape[0]) < 2:
            raise ValueError(
                "cross-covariance needs a (B, d) block with B >= 2 probe "
                f"batches; got {tuple(iterate_batch.shape)}"
            )
        if int(iterate_batch.shape[1]) != self.d_bias:
            raise ValueError(
                f"the accumulator was built for d_bias={self.d_bias} but "
                f"got blocks of width {int(iterate_batch.shape[1])}. The "
                "JL projection and every persisted matrix would be the "
                "wrong size."
            )

        with torch.no_grad():
            j = iterate_batch.detach().to(self.device).float()
            g = grad_batch.detach().to(self.device).float()
            _check_finite_or_raise(j, g, where="CrossCovAccumulator.record")
            j_mean = j.mean(dim=0, keepdim=True)
            g_mean = g.mean(dim=0, keepdim=True)
            jc = j - j_mean
            gc = g - g_mean
            self._check_distinct(jc, gc)
            n = int(j.shape[0])
            nm1 = float(n - 1)

            # ---- exact scalars, via the three B x B Grams ----
            A, Bm, K = _gram_triple(jc, gc)
            tr_exact = float(torch.diagonal(K).sum()) / nm1
            frob_sq = float((A * Bm).sum()) / nm1 ** 2
            tr_sq = float((K * K.T).sum()) / nm1 ** 2
            frob = math.sqrt(max(frob_sq, 0.0))
            asym_sq = max(2.0 * frob_sq - 2.0 * tr_sq, 0.0)
            asymmetry = (
                math.sqrt(asym_sq) / frob if frob > 0.0 else float("nan")
            )
            self._check_asymmetric(frob, asymmetry)

            sum_jc2 = float(torch.diagonal(A).sum())
            sum_gc2 = float(torch.diagonal(Bm).sum())
            var_j = sum_jc2 / (nm1 * float(self.d_bias))
            var_g = sum_gc2 / (nm1 * float(self.d_bias))
            # sum(x^2) = sum(xc^2) + n * ||xbar||^2 -- no second pass.
            sum_j2 = sum_jc2 + n * float(j_mean.double().pow(2).sum())
            sum_g2 = sum_gc2 + n * float(g_mean.double().pow(2).sum())
            rel_j = sum_jc2 / sum_j2 if sum_j2 > 0.0 else float("nan")
            rel_g = sum_gc2 / sum_g2 if sum_g2 > 0.0 else float("nan")

            perm = (
                _permutation_null(
                    A, Bm, K, n_perm=self.n_permutations, nm1=nm1,
                    seed=int(iter_idx),
                )
                if self.n_permutations > 0
                else {
                    "perm_p_trace": float("nan"),
                    "perm_p_trace_greater": float("nan"),
                    "frob_null_median": float("nan"),
                    "frob_excess": float("nan"),
                }
            )

            # ---- the picture: JL-projected (k, k) heatmap ----
            # Projection and centering commute, so this is the projection of
            # the same estimate.
            jp = jc @ self.jl if self.jl is not None else jc
            gp = gc @ self.jl if self.jl is not None else gc
            mat_t = (jp.T @ gp) / nm1
            mat = mat_t.detach().cpu().numpy().astype(np.float32)

        fro_proj = float(np.linalg.norm(mat, ord="fro"))
        self.matrices.append(mat)
        self.recorded_iters.append(int(iter_idx))
        self.sigma_Jac_sq.append(tr_exact)
        self.frob.append(frob)
        self.asymmetry.append(asymmetry)
        self.iterate_var.append(var_j)
        self.grad_var.append(var_g)
        self.iterate_rel_var.append(rel_j)
        self.grad_rel_var.append(rel_g)
        self.n_samples.append(n)
        self.sigma_Jac_sq_proj.append(float(np.trace(mat)))
        self.frob_proj.append(fro_proj)
        self.perm_p_trace.append(perm["perm_p_trace"])
        self.perm_p_trace_greater.append(perm["perm_p_trace_greater"])
        self.frob_null_median.append(perm["frob_null_median"])
        self.frob_excess.append(perm["frob_excess"])
        return {
            "sigma_Jac_sq": tr_exact,
            "frob": frob,
            "asymmetry": asymmetry,
            "iterate_var": var_j,
            "grad_var": var_g,
            "iterate_rel_var": rel_j,
            "grad_rel_var": rel_g,
            "sigma_Jac_sq_proj": float(np.trace(mat)),
            "frob_proj": fro_proj,
            **perm,
        }

    def to_npz(self, path: str) -> None:
        """Save the persisted state as a single NPZ archive.

        Schema ``CROSS_COV_SCHEMA_VERSION``. Schema-1 archives carry none
        of the provenance fields and hold an auto-covariance read off a JL
        projection; the ``left_slot``, ``right_slot`` and ``quantity``
        strings and the per-snapshot ``asymmetry`` keep that distinction
        readable from the archive itself.
        """
        np.savez(
            path,
            matrices=np.stack(self.matrices, axis=0) if self.matrices else
                np.zeros((0, self._eff_d, self._eff_d), dtype=np.float32),
            iters=np.array(self.recorded_iters, dtype=np.int64),
            sigma_Jac_sq=np.array(self.sigma_Jac_sq, dtype=np.float64),
            frob=np.array(self.frob, dtype=np.float64),
            is_pgd=np.array(int(self.is_pgd), dtype=np.int8),
            d_bias=np.array(int(self.d_bias), dtype=np.int64),
            jl_proj_dim=np.array(
                -1 if self.jl is None else int(self.jl_proj_dim),
                dtype=np.int64,
            ),
            # ---- provenance ----
            schema_version=np.array(
                CROSS_COV_SCHEMA_VERSION, dtype=np.int64,
            ),
            quantity=np.array("Sigma_slack = E[d(theta_tilde) d(g)^T]"),
            left_slot=np.array("iterate_fluctuation"),
            right_slot=np.array("grad_fluctuation"),
            iterate_source=np.array(self.iterate_source),
            probe_protocol=np.array(self.probe_protocol),
            estimator_normalisation=np.array("1/(B-1) (unbiased)"),
            scalars_are_exact=np.array(1, dtype=np.int8),
            asymmetry=np.array(self.asymmetry, dtype=np.float64),
            iterate_var=np.array(self.iterate_var, dtype=np.float64),
            grad_var=np.array(self.grad_var, dtype=np.float64),
            iterate_rel_var=np.array(self.iterate_rel_var, dtype=np.float64),
            grad_rel_var=np.array(self.grad_rel_var, dtype=np.float64),
            n_samples=np.array(self.n_samples, dtype=np.int64),
            # ---- what the projection alone would have said ----
            sigma_Jac_sq_proj=np.array(
                self.sigma_Jac_sq_proj, dtype=np.float64,
            ),
            frob_proj=np.array(self.frob_proj, dtype=np.float64),
            # ---- permutation null on the probe pairing ----
            n_permutations=np.array(int(self.n_permutations), dtype=np.int64),
            perm_p_trace=np.array(self.perm_p_trace, dtype=np.float64),
            perm_p_trace_greater=np.array(
                self.perm_p_trace_greater, dtype=np.float64,
            ),
            frob_null_median=np.array(
                self.frob_null_median, dtype=np.float64,
            ),
            frob_excess=np.array(self.frob_excess, dtype=np.float64),
        )


# ------------------------------------------------------------ Moreau probe


class MoreauProbe:
    """Trajectory-resolved measurement of ``mu_eff``.

    Each call to :meth:`record` consumes one ``(B, d)`` probe block pair
    and returns the two ingredients, ``sigma_Jac_sq`` and ``kappa``, plus
    ``mu_eff_hat = sigma_Jac^2 / (d kappa^2)``, where the ``1/d`` is part
    of the definition. The series is persisted in the trainer's metric
    dict.

    The measurement forms only the trace of the cross-covariance, so the
    per-iteration cost is ``O(B * d_bias)`` rather than the
    ``O(B^2 * d_bias)`` of the full :class:`CrossCovAccumulator`.

    The slot contract and the guards of :class:`CrossCovAccumulator`
    apply here too. In particular ``B >= 2`` is required, since a single
    probe would return a measured zero for an absent measurement.
    """

    def __init__(self, *, n_hutchinson_probes: int = 1,
                 on_degenerate: str = "raise") -> None:
        if on_degenerate not in ("raise", "warn"):
            raise ValueError(
                f"on_degenerate must be 'raise' or 'warn'; got "
                f"{on_degenerate!r}"
            )
        self.n_probes = int(n_hutchinson_probes)
        self.on_degenerate = str(on_degenerate)
        self._warned_degenerate = False
        self.iters: List[int] = []
        self.sigma_Jac_sq: List[float] = []
        self.kappa: List[float] = []
        self.mu_eff_hat: List[float] = []

    @staticmethod
    def kappa_from_emission(emission: torch.Tensor) -> float:
        r"""``kappa = ||theta-tilde||_inf`` from a captured emission.

        This is the reading to use on the lift: the readout scale is a
        property of the emitted pre-positivity tensor, which is not in
        ``state_dict``, so :meth:`_kappa_from_state_dict` cannot find it.
        :meth:`lift.models.hypernet.HyperNetwork.capture_pre_positivity`
        supplies exactly this tensor.
        """
        return max(float(emission.detach().abs().max()), 1e-12)

    @staticmethod
    def _kappa_from_state_dict(
        sd: Mapping[str, torch.Tensor],
        pos_param_names: Optional[set[str]] = None,
    ) -> float:
        """Compute kappa = ||W_tilde||_inf over the positivity-tagged tensors.

        When ``pos_param_names`` is None this defaults to every
        ``z_layers.*.weight`` and ``output_layer.weight`` key, the
        canonical ICNN positivity set of Amos 2017.

        Raises if none of the requested names are present, which is what
        happens on the lift, where the positivity-tagged tensors are
        emitted and therefore absent from ``state_dict``; use
        :meth:`kappa_from_emission` there.
        """
        if pos_param_names is None:
            pos_param_names = {
                k for k in sd
                if (k.endswith(".weight") and (
                    "z_layers" in k or "output_layer" in k
                ))
            }
        present = [n for n in pos_param_names if n in sd]
        if not present:
            raise KeyError(
                "none of the positivity-tagged names are in this "
                "state_dict, so kappa cannot be read from it. On a "
                "lifted model the tensors are EMITTED, not stored: use "
                "MoreauProbe.kappa_from_emission on the captured "
                f"pre-positivity emission instead. Looked for: "
                f"{sorted(pos_param_names)[:4]}"
            )
        m = 0.0
        for name in present:
            m = max(m, float(sd[name].detach().abs().max().item()))
        return max(m, 1e-12)

    def record(
        self,
        *,
        iter_idx: int,
        iterate_batch: torch.Tensor,
        grad_batch: torch.Tensor,
        kappa: float,
    ) -> Dict[str, float]:
        """Persist a single trajectory entry.

        ``iterate_batch`` and ``grad_batch`` carry the same contract as
        :meth:`CrossCovAccumulator.record`: left is the emitted
        pre-positivity iterate, right is the readout-bias gradient, and
        the two must be index-aligned and genuinely different sequences.
        """
        if iterate_batch.shape != grad_batch.shape:
            raise ValueError(
                "iterate_batch and grad_batch must be index-aligned and "
                f"share a shape; got {tuple(iterate_batch.shape)} vs "
                f"{tuple(grad_batch.shape)}"
            )
        if iterate_batch.dim() != 2 or int(iterate_batch.shape[0]) < 2:
            raise ValueError(
                "cross-covariance needs a (B, d) block with B >= 2 probe "
                f"batches; got {tuple(iterate_batch.shape)}"
            )
        with torch.no_grad():
            j = iterate_batch.detach().float()
            g = grad_batch.detach().float()
            _check_finite_or_raise(j, g, where="MoreauProbe.record")
            jc = j - j.mean(dim=0, keepdim=True)
            gc = g - g.mean(dim=0, keepdim=True)
            rel = _alias_defect_rel(jc, gc)
            if rel is not None:
                exc = CrossCovDegenerateInputs(
                    "MoreauProbe.record: " + _alias_defect_message(rel)
                )
                if self.on_degenerate == "raise":
                    raise exc
                if not self._warned_degenerate:
                    self._warned_degenerate = True
                    warnings.warn(str(exc), RuntimeWarning, stacklevel=3)
            d = int(j.shape[1])
            nm1 = float(int(j.shape[0]) - 1)
            # tr(jc^T gc / (B-1)) on the unprojected block: exact, O(B d),
            # and signed.
            sigma_Jac_sq = float(
                (jc.double() * gc.double()).sum() / nm1
            )
        # Not clamped at zero: sigma_Jac^2 is the trace of a cross-covariance
        # and carries a sign, so clamping would make positivity true by
        # construction.
        kappa = float(max(kappa, 1e-12))
        mu_eff_hat = sigma_Jac_sq / (float(d) * kappa ** 2)
        self.iters.append(int(iter_idx))
        self.sigma_Jac_sq.append(sigma_Jac_sq)
        self.kappa.append(kappa)
        self.mu_eff_hat.append(mu_eff_hat)
        return {
            "sigma_Jac_sq": sigma_Jac_sq,
            "kappa": kappa,
            "mu_eff_hat": mu_eff_hat,
        }

    def to_npz(self, path: str) -> None:
        """Save the persisted state as a single NPZ archive."""
        np.savez(
            path,
            iters=np.array(self.iters, dtype=np.int64),
            sigma_Jac_sq=np.array(self.sigma_Jac_sq, dtype=np.float64),
            kappa=np.array(self.kappa, dtype=np.float64),
            mu_eff_hat=np.array(self.mu_eff_hat, dtype=np.float64),
            schema_version=np.array(
                CROSS_COV_SCHEMA_VERSION, dtype=np.int64,
            ),
            quantity=np.array("mu_eff = tr(Sigma_slack) / (d kappa^2)"),
        )


# ---------------------------------------- per-sample gradient extractors


def per_sample_bias_grads(
    hypernet: nn.Module,
    target_param_names: List[str],
    per_sample_losses: torch.Tensor,
) -> torch.Tensor:
    r"""Per-sample loss gradients on the hypernet's readout biases.

    Returns one ``(B, d_bias)`` tensor: row ``b`` is the gradient of
    ``per_sample_losses[b]`` with respect to the readout biases
    ``b_h``, flattened and concatenated over the readouts whose target
    name is in ``target_param_names``, in ``hypernet._param_names``
    order.

    .. warning::

        This is the right factor of ``Sigma_slack`` only. Do not
        manufacture the left factor from this return value by any
        arithmetic: mean-removing this tensor and pairing it with itself
        measures ``Cov(g)``, an auto-covariance.

    The left factor is the emitted pre-positivity iterate, a property of
    the forward pass that cannot be recovered from gradients. Capture it
    with
    :meth:`lift.models.hypernet.HyperNetwork.capture_pre_positivity`
    around the forward whose loss is differentiated here.

    The hypernetwork emits one weight set per pooled conditioning set, so
    there is no per-sample iterate to pair with these rows: the sample
    axis of a fixed-iterate cross-covariance is probe batches, not
    examples within a batch.

    The per-sample-loss decomposition is supplied by the caller, so this
    function assumes nothing about which side of the estimator, sampler or
    EBM, the gradient lives on.
    """
    if not hasattr(hypernet, "weight_predictors"):
        raise RuntimeError(
            "per_sample_bias_grads expects a HyperNetwork with a "
            "`weight_predictors` ModuleList; got "
            f"{type(hypernet).__name__}."
        )

    bias_params = []
    for name, predictor in zip(
        hypernet._param_names, hypernet.weight_predictors,
    ):
        if name in target_param_names and predictor.bias is not None:
            bias_params.append(predictor.bias)
    if not bias_params:
        raise RuntimeError(
            "no positivity-tagged readouts have bias parameters; the "
            "cross-covariance accumulator has no targets to attach to."
        )

    B = int(per_sample_losses.shape[0])
    grads_per_sample = []
    for b in range(B):
        gs = torch.autograd.grad(
            per_sample_losses[b],
            bias_params,
            retain_graph=(b < B - 1),
            create_graph=False,
            allow_unused=True,
        )
        flat = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros_like(
                p.reshape(-1)
            )
            for g, p in zip(gs, bias_params)
        ])
        grads_per_sample.append(flat.detach())
    return torch.stack(grads_per_sample, dim=0)


__all__ = [
    "ALIAS_RTOL",
    "ASYMMETRY_FLOOR",
    "CROSS_COV_SCHEMA_VERSION",
    "MAX_EFF_D",
    "CrossCovAccumulator",
    "CrossCovDegenerateInputs",
    "CrossCovNonFinite",
    "CrossCovSymmetricEstimate",
    "MoreauProbe",
    "per_sample_bias_grads",
]
