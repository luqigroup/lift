import Mathlib
import CrossCov
import CrossCovIndep
import CrossCovSLLN

/-!
# Theorem 1 (`thm:joint-necessity`) for the fixed-iterate probe estimator of `eq:cross-cov`

The v5 manuscript (`docs/paper/v5/sections/03b_mechanism.tex`, `eq:cross-cov`) defines the
cross-covariance estimator on the **fixed-iterate probe**: at one iterate the body parameters and
the slack are held fixed, `T` fresh batches are sampled, the latent weight `θ̃_k` and the gradient `g_k`
are recorded on each, and the pairs are centred at their sample means and averaged with Bessel's
divisor,

  `Σ̂⁽ᵗ⁾ = (T − 1)⁻¹ ∑_{k=1}^{T} δθ̃_k δg_kᵀ`,  `δθ̃_k = θ̃_k − θ̃̄`,  `δg_k = g_k − ḡ`.

The earlier development (`CrossCov.lean`, `CrossCovSLLN.lean`) indexes a trailing window of
training steps `t − T + s` with the divisor `1/T`. Theorem 1 of the paper is now stated for the
probe, with the quantifier "for every iterate and every probe count `T`", and it adds two
clauses: that the theorem is indifferent to the index set of the probes, and that it is
indifferent to the divisor. This file supplies the probe-indexed statements those clauses point
to, so that every pointer in the appendix is exact rather than transported by hand.

**What is proved.** The estimator is `probeCrossCovWith c T`, the sum of the `T` outer products
with an arbitrary divisor `c`; `probeCrossCov` is the paper's Bessel divisor `(T − 1)⁻¹` and
`probeCrossCovMean` the divisor `T⁻¹`. Cases (i) and (ii) of Theorem 1 hold identically for every
probe count and every divisor, exactly as in `CrossCov.lean`: a vanishing slack Jacobian zeroes the
bias-channel reading, and a batch-independent body has zero sample-mean fluctuation and hence a
zero estimator (`probeSlackReading_eq_zero_of_jacobian_eq_zero`,
`probeCrossCov_eq_zero_of_body_const`, `probe_joint_necessity_finite`). Case (iii) is the
population statement of `CrossCovIndep.lean`, instantiated probe by probe: the expectation of the
estimator is the divisor times the sum of the per-probe population cross-covariances
(`integral_probeCrossCovWith_entry`), so when each of those is zero — which conditional
independence of the latent weight and the reference-iterate gradient supplies through
`popCrossCov_eq_zero_of_indepFun_of_centered` — the estimator is unbiased for the zero target with
**every** divisor (`probeCrossCovWith_unbiased_of_popCrossCov_eq_zero`,
`probeCrossCovWith_unbiased_of_indepFun`); this is the "(T − 1)/T rescaling of zero" of the theory
contract, made a theorem. The sample-mean-centred estimator the paper actually forms obeys the
sample-cross-covariance identity `probe_sample_crossCov_identity`, exact finite algebra with the
factor `c · T` on the mean-product term, and it is unbiased for the zero target as well, again
with every divisor, once the two probe collections are independent
(`probeCrossCovWith_centred_unbiased_of_indepFun`).

**Bessel's correction is a theorem here, not a convention.** For a nonzero population
cross-covariance `Σ` the two divisors differ: with per-probe second moments equal to `Σ` and
cross-probe second moments zero, the expectation of the sample-mean-centred estimator with divisor
`c` is exactly `c (T − 1) Σ` (`probeCrossCovWith_centred_expectation_of_iid`), so the Bessel
divisor is the unbiased one (`probeCrossCov_centred_unbiased_of_iid`) and the divisor `T⁻¹`
returns `(T − 1)/T · Σ` (`probeCrossCovMean_centred_expectation_of_iid`). The paper's sentence
"the divisor `T − 1` is Bessel's correction, and Theorem 1 holds with `T` in its place" is these
two theorems together with the zero-target results: the two divisors agree exactly when the target
is zero, which is the only case Theorem 1 speaks about.

**The two estimators on the same data.** On a common family of `T` pairs, the probe estimator
with Bessel's divisor is `T/(T − 1)` times the step-window estimator of `CrossCov.lean` with its
divisor `1/T` (`probeCrossCov_eq_smul_crossCovEstimator`, for every window position `t`, and
`probeCrossCov_prefix_eq_smul_crossCovEstimator` for the prefix), and the two probe divisors differ
by the same factor (`probeCrossCov_eq_smul_probeCrossCovMean`). Every statement of `CrossCov.lean`
and `CrossCovSLLN.lean` therefore transports to the probe with an explicit scalar, and the `L²`
rate does so with the factor squared: the mean square of an entry of the Bessel-divisor estimator
built from centred i.i.d. probes is `T σ² / (T − 1)²` (`integral_sq_probeCrossCov_entry`), so its
`L²` norm is `√T σ / (T − 1)`, which is the paper's `O(T^{-1/2})` in the only form this mathlib
can carry (`probeCrossCov_entry_l2_rate`). The almost-sure limit transports as well
(`probeCrossCov_centred_tendsto_zero_ae`): as the probe count grows, the sample-mean-centred
probe estimator converges almost surely to the zero matrix, because the factor `T/(T − 1)` tends
to one.

**The conditioning.** As in `CrossCovSLLN.lean`, the measure `μ` is the conditional law given the
iterate at which the probe is taken: the body parameters and the slack are fixed, so every
expectation below is the paper's `E_X[·]` at that iterate, and the independence hypotheses are the
paper's conditional independence there. Nothing in the fixed-iterate probe requires more, and the
probes are i.i.d. by construction, which is why the collection-level independence hypotheses are
natural for it.

## Results
* `probeCrossCovWith`, `probeCrossCov`, `probeCrossCovMean` — the probe estimator with an
  arbitrary divisor, with Bessel's divisor (`eq:cross-cov`), and with the divisor `T⁻¹`.
* `probeMean`, `probeCentre`, `probeBodyFluct`, `probeSlackReading` — the sample mean, the
  sample-mean centring, the body's fluctuation across probes, and the bias-channel reading.
* `probeSlackReading_eq_zero_of_jacobian_eq_zero` — case (i): no slack, no reading, every `T`.
* `probeCrossCovWith_eq_zero_of_fluct_eq_zero`, `probeBodyFluct_eq_zero_of_const`,
  `probeCrossCov_eq_zero_of_body_const` — case (ii): a batch-independent body zeroes the
  estimator identically, every `T`, every divisor.
* `probe_joint_necessity_finite` — the shared contrapositive.
* `probeCrossCov_eq_smul_probeCrossCovMean` — the two divisors differ by `T/(T − 1)`.
* `probeCrossCov_eq_smul_crossCovEstimator`, `probeCrossCov_prefix_eq_smul_crossCovEstimator` —
  the probe estimator is `T/(T − 1)` times the step-window estimator on the same data.
* `probe_centred_sum_identity`, `probe_sample_crossCov_identity`,
  `probeCrossCov_sample_identity` — the sample-cross-covariance identity for the probe.
* `probeCentre_prefix_eq_windowCentre`, `probeCrossCov_centred_prefix_eq` — sample-mean centring
  over the probes is window-mean centring over the prefix.
* `integral_probeCrossCovWith_entry` — the expectation is the divisor times the sum of the
  per-probe population cross-covariances.
* `probeCrossCovWith_unbiased_of_popCrossCov_eq_zero`, `probeCrossCovWith_unbiased_of_indepFun`,
  `probeCrossCov_unbiased_of_indepFun`, `probeCrossCovMean_unbiased_of_indepFun` — case (iii):
  unbiased for the zero target with either divisor.
* `indepFun_probeMean_probeMean`, `integral_probeMean_mul_eq_zero`,
  `probeCrossCovWith_centred_unbiased_of_indepFun` — the same for the sample-mean-centred
  estimator, every divisor.
* `integral_probeMean_mul_of_iid`, `probeCrossCovWith_centred_expectation_of_iid`,
  `probeCrossCov_centred_unbiased_of_iid`, `probeCrossCovMean_centred_expectation_of_iid` —
  Bessel's correction for a nonzero target.
* `integral_sq_probeCrossCov_entry`, `probeCrossCov_entry_l2_rate` — the `L²` form of
  `O(T^{-1/2})` for the Bessel-divisor estimator.
* `tendsto_bessel_factor` , `probeCrossCov_centred_tendsto_zero_ae` — the almost-sure limit.
* `probeCrossCov_centred_ne_zero` — the probe estimator of case (iii) is not identically zero.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The hypotheses carried are those of the modules this file instantiates: integrability of the
coordinates and their products, which the Bochner integral needs; per-probe or collection-level
independence, which is case (iii)'s conditional independence at the fixed iterate; and, for the
`L²` and almost-sure statements, the `IsSquareCentredIID` and `IsCentredIID` packages of
`CrossCovSLLN.lean`, which are (A1)'s i.i.d. batches together with the population zero. The
Bessel theorems take the per-probe second moments and the vanishing of the cross-probe second
moments as hypotheses; for population-centred i.i.d. probes both follow from independence and
centring exactly as in `CrossCovIndep.lean`, and `integral_mul_eq_zero_of_indepFun_of_centered`
records that derivation. The divisor `(T − 1)⁻¹` is Lean's `0` at `T = 1`, so statements at
`T = 1` are vacuous and are excluded by the hypotheses `2 ≤ T` where the Bessel value is asserted;
the identities hold for every `T` and say so.
-/

open Matrix Finset MeasureTheory ProbabilityTheory Filter Topology

namespace IcnnLift

variable {d : ℕ} {Ω : Type*}

/-! ### The probe estimator -/

section Definitions

/-- The probe-indexed cross-covariance estimator with an arbitrary divisor `c`: the sum over the
`T` probes at one iterate of the outer products `δθ̃_k δg_kᵀ`, multiplied by `c`. The paper's
`eq:cross-cov` is the case `c = (T − 1)⁻¹`; the divisor `T⁻¹` is the other case the theorem
mentions. There is no window offset: the probes are `T` fresh batches at a single iterate. -/
noncomputable def probeCrossCovWith (c : ℝ) (T : ℕ) (dθ dg : Fin T → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  c • ∑ k : Fin T, Matrix.vecMulVec (dθ k) (dg k)

/-- The estimator of `eq:cross-cov`, with Bessel's divisor `(T − 1)⁻¹`. -/
noncomputable def probeCrossCov (T : ℕ) (dθ dg : Fin T → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  probeCrossCovWith ((T : ℝ) - 1)⁻¹ T dθ dg

/-- The same estimator with the divisor `T⁻¹`, which Theorem 1 says may replace Bessel's. -/
noncomputable def probeCrossCovMean (T : ℕ) (dθ dg : Fin T → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  probeCrossCovWith (T : ℝ)⁻¹ T dθ dg

/-- The sample mean over the `T` probes. -/
noncomputable def probeMean (T : ℕ) (u : Fin T → Fin d → ℝ) : Fin d → ℝ :=
  (T : ℝ)⁻¹ • ∑ k : Fin T, u k

/-- The sample-mean centring `u_k − ū` of `eq:cross-cov`. -/
noncomputable def probeCentre (T : ℕ) (u : Fin T → Fin d → ℝ) : Fin T → Fin d → ℝ :=
  fun k => u k - probeMean T u

/-- The body's fluctuation across the probes, `eq:fluct` read on the probe: the latent weight on the
`k`-th batch minus the sample mean over the probes. The slack is common to every probe and
cancels, so the body carries the entire fluctuation. -/
noncomputable def probeBodyFluct (h : Ω → Fin d → ℝ) {T : ℕ} (X : Fin T → Ω) :
    Fin T → Fin d → ℝ :=
  probeCentre T fun k => h (X k)

/-- The bias-channel reading of the probe estimator: its contraction with the slack Jacobian
`J_b`. -/
noncomputable def probeSlackReading (Jb : Matrix (Fin d) (Fin d) ℝ) (T : ℕ)
    (dθ dg : Fin T → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Jbᵀ * probeCrossCov T dθ dg

@[simp] lemma probeMean_apply (T : ℕ) (u : Fin T → Fin d → ℝ) (i : Fin d) :
    probeMean T u i = (T : ℝ)⁻¹ * ∑ k : Fin T, u k i := by
  simp [probeMean, Finset.sum_apply]

lemma probeCrossCovWith_apply (c : ℝ) (T : ℕ) (dθ dg : Fin T → Fin d → ℝ) (i j : Fin d) :
    probeCrossCovWith c T dθ dg i j = c * ∑ k : Fin T, dθ k i * dg k j := by
  simp [probeCrossCovWith, Matrix.sum_apply, Matrix.vecMulVec_apply]

end Definitions

/-! ### Cases (i) and (ii): identities for every probe count and every divisor -/

section FiniteDeletions

/-- **Case (i) on the probe.** A vanishing slack Jacobian zeroes the bias-channel reading
identically, for every probe count. -/
theorem probeSlackReading_eq_zero_of_jacobian_eq_zero {Jb : Matrix (Fin d) (Fin d) ℝ}
    (hJ : Jb = 0) (T : ℕ) (dθ dg : Fin T → Fin d → ℝ) :
    probeSlackReading Jb T dθ dg = 0 := by
  simp [probeSlackReading, hJ]

/-- **Case (ii) on the probe, abstract form.** A vanishing latent-weight fluctuation zeroes every
summand and hence the estimator, for every probe count and every divisor. -/
theorem probeCrossCovWith_eq_zero_of_fluct_eq_zero {T : ℕ} {dθ dg : Fin T → Fin d → ℝ}
    (hθ : ∀ k, dθ k = 0) (c : ℝ) :
    probeCrossCovWith c T dθ dg = 0 := by
  have : ∀ k : Fin T, Matrix.vecMulVec (dθ k) (dg k) = (0 : Matrix (Fin d) (Fin d) ℝ) := by
    intro k
    ext i j
    simp [Matrix.vecMulVec_apply, hθ]
  simp [probeCrossCovWith, this]

/-- A body that does not depend on the batch has zero fluctuation across the probes: its sample
mean is its common value. The hypothesis `T ≠ 0` is genuine, since the sample mean over no probes
is Lean's `0` rather than the common value. -/
theorem probeBodyFluct_eq_zero_of_const {h : Ω → Fin d → ℝ} (hconst : ∀ x y, h x = h y)
    {T : ℕ} (X : Fin T → Ω) (hT : T ≠ 0) (k : Fin T) : probeBodyFluct h X k = 0 := by
  have hmean : ∑ k' : Fin T, h (X k') = (T : ℝ) • h (X k) := by
    rw [Finset.sum_congr rfl (fun k' _ => hconst (X k') (X k)), Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, ← Nat.cast_smul_eq_nsmul ℝ]
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  ext i
  simp [probeBodyFluct, probeCentre, probeMean, hmean, smul_smul, inv_mul_cancel₀ hTne]

/-- **Case (ii) on the probe, from the paper's own hypothesis.** A batch-independent body zeroes
the estimator identically, for every probe count and every divisor. -/
theorem probeCrossCovWith_eq_zero_of_body_const {h : Ω → Fin d → ℝ} (hconst : ∀ x y, h x = h y)
    {T : ℕ} (X : Fin T → Ω) (dg : Fin T → Fin d → ℝ) (hT : T ≠ 0) (c : ℝ) :
    probeCrossCovWith c T (probeBodyFluct h X) dg = 0 :=
  probeCrossCovWith_eq_zero_of_fluct_eq_zero (fun k => probeBodyFluct_eq_zero_of_const hconst X hT k) c

/-- Case (ii) for the estimator of `eq:cross-cov` itself. -/
theorem probeCrossCov_eq_zero_of_body_const {h : Ω → Fin d → ℝ} (hconst : ∀ x y, h x = h y)
    {T : ℕ} (X : Fin T → Ω) (dg : Fin T → Fin d → ℝ) (hT : T ≠ 0) :
    probeCrossCov T (probeBodyFluct h X) dg = 0 :=
  probeCrossCovWith_eq_zero_of_body_const hconst X dg hT _

/-- **The contrapositive Theorem 1 states in the main text**, on the probe: a nonzero
bias-channel reading forces a nonzero slack Jacobian and a batch-dependent body. -/
theorem probe_joint_necessity_finite {Jb : Matrix (Fin d) (Fin d) ℝ} {T : ℕ}
    {dθ dg : Fin T → Fin d → ℝ} (hne : probeSlackReading Jb T dθ dg ≠ 0) :
    Jb ≠ 0 ∧ ∃ k, dθ k ≠ 0 := by
  constructor
  · intro hJ
    exact hne (probeSlackReading_eq_zero_of_jacobian_eq_zero hJ T dθ dg)
  · by_contra hall
    have hz : ∀ k, dθ k = 0 := fun k => by
      by_contra hk
      exact hall ⟨k, hk⟩
    exact hne (by
      simp [probeSlackReading, probeCrossCov, probeCrossCovWith_eq_zero_of_fluct_eq_zero hz])

end FiniteDeletions

/-! ### The two divisors, and the step-window estimator on the same data -/

section Divisors

/-- **The two divisors differ by the explicit factor `T/(T − 1)`** on the same data. This is
the theorem behind the paper's clause that Theorem 1 is indifferent to the divisor: everything
the theorem asserts about one estimator transfers to the other through a nonzero scalar. -/
theorem probeCrossCov_eq_smul_probeCrossCovMean {T : ℕ} (hT : T ≠ 0) (dθ dg : Fin T → Fin d → ℝ) :
    probeCrossCov T dθ dg = ((T : ℝ) / ((T : ℝ) - 1)) • probeCrossCovMean T dθ dg := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  rw [probeCrossCov, probeCrossCovMean, probeCrossCovWith, probeCrossCovWith, smul_smul,
    div_mul_eq_mul_div, mul_inv_cancel₀ hTne, one_div]

/-- **The probe estimator is `T/(T − 1)` times the step-window estimator of `CrossCov.lean` on the
same data.** Reading the window `t − T + s`, `s < T`, as the `T` probes, the Bessel-divisor
probe estimator and the `1/T` window estimator differ by that scalar and by nothing else. -/
theorem probeCrossCov_eq_smul_crossCovEstimator {T : ℕ} (hT : T ≠ 0) (t : ℕ)
    (dθ dg : ℕ → Fin d → ℝ) :
    probeCrossCov T (fun k : Fin T => dθ (t - T + k)) (fun k : Fin T => dg (t - T + k))
      = ((T : ℝ) / ((T : ℝ) - 1)) • crossCovEstimator T t dθ dg := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  rw [probeCrossCov, probeCrossCovWith, crossCovEstimator, smul_smul, div_mul_eq_mul_div,
    mul_inv_cancel₀ hTne, one_div]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun s => Matrix.vecMulVec (dθ (t - T + s)) (dg (t - T + s))) T

/-- The prefix form: the `T` probes are the first `T` members of a family. -/
theorem probeCrossCov_prefix_eq_smul_crossCovEstimator {T : ℕ} (hT : T ≠ 0)
    (dθ dg : ℕ → Fin d → ℝ) :
    probeCrossCov T (fun k : Fin T => dθ k) (fun k : Fin T => dg k)
      = ((T : ℝ) / ((T : ℝ) - 1)) • crossCovEstimator T T dθ dg := by
  have h := probeCrossCov_eq_smul_crossCovEstimator hT T dθ dg
  simpa [Nat.sub_self] using h

end Divisors

/-! ### The sample-cross-covariance identity on the probe -/

section SampleIdentity

/-- The scalar core of the sample-cross-covariance identity over `Fin T`: centring two real
families at their sample means and summing the products gives the sum of the raw products minus
`T` times the product of the means. Exact for every `T`, the empty case included. -/
lemma probe_centred_sum_identity (T : ℕ) (a b : Fin T → ℝ) :
    ∑ k : Fin T, (a k - (T : ℝ)⁻¹ * ∑ l : Fin T, a l) * (b k - (T : ℝ)⁻¹ * ∑ l : Fin T, b l)
      = (∑ k : Fin T, a k * b k)
        - (T : ℝ) * (((T : ℝ)⁻¹ * ∑ l : Fin T, a l) * ((T : ℝ)⁻¹ * ∑ l : Fin T, b l)) := by
  rcases eq_or_ne T 0 with rfl | hT0
  · simp
  have hT : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT0
  set A : ℝ := ∑ l : Fin T, a l with hA
  set B : ℝ := ∑ l : Fin T, b l with hB
  have hexp : ∀ k : Fin T, (a k - (T : ℝ)⁻¹ * A) * (b k - (T : ℝ)⁻¹ * B)
      = a k * b k - ((T : ℝ)⁻¹ * B) * a k - ((T : ℝ)⁻¹ * A) * b k
        + ((T : ℝ)⁻¹ * A) * ((T : ℝ)⁻¹ * B) := by
    intro k; ring
  rw [Finset.sum_congr rfl (fun k _ => hexp k), Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← hA, ← hB]
  field_simp
  ring

/-- **The sample-cross-covariance identity for the probe**, with an arbitrary divisor: the
estimator built from the sample-mean-centred pairs equals the estimator of the raw pairs minus
`c · T` times the outer product of the two sample means. Exact for every probe count. -/
theorem probe_sample_crossCov_identity (c : ℝ) (T : ℕ) (u v : Fin T → Fin d → ℝ) :
    probeCrossCovWith c T (probeCentre T u) (probeCentre T v)
      = probeCrossCovWith c T u v
        - (c * (T : ℝ)) • Matrix.vecMulVec (probeMean T u) (probeMean T v) := by
  ext i j
  simp only [probeCrossCovWith_apply, Matrix.sub_apply, Matrix.smul_apply, Matrix.vecMulVec_apply,
    probeCentre, Pi.sub_apply, smul_eq_mul, probeMean_apply]
  rw [probe_centred_sum_identity T (fun k => u k i) (fun k => v k j)]
  ring

/-- The identity for the estimator of `eq:cross-cov`: the mean-product term carries the factor
`T/(T − 1)`. -/
theorem probeCrossCov_sample_identity (T : ℕ) (u v : Fin T → Fin d → ℝ) :
    probeCrossCov T (probeCentre T u) (probeCentre T v)
      = probeCrossCov T u v
        - ((T : ℝ) / ((T : ℝ) - 1)) • Matrix.vecMulVec (probeMean T u) (probeMean T v) := by
  rw [probeCrossCov, probe_sample_crossCov_identity, ← probeCrossCov, div_eq_inv_mul]

/-- Sample-mean centring over the probes is window-mean centring over the prefix of a family:
the two files speak about the same fluctuation. -/
theorem probeCentre_prefix_eq_windowCentre (T : ℕ) (u : ℕ → Fin d → ℝ) (k : Fin T) :
    probeCentre T (fun k : Fin T => u k) k = windowCentre T 0 u k := by
  funext i
  simp [probeCentre, windowCentre, windowMean_apply, probeMean_apply,
    Fin.sum_univ_eq_sum_range (fun s => u s i) T]

/-- The centred probe estimator is `T/(T − 1)` times the window-mean-centred estimator of
`CrossCovSLLN.lean` on the prefix. -/
theorem probeCrossCov_centred_prefix_eq {T : ℕ} (hT : T ≠ 0) (u v : ℕ → Fin d → ℝ) :
    probeCrossCov T (probeCentre T fun k : Fin T => u k) (probeCentre T fun k : Fin T => v k)
      = ((T : ℝ) / ((T : ℝ) - 1)) •
          crossCovEstimator T 0 (windowCentre T 0 u) (windowCentre T 0 v) := by
  have h := probeCrossCov_eq_smul_crossCovEstimator hT 0 (windowCentre T 0 u) (windowCentre T 0 v)
  simp only [Nat.zero_sub, zero_add] at h
  rw [← h]
  congr 1 <;> funext k <;> exact probeCentre_prefix_eq_windowCentre T _ k

/-- **The probe estimator of case (iii) is not identically zero**: an explicit probe of size `2`
in one dimension on which the sample-mean-centred estimator equals `1/2`. -/
theorem probeCrossCov_centred_ne_zero :
    ∃ (T : ℕ) (u v : Fin T → Fin 1 → ℝ),
      probeCrossCov T (probeCentre T u) (probeCentre T v) ≠ 0 := by
  refine ⟨2, (fun k _ => (k : ℝ)), (fun k _ => (k : ℝ)), ?_⟩
  intro h
  have h00 : probeCrossCov 2 (probeCentre 2 fun k (_ : Fin 1) => (k : ℝ))
      (probeCentre 2 fun k (_ : Fin 1) => (k : ℝ)) 0 0 = 0 := by rw [h]; rfl
  rw [probeCrossCov, probeCrossCovWith_apply] at h00
  norm_num [probeCentre, probeMean, Fin.sum_univ_two] at h00

end SampleIdentity

/-! ### Case (iii): unbiased for the zero target, with either divisor -/

section Unbiased

variable [MeasurableSpace Ω]

/-- **The expectation of the probe estimator is the divisor times the sum of the per-probe
population cross-covariances** `popCrossCov` of `CrossCovIndep.lean`. This is linearity of the
integral over the finite probe, and it holds for every divisor. -/
theorem integral_probeCrossCovWith_entry {μ : Measure Ω} (c : ℝ) (T : ℕ)
    (U V : Fin T → Ω → Fin d → ℝ)
    (hprodInt : ∀ k i j, Integrable (fun ω => U k ω i * V k ω j) μ) (i j : Fin d) :
    ∫ ω, probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j ∂μ
      = c * ∑ k : Fin T, popCrossCov μ (U k) (V k) i j := by
  simp only [probeCrossCovWith_apply]
  rw [integral_const_mul, integral_finsetSum _ (fun k _ => hprodInt k i j)]
  simp [popCrossCov]

/-- **Case (iii), the (T − 1)/T rescaling of zero.** If every probe's population cross-covariance
vanishes, the estimator is unbiased for the zero target with every divisor, in particular with
Bessel's `(T − 1)⁻¹` and with `T⁻¹`. -/
theorem probeCrossCovWith_unbiased_of_popCrossCov_eq_zero {μ : Measure Ω} (c : ℝ) (T : ℕ)
    (U V : Fin T → Ω → Fin d → ℝ)
    (hprodInt : ∀ k i j, Integrable (fun ω => U k ω i * V k ω j) μ)
    (hpop : ∀ k, popCrossCov μ (U k) (V k) = 0) (i j : Fin d) :
    ∫ ω, probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j ∂μ = 0 := by
  rw [integral_probeCrossCovWith_entry c T U V hprodInt i j]
  simp [hpop]

/-- The product of two coordinates of independent integrable random vectors is integrable. -/
theorem integrable_coord_mul_of_indepFun {μ : Measure Ω} {U V : Ω → Fin d → ℝ}
    (hindep : IndepFun U V μ) (hU : Integrable U μ) (hV : Integrable V μ) (i j : Fin d) :
    Integrable (fun ω => U ω i * V ω j) μ :=
  (hindep.comp (measurable_pi_apply i) (measurable_pi_apply j)).integrable_mul (hU.eval i)
    (hV.eval j)

/-- **Case (iii) from the paper's own hypothesis, every divisor.** If at each probe the latent weight
fluctuation and the reference-iterate gradient fluctuation are independent and integrable and the
latent-weight fluctuation is centred, the population cross-covariance of each probe is zero by
`popCrossCov_eq_zero_of_indepFun_of_centered`, and the estimator is unbiased for that zero with
every divisor. -/
theorem probeCrossCovWith_unbiased_of_indepFun {μ : Measure Ω} (c : ℝ) (T : ℕ)
    (U V : Fin T → Ω → Fin d → ℝ)
    (hindep : ∀ k, IndepFun (U k) (V k) μ)
    (hU : ∀ k, Integrable (U k) μ) (hV : ∀ k, Integrable (V k) μ)
    (hUc : ∀ k, popMean μ (U k) = 0) (i j : Fin d) :
    ∫ ω, probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j ∂μ = 0 :=
  probeCrossCovWith_unbiased_of_popCrossCov_eq_zero c T U V
    (fun k i j => integrable_coord_mul_of_indepFun (hindep k) (hU k) (hV k) i j)
    (fun k => popCrossCov_eq_zero_of_indepFun_of_centered (hindep k) (hU k) (hV k) (hUc k)) i j

/-- Case (iii) for the estimator of `eq:cross-cov`, Bessel's divisor. -/
theorem probeCrossCov_unbiased_of_indepFun {μ : Measure Ω} (T : ℕ)
    (U V : Fin T → Ω → Fin d → ℝ)
    (hindep : ∀ k, IndepFun (U k) (V k) μ)
    (hU : ∀ k, Integrable (U k) μ) (hV : ∀ k, Integrable (V k) μ)
    (hUc : ∀ k, popMean μ (U k) = 0) (i j : Fin d) :
    ∫ ω, probeCrossCov T (fun k => U k ω) (fun k => V k ω) i j ∂μ = 0 :=
  probeCrossCovWith_unbiased_of_indepFun _ T U V hindep hU hV hUc i j

/-- Case (iii) with the divisor `T⁻¹` in place of Bessel's: the same zero. -/
theorem probeCrossCovMean_unbiased_of_indepFun {μ : Measure Ω} (T : ℕ)
    (U V : Fin T → Ω → Fin d → ℝ)
    (hindep : ∀ k, IndepFun (U k) (V k) μ)
    (hU : ∀ k, Integrable (U k) μ) (hV : ∀ k, Integrable (V k) μ)
    (hUc : ∀ k, popMean μ (U k) = 0) (i j : Fin d) :
    ∫ ω, probeCrossCovMean T (fun k => U k ω) (fun k => V k ω) i j ∂μ = 0 :=
  probeCrossCovWith_unbiased_of_indepFun _ T U V hindep hU hV hUc i j

/-- The two sample means are independent when the two probe collections are: any measurable
functional of one collection is independent of any measurable functional of the other. This is
(A1)'s i.i.d. probes together with case (iii)'s conditional independence, at the collection
level. -/
theorem indepFun_probeMean_probeMean {μ : Measure Ω} (T : ℕ) (U V : Fin T → Ω → Fin d → ℝ)
    (i j : Fin d)
    (hcoll : IndepFun (fun ω (k : Fin T) => U k ω) (fun ω (k : Fin T) => V k ω) μ) :
    IndepFun (fun ω => probeMean T (fun k => U k ω) i)
             (fun ω => probeMean T (fun k => V k ω) j) μ := by
  have hφ : Measurable fun f : Fin T → Fin d → ℝ => probeMean T f i := by
    simp only [probeMean_apply]; fun_prop
  have hψ : Measurable fun f : Fin T → Fin d → ℝ => probeMean T f j := by
    simp only [probeMean_apply]; fun_prop
  exact hcoll.comp hφ hψ

/-- The sample mean of a centred family is integrable and centred. -/
theorem integrable_probeMean {μ : Measure Ω} (T : ℕ) (U : Fin T → Ω → Fin d → ℝ) (i : Fin d)
    (hUint : ∀ k, Integrable (fun ω => U k ω i) μ) :
    Integrable (fun ω => probeMean T (fun k => U k ω) i) μ := by
  simp only [probeMean_apply]
  exact (integrable_finsetSum _ fun k _ => hUint k).const_mul _

theorem integral_probeMean_eq_zero {μ : Measure Ω} (T : ℕ) (U : Fin T → Ω → Fin d → ℝ) (i : Fin d)
    (hUint : ∀ k, Integrable (fun ω => U k ω i) μ)
    (hUc : ∀ k, ∫ ω, U k ω i ∂μ = 0) :
    ∫ ω, probeMean T (fun k => U k ω) i ∂μ = 0 := by
  simp only [probeMean_apply]
  rw [integral_const_mul, integral_finsetSum _ (fun k _ => hUint k)]
  simp [hUc]

/-- The mean-product term of the identity has zero mean under collection-level independence and
centring of the latent-weight fluctuation. -/
theorem integral_probeMean_mul_eq_zero {μ : Measure Ω} (T : ℕ) (U V : Fin T → Ω → Fin d → ℝ)
    (i j : Fin d)
    (hcoll : IndepFun (fun ω (k : Fin T) => U k ω) (fun ω (k : Fin T) => V k ω) μ)
    (hUint : ∀ k, Integrable (fun ω => U k ω i) μ)
    (hVint : ∀ k, Integrable (fun ω => V k ω j) μ)
    (hUc : ∀ k, ∫ ω, U k ω i ∂μ = 0) :
    ∫ ω, probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j ∂μ = 0 := by
  have hM := indepFun_probeMean_probeMean T U V i j hcoll
  have h := hM.integral_mul_eq_mul_integral (integrable_probeMean T U i hUint).aestronglyMeasurable
    (integrable_probeMean T V j hVint).aestronglyMeasurable
  simp only [Pi.mul_apply] at h
  rw [h, integral_probeMean_eq_zero T U i hUint hUc, zero_mul]

/-- **Case (iii) for the sample-mean-centred estimator the paper forms, every divisor.** Under
collection-level independence of the latent weight fluctuations and the reference-iterate gradient
fluctuations across the probes, with the latent weight fluctuations centred and every coordinate
integrable, the centred probe estimator is unbiased for the zero target with every divisor — with
Bessel's `(T − 1)⁻¹` and with `T⁻¹` alike. The per-probe independence, the product
integrability, and the vanishing of the mean-product term are derived. -/
theorem probeCrossCovWith_centred_unbiased_of_indepFun {μ : Measure Ω} (c : ℝ) (T : ℕ)
    (U V : Fin T → Ω → Fin d → ℝ) (i j : Fin d)
    (hcoll : IndepFun (fun ω (k : Fin T) => U k ω) (fun ω (k : Fin T) => V k ω) μ)
    (hUint : ∀ k i', Integrable (fun ω => U k ω i') μ)
    (hVint : ∀ k j', Integrable (fun ω => V k ω j') μ)
    (hUc : ∀ k i', ∫ ω, U k ω i' ∂μ = 0) :
    ∫ ω, probeCrossCovWith c T (probeCentre T fun k => U k ω)
      (probeCentre T fun k => V k ω) i j ∂μ = 0 := by
  have hstep : ∀ k, IndepFun (U k) (V k) μ := fun k =>
    hcoll.comp (measurable_pi_apply k) (measurable_pi_apply k)
  have hprodInt : ∀ k i' j', Integrable (fun ω => U k ω i' * V k ω j') μ := fun k i' j' =>
    ((hstep k).comp (measurable_pi_apply i') (measurable_pi_apply j')).integrable_mul
      (hUint k i') (hVint k j')
  have hprod : ∀ k, popCrossCov μ (U k) (V k) = 0 := by
    intro k
    ext i' j'
    have hij : IndepFun (fun ω => U k ω i') (fun ω => V k ω j') μ :=
      (hstep k).comp (measurable_pi_apply i') (measurable_pi_apply j')
    have := hij.integral_mul_eq_mul_integral (hUint k i').aestronglyMeasurable
      (hVint k j').aestronglyMeasurable
    simp only [Pi.mul_apply] at this
    simp [popCrossCov, this, hUc k i']
  have hMeanInt : Integrable (fun ω =>
      probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j) μ :=
    (indepFun_probeMean_probeMean T U V i j hcoll).integrable_mul
      (integrable_probeMean T U i (fun k => hUint k i))
      (integrable_probeMean T V j (fun k => hVint k j))
  have hRawInt : Integrable (fun ω =>
      probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j) μ := by
    simp only [probeCrossCovWith_apply]
    exact (integrable_finsetSum _ fun k _ => hprodInt k i j).const_mul _
  have hid : ∀ ω, probeCrossCovWith c T (probeCentre T fun k => U k ω)
      (probeCentre T fun k => V k ω) i j
      = probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j
        - (c * (T : ℝ)) * (probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j) := by
    intro ω
    rw [probe_sample_crossCov_identity]
    simp [Matrix.sub_apply, Matrix.vecMulVec_apply]
  simp only [hid]
  rw [integral_sub hRawInt (hMeanInt.const_mul _), integral_const_mul,
    integral_probeMean_mul_eq_zero T U V i j hcoll (fun k => hUint k i) (fun k => hVint k j)
      (fun k => hUc k i),
    probeCrossCovWith_unbiased_of_popCrossCov_eq_zero c T U V hprodInt hprod i j]
  ring

end Unbiased

/-! ### Bessel's correction for a nonzero target -/

section Bessel

variable [MeasurableSpace Ω]

/-- For probes whose second moments are `Σ` on the diagonal and zero off it, the mean product of
the two sample means is `Σ/T`. -/
theorem integral_probeMean_mul_of_iid {μ : Measure Ω} {T : ℕ} (hT : T ≠ 0)
    (U V : Fin T → Ω → Fin d → ℝ) (i j : Fin d) {sig : ℝ}
    (hprod : ∀ k l, Integrable (fun ω => U k ω i * V l ω j) μ)
    (hdiag : ∀ k, ∫ ω, U k ω i * V k ω j ∂μ = sig)
    (hoff : ∀ k l, k ≠ l → ∫ ω, U k ω i * V l ω j ∂μ = 0) :
    ∫ ω, probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j ∂μ
      = sig / (T : ℝ) := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  have hexp : ∀ ω, probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j
      = ((T : ℝ)⁻¹ * (T : ℝ)⁻¹) * ∑ k : Fin T, ∑ l : Fin T, U k ω i * V l ω j := by
    intro ω
    simp only [probeMean_apply]
    rw [show ((T : ℝ)⁻¹ * ∑ k, U k ω i) * ((T : ℝ)⁻¹ * ∑ k, V k ω j)
        = ((T : ℝ)⁻¹ * (T : ℝ)⁻¹) * ((∑ k, U k ω i) * ∑ k, V k ω j) by ring, Finset.sum_mul_sum]
  simp only [hexp]
  rw [integral_const_mul, integral_finsetSum _ (fun k _ => integrable_finsetSum _
    (fun l _ => hprod k l))]
  have hinner : ∀ k : Fin T, ∫ ω, ∑ l : Fin T, U k ω i * V l ω j ∂μ = sig := by
    intro k
    rw [integral_finsetSum _ (fun l _ => hprod k l)]
    rw [Finset.sum_eq_single k (fun l _ hlk => hoff k l (Ne.symm hlk)) (fun h => absurd
      (Finset.mem_univ k) h)]
    exact hdiag k
  rw [Finset.sum_congr rfl (fun k _ => hinner k), Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- **The expectation of the sample-mean-centred probe estimator with divisor `c` is
`c (T − 1) Σ`.** The hypotheses are the second moments of i.i.d. probes: each probe's own
second moment is the population cross-covariance entry `Σ`, and the second moments across two
distinct probes vanish. Nothing is assumed about `Σ`, which may be nonzero. -/
theorem probeCrossCovWith_centred_expectation_of_iid {μ : Measure Ω} (c : ℝ) {T : ℕ}
    (hT : T ≠ 0) (U V : Fin T → Ω → Fin d → ℝ) (i j : Fin d) {sig : ℝ}
    (hprod : ∀ k l, Integrable (fun ω => U k ω i * V l ω j) μ)
    (hdiag : ∀ k, ∫ ω, U k ω i * V k ω j ∂μ = sig)
    (hoff : ∀ k l, k ≠ l → ∫ ω, U k ω i * V l ω j ∂μ = 0) :
    ∫ ω, probeCrossCovWith c T (probeCentre T fun k => U k ω)
      (probeCentre T fun k => V k ω) i j ∂μ = c * ((T : ℝ) - 1) * sig := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  have hMeanInt : Integrable (fun ω =>
      probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j) μ := by
    have hexp : (fun ω => probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j)
        = fun ω => ((T : ℝ)⁻¹ * (T : ℝ)⁻¹) * ∑ k : Fin T, ∑ l : Fin T, U k ω i * V l ω j := by
      funext ω
      simp only [probeMean_apply]
      rw [show ((T : ℝ)⁻¹ * ∑ k, U k ω i) * ((T : ℝ)⁻¹ * ∑ k, V k ω j)
          = ((T : ℝ)⁻¹ * (T : ℝ)⁻¹) * ((∑ k, U k ω i) * ∑ k, V k ω j) by ring, Finset.sum_mul_sum]
    rw [hexp]
    exact (integrable_finsetSum _ fun k _ => integrable_finsetSum _ fun l _ => hprod k l).const_mul _
  have hRawInt : Integrable (fun ω =>
      probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j) μ := by
    simp only [probeCrossCovWith_apply]
    exact (integrable_finsetSum _ fun k _ => hprod k k).const_mul _
  have hid : ∀ ω, probeCrossCovWith c T (probeCentre T fun k => U k ω)
      (probeCentre T fun k => V k ω) i j
      = probeCrossCovWith c T (fun k => U k ω) (fun k => V k ω) i j
        - (c * (T : ℝ)) * (probeMean T (fun k => U k ω) i * probeMean T (fun k => V k ω) j) := by
    intro ω
    rw [probe_sample_crossCov_identity]
    simp [Matrix.sub_apply, Matrix.vecMulVec_apply]
  simp only [hid]
  rw [integral_sub hRawInt (hMeanInt.const_mul _), integral_const_mul,
    integral_probeMean_mul_of_iid hT U V i j hprod hdiag hoff]
  simp only [probeCrossCovWith_apply]
  rw [integral_const_mul, integral_finsetSum _ (fun k _ => hprod k k),
    Finset.sum_congr rfl (fun k _ => hdiag k), Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  have h1 : c * (T : ℝ) * (sig / (T : ℝ)) = c * sig := by field_simp
  rw [h1]
  ring

/-- **Bessel's divisor is exact.** With `2 ≤ T` probes, the sample-mean-centred estimator of
`eq:cross-cov` has expectation exactly the population cross-covariance entry `Σ`. -/
theorem probeCrossCov_centred_unbiased_of_iid {μ : Measure Ω} {T : ℕ} (hT : 2 ≤ T)
    (U V : Fin T → Ω → Fin d → ℝ) (i j : Fin d) {sig : ℝ}
    (hprod : ∀ k l, Integrable (fun ω => U k ω i * V l ω j) μ)
    (hdiag : ∀ k, ∫ ω, U k ω i * V k ω j ∂μ = sig)
    (hoff : ∀ k l, k ≠ l → ∫ ω, U k ω i * V l ω j ∂μ = 0) :
    ∫ ω, probeCrossCov T (probeCentre T fun k => U k ω)
      (probeCentre T fun k => V k ω) i j ∂μ = sig := by
  have hT0 : T ≠ 0 := by omega
  have hT1 : (T : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ T := by exact_mod_cast hT
    linarith
  have h := probeCrossCovWith_centred_expectation_of_iid ((T : ℝ) - 1)⁻¹ hT0 U V i j hprod hdiag
    hoff
  rw [inv_mul_cancel₀ hT1, one_mul] at h
  exact h

/-- **The divisor `T⁻¹` is biased by the factor `(T − 1)/T`** on a nonzero target: this is why the
paper keeps Bessel's divisor, and why Theorem 1, whose target is zero, does not care. -/
theorem probeCrossCovMean_centred_expectation_of_iid {μ : Measure Ω} {T : ℕ} (hT : T ≠ 0)
    (U V : Fin T → Ω → Fin d → ℝ) (i j : Fin d) {sig : ℝ}
    (hprod : ∀ k l, Integrable (fun ω => U k ω i * V l ω j) μ)
    (hdiag : ∀ k, ∫ ω, U k ω i * V k ω j ∂μ = sig)
    (hoff : ∀ k l, k ≠ l → ∫ ω, U k ω i * V l ω j ∂μ = 0) :
    ∫ ω, probeCrossCovMean T (probeCentre T fun k => U k ω)
      (probeCentre T fun k => V k ω) i j ∂μ = (((T : ℝ) - 1) / (T : ℝ)) * sig := by
  have h := probeCrossCovWith_centred_expectation_of_iid (T : ℝ)⁻¹ hT U V i j hprod hdiag hoff
  rw [show (((T : ℝ) - 1) / (T : ℝ)) * sig = (T : ℝ)⁻¹ * ((T : ℝ) - 1) * sig by ring]
  exact h

/-- For population-centred independent probes the cross-probe second moments vanish: the
hypothesis `hoff` of the Bessel theorems is derived from independence and centring, exactly as the
population zero of `CrossCovIndep.lean` is. -/
theorem integral_mul_eq_zero_of_indepFun_of_centered {μ : Measure Ω} {X Y : Ω → ℝ}
    (hindep : IndepFun X Y μ) (hX : AEStronglyMeasurable X μ) (hY : AEStronglyMeasurable Y μ)
    (hX0 : ∫ ω, X ω ∂μ = 0) : ∫ ω, X ω * Y ω ∂μ = 0 := by
  have h := hindep.integral_mul_eq_mul_integral hX hY
  simp only [Pi.mul_apply] at h
  rw [h, hX0, zero_mul]

end Bessel

/-! ### The `L²` rate and the almost-sure limit, transported from the step-window estimator -/

section Rates

variable [MeasurableSpace Ω]

/-- **The `L²` form of `O(T^{-1/2})` for the Bessel-divisor probe estimator.** The mean square of
an entry of the estimator built from `T` centred i.i.d. probes is `T σ² / (T − 1)²`, with `σ²`
the variance of a single summand `δθ̃ᵢ δgⱼ`. It is the window statement
`integral_sq_crossCovEstimator_entry` transported by the factor `T/(T − 1)`, squared. -/
theorem integral_sq_probeCrossCov_entry {μ : Measure Ω} [IsFiniteMeasure μ] {T : ℕ} (hT : T ≠ 0)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (h : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j) :
    ∫ ω, (probeCrossCov T (fun k : Fin T => U k ω) (fun k : Fin T => V k ω) i j) ^ 2 ∂μ
      = (T : ℝ) * Var[fun ω => U 0 ω i * V 0 ω j; μ] / ((T : ℝ) - 1) ^ 2 := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  have hentry : ∀ ω, probeCrossCov T (fun k : Fin T => U k ω) (fun k : Fin T => V k ω) i j
      = ((T : ℝ) / ((T : ℝ) - 1)) * crossCovEstimator T T (fun s => U s ω) (fun s => V s ω) i j := by
    intro ω
    have h := congrFun (congrFun (probeCrossCov_prefix_eq_smul_crossCovEstimator hT
      (fun s => U s ω) (fun s => V s ω)) i) j
    simpa using h
  simp only [hentry, mul_pow]
  rw [integral_const_mul, integral_sq_crossCovEstimator_entry T T U V i j h]
  field_simp

/-- The `L²` norm of an entry of the Bessel-divisor probe estimator is `√T σ / (T − 1)`, which is
`σ/√T` up to the factor `T/(T − 1)`: the paper's `O(T^{-1/2})`. -/
theorem probeCrossCov_entry_l2_rate {μ : Measure Ω} [IsFiniteMeasure μ] {T : ℕ} (hT : 1 ≤ T)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (h : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j) :
    √(∫ ω, (probeCrossCov T (fun k : Fin T => U k ω) (fun k : Fin T => V k ω) i j) ^ 2 ∂μ)
      = √(T : ℝ) * √(Var[fun ω => U 0 ω i * V 0 ω j; μ]) / ((T : ℝ) - 1) := by
  have hT0 : T ≠ 0 := by omega
  have hT1 : (0 : ℝ) ≤ (T : ℝ) - 1 := by
    have : (1 : ℝ) ≤ T := by exact_mod_cast hT
    linarith
  rw [integral_sq_probeCrossCov_entry hT0 U V i j h, Real.sqrt_div' _ (sq_nonneg _),
    Real.sqrt_mul (Nat.cast_nonneg T), Real.sqrt_sq hT1]

/-- The Bessel factor tends to one as the probe count grows. -/
theorem tendsto_bessel_factor :
    Tendsto (fun T : ℕ => (T : ℝ) / ((T : ℝ) - 1)) atTop (𝓝 1) := by
  have h := tendsto_natCast_div_add_atTop (𝕜 := ℝ) (-1)
  refine h.congr fun T => ?_
  simp [sub_eq_add_neg]

/-- **The almost-sure limit for the sample-mean-centred probe estimator.** Under the packaged
i.i.d.-and-centred hypotheses of `CrossCovSLLN.lean` on the two families, the centred probe
estimator converges almost surely to the zero matrix as the probe count grows: it is the
window-mean-centred prefix estimator times the Bessel factor, and the factor tends to one. -/
theorem probeCrossCov_centred_tendsto_zero_ae {μ : Measure Ω} (U V : ℕ → Ω → Fin d → ℝ)
    (hY : IsCentredIID μ fun s ω => outerE (U s ω) (V s ω))
    (hU : IsCentredIID μ fun s ω => vecE (U s ω))
    (hV : IsCentredIID μ fun s ω => vecE (V s ω)) :
    ∀ᵐ ω ∂μ, Tendsto (fun T : ℕ =>
        probeCrossCov T (probeCentre T fun k : Fin T => U k ω)
          (probeCentre T fun k : Fin T => V k ω)) atTop (𝓝 0) := by
  filter_upwards [crossCovEstimator_tendsto_zero_ae_matrix 0 U V hY hU hV] with ω hω
  have hlim : Tendsto (fun T : ℕ => ((T : ℝ) / ((T : ℝ) - 1)) •
      crossCovEstimator T 0 (windowCentre T 0 fun s => U s ω)
        (windowCentre T 0 fun s => V s ω)) atTop (𝓝 ((1 : ℝ) • (0 : Matrix (Fin d) (Fin d) ℝ))) :=
    tendsto_bessel_factor.smul hω
  rw [one_smul] at hlim
  refine hlim.congr' ?_
  filter_upwards [eventually_ge_atTop 1] with T hT
  exact (probeCrossCov_centred_prefix_eq (by omega) _ _).symm

end Rates

end IcnnLift
