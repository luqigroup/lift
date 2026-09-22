import Mathlib
import CrossCov

/-!
# Theorem 1, case (iii): the estimator inherits the population zero

Case (iii) of `thm:joint-necessity` (`docs/paper/v4/A1_proofs.tex`, subsection
`app:proof-joint-necessity`) is the only one of the three deletions that does not make the
finite-window estimator identically zero. Once conditional independence has zeroed the population
cross-covariance `Σ_slack = E[δθ̃ δgᵀ | φ]`, the paper asserts four further things about the
trailing-window estimator `Σ̂⁽ᵗ⁾ = (1/T) Σ_{s=t-T}^{t-1} δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ` of `eq:cross-cov`: an
algebraic rewriting of the window-mean-centred estimator as an independent average minus a
mean-product term, unbiasedness, almost-sure convergence to zero as the window grows, and
finite-window fluctuations of order `O(T^{-1/2})`. `CrossCov.lean` proves the two
identically-zero deletions (i) and (ii); `CrossCovIndep.lean` proves the population zero itself.
This file proves those four assertions, and it proves them about the same object: the estimator
of this file is `IcnnLift.crossCovEstimator` from `CrossCov.lean`, carried through the bridge
`matOfEuclid_crossCovEstimatorE` when the probabilistic statements need a normed carrier.

Four modelling decisions are worth naming, because they are what makes the Lean statements the
paper's statements.

*The conditioning.* The paper conditions on the readout parameters `φ`. Rather than routing every
statement through `MeasureTheory.condExp` and `ProbabilityTheory.CondIndepFun` — which in this
mathlib would additionally require a standard Borel hypothesis on the sample space — the measure
`μ` here **is** the conditional law given `φ`. Every expectation below is therefore the paper's
`E[· | φ]` at a fixed conditioning, and the independence hypotheses are the paper's conditional
independence at that conditioning. Nothing is lost for case (iii), whose entire content is a
statement at fixed `φ`.

*The growing window.* `crossCovEstimator T t` sums the window `s = t - T, …, t - 1` with
truncated natural subtraction, so once `T ≥ t` the window is the whole prefix `{0, …, T - 1}`
(`crossCovEstimator_of_le`). The large-window limit `T → ∞` at a fixed final iteration `t` is
therefore literally the limit of prefix averages, which is what the strong law speaks about. The
`L²` statements need no such restriction: a trailing window is a time shift of a prefix window,
and `IsSquareCentredIID.shift` transports the hypotheses along that shift, so they hold for
**every** `t` and every `T`.

*Raw iterates versus population-centred ones.* The paper's `Σ̂⁽ᵗ⁾` is built from the raw
latent iterates and raw gradients, centred at their *window* means; the identity it invokes
rewrites that in terms of the *population*-centred fluctuations `θ̃⁽ˢ⁾ - E[θ̃ | φ]` and
`g⁽ˢ⁾ - E[g | φ]`, whose independence and centring are the probabilistic hypotheses. The two
descriptions coincide exactly, not merely asymptotically: window-mean centring annihilates a
constant shift (`windowCentre_sub_const`), so for every nonempty window the estimator built from
the raw families equals the one built from the population-centred families
(`crossCovEstimator_windowCentred_sub_const`). Every headline conclusion is therefore also stated
for the raw families — `crossCovEstimator_raw_windowCentred_unbiased_of_indepFun`,
`crossCovEstimator_raw_tendsto_zero_ae`, `crossCovEstimator_raw_windowCentred_l2_bound`.

*The carrier for the strong law.* `Matrix (Fin d) (Fin d) ℝ` carries no global norm, measurable
space, or Borel structure in mathlib, so `ProbabilityTheory.strong_law_ae` does not typecheck at
that type. Following the verified capability audit, the almost-sure statements are proved on
`EuclideanSpace ℝ (Fin d × Fin d)` — with `outerE` the outer product and `matOfEuclid` the matrix
reading — and `matOfEuclid_crossCovEstimatorE` identifies the transported object with
`crossCovEstimator` entry by entry. Mathlib *does* put the product topology on `Matrix`, however,
so the conclusion is not confined to the entries: `crossCovEstimator_tendsto_zero_ae_matrix`
states the almost-sure limit as `Σ̂⁽ᵗ⁾ → 0` in `Matrix (Fin d) (Fin d) ℝ`, which is the paper's
own formulation, with the entrywise statement as the intermediate step
(`tendsto_matrix_of_entries`).

**What is proved outright.** The sample-cross-covariance identity is exact finite algebra and is
derived, not assumed (`sample_crossCov_identity`). Unbiasedness is the linearity of the Bochner
integral together with the population zero of each summand, and the population zero of a summand
is itself derived from independence and centring (`integral_outerE_eq_zero_of_indepFun`). The
almost-sure limit is Etemadi's strong law, so it needs only pairwise independence — strictly
weaker than the paper's i.i.d. hypothesis. The variance of the window average is exactly the
summand variance divided by `T`.

**What is not proved, and why.** The paper's `O(T^{-1/2})` is delivered here in its `L²` form:
`variance_average_of_iid` gives the exact identity `Var[T⁻¹ Σ Z] = Var[Z] / T`, which
`integral_sq_crossCovEstimator_entry` specializes to `E[(Σ̂ᵢⱼ)²] = σᵢⱼ² / T` and
`crossCovEstimator_entry_l2_rate` reads as `‖Σ̂ᵢⱼ‖_{L²} = σᵢⱼ / √T`. An almost-sure `O(T^{-1/2})`
rate is **not** claimed and is not provable here: it would need a law of the iterated logarithm,
which this mathlib does not have, and the quantitative central limit theory (Berry–Esseen) that
would give a distributional rate is absent as well — mathlib's central limit theorem is
one-dimensional and carries no rate. Likewise the paper's `O(T^{-1})` for the mean-product term is
delivered in `L²` form (`integral_sq_windowMean_mul`: its mean square is exactly
`σ_u² σ_v² / T²`, so its `L²` norm is `σ_u σ_v / T`).
`crossCovEstimator_windowCentred_l2_bound` then bounds the mean square of the
window-mean-centred estimator the paper actually forms by `2σ²/T + 2σ_u²σ_v²/T²`, so the
`O(T^{-1})` correction that the paper only names appears explicitly in the conclusion rather than
being absorbed into an asymptotic symbol.

**Non-vacuity.** Case (iii) differs from the other two deletions precisely in that it does *not*
force the finite-window estimator to vanish, and the paper says so in words. Both halves of that
sentence are theorems here. `crossCovEstimator_windowCentred_ne_zero` exhibits a window on which
the window-mean-centred estimator is nonzero, so the algebraic statements are not about the zero
function. And `exists_nondegenerate_case_iii_model` builds, from `ProbabilityTheory.exists_iid`
and a pair of independent standard Gaussians, a probability model — not the zero measure, under
which every independence and centring hypothesis is trivially true — in which the per-step
independence `δθ̃ ⫫ δg`, the two centrings, all three `IsCentredIID` packages that the
almost-sure theorem consumes, and the `IsSquareCentredIID` package that the `L²` theorems consume
hold simultaneously, while the mean square of the estimator's `(0,0)` entry is exactly `1/T`, so
that for every nonempty window it is **not** almost surely zero. Without those two the theorems
below would be consistent with a vacuous reading.

## Results
* `outerE`, `vecE`, `matOfEuclid`, `crossCovEstimatorE` — the Euclidean carrier and its matrix
  reading.
* `matOfEuclid_crossCovEstimatorE` — the bridge: the Euclidean estimator reads as
  `IcnnLift.crossCovEstimator`.
* `tendsto_matrix_of_entries` — entrywise convergence over a finite index is convergence in the
  matrix topology.
* `windowMean`, `windowCentre`, `bodyFluct_eq_windowCentre` — the window-mean centring of
  `eq:fluct`, agreeing with `CrossCov.bodyFluct`.
* `centred_sum_identity`, `sample_crossCov_identity` — the sample-cross-covariance identity, as
  exact finite algebra, scalar core and matrix form.
* `windowCentre_sub_const`, `crossCovEstimator_windowCentred_sub_const` — window-mean centring
  annihilates a constant shift, so the raw-iterate estimator *is* the population-centred one.
* `crossCovEstimator_windowCentred_ne_zero` — the estimator of case (iii) is not identically zero.
* `crossCovEstimator_of_le`, `windowMean_of_le`, `crossCovEstimatorE_of_le` — a window at least
  as long as the iteration index is the whole prefix.
* `integrable_vecE_of_coords`, `integrable_outerE_of_coords` — Bochner integrability on the two
  Euclidean carriers from integrability of the scalar coordinates, which is what makes the
  `Integrable` hypotheses of the strong-law packages reachable from ordinary scalar hypotheses.
* `integral_outerE_eq_zero_of_indepFun`, `integral_vecE_eq_zero` — independence and centring zero
  the population outer product, and a centred fluctuation is centred on the Euclidean carrier.
* `crossCovEstimator_unbiased` — `E[Σ̂⁽ᵗ⁾] = 0` entrywise from a zero-mean summand.
* `crossCovEstimator_unbiased_of_indepFun` — the same from independence and centring.
* `integral_windowMean_eq_zero`, `integral_windowMean_mul_eq_zero` — the window means and their
  product are centred.
* `crossCovEstimator_windowCentred_unbiased` — the window-mean-centred estimator is exactly
  unbiased at every finite `T`, via the identity, from the two centrings as hypotheses.
* `indepFun_windowMean_windowMean` — the two window means are independent, from the paper's
  collection-level conditional independence.
* `crossCovEstimator_windowCentred_unbiased_of_indepFun` — exact unbiasedness with both centrings
  derived from case (iii)'s hypotheses.
* `crossCovEstimator_raw_windowCentred_unbiased_of_indepFun` — the same for the estimator built
  from the raw iterates.
* `IsCentredIID`, `IsCentredIID.tendsto_average_ae`, `IsCentredIID.tendsto_coord_ae` — the
  strong law for a pairwise independent, identically distributed, centred family.
* `crossCovEstimatorE_tendsto_zero_ae` — `Σ̂⁽ᵗ⁾ → 0` almost surely on the Euclidean carrier.
* `crossCovEstimator_tendsto_zero_ae`, `crossCovEstimator_tendsto_zero_ae_matrix` — the
  window-mean-centred estimator converges almost surely to zero, entrywise and then in the matrix
  topology, both terms of the identity handled.
* `isCentredIID_outerE`, `isCentredIID_vecE_fst`, `isCentredIID_vecE_snd`,
  `crossCovEstimator_tendsto_zero_ae_of_indepFun` — the almost-sure limit from the paper-shaped
  hypotheses, with the population zero of the summands derived rather than packaged.
* `crossCovEstimator_raw_tendsto_zero_ae` — the almost-sure matrix limit for the estimator built
  from the raw iterates.
* `IsSquareCentredIID`, `IsSquareCentredIID.shift` — the `L²` package, and its invariance under
  the time shift that turns a prefix window into a trailing window.
* `variance_average_of_iid` — `Var[T⁻¹ Σ Z] = Var[Z]/T`.
* `integral_sq_average_of_iid`, `l2_error_average` — the `L²` error is `σ/√T`.
* `integral_sq_crossCovEstimator_entry`, `crossCovEstimator_entry_l2_rate` — the paper's
  `O(T^{-1/2})`, in `L²` form, for an entry of the estimator.
* `integral_sq_windowMean_mul` — the paper's `O(T^{-1})` for the mean-product term, in `L²` form:
  its mean square is `σ_u² σ_v² / T²`.
* `memLp_crossCovEstimator_entry`, `memLp_windowMean`, `integrable_sq_windowMean_mul` — the
  integrability side conditions of the `L²` bound, derived from the packaged hypotheses.
* `crossCovEstimator_windowCentred_l2_bound` — the two combined, for the window-mean-centred
  estimator the paper actually forms, with the mean-product remainder explicit in the conclusion.
* `crossCovEstimator_windowCentred_l2_bound_of_indepFun`,
  `crossCovEstimator_raw_windowCentred_l2_bound` — the same bound with every side condition
  discharged, and for the raw iterates.
* `integral_sq_gaussianReal_std`, `integrable_sq_gaussianReal_std`, `integral_mul_gaussianProd`,
  `integral_sq_mul_gaussianProd`, `memLp_mul_gaussianProd` — the moments of the witness law.
* `exists_nondegenerate_case_iii_model` — a probability model satisfying every packaged hypothesis
  of case (iii) in which the estimator has mean square `1/T`, hence is not almost surely zero.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses this file carries, and what each would take to discharge:

* `IsCentredIID` and `IsSquareCentredIID` package (A1)'s i.i.d.-batch hypothesis together with the
  case-(iii) population zero. They are hypotheses because they are modelling assumptions about the
  optimizer, not theorems: nothing in mathlib can produce the joint law of an SGD trajectory. The
  `mean_zero` field is exactly the population zero, and it is *derivable* rather than magical —
  `integral_outerE_eq_zero_of_indepFun` derives it from conditional independence and centring, and
  `isCentredIID_outerE` assembles the whole package from the paper's own hypotheses. That the
  package is satisfiable at all, and satisfiable without collapsing the estimator to zero, is
  itself proved: see `exists_nondegenerate_case_iii_model`.
* The independence of the two window means, which the mean-product statements consume as `hM`, is
  no longer left dangling: `indepFun_windowMean_windowMean` derives it from the paper's own step —
  under (A1) the whole window collection `{θ̃⁽ˢ⁾}` is independent of `{g⁽ˢ⁾}` given `φ`, so any
  function of one is independent of any function of the other — and the `_of_indepFun` forms of
  the unbiasedness and `L²` statements consume that collection-level hypothesis instead. `hM` is
  retained as a hypothesis only in the intermediate lemmas, so that each says exactly which
  independence it uses.
* Integrability and square-integrability hypotheses (`Integrable`, `MemLp _ 2`) are genuine
  analytic side conditions of the Bochner integral and of the variance, not evasions. The three
  that `crossCovEstimator_windowCentred_l2_bound` carries are derived from the packaged
  hypotheses in `crossCovEstimator_windowCentred_l2_bound_of_indepFun`, so the headline `L²`
  statement carries none of them.
* `crossCovEstimator_windowCentred_unbiased` takes the centring of each summand and of the
  mean-product term as hypotheses; both are derived one theorem later. It is stated separately
  because it isolates the purely integral-theoretic step.

Nothing in the paper's case (iii) is assumed that this file could have proved. In particular the
sample-cross-covariance identity, which the paper asserts as "standard", is derived here, as is
the order of the mean-product term, which the paper asserts without derivation.
-/

open MeasureTheory ProbabilityTheory Finset Filter Topology Matrix

namespace IcnnLift

/-! ### The Euclidean carrier and its matrix reading -/

section Carrier

variable {d : ℕ}

/-- The outer product `u vᵀ` of two vectors, carried on `EuclideanSpace ℝ (Fin d × Fin d)`.
This is the summand `δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ` of `eq:cross-cov`, placed on the one matrix-shaped carrier
that mathlib equips with a norm, a measurable space and a Borel structure — `Matrix` has none of
the three, so the strong law cannot be stated at that type. -/
noncomputable def outerE (u v : Fin d → ℝ) : EuclideanSpace ℝ (Fin d × Fin d) :=
  WithLp.toLp 2 fun p => u p.1 * v p.2

@[simp] lemma outerE_apply (u v : Fin d → ℝ) (p : Fin d × Fin d) :
    outerE u v p = u p.1 * v p.2 := rfl

/-- A vector of `Fin d → ℝ`, carried on `EuclideanSpace ℝ (Fin d)`; used for the window means,
whose almost-sure limit is also supplied by the strong law. -/
noncomputable def vecE (u : Fin d → ℝ) : EuclideanSpace ℝ (Fin d) := WithLp.toLp 2 u

@[simp] lemma vecE_apply (u : Fin d → ℝ) (i : Fin d) : vecE u i = u i := rfl

/-- The matrix reading of a vector on the Euclidean carrier. -/
def matOfEuclid (M : EuclideanSpace ℝ (Fin d × Fin d)) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => M (i, j)

@[simp] lemma matOfEuclid_apply (M : EuclideanSpace ℝ (Fin d × Fin d)) (i j : Fin d) :
    matOfEuclid M i j = M (i, j) := rfl

lemma matOfEuclid_outerE (u v : Fin d → ℝ) :
    matOfEuclid (outerE u v) = Matrix.vecMulVec u v := by
  ext i j
  simp [Matrix.vecMulVec_apply]

/-- The outer-product map is measurable on the pair carrier. This is the measurable-glue fact
that lets the paper's parenthetical — each summand `δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ` "depends on batch `s`
alone" — be transported through `IndepFun.comp` and `IdentDistrib.comp` below. -/
lemma measurable_outerE_pair :
    Measurable fun uv : (Fin d → ℝ) × (Fin d → ℝ) => outerE uv.1 uv.2 :=
  (WithLp.measurable_toLp 2 _).comp <| measurable_pi_lambda _ fun p =>
    ((measurable_pi_apply p.1).comp measurable_fst).mul
      ((measurable_pi_apply p.2).comp measurable_snd)

/-- The first component of a pair, read on the Euclidean carrier, is measurable. -/
lemma measurable_vecE_fst :
    Measurable fun uv : (Fin d → ℝ) × (Fin d → ℝ) => vecE uv.1 :=
  (WithLp.measurable_toLp 2 _).comp measurable_fst

/-- The second component of a pair, read on the Euclidean carrier, is measurable. -/
lemma measurable_vecE_snd :
    Measurable fun uv : (Fin d → ℝ) × (Fin d → ℝ) => vecE uv.2 :=
  (WithLp.measurable_toLp 2 _).comp measurable_snd

/-- Coordinate evaluation on a Euclidean space commutes with finite sums, because it is the
continuous linear functional `EuclideanSpace.proj`. -/
lemma euclidCoord_sum {ι κ : Type*} [Fintype κ] (s : Finset ι) (f : ι → EuclideanSpace ℝ κ)
    (p : κ) : (∑ i ∈ s, f i) p = ∑ i ∈ s, (f i) p :=
  map_sum (EuclideanSpace.proj (𝕜 := ℝ) p) f s

lemma euclidCoord_smul {κ : Type*} [Fintype κ] (c : ℝ) (M : EuclideanSpace ℝ κ) (p : κ) :
    (c • M) p = c * M p := rfl

/-- The estimator `Σ̂⁽ᵗ⁾` of `eq:cross-cov`, carried on `EuclideanSpace ℝ (Fin d × Fin d)`. -/
noncomputable def crossCovEstimatorE (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    EuclideanSpace ℝ (Fin d × Fin d) :=
  (T : ℝ)⁻¹ • ∑ s ∈ range T, outerE (dθ (t - T + s)) (dg (t - T + s))

/-- **The bridge.** The matrix reading of the Euclidean estimator is exactly
`IcnnLift.crossCovEstimator` of `CrossCov.lean`: the two carriers describe the same object, so
the probabilistic statements proved on the Euclidean side are statements about the paper's
estimator. -/
theorem matOfEuclid_crossCovEstimatorE (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    matOfEuclid (crossCovEstimatorE T t dθ dg) = crossCovEstimator T t dθ dg := by
  ext i j
  simp only [matOfEuclid_apply, crossCovEstimatorE, euclidCoord_smul, euclidCoord_sum,
    outerE_apply, crossCovEstimator, Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply,
    smul_eq_mul]

/-- **Convergence in the matrix topology from convergence of the entries.** `Matrix m n R` is a
type synonym for `m → n → R` and mathlib equips it with the product topology
(`Mathlib/Topology/Instances/Matrix.lean`), even though it carries no global norm, measurable
space or Borel structure. Entrywise convergence over the finite index `Fin d` is therefore
literally convergence of matrices, and the almost-sure conclusions below can be stated at the
matrix level, as the paper states them, rather than only entry by entry. -/
theorem tendsto_matrix_of_entries {ι : Type*} {l : Filter ι}
    {f : ι → Matrix (Fin d) (Fin d) ℝ} {L : Matrix (Fin d) (Fin d) ℝ}
    (h : ∀ i j, Tendsto (fun T => f T i j) l (𝓝 (L i j))) :
    Tendsto f l (𝓝 L) :=
  (tendsto_pi_nhds (f := fun T => (f T : Fin d → Fin d → ℝ))
    (g := (L : Fin d → Fin d → ℝ))).2 fun i => tendsto_pi_nhds.2 fun j => h i j

end Carrier

/-! ### The window means, and the sample-cross-covariance identity -/

section WindowAlgebra

variable {d : ℕ} {Ω : Type*}

/-- The window mean `(1/T) Σ_{s=t-T}^{t-1} u⁽ˢ⁾` of a family over the trailing window. -/
noncomputable def windowMean (T t : ℕ) (u : ℕ → Fin d → ℝ) : Fin d → ℝ :=
  (T : ℝ)⁻¹ • ∑ s ∈ range T, u (t - T + s)

/-- The window-mean centring `u⁽ˢ⁾ - ū` of `eq:fluct`. -/
noncomputable def windowCentre (T t : ℕ) (u : ℕ → Fin d → ℝ) : ℕ → Fin d → ℝ :=
  fun s => u s - windowMean T t u

/-- `CrossCov.bodyFluct` is the window-mean centring of the body values: the two files speak
about the same fluctuation. -/
theorem bodyFluct_eq_windowCentre (h : Ω → Fin d → ℝ) (X : ℕ → Ω) (T t : ℕ) :
    bodyFluct h X T t = windowCentre T t (fun s => h (X s)) := rfl

@[simp] lemma windowMean_apply (T t : ℕ) (u : ℕ → Fin d → ℝ) (i : Fin d) :
    windowMean T t u i = (T : ℝ)⁻¹ * ∑ s ∈ range T, u (t - T + s) i := by
  simp [windowMean, Finset.sum_apply]

/-- The scalar core of the sample-cross-covariance identity: centring two real windows at their
own means and averaging the products gives the average of the raw products minus the product of
the means. Pure finite algebra; no probability enters, and every window length is covered —
`T = 0` included, both sides then vanishing. -/
lemma centred_sum_identity (T : ℕ) (a b : ℕ → ℝ) :
    (T : ℝ)⁻¹ * ∑ s ∈ range T, (a s - (T : ℝ)⁻¹ * ∑ c ∈ range T, a c)
                              * (b s - (T : ℝ)⁻¹ * ∑ c ∈ range T, b c)
      = (T : ℝ)⁻¹ * (∑ s ∈ range T, a s * b s)
        - ((T : ℝ)⁻¹ * ∑ c ∈ range T, a c) * ((T : ℝ)⁻¹ * ∑ c ∈ range T, b c) := by
  rcases eq_or_ne T 0 with rfl | hT0
  · simp
  have hT : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT0
  set A : ℝ := ∑ c ∈ range T, a c with hA
  set B : ℝ := ∑ c ∈ range T, b c with hB
  have hkey : ∑ s ∈ range T, (a s - (T : ℝ)⁻¹ * A) * (b s - (T : ℝ)⁻¹ * B)
      = (∑ s ∈ range T, a s * b s) - ((T : ℝ)⁻¹ * B) * A - ((T : ℝ)⁻¹ * A) * B
        + (T : ℝ) * (((T : ℝ)⁻¹ * A) * ((T : ℝ)⁻¹ * B)) := by
    have hexp : ∀ s ∈ range T, (a s - (T : ℝ)⁻¹ * A) * (b s - (T : ℝ)⁻¹ * B)
        = a s * b s - ((T : ℝ)⁻¹ * B) * a s - ((T : ℝ)⁻¹ * A) * b s
          + ((T : ℝ)⁻¹ * A) * ((T : ℝ)⁻¹ * B) := by
      intro s _; ring
    rw [Finset.sum_congr rfl hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib,
      Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul, ← hA, ← hB]
  rw [hkey]
  field_simp
  ring

/-- **The sample-cross-covariance identity** the paper invokes in case (iii). Writing `u` and `v`
for the population-centred fluctuations `θ̃⁽ˢ⁾ - E[θ̃ | φ]` and `g⁽ˢ⁾ - E[g | φ]`, the estimator
built from the *window*-mean-centred fluctuations equals the average of the population-centred
outer products minus the outer product of the two window means:
`Σ̂⁽ᵗ⁾ = (1/T) Σ u⁽ˢ⁾ (v⁽ˢ⁾)ᵀ - ū v̄ᵀ`.
The paper asserts this as standard; it is derived here, as exact finite algebra valid for every
`t` and every window length `T` — `T = 0` included, both sides then vanishing. -/
theorem sample_crossCov_identity (T t : ℕ) (u v : ℕ → Fin d → ℝ) :
    crossCovEstimator T t (windowCentre T t u) (windowCentre T t v)
      = crossCovEstimator T t u v - Matrix.vecMulVec (windowMean T t u) (windowMean T t v) := by
  ext i j
  simp only [crossCovEstimator, Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply,
    Matrix.sub_apply, windowCentre, Pi.sub_apply, smul_eq_mul, windowMean_apply]
  exact centred_sum_identity T (fun s => u (t - T + s) i) (fun s => v (t - T + s) j)

/-- **Window-centring is invariant under population centring.** For a nonempty window,
subtracting a constant from every member of the family changes the window mean by that constant
and the window-centred fluctuation not at all. In particular the window-centred fluctuation of
the raw iterates — `eq:fluct`, the `δθ̃⁽ˢ⁾` the paper's estimator actually uses — is literally
the window-centred fluctuation of the population-centred iterates `θ̃⁽ˢ⁾ - E[θ̃ | φ]`, so the
theorems below about `windowCentre` of centred families are theorems about the paper's estimator
of raw iterates. The hypothesis `T ≠ 0` is genuine: at `T = 0` the window mean is `0`, not the
mean over an empty window shifted by `c`. -/
theorem windowCentre_sub_const (T t : ℕ) (hT : T ≠ 0) (u : ℕ → Fin d → ℝ) (c : Fin d → ℝ) :
    windowCentre T t (fun s => u s - c) = windowCentre T t u := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  have hmean : windowMean T t (fun s => u s - c) = windowMean T t u - c := by
    funext i
    simp only [windowMean_apply, Pi.sub_apply, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul, mul_sub]
    field_simp
  funext s i
  simp only [windowCentre, hmean, Pi.sub_apply]
  ring

/-- **The estimator the paper forms from the raw iterates is the estimator formed from the
population-centred ones.** Window-mean centring annihilates a constant shift of the family, so
for a nonempty window the window-mean-centred estimator built from the raw latent iterates
`θ̃⁽ˢ⁾` and raw gradients `g⁽ˢ⁾` is *equal* — not merely asymptotically equal — to the one built
from `θ̃⁽ˢ⁾ - E[θ̃ | φ]` and `g⁽ˢ⁾ - E[g | φ]`. This is what makes the probabilistic theorems
below, whose families are population-centred by hypothesis, theorems about the paper's `Σ̂⁽ᵗ⁾`,
which is formed from raw iterates. -/
theorem crossCovEstimator_windowCentred_sub_const (T t : ℕ) (hT : T ≠ 0)
    (u v : ℕ → Fin d → ℝ) (c e : Fin d → ℝ) :
    crossCovEstimator T t (windowCentre T t fun s => u s - c)
        (windowCentre T t fun s => v s - e)
      = crossCovEstimator T t (windowCentre T t u) (windowCentre T t v) := by
  rw [windowCentre_sub_const T t hT, windowCentre_sub_const T t hT]

/-- **Case (iii) does not make the finite-window estimator identically zero.** The paper says so
in words — "Unlike cases (i) and (ii), this does not make the finite-window estimator identically
zero" — and this is that sentence as a theorem: an explicit window on which the window-mean-centred
estimator is nonzero. Without it, every conclusion of this file would be compatible with the
estimator being the zero function, and case (iii) would not be distinguishable from the two
identically-zero deletions of `CrossCov.lean`. The witness takes `d = 1`, the window
`T = 2` ending at `t = 2`, and the family `u⁽ˢ⁾ = v⁽ˢ⁾ = s`, for which the estimator is `1/4`. -/
theorem crossCovEstimator_windowCentred_ne_zero :
    ∃ (T t : ℕ) (u v : ℕ → Fin 1 → ℝ),
      crossCovEstimator T t (windowCentre T t u) (windowCentre T t v) ≠ 0 := by
  refine ⟨2, 2, (fun s _ => (s : ℝ)), (fun s _ => (s : ℝ)), ?_⟩
  intro h
  have h00 : crossCovEstimator 2 2 (windowCentre 2 2 fun s (_ : Fin 1) => (s : ℝ))
      (windowCentre 2 2 fun s (_ : Fin 1) => (s : ℝ)) 0 0 = 0 := by rw [h]; rfl
  rw [crossCovEstimator] at h00
  norm_num [windowCentre, windowMean, Finset.sum_range_succ, Matrix.vecMulVec_apply] at h00
/-- A trailing window at least as long as the current iteration index is the whole prefix
`{0, …, T-1}`: the truncated subtraction `t - T` is zero. This is what lets `T → ∞` at fixed `t`
be read as a limit of prefix averages, which is the form the strong law takes. -/
theorem crossCovEstimator_of_le (T t : ℕ) (htT : t ≤ T) (u v : ℕ → Fin d → ℝ) :
    crossCovEstimator T t u v = (T : ℝ)⁻¹ • ∑ s ∈ range T, Matrix.vecMulVec (u s) (v s) := by
  simp [crossCovEstimator, Nat.sub_eq_zero_of_le htT]

theorem crossCovEstimatorE_of_le (T t : ℕ) (htT : t ≤ T) (u v : ℕ → Fin d → ℝ) :
    crossCovEstimatorE T t u v = (T : ℝ)⁻¹ • ∑ s ∈ range T, outerE (u s) (v s) := by
  simp [crossCovEstimatorE, Nat.sub_eq_zero_of_le htT]

theorem windowMean_of_le (T t : ℕ) (htT : t ≤ T) (u : ℕ → Fin d → ℝ) :
    windowMean T t u = (T : ℝ)⁻¹ • ∑ s ∈ range T, u s := by
  simp [windowMean, Nat.sub_eq_zero_of_le htT]

end WindowAlgebra

/-! ### Unbiasedness -/

section Unbiased

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- **Integrability on the Euclidean vector carrier from integrability of the coordinates.**
`vecE u` is the finite sum `Σ i, uᵢ • eᵢ` of scalar multiples of the standard basis, so it is
Bochner integrable as soon as every coordinate is. Without this the `Integrable` hypotheses that
`isCentredIID_vecE_fst` and `isCentredIID_vecE_snd` carry would have to be established by hand on
a carrier the paper never mentions. -/
theorem integrable_vecE_of_coords {μ : Measure Ω} {U : Ω → Fin d → ℝ}
    (h : ∀ i, Integrable (fun ω => U ω i) μ) :
    Integrable (fun ω => vecE (U ω)) μ := by
  have hdec : (fun ω => vecE (U ω))
      = fun ω => ∑ i : Fin d, U ω i • (EuclideanSpace.single i (1 : ℝ)) := by
    funext ω; ext i; simp [euclidCoord_sum]
  rw [hdec]
  exact integrable_finsetSum _ fun i _ => (h i).smul_const _

/-- **Integrability on the Euclidean matrix carrier from integrability of the entries.** The same
decomposition for the outer product `δθ̃ δgᵀ`: it is the finite sum `Σ_{ij} δθ̃ᵢ δgⱼ • e_{ij}`, so
integrability of each scalar entry suffices. This is what makes the `Integrable` hypothesis of
`integral_outerE_eq_zero_of_indepFun` and `isCentredIID_outerE` reachable from ordinary scalar
integrability of the two channels. -/
theorem integrable_outerE_of_coords {μ : Measure Ω} {U V : Ω → Fin d → ℝ}
    (h : ∀ p : Fin d × Fin d, Integrable (fun ω => U ω p.1 * V ω p.2) μ) :
    Integrable (fun ω => outerE (U ω) (V ω)) μ := by
  have hdec : (fun ω => outerE (U ω) (V ω))
      = fun ω => ∑ p : Fin d × Fin d,
          (U ω p.1 * V ω p.2) • (EuclideanSpace.single p (1 : ℝ)) := by
    funext ω; ext p; simp [euclidCoord_sum]
  rw [hdec]
  exact integrable_finsetSum _ fun p _ => (h p).smul_const _

/-- **The population zero of a summand.** If the iterate fluctuation `δθ̃` and the gradient
fluctuation `δg` are independent (under `μ`, which is the paper's conditional law given `φ`) and
`δθ̃` is centred, then the population outer product `E[δθ̃ δgᵀ]` is zero. This is the
case-(iii) factorization `Σ_slack = E[δθ̃] E[δgᵀ] = 0`, proved coordinate by coordinate through
the continuous linear functional `EuclideanSpace.proj`. -/
theorem integral_outerE_eq_zero_of_indepFun {μ : Measure Ω} {U V : Ω → Fin d → ℝ}
    (hindep : IndepFun U V μ)
    (hUm : ∀ i, AEStronglyMeasurable (fun ω => U ω i) μ)
    (hVm : ∀ j, AEStronglyMeasurable (fun ω => V ω j) μ)
    (hUc : ∀ i, ∫ ω, U ω i ∂μ = 0)
    (hint : Integrable (fun ω => outerE (U ω) (V ω)) μ) :
    ∫ ω, outerE (U ω) (V ω) ∂μ = 0 := by
  ext p
  have hproj : (∫ ω, outerE (U ω) (V ω) ∂μ) p = ∫ ω, U ω p.1 * V ω p.2 ∂μ :=
    (ContinuousLinearMap.integral_comp_comm (EuclideanSpace.proj (𝕜 := ℝ) p) hint).symm
  have hmul : ∫ ω, U ω p.1 * V ω p.2 ∂μ = 0 := by
    have hij : IndepFun (fun ω => U ω p.1) (fun ω => V ω p.2) μ :=
      hindep.comp (measurable_pi_apply p.1) (measurable_pi_apply p.2)
    have := hij.integral_mul_eq_mul_integral (hUm p.1) (hVm p.2)
    simpa [Pi.mul_apply, hUc p.1] using this
  rw [hproj, hmul]
  rfl

/-- A fluctuation that is centred entrywise is centred on the Euclidean carrier: the population
mean of `vecE ∘ U` vanishes coordinate by coordinate through `EuclideanSpace.proj`. This is the
centring `E[δθ̃ | φ] = 0` (and `E[δg | φ] = 0`) of the paper's construction, in the form the
strong-law carrier needs. -/
theorem integral_vecE_eq_zero {μ : Measure Ω} {U : Ω → Fin d → ℝ}
    (hint : Integrable (fun ω => vecE (U ω)) μ)
    (hUc : ∀ i, ∫ ω, U ω i ∂μ = 0) :
    ∫ ω, vecE (U ω) ∂μ = 0 := by
  ext i
  have hproj : (∫ ω, vecE (U ω) ∂μ) i = ∫ ω, U ω i ∂μ :=
    (ContinuousLinearMap.integral_comp_comm (EuclideanSpace.proj (𝕜 := ℝ) i) hint).symm
  rw [hproj, hUc i]
  rfl

/-- **Unbiasedness, entrywise.** If every summand of the trailing window has zero mean, then so
does the estimator: this is the linearity of the integral over a finite window, and holds for
every iteration `t` and every window length `T`. -/
theorem crossCovEstimator_unbiased {μ : Measure Ω} (T t : ℕ) (U V : ℕ → Ω → Fin d → ℝ)
    (hprodInt : ∀ s i j, Integrable (fun ω => U s ω i * V s ω j) μ)
    (hprod : ∀ s i j, ∫ ω, U s ω i * V s ω j ∂μ = 0) (i j : Fin d) :
    ∫ ω, crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j ∂μ = 0 := by
  have hentry : ∀ ω, crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j
      = (T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i * V (t - T + s) ω j := by
    intro ω
    simp only [crossCovEstimator, Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply,
      smul_eq_mul]
  simp only [hentry]
  rw [integral_const_mul, integral_finsetSum _ (fun s _ => hprodInt (t - T + s) i j)]
  simp [hprod]

/-- **Unbiasedness from the paper's own hypothesis.** Independence of the two fluctuations at
each step, together with the centring of the iterate fluctuation, gives `E[Σ̂⁽ᵗ⁾] = 0`. -/
theorem crossCovEstimator_unbiased_of_indepFun {μ : Measure Ω} (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ)
    (hindep : ∀ s, IndepFun (U s) (V s) μ)
    (hUm : ∀ s i, AEStronglyMeasurable (fun ω => U s ω i) μ)
    (hVm : ∀ s j, AEStronglyMeasurable (fun ω => V s ω j) μ)
    (hUc : ∀ s i, ∫ ω, U s ω i ∂μ = 0)
    (hprodInt : ∀ s i j, Integrable (fun ω => U s ω i * V s ω j) μ) (i j : Fin d) :
    ∫ ω, crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j ∂μ = 0 := by
  refine crossCovEstimator_unbiased T t U V hprodInt ?_ i j
  intro s i' j'
  have hij : IndepFun (fun ω => U s ω i') (fun ω => V s ω j') μ :=
    (hindep s).comp (measurable_pi_apply i') (measurable_pi_apply j')
  have := hij.integral_mul_eq_mul_integral (hUm s i') (hVm s j')
  simpa [Pi.mul_apply, hUc s i'] using this

/-- The window mean of a centred family is itself centred. -/
theorem integral_windowMean_eq_zero {μ : Measure Ω} (T t : ℕ) (U : ℕ → Ω → Fin d → ℝ) (i : Fin d)
    (hUint : ∀ s, Integrable (fun ω => U s ω i) μ)
    (hUc : ∀ s, ∫ ω, U s ω i ∂μ = 0) :
    ∫ ω, windowMean T t (fun s => U s ω) i ∂μ = 0 := by
  simp only [windowMean_apply]
  rw [integral_const_mul, integral_finsetSum _ (fun s _ => hUint (t - T + s))]
  simp [hUc]

/-- The mean-product term of the sample-cross-covariance identity has zero mean, provided the two
window means are independent — which is exactly the paper's step that, under (A1), the whole
window collection of iterate fluctuations is independent of the whole collection of gradient
fluctuations given `φ`, so that any functional of one is independent of any functional of the
other. -/
theorem integral_windowMean_mul_eq_zero {μ : Measure Ω} (T t : ℕ) (U V : ℕ → Ω → Fin d → ℝ)
    (i j : Fin d)
    (hM : IndepFun (fun ω => windowMean T t (fun s => U s ω) i)
                   (fun ω => windowMean T t (fun s => V s ω) j) μ)
    (hVmeas : AEStronglyMeasurable (fun ω => windowMean T t (fun s => V s ω) j) μ)
    (hUint : ∀ s, Integrable (fun ω => U s ω i) μ)
    (hUc : ∀ s, ∫ ω, U s ω i ∂μ = 0) :
    ∫ ω, windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j ∂μ = 0 := by
  have hUmeas : AEStronglyMeasurable (fun ω => windowMean T t (fun s => U s ω) i) μ := by
    have hint : Integrable (fun ω => windowMean T t (fun s => U s ω) i) μ := by
      simp only [windowMean_apply]
      exact (integrable_finsetSum _ fun s _ => hUint (t - T + s)).const_mul _
    exact hint.aestronglyMeasurable
  have h := hM.integral_mul_eq_mul_integral hUmeas hVmeas
  simp only [Pi.mul_apply] at h
  rw [h, integral_windowMean_eq_zero T t U i hUint hUc, zero_mul]

/-- **The window-mean-centred estimator is exactly unbiased at every finite `T`.** This is the
purely integral-theoretic step: the sample-cross-covariance identity splits the estimator into an
average of summands and a mean-product term, and the two centrings `hprod` and `hMeanZero` are
taken here as **hypotheses** and combined by linearity of the integral. Neither is derived at this
level; `crossCovEstimator_windowCentred_unbiased_of_indepFun` below derives both from case
(iii)'s conditional independence together with the centring construction, and
`crossCovEstimator_raw_windowCentred_unbiased_of_indepFun` states the conclusion for the estimator
built from the raw iterates. The conclusion holds at every iteration `t` and every window
length `T`, and it is exact, not asymptotic. -/
theorem crossCovEstimator_windowCentred_unbiased {μ : Measure Ω} (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (hprodInt : ∀ s i' j', Integrable (fun ω => U s ω i' * V s ω j') μ)
    (hprod : ∀ s i' j', ∫ ω, U s ω i' * V s ω j' ∂μ = 0)
    (hMeanInt : Integrable (fun ω =>
      windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j) μ)
    (hMeanZero : ∫ ω, windowMean T t (fun s => U s ω) i
      * windowMean T t (fun s => V s ω) j ∂μ = 0) :
    ∫ ω, crossCovEstimator T t (windowCentre T t fun s => U s ω)
      (windowCentre T t fun s => V s ω) i j ∂μ = 0 := by
  have hRawInt : Integrable (fun ω =>
      crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) μ := by
    have hentry : ∀ ω, crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j
        = (T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i * V (t - T + s) ω j := by
      intro ω
      simp only [crossCovEstimator, Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply,
        smul_eq_mul]
    simp only [hentry]
    exact (integrable_finsetSum _ fun s _ => hprodInt (t - T + s) i j).const_mul _
  have hid : ∀ ω, crossCovEstimator T t (windowCentre T t fun s => U s ω)
      (windowCentre T t fun s => V s ω) i j
      = crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j
        - windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j := by
    intro ω
    rw [sample_crossCov_identity T t]
    simp [Matrix.sub_apply, Matrix.vecMulVec_apply]
  simp only [hid]
  rw [integral_sub hRawInt hMeanInt, hMeanZero,
    crossCovEstimator_unbiased T t U V hprodInt hprod i j]
  ring

/-- **Independence of the two window means, from the paper's own collection-level hypothesis.**
Under (A1) the whole window collection `{θ̃⁽ˢ⁾}` is independent of the whole collection `{g⁽ˢ⁾}`
given `φ`; stating that as independence of the two maps `ω ↦ (s ↦ U s ω)` and
`ω ↦ (s ↦ V s ω)` into the sequence space, any functional of one is independent of any
functional of the other — in particular the two window means. This discharges the `hM`
hypothesis carried by the mean-product statements below. -/
theorem indepFun_windowMean_windowMean {μ : Measure Ω} (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (hcoll : IndepFun (fun ω s => U s ω) (fun ω s => V s ω) μ) :
    IndepFun (fun ω => windowMean T t (fun s => U s ω) i)
             (fun ω => windowMean T t (fun s => V s ω) j) μ := by
  have hφ : Measurable fun f : ℕ → Fin d → ℝ => windowMean T t f i := by
    simp only [windowMean_apply]; fun_prop
  have hψ : Measurable fun f : ℕ → Fin d → ℝ => windowMean T t f j := by
    simp only [windowMean_apply]; fun_prop
  exact hcoll.comp hφ hψ

/-- **Exact unbiasedness of the window-mean-centred estimator, from the paper's own
hypotheses.** The inputs are case (iii)'s conditional independence of the two window
collections (`hcoll`), the centring of the iterate fluctuation, and integrability side
conditions; everything else — the per-step independence, the vanishing of each summand mean,
the independence and integrability of the two window means, and the vanishing of the
mean-product term — is derived. This is `E[Σ̂⁽ᵗ⁾] = 0` for the estimator the paper actually
forms, at every iteration `t` and every window length `T`. -/
theorem crossCovEstimator_windowCentred_unbiased_of_indepFun {μ : Measure Ω} (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (hcoll : IndepFun (fun ω s => U s ω) (fun ω s => V s ω) μ)
    (hUm : ∀ s i', AEStronglyMeasurable (fun ω => U s ω i') μ)
    (hVm : ∀ s j', AEStronglyMeasurable (fun ω => V s ω j') μ)
    (hUc : ∀ s i', ∫ ω, U s ω i' ∂μ = 0)
    (hUint : ∀ s, Integrable (fun ω => U s ω i) μ)
    (hVint : ∀ s, Integrable (fun ω => V s ω j) μ)
    (hprodInt : ∀ s i' j', Integrable (fun ω => U s ω i' * V s ω j') μ) :
    ∫ ω, crossCovEstimator T t (windowCentre T t fun s => U s ω)
      (windowCentre T t fun s => V s ω) i j ∂μ = 0 := by
  have hstep : ∀ s, IndepFun (U s) (V s) μ := fun s =>
    hcoll.comp (measurable_pi_apply s) (measurable_pi_apply s)
  have hprod : ∀ s i' j', ∫ ω, U s ω i' * V s ω j' ∂μ = 0 := by
    intro s i' j'
    have hij : IndepFun (fun ω => U s ω i') (fun ω => V s ω j') μ :=
      (hstep s).comp (measurable_pi_apply i') (measurable_pi_apply j')
    have := hij.integral_mul_eq_mul_integral (hUm s i') (hVm s j')
    simpa [Pi.mul_apply, hUc s i'] using this
  have hM : IndepFun (fun ω => windowMean T t (fun s => U s ω) i)
      (fun ω => windowMean T t (fun s => V s ω) j) μ :=
    indepFun_windowMean_windowMean T t U V i j hcoll
  have hMuInt : Integrable (fun ω => windowMean T t (fun s => U s ω) i) μ := by
    simp only [windowMean_apply]
    exact (integrable_finsetSum _ fun s _ => hUint (t - T + s)).const_mul _
  have hMvInt : Integrable (fun ω => windowMean T t (fun s => V s ω) j) μ := by
    simp only [windowMean_apply]
    exact (integrable_finsetSum _ fun s _ => hVint (t - T + s)).const_mul _
  have hMeanInt : Integrable (fun ω =>
      windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j) μ :=
    hM.integrable_mul hMuInt hMvInt
  have hMeanZero := integral_windowMean_mul_eq_zero T t U V i j hM
    hMvInt.aestronglyMeasurable hUint (fun s => hUc s i)
  exact crossCovEstimator_windowCentred_unbiased T t U V i j hprodInt hprod hMeanInt hMeanZero

/-- **Exact unbiasedness for the estimator built from the *raw* iterates.** This is
`crossCovEstimator_windowCentred_unbiased_of_indepFun` transported along
`crossCovEstimator_windowCentred_sub_const`: the hypotheses are still about the
population-centred fluctuations `θ̃⁽ˢ⁾ - c_θ` and `g⁽ˢ⁾ - c_g` (with `c_θ = E[θ̃ | φ]` and
`c_g = E[g | φ]` the population means, so that `hΘc` is the paper's centring construction), but
the conclusion is about `Σ̂⁽ᵗ⁾` formed from the raw families `Θ` and `G`, which is the estimator
the paper actually writes down. The hypothesis `T ≠ 0` is needed because window-mean centring is
what removes the shift, and there is no window to average over at `T = 0`. -/
theorem crossCovEstimator_raw_windowCentred_unbiased_of_indepFun {μ : Measure Ω} (T t : ℕ)
    (hT : T ≠ 0) (Θ G : ℕ → Ω → Fin d → ℝ) (cΘ cG : Fin d → ℝ) (i j : Fin d)
    (hcoll : IndepFun (fun ω s => Θ s ω - cΘ) (fun ω s => G s ω - cG) μ)
    (hΘm : ∀ s i', AEStronglyMeasurable (fun ω => Θ s ω i' - cΘ i') μ)
    (hGm : ∀ s j', AEStronglyMeasurable (fun ω => G s ω j' - cG j') μ)
    (hΘc : ∀ s i', ∫ ω, (Θ s ω i' - cΘ i') ∂μ = 0)
    (hΘint : ∀ s, Integrable (fun ω => Θ s ω i - cΘ i) μ)
    (hGint : ∀ s, Integrable (fun ω => G s ω j - cG j) μ)
    (hprodInt : ∀ s i' j', Integrable (fun ω => (Θ s ω i' - cΘ i') * (G s ω j' - cG j')) μ) :
    ∫ ω, crossCovEstimator T t (windowCentre T t fun s => Θ s ω)
      (windowCentre T t fun s => G s ω) i j ∂μ = 0 := by
  have h := crossCovEstimator_windowCentred_unbiased_of_indepFun T t
    (fun s ω => Θ s ω - cΘ) (fun s ω => G s ω - cG) i j hcoll hΘm hGm hΘc hΘint hGint hprodInt
  simpa only [crossCovEstimator_windowCentred_sub_const T t hT] using h

end Unbiased

/-! ### The almost-sure limit -/

section StrongLaw

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- A pairwise independent, identically distributed, integrable and centred family of
Banach-space-valued random variables — the Lean form of (A1)'s i.i.d. window together with the
case-(iii) population zero. Only *pairwise* independence is required, because the strong law used
below is Etemadi's; this is strictly weaker than the paper's i.i.d. hypothesis. -/
structure IsCentredIID {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    [MeasurableSpace E] [BorelSpace E] (μ : Measure Ω) (Z : ℕ → Ω → E) : Prop where
  /-- the summand is integrable -/
  integrable : Integrable (Z 0) μ
  /-- the summands are pairwise independent -/
  indep : Pairwise fun i j => IndepFun (Z i) (Z j) μ
  /-- the summands are identically distributed -/
  ident : ∀ i, IdentDistrib (Z i) (Z 0) μ μ
  /-- the population mean is zero: the case-(iii) population cross-covariance vanishes -/
  mean_zero : ∫ ω, Z 0 ω ∂μ = 0

/-- The strong law for a centred family: prefix averages converge almost surely to zero. -/
theorem IsCentredIID.tendsto_average_ae {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] {μ : Measure Ω} {Z : ℕ → Ω → E}
    (h : IsCentredIID μ Z) :
    ∀ᵐ ω ∂μ, Tendsto (fun T : ℕ => (T : ℝ)⁻¹ • ∑ s ∈ range T, Z s ω) atTop (𝓝 0) := by
  simpa [h.mean_zero] using strong_law_ae Z h.integrable h.indep h.ident

/-- The same, read coordinatewise on a Euclidean carrier. -/
theorem IsCentredIID.tendsto_coord_ae {κ : Type*} [Fintype κ] {μ : Measure Ω}
    {Z : ℕ → Ω → EuclideanSpace ℝ κ} (h : IsCentredIID μ Z) :
    ∀ᵐ ω ∂μ, ∀ p : κ,
      Tendsto (fun T : ℕ => (T : ℝ)⁻¹ * ∑ s ∈ range T, (Z s ω) p) atTop (𝓝 0) := by
  filter_upwards [h.tendsto_average_ae] with ω hω
  intro p
  have hc := ((EuclideanSpace.proj (𝕜 := ℝ) p).continuous.tendsto
    (0 : EuclideanSpace ℝ κ)).comp hω
  simp only [Function.comp_def, map_zero] at hc
  simpa [euclidCoord_smul, euclidCoord_sum] using hc

/-- **The almost-sure limit on the Euclidean carrier.** For a centred i.i.d. family of outer
products, the estimator of `eq:cross-cov` converges almost surely to the zero matrix as the
trailing window grows. Once `T ≥ t` the window is the whole prefix, so the strong law applies
verbatim. -/
theorem crossCovEstimatorE_tendsto_zero_ae {μ : Measure Ω} (t : ℕ) (U V : ℕ → Ω → Fin d → ℝ)
    (hY : IsCentredIID μ fun s ω => outerE (U s ω) (V s ω)) :
    ∀ᵐ ω ∂μ, Tendsto (fun T : ℕ => crossCovEstimatorE T t (fun s => U s ω) (fun s => V s ω))
      atTop (𝓝 0) := by
  filter_upwards [hY.tendsto_average_ae] with ω hω
  refine Tendsto.congr' ?_ hω
  filter_upwards [eventually_ge_atTop t] with T hT
  exact (crossCovEstimatorE_of_le T t hT _ _).symm

/-- **Case (iii)'s almost-sure conclusion, for the estimator the paper actually forms.** The
window-mean-centred estimator converges almost surely to zero, entrywise. Both terms of the
sample-cross-covariance identity are handled: the independent average by the strong law on
`EuclideanSpace ℝ (Fin d × Fin d)`, and the mean-product term by the strong law applied to each
of the two window means on `EuclideanSpace ℝ (Fin d)`. No rate is claimed here; see
`crossCovEstimator_entry_l2_rate` for the `L²` form of the paper's `O(T^{-1/2})`. -/
theorem crossCovEstimator_tendsto_zero_ae {μ : Measure Ω} (t : ℕ) (U V : ℕ → Ω → Fin d → ℝ)
    (hY : IsCentredIID μ fun s ω => outerE (U s ω) (V s ω))
    (hU : IsCentredIID μ fun s ω => vecE (U s ω))
    (hV : IsCentredIID μ fun s ω => vecE (V s ω)) :
    ∀ᵐ ω ∂μ, ∀ i j : Fin d, Tendsto (fun T : ℕ =>
        crossCovEstimator T t (windowCentre T t fun s => U s ω)
          (windowCentre T t fun s => V s ω) i j) atTop (𝓝 0) := by
  filter_upwards [hY.tendsto_coord_ae, hU.tendsto_coord_ae, hV.tendsto_coord_ae] with ω hy hu hv
  intro i j
  have hev : ∀ᶠ T : ℕ in atTop,
      ((T : ℝ)⁻¹ * ∑ s ∈ range T, (outerE (U s ω) (V s ω)) (i, j))
        - ((T : ℝ)⁻¹ * ∑ s ∈ range T, (vecE (U s ω)) i)
          * ((T : ℝ)⁻¹ * ∑ s ∈ range T, (vecE (V s ω)) j)
      = crossCovEstimator T t (windowCentre T t fun s => U s ω)
          (windowCentre T t fun s => V s ω) i j := by
    filter_upwards [eventually_ge_atTop t] with T htT
    rw [sample_crossCov_identity T t, crossCovEstimator_of_le T t htT]
    simp only [windowMean_of_le T t htT, Matrix.sub_apply, Matrix.smul_apply, Matrix.sum_apply,
      Matrix.vecMulVec_apply, Pi.smul_apply, Finset.sum_apply, smul_eq_mul, outerE_apply,
      vecE_apply]
  refine Tendsto.congr' hev ?_
  simpa using (hy (i, j)).sub ((hu i).mul (hv j))

/-- **Case (iii)'s almost-sure conclusion at the matrix level**, which is how the paper states it:
`Σ̂⁽ᵗ⁾ → 0` almost surely as the window grows, in the topology of `Matrix (Fin d) (Fin d) ℝ`, not
merely entry by entry. The entrywise statement `crossCovEstimator_tendsto_zero_ae` carries the
work; `tendsto_matrix_of_entries` assembles it, using the product topology that mathlib does put
on `Matrix`. Still no rate; see `crossCovEstimator_entry_l2_rate`. -/
theorem crossCovEstimator_tendsto_zero_ae_matrix {μ : Measure Ω} (t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ)
    (hY : IsCentredIID μ fun s ω => outerE (U s ω) (V s ω))
    (hU : IsCentredIID μ fun s ω => vecE (U s ω))
    (hV : IsCentredIID μ fun s ω => vecE (V s ω)) :
    ∀ᵐ ω ∂μ, Tendsto (fun T : ℕ =>
        crossCovEstimator T t (windowCentre T t fun s => U s ω)
          (windowCentre T t fun s => V s ω)) atTop (𝓝 0) := by
  filter_upwards [crossCovEstimator_tendsto_zero_ae t U V hY hU hV] with ω hω
  exact tendsto_matrix_of_entries fun i j => by simpa using hω i j
/-- **The outer-product family is centred i.i.d., from the paper's own hypotheses.** Under (A1)
the pairs `(θ̃⁽ˢ⁾, g⁽ˢ⁾)` are pairwise independent and identically distributed across the window
(each summand "depends on batch `s` alone"), and case (iii) makes the two components of a pair
independent; the population zero of the outer product is then *derived* by
`integral_outerE_eq_zero_of_indepFun`, not packaged. The transport is `IndepFun.comp` and
`IdentDistrib.comp` along `measurable_outerE_pair`. -/
theorem isCentredIID_outerE {μ : Measure Ω} (U V : ℕ → Ω → Fin d → ℝ)
    (hpair : Pairwise fun a b =>
      IndepFun (fun ω => (U a ω, V a ω)) (fun ω => (U b ω, V b ω)) μ)
    (hident : ∀ s, IdentDistrib (fun ω => (U s ω, V s ω)) (fun ω => (U 0 ω, V 0 ω)) μ μ)
    (hstep : IndepFun (U 0) (V 0) μ)
    (hUm : ∀ i, AEStronglyMeasurable (fun ω => U 0 ω i) μ)
    (hVm : ∀ j, AEStronglyMeasurable (fun ω => V 0 ω j) μ)
    (hUc : ∀ i, ∫ ω, U 0 ω i ∂μ = 0)
    (hint : Integrable (fun ω => outerE (U 0 ω) (V 0 ω)) μ) :
    IsCentredIID μ fun s ω => outerE (U s ω) (V s ω) where
  integrable := hint
  indep := fun _ _ hab => (hpair hab).comp measurable_outerE_pair measurable_outerE_pair
  ident := fun s => (hident s).comp measurable_outerE_pair
  mean_zero := integral_outerE_eq_zero_of_indepFun hstep hUm hVm hUc hint

/-- The first components of a pairwise independent, identically distributed family of pairs form
a centred i.i.d. family on the Euclidean carrier; transport along `measurable_vecE_fst`. -/
theorem isCentredIID_vecE_fst {μ : Measure Ω} (U V : ℕ → Ω → Fin d → ℝ)
    (hpair : Pairwise fun a b =>
      IndepFun (fun ω => (U a ω, V a ω)) (fun ω => (U b ω, V b ω)) μ)
    (hident : ∀ s, IdentDistrib (fun ω => (U s ω, V s ω)) (fun ω => (U 0 ω, V 0 ω)) μ μ)
    (hint : Integrable (fun ω => vecE (U 0 ω)) μ)
    (hUc : ∀ i, ∫ ω, U 0 ω i ∂μ = 0) :
    IsCentredIID μ fun s ω => vecE (U s ω) where
  integrable := hint
  indep := fun _ _ hab => (hpair hab).comp measurable_vecE_fst measurable_vecE_fst
  ident := fun s => (hident s).comp measurable_vecE_fst
  mean_zero := integral_vecE_eq_zero hint hUc

/-- The second components likewise; transport along `measurable_vecE_snd`. -/
theorem isCentredIID_vecE_snd {μ : Measure Ω} (U V : ℕ → Ω → Fin d → ℝ)
    (hpair : Pairwise fun a b =>
      IndepFun (fun ω => (U a ω, V a ω)) (fun ω => (U b ω, V b ω)) μ)
    (hident : ∀ s, IdentDistrib (fun ω => (U s ω, V s ω)) (fun ω => (U 0 ω, V 0 ω)) μ μ)
    (hint : Integrable (fun ω => vecE (V 0 ω)) μ)
    (hVc : ∀ j, ∫ ω, V 0 ω j ∂μ = 0) :
    IsCentredIID μ fun s ω => vecE (V s ω) where
  integrable := hint
  indep := fun _ _ hab => (hpair hab).comp measurable_vecE_snd measurable_vecE_snd
  ident := fun s => (hident s).comp measurable_vecE_snd
  mean_zero := integral_vecE_eq_zero hint hVc

/-- **Case (iii)'s almost-sure conclusion, from the paper-shaped hypotheses.** The inputs are
(A1)'s modelling of the window — the pairs `(θ̃⁽ˢ⁾, g⁽ˢ⁾)` pairwise independent and identically
distributed across `s` — together with case (iii)'s independence of the two components at a
step, the centring `E[δθ̃ | φ] = E[δg | φ] = 0` of the construction, and integrability side
conditions. The population zero of the summands is **derived** from independence and centring
(`integral_outerE_eq_zero_of_indepFun`), not assumed; the three packaged families of
`crossCovEstimator_tendsto_zero_ae` are assembled by the transport lemmas above. -/
theorem crossCovEstimator_tendsto_zero_ae_of_indepFun {μ : Measure Ω} (t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ)
    (hpair : Pairwise fun a b =>
      IndepFun (fun ω => (U a ω, V a ω)) (fun ω => (U b ω, V b ω)) μ)
    (hident : ∀ s, IdentDistrib (fun ω => (U s ω, V s ω)) (fun ω => (U 0 ω, V 0 ω)) μ μ)
    (hstep : IndepFun (U 0) (V 0) μ)
    (hUm : ∀ i, AEStronglyMeasurable (fun ω => U 0 ω i) μ)
    (hVm : ∀ j, AEStronglyMeasurable (fun ω => V 0 ω j) μ)
    (hUc : ∀ i, ∫ ω, U 0 ω i ∂μ = 0)
    (hVc : ∀ j, ∫ ω, V 0 ω j ∂μ = 0)
    (hYint : Integrable (fun ω => outerE (U 0 ω) (V 0 ω)) μ)
    (hUint : Integrable (fun ω => vecE (U 0 ω)) μ)
    (hVint : Integrable (fun ω => vecE (V 0 ω)) μ) :
    ∀ᵐ ω ∂μ, ∀ i j : Fin d, Tendsto (fun T : ℕ =>
        crossCovEstimator T t (windowCentre T t fun s => U s ω)
          (windowCentre T t fun s => V s ω) i j) atTop (𝓝 0) :=
  crossCovEstimator_tendsto_zero_ae t U V
    (isCentredIID_outerE U V hpair hident hstep hUm hVm hUc hYint)
    (isCentredIID_vecE_fst U V hpair hident hUint hUc)
    (isCentredIID_vecE_snd U V hpair hident hVint hVc)

/-- **Case (iii)'s almost-sure conclusion for the estimator built from the *raw* iterates**, at
the matrix level. The hypotheses are the packaged i.i.d.-and-centred conditions on the
population-centred fluctuations `θ̃⁽ˢ⁾ - c_θ`, `g⁽ˢ⁾ - c_g` — assembled from the paper's own
hypotheses by `isCentredIID_outerE`, `isCentredIID_vecE_fst` and `isCentredIID_vecE_snd` — while
the conclusion is about `Σ̂⁽ᵗ⁾` formed from the raw families, which is what the paper writes.
The transport is exact rather than asymptotic: for every `T ≠ 0` the two estimators are equal
(`crossCovEstimator_windowCentred_sub_const`), and every `T ≥ 1` is eventually in the filter. -/
theorem crossCovEstimator_raw_tendsto_zero_ae {μ : Measure Ω} (t : ℕ)
    (Θ G : ℕ → Ω → Fin d → ℝ) (cΘ cG : Fin d → ℝ)
    (hY : IsCentredIID μ fun s ω => outerE (Θ s ω - cΘ) (G s ω - cG))
    (hU : IsCentredIID μ fun s ω => vecE (Θ s ω - cΘ))
    (hV : IsCentredIID μ fun s ω => vecE (G s ω - cG)) :
    ∀ᵐ ω ∂μ, Tendsto (fun T : ℕ =>
        crossCovEstimator T t (windowCentre T t fun s => Θ s ω)
          (windowCentre T t fun s => G s ω)) atTop (𝓝 0) := by
  filter_upwards [crossCovEstimator_tendsto_zero_ae_matrix t
    (fun s ω => Θ s ω - cΘ) (fun s ω => G s ω - cG) hY hU hV] with ω hω
  refine hω.congr' ?_
  filter_upwards [eventually_ge_atTop 1] with T hT
  exact crossCovEstimator_windowCentred_sub_const T t (by omega) _ _ _ _

end StrongLaw

/-! ### The finite-window fluctuation: the `L²` form of `O(T^{-1/2})` -/

section L2Rate

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- A square-integrable, pairwise independent, identically distributed, centred real family. -/
structure IsSquareCentredIID (μ : Measure Ω) (Z : ℕ → Ω → ℝ) : Prop where
  /-- each summand is square integrable -/
  memLp : ∀ s, MemLp (Z s) 2 μ
  /-- the summands are pairwise independent -/
  indep : Pairwise fun i j => IndepFun (Z i) (Z j) μ
  /-- the summands are identically distributed -/
  ident : ∀ i, IdentDistrib (Z i) (Z 0) μ μ
  /-- the population mean is zero -/
  mean_zero : ∫ ω, Z 0 ω ∂μ = 0

theorem IsSquareCentredIID.mean_zero' {μ : Measure Ω} {Z : ℕ → Ω → ℝ}
    (h : IsSquareCentredIID μ Z) (s : ℕ) : ∫ ω, Z s ω ∂μ = 0 := by
  rw [(h.ident s).integral_eq]; exact h.mean_zero

/-- **A time shift of a square-integrable centred i.i.d. family is again such a family.** The
trailing window `{t - T, …, t - 1}` of the paper's estimator is the shift by `t - T` of the
prefix window, so the `L²` statements below hold for **every** iteration `t` and every window
length `T`, not merely in the prefix regime `t ≤ T`. -/
theorem IsSquareCentredIID.shift {μ : Measure Ω} {Z : ℕ → Ω → ℝ}
    (h : IsSquareCentredIID μ Z) (c : ℕ) : IsSquareCentredIID μ fun s => Z (c + s) where
  memLp := fun s => h.memLp (c + s)
  indep := fun _ _ hab => h.indep fun hcc => hab (Nat.add_left_cancel hcc)
  ident := fun i => (h.ident (c + i)).trans (h.ident c).symm
  mean_zero := h.mean_zero' c

/-- **The variance of the window average.** For pairwise independent, identically distributed,
square-integrable summands, `Var[(1/T) Σ Z] = Var[Z] / T` exactly. This identity, not a
central limit theorem, is what carries the paper's `O(T^{-1/2})`: mathlib's central limit theorem
is one-dimensional and rate-free, and no Berry–Esseen bound exists in this version, so a genuine
`T^{-1/2}` rate can only be stated in `L²`. -/
theorem variance_average_of_iid {μ : Measure Ω} (Z : ℕ → Ω → ℝ) (T : ℕ)
    (h : IsSquareCentredIID μ Z) :
    Var[fun ω => (T : ℝ)⁻¹ * ∑ s ∈ range T, Z s ω; μ] = Var[Z 0; μ] / T := by
  rcases Nat.eq_zero_or_pos T with rfl | hT
  · have h0 : (fun ω => ((0 : ℕ) : ℝ)⁻¹ * ∑ s ∈ range 0, Z s ω) = (0 : Ω → ℝ) := by
      funext ω; simp
    rw [h0, variance_zero, Nat.cast_zero, div_zero]
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT.ne'
  have hsum : (fun ω => ∑ s ∈ range T, Z s ω) = (∑ s ∈ range T, Z s) := by
    funext ω; simp [Finset.sum_apply]
  have h1 : Var[fun ω => (T : ℝ)⁻¹ * ∑ s ∈ range T, Z s ω; μ]
      = ((T : ℝ)⁻¹) ^ 2 * Var[fun ω => ∑ s ∈ range T, Z s ω; μ] := variance_const_mul _ _ _
  have h2 : Var[fun ω => ∑ s ∈ range T, Z s ω; μ] = ∑ s ∈ range T, Var[Z s; μ] := by
    rw [hsum]
    exact IndepFun.variance_sum (fun i _ => h.memLp i) (fun i _ j _ hij => h.indep hij)
  have h3 : ∀ s ∈ range T, Var[Z s; μ] = Var[Z 0; μ] := fun s _ => (h.ident s).variance_eq
  rw [h1, h2, Finset.sum_congr rfl h3, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  field_simp

/-- The mean square of the window average is the summand variance divided by `T`. -/
theorem integral_sq_average_of_iid {μ : Measure Ω} [IsFiniteMeasure μ] (Z : ℕ → Ω → ℝ) (T : ℕ)
    (h : IsSquareCentredIID μ Z) :
    ∫ ω, ((T : ℝ)⁻¹ * ∑ s ∈ range T, Z s ω) ^ 2 ∂μ = Var[Z 0; μ] / T := by
  have hAE : AEMeasurable (fun ω => (T : ℝ)⁻¹ * ∑ s ∈ range T, Z s ω) μ := by
    have heq : (fun ω => ∑ s ∈ range T, Z s ω) = (∑ s ∈ range T, Z s) := by
      funext ω; simp [Finset.sum_apply]
    have hs : AEMeasurable (fun ω => ∑ s ∈ range T, Z s ω) μ := by
      rw [heq]; exact Finset.aemeasurable_sum _ (fun s _ => (h.memLp s).aemeasurable)
    exact AEMeasurable.const_mul hs _
  have hint : ∫ ω, (T : ℝ)⁻¹ * ∑ s ∈ range T, Z s ω ∂μ = 0 := by
    rw [integral_const_mul,
      integral_finsetSum _ (fun s _ => (h.memLp s).integrable one_le_two)]
    simp [h.mean_zero']
  rw [← variance_of_integral_eq_zero hAE hint]
  exact variance_average_of_iid Z T h

/-- **The `L²` error of the window average is `σ / √T`.** This is the paper's `O(T^{-1/2})`, in
the only form this mathlib can carry it honestly. An almost-sure `O(T^{-1/2})` rate is not
claimed: it would require a law of the iterated logarithm, which mathlib does not have. -/
theorem l2_error_average {μ : Measure Ω} [IsFiniteMeasure μ] (Z : ℕ → Ω → ℝ) (T : ℕ)
    (h : IsSquareCentredIID μ Z) :
    √(∫ ω, ((T : ℝ)⁻¹ * ∑ s ∈ range T, Z s ω) ^ 2 ∂μ) = √(Var[Z 0; μ]) / √T := by
  rw [integral_sq_average_of_iid Z T h, Real.sqrt_div (variance_nonneg _ _)]

/-- The mean square of an entry of the estimator built from the population-centred fluctuations
is the summand variance divided by `T` — for **every** iteration `t` and every window length
`T`: the trailing window is the shift by `t - T` of the prefix window
(`IsSquareCentredIID.shift`), and shifting an i.i.d. family changes neither the variance
identity nor the summand variance. -/
theorem integral_sq_crossCovEstimator_entry {μ : Measure Ω} [IsFiniteMeasure μ] (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (h : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j) :
    ∫ ω, (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) ^ 2 ∂μ
      = Var[fun ω => U 0 ω i * V 0 ω j; μ] / T := by
  have hentry : ∀ ω, crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j
      = (T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i * V (t - T + s) ω j := by
    intro ω
    simp only [crossCovEstimator, Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply,
      smul_eq_mul]
  simp only [hentry]
  calc ∫ ω, ((T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i * V (t - T + s) ω j) ^ 2 ∂μ
      = Var[fun ω => U (t - T) ω i * V (t - T) ω j; μ] / T :=
        integral_sq_average_of_iid _ T (h.shift (t - T))
    _ = Var[fun ω => U 0 ω i * V 0 ω j; μ] / T :=
        congrArg (· / (T : ℝ)) (h.ident (t - T)).variance_eq

/-- **The finite-window fluctuation of an entry of the estimator is `O(T^{-1/2})` in `L²`.**
For every iteration `t` and every window length `T`, the `(i,j)` entry of the estimator built
from the population-centred fluctuations has `L²` norm `σᵢⱼ / √T`, with `σᵢⱼ²` the variance of a
single summand `δθ̃ᵢ δgⱼ`. -/
theorem crossCovEstimator_entry_l2_rate {μ : Measure Ω} [IsFiniteMeasure μ] (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (h : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j) :
    √(∫ ω, (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) ^ 2 ∂μ)
      = √(Var[fun ω => U 0 ω i * V 0 ω j; μ]) / √T := by
  rw [integral_sq_crossCovEstimator_entry T t U V i j h,
    Real.sqrt_div (variance_nonneg _ _)]

/-- **The mean-product term of the identity is `O(T^{-1})` in `L²`.** Its mean square is the
product of the two window-mean variances, each of which is `σ²/T`; so its `L²` norm is
`σ_u σ_v / T`, one order smaller than the leading average. This is the paper's `O(T^{-1})` claim
for that term, again in `L²` form.

The hypothesis `hM` — independence of the two window means — is the paper's own step: under (A1)
the whole window collection of iterate fluctuations is independent of the whole collection of
gradient fluctuations given `φ`, so any functional of one is independent of any functional of the
other. It is carried here at the level at which it is used, and
`indepFun_windowMean_windowMean` discharges it from the collection-level hypothesis. The
statement holds for every iteration `t` and every window length `T`. -/
theorem integral_sq_windowMean_mul {μ : Measure Ω} [IsFiniteMeasure μ] (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (hU : IsSquareCentredIID μ fun s ω => U s ω i)
    (hV : IsSquareCentredIID μ fun s ω => V s ω j)
    (hM : IndepFun (fun ω => windowMean T t (fun s => U s ω) i)
                   (fun ω => windowMean T t (fun s => V s ω) j) μ) :
    ∫ ω, (windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j) ^ 2 ∂μ
      = (Var[fun ω => U 0 ω i; μ] / T) * (Var[fun ω => V 0 ω j; μ] / T) := by
  have hUeq : ∀ ω, windowMean T t (fun s => U s ω) i
      = (T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i := by
    intro ω; simp [windowMean_apply]
  have hVeq : ∀ ω, windowMean T t (fun s => V s ω) j
      = (T : ℝ)⁻¹ * ∑ s ∈ range T, V (t - T + s) ω j := by
    intro ω; simp [windowMean_apply]
  have hUm : AEStronglyMeasurable (fun ω => windowMean T t (fun s => U s ω) i ^ 2) μ := by
    have heq : (fun ω => ∑ s ∈ range T, U (t - T + s) ω i)
        = (∑ s ∈ range T, fun ω => U (t - T + s) ω i) := by
      funext ω; simp [Finset.sum_apply]
    have hs : AEMeasurable (fun ω => ∑ s ∈ range T, U (t - T + s) ω i) μ := by
      rw [heq]; exact Finset.aemeasurable_sum _ (fun s _ => (hU.memLp (t - T + s)).aemeasurable)
    have hmean : AEMeasurable (fun ω => windowMean T t (fun s => U s ω) i) μ := by
      simp only [hUeq]
      exact AEMeasurable.const_mul hs _
    exact (hmean.pow_const 2).aestronglyMeasurable
  have hVm : AEStronglyMeasurable (fun ω => windowMean T t (fun s => V s ω) j ^ 2) μ := by
    have heq : (fun ω => ∑ s ∈ range T, V (t - T + s) ω j)
        = (∑ s ∈ range T, fun ω => V (t - T + s) ω j) := by
      funext ω; simp [Finset.sum_apply]
    have hs : AEMeasurable (fun ω => ∑ s ∈ range T, V (t - T + s) ω j) μ := by
      rw [heq]; exact Finset.aemeasurable_sum _ (fun s _ => (hV.memLp (t - T + s)).aemeasurable)
    have hmean : AEMeasurable (fun ω => windowMean T t (fun s => V s ω) j) μ := by
      simp only [hVeq]
      exact AEMeasurable.const_mul hs _
    exact (hmean.pow_const 2).aestronglyMeasurable
  have hsq : IndepFun (fun ω => windowMean T t (fun s => U s ω) i ^ 2)
      (fun ω => windowMean T t (fun s => V s ω) j ^ 2) μ :=
    hM.comp (measurable_id.pow_const 2) (measurable_id.pow_const 2)
  have hfac := hsq.integral_mul_eq_mul_integral hUm hVm
  have hUint : ∫ ω, windowMean T t (fun s => U s ω) i ^ 2 ∂μ = Var[fun ω => U 0 ω i; μ] / T := by
    simp only [hUeq]
    calc ∫ ω, ((T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i) ^ 2 ∂μ
        = Var[fun ω => U (t - T) ω i; μ] / T :=
          integral_sq_average_of_iid _ T (hU.shift (t - T))
      _ = Var[fun ω => U 0 ω i; μ] / T :=
          congrArg (· / (T : ℝ)) (hU.ident (t - T)).variance_eq
  have hVint : ∫ ω, windowMean T t (fun s => V s ω) j ^ 2 ∂μ = Var[fun ω => V 0 ω j; μ] / T := by
    simp only [hVeq]
    calc ∫ ω, ((T : ℝ)⁻¹ * ∑ s ∈ range T, V (t - T + s) ω j) ^ 2 ∂μ
        = Var[fun ω => V (t - T) ω j; μ] / T :=
          integral_sq_average_of_iid _ T (hV.shift (t - T))
      _ = Var[fun ω => V 0 ω j; μ] / T :=
          congrArg (· / (T : ℝ)) (hV.ident (t - T)).variance_eq
  calc ∫ ω, (windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j) ^ 2 ∂μ
      = ∫ ω, windowMean T t (fun s => U s ω) i ^ 2
          * windowMean T t (fun s => V s ω) j ^ 2 ∂μ := by
        simp [mul_pow]
    _ = (∫ ω, windowMean T t (fun s => U s ω) i ^ 2 ∂μ)
          * ∫ ω, windowMean T t (fun s => V s ω) j ^ 2 ∂μ := by
        simpa [Pi.mul_apply] using hfac
    _ = (Var[fun ω => U 0 ω i; μ] / T) * (Var[fun ω => V 0 ω j; μ] / T) := by
        rw [hUint, hVint]

/-- An entry of the estimator is square integrable when the summands are: it is a fixed finite
linear combination of them. This discharges one of the integrability side conditions that
`crossCovEstimator_windowCentred_l2_bound` carries. -/
theorem memLp_crossCovEstimator_entry {μ : Measure Ω} (T t : ℕ) (U V : ℕ → Ω → Fin d → ℝ)
    (i j : Fin d) (hY : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j) :
    MemLp (fun ω => crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) 2 μ := by
  have hentry : (fun ω => crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j)
      = fun ω => (T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i * V (t - T + s) ω j := by
    funext ω
    simp only [crossCovEstimator, Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply,
      smul_eq_mul]
  rw [hentry]
  exact (memLp_finsetSum (range T) fun s _ => hY.memLp (t - T + s)).const_mul _

/-- A window mean is square integrable when the summands are. -/
theorem memLp_windowMean {μ : Measure Ω} (T t : ℕ) (U : ℕ → Ω → Fin d → ℝ) (i : Fin d)
    (hU : IsSquareCentredIID μ fun s ω => U s ω i) :
    MemLp (fun ω => windowMean T t (fun s => U s ω) i) 2 μ := by
  have hentry : (fun ω => windowMean T t (fun s => U s ω) i)
      = fun ω => (T : ℝ)⁻¹ * ∑ s ∈ range T, U (t - T + s) ω i := by
    funext ω; simp [windowMean_apply]
  rw [hentry]
  exact (memLp_finsetSum (range T) fun s _ => hU.memLp (t - T + s)).const_mul _

/-- The mean-product term has an integrable square: each window mean is square integrable, and
the two are independent, so the product of their squares is integrable. Note that independence is
genuinely used — two square-integrable factors need not have an integrable product. -/
theorem integrable_sq_windowMean_mul {μ : Measure Ω} (T t : ℕ) (U V : ℕ → Ω → Fin d → ℝ)
    (i j : Fin d)
    (hU : IsSquareCentredIID μ fun s ω => U s ω i)
    (hV : IsSquareCentredIID μ fun s ω => V s ω j)
    (hM : IndepFun (fun ω => windowMean T t (fun s => U s ω) i)
                   (fun ω => windowMean T t (fun s => V s ω) j) μ) :
    Integrable (fun ω =>
      (windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j) ^ 2) μ := by
  have hsq : IndepFun (fun ω => windowMean T t (fun s => U s ω) i ^ 2)
      (fun ω => windowMean T t (fun s => V s ω) j ^ 2) μ :=
    hM.comp (measurable_id.pow_const 2) (measurable_id.pow_const 2)
  refine (hsq.integrable_mul (memLp_windowMean T t U i hU).integrable_sq
    (memLp_windowMean T t V j hV).integrable_sq).congr ?_
  filter_upwards with ω
  simp [Pi.mul_apply, mul_pow]
/-- **The finite-window fluctuation of the estimator the paper actually forms, with the
mean-product remainder carried explicitly into the conclusion.** The window-mean-centred
estimator is the difference of the two objects just measured, so `(x - y)^2 ≤ 2x^2 + 2y^2` gives
`E[(Σ̂⁽ᵗ⁾)ᵢⱼ²] ≤ 2 σᵢⱼ²/T + 2 σ_uᵢ² σ_vⱼ²/T²`, for every iteration `t` and every window length
`T` (at `T = 0` both sides vanish).
The leading term is the paper's `O(T^{-1/2})` fluctuation in `L²`; the second term is the paper's
`O(T^{-1})` mean-product correction, and it appears in the conclusion at its own order rather than
being dropped. The constant `2` is the crude constant of the elementary inequality
`(x - y)² ≤ 2x² + 2y²`; a Minkowski argument would replace the bound by
`√(E[(Σ̂⁽ᵗ⁾)ᵢⱼ²]) ≤ σᵢⱼ/√T + σ_uᵢ σ_vⱼ/T`, which is sharper but not stronger in order.
The independence `hM` and the three integrability side conditions are carried explicitly here so
that the statement says exactly what it consumes; they are all derived from the packaged
hypotheses in `crossCovEstimator_windowCentred_l2_bound_of_indepFun`. -/
theorem crossCovEstimator_windowCentred_l2_bound {μ : Measure Ω} [IsFiniteMeasure μ] (T t : ℕ)
    (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (hY : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j)
    (hU : IsSquareCentredIID μ fun s ω => U s ω i)
    (hV : IsSquareCentredIID μ fun s ω => V s ω j)
    (hM : IndepFun (fun ω => windowMean T t (fun s => U s ω) i)
                   (fun ω => windowMean T t (fun s => V s ω) j) μ)
    (hAint : Integrable (fun ω =>
      (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) ^ 2) μ)
    (hBint : Integrable (fun ω =>
      (windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j) ^ 2) μ)
    (hDint : Integrable (fun ω =>
      (crossCovEstimator T t (windowCentre T t fun s => U s ω)
        (windowCentre T t fun s => V s ω) i j) ^ 2) μ) :
    ∫ ω, (crossCovEstimator T t (windowCentre T t fun s => U s ω)
        (windowCentre T t fun s => V s ω) i j) ^ 2 ∂μ
      ≤ 2 * (Var[fun ω => U 0 ω i * V 0 ω j; μ] / T)
        + 2 * ((Var[fun ω => U 0 ω i; μ] / T) * (Var[fun ω => V 0 ω j; μ] / T)) := by
  have hid : ∀ ω, crossCovEstimator T t (windowCentre T t fun s => U s ω)
      (windowCentre T t fun s => V s ω) i j
      = crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j
        - windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j := by
    intro ω
    rw [sample_crossCov_identity T t]
    simp [Matrix.sub_apply, Matrix.vecMulVec_apply]
  have hbound : (fun ω => (crossCovEstimator T t (windowCentre T t fun s => U s ω)
        (windowCentre T t fun s => V s ω) i j) ^ 2)
      ≤ fun ω => 2 * (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) ^ 2
        + 2 * (windowMean T t (fun s => U s ω) i
              * windowMean T t (fun s => V s ω) j) ^ 2 := by
    intro ω
    dsimp only
    rw [hid ω]
    nlinarith [sq_nonneg (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j
      + windowMean T t (fun s => U s ω) i * windowMean T t (fun s => V s ω) j)]
  have hgint : Integrable (fun ω =>
      2 * (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) ^ 2
        + 2 * (windowMean T t (fun s => U s ω) i
              * windowMean T t (fun s => V s ω) j) ^ 2) μ :=
    (hAint.const_mul 2).add (hBint.const_mul 2)
  have hmono := integral_mono hDint hgint hbound
  have hval : ∫ ω, (2 * (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j) ^ 2
        + 2 * (windowMean T t (fun s => U s ω) i
              * windowMean T t (fun s => V s ω) j) ^ 2) ∂μ
      = 2 * (Var[fun ω => U 0 ω i * V 0 ω j; μ] / T)
        + 2 * ((Var[fun ω => U 0 ω i; μ] / T) * (Var[fun ω => V 0 ω j; μ] / T)) := by
    rw [integral_add (hAint.const_mul 2) (hBint.const_mul 2), integral_const_mul,
      integral_const_mul, integral_sq_crossCovEstimator_entry T t U V i j hY,
      integral_sq_windowMean_mul T t U V i j hU hV hM]
  rw [hval] at hmono
  exact hmono

/-- **The finite-window `L²` bound with every side condition discharged.** The only inputs are
(A1)'s square-integrable i.i.d. window — for the summands and for each of the two channels
separately — and case (iii)'s collection-level conditional independence `hcoll`, which is exactly
the paper's step that under (A1) the whole window collection `{θ̃⁽ˢ⁾}` is independent of
`{g⁽ˢ⁾}` given `φ`. The independence of the two window means and the three integrability
conditions that `crossCovEstimator_windowCentred_l2_bound` carries are all derived here, so
nothing about the `L²` statement is left as an unexplained hypothesis. -/
theorem crossCovEstimator_windowCentred_l2_bound_of_indepFun {μ : Measure Ω} [IsFiniteMeasure μ]
    (T t : ℕ) (U V : ℕ → Ω → Fin d → ℝ) (i j : Fin d)
    (hY : IsSquareCentredIID μ fun s ω => U s ω i * V s ω j)
    (hU : IsSquareCentredIID μ fun s ω => U s ω i)
    (hV : IsSquareCentredIID μ fun s ω => V s ω j)
    (hcoll : IndepFun (fun ω s => U s ω) (fun ω s => V s ω) μ) :
    ∫ ω, (crossCovEstimator T t (windowCentre T t fun s => U s ω)
        (windowCentre T t fun s => V s ω) i j) ^ 2 ∂μ
      ≤ 2 * (Var[fun ω => U 0 ω i * V 0 ω j; μ] / T)
        + 2 * ((Var[fun ω => U 0 ω i; μ] / T) * (Var[fun ω => V 0 ω j; μ] / T)) := by
  have hM := indepFun_windowMean_windowMean T t U V i j hcoll
  have hAmem := memLp_crossCovEstimator_entry T t U V i j hY
  have hBint := integrable_sq_windowMean_mul T t U V i j hU hV hM
  have hBmem : MemLp (fun ω => windowMean T t (fun s => U s ω) i
      * windowMean T t (fun s => V s ω) j) 2 μ := by
    refine (memLp_two_iff_integrable_sq ?_).2 hBint
    exact (memLp_windowMean T t U i hU).aestronglyMeasurable.mul
      (memLp_windowMean T t V j hV).aestronglyMeasurable
  have hDmem : MemLp (fun ω => crossCovEstimator T t (windowCentre T t fun s => U s ω)
      (windowCentre T t fun s => V s ω) i j) 2 μ := by
    have heq : (fun ω => crossCovEstimator T t (windowCentre T t fun s => U s ω)
        (windowCentre T t fun s => V s ω) i j)
        = (fun ω => crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) i j)
          - (fun ω => windowMean T t (fun s => U s ω) i
              * windowMean T t (fun s => V s ω) j) := by
      funext ω
      rw [Pi.sub_apply, sample_crossCov_identity T t]
      simp [Matrix.sub_apply, Matrix.vecMulVec_apply]
    rw [heq]
    exact hAmem.sub hBmem
  exact crossCovEstimator_windowCentred_l2_bound T t U V i j hY hU hV hM
    hAmem.integrable_sq hBint hDmem.integrable_sq

/-- **The same `L²` bound for the estimator built from the *raw* iterates.** Transported along
`crossCovEstimator_windowCentred_sub_const`, exactly as with unbiasedness and the almost-sure
limit: the hypotheses are about the population-centred fluctuations, the conclusion is about the
`Σ̂⁽ᵗ⁾` the paper writes down. -/
theorem crossCovEstimator_raw_windowCentred_l2_bound {μ : Measure Ω} [IsFiniteMeasure μ]
    (T t : ℕ) (hT : T ≠ 0) (Θ G : ℕ → Ω → Fin d → ℝ) (cΘ cG : Fin d → ℝ) (i j : Fin d)
    (hY : IsSquareCentredIID μ fun s ω => (Θ s ω i - cΘ i) * (G s ω j - cG j))
    (hU : IsSquareCentredIID μ fun s ω => Θ s ω i - cΘ i)
    (hV : IsSquareCentredIID μ fun s ω => G s ω j - cG j)
    (hcoll : IndepFun (fun ω s => Θ s ω - cΘ) (fun ω s => G s ω - cG) μ) :
    ∫ ω, (crossCovEstimator T t (windowCentre T t fun s => Θ s ω)
        (windowCentre T t fun s => G s ω) i j) ^ 2 ∂μ
      ≤ 2 * (Var[fun ω => (Θ 0 ω i - cΘ i) * (G 0 ω j - cG j); μ] / T)
        + 2 * ((Var[fun ω => Θ 0 ω i - cΘ i; μ] / T)
              * (Var[fun ω => G 0 ω j - cG j; μ] / T)) := by
  have h := crossCovEstimator_windowCentred_l2_bound_of_indepFun T t
    (fun s ω => Θ s ω - cΘ) (fun s ω => G s ω - cG) i j hY hU hV hcoll
  simpa only [crossCovEstimator_windowCentred_sub_const T t hT, Pi.sub_apply] using h

end L2Rate

/-! ### Non-vacuity: case (iii)'s hypotheses have a non-degenerate model -/

section Witness

/-- The second moment of the standard real Gaussian is `1`. Used only to build the witness model
below, where it is what makes the estimator's fluctuation nonzero. -/
theorem integral_sq_gaussianReal_std : ∫ x, x ^ 2 ∂(gaussianReal 0 1) = 1 := by
  have h := variance_of_integral_eq_zero (X := (id : ℝ → ℝ)) (μ := gaussianReal 0 1)
    aemeasurable_id (by simp)
  simp only [id] at h
  rw [← h]
  simp

/-- The square of the identity is integrable against the standard real Gaussian. -/
theorem integrable_sq_gaussianReal_std : Integrable (fun x : ℝ => x ^ 2) (gaussianReal 0 1) := by
  simpa using (memLp_id_gaussianReal' (μ := 0) (v := 1) 2 (by simp)).integrable_sq

/-- The population cross-moment of the two channels vanishes under the product law — this is
case (iii)'s population zero, holding in the witness model by construction. -/
theorem integral_mul_gaussianProd :
    ∫ p : ℝ × ℝ, p.1 * p.2 ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 0 := by
  rw [integral_prod_mul (fun x => x) (fun y => y)]
  simp

/-- The summand of the estimator has second moment `1` under the product law: the model is
**not** degenerate, which is the whole point of exhibiting it. -/
theorem integral_sq_mul_gaussianProd :
    ∫ p : ℝ × ℝ, (p.1 * p.2) ^ 2 ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 1 := by
  simp only [mul_pow]
  rw [integral_prod_mul (fun x => x ^ 2) (fun y => y ^ 2), integral_sq_gaussianReal_std, mul_one]

/-- The summand of the estimator is square integrable under the product law. -/
theorem memLp_mul_gaussianProd :
    MemLp (fun p : ℝ × ℝ => p.1 * p.2) 2 ((gaussianReal 0 1).prod (gaussianReal 0 1)) := by
  refine (memLp_two_iff_integrable_sq
    (measurable_fst.mul measurable_snd).aestronglyMeasurable).2 ?_
  refine (integrable_sq_gaussianReal_std.mul_prod integrable_sq_gaussianReal_std).congr ?_
  filter_upwards with p
  simp [mul_pow]

/-- **The hypotheses of case (iii) are satisfiable, and satisfiable non-degenerately.** A referee
is entitled to ask whether the packaged hypotheses `IsCentredIID` and `IsSquareCentredIID`, the
per-step independence `δθ̃ ⫫ δg`, and the two centrings can all hold at once over a genuine
probability measure without forcing the estimator to be zero. If they could not, every theorem
above would be vacuously true and case (iii) would be indistinguishable from the two
identically-zero deletions of `CrossCov.lean`. They can. Take the i.i.d. sequence supplied by
`ProbabilityTheory.exists_iid` for the law of a pair of independent standard Gaussians, with
`δθ̃⁽ˢ⁾` the first coordinate and `δg⁽ˢ⁾` the second; then all of the following hold at once:

* `μ` is a probability measure — in particular not the zero measure, under which independence,
  identical distribution and centring are all trivially true and nothing is being asserted;
* the two channels are independent at a step and each is centred, which is case (iii)'s
  hypothesis together with the paper's centring construction;
* all three `IsCentredIID` packages that `crossCovEstimator_tendsto_zero_ae` consumes hold, so
  the almost-sure conclusion is not vacuous;
* the summands `δθ̃⁽ˢ⁾ δg⁽ˢ⁾` satisfy `IsSquareCentredIID`, so the `L²` conclusions are not
  vacuous either;
* the mean square of the `(0,0)` entry of the estimator built from these fluctuations — the
  i.i.d.-average term of the sample-cross-covariance identity, the object
  `integral_sq_crossCovEstimator_entry` measures — is exactly `1/T`, so for every nonempty window
  the estimator is **not** almost surely zero.

Together with `crossCovEstimator_windowCentred_ne_zero`, which exhibits a window on which the
*window-mean-centred* estimator is nonzero, this is the paper's sentence "Unlike cases (i) and
(ii), this does not make the finite-window estimator identically zero", proved rather than
asserted. -/
theorem exists_nondegenerate_case_iii_model :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (U V : ℕ → Ω → Fin 1 → ℝ),
      IsProbabilityMeasure μ ∧
      IndepFun (U 0) (V 0) μ ∧
      (∀ i, ∫ ω, U 0 ω i ∂μ = 0) ∧
      (∀ j, ∫ ω, V 0 ω j ∂μ = 0) ∧
      (IsCentredIID μ fun s ω => outerE (U s ω) (V s ω)) ∧
      (IsCentredIID μ fun s ω => vecE (U s ω)) ∧
      (IsCentredIID μ fun s ω => vecE (V s ω)) ∧
      (IsSquareCentredIID μ fun s ω => U s ω 0 * V s ω 0) ∧
      (∀ T t : ℕ, ∫ ω, (crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) 0 0) ^ 2 ∂μ
        = 1 / T) ∧
      ∀ T t : ℕ, T ≠ 0 →
        ¬ ∀ᵐ ω ∂μ, crossCovEstimator T t (fun s => U s ω) (fun s => V s ω) 0 0 = 0 := by
  obtain ⟨Ω, mΩ, P, W, hmeas, hlaw, hindep, hprob⟩ :=
    ProbabilityTheory.exists_iid ℕ ((gaussianReal 0 1).prod (gaussianReal 0 1))
  haveI := hprob
  have hmul : Measurable fun p : ℝ × ℝ => p.1 * p.2 := measurable_fst.mul measurable_snd
  have hconst : Measurable fun x : ℝ => (fun _ : Fin 1 => x) :=
    measurable_pi_lambda _ fun _ => measurable_id
  have hpairmap : Measurable fun p : ℝ × ℝ =>
      ((fun _ : Fin 1 => p.1), (fun _ : Fin 1 => p.2)) := by fun_prop
  have hidW : ∀ s, IdentDistrib (W s) (id : ℝ × ℝ → ℝ × ℝ) P
      ((gaussianReal 0 1).prod (gaussianReal 0 1)) := fun s =>
    ⟨(hlaw s).aemeasurable, aemeasurable_id, by rw [(hlaw s).map_eq, Measure.map_id]⟩
  have hidWW : ∀ s, IdentDistrib (W s) (W 0) P P := fun s =>
    ⟨(hlaw s).aemeasurable, (hlaw 0).aemeasurable, by rw [(hlaw s).map_eq, (hlaw 0).map_eq]⟩
  have hidProd : ∀ s, IdentDistrib (fun ω => (W s ω).1 * (W s ω).2)
      (fun p : ℝ × ℝ => p.1 * p.2) P ((gaussianReal 0 1).prod (gaussianReal 0 1)) :=
    fun s => (hidW s).comp hmul
  have hfstmap : P.map (fun ω => (W 0 ω).1) = gaussianReal 0 1 := by
    rw [show (fun ω => (W 0 ω).1) = Prod.fst ∘ W 0 from rfl,
      ← Measure.map_map measurable_fst (hmeas 0), (hlaw 0).map_eq]
    exact Measure.fst_prod
  have hsndmap : P.map (fun ω => (W 0 ω).2) = gaussianReal 0 1 := by
    rw [show (fun ω => (W 0 ω).2) = Prod.snd ∘ W 0 from rfl,
      ← Measure.map_map measurable_snd (hmeas 0), (hlaw 0).map_eq]
    exact Measure.snd_prod
  have hidFst : IdentDistrib (fun ω => (W 0 ω).1) (id : ℝ → ℝ) P (gaussianReal 0 1) :=
    ⟨(hmeas 0).fst.aemeasurable, aemeasurable_id, by rw [hfstmap, Measure.map_id]⟩
  have hidSnd : IdentDistrib (fun ω => (W 0 ω).2) (id : ℝ → ℝ) P (gaussianReal 0 1) :=
    ⟨(hmeas 0).snd.aemeasurable, aemeasurable_id, by rw [hsndmap, Measure.map_id]⟩
  have hUint : Integrable (fun ω => (W 0 ω).1) P :=
    (hidFst.memLp_iff.2 (memLp_id_gaussianReal' 1 (by simp))).integrable le_rfl
  have hVint : Integrable (fun ω => (W 0 ω).2) P :=
    (hidSnd.memLp_iff.2 (memLp_id_gaussianReal' 1 (by simp))).integrable le_rfl
  have hUc : ∫ ω, (W 0 ω).1 ∂P = 0 := by rw [hidFst.integral_eq]; simp
  have hVc : ∫ ω, (W 0 ω).2 ∂P = 0 := by rw [hidSnd.integral_eq]; simp
  have hbase : IndepFun (fun ω => (W 0 ω).1) (fun ω => (W 0 ω).2) P := by
    rw [indepFun_iff_map_prod_eq_prod_map_map (hmeas 0).fst.aemeasurable
      (hmeas 0).snd.aemeasurable,
      show (fun ω => ((W 0 ω).1, (W 0 ω).2)) = W 0 from rfl, (hlaw 0).map_eq, hfstmap, hsndmap]
  have hpair : Pairwise fun a b => IndepFun
      (fun ω => ((fun _ : Fin 1 => (W a ω).1), (fun _ : Fin 1 => (W a ω).2)))
      (fun ω => ((fun _ : Fin 1 => (W b ω).1), (fun _ : Fin 1 => (W b ω).2))) P :=
    fun a b hab => (hindep.indepFun hab).comp hpairmap hpairmap
  have hident : ∀ s, IdentDistrib
      (fun ω => ((fun _ : Fin 1 => (W s ω).1), (fun _ : Fin 1 => (W s ω).2)))
      (fun ω => ((fun _ : Fin 1 => (W 0 ω).1), (fun _ : Fin 1 => (W 0 ω).2))) P P :=
    fun s => (hidWW s).comp hpairmap
  have hYint : Integrable (fun ω =>
      outerE (fun _ : Fin 1 => (W 0 ω).1) (fun _ : Fin 1 => (W 0 ω).2)) P :=
    integrable_outerE_of_coords fun _ => hbase.integrable_mul hUint hVint
  have hUvec : Integrable (fun ω => vecE (fun _ : Fin 1 => (W 0 ω).1)) P :=
    integrable_vecE_of_coords fun _ => hUint
  have hVvec : Integrable (fun ω => vecE (fun _ : Fin 1 => (W 0 ω).2)) P :=
    integrable_vecE_of_coords fun _ => hVint
  have hZ : IsSquareCentredIID P fun s ω => (W s ω).1 * (W s ω).2 := by
    refine ⟨fun s => ?_, fun i j hij => ?_, fun i => ?_, ?_⟩
    · exact (hidProd s).memLp_iff.2 memLp_mul_gaussianProd
    · exact (hindep.indepFun hij).comp hmul hmul
    · exact (hidProd i).trans (hidProd 0).symm
    · rw [(hidProd 0).integral_eq]; exact integral_mul_gaussianProd
  have hvar : Var[fun ω => (W 0 ω).1 * (W 0 ω).2; P] = 1 := by
    rw [(hidProd 0).variance_eq, variance_of_integral_eq_zero hmul.aemeasurable
      integral_mul_gaussianProd, integral_sq_mul_gaussianProd]
  have hval : ∀ T t : ℕ, ∫ ω, (crossCovEstimator (d := 1) T t
      (fun s _ => (W s ω).1) (fun s _ => (W s ω).2) 0 0) ^ 2 ∂P = 1 / T := by
    intro T t
    rw [integral_sq_crossCovEstimator_entry (d := 1) (μ := P) T t
      (fun s ω _ => (W s ω).1) (fun s ω _ => (W s ω).2) 0 0 hZ, hvar]
  refine ⟨Ω, mΩ, P, (fun s ω _ => (W s ω).1), (fun s ω _ => (W s ω).2), hprob,
    hbase.comp hconst hconst, fun _ => hUc, fun _ => hVc,
    isCentredIID_outerE _ _ hpair hident (hbase.comp hconst hconst)
      (fun _ => hUint.aestronglyMeasurable) (fun _ => hVint.aestronglyMeasurable)
      (fun _ => hUc) hYint,
    isCentredIID_vecE_fst _ (fun s ω _ => (W s ω).2) hpair hident hUvec (fun _ => hUc),
    isCentredIID_vecE_snd (fun s ω _ => (W s ω).1) _ hpair hident hVvec (fun _ => hVc),
    hZ, hval, ?_⟩
  intro T t hT hae
  have h0 : ∫ ω, (crossCovEstimator (d := 1) T t
      (fun s _ => (W s ω).1) (fun s _ => (W s ω).2) 0 0) ^ 2 ∂P = 0 := by
    refine integral_eq_zero_of_ae ?_
    filter_upwards [hae] with ω hω
    simp [hω]
  rw [hval T t] at h0
  exact one_div_ne_zero (Nat.cast_ne_zero.mpr hT) h0

end Witness

end IcnnLift
