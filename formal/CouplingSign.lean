import Mathlib
import TraceLemmas

/-!
# The sign of the coupling channel

This file settles, by machine-checked proof, a tension inside the authors' own design note
`docs/design/lemma1_rederivation.md`, and it is the tension on which the corrected mechanism
turns.

Section 2 of that note derives the update-covariance decomposition that replaces the refuted
Hessian decomposition of the shipped Lemma 1. Writing `δθ` for the batch-resampling fluctuation
of the latent iterate, `δg` for the gradient fluctuation, `H̄` for the mean Hessian,
`V = E[δθ δθᵀ]` and `Σ_slack = E[δθ δgᵀ]`, the projection of the update covariance on a
direction `e` carries the **coupling term**

  `eᵀ C e`, with `C = H̄ Σ_slack + Σ_slackᵀ H̄`,

alongside the attenuated objective noise and the magnitude term `eᵀ H̄ V H̄ e`. Section 2 closes
with the claim that under assumption (A4)'s forward-KL chain `δg = H̄ δθ + r` the coupling term
"is `2 eᵀ H̄ V H̄ e ≥ 0` automatically: THE SIGN COMES FOR FREE, and (A4) reduces to the
remainder-control clause". Section 11.3 (FW-2') then contradicts this: "THE SIGN IS NOT
AUTOMATIC, and this is the honest core of the mechanism. `H̄ V H̄` is PSD always; `C` is
symmetric but NOT PSD in general", and it introduces the positive alignment of `C` on the
shoulder band as a **labelled assumption (A6)** rather than as a derived fact.

Section 11.3 is right and section 2 overstates the case, and the overstatement is locatable: it
is the silent deletion of the remainder `r`. Section 2's sentence is true of the *exact* chain
`δg = H̄ δθ`, where `crossTerm_nonneg_of_exact_chain` proves it, and false of the chain
`δg = H̄ δθ + r` that (A4) actually asserts, where `crossTerm_eq_add_remainder` shows that the
cross term acquires the further summand `2 eᵀ H̄ E[δθ rᵀ] e`, of no fixed sign. The honest
quantitative replacement is `crossTerm_ge_leading_sub_of_remainder_bound`, namely
`eᵀ C e ≥ 2 eᵀ H̄ V H̄ e - 2 ‖H̄ e‖ ‖e‖ δ` whenever `E[‖δθ‖ ‖r‖] ≤ δ`. The remainder is never
assumed to vanish, and `outerMoment_chain` derives the matrix identity
`Σ_slack = V H̄ + E[δθ rᵀ]` from the pointwise chain rather than asserting it.

That much is only half the story, and the other half is what makes (A6) load-bearing rather
than decorative. Section 8's FW-1 does **not** leave the remainder uncontrolled: it carries the
explicit clause

  `‖E[δθ rᵀ]‖_op ≤ (3/2) λ_min(H̄ V H̄) / ‖H̄‖_op`                                    (A4'-q)

and FW-2' of section 11.3 assumes (A6) *together with* (A4'-q). A counterexample that violates
(A4'-q) therefore does not show that (A6) is needed; it shows only that some remainder control
is needed. This file measures the gap exactly. Writing `RemainderControlledBy H V R ρ` for the
consequence of (A4'-q) that FW-1's own proof uses — `2 |xᵀ H R x| ≤ ρ xᵀ H V H x` for every `x`,
which the displayed operator-norm clause supplies with `ρ = 3`, through
`remainderControlledBy_of_bounds` from numerical bounds on `‖H̄‖_op ‖E[δθ rᵀ]‖_op` and
`λ_min(H̄ V H̄)`, and through `remainderControlledBy_of_opNormClause` from the clause itself —
three theorems locate the boundary:

* `totalExcess_nonneg_of_remainderControlled`: at `ρ ≤ 3`, which is what (A4'-q) buys, the
  *total* excess `C + H̄ V H̄` of (FW-1b) is non-negative: (FW-1b)'s PSD claim is correct as
  stated. Its closing "in particular `E ⪰ s² H̄ V H̄`" refinement is not; see below.
* `crossTerm_nonneg_of_remainderControlled`: the *coupling term alone*, which is what section
  11.3 needs because section 10 reports the magnitude term to be empirically inert, is
  non-negative from `ρ ≤ 2` downwards.
* `remainderControl_threshold_sharp`: the constant `2` cannot be raised. For every `ρ > 2` there
  is a realizable configuration controlled at level `ρ` whose coupling term is strictly
  negative; `crossTerm_ge_neg_leading_of_remainderControlled` gives the matching lower bound
  `-eᵀ H̄ V H̄ e` at `ρ ≤ 3`.

The window `2 < ρ ≤ 3` is therefore exactly where (A6) does its work, and it is non-empty.
`coupling_sign_not_automatic_under_remainder_control` exhibits a completely explicit four-point
sample space in dimension two, with `H̄` the identity (hence symmetric and positive definite)
and `V` positive definite, whose nonzero remainder satisfies not merely the quadratic-form
surrogate but the displayed operator-norm clause of (A4'-q) itself. On this witness `E[δθ rᵀ]`,
`H̄` and `H̄ V H̄` all act as scalar multiples of the identity, so the three quantities the
clause names can be pinned by equations rather than bounded, and the witness theorem carries
them in that form inside its own conclusion. `RemainderControlOpNorm` asks for
`vecNorm (H̄ *ᵥ x) = 1 · vecNorm x` and `vecNorm (E[δθ rᵀ] *ᵥ x) = (5/8) · vecNorm x` for
*every* `x`, and `xᵀ H̄ V H̄ x = (1/2) ‖x‖²` for every `x`, which is what `‖H̄‖_op = 1`,
`‖E[δθ rᵀ]‖_op = 5/8` and `λ_min(H̄ V H̄) = 1/2` mean for matrices that act as dilations; the
clause then reads `5/8 ≤ (3/2)·(1/2)/1`. The equational characterizations are supplied by
`coll_R_mulVec` and `ceH_mulVec`, and `remainderControlledBy_of_opNormClause` derives the
quadratic-form surrogate at level `ρ = 3` from the clause, so the identification of the note's
assumption with `ρ = 3` is proved and not assumed. On that configuration the total excess of
(FW-1b) is non-negative in every direction — the conjunct is quantified over all `x`, not only
over the direction that witnesses the negativity — exactly as section 8 claims, and yet
`eᵀ C e < 0`. A corollary worth a sentence of its own: section 8's further clause "in particular
`E ⪰ s² H̄ V H̄`" is, for `s ≠ 0`, *equivalent* to positive semidefiniteness of the coupling
term (subtract `s² H̄ V H̄` from both sides), so the same witness refutes that clause under
(A4'-q); like section 2's sentence, it holds only from `ρ ≤ 2` downwards.
`coupling_sign_negative_with_arbitrarily_small_remainder` rules out the remaining escape, that
the failure is an artefact of a large remainder: for every `ε > 0` the same construction,
rescaled, has remainder budget `E[‖δθ‖ ‖r‖] = ε` and still a strictly negative coupling term.
Smallness of `r` alone never restores section 2's sentence; only the *direction* of `r` does,
through (A6) or through the uncorrelated-remainder hypothesis `E[δθ rᵀ] = 0` of
`crossTerm_nonneg_of_uncorrelated_remainder`, which `uncorrelated_remainder_satisfiable` shows
is satisfiable with a nowhere-vanishing `r`.

The file also retains the cruder counterexample `coupling_sign_not_automatic` — the same sample
space with a remainder orthogonal to the jitter — and is explicit about its limit:
`ce_not_remainderControlled` proves that that configuration does *not* satisfy (A4'-q), so on
its own it establishes only the weaker statement that the sign is not automatic without
remainder control. It is retained because it is the configuration on which `tr Σ_slack > 0`
while `C` has a negative eigendirection (`counterexample_trace_slack_pos`), which
machine-checks the caveat of the note's section 11.4: the measured positivity of the trace
reported there ("`tr Σ_slack` is strictly positive at 8997 of 9000 iterations") is genuinely
compatible with a negative eigendirection of `C`, exactly as the note warns.

Assumption (A6) is stated as the Lean definition `PositiveAlignmentOn` and used as a hypothesis.
It is not proved, because the counterexamples show it is not provable — and not merely for the
uncontrolled remainder: `positiveAlignment_fails_under_remainder_control` proves that (A6) fails
for every `κ` on the configuration that satisfies (A4'-q). It is also not vacuous.
`positiveAlignment_satisfiable` exhibits a realizable family with a nonzero remainder at every
point, indexed by a band with a continuum of reference iterates, on which the alignment holds
with the *single* constant `κ = 5/4` uniformly over the band, which is the form (A6) is used in;
a witness on a one-point band would not have exhibited that uniformity. Correspondingly,
`projDiffusion_strict_increase_nonvacuous` discharges every hypothesis of
`projDiffusion_strict_increase_on_band` simultaneously on that band, with a third-order term
that is nonzero and carries the adversarial sign, `T3(e) = -‖e‖²/2` controlled by
`τ = 1/2 < s² κ`, and reads off the strict inequality at every iterate of the band. The nonzero
`T3` matters: a witness taken with `τ = 0` would have forced `T3 = 0` and so would have left the
one hypothesis that carries the discretization error untested.

What this file does **not** do. It is pure finite-dimensional linear algebra together with
finite weighted expectations; no stochastic analysis enters, and none is available in this
mathlib. In particular the passage from the discrete SGD recursion to the diffusion coefficient,
the third-and-higher order term `T3` of (FW-1a), and the band geometry of (A2) are not derived
here: `T3` enters `projDiffusionLift` as an explicit real number and is controlled by the
explicit hypothesis `hT3`, and the band enters `PositiveAlignmentOn` as an abstract set.
The general operator norm and least eigenvalue of a matrix are not developed here either. What
the theorems consume is the quadratic-form consequence `RemainderControlledBy` that FW-1's own
proof uses; two bridges connect it to the note's operator-norm clause, and neither is an
assumption. `remainderControlledBy_of_bounds` derives it from numerical bounds `a` and `b`
playing the roles of `‖H̄‖_op ‖E[δθ rᵀ]‖_op` and `λ_min(H̄ V H̄)`; and `RemainderControlOpNorm`
states the clause itself for matrices that act as dilations, with the three quantities pinned by
equational characterizations, from which `remainderControlledBy_of_opNormClause` derives the
surrogate at level `ρ = 3`. The witnesses are of exactly that dilation form, so nothing about
them rests on an unformalized identification.

## Results
* `couplingTerm`, `crossTerm` — the coupling term `C = H Σ + Σᵀ H` of (FW-2') and its quadratic
  form, the "cross term" of section 2's covariance decomposition.
* `crossTerm_eq_two_mul` — `eᵀ C e = 2 eᵀ H Σ e` for symmetric `H`, with no other hypothesis.
* `crossTerm_eq_two_mul_conj_of_exact_chain` — under the exact chain `Σ = V H`, the cross term
  is `2 eᵀ H V H e`.
* `crossTerm_nonneg_of_exact_chain` — hence it is non-negative: section 2's claim, true here.
* `sigmaJacSq_nonneg_of_exact_chain` — the trace form `σ_Jac² = tr Σ_slack = tr (V H) ≥ 0`.
* `trace_couplingTerm_nonneg_of_exact_chain` — `tr C ≥ 0` for symmetric `H`, `V` PSD.
* `crossTerm_eq_add_remainder` — under `Σ = V H + R` the cross term is `2 eᵀ H V H e` **plus**
  the explicit remainder term `2 eᵀ H R e`.
* `crossTerm_ge_sub_abs_remainder` — the same identity as a lower bound, with the remainder
  functional in absolute value.
* `crossTerm_frozen_eq_slack_sub`, `crossTerm_frozen_neg_of_slack_neg`,
  `crossTerm_frozen_le_slack` (added 2026-09-13) — the subtraction the appendix's (A3)
  discussion makes in words: at a vanishing chain remainder the coupling form of the **frozen**
  coupling `Σ_ξ` is that of the full slack cross-covariance less `2 eᵀ H V H e`, so a direction
  in which the slack channel's coupling form is negative is one in which the frozen coupling's
  is. This is what carries this file's counterexamples from `Σ_slack` to `Σ_ξ`, and it
  discharges the appendix sentence that said the subtraction was outside the formalization.
* `outerMoment`, `outerMoment_self_posSemidef`, `outerMoment_chain`, `outerMoment_smul_smul` —
  finite weighted second moments, positive semidefiniteness of `V = E[δθ δθᵀ]`, the derivation
  of `Σ_slack = V H + E[δθ rᵀ]` from the pointwise chain `δg = H δθ + r`, and their behaviour
  under rescaling.
* `outerMoment_prod_eq_zero_of_mean_zero`, `crossTerm_eq_zero_of_independent`,
  `crossTerm_eq_zero_of_slack_eq_zero`, `crossTerm_eq_zero_of_independent_satisfiable` — the
  coupling channel is empty when the latent-weight fluctuation is independent of the batch driving
  the gradient **and has zero mean**, the second clause being automatic for the paper's
  `δθ = h(X) - E[h]` but a hypothesis here; derived on a product sample space rather than
  asserted, and exhibited on a configuration where neither fluctuation vanishes. This is
  section 11.3's isotropic-injected-noise case.
* `crossTerm_nonneg_of_uncorrelated_remainder`, `uncorrelated_remainder_satisfiable` — a clean
  extra hypothesis under which section 2's sentence survives a remainder, `E[δθ rᵀ] = 0`, and a
  configuration with a nowhere-vanishing remainder that satisfies it.
* `crossTerm_ge_leading_sub_of_remainder_bound` — the quantitative corollary: if
  `E[‖δθ‖ ‖r‖] ≤ δ` then `eᵀ C e ≥ 2 eᵀ H V H e - 2 ‖H e‖ ‖e‖ δ`.
* `RemainderControlledBy`, `remainderControlledBy_of_bounds` — assumption (A4'-q) of section 8,
  in the quadratic-form shape its own proof uses, and the bridge from numerical bounds on
  `‖H̄‖_op ‖E[δθ rᵀ]‖_op` and `λ_min(H̄ V H̄)`.
* `RemainderControlOpNorm`, `remainderControlledBy_of_opNormClause` — the displayed
  operator-norm clause of (A4'-q) itself, for matrices acting as dilations, with `‖H̄‖_op`,
  `‖E[δθ rᵀ]‖_op` and `λ_min(H̄ V H̄)` pinned by equational characterizations; and the proof that
  it implies the quadratic-form surrogate at level `ρ = 3`, so that identification is derived.
* `ceD_mulVec`, `ceH_mulVec`, `coll_R_mulVec`, `coll_remainderControlOpNorm` — the dilation
  characterizations that make the operator-norm clause checkable on the explicit witnesses.
* `totalExcess_nonneg_of_remainderControlled` — at (A4'-q)'s level `ρ ≤ 3` the total excess
  `C + H V H` of (FW-1b) is non-negative: section 8 is right.
* `crossTerm_nonneg_of_remainderControlled` — the coupling term alone is non-negative at
  `ρ ≤ 2`, and `crossTerm_ge_neg_leading_of_remainderControlled` is all that survives at `ρ ≤ 3`.
* `remainderControl_threshold_sharp` — the threshold `ρ ≤ 2` cannot be raised.
* `coupling_sign_not_automatic` — a concrete realizable configuration with `H` symmetric
  positive definite, `V` positive definite and a nonzero remainder on which `eᵀ C e < 0`;
  `ce_not_remainderControlled` records that it does not satisfy (A4'-q).
* `coupling_sign_not_automatic_under_remainder_control` — the same conclusion for a
  configuration that **does** satisfy (A4'-q) — the operator-norm clause itself, carried in the
  conclusion through the equational characterizations that `coll_R_mulVec` and `ceH_mulVec`
  supply — with the total excess of (FW-1b) non-negative in every direction; this is what makes
  (A6) necessary.
* `coupling_sign_negative_with_arbitrarily_small_remainder` — the same, with the operator-norm
  clause again verified and with remainder budget equal to any prescribed `ε`, so no smallness
  assumption on `r` can replace (A6).
* `counterexample_trace_slack_pos` — on the first configuration `tr Σ_slack > 0`.
* `ce_remainder_bound`, `ce_bound_consistent`, `ce_bound_value` — that configuration's remainder
  budget is `E[‖δθ‖ ‖r‖] = 2`, the general quantitative bound applies to it verbatim, and the
  guaranteed lower bound it delivers there is `-6` against a true cross term of `-2`.
* `PositiveAlignmentOn` — assumption (A6) of section 11.3, as a Lean definition.
* `projDiffusion_ge_of_positiveAlignment`, `projDiffusionDirect_lt_projDiffusionLift`,
  `projDiffusion_strict_increase_on_band` — under (A6) and an explicit bound on the third-order
  term, the coupling channel increases the projected diffusion, with the explicit margin
  `(s² κ - τ) ‖e‖²`, strictly in every nonzero direction and at every reference iterate on the
  band.
* `positiveAlignment_satisfiable`, `projDiffusion_strict_increase_nonvacuous` — (A6) holds with
  a single `κ` uniformly over a band with a continuum of reference iterates, on a realizable
  family with a nonzero remainder at every point; and every hypothesis of the band theorem is
  jointly satisfiable there together with a nonzero, adversarially signed third-order term, so
  the conditional results are not vacuous and do not secretly require `T3 = 0`.
* `couplingTerm_isHermitian` and `ce_couplingTerm_not_posSemidef` — the two halves of section
  11.3's sentence: `C` is symmetric, and `C` is not positive semidefinite in general.
* `positiveAlignment_fails_of_crossTerm_neg`, `positiveAlignment_fails_counterexample`,
  `positiveAlignment_fails_under_remainder_control` — (A6) fails for every `κ`, both on the
  first counterexample and on the configuration that satisfies (A4'-q).

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses carried, and what each would cost to discharge:

* `hchain : ∀ ω, δg ω = H *ᵥ δθ ω + r ω` is assumption (A4)'s forward-KL chain, taken as a
  definition of the remainder `r` rather than as an approximation; nothing is assumed about `r`
  except where a bound is named. Discharging it means deriving the chain from the forward-KL
  objective, which is a modelling statement about the loss and not a theorem of linear algebra.
* `hδ : ∑ ω, w ω * (‖δθ ω‖ ‖r ω‖) ≤ δ` is (A4)'s remainder-control clause, in the weakest form
  that makes the conclusion quantitative. It is a hypothesis because the size of `r` depends on
  third derivatives of the loss, which are outside this development.
* `hw : ∀ ω, 0 ≤ w ω` is non-negativity of the sample weights. It is what makes `outerMoment` an
  expectation rather than a signed sum, and it is needed for `V = E[δθ δθᵀ]` to be positive
  semidefinite and for the triangle inequality in the quantitative bound. It is not an
  assumption about the mathematics of the mechanism, and every explicit configuration in this
  file discharges it.
* `RemainderControlledBy H V R ρ` is the quadratic-form consequence of (A4'-q), and it is what
  the theorems consume. It is a hypothesis for the same reason as `hδ`. It is stated in this
  shape rather than in the note's operator-norm shape because the general `λ_min` and `‖·‖_op`
  for matrices are not developed here; two lemmas connect the shapes, and neither is an
  assumption. `remainderControlledBy_of_bounds` is the interface through which a numerical bound
  on `‖H̄‖_op ‖E[δθ rᵀ]‖_op` and `λ_min(H̄ V H̄)` would enter, and
  `remainderControlledBy_of_opNormClause` derives the level `ρ = 3` from `RemainderControlOpNorm`,
  which is the displayed clause itself; so the identification of the note's clause with `ρ = 3`
  is proved and not assumed. For the sharp witness nothing rests even on that: the witness's
  `E[δθ rᵀ]`, `H̄` and `H̄ V H̄` act as dilations, and
  `coupling_sign_not_automatic_under_remainder_control` carries the operator-norm clause, with
  the three quantities pinned by equational characterizations, as a conjunct of its own
  conclusion.
* `hT3 : |T3| ≤ τ * (e ⬝ᵥ e)` is the (FW-1a) third-and-higher-moment term, retained as an
  explicit real number in `projDiffusionLift` and bounded rather than discarded. Discharging it
  requires (A1)'s moment bounds and a Taylor remainder for the update map, neither of which is
  formalized here. Note that this is stronger than the design note, which absorbs `T3` into
  (A4'-q) without a bookkeeping of the margin; here the margin `(s² κ - τ) ‖e‖²` degrades
  continuously with `τ` and the conclusion holds only while `τ < s² κ`.
* `PositiveAlignmentOn` is (A6). It cannot be discharged: on the configuration of
  `coupling_sign_not_automatic_under_remainder_control`, which satisfies (A4'-q) itself,
  `positiveAlignment_fails_under_remainder_control` proves that it fails for every `κ`. It is
  not vacuous either: `positiveAlignment_satisfiable` gives a band with a continuum of iterates
  on which it holds with one `κ`, and `projDiffusion_strict_increase_nonvacuous` discharges the
  whole hypothesis set of the band theorem there, with a nonzero third-order term.

Finally, the objects `projDiffusionDirect` and `projDiffusionLift` are *definitions* of the
projected update covariance in terms of its section-2 decomposition, not derivations of that
decomposition from the SGD recursion. The derivation is section 2's and lives in
`UpdateCovariance.lean`; what is proved here is the ordering between the two given the
decomposition.

One finding for the author that goes beyond the assignment. Section 2's overstatement is not a
harmless simplification of a true-in-spirit claim, and it is not repaired by (A4'-q) either.
Section 2 writes the chain with the remainder and then reports a cross term computed without it;
the clause that would restore the sentence is not "`r` is small" but `E[δθ rᵀ] = 0`, or the
strictly stronger-than-(A4'-q) control `ρ ≤ 2`. If any version of the "sign comes for free"
sentence is retained in the tex, that is the clause it must carry. The same deletion surfaces
once more inside section 8 itself: (FW-1b)'s closing "in particular `E ⪰ s² H̄ V H̄`" is
equivalent, for `s ≠ 0`, to the positive semidefiniteness of the coupling term, and therefore
fails under (A4'-q) alone, on the same witness; only the PSD claim `E ⪰ 0` survives at `ρ = 3`.
-/

open Matrix Finset

namespace IcnnLift

variable {n : Type*} [Fintype n] {Ω : Type*} [Fintype Ω]

/-! ## Symmetry bookkeeping -/

omit [Fintype n] in
/-- Over `ℝ` a Hermitian matrix is exactly a symmetric one. -/
theorem transpose_eq_of_isHermitian {H : Matrix n n ℝ} (hH : H.IsHermitian) : Hᵀ = H := by
  ext i j
  have h := congrFun (congrFun hH.eq i) j
  simpa [Matrix.conjTranspose_apply] using h

/-- A quadratic form does not see transposition: `eᵀ Mᵀ e = eᵀ M e`. -/
theorem dotProduct_transpose_mulVec (M : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ (Mᵀ *ᵥ e) = e ⬝ᵥ (M *ᵥ e) := by
  rw [dotProduct_mulVec, vecMul_transpose, dotProduct_comm]

/-- The squared Euclidean length is non-negative. -/
theorem dotProduct_self_nonneg (x : n → ℝ) : 0 ≤ x ⬝ᵥ x := by
  rw [dotProduct]
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg _

/-- The squared Euclidean length of a nonzero vector is strictly positive. -/
theorem dotProduct_self_pos {x : n → ℝ} (hx : x ≠ 0) : 0 < x ⬝ᵥ x := by
  refine lt_of_le_of_ne (dotProduct_self_nonneg x) ?_
  intro hc
  exact hx (dotProduct_self_eq_zero.mp hc.symm)

/-! ## The coupling term and its quadratic form -/

/-- The **coupling term** `C = H Σ + Σᵀ H` of (FW-2'), section 11.3 of the design note: the
part of the update-covariance excess that is carried by the slack-channel cross-covariance
`Σ = Σ_slack = E[δθ δgᵀ]` rather than by the magnitude of the jitter. It is symmetric by
construction; it is *not* positive semidefinite in general, which is the point of this file. -/
def couplingTerm (H S : Matrix n n ℝ) : Matrix n n ℝ := H * S + Sᵀ * H

/-- The **cross term** of section 2's covariance decomposition: the quadratic form of the
coupling term in the projection direction `e` (for the paper, the bias-channel direction
`e_b`). -/
def crossTerm (H S : Matrix n n ℝ) (e : n → ℝ) : ℝ := e ⬝ᵥ (couplingTerm H S *ᵥ e)

/-- The coupling term really is symmetric. -/
theorem couplingTerm_isHermitian {H S : Matrix n n ℝ} (hH : H.IsHermitian) :
    (couplingTerm H S).IsHermitian := by
  have hHt : Hᵀ = H := transpose_eq_of_isHermitian hH
  have : (couplingTerm H S)ᵀ = couplingTerm H S := by
    simp only [couplingTerm, Matrix.transpose_add, Matrix.transpose_mul, Matrix.transpose_transpose,
      hHt]
    exact add_comm _ _
  ext i j
  have h := congrFun (congrFun this i) j
  simpa [Matrix.conjTranspose_apply] using h

/-- For symmetric `H` the cross term is exactly twice `eᵀ H Σ e`. It holds for every `Σ`
whatsoever: no positivity, and no chain relating `Σ` to `H`, is used. -/
theorem crossTerm_eq_two_mul {H : Matrix n n ℝ} (hH : H.IsHermitian) (S : Matrix n n ℝ)
    (e : n → ℝ) : crossTerm H S e = 2 * (e ⬝ᵥ ((H * S) *ᵥ e)) := by
  have hHt : Hᵀ = H := transpose_eq_of_isHermitian hH
  have hT : Sᵀ * H = (H * S)ᵀ := by rw [Matrix.transpose_mul, hHt]
  rw [crossTerm, couplingTerm, hT, Matrix.add_mulVec, dotProduct_add,
    dotProduct_transpose_mulVec]
  ring

/-- A configuration whose slack-channel cross-covariance vanishes gets nothing at all from the
coupling channel. This is the `C = 0` construction of section 11.3. The step from "isotropic
injected noise" to `Σ_slack = 0` is not asserted here but derived, on a product sample space,
in `crossTerm_eq_zero_of_independent` below. -/
theorem crossTerm_eq_zero_of_slack_eq_zero (H : Matrix n n ℝ) (e : n → ℝ) :
    crossTerm H 0 e = 0 := by
  simp [crossTerm, couplingTerm]

/-! ## The clean case: the exact chain `δg = H δθ`

Under the exact chain the slack cross-covariance is `Σ_slack = V H`, and section 2's sentence
is correct: the sign of the coupling term does come for free. -/

/-- Under the **exact** chain `δg = H δθ`, so that `Σ_slack = V H`, the cross term is
`2 eᵀ H V H e`. -/
theorem crossTerm_eq_two_mul_conj_of_exact_chain {H V S : Matrix n n ℝ} (hH : H.IsHermitian)
    (hS : S = V * H) (e : n → ℝ) :
    crossTerm H S e = 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)) := by
  rw [crossTerm_eq_two_mul hH, hS, Matrix.mul_assoc]

/-- Section 2's claim, and the exact hypothesis under which it is true: with the **exact** chain
`δg = H δθ`, symmetric `H` and positive semidefinite `V = E[δθ δθᵀ]`, the coupling channel adds
diffusion rather than removing it. The sign is supplied by `IcnnLift.quadForm_conj_nonneg`. -/
theorem crossTerm_nonneg_of_exact_chain {H V S : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (hS : S = V * H) (e : n → ℝ) : 0 ≤ crossTerm H S e := by
  rw [crossTerm_eq_two_mul_conj_of_exact_chain hH hS]
  have := quadForm_conj_nonneg hV hH e
  linarith

/-- The trace form of the same fact: under the exact chain the scalar `σ_Jac²` of section 2,
which is the trace of the slack-channel cross-covariance, equals `tr (V H)` and is
non-negative whenever both the jitter covariance and the mean Hessian are positive
semidefinite. This is `IcnnLift.trace_mul_nonneg_of_posSemidef`. -/
theorem sigmaJacSq_nonneg_of_exact_chain [DecidableEq n] {H V S : Matrix n n ℝ}
    (hV : V.PosSemidef) (hH : H.PosSemidef) (hS : S = V * H) : 0 ≤ S.trace := by
  rw [hS]
  exact trace_mul_nonneg_of_posSemidef hV hH

/-- The trace of the coupling term is non-negative under the exact chain, and this needs only
symmetry of `H`, not its positivity: `tr C = 2 tr (V H²)` and `H² = Hᴴ H` is positive
semidefinite. -/
theorem trace_couplingTerm_nonneg_of_exact_chain [DecidableEq n] {H V S : Matrix n n ℝ}
    (hV : V.PosSemidef) (hH : H.IsHermitian) (hS : S = V * H) :
    0 ≤ (couplingTerm H S).trace := by
  have hHt : Hᵀ = H := transpose_eq_of_isHermitian hH
  have hsq : (H * H).PosSemidef := by
    have := Matrix.posSemidef_conjTranspose_mul_self H
    rwa [hH.eq] at this
  have hstep : (couplingTerm H S).trace = 2 * (V * (H * H)).trace := by
    rw [couplingTerm, Matrix.trace_add, hS]
    have h1 : (Sᵀ * H).trace = (H * S).trace := by
      rw [← Matrix.trace_transpose (Sᵀ * H), Matrix.transpose_mul, Matrix.transpose_transpose, hHt]
    rw [hS] at h1
    rw [h1, Matrix.trace_mul_comm H (V * H), Matrix.mul_assoc]
    ring
  rw [hstep]
  have := trace_mul_nonneg_of_posSemidef hV hsq
  linarith

/-! ## The honest case: the chain with a remainder

Assumption (A4) asserts `δg = H δθ + r`, not `δg = H δθ`. The remainder contributes to the
cross term, and its contribution has no sign. -/

/-- Under the chain `δg = H δθ + r`, whose matrix consequence is `Σ_slack = V H + R` with
`R = E[δθ rᵀ]`, the cross term is section 2's `2 eᵀ H V H e` **plus** an explicit remainder
term. Nothing here assumes `R = 0`. -/
theorem crossTerm_eq_add_remainder {H V R S : Matrix n n ℝ} (hH : H.IsHermitian)
    (hS : S = V * H + R) (e : n → ℝ) :
    crossTerm H S e = 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)) + 2 * (e ⬝ᵥ ((H * R) *ᵥ e)) := by
  rw [crossTerm_eq_two_mul hH, hS, Matrix.mul_add, Matrix.add_mulVec, dotProduct_add,
    Matrix.mul_assoc]
  ring

/-- The remainder form as a lower bound: the coupling term is at least the section-2 leading
term minus twice the absolute value of the remainder functional. The leading term is
non-negative, but the difference need not be. -/
theorem crossTerm_ge_sub_abs_remainder {H V R S : Matrix n n ℝ} (hH : H.IsHermitian)
    (hS : S = V * H + R) (e : n → ℝ) :
    2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)) - 2 * |e ⬝ᵥ ((H * R) *ᵥ e)|
      ≤ crossTerm H S e := by
  rw [crossTerm_eq_add_remainder hH hS]
  have := neg_abs_le (e ⬝ᵥ ((H * R) *ᵥ e))
  linarith

/-! ## The subtraction that carries the sign from the slack channel to the frozen coupling

The appendix sentence this discharges, from the (A3) discussion of `app_proofs.tex`:

> "The coupling form of `Σ_ξ` is that of `Σ_slack` less the non-negative quantity
> `2 eᵀ H V H e`, by the split of `lem:diffusion`(i) at `r₂ = 0`, so it is negative in the same
> direction."

The split of `lem:diffusion`(i) is `Σ_slack = Σ_ξ + V H + Cov[δθ̃, r₂]`
(`MinibatchNoise.covMat_slack_refined`); at `r₂ = 0` it is `Σ_slack = Σ_ξ + V H`, which is the
hypothesis `hS` below. The subtraction is then exact, and the sign transfers because the
subtracted quantity is a quadratic form in a positive semidefinite matrix
(`IcnnLift.quadForm_conj_nonneg`). This is what lets the counterexamples of this file, which are
built on the full cross-covariance, be read as counterexamples on the frozen coupling. -/

/-- **The subtraction.** At a vanishing chain remainder, the coupling form of the frozen coupling
is the coupling form of the full slack cross-covariance less `2 eᵀ H V H e`. -/
theorem crossTerm_frozen_eq_slack_sub {H V Xi S : Matrix n n ℝ} (hH : H.IsHermitian)
    (hS : S = Xi + V * H) (e : n → ℝ) :
    crossTerm H Xi e = crossTerm H S e - 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)) := by
  have hS' : S = V * H + Xi := by rw [hS]; abel
  have h1 := crossTerm_eq_add_remainder (H := H) (V := V) (R := Xi) hH hS' e
  have h2 := crossTerm_eq_two_mul hH Xi e
  rw [h1, h2]
  ring

/-- **The sign conclusion.** A direction in which the full slack cross-covariance has a negative
coupling form is a direction in which the frozen coupling has one: the subtracted quantity is
non-negative, so the frozen coupling's form is the smaller of the two. This is the transfer the
appendix makes in words, and it is what makes `ce_crossTerm_neg` and
`counterexample_trace_slack_pos` statements about `Σ_ξ` as well as about `Σ_slack`. -/
theorem crossTerm_frozen_neg_of_slack_neg {H V Xi S : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (hS : S = Xi + V * H) {e : n → ℝ} (hneg : crossTerm H S e < 0) :
    crossTerm H Xi e < 0 := by
  have hq := quadForm_conj_nonneg hV hH e
  rw [crossTerm_frozen_eq_slack_sub hH hS e]
  linarith

/-- The same transfer as an inequality rather than a sign: the frozen coupling's form never
exceeds the full slack channel's. -/
theorem crossTerm_frozen_le_slack {H V Xi S : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (hS : S = Xi + V * H) (e : n → ℝ) :
    crossTerm H Xi e ≤ crossTerm H S e := by
  have hq := quadForm_conj_nonneg hV hH e
  rw [crossTerm_frozen_eq_slack_sub hH hS e]
  linarith

/-! ## Finite weighted expectations, and the quantitative remainder bound -/

/-- The Euclidean length of a vector, written through the dot product so that it composes with
`Matrix.mulVec` without a change of carrier type. -/
noncomputable def vecNorm (v : n → ℝ) : ℝ := Real.sqrt (v ⬝ᵥ v)

/-- The Euclidean length is non-negative. -/
theorem vecNorm_nonneg (v : n → ℝ) : 0 ≤ vecNorm v := Real.sqrt_nonneg _

/-- The Euclidean length as the root of a sum of squares, the form Cauchy–Schwarz wants. -/
theorem vecNorm_eq_sqrt_sum_sq (v : n → ℝ) : vecNorm v = Real.sqrt (∑ i, v i ^ 2) := by
  rw [vecNorm, dotProduct]
  congr 1
  exact Finset.sum_congr rfl fun i _ => by ring

/-- Cauchy–Schwarz for the dot product on `n → ℝ`. -/
theorem dotProduct_le_vecNorm_mul (u v : n → ℝ) : u ⬝ᵥ v ≤ vecNorm u * vecNorm v := by
  rw [vecNorm_eq_sqrt_sum_sq, vecNorm_eq_sqrt_sum_sq, dotProduct]
  exact Real.sum_mul_le_sqrt_mul_sqrt Finset.univ u v

/-- Cauchy–Schwarz with the absolute value on the left-hand side. -/
theorem abs_dotProduct_le_vecNorm_mul (u v : n → ℝ) : |u ⬝ᵥ v| ≤ vecNorm u * vecNorm v := by
  refine abs_le.mpr ⟨?_, dotProduct_le_vecNorm_mul u v⟩
  have h := dotProduct_le_vecNorm_mul (-u) v
  have hn : vecNorm (-u) = vecNorm u := by
    rw [vecNorm, vecNorm]
    congr 1
    simp [dotProduct]
  rw [hn, neg_dotProduct] at h
  linarith

/-- The Euclidean length is absolutely homogeneous. -/
theorem vecNorm_smul (a : ℝ) (v : n → ℝ) : vecNorm (a • v) = |a| * vecNorm v := by
  have h : (a • v) ⬝ᵥ (a • v) = a ^ 2 * (v ⬝ᵥ v) := by
    simp only [dotProduct, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [vecNorm, vecNorm, h, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq_eq_abs]

/-- The weighted second-moment matrix `E[u vᵀ] = ∑_ω w(ω) u(ω) v(ω)ᵀ` on a finite sample space.
Taking `w` to be the uniform weights `1/N` gives the empirical second moment of `CrossCov.lean`;
taking `w` to be an arbitrary probability vector gives a general finite expectation. -/
noncomputable def outerMoment (w : Ω → ℝ) (u v : Ω → n → ℝ) : Matrix n n ℝ :=
  ∑ ω, w ω • Matrix.vecMulVec (u ω) (v ω)

/-- `Matrix.vecMulVec_mulVec` with the scalar on the commutative side. -/
theorem vecMulVec_mulVec_real (u v w : n → ℝ) :
    Matrix.vecMulVec u v *ᵥ w = (v ⬝ᵥ w) • u := by
  ext i
  simp only [Matrix.mulVec, dotProduct, Matrix.vecMulVec_apply, Pi.smul_apply, smul_eq_mul,
    Finset.sum_mul]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- The quadratic form of a weighted second moment, in the form the estimates need. -/
theorem dotProduct_outerMoment_mulVec (w : Ω → ℝ) (u v : Ω → n → ℝ) (a b : n → ℝ) :
    a ⬝ᵥ (outerMoment w u v *ᵥ b) = ∑ ω, w ω * ((a ⬝ᵥ u ω) * (v ω ⬝ᵥ b)) := by
  rw [outerMoment, Matrix.sum_mulVec, dotProduct_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Matrix.smul_mulVec, vecMulVec_mulVec_real, dotProduct_smul, dotProduct_smul, smul_eq_mul,
    smul_eq_mul]
  ring

/-- `V = E[δθ δθᵀ]` is positive semidefinite for non-negative weights. -/
theorem outerMoment_self_posSemidef {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω) (u : Ω → n → ℝ) :
    (outerMoment w u u).PosSemidef := by
  refine Matrix.posSemidef_sum Finset.univ fun ω _ => Matrix.PosSemidef.smul ?_ (hw ω)
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ ?_
  · ext i j
    simp [Matrix.conjTranspose_apply, Matrix.vecMulVec_apply, mul_comm]
  · intro x
    have hstar : (star x : n → ℝ) = x := rfl
    rw [hstar, vecMulVec_mulVec_real, dotProduct_smul, smul_eq_mul, dotProduct_comm x (u ω)]
    exact mul_self_nonneg _

/-- The matrix consequence of assumption (A4)'s chain: if `δg = H δθ + r` pointwise on the
sample space and `H` is symmetric, then the slack-channel cross-covariance splits as
`Σ_slack = V H + E[δθ rᵀ]`. This is the substitution the design note performs in the proof of
(FW-1b), derived here rather than asserted. -/
theorem outerMoment_chain {w : Ω → ℝ} {dth dg r : Ω → n → ℝ} {H : Matrix n n ℝ}
    (hH : H.IsHermitian) (hchain : ∀ ω, dg ω = H *ᵥ dth ω + r ω) :
    outerMoment w dth dg = outerMoment w dth dth * H + outerMoment w dth r := by
  have hHt : Hᵀ = H := transpose_eq_of_isHermitian hH
  rw [outerMoment, outerMoment, outerMoment, Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun ω _ => ?_
  have hstep : Matrix.vecMulVec (dth ω) (dth ω) * H
      = Matrix.vecMulVec (dth ω) (H *ᵥ dth ω) := by
    rw [Matrix.vecMulVec_mul, ← hHt, vecMul_transpose, hHt]
  rw [hchain ω, Matrix.vecMulVec_add, smul_add, smul_mul_assoc, hstep]

/-- Pulling a matrix into a weighted second moment on the left. -/
theorem mul_outerMoment (H : Matrix n n ℝ) (w : Ω → ℝ) (u v : Ω → n → ℝ) :
    H * outerMoment w u v = outerMoment w (fun ω => H *ᵥ u ω) v := by
  rw [outerMoment, outerMoment, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Matrix.mul_smul, Matrix.mul_vecMulVec]

omit [Fintype n] in
/-- Rescaling both arguments of a weighted second moment rescales the moment. -/
theorem outerMoment_smul_smul (w : Ω → ℝ) (a b : ℝ) (u v : Ω → n → ℝ) :
    outerMoment w (fun ω => a • u ω) (fun ω => b • v ω) = (a * b) • outerMoment w u v := by
  rw [outerMoment, outerMoment, Finset.smul_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  ext i j
  simp [Matrix.vecMulVec_apply]
  ring

omit [Fintype n] in
/-- Independence kills the slack channel, derived rather than asserted. On a product sample
space with product weights, if the latent-weight fluctuation depends only on the first factor and has
zero weighted mean there, and the gradient fluctuation depends only on the second, then
`Σ_slack = E[δθ δgᵀ] = 0`. This is the matrix content of section 11.3's sentence that isotropic
injected noise, whose latent-weight fluctuation is independent of the batch that drives the gradient,
has `C = 0`. -/
theorem outerMoment_prod_eq_zero_of_mean_zero {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    (w₁ : Ω₁ → ℝ) (w₂ : Ω₂ → ℝ) (u : Ω₁ → n → ℝ) (v : Ω₂ → n → ℝ)
    (hu : ∀ i, ∑ a, w₁ a * u a i = 0) :
    outerMoment (fun p : Ω₁ × Ω₂ => w₁ p.1 * w₂ p.2) (fun p => u p.1) (fun p => v p.2) = 0 := by
  ext i j
  simp only [outerMoment, Matrix.sum_apply, Matrix.smul_apply, Matrix.vecMulVec_apply,
    smul_eq_mul, Matrix.zero_apply]
  rw [Fintype.sum_prod_type]
  have hstep : ∀ a : Ω₁, ∑ b : Ω₂, w₁ a * w₂ b * (u a i * v b j)
      = (w₁ a * u a i) * ∑ b : Ω₂, w₂ b * v b j := by
    intro a
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun b _ => by ring
  rw [Finset.sum_congr rfl fun a _ => hstep a, ← Finset.sum_mul, hu i, zero_mul]

/-- The coupling channel is empty under independence: the `C = 0` case of section 11.3, with the
hypotheses on the sample space rather than on the matrix. Two clauses are needed and both are
named: the product structure, which is independence, and the vanishing of the weighted mean of
the latent-weight fluctuation, which is automatic for the paper's centred `δθ = h(X) - E[h]` but is
not a consequence of independence and so is carried explicitly. -/
theorem crossTerm_eq_zero_of_independent {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    (w₁ : Ω₁ → ℝ) (w₂ : Ω₂ → ℝ) (u : Ω₁ → n → ℝ) (v : Ω₂ → n → ℝ)
    (hu : ∀ i, ∑ a, w₁ a * u a i = 0) (H : Matrix n n ℝ) (e : n → ℝ) :
    crossTerm H (outerMoment (fun p : Ω₁ × Ω₂ => w₁ p.1 * w₂ p.2)
      (fun p => u p.1) (fun p => v p.2)) e = 0 := by
  rw [outerMoment_prod_eq_zero_of_mean_zero w₁ w₂ u v hu, crossTerm_eq_zero_of_slack_eq_zero]

/-- **The quantitative corollary, and the honest replacement for section 2's "the sign comes for
free".** Under the chain `δg = H δθ + r` with `H` symmetric and non-negative weights, if the
remainder satisfies the (A4)-style control `E[‖δθ‖ ‖r‖] ≤ δ`, then the coupling term is at least
the section-2 leading term `2 eᵀ H V H e` minus `c δ`, with the constant written out,
`c = 2 ‖H e‖ ‖e‖`. The remainder is not assumed to vanish, and for `δ` large enough the bound is
vacuous — which is exactly what `coupling_sign_not_automatic` shows is unavoidable. -/
theorem crossTerm_ge_leading_sub_of_remainder_bound {w : Ω → ℝ} {dth dg r : Ω → n → ℝ}
    {H : Matrix n n ℝ} {delta : ℝ} (e : n → ℝ) (hw : ∀ ω, 0 ≤ w ω) (hH : H.IsHermitian)
    (hchain : ∀ ω, dg ω = H *ᵥ dth ω + r ω)
    (hdelta : ∑ ω, w ω * (vecNorm (dth ω) * vecNorm (r ω)) ≤ delta) :
    2 * (e ⬝ᵥ ((H * outerMoment w dth dth * H) *ᵥ e))
        - 2 * (vecNorm (H *ᵥ e) * vecNorm e) * delta
      ≤ crossTerm H (outerMoment w dth dg) e := by
  have hHt : Hᵀ = H := transpose_eq_of_isHermitian hH
  set V := outerMoment w dth dth with hV
  set R := outerMoment w dth r with hR
  -- the remainder functional, written out on the sample space
  have hrem : e ⬝ᵥ ((H * R) *ᵥ e)
      = ∑ ω, w ω * ((e ⬝ᵥ (H *ᵥ dth ω)) * (r ω ⬝ᵥ e)) := by
    rw [hR, mul_outerMoment, dotProduct_outerMoment_mulVec]
  -- each summand is dominated by the Cauchy–Schwarz product
  have hterm : ∀ ω : Ω, |w ω * ((e ⬝ᵥ (H *ᵥ dth ω)) * (r ω ⬝ᵥ e))|
      ≤ (vecNorm (H *ᵥ e) * vecNorm e) * (w ω * (vecNorm (dth ω) * vecNorm (r ω))) := by
    intro ω
    have hrw : e ⬝ᵥ (H *ᵥ dth ω) = (H *ᵥ e) ⬝ᵥ dth ω := by
      rw [dotProduct_mulVec, ← hHt, vecMul_transpose, hHt]
    have h1 : |e ⬝ᵥ (H *ᵥ dth ω)| ≤ vecNorm (H *ᵥ e) * vecNorm (dth ω) := by
      rw [hrw]; exact abs_dotProduct_le_vecNorm_mul _ _
    have h2 : |r ω ⬝ᵥ e| ≤ vecNorm (r ω) * vecNorm e :=
      abs_dotProduct_le_vecNorm_mul _ _
    have habs : |w ω * ((e ⬝ᵥ (H *ᵥ dth ω)) * (r ω ⬝ᵥ e))|
        = w ω * (|e ⬝ᵥ (H *ᵥ dth ω)| * |r ω ⬝ᵥ e|) := by
      rw [abs_mul, abs_mul, abs_of_nonneg (hw ω)]
    rw [habs]
    have hb1 : (0:ℝ) ≤ |e ⬝ᵥ (H *ᵥ dth ω)| := abs_nonneg _
    have hb2 : (0:ℝ) ≤ |r ω ⬝ᵥ e| := abs_nonneg _
    have hn1 : (0:ℝ) ≤ vecNorm (H *ᵥ e) * vecNorm (dth ω) :=
      mul_nonneg (vecNorm_nonneg _) (vecNorm_nonneg _)
    nlinarith [hw ω, mul_le_mul h1 h2 hb2 hn1]
  -- sum the pointwise bounds
  have hsum : |∑ ω, w ω * ((e ⬝ᵥ (H *ᵥ dth ω)) * (r ω ⬝ᵥ e))|
      ≤ (vecNorm (H *ᵥ e) * vecNorm e) * delta := by
    calc |∑ ω, w ω * ((e ⬝ᵥ (H *ᵥ dth ω)) * (r ω ⬝ᵥ e))|
        ≤ ∑ ω, |w ω * ((e ⬝ᵥ (H *ᵥ dth ω)) * (r ω ⬝ᵥ e))| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ ω, (vecNorm (H *ᵥ e) * vecNorm e) * (w ω * (vecNorm (dth ω) * vecNorm (r ω))) :=
          Finset.sum_le_sum fun ω _ => hterm ω
      _ = (vecNorm (H *ᵥ e) * vecNorm e)
            * ∑ ω, w ω * (vecNorm (dth ω) * vecNorm (r ω)) := by rw [Finset.mul_sum]
      _ ≤ (vecNorm (H *ᵥ e) * vecNorm e) * delta := by
          exact mul_le_mul_of_nonneg_left hdelta
            (mul_nonneg (vecNorm_nonneg _) (vecNorm_nonneg _))
  have hchainM : outerMoment w dth dg = V * H + R := outerMoment_chain hH hchain
  have hsplit := crossTerm_eq_add_remainder (V := V) (R := R) hH hchainM e
  rw [hsplit, hrem]
  linarith [(abs_le.mp hsum).1]

/-- A clean extra hypothesis under which section 2's sentence survives the chain **with** a
remainder: that the remainder be uncorrelated with the jitter, `E[δθ rᵀ] = 0`. Under it — which
assumption (A4) does not assert, and which the counterexamples below violate — the sign does
come for free, because the cross term reverts to `2 eᵀ H V H e`. It is sufficient but not
necessary; the sharp scalar condition is `RemainderControlledBy H V R 2` below. The hypothesis
is satisfiable with a nonzero remainder: see `uncorrelated_remainder_satisfiable`. -/
theorem crossTerm_nonneg_of_uncorrelated_remainder {w : Ω → ℝ} {dth dg r : Ω → n → ℝ}
    {H : Matrix n n ℝ} (hw : ∀ ω, 0 ≤ w ω) (hH : H.IsHermitian)
    (hchain : ∀ ω, dg ω = H *ᵥ dth ω + r ω) (hR : outerMoment w dth r = 0) (e : n → ℝ) :
    0 ≤ crossTerm H (outerMoment w dth dg) e := by
  have hS : outerMoment w dth dg = outerMoment w dth dth * H := by
    rw [outerMoment_chain hH hchain, hR, add_zero]
  exact crossTerm_nonneg_of_exact_chain (outerMoment_self_posSemidef hw dth) hH hS e

/-! ## What (A4'-q) buys, and where it stops

Section 8 (FW-1) of the design note does not leave the remainder uncontrolled. It carries the
explicit clause

  `‖E[δθ rᵀ]‖_op ≤ (3/2) λ_min(H̄ V H̄) / ‖H̄‖_op`                                    (A4'-q)

and section 11.3's FW-2' assumes (A6) *together with* (A4'-q). A counterexample violating
(A4'-q) therefore shows only that some remainder control is needed, not that (A6) is needed.
The definition below is the consequence of (A4'-q) that FW-1's own proof actually uses, at a
general level `ρ`; the displayed clause gives `ρ = 3`, via `remainderControlledBy_of_bounds`.
The three theorems that follow locate the gap exactly: `ρ ≤ 3` suffices for the *total* excess
of (FW-1b), `ρ ≤ 2` is needed and sufficient for the *coupling term alone*, and in the window
`2 < ρ ≤ 3` the coupling term can be strictly negative. That window is where (A6) does its
work. -/

/-- The consequence of assumption (A4'-q) that the proof of (FW-1b) uses: the remainder
functional of the coupling term is dominated, in the quadratic form sense and uniformly in the
direction, by `ρ` times the leading term. `R` is `E[δθ rᵀ]`, and the design note's own clause is
the case `ρ = 3`. -/
def RemainderControlledBy (H V R : Matrix n n ℝ) (rho : ℝ) : Prop :=
  ∀ x : n → ℝ, 2 * |x ⬝ᵥ ((H * R) *ᵥ x)| ≤ rho * (x ⬝ᵥ ((H * V * H) *ᵥ x))

/-- (A4'-q) in its operator-norm form implies the quadratic-form surrogate. Here `a` plays
`‖H̄‖_op ‖E[δθ rᵀ]‖_op` and `b` plays `λ_min(H̄ V H̄)`, so the design note's clause
`‖E[δθ rᵀ]‖_op ≤ (3/2) λ_min(H̄ V H̄) / ‖H̄‖_op` is exactly `2 a ≤ 3 b`, giving `ρ = 3`. The
operator norm and the least eigenvalue are not themselves formalized here; this lemma is the
interface through which any bound on them enters. -/
theorem remainderControlledBy_of_bounds {H V R : Matrix n n ℝ} {a b rho : ℝ}
    (ha : ∀ x : n → ℝ, |x ⬝ᵥ ((H * R) *ᵥ x)| ≤ a * (x ⬝ᵥ x))
    (hb : ∀ x : n → ℝ, b * (x ⬝ᵥ x) ≤ x ⬝ᵥ ((H * V * H) *ᵥ x))
    (hrho0 : 0 ≤ rho) (hab : 2 * a ≤ rho * b) :
    RemainderControlledBy H V R rho := by
  intro x
  have h1 := ha x
  have h2 := hb x
  have h3 := dotProduct_self_nonneg x
  nlinarith [mul_le_mul_of_nonneg_left h2 hrho0, mul_le_mul_of_nonneg_right hab h3]

/-- **Assumption (A4'-q) in the note's own operator-norm shape**, for the special case in which
the three matrices it mentions act as scalars. The general operator norm and least eigenvalue
are not developed in this file; instead the three quantities the clause names are *pinned by
equational characterizations*, which is exactly what an operator norm and a least eigenvalue
mean for a matrix that acts as a dilation: `nH` is `‖H̄‖_op` because `H̄` stretches every vector
by `nH`, `nR` is `‖E[δθ rᵀ]‖_op` for the same reason, and `lam` is `λ_min(H̄ V H̄)` because the
quadratic form of `H̄ V H̄` equals `lam ‖x‖²` in every direction. The final clause
`nR ≤ (3/2) lam / nH` is then the displayed (A4'-q) verbatim. -/
def RemainderControlOpNorm (H V R : Matrix n n ℝ) (nH nR lam : ℝ) : Prop :=
  (∀ x : n → ℝ, vecNorm (H *ᵥ x) = nH * vecNorm x) ∧
  (∀ x : n → ℝ, vecNorm (R *ᵥ x) = nR * vecNorm x) ∧
  (∀ x : n → ℝ, x ⬝ᵥ ((H * V * H) *ᵥ x) = lam * (x ⬝ᵥ x)) ∧
  0 < nH ∧ nR ≤ 3 / 2 * lam / nH

/-- The operator-norm form of (A4'-q) really does imply the quadratic-form surrogate at the
level `ρ = 3`, so the identification of the design note's clause with `ρ = 3` is derived here and
not assumed. Together with `remainderControlledBy_of_bounds` this closes the loop between the
note's shape of the assumption and the shape the theorems of this file consume. -/
theorem remainderControlledBy_of_opNormClause {H V R : Matrix n n ℝ} {nH nR lam : ℝ}
    (h : RemainderControlOpNorm H V R nH nR lam) : RemainderControlledBy H V R 3 := by
  obtain ⟨hHn, hRn, hlam, hnH, hclause⟩ := h
  intro x
  have hx : vecNorm x * vecNorm x = x ⬝ᵥ x := Real.mul_self_sqrt (dotProduct_self_nonneg x)
  have hstep : |x ⬝ᵥ ((H * R) *ᵥ x)| ≤ (nH * nR) * (x ⬝ᵥ x) := by
    have h1 : (H * R) *ᵥ x = H *ᵥ (R *ᵥ x) := (Matrix.mulVec_mulVec x H R).symm
    rw [h1]
    calc |x ⬝ᵥ (H *ᵥ (R *ᵥ x))| ≤ vecNorm x * vecNorm (H *ᵥ (R *ᵥ x)) :=
          abs_dotProduct_le_vecNorm_mul _ _
      _ = vecNorm x * (nH * (nR * vecNorm x)) := by rw [hHn, hRn]
      _ = (nH * nR) * (vecNorm x * vecNorm x) := by ring
      _ = (nH * nR) * (x ⬝ᵥ x) := by rw [hx]
  have hnum : nH * nR ≤ 3 / 2 * lam := by
    have h1 : nH * nR ≤ nH * (3 / 2 * lam / nH) :=
      mul_le_mul_of_nonneg_left hclause (le_of_lt hnH)
    have h2 : nH * (3 / 2 * lam / nH) = 3 / 2 * lam := by field_simp
    linarith [h1, h2.le, h2.ge]
  rw [hlam x]
  have hq := dotProduct_self_nonneg x
  nlinarith [mul_le_mul_of_nonneg_right hnum hq]

/-- **(FW-1b)'s PSD claim is correct as stated.** At the level `ρ ≤ 3` that assumption (A4'-q)
supplies, the *total* excess of (FW-1a) — the coupling term plus the magnitude term — is
non-negative in every direction. This is the design note's
`E = 3 H̄ V H̄ + H̄ E[δθ rᵀ] + E[r δθᵀ] H̄ ⪰ 0`, and it is why section 8's non-strict diffusion
ordering does not need (A6). Section 8's closing "in particular `E ⪰ s² H̄ V H̄`" is a strictly
stronger claim — for `s ≠ 0` it is equivalent, by subtracting `s² H̄ V H̄`, to positive
semidefiniteness of the coupling term itself — and it fails in the window `2 < ρ ≤ 3`: the
witness of `coupling_sign_not_automatic_under_remainder_control` satisfies (A4'-q)'s clause and
has `eᵀ C e < 0`. -/
theorem totalExcess_nonneg_of_remainderControlled {H V R S : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (hS : S = V * H + R) {rho : ℝ} (hrho : rho ≤ 3)
    (hctrl : RemainderControlledBy H V R rho) (e : n → ℝ) :
    0 ≤ crossTerm H S e + e ⬝ᵥ ((H * V * H) *ᵥ e) := by
  have hq : 0 ≤ e ⬝ᵥ ((H * V * H) *ᵥ e) := quadForm_conj_nonneg hV hH e
  have hc := hctrl e
  have habs := neg_abs_le (e ⬝ᵥ ((H * R) *ᵥ e))
  rw [crossTerm_eq_add_remainder hH hS]
  nlinarith [abs_nonneg (e ⬝ᵥ ((H * R) *ᵥ e))]

/-- **The threshold.** The coupling term *alone* — which is what section 11.3's FW-2' needs,
because section 10 reports the magnitude term to be empirically inert — is non-negative as soon
as the remainder is controlled at level `ρ ≤ 2`. The constant `2` cannot be raised:
`remainderControl_threshold_sharp` produces, for every `ρ > 2`, a realizable configuration
controlled at level `ρ` whose coupling term is strictly negative. Since (A4'-q) only supplies
`ρ = 3`, the assumption set of FW-1 does not by itself deliver the sign of `C`. -/
theorem crossTerm_nonneg_of_remainderControlled {H V R S : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (hS : S = V * H + R) {rho : ℝ} (hrho : rho ≤ 2)
    (hctrl : RemainderControlledBy H V R rho) (e : n → ℝ) :
    0 ≤ crossTerm H S e := by
  have hq : 0 ≤ e ⬝ᵥ ((H * V * H) *ᵥ e) := quadForm_conj_nonneg hV hH e
  have hc := hctrl e
  have habs := neg_abs_le (e ⬝ᵥ ((H * R) *ᵥ e))
  rw [crossTerm_eq_add_remainder hH hS]
  nlinarith [abs_nonneg (e ⬝ᵥ ((H * R) *ᵥ e))]

/-- What (A4'-q) does guarantee about the coupling term alone, and no more: at level `ρ ≤ 3` it
is bounded below by minus the leading term. The bound is attained in sign — the right-hand side
is genuinely negative — which is precisely the room in which assumption (A6) is doing work. -/
theorem crossTerm_ge_neg_leading_of_remainderControlled {H V R S : Matrix n n ℝ}
    (hV : V.PosSemidef) (hH : H.IsHermitian) (hS : S = V * H + R) {rho : ℝ} (hrho : rho ≤ 3)
    (hctrl : RemainderControlledBy H V R rho) (e : n → ℝ) :
    -(e ⬝ᵥ ((H * V * H) *ᵥ e)) ≤ crossTerm H S e := by
  have hq : 0 ≤ e ⬝ᵥ ((H * V * H) *ᵥ e) := quadForm_conj_nonneg hV hH e
  have hc := hctrl e
  have habs := neg_abs_le (e ⬝ᵥ ((H * R) *ᵥ e))
  rw [crossTerm_eq_add_remainder hH hS]
  nlinarith [abs_nonneg (e ⬝ᵥ ((H * R) *ᵥ e))]

/-! ## The first counterexample: the sign is not automatic

A fully explicit configuration in dimension two on a four-point sample space. The mean Hessian
is the identity — symmetric and positive definite, so no pathology there — the jitter covariance
`V` is positive definite, and the remainder is a genuine, bounded perturbation orthogonal to the
jitter. The cross term in the direction `e = (1,1)` is `-2`, so `C` is not positive
semidefinite: this is the machine-checked version of the second half of section 11.3's "`C` is
symmetric but NOT PSD in general".

The limit of this configuration is recorded, not hidden. Its remainder budget
`E[‖δθ‖ ‖r‖] = 2` equals the leading term `2 eᵀ H V H e = 2`, and `ce_not_remainderControlled`
proves that it does **not** satisfy assumption (A4'-q). It therefore establishes that the sign
is not automatic *without* remainder control, which is weaker than what section 11.3 needs,
since FW-2' assumes (A6) together with (A4'-q). The configuration that closes that gap is the
collinear family of the next section. What this one is uniquely good for is section 11.4's
caveat: it is the configuration on which `tr Σ_slack > 0` while `C` has a negative
eigendirection. -/

/-- Uniform weights on a four-point sample space. -/
noncomputable def ceWeight : Fin 4 → ℝ := fun _ => (1 : ℝ) / 4

/-- The latent-weight fluctuation `δθ` of the counterexample: `±e₁` and `±e₂`, so that the jitter
covariance is `(1/2) I`, positive definite. -/
def ceDtheta : Fin 4 → Fin 2 → ℝ := ![![1, 0], ![-1, 0], ![0, 1], ![0, -1]]

/-- The remainder `r` of the counterexample: supported on the two samples where `δθ = ±e₁`, and
pointing along `∓e₂`. It is anti-aligned with the curvature-weighted jitter in the direction
`(1,1)`. -/
def ceRem : Fin 4 → Fin 2 → ℝ := ![![0, -4], ![0, 4], ![0, 0], ![0, 0]]

/-- The mean Hessian of the counterexample: the identity, hence symmetric and positive
definite. -/
def ceH : Matrix (Fin 2) (Fin 2) ℝ := 1

/-- The gradient fluctuation of the counterexample, defined so that assumption (A4)'s chain
`δg = H δθ + r` holds exactly, by construction. -/
def ceDg : Fin 4 → Fin 2 → ℝ := fun ω => ceH *ᵥ ceDtheta ω + ceRem ω

/-- The projection direction of the counterexample. -/
def ceE : Fin 2 → ℝ := ![1, 1]

/-- The chain of assumption (A4) holds exactly on the counterexample. -/
theorem ce_chain (ω : Fin 4) : ceDg ω = ceH *ᵥ ceDtheta ω + ceRem ω := rfl

/-- The mean Hessian of the counterexample is symmetric. -/
theorem ce_isHermitian : ceH.IsHermitian := by
  simp [ceH]

/-- And it is positive definite, so the negativity below is not an artefact of a degenerate or
indefinite curvature either. -/
theorem ce_H_posDef : ceH.PosDef := by
  rw [ceH]
  exact Matrix.PosDef.one

/-- The jitter covariance of the counterexample is `(1/2) I`. -/
theorem ce_V_eq : outerMoment ceWeight ceDtheta ceDtheta = !![1/2, 0; 0, 1/2] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [outerMoment, ceWeight, ceDtheta, Fin.sum_univ_four, Matrix.sum_apply,
      Matrix.vecMulVec_apply] <;> norm_num

/-- The jitter covariance of the counterexample is positive **definite**, so the negativity
below is not an artefact of a degenerate `V`. -/
theorem ce_V_posDef : (outerMoment ceWeight ceDtheta ceDtheta).PosDef := by
  rw [ce_V_eq]
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [Matrix.conjTranspose_apply]
  · intro x hx
    have hstar : (star x : Fin 2 → ℝ) = x := rfl
    have hval : star x ⬝ᵥ ((!![1/2, 0; 0, 1/2] : Matrix (Fin 2) (Fin 2) ℝ) *ᵥ x)
        = (1/2) * (x 0 ^ 2 + x 1 ^ 2) := by
      rw [hstar]
      simp [dotProduct, mulVec, Fin.sum_univ_two]
      ring
    rw [hval]
    have hne : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
      by_contra hc
      have h0 : x 0 = 0 := by
        by_contra h
        exact hc (Or.inl h)
      have h1 : x 1 = 0 := by
        by_contra h
        exact hc (Or.inr h)
      refine hx (funext fun i => ?_)
      fin_cases i
      · simpa using h0
      · simpa using h1
    rcases hne with h | h
    · have : 0 < x 0 ^ 2 := by positivity
      nlinarith [sq_nonneg (x 1)]
    · have : 0 < x 1 ^ 2 := by positivity
      nlinarith [sq_nonneg (x 0)]

/-- The slack-channel cross-covariance of the counterexample. Its trace is `1 > 0`. -/
theorem ce_S_eq : outerMoment ceWeight ceDtheta ceDg = !![1/2, -2; 0, 1/2] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [outerMoment, ceWeight, ceDtheta, ceDg, ceRem, ceH, Fin.sum_univ_four, Matrix.sum_apply,
      Matrix.vecMulVec_apply] <;> norm_num

/-- **The sign is not automatic.** On the explicit configuration above — mean Hessian the
identity, jitter covariance positive definite, remainder bounded and anti-aligned — the coupling
term is strictly negative in the direction `(1,1)`. -/
theorem ce_crossTerm_neg :
    crossTerm ceH (outerMoment ceWeight ceDtheta ceDg) ceE = -2 := by
  rw [ce_S_eq, crossTerm, couplingTerm, ceH, ceE]
  simp only [Matrix.one_mul, Matrix.mul_one, dotProduct, mulVec, Fin.sum_univ_two,
    Matrix.add_apply, Matrix.transpose_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.empty_val', Matrix.cons_val_fin_one, Matrix.of_apply]
  norm_num

/-- Section 11.3's sentence, machine-checked in full: the coupling term `C` **is** symmetric
(`couplingTerm_isHermitian`) but is **not** positive semidefinite in general. -/
theorem ce_couplingTerm_not_posSemidef :
    ¬ (couplingTerm ceH (outerMoment ceWeight ceDtheta ceDg)).PosSemidef := by
  intro hpsd
  have h := (Matrix.posSemidef_iff_dotProduct_mulVec.mp hpsd).2 ceE
  rw [show (star ceE : Fin 2 → ℝ) = ceE from rfl] at h
  rw [show ceE ⬝ᵥ (couplingTerm ceH (outerMoment ceWeight ceDtheta ceDg) *ᵥ ceE)
      = crossTerm ceH (outerMoment ceWeight ceDtheta ceDg) ceE from rfl, ce_crossTerm_neg] at h
  norm_num at h

/-- The leading term that section 2 says carries the sign is strictly positive here, and is
nevertheless overwhelmed by the remainder: `2 eᵀ H V H e = 2` while the whole cross term is
`-2`. -/
theorem ce_leading_pos :
    2 * (ceE ⬝ᵥ ((ceH * outerMoment ceWeight ceDtheta ceDtheta * ceH) *ᵥ ceE)) = 2 := by
  rw [ce_V_eq, ceH, ceE]
  simp only [Matrix.one_mul, Matrix.mul_one, dotProduct, mulVec, Fin.sum_univ_two,
    Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val',
    Matrix.cons_val_fin_one, Matrix.of_apply]
  norm_num

/-- The trace of the slack-channel cross-covariance is strictly positive on the very
configuration whose coupling term has a negative eigendirection. This machine-checks the caveat
of section 11.4: the measured positivity of `tr Σ_slack` supports the scalar reduction of (A6)
but does not close its multi-dimensional form. -/
theorem counterexample_trace_slack_pos :
    0 < (outerMoment ceWeight ceDtheta ceDg).trace := by
  rw [ce_S_eq]
  simp [Matrix.trace, Matrix.diag, Fin.sum_univ_two]

/-- The remainder of this counterexample is bounded, with (A4)-style budget `E[‖δθ‖ ‖r‖] = 2`.
That budget is not small: it equals the leading term `2 eᵀ H V H e = 2`, and
`ce_not_remainderControlled` shows the configuration fails (A4'-q). The sharper statement — a
negative coupling term under (A4'-q), and with a budget below any prescribed `ε` — is
`coupling_sign_not_automatic_under_remainder_control` and
`coupling_sign_negative_with_arbitrarily_small_remainder` below. -/
theorem ce_remainder_bound :
    ∑ ω, ceWeight ω * (vecNorm (ceDtheta ω) * vecNorm (ceRem ω)) = 2 := by
  have h1 : Real.sqrt 1 = 1 := Real.sqrt_one
  have h16 : Real.sqrt 16 = 4 := by
    rw [show (16:ℝ) = 4 ^ 2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  have h0 : Real.sqrt 0 = 0 := Real.sqrt_zero
  simp only [Fin.sum_univ_four, ceWeight, ceDtheta, ceRem, vecNorm, dotProduct, Fin.sum_univ_two,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
    Matrix.tail_cons, Matrix.cons_val_three]
  norm_num [h1, h16, h0]

/-- The two halves of this file, checked against each other. The general quantitative bound
`crossTerm_ge_leading_sub_of_remainder_bound` applies verbatim to the counterexample, with the
remainder budget `δ = 2` supplied by `ce_remainder_bound`. -/
theorem ce_bound_consistent :
    2 * (ceE ⬝ᵥ ((ceH * outerMoment ceWeight ceDtheta ceDtheta * ceH) *ᵥ ceE))
        - 2 * (vecNorm (ceH *ᵥ ceE) * vecNorm ceE) * 2
      ≤ crossTerm ceH (outerMoment ceWeight ceDtheta ceDg) ceE :=
  crossTerm_ge_leading_sub_of_remainder_bound ceE (fun _ => by norm_num [ceWeight])
    ce_isHermitian ce_chain (le_of_eq ce_remainder_bound)

/-- And the value of that bound is `-6`, against a true cross term of `-2`: the guaranteed lower
bound of the quantitative corollary is genuinely negative here, as it must be. -/
theorem ce_bound_value :
    2 * (ceE ⬝ᵥ ((ceH * outerMoment ceWeight ceDtheta ceDtheta * ceH) *ᵥ ceE))
        - 2 * (vecNorm (ceH *ᵥ ceE) * vecNorm ceE) * 2 = -6 := by
  have hHe : ceH *ᵥ ceE = ceE := by
    rw [ceH]
    exact Matrix.one_mulVec ceE
  have hn : vecNorm ceE * vecNorm ceE = 2 := by
    rw [vecNorm]
    rw [Real.mul_self_sqrt (by rw [ceE, dotProduct, Fin.sum_univ_two]; norm_num)]
    rw [ceE, dotProduct, Fin.sum_univ_two]
    norm_num
  rw [hHe, hn, ce_leading_pos]
  norm_num

/-- **Section 11.3 is right and section 2 overstates the case.** There is a symmetric, positive
definite `H`, a positive definite `V`, a remainder `r` and a direction `e` for which the chain
`δg = H δθ + r` of assumption (A4) holds exactly on a finite sample space with non-negative
weights, and for which the coupling term is strictly negative. Hence the sign of the coupling
channel does *not* come for free from (A4)'s chain alone. This configuration does not satisfy
(A4'-q), so on its own it does not yet show that (A6) is needed on top of the remainder control
of FW-1; that is `coupling_sign_not_automatic_under_remainder_control`. -/
theorem coupling_sign_not_automatic :
    ∃ (w : Fin 4 → ℝ) (dth dg r : Fin 4 → Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ)
      (e : Fin 2 → ℝ),
      (∀ ω, 0 ≤ w ω) ∧ H.PosDef ∧ (outerMoment w dth dth).PosDef ∧
      (∀ ω, dg ω = H *ᵥ dth ω + r ω) ∧
      (∑ ω, w ω * (vecNorm (dth ω) * vecNorm (r ω)) = 2) ∧
      crossTerm H (outerMoment w dth dg) e < 0 := by
  refine ⟨ceWeight, ceDtheta, ceDg, ceRem, ceH, ceE, fun _ => by norm_num [ceWeight],
    ce_H_posDef, ce_V_posDef, ce_chain, ce_remainder_bound, ?_⟩
  rw [ce_crossTerm_neg]
  norm_num

/-! ## The sharp counterexample: a negative sign under (A4'-q), with an arbitrarily small
remainder

The configuration above is not enough to justify (A6), because FW-2' assumes (A6) *together
with* (A4'-q) and that configuration violates (A4'-q). This section closes the gap with a second
family on the same four-point sample space, in which the remainder is collinear with the jitter,
`r = c δθ`, and the whole configuration is scaled by `t`. Collinearity makes every quantity a
multiple of the same matrix, so the family realizes every control level `ρ = 2 |c|` exactly, and
the scaling makes the remainder budget `|c| t²` arbitrarily small without changing anything
else. The coupling term is `(1 + c) t² ‖e‖²`, so it is negative precisely when `c < -1`, that
is precisely when `ρ > 2` — which matches the abstract threshold of
`crossTerm_nonneg_of_remainderControlled` and shows that threshold to be sharp. -/

/-- The self-dot product of the counterexample's direction. -/
theorem ceE_dot : ceE ⬝ᵥ ceE = 2 := by
  simp [ceE, dotProduct, Fin.sum_univ_two]
  norm_num

/-- The counterexample's direction is nonzero, so the strictness clauses apply to it. -/
theorem ceE_ne_zero : ceE ≠ 0 := by
  intro hc
  have h0 : ceE 0 = 0 := by rw [hc]; rfl
  norm_num [ceE] at h0

/-- Each jitter sample of the four-point space is a unit vector. -/
theorem ce_vecNorm_dtheta (ω : Fin 4) : vecNorm (ceDtheta ω) = 1 := by
  fin_cases ω <;> simp [vecNorm, ceDtheta, dotProduct, Fin.sum_univ_two]

/-- The jitter covariance of the four-point sample space, `(1/2) I`, as a named matrix. -/
noncomputable def ceD : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, 0; 0, 1/2]

/-- The four-point jitter covariance, under its short name `ceD`. -/
theorem ce_V_eq_ceD : outerMoment ceWeight ceDtheta ceDtheta = ceD := ce_V_eq

/-- `ceD` is positive definite. -/
theorem ceD_posDef : ceD.PosDef := ce_V_eq_ceD ▸ ce_V_posDef

/-- `ceD` acts on every vector as the dilation by `1/2`; in particular its operator norm and its
least eigenvalue are both `1/2`, pinned without any general operator-norm development. -/
theorem ceD_mulVec (x : Fin 2 → ℝ) : ceD *ᵥ x = (1/2 : ℝ) • x := by
  ext i
  fin_cases i <;> simp [ceD, mulVec, dotProduct, Fin.sum_univ_two]

/-- The mean Hessian of these configurations acts as the identity, so `‖H̄‖_op = 1`. -/
theorem ceH_mulVec (x : Fin 2 → ℝ) : ceH *ᵥ x = x := by
  rw [ceH]; exact Matrix.one_mulVec x

/-- The quadratic form of a multiple of `ceD`. -/
theorem quadForm_ceD (c : ℝ) (e : Fin 2 → ℝ) : e ⬝ᵥ ((c • ceD) *ᵥ e) = (c / 2) * (e ⬝ᵥ e) := by
  simp [ceD, dotProduct, mulVec, Fin.sum_univ_two]
  ring

/-- The cross term of a multiple of `ceD` against the identity mean Hessian. -/
theorem crossTerm_ceH_smul_ceD (c : ℝ) (e : Fin 2 → ℝ) :
    crossTerm ceH (c • ceD) e = c * (e ⬝ᵥ e) := by
  rw [crossTerm_eq_two_mul ce_isHermitian, ceH, Matrix.one_mul, quadForm_ceD]
  ring

/-- The remainder cross-moment `E[δθ rᵀ]` of the first counterexample. -/
theorem ce_R_eq : outerMoment ceWeight ceDtheta ceRem = !![0, -2; 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [outerMoment, ceWeight, ceDtheta, ceRem, Fin.sum_univ_four, Matrix.sum_apply,
      Matrix.vecMulVec_apply, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons,
      Matrix.head_cons]

/-- The honest limitation of the first counterexample, machine-checked: it does **not** satisfy
assumption (A4'-q). In the direction `(1,1)` the remainder functional is `-2` against a leading
term of `1`, so the smallest admissible level there is `ρ = 4`, above the `ρ = 3` that (A4'-q)
supplies. A referee who reads section 11.3 alongside FW-1 is therefore right to say that the
first counterexample alone does not establish the need for (A6); the family below does. -/
theorem ce_not_remainderControlled :
    ¬ RemainderControlledBy ceH (outerMoment ceWeight ceDtheta ceDtheta)
        (outerMoment ceWeight ceDtheta ceRem) 3 := by
  intro h
  have hx := h ceE
  rw [ce_R_eq, ce_V_eq_ceD, ceH] at hx
  simp only [Matrix.one_mul, Matrix.mul_one] at hx
  have h1 : ceE ⬝ᵥ ((!![0, -2; 0, 0] : Matrix (Fin 2) (Fin 2) ℝ) *ᵥ ceE) = -2 := by
    simp [ceE, dotProduct, mulVec, Fin.sum_univ_two]
  have h2 : ceE ⬝ᵥ (ceD *ᵥ ceE) = 1 := by
    simp [ceD, ceE, dotProduct, mulVec, Fin.sum_univ_two]
    norm_num
  rw [h1, h2] at hx
  norm_num at hx

/-- The jitter of the collinear family: the four-point jitter scaled by `t`. -/
def collDtheta (t : ℝ) : Fin 4 → Fin 2 → ℝ := fun ω => t • ceDtheta ω

/-- The remainder of the collinear family: exactly `c` times the jitter. Negative `c` means the
remainder opposes the curvature-weighted jitter, which is the anti-alignment section 11.3 warns
about. -/
def collRem (c t : ℝ) : Fin 4 → Fin 2 → ℝ := fun ω => (c * t) • ceDtheta ω

/-- The gradient fluctuation of the collinear family, defined so that assumption (A4)'s chain
holds exactly, by construction. -/
def collDg (c t : ℝ) : Fin 4 → Fin 2 → ℝ := fun ω => ceH *ᵥ collDtheta t ω + collRem c t ω

/-- Unfolding lemma for the collinear jitter. -/
theorem collDtheta_eq (t : ℝ) : collDtheta t = fun ω => t • ceDtheta ω := rfl

/-- Unfolding lemma for the collinear remainder. -/
theorem collRem_eq (c t : ℝ) : collRem c t = fun ω => (c * t) • ceDtheta ω := rfl

/-- The chain of assumption (A4) holds exactly on the collinear family. -/
theorem coll_chain (c t : ℝ) (ω : Fin 4) :
    collDg c t ω = ceH *ᵥ collDtheta t ω + collRem c t ω := rfl

/-- The collinear gradient fluctuation is itself collinear with the jitter, with slope
`(1 + c) t`. -/
theorem collDg_eq (c t : ℝ) : collDg c t = fun ω => ((1 + c) * t) • ceDtheta ω := by
  funext ω
  simp only [collDg, collDtheta, collRem, ceH, Matrix.one_mulVec, ← add_smul]
  congr 1
  ring

/-- The jitter covariance of the collinear family. -/
theorem coll_V_eq (t : ℝ) : outerMoment ceWeight (collDtheta t) (collDtheta t) = (t * t) • ceD := by
  rw [collDtheta_eq, outerMoment_smul_smul, ce_V_eq_ceD]

/-- The remainder cross-moment `E[δθ rᵀ]` of the collinear family. -/
theorem coll_R_eq (c t : ℝ) :
    outerMoment ceWeight (collDtheta t) (collRem c t) = (t * (c * t)) • ceD := by
  rw [collDtheta_eq, collRem_eq, outerMoment_smul_smul, ce_V_eq_ceD]

/-- The slack-channel cross-covariance of the collinear family. -/
theorem coll_S_eq (c t : ℝ) :
    outerMoment ceWeight (collDtheta t) (collDg c t) = (t * ((1 + c) * t)) • ceD := by
  rw [collDtheta_eq, collDg_eq, outerMoment_smul_smul, ce_V_eq_ceD]

/-- The coupling term of the collinear family, in closed form: it is negative exactly when
`c < -1`. -/
theorem coll_crossTerm (c t : ℝ) (e : Fin 2 → ℝ) :
    crossTerm ceH (outerMoment ceWeight (collDtheta t) (collDg c t)) e
      = ((1 + c) * t ^ 2) * (e ⬝ᵥ e) := by
  rw [coll_S_eq, crossTerm_ceH_smul_ceD]
  ring

/-- The leading term `eᵀ H V H e` of the collinear family, in closed form. -/
theorem coll_leading (t : ℝ) (e : Fin 2 → ℝ) :
    e ⬝ᵥ ((ceH * outerMoment ceWeight (collDtheta t) (collDtheta t) * ceH) *ᵥ e)
      = (t ^ 2 / 2) * (e ⬝ᵥ e) := by
  rw [coll_V_eq, ceH]
  simp only [Matrix.one_mul, Matrix.mul_one]
  rw [quadForm_ceD]
  ring

/-- The remainder cross-moment `E[δθ rᵀ]` of the collinear family acts as a dilation, so its
operator norm is pinned at `|c| t² / 2` by an equation rather than by an inequality. -/
theorem coll_R_mulVec (c t : ℝ) (x : Fin 2 → ℝ) :
    outerMoment ceWeight (collDtheta t) (collRem c t) *ᵥ x = (c * t ^ 2 / 2) • x := by
  rw [coll_R_eq, Matrix.smul_mulVec, ceD_mulVec, smul_smul]
  congr 1
  ring

/-- **The collinear family satisfies (A4'-q) in the design note's own operator-norm shape**,
with the three quantities the clause names pinned by equations: `‖H̄‖_op = 1`,
`‖E[δθ rᵀ]‖_op = |c| t² / 2` and `λ_min(H̄ V H̄) = t² / 2`. The hypothesis `h` is the displayed
clause itself. Nothing here rests on the identification of the note's clause with the level
`ρ = 3`; that identification is a conclusion, drawn by `remainderControlledBy_of_opNormClause`. -/
theorem coll_remainderControlOpNorm (c t : ℝ) {nR lam : ℝ}
    (hnR : nR = |c| * (t ^ 2 / 2)) (hlam : lam = t ^ 2 / 2)
    (h : nR ≤ 3 / 2 * lam / 1) :
    RemainderControlOpNorm ceH (outerMoment ceWeight (collDtheta t) (collDtheta t))
      (outerMoment ceWeight (collDtheta t) (collRem c t)) 1 nR lam := by
  refine ⟨fun x => ?_, fun x => ?_, fun x => ?_, one_pos, h⟩
  · rw [ceH_mulVec, one_mul]
  · rw [coll_R_mulVec, vecNorm_smul, hnR,
      show c * t ^ 2 / 2 = c * (t ^ 2 / 2) by ring, abs_mul,
      abs_of_nonneg (by positivity : (0:ℝ) ≤ t ^ 2 / 2)]
  · rw [hlam]; exact coll_leading t x

/-- The collinear family is remainder-controlled at exactly the level `ρ = 2 |c|`, so it realizes
every level of (A4'-q)-style control. -/
theorem coll_remainderControlled (c t : ℝ) {rho : ℝ} (h : 2 * |c| ≤ rho) :
    RemainderControlledBy ceH (outerMoment ceWeight (collDtheta t) (collDtheta t))
      (outerMoment ceWeight (collDtheta t) (collRem c t)) rho := by
  intro x
  rw [coll_R_eq, coll_V_eq, ceH]
  simp only [Matrix.one_mul, Matrix.mul_one]
  rw [quadForm_ceD, quadForm_ceD]
  have hq : (0:ℝ) ≤ x ⬝ᵥ x := dotProduct_self_nonneg x
  have habs : |t * (c * t) / 2 * (x ⬝ᵥ x)| = |c| * (t ^ 2 / 2) * (x ⬝ᵥ x) := by
    rw [abs_mul, abs_of_nonneg hq, show t * (c * t) / 2 = c * (t ^ 2 / 2) by ring, abs_mul,
      abs_of_nonneg (by positivity : (0:ℝ) ≤ t ^ 2 / 2)]
  rw [habs]
  nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.mpr h) (sq_nonneg t)) hq]

/-- The (A4)-style remainder budget of the collinear family, in closed form. It tends to zero
with `t`, while the sign of the coupling term does not depend on `t` at all. -/
theorem coll_budget (c t : ℝ) :
    ∑ ω, ceWeight ω * (vecNorm (collDtheta t ω) * vecNorm (collRem c t ω)) = |t| * |c * t| := by
  have h : ∀ ω : Fin 4, ceWeight ω * (vecNorm (collDtheta t ω) * vecNorm (collRem c t ω))
      = (1/4) * (|t| * |c * t|) := by
    intro ω
    rw [show collDtheta t ω = t • ceDtheta ω from rfl,
      show collRem c t ω = (c * t) • ceDtheta ω from rfl,
      vecNorm_smul, vecNorm_smul, ce_vecNorm_dtheta, ceWeight]
    ring
  rw [Finset.sum_congr rfl fun ω _ => h ω]
  simp

/-- The jitter covariance of the collinear family is positive definite whenever the scale is
nonzero. -/
theorem coll_V_posDef {t : ℝ} (ht : t ≠ 0) :
    (outerMoment ceWeight (collDtheta t) (collDtheta t)).PosDef := by
  rw [coll_V_eq]
  exact Matrix.PosDef.smul ceD_posDef (mul_self_pos.mpr ht)

/-- The remainder of the collinear family is nonzero whenever `c` and `t` are, so the
configurations below are not disguised instances of the exact chain. -/
theorem collRem_ne_zero {c t : ℝ} (hc : c ≠ 0) (ht : t ≠ 0) : collRem c t ≠ 0 := by
  intro hcon
  have h0 : collRem c t 0 0 = 0 := by rw [hcon]; rfl
  rw [show collRem c t 0 0 = (c * t) * ceDtheta 0 0 from rfl] at h0
  norm_num [ceDtheta] at h0
  rcases h0 with h | h
  · exact hc h
  · exact ht h

/-- **(A6) is needed on top of (A4'-q), not merely on top of (A4).** There is a realizable
configuration — symmetric positive definite `H`, positive definite `V`, non-negative weights,
the chain of (A4) holding exactly with a nonzero remainder — whose remainder satisfies the
displayed operator-norm clause of (A4'-q) itself, with the three quantities that clause names
pinned by equations rather than bounded: `‖H̄‖_op = 1`, `‖E[δθ rᵀ]‖_op = 5/8` and
`λ_min(H̄ V H̄) = 1/2`, so that the clause reads `5/8 ≤ (3/2) · (1/2) / 1`. On that configuration
the total excess of (FW-1b) is non-negative **in every direction**, exactly as section 8 claims,
and yet the coupling term alone is strictly negative in the direction `(1,1)`. This is the
machine-checked form of section 11.3's "THE SIGN IS NOT AUTOMATIC" as that sentence is actually
used, namely against the assumption set FW-1 already carries. The quadratic-form level that the
same configuration realizes is `ρ = 5/2` (`coll_remainderControlled`), inside the window
`2 < ρ ≤ 3` identified above. -/
theorem coupling_sign_not_automatic_under_remainder_control :
    ∃ (w : Fin 4 → ℝ) (dth dg r : Fin 4 → Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ)
      (e : Fin 2 → ℝ),
      (∀ ω, 0 ≤ w ω) ∧ H.PosDef ∧ (outerMoment w dth dth).PosDef ∧
      (∀ ω, dg ω = H *ᵥ dth ω + r ω) ∧ r ≠ 0 ∧
      RemainderControlOpNorm H (outerMoment w dth dth) (outerMoment w dth r) 1 (5/8) (1/2) ∧
      RemainderControlledBy H (outerMoment w dth dth) (outerMoment w dth r) 3 ∧
      (∀ x : Fin 2 → ℝ, 0 ≤ crossTerm H (outerMoment w dth dg) x
          + x ⬝ᵥ ((H * outerMoment w dth dth * H) *ᵥ x)) ∧
      crossTerm H (outerMoment w dth dg) e < 0 := by
  have hop : RemainderControlOpNorm ceH (outerMoment ceWeight (collDtheta 1) (collDtheta 1))
      (outerMoment ceWeight (collDtheta 1) (collRem (-5/4) 1)) 1 (5/8) (1/2) :=
    coll_remainderControlOpNorm (-5/4) 1 (by norm_num) (by norm_num) (by norm_num)
  refine ⟨ceWeight, collDtheta 1, collDg (-5/4) 1, collRem (-5/4) 1, ceH, ceE,
    fun _ => by norm_num [ceWeight], ce_H_posDef, coll_V_posDef one_ne_zero,
    coll_chain (-5/4) 1, collRem_ne_zero (by norm_num) one_ne_zero, hop,
    remainderControlledBy_of_opNormClause hop, fun x => ?_, ?_⟩
  · rw [coll_crossTerm, coll_leading]
    nlinarith [dotProduct_self_nonneg x]
  · rw [coll_crossTerm, ceE_dot]
    norm_num

/-- **Smallness of the remainder does not restore the sign.** For every `ε > 0` the collinear
family, rescaled, gives a realizable configuration whose (A4)-style remainder budget
`E[‖δθ‖ ‖r‖]` is exactly `ε`, which still satisfies the operator-norm clause of (A4'-q) — there
`‖E[δθ rᵀ]‖_op = ε/2`, `‖H̄‖_op = 1` and `λ_min(H̄ V H̄) = 2ε/5`, so the clause reads
`ε/2 ≤ (3/2) · (2ε/5) / 1` — and whose coupling term is still strictly negative. The failure is
therefore not an artefact of a large remainder: it is carried by the remainder's direction, and
no bound on its size alone can rule it out. This is why the honest replacement for section 2's
sentence is (A6) or the uncorrelated-remainder hypothesis, and not "the remainder is small". -/
theorem coupling_sign_negative_with_arbitrarily_small_remainder {eps : ℝ} (heps : 0 < eps) :
    ∃ (w : Fin 4 → ℝ) (dth dg r : Fin 4 → Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ)
      (e : Fin 2 → ℝ),
      (∀ ω, 0 ≤ w ω) ∧ H.PosDef ∧ (outerMoment w dth dth).PosDef ∧
      (∀ ω, dg ω = H *ᵥ dth ω + r ω) ∧ r ≠ 0 ∧
      RemainderControlOpNorm H (outerMoment w dth dth) (outerMoment w dth r) 1
        (eps / 2) (2 * eps / 5) ∧
      RemainderControlledBy H (outerMoment w dth dth) (outerMoment w dth r) 3 ∧
      (∑ ω, w ω * (vecNorm (dth ω) * vecNorm (r ω))) = eps ∧
      crossTerm H (outerMoment w dth dg) e < 0 := by
  set t : ℝ := Real.sqrt (4 * eps / 5) with ht
  have htpos : 0 < t := Real.sqrt_pos.mpr (by positivity)
  have ht2 : t ^ 2 = 4 * eps / 5 := Real.sq_sqrt (by positivity)
  have hop : RemainderControlOpNorm ceH (outerMoment ceWeight (collDtheta t) (collDtheta t))
      (outerMoment ceWeight (collDtheta t) (collRem (-5/4) t)) 1 (eps / 2) (2 * eps / 5) := by
    refine coll_remainderControlOpNorm (-5/4) t ?_ ?_ (by linarith)
    · rw [ht2, show |(-5/4:ℝ)| = 5/4 by norm_num]; ring
    · rw [ht2]; ring
  refine ⟨ceWeight, collDtheta t, collDg (-5/4) t, collRem (-5/4) t, ceH, ceE,
    fun _ => by norm_num [ceWeight], ce_H_posDef, coll_V_posDef (ne_of_gt htpos),
    coll_chain (-5/4) t, collRem_ne_zero (by norm_num) (ne_of_gt htpos), hop,
    remainderControlledBy_of_opNormClause hop, ?_, ?_⟩
  · rw [coll_budget, abs_of_pos htpos, abs_of_nonpos (by nlinarith : (-5/4 : ℝ) * t ≤ 0)]
    nlinarith [ht2]
  · rw [coll_crossTerm, ceE_dot]
    nlinarith [ht2, htpos]

/-- **The threshold `ρ ≤ 2` of `crossTerm_nonneg_of_remainderControlled` is sharp.** For every
`ρ > 2` there is a realizable configuration remainder-controlled at level `ρ` whose coupling term
is strictly negative. Together with that theorem this pins the exact strength of remainder
control needed to make the sign of `C` automatic, and shows that assumption (A4'-q), which
supplies only `ρ = 3`, falls on the wrong side of it. -/
theorem remainderControl_threshold_sharp {rho : ℝ} (hrho : 2 < rho) :
    ∃ (w : Fin 4 → ℝ) (dth dg r : Fin 4 → Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ)
      (e : Fin 2 → ℝ),
      (∀ ω, 0 ≤ w ω) ∧ H.PosDef ∧ (outerMoment w dth dth).PosDef ∧
      (∀ ω, dg ω = H *ᵥ dth ω + r ω) ∧ r ≠ 0 ∧
      RemainderControlledBy H (outerMoment w dth dth) (outerMoment w dth r) rho ∧
      crossTerm H (outerMoment w dth dg) e < 0 := by
  refine ⟨ceWeight, collDtheta 1, collDg (-(rho/2)) 1, collRem (-(rho/2)) 1, ceH, ceE,
    fun _ => by norm_num [ceWeight], ce_H_posDef, coll_V_posDef one_ne_zero,
    coll_chain (-(rho/2)) 1, collRem_ne_zero (by intro hc; nlinarith [hc]) one_ne_zero,
    coll_remainderControlled (-(rho/2)) 1 ?_, ?_⟩
  · rw [abs_of_nonpos (by linarith : -(rho/2) ≤ 0)]
    linarith
  · rw [coll_crossTerm, ceE_dot]
    nlinarith

/-! ## The uncorrelated-remainder hypothesis is satisfiable with a nonzero remainder -/

/-- A constant remainder on the four-point sample space. Because the jitter samples cancel in
pairs, this remainder is uncorrelated with the jitter although it is nowhere zero. -/
def ceConstRem : Fin 4 → Fin 2 → ℝ := fun _ => ![0, 3]

theorem ceConstRem_uncorrelated : outerMoment ceWeight ceDtheta ceConstRem = 0 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [outerMoment, ceWeight, ceDtheta, ceConstRem, Fin.sum_univ_four, Matrix.sum_apply,
      Matrix.vecMulVec_apply, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons,
      Matrix.head_cons]

theorem ceConstRem_ne_zero : ceConstRem ≠ 0 := by
  intro hcon
  have h0 : ceConstRem 0 1 = 0 := by rw [hcon]; rfl
  norm_num [ceConstRem] at h0

/-- The gradient fluctuation carrying the constant remainder. -/
noncomputable def ceDgConst : Fin 4 → Fin 2 → ℝ := fun ω => ceH *ᵥ ceDtheta ω + ceConstRem ω

/-- `crossTerm_nonneg_of_uncorrelated_remainder` is not vacuous, and not a disguised statement
about the exact chain: there is a configuration with a nowhere-vanishing remainder `r` for which
`E[δθ rᵀ] = 0`, and on it the coupling term is non-negative in every direction. -/
theorem uncorrelated_remainder_satisfiable :
    ∃ (w : Fin 4 → ℝ) (dth dg r : Fin 4 → Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ),
      (∀ ω, 0 ≤ w ω) ∧ H.IsHermitian ∧ (∀ ω, dg ω = H *ᵥ dth ω + r ω) ∧
      outerMoment w dth r = 0 ∧ r ≠ 0 ∧
      ∀ e : Fin 2 → ℝ, 0 ≤ crossTerm H (outerMoment w dth dg) e :=
  ⟨ceWeight, ceDtheta, ceDgConst, ceConstRem, ceH, fun _ => by norm_num [ceWeight],
    ce_isHermitian, fun _ => rfl, ceConstRem_uncorrelated, ceConstRem_ne_zero,
    fun e => crossTerm_nonneg_of_uncorrelated_remainder (fun _ => by norm_num [ceWeight])
      ce_isHermitian (fun _ => rfl) ceConstRem_uncorrelated e⟩

/-- The `C = 0` case is a real one and not a disguised statement about the zero configuration:
there is a product sample space, with non-negative weights on both factors, on which neither the
latent-weight fluctuation nor the gradient fluctuation is the zero function, and yet `Σ_slack = 0`, so
the coupling channel contributes nothing in any direction and for any mean Hessian. -/
theorem crossTerm_eq_zero_of_independent_satisfiable :
    ∃ (w₁ : Fin 4 → ℝ) (w₂ : Fin 2 → ℝ) (u : Fin 4 → Fin 2 → ℝ) (v : Fin 2 → Fin 2 → ℝ),
      (∀ a, 0 ≤ w₁ a) ∧ (∀ b, 0 ≤ w₂ b) ∧ u ≠ 0 ∧ v ≠ 0 ∧
      (∀ i, ∑ a, w₁ a * u a i = 0) ∧
      ∀ (H : Matrix (Fin 2) (Fin 2) ℝ) (e : Fin 2 → ℝ),
        crossTerm H (outerMoment (fun p : Fin 4 × Fin 2 => w₁ p.1 * w₂ p.2)
          (fun p => u p.1) (fun p => v p.2)) e = 0 := by
  have hmean : ∀ i, ∑ a, ceWeight a * ceDtheta a i = 0 := by
    intro i
    fin_cases i <;>
      norm_num [ceWeight, ceDtheta, Fin.sum_univ_four, Matrix.cons_val_two, Matrix.cons_val_three,
        Matrix.tail_cons, Matrix.head_cons]
  refine ⟨ceWeight, fun _ => 1/2, ceDtheta, ![![1, 0], ![-1, 0]], fun _ => by norm_num [ceWeight],
    fun _ => by norm_num, ?_, ?_, hmean, fun H e => ?_⟩
  · intro hc
    have h0 : ceDtheta 0 0 = 0 := by rw [hc]; rfl
    norm_num [ceDtheta] at h0
  · intro hc
    have h0 : (![![1, 0], ![-1, 0]] : Fin 2 → Fin 2 → ℝ) 0 0 = 0 := by rw [hc]; rfl
    norm_num at h0
  · exact crossTerm_eq_zero_of_independent ceWeight (fun _ => 1/2) ceDtheta _ hmean H e

/-! ## Assumption (A6), and what it buys -/

/-- **Assumption (A6)** of section 11.3, verbatim as a quadratic-form statement: there is a
`κ > 0` such that on the shoulder band `B` the coupling term dominates `κ I`, that is,
`eᵀ C(x) e ≥ κ ‖e‖²` for every reference iterate `x` on the band and every direction `e`.

This is a *labelled assumption* and not a theorem:
`coupling_sign_not_automatic_under_remainder_control` above exhibits a configuration satisfying
every other hypothesis of FW-2', including (A4'-q), and
`positiveAlignment_fails_under_remainder_control` proves that (A6) fails on that very
configuration for every `κ`. It is also not vacuous: `positiveAlignment_satisfiable` exhibits a
realizable family with a nonzero remainder, indexed by a band with a continuum of reference
iterates, on which it holds with a single `κ`. The design
note's own reading is that "the lift helps exactly to the extent that `C` is positively aligned
on the band", and it records that the assumption is directly measurable, reducing in one
dimension to the sign of `H̄ σ_Jac²`. -/
def PositiveAlignmentOn {ι : Type*} (B : Set ι) (H S : ι → Matrix n n ℝ) (kappa : ℝ) : Prop :=
  0 < kappa ∧ ∀ x ∈ B, ∀ e : n → ℝ, kappa * (e ⬝ᵥ e) ≤ crossTerm (H x) (S x) e

/-- (A6) fails wherever the coupling term has a strictly negative direction, in particular on
the counterexample above. -/
theorem positiveAlignment_fails_of_crossTerm_neg {ι : Type*} {B : Set ι}
    {H S : ι → Matrix n n ℝ} {kappa : ℝ} {x : ι} (hx : x ∈ B) {e : n → ℝ}
    (hneg : crossTerm (H x) (S x) e < 0) : ¬ PositiveAlignmentOn B H S kappa := by
  rintro ⟨hk, halign⟩
  have h1 := halign x hx e
  have h2 : (0:ℝ) ≤ e ⬝ᵥ e := by
    rw [dotProduct]
    exact Finset.sum_nonneg fun i _ => mul_self_nonneg _
  nlinarith

/-- The projected update covariance of the **direct** baseline at the common reference iterate
of (FW-0): the latent-weight fluctuation is identically zero, so only the once-attenuated objective
noise survives. -/
noncomputable def projDiffusionDirect (s : ℝ) (Sobj : Matrix n n ℝ) (e : n → ℝ) : ℝ :=
  s ^ 2 * (e ⬝ᵥ (Sobj *ᵥ e))

/-- The projected update covariance of the **lift** at the same reference iterate, in the
decomposition of section 2 and (FW-1a): the direct baseline, plus the coupling term, plus the
magnitude term `eᵀ H V H e`, plus the third-and-higher-moment term `T3`, which is carried here
as an explicit real number rather than discarded. -/
noncomputable def projDiffusionLift (s : ℝ) (Sobj V H S : Matrix n n ℝ) (T3 : ℝ)
    (e : n → ℝ) : ℝ :=
  projDiffusionDirect s Sobj e
    + s ^ 2 * (crossTerm H S e + e ⬝ᵥ ((H * V * H) *ᵥ e)) + T3

/-- **Under (A6) the coupling channel strictly increases the projected diffusion, with an
explicit margin.** The third-order term is not assumed away: it is bounded by `τ ‖e‖²`, and the
conclusion holds whenever that bound is smaller than the alignment gain `s² κ`. The margin
`(s² κ - τ) ‖e‖²` is the quantity that (FW-2') feeds into the quasipotential estimate. -/
theorem projDiffusion_ge_of_positiveAlignment {Sobj V H S : Matrix n n ℝ} {e : n → ℝ}
    {s kappa tau T3 : ℝ} (hV : V.PosSemidef) (hH : H.IsHermitian)
    (halign : kappa * (e ⬝ᵥ e) ≤ crossTerm H S e)
    (hT3 : |T3| ≤ tau * (e ⬝ᵥ e)) :
    projDiffusionDirect s Sobj e + (s ^ 2 * kappa - tau) * (e ⬝ᵥ e)
      ≤ projDiffusionLift s Sobj V H S T3 e := by
  have hquad : 0 ≤ e ⬝ᵥ ((H * V * H) *ᵥ e) := quadForm_conj_nonneg hV hH e
  have hs : (0:ℝ) ≤ s ^ 2 := sq_nonneg s
  have hT3' : -(tau * (e ⬝ᵥ e)) ≤ T3 := (abs_le.mp hT3).1
  have hstep : s ^ 2 * (kappa * (e ⬝ᵥ e))
      ≤ s ^ 2 * (crossTerm H S e + e ⬝ᵥ ((H * V * H) *ᵥ e)) := by
    apply mul_le_mul_of_nonneg_left _ hs
    linarith
  rw [projDiffusionLift]
  nlinarith

/-- The strict form: under (A6) with a third-order term strictly dominated by the alignment
gain, the lift's projected diffusion is strictly larger than the direct baseline's in every
nonzero direction. -/
theorem projDiffusionDirect_lt_projDiffusionLift {Sobj V H S : Matrix n n ℝ} {e : n → ℝ}
    {s kappa tau T3 : ℝ} (hV : V.PosSemidef) (hH : H.IsHermitian)
    (halign : kappa * (e ⬝ᵥ e) ≤ crossTerm H S e)
    (hT3 : |T3| ≤ tau * (e ⬝ᵥ e)) (htau : tau < s ^ 2 * kappa) (he : e ≠ 0) :
    projDiffusionDirect s Sobj e < projDiffusionLift s Sobj V H S T3 e := by
  have hpos : 0 < e ⬝ᵥ e := by
    refine lt_of_le_of_ne ?_ ?_
    · rw [dotProduct]
      exact Finset.sum_nonneg fun i _ => mul_self_nonneg _
    · intro hc
      exact he (dotProduct_self_eq_zero.mp hc.symm)
  have hmain := projDiffusion_ge_of_positiveAlignment (s := s) (Sobj := Sobj) hV hH halign hT3
  nlinarith

/-- The band form of the previous theorem: under (A6) on the band `B`, at every reference
iterate on the band and in every nonzero direction, the coupling channel strictly increases the
projected diffusion. -/
theorem projDiffusion_strict_increase_on_band {ι : Type*} {B : Set ι}
    {Hf Sf Vf Sobjf : ι → Matrix n n ℝ} {T3f : ι → (n → ℝ) → ℝ} {s kappa tau : ℝ}
    (hA6 : PositiveAlignmentOn B Hf Sf kappa)
    (hV : ∀ x ∈ B, (Vf x).PosSemidef) (hH : ∀ x ∈ B, (Hf x).IsHermitian)
    (hT3 : ∀ x ∈ B, ∀ e : n → ℝ, |T3f x e| ≤ tau * (e ⬝ᵥ e))
    (htau : tau < s ^ 2 * kappa) {x : ι} (hx : x ∈ B) {e : n → ℝ} (he : e ≠ 0) :
    projDiffusionDirect s (Sobjf x) e
      < projDiffusionLift s (Sobjf x) (Vf x) (Hf x) (Sf x) (T3f x e) e :=
  projDiffusionDirect_lt_projDiffusionLift (hV x hx) (hH x hx) (hA6.2 x hx e) (hT3 x hx e) htau he

/-- Assumption (A6) fails outright on the counterexample configuration, for every `κ`: the
coupling term is negative in the direction `(1,1)`, so no positive lower bound of the form
`κ ‖e‖²` can hold there.  (A6) is therefore a genuine hypothesis and not a formality. -/
theorem positiveAlignment_fails_counterexample (kappa : ℝ) :
    ¬ PositiveAlignmentOn (Set.univ : Set Unit) (fun _ => ceH)
      (fun _ => outerMoment ceWeight ceDtheta ceDg) kappa :=
  positiveAlignment_fails_of_crossTerm_neg (B := Set.univ) (Set.mem_univ ()) (e := ceE)
    (by rw [ce_crossTerm_neg]; norm_num)

/-- The same, on the configuration that **does** satisfy (A4'-q): assumption (A6) fails there
too, for every `κ`. This is the statement that makes (A6) unprovable from the assumption set
FW-1 already carries, and it is what the docstring of `PositiveAlignmentOn` appeals to; the
configuration is the one of `coupling_sign_not_automatic_under_remainder_control`. -/
theorem positiveAlignment_fails_under_remainder_control (kappa : ℝ) :
    ¬ PositiveAlignmentOn (Set.univ : Set Unit) (fun _ => ceH)
      (fun _ => outerMoment ceWeight (collDtheta 1) (collDg (-5/4) 1)) kappa :=
  positiveAlignment_fails_of_crossTerm_neg (B := Set.univ) (Set.mem_univ ()) (e := ceE)
    (by rw [coll_crossTerm, ceE_dot]; norm_num)

/-- **Assumption (A6) is satisfiable, and not only in the degenerate case.** There is a
realizable family — non-negative weights, symmetric positive definite `H`, positive definite
`V`, the chain of (A4) holding exactly with a **nonzero** remainder at every point — indexed by
a band `B` with a continuum of reference iterates, on which the positive alignment holds with a
**single** `κ = 5/4` uniformly over the band. The uniformity is the content of (A6): a witness
on a one-point band would establish that the inequality can hold somewhere, but not that one
constant serves the whole shoulder, which is what FW-2' consumes. Note the contrast with
`coupling_sign_not_automatic_under_remainder_control`: the configurations differ only in the
sign of the collinearity coefficient `c`, which is exactly the design note's reading that
"the lift helps exactly to the extent that `C` is positively aligned on the band". -/
theorem positiveAlignment_satisfiable :
    ∃ (w : Fin 4 → ℝ) (dth : Fin 4 → Fin 2 → ℝ) (dg r : ℝ → Fin 4 → Fin 2 → ℝ)
      (H : Matrix (Fin 2) (Fin 2) ℝ) (kappa : ℝ),
      (∀ ω, 0 ≤ w ω) ∧ H.PosDef ∧ (outerMoment w dth dth).PosDef ∧
      (∀ x ∈ Set.Icc (0:ℝ) 1, ∀ ω, dg x ω = H *ᵥ dth ω + r x ω) ∧
      (∀ x ∈ Set.Icc (0:ℝ) 1, r x ≠ 0) ∧ 0 < kappa ∧
      PositiveAlignmentOn (Set.Icc (0:ℝ) 1) (fun _ => H)
        (fun x => outerMoment w dth (dg x)) kappa := by
  refine ⟨ceWeight, collDtheta 1, fun x => collDg (1/4 + x/4) 1,
    fun x => collRem (1/4 + x/4) 1, ceH, 5/4, fun _ => by norm_num [ceWeight], ce_H_posDef,
    coll_V_posDef one_ne_zero, fun x _ ω => coll_chain _ 1 ω,
    fun x hx => collRem_ne_zero (ne_of_gt (by linarith [hx.1] : (0:ℝ) < 1/4 + x/4)) one_ne_zero,
    by norm_num, ⟨by norm_num, fun x hx e => ?_⟩⟩
  rw [coll_crossTerm]
  nlinarith [dotProduct_self_nonneg e, hx.1]

/-- **`projDiffusion_strict_increase_on_band` is not vacuous.** Every hypothesis of that theorem
is discharged simultaneously on the positively aligned band of `positiveAlignment_satisfiable`,
with `s = 1`, `κ = 5/4`, and a third-order term that is genuinely nonzero and carries the
adversarial sign: `T3(e) = -‖e‖²/2`, controlled by `τ = 1/2 < s² κ`. The strict inequality is
then read off at every iterate of the band. Two things are being checked, not one: that the
hypothesis set — (A6) uniformly on a nondegenerate band, together with a third-order term
dominated by the alignment gain — is jointly satisfiable; and that satisfying it does not
require the third-order term to vanish, which a witness with `τ = 0` would have forced. -/
theorem projDiffusion_strict_increase_nonvacuous :
    PositiveAlignmentOn (Set.Icc (0:ℝ) 1) (fun _ => ceH)
        (fun x => outerMoment ceWeight (collDtheta 1) (collDg (1/4 + x/4) 1)) (5/4) ∧
      (∀ e : Fin 2 → ℝ, |(-(1:ℝ)/2) * (e ⬝ᵥ e)| ≤ 1/2 * (e ⬝ᵥ e)) ∧
      (-(1:ℝ)/2) * (ceE ⬝ᵥ ceE) ≠ 0 ∧
      ∀ x ∈ Set.Icc (0:ℝ) 1,
        projDiffusionDirect 1 ceD ceE
          < projDiffusionLift 1 ceD (outerMoment ceWeight (collDtheta 1) (collDtheta 1)) ceH
              (outerMoment ceWeight (collDtheta 1) (collDg (1/4 + x/4) 1))
              ((-(1:ℝ)/2) * (ceE ⬝ᵥ ceE)) ceE := by
  have hA6 : PositiveAlignmentOn (Set.Icc (0:ℝ) 1) (fun _ => ceH)
      (fun x => outerMoment ceWeight (collDtheta 1) (collDg (1/4 + x/4) 1)) (5/4) := by
    refine ⟨by norm_num, fun x hx e => ?_⟩
    rw [coll_crossTerm]
    nlinarith [dotProduct_self_nonneg e, hx.1]
  have hT3 : ∀ e : Fin 2 → ℝ, |(-(1:ℝ)/2) * (e ⬝ᵥ e)| ≤ 1/2 * (e ⬝ᵥ e) := by
    intro e
    rw [abs_mul, abs_of_nonneg (dotProduct_self_nonneg e)]
    norm_num
  refine ⟨hA6, hT3, ?_, fun x hx => ?_⟩
  · rw [ceE_dot]; norm_num
  · exact projDiffusion_strict_increase_on_band
      (Vf := fun _ => outerMoment ceWeight (collDtheta 1) (collDtheta 1)) (Sobjf := fun _ => ceD)
      (T3f := fun _ e => (-(1:ℝ)/2) * (e ⬝ᵥ e)) (s := 1) (tau := 1/2) hA6
      (fun _ _ => (coll_V_posDef one_ne_zero).posSemidef) (fun _ _ => ce_isHermitian)
      (fun _ _ e => hT3 e) (by norm_num) hx ceE_ne_zero

end IcnnLift
