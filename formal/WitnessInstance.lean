import Mathlib
import TraceLemmas
import CouplingSign
import BarrierRescaling
import KramersExitTime
import StationaryDensity

/-!
# One instance on which the whole hypothesis stack holds at once

Every conditional theorem of this development carries named hypotheses: positive
semidefiniteness of the jitter covariance `V` and symmetry of the mean Hessian `H̄` together with
assumption (A4)'s chain `δg = H̄ δθ + r`; the positive alignment (A6) of the coupling term on the
shoulder band; a positive continuous effective diffusion; a single-barrier pullback landscape
with a positive barrier action; and, for the first-passage statements, Dynkin's boundary-value
problem. Each module discharges its own hypotheses separately, and several carry a local
satisfiability witness. What none of them establishes is that the hypotheses can all be met **by
one and the same object**: a referee is entitled to suspect that the stack is inconsistent, that
(A6) can be bought only at the price of a degenerate `V`, or that the diffusion ordering of §2
and the single-barrier geometry of (A5) pull in opposite directions. This file removes that
suspicion by exhibiting one completely explicit model on which every hypothesis of the main
chain holds simultaneously, and by running the heavyweight theorems on it end to end.

The instance is `d = 2` over a four-point sample space with uniform weights — four equiprobable
minibatches, the finite-sample stand-in for the i.i.d. resampling of `eq:fluct`. The latent weight
fluctuation `δθ` takes the four Rademacher values `±(1,1)`, `±(1,-1)`, so it has zero mean and
jitter covariance `V = E[δθ δθᵀ] = I`, positive definite. The mean Hessian is
`H̄ = [[2,1],[1,2]]`, symmetric and positive definite with eigenvalues `1` and `3`. The
remainder of (A4) is **not** zero: it is `r = (1/6) δθ`, collinear with the jitter, which is the
positively aligned configuration that `CouplingSign.positiveAlignment_satisfiable` isolates as
the one on which the lift helps. The gradient fluctuation is then `δg = H̄ δθ + r` by
construction, so the chain of (A4) holds exactly and pointwise rather than in the mean, and the
slack-channel cross-covariance is the computed matrix `Σ_slack = V H̄ + E[δθ rᵀ]`. The positivity-map
slope is not an arbitrary constant either: `s = ψ'(0) = 1/2` is the derivative of the softplus
positivity map at the origin, proved here rather than posited (`wi_softplus_slope`).

Two facts about this configuration are worth stating in advance, because they are what make the
instance non-vacuous rather than merely consistent. First, the coupling channel is strictly
positive: `2 e_bᵀ H̄ V H̄ e_b = 36` and the full coupling term is `eᵀ C e = 38` in the bias
direction `e_b = (1,1)`, and (A6) holds **uniformly in the direction** with `κ = 7/3`, which is
the least eigenvalue of `C = H̄ Σ_slack + Σ_slackᵀ H̄ = [[32/3,25/3],[25/3,32/3]]`. Second, the
sign of the coupling term is here *derived and not assumed*: the remainder satisfies the
quadratic-form surrogate of (A4'-q) at level `ρ = 1`, strictly inside the `ρ ≤ 2` regime in
which `CouplingSign.crossTerm_nonneg_of_remainderControlled` proves the sign outright. The
instance therefore sits inside the intersection of the two sufficient conditions the development
identifies, not on the boundary of either.

The diffusion layer is computed from the linear-algebra layer and nowhere assumed. With
`σ_obj² = e_bᵀ Σ_obj e_b = 4`, the direct baseline's projected diffusion is
`σ²_dir = s² σ_obj² = 1` and the lift's is
`σ²_lift = s²σ_obj² + s² eᵀ C e + s² eᵀ H̄ V H̄ e = 1 + 19/2 + 9/2 = 15`, and these two numbers
are simultaneously the values of `CouplingSign.projDiffusionDirect`/`projDiffusionLift` and of
`BarrierRescaling.directDiffusion`/`liftDiffusion`, so no seam is crossed between the two
accountings. The landscape is the cubic `L̃(w) = w - w³/3` on `[-3/2, 3/2]`, whose basin bottom
is `w_b = -1` and whose barrier top is `w_s = 1`, with barrier action `α = 4/3 > 0`, monotone
climbing `L̃' = 1 - w² ≥ 0` on `[w_b, w_s]`, and the single-barrier structure `L̃ ≤ L̃(w_s)`,
`L̃(w_b) ≤ L̃` on the whole interval. The learning rate is `η = 1/10`, so the two Itô diffusion
coefficients of `dw = -L̃' dt + √η σ_eff dB` are `η σ²_dir = 1/10` and `η σ²_lift = 3/2`, and
both lie strictly inside the metastable window `σ² < 2α = 8/3` of (A5): the lift accelerates
escape without leaving the regime in which the escape problem is posed.

What this file does **not** claim. It does not prove any of the heavyweight theorems; it applies
them. It does not show that the paper's trained networks realize this instance — it shows that
the hypothesis stack is inhabited, which is the question "are we satisfying their conditions?"
in its machine-checkable form. It does not remove Dynkin's formula from the development: the
first-passage statement `wi_mean_first_passage_ordering` discharges `MeanExitTimeBVP` from
`KramersExitTime.meanExitTime_satisfies_bvp`, that is, by exhibiting the closed-form solution,
which is legitimate for a satisfiability claim but is not a derivation of Dynkin's formula from
an Itô calculus this mathlib does not have. And the third-order term `T3` of the update-covariance
decomposition is taken to be zero at the instance, so that the projected lift diffusion coincides
exactly with `liftDiffusion`; that this is a choice rather than a necessity is
`wi_third_order_budget_live`, which reruns the same theorem with a genuinely nonzero `T3` inside
a budget `τ = 1/4` strictly below the alignment gain `s²κ = 7/12`.

Every declaration of the instance carries the prefix `wi`.

## Results
* `wi_softplus_slope` — the positivity-map slope `s = 1/2` is `ψ'(0)` for `ψ = softplus`, proved.
* `wi_jitter_mean_zero`, `wi_V_eq`, `wi_cov_posDef`, `wi_V_posDef` — the latent-weight fluctuation is
  centred and its covariance `V = I` is positive definite.
* `wi_H_isHermitian`, `wi_H_posDef` — the mean Hessian is symmetric and positive definite.
* `wi_chain`, `wi_remainder_ne_zero`, `wi_R_eq`, `wi_S_eq`, `wi_S_eq_chain` — (A4)'s chain holds
  pointwise with a nonzero remainder, the slack-channel cross-covariance is computed, and it
  splits as `Σ_slack = V H̄ + E[δθ rᵀ]`.
* `wi_trace_slack` — the measured scalar `σ_Jac² = tr Σ_slack = 13/3` is strictly positive.
* `wi_HVH`, `wi_HS`, `wi_HR` — the three matrix products the estimates run through.
* `wi_leading_pos`, `wi_quadratic_channel` — `2 e_bᵀ H̄ V H̄ e_b = 36 > 0`: the coupling term of
  the exact chain, and the magnitude channel `18`.
* `wi_crossTerm_eq` — the full coupling term is `38` in the bias direction.
* `wi_alignment`, `wi_positiveAlignment`, `wi_alignment_sharp` — **(A6) holds** on the instance,
  uniformly in the direction, with `κ = 7/3`, and that constant is sharp.
* `wi_remainderControlled`, `wi_crossTerm_nonneg_derived` — (A4'-q)'s quadratic surrogate holds at
  `ρ = 1 ≤ 2`, so on this instance the sign of the coupling channel is derived, not assumed.
* `wi_direction_ne_zero`, `wi_objNoise_scalar` — the bias direction is nonzero and
  `σ_obj² = e_bᵀ Σ_obj e_b = 4`.
* `wi_coupling_eq`, `wi_quadratic_eq` — the two channels of the corrected §2 decomposition are
  `19/2` and `9/2`.
* `wi_sigma2Dir_eq`, `wi_sigma2Lift_eq`, `wi_projDiffusionDirect_eq`, `wi_projDiffusionLift_eq` —
  the two effective diffusions are `1` and `15`, on the matrix accounting and the scalar
  accounting alike.
* `wi_diffusion_ordering` — `0 < σ²_dir < σ²_lift`.
* `wi_projDiffusion_strict_increase`, `wi_projDiffusion_strict_increase_value` — **heavyweight
  (i)**: `CouplingSign`'s band theorem, all hypotheses discharged, conclusion `1 < 15`.
* `wi_third_order_budget_live`, `wi_third_order_lift_value` — the same with a genuinely nonzero
  third-order term inside its budget, giving `61/4`.
* `wi_sdeDir_eq`, `wi_sdeLift_eq`, `wi_sde_additive` — the two Itô diffusion coefficients
  `η σ²_dir = 1/10` and `η σ²_lift = 3/2`, additively coupled through `η σ_Jac² = 7/5`.
* `wi_hasDerivAt_landscape`, `wi_hasDerivAt_quasiDir`, `wi_hasDerivAt_quasiLift`, `wi_contScore`,
  `wi_contLandscape` — the regularity of (A1), and the two quasipotential equations.
* `wi_barrier_action`, `wi_grad_nonneg`, `wi_max`, `wi_min` — the single-barrier geometry of (A5)
  with `α = 4/3`, climbed monotonically.
* `wi_quasipotential_bracket`, `wi_quasipotential_bracket_strict` — **heavyweight (ii)**: the
  quasipotential bracket at explicit numbers, `1/12 ≤ 4/45 ≤ 1/9`, strict on both sides.
* `wi_barrier_exponent_lift_lt_direct`, `wi_barrier_exponents_eq` — **heavyweight (iii)**: the
  joined (L2') and Corollary 1 comparison, at the explicit numbers `16/9 < 80/3`.
* `wi_arrhenius_exponents_eq`, `wi_both_metastable`, `wi_metastable_window` — the same two
  exponents read off `arrheniusExponent`, and both diffusions strictly inside the metastable
  window of (A5).
* `wi_kramers_exponent_agree` — the two modules' exponent conventions agree at the instance.
* `wi_exit_time_upper_bound` — **heavyweight (iv)**: the lift's mean exit time is at most
  `10 exp(16/9)`, an explicit number.
* `wi_exit_time_lower_bound` — the matching Arrhenius lower bound at the direct diffusion.
* `wi_eventually_exit_ordering`, `wi_eventually_exit_ordering_additive` — the exit-time ordering,
  the second for the paper's own additively coupled pair `σ²` versus `σ² + σ_Jac²`.
* `wi_mean_first_passage_ordering` — **heavyweight (v)**: the paper-facing Dynkin form.
* `wi_stationary_density`, `wi_stationary_density_pos` — **heavyweight (vi)**: the density `(*)`
  is zero-flux stationary for the instance's lift diffusion, and is everywhere positive.
* `hypothesis_stack_is_inhabited` — the kill-shot: every hypothesis of the main chain, and the
  conclusions the chain delivers, in one statement the kernel checks.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. This file introduces **no new hypotheses**: every theorem in it
is unconditional, because its whole purpose is to discharge hypotheses rather than to state them.
The three things it does not do are named in the paragraph beginning "What this file does not
claim" above: it does not prove the
theorems it applies, it does not claim that a trained network realizes this instance, and its
`MeanExitTimeBVP` is discharged by exhibiting the closed-form solution rather than by deriving
Dynkin's formula, which remains the development's single largest unformalized analytic input.
-/

open Matrix Finset Set MeasureTheory Filter
open scoped Topology

namespace IcnnLift

/-! ## The instance

Dimension two, four equiprobable minibatches, and one explicit matrix for each object the
mechanism names. Nothing below is a variable: every number is written out. -/

/-- Uniform weights on the four-point sample space: four equiprobable minibatches, the
finite-sample form of the i.i.d. resampling of `eq:fluct`. -/
noncomputable def wiWeight : Fin 4 → ℝ := fun _ => (1 : ℝ) / 4

/-- The latent-weight fluctuation `δθ` of the latent iterate: the four Rademacher values
`±(1,1)` and `±(1,-1)`. It is centred (`wi_jitter_mean_zero`) and its covariance is the identity
(`wi_V_eq`). -/
def wiJitter : Fin 4 → Fin 2 → ℝ := ![![1, 1], ![-1, -1], ![1, -1], ![-1, 1]]

/-- The mean Hessian `H̄` of the pullback landscape at the reference iterate: symmetric and
positive definite, with eigenvalues `1` and `3`. Its off-diagonal entry is nonzero, so the
instance is genuinely two-dimensional and not a pair of decoupled scalar problems. -/
def wiHessian : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 1, 2]

/-- The jitter covariance `V = E[δθ δθᵀ]`, as a named matrix; `wi_V_eq` proves that the sample
space realizes it. -/
def wiCov : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 1]

/-- The remainder `r` of assumption (A4)'s chain `δg = H̄ δθ + r`. It is **not** zero: it is one
sixth of the jitter, collinear with it, which is the positively aligned configuration that
`CouplingSign.positiveAlignment_satisfiable` identifies as the one on which the lift helps. -/
noncomputable def wiRemainder : Fin 4 → Fin 2 → ℝ := fun ω => (1 / 6 : ℝ) • wiJitter ω

/-- The gradient fluctuation `δg`, defined so that (A4)'s chain holds exactly and pointwise on
the sample space rather than merely in the mean. -/
noncomputable def wiGradFluct : Fin 4 → Fin 2 → ℝ :=
  fun ω => wiHessian *ᵥ wiJitter ω + wiRemainder ω

/-- The bias-channel direction `e_b` on which the update covariance is projected. -/
def wiDirection : Fin 2 → ℝ := ![1, 1]

/-- The objective-gradient noise covariance `Σ_obj` at the reference iterate; its quadratic form
in the bias direction is `σ_obj² = 4`. -/
def wiObjNoise : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]

/-- The remainder cross-moment `R = E[δθ rᵀ]`, as a named matrix; `wi_R_eq` computes it. -/
noncomputable def wiRemCov : Matrix (Fin 2) (Fin 2) ℝ := (1 / 6 : ℝ) • wiCov

/-- The slack-channel cross-covariance `Σ_slack = E[δθ δgᵀ]`, as a named matrix; `wi_S_eq`
computes it from the sample space. -/
noncomputable def wiSlack : Matrix (Fin 2) (Fin 2) ℝ := !![13 / 6, 1; 1, 13 / 6]

/-- The positivity-map slope `s = ψ'(θ̄)` of the single-prefactor idealization (A2). Its value is not
posited: `wi_softplus_slope` proves that it is the derivative of the softplus positivity map at the
origin. -/
noncomputable def wiSlope : ℝ := 1 / 2

/-- The scalar objective-noise scale `σ_obj`, fixed by `σ_obj² = e_bᵀ Σ_obj e_b = 4`. -/
noncomputable def wiSigmaObj : ℝ := 2

/-- The learning rate `η` of the small-step diffusion approximation. -/
noncomputable def wiEta : ℝ := 1 / 10

/-! ## The positivity map, and where the number `1/2` comes from -/

/-- **The positivity-map slope is the softplus derivative at the origin.** `ψ(t) = log(1 + eᵗ)` is the
softplus positivity map the paper trains, and `ψ'(0) = 1/2`, so the single-prefactor idealization (A2)
is instantiated at a value the positivity map actually takes rather than at an arbitrary constant. -/
theorem wi_softplus_slope :
    HasDerivAt (fun t : ℝ => Real.log (1 + Real.exp t)) wiSlope 0 := by
  have h1 : HasDerivAt (fun t : ℝ => 1 + Real.exp t) (Real.exp 0) 0 :=
    (Real.hasDerivAt_exp 0).const_add 1
  have h2 : (fun t : ℝ => 1 + Real.exp t) 0 ≠ 0 := by simp
  have h3 := h1.log h2
  simpa [wiSlope, Real.exp_zero, show ((1 : ℝ) + 1) = 2 by norm_num] using h3

/-! ## Quadratic forms in dimension two

Every positivity check below is an explicit polynomial inequality in the two coordinates of the
direction, closed by `nlinarith` from a named square. No decision procedure and no numerical
evaluation is used. -/

/-- The quadratic form of an explicit `2 × 2` matrix, written out in coordinates. -/
theorem quadForm_fin_two (a b c d : ℝ) (x : Fin 2 → ℝ) :
    x ⬝ᵥ ((!![a, b; c, d] : Matrix (Fin 2) (Fin 2) ℝ) *ᵥ x)
      = a * x 0 ^ 2 + (b + c) * (x 0 * x 1) + d * x 1 ^ 2 := by
  simp [dotProduct, Matrix.mulVec, Fin.sum_univ_two]
  ring

/-- The Euclidean square norm in dimension two, written out in coordinates. -/
theorem dotProduct_self_fin_two (x : Fin 2 → ℝ) : x ⬝ᵥ x = x 0 ^ 2 + x 1 ^ 2 := by
  simp [dotProduct, Fin.sum_univ_two]
  ring

/-- A vector of `Fin 2 → ℝ` is nonzero exactly when one of its two coordinates is. -/
theorem exists_ne_zero_fin_two {x : Fin 2 → ℝ} (hx : x ≠ 0) : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
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

/-! ## The linear-algebra layer: (A2), (A4) and (A6) -/

/-- The latent-weight fluctuation is centred: `E[δθ] = 0`. This is what makes `E[δθ δθᵀ]` a
covariance rather than a bare second moment. -/
theorem wi_jitter_mean_zero (i : Fin 2) : ∑ ω, wiWeight ω * wiJitter ω i = 0 := by
  fin_cases i <;> simp [wiWeight, wiJitter, Fin.sum_univ_four]

/-- The jitter covariance of the instance is the identity. -/
theorem wi_V_eq : outerMoment wiWeight wiJitter wiJitter = wiCov := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [outerMoment, wiWeight, wiJitter, wiCov, Fin.sum_univ_four, Matrix.sum_apply,
      Matrix.vecMulVec_apply] <;>
    norm_num

/-- The mean Hessian is symmetric, which is what the coupling-term algebra of `CouplingSign`
and the congruence lemma of `TraceLemmas` require. -/
theorem wi_H_isHermitian : wiHessian.IsHermitian := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [wiHessian, Matrix.conjTranspose_apply]

/-- The mean Hessian is positive **definite**, so nothing below is an artefact of a degenerate
curvature. -/
theorem wi_H_posDef : wiHessian.PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos wi_H_isHermitian ?_
  intro x hx
  have hstar : (star x : Fin 2 → ℝ) = x := rfl
  rw [hstar, wiHessian, quadForm_fin_two]
  rcases exists_ne_zero_fin_two hx with h | h
  · have hpos : 0 < x 0 ^ 2 := by positivity
    nlinarith [sq_nonneg (x 0 + x 1), sq_nonneg (x 1)]
  · have hpos : 0 < x 1 ^ 2 := by positivity
    nlinarith [sq_nonneg (x 0 + x 1), sq_nonneg (x 0)]

/-- The jitter covariance is positive **definite**, so (A6) below is not bought at the price of
a degenerate `V`. -/
theorem wi_cov_posDef : wiCov.PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [wiCov, Matrix.conjTranspose_apply]
  · intro x hx
    have hstar : (star x : Fin 2 → ℝ) = x := rfl
    rw [hstar, wiCov, quadForm_fin_two]
    rcases exists_ne_zero_fin_two hx with h | h
    · have hpos : 0 < x 0 ^ 2 := by positivity
      nlinarith [sq_nonneg (x 1)]
    · have hpos : 0 < x 1 ^ 2 := by positivity
      nlinarith [sq_nonneg (x 0)]

/-- Positive semidefiniteness of the jitter covariance, the form the band theorems consume. -/
theorem wi_cov_posSemidef : wiCov.PosSemidef := wi_cov_posDef.posSemidef

/-- The same, stated of the sample-space second moment rather than of the named matrix. -/
theorem wi_V_posDef : (outerMoment wiWeight wiJitter wiJitter).PosDef := wi_V_eq ▸ wi_cov_posDef

/-- **Assumption (A4)'s chain holds exactly and pointwise** on the sample space: the gradient
fluctuation is the curvature-transported jitter plus the remainder at every one of the four
minibatches, not merely in the mean. -/
theorem wi_chain (ω : Fin 4) : wiGradFluct ω = wiHessian *ᵥ wiJitter ω + wiRemainder ω := rfl

/-- The remainder of (A4) is genuinely nonzero on the instance, so the chain is the honest one
`δg = H̄ δθ + r` and not the exact chain `δg = H̄ δθ` under which the sign of the coupling
channel would come for free. -/
theorem wi_remainder_ne_zero : wiRemainder ≠ 0 := by
  intro h
  have h0 := congrFun (congrFun h 0) 0
  norm_num [wiRemainder, wiJitter] at h0

/-- The remainder cross-moment `E[δθ rᵀ] = (1/6) I`. -/
theorem wi_R_eq : outerMoment wiWeight wiJitter wiRemainder = wiRemCov := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [outerMoment, wiWeight, wiJitter, wiRemainder, wiRemCov, wiCov, Fin.sum_univ_four,
      Matrix.sum_apply, Matrix.vecMulVec_apply] <;>
    norm_num

/-- The slack-channel cross-covariance `Σ_slack = E[δθ δgᵀ]` of the instance, computed from the
sample space. -/
theorem wi_S_eq : outerMoment wiWeight wiJitter wiGradFluct = wiSlack := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [outerMoment, wiWeight, wiJitter, wiGradFluct, wiRemainder, wiHessian, wiSlack,
      Fin.sum_univ_four, Matrix.sum_apply, dotProduct, Fin.sum_univ_two] <;>
    norm_num

/-- The matrix consequence of the chain, `Σ_slack = V H̄ + E[δθ rᵀ]`, verified on the instance
against `CouplingSign.outerMoment_chain`. -/
theorem wi_S_eq_chain : wiSlack = wiCov * wiHessian + wiRemCov := by
  rw [← wi_S_eq, ← wi_V_eq, ← wi_R_eq]
  exact outerMoment_chain wi_H_isHermitian wi_chain

/-- The trace of the slack-channel cross-covariance, the scalar `σ_Jac²` the paper measures, is
strictly positive on the instance. -/
theorem wi_trace_slack : wiSlack.trace = 13 / 3 := by
  simp [wiSlack, Matrix.trace_fin_two]
  norm_num

/-- The curvature congruence `H̄ V H̄` of the instance. -/
theorem wi_HVH : wiHessian * wiCov * wiHessian = !![5, 4; 4, 5] := by
  simp [wiHessian, wiCov]
  norm_num [Matrix.mul_fin_two]

/-- The product `H̄ Σ_slack`, whose quadratic form is half the coupling term. -/
theorem wi_HS : wiHessian * wiSlack = !![16 / 3, 25 / 6; 25 / 6, 16 / 3] := by
  simp [wiHessian, wiSlack]
  norm_num [Matrix.mul_fin_two]

/-- The product `H̄ E[δθ rᵀ]`, whose quadratic form is the remainder functional of (A4'-q). -/
theorem wi_HR : wiHessian * wiRemCov = !![1 / 3, 1 / 6; 1 / 6, 1 / 3] := by
  simp [wiHessian, wiRemCov, wiCov, Matrix.smul_of]
  norm_num [Matrix.mul_fin_two]

/-- **The coupling term of the exact chain is strictly positive at the bias direction**:
`2 e_bᵀ H̄ V H̄ e_b = 36`. This is the quantity §2 of the design note says carries the sign, and
on the instance it does. -/
theorem wi_leading_pos :
    2 * (wiDirection ⬝ᵥ ((wiHessian * wiCov * wiHessian) *ᵥ wiDirection)) = 36 := by
  rw [wi_HVH, wiDirection, quadForm_fin_two]
  norm_num

/-- The magnitude channel `e_bᵀ H̄ V H̄ e_b = 18`, strictly positive. -/
theorem wi_quadratic_channel :
    wiDirection ⬝ᵥ ((wiHessian * wiCov * wiHessian) *ᵥ wiDirection) = 18 := by
  rw [wi_HVH, wiDirection, quadForm_fin_two]
  norm_num

/-- **The full coupling term at the bias direction is `38`**, strictly larger than the `36` of
the exact chain: on this instance the remainder helps rather than hurts, which is exactly what
distinguishes it from the counterexample family of `CouplingSign`. -/
theorem wi_crossTerm_eq : crossTerm wiHessian wiSlack wiDirection = 38 := by
  rw [crossTerm_eq_two_mul wi_H_isHermitian, wi_HS, wiDirection, quadForm_fin_two]
  norm_num

/-- **Assumption (A6) holds on the instance, uniformly in the direction, with `κ = 7/3`.** The
coupling matrix is `C = H̄ Σ_slack + Σ_slackᵀ H̄ = [[32/3, 25/3], [25/3, 32/3]]`, whose least
eigenvalue is `32/3 - 25/3 = 7/3`; the inequality reduces to the single square
`(25/3)(e₀ + e₁)² ≥ 0`. This is the hypothesis that `CouplingSign` proves cannot be derived in
general, so exhibiting it here is the point of the file. -/
theorem wi_alignment (e : Fin 2 → ℝ) : (7 / 3 : ℝ) * (e ⬝ᵥ e) ≤ crossTerm wiHessian wiSlack e := by
  rw [crossTerm_eq_two_mul wi_H_isHermitian, wi_HS, quadForm_fin_two, dotProduct_self_fin_two]
  nlinarith [sq_nonneg (e 0 + e 1)]

/-- **The alignment constant `κ = 7/3` is sharp**: it is attained in the direction `(1,-1)`, the
eigendirection of `C` for its least eigenvalue, so no larger `κ` satisfies (A6) here. The
instance is therefore reported at its true alignment strength and not at a padded one. -/
theorem wi_alignment_sharp :
    crossTerm wiHessian wiSlack ![1, -1] = (7 / 3 : ℝ) * ((![1, -1] : Fin 2 → ℝ) ⬝ᵥ ![1, -1]) := by
  rw [crossTerm_eq_two_mul wi_H_isHermitian, wi_HS, quadForm_fin_two, dotProduct_self_fin_two]
  norm_num

/-- (A6) in the form `CouplingSign.PositiveAlignmentOn` states it, at the single reference
iterate of the band. -/
theorem wi_positiveAlignment :
    PositiveAlignmentOn (Set.univ : Set Unit) (fun _ => wiHessian) (fun _ => wiSlack) (7 / 3) :=
  ⟨by norm_num, fun _ _ e => wi_alignment e⟩

/-- **The remainder is controlled at level `ρ = 1`**, strictly inside the `ρ ≤ 2` regime of
`CouplingSign.crossTerm_nonneg_of_remainderControlled`, and well inside the `ρ = 3` that (A4'-q)
supplies. The inequality is `(1/3)(x₀² + x₀x₁ + x₁²) ≤ 5x₀² + 8x₀x₁ + 5x₁²`. -/
theorem wi_remainderControlled : RemainderControlledBy wiHessian wiCov wiRemCov 1 := by
  intro x
  rw [wi_HR, wi_HVH, quadForm_fin_two, quadForm_fin_two]
  rcases abs_cases ((1 / 3 : ℝ) * x 0 ^ 2 + (1 / 6 + 1 / 6) * (x 0 * x 1) + (1 / 3) * x 1 ^ 2)
    with ⟨h1, _⟩ | ⟨h1, _⟩ <;> rw [h1] <;>
    nlinarith [sq_nonneg (x 0 + x 1), sq_nonneg (x 0 - x 1)]

/-- **On the instance the sign of the coupling channel is derived, not assumed.** Because the
remainder is controlled at `ρ = 1 ≤ 2`, `CouplingSign.crossTerm_nonneg_of_remainderControlled`
delivers the non-negativity of the coupling term in every direction without appeal to (A6). The
instance therefore lies inside the intersection of the two sufficient conditions this
development identifies, not on the boundary of either. -/
theorem wi_crossTerm_nonneg_derived (e : Fin 2 → ℝ) : 0 ≤ crossTerm wiHessian wiSlack e :=
  crossTerm_nonneg_of_remainderControlled (V := wiCov) (R := wiRemCov) wi_cov_posSemidef
    wi_H_isHermitian wi_S_eq_chain (by norm_num) wi_remainderControlled e

/-! ## The diffusion layer

The two effective diffusions are **computed** from the linear-algebra layer above. Nothing here
is posited: `wiCoupling` and `wiQuadratic` are the two channels of the corrected §2
decomposition evaluated at the instance, and the numbers `1` and `15` follow. -/

/-- The bias direction is nonzero, which is what the strict form of the band theorem needs. -/
theorem wi_direction_ne_zero : wiDirection ≠ 0 := by
  intro h
  have h0 := congrFun h 0
  norm_num [wiDirection] at h0

/-- The objective-noise scale of (A3) read off the matrix: `e_bᵀ Σ_obj e_b = σ_obj² = 4`. -/
theorem wi_objNoise_scalar :
    wiDirection ⬝ᵥ (wiObjNoise *ᵥ wiDirection) = wiSigmaObj ^ 2 := by
  rw [wiObjNoise, wiDirection, quadForm_fin_two]
  norm_num [wiSigmaObj]

/-- The **coupling channel** `s² e_bᵀ C e_b` of the corrected decomposition, at the instance. -/
noncomputable def wiCoupling : ℝ := wiSlope ^ 2 * crossTerm wiHessian wiSlack wiDirection

/-- The **magnitude channel** `s² e_bᵀ H̄ V H̄ e_b` of the corrected decomposition, at the
instance. -/
noncomputable def wiQuadratic : ℝ :=
  wiSlope ^ 2 * (wiDirection ⬝ᵥ ((wiHessian * wiCov * wiHessian) *ᵥ wiDirection))

/-- The coupling channel is `19/2`, strictly positive. -/
theorem wi_coupling_eq : wiCoupling = 19 / 2 := by
  rw [wiCoupling, wi_crossTerm_eq]
  norm_num [wiSlope]

/-- The magnitude channel is `9/2`, strictly positive. -/
theorem wi_quadratic_eq : wiQuadratic = 9 / 2 := by
  rw [wiQuadratic, wi_quadratic_channel]
  norm_num [wiSlope]

/-- The direct baseline's effective diffusion on the shoulder band: constant, because the
baseline has no latent-weight fluctuation to make it vary. -/
noncomputable def wiSigma2Dir : ℝ → ℝ := fun _ => directDiffusion wiSlope wiSigmaObj

/-- The lift's effective diffusion on the shoulder band, in the corrected §2 form
`s²σ_obj² + coupling + quadratic`. -/
noncomputable def wiSigma2Lift : ℝ → ℝ :=
  fun _ => liftDiffusion wiSlope wiSigmaObj wiCoupling wiQuadratic

/-- The direct baseline's effective diffusion is `1`. -/
theorem wi_sigma2Dir_eq (u : ℝ) : wiSigma2Dir u = 1 := by
  norm_num [wiSigma2Dir, directDiffusion, wiSlope, wiSigmaObj]

/-- The lift's effective diffusion is `15`. -/
theorem wi_sigma2Lift_eq (u : ℝ) : wiSigma2Lift u = 15 := by
  rw [wiSigma2Lift, liftDiffusion, wi_coupling_eq, wi_quadratic_eq]
  norm_num [wiSlope, wiSigmaObj]

/-- **The two effective diffusions are positive and strictly ordered**, `0 < 1 < 15`. -/
theorem wi_diffusion_ordering :
    0 < wiSigma2Dir 0 ∧ wiSigma2Dir 0 < wiSigma2Lift 0 := by
  rw [wi_sigma2Dir_eq, wi_sigma2Lift_eq]
  norm_num

/-- The direct baseline's **projected** update covariance, computed from the matrix data, is the
same number `1` as its scalar effective diffusion: the matrix accounting of `CouplingSign` and
the scalar accounting of `BarrierRescaling` agree at the instance. -/
theorem wi_projDiffusionDirect_eq :
    projDiffusionDirect wiSlope wiObjNoise wiDirection = wiSigma2Dir 0 := by
  rw [projDiffusionDirect, wiObjNoise, wiDirection, quadForm_fin_two, wi_sigma2Dir_eq]
  norm_num [wiSlope]

/-- The lift's **projected** update covariance, computed from the matrix data with a vanishing
third-order term, is the same number `15` as its scalar effective diffusion. -/
theorem wi_projDiffusionLift_eq :
    projDiffusionLift wiSlope wiObjNoise wiCov wiHessian wiSlack 0 wiDirection
      = wiSigma2Lift 0 := by
  rw [projDiffusionLift, wi_projDiffusionDirect_eq, wi_crossTerm_eq, wi_quadratic_channel,
    wi_sigma2Dir_eq, wi_sigma2Lift_eq]
  norm_num [wiSlope]

/-- **Heavyweight (i): `CouplingSign.projDiffusion_strict_increase_on_band` at the instance.**
Every hypothesis of the band theorem — (A6) with `κ = 7/3`, positive semidefiniteness of `V`,
symmetry of `H̄`, a third-order term inside the budget `τ = 1/4`, and the domination
`τ < s²κ = 7/12` — is discharged, and the conclusion is the strict increase of the projected
diffusion in the bias direction. -/
theorem wi_projDiffusion_strict_increase :
    projDiffusionDirect wiSlope wiObjNoise wiDirection
      < projDiffusionLift wiSlope wiObjNoise wiCov wiHessian wiSlack 0 wiDirection :=
  projDiffusion_strict_increase_on_band
    (B := (Set.univ : Set Unit)) (Hf := fun _ => wiHessian) (Sf := fun _ => wiSlack)
    (Vf := fun _ => wiCov) (Sobjf := fun _ => wiObjNoise) (T3f := fun _ _ => 0)
    (s := wiSlope) (kappa := 7 / 3) (tau := 1 / 4)
    wi_positiveAlignment (fun _ _ => wi_cov_posSemidef) (fun _ _ => wi_H_isHermitian)
    (fun _ _ e => by
      simpa using mul_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 4) (dotProduct_self_nonneg e))
    (by norm_num [wiSlope]) (Set.mem_univ ()) wi_direction_ne_zero

/-- The conclusion of heavyweight (i) at the explicit numbers: `1 < 15`. -/
theorem wi_projDiffusion_strict_increase_value : wiSigma2Dir 0 < wiSigma2Lift 0 := by
  rw [← wi_projDiffusionDirect_eq, ← wi_projDiffusionLift_eq]
  exact wi_projDiffusion_strict_increase

/-- **The third-order budget is live, not degenerate.** The instance's own third-order term is
zero, which is what makes the projected lift diffusion agree exactly with `liftDiffusion`; but
that is a choice, not a necessity. Rerunning the same band theorem with the genuinely nonzero
third-order term `T3(e) = ‖e‖²/8`, still inside the budget `τ = 1/4` and hence still strictly
below the alignment gain `s²κ = 7/12`, gives the same strict increase. -/
theorem wi_third_order_budget_live :
    projDiffusionDirect wiSlope wiObjNoise wiDirection
      < projDiffusionLift wiSlope wiObjNoise wiCov wiHessian wiSlack
          ((1 / 8) * (wiDirection ⬝ᵥ wiDirection)) wiDirection :=
  projDiffusion_strict_increase_on_band
    (B := (Set.univ : Set Unit)) (Hf := fun _ => wiHessian) (Sf := fun _ => wiSlack)
    (Vf := fun _ => wiCov) (Sobjf := fun _ => wiObjNoise)
    (T3f := fun _ e => (1 / 8) * (e ⬝ᵥ e))
    (s := wiSlope) (kappa := 7 / 3) (tau := 1 / 4)
    wi_positiveAlignment (fun _ _ => wi_cov_posSemidef) (fun _ _ => wi_H_isHermitian)
    (fun _ _ e => by
      have hnn : (0 : ℝ) ≤ e ⬝ᵥ e := dotProduct_self_nonneg e
      rw [abs_of_nonneg (by positivity : (0 : ℝ) ≤ (1 / 8) * (e ⬝ᵥ e))]
      nlinarith)
    (by norm_num [wiSlope]) (Set.mem_univ ()) wi_direction_ne_zero

/-- The value of the lift's projected diffusion when the third-order term is carried: `61/4`,
still above the direct baseline's `1`. -/
theorem wi_third_order_lift_value :
    projDiffusionLift wiSlope wiObjNoise wiCov wiHessian wiSlack
        ((1 / 8) * (wiDirection ⬝ᵥ wiDirection)) wiDirection = 61 / 4 := by
  rw [projDiffusionLift, wi_projDiffusionDirect_eq, wi_crossTerm_eq, wi_quadratic_channel,
    wi_sigma2Dir_eq, wiDirection, dotProduct_self_fin_two]
  norm_num [wiSlope]

/-! ## The landscape layer: (A1) and the single-barrier geometry of (A5)

One cubic pullback landscape serves both halves of the chain: its monotone climb across
`[w_b, w_s]` is what the quasipotential bracket of `BarrierRescaling` needs, and its
single-barrier shape on the whole of `[a, b]` is what the Arrhenius bounds of `KramersExitTime`
need. -/

/-- The pullback landscape `L̃(w) = w - w³/3`: a genuine single barrier, with a basin bottom at
`w_b = -1` and a barrier top at `w_s = 1`. -/
noncomputable def wiLandscape : ℝ → ℝ := fun u => u - u ^ 3 / 3

/-- Its derivative `L̃'(w) = 1 - w²`, continuous, which is assumption (A1). -/
def wiScore : ℝ → ℝ := fun u => 1 - u ^ 2

/-- The basin bottom `w_b`. -/
noncomputable def wiBasin : ℝ := -1

/-- The barrier top `w_s`. -/
noncomputable def wiSaddle : ℝ := 1

/-- The reflecting endpoint `a` of the escape problem. -/
noncomputable def wiLeft : ℝ := -(3 / 2)

/-- The absorbing endpoint `b` of the escape problem. -/
noncomputable def wiRight : ℝ := 3 / 2

/-- The starting point of the escape problem: the basin bottom. -/
noncomputable def wiStart : ℝ := -1

/-- The direct baseline's quasipotential, the antiderivative of `L̃'/σ_eff²` for its own
diffusion. -/
noncomputable def wiQuasiDir : ℝ → ℝ := fun u => wiLandscape u

/-- The lift's quasipotential, the antiderivative of `L̃'/σ_eff²` for its own diffusion. -/
noncomputable def wiQuasiLift : ℝ → ℝ := fun u => wiLandscape u / 15

/-- The Itô diffusion coefficient of the direct baseline's escape problem: `η σ²_dir = 1/10`.
The learning rate enters here because the small-step diffusion is
`dw = -L̃' dt + √η σ_eff dB`. -/
noncomputable def wiSdeDir : ℝ := wiEta * wiSigma2Dir 0

/-- The Itô diffusion coefficient of the lift's escape problem: `η σ²_lift = 3/2`. -/
noncomputable def wiSdeLift : ℝ := wiEta * wiSigma2Lift 0

/-- The excess the lift injects, in the same units: `η σ_Jac² = 7/5`. This is the paper's own
additively coupled pair, `σ²` against `σ² + σ_Jac²`. -/
noncomputable def wiSdeJac : ℝ := wiEta * (wiSigma2Lift 0 - wiSigma2Dir 0)

/-- The direct baseline's Itô diffusion coefficient is `1/10`. -/
theorem wi_sdeDir_eq : wiSdeDir = 1 / 10 := by
  rw [wiSdeDir, wi_sigma2Dir_eq]
  norm_num [wiEta]

/-- The lift's Itô diffusion coefficient is `3/2`. -/
theorem wi_sdeLift_eq : wiSdeLift = 3 / 2 := by
  rw [wiSdeLift, wi_sigma2Lift_eq]
  norm_num [wiEta]

/-- The injected excess is `7/5`, and the two coefficients really are additively coupled. -/
theorem wi_sde_additive : wiSdeJac = 7 / 5 ∧ wiSdeDir + wiSdeJac = wiSdeLift := by
  refine ⟨?_, ?_⟩
  · rw [wiSdeJac, wi_sigma2Dir_eq, wi_sigma2Lift_eq]
    norm_num [wiEta]
  · rw [wiSdeDir, wiSdeJac, wiSdeLift, wi_sigma2Dir_eq, wi_sigma2Lift_eq]
    norm_num [wiEta]

/-- The landscape is differentiable everywhere with derivative `L̃'`. -/
theorem wi_hasDerivAt_landscape (u : ℝ) : HasDerivAt wiLandscape (wiScore u) u := by
  have h2 : HasDerivAt (fun y : ℝ => y ^ 3) ((3 : ℝ) * u ^ 2) u := by
    simpa using hasDerivAt_pow 3 u
  have h3 := (hasDerivAt_id u).sub (h2.div_const 3)
  refine h3.congr_deriv ?_
  simp only [wiScore]
  ring

/-- The direct baseline's quasipotential solves `U' = L̃'/σ_eff²` for its own diffusion. -/
theorem wi_hasDerivAt_quasiDir (u : ℝ) :
    HasDerivAt wiQuasiDir (wiScore u / wiSigma2Dir u) u := by
  have h : HasDerivAt wiQuasiDir (wiScore u) u := wi_hasDerivAt_landscape u
  simpa [wi_sigma2Dir_eq] using h

/-- The lift's quasipotential solves `U' = L̃'/σ_eff²` for its own diffusion. -/
theorem wi_hasDerivAt_quasiLift (u : ℝ) :
    HasDerivAt wiQuasiLift (wiScore u / wiSigma2Lift u) u := by
  have h : HasDerivAt wiQuasiLift (wiScore u / 15) u := (wi_hasDerivAt_landscape u).div_const 15
  simpa [wi_sigma2Lift_eq] using h

/-- `L̃'` is continuous on the climbing band, which is assumption (A1). -/
theorem wi_contScore : ContinuousOn wiScore (Icc wiBasin wiSaddle) :=
  (continuous_const.sub (continuous_pow 2)).continuousOn

/-- `L̃` is continuous on the escape interval. -/
theorem wi_contLandscape : ContinuousOn wiLandscape (Icc wiLeft wiRight) :=
  (continuous_id.sub ((continuous_pow 3).div_const 3)).continuousOn

/-- **The barrier action is `α = 4/3 > 0`.** -/
theorem wi_barrier_action : wiLandscape wiSaddle - wiLandscape wiBasin = 4 / 3 := by
  norm_num [wiLandscape, wiSaddle, wiBasin]

/-- **The barrier is climbed monotonically**: `L̃' ≥ 0` throughout `[w_b, w_s]`, which is the
hypothesis the two-sided quasipotential bracket needs and cannot do without. -/
theorem wi_grad_nonneg : ∀ u ∈ Icc wiBasin wiSaddle, 0 ≤ wiScore u := by
  intro u hu
  have h1 : (-1 : ℝ) ≤ u := hu.1
  have h2 : u ≤ (1 : ℝ) := hu.2
  simp only [wiScore]
  nlinarith

/-- **The single-barrier structure of (A5), upper half**: the barrier top dominates the landscape
on the whole escape interval. -/
theorem wi_max : ∀ y ∈ Icc wiLeft wiRight, wiLandscape y ≤ wiLandscape wiSaddle := by
  intro y hy
  have h1 : (-(3 / 2) : ℝ) ≤ y := hy.1
  simp only [wiLandscape, wiSaddle]
  nlinarith [mul_nonneg (sq_nonneg (y - 1)) (show (0 : ℝ) ≤ y + 2 by linarith)]

/-- **The single-barrier structure of (A5), lower half**: the basin bottom is dominated by the
landscape on the whole escape interval. -/
theorem wi_min : ∀ z ∈ Icc wiLeft wiRight, wiLandscape wiBasin ≤ wiLandscape z := by
  intro z hz
  have h2 : z ≤ (3 / 2 : ℝ) := hz.2
  simp only [wiLandscape, wiBasin]
  nlinarith [mul_nonneg (sq_nonneg (z + 1)) (show (0 : ℝ) ≤ 2 - z by linarith)]

/-! ## The heavyweight theorems, instantiated -/

/-- **Heavyweight (ii): `BarrierRescaling.quasipotential_gap_bounds_of_continuousOn` at the
instance.** With the lift's diffusion bracketed by `c_min = 12` and `c_max = 16` — a genuinely
wide bracket, not a disguised equality — the quasipotential gap across the barrier is trapped
between `α/16 = 1/12` and `α/12 = 1/9`. -/
theorem wi_quasipotential_bracket :
    (wiLandscape wiSaddle - wiLandscape wiBasin) / 16 ≤ wiQuasiLift wiSaddle - wiQuasiLift wiBasin ∧
      wiQuasiLift wiSaddle - wiQuasiLift wiBasin
        ≤ (wiLandscape wiSaddle - wiLandscape wiBasin) / 12 :=
  quasipotential_gap_bounds_of_continuousOn (wb := wiBasin) (ws := wiSaddle)
    (cmin := 12) (cmax := 16) (by norm_num [wiBasin, wiSaddle]) (by norm_num)
    (fun u _ => by rw [wi_sigma2Lift_eq]; norm_num)
    (fun u _ => by rw [wi_sigma2Lift_eq]; norm_num)
    wi_grad_nonneg (fun u _ => wi_hasDerivAt_landscape u) (fun u _ => wi_hasDerivAt_quasiLift u)
    wi_contScore continuousOn_const

/-- The bracket of heavyweight (ii) at explicit numbers, **strict on both sides**: the exact gap
is `4/45`, and `1/12 < 4/45 < 1/9`. So the bracket is a genuine approximation with a measured
width, not an identity in disguise. -/
theorem wi_quasipotential_bracket_strict :
    wiQuasiLift wiSaddle - wiQuasiLift wiBasin = 4 / 45 ∧
      (wiLandscape wiSaddle - wiLandscape wiBasin) / 16 < 4 / 45 ∧
      (4 : ℝ) / 45 < (wiLandscape wiSaddle - wiLandscape wiBasin) / 12 := by
  rw [wi_barrier_action]
  refine ⟨?_, by norm_num, by norm_num⟩
  norm_num [wiQuasiLift, wiLandscape, wiSaddle, wiBasin]

/-- **Heavyweight (iii): `BarrierRescaling.barrier_exponent_lift_lt_direct_of_const_bands` at the
instance.** Every hypothesis of the joined (L2') and Corollary 1 comparison is discharged — a
positive barrier action, a positive learning rate, a positive direct diffusion, a non-negative
coupling channel, a non-negative magnitude channel, non-degeneracy of one of them, the two
constant bands, the two quasipotential equations, and interval integrability — and the
conclusion is that the lift's true quasipotential barrier exponent is strictly below the direct
baseline's. -/
theorem wi_barrier_exponent_lift_lt_direct :
    (2 / wiEta) * (wiQuasiLift wiSaddle - wiQuasiLift wiBasin)
      < (2 / wiEta) * (wiQuasiDir wiSaddle - wiQuasiDir wiBasin) :=
  barrier_exponent_lift_lt_direct_of_const_bands
    (Ltilde := wiLandscape) (dL := wiScore) (wb := wiBasin) (ws := wiSaddle)
    (UeffDir := wiQuasiDir) (UeffLift := wiQuasiLift)
    (sigma2Dir := wiSigma2Dir) (sigma2Lift := wiSigma2Lift)
    (eta := wiEta) (s := wiSlope) (sigmaObj := wiSigmaObj)
    (coupling := wiCoupling) (quadratic := wiQuadratic)
    (by rw [wi_barrier_action]; norm_num) (by norm_num [wiEta])
    (by norm_num [directDiffusion, wiSlope, wiSigmaObj])
    (by rw [wi_coupling_eq]; norm_num) (by rw [wi_quadratic_eq]; norm_num)
    (Or.inl (by rw [wi_coupling_eq]; norm_num))
    (fun u _ => rfl) (fun u _ => rfl)
    (fun u _ => wi_hasDerivAt_landscape u)
    (fun u _ => wi_hasDerivAt_quasiDir u)
    (fun u _ => wi_hasDerivAt_quasiLift u)
    (wi_contScore.intervalIntegrable_of_Icc (by norm_num [wiBasin, wiSaddle]))

/-- The conclusion of heavyweight (iii) at explicit numbers: the lift's barrier exponent is
`16/9`, the direct baseline's is `80/3`. -/
theorem wi_barrier_exponents_eq :
    (2 / wiEta) * (wiQuasiLift wiSaddle - wiQuasiLift wiBasin) = 16 / 9 ∧
      (2 / wiEta) * (wiQuasiDir wiSaddle - wiQuasiDir wiBasin) = 80 / 3 := by
  constructor
  · norm_num [wiEta, wiQuasiLift, wiLandscape, wiSaddle, wiBasin]
  · norm_num [wiEta, wiQuasiDir, wiLandscape, wiSaddle, wiBasin]

/-- The same two numbers read off `BarrierRescaling.arrheniusExponent`, which is how Corollary 1
writes them, together with the strict ordering. -/
theorem wi_arrhenius_exponents_eq :
    arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta (wiSigma2Dir 0)
        = 80 / 3 ∧
      arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta (wiSigma2Lift 0)
        = 16 / 9 := by
  rw [wi_barrier_action, wi_sigma2Dir_eq, wi_sigma2Lift_eq]
  constructor <;> norm_num [arrheniusExponent, wiEta]

/-- **Both diffusions lie strictly inside the metastable window of (A5).** The Kramers exponent
exceeds one for the direct baseline (`80/3`) and, crucially, still exceeds one for the lift
(`16/9`): the lift accelerates escape without leaving the regime in which the escape problem is
posed. A witness in which the lift had left the window would satisfy the theorems' hypotheses
but would not be a witness for the paper's setting. -/
theorem wi_both_metastable :
    1 < arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta (wiSigma2Dir 0) ∧
      1 < arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta
            (wiSigma2Lift 0) := by
  rw [wi_arrhenius_exponents_eq.1, wi_arrhenius_exponents_eq.2]
  norm_num

/-- **The metastable window of (A5) in its direct form.** Both Itô diffusion coefficients lie
strictly below `2α = 8/3`: `η σ²_dir = 1/10` and `η σ²_lift = 3/2`. This is the same statement as
`wi_both_metastable`, written as a bound on the diffusion rather than on the exponent. -/
theorem wi_metastable_window :
    wiSdeDir < 2 * (wiLandscape wiSaddle - wiLandscape wiBasin) ∧
      wiSdeLift < 2 * (wiLandscape wiSaddle - wiLandscape wiBasin) := by
  rw [wi_barrier_action, wi_sdeDir_eq, wi_sdeLift_eq]
  constructor <;> norm_num

/-- The two modules' conventions agree at the instance: the Arrhenius exponent
`2α/(η σ_eff²)` of `BarrierRescaling` is the exponent `2α/σ²` of `KramersExitTime` evaluated at
the Itô diffusion coefficient `σ² = η σ_eff²`. This is the seam between the two halves of the
chain, and it is checked rather than assumed. -/
theorem wi_kramers_exponent_agree :
    2 * (wiLandscape wiSaddle - wiLandscape wiBasin) / wiSdeDir
        = arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta (wiSigma2Dir 0) ∧
      2 * (wiLandscape wiSaddle - wiLandscape wiBasin) / wiSdeLift
        = arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta
            (wiSigma2Lift 0) := by
  rw [wi_barrier_action, wi_sdeDir_eq, wi_sdeLift_eq, wi_sigma2Dir_eq, wi_sigma2Lift_eq]
  constructor <;> norm_num [arrheniusExponent, wiEta]

/-- **Heavyweight (iv): `KramersExitTime.meanExitTime_upper_bound` at the instance.** The lift's
mean exit time from the basin, started at the basin bottom, is at most the explicit number
`10 exp(16/9)`. -/
theorem wi_exit_time_upper_bound :
    meanExitTime wiSdeLift wiLandscape wiLeft wiRight wiStart ≤ 10 * Real.exp (16 / 9) := by
  have h := meanExitTime_upper_bound (sigma2 := wiSdeLift) (M := wiLandscape wiSaddle)
    (m := wiLandscape wiBasin) (a := wiLeft) (b := wiRight) (x := wiStart)
    (by rw [wi_sdeLift_eq]; norm_num) (by norm_num [wiLeft, wiStart])
    (by norm_num [wiRight, wiStart]) wi_contLandscape wi_max wi_min
  refine h.trans (le_of_eq ?_)
  rw [wi_barrier_action, wi_sdeLift_eq]
  norm_num [wiLeft, wiRight, wiStart]

/-- The matching Arrhenius lower bound at the direct baseline's diffusion: with the Laplace
tolerance `δ = 1/3`, the direct exit time is at least a positive constant times `exp 20`. -/
theorem wi_exit_time_lower_bound :
    ∃ c > 0, c * Real.exp 20 ≤ meanExitTime wiSdeDir wiLandscape wiLeft wiRight wiStart := by
  obtain ⟨c, hc, hbound⟩ := meanExitTime_arrhenius_lower_bound (sigma2 := wiSdeDir)
    (U := wiLandscape) (a := wiLeft) (b := wiRight) (x := wiStart)
    (wb := wiBasin) (ws := wiSaddle) (delta := 1 / 3)
    (by rw [wi_sdeDir_eq]; norm_num) wi_contLandscape (by norm_num [wiLeft, wiStart])
    (by norm_num [wiLeft, wiBasin]) (by norm_num [wiBasin, wiSaddle])
    (by norm_num [wiSaddle, wiRight]) (by norm_num [wiStart, wiSaddle]) (by norm_num)
  refine ⟨c, hc, ?_⟩
  have hexp : 2 * (wiLandscape wiSaddle - wiLandscape wiBasin - 1 / 3) / wiSdeDir = 20 := by
    rw [wi_barrier_action, wi_sdeDir_eq]
    norm_num
  rwa [hexp] at hbound

/-- **The exit-time ordering in the small-noise regime.** With the lift's diffusion held fixed at
`3/2`, every sufficiently small direct diffusion gives a strictly larger direct exit time. No
relation between the two Arrhenius prefactors is assumed. -/
theorem wi_eventually_exit_ordering :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      meanExitTime wiSdeLift wiLandscape wiLeft wiRight wiStart
        < meanExitTime sd wiLandscape wiLeft wiRight wiStart :=
  eventually_meanExitTime_lift_lt_direct (sl := wiSdeLift) (wb := wiBasin) (ws := wiSaddle)
    (by rw [wi_sdeLift_eq]; norm_num) wi_contLandscape (by norm_num [wiLeft, wiStart])
    (by norm_num [wiLeft, wiBasin]) (by norm_num [wiBasin, wiSaddle])
    (by norm_num [wiSaddle, wiRight]) (by norm_num [wiStart, wiSaddle])
    (by norm_num [wiLandscape, wiBasin, wiSaddle]) wi_max wi_min

/-- **The exit-time ordering for the paper's own additively coupled pair.** Equation (5) compares
`σ²` with `σ² + σ_Jac²`, not two free-floating diffusions; here `σ_Jac²` enters at the instance's
own value `η σ_Jac² = 7/5`, and for every sufficiently small direct diffusion the lifted exit
time is strictly the smaller. -/
theorem wi_eventually_exit_ordering_additive :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      meanExitTime (sd + wiSdeJac) wiLandscape wiLeft wiRight wiStart
        < meanExitTime sd wiLandscape wiLeft wiRight wiStart :=
  eventually_meanExitTime_additive_lift_lt_direct (jac2 := wiSdeJac)
    (wb := wiBasin) (ws := wiSaddle)
    (by rw [wi_sde_additive.1]; norm_num) wi_contLandscape (by norm_num [wiLeft, wiStart])
    (by norm_num [wiLeft, wiBasin]) (by norm_num [wiBasin, wiSaddle])
    (by norm_num [wiSaddle, wiRight]) (by norm_num [wiStart, wiSaddle])
    (by norm_num [wiLandscape, wiBasin, wiSaddle]) wi_max wi_min

/-- **Heavyweight (v): the paper-facing Dynkin form,
`KramersExitTime.mean_first_passage_ordering_of_dynkin`, at the instance.** The two
`MeanExitTimeBVP` hypotheses are discharged by `meanExitTime_satisfies_bvp`, that is, by
exhibiting a solution of the boundary-value problem; this establishes satisfiability of the
hypothesis stack, and it is not, and does not pretend to be, a derivation of Dynkin's formula
from an Itô calculus this mathlib does not have. -/
theorem wi_mean_first_passage_ordering :
    ∃ c > 0,
      c * Real.exp 20 ≤ meanExitTime wiSdeDir wiLandscape wiLeft wiRight wiStart ∧
        (2 / wiSdeLift * ((wiRight - wiStart) * (wiRight - wiLeft))
              * Real.exp (2 * (wiLandscape wiSaddle - wiLandscape wiBasin) / wiSdeLift)
            < c * Real.exp 20 →
          meanExitTime wiSdeLift wiLandscape wiLeft wiRight wiStart
            < meanExitTime wiSdeDir wiLandscape wiLeft wiRight wiStart) := by
  have hlt : wiLeft < wiRight := by norm_num [wiLeft, wiRight]
  have hsd : (0 : ℝ) < wiSdeDir := by rw [wi_sdeDir_eq]; norm_num
  have hsl : (0 : ℝ) < wiSdeLift := by rw [wi_sdeLift_eq]; norm_num
  have hU' : ∀ y ∈ Ioo wiLeft wiRight, HasDerivAt wiLandscape (wiScore y) y :=
    fun y _ => wi_hasDerivAt_landscape y
  obtain ⟨c, hc, hlow, hgap⟩ := mean_first_passage_ordering_of_dynkin
    (U := wiLandscape) (U' := wiScore) (a := wiLeft) (b := wiRight) (x := wiStart)
    (wb := wiBasin) (ws := wiSaddle) (delta := 1 / 3) (sd := wiSdeDir) (sl := wiSdeLift)
    (meanExitTime wiSdeDir wiLandscape wiLeft wiRight)
    (meanExitTimeDeriv wiSdeDir wiLandscape wiLeft)
    (meanExitTimeDeriv2 wiSdeDir wiLandscape wiScore wiLeft)
    (meanExitTime wiSdeLift wiLandscape wiLeft wiRight)
    (meanExitTimeDeriv wiSdeLift wiLandscape wiLeft)
    (meanExitTimeDeriv2 wiSdeLift wiLandscape wiScore wiLeft)
    (meanExitTime_satisfies_bvp hsd hlt wi_contLandscape hU')
    (meanExitTime_satisfies_bvp hsl hlt wi_contLandscape hU')
    (continuousOn_meanExitTime wiSdeDir hlt.le wi_contLandscape)
    (continuousOn_meanExitTime wiSdeLift hlt.le wi_contLandscape)
    hsd hsl wi_contLandscape hU' (by norm_num [wiLeft, wiStart])
    (by norm_num [wiLeft, wiBasin]) (by norm_num [wiBasin, wiSaddle])
    (by norm_num [wiSaddle, wiRight]) (by norm_num [wiStart, wiSaddle]) (by norm_num)
    wi_max wi_min
  have hexp : 2 * (wiLandscape wiSaddle - wiLandscape wiBasin - 1 / 3) / wiSdeDir = 20 := by
    rw [wi_barrier_action, wi_sdeDir_eq]
    norm_num
  rw [hexp] at hlow hgap
  exact ⟨c, hc, hlow, hgap⟩

/-- **Heavyweight (vi): `StationaryDensity.stationaryDensity_zeroFlux` at the instance.** The
density `π = (C/σ_eff²) exp(-(2/η) U_eff)` of equation `(*)` solves the zero-flux stationary
Fokker–Planck equation for the drift `-L̃'` and the lift's diffusion. -/
theorem wi_stationary_density :
    IsZeroFluxStationary (fun u => -wiScore u) (fokkerPlanckDiffusion wiEta wiSigma2Lift)
      (stationaryDensity 1 wiEta wiSigma2Lift wiQuasiLift) :=
  stationaryDensity_zeroFlux (by norm_num [wiEta])
    (fun w => by rw [wi_sigma2Lift_eq]; norm_num) wi_hasDerivAt_quasiLift

/-- The stationary density of the instance is strictly positive everywhere, so it is a density
and not a degenerate object. -/
theorem wi_stationary_density_pos (u : ℝ) :
    0 < stationaryDensity 1 wiEta wiSigma2Lift wiQuasiLift u :=
  stationaryDensity_pos one_pos (fun w => by rw [wi_sigma2Lift_eq]; norm_num) u

/-! ## The kill-shot

Everything above, in one statement. The existential is deliberately long: a referee reading it
sees the hypothesis stack of the main chain written out, and the kernel has checked that one
tuple satisfies all of it at once. Putting the certificate in the statement rather than in the
proof is what makes it robust to a later edit — a proof can be weakened silently, a statement
cannot. -/

/-- **The hypothesis stack of the main chain is inhabited.**

There is one model — an explicit four-point sample space in dimension two, an explicit symmetric
positive definite mean Hessian, an explicit nonzero remainder, an explicit softplus-derived
positivity-map slope, an explicit cubic single-barrier landscape and an explicit small learning rate —
that simultaneously satisfies

* the latent weight model of (A2)–(A4): non-negative weights, a centred latent-weight fluctuation, a
  positive definite jitter covariance, a symmetric positive definite mean Hessian, and the chain
  `δg = H̄ δθ + r` holding pointwise with `r ≠ 0`;
* assumption (A6), uniformly in the direction, with `κ = 7/3`, **and** the remainder control at
  level `ρ = 1` from which `CouplingSign` derives the same sign without (A6);
* the diffusion layer of the corrected §2, with the projected and the scalar accountings
  agreeing and the ordering `0 < σ²_dir < σ²_lift` strict;
* the regularity of (A1) and the single-barrier geometry of (A5), with a positive barrier action
  monotonically climbed;

and on which the chain then delivers, as theorems and not as hypotheses, the strict increase of
the projected diffusion, the strict decrease of the quasipotential barrier exponent, the strict
decrease of the Arrhenius exponent with **both** diffusions still inside the metastable window,
an explicit Arrhenius ceiling on the lift's mean exit time, and the zero-flux stationarity of the
density `(*)`.

The instance's numbers are pinned inside the statement: `α = 4/3`, `κ = 7/3`, `σ²_dir = 1`,
`σ²_lift = 15`. No theorem in this development is therefore vacuously true for want of a model. -/
theorem hypothesis_stack_is_inhabited :
    ∃ (w : Fin 4 → ℝ) (dth dg r : Fin 4 → Fin 2 → ℝ)
      (H V S Sobj : Matrix (Fin 2) (Fin 2) ℝ) (e : Fin 2 → ℝ)
      (s sigmaObj eta kappa alpha : ℝ)
      (Ltilde dL UeffDir UeffLift sigma2Dir sigma2Lift : ℝ → ℝ)
      (a b x wb ws : ℝ),
      (∀ ω, 0 ≤ w ω) ∧ (∀ i, ∑ ω, w ω * dth ω i = 0) ∧
      outerMoment w dth dth = V ∧ V.PosDef ∧ H.IsHermitian ∧ H.PosDef ∧
      (∀ ω, dg ω = H *ᵥ dth ω + r ω) ∧ r ≠ 0 ∧ outerMoment w dth dg = S ∧
      0 < kappa ∧ kappa = 7 / 3 ∧
      PositiveAlignmentOn (Set.univ : Set Unit) (fun _ => H) (fun _ => S) kappa ∧
      RemainderControlledBy H V (outerMoment w dth r) 1 ∧
      0 < 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)) ∧
      projDiffusionDirect s Sobj e = sigma2Dir 0 ∧
      projDiffusionLift s Sobj V H S 0 e = sigma2Lift 0 ∧
      (∀ u, sigma2Dir u = directDiffusion s sigmaObj) ∧
      (∀ u, sigma2Lift u = liftDiffusion s sigmaObj (s ^ 2 * crossTerm H S e)
        (s ^ 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)))) ∧
      sigma2Dir 0 = 1 ∧ sigma2Lift 0 = 15 ∧
      0 < sigma2Dir 0 ∧ sigma2Dir 0 < sigma2Lift 0 ∧
      0 < eta ∧ a < wb ∧ wb < ws ∧ ws < b ∧ a ≤ x ∧ x < ws ∧
      alpha = Ltilde ws - Ltilde wb ∧ alpha = 4 / 3 ∧ 0 < alpha ∧
      ContinuousOn Ltilde (Icc a b) ∧ (∀ u, HasDerivAt Ltilde (dL u) u) ∧
      (∀ u ∈ Icc wb ws, 0 ≤ dL u) ∧
      (∀ y ∈ Icc a b, Ltilde y ≤ Ltilde ws) ∧ (∀ z ∈ Icc a b, Ltilde wb ≤ Ltilde z) ∧
      (∀ u, HasDerivAt UeffDir (dL u / sigma2Dir u) u) ∧
      (∀ u, HasDerivAt UeffLift (dL u / sigma2Lift u) u) ∧
      projDiffusionDirect s Sobj e < projDiffusionLift s Sobj V H S 0 e ∧
      2 / eta * (UeffLift ws - UeffLift wb) < 2 / eta * (UeffDir ws - UeffDir wb) ∧
      arrheniusExponent alpha eta (sigma2Lift 0) < arrheniusExponent alpha eta (sigma2Dir 0) ∧
      1 < arrheniusExponent alpha eta (sigma2Lift 0) ∧
      meanExitTime (eta * sigma2Lift 0) Ltilde a b x ≤ 10 * Real.exp (16 / 9) ∧
      IsZeroFluxStationary (fun u => -dL u) (fokkerPlanckDiffusion eta sigma2Lift)
        (stationaryDensity 1 eta sigma2Lift UeffLift) := by
  have hexp : arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta
      (wiSigma2Lift 0)
        < arrheniusExponent (wiLandscape wiSaddle - wiLandscape wiBasin) wiEta
            (wiSigma2Dir 0) := by
    rw [wi_arrhenius_exponents_eq.1, wi_arrhenius_exponents_eq.2]
    norm_num
  exact ⟨wiWeight, wiJitter, wiGradFluct, wiRemainder, wiHessian, wiCov, wiSlack, wiObjNoise,
    wiDirection, wiSlope, wiSigmaObj, wiEta, 7 / 3,
    wiLandscape wiSaddle - wiLandscape wiBasin,
    wiLandscape, wiScore, wiQuasiDir, wiQuasiLift, wiSigma2Dir, wiSigma2Lift,
    wiLeft, wiRight, wiStart, wiBasin, wiSaddle,
    fun _ => by norm_num [wiWeight],
    wi_jitter_mean_zero, wi_V_eq, wi_cov_posDef, wi_H_isHermitian, wi_H_posDef,
    wi_chain, wi_remainder_ne_zero, wi_S_eq,
    by norm_num, rfl, wi_positiveAlignment, wi_R_eq ▸ wi_remainderControlled,
    by rw [wi_leading_pos]; norm_num,
    wi_projDiffusionDirect_eq, wi_projDiffusionLift_eq, fun _ => rfl, fun _ => rfl,
    wi_sigma2Dir_eq 0, wi_sigma2Lift_eq 0,
    wi_diffusion_ordering.1, wi_diffusion_ordering.2,
    by norm_num [wiEta],
    by norm_num [wiLeft, wiBasin], by norm_num [wiBasin, wiSaddle],
    by norm_num [wiSaddle, wiRight], by norm_num [wiLeft, wiStart],
    by norm_num [wiStart, wiSaddle],
    rfl, wi_barrier_action, by rw [wi_barrier_action]; norm_num,
    wi_contLandscape, wi_hasDerivAt_landscape, wi_grad_nonneg, wi_max, wi_min,
    wi_hasDerivAt_quasiDir, wi_hasDerivAt_quasiLift,
    wi_projDiffusion_strict_increase, wi_barrier_exponent_lift_lt_direct,
    hexp, wi_both_metastable.2, wi_exit_time_upper_bound, wi_stationary_density⟩

end IcnnLift
