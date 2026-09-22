import Mathlib
import TraceLemmas
import UpdateCovariance
import CouplingSign
import DiffusionOrdering
import MinibatchNoise
import ShoulderAttenuation
import PowerOfS

/-!
# The power of the positivity-map prefactor in the coupling term of `eq:sigma-eff`

The v5 manuscript (`docs/paper/v5/paper_v5.tex`, Section 4.2) states the bias-channel effective
diffusion of the lifted latent iterate as

  `σ²_eff = σ_s² σ²_obj + 2 e_bᵀ H_θ̃ Σ_ξ e_b + e_bᵀ H_θ̃ V H_θ̃ e_b + ϱ`   (`eq:sigma-eff`),

with `H_θ̃ = E_X[∇²_θ̃ L]` the mean latent-weight Hessian, `Σ_ξ = E_X[δθ̃ ξᵀ]` the frozen coupling of
the latent-weight fluctuation with the minibatch noise at the reference iterate, and `V = E_X[δθ̃ δθ̃ᵀ]`.
It then says, beside Corollary 1, that the direct-softplus exponent is of order `σ_s⁻²` while the
lift's is of order `σ_s⁻¹` "where the coupling keeps its single power of `σ_s`", citing
`PowerOfS.liftExponent_isTheta_inv`. That theorem is about the additive form
`s² σ_obj² + σ_Jac²`, in which the once-attenuated trace `σ_Jac² = Θ(s)` is added to the
twice-attenuated objective term; the manuscript itself retired that form for adding
incommensurable units, and `eq:sigma-eff` is its replacement. This file asks the question the
retirement leaves open: **what power of the prefactor does the coupling term of `eq:sigma-eff`
actually carry?** The answer is two, not one, and it changes the order claimed for the lift.

**The computation.** The lifted iterate reaches the loss through the coordinatewise positivity map
`θ = ψ(θ̃)`, so the mean latent-weight gradient map is `w ↦ ψ'(w) ⊙ ḡ(ψ(w))` and its Jacobian at the
reference iterate — the paper's `H_θ̃` — is, by the product rule,

  `H_θ̃ = diag(ψ') Hθ diag(ψ') + diag(ψ'' ⊙ ḡ)`,

with `Hθ` the mean constrained Hessian and `ḡ` the mean constrained gradient
(`hasFDerivAt_preReadoutGrad`, `fderiv_preReadoutGrad_apply`; this is calculus, not a
stipulation). The latent-weight minibatch noise at the reference iterate is `ξ = ψ'(θ̃_ref) ⊙ ξ_θ`
with the positivity-map derivative deterministic there, so `Σ_ξ = Ξ_θ diag(ψ')` with
`Ξ_θ = E_X[δθ̃ ξ_θᵀ]` the constrained-unit coupling: attenuated once, exactly as `PowerOfS.lean`
counts (`covMat_hadamard_right`). Contracting the two under (A2) read exactly, `ψ'(θ̃_ref) = s`
and `ψ''(θ̃_ref) = c₂` in every coordinate, gives

  `2 e_bᵀ H_θ̃ Σ_ξ e_b = 2 s³ · e_bᵀ Hθ Ξ_θ e_b + 2 s c₂ · e_bᵀ diag(ḡ) Ξ_θ e_b`

(`couplingTerm_single_prefactor`). For the softplus positivity map `ψ'' = ψ'(1 − ψ')`, proved from
the derivative of the logistic function (`hasDerivAt_logistic`), so `c₂ = s(1 − s)` and the
coupling term is `2 s³ a + 2 s² (1 − s) b` with `a = e_bᵀ Hθ Ξ_θ e_b` the curvature-weighted and
`b = e_bᵀ diag(ḡ) Ξ_θ e_b` the mean-gradient-weighted frozen coupling; for a coordinate direction
`b = ḡ_b (Ξ_θ)_{bb}` (`couplingTerm_softplus`, `gradCoupling_single`). The single power of `s`
that `Σ_ξ` carries is multiplied by at least one further power from the Hessian, because every
term of the latent-weight Hessian is attenuated at least once. The coupling term is therefore
`O(s²)` always, `Θ(s²)` when `b ≠ 0` — the mean constrained gradient at the bias coordinate is
nonzero, which is what being on the shoulder means, and the frozen coupling is nonzero there —
`Θ(s³)` when `b = 0 ≠ a`, and never `Θ(s)` (`couplingPoly_isBigO_sq`,
`couplingPoly_isTheta_sq_of_ne`, `couplingPoly_isTheta_cube_of_eq`,
`couplingPoly_not_isTheta_id`). The relative reading of (A2), `|ψ'(θ̃_ref)_i| ≤ (1 + ε) s`, with
`|ψ''(θ̃_ref)_i| ≤ c s`, leaves the `O(s²)` bound and the negative conclusion intact
(`couplingTerm_isBigO_sq_of_relative`, `couplingTerm_not_isTheta_id_of_relative`), with the
softplus positivity map satisfying the second-derivative bound at `c = 1`
(`softplus_second_derivative_bound`).

**The whole of `eq:sigma-eff`.** The jitter term `e_bᵀ H_θ̃ V H_θ̃ e_b` is likewise a polynomial
whose lowest term is `s² (1 − s)² e_bᵀ diag(ḡ) V diag(ḡ) e_b` (`jitterTerm_softplus`), and the
three remainder scalars are `s²` times constrained-unit scalars once the remainder is attenuated
once, as the chain rule makes it (`remainderScalars_once_attenuated`,
`abs_remainderScalars_le_of_once_attenuated`). Substituting all of this into
`MinibatchNoise.projDiffusion_lift_eq_frozen_form` at positivity-map slope one — the convention under
which that identity is `eq:sigma-eff` — gives the projected diffusion of the lift as an exact
polynomial in the prefactor plus the remainder (`projDiffusion_lift_softplus_form`):

  `σ²_eff(s) = s² (σ_obj² + 2 b + j₀) + s³ R(s) + ϱ(s)`,

with `R` bounded on the unit interval. When the leading coefficient `c₂ = σ_obj² + 2 b + j₀`
exceeds the remainder constant `τ` of `|ϱ(s)| ≤ τ s²`, the diffusion is `Θ(s²)`
(`sigmaEffPoly_isTheta_sq`) and the exponent `2α/σ²_eff` is `Θ(s⁻²)`
(`liftExponent_sigmaEff_isTheta_inv_sq`) — the order of the direct-softplus exponent
(`directExponent_isTheta_inv_sq'`). The ratio of the two exponents is eventually bounded between
two explicit constants (`exponentRatio_sigmaEff_eventually_bounded`), so it does not diverge
(`exponentRatio_sigmaEff_not_tendsto_atTop`), which is the negation on the paper's own
diffusion of the conclusion of `PowerOfS.exponentRatio_tendsto_atTop`; where `ϱ(s)/s²` has a
limit `ϱ₀` the ratio converges to `(c₂ + ϱ₀)/σ_obj²` (`exponentRatio_sigmaEff_tendsto`). The
leading coefficient is itself a variance, that of the projected once-attenuated gradient
fluctuation `e_bᵀ ξ_θ + (ḡ ⊙ e_b)ᵀ δθ̃` (`leadingCoeff_eq_projDiffusion`), and its excess over the
direct value is `2 b + j₀` with `j₀ ≥ 0`, so a nonnegative `b` — (A6) read at leading order in
the prefactor — makes the lift's constant at least the direct one
(`leadingCoeff_ge_direct_of_gradCoupling_nonneg`).

**What this decides for the paper.** On `eq:sigma-eff`, under the chain rule, (A2) and the refined
chain of (A4), the lift lowers the shoulder exponent by a bounded factor and not by an order: both
constructions have exponent `Θ(σ_s⁻²)` with the barrier held fixed (and both `Θ(σ_s⁻¹)` if the
barrier is instead read in latent-weight units, since the barrier convention is common to the two).
The sentence "the lift's exponent is of order `σ_s⁻¹` where the coupling keeps its single power of
`σ_s`" is not available on `eq:sigma-eff`; the mechanism at leading order is the positivity map's second
derivative multiplying the mean constrained gradient, `2 s² ḡ_b (Ξ_θ)_{bb}`, and the curvature
route `2 s³ e_bᵀ Hθ Ξ_θ e_b` enters one order later. The constants are explicit throughout, and
one-dimensional witnesses show that none of the headline statements is vacuous
(`couplingTerm_witness`, `couplingPoly_witness_isTheta_sq`, `liftExponent_witness_isTheta_inv_sq`,
`exponentRatio_witness_tendsto`).

## Results
* `preHessian`, `preReadoutGrad`, `preReadoutJacobian`, `hasFDerivAt_preReadoutGrad`,
  `preReadoutJacobian_apply`, `fderiv_preReadoutGrad_apply` — the latent-weight Hessian is
  `diag(ψ') Hθ diag(ψ') + diag(ψ'' ⊙ ḡ)`, by the product rule.
* `covMat_hadamard_right`, `covMat_hadamard_self` — the frozen coupling is attenuated once, the
  objective contribution twice.
* `preHessian_transpose`, `preHessian_single` — symmetry, and the single-prefactor form
  `s² Hθ + c₂ diag(ḡ)`.
* `curvCoupling`, `gradCoupling`, `couplingTerm_single_prefactor` — the coupling term is
  `2 s³ a + 2 s c₂ b`.
* `hasDerivAt_logistic`, `softplus_second_derivative`, `couplingTerm_softplus`,
  `couplingTerm_exp` — `ψ'' = ψ'(1 − ψ')` for softplus, `ψ'' = ψ'` for the exponential positivity map,
  and the coupling term for each.
* `couplingPoly`, `abs_couplingPoly_le`, `couplingPoly_isBigO_sq`,
  `couplingPoly_div_id_tendsto_zero`, `couplingPoly_div_direct_tendsto`,
  `not_isTheta_id_of_isBigO_sq`, `couplingPoly_not_isTheta_id`, `couplingPoly_isTheta_sq_of_ne`,
  `couplingPoly_isTheta_cube_of_eq` — the prefactor asymptotics of the coupling term, and the
  bounded coupling ratio.
* `jitterPoly`, `jitterTerm_softplus`, `remainderScalars`, `projDiffusion_lift_softplus_form`,
  `remainderScalars_once_attenuated`, `abs_remainderScalars_le_of_once_attenuated` — the whole of
  `eq:sigma-eff` in powers of the prefactor.
* `sigmaEffPoly`, `leadingCoeff`, `higherOrderBound`, `sigmaEffPoly_eq`, `abs_cubicCoeff_le`,
  `sigmaEffPoly_two_sided`, `sigmaEffPoly_two_sided_near`, `sigmaEffPoly_isTheta_sq` — the
  diffusion is `Θ(s²)`.
* `shoulderExponent_isTheta_inv_sq_of_two_sided`, `liftExponent_sigmaEff_isTheta_inv_sq`,
  `directExponent_isTheta_inv_sq'` — both exponents are `Θ(s⁻²)`.
* `exponentRatio_sigmaEff_eq`, `exponentRatio_sigmaEff_eventually_bounded`,
  `exponentRatio_sigmaEff_not_tendsto_atTop`, `exponentRatio_sigmaEff_tendsto` — the ratio of
  exponents is bounded, does not diverge, and converges to `(c₂ + ϱ₀)/σ_obj²`.
* `leadingCoeff_eq_projDiffusion`, `leadingCoeff_nonneg`, `leadingJitter_nonneg`,
  `leadingCoeff_ge_direct_of_gradCoupling_nonneg`, `gradCoupling_single` — the leading
  coefficient is a variance, and (A6) at leading order.
* `couplingTerm_witness`, `gradCoupling_witness_ne_zero`, `couplingPoly_witness_isTheta_sq`,
  `liftExponent_witness_isTheta_inv_sq`, `exponentRatio_witness_tendsto` — non-vacuity.
* `abs_dotProduct_mulVec_le`, `abs_mulVec_apply_le`, `curvAbsSum`, `gradAbsSum`,
  `crossTerm_preHessian_eq`, `abs_crossTerm_preHessian_le`, `curvAbsSum_nonneg`,
  `gradAbsSum_nonneg`, `couplingTerm_isBigO_sq_of_relative`,
  `couplingTerm_not_isTheta_id_of_relative`, `softplus_second_derivative_bound` — the same
  conclusions under (A2) read with a tolerance.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The hypotheses this file carries, and what each would take to discharge. The chain rule for the
latent-weight Hessian is proved from differentiability of the positivity map and of the mean constrained
gradient map, and is not a hypothesis. The single-prefactor reading of (A2) enters the exact
identities as the equality `ψ'(θ̃_ref) = s`; the asymptotic conclusions are re-derived under the
relative tolerance `|ψ'(θ̃_ref)_i| ≤ (1 + ε) s` in Section 9, which is the reading
`PowerOfS.lean` and `SoftplusBand.lean` argue is the defensible one. The second-derivative bound
`|ψ''(θ̃_ref)_i| ≤ c s` is proved for the softplus positivity map with `c = 1` and holds for the
exponential positivity map with `c = 1` by `ψ'' = ψ'`; it is a hypothesis for a general positivity map. The
remainder bound `|ϱ(s)| ≤ τ s²` is (A4′-r) read at the positivity-map scale, and
`abs_remainderScalars_le_of_once_attenuated` derives it from the once-attenuated form of the
remainder with an explicit `τ`; what is not proved is that the constrained-unit remainder scalars
are bounded uniformly in `s`, a moment condition on the batch law. The gap `τ < c₂` is the
(A6)-type condition under which the leading coefficient is not cancelled by the remainder, and
`leadingCoeff_ge_direct_of_gradCoupling_nonneg` shows it is implied by `τ < σ_obj²` together with
`b ≥ 0`. The barrier `α` is held fixed as the prefactor varies, as in `PowerOfS.lean`; a barrier
read in latent-weight units would scale both exponents alike and leave every comparison here
unchanged. Nothing in this file touches escape times, stochastic differential equations or the
Adam normalization: `shoulderExponent` is the Arrhenius exponent as a function of the prefactor,
and the statements are about that function.
-/

open Matrix Finset MeasureTheory ProbabilityTheory Filter Topology Asymptotics
open scoped MatrixOrder

namespace IcnnLift

variable {d : ℕ}

/-! ## 1. The latent-weight Hessian by the chain rule

The lifted iterate reaches the loss through the coordinatewise positivity map `θ = ψ(θ̃)`, so the
gradient with respect to the latent weight is `∇_θ̃ L = ψ'(θ̃) ⊙ ∇_θ L` (`eq:chain`, `ShoulderAttenuation.lean`) and the
latent-weight Hessian is the Jacobian of that map. The section computes it from calculus rather than
positing it: the mean latent-weight gradient map is `w ↦ ψ'(w) ⊙ ḡ(ψ(w))`, with `ḡ` the mean
constrained gradient map, and its Jacobian at the reference iterate is
`diag(ψ') Hθ diag(ψ') + diag(ψ'' ⊙ ḡ)`. -/

section ChainRule

/-- **The latent-weight Hessian** `H_θ̃ = E_X[∇²_θ̃ L]` at the reference iterate, written by the
chain rule through the coordinatewise positivity map: `p = ψ'(θ̃_ref)` and `p₂ = ψ''(θ̃_ref)` are the
positivity map's first and second derivatives there, `gbar` is the mean constrained gradient
`E_X[∇_θ L]` and `Hθ` the mean constrained Hessian `E_X[∇²_θ L]`, and

  `H_θ̃ = diag(p) Hθ diag(p) + diag(p₂ ⊙ gbar)`.

The second summand is the term the additive form `s² σ_obj² + σ_Jac²` never sees. -/
def preHessian (p p₂ gbar : Fin d → ℝ) (Hθ : Matrix (Fin d) (Fin d) ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  diagonal p * Hθ * diagonal p + diagonal (fun i => p₂ i * gbar i)

/-- The mean latent-weight gradient map `w ↦ ψ'(w) ⊙ ḡ(ψ(w))`, whose Jacobian is the Hessian
in the latent weights. Here `gradL` is the mean constrained gradient map on the constrained coordinate. -/
def preReadoutGrad (psi p : ℝ → ℝ) (gradL : (Fin d → ℝ) → (Fin d → ℝ)) (w : Fin d → ℝ) :
    Fin d → ℝ :=
  fun i => p (w i) * gradL (fun j => psi (w j)) i

/-- The Jacobian of the mean latent-weight gradient map, as a continuous linear map, in the form
the product rule produces: coordinate `i` is `ψ'(w_i)` times the `i`-th row of the constrained
Jacobian composed with the Hadamard map `v ↦ ψ'(w) ⊙ v`, plus `ḡ_i ψ''(w_i)` times the `i`-th
coordinate projection. -/
noncomputable def preReadoutJacobian (psi p p₂ : ℝ → ℝ) (gradL : (Fin d → ℝ) → (Fin d → ℝ))
    (HL : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ)) (w : Fin d → ℝ) : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ) :=
  ContinuousLinearMap.pi fun i =>
    p (w i) • ((ContinuousLinearMap.proj i).comp (HL.comp (hadamardCLM fun j => p (w j))))
      + gradL (fun j => psi (w j)) i • (p₂ (w i) • ContinuousLinearMap.proj i)

/-- **The chain rule for the latent-weight Hessian, as a differentiability statement.** If the
positivity map has first derivative `p` and second derivative `p₂` at every coordinate of `w`, and the
mean constrained gradient map is differentiable at `ψ(w)` with Jacobian `HL`, then the mean
latent-weight gradient map is differentiable at `w` with Jacobian `preReadoutJacobian`. This is the
product rule applied coordinatewise to `ψ'(w_i) · ḡ_i(ψ(w))`. -/
theorem hasFDerivAt_preReadoutGrad {psi p p₂ : ℝ → ℝ} {gradL : (Fin d → ℝ) → (Fin d → ℝ)}
    {HL : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ)} (w : Fin d → ℝ)
    (hpsi : ∀ i, HasDerivAt psi (p (w i)) (w i))
    (hp : ∀ i, HasDerivAt p (p₂ (w i)) (w i))
    (hL : HasFDerivAt gradL HL (fun j => psi (w j))) :
    HasFDerivAt (preReadoutGrad psi p gradL) (preReadoutJacobian psi p p₂ gradL HL w) w := by
  have hΨ : HasFDerivAt (fun u : Fin d → ℝ => fun j => psi (u j))
      (hadamardCLM fun j => p (w j)) w := by
    unfold hadamardCLM
    refine hasFDerivAt_pi.mpr fun i => ?_
    exact (hpsi i).comp_hasFDerivAt w (hasFDerivAt_apply i w)
  have hG : HasFDerivAt (fun u : Fin d → ℝ => gradL (fun j => psi (u j)))
      (HL.comp (hadamardCLM fun j => p (w j))) w := hL.comp w hΨ
  unfold preReadoutGrad preReadoutJacobian
  refine hasFDerivAt_pi.mpr fun i => ?_
  have h1 : HasFDerivAt (p ∘ fun u : Fin d → ℝ => u i)
      (p₂ (w i) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin d => ℝ) i) w :=
    (hp i).comp_hasFDerivAt w (hasFDerivAt_apply i w)
  have h2 : HasFDerivAt
      ((fun f : Fin d → ℝ => f i) ∘ fun u : Fin d → ℝ => gradL (fun j => psi (u j)))
      ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin d => ℝ) i).comp
        (HL.comp (hadamardCLM fun j => p (w j)))) w :=
    (hasFDerivAt_apply i (gradL fun j => psi (w j))).comp w hG
  exact h1.mul h2

/-- **The Jacobian is the latent-weight Hessian.** Applied to a direction `v`, the product-rule
Jacobian is `(diag(p) Hθ diag(p) + diag(p₂ ⊙ ḡ)) v`, once the constrained Jacobian `HL` is the
constrained Hessian `Hθ` acting by matrix-vector multiplication. -/
theorem preReadoutJacobian_apply {psi p p₂ : ℝ → ℝ} {gradL : (Fin d → ℝ) → (Fin d → ℝ)}
    {HL : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ)} {Hθ : Matrix (Fin d) (Fin d) ℝ}
    (hHL : ∀ v, HL v = Hθ *ᵥ v) (w v : Fin d → ℝ) :
    preReadoutJacobian psi p p₂ gradL HL w v
      = preHessian (fun i => p (w i)) (fun i => p₂ (w i)) (gradL fun j => psi (w j)) Hθ *ᵥ v := by
  funext i
  have hv : hadamardCLM (fun j => p (w j)) v = (diagonal fun j => p (w j)) *ᵥ v := by
    funext j
    simp [hadamardCLM_apply, mulVec_diagonal]
  simp only [preReadoutJacobian, preHessian, ContinuousLinearMap.pi_apply,
    _root_.add_apply, _root_.smul_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.proj_apply, hHL, smul_eq_mul, Matrix.add_mulVec, Pi.add_apply,
    ← Matrix.mulVec_mulVec, mulVec_diagonal, hv]
  ring

/-- The same statement read off the `fderiv` operator: the Jacobian of the mean latent-weight
gradient map at the reference iterate is the latent-weight Hessian. -/
theorem fderiv_preReadoutGrad_apply {psi p p₂ : ℝ → ℝ} {gradL : (Fin d → ℝ) → (Fin d → ℝ)}
    {HL : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ)} {Hθ : Matrix (Fin d) (Fin d) ℝ} (w : Fin d → ℝ)
    (hpsi : ∀ i, HasDerivAt psi (p (w i)) (w i))
    (hp : ∀ i, HasDerivAt p (p₂ (w i)) (w i))
    (hL : HasFDerivAt gradL HL (fun j => psi (w j))) (hHL : ∀ v, HL v = Hθ *ᵥ v)
    (v : Fin d → ℝ) :
    fderiv ℝ (preReadoutGrad psi p gradL) w v
      = preHessian (fun i => p (w i)) (fun i => p₂ (w i)) (gradL fun j => psi (w j)) Hθ *ᵥ v := by
  rw [(hasFDerivAt_preReadoutGrad w hpsi hp hL).fderiv, preReadoutJacobian_apply hHL]

end ChainRule

/-! ## 2. The frozen coupling in latent-weight units, and the coupling term of `eq:sigma-eff`

At the reference iterate the positivity-map derivative is deterministic, so the latent-weight minibatch
noise is `ξ = ψ'(θ̃_ref) ⊙ ξ_θ` with `ξ_θ` the constrained-coordinate noise, and the frozen coupling
`Σ_ξ = E[δθ̃ ξᵀ]` is the constrained-unit coupling `Ξ_θ = E[δθ̃ ξ_θᵀ]` times `diag(ψ')`: it is
attenuated exactly once, as `PowerOfS.lean` counts. The coupling term of `eq:sigma-eff`,
`2 e_bᵀ H_θ̃ Σ_ξ e_b`, then contracts a once-attenuated coupling with a Hessian that is itself
attenuated through the chain rule, and the section computes the product. -/

section FrozenCoupling

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **The frozen coupling is attenuated once.** With the latent-weight noise `ξ = p ⊙ ξ_θ` at the
reference iterate, `Cov[δθ̃, ξ] = Cov[δθ̃, ξ_θ] diag(p)`. The covariance centres its arguments, so
the constrained gradient may be supplied uncentred. -/
theorem covMat_hadamard_right [IsFiniteMeasure μ] (p : Fin d → ℝ) {dth g : Ω → Fin d → ℝ}
    (hθ : SqIntegrableVec μ dth) (hg : SqIntegrableVec μ g) :
    covMat μ dth (fun ω i => p i * g ω i) = covMat μ dth g * diagonal p := by
  have h : (fun ω i => p i * g ω i) = fun ω => diagonal p *ᵥ g ω := by
    funext ω i
    simp [mulVec_diagonal]
  rw [h, covMat_mulVec_right (diagonal p) hθ hg, diagonal_transpose]

/-- The objective contribution is attenuated twice: `Cov[p ⊙ ξ_θ] = diag(p) Cov[ξ_θ] diag(p)`. -/
theorem covMat_hadamard_self [IsFiniteMeasure μ] (p : Fin d → ℝ) {g : Ω → Fin d → ℝ}
    (hg : SqIntegrableVec μ g) :
    covMat μ (fun ω i => p i * g ω i) (fun ω i => p i * g ω i)
      = diagonal p * covMat μ g g * diagonal p := by
  have h : (fun ω i => p i * g ω i) = fun ω => diagonal p *ᵥ g ω := by
    funext ω i
    simp [mulVec_diagonal]
  rw [h, covMat_mulVec_left (diagonal p) hg (hg.mulVec _), covMat_mulVec_right (diagonal p) hg hg,
    diagonal_transpose, Matrix.mul_assoc]

/-- The latent-weight Hessian is symmetric whenever the constrained Hessian is. -/
theorem preHessian_transpose {p p₂ gbar : Fin d → ℝ} {Hθ : Matrix (Fin d) (Fin d) ℝ}
    (hH : Hθᵀ = Hθ) : (preHessian p p₂ gbar Hθ)ᵀ = preHessian p p₂ gbar Hθ := by
  simp [preHessian, Matrix.transpose_add, Matrix.transpose_mul, Matrix.mul_assoc, hH]

/-- **The single-prefactor form of the latent-weight Hessian.** Under (A2) read exactly,
`ψ'(θ̃_ref) = s` in every coordinate, and with `ψ''(θ̃_ref) = c₂` in every coordinate,
`H_θ̃ = s² Hθ + c₂ diag(ḡ)`. -/
theorem preHessian_single (s c₂ : ℝ) (gbar : Fin d → ℝ) (Hθ : Matrix (Fin d) (Fin d) ℝ) :
    preHessian (fun _ => s) (fun _ => c₂) gbar Hθ = s ^ 2 • Hθ + c₂ • diagonal gbar := by
  rw [preHessian, ← smul_one_eq_diagonal]
  have h : (diagonal fun i => c₂ * gbar i) = c₂ • diagonal gbar := by
    rw [← diagonal_smul]
    rfl
  rw [h, Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul, Matrix.mul_one, smul_smul, pow_two]

/-- The **curvature-weighted frozen coupling** in constrained units, `e_bᵀ Hθ Ξ_θ e_b`. -/
def curvCoupling (Hθ Ξ : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) : ℝ :=
  e ⬝ᵥ ((Hθ * Ξ) *ᵥ e)

/-- The **mean-gradient-weighted frozen coupling** in constrained units,
`e_bᵀ diag(ḡ) Ξ_θ e_b`; for a coordinate direction `e_b` it is `ḡ_b (Ξ_θ)_{bb}`, the product of the
mean constrained gradient at the bias coordinate with the frozen coupling there. -/
def gradCoupling (gbar : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) : ℝ :=
  e ⬝ᵥ ((diagonal gbar * Ξ) *ᵥ e)

/-- **The coupling term of `eq:sigma-eff` under the single-prefactor reading.** With
`H_θ̃ = s² Hθ + c₂ diag(ḡ)` and `Σ_ξ = Ξ_θ diag(s)`, the coupling term
`2 e_bᵀ H_θ̃ Σ_ξ e_b` (the cross term `crossTerm` of `CouplingSign.lean`, which is
`2 e_bᵀ H Σ e_b` for symmetric `H`) equals

  `2 s³ · e_bᵀ Hθ Ξ_θ e_b + 2 s c₂ · e_bᵀ diag(ḡ) Ξ_θ e_b`.

Three powers of the prefactor on the curvature route, and one power times the positivity map's second
derivative on the mean-gradient route; no term carries a single power on its own. -/
theorem couplingTerm_single_prefactor {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    (Ξ : Matrix (Fin d) (Fin d) ℝ) (gbar e : Fin d → ℝ) (s c₂ : ℝ) :
    crossTerm (preHessian (fun _ => s) (fun _ => c₂) gbar Hθ) (Ξ * diagonal fun _ => s) e
      = 2 * s ^ 3 * curvCoupling Hθ Ξ e + 2 * s * c₂ * gradCoupling gbar Ξ e := by
  have hsym : (preHessian (fun _ => s) (fun _ => c₂) gbar Hθ).IsHermitian :=
    isHermitian_of_transpose_eq (preHessian_transpose hH)
  rw [crossTerm_eq_two_mul hsym, preHessian_single, ← smul_one_eq_diagonal, Matrix.mul_smul,
    Matrix.mul_one, Matrix.add_mul, Matrix.mul_smul, Matrix.mul_smul, Matrix.smul_mul,
    Matrix.smul_mul, quadForm_add, quadForm_smul, quadForm_smul, quadForm_smul, quadForm_smul,
    curvCoupling, gradCoupling]
  ring

end FrozenCoupling

/-! ## 3. The positivity map's second derivative on the shoulder

For the softplus positivity map `ψ' = logistic` and `ψ'' = logistic · (1 − logistic)`, so deep on the
shoulder the second derivative is of the same order as the first: `ψ'' = s (1 − s)` where
`ψ' = s`. For the exponential positivity map `ψ'' = ψ' = s`. In both cases the mean-gradient route of
`couplingTerm_single_prefactor` carries `s · c₂ = Θ(s²)`, the same power as the direct-softplus
term `s² σ_obj²`. -/

section Softplus

/-- **`ψ'' = ψ'(1 − ψ')` for softplus**: the logistic function satisfies the logistic
differential equation. -/
theorem hasDerivAt_logistic (w : ℝ) :
    HasDerivAt logistic (logistic w * (1 - logistic w)) w := by
  have hnum : HasDerivAt (fun u : ℝ => Real.exp u) (Real.exp w) w := Real.hasDerivAt_exp w
  have hden : HasDerivAt (fun u : ℝ => 1 + Real.exp u) (Real.exp w) w := by
    simpa using (Real.hasDerivAt_exp w).const_add 1
  have h := hnum.div hden (ne_of_gt (one_add_exp_pos w))
  have hpos : (0 : ℝ) < 1 + Real.exp w := one_add_exp_pos w
  have heq : (Real.exp w * (1 + Real.exp w) - Real.exp w * Real.exp w) / (1 + Real.exp w) ^ 2
      = logistic w * (1 - logistic w) := by
    rw [logistic_def]
    field_simp
  rw [heq] at h
  exact h

/-- The softplus positivity map's second derivative, read as the derivative of the prefactor field
`ψ' = logistic` at the reference iterate: `p₂ = p (1 − p)` coordinate by coordinate. -/
theorem softplus_second_derivative (w : Fin d → ℝ) (i : Fin d) :
    HasDerivAt logistic (logistic (w i) * (1 - logistic (w i))) (w i) :=
  hasDerivAt_logistic (w i)

/-- **The coupling term for the softplus positivity map under (A2).** Where the prefactor is `s` in
every coordinate the second derivative is `s (1 − s)`, and the coupling term of `eq:sigma-eff`
is `2 s³ · e_bᵀ Hθ Ξ_θ e_b + 2 s² (1 − s) · e_bᵀ diag(ḡ) Ξ_θ e_b`. -/
theorem couplingTerm_softplus {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    (Ξ : Matrix (Fin d) (Fin d) ℝ) (gbar e : Fin d → ℝ) (s : ℝ) :
    crossTerm (preHessian (fun _ => s) (fun _ => s * (1 - s)) gbar Hθ) (Ξ * diagonal fun _ => s) e
      = 2 * s ^ 3 * curvCoupling Hθ Ξ e + 2 * s ^ 2 * (1 - s) * gradCoupling gbar Ξ e := by
  rw [couplingTerm_single_prefactor hH]
  ring

/-- **The coupling term for the exponential positivity map under (A2).** With `ψ = exp` the second
derivative equals the first, and the coupling term is
`2 s³ · e_bᵀ Hθ Ξ_θ e_b + 2 s² · e_bᵀ diag(ḡ) Ξ_θ e_b`. -/
theorem couplingTerm_exp {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    (Ξ : Matrix (Fin d) (Fin d) ℝ) (gbar e : Fin d → ℝ) (s : ℝ) :
    crossTerm (preHessian (fun _ => s) (fun _ => s) gbar Hθ) (Ξ * diagonal fun _ => s) e
      = 2 * s ^ 3 * curvCoupling Hθ Ξ e + 2 * s ^ 2 * gradCoupling gbar Ξ e := by
  rw [couplingTerm_single_prefactor hH]
  ring

end Softplus

/-! ## 4. The prefactor asymptotics of the coupling term

The coupling term of `eq:sigma-eff` is, for the softplus positivity map under (A2), the polynomial
`C(s) = 2 s³ a + 2 s² (1 − s) b` in the prefactor, with `a = e_bᵀ Hθ Ξ_θ e_b` the curvature-weighted
and `b = e_bᵀ diag(ḡ) Ξ_θ e_b` the mean-gradient-weighted frozen coupling. It is `O(s²)` always,
`Θ(s²)` when `b ≠ 0`, `Θ(s³)` when `b = 0 ≠ a`, and never `Θ(s)`. -/

section Asymptotics

/-- The coupling term of `eq:sigma-eff` as a function of the prefactor, in the softplus form of
`couplingTerm_softplus`. -/
def couplingPoly (a b s : ℝ) : ℝ := 2 * s ^ 3 * a + 2 * s ^ 2 * (1 - s) * b

theorem couplingPoly_eq_sq_mul (a b s : ℝ) :
    couplingPoly a b s = s ^ 2 * (2 * s * a + 2 * (1 - s) * b) := by
  unfold couplingPoly
  ring

/-- On the unit interval of prefactors the coupling term is at most `2 (|a| + |b|) s²`. -/
theorem abs_couplingPoly_le (a b : ℝ) {s : ℝ} (hs : 0 ≤ s) (hs1 : s ≤ 1) :
    |couplingPoly a b s| ≤ 2 * (|a| + |b|) * s ^ 2 := by
  rw [couplingPoly_eq_sq_mul, abs_mul, abs_of_nonneg (sq_nonneg s)]
  have h1 : |2 * s * a + 2 * (1 - s) * b| ≤ 2 * (|a| + |b|) := by
    calc |2 * s * a + 2 * (1 - s) * b| ≤ |2 * s * a| + |2 * (1 - s) * b| := abs_add_le _ _
      _ = 2 * s * |a| + 2 * (1 - s) * |b| := by
          simp only [abs_mul, abs_two, abs_of_nonneg hs, abs_of_nonneg (sub_nonneg.mpr hs1)]
      _ ≤ 2 * 1 * |a| + 2 * 1 * |b| := by
          gcongr
          linarith
      _ = 2 * (|a| + |b|) := by ring
  calc s ^ 2 * |2 * s * a + 2 * (1 - s) * b| ≤ s ^ 2 * (2 * (|a| + |b|)) := by
        gcongr
    _ = 2 * (|a| + |b|) * s ^ 2 := by ring

/-- **The coupling term is `O(s²)` as the positivity map flattens.** -/
theorem couplingPoly_isBigO_sq (a b : ℝ) :
    couplingPoly a b =O[𝓝[>] (0:ℝ)] fun s => s ^ 2 := by
  have hmem : Set.Ioi (0:ℝ) ∩ Set.Iio 1 ∈ 𝓝[>] (0:ℝ) :=
    Filter.inter_mem self_mem_nhdsWithin (nhdsWithin_le_nhds (Iio_mem_nhds (by norm_num)))
  rw [isBigO_iff]
  refine ⟨2 * (|a| + |b|), Filter.eventually_of_mem hmem fun s hs => ?_⟩
  obtain ⟨hs0, hs1⟩ := hs
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (sq_nonneg s)]
  exact abs_couplingPoly_le a b (le_of_lt hs0) (le_of_lt hs1)

/-- **The coupling term is `o(s)`**: divided by the prefactor it vanishes as the positivity map
flattens. This is the statement that refutes a single power of `σ_s` on `eq:sigma-eff`. -/
theorem couplingPoly_div_id_tendsto_zero (a b : ℝ) :
    Tendsto (fun s => couplingPoly a b s / s) (𝓝[>] (0:ℝ)) (𝓝 0) := by
  have hcont : Tendsto (fun s : ℝ => 2 * s ^ 2 * a + 2 * s * (1 - s) * b) (𝓝 0) (𝓝 0) := by
    have hc : Continuous (fun s : ℝ => 2 * s ^ 2 * a + 2 * s * (1 - s) * b) := by fun_prop
    simpa using hc.tendsto 0
  refine (tendsto_nhdsWithin_of_tendsto_nhds hcont).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hs' : (s:ℝ) ≠ 0 := ne_of_gt hs
  rw [couplingPoly]
  field_simp

/-- A function that is `O(s²)` on the right of `0` is not `Θ(s)` there. -/
theorem not_isTheta_id_of_isBigO_sq {f : ℝ → ℝ} (hf : f =O[𝓝[>] (0:ℝ)] fun s => s ^ 2) :
    ¬ (f =Θ[𝓝[>] (0:ℝ)] fun s => s) := by
  intro hΘ
  have h1 : (fun s : ℝ => s) =O[𝓝[>] (0:ℝ)] fun s => s ^ 2 := hΘ.isBigO_symm.trans hf
  rw [isBigO_iff] at h1
  obtain ⟨C, hC⟩ := h1
  have hsmall : ∀ᶠ s in 𝓝[>] (0:ℝ), s < 1 / (|C| + 1) := by
    apply nhdsWithin_le_nhds
    exact Iio_mem_nhds (by positivity)
  have hpos : ∀ᶠ s in 𝓝[>] (0:ℝ), 0 < s :=
    Filter.eventually_of_mem self_mem_nhdsWithin fun s hs => hs
  obtain ⟨s, hs0, hCs, hlt⟩ := (hpos.and (hC.and hsmall)).exists
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hs0, abs_of_nonneg (sq_nonneg s)] at hCs
  have h2 : s ≤ |C| * s ^ 2 := le_trans hCs (by
    have := le_abs_self C
    nlinarith [sq_nonneg s])
  have h3 : 1 ≤ |C| * s := by nlinarith
  have h4 : |C| * s < 1 := by
    have hpos : (0:ℝ) < |C| + 1 := by positivity
    calc |C| * s ≤ (|C| + 1) * s := by nlinarith [abs_nonneg C]
      _ < (|C| + 1) * (1 / (|C| + 1)) := by gcongr
      _ = 1 := by field_simp
  linarith

/-- **The dimensionless coupling ratio is bounded.** The ratio of the coupling term of
`eq:sigma-eff` to the direct-softplus term `s² σ_obj²` converges to `2 b/σ_obj²` as the positivity map
flattens, where the additive form's ratio `ϱ = σ_Jac²/(s² σ_obj²)` diverges
(`PowerOfS.couplingRatio_tendsto_atTop`). A directional gate that compares this ratio against a
bar growing like `s⁻²` therefore cannot clear deep on the shoulder. -/
theorem couplingPoly_div_direct_tendsto (a b : ℝ) {o : ℝ} (ho : o ≠ 0) :
    Tendsto (fun s => couplingPoly a b s / (s ^ 2 * o)) (𝓝[>] (0:ℝ)) (𝓝 (2 * b / o)) := by
  have hcont : Tendsto (fun s : ℝ => (2 * s * a + 2 * (1 - s) * b) / o) (𝓝 0)
      (𝓝 (2 * b / o)) := by
    have hc : Continuous (fun s : ℝ => (2 * s * a + 2 * (1 - s) * b) / o) := by fun_prop
    have h := hc.tendsto 0
    simpa using h
  refine (tendsto_nhdsWithin_of_tendsto_nhds hcont).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hs' : (s:ℝ) ≠ 0 := ne_of_gt hs
  rw [couplingPoly_eq_sq_mul]
  field_simp

/-- **The coupling term of `eq:sigma-eff` is never `Θ(s)`**, whatever the values of the two
frozen couplings. The single power of the prefactor that `σ_Jac²` carries on its own does not
survive the contraction with the latent-weight Hessian. -/
theorem couplingPoly_not_isTheta_id (a b : ℝ) :
    ¬ (couplingPoly a b =Θ[𝓝[>] (0:ℝ)] fun s => s) :=
  not_isTheta_id_of_isBigO_sq (couplingPoly_isBigO_sq a b)

/-- **The coupling term is `Θ(s²)` when the mean-gradient route is open**, that is when
`b = e_bᵀ diag(ḡ) Ξ_θ e_b ≠ 0`: the mean constrained gradient at the bias coordinate is nonzero,
which is what being on the shoulder means, and the frozen coupling is nonzero there. The
two-sided constants are explicit, `|b|/2` below and `2 (|a| + |b|)` above, on the band
`0 < s ≤ min (1/2) (|b| / (4 (|a| + 1)))`. -/
theorem couplingPoly_isTheta_sq_of_ne {a b : ℝ} (hb : b ≠ 0) :
    couplingPoly a b =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 2 := by
  have hbpos : 0 < |b| := abs_pos.mpr hb
  have hs₀ : 0 < min (1 / 2 : ℝ) (|b| / (4 * (|a| + 1))) := by positivity
  have habs : (fun s => |couplingPoly a b s|) =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 2 := by
    refine isTheta_nhdsGT_zero_of_two_sided_near (c₁ := |b| / 2) (c₂ := 2 * (|a| + |b|))
      (by positivity) hs₀ (fun s _ _ => sq_nonneg s) (fun s hs hs0 => ?_) (fun s hs hs0 => ?_)
    · have hs1 : s ≤ 1 / 2 := le_trans hs0 (min_le_left _ _)
      have hs2 : s ≤ |b| / (4 * (|a| + 1)) := le_trans hs0 (min_le_right _ _)
      have hsa : 2 * s * |a| ≤ |b| / 2 := by
        have h4 : 0 < 4 * (|a| + 1) := by positivity
        have : s * (4 * (|a| + 1)) ≤ |b| := by
          rw [le_div_iff₀ h4] at hs2
          exact hs2
        nlinarith [abs_nonneg a]
      rw [couplingPoly_eq_sq_mul, abs_mul, abs_of_nonneg (sq_nonneg s)]
      have hinner : |b| / 2 ≤ |2 * s * a + 2 * (1 - s) * b| := by
        have h1 : |2 * (1 - s) * b| - |2 * s * a| ≤ |2 * s * a + 2 * (1 - s) * b| := by
          have := abs_sub_abs_le_abs_sub (2 * (1 - s) * b) (-(2 * s * a))
          rw [abs_neg, sub_neg_eq_add, add_comm] at this
          exact this
        have h2 : |2 * (1 - s) * b| = 2 * (1 - s) * |b| := by
          rw [abs_mul, abs_mul, abs_two, abs_of_nonneg (by linarith : (0:ℝ) ≤ 1 - s)]
        have h3 : |2 * s * a| = 2 * s * |a| := by
          rw [abs_mul, abs_mul, abs_two, abs_of_pos hs]
        rw [h2, h3] at h1
        nlinarith
      calc |b| / 2 * s ^ 2 = s ^ 2 * (|b| / 2) := by ring
        _ ≤ s ^ 2 * |2 * s * a + 2 * (1 - s) * b| := by gcongr
    · have hs1 : s ≤ 1 := le_trans (le_trans hs0 (min_le_left _ _)) (by norm_num)
      have := abs_couplingPoly_le a b hs.le hs1
      linarith
  have hnorm : (fun s => ‖couplingPoly a b s‖) =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 2 := by
    simpa only [Real.norm_eq_abs] using habs
  exact isTheta_norm_left.mp hnorm

/-- **The coupling term is `Θ(s³)` when the mean-gradient route is closed**, `b = 0`, and the
curvature route is open, `a = e_bᵀ Hθ Ξ_θ e_b ≠ 0`. -/
theorem couplingPoly_isTheta_cube_of_eq {a : ℝ} (ha : a ≠ 0) :
    couplingPoly a 0 =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 3 := by
  have hapos : 0 < |a| := abs_pos.mpr ha
  have habs : (fun s => |couplingPoly a 0 s|) =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 3 := by
    refine isTheta_nhdsGT_zero_of_two_sided (c₁ := 2 * |a|) (c₂ := 2 * |a|) (by positivity)
      (fun s hs => by positivity) (fun s hs => ?_) (fun s hs => ?_)
    · have : |couplingPoly a 0 s| = 2 * |a| * s ^ 3 := by
        simp only [couplingPoly, mul_zero, add_zero, abs_mul, abs_two, abs_of_pos hs, abs_pow]
        ring
      rw [this]
    · have : |couplingPoly a 0 s| = 2 * |a| * s ^ 3 := by
        simp only [couplingPoly, mul_zero, add_zero, abs_mul, abs_two, abs_of_pos hs, abs_pow]
        ring
      rw [this]
  have hnorm : (fun s => ‖couplingPoly a 0 s‖) =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 3 := by
    simpa only [Real.norm_eq_abs] using habs
  exact isTheta_norm_left.mp hnorm

end Asymptotics

/-! ## 5. The whole of `eq:sigma-eff` in powers of the prefactor

The three latent weight terms of `eq:sigma-eff` — coupling, jitter and remainder — are assembled with
the direct-softplus term on the latent-weight objects of the chain rule, and every one of them
carries at least two powers of the prefactor. The projected diffusion of the lift is therefore
`Θ(s²)`, as the direct one is, and the exponent `2α/σ_eff²` is `Θ(s⁻²)` for both constructions.
The lift changes the constant in the exponent and not its order. -/

section Assembled

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The jitter term of `eq:sigma-eff` as a polynomial in the prefactor, softplus form:
`j₀ = e_bᵀ diag(ḡ) V diag(ḡ) e_b`, `m = e_bᵀ Hθ V diag(ḡ) e_b + e_bᵀ diag(ḡ) V Hθ e_b`,
`j₂ = e_bᵀ Hθ V Hθ e_b`. -/
def jitterPoly (j₀ m j₂ s : ℝ) : ℝ :=
  s ^ 2 * (1 - s) ^ 2 * j₀ + s ^ 3 * (1 - s) * m + s ^ 4 * j₂

/-- **The jitter term under (A2) for the softplus positivity map**: `e_bᵀ H_θ̃ V H_θ̃ e_b` with
`H_θ̃ = s² Hθ + s (1 − s) diag(ḡ)` is `jitterPoly` evaluated at the three constrained-unit
scalars. Its leading term `s² (1 − s)² e_bᵀ diag(ḡ) V diag(ḡ) e_b` is again attenuated twice. -/
theorem jitterTerm_softplus (Hθ V : Matrix (Fin d) (Fin d) ℝ) (gbar e : Fin d → ℝ) (s : ℝ) :
    e ⬝ᵥ (magnitudeMatrix (preHessian (fun _ => s) (fun _ => s * (1 - s)) gbar Hθ) V *ᵥ e)
      = jitterPoly (e ⬝ᵥ ((diagonal gbar * V * diagonal gbar) *ᵥ e))
          (e ⬝ᵥ ((Hθ * V * diagonal gbar) *ᵥ e) + e ⬝ᵥ ((diagonal gbar * V * Hθ) *ᵥ e))
          (e ⬝ᵥ ((Hθ * V * Hθ) *ᵥ e)) s := by
  rw [preHessian_single, magnitudeMatrix, jitterPoly]
  simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul, quadForm_add,
    quadForm_smul]
  ring

/-- The three remainder scalars of `MinibatchNoise.projDiffusion_lift_eq_frozen_form`, in the
latent-weight units: the pairing of the frozen gradient with `r₂`, the pairing of the
curvature-weighted jitter with `r₂`, and the variance of `r₂`. This is the `ϱ` of
`eq:sigma-eff`. -/
noncomputable def remainderScalars (μ : Measure Ω) (H : Matrix (Fin d) (Fin d) ℝ)
    (g₀ dth r₂ : Ω → Fin d → ℝ) (e : Fin d → ℝ) : ℝ :=
  2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e))
    + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e)

/-- **`eq:sigma-eff` on the latent-weight objects, in powers of the prefactor.** Take the refined
chain of (A4) in the latent-weight coordinate at positivity-map level `s`: the per-probe gradient
with respect to the latent weight is `g = s ⊙ g_θ + H_θ̃ δθ̃ + r₂`, with `g_θ` the frozen constrained gradient,
`H_θ̃ = s² Hθ + s (1 − s) diag(ḡ)` the latent-weight Hessian of the softplus positivity map under (A2), and
`r₂` the remainder. Then the projected diffusion of the lift is

  `σ²_eff = s² · e_bᵀ Cov[g_θ] e_b + C(s) + J(s) + ϱ`,

with `C` the coupling polynomial of `couplingTerm_softplus`, `J` the jitter polynomial of
`jitterTerm_softplus`, and `ϱ` the three remainder scalars; the first summand is the
direct-softplus diffusion `s² σ_obj²`. The identity is
`MinibatchNoise.projDiffusion_lift_eq_frozen_form` at positivity-map slope one, with the chain-rule
forms of the frozen coupling and of the Hessian substituted; it is exact. -/
theorem projDiffusion_lift_softplus_form [IsFiniteMeasure μ] {Hθ : Matrix (Fin d) (Fin d) ℝ}
    (hH : Hθᵀ = Hθ) (gbar : Fin d → ℝ) (s : ℝ) {g gθ dth r₂ : Ω → Fin d → ℝ}
    (hgθ : SqIntegrableVec μ gθ) (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = (fun i => s * gθ ω i)
      + preHessian (fun _ => s) (fun _ => s * (1 - s)) gbar Hθ *ᵥ dth ω + r₂ ω)
    (e : Fin d → ℝ) :
    projDiffusion μ e g
      = s ^ 2 * (e ⬝ᵥ (covMat μ gθ gθ *ᵥ e))
        + couplingPoly (curvCoupling Hθ (covMat μ dth gθ) e)
            (gradCoupling gbar (covMat μ dth gθ) e) s
        + jitterPoly (e ⬝ᵥ ((diagonal gbar * covMat μ dth dth * diagonal gbar) *ᵥ e))
            (e ⬝ᵥ ((Hθ * covMat μ dth dth * diagonal gbar) *ᵥ e)
              + e ⬝ᵥ ((diagonal gbar * covMat μ dth dth * Hθ) *ᵥ e))
            (e ⬝ᵥ ((Hθ * covMat μ dth dth * Hθ) *ᵥ e)) s
        + remainderScalars μ (preHessian (fun _ => s) (fun _ => s * (1 - s)) gbar Hθ)
            (fun ω i => s * gθ ω i) dth r₂ e := by
  have hg₀ : SqIntegrableVec μ (fun ω i => s * gθ ω i) := fun i => (hgθ i).const_mul s
  have hsym := preHessian_transpose (p := fun _ => s) (p₂ := fun _ => s * (1 - s))
    (gbar := gbar) hH
  have h := projDiffusion_lift_eq_frozen_form (μ := μ) 1 hsym hg₀ hθ hr hchain
    (Glift := g) (Gdir := fun ω i => s * gθ ω i) (fun ω => (one_smul ℝ (g ω)).symm)
    (fun ω => (one_smul ℝ _).symm) e
  rw [h, one_pow, one_mul, one_mul, remainderScalars]
  have hΞ : covMat μ dth (fun ω i => s * gθ ω i) = covMat μ dth gθ * diagonal fun _ => s :=
    covMat_hadamard_right (fun _ => s) hθ hgθ
  have hdir : projDiffusion μ e (fun ω i => s * gθ ω i) = s ^ 2 * (e ⬝ᵥ (covMat μ gθ gθ *ᵥ e)) := by
    rw [projDiffusion, covMat_hadamard_self (fun _ => s) hgθ, ← smul_one_eq_diagonal,
      Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul, Matrix.mul_one, smul_smul, quadForm_smul,
      pow_two]
  rw [hdir, hΞ, couplingTerm_softplus hH, jitterTerm_softplus, couplingPoly]
  ring

/-- **The remainder scalars are attenuated twice when the remainder is attenuated once.** If the
latent-weight remainder is `r₂ = s ⊙ r̃`, as the chain rule makes it (the per-batch Hessian
fluctuation in the latent-weight coordinate is `diag(ψ'') diag(ξ_θ) + diag(ψ') δHθ diag(ψ')`, one
prefactor at least on every term), then the three remainder scalars are `s²` times the same
three scalars formed with `r̃`, with the mixed pairing weighted by `s Hθ + (1 − s) diag(ḡ)`. -/
theorem remainderScalars_once_attenuated (Hθ : Matrix (Fin d) (Fin d) ℝ)
    (gbar : Fin d → ℝ) (s : ℝ) (gθ dth rt : Ω → Fin d → ℝ) (e : Fin d → ℝ) :
    remainderScalars μ (preHessian (fun _ => s) (fun _ => s * (1 - s)) gbar Hθ)
        (fun ω i => s * gθ ω i) dth (fun ω i => s * rt ω i) e
      = s ^ 2 * (2 * (e ⬝ᵥ (covMat μ gθ rt *ᵥ e))
          + 2 * (e ⬝ᵥ (((s • Hθ + (1 - s) • diagonal gbar) * covMat μ dth rt) *ᵥ e))
          + e ⬝ᵥ (covMat μ rt rt *ᵥ e)) := by
  have h1 : covMat μ (fun ω i => s * gθ ω i) (fun ω i => s * rt ω i)
      = (s * s) • covMat μ gθ rt := by
    have hl : (fun ω i => s * gθ ω i) = fun ω => s • gθ ω := by funext ω i; rfl
    have hr : (fun ω i => s * rt ω i) = fun ω => s • rt ω := by funext ω i; rfl
    rw [hl, hr, covMat_smul_left, covMat_smul_right, smul_smul]
  have h2 : covMat μ dth (fun ω i => s * rt ω i) = s • covMat μ dth rt := by
    have hr : (fun ω i => s * rt ω i) = fun ω => s • rt ω := by funext ω i; rfl
    rw [hr, covMat_smul_right]
  have h3 : covMat μ (fun ω i => s * rt ω i) (fun ω i => s * rt ω i)
      = (s * s) • covMat μ rt rt := by
    have hr : (fun ω i => s * rt ω i) = fun ω => s • rt ω := by funext ω i; rfl
    rw [hr, covMat_smul_left, covMat_smul_right, smul_smul]
  rw [remainderScalars, h1, h2, h3, preHessian_single]
  simp only [Matrix.add_mul, Matrix.smul_mul, Matrix.mul_smul, quadForm_add, quadForm_smul]
  ring

/-- Under the once-attenuated remainder of `remainderScalars_once_attenuated`, the remainder
scalars are bounded by `τ s²` on the unit interval of prefactors, with `τ` the sum of the absolute
values of the constrained-unit scalars: this is (A4′-r) read at the positivity-map scale, and it is the
hypothesis `hϱ` of the asymptotic statements below. -/
theorem abs_remainderScalars_le_of_once_attenuated
    (Hθ : Matrix (Fin d) (Fin d) ℝ) (gbar : Fin d → ℝ) {s : ℝ} (hs : 0 ≤ s) (hs1 : s ≤ 1)
    (gθ dth rt : Ω → Fin d → ℝ) (e : Fin d → ℝ) :
    |remainderScalars μ (preHessian (fun _ => s) (fun _ => s * (1 - s)) gbar Hθ)
        (fun ω i => s * gθ ω i) dth (fun ω i => s * rt ω i) e|
      ≤ (2 * |e ⬝ᵥ (covMat μ gθ rt *ᵥ e)|
          + 2 * (|e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e)|
            + |e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)|)
          + |e ⬝ᵥ (covMat μ rt rt *ᵥ e)|) * s ^ 2 := by
  rw [remainderScalars_once_attenuated Hθ gbar s gθ dth rt e, abs_mul, abs_of_nonneg (sq_nonneg s),
    mul_comm]
  gcongr
  simp only [Matrix.add_mul, Matrix.smul_mul, quadForm_add, quadForm_smul]
  have hs' : |s| = s := abs_of_nonneg hs
  have h1s : |1 - s| = 1 - s := abs_of_nonneg (by linarith)
  calc |2 * (e ⬝ᵥ (covMat μ gθ rt *ᵥ e))
        + 2 * (s * (e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e))
          + (1 - s) * (e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)))
        + e ⬝ᵥ (covMat μ rt rt *ᵥ e)|
      ≤ |2 * (e ⬝ᵥ (covMat μ gθ rt *ᵥ e))|
        + |2 * (s * (e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e))
          + (1 - s) * (e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)))|
        + |e ⬝ᵥ (covMat μ rt rt *ᵥ e)| := abs_add_three _ _ _
    _ ≤ 2 * |e ⬝ᵥ (covMat μ gθ rt *ᵥ e)|
        + 2 * (s * |e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e)|
          + (1 - s) * |e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)|)
        + |e ⬝ᵥ (covMat μ rt rt *ᵥ e)| := by
          gcongr
          · rw [abs_mul, abs_two]
          · rw [abs_mul, abs_two]
            gcongr
            calc |s * (e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e))
                  + (1 - s) * (e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e))|
                ≤ |s * (e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e))|
                  + |(1 - s) * (e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e))| := abs_add_le _ _
              _ = s * |e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e)|
                  + (1 - s) * |e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)| := by
                  rw [abs_mul, abs_mul, hs', h1s]
    _ ≤ 2 * |e ⬝ᵥ (covMat μ gθ rt *ᵥ e)|
        + 2 * (1 * |e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e)|
          + 1 * |e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)|)
        + |e ⬝ᵥ (covMat μ rt rt *ᵥ e)| := by
          gcongr
          linarith
    _ = 2 * |e ⬝ᵥ (covMat μ gθ rt *ᵥ e)|
        + 2 * (|e ⬝ᵥ ((Hθ * covMat μ dth rt) *ᵥ e)|
          + |e ⬝ᵥ ((diagonal gbar * covMat μ dth rt) *ᵥ e)|)
        + |e ⬝ᵥ (covMat μ rt rt *ᵥ e)| := by ring

end Assembled

/-! ## 6. The exponent: `Θ(s⁻²)` for the lift as for direct softplus

The scalar content of the previous section, isolated: `eq:sigma-eff` as a function of the
prefactor is `s² c₂ + s³ R(s) + ϱ(s)` with `c₂ = σ_obj² + 2 b + j₀` and `R` bounded on the unit
interval, and any remainder bounded by `τ s²` with `τ < c₂` leaves it `Θ(s²)`. -/

section Exponent

/-- `eq:sigma-eff` as a function of the prefactor: the direct term, the coupling polynomial, the
jitter polynomial and the remainder. -/
def sigmaEffPoly (o a b j₀ m j₂ : ℝ) (ϱ : ℝ → ℝ) (s : ℝ) : ℝ :=
  s ^ 2 * o + couplingPoly a b s + jitterPoly j₀ m j₂ s + ϱ s

/-- The leading coefficient `c₂ = σ_obj² + 2 b + j₀` of `eq:sigma-eff` in the prefactor. -/
def leadingCoeff (o b j₀ : ℝ) : ℝ := o + 2 * b + j₀

/-- The constant bounding the cubic and higher terms of `eq:sigma-eff` on the unit interval. -/
def higherOrderBound (a b j₀ m j₂ : ℝ) : ℝ := 2 * |a| + 2 * |b| + 2 * |j₀| + |m| + |j₂|

theorem sigmaEffPoly_eq (o a b j₀ m j₂ : ℝ) (ϱ : ℝ → ℝ) (s : ℝ) :
    sigmaEffPoly o a b j₀ m j₂ ϱ s
      = s ^ 2 * leadingCoeff o b j₀
        + s ^ 3 * (2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m + s * j₂) + ϱ s := by
  unfold sigmaEffPoly couplingPoly jitterPoly leadingCoeff
  ring

/-- On the unit interval the cubic coefficient is bounded by `higherOrderBound`. -/
theorem abs_cubicCoeff_le (a b j₀ m j₂ : ℝ) {s : ℝ} (hs : 0 ≤ s) (hs1 : s ≤ 1) :
    |2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m + s * j₂| ≤ higherOrderBound a b j₀ m j₂ := by
  unfold higherOrderBound
  have h1 : |(s - 2) * j₀| ≤ 2 * |j₀| := by
    rw [abs_mul]
    have : |s - 2| ≤ 2 := by rw [abs_le]; constructor <;> linarith
    nlinarith [abs_nonneg j₀]
  have h2 : |(1 - s) * m| ≤ |m| := by
    rw [abs_mul]
    have : |1 - s| ≤ 1 := by rw [abs_le]; constructor <;> linarith
    nlinarith [abs_nonneg m]
  have h3 : |s * j₂| ≤ |j₂| := by
    rw [abs_mul]
    have : |s| ≤ 1 := by rw [abs_le]; constructor <;> linarith
    nlinarith [abs_nonneg j₂]
  have h4 : |2 * a - 2 * b| ≤ 2 * |a| + 2 * |b| := by
    calc |2 * a - 2 * b| ≤ |2 * a| + |2 * b| := abs_sub _ _
      _ = 2 * |a| + 2 * |b| := by rw [abs_mul, abs_mul, abs_two]
  calc |2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m + s * j₂|
      ≤ |2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m| + |s * j₂| := abs_add_le _ _
    _ ≤ (|2 * a - 2 * b + (s - 2) * j₀| + |(1 - s) * m|) + |s * j₂| := by
        gcongr
        exact abs_add_le _ _
    _ ≤ ((|2 * a - 2 * b| + |(s - 2) * j₀|) + |(1 - s) * m|) + |s * j₂| := by
        gcongr
        exact abs_add_le _ _
    _ ≤ ((2 * |a| + 2 * |b| + 2 * |j₀|) + |m|) + |j₂| := by gcongr
    _ = 2 * |a| + 2 * |b| + 2 * |j₀| + |m| + |j₂| := by ring

/-- **Two-sided bounds for `eq:sigma-eff` near a flat positivity map.** With the remainder bounded by
`τ s²`, on the unit interval

  `s² (c₂ − τ) − s³ K ≤ σ²_eff(s) ≤ s² (c₂ + τ) + s³ K`,

`K = higherOrderBound`. -/
theorem sigmaEffPoly_two_sided (o a b j₀ m j₂ : ℝ) {ϱ : ℝ → ℝ} {τ s : ℝ} (hs : 0 ≤ s)
    (hs1 : s ≤ 1) (hϱ : |ϱ s| ≤ τ * s ^ 2) :
    s ^ 2 * (leadingCoeff o b j₀ - τ) - s ^ 3 * higherOrderBound a b j₀ m j₂
        ≤ sigmaEffPoly o a b j₀ m j₂ ϱ s
      ∧ sigmaEffPoly o a b j₀ m j₂ ϱ s
        ≤ s ^ 2 * (leadingCoeff o b j₀ + τ) + s ^ 3 * higherOrderBound a b j₀ m j₂ := by
  rw [sigmaEffPoly_eq]
  have hK := abs_cubicCoeff_le a b j₀ m j₂ hs hs1
  have hs3 : 0 ≤ s ^ 3 := by positivity
  obtain ⟨hK1, hK2⟩ := abs_le.mp hK
  obtain ⟨hϱ1, hϱ2⟩ := abs_le.mp hϱ
  have hm1 := mul_le_mul_of_nonneg_left hK1 hs3
  have hm2 := mul_le_mul_of_nonneg_left hK2 hs3
  constructor <;> nlinarith

/-- **`eq:sigma-eff` is `Θ(s²)`.** If the leading coefficient `c₂ = σ_obj² + 2 b + j₀` exceeds
the remainder constant `τ` and the remainder obeys `|ϱ(s)| ≤ τ s²` near a flat positivity map, the
projected diffusion of the lift is of the order `s²` exactly: the same order as the
direct-softplus diffusion `s² σ_obj²`, and not the order `s` of the additive form. -/
theorem sigmaEffPoly_isTheta_sq {o a b j₀ m j₂ τ s₀ : ℝ} {ϱ : ℝ → ℝ}
    (hc : τ < leadingCoeff o b j₀) (hs₀ : 0 < s₀)
    (hϱ : ∀ s, 0 < s → s ≤ s₀ → |ϱ s| ≤ τ * s ^ 2) :
    sigmaEffPoly o a b j₀ m j₂ ϱ =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 2 := by
  set K := higherOrderBound a b j₀ m j₂ with hK_def
  have hK : 0 ≤ K := by
    unfold higherOrderBound at hK_def
    rw [hK_def]
    positivity
  set c := leadingCoeff o b j₀ with hc_def
  have hgap : 0 < c - τ := by linarith
  set s₁ := min s₀ (min 1 ((c - τ) / (2 * (K + 1)))) with hs₁_def
  have hs₁ : 0 < s₁ := by
    rw [hs₁_def]
    apply lt_min hs₀
    apply lt_min one_pos
    positivity
  refine isTheta_nhdsGT_zero_of_two_sided_near (c₁ := (c - τ) / 2) (c₂ := c + τ + K)
    (by positivity) hs₁ (fun s _ _ => sq_nonneg s) (fun s hs hs0 => ?_) (fun s hs hs0 => ?_)
  · have hA : s ≤ s₀ := le_trans hs0 (min_le_left _ _)
    have hB : s ≤ 1 := le_trans hs0 (le_trans (min_le_right _ _) (min_le_left _ _))
    have hC : s ≤ (c - τ) / (2 * (K + 1)) :=
      le_trans hs0 (le_trans (min_le_right _ _) (min_le_right _ _))
    have hsK : s * K ≤ (c - τ) / 2 := by
      have h2 : 0 < 2 * (K + 1) := by positivity
      rw [le_div_iff₀ h2] at hC
      nlinarith
    obtain ⟨hlo, _⟩ := sigmaEffPoly_two_sided o a b j₀ m j₂ hs.le hB (hϱ s hs hA)
    have hs2 : 0 ≤ s ^ 2 := sq_nonneg s
    have : s ^ 3 * K ≤ s ^ 2 * ((c - τ) / 2) := by
      calc s ^ 3 * K = s ^ 2 * (s * K) := by ring
        _ ≤ s ^ 2 * ((c - τ) / 2) := by gcongr
    linarith
  · have hA : s ≤ s₀ := le_trans hs0 (min_le_left _ _)
    have hB : s ≤ 1 := le_trans hs0 (le_trans (min_le_right _ _) (min_le_left _ _))
    obtain ⟨_, hhi⟩ := sigmaEffPoly_two_sided o a b j₀ m j₂ hs.le hB (hϱ s hs hA)
    have : s ^ 3 * K ≤ s ^ 2 * K := by
      have hs2 : 0 ≤ s ^ 2 := sq_nonneg s
      calc s ^ 3 * K = s ^ 2 * (s * K) := by ring
        _ ≤ s ^ 2 * (1 * K) := by gcongr
        _ = s ^ 2 * K := by ring
    linarith

/-- A positive function that is `Θ(s²)` on the right of `0` has `2α/f` of order `s⁻²`. -/
theorem shoulderExponent_isTheta_inv_sq_of_two_sided {f : ℝ → ℝ} {α c₁ c₂ s₀ : ℝ} (hα : 0 < α)
    (hc₁ : 0 < c₁) (hs₀ : 0 < s₀)
    (hlo : ∀ s, 0 < s → s ≤ s₀ → c₁ * s ^ 2 ≤ f s)
    (hhi : ∀ s, 0 < s → s ≤ s₀ → f s ≤ c₂ * s ^ 2) :
    (fun s => shoulderExponent α (f s)) =Θ[𝓝[>] (0:ℝ)] fun s => (s ^ 2)⁻¹ := by
  have hc₂ : 0 < c₂ := by
    have h1 := hlo s₀ hs₀ le_rfl
    have h2 := hhi s₀ hs₀ le_rfl
    have : 0 < c₁ * s₀ ^ 2 := by positivity
    nlinarith [sq_nonneg s₀]
  refine isTheta_nhdsGT_zero_of_two_sided_near (c₁ := 2 * α / c₂) (c₂ := 2 * α / c₁)
    (by positivity) hs₀ (fun _s hs _ => by positivity) (fun s hs hs0 => ?_) (fun s hs hs0 => ?_)
  · have hpos : 0 < f s := lt_of_lt_of_le (by positivity) (hlo s hs hs0)
    have heq : 2 * α / c₂ * (s ^ 2)⁻¹ = 2 * α / (c₂ * s ^ 2) := by field_simp
    rw [heq, shoulderExponent, div_le_div_iff₀ (by positivity) hpos]
    have := hhi s hs hs0
    nlinarith
  · have hpos : 0 < f s := lt_of_lt_of_le (by positivity) (hlo s hs hs0)
    have heq : 2 * α / c₁ * (s ^ 2)⁻¹ = 2 * α / (c₁ * s ^ 2) := by field_simp
    rw [heq, shoulderExponent, div_le_div_iff₀ hpos (by positivity)]
    have := hlo s hs hs0
    nlinarith

/-- The two-sided bound of `sigmaEffPoly_isTheta_sq`, with its constants and its band made
explicit for reuse. -/
theorem sigmaEffPoly_two_sided_near {o a b j₀ m j₂ τ s₀ : ℝ} {ϱ : ℝ → ℝ}
    (hc : τ < leadingCoeff o b j₀) (hs₀ : 0 < s₀)
    (hϱ : ∀ s, 0 < s → s ≤ s₀ → |ϱ s| ≤ τ * s ^ 2) :
    ∃ s₁, 0 < s₁ ∧ ∀ s, 0 < s → s ≤ s₁ →
      (leadingCoeff o b j₀ - τ) / 2 * s ^ 2 ≤ sigmaEffPoly o a b j₀ m j₂ ϱ s
        ∧ sigmaEffPoly o a b j₀ m j₂ ϱ s
          ≤ (leadingCoeff o b j₀ + τ + higherOrderBound a b j₀ m j₂) * s ^ 2 := by
  set K := higherOrderBound a b j₀ m j₂ with hK_def
  have hK : 0 ≤ K := by
    unfold higherOrderBound at hK_def
    rw [hK_def]
    positivity
  set c := leadingCoeff o b j₀ with hc_def
  have hgap : 0 < c - τ := by linarith
  refine ⟨min s₀ (min 1 ((c - τ) / (2 * (K + 1)))), ?_, fun s hs hs0 => ?_⟩
  · apply lt_min hs₀
    apply lt_min one_pos
    positivity
  have hA : s ≤ s₀ := le_trans hs0 (min_le_left _ _)
  have hB : s ≤ 1 := le_trans hs0 (le_trans (min_le_right _ _) (min_le_left _ _))
  have hC : s ≤ (c - τ) / (2 * (K + 1)) :=
    le_trans hs0 (le_trans (min_le_right _ _) (min_le_right _ _))
  have hsK : s * K ≤ (c - τ) / 2 := by
    have h2 : 0 < 2 * (K + 1) := by positivity
    rw [le_div_iff₀ h2] at hC
    nlinarith
  obtain ⟨hlo, hhi⟩ := sigmaEffPoly_two_sided o a b j₀ m j₂ hs.le hB (hϱ s hs hA)
  have hs2 : 0 ≤ s ^ 2 := sq_nonneg s
  constructor
  · have : s ^ 3 * K ≤ s ^ 2 * ((c - τ) / 2) := by
      calc s ^ 3 * K = s ^ 2 * (s * K) := by ring
        _ ≤ s ^ 2 * ((c - τ) / 2) := by gcongr
    linarith
  · have : s ^ 3 * K ≤ s ^ 2 * K := by
      calc s ^ 3 * K = s ^ 2 * (s * K) := by ring
        _ ≤ s ^ 2 * (1 * K) := by gcongr
        _ = s ^ 2 * K := by ring
    linarith

/-- **The lift's exponent on `eq:sigma-eff` is `Θ(s⁻²)`.** Under the hypotheses of
`sigmaEffPoly_isTheta_sq`, the Arrhenius exponent `2α/σ²_eff` of the lift grows like `s⁻²` as
the positivity map flattens — the order of the direct-softplus exponent, and one order faster than the
`Θ(s⁻¹)` that `PowerOfS.liftExponent_isTheta_inv` derives for the additive form. As there, this
is a statement about the exponent as a function of the prefactor and not about an escape
time. -/
theorem liftExponent_sigmaEff_isTheta_inv_sq {o a b j₀ m j₂ τ s₀ α : ℝ} {ϱ : ℝ → ℝ}
    (hα : 0 < α) (hc : τ < leadingCoeff o b j₀) (hs₀ : 0 < s₀)
    (hϱ : ∀ s, 0 < s → s ≤ s₀ → |ϱ s| ≤ τ * s ^ 2) :
    (fun s => shoulderExponent α (sigmaEffPoly o a b j₀ m j₂ ϱ s)) =Θ[𝓝[>] (0:ℝ)]
      fun s => (s ^ 2)⁻¹ := by
  obtain ⟨s₁, hs₁, hb⟩ := sigmaEffPoly_two_sided_near (a := a) (m := m) (j₂ := j₂) hc hs₀ hϱ
  exact shoulderExponent_isTheta_inv_sq_of_two_sided hα (by linarith) hs₁
    (fun s hs hs0 => (hb s hs hs0).1) (fun s hs hs0 => (hb s hs hs0).2)

/-- The direct-softplus exponent `2α/(s² σ_obj²)` is `Θ(s⁻²)`; this is
`PowerOfS.directExponent_isTheta_inv_sq` for an abstract positive objective contribution. -/
theorem directExponent_isTheta_inv_sq' {o α : ℝ} (hα : 0 < α) (ho : 0 < o) :
    (fun s => shoulderExponent α (s ^ 2 * o)) =Θ[𝓝[>] (0:ℝ)] fun s => (s ^ 2)⁻¹ :=
  shoulderExponent_isTheta_inv_sq_of_two_sided hα ho one_pos
    (fun s _ _ => by rw [mul_comm]) (fun s _ _ => by rw [mul_comm])

/-- The ratio of the direct exponent to the lift's exponent on `eq:sigma-eff` is
`σ²_eff(s)/(s² σ_obj²)`; the barrier `α` cancels, as it does in `PowerOfS.exponentRatio_eq`. -/
theorem exponentRatio_sigmaEff_eq {o a b j₀ m j₂ α s : ℝ} {ϱ : ℝ → ℝ} (hα : α ≠ 0) (hs : s ≠ 0)
    (ho : o ≠ 0) (heff : sigmaEffPoly o a b j₀ m j₂ ϱ s ≠ 0) :
    shoulderExponent α (s ^ 2 * o) / shoulderExponent α (sigmaEffPoly o a b j₀ m j₂ ϱ s)
      = sigmaEffPoly o a b j₀ m j₂ ϱ s / (s ^ 2 * o) := by
  simp only [shoulderExponent]
  field_simp

/-- **The ratio of the two exponents is eventually bounded.** On `eq:sigma-eff` the direct
exponent exceeds the lift's by at most the constant factor `(c₂ + τ + K)/σ_obj²` near a flat
positivity map, and by at least `(c₂ − τ)/(2 σ_obj²)`: the lift lowers the exponent by a bounded factor,
where the additive form of `PowerOfS.exponentRatio_tendsto_atTop` had the ratio diverge. -/
theorem exponentRatio_sigmaEff_eventually_bounded {o a b j₀ m j₂ τ s₀ α : ℝ} {ϱ : ℝ → ℝ}
    (hα : α ≠ 0) (ho : 0 < o) (hc : τ < leadingCoeff o b j₀) (hs₀ : 0 < s₀)
    (hϱ : ∀ s, 0 < s → s ≤ s₀ → |ϱ s| ≤ τ * s ^ 2) :
    ∀ᶠ s in 𝓝[>] (0:ℝ),
      (leadingCoeff o b j₀ - τ) / (2 * o)
        ≤ shoulderExponent α (s ^ 2 * o) / shoulderExponent α (sigmaEffPoly o a b j₀ m j₂ ϱ s)
      ∧ shoulderExponent α (s ^ 2 * o) / shoulderExponent α (sigmaEffPoly o a b j₀ m j₂ ϱ s)
        ≤ (leadingCoeff o b j₀ + τ + higherOrderBound a b j₀ m j₂) / o := by
  obtain ⟨s₁, hs₁, hb⟩ := sigmaEffPoly_two_sided_near (a := a) (m := m) (j₂ := j₂) hc hs₀ hϱ
  have hmem : Set.Ioi (0:ℝ) ∩ Set.Iic s₁ ∈ 𝓝[>] (0:ℝ) :=
    Filter.inter_mem self_mem_nhdsWithin (nhdsWithin_le_nhds (Iic_mem_nhds hs₁))
  refine Filter.eventually_of_mem hmem fun s hs => ?_
  obtain ⟨hs0, hs1⟩ := hs
  have hs0' : (0:ℝ) < s := hs0
  obtain ⟨hlo, hhi⟩ := hb s hs0' hs1
  have hgap : 0 < leadingCoeff o b j₀ - τ := by linarith
  have hpos : 0 < sigmaEffPoly o a b j₀ m j₂ ϱ s := lt_of_lt_of_le (by positivity) hlo
  rw [exponentRatio_sigmaEff_eq hα hs0'.ne' ho.ne' hpos.ne']
  have hden : 0 < s ^ 2 * o := by positivity
  constructor
  · rw [le_div_iff₀ hden]
    calc (leadingCoeff o b j₀ - τ) / (2 * o) * (s ^ 2 * o)
        = (leadingCoeff o b j₀ - τ) / 2 * s ^ 2 := by field_simp
      _ ≤ sigmaEffPoly o a b j₀ m j₂ ϱ s := hlo
  · rw [div_le_iff₀ hden]
    calc sigmaEffPoly o a b j₀ m j₂ ϱ s
        ≤ (leadingCoeff o b j₀ + τ + higherOrderBound a b j₀ m j₂) * s ^ 2 := hhi
      _ = (leadingCoeff o b j₀ + τ + higherOrderBound a b j₀ m j₂) / o * (s ^ 2 * o) := by
          field_simp

/-- **The ratio of exponents does not diverge on `eq:sigma-eff`.** This is the negation, on the
paper's own effective diffusion, of the conclusion of `PowerOfS.exponentRatio_tendsto_atTop`:
the lift does not lower the order of the shoulder exponent. -/
theorem exponentRatio_sigmaEff_not_tendsto_atTop {o a b j₀ m j₂ τ s₀ α : ℝ} {ϱ : ℝ → ℝ}
    (hα : α ≠ 0) (ho : 0 < o) (hc : τ < leadingCoeff o b j₀) (hs₀ : 0 < s₀)
    (hϱ : ∀ s, 0 < s → s ≤ s₀ → |ϱ s| ≤ τ * s ^ 2) :
    ¬ Tendsto (fun s => shoulderExponent α (s ^ 2 * o)
        / shoulderExponent α (sigmaEffPoly o a b j₀ m j₂ ϱ s)) (𝓝[>] (0:ℝ)) atTop := by
  intro h
  set M := (leadingCoeff o b j₀ + τ + higherOrderBound a b j₀ m j₂) / o
  have h1 := h.eventually_gt_atTop M
  have h2 := exponentRatio_sigmaEff_eventually_bounded (a := a) (m := m) (j₂ := j₂) hα ho hc hs₀ hϱ
  obtain ⟨s, hgt, _, hle⟩ := (h1.and h2).exists
  exact absurd hle (not_le.mpr hgt)

/-- **The limit of the ratio, when the remainder has one.** If `ϱ(s)/s²` converges to `ϱ₀` as
the positivity map flattens, the ratio of the direct exponent to the lift's converges to
`(c₂ + ϱ₀)/σ_obj² = 1 + (2 b + j₀ + ϱ₀)/σ_obj²`: a constant, whose excess over one is the
leading-order excess of `eq:sigma-eff` over the direct diffusion, in units of the direct
diffusion. -/
theorem exponentRatio_sigmaEff_tendsto {o a b j₀ m j₂ α ϱ₀ : ℝ} {ϱ : ℝ → ℝ} (hα : α ≠ 0)
    (ho : 0 < o) (hlim : Tendsto (fun s => ϱ s / s ^ 2) (𝓝[>] (0:ℝ)) (𝓝 ϱ₀))
    (hne : ∀ᶠ s in 𝓝[>] (0:ℝ), sigmaEffPoly o a b j₀ m j₂ ϱ s ≠ 0) :
    Tendsto (fun s => shoulderExponent α (s ^ 2 * o)
        / shoulderExponent α (sigmaEffPoly o a b j₀ m j₂ ϱ s)) (𝓝[>] (0:ℝ))
      (𝓝 ((leadingCoeff o b j₀ + ϱ₀) / o)) := by
  have hR : Tendsto (fun s : ℝ => s * (2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m + s * j₂))
      (𝓝 0) (𝓝 0) := by
    have hc : Continuous (fun s : ℝ => s * (2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m + s * j₂)) :=
      by fun_prop
    simpa using hc.tendsto 0
  have hmain : Tendsto (fun s : ℝ => (leadingCoeff o b j₀
      + s * (2 * a - 2 * b + (s - 2) * j₀ + (1 - s) * m + s * j₂) + ϱ s / s ^ 2) / o)
      (𝓝[>] (0:ℝ)) (𝓝 ((leadingCoeff o b j₀ + 0 + ϱ₀) / o)) :=
    (((tendsto_const_nhds).add (tendsto_nhdsWithin_of_tendsto_nhds hR)).add hlim).div_const o
  rw [add_zero] at hmain
  refine hmain.congr' ?_
  filter_upwards [self_mem_nhdsWithin, hne] with s hs hne'
  have hs' : (s:ℝ) ≠ 0 := ne_of_gt hs
  rw [exponentRatio_sigmaEff_eq hα hs' ho.ne' hne', sigmaEffPoly_eq]
  field_simp

end Exponent

/-! ## 7. What the leading coefficient is

The `s²` coefficient of `eq:sigma-eff`, `c₂ = σ_obj² + 2 b + j₀`, is the variance of the projected
once-attenuated gradient fluctuation `e_bᵀ ξ_θ + (ḡ ⊙ e_b)ᵀ δθ̃`: the constrained noise plus the
mean constrained gradient carried by the latent weight's fluctuation of the positivity-map slope. Its excess
over the direct value, `2 b + j₀`, is nonnegative whenever the mean-gradient-weighted frozen
coupling `b` is, which is (A6) read at leading order in the prefactor. -/

section Leading

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **The leading coefficient is a variance.** `σ_obj² + 2 b + j₀` is the projected covariance of
`ξ_θ + diag(ḡ) δθ̃`, the gradient fluctuation at leading order in the prefactor: the direct-softplus
noise plus the mean constrained gradient multiplied by the latent weight's fluctuation. -/
theorem leadingCoeff_eq_projDiffusion [IsFiniteMeasure μ] {gθ dth : Ω → Fin d → ℝ}
    (hgθ : SqIntegrableVec μ gθ) (hθ : SqIntegrableVec μ dth) (gbar e : Fin d → ℝ) :
    leadingCoeff (e ⬝ᵥ (covMat μ gθ gθ *ᵥ e)) (gradCoupling gbar (covMat μ dth gθ) e)
        (e ⬝ᵥ ((diagonal gbar * covMat μ dth dth * diagonal gbar) *ᵥ e))
      = projDiffusion μ e (fun ω => gθ ω + diagonal gbar *ᵥ dth ω) := by
  have hG : SqIntegrableVec μ (fun ω => diagonal gbar *ᵥ dth ω) := hθ.mulVec _
  rw [projDiffusion, covMat_add_self hgθ hG, covMat_mulVec_left (diagonal gbar) hθ hgθ,
    covMat_mulVec_right (diagonal gbar) hgθ hθ, covMat_mulVec_left (diagonal gbar) hθ hG,
    covMat_mulVec_right (diagonal gbar) hθ hθ, diagonal_transpose, quadForm_add, quadForm_add,
    quadForm_add]
  have hT : e ⬝ᵥ ((covMat μ gθ dth * diagonal gbar) *ᵥ e)
      = e ⬝ᵥ ((diagonal gbar * covMat μ dth gθ) *ᵥ e) := by
    have h : covMat μ gθ dth * diagonal gbar = (diagonal gbar * covMat μ dth gθ)ᵀ := by
      rw [Matrix.transpose_mul, covMat_transpose, diagonal_transpose]
    rw [h, quadForm_transpose]
  rw [hT, leadingCoeff, gradCoupling, Matrix.mul_assoc]
  ring

/-- The leading coefficient is nonnegative, being a variance. -/
theorem leadingCoeff_nonneg [IsFiniteMeasure μ] {gθ dth : Ω → Fin d → ℝ}
    (hgθ : SqIntegrableVec μ gθ) (hθ : SqIntegrableVec μ dth) (gbar e : Fin d → ℝ) :
    0 ≤ leadingCoeff (e ⬝ᵥ (covMat μ gθ gθ *ᵥ e)) (gradCoupling gbar (covMat μ dth gθ) e)
        (e ⬝ᵥ ((diagonal gbar * covMat μ dth dth * diagonal gbar) *ᵥ e)) := by
  rw [leadingCoeff_eq_projDiffusion hgθ hθ gbar e]
  exact projDiffusion_nonneg (hgθ.add (hθ.mulVec _)) e

/-- The leading jitter scalar `j₀ = e_bᵀ diag(ḡ) V diag(ḡ) e_b` is nonnegative. -/
theorem leadingJitter_nonneg [IsFiniteMeasure μ] {dth : Ω → Fin d → ℝ}
    (hθ : SqIntegrableVec μ dth) (gbar e : Fin d → ℝ) :
    0 ≤ e ⬝ᵥ ((diagonal gbar * covMat μ dth dth * diagonal gbar) *ᵥ e) :=
  quadForm_conj_nonneg (covMat_self_posSemidef hθ)
    (isHermitian_of_transpose_eq (diagonal_transpose gbar)) e

/-- **(A6) at leading order in the prefactor.** If the mean-gradient-weighted frozen coupling
`b = e_bᵀ diag(ḡ) Ξ_θ e_b` is nonnegative, the leading coefficient of the lift's diffusion is at
least the direct value `σ_obj²`; strictly greater if `b > 0`, or if the leading jitter `j₀` is
positive. -/
theorem leadingCoeff_ge_direct_of_gradCoupling_nonneg [IsFiniteMeasure μ]
    {dth : Ω → Fin d → ℝ} (hθ : SqIntegrableVec μ dth) {o b : ℝ} (hb : 0 ≤ b)
    (gbar e : Fin d → ℝ) :
    o ≤ leadingCoeff o b (e ⬝ᵥ ((diagonal gbar * covMat μ dth dth * diagonal gbar) *ᵥ e)) := by
  have := leadingJitter_nonneg hθ gbar e
  unfold leadingCoeff
  linarith

/-- For a coordinate direction `e_b` the mean-gradient-weighted frozen coupling is the product
of the mean constrained gradient at the bias coordinate with the frozen coupling's diagonal
entry there, `b = ḡ_b (Ξ_θ)_{bb}`. -/
theorem gradCoupling_single (gbar : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) (i : Fin d) :
    gradCoupling gbar Ξ (Pi.single i 1) = gbar i * Ξ i i := by
  simp only [gradCoupling, dotProduct, Matrix.mulVec, Pi.single_apply, mul_ite, mul_one,
    mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true, ite_mul, one_mul, zero_mul,
    Matrix.mul_apply, diagonal_apply]
  simp

end Leading

/-! ## 8. Non-vacuity

One-dimensional constants at which every hypothesis of the headline statements holds, so that
none of them is true for want of an instance. -/

section Witness

/-- At unit constrained curvature, unit frozen coupling and unit mean gradient in one dimension,
the coupling term of `eq:sigma-eff` is exactly `2 s²`: the curvature route contributes `2 s³`
and the mean-gradient route `2 s² (1 − s)`. -/
theorem couplingTerm_witness (s : ℝ) :
    crossTerm (preHessian (fun _ : Fin 1 => s) (fun _ => s * (1 - s)) (fun _ => 1) 1)
        ((1 : Matrix (Fin 1) (Fin 1) ℝ) * diagonal fun _ => s) (Pi.single 0 1)
      = 2 * s ^ 2 := by
  rw [couplingTerm_softplus (Matrix.transpose_one), gradCoupling_single]
  have ha : curvCoupling (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 (Pi.single 0 1) = 1 := by
    simp [curvCoupling, Matrix.mulVec, dotProduct]
  rw [ha]
  simp
  ring

/-- The mean-gradient route is open at the witness, so `couplingPoly_isTheta_sq_of_ne`
applies there and the `Θ(s²)` law is not vacuous. -/
theorem gradCoupling_witness_ne_zero :
    gradCoupling (fun _ : Fin 1 => (1:ℝ)) 1 (Pi.single 0 1) ≠ 0 := by
  rw [gradCoupling_single]
  simp

/-- The coupling term at the witness is `Θ(s²)`. -/
theorem couplingPoly_witness_isTheta_sq :
    couplingPoly 1 1 =Θ[𝓝[>] (0:ℝ)] fun s => s ^ 2 :=
  couplingPoly_isTheta_sq_of_ne one_ne_zero

/-- With unit objective noise, unit couplings and unit jitter scalars and no remainder, the lift's
exponent at the witness is `Θ(s⁻²)`: the hypotheses of `liftExponent_sigmaEff_isTheta_inv_sq`
are satisfiable. -/
theorem liftExponent_witness_isTheta_inv_sq :
    (fun s => shoulderExponent 1 (sigmaEffPoly 1 1 1 1 1 1 (fun _ => 0) s)) =Θ[𝓝[>] (0:ℝ)]
      fun s => (s ^ 2)⁻¹ :=
  liftExponent_sigmaEff_isTheta_inv_sq (τ := 0) (s₀ := 1) one_pos
    (by unfold leadingCoeff; norm_num) one_pos (fun s _ _ => by simp)

/-- At the witness the ratio of exponents converges to `4`, the leading coefficient
`1 + 2 + 1` over the unit objective noise: a bounded factor, not a divergence. -/
theorem exponentRatio_witness_tendsto :
    Tendsto (fun s => shoulderExponent 1 (s ^ 2 * 1)
        / shoulderExponent 1 (sigmaEffPoly 1 1 1 1 1 1 (fun _ => 0) s)) (𝓝[>] (0:ℝ)) (𝓝 4) := by
  have h := exponentRatio_sigmaEff_tendsto (o := 1) (a := 1) (b := 1) (j₀ := 1) (m := 1)
    (j₂ := 1) (α := 1) (ϱ₀ := 0) (ϱ := fun _ => 0) one_ne_zero one_pos
    (by simp) ?_
  · have h4 : (leadingCoeff 1 1 1 + 0) / 1 = (4:ℝ) := by unfold leadingCoeff; norm_num
    rw [h4] at h
    exact h
  · obtain ⟨s₁, hs₁, hb⟩ := sigmaEffPoly_two_sided_near (o := 1) (a := 1) (b := 1) (j₀ := 1)
      (m := 1) (j₂ := 1) (τ := 0) (s₀ := 1) (ϱ := fun _ => 0)
      (by unfold leadingCoeff; norm_num) one_pos (fun s _ _ => by simp)
    have hmem : Set.Ioi (0:ℝ) ∩ Set.Iic s₁ ∈ 𝓝[>] (0:ℝ) :=
      Filter.inter_mem self_mem_nhdsWithin (nhdsWithin_le_nhds (Iic_mem_nhds hs₁))
    refine Filter.eventually_of_mem hmem fun s hs => ?_
    obtain ⟨hs0, hs1⟩ := hs
    have hs0' : (0:ℝ) < s := hs0
    obtain ⟨hlo, _⟩ := hb s hs0' hs1
    have : 0 < (leadingCoeff 1 1 1 - 0) / 2 * s ^ 2 := by
      unfold leadingCoeff
      positivity
    exact ne_of_gt (lt_of_lt_of_le this hlo)

end Witness

/-! ## 9. Robustness under (A2) read with a tolerance

The single-prefactor computation above idealizes `ψ'(θ̃_ref) = s` in every coordinate. Under
(A2) read at relative accuracy, `|ψ'(θ̃_ref)_i| ≤ (1 + ε) s`, together with a bound
`|ψ''(θ̃_ref)_i| ≤ c s` on the second derivative (which the softplus and exponential positivity maps
satisfy with `c = 1`), the coupling term of `eq:sigma-eff` is still `O(s²)`, and therefore still
not `Θ(s)`. The bounds are entrywise and carry explicit constants. -/

section Relative

/-- `|uᵀ M w| ≤ ∑ᵢⱼ |uᵢ| |Mᵢⱼ| |wⱼ|`. -/
theorem abs_dotProduct_mulVec_le (u w : Fin d → ℝ) (M : Matrix (Fin d) (Fin d) ℝ) :
    |u ⬝ᵥ (M *ᵥ w)| ≤ ∑ i, ∑ j, |u i| * |M i j| * |w j| := by
  simp only [dotProduct, Matrix.mulVec]
  calc |∑ i, u i * ∑ j, M i j * w j| ≤ ∑ i, |u i * ∑ j, M i j * w j| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |u i| * |∑ j, M i j * w j| := by simp only [abs_mul]
    _ ≤ ∑ i, |u i| * ∑ j, |M i j * w j| := by
        gcongr with i
        exact Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, ∑ j, |u i| * |M i j| * |w j| := by
        simp only [Finset.mul_sum, abs_mul, mul_assoc]

/-- `|(M w)ⱼ| ≤ ∑ₖ |Mⱼₖ| |wₖ|`. -/
theorem abs_mulVec_apply_le (M : Matrix (Fin d) (Fin d) ℝ) (w : Fin d → ℝ) (j : Fin d) :
    |(M *ᵥ w) j| ≤ ∑ k, |M j k| * |w k| := by
  simp only [Matrix.mulVec, dotProduct]
  calc |∑ k, M j k * w k| ≤ ∑ k, |M j k * w k| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ k, |M j k| * |w k| := by simp only [abs_mul]

/-- The curvature-route absolute sum `∑ᵢⱼₖ |eᵢ| |Hθᵢⱼ| |Ξⱼₖ| |eₖ|`. -/
def curvAbsSum (Hθ Ξ : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) : ℝ :=
  ∑ i, ∑ j, |e i| * |Hθ i j| * ∑ k, |Ξ j k| * |e k|

/-- The mean-gradient-route absolute sum `∑ᵢₖ |eᵢ| |ḡᵢ| |Ξᵢₖ| |eₖ|`. -/
def gradAbsSum (gbar : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) : ℝ :=
  ∑ i, ∑ k, (|e i| * |gbar i|) * |Ξ i k| * |e k|

/-- For symmetric `Hθ` the coupling term with general prefactor fields splits into the
curvature route `2 eᵀ D Hθ D Ξ D e` and the mean-gradient route `2 eᵀ diag(p₂ ⊙ ḡ) Ξ D e`, both
written as dot products with `u = p ⊙ e`. -/
theorem crossTerm_preHessian_eq {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    (p p₂ gbar e : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) :
    crossTerm (preHessian p p₂ gbar Hθ) (Ξ * diagonal p) e
      = 2 * ((fun i => p i * e i) ⬝ᵥ
            (Hθ *ᵥ (fun j => p j * (Ξ *ᵥ fun k => p k * e k) j)))
        + 2 * ((fun i => e i * (p₂ i * gbar i)) ⬝ᵥ (Ξ *ᵥ fun k => p k * e k)) := by
  have hsym : (preHessian p p₂ gbar Hθ).IsHermitian :=
    isHermitian_of_transpose_eq (preHessian_transpose hH)
  rw [crossTerm_eq_two_mul hsym, preHessian, Matrix.add_mul, quadForm_add]
  have hu : (diagonal p *ᵥ e) = fun k => p k * e k := by
    funext k
    rw [mulVec_diagonal]
  have h1 : e ⬝ᵥ ((diagonal p * Hθ * diagonal p * (Ξ * diagonal p)) *ᵥ e)
      = (fun i => p i * e i) ⬝ᵥ (Hθ *ᵥ (fun j => p j * (Ξ *ᵥ fun k => p k * e k) j)) := by
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
      ← Matrix.mulVec_mulVec, hu]
    simp only [dotProduct, mulVec_diagonal]
    refine Finset.sum_congr rfl fun i _ => ?_
    have : (fun j => p j * (Ξ *ᵥ fun k => p k * e k) j) = diagonal p *ᵥ (Ξ *ᵥ fun k => p k * e k) := by
      funext j
      rw [mulVec_diagonal]
    rw [this]
    ring
  have h2 : e ⬝ᵥ ((diagonal (fun i => p₂ i * gbar i) * (Ξ * diagonal p)) *ᵥ e)
      = (fun i => e i * (p₂ i * gbar i)) ⬝ᵥ (Ξ *ᵥ fun k => p k * e k) := by
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, hu]
    simp only [dotProduct, mulVec_diagonal]
    refine Finset.sum_congr rfl fun i _ => ?_
    ring
  rw [h1, h2]
  ring

/-- **The entrywise bound on the coupling term for general prefactor fields.** If every
coordinate of `ψ'(θ̃_ref)` is at most `P` in absolute value and every coordinate of `ψ''(θ̃_ref)`
at most `Q`, then

  `|2 e_bᵀ H_θ̃ Σ_ξ e_b| ≤ 2 (P³ · curvAbsSum + Q P · gradAbsSum)`.

No single-prefactor idealization enters: `P` and `Q` are bounds, not values. -/
theorem abs_crossTerm_preHessian_le {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    {p p₂ : Fin d → ℝ} (gbar e : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) {P Q : ℝ}
    (hP : 0 ≤ P) (hQ : 0 ≤ Q) (hp : ∀ i, |p i| ≤ P) (hp₂ : ∀ i, |p₂ i| ≤ Q) :
    |crossTerm (preHessian p p₂ gbar Hθ) (Ξ * diagonal p) e|
      ≤ 2 * (P ^ 3 * curvAbsSum Hθ Ξ e + Q * P * gradAbsSum gbar Ξ e) := by
  rw [crossTerm_preHessian_eq hH]
  set u : Fin d → ℝ := fun k => p k * e k with hu_def
  have hu : ∀ k, |u k| ≤ P * |e k| := fun k => by
    rw [hu_def, abs_mul]
    exact mul_le_mul_of_nonneg_right (hp k) (abs_nonneg _)
  set w : Fin d → ℝ := fun j => p j * (Ξ *ᵥ u) j with hw_def
  have hw : ∀ j, |w j| ≤ P * (P * ∑ k, |Ξ j k| * |e k|) := fun j => by
    rw [hw_def, abs_mul]
    have h1 : |(Ξ *ᵥ u) j| ≤ ∑ k, |Ξ j k| * |u k| := abs_mulVec_apply_le Ξ u j
    have h2 : ∑ k, |Ξ j k| * |u k| ≤ ∑ k, |Ξ j k| * (P * |e k|) := by
      gcongr with k
      exact hu k
    have h3 : ∑ k, |Ξ j k| * (P * |e k|) = P * ∑ k, |Ξ j k| * |e k| := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun k _ => by ring
    calc |p j| * |(Ξ *ᵥ u) j| ≤ P * (∑ k, |Ξ j k| * |u k|) :=
          mul_le_mul (hp j) h1 (abs_nonneg _) hP
      _ ≤ P * (∑ k, |Ξ j k| * (P * |e k|)) := by gcongr
      _ = P * (P * ∑ k, |Ξ j k| * |e k|) := by rw [h3]
  set v : Fin d → ℝ := fun i => e i * (p₂ i * gbar i) with hv_def
  have hv : ∀ i, |v i| ≤ |e i| * (Q * |gbar i|) := fun i => by
    rw [hv_def, abs_mul, abs_mul]
    gcongr
    exact hp₂ i
  have hT1 : |u ⬝ᵥ (Hθ *ᵥ w)| ≤ P ^ 3 * curvAbsSum Hθ Ξ e := by
    calc |u ⬝ᵥ (Hθ *ᵥ w)| ≤ ∑ i, ∑ j, |u i| * |Hθ i j| * |w j| := abs_dotProduct_mulVec_le u w Hθ
      _ ≤ ∑ i, ∑ j, (P * |e i|) * |Hθ i j| * (P * (P * ∑ k, |Ξ j k| * |e k|)) := by
          gcongr with i _ j _
          · exact hu i
          · exact hw j
      _ = P ^ 3 * curvAbsSum Hθ Ξ e := by
          rw [curvAbsSum, Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun j _ => ?_
          ring
  have hT2 : |v ⬝ᵥ (Ξ *ᵥ u)| ≤ Q * P * gradAbsSum gbar Ξ e := by
    calc |v ⬝ᵥ (Ξ *ᵥ u)| ≤ ∑ i, ∑ k, |v i| * |Ξ i k| * |u k| := abs_dotProduct_mulVec_le v u Ξ
      _ ≤ ∑ i, ∑ k, (|e i| * (Q * |gbar i|)) * |Ξ i k| * (P * |e k|) := by
          gcongr with i _ k _
          · exact hv i
          · exact hu k
      _ = Q * P * gradAbsSum gbar Ξ e := by
          rw [gradAbsSum, Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun k _ => ?_
          ring
  calc |2 * (u ⬝ᵥ (Hθ *ᵥ w)) + 2 * (v ⬝ᵥ (Ξ *ᵥ u))|
      ≤ |2 * (u ⬝ᵥ (Hθ *ᵥ w))| + |2 * (v ⬝ᵥ (Ξ *ᵥ u))| := abs_add_le _ _
    _ = 2 * |u ⬝ᵥ (Hθ *ᵥ w)| + 2 * |v ⬝ᵥ (Ξ *ᵥ u)| := by rw [abs_mul, abs_mul, abs_two]
    _ ≤ 2 * (P ^ 3 * curvAbsSum Hθ Ξ e) + 2 * (Q * P * gradAbsSum gbar Ξ e) := by gcongr
    _ = 2 * (P ^ 3 * curvAbsSum Hθ Ξ e + Q * P * gradAbsSum gbar Ξ e) := by ring

theorem curvAbsSum_nonneg (Hθ Ξ : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) :
    0 ≤ curvAbsSum Hθ Ξ e := by
  unfold curvAbsSum
  positivity

theorem gradAbsSum_nonneg (gbar : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) :
    0 ≤ gradAbsSum gbar Ξ e := by
  unfold gradAbsSum
  positivity

/-- **The coupling term is `O(s²)` under (A2) read with a tolerance.** Let the prefactor field
and the second-derivative field depend on the positivity-map level `s`, with `|ψ'(θ̃_ref)_i| ≤ (1 + ε) s`
— (A2) at relative accuracy `ε` — and `|ψ''(θ̃_ref)_i| ≤ c s` on the band `0 < s ≤ s₀`. Then the
coupling term of `eq:sigma-eff` is `O(s²)` as the positivity map flattens, with the explicit constant
`2 ((1 + ε)³ · curvAbsSum + c (1 + ε) · gradAbsSum)`. -/
theorem couplingTerm_isBigO_sq_of_relative {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    (gbar e : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) {p p₂ : ℝ → Fin d → ℝ} {ε c s₀ : ℝ}
    (hε : 0 ≤ ε) (hc : 0 ≤ c) (hs₀ : 0 < s₀)
    (hp : ∀ s, 0 < s → s ≤ s₀ → ∀ i, |p s i| ≤ (1 + ε) * s)
    (hp₂ : ∀ s, 0 < s → s ≤ s₀ → ∀ i, |p₂ s i| ≤ c * s) :
    (fun s => crossTerm (preHessian (p s) (p₂ s) gbar Hθ) (Ξ * diagonal (p s)) e)
      =O[𝓝[>] (0:ℝ)] fun s => s ^ 2 := by
  have hmem : Set.Ioi (0:ℝ) ∩ Set.Iic (min s₀ 1) ∈ 𝓝[>] (0:ℝ) :=
    Filter.inter_mem self_mem_nhdsWithin
      (nhdsWithin_le_nhds (Iic_mem_nhds (lt_min hs₀ one_pos)))
  rw [isBigO_iff]
  refine ⟨2 * ((1 + ε) ^ 3 * curvAbsSum Hθ Ξ e + c * (1 + ε) * gradAbsSum gbar Ξ e),
    Filter.eventually_of_mem hmem fun s hs => ?_⟩
  obtain ⟨hs0, hs1⟩ := hs
  have hs0' : (0:ℝ) < s := hs0
  have hsA : s ≤ s₀ := le_trans hs1 (min_le_left _ _)
  have hsB : s ≤ 1 := le_trans hs1 (min_le_right _ _)
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (sq_nonneg s)]
  have hA := curvAbsSum_nonneg Hθ Ξ e
  have hB := gradAbsSum_nonneg gbar Ξ e
  have hbound := abs_crossTerm_preHessian_le hH gbar e Ξ (P := (1 + ε) * s) (Q := c * s)
    (by positivity) (by positivity) (hp s hs0' hsA) (hp₂ s hs0' hsA)
  refine le_trans hbound ?_
  have h1 : ((1 + ε) * s) ^ 3 * curvAbsSum Hθ Ξ e ≤ (1 + ε) ^ 3 * curvAbsSum Hθ Ξ e * s ^ 2 := by
    have : ((1 + ε) * s) ^ 3 = (1 + ε) ^ 3 * s ^ 2 * s := by ring
    rw [this]
    have hle : (1 + ε) ^ 3 * s ^ 2 * s ≤ (1 + ε) ^ 3 * s ^ 2 * 1 := by
      gcongr
    nlinarith
  have h2 : c * s * ((1 + ε) * s) * gradAbsSum gbar Ξ e
      ≤ c * (1 + ε) * gradAbsSum gbar Ξ e * s ^ 2 := by
    have : c * s * ((1 + ε) * s) * gradAbsSum gbar Ξ e
        = c * (1 + ε) * gradAbsSum gbar Ξ e * s ^ 2 := by ring
    rw [this]
  nlinarith

/-- **The coupling term is not `Θ(s)` under (A2) read with a tolerance**: the conclusion of
`couplingPoly_not_isTheta_id` survives the relative reading of the single-prefactor
idealization. -/
theorem couplingTerm_not_isTheta_id_of_relative {Hθ : Matrix (Fin d) (Fin d) ℝ} (hH : Hθᵀ = Hθ)
    (gbar e : Fin d → ℝ) (Ξ : Matrix (Fin d) (Fin d) ℝ) {p p₂ : ℝ → Fin d → ℝ} {ε c s₀ : ℝ}
    (hε : 0 ≤ ε) (hc : 0 ≤ c) (hs₀ : 0 < s₀)
    (hp : ∀ s, 0 < s → s ≤ s₀ → ∀ i, |p s i| ≤ (1 + ε) * s)
    (hp₂ : ∀ s, 0 < s → s ≤ s₀ → ∀ i, |p₂ s i| ≤ c * s) :
    ¬ ((fun s => crossTerm (preHessian (p s) (p₂ s) gbar Hθ) (Ξ * diagonal (p s)) e)
        =Θ[𝓝[>] (0:ℝ)] fun s => s) :=
  not_isTheta_id_of_isBigO_sq (couplingTerm_isBigO_sq_of_relative hH gbar e Ξ hε hc hs₀ hp hp₂)

/-- The softplus positivity map satisfies the second-derivative bound of the relative statements with
`c = 1`: `|ψ''| = ψ' (1 − ψ') ≤ ψ'`, so `|ψ''(θ̃_ref)_i| ≤ (1 + ε) s` wherever
`|ψ'(θ̃_ref)_i| ≤ (1 + ε) s`. -/
theorem softplus_second_derivative_bound (w : ℝ) :
    |logistic w * (1 - logistic w)| ≤ |logistic w| := by
  have h0 := logistic_pos w
  have h1 := logistic_lt_one w
  rw [abs_mul, abs_of_pos h0, abs_of_pos (by linarith : (0:ℝ) < 1 - logistic w)]
  nlinarith

end Relative

end IcnnLift
