import Mathlib

/-!
# Case (iii) of Theorem 1 (`thm:joint-necessity`): conditional independence zeroes the population
cross-covariance

Theorem 1 of the paper (stated in `docs/paper/v4/04_mechanism.tex` and restated with its proof in
`docs/paper/v4/A1_proofs.tex`, `app:proof-joint-necessity`) lists three deletions, each of which
zeroes the slack-channel cross-covariance reading. Cases (i) and (ii) are finite-sample identities
and are machine-checked in `CrossCov.lean`. Case (iii) is different in kind: it is a statement
about the *population* object that the estimator targets,
`Σ_slack ≡ E[δθ̃ (δg)ᵀ | φ]`, and it is what this file proves.

The paper's argument is a chain of two equalities. If the produced latent iterate `θ̃` and the
loss gradient `g` are independent conditional on the body parameters `φ`, then their centred
fluctuations `δθ̃ = θ̃ - E[θ̃ | φ]` and `δg = g - E[g | φ]` are conditionally independent as well,
being fixed measurable functions of `θ̃` and of `g` respectively; the conditional expectation of
their outer product therefore factorizes as `E[δθ̃ (δg)ᵀ | φ] = E[δθ̃ | φ] (E[δg | φ])ᵀ`; and both
factors vanish because the fluctuations are centred by construction. Every link of that chain is
proved here, and proved at both levels: unconditionally, by `indepFun_centered`,
`popMean_centered`, `popCrossCov_centered_of_indepFun` and
`popCrossCov_centered_eq_zero_of_indepFun`; and conditionally, by
`condPopCrossCov_centered_ae_factorizes`, `condPopMean_condExp_centered_ae_eq_zero` and
`condPopCrossCov_condExp_centered_ae_eq_zero`. Nothing in case (iii) is asymptotic or "to leading
order", so no remainder term arises and none is assumed away: the results below are exact
identities, and the only hypotheses carried are independence, integrability, and — where a constant
has to be integrated — that the measure is a probability measure.

Two modelling choices deserve comment, because they are what make the Lean statements the paper's
statements rather than weaker cousins. First, the cross-covariance is defined entrywise as
`Σ i j = ∫ X ω i * Y ω j ∂μ`, which needs no norm on the space of matrices; `popCrossCov_of_indepFun`
nevertheless routes through the genuinely vector-valued Bochner integral of the outer product,
because `popCrossCov_eq_integral_outerProd` identifies the entrywise definition with
`∫ ω, X ω (Y ω)ᵀ ∂μ` whenever that integral exists, and the factorization is then
`ProbabilityTheory.IndepFun.integral_bilin` applied to the outer product `outerProd`, exhibited
here as a genuine continuous bilinear map. So the conclusion is a matrix identity, not a family of
scalar identities. Second, the conditioning on `φ` is not modelled by fixing `φ`: it is modelled by
mathlib's `ProbabilityTheory.CondIndepFun` and the disintegration kernel
`ProbabilityTheory.condExpKernel μ m'`, whose fibre means `popMean_condExpKernel_ae_eq_condExp`
identifies almost everywhere with the conditional expectations `μ[· | m']` that the paper writes
`E[· | φ]`. The conditional conclusions are stated in both readings — with the fibre means, and
with `μ[· | m']` itself — so that the second can be set beside the paper's display without any
identification carried out by hand.

Two further sections keep the development honest. `condIndepFun_bot_of_indepFun` embeds ordinary
independence into conditional independence given the trivial σ-algebra, and
`condExpKernel_bot_apply` identifies every fibre of the trivial conditioning with `μ` itself, so
that at `m' = ⊥` the fibre measures and fibre means appearing in the conditional theorems are
literally `μ` and the unconditional means — the two halves of the file are one development and not
two, at the level of the statements and not only of the hypotheses. The final section then
exhibits explicit instances, because
an implication proves nothing if its hypotheses are unsatisfiable or its conclusion automatic.
`exists_centered_popCrossCov_ne_zero` produces a probability space on which the centred population
cross-covariance and `σ_Jac²` are *not* zero, which is what shows that independence, and not the
shape of the definitions or the centring alone, is what produces the zero; and
`condPopCrossCov_coinPair_ae_eq_zero` instantiates the headline conditional theorem at a witness
that satisfies its entire hypothesis stack with both random vectors non-constant.

The vocabulary is aligned with `CrossCov.lean`: fluctuations are `Fin d → ℝ`-valued, and the outer
product is `Matrix.vecMulVec`, so that `popCrossCov` is the population limit of the summands of
`IcnnLift.crossCovEstimator`.

## Results
* `outerProd` — the outer product `(v, w) ↦ v wᵀ` as a continuous bilinear map, with unit bound.
* `of_outerProd` — read as a matrix, that outer product is `Matrix.vecMulVec`, the summand of the
  estimator in `CrossCov.lean`.
* `popMean` — the population mean `E[X]` of an `ℝ^d`-valued random vector.
* `popCrossCov` — the population cross-covariance `Σ = E[X Yᵀ]`, entrywise.
* `sigmaJacSq` — the paper's `σ_Jac² ≡ tr Σ_slack` (`04_mechanism.tex`, `eq:cross-cov`), and
  `sigmaJacSq_eq_sum`, which writes it as the sum of the diagonal second moments.
* `popCrossCov_eq_integral_outerProd` — the entrywise definition is the Bochner integral of the
  outer product whenever the latter is integrable.
* `popCrossCov_of_indepFun` — the paper's first equality: independence factorizes the population
  cross-covariance, `E[X Yᵀ] = E[X] (E[Y])ᵀ`.
* `popCrossCov_eq_zero_of_indepFun_of_centered` — the unconditional zero-mean statement:
  independent, integrable, centred `X` and `Y` have `E[X Yᵀ] = 0` as a matrix.
* `indepFun_centered` — centring preserves independence, which is the paper's step from
  independence of `(θ̃, g)` to independence of the fluctuations `(δθ̃, δg)`.
* `popMean_centered` — the centred fluctuation has zero mean.
* `popCrossCov_centered_of_indepFun` — the factorization with the centred fluctuations in place,
  `E[δX (δY)ᵀ] = E[δX] (E[δY])ᵀ`.
* `popCrossCov_centered_eq_zero_of_indepFun` — the centring version the paper actually uses: for
  arbitrary integrable independent `X` and `Y`, `E[(X - E X)(Y - E Y)ᵀ] = 0`.
* `sigmaJacSq_eq_zero_of_indepFun_of_centered`, `sigmaJacSq_centered_eq_zero_of_indepFun` —
  the corollary in the paper's own terms: `σ_Jac² = tr Σ_slack = 0`.
* `ae_indepFun_of_condIndepFun` — conditional independence gives ordinary independence under the
  disintegration kernel, for almost every realization of the conditioning σ-algebra.
* `popMean_condExpKernel_ae_eq_condExp` — the fibre mean is the conditional expectation.
* `condPopMean_condExp_centered_ae_eq_zero` — the paper's second link, conditionally:
  `E[δθ̃ | φ] = 0` almost surely.
* `condPopCrossCov_centered_ae_factorizes` — the paper's first link, conditionally:
  `E[δθ̃ (δg)ᵀ | φ] = E[δθ̃ | φ] (E[δg | φ])ᵀ` almost surely.
* `condPopCrossCov_centered_ae_eq_zero` — case (iii) with the centring written as fibre means of
  the disintegration kernel: `Σ_slack = 0` almost surely.
* `condSigmaJacSq_centered_ae_eq_zero` — hence `σ_Jac² = 0` almost surely.
* `condPopCrossCov_condExp_centered_ae_eq_zero` — case (iii) as the paper writes it, with the
  centring by the conditional expectations `μ[· | m']` themselves, and
  `condSigmaJacSq_condExp_centered_ae_eq_zero` for the trace.
* `condIndepFun_bot_of_indepFun` — independence is conditional independence given `⊥`, so the
  unconditional results above are the trivial-conditioning case of the conditional ones.
* `condExpKernel_bot_apply` — every fibre of the trivial conditioning is `μ` itself, so that
  specialization is an identity of statements and not only of hypotheses.
* `fairCoin`, `signVec`, `integral_fairCoin`, `popMean_fairCoin_signVec`,
  `popCrossCov_fairCoin_signVec`, `sigmaJacSq_fairCoin_signVec` — an explicit probability space and
  an explicit centred random vector whose cross-covariance is the all-ones matrix and whose
  `σ_Jac²` is `d`.
* `exists_centered_popCrossCov_ne_zero` — hence the centred population cross-covariance is not
  identically zero: the independence hypothesis of the theorems above is load-bearing.
* `coinPair`, `indepFun_coinPair`, `condIndepFun_coinPair`, `condPopCrossCov_coinPair_ae_eq_zero` —
  the conditional hypothesis stack instantiated with both random vectors non-constant.
* `condIndepFun_coinPair_snd` — a non-trivial conditioning σ-algebra is admissible as well.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The hypotheses carried, and why each is a hypothesis rather than a theorem:

* `Integrable X μ` and `Integrable Y μ`. These are needed for the means and the cross-covariance to
  exist at all, and are exactly the hypotheses of `ProbabilityTheory.IndepFun.integral_bilin`. They
  are not a weakening: without them the paper's `E[δθ̃ | φ]` is undefined.
* `[IsProbabilityMeasure μ]`, in the centring statements only. It is used once, to evaluate
  `∫ ω, c ∂μ = c` for the constant `c = E[X]`, and it holds for the paper's conditional law by
  construction — the disintegration kernel `condExpKernel μ m'` is a Markov kernel, so each of its
  fibres is a probability measure. The zero-mean statements, which take vanishing means as a
  hypothesis instead of producing them by centring, need no such assumption; the bare factorization
  `popCrossCov_centered_of_indepFun` needs only `[IsFiniteMeasure μ]`, so that the constant being
  subtracted is integrable.
* `Measurable X`, `Measurable Y`, `[StandardBorelSpace Ω]`, `[IsFiniteMeasure μ]` and `m' ≤ mΩ`, in
  the conditional statements only. These are the standing hypotheses under which mathlib defines
  `ProbabilityTheory.CondIndepFun` and `ProbabilityTheory.condExpKernel`, and under which the
  disintegration exists; they are not restrictions the paper's setting violates. That they are
  jointly satisfiable, and satisfiable with both random vectors non-constant, is not asserted but
  proved, by `condIndepFun_coinPair` and `condPopCrossCov_coinPair_ae_eq_zero`.

What this file does **not** prove, and where it belongs instead:

* The rest of case (iii) — that the finite-window estimator is unbiased for this population zero
  and converges to it almost surely, with `O(T^{-1/2})` finite-window fluctuations — is a
  statement about the estimator and not about the population cross-covariance. It is assigned to
  `CrossCovSLLN.lean` and is not attempted here.
* The paper's remark that, under (A1), conditional independence at a fixed step together with
  cross-iteration independence of the i.i.d. batches makes the *whole window collections*
  `{θ̃⁽ˢ⁾}` and `{g⁽ˢ⁾}` conditionally independent is a statement about a family of random
  variables across the window. This file proves the population identity for a single pair, which
  is the level at which case (iii) states its conclusion; the window-level independence transfer
  is not formalized here.
* No claim is made here that the paper's `δθ̃` and `δg` are the specific random vectors arising
  from the SGD model of (A1). The results are proved for arbitrary `ℝ^d`-valued random vectors, so
  they apply to those in particular, but the identification is a modelling step outside this file.
* The two witnesses of the final section are not claimed to be exhaustive. The non-degenerate
  conditional witness `condIndepFun_coinPair` conditions on the trivial σ-algebra, and the
  non-trivially conditioned witness `condIndepFun_coinPair_snd` obtains its conditional
  independence from the second variable being measurable with respect to the conditioning
  σ-algebra; a single instance that is both non-trivially conditioned and independent for a
  non-degenerate reason is not constructed. Nothing in the theorems depends on this — the witnesses
  are there to refute vacuity, and each refutes it on its own axis. The fibres of the trivial
  conditioning are no longer a gap: `condExpKernel_bot_apply` proves that every fibre
  `condExpKernel μ ⊥ ω` is `μ` itself, so the conditional laws of the coin-pair witness are
  `coinPair` and its fibre-level random vectors are non-constant by
  `signVec_true_ne_signVec_false`.
-/

open MeasureTheory ProbabilityTheory Matrix
open scoped ProbabilityTheory

namespace IcnnLift

variable {d : ℕ}

/-! ### The outer product as a continuous bilinear map -/

/-- The outer product `(v, w) ↦ v wᵀ` of two vectors in `ℝ^d`, as a bilinear map into the space of
`d × d` arrays. Its expectation is what the paper calls the cross-covariance; the continuous
version, which is the one the integration lemmas need, is `outerProd` below. -/
noncomputable def outerProdₗ (d : ℕ) :
    (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ) →ₗ[ℝ] (Fin d → Fin d → ℝ) :=
  LinearMap.mk₂ ℝ (fun v w i j => v i * w j)
    (fun _ _ _ => by ext i j; simp [add_mul])
    (fun _ _ _ => by ext i j; simp [mul_assoc])
    (fun _ _ _ => by ext i j; simp [mul_add])
    (fun _ _ _ => by ext i j; simp [mul_left_comm])

/-- The outer product `(v, w) ↦ v wᵀ` as a **continuous** bilinear map. Continuity is what makes
`ProbabilityTheory.IndepFun.integral_bilin` applicable, and hence what turns the paper's
factorization of `E[δθ̃ (δg)ᵀ]` into a theorem about a Bochner integral rather than a formal
manipulation of entries. The operator bound is `1` in the supremum norms, since
`|v i * w j| = |v i| * |w j| ≤ ‖v‖ * ‖w‖` coordinatewise. -/
noncomputable def outerProd (d : ℕ) :
    (Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] (Fin d → Fin d → ℝ) :=
  LinearMap.mkContinuous₂ (outerProdₗ d) 1 <| by
    intro v w
    rw [pi_norm_le_iff_of_nonneg (by positivity)]
    intro i
    rw [pi_norm_le_iff_of_nonneg (by positivity)]
    intro j
    have hv : ‖v i‖ ≤ ‖v‖ := norm_le_pi_norm v i
    have hw : ‖w j‖ ≤ ‖w‖ := norm_le_pi_norm w j
    have hentry : ‖(outerProdₗ d v w) i j‖ = ‖v i‖ * ‖w j‖ := by
      simp [outerProdₗ, LinearMap.mk₂_apply, norm_mul]
    rw [hentry, one_mul]
    exact mul_le_mul hv hw (norm_nonneg _) (norm_nonneg _)

@[simp] theorem outerProd_apply (v w : Fin d → ℝ) (i j : Fin d) :
    outerProd d v w i j = v i * w j := rfl

/-- The outer product, read as a matrix, is `Matrix.vecMulVec` — the same object that
`IcnnLift.crossCovEstimator` averages over the trailing window in `CrossCov.lean`. -/
theorem of_outerProd (v w : Fin d → ℝ) :
    Matrix.of (outerProd d v w) = Matrix.vecMulVec v w := by
  ext i j
  simp [Matrix.vecMulVec_apply]

/-! ### The population mean and the population cross-covariance -/

section Population

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The population mean `E[X]` of an `ℝ^d`-valued random vector, as a Bochner integral. Under the
conditional law of the paper this is `E[θ̃ | φ]`, respectively `E[g | φ]`. -/
noncomputable def popMean (μ : Measure Ω) (X : Ω → Fin d → ℝ) : Fin d → ℝ := ∫ ω, X ω ∂μ

/-- The population cross-covariance `Σ = E[X Yᵀ]` of a pair of `ℝ^d`-valued random vectors, defined
entrywise. This is the paper's `Σ_slack` of `eq:cross-cov` once `X` is the iterate fluctuation
`δθ̃` and `Y` is the gradient fluctuation `δg`. -/
noncomputable def popCrossCov (μ : Measure Ω) (X Y : Ω → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => ∫ ω, X ω i * Y ω j ∂μ

/-- The paper's slack-channel diffusion coefficient `σ_Jac² ≡ tr E[δθ̃ (δg)ᵀ]`
(`04_mechanism.tex`, `eq:cross-cov`), the trace of the population cross-covariance. -/
noncomputable def sigmaJacSq (μ : Measure Ω) (X Y : Ω → Fin d → ℝ) : ℝ :=
  (popCrossCov μ X Y).trace

/-- `σ_Jac²` is the sum of the diagonal second moments, as the paper's trace notation says. -/
theorem sigmaJacSq_eq_sum (μ : Measure Ω) (X Y : Ω → Fin d → ℝ) :
    sigmaJacSq μ X Y = ∑ i, ∫ ω, X ω i * Y ω i ∂μ := rfl

/-- The entrywise definition of the population cross-covariance agrees with the Bochner integral of
the vector-valued outer product, whenever that integral exists. This is what licenses reading
`popCrossCov` as `E[X Yᵀ]` rather than as a family of unrelated scalar integrals. -/
theorem popCrossCov_eq_integral_outerProd {X Y : Ω → Fin d → ℝ}
    (h : Integrable (fun ω => outerProd d (X ω) (Y ω)) μ) :
    popCrossCov μ X Y = Matrix.of (∫ ω, outerProd d (X ω) (Y ω) ∂μ) := by
  have hrow : ∀ i, Integrable (fun ω => outerProd d (X ω) (Y ω) i) μ := fun i => h.eval i
  ext i j
  rw [popCrossCov, Matrix.of_apply, Matrix.of_apply, eval_integral hrow i,
    eval_integral (fun j => (hrow i).eval j) j]
  simp

/-! ### The unconditional statement -/

/-- **The paper's first equality.** If `X` and `Y` are independent and integrable, the population
cross-covariance factorizes into the outer product of the means,
`E[X Yᵀ] = E[X] (E[Y])ᵀ`.

The proof is `ProbabilityTheory.IndepFun.integral_bilin` applied to the continuous bilinear map
`outerProd`, so the factorization happens at the level of the vector-valued Bochner integral and is
then read off entrywise; it is not an entrywise argument dressed up as a matrix one. -/
theorem popCrossCov_of_indepFun {X Y : Ω → Fin d → ℝ}
    (hindep : IndepFun X Y μ) (hX : Integrable X μ) (hY : Integrable Y μ) :
    popCrossCov μ X Y = Matrix.vecMulVec (popMean μ X) (popMean μ Y) := by
  have hint : Integrable (fun ω => outerProd d (X ω) (Y ω)) μ :=
    hindep.integrable_bilin hX hY (outerProd d)
  rw [popCrossCov_eq_integral_outerProd hint,
    hindep.integral_bilin (𝕜 := ℝ) hX hY (outerProd d)]
  ext i j
  simp [popMean, Matrix.vecMulVec_apply]

/-- **The unconditional zero-mean statement.** If `X` and `Y` are independent and integrable and
`X` has zero mean, then `E[X Yᵀ] = 0` as a matrix. This is the paper's chain
`E[δθ̃ (δg)ᵀ] = E[δθ̃] (E[δg])ᵀ = 0` with the second equality supplied by the centring. -/
theorem popCrossCov_eq_zero_of_indepFun_of_centered {X Y : Ω → Fin d → ℝ}
    (hindep : IndepFun X Y μ) (hX : Integrable X μ) (hY : Integrable Y μ)
    (hX0 : popMean μ X = 0) : popCrossCov μ X Y = 0 := by
  rw [popCrossCov_of_indepFun hindep hX hY, hX0, Matrix.zero_vecMulVec]

/-- The corollary in the paper's own terms: under the hypotheses of case (iii) the slack-channel
diffusion coefficient `σ_Jac² = tr Σ_slack` vanishes. -/
theorem sigmaJacSq_eq_zero_of_indepFun_of_centered {X Y : Ω → Fin d → ℝ}
    (hindep : IndepFun X Y μ) (hX : Integrable X μ) (hY : Integrable Y μ)
    (hX0 : popMean μ X = 0) : sigmaJacSq μ X Y = 0 := by
  rw [sigmaJacSq, popCrossCov_eq_zero_of_indepFun_of_centered hindep hX hY hX0,
    Matrix.trace_zero]

/-! ### The centring version the paper uses -/

/-- **Centring preserves independence.** This is the paper's step from "`θ̃` and `g` are independent
conditional on `φ`" to "the centred fluctuations `δθ̃` and `δg` are conditionally independent":
each fluctuation is a fixed measurable function — translation by a constant — of one of the two
original variables. -/
theorem indepFun_centered {X Y : Ω → Fin d → ℝ} (hindep : IndepFun X Y μ) (a b : Fin d → ℝ) :
    IndepFun (fun ω => X ω - a) (fun ω => Y ω - b) μ :=
  hindep.comp (φ := fun v => v - a) (ψ := fun v => v - b) (by fun_prop) (by fun_prop)

/-- The centred fluctuation has zero mean: `E[X - E X] = 0`. This is the paper's "both factors are
centred by construction", derived rather than assumed. -/
theorem popMean_centered [IsProbabilityMeasure μ] {X : Ω → Fin d → ℝ} (hX : Integrable X μ) :
    popMean μ (fun ω => X ω - popMean μ X) = 0 := by
  have h : ∫ ω, (X ω - popMean μ X) ∂μ = (∫ ω, X ω ∂μ) - popMean μ X := by
    rw [integral_sub hX (integrable_const _), integral_const]
    simp
  simpa [popMean] using h

/-- **Case (iii) of Theorem 1, unconditional form.** For arbitrary integrable independent `X` and
`Y` on a probability space, the population cross-covariance of the centred fluctuations vanishes:
`E[(X - E X)(Y - E Y)ᵀ] = 0`.

This is the statement the paper actually uses, since `δθ̃` and `δg` are defined by centring. Both
of its ingredients are theorems above: `indepFun_centered` carries independence to the
fluctuations, and `popMean_centered` supplies the vanishing factor. -/
theorem popCrossCov_centered_eq_zero_of_indepFun [IsProbabilityMeasure μ] {X Y : Ω → Fin d → ℝ}
    (hindep : IndepFun X Y μ) (hX : Integrable X μ) (hY : Integrable Y μ) :
    popCrossCov μ (fun ω => X ω - popMean μ X) (fun ω => Y ω - popMean μ Y) = 0 :=
  popCrossCov_eq_zero_of_indepFun_of_centered (indepFun_centered hindep _ _)
    (hX.sub (integrable_const _)) (hY.sub (integrable_const _)) (popMean_centered hX)

/-- The centred factorization written out, `E[δX (δY)ᵀ] = E[δX] (E[δY])ᵀ`: the paper's first
equality with the centred fluctuations in place. -/
theorem popCrossCov_centered_of_indepFun [IsFiniteMeasure μ] {X Y : Ω → Fin d → ℝ}
    (hindep : IndepFun X Y μ) (hX : Integrable X μ) (hY : Integrable Y μ) :
    popCrossCov μ (fun ω => X ω - popMean μ X) (fun ω => Y ω - popMean μ Y) =
      Matrix.vecMulVec (popMean μ (fun ω => X ω - popMean μ X))
        (popMean μ (fun ω => Y ω - popMean μ Y)) :=
  popCrossCov_of_indepFun (indepFun_centered hindep _ _)
    (hX.sub (integrable_const _)) (hY.sub (integrable_const _))

/-- The corollary in the paper's own terms, centring version: `σ_Jac² = tr Σ_slack = 0`. -/
theorem sigmaJacSq_centered_eq_zero_of_indepFun [IsProbabilityMeasure μ] {X Y : Ω → Fin d → ℝ}
    (hindep : IndepFun X Y μ) (hX : Integrable X μ) (hY : Integrable Y μ) :
    sigmaJacSq μ (fun ω => X ω - popMean μ X) (fun ω => Y ω - popMean μ Y) = 0 := by
  rw [sigmaJacSq, popCrossCov_centered_eq_zero_of_indepFun hindep hX hY, Matrix.trace_zero]

end Population

/-! ### The conditional statement

Here the conditioning on the body parameters `φ` is modelled by a sub-σ-algebra `m'` of the ambient
σ-algebra, and the conditional law by mathlib's disintegration kernel `condExpKernel μ m'`. The
first lemma turns mathlib's `CondIndepFun` — which is a kernel-level statement, an almost-sure
identity of measures of intersections — into ordinary independence under the fibre measure, for
almost every realization of `m'`. Everything above then applies fibrewise. -/

section Conditional

variable {Ω : Type*} {m' : MeasurableSpace Ω} [mΩ : MeasurableSpace Ω] [StandardBorelSpace Ω]
  {μ : Measure Ω} [IsFiniteMeasure μ]

/-- Conditional independence gives ordinary independence under the disintegration kernel, for
almost every realization of the conditioning σ-algebra.

The route is mathlib's `condIndepFun_iff_map_prod_eq_prod_map_map`, which characterizes
`CondIndepFun` as the almost-sure identity of the conditional joint law of `(X, Y)` with the
product of the conditional marginals, together with the corresponding characterization
`indepFun_iff_map_prod_eq_prod_map_map` of ordinary independence. -/
theorem ae_indepFun_of_condIndepFun (hm' : m' ≤ mΩ) {X Y : Ω → Fin d → ℝ}
    (hX : Measurable X) (hY : Measurable Y) (h : CondIndepFun m' hm' X Y μ) :
    ∀ᵐ a ∂(μ.trim hm'), IndepFun X Y (condExpKernel μ m' a) := by
  have hmap := (condIndepFun_iff_map_prod_eq_prod_map_map hX hY).mp h
  filter_upwards [hmap] with a ha
  rw [indepFun_iff_map_prod_eq_prod_map_map hX.aemeasurable hY.aemeasurable]
  simpa [Kernel.map_apply _ (hX.prodMk hY), Kernel.map_apply _ hX, Kernel.map_apply _ hY,
    Kernel.prod_apply] using ha

/-- The fibre mean under the disintegration kernel is the conditional expectation: almost surely,
`popMean (condExpKernel μ m' ω) X = (μ[X | m']) ω`. This is the lemma that entitles the results
below to be read as statements about the paper's `E[· | φ]`. -/
theorem popMean_condExpKernel_ae_eq_condExp (hm' : m' ≤ mΩ) {X : Ω → Fin d → ℝ}
    (hX : Integrable X μ) :
    (fun ω => popMean (condExpKernel μ m' ω) X) =ᵐ[μ] μ[X | m'] :=
  (condExp_ae_eq_integral_condExpKernel hm' hX).symm

/-- **The second link of the paper's chain, conditionally and in the paper's own notation.** The
conditional fluctuation is centred: almost surely the fibre mean of `X - E[X | φ]` vanishes. This
is the paper's "the last equality because both factors are centred by construction", derived rather
than assumed, with `E[· | φ]` read as mathlib's conditional expectation `μ[· | m']`. -/
theorem condPopMean_condExp_centered_ae_eq_zero (hm' : m' ≤ mΩ) {X : Ω → Fin d → ℝ}
    (hX : Integrable X μ) :
    ∀ᵐ ω ∂μ, popMean (condExpKernel μ m' ω) (fun x => X x - (μ[X | m']) ω) = 0 := by
  filter_upwards [hX.condExpKernel_ae (m := m'), popMean_condExpKernel_ae_eq_condExp hm' hX]
    with ω hxi hmean
  rw [← hmean]
  exact popMean_centered hxi

/-- **The first link of the paper's chain, conditionally.** Conditional independence factorizes the
conditional cross-covariance of the centred fluctuations,
`E[δθ̃ (δg)ᵀ | φ] = E[δθ̃ | φ] (E[δg | φ])ᵀ`, for almost every realization of the conditioning
σ-algebra. Together with `condPopMean_condExp_centered_ae_eq_zero` this is the paper's displayed
two-equality chain proved link by link at the conditional level, and not only at its endpoint. -/
theorem condPopCrossCov_centered_ae_factorizes (hm' : m' ≤ mΩ) {X Y : Ω → Fin d → ℝ}
    (hX : Measurable X) (hY : Measurable Y) (hXi : Integrable X μ) (hYi : Integrable Y μ)
    (h : CondIndepFun m' hm' X Y μ) :
    ∀ᵐ ω ∂μ, popCrossCov (condExpKernel μ m' ω)
        (fun x => X x - popMean (condExpKernel μ m' ω) X)
        (fun x => Y x - popMean (condExpKernel μ m' ω) Y) =
      Matrix.vecMulVec
        (popMean (condExpKernel μ m' ω) (fun x => X x - popMean (condExpKernel μ m' ω) X))
        (popMean (condExpKernel μ m' ω) (fun x => Y x - popMean (condExpKernel μ m' ω) Y)) := by
  filter_upwards [ae_of_ae_trim hm' (ae_indepFun_of_condIndepFun hm' hX hY h),
    hXi.condExpKernel_ae (m := m'), hYi.condExpKernel_ae (m := m')] with ω hindep hxi hyi
  exact popCrossCov_centered_of_indepFun hindep hxi hyi

/-- **Case (iii) of Theorem 1, as the paper states it.** If the produced iterate `X` and the loss
gradient `Y` are independent conditional on the body parameters, then the population
cross-covariance of their centred fluctuations vanishes almost surely:
`Σ_slack = E[δθ̃ (δg)ᵀ | φ] = 0`.

The conditional means subtracted here are the fibre means `popMean (condExpKernel μ m' ω) ·`, which
`popMean_condExpKernel_ae_eq_condExp` identifies almost everywhere with `μ[· | m']`, so the
fluctuations really are the paper's `δθ̃ = θ̃ - E[θ̃ | φ]` and `δg = g - E[g | φ]`. -/
theorem condPopCrossCov_centered_ae_eq_zero (hm' : m' ≤ mΩ) {X Y : Ω → Fin d → ℝ}
    (hX : Measurable X) (hY : Measurable Y) (hXi : Integrable X μ) (hYi : Integrable Y μ)
    (h : CondIndepFun m' hm' X Y μ) :
    ∀ᵐ ω ∂μ, popCrossCov (condExpKernel μ m' ω)
        (fun x => X x - popMean (condExpKernel μ m' ω) X)
        (fun x => Y x - popMean (condExpKernel μ m' ω) Y) = 0 := by
  filter_upwards [ae_of_ae_trim hm' (ae_indepFun_of_condIndepFun hm' hX hY h),
    hXi.condExpKernel_ae (m := m'), hYi.condExpKernel_ae (m := m')] with ω hindep hxi hyi
  exact popCrossCov_centered_eq_zero_of_indepFun hindep hxi hyi

/-- The corollary in the paper's own terms, conditional version: almost surely
`σ_Jac² = tr Σ_slack = 0` under the conditional-independence deletion. -/
theorem condSigmaJacSq_centered_ae_eq_zero (hm' : m' ≤ mΩ) {X Y : Ω → Fin d → ℝ}
    (hX : Measurable X) (hY : Measurable Y) (hXi : Integrable X μ) (hYi : Integrable Y μ)
    (h : CondIndepFun m' hm' X Y μ) :
    ∀ᵐ ω ∂μ, sigmaJacSq (condExpKernel μ m' ω)
        (fun x => X x - popMean (condExpKernel μ m' ω) X)
        (fun x => Y x - popMean (condExpKernel μ m' ω) Y) = 0 := by
  filter_upwards [condPopCrossCov_centered_ae_eq_zero hm' hX hY hXi hYi h] with ω hω
  rw [sigmaJacSq, hω, Matrix.trace_zero]

/-- **Case (iii) of Theorem 1, in the paper's own notation.** The same conclusion as
`condPopCrossCov_centered_ae_eq_zero`, but with the subtracted conditional means written as
mathlib's conditional expectations `μ[X | m']` and `μ[Y | m']` rather than as fibre means of the
disintegration kernel. This is the form the paper writes,
`Σ_slack ≡ E[δθ̃ (δg)ᵀ | φ] = 0` with `δθ̃ = θ̃ - E[θ̃ | φ]` and `δg = g - E[g | φ]`. The two forms
are the same statement — `popMean_condExpKernel_ae_eq_condExp` is what identifies them — but only
this one can be compared with the paper without performing that identification by hand. -/
theorem condPopCrossCov_condExp_centered_ae_eq_zero (hm' : m' ≤ mΩ) {X Y : Ω → Fin d → ℝ}
    (hX : Measurable X) (hY : Measurable Y) (hXi : Integrable X μ) (hYi : Integrable Y μ)
    (h : CondIndepFun m' hm' X Y μ) :
    ∀ᵐ ω ∂μ, popCrossCov (condExpKernel μ m' ω)
        (fun x => X x - (μ[X | m']) ω) (fun x => Y x - (μ[Y | m']) ω) = 0 := by
  filter_upwards [condPopCrossCov_centered_ae_eq_zero hm' hX hY hXi hYi h,
    popMean_condExpKernel_ae_eq_condExp hm' hXi,
    popMean_condExpKernel_ae_eq_condExp hm' hYi] with ω h0 hx hy
  rwa [← hx, ← hy]

/-- The corollary in the paper's own terms and the paper's own notation: almost surely
`σ_Jac² = tr Σ_slack = 0`, with the fluctuations centred by the conditional expectations
`μ[· | m']` that the paper writes `E[· | φ]`. -/
theorem condSigmaJacSq_condExp_centered_ae_eq_zero (hm' : m' ≤ mΩ) {X Y : Ω → Fin d → ℝ}
    (hX : Measurable X) (hY : Measurable Y) (hXi : Integrable X μ) (hYi : Integrable Y μ)
    (h : CondIndepFun m' hm' X Y μ) :
    ∀ᵐ ω ∂μ, sigmaJacSq (condExpKernel μ m' ω)
        (fun x => X x - (μ[X | m']) ω) (fun x => Y x - (μ[Y | m']) ω) = 0 := by
  filter_upwards [condPopCrossCov_condExp_centered_ae_eq_zero hm' hX hY hXi hYi h] with ω hω
  rw [sigmaJacSq, hω, Matrix.trace_zero]
end Conditional

/-! ### Ordinary independence is the trivial-conditioning case

The unconditional statements above are not a separate development from the conditional one: they
are its `m' = ⊥` case. The two lemmas below make that precise from both ends —
`condIndepFun_bot_of_indepFun` embeds the unconditional hypothesis into the conditional one, and
`condExpKernel_bot_apply` identifies every fibre of the trivial conditioning with `μ` itself, so
the conditional conclusions at `m' = ⊥` are the unconditional conclusions. The first lemma is also
what lets the non-vacuity witness at the end of this file produce a genuine `CondIndepFun`
hypothesis without appealing to a measurability degeneracy, that is, without making one of the two
fluctuations vanish. -/

section BotConditioning

variable {Ω : Type*} [mΩ : MeasurableSpace Ω] [StandardBorelSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Ordinary independence is conditional independence given the trivial σ-algebra. Conditioning on
`⊥` carries no information, so `μ[· | ⊥]` is the unconditional expectation and the defining
factorization of `CondIndepFun` collapses to the defining factorization of `IndepFun`. -/
theorem condIndepFun_bot_of_indepFun {β γ : Type*} [MeasurableSpace β] [MeasurableSpace γ]
    {X : Ω → β} {Y : Ω → γ} (hX : Measurable X) (hY : Measurable Y) (h : IndepFun X Y μ) :
    CondIndepFun (⊥ : MeasurableSpace Ω) bot_le X Y μ := by
  rw [condIndepFun_iff_condExp_inter_preimage_eq_mul hX hY]
  intro s t hs ht
  have hmul := (indepFun_iff_measure_inter_preimage_eq_mul.mp h) s t hs ht
  refine Filter.EventuallyEq.of_eq ?_
  rw [condExp_bot, condExp_bot, condExp_bot]
  funext ω
  simp only [integral_indicator_const, smul_eq_mul, mul_one, measureReal_def, hmul,
    ENNReal.toReal_mul, ((hX hs).inter (hY ht)), (hX hs), (hY ht)]

/-- **Every fibre of the trivial conditioning is the measure itself:**
`condExpKernel μ ⊥ ω = μ`, for every `ω` and not only almost every one. Conditioning on `⊥`
carries no information, so the conditional law is the unconditional law. Together with
`condIndepFun_bot_of_indepFun` this makes the specialization of the conditional theorems to
`m' = ⊥` an identity of statements and not only of hypotheses: at `m' = ⊥` the fibre measures,
the fibre means, and hence the conditional conclusions of this file are literally the
unconditional ones.

The proof pins the kernel down from two sides. The map `ω' ↦ condExpKernel μ ⊥ ω' s` is
measurable for the trivial σ-algebra, hence constant; and `condExpKernel_ae_eq_condExp`
identifies it almost everywhere with `μ⟦s | ⊥⟧`, which `condExp_bot` evaluates to the constant
`μ.real s`. A constant that agrees almost everywhere with `μ.real s`, under a measure that is
not zero, equals it everywhere. -/
theorem condExpKernel_bot_apply (ω : Ω) :
    condExpKernel μ (⊥ : MeasurableSpace Ω) ω = μ := by
  ext s hs
  have hconst : ∀ ω' : Ω,
      condExpKernel μ (⊥ : MeasurableSpace Ω) ω' s
        = condExpKernel μ (⊥ : MeasurableSpace Ω) ω s := by
    intro ω'
    have hmeas : Measurable[⊥] fun a => condExpKernel μ (⊥ : MeasurableSpace Ω) a s :=
      measurable_condExpKernel hs
    have hpre := hmeas (measurableSet_singleton
      (condExpKernel μ (⊥ : MeasurableSpace Ω) ω s))
    rw [MeasurableSpace.measurableSet_bot_iff] at hpre
    rcases hpre with h | h
    · have hmem : ω ∈ (fun a => condExpKernel μ (⊥ : MeasurableSpace Ω) a s) ⁻¹'
          {condExpKernel μ (⊥ : MeasurableSpace Ω) ω s} := rfl
      rw [h] at hmem
      exact absurd hmem (Set.notMem_empty ω)
    · exact Set.eq_univ_iff_forall.mp h ω'
  have hae := condExpKernel_ae_eq_condExp (μ := μ)
    (bot_le : (⊥ : MeasurableSpace Ω) ≤ mΩ) hs
  rw [condExp_bot] at hae
  haveI : (ae μ).NeBot := ae_neBot.mpr (IsProbabilityMeasure.ne_zero μ)
  obtain ⟨a, ha⟩ := hae.exists
  have hval : (condExpKernel μ (⊥ : MeasurableSpace Ω) a s).toReal = (μ s).toReal := by
    simpa [integral_indicator_const (1 : ℝ) hs, measureReal_def] using ha
  rw [← hconst a]
  exact (ENNReal.toReal_eq_toReal_iff' (measure_ne_top _ _) (measure_ne_top _ _)).mp hval

end BotConditioning

/-! ### The hypotheses are satisfiable, and the conclusion is not automatic

Every result above is an implication, and an implication is worth nothing if no instance satisfies
its hypotheses, or if its conclusion holds regardless of them. This section settles both questions
by exhibiting instances rather than by argument.

The first witness shows that `popCrossCov` and `sigmaJacSq` are not identically zero, even after
centring and even on a probability space: for the `±1` sign vector on a fair coin the centred
population cross-covariance is the all-ones matrix and `σ_Jac² = d`. So the independence hypothesis
of `popCrossCov_centered_eq_zero_of_indepFun` is load-bearing, and the vanishing it concludes comes
from that hypothesis and not from the way `popCrossCov` and `sigmaJacSq` are defined.

The second witness satisfies the entire hypothesis stack of the conditional theorems — a standard
Borel space, a probability measure, measurable `X` and `Y`, and a genuine `CondIndepFun` — with
*both* random vectors non-constant, so the conditional results are not statements about a family
that only degenerate pairs can join. Its conditioning σ-algebra is `⊥`, and by
`condExpKernel_bot_apply` its conditional laws are `coinPair` itself, so the fibre-level random
vectors of the witness are the two sign vectors, which are non-constant. The last theorem of the
section shows that a non-trivial conditioning σ-algebra is admissible too, at the cost, in that
particular instance, of making one of the two fluctuations vanish. A witness that is simultaneously
non-trivially conditioned and non-degenerate on both sides is not constructed here. -/

section Witness

/-- The fair coin: the smallest probability space carrying a random variable that fluctuates. -/
noncomputable def fairCoin : Measure Bool := (PMF.uniformOfFintype Bool).toMeasure

instance : IsProbabilityMeasure fairCoin := by rw [fairCoin]; infer_instance

/-- Expectation against the fair coin is the average of the two values. -/
theorem integral_fairCoin {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (f : Bool → E) : ∫ b, f b ∂fairCoin = (2 : ℝ)⁻¹ • (f true + f false) := by
  rw [fairCoin, PMF.integral_eq_sum, Fintype.sum_bool]
  simp [PMF.uniformOfFintype_apply, Fintype.card_bool, smul_add]

/-- The `±1` sign vector in `ℝ^d`: every coordinate is `+1` on one side of the coin and `-1` on the
other. It stands in for a maximally coupled iterate fluctuation. -/
def signVec (d : ℕ) (b : Bool) : Fin d → ℝ := fun _ => if b then 1 else -1

theorem measurable_signVec : Measurable (signVec d) := Measurable.of_discrete

/-- The sign vector is already centred, so centring leaves it unchanged. -/
theorem popMean_fairCoin_signVec : popMean fairCoin (signVec d) = 0 := by
  rw [popMean, integral_fairCoin]
  ext i
  simp [signVec]

theorem signVec_sub_popMean (b : Bool) :
    signVec d b - popMean fairCoin (signVec d) = signVec d b := by
  rw [popMean_fairCoin_signVec, sub_zero]

/-- The population cross-covariance of the sign vector with itself is the all-ones matrix. -/
theorem popCrossCov_fairCoin_signVec :
    popCrossCov fairCoin (signVec d) (signVec d) = Matrix.of fun _ _ => (1 : ℝ) := by
  ext i j
  rw [popCrossCov, Matrix.of_apply, integral_fairCoin]
  norm_num [signVec]

/-- Hence `σ_Jac² = d` for this pair, and in particular `σ_Jac²` is not identically zero. -/
theorem sigmaJacSq_fairCoin_signVec : sigmaJacSq fairCoin (signVec d) (signVec d) = d := by
  rw [sigmaJacSq, popCrossCov_fairCoin_signVec, Matrix.trace]
  simp

theorem popCrossCov_centered_fairCoin_signVec_ne_zero (hd : 0 < d) :
    popCrossCov fairCoin (fun b => signVec d b - popMean fairCoin (signVec d))
      (fun b => signVec d b - popMean fairCoin (signVec d)) ≠ 0 := by
  simp only [signVec_sub_popMean, popCrossCov_fairCoin_signVec]
  intro hcon
  have h0 := congrArg (fun M => M ⟨0, hd⟩ ⟨0, hd⟩) hcon
  norm_num at h0

theorem sigmaJacSq_centered_fairCoin_signVec_ne_zero (hd : 0 < d) :
    sigmaJacSq fairCoin (fun b => signVec d b - popMean fairCoin (signVec d))
      (fun b => signVec d b - popMean fairCoin (signVec d)) ≠ 0 := by
  simp only [signVec_sub_popMean, sigmaJacSq_fairCoin_signVec]
  exact_mod_cast hd.ne'

/-- **The independence hypothesis of the unconditional theorems cannot be dropped.** There is a
probability space and a pair of integrable `ℝ^d`-valued random vectors, `d ≥ 1`, whose centred
fluctuations have non-vanishing population cross-covariance and non-vanishing `σ_Jac²`. So
`popCrossCov_centered_eq_zero_of_indepFun` and `sigmaJacSq_centered_eq_zero_of_indepFun` are not
consequences of the definitions or of centring alone: independence is what produces the zero. -/
theorem exists_centered_popCrossCov_ne_zero (d : ℕ) (hd : 0 < d) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (X Y : Ω → Fin d → ℝ),
      IsProbabilityMeasure μ ∧ Integrable X μ ∧ Integrable Y μ ∧
        popCrossCov μ (fun ω => X ω - popMean μ X) (fun ω => Y ω - popMean μ Y) ≠ 0 ∧
        sigmaJacSq μ (fun ω => X ω - popMean μ X) (fun ω => Y ω - popMean μ Y) ≠ 0 :=
  ⟨Bool, inferInstance, fairCoin, signVec d, signVec d, inferInstance, Integrable.of_finite,
    Integrable.of_finite, popCrossCov_centered_fairCoin_signVec_ne_zero hd,
    sigmaJacSq_centered_fairCoin_signVec_ne_zero hd⟩

/-- Two independent fair coins, the carrier of the conditional witness. -/
noncomputable def coinPair : Measure (Bool × Bool) := fairCoin.prod fairCoin

instance : IsProbabilityMeasure coinPair := by rw [coinPair]; infer_instance

/-- Neither factor of the witness is a constant in disguise: the sign vector takes different values
on the two sides of the coin. -/
theorem signVec_true_ne_signVec_false (hd : 0 < d) : signVec d true ≠ signVec d false := by
  intro hcon
  have h0 := congrFun hcon ⟨0, hd⟩
  norm_num [signVec] at h0

theorem indepFun_coinPair :
    IndepFun (fun ω : Bool × Bool => signVec d ω.1) (fun ω : Bool × Bool => signVec d ω.2)
      coinPair :=
  indepFun_prod measurable_signVec measurable_signVec

/-- The conditional-independence hypothesis is satisfiable with both random vectors non-constant:
on two independent fair coins the two sign vectors are conditionally independent given the trivial
σ-algebra, and the conditional independence comes from `condIndepFun_bot_of_indepFun`, that is from
genuine independence, rather than from one of the two being measurable with respect to the
conditioning σ-algebra. -/
theorem condIndepFun_coinPair :
    CondIndepFun (⊥ : MeasurableSpace (Bool × Bool)) bot_le
      (fun ω : Bool × Bool => signVec d ω.1) (fun ω : Bool × Bool => signVec d ω.2) coinPair :=
  condIndepFun_bot_of_indepFun (measurable_signVec.comp measurable_fst)
    (measurable_signVec.comp measurable_snd) indepFun_coinPair

/-- The headline conditional theorem `condPopCrossCov_condExp_centered_ae_eq_zero` instantiated at
that witness, which is what shows its hypotheses are jointly satisfiable rather than merely
plausible. -/
theorem condPopCrossCov_coinPair_ae_eq_zero (d : ℕ) :
    ∀ᵐ ω ∂coinPair, popCrossCov (condExpKernel coinPair ⊥ ω)
        (fun x => signVec d x.1 - (coinPair[(fun x : Bool × Bool => signVec d x.1) | ⊥]) ω)
        (fun x => signVec d x.2 - (coinPair[(fun x : Bool × Bool => signVec d x.2) | ⊥]) ω) = 0 :=
  condPopCrossCov_condExp_centered_ae_eq_zero bot_le (measurable_signVec.comp measurable_fst)
    (measurable_signVec.comp measurable_snd) Integrable.of_finite Integrable.of_finite
    condIndepFun_coinPair

/-- A non-trivial conditioning σ-algebra is admissible as well: conditioning on the second coin,
the two sign vectors remain conditionally independent. Here the conditional independence is
obtained the degenerate way, from the second variable being measurable with respect to the
conditioning σ-algebra; the point of the witness is only that `m'` in the conditional theorems is
not forced to be `⊥`. -/
theorem condIndepFun_coinPair_snd :
    CondIndepFun (MeasurableSpace.comap Prod.snd inferInstance) measurable_snd.comap_le
      (fun ω : Bool × Bool => signVec d ω.1) (fun ω : Bool × Bool => signVec d ω.2) coinPair :=
  condIndepFun_of_measurable_right (measurable_signVec.comp measurable_fst)
    (measurable_signVec.comp (_root_.comap_measurable Prod.snd))

end Witness
end IcnnLift
