import Mathlib
import TraceLemmas
import UpdateCovariance
import CouplingSign
import DiffusionOrdering
import TaylorChain

/-!
# The frozen-iterate minibatch noise in (A4), and the coupling channel it turns out to be

Assumption (A4) of the paper (`docs/paper/v4/A1_proofs.tex`, "Regularity assumptions") writes the
fluctuation of the per-probe forward-KL gradient as `δg = H δθ̃ + r`, with `H = E_X[∇²L]` the mean
Hessian at the reference iterate and `r` a remainder budgeted as `‖r‖ ≤ C_r ‖δθ̃‖²`. The clause
(A4'-q) that the appendix adopts then asks the remainder's share of the cross-covariance,
`R = E[δθ̃ rᵀ]`, to be dominated by the magnitude channel, `2β ≤ 3λ`. This file records that the
chain as written omits a term that is not a remainder at all, works out what the omission does to
every downstream statement, and states the clause that the strict diffusion ordering actually
needs.

**The omitted term.** Fix the reference iterate `θ̄` and draw a batch `X`. The gradient the probe
returns is evaluated at the probe's own iterate `θ̄ + δθ̃(X)` on the batch `X`, so a first-order
expansion in the iterate reads

  `g(X) = g₀(X) + H δθ̃(X) + r₂(X)`,   `g₀(X) = ∇L(θ̄; X)`,

where `g₀` is the gradient of the *same* batch at the *frozen* iterate `θ̄`, and `r₂` collects the
pairing of the per-batch Hessian fluctuation `H(X) − H` with the jitter and the genuine second-order
Taylor term. Centring, `δg = ξ + H δθ̃ + (r₂ − E r₂)` with `ξ = g₀ − E g₀` the minibatch noise at the
frozen iterate. The old chain therefore holds only with `r = ξ + r₂ − E r₂`, and `ξ` is of order one
in the jitter: along the family that scales the latent-weight fluctuation to zero at fixed batch law the
bound `‖r‖ ≤ C_r ‖δθ̃‖²` forces `ξ ≡ 0` (`noise_eq_zero_of_quadratic_bound`), that is, it is
compatible only with a vanishing objective noise. This is the file's first finding, and it is the
one the hypothesis audit raised.

**The three-term decomposition.** For the real covariances of `UpdateCovariance.lean`, the
slack-channel cross-covariance the estimator of `eq:cross-cov` targets splits exactly as

  `Σ_slack = E[δθ̃ δgᵀ] = Ξ + V H + R₂`,   `Ξ = E[δθ̃ ξᵀ]`,  `V = E[δθ̃ δθ̃ᵀ]`,  `R₂ = E[δθ̃ r₂ᵀ]`

(`covMat_slack_refined`). The middle term is the latent weight's self-coupling through the curvature;
the first is the covariance of the latent weight with the frozen-iterate minibatch noise, both drawn
from the same batch. `Ξ` vanishes for every latent weight that reads the batch through a given statistic
exactly when the noise has zero conditional mean on the fibers of that statistic
(`outerMoment_eq_zero_of_fiberMeanZero`, with the split-batch and the independent-noise cases as
corollaries), and that condition is sharp in the quantified sense: a nonzero fiber mean is detected
by some latent weight reading the batch through that statistic
(`exists_body_coupling_ne_zero_of_fiberMean_ne_zero`), although a particular latent weight may have
`Ξ = 0` by cancellation. Two four-point and two-point witnesses show `Ξ ≠ 0` with both signs for
values of the latent weight and noises that are different functions of one batch.

**Mechanism, not nuisance.** Substituting the refined chain into the exact update-covariance
decomposition gives the projected diffusion of the lifted update as

  `σ²_lift = σ²_dir + s² (eᵀ (H Ξ + Ξᵀ H) e + eᵀ H V H e) + s² ϱ₂(e)`

(`projDiffusion_lift_eq_frozen_form`), where `ϱ₂` collects the three scalars the genuine remainder
`r₂` contributes. So `Ξ` is the object the coupling channel `C = H Σ + Σᵀ H` of `CouplingSign.lean`
and `DiffusionOrdering.lean` is built on: those modules write their `Σ_slack` as `Cov[δθ̃, g]` with
`g` the frozen-iterate gradient of the expansion `G = s g + s H δθ̃ + R`, which is `Ξ`, and every
ordering theorem there is a theorem about `Ξ` and needs no change. What does change is the reading
of the displayed lemma. Written on the full cross-covariance the same identity is

  `σ²_lift = σ²_dir + s² (eᵀ C_full e − eᵀ H V H e) + s² ϱ'(e)`

(`projDiffusion_lift_eq_full_form`), with the jitter term entering with a **minus** sign, because
`C_full` already counts it twice; the display with `+ eᵀ H V H e` exceeds the true value by
`2 s² eᵀ H V H e` up to the remainder scalars (`display_sub_projDiffusion_lift`). The display is
exact when its `Σ_slack` is read as `Ξ`, and that is the reading the paper must adopt.

**The corrected regime statement.** The strict ordering `σ²_dir < σ²_lift` follows from positive
alignment of the frozen coupling, `κ ‖e‖² ≤ eᵀ (H Ξ + Ξᵀ H) e`, together with control of the
remainder scalars `|ϱ₂(e)| ≤ τ ‖e‖²` at `τ < κ` (`projDiffusion_direct_lt_lift_of_frozenAlignment`);
the jitter term is a free non-negative summand. This replaces (A4'-q). The clause it replaces is
not merely unnecessary but false in the operating regime: under (A4'-q) applied to the chain the
appendix writes, the frozen coupling is capped by five halves of the jitter term,
`eᵀ H Ξ e ≤ (5/2) eᵀ H V H e` (`remainderControl_caps_frozen_coupling`), and along the family
`δθ̃ = t u` at fixed batch law the left side is first order in `t` while the right side is second
order, so the cap fails for every sufficiently small `t` whenever the frozen coupling is positive
(`remainderControl_fails_small_jitter`). Along the same family the projected excess is a polynomial
in `t` whose linear coefficient is the frozen coupling plus the first-order remainder pairing
(`projDiffusion_excess_small_jitter`); a positive linear coefficient gives the strict ordering at
every small scale (`projDiffusion_direct_lt_lift_small_jitter`), a negative one reverses it
(`projDiffusion_lift_lt_direct_small_jitter`), and the jitter term, being second order, decides
nothing at small scale. That last point is the scaling reading of the matched-magnitude control:
a latent weight independent of the batch has `Ξ = 0` (`covMat_eq_zero_of_indepFun`) and, in the
finite-sample model, no first-order remainder pairing either
(`outerMoment_prod_hessianFluct_eq_zero`), so its excess starts at second order
(`projDiffusion_excess_small_jitter_of_no_frozen_coupling`).

**Where the refined chain comes from.** The chain is not an assumption. For a per-batch gradient map
differentiable in the iterate it is the first-order expansion at the frozen iterate, batch by batch,
with the frozen gradient appearing by construction
(`probeGrad_eq_frozen_add_hessian_add_remainder`), and the Taylor part of `r₂` is quadratic in the
jitter with the per-batch Hessian-Lipschitz constant (`l2Norm_perBatch_remainder_le`), by the
mean-value bound of `TaylorChain.lean`. What `TaylorChain.lean` discharges under the name (A4) is the
expansion of the gradient of the *expected* loss, which carries neither `ξ` nor the Hessian
fluctuation; that object is a third one, and the paper's `δg` is not it.

## Results
* `refinedRemainder`, `refinedChain_of_remainder` — the refined chain, with the remainder defined as
  the discrepancy so that the chain is an identity.
* `oldChain_remainder_eq_noise_add_refined` — the old remainder is the frozen gradient plus the
  refined remainder: (A4)'s `r` contains `ξ`.
* `noise_eq_zero_of_quadratic_bound` — the old quadratic bound along the small-jitter family forces
  the minibatch noise to vanish.
* `FiberMeanZero`, `outerMoment_eq_zero_of_fiberMeanZero` — the vanishing theorem for `Ξ`: zero
  conditional mean of the noise on the fibers of the latent weight's statistic.
* `fiberMeanZero_of_product`, `outerMoment_prod_eq_zero_of_noise_mean_zero` — the split-batch and
  independent-noise cases as instances.
* `fiberMeanZero_hessianFluct_of_product`, `outerMoment_prod_hessianFluct_eq_zero` — a perturbation
  independent of the batch has no first-order remainder pairing either, which discharges the second
  hypothesis of `projDiffusion_excess_small_jitter_of_no_frozen_coupling` in the finite-sample
  model.
* `exists_body_coupling_ne_zero_of_fiberMean_ne_zero` — sharpness of the condition.
* `SameBatchWitness.coupling_ne_zero`, `LocationWitness.coupling_neg`, `LocationWitness.coupling_pos`
  — `Ξ ≠ 0`, with either sign, for two different functions of one batch.
* `covMat_slack_refined` — the three-term decomposition for the real covariances.
* `covMat_eq_zero_of_indepFun`, `covMat_slack_refined_of_indepFun` — a latent weight independent of the
  batch driving the gradient has `Ξ = 0` while `V H` survives.
* `trace_covMat_slack_refined` — the measured trace is `tr Ξ + tr (V H) + tr R₂` with `tr (V H) ≥ 0`,
  so a positive measured trace does not by itself sign `tr Ξ`.
* `projDiffusion_lift_eq_frozen_form`, `projDiffusion_lift_eq_full_form`,
  `display_sub_projDiffusion_lift` — the projected identity with the `ξ` cross-term explicit, in
  both readings, and the over-count of the display on the full reading.
* `projDiffusion_direct_lt_lift_of_frozenAlignment` — the corrected regime statement.
* `remainderControl_caps_frozen_coupling`, `remainderControl_fails_small_jitter` — (A4'-q) caps the
  mechanism by the magnitude channel and fails along the small-jitter family.
* `projDiffusion_excess_small_jitter`, `projDiffusion_direct_lt_lift_small_jitter`,
  `projDiffusion_lift_lt_direct_small_jitter`,
  `projDiffusion_excess_small_jitter_of_no_frozen_coupling` — the scaling reading.
* `probeGrad_eq_frozen_add_hessian_add_remainder`, `l2Norm_perBatch_remainder_le` — the refined
  chain from per-batch Taylor expansion, with the Taylor part bounded.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The hypotheses this file carries, and what would discharge each. The refined chain `hchain` is an
identity once `r₂` is read as the discrepancy, and `probeGrad_eq_frozen_add_hessian_add_remainder`
derives it from differentiability; what is not proved is the size of `r₂` beyond its Taylor part,
which is the pairing of the per-batch Hessian fluctuation with the jitter and depends on the
batch law. The alignment hypothesis `hA6` on the frozen coupling is assumption (A6) of the paper
read on `Ξ`; `CouplingSign.lean` shows that no chain-plus-remainder-control hypothesis implies it,
and the two-point witnesses here show that `Ξ` takes either sign, so it cannot be discharged. The
remainder control `hrem` bounds three covariances involving `r₂`. Along the small-jitter family the
first of them, the pairing of the frozen gradient with the Hessian-fluctuation part of `r₂`, is of
first order in the scale, as is the alignment margin, while the other two are of order two and
higher; `projDiffusion_excess_small_jitter` makes this explicit, its linear coefficient being the
frozen coupling plus that pairing. At small scale the clause `τ < κ` therefore asks the pairing to
fall below the margin, a hypothesis on third-order batch moments that nothing here computes, and
not a consequence of the chain. The positivity-map slope enters as the scalar `s` of (A2) and the
band geometry of (A2) is not modelled, exactly as in `DiffusionOrdering.lean`. The scaling family
`δθ̃ = t u`, `r₂ = t w₁ + t² w₂` at fixed frozen gradient models a body whose output scale is turned
down at fixed batch law, which is the matched-magnitude control's own parametrization; it is a
model of the small-jitter limit and not a theorem about training. Nothing in this file touches the
stochastic-analysis side of the paper, and the integrability hypotheses are the minimal ones under
which the covariances exist.
-/

open Matrix Finset MeasureTheory ProbabilityTheory Set
open scoped MatrixOrder

namespace IcnnLift

/-! ## The refined chain, pointwise

The frozen-iterate gradient `g₀`, the per-probe gradient `g` and the latent-weight fluctuation `dth` are
coordinate vectors indexed by the sample space; nothing probabilistic enters this section. -/

section Pointwise

variable {n : Type*} [Fintype n] {Ω : Type*}

/-- **The refined remainder**: the discrepancy between the per-probe gradient and its expansion
`g₀ + H δθ̃` about the frozen iterate, with the frozen-iterate gradient `g₀` kept as a separate
summand. In the paper's vocabulary this is `(H(X) − H) δθ̃ + r_T`, the Hessian-fluctuation pairing
plus the second-order Taylor term; nothing is asserted about its size by the definition. -/
def refinedRemainder (H : Matrix n n ℝ) (g g₀ dth : Ω → n → ℝ) (ω : Ω) : n → ℝ :=
  g ω - g₀ ω - H *ᵥ dth ω

/-- **The refined chain holds by construction**: `g = g₀ + H δθ̃ + r₂`. -/
theorem refinedChain_of_remainder (H : Matrix n n ℝ) (g g₀ dth : Ω → n → ℝ) (ω : Ω) :
    g ω = g₀ ω + H *ᵥ dth ω + refinedRemainder H g g₀ dth ω := by
  rw [refinedRemainder]; abel

/-- **(A4)'s remainder contains the frozen-iterate gradient.** If the old chain `g = H δθ̃ + r` is
imposed on the per-probe gradient, its remainder is the frozen gradient plus the refined remainder,
so it carries the whole of the minibatch noise `ξ = g₀ − E g₀` in addition to the second-order
terms. -/
theorem oldChain_remainder_eq_noise_add_refined {H : Matrix n n ℝ} {g g₀ dth r : Ω → n → ℝ}
    (hold : ∀ ω, g ω = H *ᵥ dth ω + r ω) (ω : Ω) :
    r ω = g₀ ω + refinedRemainder H g g₀ dth ω := by
  have h := hold ω
  rw [refinedRemainder, h]; abel

end Pointwise

/-- **The old quadratic bound is compatible only with a vanishing minibatch noise.** Along the
family that scales the latent-weight fluctuation as `t u` at a fixed batch law, the old remainder at a
sample point is `z + t • w₁ + t² • w₂`, with `z` the noise value there and the other two terms the
refined remainder's first- and second-order parts, while the bound `C_r ‖δθ̃‖²` scales as `C t²`.
If the bound holds for every small scale then `z = 0`. Stated in an arbitrary real normed space,
since only the triangle inequality is used. -/
theorem noise_eq_zero_of_quadratic_bound {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    {z w₁ w₂ : F} {C : ℝ}
    (h : ∀ t : ℝ, 0 < t → t ≤ 1 → ‖z + t • w₁ + t ^ 2 • w₂‖ ≤ C * t ^ 2) : z = 0 := by
  by_contra hz
  have hpos : 0 < ‖z‖ := norm_pos_iff.mpr hz
  set M : ℝ := |C| + ‖w₁‖ + ‖w₂‖ + 1 with hM
  have hMpos : 0 < M := by positivity
  have hbound : ∀ t : ℝ, 0 < t → t ≤ 1 → ‖z‖ ≤ t * M := by
    intro t ht0 ht1
    have h1 : ‖z‖ ≤ ‖z + t • w₁ + t ^ 2 • w₂‖ + ‖t • w₁‖ + ‖t ^ 2 • w₂‖ := by
      have ha := norm_sub_le (z + t • w₁ + t ^ 2 • w₂) (t ^ 2 • w₂)
      have hb := norm_sub_le (z + t • w₁) (t • w₁)
      simp only [add_sub_cancel_right] at ha hb
      linarith
    have h2 : ‖t • w₁‖ = t * ‖w₁‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos ht0]
    have h2' : ‖t ^ 2 • w₂‖ = t ^ 2 * ‖w₂‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity)]
    have ht2 : t ^ 2 ≤ t := by nlinarith
    have h3 : C * t ^ 2 ≤ |C| * t := by
      have hC : C ≤ |C| := le_abs_self C
      have hCt : C * t ^ 2 ≤ |C| * t ^ 2 := mul_le_mul_of_nonneg_right hC (sq_nonneg t)
      have : |C| * t ^ 2 ≤ |C| * t := mul_le_mul_of_nonneg_left ht2 (abs_nonneg C)
      linarith
    have h4 : t ^ 2 * ‖w₂‖ ≤ t * ‖w₂‖ := mul_le_mul_of_nonneg_right ht2 (norm_nonneg w₂)
    calc ‖z‖ ≤ ‖z + t • w₁ + t ^ 2 • w₂‖ + ‖t • w₁‖ + ‖t ^ 2 • w₂‖ := h1
      _ ≤ C * t ^ 2 + t * ‖w₁‖ + t ^ 2 * ‖w₂‖ := by rw [h2, h2']; linarith [h t ht0 ht1]
      _ ≤ |C| * t + t * ‖w₁‖ + t * ‖w₂‖ := by linarith
      _ ≤ t * M := by rw [hM]; nlinarith
  set t : ℝ := min 1 (‖z‖ / (2 * M)) with ht
  have ht0 : 0 < t := by
    rw [ht]; exact lt_min one_pos (div_pos hpos (by positivity))
  have ht1 : t ≤ 1 := by rw [ht]; exact min_le_left _ _
  have htle : t ≤ ‖z‖ / (2 * M) := by rw [ht]; exact min_le_right _ _
  have := hbound t ht0 ht1
  have h4 : t * M ≤ ‖z‖ / 2 := by
    calc t * M ≤ ‖z‖ / (2 * M) * M := mul_le_mul_of_nonneg_right htle hMpos.le
      _ = ‖z‖ / 2 := by field_simp
  linarith

/-! ## When the frozen coupling vanishes, on a finite weighted sample space

The vocabulary is `CouplingSign.outerMoment`, the weighted second moment `∑ w(ω) u(ω) v(ω)ᵀ`. The
latent-weight fluctuation reads the batch through a statistic `S`, `δθ̃ = a ∘ S − c`, and the noise `ξ`
is any other function of the batch. -/

section FiniteSample

variable {n : Type*} {Ω : Type*} [Fintype Ω]

/-- **Zero conditional mean on the fibers.** The weighted total of `v` over each fiber
`{ω : S ω = i}` of the statistic vanishes; for a probability weighting this is `E[v | S] = 0`. -/
def FiberMeanZero {ι : Type*} [DecidableEq ι] (w : Ω → ℝ) (S : Ω → ι) (v : Ω → n → ℝ) : Prop :=
  ∀ i : ι, ∑ ω ∈ univ.filter (fun ω => S ω = i), w ω • v ω = 0

/-- An outer product distributes over a weighted sum in its second factor. -/
theorem vecMulVec_sum_smul {ι : Type*} (s : Finset ι) (a : n → ℝ) (c : ι → ℝ) (v : ι → n → ℝ) :
    Matrix.vecMulVec a (∑ k ∈ s, c k • v k) = ∑ k ∈ s, c k • Matrix.vecMulVec a (v k) := by
  ext i j
  simp only [Matrix.vecMulVec_apply, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Matrix.sum_apply,
    Matrix.smul_apply, Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- **The vanishing theorem for the frozen coupling.** A body that reads the batch through a
statistic `S`, up to a centring constant, is uncorrelated with a noise whose conditional mean on
every fiber of `S` is zero: `E[δθ̃ ξᵀ] = 0`. This is the tower property, summed fiber by fiber. -/
theorem outerMoment_eq_zero_of_fiberMeanZero {ι : Type*} [Fintype ι] [DecidableEq ι]
    {w : Ω → ℝ} {S : Ω → ι} {v : Ω → n → ℝ} (a : ι → n → ℝ) (c : n → ℝ)
    (hv : FiberMeanZero w S v) :
    outerMoment w (fun ω => a (S ω) - c) v = 0 := by
  rw [outerMoment, ← Finset.sum_fiberwise univ S]
  refine Finset.sum_eq_zero fun i _ => ?_
  have hcongr : ∀ ω ∈ univ.filter (fun ω => S ω = i),
      w ω • Matrix.vecMulVec (a (S ω) - c) (v ω) = w ω • Matrix.vecMulVec (a i - c) (v ω) := by
    intro ω hω
    rw [Finset.mem_filter] at hω
    rw [hω.2]
  rw [Finset.sum_congr rfl hcongr]
  have : ∑ ω ∈ univ.filter (fun ω => S ω = i), w ω • Matrix.vecMulVec (a i - c) (v ω)
      = Matrix.vecMulVec (a i - c) (∑ ω ∈ univ.filter (fun ω => S ω = i), w ω • v ω) := by
    rw [vecMulVec_sum_smul]
  rw [this, hv i, Matrix.vecMulVec_zero]

/-- A sum over the fiber of the first coordinate of a product is a sum over the second factor. -/
theorem sum_filter_prod_fst {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂] [DecidableEq Ω₁]
    {M : Type*} [AddCommMonoid M] (f : Ω₁ × Ω₂ → M) (i : Ω₁) :
    ∑ p ∈ univ.filter (fun p : Ω₁ × Ω₂ => p.1 = i), f p = ∑ b, f (i, b) := by
  rw [Finset.sum_filter, Fintype.sum_prod_type]
  simp only
  rw [Finset.sum_eq_single i]
  · simp
  · intro a _ ha
    exact Finset.sum_eq_zero fun b _ => by simp [ha]
  · intro h; exact absurd (Finset.mem_univ i) h

/-- **A split batch is the fiber condition with the conditioning half as the statistic.** On a
product sample space with product weights, a noise that depends only on the second factor and has
zero weighted mean there has zero conditional mean on every fiber of the first projection. -/
theorem fiberMeanZero_of_product {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂] [DecidableEq Ω₁]
    (w₁ : Ω₁ → ℝ) (w₂ : Ω₂ → ℝ) (v : Ω₂ → n → ℝ) (hv : ∑ b, w₂ b • v b = 0) :
    FiberMeanZero (fun p : Ω₁ × Ω₂ => w₁ p.1 * w₂ p.2) Prod.fst (fun p => v p.2) := by
  intro i
  rw [sum_filter_prod_fst (fun p : Ω₁ × Ω₂ => (w₁ p.1 * w₂ p.2) • v p.2) i]
  simp only [mul_smul]
  rw [← Finset.smul_sum, hv, smul_zero]

/-- **Independence empties the frozen coupling**, as an instance of the vanishing theorem: an
latent weight conditioned on a batch drawn independently of the one that forms the gradient, or an
injected perturbation independent of the batch altogether, has `Ξ = 0`. This is the deletion of
Theorem 1's third ingredient, read on the mechanism object rather than on the full
cross-covariance; the self-coupling `V H` is untouched by it. -/
theorem outerMoment_prod_eq_zero_of_noise_mean_zero {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    [DecidableEq Ω₁] (w₁ : Ω₁ → ℝ) (w₂ : Ω₂ → ℝ) (a : Ω₁ → n → ℝ) (c : n → ℝ) (v : Ω₂ → n → ℝ)
    (hv : ∑ b, w₂ b • v b = 0) :
    outerMoment (fun p : Ω₁ × Ω₂ => w₁ p.1 * w₂ p.2) (fun p => a p.1 - c) (fun p => v p.2) = 0 :=
  outerMoment_eq_zero_of_fiberMeanZero (S := Prod.fst) a c (fiberMeanZero_of_product w₁ w₂ v hv)

/-- **A perturbation independent of the batch has no first-order remainder pairing.** On the same
product sample space, with the per-batch Hessian a function of the batch factor alone and the
perturbation a centred function of the other factor, the Hessian-fluctuation part of the refined
remainder, `(H(X) − H) u`, has zero conditional mean on every fiber of the batch projection. -/
theorem fiberMeanZero_hessianFluct_of_product {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    [DecidableEq Ω₁] [Fintype n] (w₁ : Ω₁ → ℝ) (w₂ : Ω₂ → ℝ) (Hb : Ω₁ → Matrix n n ℝ)
    (H : Matrix n n ℝ) (u : Ω₂ → n → ℝ) (hu : ∑ b, w₂ b • u b = 0) :
    FiberMeanZero (fun p : Ω₁ × Ω₂ => w₁ p.1 * w₂ p.2) Prod.fst
      (fun p => (Hb p.1 - H) *ᵥ u p.2) := by
  intro i
  rw [sum_filter_prod_fst (fun p : Ω₁ × Ω₂ => (w₁ p.1 * w₂ p.2) • ((Hb p.1 - H) *ᵥ u p.2)) i]
  simp only [mul_smul]
  have h : ∑ b, w₁ i • w₂ b • ((Hb i - H) *ᵥ u b)
      = w₁ i • ((Hb i - H) *ᵥ ∑ b, w₂ b • u b) := by
    rw [Matrix.mulVec_sum, ← Finset.smul_sum]
    congr 1
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Matrix.mulVec_smul]
  rw [h, hu, Matrix.mulVec_zero, smul_zero]

/-- **The matched-magnitude cell has no first-order remainder pairing.** The pairing of the frozen
gradient with the Hessian-fluctuation part of `r₂` vanishes for a perturbation independent of the
batch: `E[g₀ ((H(X) − H) u)ᵀ] = 0`. Together with `outerMoment_prod_eq_zero_of_noise_mean_zero`
this discharges, in the finite-sample model, both hypotheses of
`projDiffusion_excess_small_jitter_of_no_frozen_coupling` for such a perturbation. -/
theorem outerMoment_prod_hessianFluct_eq_zero {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    [DecidableEq Ω₁] [Fintype n] (w₁ : Ω₁ → ℝ) (w₂ : Ω₂ → ℝ) (g₀ : Ω₁ → n → ℝ)
    (Hb : Ω₁ → Matrix n n ℝ) (H : Matrix n n ℝ) (u : Ω₂ → n → ℝ)
    (hu : ∑ b, w₂ b • u b = 0) :
    outerMoment (fun p : Ω₁ × Ω₂ => w₁ p.1 * w₂ p.2) (fun p => g₀ p.1)
      (fun p => (Hb p.1 - H) *ᵥ u p.2) = 0 := by
  have h := outerMoment_eq_zero_of_fiberMeanZero (S := Prod.fst) g₀ (0 : n → ℝ)
    (fiberMeanZero_hessianFluct_of_product w₁ w₂ Hb H u hu)
  simpa using h

/-- An outer product of two nonzero vectors is nonzero. -/
theorem vecMulVec_ne_zero {a m : n → ℝ} (ha : a ≠ 0) (hm : m ≠ 0) :
    Matrix.vecMulVec a m ≠ 0 := by
  intro h
  obtain ⟨j, hj⟩ : ∃ j, a j ≠ 0 := Function.ne_iff.mp ha
  obtain ⟨k, hk⟩ : ∃ k, m k ≠ 0 := Function.ne_iff.mp hm
  have := congrFun (congrFun h j) k
  rw [Matrix.vecMulVec_apply, Matrix.zero_apply] at this
  exact mul_ne_zero hj hk this

/-- The frozen coupling of a latent weight supported on one fiber of the statistic, centred by any
multiple of its direction, is the outer product of that direction with the fiber total of the
noise, provided the noise has zero total mean. -/
theorem outerMoment_indicator_eq {ι : Type*} [Fintype ι] [DecidableEq ι]
    (w : Ω → ℝ) (S : Ω → ι) (v : Ω → n → ℝ) (a : n → ℝ) (i : ι) (p : ℝ)
    (hv : ∑ ω, w ω • v ω = 0) :
    outerMoment w (fun ω => (if S ω = i then a else 0) - p • a) v
      = Matrix.vecMulVec a (∑ ω ∈ univ.filter (fun ω => S ω = i), w ω • v ω) := by
  rw [outerMoment]
  simp only [Matrix.sub_vecMulVec, smul_sub, Finset.sum_sub_distrib]
  have h1 : ∑ ω, w ω • Matrix.vecMulVec (p • a) (v ω)
      = Matrix.vecMulVec (p • a) (∑ ω, w ω • v ω) := by rw [vecMulVec_sum_smul]
  rw [h1, hv, Matrix.vecMulVec_zero, sub_zero, vecMulVec_sum_smul, Finset.sum_filter]
  refine Finset.sum_congr rfl fun ω _ => ?_
  split_ifs <;> simp

/-- **Sharpness of the vanishing condition.** If the noise has zero mean but a nonzero conditional
mean on some fiber of the statistic, then a latent weight reading the batch through that statistic
alone — the indicator of the fiber in any nonzero direction, centred by any multiple of that
direction — has a nonzero frozen coupling. The fiber condition is therefore exactly what makes
`Ξ = 0` hold for every latent weight that reads the batch through `S`. -/
theorem exists_body_coupling_ne_zero_of_fiberMean_ne_zero {ι : Type*} [Fintype ι]
    [DecidableEq ι] (w : Ω → ℝ) (S : Ω → ι) (v : Ω → n → ℝ) (hv : ∑ ω, w ω • v ω = 0)
    {i : ι} (hi : ∑ ω ∈ univ.filter (fun ω => S ω = i), w ω • v ω ≠ 0) {a : n → ℝ} (ha : a ≠ 0)
    (p : ℝ) :
    outerMoment w (fun ω => (if S ω = i then a else 0) - p • a) v ≠ 0 := by
  rw [outerMoment_indicator_eq w S v a i p hv]
  exact vecMulVec_ne_zero ha hi

end FiniteSample

/-! ## Two witnesses: different functions of one batch, correlated

The first is a four-point batch space on which the latent weight reads the batch through a two-valued
statistic, the noise is a different function of the same batch, both are centred, and the frozen
coupling is one. The second is the two-point sample space of a Gaussian location model, on which
the sign of the coupling is that of the alignment between the latent weight and the gradient noise:
the same construction realizes `Ξ = −1` and `Ξ = +1`. -/

namespace SameBatchWitness

/-- Uniform weights on a four-point batch space. -/
noncomputable def w : Fin 4 → ℝ := fun _ => (1 : ℝ) / 4

/-- The statistic through which the latent weight reads the batch. -/
def S : Fin 4 → Fin 2 := ![0, 0, 1, 1]

/-- The latent weight's two values, one per fiber. -/
def a : Fin 2 → Fin 1 → ℝ := ![![1], ![-1]]

/-- The latent-weight fluctuation `δθ̃ = a ∘ S`, already centred. -/
def dth : Fin 4 → Fin 1 → ℝ := fun ω => a (S ω)

/-- The minibatch noise `ξ`, a different function of the same batch, centred. -/
def xi : Fin 4 → Fin 1 → ℝ := ![![2], ![0], ![0], ![-2]]

theorem dth_eq : dth = ![![1], ![1], ![-1], ![-1]] := by
  funext ω; fin_cases ω <;> rfl

/-- The latent-weight fluctuation is centred. -/
theorem dth_mean_zero : ∑ ω, w ω • dth ω = 0 := by
  rw [dth_eq]
  ext i; fin_cases i
  simp [Fin.sum_univ_four, w]

/-- The noise is centred. -/
theorem xi_mean_zero : ∑ ω, w ω • xi ω = 0 := by
  ext i; fin_cases i
  simp [Fin.sum_univ_four, w, xi]

/-- The noise has a nonzero conditional mean on the first fiber of the statistic: the vanishing
condition fails, as the sharpness theorem says it must. -/
theorem xi_fiber_ne_zero :
    ∑ ω ∈ univ.filter (fun ω => S ω = 0), w ω • xi ω ≠ 0 := by
  have h : univ.filter (fun ω : Fin 4 => S ω = 0) = {0, 1} := by decide
  rw [h]
  intro hc
  have := congrFun hc 0
  simp [w, xi] at this

/-- The frozen coupling of the witness is one. -/
theorem coupling_eq : outerMoment w dth xi = !![1] := by
  rw [dth_eq]
  ext i j
  fin_cases i; fin_cases j
  simp [outerMoment, Fin.sum_univ_four, w, xi]
  norm_num

/-- **Two different functions of one batch are coupled**: `Ξ ≠ 0` on the witness. -/
theorem coupling_ne_zero : outerMoment w dth xi ≠ 0 := by
  rw [coupling_eq]
  intro h
  have := congrFun (congrFun h 0) 0
  simp at this

end SameBatchWitness

namespace LocationWitness

/-- Uniform weights on a two-point sample space. -/
noncomputable def w : Fin 2 → ℝ := fun _ => (1 : ℝ) / 2

/-- The latent weight, `h(X) = X` for a centred two-point `X`. -/
def h : Fin 2 → Fin 1 → ℝ := ![![1], ![-1]]

/-- The gradient noise of the location loss `(θ − X)²/2` at any iterate, `ξ = −X`. -/
def xi : Fin 2 → Fin 1 → ℝ := ![![-1], ![1]]

theorem h_mean_zero : ∑ ω, w ω • h ω = 0 := by
  ext i; fin_cases i
  simp [Fin.sum_univ_two, w, h]

theorem xi_mean_zero : ∑ ω, w ω • xi ω = 0 := by
  ext i; fin_cases i
  simp [Fin.sum_univ_two, w, xi]

/-- With the latent weight aligned against the gradient noise the frozen coupling is negative. -/
theorem coupling_neg : outerMoment w h xi = !![-1] := by
  ext i j
  fin_cases i; fin_cases j
  simp [outerMoment, Fin.sum_univ_two, w, h, xi]
  norm_num

/-- With the latent weight's sign flipped the frozen coupling is positive. The sign of `Ξ` is the sign
of the alignment, which is what assumption (A6) assumes and the construction does not fix. -/
theorem coupling_pos : outerMoment w (fun ω => -h ω) xi = !![1] := by
  ext i j
  fin_cases i; fin_cases j
  simp [outerMoment, Fin.sum_univ_two, w, h, xi]
  norm_num

end LocationWitness

/-! ## The three-term decomposition and the projected identities, for the real covariances

The vocabulary is `UpdateCovariance.covMat`, the covariance matrix of two square-integrable random
vectors on a finite measure space, and `UpdateCovariance.projDiffusion`, the projected update
covariance. The frozen coupling is `covMat μ dth g₀ = E[δθ̃ ξᵀ]`, since the covariance centres
both arguments. -/

section MeasureTheoretic

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Type*} [Fintype n]

/-- **The three-term decomposition of the slack-channel cross-covariance.** Under the refined chain
`g = g₀ + H δθ̃ + r₂` with symmetric `H`,

  `Σ_slack = E[δθ̃ δgᵀ] = Ξ + V H + R₂`,

with `Ξ = Cov[δθ̃, g₀]` the frozen coupling, `V = Cov[δθ̃]` and `R₂ = Cov[δθ̃, r₂]`. It is bilinearity
of the covariance and nothing else; no term is dropped and no size is assumed. -/
theorem covMat_slack_refined [IsFiniteMeasure μ] {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g g₀ dth r₂ : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth) (h₀ : SqIntegrableVec μ g₀)
    (hr : SqIntegrableVec μ r₂) (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω) :
    covMat μ dth g = covMat μ dth g₀ + covMat μ dth dth * H + covMat μ dth r₂ := by
  have hg : g = fun ω => (g₀ ω + H *ᵥ dth ω) + r₂ ω := funext hchain
  subst hg
  rw [covMat_add_right hθ (h₀.add (hθ.mulVec H)) hr, covMat_add_right hθ h₀ (hθ.mulVec H),
    covMat_mulVec_right H hθ hθ, hH]

omit [Fintype n] in
/-- **A latent weight independent of the batch driving the gradient has zero frozen coupling.**
Independence of the two random vectors passes to every pair of coordinates, and independent
square-integrable variables have zero covariance. -/
theorem covMat_eq_zero_of_indepFun [IsFiniteMeasure μ] {U W : Ω → n → ℝ}
    (hind : IndepFun U W μ) (hU : SqIntegrableVec μ U) (hW : SqIntegrableVec μ W) :
    covMat μ U W = 0 := by
  ext i j
  rw [covMat_apply, Matrix.zero_apply]
  have h : IndepFun (fun ω => U ω i) (fun ω => W ω j) μ :=
    hind.comp (measurable_pi_apply i) (measurable_pi_apply j)
  exact h.covariance_eq_zero (hU i) (hW j)

/-- **Under independence the self-coupling survives.** With the latent-weight fluctuation independent of
the frozen gradient, the cross-covariance reduces to `V H + R₂`: the estimator of `eq:cross-cov`
does not read zero in the matched-magnitude cell, although the coupling channel is empty there.
The population zero of Theorem 1's third deletion is a zero of `Ξ`. -/
theorem covMat_slack_refined_of_indepFun [IsFiniteMeasure μ] {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g g₀ dth r₂ : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth) (h₀ : SqIntegrableVec μ g₀)
    (hr : SqIntegrableVec μ r₂) (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hind : IndepFun dth g₀ μ) :
    covMat μ dth g = covMat μ dth dth * H + covMat μ dth r₂ := by
  rw [covMat_slack_refined hH hθ h₀ hr hchain, covMat_eq_zero_of_indepFun hind hθ h₀, zero_add]

/-- **The subtraction of the (A3) discussion, taken from the chain itself.** At a vanishing
chain remainder the split of `lem:diffusion`(i) reads `Σ_slack = Σ_ξ + V H`, so
`CouplingSign.crossTerm_frozen_neg_of_slack_neg` applies with the split discharged rather than
assumed: a direction in which the full slack channel's coupling form is negative is one in which
the frozen coupling's is. This is the appendix sentence "the coupling form of `Σ_ξ` is that of
`Σ_slack` less the non-negative quantity `2 eᵀ H V H e`, by the split of `lem:diffusion`(i) at
`r₂ = 0`, so it is negative in the same direction", end to end. -/
theorem crossTerm_frozen_neg_of_slack_neg_of_exactChain [IsFiniteMeasure μ] {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g g₀ dth : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth)
    (h₀ : SqIntegrableVec μ g₀) (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω) {e : n → ℝ}
    (hneg : crossTerm H (covMat μ dth g) e < 0) :
    crossTerm H (covMat μ dth g₀) e < 0 := by
  have hsplit : covMat μ dth g = covMat μ dth g₀ + covMat μ dth dth * H := by
    have hg : g = fun ω => g₀ ω + H *ᵥ dth ω := funext hchain
    subst hg
    rw [covMat_add_right hθ h₀ (hθ.mulVec H), covMat_mulVec_right H hθ hθ, hH]
  exact crossTerm_frozen_neg_of_slack_neg (covMat_self_posSemidef hθ)
    (isHermitian_of_transpose_eq hH) hsplit hneg

/-- **What the measured trace reports.** The trace of the cross-covariance the estimator targets is
`tr Ξ + tr (V H) + tr R₂`, and the middle term is non-negative whenever the mean Hessian is
positive semidefinite. A positive measured `σ²_Jac` therefore bounds `tr Ξ + tr R₂` from above and
does not by itself sign the frozen coupling. -/
theorem trace_covMat_slack_refined [DecidableEq n] [IsFiniteMeasure μ] {H : Matrix n n ℝ}
    (hH : H.PosSemidef) {g g₀ dth r₂ : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth)
    (h₀ : SqIntegrableVec μ g₀) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω) :
    (covMat μ dth g).trace
      = (covMat μ dth g₀).trace + (covMat μ dth dth * H).trace + (covMat μ dth r₂).trace
      ∧ 0 ≤ (covMat μ dth dth * H).trace := by
  have hHt : Hᵀ = H := transpose_eq_of_isHermitian hH.isHermitian
  refine ⟨?_, trace_mul_nonneg_of_posSemidef (covMat_self_posSemidef hθ) hH⟩
  rw [covMat_slack_refined hHt hθ h₀ hr hchain, Matrix.trace_add, Matrix.trace_add]

/-- The cross term is additive in the cross-covariance. -/
theorem crossTerm_add (H S₁ S₂ : Matrix n n ℝ) (e : n → ℝ) :
    crossTerm H (S₁ + S₂) e = crossTerm H S₁ e + crossTerm H S₂ e := by
  have h : couplingTerm H (S₁ + S₂) = couplingTerm H S₁ + couplingTerm H S₂ := by
    simp only [couplingTerm, Matrix.mul_add, Matrix.transpose_add, Matrix.add_mul]
    abel
  rw [crossTerm, h, Matrix.add_mulVec, dotProduct_add]
  rfl

/-- The cross term is homogeneous in the cross-covariance. -/
theorem crossTerm_smul (c : ℝ) (H S : Matrix n n ℝ) (e : n → ℝ) :
    crossTerm H (c • S) e = c * crossTerm H S e := by
  have h : couplingTerm H (c • S) = c • couplingTerm H S := by
    simp only [couplingTerm, Matrix.mul_smul, Matrix.transpose_smul, Matrix.smul_mul, smul_add]
  rw [crossTerm, h, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
  rfl

/-- The magnitude matrix is homogeneous in the jitter covariance. -/
theorem magnitudeMatrix_smul (c : ℝ) (H V : Matrix n n ℝ) :
    magnitudeMatrix H (c • V) = c • magnitudeMatrix H V := by
  simp only [magnitudeMatrix, Matrix.mul_smul, Matrix.smul_mul]

/-- For symmetric `H`, `eᵀ (H M) e = (H e) ⬝ (M e)`. -/
theorem quadForm_mul_left_eq (H : Matrix n n ℝ) (hH : Hᵀ = H) (M : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ ((H * M) *ᵥ e) = (H *ᵥ e) ⬝ᵥ (M *ᵥ e) := by
  have hx : e ᵥ* H = H *ᵥ e := by
    conv_rhs => rw [← hH]
    rw [Matrix.mulVec_transpose]
  rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, hx]

/-- **The remainder channel of the refined chain, projected.** With the lift's remainder `s r₂`,
the six remainder matrices of `DiffusionOrdering.updateRemainderMatrix` project to three scalars:
the pairing of the frozen gradient with `r₂`, the pairing of the curvature-weighted jitter with
`r₂`, and the variance of `r₂`. -/
theorem remainderChannel_refined (s : ℝ) (H : Matrix n n ℝ) (g₀ dth r₂ : Ω → n → ℝ)
    (e : n → ℝ) :
    remainderChannel (updateRemainderMatrix μ s H g₀ dth (fun ω => s • r₂ ω)) e
      = s ^ 2 * (2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e))
          + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e)) := by
  have h1 : covMat μ g₀ (fun ω => s • r₂ ω) = s • covMat μ g₀ r₂ := covMat_smul_right μ s g₀ r₂
  have h2 : covMat μ (fun ω => s • r₂ ω) g₀ = s • covMat μ r₂ g₀ := covMat_smul_left μ s r₂ g₀
  have h3 : covMat μ dth (fun ω => s • r₂ ω) = s • covMat μ dth r₂ := covMat_smul_right μ s dth r₂
  have h4 : covMat μ (fun ω => s • r₂ ω) dth = s • covMat μ r₂ dth := covMat_smul_left μ s r₂ dth
  have h5 : covMat μ (fun ω => s • r₂ ω) (fun ω => s • r₂ ω) = (s * s) • covMat μ r₂ r₂ := by
    rw [covMat_smul_left, covMat_smul_right, smul_smul]
  have q1 : e ⬝ᵥ (covMat μ r₂ g₀ *ᵥ e) = e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e) := by
    rw [← covMat_transpose μ g₀ r₂, quadForm_transpose]
  have q2 : e ⬝ᵥ ((covMat μ r₂ dth * Hᵀ) *ᵥ e) = e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e) := by
    have h : covMat μ r₂ dth * Hᵀ = (H * covMat μ dth r₂)ᵀ := by
      rw [Matrix.transpose_mul, covMat_transpose]
    rw [h, quadForm_transpose]
  rw [remainderChannel, updateRemainderMatrix, h1, h2, h3, h4, h5]
  simp only [quadForm_add, quadForm_smul, Matrix.mul_smul, Matrix.smul_mul]
  rw [q1, q2]
  ring

/-- **The projected diffusion of the lift with the `ξ` cross-term explicit** (frozen-coupling
form). Under the refined chain, with the lifted update `s g` and the direct update `s g₀`,

  `σ²_lift = σ²_dir + s² (eᵀ (H Ξ + Ξᵀ H) e + eᵀ H V H e) + s² ϱ₂(e)`,

`Ξ = Cov[δθ̃, g₀]`, `ϱ₂(e) = 2 eᵀ Cov[g₀, r₂] e + 2 eᵀ H Cov[δθ̃, r₂] e + eᵀ Cov[r₂] e`. This is
`DiffusionOrdering.projDiffusion_excess_split` with its `g` recognized as the frozen-iterate
gradient and its remainder group computed; it is exact. -/
theorem projDiffusion_lift_eq_frozen_form [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g g₀ dth r₂ Glift Gdir : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hlift : ∀ ω, Glift ω = s • g ω) (hdir : ∀ ω, Gdir ω = s • g₀ ω) (e : n → ℝ) :
    projDiffusion μ e Glift
      = projDiffusion μ e Gdir
        + s ^ 2 * (crossTerm H (covMat μ dth g₀) e
            + e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
        + s ^ 2 * (2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e))
            + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e)) := by
  have hlift' : ∀ ω, Glift ω = s • g₀ ω + s • (H *ᵥ dth ω) + (fun ω => s • r₂ ω) ω := by
    intro ω
    rw [hlift ω, hchain ω]
    simp only [smul_add]
  have hsplit := projDiffusion_excess_split s hH h₀ hθ (hr.smul s) hlift' hdir e
  rw [hsplit, remainderChannel_refined s H g₀ dth r₂ e, couplingChannel, magnitudeChannel]
  ring

/-- **The same identity on the full cross-covariance: the jitter term enters with a minus sign.**
Writing the coupling term on the cross-covariance the estimator targets, `C_full = H Σ + Σᵀ H`
with `Σ = Cov[δθ̃, g]`,

  `σ²_lift = σ²_dir + s² (eᵀ C_full e − eᵀ H V H e) + s² ϱ'(e)`,

`ϱ'(e) = 2 eᵀ Cov[g₀, r₂] e + eᵀ Cov[r₂] e`. The sign is forced by the three-term decomposition:
`C_full` contains `2 H V H`, so the self-coupling is counted twice there and once in the truth. -/
theorem projDiffusion_lift_eq_full_form [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g g₀ dth r₂ Glift Gdir : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hlift : ∀ ω, Glift ω = s • g ω) (hdir : ∀ ω, Gdir ω = s • g₀ ω) (e : n → ℝ) :
    projDiffusion μ e Glift
      = projDiffusion μ e Gdir
        + s ^ 2 * (crossTerm H (covMat μ dth g) e
            - e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
        + s ^ 2 * (2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e)) := by
  have hHerm : H.IsHermitian := isHermitian_of_transpose_eq hH
  have hV : crossTerm H (covMat μ dth dth * H) e
      = 2 * (e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e)) :=
    crossTerm_eq_two_mul_conj_of_exact_chain hHerm rfl e
  have hR : crossTerm H (covMat μ dth r₂) e = 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e)) :=
    crossTerm_eq_two_mul hHerm _ e
  have hfull : crossTerm H (covMat μ dth g) e
      = crossTerm H (covMat μ dth g₀) e
        + 2 * (e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
        + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e)) := by
    rw [covMat_slack_refined hH hθ h₀ hr hchain, crossTerm_add, crossTerm_add, hV, hR]
  rw [projDiffusion_lift_eq_frozen_form s hH h₀ hθ hr hchain hlift hdir e, hfull]
  ring

/-- **The display over-counts on the full reading.** The value the displayed lemma assigns when its
`Σ_slack` is read as the full cross-covariance, `σ²_dir + s² (eᵀ C_full e + eᵀ H V H e)`, exceeds
the true projected diffusion by `2 s² eᵀ H V H e` less the genuine remainder scalars. Since the
jitter term is non-negative and in general positive, the display is exact only when its `Σ_slack`
is the frozen coupling `Ξ`. -/
theorem display_sub_projDiffusion_lift [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g g₀ dth r₂ Glift Gdir : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hlift : ∀ ω, Glift ω = s • g ω) (hdir : ∀ ω, Gdir ω = s • g₀ ω) (e : n → ℝ) :
    (projDiffusion μ e Gdir
        + s ^ 2 * (crossTerm H (covMat μ dth g) e
            + e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e)))
      - projDiffusion μ e Glift
      = s ^ 2 * (2 * (e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
          - (2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e))) := by
  rw [projDiffusion_lift_eq_full_form s hH h₀ hθ hr hchain hlift hdir e]
  ring

end MeasureTheoretic

/-! ## The corrected regime statement, and the clause it replaces -/

section Regime

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Type*} [Fintype n]

/-- **The corrected regime statement.** Under the refined chain, positive alignment of the frozen
coupling in the direction `e` — assumption (A6) read on `Ξ`, `κ ‖e‖² ≤ eᵀ (H Ξ + Ξᵀ H) e` — together
with control of the three remainder scalars, `|ϱ₂(e)| ≤ τ ‖e‖²` with `τ < κ`, gives the strict
ordering `σ²_dir < σ²_lift`. The jitter term is a free non-negative summand and no bound relating
the coupling to the magnitude channel is asked for. This is what replaces (A4'-q). -/
theorem projDiffusion_direct_lt_lift_of_frozenAlignment [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g g₀ dth r₂ Glift Gdir : Ω → n → ℝ}
    (h₀ : SqIntegrableVec μ g₀) (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hlift : ∀ ω, Glift ω = s • g ω) (hdir : ∀ ω, Gdir ω = s • g₀ ω)
    {kappa tau : ℝ} {e : n → ℝ}
    (hA6 : kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g₀) e)
    (hrem : |2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e))
        + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e)| ≤ tau * (e ⬝ᵥ e))
    (htau : tau < kappa) (he : e ≠ 0) (hs : s ≠ 0) :
    projDiffusion μ e Gdir < projDiffusion μ e Glift := by
  rw [projDiffusion_lift_eq_frozen_form s hH h₀ hθ hr hchain hlift hdir e]
  have hJ : 0 ≤ e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e) :=
    quadForm_conj_nonneg (covMat_self_posSemidef hθ) (isHermitian_of_transpose_eq hH) e
  have hpos : 0 < e ⬝ᵥ e := dotProduct_self_pos he
  have hs2 : 0 < s ^ 2 := by positivity
  have hlow := (abs_le.mp hrem).1
  have hgain : 0 < (kappa - tau) * (e ⬝ᵥ e) := mul_pos (by linarith) hpos
  nlinarith

/-- **Net positive alignment (the merged hypothesis of the paper's (A3), 2026-09-11).**
The paper's earlier pair -- alignment `κ_C ‖e‖² ≤ eᵀ(HΣ_ξ + Σ_ξᵀH)e` (old (A6)) and the
remainder bound `|ϱ(e)| ≤ τ ‖e‖²` with `τ < κ_C` (old (A4'-r)) -- only ever entered the
ordering through the difference `κ_C - τ`. This statement carries the one inequality the
proof consumes: the coupling term plus the remainder channel dominates `κ ‖e‖²`. Under it
the lifted projected diffusion exceeds the direct one by at least `s² κ ‖e‖²`, because the
jitter term `eᵀ H V H e` is a quadratic form of a PSD matrix and never subtracts. -/
theorem projDiffusion_lift_ge_direct_add_of_netAlignment [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g g₀ dth r₂ Glift Gdir : Ω → n → ℝ}
    (h₀ : SqIntegrableVec μ g₀) (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hlift : ∀ ω, Glift ω = s • g ω) (hdir : ∀ ω, Gdir ω = s • g₀ ω)
    {kappa : ℝ} {e : n → ℝ}
    (hnet : kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g₀) e
        + (2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e))
          + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e))) :
    projDiffusion μ e Gdir + s ^ 2 * (kappa * (e ⬝ᵥ e)) ≤ projDiffusion μ e Glift := by
  rw [projDiffusion_lift_eq_frozen_form s hH h₀ hθ hr hchain hlift hdir e]
  have hJ : 0 ≤ e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e) :=
    quadForm_conj_nonneg (covMat_self_posSemidef hθ) (isHermitian_of_transpose_eq hH) e
  have hs2 : 0 ≤ s ^ 2 := sq_nonneg s
  nlinarith [mul_le_mul_of_nonneg_left hnet hs2, mul_nonneg hs2 hJ]

/-- The strict ordering under net positive alignment: `κ > 0`, `e ≠ 0`, `s ≠ 0`. This is
the paper's Theorem 1(ii) / Lemma 1(iii) with the merged hypothesis. -/
theorem projDiffusion_direct_lt_lift_of_netAlignment [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g g₀ dth r₂ Glift Gdir : Ω → n → ℝ}
    (h₀ : SqIntegrableVec μ g₀) (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r₂)
    (hchain : ∀ ω, g ω = g₀ ω + H *ᵥ dth ω + r₂ ω)
    (hlift : ∀ ω, Glift ω = s • g ω) (hdir : ∀ ω, Gdir ω = s • g₀ ω)
    {kappa : ℝ} {e : n → ℝ}
    (hnet : kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g₀) e
        + (2 * (e ⬝ᵥ (covMat μ g₀ r₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ dth r₂) *ᵥ e))
          + e ⬝ᵥ (covMat μ r₂ r₂ *ᵥ e)))
    (hkappa : 0 < kappa) (he : e ≠ 0) (hs : s ≠ 0) :
    projDiffusion μ e Gdir < projDiffusion μ e Glift := by
  have h := projDiffusion_lift_ge_direct_add_of_netAlignment s hH h₀ hθ hr hchain hlift hdir hnet
  have hpos : 0 < e ⬝ᵥ e := dotProduct_self_pos he
  have hs2 : 0 < s ^ 2 := by positivity
  have hgain : 0 < s ^ 2 * (kappa * (e ⬝ᵥ e)) := by positivity
  linarith

/-- The earlier pair (old (A6) alignment with `κ_C`, old (A4'-r) remainder bound with `τ`,
`τ < κ_C`) implies net positive alignment with `κ = κ_C - τ > 0`; so the merged
hypothesis is the weaker one and the old theorem is a corollary of the new. -/
theorem netAlignment_of_frozenAlignment_and_remainder {H S R₀ R₁ R₂ : Matrix n n ℝ}
    {kappa tau : ℝ} {e : n → ℝ}
    (hA6 : kappa * (e ⬝ᵥ e) ≤ crossTerm H S e)
    (hrem : |2 * (e ⬝ᵥ (R₀ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * R₁) *ᵥ e)) + e ⬝ᵥ (R₂ *ᵥ e)|
        ≤ tau * (e ⬝ᵥ e)) :
    (kappa - tau) * (e ⬝ᵥ e) ≤ crossTerm H S e
      + (2 * (e ⬝ᵥ (R₀ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * R₁) *ᵥ e)) + e ⬝ᵥ (R₂ *ᵥ e)) := by
  have hlow := (abs_le.mp hrem).1
  nlinarith

/-- **(A4'-q) caps the mechanism by the magnitude channel.** Take the chain as the appendix writes
it, on the gradient of the expansion, `g₀ = H δθ̃ + r`, so that (A4'-q)'s `R = Cov[δθ̃, r]` equals
`Ξ − V H`; with the magnitude channel bounded below by `λ`, the symmetrized remainder pairing
bounded by `2ρ`, and the control clause `2ρ ≤ 3λ`, the frozen coupling in every direction is at
most five halves of the jitter term. -/
theorem remainderControl_caps_frozen_coupling [IsFiniteMeasure μ] {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g₀ dth r : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth)
    (hr : SqIntegrableVec μ r) (hchain : ∀ ω, g₀ ω = H *ᵥ dth ω + r ω) {lam rho : ℝ}
    (hlam : ∀ e : n → ℝ,
      lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
    (hrho : ∀ e : n → ℝ,
      |e ⬝ᵥ ((H * covMat μ dth r + (covMat μ dth r)ᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    (hA4q : 2 * rho ≤ 3 * lam) (e : n → ℝ) :
    e ⬝ᵥ ((H * covMat μ dth g₀) *ᵥ e)
      ≤ (5 / 2) * (e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e)) := by
  have hsl := covMat_slack_of_chain hH hθ hr hchain
  have hX : e ⬝ᵥ ((H * covMat μ dth g₀) *ᵥ e)
      = e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e)
        + e ⬝ᵥ ((H * covMat μ dth r) *ᵥ e) := by
    rw [hsl, Matrix.mul_add, quadForm_add, magnitudeMatrix, Matrix.mul_assoc]
  have hpair := dotProduct_symmetrizedPairing hH (covMat μ dth r) e
  have hR : e ⬝ᵥ ((H * covMat μ dth r) *ᵥ e) = (H *ᵥ e) ⬝ᵥ (covMat μ dth r *ᵥ e) :=
    quadForm_mul_left_eq H hH _ e
  have hup := (abs_le.mp (hrho e)).2
  have hl := hlam e
  have hee : 0 ≤ e ⬝ᵥ e := dotProduct_self_nonneg e
  have hrl : rho * (e ⬝ᵥ e) ≤ (3 / 2) * lam * (e ⬝ᵥ e) := by nlinarith
  rw [hX, hR]
  linarith

/-- **(A4'-q) is violated by the small-jitter regime.** Along the family `δθ̃ = t u` at a fixed
frozen gradient, the frozen coupling scales as `t` and the jitter term as `t²`, so the cap of the
previous theorem fails for every sufficiently small scale as soon as the frozen coupling is
positive in the direction `e`: no remainder `r`, no moduli `λ`, `ρ` and no control constant can
satisfy the three clauses of (A4'-q) together. The clause is therefore not a smallness regime that
the operating point can be brought into, but one it leaves as the jitter shrinks. -/
theorem remainderControl_fails_small_jitter [IsFiniteMeasure μ] {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g₀ u : Ω → n → ℝ} (hu : SqIntegrableVec μ u) {e : n → ℝ}
    (hX : 0 < e ⬝ᵥ ((H * covMat μ u g₀) *ᵥ e)) :
    ∃ t₀ > 0, ∀ t : ℝ, 0 < t → t < t₀ →
      ∀ (r : Ω → n → ℝ) (lam rho : ℝ), SqIntegrableVec μ r →
        (∀ ω, g₀ ω = H *ᵥ (t • u ω) + r ω) →
        (∀ e' : n → ℝ, lam * (e' ⬝ᵥ e')
            ≤ e' ⬝ᵥ (magnitudeMatrix H (covMat μ (fun ω => t • u ω) (fun ω => t • u ω)) *ᵥ e')) →
        (∀ e' : n → ℝ, |e' ⬝ᵥ ((H * covMat μ (fun ω => t • u ω) r
            + (covMat μ (fun ω => t • u ω) r)ᵀ * H) *ᵥ e')| ≤ 2 * rho * (e' ⬝ᵥ e')) →
        ¬ (2 * rho ≤ 3 * lam) := by
  set X : ℝ := e ⬝ᵥ ((H * covMat μ u g₀) *ᵥ e) with hXdef
  set J : ℝ := e ⬝ᵥ (magnitudeMatrix H (covMat μ u u) *ᵥ e) with hJdef
  have hJ : 0 ≤ J :=
    quadForm_conj_nonneg (covMat_self_posSemidef hu) (isHermitian_of_transpose_eq hH) e
  refine ⟨X / ((5 / 2) * J + 1), by positivity, ?_⟩
  intro t ht0 ht1 r lam rho hr hchain hlam hrho hA4q
  have hcap := remainderControl_caps_frozen_coupling hH (hu.smul t) hr hchain hlam hrho hA4q e
  have hΞ : covMat μ (fun ω => t • u ω) g₀ = t • covMat μ u g₀ := covMat_smul_left μ t u g₀
  have hV : covMat μ (fun ω => t • u ω) (fun ω => t • u ω) = (t * t) • covMat μ u u := by
    rw [covMat_smul_left, covMat_smul_right, smul_smul]
  rw [hΞ, hV, Matrix.mul_smul, magnitudeMatrix_smul, quadForm_smul, quadForm_smul] at hcap
  have hcap' : X ≤ (5 / 2) * t * J := by
    have : t * X ≤ (5 / 2) * (t * t * J) := by rw [hXdef, hJdef]; linarith
    nlinarith
  have hlt : t * ((5 / 2) * J + 1) < X := by
    have hden : 0 < (5 / 2) * J + 1 := by positivity
    calc t * ((5 / 2) * J + 1) < X / ((5 / 2) * J + 1) * ((5 / 2) * J + 1) :=
          mul_lt_mul_of_pos_right ht1 hden
      _ = X := by field_simp
  nlinarith

end Regime

/-! ## The small-jitter family

The latent-weight fluctuation is scaled as `t u` at a fixed frozen gradient, and the refined remainder
as `t w₁ + t² w₂`: the first-order part is the pairing of the per-batch Hessian fluctuation with the
jitter, whose coefficient is random, and the second-order part is the Taylor term. The projected
excess is then a polynomial in `t`, and its linear coefficient decides the ordering at small scale.
-/

section SmallJitter

/-- A real polynomial with positive linear coefficient and no constant term is positive on a
right neighbourhood of zero. -/
theorem pos_of_small_of_linear_coeff_pos {a b c d : ℝ} (ha : 0 < a) :
    ∃ t₀ > 0, ∀ t : ℝ, 0 < t → t < t₀ → 0 < t * a + t ^ 2 * b + t ^ 3 * c + t ^ 4 * d := by
  set M : ℝ := |b| + |c| + |d| with hM
  have hM0 : 0 ≤ M := by positivity
  refine ⟨min 1 (a / (M + 1)), lt_min one_pos (by positivity), ?_⟩
  intro t ht0 ht
  have ht1 : t < 1 := lt_of_lt_of_le ht (min_le_left _ _)
  have ht2 : t < a / (M + 1) := lt_of_lt_of_le ht (min_le_right _ _)
  have htM : t * (M + 1) < a := by
    have := mul_lt_mul_of_pos_right ht2 (by positivity : (0:ℝ) < M + 1)
    rwa [div_mul_cancel₀ _ (by positivity)] at this
  have ht2pos : 0 < t ^ 2 := by positivity
  have ht3le : t ^ 3 ≤ t ^ 2 := by nlinarith
  have ht4le : t ^ 4 ≤ t ^ 2 := by nlinarith
  have h1 : t ^ 2 * b ≥ -(t ^ 2 * |b|) := by
    have := mul_le_mul_of_nonneg_left (neg_abs_le b) ht2pos.le
    linarith
  have h2 : t ^ 3 * c ≥ -(t ^ 2 * |c|) := by
    have h2a : t ^ 3 * c ≥ t ^ 3 * (-|c|) :=
      mul_le_mul_of_nonneg_left (neg_abs_le c) (by positivity)
    have h2b : t ^ 3 * |c| ≤ t ^ 2 * |c| := mul_le_mul_of_nonneg_right ht3le (abs_nonneg c)
    linarith
  have h3 : t ^ 4 * d ≥ -(t ^ 2 * |d|) := by
    have h3a : t ^ 4 * d ≥ t ^ 4 * (-|d|) :=
      mul_le_mul_of_nonneg_left (neg_abs_le d) (by positivity)
    have h3b : t ^ 4 * |d| ≤ t ^ 2 * |d| := mul_le_mul_of_nonneg_right ht4le (abs_nonneg d)
    linarith
  have hsum : t * a + t ^ 2 * b + t ^ 3 * c + t ^ 4 * d ≥ t * a - t ^ 2 * M := by
    rw [hM]; linarith
  have h4 : 0 < t * a - t ^ 2 * M := by
    have : t * a - t ^ 2 * M = t * (a - t * M) := by ring
    rw [this]
    exact mul_pos ht0 (by nlinarith)
  linarith

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Type*} [Fintype n]

/-- **The projected excess along the small-jitter family is a polynomial in the scale**, exactly:

  `σ²_lift(t) − σ²_dir = s² [ t (eᵀ (H Ξ₁ + Ξ₁ᵀ H) e + 2 eᵀ Cov[g₀, w₁] e) + t² (…) + t³ (…) + t⁴ (…) ]`,

with `Ξ₁ = Cov[u, g₀]`. The linear coefficient is the frozen coupling plus the first-order
remainder pairing; the jitter term `eᵀ H V₁ H e` sits in the quadratic coefficient. -/
theorem projDiffusion_excess_small_jitter [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g₀ u w₁ w₂ : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hu : SqIntegrableVec μ u) (hw₁ : SqIntegrableVec μ w₁) (hw₂ : SqIntegrableVec μ w₂)
    (t : ℝ) {Glift Gdir : Ω → n → ℝ}
    (hlift : ∀ ω, Glift ω = s • (g₀ ω + H *ᵥ (t • u ω) + (t • w₁ ω + t ^ 2 • w₂ ω)))
    (hdir : ∀ ω, Gdir ω = s • g₀ ω) (e : n → ℝ) :
    projDiffusion μ e Glift - projDiffusion μ e Gdir
      = s ^ 2 * (t * (crossTerm H (covMat μ u g₀) e + 2 * (e ⬝ᵥ (covMat μ g₀ w₁ *ᵥ e)))
        + t ^ 2 * (e ⬝ᵥ (magnitudeMatrix H (covMat μ u u) *ᵥ e)
            + 2 * (e ⬝ᵥ (covMat μ g₀ w₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ u w₁) *ᵥ e))
            + e ⬝ᵥ (covMat μ w₁ w₁ *ᵥ e))
        + t ^ 3 * (2 * (e ⬝ᵥ ((H * covMat μ u w₂) *ᵥ e)) + 2 * (e ⬝ᵥ (covMat μ w₁ w₂ *ᵥ e)))
        + t ^ 4 * (e ⬝ᵥ (covMat μ w₂ w₂ *ᵥ e))) := by
  set dth : Ω → n → ℝ := fun ω => t • u ω with hdth
  set r₂ : Ω → n → ℝ := fun ω => t • w₁ ω + t ^ 2 • w₂ ω with hr₂
  have hθ : SqIntegrableVec μ dth := hu.smul t
  have hr : SqIntegrableVec μ r₂ := (hw₁.smul t).add (hw₂.smul (t ^ 2))
  have hchain : ∀ ω, (fun ω => g₀ ω + H *ᵥ dth ω + r₂ ω) ω = g₀ ω + H *ᵥ dth ω + r₂ ω :=
    fun _ => rfl
  have hlift' : ∀ ω, Glift ω = s • (fun ω => g₀ ω + H *ᵥ dth ω + r₂ ω) ω := fun ω => hlift ω
  have hmain := projDiffusion_lift_eq_frozen_form s hH h₀ hθ hr hchain hlift' hdir e
  have c1 : covMat μ dth g₀ = t • covMat μ u g₀ := covMat_smul_left μ t u g₀
  have c2 : covMat μ dth dth = (t * t) • covMat μ u u := by
    rw [hdth, covMat_smul_left, covMat_smul_right, smul_smul]
  have c3 : covMat μ g₀ r₂ = t • covMat μ g₀ w₁ + t ^ 2 • covMat μ g₀ w₂ := by
    rw [hr₂, covMat_add_right h₀ (hw₁.smul t) (hw₂.smul (t ^ 2)), covMat_smul_right,
      covMat_smul_right]
  have c4 : covMat μ dth r₂ = t • (t • covMat μ u w₁ + t ^ 2 • covMat μ u w₂) := by
    rw [hr₂, hdth, covMat_smul_left, covMat_add_right hu (hw₁.smul t) (hw₂.smul (t ^ 2)),
      covMat_smul_right, covMat_smul_right]
  have c5 : covMat μ r₂ r₂
      = (t * t) • covMat μ w₁ w₁ + (t ^ 2 * t ^ 2) • covMat μ w₂ w₂
        + ((t ^ 2 * t) • covMat μ w₁ w₂ + (t * t ^ 2) • covMat μ w₂ w₁) := by
    rw [hr₂, covMat_add_self (hw₁.smul t) (hw₂.smul (t ^ 2))]
    simp only [covMat_smul_left, covMat_smul_right, smul_smul]
  have q1 : e ⬝ᵥ (covMat μ w₂ w₁ *ᵥ e) = e ⬝ᵥ (covMat μ w₁ w₂ *ᵥ e) := by
    rw [← covMat_transpose μ w₁ w₂, quadForm_transpose]
  rw [hmain, c1, c2, c3, c4, c5]
  simp only [crossTerm_smul, magnitudeMatrix_smul, Matrix.mul_add, Matrix.mul_smul, quadForm_add,
    quadForm_smul]
  rw [q1]
  ring

/-- **Positive alignment at first order gives the strict ordering at every small scale.** The
jitter term and the remaining remainder scalars are of order two and higher, so no bound relating
them to the coupling is needed: the sign of the linear coefficient, the frozen coupling plus the
first-order remainder pairing, decides. -/
theorem projDiffusion_direct_lt_lift_small_jitter [IsFiniteMeasure μ] {s : ℝ} (hs : s ≠ 0)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g₀ u w₁ w₂ : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hu : SqIntegrableVec μ u) (hw₁ : SqIntegrableVec μ w₁) (hw₂ : SqIntegrableVec μ w₂)
    {e : n → ℝ}
    (hfirst : 0 < crossTerm H (covMat μ u g₀) e + 2 * (e ⬝ᵥ (covMat μ g₀ w₁ *ᵥ e))) :
    ∃ t₀ > 0, ∀ t : ℝ, 0 < t → t < t₀ → ∀ Glift Gdir : Ω → n → ℝ,
      (∀ ω, Glift ω = s • (g₀ ω + H *ᵥ (t • u ω) + (t • w₁ ω + t ^ 2 • w₂ ω))) →
      (∀ ω, Gdir ω = s • g₀ ω) →
      projDiffusion μ e Gdir < projDiffusion μ e Glift := by
  obtain ⟨t₀, ht₀, hpoly⟩ := pos_of_small_of_linear_coeff_pos
    (b := e ⬝ᵥ (magnitudeMatrix H (covMat μ u u) *ᵥ e)
            + 2 * (e ⬝ᵥ (covMat μ g₀ w₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ u w₁) *ᵥ e))
            + e ⬝ᵥ (covMat μ w₁ w₁ *ᵥ e))
    (c := 2 * (e ⬝ᵥ ((H * covMat μ u w₂) *ᵥ e)) + 2 * (e ⬝ᵥ (covMat μ w₁ w₂ *ᵥ e)))
    (d := e ⬝ᵥ (covMat μ w₂ w₂ *ᵥ e)) hfirst
  refine ⟨t₀, ht₀, fun t ht0 ht Glift Gdir hlift hdir => ?_⟩
  have h := projDiffusion_excess_small_jitter s hH h₀ hu hw₁ hw₂ t hlift hdir e
  have hp := hpoly t ht0 ht
  have hs2 : 0 < s ^ 2 := by positivity
  have : 0 < projDiffusion μ e Glift - projDiffusion μ e Gdir := by
    rw [h]; exact mul_pos hs2 hp
  linarith

/-- **Negative alignment at first order reverses the ordering at every small scale**: the lift
then diffuses less than the direct baseline in the direction `e`, and the jitter term, being of
second order, cannot rescue it. Alignment is a genuine hypothesis and not a formality. -/
theorem projDiffusion_lift_lt_direct_small_jitter [IsFiniteMeasure μ] {s : ℝ} (hs : s ≠ 0)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g₀ u w₁ w₂ : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hu : SqIntegrableVec μ u) (hw₁ : SqIntegrableVec μ w₁) (hw₂ : SqIntegrableVec μ w₂)
    {e : n → ℝ}
    (hfirst : crossTerm H (covMat μ u g₀) e + 2 * (e ⬝ᵥ (covMat μ g₀ w₁ *ᵥ e)) < 0) :
    ∃ t₀ > 0, ∀ t : ℝ, 0 < t → t < t₀ → ∀ Glift Gdir : Ω → n → ℝ,
      (∀ ω, Glift ω = s • (g₀ ω + H *ᵥ (t • u ω) + (t • w₁ ω + t ^ 2 • w₂ ω))) →
      (∀ ω, Gdir ω = s • g₀ ω) →
      projDiffusion μ e Glift < projDiffusion μ e Gdir := by
  obtain ⟨t₀, ht₀, hpoly⟩ := pos_of_small_of_linear_coeff_pos
    (b := -(e ⬝ᵥ (magnitudeMatrix H (covMat μ u u) *ᵥ e)
            + 2 * (e ⬝ᵥ (covMat μ g₀ w₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ u w₁) *ᵥ e))
            + e ⬝ᵥ (covMat μ w₁ w₁ *ᵥ e)))
    (c := -(2 * (e ⬝ᵥ ((H * covMat μ u w₂) *ᵥ e)) + 2 * (e ⬝ᵥ (covMat μ w₁ w₂ *ᵥ e))))
    (d := -(e ⬝ᵥ (covMat μ w₂ w₂ *ᵥ e))) (neg_pos.mpr hfirst)
  refine ⟨t₀, ht₀, fun t ht0 ht Glift Gdir hlift hdir => ?_⟩
  have h := projDiffusion_excess_small_jitter s hH h₀ hu hw₁ hw₂ t hlift hdir e
  have hp := hpoly t ht0 ht
  have hs2 : 0 < s ^ 2 := by positivity
  have : projDiffusion μ e Glift - projDiffusion μ e Gdir < 0 := by
    rw [h]
    have : t * (crossTerm H (covMat μ u g₀) e + 2 * (e ⬝ᵥ (covMat μ g₀ w₁ *ᵥ e)))
        + t ^ 2 * (e ⬝ᵥ (magnitudeMatrix H (covMat μ u u) *ᵥ e)
            + 2 * (e ⬝ᵥ (covMat μ g₀ w₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ u w₁) *ᵥ e))
            + e ⬝ᵥ (covMat μ w₁ w₁ *ᵥ e))
        + t ^ 3 * (2 * (e ⬝ᵥ ((H * covMat μ u w₂) *ᵥ e)) + 2 * (e ⬝ᵥ (covMat μ w₁ w₂ *ᵥ e)))
        + t ^ 4 * (e ⬝ᵥ (covMat μ w₂ w₂ *ᵥ e)) < 0 := by linarith
    exact mul_neg_of_pos_of_neg hs2 this
  linarith

/-- **The matched-magnitude cell starts at second order.** With the frozen coupling empty and the
first-order remainder pairing zero — a latent weight independent of the batch has both, the first by
`covMat_eq_zero_of_indepFun` and the second, in the finite-sample model, by
`outerMoment_prod_hessianFluct_eq_zero` — the projected excess along the family is `O(t²)`: the jitter term
is the leading contribution and it is of second order in the scale. This is the scaling reading of
the matched-magnitude control, in which injected noise at the lift's own scale is inert while the
lift is not. -/
theorem projDiffusion_excess_small_jitter_of_no_frozen_coupling [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g₀ u w₁ w₂ : Ω → n → ℝ} (h₀ : SqIntegrableVec μ g₀)
    (hu : SqIntegrableVec μ u) (hw₁ : SqIntegrableVec μ w₁) (hw₂ : SqIntegrableVec μ w₂)
    (t : ℝ) {Glift Gdir : Ω → n → ℝ}
    (hlift : ∀ ω, Glift ω = s • (g₀ ω + H *ᵥ (t • u ω) + (t • w₁ ω + t ^ 2 • w₂ ω)))
    (hdir : ∀ ω, Gdir ω = s • g₀ ω) (e : n → ℝ)
    (hΞ : covMat μ u g₀ = 0) (hw : covMat μ g₀ w₁ = 0) :
    projDiffusion μ e Glift - projDiffusion μ e Gdir
      = s ^ 2 * (t ^ 2 * (e ⬝ᵥ (magnitudeMatrix H (covMat μ u u) *ᵥ e)
            + 2 * (e ⬝ᵥ (covMat μ g₀ w₂ *ᵥ e)) + 2 * (e ⬝ᵥ ((H * covMat μ u w₁) *ᵥ e))
            + e ⬝ᵥ (covMat μ w₁ w₁ *ᵥ e))
        + t ^ 3 * (2 * (e ⬝ᵥ ((H * covMat μ u w₂) *ᵥ e)) + 2 * (e ⬝ᵥ (covMat μ w₁ w₂ *ᵥ e)))
        + t ^ 4 * (e ⬝ᵥ (covMat μ w₂ w₂ *ᵥ e))) := by
  rw [projDiffusion_excess_small_jitter s hH h₀ hu hw₁ hw₂ t hlift hdir e, hΞ, hw,
    crossTerm_eq_zero_of_slack_eq_zero]
  simp only [Matrix.zero_mulVec, dotProduct_zero, mul_zero, add_zero, zero_add]

end SmallJitter

/-! ## Where the refined chain comes from: per-batch Taylor expansion

The per-batch gradient map `G ω` is a map on `EuclideanSpace ℝ (Fin d)`, as in `TaylorChain.lean`,
and the first-order expansion at the frozen iterate is applied batch by batch. The frozen gradient
appears by construction, and so does the Hessian fluctuation once a mean Hessian is separated out.
-/

section PerBatchTaylor

/-- The per-batch gradient at the probe's own iterate `θ̄ + δθ̃(ω)`, in coordinates. -/
noncomputable def probeGrad {d : ℕ} {Ω : Type*}
    (G : Ω → EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (thb : EuclideanSpace ℝ (Fin d)) (dth : Ω → Fin d → ℝ) (ω : Ω) : Fin d → ℝ :=
  WithLp.ofLp (G ω (thb + WithLp.toLp 2 (dth ω)))

/-- The per-batch gradient at the frozen reference iterate `θ̄`, in coordinates; its fluctuation
across batches is the minibatch noise `ξ`. -/
noncomputable def frozenGrad {d : ℕ} {Ω : Type*}
    (G : Ω → EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (thb : EuclideanSpace ℝ (Fin d)) (ω : Ω) : Fin d → ℝ :=
  WithLp.ofLp (G ω thb)

/-- **The refined chain from the per-batch expansion.** If `A ω` is the derivative of the per-batch
gradient map at the frozen iterate, presented by the matrix `Hb ω`, then for any choice of mean
Hessian `H`

  `g(ω) = g₀(ω) + H δθ̃(ω) + ((H(ω) − H) δθ̃(ω) + r_T(ω))`,

with `r_T` the per-batch Taylor remainder of `TaylorChain.gradRemainder`. The frozen gradient is
not an added term: it is where the expansion starts. -/
theorem probeGrad_eq_frozen_add_hessian_add_remainder {d : ℕ} {Ω : Type*}
    (G : Ω → EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (A : Ω → EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (thb : EuclideanSpace ℝ (Fin d)) (Hb : Ω → Matrix (Fin d) (Fin d) ℝ)
    (H : Matrix (Fin d) (Fin d) ℝ) (dth : Ω → Fin d → ℝ)
    (hA : ∀ ω w, A ω w = WithLp.toLp 2 (Hb ω *ᵥ WithLp.ofLp w)) (ω : Ω) :
    probeGrad G thb dth ω
      = frozenGrad G thb ω + H *ᵥ dth ω
        + ((Hb ω - H) *ᵥ dth ω + gradRemainder (G ω) (A ω) thb dth ω) := by
  have h := gradFluct_eq_mulVec_add_remainder (G ω) (A ω) thb (Hb ω) dth (hA ω) ω
  rw [gradFluct, WithLp.ofLp_sub] at h
  rw [probeGrad, frozenGrad, Matrix.sub_mulVec]
  have h' : WithLp.ofLp (G ω (thb + WithLp.toLp 2 (dth ω)))
      = WithLp.ofLp (G ω thb) + (Hb ω *ᵥ dth ω + gradRemainder (G ω) (A ω) thb dth ω) := by
    rw [← h]; abel
  rw [h']; abel

/-- **The Taylor part of the refined remainder is quadratic in the jitter, batch by batch**, with
the per-batch Hessian-Lipschitz modulus `K ω`: `‖r_T(ω)‖ ≤ (K ω / 2) ‖δθ̃(ω)‖²`. The
Hessian-fluctuation part `(H(ω) − H) δθ̃(ω)` is linear in the jitter with a random coefficient and
is not covered by this bound; it is the first-order remainder `t w₁` of the small-jitter family. -/
theorem l2Norm_perBatch_remainder_le {d : ℕ} {Ω : Type*}
    {G : Ω → EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d)}
    {DG : Ω → EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)}
    {thb : EuclideanSpace ℝ (Fin d)} {s : Set (EuclideanSpace ℝ (Fin d))}
    {dth : Ω → Fin d → ℝ} {K : Ω → ℝ}
    (hs : Convex ℝ s) (hthb : thb ∈ s)
    (hG : ∀ ω, ∀ y ∈ s, HasFDerivAt (G ω) (DG ω y) y)
    (hK : ∀ ω, ∀ y ∈ s, ‖DG ω y - DG ω thb‖ ≤ K ω * ‖y - thb‖)
    (hmem : ∀ ω, thb + WithLp.toLp 2 (dth ω) ∈ s) (ω : Ω) :
    l2Norm (gradRemainder (G ω) (DG ω thb) thb dth ω) ≤ K ω / 2 * l2Norm (dth ω) ^ 2 :=
  l2Norm_gradRemainder_le hs hthb (hG ω) (hK ω) hmem ω

end PerBatchTaylor

end IcnnLift
