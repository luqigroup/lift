import Mathlib

/-!
# The mean slack update is exactly the gradient of the pullback landscape

The paper trains the lifted parametrization `θ̃ = b + h_φ(X)`, `θ = ψ(θ̃)`, by stochastic gradient
descent on the slack `b`, with the *same* mini-batch conditioning the latent weight and evaluating the
loss (`rem:bcond`). Written out, the per-step update of the slack is
`b_{t+1} = b_t - η G(b_t; X_t)` with `G(b; X) = ψ'(θ̃) ⬝ ∇_θ L(ψ(θ̃); X)` and `θ̃ = b + h_φ(X)`.

This file machine-checks fact (i) of `docs/design/lemma1_rederivation.md` §2 (lines 27–50): the
mean of that update is exactly the derivative of the pullback landscape
`L̃(b) = E_X[L(ψ(b + h_φ(X)); X)]`. What makes the statement worth checking is that no
independence of any kind is assumed or used anywhere below: the per-batch loss `L ω` and the body
`h ω` are evaluated at the same sample point `ω` and averaged under one measure, exactly as in the
paper's protocol, and the mean update is nevertheless an exact gradient field. The precise reading
matters, and §9 of the note warns against the looser one: the coupling between latent weight and loss
is not claimed to leave `L̃` itself alone — with a batch-dependent loss the landscape is a joint
average over the coupled pair and does depend on that coupling — but the drift it produces is
`∇L̃` and nothing besides, so no first-order effect of the coupling survives once the landscape is
accounted for. The identity is an equality, not a leading-order statement, so there is no
remainder term to carry.

Two things are proved rather than assumed. The first is the chain rule through the positivity map, which
is where the prefactor `ψ'(θ̃)` in `G` comes from. `updateDirection` is *defined* the way the note
defines `G` — the differential of the per-batch loss at the produced weight, precomposed with the
differential of the positivity map — and `hasFDerivAt_updateDirection` proves that this composite is the
derivative of `b ↦ L(ψ(b + h_φ(X)); X)`. The slack's identity Jacobian `∂θ̃/∂b = I` of
`sec:lift-channels` appears separately as `hasFDerivAt_preReadout`. In the Hilbert-space setting
the same prefactor is the adjoint of the positivity-map differential
(`hasGradientAt_updateGradient`), and it loses its transpose exactly when that differential is
self-adjoint, which is the case for the paper's coordinatewise positivity map
(`updateGradient_of_isSelfAdjoint`).

The second is differentiation under the expectation, in the two regimes the assignment asks for.
`hasFDerivAt_pullbackLandscape_finiteBatch` is unconditional for a finite batch — a finite
measure on a finite sample space — where the expectation is a finite weighted sum and no
integrability or domination hypothesis is needed. `hasFDerivAt_pullbackLandscape_of_dominated`
carries the standard Leibniz hypotheses for a general batch law: local a.e. measurability,
integrability of the integrand at the base point, and a locally uniform integrable bound on `‖G‖`.
Those hypotheses are not a way of assuming the conclusion; differentiation under an integral is
false without them, and the proof is a direct application of mathlib's
`hasFDerivAt_integral_of_dominated_of_fderiv_le`.

The consequences the note draws are then corollaries. Written at the level of the update rule
itself, one mean step of `b_{t+1} = b_t - η G(b_t; X_t)` is a plain gradient step on the pullback
landscape (`integral_slackStep`). At a critical point of that landscape the mean update vanishes,
and conversely (`meanUpdate_eq_zero_iff`). The mean update depends on the batch law *only* through
`L̃`, so two latent weight schemes with the same pullback landscape near `b` have literally the same
drift there (`meanUpdate_eq_of_pullbackLandscape_eventuallyEq`), whatever the coupling between
latent weight and gradient inside each scheme; `meanUpdate_coupledTwoPoint_eq_uncoupled` exhibits an
actual such pair — a jitter-free one-point scheme, and a two-point scheme whose latent weight and whose
per-batch loss ride the same draw — while `meanUpdate_coupledTwoPoint` computes their common
drift in closed form and `meanUpdate_coupledTwoPoint_ne_zero` shows it is nonzero, so that
corollary is not the identity `0 = 0` in disguise. When the loss does not
itself depend on the batch, `L̃` is the composite `Λ ∘ ψ` averaged against the law of the latent weight
jitter (`pullbackLandscape_eq_integral_jitterLaw`) — the convolution reading of §9 — and the drift
then sees the latent weight only through that single marginal law (`meanUpdate_eq_of_jitterLaw_eq`).
That last step is not free: the landscape identity holds pointwise, a pointwise identity
constrains no derivative, and the drift statement therefore needs the identity, and with it the
measurability hypothesis, on a whole neighbourhood of `b`. A batch-independent body, the
direct-softplus baseline included, descends the unsmoothed landscape
(`pullbackLandscape_of_body_const`). Finally, `hasDerivAt_pullbackLandscape_along_meanUpdate` and
`pullbackLandscape_lt_of_meanUpdate_ne_zero` give the word "descends" its literal meaning: the
pullback landscape has derivative `-‖E_X[G]‖²` along the mean update, and strictly decreases for all
small enough step sizes whenever the mean update is nonzero.

One point of scope about the jitter-law group, since the design note has moved past §9. Section 9
read the convolution as the paper's mechanism (its "Leg 1"); §10 of the same note records the E1
verdict refuting that reading, on the ground that matched-marginal noise reproduces none of the
lift's advantage. Nothing below depends on §9's reading and nothing below endorses it. What the
jitter-law group says is only that the drift is a functional of the marginal jitter law, which is
a statement about what the first-order dynamics can see; read together with §10 it is one more
reason the mechanism cannot be a drift effect.

Together these are the precise sense in which the paper's mechanism cannot live in the drift: any
"added drift curvature" reading of Lemma 1 is refuted at first order, which is what forces the
corrected route to the diffusion (§2 fact (ii)).

## Results
* `preReadout`, `pullbackLandscape`, `updateDirection` — the lifted iterate `θ̃`, the landscape
  `L̃`, and the per-batch update direction `G(b; X)` of §2.
* `hasFDerivAt_preReadout` — the slack channel has identity Jacobian, `∂θ̃/∂b = I`.
* `hasFDerivAt_updateDirection` — the chain rule through the positivity map: `G(b; X)` is the derivative
  of `b ↦ L(ψ(b + h_φ(X)); X)`.
* `pullbackLandscape_eq_weightedSum` — on a finite batch the landscape is a weighted sum, the
  bridge to the finite-weight model of `PullbackHessian.lean`.
* `hasFDerivAt_pullbackLandscape_finiteBatch` — fact (i) for a finite batch, with no analytic side
  condition beyond differentiability.
* `integrable_updateDirection_of_dominated` — the domination bound makes `G(b; ·)` integrable.
* `hasFDerivAt_pullbackLandscape_of_dominated` — fact (i) for a general batch law under domination.
* `fderiv_pullbackLandscape_eq_meanUpdate` — the same statement as an equation,
  `∇_b L̃(b) = E_X[G]`.
* `updateGradient`, `toDual_updateGradient`, `hasGradientAt_updateGradient` — the gradient form of
  the update direction, with `ψ'(θ̃)` appearing as the adjoint of the positivity-map differential.
* `updateGradient_of_isSelfAdjoint` — for a self-adjoint positivity-map differential, the paper's
  untransposed formula `G = ψ'(θ̃) ∇_θ L`.
* `toDual_integral` — the Riesz isometry commutes with the Bochner integral.
* `hasGradientAt_pullbackLandscape`, `gradient_pullbackLandscape_eq_meanUpdate` — fact (i) in the
  paper's own notation, `∇_b L̃(b) = E_X[G(b; X)]` as vectors.
* `gradient_pullbackLandscape_finiteBatch` — the vector form on a finite batch, with
  differentiability as the only hypothesis: the reader-facing statement for the paper's actual
  mini-batches.
* `integral_slackStep` — the paper's per-step rule in the mean:
  `E_X[b_{t+1}] = b_t - η ∇_b L̃(b_t)`.
* `meanUpdate_eq_zero_iff` — the mean update vanishes exactly at critical points of `L̃`.
* `meanUpdate_eq_of_pullbackLandscape_eventuallyEq` — pullback landscapes agreeing near `b` give
  equal drift at `b`, for two schemes sharing nothing but the slack space: sample spaces, batch
  laws, bodies, positivity maps, positivity-map codomains and per-batch losses are all unrelated.
* `hasDerivAt_along_step`, `eventually_lt_of_hasDerivAt_neg`,
  `hasDerivAt_pullbackLandscape_along_meanUpdate`, `pullbackLandscape_lt_of_meanUpdate_ne_zero` —
  the mean update is a strict descent direction for `L̃`.
* `pullbackLandscape_eq_integral_jitterLaw`, `pullbackLandscape_eq_of_jitterLaw_eq`,
  `pullbackLandscape_eventuallyEq_of_jitterLaw_eq` — for a batch-independent loss the landscape is
  `Λ ∘ ψ` averaged against the jitter law; equal jitter laws give equal landscapes, at the base
  point and then on a whole neighbourhood of it.
* `meanUpdate_eq_of_jitterLaw_eq` — hence, for a batch-independent loss, equal jitter laws give
  equal drift. This is the drift statement; the pointwise landscape identity alone does not give
  it.
* `pullbackLandscape_of_body_const` — a batch-independent body, the direct-softplus baseline
  included, leaves the landscape unsmoothed. The per-batch loss may still depend on the batch.
* `updateDirection_linear_model` — non-vacuity of the definition: in a scalar linear model the
  update direction evaluates in closed form to `(s·c) • id`, literally `ψ'·∇L`, and in particular
  is nonzero whenever `s·c ≠ 0`.
* `hasFDerivAt_pullbackLandscape_twoPointBatch`, `hasFDerivAt_pullbackLandscape_uniformBatch`,
  `gradient_pullbackLandscape_twoPointBatch` — non-vacuity of the hypotheses: both
  differentiation regimes fire on concrete batch laws, a two-point batch and the uniform law on
  `[0, 1]`, and on the former the vector form of the identity is reached end to end.
* `pullbackLandscape_coupledTwoPoint_eq_uncoupled`, `meanUpdate_coupledTwoPoint_eq_uncoupled`,
  `meanUpdate_coupledTwoPoint`, `meanUpdate_coupledTwoPoint_ne_zero` — non-vacuity of the corollary
  that carries the note's conclusion: two schemes on different sample spaces, one of them with
  latent weight and loss on the same draw, have equal landscapes and hence equal drift, and that common
  drift is `(4 s² b) • id`, nonzero away from the minimum.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. Every statement here is an exact identity or an exact
consequence of one; nothing is "to leading order", so no remainder is suppressed.

The hypotheses carried are of four kinds, none of which assumes the conclusion. (a)
Differentiability of the positivity map `ψ` and of the per-batch loss `L(·; X)` at the points where the
paper's formula evaluates them — at the base point only in the finite-batch regime, and at every
point of a fixed neighbourhood of `b`, for almost every batch, in the dominated regime: without
these the object `G(b; X)` the paper writes does not exist. (b) For a general batch law, the
dominated-differentiation package of the Leibniz rule — a.e. measurability of the integrand near
the base point and of `G(b; ·)` at it, integrability of the integrand at the base point, and a
locally uniform integrable bound on `‖G‖`. These are exactly mathlib's hypotheses; differentiation
under an integral genuinely fails without them, and the finite-batch theorem shows that on the
paper's actual mini-batches they cost nothing. (c) For the gradient form, a real inner-product
structure and completeness of the parameter space, without which `gradient` and the adjoint do
not exist, together with integrability of `G(b; ·)`; the integrability is not an extra assumption
in practice, since `integrable_updateDirection_of_dominated` derives it from the same domination
bound and `Integrable.of_finite` supplies it on a finite batch; and, for `integral_slackStep`
alone, that the batch law is a probability measure, which the paper's protocol supplies and
without which the constant iterate would be reweighted by the total mass. (d) For the jitter-law
group, almost-everywhere measurability of the body and of the smoothed integrand against the
jitter law, and — for the drift statement `meanUpdate_eq_of_jitterLaw_eq` — that measurability at
every point of a neighbourhood of `b` rather than at `b` alone, since a landscape identity holding
at a single point constrains no derivative. None of these hypothesis packages is vacuous, and the
file certifies this rather than asserting it: `updateDirection_linear_model`
evaluates the update direction in closed form (it is `ψ'·∇L`, nonzero whenever `s·c ≠ 0`),
`hasFDerivAt_pullbackLandscape_twoPointBatch` discharges the finite-batch hypotheses on a
two-point batch carrying a batch-dependent loss and a nonlinear positivity map,
`hasFDerivAt_pullbackLandscape_uniformBatch` discharges the entire dominated package on a
genuinely infinite batch space under the uniform law on `[0, 1]`, and
`gradient_pullbackLandscape_twoPointBatch` carries the first instance through the Riesz bridge,
so the vector form of fact (i) is certified reachable on an actual batch law rather than merely
stated. The corollary that carries the note's conclusion is certified the same way and not only
for satisfiability: `meanUpdate_coupledTwoPoint_eq_uncoupled` produces two schemes on different
sample spaces, one jitter-free and one whose body and whose per-batch loss are driven by the same
draw, whose landscapes coincide and whose drifts are therefore equal, while
`meanUpdate_coupledTwoPoint` and `meanUpdate_coupledTwoPoint_ne_zero` evaluate that common drift
and show it is not zero, so the corollary is not read on an instance where both sides vanish.

What this file does not do. It says nothing about the second derivative of `L̃`, which is the
subject of `PullbackHessian.lean`, and nothing about the update covariance, which is fact (ii) and
the subject of `UpdateCovariance.lean`; the present file is only the first-order half of §2. It
does not derive that the positivity-map differential of a coordinatewise positivity map is the diagonal matrix
`diag(ψ'(θ̃))` — that computation belongs to `ShoulderAttenuation.lean`, and here the
coordinatewise structure enters only through the hypothesis `IsSelfAdjoint` of
`updateGradient_of_isSelfAdjoint`, which is what removes the transpose from the paper's formula.
Finally, the convolution reading `pullbackLandscape_eq_integral_jitterLaw` is stated, as the
design note states it, for a loss that does not itself depend on the batch; with a batch-dependent
loss the pullback is a joint average and is not a convolution; what remains in that generality is
`meanUpdate_eq_of_pullbackLandscape_eventuallyEq`, which is more general in its data but assumes
agreement of the landscapes themselves rather than of the jitter laws.
-/

open MeasureTheory Filter InnerProductSpace
open scoped Topology RealInnerProductSpace

namespace IcnnLift

/-! ### The lifted iterate, its landscape, and the per-batch update direction -/

section Basic

variable {Ω : Type*} {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- The **latent iterate** `θ̃ = b + h_φ(X)` of `eq:lift`: the slack `b`, which is the
optimization variable, offset by the body's latent weight on the batch `ω`. -/
def preReadout (h : Ω → E) (b : E) (ω : Ω) : E := b + h ω

/-- The **pullback landscape** `L̃(b) = E_X[L(ψ(b + h_φ(X)); X)]` of
`docs/design/lemma1_rederivation.md` §1: the per-batch loss, evaluated at the produced weight
`θ = ψ(θ̃)` and averaged over the batch. The same sample point `ω` supplies both the body's
latent weight and the loss, so the batch coupling of `rem:bcond` is present in this definition. -/
noncomputable def pullbackLandscape [MeasurableSpace Ω] (L : Ω → F → ℝ) (ψ : E → F) (h : Ω → E)
    (μ : Measure Ω) (b : E) : ℝ :=
  ∫ ω, L ω (ψ (preReadout h b ω)) ∂μ

/-- The **per-batch update direction** `G(b; X) = ψ'(θ̃) ⬝ ∇_θ L(ψ(θ̃); X)` of §2, as a linear
functional on the slack space. It is defined by the paper's formula, as the differential of the
per-batch loss at the produced weight precomposed with the differential of the positivity map; that this
composite really is the derivative of `b ↦ L(ψ(b + h_φ(X)); X)` is the content of
`hasFDerivAt_updateDirection` and is proved, not built in. -/
noncomputable def updateDirection (L : Ω → F → ℝ) (ψ : E → F) (h : Ω → E) (b : E) (ω : Ω) :
    E →L[ℝ] ℝ :=
  (fderiv ℝ (L ω) (ψ (preReadout h b ω))).comp (fderiv ℝ ψ (preReadout h b ω))

/-- The slack channel has identity Jacobian: `∂θ̃/∂b = I`, the first-order statement of
`sec:lift-channels` that the slack enters the latent iterate additively. -/
theorem hasFDerivAt_preReadout (h : Ω → E) (b : E) (ω : Ω) :
    HasFDerivAt (fun b' : E => preReadout h b' ω) (ContinuousLinearMap.id ℝ E) b := by
  simpa [preReadout] using (hasFDerivAt_id b).add_const (h ω)

/-- **The chain rule through the positivity map.** For a fixed batch `ω`, the derivative in the slack of
`b ↦ L(ψ(b + h_φ(X)); X)` is exactly `G(b; X)`, the loss differential at the produced weight
composed with the positivity-map differential. This is where the prefactor `ψ'(θ̃)` of `eq:lift`'s
chain rule comes from. -/
theorem hasFDerivAt_updateDirection {L : Ω → F → ℝ} {ψ : E → F} {h : Ω → E} {b : E} {ω : Ω}
    (hψ : DifferentiableAt ℝ ψ (preReadout h b ω))
    (hL : DifferentiableAt ℝ (L ω) (ψ (preReadout h b ω))) :
    HasFDerivAt (fun b' => L ω (ψ (preReadout h b' ω))) (updateDirection L ψ h b ω) b := by
  have hinner : HasFDerivAt (fun b' : E => ψ (preReadout h b' ω))
      (fderiv ℝ ψ (preReadout h b ω)) b := by
    have := hψ.hasFDerivAt.comp b (hasFDerivAt_preReadout h b ω)
    simpa [Function.comp_def] using this
  exact hL.hasFDerivAt.comp b hinner

end Basic

/-! ### Fact (i): the mean update is the derivative of the pullback landscape -/

section Drift

variable {Ω : Type*} [MeasurableSpace Ω] {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

omit [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] in
/-- On a finite batch the pullback landscape is the weighted sum `∑ᵢ wᵢ (L ∘ ψ)(b + hᵢ)` with
`wᵢ` the mass the batch law puts on the `i`-th sample. This is the bridge to the finite-weight
model of `PullbackHessian.lean`, whose `pullbackLoss` takes the composite `L ∘ ψ` already formed;
here `ψ` is kept separate because the chain rule through the positivity map is what has to be proved. -/
theorem pullbackLandscape_eq_weightedSum [Fintype Ω] [MeasurableSingletonClass Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (L : Ω → F → ℝ) (ψ : E → F) (h : Ω → E) (b : E) :
    pullbackLandscape L ψ h μ b = ∑ ω, μ.real {ω} • L ω (ψ (b + h ω)) :=
  integral_fintype Integrable.of_finite

/-- **Fact (i) for a finite batch, unconditionally.** When the batch law is a finite measure on a
finite sample space — the paper's actual mini-batch, whose expectation is a finite weighted sum —
the mean update `E_X[G(b; X)]` is the derivative of the pullback landscape at `b`, with no
integrability or domination side condition: the only hypotheses are that the positivity map and each
per-batch loss are differentiable at the points where the paper's formula evaluates them. -/
theorem hasFDerivAt_pullbackLandscape_finiteBatch [Fintype Ω] [MeasurableSingletonClass Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] {L : Ω → F → ℝ} {ψ : E → F} {h : Ω → E} {b : E}
    (hψ : ∀ ω, DifferentiableAt ℝ ψ (preReadout h b ω))
    (hL : ∀ ω, DifferentiableAt ℝ (L ω) (ψ (preReadout h b ω))) :
    HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b := by
  have hval : ∀ b' : E, pullbackLandscape L ψ h μ b'
      = ∑ ω, μ.real {ω} • L ω (ψ (preReadout h b' ω)) := fun b' =>
    pullbackLandscape_eq_weightedSum μ L ψ h b'
  have hmean : ∫ ω, updateDirection L ψ h b ω ∂μ
      = ∑ ω, μ.real {ω} • updateDirection L ψ h b ω := integral_fintype Integrable.of_finite
  rw [hmean]
  have hsum : HasFDerivAt (fun b' : E => ∑ ω, μ.real {ω} • L ω (ψ (preReadout h b' ω)))
      (∑ ω, μ.real {ω} • updateDirection L ψ h b ω) b :=
    HasFDerivAt.fun_sum fun ω _ =>
      (hasFDerivAt_updateDirection (hψ ω) (hL ω)).const_smul (μ.real {ω})
  simpa only [← hval] using hsum

/-- The domination bound of the Leibniz rule already makes the update direction integrable at the
base point, so integrability of `G(b; ·)` is never an extra assumption in the dominated regime. -/
theorem integrable_updateDirection_of_dominated {μ : Measure Ω} {L : Ω → F → ℝ} {ψ : E → F}
    {h : Ω → E} {b : E} {s : Set E} {bound : Ω → ℝ} (hs : s ∈ 𝓝 b)
    (hmeas : AEStronglyMeasurable (updateDirection L ψ h b) μ)
    (hbound : ∀ᵐ ω ∂μ, ∀ b' ∈ s, ‖updateDirection L ψ h b' ω‖ ≤ bound ω)
    (hbint : Integrable bound μ) : Integrable (updateDirection L ψ h b) μ := by
  refine hbint.mono' hmeas ?_
  filter_upwards [hbound] with ω hω
  exact hω b (mem_of_mem_nhds hs)

/-- **Fact (i) for a general batch law, under domination.** If the per-batch loss and the positivity map
are differentiable near `b` for almost every batch, `‖G(b'; X)‖` is bounded on a fixed neighbourhood
of `b` by an integrable function of the batch, and the usual measurability and base-point
integrability hold, then the mean update `E_X[G(b; X)]` is exactly the derivative of the pullback
landscape at `b`.

The hypotheses are mathlib's `hasFDerivAt_integral_of_dominated_of_fderiv_le`, i.e. the classical
Leibniz conditions; differentiation under an integral is false without a hypothesis of this kind,
so they are the analytic content of the statement rather than a way around it. Note that nothing
here asks the latent weight and the gradient to be independent: both are functions of the same sample
point and are averaged under the same measure. -/
theorem hasFDerivAt_pullbackLandscape_of_dominated {μ : Measure Ω} {L : Ω → F → ℝ} {ψ : E → F}
    {h : Ω → E} {b : E} {s : Set E} {bound : Ω → ℝ} (hs : s ∈ 𝓝 b)
    (hmeas : ∀ᶠ b' in 𝓝 b, AEStronglyMeasurable (fun ω => L ω (ψ (preReadout h b' ω))) μ)
    (hint : Integrable (fun ω => L ω (ψ (preReadout h b ω))) μ)
    (hmeas' : AEStronglyMeasurable (updateDirection L ψ h b) μ)
    (hbound : ∀ᵐ ω ∂μ, ∀ b' ∈ s, ‖updateDirection L ψ h b' ω‖ ≤ bound ω)
    (hbint : Integrable bound μ)
    (hdiff : ∀ᵐ ω ∂μ, ∀ b' ∈ s, DifferentiableAt ℝ ψ (preReadout h b' ω) ∧
      DifferentiableAt ℝ (L ω) (ψ (preReadout h b' ω))) :
    HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b := by
  refine hasFDerivAt_integral_of_dominated_of_fderiv_le
    (F' := fun b' ω => updateDirection L ψ h b' ω) hs hmeas hint hmeas' hbound hbint ?_
  filter_upwards [hdiff] with ω hω b' hb'
  exact hasFDerivAt_updateDirection (hω b' hb').1 (hω b' hb').2

/-- Fact (i) restated as the equation the note writes: `∇_b L̃(b) = E_X[G(b; X)]`. -/
theorem fderiv_pullbackLandscape_eq_meanUpdate {μ : Measure Ω} {L : Ω → F → ℝ} {ψ : E → F}
    {h : Ω → E} {b : E}
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b) :
    fderiv ℝ (pullbackLandscape L ψ h μ) b = ∫ ω, updateDirection L ψ h b ω ∂μ := hD.fderiv

/-- **The mean update vanishes exactly at the critical points of the pullback landscape.** In
particular a stationary point of the lifted dynamics in the mean is a stationary point of `L̃`,
and no batch coupling can create or destroy one. -/
theorem meanUpdate_eq_zero_iff {μ : Measure Ω} {L : Ω → F → ℝ} {ψ : E → F} {h : Ω → E} {b : E}
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b) :
    (∫ ω, updateDirection L ψ h b ω ∂μ) = 0 ↔ fderiv ℝ (pullbackLandscape L ψ h μ) b = 0 := by
  rw [fderiv_pullbackLandscape_eq_meanUpdate hD]

/-- **The drift depends on the batch law only through the pullback landscape.** Two latent weight
schemes — different sample spaces, batch laws, bodies, positivity maps, positivity-map codomains and per-batch
losses, and therefore different couplings between latent weight and gradient — that produce the same
pullback landscape near `b` produce literally the same mean update at `b`, an element of the one
space they do share. This is the precise content of the note's
"the batch coupling deforms no potential at first order": the coupling may shape `L̃` itself — for
a batch-dependent loss it does — but it cannot make the drift anything other than `∇L̃`, so no
drift-level measurement separates two schemes whose landscapes agree. This is what rules out an
"added drift curvature" reading of the lemma. -/
theorem meanUpdate_eq_of_pullbackLandscape_eventuallyEq {Ω₁ Ω₂ F₁ F₂ : Type*} [MeasurableSpace Ω₁]
    [MeasurableSpace Ω₂] [NormedAddCommGroup F₁] [NormedSpace ℝ F₁] [NormedAddCommGroup F₂]
    [NormedSpace ℝ F₂] {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} {L₁ : Ω₁ → F₁ → ℝ} {L₂ : Ω₂ → F₂ → ℝ}
    {ψ₁ : E → F₁} {ψ₂ : E → F₂} {h₁ : Ω₁ → E} {h₂ : Ω₂ → E} {b : E}
    (hD₁ : HasFDerivAt (pullbackLandscape L₁ ψ₁ h₁ μ₁) (∫ ω, updateDirection L₁ ψ₁ h₁ b ω ∂μ₁) b)
    (hD₂ : HasFDerivAt (pullbackLandscape L₂ ψ₂ h₂ μ₂) (∫ ω, updateDirection L₂ ψ₂ h₂ b ω ∂μ₂) b)
    (heq : pullbackLandscape L₁ ψ₁ h₁ μ₁ =ᶠ[𝓝 b] pullbackLandscape L₂ ψ₂ h₂ μ₂) :
    (∫ ω, updateDirection L₁ ψ₁ h₁ b ω ∂μ₁) = ∫ ω, updateDirection L₂ ψ₂ h₂ b ω ∂μ₂ :=
  (hD₁.congr_of_eventuallyEq heq.symm).unique hD₂

end Drift

/-! ### Descent along the mean update -/

section Descent

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The derivative of a differentiable function along the step `b ↦ b - t v`, at `t = 0`. -/
theorem hasDerivAt_along_step {Lt : E → ℝ} {D : E →L[ℝ] ℝ} {b v : E} (hD : HasFDerivAt Lt D b) :
    HasDerivAt (fun t : ℝ => Lt (b - t • v)) (-(D v)) 0 := by
  have hline : HasDerivAt (fun t : ℝ => b - t • v) (-v) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_sub b
  have hcomp := hD.comp_hasDerivAt_of_eq 0 hline (by simp)
  simpa [Function.comp_def] using hcomp

/-- A strictly negative derivative at zero step size gives a genuine decrease for all small enough
positive step sizes. This is what upgrades "the derivative along the mean update is negative" to
"the step decreases the loss". -/
theorem eventually_lt_of_hasDerivAt_neg {Lt : E → ℝ} {b v : E} {c : ℝ} (hc : 0 < c)
    (hd : HasDerivAt (fun t : ℝ => Lt (b - t • v)) (-c) 0) :
    ∀ᶠ t in 𝓝[>] (0 : ℝ), Lt (b - t • v) < Lt b := by
  set G : ℝ → ℝ := fun t => Lt (b - t • v) with hG
  have hslope : Tendsto (slope G 0) (𝓝[≠] (0 : ℝ)) (𝓝 (-c)) := hasDerivAt_iff_tendsto_slope.1 hd
  have hev : ∀ᶠ t in 𝓝[≠] (0 : ℝ), slope G 0 t < 0 :=
    hslope.eventually_lt_const (by linarith)
  have hsub : 𝓝[>] (0 : ℝ) ≤ 𝓝[≠] (0 : ℝ) := nhdsWithin_mono _ fun x hx => ne_of_gt hx
  filter_upwards [hsub hev, self_mem_nhdsWithin] with t ht htpos
  have htpos' : (0 : ℝ) < t := htpos
  rw [slope_def_field, div_neg_iff] at ht
  have hlt : G t - G 0 < 0 := by
    rcases ht with ⟨h1, h2⟩ | ⟨h1, _⟩
    · exact absurd h2 (by simpa using htpos'.le)
    · exact h1
  have h0 : G 0 = Lt b := by simp [hG]
  have : G t < Lt b := by rw [← h0]; linarith
  simpa [hG] using this

end Descent

/-! ### The gradient form, and the positivity-map prefactor as an adjoint -/

section Hilbert

variable {Ω : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]

/-- The **per-batch update direction as a vector**, `G(b; X) = ψ'(θ̃)ᵀ ∇_θ L(ψ(θ̃); X)`: the
gradient of the per-batch loss at the produced weight, pulled back through the adjoint of the
positivity-map differential. The adjoint is the coordinate-free form of the prefactor `ψ'(θ̃)` of
`sec:lift-channels`, and it is what `updateDirection` becomes under the Riesz isometry. -/
noncomputable def updateGradient (L : Ω → H → ℝ) (ψ : H → H) (h : Ω → H) (b : H) (ω : Ω) : H :=
  ContinuousLinearMap.adjoint (fderiv ℝ ψ (preReadout h b ω))
    (gradient (L ω) (ψ (preReadout h b ω)))

/-- The vector and functional forms of the update direction correspond under the Riesz isometry. -/
theorem toDual_updateGradient (L : Ω → H → ℝ) (ψ : H → H) (h : Ω → H) (b : H) (ω : Ω) :
    toDual ℝ H (updateGradient L ψ h b ω) = updateDirection L ψ h b ω := by
  ext u
  simp [updateGradient, updateDirection, toDual_apply_apply,
    ContinuousLinearMap.adjoint_inner_left, gradient]

/-- **The chain rule through the positivity map, in gradient form.** For a fixed batch, the gradient in
the slack of `b ↦ L(ψ(b + h_φ(X)); X)` is the loss gradient at the produced weight pulled back
through the adjoint of the positivity-map differential. -/
theorem hasGradientAt_updateGradient {L : Ω → H → ℝ} {ψ : H → H} {h : Ω → H} {b : H} {ω : Ω}
    (hψ : DifferentiableAt ℝ ψ (preReadout h b ω))
    (hL : DifferentiableAt ℝ (L ω) (ψ (preReadout h b ω))) :
    HasGradientAt (fun b' => L ω (ψ (preReadout h b' ω))) (updateGradient L ψ h b ω) b := by
  rw [hasGradientAt_iff_hasFDerivAt, toDual_updateGradient]
  exact hasFDerivAt_updateDirection hψ hL

/-- When the positivity-map differential is self-adjoint — which is the case for the paper's
coordinatewise positivity map, whose differential is the diagonal matrix `diag(ψ'(θ̃))` — the adjoint in
`updateGradient` disappears and the update direction is the paper's untransposed
`ψ'(θ̃) ∇_θ L(ψ(θ̃); X)`. The diagonal form of the differential itself is not proved here; it is
the subject of `ShoulderAttenuation.lean`. -/
theorem updateGradient_of_isSelfAdjoint {L : Ω → H → ℝ} {ψ : H → H} {h : Ω → H} {b : H} {ω : Ω}
    (hA : IsSelfAdjoint (fderiv ℝ ψ (preReadout h b ω))) :
    updateGradient L ψ h b ω =
      fderiv ℝ ψ (preReadout h b ω) (gradient (L ω) (ψ (preReadout h b ω))) := by
  rw [updateGradient, ContinuousLinearMap.isSelfAdjoint_iff'.mp hA]

variable [MeasurableSpace Ω]

/-- The Riesz isometry commutes with the Bochner integral: the functional attached to a mean vector
is the mean of the attached functionals. -/
theorem toDual_integral {μ : Measure Ω} {g : Ω → H} (hg : Integrable g μ) :
    toDual ℝ H (∫ ω, g ω ∂μ) = ∫ ω, toDual ℝ H (g ω) ∂μ := by
  have hg' : Integrable (fun ω => toDual ℝ H (g ω)) μ :=
    (LinearIsometryEquiv.integrable_comp_iff (toDual ℝ H)).2 hg
  ext u
  rw [ContinuousLinearMap.integral_apply hg']
  have hswap : ∀ ω, (toDual ℝ H (g ω)) u = (toDual ℝ H u) (g ω) := by
    intro ω
    simp only [toDual_apply_apply]
    exact real_inner_comm _ _
  simp only [hswap]
  rw [ContinuousLinearMap.integral_comp_comm _ hg]
  simp only [toDual_apply_apply]
  exact real_inner_comm _ _

/-- **Fact (i) in the paper's own notation:** `∇_b L̃(b) = E_X[G(b; X)]`, as vectors. The
hypothesis `hD` is the functional form supplied by `hasFDerivAt_pullbackLandscape_finiteBatch` or
`hasFDerivAt_pullbackLandscape_of_dominated`; the integrability hypothesis is supplied by
`integrable_updateDirection_of_dominated` in the dominated regime and by `Integrable.of_finite` on
a finite batch. -/
theorem hasGradientAt_pullbackLandscape {μ : Measure Ω} {L : Ω → H → ℝ} {ψ : H → H} {h : Ω → H}
    {b : H} (hint : Integrable (updateDirection L ψ h b) μ)
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b) :
    HasGradientAt (pullbackLandscape L ψ h μ) (∫ ω, updateGradient L ψ h b ω ∂μ) b := by
  have hgrad : Integrable (updateGradient L ψ h b) μ := by
    refine (LinearIsometryEquiv.integrable_comp_iff (toDual ℝ H)).1 ?_
    simpa only [toDual_updateGradient] using hint
  rw [hasGradientAt_iff_hasFDerivAt, toDual_integral hgrad]
  simpa only [toDual_updateGradient] using hD

/-- Fact (i) as an equation between vectors: the gradient of the pullback landscape is the mean
update. -/
theorem gradient_pullbackLandscape_eq_meanUpdate {μ : Measure Ω} {L : Ω → H → ℝ} {ψ : H → H}
    {h : Ω → H} {b : H} (hint : Integrable (updateDirection L ψ h b) μ)
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b) :
    gradient (pullbackLandscape L ψ h μ) b = ∫ ω, updateGradient L ψ h b ω ∂μ :=
  (hasGradientAt_pullbackLandscape hint hD).gradient

/-- **Fact (i) on a finite batch, in the paper's own notation.** For a finite measure on a finite
sample space the vector equation `∇_b L̃(b) = E_X[G(b; X)]` holds with differentiability of the
positivity map and of each per-batch loss as the only hypotheses — no integrability, domination, or
measurability side condition. This is the reader-facing form of the identity for the paper's
actual mini-batches; on a general batch law the same equation is
`gradient_pullbackLandscape_eq_meanUpdate` fed by the dominated regime. -/
theorem gradient_pullbackLandscape_finiteBatch [Fintype Ω] [MeasurableSingletonClass Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] {L : Ω → H → ℝ} {ψ : H → H} {h : Ω → H} {b : H}
    (hψ : ∀ ω, DifferentiableAt ℝ ψ (preReadout h b ω))
    (hL : ∀ ω, DifferentiableAt ℝ (L ω) (ψ (preReadout h b ω))) :
    gradient (pullbackLandscape L ψ h μ) b = ∫ ω, updateGradient L ψ h b ω ∂μ :=
  gradient_pullbackLandscape_eq_meanUpdate Integrable.of_finite
    (hasFDerivAt_pullbackLandscape_finiteBatch μ hψ hL)

/-- **Fact (i) at the level of the paper's update rule.** The per-step rule of §2 is
`b_{t+1} = b_t - η G(b_t; X_t)` with the step evaluated on the batch drawn at that step. Averaging
that rule over the batch law gives a plain gradient step on the pullback landscape,
`E_X[b_{t+1}] = b_t - η ∇_b L̃(b_t)`: the coupling between latent weight and gradient inside `G` leaves
no trace in the mean iterate beyond `L̃`. The batch law is a probability measure here, as it is in
the paper's protocol; without that normalization the constant `b` would be reweighted by the total
mass. -/
theorem integral_slackStep {μ : Measure Ω} [IsProbabilityMeasure μ] {L : Ω → H → ℝ} {ψ : H → H}
    {h : Ω → H} {b : H} (η : ℝ) (hint : Integrable (updateDirection L ψ h b) μ)
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b) :
    (∫ ω, (b - η • updateGradient L ψ h b ω) ∂μ)
      = b - η • gradient (pullbackLandscape L ψ h μ) b := by
  have hgrad : Integrable (updateGradient L ψ h b) μ := by
    refine (LinearIsometryEquiv.integrable_comp_iff (toDual ℝ H)).1 ?_
    simpa only [toDual_updateGradient] using hint
  have hs : Integrable (fun ω => η • updateGradient L ψ h b ω) μ := hgrad.smul η
  rw [integral_sub (integrable_const b) hs, integral_const, integral_smul,
    gradient_pullbackLandscape_eq_meanUpdate hint hD]
  simp

/-- **The mean update descends the pullback landscape.** Along the step `b ↦ b - t E_X[G]` the
pullback landscape has derivative `-‖E_X[G]‖²` at `t = 0`. -/
theorem hasDerivAt_pullbackLandscape_along_meanUpdate {μ : Measure Ω} {L : Ω → H → ℝ} {ψ : H → H}
    {h : Ω → H} {b : H} (hint : Integrable (updateDirection L ψ h b) μ)
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b) :
    HasDerivAt (fun t : ℝ =>
        pullbackLandscape L ψ h μ (b - t • ∫ ω, updateGradient L ψ h b ω ∂μ))
      (-(‖∫ ω, updateGradient L ψ h b ω ∂μ‖ ^ 2)) 0 := by
  set g : H := ∫ ω, updateGradient L ψ h b ω ∂μ with hgdef
  have hgrad : HasGradientAt (pullbackLandscape L ψ h μ) g b :=
    hasGradientAt_pullbackLandscape hint hD
  have hval : (toDual ℝ H g) g = ‖g‖ ^ 2 := by
    simp [toDual_apply_apply]
  have := hasDerivAt_along_step (v := g) hgrad.hasFDerivAt
  rwa [hval] at this

/-- Unless the mean update vanishes — that is, unless `b` is already a critical point of the
pullback landscape — every small enough step along it strictly decreases the landscape. -/
theorem pullbackLandscape_lt_of_meanUpdate_ne_zero {μ : Measure Ω} {L : Ω → H → ℝ} {ψ : H → H}
    {h : Ω → H} {b : H} (hint : Integrable (updateDirection L ψ h b) μ)
    (hD : HasFDerivAt (pullbackLandscape L ψ h μ) (∫ ω, updateDirection L ψ h b ω ∂μ) b)
    (hne : (∫ ω, updateGradient L ψ h b ω ∂μ) ≠ 0) :
    ∀ᶠ t in 𝓝[>] (0 : ℝ), pullbackLandscape L ψ h μ (b - t • ∫ ω, updateGradient L ψ h b ω ∂μ)
      < pullbackLandscape L ψ h μ b := by
  have hpos : 0 < ‖∫ ω, updateGradient L ψ h b ω ∂μ‖ ^ 2 := by
    have : 0 < ‖∫ ω, updateGradient L ψ h b ω ∂μ‖ := norm_pos_iff.2 hne
    positivity
  exact eventually_lt_of_hasDerivAt_neg hpos
    (hasDerivAt_pullbackLandscape_along_meanUpdate hint hD)

end Hilbert

/-! ### The drift sees the latent weight only through the jitter law -/

section JitterLaw

variable {E F : Type*} [NormedAddCommGroup E] [MeasurableSpace E]

/-- **The convolution reading of the pullback landscape** (`docs/design/lemma1_rederivation.md`
§9). When the loss does not itself depend on the batch, the pullback landscape is the composite
`Λ ∘ ψ` averaged against the law of the latent weight jitter `h_φ(X)` — that is, `Λ ∘ ψ` smoothed by a
fixed kernel. With a batch-dependent loss this fails: the pullback is then a joint average over the
coupled pair and is not a convolution. -/
theorem pullbackLandscape_eq_integral_jitterLaw {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (Λ : F → ℝ) (ψ : E → F) (h : Ω → E) (b : E) (hh : AEMeasurable h μ)
    (hm : AEStronglyMeasurable (fun v => Λ (ψ (b + v))) (μ.map h)) :
    pullbackLandscape (fun _ => Λ) ψ h μ b = ∫ v, Λ (ψ (b + v)) ∂(μ.map h) := by
  rw [integral_map hh hm]
  rfl

/-- Two latent weight schemes with a common batch-independent loss whose jitter laws agree have the
same pullback landscape at any point where the smoothed integrand is measurable against that law.
This is a statement about the landscape and not yet about the drift: an identity holding at one
point constrains no derivative. `pullbackLandscape_eventuallyEq_of_jitterLaw_eq` and
`meanUpdate_eq_of_jitterLaw_eq` below carry it to the drift, at the price of the measurability
hypothesis on a whole neighbourhood rather than at the base point. With a batch-dependent loss the
joint law of latent weight and loss matters and this route is unavailable; what remains there is the
coupling-agnostic `meanUpdate_eq_of_pullbackLandscape_eventuallyEq`. -/
theorem pullbackLandscape_eq_of_jitterLaw_eq {Ω₁ Ω₂ : Type*} [MeasurableSpace Ω₁]
    [MeasurableSpace Ω₂] {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} (Λ : F → ℝ) (ψ : E → F) (h₁ : Ω₁ → E)
    (h₂ : Ω₂ → E) (b : E)
    (hh₁ : AEMeasurable h₁ μ₁) (hh₂ : AEMeasurable h₂ μ₂) (hlaw : μ₁.map h₁ = μ₂.map h₂)
    (hm : AEStronglyMeasurable (fun v => Λ (ψ (b + v))) (μ₁.map h₁)) :
    pullbackLandscape (fun _ => Λ) ψ h₁ μ₁ b = pullbackLandscape (fun _ => Λ) ψ h₂ μ₂ b := by
  rw [pullbackLandscape_eq_integral_jitterLaw Λ ψ h₁ b hh₁ hm,
    pullbackLandscape_eq_integral_jitterLaw Λ ψ h₂ b hh₂ (hlaw ▸ hm), hlaw]

/-- The landscape identity of `pullbackLandscape_eq_of_jitterLaw_eq` on a whole neighbourhood of
`b`, which is what a statement about the derivative needs. The only strengthening over that lemma
is that the measurability hypothesis is asked for at every point near `b` rather than at `b`
alone; it is stated as an `∀ᶠ` so that the neighbourhood on which it holds is the one the
conclusion uses. -/
theorem pullbackLandscape_eventuallyEq_of_jitterLaw_eq {Ω₁ Ω₂ : Type*} [MeasurableSpace Ω₁]
    [MeasurableSpace Ω₂] {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} (Λ : F → ℝ) (ψ : E → F) (h₁ : Ω₁ → E)
    (h₂ : Ω₂ → E) (b : E)
    (hh₁ : AEMeasurable h₁ μ₁) (hh₂ : AEMeasurable h₂ μ₂) (hlaw : μ₁.map h₁ = μ₂.map h₂)
    (hm : ∀ᶠ b' in 𝓝 b, AEStronglyMeasurable (fun v => Λ (ψ (b' + v))) (μ₁.map h₁)) :
    pullbackLandscape (fun _ => Λ) ψ h₁ μ₁ =ᶠ[𝓝 b] pullbackLandscape (fun _ => Λ) ψ h₂ μ₂ := by
  filter_upwards [hm] with b' hb'
  exact pullbackLandscape_eq_of_jitterLaw_eq Λ ψ h₁ h₂ b' hh₁ hh₂ hlaw hb'

/-- **For a batch-independent loss the drift sees the latent weight only through the jitter law.** Two
latent weight schemes on unrelated sample spaces, with unrelated batch laws and bodies, that share the
loss and the positivity map and induce the same law for the latent weight jitter `h_φ(X)` have literally the
same mean update at `b`. This is the drift statement — the one §9 of the design note reads as the
convolution picture — and it is a strict consequence of the landscape statement rather than a
restatement of it, since it needs the landscapes to agree near `b` and each scheme to satisfy fact
(i) there. With a batch-dependent loss the pullback is a joint average over the coupled pair, is
not a convolution, and no statement of this shape is available. -/
theorem meanUpdate_eq_of_jitterLaw_eq [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    {Ω₁ Ω₂ : Type*} [MeasurableSpace Ω₁] [MeasurableSpace Ω₂] {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂}
    {Λ : F → ℝ} {ψ : E → F} {h₁ : Ω₁ → E} {h₂ : Ω₂ → E} {b : E}
    (hh₁ : AEMeasurable h₁ μ₁) (hh₂ : AEMeasurable h₂ μ₂) (hlaw : μ₁.map h₁ = μ₂.map h₂)
    (hm : ∀ᶠ b' in 𝓝 b, AEStronglyMeasurable (fun v => Λ (ψ (b' + v))) (μ₁.map h₁))
    (hD₁ : HasFDerivAt (pullbackLandscape (fun _ => Λ) ψ h₁ μ₁)
      (∫ ω, updateDirection (fun _ => Λ) ψ h₁ b ω ∂μ₁) b)
    (hD₂ : HasFDerivAt (pullbackLandscape (fun _ => Λ) ψ h₂ μ₂)
      (∫ ω, updateDirection (fun _ => Λ) ψ h₂ b ω ∂μ₂) b) :
    (∫ ω, updateDirection (fun _ => Λ) ψ h₁ b ω ∂μ₁)
      = ∫ ω, updateDirection (fun _ => Λ) ψ h₂ b ω ∂μ₂ :=
  meanUpdate_eq_of_pullbackLandscape_eventuallyEq hD₁ hD₂
    (pullbackLandscape_eventuallyEq_of_jitterLaw_eq Λ ψ h₁ h₂ b hh₁ hh₂ hlaw hm)

end JitterLaw

section ConstantBody

variable {E F : Type*} [NormedAddCommGroup E]

/-- A body that does not depend on the batch — the direct-softplus baseline of the paper is the
case `c = 0`, and `rem:moreau-scope`'s frozen-anchor regime is the general case — leaves the
landscape unsmoothed: the pullback landscape is then the plain batch-averaged loss evaluated at the
one fixed shifted point, with no averaging over latent weight jitter at all. -/
theorem pullbackLandscape_of_body_const {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (L : Ω → F → ℝ) (ψ : E → F) {h : Ω → E} {c : E} (hc : ∀ ω, h ω = c) (b : E) :
    pullbackLandscape L ψ h μ b = ∫ ω, L ω (ψ (b + c)) ∂μ := by
  simp only [pullbackLandscape, preReadout, hc]

end ConstantBody

/-! ### Non-vacuity

Concrete instances certifying that the hypotheses above are satisfiable and the definitions
non-degenerate. Without these, a hostile reading could ask whether any `(L, ψ, h, μ)` at all
discharges the differentiability or domination packages; with them, the question is settled by
the kernel. The specific functions are immaterial — these are satisfiability certificates for
the analytic hypotheses, not models of the paper's training setup. -/

section NonVacuity

/-- **The update direction is the paper's formula, and it is not degenerate.** In the scalar
linear model `L(θ; X) = c·θ`, `ψ(θ̃) = s·θ̃`, the update direction evaluates in closed form to
`(s·c) • id` — literally the positivity-map slope `ψ' = s` times the loss gradient `∇_θL = c` — and is
nonzero whenever `s·c ≠ 0`. In particular the `fderiv`-based definition produces the honest
composite, not some junk value. -/
theorem updateDirection_linear_model (c s : ℝ) {Ω : Type*} (h : Ω → ℝ) (b : ℝ) (ω : Ω) :
    updateDirection (fun _ x => c * x) (fun x => s * x) h b ω
      = (s * c) • ContinuousLinearMap.id ℝ ℝ := by
  have hlin : ∀ (a y : ℝ), fderiv ℝ (fun x : ℝ => a * x) y
      = a • ContinuousLinearMap.id ℝ ℝ := by
    intro a y
    rw [fderiv_const_mul differentiableAt_fun_id a, fderiv_fun_id]
  simp only [updateDirection, hlin]
  simp [smul_smul]

/-- **Fact (i) fires on a concrete finite batch.** The two-point batch `Ω = Fin 2` with counting
law, the batch-dependent quadratic loss `L(θ; ω) = (θ − ω)²` and the nonlinear positivity map
`ψ(θ̃) = θ̃² + θ̃` satisfy every hypothesis of `hasFDerivAt_pullbackLandscape_finiteBatch`; the
loss depends on the same `ω` that drives the latent weight, so the certified instance carries the
batch coupling of `rem:bcond`, not a decoupled surrogate. -/
theorem hasFDerivAt_pullbackLandscape_twoPointBatch (b : ℝ) :
    HasFDerivAt
      (pullbackLandscape (fun (ω : Fin 2) (x : ℝ) => (x - (ω : ℝ)) ^ 2)
        (fun x : ℝ => x ^ 2 + x) (fun ω : Fin 2 => (ω : ℝ)) Measure.count)
      (∫ ω, updateDirection (fun (ω : Fin 2) (x : ℝ) => (x - (ω : ℝ)) ^ 2)
        (fun x : ℝ => x ^ 2 + x) (fun ω : Fin 2 => (ω : ℝ)) b ω ∂Measure.count) b :=
  hasFDerivAt_pullbackLandscape_finiteBatch Measure.count
    (fun _ => by fun_prop) (fun _ => by fun_prop)

/-- **The vector form of fact (i) is reached end to end on the same concrete instance.** On the
two-point batch above, the hypotheses of the Hilbert-space bridge — integrability of the update
direction and the functional form of the identity — are both dischargeable, so
`∇_b L̃(b) = E_X[G(b; X)]` holds as an equation between vectors on an actual batch law. This
certifies that the gradient-form theorems are not vacuously quantified: their instance
requirements (a complete real inner-product space, here `ℝ`) and hypothesis package are
satisfiable simultaneously. -/
theorem gradient_pullbackLandscape_twoPointBatch (b : ℝ) :
    gradient (pullbackLandscape (fun (ω : Fin 2) (x : ℝ) => (x - (ω : ℝ)) ^ 2)
      (fun x : ℝ => x ^ 2 + x) (fun ω : Fin 2 => (ω : ℝ)) Measure.count) b
    = ∫ ω, updateGradient (fun (ω : Fin 2) (x : ℝ) => (x - (ω : ℝ)) ^ 2)
        (fun x : ℝ => x ^ 2 + x) (fun ω : Fin 2 => (ω : ℝ)) b ω ∂Measure.count :=
  gradient_pullbackLandscape_finiteBatch Measure.count
    (fun _ => by fun_prop) (fun _ => by fun_prop)

/-- **The dominated package is satisfiable on a genuinely infinite batch space**, where the
finite-batch theorem does not apply: batch space `ℝ` under the uniform law on `[0, 1]`, body
`h = id` (the latent weight jitter is the batch draw itself), positivity map `ψ = sin`, per-batch loss
`L = sin`, dominating bound the constant `1`. Every one of the seven Leibniz hypotheses of
`hasFDerivAt_pullbackLandscape_of_dominated` is discharged, so those hypotheses are honest
analytic inputs, not an unsatisfiable escape hatch. -/
theorem hasFDerivAt_pullbackLandscape_uniformBatch (b : ℝ) :
    HasFDerivAt
      (pullbackLandscape (fun _ : ℝ => Real.sin) Real.sin id
        (volume.restrict (Set.Icc (0:ℝ) 1)))
      (∫ ω, updateDirection (fun _ : ℝ => Real.sin) Real.sin id b ω
        ∂(volume.restrict (Set.Icc (0:ℝ) 1))) b := by
  have hsin : ∀ y : ℝ, fderiv ℝ Real.sin y = Real.cos y • ContinuousLinearMap.id ℝ ℝ := by
    intro y
    have h1 : HasFDerivAt Real.sin
        ((1 : ℝ →L[ℝ] ℝ).smulRight (Real.cos y)) y :=
      hasDerivAt_iff_hasFDerivAt.1 (Real.hasDerivAt_sin y)
    have h2 : (1 : ℝ →L[ℝ] ℝ).smulRight (Real.cos y)
        = Real.cos y • ContinuousLinearMap.id ℝ ℝ := by
      ext; simp [mul_comm]
    rw [← h2]; exact h1.fderiv
  have hUD : ∀ b' ω : ℝ, updateDirection (fun _ : ℝ => Real.sin) Real.sin id b' ω
      = (Real.cos (b' + ω) * Real.cos (Real.sin (b' + ω))) • ContinuousLinearMap.id ℝ ℝ := by
    intro b' ω
    simp only [updateDirection, preReadout, id_eq, hsin]
    simp [smul_smul]
  have hUDnorm : ∀ b' ω : ℝ,
      ‖updateDirection (fun _ : ℝ => Real.sin) Real.sin id b' ω‖ ≤ 1 := by
    intro b' ω
    rw [hUD, norm_smul, ContinuousLinearMap.norm_id, mul_one, Real.norm_eq_abs, abs_mul]
    exact mul_le_one₀ (Real.abs_cos_le_one _) (abs_nonneg _) (Real.abs_cos_le_one _)
  refine hasFDerivAt_pullbackLandscape_of_dominated (s := Set.univ) (bound := fun _ => 1)
    Filter.univ_mem ?_ ?_ ?_ ?_ ?_ ?_
  · exact Eventually.of_forall fun b' =>
      ((by fun_prop : Continuous fun ω : ℝ => Real.sin (Real.sin (b' + ω))).aestronglyMeasurable)
  · exact (by fun_prop : Continuous fun ω : ℝ => Real.sin (Real.sin (b + ω))).integrableOn_Icc
  · have hcont : Continuous fun ω : ℝ =>
        (Real.cos (b + ω) * Real.cos (Real.sin (b + ω))) • ContinuousLinearMap.id ℝ ℝ :=
      (by fun_prop : Continuous fun ω : ℝ =>
        Real.cos (b + ω) * Real.cos (Real.sin (b + ω))).smul continuous_const
    exact hcont.aestronglyMeasurable.congr
      (Eventually.of_forall fun ω => (hUD b ω).symm)
  · exact Eventually.of_forall fun ω b' _ => hUDnorm b' ω
  · exact integrableOn_const (by simp)
  · exact Eventually.of_forall fun ω b' _ =>
      ⟨Real.differentiable_sin.differentiableAt, Real.differentiable_sin.differentiableAt⟩

/-- **Two genuinely different latent weight schemes with the same pullback landscape.** The uncoupled
scheme is a one-point batch with no latent weight jitter at all (`Ω = Fin 1`, `h ≡ 0`) carrying the loss
`2x²`; the coupled scheme is a two-point batch whose body outputs `±1` and whose per-batch loss
`(x − s·(±1))²` is centred on the very same draw, so that latent weight and gradient ride one batch in
the sense of `rem:bcond`. Both use the positivity map `ψ(x) = s·x`, whose derivative is the prefactor `s`.
Their pullback landscapes are equal at every `b`, both being `2 s² b²`: turning the coupled
latent weight on has deformed no potential. -/
theorem pullbackLandscape_coupledTwoPoint_eq_uncoupled (s b : ℝ) :
    pullbackLandscape (fun _ : Fin 1 => fun x : ℝ => 2 * x ^ 2) (fun x : ℝ => s * x)
        (fun _ : Fin 1 => (0 : ℝ)) Measure.count b
      = pullbackLandscape (fun (ω : Fin 2) (x : ℝ) => (x - s * (1 - 2 * (ω : ℝ))) ^ 2)
        (fun x : ℝ => s * x) (fun ω : Fin 2 => 1 - 2 * (ω : ℝ)) Measure.count b := by
  rw [pullbackLandscape_eq_weightedSum, pullbackLandscape_eq_weightedSum]
  simp [measureReal_def]
  ring

/-- **The corollary that carries the note's conclusion, on an actual pair of schemes.** The two
schemes above — different sample spaces, different bodies, one of them with the batch coupling of
`rem:bcond` live and the other with no batch fluctuation whatever — have literally the same mean
update at every `b`. This is `meanUpdate_eq_of_pullbackLandscape_eventuallyEq` instantiated, so its
hypotheses are certified satisfiable by schemes that genuinely differ, not only by a scheme
compared with itself. -/
theorem meanUpdate_coupledTwoPoint_eq_uncoupled (s b : ℝ) :
    (∫ ω, updateDirection (fun _ : Fin 1 => fun x : ℝ => 2 * x ^ 2) (fun x : ℝ => s * x)
        (fun _ : Fin 1 => (0 : ℝ)) b ω ∂Measure.count)
      = ∫ ω, updateDirection (fun (ω : Fin 2) (x : ℝ) => (x - s * (1 - 2 * (ω : ℝ))) ^ 2)
        (fun x : ℝ => s * x) (fun ω : Fin 2 => 1 - 2 * (ω : ℝ)) b ω ∂Measure.count :=
  meanUpdate_eq_of_pullbackLandscape_eventuallyEq
    (hasFDerivAt_pullbackLandscape_finiteBatch Measure.count (fun _ => by fun_prop)
      (fun _ => by fun_prop))
    (hasFDerivAt_pullbackLandscape_finiteBatch Measure.count (fun _ => by fun_prop)
      (fun _ => by fun_prop))
    (Filter.Eventually.of_forall (pullbackLandscape_coupledTwoPoint_eq_uncoupled s))

/-- The common drift of the two schemes, in closed form: `(4 s² b) • id`, which is the derivative
of the shared landscape `2 s² b²` and carries the positivity-map prefactor squared, once from the chain
rule and once from the positivity map's own argument. -/
theorem meanUpdate_coupledTwoPoint (s b : ℝ) :
    (∫ ω, updateDirection (fun (ω : Fin 2) (x : ℝ) => (x - s * (1 - 2 * (ω : ℝ))) ^ 2)
        (fun x : ℝ => s * x) (fun ω : Fin 2 => 1 - 2 * (ω : ℝ)) b ω ∂Measure.count)
      = (4 * s ^ 2 * b) • ContinuousLinearMap.id ℝ ℝ := by
  rw [← meanUpdate_coupledTwoPoint_eq_uncoupled s b]
  have hlin : fderiv ℝ (fun x : ℝ => s * x) b = s • ContinuousLinearMap.id ℝ ℝ := by
    rw [fderiv_const_mul differentiableAt_fun_id s, fderiv_fun_id]
  have hq : fderiv ℝ (fun x : ℝ => 2 * x ^ 2) (s * b)
      = (4 * (s * b)) • ContinuousLinearMap.id ℝ ℝ := by
    have h : HasDerivAt (fun x : ℝ => x ^ 2) (2 * (s * b)) (s * b) := by
      simpa using hasDerivAt_pow 2 (s * b)
    have h1 : HasDerivAt (fun x : ℝ => 2 * x ^ 2) (4 * (s * b)) (s * b) := by
      have h2 := h.const_mul (2 : ℝ)
      rwa [show (2 : ℝ) * (2 * (s * b)) = 4 * (s * b) by ring] at h2
    have h3 : HasFDerivAt (fun x : ℝ => 2 * x ^ 2)
        ((1 : ℝ →L[ℝ] ℝ).smulRight (4 * (s * b))) (s * b) := hasDerivAt_iff_hasFDerivAt.1 h1
    have h4 : (1 : ℝ →L[ℝ] ℝ).smulRight (4 * (s * b))
        = (4 * (s * b)) • ContinuousLinearMap.id ℝ ℝ := by ext; simp [mul_comm]
    rw [← h4]; exact h3.fderiv
  rw [integral_fintype Integrable.of_finite]
  simp only [updateDirection, preReadout, add_zero, hlin, hq, measureReal_def]
  ext
  simp [mul_comm, mul_left_comm, mul_assoc]
  ring

/-- The common drift is nonzero away from the shared minimum, for any nonzero positivity-map slope. The
instance of `meanUpdate_eq_of_pullbackLandscape_eventuallyEq` above is therefore an equality of two
nonzero mean updates and not the vacuous `0 = 0`. -/
theorem meanUpdate_coupledTwoPoint_ne_zero {s b : ℝ} (hs : s ≠ 0) (hb : b ≠ 0) :
    (∫ ω, updateDirection (fun (ω : Fin 2) (x : ℝ) => (x - s * (1 - 2 * (ω : ℝ))) ^ 2)
        (fun x : ℝ => s * x) (fun ω : Fin 2 => 1 - 2 * (ω : ℝ)) b ω ∂Measure.count) ≠ 0 := by
  rw [meanUpdate_coupledTwoPoint s b]
  intro hzero
  have h1 : ((4 * s ^ 2 * b) • ContinuousLinearMap.id ℝ ℝ) 1 = 0 := by rw [hzero]; simp
  simp at h1
  rcases h1 with h | h
  · exact hs h
  · exact hb h

end NonVacuity

end IcnnLift
