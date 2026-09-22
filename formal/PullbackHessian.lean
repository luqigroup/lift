import Mathlib
import TraceLemmas
import CrossCov

/-!
# The exact pullback Hessian, and the refutation of the shipped Lemma 1 decomposition

The shipped proof of `lem:moreau` (`docs/paper/v4/A1_proofs.tex`, the paragraph beginning
"The Hessian `H⋆`, a symmetric matrix, decomposes …") asserts, without derivation, that the
Hessian of the pullback landscape splits into an expected-loss curvature plus a
cross-covariance contribution,
`H⋆ = E_X[∇²_θ̃ L] + κ⁻² · ½ (Σ_slack + Σ_slackᵀ)`.
The authors' design note (`docs/design/lemma1_rederivation.md` §1) concedes that this assertion
is false, and this file is the machine-checked form of that concession. The pullback landscape
is `L̃(b) = E_X[(L ∘ ψ)(b + h_φ(X))]`; the batch variable enters only through the *translation*
`b ↦ b + h_φ(X)`, so differentiation in `b` commutes with the batch average and the second
derivative of `L̃` is the batch average of the second derivatives of the per-batch composite
losses — **exactly, with no cross-covariance term of any size**. The cross-covariance is a
property of the update *dynamics* (the covariance of the stochastic gradient), not of the
curvature of the averaged landscape, and the corrected route of the design note (§2, §3) puts
it there instead.

Three forms of the exact identity are proved. The first is the finite-batch case, where the
batch law is a finite family of weights over a `Fintype`: there the identity is
*unconditional* — no domination, no integrability, no smallness — and needs only that each
per-batch composite loss carries a first derivative at every point and a second derivative at
the shifted point (the first-order hypothesis is global, because identifying the derivative
*function* of the batch average requires a derivative wherever that function is evaluated; the
second-order hypothesis is needed only at the point itself). It is proved in the
generality of an arbitrary finite batch average `x ↦ ∑ᵢ wᵢ Fᵢ(x)`, so it does not depend on the
translation structure `θ̃ = b + h_φ(X)`; in particular it holds verbatim for differentiation in
the full body parameter `φ`, where it says that `∇²_φ L̃` is the batch average of the per-sample
`φ`-curvatures. One limit of that generality has to be stated plainly, because the batch-average
results are quoted below against the `φ` reading of the shipped equation. Under that reading the
identity does not by itself identify its own right-hand side with the shipped equation's first
term `E_X[∇²_θ̃ L]`, which is a curvature in `θ̃` and not in `φ`: the two are separated by the
chain rule through the body Jacobian `∂θ̃/∂φ`, a deterministic term that this file does not
compute. What the batch-average form settles under the `φ` reading is the point at issue — that
the averaging itself contributes nothing beyond the average, so no cross-covariance of any size
arises from it — and not the size of that chain-rule term. The reading on which the identity
confronts the shipped equation term for term is therefore the slack reading `∇²_b`, and that is
the paper's own reading as well: `lem:moreau` states its trace inequality "on the slack
subspace", and its assumption (A3) sets `∂θ̃/∂b = I_d`, so on the slack block the chain rule is
the identity and the first terms of the two sides are literally the same object.

The second form is the same identity at the level of the `d × d` Hessian matrix in the slack
block, obtained from the second Fréchet derivative; this is the form that confronts the shipped
equation directly, because both sides are then matrices of the same size and the equation can be
compared entry by entry. The third is the general-measure case, where the batch law is an
arbitrary measure and the identity holds under explicitly named domination hypotheses; those
hypotheses are exactly what "enough regularity to differentiate under the integral sign twice"
means, and they are supplied to mathlib's `hasDerivAt_integral_of_dominated_loc_of_deriv_le`. They
are honest hypotheses: without domination the interchange can genuinely fail, and no part of the
*conclusion* is assumed in them.

The refutation is then stated as a theorem rather than as a remark. If the shipped
decomposition held alongside the exact identity, the added term would have to vanish — and it
vanishes as a matrix, not only in trace: the symmetric part `½(Σ_slack + Σ_slackᵀ)` is forced to
be the zero matrix, so no quadratic form is added in any direction. That closes the anisotropy
hedge of the shipped text ("the added curvature is in general anisotropic … and `eq:moreau`
reports its trace"), and `tr Σ_slack = 0` is the special case the paper measures. A
remainder-carrying version is also proved: if the shipped equation is
weakened to hold only up to a remainder `R` — which is how the surrounding lemma statement
hedges, with an `O(η σ_Jac²/(d κ²))` Itô–Taylor correction — then `tr Σ_slack = -κ² tr R`
exactly, so a bound `|tr R| ≤ δ` forces `|tr Σ_slack| ≤ κ² δ`. The cross-covariance
contribution is therefore not "added curvature up to higher-order terms"; it is exactly as
large as the term the paper discards. The displayed conclusion of the lemma fares no better
than its proof: the stated trace inequality `tr H⋆ ≥ tr E_X[∇²_θ̃ L] + d μ_eff` fails outright
for the genuine pullback Hessian whenever `σ_Jac² > 0` — and it fails for *every* positive
coefficient on `σ_Jac²/κ²`, not only for the proof's own extraction `d μ_eff = σ_Jac²/κ²`, so
the `Θ(σ_Jac²/(d κ²))` hedge on `μ_eff` and the subleading Itô–Taylor correction (which can
only shift that coefficient) do not rescue it, since the exact identity makes the two
traces equal. Finally a concrete witness is exhibited: the rank-one
outer product `δθ̃ δgᵀ` with `δθ̃ = δg = e_j`, of trace `1 > 0`. It is realizable by the
measurement protocol itself: on a window of length two whose values of the latent weight move by `2 eⱼ` between
the draws, the window-centred fluctuations of `eq:fluct` — computed by `bodyFluct` of
`CrossCov.lean`, derived rather than assumed — come out to `∓ eⱼ`, and the trailing-window
estimator of `eq:cross-cov` returns exactly this witness. A window of length one could not do
it: the centring subtracts the window mean, which over a single sample is the sample itself, so
the protocol's estimator at window length one is identically zero — a fact proved here — and
the length-one equation proved alongside it is an algebraic statement about the estimator
function, not a run of the protocol. Positive trace is the regime the paper measures — the
design note records `tr Σ_slack > 0` at 8997 of 9000 iterations — so the shipped decomposition
is incompatible with the measurement it is invoked to explain, for every positivity-map scale `κ ≠ 0`.
Nothing in this turns on the `1/κ²` factor, which the design note separately flags as asserted
with no derivation: the refutation is proved with that factor replaced by an arbitrary nonzero
scalar, so no rederivation of the unit conversion can repair the equation.

Non-vacuity is machine-checked on both sides of the refutation, because a negation is worth
nothing if either side of it is empty. On the side of the hypotheses: a closed instance with the
concrete quadratic per-batch loss `x ↦ (x j)²` discharges every differentiability hypothesis of
both headline refutations, and — for weights summing to one, the case the paper is about — the
pullback Hessian it produces is computed outright to be the nonzero matrix `2 E_jj` of trace two,
so the instance is not a degenerate one in which both sides collapse to zero; and two
quadratic-integrand instances under an arbitrary probability measure discharge the entire
domination package of the general-measure identity — once with a constant body, and once with the
genuinely batch-dependent body `h(x) = sin x`, so that the per-batch shifted balls of the
domination hypotheses are exercised as actually distinct balls rather than as a single common one.
On the side of the refuted statement: the shipped equation is a satisfiable constraint, and the
refutation is sharp. Given the exact identity it holds *exactly* when the symmetric part of
`Σ_slack` vanishes, and a nonzero antisymmetric cross-covariance that satisfies it is exhibited.
So what the negative results rule out is the measured positivity of `tr Σ_slack`, and not the
shape of an equation that nothing could have satisfied.

The shipped equation carries a further defect, independent of the two just discussed, which the
design note also flags and which this file makes visible in the types: `H⋆ ≡ ∇²_φ L̃(φ⋆)` is a `p
× p` object, where `p` is the dimension of the body parameter vector `φ`, while `E_X[∇²_θ̃ L]` and
`Σ_slack` are `d × d` objects in the slack dimension `d`. The sum cannot be formed at all without
an identification of the two index sets. `HessianCrossCovDecompositionAcrossBlocks` carries that
identification as an explicit hypothesis `hpd : p = d` and reindexes through it, so that the
assumption the paper leaves silent is a visible, undischarged hypothesis in the Lean statement.
Under the alternative reading in which `H⋆` denotes only the `d × d` slack block, the equation is
well formed, and it is then refuted by the exact identity: both readings are covered.

A reading convention, stated because it is load-bearing. Throughout, `E_X[∇²_θ̃ L]` is read as
the batch average of the second derivative of the **composite** objective `L ∘ ψ` in the
latent weight `θ̃`. That is the reading of the design note (§1, which writes the exact
identity as `∇²_b L̃ = E_X[∇²_θ (L ∘ ψ)]`), it is the reading under which the two sides of the
shipped equation are curvatures of the same objective, and it is the most favourable reading for
the shipped equation, since it is the one on which the equation's first term is exactly right and
only the added term is in question. Under the alternative reading, in which `∇²_θ̃ L` denotes the
Hessian of `L` in its own argument before the positivity map, the shipped equation carries a further
defect — a missing chain rule through `ψ`, which contributes a `ψ'`-squared factor and a `ψ''`
term — and that defect is not analysed here; the chain rule through the positivity map is the business
of `ShoulderAttenuation.lean`. The refutation below is therefore stated against the strongest
form of the claim.

What this file does **not** do. It does not formalize the invariant measure of the Itô process,
the small-`η` ergodic expansion, or anything else from the first two steps of the shipped proof;
this mathlib has no Itô calculus, no stochastic differential equations and no Fokker–Planck
equation. Nothing here depends on those steps: the refutation is a statement about the third
step alone, the Hessian decomposition, and that step is a claim of ordinary multivariable
calculus which is settled here outright. It also does not prove that the corrected route of the
design note is correct; that is the business of `UpdateDrift.lean`, `UpdateCovariance.lean` and
the modules downstream of them. What it establishes is the negative result those modules are
built to replace. One scope limit is worth naming: the general-measure form of the exact identity
is proved in one real slack variable, because mathlib's differentiation-under-the-integral-sign
API in `Mathlib/Analysis/Calculus/ParametricIntegral.lean` is what supplies it and the second-order
step is cleanest there; the matrix-valued form of the exact identity, which is the one that
confronts the shipped equation, is proved for finite batch laws, where it needs no domination at
all. Reaching the matrix form for a general batch law would require iterating
`hasFDerivAt_integral_of_dominated_loc_of_lip` for a Bochner integral of operator-valued second
derivatives; it would add hypotheses without adding content, since the finite-batch form already
carries the refutation unconditionally.

## Results
* `pullbackLoss` — the pullback landscape `b ↦ ∑ᵢ wᵢ (L ∘ ψ)(b + hᵢ)` of a finite batch law.
* `iteratedDeriv_two_weightedSum` — the second derivative of a finite weighted sum is the
  weighted sum of the second derivatives.
* `iteratedDeriv_two_pullbackLoss` — the exact identity in one variable: `L̃''(b) = ∑ᵢ wᵢ fᵢ''(b + hᵢ)`.
* `pullbackLoss_no_extra_term` — the scalar refutation: any additive extra term is zero.
* `hasFDerivAt_fderiv_batchAverage` — differentiation commutes with a finite batch average,
  twice, with no assumption on how the batch enters the per-sample loss.
* `hessianMatrix_batchAverage` — the same as an equality of `d × d` matrices.
* `hasFDerivAt_fderiv_pullbackLoss` — the exact identity as a second Fréchet derivative.
* `hessianMatrix_pullbackLoss` — the exact identity as an equality of `d × d` matrices.
* `hessianMatrix_pullbackLoss_of_const` — a batch law averages: constant per-batch Hessian gives
  that Hessian back, which is where `∑ᵢ wᵢ = 1` is used.
* `trace_hessianMatrix_pullbackLoss` — the trace form of the exact identity.
* `iteratedDeriv_two_pullbackLossIntegral` — the exact identity for a general batch law, under
  named domination hypotheses.
* `iteratedDeriv_two_pullbackLossIntegral_eq_integral` — the same, with the integrand written as
  the second derivative of the per-batch composite loss.
* `iteratedDeriv_two_pullbackLossIntegral_sq` — the domination package is satisfiable: the
  general-measure identity instantiated end to end for a quadratic integrand under an arbitrary
  probability measure, returning `L̃'' = 2`.
* `iteratedDeriv_two_pullbackLossIntegral_sq_sin` — the same with the genuinely batch-dependent
  body `h(x) = sin x`, so the per-batch shifted balls of the domination package are actually
  distinct.
* `HessianCrossCovDecomposition` — the shipped equation, as a predicate.
* `trace_eq_zero_of_hessianCrossCovDecomposition` — the shipped equation plus the exact identity
  forces `tr Σ = 0`.
* `symmPart_eq_zero_of_hessianCrossCovDecomposition` — stronger: the added term vanishes as a
  matrix, so the anisotropic version of the shipped claim fails as well.
* `symmPart_eq_zero_of_scaledCrossCovDecomposition`,
  `trace_eq_zero_of_scaledCrossCovDecomposition` — the same with the shipped equation's
  underived `1/κ²` unit conversion replaced by an arbitrary nonzero scalar, so that no
  rederivation of the conversion factor can rescue the equation.
* `hessianCrossCovDecomposition_iff_symmPart_eq_zero` — the refutation is sharp: given the exact
  identity, the shipped equation holds exactly when the symmetric part of `Σ` vanishes.
* `antisymmCrossCov`, `antisymmCrossCov_ne_zero`, `trace_antisymmCrossCov`,
  `hessianCrossCovDecomposition_antisymmCrossCov` — a nonzero cross-covariance that *does*
  satisfy the shipped equation, so the negations below are not vacuously true of every `Σ`.
* `trace_of_hessianCrossCovDecomposition_remainder` — with an explicit remainder `R`,
  `tr Σ = -κ² tr R`.
* `abs_trace_le_of_hessianCrossCovDecomposition_remainder` — hence `|tr Σ| ≤ κ² δ` when
  `|tr R| ≤ δ`.
* `HessianCrossCovDecompositionAcrossBlocks` and
  `trace_eq_zero_of_hessianCrossCovDecompositionAcrossBlocks` — the same, with the `p = d`
  identification the shipped equation needs made explicit.
* `rankOneCrossCov`, `trace_rankOneCrossCov`, `rankOneCrossCov_eq_crossCovEstimator` — the
  rank-one witness of trace one, and its algebraic evaluation by the estimator function.
* `crossCovEstimator_one_bodyFluct_eq_zero` — a centred window of length one is degenerate: the
  measurement protocol's estimator at `T = 1` is identically the zero matrix.
* `rankOneCrossCov_realizable` — the witness is produced by the estimator on a genuinely
  window-centred run of length two, with the centring derived by `bodyFluct`.
* `not_hessianCrossCovDecomposition_of_trace_pos`, `not_hessianCrossCovDecomposition_rankOne` —
  no positive-trace cross-covariance satisfies the shipped equation, and in particular the
  rank-one witness does not.
* `pullbackHessian_not_hessianCrossCovDecomposition` — the refutation applied to the genuine
  pullback Hessian of a genuine pullback landscape.
* `pullbackHessian_not_scaledCrossCovDecomposition` — the same for an arbitrary nonzero unit
  conversion in place of `1/κ²`.
* `pullbackHessian_not_trace_lower_bound` — the displayed trace inequality of `lem:moreau`
  fails for the genuine pullback Hessian whenever `tr Σ > 0`, for **every** positive coefficient
  on `σ_Jac²/κ²`, so the `Θ(·)` hedge on `μ_eff` does not rescue it.
* `hessianMatrix_sq`, `hessianMatrix_pullbackLoss_sq`, `trace_hessianMatrix_pullbackLoss_sq` —
  the Hessian of the concrete quadratic `x ↦ (x j)²`, and the pullback Hessian it produces,
  computed outright: the nonzero matrix `2 E_jj`, of trace two.
* `quadraticPullback_not_hessianCrossCovDecomposition`,
  `quadraticPullback_not_trace_lower_bound` — closed instances of the two headline refutations,
  with every hypothesis discharged.
* `batchAverageHessian_not_hessianCrossCovDecomposition` — the refutation for an arbitrary finite
  batch average, with no translation structure assumed.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses carried, and why each is a hypothesis rather than a theorem:

* Twice-differentiability of the per-batch composite loss (`hf1`, `hf2`, in `HasDerivAt` and
  `HasFDerivAt` form). To be precise about the quantifiers: `hf1` supplies a first derivative at
  *every* point — global differentiability, with `f1` the derivative function — because the
  derivative function of the batch average must be identified wherever it is differentiated
  again; `hf2` is needed only at the shifted base point itself. This is a regularity assumption
  about `L ∘ ψ`, not about the conclusion; it cannot be discharged because `L` and `ψ` are
  arbitrary here (the paper's forward-KL loss composed with softplus is smooth everywhere, so
  the global form costs nothing in the intended model, and
  `quadraticPullback_not_hessianCrossCovDecomposition` discharges it outright for a concrete
  loss). In the finite-batch results this is the *only* hypothesis, which is why the
  finite-batch statement is the cleanest form of the refutation.
* The domination package of the general-measure results (`hf_meas`, `hf_int`, `hf1_meas`,
  `hf1_deriv`, `hf1_bound`, `hbound1`, `hf2_meas`, `hf2_deriv`, `hf2_bound`, `hbound2`, all on a
  ball of radius `ε` around the base point). These are the hypotheses of mathlib's
  `hasDerivAt_integral_of_dominated_loc_of_deriv_le`, applied twice. They would be discharged in
  any concrete model by uniform bounds on the first two derivatives of `L ∘ ψ` over the operative
  region together with integrability of those bounds against the batch law; they are not
  discharged here because no concrete `L`, `ψ` or batch law is fixed. Note that the finite-batch
  results need none of them, so the refutation does not rest on them.
* `hpd : p = d` in the across-blocks statements. This is *not* a hypothesis this file believes;
  it is the silent identification the shipped equation requires in order to be well formed, made
  visible. It is false in the paper's own setting whenever the body carries parameters beyond the
  slack block.
* `hκ : κ ≠ 0` in the refutation statements. The shipped equation divides by `κ²`, so `κ = 0` is
  outside its own scope. `ha : a ≠ 0` plays the same role in the scaled statements, where the
  unit conversion is left free.
* `hexact : H⋆ = H̄` in the *abstract matrix* lemmas — `trace_eq_zero_of_…` and
  `symmPart_eq_zero_of_…` in both their `κ` and their scaled forms,
  `hessianCrossCovDecomposition_iff_symmPart_eq_zero`, the two remainder lemmas,
  `not_hessianCrossCovDecomposition_of_trace_pos`, `not_hessianCrossCovDecomposition_rankOne`
  and `trace_eq_zero_of_hessianCrossCovDecompositionAcrossBlocks`. Those lemmas are pure matrix
  algebra and take the exact identity as an *input*, so a reader has to know it is one there. It
  is not a hypothesis of the results that carry the refutation: in
  `pullbackHessian_not_hessianCrossCovDecomposition`,
  `pullbackHessian_not_scaledCrossCovDecomposition`, `pullbackHessian_not_trace_lower_bound`,
  `batchAverageHessian_not_hessianCrossCovDecomposition` and the two closed quadratic instances
  it is discharged by `hessianMatrix_pullbackLoss` or `hessianMatrix_batchAverage`, so those
  statements assume nothing whatever about the two Hessians they compare.
* `hε : 0 < ε` in the general-measure results — the radius of the ball on which the domination
  package is imposed — and `hw : ∑ᵢ wᵢ = 1` in `hessianMatrix_pullbackLoss_of_const`,
  `hessianMatrix_pullbackLoss_sq` and `trace_hessianMatrix_pullbackLoss_sq`, where it is what
  makes a weighted sum an expectation. The exact identity itself needs no normalization of the
  weights, and none is imposed on it.

Satisfiability instances guard against vacuity on both sides. On the side of the hypotheses:
`quadraticPullback_not_hessianCrossCovDecomposition` discharges the differentiability package of
the headline refutation with a concrete quadratic loss of nonzero curvature, and
`iteratedDeriv_two_pullbackLossIntegral_sq` and `iteratedDeriv_two_pullbackLossIntegral_sq_sin`
discharge the entire domination package of the general-measure identity under an arbitrary
probability measure — the first with a constant body, the second with the batch-dependent body
`sin x`, whose shifted balls genuinely differ from batch to batch. None adds a hypothesis
beyond `κ ≠ 0`. `quadraticPullback_not_trace_lower_bound` does the same for the
displayed-inequality refutation, and `hessianMatrix_pullbackLoss_sq` evaluates the pullback
Hessian of those closed instances — as soon as the weights sum to one, the case the paper is
about — to the nonzero matrix `2 E_jj` of trace two, which by the exact identity is also the
value of the batch-average argument they are compared against; so the instances are not
degenerate ones in which both sides collapse to zero. On the side of the refuted statement:
`hessianCrossCovDecomposition_antisymmCrossCov` exhibits a nonzero cross-covariance that *does*
satisfy the shipped equation, and `hessianCrossCovDecomposition_iff_symmPart_eq_zero` shows the
refutation is sharp — so `¬ HessianCrossCovDecomposition …` is a statement about the
cross-covariance the paper measures and not an artefact of an equation nothing could satisfy.
The realizability of the rank-one witness is claimed only in the form actually
proved: `rankOneCrossCov_eq_crossCovEstimator` is an algebraic evaluation of the estimator
function at window length one on uncentred inputs — the protocol's own centring makes every
length-one estimate zero (`crossCovEstimator_one_bodyFluct_eq_zero`) — and the
protocol-faithful realization is the window-length-two run of `rankOneCrossCov_realizable`.

Nothing assigned to this file was left unproved.
-/

open Finset Matrix Metric MeasureTheory

namespace IcnnLift

/-! ### The pullback landscape -/

section Definition

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The **pullback landscape** of the lift, for a batch law carried by a finite family of
weights: `L̃(b) = ∑ᵢ wᵢ · (L ∘ ψ)(b + hᵢ)`, where `f i` is the composite loss `L ∘ ψ` evaluated
on the batch drawn with weight `w i` and `h i` is the body output `h_φ(Xᵢ)` on that batch. The
slack `b` enters every summand through the same translation, which is the entire content of the
exact identity below. -/
noncomputable def pullbackLoss (w : ι → ℝ) (f : ι → E → ℝ) (h : ι → E) (b : E) : ℝ :=
  ∑ i, w i * f i (b + h i)

end Definition

/-! ### The exact identity in one variable, unconditionally -/

section FiniteScalar

variable {ι : Type*} [Fintype ι] {w : ι → ℝ} {g g1 : ι → ℝ → ℝ} {g2 : ι → ℝ} {t : ℝ}

/-- The derivative of a finite weighted sum is the weighted sum of the derivatives. -/
theorem hasDerivAt_weightedSum (hg1 : ∀ i y, HasDerivAt (g i) (g1 i y) y) (t : ℝ) :
    HasDerivAt (fun s => ∑ i, w i * g i s) (∑ i, w i * g1 i t) t := by
  have key : (fun s => ∑ i, w i * g i s) = ∑ i : ι, (fun s => w i * g i s) := by
    funext s; simp
  rw [key]
  exact HasDerivAt.sum fun i _ => (hg1 i t).const_mul (w i)

/-- The derivative function of a finite weighted sum, globally. -/
theorem deriv_weightedSum (hg1 : ∀ i y, HasDerivAt (g i) (g1 i y) y) :
    (deriv fun s => ∑ i, w i * g i s) = fun s => ∑ i, w i * g1 i s :=
  funext fun s => (hasDerivAt_weightedSum hg1 s).deriv

/-- Differentiating the weighted sum a second time again produces only a weighted sum: there is
no term coupling the weights to the summands. -/
theorem hasDerivAt_deriv_weightedSum (hg1 : ∀ i y, HasDerivAt (g i) (g1 i y) y)
    (hg2 : ∀ i, HasDerivAt (g1 i) (g2 i) t) :
    HasDerivAt (deriv fun s => ∑ i, w i * g i s) (∑ i, w i * g2 i) t := by
  rw [deriv_weightedSum hg1]
  have key : (fun s => ∑ i, w i * g1 i s) = ∑ i : ι, (fun s => w i * g1 i s) := by
    funext s; simp
  rw [key]
  exact HasDerivAt.sum fun i _ => (hg2 i).const_mul (w i)

/-- **The second derivative of a finite weighted sum is the weighted sum of the second
derivatives.** This is the mathematical core of the refutation, stripped of the paper's notation:
a batch average is a finite linear combination, and differentiation is linear. -/
theorem iteratedDeriv_two_weightedSum (hg1 : ∀ i y, HasDerivAt (g i) (g1 i y) y)
    (hg2 : ∀ i, HasDerivAt (g1 i) (g2 i) t) :
    iteratedDeriv 2 (fun s => ∑ i, w i * g i s) t = ∑ i, w i * iteratedDeriv 2 (g i) t := by
  have hd : ∀ i, iteratedDeriv 2 (g i) t = g2 i := by
    intro i
    have h1 : deriv (g i) = g1 i := funext fun y => (hg1 i y).deriv
    rw [iteratedDeriv_succ, iteratedDeriv_one, h1]
    exact (hg2 i).deriv
  rw [iteratedDeriv_succ, iteratedDeriv_one, (hasDerivAt_deriv_weightedSum hg1 hg2).deriv]
  exact Finset.sum_congr rfl fun i _ => by rw [hd i]

end FiniteScalar

section PullbackScalar

variable {ι : Type*} [Fintype ι] {w : ι → ℝ} {f f1 : ι → ℝ → ℝ} {f2 : ι → ℝ} {h : ι → ℝ} {b : ℝ}

/-- The first derivative of the pullback landscape in one slack variable: the batch average of
the per-batch derivatives, each evaluated at its own shifted point `b + hᵢ`. -/
theorem hasDerivAt_pullbackLoss (hf1 : ∀ i y, HasDerivAt (f i) (f1 i y) y) (b : ℝ) :
    HasDerivAt (pullbackLoss w f h) (∑ i, w i * f1 i (b + h i)) b :=
  hasDerivAt_weightedSum (g := fun i s => f i (s + h i)) (g1 := fun i s => f1 i (s + h i))
    (fun i y => HasDerivAt.comp_add_const y (h i) (hf1 i (y + h i))) b

/-- The second derivative of the pullback landscape in one slack variable, in `HasDerivAt`
form. -/
theorem hasDerivAt_deriv_pullbackLoss (hf1 : ∀ i y, HasDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasDerivAt (f1 i) (f2 i) (b + h i)) :
    HasDerivAt (deriv (pullbackLoss w f h)) (∑ i, w i * f2 i) b :=
  hasDerivAt_deriv_weightedSum (g := fun i s => f i (s + h i)) (g1 := fun i s => f1 i (s + h i))
    (fun i y => HasDerivAt.comp_add_const y (h i) (hf1 i (y + h i)))
    (fun i => HasDerivAt.comp_add_const b (h i) (hf2 i))

/-- **The exact identity, one slack variable, finite batch law, unconditionally.**
`L̃''(b) = ∑ᵢ wᵢ (L ∘ ψ)''(b + hᵢ)`. The right-hand side is the batch average of the per-batch
loss curvatures; there is no further term. Compare the shipped equation of
`A1_proofs.tex`, which adds `κ⁻² · ½ (Σ_slack + Σ_slackᵀ)` to exactly this right-hand side. -/
theorem iteratedDeriv_two_pullbackLoss (hf1 : ∀ i y, HasDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasDerivAt (f1 i) (f2 i) (b + h i)) :
    iteratedDeriv 2 (pullbackLoss w f h) b = ∑ i, w i * iteratedDeriv 2 (f i) (b + h i) := by
  have h0 : iteratedDeriv 2 (pullbackLoss w f h) b
      = ∑ i, w i * iteratedDeriv 2 (fun s => f i (s + h i)) b :=
    iteratedDeriv_two_weightedSum (g := fun i s => f i (s + h i))
      (g1 := fun i s => f1 i (s + h i))
      (fun i y => HasDerivAt.comp_add_const y (h i) (hf1 i (y + h i)))
      (fun i => HasDerivAt.comp_add_const b (h i) (hf2 i))
  rw [h0]
  exact Finset.sum_congr rfl fun i _ => by rw [iteratedDeriv_comp_add_const 2 (f i) (h i)]

/-- **The refutation in its sharpest scalar form.** If the pullback second derivative were the
batch-averaged loss curvature *plus* an extra term `c`, then `c = 0`. No hypothesis is placed on
`c`: it is not assumed small, not assumed to be a symmetrized cross-covariance, and not assumed
to carry any particular scaling in the positivity-map scale `κ`. -/
theorem pullbackLoss_no_extra_term (hf1 : ∀ i y, HasDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasDerivAt (f1 i) (f2 i) (b + h i)) {c : ℝ}
    (hc : iteratedDeriv 2 (pullbackLoss w f h) b
      = (∑ i, w i * iteratedDeriv 2 (f i) (b + h i)) + c) : c = 0 := by
  have := iteratedDeriv_two_pullbackLoss hf1 hf2 (w := w)
  rw [this] at hc
  linarith

end PullbackScalar

/-! ### The exact identity as an equality of Hessian matrices -/

section BatchAverage

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  {w : ι → ℝ} {F : ι → E → ℝ} {F1 : ι → E → (E →L[ℝ] ℝ)}
  {F2 : ι → (E →L[ℝ] E →L[ℝ] ℝ)} {c : E}

/-- The Fréchet derivative of a finite batch average is the batch average of the Fréchet
derivatives. Nothing is assumed about how the batch index enters `F i`. -/
theorem hasFDerivAt_batchAverage (hF1 : ∀ i y, HasFDerivAt (F i) (F1 i y) y) (c : E) :
    HasFDerivAt (fun x => ∑ i, w i * F i x) (∑ i, w i • F1 i c) c := by
  have key : (fun x => ∑ i, w i * F i x) = ∑ i : ι, (fun x => w i * F i x) := by
    funext x; simp
  rw [key]
  exact HasFDerivAt.sum fun i _ => (hF1 i c).const_mul (w i)

/-- The Fréchet derivative function of a finite batch average, globally. -/
theorem fderiv_batchAverage (hF1 : ∀ i y, HasFDerivAt (F i) (F1 i y) y) :
    fderiv ℝ (fun x => ∑ i, w i * F i x) = fun c => ∑ i, w i • F1 i c :=
  funext fun c => (hasFDerivAt_batchAverage hF1 c).fderiv

/-- **Differentiation commutes with the batch average, twice.** The second Fréchet derivative of
a finite batch average is the batch average of the second Fréchet derivatives, with no further
term. This is the general form of the refutation: it assumes nothing whatsoever about how the
batch enters the per-sample loss, so it is not an artefact of the translation structure
`θ̃ = b + h_φ(X)` and it applies equally to differentiation in the full body parameter `φ`. -/
theorem hasFDerivAt_fderiv_batchAverage (hF1 : ∀ i y, HasFDerivAt (F i) (F1 i y) y)
    (hF2 : ∀ i, HasFDerivAt (F1 i) (F2 i) c) :
    HasFDerivAt (fderiv ℝ (fun x => ∑ i, w i * F i x)) (∑ i, w i • F2 i) c := by
  rw [fderiv_batchAverage hF1]
  have key : (fun x => ∑ i, w i • F1 i x) = ∑ i : ι, (fun x => w i • F1 i x) := by
    funext x; simp
  rw [key]
  exact HasFDerivAt.sum fun i _ => (hF2 i).const_smul (w i)

end BatchAverage

section PullbackFrechet

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  {w : ι → ℝ} {f : ι → E → ℝ} {f1 : ι → E → (E →L[ℝ] ℝ)}
  {f2 : ι → (E →L[ℝ] E →L[ℝ] ℝ)} {h : ι → E} {b : E}

omit [Fintype ι] in
private theorem hasFDerivAt_shift (i : ι) (y : E) (hf1 : ∀ z, HasFDerivAt (f i) (f1 i z) z) :
    HasFDerivAt (fun x => f i (x + h i)) (f1 i (y + h i)) y := by
  have hshift : HasFDerivAt (fun x : E => x + h i) (ContinuousLinearMap.id ℝ E) y :=
    (hasFDerivAt_id y).add_const (h i)
  simpa only [Function.comp_def, ContinuousLinearMap.comp_id] using
    (hf1 (y + h i)).comp y hshift

omit [Fintype ι] in
private theorem hasFDerivAt_shift' (i : ι) (hf2 : HasFDerivAt (f1 i) (f2 i) (b + h i)) :
    HasFDerivAt (fun x => f1 i (x + h i)) (f2 i) b := by
  have hshift : HasFDerivAt (fun x : E => x + h i) (ContinuousLinearMap.id ℝ E) b :=
    (hasFDerivAt_id b).add_const (h i)
  simpa only [Function.comp_def, ContinuousLinearMap.comp_id] using hf2.comp b hshift

/-- The Fréchet derivative of the pullback landscape in the slack block. -/
theorem hasFDerivAt_pullbackLoss (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y) (b : E) :
    HasFDerivAt (pullbackLoss w f h) (∑ i, w i • f1 i (b + h i)) b :=
  hasFDerivAt_batchAverage (F := fun i x => f i (x + h i)) (F1 := fun i x => f1 i (x + h i))
    (fun i y => hasFDerivAt_shift i y (hf1 i)) b

/-- The Fréchet derivative function of the pullback landscape, globally. -/
theorem fderiv_pullbackLoss (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y) :
    fderiv ℝ (pullbackLoss w f h) = fun b => ∑ i, w i • f1 i (b + h i) :=
  funext fun b => (hasFDerivAt_pullbackLoss hf1 b).fderiv

/-- **The exact identity as a second Fréchet derivative.** The second derivative of the pullback
landscape at `b` is the batch average `∑ᵢ wᵢ • f2 i` of the per-batch second derivatives, each
taken at its own shifted point. No bilinear form beyond that average appears. -/
theorem hasFDerivAt_fderiv_pullbackLoss (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i)) :
    HasFDerivAt (fderiv ℝ (pullbackLoss w f h)) (∑ i, w i • f2 i) b :=
  hasFDerivAt_fderiv_batchAverage (F := fun i x => f i (x + h i))
    (F1 := fun i x => f1 i (x + h i))
    (fun i y => hasFDerivAt_shift i y (hf1 i)) (fun i => hasFDerivAt_shift' i (hf2 i))

end PullbackFrechet

section HessianMatrix

variable {ι : Type*} [Fintype ι] {d : ℕ}

/-- The `d × d` **Hessian matrix in the slack block**: entry `(j, k)` is the second derivative of
`F` at `b` in the directions of the `j`-th and `k`-th slack coordinates. Writing the Hessian as a
matrix of this size is what makes the dimensional defect of the shipped equation visible: the
paper's `H⋆ = ∇²_φ L̃` is `p × p` in the body-parameter dimension `p`, while `E_X[∇²_θ̃ L]` and
`Σ_slack` are `d × d` in the slack dimension. -/
noncomputable def hessianMatrix (F : (Fin d → ℝ) → ℝ) (b : Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun j k => fderiv ℝ (fderiv ℝ F) b (Pi.single j 1) (Pi.single k 1)

/-- **The exact identity as an equality of `d × d` matrices, in full generality.** The Hessian
matrix of a finite batch average is the batch average of the per-sample Hessian matrices, entry
by entry. Nothing is assumed about how the batch index enters the per-sample loss, so this holds
for differentiation in the full body parameter `φ` as well as in the slack `b`: whichever
parameters are differentiated, the batch average contributes nothing beyond the average of the
per-sample curvatures, and in particular no cross-covariance. Read in `φ`, the right-hand side is
the batch average of the per-sample `φ`-curvatures; that is the shipped equation's first term
`E_X[∇²_θ̃ L]` on the slack block, where (A3) makes the body Jacobian the identity, but off that
block the two are separated by a deterministic chain-rule term through `∂θ̃/∂φ` which this file
does not compute. -/
theorem hessianMatrix_batchAverage {w : ι → ℝ} {F : ι → (Fin d → ℝ) → ℝ}
    {F1 : ι → (Fin d → ℝ) → ((Fin d → ℝ) →L[ℝ] ℝ)}
    {F2 : ι → ((Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] ℝ)} {c : Fin d → ℝ}
    (hF1 : ∀ i y, HasFDerivAt (F i) (F1 i y) y) (hF2 : ∀ i, HasFDerivAt (F1 i) (F2 i) c) :
    hessianMatrix (fun x => ∑ i, w i * F i x) c = ∑ i, w i • hessianMatrix (F i) c := by
  have hL : fderiv ℝ (fderiv ℝ (fun x => ∑ i, w i * F i x)) c = ∑ i, w i • F2 i :=
    (hasFDerivAt_fderiv_batchAverage hF1 hF2).fderiv
  have hi : ∀ i, fderiv ℝ (fderiv ℝ (F i)) c = F2 i := by
    intro i
    have hfd : fderiv ℝ (F i) = F1 i := funext fun y => (hF1 i y).fderiv
    rw [hfd]
    exact (hF2 i).fderiv
  ext j k
  simp [hessianMatrix, hL, hi, Matrix.sum_apply]

variable {w : ι → ℝ} {f : ι → (Fin d → ℝ) → ℝ}
  {f1 : ι → (Fin d → ℝ) → ((Fin d → ℝ) →L[ℝ] ℝ)}
  {f2 : ι → ((Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] ℝ)}
  {h : ι → (Fin d → ℝ)} {b : Fin d → ℝ}

/-- **The exact identity as an equality of `d × d` matrices.**
`∇²_b L̃(b) = ∑ᵢ wᵢ · ∇²_θ̃ (L ∘ ψ)(b + hᵢ)`, entry by entry, with no cross-covariance term.
This is the statement the shipped decomposition of `A1_proofs.tex` contradicts: the shipped
equation adds `κ⁻² · ½ (Σ_slack + Σ_slackᵀ)` to the right-hand side of exactly this identity. -/
theorem hessianMatrix_pullbackLoss (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i)) :
    hessianMatrix (pullbackLoss w f h) b = ∑ i, w i • hessianMatrix (f i) (b + h i) := by
  have hL : fderiv ℝ (fderiv ℝ (pullbackLoss w f h)) b = ∑ i, w i • f2 i :=
    (hasFDerivAt_fderiv_pullbackLoss hf1 hf2).fderiv
  have hi : ∀ i, fderiv ℝ (fderiv ℝ (f i)) (b + h i) = f2 i := by
    intro i
    have hfd : fderiv ℝ (f i) = f1 i := funext fun y => (hf1 i y).fderiv
    rw [hfd]
    exact (hf2 i).fderiv
  ext j k
  simp [hessianMatrix, hL, hi, Matrix.sum_apply]

/-- A batch law averages. If every per-batch loss has the same curvature `H` at its own shifted
point, the pullback Hessian is `H` itself. This is where the normalization `∑ᵢ wᵢ = 1` is used,
and it is what licenses reading `∑ᵢ wᵢ • ·` as the expectation `E_X[·]` of the paper. -/
theorem hessianMatrix_pullbackLoss_of_const (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i)) (hw : ∑ i, w i = 1)
    {H : Matrix (Fin d) (Fin d) ℝ} (hconst : ∀ i, hessianMatrix (f i) (b + h i) = H) :
    hessianMatrix (pullbackLoss w f h) b = H := by
  rw [hessianMatrix_pullbackLoss hf1 hf2]
  calc ∑ i, w i • hessianMatrix (f i) (b + h i) = ∑ i, w i • H :=
        Finset.sum_congr rfl fun i _ => by rw [hconst i]
    _ = (∑ i, w i) • H := (Finset.sum_smul).symm
    _ = H := by rw [hw, one_smul]

/-- The trace form of the exact identity: `tr ∇²_b L̃ = ∑ᵢ wᵢ tr ∇²_θ̃ (L ∘ ψ)(b + hᵢ)`. The
shipped trace equation of `A1_proofs.tex` adds `σ_Jac²/κ²` to the right-hand side of exactly
this identity, and it is that addition which the refutation below rules out. -/
theorem trace_hessianMatrix_pullbackLoss (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i)) :
    (hessianMatrix (pullbackLoss w f h) b).trace
      = ∑ i, w i * (hessianMatrix (f i) (b + h i)).trace := by
  rw [hessianMatrix_pullbackLoss hf1 hf2, Matrix.trace_sum]
  exact Finset.sum_congr rfl fun i _ => by rw [Matrix.trace_smul]; rfl

end HessianMatrix

/-! ### The exact identity for a general batch law, under stated domination -/

section GeneralMeasure

variable {α : Type*} [MeasurableSpace α]

/-- The **pullback landscape for a general batch law** `μ`:
`L̃(b) = ∫ (L ∘ ψ)(b + h_φ(x)) dμ(x)`. -/
noncomputable def pullbackLossIntegral (μ : Measure α) (f : α → ℝ → ℝ) (h : α → ℝ) (b : ℝ) : ℝ :=
  ∫ x, f x (b + h x) ∂μ

variable {μ : Measure α} {f f1 f2 : α → ℝ → ℝ} {h : α → ℝ} {b₀ ε : ℝ} {bound1 bound2 : α → ℝ}

omit [MeasurableSpace α] in
private theorem mem_ball_shift {b b₀ c ε : ℝ} (hb : b ∈ ball b₀ ε) : b + c ∈ ball (b₀ + c) ε := by
  simpa [Metric.mem_ball, dist_add_right] using hb

/-- Differentiation under the integral sign, first order, for the pullback landscape. The
hypotheses are mathlib's domination package for
`hasDerivAt_integral_of_dominated_loc_of_deriv_le`, transported through the translation
`b ↦ b + h x`: measurability and integrability of the integrand on a ball around the base point,
a derivative `f1 x` for almost every batch on the corresponding shifted ball, and an integrable
envelope `bound1` for that derivative. The conclusion also records that the derivative integrand
is itself integrable, which is what feeds the second-order step. -/
theorem hasDerivAt_pullbackLossIntegral
    (hf_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f x (b + h x)) μ)
    (hf_int : ∀ b ∈ ball b₀ ε, Integrable (fun x => f x (b + h x)) μ)
    (hf1_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f1 x (b + h x)) μ)
    (hf1_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f x) (f1 x y) y)
    (hf1_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f1 x y‖ ≤ bound1 x)
    (hbound1 : Integrable bound1 μ) :
    ∀ b ∈ ball b₀ ε,
      Integrable (fun x => f1 x (b + h x)) μ ∧
      HasDerivAt (pullbackLossIntegral μ f h) (∫ x, f1 x (b + h x) ∂μ) b := by
  intro b hb
  have hs : ball b₀ ε ∈ nhds b := Metric.isOpen_ball.mem_nhds hb
  have hbnd : ∀ᵐ x ∂μ, ∀ y ∈ ball b₀ ε, ‖f1 x (y + h x)‖ ≤ bound1 x := by
    filter_upwards [hf1_bound] with x hx y hy
    exact hx (y + h x) (mem_ball_shift hy)
  have hdiff : ∀ᵐ x ∂μ, ∀ y ∈ ball b₀ ε,
      HasDerivAt (fun z => f x (z + h x)) (f1 x (y + h x)) y := by
    filter_upwards [hf1_deriv] with x hx y hy
    exact HasDerivAt.comp_add_const y (h x) (hx (y + h x) (mem_ball_shift hy))
  exact hasDerivAt_integral_of_dominated_loc_of_deriv_le (bound := bound1)
    (F := fun z x => f x (z + h x)) (F' := fun z x => f1 x (z + h x)) hs
    (Filter.eventually_of_mem hs hf_meas) (hf_int b hb) (hf1_meas b hb) hbnd hbound1 hdiff

/-- Differentiation under the integral sign, second order: the derivative of the derivative of
the pullback landscape is the batch average of the per-batch second derivatives. The domination
package is supplied twice, once for each differentiation; the first-order package must hold on a
whole ball rather than at a point, because the second derivative is a derivative of the function
`b ↦ L̃'(b)` and that function has to be identified near the base point. -/
theorem hasDerivAt_deriv_pullbackLossIntegral
    (hε : 0 < ε)
    (hf_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f x (b + h x)) μ)
    (hf_int : ∀ b ∈ ball b₀ ε, Integrable (fun x => f x (b + h x)) μ)
    (hf1_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f1 x (b + h x)) μ)
    (hf1_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f x) (f1 x y) y)
    (hf1_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f1 x y‖ ≤ bound1 x)
    (hbound1 : Integrable bound1 μ)
    (hf2_meas : AEStronglyMeasurable (fun x => f2 x (b₀ + h x)) μ)
    (hf2_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f1 x) (f2 x y) y)
    (hf2_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f2 x y‖ ≤ bound2 x)
    (hbound2 : Integrable bound2 μ) :
    HasDerivAt (deriv (pullbackLossIntegral μ f h)) (∫ x, f2 x (b₀ + h x) ∂μ) b₀ := by
  have hstep := hasDerivAt_pullbackLossIntegral hf_meas hf_int hf1_meas hf1_deriv hf1_bound hbound1
  have hb₀ : b₀ ∈ ball b₀ ε := Metric.mem_ball_self hε
  have hs : ball b₀ ε ∈ nhds b₀ := Metric.isOpen_ball.mem_nhds hb₀
  have hbnd : ∀ᵐ x ∂μ, ∀ y ∈ ball b₀ ε, ‖f2 x (y + h x)‖ ≤ bound2 x := by
    filter_upwards [hf2_bound] with x hx y hy
    exact hx (y + h x) (mem_ball_shift hy)
  have hdiff : ∀ᵐ x ∂μ, ∀ y ∈ ball b₀ ε,
      HasDerivAt (fun z => f1 x (z + h x)) (f2 x (y + h x)) y := by
    filter_upwards [hf2_deriv] with x hx y hy
    exact HasDerivAt.comp_add_const y (h x) (hx (y + h x) (mem_ball_shift hy))
  have h2 : HasDerivAt (fun z => ∫ x, f1 x (z + h x) ∂μ) (∫ x, f2 x (b₀ + h x) ∂μ) b₀ :=
    (hasDerivAt_integral_of_dominated_loc_of_deriv_le (bound := bound2)
      (F := fun z x => f1 x (z + h x)) (F' := fun z x => f2 x (z + h x)) hs
      (Filter.eventually_of_mem hs hf1_meas) (hstep b₀ hb₀).1 hf2_meas hbnd hbound2 hdiff).2
  exact h2.congr_of_eventuallyEq (Filter.eventually_of_mem hs fun b hb => (hstep b hb).2.deriv)

/-- **The exact identity for a general batch law.** Under the stated domination hypotheses,
`L̃''(b₀) = ∫ f2 x (b₀ + h x) dμ(x) = E_X[(L ∘ ψ)''(b₀ + h_φ(X))]`, with no cross-covariance
term. Differentiation under the integral sign is where "the batch expectation commutes with
differentiation" becomes a theorem rather than a slogan, and the domination hypotheses are
exactly its price. -/
theorem iteratedDeriv_two_pullbackLossIntegral
    (hε : 0 < ε)
    (hf_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f x (b + h x)) μ)
    (hf_int : ∀ b ∈ ball b₀ ε, Integrable (fun x => f x (b + h x)) μ)
    (hf1_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f1 x (b + h x)) μ)
    (hf1_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f x) (f1 x y) y)
    (hf1_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f1 x y‖ ≤ bound1 x)
    (hbound1 : Integrable bound1 μ)
    (hf2_meas : AEStronglyMeasurable (fun x => f2 x (b₀ + h x)) μ)
    (hf2_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f1 x) (f2 x y) y)
    (hf2_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f2 x y‖ ≤ bound2 x)
    (hbound2 : Integrable bound2 μ) :
    iteratedDeriv 2 (pullbackLossIntegral μ f h) b₀ = ∫ x, f2 x (b₀ + h x) ∂μ := by
  rw [iteratedDeriv_succ, iteratedDeriv_one]
  exact (hasDerivAt_deriv_pullbackLossIntegral hε hf_meas hf_int hf1_meas hf1_deriv hf1_bound
    hbound1 hf2_meas hf2_deriv hf2_bound hbound2).deriv

/-- The same identity with the integrand written intrinsically, as the second derivative of the
per-batch composite loss: `L̃''(b₀) = ∫ (L ∘ ψ)''(b₀ + h_φ(x)) dμ(x)`. This is the form quoted
in `docs/design/lemma1_rederivation.md` §1. -/
theorem iteratedDeriv_two_pullbackLossIntegral_eq_integral
    (hε : 0 < ε)
    (hf_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f x (b + h x)) μ)
    (hf_int : ∀ b ∈ ball b₀ ε, Integrable (fun x => f x (b + h x)) μ)
    (hf1_meas : ∀ b ∈ ball b₀ ε, AEStronglyMeasurable (fun x => f1 x (b + h x)) μ)
    (hf1_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f x) (f1 x y) y)
    (hf1_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f1 x y‖ ≤ bound1 x)
    (hbound1 : Integrable bound1 μ)
    (hf2_meas : AEStronglyMeasurable (fun x => f2 x (b₀ + h x)) μ)
    (hf2_deriv : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, HasDerivAt (f1 x) (f2 x y) y)
    (hf2_bound : ∀ᵐ x ∂μ, ∀ y ∈ ball (b₀ + h x) ε, ‖f2 x y‖ ≤ bound2 x)
    (hbound2 : Integrable bound2 μ) :
    iteratedDeriv 2 (pullbackLossIntegral μ f h) b₀
      = ∫ x, iteratedDeriv 2 (f x) (b₀ + h x) ∂μ := by
  rw [iteratedDeriv_two_pullbackLossIntegral hε hf_meas hf_int hf1_meas hf1_deriv hf1_bound
    hbound1 hf2_meas hf2_deriv hf2_bound hbound2]
  refine (MeasureTheory.integral_congr_ae ?_).symm
  filter_upwards [hf1_deriv, hf2_deriv] with x hx1 hx2
  have hnb : ball (b₀ + h x) ε ∈ nhds (b₀ + h x) :=
    Metric.isOpen_ball.mem_nhds (Metric.mem_ball_self hε)
  have hev : deriv (f x) =ᶠ[nhds (b₀ + h x)] f1 x :=
    Filter.eventually_of_mem hnb fun y hy => (hx1 y hy).deriv
  rw [iteratedDeriv_succ, iteratedDeriv_one, hev.deriv_eq]
  exact (hx2 (b₀ + h x) (Metric.mem_ball_self hε)).deriv

end GeneralMeasure

section GeneralMeasureWitness

variable {α : Type*} [MeasurableSpace α]

/-- **The domination package is satisfiable, witnessed end to end.** For the quadratic
integrand `f x y = y²`, a zero body and an arbitrary probability measure as the batch law,
every hypothesis of `iteratedDeriv_two_pullbackLossIntegral` is discharged outright with
constant envelopes, and the identity returns `L̃''(b₀) = ∫ 2 dμ = 2` — the correct second
derivative of `L̃(b) = b²`. The general-measure theorem above is therefore not vacuously
conditioned: its hypotheses are exactly the price of differentiating under the integral sign,
and a genuine model pays it. -/
theorem iteratedDeriv_two_pullbackLossIntegral_sq (μ : Measure α) [IsProbabilityMeasure μ]
    (b₀ : ℝ) :
    iteratedDeriv 2 (pullbackLossIntegral μ (fun _ y => y ^ 2) (fun _ => 0)) b₀ = 2 := by
  have key := iteratedDeriv_two_pullbackLossIntegral (μ := μ) (f := fun _ y => y ^ 2)
    (f1 := fun _ y => 2 * y) (f2 := fun _ _ => 2) (h := fun _ => 0) (b₀ := b₀) (ε := 1)
    (bound1 := fun _ => 2 * (|b₀| + 1)) (bound2 := fun _ => 2)
    one_pos
    (fun _ _ => aestronglyMeasurable_const)
    (fun _ _ => integrable_const _)
    (fun _ _ => aestronglyMeasurable_const)
    (Filter.Eventually.of_forall fun x y _ => by
      simpa [mul_comm] using hasDerivAt_pow 2 y)
    (Filter.Eventually.of_forall fun x y hy => by
      have h1 : |y - (b₀ + 0)| < 1 := by simpa [Real.dist_eq] using hy
      have h2 : |y| - |b₀| ≤ |y - b₀| := by
        simpa using abs_sub_abs_le_abs_sub y b₀
      rw [Real.norm_eq_abs, abs_mul, abs_two]
      have h3 : |y| ≤ |b₀| + 1 := by
        rw [add_zero] at h1
        linarith
      linarith)
    (integrable_const _)
    aestronglyMeasurable_const
    (Filter.Eventually.of_forall fun x y _ => by
      simpa using (hasDerivAt_id y).const_mul (2 : ℝ))
    (Filter.Eventually.of_forall fun x y _ => by
      rw [Real.norm_eq_abs, abs_two])
    (integrable_const _)
  rw [key]
  simp

/-- **The domination package is satisfiable with a genuinely batch-dependent body.** The
constant-body witness above leaves every shifted ball `ball (b₀ + h x) ε` equal to the same
ball, so it does not exercise the part of the domination package that varies with the batch.
Here the body is `h(x) = sin x` under an arbitrary probability measure on `ℝ`: the translation
`b ↦ b + sin x` genuinely moves with the batch, the per-batch balls are genuinely distinct, and
every hypothesis of `iteratedDeriv_two_pullbackLossIntegral` is still discharged outright, with
constant envelopes valid across all of them at once. The identity returns
`L̃''(b₀) = ∫ 2 dμ = 2`, the correct second derivative of
`L̃(b) = ∫ (b + sin x)² dμ(x) = b² + 2b ∫ sin dμ + ∫ sin² dμ`. The interchange of
differentiation and batch expectation is therefore witnessed in the regime the paper occupies —
a body that moves with the conditioning batch — not only in the degenerate one. -/
theorem iteratedDeriv_two_pullbackLossIntegral_sq_sin (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (b₀ : ℝ) :
    iteratedDeriv 2 (pullbackLossIntegral μ (fun _ y => y ^ 2) Real.sin) b₀ = 2 := by
  have habs : ∀ (b : ℝ) (x : ℝ), b ∈ ball b₀ 1 → |b + Real.sin x| ≤ |b₀| + 2 := by
    intro b x hb
    have hb' : |b - b₀| < 1 := by simpa [Real.dist_eq] using hb
    have h1 : |b| - |b₀| ≤ |b - b₀| := by simpa using abs_sub_abs_le_abs_sub b b₀
    calc |b + Real.sin x| ≤ |b| + |Real.sin x| := abs_add_le _ _
      _ ≤ (|b₀| + 1) + 1 := add_le_add (by linarith) (Real.abs_sin_le_one x)
      _ = |b₀| + 2 := by ring
  have key := iteratedDeriv_two_pullbackLossIntegral (μ := μ) (f := fun _ y => y ^ 2)
    (f1 := fun _ y => 2 * y) (f2 := fun _ _ => 2) (h := Real.sin) (b₀ := b₀) (ε := 1)
    (bound1 := fun _ => 2 * (|b₀| + 3)) (bound2 := fun _ => 2)
    one_pos
    (fun b _ => ((continuous_const.add Real.continuous_sin).pow 2).aestronglyMeasurable)
    (fun b hb => by
      refine (integrable_const ((|b₀| + 2) ^ 2)).mono'
        (((continuous_const.add Real.continuous_sin).pow 2).aestronglyMeasurable)
        (Filter.Eventually.of_forall fun x => ?_)
      have h1 := habs b x hb
      have hnn : ‖(b + Real.sin x) ^ 2‖ = (b + Real.sin x) ^ 2 := by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      rw [hnn]
      nlinarith [sq_abs (b + Real.sin x), abs_nonneg (b + Real.sin x)])
    (fun b _ => ((continuous_const.add Real.continuous_sin).const_mul 2).aestronglyMeasurable)
    (Filter.Eventually.of_forall fun x y _ => by
      simpa [mul_comm] using hasDerivAt_pow 2 y)
    (Filter.Eventually.of_forall fun x y hy => by
      have h1 : |y - (b₀ + Real.sin x)| < 1 := by simpa [Real.dist_eq] using hy
      have h2 : |y| - |b₀ + Real.sin x| ≤ |y - (b₀ + Real.sin x)| := by
        simpa using abs_sub_abs_le_abs_sub y (b₀ + Real.sin x)
      have h3 : |b₀ + Real.sin x| ≤ |b₀| + 1 := by
        have ht := abs_add_le b₀ (Real.sin x)
        have hs := Real.abs_sin_le_one x
        linarith
      rw [Real.norm_eq_abs, abs_mul, abs_two]
      have h4 : |y| ≤ |b₀| + 3 := by linarith
      linarith)
    (integrable_const _)
    aestronglyMeasurable_const
    (Filter.Eventually.of_forall fun x y _ => by
      simpa using (hasDerivAt_id y).const_mul (2 : ℝ))
    (Filter.Eventually.of_forall fun x y _ => by
      rw [Real.norm_eq_abs, abs_two])
    (integrable_const _)
  rw [key]
  simp

end GeneralMeasureWitness

/-! ### The refutation of the shipped decomposition -/

section Refutation

variable {d p : ℕ}

/-- **The shipped decomposition of `A1_proofs.tex`**, as a predicate on matrices:
`H⋆ = E_X[∇²_θ̃ L] + κ⁻² · ½ (Σ + Σᵀ)`. This is the equation the shipped proof asserts without
derivation, written in the reading under which `H⋆` denotes the `d × d` slack block, so that both
sides have the same size. -/
def HessianCrossCovDecomposition (κ : ℝ) (Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  Hstar = Hbar + (κ ^ 2)⁻¹ • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ))

/-- **The refutation, with the shipped equation's unit conversion left unspecified.** The design
note flags, as a defect independent of the spurious term itself, that the `1/κ²` factor by which
the shipped equation converts `θ̃`-units to `φ`-units is asserted with no derivation. This lemma
removes that factor from the argument: whatever nonzero scalar `a` is chosen as the conversion,
the exact identity forces the symmetric part of the cross-covariance to be the zero matrix. Only
`a ≠ 0` is used, so no rederivation of the conversion factor rescues the equation. The shipped
form is the case `a = (κ²)⁻¹`, and the two lemmas below are that case. -/
theorem symmPart_eq_zero_of_scaledCrossCovDecomposition {a : ℝ} (ha : a ≠ 0)
    {Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar)
    (hship : Hstar = Hbar + a • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ))) :
    ((1 : ℝ) / 2) • (Sigma + Sigmaᵀ) = 0 := by
  rw [hexact] at hship
  have hzero : a • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ)) = 0 := left_eq_add.mp hship
  rcases smul_eq_zero.mp hzero with hc | hM
  · exact absurd hc ha
  · exact hM

/-- The trace form of the previous lemma: for every nonzero unit conversion `a`, the shipped
equation together with the exact identity forces `tr Σ_slack = 0`. -/
theorem trace_eq_zero_of_scaledCrossCovDecomposition {a : ℝ} (ha : a ≠ 0)
    {Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar)
    (hship : Hstar = Hbar + a • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ))) : Sigma.trace = 0 := by
  have hsym := symmPart_eq_zero_of_scaledCrossCovDecomposition ha hexact hship
  have htr := congrArg Matrix.trace hsym
  rwa [trace_symmetrization, Matrix.trace_zero] at htr

/-- **The refutation, trace form.** If the shipped decomposition held alongside the exact
identity `H⋆ = E_X[∇²_θ̃ L]` proved above, then the term it adds would have to be traceless:
`tr Σ_slack = 0`. Since the paper defines `σ_Jac² = tr Σ_slack` and reports it strictly positive,
the two cannot both hold. -/
theorem trace_eq_zero_of_hessianCrossCovDecomposition {κ : ℝ} (hκ : κ ≠ 0)
    {Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar)
    (hship : HessianCrossCovDecomposition κ Hstar Hbar Sigma) : Sigma.trace = 0 :=
  trace_eq_zero_of_scaledCrossCovDecomposition (inv_ne_zero (pow_ne_zero 2 hκ)) hexact hship

/-- **The refutation, matrix form — stronger than the trace form.** Under the same hypotheses,
the entire added term vanishes: the symmetric part `½(Σ + Σᵀ)` is the zero matrix. This closes
the anisotropy fallback of `A1_proofs.tex`, which hedges that "the added curvature is in general
anisotropic — a quadratic form aligned with the batch-coupled gradient noise" and that the trace
is only its dimension-averaged summary: no quadratic form at all is added, in any direction,
because the matrix carrying it is zero. `trace_eq_zero_of_hessianCrossCovDecomposition` is the
special case the paper measures. -/
theorem symmPart_eq_zero_of_hessianCrossCovDecomposition {κ : ℝ} (hκ : κ ≠ 0)
    {Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar)
    (hship : HessianCrossCovDecomposition κ Hstar Hbar Sigma) :
    ((1 : ℝ) / 2) • (Sigma + Sigmaᵀ) = 0 :=
  symmPart_eq_zero_of_scaledCrossCovDecomposition (inv_ne_zero (pow_ne_zero 2 hκ)) hexact hship

/-- **The refutation is sharp, and the equation it refutes is not an empty one.** Under the exact
identity the shipped equation holds *precisely* when the symmetric part of the cross-covariance
vanishes. Two things follow that a bare negation does not deliver. First, the shipped equation is
a satisfiable constraint, so the negative results below are statements about the cross-covariance
the paper measures rather than artefacts of an equation no matrix could satisfy. Second, the
refutation turns on the measured quantity and nothing else: what fails is `tr Σ_slack > 0`, not
the shape of the equation, the size of `κ`, or the dimension `d`. -/
theorem hessianCrossCovDecomposition_iff_symmPart_eq_zero {κ : ℝ} (hκ : κ ≠ 0)
    {Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar) :
    HessianCrossCovDecomposition κ Hstar Hbar Sigma ↔ ((1 : ℝ) / 2) • (Sigma + Sigmaᵀ) = 0 := by
  refine ⟨symmPart_eq_zero_of_hessianCrossCovDecomposition hκ hexact, fun hsym => ?_⟩
  rw [HessianCrossCovDecomposition, hexact, hsym, smul_zero, add_zero]

/-- A **nonzero cross-covariance that does satisfy the shipped equation**: the antisymmetric
`2 × 2` matrix `[[0, 1], [-1, 0]]`, in the reading `H⋆ = E_X[∇²_θ̃ L]` supplied by the exact
identity. Its symmetric part is zero, so by `hessianCrossCovDecomposition_iff_symmPart_eq_zero`
it is compatible with the exact identity, and its trace is zero, which is exactly the constraint
the refutation extracts. It is exhibited so that the negations below are visibly about the
measured positivity of `tr Σ_slack` rather than about an equation that nothing satisfies: an
antisymmetric cross-covariance would be consistent with the shipped decomposition, and the
paper's own measurement reports one that is not. -/
noncomputable def antisymmCrossCov : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

/-- The antisymmetric witness is a genuinely nonzero matrix. -/
theorem antisymmCrossCov_ne_zero : antisymmCrossCov ≠ 0 := by
  intro hzero
  have h : antisymmCrossCov 0 1 = (0 : Matrix (Fin 2) (Fin 2) ℝ) 0 1 := by rw [hzero]
  norm_num [antisymmCrossCov] at h

/-- The antisymmetric witness has zero trace, which is the constraint the refutation extracts. -/
theorem trace_antisymmCrossCov : antisymmCrossCov.trace = 0 := by
  simp [antisymmCrossCov, Matrix.trace_fin_two]

/-- **The shipped equation is satisfiable by a nonzero cross-covariance.** Together with
`antisymmCrossCov_ne_zero` this shows that `¬ HessianCrossCovDecomposition …` below is not
vacuously true of every `Σ`: the exact identity kills the shipped decomposition exactly for the
cross-covariances whose symmetric part does not vanish, and in particular for every one of
strictly positive trace. -/
theorem hessianCrossCovDecomposition_antisymmCrossCov {κ : ℝ} (hκ : κ ≠ 0)
    (H : Matrix (Fin 2) (Fin 2) ℝ) :
    HessianCrossCovDecomposition κ H H antisymmCrossCov := by
  rw [hessianCrossCovDecomposition_iff_symmPart_eq_zero hκ rfl]
  ext i j
  simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.transpose_apply, Matrix.zero_apply,
    smul_eq_mul]
  fin_cases i <;> fin_cases j <;> norm_num [antisymmCrossCov]

/-- **The refutation with an explicit remainder carried.** The surrounding lemma statement of
`A1_proofs.tex` hedges its conclusion with an `O(η σ_Jac²/(d κ²))` Itô–Taylor correction, so the
honest question is whether the shipped decomposition can hold *up to a remainder* `R`. It can,
but only at the cost of `tr Σ_slack = -κ² tr R` exactly: the cross-covariance contribution is
not a leading-order addition with a subleading correction, it is precisely the negative of the
correction. Nothing is assumed about `R` here. -/
theorem trace_of_hessianCrossCovDecomposition_remainder {κ : ℝ} (hκ : κ ≠ 0)
    {Hstar Hbar Sigma R : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar)
    (hship : Hstar = Hbar + (κ ^ 2)⁻¹ • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ)) + R) :
    Sigma.trace = -(κ ^ 2) * R.trace := by
  have hzero : (κ ^ 2)⁻¹ • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ)) + R = 0 := by
    rw [hexact, add_assoc] at hship
    exact left_eq_add.mp hship
  have htr : (κ ^ 2)⁻¹ * Sigma.trace + R.trace = 0 := by
    have := congrArg Matrix.trace hzero
    rwa [Matrix.trace_add, Matrix.trace_smul, trace_symmetrization, Matrix.trace_zero,
      smul_eq_mul] at this
  have hk2 : (κ : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 hκ
  field_simp at htr
  linarith

/-- The quantitative form: if the remainder in the shipped decomposition is bounded in trace by
`δ`, then the cross-covariance trace the paper calls `σ_Jac²` is bounded by `κ² δ`. So the
shipped equation cannot deliver a smoothing modulus of order `σ_Jac²/(d κ²)` unless `σ_Jac²`
is itself of remainder size, which is the opposite of the claim it is used to support. -/
theorem abs_trace_le_of_hessianCrossCovDecomposition_remainder {κ δ : ℝ} (hκ : κ ≠ 0)
    {Hstar Hbar Sigma R : Matrix (Fin d) (Fin d) ℝ} (hR : |R.trace| ≤ δ) (hexact : Hstar = Hbar)
    (hship : Hstar = Hbar + (κ ^ 2)⁻¹ • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ)) + R) :
    |Sigma.trace| ≤ κ ^ 2 * δ := by
  rw [trace_of_hessianCrossCovDecomposition_remainder hκ hexact hship, abs_mul, abs_neg,
    abs_of_nonneg (sq_nonneg κ)]
  exact mul_le_mul_of_nonneg_left hR (sq_nonneg κ)

/-- **The shipped decomposition across parameter blocks.** The paper's `H⋆ ≡ ∇²_φ L̃(φ⋆)` is a
`p × p` matrix in the body-parameter dimension `p`, while `E_X[∇²_θ̃ L]` and `Σ_slack` are `d × d`
matrices in the slack dimension `d`. The sum in the shipped equation therefore cannot be formed
without identifying the two index sets. This predicate carries that identification as the
explicit hypothesis `hpd : p = d` and reindexes `H⋆` through it, so that the assumption the
shipped proof leaves silent is visible in the Lean statement. It is a genuine assumption: it is
false whenever the body carries parameters beyond the slack block, which is the setting of the
paper. -/
def HessianCrossCovDecompositionAcrossBlocks (hpd : p = d) (κ : ℝ)
    (Hstar : Matrix (Fin p) (Fin p) ℝ) (Hbar Sigma : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  Matrix.reindex (finCongr hpd) (finCongr hpd) Hstar
    = Hbar + (κ ^ 2)⁻¹ • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ))

/-- The refutation survives the dimensional identification: even granting `p = d`, the shipped
decomposition together with the exact identity forces `tr Σ_slack = 0`. The dimensional defect
and the analytic defect are independent; repairing the first does not repair the second. -/
theorem trace_eq_zero_of_hessianCrossCovDecompositionAcrossBlocks (hpd : p = d) {κ : ℝ}
    (hκ : κ ≠ 0) {Hstar : Matrix (Fin p) (Fin p) ℝ} {Hbar Sigma : Matrix (Fin d) (Fin d) ℝ}
    (hexact : Matrix.reindex (finCongr hpd) (finCongr hpd) Hstar = Hbar)
    (hship : HessianCrossCovDecompositionAcrossBlocks hpd κ Hstar Hbar Sigma) :
    Sigma.trace = 0 :=
  trace_eq_zero_of_hessianCrossCovDecomposition hκ hexact hship

/-- A **realizable slack-channel cross-covariance with strictly positive trace**: the rank-one
outer product `δθ̃ (δg)ᵀ` of a single batch draw on which the iterate fluctuation and the gradient
fluctuation are both the `j`-th coordinate direction. This is the shape of `eq:cross-cov`, not an
artificial matrix. -/
noncomputable def rankOneCrossCov (j : Fin d) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.vecMulVec (Pi.single j 1) (Pi.single j 1)

/-- The witness is a value of the estimator *function* of `eq:cross-cov`: feeding the
trailing-window estimator of `CrossCov.lean` the constant fluctuation series `δθ̃ = δg = e_j` at
window length one returns exactly `rankOneCrossCov j`. This is an algebraic evaluation, not a
run of the measurement protocol: the protocol centres each window at its own mean (`eq:fluct`),
and a centred window of length one is degenerate — that is
`crossCovEstimator_one_bodyFluct_eq_zero` next. The protocol-faithful realization, at window
length two with the centring derived rather than assumed, is `rankOneCrossCov_realizable`
below. -/
theorem rankOneCrossCov_eq_crossCovEstimator (j : Fin d) (t : ℕ) :
    crossCovEstimator 1 t (fun _ => Pi.single j 1) (fun _ => Pi.single j 1)
      = rankOneCrossCov j := by
  simp [crossCovEstimator, rankOneCrossCov]

/-- **A window of length one cannot realize a nonzero estimate.** The window-centred fluctuation
of `eq:fluct` subtracts the window mean, and over a window of length one the mean is the sample
itself, so the measurement protocol's estimator at `T = 1` is identically the zero matrix — for
every body, every batch sequence, every gradient series and every iteration. This is why the
realization below uses a window of length two, and why the evaluation above is a statement about
the estimator function rather than about the protocol. -/
theorem crossCovEstimator_one_bodyFluct_eq_zero {Ω : Type*}
    (h : Ω → Fin d → ℝ) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ) (t : ℕ) :
    crossCovEstimator 1 t (bodyFluct h X 1 t) dg = 0 := by
  ext i k
  simp [crossCovEstimator, bodyFluct]

/-- **The witness is realizable by the measurement protocol itself.** Take the window of length
two ending at `t = 2` and let the body's latent weight on the `s`-th draw be `2s • e_j`, the gradient
fluctuation riding the same draws — the paper's same-batch protocol. The window-centred
fluctuations of `eq:fluct` are computed by `bodyFluct` of `CrossCov.lean` — derived from the
values of the latent weight, not assumed — and come out to `∓ e_j` on the two draws; the trailing-window estimator
of `eq:cross-cov` then returns exactly `rankOneCrossCov j`. Together with
`trace_rankOneCrossCov`, a genuinely realizable run of the estimator produces a cross-covariance
of strictly positive trace, to which `not_hessianCrossCovDecomposition_of_trace_pos` applies. -/
theorem rankOneCrossCov_realizable (j : Fin d) :
    crossCovEstimator 2 2
      (bodyFluct (fun n : ℕ => (2 * (n : ℝ)) • Pi.single j 1) id 2 2)
      (bodyFluct (fun n : ℕ => (2 * (n : ℝ)) • Pi.single j 1) id 2 2)
      = rankOneCrossCov j := by
  ext i k
  simp [crossCovEstimator, bodyFluct, rankOneCrossCov, Finset.sum_range_succ,
    Matrix.vecMulVec_apply, Matrix.smul_apply, smul_eq_mul]
  ring

/-- The witness has trace one, hence strictly positive trace: `σ_Jac² = 1 > 0`. -/
theorem trace_rankOneCrossCov (j : Fin d) : (rankOneCrossCov j).trace = 1 := by
  simp [rankOneCrossCov, Matrix.trace_vecMulVec]

/-- **No cross-covariance of positive trace satisfies the shipped decomposition**, for any
positivity-map scale `κ ≠ 0`, once the exact identity is in force. The design note records
`tr Σ_slack > 0` at 8997 of 9000 measured iterations, so this is the empirically operative
case. -/
theorem not_hessianCrossCovDecomposition_of_trace_pos {κ : ℝ} (hκ : κ ≠ 0)
    {Hstar Hbar Sigma : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar)
    (hpos : 0 < Sigma.trace) : ¬ HessianCrossCovDecomposition κ Hstar Hbar Sigma := by
  intro hship
  exact absurd (trace_eq_zero_of_hessianCrossCovDecomposition hκ hexact hship) (ne_of_gt hpos)

/-- The concrete instance: the rank-one single-sample cross-covariance of `rankOneCrossCov`
cannot appear in the shipped decomposition, for any `κ ≠ 0`. -/
theorem not_hessianCrossCovDecomposition_rankOne {κ : ℝ} (hκ : κ ≠ 0) (j : Fin d)
    {Hstar Hbar : Matrix (Fin d) (Fin d) ℝ} (hexact : Hstar = Hbar) :
    ¬ HessianCrossCovDecomposition κ Hstar Hbar (rankOneCrossCov j) :=
  not_hessianCrossCovDecomposition_of_trace_pos hκ hexact
    (by rw [trace_rankOneCrossCov]; norm_num)

variable {ι : Type*} [Fintype ι]

/-- **The refutation, applied to the genuine pullback Hessian.** Take the actual pullback
landscape `L̃(b) = ∑ᵢ wᵢ (L ∘ ψ)(b + hᵢ)` of a finite batch law, its actual `d × d` Hessian in the
slack block, and the actual batch average of the per-batch loss Hessians. Then no
cross-covariance of strictly positive trace can relate them the way the shipped decomposition of
`A1_proofs.tex` asserts, for any positivity-map scale `κ ≠ 0`. The only hypotheses are that each
per-batch composite loss is differentiable everywhere with a second derivative at its shifted
point; nothing about the batch law, the positivity map, the learning rate or the invariant measure is
assumed, and no remainder is discarded. -/
theorem pullbackHessian_not_hessianCrossCovDecomposition
    {w : ι → ℝ} {f : ι → (Fin d → ℝ) → ℝ}
    {f1 : ι → (Fin d → ℝ) → ((Fin d → ℝ) →L[ℝ] ℝ)}
    {f2 : ι → ((Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] ℝ)}
    {h : ι → (Fin d → ℝ)} {b : Fin d → ℝ}
    (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i))
    {κ : ℝ} (hκ : κ ≠ 0) {Sigma : Matrix (Fin d) (Fin d) ℝ} (hpos : 0 < Sigma.trace) :
    ¬ HessianCrossCovDecomposition κ (hessianMatrix (pullbackLoss w f h) b)
        (∑ i, w i • hessianMatrix (f i) (b + h i)) Sigma :=
  not_hessianCrossCovDecomposition_of_trace_pos hκ (hessianMatrix_pullbackLoss hf1 hf2) hpos

/-- **The same, with the shipped equation's unit conversion left free.** The `1/κ²` factor of the
shipped equation is asserted without derivation, so the refutation should not depend on it. It
does not: for the genuine pullback Hessian of a genuine pullback landscape, no additive
symmetrized cross-covariance of strictly positive trace can appear under *any* nonzero scalar
conversion whatsoever. Repairing the undelivered unit conversion therefore cannot repair the
equation. -/
theorem pullbackHessian_not_scaledCrossCovDecomposition
    {w : ι → ℝ} {f : ι → (Fin d → ℝ) → ℝ}
    {f1 : ι → (Fin d → ℝ) → ((Fin d → ℝ) →L[ℝ] ℝ)}
    {f2 : ι → ((Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] ℝ)}
    {h : ι → (Fin d → ℝ)} {b : Fin d → ℝ}
    (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i))
    {a : ℝ} (ha : a ≠ 0) {Sigma : Matrix (Fin d) (Fin d) ℝ} (hpos : 0 < Sigma.trace) :
    ¬ (hessianMatrix (pullbackLoss w f h) b
        = (∑ i, w i • hessianMatrix (f i) (b + h i))
            + a • (((1 : ℝ) / 2) • (Sigma + Sigmaᵀ))) := by
  intro hship
  exact absurd (trace_eq_zero_of_scaledCrossCovDecomposition ha
    (hessianMatrix_pullbackLoss hf1 hf2) hship) (ne_of_gt hpos)

/-- **The displayed conclusion of `lem:moreau` fails as well, not only its proof.** The lemma's
statement in `A1_proofs.tex` is the trace inequality `tr H⋆ ≥ tr E_X[∇²_θ̃ L] + d μ_eff` with
`μ_eff = Θ(σ_Jac²/(d κ²))` and `σ_Jac² = tr Σ` as `eq:cross-cov` defines it — an added term
that is some *unspecified positive multiple* of `σ_Jac²/κ²`, further hedged by an
`O(η σ_Jac²/(d κ²))` Itô–Taylor correction whose subtraction only shifts that multiple. A
defender of the shipped lemma might concede the undelivered decomposition equation and retreat
to this weaker displayed inequality; it does not survive either, and the refutation quantifies
over the hedge: the inequality fails for **every** coefficient `c > 0` on `σ_Jac²/κ²`, so no
choice of the `Θ(·)` constant and no subleading correction rescues it. The proof's own
extraction `d μ_eff = σ_Jac²/κ²` is the case `c = 1`. For the genuine pullback Hessian the two
traces are equal by the exact identity, so the inequality demands `c · tr Σ / κ² ≤ 0`, which
fails for every cross-covariance of strictly positive trace and every positivity-map scale `κ ≠ 0`. -/
theorem pullbackHessian_not_trace_lower_bound
    {w : ι → ℝ} {f : ι → (Fin d → ℝ) → ℝ}
    {f1 : ι → (Fin d → ℝ) → ((Fin d → ℝ) →L[ℝ] ℝ)}
    {f2 : ι → ((Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] ℝ)}
    {h : ι → (Fin d → ℝ)} {b : Fin d → ℝ}
    (hf1 : ∀ i y, HasFDerivAt (f i) (f1 i y) y)
    (hf2 : ∀ i, HasFDerivAt (f1 i) (f2 i) (b + h i))
    {c κ : ℝ} (hc : 0 < c) (hκ : κ ≠ 0) {Sigma : Matrix (Fin d) (Fin d) ℝ}
    (hpos : 0 < Sigma.trace) :
    ¬ ((∑ i, w i • hessianMatrix (f i) (b + h i)).trace + c * (Sigma.trace / κ ^ 2)
        ≤ (hessianMatrix (pullbackLoss w f h) b).trace) := by
  intro habs
  have hbar : (∑ i, w i • hessianMatrix (f i) (b + h i)).trace
      = ∑ i, w i * (hessianMatrix (f i) (b + h i)).trace := by
    rw [Matrix.trace_sum]
    exact Finset.sum_congr rfl fun i _ => by rw [Matrix.trace_smul]; rfl
  have hexact := trace_hessianMatrix_pullbackLoss hf1 hf2 (w := w)
  have hκ2 : 0 < κ ^ 2 := lt_of_le_of_ne (sq_nonneg κ) (Ne.symm (pow_ne_zero 2 hκ))
  have hquot : 0 < c * (Sigma.trace / κ ^ 2) := mul_pos hc (div_pos hpos hκ2)
  rw [hbar, hexact] at habs
  linarith

/-- **The refutation without the translation structure.** The same conclusion for an arbitrary
finite batch average `x ↦ ∑ᵢ wᵢ Fᵢ(x)`, whose Hessian is the batch average of the per-sample
Hessians: no additive cross-covariance term of positive trace can appear, whatever the batch
index does inside `Fᵢ`. This forecloses the `φ`-reading of the shipped decomposition in the form
in which its first term is the batch average of the per-sample curvatures in the same variable
that is being differentiated. It does not address the variant `φ`-reading in which that first
term is kept as the `θ̃`-curvature `E_X[∇²_θ̃ L]`: there the shipped equation additionally omits
the chain rule through the body Jacobian, and this file does not quantify that omission. On the
slack block — where `lem:moreau` states its inequality, and where (A3) makes the body Jacobian
the identity — no such gap exists, and `pullbackHessian_not_hessianCrossCovDecomposition` is
decisive. -/
theorem batchAverageHessian_not_hessianCrossCovDecomposition
    {w : ι → ℝ} {F : ι → (Fin d → ℝ) → ℝ}
    {F1 : ι → (Fin d → ℝ) → ((Fin d → ℝ) →L[ℝ] ℝ)}
    {F2 : ι → ((Fin d → ℝ) →L[ℝ] (Fin d → ℝ) →L[ℝ] ℝ)} {c : Fin d → ℝ}
    (hF1 : ∀ i y, HasFDerivAt (F i) (F1 i y) y) (hF2 : ∀ i, HasFDerivAt (F1 i) (F2 i) c)
    {κ : ℝ} (hκ : κ ≠ 0) {Sigma : Matrix (Fin d) (Fin d) ℝ} (hpos : 0 < Sigma.trace) :
    ¬ HessianCrossCovDecomposition κ (hessianMatrix (fun x => ∑ i, w i * F i x) c)
        (∑ i, w i • hessianMatrix (F i) c) Sigma :=
  not_hessianCrossCovDecomposition_of_trace_pos hκ (hessianMatrix_batchAverage hF1 hF2) hpos

/-- The first Fréchet derivative of the concrete quadratic `x ↦ (x j)²`, named so that the closed
instances below can supply it as the `f1` of the general theorems. -/
private noncomputable def sqLossD1 (j : Fin d) (y : Fin d → ℝ) : (Fin d → ℝ) →L[ℝ] ℝ :=
  y j • (ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ)
    + y j • (ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ)

/-- The (constant) second Fréchet derivative of the concrete quadratic `x ↦ (x j)²`. -/
private noncomputable def sqLossD2 (j : Fin d) : (Fin d → ℝ) →L[ℝ] ((Fin d → ℝ) →L[ℝ] ℝ) :=
  (ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ).smulRight
      (ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ)
    + (ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ).smulRight
      (ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ)

private theorem hasFDerivAt_sqLoss (j : Fin d) (y : Fin d → ℝ) :
    HasFDerivAt (fun x : Fin d → ℝ => x j * x j) (sqLossD1 j y) y :=
  ((ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ).hasFDerivAt).mul
    ((ContinuousLinearMap.proj j : (Fin d → ℝ) →L[ℝ] ℝ).hasFDerivAt)

private theorem hasFDerivAt_sqLossD1 (j : Fin d) (y : Fin d → ℝ) :
    HasFDerivAt (sqLossD1 j) (sqLossD2 j) y := by
  have hcoe : sqLossD1 j = ⇑(sqLossD2 j) := by
    funext z
    ext v
    simp [sqLossD1, sqLossD2, ContinuousLinearMap.smulRight_apply]
  rw [hcoe]
  exact (sqLossD2 j).hasFDerivAt

/-- The Hessian matrix of the concrete quadratic per-batch loss `x ↦ (x j)²`, computed outright:
twice the single-entry matrix `E_jj`, which is `2 • rankOneCrossCov j`. First derivative
`y ↦ 2 yⱼ • projⱼ`, second derivative the constant bilinear map `2 projⱼ ⊗ projⱼ`; nothing is
hypothesized. This is the loss that discharges every hypothesis of the closed instance below
with genuinely nonzero curvature. -/
theorem hessianMatrix_sq (j : Fin d) (c : Fin d → ℝ) :
    hessianMatrix (fun x : Fin d → ℝ => x j * x j) c = (2 : ℝ) • rankOneCrossCov j := by
  set P : (Fin d → ℝ) →L[ℝ] ℝ := ContinuousLinearMap.proj j with hP
  set L : (Fin d → ℝ) →L[ℝ] ((Fin d → ℝ) →L[ℝ] ℝ) := P.smulRight P + P.smulRight P with hL
  have hproj : ∀ y : Fin d → ℝ, HasFDerivAt (fun x : Fin d → ℝ => x j) P y :=
    fun _ => P.hasFDerivAt
  have hf1 : ∀ y : Fin d → ℝ,
      HasFDerivAt (fun x : Fin d → ℝ => x j * x j) (y j • P + y j • P) y :=
    fun y => (hproj y).mul (hproj y)
  have hcoe : (fun y : Fin d → ℝ => y j • P + y j • P) = ⇑L := by
    funext y
    ext v
    simp [hL, hP, ContinuousLinearMap.smulRight_apply]
  have hfd1 : fderiv ℝ (fun x : Fin d → ℝ => x j * x j) = fun y => y j • P + y j • P :=
    funext fun y => (hf1 y).fderiv
  have hfd2 : fderiv ℝ (fderiv ℝ (fun x : Fin d → ℝ => x j * x j)) c = L := by
    rw [hfd1, hcoe]
    exact L.hasFDerivAt.fderiv
  ext a b'
  simp only [hessianMatrix, Matrix.of_apply, hfd2, hL, _root_.add_apply,
    ContinuousLinearMap.smulRight_apply, _root_.smul_apply, hP,
    ContinuousLinearMap.proj_apply, Matrix.smul_apply, rankOneCrossCov,
    Matrix.vecMulVec_apply, smul_eq_mul, Pi.single_apply]
  by_cases haj : a = j <;> by_cases hbj : b' = j <;> simp [haj, hbj] <;>
    first
    | exact fun hh => hbj hh.symm
    | exact fun hh => haj hh.symm
    | exact fun _ hh => haj hh.symm
    | norm_num

/-- **The pullback Hessian of the concrete quadratic, evaluated.** For the per-batch composite
loss `x ↦ (x j)²` and a batch law whose weights sum to one, the pullback landscape has the
genuinely nonzero Hessian `2 E_jj`, whatever the body outputs `hᵢ` and the base point `b` are.
This is what makes the closed instances below configurations with real curvature rather than
formally correct but degenerate ones: the statements of the general refutations leave the two
Hessian arguments unevaluated, and with an empty batch family or zero weights they would both
collapse to the zero matrix. -/
theorem hessianMatrix_pullbackLoss_sq {w : ι → ℝ} {h : ι → (Fin d → ℝ)} (b : Fin d → ℝ)
    (j : Fin d) (hw : ∑ i, w i = 1) :
    hessianMatrix (pullbackLoss w (fun (_ : ι) (x : Fin d → ℝ) => x j * x j) h) b
      = (2 : ℝ) • rankOneCrossCov j :=
  hessianMatrix_pullbackLoss_of_const (w := w) (f := fun (_ : ι) (x : Fin d → ℝ) => x j * x j)
    (f1 := fun _ => sqLossD1 j) (f2 := fun _ => sqLossD2 j) (h := h) (b := b)
    (fun _ y => hasFDerivAt_sqLoss j y) (fun i => hasFDerivAt_sqLossD1 j (b + h i)) hw
    (fun i => hessianMatrix_sq j (b + h i))

/-- The trace of that Hessian is `2`, not `0`: the closed instances below are not statements
about a flat landscape. -/
theorem trace_hessianMatrix_pullbackLoss_sq {w : ι → ℝ} {h : ι → (Fin d → ℝ)} (b : Fin d → ℝ)
    (j : Fin d) (hw : ∑ i, w i = 1) :
    (hessianMatrix (pullbackLoss w (fun (_ : ι) (x : Fin d → ℝ) => x j * x j) h) b).trace = 2 := by
  rw [hessianMatrix_pullbackLoss_sq b j hw, Matrix.trace_smul, trace_rankOneCrossCov,
    smul_eq_mul, mul_one]

/-- **A closed instance: the hypothesis package of the headline refutation is satisfiable, and
here discharged.** For the per-batch composite loss `x ↦ (x j)²` — any finite batch family, any
weights, any body outputs `hᵢ`, any base point `b` — the differentiability hypotheses of
`pullbackHessian_not_hessianCrossCovDecomposition` are proved rather than assumed, and the
witness cross-covariance is `rankOneCrossCov j` of trace one, realizable by
`rankOneCrossCov_realizable`. The refutation is therefore not vacuous on the side of its
hypotheses. The two Hessian arguments are left in the unevaluated form the general theorem
states them in; `hessianMatrix_pullbackLoss_sq` evaluates them, and shows that as soon as the
weights sum to one — the case the paper is about — the curvature they carry is the nonzero
`2 E_jj`. -/
theorem quadraticPullback_not_hessianCrossCovDecomposition
    {w : ι → ℝ} {h : ι → (Fin d → ℝ)} (b : Fin d → ℝ) (j : Fin d)
    {κ : ℝ} (hκ : κ ≠ 0) :
    ¬ HessianCrossCovDecomposition κ
        (hessianMatrix (pullbackLoss w (fun _ x => x j * x j) h) b)
        (∑ i, w i • hessianMatrix (fun x : Fin d → ℝ => x j * x j) (b + h i))
        (rankOneCrossCov j) :=
  pullbackHessian_not_hessianCrossCovDecomposition
    (f1 := fun _ => sqLossD1 j) (f2 := fun _ => sqLossD2 j)
    (fun _ y => hasFDerivAt_sqLoss j y) (fun i => hasFDerivAt_sqLossD1 j (b + h i)) hκ
    (by rw [trace_rankOneCrossCov]; norm_num)

/-- **A closed instance of the displayed-inequality refutation as well.** The same concrete
quadratic discharges every hypothesis of `pullbackHessian_not_trace_lower_bound`: for the
rank-one cross-covariance of trace one, the trace inequality of `lem:moreau` fails for every
positive coefficient `c` on `σ_Jac²/κ²` and every positivity-map scale `κ ≠ 0`, with no hypothesis left
standing. So the displayed conclusion of the lemma, and not only its proof, is refuted by an
instance that exists. -/
theorem quadraticPullback_not_trace_lower_bound
    {w : ι → ℝ} {h : ι → (Fin d → ℝ)} (b : Fin d → ℝ) (j : Fin d)
    {c κ : ℝ} (hc : 0 < c) (hκ : κ ≠ 0) :
    ¬ ((∑ i, w i • hessianMatrix (fun x : Fin d → ℝ => x j * x j) (b + h i)).trace
        + c * ((rankOneCrossCov j).trace / κ ^ 2)
        ≤ (hessianMatrix (pullbackLoss w (fun (_ : ι) (x : Fin d → ℝ) => x j * x j) h) b).trace) :=
  pullbackHessian_not_trace_lower_bound
    (f1 := fun _ => sqLossD1 j) (f2 := fun _ => sqLossD2 j)
    (fun _ y => hasFDerivAt_sqLoss j y) (fun i => hasFDerivAt_sqLossD1 j (b + h i)) hc hκ
    (by rw [trace_rankOneCrossCov]; norm_num)

end Refutation

end IcnnLift
