import Mathlib

/-!
# The exact update-covariance decomposition and the projected effective diffusion

This file machine-checks fact (ii) of `docs/design/lemma1_rederivation.md` §2 (lines 50-87), the
diffusion half of the corrected mechanism. Section 1 of that note concedes that the shipped
Hessian decomposition of Lemma 1 is false — differentiation commutes with the batch expectation,
so the pullback Hessian carries no cross-covariance term — and relocates the cross-covariance
from the Hessian of the averaged loss to the *covariance of the per-step update*. What is proved
here is that relocation, in the only form in which it is a theorem rather than an approximation.

The note expands the slack update around the mean iterate. Writing `dθ(X) = h(X) - E[h]` for the
latent-weight fluctuation, `g(X)` for the loss gradient at the mean iterate, `H` for the loss Hessian
there and `s = ψ'(θ̄)` for the single-prefactor idealization (A2),

    G(b; X) = s g(X) + s H dθ(X) + R(X),

and the claim is that the update covariance splits into an attenuated objective-noise term, a
cross term carrying the slack-channel cross-covariance, and a term quadratic in the latent weight
jitter. The essential point, and the reason this file exists, is that **the split is an exact
identity**: the covariance of a sum is the sum of the covariances plus the cross-covariances,
with nothing discarded. Everything the note writes as `+ h.o.t.` is the contribution of `R`,
and it is carried here, term by term, into the conclusion of `updateCov_decomposition`, and then
bounded — never dropped — in `projDiffusion_remainder_bound`.

What mathlib provides for vector-valued variables is `ProbabilityTheory.covarianceBilin`, the
covariance bilinear form of a single *measure* on a Hilbert space: the law is fixed once, and the
bilinearity is in the two direction vectors. The decomposition needs a different object — the
cross-covariance matrix of two *distinct* random vectors over an abstract sample space, bilinear
in the random arguments and composable with the curvature through matrix products — and that
object mathlib does not have. So `covMat μ U W = E[(U - EU)(W - EW)ᵀ]` is defined entrywise from
the scalar `ProbabilityTheory.covariance`, and its bilinearity is proved here (`covMat_add_left`,
`covMat_add_right`, `covMat_smul_left`, `covMat_smul_right`, `covMat_mulVec_left`,
`covMat_mulVec_right`). The nine-term expansion of a three-summand covariance,
`covMat_add_three_self`, is the exact identity the note's display abbreviates.

Contracting with a direction `e_b` (`projDiffusion`) turns the matrix identity into the note's
scalar formula

    σ_eff²(b) = s² σ_obj² + 2 s² e_bᵀ E[δg δθᵀ] H e_b + s² e_bᵀ H V H e_b,   V = Cov[dθ],

which is `projDiffusion_decomposition`. That statement is the remainder-free one, stated for an
update satisfying `G = s g + s H dθ` exactly, because those three terms are what the note's
display names; it is not, however, the general one. The projected identity holds with the
remainder carried and nothing at all assumed about it
(`projDiffusion_decomposition_with_remainder`), where projection collapses the five matrix
contributions of `R` to three scalars and no symmetry of `H` is needed either, so the scalar
statement is exact wherever the matrix statement is. `projDiffusion_remainder_bound` then bounds
those three scalars rather than discarding them, and is the statement to read when `R ≠ 0`.

Note the dimensional point the note makes: here `σ_Jac²`-type quantities enter **multiplied by the
curvature** — `E[δg δθᵀ] H` and `H V H` — so all three terms carry the units of a squared
gradient, whereas the shipped `σ_eff² = s² σ_obj² + σ_Jac²` adds incommensurables. The corrected
decomposition is dimensionally homogeneous because the curvature factors are derived rather than
inserted.

The direct baseline is the same theorem at `dθ = 0`: both the cross and the quadratic term are
covariances with a constant, hence vanish, and `σ_eff² = s² σ_obj²`
(`projDiffusion_direct_baseline`). The decomposition alone does **not** establish the structural
asymmetry itself — that the lift's projected diffusion is at least the baseline's — and this file
does not claim it. What the decomposition does is reduce that claim to the sign of a single term
and to nothing else: the jitter term `s² e_bᵀ H V H e_b` is non-negative whatever the coupling,
needing only that `H` be symmetric (`projDiffusion_jitter_term_nonneg`, because it is the
variance of `⟨H e_b, dθ⟩`), so the asymmetry follows as soon as the cross term is non-negative,
which is `projDiffusion_ge_baseline_of_cross_nonneg`. Whether the cross term is in fact
non-negative is (A4)'s question, and in general it belongs to `CouplingSign.lean`, not here. The
one case the note settles by inspection is settled here by a theorem instead: under the
forward-KL chain taken exactly, `g = c + H dθ`, the cross form *equals* the jitter form and is
therefore non-negative (`covMat_of_gradient_chain`,
`projDiffusion_cross_nonneg_of_gradient_chain`), so the asymmetry holds outright
(`projDiffusion_ge_baseline_of_gradient_chain`). The note's remainder clause `r ≠ 0`, which is
what (A4) must actually control, is untouched here.

Two further points of exactness are settled by theorems rather than by this prose. First, the
expansion hypothesis `hG` restricts nothing: `updateCov_decomposition_of_sqIntegrable` produces
the remainder for an arbitrary square-integrable update, so the decomposition applies to every
`G`, whatever `s`, `H`, `g` and `dθ` one chooses to name. Second, the note's Hessian `H(X)` is
random, and its reduction to the mean Hessian is not an approximation but a redefinition of the
remainder; `updateCov_decomposition_randomHessian` and
`projDiffusion_randomHessian_remainder_bound` state the random-Hessian results outright, with the
`δH` pairing counted inside the remainder that `ε` must control.

## Results
* `SqIntegrableVec` — componentwise square-integrability of a random vector, with closure under
  sums (`SqIntegrableVec.add`), differences (`SqIntegrableVec.sub`), scalar multiples
  (`SqIntegrableVec.smul`), deterministic linear images (`SqIntegrableVec.mulVec`) and scalar
  projections (`SqIntegrableVec.projection`).
* `covMat` — the covariance matrix `E[(U - EU)(W - EW)ᵀ]` of two square-integrable random
  vectors, with `covMat_apply`, `covMat_transpose` and `covMat_zero_left`/`covMat_zero_right`.
* `covMat_add_left`, `covMat_add_right`, `covMat_smul_left`, `covMat_smul_right` — bilinearity.
* `covMat_const_add_left` — a deterministic shift of either argument leaves the covariance matrix
  unchanged.
* `covMat_mulVec_left`, `covMat_mulVec_right` — `Cov[M U, W] = M Cov[U, W]` and
  `Cov[U, M W] = Cov[U, W] Mᵀ` for a deterministic matrix `M`.
* `covMat_add_self`, `covMat_add_three_self` — the exact expansion of the covariance of a sum of
  two, respectively three, square-integrable random vectors.
* `updateExpansion_absorb_hessian_fluctuation` — a random Hessian `H(X)` is reduced to a constant
  one by absorbing `(H(X) - H) dθ(X)` into the remainder, exactly as the note's `T3` does.
* `updateCov_decomposition` — the exact decomposition of `Cov[G]` for `G = s g + s H dθ + R`,
  with every term contributed by `R` displayed.
* `updateCov_decomposition_of_sqIntegrable` — the same decomposition for an *arbitrary*
  square-integrable `G`, with the remainder constructed rather than hypothesized. This is the
  proof that the expansion hypothesis `hG` is not a restriction.
* `absorbedRemainder`, `updateCov_decomposition_randomHessian` — the decomposition for a random
  Hessian `H(X)`, with the `δH` pairing counted inside the remainder.
* `updateCov_decomposition_of_remainder_zero` — the three-term form of the note's display.
* `projDiffusion` — the projected effective diffusion `σ_eff²(b) = e_bᵀ Cov[G] e_b`, with the
  linearity of the quadratic form in `quadForm_add`, `quadForm_smul` and `quadForm_transpose`.
* `covMat_quadForm_eq_covariance` — the projection identity `eᵀ Cov[U, W] e = cov[⟨e,U⟩, ⟨e,W⟩]`.
* `projDiffusion_decomposition` — the note's scalar `σ_eff²` formula, each term identified, for a
  remainder-free update.
* `projDiffusion_decomposition_with_remainder` — the same scalar formula for an update carrying an
  arbitrary square-integrable remainder, with the three scalars the remainder contributes
  displayed and with no hypothesis on `H`. This is the projection of the full matrix identity, and
  it is exact; `projDiffusion_decomposition_with_remainder_symmetric` writes it with `Hᵀ` folded
  into `H`.
* `projDiffusion_direct_baseline` — at `dθ = 0`, `σ_eff² = s² σ_obj²`, with no symmetry assumed
  of `H`.
* `projDiffusion_nonneg` — the note's `Cov_X[G] ≥ 0`, in projected form.
* `quadForm_conj_eq_quadForm_mulVec`, `projDiffusion_jitter_term_nonneg` — for symmetric `H` the
  jitter term is the variance of `⟨H e_b, dθ⟩`, hence non-negative with no assumption on the
  coupling.
* `projDiffusion_ge_baseline_of_cross_nonneg` — the lift/direct asymmetry, and exactly the input
  it needs: a non-negative cross term. The sign in general is (A4)'s, and is not proved here.
* `covMat_of_gradient_chain`, `projDiffusion_cross_nonneg_of_gradient_chain`,
  `projDiffusion_ge_baseline_of_gradient_chain` — under the note's forward-KL chain taken exactly,
  `g = c + H dθ`, one has `Cov[g, dθ] = H V`, the cross form equals the jitter form and is
  therefore non-negative, and the asymmetry follows with no sign hypothesis left over. The note's
  remainder clause `r ≠ 0` is not covered.
* `abs_covariance_le_sqrt_variance_mul_sqrt_variance` — Cauchy-Schwarz for
  `ProbabilityTheory.covariance`, which mathlib does not provide.
* `abs_variance_add_sub_variance_le` — the L² perturbation bound behind the remainder estimate.
* `variance_projection_le` — the projected remainder's variance is at most `ε²` when the
  remainder's second moment is and the projection direction is a unit vector.
* `projDiffusion_remainder_bound` — if `E[|R|²] ≤ ε²` and `e_b ⬝ᵥ e_b = 1` then the projected
  diffusion differs from the three named terms by at most `2 ε √(named) + ε²`. This is the honest
  content of `+ h.o.t.`.
* `projDiffusion_randomHessian_remainder_bound` — the same bound for a random Hessian, with `ε`
  required to control `s (H(X) - H) dθ(X)` as well as the Taylor remainder.
* `stdGaussianWitness_*` — a one-dimensional instance in which all three named terms are nonzero
  and the decomposition is checked against a directly computed `projDiffusion`. Its purpose is to
  rule out vacuity and to make the factor `2` falsifiable:
  `stdGaussianWitness_factor_two_necessary` shows the same identity with the factor `2` deleted is
  false in that instance.
* `planeWitness_*` — a two-dimensional instance with distinct, non-parallel gradient and latent weight
  directions and prefactor `2`, in which the three covariance matrices differ, the
  cross-covariance is not symmetric, and the decomposition again matches a directly computed
  `projDiffusion`. It is what makes three further mis-statements falsifiable, and they are refuted
  outright: `planeWitness_prefactor_squared_necessary` (carrying `s` in place of `s²`),
  `planeWitness_curvature_side_necessary` (the curvature multiplying the cross-covariance from the
  left) and `planeWitness_cross_covariance_necessary` (either variance in place of the
  cross-covariance). The one-dimensional instance can refute none of the three.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

Six inputs are carried as explicit hypotheses rather than proved, and each is genuinely outside
this file.

1. `hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + R ω`. This is the Taylor expansion of the update
   map itself. It is not an approximation in the Lean statement, and that is a theorem rather than
   a promise: `updateCov_decomposition_of_sqIntegrable` constructs an `R` satisfying it for every
   square-integrable `G`, so `hG` excludes no update and assumes nothing away. What is not proved
   here is that `R` is *small*, which is a statement about the second derivative of `L ∘ ψ` and
   about `ψ''`, belonging to the regularity assumptions (A1)-(A2) and to a second-order Taylor
   argument this file does not contain. To discharge it one would apply
   `taylor_mean_remainder_lagrange` coordinatewise to `θ ↦ ψ'(θ) ∇L(ψ(θ); X)` at `θ̄`, which
   requires the `C²` hypotheses of (A1). One further point of shape: `hG` is a pointwise
   identity, demanded at every `ω` rather than `μ`-almost everywhere. That is the form the note
   writes and the form a coordinatewise Taylor theorem delivers, so nothing is conceded against
   the paper; it does mean the theorems below ask for the expansion everywhere, even though the
   covariances in their conclusions cannot distinguish functions equal almost everywhere, so an
   almost-everywhere variant would be a routine generalization rather than new content.
2. `hRε : ∫ ω, R ω ⬝ᵥ R ω ∂μ ≤ ε ^ 2`, the second-moment bound on that remainder, carried with
   `hε : 0 ≤ ε`. It is the quantitative form of (A1)'s moment bounds; supplying it is the same
   analytic task as 1. For a random Hessian it must additionally dominate the `δH` pairing
   `s (H(X) - H) dθ(X)`, which is what
   `projDiffusion_randomHessian_remainder_bound` makes explicit rather than leaving to prose.
3. `hH : Hᵀ = H`, symmetry of the loss Hessian at the mean iterate. This is Clairaut's theorem for
   `L ∘ ψ`, available from `ContDiff` regularity; it is a hypothesis here only because this file
   never constructs `H` from a loss. It is carried exactly where a displayed form folds `Hᵀ` into
   `H`: the three-term display, the remainder-free scalar decomposition, the symmetric form of the
   remainder-carrying one, the remainder bounds stated against that scalar form, the two
   comparisons, and the jitter-term sign. That this is all it does is itself a theorem rather than
   a claim: `projDiffusion_decomposition_with_remainder` is the projected identity with the
   factors `2` in place and no hypothesis whatever on `H`, and
   `projDiffusion_decomposition_with_remainder_symmetric` is that theorem followed by one rewrite
   of `Hᵀ` to `H`. Neither `updateCov_decomposition`, which carries `Hᵀ` explicitly, nor the direct
   baseline assumes symmetry either.
4. `he : e ⬝ᵥ e = 1`, the unit-direction condition on the bias-channel projection `e_b`. It is
   carried by `variance_projection_le`, `projDiffusion_remainder_bound` and
   `projDiffusion_randomHessian_remainder_bound`, and by nothing else: the decomposition and its
   projection hold for every `e`. It is written as a dot product rather than as `‖e‖ = 1` on
   purpose, because the norm the `Pi` instance puts on `n → ℝ` is the supremum norm, not the
   Euclidean one, and it is the Euclidean condition that Cauchy-Schwarz needs.
5. `hcross : 0 ≤ e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e)`, the sign of the projected cross term, carried
   by `projDiffusion_ge_baseline_of_cross_nonneg` and by nothing else. It is (A4)'s content and is
   owned by `CouplingSign.lean`; the comparison theorem exists precisely to show that the
   lift/direct asymmetry needs this input and nothing more, so it is stated as a hypothesis here
   rather than smuggled in as a fact. In one case it is discharged here rather than assumed, and
   that case is 6.
6. `hchain : ∀ ω, g ω = c + H *ᵥ dθ ω`, the note's forward-KL chain `δg = H dθ + r` taken with
   `r = 0`, carried by `covMat_of_gradient_chain`, `projDiffusion_cross_nonneg_of_gradient_chain`
   and `projDiffusion_ge_baseline_of_gradient_chain`. It is a modelling hypothesis, not an
   analytic gap: nothing in this file argues that the chain holds, and the note itself states it
   only with a remainder `r`. What these three theorems establish is that the exact chain is
   sufficient for the sign of the cross term, so that (A4)'s real work is the control of `r`
   alone. Neither the decomposition nor its projection assumes it.

Two further conditions appear but are not concessions: componentwise square-integrability, which
is the minimal condition under which the covariances in the statements exist, and the measure
class. `IsFiniteMeasure` suffices for the whole decomposition, its projection, the sign facts and
Cauchy-Schwarz. Two families strengthen this to `IsProbabilityMeasure`: `variance_projection_le`
and the two remainder bounds built on it, because `variance_le_expectation_sq` requires it, and
`covMat_const_add_left` with the three forward-KL chain results that use it, because recentring
absorbs a deterministic shift only once the measure has unit mass; for a finite measure of some
other mass the shift leaves a residue proportional to `(1 - μ(univ))²`.

The single-prefactor idealization (A2) is visible in the type of `s`: it is a scalar, not a
diagonal matrix of positivity-map derivatives. With a genuine matrix prefactor the same proofs go
through with `s • ·` replaced by a `mulVec`, using `covMat_mulVec_left` and
`covMat_mulVec_right`, which are proved here; the scalar form is what the note states, so it is
what is stated here.

The hypotheses are satisfied by non-degenerate instances, not only by degenerate ones. This
matters because every statement below is also satisfied by a Dirac measure and constant data, in
which all three named terms are zero and nothing is being tested. Two instances are given, and
the second exists because the first is not enough. `stdGaussianWitness_*` is a standard Gaussian
in one dimension in which the objective, cross and jitter terms take the nonzero values `1`, `4`
and `4`, the projected diffusion computed directly from `Cov[G]` is `9`, and the same identity
with the factor `2` deleted is false. That instance, however, takes the gradient and the latent weight
fluctuation to be the *same* random vector and the prefactor to be `1`, so in it `Cov[g, dθ]`,
`Cov[g, g]` and `Cov[dθ, dθ]` coincide, `s` and `s²` coincide, and matrices commute; a
decomposition that named the wrong covariance, carried `s` for `s²`, or multiplied the curvature
onto the cross-covariance from the wrong side would pass it unrefuted. `planeWitness_*` closes
all three gaps: two dimensions, prefactor `2`, non-parallel gradient and latent weight directions, so
that the three covariance matrices are distinct and the cross-covariance is not symmetric. In it
the three named forms are `1`, `2` and `4`, the projected diffusion is `36` by both routes, and
each of the three further mis-statements is refuted outright.

What this file does **not** claim: it says nothing about the sign of the cross term beyond the
exact-chain case of 6 above, the general case being (A4)'s job and belonging to
`CouplingSign.lean`; nothing about the size of the Taylor remainder, only about its effect once a
bound on it is supplied; nothing about the stochastic differential equation the diffusion
coefficient feeds, there being no Itô calculus in this mathlib; and nothing about exit times.
The `b`-dependence of the note's `σ_eff²(b)` is carried by the universal quantification over `μ`,
`g`, `dθ`, `H`, `s` and `e`, one instance per slack point; no statement below is specialized to a
particular `b`.
-/

open MeasureTheory Matrix ProbabilityTheory

namespace IcnnLift

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Type*} [Fintype n]

/-! ### Square-integrable random vectors -/

/-- Componentwise square-integrability of an `n`-indexed random vector. This is the standing
hypothesis under which every covariance below is finite and bilinear. -/
def SqIntegrableVec (μ : Measure Ω) (U : Ω → n → ℝ) : Prop :=
  ∀ i, MemLp (fun ω => U ω i) 2 μ

omit [Fintype n] in
/-- A sum of square-integrable random vectors is square-integrable. -/
theorem SqIntegrableVec.add {U W : Ω → n → ℝ} (hU : SqIntegrableVec μ U) (hW : SqIntegrableVec μ W) :
    SqIntegrableVec μ (fun ω => U ω + W ω) := fun i => (hU i).add (hW i)

omit [Fintype n] in
/-- A difference of square-integrable random vectors is square-integrable. This is what lets the
remainder of the update expansion be *constructed* from an arbitrary update rather than
hypothesized; see `updateCov_decomposition_of_sqIntegrable`. -/
theorem SqIntegrableVec.sub {U W : Ω → n → ℝ} (hU : SqIntegrableVec μ U) (hW : SqIntegrableVec μ W) :
    SqIntegrableVec μ (fun ω => U ω - W ω) := fun i => (hU i).sub (hW i)

omit [Fintype n] in
/-- A scalar multiple of a square-integrable random vector is square-integrable. -/
theorem SqIntegrableVec.smul (c : ℝ) {U : Ω → n → ℝ} (hU : SqIntegrableVec μ U) :
    SqIntegrableVec μ (fun ω => c • U ω) := fun i => (hU i).const_mul c

/-- A deterministic linear image of a square-integrable random vector is square-integrable. -/
theorem SqIntegrableVec.mulVec (M : Matrix n n ℝ) {U : Ω → n → ℝ} (hU : SqIntegrableVec μ U) :
    SqIntegrableVec μ (fun ω => M *ᵥ U ω) := fun i => by
  have h : (fun ω => (M *ᵥ U ω) i) = fun ω => ∑ k, M i k * U ω k := by
    funext ω; simp [Matrix.mulVec, dotProduct]
  rw [h]
  exact memLp_finsetSum _ fun k _ => (hU k).const_mul (M i k)

/-- The scalar projection of a square-integrable random vector on a fixed direction is a
square-integrable random variable. -/
theorem SqIntegrableVec.projection (e : n → ℝ) {U : Ω → n → ℝ} (hU : SqIntegrableVec μ U) :
    MemLp (fun ω => e ⬝ᵥ U ω) 2 μ := by
  have h : (fun ω => e ⬝ᵥ U ω) = fun ω => ∑ i, e i * U ω i := by
    funext ω; simp [dotProduct]
  rw [h]
  exact memLp_finsetSum _ fun i _ => (hU i).const_mul (e i)

/-! ### The covariance matrix of two random vectors, and its bilinearity -/

/-- The covariance matrix `E[(U - E U)(W - E W)ᵀ]` of two `n`-indexed random vectors, defined
entrywise from `ProbabilityTheory.covariance`. For `U = W` this is the update covariance
`Cov_X[G]` of the design note. -/
noncomputable def covMat (μ : Measure Ω) (U W : Ω → n → ℝ) : Matrix n n ℝ :=
  Matrix.of fun i j => cov[fun ω => U ω i, fun ω => W ω j; μ]

omit [Fintype n] in
@[simp]
theorem covMat_apply (μ : Measure Ω) (U W : Ω → n → ℝ) (i j : n) :
    covMat μ U W i j = cov[fun ω => U ω i, fun ω => W ω j; μ] := rfl

omit [Fintype n] in
/-- Transposing a covariance matrix swaps its arguments. -/
theorem covMat_transpose (μ : Measure Ω) (U W : Ω → n → ℝ) :
    (covMat μ U W)ᵀ = covMat μ W U := by
  ext i j
  simp [Matrix.transpose_apply, covariance_comm]

omit [Fintype n] in
/-- The covariance of anything with the identically zero random vector vanishes. -/
theorem covMat_zero_right (μ : Measure Ω) (U : Ω → n → ℝ) :
    covMat μ U (fun _ => (0 : n → ℝ)) = 0 := by
  ext i j
  simp [covMat, ProbabilityTheory.covariance]

omit [Fintype n] in
/-- The covariance of the identically zero random vector with anything vanishes. -/
theorem covMat_zero_left (μ : Measure Ω) (U : Ω → n → ℝ) :
    covMat μ (fun _ => (0 : n → ℝ)) U = 0 := by
  ext i j
  simp [covMat, ProbabilityTheory.covariance]

omit [Fintype n] in
/-- Additivity of the covariance matrix in its left argument. -/
theorem covMat_add_left [IsFiniteMeasure μ] {U V W : Ω → n → ℝ}
    (hU : SqIntegrableVec μ U) (hV : SqIntegrableVec μ V) (hW : SqIntegrableVec μ W) :
    covMat μ (fun ω => U ω + V ω) W = covMat μ U W + covMat μ V W := by
  ext i j
  exact covariance_add_left (hU i) (hV i) (hW j)

omit [Fintype n] in
/-- Additivity of the covariance matrix in its right argument. -/
theorem covMat_add_right [IsFiniteMeasure μ] {U V W : Ω → n → ℝ}
    (hU : SqIntegrableVec μ U) (hV : SqIntegrableVec μ V) (hW : SqIntegrableVec μ W) :
    covMat μ U (fun ω => V ω + W ω) = covMat μ U V + covMat μ U W := by
  ext i j
  exact covariance_add_right (hU i) (hV j) (hW j)

omit [Fintype n] in
/-- Homogeneity of the covariance matrix in its left argument. -/
theorem covMat_smul_left (μ : Measure Ω) (c : ℝ) (U W : Ω → n → ℝ) :
    covMat μ (fun ω => c • U ω) W = c • covMat μ U W := by
  ext i j
  exact covariance_const_mul_left (X := fun ω => U ω i) (Y := fun ω => W ω j) (μ := μ) c

omit [Fintype n] in
/-- Homogeneity of the covariance matrix in its right argument. -/
theorem covMat_smul_right (μ : Measure Ω) (c : ℝ) (U W : Ω → n → ℝ) :
    covMat μ U (fun ω => c • W ω) = c • covMat μ U W := by
  ext i j
  exact covariance_const_mul_right (X := fun ω => U ω i) (Y := fun ω => W ω j) (μ := μ) c

omit [Fintype n] in
/-- Adding a deterministic vector to one argument leaves the covariance matrix unchanged. This is
what lets a chain hypothesis written in the note's uncentred form, `g = c + H dθ`, be used without
subtracting means by hand; see `covMat_of_gradient_chain`. -/
theorem covMat_const_add_left [IsProbabilityMeasure μ] (c : n → ℝ) {U W : Ω → n → ℝ}
    (hU : SqIntegrableVec μ U) : covMat μ (fun ω => c + U ω) W = covMat μ U W := by
  ext i j
  exact covariance_const_add_left ((hU i).integrable (by norm_num)) (c i)

/-- Pulling a deterministic matrix out of the left argument: `Cov[M U, W] = M Cov[U, W]`. -/
theorem covMat_mulVec_left [IsFiniteMeasure μ] (M : Matrix n n ℝ) {U W : Ω → n → ℝ}
    (hU : SqIntegrableVec μ U) (hW : SqIntegrableVec μ W) :
    covMat μ (fun ω => M *ᵥ U ω) W = M * covMat μ U W := by
  ext i j
  have h : (fun ω => (M *ᵥ U ω) i) = fun ω => ∑ k, M i k * U ω k := by
    funext ω; simp [Matrix.mulVec, dotProduct]
  rw [covMat_apply, h,
    covariance_fun_sum_left (fun k => (hU k).const_mul (M i k)) (hW j), Matrix.mul_apply]
  exact Finset.sum_congr rfl fun k _ => covariance_const_mul_left _

/-- Pulling a deterministic matrix out of the right argument: `Cov[U, M W] = Cov[U, W] Mᵀ`. -/
theorem covMat_mulVec_right [IsFiniteMeasure μ] (M : Matrix n n ℝ) {U W : Ω → n → ℝ}
    (hU : SqIntegrableVec μ U) (hW : SqIntegrableVec μ W) :
    covMat μ U (fun ω => M *ᵥ W ω) = covMat μ U W * Mᵀ := by
  ext i j
  have h : (fun ω => (M *ᵥ W ω) j) = fun ω => ∑ k, M j k * W ω k := by
    funext ω; simp [Matrix.mulVec, dotProduct]
  rw [covMat_apply, h,
    covariance_fun_sum_right (fun k => (hW k).const_mul (M j k)) (hU i), Matrix.mul_apply]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [covariance_const_mul_right, Matrix.transpose_apply, covMat_apply]
  ring

/-! ### The exact expansion of the covariance of a sum -/

omit [Fintype n] in
/-- The covariance of a sum of two square-integrable random vectors, exactly: the two
covariances plus the two cross-covariances. No approximation enters. -/
theorem covMat_add_self [IsFiniteMeasure μ] {A B : Ω → n → ℝ}
    (hA : SqIntegrableVec μ A) (hB : SqIntegrableVec μ B) :
    covMat μ (fun ω => A ω + B ω) (fun ω => A ω + B ω)
      = covMat μ A A + covMat μ B B + (covMat μ A B + covMat μ B A) := by
  rw [covMat_add_left hA hB (hA.add hB), covMat_add_right hA hA hB,
    covMat_add_right hB hA hB]
  abel

omit [Fintype n] in
/-- The covariance of a sum of three square-integrable random vectors, exactly: the three
covariances plus the three symmetrized cross-covariance pairs. This is the identity the design
note's `Cov_X[G] = …` display abbreviates; it is a theorem, not a leading-order statement. -/
theorem covMat_add_three_self [IsFiniteMeasure μ] {A B C : Ω → n → ℝ}
    (hA : SqIntegrableVec μ A) (hB : SqIntegrableVec μ B) (hC : SqIntegrableVec μ C) :
    covMat μ (fun ω => A ω + B ω + C ω) (fun ω => A ω + B ω + C ω)
      = covMat μ A A + covMat μ B B + covMat μ C C
        + (covMat μ A B + covMat μ B A)
        + (covMat μ A C + covMat μ C A)
        + (covMat μ B C + covMat μ C B) := by
  have hAB : SqIntegrableVec μ (fun ω => A ω + B ω) := hA.add hB
  rw [covMat_add_self hAB hC, covMat_add_self hA hB, covMat_add_left hA hB hC,
    covMat_add_right hC hA hB]
  abel

/-! ### The update covariance -/

omit [MeasurableSpace Ω] in
/-- Reducing a *random* Hessian to the mean Hessian is bookkeeping, not an approximation: if the
update expands with `H(X)` then it expands with any fixed `H` once `(H(X) - H) dθ(X)` is moved
into the remainder. This is the note's step "the `δH` pairings contribute only mixed third
moments, absorbed into `T3`", made explicit.

The reduction is pointwise algebra and carries no probabilistic content, so it does not on its
own transport the theorems below to a random Hessian: the enlarged remainder must still be
square-integrable, and any bound on the remainder must now dominate `s (H(X) - H) dθ(X)` as well.
Both requirements are stated where they are used, in
`updateCov_decomposition_randomHessian` and `projDiffusion_randomHessian_remainder_bound`, which
are the random-Hessian statements proper. -/
theorem updateExpansion_absorb_hessian_fluctuation (s : ℝ) (H : Matrix n n ℝ)
    (Hr : Ω → Matrix n n ℝ) (g dθ R₀ G : Ω → n → ℝ)
    (hG : ∀ ω, G ω = s • g ω + s • (Hr ω *ᵥ dθ ω) + R₀ ω) (ω : Ω) :
    G ω = s • g ω + s • (H *ᵥ dθ ω) + (s • ((Hr ω - H) *ᵥ dθ ω) + R₀ ω) := by
  rw [hG ω, Matrix.sub_mulVec, smul_sub]
  abel

/-- **The exact update-covariance decomposition** — fact (ii) of `lemma1_rederivation.md` §2.

For an update that expands as `G = s g + s H dθ + R` with `R` an arbitrary square-integrable
remainder, the update covariance is *exactly*

  `s² Cov[g]`, the attenuated objective noise;
  `+ s² (Cov[g, dθ] Hᵀ + H Cov[dθ, g])`, the cross term, carrying the slack-channel
  cross-covariance `Σ_slack = E[δθ δgᵀ]` contracted with the curvature;
  `+ s² H Cov[dθ] Hᵀ`, quadratic in the latent weight jitter;
  `+` the five terms the remainder contributes.

Nothing is discarded: this is the bilinear expansion of a covariance, an identity. The note's
`+ h.o.t.` is the parenthesized group, and `projDiffusion_remainder_bound` bounds it. -/
theorem updateCov_decomposition [IsFiniteMeasure μ] (s : ℝ) (H : Matrix n n ℝ)
    {g dθ R G : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ)
    (hR : SqIntegrableVec μ R)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + R ω) :
    covMat μ G G
      = (s ^ 2) • covMat μ g g
        + (s ^ 2) • (covMat μ g dθ * Hᵀ + H * covMat μ dθ g)
        + (s ^ 2) • (H * covMat μ dθ dθ * Hᵀ)
        + (s • (covMat μ g R + covMat μ R g)
            + s • (H * covMat μ dθ R + covMat μ R dθ * Hᵀ)
            + covMat μ R R) := by
  have hGeq : G = fun ω => s • g ω + s • (H *ᵥ dθ ω) + R ω := funext hG
  subst hGeq
  have hHθ : SqIntegrableVec μ fun ω => H *ᵥ dθ ω := hθ.mulVec H
  have hA : SqIntegrableVec μ fun ω => s • g ω := hg.smul s
  have hB : SqIntegrableVec μ fun ω => s • (H *ᵥ dθ ω) := hHθ.smul s
  rw [covMat_add_three_self hA hB hR]
  -- Each of the nine blocks, evaluated.
  have e1 : covMat μ (fun ω => s • g ω) (fun ω => s • g ω) = (s ^ 2) • covMat μ g g := by
    rw [covMat_smul_left, covMat_smul_right, smul_smul, sq]
  have e2 : covMat μ (fun ω => s • (H *ᵥ dθ ω)) (fun ω => s • (H *ᵥ dθ ω))
      = (s ^ 2) • (H * covMat μ dθ dθ * Hᵀ) := by
    rw [covMat_smul_left, covMat_smul_right, covMat_mulVec_left H hθ hHθ,
      covMat_mulVec_right H hθ hθ, smul_smul, sq, Matrix.mul_assoc]
  have e3 : covMat μ (fun ω => s • g ω) (fun ω => s • (H *ᵥ dθ ω))
      = (s ^ 2) • (covMat μ g dθ * Hᵀ) := by
    rw [covMat_smul_left, covMat_smul_right, covMat_mulVec_right H hg hθ, smul_smul, sq]
  have e4 : covMat μ (fun ω => s • (H *ᵥ dθ ω)) (fun ω => s • g ω)
      = (s ^ 2) • (H * covMat μ dθ g) := by
    rw [covMat_smul_left, covMat_smul_right, covMat_mulVec_left H hθ hg, smul_smul, sq]
  have e5 : covMat μ (fun ω => s • g ω) R = s • covMat μ g R := covMat_smul_left _ _ _ _
  have e6 : covMat μ R (fun ω => s • g ω) = s • covMat μ R g := covMat_smul_right _ _ _ _
  have e7 : covMat μ (fun ω => s • (H *ᵥ dθ ω)) R = s • (H * covMat μ dθ R) := by
    rw [covMat_smul_left, covMat_mulVec_left H hθ hR]
  have e8 : covMat μ R (fun ω => s • (H *ᵥ dθ ω)) = s • (covMat μ R dθ * Hᵀ) := by
    rw [covMat_smul_right, covMat_mulVec_right H hR hθ]
  rw [e1, e2, e3, e4, e5, e6, e7, e8]
  simp only [smul_add]
  abel

/-- **The expansion hypothesis is not a restriction.** For an *arbitrary* square-integrable update
`G`, and whatever `s`, `H`, `g` and `dθ` one chooses to name, there is a square-integrable
remainder `R` for which `G = s g + s H dθ + R` holds pointwise, and the decomposition of
`updateCov_decomposition` then applies to `G`.

This is the formal content of the claim that `R` is *defined* by the expansion rather than assumed
small. It says nothing about the size of `R`, which is where the analytic work of (A1)-(A2) lies;
it says only that `hG` rules out no update, so no generality is lost by stating the decomposition
under it. -/
theorem updateCov_decomposition_of_sqIntegrable [IsFiniteMeasure μ] (s : ℝ) (H : Matrix n n ℝ)
    {g dθ G : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ)
    (hGsq : SqIntegrableVec μ G) :
    ∃ R : Ω → n → ℝ, SqIntegrableVec μ R
      ∧ (∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + R ω)
      ∧ covMat μ G G
        = (s ^ 2) • covMat μ g g
          + (s ^ 2) • (covMat μ g dθ * Hᵀ + H * covMat μ dθ g)
          + (s ^ 2) • (H * covMat μ dθ dθ * Hᵀ)
          + (s • (covMat μ g R + covMat μ R g)
              + s • (H * covMat μ dθ R + covMat μ R dθ * Hᵀ)
              + covMat μ R R) := by
  have hsq : SqIntegrableVec μ (fun ω => G ω - (s • g ω + s • (H *ᵥ dθ ω))) :=
    hGsq.sub ((hg.smul s).add ((hθ.mulVec H).smul s))
  have hexp : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω)
      + (fun ω => G ω - (s • g ω + s • (H *ᵥ dθ ω))) ω :=
    fun ω => (add_sub_cancel (s • g ω + s • (H *ᵥ dθ ω)) (G ω)).symm
  exact ⟨_, hsq, hexp, updateCov_decomposition s H hg hθ hsq hexp⟩

/-- The remainder produced by replacing the note's random Hessian `H(X)` by a fixed `H`: the `δH`
pairing `s (H(X) - H) dθ(X)` added to whatever remainder `R₀` the Taylor expansion already left.
This is the object that the note's `+ h.o.t.` must control once the Hessian is allowed to
fluctuate with the batch. -/
def absorbedRemainder (s : ℝ) (H : Matrix n n ℝ) (Hr : Ω → Matrix n n ℝ) (dθ R₀ : Ω → n → ℝ) :
    Ω → n → ℝ := fun ω => s • ((Hr ω - H) *ᵥ dθ ω) + R₀ ω

/-- **The decomposition for a random Hessian**, which is the form the design note actually writes:
`H(X) = ∇²_θ L(θ̄; X)` fluctuates with the batch. Against a fixed `H` — the mean Hessian, in the
note's reading — the three named terms are unchanged and the `δH` pairing is counted inside the
remainder, which is `absorbedRemainder`.

The one substantive hypothesis this adds is `hR`, the square-integrability of that absorbed
remainder; nothing forces it, since `Hr` is an arbitrary matrix-valued function of the batch, and
so it is stated rather than derived. Beyond that the statement is exact: no third moment is
dropped, it is relocated and named. -/
theorem updateCov_decomposition_randomHessian [IsFiniteMeasure μ] (s : ℝ) (H : Matrix n n ℝ)
    (Hr : Ω → Matrix n n ℝ) {g dθ R₀ G : Ω → n → ℝ}
    (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ)
    (hR : SqIntegrableVec μ (absorbedRemainder s H Hr dθ R₀))
    (hG : ∀ ω, G ω = s • g ω + s • (Hr ω *ᵥ dθ ω) + R₀ ω) :
    covMat μ G G
      = (s ^ 2) • covMat μ g g
        + (s ^ 2) • (covMat μ g dθ * Hᵀ + H * covMat μ dθ g)
        + (s ^ 2) • (H * covMat μ dθ dθ * Hᵀ)
        + (s • (covMat μ g (absorbedRemainder s H Hr dθ R₀)
                + covMat μ (absorbedRemainder s H Hr dθ R₀) g)
            + s • (H * covMat μ dθ (absorbedRemainder s H Hr dθ R₀)
                + covMat μ (absorbedRemainder s H Hr dθ R₀) dθ * Hᵀ)
            + covMat μ (absorbedRemainder s H Hr dθ R₀) (absorbedRemainder s H Hr dθ R₀)) :=
  updateCov_decomposition s H hg hθ hR
    (fun ω => updateExpansion_absorb_hessian_fluctuation s H Hr g dθ R₀ G hG ω)

/-- The design note's three-term display, exactly, when the remainder vanishes and the Hessian is
symmetric: `Cov[G] = s² Cov[g] + s² (Σ_gθ H + H Σ_θg) + s² H V H` with `V = Cov[dθ]`.

Every term carries the units of a squared gradient, because the latent-weight-fluctuation statistics
enter *multiplied by the curvature*. The shipped `σ_eff² = s² σ_obj² + σ_Jac²` adds
incommensurables; this does not. -/
theorem updateCov_decomposition_of_remainder_zero [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g dθ G : Ω → n → ℝ}
    (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω)) :
    covMat μ G G
      = (s ^ 2) • covMat μ g g
        + (s ^ 2) • (covMat μ g dθ * H + H * covMat μ dθ g)
        + (s ^ 2) • (H * covMat μ dθ dθ * H) := by
  have hzero : SqIntegrableVec μ (fun _ : Ω => (0 : n → ℝ)) := fun _ => memLp_const 0
  have hG' : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + (fun _ : Ω => (0 : n → ℝ)) ω := by
    intro ω; simpa using hG ω
  have h := updateCov_decomposition s H hg hθ hzero hG'
  rw [covMat_zero_right μ g, covMat_zero_left μ g, covMat_zero_right μ dθ,
    covMat_zero_left μ dθ, covMat_zero_right μ (fun _ : Ω => (0 : n → ℝ)), hH] at h
  simpa using h

/-! ### Projection on the bias-channel direction -/

/-- The projected effective diffusion `σ_eff²(b)` of the design note: the update covariance
contracted with the bias-channel direction `e_b`. -/
noncomputable def projDiffusion (μ : Measure Ω) (e : n → ℝ) (G : Ω → n → ℝ) : ℝ :=
  e ⬝ᵥ (covMat μ G G *ᵥ e)

/-- A quadratic form is additive in the matrix. -/
theorem quadForm_add (M N : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ ((M + N) *ᵥ e) = e ⬝ᵥ (M *ᵥ e) + e ⬝ᵥ (N *ᵥ e) := by
  rw [Matrix.add_mulVec, dotProduct_add]

/-- A quadratic form is homogeneous in the matrix. -/
theorem quadForm_smul (c : ℝ) (M : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ ((c • M) *ᵥ e) = c * (e ⬝ᵥ (M *ᵥ e)) := by
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- A quadratic form does not see transposition; this is what turns the two cross-covariances
into the single factor `2` of the note's `σ_eff²` formula. -/
theorem quadForm_transpose (M : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ (Mᵀ *ᵥ e) = e ⬝ᵥ (M *ᵥ e) := by
  rw [Matrix.dotProduct_mulVec, Matrix.vecMul_transpose, dotProduct_comm]

/-- **The projection identity.** Contracting the covariance matrix with a direction `e` is the
scalar covariance of the projected random variables. -/
theorem covMat_quadForm_eq_covariance [IsFiniteMeasure μ] (e : n → ℝ) {U W : Ω → n → ℝ}
    (hU : SqIntegrableVec μ U) (hW : SqIntegrableVec μ W) :
    e ⬝ᵥ (covMat μ U W *ᵥ e) = cov[fun ω => e ⬝ᵥ U ω, fun ω => e ⬝ᵥ W ω; μ] := by
  have h1 : (fun ω => e ⬝ᵥ U ω) = fun ω => ∑ i, e i * U ω i := by
    funext ω; simp [dotProduct]
  have h2 : (fun ω => e ⬝ᵥ W ω) = fun ω => ∑ j, e j * W ω j := by
    funext ω; simp [dotProduct]
  rw [h1, h2, covariance_fun_sum_fun_sum (fun i => (hU i).const_mul (e i))
    (fun j => (hW j).const_mul (e j))]
  simp only [covariance_const_mul_left, covariance_const_mul_right, dotProduct, Matrix.mulVec,
    covMat_apply, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- **The projected effective diffusion**, fact (ii) of §2 in its scalar form:

`σ_eff²(b) = s² σ_obj² + 2 s² e_bᵀ Cov[g, dθ] H e_b + s² e_bᵀ H V H e_b`, `V = Cov[dθ]`,

with `σ_obj² = e_bᵀ Cov[g] e_b` the projected objective noise. The factor `2` is the sum of the
two cross-covariances, which agree after projection by `quadForm_transpose`. All three terms are
squared-gradient-valued: the latent weight statistics `Cov[g, dθ]` and `V` appear contracted with the
curvature `H`, never bare.

Two hypotheses are carried and are worth naming, because the statement is narrower than
`updateCov_decomposition` in both. The update is assumed *remainder-free*, `G = s g + s H dθ`
exactly, so this is the note's display and not the full identity; for a nonzero remainder the
statement to use is `projDiffusion_remainder_bound`, which bounds the difference rather than
ignoring it. And `H` is assumed symmetric, which is what merges the two cross-covariances into
the single factor `2`. No condition is imposed on `e`: the identity holds for every direction,
and only the Cauchy-Schwarz estimates later in the file need `e ⬝ᵥ e = 1`. -/
theorem projDiffusion_decomposition [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dθ G : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ) (e : n → ℝ)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω)) :
    projDiffusion μ e G
      = s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
        + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
        + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e)) := by
  have hcross : H * covMat μ dθ g = (covMat μ g dθ * H)ᵀ := by
    rw [Matrix.transpose_mul, covMat_transpose, hH]
  rw [projDiffusion, updateCov_decomposition_of_remainder_zero s hH hg hθ hG, hcross,
    quadForm_add, quadForm_add, quadForm_smul, quadForm_smul, quadForm_smul, quadForm_add,
    quadForm_transpose]
  ring

/-- **The projected decomposition with the remainder carried, exactly, and with no symmetry
assumed.** This is `projDiffusion_decomposition` with nothing whatever assumed about `R` and
nothing assumed about `H`: contracting the matrix identity of `updateCov_decomposition` with the
direction `e_b` yields the three named terms together with the three scalars the remainder
contributes, and the equality is exact.

Two things are settled by this statement rather than by prose. First, the remainder contributes
three scalars here rather than the five matrices it contributes to `updateCov_decomposition`,
because projection identifies each cross-covariance with its transpose (`quadForm_transpose`):
`e_bᵀ Cov[R, g] e_b = e_bᵀ Cov[g, R] e_b` and `e_bᵀ Cov[R, dθ] Hᵀ e_b = e_bᵀ H Cov[dθ, R] e_b`.
Setting `R = 0` returns `projDiffusion_decomposition` and bounding the same three scalars by
Cauchy-Schwarz gives `projDiffusion_remainder_bound`, so those two are the two ends of one exact
statement rather than two separate claims, and the note's `+ h.o.t.` is at no stage of the
projection an approximation. Second, the factors `2` are not a consequence of the Hessian being
symmetric: no hypothesis on `H` appears here at all. Symmetry enters only to write `Hᵀ` as `H`,
which is `projDiffusion_decomposition_with_remainder_symmetric`. -/
theorem projDiffusion_decomposition_with_remainder [IsFiniteMeasure μ] (s : ℝ)
    (H : Matrix n n ℝ) {g dθ R G : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dθ) (hR : SqIntegrableVec μ R) (e : n → ℝ)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + R ω) :
    projDiffusion μ e G
      = s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
        + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * Hᵀ) *ᵥ e))
        + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * Hᵀ) *ᵥ e))
        + (2 * s * (e ⬝ᵥ (covMat μ g R *ᵥ e))
            + 2 * s * (e ⬝ᵥ ((H * covMat μ dθ R) *ᵥ e))
            + e ⬝ᵥ (covMat μ R R *ᵥ e)) := by
  have hcov := updateCov_decomposition (μ := μ) s H hg hθ hR hG
  have q1 : e ⬝ᵥ ((H * covMat μ dθ g) *ᵥ e) = e ⬝ᵥ ((covMat μ g dθ * Hᵀ) *ᵥ e) := by
    have h : H * covMat μ dθ g = (covMat μ g dθ * Hᵀ)ᵀ := by
      rw [Matrix.transpose_mul, Matrix.transpose_transpose, covMat_transpose]
    rw [h, quadForm_transpose]
  have q2 : e ⬝ᵥ (covMat μ R g *ᵥ e) = e ⬝ᵥ (covMat μ g R *ᵥ e) := by
    rw [← covMat_transpose μ g R, quadForm_transpose]
  have q3 : e ⬝ᵥ ((covMat μ R dθ * Hᵀ) *ᵥ e) = e ⬝ᵥ ((H * covMat μ dθ R) *ᵥ e) := by
    have h : covMat μ R dθ * Hᵀ = (H * covMat μ dθ R)ᵀ := by
      rw [Matrix.transpose_mul, covMat_transpose]
    rw [h, quadForm_transpose]
  rw [projDiffusion, hcov]
  simp only [quadForm_add, quadForm_smul]
  rw [q1, q2, q3]
  ring

/-- The same identity for a symmetric Hessian, which is the form in which the note's three named
terms are written and the form the remainder bounds are stated against. Its proof is the
preceding result together with a single rewrite of `Hᵀ` to `H`, which is exactly the role
symmetry plays in this file. -/
theorem projDiffusion_decomposition_with_remainder_symmetric [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g dθ R G : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dθ) (hR : SqIntegrableVec μ R) (e : n → ℝ)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + R ω) :
    projDiffusion μ e G
      = s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
        + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
        + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e))
        + (2 * s * (e ⬝ᵥ (covMat μ g R *ᵥ e))
            + 2 * s * (e ⬝ᵥ ((H * covMat μ dθ R) *ᵥ e))
            + e ⬝ᵥ (covMat μ R R *ᵥ e)) := by
  have h := projDiffusion_decomposition_with_remainder (μ := μ) s H hg hθ hR e hG
  rwa [hH] at h

/-- **The direct baseline.** With no latent weight jitter, `dθ = 0`, the cross term and the quadratic
term are covariances with a constant and vanish identically, leaving `σ_eff² = s² σ_obj²`. The
statement is hypothesis-minimal in the way the paper's baseline claim is: no symmetry of `H` is
assumed, because every term that would carry `H` dies whatever `H` is, and so the result is
derived from the exact matrix identity `updateCov_decomposition` rather than from the symmetric
scalar form. The remainder-free hypothesis is likewise not a concession: under the direct
parameterization the latent iterate carries no batch fluctuation at all, so `G = s g` holds
exactly and there is nothing to expand.

This is the baseline *value* only. It does not by itself say that the lift's projected diffusion
exceeds it — that comparison is `projDiffusion_ge_baseline_of_cross_nonneg`, and it needs the sign
of the cross term, which this file does not supply. -/
theorem projDiffusion_direct_baseline [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    {g dθ G : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ0 : ∀ ω, dθ ω = 0) (e : n → ℝ)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω)) :
    projDiffusion μ e G = s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e)) := by
  have hdθ : dθ = fun _ : Ω => (0 : n → ℝ) := funext hθ0
  subst hdθ
  have hzero : SqIntegrableVec μ (fun _ : Ω => (0 : n → ℝ)) := fun _ => memLp_const 0
  have hG' : ∀ ω, G ω = s • g ω + s • (H *ᵥ (fun _ : Ω => (0 : n → ℝ)) ω)
      + (fun _ : Ω => (0 : n → ℝ)) ω := by
    intro ω; simpa using hG ω
  have h := updateCov_decomposition s H hg hzero hzero hG'
  rw [covMat_zero_right μ g, covMat_zero_left μ g,
    covMat_zero_right μ (fun _ : Ω => (0 : n → ℝ))] at h
  have hclean : covMat μ G G = (s ^ 2) • covMat μ g g := by rw [h]; simp
  rw [projDiffusion, hclean, quadForm_smul]

/-! ### What the decomposition settles about the lift/direct asymmetry, and what it does not -/

/-- Conjugation by a symmetric matrix is a change of direction in the quadratic form:
`eᵀ (H M H) e = (H e)ᵀ M (H e)`. This is the step that turns the jitter term of the decomposition
into a variance. -/
theorem quadForm_conj_eq_quadForm_mulVec {H : Matrix n n ℝ} (hH : Hᵀ = H) (M : Matrix n n ℝ)
    (e : n → ℝ) : e ⬝ᵥ ((H * M * H) *ᵥ e) = (H *ᵥ e) ⬝ᵥ (M *ᵥ (H *ᵥ e)) := by
  have h1 : H *ᵥ (M *ᵥ (H *ᵥ e)) = (H * M * H) *ᵥ e := by
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, Matrix.mul_assoc]
  rw [← h1, Matrix.dotProduct_mulVec]
  congr 1
  conv_lhs => rw [← hH]
  rw [Matrix.vecMul_transpose]

/-- The design note's parenthetical `Cov_X[G] ≥ 0 always`, in the projected form in which it is
used: the projected effective diffusion is a variance, hence non-negative in every direction and
for every update, with no hypothesis on `s`, `H` or `dθ`. -/
theorem projDiffusion_nonneg [IsFiniteMeasure μ] {G : Ω → n → ℝ} (hG : SqIntegrableVec μ G)
    (e : n → ℝ) : 0 ≤ projDiffusion μ e G := by
  rw [projDiffusion, covMat_quadForm_eq_covariance e hG hG,
    covariance_self (hG.projection e).aemeasurable]
  exact variance_nonneg _ _

/-- **The jitter term is non-negative unconditionally.** `e_bᵀ H V H e_b` is the variance of the
latent-weight fluctuation projected on `H e_b`, so it is non-negative for every direction and every
symmetric `H`, with no assumption whatever on the coupling. This is the half of the note's
`σ_Jac²` contribution whose sign is free, and it is what leaves the sign of the *cross* term as
the only open question in the comparison below. -/
theorem projDiffusion_jitter_term_nonneg [IsFiniteMeasure μ] {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {dθ : Ω → n → ℝ} (hθ : SqIntegrableVec μ dθ) (e : n → ℝ) :
    0 ≤ e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e) := by
  rw [quadForm_conj_eq_quadForm_mulVec hH, covMat_quadForm_eq_covariance _ hθ hθ,
    covariance_self (hθ.projection _).aemeasurable]
  exact variance_nonneg _ _

/-- **The structural asymmetry, and exactly the input it needs.** The lift's projected diffusion
is at least the direct baseline's, `s² σ_obj²`, as soon as the cross term is non-negative. The
jitter term contributes its sign for free (`projDiffusion_jitter_term_nonneg`), so the hypothesis
`hcross` is the whole of what the comparison requires beyond the decomposition.

The comparison is at fixed `s`, `g` and `H`: the lift and the direct parameterization are being
read at the same mean iterate, with the same objective noise and the same curvature, and they
differ only in whether `dθ` is present. That is the note's comparison, and stating it this way is
the honest form of it: the asymmetry is a corollary of the decomposition *together with* (A4),
not of the decomposition alone. Whether `hcross` holds in general is not decided here. The one
case the note singles out is settled immediately below rather than asserted in prose: under the
forward-KL chain `δg = H dθ + r` taken with `r = 0`, the projected cross form is exactly
`e_bᵀ H V H e_b`, so `hcross` holds and the comparison follows
(`projDiffusion_cross_nonneg_of_gradient_chain`, `projDiffusion_ge_baseline_of_gradient_chain`).
The case `r ≠ 0`, which is what (A4) must actually control, is `CouplingSign.lean`'s work and is
not touched here. -/
theorem projDiffusion_ge_baseline_of_cross_nonneg [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dθ G : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ)
    (e : n → ℝ) (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω))
    (hcross : 0 ≤ e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e)) :
    s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e)) ≤ projDiffusion μ e G := by
  rw [projDiffusion_decomposition s hH hg hθ e hG]
  have hjit : 0 ≤ e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e) :=
    projDiffusion_jitter_term_nonneg hH hθ e
  nlinarith [sq_nonneg s, hcross, hjit]

/-- **The note's forward-KL chain, at the matrix level.** If the gradient fluctuation is carried
by the latent-weight fluctuation through the curvature — `g = c + H dθ` pointwise, which is the note's
`δg = H dθ + r` with `r = 0`, written before centring — then the slack-channel cross-covariance is
the curvature applied to the latent weight covariance, `Cov[g, dθ] = H V`.

The constant `c` is free and is not required to be `E[g]`: the covariance does not see it
(`covMat_const_add_left`), which is why the chain may be stated in the uncentred form the note
uses. What is assumed is that the chain holds exactly; the remainder `r` of the note's chain is
absent, and this file proves nothing about it. -/
theorem covMat_of_gradient_chain [IsProbabilityMeasure μ] (H : Matrix n n ℝ) (c : n → ℝ)
    {g dθ : Ω → n → ℝ} (hθ : SqIntegrableVec μ dθ) (hchain : ∀ ω, g ω = c + H *ᵥ dθ ω) :
    covMat μ g dθ = H * covMat μ dθ dθ := by
  have hg : g = fun ω => c + H *ᵥ dθ ω := funext hchain
  subst hg
  rw [covMat_const_add_left c (hθ.mulVec H), covMat_mulVec_left H hθ hθ]

/-- **Under the exact forward-KL chain the sign of the cross term is free.** With `g = c + H dθ`
the projected cross form coincides with the jitter form, `e_bᵀ Cov[g, dθ] H e_b = e_bᵀ H V H e_b`,
which is a variance and hence non-negative. This is the note's sentence "the sign comes for free",
proved rather than asserted, and it is the exact-chain half of (A4). The half that remains, the
note's remainder clause `r ≠ 0`, is not addressed here. -/
theorem projDiffusion_cross_nonneg_of_gradient_chain [IsProbabilityMeasure μ]
    {H : Matrix n n ℝ} (hH : Hᵀ = H) (c : n → ℝ) {g dθ : Ω → n → ℝ}
    (hθ : SqIntegrableVec μ dθ) (hchain : ∀ ω, g ω = c + H *ᵥ dθ ω) (e : n → ℝ) :
    0 ≤ e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e) := by
  rw [covMat_of_gradient_chain H c hθ hchain]
  exact projDiffusion_jitter_term_nonneg hH hθ e

/-- **The structural asymmetry under the exact chain, with no sign hypothesis left over.** For an
update whose gradient obeys `g = c + H dθ` exactly, the lift's projected diffusion is at least the
direct baseline's `s² σ_obj²`. This is `projDiffusion_ge_baseline_of_cross_nonneg` with its one
hypothesis discharged, and it is the strongest form of the note's asymmetry claim that this file
establishes. It is not the paper's assumption (A4), which asserts the comparison under the weaker
chain `δg = H dθ + r` with `r` merely controlled; that weaker statement is `CouplingSign.lean`'s
and nothing here bears on it. -/
theorem projDiffusion_ge_baseline_of_gradient_chain [IsProbabilityMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) (c : n → ℝ) {g dθ G : Ω → n → ℝ}
    (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ) (e : n → ℝ)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω)) (hchain : ∀ ω, g ω = c + H *ᵥ dθ ω) :
    s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e)) ≤ projDiffusion μ e G :=
  projDiffusion_ge_baseline_of_cross_nonneg s hH hg hθ e hG
    (projDiffusion_cross_nonneg_of_gradient_chain hH c hθ hchain e)

/-! ### The remainder, bounded rather than dropped -/

/-- Cauchy-Schwarz for `ProbabilityTheory.covariance`, proved from bilinearity and the
non-negativity of the variance through the discriminant of `t ↦ Var[X + t Y]`. mathlib does not
provide it. -/
theorem abs_covariance_le_sqrt_variance_mul_sqrt_variance [IsFiniteMeasure μ] {X Y : Ω → ℝ}
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ) :
    |cov[X, Y; μ]| ≤ Real.sqrt (Var[X; μ]) * Real.sqrt (Var[Y; μ]) := by
  have key : ∀ t : ℝ, 0 ≤ Var[Y; μ] * (t * t) + 2 * cov[X, Y; μ] * t + Var[X; μ] := by
    intro t
    have h := variance_add (μ := μ) hX (hY.const_mul t)
    have hs : Var[fun ω => t * Y ω; μ] = t ^ 2 * Var[Y; μ] := variance_const_mul t Y μ
    have hc : cov[X, fun ω => t * Y ω; μ] = t * cov[X, Y; μ] := covariance_const_mul_right t
    have hnn : 0 ≤ Var[X + fun ω => t * Y ω; μ] := variance_nonneg _ _
    rw [h, hs, hc] at hnn
    nlinarith [hnn]
  have hdisc : discrim (Var[Y; μ]) (2 * cov[X, Y; μ]) (Var[X; μ]) ≤ 0 := discrim_le_zero key
  rw [discrim] at hdisc
  have hsq : cov[X, Y; μ] ^ 2 ≤ Var[X; μ] * Var[Y; μ] := by nlinarith [hdisc]
  calc |cov[X, Y; μ]| = Real.sqrt (cov[X, Y; μ] ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (Var[X; μ] * Var[Y; μ]) := Real.sqrt_le_sqrt hsq
    _ = Real.sqrt (Var[X; μ]) * Real.sqrt (Var[Y; μ]) :=
        Real.sqrt_mul (variance_nonneg _ _) _

omit [Fintype n] in
/-- The L² perturbation bound behind the remainder estimate: perturbing a square-integrable
random variable by one of standard deviation at most `ε` moves its variance by at most
`2 ε √(Var) + ε²`. Every statement about the decomposition proper is an identity; the three
inequalities of this file all live in this final section, and this is the one that converts the
note's `+ h.o.t.` into a bound. -/
theorem abs_variance_add_sub_variance_le [IsFiniteMeasure μ] {p q : Ω → ℝ}
    (hp : MemLp p 2 μ) (hq : MemLp q 2 μ) {ε : ℝ} (hε : 0 ≤ ε) (hvq : Var[q; μ] ≤ ε ^ 2) :
    |Var[fun ω => p ω + q ω; μ] - Var[p; μ]| ≤ 2 * ε * Real.sqrt (Var[p; μ]) + ε ^ 2 := by
  rw [variance_fun_add hp hq]
  have hsqvq : Real.sqrt (Var[q; μ]) ≤ ε := by
    have h := Real.sqrt_le_sqrt hvq
    rwa [Real.sqrt_sq hε] at h
  have hcs : |cov[p, q; μ]| ≤ Real.sqrt (Var[p; μ]) * Real.sqrt (Var[q; μ]) :=
    abs_covariance_le_sqrt_variance_mul_sqrt_variance hp hq
  have hcs' : |cov[p, q; μ]| ≤ Real.sqrt (Var[p; μ]) * ε :=
    hcs.trans (mul_le_mul_of_nonneg_left hsqvq (Real.sqrt_nonneg _))
  have hsplit : Var[p; μ] + 2 * cov[p, q; μ] + Var[q; μ] - Var[p; μ]
      = 2 * cov[p, q; μ] + Var[q; μ] := by ring
  rw [hsplit]
  have habs : |2 * cov[p, q; μ] + Var[q; μ]| ≤ 2 * |cov[p, q; μ]| + Var[q; μ] := by
    have h1 := abs_add_le (2 * cov[p, q; μ]) (Var[q; μ])
    have h2 : |2 * cov[p, q; μ]| = 2 * |cov[p, q; μ]| := by rw [abs_mul]; norm_num
    have h3 : |Var[q; μ]| = Var[q; μ] := abs_of_nonneg (variance_nonneg _ _)
    linarith [h1, h2.le, h3.le]
  have hfin : 2 * |cov[p, q; μ]| + Var[q; μ] ≤ 2 * ε * Real.sqrt (Var[p; μ]) + ε ^ 2 := by
    nlinarith [hcs', hvq, Real.sqrt_nonneg (Var[p; μ]), hε]
  linarith [habs, hfin]

/-- The projected remainder has variance at most its vector second moment, hence at most `ε²`
for a unit direction. -/
theorem variance_projection_le [IsProbabilityMeasure μ] {R : Ω → n → ℝ}
    (hR : SqIntegrableVec μ R) {e : n → ℝ} (he : e ⬝ᵥ e = 1) {ε : ℝ}
    (hRε : ∫ ω, R ω ⬝ᵥ R ω ∂μ ≤ ε ^ 2) :
    Var[fun ω => e ⬝ᵥ R ω; μ] ≤ ε ^ 2 := by
  have hproj : MemLp (fun ω => e ⬝ᵥ R ω) 2 μ := hR.projection e
  have hint1 : Integrable (fun ω => (e ⬝ᵥ R ω) ^ 2) μ := hproj.integrable_sq
  have hint2 : Integrable (fun ω => R ω ⬝ᵥ R ω) μ := by
    have h : (fun ω => R ω ⬝ᵥ R ω) = fun ω => ∑ i, R ω i * R ω i := by
      funext ω; simp [dotProduct]
    rw [h]
    exact integrable_finsetSum _ fun i _ => (hR i).integrable_mul (hR i)
  have hpt : ∀ ω, (e ⬝ᵥ R ω) ^ 2 ≤ R ω ⬝ᵥ R ω := by
    intro ω
    have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ e fun i => R ω i
    have hee : ∑ i, e i ^ 2 = 1 := by rw [← he]; simp [dotProduct, sq]
    have hrr : ∑ i, R ω i ^ 2 = R ω ⬝ᵥ R ω := by simp [dotProduct, sq]
    have hlhs : (∑ i, e i * R ω i) = e ⬝ᵥ R ω := by simp [dotProduct]
    rw [hee, hrr, hlhs, one_mul] at hcs
    exact hcs
  calc Var[fun ω => e ⬝ᵥ R ω; μ] ≤ ∫ ω, (e ⬝ᵥ R ω) ^ 2 ∂μ := by
        simpa using variance_le_expectation_sq (μ := μ) hproj.aestronglyMeasurable
    _ ≤ ∫ ω, R ω ⬝ᵥ R ω ∂μ := integral_mono hint1 hint2 hpt
    _ ≤ ε ^ 2 := hRε

/-- **The honest form of the note's `+ h.o.t.`** For an update `G = s g + s H dθ + R` with a
remainder of second moment at most `ε²` and a unit projection direction `e_b`, the projected
effective diffusion differs from the three named terms

`N = s² σ_obj² + 2 s² e_bᵀ Cov[g, dθ] H e_b + s² e_bᵀ H V H e_b`

by at most `2 ε √N + ε²`, an explicit expression in `ε` obtained from Cauchy-Schwarz. At `ε = 0`
this recovers `projDiffusion_decomposition`; nothing here assumes that `R` vanishes, and the
remainder is bounded rather than discarded. -/
theorem projDiffusion_remainder_bound [IsProbabilityMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dθ R G : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dθ) (hR : SqIntegrableVec μ R)
    (hG : ∀ ω, G ω = s • g ω + s • (H *ᵥ dθ ω) + R ω)
    {e : n → ℝ} (he : e ⬝ᵥ e = 1) {ε : ℝ} (hε : 0 ≤ ε)
    (hRε : ∫ ω, R ω ⬝ᵥ R ω ∂μ ≤ ε ^ 2) :
    |projDiffusion μ e G
        - (s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
            + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
            + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e)))|
      ≤ 2 * ε * Real.sqrt (s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
            + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
            + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e))) + ε ^ 2 := by
  have hGeq : G = fun ω => s • g ω + s • (H *ᵥ dθ ω) + R ω := funext hG
  subst hGeq
  have hHθ : SqIntegrableVec μ fun ω => H *ᵥ dθ ω := hθ.mulVec H
  have hPsq : SqIntegrableVec μ fun ω => s • g ω + s • (H *ᵥ dθ ω) := (hg.smul s).add (hHθ.smul s)
  have hpL : MemLp (fun ω => e ⬝ᵥ (s • g ω + s • (H *ᵥ dθ ω))) 2 μ := hPsq.projection e
  have hqL : MemLp (fun ω => e ⬝ᵥ R ω) 2 μ := hR.projection e
  -- The three named terms are the variance of the projected named part of the update.
  have hN : Var[fun ω => e ⬝ᵥ (s • g ω + s • (H *ᵥ dθ ω)); μ]
      = s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
        + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
        + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e)) := by
    have h := projDiffusion_decomposition (μ := μ) s hH hg hθ e
      (G := fun ω => s • g ω + s • (H *ᵥ dθ ω)) fun _ => rfl
    rw [projDiffusion, covMat_quadForm_eq_covariance e hPsq hPsq,
      covariance_self hpL.aemeasurable] at h
    exact h
  -- The projected update is the variance of the projected sum.
  have hGP : projDiffusion μ e (fun ω => s • g ω + s • (H *ᵥ dθ ω) + R ω)
      = Var[fun ω => (e ⬝ᵥ (s • g ω + s • (H *ᵥ dθ ω))) + e ⬝ᵥ R ω; μ] := by
    rw [projDiffusion, covMat_quadForm_eq_covariance e (hPsq.add hR) (hPsq.add hR)]
    have hsum : (fun ω => e ⬝ᵥ (s • g ω + s • (H *ᵥ dθ ω) + R ω))
        = fun ω => (e ⬝ᵥ (s • g ω + s • (H *ᵥ dθ ω))) + e ⬝ᵥ R ω := by
      funext ω; rw [dotProduct_add]
    rw [hsum, covariance_self (hpL.aemeasurable.add hqL.aemeasurable)]
  rw [hGP, ← hN]
  exact abs_variance_add_sub_variance_le hpL hqL hε (variance_projection_le hR he hRε)


/-- **The remainder bound for a random Hessian.** The note's Hessian fluctuates with the batch,
and the price of naming a fixed `H` in the three leading terms is that `ε` must now control the
`δH` pairing `s (H(X) - H) dθ(X)` as well as the Taylor remainder `R₀`. That is exactly what
`hRε` says here, since the integrand is `absorbedRemainder`, and it is the reason the note's
sentence "the `δH` pairings contribute only mixed third moments" is a hypothesis rather than a
conclusion: nothing in this file bounds those mixed moments, and supplying that bound is part of
the same analytic task as (A1)'s moment conditions. -/
theorem projDiffusion_randomHessian_remainder_bound [IsProbabilityMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) (Hr : Ω → Matrix n n ℝ) {g dθ R₀ G : Ω → n → ℝ}
    (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dθ)
    (hR : SqIntegrableVec μ (absorbedRemainder s H Hr dθ R₀))
    (hG : ∀ ω, G ω = s • g ω + s • (Hr ω *ᵥ dθ ω) + R₀ ω)
    {e : n → ℝ} (he : e ⬝ᵥ e = 1) {ε : ℝ} (hε : 0 ≤ ε)
    (hRε : ∫ ω, absorbedRemainder s H Hr dθ R₀ ω ⬝ᵥ absorbedRemainder s H Hr dθ R₀ ω ∂μ ≤ ε ^ 2) :
    |projDiffusion μ e G
        - (s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
            + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
            + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e)))|
      ≤ 2 * ε * Real.sqrt (s ^ 2 * (e ⬝ᵥ (covMat μ g g *ᵥ e))
            + 2 * s ^ 2 * (e ⬝ᵥ ((covMat μ g dθ * H) *ᵥ e))
            + s ^ 2 * (e ⬝ᵥ ((H * covMat μ dθ dθ * H) *ᵥ e))) + ε ^ 2 :=
  projDiffusion_remainder_bound s hH hg hθ hR
    (fun ω => updateExpansion_absorb_hessian_fluctuation s H Hr g dθ R₀ G hG ω) he hε hRε

/-! ### A non-degenerate instance

Every statement above is also satisfied by a Dirac measure carrying constant data, in which the
three named terms are all zero and the decomposition asserts `0 = 0`. That is a real risk for a
formalization whose point is a factor and an ordering of matrix products, so this section pins
down one instance in which nothing is degenerate and checks the decomposition against a
`projDiffusion` computed by a different route.

The instance is the standard Gaussian on `ℝ` with `d = 1`, gradient and latent-weight fluctuation both
equal to the coordinate, curvature `H = 2` and prefactor `s = 1`, so that the update is `3 ω` and
`Cov[G] = 9`. The three named terms come out `1`, `4` and `4`. Their sum is the directly computed
`9`, and `stdGaussianWitness_factor_two_necessary` records that the same identity with the factor
`2` deleted would give `7`, hence is false: the factor is falsifiable here, and it survives. -/

namespace StdGaussianWitness

/-- The standard Gaussian on `ℝ`, the batch law of the witness instance. -/
noncomputable def batchLaw : Measure ℝ := gaussianReal 0 1

instance : IsProbabilityMeasure batchLaw := by unfold batchLaw; infer_instance

/-- The witness gradient at the mean iterate: the coordinate itself, so its projected variance is
`1` rather than `0`. -/
def grad : ℝ → Fin 1 → ℝ := fun ω _ => ω

/-- The witness latent-weight fluctuation. Taking it equal to `grad` makes the cross term as large as
Cauchy-Schwarz allows, which is what distinguishes the factor `2` from the factor `1`. -/
def jitter : ℝ → Fin 1 → ℝ := fun ω _ => ω

/-- The witness curvature, `H = 2`. It is not `1`, so a term that ought to carry one factor of
`H` cannot be confused with a term that carries two. -/
def hessian : Matrix (Fin 1) (Fin 1) ℝ := fun _ _ => 2

/-- The witness bias-channel direction, a unit vector. -/
def dir : Fin 1 → ℝ := fun _ => 1

/-- The witness update, defined as the expansion itself with `s = 1` and no remainder, so that
`stdGaussianWitness_expansion` holds definitionally and the instance is genuinely an instance of
`projDiffusion_decomposition`. -/
noncomputable def update : ℝ → Fin 1 → ℝ :=
  fun ω => (1 : ℝ) • grad ω + (1 : ℝ) • (hessian *ᵥ jitter ω)

end StdGaussianWitness

open StdGaussianWitness in
/-- The witness gradient is square-integrable: all moments of a real Gaussian are finite. -/
theorem stdGaussianWitness_grad_sqIntegrable : SqIntegrableVec batchLaw grad :=
  fun _ => memLp_id_gaussianReal' 2 (by simp)

open StdGaussianWitness in
/-- The witness latent-weight fluctuation is square-integrable. -/
theorem stdGaussianWitness_jitter_sqIntegrable : SqIntegrableVec batchLaw jitter :=
  fun _ => memLp_id_gaussianReal' 2 (by simp)

open StdGaussianWitness in
/-- The witness curvature is symmetric, as `projDiffusion_decomposition` requires. -/
theorem stdGaussianWitness_hessian_symm : hessianᵀ = hessian := by ext i j; simp [hessian]

open StdGaussianWitness in
/-- The witness direction is a unit vector in the Euclidean sense the Cauchy-Schwarz bounds use. -/
theorem stdGaussianWitness_dir_unit : dir ⬝ᵥ dir = 1 := by simp [dotProduct, dir]

open StdGaussianWitness in
/-- The witness update satisfies the expansion hypothesis with `s = 1` and zero remainder. -/
theorem stdGaussianWitness_expansion :
    ∀ ω, update ω = (1 : ℝ) • grad ω + (1 : ℝ) • (hessian *ᵥ jitter ω) := fun _ => rfl

open StdGaussianWitness in
/-- The coordinate has unit variance under the witness measure, which is what makes every named
term below nonzero. -/
theorem stdGaussianWitness_covariance_coord :
    cov[fun x : ℝ => x, fun x : ℝ => x; batchLaw] = 1 := by
  rw [covariance_self (by fun_prop)]
  unfold batchLaw
  simp

open StdGaussianWitness in
/-- The projected objective noise of the witness, `σ_obj² = 1`. -/
theorem stdGaussianWitness_objective_term :
    dir ⬝ᵥ (covMat batchLaw grad grad *ᵥ dir) = 1 := by
  have h : covMat batchLaw grad grad = fun _ _ => 1 := by
    ext i j; exact stdGaussianWitness_covariance_coord
  rw [h]; simp [dotProduct, Matrix.mulVec, dir]

open StdGaussianWitness in
/-- The projected cross form of the witness, `e_bᵀ Cov[g, dθ] H e_b = 2`. Note that it carries one
factor of the curvature, not two. -/
theorem stdGaussianWitness_cross_term :
    dir ⬝ᵥ ((covMat batchLaw grad jitter * hessian) *ᵥ dir) = 2 := by
  have h : covMat batchLaw grad jitter = fun _ _ => 1 := by
    ext i j; exact stdGaussianWitness_covariance_coord
  rw [h]; simp [dotProduct, Matrix.mulVec, Matrix.mul_apply, dir, hessian]

open StdGaussianWitness in
/-- The projected jitter form of the witness, `e_bᵀ H V H e_b = 4`. It carries two factors of the
curvature, which is why it differs from the cross form. -/
theorem stdGaussianWitness_jitter_term :
    dir ⬝ᵥ ((hessian * covMat batchLaw jitter jitter * hessian) *ᵥ dir) = 4 := by
  have h : covMat batchLaw jitter jitter = fun _ _ => 1 := by
    ext i j; exact stdGaussianWitness_covariance_coord
  rw [h]
  simp [dotProduct, Matrix.mulVec, Matrix.mul_apply, dir, hessian]
  norm_num

open StdGaussianWitness in
/-- The projected diffusion of the witness computed *directly* from `Cov[G]`, without reference to
the decomposition: the update is `3 ω`, so `σ_eff² = 9`. This is the independent value the
decomposition is checked against. -/
theorem stdGaussianWitness_projDiffusion : projDiffusion batchLaw dir update = 9 := by
  have hu : ∀ ω, update ω = fun _ : Fin 1 => 3 * ω := by
    intro ω; ext i; simp [update, grad, jitter, hessian, Matrix.mulVec, dotProduct]; ring
  have h : covMat batchLaw update update = fun _ _ => 9 := by
    ext i j
    have h1 : (fun ω => update ω i) = fun ω : ℝ => 3 * ω := by funext ω; rw [hu ω]
    have h2 : (fun ω => update ω j) = fun ω : ℝ => 3 * ω := by funext ω; rw [hu ω]
    rw [covMat_apply, h1, h2, covariance_const_mul_left, covariance_const_mul_right,
      stdGaussianWitness_covariance_coord]
    norm_num
  rw [projDiffusion, h]; simp [dotProduct, Matrix.mulVec, dir]

open StdGaussianWitness in
/-- **The decomposition, checked against an independent computation.** The left-hand side is
reached here through `projDiffusion_decomposition`, and in `stdGaussianWitness_projDiffusion`
through `Cov[G]` directly; the two routes share no step beyond the definition of `covMat`, so a
decomposition stated with the wrong factor would be refuted by this instance outright. One limit
of the instance is stated rather than implied: in one dimension matrices commute, so the *side*
on which the curvature multiplies the cross-covariance is not discriminated here — that is pinned
by the proof of `projDiffusion_decomposition` itself, which derives the side from
`covMat_mulVec_left` and `covMat_mulVec_right` rather than asserting it. Nothing
in the instance is degenerate: the three terms are `1`, `4` and `4`, none of them zero, and the
cross form (`2`) is distinguishable from the jitter form (`4`). -/
theorem stdGaussianWitness_decomposition_checks_out :
    projDiffusion batchLaw dir update = 1 + 2 * 2 + 4 := by
  rw [projDiffusion_decomposition (μ := batchLaw) 1 stdGaussianWitness_hessian_symm
      stdGaussianWitness_grad_sqIntegrable stdGaussianWitness_jitter_sqIntegrable dir
      stdGaussianWitness_expansion,
    stdGaussianWitness_objective_term, stdGaussianWitness_cross_term,
    stdGaussianWitness_jitter_term]
  norm_num

open StdGaussianWitness in
/-- **The factor `2` is falsifiable, and it survives.** In the witness instance the same identity
with the cross term counted once rather than twice would read `1 + 2 + 4 = 7`, and the projected
diffusion is `9`. A formalization that had merged the two cross-covariances incorrectly would be
refuted here rather than merely unproved. -/
theorem stdGaussianWitness_factor_two_necessary :
    projDiffusion batchLaw dir update
      ≠ (1 : ℝ) ^ 2 * (dir ⬝ᵥ (covMat batchLaw grad grad *ᵥ dir))
        + (1 : ℝ) ^ 2 * (dir ⬝ᵥ ((covMat batchLaw grad jitter * hessian) *ᵥ dir))
        + (1 : ℝ) ^ 2 * (dir ⬝ᵥ ((hessian * covMat batchLaw jitter jitter * hessian) *ᵥ dir)) := by
  rw [stdGaussianWitness_projDiffusion, stdGaussianWitness_objective_term,
    stdGaussianWitness_cross_term, stdGaussianWitness_jitter_term]
  norm_num

/-! ### A witness that discriminates the statistics, the curvature side and the prefactor

The Gaussian instance above rules out the crudest form of vacuity, but it cannot rule out the
form that matters here. In it the gradient and the latent-weight fluctuation are the *same* random
vector and the prefactor is `1`, so `Cov[g, dθ]`, `Cov[g, g]` and `Cov[dθ, dθ]` coincide, `s` and
`s²` coincide, and the dimension is `1`, in which matrices commute. A decomposition that named the
wrong one of the three covariances, that carried `s` where it should carry `s²`, or that
multiplied the curvature onto the cross-covariance from the wrong side would pass that instance
unrefuted.

This section closes those three gaps. The sample space is the same standard Gaussian, the
dimension is `2`, the prefactor is `2`, and the gradient and the latent-weight fluctuation are the
distinct rank-one vectors `ω ↦ ω (1, 2)` and `ω ↦ ω (3, -1)`, so that

    Cov[g, g] = ((1,2),(2,4)),   Cov[g, dθ] = ((3,-1),(6,-2)),   Cov[dθ, dθ] = ((9,-3),(-3,1)),

three different matrices, of which the middle one is not symmetric. With the curvature
`H = ((1,1),(1,2))` and the unit direction `e_b = (1,0)` the three named forms take the three
distinct values `1`, `2` and `4`, the projected diffusion is `36`, and the decomposition returns
`2² · 1 + 2 · 2² · 2 + 2² · 4 = 36`. Each of the four mis-statements above is then refuted
outright, by `planeWitness_factor_two_necessary`,
`planeWitness_prefactor_squared_necessary`, `planeWitness_curvature_side_necessary` and
`planeWitness_cross_covariance_necessary`. -/

namespace PlaneWitness

/-- The gradient direction of the plane witness. -/
def gradVec : Fin 2 → ℝ := ![1, 2]

/-- The latent-weight-fluctuation direction of the plane witness, chosen not parallel to `gradVec` so
that the cross-covariance is neither of the two variances and is not symmetric. -/
def jitterVec : Fin 2 → ℝ := ![3, -1]

/-- The curvature of the plane witness: symmetric, as a Hessian is, and not a multiple of the
identity, so that the side on which it multiplies the cross-covariance is visible. -/
def hessian : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 1, 2]

/-- The bias-channel direction of the plane witness, a Euclidean unit vector. -/
def dir : Fin 2 → ℝ := ![1, 0]

/-- The positivity-map prefactor of the plane witness. It is `2` rather than `1` precisely so that `s`
and `s²` are told apart. -/
def prefactor : ℝ := 2

/-- The witness gradient at the mean iterate. -/
def grad : ℝ → Fin 2 → ℝ := fun ω => ω • gradVec

/-- The witness latent-weight fluctuation. -/
def jitter : ℝ → Fin 2 → ℝ := fun ω => ω • jitterVec

/-- The witness update, defined as the expansion itself with no remainder, so that the instance is
genuinely an instance of `projDiffusion_decomposition`. -/
noncomputable def update : ℝ → Fin 2 → ℝ :=
  fun ω => prefactor • grad ω + prefactor • (hessian *ᵥ jitter ω)

end PlaneWitness

/-- Every random vector of the witness is a fixed vector scaled by the Gaussian coordinate, hence
square-integrable. -/
theorem planeWitness_sqIntegrable (v : Fin 2 → ℝ) :
    SqIntegrableVec StdGaussianWitness.batchLaw (fun ω => ω • v) := by
  intro i
  have h : (fun ω : ℝ => (ω • v) i) = fun ω : ℝ => v i * ω := by
    funext ω; simp [mul_comm]
  rw [h]
  exact (memLp_id_gaussianReal' 2 (by simp)).const_mul (v i)

/-- The covariance matrix of two rank-one witness vectors is their outer product, because the
Gaussian coordinate has unit variance. -/
theorem planeWitness_covMat (u v : Fin 2 → ℝ) :
    covMat StdGaussianWitness.batchLaw (fun ω => ω • u) (fun ω => ω • v)
      = Matrix.vecMulVec u v := by
  ext i j
  have h1 : (fun ω : ℝ => (ω • u) i) = fun ω : ℝ => u i * ω := by funext ω; simp [mul_comm]
  have h2 : (fun ω : ℝ => (ω • v) j) = fun ω : ℝ => v j * ω := by funext ω; simp [mul_comm]
  rw [covMat_apply, h1, h2, covariance_const_mul_left, covariance_const_mul_right,
    stdGaussianWitness_covariance_coord, Matrix.vecMulVec_apply]
  ring

open PlaneWitness in
/-- Contracting with the witness direction reads off the upper-left entry. -/
theorem planeWitness_quadForm (M : Matrix (Fin 2) (Fin 2) ℝ) : dir ⬝ᵥ (M *ᵥ dir) = M 0 0 := by
  simp [dotProduct, Matrix.mulVec, dir, Fin.sum_univ_two]

open PlaneWitness in
/-- The witness curvature is symmetric, as `projDiffusion_decomposition` requires. -/
theorem planeWitness_hessian_symm : hessianᵀ = hessian := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [hessian]

open PlaneWitness in
/-- The witness direction is a unit vector in the Euclidean sense the file's bounds use. -/
theorem planeWitness_dir_unit : dir ⬝ᵥ dir = 1 := by
  simp [dotProduct, dir, Fin.sum_univ_two]

open PlaneWitness in
/-- The witness gradient is square-integrable. -/
theorem planeWitness_grad_sqIntegrable : SqIntegrableVec StdGaussianWitness.batchLaw grad :=
  planeWitness_sqIntegrable gradVec

open PlaneWitness in
/-- The witness latent-weight fluctuation is square-integrable. -/
theorem planeWitness_jitter_sqIntegrable : SqIntegrableVec StdGaussianWitness.batchLaw jitter :=
  planeWitness_sqIntegrable jitterVec

open PlaneWitness in
/-- The witness update satisfies the expansion hypothesis with zero remainder. -/
theorem planeWitness_expansion :
    ∀ ω, update ω = prefactor • grad ω + prefactor • (hessian *ᵥ jitter ω) := fun _ => rfl

open PlaneWitness in
/-- The witness objective covariance. -/
theorem planeWitness_covMat_grad_grad :
    covMat StdGaussianWitness.batchLaw grad grad = !![1, 2; 2, 4] := by
  rw [show grad = fun ω : ℝ => ω • gradVec from rfl, planeWitness_covMat]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply, gradVec]

open PlaneWitness in
/-- The witness slack-channel cross-covariance `Σ_slack = E[δg δθᵀ]`. It is neither of the two
variances and it is not symmetric, which is what makes the cross term identifiable. -/
theorem planeWitness_covMat_grad_jitter :
    covMat StdGaussianWitness.batchLaw grad jitter = !![3, -1; 6, -2] := by
  rw [show grad = fun ω : ℝ => ω • gradVec from rfl,
      show jitter = fun ω : ℝ => ω • jitterVec from rfl, planeWitness_covMat]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply, gradVec, jitterVec]

open PlaneWitness in
/-- The witness latent weight covariance `V`. -/
theorem planeWitness_covMat_jitter_jitter :
    covMat StdGaussianWitness.batchLaw jitter jitter = !![9, -3; -3, 1] := by
  rw [show jitter = fun ω : ℝ => ω • jitterVec from rfl, planeWitness_covMat]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply, jitterVec]

open PlaneWitness in
/-- The projected objective noise of the plane witness, `σ_obj² = 1`. -/
theorem planeWitness_objective_term :
    dir ⬝ᵥ (covMat StdGaussianWitness.batchLaw grad grad *ᵥ dir) = 1 := by
  rw [planeWitness_covMat_grad_grad, planeWitness_quadForm]; norm_num

open PlaneWitness in
/-- The projected cross form of the plane witness, `e_bᵀ Cov[g, dθ] H e_b = 2`. -/
theorem planeWitness_cross_term :
    dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw grad jitter * hessian) *ᵥ dir) = 2 := by
  rw [planeWitness_covMat_grad_jitter, planeWitness_quadForm]
  norm_num [Matrix.mul_apply, hessian, Fin.sum_univ_two]

open PlaneWitness in
/-- The projected jitter form of the plane witness, `e_bᵀ H V H e_b = 4`. -/
theorem planeWitness_jitter_term :
    dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw jitter jitter * hessian) *ᵥ dir) = 4 := by
  rw [planeWitness_covMat_jitter_jitter, planeWitness_quadForm]
  norm_num [Matrix.mul_apply, hessian, Fin.sum_univ_two]

open PlaneWitness in
/-- The witness update is again rank one: `G(ω) = ω (6, 6)`. -/
theorem planeWitness_update_rankOne : update = fun ω : ℝ => ω • (![6, 6] : Fin 2 → ℝ) := by
  funext ω
  ext i
  fin_cases i <;>
    norm_num [update, grad, jitter, prefactor, hessian, gradVec, jitterVec, Matrix.mulVec,
      dotProduct, Fin.sum_univ_two] <;> ring

open PlaneWitness in
/-- The projected diffusion of the plane witness computed *directly* from `Cov[G]`, without
reference to the decomposition: `σ_eff² = 36`. This is the independent value the decomposition is
checked against. -/
theorem planeWitness_projDiffusion :
    projDiffusion StdGaussianWitness.batchLaw dir update = 36 := by
  rw [projDiffusion, planeWitness_update_rankOne, planeWitness_covMat, planeWitness_quadForm,
    Matrix.vecMulVec_apply]
  norm_num

open PlaneWitness in
/-- **The decomposition, checked against an independent computation, in a plane.** The left-hand
side is reached here through `projDiffusion_decomposition` and in `planeWitness_projDiffusion`
through `Cov[G]` directly, and the two agree at `2² · 1 + 2 · 2² · 2 + 2² · 4 = 36`. Nothing in
the instance is degenerate: the three named forms are `1`, `2` and `4`, the three covariance
matrices are distinct, the cross-covariance is not symmetric, and the prefactor is not `1`. -/
theorem planeWitness_decomposition_checks_out :
    projDiffusion StdGaussianWitness.batchLaw dir update
      = prefactor ^ 2 * 1 + 2 * prefactor ^ 2 * 2 + prefactor ^ 2 * 4 := by
  rw [projDiffusion_decomposition (μ := StdGaussianWitness.batchLaw) prefactor
      planeWitness_hessian_symm planeWitness_grad_sqIntegrable planeWitness_jitter_sqIntegrable
      dir planeWitness_expansion,
    planeWitness_objective_term, planeWitness_cross_term, planeWitness_jitter_term]

open PlaneWitness in
/-- The cross form with the curvature multiplying from the left, `e_bᵀ H Cov[g, dθ] e_b = 9`. It
differs from the value the decomposition names, which is why the plane witness sees the side. -/
theorem planeWitness_cross_term_curvature_on_the_left :
    dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw grad jitter) *ᵥ dir) = 9 := by
  rw [planeWitness_covMat_grad_jitter, planeWitness_quadForm]
  norm_num [Matrix.mul_apply, hessian, Fin.sum_univ_two]

open PlaneWitness in
/-- The cross form built from the latent weight covariance instead of the cross-covariance, `6`. -/
theorem planeWitness_cross_form_of_jitter_covariance :
    dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw jitter jitter * hessian) *ᵥ dir) = 6 := by
  rw [planeWitness_covMat_jitter_jitter, planeWitness_quadForm]
  norm_num [Matrix.mul_apply, hessian, Fin.sum_univ_two]

open PlaneWitness in
/-- The cross form built from the objective covariance instead of the cross-covariance, `3`. -/
theorem planeWitness_cross_form_of_objective_covariance :
    dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw grad grad * hessian) *ᵥ dir) = 3 := by
  rw [planeWitness_covMat_grad_grad, planeWitness_quadForm]
  norm_num [Matrix.mul_apply, hessian, Fin.sum_univ_two]

open PlaneWitness in
/-- **The factor `2` is falsifiable, and it survives.** Counting the cross term once rather than
twice gives `28` where the projected diffusion is `36`. -/
theorem planeWitness_factor_two_necessary :
    projDiffusion StdGaussianWitness.batchLaw dir update
      ≠ prefactor ^ 2 * (dir ⬝ᵥ (covMat StdGaussianWitness.batchLaw grad grad *ᵥ dir))
        + prefactor ^ 2 *
            (dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw grad jitter * hessian) *ᵥ dir))
        + prefactor ^ 2 *
            (dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw jitter jitter * hessian)
              *ᵥ dir)) := by
  rw [planeWitness_projDiffusion, planeWitness_objective_term, planeWitness_cross_term,
    planeWitness_jitter_term]
  norm_num [prefactor]

open PlaneWitness in
/-- **The prefactor is squared, and that is falsifiable too.** Carrying `s` where the
decomposition carries `s²` gives `18` where the projected diffusion is `36`. The Gaussian instance
of the previous section, having `s = 1`, cannot see this. -/
theorem planeWitness_prefactor_squared_necessary :
    projDiffusion StdGaussianWitness.batchLaw dir update
      ≠ prefactor * (dir ⬝ᵥ (covMat StdGaussianWitness.batchLaw grad grad *ᵥ dir))
        + 2 * prefactor *
            (dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw grad jitter * hessian) *ᵥ dir))
        + prefactor *
            (dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw jitter jitter * hessian)
              *ᵥ dir)) := by
  rw [planeWitness_projDiffusion, planeWitness_objective_term, planeWitness_cross_term,
    planeWitness_jitter_term]
  norm_num [prefactor]

open PlaneWitness in
/-- **The curvature multiplies the cross-covariance on the right, and that is falsifiable.**
Multiplying from the left gives `92` where the projected diffusion is `36`. In one dimension the
two sides agree, so this is the discrimination the plane witness adds. -/
theorem planeWitness_curvature_side_necessary :
    projDiffusion StdGaussianWitness.batchLaw dir update
      ≠ prefactor ^ 2 * (dir ⬝ᵥ (covMat StdGaussianWitness.batchLaw grad grad *ᵥ dir))
        + 2 * prefactor ^ 2 *
            (dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw grad jitter) *ᵥ dir))
        + prefactor ^ 2 *
            (dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw jitter jitter * hessian)
              *ᵥ dir)) := by
  rw [planeWitness_projDiffusion, planeWitness_objective_term,
    planeWitness_cross_term_curvature_on_the_left, planeWitness_jitter_term]
  norm_num [prefactor]

open PlaneWitness in
/-- **The cross term carries the cross-covariance and neither variance, and that is falsifiable.**
Substituting `Cov[dθ, dθ]` gives `68` and substituting `Cov[g, g]` gives `44`, where the projected
diffusion is `36`. This is the discrimination that a witness with `g = dθ` cannot supply, and it
is the one that matters, since the paper's claim is precisely that the slack-channel
cross-covariance `Σ_slack` is what the cross term carries. -/
theorem planeWitness_cross_covariance_necessary :
    projDiffusion StdGaussianWitness.batchLaw dir update
        ≠ prefactor ^ 2 * (dir ⬝ᵥ (covMat StdGaussianWitness.batchLaw grad grad *ᵥ dir))
          + 2 * prefactor ^ 2 *
              (dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw jitter jitter * hessian) *ᵥ dir))
          + prefactor ^ 2 *
              (dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw jitter jitter * hessian)
                *ᵥ dir))
      ∧ projDiffusion StdGaussianWitness.batchLaw dir update
        ≠ prefactor ^ 2 * (dir ⬝ᵥ (covMat StdGaussianWitness.batchLaw grad grad *ᵥ dir))
          + 2 * prefactor ^ 2 *
              (dir ⬝ᵥ ((covMat StdGaussianWitness.batchLaw grad grad * hessian) *ᵥ dir))
          + prefactor ^ 2 *
              (dir ⬝ᵥ ((hessian * covMat StdGaussianWitness.batchLaw jitter jitter * hessian)
                *ᵥ dir)) := by
  constructor
  · rw [planeWitness_projDiffusion, planeWitness_objective_term,
      planeWitness_cross_form_of_jitter_covariance, planeWitness_jitter_term]
    norm_num [prefactor]
  · rw [planeWitness_projDiffusion, planeWitness_objective_term,
      planeWitness_cross_form_of_objective_covariance, planeWitness_jitter_term]
    norm_num [prefactor]

end IcnnLift
