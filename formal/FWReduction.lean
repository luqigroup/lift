import Mathlib

/-!
# (FW-4) The multi-dimensional Freidlin–Wentzell action projects onto the scalar statement

This file machine-checks (FW-4) of `docs/design/lemma1_rederivation.md` §7, the *reduction check*
of the corrected mechanism: the claim that projecting the multi-dimensional Freidlin–Wentzell
action onto the bias-channel direction recovers the one-dimensional statement (L2') of §3
exactly, so that the paper's scalar barrier-rescaling result is the projection of the
multi-dimensional programme and not a separate theorem about a different object. §11.4 of the
same note records (FW-4) as the last item of the chain and calls it "mechanical given FW-1's
consistency check". This file is what that word has to mean if it is to mean anything: the
reduction is carried out as an exact computation, with every hypothesis it needs named.

**The setting.** The corrected route replaces the shipped Hessian decomposition, which §1 of the
note refutes, by a statement about the *diffusion* of the pullback dynamics. The multi-dimensional
target is the Itô diffusion `dφ = -∇L̃(φ) dt + √η Σ_eff(φ)^{1/2} dB` with `Σ_eff` the full,
anisotropic, state-dependent update covariance of §2, and the object one compares across methods
is the Freidlin–Wentzell action of a path together with its infimum, the quasipotential. The
action is defined here in the standard form
`S_T[γ] = (1/2) ∫₀ᵀ (γ̇(t) - μ(γ(t)))ᵀ D(γ(t))⁻¹ (γ̇(t) - μ(γ(t))) dt`
for a drift field `μ` and a diffusion matrix field `D`, and in the scalar form
`(1/2) ∫₀ᵀ (q̇(t) - m(q(t)))² / d(q(t)) dt` in one dimension.

**mathlib supplies no large-deviations theory**, and the audit of this version is unambiguous
about it: there is no Itô integral, no stochastic differential equation, no Fokker–Planck
equation, no exit-time theory and no Freidlin–Wentzell rate function anywhere in mathlib v4.31.0.
Nothing below therefore *derives* the action from the diffusion; the file computes with the
functional, which is the only honest thing to do here, and every statement is a statement about
that functional. The identification of this functional as the large-deviation rate function of
the diffusion is the citation to Freidlin and Wentzell (3rd ed., Ch. 3–4) that the note already
carries, and it is an input, not a result.

**A convention that the reduction check settles.** §7 of the note writes the action with the
constant `1/4`. For the stochastic differential equation displayed immediately above it,
`dφ = -∇L̃ dt + √η Σ_eff^{1/2} dB`, the Freidlin–Wentzell rate function carries `1/2`, and `1/4`
is the constant belonging to the alternative normalization `√(2η) Σ_eff^{1/2} dB`. The two differ
by a factor of two in the absolute exponent. The reduction check decides the matter, since only
`1/2` makes the projected action reproduce (L2')'s Kramers exponent `2α/(η σ_eff²)`: with `1/2`,
`fwActionMulti_alongLine_barrier` gives `S = 2 (α / σ_eff²)` and `S / η = 2α/(η σ_eff²)`, which is
`BarrierRescaling.arrheniusExponent α η σ_eff²` on the nose, whereas
`quarter_normalized_fwActionScalar_eq_barrier_div` computes the value `1/4` would have given,
`α / σ_eff²`, whose exponent `α/(η σ_eff²)` is half of (L2')'s. Both values are theorems of this
file, so the discrepancy is checked and not merely asserted. Every comparison in the note is a
ratio or an inequality between two actions carrying the same constant and is unaffected; only the
absolute exponent is, and `1/2` is the value that makes the two levels of the paper agree. The
constant `1/2` is therefore the one used throughout this file, and the note's `1/4` should be
corrected to `1/2` (or, equivalently, its exit asymptotic changed to `exp(2V*/η)`).

**What is proved.** Four things, in order.

First, the projection itself. If the path is confined to the line `{γ₀ + r e : r ∈ ℝ}` through a
unit vector `e`, if the drift is tangent to that line along the path, and if the diffusion matrix
*acts on that line as the scalar* `eᵀ D e` — the predicate `ActsAsScalarOn`, which for a unit `e`
says exactly that `e` is an eigenvector of `D` — then the multi-dimensional action of the path
equals the
one-dimensional action of its coordinate `q`, with scalar diffusion `eᵀ D e`. This is
`fwActionMulti_alongLine`, and it is an identity between the two integrals with no error term and
no integrability hypothesis, because the two integrands are equal pointwise on the interval.

Second, what survives when `e` is not an eigendirection, and how much the eigendirection
hypothesis is really assuming. The identity `eᵀ D⁻¹ e = (eᵀ D e)⁻¹` is false in general; what is
true for positive definite `D` is the Cauchy–Schwarz bound `1/(eᵀ D e) ≤ eᵀ D⁻¹ e`, proved here as
`one_div_diffusionAlong_le_diffusionAlong_inv` from the Kantorovich residual identity
`quadForm_inv_residual`. The same identity gives the equality case:
`diffusionAlong_inv_eq_iff_actsAsScalarOn` shows that `eᵀ D⁻¹ e = (eᵀ D e)⁻¹` holds *if and only
if* `e` is an eigenvector of `D`, so `ActsAsScalarOn` is not merely a convenient sufficient
condition for the pointwise projection but exactly its necessary and sufficient one. That it
cannot be dropped is proved twice on the same explicit witness `Σ = !![1, 1; 1, 2]`,
`e = (1, 0)`: at the level of the matrices by `exists_posDef_lt_diffusionAlong_inv`, and at the
level of the functionals by `exists_alongLine_fwActionScalar_lt_fwActionMulti`, which exhibits a
positive definite constant diffusion, a line-confined path and a tangent drift satisfying every
other hypothesis of the projection theorem and for which its conclusion is false, the scalar
action being `1/2` against a multi-dimensional action of `1`. Consequently the one-dimensional
action is in general a *lower* bound for the multi-dimensional action of the same line-confined
path, `fwActionScalar_le_fwActionMulti_alongLine`: the scalar reduction never overstates the cost
of a line-confined crossing, and understates it exactly to the extent that the bias direction
fails to be an eigendirection of `Σ_eff`.

Third, the consistency check with (L2'). On a band where the projected diffusion is the constant
`c`, every path that climbs from `q(0)` to `q(T)` against the pullback potential `L̃` has action at
least `2 (α / c)` with `α = L̃(q(T)) - L̃(q(0))`, and the reversed relaxation `q̇ = L̃'(q)` — the
classical Freidlin–Wentzell instanton for a one-dimensional gradient system — achieves it exactly.
That is `two_mul_barrier_div_le_fwActionScalar` and `fwActionScalar_eq_two_mul_barrier_div`; the
lower bound holds for every admissible path and not only for minimizers, and is proved by
completing the square, `(a + b)² ≥ 4ab`, rather than by a variational argument. Combining with
the projection gives
`fwActionMulti_alongLine_barrier`, whose right-hand side `2 * (α / c)` displays the effective
action `α_eff = α / σ_eff²` of (L2') literally as a subterm, and
`fwActionMulti_alongLine_arrheniusExponent`, which divides by `η` to give `2α/(η c)` — the
Arrhenius exponent that `BarrierRescaling.lean` compares across the two methods and that
`BarrierRescaling.quasipotential_gap_of_const_diffusion` reaches by the entirely different route
of the stationary density (*) of §3. Those two modules are cited, not reproved: nothing about the
stationary density, the quasipotential `U_eff`, or the Kramers relation is redone here. The
one-dimensional action of `QuasipotentialMonotone.fwAction`, at normalization `1/2` and drift
`-L̃'`, is the `fwActionScalar` of this file, so the reduction lands in the vocabulary (FW-2)
already uses. Those correspondences were checked by reading the definitions in the two modules,
not by importing them; the identifications are recorded in prose and are not machine-checked here.

Fourth, and because a theorem carrying nine hypotheses can be true for the wrong reason, an
explicit witness. `fwActionMulti_barrierWitness` instantiates every hypothesis of
`fwActionMulti_alongLine_barrier` in the two-dimensional gradient system
`dφ = -∇Φ(φ) dt + √η Σ^{1/2} dB` with `Φ(x) = x₀²/2` and the constant diagonal
`Σ = !![c, 0; 0, 1]`, positive definite on the physical range `c > 0`
(`barrierWitnessDiffusion_posDef`), along the reversed relaxation `q(t) = e^t` on the
eigendirection `e = (1, 0)`; it evaluates the action to `(e^{2T} - 1)/c`, which is strictly
positive for `T, c > 0` (`fwActionMulti_barrierWitness_pos`). The drift of that instance is the
honest gradient drift of `Φ`, coordinate by coordinate
(`hasDerivAt_barrierWitnessPotential_update`), and `Φ` restricted to the bias-channel line is the
scalar potential the barrier `α` is measured with (`barrierWitnessPotential_alongLine`). So the
assembled statement is not satisfied only by constant paths across a zero barrier, where it would
degenerate to `0 = 0`. The witness bears on satisfiability and on nothing else: it is one path in
one system, and it says as little about the minimization as the rest of the file does.

**What is NOT proved, and must not be read as proved.** The reduction is *not* the minimization.
This file shows that *if* a path is confined to the bias-channel line then its multi-dimensional
action is the one-dimensional action of its coordinate, and it computes the value of the
one-dimensional problem. It does not show that the minimizer of the multi-dimensional problem is
confined to that line. That is a genuine variational claim about the full anisotropic,
state-dependent diffusion, it is not established anywhere in the note, and nothing here bears on
it; the note lists (FW-4) as mechanical for the *reduction*, and the two must not be conflated.
Without that confinement statement the most these results support is a one-sided reading:
restricting the competitors of the multi-dimensional variational problem to line-confined paths
can only raise an infimum, so the value computed here is an upper bound for the multi-dimensional
quasipotential and never a lower one. Even that inequality is not formalized — no infimum over a
path space is taken anywhere in this file — and it is recorded only so that the reduction is not
mistaken for the comparison the escape-rate argument needs. Two further gaps are equally
deliberate. The infimum
`2 (α / c)` is approached but not attained on a compact time interval, since the reversed
relaxation leaves a critical point only in infinite time: the lower bound holds for every
admissible path, and the exact value is proved for whichever reversed-relaxation path is supplied,
so the identification of the infimum is complete only in the limit `T → ∞`, which is not taken
here. And the passage from a quasipotential to a mean first-passage time — the Freidlin–Wentzell
exit asymptotic `E[τ] ≍ exp(V*/η)`, prefactors excluded — is not available in this mathlib and is
not attempted; it is `KramersExitTime.lean`'s business and is carried there as a hypothesis.

The paths are restricted throughout to those differentiable on the compact interval `[0, T]`, in
the sense that each theorem hypothesizes `HasDerivAt` or `DifferentiableAt` at every point of
`uIcc 0 T`. The absolutely continuous paths of the general theory are not modelled; a clean
statement about a restricted class was preferred to an unfinished statement about the general one.
The action is nevertheless defined for an arbitrary path, using `deriv`, so the definition is total
and the regularity enters only where it is used. Three of Lean's conventions are thereby visible in
the definition and are not the mathematics — a non-integrable integrand, a non-differentiable path
and a *singular* diffusion matrix all record the value zero where the Freidlin–Wentzell cost is
`+∞`, and for `T < 0` the expression is an oriented integral rather than an action. The docstring
of `fwActionMulti` says which theorem controls which; the singular case is worth naming twice,
because the degeneracy of `Σ_eff^direct` as `s → 0` is exactly (FW-3)'s subject and nothing here
speaks to that limit.

## Results
* `diffusionAlong` — the diffusion coefficient `eᵀ D e` of `D` along a direction `e`.
* `ActsAsScalarOn` — `D` acts on the line `ℝ e` as multiplication by `eᵀ D e`.
* `diffusionAlong_smul` — `(c e)ᵀ M (c e) = c² (eᵀ M e)`.
* `diffusionAlong_eq_of_mulVec_eq_smul`, `actsAsScalarOn_of_mulVec_eq_smul` — the eigenvalue of a
  unit eigenvector is its Rayleigh quotient.
* `diffusionAlong_pos` — a positive definite diffusion is positive along every unit direction.
* `inv_mulVec_of_actsAsScalarOn` — on an eigendirection, `D⁻¹ e = (eᵀ D e)⁻¹ e`.
* `diffusionAlong_inv_of_actsAsScalarOn` — hence `eᵀ D⁻¹ e = (eᵀ D e)⁻¹`.
* `quadForm_inv_smul_of_actsAsScalarOn` — hence `(c e)ᵀ D⁻¹ (c e) = c² / (eᵀ D e)`.
* `diffusionAlong_ne_zero_of_actsAsScalarOn` — an invertible `D` has a non-zero coefficient along
  any unit direction it scales, so non-degeneracy of the projected diffusion is not a hypothesis.
* `quadForm_inv_residual` — the Kantorovich residual identity
  `wᵀ D⁻¹ w = eᵀ D⁻¹ e - (eᵀ D e)⁻¹` at `w = e - (eᵀ D e)⁻¹ D e`.
* `one_div_diffusionAlong_le_diffusionAlong_inv` — in general only `1/(eᵀ D e) ≤ eᵀ D⁻¹ e`.
* `diffusionAlong_inv_eq_iff_actsAsScalarOn` — equality holds **exactly** on eigendirections, so
  `ActsAsScalarOn` is the necessary and sufficient hypothesis and not merely a sufficient one.
* `tiltedWitness`, `tiltedWitnessDirection`, `tiltedWitness_posDef`, `tiltedWitness_inv`,
  `diffusionAlong_tiltedWitness`, `diffusionAlong_inv_tiltedWitness`,
  `not_actsAsScalarOn_tiltedWitness` — the explicit positive definite `2 × 2` counterexample.
* `exists_posDef_lt_diffusionAlong_inv` — the Cauchy–Schwarz inequality is strict for that
  witness, so the eigendirection hypothesis is not removable from the matrix identity.
* `alongLine`, `hasDerivAt_alongLine`, `deriv_alongLine` — the line-confined path and its velocity.
* `fwIntegrandMulti`, `fwActionMulti` — the multi-dimensional Freidlin–Wentzell action.
* `fwIntegrandScalar`, `fwActionScalar` — its one-dimensional counterpart.
* `fwActionScalar_congr_diffusion` — the scalar action sees the diffusion only along the path.
* `fwActionMulti_alongLine` — **the projection**: on an eigendirection the multi-dimensional
  action of a line-confined path equals the one-dimensional action with diffusion `eᵀ D e`.
* `fwIntegrandScalar_le_fwIntegrandMulti_alongLine`, `fwActionScalar_le_fwActionMulti_alongLine` —
  without the eigendirection hypothesis one inequality survives, in the stated direction.
* `exists_alongLine_fwActionScalar_lt_fwActionMulti` — the eigendirection hypothesis is not
  removable from the projection theorem itself: an instance satisfying every other hypothesis on
  which the projected action is `1/2` and the multi-dimensional action is `1`.
* `integral_potential_work_eq_barrier` — the work identity `∫ L̃'(q) q̇ = L̃(q(T)) - L̃(q(0))`.
* `two_mul_barrier_div_le_fwActionScalar` — every crossing costs at least `2 (α / c)`.
* `fwActionScalar_eq_two_mul_barrier_div` — the reversed relaxation costs exactly `2 (α / c)`.
* `quarter_normalized_fwActionScalar_eq_barrier_div` — under §7's literal constant `1/4` the same
  crossing costs `α / c`, one half of the exponent (L2') states; this is the normalization
  discrepancy above, checked rather than argued.
* `fwActionMulti_alongLine_barrier` — **(FW-4)**: the multi-dimensional action of the
  line-confined reversed relaxation is `2 (α / c)`, with `α / c` the effective action of (L2').
* `fwActionMulti_alongLine_arrheniusExponent` — divided by `η`, it is `2α/(η c)`, the Arrhenius
  exponent of `BarrierRescaling.arrheniusExponent`.
* `barrierWitnessDirection`, `barrierWitnessDiffusion`, `barrierWitnessPotential`,
  `barrierWitnessDrift` and their lemmas — the explicit gradient-system instance.
* `hasDerivAt_barrierWitnessPotential_update` — the witness drift is the exact gradient drift
  `-∇Φ` of the witness potential, coordinate by coordinate.
* `fwActionMulti_barrierWitness`, `fwActionMulti_barrierWitness_pos` — **(FW-4) is not vacuous**:
  every hypothesis of the assembled theorem is satisfiable, with action `(e^{2T} - 1)/c > 0`.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. The explicit hypotheses carried, and what each would take to
discharge:

* `he : e ⬝ᵥ e = 1` — the bias-channel direction is a unit vector. A normalization, free of
  content: rescaling `e` rescales `diffusionAlong` quadratically and the coordinate `q` inversely.
* `hscal : ActsAsScalarOn (D _) e` — the diffusion matrix acts on the bias-channel line as a
  scalar. This is the substantive linear-algebra hypothesis of the reduction, and it is not
  removable: it is *equivalent* to the pointwise identity the projection rests on
  (`diffusionAlong_inv_eq_iff_actsAsScalarOn`), it is false for a general positive definite
  `Σ_eff` (`exists_posDef_lt_diffusionAlong_inv`), and the conclusion of the projection theorem
  itself fails without it (`exists_alongLine_fwActionScalar_lt_fwActionMulti`). Discharging it
  means showing that the bias channel is an eigendirection of the update covariance of §2, which
  is a claim about `Σ_eff` that the note does not make. The inequality
  `fwActionScalar_le_fwActionMulti_alongLine` is what remains when it is dropped, and it is proved
  without it.
* `hmu : mu _ = m (q _) • e` — the drift is tangent to the bias-channel line along the path. For
  the gradient drift `-∇L̃` this says the gradient has no component transverse to the line at the
  points the path visits, which is a statement about `L̃` on the band and not a triviality.
* `hdet : IsUnit (D _).det` — invertibility of the diffusion matrix, that is, ellipticity. This is
  the input of (FW-3) in §7, where the note works at fixed small `s > 0` and quotes uniform
  ellipticity with constant `λ(s) = s² λ₀`, and which §11.2 records as resolved because (A2) pins
  `s` near `σ_s`. It is carried pointwise here rather than uniformly, which is all the pointwise
  computation needs.
* `hpos : (D _).PosDef` — positive definiteness, used only for the inequality half, where it is
  what makes the Cauchy–Schwarz argument available.
* `hband : diffusionAlong (D _) e = c` — the projected diffusion is constant on the band. This is
  the note's own "crossed within the shoulder band where `σ_eff²` is approximately constant".
  `BarrierRescaling.quasipotential_gap_bounds` replaces the same phrase by an explicit two-sided
  bracket on its side of the argument; the corresponding bracket for the action is not proved here,
  and the constant-band case is the one carried, so the approximation is visible as a hypothesis
  rather than hidden in a "to leading order".
* `hc : 0 < c`, `hc : c ≠ 0` — non-degeneracy of the projected diffusion on the band. The lower
  bound `two_mul_barrier_div_le_fwActionScalar` needs the sign, since it divides an inequality by
  `c`; the identity `fwActionScalar_eq_two_mul_barrier_div` needs only `c ≠ 0`, and is stated at
  that strength, but a negative `c` is not a diffusion and the physical case is `c > 0`
  throughout. Nothing is assumed about the *value* of `c`, which is where the comparison between
  the two methods lives.
* `heta : eta ≠ 0` — the learning rate is non-zero, needed only to divide the action by it in
  `fwActionMulti_alongLine_arrheniusExponent`.
* `hq`, `hLtilde`, `hint`, `hintW`, `hT` — differentiability of the path on `uIcc 0 T`,
  differentiability of the potential along it, interval integrability of the two integrands, and
  `0 ≤ T`. Ordinary regularity, supplied in the paper by (A1); the integrability hypotheses are
  stated rather than derived from continuity because the class of admissible paths is left open.

Non-degeneracy of the projected diffusion, `eᵀ D e ≠ 0`, is *not* a hypothesis of the projection
theorem: `diffusionAlong_ne_zero_of_actsAsScalarOn` derives it from `he`, `hdet` and `hscal`,
since a matrix scaling a unit vector by zero annihilates it and so is singular. All the hypotheses
above are jointly satisfiable, and satisfiable with a strictly positive barrier: see
`fwActionMulti_barrierWitness` and `fwActionMulti_barrierWitness_pos`, which exhibit an elliptic
gradient system meeting every one of them and compute an action of `(e^{2T} - 1)/c > 0`.

The one thing that would close the remaining gap in (FW-4) is a proof that the minimizing path of
the multi-dimensional problem is confined to the bias-channel line. It is not proved here, it is
not proved in the note, and no statement below should be read as evidence for it.
-/

open Matrix Set MeasureTheory

namespace IcnnLift

/-! ### The diffusion coefficient along a direction -/

section Direction

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The **diffusion coefficient of `D` along the direction `e`**, the Rayleigh quotient
`eᵀ D e`. For a unit vector `e` this is the variance the diffusion matrix `D` assigns to the
coordinate along `e`, and it is the scalar `σ_eff²` that the one-dimensional statement (L2') of
`docs/design/lemma1_rederivation.md` §3 uses once `e` is the bias-channel direction `e_b`. -/
def diffusionAlong (D : Matrix n n ℝ) (e : n → ℝ) : ℝ := e ⬝ᵥ (D *ᵥ e)

/-- `D` **acts on the line `ℝ e` as the scalar `eᵀ D e`**, that is, `e` is an eigenvector of `D`
with eigenvalue its own Rayleigh quotient. This is the hypothesis under which the projection of
the multi-dimensional Freidlin–Wentzell action onto the line `ℝ e` is exact; it is the Lean form
of §7's phrase "projecting the multi-d action onto the bias-channel direction". -/
def ActsAsScalarOn (D : Matrix n n ℝ) (e : n → ℝ) : Prop :=
  D *ᵥ e = diffusionAlong D e • e

omit [DecidableEq n] in
/-- Symmetry of the bilinear form of a symmetric matrix, in the `dotProduct`/`mulVec` vocabulary
the rest of the development uses. -/
theorem dotProduct_mulVec_comm_of_transpose_eq {M : Matrix n n ℝ} (hM : Mᵀ = M) (x y : n → ℝ) :
    x ⬝ᵥ (M *ᵥ y) = y ⬝ᵥ (M *ᵥ x) := by
  rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hM, dotProduct_comm]

omit [DecidableEq n] in
/-- The quadratic form of `M` is homogeneous of degree two along a fixed direction:
`(c e)ᵀ M (c e) = c² (eᵀ M e)`. This is the only place the scalar coordinate of a line-confined
path meets the matrix. -/
theorem diffusionAlong_smul (M : Matrix n n ℝ) (e : n → ℝ) (c : ℝ) :
    (c • e) ⬝ᵥ (M *ᵥ (c • e)) = c ^ 2 * diffusionAlong M e := by
  rw [Matrix.mulVec_smul, smul_dotProduct, dotProduct_smul, diffusionAlong]
  simp only [smul_eq_mul]
  ring

omit [DecidableEq n] in
/-- The eigenvalue of a **unit** eigenvector is its Rayleigh quotient: if `D e = c e` and
`eᵀ e = 1` then `c = eᵀ D e`. This is why `ActsAsScalarOn` can name the eigenvalue rather than
quantify over it. -/
theorem diffusionAlong_eq_of_mulVec_eq_smul {D : Matrix n n ℝ} {e : n → ℝ} {c : ℝ}
    (he : e ⬝ᵥ e = 1) (h : D *ᵥ e = c • e) : diffusionAlong D e = c := by
  rw [diffusionAlong, h, dotProduct_smul, he, smul_eq_mul, mul_one]

omit [DecidableEq n] in
/-- A unit eigenvector of `D` is a direction on which `D` acts as a scalar, in the sense of
`ActsAsScalarOn`. -/
theorem actsAsScalarOn_of_mulVec_eq_smul {D : Matrix n n ℝ} {e : n → ℝ} {c : ℝ}
    (he : e ⬝ᵥ e = 1) (h : D *ᵥ e = c • e) : ActsAsScalarOn D e := by
  rw [ActsAsScalarOn, diffusionAlong_eq_of_mulVec_eq_smul he h, h]

omit [DecidableEq n] in
/-- A positive definite diffusion matrix has a strictly positive coefficient along every unit
direction. This is the ellipticity that the scalar statement needs in order to divide by
`σ_eff²`. -/
theorem diffusionAlong_pos {D : Matrix n n ℝ} {e : n → ℝ} (hD : D.PosDef) (he : e ⬝ᵥ e = 1) :
    0 < diffusionAlong D e := by
  have he0 : e ≠ 0 := by
    intro h
    rw [h] at he
    simp at he
  simpa [diffusionAlong] using hD.dotProduct_mulVec_pos he0

/-- A **unit vector on which `D` acts as a scalar with coefficient zero is annihilated by `D`**,
so an invertible `D` cannot have a vanishing coefficient along such a direction. This is why the
non-degeneracy of the projected diffusion is not an extra hypothesis of the projection theorem:
ellipticity and the eigendirection property already supply it. -/
theorem diffusionAlong_ne_zero_of_actsAsScalarOn {D : Matrix n n ℝ} {e : n → ℝ}
    (he : e ⬝ᵥ e = 1) (hdet : IsUnit D.det) (hD : ActsAsScalarOn D e) :
    diffusionAlong D e ≠ 0 := by
  intro h0
  have he0 : e ≠ 0 := by
    intro h
    rw [h] at he
    simp at he
  have hDe : D *ᵥ e = 0 := by rw [hD, h0, zero_smul]
  have h1 : D⁻¹ *ᵥ (D *ᵥ e) = e := by
    rw [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mulVec]
  rw [hDe, Matrix.mulVec_zero] at h1
  exact he0 h1.symm

/-- On a direction that `D` scales, the inverse scales it by the reciprocal:
`D⁻¹ e = (eᵀ D e)⁻¹ e`. Invertibility of `D` and non-degeneracy of the coefficient are both
needed, and both are stated. -/
theorem inv_mulVec_of_actsAsScalarOn {D : Matrix n n ℝ} {e : n → ℝ} (hdet : IsUnit D.det)
    (hD : ActsAsScalarOn D e) (hne : diffusionAlong D e ≠ 0) :
    D⁻¹ *ᵥ e = (diffusionAlong D e)⁻¹ • e := by
  have h1 : D⁻¹ *ᵥ (D *ᵥ e) = e := by
    rw [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mulVec]
  calc D⁻¹ *ᵥ e = D⁻¹ *ᵥ ((diffusionAlong D e)⁻¹ • (D *ᵥ e)) := by
        rw [hD, smul_smul, inv_mul_cancel₀ hne, one_smul]
    _ = (diffusionAlong D e)⁻¹ • (D⁻¹ *ᵥ (D *ᵥ e)) := by rw [Matrix.mulVec_smul]
    _ = (diffusionAlong D e)⁻¹ • e := by rw [h1]

/-- Hence the diffusion coefficient of the inverse along an eigendirection is the reciprocal of
the coefficient of `D`: `eᵀ D⁻¹ e = (eᵀ D e)⁻¹`. This identity is the entire content of the
projection, and `exists_posDef_lt_diffusionAlong_inv` shows that it genuinely needs the
eigendirection hypothesis. -/
theorem diffusionAlong_inv_of_actsAsScalarOn {D : Matrix n n ℝ} {e : n → ℝ} (he : e ⬝ᵥ e = 1)
    (hdet : IsUnit D.det) (hD : ActsAsScalarOn D e) (hne : diffusionAlong D e ≠ 0) :
    diffusionAlong D⁻¹ e = (diffusionAlong D e)⁻¹ := by
  rw [diffusionAlong, inv_mulVec_of_actsAsScalarOn hdet hD hne, dotProduct_smul, he, smul_eq_mul,
    mul_one]

/-- The Freidlin–Wentzell integrand of a velocity along an eigendirection:
`(c e)ᵀ D⁻¹ (c e) = c² / (eᵀ D e)`, the one-dimensional integrand with diffusion `eᵀ D e`. -/
theorem quadForm_inv_smul_of_actsAsScalarOn {D : Matrix n n ℝ} {e : n → ℝ} (he : e ⬝ᵥ e = 1)
    (hdet : IsUnit D.det) (hD : ActsAsScalarOn D e) (hne : diffusionAlong D e ≠ 0) (c : ℝ) :
    (c • e) ⬝ᵥ (D⁻¹ *ᵥ (c • e)) = c ^ 2 / diffusionAlong D e := by
  rw [diffusionAlong_smul, diffusionAlong_inv_of_actsAsScalarOn he hdet hD hne, div_eq_mul_inv]

/-- The **Kantorovich residual identity**. For a positive definite `D` and a unit vector `e`, the
`D⁻¹`-quadratic form of the residual `w = e - (eᵀ D e)⁻¹ D e` — the part of `e` that the
eigendirection hypothesis would kill — is exactly the gap in the Cauchy–Schwarz bound:

`wᵀ D⁻¹ w = eᵀ D⁻¹ e - (eᵀ D e)⁻¹`.

Both halves of what the projection needs to know about a non-eigendirection follow from this one
identity: the inequality, because the left-hand side is non-negative, and the characterization of
its equality case, because `D⁻¹` is positive definite and so the left-hand side vanishes only at
`w = 0`. -/
theorem quadForm_inv_residual {D : Matrix n n ℝ} {e : n → ℝ} (hD : D.PosDef) (he : e ⬝ᵥ e = 1) :
    (e - (diffusionAlong D e)⁻¹ • (D *ᵥ e)) ⬝ᵥ
        (D⁻¹ *ᵥ (e - (diffusionAlong D e)⁻¹ • (D *ᵥ e)))
      = diffusionAlong D⁻¹ e - (diffusionAlong D e)⁻¹ := by
  have hlam : 0 < diffusionAlong D e := diffusionAlong_pos hD he
  have hdet : IsUnit D.det := (Matrix.isUnit_iff_isUnit_det D).mp hD.isUnit
  set A : Matrix n n ℝ := D⁻¹ with hA
  have hAps : A.PosSemidef := (hD.inv).posSemidef
  have hAsymm : Aᵀ = A := by simpa using hAps.isHermitian.eq
  have hADe : A *ᵥ (D *ᵥ e) = e := by
    rw [Matrix.mulVec_mulVec, hA, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mulVec]
  set t : ℝ := (diffusionAlong D e)⁻¹ with ht
  have h2 : e ⬝ᵥ (A *ᵥ (D *ᵥ e)) = 1 := by rw [hADe, he]
  have h3 : (D *ᵥ e) ⬝ᵥ (A *ᵥ e) = 1 := by
    rw [dotProduct_mulVec_comm_of_transpose_eq hAsymm, hADe, he]
  have h4 : (D *ᵥ e) ⬝ᵥ (A *ᵥ (D *ᵥ e)) = diffusionAlong D e := by
    rw [hADe, dotProduct_comm, diffusionAlong]
  have hexp : (e - t • (D *ᵥ e)) ⬝ᵥ (A *ᵥ (e - t • (D *ᵥ e)))
      = diffusionAlong A e - 2 * t + t ^ 2 * diffusionAlong D e := by
    simp only [Matrix.mulVec_sub, Matrix.mulVec_smul, sub_dotProduct, dotProduct_sub,
      smul_dotProduct, dotProduct_smul, smul_eq_mul, h2, h3, h4]
    simp only [diffusionAlong]
    ring
  have hts : t ^ 2 * diffusionAlong D e = t := by
    rw [ht]; field_simp
  rw [hexp, hts]
  ring

/-- **What survives without the eigendirection hypothesis.** For a positive definite `D` and a
unit vector `e`, `1 / (eᵀ D e) ≤ eᵀ D⁻¹ e`. This is the Cauchy–Schwarz (Kantorovich) bound, and it
fixes the direction in which the scalar reduction can err on a line-confined path: the scalar
integrand never exceeds the multi-dimensional one. Its equality case is
`diffusionAlong_inv_eq_iff_actsAsScalarOn`, and the inequality is strict for the explicit witness
of `exists_posDef_lt_diffusionAlong_inv`. -/
theorem one_div_diffusionAlong_le_diffusionAlong_inv {D : Matrix n n ℝ} {e : n → ℝ}
    (hD : D.PosDef) (he : e ⬝ᵥ e = 1) : 1 / diffusionAlong D e ≤ diffusionAlong D⁻¹ e := by
  have hres := quadForm_inv_residual hD he
  have hAps : (D⁻¹).PosSemidef := (hD.inv).posSemidef
  set w : n → ℝ := e - (diffusionAlong D e)⁻¹ • (D *ᵥ e) with hwdef
  have key : 0 ≤ w ⬝ᵥ (D⁻¹ *ᵥ w) := by simpa using hAps.dotProduct_mulVec_nonneg w
  rw [hres] at key
  rw [one_div]
  linarith

/-- **The equality case of the Cauchy–Schwarz bound is exactly the eigendirection hypothesis.**
For a positive definite `D` and a unit vector `e`,

`eᵀ D⁻¹ e = (eᵀ D e)⁻¹  ↔  D acts on ℝ e as the scalar eᵀ D e`.

So `ActsAsScalarOn` is not merely sufficient for the projection of the Freidlin–Wentzell
integrand to be exact — it is necessary and sufficient for the pointwise identity the projection
rests on. The forward direction is the positive definiteness of `D⁻¹` applied to the Kantorovich
residual; the reverse is `diffusionAlong_inv_of_actsAsScalarOn`. -/
theorem diffusionAlong_inv_eq_iff_actsAsScalarOn {D : Matrix n n ℝ} {e : n → ℝ}
    (hD : D.PosDef) (he : e ⬝ᵥ e = 1) :
    diffusionAlong D⁻¹ e = (diffusionAlong D e)⁻¹ ↔ ActsAsScalarOn D e := by
  have hlam : 0 < diffusionAlong D e := diffusionAlong_pos hD he
  have hdet : IsUnit D.det := (Matrix.isUnit_iff_isUnit_det D).mp hD.isUnit
  constructor
  · intro heq
    have hres := quadForm_inv_residual hD he
    rw [heq, sub_self] at hres
    set w : n → ℝ := e - (diffusionAlong D e)⁻¹ • (D *ᵥ e) with hwdef
    have hw : w = 0 := by
      by_contra hne
      have hpos : 0 < w ⬝ᵥ (D⁻¹ *ᵥ w) := by
        simpa using (hD.inv).dotProduct_mulVec_pos hne
      linarith
    have he' : e = (diffusionAlong D e)⁻¹ • (D *ᵥ e) := by
      have : e - (diffusionAlong D e)⁻¹ • (D *ᵥ e) = 0 := by rw [← hwdef]; exact hw
      exact sub_eq_zero.mp this
    have hkey : diffusionAlong D e • e = D *ᵥ e := by
      calc diffusionAlong D e • e
          = diffusionAlong D e • ((diffusionAlong D e)⁻¹ • (D *ᵥ e)) := by rw [← he']
        _ = D *ᵥ e := by rw [smul_smul, mul_inv_cancel₀ hlam.ne', one_smul]
    exact hkey.symm
  · intro hsc
    exact diffusionAlong_inv_of_actsAsScalarOn he hdet hsc hlam.ne'

end Direction

/-! ### The eigendirection hypothesis is not removable -/

/-- The **tilted witness**, the positive definite matrix `!![1, 1; 1, 2]`. It is used twice below:
first to show that the identity `eᵀ D⁻¹ e = (eᵀ D e)⁻¹` fails when `e` is not an eigenvector, and
then, once the action has been defined, to show that the projection of the *action* fails for the
same reason. -/
def tiltedWitness : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 1, 2]

/-- The direction the tilted witness is tested against: the first basis vector `(1, 0)`, a unit
vector which is not an eigenvector of `tiltedWitness`. -/
def tiltedWitnessDirection : Fin 2 → ℝ := ![1, 0]

/-- The tested direction is a unit vector. -/
theorem tiltedWitnessDirection_dotProduct_self :
    tiltedWitnessDirection ⬝ᵥ tiltedWitnessDirection = 1 := by
  simp [tiltedWitnessDirection, dotProduct, Fin.sum_univ_two]

/-- The tilted witness is positive definite, so it is an admissible — indeed uniformly elliptic —
diffusion matrix, and the failure below is not a failure of ellipticity. -/
theorem tiltedWitness_posDef : tiltedWitness.PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · rw [tiltedWitness, Matrix.IsHermitian]
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  · intro x hx
    have hx' : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
      by_contra hcon
      simp only [not_or, ne_eq, not_not] at hcon
      exact hx (by ext i; fin_cases i <;> simp [hcon.1, hcon.2])
    have hval : star x ⬝ᵥ (tiltedWitness *ᵥ x) = (x 0 + x 1) ^ 2 + x 1 ^ 2 := by
      simp [tiltedWitness, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
      ring
    rw [hval]
    rcases hx' with h0 | h1
    · rcases eq_or_ne (x 0 + x 1) 0 with hsum | hsum
      · have : x 1 ≠ 0 := by
          intro h1
          rw [h1, add_zero] at hsum
          exact h0 hsum
        positivity
      · positivity
    · positivity

/-- The tilted witness has determinant one, and its inverse is `!![2, -1; -1, 1]`. -/
theorem tiltedWitness_inv : tiltedWitness⁻¹ = !![2, -1; -1, 1] := by
  refine Matrix.inv_eq_right_inv ?_
  rw [tiltedWitness]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_two]

/-- The diffusion coefficient of the tilted witness along the tested direction is `1`. -/
theorem diffusionAlong_tiltedWitness : diffusionAlong tiltedWitness tiltedWitnessDirection = 1 := by
  simp [diffusionAlong, tiltedWitness, tiltedWitnessDirection, dotProduct, Matrix.mulVec,
    Fin.sum_univ_two]

/-- The diffusion coefficient of the *inverse* of the tilted witness along the same direction is
`2`, not the reciprocal `1` that an eigendirection would give. -/
theorem diffusionAlong_inv_tiltedWitness :
    diffusionAlong tiltedWitness⁻¹ tiltedWitnessDirection = 2 := by
  rw [diffusionAlong, tiltedWitness_inv]
  simp [tiltedWitnessDirection, dotProduct, Matrix.mulVec, Fin.sum_univ_two]

/-- **Sharpness of the Cauchy–Schwarz bound.** There is a positive definite `2 × 2` matrix and a
unit vector for which the identity `eᵀ D⁻¹ e = (eᵀ D e)⁻¹` fails, in the direction the
Cauchy–Schwarz bound predicts. Take `D = !![1, 1; 1, 2]`, whose determinant is one and whose
inverse is `!![2, -1; -1, 1]`, and `e = (1, 0)`: then `eᵀ D e = 1` while `eᵀ D⁻¹ e = 2`. So the
hypothesis `ActsAsScalarOn` cannot be dropped in favour of positive definiteness or ellipticity
alone. That this failure of the linear-algebra identity really does break the projection of the
*action* is `exists_alongLine_fwActionScalar_lt_fwActionMulti`, proved below on the same
witness. -/
theorem exists_posDef_lt_diffusionAlong_inv :
    ∃ (D : Matrix (Fin 2) (Fin 2) ℝ) (e : Fin 2 → ℝ),
      D.PosDef ∧ e ⬝ᵥ e = 1 ∧ (diffusionAlong D e)⁻¹ < diffusionAlong D⁻¹ e :=
  ⟨tiltedWitness, tiltedWitnessDirection, tiltedWitness_posDef,
    tiltedWitnessDirection_dotProduct_self, by
      rw [diffusionAlong_tiltedWitness, diffusionAlong_inv_tiltedWitness]; norm_num⟩

/-- **The tested direction is genuinely not an eigendirection.** Stated as the negation of the
hypothesis the projection theorem carries, so that the witness below is visibly a counterexample
to dropping it rather than to something weaker. -/
theorem not_actsAsScalarOn_tiltedWitness :
    ¬ ActsAsScalarOn tiltedWitness tiltedWitnessDirection := by
  intro hsc
  have h := (diffusionAlong_inv_eq_iff_actsAsScalarOn tiltedWitness_posDef
    tiltedWitnessDirection_dotProduct_self).mpr hsc
  rw [diffusionAlong_tiltedWitness, diffusionAlong_inv_tiltedWitness] at h
  norm_num at h

/-! ### Paths confined to a line -/

section Paths

variable {n : Type*} [Fintype n]

/-- The path `t ↦ γ₀ + q(t) e` confined to the line through `γ₀` in the direction `e`. In the
application `e` is the bias-channel direction `e_b` and `q` is the scalar coordinate `w` of §3. -/
def alongLine (gamma0 e : n → ℝ) (q : ℝ → ℝ) : ℝ → (n → ℝ) := fun t => gamma0 + q t • e

/-- The velocity of a line-confined path is the scalar velocity times the direction. -/
theorem hasDerivAt_alongLine {q : ℝ → ℝ} {q' t : ℝ} (h : HasDerivAt q q' t) (gamma0 e : n → ℝ) :
    HasDerivAt (alongLine gamma0 e q) (q' • e) t :=
  (h.smul_const e).const_add gamma0

/-- The same in `deriv` form, for a path differentiable at the point in question. -/
theorem deriv_alongLine {q : ℝ → ℝ} {t : ℝ} (h : DifferentiableAt ℝ q t) (gamma0 e : n → ℝ) :
    deriv (alongLine gamma0 e q) t = deriv q t • e :=
  (hasDerivAt_alongLine h.hasDerivAt gamma0 e).deriv

end Paths

/-! ### The Freidlin–Wentzell action, in `n` dimensions and in one -/

section Action

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The integrand of the multi-dimensional Freidlin–Wentzell action at time `t`:
`(γ̇(t) - μ(γ(t)))ᵀ D(γ(t))⁻¹ (γ̇(t) - μ(γ(t)))`, for the drift field `μ` and the state-dependent
diffusion matrix field `D`. In the application `μ = -∇L̃` and `D = Σ_eff`. -/
noncomputable def fwIntegrandMulti (D : (n → ℝ) → Matrix n n ℝ) (mu : (n → ℝ) → (n → ℝ))
    (gamma : ℝ → (n → ℝ)) (t : ℝ) : ℝ :=
  (deriv gamma t - mu (gamma t)) ⬝ᵥ ((D (gamma t))⁻¹ *ᵥ (deriv gamma t - mu (gamma t)))

/-- The **multi-dimensional Freidlin–Wentzell action** of a path on `[0, T]`, in the standard form
`S_T[γ] = (1/2) ∫₀ᵀ (γ̇ - μ(γ))ᵀ D(γ)⁻¹ (γ̇ - μ(γ)) dt`.

This is a definition, not a derivation: mathlib v4.31.0 contains no large-deviations theory, so
the identification of this functional as the rate function of `dφ = μ(φ) dt + √η D(φ)^{1/2} dB` is
the citation to Freidlin and Wentzell that `docs/design/lemma1_rederivation.md` §7 already carries,
and it is an input to this file rather than one of its results. The constant is `1/2`, the
constant belonging to that normalization of the noise; see the file docstring on §7's `1/4`.

The velocity is `deriv gamma`, so the action is a functional of the path alone.

Three of Lean's conventions are visible in this definition and none of them is the mathematics.
For a path that fails to be differentiable, or an integrand that fails to be integrable, the value
recorded is zero rather than `+∞`. For a *singular* `D(γ(t))` the value recorded is likewise zero,
because `Matrix.inv` returns the zero matrix on a non-unit determinant, whereas the
Freidlin–Wentzell cost of moving transversally to a degenerate diffusion is infinite; this is
worth naming because (FW-3) of §7 is precisely about the degeneracy of `Σ_eff^direct` as `s → 0`,
and nothing in this file speaks to that limit. And for `T < 0` the expression is an oriented
integral rather than an action, so it is the action of a path only when `0 ≤ T`. Every theorem
below that uses the value hypothesizes the regularity, the invertibility, or the sign of `T` that
it needs; where a theorem omits one of these — `fwActionScalar_eq_two_mul_barrier_div` and
`fwActionMulti_alongLine_barrier` do not require `0 ≤ T` — the identity is true as stated for
oriented integrals, and is a statement about an action only on the intended range. -/
noncomputable def fwActionMulti (D : (n → ℝ) → Matrix n n ℝ) (mu : (n → ℝ) → (n → ℝ)) (T : ℝ)
    (gamma : ℝ → (n → ℝ)) : ℝ :=
  (1 / 2) * ∫ t in (0 : ℝ)..T, fwIntegrandMulti D mu gamma t

end Action

/-- The integrand of the one-dimensional Freidlin–Wentzell action at time `t`,
`(q̇(t) - m(q(t)))² / d(q(t))`, for the scalar drift `m` and the scalar diffusion `d`. -/
noncomputable def fwIntegrandScalar (d m q : ℝ → ℝ) (t : ℝ) : ℝ :=
  (deriv q t - m (q t)) ^ 2 / d (q t)

/-- The **one-dimensional Freidlin–Wentzell action** of a path on `[0, T]`,
`(1/2) ∫₀ᵀ (q̇ - m(q))² / d(q) dt`. With `m = -L̃'` and `d = σ_eff²` this is
`QuasipotentialMonotone.fwAction (1/2) L̃' σ_eff² q (deriv q) T`, the action (FW-2) compares; the
present file does not import that module, and the correspondence is recorded rather than used. -/
noncomputable def fwActionScalar (d m : ℝ → ℝ) (T : ℝ) (q : ℝ → ℝ) : ℝ :=
  (1 / 2) * ∫ t in (0 : ℝ)..T, fwIntegrandScalar d m q t

/-- The scalar action sees the diffusion only through its values along the path, so two diffusion
profiles agreeing there give the same action. This is what turns the band hypothesis
"`σ_eff²` equals `c` on the band" into a statement about a constant-diffusion action. -/
theorem fwActionScalar_congr_diffusion {d₁ d₂ m q : ℝ → ℝ} {T : ℝ}
    (h : ∀ t ∈ uIcc (0 : ℝ) T, d₁ (q t) = d₂ (q t)) :
    fwActionScalar d₁ m T q = fwActionScalar d₂ m T q := by
  unfold fwActionScalar
  congr 1
  refine intervalIntegral.integral_congr (fun t ht => ?_)
  unfold fwIntegrandScalar
  rw [h t ht]

/-! ### The projection -/

section Projection

variable {n : Type*} [Fintype n] [DecidableEq n]
variable {D : (n → ℝ) → Matrix n n ℝ} {mu : (n → ℝ) → (n → ℝ)} {gamma0 e : n → ℝ}
variable {q m : ℝ → ℝ} {T : ℝ}

/-- **The projection, (FW-4)'s reduction step.** Let `e` be a unit vector, let the path be
confined to the line `{γ₀ + r e}`, let the drift be tangent to that line along the path, and let
the diffusion matrix act on that line as the scalar `eᵀ D e` at every point the path visits. Then
the multi-dimensional Freidlin–Wentzell action of the path is *exactly* the one-dimensional action
of its scalar coordinate `q`, with scalar diffusion `r ↦ eᵀ D(γ₀ + r e) e`.

Nothing is approximated and no integrability is required: the two integrands are equal at every
point of `uIcc 0 T`, by `quadForm_inv_smul_of_actsAsScalarOn`, so the two interval integrals agree
whatever their integrability status. This is the sense in which the one-dimensional statement (L2')
of §3 is the projection of the multi-dimensional programme of §7 and not a different theorem.

Non-degeneracy of the projected diffusion is not assumed: it follows from the eigendirection
hypothesis together with invertibility, by `diffusionAlong_ne_zero_of_actsAsScalarOn`. The
eigendirection hypothesis itself is not removable —
`exists_alongLine_fwActionScalar_lt_fwActionMulti` exhibits an instance satisfying every other
hypothesis, with a positive definite constant diffusion, on which the conclusion is false.

It is *not* a statement about minimizers. The path is given, and confined to the line by
hypothesis; that the minimizer of the multi-dimensional problem is so confined is a separate
variational claim which is not proved anywhere in this development. -/
theorem fwActionMulti_alongLine (he : e ⬝ᵥ e = 1)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, DifferentiableAt ℝ q t)
    (hdet : ∀ t ∈ uIcc (0 : ℝ) T, IsUnit (D (alongLine gamma0 e q t)).det)
    (hscal : ∀ t ∈ uIcc (0 : ℝ) T, ActsAsScalarOn (D (alongLine gamma0 e q t)) e)
    (hmu : ∀ t ∈ uIcc (0 : ℝ) T, mu (alongLine gamma0 e q t) = m (q t) • e) :
    fwActionMulti D mu T (alongLine gamma0 e q)
      = fwActionScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e) m T q := by
  unfold fwActionMulti fwActionScalar
  congr 1
  refine intervalIntegral.integral_congr (fun t ht => ?_)
  have hne : diffusionAlong (D (alongLine gamma0 e q t)) e ≠ 0 :=
    diffusionAlong_ne_zero_of_actsAsScalarOn he (hdet t ht) (hscal t ht)
  have hd : deriv (alongLine gamma0 e q) t = deriv q t • e := deriv_alongLine (hq t ht) _ _
  have hv : deriv (alongLine gamma0 e q) t - mu (alongLine gamma0 e q t)
      = (deriv q t - m (q t)) • e := by
    rw [hd, hmu t ht, ← sub_smul]
  unfold fwIntegrandMulti fwIntegrandScalar
  rw [hv, quadForm_inv_smul_of_actsAsScalarOn he (hdet t ht) (hscal t ht) hne]
  rfl

/-- **Without the eigendirection hypothesis, pointwise.** For a positive definite diffusion matrix
the one-dimensional integrand at diffusion `eᵀ D e` is a lower bound for the multi-dimensional
integrand of the line-confined path, by the Cauchy–Schwarz bound
`1/(eᵀ D e) ≤ eᵀ D⁻¹ e`. Equality holds precisely on eigendirections, by
`diffusionAlong_inv_eq_iff_actsAsScalarOn`, which is where `fwActionMulti_alongLine` applies; off
them the inequality is strict, and `exists_alongLine_fwActionScalar_lt_fwActionMulti` exhibits an
instance where it is. -/
theorem fwIntegrandScalar_le_fwIntegrandMulti_alongLine {t : ℝ} (he : e ⬝ᵥ e = 1)
    (hq : DifferentiableAt ℝ q t) (hpos : (D (alongLine gamma0 e q t)).PosDef)
    (hmu : mu (alongLine gamma0 e q t) = m (q t) • e) :
    fwIntegrandScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e) m q t
      ≤ fwIntegrandMulti D mu (alongLine gamma0 e q) t := by
  have hd : deriv (alongLine gamma0 e q) t = deriv q t • e := deriv_alongLine hq _ _
  have hv : deriv (alongLine gamma0 e q) t - mu (alongLine gamma0 e q t)
      = (deriv q t - m (q t)) • e := by
    rw [hd, hmu, ← sub_smul]
  have hcs : 1 / diffusionAlong (D (alongLine gamma0 e q t)) e
      ≤ diffusionAlong (D (alongLine gamma0 e q t))⁻¹ e :=
    one_div_diffusionAlong_le_diffusionAlong_inv hpos he
  have hsq : (0 : ℝ) ≤ (deriv q t - m (q t)) ^ 2 := sq_nonneg _
  unfold fwIntegrandMulti fwIntegrandScalar
  rw [hv, diffusionAlong_smul]
  have hrw : (deriv q t - m (q t)) ^ 2 / diffusionAlong (D (gamma0 + q t • e)) e
      = (deriv q t - m (q t)) ^ 2 * (1 / diffusionAlong (D (alongLine gamma0 e q t)) e) := by
    rw [alongLine]
    ring
  rw [hrw]
  exact mul_le_mul_of_nonneg_left hcs hsq

/-- **Without the eigendirection hypothesis, integrated.** The one-dimensional action at diffusion
`eᵀ Σ_eff e` never exceeds the multi-dimensional action of the same line-confined path: the scalar
reduction of (L2') can understate the cost of a line-confined crossing, never overstate it, and it
understates it exactly to the extent that the bias-channel direction fails to be an eigendirection
of the update covariance. The inequality is strict at a genuine instance, by
`exists_alongLine_fwActionScalar_lt_fwActionMulti`.

This is the honest residue of (FW-4) when the eigendirection hypothesis is dropped, and it should
not be read as making the scalar reduction safe for the paper's comparison. It bounds the two
methods' line-confined actions from below in the same direction, and a common lower bound orders
neither: the comparison the escape-rate argument needs is (FW-2)'s, between two multi-dimensional
actions of the same path, and it is made elsewhere. Nor is this a statement about minimizers. -/
theorem fwActionScalar_le_fwActionMulti_alongLine (hT : 0 ≤ T) (he : e ⬝ᵥ e = 1)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, DifferentiableAt ℝ q t)
    (hpos : ∀ t ∈ uIcc (0 : ℝ) T, (D (alongLine gamma0 e q t)).PosDef)
    (hmu : ∀ t ∈ uIcc (0 : ℝ) T, mu (alongLine gamma0 e q t) = m (q t) • e)
    (hint₁ : IntervalIntegrable
      (fwIntegrandScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e) m q) volume 0 T)
    (hint₂ : IntervalIntegrable (fwIntegrandMulti D mu (alongLine gamma0 e q)) volume 0 T) :
    fwActionScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e) m T q
      ≤ fwActionMulti D mu T (alongLine gamma0 e q) := by
  have hsub : Icc (0 : ℝ) T ⊆ uIcc (0 : ℝ) T := by
    rw [Set.uIcc_of_le hT]
  have hmono : (∫ t in (0 : ℝ)..T,
        fwIntegrandScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e) m q t)
      ≤ ∫ t in (0 : ℝ)..T, fwIntegrandMulti D mu (alongLine gamma0 e q) t := by
    refine intervalIntegral.integral_mono_on hT hint₁ hint₂ (fun t ht => ?_)
    exact fwIntegrandScalar_le_fwIntegrandMulti_alongLine he (hq t (hsub ht)) (hpos t (hsub ht))
      (hmu t (hsub ht))
  unfold fwActionScalar fwActionMulti
  linarith

/-- **The eigendirection hypothesis is not removable from the projection theorem itself.** On the
tilted witness `Σ = !![1, 1; 1, 2]` — constant in space, symmetric, positive definite, hence
uniformly elliptic — with the unit direction `e = (1, 0)`, zero drift, and the line-confined path
`t ↦ q(t) e` with `q(t) = t` on `[0, 1]`, every hypothesis of `fwActionMulti_alongLine` holds
except `ActsAsScalarOn`, and its conclusion fails: the scalar action is `1/2` while the
multi-dimensional action is `1`.

This is what `exists_posDef_lt_diffusionAlong_inv` means at the level of the functionals rather
than of the matrices, and it also shows that the inequality
`fwActionScalar_le_fwActionMulti_alongLine` is strict at a genuine instance, so neither that
inequality nor the eigendirection hypothesis can be improved to an identity in general. -/
theorem exists_alongLine_fwActionScalar_lt_fwActionMulti :
    ∃ (D : (Fin 2 → ℝ) → Matrix (Fin 2) (Fin 2) ℝ) (mu : (Fin 2 → ℝ) → (Fin 2 → ℝ))
      (gamma0 e : Fin 2 → ℝ) (q m : ℝ → ℝ) (T : ℝ),
      0 < T ∧ e ⬝ᵥ e = 1 ∧
        (∀ t ∈ uIcc (0 : ℝ) T, DifferentiableAt ℝ q t) ∧
        (∀ t ∈ uIcc (0 : ℝ) T, (D (alongLine gamma0 e q t)).PosDef) ∧
        (∀ t ∈ uIcc (0 : ℝ) T, IsUnit (D (alongLine gamma0 e q t)).det) ∧
        (∀ t ∈ uIcc (0 : ℝ) T, mu (alongLine gamma0 e q t) = m (q t) • e) ∧
        (∀ t ∈ uIcc (0 : ℝ) T, ¬ ActsAsScalarOn (D (alongLine gamma0 e q t)) e) ∧
        fwActionScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e) m T q
          < fwActionMulti D mu T (alongLine gamma0 e q) := by
  refine ⟨fun _ => tiltedWitness, fun _ => 0, 0, tiltedWitnessDirection, (fun t => t),
    (fun _ => 0), 1, one_pos, tiltedWitnessDirection_dotProduct_self,
    (fun t _ => differentiableAt_id), (fun t _ => tiltedWitness_posDef),
    (fun t _ => (Matrix.isUnit_iff_isUnit_det _).mp tiltedWitness_posDef.isUnit),
    (fun t _ => (zero_smul ℝ tiltedWitnessDirection).symm),
    (fun t _ => not_actsAsScalarOn_tiltedWitness), ?_⟩
  have hderiv : ∀ t : ℝ, deriv (fun x : ℝ => x) t = 1 := fun t => by simp
  have hs : fwActionScalar (fun _ => diffusionAlong tiltedWitness tiltedWitnessDirection)
      (fun _ => 0) 1 (fun t => t) = 1 / 2 := by
    unfold fwActionScalar
    have hcong : ∀ t ∈ uIcc (0 : ℝ) 1,
        fwIntegrandScalar (fun _ => diffusionAlong tiltedWitness tiltedWitnessDirection)
          (fun _ => 0) (fun t => t) t = 1 := by
      intro t _
      unfold fwIntegrandScalar
      rw [hderiv t, diffusionAlong_tiltedWitness]
      norm_num
    rw [intervalIntegral.integral_congr hcong, intervalIntegral.integral_const]
    norm_num
  have hm : fwActionMulti (fun _ => tiltedWitness) (fun _ => (0 : Fin 2 → ℝ)) 1
      (alongLine 0 tiltedWitnessDirection (fun t => t)) = 1 := by
    unfold fwActionMulti
    have hcong : ∀ t ∈ uIcc (0 : ℝ) 1,
        fwIntegrandMulti (fun _ => tiltedWitness) (fun _ => (0 : Fin 2 → ℝ))
          (alongLine 0 tiltedWitnessDirection (fun t => t)) t = 2 := by
      intro t _
      unfold fwIntegrandMulti
      rw [deriv_alongLine (show DifferentiableAt ℝ (fun x : ℝ => x) t from differentiableAt_id)
        0 tiltedWitnessDirection, hderiv t]
      simp only [sub_zero, one_smul]
      exact diffusionAlong_inv_tiltedWitness
    rw [intervalIntegral.integral_congr hcong, intervalIntegral.integral_const]
    norm_num
  rw [hs, hm]
  norm_num

end Projection

/-! ### The value of the reduced action on a barrier crossing -/

section Barrier

variable {c T : ℝ} {Ltilde dLtilde q : ℝ → ℝ}

/-- The **work identity**: the integral of `L̃'(q(t)) q̇(t)` over `[0, T]` is the change
`L̃(q(T)) - L̃(q(0))` in the pullback potential across the path. This is the chain rule together
with the fundamental theorem of calculus, and it is the only analytic input the two barrier
statements need. -/
theorem integral_potential_work_eq_barrier
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, DifferentiableAt ℝ q t)
    (hLtilde : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt Ltilde (dLtilde (q t)) (q t))
    (hintW : IntervalIntegrable (fun t => dLtilde (q t) * deriv q t) volume 0 T) :
    ∫ t in (0 : ℝ)..T, dLtilde (q t) * deriv q t = Ltilde (q T) - Ltilde (q 0) := by
  refine intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun t => Ltilde (q t))
    (fun t ht => ?_) hintW
  exact (hLtilde t ht).comp t (hq t ht).hasDerivAt

/-- **Every crossing costs at least `2 (α / c)`.** On a band where the projected diffusion is the
constant `c > 0`, the one-dimensional Freidlin–Wentzell action of *any* differentiable path with
the gradient drift `-L̃'` is at least `2 (α / c)`, where `α = L̃(q(T)) - L̃(q(0))` is the barrier
climbed by the path.

The proof is the elementary completion of the square `(a + b)² ≥ 4 a b` applied pointwise to
`a = q̇` and `b = L̃'(q)`, followed by the work identity; no calculus of variations is used and no
minimizer is exhibited. The effective action `α / c = α / σ_eff²` of (L2') appears literally on the
left, multiplied by the factor two that the Freidlin–Wentzell normalization contributes.

The barrier `α` is read off the path's own endpoints, so the statement is one inequality per path.
Read as a bound over a competitor class it is the statement that every admissible path from a
given `q(0)` to a given `q(T)` costs at least `2 (α / c)`, which is the lower half of the
quasipotential across that barrier. -/
theorem two_mul_barrier_div_le_fwActionScalar (hc : 0 < c) (hT : 0 ≤ T)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, DifferentiableAt ℝ q t)
    (hLtilde : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt Ltilde (dLtilde (q t)) (q t))
    (hint : IntervalIntegrable
      (fwIntegrandScalar (fun _ => c) (fun w => -dLtilde w) q) volume 0 T)
    (hintW : IntervalIntegrable (fun t => dLtilde (q t) * deriv q t) volume 0 T) :
    2 * ((Ltilde (q T) - Ltilde (q 0)) / c)
      ≤ fwActionScalar (fun _ => c) (fun w => -dLtilde w) T q := by
  have hmono : (∫ t in (0 : ℝ)..T, 4 / c * (dLtilde (q t) * deriv q t))
      ≤ ∫ t in (0 : ℝ)..T, fwIntegrandScalar (fun _ => c) (fun w => -dLtilde w) q t := by
    refine intervalIntegral.integral_mono_on hT (hintW.const_mul (4 / c)) hint (fun t _ => ?_)
    have hsq : 4 * (dLtilde (q t) * deriv q t) ≤ (deriv q t + dLtilde (q t)) ^ 2 := by
      nlinarith [sq_nonneg (deriv q t - dLtilde (q t))]
    have hfac : 4 / c * (dLtilde (q t) * deriv q t)
        = 4 * (dLtilde (q t) * deriv q t) / c := by ring
    rw [hfac]
    unfold fwIntegrandScalar
    have hrw : (deriv q t - -dLtilde (q t)) ^ 2 / c = (deriv q t + dLtilde (q t)) ^ 2 / c := by
      ring_nf
    rw [hrw]
    gcongr
  rw [intervalIntegral.integral_const_mul,
    integral_potential_work_eq_barrier hq hLtilde hintW] at hmono
  have hrw : 4 / c * (Ltilde (q T) - Ltilde (q 0))
      = 4 * ((Ltilde (q T) - Ltilde (q 0)) / c) := by ring
  rw [hrw] at hmono
  unfold fwActionScalar
  linarith

/-- **The reversed relaxation costs exactly `2 (α / c)`.** A path solving `q̇ = L̃'(q)` — the
classical Freidlin–Wentzell instanton for a one-dimensional gradient system, the relaxation flow
run backwards — has action exactly `2 (α / c)` on a band of constant diffusion `c ≠ 0`.

Together with `two_mul_barrier_div_le_fwActionScalar` this identifies the reduced quasipotential
across the barrier the path climbs: the bound proved there is `2 (α / c) ≤ S[q]` with `α` read off
the endpoint values `L̃(q(T))` and `L̃(q(0))` of the path itself, so it is a lower bound valid over
every admissible path sharing those two values — the competitor class of the quasipotential
`V(w_b, w_s)` — and the reversed relaxation attains it. Two qualifications the note's asymptotics
need are *not* proved here. A reversed relaxation connecting a critical point to the saddle exists
only in the limit `T → ∞`; the theorem is stated for whatever finite-time reversed-relaxation path
is supplied, so the infimum is identified in the limit rather than attained. And no infimum is
taken anywhere in this file, so "quasipotential" is here a name for the value `2 (α/c)` that the
two statements bracket, not a formalized infimum over a path space. -/
theorem fwActionScalar_eq_two_mul_barrier_div (hc : c ≠ 0)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt q (dLtilde (q t)) t)
    (hLtilde : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt Ltilde (dLtilde (q t)) (q t))
    (hint : IntervalIntegrable (fun t => dLtilde (q t) ^ 2) volume 0 T) :
    fwActionScalar (fun _ => c) (fun w => -dLtilde w) T q
      = 2 * ((Ltilde (q T) - Ltilde (q 0)) / c) := by
  have hwork : ∫ t in (0 : ℝ)..T, dLtilde (q t) ^ 2 = Ltilde (q T) - Ltilde (q 0) := by
    refine intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun t => Ltilde (q t))
      (fun t ht => ?_) hint
    have h := (hLtilde t ht).comp t (hq t ht)
    rw [← pow_two] at h
    exact h
  have hcongr : (∫ t in (0 : ℝ)..T, fwIntegrandScalar (fun _ => c) (fun w => -dLtilde w) q t)
      = ∫ t in (0 : ℝ)..T, 4 / c * dLtilde (q t) ^ 2 := by
    refine intervalIntegral.integral_congr (fun t ht => ?_)
    unfold fwIntegrandScalar
    rw [(hq t ht).deriv]
    field_simp
    ring
  unfold fwActionScalar
  rw [hcongr, intervalIntegral.integral_const_mul, hwork]
  field_simp
  ring

/-- **The normalization constant of §7, checked rather than argued.** §7 of
`docs/design/lemma1_rederivation.md` writes the Freidlin–Wentzell action with the constant `1/4`
and the exit asymptotic with `E[τ] ≍ exp(V*/η)`. Halving the action of this file — which carries
`1/2` — gives the quarter-normalized action, and on the reversed relaxation across a
constant-diffusion band its value is

`(1/4) ∫₀ᵀ (q̇ + L̃'(q))² / c dt = α / c`,   `α = L̃(q(T)) - L̃(q(0))`.

That is (L2')'s effective action `α_eff = α / σ_eff²` on the nose, but divided by `η` it gives the
exponent `α / (η σ_eff²)`, which is one half of the exponent `2α / (η σ_eff²)` that (L2') states
three sections earlier. The note's two displays are therefore inconsistent by a factor of two, and
`1/2` — the constant belonging to the noise normalization `√η Σ_eff^{1/2} dB` that §7 itself
writes — is the one that reconciles them; compare `fwActionScalar_eq_two_mul_barrier_div`, whose
value `2 (α/c)` divided by `η` is (L2')'s exponent. This file therefore carries `1/2` throughout,
and this theorem records what the alternative would have given. -/
theorem quarter_normalized_fwActionScalar_eq_barrier_div (hc : c ≠ 0)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt q (dLtilde (q t)) t)
    (hLtilde : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt Ltilde (dLtilde (q t)) (q t))
    (hint : IntervalIntegrable (fun t => dLtilde (q t) ^ 2) volume 0 T) :
    (1 / 2) * fwActionScalar (fun _ => c) (fun w => -dLtilde w) T q
      = (Ltilde (q T) - Ltilde (q 0)) / c := by
  rw [fwActionScalar_eq_two_mul_barrier_div hc hq hLtilde hint]
  ring

end Barrier

/-! ### (FW-4): the multi-dimensional action of a bias-channel crossing -/

section Consistency

variable {n : Type*} [Fintype n] [DecidableEq n]
variable {D : (n → ℝ) → Matrix n n ℝ} {mu : (n → ℝ) → (n → ℝ)} {gamma0 e : n → ℝ}
variable {Ltilde dLtilde q : ℝ → ℝ} {c T : ℝ}

/-- **(FW-4), assembled.** Let `e` be the unit bias-channel direction, let the diffusion matrix
`Σ_eff` act on the bias-channel line as a scalar and take the constant value `c` there along the
path — the shoulder band of (A2) — let the drift be the tangential gradient drift `-L̃'`, and let
the path be the line-confined reversed relaxation `q̇ = L̃'(q)`. Then the *multi-dimensional*
Freidlin–Wentzell action of that path is

`S = 2 * (α / c)`,   `α = L̃(q(T)) - L̃(q(0))`,   `c = eᵀ Σ_eff e`,

whose factor `α / c` is precisely the effective action `α_eff = α / σ_eff²` that (L2') of §3
assigns to the crossing, and which `BarrierRescaling.quasipotential_gap_of_const_diffusion`
obtains, from the stationary density (*) rather than from an action, as the quasipotential gap
`U_eff(w_s) - U_eff(w_b)`. The two levels of the paper therefore agree, which is what (FW-4) was
introduced to check.

The reduction is exact, but it is a reduction and not a minimization: the confinement of the path
to the bias-channel line is a hypothesis here, and the claim that the multi-dimensional minimizer
is so confined is neither proved nor used.

Two readings to guard against. The hypotheses are not vacuous and not satisfiable only by a
constant path across a zero barrier: `fwActionMulti_barrierWitness` instantiates all of them in an
elliptic two-dimensional gradient system and computes a strictly positive action. And `T` is not
required to be non-negative, because the identity is true for the oriented integral either way;
it is a statement about an action only for `0 ≤ T`. -/
theorem fwActionMulti_alongLine_barrier (he : e ⬝ᵥ e = 1) (hc : c ≠ 0)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt q (dLtilde (q t)) t)
    (hLtilde : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt Ltilde (dLtilde (q t)) (q t))
    (hint : IntervalIntegrable (fun t => dLtilde (q t) ^ 2) volume 0 T)
    (hdet : ∀ t ∈ uIcc (0 : ℝ) T, IsUnit (D (alongLine gamma0 e q t)).det)
    (hscal : ∀ t ∈ uIcc (0 : ℝ) T, ActsAsScalarOn (D (alongLine gamma0 e q t)) e)
    (hband : ∀ t ∈ uIcc (0 : ℝ) T, diffusionAlong (D (alongLine gamma0 e q t)) e = c)
    (hmu : ∀ t ∈ uIcc (0 : ℝ) T, mu (alongLine gamma0 e q t) = (-dLtilde (q t)) • e) :
    fwActionMulti D mu T (alongLine gamma0 e q)
      = 2 * ((Ltilde (q T) - Ltilde (q 0)) / c) := by
  have hproj := fwActionMulti_alongLine (m := fun w => -dLtilde w) he
    (fun t ht => (hq t ht).differentiableAt) hdet hscal hmu
  have hconst : fwActionScalar (fun r => diffusionAlong (D (gamma0 + r • e)) e)
      (fun w => -dLtilde w) T q
      = fwActionScalar (fun _ => c) (fun w => -dLtilde w) T q :=
    fwActionScalar_congr_diffusion (fun t ht => hband t ht)
  rw [hproj, hconst, fwActionScalar_eq_two_mul_barrier_div hc hq hLtilde hint]

/-- **(FW-4) in Arrhenius form.** Dividing the action of `fwActionMulti_alongLine_barrier` by the
learning rate `η` gives `2 α / (η c)`, which is `BarrierRescaling.arrheniusExponent α η c`
verbatim: the exponent in which the corrected Corollary 1 compares the direct baseline's effective
diffusion `s² σ_obj²` against the lift's `s² σ_obj² + c₁ σ_Jac² |H| + s² (HVH)`.

This is an identity between the projected action and the exponent, and nothing more. It does not
prove the Freidlin–Wentzell exit asymptotic `E[τ] ≍ exp(V*/η)` that would turn the exponent into a
mean first-passage time; that asymptotic has no formal content in mathlib v4.31.0 and is carried
as a hypothesis in `KramersExitTime.lean`. Prefactors are nowhere addressed. And the word
"verbatim" above records a reading of `BarrierRescaling.arrheniusExponent`, whose body is
`2 * alpha / (eta * sigma2)`, and not an imported identity: this file imports only `Mathlib`, so
the agreement of the two right-hand sides is checked by eye and not by the kernel. -/
theorem fwActionMulti_alongLine_arrheniusExponent {eta : ℝ} (heta : eta ≠ 0) (he : e ⬝ᵥ e = 1)
    (hc : c ≠ 0)
    (hq : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt q (dLtilde (q t)) t)
    (hLtilde : ∀ t ∈ uIcc (0 : ℝ) T, HasDerivAt Ltilde (dLtilde (q t)) (q t))
    (hint : IntervalIntegrable (fun t => dLtilde (q t) ^ 2) volume 0 T)
    (hdet : ∀ t ∈ uIcc (0 : ℝ) T, IsUnit (D (alongLine gamma0 e q t)).det)
    (hscal : ∀ t ∈ uIcc (0 : ℝ) T, ActsAsScalarOn (D (alongLine gamma0 e q t)) e)
    (hband : ∀ t ∈ uIcc (0 : ℝ) T, diffusionAlong (D (alongLine gamma0 e q t)) e = c)
    (hmu : ∀ t ∈ uIcc (0 : ℝ) T, mu (alongLine gamma0 e q t) = (-dLtilde (q t)) • e) :
    fwActionMulti D mu T (alongLine gamma0 e q) / eta
      = 2 * (Ltilde (q T) - Ltilde (q 0)) / (eta * c) := by
  rw [fwActionMulti_alongLine_barrier he hc hq hLtilde hint hdet hscal hband hmu]
  field_simp

end Consistency

/-! ### A witness: (FW-4)'s hypotheses are satisfiable with a strictly positive barrier -/

section BarrierWitness

variable {c T : ℝ}

/-- The bias-channel direction of the witness, the first basis vector of `ℝ²`. -/
def barrierWitnessDirection : Fin 2 → ℝ := ![1, 0]

/-- The witness diffusion field: the constant diagonal matrix `!![c, 0; 0, 1]`, positive definite
for `c > 0` and hence uniformly elliptic, which acts on the bias-channel line as the scalar `c`. -/
def barrierWitnessDiffusion (c : ℝ) : (Fin 2 → ℝ) → Matrix (Fin 2) (Fin 2) ℝ :=
  fun _ => !![c, 0; 0, 1]

/-- The witness potential on `ℝ²`, `Φ(x) = x₀² / 2`. Its restriction to the bias-channel line is
the scalar potential `L̃(w) = w² / 2`, by `barrierWitnessPotential_alongLine`. -/
noncomputable def barrierWitnessPotential : (Fin 2 → ℝ) → ℝ := fun x => x 0 ^ 2 / 2

/-- The witness drift field, `μ(x) = (-x₀, 0)`. This is the exact gradient drift `-∇Φ` of the
witness potential, as `hasDerivAt_barrierWitnessPotential_update` records, so the instance below is
a Freidlin–Wentzell gradient system and not merely a formal solution of the hypotheses. -/
def barrierWitnessDrift : (Fin 2 → ℝ) → (Fin 2 → ℝ) := fun x => ![-(x 0), 0]

/-- The witness direction is a unit vector. -/
theorem barrierWitnessDirection_dotProduct_self :
    barrierWitnessDirection ⬝ᵥ barrierWitnessDirection = 1 := by
  simp [barrierWitnessDirection, dotProduct, Fin.sum_univ_two]

/-- The witness diffusion is positive definite whenever `c > 0`, so the instance satisfies the
ellipticity that (FW-3) supplies and not merely the invertibility the projection theorem asks
for. -/
theorem barrierWitnessDiffusion_posDef (hc : 0 < c) (x : Fin 2 → ℝ) :
    (barrierWitnessDiffusion c x).PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · rw [barrierWitnessDiffusion, Matrix.IsHermitian]
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  · intro y hy
    have hy' : y 0 ≠ 0 ∨ y 1 ≠ 0 := by
      by_contra hcon
      simp only [not_or, ne_eq, not_not] at hcon
      exact hy (by ext i; fin_cases i <;> simp [hcon.1, hcon.2])
    have hval : star y ⬝ᵥ (barrierWitnessDiffusion c x *ᵥ y) = c * y 0 ^ 2 + y 1 ^ 2 := by
      simp [barrierWitnessDiffusion, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
      ring
    rw [hval]
    rcases hy' with h0 | h1
    · have : 0 < c * y 0 ^ 2 := by positivity
      nlinarith [sq_nonneg (y 1)]
    · have : 0 < y 1 ^ 2 := by positivity
      nlinarith [sq_nonneg (y 0), hc.le]

/-- The witness diffusion coefficient along the bias-channel direction is the constant `c`; this
discharges the band hypothesis `hband` exactly rather than approximately. -/
theorem diffusionAlong_barrierWitnessDiffusion (x : Fin 2 → ℝ) :
    diffusionAlong (barrierWitnessDiffusion c x) barrierWitnessDirection = c := by
  simp [diffusionAlong, barrierWitnessDiffusion, barrierWitnessDirection, dotProduct,
    Matrix.mulVec, Fin.sum_univ_two]

/-- The witness direction is an eigendirection of the witness diffusion, which discharges the
substantive hypothesis `ActsAsScalarOn`. -/
theorem actsAsScalarOn_barrierWitnessDiffusion (x : Fin 2 → ℝ) :
    ActsAsScalarOn (barrierWitnessDiffusion c x) barrierWitnessDirection := by
  refine actsAsScalarOn_of_mulVec_eq_smul (c := c) barrierWitnessDirection_dotProduct_self ?_
  ext i
  fin_cases i <;>
    simp [barrierWitnessDiffusion, barrierWitnessDirection, Matrix.mulVec, dotProduct,
      Fin.sum_univ_two]

/-- The witness diffusion is invertible for `c ≠ 0`, its determinant being `c`. -/
theorem isUnit_det_barrierWitnessDiffusion (hc : c ≠ 0) (x : Fin 2 → ℝ) :
    IsUnit (barrierWitnessDiffusion c x).det := by
  have hdet : (barrierWitnessDiffusion c x).det = c := by
    simp [barrierWitnessDiffusion, Matrix.det_fin_two_of]
  rw [hdet]
  exact isUnit_iff_ne_zero.mpr hc

/-- The witness scalar potential `L̃(w) = w² / 2` has derivative `L̃'(w) = w`. -/
theorem hasDerivAt_barrierWitnessScalarPotential (x : ℝ) :
    HasDerivAt (fun w : ℝ => w ^ 2 / 2) x x := by
  simpa using (hasDerivAt_pow 2 x).div_const 2

/-- **The witness drift is a genuine gradient drift.** Coordinate by coordinate,
`∂Φ/∂xᵢ (x) = -μᵢ(x)` for the witness potential `Φ(x) = x₀²/2` and the witness drift
`μ(x) = (-x₀, 0)`. The instance below is therefore an instance of the Itô diffusion
`dφ = -∇L̃(φ) dt + √η Σ_eff(φ)^{1/2} dB` of §7 with `L̃ = Φ` and
`Σ_eff = barrierWitnessDiffusion c`, and not an artefact of hypothesizing tangency directly. -/
theorem hasDerivAt_barrierWitnessPotential_update (x : Fin 2 → ℝ) (i : Fin 2) :
    HasDerivAt (fun s => barrierWitnessPotential (Function.update x i s))
      (-(barrierWitnessDrift x i)) (x i) := by
  by_cases hi : i = 0
  · subst hi
    have hup : (fun s : ℝ => barrierWitnessPotential (Function.update x 0 s))
        = fun s : ℝ => s ^ 2 / 2 := by
      funext s
      simp [barrierWitnessPotential]
    rw [hup]
    have hdr : -(barrierWitnessDrift x 0) = x 0 := by
      simp [barrierWitnessDrift]
    rw [hdr]
    exact hasDerivAt_barrierWitnessScalarPotential (x 0)
  · have hone : i = 1 := by
      refine Fin.ext ?_
      have h2 : i.val < 2 := i.isLt
      have h0 : i.val ≠ 0 := fun h => hi (Fin.ext h)
      simpa using by omega
    subst hone
    have hup : (fun s : ℝ => barrierWitnessPotential (Function.update x 1 s))
        = fun _ : ℝ => barrierWitnessPotential x := by
      funext s
      simp [barrierWitnessPotential]
    rw [hup]
    have hdr : -(barrierWitnessDrift x 1) = 0 := by
      simp [barrierWitnessDrift]
    rw [hdr]
    exact hasDerivAt_const _ _

/-- The witness potential restricted to the bias-channel line is the scalar potential
`L̃(w) = w² / 2`, so the barrier `α` appearing in the conclusion below is the barrier of the
multi-dimensional potential along the line and not a separately chosen scalar. -/
theorem barrierWitnessPotential_alongLine (q : ℝ → ℝ) (t : ℝ) :
    barrierWitnessPotential (alongLine 0 barrierWitnessDirection q t) = q t ^ 2 / 2 := by
  simp [barrierWitnessPotential, alongLine, barrierWitnessDirection]

/-- The witness drift is tangent to the bias-channel line along the witness path, with scalar
component `-L̃'(q(t)) = -q(t)`. -/
theorem barrierWitnessDrift_alongLine (q : ℝ → ℝ) (t : ℝ) :
    barrierWitnessDrift (alongLine 0 barrierWitnessDirection q t)
      = (-(q t)) • barrierWitnessDirection := by
  ext i
  fin_cases i <;> simp [barrierWitnessDrift, alongLine, barrierWitnessDirection]

/-- **(FW-4) is not vacuous.** Every hypothesis of `fwActionMulti_alongLine_barrier` is satisfied
by the following instance, and the action it computes is `(e^{2T} - 1) / c`, which is strictly
positive for `T > 0` and `c > 0`.

The instance is the two-dimensional gradient system `dφ = -∇Φ(φ) dt + √η Σ^{1/2} dB` with
`Φ(x) = x₀²/2` and the constant diagonal `Σ = !![c, 0; 0, 1]`; the bias-channel direction is
`e = (1, 0)`, which is an eigendirection of `Σ` with eigenvalue `c`; and the path is the
line-confined reversed relaxation `q(t) = e^t`, which solves `q̇ = L̃'(q) = q`. Nothing is
degenerate: the drift is the honest gradient drift of `Φ`
(`hasDerivAt_barrierWitnessPotential_update`), the potential restricted to the line is the scalar
potential the barrier is measured with (`barrierWitnessPotential_alongLine`), the barrier
`α = L̃(q(T)) - L̃(q(0)) = (e^{2T} - 1)/2` is strictly positive for `T > 0`, and on the physical
range `c > 0` the diffusion is positive definite and hence uniformly elliptic, by
`barrierWitnessDiffusion_posDef`. Only `c ≠ 0` is needed for the identity itself, which is why
that is what is assumed here; `fwActionMulti_barrierWitness_pos` restricts to `c > 0`.

This is what rules out the reading under which the assembled theorem is satisfied only by
constant paths across a zero barrier, where it would degenerate to `0 = 0`. -/
theorem fwActionMulti_barrierWitness (hc : c ≠ 0) (T : ℝ) :
    fwActionMulti (barrierWitnessDiffusion c) barrierWitnessDrift T
        (alongLine 0 barrierWitnessDirection Real.exp)
      = (Real.exp T ^ 2 - 1) / c := by
  have hkey := fwActionMulti_alongLine_barrier (D := barrierWitnessDiffusion c)
    (mu := barrierWitnessDrift) (gamma0 := 0) (e := barrierWitnessDirection)
    (Ltilde := fun w : ℝ => w ^ 2 / 2) (dLtilde := fun w : ℝ => w) (q := Real.exp) (c := c) (T := T)
    barrierWitnessDirection_dotProduct_self hc
    (fun t _ => Real.hasDerivAt_exp t)
    (fun t _ => hasDerivAt_barrierWitnessScalarPotential (Real.exp t))
    ((Real.continuous_exp.pow 2).intervalIntegrable 0 T)
    (fun t _ => isUnit_det_barrierWitnessDiffusion hc _)
    (fun t _ => actsAsScalarOn_barrierWitnessDiffusion _)
    (fun t _ => diffusionAlong_barrierWitnessDiffusion _)
    (fun t _ => barrierWitnessDrift_alongLine Real.exp t)
  rw [hkey, Real.exp_zero]
  ring

/-- The witness action is strictly positive, so the barrier it crosses is a real one. -/
theorem fwActionMulti_barrierWitness_pos (hc : 0 < c) (hT : 0 < T) :
    0 < fwActionMulti (barrierWitnessDiffusion c) barrierWitnessDrift T
      (alongLine 0 barrierWitnessDirection Real.exp) := by
  rw [fwActionMulti_barrierWitness hc.ne' T]
  have h1 : (1 : ℝ) < Real.exp T := by
    rw [← Real.exp_zero]
    exact Real.exp_lt_exp.mpr hT
  have h2 : 1 < Real.exp T ^ 2 := by nlinarith
  exact div_pos (by linarith) hc

end BarrierWitness

end IcnnLift
