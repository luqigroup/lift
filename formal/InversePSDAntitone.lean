import Mathlib

/-!
# Inverse anti-monotonicity in the positive-semidefinite order (the FW-2 auxiliary lemma)

The corrected mechanism of the paper introduces, in `docs/design/lemma1_rederivation.md` §8
between FW-1 and FW-2, an auxiliary lemma about the Loewner order: *if `A ≥ B > 0` then
`B⁻¹ ≥ A⁻¹`*. It is the step that turns FW-1's conclusion — the lift's diffusion matrix
dominates the direct method's — into the pointwise integrand comparison on which FW-2 rests: the
Freidlin–Wentzell action density `¼ uᵀ Σ(γ)⁻¹ u` involves the *inverse* diffusion, so the larger
diffusion carries the smaller density. It is the matrix version of the scalar step
`σ₁² ≤ σ₂² ⟹ 1/σ₂² ≤ 1/σ₁²` on which the whole one-dimensional argument rests, and both versions
are proved here, the scalar one twice: once directly and once *as a consequence of the matrix
theorem*, so that the claim that the two are one fact is a theorem rather than a remark.

**Status in mathlib.** The operator-algebra statement *does* exist:
`CStarAlgebra.inv_le_inv` in `Mathlib/Analysis/CStarAlgebra/ContinuousFunctionalCalculus/Order.lean`
proves `0 ≤ a → a ≤ b → b⁻¹ ≤ a⁻¹` for units `a b` of a C⋆-algebra. It does **not** specialize to
the matrices this development works with. `Mathlib.CStarAlgebra` is by definition a *complex*
C⋆-algebra (it extends `NormedAlgebra ℂ A`), and this mathlib registers no such instance on
matrices at all: `Matrix n n ℝ` is not a complex algebra, and even for `Matrix n n ℂ` the C⋆-norm
lives in the scoped instances of `Matrix.Norms.L2Operator` and no `CStarAlgebra` instance is
assembled from them. Both instance searches were run against this mathlib and both fail. So the
matrix statement is not a citation; it is proved here from the structure matrices do carry — the
scoped Loewner order of `Mathlib/Analysis/Matrix/Order.lean` (`A ≤ B` is by definition
`(B - A).PosSemidef`), the star-ordered-ring structure, and the continuous-functional-calculus
square root `CFC.sqrt`.

**On the design note's own proofs.** Two steps of the note are asserted rather than derived, and
both are supplied here. First, §8 argues `A ≥ B` gives `B^{-1/2} A B^{-1/2} ≥ I`, "so its inverse
satisfies `B^{1/2} A⁻¹ B^{1/2} ≤ I`"; the quoted step is the base case `I ≤ M ⟹ M⁻¹ ≤ I` of the
very statement being proved. It is discharged here without spectral theory, by a congruence of
its own: with `S = CFC.sqrt M` one has `1 - M⁻¹ = S⁻¹ (M - 1) S⁻¹`, positive semidefinite because
congruence preserves the cone. Second, §11.3 (FW-2') derives the constant
`δ = η / (Λ (Λ + η))` in FW-2's strictness clause through an *eigenvalue* argument — "a matrix
with eigenvalues `β/(β+η)` over the eigenvalues `β` of `B`; that map is increasing in `β`". That
argument is replaced here (`posDef_inv_gap`) by a congruence and a resolvent identity, with no
appeal to the spectrum, and the constant is confirmed to be exactly the note's. It is also shown
to be attained (`inv_gap_isotropic_eq`), with no larger constant possible
(`inv_gap_constant_sharp`), so nothing is lost relative to the note's derivation.

**On what "strict" means.** Two inequations must not be confused, and this file keeps them
apart. `posDef_inv_strictAnti` gives strictness in the Loewner *order* (`B⁻¹ < A⁻¹`, that is,
`A⁻¹ - B⁻¹` is positive semidefinite and nonzero). FW-2's strictness clause needs something
strictly stronger, a *uniform quadratic gap* `uᵀ (Σ_B⁻¹ - Σ_A⁻¹) u ≥ δ ‖u‖²`, which Loewner
strictness does not supply. The gap is proved separately, from the note's §11.3 hypotheses, by
`posDef_inv_gap` and `quadForm_inv_gap`.

Everything in this file is unconditional: there are **no** analytic hypotheses, no remainder
terms, and no "to leading order" clauses, because the statement is a finite-dimensional algebraic
identity chain. The hypotheses that appear (`PosDef`, Loewner inequalities, positivity of a
scalar) are the hypotheses of the lemma itself, and the closing section exhibits explicit
witnesses showing that they are satisfiable, that the Loewner order is a genuine (non-total,
non-trivial) order, and that the positivity hypothesis cannot be dropped.

## Results

### The scalar case
* `inv_antitone_of_pos`, `inv_strictAnti_of_pos` — `0 < x ≤ y ⟹ y⁻¹ ≤ x⁻¹`, with the strict
  version.
* `one_div_antitone_of_pos`, `one_div_strictAnti_of_pos` — the same in the `1/σ²` form the
  one-dimensional modules read off.

### Structural facts about the Loewner order
* `posDef_of_le` — a matrix Loewner-above a positive-definite matrix is positive definite.
* `posDef_one` — the identity matrix is positive definite.
* `conj_le_conj` — congruence by an arbitrary matrix preserves the Loewner order.
* `smul_le_smul_of_nonneg` — scaling by a nonnegative real preserves the Loewner order.
* `quadForm_le_of_le` — the Loewner order is the pointwise order of quadratic forms.
* `quadForm_smul_one`, `quadForm_sub` — the two computations that turn a Loewner statement about
  `c • 1` and about a difference into a scalar inequality between quadratic forms.
* `posDef_sqrt` — the continuous-functional-calculus square root of a positive-definite matrix is
  positive definite.
* `posDef_mul_self` — the square of a positive-definite matrix is positive definite.

### The auxiliary lemma itself
* `posDef_inv_le_one_of_one_le` — the base case the note leaves implicit: `1 ≤ M` gives
  `M⁻¹ ≤ 1`. Positive definiteness of `M` is *derived*, not assumed.
* `posDef_inv_antitone` — **the auxiliary lemma**: `A` positive definite and `A ≤ B` give
  `B⁻¹ ≤ A⁻¹`.
* `inv_sub_inv_posSemidef` — the same statement phrased without the scoped Loewner order.
* `posDef_inv_strictAnti` — strictness in the Loewner order only: `A < B` gives `B⁻¹ < A⁻¹`.
* `quadForm_inv_antitone` — the form FW-2 uses: `uᵀ B⁻¹ u ≤ uᵀ A⁻¹ u` for every direction `u`.

### The isotropic dictionary
* `posDef_smul_one`, `smul_one_le_smul_one`, `smul_one_le_smul_one_iff`, `inv_smul_one` — on
  multiples of the identity the Loewner order *is* the order of the scalars, and inversion is
  inversion of the scalar.
* `isotropic_inv_antitone` — the matrix lemma restricted to isotropic diffusions `σ² · I`.
* `inv_antitone_of_pos_via_matrix` — the scalar lemma *deduced from* `posDef_inv_antitone`
  through that dictionary. This is what makes "the one-dimensional step and FW-2's matrix step
  are one fact" a proved statement rather than an assertion.

### FW-2's pointwise integrand comparison
* `uniformly_elliptic_quadForm_inv_antitone` — the comparison at a *single* point, in the note's
  hypothesis shape `Σ_A ≥ Σ_B ≥ λ I > 0`.
* `actionDensity_antitone_on` — the same quantified as the note quantifies it: over matrix
  *fields* `Σ_A, Σ_B : X → Matrix n n ℝ`, at every point of a domain `D` and in every direction,
  for the action density `¼ uᵀ Σ⁻¹ u` itself.
* `actionDensity_antitone_along_path` — the corollary the note's proof of FW-2 actually invokes:
  along every path that remains in `D` on its own time set, at every time of that set, and with
  every velocity field.

### FW-2's strictness clause (the note's §11.3 constant)
* `mul_self_le_of_le_smul_one` — `0 < A ≤ Λ I` gives `A² ≤ Λ² I`, proved by congruence rather
  than through the spectrum.
* `inv_sub_inv_shift` — the resolvent identity `A⁻¹ - (A + η I)⁻¹ = η ((A + η I) A)⁻¹`.
* `posDef_inv_gap` — **the note's FW-2' bound**: `0 < A ≤ Λ I` and `A + η I ≤ B` give
  `A⁻¹ - B⁻¹ ≥ (η / (Λ (Λ + η))) I`.
* `quadForm_inv_gap` — the same in the shape the strictness clause is stated in,
  `uᵀ (A⁻¹ - B⁻¹) u ≥ δ ‖u‖²` with `δ = η / (Λ (Λ + η))`.
* `inv_gap_isotropic_eq`, `inv_gap_constant_sharp` — the bound is attained on isotropic
  matrices, and no larger constant is admissible, so the note's `δ` is sharp.

### Witnesses
* `witness_posDef`, `witness_le`, `witness_excess_singular`, `witness_not_ge` — an explicit
  anisotropic pair in `Matrix (Fin 2) (Fin 2) ℝ` satisfying the hypotheses of
  `posDef_inv_antitone` and of `posDef_inv_strictAnti`, with a singular excess, together with a
  proof that the reverse Loewner inequality fails.
* `witness_incomparable` — the Loewner order is not total, so `A ≤ B` is a genuine restriction.
* `posDef_hypothesis_is_necessary` — the positive-definiteness hypothesis of
  `posDef_inv_antitone` cannot be dropped: `0 ≤ 1` holds while `1⁻¹ ≤ 0⁻¹` fails.
* `gap_hypotheses_satisfiable` — the hypotheses of `posDef_inv_gap` are satisfiable.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. This file carries **no** explicit analytic hypothesis: every
result is proved outright from mathlib's matrix and order API.

What it deliberately does *not* do is prove FW-2 itself. FW-2 quantifies over absolutely
continuous paths in a bounded domain, integrates the action density along each, and takes an
infimum over paths. This file supplies the integrand comparison at every point of the domain, in
every direction, along every path and at every time (`actionDensity_antitone_along_path`), and the
uniform quadratic gap that FW-2's strictness clause requires (`quadForm_inv_gap`); what remains
is the integration and the passage to the infimum, together with the geometric input of FW-3a
(that near-minimizing paths spend time `τ` on the band), and those belong to
`QuasipotentialMonotone.lean` and `BandOccupancy.lean`. Nothing about Freidlin–Wentzell theory,
exit times, or stochastic differential equations is asserted in this file, and none of it is
available in this mathlib.

Two further scope limits, stated for accuracy. First, everything is proved for real matrices
`Matrix n n ℝ` over a `Fintype`/`DecidableEq` index type, which is what this development uses;
no complex-matrix or general-operator version is claimed. Second, `posDef_inv_gap` derives the
note's constant `δ = η / (Λ (Λ + η))` from the note's own §11.3 hypotheses (`A + η I ≤ B` and
`A ≤ Λ I`); it does not establish those hypotheses, which is the content of FW-1 together with
assumption (A6) and belongs to `DiffusionOrdering.lean` and `CouplingSign.lean`.
-/

open Matrix
open scoped MatrixOrder

namespace IcnnLift

/-! ## The scalar case

This is the step the one-dimensional modules of the development actually use, in the form
`σ₁² ≤ σ₂² ⟹ 1/σ₂² ≤ 1/σ₁²`. It is named here so that those modules cite a statement rather
than re-deriving it, and so that it is visible which of the two cases — scalar or matrix — a
given downstream argument depends on. -/

/-- Anti-monotonicity of inversion on the positive reals: if `0 < x` and `x ≤ y` then
`y⁻¹ ≤ x⁻¹`. This is the scalar shadow of `posDef_inv_antitone`; that it is genuinely a shadow,
and not an independent fact, is proved below as `inv_antitone_of_pos_via_matrix`. -/
theorem inv_antitone_of_pos {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) : y⁻¹ ≤ x⁻¹ :=
  inv_anti₀ hx hxy

/-- Strict anti-monotonicity of inversion on the positive reals. -/
theorem inv_strictAnti_of_pos {x y : ℝ} (hx : 0 < x) (hxy : x < y) : y⁻¹ < x⁻¹ := by
  have h := one_div_lt_one_div_of_lt hx hxy
  simpa [one_div] using h

/-- The scalar step in the `1/σ²` form in which the one-dimensional diffusion comparison reads
it off: a larger diffusion coefficient gives a smaller action density. -/
theorem one_div_antitone_of_pos {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) : 1 / y ≤ 1 / x :=
  one_div_le_one_div_of_le hx hxy

/-- The strict form of `one_div_antitone_of_pos`. -/
theorem one_div_strictAnti_of_pos {x y : ℝ} (hx : 0 < x) (hxy : x < y) : 1 / y < 1 / x :=
  one_div_lt_one_div_of_lt hx hxy

/-! ## Structural facts about the Loewner order

In the scoped order of `Mathlib/Analysis/Matrix/Order.lean`, `A ≤ B` unfolds by definition to
`(B - A).PosSemidef`. The five facts collected here are the entire geometric input to the
auxiliary lemma. -/

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- A matrix Loewner-above a positive-definite matrix is itself positive definite. In FW-2 this
is what promotes the uniform-ellipticity bound `Σ ≥ λ I > 0` to invertibility of `Σ`. -/
theorem posDef_of_le {A B : Matrix n n ℝ} (hA : A.PosDef) (hAB : A ≤ B) : B.PosDef := by
  have h : (B - A).PosSemidef := Matrix.le_iff.mp hAB
  have hAdd := hA.add_posSemidef h
  rwa [add_sub_cancel] at hAdd

omit [Fintype n] in
/-- The identity matrix is positive definite. Stated separately because it is what makes the
positive-definiteness hypothesis of `posDef_inv_le_one_of_one_le` redundant, and therefore
removable. -/
theorem posDef_one : (1 : Matrix n n ℝ).PosDef := by
  rw [← Matrix.diagonal_one]
  exact Matrix.posDef_diagonal_iff.mpr fun _ => one_pos

omit [DecidableEq n] in
/-- Congruence preserves the Loewner order: if `A ≤ B` then `Cᴴ A C ≤ Cᴴ B C` for every `C`.
This is the only geometric input the auxiliary lemma needs, and it is nothing more than the fact
that the positive-semidefinite cone is stable under congruence. -/
theorem conj_le_conj {A B : Matrix n n ℝ} (h : A ≤ B) (C : Matrix n n ℝ) :
    Cᴴ * A * C ≤ Cᴴ * B * C := by
  rw [Matrix.le_iff] at h ⊢
  have hc := h.conjTranspose_mul_mul_same C
  have hEq : Cᴴ * (B - A) * C = Cᴴ * B * C - Cᴴ * A * C := by
    rw [Matrix.mul_sub, Matrix.sub_mul]
  rwa [hEq] at hc

omit [Fintype n] [DecidableEq n] in
/-- Scaling by a nonnegative real preserves the Loewner order. This is the second half of the
scalar dictionary: `conj_le_conj` handles congruences, this handles dilations. -/
theorem smul_le_smul_of_nonneg {A B : Matrix n n ℝ} (h : A ≤ B) {c : ℝ} (hc : 0 ≤ c) :
    c • A ≤ c • B := by
  rw [Matrix.le_iff] at h ⊢
  rw [← smul_sub]
  exact h.smul hc

omit [DecidableEq n] in
/-- The Loewner order is the pointwise order of quadratic forms: `M ≤ N` gives
`uᵀ M u ≤ uᵀ N u` in every direction `u`. This is how the matrix statement is consumed by an
action integrand. -/
theorem quadForm_le_of_le {M N : Matrix n n ℝ} (h : M ≤ N) (u : n → ℝ) :
    u ⬝ᵥ (M *ᵥ u) ≤ u ⬝ᵥ (N *ᵥ u) := by
  have hpsd : (N - M).PosSemidef := Matrix.le_iff.mp h
  have h0 : 0 ≤ star u ⬝ᵥ ((N - M) *ᵥ u) :=
    (Matrix.posSemidef_iff_dotProduct_mulVec.mp hpsd).2 u
  rw [Matrix.sub_mulVec, dotProduct_sub] at h0
  simp only [star_trivial] at h0
  linarith

/-- The quadratic form of a multiple of the identity is that multiple of `‖u‖²`, where `u ⬝ᵥ u`
is the sum of squares of the coordinates. This is what turns a Loewner lower bound by `δ • 1`
into the `δ ‖u‖²` of FW-2's strictness clause. -/
theorem quadForm_smul_one (c : ℝ) (u : n → ℝ) :
    u ⬝ᵥ ((c • (1 : Matrix n n ℝ)) *ᵥ u) = c * (u ⬝ᵥ u) := by
  simp [Matrix.smul_mulVec, dotProduct_smul]

omit [DecidableEq n] in
/-- The quadratic form of a difference is the difference of the quadratic forms. -/
theorem quadForm_sub (M N : Matrix n n ℝ) (u : n → ℝ) :
    u ⬝ᵥ ((M - N) *ᵥ u) = u ⬝ᵥ (M *ᵥ u) - u ⬝ᵥ (N *ᵥ u) := by
  rw [Matrix.sub_mulVec, dotProduct_sub]

/-- The continuous-functional-calculus square root of a positive-definite matrix is positive
definite. Positive semidefiniteness is `CFC.sqrt_nonneg`; definiteness follows because
`det A = (det (CFC.sqrt A))²` is a unit, hence so is `det (CFC.sqrt A)`. -/
theorem posDef_sqrt {A : Matrix n n ℝ} (hA : A.PosDef) : (CFC.sqrt A).PosDef := by
  have hSpsd : (CFC.sqrt A).PosSemidef := Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg A)
  have hSS : CFC.sqrt A * CFC.sqrt A = A := CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg
  have hdetA : IsUnit A.det := (Matrix.isUnit_iff_isUnit_det A).mp hA.isUnit
  have hdet : IsUnit (CFC.sqrt A).det := by
    apply isUnit_of_mul_isUnit_left (y := (CFC.sqrt A).det)
    rw [← Matrix.det_mul, hSS]
    exact hdetA
  exact hSpsd.posDef_iff_isUnit.mpr ((Matrix.isUnit_iff_isUnit_det _).mpr hdet)

/-- The square of a positive-definite matrix is positive definite, because `A A = Aᴴ 1 A` and
congruence by an invertible matrix preserves definiteness. -/
theorem posDef_mul_self {A : Matrix n n ℝ} (hA : A.PosDef) : (A * A).PosDef := by
  have hinj : Function.Injective A.mulVec := Matrix.mulVec_injective_iff_isUnit.mpr hA.isUnit
  have h := (posDef_one (n := n)).conjTranspose_mul_mul_same hinj
  rwa [hA.isHermitian.eq, Matrix.mul_one] at h

/-! ## The auxiliary lemma -/

/-- The base case of the auxiliary lemma, which the design note's proof leaves implicit: if
`1 ≤ M` in the Loewner order then `M⁻¹ ≤ 1`.

Positive definiteness of `M` is **not** a hypothesis. It follows from `1 ≤ M` by `posDef_of_le`
applied to `posDef_one`, and stating it as a hypothesis would make the base case look as though
it needed an input it does not.

The proof is a congruence, not a spectral argument. With `S = CFC.sqrt M` one has `S S = M`,
`S⁻¹ M S⁻¹ = 1` and `S⁻¹ S⁻¹ = M⁻¹`, so `1 - M⁻¹ = S⁻¹ (M - 1) S⁻¹`, and the right-hand side is
positive semidefinite because `M - 1` is and `S⁻¹` is Hermitian. -/
theorem posDef_inv_le_one_of_one_le {M : Matrix n n ℝ} (h1 : (1 : Matrix n n ℝ) ≤ M) : M⁻¹ ≤ 1 := by
  have hM : M.PosDef := posDef_of_le posDef_one h1
  set S : Matrix n n ℝ := CFC.sqrt M with hSdef
  have hSpd : S.PosDef := posDef_sqrt hM
  have hSS : S * S = M := CFC.sqrt_mul_sqrt_self M hM.posSemidef.nonneg
  have hdetS : IsUnit S.det := (Matrix.isUnit_iff_isUnit_det S).mp hSpd.isUnit
  have hTS : S⁻¹ * S = 1 := Matrix.nonsing_inv_mul S hdetS
  have hST : S * S⁻¹ = 1 := Matrix.mul_nonsing_inv S hdetS
  have hTherm : (S⁻¹)ᴴ = S⁻¹ := hSpd.isHermitian.inv
  have hTT : S⁻¹ * S⁻¹ = M⁻¹ := by rw [← Matrix.mul_inv_rev, hSS]
  have hTMT : S⁻¹ * M * S⁻¹ = 1 := by
    have e : S⁻¹ * M * S⁻¹ = (S⁻¹ * S) * (S * S⁻¹) := by rw [← hSS]; simp [mul_assoc]
    rw [e, hTS, hST, one_mul]
  have hkey : S⁻¹ * (1 : Matrix n n ℝ) * S⁻¹ ≤ S⁻¹ * M * S⁻¹ := by
    have h := conj_le_conj h1 (S⁻¹)
    rwa [hTherm] at h
  rw [hTMT, Matrix.mul_one, hTT] at hkey
  exact hkey

/-- **The FW-2 auxiliary lemma** (`docs/design/lemma1_rederivation.md` §8): inversion is
anti-monotone on positive-definite matrices. If `A` is positive definite and `A ≤ B` in the
Loewner order, then `B⁻¹ ≤ A⁻¹`. This is the note's `A ≥ B > 0 ⟹ B⁻¹ ≥ A⁻¹` with the two
matrix names interchanged; the positivity hypothesis is placed on the smaller matrix, which is
where it is needed and where the note places it. Positive definiteness of the larger matrix is
derived, not assumed.

That the hypothesis on the smaller matrix cannot be dropped is proved below as
`posDef_hypothesis_is_necessary`.

Proof, following the note but with its implicit inversion step supplied by
`posDef_inv_le_one_of_one_le`: put `R = CFC.sqrt A`, so `R⁻¹ A R⁻¹ = 1`; congruence by `R⁻¹`
turns `A ≤ B` into `1 ≤ R⁻¹ B R⁻¹ =: M`; the base case gives `M⁻¹ ≤ 1`; and `M⁻¹ = R B⁻¹ R`, so
congruence by `R⁻¹` a second time gives `B⁻¹ ≤ R⁻¹ R⁻¹ = A⁻¹`. -/
theorem posDef_inv_antitone {A B : Matrix n n ℝ} (hA : A.PosDef) (hAB : A ≤ B) : B⁻¹ ≤ A⁻¹ := by
  set R : Matrix n n ℝ := CFC.sqrt A with hRdef
  have hRpd : R.PosDef := posDef_sqrt hA
  have hRR : R * R = A := CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg
  have hdetR : IsUnit R.det := (Matrix.isUnit_iff_isUnit_det R).mp hRpd.isUnit
  have hPR : R⁻¹ * R = 1 := Matrix.nonsing_inv_mul R hdetR
  have hRP : R * R⁻¹ = 1 := Matrix.mul_nonsing_inv R hdetR
  have hPherm : (R⁻¹)ᴴ = R⁻¹ := hRpd.isHermitian.inv
  have hPP : R⁻¹ * R⁻¹ = A⁻¹ := by rw [← Matrix.mul_inv_rev, hRR]
  have hPAP : R⁻¹ * A * R⁻¹ = 1 := by
    have e : R⁻¹ * A * R⁻¹ = (R⁻¹ * R) * (R * R⁻¹) := by rw [← hRR]; simp [mul_assoc]
    rw [e, hPR, hRP, one_mul]
  have h1M : (1 : Matrix n n ℝ) ≤ R⁻¹ * B * R⁻¹ := by
    have h := conj_le_conj hAB (R⁻¹)
    rwa [hPherm, hPAP] at h
  have hMinv : (R⁻¹ * B * R⁻¹)⁻¹ = R * B⁻¹ * R := by
    rw [Matrix.mul_inv_rev, Matrix.mul_inv_rev, Matrix.nonsing_inv_nonsing_inv R hdetR, mul_assoc]
  have hstep := posDef_inv_le_one_of_one_le h1M
  rw [hMinv] at hstep
  have h2 := conj_le_conj hstep (R⁻¹)
  rw [hPherm] at h2
  have eL : R⁻¹ * (R * B⁻¹ * R) * R⁻¹ = B⁻¹ := by
    have e : R⁻¹ * (R * B⁻¹ * R) * R⁻¹ = (R⁻¹ * R) * B⁻¹ * (R * R⁻¹) := by simp [mul_assoc]
    rw [e, hPR, hRP, one_mul, mul_one]
  have eR : R⁻¹ * (1 : Matrix n n ℝ) * R⁻¹ = A⁻¹ := by rw [Matrix.mul_one, hPP]
  rwa [eL, eR] at h2

/-- The auxiliary lemma stated without the scoped Loewner order, for callers that prefer to work
with `Matrix.PosSemidef` directly. -/
theorem inv_sub_inv_posSemidef {A B : Matrix n n ℝ} (hA : A.PosDef)
    (hAB : (B - A).PosSemidef) : (A⁻¹ - B⁻¹).PosSemidef :=
  Matrix.le_iff.mp (posDef_inv_antitone hA (Matrix.le_iff.mpr hAB))

/-- Strictness **in the Loewner order**: if `A` is positive definite and `A < B`, then
`B⁻¹ < A⁻¹`. Strictness transfers because inversion is injective on invertible matrices.

This is weaker than the strictness FW-2 needs, and the two should not be confused. `A < B` in a
partial order means `A ≤ B` together with `A ≠ B`, so the conclusion says only that `A⁻¹ - B⁻¹`
is positive semidefinite and nonzero: the gap may be zero in most directions, and may be
arbitrarily small in the rest. FW-2's strictness clause needs the uniform bound
`uᵀ (A⁻¹ - B⁻¹) u ≥ δ ‖u‖²`, which is `posDef_inv_gap` and `quadForm_inv_gap` below, and which
requires genuinely more hypotheses. -/
theorem posDef_inv_strictAnti {A B : Matrix n n ℝ} (hA : A.PosDef) (hAB : A < B) : B⁻¹ < A⁻¹ := by
  have hB : B.PosDef := posDef_of_le hA hAB.le
  have hne : B⁻¹ ≠ A⁻¹ := by
    intro hEq
    have hdA : IsUnit A.det := (Matrix.isUnit_iff_isUnit_det A).mp hA.isUnit
    have hdB : IsUnit B.det := (Matrix.isUnit_iff_isUnit_det B).mp hB.isUnit
    have : B = A := by
      rw [← Matrix.nonsing_inv_nonsing_inv B hdB, hEq, Matrix.nonsing_inv_nonsing_inv A hdA]
    exact hAB.ne this.symm
  exact lt_of_le_of_ne (posDef_inv_antitone hA hAB.le) hne

/-- The form in which FW-2 consumes the auxiliary lemma: the quadratic form of the inverse is
anti-monotone, so the larger diffusion carries the smaller Freidlin–Wentzell action density
`¼ uᵀ Σ⁻¹ u`, in every direction `u`. The statement is about a single pair of matrices; the
version quantified over the points of a domain, as FW-2 requires, is
`actionDensity_antitone_on`. -/
theorem quadForm_inv_antitone {A B : Matrix n n ℝ} (hA : A.PosDef) (hAB : A ≤ B) (u : n → ℝ) :
    u ⬝ᵥ (B⁻¹ *ᵥ u) ≤ u ⬝ᵥ (A⁻¹ *ᵥ u) :=
  quadForm_le_of_le (posDef_inv_antitone hA hAB) u

/-! ## The isotropic dictionary

FW-2 is a statement about matrix-valued diffusion coefficients, while the paper's
one-dimensional argument is a statement about scalars. The lemmas below show that the two are
the same statement on multiples of the identity: the Loewner order restricted to `c · I` is the
order of `c`, inversion is inversion of `c`, and `inv_antitone_of_pos_via_matrix` closes the
loop by deriving the scalar lemma from the matrix theorem. -/

omit [Fintype n] in
/-- A positive multiple of the identity is positive definite. -/
theorem posDef_smul_one {c : ℝ} (hc : 0 < c) : ((c • (1 : Matrix n n ℝ))).PosDef := by
  rw [Matrix.smul_one_eq_diagonal]
  exact Matrix.posDef_diagonal_iff.mpr fun _ => hc

omit [Fintype n] in
/-- Multiples of the identity are Loewner-ordered by their scalars. -/
theorem smul_one_le_smul_one {c d : ℝ} (h : c ≤ d) :
    (c • (1 : Matrix n n ℝ)) ≤ d • 1 := by
  rw [Matrix.le_iff, ← sub_smul, Matrix.smul_one_eq_diagonal]
  exact Matrix.posSemidef_diagonal_iff.mpr fun _ => by linarith

omit [Fintype n] in
/-- On multiples of the identity the Loewner order **is** the order of the scalars. This is the
precise sense in which the matrix statement of the auxiliary lemma contains the scalar one. The
forward direction is false for an empty index type, where every matrix is `0`, hence the
`Nonempty` hypothesis; the direction used downstream, `smul_one_le_smul_one`, does not need it. -/
theorem smul_one_le_smul_one_iff [Nonempty n] {c d : ℝ} :
    (c • (1 : Matrix n n ℝ)) ≤ d • 1 ↔ c ≤ d := by
  constructor
  · intro h
    rw [Matrix.le_iff, ← sub_smul, Matrix.smul_one_eq_diagonal] at h
    have := Matrix.posSemidef_diagonal_iff.mp h (Classical.arbitrary n)
    linarith
  · exact smul_one_le_smul_one

/-- Inversion of an isotropic matrix is inversion of its scalar. -/
theorem inv_smul_one {c : ℝ} (hc : c ≠ 0) :
    ((c • (1 : Matrix n n ℝ)))⁻¹ = c⁻¹ • 1 := by
  refine Matrix.inv_eq_right_inv ?_
  rw [Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul, smul_smul, mul_inv_cancel₀ hc, one_smul]

/-- The matrix auxiliary lemma, restricted to isotropic diffusions `σ² · I`. The proof goes
through `posDef_inv_antitone`, so this is a genuine specialization and not a second, independent
argument. -/
theorem isotropic_inv_antitone {c d : ℝ} (hc : 0 < c) (hcd : c ≤ d) :
    (d • (1 : Matrix n n ℝ))⁻¹ ≤ (c • (1 : Matrix n n ℝ))⁻¹ :=
  posDef_inv_antitone (posDef_smul_one hc) (smul_one_le_smul_one hcd)

/-- The scalar step `0 < x ≤ y ⟹ y⁻¹ ≤ x⁻¹`, **deduced from the matrix theorem**: apply
`posDef_inv_antitone` to `x · I` and `y · I` on a one-point index type, then read the conclusion
back through `inv_smul_one` and `smul_one_le_smul_one_iff`.

The statement is identical to `inv_antitone_of_pos`; only the proof differs. The point of
proving it twice is that the assertion "the one-dimensional argument of the paper and FW-2's
matrix argument are one and the same fact" is thereby a theorem of this file rather than a
remark in its docstring. -/
theorem inv_antitone_of_pos_via_matrix {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) : y⁻¹ ≤ x⁻¹ := by
  have hy : 0 < y := lt_of_lt_of_le hx hxy
  have h := isotropic_inv_antitone (n := Fin 1) hx hxy
  rw [inv_smul_one hy.ne', inv_smul_one hx.ne'] at h
  exact smul_one_le_smul_one_iff.mp h

/-! ## FW-2's pointwise integrand comparison

The design note states FW-2 for coefficient *fields* on a bounded domain: `Σ_A(x) ≥ Σ_B(x) ≥
λ I > 0` for every `x ∈ D`, and its proof begins "for any absolutely continuous path `γ` in `D`
and a.e. `t`, the integrand comparison holds pointwise". The three results below record that
comparison at exactly those three levels of quantification — one point, every point of `D`, and
every time along every path in `D` — so that what remains to FW-2 is the integration and the
infimum, and nothing pointwise. -/

/-- FW-2's integrand comparison at a single point, in the hypothesis shape the design note
states it in: under uniform ellipticity `Σ_B ≥ λ I > 0` and the FW-1 ordering `Σ_B ≤ Σ_A`, the
action density of `Σ_A` is dominated by that of `Σ_B` in every direction `u`.

This is a statement about one pair of matrices. The ellipticity hypothesis is not used for
uniformity here — there is no domain to be uniform over — but only to supply positive
definiteness of `Σ_B`; the genuinely domain-quantified statement is `actionDensity_antitone_on`. -/
theorem uniformly_elliptic_quadForm_inv_antitone {lam : ℝ} (hlam : 0 < lam)
    {SB SA : Matrix n n ℝ} (hell : (lam • (1 : Matrix n n ℝ)) ≤ SB) (hord : SB ≤ SA)
    (u : n → ℝ) : u ⬝ᵥ (SA⁻¹ *ᵥ u) ≤ u ⬝ᵥ (SB⁻¹ *ᵥ u) :=
  quadForm_inv_antitone (posDef_of_le (posDef_smul_one hlam) hell) hord u

/-- FW-2's integrand comparison quantified as the design note quantifies it: over coefficient
*fields* `Σ_A, Σ_B : X → Matrix n n ℝ` satisfying `Σ_A(x) ≥ Σ_B(x) ≥ λ I > 0` at every point of
a domain `D`, the Freidlin–Wentzell action density `¼ uᵀ Σ(x)⁻¹ u` of `Σ_A` is dominated by that
of `Σ_B` at every point of `D` and in every direction.

The factor `¼` is the note's own normalization of the action integrand; it is carried explicitly
so that the statement is literally the note's integrand comparison and not a rescaled cousin. -/
theorem actionDensity_antitone_on {X : Type*} {D : Set X} {lam : ℝ} (hlam : 0 < lam)
    {SB SA : X → Matrix n n ℝ}
    (hell : ∀ x ∈ D, (lam • (1 : Matrix n n ℝ)) ≤ SB x)
    (hord : ∀ x ∈ D, SB x ≤ SA x) :
    ∀ x ∈ D, ∀ u : n → ℝ,
      (1 / 4 : ℝ) * (u ⬝ᵥ ((SA x)⁻¹ *ᵥ u)) ≤ (1 / 4 : ℝ) * (u ⬝ᵥ ((SB x)⁻¹ *ᵥ u)) := by
  intro x hx u
  have h := uniformly_elliptic_quadForm_inv_antitone hlam (hell x hx) (hord x hx) u
  linarith

/-- The corollary the note's proof of FW-2 actually invokes: along every path `γ` remaining in
`D`, at every time `t`, and for every velocity field `v` — in FW-2, `v t = γ'(t) + ∇Ũ(γ t)` —
the action density of `Σ_A` is dominated by that of `Σ_B`.

The path is required to lie in `D` only on its own time set `I`, which in FW-2 is the interval
`[0, T]`; nothing is assumed about `γ` outside `I`, and no regularity of `γ` or of `v` is
assumed anywhere. The conclusion holds at *every* `t ∈ I`, not merely almost every `t`, because
it is an algebraic identity in the values of `γ` and `v`. Absolute continuity of `γ` and
integrability of the density enter only in the step this file does not take, integration along
the path. -/
theorem actionDensity_antitone_along_path {X : Type*} {D : Set X} {lam : ℝ} (hlam : 0 < lam)
    {SB SA : X → Matrix n n ℝ}
    (hell : ∀ x ∈ D, (lam • (1 : Matrix n n ℝ)) ≤ SB x)
    (hord : ∀ x ∈ D, SB x ≤ SA x)
    (gamma : ℝ → X) {I : Set ℝ} (hgamma : ∀ t ∈ I, gamma t ∈ D) (v : ℝ → n → ℝ)
    {t : ℝ} (ht : t ∈ I) :
    (1 / 4 : ℝ) * (v t ⬝ᵥ ((SA (gamma t))⁻¹ *ᵥ v t))
      ≤ (1 / 4 : ℝ) * (v t ⬝ᵥ ((SB (gamma t))⁻¹ *ᵥ v t)) :=
  actionDensity_antitone_on hlam hell hord _ (hgamma t ht) (v t)

/-! ## FW-2's strictness clause: the quantitative gap of the note's §11.3

FW-2's strictness clause asks for `δ > 0` with `uᵀ (Σ_B⁻¹ - Σ_A⁻¹) u ≥ δ ‖u‖²`, and §11.3
(FW-2') produces `δ = η / (Λ (Λ + η))` from `Σ_A ≥ Σ_B + η I` and `Σ_B ≤ Λ I`. The note's
derivation of that constant passes through the eigenvalues of `B^{1/2} (B + η I)^{-1} B^{1/2}`.
The derivation below reaches the same constant without any reference to the spectrum: it uses
one congruence (to get `A² ≤ Λ² I` from `A ≤ Λ I`), one resolvent identity, and the
auxiliary lemma itself. In the names used here `A` is the note's `Σ_B`, the *smaller* matrix,
which is the one the note bounds above. -/

/-- A congruence proof that squaring is monotone against a multiple of the identity: if `A` is
positive definite and `A ≤ Λ I`, then `A² ≤ Λ² I`.

Conjugating `A ≤ Λ I` by `R = CFC.sqrt A` gives `R A R ≤ Λ R R`, and `R A R = A A`, `R R = A` by
associativity alone, so `A² ≤ Λ A ≤ Λ² I`. No eigenvalue of `A` is mentioned. -/
theorem mul_self_le_of_le_smul_one {A : Matrix n n ℝ} {lam : ℝ} (hA : A.PosDef) (hlam : 0 ≤ lam)
    (h : A ≤ lam • (1 : Matrix n n ℝ)) : A * A ≤ (lam * lam) • (1 : Matrix n n ℝ) := by
  set R : Matrix n n ℝ := CFC.sqrt A with hRdef
  have hRR : R * R = A := CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg
  have hRherm : Rᴴ = R := (posDef_sqrt hA).isHermitian
  have hc := conj_le_conj h R
  rw [hRherm] at hc
  have eL : R * A * R = A * A := by rw [← hRR]; noncomm_ring
  have eR : R * (lam • (1 : Matrix n n ℝ)) * R = lam • A := by
    rw [← hRR, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one]
  rw [eL, eR] at hc
  refine hc.trans ?_
  have h2 := smul_le_smul_of_nonneg h hlam
  rwa [smul_smul] at h2

/-- The resolvent identity `A⁻¹ - (A + η I)⁻¹ = η ((A + η I) A)⁻¹`, for `A` and `A + η I` both
positive definite. It is the algebraic heart of the quantitative gap: it converts a difference of
inverses, which is not directly comparable to anything, into a single inverse, to which the
auxiliary lemma applies. -/
theorem inv_sub_inv_shift {A : Matrix n n ℝ} {eta : ℝ} (hA : A.PosDef)
    (hC : (A + eta • (1 : Matrix n n ℝ)).PosDef) :
    A⁻¹ - (A + eta • (1 : Matrix n n ℝ))⁻¹ = eta • (((A + eta • (1 : Matrix n n ℝ)) * A)⁻¹) := by
  set C : Matrix n n ℝ := A + eta • 1 with hCdef
  have hdA : IsUnit A.det := (Matrix.isUnit_iff_isUnit_det A).mp hA.isUnit
  have hdC : IsUnit C.det := (Matrix.isUnit_iff_isUnit_det C).mp hC.isUnit
  have hAA : A⁻¹ * A = 1 := Matrix.nonsing_inv_mul A hdA
  have hCC : C * C⁻¹ = 1 := Matrix.mul_nonsing_inv C hdC
  have hCA : C - A = eta • (1 : Matrix n n ℝ) := by rw [hCdef]; abel
  have key : A⁻¹ * (C - A) * C⁻¹ = A⁻¹ - C⁻¹ := by
    rw [Matrix.mul_sub, Matrix.sub_mul, mul_assoc A⁻¹ C C⁻¹, hCC, Matrix.mul_one, hAA,
      Matrix.one_mul]
  rw [← key, hCA, Matrix.mul_inv_rev, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one]

/-- **The quantitative gap of `docs/design/lemma1_rederivation.md` §11.3 (FW-2').** If `A` is
positive definite with `A ≤ Λ I`, and `B` exceeds `A` by at least `η I`, then

  `A⁻¹ - B⁻¹ ≥ (η / (Λ (Λ + η))) I`.

In the note's names `A` is `Σ_dir` and `B` is `Σ_lift`, and this is exactly the note's
`δ = η / (Λ (Λ + η))`. As the note observes, only the *smaller* matrix needs an upper bound; no
bound of any kind is imposed on `B`.

The note derives the constant from the eigenvalues of `B^{1/2}(B + η I)^{-1}B^{1/2}`. Here the
route is: `B⁻¹ ≤ (A + η I)⁻¹` by the auxiliary lemma; `A⁻¹ - (A + η I)⁻¹ = η ((A + η I) A)⁻¹` by
`inv_sub_inv_shift`; `(A + η I) A = A² + η A ≤ (Λ² + η Λ) I = Λ (Λ + η) I` by
`mul_self_le_of_le_smul_one`; and the auxiliary lemma once more turns that upper bound into the
lower bound `((A + η I) A)⁻¹ ≥ (Λ (Λ + η))⁻¹ I`. No spectrum is used at any step. -/
theorem posDef_inv_gap {A B : Matrix n n ℝ} {eta lam : ℝ}
    (hA : A.PosDef) (heta : 0 < eta) (hlam : 0 < lam)
    (hupper : A ≤ lam • (1 : Matrix n n ℝ)) (hgap : A + eta • (1 : Matrix n n ℝ) ≤ B) :
    (eta / (lam * (lam + eta))) • (1 : Matrix n n ℝ) ≤ A⁻¹ - B⁻¹ := by
  have hEta : ((eta • (1 : Matrix n n ℝ))).PosDef := posDef_smul_one heta
  have hC : (A + eta • (1 : Matrix n n ℝ)).PosDef := hA.add_posSemidef hEta.posSemidef
  have hBinv : B⁻¹ ≤ (A + eta • (1 : Matrix n n ℝ))⁻¹ := posDef_inv_antitone hC hgap
  have hres := inv_sub_inv_shift hA hC
  have hCA_eq : (A + eta • (1 : Matrix n n ℝ)) * A = A * A + eta • A := by
    rw [Matrix.add_mul, Matrix.smul_mul, Matrix.one_mul]
  have hsq : A * A ≤ (lam * lam) • (1 : Matrix n n ℝ) :=
    mul_self_le_of_le_smul_one hA hlam.le hupper
  have hetaA : eta • A ≤ (eta * lam) • (1 : Matrix n n ℝ) := by
    have h := smul_le_smul_of_nonneg hupper heta.le
    rwa [smul_smul] at h
  have hsum : A * A + eta • A ≤ (lam * (lam + eta)) • (1 : Matrix n n ℝ) := by
    have h := add_le_add hsq hetaA
    rwa [← add_smul, show lam * lam + eta * lam = lam * (lam + eta) by ring] at h
  have hCA_pd : ((A + eta • (1 : Matrix n n ℝ)) * A).PosDef := by
    rw [hCA_eq]
    exact (posDef_mul_self hA).add_posSemidef (hA.posSemidef.smul heta.le)
  have hCA_le : (A + eta • (1 : Matrix n n ℝ)) * A ≤ (lam * (lam + eta)) • (1 : Matrix n n ℝ) := by
    rw [hCA_eq]; exact hsum
  have hne : lam * (lam + eta) ≠ 0 := by positivity
  have hinvle : (lam * (lam + eta))⁻¹ • (1 : Matrix n n ℝ)
      ≤ ((A + eta • (1 : Matrix n n ℝ)) * A)⁻¹ := by
    have h := posDef_inv_antitone hCA_pd hCA_le
    rwa [inv_smul_one hne] at h
  have hscaled := smul_le_smul_of_nonneg hinvle heta.le
  rw [smul_smul, ← div_eq_mul_inv] at hscaled
  calc (eta / (lam * (lam + eta))) • (1 : Matrix n n ℝ)
      ≤ eta • (((A + eta • (1 : Matrix n n ℝ)) * A)⁻¹) := hscaled
    _ = A⁻¹ - (A + eta • (1 : Matrix n n ℝ))⁻¹ := hres.symm
    _ ≤ A⁻¹ - B⁻¹ := sub_le_sub_left hBinv _

/-- The quantitative gap in the shape FW-2's strictness clause is stated in: with
`δ = η / (Λ (Λ + η))`,

  `uᵀ A⁻¹ u - uᵀ B⁻¹ u ≥ δ ‖u‖²`   for every direction `u`,

where `‖u‖²` is `u ⬝ᵥ u`, the sum of squares of the coordinates. This is the input FW-2's
strictness argument integrates over the time set supplied by FW-3a; the integration and the
near-minimizing-path selection are not performed here. -/
theorem quadForm_inv_gap {A B : Matrix n n ℝ} {eta lam : ℝ}
    (hA : A.PosDef) (heta : 0 < eta) (hlam : 0 < lam)
    (hupper : A ≤ lam • (1 : Matrix n n ℝ)) (hgap : A + eta • (1 : Matrix n n ℝ) ≤ B)
    (u : n → ℝ) :
    (eta / (lam * (lam + eta))) * (u ⬝ᵥ u) ≤ u ⬝ᵥ (A⁻¹ *ᵥ u) - u ⬝ᵥ (B⁻¹ *ᵥ u) := by
  have h := quadForm_le_of_le (posDef_inv_gap hA heta hlam hupper hgap) u
  rwa [quadForm_smul_one, quadForm_sub] at h

/-- The bound of `posDef_inv_gap` is **attained**: the isotropic pair `A = Λ I`,
`B = (Λ + η) I` satisfies its hypotheses (`A ≤ Λ I` and `A + η I ≤ B` both hold, as equalities)
and turns its conclusion into an equality. -/
theorem inv_gap_isotropic_eq {lam eta : ℝ} (hlam : 0 < lam) (heta : 0 < eta) :
    (lam • (1 : Matrix n n ℝ))⁻¹ - ((lam + eta) • (1 : Matrix n n ℝ))⁻¹
      = (eta / (lam * (lam + eta))) • (1 : Matrix n n ℝ) := by
  have h2 : lam + eta ≠ 0 := by positivity
  rw [inv_smul_one hlam.ne', inv_smul_one h2, ← sub_smul]
  congr 1
  field_simp
  ring

/-- The constant `η / (Λ (Λ + η))` of `posDef_inv_gap` is **sharp**: any constant `c` that works
for the isotropic pair `A = Λ I`, `B = (Λ + η) I` — which satisfies that theorem's hypotheses —
already satisfies `c ≤ η / (Λ (Λ + η))`. The note's `δ` therefore cannot be improved under its
own hypotheses, and the eigenvalue-free derivation above loses nothing to the note's. -/
theorem inv_gap_constant_sharp [Nonempty n] {lam eta c : ℝ} (hlam : 0 < lam) (heta : 0 < eta)
    (h : c • (1 : Matrix n n ℝ)
      ≤ (lam • (1 : Matrix n n ℝ))⁻¹ - ((lam + eta) • (1 : Matrix n n ℝ))⁻¹) :
    c ≤ eta / (lam * (lam + eta)) := by
  rw [inv_gap_isotropic_eq hlam heta] at h
  exact smul_one_le_smul_one_iff.mp h

/-! ## Witnesses

The theorems above are conditional statements, and a conditional statement is worthless if
nothing satisfies its hypotheses, or if its hypotheses are satisfied so easily that the
conclusion carries no information. The results below rule both out by explicit construction in
`Matrix (Fin 2) (Fin 2) ℝ`: the hypotheses are satisfiable, and satisfiable anisotropically;
the Loewner order is neither trivial (`witness_not_ge`) nor total (`witness_incomparable`), so
`A ≤ B` is a real restriction on the pair; and the positive-definiteness hypothesis is
load-bearing (`posDef_hypothesis_is_necessary`). -/

/-- The smaller matrix of the witness pair, the identity, is positive definite. -/
theorem witness_posDef : (1 : Matrix (Fin 2) (Fin 2) ℝ).PosDef := posDef_one

/-- The witness pair satisfies the Loewner hypothesis of `posDef_inv_antitone`, and does so
anisotropically: the excess is `diagonal ![0, 1]`, which is positive semidefinite but singular
(`witness_excess_singular`). This is the situation FW-2 is actually in — the diffusion excess
acts non-degenerately only on part of the space. -/
theorem witness_le :
    (1 : Matrix (Fin 2) (Fin 2) ℝ) ≤ 1 + Matrix.diagonal ![0, 1] := by
  rw [Matrix.le_iff, add_sub_cancel_left]
  refine Matrix.posSemidef_diagonal_iff.mpr ?_
  intro i; fin_cases i <;> norm_num

/-- The excess of the witness pair is singular, so `witness_le` is a genuinely non-strict
Loewner inequality rather than a positive-definite one. -/
theorem witness_excess_singular : (Matrix.diagonal ![(0 : ℝ), 1]).det = 0 := by
  rw [Matrix.det_diagonal]
  simp

/-- The reverse Loewner inequality fails for the witness pair. Consequently the witness pair is
*strictly* ordered, so the hypothesis of `posDef_inv_strictAnti` is satisfiable too, and the
Loewner order is not the trivial "everything below everything" relation. -/
theorem witness_not_ge :
    ¬ ((1 : Matrix (Fin 2) (Fin 2) ℝ) + Matrix.diagonal ![0, 1] ≤ 1) := by
  intro h
  rw [Matrix.le_iff] at h
  have hd := h.diag_nonneg (i := 1)
  simp [Matrix.diagonal] at hd
  linarith

/-- The positive-definiteness hypothesis of `posDef_inv_antitone` is **load-bearing**: without
it the conclusion is false. Taking `A = 0` and `B = 1` gives `A ≤ B` while `B⁻¹ ≤ A⁻¹` fails,
because mathlib's inverse of a singular matrix is `0` and `1 ≤ 0` is false in the Loewner order.
The hypothesis therefore does not trivialize anything; it is what the theorem needs. -/
theorem posDef_hypothesis_is_necessary :
    ¬ (∀ A B : Matrix (Fin 2) (Fin 2) ℝ, A ≤ B → B⁻¹ ≤ A⁻¹) := by
  intro h
  have h0 : (0 : Matrix (Fin 2) (Fin 2) ℝ) ≤ 1 := witness_posDef.posSemidef.nonneg
  have hcon := h 0 1 h0
  rw [inv_one, Matrix.inv_zero, Matrix.le_iff] at hcon
  have hd := hcon.diag_nonneg (i := 0)
  norm_num at hd

/-- The Loewner order is **not total**: `diagonal ![1, 0]` and `diagonal ![0, 1]` are
incomparable in either direction. The hypothesis `A ≤ B` of `posDef_inv_antitone` is therefore a
genuine restriction on the pair, not a condition that any two positive-semidefinite matrices
automatically satisfy. -/
theorem witness_incomparable :
    ¬ ((Matrix.diagonal ![(1 : ℝ), 0] ≤ Matrix.diagonal ![(0 : ℝ), 1])
        ∨ (Matrix.diagonal ![(0 : ℝ), 1] ≤ Matrix.diagonal ![(1 : ℝ), 0])) := by
  rintro (hc | hc) <;> rw [Matrix.le_iff] at hc
  · have hd := hc.diag_nonneg (i := 0); simp [Matrix.diagonal] at hd; linarith
  · have hd := hc.diag_nonneg (i := 1); simp [Matrix.diagonal] at hd; linarith

/-- The hypotheses of `posDef_inv_gap` are satisfiable, and the conclusion is then an equality:
`A = I`, `B = 2 I`, `Λ = η = 1` give `δ = 1/2` and `A⁻¹ - B⁻¹ = (1/2) I` exactly. -/
theorem gap_hypotheses_satisfiable :
    ((1 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).PosDef ∧
      ((1 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) ≤ (1 : ℝ) • 1 ∧
      ((1 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) + (1 : ℝ) • 1 ≤ (2 : ℝ) • 1 ∧
      ((1 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ))⁻¹ - ((2 : ℝ) • 1)⁻¹
        = (1 / (1 * (1 + 1)) : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  refine ⟨posDef_smul_one one_pos, le_refl _, ?_, ?_⟩
  · rw [← add_smul]
    norm_num
  · have h := inv_gap_isotropic_eq (n := Fin 2) (lam := 1) (eta := 1) one_pos one_pos
    norm_num at h ⊢
    exact h

end IcnnLift
