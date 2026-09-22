import Mathlib.Analysis.Matrix.Order

/-!
# Trace and quadratic-form toolkit for the cross-covariance argument

The corrected mechanism of the paper (`docs/design/lemma1_rederivation.md` §2) turns on two
matrix facts that mathlib does not name, and that every later module reuses.

The first is **self-duality of the positive-semidefinite cone in trace form**: for `V` and `H`
positive semidefinite, `0 ≤ tr (V * H)`. This is what signs the trace of the slack-channel
cross-covariance once (A4)'s forward-KL chain `δg = H δθ̃ + r` has been substituted, since the
leading term of `Σ_slack = E[δθ̃ δgᵀ]` is then `V * H` with `V = E[δθ̃ δθ̃ᵀ]` the iterate-fluctuation
covariance. mathlib has `Matrix.PosSemidef.trace_nonneg` for a *single* positive-semidefinite
matrix; the product statement is not there, and is proved here from the continuous functional
calculus square root, which matrices carry through their C⋆-algebra structure.

The second is the **cross-term quadratic form** `0 ≤ eᵀ (H * V * H) e` of the update-covariance
decomposition, the fact that makes the coupling channel add rather than subtract diffusion.

The third is bookkeeping: the trace does not see symmetrization, `tr (½(M + Mᵀ)) = tr M`.

## Results
* `trace_mul_nonneg_of_posSemidef` — `0 ≤ (V * H).trace` for `V H` positive semidefinite.
* `posSemidef_conj_symm` — `H * V * H` is positive semidefinite when `V` is and `H` is symmetric.
* `quadForm_conj_nonneg` — hence `0 ≤ e ⬝ᵥ ((H * V * H) *ᵥ e)` for every `e`.
* `trace_symmetrization` — `(½ • (M + Mᵀ)).trace = M.trace`.
* `trace_add_nonneg_of_posSemidef` — traces of positive-semidefinite matrices add non-negatively.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open Matrix
open scoped MatrixOrder

namespace IcnnLift

variable {n : Type*} [Fintype n]

/-- Self-duality of the positive-semidefinite cone, trace form: the trace of a product of two
positive-semidefinite matrices is non-negative.

Proof: write `V = S * S` with `S = CFC.sqrt V` positive semidefinite (hence Hermitian), then
`tr (V * H) = tr (S * (S * H)) = tr (S * H * S) = tr (Sᴴ * H * S)`, and `Sᴴ * H * S` is
positive semidefinite, so its trace is non-negative. -/
theorem trace_mul_nonneg_of_posSemidef [DecidableEq n] {V H : Matrix n n ℝ}
    (hV : V.PosSemidef) (hH : H.PosSemidef) : 0 ≤ (V * H).trace := by
  have hV0 : (0 : Matrix n n ℝ) ≤ V := Matrix.nonneg_iff_posSemidef.mpr hV
  set S : Matrix n n ℝ := CFC.sqrt V with hSdef
  have hSpsd : S.PosSemidef := Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg V)
  have hSS : S * S = V := CFC.sqrt_mul_sqrt_self V hV0
  have h1 : (V * H).trace = (S * (S * H)).trace := by rw [← Matrix.mul_assoc, hSS]
  have h2 : (S * (S * H)).trace = (S * H * S).trace := Matrix.trace_mul_comm S (S * H)
  have h3 : (S * H * S).PosSemidef := by
    have := hH.conjTranspose_mul_mul_same S
    rwa [hSpsd.isHermitian.eq] at this
  rw [h1, h2]
  exact h3.trace_nonneg

/-- The congruence `H * V * H` of a positive-semidefinite `V` by a symmetric `H` is itself
positive semidefinite. This is the cross term of the update-covariance decomposition. -/
theorem posSemidef_conj_symm {V H : Matrix n n ℝ}
    (hV : V.PosSemidef) (hH : H.IsHermitian) : (H * V * H).PosSemidef := by
  have := hV.conjTranspose_mul_mul_same H
  rwa [hH.eq] at this

/-- The cross-term quadratic form is non-negative in every direction: the coupling channel adds
diffusion rather than removing it. -/
theorem quadForm_conj_nonneg {V H : Matrix n n ℝ}
    (hV : V.PosSemidef) (hH : H.IsHermitian) (e : n → ℝ) :
    0 ≤ e ⬝ᵥ ((H * V * H) *ᵥ e) := by
  have h := (Matrix.posSemidef_iff_dotProduct_mulVec.mp (posSemidef_conj_symm hV hH)).2 e
  simpa using h

/-- The trace is insensitive to symmetrization: only the symmetric part of a cross-covariance
can contribute to a curvature, but the trace of the symmetric part is the trace itself. -/
theorem trace_symmetrization (M : Matrix n n ℝ) :
    (((1 : ℝ)/2) • (M + Mᵀ)).trace = M.trace := by
  rw [Matrix.trace_smul, Matrix.trace_add, Matrix.trace_transpose]
  ring

/-- Traces of positive-semidefinite matrices add non-negatively. -/
theorem trace_add_nonneg_of_posSemidef {A B : Matrix n n ℝ}
    (hA : A.PosSemidef) (hB : B.PosSemidef) : 0 ≤ (A + B).trace := by
  rw [Matrix.trace_add]
  exact add_nonneg hA.trace_nonneg hB.trace_nonneg

end IcnnLift
