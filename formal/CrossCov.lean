import Mathlib

/-!
# The slack-channel cross-covariance estimator, and the two finite-sample substitutions

This file fixes the vocabulary the rest of the development shares, and machine-checks the two
substitutions printed after Theorem 2 (`thm:joint-necessity`) of the paper, the ones that
hold **identically, for every iteration `t` and every window length `T`**: a latent weight with no
learnable slack, and a hypernetwork constant in the batch. In this file they are called deletions
(i) and (ii). Theorem 2 itself, the conditional-independence clause, is a statement about the
population coupling and its estimator, and lives in `CrossCovIndep.lean` and `CrossCovSLLN.lean`.

The estimator here is the **trailing-window** form of `eq:cross-cov`, the empirical second moment
of the pair `(δθ̃, δg)` over the last `T` training steps,
`Σ̂⁽ᵗ⁾ = (1/T) Σ_{s = t-T}^{t-1} δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ`,
and its **slack-channel reading** is the contraction `J_bᵀ Σ̂⁽ᵗ⁾` with the slack Jacobian
`J_b = ∂θ̃/∂b`. The paper's `eq:cross-cov` itself is the `T`-probe estimator at one iterate with
Bessel's divisor `1/(T − 1)`, and it is formalized in `CrossCovProbe.lean`. The opening of the
paper's appendix records that on the same data the trailing-window form with divisor `1/T`
differs from it by the factor `T/(T − 1)`, so every statement about one transports to the other;
the two substitutions below are summand-by-summand zeros and hold for both. The estimator and its
reading are modeled here exactly: a window of `T` samples indexed from `t - T`, an outer product per sample, and a `1/T`
average.

Two modeling choices are worth naming, because they are what make the Lean statements the
paper's statements rather than weaker cousins.

* The window is indexed by `Finset.range T` offset by `t - T`, so the "at every iterate and for
  every `T`" quantifier of the paper is a genuine `∀ t T` in the Lean statement.
* Deletion (ii)'s hypothesis is the paper's, not a restatement of its conclusion: the body
  `h` is **constant as a function of the batch**, and the vanishing of the fluctuation is
  derived from that, not assumed. `bodyFluct` is the window-centered fluctuation
  `h (X s) - (1/T) Σ h (X s')` of `eq:fluct`.

## Results
* `crossCovEstimator` — the trailing-window form of the estimator of `eq:cross-cov`.
* `slackReading` — its contraction with the slack Jacobian.
* `slackReading_eq_zero_of_jacobian_eq_zero` — deletion (i), the paper's first substitution: no
  learnable slack, reading zero.
* `crossCovEstimator_eq_zero_of_fluct_eq_zero` — deletion (ii), abstract form.
* `bodyFluct_eq_zero_of_const` — a batch-independent body has zero resampling fluctuation.
* `crossCovEstimator_eq_zero_of_body_const` — deletion (ii), the paper's second substitution, from
  the paper's own hypothesis.
* `joint_necessity_finite` — the contrapositive both substitutions share, displayed in the
  appendix proof of Theorem 2: a nonzero reading forces a nonzero slack Jacobian **and** a
  batch-dependent body.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. Theorem 2's own clause, conditional independence, is not in
this file: it is not a finite-sample identity and is proved in `CrossCovIndep.lean` and
`CrossCovSLLN.lean`.
-/

open Matrix Finset

namespace IcnnLift

variable {d : ℕ} {Ω : Type*}

/-- The trailing-window form of the slack-channel cross-covariance estimator `Σ̂⁽ᵗ⁾` of
`eq:cross-cov` (the paper's display is the `T`-probe form with divisor `1/(T − 1)`, in
`CrossCovProbe.lean`): the `1/T`-average of the outer products `δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ` over the window `s = t-T, …, t-1`. -/
noncomputable def crossCovEstimator (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  (T : ℝ)⁻¹ • ∑ s ∈ range T, Matrix.vecMulVec (dθ (t - T + s)) (dg (t - T + s))

/-- The **slack-channel reading** of the estimator: its contraction with the slack Jacobian
`J_b = ∂θ̃/∂b`. This is the object of the first substitution after Theorem 2: the paper's
conclusion concerns the coupling in the update of the slack, `(∂θ̃/∂b)ᵀ Σ_ξ`. -/
noncomputable def slackReading (Jb : Matrix (Fin d) (Fin d) ℝ) (T t : ℕ)
    (dθ dg : ℕ → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Jbᵀ * crossCovEstimator T t dθ dg

/-- Deletion (i), the paper's first substitution — **no learnable slack.** If the slack Jacobian vanishes then the
slack-channel reading vanishes identically, for every iteration and every window length: each
summand is pre-multiplied by the zero Jacobian. -/
theorem slackReading_eq_zero_of_jacobian_eq_zero {Jb : Matrix (Fin d) (Fin d) ℝ}
    (hJ : Jb = 0) (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    slackReading Jb T t dθ dg = 0 := by
  simp [slackReading, hJ]

/-- Deletion (ii), abstract form — a vanishing iterate fluctuation zeroes every summand, hence
the estimator itself, for every iteration and every window length. -/
theorem crossCovEstimator_eq_zero_of_fluct_eq_zero {dθ dg : ℕ → Fin d → ℝ}
    (hθ : ∀ s, dθ s = 0) (T t : ℕ) :
    crossCovEstimator T t dθ dg = 0 := by
  have : ∀ s ∈ range T,
      Matrix.vecMulVec (dθ (t - T + s)) (dg (t - T + s)) = (0 : Matrix (Fin d) (Fin d) ℝ) := by
    intro s _
    ext i j
    simp [Matrix.vecMulVec_apply, hθ]
  simp [crossCovEstimator, Finset.sum_congr rfl this]

/-- The window-centered resampling fluctuation of the latent iterate, `eq:fluct`: the body's
value on the batch drawn at step `s`, minus the window mean of those values. The slack `b` is
common to every step of the window and cancels, which is exactly the paper's statement that the
body carries the entire batch-induced fluctuation. -/
noncomputable def bodyFluct (h : Ω → Fin d → ℝ) (X : ℕ → Ω) (T t : ℕ) (s : ℕ) : Fin d → ℝ :=
  h (X s) - (T : ℝ)⁻¹ • ∑ s' ∈ range T, h (X (t - T + s'))

/-- A body that does not depend on the conditioning batch has zero resampling fluctuation.
This is the second substitution's hypothesis as the paper states it — a hypernetwork constant in
the batch — and the
vanishing of `δθ̃` is derived from it rather than assumed. -/
theorem bodyFluct_eq_zero_of_const {h : Ω → Fin d → ℝ} (hconst : ∀ x y, h x = h y)
    (X : ℕ → Ω) (T t s : ℕ) (hT : T ≠ 0) : bodyFluct h X T t s = 0 := by
  have hmean : ∑ s' ∈ range T, h (X (t - T + s')) = (T : ℝ) • h (X s) := by
    rw [Finset.sum_congr rfl (fun s' _ => hconst (X (t - T + s')) (X s)), Finset.sum_const,
      Finset.card_range, ← Nat.cast_smul_eq_nsmul ℝ]
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  ext i
  simp [bodyFluct, hmean, smul_smul, inv_mul_cancel₀ hTne]

/-- Deletion (ii) from the paper's own hypothesis: a batch-independent body zeroes the estimator
identically, for every iteration and every window length. -/
theorem crossCovEstimator_eq_zero_of_body_const {h : Ω → Fin d → ℝ}
    (hconst : ∀ x y, h x = h y) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct h X T t) dg = 0 :=
  crossCovEstimator_eq_zero_of_fluct_eq_zero
    (fun s => bodyFluct_eq_zero_of_const hconst X T t s hT) T t

/-- The contrapositive the two finite-sample substitutions share, the display the appendix proof
of Theorem 2 calls the sentence printed after the theorem: a **nonzero** slack-channel reading forces both a nonzero slack Jacobian and a
batch-dependent body. -/
theorem joint_necessity_finite {Jb : Matrix (Fin d) (Fin d) ℝ} {dθ dg : ℕ → Fin d → ℝ} {T t : ℕ}
    (hne : slackReading Jb T t dθ dg ≠ 0) :
    Jb ≠ 0 ∧ ∃ s, dθ s ≠ 0 := by
  constructor
  · intro hJ
    exact hne (slackReading_eq_zero_of_jacobian_eq_zero hJ T t dθ dg)
  · by_contra hall
    have hz : ∀ s, dθ s = 0 := fun s => by
      by_contra hs
      exact hall ⟨s, hs⟩
    exact hne (by
      simp [slackReading, crossCovEstimator_eq_zero_of_fluct_eq_zero hz T t])

end IcnnLift
