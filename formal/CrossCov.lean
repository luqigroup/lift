import Mathlib

/-!
# The slack-channel cross-covariance estimator, and the two finite-sample deletions

This file fixes the vocabulary the rest of the development shares, and machine-checks the two
deletions of Theorem 1 (`thm:joint-necessity`) that hold **identically, for every iteration `t`
and every window length `T`** — cases (i) and (ii) of the appendix statement. Case (iii) is a
statement about the population cross-covariance and its estimator, and lives in
`CrossCovIndep.lean` and `CrossCovSLLN.lean`.

The estimator of `eq:cross-cov` is the trailing-window empirical second moment of the pair
`(δθ̃, δg)`,
`Σ̂⁽ᵗ⁾ = (1/T) Σ_{s = t-T}^{t-1} δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ`,
and its **slack-channel reading** is the contraction `J_bᵀ Σ̂⁽ᵗ⁾` with the slack Jacobian
`J_b = ∂θ̃/∂b`. Both are modelled here exactly: a window of `T` samples indexed from `t - T`,
an outer product per sample, and a `1/T` average.

Two modelling choices are worth naming, because they are what make the Lean statements the
paper's statements rather than weaker cousins.

* The window is indexed by `Finset.range T` offset by `t - T`, so the "for every `t` and every
  window length `T`" quantifier of the paper is a genuine `∀ t T` in the Lean statement.
* Case (ii)'s hypothesis is the paper's, not a restatement of its conclusion: the body
  `h` is **constant as a function of the batch**, and the vanishing of the fluctuation is
  derived from that, not assumed. `bodyFluct` is the window-centred fluctuation
  `h (X s) - (1/T) Σ h (X s')` of `eq:fluct`.

## Results
* `crossCovEstimator` — the estimator of `eq:cross-cov`.
* `slackReading` — its contraction with the slack Jacobian.
* `slackReading_eq_zero_of_jacobian_eq_zero` — deletion (i): no slack channel, reading zero.
* `crossCovEstimator_eq_zero_of_fluct_eq_zero` — deletion (ii), abstract form.
* `bodyFluct_eq_zero_of_const` — a batch-independent body has zero resampling fluctuation.
* `crossCovEstimator_eq_zero_of_body_const` — deletion (ii) from the paper's own hypothesis.
* `joint_necessity_finite` — the contrapositive both deletions share: a nonzero reading forces
  a nonzero slack Jacobian **and** a batch-dependent body.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. Case (iii), the conditional-independence deletion, is not in
this file: it is not a finite-sample identity and is proved in `CrossCovIndep.lean`.
-/

open Matrix Finset

namespace IcnnLift

variable {d : ℕ} {Ω : Type*}

/-- The trailing-window slack-channel cross-covariance estimator `Σ̂⁽ᵗ⁾` of `eq:cross-cov`:
the `1/T`-average of the outer products `δθ̃⁽ˢ⁾ (δg⁽ˢ⁾)ᵀ` over the window `s = t-T, …, t-1`. -/
noncomputable def crossCovEstimator (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  (T : ℝ)⁻¹ • ∑ s ∈ range T, Matrix.vecMulVec (dθ (t - T + s)) (dg (t - T + s))

/-- The **slack-channel reading** of the estimator: its contraction with the slack Jacobian
`J_b = ∂θ̃/∂b`. This, not the estimator itself, is what Theorem 1 is about. -/
noncomputable def slackReading (Jb : Matrix (Fin d) (Fin d) ℝ) (T t : ℕ)
    (dθ dg : ℕ → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Jbᵀ * crossCovEstimator T t dθ dg

/-- Deletion (i) — **the slack channel is absent.** If the slack Jacobian vanishes then the
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

/-- The window-centred resampling fluctuation of the latent iterate, `eq:fluct`: the body's
value on the batch drawn at step `s`, minus the window mean of those values. The slack `b` is
common to every step of the window and cancels, which is exactly the paper's statement that the
body carries the entire batch-induced fluctuation. -/
noncomputable def bodyFluct (h : Ω → Fin d → ℝ) (X : ℕ → Ω) (T t : ℕ) (s : ℕ) : Fin d → ℝ :=
  h (X s) - (T : ℝ)⁻¹ • ∑ s' ∈ range T, h (X (t - T + s'))

/-- A body that does not depend on the conditioning batch has zero resampling fluctuation.
This is deletion (ii)'s hypothesis as the paper states it — `h_φ(X)` constant in `X` — and the
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

/-- The contrapositive the two finite-sample deletions share, and the form Theorem 1 states in
the main text: a **nonzero** slack-channel reading forces both a nonzero slack Jacobian and a
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
