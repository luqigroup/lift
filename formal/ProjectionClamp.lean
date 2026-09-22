import Mathlib

/-!
# One projected coordinate at the boundary: the two elementary clauses of `prop:clamp`

Proposition `prop:clamp` of the paper (Section 3.3 of `paper_v5.tex`, proved in
`app_proofs.tex`) is a model of a single constrained coordinate under projected stochastic
gradient descent, held at the boundary of the cone. In the paper's letters: `γ, σ, η > 0`,
`ζ₁, ζ₂, …` independent standard normal with distribution function `Φ`,

    W_{k+1} = max(0, W_k - η(γ + σ ζ_k)),      ρ = γ/σ,

and the proposition makes three assertions.

1. **The dwell.** "A visit to the boundary ends at the first step whose gradient points into the
   cone, so it lasts a geometric number of steps of mean `1/Φ(-ρ)`."
2. **The clamped fraction.** "The recursion has a unique stationary law, under which
   `P(W_∞ = 0) = exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))`, a function of `ρ` alone that increases from `0`
   to `1`."
3. **The offset.** "`E[W_∞] = η σ m(ρ)` with `m` finite and depending on `ρ` alone."

**This file formalizes clause 1 in full, and the homogeneity content of clause 3. It does not
formalize clause 2 at all, and it does not prove that `m` is finite.** That split is not a
matter of taste: clause 2 is the Spitzer/Sparre-Andersen ladder-epoch identity for a Gaussian
random walk, and this mathlib (v4.31.0) has no fluctuation theory — no ladder heights, no
Sparre-Andersen combinatorial lemma, no Spitzer identity, and no Wiener--Hopf factorization. A
whole-tree search for `ladder`, `Spitzer`, `Sparre` and `fluctuation` returns nothing usable.
The same is true of the finiteness of `m = E[sup_j S_j]`, which in the paper's proof is a
citation to Asmussen's negative-drift theory. Both stay cited classical facts, exactly as the
paper's proof uses them, and neither is asserted anywhere in this file.

## What clause 1 becomes here

The paper's "first step whose gradient points into the cone" is `VisitEndsAt`: the first `n`
draws leave the iterate clamped and the draw at index `n` releases it. That this is the
*first* such step, in the literal sense of an infimum, is `clampRun_eq_of_visitEndsAt` and
`visitEndsAt_clampRun`, which identify `VisitEndsAt ρ z n` with `sInf {k | z k < -ρ} = n`.
That a clamped step is released exactly when `ζ < -ρ` is `clampStep_pos_iff`, proved from the
paper's own display `max(0, 0 - η(γ + σζ))` with `ρ = γ/σ`, so the reduction of the
two-parameter recursion to the single ratio `ρ` is machine-checked rather than asserted
(`eta_mul_add_eq`, `clampIterate_succ_paper`).

The law is then `measureReal_visitEndsAt`: `P(visit ends at n) = (1 - p)ⁿ p` with
`p = Φ(-ρ)`, proved from mutual independence through `iIndepFun.meas_biInter`, and the mean is
`tsum_visitLength`: `∑ₙ (n+1) · P(visit ends at n) = 1/Φ(-ρ)`. **The off-by-one is the paper's
own convention and is worth naming.** `VisitEndsAt ρ z n` holds when the visit lasts `n + 1`
steps: `n` steps on which the clamp is active and one further step that ends the visit. The
paper's "geometric number of steps of mean `1/Φ(-ρ)`" is the length `n + 1`, which is why the
summand carries `(n + 1)` and the total is `1/p` rather than `(1-p)/p`. The index `n` alone is
geometric in mathlib's convention (`geometricMeasure`, the number of failures before the first
success), with mean `(1-p)/p`.

`Φ` is `ProbabilityTheory.cdf (gaussianReal 0 1)`, so `clampExitProb ρ = Φ(-ρ)` is the genuine
standard-normal distribution function and not a stand-in; `clampExitProb_pos` and
`clampExitProb_lt_one` are the two facts the geometric law needs, and both come from the
Gaussian's density being positive everywhere.

## What clause 3 becomes here

`clampIterate c ρ z` is the Lindley recursion at scale `c`; at `c = η σ` it is the paper's
recursion verbatim (`clampIterate_succ_paper`). `clampIterate_homogeneous` is the algebraic
heart: for every `c ≥ 0` and every path, `W_k^{(c)} = c · W_k^{(1)}`, by induction, with no
probability at all. Because the unit-scale recursion contains neither `η` nor `σ` — only `ρ`
and the noise — every functional of it is "a function of `ρ` alone" in the paper's sense, and
the scaling passes to expectations (`integral_clampIterate_homogeneous`), to limits of
expectations (`tendsto_integral_clampIterate_homogeneous`), and to the pathwise supremum
`M = sup_j S_j` that the paper's proof uses (`iSup_clampIterate_homogeneous`) — a scaling
identity about that supremum, asserting nothing about its existence or finiteness, and
junk-valued (`0 = c·0`) on a path where the supremum is infinite.

**The existence of the stationary mean is a hypothesis here, not a theorem.** The paper asserts
`E[W_∞] = η σ m(ρ)`; what is proved here is that *if* the unit-scale means converge to `m` then
the `η σ`-scale means converge to `η σ m`. Convergence itself, and the finiteness of `m`, are
the Asmussen citation of the paper's proof, and are not formalized. This is the one place where
the Lean statement is weaker than the LaTeX one, and it is weaker in the same way the paper's
own proof is: the proof establishes the scaling and cites the rest.

## Results
* `eta_mul_add_eq` — the paper's increment `η(γ + σζ)` is `ησ(ρ + ζ)`: the reduction to `ρ`.
* `clampStep_pos_iff`, `clampStep_eq_zero_iff` — a clamped step is released exactly when
  `ζ < -ρ`, which is the paper's "the first step whose gradient points into the cone".
* `clampExitProb` — `Φ(-ρ)`, with `clampExitProb_pos`, `clampExitProb_lt_one`,
  `clampExitProb_eq_measureReal_Iio`.
* `VisitEndsAt`, `clampRun`, `clampRun_eq_of_visitEndsAt`, `visitEndsAt_clampRun` — the visit
  length and its identification with a genuine first passage.
* `measure_visitEndsAt`, `measureReal_visitEndsAt` — the geometric law `(1-p)ⁿ p`;
  `measureReal_visitEndsAt_eq_geometricMeasure` names it as mathlib's own geometric
  distribution, and `measureReal_clampRun_eq` reads it on the first-passage index.
* `tsum_visitLength` — the mean visit length `1/Φ(-ρ)`; `tsum_measureReal_visitEndsAt` records
  that the law is a probability distribution on the visit length.
* `clamp_hypotheses_satisfiable` — the hypothesis stack is inhabited: an i.i.d. standard-normal
  sequence exists (`ProbabilityTheory.exists_iid`), so the geometric law is not vacuous.
* `clampIterate`, `clampIterate_succ_paper` — the Lindley recursion, in the paper's letters.
* `clampIterate_eq_zero_of_forall_le`, `clampIterate_eq_zero_of_visitEndsAt`,
  `clampIterate_pos_of_visitEndsAt` — the visit read on the iterate: through a visit `W` stays
  at the boundary, and at the step the visit names it is released.
* `clampIterate_homogeneous`, `iSup_clampIterate_homogeneous`,
  `integral_clampIterate_homogeneous`, `tendsto_integral_clampIterate_homogeneous` — the
  homogeneity: `η σ` times a quantity in which neither `η` nor `σ` appears.
* `clampIterate_eq_zero_iff_unit`, `measureReal_clampIterate_eq_zero_scale_free` — the clamped
  fraction at every step is the same number at every scale `ησ`: a function of `ρ` and the law of
  the draws alone.
* `clampIterate_antitone_in_rho`, `clampIterate_eq_zero_of_le_rho`,
  `measureReal_clampIterate_eq_zero_mono_rho` — the coupling: a larger ratio gives a pointwise
  smaller iterate on the same draws, so the clamped fraction is non-decreasing in `ρ`.
* `measure_forall_le`, `measureReal_clampIterate_eq_zero_ge`,
  `tendsto_measureReal_clampIterate_eq_zero_atTop` — the clamped fraction at every step is at
  least `(1 - Φ(-ρ))^k` and tends to `1` as `ρ → ∞`.
* `walkSum`, `walkSum_homogeneous`, `iSup_walkSum_homogeneous` — the paper's own
  `M = ησ · sup_j(-∑_{i≤j}(ρ + ζ_i))`, and its `ησ` scaling.
* `walkSum_succ`, `clampIterate_eq_sup'_sub` — the recursion unrolled:
  `W_k = max_{0≤j≤k}(S_k - S_j)`, pathwise.
* `maxWalk`, `maxWalk_eq_sup'`, `revDraws`, `revIndex`, `revIndex_involutive`,
  `walkSum_revDraws`, `clampIterate_eq_maxWalk_revDraws` — reversing the first `k` draws turns
  those rises into partial sums from the start, still pathwise.
* `map_revDraws_eq`, `map_clampIterate_eq_map_maxWalk` — **the reversal identity**:
  `W_k =ᵈ max_{0≤j≤k} S_j`, because an i.i.d. block has the same law reversed. This is the step
  that carries the facts below about the walk over to the recursion.
* `hasSubgaussianMGF_of_hasLaw`, `hasSubgaussianMGF_negSum`, `measureReal_walk_pos_le`,
  `ae_bddAbove_walkSum` — **`M` is finite almost surely**: the draws are sub-Gaussian with
  constant one, Chernoff gives `P(S_j > 0) ≤ exp(-ρ²j/2)`, and Borel--Cantelli leaves only
  finitely many positive partial sums. No strong law, no fluctuation theory.
* `le_inv_mul_exp`, `lintegral_ofReal_walkSum_le`, `lintegral_iSup_ofReal_walkSum_lt_top` —
  **`M` has a finite mean**: the supremum is at most the sum of the positive parts, and the
  tilted bound makes those a geometric series.
* `lintegral_iSup_ofReal_walkSum_homogeneous` — and that mean scales by `ησ`, so it is
  `ησ · m(ρ)` with `m(ρ)` finite and free of `η` and `σ`.
* `maxWalk_mono`, `tendsto_maxWalk_atTop`, `tendsto_integral_clampIterate` — **`W_k → M` in
  distribution**: the running maximum increases to the walk's supremum almost surely, and the
  reversal identity carries that to the laws of the iterates, tested against bounded continuous
  functions.
* `offsetWitnessPath`, `clampIterate_offsetWitnessPath_succ`,
  `homogeneity_hypotheses_satisfiable` — the offset clause's hypothesis is inhabited, at a
  nonzero limit `m = 1`, with the instantiated conclusion `η σ` carried in the statement.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. What is carried rather than proved:

* **Clause 2 of `prop:clamp`: three of its four assertions are proved at every step of the
  recursion, and the stationary law itself is not.** What is proved (2026-09-13) is that the
  clamped fraction `P(W_k = 0)` is scale-free, hence a function of `ρ` and the law of the draws
  alone; that it is non-decreasing in `ρ`, by a pathwise coupling; and that it tends to `1` as
  `ρ → ∞`. What is **not** proved, and is not assumed either: the Spitzer series for the
  stationary clamped fraction, and the limit `0` as `ρ ↓ 0` that the series supplies. (That
  `P(W_k = 0)` converges as `k → ∞` does not follow from `tendsto_integral_clampIterate`: the
  indicator of `{0}` is not continuous and the limit law has an atom there. The monotone-events
  argument that would give it is not formalized.)
  The last is not available at finite `k` — at `ρ = 0` the clamped fraction at step `k` is
  positive — so it is a statement about the limit object alone. Its two classical routes are the
  Spitzer series, which needs fluctuation theory mathlib does not carry, and a zero-one argument
  at zero drift followed by a continuity-in-`ρ` passage; the second is supported in principle
  (mathlib has the Kolmogorov zero-one machinery in `Probability/Independence/ZeroOne.lean`, the
  Gaussian convolution, and a central limit theorem) but needs tail-σ-algebra scaffolding and a
  parametric continuity argument that this development has no analogue for.
* **The finiteness of `m` is proved for the paper's `M`, not for the stationary law.**
  `lintegral_iSup_ofReal_walkSum_lt_top` shows `E[sup_j S_j] < ∞` and
  `ae_bddAbove_walkSum` that the supremum is finite almost surely — the two facts the paper's
  proof takes from Asmussen — and
  `lintegral_iSup_ofReal_walkSum_homogeneous` gives `E[M^{(ησ)}] = ησ · E[M^{(1)}]`. The
  reversal identity `W_k =ᵈ max_{j≤k} S_j` is proved as well
  (`map_clampIterate_eq_map_maxWalk`), and so is the passage to the limit,
  `W_k → M` in distribution (`tendsto_integral_clampIterate`). **What is still not proved is
  that the law of `M` is stationary for the recursion, or that it is the unique such law**:
  nothing here states invariance. The paper's "the recursion has a unique stationary law" is
  therefore cited, while "the iterates converge in distribution to the law of `M`, which is
  finite with finite mean `ησ m(ρ)`" is proved.
  `tendsto_integral_clampIterate_homogeneous` still carries the convergence of the *means* of
  the recursion as a hypothesis — weak convergence does not deliver convergence of means without
  uniform integrability — inhabited by `homogeneity_hypotheses_satisfiable`.
* **The dwell is stated for the boundary visit, not for the full recursion.** The geometric law
  is proved for `VisitEndsAt`, a predicate on the draws, and the bridge to the iterate is
  `clampIterate_eq_zero_of_visitEndsAt` (through a visit `W` is `0` at every step up to and
  including `n`) together with `clampIterate_pos_of_visitEndsAt` (at step `n + 1` it is
  released), added 2026-09-13 so that the paper's "a visit to the boundary" is a statement about
  `W` and not only about `ζ`. What is still **not** proved is that the recursion ever returns to
  the boundary: recurrence is part of the stationary-law clause and stays unclaimed.
* **Independence is mutual and the laws are exact.** `measure_visitEndsAt` assumes
  `iIndepFun ζ P` (mutual independence, the paper's "independent") and
  `HasLaw (ζ i) (gaussianReal 0 1) P` (exactly standard normal, the paper's hypothesis). No
  weaker hypothesis is used and none is claimed.
* The single coordinate, the constant drift `γ`, the Gaussian noise and the absence of
  curvature are idealizations of the paper's own statement, which says so: `prop:clamp` is "a
  model of one constrained coordinate at the boundary, not a statement about the network".
-/

open MeasureTheory Filter Set ProbabilityTheory
open scoped Topology ENNReal NNReal

namespace IcnnLift

/-! ### The clamped step, and the reduction to the ratio `ρ = γ/σ` -/

/-- The paper's increment `η(γ + σζ)` is `ησ(ρ + ζ)` with `ρ = γ/σ`: the two-parameter
recursion depends on `(γ, σ)` only through the scale `ησ` and the ratio `ρ`. -/
theorem eta_mul_add_eq (eta gamma sigma z : ℝ) (hsigma : sigma ≠ 0) :
    eta * (gamma + sigma * z) = eta * sigma * (gamma / sigma + z) := by
  field_simp

/-- **A clamped step is released exactly when the draw is below `-ρ`.** From `W_k = 0` the next
iterate `max(0, 0 - η(γ + σζ))` is strictly positive precisely when `ζ < -γ/σ = -ρ`. This is
the paper's "a visit to the boundary ends at the first step whose gradient points into the
cone", written out on the paper's own display. -/
theorem clampStep_pos_iff {eta gamma sigma z : ℝ} (heta : 0 < eta) (hsigma : 0 < sigma) :
    0 < max 0 (0 - eta * (gamma + sigma * z)) ↔ z < -(gamma / sigma) := by
  have hiff : (0 : ℝ) < 0 - eta * (gamma + sigma * z) ↔ z < -(gamma / sigma) := by
    rw [show -(gamma / sigma) = -gamma / sigma by ring, lt_div_iff₀ hsigma]
    constructor
    · intro h
      have h2 : eta * (gamma + sigma * z) < 0 := by linarith
      have h3 : gamma + sigma * z < 0 := by
        by_contra hc
        rw [not_lt] at hc
        nlinarith [mul_nonneg heta.le hc]
      linarith
    · intro h
      have h3 : gamma + sigma * z < 0 := by linarith
      have h4 : eta * (gamma + sigma * z) < 0 := mul_neg_of_pos_of_neg heta h3
      linarith
  rw [lt_max_iff, hiff]
  simp

/-- The complementary form: the step stays clamped exactly when the draw is at or above `-ρ`. -/
theorem clampStep_eq_zero_iff {eta gamma sigma z : ℝ} (heta : 0 < eta) (hsigma : 0 < sigma) :
    max 0 (0 - eta * (gamma + sigma * z)) = 0 ↔ -(gamma / sigma) ≤ z := by
  constructor
  · intro h
    by_contra hc
    rw [not_le] at hc
    have := (clampStep_pos_iff heta hsigma).mpr hc
    rw [h] at this
    exact lt_irrefl 0 this
  · intro h
    by_contra hc
    have hpos : 0 < max 0 (0 - eta * (gamma + sigma * z)) :=
      lt_of_le_of_ne (le_max_left _ _) (Ne.symm hc)
    have := (clampStep_pos_iff heta hsigma).mp hpos
    linarith

/-! ### `Φ(-ρ)`, the release probability of one clamped step -/

/-- `Φ(-ρ)`: the standard-normal distribution function at `-ρ`, the probability that one draw
releases a clamped step. -/
noncomputable def clampExitProb (rho : ℝ) : ℝ :=
  ProbabilityTheory.cdf (gaussianReal 0 1) (-rho)

theorem clampExitProb_eq_measureReal_Iic (rho : ℝ) :
    clampExitProb rho = (gaussianReal 0 1).real (Iic (-rho)) :=
  ProbabilityTheory.cdf_eq_real _ _

/-- `Φ(-ρ)` is the mass of the open half-line, the Gaussian having no atoms; this is the form
the geometric law consumes, the release event being `ζ < -ρ`. -/
theorem clampExitProb_eq_measureReal_Iio (rho : ℝ) :
    clampExitProb rho = (gaussianReal 0 1).real (Iio (-rho)) := by
  have : NoAtoms (gaussianReal (0 : ℝ) 1) := noAtoms_gaussianReal one_ne_zero
  rw [clampExitProb_eq_measureReal_Iic]
  exact (measureReal_congr Iio_ae_eq_Iic).symm

theorem measure_Iio_gaussian (rho : ℝ) :
    gaussianReal (0 : ℝ) 1 (Iio (-rho)) = ENNReal.ofReal (clampExitProb rho) := by
  rw [clampExitProb_eq_measureReal_Iio, measureReal_def,
    ENNReal.ofReal_toReal (measure_ne_top _ _)]

/-- The release probability is positive: the Gaussian charges every half-line, because Lebesgue
measure is absolutely continuous with respect to it. -/
theorem clampExitProb_pos (rho : ℝ) : 0 < clampExitProb rho := by
  rw [clampExitProb_eq_measureReal_Iio, measureReal_def]
  refine ENNReal.toReal_pos ?_ (measure_ne_top _ _)
  intro h
  have hac := gaussianReal_absolutelyContinuous' (0 : ℝ) (v := 1) one_ne_zero
  have hv : (volume : Measure ℝ) (Iio (-rho)) = 0 := hac h
  rw [Real.volume_Iio] at hv
  exact ENNReal.top_ne_zero hv

/-- The release probability is below one: the Gaussian also charges the complementary
half-line. -/
theorem clampExitProb_lt_one (rho : ℝ) : clampExitProb rho < 1 := by
  have hcompl : gaussianReal (0 : ℝ) 1 (Ici (-rho)) ≠ 0 := by
    intro h
    have hac := gaussianReal_absolutelyContinuous' (0 : ℝ) (v := 1) one_ne_zero
    have hv : (volume : Measure ℝ) (Ici (-rho)) = 0 := hac h
    rw [Real.volume_Ici] at hv
    exact ENNReal.top_ne_zero hv
  have hlt : gaussianReal (0 : ℝ) 1 (Iio (-rho)) < 1 := by
    have hc : gaussianReal (0 : ℝ) 1 (Iio (-rho))ᶜ = 1 - gaussianReal (0 : ℝ) 1 (Iio (-rho)) :=
      prob_compl_eq_one_sub measurableSet_Iio
    rw [compl_Iio] at hc
    have hpos : 0 < 1 - gaussianReal (0 : ℝ) 1 (Iio (-rho)) := by
      rw [← hc]
      exact pos_iff_ne_zero.mpr hcompl
    exact tsub_pos_iff_lt.mp hpos
  rw [clampExitProb_eq_measureReal_Iio, measureReal_def]
  have := (ENNReal.toReal_lt_toReal (measure_ne_top _ _) (by simp)).mpr hlt
  simpa using this

/-! ### The visit length, and its geometric law -/

/-- **The visit to the boundary ends at step `n`.** The first `n` draws leave the iterate
clamped (`-ρ ≤ z j`, by `clampStep_eq_zero_iff`) and the draw at index `n` releases it
(`z n < -ρ`, by `clampStep_pos_iff`). The visit then lasts `n + 1` steps: `n` clamped steps and
the step that ends it. -/
def VisitEndsAt (rho : ℝ) (z : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ j < n, -rho ≤ z j) ∧ z n < -rho

/-- The index of the first releasing draw: a genuine first passage of the noise sequence below
`-ρ`. -/
noncomputable def clampRun (rho : ℝ) (z : ℕ → ℝ) : ℕ := sInf {k | z k < -rho}

/-- `VisitEndsAt` really is the *first* releasing step. -/
theorem clampRun_eq_of_visitEndsAt {rho : ℝ} {z : ℕ → ℝ} {n : ℕ} (h : VisitEndsAt rho z n) :
    clampRun rho z = n := by
  have hne : {k | z k < -rho}.Nonempty := ⟨n, h.2⟩
  have hle : clampRun rho z ≤ n := Nat.sInf_le h.2
  rcases lt_or_eq_of_le hle with hlt | heq
  · have hmem : z (clampRun rho z) < -rho := Nat.sInf_mem hne
    exact absurd hmem (not_lt.mpr (h.1 _ hlt))
  · exact heq

/-- Conversely, if some draw releases the step then the first one that does ends the visit. -/
theorem visitEndsAt_clampRun {rho : ℝ} {z : ℕ → ℝ} (h : ∃ k, z k < -rho) :
    VisitEndsAt rho z (clampRun rho z) := by
  refine ⟨fun j hj => ?_, Nat.sInf_mem h⟩
  have := Nat.notMem_of_lt_sInf hj
  simpa using this

variable {Ω : Type*} [MeasurableSpace Ω]

omit [MeasurableSpace Ω] in
/-- The visit-ends-at-`n` event as an intersection of `n + 1` single-draw events, the form the
independence hypothesis consumes. -/
theorem visitEndsAt_setOf (rho : ℝ) (zeta : ℕ → Ω → ℝ) (n : ℕ) :
    {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = ⋂ j ∈ Finset.range (n + 1),
          zeta j ⁻¹' (if j < n then Ici (-rho) else Iio (-rho)) := by
  ext ω
  simp only [mem_setOf_eq, VisitEndsAt, mem_iInter, Finset.mem_range, mem_preimage]
  constructor
  · rintro ⟨h1, h2⟩ j hj
    by_cases hjn : j < n
    · simpa [hjn] using h1 j hjn
    · have hjeq : j = n := by omega
      subst hjeq
      simpa [hjn] using h2
  · intro h
    refine ⟨fun j hj => ?_, ?_⟩
    · have := h j (by omega)
      simpa [hj] using this
    · have := h n (by omega)
      simpa using this

theorem measure_preimage_Iio {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (rho : ℝ) (j : ℕ) :
    P (zeta j ⁻¹' Iio (-rho)) = ENNReal.ofReal (clampExitProb rho) := by
  have h := (hlaw j).measure_eq (p := fun x : ℝ => x < -rho) measurableSet_Iio
  rw [show zeta j ⁻¹' Iio (-rho) = {ω | zeta j ω < -rho} from rfl, h]
  exact measure_Iio_gaussian rho

theorem measure_Ici_gaussian (rho : ℝ) :
    gaussianReal (0 : ℝ) 1 (Ici (-rho)) = ENNReal.ofReal (1 - clampExitProb rho) := by
  have hc : gaussianReal (0 : ℝ) 1 (Iio (-rho))ᶜ = 1 - gaussianReal (0 : ℝ) 1 (Iio (-rho)) :=
    prob_compl_eq_one_sub measurableSet_Iio
  rw [compl_Iio] at hc
  rw [hc, measure_Iio_gaussian, ENNReal.ofReal_sub _ (clampExitProb_pos rho).le,
    ENNReal.ofReal_one]

theorem measure_preimage_Ici {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (rho : ℝ) (j : ℕ) :
    P (zeta j ⁻¹' Ici (-rho)) = ENNReal.ofReal (1 - clampExitProb rho) := by
  have h := (hlaw j).measure_eq (p := fun x : ℝ => -rho ≤ x) measurableSet_Ici
  rw [show zeta j ⁻¹' Ici (-rho) = {ω | -rho ≤ zeta j ω} from rfl, h]
  exact measure_Ici_gaussian rho

/-- **The geometric law of the visit length.** Under mutual independence and the standard-normal
law of the draws, the probability that the visit ends at step `n` is `(1 - Φ(-ρ))ⁿ Φ(-ρ)`: `n`
draws that leave the step clamped, then one that releases it. -/
theorem measure_visitEndsAt {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P)
    (rho : ℝ) (n : ℕ) :
    P {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = ENNReal.ofReal ((1 - clampExitProb rho) ^ n * clampExitProb rho) := by
  classical
  set s : ℕ → Set Ω := fun j => zeta j ⁻¹' (if j < n then Ici (-rho) else Iio (-rho)) with hs
  have hmeas : ∀ j ∈ Finset.range (n + 1),
      MeasurableSet[MeasurableSpace.comap (zeta j) inferInstance] (s j) := by
    intro j _
    refine ⟨if j < n then Ici (-rho) else Iio (-rho), ?_, rfl⟩
    split_ifs
    · exact measurableSet_Ici
    · exact measurableSet_Iio
  have h1 : ∀ j ∈ Finset.range n, P (s j) = ENNReal.ofReal (1 - clampExitProb rho) := by
    intro j hj
    rw [Finset.mem_range] at hj
    simp only [hs, if_pos hj]
    exact measure_preimage_Ici hlaw rho j
  have h2 : P (s n) = ENNReal.ofReal (clampExitProb rho) := by
    simp only [hs, if_neg (lt_irrefl n)]
    exact measure_preimage_Iio hlaw rho n
  have hnn : (0 : ℝ) ≤ 1 - clampExitProb rho := by
    have := clampExitProb_lt_one rho
    linarith
  rw [visitEndsAt_setOf rho zeta n, hindep.meas_biInter hmeas, Finset.prod_range_succ,
    Finset.prod_congr rfl h1, Finset.prod_const, Finset.card_range, h2,
    ← ENNReal.ofReal_pow hnn, ← ENNReal.ofReal_mul (pow_nonneg hnn n)]

/-- The geometric law, in real form. -/
theorem measureReal_visitEndsAt {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P)
    (rho : ℝ) (n : ℕ) :
    P.real {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = (1 - clampExitProb rho) ^ n * clampExitProb rho := by
  have hnn : (0 : ℝ) ≤ 1 - clampExitProb rho := by
    have := clampExitProb_lt_one rho
    linarith
  rw [measureReal_def, measure_visitEndsAt hindep hlaw rho n,
    ENNReal.toReal_ofReal (mul_nonneg (pow_nonneg hnn n) (clampExitProb_pos rho).le)]

theorem clampExitProb_mem_unitInterval (rho : ℝ) : clampExitProb rho ∈ unitInterval :=
  ⟨(clampExitProb_pos rho).le, (clampExitProb_lt_one rho).le⟩

/-- **The law is geometric in mathlib's own sense**: the visit-ends-at-`n` event carries exactly
the mass that mathlib's geometric distribution of success probability `Φ(-ρ)` gives to `n`, the
number of clamped steps that precede the release. This is the paper's word "geometric",
machine-checked against the library's own definition rather than against a formula. -/
theorem measureReal_visitEndsAt_eq_geometricMeasure {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P)
    (rho : ℝ) (n : ℕ) :
    P.real {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = (ProbabilityTheory.geometricMeasure
          ⟨clampExitProb rho, clampExitProb_mem_unitInterval rho⟩).real {n} := by
  have hne : (⟨clampExitProb rho, clampExitProb_mem_unitInterval rho⟩ : unitInterval) ≠ 0 := by
    intro h
    have h0 := Subtype.ext_iff.mp h
    simp only at h0
    exact (clampExitProb_pos rho).ne' h0
  rw [ProbabilityTheory.geometricMeasure_real_singleton hne]
  exact measureReal_visitEndsAt hindep hlaw rho n

omit [MeasurableSpace Ω] in
/-- The visit-ends-at-`n` event is the event that the first releasing draw is the `n`-th, which
is what makes `clampRun` the dwell of the paper's sentence. -/
theorem setOf_visitEndsAt_eq (rho : ℝ) (zeta : ℕ → Ω → ℝ) (n : ℕ) :
    {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = {ω | (∃ k, zeta k ω < -rho) ∧ clampRun rho (fun k => zeta k ω) = n} := by
  ext ω
  simp only [mem_setOf_eq]
  constructor
  · intro h
    exact ⟨⟨n, h.2⟩, clampRun_eq_of_visitEndsAt h⟩
  · rintro ⟨hex, hrun⟩
    have := visitEndsAt_clampRun hex
    rwa [hrun] at this

/-- The geometric law, read on the first-passage index `clampRun` itself. -/
theorem measureReal_clampRun_eq {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P)
    (rho : ℝ) (n : ℕ) :
    P.real {ω | (∃ k, zeta k ω < -rho) ∧ clampRun rho (fun k => zeta k ω) = n}
      = (1 - clampExitProb rho) ^ n * clampExitProb rho := by
  rw [← setOf_visitEndsAt_eq]
  exact measureReal_visitEndsAt hindep hlaw rho n

/-- The visit ends almost surely: the geometric law is a probability distribution on the index
of the releasing draw. -/
theorem tsum_measureReal_visitEndsAt {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (rho : ℝ) :
    ∑' n : ℕ, P.real {ω | VisitEndsAt rho (fun k => zeta k ω) n} = 1 := by
  have hp0 := clampExitProb_pos rho
  have hp1 := clampExitProb_lt_one rho
  have hnn : (0 : ℝ) ≤ 1 - clampExitProb rho := by linarith
  have hlt : 1 - clampExitProb rho < 1 := by linarith
  calc ∑' n : ℕ, P.real {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = ∑' n : ℕ, (1 - clampExitProb rho) ^ n * clampExitProb rho := by
        exact tsum_congr fun n => measureReal_visitEndsAt hindep hlaw rho n
    _ = (∑' n : ℕ, (1 - clampExitProb rho) ^ n) * clampExitProb rho := by
        rw [tsum_mul_right]
    _ = 1 := by
        rw [tsum_geometric_of_lt_one hnn hlt, sub_sub_cancel, inv_mul_cancel₀ hp0.ne']

/-- **The mean visit length is `1/Φ(-ρ)`.** The visit that ends at index `n` lasts `n + 1`
steps — `n` clamped steps and the step that releases it — so the paper's "geometric number of
steps of mean `1/Φ(-ρ)`" is this sum. The dwell depends on `γ` and `σ` only through `ρ`, and
not at all on the step size `η`. -/
theorem tsum_visitLength {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (rho : ℝ) :
    ∑' n : ℕ, ((n : ℝ) + 1) * P.real {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = 1 / clampExitProb rho := by
  have hp0 := clampExitProb_pos rho
  have hp1 := clampExitProb_lt_one rho
  set p := clampExitProb rho with hpdef
  have hnn : (0 : ℝ) ≤ 1 - p := by linarith
  have hlt : 1 - p < 1 := by linarith
  have hnorm : ‖1 - p‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_nonneg hnn]
    exact hlt
  have hs1 : Summable (fun n : ℕ => (n : ℝ) * (1 - p) ^ n) := by
    simpa using summable_pow_mul_geometric_of_norm_lt_one (R := ℝ) 1 hnorm
  have hs2 : Summable (fun n : ℕ => (1 - p) ^ n) := summable_geometric_of_lt_one hnn hlt
  calc ∑' n : ℕ, ((n : ℝ) + 1) * P.real {ω | VisitEndsAt rho (fun k => zeta k ω) n}
      = ∑' n : ℕ, ((n : ℝ) * (1 - p) ^ n + (1 - p) ^ n) * p := by
        refine tsum_congr fun n => ?_
        rw [measureReal_visitEndsAt hindep hlaw rho n]
        ring
    _ = (∑' n : ℕ, ((n : ℝ) * (1 - p) ^ n + (1 - p) ^ n)) * p := by rw [tsum_mul_right]
    _ = ((∑' n : ℕ, (n : ℝ) * (1 - p) ^ n) + ∑' n : ℕ, (1 - p) ^ n) * p := by
        rw [hs1.tsum_add hs2]
    _ = ((1 - p) / (1 - (1 - p)) ^ 2 + (1 - (1 - p))⁻¹) * p := by
        rw [tsum_coe_mul_geometric_of_norm_lt_one hnorm, tsum_geometric_of_lt_one hnn hlt]
    _ = 1 / p := by
        rw [sub_sub_cancel]
        field_simp
        ring

/-- **The hypothesis stack of the geometric law is inhabited.** An i.i.d. standard-normal
sequence exists, so `measure_visitEndsAt` and `tsum_visitLength` are not vacuous. -/
theorem clamp_hypotheses_satisfiable :
    ∃ (Ω : Type) (_mΩ : MeasurableSpace Ω) (P : Measure Ω) (zeta : ℕ → Ω → ℝ),
      IsProbabilityMeasure P ∧ (∀ i, Measurable (zeta i)) ∧
        (∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) ∧ iIndepFun zeta P := by
  obtain ⟨Ω, mΩ, P, X, hmeas, hlaw, hindep, hP⟩ := exists_iid ℕ (gaussianReal (0 : ℝ) 1)
  exact ⟨Ω, mΩ, P, X, hP, hmeas, hlaw, hindep⟩

/-! ### The Lindley recursion and its homogeneity -/

/-- The **Lindley recursion at scale `c`**: `W₀ = 0`, `W_{k+1} = max(0, W_k - c(ρ + z_k))`. At
`c = η σ` this is the paper's `W_{k+1} = max(0, W_k - η(γ + σ ζ_k))` with `ρ = γ/σ`, which is
`clampIterate_succ_paper`. -/
noncomputable def clampIterate (c rho : ℝ) (z : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => max 0 (clampIterate c rho z k - c * (rho + z k))

@[simp] theorem clampIterate_zero (c rho : ℝ) (z : ℕ → ℝ) : clampIterate c rho z 0 = 0 := rfl

theorem clampIterate_succ (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    clampIterate c rho z (k + 1) = max 0 (clampIterate c rho z k - c * (rho + z k)) := rfl

/-- **The recursion in the paper's letters.** At scale `c = η σ` and ratio `ρ = γ/σ`, the step
of `clampIterate` is exactly the paper's `W_{k+1} = max(0, W_k - η(γ + σ ζ_k))`. -/
theorem clampIterate_succ_paper {eta gamma sigma : ℝ} (hsigma : sigma ≠ 0) (z : ℕ → ℝ) (k : ℕ) :
    clampIterate (eta * sigma) (gamma / sigma) z (k + 1)
      = max 0 (clampIterate (eta * sigma) (gamma / sigma) z k - eta * (gamma + sigma * z k)) := by
  rw [clampIterate_succ, eta_mul_add_eq eta gamma sigma (z k) hsigma]

theorem clampIterate_nonneg (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) : 0 ≤ clampIterate c rho z k := by
  cases k with
  | zero => exact le_rfl
  | succ k => exact le_max_left _ _

/-! ### The visit, read on the iterate and not only on the draws

`VisitEndsAt` is a predicate on the noise sequence. The paper's sentence is about `W`, so the
two lemmas below carry it across: through a visit the iterate really does stay at the boundary,
and at the step the visit names it really is released. What stays unproved is that the recursion
returns to the boundary at all, which belongs to the unformalized stationary-law clause. -/

/-- **The iterate stays at the boundary while the draws fail to release it.** From an iterate
sitting at `0`, every further step whose draw satisfies `-ρ ≤ z` leaves it at `0`. -/
theorem clampIterate_eq_zero_of_forall_le {c rho : ℝ} (hc : 0 ≤ c) {z : ℕ → ℝ} {k : ℕ}
    (hk : clampIterate c rho z k = 0) :
    ∀ j : ℕ, (∀ i < j, -rho ≤ z (k + i)) → clampIterate c rho z (k + j) = 0 := by
  intro j
  induction j with
  | zero => intro _; simpa using hk
  | succ j ih =>
    intro h
    have hprev : clampIterate c rho z (k + j) = 0 :=
      ih fun i hi => h i (hi.trans (Nat.lt_succ_self j))
    have hz : -rho ≤ z (k + j) := h j (Nat.lt_succ_self j)
    have hstep : clampIterate c rho z (k + j + 1)
        = max 0 (clampIterate c rho z (k + j) - c * (rho + z (k + j))) :=
      clampIterate_succ c rho z (k + j)
    have hnn : 0 ≤ c * (rho + z (k + j)) := mul_nonneg hc (by linarith)
    have : k + (j + 1) = k + j + 1 := by omega
    rw [this, hstep, hprev]
    exact max_eq_left (by linarith)

/-- **Through a visit, the iterate is clamped.** If the visit started at the boundary ends at
step `n`, then the iterate is `0` at every step up to and including `n`: the paper's "a visit to
the boundary", read on `W` and not only on the draws. -/
theorem clampIterate_eq_zero_of_visitEndsAt {c rho : ℝ} (hc : 0 ≤ c) {z : ℕ → ℝ} {n : ℕ}
    (h : VisitEndsAt rho z n) : ∀ j ≤ n, clampIterate c rho z j = 0 := by
  intro j hj
  have h0 : clampIterate c rho z 0 = 0 := rfl
  have := clampIterate_eq_zero_of_forall_le (k := 0) hc h0 j
    (fun i hi => by simpa using h.1 i (lt_of_lt_of_le hi hj))
  simpa using this

/-- **And at the end of the visit it is released.** The iterate leaves the boundary at step
`n + 1`, so the visit the geometric law counts is a visit of the recursion: `n` clamped steps
and the step that ends it. -/
theorem clampIterate_pos_of_visitEndsAt {c rho : ℝ} (hc : 0 < c) {z : ℕ → ℝ} {n : ℕ}
    (h : VisitEndsAt rho z n) : 0 < clampIterate c rho z (n + 1) := by
  have hn : clampIterate c rho z n = 0 := clampIterate_eq_zero_of_visitEndsAt hc.le h n le_rfl
  have hstep : clampIterate c rho z (n + 1)
      = max 0 (clampIterate c rho z n - c * (rho + z n)) := clampIterate_succ c rho z n
  have hneg : c * (rho + z n) < 0 := mul_neg_of_pos_of_neg hc (by linarith [h.2])
  rw [hstep, hn]
  exact lt_max_of_lt_right (by linarith)

/-- **The homogeneity of `prop:clamp`.** Every iterate of the recursion at scale `c ≥ 0` is `c`
times the corresponding iterate of the unit-scale recursion, pathwise and at every step. The
unit-scale recursion contains neither `η` nor `σ`: it is built from `ρ` and the draws alone. -/
theorem clampIterate_homogeneous {c : ℝ} (hc : 0 ≤ c) (rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    clampIterate c rho z k = c * clampIterate 1 rho z k := by
  induction k with
  | zero => simp
  | succ k ih =>
      have h := mul_max_of_nonneg (0 : ℝ) (clampIterate 1 rho z k - 1 * (rho + z k)) hc
      rw [clampIterate_succ, clampIterate_succ, ih, h, mul_zero]
      congr 1
      ring

/-- The supremum of the **iterates** scales likewise: `sup_k W_k^{(ησ)} = ησ · sup_k W_k^{(1)}`.

**This is not the paper's `M`, and the correction is worth stating.** The paper's proof writes
`M = ησ · sup_j (-∑_{i≤j}(ρ + ζ_i))`, a supremum of the *walk's partial sums*; the supremum here
is over the *recursion's values*. The two are different pathwise objects — `W_k` equals
`max_{j≤k}(S_k - S_j)`, not `max_{j≤k} S_j`, and they agree only in distribution at each fixed
time — and nothing here claims the iterates' supremum is finite. `iSup_walkSum_homogeneous`
below is the paper's object. -/
theorem iSup_clampIterate_homogeneous {c : ℝ} (hc : 0 ≤ c) (rho : ℝ) (z : ℕ → ℝ) :
    ⨆ k, clampIterate c rho z k = c * ⨆ k, clampIterate 1 rho z k := by
  rw [Real.mul_iSup_of_nonneg hc]
  exact iSup_congr fun k => clampIterate_homogeneous hc rho z k

/-- The partial sums of the paper's proof, `S_j = -c ∑_{i<j} (ρ + z_i)` at scale `c = ησ`. -/
noncomputable def walkSum (c rho : ℝ) (z : ℕ → ℝ) (j : ℕ) : ℝ :=
  -(c * ∑ i ∈ Finset.range j, (rho + z i))

theorem walkSum_homogeneous (c rho : ℝ) (z : ℕ → ℝ) (j : ℕ) :
    walkSum c rho z j = c * walkSum 1 rho z j := by
  simp only [walkSum, one_mul]
  ring

/-- **The paper's `M` scales by `ησ`.** `M = ησ · sup_j (-∑_{i≤j}(ρ + ζ_i))`, the identity the
offset clause of `prop:clamp` rests on, with the supremum on the right depending on `ρ` and the
draws alone. Unconditional, hence junk-valued (`0 = c · 0`) on a path whose supremum is
infinite; it has content exactly where `M` is finite, which is the negative-drift fact this
development does not prove. -/
theorem iSup_walkSum_homogeneous {c : ℝ} (hc : 0 ≤ c) (rho : ℝ) (z : ℕ → ℝ) :
    ⨆ j, walkSum c rho z j = c * ⨆ j, walkSum 1 rho z j := by
  rw [Real.mul_iSup_of_nonneg hc]
  exact iSup_congr fun j => walkSum_homogeneous c rho z j

/-- The expectation of every iterate scales by `c`. -/
theorem integral_clampIterate_homogeneous {P : Measure Ω} (zeta : ℕ → Ω → ℝ)
    {c : ℝ} (hc : 0 ≤ c) (rho : ℝ) (k : ℕ) :
    ∫ ω, clampIterate c rho (fun j => zeta j ω) k ∂P
      = c * ∫ ω, clampIterate 1 rho (fun j => zeta j ω) k ∂P := by
  rw [← integral_const_mul]
  exact integral_congr_ae (Eventually.of_forall fun ω => clampIterate_homogeneous hc rho _ k)

/-- **The offset clause, at the strength this development proves it.** If the unit-scale means
converge to `m` — the existence of the stationary mean, which the paper's proof takes from
Asmussen and which is *not* proved here — then the means of the paper's own recursion converge
to `η σ m`. Since `m` is read off a recursion in which neither `η` nor `σ` occurs, it is "a
function of `ρ` alone" in the paper's sense. -/
theorem tendsto_integral_clampIterate_homogeneous {P : Measure Ω} (zeta : ℕ → Ω → ℝ)
    {eta sigma gamma m : ℝ} (heta : 0 < eta) (hsigma : 0 < sigma)
    (hm : Tendsto (fun k => ∫ ω, clampIterate 1 (gamma / sigma) (fun j => zeta j ω) k ∂P)
      atTop (𝓝 m)) :
    Tendsto (fun k => ∫ ω, clampIterate (eta * sigma) (gamma / sigma) (fun j => zeta j ω) k ∂P)
      atTop (𝓝 (eta * sigma * m)) := by
  refine (hm.const_mul (eta * sigma)).congr fun k => ?_
  exact (integral_clampIterate_homogeneous zeta (by positivity) (gamma / sigma) k).symm

/-! ### The clamped fraction: what it depends on, and how it moves in `ρ`

Clause 2 of `prop:clamp` asserts four things about `P(W_∞ = 0)`: that it is a function of `ρ`
alone, that it increases in `ρ`, that it tends to `1` and to `0` at the two ends, and that it
equals the Spitzer series. The stationary law `W_∞` is not definable here, so the three
non-series clauses are proved instead **at every step of the recursion**, where they are
pathwise facts about the iterate and need no fluctuation theory at all:

* the clamped fraction at step `k` does not depend on the scale `ησ` — it is the same number for
  every step size and every noise scale, hence a function of `ρ` and the law of the draws alone
  (`clampIterate_eq_zero_iff_unit`, `measureReal_clampIterate_eq_zero_scale_free`);
* it is non-decreasing in `ρ`, by a pathwise coupling on one and the same draws
  (`clampIterate_antitone_in_rho`, `measureReal_clampIterate_eq_zero_mono_rho`);
* it tends to `1` as `ρ → ∞` (`tendsto_measureReal_clampIterate_eq_zero_atTop`).

What is **not** here, and why: the limit `k → ∞` (the stationary law itself), the limit
`ρ ↓ 0`, and the Spitzer series. See the module docstring. -/

/-- Being clamped is scale-free: at scale `c > 0` the iterate vanishes exactly when the
unit-scale iterate does, because the two differ by the factor `c`. -/
theorem clampIterate_eq_zero_iff_unit {c : ℝ} (hc : 0 < c) (rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    clampIterate c rho z k = 0 ↔ clampIterate 1 rho z k = 0 := by
  rw [clampIterate_homogeneous hc.le rho z k]
  constructor
  · intro h
    exact (mul_eq_zero.mp h).resolve_left hc.ne'
  · intro h
    rw [h, mul_zero]

/-- **The clamped fraction is a function of `ρ` alone.** The event that the iterate is clamped at
step `k` is literally the same event at every scale `c = ησ > 0`, so its probability carries no
dependence on the step size or the noise scale — only on `ρ` and on the law of the draws. This is
the first clause of `prop:clamp`'s clamped-fraction sentence, at every finite step. -/
theorem measureReal_clampIterate_eq_zero_scale_free {P : Measure Ω} {c : ℝ} (hc : 0 < c)
    (rho : ℝ) (zeta : ℕ → Ω → ℝ) (k : ℕ) :
    P.real {ω | clampIterate c rho (fun j => zeta j ω) k = 0}
      = P.real {ω | clampIterate 1 rho (fun j => zeta j ω) k = 0} := by
  congr 1
  ext ω
  exact clampIterate_eq_zero_iff_unit hc rho _ k

/-- **The coupling.** On one and the same draws, a larger drift-to-noise ratio gives a pointwise
smaller iterate, at every step. No probability enters: this is an induction on the recursion. -/
theorem clampIterate_antitone_in_rho {c : ℝ} (hc : 0 ≤ c) {rho rho' : ℝ} (h : rho ≤ rho')
    (z : ℕ → ℝ) (k : ℕ) : clampIterate c rho' z k ≤ clampIterate c rho z k := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [clampIterate_succ, clampIterate_succ]
    refine max_le_max le_rfl ?_
    have hstep : c * (rho + z k) ≤ c * (rho' + z k) :=
      mul_le_mul_of_nonneg_left (by linarith) hc
    linarith

/-- Hence a draw sequence that leaves the iterate clamped at ratio `ρ` leaves it clamped at every
larger ratio. -/
theorem clampIterate_eq_zero_of_le_rho {c : ℝ} (hc : 0 ≤ c) {rho rho' : ℝ} (h : rho ≤ rho')
    {z : ℕ → ℝ} {k : ℕ} (hz : clampIterate c rho z k = 0) : clampIterate c rho' z k = 0 := by
  have hle : clampIterate c rho' z k ≤ 0 := by
    have := clampIterate_antitone_in_rho hc h z k
    rw [hz] at this
    exact this
  exact le_antisymm hle (clampIterate_nonneg c rho' z k)

/-- **The clamped fraction is non-decreasing in `ρ`**, by the pathwise coupling: the clamped
event at ratio `ρ` is contained in the clamped event at any larger ratio, on the same draws. This
is the monotonicity clause of `prop:clamp`, at every finite step and with no fluctuation theory.
It is monotone, not strictly increasing; strictness is not claimed here. -/
theorem measureReal_clampIterate_eq_zero_mono_rho {P : Measure Ω} [IsFiniteMeasure P] {c : ℝ}
    (hc : 0 ≤ c) {rho rho' : ℝ} (h : rho ≤ rho') (zeta : ℕ → Ω → ℝ) (k : ℕ) :
    P.real {ω | clampIterate c rho (fun j => zeta j ω) k = 0}
      ≤ P.real {ω | clampIterate c rho' (fun j => zeta j ω) k = 0} :=
  measureReal_mono (fun _ hω => clampIterate_eq_zero_of_le_rho hc h hω) (measure_ne_top P _)

omit [MeasurableSpace Ω] in
/-- The event that the first `k` draws all fail to release the clamp, as an intersection of
single-draw events. -/
theorem setOf_forall_le_eq (rho : ℝ) (zeta : ℕ → Ω → ℝ) (k : ℕ) :
    {ω | ∀ i < k, -rho ≤ zeta i ω} = ⋂ i ∈ Finset.range k, zeta i ⁻¹' Ici (-rho) := by
  ext ω
  simp only [mem_setOf_eq, mem_iInter, Finset.mem_range, mem_preimage, mem_Ici]

/-- Its probability is `(1 - Φ(-ρ))^k`, by mutual independence. -/
theorem measure_forall_le {P : Measure Ω} {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P)
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (rho : ℝ) (k : ℕ) :
    P {ω | ∀ i < k, -rho ≤ zeta i ω} = ENNReal.ofReal ((1 - clampExitProb rho) ^ k) := by
  classical
  have hmeas : ∀ j ∈ Finset.range k,
      MeasurableSet[MeasurableSpace.comap (zeta j) inferInstance] (zeta j ⁻¹' Ici (-rho)) := by
    intro j _
    exact ⟨Ici (-rho), measurableSet_Ici, rfl⟩
  have hnn : (0 : ℝ) ≤ 1 - clampExitProb rho := by
    have := clampExitProb_lt_one rho
    linarith
  rw [setOf_forall_le_eq, hindep.meas_biInter hmeas]
  rw [Finset.prod_congr rfl (fun j _ => measure_preimage_Ici hlaw rho j), Finset.prod_const,
    Finset.card_range, ← ENNReal.ofReal_pow hnn]

/-- **A lower bound on the clamped fraction at every step**: the iterate is still at the boundary
whenever none of the first `k` draws released it. -/
theorem measureReal_clampIterate_eq_zero_ge {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P)
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c : ℝ} (hc : 0 ≤ c) (rho : ℝ) (k : ℕ) :
    (1 - clampExitProb rho) ^ k
      ≤ P.real {ω | clampIterate c rho (fun j => zeta j ω) k = 0} := by
  have hnn : (0 : ℝ) ≤ 1 - clampExitProb rho := by
    have := clampExitProb_lt_one rho
    linarith
  have hsub : {ω | ∀ i < k, -rho ≤ zeta i ω}
      ⊆ {ω | clampIterate c rho (fun j => zeta j ω) k = 0} := by
    intro ω hω
    have h0 : clampIterate c rho (fun j => zeta j ω) 0 = 0 := rfl
    have := clampIterate_eq_zero_of_forall_le (k := 0) hc h0 k (fun i hi => by simpa using hω i hi)
    simpa using this
  have hmono := measureReal_mono hsub (measure_ne_top P _)
  have hval : P.real {ω | ∀ i < k, -rho ≤ zeta i ω} = (1 - clampExitProb rho) ^ k := by
    rw [measureReal_def, measure_forall_le hindep hlaw rho k,
      ENNReal.toReal_ofReal (pow_nonneg hnn k)]
  linarith [hval ▸ hmono]

/-- **The clamped fraction tends to one as the drift-to-noise ratio grows**, at every step: the
release probability of a single draw is `Φ(-ρ) → 0`, so the bound above forces the fraction up to
`1`. This is the upper limit of `prop:clamp`'s clamped-fraction sentence, at every finite step. -/
theorem tendsto_measureReal_clampIterate_eq_zero_atTop {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P)
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c : ℝ} (hc : 0 ≤ c) (k : ℕ) :
    Tendsto (fun rho : ℝ => P.real {ω | clampIterate c rho (fun j => zeta j ω) k = 0})
      atTop (𝓝 1) := by
  have hp : Tendsto (fun rho : ℝ => clampExitProb rho) atTop (𝓝 0) := by
    have h := (ProbabilityTheory.tendsto_cdf_atBot (μ := gaussianReal 0 1)).comp
      tendsto_neg_atTop_atBot
    exact h
  have hlow : Tendsto (fun rho : ℝ => (1 - clampExitProb rho) ^ k) atTop (𝓝 1) := by
    have h1 : Tendsto (fun rho : ℝ => 1 - clampExitProb rho) atTop (𝓝 1) := by
      simpa using (tendsto_const_nhds (x := (1 : ℝ))).sub hp
    simpa using h1.pow k
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le hlow tendsto_const_nhds
    (fun rho => measureReal_clampIterate_eq_zero_ge hindep hlaw hc rho k) (fun rho => ?_)
  exact measureReal_le_one

/-! ### The reversal identity, first half: the pathwise form

The paper's proof writes "iterating the recursion from `W_0 = 0` gives `W_k = max_{0≤j≤k} S_j`
in distribution". The pathwise statement behind it is an identity, not a distributional one:

    W_k = max_{0≤j≤k} (S_k - S_j),

which is the recursion unrolled. The distributional statement follows from it by reversing the
block of draws, which is the second half. Nothing here is probabilistic. -/

/-- One step of the walk. -/
theorem walkSum_succ (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    walkSum c rho z (k + 1) = walkSum c rho z k - c * (rho + z k) := by
  simp only [walkSum, Finset.sum_range_succ]
  ring

/-- **The Lindley recursion, unrolled.** The iterate at step `k` is the largest rise of the walk
ending at step `k`: `W_k = max_{0≤j≤k} (S_k - S_j)`. Pathwise, for every sequence of draws. -/
theorem clampIterate_eq_sup'_sub (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    clampIterate c rho z k
      = (Finset.range (k + 1)).sup' Finset.nonempty_range_add_one
          (fun j => walkSum c rho z k - walkSum c rho z j) := by
  induction k with
  | zero => simp [walkSum]
  | succ k ih =>
    have hstep : walkSum c rho z (k + 1) = walkSum c rho z k - c * (rho + z k) :=
      walkSum_succ c rho z k
    have hshift : ((Finset.range (k + 1)).sup' Finset.nonempty_range_add_one
          (fun j => walkSum c rho z k - walkSum c rho z j)) - c * (rho + z k)
        = (Finset.range (k + 1)).sup' Finset.nonempty_range_add_one
          (fun j => walkSum c rho z (k + 1) - walkSum c rho z j) := by
      have hg : ∀ x y : ℝ, (x ⊔ y) + -(c * (rho + z k))
          = (x + -(c * (rho + z k))) ⊔ (y + -(c * (rho + z k))) :=
        fun x y => (max_add_add_right x y _).symm
      have h := Finset.apply_sup'_eq_sup'_comp (s := Finset.range (k + 1))
        Finset.nonempty_range_add_one
        (f := fun j => walkSum c rho z k - walkSum c rho z j)
        (fun x => x + -(c * (rho + z k))) hg
      have hrw : ((Finset.range (k + 1)).sup' Finset.nonempty_range_add_one
            (fun j => walkSum c rho z k - walkSum c rho z j)) - c * (rho + z k)
          = ((Finset.range (k + 1)).sup' Finset.nonempty_range_add_one
            (fun j => walkSum c rho z k - walkSum c rho z j)) + -(c * (rho + z k)) := by
        ring
      rw [hrw, h]
      refine Finset.sup'_congr _ rfl fun j _ => ?_
      simp only [Function.comp_apply]
      rw [hstep]
      ring
    rw [clampIterate_succ, ih, hshift]
    set g : ℕ → ℝ := fun j => walkSum c rho z (k + 1) - walkSum c rho z j with hgdef
    have hgtop : g (k + 1) = 0 := by rw [hgdef]; simp
    refine le_antisymm (max_le ?_ ?_) ?_
    · have h := Finset.le_sup' (s := Finset.range (k + 1 + 1)) g (Finset.self_mem_range_succ (k + 1))
      rw [hgtop] at h
      exact h
    · refine Finset.sup'_le _ _ fun j hj => ?_
      have hj' : j < k + 1 := Finset.mem_range.mp hj
      exact Finset.le_sup' (s := Finset.range (k + 1 + 1)) g
        (Finset.mem_range.mpr (by omega))
    · refine Finset.sup'_le _ _ fun j hj => ?_
      rcases Nat.lt_succ_iff_lt_or_eq.mp (Finset.mem_range.mp hj) with h | h
      · exact le_max_of_le_right
          (Finset.le_sup' (s := Finset.range (k + 1)) g (Finset.mem_range.mpr h))
      · rw [h, hgtop]
        exact le_max_left _ _

/-! ### The reversal identity, second half: reversing the block

The pathwise identity above says `W_k` is the largest rise of the walk ending at `k`. Reversing
the first `k` draws turns "rises ending at `k`" into "partial sums from the start", so `W_k` is
the running maximum of the walk driven by the reversed block (`clampIterate_eq_maxWalk_revDraws`,
still pathwise). The block of draws is i.i.d., so reversing it does not change its law, and the
two random variables therefore have the same law (`map_clampIterate_eq_map_maxWalk`). That is the
paper's "`W_k = max_{0≤j≤k} S_j` in distribution". -/

/-- The running maximum of the walk, `max_{0≤j≤k} S_j`, as a recursion — the form that makes
measurability an induction rather than a lemma about `Finset.sup'`. -/
noncomputable def maxWalk (c rho : ℝ) (z : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => max (maxWalk c rho z k) (walkSum c rho z (k + 1))

theorem maxWalk_eq_sup' (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    maxWalk c rho z k
      = (Finset.range (k + 1)).sup' Finset.nonempty_range_add_one (fun j => walkSum c rho z j) := by
  induction k with
  | zero => simp [maxWalk, walkSum]
  | succ k ih =>
    rw [maxWalk, ih]
    refine le_antisymm (max_le ?_ ?_) ?_
    · refine Finset.sup'_le _ _ fun j hj => ?_
      have hj' : j < k + 1 := Finset.mem_range.mp hj
      exact Finset.le_sup' (s := Finset.range (k + 1 + 1)) (fun i => walkSum c rho z i)
        (Finset.mem_range.mpr (by omega))
    · exact Finset.le_sup' (s := Finset.range (k + 1 + 1)) (fun i => walkSum c rho z i)
        (Finset.self_mem_range_succ (k + 1))
    · refine Finset.sup'_le _ _ fun j hj => ?_
      rcases Nat.lt_succ_iff_lt_or_eq.mp (Finset.mem_range.mp hj) with h | h
      · exact le_max_of_le_left
          (Finset.le_sup' (s := Finset.range (k + 1)) (fun i => walkSum c rho z i)
            (Finset.mem_range.mpr h))
      · rw [h]
        exact le_max_right _ _

/-- The first `k` draws, reversed. -/
def revDraws (k : ℕ) (z : ℕ → ℝ) : ℕ → ℝ := fun i => if i < k then z (k - 1 - i) else z i

/-- The index permutation that `revDraws` applies. -/
def revIndex (k : ℕ) : ℕ → ℕ := fun i => if i < k then k - 1 - i else i

theorem revDraws_eq_comp (k : ℕ) (z : ℕ → ℝ) : revDraws k z = fun i => z (revIndex k i) := by
  funext i
  rw [revDraws, revIndex]
  split_ifs <;> rfl

/-- `revIndex` is an involution, hence injective. -/
theorem revIndex_involutive (k : ℕ) : Function.Involutive (revIndex k) := by
  intro i
  rw [revIndex, revIndex]
  by_cases h : i < k
  · have h1 : k - 1 - i < k := by omega
    rw [if_pos h, if_pos h1]
    omega
  · rw [if_neg h, if_neg h]

theorem revIndex_injective (k : ℕ) : Function.Injective (revIndex k) :=
  (revIndex_involutive k).injective

/-- The walk driven by the reversed block, at a time `j ≤ k`, is the rise of the original walk
over the last `j` steps: `S_j(rev z) = S_k(z) - S_{k-j}(z)`. -/
theorem walkSum_revDraws (c rho : ℝ) (z : ℕ → ℝ) {k j : ℕ} (hjk : j ≤ k) :
    walkSum c rho (revDraws k z) j = walkSum c rho z k - walkSum c rho z (k - j) := by
  have hsum : ∀ j : ℕ, j ≤ k →
      (∑ i ∈ Finset.range j, (rho + revDraws k z i))
        = (∑ m ∈ Finset.range k, (rho + z m)) - ∑ m ∈ Finset.range (k - j), (rho + z m) := by
    intro j
    induction j with
    | zero => simp
    | succ j ih =>
      intro hjk'
      have hjk'' : j ≤ k := by omega
      have hlt : j < k := by omega
      rw [Finset.sum_range_succ, ih hjk'']
      have hrev : revDraws k z j = z (k - 1 - j) := by
        rw [revDraws, if_pos hlt]
      have hk1 : k - j = (k - (j + 1)) + 1 := by omega
      have hidx : k - (j + 1) = k - 1 - j := by omega
      rw [hrev, hk1, Finset.sum_range_succ, hidx]
      ring
  rw [walkSum, walkSum, walkSum, hsum j hjk]
  ring

/-- **The reversal identity, pathwise.** The iterate at step `k` is the running maximum of the
walk driven by the block of the first `k` draws, reversed. -/
theorem clampIterate_eq_maxWalk_revDraws (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    clampIterate c rho z k = maxWalk c rho (revDraws k z) k := by
  rw [clampIterate_eq_sup'_sub, maxWalk_eq_sup']
  refine le_antisymm (Finset.sup'_le _ _ fun j hj => ?_) (Finset.sup'_le _ _ fun j hj => ?_)
  · have hj' : j ≤ k := Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
    have hval : walkSum c rho (revDraws k z) (k - j)
        = walkSum c rho z k - walkSum c rho z j := by
      rw [walkSum_revDraws c rho z (by omega : k - j ≤ k)]
      congr 2
      omega
    rw [← hval]
    exact Finset.le_sup' (s := Finset.range (k + 1))
      (fun i => walkSum c rho (revDraws k z) i)
      (Finset.mem_range.mpr (by omega : k - j < k + 1))
  · have hj' : j ≤ k := Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
    rw [walkSum_revDraws c rho z hj']
    exact Finset.le_sup' (s := Finset.range (k + 1))
      (fun i => walkSum c rho z k - walkSum c rho z i)
      (Finset.mem_range.mpr (by omega : k - j < k + 1))

theorem measurable_walkSum_apply (c rho : ℝ) (j : ℕ) :
    Measurable (fun z : ℕ → ℝ => walkSum c rho z j) := by
  simp only [walkSum]
  fun_prop

theorem measurable_maxWalk_apply (c rho : ℝ) (k : ℕ) :
    Measurable (fun z : ℕ → ℝ => maxWalk c rho z k) := by
  induction k with
  | zero => simp [maxWalk]
  | succ k ih =>
    simp only [maxWalk]
    exact ih.max (measurable_walkSum_apply c rho (k + 1))

theorem measurable_clampIterate_apply (c rho : ℝ) (k : ℕ) :
    Measurable (fun z : ℕ → ℝ => clampIterate c rho z k) := by
  induction k with
  | zero => simp [clampIterate]
  | succ k ih =>
    simp only [clampIterate_succ]
    have h2 : Measurable (fun z : ℕ → ℝ => c * (rho + z k)) := by fun_prop
    exact measurable_const.max (ih.sub h2)

theorem measurable_revDraws (k : ℕ) : Measurable (revDraws k) := by
  have h : revDraws k = fun (z : ℕ → ℝ) i => z (revIndex k i) := by
    funext z
    exact revDraws_eq_comp k z
  rw [h]
  exact measurable_pi_lambda _ fun i => measurable_pi_apply (revIndex k i)

/-- **Reversing the first `k` draws does not change the law of the block.** The reversed family
is i.i.d. with the same marginals, so both blocks have the same infinite product law. This is
where independence and identical distribution are used, and it is all they are used for. -/
theorem map_revDraws_eq {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (k : ℕ) :
    (P.map fun ω i => zeta i ω).map (revDraws k) = P.map fun ω i => zeta i ω := by
  have hblock : Measurable (fun ω i => zeta i ω) := measurable_pi_lambda _ hmeas
  have hblock' : Measurable (fun ω i => zeta (revIndex k i) ω) :=
    measurable_pi_lambda _ fun i => hmeas (revIndex k i)
  have hpi : P.map (fun ω i => zeta i ω)
      = Measure.infinitePi (fun _ : ℕ => gaussianReal 0 1) := by
    rw [(iIndepFun_iff_map_fun_eq_infinitePi_map₀ hblock.aemeasurable).mp hindep]
    congr 1
    funext i
    exact (hlaw i).map_eq
  have hpi' : P.map (fun ω i => zeta (revIndex k i) ω)
      = Measure.infinitePi (fun _ : ℕ => gaussianReal 0 1) := by
    have hindep' : iIndepFun (fun i => zeta (revIndex k i)) P :=
      hindep.precomp (revIndex_injective k)
    rw [(iIndepFun_iff_map_fun_eq_infinitePi_map₀ hblock'.aemeasurable).mp hindep']
    congr 1
    funext i
    exact (hlaw (revIndex k i)).map_eq
  have hcomp : (revDraws k) ∘ (fun ω i => zeta i ω) = fun ω i => zeta (revIndex k i) ω := by
    funext ω
    exact revDraws_eq_comp k _
  calc (P.map fun ω i => zeta i ω).map (revDraws k)
      = P.map ((revDraws k) ∘ fun ω i => zeta i ω) :=
        Measure.map_map (measurable_revDraws k) hblock
    _ = P.map (fun ω i => zeta (revIndex k i) ω) := by rw [hcomp]
    _ = Measure.infinitePi (fun _ : ℕ => gaussianReal 0 1) := hpi'
    _ = P.map (fun ω i => zeta i ω) := hpi.symm

/-- **The reversal identity.** `W_k` and the running maximum `max_{0≤j≤k} S_j` have the same
law, for every `k`. This is the paper's "iterating the recursion from `W_0 = 0` gives
`W_k = max_{0≤j≤k} S_j` in distribution", and it is the step that carries the facts proved below
about the walk's supremum over to the recursion. -/
theorem map_clampIterate_eq_map_maxWalk {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (c rho : ℝ) (k : ℕ) :
    P.map (fun ω => clampIterate c rho (fun i => zeta i ω) k)
      = P.map (fun ω => maxWalk c rho (fun i => zeta i ω) k) := by
  have hblock : Measurable (fun ω i => zeta i ω) := measurable_pi_lambda _ hmeas
  have h1 : (fun ω => clampIterate c rho (fun i => zeta i ω) k)
      = ((fun z : ℕ → ℝ => maxWalk c rho z k) ∘ revDraws k) ∘ (fun ω i => zeta i ω) := by
    funext ω
    exact clampIterate_eq_maxWalk_revDraws c rho _ k
  have h2 : (fun ω => maxWalk c rho (fun i => zeta i ω) k)
      = (fun z : ℕ → ℝ => maxWalk c rho z k) ∘ (fun ω i => zeta i ω) := rfl
  rw [h1, h2]
  calc P.map (((fun z : ℕ → ℝ => maxWalk c rho z k) ∘ revDraws k) ∘ fun ω i => zeta i ω)
      = (P.map fun ω i => zeta i ω).map ((fun z : ℕ → ℝ => maxWalk c rho z k) ∘ revDraws k) :=
        (Measure.map_map ((measurable_maxWalk_apply c rho k).comp (measurable_revDraws k))
          hblock).symm
    _ = ((P.map fun ω i => zeta i ω).map (revDraws k)).map
          (fun z : ℕ → ℝ => maxWalk c rho z k) :=
        (Measure.map_map (measurable_maxWalk_apply c rho k) (measurable_revDraws k)).symm
    _ = (P.map fun ω i => zeta i ω).map (fun z : ℕ → ℝ => maxWalk c rho z k) := by
        rw [map_revDraws_eq hindep hmeas hlaw k]
    _ = P.map ((fun z : ℕ → ℝ => maxWalk c rho z k) ∘ fun ω i => zeta i ω) :=
        Measure.map_map (measurable_maxWalk_apply c rho k) hblock

/-! ### The walk's supremum is almost surely finite

The paper's proof takes this from Asmussen, through the strong law. It needs neither: the draws
are exactly Gaussian, hence sub-Gaussian with constant one, so the Chernoff bound gives
`P(S_j > 0) ≤ exp(-ρ²j/2)`, a geometric series, and Borel--Cantelli says that almost surely only
finitely many partial sums are positive. A sequence that is eventually non-positive is bounded
above by the maximum of its finitely many early terms and `0`.

Note where the scale `c = ησ` goes: the event `S_j > 0` is `∑_{i<j}(-ζ_i) > ρ j`, free of `c`
entirely, which is why the bound depends on `ρ` alone. -/

/-- The walk written as a sum of independent terms plus the drift constant. -/
theorem walkSum_eq_sum_add (c rho : ℝ) (z : ℕ → ℝ) (j : ℕ) :
    walkSum c rho z j = (∑ i ∈ Finset.range j, -(c * z i)) + -(c * rho * j) := by
  have h1 : ∑ i ∈ Finset.range j, (rho + z i) = j * rho + ∑ i ∈ Finset.range j, z i := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have h2 : (∑ i ∈ Finset.range j, -(c * z i)) = -(c * ∑ i ∈ Finset.range j, z i) := by
    rw [Finset.sum_neg_distrib, Finset.mul_sum]
  rw [walkSum, h1, h2]
  ring

/-- The positive part of a real is at most `t⁻¹ exp(t ·)` for every `t > 0`: the tilt that turns
a mean bound into a geometric one. -/
theorem le_inv_mul_exp {t x : ℝ} (ht : 0 < t) : x ≤ t⁻¹ * Real.exp (t * x) := by
  have h := Real.add_one_le_exp (t * x)
  have hle : t * x ≤ Real.exp (t * x) := by linarith
  calc x = t⁻¹ * (t * x) := by field_simp
    _ ≤ t⁻¹ * Real.exp (t * x) := mul_le_mul_of_nonneg_left hle (by positivity)

/-- A draw with the standard normal law is sub-Gaussian with constant one: its moment generating
function is exactly `exp(t²/2)`. -/
theorem hasSubgaussianMGF_of_hasLaw {P : Measure Ω} {X : Ω → ℝ} (hmeas : Measurable X)
    (hlaw : HasLaw X (gaussianReal 0 1) P) : HasSubgaussianMGF X 1 P where
  integrable_exp_mul t := by
    have h := integrable_exp_mul_gaussianReal (μ := (0 : ℝ)) (v := 1) t
    rw [← hlaw.map_eq] at h
    exact (integrable_map_measure (by fun_prop) hmeas.aemeasurable).mp h
  mgf_le t := by
    rw [mgf_gaussianReal hlaw.map_eq]
    norm_num

/-- The reversed partial sums `∑_{i<j} (-ζ_i)` are sub-Gaussian with constant `j`. -/
theorem hasSubgaussianMGF_negSum {P : Measure Ω} {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P)
    (hmeas : ∀ i, Measurable (zeta i)) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P)
    (j : ℕ) :
    HasSubgaussianMGF (fun ω => ∑ i ∈ Finset.range j, -zeta i ω) (j : ℝ≥0) P := by
  have hneg : iIndepFun (fun i ω => -zeta i ω) P :=
    hindep.comp (fun _ x => -x) (fun _ => measurable_neg)
  have h := HasSubgaussianMGF.sum_of_iIndepFun (X := fun i ω => -zeta i ω) (c := fun _ => 1)
    hneg (s := Finset.range j) (fun i _ => ((hasSubgaussianMGF_of_hasLaw (hmeas i) (hlaw i)).neg))
  simpa using h

/-- **The Chernoff bound on a positive partial sum.** `P(S_j > 0) ≤ exp(-ρ²j/2)`: the event is
free of the scale, and the exponent is linear in `j`. -/
theorem measureReal_walk_pos_le {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {rho : ℝ} (hrho : 0 < rho) (j : ℕ) :
    P.real {ω | rho * j ≤ ∑ i ∈ Finset.range j, -zeta i ω}
      ≤ Real.exp (-(rho ^ 2 / 2) * j) := by
  have hsub := hasSubgaussianMGF_negSum hindep hmeas hlaw j
  have hε : (0 : ℝ) ≤ rho * j := by positivity
  have h := hsub.measure_ge_le hε
  refine h.trans (le_of_eq ?_)
  rcases Nat.eq_zero_or_pos j with rfl | hj
  · norm_num
  · have hjne : (j : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hj.ne'
    congr 1
    push_cast
    field_simp

/-- **The walk is almost surely bounded above.** Only finitely many partial sums are positive,
by Borel--Cantelli on the geometric bound, so the supremum is a maximum over finitely many terms
and `0`. This is the paper's "`M = sup_j S_j` is finite almost surely", proved rather than
cited, and with no strong law and no fluctuation theory. -/
theorem ae_bddAbove_walkSum {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    ∀ᵐ ω ∂P, BddAbove (Set.range (fun j => walkSum c rho (fun i => zeta i ω) j)) := by
  classical
  set A : ℕ → Set Ω := fun j => {ω | rho * j ≤ ∑ i ∈ Finset.range j, -zeta i ω} with hA
  have hAmeas : ∀ j, MeasurableSet (A j) := by
    intro j
    have : Measurable (fun ω => ∑ i ∈ Finset.range j, -zeta i ω) :=
      Finset.measurable_sum _ fun i _ => (hmeas i).neg
    exact measurableSet_le measurable_const this
  -- the geometric bound, in `ℝ≥0∞`
  have hbound : ∀ j, P (A j) ≤ ENNReal.ofReal (Real.exp (-(rho ^ 2 / 2)) ^ j) := by
    intro j
    have h := measureReal_walk_pos_le hindep hmeas hlaw hrho j
    have hmeasreal : P (A j) = ENNReal.ofReal (P.real (A j)) := by
      rw [measureReal_def, ENNReal.ofReal_toReal (measure_ne_top P _)]
    rw [hmeasreal]
    refine ENNReal.ofReal_le_ofReal ?_
    calc P.real (A j) ≤ Real.exp (-(rho ^ 2 / 2) * j) := h
      _ = Real.exp (-(rho ^ 2 / 2)) ^ j := by rw [← Real.exp_nat_mul]; ring_nf
  -- hence the series of probabilities converges
  have hsum : ∑' j, P (A j) ≠ ⊤ := by
    have hr : |Real.exp (-(rho ^ 2 / 2))| < 1 := by
      rw [abs_of_pos (Real.exp_pos _), Real.exp_lt_one_iff]
      have : 0 < rho ^ 2 / 2 := by positivity
      linarith
    have hsummable : Summable (fun j : ℕ => Real.exp (-(rho ^ 2 / 2)) ^ j) :=
      summable_geometric_of_abs_lt_one hr
    have hle : ∑' j, P (A j) ≤ ∑' j, ENNReal.ofReal (Real.exp (-(rho ^ 2 / 2)) ^ j) :=
      ENNReal.tsum_le_tsum hbound
    refine ne_top_of_le_ne_top ?_ hle
    rw [← ENNReal.ofReal_tsum_of_nonneg (fun j => (pow_nonneg (Real.exp_pos _).le j)) hsummable]
    exact ENNReal.ofReal_ne_top
  -- Borel--Cantelli: almost surely, only finitely many partial sums are positive
  have hlimsup := MeasureTheory.measure_limsup_atTop_eq_zero hsum
  have hae : ∀ᵐ ω ∂P, ω ∉ Filter.limsup A Filter.atTop := by
    rw [ae_iff]
    simpa using hlimsup
  filter_upwards [hae] with ω hω
  have hev : ∀ᶠ j in Filter.atTop, ω ∉ A j := by
    rw [Filter.mem_limsup_iff_frequently_mem, Filter.not_frequently] at hω
    exact hω
  obtain ⟨N, hN⟩ := Filter.eventually_atTop.mp hev
  have hle : ∀ j, N ≤ j → walkSum c rho (fun i => zeta i ω) j ≤ 0 := by
    intro j hj
    have hnot : ¬ rho * j ≤ ∑ i ∈ Finset.range j, -zeta i ω := hN j hj
    rw [not_le] at hnot
    have hexp : walkSum c rho (fun i => zeta i ω) j
        = c * ((∑ i ∈ Finset.range j, -zeta i ω) - rho * j) := by
      simp only [walkSum, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
        nsmul_eq_mul, Finset.sum_neg_distrib]
      ring
    rw [hexp]
    have : (∑ i ∈ Finset.range j, -zeta i ω) - rho * j ≤ 0 := by linarith
    exact mul_nonpos_of_nonneg_of_nonpos hc.le this
  have hsubset : Set.range (fun j => walkSum c rho (fun i => zeta i ω) j)
      ⊆ (fun j => walkSum c rho (fun i => zeta i ω) j) '' (Set.Iio N) ∪ Set.Iic 0 := by
    rintro _ ⟨j, rfl⟩
    rcases lt_or_ge j N with h | h
    · exact Or.inl ⟨j, h, rfl⟩
    · exact Or.inr (hle j h)
  exact BddAbove.mono hsubset
    (((Set.finite_Iio N).image _).bddAbove.union bddAbove_Iic)

/-! ### The expected supremum is finite, and scales by `ησ`

The paper takes the finite mean from Asmussen too. It is the same geometric estimate: the
supremum is at most the sum of the positive parts, the positive part of a real is at most
`t⁻¹exp(t·)` (`le_inv_mul_exp`), and at the tilt `t = ρ` the sub-Gaussian bound on the reversed
partial sums gives `E[(S_j)^+] ≤ (c/ρ)exp(-ρ²j/2)`, a convergent geometric series.

Everything is in `ℝ≥0∞`, through `∫⁻`. That is deliberate: the supremum of a countable family is
then measurable with no hypothesis at all, the bound is a genuine statement about the expected
supremum, and the scaling is an identity of `∫⁻`s rather than of integrals whose integrability
would have to be carried around. -/

/-- The expected positive part of the `j`-th partial sum is at most `(c/ρ)exp(-ρ²j/2)`. -/
theorem lintegral_ofReal_walkSum_le {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho)
    (j : ℕ) :
    ∫⁻ ω, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P
      ≤ ENNReal.ofReal (c / rho * Real.exp (-(rho ^ 2 / 2)) ^ j) := by
  classical
  have hsub := hasSubgaussianMGF_negSum hindep hmeas hlaw j
  have hexp : ∀ ω, walkSum c rho (fun i => zeta i ω) j
      = c * ((∑ i ∈ Finset.range j, -zeta i ω) - rho * j) := by
    intro ω
    have h2 : (∑ i ∈ Finset.range j, -zeta i ω) = -(∑ i ∈ Finset.range j, zeta i ω) :=
      Finset.sum_neg_distrib _
    rw [walkSum_eq_sum_add c rho _ j, h2]
    have h3 : (∑ i ∈ Finset.range j, -(c * zeta i ω)) = -(c * ∑ i ∈ Finset.range j, zeta i ω) := by
      rw [Finset.sum_neg_distrib, Finset.mul_sum]
    rw [h3]
    ring
  -- the tilted dominating function
  set K : ℝ := c * rho⁻¹ * Real.exp (-(rho * (rho * j))) with hK
  have hKpos : 0 < K := by rw [hK]; positivity
  have hgfun : ∀ ω, c * ((∑ i ∈ Finset.range j, -zeta i ω) - rho * j)
      ≤ K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω) := by
    intro ω
    have h := le_inv_mul_exp (t := rho) (x := (∑ i ∈ Finset.range j, -zeta i ω) - rho * j) hrho
    have hsplit : Real.exp (rho * ((∑ i ∈ Finset.range j, -zeta i ω) - rho * j))
        = Real.exp (-(rho * (rho * j))) * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω) := by
      rw [← Real.exp_add]
      congr 1
      ring
    calc c * ((∑ i ∈ Finset.range j, -zeta i ω) - rho * j)
        ≤ c * (rho⁻¹ * Real.exp (rho * ((∑ i ∈ Finset.range j, -zeta i ω) - rho * j))) :=
          mul_le_mul_of_nonneg_left h hc.le
      _ = K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω) := by
          rw [hsplit, hK]; ring
  have hgint : Integrable
      (fun ω => K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω)) P :=
    (hsub.integrable_exp_mul rho).const_mul _
  have hgnn : 0 ≤ᵐ[P] fun ω => K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω) := by
    filter_upwards with ω
    positivity
  have hval : ∫ ω, K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω) ∂P
      = K * mgf (fun ω => ∑ i ∈ Finset.range j, -zeta i ω) P rho := by
    rw [integral_const_mul]
    rfl
  have hKbound : K * mgf (fun ω => ∑ i ∈ Finset.range j, -zeta i ω) P rho
      ≤ c / rho * Real.exp (-(rho ^ 2 / 2)) ^ j := by
    have hmle : mgf (fun ω => ∑ i ∈ Finset.range j, -zeta i ω) P rho
        ≤ Real.exp ((j : ℝ) * rho ^ 2 / 2) := by
      have h := hsub.mgf_le rho
      refine h.trans (le_of_eq ?_)
      congr 1
    have hstep : K * mgf (fun ω => ∑ i ∈ Finset.range j, -zeta i ω) P rho
        ≤ K * Real.exp ((j : ℝ) * rho ^ 2 / 2) := mul_le_mul_of_nonneg_left hmle hKpos.le
    refine hstep.trans (le_of_eq ?_)
    rw [hK, mul_assoc, ← Real.exp_add, ← Real.exp_nat_mul, div_eq_mul_inv]
    congr 2
    ring
  calc ∫⁻ ω, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P
      ≤ ∫⁻ ω, ENNReal.ofReal (K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω)) ∂P := by
        refine lintegral_mono fun ω => ENNReal.ofReal_le_ofReal ?_
        rw [hexp ω]
        exact hgfun ω
    _ = ENNReal.ofReal (∫ ω, K * Real.exp (rho * ∑ i ∈ Finset.range j, -zeta i ω) ∂P) :=
        (ofReal_integral_eq_lintegral_ofReal hgint hgnn).symm
    _ ≤ ENNReal.ofReal (c / rho * Real.exp (-(rho ^ 2 / 2)) ^ j) := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [hval]
        exact hKbound

/-- **The expected supremum of the walk is finite.** The supremum is at most the sum of the
positive parts, and those sum geometrically. This is the paper's "`M` … has a finite mean, the
walk having negative drift and finite variance", with the Asmussen citation replaced by a
Chernoff bound. -/
theorem lintegral_iSup_ofReal_walkSum_lt_top {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    ∫⁻ ω, ⨆ j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P < ⊤ := by
  classical
  have hmeasw : ∀ j, Measurable (fun ω => walkSum c rho (fun i => zeta i ω) j) := by
    intro j
    have hfun : (fun ω => walkSum c rho (fun i => zeta i ω) j)
        = fun ω => (∑ i ∈ Finset.range j, -(c * zeta i ω)) + -(c * rho * j) := by
      funext ω
      exact walkSum_eq_sum_add c rho _ j
    rw [hfun]
    exact (Finset.measurable_sum _ fun i _ => ((hmeas i).const_mul c).neg).add_const _
  have hpt : ∀ ω, (⨆ j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j))
      ≤ ∑' j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) :=
    fun ω => iSup_le fun j => ENNReal.le_tsum j
  have hsum : ∫⁻ ω, ∑' j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P
      = ∑' j, ∫⁻ ω, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P :=
    lintegral_tsum fun j => ((hmeasw j).ennreal_ofReal).aemeasurable
  have hgeo : Summable (fun j : ℕ => c / rho * Real.exp (-(rho ^ 2 / 2)) ^ j) := by
    have hr : |Real.exp (-(rho ^ 2 / 2))| < 1 := by
      rw [abs_of_pos (Real.exp_pos _), Real.exp_lt_one_iff]
      have : 0 < rho ^ 2 / 2 := by positivity
      linarith
    exact (summable_geometric_of_abs_lt_one hr).mul_left _
  have hfin : ∑' j, ENNReal.ofReal (c / rho * Real.exp (-(rho ^ 2 / 2)) ^ j) ≠ ⊤ := by
    rw [← ENNReal.ofReal_tsum_of_nonneg (fun j => by positivity) hgeo]
    exact ENNReal.ofReal_ne_top
  calc ∫⁻ ω, ⨆ j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P
      ≤ ∫⁻ ω, ∑' j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P :=
        lintegral_mono hpt
    _ = ∑' j, ∫⁻ ω, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P := hsum
    _ ≤ ∑' j, ENNReal.ofReal (c / rho * Real.exp (-(rho ^ 2 / 2)) ^ j) :=
        ENNReal.tsum_le_tsum fun j => lintegral_ofReal_walkSum_le hindep hmeas hlaw hc hrho j
    _ < ⊤ := lt_top_iff_ne_top.mpr hfin

/-- **The expected supremum scales by `ησ`.** Pathwise homogeneity carried through the integral,
with the right-hand factor finite by the theorem above and containing neither `η` nor `σ`. This
is `E[W_∞] = ησ·m(ρ)` at the level this development can state it: `m(ρ)` is the expected
supremum of the unit-scale walk, a function of `ρ` and the law of the draws alone. -/
theorem lintegral_iSup_ofReal_walkSum_homogeneous {P : Measure Ω} {zeta : ℕ → Ω → ℝ} {c : ℝ}
    (hc : 0 ≤ c) (rho : ℝ) :
    ∫⁻ ω, ⨆ j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j) ∂P
      = ENNReal.ofReal c
        * ∫⁻ ω, ⨆ j, ENNReal.ofReal (walkSum 1 rho (fun i => zeta i ω) j) ∂P := by
  have hpt : ∀ ω, (⨆ j, ENNReal.ofReal (walkSum c rho (fun i => zeta i ω) j))
      = ENNReal.ofReal c * ⨆ j, ENNReal.ofReal (walkSum 1 rho (fun i => zeta i ω) j) := by
    intro ω
    rw [ENNReal.mul_iSup]
    refine iSup_congr fun j => ?_
    rw [walkSum_homogeneous c rho _ j, ENNReal.ofReal_mul hc]
  simp_rw [hpt]
  exact lintegral_const_mul' _ _ ENNReal.ofReal_ne_top

/-! ### Passing to the limit: `W_k → M` in distribution

The last step of the paper's sentence. The running maximum is monotone in `k` and its supremum
is the walk's supremum, which is almost surely finite, so `max_{0≤j≤k} S_j → M` pointwise almost
surely; the reversal identity turns that into a statement about the laws of the iterates.

**Which formulation.** Convergence in distribution is stated here in its defining form — the
integrals against bounded continuous test functions converge — rather than through a weak
topology on measures. That is the version the paper's use needs (it is a statement about the
laws, not about any coupling), it is what dominated convergence delivers directly, and it makes
the hypotheses on the test function visible at the use site. -/

theorem maxWalk_mono (c rho : ℝ) (z : ℕ → ℝ) : Monotone (maxWalk c rho z) := by
  refine monotone_nat_of_le_succ fun k => ?_
  rw [maxWalk]
  exact le_max_left _ _

theorem walkSum_le_maxWalk (c rho : ℝ) (z : ℕ → ℝ) {j k : ℕ} (hjk : j ≤ k) :
    walkSum c rho z j ≤ maxWalk c rho z k := by
  rw [maxWalk_eq_sup']
  exact Finset.le_sup' (s := Finset.range (k + 1)) (fun i => walkSum c rho z i)
    (Finset.mem_range.mpr (by omega))

theorem maxWalk_le_of_walkSum_le {c rho : ℝ} {z : ℕ → ℝ} {b : ℝ}
    (hb : ∀ j, walkSum c rho z j ≤ b) (k : ℕ) : maxWalk c rho z k ≤ b := by
  rw [maxWalk_eq_sup']
  exact Finset.sup'_le _ _ fun j _ => hb j

/-- Where the walk is bounded above, the running maximum increases to the walk's supremum. -/
theorem tendsto_maxWalk_atTop {c rho : ℝ} {z : ℕ → ℝ}
    (hbdd : BddAbove (Set.range (fun j => walkSum c rho z j))) :
    Filter.Tendsto (maxWalk c rho z) Filter.atTop (𝓝 (⨆ j, walkSum c rho z j)) := by
  obtain ⟨b, hb⟩ := id hbdd
  have hble : ∀ j, walkSum c rho z j ≤ b := fun j => hb ⟨j, rfl⟩
  have hbdd' : BddAbove (Set.range (maxWalk c rho z)) :=
    ⟨b, by rintro _ ⟨k, rfl⟩; exact maxWalk_le_of_walkSum_le hble k⟩
  have hlim := tendsto_atTop_ciSup (maxWalk_mono c rho z) hbdd'
  have hsup : (⨆ k, maxWalk c rho z k) = ⨆ j, walkSum c rho z j := by
    refine le_antisymm (ciSup_le fun k => ?_) (ciSup_le fun j => ?_)
    · rw [maxWalk_eq_sup']
      exact Finset.sup'_le _ _ fun j _ => le_ciSup hbdd j
    · exact le_ciSup_of_le hbdd' j (walkSum_le_maxWalk c rho z le_rfl)
  rwa [hsup] at hlim

/-- **`W_k → M` in distribution.** For every bounded continuous test function, the expectation
against the law of the iterate at step `k` converges to the expectation against the law of the
walk's supremum. With the reversal identity this is the paper's "`W_k → M` in distribution", and
it is the last step of the stationary-law sentence that this development can take: what remains
is the Spitzer series and the limit at `ρ ↓ 0` it supplies. -/
theorem tendsto_integral_clampIterate {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho)
    {f : ℝ → ℝ} (hfc : Continuous f) {C : ℝ} (hfb : ∀ x, |f x| ≤ C) :
    Filter.Tendsto (fun k => ∫ ω, f (clampIterate c rho (fun i => zeta i ω) k) ∂P)
      Filter.atTop (𝓝 (∫ ω, f (⨆ j, walkSum c rho (fun i => zeta i ω) j) ∂P)) := by
  classical
  have hmeasM : ∀ k, Measurable (fun ω => maxWalk c rho (fun i => zeta i ω) k) := by
    intro k
    exact (measurable_maxWalk_apply c rho k).comp (measurable_pi_lambda _ hmeas)
  have hmeasW : ∀ k, Measurable (fun ω => clampIterate c rho (fun i => zeta i ω) k) := by
    intro k
    exact (measurable_clampIterate_apply c rho k).comp (measurable_pi_lambda _ hmeas)
  -- almost surely, the running maximum increases to the supremum
  have hae : ∀ᵐ ω ∂P, Filter.Tendsto (fun k => maxWalk c rho (fun i => zeta i ω) k)
      Filter.atTop (𝓝 (⨆ j, walkSum c rho (fun i => zeta i ω) j)) := by
    filter_upwards [ae_bddAbove_walkSum hindep hmeas hlaw hc hrho] with ω hω
    exact tendsto_maxWalk_atTop hω
  -- hence the supremum is almost everywhere measurable, and dominated convergence applies
  have hsupmeas : AEMeasurable (fun ω => ⨆ j, walkSum c rho (fun i => zeta i ω) j) P :=
    aemeasurable_of_tendsto_metrizable_ae' (fun k => (hmeasM k).aemeasurable) hae
  have hdom : Filter.Tendsto (fun k => ∫ ω, f (maxWalk c rho (fun i => zeta i ω) k) ∂P)
      Filter.atTop (𝓝 (∫ ω, f (⨆ j, walkSum c rho (fun i => zeta i ω) j) ∂P)) := by
    refine tendsto_integral_of_dominated_convergence (fun _ => C)
      (fun k => (hfc.measurable.comp (hmeasM k)).aestronglyMeasurable)
      (integrable_const C) (fun k => ?_) ?_
    · filter_upwards with ω
      simpa [Real.norm_eq_abs] using hfb _
    · filter_upwards [hae] with ω hω
      exact (hfc.tendsto _).comp hω
  -- and the two laws agree at every step
  refine hdom.congr fun k => ?_
  have hfae : AEStronglyMeasurable f (P.map (fun ω => maxWalk c rho (fun i => zeta i ω) k)) :=
    hfc.aestronglyMeasurable
  have h1 : ∫ ω, f (clampIterate c rho (fun i => zeta i ω) k) ∂P
      = ∫ x, f x ∂(P.map fun ω => clampIterate c rho (fun i => zeta i ω) k) := by
    rw [integral_map (hmeasW k).aemeasurable]
    rw [map_clampIterate_eq_map_maxWalk hindep hmeas hlaw c rho k]
    exact hfae
  have h2 : ∫ ω, f (maxWalk c rho (fun i => zeta i ω) k) ∂P
      = ∫ x, f x ∂(P.map fun ω => maxWalk c rho (fun i => zeta i ω) k) :=
    (integral_map (hmeasM k).aemeasurable hfae).symm
  rw [h2, ← map_clampIterate_eq_map_maxWalk hindep hmeas hlaw c rho k, ← h1]

/-! ### The offset clause's hypothesis is inhabited

`tendsto_integral_clampIterate_homogeneous` carries the convergence of the unit-scale means as a
hypothesis, so it owes a witness, like every other hypothesis stack in this development. The
witness below is deliberately the informative one rather than the trivial one: the limit is
`m = 1`, not `0`, so the instantiated conclusion is an offset of exactly `η σ` — the paper's
"the stationary iterate sits above the constrained optimum at a distance proportional to the
step size", with a nonzero constant, checked by the kernel in the statement.

**The witness is degenerate in its randomness, and it has to be.** It is a single deterministic
path, carried on a Dirac probability space: one draw that releases the iterate, then draws that
hold it where it lands. A witness whose draws are genuinely i.i.d. standard normal *would be*
the convergence theorem the paper takes from Asmussen, which is exactly what is not formalized
here; exhibiting one would close the gap rather than witness it. What this theorem establishes
is therefore the only thing a witness can establish: the hypothesis is not empty, so the
homogeneity theorem is not vacuously true. -/

/-- The witness path for the offset clause: one draw below `-ρ`, which releases the iterate, and
then draws at exactly `-ρ`, which hold it. -/
noncomputable def offsetWitnessPath (rho : ℝ) : ℕ → ℝ :=
  fun j => if j = 0 then -rho - 1 else -rho

theorem offsetWitnessPath_zero (rho : ℝ) : offsetWitnessPath rho 0 = -rho - 1 := if_pos rfl

theorem offsetWitnessPath_succ (rho : ℝ) (k : ℕ) : offsetWitnessPath rho (k + 1) = -rho :=
  if_neg (Nat.succ_ne_zero k)

/-- On the witness path the unit-scale iterate is `1` from the first step onward. -/
theorem clampIterate_offsetWitnessPath_succ (rho : ℝ) (k : ℕ) :
    clampIterate 1 rho (offsetWitnessPath rho) (k + 1) = 1 := by
  induction k with
  | zero =>
    have h : (0 : ℝ) - 1 * (rho + offsetWitnessPath rho 0) = 1 := by
      rw [offsetWitnessPath_zero]; ring
    rw [clampIterate_succ, clampIterate_zero, h]
    norm_num
  | succ k ih =>
    have h : (1 : ℝ) - 1 * (rho + offsetWitnessPath rho (k + 1)) = 1 := by
      rw [offsetWitnessPath_succ]; ring
    rw [clampIterate_succ, ih, h]
    norm_num

/-- The means of the witness path, at unit scale: constantly `1` from the first step onward. -/
theorem integral_clampIterate_offsetWitnessPath (rho : ℝ) (k : ℕ) :
    ∫ _ω : ℝ, clampIterate 1 rho (offsetWitnessPath rho) (k + 1) ∂(Measure.dirac (0 : ℝ))
      = 1 := by
  rw [integral_const, probReal_univ, one_smul, clampIterate_offsetWitnessPath_succ]

/-- **The hypothesis stack of the offset clause is inhabited**, with a nonzero limit. A
probability space, a sequence of draws and a limit `m = 1` exist for which the unit-scale means
converge; the instantiated conclusion of `tendsto_integral_clampIterate_homogeneous` is then
carried in the statement, so the kernel checks the whole chain rather than only the hypotheses:
the means of the paper's own recursion converge to `η σ`. See the section docstring above for
what this witness does and does not show. -/
theorem homogeneity_hypotheses_satisfiable (eta sigma gamma : ℝ) (heta : 0 < eta)
    (hsigma : 0 < sigma) :
    ∃ (Ω : Type) (_mΩ : MeasurableSpace Ω) (P : Measure Ω) (zeta : ℕ → Ω → ℝ),
      IsProbabilityMeasure P ∧ (∀ i, Measurable (zeta i)) ∧
        Tendsto (fun k => ∫ ω, clampIterate 1 (gamma / sigma) (fun j => zeta j ω) k ∂P)
          atTop (𝓝 1) ∧
        Tendsto (fun k => ∫ ω, clampIterate (eta * sigma) (gamma / sigma)
          (fun j => zeta j ω) k ∂P) atTop (𝓝 (eta * sigma)) := by
  have hconv : Tendsto (fun k => ∫ _ω : ℝ,
      clampIterate 1 (gamma / sigma) (offsetWitnessPath (gamma / sigma)) k
        ∂(Measure.dirac (0 : ℝ))) atTop (𝓝 1) := by
    refine tendsto_const_nhds.congr' ?_
    filter_upwards [eventually_ge_atTop 1] with k hk
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    exact (integral_clampIterate_offsetWitnessPath (gamma / sigma) k').symm
  have h := tendsto_integral_clampIterate_homogeneous (P := Measure.dirac (0 : ℝ))
    (fun j (_ : ℝ) => offsetWitnessPath (gamma / sigma) j) heta hsigma hconv
  rw [mul_one] at h
  exact ⟨ℝ, inferInstance, Measure.dirac 0, fun j _ => offsetWitnessPath (gamma / sigma) j,
    inferInstance, fun _ => measurable_const, hconv, h⟩

end IcnnLift
