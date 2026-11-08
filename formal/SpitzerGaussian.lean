import Mathlib
import ProjectionClamp

/-!
# The Gaussian side of the Spitzer series in `prop:clamp`

Proposition `prop:clamp` of the paper (the Lindley recursion at the boundary)
states that for independent standard normal draws `ζ₀, ζ₁, …`, drift ratio `ρ > 0` and scale
`c = ησ > 0`, the clamped fraction at the stationary law is

    P(W_∞ = 0) = exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k)),

and that it is non-decreasing in `ρ` and tends to `0` as `ρ ↓ 0`. Its proof has two halves.
The general half is Spitzer's identity `P(S_n ≤ 0 for all n ≥ 1) = exp(-∑_{n≥1} n⁻¹ P(S_n > 0))`
for a walk with independent, identically distributed increments, together with the
identification of the stationary law with the law of `M = sup_j S_j`, so that
`P(W_∞ = 0) = P(M = 0)`. That half is not in this file. **This file is the Gaussian half:** it
computes `P(S_n > 0)` for the paper's walk, and it studies the resulting series as a function
of `ρ`.

## The probability of a positive partial sum

`walkSum c ρ ζ n = -c ∑_{i<n} (ρ + ζ_i)` is the paper's `S_n`. For `c > 0` it is positive
exactly when `∑_{i<n} ζ_i < -ρn` (`setOf_walkSum_pos_eq`), an event free of the scale `c`. The
sum of `n` independent standard normals is `N(0, n)` (`hasLaw_sum_range_gaussian`, by induction
on `n` with the Gaussian convolution `gaussianReal_conv_gaussianReal`, the partial sum being
independent of the next draw), and `N(0, n)` is the image of `N(0, 1)` under `x ↦ √n x`
(`gaussianReal_natCast_eq_map_sqrt`). Hence, for every real `ρ` and every `n ≥ 1`,

    P(S_n > 0) = Φ(-ρn/√n) = Φ(-ρ√n)            (`measureReal_walkSum_pos_eq`).

The event is a strict inequality and `Φ` is the distribution function, a non-strict one; the
two agree because the Gaussian has no atoms (`clampExitProb_eq_measureReal_Iio` of
`ProjectionClamp`). At `n = 0` the walk is `0`, the event is empty, and the identity fails,
which is why `n ≥ 1` is a hypothesis.

## The series

`spitzerSeries ρ = ∑_{k≥0} Φ(-ρ√(k+1)) / (k+1)` is the paper's `∑_{k≥1} k⁻¹ Φ(-ρ√k)` with the
index shifted by one.

* **Summable for `ρ > 0`** (`summable_spitzerTerm`): the Gaussian tail `Φ(-x) ≤ exp(-x²/2)` for
  `x ≥ 0` (`clampExitProb_le_exp`, the Chernoff bound for a variable that is sub-Gaussian with
  constant one) bounds the `k`-th term by `exp(-ρ²(k+1)/2)`, a geometric sequence.
* **Strictly decreasing on `(0, ∞)`** (`spitzerSeries_antitoneOn`,
  `spitzerSeries_strictAntiOn`): each term is non-increasing in `ρ` because `Φ` is
  non-decreasing, and the first term is strictly decreasing because `Φ` is strictly increasing,
  the Gaussian charging every interval (`clampExitProb_strictAnti`).
* **Tends to `0` as `ρ → ∞`** (`tendsto_spitzerSeries_atTop`): dominated convergence for series
  (Tannery's theorem), each term tending to `0` and, for `ρ ≥ 1`, each term bounded by
  `exp(-(k+1)/2)`.
* **Tends to `+∞` as `ρ ↓ 0`** (`tendsto_spitzerSeries_nhdsGT`): for `0 < ρ < 1/(√K + 1)` the
  first `K` terms are each at least `Φ(-1)/(k+1)`, so the series is at least `Φ(-1)` times the
  `K`-th harmonic number, and the harmonic numbers diverge
  (`Real.tendsto_sum_range_one_div_nat_succ_atTop`).

**At `ρ ≤ 0` the series diverges and the value is a convention.** There every term is at least
`Φ(0)/(k+1)`, so the series is not summable (`not_summable_spitzerTerm_of_nonpos`), and Lean's
`tsum` of a non-summable family is `0` (`spitzerSeries_of_nonpos`). The values
`spitzerSeries ρ = 0` and `clampedFraction ρ = 1` at `ρ ≤ 0` are therefore junk values, not the
paper's statement, and every monotonicity statement below is restricted to `Ioi 0`.

## The clamped fraction

`clampedFraction ρ = exp(-spitzerSeries ρ)` is the right-hand side of the paper's display. It is
strictly increasing on `(0, ∞)` (`clampedFraction_strictMonoOn`, and its non-strict form
`clampedFraction_monotoneOn`), tends to `0` as `ρ → 0⁺` (`tendsto_clampedFraction_nhdsGT`) and to
`1` as `ρ → ∞` (`tendsto_clampedFraction_atTop`), and lies in `(0, 1]` for every `ρ`
(`clampedFraction_mem_Ioc`), strictly below `1` when `ρ > 0` (`clampedFraction_lt_one`).

## The bridge to Spitzer's identity

The increments of the paper's proof are `X_i = -c(ρ + ζ_i)`, written below as the explicit
family `fun i ω => -(c * (rho + zeta i ω))`. Their partial sums are `walkSum`
(`sum_walkIncrements_eq_walkSum`); they are mutually independent
(`iIndepFun_walkIncrements`), measurable (`measurable_walkIncrements`) and identically
distributed (`identDistrib_walkIncrements`); and for `c > 0` and `ρ > 0` the family
`n ↦ P(X_0 + ⋯ + X_n > 0)/(n+1)` is summable (`summable_measureReal_walkIncrements_pos`) with
sum `spitzerSeries ρ` (`tsum_measureReal_walkIncrements_pos`). So
`clampedFraction ρ = exp(-∑_{n≥1} n⁻¹ P(S_n > 0))` for the paper's walk
(`clampedFraction_eq_exp_neg_tsum`), which is the right-hand side of Spitzer's identity.

## Honesty

No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. What is not proved here:

* **Spitzer's identity**, and the identification of the stationary law of the recursion with the
  law of `M`. This file does not state that `clampedFraction ρ` is the clamped fraction of any
  recursion: it is the closed form, studied as a function of `ρ`, and the bridge lemmas are the
  hooks by which the general identity, applied to the increments above, turns it into
  `P(W_∞ = 0)`.
* **The values at `ρ ≤ 0`** carry no content, as recorded above; nothing is claimed there beyond
  the divergence of the series.
-/

open MeasureTheory Filter Set ProbabilityTheory
open scoped Topology ENNReal NNReal

namespace IcnnLift

variable {Ω : Type*} [MeasurableSpace Ω]

/-! ### The law of a Gaussian partial sum -/

/-- The sum of `n` independent standard normal draws is `N(0, n)`. -/
theorem hasLaw_sum_range_gaussian {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (n : ℕ) :
    HasLaw (fun ω => ∑ i ∈ Finset.range n, zeta i ω) (gaussianReal 0 (n : ℝ≥0)) P := by
  induction n with
  | zero =>
    simp only [Finset.range_zero, Finset.sum_empty, Nat.cast_zero, gaussianReal_zero_var]
    exact hasLaw_dirac_of_ae_eq (ae_of_all _ fun _ => rfl)
  | succ n ih =>
    have hind : IndepFun (∑ i ∈ Finset.range n, zeta i) (zeta n) P :=
      hindep.indepFun_finsetSum_of_notMem hmeas (by simp)
    have hih : HasLaw (∑ i ∈ Finset.range n, zeta i) (gaussianReal 0 (n : ℝ≥0)) P := by
      convert ih using 1
      ext ω
      simp [Finset.sum_apply]
    have h := hind.hasLaw_fun_add hih (hlaw n)
    rw [gaussianReal_conv_gaussianReal, add_zero] at h
    convert h using 2 with ω
    · rw [Finset.sum_range_succ, Finset.sum_apply]
    · push_cast
      ring

/-- `N(0, n)` is the image of `N(0, 1)` under `x ↦ √n x`. -/
theorem gaussianReal_natCast_eq_map_sqrt (n : ℕ) :
    gaussianReal 0 (n : ℝ≥0) = (gaussianReal 0 1).map (fun x => Real.sqrt n * x) := by
  rw [gaussianReal_map_const_mul, mul_zero, mul_one]
  congr 1
  ext
  simp [Real.sq_sqrt (Nat.cast_nonneg n)]

/-- Under `N(0, n)` with `n ≥ 1`, the half-line below `-ρn` has mass `Φ(-ρ√n)`. -/
theorem measureReal_gaussianReal_natCast_Iio {n : ℕ} (hn : 1 ≤ n) (rho : ℝ) :
    (gaussianReal 0 (n : ℝ≥0)).real (Iio (-(rho * n))) = clampExitProb (rho * Real.sqrt n) := by
  have hs : 0 < Real.sqrt n := Real.sqrt_pos.mpr (by exact_mod_cast hn)
  have hsq : Real.sqrt n * Real.sqrt n = n := Real.mul_self_sqrt (Nat.cast_nonneg n)
  rw [gaussianReal_natCast_eq_map_sqrt, map_measureReal_apply (by fun_prop) measurableSet_Iio,
    clampExitProb_eq_measureReal_Iio]
  congr 1
  ext x
  simp only [mem_preimage, mem_Iio]
  rw [show -(rho * (n : ℝ)) = Real.sqrt n * (-(rho * Real.sqrt n)) by
    linear_combination rho * hsq]
  constructor
  · intro h
    exact lt_of_mul_lt_mul_left h hs.le
  · intro h
    exact mul_lt_mul_of_pos_left h hs

omit [MeasurableSpace Ω] in
/-- For `c > 0` the walk is positive at step `n` exactly when `∑_{i<n} ζ_i < -ρn`: the event is
free of the scale. -/
theorem setOf_walkSum_pos_eq {c : ℝ} (hc : 0 < c) (rho : ℝ) (zeta : ℕ → Ω → ℝ) (n : ℕ) :
    {ω | 0 < walkSum c rho (fun i => zeta i ω) n}
      = {ω | ∑ i ∈ Finset.range n, zeta i ω < -(rho * n)} := by
  ext ω
  simp only [mem_setOf_eq, walkSum, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
    nsmul_eq_mul]
  constructor
  · intro h
    by_contra h'
    rw [not_lt] at h'
    have : 0 ≤ (n : ℝ) * rho + ∑ i ∈ Finset.range n, zeta i ω := by linarith
    nlinarith [mul_nonneg hc.le this]
  · intro h
    have : (n : ℝ) * rho + ∑ i ∈ Finset.range n, zeta i ω < 0 := by linarith
    linarith [mul_neg_of_pos_of_neg hc this]

/-- **`P(S_n > 0) = Φ(-ρ√n)`.** For independent standard normal draws, any scale `c > 0`, any
real `ρ` and any `n ≥ 1`, the probability that the paper's walk
`S_n = -c ∑_{i<n} (ρ + ζ_i)` is positive is `Φ(-ρ√n)`. -/
theorem measureReal_walkSum_pos_eq {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c : ℝ} (hc : 0 < c) (rho : ℝ)
    {n : ℕ} (hn : 1 ≤ n) :
    P.real {ω | 0 < walkSum c rho (fun i => zeta i ω) n} = clampExitProb (rho * Real.sqrt n) := by
  rw [setOf_walkSum_pos_eq hc, (hasLaw_sum_range_gaussian hindep hmeas hlaw n).measureReal_eq
    (p := fun x => x < -(rho * n)) measurableSet_Iio]
  exact measureReal_gaussianReal_natCast_Iio hn rho

/-! ### The Gaussian tail, and the monotonicity of `Φ` -/

/-- **The Gaussian tail.** `Φ(-x) ≤ exp(-x²/2)` for `x ≥ 0`: the Chernoff bound for a variable
that is sub-Gaussian with constant one. -/
theorem clampExitProb_le_exp {x : ℝ} (hx : 0 ≤ x) :
    clampExitProb x ≤ Real.exp (-(x ^ 2 / 2)) := by
  have hsub : HasSubgaussianMGF (fun y : ℝ => -y) 1 (gaussianReal 0 1) :=
    (hasSubgaussianMGF_of_hasLaw (P := gaussianReal 0 1) measurable_id HasLaw.id).neg
  have h := hsub.measure_ge_le hx
  rw [clampExitProb_eq_measureReal_Iic]
  convert h using 2
  · ext y
    simp [le_neg]
  · simp [neg_div]

/-- `ρ ↦ Φ(-ρ)` is non-increasing, `Φ` being a distribution function. -/
theorem clampExitProb_antitone : Antitone clampExitProb :=
  fun _ _ hab => monotone_cdf _ (neg_le_neg hab)

/-- `ρ ↦ Φ(-ρ)` is strictly decreasing: the Gaussian charges every interval, because Lebesgue
measure is absolutely continuous with respect to it. -/
theorem clampExitProb_strictAnti : StrictAnti clampExitProb := by
  intro a b hab
  have hlt : -b < -a := neg_lt_neg hab
  have hI : gaussianReal (0 : ℝ) 1 (Ioc (-b) (-a)) ≠ 0 := by
    intro h
    have hac := gaussianReal_absolutelyContinuous' (0 : ℝ) (v := 1) one_ne_zero
    have hv : (volume : Measure ℝ) (Ioc (-b) (-a)) = 0 := hac h
    rw [Real.volume_Ioc, ENNReal.ofReal_eq_zero] at hv
    linarith
  have hm := (cdf (gaussianReal 0 1)).measure_Ioc (-b) (-a)
  rw [measure_cdf] at hm
  rw [hm, ne_eq, ENNReal.ofReal_eq_zero, not_le, sub_pos] at hI
  exact hI

/-! ### The Spitzer series -/

/-- **The Spitzer series** `∑_{k≥1} k⁻¹ Φ(-ρ√k)`, indexed from `0` with `k + 1` in place of `k`.
It is summable exactly when `ρ > 0`; at `ρ ≤ 0` the `tsum` is `0` by convention. -/
noncomputable def spitzerSeries (rho : ℝ) : ℝ :=
  ∑' k : ℕ, clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1)

theorem spitzerTerm_nonneg (rho : ℝ) (k : ℕ) :
    0 ≤ clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1) :=
  div_nonneg (clampExitProb_pos _).le (by positivity)

theorem spitzerTerm_pos (rho : ℝ) (k : ℕ) :
    0 < clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1) :=
  div_pos (clampExitProb_pos _) (by positivity)

/-- The `k`-th term is at most `exp(-ρ²(k+1)/2)` for `ρ ≥ 0`. -/
theorem spitzerTerm_le_exp {rho : ℝ} (hrho : 0 ≤ rho) (k : ℕ) :
    clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1)
      ≤ Real.exp (-(rho ^ 2 / 2) * ((k : ℝ) + 1)) := by
  have hk : (1 : ℝ) ≤ k + 1 := by
    have := (Nat.cast_nonneg k : (0 : ℝ) ≤ k)
    linarith
  calc clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1)
      ≤ clampExitProb (rho * Real.sqrt (k + 1)) := div_le_self (clampExitProb_pos _).le hk
    _ ≤ Real.exp (-((rho * Real.sqrt (k + 1)) ^ 2 / 2)) := clampExitProb_le_exp (by positivity)
    _ = Real.exp (-(rho ^ 2 / 2) * ((k : ℝ) + 1)) := by
        congr 1
        rw [mul_pow, Real.sq_sqrt (by positivity)]
        ring

/-- **The series converges for `ρ > 0`**, its terms being dominated by a geometric sequence. -/
theorem summable_spitzerTerm {rho : ℝ} (hrho : 0 < rho) :
    Summable fun k : ℕ => clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1) := by
  have hneg : -(rho ^ 2 / 2) < 0 := by
    have := pow_pos hrho 2
    linarith
  have hb : Summable fun k : ℕ => Real.exp (-(rho ^ 2 / 2) * ((k : ℝ) + 1)) :=
    Real.summable_exp_nat_mul_of_ge (f := fun k : ℕ => (k : ℝ) + 1) hneg (fun i => by simp)
  exact hb.of_nonneg_of_le (spitzerTerm_nonneg rho) (spitzerTerm_le_exp hrho.le)

/-- At `ρ ≤ 0` the series diverges: every term is at least `Φ(0)/(k+1)`. -/
theorem not_summable_spitzerTerm_of_nonpos {rho : ℝ} (hrho : rho ≤ 0) :
    ¬ Summable fun k : ℕ => clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1) := by
  intro hs
  have hle : ∀ k : ℕ, clampExitProb 0 * (1 / ((k : ℝ) + 1))
      ≤ clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1) := by
    intro k
    rw [mul_one_div]
    gcongr
    exact clampExitProb_antitone (mul_nonpos_of_nonpos_of_nonneg hrho (Real.sqrt_nonneg _))
  have h1 : Summable fun k : ℕ => clampExitProb 0 * (1 / ((k : ℝ) + 1)) :=
    hs.of_nonneg_of_le (fun k => mul_nonneg (clampExitProb_pos 0).le (by positivity)) hle
  have h2 : Summable fun k : ℕ => 1 / ((k : ℝ) + 1) :=
    (summable_mul_left_iff (clampExitProb_pos 0).ne').mp h1
  have h3 : Summable fun k : ℕ => 1 / ((k + 1 : ℕ) : ℝ) := by
    simpa using h2
  exact Real.not_summable_one_div_natCast ((summable_nat_add_iff 1).mp h3)

/-- At `ρ ≤ 0` the `tsum` takes its junk value `0`. -/
theorem spitzerSeries_of_nonpos {rho : ℝ} (hrho : rho ≤ 0) : spitzerSeries rho = 0 :=
  tsum_eq_zero_of_not_summable (not_summable_spitzerTerm_of_nonpos hrho)

theorem spitzerSeries_nonneg (rho : ℝ) : 0 ≤ spitzerSeries rho :=
  tsum_nonneg (spitzerTerm_nonneg rho)

theorem spitzerSeries_pos {rho : ℝ} (hrho : 0 < rho) : 0 < spitzerSeries rho :=
  (summable_spitzerTerm hrho).tsum_pos (spitzerTerm_nonneg rho) 0 (spitzerTerm_pos rho 0)

/-- The series is non-increasing in `ρ` on `(0, ∞)`. -/
theorem spitzerSeries_antitoneOn : AntitoneOn spitzerSeries (Ioi 0) := by
  intro a ha b hb hab
  refine (summable_spitzerTerm hb).tsum_le_tsum (fun k => ?_) (summable_spitzerTerm ha)
  gcongr
  exact clampExitProb_antitone (mul_le_mul_of_nonneg_right hab (Real.sqrt_nonneg _))

/-- **The series is strictly decreasing in `ρ` on `(0, ∞)`.** -/
theorem spitzerSeries_strictAntiOn : StrictAntiOn spitzerSeries (Ioi 0) := by
  intro a ha b hb hab
  have hs1 : 0 < Real.sqrt (((0 : ℕ) : ℝ) + 1) := Real.sqrt_pos.mpr (by positivity)
  refine (summable_spitzerTerm hb).tsum_lt_tsum (i := 0) (fun k => ?_) ?_
    (summable_spitzerTerm ha)
  · gcongr
    exact clampExitProb_antitone (mul_le_mul_of_nonneg_right hab.le (Real.sqrt_nonneg _))
  · gcongr
    exact clampExitProb_strictAnti (mul_lt_mul_of_pos_right hab hs1)

/-- **The series tends to `0` as `ρ → ∞`**, by dominated convergence for series. -/
theorem tendsto_spitzerSeries_atTop : Tendsto spitzerSeries atTop (𝓝 0) := by
  have hb : Summable fun k : ℕ => Real.exp (-(1 / 2) * ((k : ℝ) + 1)) :=
    Real.summable_exp_nat_mul_of_ge (f := fun k : ℕ => (k : ℝ) + 1) (by norm_num)
      (fun i => by simp)
  rw [show (0 : ℝ) = ∑' _ : ℕ, (0 : ℝ) by simp]
  refine tendsto_tsum_of_dominated_convergence (𝓕 := atTop)
    (f := fun rho (k : ℕ) => clampExitProb (rho * Real.sqrt (k + 1)) / (k + 1)) hb
    (fun k => ?_) ?_
  · have hs : 0 < Real.sqrt ((k : ℝ) + 1) := Real.sqrt_pos.mpr (by positivity)
    have h1 : Tendsto (fun rho : ℝ => -(rho * Real.sqrt (k + 1))) atTop atBot :=
      tendsto_neg_atTop_atBot.comp (tendsto_id.atTop_mul_const hs)
    have h2 := ((tendsto_cdf_atBot (gaussianReal 0 1)).comp h1).div_const ((k : ℝ) + 1)
    simpa [clampExitProb, Function.comp_def] using h2
  · filter_upwards [eventually_ge_atTop 1] with rho hrho k
    rw [Real.norm_of_nonneg (spitzerTerm_nonneg _ _)]
    refine (spitzerTerm_le_exp (by linarith) k).trans (Real.exp_le_exp.mpr ?_)
    have hk : (0 : ℝ) ≤ k + 1 := by positivity
    have : (1 : ℝ) ≤ rho ^ 2 := by nlinarith
    nlinarith

/-- **The series tends to `+∞` as `ρ ↓ 0`.** For `0 < ρ < 1/(√K + 1)` the first `K` terms are
each at least `Φ(-1)/(k+1)`, and the harmonic numbers diverge. -/
theorem tendsto_spitzerSeries_nhdsGT : Tendsto spitzerSeries (𝓝[>] 0) atTop := by
  rw [tendsto_atTop]
  intro b
  obtain ⟨K, hK⟩ := ((Real.tendsto_sum_range_one_div_nat_succ_atTop.const_mul_atTop
    (clampExitProb_pos 1)).eventually_ge_atTop b).exists
  have hε : (0 : ℝ) < 1 / (Real.sqrt K + 1) := by positivity
  filter_upwards [Ioo_mem_nhdsGT hε] with rho hrho
  obtain ⟨h0, h1⟩ := hrho
  have hρK : rho * Real.sqrt K ≤ 1 := by
    have hpos : 0 < Real.sqrt K + 1 := by positivity
    rw [lt_div_iff₀ hpos] at h1
    nlinarith
  calc b ≤ clampExitProb 1 * ∑ i ∈ Finset.range K, 1 / ((i : ℝ) + 1) := hK
    _ = ∑ i ∈ Finset.range K, clampExitProb 1 / ((i : ℝ) + 1) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun i _ => ?_
        ring
    _ ≤ ∑ i ∈ Finset.range K, clampExitProb (rho * Real.sqrt (i + 1)) / ((i : ℝ) + 1) := by
        refine Finset.sum_le_sum fun i hi => ?_
        gcongr
        apply clampExitProb_antitone
        have hiK : (i : ℝ) + 1 ≤ K := by
          have := Finset.mem_range.mp hi
          exact_mod_cast this
        calc rho * Real.sqrt (i + 1) ≤ rho * Real.sqrt K := by gcongr
          _ ≤ 1 := hρK
    _ ≤ spitzerSeries rho :=
        (summable_spitzerTerm h0).sum_le_tsum _ (fun i _ => spitzerTerm_nonneg _ _)

/-! ### The clamped fraction -/

/-- **The clamped fraction in closed form**, `exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))`. Meaningful for
`ρ > 0`; at `ρ ≤ 0` it takes the junk value `1`. -/
noncomputable def clampedFraction (rho : ℝ) : ℝ := Real.exp (-spitzerSeries rho)

/-- The clamped fraction is non-decreasing in `ρ` on `(0, ∞)`. -/
theorem clampedFraction_monotoneOn : MonotoneOn clampedFraction (Ioi 0) := by
  intro a ha b hb hab
  exact Real.exp_le_exp.mpr (neg_le_neg (spitzerSeries_antitoneOn ha hb hab))

/-- **The clamped fraction is strictly increasing in `ρ` on `(0, ∞)`.** -/
theorem clampedFraction_strictMonoOn : StrictMonoOn clampedFraction (Ioi 0) := by
  intro a ha b hb hab
  exact Real.exp_lt_exp.mpr (neg_lt_neg (spitzerSeries_strictAntiOn ha hb hab))

/-- **The clamped fraction tends to `0` as `ρ ↓ 0`.** -/
theorem tendsto_clampedFraction_nhdsGT : Tendsto clampedFraction (𝓝[>] 0) (𝓝 0) :=
  Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp tendsto_spitzerSeries_nhdsGT)

/-- **The clamped fraction tends to `1` as `ρ → ∞`.** -/
theorem tendsto_clampedFraction_atTop : Tendsto clampedFraction atTop (𝓝 1) := by
  have hneg := tendsto_spitzerSeries_atTop.neg
  rw [neg_zero] at hneg
  have := (Real.continuous_exp.tendsto 0).comp hneg
  rw [Real.exp_zero] at this
  exact this

/-- The clamped fraction lies in `(0, 1]`. -/
theorem clampedFraction_mem_Ioc (rho : ℝ) : clampedFraction rho ∈ Ioc 0 1 :=
  ⟨Real.exp_pos _, Real.exp_le_one_iff.mpr (neg_nonpos.mpr (spitzerSeries_nonneg rho))⟩

/-- For `ρ > 0` the clamped fraction is strictly below `1`. -/
theorem clampedFraction_lt_one {rho : ℝ} (hrho : 0 < rho) : clampedFraction rho < 1 :=
  Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr (spitzerSeries_pos hrho))

/-! ### The bridge to Spitzer's identity: the increments `X_i = -c(ρ + ζ_i)` -/

omit [MeasurableSpace Ω] in
/-- The partial sums of the increments `X_i = -c(ρ + ζ_i)` are the paper's walk. -/
theorem sum_walkIncrements_eq_walkSum (c rho : ℝ) (zeta : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) :
    ∑ i ∈ Finset.range n, -(c * (rho + zeta i ω)) = walkSum c rho (fun i => zeta i ω) n := by
  rw [walkSum, Finset.mul_sum, Finset.sum_neg_distrib]

/-- The increments are mutually independent. -/
theorem iIndepFun_walkIncrements {P : Measure Ω} {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P)
    (c rho : ℝ) : iIndepFun (fun i ω => -(c * (rho + zeta i ω))) P :=
  hindep.comp (fun _ x => -(c * (rho + x))) (fun _ => by fun_prop)

/-- The increments are measurable. -/
theorem measurable_walkIncrements {zeta : ℕ → Ω → ℝ} (hmeas : ∀ i, Measurable (zeta i))
    (c rho : ℝ) (i : ℕ) : Measurable fun ω => -(c * (rho + zeta i ω)) := by
  have := hmeas i
  fun_prop

/-- The increments are identically distributed. -/
theorem identDistrib_walkIncrements {P : Measure Ω} {zeta : ℕ → Ω → ℝ}
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (c rho : ℝ) (i : ℕ) :
    IdentDistrib (fun ω => -(c * (rho + zeta i ω))) (fun ω => -(c * (rho + zeta 0 ω))) P P := by
  have h : IdentDistrib (zeta i) (zeta 0) P P :=
    ⟨(hlaw i).aemeasurable, (hlaw 0).aemeasurable, (hlaw i).map_eq.trans (hlaw 0).map_eq.symm⟩
  exact h.comp (u := fun x => -(c * (rho + x))) (by fun_prop)

/-- The `n`-th term of Spitzer's series for the increments is the `n`-th term of
`spitzerSeries`. -/
theorem measureReal_walkIncrements_pos_div {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c : ℝ} (hc : 0 < c) (rho : ℝ) (n : ℕ) :
    P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), -(c * (rho + zeta i ω))} / (n + 1)
      = clampExitProb (rho * Real.sqrt (n + 1)) / (n + 1) := by
  simp_rw [sum_walkIncrements_eq_walkSum]
  rw [measureReal_walkSum_pos_eq hindep hmeas hlaw hc rho (Nat.le_add_left 1 n)]
  push_cast
  rfl

/-- **Spitzer's series for the paper's walk converges for `ρ > 0`.** -/
theorem summable_measureReal_walkIncrements_pos {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c)
    (hrho : 0 < rho) :
    Summable fun n : ℕ =>
      P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), -(c * (rho + zeta i ω))} / (n + 1) := by
  simp_rw [measureReal_walkIncrements_pos_div hindep hmeas hlaw hc rho]
  exact summable_spitzerTerm hrho

/-- **Spitzer's series for the paper's walk is `spitzerSeries ρ`.** -/
theorem tsum_measureReal_walkIncrements_pos {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c : ℝ} (hc : 0 < c) (rho : ℝ) :
    ∑' n : ℕ, P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), -(c * (rho + zeta i ω))} / (n + 1)
      = spitzerSeries rho := by
  simp_rw [measureReal_walkIncrements_pos_div hindep hmeas hlaw hc rho]
  rfl

/-- **The closed form is the right-hand side of Spitzer's identity for the paper's walk.** -/
theorem clampedFraction_eq_exp_neg_tsum {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c : ℝ} (hc : 0 < c) (rho : ℝ) :
    clampedFraction rho = Real.exp (-∑' n : ℕ,
      P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), -(c * (rho + zeta i ω))} / (n + 1)) := by
  rw [tsum_measureReal_walkIncrements_pos hindep hmeas hlaw hc rho]
  rfl

end IcnnLift
