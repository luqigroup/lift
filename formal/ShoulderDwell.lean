import Mathlib
import KramersExitTime
import KramersBridge

/-!
# The shoulder dwell: the occupation-time corollary `cor:dwell`

Corollary `cor:dwell` of the paper (stated in Section 3.3 of `paper_v5.tex`, proved in
`app_proofs.tex` at the end of `app:proof-fpt`) reads, under the hypotheses of
`thm:main`(iii):

> let `τ` be the first passage of `w` to `b` and let `D(x) = E_x[∫₀^τ 1_{[a,w_s]}(w_t) dt]` be
> the mean time the diffusion spends below the barrier top before escape. Then
> `ησ²_eff log D(x) → 2α` and `ησ²_eff log (T(x)/D(x)) → 0` as `ησ²_eff → 0⁺`. The dwell
> therefore carries the exponent of `thm:main`(iii) and is strictly the shorter under the lift
> for every sufficiently small step size, while the fraction `D(x)/T(x)` of the time before
> escape spent below the barrier top carries none of that ordering.

This file is its machine-checked counterpart, and it is built exactly the way `KramersExitTime`
builds the exit time: the paper's proof says the exit-time argument "applies verbatim to a
bounded measurable right-hand side", and that is what is done here — the boundary-value problem,
its closed form, its uniqueness, the two Arrhenius bounds and the exponent limit are
**generalized** from the constant right-hand side to an arbitrary one, and then instantiated at
the band indicator. Nothing already proved is weakened: the exit-time statements of
`KramersExitTime` are the case `f ≡ 1` of the statements here, and both directions of that
identification are theorems (`occupationTimeBVP_one_iff`, `occupationTime_one`,
`meanExitTime_satisfies_bvp_of_occupation`, `eq_meanExitTime_of_occupationTimeBVP`).

**What is unavailable, and what is done instead.** As in `KramersExitTime`: this mathlib
(v4.31.0) has no Itô integral, no diffusion generator and no occupation time of a process, so
`E_x[∫₀^τ 1_{[a,w_s]}(w_t) dt]` cannot be stated here. What can be stated, and is, is the
boundary-value problem that it satisfies,

    (σ²/2) D''(x) - U'(x) D'(x) = -1_{[a,w_s]}(x)   on (a, b),   D(b) = 0,   D'(a⁺) = 0,

together with its Green-function solution

    D(x) = (2/σ²) ∫ₓᵇ exp(2U(y)/σ²) (∫ₐʸ 1_{[a,w_s]}(z) exp(-2U(z)/σ²) dz) dy,

which is the paper's `eq:pf-dwell`, that is `eq:pf-closed` with the inner integrand restricted.
The passage from the stochastic differential equation to that problem is **Dynkin's formula**
for the generator `A = (σ²/2) ∂² - U' ∂` applied to the occupation functional. It is the same
modeling input that `MeanExitTimeBVP` is for the exit time — not a second one — and it is
carried as a named hypothesis of the paper-facing theorems and never as an axiom.

**The one place where the indicator costs something.** The paper says the two differentiations
and the uniqueness argument apply verbatim to a bounded measurable right-hand side. The second
differentiation does not, quite: at the jump `w_s` of the indicator, the closed form's first
derivative has a corner and `D''(w_s)` does not exist. `OccupationTimeBVP` therefore asks for
the interior equation only at the points where the right-hand side is continuous, which for the
band indicator is every point of `(a, b)` except `w_s` itself (`continuousAt_bandIndicator`,
`not_continuousAt_bandIndicator`). That is the honest form, and it is not a weakening in
substance: uniqueness still holds (`eq_of_occupationTimeBVP`, through
`constant_of_hasDerivAt_zero_off_point`, which glues the two sides of the jump by continuity),
so the Dynkin hypothesis still pins its solution, and the closed form still satisfies the
problem (`dwellTime_satisfies_bvp`), so the hypothesis is not vacuous. The price is one extra
clause in the structure, `continuousOn_fst`, which is free for a continuous right-hand side
(`occupationTimeBVP_of_meanExitTimeBVP`) and is what the gluing consumes at a jump.

**What is proved.** The general closed form solves the general problem, by the same two
applications of the fundamental theorem of calculus, the inner one now at points of continuity
of `f` only; the problem has no other solution continuous up to `[a, b]`; the occupation time is
at most the exit time whenever `f ≤ 1`, by monotonicity of the integral, which is the paper's
"the integrands being non-negative"; the windowed Arrhenius lower bound holds with the same
window constant `L = 4ρ²` valid for every `σ² > 0` at once, because the inner window
`[w_b - ρ, w_b + ρ]` constructed there already lies inside the band `[a, w_s]` — the paper's
"that the inner window lies below the barrier top is the ordering `a < w_b < w_s` of (A2), and
it is the only step of `app:proof-fpt` that the indicator touches"; and the two bounds close, in
the small-noise limit, into `σ² log D(x) → 2α` and `σ² log (T(x)/D(x)) → 0`. The ordering clause
is the exit time's, on the family the corollary names — one shared step size `η → 0⁺` at two
fixed effective variances, the multiplicatively coupled pair of `thm:main`(iii) and of Step 3 of
`app:proof-fpt`.

## Results

The three clauses of `cor:dwell`, for the closed form and then given Dynkin's formula:

| clause of `cor:dwell` | closed form | given Dynkin |
|---|---|---|
| `ησ²_eff log D(x) → 2α` | `tendsto_mul_log_dwellTime` | `mean_dwell_kramers_exponent_of_dynkin` |
| `ησ²_eff log (T(x)/D(x)) → 0` | `tendsto_mul_log_meanExitTime_div_dwellTime` | `mean_dwell_ratio_of_dynkin` |
| the lift's dwell is the shorter, for every sufficiently small **step size** (`η → 0⁺`, variances `sd < sl` fixed) | `eventually_dwellTime_lift_lt_direct_smallLearningRate` | `eventually_mean_dwell_lift_lt_direct_smallLearningRate` |
| *companion, not this clause*: the same for the additively coupled family (`σ²` against `σ² + σ_Jac²`, the small parameter being the shared **variance**) | `eventually_dwellTime_additive_lift_lt_direct` | `eventually_mean_dwell_additive_lift_lt_direct` |

and the pieces they are built from:

* `occupationIntegral`, `occupationIntegrand`, `occupationTime`, `occupationTimeDeriv`,
  `occupationTimeDeriv2` — the Green-function closed form for a general right-hand side `f`, and
  its two derivatives; `OccupationIntegrable` is the admissibility hypothesis, discharged for
  bounded measurable `f` by `occupationIntegrable_of_bounded`.
* `hasDerivAt_occupationIntegral`, `hasDerivAt_occupationTime`,
  `hasDerivAt_occupationTimeDeriv` — the two differentiations; the second needs continuity of
  `f` at the point of differentiation and nowhere else.
* `OccupationTimeBVP` — the Dynkin boundary-value problem with right-hand side `-f`.
* `occupationTime_satisfies_bvp` — the closed form solves it.
* `constant_of_hasDerivAt_zero_off_point` — a continuous function with vanishing derivative off
  one point is constant; the gluing lemma the jump forces.
* `eq_of_occupationTimeBVP` — uniqueness, for a right-hand side continuous off at most one
  point.
* `occupationTime_one`, `occupationTimeDeriv_one`, `occupationTimeDeriv2_one`,
  `occupationTimeBVP_one_iff`, `meanExitTime_satisfies_bvp_of_occupation`,
  `eq_meanExitTime_of_occupationTimeBVP` — the case `f ≡ 1` is `KramersExitTime` verbatim, so
  the generalization weakens nothing.
* `bandIndicator` and its lemmas — `1_{[a,w_s]}`: measurable, in `[0,1]`, one on the band,
  continuous off `w_s` and discontinuous at `w_s`.
* `occupationTime_le_meanExitTime`, `dwellTime_le_meanExitTime` — `D ≤ T`.
* `intervalIntegral_le_of_subinterval'`, `const_mul_le_intervalIntegral'` — the two estimates of
  `KramersExitTime` with integrability in place of continuity, which the indicator needs.
* `occupationIntegral_window_lower_bound`, `occupationTime_window_lower_bound`,
  `occupationTime_arrhenius_lower_bound_uniform` — the Laplace lower bound, with the inner
  window inside the band.
* `BandRHS` — the right-hand sides the estimates are stated for: `0 ≤ f ≤ 1`, `f ≥ 1` on
  `[a, w_s]`, measurable. `bandRHS_one` and `bandRHS_bandIndicator` are the two instances, so
  every estimate below is a statement about `T` and about `D` at once.
* `occupationTime_pos`, `tendsto_mul_log_occupationTime`,
  `tendsto_mul_log_meanExitTime_div_occupationTime`,
  `eventually_occupationTime_lift_lt_direct_smallLearningRate`,
  `eventually_occupationTime_additive_lift_lt_direct` — the general forms of the three clauses,
  the ordering in both couplings.
* `meanExitTime_additive_upper_ceiling` — the one ceiling that bounds the whole lifted family,
  extracted from `KramersExitTime.eventually_meanExitTime_additive_lift_lt_direct`.
* `dwellTime`, `dwellTimeDeriv`, `dwellTimeDeriv2`, `dwellTime_satisfies_bvp`,
  `continuousOn_dwellTime` — the paper's `D(x)` and its boundary-value problem.
* `dwell_hypotheses_satisfiable`, `cubicBarrier_dwell_kramers_exponent`,
  `cubicBarrier_dwell_ratio`, `cubicBarrier_dwell_lift_lt_direct_smallLearningRate` —
  non-vacuity, on the cubic barrier of `KramersExitTime` with band
  `[-3/2, 1]`: the hypothesis stack is inhabited and the exponent runs to `8/3 = 2α`, the same
  number the exit time's does.

## Honesty

No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. What is carried rather than proved, and where the Lean
statement and the LaTeX one differ:

* **Dynkin's formula for the occupation functional**, `hDynkin : ∀ s > 0, OccupationTimeBVP s U'
  (bandIndicator a ws) a b (D s) (DD s) (DDD s)`, together with `ContinuousOn (D s) [a,b]`, in
  the three paper-facing theorems. This is the same modeling input as for the exit time, now for
  the occupation time, exactly as the paper's proof says ("the same modeling step that
  `eq:pf-bvp` is"). Discharging it needs an Itô integral and an optional-stopping argument,
  neither of which exists in this mathlib. It is not vacuous: `dwellTime_satisfies_bvp` exhibits
  a solution and `dwell_hypotheses_satisfiable` exhibits the whole stack.
* **The interior equation is required only off the jump.** The LaTeX says `D` solves
  `½σ²D'' - L̃'D' = -1_{[a,w_s]}` on `(a,b)`; the Lean asks for it at every point of `(a,b)` at
  which the right-hand side is continuous, that is at every point except `w_s`. At `w_s` the
  closed form has no second derivative, so the LaTeX reading is the one no function satisfies
  and the Lean one is the one the paper's own `eq:pf-dwell` satisfies. This is the single
  substantive gap between the two statements, and it cuts in the direction of the Lean being
  satisfiable rather than of its being stronger.
* **Constant diffusion coefficient.** `σ²` is a real number and not a function, in every
  definition and theorem, exactly as in `KramersExitTime`; the paper's `σ_eff(w̃)` is read as
  constant across the barrier band, which is the reduction `lem:diffusion`(v) brackets. Built
  into the definitions, not carried as a hypothesis.
* **`α` is `U(w_s) - U(w_b)`, and `σ²` is the paper's `ησ²_eff`.** The limits here are taken as
  `σ² → 0⁺` along a single real parameter, which is the paper's `ησ²_eff → 0⁺`; the Lean carries
  no `η` and no `σ²_eff` separately.
* **Two couplings, and only one of them is the corollary's clause.** The paper says "strictly
  the shorter under the lift for every sufficiently small step size", which is one shared
  learning rate going to zero at two *fixed* effective variances: the pair `(η·sd, η·sl)` with
  `sd < sl`, multiplicatively coupled, as in `thm:main`(iii) and Step 3 of `app:proof-fpt`. That
  is `eventually_dwellTime_lift_lt_direct_smallLearningRate` and its Dynkin form, added
  2026-09-13, mirroring `KramersBridge.eventually_meanExitTime_lift_lt_direct_smallLearningRate`,
  which draws the same distinction for the exit time.
  `eventually_dwellTime_additive_lift_lt_direct` is a **different** statement, kept as a
  companion: there the small parameter is itself the direct arm's diffusion and the lift's is
  that plus a fixed `σ_Jac² > 0`, so the gap does not shrink with the parameter. Neither implies
  the other, and only the first discharges the corollary's clause as written. In both, no
  threshold is produced — "sufficiently small" is a filter statement, exactly as in
  `KramersExitTime` — and no claim is made at a prescribed pair of diffusions.
* **`D(x)/T(x) → 1` is not claimed**, and neither is any lower bound on the ratio beyond the
  exponent statement. The paper's `rem:dwell-band` says the same, and names the condition
  (`inf_{(w_s,b]} L̃ > L̃(w_b)`) that would be needed; that condition appears nowhere here.
* **The general band of `rem:dwell-band` is not formalized.** The remark's exponent
  `2(L̃(w_s) - inf_B L̃)` for an arbitrary sub-interval `B` is not stated. What is stated is the
  band `[a, w_s]` of the corollary, through `BandRHS`, whose clause `one_le_on_band` is exactly
  the containment the remark names as the extra condition a general band would need.
* **The sharp Kramers prefactor** is not controlled, here as in `KramersExitTime`: the lower
  constant degenerates as the Laplace tolerance `δ → 0`, so at a fixed `σ²` the two bounds do not
  close, and they close only at the level of exponents in the small-noise limit.
-/

open MeasureTheory intervalIntegral Set Filter
open scoped Topology

namespace IcnnLift

/-! ### The occupation-time closed form, for a general right-hand side -/

/-- The inner integral `∫ₐʸ f(z) exp(-2U(z)/σ²) dz` of the occupation-time closed form: the
speed measure of `[a, y]` weighted by the right-hand side `f`, up to the constant `2/σ²`. At
`f ≡ 1` this is `speedIntegral`. -/
noncomputable def occupationIntegral (sigma2 : ℝ) (U f : ℝ → ℝ) (a y : ℝ) : ℝ :=
  ∫ z in a..y, f z * speedDensity sigma2 U z

/-- The integrand of the outer integral,
`exp(2U(y)/σ²) ∫ₐʸ f(z) exp(-2U(z)/σ²) dz`. At `f ≡ 1` this is `exitIntegrand`. -/
noncomputable def occupationIntegrand (sigma2 : ℝ) (U f : ℝ → ℝ) (a y : ℝ) : ℝ :=
  scaleDensity sigma2 U y * occupationIntegral sigma2 U f a y

/-- The **Green-function closed form** for the boundary-value problem with right-hand side `-f`,

`T(x) = (2/σ²) ∫ₓᵇ exp(2U(y)/σ²) (∫ₐʸ f(z) exp(-2U(z)/σ²) dz) dy`,

on `(a, b)`, reflecting at `a` and absorbing at `b`. At `f ≡ 1` this is `meanExitTime`
(`occupationTime_one`); at `f = 1_{[a,w_s]}` it is the mean occupation time of the band
`[a, w_s]` before absorption, which is the paper's `D(x)`. -/
noncomputable def occupationTime (sigma2 : ℝ) (U f : ℝ → ℝ) (a b x : ℝ) : ℝ :=
  2 / sigma2 * ∫ y in x..b, occupationIntegrand sigma2 U f a y

/-- The first derivative of `occupationTime`. -/
noncomputable def occupationTimeDeriv (sigma2 : ℝ) (U f : ℝ → ℝ) (a x : ℝ) : ℝ :=
  -(2 / sigma2) * occupationIntegrand sigma2 U f a x

/-- The second derivative of `occupationTime`, in the form the interior equation produces:
`T'' = (2U'/σ²) T' - 2f/σ²`. As in `KramersExitTime`, this is a definition, and the
mathematical content is `hasDerivAt_occupationTimeDeriv`, which proves that this closed formula
really is the derivative of `occupationTimeDeriv`. -/
noncomputable def occupationTimeDeriv2 (sigma2 : ℝ) (U U' f : ℝ → ℝ) (a x : ℝ) : ℝ :=
  2 * U' x / sigma2 * occupationTimeDeriv sigma2 U f a x - 2 * f x / sigma2

/-- The standing integrability hypothesis on the right-hand side: the weighted speed density is
integrable on the escape interval. A bounded measurable `f` satisfies it
(`occupationIntegrable_of_bounded`), which covers both `f ≡ 1` and the band indicator. -/
def OccupationIntegrable (sigma2 : ℝ) (U f : ℝ → ℝ) (a b : ℝ) : Prop :=
  IntegrableOn (fun z => f z * speedDensity sigma2 U z) (Set.Icc a b) volume

/-- A bounded measurable right-hand side is admissible. -/
theorem occupationIntegrable_of_bounded {sigma2 : ℝ} {U f : ℝ → ℝ} {a b c : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hfm : Measurable f) (hbd : ∀ z, |f z| ≤ c) :
    OccupationIntegrable sigma2 U f a b := by
  have hspeed : IntegrableOn (speedDensity sigma2 U) (Set.Icc a b) volume :=
    (continuousOn_speedDensity sigma2 hU).integrableOn_Icc
  refine hspeed.bdd_mul (c := c) hfm.aestronglyMeasurable ?_
  filter_upwards with z
  simpa [Real.norm_eq_abs] using hbd z

/-- Every subinterval of `[a, b]` carries an interval-integrable weighted speed density. -/
theorem intervalIntegrable_occupation {sigma2 : ℝ} {U f : ℝ → ℝ} {a b u v : ℝ}
    (hfi : OccupationIntegrable sigma2 U f a b) (hau : a ≤ u) (huv : u ≤ v) (hvb : v ≤ b) :
    IntervalIntegrable (fun z => f z * speedDensity sigma2 U z) volume u v := by
  refine MeasureTheory.IntegrableOn.intervalIntegrable (hfi.mono_set ?_)
  rw [Set.uIcc_of_le huv]
  exact Set.Icc_subset_Icc hau hvb

theorem occupationIntegral_self (sigma2 : ℝ) (U f : ℝ → ℝ) (a : ℝ) :
    occupationIntegral sigma2 U f a a = 0 := by
  simp [occupationIntegral]

theorem occupationIntegral_nonneg {sigma2 : ℝ} {U f : ℝ → ℝ} {a y : ℝ} (hay : a ≤ y)
    (hf : ∀ z ∈ Set.Icc a y, 0 ≤ f z) : 0 ≤ occupationIntegral sigma2 U f a y :=
  intervalIntegral.integral_nonneg hay fun z hz =>
    mul_nonneg (hf z hz) (speedDensity_pos sigma2 U z).le

theorem continuousOn_occupationIntegral {sigma2 : ℝ} {U f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
    (hfi : OccupationIntegrable sigma2 U f a b) :
    ContinuousOn (occupationIntegral sigma2 U f a) (Set.Icc a b) := by
  have hint : IntegrableOn (fun z => f z * speedDensity sigma2 U z) (Set.uIcc a b) volume := by
    rwa [Set.uIcc_of_le hab]
  have h := intervalIntegral.continuousOn_primitive_interval hint
  rwa [Set.uIcc_of_le hab] at h

theorem continuousOn_occupationIntegrand {sigma2 : ℝ} {U f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) (hfi : OccupationIntegrable sigma2 U f a b) :
    ContinuousOn (occupationIntegrand sigma2 U f a) (Set.Icc a b) :=
  (continuousOn_scaleDensity sigma2 hU).mul (continuousOn_occupationIntegral hab hfi)

theorem occupationIntegrand_nonneg {sigma2 : ℝ} {U f : ℝ → ℝ} {a y : ℝ} (hay : a ≤ y)
    (hf : ∀ z ∈ Set.Icc a y, 0 ≤ f z) : 0 ≤ occupationIntegrand sigma2 U f a y :=
  mul_nonneg (scaleDensity_pos sigma2 U y).le (occupationIntegral_nonneg hay hf)

/-! ### The two differentiations, for a general right-hand side

The outer differentiation goes through for every admissible `f`. The inner one is the only
place where the right-hand side has to be continuous, and it is needed **only at the point of
differentiation**: that is what lets the band indicator through, since it is continuous at every
point of `(a, b)` except the barrier top itself. -/

/-- Fundamental theorem of calculus, inner integral, at any point of continuity of `f`. -/
theorem hasDerivAt_occupationIntegral {sigma2 : ℝ} {U f : ℝ → ℝ} {a b x : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hfm : Measurable f)
    (hfi : OccupationIntegrable sigma2 U f a b) (hx : x ∈ Set.Ioo a b)
    (hfc : ContinuousAt f x) :
    HasDerivAt (occupationIntegral sigma2 U f a) (f x * speedDensity sigma2 U x) x := by
  have hspeedOoo : ContinuousOn (speedDensity sigma2 U) (Set.Ioo a b) :=
    (continuousOn_speedDensity sigma2 hU).mono Set.Ioo_subset_Icc_self
  refine intervalIntegral.integral_hasDerivAt_right
    (intervalIntegrable_occupation hfi le_rfl hx.1.le hx.2.le) ?_ ?_
  · refine ⟨Set.Ioo a b, isOpen_Ioo.mem_nhds hx, ?_⟩
    exact hfm.aestronglyMeasurable.mul (hspeedOoo.aestronglyMeasurable measurableSet_Ioo)
  · exact hfc.mul (hspeedOoo.continuousAt (isOpen_Ioo.mem_nhds hx))

/-- Fundamental theorem of calculus, outer integral: the closed form is differentiable in the
starting point for every admissible right-hand side, continuous or not. -/
theorem hasDerivAt_occupationTime {sigma2 : ℝ} {U f : ℝ → ℝ} {a b x : ℝ} (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) (hfi : OccupationIntegrable sigma2 U f a b)
    (hx : x ∈ Set.Ioo a b) :
    HasDerivAt (occupationTime sigma2 U f a b) (occupationTimeDeriv sigma2 U f a x) x := by
  have hcont : ContinuousOn (occupationIntegrand sigma2 U f a) (Set.Icc a b) :=
    continuousOn_occupationIntegrand hab hU hfi
  have hint : IntervalIntegrable (occupationIntegrand sigma2 U f a) volume x b := by
    refine ContinuousOn.intervalIntegrable (hcont.mono ?_)
    rw [Set.uIcc_of_le hx.2.le]
    exact Set.Icc_subset_Icc hx.1.le le_rfl
  have hbase : HasDerivAt (fun u => ∫ y in u..b, occupationIntegrand sigma2 U f a y)
      (-occupationIntegrand sigma2 U f a x) x := by
    refine intervalIntegral.integral_hasDerivAt_left hint ?_ ?_
    · exact ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo
        (hcont.mono Set.Ioo_subset_Icc_self) x hx
    · exact (hcont.mono Set.Ioo_subset_Icc_self).continuousAt (isOpen_Ioo.mem_nhds hx)
  have h2 : HasDerivAt (occupationTime sigma2 U f a b)
      (2 / sigma2 * -occupationIntegrand sigma2 U f a x) x := hbase.const_mul (2 / sigma2)
  refine h2.congr_deriv ?_
  simp only [occupationTimeDeriv]
  ring

/-- The second differentiation, at any point of continuity of the right-hand side. The
reciprocity `scaleDensity · speedDensity = 1` turns the product rule into the inhomogeneous term
`-2f(x)/σ²` of the interior equation. -/
theorem hasDerivAt_occupationTimeDeriv {sigma2 : ℝ} {U U' f : ℝ → ℝ} {a b x : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y)
    (hfm : Measurable f) (hfi : OccupationIntegrable sigma2 U f a b) (hx : x ∈ Set.Ioo a b)
    (hfc : ContinuousAt f x) :
    HasDerivAt (occupationTimeDeriv sigma2 U f a) (occupationTimeDeriv2 sigma2 U U' f a x) x := by
  have hscale : HasDerivAt (scaleDensity sigma2 U)
      (scaleDensity sigma2 U x * (2 * U' x / sigma2)) x :=
    (((hU' x hx).const_mul (2 : ℝ)).div_const sigma2).exp
  have hinner : HasDerivAt (occupationIntegral sigma2 U f a) (f x * speedDensity sigma2 U x) x :=
    hasDerivAt_occupationIntegral hU hfm hfi hx hfc
  have hprod : HasDerivAt (occupationIntegrand sigma2 U f a)
      (scaleDensity sigma2 U x * (2 * U' x / sigma2) * occupationIntegral sigma2 U f a x
        + scaleDensity sigma2 U x * (f x * speedDensity sigma2 U x)) x := hscale.mul hinner
  have hone : scaleDensity sigma2 U x * (f x * speedDensity sigma2 U x) = f x := by
    linear_combination f x * scaleDensity_mul_speedDensity sigma2 U x
  rw [hone] at hprod
  have hfinal := hprod.const_mul (-(2 / sigma2))
  refine hfinal.congr_deriv ?_
  simp only [occupationTimeDeriv2, occupationTimeDeriv, occupationIntegrand]
  ring

/-! ### The boundary-value problem with a general right-hand side -/

/-- The **Dynkin boundary-value problem for the mean occupation time** of a set with indicator
`f`: for the diffusion `dX = -U'(X) dt + σ dB` on `(a, b)`, reflecting at `a` and absorbing at
`b`, the mean time spent in `{f = 1}` before absorption solves

    (σ²/2) T''(x) - U'(x) T'(x) = -f(x)   on (a, b),      T(b) = 0,      T'(a⁺) = 0.

`MeanExitTimeBVP` is the case `f ≡ 1`, and the two are equivalent there
(`occupationTimeBVP_one_iff`), so nothing proved in `KramersExitTime` is weakened by passing to
this form.

Two departures from `MeanExitTimeBVP`, both forced by the fact that the right-hand side the
paper needs is an **indicator** and not a continuous function.

* The interior clauses are required only at the points where `f` is continuous. A function whose
  second derivative exists at a jump of `f` would have to satisfy an equation with two different
  right-hand sides there, so requiring them everywhere would make the problem unsatisfiable at
  the indicator, and every theorem carrying it vacuous. The closed form really does fail to be
  twice differentiable at the jump: its first derivative has a corner there.
* `continuousOn_fst` is a clause here and is not one in `MeanExitTimeBVP`. For a continuous
  right-hand side it is implied by `hasDerivAt_snd` and costs nothing
  (`occupationTimeBVP_of_meanExitTimeBVP`); at a jump it is what glues the two sides of the
  uniqueness argument together, and the closed form satisfies it. -/
structure OccupationTimeBVP (sigma2 : ℝ) (U' f : ℝ → ℝ) (a b : ℝ) (T DT DDT : ℝ → ℝ) : Prop where
  /-- `DT` is the derivative of `T` on the open interval. -/
  hasDerivAt_fst : ∀ x ∈ Set.Ioo a b, HasDerivAt T (DT x) x
  /-- `DT` is continuous on the open interval. -/
  continuousOn_fst : ContinuousOn DT (Set.Ioo a b)
  /-- `DDT` is the derivative of `DT` wherever the right-hand side is continuous. -/
  hasDerivAt_snd : ∀ x ∈ Set.Ioo a b, ContinuousAt f x → HasDerivAt DT (DDT x) x
  /-- The interior equation `(σ²/2) T''(x) - U'(x) T'(x) = -f(x)`, wherever `f` is continuous. -/
  generator : ∀ x ∈ Set.Ioo a b, ContinuousAt f x →
    sigma2 / 2 * DDT x - U' x * DT x = -f x
  /-- Absorbing boundary at `b`. -/
  absorbing : T b = 0
  /-- Reflecting boundary at `a`, as the one-sided limit `T'(a⁺) = 0`. -/
  reflecting : Filter.Tendsto DT (𝓝[>] a) (𝓝 0)

theorem occupationTime_right (sigma2 : ℝ) (U f : ℝ → ℝ) (a b : ℝ) :
    occupationTime sigma2 U f a b b = 0 := by
  simp [occupationTime]

theorem occupationTimeDeriv_left (sigma2 : ℝ) (U f : ℝ → ℝ) (a : ℝ) :
    occupationTimeDeriv sigma2 U f a a = 0 := by
  simp [occupationTimeDeriv, occupationIntegrand, occupationIntegral_self]

theorem continuousOn_occupationTimeDeriv {sigma2 : ℝ} {U f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) (hfi : OccupationIntegrable sigma2 U f a b) :
    ContinuousOn (occupationTimeDeriv sigma2 U f a) (Set.Icc a b) :=
  continuousOn_const.mul (continuousOn_occupationIntegrand hab hU hfi)

theorem tendsto_occupationTimeDeriv_right {sigma2 : ℝ} {U f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
    (hU : ContinuousOn U (Set.Icc a b)) (hfi : OccupationIntegrable sigma2 U f a b) :
    Filter.Tendsto (occupationTimeDeriv sigma2 U f a) (𝓝[>] a) (𝓝 0) := by
  have hcw : ContinuousWithinAt (occupationTimeDeriv sigma2 U f a) (Set.Icc a b) a :=
    continuousOn_occupationTimeDeriv hab.le hU hfi a ⟨le_rfl, hab.le⟩
  have hle : 𝓝[>] a ≤ 𝓝[Set.Icc a b] a := nhdsWithin_le_iff.mpr (Icc_mem_nhdsGT_left hab)
  have := hcw.tendsto.mono_left hle
  rwa [occupationTimeDeriv_left] at this

/-- **The Green-function closed form solves the boundary-value problem**, for every admissible
right-hand side. This is what makes the Dynkin hypothesis of the paper-facing theorems below
non-vacuous at the band indicator. -/
theorem occupationTime_satisfies_bvp {sigma2 : ℝ} {U U' f : ℝ → ℝ} {a b : ℝ} (hs : 0 < sigma2)
    (hab : a < b) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hfm : Measurable f)
    (hfi : OccupationIntegrable sigma2 U f a b) :
    OccupationTimeBVP sigma2 U' f a b (occupationTime sigma2 U f a b)
      (occupationTimeDeriv sigma2 U f a) (occupationTimeDeriv2 sigma2 U U' f a) where
  hasDerivAt_fst x hx := hasDerivAt_occupationTime hab.le hU hfi hx
  continuousOn_fst :=
    (continuousOn_occupationTimeDeriv hab.le hU hfi).mono Set.Ioo_subset_Icc_self
  hasDerivAt_snd x hx hfc := hasDerivAt_occupationTimeDeriv hU hU' hfm hfi hx hfc
  generator x _ _ := by
    have hne : sigma2 ≠ 0 := ne_of_gt hs
    simp only [occupationTimeDeriv2]
    field_simp
    ring
  absorbing := occupationTime_right sigma2 U f a b
  reflecting := tendsto_occupationTimeDeriv_right hab hU hfi

/-! ### Uniqueness, across a jump of the right-hand side

The classical uniqueness argument is a Wronskian computation, and the right-hand side cancels
out of it: two solutions of the *same* problem satisfy the same interior equation, so their
difference satisfies the homogeneous one whatever `f` is. The only thing a jump of `f` costs is
that the Wronskian is known to have vanishing derivative off the jump rather than everywhere,
and `constant_of_hasDerivAt_zero_off_point` closes that gap by continuity. -/

/-- A continuous function on an open interval whose derivative vanishes at every point except
possibly one is constant. The exceptional point is reached from the left by the interval
constancy and from the right by a limit, which is why continuity is a hypothesis and not a
consequence. -/
theorem constant_of_hasDerivAt_zero_off_point {W : ℝ → ℝ} {a b c : ℝ}
    (hcont : ContinuousOn W (Set.Ioo a b))
    (hderiv : ∀ x ∈ Set.Ioo a b, x ≠ c → HasDerivAt W 0 x) :
    ∀ u ∈ Set.Ioo a b, ∀ v ∈ Set.Ioo a b, W u = W v := by
  have key : ∀ u ∈ Set.Ioo a b, ∀ v ∈ Set.Ioo a b, u ≤ v →
      (∀ y ∈ Set.Ico u v, y ≠ c) → W u = W v := by
    intro u hu v hv huv hne
    have hsub : Set.Icc u v ⊆ Set.Ioo a b := Set.Icc_subset_Ioo hu.1 hv.2
    have hcw : ContinuousOn W (Set.Icc u v) := hcont.mono hsub
    have h := constant_of_has_deriv_right_zero hcw
      (fun y hy => (hderiv y (hsub (Set.Ico_subset_Icc_self hy)) (hne y hy)).hasDerivWithinAt) v
      (Set.right_mem_Icc.mpr huv)
    exact h.symm
  have mono : ∀ u ∈ Set.Ioo a b, ∀ v ∈ Set.Ioo a b, u ≤ v → W u = W v := by
    intro u hu v hv huv
    rcases le_or_gt v c with hvc | hcv
    · exact key u hu v hv huv fun y hy => ne_of_lt (lt_of_lt_of_le hy.2 hvc)
    · rcases lt_or_ge c u with hcu | huc
      · exact key u hu v hv huv fun y hy => ne_of_gt (lt_of_lt_of_le hcu hy.1)
      · have hcIoo : c ∈ Set.Ioo a b := ⟨lt_of_lt_of_le hu.1 huc, lt_trans hcv hv.2⟩
        have h1 : W u = W c := key u hu c hcIoo huc fun y hy => ne_of_lt hy.2
        have hlim : Filter.Tendsto W (𝓝[>] c) (𝓝 (W c)) :=
          (hcont c hcIoo).tendsto.mono_left
            (nhdsWithin_le_iff.mpr (mem_nhdsWithin_of_mem_nhds (isOpen_Ioo.mem_nhds hcIoo)))
        have hev : W =ᶠ[𝓝[>] c] fun _ => W v := by
          filter_upwards [Ioo_mem_nhdsGT hcv] with w hw
          exact key w ⟨lt_trans hcIoo.1 hw.1, lt_trans hw.2 hv.2⟩ v hv hw.2.le
            fun y hy => ne_of_gt (lt_of_lt_of_le hw.1 hy.1)
        have h2 : W c = W v := tendsto_nhds_unique (hlim.congr' hev) tendsto_const_nhds
        rw [h1, h2]
  intro u hu v hv
  rcases le_total u v with h | h
  · exact mono u hu v hv h
  · exact (mono v hv u hu h).symm

/-- **Uniqueness for the occupation-time boundary-value problem.** Any two solutions that are
continuous up to the closed interval coincide, for a right-hand side continuous off at most one
point. Nothing is assumed about the closed form, so the paper-facing theorems below can carry
Dynkin's formula alone. -/
theorem eq_of_occupationTimeBVP {sigma2 : ℝ} {U U' f : ℝ → ℝ} {a b c : ℝ}
    {T DT DDT V DV DDV : ℝ → ℝ} (hs : 0 < sigma2) (hab : a < b)
    (hU : ContinuousOn U (Set.Icc a b)) (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y)
    (hfc : ∀ x ∈ Set.Ioo a b, x ≠ c → ContinuousAt f x)
    (hTcont : ContinuousOn T (Set.Icc a b)) (hVcont : ContinuousOn V (Set.Icc a b))
    (hT : OccupationTimeBVP sigma2 U' f a b T DT DDT)
    (hV : OccupationTimeBVP sigma2 U' f a b V DV DDV) :
    ∀ x ∈ Set.Icc a b, T x = V x := by
  have hne : sigma2 ≠ 0 := ne_of_gt hs
  -- the Wronskian-type quantity has vanishing derivative off the jump
  have hgderiv : ∀ x ∈ Set.Ioo a b, x ≠ c →
      HasDerivAt (fun y => (DT y - DV y) * speedDensity sigma2 U y) 0 x := by
    intro x hx hxc
    have h1 : HasDerivAt (fun y => DT y - DV y) (DDT x - DDV x) x :=
      (hT.hasDerivAt_snd x hx (hfc x hx hxc)).sub (hV.hasDerivAt_snd x hx (hfc x hx hxc))
    have h2 : HasDerivAt (speedDensity sigma2 U)
        (speedDensity sigma2 U x * -(2 * U' x / sigma2)) x :=
      ((((hU' x hx).const_mul (2 : ℝ)).div_const sigma2).neg).exp
    have h3 : HasDerivAt (fun y => (DT y - DV y) * speedDensity sigma2 U y)
        ((DDT x - DDV x) * speedDensity sigma2 U x
          + (DT x - DV x) * (speedDensity sigma2 U x * -(2 * U' x / sigma2))) x := h1.mul h2
    refine h3.congr_deriv ?_
    have e1 : sigma2 / 2 * DDT x - U' x * DT x = -f x := hT.generator x hx (hfc x hx hxc)
    have e2 : sigma2 / 2 * DDV x - U' x * DV x = -f x := hV.generator x hx (hfc x hx hxc)
    have hnum : sigma2 * (DDT x - DDV x) - 2 * (U' x * (DT x - DV x)) = 0 := by
      linear_combination 2 * e1 - 2 * e2
    have hrw : (DDT x - DDV x) * speedDensity sigma2 U x
        + (DT x - DV x) * (speedDensity sigma2 U x * -(2 * U' x / sigma2))
        = speedDensity sigma2 U x
            * ((sigma2 * (DDT x - DDV x) - 2 * (U' x * (DT x - DV x))) / sigma2) := by
      field_simp
      ring
    rw [hrw, hnum]
    simp
  -- it is continuous on the open interval, hence constant there
  have hgcont : ContinuousOn (fun y => (DT y - DV y) * speedDensity sigma2 U y)
      (Set.Ioo a b) :=
    (hT.continuousOn_fst.sub hV.continuousOn_fst).mul
      ((continuousOn_speedDensity sigma2 hU).mono Set.Ioo_subset_Icc_self)
  have hgconst := constant_of_hasDerivAt_zero_off_point (c := c) hgcont hgderiv
  -- and the constant is zero, by the reflecting boundary condition
  have hgzero : ∀ x ∈ Set.Ioo a b, (DT x - DV x) * speedDensity sigma2 U x = 0 := by
    intro x hx
    have hE : Filter.Tendsto (speedDensity sigma2 U) (𝓝[>] a)
        (𝓝 (speedDensity sigma2 U a)) :=
      ((continuousOn_speedDensity sigma2 hU) a ⟨le_rfl, hab.le⟩).tendsto.mono_left
        (nhdsWithin_le_iff.mpr (Icc_mem_nhdsGT_left hab))
    have hlim : Filter.Tendsto (fun y => (DT y - DV y) * speedDensity sigma2 U y)
        (𝓝[>] a) (𝓝 0) := by
      have h := (hT.reflecting.sub hV.reflecting).mul hE
      simpa using h
    have hev : (fun y => (DT y - DV y) * speedDensity sigma2 U y) =ᶠ[𝓝[>] a]
        fun _ => (DT x - DV x) * speedDensity sigma2 U x := by
      filter_upwards [Ioo_mem_nhdsGT hab] with y hy
      exact hgconst y hy x hx
    exact tendsto_nhds_unique tendsto_const_nhds (hlim.congr' hev)
  -- so the two derivatives agree on the open interval
  have hDTeq : ∀ x ∈ Set.Ioo a b, DT x = DV x := by
    intro x hx
    rcases mul_eq_zero.mp (hgzero x hx) with h | h
    · linarith
    · exact absurd h (ne_of_gt (speedDensity_pos sigma2 U x))
  -- and therefore the two solutions agree, by the absorbing boundary condition
  have hDcont : ContinuousOn (fun y => T y - V y) (Set.Icc a b) := hTcont.sub hVcont
  have hdiff : ∀ x ∈ Set.Ioo a b, HasDerivAt (fun y => T y - V y) 0 x := by
    intro x hx
    have h3 : HasDerivAt (fun y => T y - V y) (DT x - DV x) x :=
      (hT.hasDerivAt_fst x hx).sub (hV.hasDerivAt_fst x hx)
    rwa [hDTeq x hx, sub_self] at h3
  have hIoo : ∀ x ∈ Set.Ioo a b, T x = V x := by
    intro x hx
    have hc := constant_of_has_deriv_right_zero (a := x) (b := b)
      (hDcont.mono (Set.Icc_subset_Icc hx.1.le le_rfl))
      (fun y hy => (hdiff y ⟨lt_of_lt_of_le hx.1 hy.1, hy.2⟩).hasDerivWithinAt)
    have hb := hc b (Set.right_mem_Icc.mpr hx.2.le)
    have hDb : T b - V b = 0 := by rw [hT.absorbing, hV.absorbing]; ring
    have hzero : T x - V x = 0 := by rw [← hb]; exact hDb
    linarith
  intro x hx
  rcases eq_or_lt_of_le hx.1 with hxa | hxa
  · have hlim : Filter.Tendsto (fun y => T y - V y) (𝓝[>] a) (𝓝 (T a - V a)) :=
      (hDcont a ⟨le_rfl, hab.le⟩).tendsto.mono_left
        (nhdsWithin_le_iff.mpr (Icc_mem_nhdsGT_left hab))
    have hev : (fun y => T y - V y) =ᶠ[𝓝[>] a] fun _ => (0 : ℝ) := by
      filter_upwards [Ioo_mem_nhdsGT hab] with y hy
      rw [hIoo y hy]; ring
    have h0 : T a - V a = 0 := tendsto_nhds_unique (hlim.congr' hev) tendsto_const_nhds
    rw [← hxa]
    linarith
  · rcases eq_or_lt_of_le hx.2 with hxb | hxb
    · rw [hxb, hT.absorbing, hV.absorbing]
    · exact hIoo x ⟨hxa, hxb⟩

/-! ### The constant right-hand side is the case `f ≡ 1`

Nothing proved in `KramersExitTime` is weakened by the passage to a general right-hand side:
the closed form, the boundary-value problem, the fact that the closed form solves it and the
uniqueness theorem all specialize back at `f ≡ 1`, and the two boundary-value problems are
equivalent there. -/

theorem occupationIntegral_one (sigma2 : ℝ) (U : ℝ → ℝ) (a y : ℝ) :
    occupationIntegral sigma2 U (fun _ => 1) a y = speedIntegral sigma2 U a y := by
  simp [occupationIntegral, speedIntegral]

theorem occupationIntegrand_one (sigma2 : ℝ) (U : ℝ → ℝ) (a y : ℝ) :
    occupationIntegrand sigma2 U (fun _ => 1) a y = exitIntegrand sigma2 U a y := by
  simp [occupationIntegrand, exitIntegrand, occupationIntegral_one]

theorem occupationTime_one (sigma2 : ℝ) (U : ℝ → ℝ) (a b x : ℝ) :
    occupationTime sigma2 U (fun _ => 1) a b x = meanExitTime sigma2 U a b x := by
  simp only [occupationTime, meanExitTime, occupationIntegrand_one]

theorem occupationTimeDeriv_one (sigma2 : ℝ) (U : ℝ → ℝ) (a x : ℝ) :
    occupationTimeDeriv sigma2 U (fun _ => 1) a x = meanExitTimeDeriv sigma2 U a x := by
  simp only [occupationTimeDeriv, meanExitTimeDeriv, occupationIntegrand_one]

theorem occupationTimeDeriv2_one (sigma2 : ℝ) (U U' : ℝ → ℝ) (a x : ℝ) :
    occupationTimeDeriv2 sigma2 U U' (fun _ => 1) a x = meanExitTimeDeriv2 sigma2 U U' a x := by
  simp only [occupationTimeDeriv2, meanExitTimeDeriv2, occupationTimeDeriv_one]
  norm_num

theorem occupationIntegrable_one {sigma2 : ℝ} {U : ℝ → ℝ} {a b : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) : OccupationIntegrable sigma2 U (fun _ => 1) a b :=
  occupationIntegrable_of_bounded (c := 1) hU measurable_const fun _ => by norm_num

/-- The exit-time boundary-value problem is the occupation-time one at `f ≡ 1`. The continuity
clause `continuousOn_fst` costs nothing here: a constant right-hand side makes `DT`
differentiable at every interior point. -/
theorem occupationTimeBVP_of_meanExitTimeBVP {sigma2 : ℝ} {U' : ℝ → ℝ} {a b : ℝ}
    {T DT DDT : ℝ → ℝ} (h : MeanExitTimeBVP sigma2 U' a b T DT DDT) :
    OccupationTimeBVP sigma2 U' (fun _ => 1) a b T DT DDT where
  hasDerivAt_fst := h.hasDerivAt_fst
  continuousOn_fst x hx := ((h.hasDerivAt_snd x hx).continuousAt).continuousWithinAt
  hasDerivAt_snd x hx _ := h.hasDerivAt_snd x hx
  generator x hx _ := by simpa using h.generator x hx
  absorbing := h.absorbing
  reflecting := h.reflecting

/-- And conversely. -/
theorem meanExitTimeBVP_of_occupationTimeBVP {sigma2 : ℝ} {U' : ℝ → ℝ} {a b : ℝ}
    {T DT DDT : ℝ → ℝ} (h : OccupationTimeBVP sigma2 U' (fun _ => 1) a b T DT DDT) :
    MeanExitTimeBVP sigma2 U' a b T DT DDT where
  hasDerivAt_fst := h.hasDerivAt_fst
  hasDerivAt_snd x hx := h.hasDerivAt_snd x hx continuousAt_const
  generator x hx := by simpa using h.generator x hx continuousAt_const
  absorbing := h.absorbing
  reflecting := h.reflecting

theorem occupationTimeBVP_one_iff {sigma2 : ℝ} {U' : ℝ → ℝ} {a b : ℝ} {T DT DDT : ℝ → ℝ} :
    OccupationTimeBVP sigma2 U' (fun _ => 1) a b T DT DDT ↔ MeanExitTimeBVP sigma2 U' a b T DT DDT :=
  ⟨meanExitTimeBVP_of_occupationTimeBVP, occupationTimeBVP_of_meanExitTimeBVP⟩

/-- `KramersExitTime.meanExitTime_satisfies_bvp`, re-derived from the general form. -/
theorem meanExitTime_satisfies_bvp_of_occupation {U U' : ℝ → ℝ} {a b : ℝ} {sigma2 : ℝ}
    (hs : 0 < sigma2) (hab : a < b) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) :
    MeanExitTimeBVP sigma2 U' a b (meanExitTime sigma2 U a b) (meanExitTimeDeriv sigma2 U a)
      (meanExitTimeDeriv2 sigma2 U U' a) := by
  have h := occupationTime_satisfies_bvp (f := fun _ => (1 : ℝ)) hs hab hU hU' measurable_const
    (occupationIntegrable_one hU)
  rw [show occupationTime sigma2 U (fun _ => 1) a b = meanExitTime sigma2 U a b from
      funext (occupationTime_one sigma2 U a b),
    show occupationTimeDeriv sigma2 U (fun _ => 1) a = meanExitTimeDeriv sigma2 U a from
      funext (occupationTimeDeriv_one sigma2 U a),
    show occupationTimeDeriv2 sigma2 U U' (fun _ => 1) a = meanExitTimeDeriv2 sigma2 U U' a from
      funext (occupationTimeDeriv2_one sigma2 U U' a)] at h
  exact meanExitTimeBVP_of_occupationTimeBVP h

/-- `KramersExitTime.eq_meanExitTime_of_bvp`, re-derived from the general uniqueness theorem. -/
theorem eq_meanExitTime_of_occupationTimeBVP {U U' : ℝ → ℝ} {a b : ℝ} {sigma2 : ℝ}
    {T DT DDT : ℝ → ℝ} (hs : 0 < sigma2) (hab : a < b) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hTcont : ContinuousOn T (Set.Icc a b))
    (hT : MeanExitTimeBVP sigma2 U' a b T DT DDT) :
    ∀ x ∈ Set.Icc a b, T x = meanExitTime sigma2 U a b x := by
  have hV := occupationTime_satisfies_bvp (f := fun _ => (1 : ℝ)) hs hab hU hU' measurable_const
    (occupationIntegrable_one hU)
  have hVcont : ContinuousOn (occupationTime sigma2 U (fun _ => 1) a b) (Set.Icc a b) := by
    rw [show occupationTime sigma2 U (fun _ => 1) a b = meanExitTime sigma2 U a b from
      funext (occupationTime_one sigma2 U a b)]
    exact continuousOn_meanExitTime sigma2 hab.le hU
  have h := eq_of_occupationTimeBVP (c := a) hs hab hU hU'
    (fun _ _ _ => continuousAt_const) hTcont hVcont
    (occupationTimeBVP_of_meanExitTimeBVP hT) hV
  intro x hx
  rw [h x hx, occupationTime_one]

/-! ### The band indicator

The right-hand side the corollary needs is the indicator of the band `[a, w_s]`, the part of the
escape interval below the barrier top. It is bounded and measurable, hence admissible, and it is
continuous at every interior point except the barrier top itself. -/

/-- The indicator `1_{[a,w_s]}` of the band below the barrier top. -/
noncomputable def bandIndicator (a ws : ℝ) : ℝ → ℝ :=
  Set.indicator (Set.Icc a ws) (fun _ => 1)

theorem measurable_bandIndicator (a ws : ℝ) : Measurable (bandIndicator a ws) :=
  measurable_const.indicator measurableSet_Icc

theorem bandIndicator_nonneg (a ws z : ℝ) : 0 ≤ bandIndicator a ws z :=
  Set.indicator_nonneg (fun _ _ => zero_le_one) z

theorem bandIndicator_le_one (a ws z : ℝ) : bandIndicator a ws z ≤ 1 := by
  unfold bandIndicator Set.indicator
  split_ifs <;> norm_num

theorem bandIndicator_abs_le_one (a ws z : ℝ) : |bandIndicator a ws z| ≤ 1 :=
  abs_le.mpr ⟨by linarith [bandIndicator_nonneg a ws z], bandIndicator_le_one a ws z⟩

theorem bandIndicator_eq_one {a ws z : ℝ} (hz : z ∈ Set.Icc a ws) : bandIndicator a ws z = 1 :=
  Set.indicator_of_mem hz _

theorem one_le_bandIndicator {a ws z : ℝ} (hz : z ∈ Set.Icc a ws) : 1 ≤ bandIndicator a ws z :=
  le_of_eq (bandIndicator_eq_one hz).symm

/-- The band indicator is continuous at every point of the escape interval except the barrier
top. -/
theorem continuousAt_bandIndicator {a b ws x : ℝ} (hx : x ∈ Set.Ioo a b) (hxws : x ≠ ws) :
    ContinuousAt (bandIndicator a ws) x := by
  rcases lt_or_gt_of_ne hxws with hlt | hgt
  · refine (continuousAt_const (y := (1 : ℝ))).congr ?_
    filter_upwards [isOpen_Ioo.mem_nhds (show x ∈ Set.Ioo a ws from ⟨hx.1, hlt⟩)] with z hz
    exact (bandIndicator_eq_one ⟨hz.1.le, hz.2.le⟩).symm
  · refine (continuousAt_const (y := (0 : ℝ))).congr ?_
    filter_upwards [isOpen_Ioi.mem_nhds (show x ∈ Set.Ioi ws from hgt)] with z hz
    exact (Set.indicator_of_notMem (fun hmem => absurd hmem.2 (not_le.mpr hz)) _).symm

/-- And it is **not** continuous at the barrier top, which is why the interior clauses of
`OccupationTimeBVP` are guarded by continuity: the closed form has a corner in its first
derivative there, and a problem demanding the interior equation at `w_s` would have no
solution. -/
theorem not_continuousAt_bandIndicator {a ws : ℝ} (haws : a ≤ ws) :
    ¬ ContinuousAt (bandIndicator a ws) ws := by
  intro hc
  have h1 : Filter.Tendsto (bandIndicator a ws) (𝓝[>] ws) (𝓝 (bandIndicator a ws ws)) :=
    hc.continuousWithinAt.tendsto
  have h0 : bandIndicator a ws ws = 1 := bandIndicator_eq_one ⟨haws, le_rfl⟩
  have hev : bandIndicator a ws =ᶠ[𝓝[>] ws] fun _ => (0 : ℝ) := by
    filter_upwards [self_mem_nhdsWithin] with z hz
    exact Set.indicator_of_notMem (fun hmem => absurd hmem.2 (not_le.mpr hz)) _
  rw [h0] at h1
  have : (0 : ℝ) = 1 := tendsto_nhds_unique tendsto_const_nhds (h1.congr' hev)
  norm_num at this

theorem occupationIntegrable_bandIndicator {sigma2 : ℝ} {U : ℝ → ℝ} {a b ws : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) :
    OccupationIntegrable sigma2 U (bandIndicator a ws) a b :=
  occupationIntegrable_of_bounded (c := 1) hU (measurable_bandIndicator a ws)
    (bandIndicator_abs_le_one a ws)

/-! ### The upper bound: the dwell is at most the exit time

`D(x) ≤ T(x)` is monotonicity of the closed form in the right-hand side, which is the paper's
"the integrands being non-negative". Everything the exit time's upper bound says about `T(x)`
therefore says the same about `D(x)`. -/

/-- A right-hand side between `0` and `1` produces an occupation time between `0` and the mean
exit time. -/
theorem occupationTime_le_meanExitTime {sigma2 : ℝ} {U f : ℝ → ℝ} {a b x : ℝ} (hs : 0 < sigma2)
    (hax : a ≤ x) (hxb : x ≤ b) (hU : ContinuousOn U (Set.Icc a b))
    (hfi : OccupationIntegrable sigma2 U f a b)
    (hf1 : ∀ z ∈ Set.Icc a b, f z ≤ 1) :
    occupationTime sigma2 U f a b x ≤ meanExitTime sigma2 U a b x := by
  have hab : a ≤ b := hax.trans hxb
  have hinner : ∀ y ∈ Set.Icc x b,
      occupationIntegral sigma2 U f a y ≤ speedIntegral sigma2 U a y := by
    intro y hy
    have hay : a ≤ y := hax.trans hy.1
    refine intervalIntegral.integral_mono_on hay
      (intervalIntegrable_occupation hfi le_rfl hay hy.2)
      (intervalIntegrable_speedDensity sigma2 hU le_rfl hay hy.2) ?_
    intro z hz
    have hz' : z ∈ Set.Icc a b := ⟨hz.1, hz.2.trans hy.2⟩
    nlinarith [speedDensity_pos sigma2 U z, hf1 z hz']
  have hpt : ∀ y ∈ Set.Icc x b,
      occupationIntegrand sigma2 U f a y ≤ exitIntegrand sigma2 U a y := by
    intro y hy
    exact mul_le_mul_of_nonneg_left (hinner y hy) (scaleDensity_pos sigma2 U y).le
  have houter : (∫ y in x..b, occupationIntegrand sigma2 U f a y)
      ≤ ∫ y in x..b, exitIntegrand sigma2 U a y := by
    refine intervalIntegral.integral_mono_on hxb ?_ ?_ hpt
    · refine ContinuousOn.intervalIntegrable ?_
      rw [Set.uIcc_of_le hxb]
      exact (continuousOn_occupationIntegrand hab hU hfi).mono (Set.Icc_subset_Icc hax le_rfl)
    · refine ContinuousOn.intervalIntegrable ?_
      rw [Set.uIcc_of_le hxb]
      exact (continuousOn_exitIntegrand sigma2 hab hU).mono (Set.Icc_subset_Icc hax le_rfl)
  have hfront : (0 : ℝ) ≤ 2 / sigma2 := div_nonneg (by norm_num) hs.le
  rw [occupationTime, meanExitTime]
  exact mul_le_mul_of_nonneg_left houter hfront

/-! ### The lower bound: the Laplace windows lie inside the band

This is the only step of the escape-time argument that the indicator touches. The inner window
of the exit-time lower bound is a neighbourhood of the basin bottom `w_b`, and the ordering
`a < w_b < w_s` of (A2) puts it inside the band `[a, w_s]`, where the indicator is one; the
bound is then the same bound. -/

/-- Restricting a non-negative integrand to a subinterval only decreases the integral —
`KramersExitTime.intervalIntegral_le_of_subinterval` with integrability in place of continuity,
because the band indicator is not continuous. -/
theorem intervalIntegral_le_of_subinterval' {f : ℝ → ℝ} {p u v q : ℝ}
    (hpu : p ≤ u) (huv : u ≤ v) (hvq : v ≤ q)
    (hint : ∀ r s : ℝ, p ≤ r → r ≤ s → s ≤ q → IntervalIntegrable f volume r s)
    (hpos : ∀ y ∈ Set.Icc p q, 0 ≤ f y) :
    (∫ y in u..v, f y) ≤ ∫ y in p..q, f y := by
  have huq : u ≤ q := huv.trans hvq
  have hpv : p ≤ v := hpu.trans huv
  have e1 : (∫ y in p..u, f y) + ∫ y in u..q, f y = ∫ y in p..q, f y :=
    intervalIntegral.integral_add_adjacent_intervals (hint p u le_rfl hpu huq)
      (hint u q hpu huq le_rfl)
  have e2 : (∫ y in u..v, f y) + ∫ y in v..q, f y = ∫ y in u..q, f y :=
    intervalIntegral.integral_add_adjacent_intervals (hint u v hpu huv hvq)
      (hint v q hpv hvq le_rfl)
  have n1 : 0 ≤ ∫ y in p..u, f y :=
    intervalIntegral.integral_nonneg hpu fun y hy => hpos y ⟨hy.1, hy.2.trans huq⟩
  have n2 : 0 ≤ ∫ y in v..q, f y :=
    intervalIntegral.integral_nonneg hvq fun y hy => hpos y ⟨hpv.trans hy.1, hy.2⟩
  linarith

/-- A constant lower bound on an interval integrates to a lower bound on the integral —
`KramersExitTime.const_mul_le_intervalIntegral` with integrability in place of continuity. -/
theorem const_mul_le_intervalIntegral' {f : ℝ → ℝ} {u v c : ℝ} (huv : u ≤ v)
    (hint : IntervalIntegrable f volume u v) (hle : ∀ y ∈ Set.Icc u v, c ≤ f y) :
    (v - u) * c ≤ ∫ y in u..v, f y := by
  have h := intervalIntegral.integral_mono_on huv intervalIntegral.intervalIntegrable_const hint hle
  simpa using h

/-- The inner integral, bounded below by its restriction to a window on which `U` does not exceed
`cb` **and on which the right-hand side is at least one**. This is the basin-bottom half of the
Laplace estimate, and the second condition is the only new hypothesis the occupation time needs
over the exit time. -/
theorem occupationIntegral_window_lower_bound {U f : ℝ → ℝ} {a b t₁ t₂ y cb : ℝ} {sigma2 : ℝ}
    (hs : 0 < sigma2)
    (hfi : OccupationIntegrable sigma2 U f a b) (hf0 : ∀ z ∈ Set.Icc a b, 0 ≤ f z)
    (hf1 : ∀ z ∈ Set.Icc t₁ t₂, 1 ≤ f z)
    (hat₁ : a ≤ t₁) (ht : t₁ ≤ t₂) (hty : t₂ ≤ y) (hyb : y ≤ b)
    (hbasin : ∀ z ∈ Set.Icc t₁ t₂, U z ≤ cb) :
    (t₂ - t₁) * Real.exp (-(2 * cb / sigma2)) ≤ occupationIntegral sigma2 U f a y := by
  have hay : a ≤ y := hat₁.trans (ht.trans hty)
  have hstep : (t₂ - t₁) * Real.exp (-(2 * cb / sigma2))
      ≤ ∫ z in t₁..t₂, f z * speedDensity sigma2 U z := by
    refine const_mul_le_intervalIntegral' ht
      (intervalIntegrable_occupation hfi hat₁ ht (hty.trans hyb)) ?_
    intro z hz
    have hspeed : Real.exp (-(2 * cb / sigma2)) ≤ speedDensity sigma2 U z := by
      rw [speedDensity, Real.exp_le_exp]
      have h2 : 2 * U z / sigma2 ≤ 2 * cb / sigma2 :=
        two_mul_div_le_two_mul_div hs (hbasin z hz)
      linarith
    nlinarith [hf1 z hz, Real.exp_pos (-(2 * cb / sigma2))]
  refine hstep.trans ?_
  refine intervalIntegral_le_of_subinterval' hat₁ ht hty
    (fun r s hr hrs hsy => intervalIntegrable_occupation hfi hr hrs (hsy.trans hyb)) ?_
  intro z hz
  exact mul_nonneg (hf0 z ⟨hz.1, hz.2.trans hyb⟩) (speedDensity_pos sigma2 U z).le

/-- **The Arrhenius lower bound for the occupation time, with explicit windows.** Identical to
`KramersExitTime.meanExitTime_window_lower_bound` except for the clause `hf1`, which asks that
the right-hand side be at least one on the inner window. -/
theorem occupationTime_window_lower_bound {U f : ℝ → ℝ} {a b x s₁ s₂ t₁ t₂ cs cb : ℝ}
    {sigma2 : ℝ} (hs : 0 < sigma2) (hU : ContinuousOn U (Set.Icc a b))
    (hfi : OccupationIntegrable sigma2 U f a b) (hf0 : ∀ z ∈ Set.Icc a b, 0 ≤ f z)
    (hf1 : ∀ z ∈ Set.Icc t₁ t₂, 1 ≤ f z)
    (hax : a ≤ x) (hxs₁ : x ≤ s₁) (hs₁₂ : s₁ ≤ s₂) (hs₂b : s₂ ≤ b)
    (hat₁ : a ≤ t₁) (ht₁₂ : t₁ ≤ t₂) (ht₂s₁ : t₂ ≤ s₁)
    (hbarrier : ∀ y ∈ Set.Icc s₁ s₂, cs ≤ U y)
    (hbasin : ∀ z ∈ Set.Icc t₁ t₂, U z ≤ cb) :
    2 / sigma2 * ((s₂ - s₁) * (t₂ - t₁)) * Real.exp (2 * (cs - cb) / sigma2)
      ≤ occupationTime sigma2 U f a b x := by
  have hxb : x ≤ b := hxs₁.trans (hs₁₂.trans hs₂b)
  have hab : a ≤ b := hax.trans hxb
  have hcont : ContinuousOn (occupationIntegrand sigma2 U f a) (Set.Icc x b) :=
    (continuousOn_occupationIntegrand hab hU hfi).mono (Set.Icc_subset_Icc hax le_rfl)
  have hpt : ∀ y ∈ Set.Icc s₁ s₂,
      Real.exp (2 * cs / sigma2) * ((t₂ - t₁) * Real.exp (-(2 * cb / sigma2)))
        ≤ occupationIntegrand sigma2 U f a y := by
    intro y hy
    have hyb : y ≤ b := hy.2.trans hs₂b
    have hscale : Real.exp (2 * cs / sigma2) ≤ scaleDensity sigma2 U y := by
      rw [scaleDensity, Real.exp_le_exp]
      exact two_mul_div_le_two_mul_div hs (hbarrier y hy)
    have hspeed : (t₂ - t₁) * Real.exp (-(2 * cb / sigma2))
        ≤ occupationIntegral sigma2 U f a y :=
      occupationIntegral_window_lower_bound hs hfi hf0 hf1 hat₁ ht₁₂
        (ht₂s₁.trans hy.1) hyb hbasin
    have hnn : 0 ≤ (t₂ - t₁) * Real.exp (-(2 * cb / sigma2)) :=
      mul_nonneg (by linarith) (Real.exp_pos _).le
    exact mul_le_mul hscale hspeed hnn (scaleDensity_pos sigma2 U y).le
  have hwindow : (s₂ - s₁) *
      (Real.exp (2 * cs / sigma2) * ((t₂ - t₁) * Real.exp (-(2 * cb / sigma2))))
        ≤ ∫ y in s₁..s₂, occupationIntegrand sigma2 U f a y :=
    const_mul_le_intervalIntegral hs₁₂ (hcont.mono (Set.Icc_subset_Icc hxs₁ hs₂b)) hpt
  have houter : (∫ y in s₁..s₂, occupationIntegrand sigma2 U f a y)
      ≤ ∫ y in x..b, occupationIntegrand sigma2 U f a y :=
    intervalIntegral_le_of_subinterval hxs₁ hs₁₂ hs₂b hcont
      (fun y hy => occupationIntegrand_nonneg (hax.trans hy.1)
        (fun z hz => hf0 z ⟨hz.1, hz.2.trans (hy.2.trans le_rfl)⟩))
  have hexp : Real.exp (2 * cs / sigma2) * Real.exp (-(2 * cb / sigma2))
      = Real.exp (2 * (cs - cb) / sigma2) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hfront : (0 : ℝ) < 2 / sigma2 := by positivity
  rw [occupationTime]
  have hkey : (s₂ - s₁) * (t₂ - t₁) * Real.exp (2 * (cs - cb) / sigma2)
      ≤ ∫ y in x..b, occupationIntegrand sigma2 U f a y := by
    have heq : (s₂ - s₁) * (t₂ - t₁) * Real.exp (2 * (cs - cb) / sigma2)
        = (s₂ - s₁) *
          (Real.exp (2 * cs / sigma2) * ((t₂ - t₁) * Real.exp (-(2 * cb / sigma2)))) := by
      rw [← hexp]; ring
    rw [heq]
    exact hwindow.trans houter
  calc 2 / sigma2 * ((s₂ - s₁) * (t₂ - t₁)) * Real.exp (2 * (cs - cb) / sigma2)
      = 2 / sigma2 * ((s₂ - s₁) * (t₂ - t₁) * Real.exp (2 * (cs - cb) / sigma2)) := by ring
    _ ≤ 2 / sigma2 * ∫ y in x..b, occupationIntegrand sigma2 U f a y :=
        mul_le_mul_of_nonneg_left hkey hfront.le

/-! ### The right-hand sides the corollary is stated for

`BandRHS` collects exactly what the two-sided estimate needs of the right-hand side: measurable,
between `0` and `1`, and at least `1` on the band `[a, w_s]`. The constant `1` and the band
indicator both satisfy it, so every theorem below is a statement about the exit time and about
the dwell at once, and the exit-time case is literally the same theorem. -/

/-- A **band right-hand side**: `0 ≤ f ≤ 1` everywhere, `f ≥ 1` on the band `[a, w_s]`,
measurable. `f ≡ 1` gives the mean exit time `T`; `f = 1_{[a,w_s]}` gives the dwell `D`. -/
structure BandRHS (f : ℝ → ℝ) (a ws : ℝ) : Prop where
  /-- Measurability, for the fundamental theorem of calculus. -/
  measurable : Measurable f
  /-- Non-negativity, for the monotonicity estimates. -/
  nonneg : ∀ z, 0 ≤ f z
  /-- Bounded by one, which is what makes the occupation time at most the exit time. -/
  le_one : ∀ z, f z ≤ 1
  /-- At least one on the band below the barrier top, which is what carries the lower bound. -/
  one_le_on_band : ∀ z ∈ Set.Icc a ws, 1 ≤ f z

theorem BandRHS.integrable {f U : ℝ → ℝ} {a ws b sigma2 : ℝ} (hf : BandRHS f a ws)
    (hU : ContinuousOn U (Set.Icc a b)) : OccupationIntegrable sigma2 U f a b :=
  occupationIntegrable_of_bounded (c := 1) hU hf.measurable fun z =>
    abs_le.mpr ⟨by linarith [hf.nonneg z], hf.le_one z⟩

/-- The constant right-hand side of the mean exit time is a band right-hand side. -/
theorem bandRHS_one (a ws : ℝ) : BandRHS (fun _ => 1) a ws where
  measurable := measurable_const
  nonneg _ := zero_le_one
  le_one _ := le_rfl
  one_le_on_band _ _ := le_rfl

/-- The band indicator of the corollary is a band right-hand side. -/
theorem bandRHS_bandIndicator (a ws : ℝ) : BandRHS (bandIndicator a ws) a ws where
  measurable := measurable_bandIndicator a ws
  nonneg := bandIndicator_nonneg a ws
  le_one := bandIndicator_le_one a ws
  one_le_on_band _ hz := one_le_bandIndicator hz

/-- **The Arrhenius lower bound for the occupation time, uniformly in the diffusion.** Word for
word `KramersExitTime.meanExitTime_arrhenius_lower_bound_uniform`, with the same window constant
`L = 4ρ²` valid for every `σ² > 0` at once. The one new step is the arithmetic that puts the
inner window inside the band: the common radius `ρ` already satisfies `ρ ≤ w_b - a` and
`ρ ≤ (w_s - w_b)/2`, so `[w_b - ρ, w_b + ρ] ⊆ [a, w_s]` and the right-hand side is at least one
there. That is the paper's "the inner window lies below the barrier top is the ordering
`a < w_b < w_s` of (A2), and it is the only step the indicator touches". -/
theorem occupationTime_arrhenius_lower_bound_uniform {U f : ℝ → ℝ} {a b x wb ws delta : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hf : BandRHS f a ws) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta) :
    ∃ L > 0, ∀ sigma2 : ℝ, 0 < sigma2 →
      2 / sigma2 * L * Real.exp (2 * (U ws - U wb - delta) / sigma2)
        ≤ occupationTime sigma2 U f a b x := by
  have hwsIoo : ws ∈ Set.Ioo a b := ⟨hawb.trans hwbws, hwsb⟩
  have hwbIoo : wb ∈ Set.Ioo a b := ⟨hawb, hwbws.trans hwsb⟩
  have hUoo : ContinuousOn U (Set.Ioo a b) := hU.mono Set.Ioo_subset_Icc_self
  have hcs : ContinuousAt U ws := hUoo.continuousAt (isOpen_Ioo.mem_nhds hwsIoo)
  have hcb : ContinuousAt U wb := hUoo.continuousAt (isOpen_Ioo.mem_nhds hwbIoo)
  obtain ⟨rs, hrs, hrsp⟩ := Metric.eventually_nhds_iff.mp
    (hcs.eventually (eventually_gt_nhds (show U ws - delta / 2 < U ws by linarith)))
  obtain ⟨rb, hrb, hrbp⟩ := Metric.eventually_nhds_iff.mp
    (hcb.eventually (eventually_lt_nhds (show U wb < U wb + delta / 2 by linarith)))
  obtain ⟨ρ, hρpos, hρ1, hρ2, hρ3, hρ4, hρ5, hρ6⟩ :
      ∃ ρ : ℝ, 0 < ρ ∧ ρ ≤ rs / 2 ∧ ρ ≤ rb / 2 ∧ ρ ≤ ws - x ∧ ρ ≤ b - ws ∧ ρ ≤ wb - a ∧
        ρ ≤ (ws - wb) / 2 := by
    refine ⟨min (min (rs / 2) (rb / 2))
      (min (min (ws - x) (b - ws)) (min (wb - a) ((ws - wb) / 2))), ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · repeat' apply lt_min
      all_goals linarith
    · exact (min_le_left _ _).trans (min_le_left _ _)
    · exact (min_le_left _ _).trans (min_le_right _ _)
    · exact (min_le_right _ _).trans ((min_le_left _ _).trans (min_le_left _ _))
    · exact (min_le_right _ _).trans ((min_le_left _ _).trans (min_le_right _ _))
    · exact (min_le_right _ _).trans ((min_le_right _ _).trans (min_le_left _ _))
    · exact (min_le_right _ _).trans ((min_le_right _ _).trans (min_le_right _ _))
  have hbarrier : ∀ y ∈ Set.Icc (ws - ρ) (ws + ρ), U ws - delta / 2 ≤ U y := by
    intro y hy
    obtain ⟨hy1, hy2⟩ := hy
    have hd : dist y ws < rs := by
      rw [Real.dist_eq, abs_lt]
      constructor <;> linarith
    exact (hrsp hd).le
  have hbasin : ∀ z ∈ Set.Icc (wb - ρ) (wb + ρ), U z ≤ U wb + delta / 2 := by
    intro z hz
    obtain ⟨hz1, hz2⟩ := hz
    have hd : dist z wb < rb := by
      rw [Real.dist_eq, abs_lt]
      constructor <;> linarith
    exact (hrbp hd).le
  -- the inner window lies inside the band, so the right-hand side is at least one on it
  have hf1 : ∀ z ∈ Set.Icc (wb - ρ) (wb + ρ), 1 ≤ f z := by
    intro z hz
    exact hf.one_le_on_band z ⟨by linarith [hz.1], by linarith [hz.2]⟩
  refine ⟨4 * ρ ^ 2, mul_pos (by norm_num) (pow_pos hρpos 2), fun sigma2 hs => ?_⟩
  have hmain := occupationTime_window_lower_bound (U := U) (f := f) (a := a) (b := b) (x := x)
    (s₁ := ws - ρ) (s₂ := ws + ρ) (t₁ := wb - ρ) (t₂ := wb + ρ)
    (cs := U ws - delta / 2) (cb := U wb + delta / 2) (sigma2 := sigma2)
    hs hU (hf.integrable hU) (fun z _ => hf.nonneg z) hf1 hax (by linarith) (by linarith)
    (by linarith) (by linarith) (by linarith) (by linarith) hbarrier hbasin
  have he : 2 * (U ws - delta / 2 - (U wb + delta / 2)) / sigma2
      = 2 * (U ws - U wb - delta) / sigma2 := by ring
  have hcconst : 2 / sigma2 * ((ws + ρ - (ws - ρ)) * (wb + ρ - (wb - ρ)))
      = 2 / sigma2 * (4 * ρ ^ 2) := by ring
  rw [he, hcconst] at hmain
  exact hmain

/-- The occupation time is strictly positive: the band carries a positive amount of speed
measure. -/
theorem occupationTime_pos {U f : ℝ → ℝ} {a b x wb ws : ℝ} {sigma2 : ℝ} (hs : 0 < sigma2)
    (hU : ContinuousOn U (Set.Icc a b)) (hf : BandRHS f a ws) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) :
    0 < occupationTime sigma2 U f a b x := by
  obtain ⟨L, hL, hbound⟩ := occupationTime_arrhenius_lower_bound_uniform (delta := 1)
    hU hf hax hawb hwbws hwsb hxws one_pos
  have h := hbound sigma2 hs
  have hpos : (0 : ℝ) < 2 / sigma2 * L * Real.exp (2 * (U ws - U wb - 1) / sigma2) :=
    mul_pos (mul_pos (div_pos (by norm_num) hs) hL) (Real.exp_pos _)
  linarith

/-- **The Kramers exponent of the occupation time, as a genuine limit.**

    `σ² · log D(x)  →  2α = 2 (U(w_s) - U(w_b))`   as `σ² → 0⁺`,

for every band right-hand side at once. At `f ≡ 1` this is
`KramersExitTime.tendsto_mul_log_meanExitTime`; at the band indicator it is the first clause of
the paper's `cor:dwell`. The lower bound is the windowed one, the upper bound is monotonicity
`D ≤ T` followed by the exit time's own upper bound, and the two crude prefactors disappear
because they are sub-exponential. -/
theorem tendsto_mul_log_occupationTime {U f : ℝ → ℝ} {a b x wb ws : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hf : BandRHS f a ws) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun s => s * Real.log (occupationTime s U f a b x)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * (U ws - U wb))) := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hab : a ≤ b := hax.trans hxb
  have hbx : (0 : ℝ) < b - x := by linarith
  have hba : (0 : ℝ) < b - a := by linarith
  rw [tendsto_order]
  constructor
  · intro c hc
    have heps : (0 : ℝ) < 2 * (U ws - U wb) - c := by linarith
    set d : ℝ := min ((U ws - U wb) / 2) ((2 * (U ws - U wb) - c) / 8) with hd_def
    have hd_pos : 0 < d := lt_min (by linarith) (by linarith)
    have hdle : d ≤ (2 * (U ws - U wb) - c) / 8 := by rw [hd_def]; exact min_le_right _ _
    obtain ⟨L, hL, hlow⟩ :=
      occupationTime_arrhenius_lower_bound_uniform (delta := d) hU hf hax hawb hwbws hwsb hxws
        hd_pos
    have h2L : (0 : ℝ) < 2 * L := by linarith
    have hpref := (tendsto_mul_log_const_div h2L).eventually
      (eventually_gt_nhds
        (show -((2 * (U ws - U wb) - c) / 4) < (0 : ℝ) by linarith))
    filter_upwards [hpref, self_mem_nhdsWithin] with s hs1 hs2
    have hspos : (0 : ℝ) < s := hs2
    have hlow' := hlow s hspos
    have heq : 2 / s * L = 2 * L / s := by ring
    rw [heq] at hlow'
    have hPpos : (0 : ℝ) < 2 * L / s * Real.exp (2 * (U ws - U wb - d) / s) :=
      mul_pos (div_pos h2L hspos) (Real.exp_pos _)
    have hlog : Real.log (2 * L / s * Real.exp (2 * (U ws - U wb - d) / s))
        ≤ Real.log (occupationTime s U f a b x) := Real.log_le_log hPpos hlow'
    have hmul := mul_le_mul_of_nonneg_left hlog hspos.le
    have hkey : s * Real.log (2 * L / s * Real.exp (2 * (U ws - U wb - d) / s))
        = s * Real.log (2 * L / s) + 2 * (U ws - U wb - d) := by
      rw [Real.log_mul (div_pos h2L hspos).ne' (Real.exp_ne_zero _), Real.log_exp, mul_add]
      congr 1
      field_simp
    rw [hkey] at hmul
    linarith
  · intro c hc
    have heps : (0 : ℝ) < c - 2 * (U ws - U wb) := by linarith
    have hA : (0 : ℝ) < 2 * ((b - x) * (b - a)) :=
      mul_pos (by norm_num) (mul_pos hbx hba)
    have hpref := (tendsto_mul_log_const_div hA).eventually
      (eventually_lt_nhds (show (0 : ℝ) < (c - 2 * (U ws - U wb)) / 2 by linarith))
    filter_upwards [hpref, self_mem_nhdsWithin] with s hs1 hs2
    have hspos : (0 : ℝ) < s := hs2
    have hDpos : (0 : ℝ) < occupationTime s U f a b x :=
      occupationTime_pos hspos hU hf hax hawb hwbws hwsb hxws
    have hupbd : occupationTime s U f a b x
        ≤ 2 / s * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / s) :=
      (occupationTime_le_meanExitTime hspos hax hxb hU (hf.integrable hU)
        (fun z _ => hf.le_one z)).trans
        (meanExitTime_upper_bound (sigma2 := s) (M := U ws) (m := U wb)
          hspos hax hxb hU hmax hmin)
    have heq : 2 / s * ((b - x) * (b - a)) = 2 * ((b - x) * (b - a)) / s := by ring
    rw [heq] at hupbd
    have hlog : Real.log (occupationTime s U f a b x)
        ≤ Real.log (2 * ((b - x) * (b - a)) / s * Real.exp (2 * (U ws - U wb) / s)) :=
      Real.log_le_log hDpos hupbd
    have hmul := mul_le_mul_of_nonneg_left hlog hspos.le
    have hkey : s * Real.log (2 * ((b - x) * (b - a)) / s * Real.exp (2 * (U ws - U wb) / s))
        = s * Real.log (2 * ((b - x) * (b - a)) / s) + 2 * (U ws - U wb) := by
      rw [Real.log_mul (div_pos hA hspos).ne' (Real.exp_ne_zero _), Real.log_exp, mul_add]
      congr 1
      field_simp
    rw [hkey] at hmul
    linarith

/-- **The fraction of the pre-escape time spent on the band carries no exponent.**

    `σ² · log (T(x) / D(x))  →  0`   as `σ² → 0⁺`.

This is the second clause of `cor:dwell`, and it is a subtraction: both `σ² log T` and
`σ² log D` tend to `2α`, so their difference tends to `0`. Note what it does **not** say: the
ratio `D/T` is not claimed to tend to `1` — that needs a hypothesis on `U` past the barrier top
which (A2) does not supply — only that it is sub-exponential in the effective noise, which is
what makes it carry no part of an ordering that lives in an exponent. -/
theorem tendsto_mul_log_meanExitTime_div_occupationTime {U f : ℝ → ℝ} {a b x wb ws : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hf : BandRHS f a ws) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto
      (fun s => s * Real.log (meanExitTime s U a b x / occupationTime s U f a b x))
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hT := tendsto_mul_log_occupationTime (f := fun _ => 1) hU (bandRHS_one a ws) hax
    hawb hwbws hwsb hxws hbarrier hmax hmin
  have hD := tendsto_mul_log_occupationTime (f := f) hU hf hax hawb hwbws hwsb hxws hbarrier
    hmax hmin
  have hdiff := hT.sub hD
  rw [sub_self] at hdiff
  refine hdiff.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hspos : (0 : ℝ) < s := hs
  have hDpos : (0 : ℝ) < occupationTime s U f a b x :=
    occupationTime_pos hspos hU hf hax hawb hwbws hwsb hxws
  have hTpos : (0 : ℝ) < occupationTime s U (fun _ => 1) a b x :=
    occupationTime_pos hspos hU (bandRHS_one a ws) hax hawb hwbws hwsb hxws
  rw [occupationTime_one] at hTpos ⊢
  rw [Real.log_div hTpos.ne' hDpos.ne']
  ring

/-- The whole additively coupled family sits below one ceiling, because a larger diffusion
improves both the prefactor and the exponent of the exit time's upper bound. This is the `hup`
step inside `KramersExitTime.eventually_meanExitTime_additive_lift_lt_direct`, isolated here so
that the occupation time can reuse it through `D ≤ T`. -/
theorem meanExitTime_additive_upper_ceiling {U : ℝ → ℝ} {a b x wb ws jac2 sd : ℝ}
    (hjac : 0 < jac2) (hsd : 0 < sd) (hax : a ≤ x) (hxb : x ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) (hle : U wb ≤ U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    meanExitTime (sd + jac2) U a b x
      ≤ 2 / jac2 * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / jac2) := by
  have hab : a ≤ b := hax.trans hxb
  have hsl : (0 : ℝ) < sd + jac2 := by linarith
  have h1 := meanExitTime_upper_bound (sigma2 := sd + jac2) (M := U ws) (m := U wb)
    hsl hax hxb hU hmax hmin
  have hAnn : (0 : ℝ) ≤ (b - x) * (b - a) := mul_nonneg (by linarith) (by linarith)
  have hfrac : 2 / (sd + jac2) ≤ 2 / jac2 := by
    rw [div_le_div_iff₀ hsl hjac]
    linarith
  have hexple : Real.exp (2 * (U ws - U wb) / (sd + jac2))
      ≤ Real.exp (2 * (U ws - U wb) / jac2) := by
    rw [Real.exp_le_exp, div_le_div_iff₀ hsl hjac]
    nlinarith [mul_nonneg hsd.le (show (0 : ℝ) ≤ 2 * (U ws - U wb) by linarith)]
  refine h1.trans ?_
  have hstep1 : 2 / (sd + jac2) * ((b - x) * (b - a))
      ≤ 2 / jac2 * ((b - x) * (b - a)) := mul_le_mul_of_nonneg_right hfrac hAnn
  have hnn : (0 : ℝ) ≤ 2 / jac2 * ((b - x) * (b - a)) := mul_nonneg (by positivity) hAnn
  exact mul_le_mul hstep1 hexple (Real.exp_pos _).le hnn

/-- **The dwell is strictly the shorter under the lift, for every sufficiently small step
size.** The paper's own coupled family: the direct process at effective variance `sd`, the
lifted one at `sd + σ_Jac²` with `σ_Jac² > 0` fixed. The proof is the exit time's, with the
lower bound taken at the band right-hand side and the upper bound routed through `D ≤ T`. -/
theorem eventually_occupationTime_additive_lift_lt_direct {U f : ℝ → ℝ} {a b x wb ws jac2 : ℝ}
    (hjac : 0 < jac2) (hU : ContinuousOn U (Set.Icc a b)) (hf : BandRHS f a ws) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      occupationTime (sd + jac2) U f a b x < occupationTime sd U f a b x := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hdelta : (0 : ℝ) < (U ws - U wb) / 2 := by linarith
  obtain ⟨L, hL, hbound⟩ :=
    occupationTime_arrhenius_lower_bound_uniform (delta := (U ws - U wb) / 2)
      hU hf hax hawb hwbws hwsb hxws hdelta
  have hup : ∀ sd : ℝ, 0 < sd →
      occupationTime (sd + jac2) U f a b x
        ≤ 2 / jac2 * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / jac2) := by
    intro sd hsd
    have hsl : (0 : ℝ) < sd + jac2 := by linarith
    have h0 : occupationTime (sd + jac2) U f a b x ≤ meanExitTime (sd + jac2) U a b x :=
      occupationTime_le_meanExitTime hsl hax hxb hU (hf.integrable hU) fun z _ => hf.le_one z
    exact h0.trans (meanExitTime_additive_upper_ceiling hjac hsd hax hxb hU hbarrier.le hmax hmin)
  have hbeta : 0 < 2 * (U ws - U wb - (U ws - U wb) / 2) := by linarith
  have hgap := eventually_lower_bound_gt (beta := 2 * (U ws - U wb - (U ws - U wb) / 2))
    (L := L)
    (kappa := 2 / jac2 * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / jac2)) hbeta hL
  filter_upwards [hgap, self_mem_nhdsWithin] with sd hsd1 hsd2
  have hsdpos : (0 : ℝ) < sd := hsd2
  have hlow := hbound sd hsdpos
  linarith [hup sd hsdpos]

/-- **The ordering at one shared step size, for the multiplicatively coupled pair.** This is
the family `cor:dwell`'s third clause is stated on, and the one Step 3 of `app:proof-fpt` uses:
one learning rate `eta` going to zero, two *fixed* effective variances `sd < sl`, so that the
two diffusion coefficients are `eta * sd` and `eta * sl`. It is a different statement from
`eventually_occupationTime_additive_lift_lt_direct`, where the small parameter is itself the
direct arm's diffusion and the two members differ by a fixed constant; neither implies the
other, and `KramersBridge` draws exactly this distinction for the exit time
(`eventually_meanExitTime_lift_lt_direct_smallLearningRate`, which this mirrors).

The proof is the bridge's: the Laplace tolerance `δ = α(sl - sd)/(2 sl)` separates the two
exponent scales by `γ = α(sl - sd)/(sl sd) > 0`, the lower bound is taken at `eta * sd` with the
diffusion-uniform window constant, and the upper bound at `eta * sl` is routed through
`D ≤ T`. -/
theorem eventually_occupationTime_lift_lt_direct_smallLearningRate {U f : ℝ → ℝ}
    {a b x wb ws sd sl : ℝ} (hsd : 0 < sd) (hsdl : sd < sl)
    (hU : ContinuousOn U (Set.Icc a b)) (hf : BandRHS f a ws) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ),
      occupationTime (eta * sl) U f a b x < occupationTime (eta * sd) U f a b x := by
  have hsl : 0 < sl := hsd.trans hsdl
  have halpha : 0 < U ws - U wb := by linarith
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  -- the Laplace tolerance, chosen so that the two exponent scales separate
  set delta : ℝ := (U ws - U wb) * (sl - sd) / (2 * sl) with hdelta_def
  have hdelta : 0 < delta := by
    rw [hdelta_def]
    exact div_pos (mul_pos halpha (by linarith)) (by linarith)
  obtain ⟨L, hL, hbound⟩ :=
    occupationTime_arrhenius_lower_bound_uniform (delta := delta) hU hf hax hawb hwbws hwsb
      hxws hdelta
  -- the separation of the two exponent scales
  set gamma : ℝ := 2 * (U ws - U wb - delta) / sd - 2 * (U ws - U wb) / sl with hgamma_def
  have hgamma_eq : gamma = (U ws - U wb) * (sl - sd) / (sl * sd) := by
    rw [hgamma_def, hdelta_def]
    field_simp
    ring
  have hgamma : 0 < gamma := by
    rw [hgamma_eq]
    exact div_pos (mul_pos halpha (by linarith)) (mul_pos hsl hsd)
  filter_upwards [eventually_lt_exp_div hgamma ((b - x) * (b - a) * sd / (sl * L)),
    self_mem_nhdsWithin] with eta h1 h2
  have heta : (0 : ℝ) < eta := h2
  have hlow := hbound (eta * sd) (mul_pos heta hsd)
  have hup : occupationTime (eta * sl) U f a b x
      ≤ 2 / (eta * sl) * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / (eta * sl)) :=
    (occupationTime_le_meanExitTime (mul_pos heta hsl) hax hxb hU (hf.integrable hU)
      (fun z _ => hf.le_one z)).trans
      (meanExitTime_upper_bound (sigma2 := eta * sl) (M := U ws) (m := U wb)
        (mul_pos heta hsl) hax hxb hU hmax hmin)
  refine lt_of_le_of_lt hup (lt_of_lt_of_le ?_ hlow)
  -- the gap between the two explicit bounds, at this learning rate
  have hexpsub : Real.exp (gamma / eta)
      = Real.exp (2 * (U ws - U wb - delta) / (eta * sd))
        / Real.exp (2 * (U ws - U wb) / (eta * sl)) := by
    rw [← Real.exp_sub]
    congr 1
    rw [hgamma_def]
    field_simp
  rw [hexpsub] at h1
  have hpos2 : (0 : ℝ) < Real.exp (2 * (U ws - U wb) / (eta * sl)) := Real.exp_pos _
  have hstep1 : (b - x) * (b - a) * sd / (sl * L)
      * Real.exp (2 * (U ws - U wb) / (eta * sl))
      < Real.exp (2 * (U ws - U wb - delta) / (eta * sd)) := by
    have h := mul_lt_mul_of_pos_right h1 hpos2
    rwa [div_mul_cancel₀ _ hpos2.ne'] at h
  have hc : (0 : ℝ) < 2 * L / (eta * sd) := by positivity
  have hstep2 := mul_lt_mul_of_pos_left hstep1 hc
  calc 2 / (eta * sl) * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / (eta * sl))
      = 2 * L / (eta * sd)
          * ((b - x) * (b - a) * sd / (sl * L) * Real.exp (2 * (U ws - U wb) / (eta * sl))) := by
        field_simp
    _ < 2 * L / (eta * sd) * Real.exp (2 * (U ws - U wb - delta) / (eta * sd)) := hstep2
    _ = 2 / (eta * sd) * L * Real.exp (2 * (U ws - U wb - delta) / (eta * sd)) := by ring

/-! ### The dwell `D(x)` of `cor:dwell`

`dwellTime` is the closed form (eq:pf-dwell) of the paper's

`D(x) = E_x[∫₀^τ 1_{[a,w_s]}(w_t) dt]`,

and the theorems below are the clauses of the corollary, at the level at which this mathlib can
state them: about the boundary-value problem and its solution, not about an occupation time of a
process. -/

theorem continuousOn_occupationTime {sigma2 : ℝ} {U f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) (hfi : OccupationIntegrable sigma2 U f a b) :
    ContinuousOn (occupationTime sigma2 U f a b) (Set.Icc a b) := by
  have hint : IntegrableOn (occupationIntegrand sigma2 U f a) (Set.uIcc a b) volume := by
    rw [Set.uIcc_of_le hab]
    exact (continuousOn_occupationIntegrand hab hU hfi).integrableOn_Icc
  have h := intervalIntegral.continuousOn_primitive_interval_left hint
  rw [Set.uIcc_of_le hab] at h
  show ContinuousOn
    (fun x => 2 / sigma2 * ∫ y in x..b, occupationIntegrand sigma2 U f a y) (Set.Icc a b)
  exact continuousOn_const.mul h

/-- **The dwell**, as the Green-function closed form of the paper's `eq:pf-dwell`:

`D(x) = (2/σ²) ∫ₓᵇ exp(2U(y)/σ²) (∫ₐʸ 1_{[a,w_s]}(z) exp(-2U(z)/σ²) dz) dy`,

which is `meanExitTime` with the inner integrand restricted to the band. -/
noncomputable def dwellTime (sigma2 : ℝ) (U : ℝ → ℝ) (a ws b x : ℝ) : ℝ :=
  occupationTime sigma2 U (bandIndicator a ws) a b x

/-- The first derivative of the dwell. -/
noncomputable def dwellTimeDeriv (sigma2 : ℝ) (U : ℝ → ℝ) (a ws x : ℝ) : ℝ :=
  occupationTimeDeriv sigma2 U (bandIndicator a ws) a x

/-- The second derivative of the dwell, off the barrier top. -/
noncomputable def dwellTimeDeriv2 (sigma2 : ℝ) (U U' : ℝ → ℝ) (a ws x : ℝ) : ℝ :=
  occupationTimeDeriv2 sigma2 U U' (bandIndicator a ws) a x

/-- **The dwell's closed form solves the dwell's boundary-value problem.** This is what makes
the Dynkin hypothesis of the paper-facing theorems non-vacuous. -/
theorem dwellTime_satisfies_bvp {sigma2 : ℝ} {U U' : ℝ → ℝ} {a b ws : ℝ} (hs : 0 < sigma2)
    (hab : a < b) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) :
    OccupationTimeBVP sigma2 U' (bandIndicator a ws) a b (dwellTime sigma2 U a ws b)
      (dwellTimeDeriv sigma2 U a ws) (dwellTimeDeriv2 sigma2 U U' a ws) :=
  occupationTime_satisfies_bvp hs hab hU hU' (measurable_bandIndicator a ws)
    (occupationIntegrable_bandIndicator hU)

theorem continuousOn_dwellTime {sigma2 : ℝ} {U : ℝ → ℝ} {a b ws : ℝ} (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) :
    ContinuousOn (dwellTime sigma2 U a ws b) (Set.Icc a b) :=
  continuousOn_occupationTime hab hU (occupationIntegrable_bandIndicator hU)

/-- **The dwell is at most the exit time**, the paper's "the integrands being non-negative". -/
theorem dwellTime_le_meanExitTime {sigma2 : ℝ} {U : ℝ → ℝ} {a b ws x : ℝ} (hs : 0 < sigma2)
    (hax : a ≤ x) (hxb : x ≤ b) (hU : ContinuousOn U (Set.Icc a b)) :
    dwellTime sigma2 U a ws b x ≤ meanExitTime sigma2 U a b x :=
  occupationTime_le_meanExitTime hs hax hxb hU (occupationIntegrable_bandIndicator hU)
    fun z _ => bandIndicator_le_one a ws z

/-- **`cor:dwell`, first clause, for the closed form**: `ησ²_eff log D(x) → 2α`. -/
theorem tendsto_mul_log_dwellTime {U : ℝ → ℝ} {a b x wb ws : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun s => s * Real.log (dwellTime s U a ws b x)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * (U ws - U wb))) :=
  tendsto_mul_log_occupationTime hU (bandRHS_bandIndicator a ws) hax hawb hwbws hwsb hxws
    hbarrier hmax hmin

/-- **`cor:dwell`, second clause, for the closed forms**: `ησ²_eff log (T(x)/D(x)) → 0`. -/
theorem tendsto_mul_log_meanExitTime_div_dwellTime {U : ℝ → ℝ} {a b x wb ws : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto
      (fun s => s * Real.log (meanExitTime s U a b x / dwellTime s U a ws b x))
      (𝓝[>] (0 : ℝ)) (𝓝 0) :=
  tendsto_mul_log_meanExitTime_div_occupationTime hU (bandRHS_bandIndicator a ws) hax hawb
    hwbws hwsb hxws hbarrier hmax hmin

/-- **`cor:dwell`, third clause, for the closed forms**: the dwell is strictly the shorter under
the lift for every sufficiently small shared variance, on the paper's own additively coupled
family `σ²` versus `σ² + σ_Jac²`. -/
theorem eventually_dwellTime_additive_lift_lt_direct {U : ℝ → ℝ} {a b x wb ws jac2 : ℝ}
    (hjac : 0 < jac2) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      dwellTime (sd + jac2) U a ws b x < dwellTime sd U a ws b x :=
  eventually_occupationTime_additive_lift_lt_direct hjac hU (bandRHS_bandIndicator a ws) hax
    hawb hwbws hwsb hxws hbarrier hmax hmin

/-- **`cor:dwell`, third clause, for the closed forms: "strictly the shorter under the lift for
every sufficiently small step size".** One shared step size `eta → 0⁺`, two fixed effective
variances `sd < sl` — the multiplicatively coupled pair of `thm:main`(iii) and of Step 3 of
`app:proof-fpt`, in which the diffusion coefficient of the boundary-value problem is
`η σ²_eff`. This is the theorem that discharges the corollary's clause as the corollary states
it; `eventually_dwellTime_additive_lift_lt_direct` above is the companion statement for the
additively coupled family of `eq:sigma-eff`'s variance offset, where the small parameter is the
shared variance and not the step size, and neither implies the other. -/
theorem eventually_dwellTime_lift_lt_direct_smallLearningRate {U : ℝ → ℝ}
    {a b x wb ws sd sl : ℝ} (hsd : 0 < sd) (hsdl : sd < sl)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ),
      dwellTime (eta * sl) U a ws b x < dwellTime (eta * sd) U a ws b x :=
  eventually_occupationTime_lift_lt_direct_smallLearningRate hsd hsdl hU
    (bandRHS_bandIndicator a ws) hax hawb hwbws hwsb hxws hbarrier hmax hmin

/-! ### The paper-facing statements

As in `KramersExitTime`, each carries one named analytic hypothesis, `hDynkin`, and nothing
else is assumed: that the mean occupation time of the band solves the boundary-value problem
with right-hand side `-1_{[a,w_s]}` is Dynkin's formula for the generator
`A = (σ²/2) ∂² - U' ∂` applied to the occupation functional, the same modeling step that
`MeanExitTimeBVP` is for the exit time and the only one this mathlib cannot express. It does not
smuggle in the closed form: that passage is `eq_of_occupationTimeBVP`, proved above. -/

/-- **`cor:dwell`, first clause.** Given Dynkin's formula for the occupation functional, the
Kramers exponent of the dwell is the barrier action: `ησ²_eff log D(x) → 2α`. -/
theorem mean_dwell_kramers_exponent_of_dynkin {U U' : ℝ → ℝ} {a b x wb ws : ℝ}
    (D DD DDD : ℝ → ℝ → ℝ)
    (hDynkin : ∀ s : ℝ, 0 < s →
      OccupationTimeBVP s U' (bandIndicator a ws) a b (D s) (DD s) (DDD s))
    (hD : ∀ s : ℝ, 0 < s → ContinuousOn (D s) (Set.Icc a b))
    (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun s => s * Real.log (D s x)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * (U ws - U wb))) := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  refine (tendsto_mul_log_dwellTime hU hax hawb hwbws hwsb hxws hbarrier hmax hmin).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hspos : (0 : ℝ) < s := hs
  have h := eq_of_occupationTimeBVP (c := ws) hspos hab hU hU'
    (fun y hy hyws => continuousAt_bandIndicator hy hyws) (hD s hspos)
    (continuousOn_dwellTime hab.le hU) (hDynkin s hspos)
    (dwellTime_satisfies_bvp hspos hab hU hU')
  rw [h x ⟨hax, hxb⟩]

/-- **`cor:dwell`, second clause.** Given Dynkin's formula for both functionals, the fraction of
the pre-escape time spent on the band carries no exponent: `ησ²_eff log (T(x)/D(x)) → 0`. -/
theorem mean_dwell_ratio_of_dynkin {U U' : ℝ → ℝ} {a b x wb ws : ℝ}
    (tau Dtau DDtau D DD DDD : ℝ → ℝ → ℝ)
    (hDynkinTau : ∀ s : ℝ, 0 < s → MeanExitTimeBVP s U' a b (tau s) (Dtau s) (DDtau s))
    (hDynkinD : ∀ s : ℝ, 0 < s →
      OccupationTimeBVP s U' (bandIndicator a ws) a b (D s) (DD s) (DDD s))
    (htau : ∀ s : ℝ, 0 < s → ContinuousOn (tau s) (Set.Icc a b))
    (hD : ∀ s : ℝ, 0 < s → ContinuousOn (D s) (Set.Icc a b))
    (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun s => s * Real.log (tau s x / D s x)) (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  refine (tendsto_mul_log_meanExitTime_div_dwellTime hU hax hawb hwbws hwsb hxws hbarrier
    hmax hmin).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hspos : (0 : ℝ) < s := hs
  have h1 := eq_meanExitTime_of_occupationTimeBVP (U := U) hspos hab hU hU' (htau s hspos)
    (hDynkinTau s hspos)
  have h2 := eq_of_occupationTimeBVP (c := ws) hspos hab hU hU'
    (fun y hy hyws => continuousAt_bandIndicator hy hyws) (hD s hspos)
    (continuousOn_dwellTime hab.le hU) (hDynkinD s hspos)
    (dwellTime_satisfies_bvp hspos hab hU hU')
  rw [h1 x ⟨hax, hxb⟩, h2 x ⟨hax, hxb⟩]

/-- **`cor:dwell`, third clause.** Given Dynkin's formula for the additively coupled family, the
lift's dwell is strictly below the direct one for every sufficiently small shared variance. -/
theorem eventually_mean_dwell_additive_lift_lt_direct {U U' : ℝ → ℝ} {a b x wb ws jac2 : ℝ}
    (Dd DDd DDDd Dl DDl DDDl : ℝ → ℝ → ℝ)
    (hDynkinD : ∀ s : ℝ, 0 < s →
      OccupationTimeBVP s U' (bandIndicator a ws) a b (Dd s) (DDd s) (DDDd s))
    (hDynkinL : ∀ s : ℝ, 0 < s →
      OccupationTimeBVP (s + jac2) U' (bandIndicator a ws) a b (Dl s) (DDl s) (DDDl s))
    (hDd : ∀ s : ℝ, 0 < s → ContinuousOn (Dd s) (Set.Icc a b))
    (hDl : ∀ s : ℝ, 0 < s → ContinuousOn (Dl s) (Set.Icc a b))
    (hjac : 0 < jac2) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ), Dl sd x < Dd sd x := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  filter_upwards [eventually_dwellTime_additive_lift_lt_direct hjac hU hax hawb hwbws hwsb hxws
      hbarrier hmax hmin, self_mem_nhdsWithin] with sd h1 h2
  have hsdpos : (0 : ℝ) < sd := h2
  have hslpos : (0 : ℝ) < sd + jac2 := by linarith
  have hd := eq_of_occupationTimeBVP (c := ws) hsdpos hab hU hU'
    (fun y hy hyws => continuousAt_bandIndicator hy hyws) (hDd sd hsdpos)
    (continuousOn_dwellTime hab.le hU) (hDynkinD sd hsdpos)
    (dwellTime_satisfies_bvp hsdpos hab hU hU')
  have hl := eq_of_occupationTimeBVP (c := ws) hslpos hab hU hU'
    (fun y hy hyws => continuousAt_bandIndicator hy hyws) (hDl sd hsdpos)
    (continuousOn_dwellTime hab.le hU) (hDynkinL sd hsdpos)
    (dwellTime_satisfies_bvp hslpos hab hU hU')
  rw [hd x ⟨hax, hxb⟩, hl x ⟨hax, hxb⟩]
  exact h1

/-- **`cor:dwell`, third clause, given Dynkin's formula: the paper-facing form.** The mean
occupation times themselves rather than the closed forms, at one shared step size and two fixed
effective variances `sd < sl`. Dynkin's formula at the band indicator is the only named
hypothesis, and uniqueness (`eq_of_occupationTimeBVP`) converts it into the closed form. This
mirrors `KramersBridge.eventually_mean_first_passage_lift_lt_direct_smallLearningRate`, which is
the same statement for the exit time. -/
theorem eventually_mean_dwell_lift_lt_direct_smallLearningRate {U U' : ℝ → ℝ}
    {a b x wb ws sd sl : ℝ}
    (Dd DDd DDDd Dl DDl DDDl : ℝ → ℝ → ℝ)
    (hDynkinD : ∀ eta : ℝ, 0 < eta →
      OccupationTimeBVP (eta * sd) U' (bandIndicator a ws) a b (Dd eta) (DDd eta) (DDDd eta))
    (hDynkinL : ∀ eta : ℝ, 0 < eta →
      OccupationTimeBVP (eta * sl) U' (bandIndicator a ws) a b (Dl eta) (DDl eta) (DDDl eta))
    (hDd : ∀ eta : ℝ, 0 < eta → ContinuousOn (Dd eta) (Set.Icc a b))
    (hDl : ∀ eta : ℝ, 0 < eta → ContinuousOn (Dl eta) (Set.Icc a b))
    (hsd : 0 < sd) (hsdl : sd < sl)
    (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ), Dl eta x < Dd eta x := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hsl : 0 < sl := hsd.trans hsdl
  filter_upwards [eventually_dwellTime_lift_lt_direct_smallLearningRate hsd hsdl hU hax hawb
      hwbws hwsb hxws hbarrier hmax hmin, self_mem_nhdsWithin] with eta h1 h2
  have hetapos : (0 : ℝ) < eta := h2
  have hd := eq_of_occupationTimeBVP (c := ws) (mul_pos hetapos hsd) hab hU hU'
    (fun y hy hyws => continuousAt_bandIndicator hy hyws) (hDd eta hetapos)
    (continuousOn_dwellTime hab.le hU) (hDynkinD eta hetapos)
    (dwellTime_satisfies_bvp (mul_pos hetapos hsd) hab hU hU')
  have hl := eq_of_occupationTimeBVP (c := ws) (mul_pos hetapos hsl) hab hU hU'
    (fun y hy hyws => continuousAt_bandIndicator hy hyws) (hDl eta hetapos)
    (continuousOn_dwellTime hab.le hU) (hDynkinL eta hetapos)
    (dwellTime_satisfies_bvp (mul_pos hetapos hsl) hab hU hU')
  rw [hd x ⟨hax, hxb⟩, hl x ⟨hax, hxb⟩]
  exact h1

/-! ### Non-vacuity

The cubic witness that discharges the exit-time stack in `KramersExitTime` discharges this one,
on the band `[a, w_s] = [-3/2, 1]`: the hypothesis stack is inhabited, and the two paper-facing
conclusions run on it, the first to a nonzero limit. -/

/-- **The hypothesis stack of `mean_dwell_kramers_exponent_of_dynkin` is satisfiable**, on the
cubic barrier `U(y) = y - y³/3` over `[-3/2, 3/2]` with `w_b = -1`, `w_s = 1`, `x = -1`, and the
dwell closed form as the Dynkin family. -/
theorem dwell_hypotheses_satisfiable :
    ∃ (U U' : ℝ → ℝ) (a b x wb ws : ℝ) (D DD DDD : ℝ → ℝ → ℝ),
      (∀ s : ℝ, 0 < s →
        OccupationTimeBVP s U' (bandIndicator a ws) a b (D s) (DD s) (DDD s)) ∧
      (∀ s : ℝ, 0 < s → ContinuousOn (D s) (Set.Icc a b)) ∧
      ContinuousOn U (Set.Icc a b) ∧
      (∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) ∧
      a ≤ x ∧ a < wb ∧ wb < ws ∧ ws < b ∧ x < ws ∧
      U wb < U ws ∧
      (∀ y ∈ Set.Icc a b, U y ≤ U ws) ∧ (∀ z ∈ Set.Icc a b, U wb ≤ U z) := by
  have hab : (-(3 / 2) : ℝ) < 3 / 2 := by norm_num
  exact ⟨cubicBarrier, cubicBarrierDeriv, -(3 / 2), 3 / 2, -1, -1, 1,
    fun s => dwellTime s cubicBarrier (-(3 / 2)) 1 (3 / 2),
    fun s => dwellTimeDeriv s cubicBarrier (-(3 / 2)) 1,
    fun s => dwellTimeDeriv2 s cubicBarrier cubicBarrierDeriv (-(3 / 2)) 1,
    fun _ hs => dwellTime_satisfies_bvp hs hab continuousOn_cubicBarrier hasDerivAt_cubicBarrier,
    fun _ _ => continuousOn_dwellTime hab.le continuousOn_cubicBarrier,
    continuousOn_cubicBarrier, hasDerivAt_cubicBarrier,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    cubicBarrier_pos, cubicBarrier_le_top, cubicBarrier_bottom_le⟩

/-- **The Kramers exponent of the dwell, evaluated on the witness**: `σ² log D(x) → 8/3 = 2α`,
the same exponent as the exit time's (`KramersExitTime.cubicBarrier_kramers_exponent`), which is
the corollary's "the dwell carries the exponent of `thm:main`(iii)" on an instance. -/
theorem cubicBarrier_dwell_kramers_exponent :
    Filter.Tendsto
      (fun s => s * Real.log (dwellTime s cubicBarrier (-(3 / 2)) 1 (3 / 2) (-1)))
      (𝓝[>] (0 : ℝ)) (𝓝 (8 / 3)) := by
  have h := tendsto_mul_log_dwellTime (U := cubicBarrier) (a := -(3 / 2)) (b := 3 / 2)
    (x := -1) (wb := -1) (ws := 1) continuousOn_cubicBarrier (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) cubicBarrier_pos cubicBarrier_le_top
    cubicBarrier_bottom_le
  have hval : (2 : ℝ) * (cubicBarrier 1 - cubicBarrier (-1)) = 8 / 3 := by
    rw [cubicBarrier_action]; norm_num
  rwa [hval] at h

/-- **The ratio clause, evaluated on the witness**: `σ² log (T(x)/D(x)) → 0`. -/
theorem cubicBarrier_dwell_ratio :
    Filter.Tendsto
      (fun s => s * Real.log (meanExitTime s cubicBarrier (-(3 / 2)) (3 / 2) (-1)
        / dwellTime s cubicBarrier (-(3 / 2)) 1 (3 / 2) (-1)))
      (𝓝[>] (0 : ℝ)) (𝓝 0) :=
  tendsto_mul_log_meanExitTime_div_dwellTime (U := cubicBarrier) (a := -(3 / 2)) (b := 3 / 2)
    (x := -1) (wb := -1) (ws := 1) continuousOn_cubicBarrier (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) cubicBarrier_pos cubicBarrier_le_top
    cubicBarrier_bottom_le

/-- **The shared-step-size ordering, evaluated on the witness.** For every pair of fixed
effective variances `sd < sl` the cubic-barrier dwell at `eta * sl` is strictly below the one at
`eta * sd` for all sufficiently small `eta`, so the corollary's third clause is not vacuous on
an instance. -/
theorem cubicBarrier_dwell_lift_lt_direct_smallLearningRate {sd sl : ℝ} (hsd : 0 < sd)
    (hsdl : sd < sl) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ),
      dwellTime (eta * sl) cubicBarrier (-(3 / 2)) 1 (3 / 2) (-1)
        < dwellTime (eta * sd) cubicBarrier (-(3 / 2)) 1 (3 / 2) (-1) :=
  eventually_dwellTime_lift_lt_direct_smallLearningRate (U := cubicBarrier) (a := -(3 / 2))
    (b := 3 / 2) (x := -1) (wb := -1) (ws := 1) hsd hsdl continuousOn_cubicBarrier (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) cubicBarrier_pos
    cubicBarrier_le_top cubicBarrier_bottom_le

end IcnnLift
