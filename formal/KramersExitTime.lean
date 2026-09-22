import Mathlib

/-!
# The one-dimensional mean-first-passage boundary-value problem behind Corollary 1 (`cor:fpt`)

Corollary 1 of the paper compares the mean first-passage time of the bias-channel projection
`dw = -L̃'(w) dt + σ_eff dB` across the softplus shoulder for two effective diffusions, and it
does so by citing Kramers' law as a black box: `E[τ] ≍ exp(2α/σ_eff²)` with `α` the barrier
action. This file supplies the analytic content of that citation, as far as it can honestly be
supplied inside this mathlib.

**What is unavailable, and what is done instead.** This mathlib (v4.31.0) contains no Itô
integral, no stochastic differential equation, no diffusion generator, no Fokker--Planck
equation and no exit time of a continuous-time process; a whole-tree search for `diffusion`,
`Fokker`, `Kramers`, `first passage`, `exit time`, `scale function` and `speed measure` returns
nothing. The object `E[τ]` therefore cannot even be *stated* here. What can be stated, and is,
is the classical boundary-value problem that `E[τ]` satisfies. For the diffusion
`dX = -U'(X) dt + σ dB` on `(a, b)` with a reflecting boundary at `a` and an absorbing boundary
at `b`, the mean exit time `T(x)` from the start point `x` solves

    (σ²/2) T''(x) - U'(x) T'(x) = -1   on (a, b),      T(b) = 0,      T'(a⁺) = 0.

`MeanExitTimeBVP` is that problem, written out. The passage from the stochastic differential
equation to this ordinary differential equation is **Dynkin's formula** applied to the
generator `A = (σ²/2) ∂² - U' ∂`, and it is exactly the step this mathlib cannot express.
Everything in this file is therefore a theorem about the boundary-value problem, not about a
stochastic process. That is the honest reading, and it is also precisely the analytic input
that the paper's Corollary 1 needs and currently leaves implicit: the corollary's proof cites
"the Kramers (Eyring--Arrhenius) escape law" without producing the boundary-value problem, its
solution, or the estimate that turns the solution into an exponential.

**What is proved.** The classical closed form

    T(x) = (2/σ²) ∫ₓᵇ exp(2U(y)/σ²) (∫ₐʸ exp(-2U(z)/σ²) dz) dy

is verified to solve the boundary-value problem, by differentiating the double integral twice
through the fundamental theorem of calculus; and the boundary-value problem is shown to have no
other continuous solution, so that on `[a, b]` a function is the closed form exactly when it
solves the problem. From the closed form the Arrhenius estimates are then proved outright, by
elementary means: a lower bound `c(δ) · exp(2(α - δ)/σ²) ≤ T(x)` for every `δ > 0`, obtained by
restricting the outer integral to a window around the barrier top and the inner integral to a
window around the basin bottom; and, when the barrier top and the basin bottom are the extreme
values of `U` on the interval — the single-barrier structure of the paper's assumption (A5) — a
matching upper bound `T(x) ≤ C · exp(2α/σ²)`. The upper constant `C` is explicit throughout; the
lower constant `c(δ)` is explicit once the two windows are named
(`meanExitTime_window_lower_bound`), and is existentially quantified when the windows are
instead manufactured from continuity of `U` alone
(`meanExitTime_arrhenius_lower_bound_uniform`), since continuity supplies no modulus. Together
these pin `T(x)` to `exp(2α/σ²)` at the level of the leading exponential, and the
small-noise limit closes them at that level into a genuine limit, `σ² · log T(x) → 2α` as
`σ² → 0⁺` (`tendsto_mul_log_meanExitTime`) — which is the paper's `≍` in the sense the paper
itself declares for it, equality of the leading exponential factor up to a sub-exponential
prefactor. The bounds then yield Corollary 1's ordering in four forms: for a prescribed pair of
diffusions with the two Laplace windows named, under a gap inequality every term of which is
data supplied by the caller (`meanExitTime_lift_lt_direct_of_window`); for a prescribed pair
with the windows left to continuity, under the same gap inequality but now with an
existentially quantified lower constant (`meanExitTime_lift_lt_direct`); for every sufficiently
small direct diffusion against a fixed lifted one, with no gap hypothesis at all; and for the
paper's own additively coupled pair `σ²` versus `σ² + σ_Jac²`, again with no gap hypothesis —
the regime (A5) postulates. A concrete cubic-potential instance discharges the hypothesis stack
(`paper_facing_hypotheses_satisfiable`) and then carries the two paper-facing conclusions
through on it, the Kramers exponent `σ² · log T(x) → 8/3` (`cubicBarrier_kramers_exponent`) and
the additively coupled ordering (`cubicBarrier_additive_lift_lt_direct`), so none of this is
vacuous. Throughout, the diffusion coefficient is a constant `σ²`: that is the reduction the
paper's own proof performs before invoking Kramers, when it treats the effective variance as
constant across the barrier band under (A2). It is an idealization of the paper's own
state-dependent `σ_eff(w̃)`, it is built into the definitions here rather than carried as a
hypothesis, and it is listed as such in the `## Honesty` section.

**What is not proved.** Three things, named again in the `## Honesty` section. First, Dynkin's
formula, the bridge from the stochastic differential equation to the boundary-value problem;
this is the single named hypothesis of the paper-facing theorems, and because uniqueness
*is* proved, that hypothesis asks for nothing beyond what Dynkin's formula delivers. Second, the
sharp Kramers prefactor `2π/√(U''(w_b)|U''(w_s)|)`, which this argument does not attempt: the
constants `c(δ)` and `C` here are crude — explicit in the window form, and in every form
degenerate as `δ → 0` for `c(δ)` — exactly as a Laplace estimate carried out without a
second-order expansion at the two extrema must be. The
consequence for the comparison is a loss of sharpness, not of content: the ordering of the two
exit times is proved outright for every sufficiently small direct diffusion — decoupled
(`eventually_meanExitTime_lift_lt_direct`) and for the paper's own coupled family
(`eventually_meanExitTime_additive_lift_lt_direct`), neither carrying a gap hypothesis — and
for a *prescribed* pair of diffusions it is proved under a gap inequality between the two
bounds, checkable term by term in `meanExitTime_lift_lt_direct_of_window`. What is not proved is
that `σ_direct² < σ_lift²` alone suffices at a fixed pair, nor an explicit threshold for
"sufficiently small"; both are statements about the sub-exponential prefactors, and the sharp
Kramers prefactor is what would supply them. Third, the state-dependence of the diffusion
coefficient: the paper's bias-channel SDE carries `σ_eff(w̃)`, and its proof replaces this by a
constant across the barrier band "to leading order" before invoking Kramers. This file makes
that replacement exactly rather than approximately — `σ²` is a constant in every definition and
every theorem — so the leading-order step of the paper is inherited, not closed. Closing it
would mean rerunning the scale-and-speed construction with `σ²(x)` in place of `σ²`, which
changes the closed form and is not attempted here.

## Results
* `MeanExitTimeBVP` — the Dynkin boundary-value problem for the mean exit time.
* `meanExitTime` — the classical double-integral solution, with `meanExitTimeDeriv` and
  `meanExitTimeDeriv2` its first and second derivatives.
* `hasDerivAt_speedIntegral`, `hasDerivAt_meanExitTime`, `hasDerivAt_meanExitTimeDeriv` — the
  two differentiations of the double integral.
* `meanExitTime_satisfies_bvp` — the closed form solves the boundary-value problem.
* `eq_meanExitTime_of_bvp` — and it is the only continuous solution.
* `continuousOn_meanExitTime` — the closed form is continuous up to the closed interval.
* `meanExitTime_window_lower_bound` — the Arrhenius lower bound with explicit windows.
* `meanExitTime_arrhenius_lower_bound_uniform` — the same bound with a window constant valid for
  every diffusion at once; `meanExitTime_arrhenius_lower_bound` is its `∃ c > 0` form.
* `meanExitTime_upper_bound` — the matching upper bound under the single-barrier structure.
* `tendsto_mul_log_meanExitTime` — the Kramers exponent as a genuine limit:
  `σ² · log T(x) → 2α` as `σ² → 0⁺`, with `tendsto_mul_log_const_div` the sub-exponentiality
  of the two crude prefactors.
* `exp_barrier_strictAnti` — the exponent comparison `exp(2α/σ₂²) < exp(2α/σ₁²)`.
* `meanExitTime_lt_of_gap`, `meanExitTime_lift_lt_direct` — the genuine ordering of the two exit
  times, under a gap condition whose lower-bound constant is existentially quantified.
* `meanExitTime_lift_lt_direct_of_window` — the same ordering with the existential removed:
  every term of the gap inequality is data supplied by the caller.
* `eventually_lower_bound_gt` — the gap condition holds for all small enough diffusion.
* `eventually_meanExitTime_lift_lt_direct` — hence the ordering holds outright for all
  sufficiently small direct diffusion, with no gap hypothesis.
* `eventually_meanExitTime_additive_lift_lt_direct` — the same, for the paper's own coupled
  pair `σ²` vs `σ² + σ_Jac²`.
* `mean_first_passage_arrhenius_of_dynkin`, `mean_first_passage_ordering_of_dynkin`,
  `eventually_mean_first_passage_lift_lt_direct`,
  `eventually_mean_first_passage_additive_lift_lt_direct`,
  `mean_first_passage_kramers_exponent_of_dynkin` — the paper-facing statements, carrying
  Dynkin's formula as a single named hypothesis.
* `cubicBarrier` — the witness potential `U(y) = y - y³/3` on `[-3/2, 3/2]`, with barrier action
  `α = 4/3`, together with its continuity, differentiability and single-barrier lemmas.
* `paper_facing_hypotheses_satisfiable` — that potential and the closed-form family satisfy the
  hypothesis stack of `mean_first_passage_kramers_exponent_of_dynkin`.
* `cubicBarrier_kramers_exponent`, `cubicBarrier_additive_lift_lt_direct` — the two paper-facing
  conclusions run end to end on that instance: `σ² · log T(x) → 8/3 = 2α`, and, for every fixed
  `σ_Jac² > 0`, the additively coupled ordering for all sufficiently small shared variance.
  Together with the previous item this is what makes the non-vacuity claim machine-checked at
  the conclusions rather than asserted about the hypotheses.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. The hypotheses carried, and what it would take to discharge
them:

* `hDynkin : MeanExitTimeBVP sigma2 U' a b tau Dtau DDtau` in the paper-facing theorems,
  together with `ContinuousOn tau (Set.Icc a b)`. This is Dynkin's formula for the generator
  `A = (σ²/2) ∂² - U' ∂` of `dX = -U'(X) dt + σ dB` applied to the exit time, and nothing more:
  uniqueness for the boundary-value problem is proved here (`eq_meanExitTime_of_bvp`), so the
  hypothesis does not have to name the closed form. Discharging it needs an Itô integral, a
  diffusion generator and an optional-stopping argument, none of which exist in this mathlib;
  it is stated as a hypothesis rather than an axiom so that the gap is visible at every use
  site.
* `hU : ContinuousOn U (Set.Icc a b)` and `hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y`.
  These are the paper's (A5) regularity, weakened: (A5) asks for a `C²` single-barrier
  potential, and only `C¹` on the closed interval is used here.
* `hmax`, `hmin` in the upper bound: the barrier top `w_s` maximizes and the basin bottom `w_b`
  minimizes `U` on `[a, b]`. This is the single-barrier structure of (A5) written out; without
  it the upper bound still holds with `α` replaced by the true oscillation of `U`, which is
  larger, and the two-sided pinning is lost.
* `hgap` in `meanExitTime_lt_of_gap`, `meanExitTime_lift_lt_direct`,
  `meanExitTime_lift_lt_direct_of_window` and `mean_first_passage_ordering_of_dynkin`: the
  proved upper bound at the lifted diffusion falls below the proved lower bound at the direct
  one. This is the small-noise clause of (A5) made quantitative. It is a genuine regime
  condition and not a disguised assumption of the conclusion: `eventually_lower_bound_gt` proves
  it is satisfied for all sufficiently small `σ²` whatever the positive lower constant, and the
  two `eventually` theorems remove it entirely in that regime. Read the two prescribed-pair
  forms with their constants in mind. In `meanExitTime_lift_lt_direct` the lower constant is
  existentially quantified, so the gap is not an inequality a reader can evaluate; what that
  theorem delivers is the constant together with the implication, and the implication itself is
  transitivity of the two bounds. `meanExitTime_lift_lt_direct_of_window` is the form in which
  every term is data, and it is the one to instantiate when a number is wanted.
* Constant diffusion coefficient. `σ²` is a real number, not a function, in every definition and
  theorem of this file. The paper's own bias-channel SDE has `σ_eff(w̃)` and its proof of
  Corollary 1 replaces that by a constant across the barrier band, "to leading order", under
  (A2); this file inherits that replacement rather than closing it. It is not carried as a
  hypothesis because it is built into the definitions `scaleDensity`, `speedDensity` and
  `meanExitTime`, so it cannot be seen at a use site the way `hDynkin` can. Discharging it means
  redoing the scale-and-speed construction with a state-dependent `σ²(x)`, which changes the
  closed form; it is not attempted here.

The relation between what is proved and the paper's `≍` should be read plainly. The paper
writes `E[τ] ≍ exp(2α/σ_eff²)` and declares `≍` to mean equality of the leading exponential
factor up to a sub-exponential prefactor. At a *fixed* diffusion what is proved is the
two-sided bound `c(δ) exp(2(α - δ)/σ²) ≤ T(x) ≤ C exp(2α/σ²)`, with `C` explicit and `c(δ)`
explicit once the two Laplace windows are named; `c(δ) → 0` as `δ → 0`, so at fixed `σ²` the two
bounds do not close up. In the small-noise
limit they do close at the level of exponents: `σ² · log T(x) → 2α`
(`tendsto_mul_log_meanExitTime`; `mean_first_passage_kramers_exponent_of_dynkin` for the
abstract mean first-passage time), which is the paper's `≍` in its declared sense. What
remains unproved is any control of the sub-exponential prefactor itself — the sharp Kramers
constant `2π/√(U''(w_b)|U''(w_s)|)` — and with it two sharper statements: the ordering of the
two exit times from `σ_direct² < σ_lift²` alone at a fixed pair, and an explicit threshold in
place of "for all sufficiently small" in the `eventually` theorems. Both would need the
second-order Laplace expansion at the barrier top and the basin bottom, which is not
attempted here.
-/

open MeasureTheory intervalIntegral Set Filter
open scoped Topology

namespace IcnnLift

/-! ### The boundary-value problem -/

/-- The **Dynkin boundary-value problem** for the mean exit time of the one-dimensional
diffusion `dX = -U'(X) dt + σ dB` from the interval `(a, b)`, with a reflecting boundary at `a`
and an absorbing boundary at `b`.

`T` is the mean exit time as a function of the starting point, `DT` and `DDT` its first and
second derivatives, supplied as data so that no `deriv`-of-a-junk-value ever appears; the first
two clauses pin them to be the derivatives they are named for, so supplying them as data costs
nothing. The remaining clauses are the interior equation `(σ²/2) T'' - U' T' = -1`, the
absorbing condition `T(b) = 0`, and the reflecting condition `T'(a⁺) = 0`, the last stated as a
one-sided limit because the equation itself only holds in the interior.

The passage from the stochastic differential equation to this problem is Dynkin's formula for
the generator `A = (σ²/2) ∂² - U' ∂`; this mathlib has no Itô calculus, so that passage is not
formalized anywhere in this file. -/
structure MeanExitTimeBVP (sigma2 : ℝ) (U' : ℝ → ℝ) (a b : ℝ) (T DT DDT : ℝ → ℝ) : Prop where
  /-- `DT` is the derivative of `T` on the open interval. -/
  hasDerivAt_fst : ∀ x ∈ Set.Ioo a b, HasDerivAt T (DT x) x
  /-- `DDT` is the derivative of `DT` on the open interval. -/
  hasDerivAt_snd : ∀ x ∈ Set.Ioo a b, HasDerivAt DT (DDT x) x
  /-- The interior equation `(σ²/2) T''(x) - U'(x) T'(x) = -1`. -/
  generator : ∀ x ∈ Set.Ioo a b, sigma2 / 2 * DDT x - U' x * DT x = -1
  /-- Absorbing boundary at `b`. -/
  absorbing : T b = 0
  /-- Reflecting boundary at `a`, as the one-sided limit `T'(a⁺) = 0`. -/
  reflecting : Filter.Tendsto DT (𝓝[>] a) (𝓝 0)

/-! ### The scale and speed densities, and the closed-form solution -/

/-- The **scale density** `exp(2U(y)/σ²)` of the diffusion `dX = -U'(X) dt + σ dB`: the
derivative of the classical scale function. -/
noncomputable def scaleDensity (sigma2 : ℝ) (U : ℝ → ℝ) (y : ℝ) : ℝ :=
  Real.exp (2 * U y / sigma2)

/-- The **speed density** `exp(-2U(z)/σ²)`, the unnormalized Gibbs weight of the potential `U`
at temperature `σ²/2`. -/
noncomputable def speedDensity (sigma2 : ℝ) (U : ℝ → ℝ) (z : ℝ) : ℝ :=
  Real.exp (-(2 * U z / sigma2))

/-- The inner integral `∫ₐʸ exp(-2U(z)/σ²) dz` of the closed form: the speed measure of
`[a, y]`, up to the constant `2/σ²`. -/
noncomputable def speedIntegral (sigma2 : ℝ) (U : ℝ → ℝ) (a y : ℝ) : ℝ :=
  ∫ z in a..y, speedDensity sigma2 U z

/-- The integrand of the outer integral, `exp(2U(y)/σ²) ∫ₐʸ exp(-2U(z)/σ²) dz`. -/
noncomputable def exitIntegrand (sigma2 : ℝ) (U : ℝ → ℝ) (a y : ℝ) : ℝ :=
  scaleDensity sigma2 U y * speedIntegral sigma2 U a y

/-- The **classical closed-form mean exit time**

`T(x) = (2/σ²) ∫ₓᵇ exp(2U(y)/σ²) (∫ₐʸ exp(-2U(z)/σ²) dz) dy`

for the diffusion `dX = -U'(X) dt + σ dB` on `(a, b)`, reflecting at `a` and absorbing at `b`.
`meanExitTime_satisfies_bvp` verifies that it solves `MeanExitTimeBVP`. -/
noncomputable def meanExitTime (sigma2 : ℝ) (U : ℝ → ℝ) (a b x : ℝ) : ℝ :=
  2 / sigma2 * ∫ y in x..b, exitIntegrand sigma2 U a y

/-- The first derivative of `meanExitTime`, namely `-(2/σ²) exp(2U(x)/σ²) ∫ₐˣ exp(-2U/σ²)`. -/
noncomputable def meanExitTimeDeriv (sigma2 : ℝ) (U : ℝ → ℝ) (a x : ℝ) : ℝ :=
  -(2 / sigma2) * exitIntegrand sigma2 U a x

/-- The second derivative of `meanExitTime`, in the form the interior equation produces:
`T'' = (2U'/σ²) T' - 2/σ²`.

This is a definition, so the interior equation `(σ²/2) T'' - U' T' = -1` is algebra once it is
substituted; the mathematical content is entirely in `hasDerivAt_meanExitTimeDeriv`, which
proves that this closed formula really is the derivative of `meanExitTimeDeriv`. Nothing is true
by fiat: `MeanExitTimeBVP` pins its `DDT` to be the derivative of its `DT` before the equation
is ever read. -/
noncomputable def meanExitTimeDeriv2 (sigma2 : ℝ) (U U' : ℝ → ℝ) (a x : ℝ) : ℝ :=
  2 * U' x / sigma2 * meanExitTimeDeriv sigma2 U a x - 2 / sigma2

/-! ### Elementary positivity and continuity -/

theorem scaleDensity_pos (sigma2 : ℝ) (U : ℝ → ℝ) (y : ℝ) : 0 < scaleDensity sigma2 U y :=
  Real.exp_pos _

theorem speedDensity_pos (sigma2 : ℝ) (U : ℝ → ℝ) (z : ℝ) : 0 < speedDensity sigma2 U z :=
  Real.exp_pos _

/-- Scale and speed densities are reciprocal; this is the identity that makes the double
integral solve the equation. -/
theorem scaleDensity_mul_speedDensity (sigma2 : ℝ) (U : ℝ → ℝ) (y : ℝ) :
    scaleDensity sigma2 U y * speedDensity sigma2 U y = 1 := by
  rw [scaleDensity, speedDensity, ← Real.exp_add, add_neg_cancel, Real.exp_zero]

theorem continuousOn_scaleDensity {U : ℝ → ℝ} {s : Set ℝ} (sigma2 : ℝ)
    (hU : ContinuousOn U s) : ContinuousOn (scaleDensity sigma2 U) s := by
  unfold scaleDensity
  fun_prop

theorem continuousOn_speedDensity {U : ℝ → ℝ} {s : Set ℝ} (sigma2 : ℝ)
    (hU : ContinuousOn U s) : ContinuousOn (speedDensity sigma2 U) s := by
  unfold speedDensity
  fun_prop

/-- Every subinterval of `[a, b]` carries an interval-integrable speed density. -/
theorem intervalIntegrable_speedDensity {U : ℝ → ℝ} {a b u v : ℝ} (sigma2 : ℝ)
    (hU : ContinuousOn U (Set.Icc a b)) (hau : a ≤ u) (huv : u ≤ v) (hvb : v ≤ b) :
    IntervalIntegrable (speedDensity sigma2 U) volume u v := by
  refine ContinuousOn.intervalIntegrable ((continuousOn_speedDensity sigma2 hU).mono ?_)
  rw [Set.uIcc_of_le huv]
  exact Set.Icc_subset_Icc hau hvb

theorem speedIntegral_self (sigma2 : ℝ) (U : ℝ → ℝ) (a : ℝ) :
    speedIntegral sigma2 U a a = 0 := by
  simp [speedIntegral]

theorem speedIntegral_nonneg {a y : ℝ} (sigma2 : ℝ) (U : ℝ → ℝ) (hay : a ≤ y) :
    0 ≤ speedIntegral sigma2 U a y :=
  intervalIntegral.integral_nonneg hay fun _ _ => (speedDensity_pos sigma2 U _).le

theorem continuousOn_speedIntegral {U : ℝ → ℝ} {a b : ℝ} (sigma2 : ℝ) (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) :
    ContinuousOn (speedIntegral sigma2 U a) (Set.Icc a b) := by
  have hint : IntegrableOn (speedDensity sigma2 U) (Set.uIcc a b) volume := by
    rw [Set.uIcc_of_le hab]
    exact (continuousOn_speedDensity sigma2 hU).integrableOn_Icc
  have h := intervalIntegral.continuousOn_primitive_interval hint
  rwa [Set.uIcc_of_le hab] at h

theorem continuousOn_exitIntegrand {U : ℝ → ℝ} {a b : ℝ} (sigma2 : ℝ) (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) :
    ContinuousOn (exitIntegrand sigma2 U a) (Set.Icc a b) :=
  (continuousOn_scaleDensity sigma2 hU).mul (continuousOn_speedIntegral sigma2 hab hU)

theorem exitIntegrand_nonneg {a y : ℝ} (sigma2 : ℝ) (U : ℝ → ℝ) (hay : a ≤ y) :
    0 ≤ exitIntegrand sigma2 U a y :=
  mul_nonneg (scaleDensity_pos sigma2 U y).le (speedIntegral_nonneg sigma2 U hay)

/-! ### The two differentiations of the double integral -/

/-- Fundamental theorem of calculus, inner integral: `d/dy ∫ₐʸ exp(-2U/σ²) = exp(-2U(y)/σ²)`. -/
theorem hasDerivAt_speedIntegral {U : ℝ → ℝ} {a b x : ℝ} (sigma2 : ℝ)
    (hU : ContinuousOn U (Set.Icc a b)) (hx : x ∈ Set.Ioo a b) :
    HasDerivAt (speedIntegral sigma2 U a) (speedDensity sigma2 U x) x := by
  have hcont : ContinuousOn (speedDensity sigma2 U) (Set.Icc a b) :=
    continuousOn_speedDensity sigma2 hU
  refine intervalIntegral.integral_hasDerivAt_right
    (intervalIntegrable_speedDensity sigma2 hU le_rfl hx.1.le hx.2.le) ?_ ?_
  · exact ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo
      (hcont.mono Set.Ioo_subset_Icc_self) x hx
  · exact (hcont.mono Set.Ioo_subset_Icc_self).continuousAt (isOpen_Ioo.mem_nhds hx)

/-- Fundamental theorem of calculus, outer integral: the closed form is differentiable in the
starting point, with derivative `meanExitTimeDeriv`. -/
theorem hasDerivAt_meanExitTime {U : ℝ → ℝ} {a b x : ℝ} (sigma2 : ℝ) (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) (hx : x ∈ Set.Ioo a b) :
    HasDerivAt (meanExitTime sigma2 U a b) (meanExitTimeDeriv sigma2 U a x) x := by
  have hcont : ContinuousOn (exitIntegrand sigma2 U a) (Set.Icc a b) :=
    continuousOn_exitIntegrand sigma2 hab hU
  have hint : IntervalIntegrable (exitIntegrand sigma2 U a) volume x b := by
    refine ContinuousOn.intervalIntegrable (hcont.mono ?_)
    rw [Set.uIcc_of_le hx.2.le]
    exact Set.Icc_subset_Icc hx.1.le le_rfl
  have hbase : HasDerivAt (fun u => ∫ y in u..b, exitIntegrand sigma2 U a y)
      (-exitIntegrand sigma2 U a x) x := by
    refine intervalIntegral.integral_hasDerivAt_left hint ?_ ?_
    · exact ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo
        (hcont.mono Set.Ioo_subset_Icc_self) x hx
    · exact (hcont.mono Set.Ioo_subset_Icc_self).continuousAt (isOpen_Ioo.mem_nhds hx)
  have h2 : HasDerivAt (meanExitTime sigma2 U a b)
      (2 / sigma2 * -exitIntegrand sigma2 U a x) x := hbase.const_mul (2 / sigma2)
  refine h2.congr_deriv ?_
  simp only [meanExitTimeDeriv]
  ring

/-- The second differentiation. The reciprocity `scaleDensity · speedDensity = 1` is what turns
the product rule into the inhomogeneous term `-2/σ²` of the interior equation. -/
theorem hasDerivAt_meanExitTimeDeriv {U U' : ℝ → ℝ} {a b x : ℝ} (sigma2 : ℝ)
    (hU : ContinuousOn U (Set.Icc a b)) (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y)
    (hx : x ∈ Set.Ioo a b) :
    HasDerivAt (meanExitTimeDeriv sigma2 U a) (meanExitTimeDeriv2 sigma2 U U' a x) x := by
  have hscale : HasDerivAt (scaleDensity sigma2 U)
      (scaleDensity sigma2 U x * (2 * U' x / sigma2)) x := by
    have h := (((hU' x hx).const_mul (2 : ℝ)).div_const sigma2).exp
    exact h
  have hspeed : HasDerivAt (speedIntegral sigma2 U a) (speedDensity sigma2 U x) x :=
    hasDerivAt_speedIntegral sigma2 hU hx
  have hprod : HasDerivAt (exitIntegrand sigma2 U a)
      (scaleDensity sigma2 U x * (2 * U' x / sigma2) * speedIntegral sigma2 U a x
        + scaleDensity sigma2 U x * speedDensity sigma2 U x) x := hscale.mul hspeed
  have hone : scaleDensity sigma2 U x * speedDensity sigma2 U x = 1 :=
    scaleDensity_mul_speedDensity sigma2 U x
  rw [hone] at hprod
  have hfinal := hprod.const_mul (-(2 / sigma2))
  refine hfinal.congr_deriv ?_
  simp only [meanExitTimeDeriv2, meanExitTimeDeriv, exitIntegrand]
  ring

/-! ### The closed form solves the boundary-value problem -/

/-- The absorbing boundary condition: the outer integral is over a degenerate interval. -/
theorem meanExitTime_right (sigma2 : ℝ) (U : ℝ → ℝ) (a b : ℝ) :
    meanExitTime sigma2 U a b b = 0 := by
  simp [meanExitTime]

/-- The reflecting boundary condition, at the level of values: the inner integral vanishes at
the reflecting endpoint. -/
theorem meanExitTimeDeriv_left (sigma2 : ℝ) (U : ℝ → ℝ) (a : ℝ) :
    meanExitTimeDeriv sigma2 U a a = 0 := by
  simp [meanExitTimeDeriv, exitIntegrand, speedIntegral_self]

/-- The derivative of the closed form extends continuously to the closed interval. -/
theorem continuousOn_meanExitTimeDeriv {U : ℝ → ℝ} {a b : ℝ} (sigma2 : ℝ) (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) :
    ContinuousOn (meanExitTimeDeriv sigma2 U a) (Set.Icc a b) :=
  continuousOn_const.mul (continuousOn_exitIntegrand sigma2 hab hU)

/-- Membership of the closed interval in the right-hand neighbourhood filter at `a`, the small
fact both boundary arguments need. -/
theorem Icc_mem_nhdsGT_left {a b : ℝ} (hab : a < b) : Set.Icc a b ∈ 𝓝[>] a := by
  have h1 : Set.Ioi a ∈ 𝓝[>] a := self_mem_nhdsWithin
  have h2 : Set.Iio b ∈ 𝓝[>] a := nhdsWithin_le_nhds (Iio_mem_nhds hab)
  exact Filter.mem_of_superset (Filter.inter_mem h1 h2) fun y hy => ⟨hy.1.le, hy.2.le⟩

/-- The reflecting boundary condition as the genuine one-sided limit `T'(a⁺) = 0`. -/
theorem tendsto_meanExitTimeDeriv_right {U : ℝ → ℝ} {a b : ℝ} (sigma2 : ℝ) (hab : a < b)
    (hU : ContinuousOn U (Set.Icc a b)) :
    Filter.Tendsto (meanExitTimeDeriv sigma2 U a) (𝓝[>] a) (𝓝 0) := by
  have hcw : ContinuousWithinAt (meanExitTimeDeriv sigma2 U a) (Set.Icc a b) a :=
    continuousOn_meanExitTimeDeriv sigma2 hab.le hU a ⟨le_rfl, hab.le⟩
  have hle : 𝓝[>] a ≤ 𝓝[Set.Icc a b] a := nhdsWithin_le_iff.mpr (Icc_mem_nhdsGT_left hab)
  have := hcw.tendsto.mono_left hle
  rwa [meanExitTimeDeriv_left] at this

/-- **The closed form solves the Dynkin boundary-value problem.** Both differentiations are
genuine applications of the fundamental theorem of calculus, and the interior equation then
holds identically. -/
theorem meanExitTime_satisfies_bvp {U U' : ℝ → ℝ} {a b : ℝ} {sigma2 : ℝ} (hs : 0 < sigma2)
    (hab : a < b) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) :
    MeanExitTimeBVP sigma2 U' a b (meanExitTime sigma2 U a b) (meanExitTimeDeriv sigma2 U a)
      (meanExitTimeDeriv2 sigma2 U U' a) where
  hasDerivAt_fst x hx := hasDerivAt_meanExitTime sigma2 hab.le hU hx
  hasDerivAt_snd x hx := hasDerivAt_meanExitTimeDeriv sigma2 hU hU' hx
  generator x _ := by
    have hne : sigma2 ≠ 0 := ne_of_gt hs
    simp only [meanExitTimeDeriv2]
    field_simp
    ring
  absorbing := meanExitTime_right sigma2 U a b
  reflecting := tendsto_meanExitTimeDeriv_right sigma2 hab hU

/-! ### Uniqueness for the boundary-value problem -/

/-- The closed form is continuous up to the closed interval. -/
theorem continuousOn_meanExitTime {U : ℝ → ℝ} {a b : ℝ} (sigma2 : ℝ) (hab : a ≤ b)
    (hU : ContinuousOn U (Set.Icc a b)) :
    ContinuousOn (meanExitTime sigma2 U a b) (Set.Icc a b) := by
  have hint : IntegrableOn (exitIntegrand sigma2 U a) (Set.uIcc a b) volume := by
    rw [Set.uIcc_of_le hab]
    exact (continuousOn_exitIntegrand sigma2 hab hU).integrableOn_Icc
  have h := intervalIntegral.continuousOn_primitive_interval_left hint
  rw [Set.uIcc_of_le hab] at h
  show ContinuousOn (fun x => 2 / sigma2 * ∫ y in x..b, exitIntegrand sigma2 U a y) (Set.Icc a b)
  exact continuousOn_const.mul h

/-- **Uniqueness for the Dynkin boundary-value problem.** Any solution that is continuous up to
the closed interval coincides with the closed form.

The argument is the classical one. The Wronskian-type quantity
`(T' - V')(x) · exp(-2U(x)/σ²)` has vanishing derivative on `(a, b)`, because the two solutions
satisfy the same interior equation and the exponential is exactly the integrating factor; it is
therefore constant there. Its limit at the reflecting boundary is `0`, since both derivatives
tend to `0` there and the exponential is continuous, so the constant is `0`; as the exponential
never vanishes, `T' = V'` on `(a, b)`. The difference `T - V` then has vanishing derivative on
`(a, b)`, hence is constant on every `[x, b]` with `a < x`, hence equals its value at the
absorbing boundary, which is `0`. Continuity extends the conclusion to `a`.

This is what makes the identification of the closed form with a mean first-passage time depend
on nothing but Dynkin's formula: the boundary-value problem determines its solution. -/
theorem eq_meanExitTime_of_bvp {U U' : ℝ → ℝ} {a b : ℝ} {sigma2 : ℝ} {T DT DDT : ℝ → ℝ}
    (hs : 0 < sigma2) (hab : a < b) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y)
    (hTcont : ContinuousOn T (Set.Icc a b))
    (hT : MeanExitTimeBVP sigma2 U' a b T DT DDT) :
    ∀ x ∈ Set.Icc a b, T x = meanExitTime sigma2 U a b x := by
  have hne : sigma2 ≠ 0 := ne_of_gt hs
  have hV := meanExitTime_satisfies_bvp hs hab hU hU'
  -- the Wronskian-type quantity has vanishing derivative on the open interval
  have hgderiv : ∀ x ∈ Set.Ioo a b,
      HasDerivAt (fun y => (DT y - meanExitTimeDeriv sigma2 U a y) * speedDensity sigma2 U y)
        0 x := by
    intro x hx
    have h1 : HasDerivAt (fun y => DT y - meanExitTimeDeriv sigma2 U a y)
        (DDT x - meanExitTimeDeriv2 sigma2 U U' a x) x :=
      (hT.hasDerivAt_snd x hx).sub (hasDerivAt_meanExitTimeDeriv sigma2 hU hU' hx)
    have h2 : HasDerivAt (speedDensity sigma2 U)
        (speedDensity sigma2 U x * -(2 * U' x / sigma2)) x :=
      ((((hU' x hx).const_mul (2 : ℝ)).div_const sigma2).neg).exp
    have h3 : HasDerivAt
        (fun y => (DT y - meanExitTimeDeriv sigma2 U a y) * speedDensity sigma2 U y)
        ((DDT x - meanExitTimeDeriv2 sigma2 U U' a x) * speedDensity sigma2 U x
          + (DT x - meanExitTimeDeriv sigma2 U a x)
              * (speedDensity sigma2 U x * -(2 * U' x / sigma2))) x := h1.mul h2
    refine h3.congr_deriv ?_
    have e1 : sigma2 / 2 * DDT x - U' x * DT x = -1 := hT.generator x hx
    have e2 : sigma2 / 2 * meanExitTimeDeriv2 sigma2 U U' a x
        - U' x * meanExitTimeDeriv sigma2 U a x = -1 := hV.generator x hx
    have hnum : sigma2 * (DDT x - meanExitTimeDeriv2 sigma2 U U' a x)
        - 2 * (U' x * (DT x - meanExitTimeDeriv sigma2 U a x)) = 0 := by
      linear_combination 2 * e1 - 2 * e2
    have hrw : (DDT x - meanExitTimeDeriv2 sigma2 U U' a x) * speedDensity sigma2 U x
        + (DT x - meanExitTimeDeriv sigma2 U a x)
            * (speedDensity sigma2 U x * -(2 * U' x / sigma2))
        = speedDensity sigma2 U x
            * ((sigma2 * (DDT x - meanExitTimeDeriv2 sigma2 U U' a x)
                - 2 * (U' x * (DT x - meanExitTimeDeriv sigma2 U a x))) / sigma2) := by
      field_simp
      ring
    rw [hrw, hnum]
    simp
  -- hence it is constant on the open interval
  have hgconst : ∀ u ∈ Set.Ioo a b, ∀ v ∈ Set.Ioo a b, u ≤ v →
      (DT v - meanExitTimeDeriv sigma2 U a v) * speedDensity sigma2 U v
        = (DT u - meanExitTimeDeriv sigma2 U a u) * speedDensity sigma2 U u := by
    intro u hu v hv huv
    have hsub : Set.Icc u v ⊆ Set.Ioo a b := Set.Icc_subset_Ioo hu.1 hv.2
    have hcont : ContinuousOn
        (fun y => (DT y - meanExitTimeDeriv sigma2 U a y) * speedDensity sigma2 U y)
        (Set.Icc u v) := fun y hy => ((hgderiv y (hsub hy)).continuousAt).continuousWithinAt
    exact constant_of_has_deriv_right_zero hcont
      (fun y hy => (hgderiv y (hsub (Set.Ico_subset_Icc_self hy))).hasDerivWithinAt) v
      (Set.right_mem_Icc.mpr huv)
  -- and the constant is zero, by the reflecting boundary condition
  have hgzero : ∀ x ∈ Set.Ioo a b,
      (DT x - meanExitTimeDeriv sigma2 U a x) * speedDensity sigma2 U x = 0 := by
    intro x hx
    have hE : Filter.Tendsto (speedDensity sigma2 U) (𝓝[>] a)
        (𝓝 (speedDensity sigma2 U a)) :=
      ((continuousOn_speedDensity sigma2 hU) a ⟨le_rfl, hab.le⟩).tendsto.mono_left
        (nhdsWithin_le_iff.mpr (Icc_mem_nhdsGT_left hab))
    have hlim : Filter.Tendsto
        (fun y => (DT y - meanExitTimeDeriv sigma2 U a y) * speedDensity sigma2 U y)
        (𝓝[>] a) (𝓝 0) := by
      have h := (hT.reflecting.sub (tendsto_meanExitTimeDeriv_right sigma2 hab hU)).mul hE
      simpa using h
    have hev : (fun y => (DT y - meanExitTimeDeriv sigma2 U a y) * speedDensity sigma2 U y)
        =ᶠ[𝓝[>] a]
        fun _ => (DT x - meanExitTimeDeriv sigma2 U a x) * speedDensity sigma2 U x := by
      filter_upwards [Ioo_mem_nhdsGT hab] with y hy
      rcases le_total y x with h | h
      · exact (hgconst y hy x hx h).symm
      · exact hgconst x hx y hy h
    exact tendsto_nhds_unique tendsto_const_nhds (hlim.congr' hev)
  -- so the two derivatives agree on the open interval
  have hDTeq : ∀ x ∈ Set.Ioo a b, DT x = meanExitTimeDeriv sigma2 U a x := by
    intro x hx
    rcases mul_eq_zero.mp (hgzero x hx) with h | h
    · linarith
    · exact absurd h (ne_of_gt (speedDensity_pos sigma2 U x))
  -- and therefore the two solutions agree, by the absorbing boundary condition
  have hDcont : ContinuousOn (fun y => T y - meanExitTime sigma2 U a b y) (Set.Icc a b) :=
    hTcont.sub (continuousOn_meanExitTime sigma2 hab.le hU)
  have hdiff : ∀ x ∈ Set.Ioo a b,
      HasDerivAt (fun y => T y - meanExitTime sigma2 U a b y) 0 x := by
    intro x hx
    have h3 : HasDerivAt (fun y => T y - meanExitTime sigma2 U a b y)
        (DT x - meanExitTimeDeriv sigma2 U a x) x :=
      (hT.hasDerivAt_fst x hx).sub (hasDerivAt_meanExitTime sigma2 hab.le hU hx)
    rwa [hDTeq x hx, sub_self] at h3
  have hIoo : ∀ x ∈ Set.Ioo a b, T x = meanExitTime sigma2 U a b x := by
    intro x hx
    have hc := constant_of_has_deriv_right_zero (a := x) (b := b)
      (hDcont.mono (Set.Icc_subset_Icc hx.1.le le_rfl))
      (fun y hy => (hdiff y ⟨lt_of_lt_of_le hx.1 hy.1, hy.2⟩).hasDerivWithinAt)
    have hb := hc b (Set.right_mem_Icc.mpr hx.2.le)
    have hDb : T b - meanExitTime sigma2 U a b b = 0 := by
      rw [hT.absorbing, meanExitTime_right]; ring
    have hzero : T x - meanExitTime sigma2 U a b x = 0 := by rw [← hb]; exact hDb
    linarith
  intro x hx
  rcases eq_or_lt_of_le hx.1 with hxa | hxa
  · -- the reflecting endpoint, reached by continuity
    have hlim : Filter.Tendsto (fun y => T y - meanExitTime sigma2 U a b y) (𝓝[>] a)
        (𝓝 (T a - meanExitTime sigma2 U a b a)) :=
      (hDcont a ⟨le_rfl, hab.le⟩).tendsto.mono_left
        (nhdsWithin_le_iff.mpr (Icc_mem_nhdsGT_left hab))
    have hev : (fun y => T y - meanExitTime sigma2 U a b y) =ᶠ[𝓝[>] a] fun _ => (0 : ℝ) := by
      filter_upwards [Ioo_mem_nhdsGT hab] with y hy
      rw [hIoo y hy]; ring
    have h0 : T a - meanExitTime sigma2 U a b a = 0 :=
      tendsto_nhds_unique (hlim.congr' hev) tendsto_const_nhds
    rw [← hxa]
    linarith
  · rcases eq_or_lt_of_le hx.2 with hxb | hxb
    · rw [hxb, hT.absorbing, meanExitTime_right]
    · exact hIoo x ⟨hxa, hxb⟩

/-! ### Elementary interval-integral estimates -/

/-- Restricting a non-negative integrand to a subinterval only decreases the integral. -/
theorem intervalIntegral_le_of_subinterval {f : ℝ → ℝ} {p u v q : ℝ}
    (hpu : p ≤ u) (huv : u ≤ v) (hvq : v ≤ q) (hcont : ContinuousOn f (Set.Icc p q))
    (hpos : ∀ y ∈ Set.Icc p q, 0 ≤ f y) :
    (∫ y in u..v, f y) ≤ ∫ y in p..q, f y := by
  have huq : u ≤ q := huv.trans hvq
  have hpv : p ≤ v := hpu.trans huv
  have hsub : ∀ r s : ℝ, p ≤ r → r ≤ s → s ≤ q → IntervalIntegrable f volume r s := by
    intro r s hr hrs hsq
    refine ContinuousOn.intervalIntegrable (hcont.mono ?_)
    rw [Set.uIcc_of_le hrs]
    exact Set.Icc_subset_Icc hr hsq
  have e1 : (∫ y in p..u, f y) + ∫ y in u..q, f y = ∫ y in p..q, f y :=
    intervalIntegral.integral_add_adjacent_intervals (hsub p u le_rfl hpu huq)
      (hsub u q hpu huq le_rfl)
  have e2 : (∫ y in u..v, f y) + ∫ y in v..q, f y = ∫ y in u..q, f y :=
    intervalIntegral.integral_add_adjacent_intervals (hsub u v hpu huv hvq)
      (hsub v q hpv hvq le_rfl)
  have n1 : 0 ≤ ∫ y in p..u, f y :=
    intervalIntegral.integral_nonneg hpu fun y hy => hpos y ⟨hy.1, hy.2.trans huq⟩
  have n2 : 0 ≤ ∫ y in v..q, f y :=
    intervalIntegral.integral_nonneg hvq fun y hy => hpos y ⟨hpv.trans hy.1, hy.2⟩
  linarith

/-- A constant lower bound on an interval integrates to a lower bound on the integral. -/
theorem const_mul_le_intervalIntegral {f : ℝ → ℝ} {u v c : ℝ} (huv : u ≤ v)
    (hcont : ContinuousOn f (Set.Icc u v)) (hle : ∀ y ∈ Set.Icc u v, c ≤ f y) :
    (v - u) * c ≤ ∫ y in u..v, f y := by
  have hint : IntervalIntegrable f volume u v := by
    refine ContinuousOn.intervalIntegrable ?_
    rwa [Set.uIcc_of_le huv]
  have h := intervalIntegral.integral_mono_on huv intervalIntegral.intervalIntegrable_const hint hle
  simpa using h

/-- A constant upper bound on an interval integrates to an upper bound on the integral. -/
theorem intervalIntegral_le_const_mul {f : ℝ → ℝ} {u v c : ℝ} (huv : u ≤ v)
    (hcont : ContinuousOn f (Set.Icc u v)) (hle : ∀ y ∈ Set.Icc u v, f y ≤ c) :
    (∫ y in u..v, f y) ≤ (v - u) * c := by
  have hint : IntervalIntegrable f volume u v := by
    refine ContinuousOn.intervalIntegrable ?_
    rwa [Set.uIcc_of_le huv]
  have h := intervalIntegral.integral_mono_on huv hint intervalIntegral.intervalIntegrable_const hle
  simpa using h

/-- Monotonicity of `u ↦ 2u/σ²` for positive `σ²`, the step that turns a bound on the potential
into a bound on the Gibbs weight. -/
theorem two_mul_div_le_two_mul_div {u v sigma2 : ℝ} (hs : 0 < sigma2) (h : u ≤ v) :
    2 * u / sigma2 ≤ 2 * v / sigma2 := by
  have hc : (0 : ℝ) ≤ 2 / sigma2 := div_nonneg (by norm_num) hs.le
  calc 2 * u / sigma2 = 2 / sigma2 * u := by ring
    _ ≤ 2 / sigma2 * v := mul_le_mul_of_nonneg_left h hc
    _ = 2 * v / sigma2 := by ring

/-! ### The Arrhenius lower bound -/

/-- The inner integral, bounded below by its restriction to a window on which `U` does not
exceed `cb`. This is the basin-bottom half of the Laplace estimate. -/
theorem speedIntegral_window_lower_bound {U : ℝ → ℝ} {a b t₁ t₂ y cb : ℝ} {sigma2 : ℝ}
    (hs : 0 < sigma2) (hU : ContinuousOn U (Set.Icc a b))
    (hat₁ : a ≤ t₁) (ht : t₁ ≤ t₂) (hty : t₂ ≤ y) (hyb : y ≤ b)
    (hbasin : ∀ z ∈ Set.Icc t₁ t₂, U z ≤ cb) :
    (t₂ - t₁) * Real.exp (-(2 * cb / sigma2)) ≤ speedIntegral sigma2 U a y := by
  have hcont : ContinuousOn (speedDensity sigma2 U) (Set.Icc a y) :=
    (continuousOn_speedDensity sigma2 hU).mono (Set.Icc_subset_Icc le_rfl hyb)
  have hstep : (t₂ - t₁) * Real.exp (-(2 * cb / sigma2)) ≤ ∫ z in t₁..t₂, speedDensity sigma2 U z := by
    refine const_mul_le_intervalIntegral ht
      (hcont.mono (Set.Icc_subset_Icc hat₁ hty)) ?_
    intro z hz
    rw [speedDensity, Real.exp_le_exp]
    have h2 : 2 * U z / sigma2 ≤ 2 * cb / sigma2 := two_mul_div_le_two_mul_div hs (hbasin z hz)
    linarith
  refine le_trans hstep ?_
  exact intervalIntegral_le_of_subinterval hat₁ ht hty hcont
    (fun z _ => (speedDensity_pos sigma2 U z).le)

/-- **The Arrhenius lower bound, with explicit windows.** Restrict the outer integral to a
window `[s₁, s₂]` on which `U ≥ cs` (a neighbourhood of the barrier top) and the inner integral
to a window `[t₁, t₂]` on which `U ≤ cb` (a neighbourhood of the basin bottom); the double
integral is then at least the product of the two restricted pieces, and the exponentials
combine into `exp(2(cs - cb)/σ²)`. -/
theorem meanExitTime_window_lower_bound {U : ℝ → ℝ} {a b x s₁ s₂ t₁ t₂ cs cb : ℝ} {sigma2 : ℝ}
    (hs : 0 < sigma2) (hU : ContinuousOn U (Set.Icc a b))
    (hax : a ≤ x) (hxs₁ : x ≤ s₁) (hs₁₂ : s₁ ≤ s₂) (hs₂b : s₂ ≤ b)
    (hat₁ : a ≤ t₁) (ht₁₂ : t₁ ≤ t₂) (ht₂s₁ : t₂ ≤ s₁)
    (hbarrier : ∀ y ∈ Set.Icc s₁ s₂, cs ≤ U y)
    (hbasin : ∀ z ∈ Set.Icc t₁ t₂, U z ≤ cb) :
    2 / sigma2 * ((s₂ - s₁) * (t₂ - t₁)) * Real.exp (2 * (cs - cb) / sigma2)
      ≤ meanExitTime sigma2 U a b x := by
  have hxb : x ≤ b := hxs₁.trans (hs₁₂.trans hs₂b)
  have hab : a ≤ b := hax.trans hxb
  have hcont : ContinuousOn (exitIntegrand sigma2 U a) (Set.Icc x b) :=
    (continuousOn_exitIntegrand sigma2 hab hU).mono (Set.Icc_subset_Icc hax le_rfl)
  -- the pointwise bound on the outer integrand over the barrier window
  have hpt : ∀ y ∈ Set.Icc s₁ s₂,
      Real.exp (2 * cs / sigma2) * ((t₂ - t₁) * Real.exp (-(2 * cb / sigma2)))
        ≤ exitIntegrand sigma2 U a y := by
    intro y hy
    have hyb : y ≤ b := hy.2.trans hs₂b
    have hscale : Real.exp (2 * cs / sigma2) ≤ scaleDensity sigma2 U y := by
      rw [scaleDensity, Real.exp_le_exp]
      exact two_mul_div_le_two_mul_div hs (hbarrier y hy)
    have hspeed : (t₂ - t₁) * Real.exp (-(2 * cb / sigma2)) ≤ speedIntegral sigma2 U a y :=
      speedIntegral_window_lower_bound hs hU hat₁ ht₁₂ (ht₂s₁.trans hy.1) hyb hbasin
    have hnn : 0 ≤ (t₂ - t₁) * Real.exp (-(2 * cb / sigma2)) :=
      mul_nonneg (by linarith) (Real.exp_pos _).le
    exact mul_le_mul hscale hspeed hnn (scaleDensity_pos sigma2 U y).le
  -- integrate the pointwise bound over the barrier window
  have hwindow : (s₂ - s₁) *
      (Real.exp (2 * cs / sigma2) * ((t₂ - t₁) * Real.exp (-(2 * cb / sigma2))))
        ≤ ∫ y in s₁..s₂, exitIntegrand sigma2 U a y :=
    const_mul_le_intervalIntegral hs₁₂
      (hcont.mono (Set.Icc_subset_Icc hxs₁ hs₂b)) hpt
  -- enlarge the window back to [x, b]
  have houter : (∫ y in s₁..s₂, exitIntegrand sigma2 U a y)
      ≤ ∫ y in x..b, exitIntegrand sigma2 U a y :=
    intervalIntegral_le_of_subinterval hxs₁ hs₁₂ hs₂b hcont
      (fun y hy => exitIntegrand_nonneg sigma2 U (hax.trans hy.1))
  -- collect
  have hexp : Real.exp (2 * cs / sigma2) * Real.exp (-(2 * cb / sigma2))
      = Real.exp (2 * (cs - cb) / sigma2) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hfront : (0 : ℝ) < 2 / sigma2 := by positivity
  rw [meanExitTime]
  have hkey : (s₂ - s₁) * (t₂ - t₁) * Real.exp (2 * (cs - cb) / sigma2)
      ≤ ∫ y in x..b, exitIntegrand sigma2 U a y := by
    have heq : (s₂ - s₁) * (t₂ - t₁) * Real.exp (2 * (cs - cb) / sigma2)
        = (s₂ - s₁) *
          (Real.exp (2 * cs / sigma2) * ((t₂ - t₁) * Real.exp (-(2 * cb / sigma2)))) := by
      rw [← hexp]; ring
    rw [heq]
    exact hwindow.trans houter
  calc 2 / sigma2 * ((s₂ - s₁) * (t₂ - t₁)) * Real.exp (2 * (cs - cb) / sigma2)
      = 2 / sigma2 * ((s₂ - s₁) * (t₂ - t₁) * Real.exp (2 * (cs - cb) / sigma2)) := by ring
    _ ≤ 2 / sigma2 * ∫ y in x..b, exitIntegrand sigma2 U a y := by
        exact mul_le_mul_of_nonneg_left hkey hfront.le

/-- **The Arrhenius lower bound, uniformly in the diffusion.** Only continuity of `U` is used:
continuity at the barrier top `w_s` and at the basin bottom `w_b` produces windows on which `U`
is within `δ/2` of the corresponding extreme value, and `meanExitTime_window_lower_bound` then
converts those windows into the bound.

The point of this form is that the window constant `L = 4ρ²`, for the common window radius `ρ`,
depends on `U`, on the geometry of `a < w_b < w_s < b` and on `δ`, but **not on the diffusion**:
the same `L` serves every `σ² > 0`. That is what makes the small-noise limit meaningful, and it
is what `eventually_meanExitTime_lift_lt_direct` consumes. The constant does degenerate as
`δ → 0`, which is the price of a Laplace estimate carried out without a second-order expansion
at the two extrema. -/
theorem meanExitTime_arrhenius_lower_bound_uniform {U : ℝ → ℝ} {a b x wb ws delta : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta) :
    ∃ L > 0, ∀ sigma2 : ℝ, 0 < sigma2 →
      2 / sigma2 * L * Real.exp (2 * (U ws - U wb - delta) / sigma2)
        ≤ meanExitTime sigma2 U a b x := by
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
  refine ⟨4 * ρ ^ 2, mul_pos (by norm_num) (pow_pos hρpos 2), fun sigma2 hs => ?_⟩
  have hmain := meanExitTime_window_lower_bound (U := U) (a := a) (b := b) (x := x)
    (s₁ := ws - ρ) (s₂ := ws + ρ) (t₁ := wb - ρ) (t₂ := wb + ρ)
    (cs := U ws - delta / 2) (cb := U wb + delta / 2) (sigma2 := sigma2)
    hs hU hax (by linarith) (by linarith) (by linarith) (by linarith) (by linarith)
    (by linarith) hbarrier hbasin
  have he : 2 * (U ws - delta / 2 - (U wb + delta / 2)) / sigma2
      = 2 * (U ws - U wb - delta) / sigma2 := by ring
  have hcconst : 2 / sigma2 * ((ws + ρ - (ws - ρ)) * (wb + ρ - (wb - ρ)))
      = 2 / sigma2 * (4 * ρ ^ 2) := by ring
  rw [he, hcconst] at hmain
  exact hmain

/-- **The Arrhenius lower bound, existential form**, the statement Corollary 1 reads: for every
`δ > 0` there is a positive constant `c` with `c · exp(2(α - δ)/σ²) ≤ T(x)`, where
`α = U(w_s) - U(w_b)` is the barrier action. A smaller effective diffusion forces an
exponentially large exit time. -/
theorem meanExitTime_arrhenius_lower_bound {U : ℝ → ℝ} {a b x wb ws delta : ℝ} {sigma2 : ℝ}
    (hs : 0 < sigma2) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta) :
    ∃ c > 0, c * Real.exp (2 * (U ws - U wb - delta) / sigma2) ≤ meanExitTime sigma2 U a b x := by
  obtain ⟨L, hL, hbound⟩ :=
    meanExitTime_arrhenius_lower_bound_uniform hU hax hawb hwbws hwsb hxws hdelta
  exact ⟨2 / sigma2 * L, mul_pos (div_pos (by norm_num) hs) hL, hbound sigma2 hs⟩

/-! ### The matching upper bound, and the comparison of Corollary 1 -/

/-- The inner integral is bounded above by the length of the interval times the largest Gibbs
weight it carries. -/
theorem speedIntegral_le {U : ℝ → ℝ} {a b y m : ℝ} {sigma2 : ℝ} (hs : 0 < sigma2)
    (hU : ContinuousOn U (Set.Icc a b)) (hay : a ≤ y) (hyb : y ≤ b)
    (hmin : ∀ z ∈ Set.Icc a b, m ≤ U z) :
    speedIntegral sigma2 U a y ≤ (b - a) * Real.exp (-(2 * m / sigma2)) := by
  have h1 : speedIntegral sigma2 U a y ≤ (y - a) * Real.exp (-(2 * m / sigma2)) := by
    refine intervalIntegral_le_const_mul hay
      ((continuousOn_speedDensity sigma2 hU).mono (Set.Icc_subset_Icc le_rfl hyb)) ?_
    intro z hz
    rw [speedDensity, Real.exp_le_exp]
    have h2 : 2 * m / sigma2 ≤ 2 * U z / sigma2 :=
      two_mul_div_le_two_mul_div hs (hmin z ⟨hz.1, hz.2.trans hyb⟩)
    linarith
  have h2 : (y - a) * Real.exp (-(2 * m / sigma2)) ≤ (b - a) * Real.exp (-(2 * m / sigma2)) :=
    mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos _).le
  linarith

/-- **The matching Arrhenius upper bound.** If `M` dominates and `m` is dominated by `U` on the
whole interval — for the single-barrier potential of the paper's (A5) these are the barrier top
and the basin bottom, so that `M - m` is exactly the barrier action `α` — then the closed form
is at most an explicit constant times `exp(2(M - m)/σ²)`. Together with
`meanExitTime_arrhenius_lower_bound` this pins the exit time to `exp(2α/σ²)` at the level of the
leading exponential, which is the precision of the paper's `≍`. -/
theorem meanExitTime_upper_bound {U : ℝ → ℝ} {a b x M m : ℝ} {sigma2 : ℝ} (hs : 0 < sigma2)
    (hax : a ≤ x) (hxb : x ≤ b) (hU : ContinuousOn U (Set.Icc a b))
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ M) (hmin : ∀ z ∈ Set.Icc a b, m ≤ U z) :
    meanExitTime sigma2 U a b x
      ≤ 2 / sigma2 * ((b - x) * (b - a)) * Real.exp (2 * (M - m) / sigma2) := by
  have hab : a ≤ b := hax.trans hxb
  have hpt : ∀ y ∈ Set.Icc x b, exitIntegrand sigma2 U a y
      ≤ Real.exp (2 * M / sigma2) * ((b - a) * Real.exp (-(2 * m / sigma2))) := by
    intro y hy
    have hay : a ≤ y := hax.trans hy.1
    have hscale : scaleDensity sigma2 U y ≤ Real.exp (2 * M / sigma2) := by
      rw [scaleDensity, Real.exp_le_exp]
      exact two_mul_div_le_two_mul_div hs (hmax y ⟨hay, hy.2⟩)
    have hspeed : speedIntegral sigma2 U a y ≤ (b - a) * Real.exp (-(2 * m / sigma2)) :=
      speedIntegral_le hs hU hay hy.2 hmin
    exact mul_le_mul hscale hspeed (speedIntegral_nonneg sigma2 U hay) (Real.exp_pos _).le
  have hint : (∫ y in x..b, exitIntegrand sigma2 U a y)
      ≤ (b - x) * (Real.exp (2 * M / sigma2) * ((b - a) * Real.exp (-(2 * m / sigma2)))) :=
    intervalIntegral_le_const_mul hxb
      ((continuousOn_exitIntegrand sigma2 hab hU).mono (Set.Icc_subset_Icc hax le_rfl)) hpt
  have hexp : Real.exp (2 * M / sigma2) * Real.exp (-(2 * m / sigma2))
      = Real.exp (2 * (M - m) / sigma2) := by
    rw [← Real.exp_add]; congr 1; ring
  have hfront : (0 : ℝ) ≤ 2 / sigma2 := div_nonneg (by norm_num) hs.le
  rw [meanExitTime]
  calc 2 / sigma2 * ∫ y in x..b, exitIntegrand sigma2 U a y
      ≤ 2 / sigma2 *
          ((b - x) * (Real.exp (2 * M / sigma2) * ((b - a) * Real.exp (-(2 * m / sigma2))))) :=
        mul_le_mul_of_nonneg_left hint hfront
    _ = 2 / sigma2 * ((b - x) * (b - a)) * Real.exp (2 * (M - m) / sigma2) := by
        rw [← hexp]; ring

/-! ### The Kramers exponent in the small-noise limit

The lower-bound constant degenerates as its tolerance `δ → 0`, so for a fixed diffusion the two
bounds do not meet. They do meet at the level of exponents: sending `σ² → 0⁺` first and `δ → 0`
second is legitimate, and it identifies the limit of `σ² · log T(x)` exactly. This section
carries out that double limit, so that "the leading exponential factor is `exp(2α/σ²)`" is
a theorem rather than a gloss on a pair of inequalities. -/

/-- As `s → 0⁺`, `s · log (C / s) → 0`: the prefactors `(2/σ²) · L` and `(2/σ²) · (b-x)(b-a)`
of the two Arrhenius bounds are sub-exponential, so they vanish from the exponent in the
small-noise limit. -/
theorem tendsto_mul_log_const_div {C : ℝ} (hC : 0 < C) :
    Filter.Tendsto (fun s : ℝ => s * Real.log (C / s)) (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have h1 : Filter.Tendsto (fun s : ℝ => s * Real.log C) (𝓝[>] (0 : ℝ)) (𝓝 0) := by
    have h : Filter.Tendsto (fun s : ℝ => s * Real.log C) (𝓝 (0 : ℝ))
        (𝓝 (0 * Real.log C)) := (continuous_mul_const _).tendsto 0
    rw [zero_mul] at h
    exact h.mono_left nhdsWithin_le_nhds
  have h2 : Filter.Tendsto (fun s : ℝ => s * Real.log s) (𝓝[>] (0 : ℝ)) (𝓝 0) := by
    have h := Real.continuous_mul_log.tendsto (0 : ℝ)
    rw [Real.log_zero, mul_zero] at h
    exact h.mono_left nhdsWithin_le_nhds
  have h12 := h1.sub h2
  rw [sub_zero] at h12
  refine h12.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hspos : (0 : ℝ) < s := hs
  rw [Real.log_div hC.ne' hspos.ne']
  ring

/-- **The Kramers exponent, as a genuine limit.** Under the single-barrier structure of (A5),

    `σ² · log T(x)  →  2α = 2 (U(w_s) - U(w_b))`   as `σ² → 0⁺`.

This is the two-sided Arrhenius bound closed at the level of exponents: the lower bound is
applied with tolerance `δ` chosen from the target accuracy, the upper bound as it stands, and
the two crude prefactors disappear because they are sub-exponential
(`tendsto_mul_log_const_div`). It is the paper's `E[τ] ≍ exp(2α/σ_eff²)` in the precise sense
of equality of leading exponential factors — `log`-scale equivalence — which is all that the
comparison of Corollary 1 consumes; the sub-exponential prefactor itself is *not* controlled,
and the sharp Kramers constant would be needed to control it. -/
theorem tendsto_mul_log_meanExitTime {U : ℝ → ℝ} {a b x wb ws : ℝ}
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun s => s * Real.log (meanExitTime s U a b x)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * (U ws - U wb))) := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hab : a ≤ b := hax.trans hxb
  have hbx : (0 : ℝ) < b - x := by linarith
  have hba : (0 : ℝ) < b - a := by linarith
  rw [tendsto_order]
  constructor
  · -- every `c < 2α` is eventually exceeded, via the lower bound at tolerance `d`
    intro c hc
    have heps : (0 : ℝ) < 2 * (U ws - U wb) - c := by linarith
    set d : ℝ := min ((U ws - U wb) / 2) ((2 * (U ws - U wb) - c) / 8) with hd_def
    have hd_pos : 0 < d := lt_min (by linarith) (by linarith)
    have hdle : d ≤ (2 * (U ws - U wb) - c) / 8 := by rw [hd_def]; exact min_le_right _ _
    obtain ⟨L, hL, hlow⟩ :=
      meanExitTime_arrhenius_lower_bound_uniform (delta := d) hU hax hawb hwbws hwsb hxws hd_pos
    have h2L : (0 : ℝ) < 2 * L := by linarith
    have hpref := (tendsto_mul_log_const_div h2L).eventually
      (eventually_gt_nhds
        (show -((2 * (U ws - U wb) - c) / 4) < (0 : ℝ) by linarith))
    filter_upwards [hpref, self_mem_nhdsWithin] with s hs1 hs2
    have hspos : (0 : ℝ) < s := hs2
    have hsne : s ≠ 0 := hspos.ne'
    have hlow' := hlow s hspos
    have heq : 2 / s * L = 2 * L / s := by ring
    rw [heq] at hlow'
    have hPpos : (0 : ℝ) < 2 * L / s * Real.exp (2 * (U ws - U wb - d) / s) :=
      mul_pos (div_pos h2L hspos) (Real.exp_pos _)
    have hlog : Real.log (2 * L / s * Real.exp (2 * (U ws - U wb - d) / s))
        ≤ Real.log (meanExitTime s U a b x) := Real.log_le_log hPpos hlow'
    have hmul := mul_le_mul_of_nonneg_left hlog hspos.le
    have hkey : s * Real.log (2 * L / s * Real.exp (2 * (U ws - U wb - d) / s))
        = s * Real.log (2 * L / s) + 2 * (U ws - U wb - d) := by
      rw [Real.log_mul (div_pos h2L hspos).ne' (Real.exp_ne_zero _), Real.log_exp, mul_add]
      congr 1
      field_simp
    rw [hkey] at hmul
    linarith
  · -- and every `c > 2α` eventually dominates, via the upper bound
    intro c hc
    have heps : (0 : ℝ) < c - 2 * (U ws - U wb) := by linarith
    have hA : (0 : ℝ) < 2 * ((b - x) * (b - a)) :=
      mul_pos (by norm_num) (mul_pos hbx hba)
    have hpref := (tendsto_mul_log_const_div hA).eventually
      (eventually_lt_nhds (show (0 : ℝ) < (c - 2 * (U ws - U wb)) / 2 by linarith))
    obtain ⟨L₀, hL₀, hlow₀⟩ :=
      meanExitTime_arrhenius_lower_bound_uniform (delta := (U ws - U wb) / 2)
        hU hax hawb hwbws hwsb hxws (by linarith)
    filter_upwards [hpref, self_mem_nhdsWithin] with s hs1 hs2
    have hspos : (0 : ℝ) < s := hs2
    have hTpos : (0 : ℝ) < meanExitTime s U a b x := by
      have h := hlow₀ s hspos
      have hp : (0 : ℝ)
          < 2 / s * L₀ * Real.exp (2 * (U ws - U wb - (U ws - U wb) / 2) / s) :=
        mul_pos (mul_pos (div_pos (by norm_num) hspos) hL₀) (Real.exp_pos _)
      linarith
    have hupbd := meanExitTime_upper_bound (sigma2 := s) (M := U ws) (m := U wb)
      hspos hax hxb hU hmax hmin
    have heq : 2 / s * ((b - x) * (b - a)) = 2 * ((b - x) * (b - a)) / s := by ring
    rw [heq] at hupbd
    have hlog : Real.log (meanExitTime s U a b x)
        ≤ Real.log (2 * ((b - x) * (b - a)) / s * Real.exp (2 * (U ws - U wb) / s)) :=
      Real.log_le_log hTpos hupbd
    have hmul := mul_le_mul_of_nonneg_left hlog hspos.le
    have hsne : s ≠ 0 := hspos.ne'
    have hkey : s * Real.log (2 * ((b - x) * (b - a)) / s * Real.exp (2 * (U ws - U wb) / s))
        = s * Real.log (2 * ((b - x) * (b - a)) / s) + 2 * (U ws - U wb) := by
      rw [Real.log_mul (div_pos hA hspos).ne' (Real.exp_ne_zero _), Real.log_exp, mul_add]
      congr 1
      field_simp
    rw [hkey] at hmul
    linarith

/-- The Kramers exponent is strictly decreasing in the effective diffusion: this is the bare
comparison `exp(2α/σ₂²) < exp(2α/σ₁²)` that Corollary 1 reads off, for a barrier `α > 0` held
fixed across the comparison. -/
theorem exp_barrier_strictAnti {alpha s₁ s₂ : ℝ} (halpha : 0 < alpha) (hs₁ : 0 < s₁)
    (h : s₁ < s₂) : Real.exp (2 * alpha / s₂) < Real.exp (2 * alpha / s₁) := by
  rw [Real.exp_lt_exp]
  exact div_lt_div_of_pos_left (by linarith) hs₁ h

/-- Transitivity of the two Arrhenius bounds: a lower bound on one exit time and an upper bound
on the other order the two exit times as soon as the two bounds themselves are ordered. -/
theorem meanExitTime_lt_of_gap {T₁ T₂ c C E₁ E₂ : ℝ} (hlow : c * Real.exp E₁ ≤ T₁)
    (hup : T₂ ≤ C * Real.exp E₂) (hgap : C * Real.exp E₂ < c * Real.exp E₁) : T₂ < T₁ :=
  lt_of_le_of_lt hup (lt_of_lt_of_le hgap hlow)

/-- **Corollary 1's comparison, in the form this development actually proves.** With one and the
same potential and barrier, the exit time at the diffusion `sl` is strictly below the exit time
at `sd` *provided* the proved upper bound at `sl` falls below the proved lower bound at `sd`.
The constant `c` is the one produced by `meanExitTime_arrhenius_lower_bound`; the upper bound is
`meanExitTime_upper_bound`, and it needs the single-barrier structure `hmax`/`hmin` of (A5),
which is what makes `U w_s - U w_b` the true oscillation of `U` on the interval and hence makes
the two exponents comparable.

Read the statement with two things in mind. First, `sd < sl` is *not* a hypothesis: the gap
condition carries the whole comparison, and it is the gap, not the bare ordering of the two
diffusions, that this development can certify. Second, `c` is existentially quantified — it
comes out of the moduli of continuity of `U` at the two extrema, which the hypotheses do not
supply — so the gap condition here is not an inequality a reader can evaluate on given data.
`meanExitTime_lift_lt_direct_of_window` is the same comparison with the existential removed.

The gap hypothesis is not vacuous: `eventually_lower_bound_gt` shows that, whatever the positive
constant, it holds for every sufficiently small `sd`, which is precisely the small-noise clause
of (A5). What is *not* proved, and what a sharp Kramers law would supply, is that the gap holds
under the sole hypothesis `sd < sl`; the prefactors here are crude, and closing that would
require the second-order Laplace expansion at the barrier top and the basin bottom. -/
theorem meanExitTime_lift_lt_direct {U : ℝ → ℝ} {a b x wb ws delta : ℝ} {sd sl : ℝ}
    (hsd : 0 < sd) (hsl : 0 < sl) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∃ c > 0, c * Real.exp (2 * (U ws - U wb - delta) / sd) ≤ meanExitTime sd U a b x ∧
      (2 / sl * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / sl)
          < c * Real.exp (2 * (U ws - U wb - delta) / sd) →
        meanExitTime sl U a b x < meanExitTime sd U a b x) := by
  obtain ⟨c, hcpos, hlow⟩ :=
    meanExitTime_arrhenius_lower_bound (sigma2 := sd) hsd hU hax hawb hwbws hwsb hxws hdelta
  refine ⟨c, hcpos, hlow, fun hgap => ?_⟩
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hup := meanExitTime_upper_bound (sigma2 := sl) (M := U ws) (m := U wb)
    hsl hax hxb hU hmax hmin
  exact meanExitTime_lt_of_gap hlow hup hgap

/-- **Corollary 1's comparison with every constant explicit.** The lower-bound constant of
`meanExitTime_lift_lt_direct` is existentially quantified: it is manufactured out of the moduli
of continuity of `U` at the barrier top and the basin bottom, so a reader who wants to *check*
the gap condition for a prescribed pair of diffusions is not handed a number. This variant
removes the existential. The caller supplies the barrier window `[s₁, s₂]` on which `U ≥ c_s`,
the basin window `[t₁, t₂]` on which `U ≤ c_b`, and the two levels; the gap hypothesis is then
an inequality between two closed-form real numbers built from the data alone, and under it the
exit time at `sl` is strictly below the exit time at `sd`.

This is the form in which the comparison is genuinely checkable. The price is that the caller
must exhibit the two windows, which is exactly the Laplace input that
`meanExitTime_arrhenius_lower_bound_uniform` manufactures from continuity alone — and which,
for a potential given by a formula, is elementary to write down. Note that `sd < sl` is not a
hypothesis: the gap condition carries the whole comparison, and it is the gap, not the bare
ordering of the two diffusions, that this development can certify. -/
theorem meanExitTime_lift_lt_direct_of_window {U : ℝ → ℝ} {a b x s₁ s₂ t₁ t₂ cs cb wb ws : ℝ}
    {sd sl : ℝ} (hsd : 0 < sd) (hsl : 0 < sl) (hU : ContinuousOn U (Set.Icc a b))
    (hax : a ≤ x) (hxs₁ : x ≤ s₁) (hs₁₂ : s₁ ≤ s₂) (hs₂b : s₂ ≤ b)
    (hat₁ : a ≤ t₁) (ht₁₂ : t₁ ≤ t₂) (ht₂s₁ : t₂ ≤ s₁)
    (hbarrier : ∀ y ∈ Set.Icc s₁ s₂, cs ≤ U y)
    (hbasin : ∀ z ∈ Set.Icc t₁ t₂, U z ≤ cb)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z)
    (hgap : 2 / sl * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / sl)
        < 2 / sd * ((s₂ - s₁) * (t₂ - t₁)) * Real.exp (2 * (cs - cb) / sd)) :
    meanExitTime sl U a b x < meanExitTime sd U a b x := by
  have hxb : x ≤ b := hxs₁.trans (hs₁₂.trans hs₂b)
  have hlow := meanExitTime_window_lower_bound hsd hU hax hxs₁ hs₁₂ hs₂b hat₁ ht₁₂ ht₂s₁
    hbarrier hbasin
  have hup := meanExitTime_upper_bound (sigma2 := sl) (M := U ws) (m := U wb)
    hsl hax hxb hU hmax hmin
  linarith

/-- **The gap hypothesis is a genuine small-noise condition, not a disguised assumption of the
conclusion.** For a fixed positive exponent scale `β = 2(α - δ)` and a fixed positive window
constant `L`, the Arrhenius lower bound `(2/s)·L·exp(β/s)` exceeds any prescribed threshold for
all sufficiently small effective diffusion `s`. -/
theorem eventually_lower_bound_gt {beta L kappa : ℝ} (hbeta : 0 < beta) (hL : 0 < L) :
    ∀ᶠ s in 𝓝[>] (0 : ℝ), kappa < 2 / s * L * Real.exp (beta / s) := by
  have h1 : Filter.Tendsto (fun s : ℝ => beta / s) (𝓝[>] (0 : ℝ)) Filter.atTop := by
    simp only [div_eq_mul_inv]
    exact Filter.Tendsto.const_mul_atTop hbeta tendsto_inv_nhdsGT_zero
  have h2 : Filter.Tendsto (fun s : ℝ => 2 * L * Real.exp (beta / s)) (𝓝[>] (0 : ℝ))
      Filter.atTop :=
    Filter.Tendsto.const_mul_atTop (by linarith) (Real.tendsto_exp_atTop.comp h1)
  have h3 : ∀ᶠ s in 𝓝[>] (0 : ℝ), kappa < 2 * L * Real.exp (beta / s) :=
    h2.eventually_gt_atTop kappa
  have h4 : ∀ᶠ s in 𝓝[>] (0 : ℝ), s ≤ 1 :=
    ((eventually_lt_nhds (by norm_num : (0 : ℝ) < 1)).filter_mono
      (nhdsWithin_le_nhds (s := Set.Ioi (0 : ℝ)))).mono fun _ hx => hx.le
  filter_upwards [h3, h4, self_mem_nhdsWithin] with s hs1 hs2 hs3
  have hspos : (0 : ℝ) < s := hs3
  have hmono : 2 * L * Real.exp (beta / s) ≤ 2 / s * L * Real.exp (beta / s) := by
    refine mul_le_mul_of_nonneg_right ?_ (Real.exp_pos _).le
    refine mul_le_mul_of_nonneg_right ?_ hL.le
    rw [le_div_iff₀ hspos]
    linarith
  linarith

/-- **Corollary 1's ordering in the small-noise regime, with no gap hypothesis at all.** Fix the
potential, the barrier, the interval and the lifted diffusion `sl > 0`. Then for every
sufficiently small direct diffusion `sd`, the lifted exit time is strictly below the direct one.
The only barrier hypothesis is the paper's own: a non-vanishing barrier, `U(w_b) < U(w_s)`; the
Laplace tolerance is chosen inside the proof (`δ = α/2`) and does not appear in the statement.

This is unconditional in the sense that matters: nothing is assumed about the relative sizes of
the two Arrhenius prefactors. It works because `meanExitTime_arrhenius_lower_bound_uniform`
supplies one window constant `L` valid for every diffusion at once, so the lower bound
`(2/sd)·L·exp(2(α - δ)/sd)` diverges as `sd → 0` while the upper bound for the fixed `sl`
stays put. Note the quantifier structure: `sl` is *fixed* while `sd` shrinks, so the two
diffusions here are decoupled; the paper's own pair is the additively coupled `σ²` vs
`σ² + σ_Jac²`, treated in `eventually_meanExitTime_additive_lift_lt_direct` below.

What is still not obtained is a threshold: "sufficiently small" is a filter statement, and
turning it into an explicit inequality between `sd` and `sl` would need the sharp Kramers
prefactor. -/
theorem eventually_meanExitTime_lift_lt_direct {U : ℝ → ℝ} {a b x wb ws : ℝ} {sl : ℝ}
    (hsl : 0 < sl) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ), meanExitTime sl U a b x < meanExitTime sd U a b x := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hdelta : (0 : ℝ) < (U ws - U wb) / 2 := by linarith
  obtain ⟨L, hL, hbound⟩ :=
    meanExitTime_arrhenius_lower_bound_uniform (delta := (U ws - U wb) / 2)
      hU hax hawb hwbws hwsb hxws hdelta
  have hup : meanExitTime sl U a b x
      ≤ 2 / sl * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / sl) :=
    meanExitTime_upper_bound (sigma2 := sl) (M := U ws) (m := U wb) hsl hax hxb hU hmax hmin
  have hbeta : 0 < 2 * (U ws - U wb - (U ws - U wb) / 2) := by linarith
  have hgap := eventually_lower_bound_gt (beta := 2 * (U ws - U wb - (U ws - U wb) / 2))
    (L := L)
    (kappa := 2 / sl * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / sl)) hbeta hL
  filter_upwards [hgap, self_mem_nhdsWithin] with sd hsd1 hsd2
  have hsdpos : (0 : ℝ) < sd := hsd2
  have hlow := hbound sd hsdpos
  linarith

/-- **Corollary 1's ordering for the paper's own coupled pair of diffusions.** Equation (5) of
the paper compares `E[τ_direct] ≍ exp(2α/(σ_s²σ_obj²))` with
`E[τ_hyper] ≍ exp(2α/(σ_s²σ_obj² + σ_Jac²))`: the two effective variances are not free-floating
but *additively coupled* — the lift's variance is the direct one plus the slack-channel
cross-covariance `σ_Jac² > 0`. This theorem states the ordering for exactly that family: with
`jac2 = σ_Jac²` held fixed, for every sufficiently small direct diffusion `sd` the exit time at
the lifted diffusion `sd + jac2` is strictly below the exit time at `sd`. The proof combines
the diffusion-uniform lower bound at `sd` with the upper bound at `sd + jac2`, which is itself
bounded by the `jac2`-only ceiling `(2/jac2)·(b-x)(b-a)·exp(2α/jac2)` because both the
prefactor and the exponent improve as the diffusion grows. -/
theorem eventually_meanExitTime_additive_lift_lt_direct {U : ℝ → ℝ} {a b x wb ws jac2 : ℝ}
    (hjac : 0 < jac2) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      meanExitTime (sd + jac2) U a b x < meanExitTime sd U a b x := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hdelta : (0 : ℝ) < (U ws - U wb) / 2 := by linarith
  obtain ⟨L, hL, hbound⟩ :=
    meanExitTime_arrhenius_lower_bound_uniform (delta := (U ws - U wb) / 2)
      hU hax hawb hwbws hwsb hxws hdelta
  -- the whole lifted family sits below one ceiling, because a larger diffusion improves both
  -- the prefactor and the exponent of the upper bound
  have hup : ∀ sd : ℝ, 0 < sd →
      meanExitTime (sd + jac2) U a b x
        ≤ 2 / jac2 * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / jac2) := by
    intro sd hsd
    have hsl : (0 : ℝ) < sd + jac2 := by linarith
    have h1 := meanExitTime_upper_bound (sigma2 := sd + jac2) (M := U ws) (m := U wb)
      hsl hax hxb hU hmax hmin
    have hAnn : (0 : ℝ) ≤ (b - x) * (b - a) :=
      mul_nonneg (by linarith) (by linarith)
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
    have hnn : (0 : ℝ) ≤ 2 / jac2 * ((b - x) * (b - a)) :=
      mul_nonneg (by positivity) hAnn
    exact mul_le_mul hstep1 hexple (Real.exp_pos _).le hnn
  have hbeta : 0 < 2 * (U ws - U wb - (U ws - U wb) / 2) := by linarith
  have hgap := eventually_lower_bound_gt (beta := 2 * (U ws - U wb - (U ws - U wb) / 2))
    (L := L)
    (kappa := 2 / jac2 * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / jac2)) hbeta hL
  filter_upwards [hgap, self_mem_nhdsWithin] with sd hsd1 hsd2
  have hsdpos : (0 : ℝ) < sd := hsd2
  have hlow := hbound sd hsdpos
  linarith [hup sd hsdpos]

/-! ### The paper-facing statements

The theorems below are the ones Corollary 1 (`cor:fpt`) needs. Each carries one named
analytic hypothesis, `hDynkin`, saying that the mean first-passage time `tau` of the diffusion
`dX = -U'(X) dt + σ dB` on `(a, b)` — reflecting at `a`, absorbing at `b` — together with its
first two derivatives, solves `MeanExitTimeBVP`. That is exactly the conclusion of Dynkin's
formula for the generator `A = (σ²/2) ∂² - U' ∂` applied to the exit time, and it is the only
step of Corollary 1 left unformalized: this mathlib has no Itô calculus and no diffusion
generator with which to prove it, so it is carried as a hypothesis and never as an axiom.

Note what `hDynkin` does *not* smuggle in. It does not assert the closed form; the passage from
the boundary-value problem to the closed form is `eq_meanExitTime_of_bvp`, proved above. It does
not assert any exponential behaviour; that is `meanExitTime_arrhenius_lower_bound` and
`meanExitTime_upper_bound`, also proved above. Everything after `hDynkin` is proved outright. -/

/-- **Corollary 1's Arrhenius lower bound for the mean first-passage time.** Given Dynkin's
formula — that is, given that the mean first-passage time solves the boundary-value problem —
the mean first-passage time across a barrier of action `α = U(w_s) - U(w_b)` is at least a
positive constant times `exp(2(α - δ)/σ²)` for every `δ > 0`: a smaller effective diffusion
forces an exponentially large exit time. -/
theorem mean_first_passage_arrhenius_of_dynkin {U U' : ℝ → ℝ} {a b x wb ws delta : ℝ}
    {sigma2 : ℝ} (tau Dtau DDtau : ℝ → ℝ)
    (hDynkin : MeanExitTimeBVP sigma2 U' a b tau Dtau DDtau)
    (htau : ContinuousOn tau (Set.Icc a b))
    (hs : 0 < sigma2) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x) (hxb : x ≤ b)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta) :
    ∃ c > 0, c * Real.exp (2 * (U ws - U wb - delta) / sigma2) ≤ tau x := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  rw [eq_meanExitTime_of_bvp hs hab hU hU' htau hDynkin x ⟨hax, hxb⟩]
  exact meanExitTime_arrhenius_lower_bound hs hU hax hawb hwbws hwsb hxws hdelta

/-- **Corollary 1's ordering of the two mean first-passage times.** With the barrier held fixed
across the comparison, as the paper requires, the process at diffusion `sl` (in the paper's
reading the lift, with the larger effective variance `σ_s²σ_obj² + σ_Jac²`) escapes strictly
faster than the process at `sd` (the direct softplus, `σ_s²σ_obj²`), once the proved upper bound
at `sl` falls below the proved lower bound at `sd`. As in `meanExitTime_lift_lt_direct`, which
this specializes, `sd < sl` is not a hypothesis and the lower constant `c` is existentially
quantified; the unconditional small-noise forms are the two `eventually` theorems below. -/
theorem mean_first_passage_ordering_of_dynkin {U U' : ℝ → ℝ} {a b x wb ws delta : ℝ}
    {sd sl : ℝ} (taud Dtaud DDtaud taul Dtaul DDtaul : ℝ → ℝ)
    (hDynkinD : MeanExitTimeBVP sd U' a b taud Dtaud DDtaud)
    (hDynkinL : MeanExitTimeBVP sl U' a b taul Dtaul DDtaul)
    (htaud : ContinuousOn taud (Set.Icc a b)) (htaul : ContinuousOn taul (Set.Icc a b))
    (hsd : 0 < sd) (hsl : 0 < sl) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∃ c > 0, c * Real.exp (2 * (U ws - U wb - delta) / sd) ≤ taud x ∧
      (2 / sl * ((b - x) * (b - a)) * Real.exp (2 * (U ws - U wb) / sl)
          < c * Real.exp (2 * (U ws - U wb - delta) / sd) → taul x < taud x) := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  rw [eq_meanExitTime_of_bvp hsd hab hU hU' htaud hDynkinD x ⟨hax, hxb⟩,
    eq_meanExitTime_of_bvp hsl hab hU hU' htaul hDynkinL x ⟨hax, hxb⟩]
  exact meanExitTime_lift_lt_direct hsd hsl hU hax hawb hwbws hwsb hxws hdelta hmax hmin

/-- **Corollary 1's ordering of the two mean first-passage times, in the small-noise regime.**
Here the direct diffusion is not prescribed: `taud sd` is the mean first-passage time of the
direct process at effective diffusion `sd`, assumed (Dynkin) to solve the boundary-value problem
for every `sd > 0`, and the conclusion is that for every sufficiently small `sd` the lift's mean
first-passage time is strictly smaller. No relation between the two Arrhenius prefactors is
assumed, and the only barrier hypothesis is the paper's `α > 0`. The lifted diffusion `sl` is
held fixed while `sd` shrinks; the paper's own coupled pair `σ²` vs `σ² + σ_Jac²` is
`eventually_mean_first_passage_additive_lift_lt_direct` below. -/
theorem eventually_mean_first_passage_lift_lt_direct {U U' : ℝ → ℝ} {a b x wb ws : ℝ}
    {sl : ℝ} (taud Dtaud DDtaud : ℝ → ℝ → ℝ) (taul Dtaul DDtaul : ℝ → ℝ)
    (hDynkinD : ∀ sd : ℝ, 0 < sd → MeanExitTimeBVP sd U' a b (taud sd) (Dtaud sd) (DDtaud sd))
    (hDynkinL : MeanExitTimeBVP sl U' a b taul Dtaul DDtaul)
    (htaud : ∀ sd : ℝ, 0 < sd → ContinuousOn (taud sd) (Set.Icc a b))
    (htaul : ContinuousOn taul (Set.Icc a b))
    (hsl : 0 < sl) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ), taul x < taud sd x := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  filter_upwards [eventually_meanExitTime_lift_lt_direct hsl hU hax hawb hwbws hwsb hxws
      hbarrier hmax hmin, self_mem_nhdsWithin] with sd h1 h2
  have hsdpos : (0 : ℝ) < sd := h2
  rw [eq_meanExitTime_of_bvp hsl hab hU hU' htaul hDynkinL x ⟨hax, hxb⟩,
    eq_meanExitTime_of_bvp hsdpos hab hU hU' (htaud sd hsdpos) (hDynkinD sd hsdpos) x ⟨hax, hxb⟩]
  exact h1

/-- **Corollary 1's ordering for the paper's own coupled pair, mean-first-passage form.**
Equation (5) of the paper compares the direct process at effective variance `σ_s²σ_obj²` with
the lifted process at `σ_s²σ_obj² + σ_Jac²`: one family, indexed by the shared attenuated
variance, with the lift offset by the fixed slack-channel cross-covariance `jac2 = σ_Jac² > 0`.
Given Dynkin's formula for both members at every index, the lift's mean first-passage time is
strictly below the direct one for every sufficiently small shared variance `sd`. This is the
faithful quantifier structure of the paper's comparison — the two diffusions are never varied
independently — and it is the machine-checked content of "the lift escapes the shoulder
strictly faster than the direct softplus" in the small-noise regime (A5) postulates. -/
theorem eventually_mean_first_passage_additive_lift_lt_direct {U U' : ℝ → ℝ}
    {a b x wb ws jac2 : ℝ}
    (taud Dtaud DDtaud taul Dtaul DDtaul : ℝ → ℝ → ℝ)
    (hDynkinD : ∀ sd : ℝ, 0 < sd → MeanExitTimeBVP sd U' a b (taud sd) (Dtaud sd) (DDtaud sd))
    (hDynkinL : ∀ sd : ℝ, 0 < sd →
      MeanExitTimeBVP (sd + jac2) U' a b (taul sd) (Dtaul sd) (DDtaul sd))
    (htaud : ∀ sd : ℝ, 0 < sd → ContinuousOn (taud sd) (Set.Icc a b))
    (htaul : ∀ sd : ℝ, 0 < sd → ContinuousOn (taul sd) (Set.Icc a b))
    (hjac : 0 < jac2) (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ), taul sd x < taud sd x := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  filter_upwards [eventually_meanExitTime_additive_lift_lt_direct hjac hU hax hawb hwbws hwsb
      hxws hbarrier hmax hmin, self_mem_nhdsWithin] with sd h1 h2
  have hsdpos : (0 : ℝ) < sd := h2
  have hslpos : (0 : ℝ) < sd + jac2 := by linarith
  rw [eq_meanExitTime_of_bvp hslpos hab hU hU' (htaul sd hsdpos) (hDynkinL sd hsdpos)
      x ⟨hax, hxb⟩,
    eq_meanExitTime_of_bvp hsdpos hab hU hU' (htaud sd hsdpos) (hDynkinD sd hsdpos) x ⟨hax, hxb⟩]
  exact h1

/-- **The paper's `≍`, at the precision actually proved, for the mean first-passage time.**
Given Dynkin's formula at every diffusion, the Kramers exponent of the mean first-passage time
is exactly the barrier action: `σ² · log E[τ(σ²)] → 2α` as `σ² → 0⁺`. This is equality of the
leading exponential factor — the content of the paper's `E[τ] ≍ exp(2α/σ_eff²)` at `log` scale
— and it is what pins each side of the comparison in equation (5) of the paper. The
sub-exponential prefactor is not controlled; see the module docstring. -/
theorem mean_first_passage_kramers_exponent_of_dynkin {U U' : ℝ → ℝ} {a b x wb ws : ℝ}
    (tau Dtau DDtau : ℝ → ℝ → ℝ)
    (hDynkin : ∀ s : ℝ, 0 < s → MeanExitTimeBVP s U' a b (tau s) (Dtau s) (DDtau s))
    (htau : ∀ s : ℝ, 0 < s → ContinuousOn (tau s) (Set.Icc a b))
    (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun s => s * Real.log (tau s x)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * (U ws - U wb))) := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  refine (tendsto_mul_log_meanExitTime hU hax hawb hwbws hwsb hxws hbarrier hmax hmin).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hspos : (0 : ℝ) < s := hs
  rw [eq_meanExitTime_of_bvp hspos hab hU hU' (htau s hspos) (hDynkin s hspos) x ⟨hax, hxb⟩]

/-! ### Non-vacuity

The theorems above carry a stack of hypotheses — the Dynkin boundary-value problem for a whole
family of diffusions, `C¹` regularity, the ordering of the five points, the single-barrier
shape, the positive barrier. A theorem whose hypotheses no instance satisfies would be
worthless, so the stack is discharged here on a concrete instance: the cubic potential
`U(y) = y - y³/3` on `[-3/2, 3/2]`, which has its basin bottom at `w_b = -1` and its barrier
top at `w_s = 1`, and the closed form itself as the family of solutions — `eq_meanExitTime_of_bvp`
says there is no other choice.

Three statements are recorded, not one. `paper_facing_hypotheses_satisfiable` exhibits the
hypothesis stack; `cubicBarrier_kramers_exponent` and `cubicBarrier_additive_lift_lt_direct`
then run the two paper-facing conclusions on that instance, so that the non-vacuity claim is
machine-checked at the conclusions rather than asserted about the hypotheses. The second of the
two exercises the shifted family `s ↦ τ(s + σ_Jac²)` that the paper's own additively coupled
comparison needs, and which the hypothesis stack alone does not exhibit. -/

/-- The **cubic barrier potential** `U(y) = y - y³/3`, the non-vacuity witness. On `[-3/2, 3/2]`
its basin bottom is `w_b = -1` with `U(w_b) = -2/3`, its barrier top is `w_s = 1` with
`U(w_s) = 2/3`, and the barrier action is `α = U(w_s) - U(w_b) = 4/3 > 0`. This is the
metastable single-barrier shape of the paper's (A5) in the smallest example that has one: the
potential falls from the reflecting wall into the basin, rises across the barrier, and falls
again to the absorbing wall. -/
noncomputable def cubicBarrier (y : ℝ) : ℝ := y - y ^ 3 / 3

/-- The derivative `U'(y) = 1 - y²` of `cubicBarrier`; the witness diffusion is
`dX = -(1 - X²) dt + σ dB`. -/
noncomputable def cubicBarrierDeriv (y : ℝ) : ℝ := 1 - y ^ 2

/-- The witness potential is continuous on the closed interval, which is the regularity `hU` of
every theorem in this file. -/
theorem continuousOn_cubicBarrier :
    ContinuousOn cubicBarrier (Set.Icc (-(3 / 2) : ℝ) (3 / 2)) :=
  (continuous_id.sub ((continuous_pow 3).div_const 3)).continuousOn

/-- The witness potential is differentiable on the open interval with the stated derivative,
which is the regularity `hU'`. -/
theorem hasDerivAt_cubicBarrier :
    ∀ y ∈ Set.Ioo (-(3 / 2) : ℝ) (3 / 2), HasDerivAt cubicBarrier (cubicBarrierDeriv y) y := by
  intro y _
  have h2 : HasDerivAt (fun y : ℝ => y ^ 3) ((3 : ℝ) * y ^ 2) y := by
    simpa using hasDerivAt_pow 3 y
  have h3 := (hasDerivAt_id y).sub (h2.div_const 3)
  refine h3.congr_deriv ?_
  show (1 : ℝ) - 3 * y ^ 2 / 3 = 1 - y ^ 2
  ring

/-- The barrier top of the witness is the maximum of `U` on the interval: this is the `hmax`
half of the single-barrier structure of (A5). -/
theorem cubicBarrier_le_top :
    ∀ y ∈ Set.Icc (-(3 / 2) : ℝ) (3 / 2), cubicBarrier y ≤ cubicBarrier 1 := by
  intro y hy
  have h1 : (-(3 / 2) : ℝ) ≤ y := hy.1
  show y - y ^ 3 / 3 ≤ 1 - 1 ^ 3 / 3
  nlinarith [mul_nonneg (sq_nonneg (y - 1)) (show (0 : ℝ) ≤ y + 2 by linarith)]

/-- The basin bottom of the witness is the minimum of `U` on the interval: this is the `hmin`
half of the single-barrier structure of (A5). -/
theorem cubicBarrier_bottom_le :
    ∀ z ∈ Set.Icc (-(3 / 2) : ℝ) (3 / 2), cubicBarrier (-1) ≤ cubicBarrier z := by
  intro z hz
  have h2 : z ≤ (3 / 2 : ℝ) := hz.2
  show (-1 : ℝ) - (-1 : ℝ) ^ 3 / 3 ≤ z - z ^ 3 / 3
  nlinarith [mul_nonneg (sq_nonneg (z + 1)) (show (0 : ℝ) ≤ 2 - z by linarith)]

/-- The barrier action of the witness is `α = 4/3`. -/
theorem cubicBarrier_action : cubicBarrier 1 - cubicBarrier (-1) = 4 / 3 := by
  show (1 : ℝ) - 1 ^ 3 / 3 - ((-1 : ℝ) - (-1 : ℝ) ^ 3 / 3) = 4 / 3
  norm_num

/-- The witness has a non-vanishing barrier, the paper's standing hypothesis `α > 0`. -/
theorem cubicBarrier_pos : cubicBarrier (-1) < cubicBarrier 1 := by
  have h := cubicBarrier_action
  linarith

/-- **The hypothesis stack of the paper-facing theorems is satisfiable.** The cubic potential
`U(y) = y - y³/3` on `[a, b] = [-3/2, 3/2]`, with `w_b = -1`, `w_s = 1`, start point `x = -1`,
derivative `U'(y) = 1 - y²`, and the closed-form `meanExitTime` family as `tau`, satisfies every
hypothesis of `mean_first_passage_kramers_exponent_of_dynkin`: the Dynkin family, the
continuity, the point ordering, the positive barrier `U(w_b) = -2/3 < 2/3 = U(w_s)`, and the
single-barrier structure. The two theorems that follow discharge the remaining paper-facing
conclusions on the same instance, the second of them with the shifted family
`s ↦ meanExitTime (s + σ_Jac²)` that the additively coupled comparison requires. -/
theorem paper_facing_hypotheses_satisfiable :
    ∃ (U U' : ℝ → ℝ) (a b x wb ws : ℝ) (tau Dtau DDtau : ℝ → ℝ → ℝ),
      (∀ s : ℝ, 0 < s → MeanExitTimeBVP s U' a b (tau s) (Dtau s) (DDtau s)) ∧
      (∀ s : ℝ, 0 < s → ContinuousOn (tau s) (Set.Icc a b)) ∧
      ContinuousOn U (Set.Icc a b) ∧
      (∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) ∧
      a ≤ x ∧ a < wb ∧ wb < ws ∧ ws < b ∧ x < ws ∧
      U wb < U ws ∧
      (∀ y ∈ Set.Icc a b, U y ≤ U ws) ∧ (∀ z ∈ Set.Icc a b, U wb ≤ U z) := by
  have hab : (-(3 / 2) : ℝ) < 3 / 2 := by norm_num
  exact ⟨cubicBarrier, cubicBarrierDeriv, -(3 / 2), 3 / 2, -1, -1, 1,
    fun s => meanExitTime s cubicBarrier (-(3 / 2)) (3 / 2),
    fun s => meanExitTimeDeriv s cubicBarrier (-(3 / 2)),
    fun s => meanExitTimeDeriv2 s cubicBarrier cubicBarrierDeriv (-(3 / 2)),
    fun _ hs =>
      meanExitTime_satisfies_bvp hs hab continuousOn_cubicBarrier hasDerivAt_cubicBarrier,
    fun s _ => continuousOn_meanExitTime s hab.le continuousOn_cubicBarrier,
    continuousOn_cubicBarrier, hasDerivAt_cubicBarrier,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    cubicBarrier_pos, cubicBarrier_le_top, cubicBarrier_bottom_le⟩

/-- **The Kramers exponent of Corollary 1, evaluated on the witness.** Running
`mean_first_passage_kramers_exponent_of_dynkin` on the cubic barrier gives

    `σ² · log E[τ(σ²)] → 8/3 = 2α`   as `σ² → 0⁺`,

with `α = 4/3` the witness barrier action. The limit is a nonzero number, so the paper-facing
Kramers statement is not merely satisfiable but has non-trivial content on an instance. -/
theorem cubicBarrier_kramers_exponent :
    Filter.Tendsto
      (fun s => s * Real.log (meanExitTime s cubicBarrier (-(3 / 2)) (3 / 2) (-1)))
      (𝓝[>] (0 : ℝ)) (𝓝 (8 / 3)) := by
  have hab : (-(3 / 2) : ℝ) < 3 / 2 := by norm_num
  have h := mean_first_passage_kramers_exponent_of_dynkin
    (U := cubicBarrier) (U' := cubicBarrierDeriv) (a := -(3 / 2)) (b := 3 / 2) (x := -1)
    (wb := -1) (ws := 1)
    (tau := fun s => meanExitTime s cubicBarrier (-(3 / 2)) (3 / 2))
    (Dtau := fun s => meanExitTimeDeriv s cubicBarrier (-(3 / 2)))
    (DDtau := fun s => meanExitTimeDeriv2 s cubicBarrier cubicBarrierDeriv (-(3 / 2)))
    (fun _ hs =>
      meanExitTime_satisfies_bvp hs hab continuousOn_cubicBarrier hasDerivAt_cubicBarrier)
    (fun s _ => continuousOn_meanExitTime s hab.le continuousOn_cubicBarrier)
    continuousOn_cubicBarrier hasDerivAt_cubicBarrier (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) cubicBarrier_pos cubicBarrier_le_top cubicBarrier_bottom_le
  have hval : (2 : ℝ) * (cubicBarrier 1 - cubicBarrier (-1)) = 8 / 3 := by
    rw [cubicBarrier_action]; norm_num
  rw [hval] at h
  exact h

/-- **Corollary 1's additively coupled ordering, evaluated on the witness.** For every fixed
slack-channel excess `σ_Jac² > 0`, running `eventually_mean_first_passage_additive_lift_lt_direct`
on the cubic barrier with the direct family `s ↦ T(s)` and the lifted family `s ↦ T(s + σ_Jac²)`
gives: for every sufficiently small shared variance `sd`, the lift's mean first-passage time is
strictly below the direct one. This is the paper's own comparison — one family, the lift offset
by a fixed `σ_Jac²` — carried out end to end on a concrete potential, and it is what makes the
shifted-family instantiation a proved statement rather than a remark. -/
theorem cubicBarrier_additive_lift_lt_direct {jac2 : ℝ} (hjac : 0 < jac2) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      meanExitTime (sd + jac2) cubicBarrier (-(3 / 2)) (3 / 2) (-1)
        < meanExitTime sd cubicBarrier (-(3 / 2)) (3 / 2) (-1) := by
  have hab : (-(3 / 2) : ℝ) < 3 / 2 := by norm_num
  exact eventually_mean_first_passage_additive_lift_lt_direct
    (U := cubicBarrier) (U' := cubicBarrierDeriv) (a := -(3 / 2)) (b := 3 / 2) (x := -1)
    (wb := -1) (ws := 1) (jac2 := jac2)
    (taud := fun s => meanExitTime s cubicBarrier (-(3 / 2)) (3 / 2))
    (Dtaud := fun s => meanExitTimeDeriv s cubicBarrier (-(3 / 2)))
    (DDtaud := fun s => meanExitTimeDeriv2 s cubicBarrier cubicBarrierDeriv (-(3 / 2)))
    (taul := fun s => meanExitTime (s + jac2) cubicBarrier (-(3 / 2)) (3 / 2))
    (Dtaul := fun s => meanExitTimeDeriv (s + jac2) cubicBarrier (-(3 / 2)))
    (DDtaul := fun s => meanExitTimeDeriv2 (s + jac2) cubicBarrier cubicBarrierDeriv (-(3 / 2)))
    (fun _ hs =>
      meanExitTime_satisfies_bvp hs hab continuousOn_cubicBarrier hasDerivAt_cubicBarrier)
    (fun _ hs => meanExitTime_satisfies_bvp (by linarith) hab continuousOn_cubicBarrier
      hasDerivAt_cubicBarrier)
    (fun s _ => continuousOn_meanExitTime s hab.le continuousOn_cubicBarrier)
    (fun s _ => continuousOn_meanExitTime (s + jac2) hab.le continuousOn_cubicBarrier)
    hjac continuousOn_cubicBarrier hasDerivAt_cubicBarrier (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) cubicBarrier_pos cubicBarrier_le_top
    cubicBarrier_bottom_le

end IcnnLift
