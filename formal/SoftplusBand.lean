import Mathlib
import Assumptions
import ShoulderAttenuation

/-!
# (A2) for the softplus positivity map: the single-prefactor tolerance with an explicit constant

`Assumptions.lean` carries the paper's assumption (A2) -- "the operative region is a narrow band
of the softplus shoulder over which `psi'` varies slowly", written in the appendix as
`psi'(thetatilde) ~ sigma_s I_d` -- as the structure `SinglePrefactor psi B s eps`, whose
substantive field is `approx : ∀ w ∈ B, |deriv psi w - s| ≤ eps`. That module proves (A2)
satisfiable for an arbitrary `C¹` non-decreasing positivity map (`exists_singlePrefactor_band`), but only
in the shape "fix the tolerance `eps`, and continuity of `psi'` produces *some* band width
`delta > 0`". No constant is exhibited, and the width produced is not quantified.

For the positivity map the paper actually trains, `psi(w) = log (1 + exp w)`, the tolerance is not an
assumption at all. It is a theorem with an explicit constant, and this module proves it in two
strengths.

The **absolute** strength comes from the curvature bound. The first derivative of softplus is the
logistic `sigma(w) = exp w / (1 + exp w)`, and the second is `sigma (1 - sigma)`, which is bounded
by `1/4` everywhere because `sigma(1-sigma) = 1/4 - (sigma - 1/2)²`. The mean value inequality
turns that into a global Lipschitz bound on `psi'` with constant `1/4`, so on a band of half-width
`delta` around any centre `w_s` one may take `s = psi'(w_s)` and `eps = delta / 4`. This is
`singlePrefactor_softplus_closedBall`, and it is the explicit `eps(delta) = delta/4` that
discharges (A2) for softplus.

The absolute strength is, however, the wrong currency on the shoulder, and this module says so
with a theorem rather than a caveat. Every downstream consumer of (A2) uses the *relative* width
`eps / s`: `SinglePrefactor.tol_lt` demands `eps < s`, `SinglePrefactor.deriv_ge` produces the
ellipticity constant `s - eps` that `deriv_pos` turns into strict positivity, and
`SinglePrefactor.diffusion_bracket` produces the pair `(s - eps)², (s + eps)²` that
`BarrierRescaling.quasipotential_gap_bounds` consumes as `c_min` and `c_max`. On the deep shoulder
`s = sigma(w_s)` is exponentially small, so `eps = delta/4` overwhelms it:
`absolute_tolerance_exceeds_prefactor_atBot` proves that for *every* fixed half-width `delta > 0`
the absolute instance fails, eventually in `w_s`, on the `tol_lt` field alone. Read literally, the
`delta/4` bracket is useless exactly where the paper operates.

The **relative** strength repairs this, and is the main content of the module. The sharper
curvature fact is `psi'' ≤ psi'`, that is `sigma(1 - sigma) ≤ sigma`, which integrates to the
exact two-sided ratio bracket `exp (-delta) * sigma(w_s) ≤ sigma(w) ≤ exp delta * sigma(w_s)` for
`|w - w_s| ≤ delta` (proved here algebraically, from `sigma(w) (1 + exp w) = exp w`, not by a
Gronwall argument). Consequently the half-width `delta = log (1 + rho)` delivers relative
tolerance exactly `rho` at *every* centre: `singlePrefactor_softplus_relative` produces
`SinglePrefactor softplus (closedBall w_s (log (1+rho))) (sigma w_s) (rho * sigma w_s)` for any
`0 < rho < 1` and any `w_s` whatsoever. The admissible half-width is a constant independent of the
depth of the shoulder, whereas the absolute route allows only `4 rho sigma(w_s)`, which collapses
to zero; `relative_over_absolute_halfWidth_tendsto_atTop` states the divergence of the ratio. This
is the precise form of "on the deep shoulder the band can be taken wide in absolute terms while
`eps/s` stays small".

A short final section treats the *other* `eps` of the development, the one
`LogDensityCurvature.abs_logDensityCurvature_sub_leading_le` carries as
`|(log sigma_eff^2)''(w)| <= eps` in place of exact constancy of the effective diffusion. If the
objective diffusion is constant then `sigma_eff^2 = psi'^2 sigma_obj^2`, and
`deriv_deriv_log_softplusDiffusion` computes that second derivative exactly, as
`-2 sigma(w) (1 - sigma(w))`. It is bounded by `1/2` unconditionally and by `2 s0` on the shoulder
`{w | sigma(w) < s0}`, so that hypothesis too becomes a theorem with an explicit constant for this
positivity map, and one that shrinks with the depth of the shoulder. This module does not import
`LogDensityCurvature.lean`, so that its own import surface stays at `Assumptions` and
`ShoulderAttenuation`; the bound is stated in exactly the shape that module's `hvar` hypothesis
takes, so that `abs_logDensityCurvature_sub_leading_le` applies to it directly with
`sigma2 := softplusDiffusion sigma_obj^2` and `eps := 2 s0`.

The module closes with the instantiation the audit asked for.
`exists_singlePrefactor_softplus_inside_shoulder` produces, for any shoulder level `s0 > 0` and
any relative tolerance `rho ∈ (0,1)`, a centre `w_s` such that the whole band of half-width
`log (1 + rho)` lies inside `ShoulderAttenuation.shoulder s0` *and* carries a `SinglePrefactor`
term. Its band is provably not a single point (`band_not_subsingleton`), which matters because
`SinglePrefactor.subsingleton_of_tol_zero` shows that the zero-tolerance reading of (A2) forces
exactly that degeneracy for this positivity map.

What this module does **not** prove. It says nothing about *where training goes*: that the
operative iterates actually sit on a shoulder band, at some `w_s` with `sigma(w_s)` small, is a
claim about the optimizer and the data and is not addressed here or anywhere in this development.
It says nothing about the vector-valued form of (A2): the paper's `psi'(thetatilde) ~ sigma_s I_d`
asserts that a single scalar serves *across coordinates*, and the theorems here are
one-dimensional statements about one coordinate's pre-activation, so the cross-coordinate claim
reduces to all coordinates' pre-activations lying in a common band -- again a statement about
training, not about `psi`. And it does not touch the remaining hypotheses of the development:
`SinglePrefactor` also carries `contDiff` and `mono`, which are discharged here for softplus, but
(A1), (A3)-(A6) are untouched.

## Results
* `logistic_eq_sigmoid` — the module's `logistic` is mathlib's `Real.sigmoid`, so mathlib's
  sigmoid calculus applies verbatim.
* `contDiff_softplus`, `contDiff_one_softplus`, `differentiable_softplus` — softplus is
  real-analytic, hence in particular `C¹`, which is `SinglePrefactor.contDiff`.
* `hasDerivAt_logistic`, `deriv_logistic`, `deriv_deriv_softplus` — `psi'' = sigma (1 - sigma)`.
* `deriv_logistic_pos`, `deriv_logistic_le_quarter`, `abs_deriv_deriv_softplus_le_quarter` —
  `0 < psi'' ≤ 1/4` everywhere.
* `deriv_deriv_softplus_zero` — the constant `1/4` is attained, at `w = 0`, so it cannot be
  lowered.
* `abs_logistic_sub_le_quarter_mul` — the mean value consequence `|psi'(w) - psi'(v)| ≤ |w-v|/4`.
* `softplus_prefactor_bracket` — the (A2) bracket `s - delta/4 ≤ psi' ≤ s + delta/4` on
  `[w_s - delta, w_s + delta]` with `s = psi'(w_s)`.
* `singlePrefactor_softplus_closedBall` — (A2) for softplus with `eps(delta) = delta/4`.
* `absolute_tolerance_exceeds_prefactor_atBot` — that instance fails eventually in `w_s`, for
  every fixed `delta > 0`, on the `tol_lt` field.
* `deriv_deriv_softplus_le_deriv_softplus` — the relative curvature bound `psi'' ≤ psi'`.
* `logistic_le_exp_mul_of_le`, `exp_mul_le_logistic_of_le`, `logistic_relative_bracket` — the
  exact ratio bracket `exp(-delta) s ≤ psi'(w) ≤ exp(delta) s` on the band.
* `abs_logistic_sub_le_relative` — `|psi'(w) - s| ≤ (exp delta - 1) s`.
* `singlePrefactor_softplus_relative` — (A2) at relative tolerance `rho`, half-width
  `log (1 + rho)`, at every centre.
* `softplus_diffusion_bracket_relative` — the `c_min`, `c_max` pair `((1∓rho) s)²` that
  `BarrierRescaling` consumes, with ratio independent of the centre.
* `absolute_halfWidth_of_relative_tolerance`, `relative_halfWidth_of_relative_tolerance`,
  `relative_over_absolute_halfWidth_tendsto_atTop` — the two admissible half-widths for a target
  relative tolerance, and the divergence of their ratio down the shoulder.
* `band_subset_shoulder`, `band_not_subsingleton`,
  `exists_singlePrefactor_softplus_inside_shoulder` — the witness: a fat band inside any
  prescribed shoulder level carrying a `SinglePrefactor` term.
* `singlePrefactor_softplus_tol_zero_subsingleton` — the zero-tolerance reading of (A2) is
  degenerate for this positivity map.
* `softplusDiffusion`, `softplusDiffusion_eq_deriv_sq`, `softplusDiffusion_pos` — the effective
  diffusion `psi'(w)² sigma_obj²` of the shoulder band.
* `deriv_deriv_log_softplusDiffusion` — `(log sigma_eff²)''(w) = -2 sigma(w) (1 - sigma(w))`.
* `abs_deriv_deriv_log_softplusDiffusion_le_half`,
  `abs_deriv_deriv_log_softplusDiffusion_le_shoulder` — the diffusion-variation `eps` of
  `LogDensityCurvature.abs_logDensityCurvature_sub_leading_le` is at most `1/2` everywhere and at
  most `2 s0` on the shoulder `{w | sigma(w) < s0}`.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

This module carries **no** analytic hypothesis of its own: every theorem below is either an
unconditional fact about `log (1 + exp w)` or a fact conditioned on explicit numerical
side-conditions (`0 ≤ delta`, `0 < rho < 1`, `delta < 4 * sigma(w_s)`) that are arithmetic, not
analytic. In that sense (A2) is fully discharged for softplus, and what remains assumed in the
paper is only the location of the operative band, which is a statement about training. The
`SinglePrefactor` structure also asks for `Monotone psi`, discharged here by
`ShoulderAttenuation.softplus_monotone`, and `ContDiff ℝ 1 psi`, discharged by `contDiff_softplus`.

Two duplications are deliberate and are recorded rather than removed. First, `softplus` and
`logistic` are `ShoulderAttenuation.lean`'s definitions, imported rather than restated;
`logistic_eq_sigmoid` identifies the latter with mathlib's `Real.sigmoid`
(`Mathlib/Analysis/SpecialFunctions/Sigmoid.lean`), and the derivative facts are then mathlib's
`Real.hasDerivAt_sigmoid` and `Real.deriv_sigmoid` rather than fresh computations. Second, the
`1/4` curvature bound is proved here and not in `ShoulderAttenuation.lean`, which stops at the
first derivative; nothing in that module is edited.
-/

open Filter Topology Set

namespace IcnnLift

/-! ## The positivity map is mathlib's sigmoid, and it is smooth -/

/-- The positivity-map derivative used throughout `ShoulderAttenuation.lean`, `exp w / (1 + exp w)`, is
mathlib's `Real.sigmoid`, defined there as `(1 + exp (-w))⁻¹`. Recording the identification once
lets every sigmoid fact in `Mathlib/Analysis/SpecialFunctions/Sigmoid.lean` be used below without
reproving it. -/
theorem logistic_eq_sigmoid : logistic = Real.sigmoid := by
  funext w
  rw [logistic_def, Real.sigmoid_def, Real.exp_neg]
  have h : (0 : ℝ) < Real.exp w := Real.exp_pos w
  field_simp
  ring

/-- The softplus positivity map is real-analytic: `ContDiff ℝ ⊤` at the top of `WithTop ℕ∞` is the
analytic class `ω`. It is the composition of the analytic `w ↦ 1 + exp w`, which never vanishes,
with `Real.log`. -/
theorem contDiff_softplus : ContDiff ℝ (⊤ : WithTop ℕ∞) softplus := by
  have hbase : ContDiff ℝ (⊤ : WithTop ℕ∞) (fun w : ℝ => 1 + Real.exp w) :=
    contDiff_const.add Real.contDiff_exp
  exact hbase.log fun w => (one_add_exp_pos w).ne'

/-- The `C¹` regularity that `SinglePrefactor.contDiff` asks for. -/
theorem contDiff_one_softplus : ContDiff ℝ 1 softplus := contDiff_softplus.of_le le_top

theorem differentiable_softplus : Differentiable ℝ softplus :=
  contDiff_one_softplus.differentiable_one

/-! ## The second derivative and the `1/4` curvature bound -/

/-- The logistic solves its own logistic equation: `sigma' = sigma (1 - sigma)`. This is
`Real.hasDerivAt_sigmoid` transported along `logistic_eq_sigmoid`. -/
theorem hasDerivAt_logistic (w : ℝ) :
    HasDerivAt logistic (logistic w * (1 - logistic w)) w := by
  rw [logistic_eq_sigmoid]
  exact Real.hasDerivAt_sigmoid w

theorem deriv_logistic (w : ℝ) : deriv logistic w = logistic w * (1 - logistic w) :=
  (hasDerivAt_logistic w).deriv

theorem differentiable_logistic : Differentiable ℝ logistic :=
  fun w => (hasDerivAt_logistic w).differentiableAt

/-- **The second derivative of the softplus positivity map.** `psi''(w) = sigma(w) (1 - sigma(w))`, where
`sigma = psi'` is the logistic. -/
theorem deriv_deriv_softplus (w : ℝ) :
    deriv (deriv softplus) w = logistic w * (1 - logistic w) := by
  rw [deriv_softplus_eq_logistic, deriv_logistic]

/-- The curvature is strictly positive: softplus is strictly convex. -/
theorem deriv_logistic_pos (w : ℝ) : 0 < deriv logistic w := by
  rw [deriv_logistic]
  have h1 := logistic_pos w
  have h2 := logistic_lt_one w
  nlinarith

/-- **The curvature bound.** `sigma (1 - sigma) = 1/4 - (sigma - 1/2)² ≤ 1/4`, with equality only
where `sigma = 1/2`, that is at `w = 0`. -/
theorem deriv_logistic_le_quarter (w : ℝ) : deriv logistic w ≤ 1 / 4 := by
  rw [deriv_logistic]
  nlinarith [sq_nonneg (logistic w - 1 / 2)]

/-- **The curvature bound is sharp.** At `w = 0` the logistic equals `1/2`, so `psi''(0) = 1/4`
exactly and no smaller constant serves in `abs_deriv_deriv_softplus_le_quarter`, nor a smaller
`eps(delta)` than `delta/4` in `softplus_prefactor_bracket`. -/
theorem deriv_deriv_softplus_zero : deriv (deriv softplus) 0 = 1 / 4 := by
  rw [deriv_deriv_softplus, logistic_zero]
  norm_num

theorem abs_deriv_logistic_le_quarter (w : ℝ) : |deriv logistic w| ≤ 1 / 4 :=
  abs_le.mpr ⟨by linarith [deriv_logistic_pos w], deriv_logistic_le_quarter w⟩

/-- **`|psi''| ≤ 1/4` everywhere**, stated for the positivity map rather than for its derivative. -/
theorem abs_deriv_deriv_softplus_le_quarter (w : ℝ) : |deriv (deriv softplus) w| ≤ 1 / 4 := by
  rw [deriv_softplus_eq_logistic]
  exact abs_deriv_logistic_le_quarter w

/-! ## The absolute (A2) bracket: `eps(delta) = delta / 4` -/

/-- The mean value consequence of `|psi''| ≤ 1/4`: the positivity-map derivative is `1/4`-Lipschitz on
all of `ℝ`. -/
theorem abs_logistic_sub_le_quarter_mul (w v : ℝ) : |logistic w - logistic v| ≤ |w - v| / 4 := by
  have hdiff : ∀ x ∈ (Set.univ : Set ℝ), DifferentiableAt ℝ logistic x :=
    fun x _ => differentiable_logistic x
  have hbound : ∀ x ∈ (Set.univ : Set ℝ), ‖deriv logistic x‖ ≤ (1 : ℝ) / 4 := by
    intro x _
    rw [Real.norm_eq_abs]
    exact abs_deriv_logistic_le_quarter x
  have h := convex_univ.norm_image_sub_le_of_norm_deriv_le hdiff hbound
    (Set.mem_univ v) (Set.mem_univ w)
  rw [Real.norm_eq_abs, Real.norm_eq_abs] at h
  linarith

/-- **The (A2) bracket for softplus, absolute form.** With `s = psi'(w_s)`, the positivity-map derivative
on the band `[w_s - delta, w_s + delta]` is trapped between `s - delta/4` and `s + delta/4`. The
tolerance `eps = delta/4` is explicit and is the paper's "varies slowly" made quantitative. -/
theorem softplus_prefactor_bracket {ws delta w : ℝ} (hw : |w - ws| ≤ delta) :
    logistic ws - delta / 4 ≤ deriv softplus w ∧ deriv softplus w ≤ logistic ws + delta / 4 := by
  rw [deriv_softplus_eq_logistic]
  have h := abs_logistic_sub_le_quarter_mul w ws
  have h2 : |w - ws| / 4 ≤ delta / 4 := by linarith
  have h3 := abs_le.mp (h.trans h2)
  exact ⟨by linarith [h3.1], by linarith [h3.2]⟩

/-- **(A2) holds for softplus with the explicit tolerance `delta / 4`.** The single side-condition
`delta < 4 * sigma(w_s)` is exactly `SinglePrefactor.tol_lt`, `eps < s`; nothing analytic is
assumed. -/
theorem singlePrefactor_softplus_closedBall {ws delta : ℝ} (hd : 0 ≤ delta)
    (hnd : delta < 4 * logistic ws) :
    SinglePrefactor softplus (Metric.closedBall ws delta) (logistic ws) (delta / 4) where
  contDiff := contDiff_one_softplus
  mono := softplus_monotone
  prefactor_pos := logistic_pos ws
  tol_nonneg := by linarith
  tol_lt := by linarith
  approx := by
    intro w hw
    rw [deriv_softplus_eq_logistic]
    have hw' : |w - ws| ≤ delta := by
      rw [Metric.mem_closedBall, Real.dist_eq] at hw
      exact hw
    exact (abs_logistic_sub_le_quarter_mul w ws).trans (by linarith)

/-- **Why the absolute bracket is the wrong currency on the shoulder.** Fix any half-width
`delta > 0`. Because `sigma(w_s) → 0` as `w_s → -∞`, eventually `sigma(w_s) ≤ delta/4`, and then
the instance of `SinglePrefactor` with prefactor `sigma(w_s)` and tolerance `delta/4` cannot exist
at all: its `tol_lt` field, `eps < s`, is false. So no fixed absolute half-width survives down the
shoulder, and every consumer of (A2) that reads `s - eps` as an ellipticity constant would read
zero. -/
theorem absolute_tolerance_exceeds_prefactor_atBot {delta : ℝ} (hd : 0 < delta) :
    ∀ᶠ ws in atBot,
      ¬ SinglePrefactor softplus (Metric.closedBall ws delta) (logistic ws) (delta / 4) := by
  have hq : (0 : ℝ) < delta / 4 := by linarith
  filter_upwards [tendsto_logistic_atBot.eventually_lt_const hq] with ws hws h
  exact absurd h.tol_lt (not_lt.mpr hws.le)

/-! ## The relative curvature bound and the exact ratio bracket -/

/-- **The relative curvature bound `psi'' ≤ psi'`.** Since `psi'' = sigma (1 - sigma)` and
`0 < 1 - sigma < 1`, the second derivative is dominated by the first, so the logarithmic
derivative of `psi'` is bounded by one. This, and not the `1/4` bound, is what controls the
*relative* variation of the prefactor. -/
theorem deriv_deriv_softplus_le_deriv_softplus (w : ℝ) :
    deriv (deriv softplus) w ≤ deriv softplus w := by
  rw [deriv_deriv_softplus, deriv_softplus_eq_logistic]
  have h1 := logistic_pos w
  have h2 := logistic_lt_one w
  nlinarith

/-- The algebraic form of the ratio bound, upward direction: for `v ≤ w`,
`sigma(w) ≤ exp (w - v) sigma(v)`. Both sides equal `exp w / (1 + exp ·)` at the two centres, so
the inequality is monotonicity of `1 + exp`. -/
theorem logistic_le_exp_mul_of_le {v w : ℝ} (h : v ≤ w) :
    logistic w ≤ Real.exp (w - v) * logistic v := by
  have hv : (0 : ℝ) < 1 + Real.exp v := one_add_exp_pos v
  have hkey : Real.exp (w - v) * logistic v = Real.exp w / (1 + Real.exp v) := by
    rw [logistic_def, Real.exp_sub]
    have hne : Real.exp v ≠ 0 := (Real.exp_pos v).ne'
    field_simp
  rw [hkey, logistic_def]
  have hmono : Real.exp v ≤ Real.exp w := Real.exp_le_exp.mpr h
  exact div_le_div_of_nonneg_left (Real.exp_pos w).le hv (by linarith)

/-- The algebraic form of the ratio bound, downward direction: for `w ≤ v`,
`exp (w - v) sigma(v) ≤ sigma(w)`. -/
theorem exp_mul_le_logistic_of_le {v w : ℝ} (h : w ≤ v) :
    Real.exp (w - v) * logistic v ≤ logistic w := by
  have hw : (0 : ℝ) < 1 + Real.exp w := one_add_exp_pos w
  have hkey : Real.exp (w - v) * logistic v = Real.exp w / (1 + Real.exp v) := by
    rw [logistic_def, Real.exp_sub]
    have hne : Real.exp v ≠ 0 := (Real.exp_pos v).ne'
    field_simp
  rw [hkey, logistic_def]
  have hmono : Real.exp w ≤ Real.exp v := Real.exp_le_exp.mpr h
  exact div_le_div_of_nonneg_left (Real.exp_pos w).le hw (by linarith)

/-- **The exact ratio bracket on a band.** For `|w - w_s| ≤ delta`,
`exp (-delta) sigma(w_s) ≤ sigma(w) ≤ exp delta sigma(w_s)`. This is the integrated form of
`psi'' ≤ psi'`, and its two constants do not depend on `w_s`. -/
theorem logistic_relative_bracket {ws delta w : ℝ} (hd : 0 ≤ delta) (hw : |w - ws| ≤ delta) :
    Real.exp (-delta) * logistic ws ≤ logistic w ∧
      logistic w ≤ Real.exp delta * logistic ws := by
  obtain ⟨hlo, hhi⟩ := abs_le.mp hw
  have hs : 0 < logistic ws := logistic_pos ws
  constructor
  · rcases le_total w ws with hle | hge
    · refine le_trans ?_ (exp_mul_le_logistic_of_le hle)
      have : Real.exp (-delta) ≤ Real.exp (w - ws) := Real.exp_le_exp.mpr (by linarith)
      exact mul_le_mul_of_nonneg_right this hs.le
    · have h1 : logistic ws ≤ logistic w := logistic_strictMono.monotone hge
      have h2 : Real.exp (-delta) ≤ 1 := by
        rw [Real.exp_le_one_iff]; linarith
      nlinarith
  · rcases le_total ws w with hge | hle
    · refine le_trans (logistic_le_exp_mul_of_le hge) ?_
      have : Real.exp (w - ws) ≤ Real.exp delta := Real.exp_le_exp.mpr (by linarith)
      exact mul_le_mul_of_nonneg_right this hs.le
    · have h1 : logistic w ≤ logistic ws := logistic_strictMono.monotone hle
      have h2 : (1 : ℝ) ≤ Real.exp delta := Real.one_le_exp hd
      nlinarith

/-- **The (A2) tolerance in relative form.** On a band of half-width `delta` the positivity-map
derivative deviates from its central value by at most `(exp delta - 1)` times that central value.
The tolerance is proportional to the prefactor, so the ratio `eps / s = exp delta - 1` is a
function of the half-width alone. -/
theorem abs_logistic_sub_le_relative {ws delta w : ℝ} (hd : 0 ≤ delta) (hw : |w - ws| ≤ delta) :
    |logistic w - logistic ws| ≤ (Real.exp delta - 1) * logistic ws := by
  obtain ⟨hlo, hhi⟩ := logistic_relative_bracket hd hw
  have hs : 0 < logistic ws := logistic_pos ws
  have hprod : Real.exp delta * Real.exp (-delta) = 1 := by
    rw [← Real.exp_add]; simp
  have hep : (0 : ℝ) < Real.exp delta := Real.exp_pos delta
  have hcosh : 2 ≤ Real.exp delta + Real.exp (-delta) := by
    nlinarith [sq_nonneg (Real.exp delta - 1)]
  refine abs_le.mpr ⟨?_, by nlinarith⟩
  have hstep : (1 - Real.exp (-delta)) * logistic ws ≤ (Real.exp delta - 1) * logistic ws := by
    apply mul_le_mul_of_nonneg_right _ hs.le
    linarith
  nlinarith

/-! ## (A2) at a prescribed relative tolerance, uniformly in the centre -/

/-- **The main instantiation.** For any relative tolerance `rho` strictly between `0` and `1`, and
at **every** centre `w_s` on the real line, the softplus positivity map satisfies (A2) on the band of
half-width `log (1 + rho)` with prefactor `s = sigma(w_s)` and tolerance `eps = rho * s`. The
half-width does not depend on `w_s`: however deep on the shoulder the band sits, its width is the
same fixed positive number, and the relative tolerance `eps / s` is exactly `rho`.

This is the machine-checked statement that the positivity map the paper trains satisfies the
single-prefactor idealization, with the tolerance produced rather than assumed. -/
theorem singlePrefactor_softplus_relative {ws rho : ℝ} (h0 : 0 < rho) (h1 : rho < 1) :
    SinglePrefactor softplus (Metric.closedBall ws (Real.log (1 + rho))) (logistic ws)
      (rho * logistic ws) where
  contDiff := contDiff_one_softplus
  mono := softplus_monotone
  prefactor_pos := logistic_pos ws
  tol_nonneg := mul_nonneg h0.le (logistic_pos ws).le
  tol_lt := by
    have hs := logistic_pos ws
    nlinarith
  approx := by
    intro w hw
    have hlogpos : 0 ≤ Real.log (1 + rho) := (Real.log_pos (by linarith)).le
    have hw' : |w - ws| ≤ Real.log (1 + rho) := by
      rw [Metric.mem_closedBall, Real.dist_eq] at hw
      exact hw
    have hexp : Real.exp (Real.log (1 + rho)) = 1 + rho := Real.exp_log (by linarith)
    rw [deriv_softplus_eq_logistic]
    have := abs_logistic_sub_le_relative hlogpos hw'
    rw [hexp] at this
    simpa using this

/-- The half-width `log (1 + rho)` is positive, so the band is a genuine interval and not a point.
Together with `singlePrefactor_softplus_tol_zero_subsingleton` this is what makes the tolerance
carried by (A2) load-bearing rather than cosmetic. -/
theorem band_not_subsingleton {ws rho : ℝ} (h0 : 0 < rho) :
    ¬ (Metric.closedBall ws (Real.log (1 + rho))).Subsingleton := by
  have hpos : 0 < Real.log (1 + rho) := Real.log_pos (by linarith)
  intro hsub
  have hl : ws - Real.log (1 + rho) ∈ Metric.closedBall ws (Real.log (1 + rho)) := by
    rw [Metric.mem_closedBall, Real.dist_eq]
    rw [abs_of_nonpos (by linarith)]
    linarith
  have hr : ws + Real.log (1 + rho) ∈ Metric.closedBall ws (Real.log (1 + rho)) := by
    rw [Metric.mem_closedBall, Real.dist_eq]
    rw [abs_of_nonneg (by linarith)]
    linarith
  have := hsub hl hr
  linarith

/-- **The `c_min`, `c_max` pair (A2) hands to the barrier module.** Applying
`SinglePrefactor.diffusion_bracket` to the relative instance brackets the attenuated objective
diffusion `psi'(w)² sigma_obj²` between `((1 - rho) s)² sigma_obj²` and `((1 + rho) s)²
sigma_obj²`. The *ratio* of the two constants is `((1+rho)/(1-rho))²`, which depends only on the
relative tolerance and not on the centre; this is the sense in which the shoulder band can be
taken wide without loosening the bracket that
`BarrierRescaling.quasipotential_gap_bounds` consumes. -/
theorem softplus_diffusion_bracket_relative {ws rho : ℝ} (h0 : 0 < rho) (h1 : rho < 1)
    {w : ℝ} (hw : w ∈ Metric.closedBall ws (Real.log (1 + rho))) {sigmaObj2 : ℝ}
    (hobj : 0 ≤ sigmaObj2) :
    ((1 - rho) * logistic ws) ^ 2 * sigmaObj2 ≤ deriv softplus w ^ 2 * sigmaObj2 ∧
      deriv softplus w ^ 2 * sigmaObj2 ≤ ((1 + rho) * logistic ws) ^ 2 * sigmaObj2 := by
  obtain ⟨hlo, hhi⟩ := (singlePrefactor_softplus_relative h0 h1).diffusion_bracket hw hobj
  have e1 : logistic ws - rho * logistic ws = (1 - rho) * logistic ws := by ring
  have e2 : logistic ws + rho * logistic ws = (1 + rho) * logistic ws := by ring
  rw [e1] at hlo
  rw [e2] at hhi
  exact ⟨hlo, hhi⟩

/-! ## Absolute against relative: the width available on the deep shoulder -/

/-- The absolute route reaches relative tolerance `rho` only on bands of half-width at most
`4 rho sigma(w_s)`, because its tolerance is `delta / 4`. -/
theorem absolute_halfWidth_of_relative_tolerance {ws delta rho : ℝ}
    (hle : delta ≤ 4 * (rho * logistic ws)) :
    ∀ w ∈ Metric.closedBall ws delta, |deriv softplus w - logistic ws| ≤ rho * logistic ws := by
  intro w hw
  rw [Metric.mem_closedBall, Real.dist_eq] at hw
  rw [deriv_softplus_eq_logistic]
  exact (abs_logistic_sub_le_quarter_mul w ws).trans (by linarith)

/-- The relative route reaches the same relative tolerance `rho` on a band of half-width
`log (1 + rho)`, at every centre, with no smallness condition on `sigma(w_s)`. -/
theorem relative_halfWidth_of_relative_tolerance {ws rho : ℝ} (h0 : 0 < rho) :
    ∀ w ∈ Metric.closedBall ws (Real.log (1 + rho)),
      |deriv softplus w - logistic ws| ≤ rho * logistic ws := by
  intro w hw
  have hlogpos : 0 ≤ Real.log (1 + rho) := (Real.log_pos (by linarith)).le
  have hw' : |w - ws| ≤ Real.log (1 + rho) := by
    rw [Metric.mem_closedBall, Real.dist_eq] at hw
    exact hw
  have hexp : Real.exp (Real.log (1 + rho)) = 1 + rho := Real.exp_log (by linarith)
  rw [deriv_softplus_eq_logistic]
  have := abs_logistic_sub_le_relative hlogpos hw'
  rw [hexp] at this
  simpa using this

/-- **The precise sense in which the band may be taken wide on the deep shoulder.** Fix a target
relative tolerance `rho > 0`. The half-width the relative route certifies is the constant
`log (1 + rho)`; the half-width the absolute route certifies is `4 rho sigma(w_s)`. Their ratio
diverges as the centre descends the shoulder, so the relative bound buys an unbounded factor in
admissible band width at a fixed relative tolerance. -/
theorem relative_over_absolute_halfWidth_tendsto_atTop {rho : ℝ} (h0 : 0 < rho) :
    Tendsto (fun ws : ℝ => Real.log (1 + rho) / (4 * rho * logistic ws)) atBot atTop := by
  have hlog : 0 < Real.log (1 + rho) := Real.log_pos (by linarith)
  have hc : 0 < Real.log (1 + rho) / (4 * rho) := by positivity
  have h := Tendsto.const_mul_atTop hc tendsto_inv_logistic_atBot
  refine h.congr fun ws => ?_
  have hpos : logistic ws ≠ 0 := (logistic_pos ws).ne'
  field_simp

/-! ## The witness: a fat band inside a prescribed shoulder -/

/-- If the centre is deep enough that `(1 + rho) sigma(w_s) < s0`, the whole band of half-width
`log (1 + rho)` lies inside the shoulder `{w | sigma(w) < s0}` of `ShoulderAttenuation.lean`. The
band therefore sits where the paper says the operative region sits, and not merely near it. -/
theorem band_subset_shoulder {ws rho s0 : ℝ} (h0 : 0 < rho)
    (h : (1 + rho) * logistic ws < s0) :
    Metric.closedBall ws (Real.log (1 + rho)) ⊆ shoulder s0 := by
  intro w hw
  have hlogpos : 0 ≤ Real.log (1 + rho) := (Real.log_pos (by linarith)).le
  have hw' : |w - ws| ≤ Real.log (1 + rho) := by
    rw [Metric.mem_closedBall, Real.dist_eq] at hw
    exact hw
  have hexp : Real.exp (Real.log (1 + rho)) = 1 + rho := Real.exp_log (by linarith)
  have hb := (logistic_relative_bracket hlogpos hw').2
  rw [hexp] at hb
  exact lt_of_le_of_lt hb h

/-- **The instantiation asked for by the audit.** For any shoulder level `s0 > 0` and any relative
tolerance `rho` strictly between `0` and `1` there is a centre `w_s` such that the band of
half-width `log (1 + rho)` around it lies entirely inside the shoulder, is not a single point, and
carries a `SinglePrefactor` term for the softplus positivity map with prefactor `sigma(w_s)` and
tolerance `rho * sigma(w_s)`.

Everything in the conclusion is produced: the `C¹` regularity, the monotonicity, the positivity of
the prefactor, the tolerance and its strict domination by the prefactor. Nothing is assumed about
`psi`. What the theorem does not and cannot say is that training visits this band; that is the
residue of (A2) and it is a statement about the optimizer. -/
theorem exists_singlePrefactor_softplus_inside_shoulder {s0 rho : ℝ} (hs0 : 0 < s0)
    (h0 : 0 < rho) (h1 : rho < 1) :
    ∃ ws : ℝ, Metric.closedBall ws (Real.log (1 + rho)) ⊆ shoulder s0 ∧
      ¬ (Metric.closedBall ws (Real.log (1 + rho))).Subsingleton ∧
      SinglePrefactor softplus (Metric.closedBall ws (Real.log (1 + rho))) (logistic ws)
        (rho * logistic ws) := by
  have hden : 0 < s0 / (1 + rho) := by positivity
  obtain ⟨ws, hws⟩ := (tendsto_logistic_atBot.eventually_lt_const hden).exists
  refine ⟨ws, band_subset_shoulder h0 ?_, band_not_subsingleton h0,
    singlePrefactor_softplus_relative h0 h1⟩
  have hone : (0 : ℝ) < 1 + rho := by linarith
  rw [lt_div_iff₀ hone] at hws
  linarith [hws]

/-! ## The other `eps`: the variation of the effective diffusion in (L1') -/

/-- The effective diffusion the lift produces on the shoulder when the objective diffusion is a
constant `sigma_obj^2`: `sigma_eff^2(w) = psi'(w)^2 sigma_obj^2`. This is the `sigma2` argument
that `LogDensityCurvature.lean` and `BarrierRescaling.lean` carry abstractly. -/
noncomputable def softplusDiffusion (sigmaObj2 w : ℝ) : ℝ := logistic w ^ 2 * sigmaObj2

theorem softplusDiffusion_def (sigmaObj2 w : ℝ) :
    softplusDiffusion sigmaObj2 w = logistic w ^ 2 * sigmaObj2 := rfl

theorem softplusDiffusion_eq_deriv_sq (sigmaObj2 w : ℝ) :
    softplusDiffusion sigmaObj2 w = deriv softplus w ^ 2 * sigmaObj2 := by
  rw [softplusDiffusion_def, deriv_softplus_eq_logistic]

theorem softplusDiffusion_pos {sigmaObj2 : ℝ} (hc : 0 < sigmaObj2) (w : ℝ) :
    0 < softplusDiffusion sigmaObj2 w := by
  rw [softplusDiffusion_def]
  have := logistic_pos w
  positivity

/-- The logarithmic derivative of the effective diffusion is `2 (1 - sigma)`: the constant
`sigma_obj^2` drops out of `log`, and `(log sigma)' = sigma''/sigma' = 1 - sigma` by the logistic
equation. -/
theorem hasDerivAt_log_softplusDiffusion {sigmaObj2 : ℝ} (hc : 0 < sigmaObj2) (w : ℝ) :
    HasDerivAt (fun u => Real.log (softplusDiffusion sigmaObj2 u)) (2 * (1 - logistic w)) w := by
  have hs : 0 < logistic w := logistic_pos w
  have hsq : HasDerivAt (fun u => softplusDiffusion sigmaObj2 u)
      (2 * logistic w ^ (2 - 1) * (logistic w * (1 - logistic w)) * sigmaObj2) w :=
    ((hasDerivAt_logistic w).pow 2).mul_const sigmaObj2
  have hne : softplusDiffusion sigmaObj2 w ≠ 0 := (softplusDiffusion_pos hc w).ne'
  have h := hsq.log hne
  convert h using 1
  rw [softplusDiffusion_def]
  field_simp
  ring

theorem deriv_log_softplusDiffusion {sigmaObj2 : ℝ} (hc : 0 < sigmaObj2) :
    (deriv fun u => Real.log (softplusDiffusion sigmaObj2 u)) = fun w => 2 * (1 - logistic w) :=
  funext fun w => (hasDerivAt_log_softplusDiffusion hc w).deriv

/-- **The second `eps` of the development, computed rather than assumed.**
`LogDensityCurvature.abs_logDensityCurvature_sub_leading_le` carries the hypothesis
`|(log sigma_eff^2)''(w)| ≤ eps` as the un-idealised replacement for exact constancy of the
diffusion on the (A2) band. For the softplus positivity map with a constant objective diffusion that
second derivative is exactly `-2 sigma(w) (1 - sigma(w))` -- the same logistic curvature that
governs the prefactor, doubled and reflected. -/
theorem deriv_deriv_log_softplusDiffusion {sigmaObj2 : ℝ} (hc : 0 < sigmaObj2) (w : ℝ) :
    deriv (deriv fun u => Real.log (softplusDiffusion sigmaObj2 u)) w
      = -(2 * (logistic w * (1 - logistic w))) := by
  rw [deriv_log_softplusDiffusion hc]
  have h := ((hasDerivAt_logistic w).const_sub (1 : ℝ)).const_mul (2 : ℝ)
  rw [h.deriv]
  ring

/-- The unconditional bound on the diffusion-variation `eps`: `|(log sigma_eff^2)''| ≤ 1/2`
everywhere, from `sigma (1 - sigma) ≤ 1/4`. -/
theorem abs_deriv_deriv_log_softplusDiffusion_le_half {sigmaObj2 : ℝ} (hc : 0 < sigmaObj2)
    (w : ℝ) :
    |deriv (deriv fun u => Real.log (softplusDiffusion sigmaObj2 u)) w| ≤ 1 / 2 := by
  rw [deriv_deriv_log_softplusDiffusion hc]
  have h1 := logistic_pos w
  have h2 := logistic_lt_one w
  rw [abs_le]
  constructor
  · nlinarith [sq_nonneg (logistic w - 1 / 2)]
  · nlinarith

/-- **The diffusion-variation `eps` is exponentially small on the shoulder.** On the shoulder
`{w | sigma(w) < s0}` the bound improves from `1/2` to `2 s0`, because
`|(log sigma_eff^2)''| = 2 sigma (1 - sigma) ≤ 2 sigma`. So the input
`LogDensityCurvature.abs_logDensityCurvature_sub_leading_le` carries as a hypothesis is, for this
positivity map on this band, a theorem with the explicit constant `eps = 2 s0`: the deeper the shoulder
the smaller the correction, and the (L1') identity of
`LogDensityCurvature.logDensityCurvature_of_const_diffusion` is recovered in the limit. -/
theorem abs_deriv_deriv_log_softplusDiffusion_le_shoulder {sigmaObj2 s0 : ℝ} (hc : 0 < sigmaObj2)
    {w : ℝ} (hw : w ∈ shoulder s0) :
    |deriv (deriv fun u => Real.log (softplusDiffusion sigmaObj2 u)) w| ≤ 2 * s0 := by
  have hlt : logistic w < s0 := hw
  rw [deriv_deriv_log_softplusDiffusion hc]
  have h1 := logistic_pos w
  have h2 := logistic_lt_one w
  rw [abs_le]
  constructor <;> nlinarith

/-- **The tolerance cannot be set to zero for this positivity map.**
`SinglePrefactor.subsingleton_of_tol_zero` specialized to softplus: since `psi' = sigma` is
strictly increasing and hence injective, a zero-tolerance single-prefactor band is a single point.
Read against `band_not_subsingleton`, this says the relative instance above is not an instance of
the exact idealization in disguise. -/
theorem singlePrefactor_softplus_tol_zero_subsingleton {B : Set ℝ} {s : ℝ}
    (h : SinglePrefactor softplus B s 0) : B.Subsingleton := by
  refine h.subsingleton_of_tol_zero ?_
  rw [deriv_softplus_eq_logistic]
  exact logistic_strictMono.injective

end IcnnLift
