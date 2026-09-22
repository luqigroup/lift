import Mathlib
import BarrierRescaling

/-!
# The quasipotential of equation (*) as an integral, and the discharge of `hUeff`

`BarrierRescaling.lean` machine-checks claim (L2') of `docs/design/lemma1_rederivation.md` §3 —
the barrier rescaling `alpha_eff = alpha / sigma_eff^2` and its two-sided band form — but it does
so with the effective potential `U_eff` left abstract, constrained only by the named hypothesis

  `hUeff : ∀ u ∈ band, HasDerivAt Ueff (Ltilde' u / sigma_eff^2 u) u`.

That hypothesis is the differentiated form of equation (*) of the design note. The note itself
does not leave `U_eff` abstract: it *defines* it, as the indefinite integral
`U_eff(w) = ∫^w Ltilde'(u) / sigma_eff^2(u) du`. Read that way, `hUeff` is not a stochastic-analysis
input at all — it is the fundamental theorem of calculus. This file takes the note's definition
literally, and discharges `hUeff` from it.

The content is therefore a chain of three steps. First, `quasipotential` is the note's integral,
with the lower limit made explicit (the note writes the indefinite `∫^w`, which fixes `U_eff` only
up to an additive constant; every statement below is invariant under that shift, because only
increments of `U_eff` are ever consumed). Second, `hasDerivAt_quasipotential` proves that this
function does satisfy `hUeff`, from continuity of `Ltilde'` and of `sigma_eff^2` together with
strict positivity of `sigma_eff^2` — which is (A1)'s regularity clause and (A2)'s uniform
ellipticity, and nothing more. Third, the capstones restate
`BarrierRescaling.quasipotential_gap_bounds`, `barrier_exponent_bounds` and
`quasipotential_gap_of_const_diffusion` with `quasipotential` substituted for `U_eff` and every
one of their hypotheses — `hUeff`, `hintU` and `hintL` alike — derived from that same continuity
and positivity, so that a caller supplies one regularity package and receives the barrier
rescaling bound.

Three observations about the discharge are worth recording, because they are what an audit of
`hUeff` should turn on.

*The band alone does not suffice; an open window around it is needed.* `BarrierRescaling` asks for
a two-sided `HasDerivAt` at every point of the closed band, its endpoints included. A function
defined only on `[w_b, w_s]` cannot have a two-sided derivative at `w_b`, so `hUeff` implicitly
requires `Ltilde'` and `sigma_eff^2` to be defined, continuous and elliptic on a neighbourhood of
the band. That is why the hypotheses below are stated on an open interval `Ioo a b` containing the
closed band rather than on the band itself. This is a real, if mild, strengthening of the paper's
(A2), and it is stated rather than smuggled.

*`hUeff` is not vacuous, and it is not over-strong either.* It is not vacuous because a solution
exists whenever the regularity holds — that is `hasDerivAt_quasipotential`, and
`quasipotential_gap_bounds_witness` exhibits a concrete instance with a genuinely non-constant
diffusion. It is not over-strong because it pins down exactly the quantity `BarrierRescaling`
consumes and no more: `quasipotential_eq_gap_of_hasDerivAt` shows that *any* `Ueff` satisfying
`hUeff` has `Ueff w_s - Ueff w_b` equal to the integral, so the gap is an invariant of the
hypothesis, and `quasipotential_gap_unique` records the corresponding uniqueness statement. The
additive constant that `hUeff` leaves free never reaches a conclusion.

*Positivity of `sigma_eff^2` is the softplus hypothesis.* In the paper's decomposition
`sigma_eff^2(w) = s^2 sigma_obj^2 + c1 sigma_Jac^2 |H| + s^2 (HVH)` of §2, with `s = psi'(w)` the
positivity-map derivative, every term carries at least one power of `psi'`, so ellipticity is exactly the
statement that the positivity-map derivative does not vanish on the window. The design note's own
ellipticity audit (§11.3, "ELLIPTICITY: RESOLVED") settles this in the paper's favour: assumption
(A2) pins `psi'` near `sigma_s > 0` across the shoulder band, giving the uniform lower bound
`sigma_eff^2 ≥ sigma_s^2 lambda_min(Sigma)`, and the note records that "softplus' > 0 strictly, so
the constant is genuinely positive". For the softplus positivity map the paper trains, `psi' = sigmoid`
is strictly positive on all of `ℝ`, so the hypothesis holds on any window whatsoever; for the
ReLU-type positivity map of the same section it would fail identically on the gated set, and with it the
whole quasipotential construction. The hypothesis is thus satisfied in the paper's setting for a
structural reason, and it is not an idle one.

**What this file does not do.** It does not prove equation (*) itself — that `pi ∝ sigma_eff^{-2}
exp(-(2/eta) U_eff)` is the stationary density of the Itô diffusion with state-dependent noise.
That is a statement about a stochastic differential equation, mathlib v4.31.0 has no Itô calculus,
and it is `StationaryDensity.lean`'s business; what is discharged here is the *differentiated*
form, which is the only form `BarrierRescaling.lean` consumes. Nor does it discharge the other
named hypotheses of (L2'): the monotone-climb condition `hgrad : 0 ≤ Ltilde'` on the band, the
bracket `cmin ≤ sigma_eff^2 ≤ cmax`, the band orientation `w_b ≤ w_s`, and the Kramers relation
itself all remain exactly where `BarrierRescaling.lean` leaves them.

## Results
* `quasipotential` — the note's `U_eff(w) = ∫_{w_0}^{w} Ltilde'(u)/sigma_eff^2(u) du`.
* `barrierPotential` — the pullback potential recovered from its derivative, `∫_{w_0}^{w} Ltilde'`.
* `quasipotential_self`, `quasipotential_symm`, `quasipotential_add` — base-point bookkeeping.
* `hasDerivAt_intervalIntegral_of_continuousOn` — the fundamental theorem of calculus on an open
  order-connected window, the single analytic engine of the file.
* `hasDerivAt_quasipotential` — `hUeff` for `quasipotential`, at one point of the window.
* `hasDerivAt_quasipotential_uIcc`, `hasDerivAt_quasipotential_Icc` — the same in the two binder
  shapes `BarrierRescaling.lean` actually consumes.
* `hasDerivAt_quasipotential_eventually` — the same in the germ shape `LogDensityCurvature.lean`
  consumes for (L1').
* `hasDerivAt_barrierPotential_Icc` — the companion `hLtilde` for `barrierPotential`.
* `intervalIntegrable_of_band`, `intervalIntegrable_div_of_band` — `hintL` and `hintU` from the
  same continuity hypotheses.
* `quasipotential_eq_gap_of_hasDerivAt` — any `Ueff` satisfying `hUeff` has the integral as its
  gap; the hypothesis determines what is consumed.
* `quasipotential_gap_unique` — two solutions of `hUeff` have the same gap.
* `quasipotential_gap_bounds_of_continuity` — the capstone: `BarrierRescaling`'s two-sided band
  bound with `U_eff` the integral and only continuity, positivity and the band data as inputs.
* `quasipotential_gap_bounds_of_continuity_selfBased` — the same at the canonical base point
  `w_0 = w_b`, where the lower bracket reads `alpha / cmax ≤ U_eff(w_s) ≤ alpha / cmin`.
* `barrier_exponent_bounds_of_continuity` — the capstone in Arrhenius-exponent form.
* `quasipotential_gap_of_const_diffusion_of_continuity` — the exact constant-band identity with
  `U_eff` the integral, in the orientation-free `uIcc` form.
* `quasipotential_gap_bounds_fully_discharged` — the end-to-end statement in which `Ltilde` is
  itself the primitive of `Ltilde'`, so that continuity of `Ltilde'`, continuity and ellipticity of
  `sigma_eff^2`, the bracket and the monotone climb are the *only* hypotheses left.
* `quasipotential_example`, `quasipotential_ftc_witness` — a worked instance with a
  genuinely non-constant diffusion, computed in closed form, certifying that the hypothesis set of
  the capstone is jointly satisfiable and that the bracket it yields is strict on both sides.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. No existing module is modified: this file imports
`BarrierRescaling.lean` and consumes its theorems as stated.

The hypotheses this file carries, and what each would take to discharge:

* `hcontL : ContinuousOn dL (Ioo a b)` and `hcontS : ContinuousOn sigma2 (Ioo a b)` — continuity
  of `Ltilde'` and of `sigma_eff^2` on an open window around the band. This is the paper's (A1)
  regularity clause. It is not dischargeable inside this file, because `Ltilde` and `sigma_eff^2`
  are abstract here; in the paper's setting it follows from smoothness of the softplus positivity map and
  of the per-sample loss, together with finiteness of the batch moments that build `sigma_eff^2`.
* `hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u` — uniform ellipticity on the window, the paper's (A2). See
  the third observation above: for the softplus positivity map it holds globally and structurally.
* `hband : Icc wb ws ⊆ Ioo a b` — the window contains the band. Forced by the two-sided
  `HasDerivAt` shape of `hUeff`, as explained above; it cannot be dropped without weakening
  `BarrierRescaling`'s hypothesis to a one-sided `HasDerivWithinAt`.
* `hwb`, `hcmin`, `hlower`, `hupper`, `hgrad`, `hLtilde` — passed through unchanged to
  `BarrierRescaling.lean`, which documents each. Nothing here weakens or strengthens them, with
  the single exception that `hLtilde` disappears from
  `quasipotential_gap_bounds_fully_discharged`, where `Ltilde` is taken to be the primitive of
  `dL` and the fundamental theorem of calculus supplies it.
-/

open Set MeasureTheory

namespace IcnnLift

/-! ### The quasipotential as the note's integral -/

section Definition

/-- The **effective quasipotential** of equation (*) of `docs/design/lemma1_rederivation.md` §3,
`U_eff(w) = ∫^w Ltilde'(u) / sigma_eff^2(u) du`, with the lower limit `w0` made explicit. The
argument `dL` is the paper's `Ltilde'` and `sigma2` is its `sigma_eff^2`.

The note writes an indefinite integral, which determines `U_eff` only up to an additive constant.
Changing `w0` shifts `U_eff` by a constant and rescales the stationary density
`pi ∝ sigma_eff^{-2} exp(-(2/eta) U_eff)` by a positive factor, so it changes neither the
normalized measure nor any barrier gap; every statement below consumes `U_eff` through an
increment, and `quasipotential_add` records the shift explicitly. -/
noncomputable def quasipotential (dL sigma2 : ℝ → ℝ) (w0 w : ℝ) : ℝ :=
  ∫ u in w0..w, dL u / sigma2 u

/-- The **pullback potential recovered from its derivative**, `Ltilde(w) = ∫_{w0}^{w} Ltilde'(u) du`.
This is used only in `quasipotential_gap_bounds_fully_discharged`, where taking `Ltilde` to be a
primitive of `dL` removes the last hypothesis of the capstone that is not continuity, positivity
or band data. -/
noncomputable def barrierPotential (dL : ℝ → ℝ) (w0 w : ℝ) : ℝ :=
  ∫ u in w0..w, dL u

/-- The quasipotential vanishes at its base point: the additive normalization is `U_eff(w0) = 0`. -/
@[simp] theorem quasipotential_self (dL sigma2 : ℝ → ℝ) (w0 : ℝ) :
    quasipotential dL sigma2 w0 w0 = 0 :=
  intervalIntegral.integral_same

/-- Reversing the endpoints negates the quasipotential increment. -/
theorem quasipotential_symm (dL sigma2 : ℝ → ℝ) (w0 w : ℝ) :
    quasipotential dL sigma2 w w0 = -quasipotential dL sigma2 w0 w :=
  intervalIntegral.integral_symm _ _

/-- **The base point is a pure additive constant.** Moving the lower limit from `w0` to `w1`
shifts `U_eff` by the fixed number `quasipotential dL sigma2 w0 w1`, so every increment
`U_eff(w_s) - U_eff(w_b)` — the only way `BarrierRescaling.lean` reads `U_eff` — is independent of
the base point. This is the formal content of the note's indefinite `∫^w`. -/
theorem quasipotential_add {dL sigma2 : ℝ → ℝ} {w0 w1 w : ℝ}
    (h01 : IntervalIntegrable (fun u => dL u / sigma2 u) volume w0 w1)
    (h1w : IntervalIntegrable (fun u => dL u / sigma2 u) volume w1 w) :
    quasipotential dL sigma2 w0 w
      = quasipotential dL sigma2 w0 w1 + quasipotential dL sigma2 w1 w :=
  (intervalIntegral.integral_add_adjacent_intervals h01 h1w).symm

end Definition

/-! ### The fundamental theorem of calculus on an open window -/

section FTC

/-- **The analytic engine of the file: the fundamental theorem of calculus on an open,
order-connected window.** If `f` is continuous on a set `s` that is open and order-connected — an
open interval, in the intended application — then for `w0` and `w` in `s` the primitive
`v ↦ ∫_{w0}^{v} f` is differentiable at `w` with derivative `f w`, in the two-sided `HasDerivAt`
sense.

Order-connectedness is what makes `uIcc w0 w ⊆ s`, hence what makes the integrand continuous, and
therefore interval-integrable, along the whole path of integration; openness is what upgrades a
one-sided derivative at the endpoints to a two-sided one. Both are needed, and both are supplied
by an open interval.

This is `intervalIntegral.integral_hasDerivAt_right` of mathlib, with its integrability and
strong-measurability side conditions discharged from continuity on `s`. The global-continuity
special case is `StationaryDensity.hasDerivAt_effectivePotential`; the point of the local form is
that the paper's `sigma_eff^2` is known to be elliptic only on the shoulder window, not on all
of `ℝ`. -/
theorem hasDerivAt_intervalIntegral_of_continuousOn {f : ℝ → ℝ} {s : Set ℝ}
    (hopen : IsOpen s) (hconn : s.OrdConnected) (hcont : ContinuousOn f s)
    {w0 w : ℝ} (hw0 : w0 ∈ s) (hw : w ∈ s) :
    HasDerivAt (fun v => ∫ u in w0..v, f u) (f w) w := by
  have hsub : uIcc w0 w ⊆ s := hconn.uIcc_subset hw0 hw
  have hint : IntervalIntegrable f volume w0 w := (hcont.mono hsub).intervalIntegrable
  have hmeas : StronglyMeasurableAtFilter f (nhds w) volume :=
    ContinuousOn.stronglyMeasurableAtFilter hopen hcont w hw
  have hca : ContinuousAt f w := hcont.continuousAt (hopen.mem_nhds hw)
  exact intervalIntegral.integral_hasDerivAt_right hint hmeas hca

/-- **`hUeff` for the note's own `U_eff`, at one point of the window.** If `Ltilde'` and
`sigma_eff^2` are continuous on an open, order-connected window `s` on which `sigma_eff^2` is
strictly positive, then the quasipotential of equation (*) is differentiable at each `w ∈ s` with

  `U_eff'(w) = Ltilde'(w) / sigma_eff^2(w)`,

which is exactly the hypothesis `hUeff` that `BarrierRescaling.lean` carries. Strict positivity
enters only through `sigma_eff^2(u) ≠ 0`, which is what keeps the integrand continuous; it is the
paper's uniform ellipticity (A2), and for the softplus positivity map it is automatic. -/
theorem hasDerivAt_quasipotential {dL sigma2 : ℝ → ℝ} {s : Set ℝ}
    (hopen : IsOpen s) (hconn : s.OrdConnected)
    (hcontL : ContinuousOn dL s) (hcontS : ContinuousOn sigma2 s)
    (hpos : ∀ u ∈ s, 0 < sigma2 u) {w0 w : ℝ} (hw0 : w0 ∈ s) (hw : w ∈ s) :
    HasDerivAt (quasipotential dL sigma2 w0) (dL w / sigma2 w) w :=
  hasDerivAt_intervalIntegral_of_continuousOn hopen hconn
    (hcontL.div hcontS fun u hu => (hpos u hu).ne') hw0 hw

/-- `hUeff` in the germ (`∀ᶠ u in 𝓝 w`) shape, which is the form `LogDensityCurvature.lean`
carries for the (L1') curvature identity: there the hypothesis is not needed on a band but on a
neighbourhood of the minimum `w*`, because the identity differentiates `U_eff` twice. Openness of
the window is exactly what supplies it. -/
theorem hasDerivAt_quasipotential_eventually {dL sigma2 : ℝ → ℝ} {s : Set ℝ}
    (hopen : IsOpen s) (hconn : s.OrdConnected)
    (hcontL : ContinuousOn dL s) (hcontS : ContinuousOn sigma2 s)
    (hpos : ∀ u ∈ s, 0 < sigma2 u) {w0 w : ℝ} (hw0 : w0 ∈ s) (hw : w ∈ s) :
    ∀ᶠ u in nhds w, HasDerivAt (quasipotential dL sigma2 w0) (dL u / sigma2 u) u :=
  Filter.eventually_of_mem (hopen.mem_nhds hw) fun _ hu =>
    hasDerivAt_quasipotential hopen hconn hcontL hcontS hpos hw0 hu

/-- `hUeff` in the `uIcc` binder shape, the form consumed by
`BarrierRescaling.quasipotential_gap_of_const_diffusion` and `barrier_exponent_of_const_diffusion`.
The band is required to sit inside the open window `Ioo a b`; see the file docstring for why an
open window, and not the closed band, is what the two-sided `HasDerivAt` demands. -/
theorem hasDerivAt_quasipotential_uIcc {dL sigma2 : ℝ → ℝ} {a b wb ws w0 : ℝ}
    (hband : uIcc wb ws ⊆ Ioo a b) (hw0 : w0 ∈ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u) :
    ∀ u ∈ uIcc wb ws, HasDerivAt (quasipotential dL sigma2 w0) (dL u / sigma2 u) u :=
  fun _ hu =>
    hasDerivAt_quasipotential isOpen_Ioo ordConnected_Ioo hcontL hcontS hpos hw0 (hband hu)

/-- `hUeff` in the `Icc` binder shape, the form consumed by
`BarrierRescaling.quasipotential_gap_bounds`, `barrier_exponent_bounds` and
`quasipotential_gap_bounds_eq_of_const`. -/
theorem hasDerivAt_quasipotential_Icc {dL sigma2 : ℝ → ℝ} {a b wb ws w0 : ℝ}
    (hband : Icc wb ws ⊆ Ioo a b) (hw0 : w0 ∈ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u) :
    ∀ u ∈ Icc wb ws, HasDerivAt (quasipotential dL sigma2 w0) (dL u / sigma2 u) u :=
  fun _ hu =>
    hasDerivAt_quasipotential isOpen_Ioo ordConnected_Ioo hcontL hcontS hpos hw0 (hband hu)

/-- The companion of `hasDerivAt_quasipotential_Icc` for the potential itself: if `Ltilde'` is
continuous on the window, its primitive is an antiderivative of it on the band, which is the
hypothesis `hLtilde` of `BarrierRescaling.quasipotential_gap_bounds`. -/
theorem hasDerivAt_barrierPotential_Icc {dL : ℝ → ℝ} {a b wb ws w0 : ℝ}
    (hband : Icc wb ws ⊆ Ioo a b) (hw0 : w0 ∈ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) :
    ∀ u ∈ Icc wb ws, HasDerivAt (barrierPotential dL w0) (dL u) u :=
  fun _ hu =>
    hasDerivAt_intervalIntegral_of_continuousOn isOpen_Ioo ordConnected_Ioo hcontL hw0 (hband hu)

end FTC

/-! ### The interval-integrability side conditions -/

section Integrability

/-- `hintL` of `BarrierRescaling.quasipotential_gap_bounds`, from continuity of `Ltilde'` on the
window. -/
theorem intervalIntegrable_of_band {dL : ℝ → ℝ} {a b wb ws : ℝ} (hwb : wb ≤ ws)
    (hband : Icc wb ws ⊆ Ioo a b) (hcontL : ContinuousOn dL (Ioo a b)) :
    IntervalIntegrable dL volume wb ws :=
  (hcontL.mono (by rw [uIcc_of_le hwb]; exact hband)).intervalIntegrable

/-- `hintU` of `BarrierRescaling.quasipotential_gap_bounds`, from continuity of `Ltilde'` and of
`sigma_eff^2` on the window together with ellipticity. Note that only nonvanishing of
`sigma_eff^2` is used, not a lower bracket: the bracket `cmin ≤ sigma_eff^2 ≤ cmax` does its work
in the comparison, not in the integrability. -/
theorem intervalIntegrable_div_of_band {dL sigma2 : ℝ → ℝ} {a b wb ws : ℝ} (hwb : wb ≤ ws)
    (hband : Icc wb ws ⊆ Ioo a b) (hcontL : ContinuousOn dL (Ioo a b))
    (hcontS : ContinuousOn sigma2 (Ioo a b)) (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u) :
    IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws :=
  ((hcontL.div hcontS fun u hu => (hpos u hu).ne').mono
    (by rw [uIcc_of_le hwb]; exact hband)).intervalIntegrable

end Integrability

/-! ### What `hUeff` determines, and what it leaves free -/

section Determinacy

variable {dL sigma2 Ueff : ℝ → ℝ} {wb ws : ℝ}

/-- **`hUeff` determines exactly the quantity that is consumed.** Any `Ueff` satisfying the
hypothesis on the band has increment equal to the note's integral:

  `∫_{w_b}^{w_s} Ltilde'(u)/sigma_eff^2(u) du = U_eff(w_s) - U_eff(w_b)`.

So `hUeff` is neither vacuous nor over-strong. It is not vacuous because
`hasDerivAt_quasipotential` produces a solution under ordinary regularity; it is not over-strong
because the additive constant it leaves free is invisible to every conclusion of
`BarrierRescaling.lean`, all of which read `U_eff` through this increment. No continuity is needed
here, only the hypothesis itself and the integrability that accompanies it. -/
theorem quasipotential_eq_gap_of_hasDerivAt
    (hUeff : ∀ u ∈ uIcc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hint : IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws) :
    quasipotential dL sigma2 wb ws = Ueff ws - Ueff wb :=
  intervalIntegral.integral_eq_sub_of_hasDerivAt hUeff hint

/-- **Uniqueness of the barrier gap under `hUeff`.** Two effective potentials satisfying the
hypothesis on the same band have the same gap across it. This is the statement that the barrier
rescaling of (L2') is a property of `Ltilde'` and `sigma_eff^2` alone, and not of the choice of
antiderivative. -/
theorem quasipotential_gap_unique {U1 U2 : ℝ → ℝ}
    (h1 : ∀ u ∈ uIcc wb ws, HasDerivAt U1 (dL u / sigma2 u) u)
    (h2 : ∀ u ∈ uIcc wb ws, HasDerivAt U2 (dL u / sigma2 u) u)
    (hint : IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws) :
    U1 ws - U1 wb = U2 ws - U2 wb := by
  rw [← quasipotential_eq_gap_of_hasDerivAt h1 hint, quasipotential_eq_gap_of_hasDerivAt h2 hint]

end Determinacy

/-! ### The capstones: (L2') with every input derived from regularity -/

section Capstone

variable {Ltilde dL sigma2 : ℝ → ℝ} {a b wb ws w0 cmin cmax : ℝ}

/-- **The discharge, end to end.** `BarrierRescaling.quasipotential_gap_bounds` restated with the
note's own `U_eff` — the integral of equation (*) — in place of the abstract one, and with its
hypotheses `hUeff`, `hintL` and `hintU` all derived. What remains is continuity of `Ltilde'` and of
`sigma_eff^2` on an open window containing the band, uniform ellipticity on that window, the band
bracket `cmin ≤ sigma_eff^2 ≤ cmax` with `cmin > 0`, the monotone climb `0 ≤ Ltilde'`, the
orientation `w_b ≤ w_s`, and the statement that `Ltilde'` is the derivative of `Ltilde`. The
conclusion is (L2')'s two-sided band form,

  `alpha / cmax ≤ U_eff(w_s) - U_eff(w_b) ≤ alpha / cmin`,   `alpha = Ltilde(w_s) - Ltilde(w_b)`.

Only the last remaining hypothesis, `hLtilde`, is not a regularity condition on the two data of
the quasipotential; `quasipotential_gap_bounds_fully_discharged` removes it too, at the cost of
naming `Ltilde` as the primitive of `Ltilde'`. -/
theorem quasipotential_gap_bounds_of_continuity (hwb : wb ≤ ws)
    (hband : Icc wb ws ⊆ Ioo a b) (hw0 : w0 ∈ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u)
    (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u) :
    (Ltilde ws - Ltilde wb) / cmax
        ≤ quasipotential dL sigma2 w0 ws - quasipotential dL sigma2 w0 wb ∧
      quasipotential dL sigma2 w0 ws - quasipotential dL sigma2 w0 wb
        ≤ (Ltilde ws - Ltilde wb) / cmin :=
  quasipotential_gap_bounds hwb hcmin hlower hupper hgrad hLtilde
    (hasDerivAt_quasipotential_Icc hband hw0 hcontL hcontS hpos)
    (intervalIntegrable_of_band hwb hband hcontL)
    (intervalIntegrable_div_of_band hwb hband hcontL hcontS hpos)

/-- The capstone at the canonical base point `w_0 = w_b`, where `U_eff(w_b) = 0` and the bracket
reads directly on the value at the saddle:

  `alpha / cmax ≤ U_eff(w_s) ≤ alpha / cmin`.

This is the form in which the note's `alpha_eff = alpha / sigma_eff^2` is most directly read. -/
theorem quasipotential_gap_bounds_of_continuity_selfBased (hwb : wb ≤ ws)
    (hband : Icc wb ws ⊆ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u)
    (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u) :
    (Ltilde ws - Ltilde wb) / cmax ≤ quasipotential dL sigma2 wb ws ∧
      quasipotential dL sigma2 wb ws ≤ (Ltilde ws - Ltilde wb) / cmin := by
  have hwbmem : wb ∈ Ioo a b := hband (left_mem_Icc.mpr hwb)
  obtain ⟨h1, h2⟩ := quasipotential_gap_bounds_of_continuity (w0 := wb) hwb hband hwbmem
    hcontL hcontS hpos hcmin hlower hupper hgrad hLtilde
  rw [quasipotential_self] at h1 h2
  exact ⟨by linarith, by linarith⟩

/-- The capstone in Arrhenius-exponent form: `BarrierRescaling.barrier_exponent_bounds` with the
note's `U_eff` and every analytic input derived from continuity and ellipticity. This is the
inequality the corrected Corollary 1 compares across the two methods. -/
theorem barrier_exponent_bounds_of_continuity {eta : ℝ} (hwb : wb ≤ ws) (heta : 0 < eta)
    (hband : Icc wb ws ⊆ Ioo a b) (hw0 : w0 ∈ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u)
    (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u) :
    arrheniusExponent (Ltilde ws - Ltilde wb) eta cmax
        ≤ 2 / eta * (quasipotential dL sigma2 w0 ws - quasipotential dL sigma2 w0 wb) ∧
      2 / eta * (quasipotential dL sigma2 w0 ws - quasipotential dL sigma2 w0 wb)
        ≤ arrheniusExponent (Ltilde ws - Ltilde wb) eta cmin :=
  barrier_exponent_bounds hwb heta hcmin hlower hupper hgrad hLtilde
    (hasDerivAt_quasipotential_Icc hband hw0 hcontL hcontS hpos)
    (intervalIntegrable_of_band hwb hband hcontL)
    (intervalIntegrable_div_of_band hwb hband hcontL hcontS hpos)

/-- The exact constant-band identity of (L2')'s first half, with the note's `U_eff` substituted
and `hUeff` and `hintL` discharged: on a band where `sigma_eff^2` is identically `c`, the
quasipotential gap is `alpha / c` exactly. The orientation of the band is unconstrained, matching
`BarrierRescaling.quasipotential_gap_of_const_diffusion`, so the hypotheses are stated on `uIcc`.
Positivity of `c` is not needed for the identity and is not assumed; it enters only through the
window's ellipticity, which the fundamental theorem of calculus does need. -/
theorem quasipotential_gap_of_const_diffusion_of_continuity {c : ℝ}
    (hwindow : uIcc wb ws ⊆ Ioo a b) (hw0 : w0 ∈ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u)
    (hband : ∀ u ∈ uIcc wb ws, sigma2 u = c)
    (hLtilde : ∀ u ∈ uIcc wb ws, HasDerivAt Ltilde (dL u) u) :
    quasipotential dL sigma2 w0 ws - quasipotential dL sigma2 w0 wb
      = (Ltilde ws - Ltilde wb) / c :=
  quasipotential_gap_of_const_diffusion hband hLtilde
    (hasDerivAt_quasipotential_uIcc hwindow hw0 hcontL hcontS hpos)
    ((hcontL.mono hwindow).intervalIntegrable)

/-- **(L2') with nothing left but regularity.** Taking `Ltilde` to be the primitive of `Ltilde'`
removes the last hypothesis of `quasipotential_gap_bounds_of_continuity` that is not a regularity
or band condition: the fundamental theorem of calculus supplies `hLtilde` as well. What is assumed
is then exactly continuity of `Ltilde'`, continuity and strict positivity of `sigma_eff^2` on an
open window containing the band, the bracket `cmin ≤ sigma_eff^2 ≤ cmax` with `cmin > 0`, the
monotone climb `0 ≤ Ltilde'` across the band, and its orientation. Nothing analytic remains.

The barrier action appears here as `barrierPotential dL wb ws = ∫_{w_b}^{w_s} Ltilde'`, which is
`alpha` by the same theorem; naming `Ltilde` this way is a normalization of the additive constant
in `Ltilde` and not a restriction, exactly as with `U_eff`. -/
theorem quasipotential_gap_bounds_fully_discharged (hwb : wb ≤ ws)
    (hband : Icc wb ws ⊆ Ioo a b)
    (hcontL : ContinuousOn dL (Ioo a b)) (hcontS : ContinuousOn sigma2 (Ioo a b))
    (hpos : ∀ u ∈ Ioo a b, 0 < sigma2 u)
    (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u) :
    barrierPotential dL wb ws / cmax ≤ quasipotential dL sigma2 wb ws ∧
      quasipotential dL sigma2 wb ws ≤ barrierPotential dL wb ws / cmin := by
  have hwbmem : wb ∈ Ioo a b := hband (left_mem_Icc.mpr hwb)
  have hL : ∀ u ∈ Icc wb ws, HasDerivAt (barrierPotential dL wb) (dL u) u :=
    hasDerivAt_barrierPotential_Icc hband hwbmem hcontL
  have hzero : barrierPotential dL wb wb = 0 := intervalIntegral.integral_same
  obtain ⟨h1, h2⟩ := quasipotential_gap_bounds_of_continuity_selfBased hwb hband hcontL hcontS
    hpos hcmin hlower hupper hgrad hL
  rw [hzero, sub_zero] at h1 h2
  exact ⟨h1, h2⟩

end Capstone

/-! ### A worked instance: the hypothesis set is satisfiable, the bracket is strict -/

section Witness

/-- The closed form of the quasipotential at the witness instance below: with `Ltilde'(u) = u` and
the genuinely non-constant diffusion `sigma_eff^2(u) = 1 + u^2`, based at `w_0 = 0`,

  `U_eff(w) = (1/2) log(1 + w^2)`.

The computation is the fundamental theorem of calculus in the other direction: the closed form is
exhibited as an antiderivative and `quasipotential_eq_gap_of_hasDerivAt` identifies it with the
integral. -/
theorem quasipotential_example (w : ℝ) :
    quasipotential (fun u => u) (fun u => 1 + u ^ 2) 0 w = Real.log (1 + w ^ 2) / 2 := by
  have hderiv : ∀ v : ℝ,
      HasDerivAt (fun x : ℝ => Real.log (1 + x ^ 2) / 2) (v / (1 + v ^ 2)) v := by
    intro v
    have hne : (1 : ℝ) + v ^ 2 ≠ 0 := by positivity
    have h1 : HasDerivAt (fun x : ℝ => 1 + x ^ 2) (2 * v) v := by
      simpa using (hasDerivAt_pow 2 v).const_add (1 : ℝ)
    have h2 : HasDerivAt (fun x : ℝ => Real.log (1 + x ^ 2)) (2 * v / (1 + v ^ 2)) v := h1.log hne
    have h3 := h2.div_const 2
    have : 2 * v / (1 + v ^ 2) / 2 = v / (1 + v ^ 2) := by
      field_simp
    rwa [this] at h3
  have hcont : Continuous fun u : ℝ => u / (1 + u ^ 2) :=
    continuous_id.div (by fun_prop) fun u => by positivity
  have hint : IntervalIntegrable (fun u : ℝ => u / (1 + u ^ 2)) volume 0 w :=
    hcont.intervalIntegrable 0 w
  have := quasipotential_eq_gap_of_hasDerivAt (dL := fun u : ℝ => u)
    (sigma2 := fun u : ℝ => 1 + u ^ 2) (Ueff := fun x : ℝ => Real.log (1 + x ^ 2) / 2)
    (wb := 0) (ws := w) (fun u _ => hderiv u) hint
  rw [this]
  norm_num

/-- **The hypothesis set of the capstone is jointly satisfiable, with a non-constant diffusion,
and the bracket it yields is strict on both sides.**

The instance is `Ltilde'(u) = u`, `Ltilde(u) = u^2/2`, `sigma_eff^2(u) = 1 + u^2`, band
`[w_b, w_s] = [0, 1]` inside the window `(-1, 2)`, bracket `cmin = 1`, `cmax = 2`. The diffusion is
genuinely non-constant across the band — `sigma_eff^2(0) = 1 ≠ 2 = sigma_eff^2(1)` — so this is not
the exact constant-band case in disguise. The barrier action is `alpha = 1/2`, so the capstone
predicts `1/4 ≤ U_eff(1) ≤ 1/2`, and the closed form gives `U_eff(1) = (log 2)/2`, which sits
strictly inside that bracket by `BarrierRescaling.quasipotential_gap_bounds_witness_strict`.

The certificate lives in the statement rather than in the proof: the conjunction below carries the
hypotheses, the derived bracket and its strictness together, so a later edit cannot hollow it out
without changing what is claimed. -/
theorem quasipotential_ftc_witness :
    ((1 : ℝ) + (0 : ℝ) ^ 2 ≠ 1 + (1 : ℝ) ^ 2) ∧
      Icc (0 : ℝ) 1 ⊆ Ioo (-1 : ℝ) 2 ∧
      ((1 : ℝ) ^ 2 / 2 - (0 : ℝ) ^ 2 / 2) / 2
          ≤ quasipotential (fun u => u) (fun u => 1 + u ^ 2) 0 1 ∧
        quasipotential (fun u => u) (fun u => 1 + u ^ 2) 0 1
          ≤ ((1 : ℝ) ^ 2 / 2 - (0 : ℝ) ^ 2 / 2) / 1 ∧
      ((1 : ℝ) ^ 2 / 2 - (0 : ℝ) ^ 2 / 2) / 2
          < quasipotential (fun u => u) (fun u => 1 + u ^ 2) 0 1 ∧
        quasipotential (fun u => u) (fun u => 1 + u ^ 2) 0 1
          < ((1 : ℝ) ^ 2 / 2 - (0 : ℝ) ^ 2 / 2) / 1 := by
  have hband : Icc (0 : ℝ) 1 ⊆ Ioo (-1 : ℝ) 2 := by
    intro u hu
    exact ⟨by linarith [hu.1], by linarith [hu.2]⟩
  have hcontL : ContinuousOn (fun u : ℝ => u) (Ioo (-1 : ℝ) 2) := continuousOn_id
  have hcontS : ContinuousOn (fun u : ℝ => 1 + u ^ 2) (Ioo (-1 : ℝ) 2) := by fun_prop
  have hpos : ∀ u ∈ Ioo (-1 : ℝ) 2, 0 < 1 + u ^ 2 := fun u _ => by positivity
  have hlower : ∀ u ∈ Icc (0 : ℝ) 1, (1 : ℝ) ≤ 1 + u ^ 2 := fun u _ => by nlinarith [sq_nonneg u]
  have hupper : ∀ u ∈ Icc (0 : ℝ) 1, (1 : ℝ) + u ^ 2 ≤ 2 := by
    intro u hu; nlinarith [hu.1, hu.2]
  have hgrad : ∀ u ∈ Icc (0 : ℝ) 1, (0 : ℝ) ≤ u := fun u hu => hu.1
  have hLtilde : ∀ u ∈ Icc (0 : ℝ) 1, HasDerivAt (fun x : ℝ => x ^ 2 / 2) u u := by
    intro u _
    simpa using ((hasDerivAt_pow 2 u).div_const 2)
  obtain ⟨h1, h2⟩ := quasipotential_gap_bounds_of_continuity_selfBased
    (Ltilde := fun x : ℝ => x ^ 2 / 2) (dL := fun u : ℝ => u)
    (sigma2 := fun u : ℝ => 1 + u ^ 2) (cmin := 1) (cmax := 2)
    (by norm_num) hband hcontL hcontS hpos one_pos hlower hupper hgrad hLtilde
  obtain ⟨hlog1, hlog2⟩ := quasipotential_gap_bounds_witness_strict
  have hval : quasipotential (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2) 0 1 = Real.log 2 / 2 := by
    rw [quasipotential_example]; norm_num
  refine ⟨by norm_num, hband, ?_, ?_, ?_, ?_⟩
  · simpa using h1
  · simpa using h2
  · rw [hval]; norm_num; linarith
  · rw [hval]; norm_num; linarith

end Witness

end IcnnLift
