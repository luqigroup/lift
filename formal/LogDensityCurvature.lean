import Mathlib
import StationaryDensity

/-!
# (L1') The log-density curvature at a minimum under multiplicative noise

This file machine-checks consequence (L1') of the corrected mechanism, stated in
`docs/design/lemma1_rederivation.md` §3 (lines 105-120), and closes the one gap that
statement leaves open.

The corrected route replaces the refuted Hessian decomposition of the shipped Lemma 1 by a
state-dependent-diffusion argument. The bias channel is modelled by the one-dimensional Itô
diffusion `dw = -Ltilde'(w) dt + sqrt(eta) sigma_eff(w) dB`, whose stationary density in one
dimension is the multiplicative-noise form

    pi(w) = (C0 / sigma_eff^2(w)) * exp (-(2/eta) * U_eff(w)),
    U_eff'(w) = Ltilde'(w) / sigma_eff^2(w).

The note writes this with a proportionality sign; the constant `C0` is carried explicitly and is
universally quantified in every statement about the density below, and
`deriv_deriv_log_stationaryDensity_normalization` records that the curvature does not depend on
it, so no statement here is sensitive to the normalisation the note leaves open.

The note then asserts that at a local minimum `w*` of `Ltilde` inside the shoulder,

    -(log pi)''(w*) = (2/eta) * Ltilde''(w*) / sigma_eff^2(w*) + C(w*),

"with `C` the sigma-variation correction", and never computes `C`. This file computes it.
Differentiating `log pi = log C0 - log (sigma_eff^2) - (2/eta) U_eff` twice gives
`(log pi)'' = -(log sigma_eff^2)'' - (2/eta) U_eff''` with
`U_eff'' = Ltilde''/sigma_eff^2 - Ltilde' (sigma_eff^2)'/(sigma_eff^2)^2`, so the general
identity carries an explicit `Ltilde'` term and the critical-point identity is its collapse.
The correction is therefore exactly `C(w*) = (log sigma_eff^2)''(w*)`, and
`sigmaVariation_correction_eq` states that as a uniqueness result: any `C` for which the note's
display holds is forced to equal `(log sigma_eff^2)''(w*)`. The forms to read as (L1') itself are
`logDensityCurvature_at_localMin_deriv` and `sigmaVariation_correction_eq_deriv`, which write the
leading term with the note's own `Ltilde''(w*)`, that is, as `deriv (deriv Ltilde) w`.

Nothing here is asymptotic. The general statement `neg_deriv_deriv_log_stationaryDensity`
retains the full `Ltilde'` term rather than dropping it as higher order, and every collapse of
that term is a consequence of a hypothesis that is itself proved from `IsLocalMin` (via
`IsLocalMin.hasDerivAt_eq_zero`) or from local constancy of the diffusion, never assumed.

**What this file does not prove, and why.** The density is not restated here: it is
`IcnnLift.stationaryDensity` of the sibling module `StationaryDensity.lean`, which proves that
this closed form solves the one-dimensional stationary Fokker-Planck equation with zero
probability flux, for drift `-Ltilde'` and Fokker-Planck coefficient `D = (eta/2) sigma_eff^2`.
What remains unformalized, there and here, is the step that identifies that Fokker-Planck
operator as the generator of the Itô diffusion
`dw = -Ltilde'(w) dt + sqrt(eta) sigma_eff(w) dB`. Mathlib v4.31.0 contains no Itô integral, no
stochastic differential equations and no notion of an invariant measure of a diffusion, so that
identification cannot be stated in this library, let alone proved; discharging it would require
formalizing one-dimensional Itô diffusions and their infinitesimal generators. The effective
potential `U_eff` is carried as an arbitrary antiderivative of `Ltilde'/sigma_eff^2` near the
point of interest; `hasDerivAt_effectivePotential_integral` shows that the note's own integral
formula — the `effectivePotential` of `StationaryDensity.lean` — supplies such an antiderivative
whenever the integrand is continuous on the line, so the hypothesis is not vacuous;
`logDensityCurvature_localMin_witness` exhibits a complete instance in which every hypothesis of
the flagship form `logDensityCurvature_at_localMin_deriv` — and hence of
`logDensityCurvature_at_localMin` — holds at once and the effective diffusion is not constant.
(That module's own `hasDerivAt_effectivePotential` proves the same differentiability fact under
the slightly stronger hypothesis that `Ltilde'` and `sigma_eff^2` are separately continuous and
`sigma_eff^2` is everywhere positive.)

The comparison result is deliberately narrow, and it is worth being exact about what it is a
statement about. `leadingCurvature_strictAntiOn` and its companions say only that, for a fixed
`Ltilde''(w*) > 0`, the *number* `(2/eta) Ltilde''(w*)/sigma_eff^2(w*)` is strictly decreasing in
`sigma_eff^2(w*)`, and that it diverges as `sigma_eff^2(w*)` tends to zero from above. They are
statements about the curvature of `log pi` at the single point `w*`; that a smaller curvature at
the minimum means a broader stationary measure is the physical reading of those statements and is
not itself proved by them. The reading is the note's: not that the lift adds convexity — the
corrected route exists precisely because that claim is false — but that the lift keeps the
temperature finite on the shoulder. The direct baseline's `sigma_eff^2 = s^2 sigma_obj^2` is tiny
there, so the same `Ltilde''` is divided by a tiny number; the lift divides the *same* `Ltilde''`
by a strictly larger `sigma_eff^2`.

One statement of this file is about the measure rather than about its curvature, so that the
note's word "needle" is not left as rhetoric. `stationaryDensity_ratio_tendsto_zero` proves that
on a band of constant effective diffusion `c` — the (A2) regime in which `(*)` reduces to the
Boltzmann form `pi ∝ exp(-(2/(eta c)) Ltilde)` — the density at any point `u` of strictly larger
pullback loss than `w`, taken relative to the density at `w`, tends to zero as `c` tends to zero
from above. That is the collapse onto the trapping point, at the level of the density; the
divergence of the curvature in `leadingCurvature_tendsto_atTop` is its local shadow.

Second derivatives are written `deriv (deriv f)` throughout, never `iteratedDeriv 2`;
`iteratedDeriv_two_eq_deriv_deriv` records that the two agree, so no generality is lost.

## Results
* `log_stationaryDensity` — the logarithm of the note's density `(*)`, taken from
  `StationaryDensity.lean`, in the additive form the calculus uses.
* `stationaryDensity_div_const` — dividing the density by a constant returns a density of the
  same form, so the normalised `pi/Z` is again one of the densities every statement here is
  quantified over.
* `deriv_deriv_log_stationaryDensity_normalization` — the curvature is unchanged by the choice of
  normalising constant, so the note's `pi ∝ …` is not an ambiguity in any statement below.
* `hasDerivAt_effectivePotential_integral` — the note's `U_eff`, stated on the
  `effectivePotential` of `StationaryDensity.lean`, is an antiderivative of
  `Ltilde'/sigma_eff^2` whenever that quotient is continuous on the line.
* `hasDerivAt_logForm` — the first derivative of `log pi`.
* `deriv_deriv_log_diffusion` — the explicit second derivative of `log sigma_eff^2`.
* `neg_deriv_deriv_log_stationaryDensity` — the general identity, with the `Ltilde'` term
  present, so the role of the critical-point hypothesis is visible.
* `neg_deriv_deriv_log_stationaryDensity_of_critical` — (L1') at a critical point of the drift.
* `logDensityCurvature_at_localMin` — (L1') at a local minimum, with the second derivative read
  off the drift function `dLtilde` that defines `U_eff`.
* `logDensityCurvature_at_localMin_deriv` — (L1') at a local minimum in the note's own symbols,
  with the leading term written `deriv (deriv Ltilde) w`, that is, literally `Ltilde''(w*)`.
* `sigmaVariation_correction_eq`, `sigmaVariation_correction_eq_deriv` — the note's uncomputed
  `C(w*)` is forced to equal `(log sigma_eff^2)''(w*)`, in the two forms above.
* `deriv_deriv_log_diffusion_eq_zero_of_eventuallyConst` — the correction vanishes on a band on
  which `sigma_eff^2` is exactly constant.
* `eventually_hasDerivAt_zero_of_eventuallyConst` — the derivative bookkeeping such a band needs.
* `logDensityCurvature_of_const_diffusion` — the curvature on such a band, where criticality is
  not needed because the `Ltilde'` term carries the factor `(sigma_eff^2)'`.
* `abs_logDensityCurvature_sub_leading_le` — the (A2) band without the exact-constancy
  idealisation: a bound `eps` on `(log sigma_eff^2)''(w*)` is carried into the conclusion instead
  of the correction being assumed to vanish.
* `leadingCurvature` — the leading curvature term as a function of the effective diffusion.
* `leadingCurvature_strictAntiOn`, `leadingCurvature_lt_of_diffusion_lt` — the leading term is
  strictly decreasing in the effective diffusion.
* `leadingCurvature_gap_pos` — the gap between the two leading terms is strictly positive, which
  is what makes the tolerance of the un-idealised comparison below a nonempty band.
* `leadingCurvature_ratio` — the curvature ratio is exactly the inverse diffusion ratio.
* `leadingCurvature_tendsto_atTop` — the needle: the curvature diverges as the effective
  diffusion tends to zero from above.
* `logDensityCurvature_lt_of_diffusion_lt` — the comparison at the level of the densities
  themselves, on a band where both diffusions are exactly constant.
* `logDensityCurvature_lt_of_diffusion_lt_deriv` — the same comparison with the shared drift
  tied to `Ltilde` on a neighbourhood, so the positive quantity is literally the note's
  `Ltilde''(w*)`.
* `logDensityCurvature_lt_of_diffusion_lt_of_correction_gap` — the same comparison for arbitrary
  effective diffusions, conditional on the corrections not swamping the gap in the leading terms.
* `logDensityCurvature_lt_of_diffusion_lt_of_correction_bound` — the same comparison with the
  tolerance stated one bound per effective diffusion, `|(log sigma_eff^2)''(w*)| <= eps` with
  `2 eps` below the gap in the leading terms, which is the form in which a reader can check how
  much variation the comparison survives.
* `stationaryDensity_ratio_tendsto_zero` — the needle at the level of the density: on a band of
  constant effective diffusion, the density at a point of strictly larger pullback loss, relative
  to the density at `w`, tends to zero as the effective diffusion tends to zero from above.
* `deriv_deriv_log_one_add_sq` — the correction is not identically zero.
* `logDensityCurvature_quadratic` — the constant-diffusion bundle is satisfiable, and on the
  Gaussian instance the identity returns the classical answer.
* `logDensityCurvature_localMin_witness` — the *full* bundle of the flagship
  `logDensityCurvature_at_localMin_deriv` (hence of `logDensityCurvature_at_localMin`) is
  satisfiable with a non-constant `sigma_eff^2`, and on that instance the correction is `2`.
* `logDensityCurvature_lt_quadratic` — the density-level comparison is satisfiable.
* `logDensityCurvature_lt_correction_gap_witness` — the un-idealised comparison is satisfiable
  with both sigma-variation corrections nonzero, so its generality over the constant band is not
  empty.
* `logDensityCurvature_lt_correction_bound_witness` — the bounded-tolerance comparison is
  satisfiable with a strictly positive tolerance and two genuinely varying effective diffusions,
  so it is not a disguised statement about the constant band.
* `iteratedDeriv_two_eq_deriv_deriv` — the second-derivative convention.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. The explicit hypotheses carried are: (i) the identification of
the zero-flux Fokker-Planck density of `StationaryDensity.lean` with the stationary law of the
Itô diffusion, which mathlib has no stochastic calculus in which to state; (ii) the existence
near the point of an antiderivative `U` of `Ltilde'/sigma_eff^2`, which
`hasDerivAt_effectivePotential_integral` discharges under continuity; (iii) the
differentiability inputs — `sigma_eff^2` differentiable on a neighbourhood of the point with its
derivative differentiable at the point (`hs1` and `hs2`; the latter is what makes
`(log sigma_eff^2)''`, the computed correction, exist at all), the drift `Ltilde'` differentiable
at the point (`hddL`), and `dLtilde` tied to the derivative of `Ltilde` — at the point alone
(`hL1`) in `logDensityCurvature_at_localMin` and `sigmaVariation_correction_eq`, on a
neighbourhood (`hL`) in the `_deriv` forms and in `abs_logDensityCurvature_sub_leading_le` —
with every neighbourhood hypothesis stated as `∀ᶠ` in `𝓝 w`, so that
"inside the shoulder" is a genuine locality condition and not a global smoothness assumption;
and (iv) positivity of `sigma_eff^2` on that neighbourhood, which is the ellipticity clause of
(A2).

Nothing is assumed to leading order: the general identity
`neg_deriv_deriv_log_stationaryDensity` retains the `Ltilde'` term and the sigma-variation
correction exactly, and no Taylor remainder is discarded anywhere. One *idealisation* is made,
and it is confined to the band results. The note's assumption (A2) is that `sigma_eff^2` is
**approximately** constant across the shoulder band;
`deriv_deriv_log_diffusion_eq_zero_of_eventuallyConst`, `logDensityCurvature_of_const_diffusion`
and `logDensityCurvature_lt_of_diffusion_lt` formalise this as **exact** local constancy, which
does set the sigma-variation correction to zero by hypothesis. That idealisation is dispensable
and is not relied on: `abs_logDensityCurvature_sub_leading_le` proves the band conclusion with a
bound `eps` on `(log sigma_eff^2)''(w*)` carried into the conclusion, and
`logDensityCurvature_lt_of_diffusion_lt_of_correction_gap` proves the comparison for arbitrary
effective diffusions under an explicit, visible hypothesis on how large the corrections may be
relative to the gap in the leading terms. That hypothesis is sharp — given the identity it is
equivalent to the conclusion — and it constrains only the difference of the two corrections, so
`logDensityCurvature_lt_of_diffusion_lt_of_correction_bound` restates it as one bound per
effective diffusion, `|(log sigma_eff^2)''(w*)| <= eps` with `2 eps` strictly below the gap in the
leading terms, and `leadingCurvature_gap_pos` records that this gap is a strictly positive number,
so the tolerance is a nonempty band and not a vacuous side condition. That the distinction has
content is itself proved: `deriv_deriv_log_one_add_sq` and `logDensityCurvature_localMin_witness`
exhibit an admissible effective diffusion for which the correction equals `2`;
`logDensityCurvature_lt_correction_gap_witness` exhibits the un-idealised comparison itself
firing between two genuinely varying effective diffusions whose corrections, `2` and `1/2`, are
both nonzero; and `logDensityCurvature_lt_correction_bound_witness` exhibits the same comparison
firing through the bounded-tolerance form, with `eps = 2` and neither correction zero.

A second point of honesty concerns what the comparison statements are statements about. Every
result in the comparison section other than `stationaryDensity_ratio_tendsto_zero` is an assertion
about the real number `-(log pi)''(w*)`, that is, about the curvature of the log-density at a
single point. The note's reading of that number — a broader stationary measure at a usable
temperature, against a needle at the trapping point — is an interpretation, and the docstrings
below say so rather than asserting it as proved. The one measure-level statement proved here is
`stationaryDensity_ratio_tendsto_zero`, and it is proved only in the constant-diffusion regime,
where `(*)` reduces to a Boltzmann form; a general statement about the width of the multiplicative
noise measure, or about its concentration as `sigma_eff^2` degenerates, is not attempted.

One further point of exactness. `neg_deriv_deriv_log_stationaryDensity`, its critical-point
collapse, `logDensityCurvature_of_const_diffusion` and `logDensityCurvature_lt_of_diffusion_lt`
mention no `Ltilde` at all — they are statements about an arbitrary drift function `dLtilde` —
and the hypotheses of `logDensityCurvature_at_localMin` tie that drift to `Ltilde` only at the
point `w`. That is all the proofs need, and it makes those statements applicable to any drift;
but it means that in them the number `ddL` is the derivative at `w` of the drift, and is not by
itself the second derivative of `Ltilde`. `logDensityCurvature_at_localMin_deriv`,
`sigmaVariation_correction_eq_deriv` and `logDensityCurvature_lt_of_diffusion_lt_deriv` remove
that slack by requiring `dLtilde` to be the derivative of `Ltilde` throughout a neighbourhood,
and write the second derivative — in the conclusion of the first two, in the positivity
hypothesis of the third — with the note's own `Ltilde''(w*)`, that is, as
`deriv (deriv Ltilde) w`. Those are the statements to read as (L1').
-/

open Filter Topology

namespace IcnnLift

/-! ### The logarithm of the stationary density

The density itself is `IcnnLift.stationaryDensity` of `StationaryDensity.lean`, imported rather
than restated: `stationaryDensity C0 eta sigma2 U w = C0 / sigma2 w * exp (-(2/eta) U w)`, the
display `(*)` of `lemma1_rederivation.md` §3. -/

/-- The logarithm of the stationary density in the additive form the curvature computation uses:
`log pi = log C0 - log (sigma_eff^2) - (2/eta) U_eff`. -/
theorem log_stationaryDensity {C0 eta : ℝ} {sigma2 U : ℝ → ℝ} {w : ℝ}
    (hC0 : C0 ≠ 0) (hsw : sigma2 w ≠ 0) :
    Real.log (stationaryDensity C0 eta sigma2 U w)
      = Real.log C0 - Real.log (sigma2 w) - 2 / eta * U w := by
  have hexp : Real.exp (-(2 / eta) * U w) ≠ 0 := Real.exp_ne_zero _
  simp only [stationaryDensity]
  rw [Real.log_mul (div_ne_zero hC0 hsw) hexp, Real.log_div hC0 hsw, Real.log_exp]
  ring

/-- Dividing the density `(*)` by a constant returns a density of the same form, with the
proportionality constant divided by that constant. This is what makes the normalisation question
a substitution rather than an appeal to the reader: the normalised probability density `pi/Z` is
`stationaryDensity (C0/Z) eta sigma_eff^2 U_eff`, so every statement of this file, each of which
is universally quantified over that constant, applies to `pi/Z` unchanged. -/
theorem stationaryDensity_div_const (C0 eta Z : ℝ) (sigma2 U : ℝ → ℝ) (w : ℝ) :
    stationaryDensity C0 eta sigma2 U w / Z = stationaryDensity (C0 / Z) eta sigma2 U w := by
  simp only [stationaryDensity]
  ring

/-- **The curvature does not depend on the normalisation.** The note writes the stationary
density with a proportionality sign, `pi ∝ (1/sigma_eff^2) exp (-(2/eta) U_eff)`, which fixes it
only up to a positive constant; the normalised probability density is `pi/Z`. Rescaling the
constant shifts `log pi` by a constant and therefore leaves every second derivative below
unchanged, so the curvature statements of this file, all of which are proved for an arbitrary
`C0`, hold verbatim for the normalised density whenever the density is normalisable — by
`stationaryDensity_div_const` that density is again of the form the statements quantify over, and
`exists_normalized_stationaryDensity` of `StationaryDensity.lean` supplies the normalising
constant under an integrability hypothesis. -/
theorem deriv_deriv_log_stationaryDensity_normalization
    {C1 C2 eta : ℝ} {sigma2 U : ℝ → ℝ} {w : ℝ} (hC1 : C1 ≠ 0) (hC2 : C2 ≠ 0)
    (hne : ∀ᶠ u in 𝓝 w, sigma2 u ≠ 0) :
    deriv (deriv fun u => Real.log (stationaryDensity C1 eta sigma2 U u)) w
      = deriv (deriv fun u => Real.log (stationaryDensity C2 eta sigma2 U u)) w := by
  have hEq : (fun u => Real.log (stationaryDensity C1 eta sigma2 U u))
      =ᶠ[𝓝 w] fun u => Real.log (stationaryDensity C2 eta sigma2 U u)
        + (Real.log C1 - Real.log C2) := by
    filter_upwards [hne] with u hu
    rw [log_stationaryDensity hC1 hu, log_stationaryDensity hC2 hu]
    ring
  have h1 : (deriv fun u => Real.log (stationaryDensity C1 eta sigma2 U u))
      =ᶠ[𝓝 w] deriv fun u => Real.log (stationaryDensity C2 eta sigma2 U u) := by
    filter_upwards [hEq.eventually_nhds] with x hx
    rw [Filter.EventuallyEq.deriv_eq hx, deriv_add_const]
  exact h1.deriv_eq

/-- The note's effective potential `U_eff(w) = ∫^w Ltilde'(u)/sigma_eff^2(u) du` — stated on the
`effectivePotential` of `StationaryDensity.lean`, which is exactly that integral with the lower
limit made explicit — is an antiderivative of `Ltilde'/sigma_eff^2` whenever that quotient is
continuous, so the antiderivative hypothesis carried by the results below is not vacuous. Any
two antiderivatives differ by a constant, which the normalizing constant `C0` absorbs
(`deriv_deriv_log_stationaryDensity_normalization`). That module's own
`hasDerivAt_effectivePotential` proves the same fact under separate continuity of `Ltilde'` and
`sigma_eff^2` plus everywhere-positivity of `sigma_eff^2`; the form here needs only continuity
of the quotient. Note that the continuity hypothesis here is global, on the whole line, whereas
everything the results below need is local: this statement is a sufficient condition for the
antiderivative hypothesis, not a characterisation of it, and a caller who has an antiderivative
only near `w` should supply it directly. -/
theorem hasDerivAt_effectivePotential_integral {sigma2 dLtilde : ℝ → ℝ}
    (hcont : Continuous fun u => dLtilde u / sigma2 u) (a w : ℝ) :
    HasDerivAt (effectivePotential a dLtilde sigma2) (dLtilde w / sigma2 w) w :=
  (hcont.integral_hasStrictDerivAt a w).hasDerivAt

/-! ### First and second derivatives -/

/-- The first derivative of `log pi` in its additive form:
`(log pi)' = -(sigma_eff^2)'/sigma_eff^2 - (2/eta) Ltilde'/sigma_eff^2`. -/
theorem hasDerivAt_logForm {C0 eta : ℝ} {sigma2 dsigma2 dLtilde U : ℝ → ℝ} {u : ℝ}
    (hsu : sigma2 u ≠ 0) (hs : HasDerivAt sigma2 (dsigma2 u) u)
    (hU : HasDerivAt U (dLtilde u / sigma2 u) u) :
    HasDerivAt (fun v => Real.log C0 - Real.log (sigma2 v) - 2 / eta * U v)
      (-(dsigma2 u / sigma2 u) - 2 / eta * (dLtilde u / sigma2 u)) u := by
  have h1 : HasDerivAt (fun v => Real.log (sigma2 v)) (dsigma2 u / sigma2 u) u := hs.log hsu
  have h2 : HasDerivAt (fun v => Real.log C0 - Real.log (sigma2 v)) (-(dsigma2 u / sigma2 u)) u :=
    h1.const_sub _
  have h3 : HasDerivAt (fun v => 2 / eta * U v) (2 / eta * (dLtilde u / sigma2 u)) u :=
    hU.const_mul _
  exact h2.sub h3

/-- The sigma-variation correction in closed form: the second derivative of
`log sigma_eff^2` at `w` is `((sigma_eff^2)'' sigma_eff^2 - ((sigma_eff^2)')^2)/(sigma_eff^2)^2`.
This is the object the note calls `C(w*)` and does not compute. -/
theorem deriv_deriv_log_diffusion {sigma2 dsigma2 : ℝ → ℝ} {w dds : ℝ}
    (hne : ∀ᶠ u in 𝓝 w, sigma2 u ≠ 0)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w) :
    deriv (deriv fun u => Real.log (sigma2 u)) w
      = (dds * sigma2 w - dsigma2 w * dsigma2 w) / sigma2 w ^ 2 := by
  have hd : (deriv fun u => Real.log (sigma2 u)) =ᶠ[𝓝 w] fun u => dsigma2 u / sigma2 u := by
    filter_upwards [hne, hs1] with u h1 h2 using (h2.log h1).deriv
  rw [hd.deriv_eq]
  exact (hs2.div hs1.self_of_nhds hne.self_of_nhds).deriv

/-! ### (L1'), general and at a critical point -/

/-- **The general log-density curvature identity.** With no assumption on `Ltilde'`,

`-(log pi)''(w) = (2/eta)(Ltilde''(w)/sigma_eff^2(w) - Ltilde'(w)(sigma_eff^2)'(w)/sigma_eff^2(w)^2)
  + (log sigma_eff^2)''(w)`.

The middle term is the one that the critical-point hypothesis of (L1') removes; it is displayed
here rather than dropped, so that the role of that hypothesis is visible. Note that the identity
is an identity of real numbers for every `eta`, including `eta = 0` under Lean's division
convention; the physical reading of course requires `eta > 0`. -/
theorem neg_deriv_deriv_log_stationaryDensity
    {C0 eta : ℝ} {sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hddL : HasDerivAt dLtilde ddL w) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
      = 2 / eta * (ddL / sigma2 w - dLtilde w * dsigma2 w / sigma2 w ^ 2)
        + deriv (deriv fun u => Real.log (sigma2 u)) w := by
  have hne : ∀ᶠ u in 𝓝 w, sigma2 u ≠ 0 := hspos.mono fun _ h => h.ne'
  have hwne : sigma2 w ≠ 0 := hne.self_of_nhds
  -- Replace `log pi` by its additive form on a neighbourhood of `w`.
  have hEq : (fun u => Real.log (stationaryDensity C0 eta sigma2 U u))
      =ᶠ[𝓝 w] fun u => Real.log C0 - Real.log (sigma2 u) - 2 / eta * U u := by
    filter_upwards [hne] with u hu using log_stationaryDensity hC0.ne' hu
  have hEq1 : (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u))
      =ᶠ[𝓝 w] deriv fun u => Real.log C0 - Real.log (sigma2 u) - 2 / eta * U u := by
    filter_upwards [hEq.eventually_nhds] with x hx using Filter.EventuallyEq.deriv_eq hx
  rw [hEq1.deriv_eq]
  -- The first derivative, as a function on a neighbourhood of `w`.
  have hD1 : (deriv fun u => Real.log C0 - Real.log (sigma2 u) - 2 / eta * U u)
      =ᶠ[𝓝 w] fun u => -(dsigma2 u / sigma2 u) - 2 / eta * (dLtilde u / sigma2 u) := by
    filter_upwards [hne, hs1, hU] with u h1 h2 h3 using (hasDerivAt_logForm h1 h2 h3).deriv
  rw [hD1.deriv_eq]
  -- The second derivative, at `w`.
  have hq1 : HasDerivAt (fun v => dsigma2 v / sigma2 v)
      ((dds * sigma2 w - dsigma2 w * dsigma2 w) / sigma2 w ^ 2) w :=
    hs2.div hs1.self_of_nhds hwne
  have hq2 : HasDerivAt (fun v => dLtilde v / sigma2 v)
      ((ddL * sigma2 w - dLtilde w * dsigma2 w) / sigma2 w ^ 2) w :=
    hddL.div hs1.self_of_nhds hwne
  have hD2 : HasDerivAt (fun u => -(dsigma2 u / sigma2 u) - 2 / eta * (dLtilde u / sigma2 u))
      (-((dds * sigma2 w - dsigma2 w * dsigma2 w) / sigma2 w ^ 2)
        - 2 / eta * ((ddL * sigma2 w - dLtilde w * dsigma2 w) / sigma2 w ^ 2)) w :=
    hq1.neg.sub (hq2.const_mul _)
  rw [hD2.deriv, deriv_deriv_log_diffusion hne hs1 hs2]
  have key : (ddL * sigma2 w - dLtilde w * dsigma2 w) / sigma2 w ^ 2
      = ddL / sigma2 w - dLtilde w * dsigma2 w / sigma2 w ^ 2 := by
    field_simp
  rw [key]
  ring

/-- **(L1') at a critical point.** Where `Ltilde'(w) = 0` the `Ltilde'` term of the general
identity collapses and

`-(log pi)''(w) = (2/eta) Ltilde''(w)/sigma_eff^2(w) + (log sigma_eff^2)''(w)`,

which is the note's display with the sigma-variation correction `C(w)` computed: it is exactly
`(log sigma_eff^2)''(w)`. -/
theorem neg_deriv_deriv_log_stationaryDensity_of_critical
    {C0 eta : ℝ} {sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hcrit : dLtilde w = 0) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
      = 2 / eta * (ddL / sigma2 w) + deriv (deriv fun u => Real.log (sigma2 u)) w := by
  rw [neg_deriv_deriv_log_stationaryDensity hC0 hspos hs1 hs2 hU hddL, hcrit]
  ring

/-- **(L1') at a local minimum** `w*` of `Ltilde` inside the shoulder. The criticality of `w*`
is derived from `IsLocalMin` by `IsLocalMin.hasDerivAt_eq_zero`, not assumed. Note what `ddL` is
here: `hL1` ties the drift function `dLtilde` to `Ltilde` only at the point `w`, which is all
that is needed to force `dLtilde w = 0`, so `ddL` is the derivative at `w` of the drift function
that defines `U_eff` and is not, on these hypotheses alone, the second derivative of `Ltilde`.
`logDensityCurvature_at_localMin_deriv` below removes that slack and states the conclusion with
the note's own `Ltilde''(w*)`. -/
theorem logDensityCurvature_at_localMin
    {C0 eta : ℝ} {Ltilde sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hL1 : HasDerivAt Ltilde (dLtilde w) w)
    (hddL : HasDerivAt dLtilde ddL w)
    (hmin : IsLocalMin Ltilde w) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
      = 2 / eta * (ddL / sigma2 w) + deriv (deriv fun u => Real.log (sigma2 u)) w :=
  neg_deriv_deriv_log_stationaryDensity_of_critical hC0 hspos hs1 hs2 hU hddL
    (hmin.hasDerivAt_eq_zero hL1)

/-- **(L1') in the note's own symbols.** The statement above is quantified over a drift function
`dLtilde` that is tied to `Ltilde` only at the point `w`, so its `ddL` is the derivative at `w` of
that drift function and not, on the strength of those hypotheses alone, the second derivative of
`Ltilde`. This version closes that gap: `dLtilde` is required to be the derivative of `Ltilde`
throughout a neighbourhood of `w`, and the conclusion is then stated with the note's own symbol,
`Ltilde''(w*)` written as `deriv (deriv Ltilde) w`:

`-(log pi)''(w*) = (2/eta) Ltilde''(w*)/sigma_eff^2(w*) + (log sigma_eff^2)''(w*)`.

Every symbol of the note's display is now the object the note means it to be. -/
theorem logDensityCurvature_at_localMin_deriv
    {C0 eta : ℝ} {Ltilde sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hL : ∀ᶠ u in 𝓝 w, HasDerivAt Ltilde (dLtilde u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hmin : IsLocalMin Ltilde w) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
      = 2 / eta * (deriv (deriv Ltilde) w / sigma2 w)
        + deriv (deriv fun u => Real.log (sigma2 u)) w := by
  have hderiv : deriv Ltilde =ᶠ[𝓝 w] dLtilde := by
    filter_upwards [hL] with u hu using hu.deriv
  have hdd : deriv (deriv Ltilde) w = ddL := by
    rw [hderiv.deriv_eq]
    exact hddL.deriv
  rw [hdd]
  exact logDensityCurvature_at_localMin hC0 hspos hs1 hs2 hU hL.self_of_nhds hddL hmin

/-- **The gap the note leaves open, closed.** The note writes the curvature at a minimum as
`(2/eta) Ltilde''(w*)/sigma_eff^2(w*) + C(w*)` "with `C` the sigma-variation correction" and
never computes `C`. Under the hypotheses of (L1') there is exactly one such `C`, and it is the
second derivative of `log sigma_eff^2` at `w*`. -/
theorem sigmaVariation_correction_eq
    {C0 eta : ℝ} {Ltilde sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds C : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hL1 : HasDerivAt Ltilde (dLtilde w) w)
    (hddL : HasDerivAt dLtilde ddL w)
    (hmin : IsLocalMin Ltilde w)
    (hC : -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
        = 2 / eta * (ddL / sigma2 w) + C) :
    C = deriv (deriv fun u => Real.log (sigma2 u)) w := by
  have h := logDensityCurvature_at_localMin (eta := eta) hC0 hspos hs1 hs2 hU hL1 hddL hmin
  rw [hC] at h
  linarith

/-- **The gap the note leaves open, closed, in the note's own symbols.** The same uniqueness
statement as `sigmaVariation_correction_eq`, but with the leading term written using the note's
`Ltilde''(w*)` rather than the derivative of a drift function tied to `Ltilde` only at `w*`. If
the note's display holds at a local minimum with any real number `C` in the place of the
sigma-variation correction, then `C` is the second derivative of `log sigma_eff^2` at that
minimum. -/
theorem sigmaVariation_correction_eq_deriv
    {C0 eta : ℝ} {Ltilde sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds C : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hL : ∀ᶠ u in 𝓝 w, HasDerivAt Ltilde (dLtilde u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hmin : IsLocalMin Ltilde w)
    (hC : -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
        = 2 / eta * (deriv (deriv Ltilde) w / sigma2 w) + C) :
    C = deriv (deriv fun u => Real.log (sigma2 u)) w := by
  have h := logDensityCurvature_at_localMin_deriv (eta := eta) hC0 hspos hs1 hs2 hU hL hddL hmin
  rw [hC] at h
  linarith

/-! ### The (A2) band: a locally constant effective diffusion -/

/-- On a band on which `sigma_eff^2` is exactly constant the sigma-variation correction
vanishes. This is an idealisation of the regime the note's (A2) band argument works in: the note
assumes only that `sigma_eff^2` is *approximately* constant across the shoulder band, and exact
constancy sets the correction to zero rather than bounding it.
`abs_logDensityCurvature_sub_leading_le` gives the un-idealised form, in which a bound on the
correction is carried into the conclusion. -/
theorem deriv_deriv_log_diffusion_eq_zero_of_eventuallyConst {sigma2 : ℝ → ℝ} {c w : ℝ}
    (hconst : sigma2 =ᶠ[𝓝 w] fun _ => c) :
    deriv (deriv fun u => Real.log (sigma2 u)) w = 0 := by
  have h1 : (fun u => Real.log (sigma2 u)) =ᶠ[𝓝 w] fun _ => Real.log c := by
    filter_upwards [hconst] with u hu using by rw [hu]
  have h2 : (deriv fun u => Real.log (sigma2 u)) =ᶠ[𝓝 w] fun _ => (0 : ℝ) := by
    filter_upwards [h1.eventually_nhds] with x hx using by
      rw [Filter.EventuallyEq.deriv_eq hx, deriv_const]
  rw [h2.deriv_eq, deriv_const]

/-- A function constant on a neighbourhood has vanishing derivative on that neighbourhood. -/
theorem eventually_hasDerivAt_zero_of_eventuallyConst {sigma2 : ℝ → ℝ} {c w : ℝ}
    (hconst : sigma2 =ᶠ[𝓝 w] fun _ => c) : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 0 u := by
  filter_upwards [hconst.eventually_nhds] with x hx
  exact (hasDerivAt_const x c).congr_of_eventuallyEq hx

/-- **The curvature on a band of constant effective diffusion.** Here

`-(log pi)''(w) = (2/eta) Ltilde''(w) / sigma_eff^2`,

with no correction term and — worth noting — with no criticality hypothesis: the `Ltilde'` term
of the general identity carries the factor `(sigma_eff^2)'`, which vanishes on such a band. This
is the form in which (L1') feeds (L2'). The hypothesis `hconst` is exact local constancy, which
is stronger than the note's "approximately constant" (A2) band;
`abs_logDensityCurvature_sub_leading_le` states the conclusion under a bound on the variation
instead, at the cost of a criticality hypothesis and of an `eps` in the conclusion. -/
theorem logDensityCurvature_of_const_diffusion
    {C0 eta c : ℝ} {sigma2 dLtilde U : ℝ → ℝ} {w ddL : ℝ}
    (hC0 : 0 < C0) (hc : 0 < c)
    (hconst : sigma2 =ᶠ[𝓝 w] fun _ => c)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hddL : HasDerivAt dLtilde ddL w) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w
      = 2 / eta * (ddL / c) := by
  have hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u := by
    filter_upwards [hconst] with u hu using by rw [hu]; exact hc
  have hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 ((fun _ : ℝ => (0 : ℝ)) u) u :=
    eventually_hasDerivAt_zero_of_eventuallyConst hconst
  have hs2 : HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 w := hasDerivAt_const w 0
  have hw : sigma2 w = c := hconst.self_of_nhds
  rw [neg_deriv_deriv_log_stationaryDensity hC0 hspos hs1 hs2 hU hddL,
    deriv_deriv_log_diffusion_eq_zero_of_eventuallyConst hconst, hw]
  ring

/-- **The (A2) band with the remainder carried, not assumed away.** The note's assumption (A2)
is that `sigma_eff^2` is *approximately* constant across the shoulder band, not exactly constant;
`logDensityCurvature_of_const_diffusion` above idealises it to exact local constancy, which sets
the sigma-variation correction to zero by hypothesis. This statement does not. It assumes only a
bound `|(log sigma_eff^2)''(w*)| <= eps` on the variation of the effective diffusion — the honest
reading of "approximately constant" for a curvature statement — and carries that bound into the
conclusion:

`| -(log pi)''(w*) - (2/eta) Ltilde''(w*)/sigma_eff^2(w*) | <= eps`.

Exact constancy is the case `eps = 0`, and `deriv_deriv_log_one_add_sq` shows that `eps` cannot
be taken to be `0` for an arbitrary admissible effective diffusion. -/
theorem abs_logDensityCurvature_sub_leading_le
    {C0 eta eps : ℝ} {Ltilde sigma2 dsigma2 dLtilde U : ℝ → ℝ} {w ddL dds : ℝ}
    (hC0 : 0 < C0)
    (hspos : ∀ᶠ u in 𝓝 w, 0 < sigma2 u)
    (hs1 : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2 (dsigma2 u) u)
    (hs2 : HasDerivAt dsigma2 dds w)
    (hU : ∀ᶠ u in 𝓝 w, HasDerivAt U (dLtilde u / sigma2 u) u)
    (hL : ∀ᶠ u in 𝓝 w, HasDerivAt Ltilde (dLtilde u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hmin : IsLocalMin Ltilde w)
    (hvar : |deriv (deriv fun u => Real.log (sigma2 u)) w| ≤ eps) :
    |(-deriv (deriv fun u => Real.log (stationaryDensity C0 eta sigma2 U u)) w)
        - 2 / eta * (deriv (deriv Ltilde) w / sigma2 w)| ≤ eps := by
  rw [logDensityCurvature_at_localMin_deriv hC0 hspos hs1 hs2 hU hL hddL hmin]
  have h : 2 / eta * (deriv (deriv Ltilde) w / sigma2 w)
      + deriv (deriv fun u => Real.log (sigma2 u)) w
      - 2 / eta * (deriv (deriv Ltilde) w / sigma2 w)
      = deriv (deriv fun u => Real.log (sigma2 u)) w := by ring
  rw [h]
  exact hvar

/-! ### The comparison: a larger effective diffusion is a lower curvature -/

/-- The leading curvature term of (L1'), `(2/eta) Ltilde''(w*) / sigma_eff^2(w*)`, read as a
function of the effective diffusion `s = sigma_eff^2(w*)` at fixed `eta` and fixed
`Ltilde''(w*)`. -/
noncomputable def leadingCurvature (eta ddL s : ℝ) : ℝ := 2 / eta * (ddL / s)

/-- For a fixed positive `Ltilde''(w*)` the leading curvature term is strictly decreasing in the
effective diffusion. What is proved is exactly that: a statement about the real number
`(2/eta) Ltilde''(w*)/sigma_eff^2(w*)`, at fixed `eta` and fixed `Ltilde''(w*)`. What it rules
out is the "added convexity" reading, since the same `Ltilde''(w*)` stands on both sides and only
the divisor changes; the note's positive reading — that the larger effective diffusion gives a
broader stationary measure governed by the same `Ltilde` — is an interpretation of this number and
is not proved here. `stationaryDensity_ratio_tendsto_zero` is the statement of this file that is
about the measure itself. -/
theorem leadingCurvature_strictAntiOn {eta ddL : ℝ} (heta : 0 < eta) (hddL : 0 < ddL) :
    StrictAntiOn (leadingCurvature eta ddL) (Set.Ioi 0) := by
  intro s₁ hs₁ s₂ hs₂ h12
  have hs₁' : (0 : ℝ) < s₁ := hs₁
  have hs₂' : (0 : ℝ) < s₂ := hs₂
  have hpre : (0 : ℝ) < 2 / eta := by positivity
  have hkey : ddL / s₂ < ddL / s₁ := by
    rw [div_lt_div_iff₀ hs₂' hs₁']
    exact mul_lt_mul_of_pos_left h12 hddL
  simp only [leadingCurvature]
  exact mul_lt_mul_of_pos_left hkey hpre

/-- Two-point form of the comparison: the direct baseline's tiny effective diffusion `s₁` gives
a strictly *larger* log-density curvature at the minimum than the lift's `s₂ > s₁`. The lift has
not added convexity — the same `Ltilde''` appears on both sides — it has divided that `Ltilde''`
by a larger effective diffusion. That this is the statement that the temperature on the shoulder
stays finite is the note's reading of the inequality, not a further conclusion drawn from it. -/
theorem leadingCurvature_lt_of_diffusion_lt {eta ddL s₁ s₂ : ℝ} (heta : 0 < eta) (hddL : 0 < ddL)
    (hs₁ : 0 < s₁) (h12 : s₁ < s₂) :
    leadingCurvature eta ddL s₂ < leadingCurvature eta ddL s₁ :=
  leadingCurvature_strictAntiOn heta hddL hs₁ (hs₁.trans h12) h12

/-- The gap between the two leading curvature terms is a strictly positive number whenever the
effective diffusions are ordered and `Ltilde''(w*) > 0`. This is what makes the tolerance carried
by `logDensityCurvature_lt_of_diffusion_lt_of_correction_bound` a nonempty band: there is always
an `eps > 0` with `2 eps` below the gap, so that statement is not a vacuous side condition, and
the size of the admissible variation in the two effective diffusions is exactly this gap. -/
theorem leadingCurvature_gap_pos {eta ddL s₁ s₂ : ℝ} (heta : 0 < eta) (hddL : 0 < ddL)
    (hs₁ : 0 < s₁) (h12 : s₁ < s₂) :
    0 < leadingCurvature eta ddL s₁ - leadingCurvature eta ddL s₂ :=
  sub_pos.mpr (leadingCurvature_lt_of_diffusion_lt heta hddL hs₁ h12)

/-- The comparison is exact, not merely ordinal: the ratio of the two leading curvatures is the
inverse ratio of the two effective diffusions. -/
theorem leadingCurvature_ratio {eta ddL s₁ s₂ : ℝ} (heta : eta ≠ 0) (hddL : ddL ≠ 0)
    (hs₁ : s₁ ≠ 0) (hs₂ : s₂ ≠ 0) :
    leadingCurvature eta ddL s₁ / leadingCurvature eta ddL s₂ = s₂ / s₁ := by
  unfold leadingCurvature
  field_simp

/-- **The needle, at the level of the curvature.** As the effective diffusion at the minimum
tends to zero from above, the leading log-density curvature diverges. This is the direct
baseline's situation on the shoulder, where `sigma_eff^2 = s^2 sigma_obj^2` is tiny. The
divergence of a curvature is not by itself the collapse of a measure, and this statement does not
prove that collapse; `stationaryDensity_ratio_tendsto_zero` proves it, in the constant-diffusion
regime, as a statement about the density. -/
theorem leadingCurvature_tendsto_atTop {eta ddL : ℝ} (heta : 0 < eta) (hddL : 0 < ddL) :
    Tendsto (leadingCurvature eta ddL) (𝓝[>] (0 : ℝ)) atTop := by
  have hpos : (0 : ℝ) < 2 / eta * ddL := by positivity
  have h := Filter.Tendsto.const_mul_atTop hpos (tendsto_inv_nhdsGT_zero (𝕜 := ℝ))
  refine h.congr fun s => ?_
  unfold leadingCurvature
  ring

/-- **The comparison at the level of the densities.** On a band on which both effective
diffusions are constant — the (A2) band regime — and at a point where the shared drift `dLtilde`
has strictly positive derivative, the density of the run with the smaller effective diffusion
has the strictly larger log-density curvature. Both sides are built from the *same* drift and
the *same* `ddL`; what differs is only the effective diffusion by which that number is divided.
Note what `ddL` is here: the derivative of the drift function, which is the note's `Ltilde''`
only once `dLtilde` is identified with `Ltilde'` — `logDensityCurvature_lt_of_diffusion_lt_deriv`
below makes that identification a hypothesis and states the positivity as the note's own
`0 < Ltilde''(w*)`. The exact local constancy assumed of both effective diffusions is again an
idealisation of the note's (A2) band;
`logDensityCurvature_lt_of_diffusion_lt_of_correction_gap` proves the same comparison for
arbitrary effective diffusions, with the tolerance for their variation stated explicitly. -/
theorem logDensityCurvature_lt_of_diffusion_lt
    {C0d C0l eta cd cl : ℝ} {sigma2d sigma2l dLtilde Ud Ul : ℝ → ℝ} {w ddL : ℝ}
    (heta : 0 < eta) (hC0d : 0 < C0d) (hC0l : 0 < C0l)
    (hcd : 0 < cd) (hcl : cd < cl) (hddL : 0 < ddL)
    (hconstd : sigma2d =ᶠ[𝓝 w] fun _ => cd)
    (hconstl : sigma2l =ᶠ[𝓝 w] fun _ => cl)
    (hUd : ∀ᶠ u in 𝓝 w, HasDerivAt Ud (dLtilde u / sigma2d u) u)
    (hUl : ∀ᶠ u in 𝓝 w, HasDerivAt Ul (dLtilde u / sigma2l u) u)
    (hddLd : HasDerivAt dLtilde ddL w) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0l eta sigma2l Ul u)) w
      < -deriv (deriv fun u => Real.log (stationaryDensity C0d eta sigma2d Ud u)) w := by
  rw [logDensityCurvature_of_const_diffusion hC0d hcd hconstd hUd hddLd,
    logDensityCurvature_of_const_diffusion hC0l (hcd.trans hcl) hconstl hUl hddLd]
  exact leadingCurvature_lt_of_diffusion_lt heta hddL hcd hcl

/-- **The density-level comparison in the note's own symbols.**
`logDensityCurvature_lt_of_diffusion_lt` ties the shared drift `dLtilde` to no potential, so its
positivity hypothesis `0 < ddL` is a statement about the derivative of the drift. This form
closes that slack exactly as `logDensityCurvature_at_localMin_deriv` does for the identity:
`dLtilde` is required to be the derivative of `Ltilde` throughout a neighbourhood of `w`, and
the positive quantity is the note's own `Ltilde''(w*)`, written `deriv (deriv Ltilde) w`. Both
stationary densities are then governed by the *same* pullback loss `Ltilde`, and the run with
the smaller constant effective diffusion has the strictly larger log-density curvature — the
same `Ltilde''(w*)` divided by a smaller number, not convexity added to either landscape. At the
note's `w*` the hypothesis `0 < Ltilde''(w*)` is the nondegeneracy of the minimum; the
criticality of `w*` itself is not needed on a band of constant effective diffusion
(`logDensityCurvature_of_const_diffusion`), so it is not assumed. -/
theorem logDensityCurvature_lt_of_diffusion_lt_deriv
    {C0d C0l eta cd cl : ℝ} {Ltilde sigma2d sigma2l dLtilde Ud Ul : ℝ → ℝ} {w ddL : ℝ}
    (heta : 0 < eta) (hC0d : 0 < C0d) (hC0l : 0 < C0l)
    (hcd : 0 < cd) (hcl : cd < cl)
    (hconstd : sigma2d =ᶠ[𝓝 w] fun _ => cd)
    (hconstl : sigma2l =ᶠ[𝓝 w] fun _ => cl)
    (hUd : ∀ᶠ u in 𝓝 w, HasDerivAt Ud (dLtilde u / sigma2d u) u)
    (hUl : ∀ᶠ u in 𝓝 w, HasDerivAt Ul (dLtilde u / sigma2l u) u)
    (hL : ∀ᶠ u in 𝓝 w, HasDerivAt Ltilde (dLtilde u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hpos : 0 < deriv (deriv Ltilde) w) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0l eta sigma2l Ul u)) w
      < -deriv (deriv fun u => Real.log (stationaryDensity C0d eta sigma2d Ud u)) w := by
  have hderiv : deriv Ltilde =ᶠ[𝓝 w] dLtilde := by
    filter_upwards [hL] with u hu using hu.deriv
  have hdd : deriv (deriv Ltilde) w = ddL := by
    rw [hderiv.deriv_eq]
    exact hddL.deriv
  rw [hdd] at hpos
  exact logDensityCurvature_lt_of_diffusion_lt heta hC0d hC0l hcd hcl hpos hconstd hconstl
    hUd hUl hddL

/-- **The comparison without the constant-band idealisation.** The statement above compares two
runs whose effective diffusions are exactly constant near `w`, which is a stronger hypothesis
than the note's "approximately constant" (A2) band. Dropping it, the comparison becomes
conditional, and the condition is exactly what one would expect: at a common critical point of
the pullback loss, the run with the smaller effective diffusion has the strictly larger
log-density curvature provided the difference of the two sigma-variation corrections does not
swamp the gap between the two leading terms. Taking both effective diffusions locally constant
makes both corrections vanish and reduces `hgap` to
`0 < (2/eta) Ltilde''(w) (1/cd - 1/cl)`, which the hypotheses of the previous statement supply;
that statement is not literally a special case of this one, because on an exactly constant band
it needs no criticality hypothesis at all.

Two things should be said plainly about `hgap`. First, it is sharp: given the identity of
`neg_deriv_deriv_log_stationaryDensity_of_critical` it is *equivalent* to the conclusion, so the
mathematical content of this statement is that identity, and the inequality is bookkeeping.
Second, it constrains only the *difference* of the two sigma-variation corrections, and therefore
does not by itself tell a reader how much either effective diffusion may vary — two wildly varying
diffusions whose corrections happen to agree satisfy it as readily as two nearly constant ones.
`logDensityCurvature_lt_of_diffusion_lt_of_correction_bound` below states the tolerance in the
form a reader can check, one bound per effective diffusion. Both hypotheses are written entirely
in terms of `eta`, `Ltilde''(w*)` and the two effective diffusions, so neither requires knowing
the two densities in advance. -/
theorem logDensityCurvature_lt_of_diffusion_lt_of_correction_gap
    {C0d C0l eta : ℝ} {sigma2d dsigma2d sigma2l dsigma2l dLtilde Ud Ul : ℝ → ℝ}
    {w ddL ddsd ddsl : ℝ}
    (hC0d : 0 < C0d) (hC0l : 0 < C0l)
    (hsposd : ∀ᶠ u in 𝓝 w, 0 < sigma2d u)
    (hs1d : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2d (dsigma2d u) u)
    (hs2d : HasDerivAt dsigma2d ddsd w)
    (hUd : ∀ᶠ u in 𝓝 w, HasDerivAt Ud (dLtilde u / sigma2d u) u)
    (hsposl : ∀ᶠ u in 𝓝 w, 0 < sigma2l u)
    (hs1l : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2l (dsigma2l u) u)
    (hs2l : HasDerivAt dsigma2l ddsl w)
    (hUl : ∀ᶠ u in 𝓝 w, HasDerivAt Ul (dLtilde u / sigma2l u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hcrit : dLtilde w = 0)
    (hgap : deriv (deriv fun u => Real.log (sigma2l u)) w
          - deriv (deriv fun u => Real.log (sigma2d u)) w
        < 2 / eta * (ddL / sigma2d w) - 2 / eta * (ddL / sigma2l w)) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0l eta sigma2l Ul u)) w
      < -deriv (deriv fun u => Real.log (stationaryDensity C0d eta sigma2d Ud u)) w := by
  rw [neg_deriv_deriv_log_stationaryDensity_of_critical hC0d hsposd hs1d hs2d hUd hddL hcrit,
    neg_deriv_deriv_log_stationaryDensity_of_critical hC0l hsposl hs1l hs2l hUl hddL hcrit]
  linarith

/-- **The comparison with a checkable tolerance on each effective diffusion.** The hypothesis
`hgap` of the previous statement is sharp but joint: it bounds the difference of the two
sigma-variation corrections, and so does not say how much either effective diffusion may vary.
This form does. Each correction is bounded separately, `|(log sigma_eff^2)''(w*)| <= eps`, which
is the honest reading of the note's "approximately constant" (A2) band for a curvature statement,
and the tolerance `eps` is required only to satisfy `2 eps < ` the gap between the two leading
curvature terms. By `leadingCurvature_gap_pos` that gap is strictly positive whenever
`Ltilde''(w*) > 0` and the effective diffusion of the lift exceeds that of the direct baseline, so
the admissible band of `eps` is nonempty and the statement has content: any pair of effective
diffusions whose variation at `w*` is small enough relative to the gap — a condition on the
diffusions alone, checkable without the densities — gives the strict comparison, with the
corrections carried rather than assumed away. Exact local constancy is the case `eps = 0`. -/
theorem logDensityCurvature_lt_of_diffusion_lt_of_correction_bound
    {C0d C0l eta eps : ℝ} {sigma2d dsigma2d sigma2l dsigma2l dLtilde Ud Ul : ℝ → ℝ}
    {w ddL ddsd ddsl : ℝ}
    (hC0d : 0 < C0d) (hC0l : 0 < C0l)
    (hsposd : ∀ᶠ u in 𝓝 w, 0 < sigma2d u)
    (hs1d : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2d (dsigma2d u) u)
    (hs2d : HasDerivAt dsigma2d ddsd w)
    (hUd : ∀ᶠ u in 𝓝 w, HasDerivAt Ud (dLtilde u / sigma2d u) u)
    (hsposl : ∀ᶠ u in 𝓝 w, 0 < sigma2l u)
    (hs1l : ∀ᶠ u in 𝓝 w, HasDerivAt sigma2l (dsigma2l u) u)
    (hs2l : HasDerivAt dsigma2l ddsl w)
    (hUl : ∀ᶠ u in 𝓝 w, HasDerivAt Ul (dLtilde u / sigma2l u) u)
    (hddL : HasDerivAt dLtilde ddL w)
    (hcrit : dLtilde w = 0)
    (hvard : |deriv (deriv fun u => Real.log (sigma2d u)) w| ≤ eps)
    (hvarl : |deriv (deriv fun u => Real.log (sigma2l u)) w| ≤ eps)
    (htol : 2 * eps
        < leadingCurvature eta ddL (sigma2d w) - leadingCurvature eta ddL (sigma2l w)) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0l eta sigma2l Ul u)) w
      < -deriv (deriv fun u => Real.log (stationaryDensity C0d eta sigma2d Ud u)) w := by
  have hd := abs_le.mp hvard
  have hl := abs_le.mp hvarl
  simp only [leadingCurvature] at htol
  exact logDensityCurvature_lt_of_diffusion_lt_of_correction_gap hC0d hC0l hsposd hs1d hs2d hUd
    hsposl hs1l hs2l hUl hddL hcrit (by linarith [hd.1, hd.2, hl.1, hl.2])

/-! ### The needle, at the level of the measure -/

/-- **The needle.** The comparison results above are statements about the number
`-(log pi)''(w*)`. This one is about the density itself, and it is what the note's word "needle"
means. On a band on which the effective diffusion is constant and equal to `c`, the effective
potential of `(*)` is `U_eff = Ltilde/c` — an antiderivative of `Ltilde'/sigma_eff^2` for that
diffusion — and `(*)` reduces to the Boltzmann form `pi ∝ exp (-(2/(eta c)) Ltilde)`. Then for
every point `u` whose pullback loss strictly exceeds that of `w`, the density at `u` relative to
the density at `w` tends to zero as `c` tends to zero from above:

`pi_c(u) / pi_c(w) = exp (-(2/eta) (Ltilde(u) - Ltilde(w))/c) → 0`.

Taking `w = w*` a local minimum, this is the note's needle for a run whose effective diffusion on
the shoulder is driven to zero — the direct baseline, whose `sigma_eff^2 = s^2 sigma_obj^2` is
tiny there: every point of strictly larger pullback loss is suppressed relative to `w*` by an
arbitrarily large factor. The ratio is taken because the statement is about the shape of the
density and not about its normalisation; by `stationaryDensity_div_const` the ratio is unchanged
if `pi_c` is replaced by the normalised density. Two things are deliberately not claimed. This is
a pointwise statement about the density, not a convergence statement about the measures: that the
normalised measures converge weakly to the point mass at `w*` would need an integrability argument
on top of it and is not proved here. And nothing is claimed for a non-constant effective
diffusion, where `(*)` is not of Boltzmann form and the sigma-variation prefactor
`1/sigma_eff^2` does not cancel in the ratio. -/
theorem stationaryDensity_ratio_tendsto_zero {C0 eta : ℝ} {Ltilde : ℝ → ℝ} {w u : ℝ}
    (hC0 : 0 < C0) (heta : 0 < eta) (hgt : Ltilde w < Ltilde u) :
    Tendsto (fun c : ℝ =>
        stationaryDensity C0 eta (fun _ => c) (fun v => Ltilde v / c) u
          / stationaryDensity C0 eta (fun _ => c) (fun v => Ltilde v / c) w)
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hk : 0 < 2 / eta * (Ltilde u - Ltilde w) :=
    mul_pos (by positivity) (sub_pos.mpr hgt)
  have h1 : Tendsto (fun c : ℝ => 2 / eta * (Ltilde u - Ltilde w) * c⁻¹) (𝓝[>] (0 : ℝ)) atTop :=
    Filter.Tendsto.const_mul_atTop hk (tendsto_inv_nhdsGT_zero (𝕜 := ℝ))
  have h2 : Tendsto (fun c : ℝ => -(2 / eta * (Ltilde u - Ltilde w) * c⁻¹))
      (𝓝[>] (0 : ℝ)) atBot := tendsto_neg_atTop_atBot.comp h1
  have h3 : Tendsto (fun c : ℝ => Real.exp (-(2 / eta * (Ltilde u - Ltilde w) * c⁻¹)))
      (𝓝[>] (0 : ℝ)) (𝓝 0) := Real.tendsto_exp_atBot.comp h2
  refine h3.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with c hc
  have hcne : c ≠ 0 := ne_of_gt hc
  have hC0c : C0 / c ≠ 0 := div_ne_zero hC0.ne' hcne
  simp only [stationaryDensity]
  rw [mul_div_mul_comm, div_self hC0c, one_mul, ← Real.exp_sub]
  congr 1
  field_simp
  ring

/-! ### Witnesses: the hypotheses are satisfiable, and the correction is genuinely present -/

/-- The sigma-variation correction is not identically zero, so computing it is not a formality.
For the admissible effective diffusion `sigma_eff^2(u) = 1 + u^2` the correction at the origin
is `2`. Any claim that the curvature at a minimum equals the leading term alone is therefore
false without a hypothesis on the variation of `sigma_eff^2`, such as the local constancy of the
(A2) band regime. -/
theorem deriv_deriv_log_one_add_sq :
    deriv (deriv fun u : ℝ => Real.log (1 + u ^ 2)) 0 = 2 := by
  have hne : ∀ᶠ u in 𝓝 (0 : ℝ), (1 + u ^ 2 : ℝ) ≠ 0 :=
    Filter.Eventually.of_forall fun u => by positivity
  have hs1 : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 1 + v ^ 2) (2 * u) u :=
    Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (1 : ℝ)
  have hs2 : HasDerivAt (fun u : ℝ => 2 * u) 2 (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)
  rw [deriv_deriv_log_diffusion hne hs1 hs2]
  norm_num

/-- The hypothesis bundle of `logDensityCurvature_of_const_diffusion` is satisfiable, and on its
simplest instance the identity returns the classical answer. This witnesses the constant-diffusion
band only, where the sigma-variation correction is zero; `logDensityCurvature_localMin_witness`
witnesses the full bundle of `logDensityCurvature_at_localMin`, with a non-constant `sigma_eff^2`
and a correction that does not vanish. Take `sigma_eff^2` constant equal to `c > 0` and the
quadratic
pullback loss `Ltilde(v) = a v^2 / 2`, whose effective potential is
`U_eff(v) = a v^2 / (2c)`; the stationary density is then the Gaussian
`pi(u) ∝ exp (-a u^2 / (eta c))`, and the curvature of its logarithm at the origin is
`2 a / (eta c)`, which is `(2/eta) Ltilde''(0) / sigma_eff^2(0)` with no correction. The identity
is stated for every real `a`, since `logDensityCurvature_of_const_diffusion` needs no criticality
hypothesis on a constant band; it is for `a > 0` that the origin is a minimum of `Ltilde` and the
number is the curvature at a minimum in the sense of (L1'). -/
theorem logDensityCurvature_quadratic {C0 eta a c : ℝ} (hC0 : 0 < C0) (hc : 0 < c) :
    -deriv (deriv fun u => Real.log
        (stationaryDensity C0 eta (fun _ => c) (fun v => a * v ^ 2 / (2 * c)) u)) 0
      = 2 / eta * (a / c) := by
  have hcne : c ≠ 0 := hc.ne'
  have hconst : (fun _ : ℝ => c) =ᶠ[𝓝 (0 : ℝ)] fun _ => c := Filter.EventuallyEq.rfl
  have hU : ∀ᶠ u in 𝓝 (0 : ℝ),
      HasDerivAt (fun v : ℝ => a * v ^ 2 / (2 * c)) (a * u / c) u := by
    refine Filter.Eventually.of_forall fun u => ?_
    have h : HasDerivAt (fun v : ℝ => a * v ^ 2 / (2 * c)) (a * (2 * u) / (2 * c)) u := by
      simpa using ((hasDerivAt_pow 2 u).const_mul a).div_const (2 * c)
    convert h using 1
    field_simp
  have hddL : HasDerivAt (fun x : ℝ => a * x) a (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul a
  exact logDensityCurvature_of_const_diffusion (dLtilde := fun x => a * x) hC0 hc hconst hU hddL

/-- **A witness for the full hypothesis bundle of (L1') at a local minimum, with a non-constant
effective diffusion.** `logDensityCurvature_quadratic` witnesses only the constant-diffusion
bundle, in which the sigma-variation correction is zero; a bundle of hypotheses that can be met
only when the object it computes vanishes would prove nothing about that object. This witness
meets every hypothesis of the flagship form `logDensityCurvature_at_localMin_deriv` — positivity
and differentiability of `sigma_eff^2` on a neighbourhood, an antiderivative `U_eff` of
`Ltilde'/sigma_eff^2`, `dLtilde` equal to the derivative of `Ltilde` on a neighbourhood (here on
all of `ℝ`), differentiability of `Ltilde'`, and `IsLocalMin` — and hence every hypothesis of
`logDensityCurvature_at_localMin` as well, with a `sigma_eff^2` that genuinely varies.
Take `sigma_eff^2(u) = 1 + u^2`, the pullback loss `Ltilde(v) = a v^2 / 2` with `a > 0`, so that
`0` is a local minimum with `Ltilde''(0) = a`, and the effective potential
`U_eff(v) = (a/2) log (1 + v^2)`, which satisfies `U_eff' = Ltilde'/sigma_eff^2`. The stationary
density is `pi(u) = C0 (1 + u^2)^(-(1 + a/eta))` and its log-density curvature at the minimum is

`-(log pi)''(0) = (2/eta) a + 2`,

the leading term `(2/eta) Ltilde''(0)/sigma_eff^2(0) = (2/eta) a` plus the sigma-variation
correction `(log sigma_eff^2)''(0) = 2`. The correction is therefore not a formality: on this
instance, reporting the leading term alone misstates the curvature by exactly `2`, whatever the
learning rate. -/
theorem logDensityCurvature_localMin_witness {C0 eta a : ℝ} (hC0 : 0 < C0) (ha : 0 < a) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta (fun v => 1 + v ^ 2)
        (fun v => a / 2 * Real.log (1 + v ^ 2)) u)) 0
      = 2 / eta * a + 2 := by
  have hspos : ∀ᶠ u in 𝓝 (0 : ℝ), 0 < (1 : ℝ) + u ^ 2 :=
    Filter.Eventually.of_forall fun u => by positivity
  have hs1 : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 1 + v ^ 2) (2 * u) u :=
    Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (1 : ℝ)
  have hs2 : HasDerivAt (fun u : ℝ => 2 * u) 2 (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)
  have hU : ∀ᶠ u in 𝓝 (0 : ℝ),
      HasDerivAt (fun v : ℝ => a / 2 * Real.log (1 + v ^ 2)) (a * u / (1 + u ^ 2)) u := by
    refine Filter.Eventually.of_forall fun u => ?_
    have hne : (1 : ℝ) + u ^ 2 ≠ 0 := by positivity
    have h1 : HasDerivAt (fun v : ℝ => 1 + v ^ 2) (2 * u) u := by
      simpa using (hasDerivAt_pow 2 u).const_add (1 : ℝ)
    have h2 : HasDerivAt (fun v : ℝ => Real.log (1 + v ^ 2)) (2 * u / (1 + u ^ 2)) u := h1.log hne
    have heq : a * u / (1 + u ^ 2) = a / 2 * (2 * u / (1 + u ^ 2)) := by ring
    rw [heq]
    exact h2.const_mul (a / 2)
  have hL : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => a * v ^ 2 / 2) (a * u) u := by
    refine Filter.Eventually.of_forall fun u => ?_
    have h : HasDerivAt (fun v : ℝ => a * v ^ 2 / 2) (a * (2 * u) / 2) u := by
      simpa using ((hasDerivAt_pow 2 u).const_mul a).div_const 2
    convert h using 1
    ring
  have hddL : HasDerivAt (fun v : ℝ => a * v) a (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul a
  have hmin : IsLocalMin (fun v : ℝ => a * v ^ 2 / 2) 0 := by
    refine Filter.Eventually.of_forall fun x => ?_
    have hx : (0 : ℝ) ≤ a * x ^ 2 / 2 := by positivity
    simpa using hx
  have hdd2 : deriv (deriv fun v : ℝ => a * v ^ 2 / 2) 0 = a := by
    have hderiv : deriv (fun v : ℝ => a * v ^ 2 / 2) =ᶠ[𝓝 (0 : ℝ)] fun v => a * v := by
      filter_upwards [hL] with u hu using hu.deriv
    rw [hderiv.deriv_eq]
    exact hddL.deriv
  have h := logDensityCurvature_at_localMin_deriv (C0 := C0) (eta := eta)
    (Ltilde := fun v : ℝ => a * v ^ 2 / 2) (sigma2 := fun v : ℝ => 1 + v ^ 2)
    (dsigma2 := fun v : ℝ => 2 * v) (dLtilde := fun v : ℝ => a * v)
    (U := fun v : ℝ => a / 2 * Real.log (1 + v ^ 2)) (w := 0) (ddL := a) (dds := 2)
    hC0 hspos hs1 hs2 hU hL hddL hmin
  rw [h, hdd2]
  norm_num [deriv_deriv_log_one_add_sq]

/-- **A witness for the density-level comparison.** The hypotheses of
`logDensityCurvature_lt_of_diffusion_lt` are met by the quadratic instance above run at two
different constant effective diffusions `cd < cl` with the *same* pullback loss
`Ltilde(v) = a v^2/2`: the curvature at the minimum is `(2/eta) a/cl` for the larger diffusion
and `(2/eta) a/cd` for the smaller. Nothing has been added to the loss; the same `Ltilde''(0) = a`
has been divided by a larger number. -/
theorem logDensityCurvature_lt_quadratic {C0 eta a cd cl : ℝ} (heta : 0 < eta) (hC0 : 0 < C0)
    (ha : 0 < a) (hcd : 0 < cd) (hcl : cd < cl) :
    -deriv (deriv fun u => Real.log
        (stationaryDensity C0 eta (fun _ => cl) (fun v => a * v ^ 2 / (2 * cl)) u)) 0
      < -deriv (deriv fun u => Real.log
        (stationaryDensity C0 eta (fun _ => cd) (fun v => a * v ^ 2 / (2 * cd)) u)) 0 := by
  rw [logDensityCurvature_quadratic hC0 (hcd.trans hcl), logDensityCurvature_quadratic hC0 hcd]
  exact leadingCurvature_lt_of_diffusion_lt heta ha hcd hcl

/-- **A witness for the un-idealised comparison, with both corrections nonzero.** The
density-level comparison is witnessed by `logDensityCurvature_lt_quadratic` only in the
constant-diffusion regime, in which both sigma-variation corrections vanish; if the
correction-gap form `logDensityCurvature_lt_of_diffusion_lt_of_correction_gap` were satisfiable
only there, its generality over the constant-band statement would be empty. It is not. Take the
shared drift `dLtilde(v) = a v` with `a > 0`, critical at `0`, and the two genuinely varying
effective diffusions `sigma_eff,d^2(u) = 1 + u^2` and `sigma_eff,l^2(u) = 4 + u^2`, with
effective potentials `(a/2) log (1 + u^2)` and `(a/2) log (4 + u^2)`. The corrections at `0`
are `2` and `1/2` — both nonzero — and the gap hypothesis reads
`1/2 - 2 < (2/eta) (a - a/4)`, which holds for every `a > 0` and `eta > 0` because the left
side is negative and the right side positive. The comparison then fires with the corrections
carried, not assumed away: the run whose effective diffusion is smaller at the common critical
point has the strictly larger log-density curvature. -/
theorem logDensityCurvature_lt_correction_gap_witness {C0 eta a : ℝ}
    (heta : 0 < eta) (hC0 : 0 < C0) (ha : 0 < a) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta (fun v => 4 + v ^ 2)
        (fun v => a / 2 * Real.log (4 + v ^ 2)) u)) 0
      < -deriv (deriv fun u => Real.log (stationaryDensity C0 eta (fun v => 1 + v ^ 2)
        (fun v => a / 2 * Real.log (1 + v ^ 2)) u)) 0 := by
  -- The correction of the larger diffusion: `(log (4 + u^2))''(0) = 1/2`.
  have h4 : deriv (deriv fun u : ℝ => Real.log (4 + u ^ 2)) 0 = 1 / 2 := by
    have hne : ∀ᶠ u in 𝓝 (0 : ℝ), (4 + u ^ 2 : ℝ) ≠ 0 :=
      Filter.Eventually.of_forall fun u => by positivity
    have hs1 : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 4 + v ^ 2) (2 * u) u :=
      Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (4 : ℝ)
    have hs2 : HasDerivAt (fun u : ℝ => 2 * u) 2 (0 : ℝ) := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)
    rw [deriv_deriv_log_diffusion hne hs1 hs2]
    norm_num
  have hsposd : ∀ᶠ u in 𝓝 (0 : ℝ), 0 < (1 : ℝ) + u ^ 2 :=
    Filter.Eventually.of_forall fun u => by positivity
  have hsposl : ∀ᶠ u in 𝓝 (0 : ℝ), 0 < (4 : ℝ) + u ^ 2 :=
    Filter.Eventually.of_forall fun u => by positivity
  have hs1d : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 1 + v ^ 2) (2 * u) u :=
    Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (1 : ℝ)
  have hs1l : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 4 + v ^ 2) (2 * u) u :=
    Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (4 : ℝ)
  have hs2 : HasDerivAt (fun u : ℝ => 2 * u) 2 (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)
  have hU : ∀ c : ℝ, 0 < c → ∀ᶠ u in 𝓝 (0 : ℝ),
      HasDerivAt (fun v : ℝ => a / 2 * Real.log (c + v ^ 2)) (a * u / (c + u ^ 2)) u := by
    intro c hc
    refine Filter.Eventually.of_forall fun u => ?_
    have hne : (c : ℝ) + u ^ 2 ≠ 0 := by positivity
    have h1 : HasDerivAt (fun v : ℝ => c + v ^ 2) (2 * u) u := by
      simpa using (hasDerivAt_pow 2 u).const_add c
    have h2 : HasDerivAt (fun v : ℝ => Real.log (c + v ^ 2)) (2 * u / (c + u ^ 2)) u :=
      h1.log hne
    have heq : a * u / (c + u ^ 2) = a / 2 * (2 * u / (c + u ^ 2)) := by ring
    rw [heq]
    exact h2.const_mul (a / 2)
  have hddL : HasDerivAt (fun v : ℝ => a * v) a (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul a
  have hcrit : (fun v : ℝ => a * v) 0 = 0 := mul_zero a
  have hgap : deriv (deriv fun u : ℝ => Real.log (4 + u ^ 2)) 0
        - deriv (deriv fun u : ℝ => Real.log (1 + u ^ 2)) 0
      < 2 / eta * (a / (1 + (0 : ℝ) ^ 2)) - 2 / eta * (a / (4 + (0 : ℝ) ^ 2)) := by
    rw [h4, deriv_deriv_log_one_add_sq]
    have hpos : 0 < 2 / eta * a := by positivity
    norm_num
    nlinarith [hpos]
  exact logDensityCurvature_lt_of_diffusion_lt_of_correction_gap (C0d := C0) (C0l := C0)
    (eta := eta) (sigma2d := fun v : ℝ => 1 + v ^ 2) (dsigma2d := fun v : ℝ => 2 * v)
    (sigma2l := fun v : ℝ => 4 + v ^ 2) (dsigma2l := fun v : ℝ => 2 * v)
    (dLtilde := fun v : ℝ => a * v)
    (Ud := fun v : ℝ => a / 2 * Real.log (1 + v ^ 2))
    (Ul := fun v : ℝ => a / 2 * Real.log (4 + v ^ 2))
    (w := 0) (ddL := a) (ddsd := 2) (ddsl := 2)
    hC0 hC0 hsposd hs1d hs2 (hU 1 one_pos) hsposl hs1l hs2 (hU 4 (by norm_num)) hddL hcrit hgap

/-- **A witness for the bounded-tolerance comparison, with both corrections nonzero.** The
tolerance form `logDensityCurvature_lt_of_diffusion_lt_of_correction_bound` would be of no use if
it could be met only with `eps = 0`, that is, only on the exactly constant band it exists to
generalise. It cannot be. Take the same two genuinely varying effective diffusions as above,
`sigma_eff,d^2(u) = 1 + u^2` and `sigma_eff,l^2(u) = 4 + u^2`, whose sigma-variation corrections
at the origin are `2` and `1/2`, and the shared drift `dLtilde(v) = a v`, critical at the origin.
Both corrections are bounded by `eps = 2`, and the gap between the leading curvature terms is
`3a/(2 eta)`, so the tolerance `2 eps < 3a/(2 eta)` is the condition `8 eta < 3 a` — satisfied, for
instance, by every `a > 3` at learning rate `eta = 1`. The comparison then fires with a strictly
positive tolerance on each effective diffusion separately, and with neither correction assumed to
vanish. -/
theorem logDensityCurvature_lt_correction_bound_witness {C0 eta a : ℝ}
    (hC0 : 0 < C0) (heta : 0 < eta) (hlarge : 8 * eta < 3 * a) :
    -deriv (deriv fun u => Real.log (stationaryDensity C0 eta (fun v => 4 + v ^ 2)
        (fun v => a / 2 * Real.log (4 + v ^ 2)) u)) 0
      < -deriv (deriv fun u => Real.log (stationaryDensity C0 eta (fun v => 1 + v ^ 2)
        (fun v => a / 2 * Real.log (1 + v ^ 2)) u)) 0 := by
  have ha : 0 < a := by nlinarith
  -- The correction of the larger effective diffusion at the origin.
  have h4 : deriv (deriv fun u : ℝ => Real.log (4 + u ^ 2)) 0 = 1 / 2 := by
    have hne : ∀ᶠ u in 𝓝 (0 : ℝ), (4 + u ^ 2 : ℝ) ≠ 0 :=
      Filter.Eventually.of_forall fun u => by positivity
    have hs1 : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 4 + v ^ 2) (2 * u) u :=
      Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (4 : ℝ)
    have hs2 : HasDerivAt (fun u : ℝ => 2 * u) 2 (0 : ℝ) := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)
    rw [deriv_deriv_log_diffusion hne hs1 hs2]
    norm_num
  have hsposd : ∀ᶠ u in 𝓝 (0 : ℝ), 0 < (1 : ℝ) + u ^ 2 :=
    Filter.Eventually.of_forall fun u => by positivity
  have hsposl : ∀ᶠ u in 𝓝 (0 : ℝ), 0 < (4 : ℝ) + u ^ 2 :=
    Filter.Eventually.of_forall fun u => by positivity
  have hs1d : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 1 + v ^ 2) (2 * u) u :=
    Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (1 : ℝ)
  have hs1l : ∀ᶠ u in 𝓝 (0 : ℝ), HasDerivAt (fun v : ℝ => 4 + v ^ 2) (2 * u) u :=
    Filter.Eventually.of_forall fun u => by simpa using (hasDerivAt_pow 2 u).const_add (4 : ℝ)
  have hs2 : HasDerivAt (fun u : ℝ => 2 * u) 2 (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)
  have hU : ∀ c : ℝ, 0 < c → ∀ᶠ u in 𝓝 (0 : ℝ),
      HasDerivAt (fun v : ℝ => a / 2 * Real.log (c + v ^ 2)) (a * u / (c + u ^ 2)) u := by
    intro c hc
    refine Filter.Eventually.of_forall fun u => ?_
    have hne : (c : ℝ) + u ^ 2 ≠ 0 := by positivity
    have h1 : HasDerivAt (fun v : ℝ => c + v ^ 2) (2 * u) u := by
      simpa using (hasDerivAt_pow 2 u).const_add c
    have h2 : HasDerivAt (fun v : ℝ => Real.log (c + v ^ 2)) (2 * u / (c + u ^ 2)) u :=
      h1.log hne
    have heq : a * u / (c + u ^ 2) = a / 2 * (2 * u / (c + u ^ 2)) := by ring
    rw [heq]
    exact h2.const_mul (a / 2)
  have hddL : HasDerivAt (fun v : ℝ => a * v) a (0 : ℝ) := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul a
  have hcrit : (fun v : ℝ => a * v) 0 = 0 := mul_zero a
  have hvard : |deriv (deriv fun u : ℝ => Real.log (1 + u ^ 2)) 0| ≤ 2 := by
    rw [deriv_deriv_log_one_add_sq]
    norm_num
  have hvarl : |deriv (deriv fun u : ℝ => Real.log (4 + u ^ 2)) 0| ≤ 2 := by
    rw [h4]
    rw [abs_of_pos (by norm_num : (0:ℝ) < 1 / 2)]
    norm_num
  have htol : 2 * (2 : ℝ)
      < leadingCurvature eta a ((1 : ℝ) + 0 ^ 2) - leadingCurvature eta a ((4 : ℝ) + 0 ^ 2) := by
    have hgapval : leadingCurvature eta a ((1 : ℝ) + 0 ^ 2)
        - leadingCurvature eta a ((4 : ℝ) + 0 ^ 2) = 3 * a / (2 * eta) := by
      simp only [leadingCurvature]
      field_simp
      ring
    rw [hgapval, lt_div_iff₀ (by positivity : (0 : ℝ) < 2 * eta)]
    linarith
  exact logDensityCurvature_lt_of_diffusion_lt_of_correction_bound (C0d := C0) (C0l := C0)
    (eta := eta) (eps := 2) (sigma2d := fun v : ℝ => 1 + v ^ 2) (dsigma2d := fun v : ℝ => 2 * v)
    (sigma2l := fun v : ℝ => 4 + v ^ 2) (dsigma2l := fun v : ℝ => 2 * v)
    (dLtilde := fun v : ℝ => a * v)
    (Ud := fun v : ℝ => a / 2 * Real.log (1 + v ^ 2))
    (Ul := fun v : ℝ => a / 2 * Real.log (4 + v ^ 2))
    (w := 0) (ddL := a) (ddsd := 2) (ddsl := 2)
    hC0 hC0 hsposd hs1d hs2 (hU 1 one_pos) hsposl hs1l hs2 (hU 4 (by norm_num)) hddL hcrit
    hvard hvarl htol

/-! ### Convention -/

/-- Second derivatives are written `deriv (deriv f)` throughout this file; this is the
`iteratedDeriv 2` of mathlib's iterated-derivative API. -/
theorem iteratedDeriv_two_eq_deriv_deriv (f : ℝ → ℝ) : iteratedDeriv 2 f = deriv (deriv f) := by
  rw [iteratedDeriv_succ, iteratedDeriv_one]

end IcnnLift
