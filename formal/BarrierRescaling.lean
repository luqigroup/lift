import Mathlib
import TraceLemmas

/-!
# (L2') Barrier rescaling by the effective diffusion, and the corrected Kramers exponent

This file machine-checks claim (L2') of `docs/design/lemma1_rederivation.md` §3 — the claim the
note itself calls "the result that actually matters" — and with it the corrected replacement for
the paper's Corollary 1.

The corrected mechanism replaces the shipped Hessian decomposition, which §1 of the same note
refutes (differentiation commutes with the batch expectation, so the pullback Hessian carries no
cross-covariance term), by a statement about the **diffusion** of the bias-channel dynamics. The
one-dimensional Itô diffusion `d w = -Ltilde'(w) dt + sqrt(eta) sigma_eff(w) dB` with
state-dependent noise has, at zero flux, the stationary density
`pi(w) ∝ sigma_eff(w)^{-2} exp(-(2/eta) U_eff(w))` with the **quasipotential**
`U_eff(w) = ∫^w Ltilde'(u) / sigma_eff^2(u) du` — equation (*) of §3. The quasipotential, not
`Ltilde` itself, is what an escape argument sees, and the content of (L2') is that a barrier of
action `alpha = Ltilde(w_s) - Ltilde(w_b)` crossed on a band where `sigma_eff^2` is constant has
quasipotential action `alpha_eff = alpha / sigma_eff^2` exactly.

Two things are proved here that the note does not prove. First, the constant-band case is proved
as an **exact identity**, not to leading order: it is the fundamental theorem of calculus applied
to `U_eff' = Ltilde' / sigma_eff^2`, and nothing is discarded. Second — and this is the point of
the file — the note's phrase "crossed within the shoulder band where `sigma_eff^2` is
approximately constant" is replaced by a **two-sided bound**: if `sigma_eff^2` is merely trapped
between `c_min > 0` and `c_max` on the band, and the potential is non-decreasing across it, then
`alpha / c_max ≤ U_eff(w_s) - U_eff(w_b) ≤ alpha / c_min`. It degenerates to the exact identity
when `c_min = c_max`, and it is what makes the band argument honest: the approximation error is
carried explicitly in the width of the bracket rather than assumed away. It is not, however, a
strengthening of the note's sentence in the logical sense, and is not advertised as one: it adds
the monotone-climb hypothesis `Ltilde' ≥ 0` on the band, which the note does not ask for and
which forbids intervening critical points of `Ltilde` between `w_b` and `w_s`. What it buys in
exchange is that no approximation is made anywhere.

The second half of the file is the exponent comparison itself. The Arrhenius exponent
`arrheniusExponent alpha eta sigma2 = 2 * alpha / (eta * sigma2)` is proved strictly decreasing
in `sigma2` on the positive reals, hence so is `exp` of it, and the comparison is then
instantiated at the two effective diffusions of the corrected decomposition of §2: the direct
baseline `s^2 sigma_obj^2`, and the lift's `s^2 sigma_obj^2 + c1 sigma_Jac^2 |H| + s^2 (HVH)`.
The sign of the coupling channel enters here as the named hypothesis `hcoupling`, and it is a
genuine assumption — the note's (A6) of §11.3 — not a derived fact: §11.3 retracts §2's "the
sign comes for free" sentence, and `CouplingSign.lean` machine-checks the retraction, exhibiting
a configuration that satisfies (A4')'s own remainder-control clause while its coupling term is
strictly negative. The sign of the quadratic channel is not assumed at all in the matrix
instantiation, where it is discharged from `IcnnLift.quadForm_conj_nonneg` of `TraceLemmas.lean`.

**What this file does NOT prove, and must not be read as proving.** It is a statement about
**exponents only**. The Kramers prefactor is nowhere addressed, and neither is the
`sigma_eff^{-2}` prefactor that (*) puts in front of the exponential — both differ between the
two methods, and both are discarded here. How much is discarded is made explicit rather than
hidden: `escapeTime_ratio_of_kramers_of_prefactors` carries the two prefactors separately and
shows that the whole content of the exponent-only convention is the factor `C_dir / C_lift` it
sets to one. The passage from a quasipotential gap to a mean first-passage time is
`KramersExitTime.lean`'s business and is not available in this mathlib at all: mathlib v4.31.0
carries a Brownian motion and the hitting time of a process, but no stochastic integral, no
solution theory for stochastic differential equations, no Fokker-Planck equation and no mean
first-passage-time formula for a diffusion, so no Arrhenius law can be derived here.
Accordingly the escape-time statements below come in two forms: an unconditional one about
`Real.exp` of the exponents, which is a theorem, and one about a time `tau` which carries the
Kramers relation
`tau sigma2 = C * exp (arrheniusExponent alpha eta sigma2)` as the explicit named hypothesis
`hkramers`. Discharging `hkramers` would require the one-dimensional mean-first-passage-time
formula for a diffusion with state-dependent noise, that is, an Itô-calculus development that
mathlib does not have. Likewise the stationary density (*) itself is not proved here; only its
differentiated consequence `U_eff' = Ltilde' / sigma_eff^2` is used, and it is carried as the
hypothesis `hUeff`, to be supplied by `StationaryDensity.lean`.

## Results
* `arrheniusExponent` — the Kramers exponent `2 alpha / (eta sigma2)`.
* `arrheniusFactor` — its exponential, the exponent-only escape-time prediction.
* `directDiffusion`, `liftDiffusion` — the two effective diffusions of the corrected §2 form.
* `escapeTimeRatio` — `exp (2 alpha (1/sigma2_dir - 1/sigma2_lift) / eta)`.
* `quasipotential_gap_of_const_diffusion` — exact: constant `sigma_eff^2 = c` on the band gives
  `U_eff(w_s) - U_eff(w_b) = alpha / c`.
* `barrier_exponent_of_const_diffusion` — hence the barrier enters the density as
  `(2/eta) (U_eff(w_s) - U_eff(w_b)) = arrheniusExponent alpha eta c`.
* `quasipotential_gap_bounds` — non-constant band: `alpha / c_max ≤ U_eff gap ≤ alpha / c_min`.
* `quasipotential_gap_bounds_of_continuousOn` — the same with the two integrability hypotheses
  discharged from continuity.
* `barrier_exponent_bounds` — the two-sided bound in Arrhenius-exponent form.
* `quasipotential_gap_bounds_eq_of_const` — the bracket collapses to the exact identity.
* `quasipotential_gap_bounds_witness_strict` — the numerical core of the witness below:
  `1/2 < log 2 < 1`, the strict bracket at the witness instance.
* `quasipotential_gap_bounds_witness` — an existential statement, checked by the kernel in the
  statement itself and not merely in a proof, that the band hypotheses are jointly satisfiable
  by a genuinely non-constant diffusion (`sigma2 wb ≠ sigma2 ws`) and that the bracket at such
  an instance is strict on both sides, hence not an identity in disguise.
* `arrheniusExponent_strictAntiOn`, `arrheniusFactor_strictAntiOn` — strict decrease in `sigma2`.
* `directDiffusion_lt_liftDiffusion` — the lift's effective diffusion strictly exceeds the
  direct baseline's under a non-negative coupling channel, a non-negative quadratic channel and
  non-degeneracy of one of the two.
* `arrheniusExponent_lift_lt_direct`, `arrheniusFactor_lift_lt_direct` — the strict ordering of
  the two exponents, and of the two exponent-only escape-time predictions.
* `arrheniusExponent_lift_lt_direct_of_quadForm` — the same with the quadratic channel's sign
  proved rather than assumed, from `TraceLemmas.quadForm_conj_nonneg`.
* `quadForm_channel_nonvacuous` — the quadratic channel can be strictly positive, so the second
  disjunct of the non-degeneracy hypothesis is satisfiable.
* `escapeTimeRatio_eq_div` — the ratio identity.
* `one_lt_escapeTimeRatio` — the ratio exceeds one.
* `one_lt_escapeTimeRatio_lift` — the same at the two effective diffusions of the corrected
  decomposition.
* `escapeTimeRatio_strictMono_lift` — it grows strictly as the excess diffusion grows.
* `escapeTime_ratio_of_kramers` — the ratio of predicted escape times, under `hkramers`.
* `escapeTime_ratio_of_kramers_of_prefactors` — the same with the two Kramers prefactors kept
  distinct, exhibiting the factor the exponent-only convention discards.
* `barrier_exponent_lift_lt_direct_of_const_bands` — the two halves joined: on a band where each
  method's effective diffusion is constant, the lift's true quasipotential barrier exponent is
  strictly below the direct baseline's.
* `barrier_exponent_lift_lt_direct_of_pointwise` — the same for non-constant bands under the
  *pointwise* ordering `0 < sigma_eff^{2,dir}(u) < sigma_eff^{2,lift}(u)` on the band, which is
  the form in which §2 delivers the ordering and which needs no bracket on either diffusion.
* `barrier_exponent_lift_lt_direct_of_bands` — its corollary for bracketed measurements, under
  the separation hypothesis `c_max^dir < c_min^lift`. Separation is sufficient, not necessary.
* `barrier_exponent_lift_lt_direct_of_channels` — the pointwise comparison instantiated at the
  corrected §2 decomposition, with the coupling and quadratic channels carried per iterate of
  the band rather than as two constants.
* `barrier_exponent_lift_lt_direct_of_channels_witness` — an existential certifying that the
  channel form's nine band hypotheses are jointly satisfiable with a coupling channel that
  varies along the band and an identically vanishing quadratic channel, together with the
  instantiated conclusion.
* `barrier_exponent_lift_lt_direct_of_bands_witness` — an existential statement that every
  hypothesis of the joined band theorem is satisfiable at once, with a genuinely non-constant
  lift diffusion (`sigma2Lift wb ≠ sigma2Lift ws`) and the pointwise ordering that
  `barrier_exponent_lift_lt_direct_of_pointwise` consumes, together with the instantiated
  conclusion, so neither headline comparison is vacuously quantified; the certificate lives in
  the statement rather than in a proof a later edit could hollow out.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. The explicit hypotheses carried, and what each would take to
discharge:

* `hUeff : ∀ u ∈ ..., HasDerivAt Ueff (dL u / sigma2 u) u` — the differentiated form of the
  stationary density (*). It is a hypothesis because (*) is a statement about the stationary
  density of an Itô diffusion with state-dependent noise, and mathlib v4.31.0 has no stochastic
  integral and no theory of stationary densities for stochastic differential equations; it is
  `StationaryDensity.lean`'s statement to supply.
* `hLtilde`, and the interval-integrability hypotheses `hintL`, `hintU` — ordinary regularity of
  `Ltilde` on the band, supplied in the paper by (A1); the `_of_continuousOn` variant discharges
  the integrability from continuity of `Ltilde'` and `sigma_eff^2`.
* `hgrad : 0 ≤ Ltilde'` on the band, in every bounds theorem — the barrier is climbed
  monotonically from the basin to the saddle. This is strictly stronger than the note's
  `alpha > 0`: it forbids intervening critical points of `Ltilde` strictly between `w_b` and
  `w_s`, which is the standard single-barrier picture but is an added restriction. It cannot be
  dropped, because the interval-integral comparison it drives reverses wherever `Ltilde'` is
  negative. It is not needed for the exact constant-band identity, which is why that theorem
  does not carry it.
* `hwb : wb ≤ ws` — orientation of the band, needed only by the bounds theorems, because the
  monotonicity lemma for the interval integral is stated for `a ≤ b`. The exact constant-band
  identity is orientation-free.
* `hdirect : 0 < directDiffusion s sigmaObj` — the direct baseline's projected diffusion does not
  vanish, that is `s ≠ 0` and `sigma_obj ≠ 0`. Strict antitonicity holds on `Ioi 0` only, so
  without it there is no comparison to make.
* `hquadratic : 0 ≤ quadratic` — the sign of the quadratic channel, in the abstract comparison
  only. It is discharged, not assumed, in `arrheniusExponent_lift_lt_direct_of_quadForm`.
* `hcoupling : 0 ≤ coupling` — the sign of the cross channel `c1 sigma_Jac^2 |H|`, the projected
  form of the note's assumption (A6) of §11.3. The sign is NOT automatic. §2's sentence — that
  under the forward-KL chain `dg = H dtheta + r` the cross term is `2 e^T H V H e` and "the sign
  comes for free" — is retracted by §11.3 ("the sign is not automatic, and this is the honest
  core of the mechanism"), and `CouplingSign.lean` machine-checks the retraction: the sign is a
  theorem for the exact chain `r = 0` and under remainder control at ratio `ρ ≤ 2`, while in the
  window `2 < ρ ≤ 3` that (A4')'s own operator-norm clause permits there are realizable
  configurations whose coupling term is strictly negative, at arbitrarily small remainder.
  `hcoupling` is therefore a load-bearing hypothesis, and this file deliberately does not
  re-derive what is not derivable. Where the coupling term is negative enough that
  `coupling + quadratic < 0`, the two exponents order the other way round
  (`arrheniusExponent_strictAntiOn` is symmetric in its two points) — the mathematical form of
  §11.3's warning that the theory does not say a lift always helps.
* `hnondeg : 0 < coupling ∨ 0 < quadratic` — non-degeneracy. Without it the two diffusions may
  coincide and no strict ordering can hold; it is a genuine hypothesis, not a technicality. It
  is a disjunction so that either channel alone suffices; per §§10–11.0 of the note the
  magnitude channel is measured empirically inert at the operating point, so the disjunct that
  carries the mechanism in practice is the coupling channel's.
* `hkramers` — the Kramers relation itself, as discussed above. This is by far the largest
  unformalized input in the file and it is confined to the two theorems
  `escapeTime_ratio_of_kramers` and `escapeTime_ratio_of_kramers_of_prefactors`; every other
  statement is unconditional in it. The first of the two additionally fixes a *common* prefactor
  across the two methods; the second does not, and exhibits the quotient `C_dir / C_lift` that
  the common-prefactor form silently sets to one.
* `hsep : cmaxDir < cminLift` in `barrier_exponent_lift_lt_direct_of_bands` — separation of the
  two brackets. It is a *sufficient* condition and not a necessary one, and it is no longer this
  file's primary form of the band comparison. `barrier_exponent_lift_lt_direct_of_pointwise`
  assumes only the pointwise ordering `sigma_eff^{2,dir}(u) < sigma_eff^{2,lift}(u)` on the band,
  which separated brackets imply and which is what §2's per-iterate covariance excess actually
  delivers; the bracketed form is retained because a bracket is what a measurement reports, and
  it is now derived from the pointwise one. Neither form assumes an upper bracket on the lift's
  diffusion.
* `hdom : sigma_eff^{2,dir} < sigma_eff^{2,lift}` on the band, in the pointwise comparison — the
  band form of `hcoupling` and `hnondeg` together: at every reference iterate of the band the
  excess `coupling(u) + quadratic(u)` is strictly positive, which is what
  `barrier_exponent_lift_lt_direct_of_channels` makes explicit. Supplying it is
  `DiffusionOrdering.lean`'s business, and `CouplingSign.lean` is why it must be supplied rather
  than derived.
* The §2 **decomposition itself is not derived here.** `liftDiffusion` is a definition, so
  `directDiffusion_lt_liftDiffusion` is arithmetic about that definition and carries no claim
  that the lift's projected update covariance really is `s^2 sigma_obj^2 + coupling + quadratic`.
  That identification is fact (ii) of §2 and is machine-checked in `UpdateCovariance.lean`; this
  file consumes its shape and nothing else about it.
* FW-0's **ceteris-paribus convention** is structural rather than a named hypothesis, and is
  flagged here for that reason. Wherever the two methods are compared across one barrier
  (`barrier_exponent_lift_lt_direct_of_const_bands`, `_of_pointwise`, `_of_bands`, `_of_channels`)
  a single `Ltilde` and a single `dL`, hence a single barrier action `alpha`, serve both methods.
  §8's FW-0 states this as a modeling convention and not a theorem, mirroring the paper's
  `rem:fpt-scope`; it is carried in the shape of the statements themselves, and a reader who
  rejects the convention rejects the comparison rather than one of its hypotheses.
-/

open Set MeasureTheory Matrix

namespace IcnnLift

/-! ### The quasipotential gap across a barrier -/

section Quasipotential

variable {Ltilde dL Ueff sigma2 : ℝ → ℝ} {wb ws : ℝ}

/-- **Exact barrier rescaling on a constant-diffusion band, (L2') first half.**

If the effective diffusion `sigma_eff^2` is equal to the constant `c` throughout the band joining
the basin `w_b` to the saddle `w_s`, then the quasipotential gap is the barrier action divided by
`c`, *exactly*:
`U_eff(w_s) - U_eff(w_b) = (Ltilde(w_s) - Ltilde(w_b)) / c = alpha / c`.

The note states this for a positive `c`, and physically `c > 0`; positivity is not needed for the
identity itself and is therefore not assumed here. It is needed downstream, in the exponent
comparison, where it is stated where it is used. One caveat about the unrestricted `c`, since it
is the kind of thing that can be mistaken for extra strength: at `c = 0` the statement is not a
mathematical claim but an artefact of Lean's convention `x / 0 = 0`. The hypothesis `hUeff` then
reads `U_eff' = 0` on the band, and both sides of the conclusion are zero. Nothing downstream
uses that case; every comparison theorem below carries a strictly positive diffusion.

No approximation is made and none is available to be made: this is the fundamental theorem of
calculus applied to `U_eff' = Ltilde' / sigma_eff^2`, which is equation (*) of §3 in
differentiated form and is carried here as the hypothesis `hUeff`. The orientation of the band is
unconstrained — `w_b` and `w_s` may lie in either order — because the argument runs through the
oriented interval integral. -/
theorem quasipotential_gap_of_const_diffusion {c : ℝ}
    (hband : ∀ u ∈ uIcc wb ws, sigma2 u = c)
    (hLtilde : ∀ u ∈ uIcc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUeff : ∀ u ∈ uIcc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hintL : IntervalIntegrable dL volume wb ws) :
    Ueff ws - Ueff wb = (Ltilde ws - Ltilde wb) / c := by
  have hEq : EqOn (fun u => dL u / c) (fun u => dL u / sigma2 u) (uIcc wb ws) := by
    intro u hu
    simp only [hband u hu]
  have hintU : IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws :=
    (hintL.div_const c).congr (hEq.mono uIoc_subset_uIcc)
  have hIU : ∫ u in wb..ws, dL u / sigma2 u = Ueff ws - Ueff wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hUeff hintU
  have hIL : ∫ u in wb..ws, dL u = Ltilde ws - Ltilde wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hLtilde hintL
  have hcongr : ∫ u in wb..ws, dL u / sigma2 u = ∫ u in wb..ws, dL u / c :=
    intervalIntegral.integral_congr hEq.symm
  rw [← hIU, hcongr, intervalIntegral.integral_div, hIL]

/-- **Two-sided barrier rescaling on a band of non-constant diffusion, (L2') second half.**

The note's hypothesis is that `sigma_eff^2` is "approximately constant" on the shoulder band.
That phrase is replaced here by an explicit bracket: if `c_min ≤ sigma_eff^2 ≤ c_max` on the band
with `c_min > 0`, and the pullback potential is non-decreasing across the whole band
(`0 ≤ Ltilde'`), then

`alpha / c_max ≤ U_eff(w_s) - U_eff(w_b) ≤ alpha / c_min`,   `alpha = Ltilde(w_s) - Ltilde(w_b)`.

The approximation the note makes is thereby carried as the explicit width of the bracket rather
than discarded, and the statement collapses to `quasipotential_gap_of_const_diffusion` when
`c_min = c_max` (see `quasipotential_gap_bounds_eq_of_const`). The proof is monotonicity of the
interval integral applied to the pointwise inequalities
`dL / c_max ≤ dL / sigma_eff^2 ≤ dL / c_min`.

The hypothesis `hgrad` is stronger than the note's, and is stated rather than smuggled. The note
asks only for a positive barrier action `alpha > 0`; monotone climbing forbids in addition any
intervening critical point of `Ltilde` strictly between `w_b` and `w_s`. That is the standard
single-barrier picture, but it is an added restriction, and it cannot be dropped: without a sign
for `Ltilde'` the two pointwise inequalities above reverse where `Ltilde'` is negative and the
bracket is false. The exact identity of `quasipotential_gap_of_const_diffusion` needs no such
hypothesis. -/
theorem quasipotential_gap_bounds {cmin cmax : ℝ} (hwb : wb ≤ ws) (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUeff : ∀ u ∈ Icc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hintL : IntervalIntegrable dL volume wb ws)
    (hintU : IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws) :
    (Ltilde ws - Ltilde wb) / cmax ≤ Ueff ws - Ueff wb ∧
      Ueff ws - Ueff wb ≤ (Ltilde ws - Ltilde wb) / cmin := by
  have huIcc : uIcc wb ws = Icc wb ws := uIcc_of_le hwb
  have hIU : ∫ u in wb..ws, dL u / sigma2 u = Ueff ws - Ueff wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (by rw [huIcc]; exact hUeff) hintU
  have hIL : ∫ u in wb..ws, dL u = Ltilde ws - Ltilde wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (by rw [huIcc]; exact hLtilde) hintL
  constructor
  · have hle : ∀ u ∈ Icc wb ws, dL u / cmax ≤ dL u / sigma2 u := fun u hu =>
      div_le_div_of_nonneg_left (hgrad u hu)
        (lt_of_lt_of_le hcmin (hlower u hu)) (hupper u hu)
    have hmono := intervalIntegral.integral_mono_on hwb (hintL.div_const cmax) hintU hle
    rwa [intervalIntegral.integral_div, hIL, hIU] at hmono
  · have hge : ∀ u ∈ Icc wb ws, dL u / sigma2 u ≤ dL u / cmin := fun u hu =>
      div_le_div_of_nonneg_left (hgrad u hu) hcmin (hlower u hu)
    have hmono := intervalIntegral.integral_mono_on hwb hintU (hintL.div_const cmin) hge
    rwa [intervalIntegral.integral_div, hIL, hIU] at hmono

/-- The two-sided barrier rescaling with the two interval-integrability hypotheses discharged
from continuity of `Ltilde'` and of `sigma_eff^2` on the band, which is the regularity the paper
assumes anyway in (A1). -/
theorem quasipotential_gap_bounds_of_continuousOn {cmin cmax : ℝ} (hwb : wb ≤ ws)
    (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUeff : ∀ u ∈ Icc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hcontL : ContinuousOn dL (Icc wb ws))
    (hcontS : ContinuousOn sigma2 (Icc wb ws)) :
    (Ltilde ws - Ltilde wb) / cmax ≤ Ueff ws - Ueff wb ∧
      Ueff ws - Ueff wb ≤ (Ltilde ws - Ltilde wb) / cmin := by
  have hne : ∀ u ∈ Icc wb ws, sigma2 u ≠ 0 := fun u hu =>
    (lt_of_lt_of_le hcmin (hlower u hu)).ne'
  exact quasipotential_gap_bounds hwb hcmin hlower hupper hgrad hLtilde hUeff
    (hcontL.intervalIntegrable_of_Icc hwb)
    ((hcontL.div hcontS hne).intervalIntegrable_of_Icc hwb)

/-- When the bracket of `quasipotential_gap_bounds` is degenerate — the diffusion really is
constant on the band — the two bounds pinch, and the exact identity of
`quasipotential_gap_of_const_diffusion` is recovered from them. This is the consistency check
that the bracket is the honest generalization of the exact case rather than a different
statement. -/
theorem quasipotential_gap_bounds_eq_of_const {c : ℝ} (hwb : wb ≤ ws) (hc : 0 < c)
    (hband : ∀ u ∈ Icc wb ws, sigma2 u = c)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUeff : ∀ u ∈ Icc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hintL : IntervalIntegrable dL volume wb ws)
    (hintU : IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws) :
    Ueff ws - Ueff wb = (Ltilde ws - Ltilde wb) / c := by
  obtain ⟨h1, h2⟩ := quasipotential_gap_bounds (cmin := c) (cmax := c) hwb hc
    (fun u hu => (hband u hu).ge) (fun u hu => (hband u hu).le) hgrad hLtilde hUeff hintL hintU
  linarith

/-- **The numerical core of the non-constant witness: `1/2 < log 2 < 1`, strictly.** The width
`alpha / c_min - alpha / c_max = 1/2` is the price the note's "approximately constant" phrase
actually costs on the witness band of `quasipotential_gap_bounds_witness` below, and neither
endpoint of the bracket is attained there. Consequently the bracket of
`quasipotential_gap_bounds` cannot be strengthened to an equality without a further hypothesis
on `sigma_eff^2`, and a comparison between two methods drawn from brackets alone requires those
brackets to be separated — which is the hypothesis `hsep` of
`barrier_exponent_lift_lt_direct_of_bands`. -/
theorem quasipotential_gap_bounds_witness_strict :
    (1 : ℝ) / 2 < Real.log 2 ∧ Real.log 2 < 1 := by
  constructor
  · have h := Real.log_two_gt_d9
    norm_num at h ⊢
    linarith
  · have h := Real.log_two_lt_d9
    norm_num at h ⊢
    linarith

/-- **The band hypotheses are satisfiable away from the constant case — certified in the
statement, not merely exhibited in a proof.**

A referee is entitled to ask whether `quasipotential_gap_bounds` says anything, or whether its
hypotheses can only be met when `sigma_eff^2` is constant, in which case the bracket would be a
restatement of `quasipotential_gap_of_const_diffusion`. This theorem states the answer as an
existential the kernel checks: there is an instance meeting every hypothesis of
`quasipotential_gap_bounds_of_continuousOn` whose diffusion is genuinely non-constant on the
band (`sigma2 wb ≠ sigma2 ws`) and whose bracket is strict on both sides, hence not an identity
in disguise. The instance is the band `[0, 1]`, the pullback potential `Ltilde(u) = u` (so
`Ltilde' = 1 ≥ 0` and the barrier action is `alpha = 1`), the effective diffusion
`sigma_eff^2(u) = u + 1` with `sigma_eff^2(0) = 1 ≠ 2 = sigma_eff^2(1)`, trapped between
`c_min = 1` and `c_max = 2`, and the quasipotential `U_eff(u) = log (u + 1)`, which satisfies
`U_eff' = Ltilde' / sigma_eff^2` on the band; the bracket for the gap
`U_eff(1) - U_eff(0) = log 2` is the strict numerical statement
`alpha / c_max = 1/2 < log 2 < 1 = alpha / c_min` of
`quasipotential_gap_bounds_witness_strict`. -/
theorem quasipotential_gap_bounds_witness :
    ∃ (Ltilde dL Ueff sigma2 : ℝ → ℝ) (wb ws cmin cmax : ℝ),
      wb ≤ ws ∧ 0 < cmin ∧
      (∀ u ∈ Icc wb ws, cmin ≤ sigma2 u) ∧
      (∀ u ∈ Icc wb ws, sigma2 u ≤ cmax) ∧
      (∀ u ∈ Icc wb ws, 0 ≤ dL u) ∧
      (∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u) ∧
      (∀ u ∈ Icc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u) ∧
      ContinuousOn dL (Icc wb ws) ∧ ContinuousOn sigma2 (Icc wb ws) ∧
      sigma2 wb ≠ sigma2 ws ∧
      (Ltilde ws - Ltilde wb) / cmax < Ueff ws - Ueff wb ∧
      Ueff ws - Ueff wb < (Ltilde ws - Ltilde wb) / cmin := by
  have hne : ∀ u ∈ Icc (0 : ℝ) 1, u + 1 ≠ 0 := by
    intro u hu
    have := hu.1
    positivity
  refine ⟨fun u => u, fun _ => 1, fun u => Real.log (u + 1), fun u => u + 1,
    0, 1, 1, 2, by norm_num, one_pos, ?_, ?_, fun _ _ => zero_le_one,
    fun u _ => hasDerivAt_id' u, ?_, continuousOn_const,
    (continuous_id.add continuous_const).continuousOn, by norm_num, ?_, ?_⟩
  · intro u hu
    have := hu.1
    linarith
  · intro u hu
    have := hu.2
    linarith
  · intro u hu
    simpa using ((hasDerivAt_id u).add_const (1 : ℝ)).log (hne u hu)
  · have h := quasipotential_gap_bounds_witness_strict.1
    norm_num
    linarith
  · have h := quasipotential_gap_bounds_witness_strict.2
    norm_num
    linarith

end Quasipotential

/-! ### The Arrhenius exponent and its strict monotonicity in the diffusion -/

/-- The **Arrhenius (Kramers) exponent** of a barrier of action `alpha` at learning rate `eta`
under effective diffusion `sigma2`: `2 alpha / (eta sigma2)`. By (L2') this is
`(2/eta)` times the quasipotential action `alpha_eff = alpha / sigma2`, so it is the quantity in
which the two methods are compared in the corrected Corollary 1. It is an exponent; no prefactor
is attached to it anywhere in this file. -/
noncomputable def arrheniusExponent (alpha eta sigma2 : ℝ) : ℝ := 2 * alpha / (eta * sigma2)

/-- The exponent-only escape-time prediction, `exp` of the Arrhenius exponent. This is *not* a
mean first-passage time: it is the exponential factor that a Kramers law would multiply by a
prefactor this file never addresses. -/
noncomputable def arrheniusFactor (alpha eta sigma2 : ℝ) : ℝ :=
  Real.exp (arrheniusExponent alpha eta sigma2)

/-- The direct baseline's projected effective diffusion, `s^2 sigma_obj^2`: the objective gradient
noise attenuated once by the positivity-map slope `s = psi'`. The baseline has `dtheta = 0`, so the
coupling and quadratic channels of §2 vanish identically for it. -/
noncomputable def directDiffusion (s sigmaObj : ℝ) : ℝ := s ^ 2 * sigmaObj ^ 2

/-- The lift's projected effective diffusion in the corrected form of §2,
`s^2 sigma_obj^2 + coupling + quadratic`, where `coupling` is the cross channel
`c1 sigma_Jac^2 |H_bb|` carried by the slack-channel cross-covariance and `quadratic` is
`s^2 (HVH)_bb`. Every term carries the same units, which was the dimensional defect of the
shipped `sigma_eff^2 = s^2 sigma_obj^2 + sigma_Jac^2`. The two accountings of §11.2 of the note
disagree about the power of `s` the coupling channel carries (attenuated once, in the
latent-iterate accounting, against twice in the parameter-space one); `coupling` is an
abstract real here precisely so that the comparison theorems below are correct under either
accounting.

This is a definition, not a derivation: that the lift's projected update covariance really takes
this form is fact (ii) of §2, machine-checked in `UpdateCovariance.lean`, and it is nowhere
re-proved here. In particular `directDiffusion_lt_liftDiffusion` below is arithmetic about this
definition and asserts nothing about whether the definition is the right one. -/
noncomputable def liftDiffusion (s sigmaObj coupling quadratic : ℝ) : ℝ :=
  s ^ 2 * sigmaObj ^ 2 + coupling + quadratic

/-- The Arrhenius exponent is **strictly decreasing** in the effective diffusion, on the positive
reals, whenever the barrier action and the learning rate are positive. This is the whole of the
comparison: a strictly larger effective diffusion is a strictly smaller exponent. -/
theorem arrheniusExponent_strictAntiOn {alpha eta : ℝ} (halpha : 0 < alpha) (heta : 0 < eta) :
    StrictAntiOn (arrheniusExponent alpha eta) (Ioi 0) := by
  intro x hx y hy hxy
  have hx0 : (0 : ℝ) < x := hx
  have hy0 : (0 : ℝ) < y := hy
  have h2a : (0 : ℝ) < 2 * alpha := by linarith
  have hex : (0 : ℝ) < eta * x := mul_pos heta hx0
  have hlt : eta * x < eta * y := by
    have := mul_lt_mul_of_pos_left hxy heta
    simpa using this
  exact div_lt_div_of_pos_left h2a hex hlt

/-- Hence the exponent-only escape-time prediction is strictly decreasing in the effective
diffusion: a strictly larger diffusion gives a strictly smaller predicted escape time. -/
theorem arrheniusFactor_strictAntiOn {alpha eta : ℝ} (halpha : 0 < alpha) (heta : 0 < eta) :
    StrictAntiOn (arrheniusFactor alpha eta) (Ioi 0) := fun _ hx _ hy hxy =>
  Real.exp_lt_exp.mpr (arrheniusExponent_strictAntiOn halpha heta hx hy hxy)

/-! ### Instantiation at the two effective diffusions of the corrected decomposition -/

section Comparison

variable {alpha eta s sigmaObj coupling quadratic : ℝ}

/-- The lift's effective diffusion **strictly exceeds** the direct baseline's, given that the
coupling channel is non-negative (the projected form of the note's assumption (A6), §11.3 — a
genuine assumption, not a derived fact: `CouplingSign.lean` shows the sign is not automatic
under (A4')'s remainder control, so it is assumed here and never derived), that the quadratic
channel is non-negative, and that at least one of the two channels is non-degenerate. Both
channels are abstract reals here.
The quadratic channel's non-negativity is a hypothesis at this level of generality and a theorem
once the channel is instantiated as the quadratic form `s^2 eᵀ H V H e` of the corrected §2, in
`arrheniusExponent_lift_lt_direct_of_quadForm`, where it is discharged from
`TraceLemmas.quadForm_conj_nonneg`. -/
theorem directDiffusion_lt_liftDiffusion (hcoupling : 0 ≤ coupling) (hquadratic : 0 ≤ quadratic)
    (hnondeg : 0 < coupling ∨ 0 < quadratic) :
    directDiffusion s sigmaObj < liftDiffusion s sigmaObj coupling quadratic := by
  rcases hnondeg with h | h <;> · simp only [directDiffusion, liftDiffusion]; linarith

/-- **The corrected Corollary 1 exponent comparison.** Under a positive barrier action, a positive
learning rate, a positive direct baseline diffusion (exactly `s ≠ 0` and `sigma_obj ≠ 0`; the
physical case on the shoulder is `s > 0` and `sigma_obj > 0`, which suffices but is not required),
a non-negative coupling channel, a non-negative quadratic channel and non-degeneracy of one of
them, the lift's Arrhenius exponent is strictly smaller than the direct baseline's:

`2 alpha / (eta [s^2 sigma_obj^2 + coupling + quadratic]) < 2 alpha / (eta s^2 sigma_obj^2)`.

This is exactly the comparison displayed in §3 of the note, with the dimensionally consistent
denominator derived in §2 in place of the shipped `s^2 sigma_obj^2 + sigma_Jac^2`. -/
theorem arrheniusExponent_lift_lt_direct (halpha : 0 < alpha) (heta : 0 < eta)
    (hdirect : 0 < directDiffusion s sigmaObj) (hcoupling : 0 ≤ coupling)
    (hquadratic : 0 ≤ quadratic) (hnondeg : 0 < coupling ∨ 0 < quadratic) :
    arrheniusExponent alpha eta (liftDiffusion s sigmaObj coupling quadratic)
      < arrheniusExponent alpha eta (directDiffusion s sigmaObj) := by
  have hlt := directDiffusion_lt_liftDiffusion (s := s) (sigmaObj := sigmaObj)
    hcoupling hquadratic hnondeg
  exact arrheniusExponent_strictAntiOn halpha heta hdirect (hdirect.trans hlt) hlt

/-- The same comparison at the level of the exponent-only escape-time prediction: the lift's
predicted escape time is strictly smaller. The prefactor is not addressed. -/
theorem arrheniusFactor_lift_lt_direct (halpha : 0 < alpha) (heta : 0 < eta)
    (hdirect : 0 < directDiffusion s sigmaObj) (hcoupling : 0 ≤ coupling)
    (hquadratic : 0 ≤ quadratic) (hnondeg : 0 < coupling ∨ 0 < quadratic) :
    arrheniusFactor alpha eta (liftDiffusion s sigmaObj coupling quadratic)
      < arrheniusFactor alpha eta (directDiffusion s sigmaObj) :=
  Real.exp_lt_exp.mpr
    (arrheniusExponent_lift_lt_direct halpha heta hdirect hcoupling hquadratic hnondeg)

/-- The exponent comparison with the **quadratic channel's sign proved rather than assumed.**

The quadratic channel of §2 is `s^2 e^T H V H e` with `V = Cov[dtheta]` positive semidefinite and
`H` the symmetric mean Hessian. Its non-negativity is `IcnnLift.quadForm_conj_nonneg` of
`TraceLemmas.lean` and is discharged here, so the only sign hypothesis that remains is
`hcoupling`, the coupling channel's — the note's assumption (A6) of §11.3, which
`CouplingSign.lean` shows cannot be traded for a derivation. -/
theorem arrheniusExponent_lift_lt_direct_of_quadForm {n : Type*} [Fintype n]
    {V H : Matrix n n ℝ} (hV : V.PosSemidef) (hH : H.IsHermitian) (e : n → ℝ)
    (halpha : 0 < alpha) (heta : 0 < eta) (hdirect : 0 < directDiffusion s sigmaObj)
    (hcoupling : 0 ≤ coupling)
    (hnondeg : 0 < coupling ∨ 0 < s ^ 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e))) :
    arrheniusExponent alpha eta
        (liftDiffusion s sigmaObj coupling (s ^ 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e))))
      < arrheniusExponent alpha eta (directDiffusion s sigmaObj) :=
  arrheniusExponent_lift_lt_direct halpha heta hdirect hcoupling
    (mul_nonneg (sq_nonneg s) (quadForm_conj_nonneg hV hH e)) hnondeg

/-- **The quadratic channel can be strictly positive.**

`arrheniusExponent_lift_lt_direct_of_quadForm` concludes a strict inequality from the
non-degeneracy disjunction `0 < coupling ∨ 0 < s^2 eᵀ (H V H) e`, and its second disjunct is a
quadratic form that `TraceLemmas.quadForm_conj_nonneg` shows is never negative. That it can be
strictly positive — so that the theorem is not carried entirely by its first disjunct, and so
that the lift's diffusion excess can come from the quadratic channel alone — is exhibited here by
a one-dimensional instance: `V = H = 1`, `e = 1`, `s = sigma_obj = 1`, for which the direct
baseline diffusion is `1 > 0` and the quadratic channel is `1 > 0`. This is a satisfiability
statement about the mathematics, not an empirical claim: §11.0 of the note measures the
magnitude channel inert at the operating point across a hundredfold variance bracket, so in the
measured regime the excess is carried by the coupling channel. -/
theorem quadForm_channel_nonvacuous :
    ∃ (V H : Matrix (Fin 1) (Fin 1) ℝ) (e : Fin 1 → ℝ) (s sigmaObj : ℝ),
      V.PosSemidef ∧ H.IsHermitian ∧ 0 < directDiffusion s sigmaObj ∧
        0 < s ^ 2 * (e ⬝ᵥ ((H * V * H) *ᵥ e)) := by
  refine ⟨1, 1, fun _ => 1, 1, 1, ?_, Matrix.isHermitian_one, ?_, ?_⟩
  · simpa using Matrix.posSemidef_conjTranspose_mul_self (1 : Matrix (Fin 1) (Fin 1) ℝ)
  · norm_num [directDiffusion]
  · norm_num [Matrix.one_mulVec, dotProduct]

end Comparison

/-! ### The escape-time ratio -/

/-- The predicted **ratio of escape times**, direct over lift, in the form the paper's figures are
compared against: `exp (2 alpha (1/sigma2_dir - 1/sigma2_lift) / eta)`. -/
noncomputable def escapeTimeRatio (alpha eta sigma2dir sigma2lift : ℝ) : ℝ :=
  Real.exp (2 * alpha * (1 / sigma2dir - 1 / sigma2lift) / eta)

section Ratio

variable {alpha eta sd sl : ℝ}

/-- The ratio identity: `escapeTimeRatio` is the quotient of the two exponent-only escape-time
predictions. -/
theorem escapeTimeRatio_eq_div (heta : eta ≠ 0) (hd : sd ≠ 0) (hl : sl ≠ 0) :
    escapeTimeRatio alpha eta sd sl
      = arrheniusFactor alpha eta sd / arrheniusFactor alpha eta sl := by
  have hexp : arrheniusExponent alpha eta sd - arrheniusExponent alpha eta sl
      = 2 * alpha * (1 / sd - 1 / sl) / eta := by
    simp only [arrheniusExponent]
    field_simp
  rw [arrheniusFactor, arrheniusFactor, ← Real.exp_sub, hexp, escapeTimeRatio]

/-- The ratio **exceeds one** whenever the lift's effective diffusion strictly exceeds the direct
baseline's: the lift is predicted to escape strictly sooner. -/
theorem one_lt_escapeTimeRatio (halpha : 0 < alpha) (heta : 0 < eta) (hd : 0 < sd)
    (hdl : sd < sl) : 1 < escapeTimeRatio alpha eta sd sl := by
  rw [escapeTimeRatio, Real.one_lt_exp_iff]
  have hinv : 1 / sl < 1 / sd := one_div_lt_one_div_of_lt hd hdl
  have hnum : 0 < 2 * alpha * (1 / sd - 1 / sl) := by nlinarith
  exact div_pos hnum heta

/-- The ratio **grows strictly as the excess diffusion grows**: holding the direct baseline fixed,
a strictly larger lift diffusion gives a strictly larger predicted separation. This is the
quantitative shape the paper's escape-time figures are read against. -/
theorem escapeTimeRatio_strictMono_lift {sl₁ sl₂ : ℝ} (halpha : 0 < alpha) (heta : 0 < eta)
    (hl₁ : 0 < sl₁) (h₁₂ : sl₁ < sl₂) :
    escapeTimeRatio alpha eta sd sl₁ < escapeTimeRatio alpha eta sd sl₂ := by
  rw [escapeTimeRatio, escapeTimeRatio, Real.exp_lt_exp]
  have hinv : 1 / sl₂ < 1 / sl₁ := one_div_lt_one_div_of_lt hl₁ h₁₂
  have hnum : 2 * alpha * (1 / sd - 1 / sl₁) < 2 * alpha * (1 / sd - 1 / sl₂) := by nlinarith
  exact div_lt_div_of_pos_right hnum heta

/-- The ratio of **mean first-passage times**, under the Kramers relation as an explicit
hypothesis.

`hkramers` states that the predicted escape time at effective diffusion `x > 0` is
`C * exp (arrheniusExponent alpha eta x)` with a common prefactor `C`. This is the single largest
unformalized input in the file: mathlib v4.31.0 carries a Brownian motion and the hitting time of
a process, but no stochastic integral, no solution theory for stochastic differential equations,
no Fokker-Planck equation and no mean first-passage-time formula, so the Arrhenius law cannot be
derived here. Discharging
`hkramers` requires the one-dimensional mean-first-passage-time formula for a diffusion with
state-dependent noise, which is `KramersExitTime.lean`'s business. Note also that the hypothesis
assumes a *common* prefactor across the two methods, which is the paper's own "exponents only, no
prefactors" convention; the Eyring-Kramers prefactor differs between the two diffusions and is
nowhere estimated here. -/
theorem escapeTime_ratio_of_kramers {tau : ℝ → ℝ} {C : ℝ} (hC : C ≠ 0) (heta : eta ≠ 0)
    (hd : 0 < sd) (hl : 0 < sl)
    (hkramers : ∀ x, 0 < x → tau x = C * Real.exp (arrheniusExponent alpha eta x)) :
    tau sd / tau sl = escapeTimeRatio alpha eta sd sl := by
  rw [hkramers sd hd, hkramers sl hl,
    escapeTimeRatio_eq_div (alpha := alpha) heta hd.ne' hl.ne']
  rw [arrheniusFactor, arrheniusFactor]
  rw [mul_div_mul_left _ _ hC]

/-- **The ratio of mean first-passage times with the two Kramers prefactors kept apart.**

`escapeTime_ratio_of_kramers` assumes one prefactor `C` for both methods, and that assumption is
not innocent: the Eyring-Kramers prefactor depends on the diffusion, so the direct baseline's and
the lift's differ in general, and the stationary density (*) carries a further `sigma_eff^{-2}`
that differs between them as well. This theorem therefore keeps the two prefactors distinct and
states what is actually true of the two predicted times,

`tau_dir(sigma2_dir) / tau_lift(sigma2_lift) = (C_dir / C_lift) * escapeTimeRatio ...`,

which exhibits the exact factor the exponent-only convention discards. `escapeTime_ratio_of_kramers`
is its `C_dir = C_lift` case. Nothing here estimates `C_dir / C_lift`; bounding it is
`KramersExitTime.lean`'s business, and until it is bounded the ratio statement is a statement
about exponents and not about times. -/
theorem escapeTime_ratio_of_kramers_of_prefactors {tauDir tauLift : ℝ → ℝ} {Cdir Clift : ℝ}
    (hClift : Clift ≠ 0) (heta : eta ≠ 0) (hd : 0 < sd) (hl : 0 < sl)
    (hkd : tauDir sd = Cdir * Real.exp (arrheniusExponent alpha eta sd))
    (hkl : tauLift sl = Clift * Real.exp (arrheniusExponent alpha eta sl)) :
    tauDir sd / tauLift sl = Cdir / Clift * escapeTimeRatio alpha eta sd sl := by
  rw [hkd, hkl, escapeTimeRatio_eq_div (alpha := alpha) heta hd.ne' hl.ne', arrheniusFactor,
    arrheniusFactor]
  field_simp

/-- **The escape-time ratio at the two effective diffusions of the corrected decomposition.**

Assembling `directDiffusion_lt_liftDiffusion` with `one_lt_escapeTimeRatio`: under a positive
barrier action, a positive learning rate, a positive direct baseline diffusion, a non-negative
coupling channel (assumption (A6) of §11.3, an assumption and not a theorem — see
`CouplingSign.lean`), a non-negative quadratic channel and non-degeneracy
of one of the two, the predicted escape-time ratio direct-over-lift exceeds one. This is the
statement the paper's escape-time figures are compared against; it is an exponent ratio, with no
prefactor. -/
theorem one_lt_escapeTimeRatio_lift {s sigmaObj coupling quadratic : ℝ} (halpha : 0 < alpha)
    (heta : 0 < eta) (hdirect : 0 < directDiffusion s sigmaObj) (hcoupling : 0 ≤ coupling)
    (hquadratic : 0 ≤ quadratic) (hnondeg : 0 < coupling ∨ 0 < quadratic) :
    1 < escapeTimeRatio alpha eta (directDiffusion s sigmaObj)
          (liftDiffusion s sigmaObj coupling quadratic) :=
  one_lt_escapeTimeRatio halpha heta hdirect
    (directDiffusion_lt_liftDiffusion hcoupling hquadratic hnondeg)

end Ratio

/-! ### Bridging the two halves: the barrier as it enters the exponent -/

section Bridge

variable {Ltilde dL Ueff sigma2 : ℝ → ℝ} {wb ws : ℝ}

/-- **(L2') assembled, constant-band case.** The quasipotential barrier as it enters the
stationary density `pi ∝ sigma_eff^{-2} exp(-(2/eta) U_eff)` is exactly the Arrhenius exponent of
the barrier action `alpha = Ltilde(w_s) - Ltilde(w_b)` at the band's diffusion `c`:

`(2 / eta) * (U_eff(w_s) - U_eff(w_b)) = arrheniusExponent alpha eta c`.

This is the sentence "(*) gives effective action `alpha_eff = alpha / sigma_eff^2`, so the Kramers
exponent compares as ..." of §3, made exact. -/
theorem barrier_exponent_of_const_diffusion {c : ℝ}
    (hband : ∀ u ∈ uIcc wb ws, sigma2 u = c)
    (hLtilde : ∀ u ∈ uIcc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUeff : ∀ u ∈ uIcc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hintL : IntervalIntegrable dL volume wb ws) (eta : ℝ) :
    (2 / eta) * (Ueff ws - Ueff wb) = arrheniusExponent (Ltilde ws - Ltilde wb) eta c := by
  rw [quasipotential_gap_of_const_diffusion hband hLtilde hUeff hintL, arrheniusExponent,
    div_mul_div_comm]

/-- **(L2') assembled, band case with the approximation carried explicitly.** On a band where the
effective diffusion is only known to lie between `c_min > 0` and `c_max`, the barrier's
contribution to the exponent is bracketed by the two Arrhenius exponents:

`arrheniusExponent alpha eta c_max ≤ (2/eta) (U_eff(w_s) - U_eff(w_b))
    ≤ arrheniusExponent alpha eta c_min`.

The comparison of `arrheniusExponent_lift_lt_direct` transfers to the true barrier exponent as
soon as the lift's bracket lies strictly below the direct baseline's; that transfer is not left
as a remark but is proved, as `barrier_exponent_lift_lt_direct_of_bands`. Two overlapping
brackets, on their own, license no conclusion. Brackets are not the weakest available hypothesis,
however, and this file does not present them as such: the pointwise ordering of the two
diffusions suffices and is proved first, as `barrier_exponent_lift_lt_direct_of_pointwise`. -/
theorem barrier_exponent_bounds {eta cmin cmax : ℝ} (hwb : wb ≤ ws) (heta : 0 < eta)
    (hcmin : 0 < cmin)
    (hlower : ∀ u ∈ Icc wb ws, cmin ≤ sigma2 u)
    (hupper : ∀ u ∈ Icc wb ws, sigma2 u ≤ cmax)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUeff : ∀ u ∈ Icc wb ws, HasDerivAt Ueff (dL u / sigma2 u) u)
    (hintL : IntervalIntegrable dL volume wb ws)
    (hintU : IntervalIntegrable (fun u => dL u / sigma2 u) volume wb ws) :
    arrheniusExponent (Ltilde ws - Ltilde wb) eta cmax ≤ (2 / eta) * (Ueff ws - Ueff wb) ∧
      (2 / eta) * (Ueff ws - Ueff wb)
        ≤ arrheniusExponent (Ltilde ws - Ltilde wb) eta cmin := by
  obtain ⟨h1, h2⟩ := quasipotential_gap_bounds hwb hcmin hlower hupper hgrad hLtilde hUeff
    hintL hintU
  have hfac : (0 : ℝ) ≤ 2 / eta := by positivity
  constructor
  · have := mul_le_mul_of_nonneg_left h1 hfac
    rwa [arrheniusExponent, ← div_mul_div_comm]
  · have := mul_le_mul_of_nonneg_left h2 hfac
    rwa [arrheniusExponent, ← div_mul_div_comm]

/-- **(L2') and the corrected Corollary 1 joined, constant-band case.**

The two halves of the file meet here. Fix one barrier of the pullback landscape, of positive
action `alpha = Ltilde(w_s) - Ltilde(w_b)`, and run the two methods across it. The direct
baseline's effective diffusion is constant and equal to `s^2 sigma_obj^2` on the band; the lift's
is constant and equal to `s^2 sigma_obj^2 + coupling + quadratic`. Each method has its own
quasipotential — `U_eff^dir` and `U_eff^lift`, both solving `U' = Ltilde' / sigma_eff^2` for its
own `sigma_eff^2` — and the conclusion is that the lift's true quasipotential barrier exponent is
strictly below the direct baseline's:

`(2/eta) (U_eff^lift(w_s) - U_eff^lift(w_b)) < (2/eta) (U_eff^dir(w_s) - U_eff^dir(w_b))`.

The barrier `alpha` — one `Ltilde`, one `dL` for both methods — is common to the two methods by
FW-0's ceteris-paribus convention, which the note states as a modeling convention and not a
theorem: the comparison is run at a common reference iterate with the barrier of the reference
potential held fixed, exactly as the paper's `rem:fpt-scope` holds `alpha` fixed. Within that
convention §2's fact (i) gives the common drift, so only the diffusion differs. This is the
exponent statement the corrected Corollary 1 makes; it is an inequality between exponents, and
the prefactors are not addressed. -/
theorem barrier_exponent_lift_lt_direct_of_const_bands
    {UeffDir UeffLift sigma2Dir sigma2Lift : ℝ → ℝ}
    {eta s sigmaObj coupling quadratic : ℝ}
    (halpha : 0 < Ltilde ws - Ltilde wb) (heta : 0 < eta)
    (hdirect : 0 < directDiffusion s sigmaObj)
    (hcoupling : 0 ≤ coupling) (hquadratic : 0 ≤ quadratic)
    (hnondeg : 0 < coupling ∨ 0 < quadratic)
    (hbandDir : ∀ u ∈ uIcc wb ws, sigma2Dir u = directDiffusion s sigmaObj)
    (hbandLift : ∀ u ∈ uIcc wb ws, sigma2Lift u = liftDiffusion s sigmaObj coupling quadratic)
    (hLtilde : ∀ u ∈ uIcc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUdir : ∀ u ∈ uIcc wb ws, HasDerivAt UeffDir (dL u / sigma2Dir u) u)
    (hUlift : ∀ u ∈ uIcc wb ws, HasDerivAt UeffLift (dL u / sigma2Lift u) u)
    (hintL : IntervalIntegrable dL volume wb ws) :
    (2 / eta) * (UeffLift ws - UeffLift wb) < (2 / eta) * (UeffDir ws - UeffDir wb) := by
  rw [barrier_exponent_of_const_diffusion hbandDir hLtilde hUdir hintL eta,
    barrier_exponent_of_const_diffusion hbandLift hLtilde hUlift hintL eta]
  exact arrheniusExponent_lift_lt_direct halpha heta hdirect hcoupling hquadratic hnondeg

/-- **(L2') and the corrected Corollary 1 joined, non-constant bands — the pointwise form.**

Neither method's effective diffusion is exactly constant on a real shoulder band, and the note
covers this with the words "approximately constant". The honest replacement is not a pair of
brackets: it is the pointwise ordering that the diffusion decomposition of §2 actually delivers.
At each reference iterate `u` of the band, §2 computes one update covariance per method and their
difference, so what §2 supports is

`0 < sigma_eff^{2,dir}(u) < sigma_eff^{2,lift}(u)`  for every `u` in `[w_b, w_s]`,

and nothing about the *ranges* of the two functions over the band. Under that hypothesis, a
barrier of strictly positive action `alpha = Ltilde(w_s) - Ltilde(w_b)` climbed monotonically
across the band, and the two quasipotential equations `U' = Ltilde' / sigma_eff^2` — each method
with its own diffusion — the lift's true quasipotential barrier exponent is strictly below the
direct baseline's:

`(2/eta) (U_eff^lift(w_s) - U_eff^lift(w_b)) < (2/eta) (U_eff^dir(w_s) - U_eff^dir(w_b))`.

This is strictly weaker in its hypotheses than the bracketed
`barrier_exponent_lift_lt_direct_of_bands` below, which is now its corollary: separated brackets
give the pointwise ordering through `sigma_eff^{2,dir} ≤ c_max^dir < c_min^lift ≤
sigma_eff^{2,lift}`, while the converse fails, since two diffusions can be ordered at every point
of a band whose ranges overlap completely. No bracket of any kind is assumed here, on either
diffusion, and only the direct baseline's positivity is required — the lift's follows from the
ordering.

Strictness is where the barrier's positivity is spent, and it is spent honestly rather than
assumed. From `alpha > 0` and `Ltilde' ≥ 0` on the band there is a point of the band at which
`Ltilde' > 0`, at which the two integrands `Ltilde'/sigma_eff^2` are strictly ordered; continuity
of `Ltilde'` and of the two diffusions then upgrades the pointwise comparison of the interval
integrals to a strict one
(`intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt`). A degenerate band
`w_b = w_s` is excluded by `alpha > 0` rather than by hypothesis.

The hypotheses are jointly satisfiable — see
`barrier_exponent_lift_lt_direct_of_bands_witness`, whose instance exhibits the pointwise
ordering explicitly alongside a nowhere-constant lift diffusion. The prefactors are not addressed
here, as nowhere in this file. -/
theorem barrier_exponent_lift_lt_direct_of_pointwise
    {UeffDir UeffLift sigma2Dir sigma2Lift : ℝ → ℝ} {eta : ℝ}
    (hwb : wb ≤ ws) (halpha : 0 < Ltilde ws - Ltilde wb) (heta : 0 < eta)
    (hposDir : ∀ u ∈ Icc wb ws, 0 < sigma2Dir u)
    (hdom : ∀ u ∈ Icc wb ws, sigma2Dir u < sigma2Lift u)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUdir : ∀ u ∈ Icc wb ws, HasDerivAt UeffDir (dL u / sigma2Dir u) u)
    (hUlift : ∀ u ∈ Icc wb ws, HasDerivAt UeffLift (dL u / sigma2Lift u) u)
    (hcontL : ContinuousOn dL (Icc wb ws))
    (hcontDir : ContinuousOn sigma2Dir (Icc wb ws))
    (hcontLift : ContinuousOn sigma2Lift (Icc wb ws)) :
    (2 / eta) * (UeffLift ws - UeffLift wb) < (2 / eta) * (UeffDir ws - UeffDir wb) := by
  have huIcc : uIcc wb ws = Icc wb ws := uIcc_of_le hwb
  have hposLift : ∀ u ∈ Icc wb ws, 0 < sigma2Lift u := fun u hu =>
    (hposDir u hu).trans (hdom u hu)
  have hcontQdir : ContinuousOn (fun u => dL u / sigma2Dir u) (Icc wb ws) :=
    hcontL.div hcontDir fun u hu => (hposDir u hu).ne'
  have hcontQlift : ContinuousOn (fun u => dL u / sigma2Lift u) (Icc wb ws) :=
    hcontL.div hcontLift fun u hu => (hposLift u hu).ne'
  have hIL : ∫ u in wb..ws, dL u = Ltilde ws - Ltilde wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (by rw [huIcc]; exact hLtilde)
      (hcontL.intervalIntegrable_of_Icc hwb)
  have hIdir : ∫ u in wb..ws, dL u / sigma2Dir u = UeffDir ws - UeffDir wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (by rw [huIcc]; exact hUdir)
      (hcontQdir.intervalIntegrable_of_Icc hwb)
  have hIlift : ∫ u in wb..ws, dL u / sigma2Lift u = UeffLift ws - UeffLift wb :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (by rw [huIcc]; exact hUlift)
      (hcontQlift.intervalIntegrable_of_Icc hwb)
  -- A barrier of positive action cannot be crossed on a degenerate band.
  have hlt : wb < ws := by
    rcases hwb.lt_or_eq with h | h
    · exact h
    · subst h
      simp at halpha
  -- Nor can it be crossed without the potential strictly climbing somewhere.
  have hex : ∃ c ∈ Icc wb ws, 0 < dL c := by
    by_contra hcon
    have hle : ∀ u ∈ Icc wb ws, dL u ≤ 0 := fun u hu => not_lt.mp fun h => hcon ⟨u, hu, h⟩
    have hzero : EqOn dL (fun _ => (0 : ℝ)) (uIcc wb ws) := by
      intro u hu
      rw [huIcc] at hu
      show dL u = (0 : ℝ)
      exact le_antisymm (hle u hu) (hgrad u hu)
    have hz : ∫ u in wb..ws, dL u = 0 := by
      rw [intervalIntegral.integral_congr hzero]
      simp
    rw [hz] at hIL
    linarith
  have hstrict : ∫ u in wb..ws, dL u / sigma2Lift u < ∫ u in wb..ws, dL u / sigma2Dir u := by
    refine intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt hlt
      hcontQlift hcontQdir (fun x hx => ?_) ?_
    · have hx' : x ∈ Icc wb ws := Ioc_subset_Icc_self hx
      exact div_le_div_of_nonneg_left (hgrad x hx') (hposDir x hx') (hdom x hx').le
    · obtain ⟨c, hc, hdLc⟩ := hex
      exact ⟨c, hc, div_lt_div_of_pos_left hdLc (hposDir c hc) (hdom c hc)⟩
  have hfac : (0 : ℝ) < 2 / eta := div_pos (by norm_num) heta
  rw [← hIdir, ← hIlift]
  exact mul_lt_mul_of_pos_left hstrict hfac

/-- **(L2') and the corrected Corollary 1 joined, non-constant bands — the bracketed corollary.**

The form in which a measurement reports the two diffusions is a pair of brackets, and this is the
comparison in those terms: if

`c_min^dir ≤ sigma_eff^{2,dir} ≤ c_max^dir < c_min^lift ≤ sigma_eff^{2,lift}`

on the band, with `c_min^dir > 0`, then the lift's quasipotential barrier exponent is strictly
below the direct baseline's. Two *overlapping* brackets license no conclusion, and the theorem
does not pretend otherwise.

Separation is nevertheless **sufficient and not necessary**, and it should not be advertised as
what the band argument of §3 needs. What that argument needs is the pointwise ordering of
`barrier_exponent_lift_lt_direct_of_pointwise`, of which this statement is a corollary in one
line: the chain above orders the two diffusions at every point of the band, and the converse
implication fails. Supplying the ordering on the band, in either form, is
`DiffusionOrdering.lean`'s business; `CouplingSign.lean` is why it must be supplied rather than
derived. No upper bracket on the lift's diffusion appears, and none is needed: the pointwise
route never asks for one. -/
theorem barrier_exponent_lift_lt_direct_of_bands
    {UeffDir UeffLift sigma2Dir sigma2Lift : ℝ → ℝ}
    {eta cminDir cmaxDir cminLift : ℝ}
    (hwb : wb ≤ ws) (halpha : 0 < Ltilde ws - Ltilde wb) (heta : 0 < eta)
    (hcminDir : 0 < cminDir) (hsep : cmaxDir < cminLift)
    (hlowerDir : ∀ u ∈ Icc wb ws, cminDir ≤ sigma2Dir u)
    (hupperDir : ∀ u ∈ Icc wb ws, sigma2Dir u ≤ cmaxDir)
    (hlowerLift : ∀ u ∈ Icc wb ws, cminLift ≤ sigma2Lift u)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUdir : ∀ u ∈ Icc wb ws, HasDerivAt UeffDir (dL u / sigma2Dir u) u)
    (hUlift : ∀ u ∈ Icc wb ws, HasDerivAt UeffLift (dL u / sigma2Lift u) u)
    (hcontL : ContinuousOn dL (Icc wb ws))
    (hcontDir : ContinuousOn sigma2Dir (Icc wb ws))
    (hcontLift : ContinuousOn sigma2Lift (Icc wb ws)) :
    (2 / eta) * (UeffLift ws - UeffLift wb) < (2 / eta) * (UeffDir ws - UeffDir wb) :=
  barrier_exponent_lift_lt_direct_of_pointwise hwb halpha heta
    (fun u hu => lt_of_lt_of_le hcminDir (hlowerDir u hu))
    (fun u hu => lt_of_le_of_lt (hupperDir u hu) (lt_of_lt_of_le hsep (hlowerLift u hu)))
    hgrad hLtilde hUdir hUlift hcontL hcontDir hcontLift

/-- **The corrected Corollary 1 on a band, with the two channels carried per iterate.**

This is the previous theorem instantiated at the corrected decomposition of §2, in the form the
note's own bookkeeping produces it. The direct baseline's diffusion is the constant
`s^2 sigma_obj^2`: §2's fact that `dtheta = 0` for the baseline removes the two extra channels
identically, and nothing that remains varies along the band. The lift's is
`s^2 sigma_obj^2 + coupling(u) + quadratic(u)`, with both channels functions of the reference
iterate `u`, because both are covariances evaluated there — `coupling` is the cross channel
`c1 sigma_Jac^2 |H_bb|` and `quadratic` is `s^2 (H V H)_bb`, each read at `u`. The conclusion is
the strict ordering of the two quasipotential barrier exponents.

The sign hypotheses are the band forms of the ones carried throughout this file: `hcoupling` is
assumption (A6) of §11.3 asked at every iterate of the band and is an assumption rather than a
theorem (`CouplingSign.lean`), `hquadratic` is a theorem once the channel is instantiated as a
quadratic form (`TraceLemmas.quadForm_conj_nonneg`, used in
`arrheniusExponent_lift_lt_direct_of_quadForm`), and `hnondeg` is the non-degeneracy without
which the two diffusions may coincide and no strict ordering can hold. -/
theorem barrier_exponent_lift_lt_direct_of_channels
    {UeffDir UeffLift sigma2Lift coupling quadratic : ℝ → ℝ} {eta s sigmaObj : ℝ}
    (hwb : wb ≤ ws) (halpha : 0 < Ltilde ws - Ltilde wb) (heta : 0 < eta)
    (hdirect : 0 < directDiffusion s sigmaObj)
    (hband : ∀ u ∈ Icc wb ws,
      sigma2Lift u = liftDiffusion s sigmaObj (coupling u) (quadratic u))
    (hcoupling : ∀ u ∈ Icc wb ws, 0 ≤ coupling u)
    (hquadratic : ∀ u ∈ Icc wb ws, 0 ≤ quadratic u)
    (hnondeg : ∀ u ∈ Icc wb ws, 0 < coupling u ∨ 0 < quadratic u)
    (hgrad : ∀ u ∈ Icc wb ws, 0 ≤ dL u)
    (hLtilde : ∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u)
    (hUdir : ∀ u ∈ Icc wb ws, HasDerivAt UeffDir (dL u / directDiffusion s sigmaObj) u)
    (hUlift : ∀ u ∈ Icc wb ws, HasDerivAt UeffLift (dL u / sigma2Lift u) u)
    (hcontL : ContinuousOn dL (Icc wb ws))
    (hcontLift : ContinuousOn sigma2Lift (Icc wb ws)) :
    (2 / eta) * (UeffLift ws - UeffLift wb) < (2 / eta) * (UeffDir ws - UeffDir wb) := by
  refine barrier_exponent_lift_lt_direct_of_pointwise
    (sigma2Dir := fun _ => directDiffusion s sigmaObj) hwb halpha heta (fun _ _ => hdirect)
    (fun u hu => ?_) hgrad hLtilde hUdir hUlift hcontL continuousOn_const hcontLift
  rw [hband u hu]
  exact directDiffusion_lt_liftDiffusion (hcoupling u hu) (hquadratic u hu) (hnondeg u hu)

/-- **The hypotheses of the joined band comparison are jointly satisfiable with a genuinely
non-constant lift diffusion — certified in the statement, not merely exhibited in a proof** — so
neither `barrier_exponent_lift_lt_direct_of_bands` nor
`barrier_exponent_lift_lt_direct_of_pointwise` is vacuously quantified. The existential lists
every hypothesis of the bracketed band theorem, adds the non-constancy
`sigma2Lift wb ≠ sigma2Lift ws`, adds the pointwise ordering
`sigma2Dir u < sigma2Lift u` that the pointwise theorem consumes in place of the brackets, and
closes with the instantiated conclusion, which the proof obtains by applying the bracketed band
comparison itself to the other conjuncts. The instance: the band `[0, 1]`, the pullback potential
`Ltilde(u) = u` (barrier action `alpha = 1`, climbed monotonically), the direct baseline at the
constant diffusion `sigma_eff^{2,dir} = 1` with quasipotential `U_eff^dir(u) = u`, and the lift
at the nowhere-constant diffusion `sigma_eff^{2,lift}(u) = u + 2` with
`sigma_eff^{2,lift}(0) = 2 ≠ 3 = sigma_eff^{2,lift}(1)`, bracketed below by `c_min^lift = 2` and
separated from `c_max^dir = 1`, with quasipotential `U_eff^lift(u) = log (u + 2)` solving
`U' = Ltilde' / sigma_eff^2`. A constant direct diffusion is the physically faithful choice — the
baseline's `s^2 sigma_obj^2` carries no `dtheta` dependence — while the lift's varies across the
band, which is exactly the situation the bracketed theorem exists for. At `eta = 2` the
instantiated conclusion is the true numerical statement that the lift's quasipotential barrier
`log 3 - log 2 = log (3/2) ≈ 0.405` lies strictly below the direct baseline's barrier `1`. -/
theorem barrier_exponent_lift_lt_direct_of_bands_witness :
    ∃ (Ltilde dL UeffDir UeffLift sigma2Dir sigma2Lift : ℝ → ℝ)
      (wb ws eta cminDir cmaxDir cminLift : ℝ),
      wb ≤ ws ∧ 0 < Ltilde ws - Ltilde wb ∧ 0 < eta ∧
        0 < cminDir ∧ cmaxDir < cminLift ∧
        (∀ u ∈ Icc wb ws, cminDir ≤ sigma2Dir u) ∧
        (∀ u ∈ Icc wb ws, sigma2Dir u ≤ cmaxDir) ∧
        (∀ u ∈ Icc wb ws, cminLift ≤ sigma2Lift u) ∧
        (∀ u ∈ Icc wb ws, 0 ≤ dL u) ∧
        (∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u) ∧
        (∀ u ∈ Icc wb ws, HasDerivAt UeffDir (dL u / sigma2Dir u) u) ∧
        (∀ u ∈ Icc wb ws, HasDerivAt UeffLift (dL u / sigma2Lift u) u) ∧
        ContinuousOn dL (Icc wb ws) ∧ ContinuousOn sigma2Dir (Icc wb ws) ∧
        ContinuousOn sigma2Lift (Icc wb ws) ∧
        sigma2Lift wb ≠ sigma2Lift ws ∧
        (∀ u ∈ Icc wb ws, sigma2Dir u < sigma2Lift u) ∧
        (2 / eta) * (UeffLift ws - UeffLift wb) < (2 / eta) * (UeffDir ws - UeffDir wb) := by
  have hne : ∀ u ∈ Icc (0 : ℝ) 1, u + 2 ≠ 0 := by
    intro u hu
    have := hu.1
    positivity
  have hwb : (0 : ℝ) ≤ 1 := by norm_num
  have halpha : (0 : ℝ) < (1 : ℝ) - 0 := by norm_num
  have heta : (0 : ℝ) < 2 := by norm_num
  have hcminDir : (0 : ℝ) < 1 := one_pos
  have hsep : (1 : ℝ) < 2 := one_lt_two
  have hlowerDir : ∀ u ∈ Icc (0 : ℝ) 1, (1 : ℝ) ≤ (1 : ℝ) := fun _ _ => le_refl 1
  have hlowerLift : ∀ u ∈ Icc (0 : ℝ) 1, (2 : ℝ) ≤ u + 2 := by
    intro u hu
    have := hu.1
    linarith
  have hgrad : ∀ u ∈ Icc (0 : ℝ) 1, (0 : ℝ) ≤ (1 : ℝ) := fun _ _ => zero_le_one
  have hLtilde : ∀ u ∈ Icc (0 : ℝ) 1, HasDerivAt (fun u : ℝ => u) (1 : ℝ) u :=
    fun u _ => hasDerivAt_id' u
  have hUdir : ∀ u ∈ Icc (0 : ℝ) 1, HasDerivAt (fun u : ℝ => u) ((1 : ℝ) / 1) u :=
    fun u _ => by simpa using hasDerivAt_id' u
  have hUlift : ∀ u ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun u : ℝ => Real.log (u + 2)) ((1 : ℝ) / (u + 2)) u := by
    intro u hu
    simpa using ((hasDerivAt_id u).add_const (2 : ℝ)).log (hne u hu)
  have hcontL : ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Icc 0 1) := continuousOn_const
  have hcontLift : ContinuousOn (fun u : ℝ => u + 2) (Icc 0 1) :=
    (continuous_id.add continuous_const).continuousOn
  have hdom : ∀ u ∈ Icc (0 : ℝ) 1, (1 : ℝ) < u + 2 := by
    intro u hu
    have := hu.1
    linarith
  exact ⟨fun u => u, fun _ => 1, fun u => u, fun u => Real.log (u + 2),
    fun _ => 1, fun u => u + 2, 0, 1, 2, 1, 1, 2,
    hwb, halpha, heta, hcminDir, hsep, hlowerDir, hlowerDir, hlowerLift, hgrad,
    hLtilde, hUdir, hUlift, hcontL, hcontL, hcontLift, by norm_num, hdom,
    barrier_exponent_lift_lt_direct_of_bands
      (Ltilde := fun u : ℝ => u) (dL := fun _ : ℝ => (1 : ℝ))
      (UeffDir := fun u : ℝ => u) (UeffLift := fun u : ℝ => Real.log (u + 2))
      (sigma2Dir := fun _ : ℝ => (1 : ℝ)) (sigma2Lift := fun u : ℝ => u + 2)
      (wb := 0) (ws := 1) (eta := 2) (cminDir := 1) (cmaxDir := 1) (cminLift := 2)
      hwb halpha heta hcminDir hsep hlowerDir hlowerDir hlowerLift hgrad
      hLtilde hUdir hUlift hcontL hcontL hcontLift⟩

/-- **The channel form of the band comparison is satisfiable, with a channel that genuinely
varies along the band — certified in the statement rather than in a proof.**

`barrier_exponent_lift_lt_direct_of_channels` carries nine hypotheses about the band, and a
referee is entitled to ask whether they can be met at once by anything other than a constant
diffusion, in which case the theorem would be `barrier_exponent_lift_lt_direct_of_const_bands`
under another name. This existential answers that: the band `[0, 1]`, the pullback potential
`Ltilde(u) = u` (barrier action `alpha = 1`, climbed monotonically), the positivity map and objective
scales `s = sigma_obj = 1` so that the direct baseline's diffusion is the constant `1` with
quasipotential `U_eff^dir(u) = u`, the coupling channel `coupling(u) = u + 1`, which is strictly
positive and non-constant along the band (`coupling(0) = 1 ≠ 2 = coupling(1)`), the quadratic
channel identically zero — the empirically inert case of §§10-11.0 — and hence the lift's
diffusion `sigma_eff^{2,lift}(u) = 1 + (u + 1) + 0 = u + 2` with quasipotential
`U_eff^lift(u) = log (u + 2)`. At `eta = 2` the instantiated conclusion is the true numerical
statement `log 3 - log 2 = log (3/2) < 1`. -/
theorem barrier_exponent_lift_lt_direct_of_channels_witness :
    ∃ (Ltilde dL UeffDir UeffLift sigma2Lift coupling quadratic : ℝ → ℝ)
      (wb ws eta s sigmaObj : ℝ),
      wb ≤ ws ∧ 0 < Ltilde ws - Ltilde wb ∧ 0 < eta ∧ 0 < directDiffusion s sigmaObj ∧
        (∀ u ∈ Icc wb ws, sigma2Lift u = liftDiffusion s sigmaObj (coupling u) (quadratic u)) ∧
        (∀ u ∈ Icc wb ws, 0 ≤ coupling u) ∧
        (∀ u ∈ Icc wb ws, 0 ≤ quadratic u) ∧
        (∀ u ∈ Icc wb ws, 0 < coupling u ∨ 0 < quadratic u) ∧
        (∀ u ∈ Icc wb ws, 0 ≤ dL u) ∧
        (∀ u ∈ Icc wb ws, HasDerivAt Ltilde (dL u) u) ∧
        (∀ u ∈ Icc wb ws, HasDerivAt UeffDir (dL u / directDiffusion s sigmaObj) u) ∧
        (∀ u ∈ Icc wb ws, HasDerivAt UeffLift (dL u / sigma2Lift u) u) ∧
        ContinuousOn dL (Icc wb ws) ∧ ContinuousOn sigma2Lift (Icc wb ws) ∧
        coupling wb ≠ coupling ws ∧
        (2 / eta) * (UeffLift ws - UeffLift wb) < (2 / eta) * (UeffDir ws - UeffDir wb) := by
  have hne : ∀ u ∈ Icc (0 : ℝ) 1, u + 2 ≠ 0 := by
    intro u hu
    have := hu.1
    positivity
  have hwb : (0 : ℝ) ≤ 1 := by norm_num
  have halpha : (0 : ℝ) < (1 : ℝ) - 0 := by norm_num
  have heta : (0 : ℝ) < 2 := by norm_num
  have hdirect : 0 < directDiffusion (1 : ℝ) 1 := by norm_num [directDiffusion]
  have hband : ∀ u ∈ Icc (0 : ℝ) 1,
      u + 2 = liftDiffusion (1 : ℝ) 1 (u + 1) 0 := by
    intro u _
    simp only [liftDiffusion]
    ring
  have hcoupling : ∀ u ∈ Icc (0 : ℝ) 1, (0 : ℝ) ≤ u + 1 := by
    intro u hu
    have := hu.1
    linarith
  have hnondeg : ∀ u ∈ Icc (0 : ℝ) 1, (0 : ℝ) < u + 1 ∨ (0 : ℝ) < 0 := by
    intro u hu
    have := hu.1
    exact Or.inl (by linarith)
  have hLtilde : ∀ u ∈ Icc (0 : ℝ) 1, HasDerivAt (fun u : ℝ => u) (1 : ℝ) u :=
    fun u _ => hasDerivAt_id' u
  have hUdir : ∀ u ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun u : ℝ => u) ((1 : ℝ) / directDiffusion (1 : ℝ) 1) u :=
    fun u _ => by simpa [directDiffusion] using hasDerivAt_id' u
  have hUlift : ∀ u ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun u : ℝ => Real.log (u + 2)) ((1 : ℝ) / (u + 2)) u := by
    intro u hu
    simpa using ((hasDerivAt_id u).add_const (2 : ℝ)).log (hne u hu)
  have hcontL : ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Icc 0 1) := continuousOn_const
  have hcontLift : ContinuousOn (fun u : ℝ => u + 2) (Icc 0 1) :=
    (continuous_id.add continuous_const).continuousOn
  exact ⟨fun u => u, fun _ => 1, fun u => u, fun u => Real.log (u + 2),
    fun u => u + 2, fun u => u + 1, fun _ => 0, 0, 1, 2, 1, 1,
    hwb, halpha, heta, hdirect, hband, hcoupling, fun _ _ => le_rfl, hnondeg,
    fun _ _ => zero_le_one, hLtilde, hUdir, hUlift, hcontL, hcontLift, by norm_num,
    barrier_exponent_lift_lt_direct_of_channels
      (Ltilde := fun u : ℝ => u) (dL := fun _ : ℝ => (1 : ℝ))
      (UeffDir := fun u : ℝ => u) (UeffLift := fun u : ℝ => Real.log (u + 2))
      (sigma2Lift := fun u : ℝ => u + 2) (coupling := fun u : ℝ => u + 1)
      (quadratic := fun _ : ℝ => (0 : ℝ)) (wb := 0) (ws := 1) (eta := 2) (s := 1) (sigmaObj := 1)
      hwb halpha heta hdirect hband hcoupling (fun _ _ => le_rfl) hnondeg
      (fun _ _ => zero_le_one) hLtilde hUdir hUlift hcontL hcontLift⟩

end Bridge

end IcnnLift
