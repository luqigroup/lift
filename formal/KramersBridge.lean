import Mathlib
import KramersExitTime
import BarrierRescaling

/-!
# The Kramers bridge: how much of `BarrierRescaling`'s `hkramers` is already a theorem

`BarrierRescaling.lean` states the Kramers relation as the named hypothesis

    hkramers : ∀ x, 0 < x → tau x = C * Real.exp (arrheniusExponent alpha eta x)

— an *exact* identity, holding at every effective diffusion `x > 0`, with one and the same
diffusion-independent prefactor `C`. It is the single largest unformalized input of that file,
and it is used there in exactly two places, `escapeTime_ratio_of_kramers` and
`escapeTime_ratio_of_kramers_of_prefactors`, both of which draw from it a statement about the
*ratio* of two escape times. `KramersExitTime.lean` has since proved, for the verified and
unique solution of the mean-first-passage boundary-value problem, the two-sided Arrhenius bound

    (2/σ²) L exp(2(α - δ)/σ²)  ≤  T(x)  ≤  (2/σ²) (b-x)(b-a) exp(2α/σ²),

with `L > 0` a window constant that is the same for every `σ²` at once. This file asks, and
answers, the only question that then remains: which of the conclusions `BarrierRescaling` draws
under `hkramers` are already theorems, and how much of the ratio identity survives the loss of
the sharp prefactor.

**The ordering survives outright, in the regime the paper works in.** The ordering conclusion
of `hkramers` — the lift's escape time is strictly the smaller of the two — is re-derived here
from the two proved bounds and nothing else. The quantifier structure matters and is not the
one `KramersExitTime` already provides. That module varies the two diffusions either
independently (`eventually_meanExitTime_lift_lt_direct`, lifted diffusion fixed) or additively
(`eventually_meanExitTime_additive_lift_lt_direct`, `σ²` against `σ² + σ_Jac²`). The pair the
corrected mechanism actually compares is *multiplicatively* coupled: the bias-channel diffusion
of §3 of `docs/design/lemma1_rederivation.md` is `dw = -L̃'(w) dt + √η σ_eff(w) dB`, so the
diffusion coefficient of the boundary-value problem is `η σ_eff²` and the two members of the
comparison are `η · directDiffusion` and `η · liftDiffusion` with one shared learning rate `η`.
`eventually_meanExitTime_lift_lt_direct_smallLearningRate` proves the strict ordering for that
family, for every sufficiently small learning rate, carrying no gap hypothesis and no hypothesis
about prefactors; `eventually_meanExitTime_lift_lt_direct_of_channels` states it in
`BarrierRescaling`'s own channel vocabulary, so that the sign hypotheses it consumes are exactly
`hcoupling`, `hquadratic` and `hnondeg` and `hkramers` has disappeared. The small learning rate
is the paper's own operating assumption, not an artefact.

**The ratio identity survives as a bracket, and, at logarithmic scale, as an exact limit.**
`meanExitTime_ratio_bracket` converts the two-sided bound into a two-sided bound on the ratio of
exit times, which is the honest replacement for the identity `tau sd / tau sl = escapeTimeRatio`
that `hkramers` delivers: the ratio is pinned between `(L sl)/(P sd) exp(2(α-δ)/sd - 2α/sl)` and
`(P sl)/(L sd) exp(2α/sd - 2(α-δ)/sl)`, where `P = (b-x)(b-a)`. The bracket is wide — its two
endpoints differ by the factor `(P/L)² exp(2δ/sd + 2δ/sl)` — and that width is precisely the
sharp prefactor this development does not have. At logarithmic scale, however, the bracket
closes completely in the small-learning-rate limit:
`tendsto_mul_log_ratio_sub_escapeTimeRatio` proves that the discrepancy between the true
log-ratio and the log-ratio `hkramers` predicts, both scaled by `η`, tends to zero. That is the
exact sense in which the ratio identity of `escapeTime_ratio_of_kramers` is recovered without
`hkramers`, and it is also the exact sense in which it is not: the identity holds in the limit
after multiplication by `η`, which controls the exponent and says nothing about the prefactor.

**`hkramers` is not merely unrefuted; it is sandwiched.** `kramers_hypothesis_sandwiched`
exhibits, for every barrier and every learning rate, a positive constant `C` and a window
`(0, v₀]` of effective diffusions on which a function of `hkramers`' literal shape,
`v ↦ C exp(arrheniusExponent α η v)`, satisfies *both* proved bounds simultaneously — the same
two bounds the true mean exit time satisfies. Consequently the sharp Kramers function and the
true exit time lie in one and the same bracket, and their ratio is pinned between
`(L/P) exp(-2δ/(ηv))` and `(P/L) exp(2δ/(ηv))`. Nothing that `KramersExitTime` proves can
contradict `hkramers`; the hypothesis is consistent with the entire development, which is the
strongest statement available short of a derivation.

**What would discharge `hkramers` outright, and why it is not done here.** The bounds proved in
`KramersExitTime` are Laplace estimates carried out to zeroth order: the outer integral is
restricted to a window around the barrier top on which `U ≥ U(w_s) - δ/2`, the inner integral to
a window around the basin bottom on which `U ≤ U(w_b) + δ/2`, and the two windows contribute
their lengths as the constants. Discharging `hkramers` requires the *second-order Laplace
expansion at the two extrema*: writing `U(y) = U(w_s) - ½|U''(w_s)|(y - w_s)² + O((y-w_s)³)` at
the barrier top and `U(z) = U(w_b) + ½U''(w_b)(z - w_b)² + O((z-w_b)³)` at the basin bottom,
evaluating the two resulting Gaussian integrals, and controlling the cubic remainders uniformly
in `σ²`. That computation yields the Eyring–Kramers prefactor
`2π / √(U''(w_b) |U''(w_s)|)`, which is `σ`-independent, and with it the exact relation
`T(x) = (2π / √(U''(w_b)|U''(w_s)|)) exp(2α/σ²) (1 + o(1))` — that is, `hkramers` with an
explicit `C` and an explicit error. Nothing in the argument needs stochastic calculus: it is an
asymptotic evaluation of the double integral `meanExitTime` already defines, and it needs `U` to
be `C³` near the two extrema with `U''(w_b) > 0` and `U''(w_s) < 0`, which is the paper's (A5)
in its full strength rather than the `C¹` weakening used so far. It is a substantial piece of
real analysis — a uniform Laplace method with remainder — and it is **future work**; it is named
here so that the remaining gap is a definite piece of mathematics and not a gesture. Until it is
carried out, `hkramers` stays a hypothesis, and every conclusion `BarrierRescaling` draws from
it that this file does not re-derive stays conditional.

**Satisfiability, and one negative result.** The chain is non-vacuous end to end:
`smallLearningRate_ordering_nonvacuous` exhibits a concrete potential, interval, start point and
channel decomposition satisfying every hypothesis of the ordering theorem *together with the
instantiated conclusion*, so that the kernel checks the whole chain in the statement rather than
only inside a proof. The potential is the cosine barrier `U(y) = -cos y` on `[-1, 4]` with basin
bottom `w_b = 0` and barrier top `w_s = π`; its two extremal hypotheses hold globally, not just
on the interval. The obvious candidate — a quadratic barrier — is **not** available, and
`no_quadratic_single_barrier` proves it: a quadratic whose basin bottom and barrier top are both
interior to `[a, b]` and are respectively the minimum and the maximum of the quadratic on
`[a, b]` must have vanishing leading and linear coefficients, hence be constant, hence have zero
barrier action. The single-barrier structure of (A5) with both extrema interior forces two
interior critical points, and so forces a potential of degree at least three (or a transcendental
one such as `-cos`). This is why `KramersExitTime`'s own witness is a cubic and why the witness
here is a cosine.

## Results
* `arrheniusExponent_eq_two_mul_div` — the dictionary: `BarrierRescaling`'s exponent at effective
  diffusion `σ_eff²` and learning rate `η` is `KramersExitTime`'s exponent at diffusion
  coefficient `η σ_eff²`.
* `escapeTimeRatio_eq_exp_sub` — `escapeTimeRatio` is the exponential of the difference of the
  two exponents, in the same variables.
* `meanExitTime_pos` — the closed-form exit time is strictly positive.
* `eventually_lt_exp_div` — `exp(γ/η)` exceeds any prescribed threshold for small `η`.
* `eventually_meanExitTime_lift_lt_direct_smallLearningRate` — the strict escape-time ordering
  for the multiplicatively coupled pair `η σ_d²` against `η σ_l²`, for every sufficiently small
  learning rate, with no gap hypothesis: `hkramers`' ordering consequence, discharged.
* `eventually_meanExitTime_lift_lt_direct_of_channels` — the same in `BarrierRescaling`'s channel
  vocabulary, consuming `hcoupling`, `hquadratic` and `hnondeg` and nothing else.
* `eventually_mean_first_passage_lift_lt_direct_smallLearningRate` — the same for the abstract
  mean first-passage times, carrying Dynkin's formula as its only named hypothesis.
* `div_mem_bracket_of_bounds` — two-sided bounds on a numerator and a denominator bracket their
  quotient.
* `meanExitTime_ratio_bracket` — the resulting two-sided bracket on the ratio of exit times: what
  survives of `escapeTime_ratio_of_kramers` without the sharp prefactor.
* `mul_exp_neg_le_one` — `t exp(-t) ≤ 1`, the elementary bound that makes the sandwich work.
* `sharp_kramers_within_arrhenius_bounds` — a function of `hkramers`' exact shape satisfies both
  Arrhenius bounds on an explicit window `(0, v₀]`, with `C` and `v₀` given in closed form.
* `kramers_hypothesis_sandwiched` — hence, at the constants `KramersExitTime` actually produces,
  the sharp Kramers function and the true exit time lie in one bracket, and their ratio is
  pinned between `(L/P) exp(-2δ/(ηv))` and `(P/L) exp(2δ/(ηv))`.
* `tendsto_nhdsGT_mul_const`, `tendsto_mul_log_meanExitTime_scaled` — the small-noise limit
  transported to the learning-rate variable.
* `tendsto_mul_log_meanExitTime_ratio` — `η log(T_d/T_l) → 2α(1/σ_d² - 1/σ_l²)`.
* `mul_log_escapeTimeRatio` — `η log(escapeTimeRatio α η σ_d² σ_l²) = 2α(1/σ_d² - 1/σ_l²)`,
  exactly.
* `tendsto_mul_log_ratio_sub_escapeTimeRatio` — hence the discrepancy between the true log-ratio
  and the one `hkramers` predicts vanishes: the ratio identity is recovered at logarithmic scale.
* `no_quadratic_single_barrier` — no quadratic potential satisfies the interior single-barrier
  hypotheses with a positive barrier.
* `smallLearningRate_ordering_nonvacuous` — the cosine-barrier instance, with the instantiated
  conclusion carried in the statement.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. Nothing here is added to `KramersExitTime.lean` or to
`BarrierRescaling.lean`; both are imported unchanged.

The hypotheses carried, and what each would take to discharge:

* `hU`, `hax`, `hawb`, `hwbws`, `hwsb`, `hxws`, `hmax`, `hmin` — `KramersExitTime`'s own
  regularity and single-barrier stack, weakened there to `C⁰` on the closed interval, plus the
  ordering `a < w_b < w_s < b` and the two extremal clauses of (A5). They are inherited verbatim,
  and `smallLearningRate_ordering_nonvacuous` discharges all of them on a concrete instance.
* `hDynkin` in the paper-facing form — Dynkin's formula for the generator
  `A = (σ²/2)∂² - U'∂`. It is `KramersExitTime`'s hypothesis, inherited unchanged; this mathlib
  has no Itô calculus, so it cannot be discharged here.
* `hcoupling : 0 ≤ coupling` in the channel form — assumption (A6) of §11.3 of the note, which
  `CouplingSign.lean` machine-checks cannot be traded for a derivation. It enters only through
  `directDiffusion_lt_liftDiffusion` and is `BarrierRescaling`'s hypothesis, not a new one.

What this file does **not** prove, and must not be read as proving:

* It does not prove `hkramers`. It proves that `hkramers`' ordering consequence holds without it
  in the small-learning-rate regime, that its ratio consequence holds without it at logarithmic
  scale in the limit, and that `hkramers` itself is consistent with everything proved. The exact
  identity, with one diffusion-independent constant valid at every `σ²`, is neither proved nor
  refuted here, and the second-order Laplace expansion described above is what would settle it.
* It does not sharpen the prefactor by any amount. The constants `L` and `P = (b-x)(b-a)` are
  `KramersExitTime`'s crude ones, and `L` still degenerates as the Laplace tolerance `δ → 0`.
  Every bracket below is therefore wide at fixed `σ²`, and the only place the two sides meet is
  the logarithmic limit.
* It does not give a threshold. "For every sufficiently small learning rate" is a statement about
  the filter `𝓝[>] 0` and is not converted into an explicit inequality anywhere; converting it
  is again a question about the prefactor.
-/

open Set Filter
open scoped Topology

namespace IcnnLift

/-! ### The dictionary between the two parametrizations

`BarrierRescaling` writes the escape exponent as `arrheniusExponent alpha eta sigma2eff`, a
function of the barrier action, the learning rate and the *effective diffusion*;
`KramersExitTime` writes it as `2 alpha / sigma2`, a function of the barrier action and the
*diffusion coefficient of the boundary-value problem*. The bias-channel diffusion of §3 of the
note is `dw = -L̃'(w) dt + √η σ_eff(w) dB`, whose diffusion coefficient is `η σ_eff²`, so the two
agree under the substitution `sigma2 = eta * sigma2eff`. Every theorem below uses that
substitution and nothing else; it is recorded here so that the identification is visible rather
than implicit. -/

/-- `BarrierRescaling`'s Arrhenius exponent at effective diffusion `sigma2` and learning rate
`eta` is `KramersExitTime`'s exponent `2 alpha / (eta sigma2)` at the diffusion coefficient
`eta * sigma2` of the boundary-value problem. -/
theorem arrheniusExponent_eq_two_mul_div (alpha eta sigma2 : ℝ) :
    arrheniusExponent alpha eta sigma2 = 2 * alpha / (eta * sigma2) := rfl

/-- The escape-time ratio of `BarrierRescaling` is the exponential of the difference of the two
Arrhenius exponents, written in the diffusion coefficients `eta * sd` and `eta * sl` that the
boundary-value problem uses. -/
theorem escapeTimeRatio_eq_exp_sub {alpha eta sd sl : ℝ} (heta : eta ≠ 0) (hd : sd ≠ 0)
    (hl : sl ≠ 0) :
    escapeTimeRatio alpha eta sd sl
      = Real.exp (2 * alpha / (eta * sd) - 2 * alpha / (eta * sl)) := by
  rw [escapeTimeRatio]
  congr 1
  field_simp

/-! ### Elementary facts about the proved exit time -/

/-- The closed-form mean exit time is strictly positive under `KramersExitTime`'s geometry: the
Arrhenius lower bound at any tolerance already forces it. This is needed wherever the ratio of
two exit times is formed or a logarithm is taken. -/
theorem meanExitTime_pos {U : ℝ → ℝ} {a b x wb ws : ℝ} {sigma2 : ℝ} (hs : 0 < sigma2)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) :
    0 < meanExitTime sigma2 U a b x := by
  obtain ⟨c, hc, hle⟩ :=
    meanExitTime_arrhenius_lower_bound (delta := 1) hs hU hax hawb hwbws hwsb hxws one_pos
  exact lt_of_lt_of_le (mul_pos hc (Real.exp_pos _)) hle

/-- For a positive exponent scale `gamma`, `exp (gamma / eta)` exceeds any prescribed threshold
once the learning rate `eta` is small enough. This is the learning-rate analogue of
`KramersExitTime.eventually_lower_bound_gt` and is what removes the gap hypothesis from the
ordering theorem below. -/
theorem eventually_lt_exp_div {gamma : ℝ} (hgamma : 0 < gamma) (kappa : ℝ) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ), kappa < Real.exp (gamma / eta) := by
  have h1 : Filter.Tendsto (fun eta : ℝ => gamma / eta) (𝓝[>] (0 : ℝ)) Filter.atTop := by
    simp only [div_eq_mul_inv]
    exact Filter.Tendsto.const_mul_atTop hgamma tendsto_inv_nhdsGT_zero
  exact (Real.tendsto_exp_atTop.comp h1).eventually_gt_atTop kappa

/-! ### The ordering consequence of `hkramers`, discharged

`BarrierRescaling` uses `hkramers` to conclude, through `escapeTime_ratio_of_kramers` and
`one_lt_escapeTimeRatio_lift`, that the lift's escape time is strictly the smaller of the two.
That conclusion is re-derived here from the two Arrhenius bounds of `KramersExitTime` alone, for
the family of diffusion coefficients the corrected mechanism actually compares. -/

/-- **The strict escape-time ordering for the multiplicatively coupled pair.**

Fix the potential, the interval, the start point and two effective diffusions `sd < sl`. The
bias-channel diffusion carries the learning rate inside the noise amplitude, so the two
boundary-value problems being compared have diffusion coefficients `eta * sd` and `eta * sl` with
one shared `eta`. Then for every sufficiently small learning rate the exit time at the larger
effective diffusion is strictly the smaller.

No gap hypothesis and no hypothesis relating the two Arrhenius prefactors is carried: the
tolerance `delta = alpha (sl - sd) / (2 sl)` is chosen inside the proof so that the two exponent
scales separate, `2(alpha - delta)/sd - 2 alpha/sl = alpha (sl - sd)/(sl sd) > 0`, and
`meanExitTime_arrhenius_lower_bound_uniform` supplies one window constant valid at both diffusion
coefficients at once, so that the separated exponents beat the two prefactors as `eta → 0⁺`.

This is `hkramers`' ordering consequence with `hkramers` removed. What it does not give, and
what the sharp Kramers prefactor would give, is a threshold: "sufficiently small" is a statement
about the filter `𝓝[>] 0` and is not made explicit. -/
theorem eventually_meanExitTime_lift_lt_direct_smallLearningRate {U : ℝ → ℝ}
    {a b x wb ws sd sl : ℝ} (hsd : 0 < sd) (hsdl : sd < sl)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ),
      meanExitTime (eta * sl) U a b x < meanExitTime (eta * sd) U a b x := by
  have hsl : 0 < sl := hsd.trans hsdl
  have halpha : 0 < U ws - U wb := by linarith
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hP : (0 : ℝ) < (b - x) * (b - a) := by
    have h1 : x < b := lt_of_lt_of_le hxws hwsb.le
    have h2 : a < b := hawb.trans (hwbws.trans hwsb)
    exact mul_pos (by linarith) (by linarith)
  -- the Laplace tolerance, chosen so that the two exponent scales separate
  set delta : ℝ := (U ws - U wb) * (sl - sd) / (2 * sl) with hdelta_def
  have hdelta : 0 < delta := by
    rw [hdelta_def]
    exact div_pos (mul_pos halpha (by linarith)) (by linarith)
  obtain ⟨L, hL, hbound⟩ :=
    meanExitTime_arrhenius_lower_bound_uniform (delta := delta) hU hax hawb hwbws hwsb hxws hdelta
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
  have hup := meanExitTime_upper_bound (sigma2 := eta * sl) (M := U ws) (m := U wb)
    (mul_pos heta hsl) hax hxb hU hmax hmin
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

/-- **The same ordering, in `BarrierRescaling`'s channel vocabulary.**

The two effective diffusions of the corrected decomposition of §2 are
`directDiffusion s sigmaObj = s² σ_obj²` and
`liftDiffusion s sigmaObj coupling quadratic = s² σ_obj² + coupling + quadratic`. Under the sign
hypotheses `BarrierRescaling` already carries — `hcoupling`, the projected form of assumption
(A6) of §11.3, which `CouplingSign.lean` shows is not derivable; `hquadratic`, discharged from
`TraceLemmas.quadForm_conj_nonneg` once the channel is instantiated as a quadratic form; and the
non-degeneracy disjunction — the lift's mean exit time is strictly below the direct baseline's
for every sufficiently small learning rate.

This is the statement `BarrierRescaling.one_lt_escapeTimeRatio_lift` makes about the *predicted*
exponent-only ratio, made instead about the exit times themselves, with `hkramers` gone. -/
theorem eventually_meanExitTime_lift_lt_direct_of_channels {U : ℝ → ℝ}
    {a b x wb ws s sigmaObj coupling quadratic : ℝ}
    (hdirect : 0 < directDiffusion s sigmaObj) (hcoupling : 0 ≤ coupling)
    (hquadratic : 0 ≤ quadratic) (hnondeg : 0 < coupling ∨ 0 < quadratic)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ),
      meanExitTime (eta * liftDiffusion s sigmaObj coupling quadratic) U a b x
        < meanExitTime (eta * directDiffusion s sigmaObj) U a b x :=
  eventually_meanExitTime_lift_lt_direct_smallLearningRate hdirect
    (directDiffusion_lt_liftDiffusion hcoupling hquadratic hnondeg) hU hax hawb hwbws hwsb hxws
    hbarrier hmax hmin

/-- **The ordering for the abstract mean first-passage times.**

The same conclusion for the mean first-passage times themselves rather than for the closed form,
under Dynkin's formula for both processes at every learning rate. This is `KramersExitTime`'s
hypothesis `hDynkin` and nothing further: uniqueness for the boundary-value problem
(`eq_meanExitTime_of_bvp`) converts it into the closed form, and the ordering is then the theorem
above. It is the paper-facing form of Corollary 1's comparison for the multiplicatively coupled
pair. -/
theorem eventually_mean_first_passage_lift_lt_direct_smallLearningRate {U U' : ℝ → ℝ}
    {a b x wb ws sd sl : ℝ}
    (taud Dtaud DDtaud taul Dtaul DDtaul : ℝ → ℝ → ℝ)
    (hDynkinD : ∀ eta : ℝ, 0 < eta →
      MeanExitTimeBVP (eta * sd) U' a b (taud eta) (Dtaud eta) (DDtaud eta))
    (hDynkinL : ∀ eta : ℝ, 0 < eta →
      MeanExitTimeBVP (eta * sl) U' a b (taul eta) (Dtaul eta) (DDtaul eta))
    (htaud : ∀ eta : ℝ, 0 < eta → ContinuousOn (taud eta) (Set.Icc a b))
    (htaul : ∀ eta : ℝ, 0 < eta → ContinuousOn (taul eta) (Set.Icc a b))
    (hsd : 0 < sd) (hsdl : sd < sl)
    (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ eta in 𝓝[>] (0 : ℝ), taul eta x < taud eta x := by
  have hab : a < b := hawb.trans (hwbws.trans hwsb)
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hsl : 0 < sl := hsd.trans hsdl
  filter_upwards [eventually_meanExitTime_lift_lt_direct_smallLearningRate hsd hsdl hU hax
      hawb hwbws hwsb hxws hbarrier hmax hmin, self_mem_nhdsWithin] with eta h1 h2
  have hetapos : (0 : ℝ) < eta := h2
  rw [eq_meanExitTime_of_bvp (mul_pos hetapos hsl) hab hU hU' (htaul eta hetapos)
      (hDynkinL eta hetapos) x ⟨hax, hxb⟩,
    eq_meanExitTime_of_bvp (mul_pos hetapos hsd) hab hU hU' (htaud eta hetapos)
      (hDynkinD eta hetapos) x ⟨hax, hxb⟩]
  exact h1

/-! ### What survives of the ratio identity: a two-sided bracket

`escapeTime_ratio_of_kramers` derives from `hkramers` the exact identity
`tau sd / tau sl = escapeTimeRatio alpha eta sd sl`. Without the sharp prefactor the identity
becomes a bracket, and the width of that bracket is exactly the prefactor information that is
missing. -/

/-- Two-sided bounds on a numerator and on a denominator bracket their quotient. Elementary, but
it is the step that converts `KramersExitTime`'s two bounds into a statement about a ratio. -/
theorem div_mem_bracket_of_bounds {Td Tl lod hid lol hil : ℝ}
    (hlod : 0 ≤ lod) (hd1 : lod ≤ Td) (hd2 : Td ≤ hid)
    (hlol : 0 < lol) (hl1 : lol ≤ Tl) (hl2 : Tl ≤ hil) :
    lod / hil ≤ Td / Tl ∧ Td / Tl ≤ hid / lol := by
  have hTl : 0 < Tl := lt_of_lt_of_le hlol hl1
  have hhil : 0 < hil := lt_of_lt_of_le hTl hl2
  have hTd : 0 ≤ Td := hlod.trans hd1
  constructor
  · rw [div_le_div_iff₀ hhil hTl]
    nlinarith
  · rw [div_le_div_iff₀ hTl hlol]
    nlinarith

/-- **The ratio of exit times, bracketed.**

With `alpha = U(w_s) - U(w_b)` the barrier action, `P = (b-x)(b-a)` the upper-bound constant and
`L > 0` the window constant of `meanExitTime_arrhenius_lower_bound_uniform`, valid at both
diffusion coefficients at once,

`(L sl)/(P sd) exp(2(alpha - delta)/sd - 2 alpha/sl) ≤ T(sd)/T(sl)
    ≤ (P sl)/(L sd) exp(2 alpha/sd - 2(alpha - delta)/sl)`.

This is the honest replacement for `escapeTime_ratio_of_kramers`. Compare the two: `hkramers`
gives the single value `exp(2 alpha/sd - 2 alpha/sl)`; the proved bounds give an interval
containing it whose two endpoints differ by the factor
`(P/L)² exp(2 delta/sd + 2 delta/sl)`. That factor is precisely the Eyring–Kramers prefactor
information this development does not have, and it does not shrink at fixed diffusion, because
`L` degenerates as `delta → 0`. The bracket does close at logarithmic scale in the
small-learning-rate limit; see `tendsto_mul_log_ratio_sub_escapeTimeRatio` below. -/
theorem meanExitTime_ratio_bracket {U : ℝ → ℝ} {a b x wb ws delta sd sl : ℝ}
    (hsd : 0 < sd) (hsl : 0 < sl) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∃ L > 0,
      L * sl / ((b - x) * (b - a) * sd)
          * Real.exp (2 * (U ws - U wb - delta) / sd - 2 * (U ws - U wb) / sl)
          ≤ meanExitTime sd U a b x / meanExitTime sl U a b x ∧
        meanExitTime sd U a b x / meanExitTime sl U a b x
          ≤ (b - x) * (b - a) * sl / (L * sd)
            * Real.exp (2 * (U ws - U wb) / sd - 2 * (U ws - U wb - delta) / sl) := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hP : (0 : ℝ) < (b - x) * (b - a) := by
    have h1 : x < b := lt_of_lt_of_le hxws hwsb.le
    have h2 : a < b := hawb.trans (hwbws.trans hwsb)
    exact mul_pos (by linarith) (by linarith)
  obtain ⟨L, hL, hbound⟩ :=
    meanExitTime_arrhenius_lower_bound_uniform (delta := delta) hU hax hawb hwbws hwsb hxws hdelta
  have hlowd := hbound sd hsd
  have hlowl := hbound sl hsl
  have hupd := meanExitTime_upper_bound (sigma2 := sd) (M := U ws) (m := U wb)
    hsd hax hxb hU hmax hmin
  have hupl := meanExitTime_upper_bound (sigma2 := sl) (M := U ws) (m := U wb)
    hsl hax hxb hU hmax hmin
  have hlodnn : (0 : ℝ) ≤ 2 / sd * L * Real.exp (2 * (U ws - U wb - delta) / sd) := by positivity
  have hlolpos : (0 : ℝ) < 2 / sl * L * Real.exp (2 * (U ws - U wb - delta) / sl) := by positivity
  obtain ⟨hbr1, hbr2⟩ := div_mem_bracket_of_bounds hlodnn hlowd hupd hlolpos hlowl hupl
  refine ⟨L, hL, le_trans (le_of_eq ?_) hbr1, le_trans hbr2 (le_of_eq ?_)⟩
  · rw [Real.exp_sub]
    field_simp
  · rw [Real.exp_sub]
    field_simp

/-! ### `hkramers` is sandwiched by the proved bounds

The bracket above shows how much of the ratio identity survives. The theorems here answer the
complementary question: is the exact relation `hkramers` postulates *compatible* with everything
`KramersExitTime` proves? It is, and explicitly so — a function of `hkramers`' literal shape
satisfies both proved bounds on an explicit window of effective diffusions. -/

/-- `t exp(-t) ≤ 1` for every real `t`. This is the elementary bound behind the sandwich: it
caps the Arrhenius lower-bound prefactor `(2L/σ²) exp(-2δ/σ²)`, which is unbounded neither as
`σ² → 0⁺` nor as `σ² → ∞`, by the diffusion-independent constant `L/δ`. -/
theorem mul_exp_neg_le_one (t : ℝ) : t * Real.exp (-t) ≤ 1 := by
  have h1 : t ≤ Real.exp t := by have := Real.add_one_le_exp t; linarith
  have h2 : t * Real.exp (-t) ≤ Real.exp t * Real.exp (-t) :=
    mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
  rwa [← Real.exp_add, add_neg_cancel, Real.exp_zero] at h2

/-- **The sharp Kramers relation is consistent with both Arrhenius bounds.**

Given a barrier action `alpha`, a learning rate `eta > 0`, a Laplace tolerance `delta > 0` and
the two bound constants `L, P > 0`, there is a positive constant `C` and a window `(0, v₀]` of
effective diffusions on which the function

    `v ↦ C * Real.exp (arrheniusExponent alpha eta v)`

— that is, a function of `BarrierRescaling.hkramers`' literal shape, with a single
diffusion-independent prefactor — satisfies both the Arrhenius lower bound and the Arrhenius
upper bound of `KramersExitTime`, at the diffusion coefficient `eta * v` of the boundary-value
problem. Both constants are explicit: `C = L / delta` and `v₀ = 2 P delta / (eta L)`.

The lower bound is the substantive half. It asks `(2L/(ηv)) exp(-2δ/(ηv)) ≤ C`, and the left
side is exactly `(L/δ) t exp(-t)` at `t = 2δ/(ηv)`, hence at most `L/δ` by
`mul_exp_neg_le_one`, uniformly in `v`. The upper bound is the reason for the window: the
proved upper-bound prefactor is `2P/(ηv)`, which degrades as `v` grows, so a *constant* `C`
can respect it only below a threshold, and `v₀` is that threshold.

The consequence for the audit is that `hkramers` is not merely unrefuted by the proved bounds;
no instance of them can contradict it. What remains unproved is that the true exit time equals
such a function, which is the second-order Laplace expansion described in the module
docstring. -/
theorem sharp_kramers_within_arrhenius_bounds {alpha eta delta L P : ℝ}
    (heta : 0 < eta) (hL : 0 < L) (hP : 0 < P) (hdelta : 0 < delta) :
    ∃ C > 0, ∃ v₀ > 0, ∀ v : ℝ, 0 < v → v ≤ v₀ →
      2 / (eta * v) * L * Real.exp (2 * (alpha - delta) / (eta * v))
          ≤ C * Real.exp (arrheniusExponent alpha eta v) ∧
        C * Real.exp (arrheniusExponent alpha eta v)
          ≤ 2 / (eta * v) * P * Real.exp (2 * alpha / (eta * v)) := by
  refine ⟨L / delta, div_pos hL hdelta, 2 * P * delta / (eta * L), by positivity,
    fun v hv hle => ?_⟩
  have hev : (0 : ℝ) < eta * v := mul_pos heta hv
  have hexp : arrheniusExponent alpha eta v = 2 * alpha / (eta * v) := rfl
  rw [hexp]
  constructor
  · have hsplit : Real.exp (2 * (alpha - delta) / (eta * v))
        = Real.exp (2 * alpha / (eta * v)) * Real.exp (-(2 * delta / (eta * v))) := by
      rw [← Real.exp_add]
      congr 1
      field_simp
      ring
    have hfac : 2 / (eta * v) * L = L / delta * (2 * delta / (eta * v)) := by field_simp
    have hkey : 2 / (eta * v) * L * Real.exp (-(2 * delta / (eta * v))) ≤ L / delta := by
      rw [hfac]
      have h := mul_exp_neg_le_one (2 * delta / (eta * v))
      calc L / delta * (2 * delta / (eta * v)) * Real.exp (-(2 * delta / (eta * v)))
          = L / delta * (2 * delta / (eta * v) * Real.exp (-(2 * delta / (eta * v)))) := by ring
        _ ≤ L / delta * 1 := mul_le_mul_of_nonneg_left h (by positivity)
        _ = L / delta := by ring
    rw [hsplit]
    calc 2 / (eta * v) * L * (Real.exp (2 * alpha / (eta * v))
            * Real.exp (-(2 * delta / (eta * v))))
        = 2 / (eta * v) * L * Real.exp (-(2 * delta / (eta * v)))
            * Real.exp (2 * alpha / (eta * v)) := by ring
      _ ≤ L / delta * Real.exp (2 * alpha / (eta * v)) :=
          mul_le_mul_of_nonneg_right hkey (Real.exp_pos _).le
  · have hcoef : L / delta ≤ 2 / (eta * v) * P := by
      have h1 : eta * L * v ≤ 2 * P * delta := by
        calc eta * L * v ≤ eta * L * (2 * P * delta / (eta * L)) :=
              mul_le_mul_of_nonneg_left hle (by positivity)
          _ = 2 * P * delta := by field_simp
      have h2 : 2 / (eta * v) * P = 2 * P / (eta * v) := by ring
      rw [h2, div_le_div_iff₀ hdelta hev]
      nlinarith
    exact mul_le_mul_of_nonneg_right hcoef (Real.exp_pos _).le

/-- **`hkramers` and the true exit time lie in one and the same bracket.**

The previous theorem instantiated at the constants `KramersExitTime` actually produces. For the
window constant `L` of `meanExitTime_arrhenius_lower_bound_uniform` and the upper-bound constant
`P = (b-x)(b-a)`, there are a prefactor `C > 0` and a window `(0, v₀]` such that the function
`tau` of `hkramers`' literal shape and the true mean exit time `T(eta * v)` satisfy the *same*
two-sided Arrhenius bound at every effective diffusion in the window. Their ratio is therefore
pinned:

`(L/P) exp(-2 delta/(eta v)) ≤ tau v / T(eta v) ≤ (P/L) exp(2 delta/(eta v))`.

This is the precise sense in which `hkramers` is sandwiched rather than merely unrefuted, and
the two-sided pin is also the precise measure of how far the hypothesis could be from the truth
without contradicting anything proved. The pin is wide, and it is wide for the same reason the
ratio bracket is: the constants are the crude ones, and `L` degenerates as `delta → 0`. -/
theorem kramers_hypothesis_sandwiched {U : ℝ → ℝ} {a b x wb ws delta eta : ℝ}
    (heta : 0 < eta) (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws) (hdelta : 0 < delta)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∃ L > 0, ∃ C > 0, ∃ v₀ > 0, ∃ tau : ℝ → ℝ,
      (∀ v : ℝ, 0 < v → tau v = C * Real.exp (arrheniusExponent (U ws - U wb) eta v)) ∧
      ∀ v : ℝ, 0 < v → v ≤ v₀ →
        (2 / (eta * v) * L * Real.exp (2 * (U ws - U wb - delta) / (eta * v)) ≤ tau v ∧
            tau v ≤ 2 / (eta * v) * ((b - x) * (b - a))
              * Real.exp (2 * (U ws - U wb) / (eta * v))) ∧
          (2 / (eta * v) * L * Real.exp (2 * (U ws - U wb - delta) / (eta * v))
              ≤ meanExitTime (eta * v) U a b x ∧
            meanExitTime (eta * v) U a b x ≤ 2 / (eta * v) * ((b - x) * (b - a))
              * Real.exp (2 * (U ws - U wb) / (eta * v))) ∧
          (L / ((b - x) * (b - a)) * Real.exp (-(2 * delta / (eta * v)))
              ≤ tau v / meanExitTime (eta * v) U a b x ∧
            tau v / meanExitTime (eta * v) U a b x
              ≤ (b - x) * (b - a) / L * Real.exp (2 * delta / (eta * v))) := by
  have hxb : x ≤ b := hxws.le.trans hwsb.le
  have hP : (0 : ℝ) < (b - x) * (b - a) := by
    have h1 : x < b := lt_of_lt_of_le hxws hwsb.le
    have h2 : a < b := hawb.trans (hwbws.trans hwsb)
    exact mul_pos (by linarith) (by linarith)
  obtain ⟨L, hL, hbound⟩ :=
    meanExitTime_arrhenius_lower_bound_uniform (delta := delta) hU hax hawb hwbws hwsb hxws hdelta
  obtain ⟨C, hC, v₀, hv₀, hsharp⟩ :=
    sharp_kramers_within_arrhenius_bounds (alpha := U ws - U wb) (delta := delta)
      (L := L) (P := (b - x) * (b - a)) heta hL hP hdelta
  refine ⟨L, hL, C, hC, v₀, hv₀,
    fun v => C * Real.exp (arrheniusExponent (U ws - U wb) eta v), fun v _ => rfl,
    fun v hv hle => ?_⟩
  have hev : (0 : ℝ) < eta * v := mul_pos heta hv
  obtain ⟨hsl, hsu⟩ := hsharp v hv hle
  have hTl := hbound (eta * v) hev
  have hTu := meanExitTime_upper_bound (sigma2 := eta * v) (M := U ws) (m := U wb)
    hev hax hxb hU hmax hmin
  have hlodnn : (0 : ℝ)
      ≤ 2 / (eta * v) * L * Real.exp (2 * (U ws - U wb - delta) / (eta * v)) := by positivity
  have hlolpos : (0 : ℝ)
      < 2 / (eta * v) * L * Real.exp (2 * (U ws - U wb - delta) / (eta * v)) := by positivity
  obtain ⟨hbr1, hbr2⟩ := div_mem_bracket_of_bounds hlodnn hsl hsu hlolpos hTl hTu
  -- the two directions of the exponent split
  have hE1 : Real.exp (2 * (U ws - U wb - delta) / (eta * v))
      = Real.exp (2 * (U ws - U wb) / (eta * v)) * Real.exp (-(2 * delta / (eta * v))) := by
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  have hE2 : Real.exp (2 * (U ws - U wb) / (eta * v))
      = Real.exp (2 * (U ws - U wb - delta) / (eta * v)) * Real.exp (2 * delta / (eta * v)) := by
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  refine ⟨⟨hsl, hsu⟩, ⟨hTl, hTu⟩, le_trans (le_of_eq ?_) hbr1, le_trans hbr2 (le_of_eq ?_)⟩
  · rw [hE1]
    field_simp
  · rw [hE2]
    field_simp

/-! ### What survives of the ratio identity: the logarithmic limit

At a fixed diffusion the bracket does not close, and it cannot without the sharp prefactor. At
logarithmic scale, in the small-learning-rate limit, it closes exactly, and the identity
`escapeTime_ratio_of_kramers` derives from `hkramers` is recovered without `hkramers`. -/

/-- Multiplying by a positive constant maps the punctured right neighbourhood of the origin into
itself; this transports `KramersExitTime`'s small-noise limit from the diffusion coefficient to
the learning rate. -/
theorem tendsto_nhdsGT_mul_const {c : ℝ} (hc : 0 < c) :
    Filter.Tendsto (fun eta : ℝ => eta * c) (𝓝[>] (0 : ℝ)) (𝓝[>] (0 : ℝ)) := by
  refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
  · have h : Filter.Tendsto (fun eta : ℝ => eta * c) (𝓝 (0 : ℝ)) (𝓝 (0 * c)) :=
      (continuous_mul_const c).tendsto 0
    rw [zero_mul] at h
    exact h.mono_left nhdsWithin_le_nhds
  · filter_upwards [self_mem_nhdsWithin] with eta h
    exact mul_pos h hc

/-- **The Kramers exponent in the learning-rate variable.** At effective diffusion `c` held
fixed, `eta * log T(eta c) → 2 alpha / c` as the learning rate tends to zero. This is
`KramersExitTime.tendsto_mul_log_meanExitTime` composed with the scaling `eta ↦ eta c`, and it is
exactly `BarrierRescaling`'s `arrheniusExponent alpha eta c` multiplied by `eta`. -/
theorem tendsto_mul_log_meanExitTime_scaled {U : ℝ → ℝ} {a b x wb ws c : ℝ} (hc : 0 < c)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto (fun eta => eta * Real.log (meanExitTime (eta * c) U a b x))
      (𝓝[>] (0 : ℝ)) (𝓝 (2 * (U ws - U wb) / c)) := by
  have hbase := (tendsto_mul_log_meanExitTime hU hax hawb hwbws hwsb hxws hbarrier hmax hmin).comp
    (tendsto_nhdsGT_mul_const hc)
  have hscaled := hbase.const_mul (1 / c)
  have hval : (1 / c) * (2 * (U ws - U wb)) = 2 * (U ws - U wb) / c := by field_simp
  rw [hval] at hscaled
  refine hscaled.congr fun eta => ?_
  simp only [Function.comp_apply]
  field_simp

/-- **The log-ratio of the two exit times converges to the exponent `hkramers` predicts.**

`eta * log (T(eta sd) / T(eta sl)) → 2 alpha (1/sd - 1/sl)` as the learning rate tends to zero.
The right-hand side is exactly `eta * log (escapeTimeRatio alpha eta sd sl)`, which is
independent of `eta`; see `mul_log_escapeTimeRatio`. -/
theorem tendsto_mul_log_meanExitTime_ratio {U : ℝ → ℝ} {a b x wb ws sd sl : ℝ}
    (hsd : 0 < sd) (hsl : 0 < sl)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto
      (fun eta => eta * Real.log
        (meanExitTime (eta * sd) U a b x / meanExitTime (eta * sl) U a b x))
      (𝓝[>] (0 : ℝ)) (𝓝 (2 * (U ws - U wb) * (1 / sd - 1 / sl))) := by
  have h1 := tendsto_mul_log_meanExitTime_scaled hsd hU hax hawb hwbws hwsb hxws hbarrier hmax hmin
  have h2 := tendsto_mul_log_meanExitTime_scaled hsl hU hax hawb hwbws hwsb hxws hbarrier hmax hmin
  have h := h1.sub h2
  have hval : 2 * (U ws - U wb) / sd - 2 * (U ws - U wb) / sl
      = 2 * (U ws - U wb) * (1 / sd - 1 / sl) := by field_simp
  rw [hval] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with eta heta
  have hetapos : (0 : ℝ) < eta := heta
  have hd := meanExitTime_pos (sigma2 := eta * sd) (mul_pos hetapos hsd) hU hax hawb hwbws hwsb hxws
  have hl := meanExitTime_pos (sigma2 := eta * sl) (mul_pos hetapos hsl) hU hax hawb hwbws hwsb hxws
  rw [Real.log_div hd.ne' hl.ne']
  ring

/-- `eta * log (escapeTimeRatio alpha eta sd sl) = 2 alpha (1/sd - 1/sl)`, exactly and for every
non-zero learning rate. `BarrierRescaling`'s predicted ratio is therefore, at logarithmic scale
and after multiplication by `eta`, a constant in the learning rate. -/
theorem mul_log_escapeTimeRatio {alpha eta sd sl : ℝ} (heta : eta ≠ 0) (hsd : sd ≠ 0)
    (hsl : sl ≠ 0) :
    eta * Real.log (escapeTimeRatio alpha eta sd sl) = 2 * alpha * (1 / sd - 1 / sl) := by
  rw [escapeTimeRatio, Real.log_exp]
  field_simp

/-- **The ratio identity of `escapeTime_ratio_of_kramers`, recovered at logarithmic scale
without `hkramers`.**

The discrepancy between the true log-ratio of the two exit times and the log-ratio
`BarrierRescaling`'s `escapeTimeRatio` predicts, both scaled by the learning rate, tends to zero
as the learning rate tends to zero. Since `hkramers` yields exactly
`tau sd / tau sl = escapeTimeRatio alpha eta sd sl`, this says that the identity it postulates
holds in the small-learning-rate limit at the level of exponents, and is a theorem there rather
than a hypothesis.

It says nothing more than that. The scaling by `eta` is what makes the statement about
exponents: it discards any sub-exponential prefactor, which is precisely the quantity the sharp
Kramers relation additionally claims to pin down. At a fixed learning rate the two sides are
related only through `meanExitTime_ratio_bracket`, whose width does not shrink. -/
theorem tendsto_mul_log_ratio_sub_escapeTimeRatio {U : ℝ → ℝ} {a b x wb ws sd sl : ℝ}
    (hsd : 0 < sd) (hsl : 0 < sl)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    Filter.Tendsto
      (fun eta => eta * Real.log
          (meanExitTime (eta * sd) U a b x / meanExitTime (eta * sl) U a b x)
        - eta * Real.log (escapeTimeRatio (U ws - U wb) eta sd sl))
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have h1 := tendsto_mul_log_meanExitTime_ratio hsd hsl hU hax hawb hwbws hwsb hxws hbarrier
    hmax hmin
  have h2 : Filter.Tendsto
      (fun eta : ℝ => eta * Real.log (escapeTimeRatio (U ws - U wb) eta sd sl))
      (𝓝[>] (0 : ℝ)) (𝓝 (2 * (U ws - U wb) * (1 / sd - 1 / sl))) := by
    refine tendsto_const_nhds.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with eta heta
    exact (mul_log_escapeTimeRatio (ne_of_gt heta) hsd.ne' hsl.ne').symm
  have h := h1.sub h2
  rwa [sub_self] at h

/-! ### Non-vacuity, and why the quadratic barrier is unavailable

The theorems above inherit `KramersExitTime`'s hypothesis stack, whose two extremal clauses
require the basin bottom and the barrier top to be *interior* to `[a, b]`. That is a real
restriction on the potential, and the first theorem here records what it excludes. -/

/-- **No quadratic potential carries an interior single barrier.**

If a quadratic `U(y) = p y² + q y + r` attains its maximum on `[a, b]` at an interior point `w_s`
and its minimum on `[a, b]` at an interior point `w_b`, with `a < w_b < w_s < b`, then `p = 0`
and `q = 0`, so `U` is constant and its barrier action `U(w_s) - U(w_b)` is zero. The
single-barrier hypotheses of `KramersExitTime` are therefore unsatisfiable at any quadratic with
a positive barrier.

The mechanism is immediate: both extrema being interior forces `U'(w_b) = U'(w_s) = 0`, and a
quadratic has at most one critical point unless it is constant. The consequence is that a
witness for the hypothesis stack must have two interior critical points, hence be of degree at
least three — `KramersExitTime.paper_facing_hypotheses_satisfiable` uses the cubic
`y - y³/3` — or be transcendental, as the cosine barrier used below is. -/
theorem no_quadratic_single_barrier {p q r a b wb ws : ℝ}
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b)
    (hmax : ∀ y ∈ Set.Icc a b, p * y ^ 2 + q * y + r ≤ p * ws ^ 2 + q * ws + r)
    (hmin : ∀ z ∈ Set.Icc a b, p * wb ^ 2 + q * wb + r ≤ p * z ^ 2 + q * z + r) :
    p = 0 ∧ q = 0 ∧ p * wb ^ 2 + q * wb + r = p * ws ^ 2 + q * ws + r := by
  have hf : ∀ t : ℝ, HasDerivAt (fun y : ℝ => p * y ^ 2 + q * y + r) (2 * p * t + q) t := by
    intro t
    have h1 : HasDerivAt (fun y : ℝ => y ^ 2) (2 * t) t := by
      simpa using hasDerivAt_pow 2 t
    have h2 := ((h1.const_mul p).add ((hasDerivAt_id t).const_mul q)).add_const r
    refine h2.congr_deriv ?_
    simp only [mul_one]
    ring
  have hwsmem : Set.Icc a b ∈ 𝓝 ws := Icc_mem_nhds (hawb.trans hwbws) hwsb
  have hwbmem : Set.Icc a b ∈ 𝓝 wb := Icc_mem_nhds hawb (hwbws.trans hwsb)
  have hlmax : IsLocalMax (fun y : ℝ => p * y ^ 2 + q * y + r) ws :=
    Filter.eventually_of_mem hwsmem hmax
  have hlmin : IsLocalMin (fun y : ℝ => p * y ^ 2 + q * y + r) wb :=
    Filter.eventually_of_mem hwbmem hmin
  have e1 : 2 * p * ws + q = 0 := hlmax.hasDerivAt_eq_zero (hf ws)
  have e2 : 2 * p * wb + q = 0 := hlmin.hasDerivAt_eq_zero (hf wb)
  have hne : ws - wb ≠ 0 := sub_ne_zero_of_ne hwbws.ne'
  have h3 : 2 * p * (ws - wb) = 0 := by linarith
  have hp : p = 0 := by
    rcases mul_eq_zero.mp h3 with h | h
    · linarith
    · exact absurd h hne
  have hq : q = 0 := by rw [hp] at e1; linarith
  exact ⟨hp, hq, by rw [hp, hq]; ring⟩

/-- **The whole chain is non-vacuous, end to end.**

The cosine barrier `U(y) = -cos y` on `[a, b] = [-1, 4]`, with basin bottom `w_b = 0`, barrier
top `w_s = π`, start point `x = 0`, positivity-map slope `s = 1`, objective noise `sigma_obj = 1`,
coupling channel `1` and quadratic channel `0`, satisfies every hypothesis of
`eventually_meanExitTime_lift_lt_direct_of_channels`, and the conclusion is carried in the
statement rather than left to a proof a later edit could hollow out. The barrier action is
`U(w_s) - U(w_b) = 1 - (-1) = 2 > 0`; the two extremal clauses hold not merely on `[-1, 4]` but
globally, since `-1 ≤ -cos y ≤ 1` everywhere; and `π < 4` places the barrier top strictly inside
the interval.

This instance is independent of `KramersExitTime.paper_facing_hypotheses_satisfiable`, which
uses the cubic `y - y³/3`; by `no_quadratic_single_barrier` no quadratic could serve in either
place. -/
theorem smallLearningRate_ordering_nonvacuous :
    ∃ (U : ℝ → ℝ) (a b x wb ws s sigmaObj coupling quadratic : ℝ),
      ContinuousOn U (Set.Icc a b) ∧
      a ≤ x ∧ a < wb ∧ wb < ws ∧ ws < b ∧ x < ws ∧ U wb < U ws ∧
      (∀ y ∈ Set.Icc a b, U y ≤ U ws) ∧ (∀ z ∈ Set.Icc a b, U wb ≤ U z) ∧
      0 < directDiffusion s sigmaObj ∧ 0 ≤ coupling ∧ 0 ≤ quadratic ∧
      (0 < coupling ∨ 0 < quadratic) ∧
      directDiffusion s sigmaObj < liftDiffusion s sigmaObj coupling quadratic ∧
      (∀ᶠ eta in 𝓝[>] (0 : ℝ),
        meanExitTime (eta * liftDiffusion s sigmaObj coupling quadratic) U a b x
          < meanExitTime (eta * directDiffusion s sigmaObj) U a b x) := by
  have hUc : ContinuousOn (fun y : ℝ => -Real.cos y) (Set.Icc (-1 : ℝ) 4) :=
    Real.continuous_cos.neg.continuousOn
  have hbar : (fun y : ℝ => -Real.cos y) 0 < (fun y : ℝ => -Real.cos y) Real.pi := by
    simp [Real.cos_zero, Real.cos_pi]
  have hmax : ∀ y ∈ Set.Icc (-1 : ℝ) 4,
      (fun y : ℝ => -Real.cos y) y ≤ (fun y : ℝ => -Real.cos y) Real.pi := by
    intro y _
    simp only [Real.cos_pi]
    have := Real.neg_one_le_cos y
    linarith
  have hmin : ∀ z ∈ Set.Icc (-1 : ℝ) 4,
      (fun y : ℝ => -Real.cos y) 0 ≤ (fun y : ℝ => -Real.cos y) z := by
    intro z _
    simp only [Real.cos_zero]
    have := Real.cos_le_one z
    linarith
  have hdirect : (0 : ℝ) < directDiffusion 1 1 := by norm_num [directDiffusion]
  refine ⟨fun y => -Real.cos y, -1, 4, 0, 0, Real.pi, 1, 1, 1, 0, hUc, by norm_num, by norm_num,
    Real.pi_pos, Real.pi_lt_four, Real.pi_pos, hbar, hmax, hmin, hdirect, by norm_num,
    le_rfl, Or.inl one_pos,
    directDiffusion_lt_liftDiffusion (by norm_num) le_rfl (Or.inl one_pos), ?_⟩
  exact eventually_meanExitTime_lift_lt_direct_of_channels hdirect (by norm_num) le_rfl
    (Or.inl one_pos) hUc (by norm_num) (by norm_num) Real.pi_pos Real.pi_lt_four Real.pi_pos
    hbar hmax hmin

end IcnnLift
