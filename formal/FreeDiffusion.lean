import Mathlib

/-!
# The driftless free-diffusion exit time, and the crossover out of the Arrhenius regime

The paper's `cor:fpt` states the Kramers (Arrhenius) mean first-passage time
`E[τ] ≍ exp(2α/σ_eff²)` under assumption (A5), which asks that the effective noise be small
relative to the barrier. `rem:fpt-scope` and the closing paragraph of `app:proof-fpt` then say
what happens when (A5) fails: "once the unattenuated `σ_Jac²` grows comparable to the barrier
`2α` … the escape leaves the Arrhenius regime and is governed instead by free diffusion over the
escape distance, with mean first-passage time the polynomial `Θ(σ_Jac^{-2})` of the standard
Brownian hitting-time identity." The design note records the same item as (FW-5). This file
machine-checks that free-diffusion law and the crossover statement attached to it.

**The modelling inputs.** Two are carried, and the `## Honesty` section states both. The second
is recorded here at the outset because it is the one a reader is likeliest to miss: the
bias-channel SDE of `app:proof-fpt` is `dw̃ = -L̃'(w̃) dt + σ_eff(w̃) dB_t`, with a drift and a
state-dependent diffusion, while the process treated below has neither. Dropping the drift is
the paper's own step past the crossover — `rem:fpt-scope` says the escape there "is governed
instead by free diffusion", and the probe of `sec:exp-e6` is drift-free by construction — and
freezing the diffusion across the escape region is its Step 4 together with (A2); neither is
derived here. Everything below is therefore a statement about the driftless, constant-diffusion
idealization of the escape, which is exactly the object `rem:fpt-scope` invokes, and not about
the SDE of `app:proof-fpt` itself.

**The first modelling input, stated once and carried as a definition rather than an axiom.** For the
driftless scalar Itô process `dX = σ dB` started at `x`, Dynkin's formula applied to the exit
time `τ` from an interval `(a,b)` gives that `T(x) = E_x[τ]` solves the Dirichlet problem for the
generator `A = (σ²/2) d²/dx²`, namely `(σ²/2) T'' = -1` on `(a,b)` with `T(a) = T(b) = 0`. This
mathlib has no Itô integral, no stochastic differential equation, no diffusion generator and no
exit time — the whole of stochastic analysis relevant here is absent, as the API audit records
(`Mathlib/Probability/BrownianMotion` supplies only the *predicates* `IsPreBrownianReal` and
`IsBrownianReal`, with no existence theorem, and a case-insensitive search of the tree for
`exit time`, `first passage`, `Fokker`, `Kramers`, `Ito` returns nothing). So the passage from
the SDE to the boundary value problem cannot be a theorem here. It is encoded instead as the
**predicate** `IsFreeExitTime`, a property of a candidate function `T` together with witnesses
for its first and second derivatives; no statement below asserts that the probabilistic mean
exit time satisfies it, and the paper-facing theorems
(`isFreeExitTime_two_sided`, `isFreeExitTime_midpoint`, `isFreeExitTime_isTheta_inv`, and the
decomposed-variable forms `isFreeExitTime_two_sided_decomposed`,
`isFreeExitTime_isTheta_inv_decomposed`) carry it as their single named hypothesis, in the
pattern of `KramersExitTime.lean`. Discharging that
link would require, in order: the Itô integral, strong
existence and uniqueness for `dX = σ dB` (trivial as an SDE but not available), the exit time as
a stopping time, and Dynkin's formula for `C²` functions of the process. That is a substantial
development in its own right and none of it exists in this mathlib.

Everything downstream of the predicate is proved outright: the exit-time results carry no
hypothesis beyond `0 < σ²`, `a < b` and the position of the starting point — plus, in the
paper-facing forms, the boundary value problem itself as the single named hypothesis, and, in
the decomposed-variable forms, the explicit interval `[0, K]` confining the attenuated channel —
and the crossover results carry only positivity and explicit regime thresholds, each named in
its own statement.
In particular the closed form is shown to be not merely *a* solution but
*the* solution: `eq_freeExitTime_of_isFreeExitTime` proves uniqueness in the full Dirichlet
class — continuous on `[a,b]`, twice differentiable on `(a,b)`, and nothing asked at the
endpoints — so the polynomial law is forced by the boundary value problem rather than fitted
to it. The regularity is stated that way deliberately: the mean exit
time read as a function on the whole line is `0` outside `(a,b)`, so its one-sided derivatives
at `a` and at `b` disagree, and a predicate demanding a two-sided derivative there would have
been false of the object it models while asserting uniqueness in a smaller class.

**The escape domain has to be bounded, and the probe of `sec:exp-e6` is not.** The polynomial
law is a statement about escape from a *bounded* interval, and that is not a convenience of the
formalization. `eq_quadratic_of_generator` shows that on any convex open set every solution of
`(σ²/2) T'' = -1` is the downward parabola of leading coefficient `-1/σ²` fixed by its value and
slope at one base point; a downward parabola is eventually negative, so
`no_nonneg_solution_on_Iio` and `no_nonneg_solution_on_Ioi` conclude that the driftless problem
has **no** non-negative solution on a half-line. Probabilistically this is the familiar fact
that driftless Brownian motion started on one side of a single absorbing level has infinite mean
hitting time. The diffusive probe of `sec:exp-e6` is exactly that configuration: the SDE is
drift-free, absorption is at the single shoulder level `w̃_s`, the start is `w̃_0 < w̃_s`, and
nothing bounds the process from below. So the idealized envelope
`E[τ] ∝ (w̃_s - w̃_0)²/σ_Jac²` that `fig:e6` draws is the bounded-domain value — which is what
the "standard Brownian hitting-time identity" of `rem:fpt-scope` and of the closing paragraph of
`app:proof-fpt` supplies — and not the mean first-passage time of the probe as literally
specified. This file formalizes the bounded-domain statement, which is the one that is true, and
records the gap rather than closing it: closing it would require a second boundary that the
probe does not have, and the two-sided reading in which it is supplied, with the escape distance
`L` as the half-width and the same value `L²/σ²`, is `freeExitTime_symmetric_interval`. Nothing
here contradicts the paper's `Θ(σ_Jac^{-2})`; it locates the escape domain that claim needs.

**On "the two laws cross exactly once".** The brief for this file asked for a proof that the
free-diffusion time `(b-a)²/(4σ²)` is smaller than the Arrhenius time `exp(2α/σ²)` for large
`σ²` and larger for small `σ²`, so that the two cross exactly once. The second half of that is
false, and the file proves the true statement instead. Writing `u = 1/σ²`, the comparison is
between the line `(b-a)²u/4` and the exponential `exp(2αu)`, which starts at `1 > 0` and
eventually dominates any line; so the free-diffusion expression is *below* the Arrhenius
expression in both extremes, and if it ever rises strictly above, it does so on a bounded
intermediate window and crosses **at least twice**, not once. How many crossings there are in
general is not determined here, and "at least" is meant: in the boundary case `8eα = (b-a)²` of
the criterion below the free-diffusion expression never rises strictly above the Arrhenius one
at all — that comparison is exactly `exp_one_mul_le_exp`, `e·y ≤ exp y` — while the two are
equal at `σ² = 2α`. `crossing_not_unique` exhibits a parameter set — barrier
`α = 1/2`, interval `(0,4)` — at which the difference of the two laws is machine-checked to
change sign between `σ² = 1/8` and `σ² = 1`, and again between `σ² = 1` and `σ² = 5`; and
`exists_two_crossings` turns those sign changes into two distinct diffusion levels at which the
two laws are equal, by the intermediate value theorem. The refutation is therefore itself a
result of this file rather than a remark about one. What is proved here in place of the false
claim is sharper than the claim: both extreme regimes are
established with explicit thresholds (`freeDiffusionTime_lt_arrheniusTime_of_large_variance`,
`freeDiffusionTime_lt_arrheniusTime_of_small_variance`), any crossing is confined to the
explicit window `8α²/(b-a)² < σ² ≤ (b-a)²/4` (`crossing_confined`), and such a window is
non-empty only when `32α² < (b-a)⁴` (`crossing_requires_small_barrier`), a necessary condition
on the barrier for the two laws to meet at all. `exists_crossing_iff` then makes that criterion
sharp in both directions: the free-diffusion time is *at least* the Arrhenius time at some
positive diffusion **if and only if** `8eα ≤ (b-a)²`, and when it is, the crossover diffusion
`σ² = 2α` already witnesses that inequality — it is not, in general, a level at which the two
are *equal*, since there the two laws take the values `(b-a)²/(8α)` and `e`. The equality form
is stated separately and holds under the same criterion: `exists_equal_crossing_iff` produces,
by the intermediate value theorem between `σ² = 2α` and the large-noise regime, a diffusion
level at which the two closed forms coincide exactly.
`quartic_lt_of_crossing_criterion` and `crossing_criterion_of_large_barrier` machine-check the
comparison between that sharp criterion and the quartic necessary condition, which the
docstrings of `crossing_requires_small_barrier` and `no_crossing_of_large_barrier` previously
only asserted. `exists_arrheniusTime_le_freeDiffusionTime` exhibits parameters satisfying the
hypothesis of `crossing_confined`, so the confinement results are not vacuous.

The *bona fide* "exactly once" statement in the neighbourhood of the paper's remark is about the
Arrhenius **exponent**, not about the two times: `2α/σ²` is strictly antitone in `σ²` and passes
the value one exactly at `σ² = 2α`, so `{σ² > 0 : 2α/σ² > 1} = (0, 2α)` exactly
(`metastable_window`, `metastable_window_eq_Ioo`). That single crossing point `σ² = 2α` is the
formal content of the paper's "once `σ_Jac²` grows comparable to the barrier `2α`", and it is
what `rem:fpt-scope` currently gestures at without a statement. That the two regimes are
genuinely different laws — and not one law written two ways — is
`arrheniusTime_not_isBigO_inv`: the Arrhenius time is not `O(σ^{-2})` as `σ² ↓ 0`, so no
constant multiple of the free-diffusion law bounds it in the metastable regime.

**What `rem:fpt-scope` does assert, and where it is proved.** The remark does not compare the
two closed forms; it says that past the crossover the escape is *governed by* the free-diffusion
law — the Kramers asymptotic is abandoned there, not reinterpreted (the remark reads the
diffusive probe of `sec:exp-e6` "at the level of the SDE rather than its Kramers asymptotics").
The formalization of that assertion is the boundary value problem and its consequences:
granting Dynkin's formula for the driftless process, the mean exit time itself obeys the
explicit two-sided polynomial bound and is `Θ((σ²)⁻¹)`
(`isFreeExitTime_two_sided`, `isFreeExitTime_isTheta_inv`). On top of that this file adds a
robustness check that is its own construction and **not** an assertion of the paper: even
retaining the Kramers form `P·exp(2α/σ²)` past the crossover, the exponential factor lies in
`(1, e]` once `σ² ≥ 2α`, so the time is squeezed between `P` and `e·P` for an arbitrary
non-negative prefactor `P`
(`arrheniusTime_le_exp_one_of_crossover`, `kramers_between_prefactor_and_exp_one_of_crossover`),
and with the free-diffusion time in the role of `P` the composite is `Θ((σ²)⁻¹)` as `σ² → ∞`,
with explicit constants `(b-a)²/4` and `e(b-a)²/4` (`kramersTime_isTheta_inv_atTop`). The
polynomial law thus survives whichever way the crossover is modelled — not because the two
closed forms cross but because the exponential has become a bounded multiplier; the paper makes
no such prefactor decomposition of the post-crossover time, and the `## Honesty` section records
that identification as this file's own.
Finally the paper puts its threshold on the *unattenuated* `σ_Jac²` and not on the total
effective variance `σ_s²σ_obj² + σ_Jac²`; `lift_leaves_metastable_window` states the crossover in
those decomposed variables, and `direct_metastable_lift_past_crossover` is the per-method scoping
of (FW-5): with the attenuated channel alone below the barrier and the two channels together
above it, the direct baseline is inside the metastable window of (A5) and the lift is outside.
The polynomial law itself is decomposed the same way. `rem:fpt-scope` and the closing paragraph
of `app:proof-fpt` write the law as `Θ(σ_Jac^{-2})` — in the Jacobian channel alone — while the
diffusion of the bias-channel SDE is the total `σ_eff²`; the paper bridges the two by the
shoulder collapse "`σ_eff² ≈ σ_Jac²`" of its Step 4, a leading-order step this file refuses to
absorb. Instead `freeExitTime_two_sided_decomposed` and `freeExitTime_isTheta_inv_decomposed`
(with the solution forms `isFreeExitTime_two_sided_decomposed`,
`isFreeExitTime_isTheta_inv_decomposed`) carry the attenuated channel as a summand bounded in
`[0, K]` and prove the sandwich and the `Θ` in `σ_Jac²` itself: past any threshold
`0 < s₀ ≤ σ_Jac²` the exit time lies between `δ²·(s₀/(s₀+K))·σ_Jac⁻²` and `((b-a)²/4)·σ_Jac⁻²`,
and as `σ_Jac² → ∞` it is `Θ(σ_Jac⁻²)` even when the attenuated channel varies arbitrarily
within its bound. At `K = 0` the undecomposed constants are recovered exactly.

## Results
* `IsFreeExitTime` — the Dirichlet problem `(σ²/2) T'' = -1`, `T(a) = T(b) = 0` for the
  driftless generator, in the Dirichlet class (continuous on `[a,b]`, twice differentiable on
  `(a,b)`), with explicit first- and second-derivative witnesses.
* `hasDerivAt_freeExitTime`, `hasDerivAt_freeExitTimeDeriv` — the two space derivatives of the
  closed form, computed exactly.
* `freeExitTime_isFreeExitTime` — `T(x) = (x-a)(b-x)/σ²` solves it, for every `σ² ≠ 0`.
* `exists_isFreeExitTime_family` — the *family* form of that witness, one solution at each
  positive diffusion, which is what the paper-facing `Θ` theorems quantify over and what the
  single-level witness does not by itself exhibit.
* `eq_of_hasDerivWithinAt_interior_zero` — the constancy lemma the uniqueness argument needs,
  assembled from mathlib's separate monotone and antitone halves.
* `eq_freeExitTime_of_isFreeExitTime` — and it is the only solution on `[a,b]`.
* `eq_quadratic_of_generator` — on any convex open set every solution of `(σ²/2) T'' = -1` is
  the downward parabola of leading coefficient `-1/σ²` fixed by its value and slope at a base
  point.
* `exists_neg_of_generator_unbounded`, `no_nonneg_solution_on_Iio`, `no_nonneg_solution_on_Ioi`
  — hence the driftless problem has **no** non-negative solution on a half-line: the escape
  domain of the polynomial law has to be bounded, and the single-absorbing-level configuration
  of the `sec:exp-e6` probe is not one on which a finite mean exit time solves this problem.
* `freeExitTime_eq_mul_inv` — the scaling law `T(x) = (x-a)(b-x) · (σ²)⁻¹`.
* `freeExitTime_isTheta_inv`, `freeExitTime_isTheta_inv_sq` — hence `T =Θ[l] (σ²)⁻¹`, resp.
  `Θ(σ^{-2})`, on every filter, for `x` strictly inside `(a,b)`.
* `freeExitTime_two_sided` — the explicit two-sided bound the `Θ` claim abbreviates, uniform
  over the sub-interval `[a+δ, b-δ]`.
* `freeExitTime_midpoint`, `freeExitTime_le_midpoint` — the maximum is `(b-a)²/(4σ²)`, attained
  at the midpoint: the `L²/σ²` form the paper quotes.
* `freeExitTime_symmetric_interval` — the closed form equals `L²/σ²` from the centre of an
  interval of half-width `L`, which is the value of the envelope drawn in `fig:e6`.
* `isFreeExitTime_two_sided`, `isFreeExitTime_midpoint`, `isFreeExitTime_isTheta_inv` — the
  same laws stated of **any** solution of the boundary value problem rather than of the closed
  form: with the Dirichlet problem as the single named hypothesis, the solution obeys the
  two-sided polynomial bound, takes the value `(b-a)²/(4σ²)` at the midpoint, and — as a family
  over the diffusion — is `Θ((σ²)⁻¹)` along every filter on which the diffusion is eventually
  positive. These are the paper-facing forms: granting Dynkin's formula, they are statements
  about the mean exit time itself.
* `freeExitTime_two_sided_decomposed`, `freeExitTime_isTheta_inv_decomposed`,
  `isFreeExitTime_two_sided_decomposed`, `isFreeExitTime_isTheta_inv_decomposed` — the same
  polynomial law in the paper's decomposed variables: with the attenuated channel confined to
  `[0, K]`, the exit time is two-sidedly a constant multiple of `σ_Jac⁻²` past any positive
  threshold on `σ_Jac²`, and `Θ(σ_Jac⁻²)` as `σ_Jac² → ∞` — the literal variable of
  `rem:fpt-scope`'s `Θ(σ_Jac^{-2})`, with the shoulder collapse `σ_eff² ≈ σ_Jac²` carried as a
  bound instead of assumed.
* `freeExitTime_pos`, `freeExitTime_strictAntiOn_variance`, `one_lt_arrheniusTime`,
  `arrheniusTime_strictAntiOn`, `arrheniusTime_at_crossover` — elementary structure of the two
  laws: both are strictly decreasing in the diffusion, the Arrhenius factor exceeds one, and it
  equals `e` exactly at the crossover `σ² = 2α`.
* `metastable_window`, `metastable_window_eq_Ioo` — the Arrhenius exponent exceeds one exactly
  on `σ² ∈ (0, 2α)`: the crossover condition, crossed exactly once, at `σ² = 2α`.
* `lift_leaves_metastable_window`, `direct_metastable_lift_past_crossover` — the same crossover
  in the paper's decomposed variables `σ_eff² = σ_s²σ_obj² + σ_Jac²`: the unattenuated channel
  alone reaching `2α` already drops the exponent to at most one, and with the attenuated channel
  alone below the barrier the direct baseline stays inside (A5) while the lift leaves it.
* `arrheniusTime_le_exp_one_of_crossover`, `kramers_between_prefactor_and_exp_one_of_crossover`
  — past `σ² ≥ 2α` the Arrhenius factor lies in `(1, e]`, so a Kramers-form first-passage time
  `P·exp(2α/σ²)` is squeezed between `P` and `e·P` for every prefactor `P ≥ 0`.
* `kramersTime_isTheta_inv_atTop` — hence, with the free-diffusion prefactor the paper names,
  the first-passage time is `Θ((σ²)⁻¹)` as `σ² → ∞`: the polynomial law of `rem:fpt-scope`.
* `freeDiffusionTime_lt_arrheniusTime_of_large_variance` — past `σ² > (b-a)²/4` the
  free-diffusion time is below one and hence below the Arrhenius time.
* `freeDiffusionTime_lt_arrheniusTime_of_small_variance` and its `c`-scaled form
  `const_mul_freeDiffusionTime_lt_arrheniusTime_of_small_variance` — below `σ² ≤ 8α²/(b-a)²` the
  Arrhenius time is the larger, and below `σ² ≤ 8α²/(c(b-a)²)` it exceeds `c` times the
  free-diffusion time, for every constant `c`.
* `crossing_confined`, `crossing_requires_small_barrier`, `no_crossing_of_large_barrier` — any
  crossing is confined to an explicit bounded window, non-empty only if `32α² < (b-a)⁴`; for a
  larger barrier the Arrhenius time is strictly the larger at every noise level.
* `exp_one_mul_le_exp`, `exists_crossing_iff` — the sharp form: the free-diffusion time is at
  least the Arrhenius time at some positive diffusion if and only if `8eα ≤ (b-a)²`, and then
  `σ² = 2α` already witnesses that inequality (not, in general, an equality).
* `continuousOn_freeDiffusionTime_sub_arrheniusTime`, `exists_equal_crossing_iff` — the equality
  form under the same criterion: the two closed forms are genuinely *equal* at some positive
  diffusion if and only if `8eα ≤ (b-a)²`, by the intermediate value theorem.
* `quartic_lt_of_crossing_criterion`, `crossing_criterion_of_large_barrier` — the comparison of
  the sharp criterion with the quartic necessary condition, in both directions.
* `exists_arrheniusTime_le_freeDiffusionTime` — parameters satisfying the hypothesis of
  `crossing_confined`, so that family of results is not vacuous.
* `crossing_not_unique`, `exists_two_crossings` — an explicit parameter set at which the
  difference of the two laws changes sign twice, and, by the intermediate value theorem, two
  crossings, one in `(1/8, 1)` and one in `(1, 5)`, at which the two laws are equal: the
  "exactly once" reading is refuted by exhibited crossings, not by an argument.
* `arrheniusTime_not_isBigO_inv` — the Arrhenius law is not of the polynomial free-diffusion
  type as `σ² ↓ 0`; the crossover is a genuine change of law.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

**Two modelling inputs are carried, not one.** The first is the one named above: that the
probabilistic mean exit time of `dX = σ dB` from `(a,b)` satisfies `IsFreeExitTime`. It is
carried as a *definition* — quantified
over by the solution and uniqueness theorems, and standing as the single named hypothesis of the
paper-facing forms `isFreeExitTime_two_sided`, `isFreeExitTime_midpoint`,
`isFreeExitTime_isTheta_inv` and their decomposed-variable forms
`isFreeExitTime_two_sided_decomposed`, `isFreeExitTime_isTheta_inv_decomposed` — never as an
axiom, and never as a hypothesis smuggling in the
conclusion: the conclusion `T(x) = (x-a)(b-x)/σ²` is derived from the boundary value problem,
not assumed, and the hypothesis is satisfiable (`freeExitTime_isFreeExitTime` is a single-level
witness and `exists_isFreeExitTime_family` the family witness the `Θ` theorems need, so the
paper-facing forms are not vacuous). Discharging it needs Itô calculus and Dynkin's formula,
neither of which exists in mathlib v4.31.0.

The second is the reduction that produces `dX = σ dB` in the first place, and it is *not* a
consequence of the first. The bias-channel SDE of `app:proof-fpt` is
`dw̃ = -L̃'(w̃) dt + σ_eff(w̃) dB_t`: it has a drift and a state-dependent diffusion, and the
object formalized here has neither. Dropping the drift is the paper's own step — `rem:fpt-scope`
says the post-crossover escape "is governed instead by free diffusion", and the probe of
`sec:exp-e6` is drift-free by construction — and freezing `σ_eff` across the escape region is
the paper's Step 4 together with the single-prefactor idealization (A2). Neither is derived
here, and neither could be without the stochastic analysis the first input already needs; the
consequence is that the results of this file speak about the driftless, constant-diffusion
idealization of the escape and not about the SDE of `app:proof-fpt` itself. What this file adds
on that point is the bounded-domain caveat recorded above: once the drift is dropped, a bounded
escape domain stops being a modelling convenience and becomes a necessity, and
`no_nonneg_solution_on_Iio` proves it — on the single-absorbing-level, unbounded-below domain of
the `sec:exp-e6` probe the driftless boundary value problem has no non-negative solution at all.

One further modelling choice appears, in one place only, and is confined to how one theorem is
read rather than to a hypothesis: `kramersTime_isTheta_inv_atTop` studies the product of the
free-diffusion time with `exp(2α/σ²)`. Reading that composite as "the first-passage time past
the crossover" identifies the sub-exponential prefactor of the Kramers law with the
free-diffusion time rather than with the Eyring–Kramers curvature prefactor. That
identification is made by this file, as a bridge between the two regimes; it is **not** made by
the paper, which abandons the Kramers form past the crossover rather than reinterpreting it,
and it is not derived from anything. The theorem itself is a statement about the composite as
written, and the prefactor-free version of the same fact,
`kramers_between_prefactor_and_exp_one_of_crossover`, quantifies over an arbitrary `P ≥ 0` and
so depends on no such choice.

No other hypothesis is carried. In particular there is no "to leading order" step anywhere in
this file: the closed form is exact, the maximum is exact, and the two-sided bound is an
inequality between explicit constants rather than an asymptotic equivalence. The passage from
the total effective variance to the paper's `σ_Jac²` is likewise not taken to leading order:
the decomposed-variable theorems replace the paper's "`σ_eff² ≈ σ_Jac²`" by the boundedness
side condition `0 ≤ σ_s²σ_obj² ≤ K`, carried through to explicit constants, which is a regime
hypothesis of the same kind as the positivity and threshold conditions and not a new modelling
input. Where a family of
results rests on a hypothesis whose satisfiability is not obvious — the existence of a diffusion
level at which the Arrhenius time falls below the free-diffusion time — a witness is exhibited
(`exists_arrheniusTime_le_freeDiffusionTime`), so that family is demonstrably not vacuous.
-/

open Set Filter Asymptotics
open scoped Topology

namespace IcnnLift

/-! ### The driftless exit-time boundary value problem -/

/-- The Dirichlet problem solved by the mean exit time of the driftless scalar diffusion
`dX = σ dB` from the interval `(a,b)`.

Dynkin's formula applied to `τ = inf {t : X_t ∉ (a,b)}` turns `T(x) = E_x[τ]` into the boundary
value problem for the generator `A = (σ²/2) d²/dx²`,
`(σ²/2) T'' = -1` on `(a,b)`, `T(a) = T(b) = 0`.
mathlib v4.31.0 has no Itô integral, no stochastic differential equation, no diffusion generator
and no exit time, so that passage cannot be a theorem here; it is the one modelling input of this
file and is recorded as this predicate. `T'` and `T''` are carried explicitly as derivative
witnesses rather than through `deriv`, so that the regularity actually used is visible in the
statement. `σsq` denotes `σ²` throughout; the diffusion enters only through its square.

The regularity demanded is exactly that of the Dirichlet problem — continuity up to the
absorbing boundary, two derivatives in the interior — and no more. In particular the predicate
does **not** ask `T` to be differentiable *at* `a` or at `b`, and that restraint is not a
convenience: the probabilistic mean exit time, read as a function on the whole line, is
`E_x[τ] = (x-a)(b-x)/σ²` inside `(a,b)` and `0` outside, which is continuous at the endpoints
but whose one-sided derivatives there disagree: at `a` they are `0` from the left and
`(b-a)/σ²` from the right, and at `b` they are `-(b-a)/σ²` and `0`. A predicate demanding a
two-sided derivative at `a` and `b` would therefore be false of the very object it models, and
would at the same time shrink the class in which the uniqueness theorem below asserts anything.
Both defects are avoided by asking for continuity on `[a,b]` and differentiability on `(a,b)`. -/
structure IsFreeExitTime (σsq a b : ℝ) (T T' T'' : ℝ → ℝ) : Prop where
  /-- `T` is continuous up to and including the absorbing boundary. -/
  continuousOn : ContinuousOn T (Set.Icc a b)
  /-- `T` is differentiable on the open interval, with derivative `T'`. -/
  hasDerivAt_interior : ∀ x ∈ Set.Ioo a b, HasDerivAt T (T' x) x
  /-- `T'` is differentiable on the open interval, with derivative `T''`. -/
  hasDerivAt_deriv : ∀ x ∈ Set.Ioo a b, HasDerivAt T' (T'' x) x
  /-- The generator equation `(σ²/2) T'' = -1` on the interior. -/
  generator : ∀ x ∈ Set.Ioo a b, σsq / 2 * T'' x = -1
  /-- Absorption at the left endpoint. -/
  boundary_left : T a = 0
  /-- Absorption at the right endpoint. -/
  boundary_right : T b = 0

/-- The closed-form driftless mean exit time from `(a,b)`, `T(x) = (x-a)(b-x)/σ²`. -/
noncomputable def freeExitTime (σsq a b x : ℝ) : ℝ := (x - a) * (b - x) / σsq

/-- The first derivative of `freeExitTime` in the space variable, `T'(x) = (a+b-2x)/σ²`. -/
noncomputable def freeExitTimeDeriv (σsq a b x : ℝ) : ℝ := (a + b - 2 * x) / σsq

/-- `freeExitTimeDeriv` is the space derivative of `freeExitTime`. -/
theorem hasDerivAt_freeExitTime (σsq a b x : ℝ) :
    HasDerivAt (freeExitTime σsq a b) (freeExitTimeDeriv σsq a b x) x := by
  have h1 : HasDerivAt (fun y : ℝ => y - a) 1 x := (hasDerivAt_id x).sub_const a
  have h2 : HasDerivAt (fun y : ℝ => b - y) (0 - 1) x :=
    (hasDerivAt_const x b).sub (hasDerivAt_id x)
  have hm : HasDerivAt (fun y : ℝ => (y - a) * (b - y)) (1 * (b - x) + (x - a) * (0 - 1)) x :=
    h1.mul h2
  have h3 : HasDerivAt (fun y : ℝ => (y - a) * (b - y) / σsq)
      ((1 * (b - x) + (x - a) * (0 - 1)) / σsq) x := hm.div_const σsq
  have hfun : freeExitTime σsq a b = fun y : ℝ => (y - a) * (b - y) / σsq := rfl
  rw [hfun]
  convert h3 using 1
  simp only [freeExitTimeDeriv]
  ring

/-- The second space derivative of the closed form is the constant `-2/σ²`. -/
theorem hasDerivAt_freeExitTimeDeriv (σsq a b x : ℝ) :
    HasDerivAt (freeExitTimeDeriv σsq a b) (-2 / σsq) x := by
  have h1 : HasDerivAt (fun y : ℝ => a + b - 2 * y) (0 - 2 * 1) x :=
    (hasDerivAt_const x (a + b)).sub ((hasDerivAt_id x).const_mul 2)
  have h2 : HasDerivAt (fun y : ℝ => (a + b - 2 * y) / σsq) ((0 - 2 * 1) / σsq) x :=
    h1.div_const σsq
  have hfun : freeExitTimeDeriv σsq a b = fun y : ℝ => (a + b - 2 * y) / σsq := rfl
  rw [hfun]
  convert h2 using 1
  ring

/-- **The closed form solves the boundary value problem.** For every non-zero `σ²`,
`T(x) = (x-a)(b-x)/σ²` satisfies `(σ²/2) T'' = -1` on `(a,b)` together with both absorbing
boundary conditions. This is the exact statement, with no leading-order truncation. -/
theorem freeExitTime_isFreeExitTime {σsq : ℝ} (hσ : σsq ≠ 0) (a b : ℝ) :
    IsFreeExitTime σsq a b (freeExitTime σsq a b) (freeExitTimeDeriv σsq a b)
      (fun _ => -2 / σsq) where
  continuousOn := fun x _ => ((hasDerivAt_freeExitTime σsq a b x).continuousAt).continuousWithinAt
  hasDerivAt_interior := fun x _ => hasDerivAt_freeExitTime σsq a b x
  hasDerivAt_deriv := fun x _ => hasDerivAt_freeExitTimeDeriv σsq a b x
  generator := by
    intro x _
    field_simp
  boundary_left := by simp [freeExitTime]
  boundary_right := by simp [freeExitTime]

/-- **The family hypothesis of the paper-facing `Θ` theorems is satisfiable.**
`isFreeExitTime_isTheta_inv` and `isFreeExitTime_isTheta_inv_decomposed` quantify over a whole
*family* of solutions, one at each diffusion level, which the single-level witness
`freeExitTime_isFreeExitTime` does not by itself exhibit. The closed form supplies the family:
`σ² ↦ freeExitTime σ² a b` solves the boundary value problem at every positive diffusion, so
those two theorems are not vacuous. -/
theorem exists_isFreeExitTime_family (a b : ℝ) :
    ∃ T T' T'' : ℝ → ℝ → ℝ, ∀ σsq : ℝ, 0 < σsq →
      IsFreeExitTime σsq a b (T σsq) (T' σsq) (T'' σsq) :=
  ⟨fun σsq => freeExitTime σsq a b, fun σsq => freeExitTimeDeriv σsq a b,
    fun σsq _ => -2 / σsq, fun _ hσ => freeExitTime_isFreeExitTime (ne_of_gt hσ) a b⟩

/-! ### Uniqueness of the solution -/

/-- A real function that is continuous on a convex set and has vanishing derivative on its
interior is constant there. mathlib names the monotone and antitone halves separately
(`monotoneOn_of_hasDerivWithinAt_nonneg`, `antitoneOn_of_hasDerivWithinAt_nonpos`); constancy is
their conjunction. -/
theorem eq_of_hasDerivWithinAt_interior_zero {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f 0 (interior D) x)
    {x y : ℝ} (hx : x ∈ D) (hy : y ∈ D) : f x = f y := by
  have hmono : MonotoneOn f D :=
    monotoneOn_of_hasDerivWithinAt_nonneg hD hf (f' := fun _ => (0 : ℝ)) hf' fun _ _ => le_rfl
  have hanti : AntitoneOn f D :=
    antitoneOn_of_hasDerivWithinAt_nonpos hD hf (f' := fun _ => (0 : ℝ)) hf' fun _ _ => le_rfl
  rcases le_total x y with h | h
  · exact le_antisymm (hmono hx hy h) (hanti hx hy h)
  · exact le_antisymm (hanti hy hx h) (hmono hy hx h)

/-- **Uniqueness.** Any solution of the driftless exit-time boundary value problem agrees with
the closed form on `[a,b]`. Hence the polynomial law is forced by the boundary value problem and
is not one solution among many: two solutions differ by a function with vanishing second
derivative on `(a,b)`, hence by an affine function, and an affine function vanishing at both
endpoints is zero. The solution class is the full Dirichlet class of `IsFreeExitTime` —
continuous on `[a,b]`, twice differentiable on `(a,b)` — and in particular no differentiability
is asked at the endpoints, so the theorem applies to the mean exit time extended by zero outside
the interval, whose one-sided derivatives at `a` and at `b` disagree. -/
theorem eq_freeExitTime_of_isFreeExitTime {σsq a b : ℝ} (hσ : 0 < σsq) (hab : a < b)
    {T T' T'' : ℝ → ℝ} (h : IsFreeExitTime σsq a b T T' T'') {x : ℝ} (hx : x ∈ Set.Icc a b) :
    T x = freeExitTime σsq a b x := by
  have hσ' : σsq ≠ 0 := ne_of_gt hσ
  -- the second derivative is pinned by the generator equation
  have hT'' : ∀ y ∈ Set.Ioo a b, T'' y = -2 / σsq := by
    intro y hy
    rw [eq_div_iff hσ']
    have := h.generator y hy
    linarith
  -- the difference of the two first derivatives has vanishing derivative on the interior
  have hu' : ∀ y ∈ Set.Ioo a b,
      HasDerivAt (fun z => T' z - freeExitTimeDeriv σsq a b z) 0 y := by
    intro y hy
    have := (h.hasDerivAt_deriv y hy).sub (hasDerivAt_freeExitTimeDeriv σsq a b y)
    rwa [hT'' y hy, sub_self] at this
  -- hence it is constant on the interior
  have hu'const : ∀ y ∈ Set.Ioo a b, ∀ z ∈ Set.Ioo a b,
      T' y - freeExitTimeDeriv σsq a b y = T' z - freeExitTimeDeriv σsq a b z := by
    intro y hy z hz
    refine eq_of_hasDerivWithinAt_interior_zero
      (f := fun w => T' w - freeExitTimeDeriv σsq a b w) (convex_Ioo a b) ?_ ?_ hy hz
    · exact fun w hw => ((hu' w hw).continuousAt).continuousWithinAt
    · rw [interior_Ioo]
      exact fun w hw => (hu' w hw).hasDerivWithinAt
  set c : ℝ := (a + b) / 2 with hc
  have hcmem : c ∈ Set.Ioo a b := by
    constructor <;> · rw [hc]; linarith
  set k : ℝ := T' c - freeExitTimeDeriv σsq a b c with hk
  -- the difference of the two solutions, corrected by the affine part
  set v : ℝ → ℝ := fun z => (T z - freeExitTime σsq a b z) - k * z with hv
  have hvcont : ContinuousOn v (Set.Icc a b) := by
    intro y hy
    have h1 : ContinuousWithinAt (fun z => T z - freeExitTime σsq a b z) (Set.Icc a b) y :=
      (h.continuousOn y hy).sub
        ((hasDerivAt_freeExitTime σsq a b y).continuousAt).continuousWithinAt
    have h2 : ContinuousWithinAt (fun z : ℝ => k * z) (Set.Icc a b) y :=
      ((continuous_const.mul continuous_id).continuousAt).continuousWithinAt
    exact h1.sub h2
  have hv' : ∀ y ∈ interior (Set.Icc a b), HasDerivWithinAt v 0 (interior (Set.Icc a b)) y := by
    rw [interior_Icc]
    intro y hy
    have hd : HasDerivAt (fun z => T z - freeExitTime σsq a b z)
        (T' y - freeExitTimeDeriv σsq a b y) y :=
      (h.hasDerivAt_interior y hy).sub (hasDerivAt_freeExitTime σsq a b y)
    have hlin : HasDerivAt (fun z : ℝ => k * z) k y := by
      simpa using (hasDerivAt_id y).const_mul k
    have := hd.sub hlin
    rw [hu'const y hy c hcmem, ← hk, sub_self] at this
    exact this.hasDerivWithinAt
  have hvconst : ∀ y ∈ Set.Icc a b, ∀ z ∈ Set.Icc a b, v y = v z := fun y hy z hz =>
    eq_of_hasDerivWithinAt_interior_zero (convex_Icc a b) hvcont hv' hy hz
  have hamem : a ∈ Set.Icc a b := ⟨le_rfl, hab.le⟩
  have hbmem : b ∈ Set.Icc a b := ⟨hab.le, le_rfl⟩
  -- the boundary conditions force the affine slope to vanish
  have hva : v a = -(k * a) := by
    rw [hv]
    simp [h.boundary_left, freeExitTime]
  have hvb : v b = -(k * b) := by
    rw [hv]
    simp [h.boundary_right, freeExitTime]
  have hk0 : k = 0 := by
    have := hvconst a hamem b hbmem
    rw [hva, hvb] at this
    have hba : b - a ≠ 0 := sub_ne_zero_of_ne (ne_of_gt hab)
    have : k * (b - a) = 0 := by linarith
    rcases mul_eq_zero.mp this with h' | h'
    · exact h'
    · exact absurd h' hba
  have hfin := hvconst x hx a hamem
  rw [hva, hk0] at hfin
  simp only [hv, hk0, zero_mul, sub_zero] at hfin
  linarith [hfin]

/-! ### The escape domain has to be bounded

The polynomial law is a statement about escape from a *bounded* interval, and that is not a
convenience of the formalization. The three results below show that the driftless generator
equation admits no non-negative solution on a half-line, so a mean exit time obeying it cannot
exist there: every solution is a downward parabola, and a downward parabola is eventually
negative. Probabilistically this is the familiar fact that driftless Brownian motion started
below a single absorbing level has infinite mean hitting time. The diffusive probe of
`sec:exp-e6` is exactly that configuration — the SDE is drift-free, absorption is at the single
shoulder level, and nothing bounds the process from below — so the envelope
`E[τ] ∝ (w̃_s - w̃_0)²/σ_Jac²` that `fig:e6` draws is the bounded-domain value rather than the
mean first-passage time of the probe as literally specified. -/

/-- **Every solution of the driftless generator equation is a downward parabola.** On a convex
open set, `(σ²/2) T'' = -1` determines `T` from its value and slope at any one base point:
`T(x) = T(x₀) + T'(x₀)(x - x₀) - (x - x₀)²/σ²`. The leading coefficient `-1/σ²` is negative, which
is the structural reason both for uniqueness on a bounded interval
(`eq_freeExitTime_of_isFreeExitTime`, where the two boundary conditions fix the two free
constants) and for non-existence of a non-negative solution on an unbounded one
(`no_nonneg_solution_on_Iio`, `no_nonneg_solution_on_Ioi`). -/
theorem eq_quadratic_of_generator {σsq : ℝ} (hσ : 0 < σsq) {D : Set ℝ} (hDc : Convex ℝ D)
    (hDo : IsOpen D) {T T' T'' : ℝ → ℝ}
    (hT : ∀ x ∈ D, HasDerivAt T (T' x) x) (hT' : ∀ x ∈ D, HasDerivAt T' (T'' x) x)
    (hgen : ∀ x ∈ D, σsq / 2 * T'' x = -1) {x₀ : ℝ} (hx₀ : x₀ ∈ D) :
    ∀ x ∈ D, T x = T x₀ + T' x₀ * (x - x₀) - (x - x₀) ^ 2 / σsq := by
  have hσ' : σsq ≠ 0 := ne_of_gt hσ
  set k : ℝ := 2 / σsq with hk
  -- the generator equation pins the second derivative
  have hT''val : ∀ x ∈ D, T'' x = -k := by
    intro x hx
    have hval : T'' x = -2 / σsq := by
      rw [eq_div_iff hσ']
      have := hgen x hx
      linarith
    rw [hval, hk]
    ring
  -- hence `T' + k·id` has vanishing derivative, so it is constant
  have hd1 : ∀ y ∈ D, HasDerivAt (fun z => T' z + k * z) 0 y := by
    intro y hy
    have h1 : HasDerivAt (fun z : ℝ => k * z) k y := by simpa using (hasDerivAt_id y).const_mul k
    have := (hT' y hy).add h1
    rwa [hT''val y hy, neg_add_cancel] at this
  have hc1 : ∀ y ∈ D, T' y + k * y = T' x₀ + k * x₀ := by
    intro y hy
    refine eq_of_hasDerivWithinAt_interior_zero (f := fun w => T' w + k * w) hDc ?_ ?_ hy hx₀
    · exact fun w hw => ((hd1 w hw).continuousAt).continuousWithinAt
    · rw [hDo.interior_eq]
      exact fun w hw => (hd1 w hw).hasDerivWithinAt
  have hT'val : ∀ y ∈ D, T' y = T' x₀ - k * (y - x₀) := by
    intro y hy
    have := hc1 y hy
    linarith
  -- the parabola with that derivative
  have hdQ : ∀ y : ℝ, HasDerivAt (fun z : ℝ => T x₀ + T' x₀ * (z - x₀) - k * (z - x₀) ^ 2 / 2)
      (T' x₀ - k * (y - x₀)) y := by
    intro y
    have h1 : HasDerivAt (fun z : ℝ => T x₀ + T' x₀ * (z - x₀)) (T' x₀ * 1) y :=
      (((hasDerivAt_id y).sub_const x₀).const_mul _).const_add (T x₀)
    have h2 : HasDerivAt (fun z : ℝ => k * (z - x₀) ^ 2 / 2)
        (k * (2 * (y - x₀) ^ 1 * 1) / 2) y := by
      have hsq : HasDerivAt (fun z : ℝ => (z - x₀) ^ 2) (2 * (y - x₀) ^ 1 * 1) y :=
        ((hasDerivAt_id y).sub_const x₀).pow 2
      exact (hsq.const_mul k).div_const 2
    have hcomb : HasDerivAt (fun z : ℝ => T x₀ + T' x₀ * (z - x₀) - k * (z - x₀) ^ 2 / 2)
        (T' x₀ * 1 - k * (2 * (y - x₀) ^ 1 * 1) / 2) y := h1.sub h2
    have hval : T' x₀ * 1 - k * (2 * (y - x₀) ^ 1 * 1) / 2 = T' x₀ - k * (y - x₀) := by ring
    rwa [hval] at hcomb
  -- so `T` minus the parabola is constant, and vanishes at the base point
  have hd2 : ∀ y ∈ D,
      HasDerivAt (fun z => T z - (T x₀ + T' x₀ * (z - x₀) - k * (z - x₀) ^ 2 / 2)) 0 y := by
    intro y hy
    have := (hT y hy).sub (hdQ y)
    rwa [hT'val y hy, sub_self] at this
  have hc2 : ∀ y ∈ D, T y - (T x₀ + T' x₀ * (y - x₀) - k * (y - x₀) ^ 2 / 2)
      = T x₀ - (T x₀ + T' x₀ * (x₀ - x₀) - k * (x₀ - x₀) ^ 2 / 2) := by
    intro y hy
    refine eq_of_hasDerivWithinAt_interior_zero
      (f := fun w => T w - (T x₀ + T' x₀ * (w - x₀) - k * (w - x₀) ^ 2 / 2)) hDc ?_ ?_ hy hx₀
    · exact fun w hw => ((hd2 w hw).continuousAt).continuousWithinAt
    · rw [hDo.interior_eq]
      exact fun w hw => (hd2 w hw).hasDerivWithinAt
  intro x hx
  have h := hc2 x hx
  have hrhs : T x₀ - (T x₀ + T' x₀ * (x₀ - x₀) - k * (x₀ - x₀) ^ 2 / 2) = 0 := by ring
  rw [hrhs, sub_eq_zero] at h
  rw [h, hk]
  field_simp

/-- **A solution on a domain reaching arbitrarily far from a base point goes negative.** The
downward parabola of `eq_quadratic_of_generator` eventually falls below zero, so a domain
containing points at unbounded distance from `x₀` contains a point at which the solution is
negative. The threshold is explicit: the point produced is at distance at least
`max 1 (σ² (|T(x₀)| + |T'(x₀)| + 1))` from the base point. -/
theorem exists_neg_of_generator_unbounded {σsq : ℝ} (hσ : 0 < σsq) {D : Set ℝ}
    (hDc : Convex ℝ D) (hDo : IsOpen D) {T T' T'' : ℝ → ℝ}
    (hT : ∀ x ∈ D, HasDerivAt T (T' x) x) (hT' : ∀ x ∈ D, HasDerivAt T' (T'' x) x)
    (hgen : ∀ x ∈ D, σsq / 2 * T'' x = -1) {x₀ : ℝ} (hx₀ : x₀ ∈ D)
    (hfar : ∀ R : ℝ, ∃ x ∈ D, R ≤ |x - x₀|) :
    ∃ x ∈ D, T x < 0 := by
  have hrep := eq_quadratic_of_generator hσ hDc hDo hT hT' hgen hx₀
  set A : ℝ := T x₀ with hA
  set B : ℝ := T' x₀ with hB
  set M : ℝ := max 1 (σsq * (|A| + |B| + 1)) with hM
  have hM1 : (1 : ℝ) ≤ M := le_max_left _ _
  have hM2 : σsq * (|A| + |B| + 1) ≤ M := le_max_right _ _
  obtain ⟨x, hxD, hxfar⟩ := hfar M
  refine ⟨x, hxD, ?_⟩
  have hu1 : (1 : ℝ) ≤ |x - x₀| := hM1.trans hxfar
  have hbig : (|A| + |B| + 1) * |x - x₀| ≤ (x - x₀) ^ 2 / σsq := by
    rw [le_div_iff₀ hσ, ← sq_abs (x - x₀)]
    nlinarith [mul_le_mul_of_nonneg_right (hM2.trans hxfar) (abs_nonneg (x - x₀))]
  have hAle : A ≤ |A| := le_abs_self A
  have hBle : B * (x - x₀) ≤ |B| * |x - x₀| :=
    (le_abs_self _).trans (le_of_eq (abs_mul B (x - x₀)))
  rw [hrep x hxD]
  nlinarith [hbig, hAle, hBle, hu1, abs_nonneg A, abs_nonneg B]

/-- **No non-negative solution on a half-line unbounded below.** This is the configuration of
the diffusive probe of `sec:exp-e6`: drift-free dynamics, a single absorbing level `b = w̃_s`,
and no lower boundary. A mean exit time is non-negative by construction, so the driftless
boundary value problem has no solution that could be one, and the envelope
`(w̃_s - w̃_0)²/σ_Jac²` of `fig:e6` must be read as the bounded-domain value
(`freeExitTime_symmetric_interval`) rather than as the mean first-passage time of the probe as
specified. Nothing here contradicts the paper's `Θ(σ_Jac^{-2})`, which is a statement about the
bounded problem; it locates the escape domain the statement needs. -/
theorem no_nonneg_solution_on_Iio {σsq b : ℝ} (hσ : 0 < σsq) {T T' T'' : ℝ → ℝ}
    (hT : ∀ x ∈ Set.Iio b, HasDerivAt T (T' x) x)
    (hT' : ∀ x ∈ Set.Iio b, HasDerivAt T' (T'' x) x)
    (hgen : ∀ x ∈ Set.Iio b, σsq / 2 * T'' x = -1) :
    ¬ ∀ x ∈ Set.Iio b, 0 ≤ T x := by
  intro hpos
  have hx₀ : b - 1 ∈ Set.Iio b := by simp only [Set.mem_Iio]; linarith
  have hfar : ∀ R : ℝ, ∃ x ∈ Set.Iio b, R ≤ |x - (b - 1)| := by
    intro R
    refine ⟨b - 1 - (|R| + 1), by simp only [Set.mem_Iio]; linarith [abs_nonneg R], ?_⟩
    have hshift : b - 1 - (|R| + 1) - (b - 1) = -(|R| + 1) := by ring
    rw [hshift, abs_neg, abs_of_nonneg (by linarith [abs_nonneg R] : (0 : ℝ) ≤ |R| + 1)]
    linarith [le_abs_self R]
  obtain ⟨x, hxD, hxneg⟩ :=
    exists_neg_of_generator_unbounded hσ (convex_Iio b) isOpen_Iio hT hT' hgen hx₀ hfar
  exact absurd (hpos x hxD) (not_le.mpr hxneg)

/-- **No non-negative solution on a half-line unbounded above**, the mirror image of
`no_nonneg_solution_on_Iio` and the statement for an escape that runs upward from a single
absorbing level. -/
theorem no_nonneg_solution_on_Ioi {σsq a : ℝ} (hσ : 0 < σsq) {T T' T'' : ℝ → ℝ}
    (hT : ∀ x ∈ Set.Ioi a, HasDerivAt T (T' x) x)
    (hT' : ∀ x ∈ Set.Ioi a, HasDerivAt T' (T'' x) x)
    (hgen : ∀ x ∈ Set.Ioi a, σsq / 2 * T'' x = -1) :
    ¬ ∀ x ∈ Set.Ioi a, 0 ≤ T x := by
  intro hpos
  have hx₀ : a + 1 ∈ Set.Ioi a := by simp only [Set.mem_Ioi]; linarith
  have hfar : ∀ R : ℝ, ∃ x ∈ Set.Ioi a, R ≤ |x - (a + 1)| := by
    intro R
    refine ⟨a + 1 + (|R| + 1), by simp only [Set.mem_Ioi]; linarith [abs_nonneg R], ?_⟩
    have hshift : a + 1 + (|R| + 1) - (a + 1) = |R| + 1 := by ring
    rw [hshift, abs_of_nonneg (by linarith [abs_nonneg R] : (0 : ℝ) ≤ |R| + 1)]
    linarith [le_abs_self R]
  obtain ⟨x, hxD, hxneg⟩ :=
    exists_neg_of_generator_unbounded hσ (convex_Ioi a) isOpen_Ioi hT hT' hgen hx₀ hfar
  exact absurd (hpos x hxD) (not_le.mpr hxneg)

/-! ### The scaling law -/

/-- The scaling law in the form the paper quotes: the exit time is the fixed geometric factor
`(x-a)(b-x)` times `(σ²)⁻¹`. -/
theorem freeExitTime_eq_mul_inv (σsq a b x : ℝ) :
    freeExitTime σsq a b x = ((x - a) * (b - x)) * σsq⁻¹ := by
  rw [freeExitTime, div_eq_mul_inv]

/-- **`Θ((σ²)⁻¹)`, on every filter.** For `x` strictly inside `(a,b)` the exit time, as a
function of the diffusion `σ²`, is `Θ` of `(σ²)⁻¹`. The relation holds on an arbitrary filter
because the dependence is an exact proportionality rather than an asymptotic one; the small-noise
(`𝓝[>] 0`) and large-noise (`atTop`) specializations are immediate. -/
theorem freeExitTime_isTheta_inv {a b x : ℝ} (hx : x ∈ Set.Ioo a b) (l : Filter ℝ) :
    (fun σsq : ℝ => freeExitTime σsq a b x) =Θ[l] fun σsq : ℝ => σsq⁻¹ := by
  have hC : (x - a) * (b - x) ≠ 0 :=
    ne_of_gt (mul_pos (sub_pos.mpr hx.1) (sub_pos.mpr hx.2))
  have hrw : (fun σsq : ℝ => freeExitTime σsq a b x)
      = fun σsq : ℝ => ((x - a) * (b - x)) * σsq⁻¹ :=
    funext fun σsq => freeExitTime_eq_mul_inv σsq a b x
  rw [hrw]
  exact (Asymptotics.isTheta_const_mul_left hC).mpr (Asymptotics.isTheta_refl _ l)

/-- The same statement written in the diffusion `σ` rather than in `σ²`: the exit time is
`Θ(σ^{-2})` in the process's **own** diffusion, which for the bias-channel SDE is the total
effective `σ_eff`. The variable of `rem:fpt-scope`'s `Θ(σ_Jac^{-2})` is instead the Jacobian
channel alone, one summand of `σ_eff² = σ_s²σ_obj² + σ_Jac²`; that decomposed form is proved
in `freeExitTime_isTheta_inv_decomposed` below, with the attenuated summand carried as a bound
rather than set to zero. -/
theorem freeExitTime_isTheta_inv_sq {a b x : ℝ} (hx : x ∈ Set.Ioo a b) (l : Filter ℝ) :
    (fun σ : ℝ => freeExitTime (σ ^ 2) a b x) =Θ[l] fun σ : ℝ => (σ ^ 2)⁻¹ := by
  have hC : (x - a) * (b - x) ≠ 0 :=
    ne_of_gt (mul_pos (sub_pos.mpr hx.1) (sub_pos.mpr hx.2))
  have hrw : (fun σ : ℝ => freeExitTime (σ ^ 2) a b x)
      = fun σ : ℝ => ((x - a) * (b - x)) * (σ ^ 2)⁻¹ :=
    funext fun σ => freeExitTime_eq_mul_inv (σ ^ 2) a b x
  rw [hrw]
  exact (Asymptotics.isTheta_const_mul_left hC).mpr (Asymptotics.isTheta_refl _ l)

/-- **The explicit two-sided bound the `Θ` claim abbreviates**, uniform over the sub-interval
`[a+δ, b-δ]`: the exit time is squeezed between `δ²(σ²)⁻¹` and `((b-a)²/4)(σ²)⁻¹` with both
constants explicit and the lower one strictly positive. -/
theorem freeExitTime_two_sided {σsq a b δ x : ℝ} (hσ : 0 < σsq) (hδ : 0 < δ)
    (h1 : a + δ ≤ x) (h2 : x ≤ b - δ) :
    δ ^ 2 * σsq⁻¹ ≤ freeExitTime σsq a b x ∧
      freeExitTime σsq a b x ≤ (b - a) ^ 2 / 4 * σsq⁻¹ := by
  have hxa : δ ≤ x - a := by linarith
  have hbx : δ ≤ b - x := by linarith
  have hinv : 0 < σsq⁻¹ := inv_pos.mpr hσ
  rw [freeExitTime_eq_mul_inv]
  constructor
  · have : δ ^ 2 ≤ (x - a) * (b - x) := by nlinarith
    exact mul_le_mul_of_nonneg_right this hinv.le
  · have : (x - a) * (b - x) ≤ (b - a) ^ 2 / 4 := by nlinarith [sq_nonneg ((x - a) - (b - x))]
    exact mul_le_mul_of_nonneg_right this hinv.le

/-- The exit time is strictly positive strictly inside the interval. -/
theorem freeExitTime_pos {σsq a b x : ℝ} (hσ : 0 < σsq) (hx : x ∈ Set.Ioo a b) :
    0 < freeExitTime σsq a b x := by
  rw [freeExitTime]
  exact div_pos (mul_pos (sub_pos.mpr hx.1) (sub_pos.mpr hx.2)) hσ

/-- The exit time is strictly decreasing in the diffusion: more noise, faster escape. -/
theorem freeExitTime_strictAntiOn_variance {a b x : ℝ} (hx : x ∈ Set.Ioo a b) :
    StrictAntiOn (fun σsq => freeExitTime σsq a b x) (Set.Ioi 0) := by
  intro s hs t _ hst
  have hC : 0 < (x - a) * (b - x) := mul_pos (sub_pos.mpr hx.1) (sub_pos.mpr hx.2)
  simp only [freeExitTime]
  exact div_lt_div_of_pos_left hC (Set.mem_Ioi.mp hs) hst

/-! ### The maximum, and the `L²/σ²` form -/

/-- The exit time from the midpoint is `(b-a)²/(4σ²)`. -/
theorem freeExitTime_midpoint (σsq a b : ℝ) :
    freeExitTime σsq a b ((a + b) / 2) = (b - a) ^ 2 / (4 * σsq) := by
  rw [freeExitTime]
  ring

/-- The midpoint maximizes the exit time, so `(b-a)²/(4σ²)` is the largest mean exit time from
the interval — the `L²/σ²` form the paper quotes for the diffusive regime. -/
theorem freeExitTime_le_midpoint {σsq : ℝ} (hσ : 0 < σsq) (a b x : ℝ) :
    freeExitTime σsq a b x ≤ freeExitTime σsq a b ((a + b) / 2) := by
  rw [freeExitTime, freeExitTime, div_eq_mul_inv, div_eq_mul_inv]
  have hnum : (x - a) * (b - x) ≤ ((a + b) / 2 - a) * (b - (a + b) / 2) := by
    nlinarith [sq_nonneg ((x - a) - (b - x))]
  exact mul_le_mul_of_nonneg_right hnum (inv_pos.mpr hσ).le

/-- Started at the centre of an interval of half-width `L`, the closed-form mean exit time is
exactly `L²/σ²`. This is the value of the idealized envelope drawn in `fig:e6`, with `L` the
nominal escape distance `w̃_s - w̃_0`.

The identification is of values, and the difference between the two problems is worth stating.
The probe of `sec:exp-e6` absorbs at the single level `w̃_s` and is unbounded below, and on such
a domain the driftless boundary value problem has no non-negative solution at all
(`no_nonneg_solution_on_Iio`); `L²/σ²` there is the bounded-domain idealization, which is what
the "standard Brownian hitting-time identity" of `rem:fpt-scope` supplies, and not the mean
first-passage time of the probe as literally specified. -/
theorem freeExitTime_symmetric_interval (σsq L : ℝ) :
    freeExitTime σsq (-L) L 0 = L ^ 2 / σsq := by
  rw [freeExitTime]
  ring

/-! ### The same laws for any solution of the boundary value problem

The scaling results above are stated of the closed form `freeExitTime`. By the uniqueness
result `eq_freeExitTime_of_isFreeExitTime` the identical laws hold for **any** function
satisfying `IsFreeExitTime` — that is, granting Dynkin's formula, for the mean exit time
itself. The three theorems below are those
paper-facing forms, with the boundary value problem as the single named hypothesis, in the
pattern of the paper-facing theorems of `KramersExitTime.lean`. The hypothesis is satisfiable
(`freeExitTime_isFreeExitTime`), so none of them is vacuous. -/

/-- **The two-sided polynomial bound, for any solution of the Dirichlet problem.** If `T`
solves the driftless exit-time boundary value problem — which, granting Dynkin's formula, the
mean exit time of `dX = σ dB` from `(a,b)` does — then `T` itself is squeezed between
`δ²(σ²)⁻¹` and `((b-a)²/4)(σ²)⁻¹` on the sub-interval `[a+δ, b-δ]`. This is the two-sided form
of the `Θ(σ^{-2})` claim of `rem:fpt-scope`, stated of the solution rather than of the closed
form. -/
theorem isFreeExitTime_two_sided {σsq a b δ x : ℝ} {T T' T'' : ℝ → ℝ}
    (h : IsFreeExitTime σsq a b T T' T'') (hσ : 0 < σsq) (hδ : 0 < δ)
    (h1 : a + δ ≤ x) (h2 : x ≤ b - δ) :
    δ ^ 2 * σsq⁻¹ ≤ T x ∧ T x ≤ (b - a) ^ 2 / 4 * σsq⁻¹ := by
  have hab : a < b := by linarith
  have hxIcc : x ∈ Set.Icc a b := ⟨by linarith, by linarith⟩
  rw [eq_freeExitTime_of_isFreeExitTime hσ hab h hxIcc]
  exact freeExitTime_two_sided hσ hδ h1 h2

/-- **The `L²/σ²` maximum, for any solution of the Dirichlet problem**: at the midpoint of the
interval any solution takes the value `(b-a)²/(4σ²)` — the free-diffusion time — so the largest
mean exit time the boundary value problem allows is the `L²/σ²` form the paper quotes. -/
theorem isFreeExitTime_midpoint {σsq a b : ℝ} {T T' T'' : ℝ → ℝ}
    (h : IsFreeExitTime σsq a b T T' T'') (hσ : 0 < σsq) (hab : a < b) :
    T ((a + b) / 2) = (b - a) ^ 2 / (4 * σsq) := by
  have hmid : (a + b) / 2 ∈ Set.Icc a b := ⟨by linarith, by linarith⟩
  rw [eq_freeExitTime_of_isFreeExitTime hσ hab h hmid]
  exact freeExitTime_midpoint σsq a b

/-- **`Θ((σ²)⁻¹)` for any family of solutions of the Dirichlet problem.** Let `T σ²` solve the
driftless exit-time boundary value problem at each diffusion level `σ² > 0` — granting Dynkin's
formula, the family of mean exit times does. Then from every start point strictly inside the
interval, `σ² ↦ T σ² x` is `Θ((σ²)⁻¹)` along any filter on which the diffusion is eventually
positive; `atTop` and `𝓝[>] 0` are the two specializations the crossover discussion uses. This
is the polynomial law of `rem:fpt-scope` stated of the exit time itself rather than of the
closed form, and Dynkin's formula is the only unformalized step between this statement and the
probabilistic one. -/
theorem isFreeExitTime_isTheta_inv {a b x : ℝ} (hx : x ∈ Set.Ioo a b)
    {T T' T'' : ℝ → ℝ → ℝ}
    (hT : ∀ σsq : ℝ, 0 < σsq → IsFreeExitTime σsq a b (T σsq) (T' σsq) (T'' σsq))
    {l : Filter ℝ} (hl : ∀ᶠ σsq in l, 0 < σsq) :
    (fun σsq : ℝ => T σsq x) =Θ[l] fun σsq : ℝ => σsq⁻¹ := by
  have hab : a < b := hx.1.trans hx.2
  have hxIcc : x ∈ Set.Icc a b := ⟨hx.1.le, hx.2.le⟩
  have heq : (fun σsq : ℝ => freeExitTime σsq a b x) =ᶠ[l] fun σsq : ℝ => T σsq x := by
    filter_upwards [hl] with σsq hσ
    exact (eq_freeExitTime_of_isFreeExitTime hσ hab (hT σsq hσ) hxIcc).symm
  exact ((freeExitTime_isTheta_inv hx l).symm.trans_eventuallyEq heq).symm

/-! ### The polynomial law in the paper's decomposed variables

`rem:fpt-scope` and the closing paragraph of `app:proof-fpt` state the post-crossover law in
the Jacobian channel alone — "mean first-passage time the polynomial `Θ(σ_Jac^{-2})`" — while
the diffusion of the bias-channel SDE is the total effective variance
`σ_eff² = σ_s²σ_obj² + σ_Jac²` of `app:proof-fpt`. The paper's own bridge is the shoulder
collapse of its Step 4, "on the shoulder the first term collapses and `σ_eff² ≈ σ_Jac²`" — a
leading-order identification. The results above are exact in the SDE's own `σ²`, that is in
`σ_eff²`; the four theorems below restate them in the paper's variable `σ_Jac²` with that
identification **carried as a hypothesis instead of applied**: the attenuated channel enters as
a summand known only to lie in `[0, K]`, and the polynomial law in `σ_Jac²` alone comes out
with the lower constant degraded by the explicit factor `s₀/(s₀+K)` at threshold `s₀`, or, in
the limit `σ_Jac² → ∞`, not at all. Setting `K = 0` recovers the undecomposed statements
exactly. -/

/-- **The two-sided bound in the paper's decomposed variables.** Write the effective diffusion
as `app:proof-fpt` does, `σ_eff² = σ_s²σ_obj² + σ_Jac²`, with the attenuated channel `σobj`
known only to lie in `[0, K]` and the Jacobian channel `σJac` past an arbitrary positive
threshold `s₀` — for the crossover discussion `s₀ = 2α`, the "once `σ_Jac²` grows comparable to
the barrier" regime of `rem:fpt-scope`. Then the exit time started in `[a+δ, b-δ]` is squeezed
between explicit constant multiples of `σJac⁻¹` in the Jacobian channel **alone**:
`δ²·(s₀/(s₀+K))` below and `(b-a)²/4` above. The shoulder collapse `σ_eff² ≈ σ_Jac²` of the
paper's Step 4 is thus a carried bound rather than an applied approximation; at `K = 0` the
constants reduce to those of `freeExitTime_two_sided`. -/
theorem freeExitTime_two_sided_decomposed {a b δ x σobj σJac K s₀ : ℝ} (hδ : 0 < δ)
    (h1 : a + δ ≤ x) (h2 : x ≤ b - δ) (hobj : 0 ≤ σobj) (hK : σobj ≤ K)
    (hs₀ : 0 < s₀) (hs : s₀ ≤ σJac) :
    δ ^ 2 * (s₀ / (s₀ + K)) * σJac⁻¹ ≤ freeExitTime (σobj + σJac) a b x ∧
      freeExitTime (σobj + σJac) a b x ≤ (b - a) ^ 2 / 4 * σJac⁻¹ := by
  have hσJ : 0 < σJac := hs₀.trans_le hs
  have hsum : 0 < σobj + σJac := by linarith
  have hK0 : 0 ≤ K := hobj.trans hK
  obtain ⟨hlo, hhi⟩ := freeExitTime_two_sided hsum hδ h1 h2
  refine ⟨?_, ?_⟩
  · have hfrac : s₀ / (s₀ + K) * σJac⁻¹ ≤ (σobj + σJac)⁻¹ := by
      rw [← div_eq_mul_inv, div_div, ← one_div (σobj + σJac),
        div_le_div_iff₀ (by positivity) hsum]
      nlinarith [mul_le_mul_of_nonneg_right hs hobj, mul_le_mul_of_nonneg_left hK hσJ.le]
    calc δ ^ 2 * (s₀ / (s₀ + K)) * σJac⁻¹
        = δ ^ 2 * (s₀ / (s₀ + K) * σJac⁻¹) := by ring
      _ ≤ δ ^ 2 * (σobj + σJac)⁻¹ := mul_le_mul_of_nonneg_left hfrac (by positivity)
      _ ≤ freeExitTime (σobj + σJac) a b x := hlo
  · exact hhi.trans
      (mul_le_mul_of_nonneg_left (inv_anti₀ hσJ (by linarith)) (by positivity))

/-- **`Θ(σ_Jac^{-2})` in the Jacobian channel alone.** Let the attenuated channel vary
arbitrarily with the Jacobian channel, subject only to staying in `[0, K]`. Then as
`σ_Jac² → ∞` the exit time at total diffusion `σobj σJac + σJac` is `Θ` of `σJac⁻¹`: the
literal `Θ(σ_Jac^{-2})` of `rem:fpt-scope` and of the closing paragraph of `app:proof-fpt`,
with the shoulder collapse `σ_eff² ≈ σ_Jac²` carried as the boundedness hypothesis rather than
assumed. The filter is `atTop` because in the decomposed variable the proportionality is no
longer exact — at bounded `σ_Jac²` the attenuated summand shifts the constants, and
`freeExitTime_two_sided_decomposed` gives that regime explicitly. -/
theorem freeExitTime_isTheta_inv_decomposed {a b x K : ℝ} (hx : x ∈ Set.Ioo a b)
    {σobj : ℝ → ℝ} (hobj : ∀ t, 0 ≤ σobj t) (hK : ∀ t, σobj t ≤ K) :
    (fun σJac : ℝ => freeExitTime (σobj σJac + σJac) a b x) =Θ[Filter.atTop]
      fun σJac : ℝ => σJac⁻¹ := by
  have hC : 0 < (x - a) * (b - x) := mul_pos (sub_pos.mpr hx.1) (sub_pos.mpr hx.2)
  constructor
  · refine Asymptotics.isBigO_iff.mpr ⟨(x - a) * (b - x), ?_⟩
    filter_upwards [Filter.eventually_ge_atTop (1 : ℝ)] with t ht
    have ht0 : 0 < t := lt_of_lt_of_le one_pos ht
    have hsum : 0 < σobj t + t := by have := hobj t; linarith
    have hTpos : 0 < freeExitTime (σobj t + t) a b x := freeExitTime_pos hsum hx
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hTpos, abs_of_pos (inv_pos.mpr ht0),
      freeExitTime_eq_mul_inv]
    exact mul_le_mul_of_nonneg_left (inv_anti₀ ht0 (by have := hobj t; linarith)) hC.le
  · refine Asymptotics.isBigO_iff.mpr ⟨2 / ((x - a) * (b - x)), ?_⟩
    filter_upwards [Filter.eventually_ge_atTop (max K 1)] with t ht
    have htK : K ≤ t := le_trans (le_max_left _ _) ht
    have ht1 : (1 : ℝ) ≤ t := le_trans (le_max_right _ _) ht
    have ht0 : 0 < t := lt_of_lt_of_le one_pos ht1
    have hsum : 0 < σobj t + t := by have := hobj t; linarith
    have hTpos : 0 < freeExitTime (σobj t + t) a b x := freeExitTime_pos hsum hx
    have hsum2 : σobj t + t ≤ 2 * t := by have := hK t; linarith
    have hCne : (x - a) * (b - x) ≠ 0 := ne_of_gt hC
    have hsumne : σobj t + t ≠ 0 := ne_of_gt hsum
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr ht0), abs_of_pos hTpos,
      freeExitTime_eq_mul_inv]
    have hcancel : 2 / ((x - a) * (b - x)) * ((x - a) * (b - x) * (σobj t + t)⁻¹)
        = 2 * (σobj t + t)⁻¹ := by
      field_simp
      exact div_self hCne
    rw [hcancel]
    calc t⁻¹ = 1 / t := (one_div t).symm
      _ ≤ 2 / (σobj t + t) := by rw [div_le_div_iff₀ ht0 hsum]; linarith
      _ = 2 * (σobj t + t)⁻¹ := by rw [div_eq_mul_inv]

/-- **The decomposed two-sided bound, for any solution of the Dirichlet problem.** If `T`
solves the driftless exit-time boundary value problem at the total diffusion `σobj + σJac` —
which, granting Dynkin's formula, the mean exit time of the lifted bias-channel SDE does —
then `T` itself obeys the `σ_Jac`-channel sandwich of `freeExitTime_two_sided_decomposed`. -/
theorem isFreeExitTime_two_sided_decomposed {a b δ x σobj σJac K s₀ : ℝ} {T T' T'' : ℝ → ℝ}
    (h : IsFreeExitTime (σobj + σJac) a b T T' T'') (hδ : 0 < δ)
    (h1 : a + δ ≤ x) (h2 : x ≤ b - δ) (hobj : 0 ≤ σobj) (hK : σobj ≤ K)
    (hs₀ : 0 < s₀) (hs : s₀ ≤ σJac) :
    δ ^ 2 * (s₀ / (s₀ + K)) * σJac⁻¹ ≤ T x ∧ T x ≤ (b - a) ^ 2 / 4 * σJac⁻¹ := by
  have hσJ : 0 < σJac := hs₀.trans_le hs
  have hsum : 0 < σobj + σJac := by linarith
  have hab : a < b := by linarith
  have hxIcc : x ∈ Set.Icc a b := ⟨by linarith, by linarith⟩
  rw [eq_freeExitTime_of_isFreeExitTime hsum hab h hxIcc]
  exact freeExitTime_two_sided_decomposed hδ h1 h2 hobj hK hs₀ hs

/-- **`Θ(σ_Jac^{-2})` for any family of solutions, in the paper's decomposed variables.** Let
`T σ²` solve the driftless exit-time boundary value problem at every diffusion `σ² > 0` —
granting Dynkin's formula, the family of mean exit times does — and let the attenuated channel
stay in `[0, K]`. Then from every start strictly inside the interval the solution at total
diffusion `σobj σJac + σJac` is `Θ(σJac⁻¹)` as `σ_Jac² → ∞`. This is the statement of
`rem:fpt-scope`'s "mean first-passage time the polynomial `Θ(σ_Jac^{-2})`" in the remark's own
variable, with Dynkin's formula the only unformalized step and the shoulder collapse carried
as a bound. -/
theorem isFreeExitTime_isTheta_inv_decomposed {a b x K : ℝ} (hx : x ∈ Set.Ioo a b)
    {T T' T'' : ℝ → ℝ → ℝ}
    (hT : ∀ σsq : ℝ, 0 < σsq → IsFreeExitTime σsq a b (T σsq) (T' σsq) (T'' σsq))
    {σobj : ℝ → ℝ} (hobj : ∀ t, 0 ≤ σobj t) (hK : ∀ t, σobj t ≤ K) :
    (fun σJac : ℝ => T (σobj σJac + σJac) x) =Θ[Filter.atTop] fun σJac : ℝ => σJac⁻¹ := by
  have hab : a < b := hx.1.trans hx.2
  have hxIcc : x ∈ Set.Icc a b := ⟨hx.1.le, hx.2.le⟩
  have heq : (fun σJac : ℝ => freeExitTime (σobj σJac + σJac) a b x)
      =ᶠ[Filter.atTop] fun σJac : ℝ => T (σobj σJac + σJac) x := by
    filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with t ht
    have hsum : 0 < σobj t + t := by have := hobj t; linarith
    exact (eq_freeExitTime_of_isFreeExitTime hsum hab (hT _ hsum) hxIcc).symm
  exact ((freeExitTime_isTheta_inv_decomposed hx hobj hK).symm.trans_eventuallyEq heq).symm

/-! ### The crossover out of the Arrhenius regime -/

/-- The largest driftless mean exit time from `(a,b)`, `(b-a)²/(4σ²)`: the free-diffusion law of
`rem:fpt-scope`, as a function of the effective diffusion `σ²`. -/
noncomputable def freeDiffusionTime (σsq a b : ℝ) : ℝ := (b - a) ^ 2 / (4 * σsq)

/-- The Arrhenius (Kramers) mean first-passage time of `eq:fpt`, `exp(2α/σ²)`, with `α > 0` the
barrier action held fixed across the comparison. -/
noncomputable def arrheniusTime (α σsq : ℝ) : ℝ := Real.exp (2 * α / σsq)

/-- The free-diffusion law is the exit time from the midpoint. -/
theorem freeDiffusionTime_eq_freeExitTime_midpoint (σsq a b : ℝ) :
    freeDiffusionTime σsq a b = freeExitTime σsq a b ((a + b) / 2) :=
  (freeExitTime_midpoint σsq a b).symm

/-- The Arrhenius time exceeds one whenever the barrier and the diffusion are positive. -/
theorem one_lt_arrheniusTime {α σsq : ℝ} (hα : 0 < α) (hσ : 0 < σsq) :
    1 < arrheniusTime α σsq := by
  rw [arrheniusTime, Real.one_lt_exp_iff]
  positivity

/-- The Arrhenius time is strictly decreasing in the diffusion. -/
theorem arrheniusTime_strictAntiOn {α : ℝ} (hα : 0 < α) :
    StrictAntiOn (arrheniusTime α) (Set.Ioi 0) := by
  intro s hs t _ hst
  rw [arrheniusTime, arrheniusTime, Real.exp_lt_exp]
  exact div_lt_div_of_pos_left (by positivity) (Set.mem_Ioi.mp hs) hst

/-- **The crossover condition.** The Arrhenius exponent `2α/σ²` exceeds one exactly when the
diffusion is below `2α`. This is the formal content of the paper's "once `σ_Jac²` grows
comparable to the barrier `2α` the lifted exponent drops to order one … and (A5) fails": the
exponent is strictly antitone in `σ²` and crosses the level one exactly once, at `σ² = 2α`. -/
theorem metastable_window {α σsq : ℝ} (hσ : 0 < σsq) : 1 < 2 * α / σsq ↔ σsq < 2 * α := by
  rw [lt_div_iff₀ hσ, one_mul]

/-- The metastable window of (A5), as a set: the small-noise regime in which the Arrhenius
exponent is larger than one is exactly the interval `(0, 2α)`, so the crossover point `σ² = 2α`
is unique. -/
theorem metastable_window_eq_Ioo (α : ℝ) :
    {σsq : ℝ | 0 < σsq ∧ 1 < 2 * α / σsq} = Set.Ioo 0 (2 * α) := by
  ext σsq
  constructor
  · rintro ⟨hpos, hexp⟩
    exact ⟨hpos, (metastable_window hpos).mp hexp⟩
  · rintro ⟨hpos, hlt⟩
    exact ⟨hpos, (metastable_window hpos).mpr hlt⟩

/-- **The crossover condition in the paper's own variables.** `rem:fpt-scope` states the
crossover in terms of the unattenuated Jacobian channel alone: "once the unattenuated `σ_Jac²`
grows comparable to the barrier `2α` … (A5) fails". The lifted effective variance of
`app:proof-fpt` is the sum `σ_eff² = σ_s²σ_obj² + σ_Jac²` of the attenuated objective channel and
the unattenuated Jacobian channel, so the Jacobian channel alone reaching `2α` is already enough:
the lifted Arrhenius exponent is then at most one whatever the attenuated channel contributes.
This is the statement of `metastable_window` in the decomposed variables the paper uses, which
matters because the paper's threshold is on `σ_Jac²` and not on the total. -/
theorem lift_leaves_metastable_window {α σobj σJac : ℝ} (hα : 0 < α) (hobj : 0 ≤ σobj)
    (h : 2 * α ≤ σJac) : ¬ (1 < 2 * α / (σobj + σJac)) := by
  have hσ : 0 < σobj + σJac := by linarith
  rw [metastable_window hσ]
  exact not_lt.mpr (by linarith)

/-- **The per-method scoping of (FW-5), stated.** If the attenuated objective channel alone is
below the barrier while the two channels together are not, then the direct parametrization sits
inside the metastable window of (A5) — its Arrhenius exponent exceeds one — and the lift sits
outside it. This is the paper's own reading of the scope of `cor:fpt`: the Arrhenius asymptotic
is where the direct baseline lives, and the lift, precisely because `σ_Jac²` is added
unattenuated, can be carried past the crossover into the free-diffusion regime of
`freeExitTime`. Nothing here asserts that the lift always crosses over; the crossing of the two
channels is a hypothesis of the theorem, exactly as it is a regime assumption in the paper. -/
theorem direct_metastable_lift_past_crossover {α σobj σJac : ℝ} (hobj : 0 < σobj)
    (hdirect : σobj < 2 * α) (hlift : 2 * α ≤ σobj + σJac) :
    1 < 2 * α / σobj ∧ ¬ (1 < 2 * α / (σobj + σJac)) := by
  have hsum : 0 < σobj + σJac := lt_of_lt_of_le (by linarith) hlift
  refine ⟨(metastable_window hobj).mpr hdirect, ?_⟩
  rw [metastable_window hsum]
  exact not_lt.mpr hlift

/-- At the crossover the Arrhenius factor is exactly `e`. -/
theorem arrheniusTime_at_crossover {α : ℝ} (hα : 0 < α) :
    arrheniusTime α (2 * α) = Real.exp 1 := by
  rw [arrheniusTime, div_self (by positivity : (2 : ℝ) * α ≠ 0)]

/-- Past the crossover `σ² ≥ 2α` the Arrhenius exponent has dropped to at most one, so the
Arrhenius factor is at most `e`. With `one_lt_arrheniusTime` it is then trapped in `(1, e]`: an
`O(1)` correction rather than an exponentially large one. -/
theorem arrheniusTime_le_exp_one_of_crossover {α σsq : ℝ} (hα : 0 < α) (hσ : 2 * α ≤ σsq) :
    arrheniusTime α σsq ≤ Real.exp 1 := by
  have hσ0 : 0 < σsq := lt_of_lt_of_le (by positivity) hσ
  rw [arrheniusTime, Real.exp_le_exp, div_le_one hσ0]
  exact hσ

/-- **Past the crossover the exponential factor drops out of the first-passage time.** The
paper's `≍` in `eq:fpt` means equality of the leading exponential factor up to a sub-exponential
prefactor, so a mean first-passage time of the Kramers form is `P · exp(2α/σ²)` with `P ≥ 0` the
prefactor. Once `σ² ≥ 2α` the exponential factor lies in `(1, e]`, so that time is squeezed
between `P` and `e·P`: past the crossover the prefactor, and not the exponential, sets the law.
The prefactor is an arbitrary non-negative real here, so nothing about its identity is used. -/
theorem kramers_between_prefactor_and_exp_one_of_crossover {α σsq P : ℝ} (hα : 0 < α)
    (hσ : 2 * α ≤ σsq) (hP : 0 ≤ P) :
    P ≤ P * arrheniusTime α σsq ∧ P * arrheniusTime α σsq ≤ Real.exp 1 * P := by
  have hσ0 : 0 < σsq := lt_of_lt_of_le (by positivity) hσ
  have h1 : 1 ≤ arrheniusTime α σsq := (one_lt_arrheniusTime hα hσ0).le
  have h2 : arrheniusTime α σsq ≤ Real.exp 1 := arrheniusTime_le_exp_one_of_crossover hα hσ
  exact ⟨by nlinarith, by nlinarith⟩

/-- **The polynomial law past the crossover, for the composite, as a `Θ`.** Put the
free-diffusion time in the role of the Kramers prefactor, so that the composite law is
`((b-a)²/(4σ²))·exp(2α/σ²)`. That composite is `Θ((σ²)⁻¹)` as `σ² → ∞`, with the explicit
constants `(b-a)²/4` and `e(b-a)²/4` supplied by the squeeze above: past the crossover the
polynomial law emerges not because the two closed forms cross, but because the exponential
factor has become a bounded multiplier. The decomposition of the post-crossover time as
free-diffusion prefactor times Arrhenius exponential is this file's bridging reading and not an
assertion of the paper — `rem:fpt-scope` abandons the Kramers form past the crossover rather
than reinterpreting it, and its own polynomial claim is formalized in
`isFreeExitTime_two_sided` and `isFreeExitTime_isTheta_inv`; the identification is recorded
again in the `## Honesty` section. The theorem is exactly about the composite as written. -/
theorem kramersTime_isTheta_inv_atTop {α a b : ℝ} (hα : 0 < α) (hab : a ≠ b) :
    (fun σsq : ℝ => freeDiffusionTime σsq a b * arrheniusTime α σsq)
      =Θ[Filter.atTop] fun σsq : ℝ => σsq⁻¹ := by
  have hba : b - a ≠ 0 := sub_ne_zero_of_ne (Ne.symm hab)
  have hk : 0 < (b - a) ^ 2 / 4 := by positivity
  constructor
  · refine Asymptotics.isBigO_iff.mpr ⟨(b - a) ^ 2 / 4 * Real.exp 1, ?_⟩
    filter_upwards [Filter.eventually_ge_atTop (2 * α)] with σsq hσ
    have hσ0 : 0 < σsq := lt_of_lt_of_le (by positivity) hσ
    have hinv : 0 < σsq⁻¹ := inv_pos.mpr hσ0
    have hfree : freeDiffusionTime σsq a b = (b - a) ^ 2 / 4 * σsq⁻¹ := by
      rw [freeDiffusionTime]
      field_simp
    have harr : arrheniusTime α σsq ≤ Real.exp 1 := arrheniusTime_le_exp_one_of_crossover hα hσ
    have hpos : 0 < freeDiffusionTime σsq a b * arrheniusTime α σsq := by
      rw [hfree, arrheniusTime]
      positivity
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hpos, abs_of_pos hinv, hfree]
    nlinarith [mul_le_mul_of_nonneg_left harr (mul_pos hk hinv).le]
  · refine Asymptotics.isBigO_iff.mpr ⟨4 / (b - a) ^ 2, ?_⟩
    filter_upwards [Filter.eventually_ge_atTop (2 * α)] with σsq hσ
    have hσ0 : 0 < σsq := lt_of_lt_of_le (by positivity) hσ
    have hinv : 0 < σsq⁻¹ := inv_pos.mpr hσ0
    have hfree : freeDiffusionTime σsq a b = (b - a) ^ 2 / 4 * σsq⁻¹ := by
      rw [freeDiffusionTime]
      field_simp
    have h1 : 1 ≤ arrheniusTime α σsq := (one_lt_arrheniusTime hα hσ0).le
    have hpos : 0 < freeDiffusionTime σsq a b * arrheniusTime α σsq := by
      rw [hfree, arrheniusTime]
      positivity
    have hcancel : 4 / (b - a) ^ 2 * ((b - a) ^ 2 / 4 * σsq⁻¹ * arrheniusTime α σsq)
        = σsq⁻¹ * arrheniusTime α σsq := by
      field_simp
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hpos, abs_of_pos hinv, hfree, hcancel]
    nlinarith [mul_le_mul_of_nonneg_left h1 hinv.le]

/-- **Large noise.** Past `σ² > (b-a)²/4` the free-diffusion time has fallen below one while the
Arrhenius prediction is still above one, so the free-diffusion law gives the smaller time. The
threshold is explicit and the inequality is strict. -/
theorem freeDiffusionTime_lt_arrheniusTime_of_large_variance {α a b σsq : ℝ} (hα : 0 < α)
    (h : (b - a) ^ 2 / 4 < σsq) : freeDiffusionTime σsq a b < arrheniusTime α σsq := by
  have hσ : 0 < σsq := lt_of_le_of_lt (by positivity) h
  have hlt1 : freeDiffusionTime σsq a b < 1 := by
    rw [freeDiffusionTime, div_lt_one (by positivity)]
    linarith
  exact hlt1.trans (one_lt_arrheniusTime hα hσ)

/-- **Small noise, with an arbitrary constant.** Below `σ² ≤ 8α²/(c (b-a)²)` the Arrhenius time
exceeds `c` times the free-diffusion time. Since the threshold depends on `c` only through the
bound, this says that *no* constant multiple of the free-diffusion law bounds the Arrhenius law
in the metastable regime: the two are genuinely different laws, not one law up to a constant. -/
theorem const_mul_freeDiffusionTime_lt_arrheniusTime_of_small_variance {α a b σsq c : ℝ}
    (hα : 0 < α) (hσ : 0 < σsq) (h : c * (b - a) ^ 2 * σsq ≤ 8 * α ^ 2) :
    c * freeDiffusionTime σsq a b < arrheniusTime α σsq := by
  have hσ' : σsq ≠ 0 := ne_of_gt hσ
  have hq : 1 + 2 * α / σsq + (2 * α / σsq) ^ 2 / 2 ≤ Real.exp (2 * α / σsq) :=
    Real.quadratic_le_exp_of_nonneg (by positivity)
  have hsplit : 1 + 2 * α / σsq + (2 * α / σsq) ^ 2 / 2 - c * ((b - a) ^ 2 / (4 * σsq))
      = (σsq ^ 2 + 2 * α * σsq + 2 * α ^ 2 - c * (b - a) ^ 2 * σsq / 4) / σsq ^ 2 := by
    field_simp
  have hpos : 0 < (σsq ^ 2 + 2 * α * σsq + 2 * α ^ 2 - c * (b - a) ^ 2 * σsq / 4) / σsq ^ 2 := by
    apply div_pos _ (by positivity)
    nlinarith [pow_pos hσ 2, mul_pos hα hσ]
  rw [← hsplit] at hpos
  rw [freeDiffusionTime, arrheniusTime]
  linarith

/-- **Small noise.** Below `σ² ≤ 8α²/(b-a)²` the Arrhenius time is the larger of the two. This
is the `c = 1` case of the previous theorem, and it is the half of the crossover claim that the
paper's remark gets right: deep in the metastable regime the exponential law dominates. -/
theorem freeDiffusionTime_lt_arrheniusTime_of_small_variance {α a b σsq : ℝ} (hα : 0 < α)
    (hσ : 0 < σsq) (h : (b - a) ^ 2 * σsq ≤ 8 * α ^ 2) :
    freeDiffusionTime σsq a b < arrheniusTime α σsq := by
  have := const_mul_freeDiffusionTime_lt_arrheniusTime_of_small_variance (c := 1) hα hσ (by
    simpa using h)
  simpa using this

/-- **Any crossing is confined to an explicit bounded window.** If the free-diffusion time ever
reaches the Arrhenius time, the diffusion must lie in `(8α²/(b-a)², (b-a)²/4]`. Both endpoints
come from the two theorems above, so the window is exactly the region the extreme-regime
arguments leave open; the free-diffusion law is strictly below the Arrhenius law outside it. -/
theorem crossing_confined {α a b σsq : ℝ} (hα : 0 < α) (hσ : 0 < σsq)
    (h : arrheniusTime α σsq ≤ freeDiffusionTime σsq a b) :
    8 * α ^ 2 < (b - a) ^ 2 * σsq ∧ 4 * σsq ≤ (b - a) ^ 2 := by
  constructor
  · by_contra hcon
    exact absurd (freeDiffusionTime_lt_arrheniusTime_of_small_variance hα hσ (not_lt.mp hcon))
      (not_lt.mpr h)
  · by_contra hcon
    have hbig : (b - a) ^ 2 / 4 < σsq := by
      have := not_le.mp hcon
      linarith
    exact absurd (freeDiffusionTime_lt_arrheniusTime_of_large_variance hα hbig) (not_lt.mpr h)

/-- **A crossing needs a small barrier.** The window of `crossing_confined` is non-empty only
when `32α² < (b-a)⁴`; equivalently, the two laws can meet only if the barrier action `α` is
small compared with the square of the escape distance. For a barrier that is not small in this
sense the Arrhenius prediction is strictly the larger at every noise level, which is the precise
sense in which the polynomial law is a large-noise phenomenon. The condition is necessary but
not sharp: the exact criterion is `8eα ≤ (b-a)²` (`exists_crossing_iff`), and it is strictly
stronger — `quartic_lt_of_crossing_criterion` derives `32α² < (b-a)⁴` from it, so this theorem's
conclusion follows from the sharp criterion and not conversely. -/
theorem crossing_requires_small_barrier {α a b σsq : ℝ} (hα : 0 < α) (hσ : 0 < σsq)
    (h : arrheniusTime α σsq ≤ freeDiffusionTime σsq a b) : 32 * α ^ 2 < (b - a) ^ 4 := by
  obtain ⟨h1, h2⟩ := crossing_confined hα hσ h
  nlinarith [sq_nonneg (b - a), sq_nonneg ((b - a) ^ 2)]

/-- **The two regimes are different laws.** The Arrhenius time is not `O((σ²)⁻¹)` as the
diffusion tends to zero, so the escape law really does change at the crossover rather than being
the same law written two ways. Combined with `freeExitTime_isTheta_inv`, which puts the
free-diffusion time in `Θ((σ²)⁻¹)` on every filter, this separates the two regimes. -/
theorem arrheniusTime_not_isBigO_inv {α : ℝ} (hα : 0 < α) :
    ¬(fun σsq : ℝ => arrheniusTime α σsq) =O[𝓝[>] (0 : ℝ)] fun σsq : ℝ => σsq⁻¹ := by
  intro hO
  obtain ⟨c, hc⟩ := Asymptotics.isBigO_iff.mp hO
  have htend : Filter.Tendsto (fun σsq : ℝ => arrheniusTime α σsq * σsq) (𝓝[>] (0 : ℝ))
      Filter.atTop := by
    refine Filter.tendsto_atTop_mono' _ ?_
      (Filter.Tendsto.const_mul_atTop (by positivity : (0 : ℝ) < 2 * α ^ 2)
        tendsto_inv_nhdsGT_zero)
    filter_upwards [self_mem_nhdsWithin] with σsq hσmem
    have hσ : 0 < σsq := Set.mem_Ioi.mp hσmem
    have hσ' : σsq ≠ 0 := ne_of_gt hσ
    have hq : 1 + 2 * α / σsq + (2 * α / σsq) ^ 2 / 2 ≤ Real.exp (2 * α / σsq) :=
      Real.quadratic_le_exp_of_nonneg (by positivity)
    have hid : (1 + 2 * α / σsq + (2 * α / σsq) ^ 2 / 2) * σsq - 2 * α ^ 2 * σsq⁻¹
        = σsq + 2 * α := by
      field_simp
      ring
    have hstep : 2 * α ^ 2 * σsq⁻¹ ≤ (1 + 2 * α / σsq + (2 * α / σsq) ^ 2 / 2) * σsq := by
      nlinarith [hid]
    have hmul : (1 + 2 * α / σsq + (2 * α / σsq) ^ 2 / 2) * σsq ≤ Real.exp (2 * α / σsq) * σsq :=
      mul_le_mul_of_nonneg_right hq hσ.le
    rw [arrheniusTime]
    linarith
  have hmem : ∀ᶠ σsq : ℝ in 𝓝[>] (0 : ℝ), (0 : ℝ) < σsq := self_mem_nhdsWithin
  obtain ⟨σsq, hσ, hbound, hgt⟩ :=
    (hmem.and (hc.and (htend.eventually_gt_atTop c))).exists
  have hnorm1 : ‖arrheniusTime α σsq‖ = arrheniusTime α σsq := by
    rw [arrheniusTime, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  have hnorm2 : ‖σsq⁻¹‖ = σsq⁻¹ := by
    rw [Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hσ)]
  rw [hnorm1, hnorm2] at hbound
  have hkey : arrheniusTime α σsq * σsq ≤ c := by
    have := mul_le_mul_of_nonneg_right hbound hσ.le
    rwa [mul_assoc, inv_mul_cancel₀ (ne_of_gt hσ), mul_one] at this
  linarith

/-- **A large barrier rules out any crossing.** When the barrier action is not small compared
with the square of the escape distance, `(b-a)⁴ ≤ 32α²`, the Arrhenius time strictly exceeds the
free-diffusion time at *every* noise level. Contrapositive of `crossing_requires_small_barrier`,
stated positively because this is the form the scope discussion needs. The hypothesis is
sufficient but not necessary: by `exists_crossing_iff` there is already no crossing whenever
`(b-a)² < 8eα`, which this hypothesis implies (`crossing_criterion_of_large_barrier`) and which
holds in strictly more cases. -/
theorem no_crossing_of_large_barrier {α a b σsq : ℝ} (hα : 0 < α) (hσ : 0 < σsq)
    (h : (b - a) ^ 4 ≤ 32 * α ^ 2) : freeDiffusionTime σsq a b < arrheniusTime α σsq := by
  by_contra hcon
  exact absurd (crossing_requires_small_barrier hα hσ (not_lt.mp hcon)) (not_lt.mpr h)

/-- `e·y ≤ exp y` for every real `y`: the exponential lies above the line through the origin of
slope `e`, which is tangent to it at `y = 1`. This is the elementary fact behind the sharp
crossing criterion below. -/
theorem exp_one_mul_le_exp (y : ℝ) : Real.exp 1 * y ≤ Real.exp y := by
  have h : y ≤ Real.exp (y - 1) := by
    have := Real.add_one_le_exp (y - 1)
    linarith
  calc Real.exp 1 * y ≤ Real.exp 1 * Real.exp (y - 1) :=
        mul_le_mul_of_nonneg_left h (Real.exp_pos 1).le
    _ = Real.exp y := by
        rw [← Real.exp_add]
        congr 1
        ring

/-- The sharp criterion `8eα ≤ (b-a)²` of `exists_crossing_iff` implies the quartic necessary
condition `32α² < (b-a)⁴` of `crossing_requires_small_barrier`, so the former is strictly the
stronger of the two. The docstrings of those two theorems previously asserted this comparison
without proving it; it is proved here, from `e > 2.7182818283`. -/
theorem quartic_lt_of_crossing_criterion {α a b : ℝ} (hα : 0 < α)
    (h : 8 * Real.exp 1 * α ≤ (b - a) ^ 2) : 32 * α ^ 2 < (b - a) ^ 4 := by
  have he : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  have hu : 21 * α ≤ (b - a) ^ 2 := by nlinarith [mul_pos (sub_pos.mpr he) hα]
  nlinarith [hu, hα, sq_nonneg ((b - a) ^ 2 - 21 * α)]

/-- The contrapositive of `quartic_lt_of_crossing_criterion`: a barrier large in the quartic
sense already fails the sharp criterion, which is the comparison the docstring of
`no_crossing_of_large_barrier` states. -/
theorem crossing_criterion_of_large_barrier {α a b : ℝ} (hα : 0 < α)
    (h : (b - a) ^ 4 ≤ 32 * α ^ 2) : (b - a) ^ 2 < 8 * Real.exp 1 * α := by
  by_contra hcon
  exact absurd (quartic_lt_of_crossing_criterion hα (not_lt.mp hcon)) (not_lt.mpr h)

/-- **The sharp criterion for the free-diffusion time to reach the Arrhenius time.** The
free-diffusion time is *at least* the Arrhenius time at some positive diffusion if and only if
`8eα ≤ (b-a)²`; and when it is, the crossover diffusion `σ² = 2α` of `metastable_window` already
witnesses that inequality. Both directions are the tangency `e·y ≤ exp y` at `y = 2α/σ² = 1`.

The conclusion is an inequality and not an equality, and the distinction is not cosmetic: at
`σ² = 2α` the two laws take the values `e` and `(b-a)²/(8α)`, which coincide only in the boundary
case `8eα = (b-a)²`. The statement that the two closed forms are genuinely *equal* somewhere,
under the same criterion, is `exists_equal_crossing_iff`; the diffusion level it produces is in
general not `2α`. This criterion is sharp where `crossing_requires_small_barrier` (necessity of
`32α² < (b-a)⁴`) and `no_crossing_of_large_barrier` (sufficiency of `(b-a)⁴ ≤ 32α²` for no
crossing) are one-sided only — `quartic_lt_of_crossing_criterion` derives the quartic condition
from this one. -/
theorem exists_crossing_iff {α a b : ℝ} (hα : 0 < α) :
    (∃ σsq : ℝ, 0 < σsq ∧ arrheniusTime α σsq ≤ freeDiffusionTime σsq a b)
      ↔ 8 * Real.exp 1 * α ≤ (b - a) ^ 2 := by
  constructor
  · rintro ⟨σsq, hσ, h⟩
    have hσ' : σsq ≠ 0 := ne_of_gt hσ
    have h1 : Real.exp 1 * (2 * α / σsq) ≤ arrheniusTime α σsq := exp_one_mul_le_exp _
    have h2 : Real.exp 1 * (2 * α / σsq) ≤ (b - a) ^ 2 / (4 * σsq) := by
      rw [freeDiffusionTime] at h
      linarith
    have h4 := mul_le_mul_of_nonneg_right h2 hσ.le
    have e1 : Real.exp 1 * (2 * α / σsq) * σsq = Real.exp 1 * (2 * α) := by
      field_simp
    have e2 : (b - a) ^ 2 / (4 * σsq) * σsq = (b - a) ^ 2 / 4 := by
      field_simp
    rw [e1, e2] at h4
    linarith
  · intro h
    refine ⟨2 * α, by positivity, ?_⟩
    rw [arrheniusTime_at_crossover hα, freeDiffusionTime, le_div_iff₀ (by positivity)]
    linarith

/-- Both closed forms are continuous away from zero diffusion, so their difference is continuous
on any closed interval of positive diffusion levels. This is the hypothesis the intermediate
value theorem needs in the two crossing theorems below. -/
theorem continuousOn_freeDiffusionTime_sub_arrheniusTime (α a b : ℝ) {c d : ℝ} (hc : 0 < c) :
    ContinuousOn (fun σsq : ℝ => freeDiffusionTime σsq a b - arrheniusTime α σsq)
      (Set.Icc c d) := by
  intro x hx
  have hx0 : x ≠ 0 := ne_of_gt (lt_of_lt_of_le hc hx.1)
  have hden : (4 : ℝ) * x ≠ 0 := mul_ne_zero (by norm_num) hx0
  have hA : ContinuousAt (fun σ : ℝ => freeDiffusionTime σ a b) x := by
    simp only [freeDiffusionTime]
    exact continuousAt_const.div (continuousAt_const.mul continuousAt_id) hden
  have hB : ContinuousAt (fun σ : ℝ => arrheniusTime α σ) x := by
    simp only [arrheniusTime]
    exact (Real.continuous_exp.continuousAt).comp (continuousAt_const.div continuousAt_id hx0)
  exact (hA.sub hB).continuousWithinAt

/-- **The two closed forms are equal somewhere, under exactly the same criterion.**
`exists_crossing_iff` gives an inequality; this gives the equality it is usually read as. The
free-diffusion time equals the Arrhenius time at some positive diffusion **if and only if**
`8eα ≤ (b-a)²`. One direction is immediate from `exists_crossing_iff`; the other runs the
intermediate value theorem on the difference between `σ² = 2α`, where the free-diffusion time is
at least the Arrhenius time, and `σ² = (b-a)²/4 + 1`, where
`freeDiffusionTime_lt_arrheniusTime_of_large_variance` puts it strictly below. The statement
asserts only that a meeting point exists; the construction places it in
`[2α, (b-a)²/4 + 1]`, and it is in general not `2α` itself. -/
theorem exists_equal_crossing_iff {α a b : ℝ} (hα : 0 < α) :
    (∃ σsq : ℝ, 0 < σsq ∧ freeDiffusionTime σsq a b = arrheniusTime α σsq)
      ↔ 8 * Real.exp 1 * α ≤ (b - a) ^ 2 := by
  constructor
  · rintro ⟨σsq, hσ, heq⟩
    exact (exists_crossing_iff hα).mp ⟨σsq, hσ, le_of_eq heq.symm⟩
  · intro h
    have he : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
    have hu : 21 * α ≤ (b - a) ^ 2 := by nlinarith [mul_pos (sub_pos.mpr he) hα]
    set M : ℝ := (b - a) ^ 2 / 4 + 1 with hMdef
    have h2αM : 2 * α < M := by rw [hMdef]; linarith
    have hg0 : 0 ≤ freeDiffusionTime (2 * α) a b - arrheniusTime α (2 * α) := by
      rw [arrheniusTime_at_crossover hα, freeDiffusionTime, sub_nonneg,
        le_div_iff₀ (by positivity)]
      linarith
    have hgM : freeDiffusionTime M a b - arrheniusTime α M < 0 := by
      have := freeDiffusionTime_lt_arrheniusTime_of_large_variance (a := a) (b := b) hα
        (show (b - a) ^ 2 / 4 < M by rw [hMdef]; linarith)
      linarith
    obtain ⟨s, hsmem, hs0⟩ : ∃ s ∈ Set.Icc (2 * α) M,
        freeDiffusionTime s a b - arrheniusTime α s = 0 :=
      intermediate_value_Icc' h2αM.le
        (continuousOn_freeDiffusionTime_sub_arrheniusTime α a b (by positivity))
        ⟨by linarith, by linarith⟩
    exact ⟨s, lt_of_lt_of_le (by positivity) hsmem.1, sub_eq_zero.mp hs0⟩

/-- **The two laws do not cross exactly once.** At `α = 1/2` and `(a,b) = (0,4)` the sign of
`freeDiffusionTime - arrheniusTime` is negative at `σ² = 1/8`, positive at `σ² = 1`, and negative
again at `σ² = 5`. Two sign changes, so the claim that the free-diffusion law overtakes the
Arrhenius law once and for all as the noise falls is false: the exponential eventually outruns
any polynomial in `1/σ²` at small noise as well as sitting above it at large noise. The step
from these three inequalities to two actual crossings is the intermediate value theorem and is
carried out in `exists_two_crossings`, so nothing here is left to the reader. What survives is
`crossing_confined` — every crossing lies in a bounded window — together with the unique
crossing of the *exponent* through one at `σ² = 2α` (`metastable_window`), which is the
statement `rem:fpt-scope` actually needs. -/
theorem crossing_not_unique :
    freeDiffusionTime (1 / 8) 0 4 < arrheniusTime (1 / 2) (1 / 8) ∧
      arrheniusTime (1 / 2) 1 < freeDiffusionTime 1 0 4 ∧
      freeDiffusionTime 5 0 4 < arrheniusTime (1 / 2) 5 := by
  refine ⟨?_, ?_, ?_⟩
  · exact freeDiffusionTime_lt_arrheniusTime_of_small_variance (by norm_num) (by norm_num)
      (by norm_num)
  · have h1 : arrheniusTime (1 / 2 : ℝ) 1 = Real.exp 1 := by
      rw [arrheniusTime]
      norm_num
    have h2 : freeDiffusionTime (1 : ℝ) 0 4 = 4 := by
      rw [freeDiffusionTime]
      norm_num
    rw [h1, h2]
    exact Real.exp_one_lt_d9.trans (by norm_num)
  · exact freeDiffusionTime_lt_arrheniusTime_of_large_variance (by norm_num) (by norm_num)

/-- **The hypothesis of `crossing_confined` is satisfiable, so those results are not vacuous.**
At barrier `α = 1/2` on the interval `(0,4)` the Arrhenius time at `σ² = 1` is `e`, which is
below the free-diffusion time `4` there. The confinement window of `crossing_confined` and the
necessary condition of `crossing_requires_small_barrier` are therefore statements about a
situation that occurs, not conclusions drawn from empty hypotheses. -/
theorem exists_arrheniusTime_le_freeDiffusionTime :
    ∃ α a b σsq : ℝ, 0 < α ∧ 0 < σsq ∧ arrheniusTime α σsq ≤ freeDiffusionTime σsq a b := by
  refine ⟨1 / 2, 0, 4, 1, by norm_num, by norm_num, ?_⟩
  have h1 : arrheniusTime (1 / 2 : ℝ) 1 = Real.exp 1 := by
    rw [arrheniusTime]
    norm_num
  have h2 : freeDiffusionTime (1 : ℝ) 0 4 = 4 := by
    rw [freeDiffusionTime]
    norm_num
  rw [h1, h2]
  exact (Real.exp_one_lt_d9.trans (by norm_num)).le

/-- **Two crossings, not one.** For barrier `α = 1/2` on the interval `(0,4)` there are two
diffusion levels at which the free-diffusion time and the Arrhenius time are *equal*: one in
`(1/8, 1)` and one in `(1, 5)`, hence two distinct ones, since the first is below `1` and the
second above it. The three sign facts of `crossing_not_unique` become genuine crossings by the
intermediate value theorem, applied to the difference of the two laws on `[1/8, 5]`, where both
are continuous. So the "exactly once" reading of the crossover is refuted here by a theorem
exhibiting two crossings, not by an argument about a theorem. -/
theorem exists_two_crossings :
    ∃ s t : ℝ, s ∈ Set.Ioo (1 / 8 : ℝ) 1 ∧ t ∈ Set.Ioo (1 : ℝ) 5 ∧
      freeDiffusionTime s 0 4 = arrheniusTime (1 / 2) s ∧
      freeDiffusionTime t 0 4 = arrheniusTime (1 / 2) t := by
  obtain ⟨hlow, hmid, hhigh⟩ := crossing_not_unique
  have hcont : ContinuousOn (fun σ : ℝ => freeDiffusionTime σ 0 4 - arrheniusTime (1 / 2) σ)
      (Set.Icc (1 / 8 : ℝ) 5) :=
    continuousOn_freeDiffusionTime_sub_arrheniusTime (1 / 2) 0 4 (by norm_num)
  obtain ⟨s, hsmem, hs0⟩ : ∃ s ∈ Set.Icc (1 / 8 : ℝ) 1,
      freeDiffusionTime s 0 4 - arrheniusTime (1 / 2) s = 0 := by
    have hsub : Set.Icc (1 / 8 : ℝ) 1 ⊆ Set.Icc (1 / 8 : ℝ) 5 :=
      Set.Icc_subset_Icc le_rfl (by norm_num)
    exact intermediate_value_Icc (by norm_num : (1 / 8 : ℝ) ≤ 1) (hcont.mono hsub)
      ⟨by linarith, by linarith⟩
  obtain ⟨t, htmem, ht0⟩ : ∃ t ∈ Set.Icc (1 : ℝ) 5,
      freeDiffusionTime t 0 4 - arrheniusTime (1 / 2) t = 0 := by
    have hsub : Set.Icc (1 : ℝ) 5 ⊆ Set.Icc (1 / 8 : ℝ) 5 :=
      Set.Icc_subset_Icc (by norm_num) le_rfl
    exact intermediate_value_Icc' (by norm_num : (1 : ℝ) ≤ 5) (hcont.mono hsub)
      ⟨by linarith, by linarith⟩
  have hs1 : s < 1 := by
    refine lt_of_le_of_ne hsmem.2 ?_
    intro hEq
    rw [hEq] at hs0
    linarith
  have hs8 : 1 / 8 < s := by
    refine lt_of_le_of_ne hsmem.1 ?_
    intro hEq
    rw [← hEq] at hs0
    linarith
  have ht1 : 1 < t := by
    refine lt_of_le_of_ne htmem.1 ?_
    intro hEq
    rw [← hEq] at ht0
    linarith
  have ht5 : t < 5 := by
    refine lt_of_le_of_ne htmem.2 ?_
    intro hEq
    rw [hEq] at ht0
    linarith
  exact ⟨s, t, ⟨hs8, hs1⟩, ⟨ht1, ht5⟩, sub_eq_zero.mp hs0, sub_eq_zero.mp ht0⟩

end IcnnLift
