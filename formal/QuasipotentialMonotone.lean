import Mathlib

/-!
# (FW-2) Quasipotential monotonicity in the diffusion, in one dimension

This file machine-checks FW-2 of `docs/design/lemma1_rederivation.md` in the one-dimensional
reduction that the paper's Corollary actually uses.  Three versions of the statement appear in
that note and they are not equivalent.  The sketch of section 7 (line 202) is superseded by the
full draft of section 8 ("FW-2 — quasipotential monotonicity in the diffusion"), which is in turn
amended by FW-2' of section 11.3, where the strictness hypothesis is moved off the magnitude
channel `H V H` and onto the coupling channel `C = H Σ_slack + Σ_slackᵀ H` alone, because
section 10's matched-noise bracket found the magnitude channel empirically inert.  Both the
section-8 form and the section-11.3 form are formalized here, and it is worth being exact about
how they differ, because on one axis 11.3 is not a refinement of 8 but a strengthening of its
hypothesis.  Section 8 assumes the strictness input directly, as a lower bound `δ` on the
reciprocal gap `uᵀ(Σ_B⁻¹ - Σ_A⁻¹)u ≥ δ‖u‖²`; section 11.3 replaces `δ` by an upper bound `Λ` on
the baseline diffusion together with a lower bound `ε` on the excess, and *derives*
`δ = ε/(Λ(Λ+ε))` from the pair.  The pair implies the reciprocal gap and not conversely — a
positive reciprocal gap survives a baseline diffusion tending to zero, which is the shoulder
regime `σ_dir² = s²σ_obj² → 0` of FW-3, whereas a positive excess `σ_lift² - σ_dir²` need not.
So the theorems are stated at the reciprocal-gap level
(`quasipotentialIncrement_le_sub_reciprocalGap`, `fwAction_le_sub_reciprocalGap`,
`fwQuasipotential_le_sub_gap_of_nearMinimizer_reciprocalGap`) and FW-2''s `(Λ, ε)` forms are
derived from them, with the modulus `δ = ε/(Λ(Λ+ε))` proved rather than quoted
(`reciprocal_gap_of_diffusion_excess`) and shown to be the largest one those hypotheses admit
(`reciprocal_gap_of_diffusion_excess_optimal`).

The mathematical content, in one dimension, is that the effective quasipotential of the
bias-channel Itô diffusion `dw = -L̃'(w) dt + √η σ_eff(w) dB` of section 3,

  `U_eff(w) = ∫^w L̃'(u) / σ_eff²(u) du`,                                                   (*)

is pointwise **anti-monotone in the diffusion** on any interval on which the drift climbs
(`L̃' ≥ 0`): raising `σ_eff²` at every point of the band cannot raise the barrier's effective
action, and strictly lowers it as soon as the raise is strict where the climb is strict.  This
is elementary calculus once (*) is granted, and it is the whole of FW-2 in the scalar reduction.

Three things are proved beyond the bare inequality, and they are what make the file useful to
the paper rather than decorative.

First, a **quantitative gap**: if on a subinterval `[c,d]` of the band the climb is at least `m`,
the baseline diffusion is at most `Λ`, and the lift's excess is at least `ε`, then the effective
action drops by at least `(d - c) · m ε / (Λ (Λ + ε))`.  The modulus `ε / (Λ (Λ + ε))` is exactly
FW-2''s `δ`, here derived rather than quoted, and — as FW-2' observes — it needs an upper bound
on the baseline diffusion only, never on the lift's.  The constant is exact rather than the
artifact of a slack estimate, and this is checked twice over: the extremal admissible datum
attains it with equality (`reciprocal_gap_of_diffusion_excess_sharp`), no larger uniform modulus
is admissible (`reciprocal_gap_of_diffusion_excess_optimal`), and on the explicit instance of
`liftQuasipotentialIncrement_le_sub_gap_witness` the promised drop is attained by the two
quasipotential increments themselves.

Second, the statement is lifted off the closed-form (*) and onto the variational object.
`fwAction` is the one-dimensional Freidlin–Wentzell action of a path, `fwQuasipotential` is its
infimum over an abstract space `Γ` of admissible paths, and
`fwQuasipotential_le_sub_gap_of_nearMinimizer_reciprocalGap`,
`fwQuasipotential_le_sub_gap_of_nearMinimizer_bandOccupancy` and
`fwQuasipotential_lt_of_nearMinimizer_bandOccupancy` prove FW-2 and FW-2' for that infimum:
pointwise domination of the actions plus an occupancy time `τ` on the band gives a quasipotential
smaller by the explicit gap `nrm · δ c² τ` — at the design note's normalization `nrm = 1/4` the
`(δ c²/4) τ` that section 8's proof writes down — and strictly smaller as soon as every factor
of that gap is positive.  The occupancy input is carried as a named hypothesis, not proved, and
it is section 8's own strictness hypothesis rather than any theorem of the note.  Its three
clauses have three different sources: the time set of measure `τ` in the band is FW-3a of
section 11.1, whose deterministic transit-time core `BandOccupancy.lean` proves while carrying
the action-to-speed implication as a named hypothesis of its own; the diffusion clause on that
set is section 8's reciprocal gap, supplied through FW-3b's ellipticity (section 11.2) together
with the FW-1/(A6) excess; and the velocity floor `c ≤ |γ' + L̃'(γ)|` is proved nowhere in the
note — its section-8 physical note argues it physically, and a formal discharge would need FW-3a
rerun on the upward-moving time set, where `u = γ' + L̃'(γ) ≥ L̃' ≥ m` as soon as the climb has a
floor.

The quantifier on that occupancy hypothesis is the one place where a plausible-looking statement
would have been materially weaker than the note's, so it is worth naming.  Section 8 asks for the
band time only of paths that nearly minimize the *baseline* action, and FW-3a supplies it only
for paths of action at most a stated `S̄`, with an occupancy time `τ(w, Λ, S̄, M)` that *decreases*
as `S̄` grows.  Over an unrestricted path space there is therefore no single positive `τ` at all,
and a theorem demanding one of every path could not be fed by FW-3a.  The statements named
`*_nearMinimizer_*` ask for occupancy only on the sublevel set `{γ : S_dir[γ] ≤ V_dir + ρ}`, which
is exactly FW-3a's hypothesis at `S̄ = V_dir + ρ`; the all-paths versions are kept as corollaries
of them and are labelled as the stronger-hypothesis, weaker statements they are.

Third, `quasipotentialIncrement_le_fwAction` connects the two levels: for **every** continuously
differentiable path the one-dimensional Freidlin–Wentzell action dominates `4 · nrm` times the
increment (*), by the completion of the square `(γ' + L̃')² = (γ' - L̃')² + 4 γ' L̃'` and the
change-of-variables formula.  This proves one half of the classical identification of (*) with
the one-dimensional quasipotential.

## What is NOT proved here, and why

* **The multi-dimensional statement is not established.**  In `d > 1` the quasipotential is an
  infimum of a path integral of `uᵀ Σ(γ)⁻¹ u`, and the pointwise step needs anti-monotonicity of
  the matrix inverse in the Loewner order — the auxiliary lemma of section 8, which lives in
  `InversePSDAntitone.lean` — followed by the projection argument of FW-4 in `FWReduction.lean`.
  The variational half of the argument *is* dimension-free and is proved here in that generality
  (`quasipotential_le_of_action_le`, `quasipotential_lt_of_action_le_sub` and, in the form section
  8 actually argues, `quasipotential_le_sub_of_action_le_sub_nearMin` and
  `quasipotential_lt_of_action_le_sub_nearMin`, all stated for an arbitrary index type), and it is
  the only part of the argument that transfers verbatim.  What `d > 1` still needs is the
  pointwise integrand comparison in the Loewner order and then, on top of it, a multi-dimensional
  `fwAction` with its own integration step: the proofs of `fwAction_antitone_in_diffusion` and
  `fwAction_le_sub_reciprocalGap` would be repeated with `uᵀΣ(γ)⁻¹u` in place of `u²/σ²(γ)`, which
  is mechanical but is not done here and should not be assumed done.  Nothing in this file asserts
  a multi-dimensional conclusion.
* **The identification of (*) with the quasipotential is proved in one direction only.**
  `quasipotentialIncrement_le_fwAction` gives `4·nrm·ΔU_eff ≤ S[γ]` for every path.  The reverse
  inequality requires exhibiting near-optimal paths, i.e. solving `γ' = L̃'(γ)` and passing to a
  limit as the transit time diverges; that is an ODE existence-and-limit argument and is not done
  here.
* **Nothing about exit times is proved.**  This mathlib has no Itô calculus, no SDE, no
  Fokker–Planck and no exit-time theory, so the step from a quasipotential comparison to a mean
  first passage time comparison cannot be a theorem here.  `kramersFactor_lt_of_quasipotential_lt`
  compares the Arrhenius *exponents* only; converting an exponent into an exit time is the named
  hypothesis of `KramersExitTime.lean`.
* **The stationary density (*) itself is not derived here.**  That it is the stationary density of
  the state-dependent-diffusion SDE is section 3's input and belongs to `StationaryDensity.lean`.
  This file takes (*) as the definition of the effective quasipotential increment and proves the
  comparison statements about it.

## Results
* `quasipotentialIncrement` — the increment of `U_eff` in (*), `∫_a^b L̃'(u)/σ²(u) du`.
* `reciprocal_gap_of_diffusion_excess` — the scalar form of FW-2''s modulus `δ = ε/(Λ(Λ+ε))`.
* `reciprocal_gap_of_diffusion_excess_sharp` — the modulus is attained on an admissible datum.
* `reciprocal_gap_of_diffusion_excess_optimal` — hence no larger uniform modulus is admissible.
* `quasipotentialIncrement_antitone_in_diffusion` — FW-2, monotone form, in one dimension.
* `quasipotentialIncrement_le_sub_reciprocalGap` — FW-2 with section 8's own hypothesis, a lower
  bound `δ` on `1/σ_dir² - 1/σ_lift²`, and the gap `(d-c)·m δ`.
* `quasipotentialIncrement_le_sub_gap` — its FW-2' specialization, with the derived gap
  `(d-c)·m ε/(Λ(Λ+ε))`.
* `quasipotentialIncrement_strictAnti_in_diffusion` — FW-2, strict form, under continuity.
* `liftQuasipotentialIncrement_lt_direct` — the paper's corollary, with (A6) explicit.
* `liftQuasipotentialIncrement_le_sub_gap` — the same with the modulus of (A6) quantified.
* `kramersFactor_lt_of_quasipotential_lt` — the Arrhenius exponent comparison.
* `quasipotential_le_of_action_le`, `quasipotential_le_sub_of_action_le_sub`,
  `quasipotential_lt_of_action_le_sub` — the variational step, in any dimension, under a gap
  holding along every path.
* `quasipotential_le_sub_of_action_le_sub_nearMin`, `quasipotential_lt_of_action_le_sub_nearMin` —
  the same with the gap required only of near-minimizing paths, which is how section 8 argues it.
* `fwAction`, `fwQuasipotential` — the one-dimensional Freidlin–Wentzell action and its infimum.
* `fwAction_nonneg`, `bddBelow_range_fwAction` — the action is non-negative, so the infimum
  defining `fwQuasipotential` is a genuine greatest lower bound.
* `fwAction_integrand_intervalIntegrable` — continuity of the path, its velocity and the
  coefficients discharges the per-path integrability hypotheses of the comparison theorems.
* `fwAction_antitone_in_diffusion` — FW-2's monotone step for a single path.
* `fwAction_le_sub_reciprocalGap` — FW-2's strictness clause for a single path, with section 8's
  own reciprocal-gap hypothesis on a measurable time set; `fwAction_le_sub_gap` is its FW-2'
  specialization, in which `δ = ε/(Λ(Λ+ε))` is derived from `(Λ, ε)` instead of assumed.
* `quasipotentialIncrement_le_fwAction` — (*) is a lower bound for the action of every path.
* `fwQuasipotential_ge_quasipotentialIncrement` — hence for the quasipotential itself.
* `fwQuasipotential_le_of_diffusion_le` — FW-2 for the variational quasipotential.
* `fwQuasipotential_le_sub_gap_of_nearMinimizer_reciprocalGap` — FW-2 for the variational
  quasipotential with section 8's own strictness hypothesis, and with the occupancy required only
  of near-minimizing paths, as FW-3a can supply it.
* `fwQuasipotential_le_sub_gap_of_nearMinimizer_bandOccupancy`,
  `fwQuasipotential_lt_of_nearMinimizer_bandOccupancy` — its FW-2' specialization and the strict
  form, with the same quantifier on the occupancy hypothesis.
* `fwQuasipotential_le_sub_gap_of_bandOccupancy`, `fwQuasipotential_lt_of_bandOccupancy` — the
  corollaries of those two under the stronger all-paths occupancy hypothesis.
* `liftQuasipotentialIncrement_lt_direct_witness`,
  `liftQuasipotentialIncrement_le_sub_gap_witness`,
  `fwQuasipotential_lt_of_nearMinimizer_bandOccupancy_witness` — explicit instances showing the
  hypothesis bundles of the three headline theorems are satisfiable, so none is vacuous; the two
  quantitative ones attain their gap with equality.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses carried, and what each would take to discharge:

* **Integrability** of the two integrands, or continuity on the closed band, in every statement
  about (*) — and of the two action integrands, path by path (`hi1`/`hi2`), in every
  `fwAction_*` and `fwQuasipotential_*` comparison.  These are genuine analytic side conditions
  of an interval integral and there is no way to remove them.  On the path side they are
  moreover load-bearing in a way specific to the formalization: Lean's Bochner integral assigns
  a non-integrable integrand the value `0`, not `+∞`, so a path along which the *baseline*
  integrand fails to be integrable would be recorded by `fwAction` as a zero-action path,
  dragging the baseline infimum down and falsifying the comparison; the hypotheses `hi1`/`hi2`
  are exactly the restriction of `Γ` to path families on which `fwQuasipotential` is the
  Freidlin–Wentzell infimum rather than that artifact.  All of them are implied by continuity:
  of `L̃'` and `σ_eff²` on the closed band for (*), which is what (A1)'s regularity clause and
  (A2)'s two-sided bound on `ψ'` (section 11.2) are there to supply, and of the path and its
  recorded velocity for the action integrands, which is `fwAction_integrand_intervalIntegrable`
  below.
* **Uniform ellipticity of the baseline**, `0 < σ_dir` on the band, and additionally
  `σ_dir ≤ Λ` in the FW-2' forms of the gap statements.  This is FW-3b of section 11.2, which
  resolves it from (A2)'s `ψ'(θ̃) ≈ σ_s I_d`; it is a hypothesis here because (A2) is a modelling
  assumption about the positivity map, not a theorem.  The upper bound `Λ` is needed only to *derive*
  the modulus: the `*_reciprocalGap` statements, which are section 8's own form, assume the
  reciprocal gap `δ ≤ 1/σ_dir² - 1/σ_lift²` instead and use no upper bound at all.
* **(A6), the coupling excess**, `0 < exc` (or `κ ≤ exc`) on the subinterval `[c,d]` of the
  band.  Section 11.3 is explicit that this is an assumption with empirical content and not a
  consequence of FW-1: `H V H` is positive semidefinite always, but
  `C = H Σ_slack + Σ_slackᵀ H` is not.  It is therefore stated as its own hypothesis and never
  folded into the conclusion.  Distinct from it is the weak ordering `0 ≤ exc` across the whole
  band (`hexc` of the two corollaries): that is FW-1's diffusion ordering with the `T3`
  remainder controlled by (A4'-q), the subject of `DiffusionOrdering.lean`, and it is a
  hypothesis here for the same reason.
* **The climb**, `0 ≤ L̃'` on the band and `0 < L̃'` where strictness is claimed.  This is the
  statement that the segment under comparison is the uphill part of the escape; it is false on
  the downhill side, where the inequality reverses, so it cannot be dropped.
* **Band occupancy** `hocc` in the five `fwQuasipotential_*` gap theorems.  This is section 8's
  strictness hypothesis itself, and no single lemma of the development discharges it: its
  time-in-band clause is FW-3a of section 11.1, whose deterministic core `BandOccupancy.lean`
  proves while carrying the action-to-speed implication as its own named hypothesis; its
  diffusion clause is FW-3b's ellipticity and the FW-1/(A6) excess restricted to the times the
  path spends in the band; and its velocity-floor clause `c ≤ |γ' + L̃'(γ)|` is proved nowhere in
  the note, whose section-8 physical note argues it physically — a formal discharge would need
  FW-3a restricted to the upward-moving times, where `u ≥ L̃' ≥ m`.  In the three
  `*_nearMinimizer_*` forms the whole bundle is asked only of paths whose baseline action is
  within `ρ` of the baseline quasipotential, which is precisely FW-3a's own hypothesis
  `S[γ] ≤ S̄` at `S̄ = V_dir + ρ`; in the two all-paths forms it is asked of every path, which
  FW-3a does not give, because its `τ` shrinks as the admissible action grows.  The all-paths
  forms are therefore usable only after `Γ` has been cut down to a sublevel set, and they are
  derived from the near-minimizer forms rather than the other way round.
* **Boundedness below of the two action families over the path space.**  Without it `⨅` is not
  the greatest lower bound but the junk value Lean's `sInf` returns, and the comparison says
  nothing.  In the one-dimensional statements it is *not* a hypothesis: `fwAction_nonneg` proves
  the action non-negative whenever the diffusion is positive along the path, and
  `bddBelow_range_fwAction` discharges the side condition from the hypotheses those theorems
  already carry.  The same hypotheses (`hnrm`, `hT`, `hin`, `hs1`) bound the *baseline* family
  below as well, so both infima appearing in those statements are genuine greatest lower bounds,
  even though only the lift's boundedness is consumed by the proofs.  Boundedness survives as an
  explicit hypothesis only in the dimension-free lemmas `quasipotential_le_of_action_le`,
  `quasipotential_le_sub_of_action_le_sub` and their `nearMin` companions, which know nothing
  about the shape of the action and are meant to be reused verbatim by `FWReduction.lean`; there
  it is asked of `SA` only, and a caller whose `SB` is unbounded below would get a true statement
  about that junk value rather than about an infimum, so `FWReduction.lean` must supply the
  baseline bound itself.
* **The path space `Γ`** is abstract.  Admissibility — endpoints, absolute continuity, remaining
  in the domain `D` — is encoded in the choice of `Γ`, so the theorems hold for whatever notion
  of admissible path the multi-dimensional development settles on.  Nothing is assumed about `Γ`
  beyond non-emptiness.  In particular the velocity family `vel` is **not** tied to the derivative
  of `traj` by any of the comparison theorems: they compare two actions of the same
  `(path, velocity, horizon)` triple, so the comparison is insensitive to that link, and it is
  the choice of `Γ` that has to impose it.  The one theorem that does need the link,
  `quasipotentialIncrement_le_fwAction`, states it as the hypothesis `hderiv`, and states it as a
  two-sided `HasDerivAt` on the closed interval `[0,T]` — slightly more than differentiability
  within `[0,T]`, and what the change-of-variables lemma used there
  (`intervalIntegral.integral_deriv_smul_comp'`) takes; mathlib's more primitive variant would
  allow one-sided derivatives on the interior at the cost of extra integrability bookkeeping,
  and neither reaches the absolutely continuous paths over which section 8 quantifies its
  integrand comparison.  The comparison theorems themselves impose no derivative link at all, so
  only this one identification lemma is confined to continuously differentiable paths.  The witness
  `fwQuasipotential_lt_of_nearMinimizer_bandOccupancy_witness` uses a path whose recorded velocity
  really is its derivative, so the non-vacuity it exhibits does not rest on the decoupling.
* **Non-vacuity.**  The hypothesis lists of the headline theorems are long enough that
  satisfiability is a real question, so it is answered in the file: the three `*_witness`
  theorems construct instances and compute both sides of each conclusion.  Two of the three
  conclusions come out as equalities, so the witnesses also certify that the gap constants are
  the exact ones and not weakened copies.
-/

open MeasureTheory Set

namespace IcnnLift

/-! ## The one-dimensional effective quasipotential -/

/-- The increment of the effective quasipotential `U_eff` of `docs/design/lemma1_rederivation.md`
section 3, equation (*): for the bias-channel Itô diffusion
`dw = -L̃'(w) dt + √η σ_eff(w) dB` with state-dependent diffusion `σ_eff²`, the stationary
density is `π(w) ∝ σ_eff⁻²(w) exp(-(2/η) U_eff(w))` with
`U_eff(w) = ∫^w L̃'(u)/σ_eff²(u) du`.  Only increments of `U_eff` are meaningful, so that is what
is defined; `Lp` is `L̃'` and `sigma2` is `σ_eff²`. -/
noncomputable def quasipotentialIncrement (Lp sigma2 : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ u in a..b, Lp u / sigma2 u

/-- The scalar form of the modulus `δ` that FW-2' of section 11.3 derives by congruence in the
Loewner order.  If the baseline diffusion `x` is positive and at most `Λ`, the lift's diffusion
`y` exceeds it by at least `ε`, and the climb `p` is at least `m ≥ 0`, then the pointwise
integrand of (*) drops by at least `m ε / (Λ (Λ + ε))`.  Note, as FW-2' does, that only an
upper bound on the **baseline** diffusion is used; the lift's diffusion is bounded below only. -/
theorem reciprocal_gap_of_diffusion_excess {p x y m lam eps : ℝ}
    (hm : 0 ≤ m) (hmp : m ≤ p) (hx : 0 < x) (heps : 0 ≤ eps) (hlam : 0 < lam)
    (hxlam : x ≤ lam) (hy : x + eps ≤ y) :
    m * eps / (lam * (lam + eps)) ≤ p / x - p / y := by
  have hp : 0 ≤ p := hm.trans hmp
  have hxe : 0 < x + eps := by linarith
  have hden : 0 < x * (x + eps) := by positivity
  have hden2 : 0 < lam * (lam + eps) := by positivity
  have hdd : x * (x + eps) ≤ lam * (lam + eps) := by nlinarith
  have h2 : p / y ≤ p / (x + eps) := div_le_div_of_nonneg_left hp hxe hy
  have h3 : p / x - p / (x + eps) = p * eps / (x * (x + eps)) := by field_simp; ring
  have h4 : m * eps / (lam * (lam + eps)) ≤ p * eps / (x * (x + eps)) := by
    rw [div_le_div_iff₀ hden2 hden]
    have hA : m * eps * (x * (x + eps)) ≤ p * eps * (x * (x + eps)) := by
      nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.mpr hmp) heps) hden.le]
    have hB : p * eps * (x * (x + eps)) ≤ p * eps * (lam * (lam + eps)) := by
      nlinarith [mul_nonneg (mul_nonneg hp heps) (sub_nonneg.mpr hdd)]
    linarith
  linarith [h3 ▸ h4]

/-- The modulus of `reciprocal_gap_of_diffusion_excess` is attained: at `p = m`, `x = Λ`,
`y = Λ + ε` — an instance that satisfies every hypothesis of that lemma as soon as `0 ≤ m` — the
inequality holds with equality.  The equation pins the constant itself: a formalization that had
silently weakened the modulus would still type-check as an inequality but would fail here. -/
theorem reciprocal_gap_of_diffusion_excess_sharp {m lam eps : ℝ}
    (hlam : 0 < lam) (heps : 0 ≤ eps) :
    m * eps / (lam * (lam + eps)) = m / lam - m / (lam + eps) := by
  have h1 : lam ≠ 0 := hlam.ne'
  have h2 : lam + eps ≠ 0 := (by positivity : (0:ℝ) < lam + eps).ne'
  field_simp
  ring

/-- **FW-2''s modulus is the largest one its hypotheses admit.**  Let `μ` be any number that lower
bounds the pointwise reciprocal gap `p/x - p/y` uniformly over the whole class of data the lemma
`reciprocal_gap_of_diffusion_excess` quantifies over — every climb `p ≥ m`, every baseline
`0 < x ≤ Λ`, every lift diffusion `y ≥ x + ε`.  Then `μ ≤ m ε / (Λ (Λ + ε))`.  So the modulus
this file carries into every gap statement is not the artifact of a slack estimate: no larger
uniform constant is available without strengthening the hypotheses, and FW-2''s `δ` cannot be
improved.  The extremal datum is the one exhibited by `reciprocal_gap_of_diffusion_excess_sharp`,
namely `p = m`, `x = Λ`, `y = Λ + ε`. -/
theorem reciprocal_gap_of_diffusion_excess_optimal {m lam eps mu : ℝ}
    (hlam : 0 < lam) (heps : 0 ≤ eps)
    (hmu : ∀ p x y : ℝ, m ≤ p → 0 < x → x ≤ lam → x + eps ≤ y → mu ≤ p / x - p / y) :
    mu ≤ m * eps / (lam * (lam + eps)) := by
  have h : mu ≤ m / lam - m / (lam + eps) := hmu m lam (lam + eps) le_rfl hlam le_rfl le_rfl
  rw [reciprocal_gap_of_diffusion_excess_sharp hlam heps]
  exact h

/-- **FW-2, monotone form, in one dimension.**  On an interval on which the drift climbs
(`0 ≤ L̃'`) and the baseline diffusion is positive, raising the diffusion pointwise cannot raise
the increment of the effective quasipotential.  This is the "integrand is monotone decreasing in
`Σ`" step of section 8's proof, in the scalar case where it is `a/y ≤ a/x` for `0 < x ≤ y`. -/
theorem quasipotentialIncrement_antitone_in_diffusion
    {Lp s1 s2 : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
    (hLp : ∀ u ∈ Icc a b, 0 ≤ Lp u)
    (hs1 : ∀ u ∈ Icc a b, 0 < s1 u)
    (hs12 : ∀ u ∈ Icc a b, s1 u ≤ s2 u)
    (hi1 : IntervalIntegrable (fun u => Lp u / s1 u) volume a b)
    (hi2 : IntervalIntegrable (fun u => Lp u / s2 u) volume a b) :
    quasipotentialIncrement Lp s2 a b ≤ quasipotentialIncrement Lp s1 a b := by
  refine intervalIntegral.integral_mono_on hab hi2 hi1 ?_
  intro u hu
  exact div_le_div_of_nonneg_left (hLp u hu) (hs1 u hu) (hs12 u hu)

/-- **The quantitative form of FW-2 at the level of the closed form (*), with section 8's own
strictness hypothesis.**  Section 8 states the strictness input of FW-2 as a lower bound `δ` on
the *reciprocal gap*, `uᵀ (Σ_B⁻¹ - Σ_A⁻¹) u ≥ δ ‖u‖²`, which in one dimension reads
`δ ≤ 1/σ_B² - 1/σ_A²`; it is FW-2' of section 11.3 that replaces `δ` by the pair `(Λ, ε)` and
derives `δ = ε/(Λ(Λ+ε))` from it.  This lemma takes the reciprocal gap directly: on a subinterval
`[c,d]` of the band where the climb is at least `m` and the reciprocal gap at least `δ`, the
increment of the effective quasipotential drops by at least `(d - c) · m δ`.

The reciprocal-gap hypothesis is genuinely weaker than FW-2''s pair, which is why both forms are
carried.  A positive reciprocal gap does force an upper bound on the baseline (`σ_B² ≤ 1/δ`), but
it forces no positive lower bound on the *difference* `σ_A² - σ_B²`: that difference degenerates
as the baseline diffusion goes to zero while `1/σ_B² - 1/σ_A²` does not, and a baseline diffusion
going to zero on the band is exactly the shoulder regime `σ_dir² = s² σ_obj² → 0` that FW-3 is
about.  No continuity is needed for this form. -/
theorem quasipotentialIncrement_le_sub_reciprocalGap
    {Lp s1 s2 : ℝ → ℝ} {a b c d m delta : ℝ}
    (hac : a ≤ c) (hcd : c ≤ d) (hdb : d ≤ b)
    (hm : 0 ≤ m) (hdelta : 0 ≤ delta)
    (hLp : ∀ u ∈ Icc a b, 0 ≤ Lp u)
    (hs1 : ∀ u ∈ Icc a b, 0 < s1 u)
    (hs12 : ∀ u ∈ Icc a b, s1 u ≤ s2 u)
    (hLpm : ∀ u ∈ Icc c d, m ≤ Lp u)
    (hgap : ∀ u ∈ Icc c d, delta ≤ 1 / s1 u - 1 / s2 u)
    (hi1 : IntervalIntegrable (fun u => Lp u / s1 u) volume a b)
    (hi2 : IntervalIntegrable (fun u => Lp u / s2 u) volume a b) :
    quasipotentialIncrement Lp s2 a b
      ≤ quasipotentialIncrement Lp s1 a b - (d - c) * (m * delta) := by
  have hab : a ≤ b := hac.trans (hcd.trans hdb)
  set F : ℝ → ℝ := fun u => Lp u / s1 u - Lp u / s2 u with hF
  have hiF : IntervalIntegrable F volume a b := hi1.sub hi2
  have huab : uIcc a b = Icc a b := uIcc_of_le hab
  have hcdsub : Icc c d ⊆ Icc a b := Icc_subset_Icc hac hdb
  have hFnn : ∀ u ∈ Icc a b, 0 ≤ F u := by
    intro u hu
    have := div_le_div_of_nonneg_left (hLp u hu) (hs1 u hu) (hs12 u hu)
    simp only [hF]
    linarith
  have hFgap : ∀ u ∈ Icc c d, m * delta ≤ F u := by
    intro u hu
    have hLpu : 0 ≤ Lp u := hm.trans (hLpm u hu)
    have h1 : m * delta ≤ Lp u * delta := mul_le_mul_of_nonneg_right (hLpm u hu) hdelta
    have h2 : Lp u * delta ≤ Lp u * (1 / s1 u - 1 / s2 u) :=
      mul_le_mul_of_nonneg_left (hgap u hu) hLpu
    have h3 : Lp u * (1 / s1 u - 1 / s2 u) = F u := by simp only [hF]; ring
    linarith
  have hiac : IntervalIntegrable F volume a c :=
    hiF.mono_set (by rw [uIcc_of_le hac, huab]; exact Icc_subset_Icc le_rfl (hcd.trans hdb))
  have hicd : IntervalIntegrable F volume c d :=
    hiF.mono_set (by rw [uIcc_of_le hcd, huab]; exact hcdsub)
  have hidb : IntervalIntegrable F volume d b :=
    hiF.mono_set (by rw [uIcc_of_le hdb, huab]; exact Icc_subset_Icc (hac.trans hcd) le_rfl)
  have hsplit :
      (∫ u in a..c, F u) + (∫ u in c..d, F u) + (∫ u in d..b, F u) = ∫ u in a..b, F u := by
    rw [intervalIntegral.integral_add_adjacent_intervals hiac hicd,
      intervalIntegral.integral_add_adjacent_intervals (hiac.trans hicd) hidb]
  have h1 : (0:ℝ) ≤ ∫ u in a..c, F u :=
    intervalIntegral.integral_nonneg hac (fun u hu => hFnn u ⟨hu.1, hu.2.trans (hcd.trans hdb)⟩)
  have h3 : (0:ℝ) ≤ ∫ u in d..b, F u :=
    intervalIntegral.integral_nonneg hdb
      (fun u hu => hFnn u ⟨(hac.trans hcd).trans hu.1, hu.2⟩)
  have h2 : (d - c) * (m * delta) ≤ ∫ u in c..d, F u := by
    calc (d - c) * (m * delta)
        = ∫ _u in c..d, (m * delta) := by
          rw [intervalIntegral.integral_const, smul_eq_mul]
      _ ≤ ∫ u in c..d, F u :=
          intervalIntegral.integral_mono_on hcd intervalIntegrable_const hicd hFgap
  have hkey : (d - c) * (m * delta) ≤ ∫ u in a..b, F u := by
    rw [← hsplit]; linarith
  have hFint : (∫ u in a..b, F u)
      = quasipotentialIncrement Lp s1 a b - quasipotentialIncrement Lp s2 a b := by
    simp only [hF, quasipotentialIncrement]
    exact intervalIntegral.integral_sub hi1 hi2
  rw [hFint] at hkey
  linarith

/-- **FW-2', quantitative form, in one dimension.**  If in addition the climb is at least `m` on
a subinterval `[c,d]` of the band, the baseline diffusion there is at most `Λ`, and the lift's
diffusion exceeds it there by at least `ε`, then the effective action across `[a,b]` drops by at
least `(d - c) · m ε / (Λ (Λ + ε))`.  The modulus is FW-2''s `δ`, and the length `d - c` plays
the role of FW-2''s occupancy time `τ`.  No continuity is needed for this form.

This is the specialization of `quasipotentialIncrement_le_sub_reciprocalGap` in which the
reciprocal gap is not assumed but derived, by `reciprocal_gap_of_diffusion_excess`, from the pair
`(Λ, ε)`; the derivation needs an upper bound on the baseline diffusion only, never on the
lift's, exactly as FW-2' observes. -/
theorem quasipotentialIncrement_le_sub_gap
    {Lp s1 s2 : ℝ → ℝ} {a b c d m lam eps : ℝ}
    (hac : a ≤ c) (hcd : c ≤ d) (hdb : d ≤ b)
    (hm : 0 ≤ m) (hlam : 0 < lam) (heps : 0 ≤ eps)
    (hLp : ∀ u ∈ Icc a b, 0 ≤ Lp u)
    (hs1 : ∀ u ∈ Icc a b, 0 < s1 u)
    (hs12 : ∀ u ∈ Icc a b, s1 u ≤ s2 u)
    (hLpm : ∀ u ∈ Icc c d, m ≤ Lp u)
    (hs1lam : ∀ u ∈ Icc c d, s1 u ≤ lam)
    (hexc : ∀ u ∈ Icc c d, s1 u + eps ≤ s2 u)
    (hi1 : IntervalIntegrable (fun u => Lp u / s1 u) volume a b)
    (hi2 : IntervalIntegrable (fun u => Lp u / s2 u) volume a b) :
    quasipotentialIncrement Lp s2 a b
      ≤ quasipotentialIncrement Lp s1 a b - (d - c) * (m * eps / (lam * (lam + eps))) := by
  have hcdsub : Icc c d ⊆ Icc a b := Icc_subset_Icc hac hdb
  have hgap : ∀ u ∈ Icc c d, eps / (lam * (lam + eps)) ≤ 1 / s1 u - 1 / s2 u := by
    intro u hu
    have h := reciprocal_gap_of_diffusion_excess (m := 1) (p := 1) zero_le_one le_rfl
      (hs1 u (hcdsub hu)) heps hlam (hs1lam u hu) (hexc u hu)
    simpa using h
  have h := quasipotentialIncrement_le_sub_reciprocalGap (delta := eps / (lam * (lam + eps)))
    hac hcd hdb hm (by positivity) hLp hs1 hs12 hLpm hgap hi1 hi2
  rw [mul_div_assoc]
  exact h

/-- **FW-2, strict form, in one dimension.**  If the diffusion is strictly larger and the climb
strictly positive on a subinterval of positive length, the effective action across the band is
strictly smaller.  Continuity on the closed band is used twice: to get integrability of the two
integrands, and to make the strict pointwise inequality on `(c,d)` produce a strictly positive
integral. -/
theorem quasipotentialIncrement_strictAnti_in_diffusion
    {Lp s1 s2 : ℝ → ℝ} {a b c d : ℝ}
    (hac : a ≤ c) (hcd : c < d) (hdb : d ≤ b)
    (hLpc : ContinuousOn Lp (Icc a b))
    (hs1c : ContinuousOn s1 (Icc a b))
    (hs2c : ContinuousOn s2 (Icc a b))
    (hLp : ∀ u ∈ Icc a b, 0 ≤ Lp u)
    (hs1 : ∀ u ∈ Icc a b, 0 < s1 u)
    (hs12 : ∀ u ∈ Icc a b, s1 u ≤ s2 u)
    (hLppos : ∀ u ∈ Icc c d, 0 < Lp u)
    (hlt : ∀ u ∈ Icc c d, s1 u < s2 u) :
    quasipotentialIncrement Lp s2 a b < quasipotentialIncrement Lp s1 a b := by
  have hab : a ≤ b := hac.trans (hcd.le.trans hdb)
  have huab : uIcc a b = Icc a b := uIcc_of_le hab
  have hs2 : ∀ u ∈ Icc a b, 0 < s2 u := fun u hu => lt_of_lt_of_le (hs1 u hu) (hs12 u hu)
  have hi1 : IntervalIntegrable (fun u => Lp u / s1 u) volume a b := by
    apply ContinuousOn.intervalIntegrable
    rw [huab]
    exact hLpc.div hs1c (fun u hu => (hs1 u hu).ne')
  have hi2 : IntervalIntegrable (fun u => Lp u / s2 u) volume a b := by
    apply ContinuousOn.intervalIntegrable
    rw [huab]
    exact hLpc.div hs2c (fun u hu => (hs2 u hu).ne')
  set F : ℝ → ℝ := fun u => Lp u / s1 u - Lp u / s2 u with hF
  have hiF : IntervalIntegrable F volume a b := hi1.sub hi2
  have hcdsub : Icc c d ⊆ Icc a b := Icc_subset_Icc hac hdb
  have hFnn : ∀ u ∈ Icc a b, 0 ≤ F u := by
    intro u hu
    have := div_le_div_of_nonneg_left (hLp u hu) (hs1 u hu) (hs12 u hu)
    simp only [hF]
    linarith
  have hFpos : ∀ u ∈ Icc c d, 0 < F u := by
    intro u hu
    have hu' := hcdsub hu
    have hstrict : Lp u / s2 u < Lp u / s1 u :=
      div_lt_div_of_pos_left (hLppos u hu) (hs1 u hu') (hlt u hu)
    simp only [hF]
    linarith
  have hiac : IntervalIntegrable F volume a c :=
    hiF.mono_set (by rw [uIcc_of_le hac, huab]; exact Icc_subset_Icc le_rfl (hcd.le.trans hdb))
  have hicd : IntervalIntegrable F volume c d :=
    hiF.mono_set (by rw [uIcc_of_le hcd.le, huab]; exact hcdsub)
  have hidb : IntervalIntegrable F volume d b :=
    hiF.mono_set (by rw [uIcc_of_le hdb, huab]; exact Icc_subset_Icc (hac.trans hcd.le) le_rfl)
  have hsplit :
      (∫ u in a..c, F u) + (∫ u in c..d, F u) + (∫ u in d..b, F u) = ∫ u in a..b, F u := by
    rw [intervalIntegral.integral_add_adjacent_intervals hiac hicd,
      intervalIntegral.integral_add_adjacent_intervals (hiac.trans hicd) hidb]
  have h1 : (0:ℝ) ≤ ∫ u in a..c, F u :=
    intervalIntegral.integral_nonneg hac
      (fun u hu => hFnn u ⟨hu.1, hu.2.trans (hcd.le.trans hdb)⟩)
  have h3 : (0:ℝ) ≤ ∫ u in d..b, F u :=
    intervalIntegral.integral_nonneg hdb
      (fun u hu => hFnn u ⟨(hac.trans hcd.le).trans hu.1, hu.2⟩)
  have h2 : (0:ℝ) < ∫ u in c..d, F u :=
    intervalIntegral.intervalIntegral_pos_of_pos_on hicd
      (fun u hu => hFpos u ⟨hu.1.le, hu.2.le⟩) hcd
  have hkey : (0:ℝ) < ∫ u in a..b, F u := by rw [← hsplit]; linarith
  have hFint : (∫ u in a..b, F u)
      = quasipotentialIncrement Lp s1 a b - quasipotentialIncrement Lp s2 a b := by
    simp only [hF, quasipotentialIncrement]
    exact intervalIntegral.integral_sub hi1 hi2
  rw [hFint] at hkey
  linarith

/-! ## The corollary the paper uses, with (A6) explicit -/

/-- **The paper's corollary, strict form.**  Write the lift's effective diffusion on the band as
the direct baseline's plus the excess `exc` that FW-1 computes,
`σ_lift² = σ_dir² + s²(C + H V H) + T₃`.  If the excess is non-negative across the band (FW-1
with the remainder controlled by (A4'-q)) and **strictly positive somewhere on it** — which is
assumption (A6) of section 11.3, carried here as its own hypothesis and not folded into the
conclusion — then the lift's effective action across the barrier is strictly smaller than the
direct baseline's.

Section 11.3 is explicit that (A6) is where the mechanism's honesty sits: `H V H` is positive
semidefinite always, but the coupling term `C = H Σ_slack + Σ_slackᵀ H` is not, so a lift whose
latent-weight fluctuation is anti-aligned with the curvature-weighted gradient fluctuation gets
nothing.  The theorem says the lift helps exactly to the extent that (A6) holds. -/
theorem liftQuasipotentialIncrement_lt_direct
    {Lp sdir exc : ℝ → ℝ} {a b c d : ℝ}
    (hac : a ≤ c) (hcd : c < d) (hdb : d ≤ b)
    (hLpc : ContinuousOn Lp (Icc a b))
    (hdirc : ContinuousOn sdir (Icc a b))
    (hexcc : ContinuousOn exc (Icc a b))
    (hLp : ∀ u ∈ Icc a b, 0 ≤ Lp u)
    (hdir : ∀ u ∈ Icc a b, 0 < sdir u)
    (hexc : ∀ u ∈ Icc a b, 0 ≤ exc u)
    (hLppos : ∀ u ∈ Icc c d, 0 < Lp u)
    (hA6 : ∀ u ∈ Icc c d, 0 < exc u) :
    quasipotentialIncrement Lp (fun u => sdir u + exc u) a b
      < quasipotentialIncrement Lp sdir a b := by
  refine quasipotentialIncrement_strictAnti_in_diffusion hac hcd hdb hLpc hdirc (hdirc.add hexcc)
    hLp hdir (fun u hu => ?_) hLppos (fun u hu => ?_)
  · have := hexc u hu; linarith
  · have := hA6 u hu; linarith

/-- **The paper's corollary, quantitative form.**  With (A6) quantified by a modulus `κ` on the
subinterval `[c,d]`, a climb of at least `m` there, and the baseline diffusion bounded by `Λ`
there (FW-3b's ellipticity), the lift's effective action across the barrier is smaller by at
least `(d - c) · m κ / (Λ (Λ + κ))`.  This is FW-2''s `δ` in the scalar reduction: in the paper's
variables the modulus `κ` here is section 11.3's diffusion excess `s² κ`, and `Λ` is the upper
bound on the direct baseline's `s² σ_obj²` on the band. -/
theorem liftQuasipotentialIncrement_le_sub_gap
    {Lp sdir exc : ℝ → ℝ} {a b c d m lam kappa : ℝ}
    (hac : a ≤ c) (hcd : c ≤ d) (hdb : d ≤ b)
    (hm : 0 ≤ m) (hlam : 0 < lam) (hkappa : 0 ≤ kappa)
    (hLp : ∀ u ∈ Icc a b, 0 ≤ Lp u)
    (hdir : ∀ u ∈ Icc a b, 0 < sdir u)
    (hexc : ∀ u ∈ Icc a b, 0 ≤ exc u)
    (hLpm : ∀ u ∈ Icc c d, m ≤ Lp u)
    (hdirlam : ∀ u ∈ Icc c d, sdir u ≤ lam)
    (hA6 : ∀ u ∈ Icc c d, kappa ≤ exc u)
    (hi1 : IntervalIntegrable (fun u => Lp u / sdir u) volume a b)
    (hi2 : IntervalIntegrable (fun u => Lp u / (sdir u + exc u)) volume a b) :
    quasipotentialIncrement Lp (fun u => sdir u + exc u) a b
      ≤ quasipotentialIncrement Lp sdir a b
          - (d - c) * (m * kappa / (lam * (lam + kappa))) := by
  refine quasipotentialIncrement_le_sub_gap hac hcd hdb hm hlam hkappa hLp hdir
    (fun u hu => ?_) hLpm hdirlam (fun u hu => ?_) hi1 hi2
  · have := hexc u hu; linarith
  · have := hA6 u hu; linarith

/-- The Arrhenius comparison that the quasipotential comparison feeds.  A strictly smaller
effective action gives a strictly smaller Kramers exponent `2V/η`, hence a strictly smaller
Arrhenius factor `exp(2V/η)`.

The constant `2` is section 3's convention, in which `V` is the increment (*) of `U_eff` and
(L2') writes the exponent as `2α/(η σ_eff²)`; the Freidlin–Wentzell convention, in which `V` is
the action itself, writes `V/η`.  Nothing here depends on the choice, since `Real.exp` is
monotone and both conventions differ by a positive factor, but the two should not be conflated
when this lemma is quoted — see the discussion of the normalization in `fwAction` below.

This is a statement about `Real.exp` and nothing more.  The identification of `exp(2V/η)` with a
mean first passage time is **not** proved anywhere in this development, because this mathlib has
no Itô calculus, no SDE and no exit-time theory; it is the named hypothesis of
`KramersExitTime.lean`. -/
theorem kramersFactor_lt_of_quasipotential_lt {V1 V2 eta : ℝ} (heta : 0 < eta) (h : V2 < V1) :
    Real.exp (2 * V2 / eta) < Real.exp (2 * V1 / eta) := by
  rw [Real.exp_lt_exp]
  have h2 : 2 * V2 < 2 * V1 := by linarith
  have := mul_lt_mul_of_pos_right h2 (inv_pos.mpr heta)
  simpa [div_eq_mul_inv] using this

/-! ## The variational step, in any dimension -/

/-- **The variational half of FW-2, in any dimension.**  If one action dominates another path by
path, the infimum over paths — the quasipotential — obeys the same inequality.  Section 8's proof
of FW-2 uses exactly this after establishing the pointwise integrand comparison: "Integrating,
`S_A[γ] ≤ S_B[γ]` for EVERY admissible path, so the infimum obeys `V_A ≤ V_B`."

Nothing here is one-dimensional: `Γ` is an arbitrary non-empty index type, so this lemma is
reusable verbatim once `FWReduction.lean` and `InversePSDAntitone.lean` supply the pointwise
comparison of `uᵀ Σ(γ)⁻¹ u` in the Loewner order.  `hbdd` cannot be dropped: without it `⨅` is
not the greatest lower bound. -/
theorem quasipotential_le_of_action_le {Γ : Type*} [Nonempty Γ] {SA SB : Γ → ℝ}
    (hbdd : BddBelow (range SA)) (h : ∀ γ, SA γ ≤ SB γ) :
    (⨅ γ, SA γ) ≤ ⨅ γ, SB γ :=
  le_ciInf fun γ => (ciInf_le hbdd γ).trans (h γ)

/-- The quantitative variational step under an action gap `g` holding along **every** path.
This is not quite section 8's strictness step: that argument takes a `B`-minimizing sequence and
needs the gap only along its terms, whereas this lemma demands it of every path in `Γ`.  The
faithful form is `quasipotential_le_sub_of_action_le_sub_nearMin` immediately below, from which
this one follows by ignoring the near-minimality hypothesis; it is kept because it is the
convenient form when the gap really is uniform. -/
theorem quasipotential_le_sub_of_action_le_sub {Γ : Type*} [Nonempty Γ] {SA SB : Γ → ℝ} {g : ℝ}
    (hbdd : BddBelow (range SA)) (h : ∀ γ, SA γ ≤ SB γ - g) :
    (⨅ γ, SA γ) ≤ (⨅ γ, SB γ) - g := by
  have h1 : ∀ γ, (⨅ γ', SA γ') + g ≤ SB γ := by
    intro γ
    have h2 := ciInf_le hbdd γ
    have h3 := h γ
    linarith
  have h4 : (⨅ γ', SA γ') + g ≤ ⨅ γ, SB γ := le_ciInf h1
  linarith

/-- **The strictness clause of FW-2, in any dimension**, under a gap holding along every path.
As with the previous lemma the faithful form is
`quasipotential_lt_of_action_le_sub_nearMin`, which asks for the gap only near the infimum. -/
theorem quasipotential_lt_of_action_le_sub {Γ : Type*} [Nonempty Γ] {SA SB : Γ → ℝ} {g : ℝ}
    (hg : 0 < g) (hbdd : BddBelow (range SA)) (h : ∀ γ, SA γ ≤ SB γ - g) :
    (⨅ γ, SA γ) < ⨅ γ, SB γ := by
  have := quasipotential_le_sub_of_action_le_sub hbdd h
  linarith

/-- **The variational gap step exactly as section 8 argues it: the gap is needed only along
near-minimizing paths.**  Section 8's strictness proof reads "take a `B`-minimizing sequence
`γ_k` with `S_B[γ_k] → V_B`; on the stated time set `S_A[γ_k] ≤ S_B[γ_k] - (δc²/4)τ`, so
`V_A ≤ V_B - (δc²/4)τ`".  The gap is thus assumed only of paths whose `B`-action lies within some
`ρ > 0` of the infimum.  Here the minimizing sequence is replaced by a single `ε`-argument, which
shortens the proof without strengthening the hypothesis.

The distinction is load-bearing rather than cosmetic, and it is why this lemma exists alongside
`quasipotential_le_sub_of_action_le_sub`.  FW-3a of section 11.1 produces its occupancy time
`τ(w, Λ, S̄, M)` only for paths of action at most `S̄`, and its own remark records that `τ` *shrinks
as `S̄` grows*; so over an unrestricted path space there is in general **no** single positive
occupancy time, and a gap hypothesis quantified over every path is not something FW-3a can
supply.  Quantified over the sublevel set `{γ : S_B[γ] ≤ V_B + ρ}` it is exactly what FW-3a
supplies, with `S̄ = V_B + ρ`.

`hbdd` cannot be dropped: it is what makes `⨅ SA` the greatest lower bound rather than the junk
value `⨅` returns on a family unbounded below.  Nothing here is one-dimensional. -/
theorem quasipotential_le_sub_of_action_le_sub_nearMin {Γ : Type*} [Nonempty Γ]
    {SA SB : Γ → ℝ} {g rho : ℝ} (hrho : 0 < rho) (hbdd : BddBelow (range SA))
    (h : ∀ γ, SB γ ≤ (⨅ γ', SB γ') + rho → SA γ ≤ SB γ - g) :
    (⨅ γ, SA γ) ≤ (⨅ γ, SB γ) - g := by
  refine le_of_forall_sub_le fun ε hε => ?_
  set VB : ℝ := ⨅ γ', SB γ' with hVB
  have hmin : (0:ℝ) < min ε rho := lt_min hε hrho
  have hlt : VB < VB + min ε rho := by linarith
  obtain ⟨γ, hγ⟩ := exists_lt_of_ciInf_lt (f := SB) (by rw [← hVB]; exact hlt)
  have hnear : SB γ ≤ VB + rho := by
    have hr : min ε rho ≤ rho := min_le_right _ _
    linarith
  have hA := h γ hnear
  have hinf : (⨅ γ', SA γ') ≤ SA γ := ciInf_le hbdd γ
  have hl : min ε rho ≤ ε := min_le_left _ _
  linarith

/-- **The strictness clause of FW-2 as section 8 argues it**, with the gap required only of
near-minimizing paths.  This is the dimension-free half of FW-2' and is reusable verbatim in
`d > 1` once the pointwise Loewner-order comparison is available. -/
theorem quasipotential_lt_of_action_le_sub_nearMin {Γ : Type*} [Nonempty Γ]
    {SA SB : Γ → ℝ} {g rho : ℝ} (hg : 0 < g) (hrho : 0 < rho) (hbdd : BddBelow (range SA))
    (h : ∀ γ, SB γ ≤ (⨅ γ', SB γ') + rho → SA γ ≤ SB γ - g) :
    (⨅ γ, SA γ) < ⨅ γ, SB γ := by
  have := quasipotential_le_sub_of_action_le_sub_nearMin hrho hbdd h
  linarith

/-! ## The one-dimensional Freidlin–Wentzell action, and FW-2 for the genuine quasipotential -/

/-- The one-dimensional Freidlin–Wentzell action of a path, for the diffusion
`dφ = -L̃'(φ) dt + √η σ_eff(φ) dB` of section 3:
`S_T[γ] = nrm · ∫₀ᵀ (γ'(t) + L̃'(γ(t)))² / σ_eff²(γ(t)) dt`.

The normalization `nrm` is carried as a parameter rather than fixed.  Section 7 of the design
note writes `1/4`; the classical Freidlin–Wentzell normalization for `dφ = b dt + √η σ dB`, the
one that makes the exit-time exponent `V/η` agree with the Kramers exponent `2 α / (η σ²)` of
`(L2')`, is `1/2`.  Every comparison below is invariant under the choice, since both actions
carry the same factor, so nothing here depends on settling it; the one place the constant is
visible is `quasipotentialIncrement_le_fwAction`, where the bound is `4 · nrm · ΔU_eff`.

The velocity `gammaDot` is a free function here rather than the derivative of `gamma`: the
comparison theorems below hold for any pair, since they compare two actions of the *same*
`(path, velocity, horizon)` triple, and only `quasipotentialIncrement_le_fwAction` ties
`gammaDot` to `gamma`, by the hypothesis `hderiv`.  Imposing the link on an admissible path is
therefore part of the choice of the index type `Γ`, not of this definition.

One caution about the encoding: Lean's integral of a non-integrable integrand is `0`, so a path
whose action integrand fails to be integrable is recorded here as having action `0`, not `+∞`.
The comparison theorems below therefore carry per-path integrability hypotheses — dischargeable
by `fwAction_integrand_intervalIntegrable` for continuous data — and those hypotheses confine
`Γ` to path families on which this action and `fwQuasipotential` agree with the
Freidlin–Wentzell ones. -/
noncomputable def fwAction (nrm : ℝ) (Lp sigma2 gamma gammaDot : ℝ → ℝ) (T : ℝ) : ℝ :=
  nrm * ∫ t in (0:ℝ)..T, (gammaDot t + Lp (gamma t)) ^ 2 / sigma2 (gamma t)

/-- The quasipotential as Freidlin–Wentzell define it: the infimum of the action over admissible
paths.  Admissibility — the endpoints, absolute continuity, and remaining inside the domain `D` —
is encoded in the choice of the index type `Γ`, about which nothing is assumed, so the statements
below hold for whatever notion of admissible path is settled on. -/
noncomputable def fwQuasipotential {Γ : Type*} (nrm : ℝ) (Lp sigma2 : ℝ → ℝ)
    (traj vel : Γ → ℝ → ℝ) (hor : Γ → ℝ) : ℝ :=
  ⨅ γ : Γ, fwAction nrm Lp sigma2 (traj γ) (vel γ) (hor γ)

/-- The one-dimensional Freidlin–Wentzell action is non-negative along any path that stays where
the diffusion is positive.  No integrability is needed: a non-integrable integrand makes the
interval integral `0`, which is still non-negative. -/
theorem fwAction_nonneg {nrm : ℝ} {Lp sigma2 gamma gammaDot : ℝ → ℝ} {T : ℝ} {D : Set ℝ}
    (hnrm : 0 ≤ nrm) (hT : 0 ≤ T)
    (hgam : ∀ t ∈ Icc (0:ℝ) T, gamma t ∈ D)
    (hs : ∀ x ∈ D, 0 < sigma2 x) :
    0 ≤ fwAction nrm Lp sigma2 gamma gammaDot T := by
  refine mul_nonneg hnrm (intervalIntegral.integral_nonneg hT fun t ht => ?_)
  exact div_nonneg (sq_nonneg _) (hs _ (hgam t ht)).le

/-- The family of path actions is bounded below by `0`, so `fwQuasipotential` is a genuine
greatest lower bound and not the junk value `⨅` returns on a family unbounded below.  Every
one-dimensional theorem below discharges its boundedness side condition with this lemma instead
of assuming it; only the dimension-free lemmas of the previous section, which know nothing about
the shape of the action, still carry it as a hypothesis. -/
theorem bddBelow_range_fwAction {Γ : Type*} {nrm : ℝ} {Lp sigma2 : ℝ → ℝ}
    {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    (hnrm : 0 ≤ nrm) (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs : ∀ x ∈ D, 0 < sigma2 x) :
    BddBelow (range fun γ => fwAction nrm Lp sigma2 (traj γ) (vel γ) (hor γ)) := by
  refine ⟨0, ?_⟩
  rintro y ⟨γ, rfl⟩
  exact fwAction_nonneg (D := D) hnrm (hT γ) (hin γ) hs

/-- Continuity discharges the per-path integrability hypotheses of the comparison theorems
below.  If the path and its recorded velocity are continuous on the closed window `[0, T]`, the
path remains in `D`, and `L̃'` and `σ²` are continuous on `D` with `σ² > 0` there, then the
action integrand `(γ' + L̃'(γ))² / σ²(γ)` is interval-integrable on `[0, T]`.  This is how the
hypotheses `hi1`/`hi2` below are meant to be discharged once `Γ` is a space of continuous
admissible paths with continuous recorded velocities: apply it once with `σ² := σ_dir²` and once
with `σ² := σ_lift²`.  Without such a restriction on `Γ` the hypotheses cannot be dropped: a
path whose baseline integrand is not integrable would have Lean action `0` rather than `+∞`,
and the quasipotential comparison would be false. -/
theorem fwAction_integrand_intervalIntegrable
    {Lp sigma2 gamma gammaDot : ℝ → ℝ} {T : ℝ} {D : Set ℝ} (hT : 0 ≤ T)
    (hgc : ContinuousOn gamma (Icc (0:ℝ) T))
    (hdotc : ContinuousOn gammaDot (Icc (0:ℝ) T))
    (hmaps : ∀ t ∈ Icc (0:ℝ) T, gamma t ∈ D)
    (hLpc : ContinuousOn Lp D) (hsc : ContinuousOn sigma2 D)
    (hspos : ∀ x ∈ D, 0 < sigma2 x) :
    IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / sigma2 (gamma t))
      volume 0 T := by
  have hmt : MapsTo gamma (Icc (0:ℝ) T) D := hmaps
  have hLpg : ContinuousOn (fun t => Lp (gamma t)) (Icc (0:ℝ) T) := hLpc.comp hgc hmt
  have hsg : ContinuousOn (fun t => sigma2 (gamma t)) (Icc (0:ℝ) T) := hsc.comp hgc hmt
  have hsgne : ∀ t ∈ Icc (0:ℝ) T, sigma2 (gamma t) ≠ 0 := fun t ht =>
    (hspos _ (hmaps t ht)).ne'
  apply ContinuousOn.intervalIntegrable
  rw [uIcc_of_le hT]
  exact ((hdotc.add hLpg).pow 2).div hsg hsgne

/-- **FW-2's pointwise step, one path at a time.**  Raising the diffusion at every point the path
visits cannot raise its action.  In one dimension the Loewner-order auxiliary lemma of section 8
degenerates to `a/y ≤ a/x` for `0 < x ≤ y` and `0 ≤ a`, with `a = (γ' + L̃'(γ))² ≥ 0`; no sign
condition on the drift is needed here, unlike for the closed-form increment (*). -/
theorem fwAction_antitone_in_diffusion {nrm : ℝ} {Lp s1 s2 gamma gammaDot : ℝ → ℝ} {T : ℝ}
    {D : Set ℝ} (hnrm : 0 ≤ nrm) (hT : 0 ≤ T)
    (hgam : ∀ t ∈ Icc (0:ℝ) T, gamma t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hi1 : IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / s1 (gamma t))
      volume 0 T)
    (hi2 : IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / s2 (gamma t))
      volume 0 T) :
    fwAction nrm Lp s2 gamma gammaDot T ≤ fwAction nrm Lp s1 gamma gammaDot T := by
  refine mul_le_mul_of_nonneg_left ?_ hnrm
  refine intervalIntegral.integral_mono_on hT hi2 hi1 ?_
  intro t ht
  exact div_le_div_of_nonneg_left (sq_nonneg _) (hs1 _ (hgam t ht)) (hs12 _ (hgam t ht))

/-- **FW-2's strictness clause, one path at a time, exactly as section 8 states it.**  Section 8
assumes a lower bound `δ` on the reciprocal gap, `uᵀ (Σ_B⁻¹ - Σ_A⁻¹) u ≥ δ ‖u‖²` — in one
dimension `δ ≤ 1/σ_B²(γ) - 1/σ_A²(γ)` — on a time set of measure at least `τ` on which the
velocity excess `u = γ' + L̃'(γ)` has `|u| ≥ c`.  The conclusion is section 8's own bound,
`S_A[γ] ≤ S_B[γ] - nrm · δ c² τ`, which at the design note's `nrm = 1/4` is the `(δ c²/4) τ` the
note writes down.

The time set is an arbitrary measurable subset of `(0, T]`, not an interval, which is what FW-3a
of section 11.1 actually produces.  `fwAction_le_sub_gap` below is FW-2''s specialization, in
which `δ` is not assumed but derived from an upper bound on the baseline diffusion together with
a lower bound on the excess; the reciprocal-gap form is the weaker hypothesis of the two, since a
positive reciprocal gap survives a baseline diffusion tending to zero while a positive excess
`σ_A² - σ_B²` need not. -/
theorem fwAction_le_sub_reciprocalGap {nrm : ℝ} {Lp s1 s2 gamma gammaDot : ℝ → ℝ}
    {T cvel delta tau : ℝ} {D E : Set ℝ}
    (hnrm : 0 ≤ nrm) (hT : 0 ≤ T)
    (hE : MeasurableSet E) (hEsub : E ⊆ Ioc 0 T) (htau : tau ≤ volume.real E)
    (hcvel : 0 ≤ cvel) (hdelta : 0 ≤ delta)
    (hgam : ∀ t ∈ Ioc (0:ℝ) T, gamma t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hgap : ∀ t ∈ E, delta ≤ 1 / s1 (gamma t) - 1 / s2 (gamma t))
    (hvel : ∀ t ∈ E, cvel ≤ |gammaDot t + Lp (gamma t)|)
    (hi1 : IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / s1 (gamma t))
      volume 0 T)
    (hi2 : IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / s2 (gamma t))
      volume 0 T) :
    fwAction nrm Lp s2 gamma gammaDot T
      ≤ fwAction nrm Lp s1 gamma gammaDot T - nrm * (tau * (cvel ^ 2 * delta)) := by
  set F : ℝ → ℝ := fun t =>
    (gammaDot t + Lp (gamma t)) ^ 2 / s1 (gamma t)
      - (gammaDot t + Lp (gamma t)) ^ 2 / s2 (gamma t) with hF
  have hiF : IntervalIntegrable F volume 0 T := hi1.sub hi2
  have hg0 : 0 ≤ cvel ^ 2 * delta := mul_nonneg (sq_nonneg _) hdelta
  have hFnn : ∀ t ∈ Ioc (0:ℝ) T, 0 ≤ F t := by
    intro t ht
    have := div_le_div_of_nonneg_left (sq_nonneg (gammaDot t + Lp (gamma t)))
      (hs1 _ (hgam t ht)) (hs12 _ (hgam t ht))
    simp only [hF]
    linarith
  have hFgap : ∀ t ∈ E, cvel ^ 2 * delta ≤ F t := by
    intro t ht
    have hcv : cvel ^ 2 ≤ (gammaDot t + Lp (gamma t)) ^ 2 := by
      have h1 : cvel ≤ |gammaDot t + Lp (gamma t)| := hvel t ht
      have h2 : cvel ^ 2 ≤ |gammaDot t + Lp (gamma t)| ^ 2 := by
        nlinarith [abs_nonneg (gammaDot t + Lp (gamma t)), hcvel]
      rwa [sq_abs] at h2
    have h1 : cvel ^ 2 * delta ≤ (gammaDot t + Lp (gamma t)) ^ 2 * delta :=
      mul_le_mul_of_nonneg_right hcv hdelta
    have h2 : (gammaDot t + Lp (gamma t)) ^ 2 * delta
        ≤ (gammaDot t + Lp (gamma t)) ^ 2 * (1 / s1 (gamma t) - 1 / s2 (gamma t)) :=
      mul_le_mul_of_nonneg_left (hgap t ht) (sq_nonneg _)
    have h3 : (gammaDot t + Lp (gamma t)) ^ 2 * (1 / s1 (gamma t) - 1 / s2 (gamma t)) = F t := by
      simp only [hF]; ring
    linarith
  have hIoc : IntegrableOn F (Ioc (0:ℝ) T) volume := hiF.1
  have hEint : IntegrableOn F E volume := hIoc.mono_set hEsub
  have hEfin : volume E < ⊤ := lt_of_le_of_lt (measure_mono hEsub) measure_Ioc_lt_top
  have hconst : IntegrableOn (fun _ : ℝ => cvel ^ 2 * delta) E volume :=
    integrableOn_const hEfin.ne
  have hnn : 0 ≤ᵐ[volume.restrict (Ioc (0:ℝ) T)] F := by
    rw [Filter.EventuallyLE, ae_restrict_iff' measurableSet_Ioc]
    exact ae_of_all _ hFnn
  have step1 : (∫ t in E, F t) ≤ ∫ t in Ioc (0:ℝ) T, F t :=
    setIntegral_mono_set hIoc hnn hEsub.eventuallyLE
  have step2 : tau * (cvel ^ 2 * delta) ≤ ∫ t in E, F t := by
    have h0 : (∫ _t in E, cvel ^ 2 * delta) ≤ ∫ t in E, F t :=
      setIntegral_mono_on hconst hEint hE hFgap
    rw [setIntegral_const, smul_eq_mul] at h0
    have h1 : tau * (cvel ^ 2 * delta) ≤ volume.real E * (cvel ^ 2 * delta) :=
      mul_le_mul_of_nonneg_right htau hg0
    linarith
  have hkey : tau * (cvel ^ 2 * delta) ≤ ∫ t in (0:ℝ)..T, F t := by
    rw [intervalIntegral.integral_of_le hT]
    linarith
  have hFint : (∫ t in (0:ℝ)..T, F t)
      = (∫ t in (0:ℝ)..T, (gammaDot t + Lp (gamma t)) ^ 2 / s1 (gamma t))
        - ∫ t in (0:ℝ)..T, (gammaDot t + Lp (gamma t)) ^ 2 / s2 (gamma t) :=
    intervalIntegral.integral_sub hi1 hi2
  rw [hFint] at hkey
  have hmul := mul_le_mul_of_nonneg_left hkey hnrm
  simp only [fwAction]
  nlinarith [hmul]

/-- **FW-2''s strictness clause, one path at a time.**  This is the form FW-2' of section 11.3
feeds: a time set `E` of measure at least `τ` on which the velocity excess `u = γ' + L̃'(γ)` has
`|u| ≥ c` and the diffusion excess is at least `ε` against a baseline of at most `Λ`.  The
conclusion is section 8's bound `S_A[γ] ≤ S_B[γ] - nrm · δ c² τ` with `δ = ε/(Λ(Λ+ε))` the
modulus FW-2' derives, here obtained from `reciprocal_gap_of_diffusion_excess` rather than
quoted, and needing an upper bound on the baseline diffusion only, never on the lift's.

The time set is an arbitrary measurable subset of `(0, T]`, not an interval, which is what FW-3a
of section 11.1 actually produces.  For section 8's own, weaker hypothesis — a lower bound on the
reciprocal gap itself — use `fwAction_le_sub_reciprocalGap`, of which this is a corollary. -/
theorem fwAction_le_sub_gap {nrm : ℝ} {Lp s1 s2 gamma gammaDot : ℝ → ℝ}
    {T cvel lam eps tau : ℝ} {D E : Set ℝ}
    (hnrm : 0 ≤ nrm) (hT : 0 ≤ T)
    (hE : MeasurableSet E) (hEsub : E ⊆ Ioc 0 T) (htau : tau ≤ volume.real E)
    (hcvel : 0 ≤ cvel) (hlam : 0 < lam) (heps : 0 ≤ eps)
    (hgam : ∀ t ∈ Ioc (0:ℝ) T, gamma t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hband : ∀ t ∈ E, s1 (gamma t) ≤ lam ∧ s1 (gamma t) + eps ≤ s2 (gamma t))
    (hvel : ∀ t ∈ E, cvel ≤ |gammaDot t + Lp (gamma t)|)
    (hi1 : IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / s1 (gamma t))
      volume 0 T)
    (hi2 : IntervalIntegrable (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / s2 (gamma t))
      volume 0 T) :
    fwAction nrm Lp s2 gamma gammaDot T
      ≤ fwAction nrm Lp s1 gamma gammaDot T
          - nrm * (tau * (cvel ^ 2 * eps / (lam * (lam + eps)))) := by
  have hgap : ∀ t ∈ E, eps / (lam * (lam + eps)) ≤ 1 / s1 (gamma t) - 1 / s2 (gamma t) := by
    intro t ht
    obtain ⟨hb1, hb2⟩ := hband t ht
    have h := reciprocal_gap_of_diffusion_excess (m := 1) (p := 1) zero_le_one le_rfl
      (hs1 _ (hgam t (hEsub ht))) heps hlam hb1 hb2
    simpa using h
  have h := fwAction_le_sub_reciprocalGap (D := D) (delta := eps / (lam * (lam + eps)))
    hnrm hT hE hEsub htau hcvel (by positivity) hgam hs1 hs12 hgap hvel hi1 hi2
  rw [mul_div_assoc]
  exact h

/-- **The closed form (*) is a lower bound for the action of every path.**  Completing the square,
`(γ' + L̃'(γ))² = (γ' - L̃'(γ))² + 4 γ' L̃'(γ)`, and changing variables `u = γ(t)` in
`∫₀ᵀ γ'(t) (L̃'/σ²)(γ(t)) dt = ∫_{γ(0)}^{γ(T)} L̃'(u)/σ²(u) du`, gives
`4 · nrm · (U_eff(γ(T)) - U_eff(γ(0))) ≤ S_T[γ]` for every continuously differentiable path
inside the domain, with no sign condition on the drift and no condition on the endpoints.

`fwQuasipotential_ge_quasipotentialIncrement` takes the infimum over paths and states the
consequence for the quasipotential itself.  That is one half of the classical one-dimensional
identification of (*) with the quasipotential.  **The reverse inequality is not proved here**: it
requires exhibiting near-optimal paths, that is solving the time-reversed relaxation
`γ' = L̃'(γ)` and passing to the limit of a diverging transit time, which is an ODE existence and
limiting argument outside the scope of this file. -/
theorem quasipotentialIncrement_le_fwAction
    {nrm : ℝ} {Lp sigma2 gamma gammaDot : ℝ → ℝ} {T : ℝ} {D : Set ℝ}
    (hnrm : 0 ≤ nrm) (hT : 0 ≤ T)
    (hderiv : ∀ t ∈ Icc (0:ℝ) T, HasDerivAt gamma (gammaDot t) t)
    (hdotc : ContinuousOn gammaDot (Icc (0:ℝ) T))
    (hmaps : ∀ t ∈ Icc (0:ℝ) T, gamma t ∈ D)
    (hLpc : ContinuousOn Lp D) (hsc : ContinuousOn sigma2 D)
    (hspos : ∀ x ∈ D, 0 < sigma2 x) :
    4 * nrm * quasipotentialIncrement Lp sigma2 (gamma 0) (gamma T)
      ≤ fwAction nrm Lp sigma2 gamma gammaDot T := by
  have huT : uIcc (0:ℝ) T = Icc 0 T := uIcc_of_le hT
  have hmt : MapsTo gamma (Icc (0:ℝ) T) D := hmaps
  have hgc : ContinuousOn gamma (Icc (0:ℝ) T) := fun t ht =>
    (hderiv t ht).continuousAt.continuousWithinAt
  have hgD : ContinuousOn (fun u => Lp u / sigma2 u) D :=
    hLpc.div hsc (fun x hx => (hspos x hx).ne')
  have hLpg : ContinuousOn (fun t => Lp (gamma t)) (Icc (0:ℝ) T) := hLpc.comp hgc hmt
  have hsg : ContinuousOn (fun t => sigma2 (gamma t)) (Icc (0:ℝ) T) := hsc.comp hgc hmt
  have hsgne : ∀ t ∈ Icc (0:ℝ) T, sigma2 (gamma t) ≠ 0 := fun t ht => (hspos _ (hmaps t ht)).ne'
  have hcov : (∫ t in (0:ℝ)..T, gammaDot t • ((fun u => Lp u / sigma2 u) ∘ gamma) t)
      = ∫ u in gamma 0..gamma T, Lp u / sigma2 u := by
    refine intervalIntegral.integral_deriv_smul_comp' ?_ ?_ ?_
    · rw [huT]; exact hderiv
    · rw [huT]; exact hdotc
    · refine hgD.mono ?_
      rw [huT]
      rintro x ⟨t, ht, rfl⟩
      exact hmaps t ht
  have hiL : IntervalIntegrable
      (fun t => gammaDot t * (Lp (gamma t) / sigma2 (gamma t))) volume 0 T := by
    apply ContinuousOn.intervalIntegrable
    rw [huT]
    exact hdotc.mul (hLpg.div hsg hsgne)
  have hiR : IntervalIntegrable
      (fun t => (gammaDot t + Lp (gamma t)) ^ 2 / sigma2 (gamma t)) volume 0 T := by
    apply ContinuousOn.intervalIntegrable
    rw [huT]
    exact ((hdotc.add hLpg).pow 2).div hsg hsgne
  have hpt : ∀ t ∈ Icc (0:ℝ) T,
      gammaDot t * (Lp (gamma t) / sigma2 (gamma t))
        ≤ (1/4) * ((gammaDot t + Lp (gamma t)) ^ 2 / sigma2 (gamma t)) := by
    intro t ht
    have hs : 0 < sigma2 (gamma t) := hspos _ (hmaps t ht)
    have key : (1/4) * ((gammaDot t + Lp (gamma t)) ^ 2 / sigma2 (gamma t))
        - gammaDot t * (Lp (gamma t) / sigma2 (gamma t))
        = (gammaDot t - Lp (gamma t)) ^ 2 / (4 * sigma2 (gamma t)) := by
      field_simp
      ring
    nlinarith [div_nonneg (sq_nonneg (gammaDot t - Lp (gamma t)))
      (by positivity : (0:ℝ) ≤ 4 * sigma2 (gamma t)), key]
  have hmono := intervalIntegral.integral_mono_on hT hiL (hiR.const_mul (1/4)) hpt
  rw [intervalIntegral.integral_const_mul] at hmono
  have hQ : quasipotentialIncrement Lp sigma2 (gamma 0) (gamma T)
      ≤ (1/4) * ∫ t in (0:ℝ)..T, (gammaDot t + Lp (gamma t)) ^ 2 / sigma2 (gamma t) := by
    rw [quasipotentialIncrement, ← hcov]
    simpa [smul_eq_mul] using hmono
  have := mul_le_mul_of_nonneg_left hQ (by positivity : (0:ℝ) ≤ 4 * nrm)
  simp only [fwAction]
  nlinarith [this]

/-- The increment (*) bounds the one-dimensional Freidlin–Wentzell quasipotential from below:
over any non-empty family of continuously differentiable paths that run from `x0` to `x` inside
the domain, `4 · nrm · (U_eff(x) - U_eff(x0)) ≤ V(x0, x)`.  This is the path-space form of
`quasipotentialIncrement_le_fwAction`, and it is one half of the classical identification of (*)
with the quasipotential in one dimension; the reverse inequality is not proved here. -/
theorem fwQuasipotential_ge_quasipotentialIncrement {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp sigma2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {x0 x : ℝ} {D : Set ℝ}
    (hnrm : 0 ≤ nrm)
    (hends : ∀ γ, traj γ 0 = x0 ∧ traj γ (hor γ) = x)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hderiv : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), HasDerivAt (traj γ) (vel γ t) t)
    (hdotc : ∀ γ, ContinuousOn (vel γ) (Icc (0:ℝ) (hor γ)))
    (hmaps : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hLpc : ContinuousOn Lp D) (hsc : ContinuousOn sigma2 D)
    (hspos : ∀ y ∈ D, 0 < sigma2 y) :
    4 * nrm * quasipotentialIncrement Lp sigma2 x0 x
      ≤ fwQuasipotential nrm Lp sigma2 traj vel hor := by
  refine le_ciInf fun γ => ?_
  have h := quasipotentialIncrement_le_fwAction (nrm := nrm) (D := D) hnrm (hT γ) (hderiv γ)
    (hdotc γ) (hmaps γ) hLpc hsc hspos
  obtain ⟨h0, hend⟩ := hends γ
  rwa [h0, hend] at h

/-- **FW-2 for the genuine variational quasipotential, in one dimension.**  If the lift's
diffusion dominates the baseline's at every point every admissible path visits, the lift's
quasipotential is at most the baseline's.  This is section 8's FW-2 with the one-dimensional
integrand comparison substituted for the Loewner-order one.

The monotone direction genuinely does hold path by path, so unlike the gap statements below it
needs no near-minimality hypothesis; this is section 8's own "for EVERY admissible path".  The
boundedness-below side condition is discharged from `bddBelow_range_fwAction` rather than
assumed. -/
theorem fwQuasipotential_le_of_diffusion_le {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp s1 s2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    (hnrm : 0 ≤ nrm)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hi1 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s1 (traj γ t)) volume 0 (hor γ))
    (hi2 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s2 (traj γ t)) volume 0 (hor γ)) :
    fwQuasipotential nrm Lp s2 traj vel hor ≤ fwQuasipotential nrm Lp s1 traj vel hor :=
  quasipotential_le_of_action_le
    (bddBelow_range_fwAction (D := D) hnrm hT hin
      (fun x hx => lt_of_lt_of_le (hs1 x hx) (hs12 x hx)))
    fun γ => fwAction_antitone_in_diffusion hnrm (hT γ) (hin γ) hs1 hs12 (hi1 γ) (hi2 γ)

/-- **FW-2 for the genuine variational quasipotential, in one dimension, with section 8's own
strictness hypothesis and with the occupancy quantified the way FW-3a can actually supply it.**

This is the statement of section 8 transcribed: the gap is required only of paths whose baseline
action lies within `ρ` of the baseline quasipotential — section 8's "every `B`-near-minimizing
path", equivalently FW-3a's "every path `γ` with `S[γ] ≤ S̄`" at `S̄ = V_dir + ρ` — and on such a
path the hypothesis asks for a measurable time set of measure at least `τ` on which the velocity
excess `u = γ' + L̃'(γ)` satisfies `|u| ≥ c` and the reciprocal gap `1/σ_dir² - 1/σ_lift²` is at
least `δ`, which is section 8's `uᵀ(Σ_B⁻¹ - Σ_A⁻¹)u ≥ δ‖u‖²` in one dimension.  The conclusion is
section 8's own gap `nrm · δ c² τ`, which at `nrm = 1/4` is the `(δ c²/4) τ` the note writes down.

`fwQuasipotential_le_sub_gap_of_nearMinimizer_bandOccupancy` below is the FW-2' specialization,
in which `δ` is replaced by the pair `(Λ, ε)` of section 11.3 and derived from it; that pair is
the stronger hypothesis, so this is the more general of the two statements.

`hocc` is a hypothesis, not a theorem, here — and it is section 8's hypothesis, not
`BandOccupancy.lean`'s conclusion.  That file proves the deterministic time-in-band core of
FW-3a and carries the action-to-speed implication as a named hypothesis of its own; the
diffusion clause comes from FW-3b and FW-1/(A6); and the velocity floor `c ≤ |u|` is the clause
the note never proves, only argues.  Assembling `hocc` from those pieces is the FW chain's job,
not this file's. -/
theorem fwQuasipotential_le_sub_gap_of_nearMinimizer_reciprocalGap {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp s1 s2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    {cvel delta tau rho : ℝ}
    (hnrm : 0 ≤ nrm) (hcvel : 0 ≤ cvel) (hdelta : 0 ≤ delta) (hrho : 0 < rho)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hocc : ∀ γ, fwAction nrm Lp s1 (traj γ) (vel γ) (hor γ)
          ≤ fwQuasipotential nrm Lp s1 traj vel hor + rho →
        ∃ E : Set ℝ, MeasurableSet E ∧ E ⊆ Ioc 0 (hor γ) ∧ tau ≤ volume.real E ∧
          (∀ t ∈ E, delta ≤ 1 / s1 (traj γ t) - 1 / s2 (traj γ t)) ∧
          (∀ t ∈ E, cvel ≤ |vel γ t + Lp (traj γ t)|))
    (hi1 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s1 (traj γ t)) volume 0 (hor γ))
    (hi2 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s2 (traj γ t)) volume 0 (hor γ)) :
    fwQuasipotential nrm Lp s2 traj vel hor
      ≤ fwQuasipotential nrm Lp s1 traj vel hor - nrm * (tau * (cvel ^ 2 * delta)) := by
  refine quasipotential_le_sub_of_action_le_sub_nearMin hrho
    (bddBelow_range_fwAction (D := D) hnrm hT hin
      (fun x hx => lt_of_lt_of_le (hs1 x hx) (hs12 x hx))) fun γ hγ => ?_
  obtain ⟨E, hE, hEsub, htauE, hgapE, hvelE⟩ := hocc γ hγ
  exact fwAction_le_sub_reciprocalGap hnrm (hT γ) hE hEsub htauE hcvel hdelta
    (fun t ht => hin γ t ⟨ht.1.le, ht.2⟩) hs1 hs12 hgapE hvelE (hi1 γ) (hi2 γ)

/-- **FW-2' for the genuine variational quasipotential, in one dimension, with the explicit gap,
and with the occupancy hypothesis quantified the way FW-3a can actually supply it.**

The occupancy hypothesis `hocc` is required only of paths whose baseline action lies within `ρ`
of the baseline quasipotential — section 8's "every `B`-near-minimizing path", equivalently
FW-3a's "every path `γ` with `S[γ] ≤ S̄`" at `S̄ = V_dir + ρ`.  On such a path the hypothesis asks
for a measurable time set of measure at least `τ` on which the velocity excess
`u = γ' + L̃'(γ)` satisfies `|u| ≥ c`, the baseline diffusion is at most `Λ`, and the lift's
diffusion exceeds it by at least `ε` — the last being assumption (A6) of section 11.3 with
modulus `ε`.  The conclusion carries section 8's own gap, `nrm · δ c² τ` with
`δ = ε/(Λ(Λ+ε))`, which at `nrm = 1/4` is the `(δ c²/4) τ` the note writes down.

Quantifying the occupancy over *every* path instead, as
`fwQuasipotential_le_sub_gap_of_bandOccupancy` below does, is strictly stronger and is not what
FW-3a delivers: section 11.1's `τ(w, Λ, S̄, M)` decreases as `S̄` grows, so over a path space
containing paths of arbitrarily large action no single positive `τ` works.  That corollary is
retained for convenience but this is the statement the FW chain can be fed.

`hocc` is a hypothesis, not a theorem, here — and it is section 8's hypothesis, not
`BandOccupancy.lean`'s conclusion.  That file proves the deterministic time-in-band core of
FW-3a and carries the action-to-speed implication as a named hypothesis of its own; the
diffusion clauses come from FW-3b and FW-1/(A6); and the velocity floor `c ≤ |u|` is the
clause the note never proves, only argues.  Assembling `hocc` from those pieces is the FW
chain's job, not this file's.

This is the corollary of `fwQuasipotential_le_sub_gap_of_nearMinimizer_reciprocalGap` in which
section 8's reciprocal gap `δ` is not assumed but derived from section 11.3's pair `(Λ, ε)`. -/
theorem fwQuasipotential_le_sub_gap_of_nearMinimizer_bandOccupancy {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp s1 s2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    {cvel lam eps tau rho : ℝ}
    (hnrm : 0 ≤ nrm) (hcvel : 0 ≤ cvel) (hlam : 0 < lam) (heps : 0 ≤ eps) (hrho : 0 < rho)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hocc : ∀ γ, fwAction nrm Lp s1 (traj γ) (vel γ) (hor γ)
          ≤ fwQuasipotential nrm Lp s1 traj vel hor + rho →
        ∃ E : Set ℝ, MeasurableSet E ∧ E ⊆ Ioc 0 (hor γ) ∧ tau ≤ volume.real E ∧
          (∀ t ∈ E, s1 (traj γ t) ≤ lam ∧ s1 (traj γ t) + eps ≤ s2 (traj γ t)) ∧
          (∀ t ∈ E, cvel ≤ |vel γ t + Lp (traj γ t)|))
    (hi1 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s1 (traj γ t)) volume 0 (hor γ))
    (hi2 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s2 (traj γ t)) volume 0 (hor γ)) :
    fwQuasipotential nrm Lp s2 traj vel hor
      ≤ fwQuasipotential nrm Lp s1 traj vel hor
          - nrm * (tau * (cvel ^ 2 * eps / (lam * (lam + eps)))) := by
  have hocc' : ∀ γ, fwAction nrm Lp s1 (traj γ) (vel γ) (hor γ)
        ≤ fwQuasipotential nrm Lp s1 traj vel hor + rho →
      ∃ E : Set ℝ, MeasurableSet E ∧ E ⊆ Ioc 0 (hor γ) ∧ tau ≤ volume.real E ∧
        (∀ t ∈ E, eps / (lam * (lam + eps)) ≤ 1 / s1 (traj γ t) - 1 / s2 (traj γ t)) ∧
        (∀ t ∈ E, cvel ≤ |vel γ t + Lp (traj γ t)|) := by
    intro γ hγ
    obtain ⟨E, hE, hEsub, htauE, hband, hvelE⟩ := hocc γ hγ
    refine ⟨E, hE, hEsub, htauE, fun t ht => ?_, hvelE⟩
    obtain ⟨hb1, hb2⟩ := hband t ht
    have hmem : traj γ t ∈ D := hin γ t ⟨(hEsub ht).1.le, (hEsub ht).2⟩
    have h := reciprocal_gap_of_diffusion_excess (m := 1) (p := 1) zero_le_one le_rfl
      (hs1 _ hmem) heps hlam hb1 hb2
    simpa using h
  have h := fwQuasipotential_le_sub_gap_of_nearMinimizer_reciprocalGap (D := D)
    (delta := eps / (lam * (lam + eps))) hnrm hcvel (by positivity) hrho hT hin hs1 hs12
    hocc' hi1 hi2
  rw [mul_div_assoc]
  exact h

/-- **FW-2', strict form, for the genuine variational quasipotential in one dimension**, with the
occupancy hypothesis quantified over near-minimizing paths only.  With a strictly positive
normalization, occupancy time, velocity floor and diffusion excess, the lift's quasipotential
across the band is strictly smaller than the baseline's.  This is FW-2''s own conclusion,
`V_lift ≤ V_dir - (δ c²/4) τ < V_dir`, in the scalar reduction, and it is the theorem of this
file that the FW chain of section 11.4 is meant to consume. -/
theorem fwQuasipotential_lt_of_nearMinimizer_bandOccupancy {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp s1 s2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    {cvel lam eps tau rho : ℝ}
    (hnrm : 0 < nrm) (hcvel : 0 < cvel) (hlam : 0 < lam) (heps : 0 < eps) (htau : 0 < tau)
    (hrho : 0 < rho)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hocc : ∀ γ, fwAction nrm Lp s1 (traj γ) (vel γ) (hor γ)
          ≤ fwQuasipotential nrm Lp s1 traj vel hor + rho →
        ∃ E : Set ℝ, MeasurableSet E ∧ E ⊆ Ioc 0 (hor γ) ∧ tau ≤ volume.real E ∧
          (∀ t ∈ E, s1 (traj γ t) ≤ lam ∧ s1 (traj γ t) + eps ≤ s2 (traj γ t)) ∧
          (∀ t ∈ E, cvel ≤ |vel γ t + Lp (traj γ t)|))
    (hi1 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s1 (traj γ t)) volume 0 (hor γ))
    (hi2 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s2 (traj γ t)) volume 0 (hor γ)) :
    fwQuasipotential nrm Lp s2 traj vel hor < fwQuasipotential nrm Lp s1 traj vel hor := by
  have hgap : 0 < nrm * (tau * (cvel ^ 2 * eps / (lam * (lam + eps)))) := by positivity
  have := fwQuasipotential_le_sub_gap_of_nearMinimizer_bandOccupancy hnrm.le hcvel.le hlam
    heps.le hrho hT hin hs1 hs12 hocc hi1 hi2
  linarith

/-- The same gap under the stronger hypothesis that **every** admissible path, near-minimizing or
not, meets the occupancy condition.  A corollary of the near-minimizer form, obtained by ignoring
the near-minimality premise.  It is the convenient statement when the path space `Γ` has already
been cut down to a sublevel set of the baseline action, and the weaker statement of the two;
FW-3a of section 11.1 supplies its hypothesis only after such a cut, because the occupancy time
it produces shrinks as the admissible action grows. -/
theorem fwQuasipotential_le_sub_gap_of_bandOccupancy {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp s1 s2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    {cvel lam eps tau : ℝ}
    (hnrm : 0 ≤ nrm) (hcvel : 0 ≤ cvel) (hlam : 0 < lam) (heps : 0 ≤ eps)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hocc : ∀ γ, ∃ E : Set ℝ, MeasurableSet E ∧ E ⊆ Ioc 0 (hor γ) ∧ tau ≤ volume.real E ∧
      (∀ t ∈ E, s1 (traj γ t) ≤ lam ∧ s1 (traj γ t) + eps ≤ s2 (traj γ t)) ∧
      (∀ t ∈ E, cvel ≤ |vel γ t + Lp (traj γ t)|))
    (hi1 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s1 (traj γ t)) volume 0 (hor γ))
    (hi2 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s2 (traj γ t)) volume 0 (hor γ)) :
    fwQuasipotential nrm Lp s2 traj vel hor
      ≤ fwQuasipotential nrm Lp s1 traj vel hor
          - nrm * (tau * (cvel ^ 2 * eps / (lam * (lam + eps)))) :=
  fwQuasipotential_le_sub_gap_of_nearMinimizer_bandOccupancy (D := D) (rho := 1) hnrm hcvel hlam
    heps one_pos hT hin hs1 hs12 (fun γ _ => hocc γ) hi1 hi2

/-- The strict form under the same stronger, all-paths occupancy hypothesis; a corollary of
`fwQuasipotential_lt_of_nearMinimizer_bandOccupancy`. -/
theorem fwQuasipotential_lt_of_bandOccupancy {Γ : Type*} [Nonempty Γ]
    {nrm : ℝ} {Lp s1 s2 : ℝ → ℝ} {traj vel : Γ → ℝ → ℝ} {hor : Γ → ℝ} {D : Set ℝ}
    {cvel lam eps tau : ℝ}
    (hnrm : 0 < nrm) (hcvel : 0 < cvel) (hlam : 0 < lam) (heps : 0 < eps) (htau : 0 < tau)
    (hT : ∀ γ, 0 ≤ hor γ)
    (hin : ∀ γ, ∀ t ∈ Icc (0:ℝ) (hor γ), traj γ t ∈ D)
    (hs1 : ∀ x ∈ D, 0 < s1 x) (hs12 : ∀ x ∈ D, s1 x ≤ s2 x)
    (hocc : ∀ γ, ∃ E : Set ℝ, MeasurableSet E ∧ E ⊆ Ioc 0 (hor γ) ∧ tau ≤ volume.real E ∧
      (∀ t ∈ E, s1 (traj γ t) ≤ lam ∧ s1 (traj γ t) + eps ≤ s2 (traj γ t)) ∧
      (∀ t ∈ E, cvel ≤ |vel γ t + Lp (traj γ t)|))
    (hi1 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s1 (traj γ t)) volume 0 (hor γ))
    (hi2 : ∀ γ, IntervalIntegrable
      (fun t => (vel γ t + Lp (traj γ t)) ^ 2 / s2 (traj γ t)) volume 0 (hor γ)) :
    fwQuasipotential nrm Lp s2 traj vel hor < fwQuasipotential nrm Lp s1 traj vel hor :=
  fwQuasipotential_lt_of_nearMinimizer_bandOccupancy (D := D) (rho := 1) hnrm hcvel hlam heps
    htau one_pos hT hin hs1 hs12 (fun γ _ => hocc γ) hi1 hi2

/-! ## Non-vacuity: the hypothesis bundles are satisfiable

The headline statements carry long hypothesis lists, and a theorem whose hypotheses no instance
satisfies is worthless however cleanly it type-checks.  The three theorems of this section
exhibit explicit instances, so the conclusions above are not vacuously true.  In each the
numbers computed are literally the two sides of the inequality that the cited theorem delivers on
that instance, so they also pin the constants: in the two quantitative witnesses the promised gap
is attained with equality, which a silently weakened modulus would fail.  They are witnesses and
nothing more: none of them is a statement about the paper's diffusion.
-/

/-- **Non-vacuity of the closed-form corollary.**  With `L̃' ≡ 1`, `σ_dir² ≡ 1` and an excess
`exc ≡ 1` on `[0,1]`, every hypothesis of `liftQuasipotentialIncrement_lt_direct` holds, and the
two effective actions are `1/2` and `1`.  The first two conjuncts compute the two sides of the
third, which is that theorem applied, so the hypothesis bundle is satisfiable and the strict
conclusion is not vacuous. -/
theorem liftQuasipotentialIncrement_lt_direct_witness :
    quasipotentialIncrement (fun _ => (1:ℝ))
        (fun u => (fun _ => (1:ℝ)) u + (fun _ => (1:ℝ)) u) 0 1 = 1 / 2 ∧
      quasipotentialIncrement (fun _ => (1:ℝ)) (fun _ => (1:ℝ)) 0 1 = 1 ∧
      quasipotentialIncrement (fun _ => (1:ℝ))
          (fun u => (fun _ => (1:ℝ)) u + (fun _ => (1:ℝ)) u) 0 1
        < quasipotentialIncrement (fun _ => (1:ℝ)) (fun _ => (1:ℝ)) 0 1 := by
  refine ⟨by norm_num [quasipotentialIncrement], by norm_num [quasipotentialIncrement], ?_⟩
  exact liftQuasipotentialIncrement_lt_direct (a := 0) (b := 1) (c := 0) (d := 1)
    le_rfl one_pos le_rfl continuousOn_const continuousOn_const continuousOn_const
    (fun _ _ => zero_le_one) (fun _ _ => one_pos) (fun _ _ => zero_le_one)
    (fun _ _ => one_pos) (fun _ _ => one_pos)

/-- **Non-vacuity, and exactness, of the quantitative corollary.**  The same instance, now with
the modulus quantified: `m = Λ = κ = 1` on the band `[c,d] = [0,1] = [a,b]`.  Every hypothesis of
`liftQuasipotentialIncrement_le_sub_gap` holds; the lift's effective action is `1/2`, and the
baseline's minus the promised gap `(d - c) · m κ/(Λ(Λ+κ)) = 1/2` is also `1/2`.  So the bound is
attained with equality on an admissible instance: the constant that FW-2' carries into the FW
chain is exact rather than the residue of a slack estimate, and a formalization that had
weakened it would fail the second conjunct. -/
theorem liftQuasipotentialIncrement_le_sub_gap_witness :
    quasipotentialIncrement (fun _ => (1:ℝ))
        (fun u => (fun _ => (1:ℝ)) u + (fun _ => (1:ℝ)) u) 0 1 = 1 / 2 ∧
      quasipotentialIncrement (fun _ => (1:ℝ)) (fun _ => (1:ℝ)) 0 1
          - (1 - 0) * (1 * 1 / (1 * (1 + 1))) = 1 / 2 ∧
      quasipotentialIncrement (fun _ => (1:ℝ))
          (fun u => (fun _ => (1:ℝ)) u + (fun _ => (1:ℝ)) u) 0 1
        ≤ quasipotentialIncrement (fun _ => (1:ℝ)) (fun _ => (1:ℝ)) 0 1
            - (1 - 0) * (1 * 1 / (1 * (1 + 1))) := by
  refine ⟨by norm_num [quasipotentialIncrement], by norm_num [quasipotentialIncrement], ?_⟩
  refine liftQuasipotentialIncrement_le_sub_gap (a := 0) (b := 1) (c := 0) (d := 1)
    (m := 1) (lam := 1) (kappa := 1) le_rfl zero_le_one le_rfl zero_le_one one_pos zero_le_one
    (fun _ _ => zero_le_one) (fun _ _ => one_pos) (fun _ _ => zero_le_one)
    (fun _ _ => le_rfl) (fun _ _ => le_rfl) (fun _ _ => le_rfl) ?_ ?_
  · simp
  · simp

/-- **Non-vacuity of the variational statement.**  Take a single admissible path, `γ(t) = t` on
`[0,1]`, whose recorded velocity really is its derivative — that is the first conjunct, and it
matters because `fwAction` carries the velocity as a free function.  With `L̃' ≡ 0`,
`σ_dir² ≡ 1`, `σ_lift² ≡ 2`, `c = Λ = ε = τ = ρ = nrm = 1` and the band time set `(0,1]`, every
hypothesis of `fwQuasipotential_lt_of_nearMinimizer_bandOccupancy` holds, and the two
quasipotentials are `1/2` and `1`.  So the strict conclusion of FW-2' as formalized here has
instances. -/
theorem fwQuasipotential_lt_of_nearMinimizer_bandOccupancy_witness :
    (∀ t : ℝ, HasDerivAt (fun r : ℝ => r) 1 t) ∧
      fwQuasipotential (Γ := Unit) 1 (fun _ => (0:ℝ)) (fun _ => (2:ℝ))
          (fun _ r => r) (fun _ _ => (1:ℝ)) (fun _ => (1:ℝ)) = 1 / 2 ∧
      fwQuasipotential (Γ := Unit) 1 (fun _ => (0:ℝ)) (fun _ => (1:ℝ))
          (fun _ r => r) (fun _ _ => (1:ℝ)) (fun _ => (1:ℝ)) = 1 ∧
      fwQuasipotential (Γ := Unit) 1 (fun _ => (0:ℝ)) (fun _ => (2:ℝ))
          (fun _ r => r) (fun _ _ => (1:ℝ)) (fun _ => (1:ℝ))
        < fwQuasipotential (Γ := Unit) 1 (fun _ => (0:ℝ)) (fun _ => (1:ℝ))
          (fun _ r => r) (fun _ _ => (1:ℝ)) (fun _ => (1:ℝ)) := by
  refine ⟨fun t => hasDerivAt_id' t, by simp [fwQuasipotential, fwAction],
    by simp [fwQuasipotential, fwAction], ?_⟩
  refine fwQuasipotential_lt_of_nearMinimizer_bandOccupancy (D := univ) (cvel := 1) (lam := 1)
    (eps := 1) (tau := 1) (rho := 1) one_pos one_pos one_pos one_pos one_pos one_pos
    (fun _ => zero_le_one) (fun _ _ _ => mem_univ _) (fun _ _ => one_pos)
    (fun _ _ => one_le_two) ?_ (fun _ => intervalIntegrable_const)
    (fun _ => intervalIntegrable_const)
  intro γ _
  refine ⟨Ioc 0 1, measurableSet_Ioc, subset_rfl, ?_, fun t _ => ⟨le_rfl, by norm_num⟩,
    fun t _ => ?_⟩
  · simp [measureReal_def, Real.volume_Ioc]
  · norm_num

end IcnnLift
