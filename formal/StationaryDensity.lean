import Mathlib

/-!
# The stationary density of a one-dimensional Itô diffusion with state-dependent noise

This file machine-checks equation `(*)` of `docs/design/lemma1_rederivation.md`, section 3 — the
analytic core of the corrected Lemma 1. The design note replaces the refuted Hessian
decomposition of the shipped proof (its section 1 establishes that differentiation commutes with
the batch expectation, so the pullback Hessian carries no cross-covariance term) with a
statement about the *dynamics*: the bias-channel iterate is modelled as the one-dimensional Itô
diffusion

  `dw = -L̃'(w) dt + √η σ_eff(w) dB_t`,

whose diffusion coefficient is state dependent because the slack-channel coupling enters the
update covariance, not the Hessian. The claim the note needs is that this diffusion has
stationary density

  `π(w) ∝ (1 / σ_eff²(w)) · exp(-(2/η) U_eff(w))`,  `U_eff(w) = ∫^w L̃'(u) / σ_eff²(u) du`.

## The register, and what is and is not proved here

mathlib v4.31.0 contains no stochastic calculus whatsoever: no Itô integral, no stochastic
differential equation, no Fokker–Planck equation, no notion of an invariant measure of a
diffusion, and no exit-time theory. (This was verified by a whole-tree audit, not assumed.) The
passage from the displayed Itô SDE to a partial differential equation for its stationary law is
therefore not available as a mathlib theorem, and it is not proved here. That passage is the
modelling register the paper adopts, following Li et al. (2017) for the step from discrete
stochastic gradient descent to the modified equation, and it is recorded as such in the paper
rather than derived.

What this file does prove — completely, from ordinary real analysis, with no `sorry` and
no added axiom — is that the stated `π` **solves the stationary Fokker–Planck equation**; that
in the zero-flux form every solution, nonnegative or not, is `(*)` with an identified constant; and
that for the full stationary equation every nonnegative solution has vanishing current and is
`(*)`, under a lower-bound hypothesis on the effective potential that is stated explicitly and
whose necessity is itself machine-checked (`exists_nonzero_constant_current`). Writing
`D = (η/2) σ_eff²` for the Fokker–Planck diffusion coefficient of the above SDE and `μ = -L̃'`
for its drift, the stationary Fokker–Planck
equation `∂_t p = -∂_w J = 0` with probability current `J = μ p - ∂_w (D p)` says exactly that
the current is constant in `w`. That equivalence is itself machine-checked rather than asserted:
`exists_constant_current_of_stationary` turns the second-order equation `∂_w J = 0` into a real
constant `J` with `∂_w (D ρ) = μ ρ - J`, which is the hypothesis shape the theorems below
consume, and `probabilityCurrent_eq_const` is its converse. The case of a vanishing constant,

  `μ(w) π(w) = ∂_w (D π)(w)` for every `w`,                                            (ZF)

is the zero-flux form, which the predicate `IsZeroFluxStationary` writes out.

## The boundary condition is derived here, not assumed

The design note obtains `(*)` "in one dimension, with reflecting or natural boundaries". Those
words carry the passage from a constant current to a vanishing one, and it would be easy — and
dishonest — to install that passage as a hypothesis. Two theorems keep it honest, and they
should not be confused with each other.

`isZeroFluxStationary_of_constant_current` is the bookkeeping version: a current that is constant
in `w` and vanishes at a single point vanishes everywhere. It assumes exactly what it needs and
establishes nothing about the diffusion; it is retained only because it names the input.

`zeroFlux_of_constant_current` is the substantive version. It takes a solution of the *full*
stationary Fokker–Planck equation — constant current `J`, with `J` an unknown real number that is
not assumed to vanish — and derives `J = 0` from two conditions that are not about boundaries at
all: that `ρ` is nonnegative, which any density is, and that the scale integral
`w ↦ ∫_0^w exp((2/η) U_eff)` is unbounded above and unbounded below. Since that integrand is
strictly positive the scale integral is increasing, so the second condition is the divergence of
`∫^{+∞} exp((2/η) U_eff)` and of `∫_{-∞} exp((2/η) U_eff)`, which is Feller's non-integrability
criterion for the scale function: a divergent scale integral makes the corresponding endpoint
unattainable by the diffusion, and that unattainability is the operative content of "natural"
boundaries on the line. (The note's alternative reading, reflecting boundaries, is a
bounded-interval notion; the bias channel lives on the whole line, and the whole-line case is
the one treated here.) `unbounded_scaleIntegral_of_bddBelow_potential` discharges the criterion
from a lower bound `m ≤ U_eff` with `m` an arbitrary real, not from `U_eff ≥ 0`: the indefinite
integral that `(*)` writes for `U_eff` is pinned to vanish at whatever base point it is given and
is therefore negative on one side of that point unless the base point is the argmin, so a
nonnegativity hypothesis would have excluded the very potentials `(*)` produces. Boundedness
below is what a confining loss supplies, for every base point at once.
`eq_stationaryDensity_of_constant_current` carries the same arbitrary `m` and packages the two
steps: every nonnegative solution of the stationary Fokker–Planck equation whose effective
potential is bounded below has zero current and is exactly `(*)`.

The mechanism is `eq_of_constant_current`, which solves the constant-current equation outright:

  `D(w) ρ(w) = (K - J ∫_0^w exp((2/η) U_eff)) · exp(-(2/η) U_eff(w))`,
  `K = D(0) ρ(0) exp((2/η) U_eff(0))`.

Setting `J = 0` recovers `(*)`; a nonzero `J` drives the bracket negative at one end of the line
or the other, which a nonnegative `ρ` cannot afford. What is *not* derived is the vanishing of
the current when the potential is unbounded below: there the diffusion may be transient and carry
a genuine nonzero current, so the restriction is real and is carried as the hypothesis `hU0`.
That failure mode is exhibited rather than asserted: `exists_nonzero_constant_current`
constructs a transient instance — constant drift `+1`, unit diffusion, `U_eff(w) = -w` unbounded
below — whose uniform nonnegative stationary solution carries the constant current `J = 1 ≠ 0`
while every hypothesis of `zeroFlux_of_constant_current` except the scale divergences holds, and
the upward scale divergence provably fails. So `hU0` is not removable.

## Further points of honesty

First, `stationaryDensity_zeroFlux` is proved *without* assuming that `σ_eff²` is
differentiable: `σ_eff²` cancels exactly inside the current `D π`, so only its
positivity is needed. Differentiability of `σ_eff²` is a genuine hypothesis of the *other*
statements — `hasDerivAt_stationaryDensity`, which supplies the score `π'/π` that Lemma 1's
consequence (L1') differentiates a second time — and it is carried there and only there.
Second, no step of this file is "to leading order": in one dimension `(*)` is exact, and the
file's identities are exact. The approximations of the corrected mechanism live upstream, in the
covariance expansion of the note's section 2, and are carried by the modules that formalize it.
Third, `exists_eq_stationaryDensity_of_zeroFlux` proves the converse: every solution of (ZF) is
of the stated form. So the file does not merely exhibit a solution, it characterizes the
solution set, which is what entitles the paper to write "the stationary density" rather than "a
stationary density".

Finally, the hypothesis sets are demonstrably satisfiable, and this is checked inside the file
rather than asserted. `not_isZeroFluxStationary_one` shows that `IsZeroFluxStationary` is a
restrictive condition and not a triviality. `stationaryDensity_zeroFlux_example` instantiates the
main theorem at the genuinely state-dependent diffusion `σ_eff²(w) = 1 + w²` with `L̃'(w) = w` and
`η = 2`, and `stationaryDensity_example` computes the resulting density in closed form as
`(1 + w²)^(-3/2)`, a tail no Boltzmann density has. `hasDerivAt_stationaryDensity_example`
instantiates the score at the same state-dependent diffusion, so the `σ`-variation term that
(L1') calls `C(w*)` is exhibited on a diffusion where it is nonzero.
`exists_normalized_stationaryDensity_gaussian` discharges every hypothesis of the
normalization result and produces an actual probability measure, and
`eq_gaussian_of_constant_current` together with `constant_current_gaussian_witness` does the same
for the derived boundary condition at constant noise, while
`eq_stationaryDensity_of_constant_current_example` does it again at the state-dependent
`σ_eff²(w) = 1 + w²`, so the derivation is exercised in the regime the correction is about and
not only in the one the shipped proof already covered.
`exists_constant_current_of_stationary_of_zeroFlux` witnesses the hypotheses of the extraction
`exists_constant_current_of_stationary`, and `exists_nonzero_constant_current` together with
`exists_nonzero_constant_current_mirror` checks the necessity of each of that derivation's two
scale hypotheses by exhibiting the transient counterexamples described above. No statement in
this file is vacuously true.

## Results
* `IsZeroFluxStationary` — the zero-probability-flux form of the stationary Fokker–Planck
  equation, `∀ w, HasDerivAt (fun u => D u * ρ u) (drift w * ρ w) w`.
* `not_isZeroFluxStationary_one` — that predicate is not vacuously true.
* `probabilityCurrent`, `probabilityCurrent_eq_zero`, `hasDerivAt_probabilityCurrent_zero` — the
  current of a zero-flux density vanishes identically, hence so does its divergence.
* `probabilityCurrent_eq_const`, `exists_constant_current_of_stationary`,
  `exists_constant_current_of_stationary_of_zeroFlux` — the second-order stationary equation
  `∂_w J = 0` and the statement "the current is the real constant `J`" are the same condition, so
  nothing is smuggled in by taking the latter as a hypothesis below.
* `isZeroFluxStationary_of_constant_current` — a constant current vanishing at one point vanishes
  everywhere. Bookkeeping only; see the next three entries for the substantive statement.
* `fokkerPlanckDiffusion`, `effectivePotential`, `stationaryDensity` — the objects of `(*)`.
* `stationaryDensity_shift` — shifting `U_eff` by an additive constant, which is all the
  indefinite integral of `(*)` leaves free, rescales `π` by a positive factor.
* `hasDerivAt_effectivePotential` — the fundamental theorem of calculus supplies `U_eff` from a
  continuous `L̃'` and a continuous positive `σ_eff²`, so `(*)`'s antiderivative genuinely exists.
* `diffusion_mul_stationaryDensity` — `D · π = (ηC/2) exp(-(2/η) U)`: the state-dependent
  prefactor `1/σ_eff²` cancels against `D` exactly. This is the mechanism of the whole result.
* `stationaryDensity_zeroFlux` — **the main theorem**: `π` of `(*)` solves (ZF) for the drift
  `-L̃'`.
* `stationaryDensity_zeroFlux_deriv` — the same statement written with `deriv`.
* `stationaryDensity_zeroFlux_of_continuous` — `(*)` with `U_eff` the actual integral.
* `eq_stationaryDensity_of_zeroFlux`, `exists_eq_stationaryDensity_of_zeroFlux`,
  `pos_const_of_zeroFlux` — converse: every zero-flux density is `(*)`, with the proportionality
  constant identified, and that constant is positive whenever the density is.
* `eq_of_constant_current` — the general solution of the stationary Fokker–Planck equation at an
  arbitrary constant current `J`, in closed form.
* `unbounded_scaleIntegral_of_bddBelow_potential` — Feller's non-integrability criterion holds at
  both endpoints whenever `U_eff` is bounded below by some real `m`.
* `zeroFlux_of_constant_current` — **the boundary condition, derived**: a nonnegative solution of
  the stationary Fokker–Planck equation with inaccessible endpoints has vanishing current.
* `eq_stationaryDensity_of_constant_current` — the two combined: every nonnegative stationary
  solution for a potential bounded below is `(*)`, and its current is zero.
* `exists_nonzero_constant_current`, `exists_nonzero_constant_current_mirror` — each scale
  hypothesis of that derivation is separately necessary: two transient diffusions with
  nonnegative stationary solutions of nonzero constant current, one for which the upward scale
  divergence provably fails and one for which the downward one does.
* `stationaryDensity_pos`, `stationaryDensity_support_univ`, `stationaryDensity_integral_pos` —
  positivity, full support, and hence strictly positive total mass for an integrable `(*)`.
* `hasDerivAt_stationaryDensity` — the score `π' = -(σ_eff²'/σ_eff² + (2/η) L̃'/σ_eff²) π`.
* `stationaryDensity_const_diffusion`, `boltzmann_zeroFlux`,
  `exists_boltzmann_of_const_diffusion` — the constant-noise special case: `(*)` reduces to the
  Boltzmann form `exp(-2 L̃ / (η σ_eff²))` that the shipped proof borrowed, so the corrected
  formula contains the shipped one.
* `exists_normalized_stationaryDensity` — under an integrability hypothesis, `π` normalizes to a
  probability density of the same form, which is again zero-flux stationary.
* `stationaryDensity_zeroFlux_example`, `effectivePotential_example`,
  `stationaryDensity_example`, `hasDerivAt_stationaryDensity_example`,
  `integrable_stationaryDensity_gaussian`, `exists_normalized_stationaryDensity_gaussian`,
  `constant_current_gaussian_witness`, `eq_gaussian_of_constant_current`,
  `eq_stationaryDensity_of_constant_current_example` — worked instances witnessing that every
  hypothesis set above is satisfiable, with the state-dependent example computed in closed form
  and the derived boundary condition instantiated at it, not only at constant noise.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses carried, and what it would take to discharge each:

* `hη : η ≠ 0` — the learning rate is nonzero. Not dischargeable: it is a modelling parameter.
* `hσ : ∀ w, 0 < σ2 w` — strict positivity of the effective diffusion `σ_eff²`. In the paper this
  is uniform ellipticity on the shoulder band, assumption (A2)/(FW-3) of the design note; it is a
  hypothesis because a diffusion that degenerates has no density of this form.
* `hU : ∀ w, HasDerivAt U (L' w / σ2 w) w` — that `U_eff` is an antiderivative of
  `L̃'/σ_eff²`. `hasDerivAt_effectivePotential` discharges it whenever `L̃'` and `σ_eff²` are
  continuous and `σ_eff²` is positive, so it is not a real assumption; it is stated separately
  only so that the main theorem does not force a continuity assumption it does not need.
* `hσd : ∀ w, HasDerivAt σ2 (σ2' w) w` — differentiability of `σ_eff²`, used only for the score
  `hasDerivAt_stationaryDensity`. This is assumption (A1)'s regularity clause.
* `hint : Integrable (stationaryDensity C η σ2 U)` — normalizability. This is a genuine
  restriction: `(*)` defines a positive solution of (ZF) whether or not it is integrable, and
  integrability is exactly the condition under which the diffusion is positive recurrent rather
  than transient. It is not dischargeable from the local hypotheses; it is a growth condition on
  `L̃` at infinity.
* `hρnn : ∀ w, 0 ≤ ρ w` — nonnegativity of the candidate solution, in the theorems that derive
  the boundary condition. This is not an analytic input but the definition of a density, and it
  is what rules out a nonzero current.
* `hU0 : ∀ u, m ≤ U u` in `eq_stationaryDensity_of_constant_current`, for an arbitrary real `m`
  carried as an implicit argument — a lower bound on `U_eff`, not a nonnegativity assumption —
  and its weaker unbundled form `hplus`/`hminus` in `zeroFlux_of_constant_current`. This is a
  real restriction rather than bookkeeping: a potential unbounded below can make the diffusion
  transient, and a transient diffusion carries a genuine nonzero current, so the conclusion
  `J = 0` is false without it. That necessity is machine-checked:
  `exists_nonzero_constant_current` satisfies every other hypothesis with current `J = 1`, and
  its potential is bounded below by no `m` at all. It is the analytic content of "reflecting or
  natural boundaries".
* `hη : 0 < η` rather than `η ≠ 0` in the theorems that derive the boundary condition, since the
  sign of `D = (η/2) σ_eff²` is what converts `ρ ≥ 0` into `D ρ ≥ 0`. The main theorem and its
  converse need only `η ≠ 0`.
* `hC : 0 < C`, and `hs : 0 < s2` in the constant-noise statements — positivity of the
  proportionality constant and of the constant noise level, carried by the positivity, support
  and normalization statements. These are sign conventions on the object being exhibited, not
  analytic inputs.

What is **not** closed, and is not closeable in this mathlib: that the diffusion of the displayed
SDE has *any* stationary law, and that a stationary law must satisfy the stationary
Fokker–Planck equation. Both require an Itô calculus and a generator/adjoint theory that
mathlib v4.31.0 does not have. They are the modelling step the paper cites rather than proves,
and no hypothesis in this file silently stands in for them: the file's statements are about the
partial differential equation, which is the object the paper's Lemma 1 and its consequences
(L1') and (L2') actually manipulate. What the file does *not* leave open, and what an earlier
draft of this module did leave open, is the boundary condition: that is now a theorem with
stated and satisfiable hypotheses, not a phrase.
-/

open MeasureTheory

namespace IcnnLift

/-! ## Stationarity as a calculus statement -/

/-- **Zero probability flux.** For a one-dimensional diffusion with drift `drift` and
Fokker–Planck diffusion coefficient `D`, a density `ρ` is *stationary with zero flux* when the
probability current `drift · ρ - ∂_w (D ρ)` vanishes identically, i.e. when

  `drift(w) ρ(w) = ∂_w (D ρ)(w)` for every `w`.

This is the stationary Fokker–Planck equation in one dimension under reflecting or natural
boundary conditions. It is stated with `HasDerivAt` rather than `deriv` so that it also asserts
the differentiability of the current, which `deriv` would silently supply as `0`. -/
def IsZeroFluxStationary (drift D ρ : ℝ → ℝ) : Prop :=
  ∀ w : ℝ, HasDerivAt (fun u => D u * ρ u) (drift w * ρ w) w

/-- **The predicate has content.** `IsZeroFluxStationary` is not satisfied by an arbitrary
triple: for unit drift, unit diffusion coefficient and the constant density `1` the product
`D ρ` is constant, so its derivative is `0`, while the equation demands `drift · ρ = 1`. A
definition that everything satisfied would prove nothing about `(*)`, so this is checked here
rather than asserted. -/
theorem not_isZeroFluxStationary_one :
    ¬ IsZeroFluxStationary (fun _ => (1:ℝ)) (fun _ => (1:ℝ)) (fun _ => (1:ℝ)) := by
  intro h
  have hc : HasDerivAt (fun _ : ℝ => (1:ℝ) * 1) 0 0 := by
    simpa using hasDerivAt_const (0:ℝ) ((1:ℝ) * 1)
  have hone := (h 0).unique hc
  norm_num at hone

/-- The probability current `J = drift · ρ - ∂_w (D ρ)` of a one-dimensional diffusion. -/
noncomputable def probabilityCurrent (drift D ρ : ℝ → ℝ) : ℝ → ℝ :=
  fun w => drift w * ρ w - deriv (fun u => D u * ρ u) w

/-- A real function whose derivative vanishes everywhere is constant. This is the mean value
theorem, in the form used twice below: once for the uniqueness of the zero-flux solution, and
once for the "up to an additive constant" clause of the constant-diffusion corollary. -/
theorem eq_of_hasDerivAt_zero {f : ℝ → ℝ} (h : ∀ w, HasDerivAt f 0 w) (w v : ℝ) : f w = f v :=
  is_const_of_deriv_eq_zero (fun x => (h x).differentiableAt) (fun x => (h x).deriv) w v

/-- A zero-flux stationary density has identically vanishing probability current. -/
theorem probabilityCurrent_eq_zero {drift D ρ : ℝ → ℝ} (h : IsZeroFluxStationary drift D ρ) :
    probabilityCurrent drift D ρ = 0 := by
  funext w
  simp [probabilityCurrent, (h w).deriv]

/-- Consequently the current has vanishing divergence: a zero-flux density solves the stationary
Fokker–Planck equation `∂_t p = -∂_w J = 0`. -/
theorem hasDerivAt_probabilityCurrent_zero {drift D ρ : ℝ → ℝ}
    (h : IsZeroFluxStationary drift D ρ) (w : ℝ) :
    HasDerivAt (probabilityCurrent drift D ρ) 0 w := by
  rw [probabilityCurrent_eq_zero h]
  exact hasDerivAt_const w 0

/-- **A constant current is a constant current.** If the derivative of `D ρ` is `drift · ρ - J`
at every point for a fixed real `J`, then the probability current of `ρ` is the constant function
`J`. This is the trivial half of the equivalence recorded in the module docstring; its converse,
`exists_constant_current_of_stationary`, is the half that matters. -/
theorem probabilityCurrent_eq_const {drift D ρ : ℝ → ℝ} {J : ℝ}
    (hρ : ∀ w, HasDerivAt (fun u => D u * ρ u) (drift w * ρ w - J) w) :
    probabilityCurrent drift D ρ = fun _ => J := by
  funext w
  simp [probabilityCurrent, (hρ w).deriv]

/-- **Stationarity is exactly constancy of the current.** The stationary Fokker–Planck equation
is the second-order statement `∂_t p = -∂_w J = 0`, that is, that the probability current
`J = drift · ρ - ∂_w (D ρ)` has vanishing derivative. This theorem extracts from that statement
the real number `J` and the first-order equation `∂_w (D ρ) = drift · ρ - J`, which is the form in
which `eq_of_constant_current`, `zeroFlux_of_constant_current` and
`eq_stationaryDensity_of_constant_current` take their hypothesis.

It is proved here rather than asserted in prose because those three theorems would otherwise be
open to the reading that "the current is the constant `J`" is a stronger hypothesis than
stationarity. It is not: given only that `D ρ` is differentiable — which the definition of the
current already presupposes, since `probabilityCurrent` is written with `deriv` — the value of
the current at the origin serves as `J`, by the mean value theorem. -/
theorem exists_constant_current_of_stationary {drift D ρ : ℝ → ℝ}
    (hd : ∀ w, DifferentiableAt ℝ (fun u => D u * ρ u) w)
    (hstat : ∀ w, HasDerivAt (probabilityCurrent drift D ρ) 0 w) :
    ∃ J : ℝ, ∀ w, HasDerivAt (fun u => D u * ρ u) (drift w * ρ w - J) w := by
  refine ⟨probabilityCurrent drift D ρ 0, fun w => ?_⟩
  have hJ : probabilityCurrent drift D ρ w = probabilityCurrent drift D ρ 0 :=
    eq_of_hasDerivAt_zero hstat w 0
  have hderiv : deriv (fun u => D u * ρ u) w
      = drift w * ρ w - probabilityCurrent drift D ρ 0 := by
    simp only [probabilityCurrent] at hJ ⊢
    linarith
  have h := (hd w).hasDerivAt
  rwa [hderiv] at h

/-- **The extraction is not vacuous.** Every zero-flux stationary density satisfies both
hypotheses of `exists_constant_current_of_stationary`: the product `D ρ` is differentiable
because `IsZeroFluxStationary` is stated with `HasDerivAt`, and the current vanishes identically
and so has vanishing derivative. Every instance the file constructs below — in particular
`stationaryDensity_zeroFlux_example` at the state-dependent diffusion — therefore inhabits that
hypothesis set, with the extracted constant equal to zero. -/
theorem exists_constant_current_of_stationary_of_zeroFlux {drift D ρ : ℝ → ℝ}
    (h : IsZeroFluxStationary drift D ρ) :
    (∀ w, DifferentiableAt ℝ (fun u => D u * ρ u) w) ∧
      ∀ w, HasDerivAt (probabilityCurrent drift D ρ) 0 w :=
  ⟨fun w => (h w).differentiableAt, hasDerivAt_probabilityCurrent_zero h⟩

/-- **The boundary condition, named.** Stationarity of the law says only that the current `J` is
constant in `w`. If in addition `J` vanishes at a single point `w₀`, then it vanishes everywhere
and the density is zero-flux stationary in the sense of `IsZeroFluxStationary`.

This theorem is bookkeeping, and is labelled as such: under its hypotheses `J` is identically
zero by the mean value theorem, so `hJ` is already the conclusion, and nothing about the
diffusion has been used. Its only purpose is to give the modelling input a name and a place. The
statement that actually *derives* the vanishing of the current — from nonnegativity of the
density and inaccessibility of the endpoints, rather than from an assumption about the current —
is `zeroFlux_of_constant_current` below. -/
theorem isZeroFluxStationary_of_constant_current {drift D ρ J : ℝ → ℝ}
    (hJ : ∀ w, HasDerivAt (fun u => D u * ρ u) (drift w * ρ w - J w) w)
    (hstat : ∀ w, HasDerivAt J 0 w) {w₀ : ℝ} (hzero : J w₀ = 0) :
    IsZeroFluxStationary drift D ρ := by
  intro w
  have hJw : J w = 0 := by rw [eq_of_hasDerivAt_zero hstat w w₀, hzero]
  have h := hJ w
  rwa [hJw, sub_zero] at h

/-! ## The objects of equation `(*)` -/

/-- The Fokker–Planck diffusion coefficient of `dw = -L̃'(w) dt + √η σ_eff(w) dB_t`, namely
`D = (η/2) σ_eff²`. The argument `σ2` is the paper's `σ_eff²`. -/
noncomputable def fokkerPlanckDiffusion (η : ℝ) (σ2 : ℝ → ℝ) : ℝ → ℝ :=
  fun w => η / 2 * σ2 w

/-- The temperature-weighted effective potential `U_eff(w) = ∫_a^w L̃'(u)/σ_eff²(u) du` of
equation `(*)`, with the lower limit `a` made explicit (the paper writes the indefinite `∫^w`,
which fixes `U_eff` only up to an additive constant; changing `a` rescales `π` by a positive
constant and so does not change the measure). -/
noncomputable def effectivePotential (a : ℝ) (L' σ2 : ℝ → ℝ) : ℝ → ℝ :=
  fun w => ∫ u in a..w, L' u / σ2 u

/-- The stationary density of equation `(*)`,
`π(w) = (C / σ_eff²(w)) exp(-(2/η) U_eff(w))`, with the proportionality constant `C` explicit. -/
noncomputable def stationaryDensity (C η : ℝ) (σ2 U : ℝ → ℝ) : ℝ → ℝ :=
  fun w => C / σ2 w * Real.exp (-(2 / η) * U w)

/-- **The additive freedom in `U_eff` is a positive rescaling of `π`.** Equation `(*)` writes
`U_eff` as an indefinite integral, which fixes it only up to an additive constant; shifting
`U_eff` by `k` multiplies `π` by the positive constant `exp(-(2/η) k)` and therefore changes
neither the normalized density nor the proportionality statement `(*)` asserts. It is recorded
so that the choice of base point in `effectivePotential` is visibly immaterial. It is *not* used
to convert a lower bound on `U_eff` into nonnegativity: the lemmas that need a lower bound,
`unbounded_scaleIntegral_of_bddBelow_potential` and
`eq_stationaryDensity_of_constant_current`, take the bound as an arbitrary real `m` and prove
the conclusion for the unshifted `U_eff` itself. -/
theorem stationaryDensity_shift (C η k : ℝ) (σ2 U : ℝ → ℝ) (w : ℝ) :
    stationaryDensity C η σ2 (fun u => U u + k) w
      = Real.exp (-(2 / η) * k) * stationaryDensity C η σ2 U w := by
  simp only [stationaryDensity, mul_add, Real.exp_add]
  ring

/-- The fundamental theorem of calculus supplies the antiderivative that equation `(*)` writes as
an indefinite integral: if `L̃'` is continuous and `σ_eff²` is continuous and everywhere positive,
then `U_eff` really is differentiable with derivative `L̃'/σ_eff²`. Continuity of `L̃'` is
assumption (A1) of the paper; positivity of `σ_eff²` is the non-degeneracy of the diffusion. -/
theorem hasDerivAt_effectivePotential {L' σ2 : ℝ → ℝ} (hL' : Continuous L') (hσc : Continuous σ2)
    (hσ : ∀ w, 0 < σ2 w) (a w : ℝ) :
    HasDerivAt (effectivePotential a L' σ2) (L' w / σ2 w) w := by
  have hcont : Continuous fun u => L' u / σ2 u := hL'.div₀ hσc fun u => (hσ u).ne'
  exact (hcont.integral_hasStrictDerivAt a w).hasDerivAt

/-! ## The main theorem -/

/-- **The cancellation that makes `(*)` work.** The state-dependent prefactor `1/σ_eff²` of the
density cancels exactly against the diffusion coefficient `D = (η/2) σ_eff²`, so the quantity
whose derivative the flux condition constrains is the pure exponential

  `D(w) π(w) = (ηC/2) exp(-(2/η) U_eff(w))`.

Note that this needs only `σ_eff²(w) ≠ 0`; no differentiability of `σ_eff²` is used, here or in
the main theorem. -/
theorem diffusion_mul_stationaryDensity (C η : ℝ) {σ2 : ℝ → ℝ} (U : ℝ → ℝ) (w : ℝ)
    (hσ : σ2 w ≠ 0) :
    fokkerPlanckDiffusion η σ2 w * stationaryDensity C η σ2 U w
      = η * C / 2 * Real.exp (-(2 / η) * U w) := by
  simp only [fokkerPlanckDiffusion, stationaryDensity]
  field_simp

/-- **Equation `(*)` of `docs/design/lemma1_rederivation.md` §3, machine-checked.**

For the one-dimensional Itô diffusion `dw = -L̃'(w) dt + √η σ_eff(w) dB_t`, whose Fokker–Planck
drift is `-L̃'` and whose diffusion coefficient is `D = (η/2) σ_eff²`, the density

  `π(w) = (C / σ_eff²(w)) exp(-(2/η) U_eff(w))`,   `U_eff' = L̃'/σ_eff²`,

satisfies the zero-flux stationary Fokker–Planck equation `-L̃'(w) π(w) = ∂_w (D π)(w)` at every
point. This is the classical stationary density for multiplicative noise, and it is what replaces
the Boltzmann form the shipped proof borrowed; in one dimension it is exact, with no leading-order
truncation.

The hypotheses are exactly: a nonzero learning rate, a strictly positive effective diffusion, and
an antiderivative `U` of `L̃'/σ_eff²`. In particular `σ_eff²` need not be differentiable. -/
theorem stationaryDensity_zeroFlux {η C : ℝ} (hη : η ≠ 0) {σ2 U L' : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w) :
    IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η σ2)
      (stationaryDensity C η σ2 U) := by
  intro w
  have hσw : σ2 w ≠ 0 := (hσ w).ne'
  have hcurrent : (fun u => fokkerPlanckDiffusion η σ2 u * stationaryDensity C η σ2 U u)
      = fun u => η * C / 2 * Real.exp (-(2 / η) * U u) :=
    funext fun u => diffusion_mul_stationaryDensity C η U u (hσ u).ne'
  have hval : ∀ E : ℝ,
      -L' w * (C / σ2 w * E) = η * C / 2 * (E * (-(2 / η) * (L' w / σ2 w))) := by
    intro E
    field_simp
  have hexp : HasDerivAt (fun u => Real.exp (-(2 / η) * U u))
      (Real.exp (-(2 / η) * U w) * (-(2 / η) * (L' w / σ2 w))) w :=
    ((hU w).const_mul (-(2 / η))).exp
  have h := hexp.const_mul (η * C / 2)
  rw [hcurrent]
  simp only [stationaryDensity]
  rw [hval]
  exact h

/-- Equation `(*)` written the way the design note writes it, with `deriv`: the drift times the
density equals the derivative of the diffusion times the density. -/
theorem stationaryDensity_zeroFlux_deriv {η C : ℝ} (hη : η ≠ 0) {σ2 U L' : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w) (w : ℝ) :
    -L' w * stationaryDensity C η σ2 U w
      = deriv (fun u => fokkerPlanckDiffusion η σ2 u * stationaryDensity C η σ2 U u) w :=
  (stationaryDensity_zeroFlux hη hσ hU w).deriv.symm

/-- Equation `(*)` with `U_eff` the honest indefinite integral of the note, obtained by feeding
`hasDerivAt_effectivePotential` into the main theorem. Here the only inputs are continuity of
`L̃'`, continuity and positivity of `σ_eff²`, and `η ≠ 0`. -/
theorem stationaryDensity_zeroFlux_of_continuous {η C : ℝ} (hη : η ≠ 0) {σ2 L' : ℝ → ℝ}
    (hL' : Continuous L') (hσc : Continuous σ2) (hσ : ∀ w, 0 < σ2 w) (a : ℝ) :
    IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η σ2)
      (stationaryDensity C η σ2 (effectivePotential a L' σ2)) :=
  stationaryDensity_zeroFlux hη hσ fun w => hasDerivAt_effectivePotential hL' hσc hσ a w

/-! ## The converse: `(*)` is the only zero-flux solution -/

/-- **Uniqueness, with the constant identified.** Every zero-flux stationary density for the
drift `-L̃'` and diffusion coefficient `(η/2) σ_eff²` is exactly the density `(*)` with
proportionality constant `C = σ_eff²(0) ρ(0) exp((2/η) U_eff(0))`. Together with
`stationaryDensity_zeroFlux` this says that `(*)` describes the whole solution set, which is
what licenses the definite article in "the stationary density". The constant is positive as soon
as `ρ` is positive at one point; that is `pos_const_of_zeroFlux` below, proved rather than
remarked, since it is what makes the identification an identification of densities.

The proof is the classical integrating-factor argument: the zero-flux equation says that the
current `D ρ` solves the linear ODE `(D ρ)' = -(2/η) U_eff' (D ρ)`, so `D ρ · exp((2/η) U_eff)`
has vanishing derivative and is therefore constant. -/
theorem eq_stationaryDensity_of_zeroFlux {η : ℝ} (hη : η ≠ 0) {σ2 U L' ρ : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w)
    (hρ : IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η σ2) ρ) (w : ℝ) :
    ρ w = stationaryDensity (σ2 0 * ρ 0 * Real.exp (2 / η * U 0)) η σ2 U w := by
  have hF : ∀ w, HasDerivAt
      (fun u => fokkerPlanckDiffusion η σ2 u * ρ u * Real.exp (2 / η * U u)) 0 w := by
    intro w
    have hσw : σ2 w ≠ 0 := (hσ w).ne'
    have h1 : HasDerivAt (fun u => fokkerPlanckDiffusion η σ2 u * ρ u) (-L' w * ρ w) w := hρ w
    have h2 : HasDerivAt (fun u => Real.exp (2 / η * U u))
        (Real.exp (2 / η * U w) * (2 / η * (L' w / σ2 w))) w := ((hU w).const_mul (2 / η)).exp
    have hval : ∀ E : ℝ, -L' w * ρ w * E
        + fokkerPlanckDiffusion η σ2 w * ρ w * (E * (2 / η * (L' w / σ2 w))) = 0 := by
      intro E
      simp only [fokkerPlanckDiffusion]
      field_simp
      ring
    rw [← hval (Real.exp (2 / η * U w))]
    exact h1.fun_mul h2
  have hconst : ∀ w, fokkerPlanckDiffusion η σ2 w * ρ w * Real.exp (2 / η * U w)
      = fokkerPlanckDiffusion η σ2 0 * ρ 0 * Real.exp (2 / η * U 0) :=
    fun w => eq_of_hasDerivAt_zero hF w 0
  have hσw : σ2 w ≠ 0 := (hσ w).ne'
  have hexpw : Real.exp (2 / η * U w) ≠ 0 := (Real.exp_pos _).ne'
  have hη2 : η / 2 ≠ 0 := div_ne_zero hη two_ne_zero
  have hkey : σ2 w * ρ w * Real.exp (2 / η * U w)
      = σ2 0 * ρ 0 * Real.exp (2 / η * U 0) := by
    refine mul_left_cancel₀ hη2 ?_
    have h := hconst w
    simp only [fokkerPlanckDiffusion] at h
    linear_combination h
  have hexpneg : Real.exp (-(2 / η) * U w) = (Real.exp (2 / η * U w))⁻¹ := by
    rw [neg_mul, Real.exp_neg]
  have halg : ∀ Ew E0 : ℝ, Ew ≠ 0 → σ2 w * ρ w * Ew = σ2 0 * ρ 0 * E0 →
      ρ w = σ2 0 * ρ 0 * E0 / σ2 w * Ew⁻¹ := by
    intro Ew E0 hEw hh
    field_simp
    linear_combination hh
  simp only [stationaryDensity, hexpneg]
  exact halg _ _ hexpw hkey

/-- **Uniqueness up to a constant**, in the form the design note's `∝` asserts: every zero-flux
stationary density is the density `(*)` for some proportionality constant. -/
theorem exists_eq_stationaryDensity_of_zeroFlux {η : ℝ} (hη : η ≠ 0) {σ2 U L' ρ : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w)
    (hρ : IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η σ2) ρ) :
    ∃ C : ℝ, ∀ w, ρ w = stationaryDensity C η σ2 U w :=
  ⟨σ2 0 * ρ 0 * Real.exp (2 / η * U 0), eq_stationaryDensity_of_zeroFlux hη hσ hU hρ⟩

/-- The proportionality constant that `eq_stationaryDensity_of_zeroFlux` identifies is strictly
positive as soon as the solution is strictly positive at a single point, because `σ_eff²` is
strictly positive and the exponential never vanishes. Together with `stationaryDensity_pos` this
is what turns the identification of solutions of a linear ordinary differential equation into an
identification of *densities*, which is the form the design note's `∝` is asserted in. -/
theorem pos_const_of_zeroFlux {η : ℝ} {σ2 U ρ : ℝ → ℝ} (hσ : ∀ w, 0 < σ2 w) (hρ0 : 0 < ρ 0) :
    0 < σ2 0 * ρ 0 * Real.exp (2 / η * U 0) :=
  mul_pos (mul_pos (hσ 0) hρ0) (Real.exp_pos _)

/-! ## The boundary condition, derived rather than assumed -/

/-- **The general solution at an arbitrary constant current.** Stationarity of the law says that
the probability current `J = -L̃' ρ - ∂_w (D ρ)` is constant in `w`, and nothing more; whether
that constant vanishes is a boundary question. This theorem solves the constant-current equation
outright, for an arbitrary real `J` that is *not* assumed to be zero:

  `D(w) ρ(w) = (K - J ∫_0^w exp((2/η) U_eff)) · exp(-(2/η) U_eff(w))`,
  `K = D(0) ρ(0) exp((2/η) U_eff(0))`.

The proof is the integrating factor of `eq_stationaryDensity_of_zeroFlux` carried one step
further: `(D ρ) exp((2/η) U_eff)` now has derivative `-J exp((2/η) U_eff)` instead of `0`, and
the fundamental theorem of calculus integrates that. Setting `J = 0` recovers equation `(*)`;
the case `J ≠ 0` is what the next two theorems rule out. The integral appearing here is the
classical scale function of the diffusion, normalized to vanish at `0`; its integrand
`exp((2/η) U_eff)` is the scale density `s'`. -/
theorem eq_of_constant_current {η J : ℝ} (hη : η ≠ 0) {σ2 U L' ρ : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w)
    (hρ : ∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion η σ2 u * ρ u)
      (-L' w * ρ w - J) w) (w : ℝ) :
    fokkerPlanckDiffusion η σ2 w * ρ w
      = (fokkerPlanckDiffusion η σ2 0 * ρ 0 * Real.exp (2 / η * U 0)
          - J * ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) * Real.exp (-(2 / η) * U w) := by
  have hUc : Continuous U := continuous_iff_continuousAt.mpr fun x => (hU x).continuousAt
  have hEc : Continuous fun u => Real.exp (2 / η * U u) := (hUc.const_mul (2 / η)).rexp
  have hg : ∀ x : ℝ, HasDerivAt (fun v => ∫ u in (0:ℝ)..v, Real.exp (2 / η * U u))
      (Real.exp (2 / η * U x)) x := fun x => (hEc.integral_hasStrictDerivAt 0 x).hasDerivAt
  have hF : ∀ x : ℝ, HasDerivAt
      (fun v => fokkerPlanckDiffusion η σ2 v * ρ v * Real.exp (2 / η * U v)
        + J * ∫ u in (0:ℝ)..v, Real.exp (2 / η * U u)) 0 x := by
    intro x
    have hσx : σ2 x ≠ 0 := (hσ x).ne'
    have h2 : HasDerivAt (fun u => Real.exp (2 / η * U u))
        (Real.exp (2 / η * U x) * (2 / η * (L' x / σ2 x))) x := ((hU x).const_mul (2 / η)).exp
    have h12 := (hρ x).fun_mul h2
    have h3 := (hg x).const_mul J
    have hval : ∀ E : ℝ, (-L' x * ρ x - J) * E
        + fokkerPlanckDiffusion η σ2 x * ρ x * (E * (2 / η * (L' x / σ2 x))) + J * E = 0 := by
      intro E
      simp only [fokkerPlanckDiffusion]
      field_simp
      ring
    have hsum := h12.add h3
    rwa [hval (Real.exp (2 / η * U x))] at hsum
  have hconst := eq_of_hasDerivAt_zero hF w 0
  simp only [intervalIntegral.integral_same, mul_zero, add_zero] at hconst
  have hE : Real.exp (2 / η * U w) ≠ 0 := (Real.exp_pos _).ne'
  have hneg : Real.exp (-(2 / η) * U w) = (Real.exp (2 / η * U w))⁻¹ := by
    rw [neg_mul, Real.exp_neg]
  have hkey : fokkerPlanckDiffusion η σ2 0 * ρ 0 * Real.exp (2 / η * U 0)
      - J * ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)
      = fokkerPlanckDiffusion η σ2 w * ρ w * Real.exp (2 / η * U w) := by linarith [hconst]
  rw [hneg, hkey, mul_assoc, mul_inv_cancel₀ hE, mul_one]

/-- **Feller's non-integrability criterion, in the form the next theorem consumes.** The scale
integral `w ↦ ∫_0^w exp((2/η) U_eff)` has a strictly positive integrand, so it is increasing;
it is unbounded above exactly when `∫^{+∞} exp((2/η) U_eff)` diverges and unbounded below exactly
when `∫_{-∞} exp((2/η) U_eff)` diverges, which is the classical criterion for the two endpoints
of the line to be inaccessible to the diffusion. Both hold as soon as the effective potential is
bounded below by some `m`, since then the integrand is at least the positive constant
`exp((2/η) m)` and the scale integral dominates the linear function `exp((2/η) m) · w` on the
right and is dominated by it on the left.

The bound is an arbitrary `m`, not `0 ≤ U`: `effectivePotential a L' σ2` vanishes at its base
point `a` and is negative wherever the integral dips below that value, so a nonnegativity
hypothesis would apply to the indefinite integral of `(*)` only when `a` happens to be the
argmin of the effective potential. Boundedness below is what a confining loss supplies, for
every base point at once. -/
theorem unbounded_scaleIntegral_of_bddBelow_potential {η : ℝ} (hη : 0 < η) {U : ℝ → ℝ}
    (hUc : Continuous U) {m : ℝ} (hU0 : ∀ u, m ≤ U u) :
    (∀ M : ℝ, ∃ w : ℝ, M < ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) ∧
      (∀ M : ℝ, ∃ w : ℝ, (∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) < M) := by
  set c : ℝ := Real.exp (2 / η * m) with hc
  have hcpos : 0 < c := Real.exp_pos _
  have hEc : Continuous fun u => Real.exp (2 / η * U u) := (hUc.const_mul (2 / η)).rexp
  have hone : ∀ u : ℝ, c ≤ Real.exp (2 / η * U u) := fun u =>
    Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (hU0 u) (by positivity))
  have hlow : ∀ w : ℝ, 0 ≤ w → c * w ≤ ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u) := by
    intro w hw
    have h : (∫ _u in (0:ℝ)..w, c) ≤ ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u) :=
      intervalIntegral.integral_mono_on hw intervalIntegral.intervalIntegrable_const
        (hEc.intervalIntegrable 0 w) fun x _ => hone x
    simp only [intervalIntegral.integral_const, smul_eq_mul, sub_zero] at h
    nlinarith [h]
  have hhigh : ∀ w : ℝ, w ≤ 0 → (∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) ≤ c * w := by
    intro w hw
    have h : (∫ _u in w..(0:ℝ), c) ≤ ∫ u in w..(0:ℝ), Real.exp (2 / η * U u) :=
      intervalIntegral.integral_mono_on hw intervalIntegral.intervalIntegrable_const
        (hEc.intervalIntegrable w 0) fun x _ => hone x
    simp only [intervalIntegral.integral_const, smul_eq_mul] at h
    rw [intervalIntegral.integral_symm w 0]
    nlinarith [h]
  refine ⟨fun M => ⟨max (M / c) 0 + 1, ?_⟩, fun M => ⟨min (M / c) 0 - 1, ?_⟩⟩
  · have h0 : (0:ℝ) ≤ max (M / c) 0 := le_max_right _ 0
    have h1 : M / c ≤ max (M / c) 0 := le_max_left _ 0
    have h2 := hlow (max (M / c) 0 + 1) (by linarith)
    have h4 : M / c < max (M / c) 0 + 1 := by linarith
    have h3 : M < c * (max (M / c) 0 + 1) := by
      calc M = M / c * c := (div_mul_cancel₀ M hcpos.ne').symm
        _ < (max (M / c) 0 + 1) * c := mul_lt_mul_of_pos_right h4 hcpos
        _ = c * (max (M / c) 0 + 1) := mul_comm _ _
    linarith
  · have h0 : min (M / c) 0 ≤ (0:ℝ) := min_le_right _ 0
    have h1 : min (M / c) 0 ≤ M / c := min_le_left _ 0
    have h2 := hhigh (min (M / c) 0 - 1) (by linarith)
    have h4 : min (M / c) 0 - 1 < M / c := by linarith
    have h3 : c * (min (M / c) 0 - 1) < M := by
      calc c * (min (M / c) 0 - 1) = (min (M / c) 0 - 1) * c := mul_comm _ _
        _ < M / c * c := mul_lt_mul_of_pos_right h4 hcpos
        _ = M := div_mul_cancel₀ M hcpos.ne'
    linarith

/-- **The boundary condition, derived.** Let `ρ` solve the *full* stationary Fokker–Planck
equation for the drift `-L̃'` and the coefficient `D = (η/2) σ_eff²`, that is, let its probability
current be some constant `J` about which nothing is assumed. (By
`exists_constant_current_of_stationary` that hypothesis is exactly the second-order stationary
equation `∂_w J = 0` for a `ρ` whose current is differentiable, so taking it in this form assumes
nothing extra.) If `ρ` is nonnegative — which any density is — and the scale integral of
`unbounded_scaleIntegral_of_bddBelow_potential` is unbounded in both directions, so that neither
endpoint of the line is accessible, then `J = 0` and `ρ` is zero-flux stationary.

This is what the phrase "reflecting or natural boundaries" in section 3 of the design note buys,
proved rather than assumed. The argument is the closed form of `eq_of_constant_current`: the
exponential factor there is strictly positive and `D ρ` is nonnegative, so the bracket
`K - J ∫_0^w exp((2/η) U_eff)` is nonnegative for every `w`; a strictly positive `J` makes it
negative far to the right and a strictly negative `J` makes it negative far to the left.

Unlike the main theorem, this one needs the sign of `η` and not merely `η ≠ 0`, because it is the
positivity of `D = (η/2) σ_eff²` that turns `ρ ≥ 0` into `D ρ ≥ 0`. -/
theorem zeroFlux_of_constant_current {η J : ℝ} (hη : 0 < η) {σ2 U L' ρ : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w) (hρnn : ∀ w, 0 ≤ ρ w)
    (hplus : ∀ M : ℝ, ∃ w : ℝ, M < ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u))
    (hminus : ∀ M : ℝ, ∃ w : ℝ, (∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) < M)
    (hρ : ∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion η σ2 u * ρ u)
      (-L' w * ρ w - J) w) :
    J = 0 ∧ IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η σ2) ρ := by
  have hnn : ∀ w : ℝ, 0 ≤ fokkerPlanckDiffusion η σ2 0 * ρ 0 * Real.exp (2 / η * U 0)
      - J * ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u) := by
    intro w
    have hDpos : 0 < fokkerPlanckDiffusion η σ2 w := by
      have hw := hσ w
      simp only [fokkerPlanckDiffusion]
      positivity
    have h1 : 0 ≤ fokkerPlanckDiffusion η σ2 w * ρ w := mul_nonneg hDpos.le (hρnn w)
    rw [eq_of_constant_current hη.ne' hσ hU hρ w] at h1
    refine le_of_mul_le_mul_right ?_ (Real.exp_pos (-(2 / η) * U w))
    simpa using h1
  have hJ0 : J = 0 := by
    rcases lt_trichotomy J 0 with h | h | h
    · exfalso
      obtain ⟨w, hw⟩ := hminus
        ((fokkerPlanckDiffusion η σ2 0 * ρ 0 * Real.exp (2 / η * U 0)) / J)
      have hlt := (lt_div_iff_of_neg h).mp hw
      have hge := hnn w
      nlinarith
    · exact h
    · exfalso
      obtain ⟨w, hw⟩ := hplus
        ((fokkerPlanckDiffusion η σ2 0 * ρ 0 * Real.exp (2 / η * U 0)) / J)
      have hlt := (div_lt_iff₀ h).mp hw
      have hge := hnn w
      nlinarith
  refine ⟨hJ0, fun w => ?_⟩
  have h := hρ w
  rwa [hJ0, sub_zero] at h

/-- **Every nonnegative stationary solution is `(*)`, and its current vanishes.** This is the
statement the design note needs, with the boundary phrase discharged. Assume the effective
potential is bounded below by *some* real `m` — not by `0`; see
`unbounded_scaleIntegral_of_bddBelow_potential` for why the distinction matters, since the
indefinite integral of `(*)` is pinned to vanish at its base point and dips below `0` on either
side of it unless that base point happens to be the argmin — and let `ρ` be any nonnegative
solution of the stationary Fokker–Planck equation whose current is the constant `J`. Then `J = 0`
and `ρ` is exactly the density of `(*)`, with the proportionality constant identified as
`σ_eff²(0) ρ(0) exp((2/η) U_eff(0))`. The conclusion refers to `U` itself, not to any shifted
version of it, so no bookkeeping is hidden in the choice of `m`.

The hypothesis `hU0` is a genuine restriction and not bookkeeping: a potential unbounded below
can make the diffusion transient, and a transient diffusion carries a genuine nonzero current, so
the conclusion `J = 0` is false without it; `exists_nonzero_constant_current` machine-checks this
by exhibiting a transient instance whose potential is bounded below by no `m` at all. -/
theorem eq_stationaryDensity_of_constant_current {η J : ℝ} (hη : 0 < η) {σ2 U L' ρ : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w) {m : ℝ} (hU0 : ∀ u, m ≤ U u)
    (hρnn : ∀ w, 0 ≤ ρ w)
    (hρ : ∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion η σ2 u * ρ u)
      (-L' w * ρ w - J) w) :
    J = 0 ∧ ∀ w, ρ w = stationaryDensity (σ2 0 * ρ 0 * Real.exp (2 / η * U 0)) η σ2 U w := by
  have hUc : Continuous U := continuous_iff_continuousAt.mpr fun x => (hU x).continuousAt
  obtain ⟨hplus, hminus⟩ := unbounded_scaleIntegral_of_bddBelow_potential hη hUc hU0
  obtain ⟨hJ0, hzf⟩ := zeroFlux_of_constant_current hη hσ hU hρnn hplus hminus hρ
  exact ⟨hJ0, eq_stationaryDensity_of_zeroFlux hη.ne' hσ hU hzf⟩

/-! ## Positivity, the score, and the constant-noise special case -/

/-- The density of `(*)` is strictly positive wherever the diffusion is, for a positive
proportionality constant. -/
theorem stationaryDensity_pos {η C : ℝ} (hC : 0 < C) {σ2 U : ℝ → ℝ} (hσ : ∀ w, 0 < σ2 w) (w : ℝ) :
    0 < stationaryDensity C η σ2 U w := by
  have hw := hσ w
  simp only [stationaryDensity]
  positivity

/-- **The score of the stationary density.** When `σ_eff²` is differentiable — assumption (A1)'s
regularity clause, and the *only* place in this file where that is needed — the density of `(*)`
satisfies

  `π'(w) = -( σ_eff²'(w)/σ_eff²(w) + (2/η) L̃'(w)/σ_eff²(w) ) π(w)`,

i.e. `(log π)' = -σ_eff²'/σ_eff² - (2/η) L̃'/σ_eff²`. The second summand is the temperature-weighted
force of `(*)`; the first is the `σ`-variation correction that consequence (L1') of the design
note calls `C(w*)` and that the Boltzmann form of the shipped proof does not have. -/
theorem hasDerivAt_stationaryDensity {η C : ℝ} (hη : η ≠ 0) {σ2 σ2' U L' : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hσd : ∀ w, HasDerivAt σ2 (σ2' w) w)
    (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w) (w : ℝ) :
    HasDerivAt (stationaryDensity C η σ2 U)
      (-(σ2' w / σ2 w + 2 / η * (L' w / σ2 w)) * stationaryDensity C η σ2 U w) w := by
  have hσw : σ2 w ≠ 0 := (hσ w).ne'
  have h1 : HasDerivAt (fun u => C / σ2 u) ((0 * σ2 w - C * σ2' w) / σ2 w ^ 2) w :=
    (hasDerivAt_const w C).fun_div (hσd w) hσw
  have h2 : HasDerivAt (fun u => Real.exp (-(2 / η) * U u))
      (Real.exp (-(2 / η) * U w) * (-(2 / η) * (L' w / σ2 w))) w :=
    ((hU w).const_mul (-(2 / η))).exp
  have hval : ∀ E : ℝ, -(σ2' w / σ2 w + 2 / η * (L' w / σ2 w)) * (C / σ2 w * E)
      = (0 * σ2 w - C * σ2' w) / σ2 w ^ 2 * E + C / σ2 w * (E * (-(2 / η) * (L' w / σ2 w))) := by
    intro E
    field_simp
    ring
  simp only [stationaryDensity]
  rw [hval]
  exact h1.fun_mul h2

/-- **The constant-noise special case, algebraic form.** When `σ_eff²` is the constant `s²`, the
effective potential `U_eff = L̃/s²` is admissible and `(*)` collapses to the Boltzmann density

  `π(w) = (C/s²) exp(-2 L̃(w) / (η s²))`,

which is exactly the form the shipped proof borrowed. The corrected formula therefore contains
the shipped one as its constant-noise case. -/
theorem stationaryDensity_const_diffusion {η C s2 : ℝ} (hη : η ≠ 0) (hs : s2 ≠ 0) (Lt : ℝ → ℝ)
    (w : ℝ) :
    stationaryDensity C η (fun _ => s2) (fun u => Lt u / s2) w
      = C / s2 * Real.exp (-(2 * Lt w) / (η * s2)) := by
  have harg : -(2 / η) * (Lt w / s2) = -(2 * Lt w) / (η * s2) := by
    field_simp
  simp only [stationaryDensity, harg]

/-- **The constant-noise special case, dynamical form.** With constant diffusion `σ_eff² = s²`,
the Boltzmann density `A exp(-2 L̃/(η s²))` is zero-flux stationary for the drift `-L̃'`. This is
the statement the shipped proof used; here it is a corollary of the corrected `(*)` rather than
an independent borrowing. -/
theorem boltzmann_zeroFlux {η s2 A : ℝ} (hη : η ≠ 0) (hs : 0 < s2) {L' Lt : ℝ → ℝ}
    (hLt : ∀ w, HasDerivAt Lt (L' w) w) :
    IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η (fun _ => s2))
      (fun w => A * Real.exp (-(2 * Lt w) / (η * s2))) := by
  have hfun : (fun w => A * Real.exp (-(2 * Lt w) / (η * s2)))
      = stationaryDensity (A * s2) η (fun _ => s2) (fun u => Lt u / s2) := by
    funext w
    rw [stationaryDensity_const_diffusion hη hs.ne' Lt w, mul_div_assoc, div_self hs.ne', mul_one]
  rw [hfun]
  exact stationaryDensity_zeroFlux hη (fun _ => hs) fun w => (hLt w).div_const s2

/-- **The constant-noise special case, up to the additive constant of `∫^w`.** For *any*
antiderivative `U` of `L̃'/s²` — not only the normalized choice `L̃/s²` — the density of `(*)` at
constant diffusion is proportional to the Boltzmann density, with a positive constant when `C` is
positive. This is the precise content of the design note's "`U_eff = L̃/σ_eff²` up to a
constant". -/
theorem exists_boltzmann_of_const_diffusion {η C s2 : ℝ} (hη : η ≠ 0) (hs : 0 < s2)
    {L' Lt U : ℝ → ℝ} (hLt : ∀ w, HasDerivAt Lt (L' w) w)
    (hU : ∀ w, HasDerivAt U (L' w / s2) w) :
    ∃ A : ℝ, (0 < C → 0 < A) ∧
      ∀ w, stationaryDensity C η (fun _ => s2) U w = A * Real.exp (-(2 * Lt w) / (η * s2)) := by
  have hzero : ∀ w, HasDerivAt (fun u => U u - Lt u / s2) 0 w := by
    intro w
    have h := (hU w).fun_sub ((hLt w).div_const s2)
    simpa using h
  have hk : ∀ w, U w - Lt w / s2 = U 0 - Lt 0 / s2 := fun w => eq_of_hasDerivAt_zero hzero w 0
  refine ⟨C / s2 * Real.exp (-(2 / η) * (U 0 - Lt 0 / s2)), ?_, ?_⟩
  · intro hC
    exact mul_pos (div_pos hC hs) (Real.exp_pos _)
  · intro w
    have hUw : U w = Lt w / s2 + (U 0 - Lt 0 / s2) := by linarith [hk w]
    have harg : -(2 / η) * (Lt w / s2) = -(2 * Lt w) / (η * s2) := by field_simp
    simp only [stationaryDensity, hUw, mul_add, Real.exp_add, harg]
    ring

/-! ## Normalization -/

/-- The support of the density of `(*)` is the whole line, since it is everywhere positive. -/
theorem stationaryDensity_support_univ {η C : ℝ} (hC : 0 < C) {σ2 U : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) : Function.support (stationaryDensity C η σ2 U) = Set.univ := by
  ext w
  simp [Function.mem_support, (stationaryDensity_pos hC hσ w).ne']

/-- An integrable density of the form `(*)` has strictly positive total mass, so it can be
normalized. Positivity of the mass is *derived*, not assumed. -/
theorem stationaryDensity_integral_pos {η C : ℝ} (hC : 0 < C) {σ2 U : ℝ → ℝ}
    (hσ : ∀ w, 0 < σ2 w) (hint : Integrable (stationaryDensity C η σ2 U)) :
    0 < ∫ w : ℝ, stationaryDensity C η σ2 U w := by
  have hnn : (0 : ℝ → ℝ) ≤ stationaryDensity C η σ2 U :=
    fun w => (stationaryDensity_pos hC hσ w).le
  rw [integral_pos_iff_support_of_nonneg hnn hint, stationaryDensity_support_univ hC hσ,
    Real.volume_univ]
  exact ENNReal.zero_lt_top

/-- **`(*)` normalizes to a probability density.** Under the single extra hypothesis that the
density of `(*)` is Lebesgue integrable — a growth condition on `L̃` at infinity, equivalently
positive recurrence of the diffusion, and not something the local hypotheses can supply — there
is a proportionality constant `C'` for which the density of `(*)` is strictly positive, has
total integral `1`, generates a probability measure on the line, and still solves the zero-flux
stationary Fokker–Planck equation. This is the content of the `∝` in equation `(*)`. -/
theorem exists_normalized_stationaryDensity {η C : ℝ} (hη : η ≠ 0) (hC : 0 < C)
    {σ2 U L' : ℝ → ℝ} (hσ : ∀ w, 0 < σ2 w) (hU : ∀ w, HasDerivAt U (L' w / σ2 w) w)
    (hint : Integrable (stationaryDensity C η σ2 U)) :
    ∃ C' : ℝ, 0 < C' ∧ (∀ w, 0 < stationaryDensity C' η σ2 U w) ∧
      (∫ w : ℝ, stationaryDensity C' η σ2 U w) = 1 ∧
      IsProbabilityMeasure
        (volume.withDensity fun w => ENNReal.ofReal (stationaryDensity C' η σ2 U w)) ∧
      IsZeroFluxStationary (fun w => -L' w) (fokkerPlanckDiffusion η σ2)
        (stationaryDensity C' η σ2 U) := by
  set Z : ℝ := ∫ w : ℝ, stationaryDensity C η σ2 U w with hZdef
  have hZ : 0 < Z := stationaryDensity_integral_pos hC hσ hint
  have hCZ : 0 < C / Z := div_pos hC hZ
  have hscale : ∀ w : ℝ, stationaryDensity (C / Z) η σ2 U w = stationaryDensity C η σ2 U w / Z := by
    intro w
    simp only [stationaryDensity]
    ring
  have hone : (∫ w : ℝ, stationaryDensity (C / Z) η σ2 U w) = 1 := by
    rw [integral_congr_ae (Filter.Eventually.of_forall hscale), integral_div, ← hZdef,
      div_self hZ.ne']
  have hintCZ : Integrable (stationaryDensity (C / Z) η σ2 U) := by
    have hfun : stationaryDensity (C / Z) η σ2 U = fun w => stationaryDensity C η σ2 U w / Z :=
      funext hscale
    rw [hfun]
    exact hint.div_const Z
  refine ⟨C / Z, hCZ, fun w => stationaryDensity_pos hCZ hσ w, hone, ?_,
    stationaryDensity_zeroFlux hη hσ hU⟩
  constructor
  rw [withDensity_apply _ MeasurableSet.univ, setLIntegral_univ,
    ← ofReal_integral_eq_lintegral_ofReal hintCZ
      (Filter.Eventually.of_forall fun w => (stationaryDensity_pos hCZ hσ w).le), hone]
  simp

/-! ## Non-vacuity -/

/-- **The main theorem is not vacuous.** Its hypotheses are satisfied by a genuinely
state-dependent diffusion: take `σ_eff²(w) = 1 + w²`, `L̃'(w) = w` and `η = 2`. The resulting
stationary density is `(1 + w²)^(-3/2)` up to normalization, which is exactly the multiplicative
noise behaviour that the Boltzmann form of the shipped proof cannot produce. -/
theorem stationaryDensity_zeroFlux_example :
    IsZeroFluxStationary (fun w => -w) (fokkerPlanckDiffusion 2 (fun w => 1 + w ^ 2))
      (stationaryDensity 1 2 (fun w => 1 + w ^ 2)
        (effectivePotential 0 (fun w => w) (fun w => 1 + w ^ 2))) := by
  have hσc : Continuous fun w : ℝ => 1 + w ^ 2 := by fun_prop
  refine stationaryDensity_zeroFlux_of_continuous (by norm_num) continuous_id hσc
    (fun w => ?_) 0
  show (0 : ℝ) < 1 + w ^ 2
  positivity

/-- **The integrability hypothesis of the normalization theorem is not vacuous.** In the
constant-noise Gaussian case `σ_eff² = 1`, `L̃'(w) = w`, `U_eff(w) = w²/2` and `η = 2`, the
density of `(*)` is `exp(-w²/2)`, which is Lebesgue integrable; so
`exists_normalized_stationaryDensity` genuinely applies and delivers a probability measure. -/
theorem integrable_stationaryDensity_gaussian :
    Integrable (stationaryDensity 1 2 (fun _ => (1 : ℝ)) (fun w => w ^ 2 / 2)) := by
  have hfun : stationaryDensity 1 2 (fun _ => (1 : ℝ)) (fun w => w ^ 2 / 2)
      = fun w => Real.exp (-(1 / 2 : ℝ) * w ^ 2) := by
    funext w
    simp only [stationaryDensity]
    norm_num
    ring_nf
  rw [hfun]
  exact integrable_exp_neg_mul_sq (by norm_num)

/-- The effective potential of the state-dependent example in closed form:
`U_eff(w) = ∫_0^w u/(1+u²) du = ½ log(1+w²)`. Both sides are antiderivatives of `w/(1+w²)` and
agree at `0`, so they agree everywhere. -/
theorem effectivePotential_example (w : ℝ) :
    effectivePotential 0 (fun u => u) (fun u => 1 + u ^ 2) w = Real.log (1 + w ^ 2) / 2 := by
  have hpos : ∀ u : ℝ, (0:ℝ) < 1 + u ^ 2 := fun u => by positivity
  have hL : ∀ x : ℝ, HasDerivAt (fun u : ℝ => Real.log (1 + u ^ 2) / 2) (x / (1 + x ^ 2)) x := by
    intro x
    have hd : HasDerivAt (fun u : ℝ => 1 + u ^ 2) (2 * x) x := by
      simpa using (hasDerivAt_pow 2 x).const_add (1:ℝ)
    have heq : x / (1 + x ^ 2) = 2 * x / (1 + x ^ 2) / 2 := by ring
    rw [heq]
    exact (hd.log (hpos x).ne').div_const 2
  have hE : ∀ x : ℝ, HasDerivAt (effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2))
      (x / (1 + x ^ 2)) x := by
    intro x
    have h := hasDerivAt_effectivePotential (L' := fun u : ℝ => u) (σ2 := fun u : ℝ => 1 + u ^ 2)
      continuous_id (by fun_prop) hpos 0 x
    simpa using h
  have hzero : ∀ x : ℝ, HasDerivAt
      (fun u => effectivePotential 0 (fun v : ℝ => v) (fun v : ℝ => 1 + v ^ 2) u
        - Real.log (1 + u ^ 2) / 2) 0 x := by
    intro x
    simpa using (hE x).fun_sub (hL x)
  have h0 : effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2) 0 = 0 := by
    simp [effectivePotential]
  have h := eq_of_hasDerivAt_zero hzero w 0
  rw [h0] at h
  norm_num at h
  linarith

/-- **The state-dependent example in closed form.** The density that
`stationaryDensity_zeroFlux_example` exhibits is `π(w) = (1 + w²)^(-3/2)`. The exponent `-3/2` is
worth reading: `-1` of it is the prefactor `1/σ_eff²` of `(*)`, and the remaining `-1/2` is the
exponential of the effective potential. A Boltzmann density `exp(-2 L̃/(η s²))` for the same
drift `L̃'(w) = w` would be the Gaussian `exp(-w²/2)`; the state-dependent noise replaces that by
a power-law tail. This is the qualitative difference between `(*)` and the form the shipped proof
borrowed, made explicit on one example. -/
theorem stationaryDensity_example (w : ℝ) :
    stationaryDensity 1 2 (fun u => 1 + u ^ 2)
        (effectivePotential 0 (fun u => u) (fun u => 1 + u ^ 2)) w
      = (1 + w ^ 2) ^ (-(3:ℝ) / 2) := by
  have hpos : (0:ℝ) < 1 + w ^ 2 := by positivity
  have hinv : (1:ℝ) / (1 + w ^ 2) = Real.exp (-Real.log (1 + w ^ 2)) := by
    rw [Real.exp_neg, Real.exp_log hpos, one_div]
  rw [Real.rpow_def_of_pos hpos]
  simp only [stationaryDensity, effectivePotential_example, hinv]
  rw [← Real.exp_add]
  congr 1
  ring

/-- **The score is not vacuous either, and its `σ`-variation term is not zero.** Instantiating
`hasDerivAt_stationaryDensity` at the state-dependent example gives
`π'(w) = -(2w/(1+w²) + (2/2)·w/(1+w²)) π(w)`. The first summand is the `σ`-variation correction
that consequence (L1') of the design note calls `C(w*)`; it is visibly nonzero away from the
origin, which is exactly what the Boltzmann form of the shipped proof cannot produce. -/
theorem hasDerivAt_stationaryDensity_example (w : ℝ) :
    HasDerivAt (stationaryDensity 1 2 (fun u => 1 + u ^ 2)
        (effectivePotential 0 (fun u => u) (fun u => 1 + u ^ 2)))
      (-(2 * w / (1 + w ^ 2) + 2 / 2 * (w / (1 + w ^ 2)))
        * stationaryDensity 1 2 (fun u => 1 + u ^ 2)
            (effectivePotential 0 (fun u => u) (fun u => 1 + u ^ 2)) w) w := by
  have hpos : ∀ u : ℝ, (0:ℝ) < 1 + u ^ 2 := fun u => by positivity
  refine hasDerivAt_stationaryDensity (L' := fun u => u) (σ2' := fun u => 2 * u)
    (by norm_num) hpos (fun x => ?_) (fun x => ?_) w
  · simpa using (hasDerivAt_pow 2 x).const_add (1:ℝ)
  · exact hasDerivAt_effectivePotential (L' := fun u : ℝ => u) (σ2 := fun u : ℝ => 1 + u ^ 2)
      continuous_id (by fun_prop) hpos 0 x

/-- The Gaussian potential `U_eff(w) = w²/2` is an antiderivative of `L̃'(w)/σ_eff²(w) = w/1`. -/
theorem hasDerivAt_gaussianPotential (w : ℝ) :
    HasDerivAt (fun u : ℝ => u ^ 2 / 2) (w / 1) w := by
  simpa using (hasDerivAt_pow 2 w).div_const 2

/-- **The normalization theorem is not vacuous.** All five hypotheses of
`exists_normalized_stationaryDensity` are discharged in the constant-noise Gaussian case
`σ_eff² = 1`, `L̃'(w) = w`, `U_eff(w) = w²/2`, `η = 2`, and the theorem then delivers an honest
probability measure on the line that is still zero-flux stationary. Without this instance the
normalization theorem would be a statement about a hypothesis set no one had exhibited. -/
theorem exists_normalized_stationaryDensity_gaussian :
    ∃ C' : ℝ, 0 < C' ∧ (∀ w, 0 < stationaryDensity C' 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) w) ∧
      (∫ w : ℝ, stationaryDensity C' 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) w) = 1 ∧
      IsProbabilityMeasure (volume.withDensity fun w =>
        ENNReal.ofReal (stationaryDensity C' 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) w)) ∧
      IsZeroFluxStationary (fun w => -w) (fokkerPlanckDiffusion 2 (fun _ => (1:ℝ)))
        (stationaryDensity C' 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2)) :=
  exists_normalized_stationaryDensity (L' := fun w => w) (by norm_num) one_pos
    (fun _ => one_pos) hasDerivAt_gaussianPotential integrable_stationaryDensity_gaussian

/-- **The derived boundary condition is not vacuous.** The Gaussian density of `(*)` satisfies
both hypotheses of `eq_gaussian_of_constant_current` below: it is nonnegative, and it solves the
stationary Fokker–Planck equation with the constant current `J = 0`. So the hypothesis set of the
derivation is inhabited, and the conclusion `J = 0` is not obtained from an empty premise. -/
theorem constant_current_gaussian_witness :
    (∀ w, 0 ≤ stationaryDensity 1 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) w) ∧
      ∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion 2 (fun _ => (1:ℝ)) u *
          stationaryDensity 1 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) u)
        (-w * stationaryDensity 1 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) w - 0) w := by
  refine ⟨fun w => (stationaryDensity_pos one_pos (fun _ => one_pos) w).le, fun w => ?_⟩
  simpa using stationaryDensity_zeroFlux (L' := fun w => w) (by norm_num) (fun _ => one_pos)
    hasDerivAt_gaussianPotential w

/-- **The derived boundary condition, instantiated.** For the Ornstein–Uhlenbeck data
`σ_eff² = 1`, `L̃'(w) = w`, `η = 2`, every nonnegative solution of the stationary Fokker–Planck
equation whose current is some constant `J` has `J = 0` and equals the Gaussian density of `(*)`
normalized by its value at the origin. Nothing here assumes the current vanishes; that is the
conclusion. -/
theorem eq_gaussian_of_constant_current {J : ℝ} {ρ : ℝ → ℝ} (hρnn : ∀ w, 0 ≤ ρ w)
    (hρ : ∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion 2 (fun _ => (1:ℝ)) u * ρ u)
      (-w * ρ w - J) w) :
    J = 0 ∧ ∀ w, ρ w = stationaryDensity (ρ 0) 2 (fun _ => (1:ℝ)) (fun u => u ^ 2 / 2) w := by
  have h := eq_stationaryDensity_of_constant_current (L' := fun w : ℝ => w) (m := 0)
    (by norm_num) (fun _ => one_pos) hasDerivAt_gaussianPotential (fun u => by positivity)
    hρnn hρ
  simpa using h

/-- **The derived boundary condition at a genuinely state-dependent diffusion.** The Gaussian
instance above discharges the hypotheses of `eq_stationaryDensity_of_constant_current` only at
constant noise, which is precisely the case the corrected `(*)` is meant to supersede, so the
same is done here for `σ_eff²(w) = 1 + w²`, `L̃'(w) = w`, `η = 2`. Its effective potential
`U_eff(w) = ½ log(1 + w²)` is bounded below by `0`, so the scale integral diverges at both
endpoints, and the conclusion is that every nonnegative solution of the stationary Fokker–Planck
equation for this multiplicative-noise diffusion has vanishing current and is the power law
`ρ(w) = ρ(0) (1 + w²)^(-3/2)`. No Boltzmann density has that tail, so this instance shows the
derivation of the boundary condition is available in the regime the correction is about, and not
only in the constant-noise regime the shipped proof already covered. -/
theorem eq_stationaryDensity_of_constant_current_example {J : ℝ} {ρ : ℝ → ℝ}
    (hρnn : ∀ w, 0 ≤ ρ w)
    (hρ : ∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion 2 (fun v : ℝ => 1 + v ^ 2) u * ρ u)
      (-w * ρ w - J) w) :
    J = 0 ∧ ∀ w, ρ w = ρ 0 * (1 + w ^ 2) ^ (-(3:ℝ) / 2) := by
  have hpos : ∀ u : ℝ, (0:ℝ) < 1 + u ^ 2 := fun u => by positivity
  have hU : ∀ w : ℝ, HasDerivAt (effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2))
      (w / (1 + w ^ 2)) w := fun w =>
    hasDerivAt_effectivePotential (L' := fun u : ℝ => u) (σ2 := fun u : ℝ => 1 + u ^ 2)
      continuous_id (by fun_prop) hpos 0 w
  have hU0 : ∀ u : ℝ,
      (0:ℝ) ≤ effectivePotential 0 (fun v : ℝ => v) (fun v : ℝ => 1 + v ^ 2) u := by
    intro u
    rw [effectivePotential_example]
    have h1 : (1:ℝ) ≤ 1 + u ^ 2 := by nlinarith [sq_nonneg u]
    have h2 : (0:ℝ) ≤ Real.log (1 + u ^ 2) := Real.log_nonneg h1
    linarith
  obtain ⟨hJ, heq⟩ := eq_stationaryDensity_of_constant_current (L' := fun w : ℝ => w) (m := 0)
    (by norm_num) hpos hU hU0 hρnn hρ
  have hc : effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2) 0 = 0 := by
    simp [effectivePotential]
  refine ⟨hJ, fun w => ?_⟩
  have hmul : ∀ Cc : ℝ, stationaryDensity Cc 2 (fun u : ℝ => 1 + u ^ 2)
      (effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2)) w
      = Cc * (1 + w ^ 2) ^ (-(3:ℝ) / 2) := by
    intro Cc
    have h1 : stationaryDensity Cc 2 (fun u : ℝ => 1 + u ^ 2)
        (effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2)) w
        = Cc * stationaryDensity 1 2 (fun u : ℝ => 1 + u ^ 2)
            (effectivePotential 0 (fun u : ℝ => u) (fun u : ℝ => 1 + u ^ 2)) w := by
      simp only [stationaryDensity]
      ring
    rw [h1, stationaryDensity_example]
  rw [heq w, hmul]
  norm_num [hc]

/-- **The upward scale hypothesis of the derived boundary condition is necessary — and it is
the only thing that fails.** The docstrings above assert that a potential unbounded below can
make the diffusion transient and that a transient diffusion carries a genuine nonzero current;
this theorem checks the assertion rather than leaving it as prose. For `η = 2`, unit effective
diffusion `σ_eff² ≡ 1` and the constant force `L̃'(w) = -1` — Fokker–Planck drift `+1`,
effective potential `U_eff(w) = -w` — the uniform function `ρ ≡ 1` is a nonnegative solution of
the full stationary Fokker–Planck equation with constant current `J = 1 ≠ 0`: mass streams to
`+∞` forever. The statement records that every hypothesis of `zeroFlux_of_constant_current`
other than the upward divergence is satisfied — the downward divergence `hminus` *holds*, since
the scale integral `∫_0^w e^{-u} du = 1 - e^{-w}` tends to `-∞` — that `U_eff` is bounded below
by no `m`, so `hU0` of `eq_stationaryDensity_of_constant_current` fails for every `m`, and that
the upward divergence `hplus` provably fails, because that same scale integral stays below `1`.
Consequently `hplus` alone cannot be dropped, and neither can `hU0`: without them the conclusion
`J = 0` is false, exactly as claimed where those hypotheses are introduced. The reflected
instance `exists_nonzero_constant_current_mirror` below does the same for `hminus`. -/
theorem exists_nonzero_constant_current :
    ∃ (η J : ℝ) (σ2 U L' ρ : ℝ → ℝ), 0 < η ∧ J ≠ 0 ∧ (∀ w, 0 < σ2 w) ∧
      (∀ w, HasDerivAt U (L' w / σ2 w) w) ∧ (∀ w, 0 ≤ ρ w) ∧
      (∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion η σ2 u * ρ u) (-L' w * ρ w - J) w) ∧
      (∀ M : ℝ, ∃ w : ℝ, (∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) < M) ∧
      (∀ m : ℝ, ∃ u : ℝ, U u < m) ∧
      ¬ (∀ M : ℝ, ∃ w : ℝ, M < ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) := by
  refine ⟨2, 1, fun _ => 1, fun u => -u, fun _ => -1, fun _ => 1, two_pos, one_ne_zero,
    fun _ => one_pos, fun w => ?_, fun _ => zero_le_one, fun w => ?_, ?_, ?_, ?_⟩
  · -- `U(w) = -w` is an antiderivative of `L̃'/σ_eff² = (-1)/1`.
    simpa using hasDerivAt_neg' w
  · -- `D ρ ≡ 1` is constant, and the prescribed current value `-L̃' ρ - J = 1 - 1` is `0`.
    have hfun : (fun u : ℝ => fokkerPlanckDiffusion 2 (fun _ : ℝ => (1:ℝ)) u
        * (fun _ : ℝ => (1:ℝ)) u) = fun _ : ℝ => (1:ℝ) := by
      funext u
      norm_num [fokkerPlanckDiffusion]
    rw [hfun]
    norm_num
    exact hasDerivAt_const w 1
  · -- The downward divergence holds: the scale integral is `1 - e^{-w} → -∞` as `w → -∞`.
    intro M
    refine ⟨-(|M| + 2), ?_⟩
    have h1 : |M| + 2 + 1 ≤ Real.exp (|M| + 2) := Real.add_one_le_exp _
    have h2 : -|M| ≤ M := neg_abs_le M
    norm_num
    linarith
  · -- `U_eff(w) = -w` is bounded below by no `m`.
    intro m
    exact ⟨-m + 1, by norm_num⟩
  · -- The upward divergence fails: the same scale integral stays below `1`, so `hplus` fails
    -- at `M = 1`.
    intro hplus
    obtain ⟨w, hw⟩ := hplus 1
    norm_num at hw
    nlinarith [Real.exp_pos (-w)]

/-- **The downward scale hypothesis is necessary too.** The reflection of
`exists_nonzero_constant_current`: for `η = 2`, unit effective diffusion and the constant force
`L̃'(w) = 1` — Fokker–Planck drift `-1`, effective potential `U_eff(w) = w` — the uniform
function `ρ ≡ 1` is a nonnegative solution of the full stationary Fokker–Planck equation with
constant current `J = -1 ≠ 0`: mass streams to `-∞` forever. Here the upward divergence
`hplus` *holds*, since the scale integral `∫_0^w e^u du = e^w - 1` is unbounded above, and the
downward divergence `hminus` provably fails, because that integral stays above `-1`.
Consequently `hminus` alone cannot be dropped from `zeroFlux_of_constant_current` either: the
two scale hypotheses are each necessary, not merely jointly. -/
theorem exists_nonzero_constant_current_mirror :
    ∃ (η J : ℝ) (σ2 U L' ρ : ℝ → ℝ), 0 < η ∧ J ≠ 0 ∧ (∀ w, 0 < σ2 w) ∧
      (∀ w, HasDerivAt U (L' w / σ2 w) w) ∧ (∀ w, 0 ≤ ρ w) ∧
      (∀ w, HasDerivAt (fun u => fokkerPlanckDiffusion η σ2 u * ρ u) (-L' w * ρ w - J) w) ∧
      (∀ M : ℝ, ∃ w : ℝ, M < ∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) ∧
      ¬ (∀ M : ℝ, ∃ w : ℝ, (∫ u in (0:ℝ)..w, Real.exp (2 / η * U u)) < M) := by
  refine ⟨2, -1, fun _ => 1, fun u => u, fun _ => 1, fun _ => 1, two_pos, by norm_num,
    fun _ => one_pos, fun w => ?_, fun _ => zero_le_one, fun w => ?_, ?_, ?_⟩
  · -- `U(w) = w` is an antiderivative of `L̃'/σ_eff² = 1/1`.
    simpa using hasDerivAt_id' w
  · -- `D ρ ≡ 1` is constant, and the prescribed current value `-L̃' ρ - J = -1 + 1` is `0`.
    have hfun : (fun u : ℝ => fokkerPlanckDiffusion 2 (fun _ : ℝ => (1:ℝ)) u
        * (fun _ : ℝ => (1:ℝ)) u) = fun _ : ℝ => (1:ℝ) := by
      funext u
      norm_num [fokkerPlanckDiffusion]
    rw [hfun]
    norm_num
    exact hasDerivAt_const w 1
  · -- The upward divergence holds: the scale integral is `e^w - 1 → ∞` as `w → ∞`.
    intro M
    refine ⟨|M| + 2, ?_⟩
    have h1 : |M| + 2 + 1 ≤ Real.exp (|M| + 2) := Real.add_one_le_exp _
    have h2 : M ≤ |M| := le_abs_self M
    norm_num
    linarith
  · -- The downward divergence fails: the same scale integral stays above `-1`, so `hminus`
    -- fails at `M = -1`.
    intro hminus
    obtain ⟨w, hw⟩ := hminus (-1)
    norm_num at hw
    nlinarith [Real.exp_pos w]

end IcnnLift
