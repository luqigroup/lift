import Mathlib
import CrossCov

/-!
# The shoulder, and the once-versus-twice attenuation count

This file machine-checks the shoulder facts of the paper's Section 2
(`docs/paper/v4/02_shoulder.tex`) and the attenuation-count asymmetry that opens Section 4
(`docs/paper/v4/04_mechanism.tex`, the paragraph beginning "The two terms are not attenuated
alike").

Section 2 parametrizes the constrained weights of an input-convex network through a positivity
map, `θ = ψ(θ̃)`, and observes that gradients reach the free coordinate only through the
positivity map's derivative, `∇_{θ̃}L = ψ'(θ̃) ⊙ ∇_θ L` (`eq:chain`). For the softplus positivity map
`ψ(w) = log(1 + e^w)` the prefactor `ψ'` is the logistic function; it is strictly positive,
strictly increasing, and tends to zero at `-∞`, so the sublevel set on which it has collapsed
below a tolerance `σ_s` — the **shoulder** `S_{σ_s} = {w̃ : ψ'(w̃) < σ_s}` of
`eq:shoulder` — is a genuine left-infinite half-line rather than a thin set. All of that is
proved here from `HasDerivAt`, with the half-line's right endpoint computed exactly as
`log(σ_s/(1-σ_s))`, and with the corresponding edge in the constrained coordinate computed as
`ψ(log(σ_s/(1-σ_s))) = -log(1-σ_s)`, the closed form behind the paper's `0.051` at
`σ_s = 0.05` (the identity is machine-checked; the decimal arithmetic is not).

The generality of the phenomenon is a claim the paper makes, but a reader of `docs/paper/v4/` will
not find it stated there, and this is worth saying plainly because the assignment for this file
cited it by label. The remark that carried it, `rem:positivity-generality`, was not migrated into
the sectioned v4 source — `log.md` records the decision, and `docs/handoff/HANDOFF.md` still lists
the label among the manuscript's formal statements. Its text survives in the pre-split manuscript
`research/paper_icnn_lift.tex`, which this repository keeps only on the branch
`worktree-agent-a1027a26523ab1482`, and reads "The lift analysis below uses only that
`ψ : ℝ → ℝ_{≥0}` is differentiable monotone, with a chain-rule prefactor `ψ'` that is small on an
extended region of parameter space, so chain-rule attenuation is genuinely active on a non-trivial
subset of parameter space". It continues that the non-differentiable projection limit
`ψ(u) = max(u, 0)` is the degenerate endpoint at which "the gated coordinates have exactly zero
gradient". Both halves are proved here rather than exhibited. `shoulder_general`: for every
differentiable non-decreasing `ψ` whose derivative tends to zero at `-∞`, and every tolerance
`s > 0`, there is a left ray on which the prefactor is below `s` and every constrained gradient is
attenuated to at most `s` times its constrained size — softplus and the exponential positivity map are
then instances rather than the content, and the positivity map's own non-negativity, which the remark
asks for, turns out not to be needed. `projection_limit_gradient_zero`: below the origin the
projection positivity map passes exactly zero gradient, so the attenuation of the smooth family
degenerates to annihilation at the endpoint the remark names.

Section 4's opening asymmetry is the reason the lift has a channel the direct parametrization
has lost. The objective term of the effective diffusion is the variance of an **already
attenuated** gradient, so under (A2)'s single-prefactor idealization `ψ'(θ̃) = s·I_d` it carries
`s` twice; the cross-covariance `σ²_Jac = tr E[δθ̃ δg^⊤]` carries it **once**, because only `δg`
passes through `ψ'` and `δθ̃`, a re-evaluation of the latent weight rather than a gradient step, does
not. Both counts are derived here, not assumed, and both are proved twice over: once for the
running estimator `crossCovEstimator` of `CrossCov.lean`, and once for the population object
`tr E[δθ̃ δg^⊤]` that `eq:cross-cov` actually defines `σ²_Jac` to be, written as a Bochner
integral (`popCrossCovTrace_attenuated_once`, `popCrossCovTrace_attenuated_twice`). The population
forms carry no integrability hypothesis, because a real scalar passes through `∫` unconditionally.
The consequence, that the ratio `ϱ = σ²_Jac/(σ_s² σ²_obj)` diverges as the positivity map flattens — the
ratio Section 4 forms, with `σ²_obj` read on the constrained coordinate before the prefactor — is
proved as a genuine `Tendsto … atTop` along `𝓝[>] 0`, for the estimator
(`crossCovRatio_tendsto_atTop`) and for the population objects
(`popCrossCovRatio_tendsto_atTop`) alike.

**The power of the prefactor, and where the correction actually lives.** The superseded
pre-split manuscript — the same `research/paper_icnn_lift.tex` named above — called `σ²_Jac`
*unattenuated*, that is `Θ(1)` in the prefactor, and the authors' own design note settles that
this was wrong (`docs/design/lemma1_rederivation.md` §11.2, which audits the occurrences line by
line): since `δg = ψ'(θ̃) ⊙ ∇_θ L` already contains one factor of the prefactor,
`σ²_Jac = Θ(s)`, attenuated once. The v4 source formalized here already carries the corrected
count in prose — Section 4 says `σ²_Jac` "carries that derivative once", and has in every
revision of `04_mechanism.tex` — so this file corrects nothing in the current text; it
machine-checks the corrected power that the text states and the design note derived.
`crossCovEstimator_attenuated_once` is the exact statement of it, and `attenuation_asymmetry`
records both halves of the corrected reading at once: the cross-covariance channel does vanish
as `s → 0⁺` — so it is *not* unattenuated — and yet the ratio to the twice-attenuated objective
channel still diverges, which is all the downstream argument ever needed. One occurrence of
"unattenuated" does survive in v4, in `A2_experiments.tex`'s drift-free bias-channel simulation,
where it attaches to the injected latent weight dial of that synthetic SDE rather than to the
population `σ²_Jac`; the design note classifies that usage as scoped to the synthetic model and
defensible, while cautioning that the dial does not share the real quantity's `s`-scaling.

**(A2) is load-bearing, and Section 7 of this file measures how much — but the measurement cuts
in the paper's favour.** The counts just described are stated for a single scalar prefactor, which
is what (A2) idealizes `diag(ψ'(θ̃))` to. Section 7 of this file puts the diagonal back and asks
what survives. A two-sided band `c₁σ_s ≤ ψ'(θ̃_i) ≤ c₂σ_s` with merely *bounded* spread `c₂/c₁`
is **not** enough: `crossCovRatio_diag_sign_flip` exhibits a two-coordinate window on which the
unattenuated coupling trace is `+1`, every coordinate's prefactor lies between `σ_s` and `10σ_s`,
and the attenuated coupling trace is `-8σ_s`, so the ratio tends to `-∞` and the divergence claim
is false. Two different hypotheses repair it, and which one is invoked matters. The first is a
sign condition the paper never states: `crossCovRatio_diag_tendsto_atTop` proves the divergence
for any diagonal band with `0 < c₁ ≤ c₂`, demanded only on a window `(0, s₀)` of prefactor
scales, provided the coupling `δθ̃_i δ(∇_θL)_i` is coordinatewise non-negative on the window. The
second is (A2) itself, read as the quantitative statement it makes rather than as a bounded band:
`crossCovRatio_diag_tendsto_atTop_of_small_spread` proves the divergence with **no** sign
condition, from `|ψ'(θ̃_i) - σ_s| ≤ ε σ_s` together with the threshold
`ε · couplingAbs < σ̂²_Jac`, where `couplingAbs` is the `ℓ¹` mass of the coordinatewise coupling —
that is, the relative spread has only to be smaller than the fraction of that mass which survives
cancellation in the signed trace. Since (A2) says the derivative is *approximately constant*
across the operative band, this is the reading the paper is entitled to, and it suffices;
`crossCovRatio_diag_small_spread_rescues_sign_flip` runs it on the very data of the counterexample,
where a relative spread of `1/4` restores the divergence that a spread of ten destroys. What a
referee should take from Section 7 is therefore not that the paper needs an unstated sign
condition, but that (A2) has to be read quantitatively: a merely bounded ratio `c₂/c₁` does not
do the work, and how small the spread must be is set by the cancellation in the coupling.

## Results
* `hasDerivAt_positivityMap_comp` — the chain rule of `eq:chain` as a derivative statement: if `L` is
  differentiable at `ψ(w)` with derivative `g` and `ψ` is differentiable at `w` with derivative
  `p`, then `L ∘ ψ` is differentiable at `w` with derivative `g * p`.
* `hasDerivAt_positivityMap_coordinate` — its coordinatewise form, which is the Hadamard product
  `∇_{θ̃}L = ψ'(θ̃) ⊙ ∇_θ L` one coordinate at a time.
* `hasFDerivAt_positivityMap_pi`, `fderiv_positivityMap_pi_single` — `eq:chain` with all `d` coordinates
  perturbed at once: the pullback is jointly differentiable with differential the constrained
  one precomposed with `v ↦ ψ'(θ̃) ⊙ v`, and its `i`-th partial derivative is
  `ψ'(θ̃_i) · (∇_θ L)_i`.
* `hasDerivAt_softplus`, `deriv_softplus_eq_logistic` — `ψ' = logistic` for the softplus positivity map.
* `logistic_pos`, `logistic_lt_one`, `logistic_strictMono`, `tendsto_logistic_atBot` — the
  prefactor is strictly positive, strictly increasing, and collapses to zero at `-∞`.
* `tendsto_logistic_div_exp_atBot` — the decay is exponential: `ψ'(w̃)/e^{w̃} → 1` at `-∞`.
* `tendsto_inv_logistic_atBot` — "the attenuation grows without bound": `1/ψ' → ∞` at `-∞`.
* `logistic_zero`, `logistic_neg_five_lt_one_and_a_half_percent` — Section 2's displayed numeric:
  a free parameter at `-5` passes under `1.5%` of the gradient it passes at zero.
* `shoulder_eq_deriv_sublevel` — the shoulder is the sublevel set of `deriv softplus`, so the
  identification `ψ' = logistic` is derived and not built into the definition.
* `shoulder_eq_Iio`, `shoulder_closed_eq_Iic` — the shoulder of `eq:shoulder` is the half-line
  `(-∞, log(s/(1-s)))`, and its closed version is `(-∞, log(s/(1-s))]`.
* `softplus_shoulder_edge` — the same edge read in the constrained coordinate: `-log(1-s)`.
* `mem_shoulder_of_softplus_lt` — Section 2's reading of that edge: every constrained weight
  below `-log(1-s)` lies in the shoulder.
* `shoulder_eq_empty_of_nonpos`, `shoulder_eq_univ_of_one_le` — the two degenerate tolerances.
* `shoulder_general` — `rem:positivity-generality` proved rather than exhibited.
* `shoulder_softplus`, `shoulder_exp` — its two instances.
* `chainRule_attenuated` — the attenuation inequality behind both: where the prefactor is at
  most `s`, the free-coordinate gradient is at most `s` times the constrained one.
* `shoulderOf_contains_ray` — a prefactor collapsing at `-∞` has a whole left ray in every
  sublevel set, which is what "extended region, not a thin set" asserts.
* `hasDerivAt_projection_of_neg`, `projection_limit_gradient_zero` — the same remark's projection
  limit: below the origin the gated coordinate passes exactly zero gradient, for an arbitrary
  loss and with no differentiability hypothesis on it.
* `projection_limit_gradient_zero_of_hasDerivAt` — the same read through `eq:chain`, with the
  prefactor `0` in place of a tolerance.
* `crossCovEstimator_attenuated_once`, `crossCovEstimator_attenuated_twice` — the attenuation
  count, as exact identities about the estimator of `eq:cross-cov`.
* `trace_crossCovEstimator_attenuated_once`, `trace_crossCovEstimator_attenuated_twice` — the
  same counts at the level of the traces `σ̂²_Jac(s) = s·σ̂²_Jac(1)` and `s²·σ²_obj`.
* `popCrossCovTrace` — the population object `tr E[δθ̃ δg^⊤]` that `eq:cross-cov` defines
  `σ²_Jac` to be, as a Bochner integral.
* `popCrossCovTrace_attenuated_once`, `popCrossCovTrace_attenuated_twice` — the two counts for
  that population object, with no integrability hypothesis.
* `popCrossCovTrace_self_nonneg` — the population objective channel is non-negative.
* `trace_crossCovEstimator` — the estimator's trace as an explicit sum over the window.
* `trace_crossCovEstimator_self_pos` — the objective channel is strictly positive as soon as the
  window is nonempty and the gradient fluctuation is nonzero somewhere on it.
* `attenuationRatio_tendsto_atTop` — `ϱ → ∞` as `s → 0⁺`, from leading-order scalings carried
  with explicit remainder bounds.
* `attenuationRatio_tendsto_atTop_with_remainders` — a witness with `C_J, C_O > 0`, so the
  remainder-carrying form of that theorem is not vacuously about the exact-scaling case.
* `attenuation_asymmetry` — both channels vanish, and yet their ratio diverges.
* `crossCovRatio_tendsto_atTop` — the same divergence with both constants read off the estimator
  rather than posited.
* `popCrossCovRatio_tendsto_atTop` — and with both constants read off the population objects,
  which is what Section 4's sentence is about.
* `diagAttenuate`, `trace_diagAttenuate_coupling_ge`, `trace_diagAttenuate_obj_le`,
  `trace_diagAttenuate_obj_ge` — the coordinatewise prefactor that (A2) idealizes to a scalar.
* `couplingAbs`, `couplingAbs_nonneg`, `trace_crossCovEstimator_le_couplingAbs` — the `ℓ¹` mass
  of the coordinatewise coupling, which bounds the trace and measures its cancellation.
* `trace_diagAttenuate_coupling_spread_ge` — the coupling trace under a prefactor within `r` of a
  constant, with no sign hypothesis.
* `crossCovRatio_diag_tendsto_atTop` — the divergence survives a genuine diagonal prefactor band,
  under a coordinatewise sign condition on the coupling.
* `crossCovRatio_diag_tendsto_atTop_of_small_spread` — and survives with no sign condition at all
  once (A2) is read quantitatively, as a bound on the *relative spread* of the prefactor against
  the coupling's `ℓ¹` mass.
* `crossCovRatio_diag_small_spread_rescues_sign_flip` — that reading run on the counterexample's
  own data, where a relative spread of `1/4` restores the divergence.
* `crossCovRatio_diag_tendsto_atTop_nonvacuous` — a witness that its hypotheses, the sign
  condition included, are simultaneously satisfiable.
* `crossCovRatio_diag_sign_flip` — and fails without one: a two-coordinate counterexample whose
  prefactor spread is bounded by a factor of ten and whose ratio tends to `-∞`.
* `not_tendsto_atTop_of_neg_on_Ioi` — the elementary fact that counterexample rests on.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses this file carries, and what each would take to discharge:

1. **The single-prefactor idealization (A2), and its price.** `crossCovEstimator_attenuated_once`
   and `crossCovEstimator_attenuated_twice` are stated for a gradient fluctuation of the exact form
   `fun n => s • dgRaw n`. That is (A2) — `ψ'(θ̃) ≈ σ_s I_d` across coordinates on the operative
   band — written as an equality rather than an approximation. It is a hypothesis here because the
   paper itself states (A2) as an idealization. What Section 7 adds is a measurement of what the
   idealization costs, and the answer is not that the constants change. Replacing the scalar by the
   diagonal `diag(ψ'(θ̃))` and asking only for two-sided bounds `c₁σ_s ≤ ψ'(θ̃_i) ≤ c₂σ_s` with
   the spread `c₂/c₁` bounded does **not** preserve the conclusion:
   `crossCovRatio_diag_sign_flip` is a two-coordinate counterexample, with spread ten, on which the
   once-attenuated coupling trace has the opposite sign to the unattenuated one and the ratio tends
   to `-∞` rather than `+∞`. Two hypotheses restore it, and the file proves both, because the
   choice between them is what a referee needs to see. `crossCovRatio_diag_tendsto_atTop` adds the
   hypothesis that the coupling `δθ̃_i δ(∇_θL)_i` is coordinatewise non-negative on the window;
   that is a genuine assumption about the data, not a consequence of `σ²_Jac > 0`, and the paper
   does not make it. `crossCovRatio_diag_tendsto_atTop_of_small_spread` instead makes (A2)'s own
   words quantitative — `|ψ'(θ̃_i) - σ_s| ≤ ε σ_s`, the derivative *approximately constant* rather
   than merely banded — and needs no sign condition, at the price of the explicit threshold
   `ε · couplingAbs T t δθ̃ δ(∇_θL) < σ̂²_Jac` relating the admissible spread to the cancellation
   in the coupling. The second is the reading the paper is entitled to, and
   `crossCovRatio_diag_small_spread_rescues_sign_flip` shows it defeating the counterexample on the
   counterexample's own data. Discharging (A2) honestly therefore means bounding the prefactor's
   relative spread on the operative band against that threshold — a quantitative claim about how
   narrow the band is, which this file does not attempt, since it is a statement about the
   trajectory rather than about the positivity map.
2. **The leading-order scalings, with remainders.** `attenuationRatio_tendsto_atTop` does *not*
   assume `σ²_Jac = sB` and `σ²_obj = A` on the nose. It assumes only
   `|σ²_Jac(s) - sB| ≤ C_J s²` and `|σ²_obj(s) - A| ≤ C_O s` on a window `(0, s₀)`, that is, the
   leading behaviour together with an explicit remainder one order smaller, and the remainder
   constants `C_J, C_O` appear in the proof (they set the width of the window on which the lower
   bound `ϱ(s) ≥ B/(3As)` holds). Setting `C_J = C_O = 0` recovers the exact scaling; no step of
   the argument assumes the remainder vanishes.
3. **Positivity of the two constants.** `0 < A` and `0 < B` are hypotheses. `A > 0` is
   non-degeneracy of the objective gradient noise; `B > 0` is exactly the nonzero cross-covariance
   reading that Theorem 1 of the paper characterizes, and `CrossCov.joint_necessity_finite` says
   what it takes structurally. Neither can be proved here: `B = 0` is a real possibility (it is
   what every architecture missing one of the three ingredients gives), and the paper's claim is
   conditional on it being nonzero.

What this file does **not** prove, and why. The identity
`σ²_eff = σ_s² σ²_obj + σ²_Jac` — the diffusion reading of stochastic gradient descent that turns
these two channels into an effective noise level — is not formalized here, because mathlib
v4.31.0 has no Itô integral, no stochastic differential equations and no diffusion generators. It
does construct Brownian motion, in `Mathlib/Probability/BrownianMotion`, and it has martingales,
filtrations and stopping times in `Mathlib/Probability/Process`; what is absent is stochastic
calculus on top of them. The identity is therefore not proved anywhere in this development, and
any module that needs it has to carry it as a named hypothesis. This file proves only the algebra
and the analysis on either side of it: the attenuation counts that feed it, and the divergence of
the ratio that comes out of it. Nor is the probabilistic *identification* of the two channels
proved: `popCrossCovTrace μ δθ̃ δg` is the paper's `σ²_Jac` only once `δθ̃` and `δg` are the
resampling fluctuations of `eq:fluct` under the law `μ`, and `popCrossCovTrace μ δg δg` is
`σ²_obj` only for a centred fluctuation, and the passage from the matrix-valued
`tr E[δθ̃ δg^⊤]` to the integrated inner product that `popCrossCovTrace` takes as its definition
is an interchange of trace and integral that is assumed rather than proved. Those are modelling
conventions, recorded in the declarations that use them, not results of this file. Of the numbers displayed in Section 2, the
one that is arithmetic
on the positivity map alone is proved — `logistic_neg_five_lt_one_and_a_half_percent`, the `1.5%` at
`w̃ = -5`. The three-decimal ones, `log(0.05/0.95) ≈ -2.944` and `ψ(-2.944) ≈ 0.051`, are not:
bounding `Real.log` to three places calls for interval arithmetic that would add no mathematical
content, and the symbolic identities behind them — the shoulder edge `log(s/(1-s))` and its image
`-log(1-s)` under the positivity map — are proved instead. The occupancy table is a measurement on a
training run and is not a mathematical claim.
-/

open Filter Topology Set

namespace IcnnLift

/-! ## 1. The positivity map and its derivative -/

/-- The softplus positivity map `ψ(w) = log(1 + e^w)`, the standard differentiable
alternative to projection in the paper's Section 2. -/
noncomputable def softplus (w : ℝ) : ℝ := Real.log (1 + Real.exp w)

/-- The logistic function `σ(w) = e^w/(1 + e^w)`. Section 2 records that it is the softplus
positivity map's derivative, hence the chain-rule prefactor of `eq:chain`. -/
noncomputable def logistic (w : ℝ) : ℝ := Real.exp w / (1 + Real.exp w)

/-- Defining equation of the softplus positivity map. -/
theorem softplus_def (w : ℝ) : softplus w = Real.log (1 + Real.exp w) := rfl

/-- Defining equation of the logistic prefactor. -/
theorem logistic_def (w : ℝ) : logistic w = Real.exp w / (1 + Real.exp w) := rfl

/-- The softplus denominator is positive, so the positivity map is well defined and its derivative
never divides by zero. -/
theorem one_add_exp_pos (w : ℝ) : (0 : ℝ) < 1 + Real.exp w := by
  have := Real.exp_pos w
  linarith

/-- The positivity map takes values in the strictly positive reals: this is the sense in which
`ψ : ℝ → ℝ_{≥0}` enforces the input-convexity constraint at every iterate. -/
theorem softplus_pos (w : ℝ) : 0 < softplus w := by
  have h1 : (1 : ℝ) < 1 + Real.exp w := by
    have := Real.exp_pos w
    linarith
  have := Real.log_lt_log (by norm_num) h1
  simpa [softplus_def, Real.log_one] using this

/-! ## 2. The chain rule through the positivity map (`eq:chain`) -/

/-- **The chain rule of `eq:chain`.** With `θ = ψ(θ̃)`, the gradient in the unconstrained
coordinate is the constrained gradient multiplied by the positivity-map derivative. Stated as a genuine
differentiability statement: if the loss `L`, read on the constrained coordinate, has derivative
`g` at `ψ w`, and the positivity map has derivative `p` at `w`, then the pullback `u ↦ L (ψ u)` has
derivative `g * p` at `w`. -/
theorem hasDerivAt_positivityMap_comp {L psi : ℝ → ℝ} {w g p : ℝ}
    (hpsi : HasDerivAt psi p w) (hL : HasDerivAt L g (psi w)) :
    HasDerivAt (fun u : ℝ => L (psi u)) (g * p) w :=
  HasDerivAt.comp w hL hpsi

/-- The same statement read off the `deriv` operator, which is the form `eq:chain` is written
in. -/
theorem deriv_positivityMap_comp {L psi : ℝ → ℝ} {w : ℝ}
    (hpsi : DifferentiableAt ℝ psi w) (hL : DifferentiableAt ℝ L (psi w)) :
    deriv (fun u : ℝ => L (psi u)) w = deriv L (psi w) * deriv psi w :=
  (hasDerivAt_positivityMap_comp hpsi.hasDerivAt hL.hasDerivAt).deriv

/-- **The coordinatewise form of `eq:chain`**, which reads the Hadamard product
`∇_{θ̃}L = ψ'(θ̃) ⊙ ∇_θ L` one coordinate at a time: perturbing the `i`-th free coordinate and
reading the loss through the coordinatewise positivity map gives the `i`-th constrained partial
derivative multiplied by `ψ'(w i)`. The positivity map is applied coordinatewise, so no coordinate
mixes with any other. The joint statement, with all coordinates perturbed at once — which is
what the displayed identity of `eq:chain` presupposes — is `hasFDerivAt_positivityMap_pi` below. -/
theorem hasDerivAt_positivityMap_coordinate {d : ℕ} {L : (Fin d → ℝ) → ℝ} {psi : ℝ → ℝ}
    (w : Fin d → ℝ) (i : Fin d) {p g : ℝ} (hpsi : HasDerivAt psi p (w i))
    (hL : HasDerivAt (fun u : ℝ => L (Function.update (fun j => psi (w j)) i u)) g (psi (w i))) :
    HasDerivAt (fun u : ℝ => L (fun j => psi (Function.update w i u j))) (g * p) (w i) := by
  have hFG : (fun u : ℝ => L (fun j => psi (Function.update w i u j)))
      = fun u : ℝ => L (Function.update (fun j => psi (w j)) i (psi u)) := by
    funext u
    congr 1
    funext j
    rcases eq_or_ne j i with rfl | hj
    · simp
    · simp [Function.update_of_ne hj]
  rw [hFG]
  exact hasDerivAt_positivityMap_comp hpsi hL

/-- The Hadamard multiplier `v ↦ a ⊙ v` as a continuous linear map on `ℝ^d`. With
`a = ψ'(θ̃)` it is the diagonal Jacobian of the coordinatewise positivity map, the `diag(ψ'(θ̃))` that
(A2) later idealizes to a scalar. -/
noncomputable def hadamardCLM {d : ℕ} (a : Fin d → ℝ) : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ) :=
  ContinuousLinearMap.pi fun i => a i • ContinuousLinearMap.proj i

/-- The Hadamard multiplier acts entrywise. -/
theorem hadamardCLM_apply {d : ℕ} (a v : Fin d → ℝ) (i : Fin d) :
    hadamardCLM a v i = a i * v i := rfl

/-- **`eq:chain` with all `d` coordinates perturbed at once.** The coordinatewise positivity map
`Ψ(θ̃) = (ψ(θ̃_1), …, ψ(θ̃_d))` pulled through a loss that is jointly differentiable at the
constrained point is jointly differentiable, with differential the constrained differential `G`
precomposed with the diagonal map `v ↦ ψ'(θ̃) ⊙ v`. This is the statement the displayed identity
`∇_{θ̃}L = ψ'(θ̃) ⊙ ∇_θ L` of `eq:chain` presupposes — a gradient identity asserts joint
differentiability of both sides, which the coordinatewise slices above do not by themselves
provide. `fderiv_positivityMap_pi_single` reads the Hadamard entries off it. -/
theorem hasFDerivAt_positivityMap_pi {d : ℕ} {L : (Fin d → ℝ) → ℝ} {psi p : ℝ → ℝ}
    {G : (Fin d → ℝ) →L[ℝ] ℝ} (w : Fin d → ℝ)
    (hpsi : ∀ i, HasDerivAt psi (p (w i)) (w i))
    (hL : HasFDerivAt L G (fun j => psi (w j))) :
    HasFDerivAt (fun u : Fin d → ℝ => L (fun j => psi (u j)))
      (G.comp (hadamardCLM fun i => p (w i))) w := by
  have hΨ : HasFDerivAt (fun u : Fin d → ℝ => fun j => psi (u j))
      (hadamardCLM fun i => p (w i)) w := by
    unfold hadamardCLM
    refine hasFDerivAt_pi.mpr fun i => ?_
    exact (hpsi i).comp_hasFDerivAt w (hasFDerivAt_apply i w)
  exact hL.comp w hΨ

/-- **The Hadamard entries of `eq:chain`, read off the joint statement**: the `i`-th partial
derivative of the pulled-back loss is `ψ'(θ̃_i)` times the `i`-th constrained partial
derivative, `(∇_{θ̃}L)_i = ψ'(θ̃_i) · (∇_θ L)_i`. Here `G (Pi.single i 1)` is the `i`-th entry
of the constrained gradient and the left-hand side is the `i`-th entry of the free one. -/
theorem fderiv_positivityMap_pi_single {d : ℕ} {L : (Fin d → ℝ) → ℝ} {psi p : ℝ → ℝ}
    {G : (Fin d → ℝ) →L[ℝ] ℝ} (w : Fin d → ℝ)
    (hpsi : ∀ i, HasDerivAt psi (p (w i)) (w i))
    (hL : HasFDerivAt L G (fun j => psi (w j))) (i : Fin d) :
    fderiv ℝ (fun u : Fin d → ℝ => L (fun j => psi (u j))) w (Pi.single i (1 : ℝ))
      = p (w i) * G (Pi.single i (1 : ℝ)) := by
  rw [(hasFDerivAt_positivityMap_pi w hpsi hL).fderiv]
  have hval : hadamardCLM (fun j => p (w j)) (Pi.single i (1 : ℝ))
      = p (w i) • (Pi.single i (1 : ℝ) : Fin d → ℝ) := by
    funext j
    rcases eq_or_ne j i with rfl | hj
    · simp [hadamardCLM_apply]
    · simp [hadamardCLM_apply, Pi.single_eq_of_ne hj]
  rw [ContinuousLinearMap.comp_apply, hval, map_smul, smul_eq_mul]

/-! ## 3. The softplus prefactor and the shoulder (`eq:shoulder`) -/

/-- **`ψ' = logistic`** for the softplus positivity map, as a derivative statement. -/
theorem hasDerivAt_softplus (w : ℝ) : HasDerivAt softplus (logistic w) w := by
  have h : HasDerivAt (fun u : ℝ => Real.log (1 + Real.exp u))
      (Real.exp w / (1 + Real.exp w)) w :=
    HasDerivAt.log (HasDerivAt.const_add 1 (Real.hasDerivAt_exp w)) (one_add_exp_pos w).ne'
  exact h

/-- The identification of Section 2, "for the softplus `ψ'` is the logistic function". -/
theorem deriv_softplus_eq_logistic : deriv softplus = logistic :=
  funext fun w => (hasDerivAt_softplus w).deriv

/-- The chain-rule prefactor is **strictly positive everywhere**: the softplus positivity map attenuates
the gradient but never annihilates it, in contrast with the projection limit. -/
theorem logistic_pos (w : ℝ) : 0 < logistic w :=
  div_pos (Real.exp_pos w) (one_add_exp_pos w)

/-- The prefactor never exceeds one, so the positivity map can only attenuate a gradient. -/
theorem logistic_lt_one (w : ℝ) : logistic w < 1 := by
  rw [logistic_def, div_lt_one (one_add_exp_pos w)]
  linarith [Real.exp_pos w]

/-- The prefactor is **strictly increasing**, so the attenuation deepens monotonically as the
free parameter moves down the half-line. -/
theorem logistic_strictMono : StrictMono logistic := by
  intro a b hab
  have ha : (0 : ℝ) < Real.exp a := Real.exp_pos a
  have hb : (0 : ℝ) < Real.exp b := Real.exp_pos b
  have hda : (0 : ℝ) < 1 + Real.exp a := by linarith
  have hdb : (0 : ℝ) < 1 + Real.exp b := by linarith
  have hlt : Real.exp a < Real.exp b := Real.exp_lt_exp.mpr hab
  rw [logistic_def, logistic_def, div_lt_div_iff₀ hda hdb]
  nlinarith

/-- The positivity map itself is non-decreasing, which is the monotonicity `rem:positivity-generality`
asks of a positivity reparametrization. -/
theorem softplus_monotone : Monotone softplus := by
  refine monotone_of_deriv_nonneg (fun w => (hasDerivAt_softplus w).differentiableAt) fun w => ?_
  rw [(hasDerivAt_softplus w).deriv]
  exact (logistic_pos w).le

/-- **The attenuation prefactor can be made arbitrarily small**: `ψ'(w̃) → 0` as `w̃ → -∞`. This
is what "the shoulder" names. The statement is qualitative; Section 2's displayed numeric, that a
free parameter at `-5` passes under `1.5%` of the gradient it would pass at zero, is the separate
`logistic_neg_five_lt_one_and_a_half_percent` below. -/
theorem tendsto_logistic_atBot : Tendsto logistic atBot (𝓝 0) := by
  have hnum : Tendsto Real.exp atBot (𝓝 0) := Real.tendsto_exp_atBot
  have hden : Tendsto (fun w : ℝ => 1 + Real.exp w) atBot (𝓝 (1 + 0)) :=
    tendsto_const_nhds.add Real.tendsto_exp_atBot
  have h := hnum.div hden (by norm_num)
  have hz : (0 : ℝ) / (1 + 0) = 0 := by norm_num
  rw [hz] at h
  exact h.congr fun w => rfl

/-- **The paper's asymptotic for the prefactor**, `ψ'(w̃) ≈ e^{w̃}` as `w̃ → -∞`, stated exactly:
the ratio of the logistic to the exponential tends to one at `-∞`. This is the sense in which
"the attenuation grows without bound the further the iterate moves into" the shoulder — the
prefactor decays exponentially in the free parameter's own value. -/
theorem tendsto_logistic_div_exp_atBot :
    Tendsto (fun w : ℝ => logistic w / Real.exp w) atBot (𝓝 1) := by
  have hratio : ∀ w : ℝ, logistic w / Real.exp w = (1 + Real.exp w)⁻¹ := by
    intro w
    have h1 : Real.exp w ≠ 0 := Real.exp_ne_zero w
    have h2 : (1 : ℝ) + Real.exp w ≠ 0 := (one_add_exp_pos w).ne'
    rw [logistic_def]
    field_simp
  have hlim : Tendsto (fun w : ℝ => (1 + Real.exp w)⁻¹) atBot (𝓝 ((1 + 0 : ℝ))⁻¹) :=
    (tendsto_const_nhds.add Real.tendsto_exp_atBot).inv₀ (by norm_num)
  rw [show ((1 : ℝ) + 0)⁻¹ = 1 by norm_num] at hlim
  exact hlim.congr fun w => (hratio w).symm

/-- **The prefactor at the origin**, `ψ'(0) = 1/2`: the reference value against which Section 2
measures the collapse on the shoulder. -/
theorem logistic_zero : logistic 0 = 1 / 2 := by
  rw [logistic_def, Real.exp_zero]
  norm_num

/-- **Section 2's displayed numeric, machine-checked**: a free parameter at `-5` passes under
`1.5%` of the gradient it would pass at zero, that is `ψ'(-5) < 0.015 · ψ'(0)`. The proof needs
only `e > 2.7`, since `ψ'(-5) < e^{-5}` and `e^5 > 2.7^5 > 134 > 1/0.0075`. This is the one
displayed number of Section 2 that is arithmetic on the positivity map rather than on measured data. -/
theorem logistic_neg_five_lt_one_and_a_half_percent :
    logistic (-5) < 0.015 * logistic 0 := by
  have he1 : (2.7 : ℝ) < Real.exp 1 := lt_trans (by norm_num) Real.exp_one_gt_d9
  have he5 : (134 : ℝ) < Real.exp 5 := by
    have h : (2.7 : ℝ) ^ (5 : ℕ) < Real.exp 1 ^ (5 : ℕ) := by gcongr
    have h2 : Real.exp 1 ^ (5 : ℕ) = Real.exp 5 := by
      rw [Real.exp_one_pow]
      norm_num
    rw [h2] at h
    nlinarith [h]
  have hmul : Real.exp (-5) * Real.exp 5 = 1 := by
    rw [← Real.exp_add]
    norm_num
  have hexp : Real.exp (-5) < 0.0075 := by
    nlinarith [Real.exp_pos (-5), he5, hmul]
  have hlt : logistic (-5) < Real.exp (-5) := by
    rw [logistic_def, div_lt_iff₀ (one_add_exp_pos (-5))]
    nlinarith [Real.exp_pos (-5)]
  rw [logistic_zero]
  linarith

/-- **The attenuation grows without bound**, which is Section 2's own phrasing for what happens
"the further the iterate moves into" the shoulder: the reciprocal of the chain-rule prefactor
diverges at `-∞`. -/
theorem tendsto_inv_logistic_atBot : Tendsto (fun w : ℝ => (logistic w)⁻¹) atBot atTop := by
  have h : Tendsto logistic atBot (𝓝[>] (0 : ℝ)) :=
    tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ tendsto_logistic_atBot
      (Filter.Eventually.of_forall fun w => logistic_pos w)
  exact h.inv_tendsto_nhdsGT_zero

/-- The sublevel set of a general chain-rule prefactor `p` at tolerance `s`. -/
def shoulderOf (p : ℝ → ℝ) (s : ℝ) : Set ℝ := {w | p w < s}

/-- **The shoulder** `S_{σ_s}` of `eq:shoulder`: the set of free coordinates on which the
softplus chain-rule prefactor has collapsed below the tolerance `σ_s`. -/
def shoulder (s : ℝ) : Set ℝ := shoulderOf logistic s

/-- The shoulder is literally the sublevel set of the positivity map's *derivative*, which is how
`eq:shoulder` reads. The definition above is phrased through `logistic` for convenience; that this
is the same set is a consequence of `deriv_softplus_eq_logistic` and not a modelling choice, so
nothing about the identification `ψ' = logistic` is smuggled into the definition. -/
theorem shoulder_eq_deriv_sublevel (s : ℝ) :
    shoulder s = {w : ℝ | deriv softplus w < s} := by
  rw [deriv_softplus_eq_logistic]
  rfl

/-- The shoulder membership test, solved for the free coordinate. -/
theorem logistic_lt_iff {s w : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    logistic w < s ↔ w < Real.log (s / (1 - s)) := by
  have he : (0 : ℝ) < Real.exp w := Real.exp_pos w
  have hden : (0 : ℝ) < 1 + Real.exp w := by linarith
  have h1s : (0 : ℝ) < 1 - s := by linarith
  have hq : (0 : ℝ) < s / (1 - s) := div_pos hs0 h1s
  rw [Real.lt_log_iff_exp_lt hq, logistic_def, div_lt_iff₀ hden, lt_div_iff₀ h1s]
  constructor <;> intro h <;> nlinarith

/-- The `≤` form of the same test. -/
theorem logistic_le_iff {s w : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    logistic w ≤ s ↔ w ≤ Real.log (s / (1 - s)) := by
  have he : (0 : ℝ) < Real.exp w := Real.exp_pos w
  have hden : (0 : ℝ) < 1 + Real.exp w := by linarith
  have h1s : (0 : ℝ) < 1 - s := by linarith
  have hq : (0 : ℝ) < s / (1 - s) := div_pos hs0 h1s
  rw [Real.le_log_iff_exp_le hq, logistic_def, div_le_iff₀ hden, le_div_iff₀ h1s]
  constructor <;> intro h <;> nlinarith

/-- **`eq:shoulder`, exactly as displayed in the paper**: for a tolerance `0 < s < 1` the
shoulder is the left-infinite interval `(-∞, log(s/(1-s)))`. The right endpoint is given in
closed form, which is what makes the shoulder an extended region rather than a thin set: at
`s = 0.05` it is the half-line below `log(1/19)`. -/
theorem shoulder_eq_Iio {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    shoulder s = Set.Iio (Real.log (s / (1 - s))) := by
  ext w
  simpa [shoulder, shoulderOf] using logistic_lt_iff (w := w) hs0 hs1

/-- The closed sublevel set `{w̃ : ψ'(w̃) ≤ s}` is the corresponding closed half-line. -/
theorem shoulder_closed_eq_Iic {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    {w : ℝ | logistic w ≤ s} = Set.Iic (Real.log (s / (1 - s))) := by
  ext w
  simpa using logistic_le_iff (w := w) hs0 hs1

/-- **The shoulder edge read in the constrained coordinate.** Applying the positivity map to the right
endpoint of `eq:shoulder` gives `-log(1-s)`; at `s = 0.05` this is the paper's `ψ(-2.944) ≈
0.051`, so every constrained weight below it lies in the shoulder. -/
theorem softplus_shoulder_edge {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    softplus (Real.log (s / (1 - s))) = -Real.log (1 - s) := by
  have h1s : (0 : ℝ) < 1 - s := by linarith
  have hq : (0 : ℝ) < s / (1 - s) := div_pos hs0 h1s
  have hexp : Real.exp (Real.log (s / (1 - s))) = s / (1 - s) := Real.exp_log hq
  have harg : 1 + s / (1 - s) = (1 - s)⁻¹ := by
    field_simp
    ring
  rw [softplus_def, hexp, harg, Real.log_inv]

/-- **Section 2's reading of the edge**: "every constrained weight below `ψ(-2.944) ≈ 0.051`
lies in it". A constrained weight strictly below `-log(1-s)`, the positivity map's value at the
shoulder edge, comes from a free coordinate inside the shoulder; monotonicity of the positivity map
carries the constrained-coordinate test back to the free coordinate. -/
theorem mem_shoulder_of_softplus_lt {s w : ℝ} (hs0 : 0 < s) (hs1 : s < 1)
    (h : softplus w < -Real.log (1 - s)) : w ∈ shoulder s := by
  rw [shoulder_eq_Iio hs0 hs1, Set.mem_Iio]
  by_contra hcon
  have hmono := softplus_monotone (not_lt.mp hcon)
  rw [softplus_shoulder_edge hs0 hs1] at hmono
  linarith

/-- A tolerance at or below zero cuts out nothing: the softplus prefactor is strictly positive,
so no free coordinate is ever fully gated. -/
theorem shoulder_eq_empty_of_nonpos {s : ℝ} (hs : s ≤ 0) : shoulder s = ∅ := by
  ext w
  simp only [shoulder, shoulderOf, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_lt]
  exact hs.trans (logistic_pos w).le

/-- A tolerance at or above one cuts out everything, since the prefactor never reaches one. -/
theorem shoulder_eq_univ_of_one_le {s : ℝ} (hs : 1 ≤ s) : shoulder s = Set.univ := by
  ext w
  simp only [shoulder, shoulderOf, Set.mem_setOf_eq, Set.mem_univ, iff_true]
  exact (logistic_lt_one w).trans_le hs

/-! ## 4. Generality: `rem:positivity-generality` as a theorem -/

/-- **The attenuation bound.** Where the chain-rule prefactor is below the tolerance `s`, the
gradient in the free coordinate is at most `s` times the gradient in the constrained coordinate.
This is what "the gradient that would bring them back out is exponentially small in their own
value" asserts, stated as an inequality rather than an asymptotic. -/
theorem chainRule_attenuated {psi p L : ℝ → ℝ} {w g s : ℝ}
    (hpsi : HasDerivAt psi (p w) w) (hL : HasDerivAt L g (psi w))
    (hnn : 0 ≤ p w) (hle : p w ≤ s) :
    HasDerivAt (fun u : ℝ => L (psi u)) (g * p w) w ∧ |g * p w| ≤ s * |g| := by
  refine ⟨hasDerivAt_positivityMap_comp hpsi hL, ?_⟩
  have habs : |g * p w| = |g| * p w := by
    rw [abs_mul, abs_of_nonneg hnn]
  rw [habs]
  nlinarith [abs_nonneg g]

/-- A prefactor that collapses at `-∞` has, for every tolerance, a whole left ray inside its
sublevel set. The shoulder is therefore "an extended region of the free-parameter space, not a
thin set an iterate can be expected to miss". -/
theorem shoulderOf_contains_ray {p : ℝ → ℝ} (hflat : Tendsto p atBot (𝓝 0)) {s : ℝ} (hs : 0 < s) :
    ∃ a : ℝ, Set.Iio a ⊆ shoulderOf p s := by
  obtain ⟨a, ha⟩ := eventually_atBot.mp (Filter.Tendsto.eventually_lt_const hs hflat)
  exact ⟨a, fun w hw => ha w (le_of_lt hw)⟩

/-- **`rem:positivity-generality`, proved.** The remark asserts that the shoulder phenomenon is
not special to softplus, and then discharges the assertion by observing that softplus and "any
other smooth monotone-positive reparametrization" satisfy its two conditions. Here it is a theorem
about an arbitrary positivity map: let `ψ` be differentiable with derivative `p`, let `p` be non-negative
(so `ψ` is non-decreasing), and let `p` tend to zero at `-∞`. Then `ψ` is monotone, and for every
tolerance `s > 0` there is a left ray `(-∞, a)` on which the chain-rule prefactor is below `s` and
on which every constrained gradient `g` is attenuated to a free-coordinate gradient of size at
most `s|g|`.

Nothing beyond those three hypotheses is used: no formula for `ψ`, no convexity, no rate of decay,
and — unlike the remark, which asks for `ψ : ℝ → ℝ_{≥0}` — no non-negativity of the positivity map
itself. The remark's other clause, the non-differentiable projection limit, is out of reach of
this statement by construction and is treated in `projection_limit_gradient_zero`. -/
theorem shoulder_general {psi p : ℝ → ℝ}
    (hderiv : ∀ w, HasDerivAt psi (p w) w) (hnonneg : ∀ w, 0 ≤ p w)
    (hflat : Tendsto p atBot (𝓝 0)) :
    Monotone psi ∧
      ∀ s : ℝ, 0 < s → ∃ a : ℝ, ∀ w < a,
        p w < s ∧ ∀ (L : ℝ → ℝ) (g : ℝ), HasDerivAt L g (psi w) →
          HasDerivAt (fun u : ℝ => L (psi u)) (g * p w) w ∧ |g * p w| ≤ s * |g| := by
  refine ⟨monotone_of_hasDerivAt_nonneg hderiv fun w => hnonneg w, fun s hs => ?_⟩
  obtain ⟨a, ha⟩ := shoulderOf_contains_ray hflat hs
  refine ⟨a, fun w hw => ?_⟩
  have hpw : p w < s := ha hw
  exact ⟨hpw, fun L g hL => chainRule_attenuated (hderiv w) hL (hnonneg w) hpw.le⟩

/-- The softplus positivity map is an instance of `shoulder_general`. -/
theorem shoulder_softplus :
    Monotone softplus ∧
      ∀ s : ℝ, 0 < s → ∃ a : ℝ, ∀ w < a,
        logistic w < s ∧ ∀ (L : ℝ → ℝ) (g : ℝ), HasDerivAt L g (softplus w) →
          HasDerivAt (fun u : ℝ => L (softplus u)) (g * logistic w) w ∧
            |g * logistic w| ≤ s * |g| :=
  shoulder_general hasDerivAt_softplus (fun w => (logistic_pos w).le)
    tendsto_logistic_atBot

/-- The exponential positivity map `ψ(u) = e^u`, the other canonical smooth positivity
reparametrization, is a second instance: its prefactor is itself, and it collapses at `-∞` for
the same reason. -/
theorem shoulder_exp :
    Monotone Real.exp ∧
      ∀ s : ℝ, 0 < s → ∃ a : ℝ, ∀ w < a,
        Real.exp w < s ∧ ∀ (L : ℝ → ℝ) (g : ℝ), HasDerivAt L g (Real.exp w) →
          HasDerivAt (fun u : ℝ => L (Real.exp u)) (g * Real.exp w) w ∧
            |g * Real.exp w| ≤ s * |g| :=
  shoulder_general Real.hasDerivAt_exp (fun w => (Real.exp_pos w).le)
    Real.tendsto_exp_atBot

/-- Below the origin the projection positivity map `ψ(u) = max(u, 0)` is locally constant, so it is
differentiable there with derivative zero. The positivity map is not differentiable at the origin, which
is why the projection limit falls outside `shoulder_general` and has to be treated
separately. -/
theorem hasDerivAt_projection_of_neg {w : ℝ} (hw : w < 0) :
    HasDerivAt (fun u : ℝ => max u 0) 0 w := by
  have hev : (fun u : ℝ => max u 0) =ᶠ[𝓝 w] fun _ : ℝ => (0 : ℝ) := by
    filter_upwards [gt_mem_nhds hw] with u hu
    exact max_eq_right hu.le
  exact (hasDerivAt_const w (0 : ℝ)).congr_of_eventuallyEq hev

/-- **The projection limit of `rem:positivity-generality`**, which the remark states and the
formalization above deliberately excludes: at the non-differentiable endpoint of the family the
attenuation is not small, it is exact. Below the origin the pullback of **any** loss through the
projection positivity map has derivative exactly `0`, so "the gated coordinates have exactly zero
gradient" and no tolerance `s > 0` is involved. This is the contrast that makes `logistic_pos`
worth stating: softplus attenuates without ever annihilating.

The loss carries no hypothesis at all here, and that is the point rather than an oversight. Below
the origin the projection positivity map is locally constant, so the pullback is locally constant too and
its derivative is zero whether or not the loss is differentiable anywhere. Routing the statement
through `hasDerivAt_positivityMap_comp` instead would force a differentiability hypothesis on `L` at
`max w 0` that the remark's claim does not need. -/
theorem projection_limit_gradient_zero (L : ℝ → ℝ) {w : ℝ} (hw : w < 0) :
    HasDerivAt (fun u : ℝ => L (max u 0)) 0 w := by
  have hev : (fun u : ℝ => L (max u 0)) =ᶠ[𝓝 w] fun _ : ℝ => L 0 := by
    filter_upwards [gt_mem_nhds hw] with u hu
    rw [max_eq_right hu.le]
  exact (hasDerivAt_const w (L 0)).congr_of_eventuallyEq hev

/-- The same fact read through `eq:chain` rather than through local constancy, for a loss that is
differentiable at the gated value: the chain-rule prefactor at the projection positivity map is `0`
below the origin, so the free-coordinate derivative is `g * 0` for every constrained gradient `g`.
This is the projection endpoint of `chainRule_attenuated`, in which the tolerance `s` may be taken
to be `0`. -/
theorem projection_limit_gradient_zero_of_hasDerivAt {L : ℝ → ℝ} {w g : ℝ} (hw : w < 0)
    (hL : HasDerivAt L g (max w 0)) :
    HasDerivAt (fun u : ℝ => L (max u 0)) (g * 0) w :=
  hasDerivAt_positivityMap_comp (hasDerivAt_projection_of_neg hw) hL

/-! ## 5. The attenuation count: once for the coupling channel, twice for the objective -/

section AttenuationCount

variable {d : ℕ}

/-- **The cross-covariance channel is attenuated ONCE.** Under (A2)'s single-prefactor reading
`ψ'(θ̃) = s·I_d`, the free-coordinate gradient fluctuation is `δg = s · δ(∇_θ L)`, while the
iterate fluctuation `δθ̃` of `eq:fluct` passes through no positivity-map derivative at all — it is a
re-evaluation of the latent weight, not a gradient step. The estimator of `eq:cross-cov` is therefore
exactly linear in `s`.

This is the corrected power of `docs/design/lemma1_rederivation.md` §11.2 — the count that
Section 4 of the v4 text states as "carries that derivative once": `σ²_Jac = Θ(s)`. The
superseded pre-split manuscript's word "unattenuated", which would mean `Θ(1)`, was wrong, and
§11.2 records the correction that the v4 wording now carries. -/
theorem crossCovEstimator_attenuated_once (T t : ℕ) (s : ℝ) (dθ dg : ℕ → Fin d → ℝ) :
    crossCovEstimator T t dθ (fun n => s • dg n) = s • crossCovEstimator T t dθ dg := by
  have hstep : ∀ n : ℕ,
      Matrix.vecMulVec (dθ n) (s • dg n) = s • Matrix.vecMulVec (dθ n) (dg n) := by
    intro n
    ext i j
    simp [Matrix.vecMulVec_apply, mul_left_comm]
  simp only [crossCovEstimator, hstep]
  rw [← Finset.smul_sum, smul_comm]

/-- **The objective channel is attenuated TWICE.** The objective term of the effective diffusion
is the variance of an already-attenuated gradient: both slots of the second moment carry the
positivity-map derivative, so the same estimator scales as `s²`. -/
theorem crossCovEstimator_attenuated_twice (T t : ℕ) (s : ℝ) (dg : ℕ → Fin d → ℝ) :
    crossCovEstimator T t (fun n => s • dg n) (fun n => s • dg n)
      = (s ^ 2) • crossCovEstimator T t dg dg := by
  have hstep : ∀ n : ℕ,
      Matrix.vecMulVec (s • dg n) (s • dg n) = (s ^ 2) • Matrix.vecMulVec (dg n) (dg n) := by
    intro n
    ext i j
    simp [Matrix.vecMulVec_apply]
    ring
  simp only [crossCovEstimator, hstep]
  rw [← Finset.smul_sum, smul_comm]

/-- The trace form of the once-attenuated count: `σ̂²_Jac(s) = s · σ̂²_Jac(1)`. -/
theorem trace_crossCovEstimator_attenuated_once (T t : ℕ) (s : ℝ) (dθ dg : ℕ → Fin d → ℝ) :
    (crossCovEstimator T t dθ (fun n => s • dg n)).trace
      = s * (crossCovEstimator T t dθ dg).trace := by
  rw [crossCovEstimator_attenuated_once, Matrix.trace_smul, smul_eq_mul]

/-- The trace form of the twice-attenuated count: the free-coordinate objective variance is
`s² σ²_obj`, with `σ²_obj` read on the constrained coordinate before the positivity-map prefactor. -/
theorem trace_crossCovEstimator_attenuated_twice (T t : ℕ) (s : ℝ) (dg : ℕ → Fin d → ℝ) :
    (crossCovEstimator T t (fun n => s • dg n) (fun n => s • dg n)).trace
      = s ^ 2 * (crossCovEstimator T t dg dg).trace := by
  rw [crossCovEstimator_attenuated_twice, Matrix.trace_smul, smul_eq_mul]

/-- The trace of the estimator written out as a sum over the window: the `1/T` average of the
inner products `⟨δθ̃⁽ˢ⁾, δg⁽ˢ⁾⟩`. Every trace-level statement below is read off this identity. -/
theorem trace_crossCovEstimator (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    (crossCovEstimator T t dθ dg).trace
      = (T : ℝ)⁻¹ * ∑ n ∈ Finset.range T, ∑ i, dθ (t - T + n) i * dg (t - T + n) i := by
  rw [crossCovEstimator, Matrix.trace_smul, smul_eq_mul, Matrix.trace_sum]
  refine congrArg (fun x : ℝ => (T : ℝ)⁻¹ * x) (Finset.sum_congr rfl fun n _ => ?_)
  exact Matrix.trace_vecMulVec _ _

/-- The empirical objective channel is a sum of squares, hence non-negative: the denominator of
the ratio below is never negative. Non-negativity alone is not a non-degeneracy statement, and in
particular it does not say the channel vanishes only on an identically zero gradient fluctuation —
an empty window (`T = 0`) makes the estimator zero whatever the data. The sharp statement is
`trace_crossCovEstimator_self_pos`. -/
theorem trace_crossCovEstimator_self_nonneg (T t : ℕ) (dg : ℕ → Fin d → ℝ) :
    0 ≤ (crossCovEstimator T t dg dg).trace := by
  rw [crossCovEstimator, Matrix.trace_smul, smul_eq_mul, Matrix.trace_sum]
  refine mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg T)) (Finset.sum_nonneg fun n _ => ?_)
  rw [Matrix.trace_vecMulVec]
  simp only [dotProduct]
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg _

/-- **When the objective channel is strictly positive.** A nonempty window on which the
constrained-coordinate gradient fluctuation is nonzero at some step and some coordinate suffices.
This is what makes the positivity hypothesis `hObj` of `crossCovRatio_tendsto_atTop` and
`crossCovRatio_diag_tendsto_atTop` a condition on the data rather than an unsatisfiable one: it is
implied by the mildest possible non-degeneracy of the gradient noise on the window. -/
theorem trace_crossCovEstimator_self_pos {T t : ℕ} {dg : ℕ → Fin d → ℝ} (hT : T ≠ 0)
    (hne : ∃ n ∈ Finset.range T, ∃ i, dg (t - T + n) i ≠ 0) :
    0 < (crossCovEstimator T t dg dg).trace := by
  obtain ⟨n₀, hn₀, i₀, hi₀⟩ := hne
  rw [trace_crossCovEstimator]
  refine mul_pos (inv_pos.mpr (Nat.cast_pos.mpr (Nat.pos_of_ne_zero hT))) ?_
  refine Finset.sum_pos' (fun n _ => Finset.sum_nonneg fun i _ => mul_self_nonneg _) ⟨n₀, hn₀, ?_⟩
  exact Finset.sum_pos' (fun i _ => mul_self_nonneg _)
    ⟨i₀, Finset.mem_univ i₀, mul_self_pos.mpr hi₀⟩

end AttenuationCount

/-! ### 5.1 The same counts for the population objects of `eq:cross-cov`

`eq:cross-cov` defines `σ²_Jac` as the **population** trace `tr E[δθ̃ δg^⊤]` and exhibits
`Σ̂⁽ᵗ⁾` as its running estimator, and Section 4's asymmetry sentence is about the population
quantities. The identities of Section 5 are about the estimator, so on their own they leave the
population claim as an inference from the sample one. They need not: the two counts pass through
the expectation by linearity alone, with no integrability hypothesis, because `∫` commutes with a
real scalar unconditionally in the Bochner theory (a non-integrable integrand yields `0` on both
sides). Both counts are therefore proved here directly for the population objects as well. -/

section PopulationCount

open MeasureTheory

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- The population cross-covariance trace of `eq:cross-cov`, `tr E[δθ̃ δg^⊤]`, taken here to be
the integral against `μ` of the inner product `⟨δθ̃, δg⟩`. The trace of an outer product is that
inner product, so for an integrable pair of fluctuations this is the paper's quantity; the one
step involved, interchanging the finite-dimensional trace with the integral, is not proved here,
because the integrated inner product is taken as the definition rather than derived from a
matrix-valued Bochner integral. No hypothesis is placed on `μ`, which need not even be a
probability measure for the scaling identities below.

Taking `δθ̃ = δg` gives `tr E[δg δg^⊤]`, the population objective channel. That is `σ²_obj` only
when the fluctuation is centred, which is how `eq:fluct` and the surrounding text define it; the
identification of the second moment with the variance is a modelling convention recorded here and
not a theorem of this file. -/
noncomputable def popCrossCovTrace (μ : Measure Ω) (dθ dg : Ω → Fin d → ℝ) : ℝ :=
  ∫ ω, ∑ i, dθ ω i * dg ω i ∂μ

/-- **The population cross-covariance is attenuated ONCE**, the population form of
`crossCovEstimator_attenuated_once`: only `δg` carries the positivity-map derivative, so under (A2)'s
single-prefactor reading `σ²_Jac` is exactly linear in the prefactor. No integrability hypothesis
is needed. -/
theorem popCrossCovTrace_attenuated_once (μ : Measure Ω) (s : ℝ) (dθ dg : Ω → Fin d → ℝ) :
    popCrossCovTrace μ dθ (fun ω i => s * dg ω i) = s * popCrossCovTrace μ dθ dg := by
  have hpt : ∀ ω : Ω, ∑ i, dθ ω i * (s * dg ω i) = s * ∑ i, dθ ω i * dg ω i := by
    intro ω
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [popCrossCovTrace, hpt]
  exact integral_const_mul s _

/-- **The population objective channel is attenuated TWICE**, the population form of
`crossCovEstimator_attenuated_twice`: both slots of the second moment carry the positivity-map
derivative. No integrability hypothesis is needed. -/
theorem popCrossCovTrace_attenuated_twice (μ : Measure Ω) (s : ℝ) (dg : Ω → Fin d → ℝ) :
    popCrossCovTrace μ (fun ω i => s * dg ω i) (fun ω i => s * dg ω i)
      = s ^ 2 * popCrossCovTrace μ dg dg := by
  have hpt : ∀ ω : Ω, ∑ i, (s * dg ω i) * (s * dg ω i) = s ^ 2 * ∑ i, dg ω i * dg ω i := by
    intro ω
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [popCrossCovTrace, hpt]
  exact integral_const_mul (s ^ 2) _

/-- The population objective channel is non-negative, being the integral of a sum of squares. -/
theorem popCrossCovTrace_self_nonneg (μ : Measure Ω) (dg : Ω → Fin d → ℝ) :
    0 ≤ popCrossCovTrace μ dg dg :=
  integral_nonneg fun _ => Finset.sum_nonneg fun _ _ => mul_self_nonneg _

end PopulationCount

/-! ## 6. The ratio diverges as the positivity map flattens -/

/-- The ratio `ϱ = σ²_Jac / (σ_s² σ²_obj)` of the mechanism section: the coupling channel
measured against the twice-attenuated objective channel. -/
noncomputable def attenuationRatio (sigmaJac sigmaObj : ℝ → ℝ) (s : ℝ) : ℝ :=
  sigmaJac s / (s ^ 2 * sigmaObj s)

/-- **The attenuation-count asymmetry, with explicit remainders.** Suppose the coupling channel
is `sB` to leading order in the positivity-map prefactor and the objective channel's constrained-side
variance is `A`, each with an explicit remainder one order smaller:
`|σ²_Jac(s) - sB| ≤ C_J s²` and `|σ²_obj(s) - A| ≤ C_O s` on a window `(0, s₀)`. Then the ratio
`ϱ(s) = σ²_Jac(s)/(s² σ²_obj(s))` diverges as `s → 0⁺`.

Nothing here assumes the remainders vanish: the proof extracts, from the remainder constants
themselves, a window `(0, s₁)` on which `ϱ(s) ≥ B/(3As)`, and it is that explicit lower bound
that diverges. Taking `C_J = C_O = 0` recovers the exact scaling. -/
theorem attenuationRatio_tendsto_atTop
    {sigmaJac sigmaObj : ℝ → ℝ} {A B Cj Co s₀ : ℝ}
    (hA : 0 < A) (hB : 0 < B) (hCj : 0 ≤ Cj) (hCo : 0 ≤ Co) (hs₀ : 0 < s₀)
    (hjac : ∀ s ∈ Set.Ioo (0 : ℝ) s₀, |sigmaJac s - s * B| ≤ Cj * s ^ 2)
    (hobj : ∀ s ∈ Set.Ioo (0 : ℝ) s₀, |sigmaObj s - A| ≤ Co * s) :
    Tendsto (attenuationRatio sigmaJac sigmaObj) (𝓝[>] (0 : ℝ)) atTop := by
  have hCo1 : (0 : ℝ) < Co + 1 := by linarith
  have hCj1 : (0 : ℝ) < Cj + 1 := by linarith
  set s₁ : ℝ := min s₀ (min (A / (2 * (Co + 1))) (B / (2 * (Cj + 1)))) with hs₁def
  have hs₁ : 0 < s₁ := by
    refine lt_min hs₀ (lt_min ?_ ?_)
    · positivity
    · positivity
  have key : ∀ s ∈ Set.Ioo (0 : ℝ) s₁,
      B / (3 * A) * s⁻¹ ≤ attenuationRatio sigmaJac sigmaObj s := by
    intro s hs
    obtain ⟨hs0, hslt⟩ := hs
    have h₀ : s < s₀ := lt_of_lt_of_le hslt (by rw [hs₁def]; exact min_le_left _ _)
    have hCoS : s < A / (2 * (Co + 1)) :=
      lt_of_lt_of_le hslt (by rw [hs₁def]; exact (min_le_right _ _).trans (min_le_left _ _))
    have hCjS : s < B / (2 * (Cj + 1)) :=
      lt_of_lt_of_le hslt (by rw [hs₁def]; exact (min_le_right _ _).trans (min_le_right _ _))
    have h2Co : s * (2 * (Co + 1)) < A := (lt_div_iff₀ (by positivity)).mp hCoS
    have h2Cj : s * (2 * (Cj + 1)) < B := (lt_div_iff₀ (by positivity)).mp hCjS
    have hCoLt : Co * s < A / 2 := by nlinarith
    have hCjLt : Cj * s < B / 2 := by nlinarith
    have hmem : s ∈ Set.Ioo (0 : ℝ) s₀ := ⟨hs0, h₀⟩
    have hoa := abs_le.mp (hobj s hmem)
    have hja := abs_le.mp (hjac s hmem)
    have hobjUB : sigmaObj s < 3 * A / 2 := by linarith [hoa.2]
    have hobjLB : A / 2 < sigmaObj s := by linarith [hoa.1]
    have hobjPos : 0 < sigmaObj s := by linarith
    have hCjSq : Cj * s ^ 2 < B / 2 * s := by nlinarith
    have hjacLB : s * B / 2 < sigmaJac s := by nlinarith [hja.1]
    have hden : 0 < s ^ 2 * sigmaObj s := by positivity
    have h3As : 0 < 3 * A * s := by positivity
    have heq : B / (3 * A) * s⁻¹ = B / (3 * A * s) := by
      rw [← div_eq_mul_inv, div_div]
    simp only [attenuationRatio]
    rw [heq, div_le_div_iff₀ h3As hden]
    have e1 : B * (s ^ 2 * sigmaObj s) < B * s ^ 2 * (3 * A / 2) := by
      nlinarith [mul_pos (mul_pos hB (pow_pos hs0 2)) (sub_pos.mpr hobjUB)]
    have e2 : s * B / 2 * (3 * A * s) < sigmaJac s * (3 * A * s) := by nlinarith
    have e3 : B * s ^ 2 * (3 * A / 2) = s * B / 2 * (3 * A * s) := by ring
    linarith
  have hlim : Tendsto (fun s : ℝ => B / (3 * A) * s⁻¹) (𝓝[>] (0 : ℝ)) atTop :=
    Filter.Tendsto.const_mul_atTop (by positivity) tendsto_inv_nhdsGT_zero
  exact tendsto_atTop_mono' _ (Filter.eventually_of_mem (Ioo_mem_nhdsGT hs₁) key) hlim

/-- **The remainder-carrying hypotheses are satisfiable with nonzero remainders.** The theorem
above is stated with remainder constants `C_J` and `C_O`, but every instance in this file uses it
at `C_J = C_O = 0`, so it is worth exhibiting a witness in which both remainders are genuinely
present: the coupling channel `sB + C_J s²` and the objective variance `A + C_O s` satisfy the
hypotheses exactly, for arbitrary non-negative `C_J` and `C_O`, and the ratio still diverges. The
divergence is therefore not an artifact of the exact-scaling case. -/
theorem attenuationRatio_tendsto_atTop_with_remainders {A B Cj Co : ℝ}
    (hA : 0 < A) (hB : 0 < B) (hCj : 0 ≤ Cj) (hCo : 0 ≤ Co) :
    Tendsto (attenuationRatio (fun s => s * B + Cj * s ^ 2) (fun s => A + Co * s))
      (𝓝[>] (0 : ℝ)) atTop := by
  refine attenuationRatio_tendsto_atTop (A := A) (B := B) (Cj := Cj) (Co := Co) (s₀ := 1)
    hA hB hCj hCo one_pos ?_ ?_
  · intro s _
    have hrw : s * B + Cj * s ^ 2 - s * B = Cj * s ^ 2 := by ring
    rw [hrw, abs_of_nonneg (mul_nonneg hCj (sq_nonneg s))]
  · intro s hs
    have hrw : A + Co * s - A = Co * s := by ring
    rw [hrw, abs_of_nonneg (mul_nonneg hCo hs.1.le)]

/-- **The corrected reading of the asymmetry, in one statement.** With the coupling channel
scaling as `sB` and the objective channel as `s²A`:

* the coupling channel *vanishes* as the positivity map flattens, so it is **not** unattenuated — the
  content of `docs/design/lemma1_rederivation.md` §11.2's correction to the superseded pre-split
  manuscript, which the v4 wording already incorporates;
* the objective channel vanishes too;
* and yet their ratio diverges, because the first is attenuated once where the second is
  attenuated twice.

The third item is the entire downstream requirement; the first is the fact the superseded
manuscript's wording got wrong and the v4 text now states correctly. -/
theorem attenuation_asymmetry {A B : ℝ} (hA : 0 < A) (hB : 0 < B) :
    Tendsto (fun s : ℝ => s * B) (𝓝[>] (0 : ℝ)) (𝓝 0) ∧
      Tendsto (fun s : ℝ => s ^ 2 * A) (𝓝[>] (0 : ℝ)) (𝓝 0) ∧
        Tendsto (attenuationRatio (fun s : ℝ => s * B) (fun _ => A)) (𝓝[>] (0 : ℝ)) atTop := by
  refine ⟨?_, ?_, ?_⟩
  · have h : Tendsto (fun s : ℝ => s * B) (𝓝 (0 : ℝ)) (𝓝 (0 * B)) :=
      tendsto_id.mul tendsto_const_nhds
    rw [zero_mul] at h
    exact h.mono_left nhdsWithin_le_nhds
  · have h : Tendsto (fun s : ℝ => s ^ 2 * A) (𝓝 (0 : ℝ)) (𝓝 (0 ^ 2 * A)) :=
      (tendsto_id.pow 2).mul tendsto_const_nhds
    rw [show (0 : ℝ) ^ 2 * A = 0 by ring] at h
    exact h.mono_left nhdsWithin_le_nhds
  · refine attenuationRatio_tendsto_atTop (A := A) (B := B) (Cj := 0) (Co := 0) (s₀ := 1)
      hA hB le_rfl le_rfl one_pos ?_ ?_ <;> intro s _ <;> simp

/-- **The divergence with both constants read off the estimator.** Let `dθ` be the iterate
fluctuation of `eq:fluct` and `dgRaw` the constrained-coordinate gradient fluctuation, and let
the positivity-map prefactor be the single scalar `s` of (A2). Then the empirical coupling channel over
the free-coordinate gradient `s • dgRaw` divided by the empirical objective channel over the same
attenuated gradient diverges as `s → 0⁺`, provided both readings are nonzero at `s = 1`.

Here neither constant is posited: `B` is the trace of the estimator of `eq:cross-cov` and `A` is
the trace of the objective second moment, both taken on the constrained coordinate. The only
hypotheses are that they are positive — `A > 0` is non-degeneracy of the gradient noise and
`B > 0` is exactly the nonzero cross-covariance reading that `CrossCov.joint_necessity_finite`
characterizes structurally. -/
theorem crossCovRatio_tendsto_atTop {d : ℕ} (T t : ℕ) (dθ dgRaw : ℕ → Fin d → ℝ)
    (hJac : 0 < (crossCovEstimator T t dθ dgRaw).trace)
    (hObj : 0 < (crossCovEstimator T t dgRaw dgRaw).trace) :
    Tendsto (fun s : ℝ => (crossCovEstimator T t dθ (fun n => s • dgRaw n)).trace /
        (crossCovEstimator T t (fun n => s • dgRaw n) (fun n => s • dgRaw n)).trace)
      (𝓝[>] (0 : ℝ)) atTop := by
  have h := attenuation_asymmetry hObj hJac
  refine h.2.2.congr fun s => ?_
  simp only [attenuationRatio]
  rw [trace_crossCovEstimator_attenuated_twice, trace_crossCovEstimator_attenuated_once]

section PopulationRatio

open MeasureTheory

/-- **The divergence for the population objects of `eq:cross-cov`.** Section 4's sentence is
about `σ²_Jac = tr E[δθ̃ δg^⊤]` and `σ²_obj`, not about their running estimators, so the
divergence is recorded here for the population quantities as well: under (A2)'s single-prefactor
reading the free-coordinate gradient fluctuation is `s · δ(∇_θL)`, and the ratio of the
once-attenuated population coupling to the twice-attenuated population objective channel diverges
as the positivity map flattens.

As in `crossCovRatio_tendsto_atTop` the two constants are read off the objects rather than
posited, and the only hypotheses are that they are positive: `σ²_obj > 0` is non-degeneracy of the
gradient noise, and `σ²_Jac > 0` is the nonzero cross-covariance that `thm:joint-necessity`
characterizes structurally. Neither can be proved here, and `σ²_Jac = 0` is a real possibility. -/
theorem popCrossCovRatio_tendsto_atTop {d : ℕ} {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (dθ dgRaw : Ω → Fin d → ℝ)
    (hJac : 0 < popCrossCovTrace μ dθ dgRaw)
    (hObj : 0 < popCrossCovTrace μ dgRaw dgRaw) :
    Tendsto (fun s : ℝ => popCrossCovTrace μ dθ (fun ω i => s * dgRaw ω i) /
        popCrossCovTrace μ (fun ω i => s * dgRaw ω i) (fun ω i => s * dgRaw ω i))
      (𝓝[>] (0 : ℝ)) atTop := by
  have h := attenuation_asymmetry hObj hJac
  refine h.2.2.congr fun s => ?_
  simp only [attenuationRatio]
  rw [popCrossCovTrace_attenuated_twice, popCrossCovTrace_attenuated_once]

end PopulationRatio

/-! ## 7. Beyond (A2): what a genuine diagonal prefactor does to the count -/

section DiagonalPrefactor

variable {d : ℕ}

/-- **The prefactor that (A2) idealizes away.** The free-coordinate gradient fluctuation obtained
from the constrained one by multiplying coordinate `i` by its own positivity-map derivative
`q i = ψ'(θ̃_i)`. Section 5's statements are the case of a constant `q`; the paper's (A2) is the
assertion that the constant case is a good approximation on the operative band. -/
def diagAttenuate (q : Fin d → ℝ) (dg : ℕ → Fin d → ℝ) : ℕ → Fin d → ℝ :=
  fun n i => q i * dg n i

/-- A constant diagonal prefactor is exactly the scalar attenuation of Section 5, so the results
below contain those of Section 5 as the case `q = σ_s · 1`. -/
theorem diagAttenuate_const (s : ℝ) (dg : ℕ → Fin d → ℝ) :
    diagAttenuate (fun _ => s) dg = fun n => s • dg n := rfl

/-- **Lower bound on the once-attenuated coupling trace under a diagonal prefactor.** Only the
lower bound `c ≤ q i` on the prefactor is used, together with the hypothesis that the coupling
`δθ̃_i δ(∇_θL)_i` is coordinatewise non-negative on the window. That sign hypothesis is not
decorative: without it the inequality is false, and `crossCovRatio_diag_sign_flip` below exhibits
data on which it fails by a change of sign. -/
theorem trace_diagAttenuate_coupling_ge {T t : ℕ} {q : Fin d → ℝ} {c : ℝ}
    (dθ dg : ℕ → Fin d → ℝ) (hq : ∀ i, c ≤ q i)
    (hsign : ∀ n ∈ Finset.range T, ∀ i, 0 ≤ dθ (t - T + n) i * dg (t - T + n) i) :
    c * (crossCovEstimator T t dθ dg).trace
      ≤ (crossCovEstimator T t dθ (diagAttenuate q dg)).trace := by
  rw [trace_crossCovEstimator, trace_crossCovEstimator, ← mul_assoc,
    mul_comm c ((T : ℝ)⁻¹), mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun n hn => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hterm : dθ (t - T + n) i * diagAttenuate q dg (t - T + n) i
      = q i * (dθ (t - T + n) i * dg (t - T + n) i) := by
    simp only [diagAttenuate]
    ring
  rw [hterm]
  exact mul_le_mul_of_nonneg_right (hq i) (hsign n hn i)

/-- **Upper bound on the twice-attenuated objective trace under a diagonal prefactor.** No sign
hypothesis is needed here: both slots carry the same prefactor, so every summand is a square. -/
theorem trace_diagAttenuate_obj_le {T t : ℕ} {q : Fin d → ℝ} {c : ℝ}
    (dg : ℕ → Fin d → ℝ) (hq0 : ∀ i, 0 ≤ q i) (hq : ∀ i, q i ≤ c) :
    (crossCovEstimator T t (diagAttenuate q dg) (diagAttenuate q dg)).trace
      ≤ c ^ 2 * (crossCovEstimator T t dg dg).trace := by
  rw [trace_crossCovEstimator, trace_crossCovEstimator, ← mul_assoc,
    mul_comm (c ^ 2) ((T : ℝ)⁻¹), mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun n _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hterm : diagAttenuate q dg (t - T + n) i * diagAttenuate q dg (t - T + n) i
      = q i ^ 2 * (dg (t - T + n) i * dg (t - T + n) i) := by
    simp only [diagAttenuate]
    ring
  rw [hterm]
  refine mul_le_mul_of_nonneg_right ?_ (mul_self_nonneg _)
  nlinarith [hq0 i, hq i]

/-- The matching lower bound for the objective channel. -/
theorem trace_diagAttenuate_obj_ge {T t : ℕ} {q : Fin d → ℝ} {c : ℝ}
    (dg : ℕ → Fin d → ℝ) (hc : 0 ≤ c) (hq : ∀ i, c ≤ q i) :
    c ^ 2 * (crossCovEstimator T t dg dg).trace
      ≤ (crossCovEstimator T t (diagAttenuate q dg) (diagAttenuate q dg)).trace := by
  rw [trace_crossCovEstimator, trace_crossCovEstimator, ← mul_assoc,
    mul_comm (c ^ 2) ((T : ℝ)⁻¹), mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun n _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hterm : diagAttenuate q dg (t - T + n) i * diagAttenuate q dg (t - T + n) i
      = q i ^ 2 * (dg (t - T + n) i * dg (t - T + n) i) := by
    simp only [diagAttenuate]
    ring
  rw [hterm]
  refine mul_le_mul_of_nonneg_right ?_ (mul_self_nonneg _)
  nlinarith [hq i, hc]

/-- **The `ℓ¹` mass of the coordinatewise coupling** over the window: the `1/T` average of
`∑_i |δθ̃_i δ(∇_θL)_i|`. It dominates the coupling trace, and the gap between the two is exactly
the cancellation that the signed sum `σ̂²_Jac` conceals — which is what a non-constant prefactor
can expose. It is the natural scale against which to measure how far from constant the positivity-map
derivative may be before the count breaks. -/
noncomputable def couplingAbs (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) : ℝ :=
  (T : ℝ)⁻¹ * ∑ n ∈ Finset.range T, ∑ i, |dθ (t - T + n) i * dg (t - T + n) i|

/-- The `ℓ¹` mass is non-negative. -/
theorem couplingAbs_nonneg (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    0 ≤ couplingAbs T t dθ dg := by
  rw [couplingAbs]
  refine mul_nonneg (by positivity) (Finset.sum_nonneg fun n _ => ?_)
  exact Finset.sum_nonneg fun i _ => abs_nonneg _

/-- The coupling trace never exceeds its `ℓ¹` mass. -/
theorem trace_crossCovEstimator_le_couplingAbs (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    (crossCovEstimator T t dθ dg).trace ≤ couplingAbs T t dθ dg := by
  rw [trace_crossCovEstimator, couplingAbs]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  exact Finset.sum_le_sum fun n _ => Finset.sum_le_sum fun i _ => le_abs_self _

/-- **The coupling trace under a prefactor that is nearly constant across coordinates.** If every
coordinate's positivity-map derivative lies within `r` of the common value `a`, the once-attenuated
coupling trace is within `r` times the coupling's `ℓ¹` mass of `a` times the unattenuated one.
This is the quantitative form of (A2)'s "approximately constant across coordinates", and unlike
`trace_diagAttenuate_coupling_ge` it needs **no** sign hypothesis: the deviation from the constant
prefactor is controlled in absolute value, so cancellation is bounded rather than excluded. -/
theorem trace_diagAttenuate_coupling_spread_ge {T t : ℕ} {q : Fin d → ℝ} {a r : ℝ}
    (dθ dg : ℕ → Fin d → ℝ) (hq : ∀ i, |q i - a| ≤ r) :
    a * (crossCovEstimator T t dθ dg).trace - r * couplingAbs T t dθ dg
      ≤ (crossCovEstimator T t dθ (diagAttenuate q dg)).trace := by
  have hT : (0 : ℝ) ≤ (T : ℝ)⁻¹ := by positivity
  have hrow : ∀ n ∈ Finset.range T,
      a * (∑ i, dθ (t - T + n) i * dg (t - T + n) i)
          - r * ∑ i, |dθ (t - T + n) i * dg (t - T + n) i|
        ≤ ∑ i, dθ (t - T + n) i * diagAttenuate q dg (t - T + n) i := by
    intro n _
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_le_sum fun i _ => ?_
    have hterm : dθ (t - T + n) i * diagAttenuate q dg (t - T + n) i
        = q i * (dθ (t - T + n) i * dg (t - T + n) i) := by
      simp only [diagAttenuate]
      ring
    have hb : |(q i - a) * (dθ (t - T + n) i * dg (t - T + n) i)|
        ≤ r * |dθ (t - T + n) i * dg (t - T + n) i| := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (hq i) (abs_nonneg _)
    have hlow : -(r * |dθ (t - T + n) i * dg (t - T + n) i|)
        ≤ (q i - a) * (dθ (t - T + n) i * dg (t - T + n) i) := (abs_le.mp hb).1
    have hsplit : q i * (dθ (t - T + n) i * dg (t - T + n) i)
        = a * (dθ (t - T + n) i * dg (t - T + n) i)
          + (q i - a) * (dθ (t - T + n) i * dg (t - T + n) i) := by ring
    rw [hterm, hsplit]
    linarith
  have hsum : a * (∑ n ∈ Finset.range T, ∑ i, dθ (t - T + n) i * dg (t - T + n) i)
        - r * ∑ n ∈ Finset.range T, ∑ i, |dθ (t - T + n) i * dg (t - T + n) i|
      ≤ ∑ n ∈ Finset.range T, ∑ i, dθ (t - T + n) i * diagAttenuate q dg (t - T + n) i := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_le_sum hrow
  rw [trace_crossCovEstimator, trace_crossCovEstimator, couplingAbs]
  have hrw : a * ((T : ℝ)⁻¹ * ∑ n ∈ Finset.range T, ∑ i, dθ (t - T + n) i * dg (t - T + n) i)
      - r * ((T : ℝ)⁻¹ * ∑ n ∈ Finset.range T, ∑ i, |dθ (t - T + n) i * dg (t - T + n) i|)
      = (T : ℝ)⁻¹ * (a * (∑ n ∈ Finset.range T, ∑ i, dθ (t - T + n) i * dg (t - T + n) i)
          - r * ∑ n ∈ Finset.range T, ∑ i, |dθ (t - T + n) i * dg (t - T + n) i|) := by
    ring
  rw [hrw]
  exact mul_le_mul_of_nonneg_left hsum hT

/-- **The divergence survives a genuine diagonal prefactor, under a sign condition.** Let the
positivity-map derivatives on the operative band lie in a band `c₁σ_s ≤ ψ'(θ̃_i) ≤ c₂σ_s` with
`0 < c₁ ≤ c₂` — a two-sided version of (A2) that no longer pretends the prefactor is constant
across coordinates — and let the coupling be coordinatewise non-negative on the window. Then the
ratio of the once-attenuated coupling trace to the twice-attenuated objective trace still diverges
as the positivity map flattens, and the proof exhibits the same `Ω(1/σ_s)` lower bound that drives the
scalar case, `c₁ σ̂²_Jac/(c₂² σ²_obj) · σ_s⁻¹`. Only that lower bound is proved; the matching
upper bound, and hence a two-sided `Θ(1/σ_s)` rate, is not claimed here.

The sign hypothesis is the price of dropping (A2), and it is a genuine assumption about the data
rather than a consequence of a nonzero cross-covariance: `crossCovRatio_diag_sign_flip` shows the
conclusion is false without it. The band itself is demanded only on a window `(0, s₀)` of
prefactor scales, not globally: (A2) scopes its idealization to "the operative shoulder region",
and the divergence is a statement about the limit `σ_s → 0⁺`, so nothing about large prefactors
is assumed. -/
theorem crossCovRatio_diag_tendsto_atTop (T t : ℕ) (dθ dgRaw : ℕ → Fin d → ℝ) {c₁ c₂ s₀ : ℝ}
    (q : ℝ → Fin d → ℝ) (hc₁ : 0 < c₁) (hc : c₁ ≤ c₂) (hs₀ : 0 < s₀)
    (hlo : ∀ s ∈ Set.Ioo (0 : ℝ) s₀, ∀ i, c₁ * s ≤ q s i)
    (hhi : ∀ s ∈ Set.Ioo (0 : ℝ) s₀, ∀ i, q s i ≤ c₂ * s)
    (hsign : ∀ n ∈ Finset.range T, ∀ i, 0 ≤ dθ (t - T + n) i * dgRaw (t - T + n) i)
    (hJac : 0 < (crossCovEstimator T t dθ dgRaw).trace)
    (hObj : 0 < (crossCovEstimator T t dgRaw dgRaw).trace) :
    Tendsto (fun s : ℝ => (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace /
        (crossCovEstimator T t (diagAttenuate (q s) dgRaw) (diagAttenuate (q s) dgRaw)).trace)
      (𝓝[>] (0 : ℝ)) atTop := by
  have hc₂ : 0 < c₂ := lt_of_lt_of_le hc₁ hc
  set J := (crossCovEstimator T t dθ dgRaw).trace with hJdef
  set O := (crossCovEstimator T t dgRaw dgRaw).trace with hOdef
  have key : ∀ s ∈ Set.Ioo (0 : ℝ) s₀,
      c₁ * J / (c₂ ^ 2 * O) * s⁻¹ ≤
        (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace /
          (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
            (diagAttenuate (q s) dgRaw)).trace := by
    intro s hs
    have hs0 : (0 : ℝ) < s := hs.1
    have hq0 : ∀ i, 0 ≤ q s i := fun i => le_trans (by positivity) (hlo s hs i)
    have hJq : c₁ * s * J ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace :=
      trace_diagAttenuate_coupling_ge dθ dgRaw (hlo s hs) hsign
    have hOle :
        (crossCovEstimator T t (diagAttenuate (q s) dgRaw) (diagAttenuate (q s) dgRaw)).trace
          ≤ (c₂ * s) ^ 2 * O := trace_diagAttenuate_obj_le dgRaw hq0 (hhi s hs)
    have hOge : (c₁ * s) ^ 2 * O
        ≤ (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
            (diagAttenuate (q s) dgRaw)).trace :=
      trace_diagAttenuate_obj_ge dgRaw (by positivity) (hlo s hs)
    have hOqpos : 0 < (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
        (diagAttenuate (q s) dgRaw)).trace := lt_of_lt_of_le (by positivity) hOge
    have hJqpos : 0 < c₁ * s * J := by positivity
    have hden : (0 : ℝ) < (c₂ * s) ^ 2 * O := by positivity
    have hnum : 0 ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace :=
      le_trans hJqpos.le hJq
    have step1 : c₁ * s * J / ((c₂ * s) ^ 2 * O)
        ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace
            / ((c₂ * s) ^ 2 * O) := by
      gcongr
    have step2 : (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace
          / ((c₂ * s) ^ 2 * O)
        ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace /
            (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
              (diagAttenuate (q s) dgRaw)).trace := by
      gcongr
    have heq : c₁ * J / (c₂ ^ 2 * O) * s⁻¹ = c₁ * s * J / ((c₂ * s) ^ 2 * O) := by
      field_simp
    rw [heq]
    exact le_trans step1 step2
  have hlim : Tendsto (fun s : ℝ => c₁ * J / (c₂ ^ 2 * O) * s⁻¹) (𝓝[>] (0 : ℝ)) atTop :=
    Filter.Tendsto.const_mul_atTop (by positivity) tendsto_inv_nhdsGT_zero
  exact tendsto_atTop_mono' _ (Filter.eventually_of_mem (Ioo_mem_nhdsGT hs₀) key) hlim

/-- **The sign condition does not empty the theorem.** Data satisfying every hypothesis of
`crossCovRatio_diag_tendsto_atTop` at once exists: a one-step window, a coordinatewise
non-negative coupling, a prefactor band of spread two, and strictly positive unattenuated traces.
The divergence proved above is therefore a statement about a nonempty class of runs, and the
coordinatewise sign hypothesis is a restriction rather than an impossibility. -/
theorem crossCovRatio_diag_tendsto_atTop_nonvacuous :
    Tendsto (fun s : ℝ =>
        (crossCovEstimator 1 1 (fun _ => ![(1 : ℝ), 1])
            (diagAttenuate ![s, 2 * s] (fun _ => ![(1 : ℝ), 1]))).trace /
          (crossCovEstimator 1 1 (diagAttenuate ![s, 2 * s] (fun _ => ![(1 : ℝ), 1]))
            (diagAttenuate ![s, 2 * s] (fun _ => ![(1 : ℝ), 1]))).trace)
      (𝓝[>] (0 : ℝ)) atTop := by
  refine crossCovRatio_diag_tendsto_atTop 1 1 (fun _ => ![(1 : ℝ), 1]) (fun _ => ![(1 : ℝ), 1])
    (c₁ := 1) (c₂ := 2) (s₀ := 1) (fun s => ![s, 2 * s]) one_pos (by norm_num) one_pos
    ?_ ?_ ?_ ?_ ?_
  · intro s hs i
    obtain ⟨hs0, -⟩ := hs
    fin_cases i
    all_goals simp
    all_goals linarith
  · intro s hs i
    obtain ⟨hs0, -⟩ := hs
    fin_cases i
    all_goals simp
    all_goals linarith
  · intro n _ i
    fin_cases i
    all_goals norm_num
  · rw [trace_crossCovEstimator]
    norm_num [Fin.sum_univ_two]
  · rw [trace_crossCovEstimator]
    norm_num [Fin.sum_univ_two]

/-- A function that is negative throughout `(0, ∞)` cannot diverge to `+∞` as its argument
decreases to zero. This is what turns the counterexample below into a refutation rather than a
suggestive computation. -/
theorem not_tendsto_atTop_of_neg_on_Ioi {f : ℝ → ℝ} (hneg : ∀ s : ℝ, 0 < s → f s < 0) :
    ¬ Tendsto f (𝓝[>] (0 : ℝ)) atTop := by
  intro hT
  have h1 : ∀ᶠ s : ℝ in 𝓝[>] (0 : ℝ), (0 : ℝ) < f s := hT.eventually (eventually_gt_atTop 0)
  have h2 : ∀ᶠ s : ℝ in 𝓝[>] (0 : ℝ), (0 : ℝ) < s := self_mem_nhdsWithin
  obtain ⟨s, hfs, hs⟩ := (h1.and h2).exists
  exact absurd hfs (not_lt.mpr (hneg s hs).le)

/-- **(A2) cannot be relaxed to a two-sided band alone.** The idealization is sometimes described
as harmless because the positivity-map derivative is nearly constant on a narrow band, so that replacing
`diag(ψ'(θ̃))` by a scalar changes only constants. It does not: the once-attenuated coupling trace
is a signed sum, and a bounded spread across coordinates is enough to reverse its sign.

Concretely, with `d = 2` and a single-step window, take
`δθ̃ = (1, -1)`, `δ(∇_θL) = (2, 1)`, and the prefactor `q(σ_s) = (σ_s, 10σ_s)`, so every
coordinate's prefactor lies between `σ_s` and `10σ_s`. The unattenuated coupling trace is `+1` and
the constrained objective trace is `5`, both strictly positive, so the hypotheses of
`crossCovRatio_tendsto_atTop` hold for the scalar reading. Yet the attenuated coupling trace is
`-8σ_s`, the attenuated objective trace is `104σ_s²`, and the ratio `-1/(13σ_s)` tends to `-∞`.

Consequently the conclusion of `crossCovRatio_diag_tendsto_atTop` genuinely needs its sign
hypothesis, and `σ²_Jac > 0` on the constrained coordinate does not by itself survive the passage
through a non-constant positivity-map derivative. -/
theorem crossCovRatio_diag_sign_flip :
    ∃ (dθ dgRaw : ℕ → Fin 2 → ℝ) (q : ℝ → Fin 2 → ℝ),
      (∀ s : ℝ, 0 < s → ∀ i, 1 * s ≤ q s i) ∧
      (∀ s : ℝ, 0 < s → ∀ i, q s i ≤ 10 * s) ∧
      0 < (crossCovEstimator 1 1 dθ dgRaw).trace ∧
      0 < (crossCovEstimator 1 1 dgRaw dgRaw).trace ∧
      ((∀ s : ℝ, 0 < s →
          (crossCovEstimator 1 1 dθ (diagAttenuate (q s) dgRaw)).trace /
            (crossCovEstimator 1 1 (diagAttenuate (q s) dgRaw)
              (diagAttenuate (q s) dgRaw)).trace < 0) ∧
        Tendsto (fun s : ℝ => (crossCovEstimator 1 1 dθ (diagAttenuate (q s) dgRaw)).trace /
            (crossCovEstimator 1 1 (diagAttenuate (q s) dgRaw)
              (diagAttenuate (q s) dgRaw)).trace) (𝓝[>] (0 : ℝ)) atBot ∧
        ¬ Tendsto (fun s : ℝ => (crossCovEstimator 1 1 dθ (diagAttenuate (q s) dgRaw)).trace /
            (crossCovEstimator 1 1 (diagAttenuate (q s) dgRaw)
              (diagAttenuate (q s) dgRaw)).trace) (𝓝[>] (0 : ℝ)) atTop) := by
  refine ⟨fun _ => ![1, -1], fun _ => ![2, 1], fun s => ![s, 10 * s], ?_, ?_, ?_, ?_, ?_⟩
  · intro s hs i
    fin_cases i
    all_goals simp
    all_goals linarith
  · intro s hs i
    fin_cases i
    all_goals simp
    all_goals linarith
  · rw [trace_crossCovEstimator]
    norm_num [Fin.sum_univ_two]
  · rw [trace_crossCovEstimator]
    norm_num [Fin.sum_univ_two]
  · have hnum : ∀ s : ℝ,
        (crossCovEstimator 1 1 (fun _ => ![(1 : ℝ), -1])
          (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))).trace = -8 * s := by
      intro s
      rw [trace_crossCovEstimator]
      simp [Fin.sum_univ_two, diagAttenuate]
      ring
    have hden : ∀ s : ℝ,
        (crossCovEstimator 1 1 (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))
          (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))).trace = 104 * s ^ 2 := by
      intro s
      rw [trace_crossCovEstimator]
      simp [Fin.sum_univ_two, diagAttenuate]
      ring
    have hneg : ∀ s : ℝ, 0 < s →
        (crossCovEstimator 1 1 (fun _ => ![(1 : ℝ), -1])
            (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))).trace /
          (crossCovEstimator 1 1 (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))
            (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))).trace < 0 := by
      intro s hs
      rw [hnum s, hden s]
      exact div_neg_of_neg_of_pos (by linarith) (by positivity)
    have hbot : Tendsto (fun s : ℝ =>
        (crossCovEstimator 1 1 (fun _ => ![(1 : ℝ), -1])
            (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))).trace /
          (crossCovEstimator 1 1 (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))
            (diagAttenuate ![s, 10 * s] (fun _ => ![(2 : ℝ), 1]))).trace)
        (𝓝[>] (0 : ℝ)) atBot := by
      have hlin : Tendsto (fun s : ℝ => (-8 / 104 : ℝ) * s⁻¹) (𝓝[>] (0 : ℝ)) atBot :=
        (tendsto_const_mul_atBot_of_neg (by norm_num)).mpr tendsto_inv_nhdsGT_zero
      refine hlin.congr' ?_
      filter_upwards [self_mem_nhdsWithin] with s hs
      have hs0 : (s : ℝ) ≠ 0 := ne_of_gt hs
      rw [hnum s, hden s]
      field_simp
    exact ⟨hneg, hbot, not_tendsto_atTop_of_neg_on_Ioi hneg⟩

/-- **The divergence survives a genuine diagonal prefactor with no sign condition, once (A2) is
read quantitatively.** (A2) does not say the positivity-map derivative is merely *bounded* between two
multiples of `σ_s`; it says it is **approximately constant** across the operative band. Made
quantitative, that is `|ψ'(θ̃_i) - σ_s| ≤ ε σ_s` for every coordinate, with `ε` the relative
spread. Under that hypothesis the ratio of the once-attenuated coupling trace to the
twice-attenuated objective trace diverges as `σ_s → 0⁺`, and no assumption is made about the sign
of the coordinatewise coupling.

The price is a threshold on `ε`, and the threshold is the honest content of the statement: the
spread has to be small relative to the fraction of the coupling's `ℓ¹` mass that survives
cancellation in the signed trace, `ε · couplingAbs < σ̂²_Jac`. That is exactly the condition the
sign-flip counterexample above violates, and violates by a wide margin: there the prefactor runs
from `σ_s` to `10σ_s`, so `ε = 9` against an `ℓ¹` mass of `3` and a coupling trace of `1`. When
the coupling *is* coordinatewise non-negative the mass equals the trace and the threshold reduces
to `ε < 1`, that is, to a prefactor band that never changes sign. As with the sign-condition
theorem, the band is demanded
only on a window `(0, s₀)` of prefactor scales, since the conclusion is a statement about the
limit. Positivity of the coupling trace is not assumed: it follows from the threshold, because the
`ℓ¹` mass is non-negative. -/
theorem crossCovRatio_diag_tendsto_atTop_of_small_spread (T t : ℕ) (dθ dgRaw : ℕ → Fin d → ℝ)
    {ε s₀ : ℝ} (q : ℝ → Fin d → ℝ) (hε : 0 ≤ ε) (hs₀ : 0 < s₀)
    (hband : ∀ s ∈ Set.Ioo (0 : ℝ) s₀, ∀ i, |q s i - s| ≤ ε * s)
    (hObj : 0 < (crossCovEstimator T t dgRaw dgRaw).trace)
    (hspread : ε * couplingAbs T t dθ dgRaw < (crossCovEstimator T t dθ dgRaw).trace) :
    Tendsto (fun s : ℝ => (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace /
        (crossCovEstimator T t (diagAttenuate (q s) dgRaw) (diagAttenuate (q s) dgRaw)).trace)
      (𝓝[>] (0 : ℝ)) atTop := by
  set J := (crossCovEstimator T t dθ dgRaw).trace with hJdef
  set O := (crossCovEstimator T t dgRaw dgRaw).trace with hOdef
  set M := couplingAbs T t dθ dgRaw with hMdef
  have hMnn : 0 ≤ M := couplingAbs_nonneg T t dθ dgRaw
  have hJac : 0 < J := lt_of_le_of_lt (mul_nonneg hε hMnn) hspread
  have hJM : J ≤ M := trace_crossCovEstimator_le_couplingAbs T t dθ dgRaw
  have hMpos : 0 < M := lt_of_lt_of_le hJac hJM
  have hε1 : ε < 1 := by
    by_contra hcon
    have hge : (1 : ℝ) ≤ ε := not_lt.mp hcon
    have h := mul_le_mul_of_nonneg_right hge hMpos.le
    rw [one_mul] at h
    linarith
  have h1pe : (0 : ℝ) < 1 + ε := by linarith
  have h1me : (0 : ℝ) < 1 - ε := by linarith
  have hnum0 : 0 < J - ε * M := by linarith
  have hOne : O ≠ 0 := ne_of_gt hObj
  have hpne : (1 : ℝ) + ε ≠ 0 := ne_of_gt h1pe
  have key : ∀ s ∈ Set.Ioo (0 : ℝ) s₀,
      (J - ε * M) / ((1 + ε) ^ 2 * O) * s⁻¹ ≤
        (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace /
          (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
            (diagAttenuate (q s) dgRaw)).trace := by
    intro s hs
    have hs0 : (0 : ℝ) < s := hs.1
    have hsne : s ≠ 0 := ne_of_gt hs0
    have hbnd : ∀ i, -(ε * s) ≤ q s i - s ∧ q s i - s ≤ ε * s :=
      fun i => abs_le.mp (hband s hs i)
    have hqlo : ∀ i, (1 - ε) * s ≤ q s i := by
      intro i
      have h := (hbnd i).1
      nlinarith
    have hqhi : ∀ i, q s i ≤ (1 + ε) * s := by
      intro i
      have h := (hbnd i).2
      nlinarith
    have hq0 : ∀ i, 0 ≤ q s i := fun i => le_trans (mul_pos h1me hs0).le (hqlo i)
    have hNum : s * J - ε * s * M
        ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace :=
      trace_diagAttenuate_coupling_spread_ge dθ dgRaw (fun i => hband s hs i)
    have hDle : (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
          (diagAttenuate (q s) dgRaw)).trace ≤ ((1 + ε) * s) ^ 2 * O :=
      trace_diagAttenuate_obj_le dgRaw hq0 hqhi
    have hDge : ((1 - ε) * s) ^ 2 * O
        ≤ (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
            (diagAttenuate (q s) dgRaw)).trace :=
      trace_diagAttenuate_obj_ge dgRaw (mul_pos h1me hs0).le hqlo
    have hDpos : 0 < (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
        (diagAttenuate (q s) dgRaw)).trace :=
      lt_of_lt_of_le (mul_pos (pow_pos (mul_pos h1me hs0) 2) hObj) hDge
    have hNumPos : 0 < s * J - ε * s * M := by nlinarith [mul_pos hs0 hnum0]
    have hnumnn : 0 ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace :=
      le_trans hNumPos.le hNum
    have hden : (0 : ℝ) < ((1 + ε) * s) ^ 2 * O := mul_pos (pow_pos (mul_pos h1pe hs0) 2) hObj
    have step1 : (s * J - ε * s * M) / (((1 + ε) * s) ^ 2 * O)
        ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace
            / (((1 + ε) * s) ^ 2 * O) := by
      gcongr
    have step2 : (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace
          / (((1 + ε) * s) ^ 2 * O)
        ≤ (crossCovEstimator T t dθ (diagAttenuate (q s) dgRaw)).trace /
            (crossCovEstimator T t (diagAttenuate (q s) dgRaw)
              (diagAttenuate (q s) dgRaw)).trace := by
      gcongr
    have heq : (J - ε * M) / ((1 + ε) ^ 2 * O) * s⁻¹
        = (s * J - ε * s * M) / (((1 + ε) * s) ^ 2 * O) := by
      field_simp
    rw [heq]
    exact le_trans step1 step2
  have hlim : Tendsto (fun s : ℝ => (J - ε * M) / ((1 + ε) ^ 2 * O) * s⁻¹)
      (𝓝[>] (0 : ℝ)) atTop :=
    Filter.Tendsto.const_mul_atTop (div_pos hnum0 (mul_pos (pow_pos h1pe 2) hObj))
      tendsto_inv_nhdsGT_zero
  exact tendsto_atTop_mono' _ (Filter.eventually_of_mem (Ioo_mem_nhdsGT hs₀) key) hlim

/-- **The quantitative reading of (A2) rescues the sign-flip data.** The counterexample above is
not a refutation of (A2); it is a refutation of the weaker "bounded spread" reading of it. On the
very same window — `δθ̃ = (1, -1)`, `δ(∇_θL) = (2, 1)`, whose coupling is *not* coordinatewise
non-negative, so `crossCovRatio_diag_tendsto_atTop` does not apply — a relative spread of `1/4`
in place of the counterexample's factor of ten restores the divergence: the prefactor
`q(σ_s) = (σ_s, (5/4)σ_s)` gives the attenuated coupling trace `(3/4)σ_s` rather than `-8σ_s`,
and the ratio tends to `+∞`. The threshold is met with room to spare, the coupling's `ℓ¹` mass
being `3` against a trace of `1`, so that `ε · 3 = 3/4 < 1`. -/
theorem crossCovRatio_diag_small_spread_rescues_sign_flip :
    Tendsto (fun s : ℝ =>
        (crossCovEstimator 1 1 (fun _ => ![(1 : ℝ), -1])
            (diagAttenuate ![s, (5 / 4) * s] (fun _ => ![(2 : ℝ), 1]))).trace /
          (crossCovEstimator 1 1 (diagAttenuate ![s, (5 / 4) * s] (fun _ => ![(2 : ℝ), 1]))
            (diagAttenuate ![s, (5 / 4) * s] (fun _ => ![(2 : ℝ), 1]))).trace)
      (𝓝[>] (0 : ℝ)) atTop := by
  refine crossCovRatio_diag_tendsto_atTop_of_small_spread 1 1 (fun _ => ![(1 : ℝ), -1])
    (fun _ => ![(2 : ℝ), 1]) (ε := 1 / 4) (s₀ := 1) (fun s => ![s, (5 / 4) * s])
    (by norm_num) one_pos ?_ ?_ ?_
  · intro s hs i
    obtain ⟨hs0, -⟩ := hs
    fin_cases i
    all_goals simp
    · linarith
    · rw [show (5 / 4 : ℝ) * s - s = 4⁻¹ * s by ring,
        abs_of_nonneg (by linarith : (0 : ℝ) ≤ 4⁻¹ * s)]
  · rw [trace_crossCovEstimator]
    norm_num [Fin.sum_univ_two]
  · rw [couplingAbs, trace_crossCovEstimator]
    norm_num [Fin.sum_univ_two]

end DiagonalPrefactor

end IcnnLift
