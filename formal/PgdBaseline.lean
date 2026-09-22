import Mathlib
import TraceLemmas
import CrossCov
import CouplingSign
import BarrierRescaling
import FWReduction
import KramersExitTime
import QuasipotentialMonotone
import StructuralZeros

/-!
# The projected-gradient baseline and the alignment threshold (Theorem A)

This file machine-checks the directional escape comparison between the lift and **projected
gradient descent** that `docs/design/pgd_theory_angle_2026-09-01.md` section 3 recommends, after
that note closed three other candidate explanations with verified negative results. The paper
(`docs/paper/v4/04_mechanism.tex`) explains the lift's advantage over the *direct* softplus arm as
a temperature story: the shoulder attenuates the slope to `s = psi'`, the effective
diffusion collapses as `Theta(eta s^2)`, and escape costs `exp(Theta(1/s^2))`. Projected gradient
descent gets no such account, and it cannot get the same one, because it outputs raw parameters into
the loss and so carries no positivity-map prefactor at all.

The note's arithmetic (its negative result N2) is that the temperature channel therefore runs the
*wrong way*: with `Sigma_PGD = Sigma_obj`, `Sigma_direct = s^2 Sigma_obj` and
`Sigma_lift = s^2 Sigma_obj + s^2 (C + H V H)`, the projected baseline explores at a strictly
higher temperature than the direct arm whenever `s < 1`, by the computable factor `1/s^2` — about
five at a typical interior constrained weight — and it is not dominated by the lift either, since
the lift pays that same factor on the whole of its diffusion and must overcome it before its
directional reading catches up. No account of escape *magnitude* can therefore favour the lift over
the projected baseline. What the lift has that the projected baseline structurally cannot have is
a component of the diffusion correlated with the gradient fluctuation itself, the coupling term
`C = H Sigma_slack + Sigma_slack^T H` that Theorem 1 certifies. Since a Freidlin-Wentzell action
sees the diffusion only through its value along the escape path, the comparison must be made on
`v^T Sigma v` and not on the trace, and it then has a threshold form.

**What this file proves.** Theorem A as a biconditional: with the barrier action `alpha > 0`, the
learning rate `eta > 0` and both directional diffusions positive, the lift's Arrhenius exponent is
strictly below the projected baseline's **if and only if**
`v^T (C + H V H) v > (s^{-2} - 1) v^T Sigma_obj v`, equivalently `rho_v > s^{-2} - 1` with `rho_v`
the framework's own temperature ratio evaluated along `v`. The biconditional is a theorem whatever
the empirical measurement of `rho_v` turns out to be: the gate decides what the paper may claim,
not whether the comparison holds. The file also carries the negative direction honestly — an
explicit two-dimensional instance with `s < 1`, a positive semidefinite magnitude channel and no
coupling at all in which the lift's directional diffusion is strictly *below* the projected
baseline's, so that the failure of the domination `Sigma_lift >= Sigma_PGD` is machine-checked and
not merely conceded — and the structural point that the projected baseline's coupling term is
identically zero, so that `rho_v = 0` for it and the alignment route is unavailable to it at any
temperature.

**What this file does not prove, and why.** Four things.

First, the note displays `log E[tau] = 2 alpha / (eta v^T Sigma v) + o(1/eta)`. The `o(1/eta)`
remainder is *not* formalized here and nothing in this file asserts it. mathlib v4.31.0 has no Ito
calculus, no stochastic differential equation, no diffusion generator and no exit time, as
`KramersExitTime.lean`'s own docstring records from a whole-tree search; the passage from the
dynamics to any Arrhenius statement is carried in that module as the named hypothesis `hDynkin`,
Dynkin's formula for the mean first-passage time, and it is inherited here unchanged. What is
proved about exponents is proved about `BarrierRescaling.arrheniusExponent`, which is the exponent
`2 alpha / (eta sigma2)` as a function and not a theorem about any process.

Second, the escape-time consequence is stated only in the regime `KramersExitTime.lean` actually
delivers. That module pins the exit time two-sided but with crude prefactors, so the ordering of
two prescribed exit times needs an explicit gap inequality between the proved bounds; the ordering
free of any gap hypothesis holds in the small-noise limit. Both forms appear below with their
conditions in the statement, never in a docstring.

Third, the alignment itself. That the coupling channel is positively aligned with the escape
direction is assumption (A7) of the note, the directional strengthening of (A6), and it is an
empirical bet, not a theorem: `CouplingSign.lean` proves that the sign of `C` is not automatic
even under the remainder control of (A4'), and exhibits a configuration where (A6) fails for every
constant. Nothing here derives it. The gate is stated so that a reader can evaluate it against
measured matrices.

Fourth, the reduction from the multi-dimensional Freidlin-Wentzell action to the scalar comparison
along `v` is `FWReduction.lean`'s business (`fwActionScalar_le_fwActionMulti_alongLine`, and the
identity `fwActionMulti_alongLine_arrheniusExponent`); this file consumes its `diffusionAlong` and
does not re-derive the projection. In particular the claim that the improving move really is a
crossing along a single fixed direction is a hypothesis of that reduction and not established
here.

**One discrepancy in the source note, reported rather than silently resolved.** Section 3.1 of the
note states the gate with numerator `v^T (C + H V H) v`, and its temperature ratio `rho` is the
framework's `(coupling + quadratic) / (s^2 sigma_obj^2)` of `docs/design/lemma1_rederivation.md`
(L3'); read along `v` the factor `s^2` cancels and that ratio is exactly
`v^T (C + H V H) v / v^T Sigma_obj v`, which is what `alignmentRatio` is defined to be and what
`alignmentRatio_eq_excess_div_direct` verifies. But the measurement recipe P0 in section 4 of the
note computes `v^T C v / v^T Sigma_obj v`, dropping the magnitude channel from the numerator. The
two disagree, and the disagreement has a sign: since `v^T H V H v >= 0` always, P0's ratio never
exceeds the gate's, so P0 clearing the bar implies the gate holds
(`gateHolds_of_couplingOnly_gate`), while P0 failing the bar does not imply the gate fails
(`exists_gateHolds_of_couplingOnly_gate_failure`). A P0 run that comes back below the bar
therefore does not by itself close the question; it closes it only together with a measurement of
the magnitude channel along the same direction.

## Results
* `pgdDiffusionAlong`, `directDiffusionAlong`, `liftDiffusionAlong` — the three arms' diffusions
  read along the escape direction, in the note's section 3.1 display.
* `pgdDiffusionMatrix`, `directDiffusionMatrix`, `liftDiffusionMatrix` — the same three as
  matrices, with `diffusionAlong_liftDiffusionMatrix` and its companions identifying the readings.
* `liftDiffusionAlong_eq_projDiffusionLift` — agreement with `CouplingSign.projDiffusionLift` at
  zero third-order term, so the object compared here is the one (FW-1) delivers.
* `directDiffusionAlong_le_liftDiffusionAlong` — the direct arm never exceeds the lift along `v`,
  given only that the coupling channel is non-negative there; the magnitude channel's sign is
  discharged from `TraceLemmas.quadForm_conj_nonneg`. Its companion
  `directDiffusionAlong_le_liftDiffusionAlong_of_positiveAlignment` takes (A6) itself.
* `directDiffusionAlong_lt_pgdDiffusionAlong` — the direct arm is strictly below the projected
  baseline whenever `s < 1`, unconditionally in the coupling.
* `exists_liftDiffusionAlong_lt_pgdDiffusionAlong` — **N2**: the lift does not dominate the
  projected baseline. An explicit instance with `s < 1`, `C = 0` and a strictly positive magnitude
  channel in which the lift's directional diffusion is strictly smaller.
* `not_posSemidef_liftDiffusionMatrix_sub_pgdDiffusionMatrix` — and the matrix domination
  `Sigma_lift >= Sigma_PGD` fails even at a gate-passing instance.
* `arrheniusExponent_lift_lt_pgd_iff` and `arrheniusExponent_lift_lt_pgd_iff_gate` — **Theorem A**,
  in both the explicit and the `rho_v` form.
* `liftDiffusionAlong_add_remainder`, `arrheniusExponent_lift_lt_pgd_iff_of_remainder`
  (added 2026-09-13) — Theorem A net of the remainder channel: the remainder enters the lift's
  directional diffusion additively, so the threshold is the same arithmetic with `vᵀ R v` added
  to the excess. This discharges the appendix sentence that said that arithmetic was outside the
  formalization; `R = 0` returns `thm:directional` as stated.
* `adamVarianceMap`, `adamVarianceMap_eq_add`, `adamVarianceMap_zero`,
  `adamVarianceMap_strictAntiOn`, `AdamQuasiStatic`, `adam_exponent_lt_iff`,
  `adamQuasiStatic_satisfiable` (added 2026-09-13) — `rem:adam`'s reduction, split into its two
  halves: the map `V ↦ (√V + ε)/V` is strictly decreasing on the positive reals (a theorem) and
  the quasi-static reading that the Adam exponent *is* `2α/η` times that map (a named
  hypothesis, in the register of the Dynkin inputs). Together they give the remark's
  equivalence between an inequality of exponents and the reverse inequality of variances, so
  every ordering conclusion of `app:proof-pgd` transports.
* `alignmentRatio`, `gateThreshold`, `GateHolds` — the gate, in the note's own convention.
* `alignmentRatio_eq_excess_div_direct` — that convention is the framework's temperature ratio.
* `gateHolds_of_couplingOnly_gate`, `exists_gateHolds_of_couplingOnly_gate_failure` — the exact
  logical strength of the P0 measurement recipe.
* `meanExitTime_lift_lt_pgd_of_window`, `eventually_meanExitTime_lift_lt_pgd_of_gate`,
  `eventually_mean_first_passage_lift_lt_pgd_of_gate` — the escape-time consequence, in the two
  regimes `KramersExitTime.lean` delivers, the third under Dynkin's formula.
* `quasipotentialIncrement_lift_le_pgd_of_gate` — (FW-2) instantiated at the projected baseline,
  which its reciprocal-gap statement accepts even though the matrix domination fails.
* `pgd_alignment_structurally_unavailable` — **Theorem C**: with no batch-conditioned body the
  coupling term vanishes, the projected baseline's directional diffusion is exactly
  `v^T Sigma_obj v`, its `rho_v` is zero and the gate fails for it identically.
* `gateHolds_witness`, `gateWitness_arrheniusExponent_lift_lt_pgd`, `gateWitness_trace_lt` —
  non-vacuity: a `2 x 2` instance clearing the gate, whose slack cross-covariance is the exact
  (A4) chain's `V H`, which by Theorem A escapes strictly faster while carrying strictly *less*
  total noise than the projected baseline, `tr Sigma_lift = 6 < 9 = tr Sigma_PGD`.
* `gate_discriminates_on_slope` — a positively aligned configuration that clears the gate at
  `s = 3/5` and fails it at `s = 1/3`, so the gate is a real discriminator and positive alignment
  alone is not sufficient.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. The explicit hypotheses this file carries are these. The
positivity of the two directional diffusions is a hypothesis of Theorem A and is not derived; it
fails, for instance, in a direction on which the objective noise vanishes. The alignment (A7) is
never assumed silently: every statement that needs the coupling channel's sign takes
`0 <= diffusionAlong C v` as a named hypothesis, and the two theorems that consume (A6) take
`CouplingSign.PositiveAlignmentOn` itself. The exit-time statements inherit `KramersExitTime`'s
single-barrier shape, continuity, point ordering, and — for the mean first-passage forms —
Dynkin's formula, which would be discharged only by a stochastic calculus that this mathlib does
not have. The reduction to a single escape direction is inherited from `FWReduction`. Nothing in
the file asserts that the gate is met on any measured data.
-/

open Matrix
open scoped Topology

namespace IcnnLift

variable {n : Type*} [Fintype n]

/-! ## 1. The three diffusions, read along the escape direction

Freidlin-Wentzell sees a diffusion only through its value along the escape path, so the three arms
are compared through `FWReduction.diffusionAlong D v = v ⬝ᵥ (D *ᵥ v)` and not through the trace.
The three matrices are the display of section 3.1 of the note. -/

/-- The **projected-gradient baseline's** diffusion matrix, `Sigma_PGD = Sigma_obj`. Projected
gradient descent outputs raw parameters into the loss, so its update gradient carries no positivity-map
prefactor `s = psi'`, and by Theorem 1 (`thm:joint-necessity`) its slack-channel cross-covariance
is a structural zero, so no coupling term is added either. Both facts are inputs, not claims of
this definition: the second is `pgd_alignment_structurally_unavailable` below. -/
def pgdDiffusionMatrix (Sobj : Matrix n n ℝ) : Matrix n n ℝ := Sobj

/-- The **direct softplus arm's** diffusion matrix, `Sigma_direct = s^2 Sigma_obj`: the objective
noise attenuated twice by the positivity-map slope, once through each factor of the update outer
product. -/
def directDiffusionMatrix (s : ℝ) (Sobj : Matrix n n ℝ) : Matrix n n ℝ := s ^ 2 • Sobj

/-- The **lift's** diffusion matrix, `Sigma_lift = s^2 (Sigma_obj + C + H V H)` with
`C = H Sigma_slack + Sigma_slack^T H` the coupling term of `CouplingSign.couplingTerm` and
`H V H` the magnitude term. The decomposition itself is fact (ii) of the corrected mechanism,
machine-checked in `UpdateCovariance.lean` and `DiffusionOrdering.lean`; it is not re-derived
here. -/
def liftDiffusionMatrix (s : ℝ) (Sobj C HVH : Matrix n n ℝ) : Matrix n n ℝ :=
  s ^ 2 • (Sobj + C + HVH)

/-- The projected-gradient baseline's diffusion **along the escape direction** `v`, the quantity
the Freidlin-Wentzell action actually sees: `v ⬝ᵥ (Sigma_obj *ᵥ v)`. -/
def pgdDiffusionAlong (v : n → ℝ) (Sobj : Matrix n n ℝ) : ℝ := diffusionAlong Sobj v

/-- The direct softplus arm's diffusion along `v`: `s^2 (v ⬝ᵥ (Sigma_obj *ᵥ v))`. -/
def directDiffusionAlong (s : ℝ) (v : n → ℝ) (Sobj : Matrix n n ℝ) : ℝ :=
  s ^ 2 * diffusionAlong Sobj v

/-- The lift's diffusion along `v`: `s^2 (v ⬝ᵥ ((Sigma_obj + C + H V H) *ᵥ v))`. -/
def liftDiffusionAlong (s : ℝ) (v : n → ℝ) (Sobj C HVH : Matrix n n ℝ) : ℝ :=
  s ^ 2 * diffusionAlong (Sobj + C + HVH) v

/-- The quadratic form is additive in the matrix, which is what lets the lift's reading split into
a baseline, a coupling channel and a magnitude channel. -/
theorem diffusionAlong_add (A B : Matrix n n ℝ) (v : n → ℝ) :
    diffusionAlong (A + B) v = diffusionAlong A v + diffusionAlong B v := by
  simp [diffusionAlong, Matrix.add_mulVec, dotProduct_add]

/-- The reading of `Sigma_PGD` along `v` is the projected baseline's directional diffusion. -/
theorem diffusionAlong_pgdDiffusionMatrix (Sobj : Matrix n n ℝ) (v : n → ℝ) :
    diffusionAlong (pgdDiffusionMatrix Sobj) v = pgdDiffusionAlong v Sobj := rfl

/-- The reading of `Sigma_direct` along `v` is the direct arm's directional diffusion. -/
theorem diffusionAlong_directDiffusionMatrix (s : ℝ) (Sobj : Matrix n n ℝ) (v : n → ℝ) :
    diffusionAlong (directDiffusionMatrix s Sobj) v = directDiffusionAlong s v Sobj := by
  simp [diffusionAlong, directDiffusionMatrix, directDiffusionAlong, Matrix.smul_mulVec,
    dotProduct_smul]

/-- The reading of `Sigma_lift` along `v` is the lift's directional diffusion. -/
theorem diffusionAlong_liftDiffusionMatrix (s : ℝ) (Sobj C HVH : Matrix n n ℝ) (v : n → ℝ) :
    diffusionAlong (liftDiffusionMatrix s Sobj C HVH) v = liftDiffusionAlong s v Sobj C HVH := by
  rw [liftDiffusionMatrix, diffusionAlong, liftDiffusionAlong, diffusionAlong,
    Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- The lift's reading, split into the direct baseline and the two excess channels. This is the
form every comparison below is proved in. -/
theorem liftDiffusionAlong_eq (s : ℝ) (v : n → ℝ) (Sobj C HVH : Matrix n n ℝ) :
    liftDiffusionAlong s v Sobj C HVH
      = directDiffusionAlong s v Sobj + s ^ 2 * diffusionAlong (C + HVH) v := by
  rw [liftDiffusionAlong, directDiffusionAlong, add_assoc, diffusionAlong_add]
  ring

/-- The object compared here is the one (FW-1) delivers. With the coupling term instantiated as
`CouplingSign.couplingTerm H Sigma_slack` and the magnitude term as `H V H`, the lift's directional
diffusion is `CouplingSign.projDiffusionLift` at third-order term zero. The third-order term is
carried explicitly in that module and is *not* assumed to vanish anywhere in this file; setting it
to zero here identifies the two definitions and nothing else. -/
theorem liftDiffusionAlong_eq_projDiffusionLift (s : ℝ) (v : n → ℝ)
    (Sobj V H S : Matrix n n ℝ) :
    liftDiffusionAlong s v Sobj (couplingTerm H S) (H * V * H)
      = projDiffusionLift s Sobj V H S 0 v := by
  rw [liftDiffusionAlong_eq, diffusionAlong_add, projDiffusionLift, projDiffusionDirect,
    directDiffusionAlong]
  simp only [diffusionAlong, crossTerm]
  ring

/-- The cross term of `CouplingSign` is the coupling term's reading along the direction: the two
vocabularies agree by definition. -/
theorem crossTerm_eq_diffusionAlong (H S : Matrix n n ℝ) (v : n → ℝ) :
    crossTerm H S v = diffusionAlong (couplingTerm H S) v := rfl

/-- **The direct arm never exceeds the lift along the escape direction.** The magnitude channel's
sign is not assumed: `H V H` is positive semidefinite for positive semidefinite `V` and symmetric
`H`, so its quadratic form is non-negative by `TraceLemmas.quadForm_conj_nonneg`. Only the coupling
channel's sign is a hypothesis, and it is exactly the directional reading of (A6) — the note's
(A7) — which `CouplingSign.lean` proves is not automatic. -/
theorem directDiffusionAlong_le_liftDiffusionAlong {V H C : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (s : ℝ) (v : n → ℝ) (Sobj : Matrix n n ℝ)
    (hC : 0 ≤ diffusionAlong C v) :
    directDiffusionAlong s v Sobj ≤ liftDiffusionAlong s v Sobj C (H * V * H) := by
  have hq : 0 ≤ diffusionAlong (H * V * H) v := quadForm_conj_nonneg hV hH v
  have hsum : 0 ≤ diffusionAlong (C + H * V * H) v := by
    rw [diffusionAlong_add]; linarith
  rw [liftDiffusionAlong_eq]
  nlinarith [sq_nonneg s]

/-- The same with (A6) itself in place of a bare sign hypothesis: on a band where the coupling term
dominates `kappa I` uniformly, the direct arm never exceeds the lift in any direction. -/
theorem directDiffusionAlong_le_liftDiffusionAlong_of_positiveAlignment {ι : Type*} {B : Set ι}
    {Hf Sf : ι → Matrix n n ℝ} {kappa : ℝ} {V : Matrix n n ℝ} {x : ι}
    (hA6 : PositiveAlignmentOn B Hf Sf kappa) (hx : x ∈ B) (hV : V.PosSemidef)
    (hH : (Hf x).IsHermitian) (s : ℝ) (v : n → ℝ) (Sobj : Matrix n n ℝ) :
    directDiffusionAlong s v Sobj
      ≤ liftDiffusionAlong s v Sobj (couplingTerm (Hf x) (Sf x)) (Hf x * V * Hf x) := by
  have hdot : 0 ≤ v ⬝ᵥ v := dotProduct_self_nonneg v
  have halign : kappa * (v ⬝ᵥ v) ≤ crossTerm (Hf x) (Sf x) v := hA6.2 x hx v
  have hC : 0 ≤ diffusionAlong (couplingTerm (Hf x) (Sf x)) v := by
    rw [← crossTerm_eq_diffusionAlong]
    nlinarith [hA6.1]
  exact directDiffusionAlong_le_liftDiffusionAlong hV hH s v Sobj hC

/-- **The direct arm is strictly below the projected baseline whenever the positivity map attenuates.**
This is the first half of the note's N2, and it holds unconditionally in the coupling: the positivity-map
prefactor is a pure loss of temperature. -/
theorem directDiffusionAlong_lt_pgdDiffusionAlong {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1)
    (v : n → ℝ) {Sobj : Matrix n n ℝ} (hpos : 0 < pgdDiffusionAlong v Sobj) :
    directDiffusionAlong s v Sobj < pgdDiffusionAlong v Sobj := by
  have hs2 : s ^ 2 < 1 := by nlinarith
  have h : s ^ 2 * diffusionAlong Sobj v < 1 * diffusionAlong Sobj v :=
    mul_lt_mul_of_pos_right hs2 hpos
  simpa [directDiffusionAlong, pgdDiffusionAlong] using h

/-! ### N2: the lift does not dominate the projected baseline

The note's second negative result says that no account of escape *magnitude* can favour the lift
over projected gradient descent, because `T_direct << T_lift << T_PGD` always. Its honest formal
content is that the ordering `liftDiffusionAlong < pgdDiffusionAlong` is genuinely attainable —
with the positivity map attenuating, no coupling at all, and a magnitude channel that is a legitimate
positive definite congruence. -/

/-- The direction the two-dimensional witnesses below are read along, the first basis vector. -/
def witnessDirection : Fin 2 → ℝ := ![1, 0]

/-- The witness direction is a unit vector, so the readings below are genuine directional
diffusions and not artefacts of a rescaling. -/
theorem witnessDirection_dotProduct_self : witnessDirection ⬝ᵥ witnessDirection = 1 := by
  simp [witnessDirection, dotProduct, Fin.sum_univ_two]

/-- The magnitude channel of the N2 witness, `V = diag(1, 0)`, read through the identity Hessian.
It is positive semidefinite and strictly positive along the escape direction, so the instance is
not degenerate: the lift really does carry an excess, and still loses. -/
def n2WitnessV : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 0]

/-- The N2 witness's magnitude matrix is positive semidefinite. -/
theorem n2WitnessV_posSemidef : n2WitnessV.PosSemidef := by
  refine Matrix.posSemidef_iff_dotProduct_mulVec.mpr ⟨?_, fun x => ?_⟩
  · rw [n2WitnessV, Matrix.IsHermitian]
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  · have : star x ⬝ᵥ (n2WitnessV *ᵥ x) = x 0 ^ 2 := by
      simp [n2WitnessV, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
      ring
    rw [this]
    positivity

/-- **N2, machine-checked.** There is an instance with a strictly attenuating positivity map `s = 1/2`, a
zero coupling term, a positive semidefinite magnitude channel that is strictly positive along the
escape direction, and a positive definite objective covariance, in which the lift's directional
diffusion is strictly **below** the projected baseline's. No comparison of escape magnitudes can
therefore favour the lift over the projected baseline; the only route left open is the direction
of the diffusion, which is Theorem A. -/
theorem exists_liftDiffusionAlong_lt_pgdDiffusionAlong :
    ∃ (Sobj V H : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) (s : ℝ),
      Sobj.PosDef ∧ V.PosSemidef ∧ H.IsHermitian ∧ 0 < s ∧ s < 1 ∧ v ⬝ᵥ v = 1 ∧
        0 < diffusionAlong (H * V * H) v ∧
        liftDiffusionAlong s v Sobj 0 (H * V * H) < pgdDiffusionAlong v Sobj := by
  refine ⟨1, n2WitnessV, 1, witnessDirection, 1 / 2, Matrix.PosDef.one, n2WitnessV_posSemidef,
    Matrix.isHermitian_one, by norm_num, by norm_num, witnessDirection_dotProduct_self, ?_, ?_⟩
  · simp [diffusionAlong, n2WitnessV, witnessDirection, dotProduct, Matrix.mulVec,
      Matrix.mul_apply, Fin.sum_univ_two]
  · simp [liftDiffusionAlong, pgdDiffusionAlong, diffusionAlong, n2WitnessV, witnessDirection,
      dotProduct, Matrix.mulVec, Matrix.add_apply, Matrix.one_apply,
      Fin.sum_univ_two]
    norm_num

/-! ## 2. Theorem A: the alignment threshold

The exponent is `BarrierRescaling.arrheniusExponent alpha eta x = 2 alpha / (eta x)`, strictly
decreasing in `x` on the positive reals; the comparison of the two arms is therefore the
comparison of their directional diffusions, and that comparison is a threshold on the excess
because the lift pays the positivity-map prefactor `s^2` on the whole of its diffusion while the
projected baseline pays it on none. -/

section TheoremA

variable {alpha eta s : ℝ} {Sobj C HVH : Matrix n n ℝ} {v : n → ℝ}

/-- The rearrangement at the heart of Theorem A: for a positive scale `S`, the attenuated total
`S (a + c)` exceeds the unattenuated baseline `a` exactly when the excess `c` clears the threshold
`(S⁻¹ - 1) a`. With `S = s^2` this is the gate. -/
theorem lt_mul_add_iff_inv_sub_one_mul_lt {S a c : ℝ} (hS : 0 < S) :
    a < S * (a + c) ↔ (S⁻¹ - 1) * a < c := by
  have heq : S * ((S⁻¹ - 1) * a) = a - S * a := by
    field_simp
  constructor
  · intro h
    have h2 : S * ((S⁻¹ - 1) * a) < S * c := by
      rw [heq]; nlinarith
    exact lt_of_mul_lt_mul_left h2 hS.le
  · intro h
    have h2 : S * ((S⁻¹ - 1) * a) < S * c := mul_lt_mul_of_pos_left h hS
    rw [heq] at h2
    nlinarith

/-- **The directional diffusion ordering is the gate.** The lift's diffusion along the escape
direction exceeds the projected baseline's exactly when the excess channels clear
`(s^{-2} - 1) v^T Sigma_obj v`. Nothing here is asymptotic and no sign hypothesis is used: this is
the algebraic content of the comparison, and it runs in both directions. -/
theorem pgdDiffusionAlong_lt_liftDiffusionAlong_iff (hs : 0 < s) :
    pgdDiffusionAlong v Sobj < liftDiffusionAlong s v Sobj C HVH
      ↔ ((s ^ 2)⁻¹ - 1) * diffusionAlong Sobj v < diffusionAlong (C + HVH) v := by
  have hS : (0 : ℝ) < s ^ 2 := by positivity
  rw [liftDiffusionAlong, add_assoc, diffusionAlong_add, pgdDiffusionAlong]
  exact lt_mul_add_iff_inv_sub_one_mul_lt hS

/-- The Arrhenius exponent orders strictly *against* the diffusion, in both directions. This is
`BarrierRescaling.arrheniusExponent_strictAntiOn` turned into the biconditional Theorem A needs;
the converse direction is where the positivity of both diffusions is consumed. -/
theorem arrheniusExponent_lt_iff_lt {x y : ℝ} (halpha : 0 < alpha) (heta : 0 < eta)
    (hx : 0 < x) (hy : 0 < y) :
    arrheniusExponent alpha eta y < arrheniusExponent alpha eta x ↔ x < y := by
  constructor
  · intro h
    rcases lt_trichotomy x y with hlt | heq | hgt
    · exact hlt
    · rw [heq] at h
      exact absurd h (lt_irrefl _)
    · exact absurd h (asymm (arrheniusExponent_strictAntiOn halpha heta hy hx hgt))
  · intro h
    exact arrheniusExponent_strictAntiOn halpha heta hx hy h

/-- **THEOREM A (directional escape comparison), explicit form.** With the barrier action and the
learning rate positive and both directional diffusions positive, the lift's Arrhenius exponent is
strictly below the projected-gradient baseline's **if and only if**

`v^T (C + H V H) v > (s^{-2} - 1) v^T Sigma_obj v`.

Both directions are proved. The theorem is a biconditional and holds whatever the empirical value
of the left-hand side turns out to be: it decides what a comparison of the two arms may claim, not
whether the arms differ. Note what is *not* asserted — the note's display
`log E[tau] = 2 alpha / (eta v^T Sigma v) + o(1/eta)` is an asymptotic statement about a mean exit
time, and neither the exit time nor the remainder appears here; this is a comparison of the two
exponents themselves. The passage to exit times is section 3 below, at the precision
`KramersExitTime.lean` proves and no better. -/
theorem arrheniusExponent_lift_lt_pgd_iff (halpha : 0 < alpha) (heta : 0 < eta) (hs : 0 < s)
    (hpgd : 0 < pgdDiffusionAlong v Sobj) (hlift : 0 < liftDiffusionAlong s v Sobj C HVH) :
    arrheniusExponent alpha eta (liftDiffusionAlong s v Sobj C HVH)
        < arrheniusExponent alpha eta (pgdDiffusionAlong v Sobj)
      ↔ ((s ^ 2)⁻¹ - 1) * diffusionAlong Sobj v < diffusionAlong (C + HVH) v :=
  (arrheniusExponent_lt_iff_lt halpha heta hpgd hlift).trans
    (pgdDiffusionAlong_lt_liftDiffusionAlong_iff hs)

/-! ### Theorem A net of the remainder channel

The appendix sentence this discharges, from the opening of `app:proof-pgd`:

> "`thm:directional` is stated at a vanishing remainder channel; the threshold of
> `app:pgd-account` reads the excess net of the remainder channel that (A3) folds into its
> inequality, which is the same arithmetic with the remainder added to the excess."

It is an identity and a biconditional, not an inequality: the remainder channel enters the
lift's directional diffusion additively (`liftDiffusionAlong_add_remainder`), so Theorem A read
with the remainder is Theorem A with `diffusionAlong R v` added to the excess side of the
threshold, and nothing else changes. No sign is assumed of `R`: the statement holds whichever way
the remainder channel points. -/

/-- **The remainder channel enters additively.** Reading the magnitude channel as `H V H + R`
adds exactly `s² vᵀ R v` to the lift's diffusion along the escape direction. -/
theorem liftDiffusionAlong_add_remainder (R : Matrix n n ℝ) :
    liftDiffusionAlong s v Sobj C (HVH + R)
      = liftDiffusionAlong s v Sobj C HVH + s ^ 2 * diffusionAlong R v := by
  have h : Sobj + C + (HVH + R) = Sobj + C + HVH + R := by abel
  rw [liftDiffusionAlong, h, diffusionAlong_add, liftDiffusionAlong]
  ring

/-- **THEOREM A net of the remainder channel.** The same biconditional as
`arrheniusExponent_lift_lt_pgd_iff`, with the remainder channel `R` carried: the threshold is
the same arithmetic with `vᵀ R v` added to the excess. Setting `R = 0` returns the theorem as
`thm:directional` states it. -/
theorem arrheniusExponent_lift_lt_pgd_iff_of_remainder {R : Matrix n n ℝ} (halpha : 0 < alpha)
    (heta : 0 < eta) (hs : 0 < s) (hpgd : 0 < pgdDiffusionAlong v Sobj)
    (hlift : 0 < liftDiffusionAlong s v Sobj C (HVH + R)) :
    arrheniusExponent alpha eta (liftDiffusionAlong s v Sobj C (HVH + R))
        < arrheniusExponent alpha eta (pgdDiffusionAlong v Sobj)
      ↔ ((s ^ 2)⁻¹ - 1) * diffusionAlong Sobj v
          < diffusionAlong (C + HVH) v + diffusionAlong R v := by
  have h := arrheniusExponent_lift_lt_pgd_iff (HVH := HVH + R) halpha heta hs hpgd hlift
  have h2 : C + (HVH + R) = C + HVH + R := by abel
  rw [h, h2, diffusionAlong_add]

end TheoremA

/-! ### The gate

The note states the threshold in the framework's own temperature ratio, evaluated along the escape
direction rather than as a trace. That ratio is `rho = (coupling + quadratic) / (s^2 sigma_obj^2)`
of `docs/design/lemma1_rederivation.md` (L3'), the relative excess of the lift's diffusion over the
direct baseline's; read along `v`, the factor `s^2` cancels between numerator and denominator and
the ratio is `v^T (C + H V H) v / v^T Sigma_obj v`. That cancellation is
`alignmentRatio_eq_excess_div_direct`, proved rather than asserted, and it is why the gate's bar is
`s^{-2} - 1` and not something with an `s^2` in it. -/

/-- **`rho_v`**, the framework's temperature ratio evaluated along the escape direction:
the excess channels' quadratic form over the objective noise's, both read along `v`. -/
noncomputable def alignmentRatio (v : n → ℝ) (Sobj C HVH : Matrix n n ℝ) : ℝ :=
  diffusionAlong (C + HVH) v / diffusionAlong Sobj v

/-- The gate's bar, `s^{-2} - 1`. At the note's typical interior constrained weight
`theta = 0.6`, where `s = psi'(psi^{-1}(0.6)) = 0.4512`, the bar is about `3.9`; the skeptic's
objection X4, that Adam moves the smooth arms' suppression from `s^2` to `s`, would move the bar to
`s^{-1} - 1`, about `1.2`. Which of the two is right is an empirical question about the optimizer
that this file does not address; the theorems below are stated for the `s^2` bookkeeping the paper
currently uses. -/
noncomputable def gateThreshold (s : ℝ) : ℝ := (s ^ 2)⁻¹ - 1

/-- **THE GATE**, `rho_v > s^{-2} - 1`. -/
def GateHolds (s : ℝ) (v : n → ℝ) (Sobj C HVH : Matrix n n ℝ) : Prop :=
  gateThreshold s < alignmentRatio v Sobj C HVH

variable {alpha eta s : ℝ} {Sobj C HVH : Matrix n n ℝ} {v : n → ℝ}

/-- The gate in cleared-denominator form, which is how it is checked against measured matrices. -/
theorem gateHolds_iff (ha : 0 < diffusionAlong Sobj v) :
    GateHolds s v Sobj C HVH
      ↔ ((s ^ 2)⁻¹ - 1) * diffusionAlong Sobj v < diffusionAlong (C + HVH) v := by
  rw [GateHolds, alignmentRatio, gateThreshold, lt_div_iff₀ ha]

/-- **THEOREM A in the note's own notation.** The lift's Arrhenius exponent is strictly below the
projected-gradient baseline's if and only if the gate `rho_v > s^{-2} - 1` holds. -/
theorem arrheniusExponent_lift_lt_pgd_iff_gate (halpha : 0 < alpha) (heta : 0 < eta) (hs : 0 < s)
    (hpgd : 0 < pgdDiffusionAlong v Sobj) (hlift : 0 < liftDiffusionAlong s v Sobj C HVH) :
    arrheniusExponent alpha eta (liftDiffusionAlong s v Sobj C HVH)
        < arrheniusExponent alpha eta (pgdDiffusionAlong v Sobj)
      ↔ GateHolds s v Sobj C HVH := by
  have ha : 0 < diffusionAlong Sobj v := hpgd
  rw [gateHolds_iff ha]
  exact arrheniusExponent_lift_lt_pgd_iff halpha heta hs hpgd hlift

/-- **The gate's numerator convention is the framework's temperature ratio.** `rho_v` is the
lift's directional excess over the *direct* baseline, divided by that baseline — the ratio
`(coupling + quadratic) / (s^2 sigma_obj^2)` of (L3') read along `v` — and the positivity-map prefactor
cancels, leaving a quantity that does not depend on `s` at all. This is the identity that makes
the note's two statements of the gate, the explicit one and the `rho_v` one, the same
statement. -/
theorem alignmentRatio_eq_excess_div_direct (hs : s ≠ 0) (ha : diffusionAlong Sobj v ≠ 0) :
    alignmentRatio v Sobj C HVH
      = (liftDiffusionAlong s v Sobj C HVH - directDiffusionAlong s v Sobj)
          / directDiffusionAlong s v Sobj := by
  have hs2 : s ^ 2 ≠ 0 := pow_ne_zero 2 hs
  rw [liftDiffusionAlong_eq, directDiffusionAlong, alignmentRatio]
  field_simp
  ring

/-- **The measurement recipe P0 is sufficient but not necessary.** Section 4 of the note computes
`v^T C v / v^T Sigma_obj v`, dropping the magnitude channel from the gate's numerator. Since
`v^T H V H v >= 0` always, P0's ratio never exceeds `rho_v`, so a P0 run that clears the bar
certifies the gate. -/
theorem gateHolds_of_couplingOnly_gate {V H : Matrix n n ℝ} (hV : V.PosSemidef)
    (hH : H.IsHermitian) (ha : 0 < diffusionAlong Sobj v)
    (hP0 : gateThreshold s < diffusionAlong C v / diffusionAlong Sobj v) :
    GateHolds s v Sobj C (H * V * H) := by
  rw [lt_div_iff₀ ha, gateThreshold] at hP0
  rw [gateHolds_iff ha, diffusionAlong_add]
  have hq : 0 ≤ diffusionAlong (H * V * H) v := quadForm_conj_nonneg hV hH v
  linarith

/-! ## 3. The escape-time consequence, at the precision the exit-time module proves

Theorem A compares exponents. Turning an exponent comparison into a comparison of mean exit times
needs the Arrhenius law, and `KramersExitTime.lean` proves it two-sided but with crude prefactors:
a window lower bound `c exp(2(alpha - delta)/sigma2)` and an upper bound
`C exp(2 alpha / sigma2)`, whose constants come from the moduli of continuity of the potential and
not from a Laplace expansion. Two regimes follow, and both are stated here with their conditions
in the statement.

The first is the prescribed pair: for two *given* diffusions the ordering follows from an explicit
inequality between the two proved bounds, which a reader can evaluate on the data. The second is
the small-noise regime, where no gap hypothesis is needed at all because the lower bound diverges;
there the two diffusions are the additively coupled family `sd` and `sd + surplus`, with the
surplus held fixed, which is the paper's own quantifier structure. Note the price of the second
form: it says the lift leads for every *sufficiently small* shared diffusion, and the measured
`v^T Sigma_obj v` need not be in that range. The first form is the one that certifies a prescribed
pair. -/

section ExitTime

variable {alpha eta s : ℝ} {Sobj C HVH : Matrix n n ℝ} {v : n → ℝ}

/-- The **gate surplus**, the amount by which the lift's directional diffusion exceeds the
projected-gradient baseline's. It is positive exactly when the gate holds, and it is the offset in
which the paper's additively coupled comparison is stated. -/
def gateSurplus (s : ℝ) (v : n → ℝ) (Sobj C HVH : Matrix n n ℝ) : ℝ :=
  liftDiffusionAlong s v Sobj C HVH - pgdDiffusionAlong v Sobj

/-- The surplus is positive exactly when the gate holds. -/
theorem gateSurplus_pos_iff_gateHolds (hs : 0 < s) (ha : 0 < diffusionAlong Sobj v) :
    0 < gateSurplus s v Sobj C HVH ↔ GateHolds s v Sobj C HVH := by
  rw [gateSurplus, sub_pos, gateHolds_iff ha]
  exact pgdDiffusionAlong_lt_liftDiffusionAlong_iff hs

/-- **The escape-time consequence for a prescribed pair of directional diffusions.** Under the
gate, the lift's Arrhenius exponent is strictly below the projected baseline's; and under the
explicit gap inequality `hgap` between the two bounds `KramersExitTime.lean` proves — the lift's
upper bound below the projected baseline's window lower bound — the lift's mean exit time is
strictly below the projected baseline's as well.

The two conclusions carry different weights and the theorem states both so that the difference is
visible. The exponent ordering is exactly Theorem A. The time ordering needs `hgap`, which is not
implied by the gate: the prefactors of the proved bracket are crude, and a sharp Kramers law with
its Eyring prefactor — which this mathlib cannot state, let alone prove — is what would be needed
to derive the time ordering from the diffusion ordering alone. `hgap` is an inequality between two
closed-form real numbers built from the data, so it is checkable; the price is that the caller must
exhibit the barrier window `[s₁, s₂]` on which `U >= cs` and the basin window `[t₁, t₂]` on which
`U <= cb`. -/
theorem meanExitTime_lift_lt_pgd_of_window {U : ℝ → ℝ} {a b x s₁ s₂ t₁ t₂ cs cb wb ws : ℝ}
    (halpha : 0 < alpha) (heta : 0 < eta) (hs : 0 < s)
    (hpgd : 0 < pgdDiffusionAlong v Sobj) (hlift : 0 < liftDiffusionAlong s v Sobj C HVH)
    (hgate : GateHolds s v Sobj C HVH)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x) (hxs₁ : x ≤ s₁) (hs₁₂ : s₁ ≤ s₂)
    (hs₂b : s₂ ≤ b) (hat₁ : a ≤ t₁) (ht₁₂ : t₁ ≤ t₂) (ht₂s₁ : t₂ ≤ s₁)
    (hbarrier : ∀ y ∈ Set.Icc s₁ s₂, cs ≤ U y)
    (hbasin : ∀ z ∈ Set.Icc t₁ t₂, U z ≤ cb)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z)
    (hgap : 2 / liftDiffusionAlong s v Sobj C HVH * ((b - x) * (b - a))
          * Real.exp (2 * (U ws - U wb) / liftDiffusionAlong s v Sobj C HVH)
        < 2 / pgdDiffusionAlong v Sobj * ((s₂ - s₁) * (t₂ - t₁))
          * Real.exp (2 * (cs - cb) / pgdDiffusionAlong v Sobj)) :
    arrheniusExponent alpha eta (liftDiffusionAlong s v Sobj C HVH)
        < arrheniusExponent alpha eta (pgdDiffusionAlong v Sobj)
      ∧ meanExitTime (liftDiffusionAlong s v Sobj C HVH) U a b x
          < meanExitTime (pgdDiffusionAlong v Sobj) U a b x := by
  refine ⟨(arrheniusExponent_lift_lt_pgd_iff_gate halpha heta hs hpgd hlift).mpr hgate, ?_⟩
  exact meanExitTime_lift_lt_direct_of_window hpgd hlift hU hax hxs₁ hs₁₂ hs₂b hat₁ ht₁₂ ht₂s₁
    hbarrier hbasin hmax hmin hgap

/-- **The escape-time consequence in the small-noise regime, with no gap hypothesis.** Hold the
gate surplus fixed — under the gate it is positive — and let the shared diffusion shrink: then for
every sufficiently small shared diffusion the exit time at the lift's value is strictly below the
exit time at the projected baseline's. This is `KramersExitTime.eventually_meanExitTime_additive_
lift_lt_direct` at the surplus the gate certifies, and it needs no relation between the two
Arrhenius prefactors.

Read the quantifier honestly. "Sufficiently small" is a filter statement with no threshold, and the
measured `v^T Sigma_obj v` is a fixed number that this theorem does not place inside the eventual
set. What the theorem establishes is that the gate's surplus is of the right kind — a fixed
additive offset in the diffusion, which is what the Arrhenius bracket converts into an exit-time
ordering — and not that the ordering holds at any particular measured diffusion. For a prescribed
pair use `meanExitTime_lift_lt_pgd_of_window`. -/
theorem eventually_meanExitTime_lift_lt_pgd_of_gate {U : ℝ → ℝ} {a b x wb ws : ℝ}
    (hs : 0 < s) (ha : 0 < diffusionAlong Sobj v) (hgate : GateHolds s v Sobj C HVH)
    (hU : ContinuousOn U (Set.Icc a b)) (hax : a ≤ x) (hawb : a < wb) (hwbws : wb < ws)
    (hwsb : ws < b) (hxws : x < ws) (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ),
      meanExitTime (sd + gateSurplus s v Sobj C HVH) U a b x < meanExitTime sd U a b x := by
  have hjac : 0 < gateSurplus s v Sobj C HVH := (gateSurplus_pos_iff_gateHolds hs ha).mpr hgate
  exact eventually_meanExitTime_additive_lift_lt_direct hjac hU hax hawb hwbws hwsb hxws
    hbarrier hmax hmin

/-- **The same for the mean first-passage time itself, under Dynkin's formula.** The statements
above are about `KramersExitTime.meanExitTime`, the closed-form solution of the exit-time
boundary-value problem. Identifying that closed form with the mean first-passage time of the
diffusion `dX = -U'(X) dt + sigma dB` is Dynkin's formula, which mathlib v4.31.0 cannot express —
it has no stochastic integral and no diffusion generator — and which is therefore carried, here as
in `KramersExitTime.lean`, as the named hypothesis `hDynkinD` / `hDynkinL`: the mean first-passage
time and its first two derivatives solve `MeanExitTimeBVP`. Everything after that hypothesis is
proved: the passage from the boundary-value problem to the closed form is
`eq_meanExitTime_of_bvp`, and the exponential bounds are proved outright. -/
theorem eventually_mean_first_passage_lift_lt_pgd_of_gate {U U' : ℝ → ℝ} {a b x wb ws : ℝ}
    (taud Dtaud DDtaud taul Dtaul DDtaul : ℝ → ℝ → ℝ)
    (hDynkinD : ∀ sd : ℝ, 0 < sd → MeanExitTimeBVP sd U' a b (taud sd) (Dtaud sd) (DDtaud sd))
    (hDynkinL : ∀ sd : ℝ, 0 < sd →
      MeanExitTimeBVP (sd + gateSurplus s v Sobj C HVH) U' a b (taul sd) (Dtaul sd) (DDtaul sd))
    (htaud : ∀ sd : ℝ, 0 < sd → ContinuousOn (taud sd) (Set.Icc a b))
    (htaul : ∀ sd : ℝ, 0 < sd → ContinuousOn (taul sd) (Set.Icc a b))
    (hs : 0 < s) (ha : 0 < diffusionAlong Sobj v) (hgate : GateHolds s v Sobj C HVH)
    (hU : ContinuousOn U (Set.Icc a b))
    (hU' : ∀ y ∈ Set.Ioo a b, HasDerivAt U (U' y) y) (hax : a ≤ x)
    (hawb : a < wb) (hwbws : wb < ws) (hwsb : ws < b) (hxws : x < ws)
    (hbarrier : U wb < U ws)
    (hmax : ∀ y ∈ Set.Icc a b, U y ≤ U ws) (hmin : ∀ z ∈ Set.Icc a b, U wb ≤ U z) :
    ∀ᶠ sd in 𝓝[>] (0 : ℝ), taul sd x < taud sd x := by
  have hjac : 0 < gateSurplus s v Sobj C HVH := (gateSurplus_pos_iff_gateHolds hs ha).mpr hgate
  exact eventually_mean_first_passage_additive_lift_lt_direct taud Dtaud DDtaud taul Dtaul DDtaul
    hDynkinD hDynkinL htaud htaul hjac hU hU' hax hawb hwbws hwsb hxws hbarrier hmax hmin

/-- **(FW-2) accepts the projected-gradient baseline.** The quasipotential estimate of
`QuasipotentialMonotone.lean` is stated at the reciprocal-gap level,
`delta <= 1/sigma_B^2 - 1/sigma_A^2`, which is a *directional* hypothesis and not a
positive-semidefinite domination. That distinction is exactly what this file needs: the domination
`Sigma_lift >= Sigma_PGD` is false
(`not_posSemidef_liftDiffusionMatrix_sub_pgdDiffusionMatrix`), and the estimate goes through
anyway. Here it is instantiated with the projected baseline as `sigma_B^2`: under the gate the
reciprocal gap is positive, and the lift's effective quasipotential increment across a climb of at
least `m` on a subinterval `[c, d]` is smaller by at least `(d - c) m delta`.

The two diffusions are constant in the band coordinate here, which is (A2)'s band idealization —
the same idealization `BarrierRescaling.lean` makes when it computes the barrier exactly on a
constant band and brackets it otherwise. What is genuinely used is only the reciprocal gap. -/
theorem quasipotentialIncrement_lift_le_pgd_of_gate {Lp : ℝ → ℝ} {a b c d m : ℝ}
    (hs : 0 < s) (ha : 0 < diffusionAlong Sobj v) (hgate : GateHolds s v Sobj C HVH)
    (hac : a ≤ c) (hcd : c ≤ d) (hdb : d ≤ b) (hm : 0 ≤ m)
    (hLp : ∀ u ∈ Set.Icc a b, 0 ≤ Lp u) (hLpm : ∀ u ∈ Set.Icc c d, m ≤ Lp u)
    (hi : IntervalIntegrable Lp MeasureTheory.volume a b) :
    quasipotentialIncrement Lp (fun _ => liftDiffusionAlong s v Sobj C HVH) a b
      ≤ quasipotentialIncrement Lp (fun _ => pgdDiffusionAlong v Sobj) a b
        - (d - c) * (m * (1 / pgdDiffusionAlong v Sobj
            - 1 / liftDiffusionAlong s v Sobj C HVH)) := by
  have hlt : pgdDiffusionAlong v Sobj < liftDiffusionAlong s v Sobj C HVH :=
    (pgdDiffusionAlong_lt_liftDiffusionAlong_iff hs).mpr ((gateHolds_iff ha).mp hgate)
  have hpgd : 0 < pgdDiffusionAlong v Sobj := ha
  have hlift : 0 < liftDiffusionAlong s v Sobj C HVH := hpgd.trans hlt
  have hdelta : 0 ≤ 1 / pgdDiffusionAlong v Sobj - 1 / liftDiffusionAlong s v Sobj C HVH :=
    sub_nonneg.mpr (one_div_le_one_div_of_le hpgd hlt.le)
  exact quasipotentialIncrement_le_sub_reciprocalGap hac hcd hdb hm hdelta hLp
    (fun _ _ => hpgd) (fun _ _ => hlt.le) hLpm (fun _ _ => le_rfl)
    (hi.div_const _) (hi.div_const _)

end ExitTime

/-! ## 4. Theorem C: alignment is structurally unavailable to the projected baseline

Theorem 1 (`thm:joint-necessity`) says that a nonzero slack-channel reading requires both a slack
channel and a batch-conditioned body. Projected gradient descent has neither: it optimizes the
constrained weights directly and projects, so its latent iterate is a function of the
conditioning batch that does not depend on it. `StructuralZeros.lean` records exactly this reading
— "PGD, having neither slack nor data-conditioned body, is covered by the same batch-independence
argument" — and derives from it that the window cross-covariance estimator vanishes identically,
for every iteration and every window length.

The consequence for the present file is Theorem C of the note: with `Sigma_slack = 0` the coupling
term `C = H Sigma_slack + Sigma_slack^T H` is identically zero, so the projected baseline's
directional diffusion is exactly `v^T Sigma_obj v` with no aligned component, its `rho_v` is zero,
and the gate fails for it at every direction and every objective covariance. Its only currency is
temperature, and temperature is directionless.

**This is the honest upgrade to Theorem 1's open converse, and it does not close it.** The
converse — that a structural zero of the estimator *implies* a loss — remains open, and nothing
below proves it. What is proved is the weaker and precise statement that the alignment route is
unavailable: a arm whose coupling term vanishes cannot clear a positive gate no matter how large
its temperature, because its `rho_v` is zero rather than small. Whether the alignment route is
what produces the measured margins is an empirical question the note's P0 measurement is designed
to answer, and it is not answered here. -/

section StructuralZero

/-- A vanishing slack-channel cross-covariance leaves no coupling term at all. -/
theorem couplingTerm_eq_zero (H : Matrix n n ℝ) : couplingTerm H 0 = 0 := by
  simp [couplingTerm]

omit [Fintype n] in
/-- With neither attenuation nor excess channels, the lift's diffusion matrix *is* the projected
baseline's: `Sigma_PGD = Sigma_obj` is the lift's own formula at `s = 1` and `C = 0`, which is the
note's section 3.1 display. -/
theorem liftDiffusionMatrix_eq_pgdDiffusionMatrix (Sobj : Matrix n n ℝ) :
    liftDiffusionMatrix 1 Sobj 0 0 = pgdDiffusionMatrix Sobj := by
  simp [liftDiffusionMatrix, pgdDiffusionMatrix]

/-- The same reading along the escape direction: with no coupling and no magnitude channel the
directional diffusion is exactly `v^T Sigma_obj v`. -/
theorem liftDiffusionAlong_eq_pgdDiffusionAlong (v : n → ℝ) (Sobj : Matrix n n ℝ) :
    liftDiffusionAlong 1 v Sobj 0 0 = pgdDiffusionAlong v Sobj := by
  rw [← diffusionAlong_liftDiffusionMatrix, liftDiffusionMatrix_eq_pgdDiffusionMatrix,
    diffusionAlong_pgdDiffusionMatrix]

/-- An arm with no coupling and no magnitude channel has `rho_v = 0` in every direction: not a
small aligned component, but none. -/
theorem alignmentRatio_eq_zero (v : n → ℝ) (Sobj : Matrix n n ℝ) :
    alignmentRatio v Sobj 0 0 = 0 := by
  simp [alignmentRatio, diffusionAlong]

/-- Hence the gate fails for such an arm identically: at `s = 1` the bar is `0` and the ratio is
`0`, and the inequality is strict. -/
theorem not_gateHolds_of_no_coupling (v : n → ℝ) (Sobj : Matrix n n ℝ) :
    ¬ GateHolds 1 v Sobj 0 0 := by
  rw [GateHolds, gateThreshold, alignmentRatio_eq_zero]
  norm_num

/-- **THEOREM C.** For an arm whose latent iterate carries no batch dependence — projected
gradient descent, which has neither a slack channel nor a data-conditioned body — the window
cross-covariance estimator of `eq:cross-cov` vanishes for every iteration and every window length
(`StructuralZeros.crossCovEstimator_eq_zero_of_batchIndependent`), hence the coupling term
vanishes, hence its diffusion matrix is exactly `Sigma_obj`, its directional diffusion is exactly
`v^T Sigma_obj v`, its `rho_v` is zero, and the gate fails for it.

The hypothesis is the architecture, not the conclusion: `BatchIndependent` says the latent weight map
is constant in the conditioning batch, and the vanishing of the fluctuation is derived from that.
As stated in the section docstring, this does not close Theorem 1's converse. -/
theorem pgd_alignment_structurally_unavailable {d : ℕ} {Ω : Type*} {emission : Ω → Fin d → ℝ}
    (hbatch : BatchIndependent emission) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0)
    (H Sobj : Matrix (Fin d) (Fin d) ℝ) (v : Fin d → ℝ)
    (Sslack : Matrix (Fin d) (Fin d) ℝ)
    (hSslack : Sslack = crossCovEstimator T t (bodyFluct emission X T t) dg) :
    couplingTerm H Sslack = 0 ∧
      liftDiffusionMatrix 1 Sobj (couplingTerm H Sslack) 0 = pgdDiffusionMatrix Sobj ∧
      liftDiffusionAlong 1 v Sobj (couplingTerm H Sslack) 0 = pgdDiffusionAlong v Sobj ∧
      alignmentRatio v Sobj (couplingTerm H Sslack) 0 = 0 ∧
      ¬ GateHolds 1 v Sobj (couplingTerm H Sslack) 0 := by
  have hzero : Sslack = 0 := by
    rw [hSslack]
    exact crossCovEstimator_eq_zero_of_batchIndependent hbatch X dg T t hT
  have hC : couplingTerm H Sslack = 0 := by
    rw [hzero]
    exact couplingTerm_eq_zero H
  rw [hC]
  exact ⟨rfl, liftDiffusionMatrix_eq_pgdDiffusionMatrix Sobj,
    liftDiffusionAlong_eq_pgdDiffusionAlong v Sobj, alignmentRatio_eq_zero v Sobj,
    not_gateHolds_of_no_coupling v Sobj⟩

end StructuralZero

/-! ## 5. Non-vacuity, and that the gate discriminates

Theorem A is a biconditional, so it is worthless in both directions if no instance clears the gate
and worthless in one direction if every instance does. Both are checked here on explicit `2 x 2`
instances, by `norm_num`.

The clearing instance is deliberately not adversarial. Its slack cross-covariance is
`Sigma_slack = V H` with `V = I`, that is, the **exact** forward-KL chain of (A4) under which
`CouplingSign.crossTerm_nonneg_of_exact_chain` proves the coupling channel's sign rather than
assuming it; only then is the gate a statement about magnitudes and not about a hand-chosen sign.
The instance also carries the note's section 3.2 point, which is the reason the trace is the wrong
monitor: it clears the gate — so by Theorem A it escapes strictly faster — while carrying **less
total noise** than the projected baseline, `tr Sigma_lift = 6 < 9 = tr Sigma_PGD`. And the matrix
domination `Sigma_lift >= Sigma_PGD` is *false* on it, which is why the chain's (FW-2) is stated at
the reciprocal-gap level in `QuasipotentialMonotone.lean` and not as a positive-semidefinite
ordering. -/

section Witness

/-- The objective noise covariance of the witness, `diag(1, 8)`: the projected baseline spreads its
temperature over both coordinates while the escape direction is the first. -/
def gateWitnessObj : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 8]

/-- The mean Hessian of the witness, `diag(2, 1)`, symmetric and positive definite. -/
def gateWitnessHessian : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]

/-- The witness's coupling term `C = H Sigma_slack + Sigma_slack^T H`, at the slack
cross-covariance `Sigma_slack = V H` of the exact (A4) chain with `V = I`. -/
def gateWitnessCoupling : Matrix (Fin 2) (Fin 2) ℝ :=
  couplingTerm gateWitnessHessian gateWitnessHessian

/-- The witness's magnitude term `H V H`, again at `V = I`. -/
def gateWitnessMagnitude : Matrix (Fin 2) (Fin 2) ℝ :=
  gateWitnessHessian * 1 * gateWitnessHessian

/-- The witness's objective covariance is positive definite. -/
theorem gateWitnessObj_posDef : gateWitnessObj.PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · rw [gateWitnessObj, Matrix.IsHermitian]
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  · intro x hx
    have hx' : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
      by_contra hcon
      simp only [not_or, ne_eq, not_not] at hcon
      exact hx (by ext i; fin_cases i <;> simp [hcon.1, hcon.2])
    have hval : star x ⬝ᵥ (gateWitnessObj *ᵥ x) = x 0 ^ 2 + 8 * x 1 ^ 2 := by
      simp [gateWitnessObj, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
      ring
    rw [hval]
    rcases hx' with h0 | h1
    · positivity
    · positivity

/-- The witness's mean Hessian is symmetric, which is what the coupling term's construction and
`TraceLemmas.quadForm_conj_nonneg` both require. -/
theorem gateWitnessHessian_isHermitian : gateWitnessHessian.IsHermitian := by
  rw [gateWitnessHessian, Matrix.IsHermitian]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

/-- The witness's slack cross-covariance really is the exact chain's `V H`, at `V = I`. -/
theorem gateWitnessSlack_eq_exact_chain :
    gateWitnessHessian = (1 : Matrix (Fin 2) (Fin 2) ℝ) * gateWitnessHessian := (one_mul _).symm

/-- Hence the witness's coupling reading is `2 v^T H V H v`, the sign-free form of the exact
chain: it is not a hand-chosen positive number but the value `CouplingSign` derives from (A4). -/
theorem gateWitness_crossTerm_eq :
    crossTerm gateWitnessHessian gateWitnessHessian witnessDirection
      = 2 * (witnessDirection ⬝ᵥ (gateWitnessMagnitude *ᵥ witnessDirection)) :=
  crossTerm_eq_two_mul_conj_of_exact_chain gateWitnessHessian_isHermitian
    gateWitnessSlack_eq_exact_chain witnessDirection

/-- The witness's coupling term, computed. -/
theorem gateWitnessCoupling_eq : gateWitnessCoupling = !![8, 0; 0, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [gateWitnessCoupling, couplingTerm, gateWitnessHessian, Matrix.mul_apply,
      Matrix.add_apply, Matrix.transpose_apply, Fin.sum_univ_two]

/-- The witness's magnitude term, computed. -/
theorem gateWitnessMagnitude_eq : gateWitnessMagnitude = !![4, 0; 0, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [gateWitnessMagnitude, gateWitnessHessian, Matrix.mul_apply, Matrix.one_apply,
      Fin.sum_univ_two]

/-- The objective noise along the escape direction is `1`. -/
theorem gateWitness_diffusionAlong_obj :
    diffusionAlong gateWitnessObj witnessDirection = 1 := by
  simp [diffusionAlong, gateWitnessObj, witnessDirection, dotProduct, Matrix.mulVec,
    Fin.sum_univ_two]

/-- The excess channels along the escape direction sum to `12`: eight from the coupling term and
four from the magnitude term. -/
theorem gateWitness_diffusionAlong_excess :
    diffusionAlong (gateWitnessCoupling + gateWitnessMagnitude) witnessDirection = 12 := by
  rw [gateWitnessCoupling_eq, gateWitnessMagnitude_eq]
  simp [diffusionAlong, witnessDirection, dotProduct, Matrix.mulVec, Matrix.add_apply,
    Fin.sum_univ_two]
  norm_num

/-- **The gate is not vacuous.** At the attenuation `s = 1/2`, where the bar is `s^{-2} - 1 = 3`,
the witness's ratio is `rho_v = 12`. -/
theorem gateHolds_witness :
    GateHolds (1 / 2 : ℝ) witnessDirection gateWitnessObj gateWitnessCoupling
      gateWitnessMagnitude := by
  rw [GateHolds, gateThreshold, alignmentRatio, gateWitness_diffusionAlong_excess,
    gateWitness_diffusionAlong_obj]
  norm_num

/-- The witness's directional diffusions: the lift reads `13/4`, the projected baseline reads `1`.
So the lift explores the escape direction more than three times as strongly, on strictly less total
noise (`gateWitness_trace_lt`). -/
theorem gateWitness_diffusionAlong_values :
    liftDiffusionAlong (1 / 2 : ℝ) witnessDirection gateWitnessObj gateWitnessCoupling
        gateWitnessMagnitude = 13 / 4
      ∧ pgdDiffusionAlong witnessDirection gateWitnessObj = 1 := by
  refine ⟨?_, gateWitness_diffusionAlong_obj⟩
  rw [liftDiffusionAlong_eq, directDiffusionAlong, gateWitness_diffusionAlong_obj,
    gateWitness_diffusionAlong_excess]
  norm_num

/-- **The escape comparison on the witness.** By Theorem A, the lift's Arrhenius exponent is
strictly below the projected-gradient baseline's, for every positive barrier action and every
positive learning rate. -/
theorem gateWitness_arrheniusExponent_lift_lt_pgd {alpha eta : ℝ} (halpha : 0 < alpha)
    (heta : 0 < eta) :
    arrheniusExponent alpha eta
        (liftDiffusionAlong (1 / 2 : ℝ) witnessDirection gateWitnessObj gateWitnessCoupling
          gateWitnessMagnitude)
      < arrheniusExponent alpha eta (pgdDiffusionAlong witnessDirection gateWitnessObj) := by
  obtain ⟨hlift, hpgd⟩ := gateWitness_diffusionAlong_values
  refine (arrheniusExponent_lift_lt_pgd_iff_gate halpha heta (by norm_num) ?_ ?_).mpr
    gateHolds_witness
  · rw [hpgd]; norm_num
  · rw [hlift]; norm_num

/-- The witness's lift diffusion matrix, computed. -/
theorem gateWitnessLiftMatrix_eq :
    liftDiffusionMatrix (1 / 2 : ℝ) gateWitnessObj gateWitnessCoupling gateWitnessMagnitude
      = !![13 / 4, 0; 0, 11 / 4] := by
  rw [liftDiffusionMatrix, gateWitnessCoupling_eq, gateWitnessMagnitude_eq, gateWitnessObj]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [Matrix.smul_apply, Matrix.add_apply]

/-- **Smaller total noise, faster escape.** The witness clears the gate — so by
`gateWitness_arrheniusExponent_lift_lt_pgd` it escapes strictly faster — while its diffusion has
**strictly smaller trace** than the projected baseline's, `6 < 9`. This is the note's section 3.2
point in machine-checked form: `sigma_Jac^2` read as a trace is blind to the property that is the
mechanism, because the mechanism is a direction and not a magnitude. -/
theorem gateWitness_trace_lt :
    (liftDiffusionMatrix (1 / 2 : ℝ) gateWitnessObj gateWitnessCoupling gateWitnessMagnitude).trace
      < (pgdDiffusionMatrix gateWitnessObj).trace := by
  rw [gateWitnessLiftMatrix_eq, pgdDiffusionMatrix, gateWitnessObj, Matrix.trace_fin_two_of,
    Matrix.trace_fin_two_of]
  norm_num

/-- **The matrix domination `Sigma_lift >= Sigma_PGD` is false, even where the gate holds.** On the
witness the difference is negative in the second coordinate, so it is not positive semidefinite.
This is why the chain's (FW-2) is stated at the reciprocal-gap level
`u^T (Sigma_B^{-1} - Sigma_A^{-1}) u >= delta ||u||^2` in `QuasipotentialMonotone.lean`, which is a
directional hypothesis and therefore accepts the projected-gradient baseline, rather than as a
positive-semidefinite ordering, which it would not. -/
theorem not_posSemidef_liftDiffusionMatrix_sub_pgdDiffusionMatrix :
    ¬ (liftDiffusionMatrix (1 / 2 : ℝ) gateWitnessObj gateWitnessCoupling gateWitnessMagnitude
        - pgdDiffusionMatrix gateWitnessObj).PosSemidef := by
  intro h
  have h1 := (Matrix.posSemidef_iff_dotProduct_mulVec.mp h).2 (![0, 1] : Fin 2 → ℝ)
  rw [gateWitnessLiftMatrix_eq, pgdDiffusionMatrix, gateWitnessObj] at h1
  simp [dotProduct, Matrix.mulVec, Matrix.sub_apply, Fin.sum_univ_two] at h1
  norm_num at h1

/-- **The gate discriminates, and it discriminates on the attenuation.** The same positively
aligned configuration — the identity Hessian, the rank-one jitter covariance `diag(1, 0)`, and the
exact chain — clears the gate at `s = 3/5` and fails it at `s = 1/3`, because the bar
`s^{-2} - 1` moves from `16/9` to `8` while the ratio stays at `rho_v = 3`. Positive alignment,
which is all that (A6) and the note's (A7) assert, is therefore **not sufficient**: the lift must
clear a threshold set by how hard the positivity map attenuates, and that is the whole content of the
recommendation. -/
theorem gate_discriminates_on_slope :
    GateHolds (3 / 5 : ℝ) witnessDirection gateWitnessObj (couplingTerm 1 n2WitnessV)
        (1 * n2WitnessV * 1)
      ∧ ¬ GateHolds (1 / 3 : ℝ) witnessDirection gateWitnessObj (couplingTerm 1 n2WitnessV)
        (1 * n2WitnessV * 1) := by
  have htrans : n2WitnessVᵀ = n2WitnessV :=
    transpose_eq_of_isHermitian n2WitnessV_posSemidef.isHermitian
  have hsum : couplingTerm 1 n2WitnessV + 1 * n2WitnessV * 1 = !![3, 0; 0, 0] := by
    simp only [couplingTerm, htrans, one_mul, mul_one]
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [n2WitnessV, Matrix.add_apply]
  have hexcess :
      diffusionAlong (couplingTerm 1 n2WitnessV + 1 * n2WitnessV * 1) witnessDirection = 3 := by
    rw [hsum]
    simp [diffusionAlong, witnessDirection, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
  constructor
  · rw [GateHolds, gateThreshold, alignmentRatio, hexcess, gateWitness_diffusionAlong_obj]
    norm_num
  · rw [GateHolds, gateThreshold, alignmentRatio, hexcess, gateWitness_diffusionAlong_obj]
    norm_num

/-- **The measurement recipe P0 is not necessary, and the gap is exhibited.** Section 4 of the note
computes `v^T C v / v^T Sigma_obj v`, dropping the magnitude channel. Here is a configuration in
which that ratio is zero — the coupling term vanishes outright — while the gate itself holds,
carried entirely by the magnitude channel. A P0 run below the bar therefore does not close the
question; it closes it only together with a measurement of the magnitude channel along the same
direction.

This is a satisfiability statement about the recipe and not a claim about the measured regime: the
note's own section 11.0 reports the magnitude channel inert at the operating point across a
hundredfold variance bracket, in which case the two recipes nearly agree. -/
theorem exists_gateHolds_of_couplingOnly_gate_failure :
    ∃ (Sobj V H : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) (s : ℝ),
      Sobj.PosDef ∧ V.PosSemidef ∧ H.IsHermitian ∧ 0 < s ∧ s < 1 ∧
        ¬ gateThreshold s
            < diffusionAlong (0 : Matrix (Fin 2) (Fin 2) ℝ) v / diffusionAlong Sobj v ∧
          GateHolds s v Sobj 0 (H * V * H) := by
  refine ⟨gateWitnessObj, 1, gateWitnessHessian, witnessDirection, 1 / 2, gateWitnessObj_posDef,
    Matrix.PosDef.one.posSemidef, gateWitnessHessian_isHermitian, by norm_num, by norm_num, ?_, ?_⟩
  · rw [gateThreshold, gateWitness_diffusionAlong_obj]
    simp [diffusionAlong]
  · rw [GateHolds, gateThreshold, alignmentRatio, gateWitness_diffusionAlong_obj]
    have hmag : diffusionAlong (0 + gateWitnessHessian * 1 * gateWitnessHessian) witnessDirection
        = 4 := by
      rw [zero_add, ← gateWitnessMagnitude, gateWitnessMagnitude_eq]
      simp [diffusionAlong, witnessDirection, dotProduct, Matrix.mulVec, Fin.sum_univ_two]
    rw [hmag]
    norm_num

end Witness

/-! ## 5. Adam: the variance still orders the exponent

The appendix sentence this discharges, from `rem:adam`:

> "The comparison of `app:proof-pgd` is unchanged, because the map `V ↦ (√V + ε)/V` through
> which the variance enters the exponent is strictly decreasing and common to the constructions,
> so any inequality between two exponents is equivalent to the reverse inequality between the two
> variances."

Two things are separated here, and the separation is the point.

**The modelling step stays a hypothesis.** That Adam's escape exponent *is*
`2α/η · (√V + ε)/V` rests on reading the second-moment estimate as constant across the band —
the quasi-static reading — and on the normalization acting identically on the two constructions.
Neither is derived anywhere in this development, and neither can be: they are statements about a
training algorithm, not about a diffusion. They are carried by the named hypothesis
`AdamQuasiStatic`, in the same register as the Dynkin inputs of `KramersExitTime` and
`ShoulderDwell`, so that the gap is visible at every use site rather than built into a
definition.

**The mathematics is a theorem.** Given that reading, the ordering conclusion is not a further
assumption: `adamVarianceMap_strictAntiOn` proves that `V ↦ (√V + ε)/V` is strictly decreasing on
the positive reals for every `ε ≥ 0`, and `adam_exponent_lt_iff` turns that into the
biconditional the remark states. The two constructions enter only through their variances, so the
map being *common* to them is what makes the equivalence an equivalence rather than an
implication.

The two regimes the remark names are both visible in `adamVarianceMap_eq_add`, which writes the
map as `V^{-1/2} + ε V⁻¹`: at `ε = 0` it is the square-rooted reading
(`adamVarianceMap_zero`), and the `ε`-dominant regime is the second summand, the original form
with the learning rate rescaled. -/

section Adam

/-- The map through which the gradient variance enters the escape exponent under Adam's
per-coordinate normalization, `V ↦ (√V + ε)/V`, with `ε` the stabilizing constant. -/
noncomputable def adamVarianceMap (eps V : ℝ) : ℝ := (Real.sqrt V + eps) / V

/-- The map splits into the square-rooted reading and the `ε`-dominant one:
`(√V + ε)/V = V^{-1/2} + ε V⁻¹`. -/
theorem adamVarianceMap_eq_add (eps : ℝ) {V : ℝ} (hV : 0 < V) :
    adamVarianceMap eps V = (Real.sqrt V)⁻¹ + eps * V⁻¹ := by
  have hspos : (0 : ℝ) < Real.sqrt V := Real.sqrt_pos.mpr hV
  have hs : Real.sqrt V * Real.sqrt V = V := Real.mul_self_sqrt hV.le
  have h1 : (Real.sqrt V)⁻¹ = Real.sqrt V / V := by
    rw [eq_div_iff hV.ne']
    nth_rewrite 2 [← hs]
    rw [← mul_assoc, inv_mul_cancel₀ hspos.ne', one_mul]
  rw [adamVarianceMap, add_div, h1, mul_comm eps V⁻¹, ← div_eq_inv_mul]

/-- At a vanishing stabilizing constant the map is `V^{-1/2}`: the gradient variance replaced by
its square root, which is the remark's first regime. -/
theorem adamVarianceMap_zero {V : ℝ} (hV : 0 < V) :
    adamVarianceMap 0 V = (Real.sqrt V)⁻¹ := by
  rw [adamVarianceMap_eq_add 0 hV, zero_mul, add_zero]

/-- **The map is strictly decreasing on the positive reals**, for every non-negative stabilizing
constant. This is the analytic content of `rem:adam`'s sentence, and the only thing in it that is
a theorem rather than a modelling step. -/
theorem adamVarianceMap_strictAntiOn {eps : ℝ} (heps : 0 ≤ eps) :
    StrictAntiOn (adamVarianceMap eps) (Set.Ioi (0 : ℝ)) := by
  intro u hu v hv huv
  have hu0 : (0 : ℝ) < u := hu
  have hv0 : (0 : ℝ) < v := hv
  have ha0 : 0 < Real.sqrt u := Real.sqrt_pos.mpr hu0
  have hb0 : 0 < Real.sqrt v := Real.sqrt_pos.mpr hv0
  have ha : Real.sqrt u * Real.sqrt u = u := Real.mul_self_sqrt hu0.le
  have hb : Real.sqrt v * Real.sqrt v = v := Real.mul_self_sqrt hv0.le
  have hab : Real.sqrt u < Real.sqrt v := Real.sqrt_lt_sqrt hu0.le huv
  rw [adamVarianceMap, adamVarianceMap, div_lt_div_iff₀ hv0 hu0]
  have h1 : (Real.sqrt v + eps) * u
      = Real.sqrt u * Real.sqrt u * Real.sqrt v + eps * (Real.sqrt u * Real.sqrt u) := by
    rw [ha]; ring
  have h2 : (Real.sqrt u + eps) * v
      = Real.sqrt u * (Real.sqrt v * Real.sqrt v) + eps * (Real.sqrt v * Real.sqrt v) := by
    rw [hb]; ring
  have h3 : Real.sqrt u * Real.sqrt u * Real.sqrt v
      < Real.sqrt u * (Real.sqrt v * Real.sqrt v) := by
    nlinarith [mul_pos ha0 hb0]
  have h4 : eps * (Real.sqrt u * Real.sqrt u) ≤ eps * (Real.sqrt v * Real.sqrt v) := by
    have hle : Real.sqrt u * Real.sqrt u ≤ Real.sqrt v * Real.sqrt v := by nlinarith
    exact mul_le_mul_of_nonneg_left hle heps
  rw [h1, h2]
  linarith

/-- **The quasi-static reading of Adam's second-moment estimate**, carried as a named hypothesis
and proved nowhere: that the escape exponent of a construction whose gradient variance along the
escape direction is `V` equals `2α/η · (√V + ε)/V`. This is the modelling step `rem:adam` calls
"stated and not derived"; it packages the constancy of the second-moment estimate across the band
and the identical action of the normalization on the two constructions. Discharging it would need
a model of the Adam recursion, which this development does not have. -/
structure AdamQuasiStatic (alpha eta eps : ℝ) (E : ℝ → ℝ) : Prop where
  /-- The exponent at directional variance `V` is `2α/η` times the Adam variance map. -/
  exponent_eq : ∀ V : ℝ, 0 < V → E V = 2 * alpha / eta * adamVarianceMap eps V

/-- **The comparison of `app:proof-pgd` is unchanged under Adam.** Given the quasi-static
reading, an inequality between the two exponents is equivalent to the reverse inequality between
the two directional variances — which is exactly what Theorem A consumes, so every conclusion of
`app:proof-pgd` that is an ordering of exponents transports. The two constructions enter only
through `V`, the map being common to them. -/
theorem adam_exponent_lt_iff {alpha eta eps : ℝ} {E : ℝ → ℝ} (halpha : 0 < alpha)
    (heta : 0 < eta) (heps : 0 ≤ eps) (hq : AdamQuasiStatic alpha eta eps E)
    {V₁ V₂ : ℝ} (h₁ : 0 < V₁) (h₂ : 0 < V₂) :
    E V₁ < E V₂ ↔ V₂ < V₁ := by
  have hc : (0 : ℝ) < 2 * alpha / eta := by positivity
  rw [hq.exponent_eq V₁ h₁, hq.exponent_eq V₂ h₂, mul_lt_mul_iff_of_pos_left hc]
  exact (adamVarianceMap_strictAntiOn heps).lt_iff_gt (Set.mem_Ioi.mpr h₁) (Set.mem_Ioi.mpr h₂)

/-- The hypothesis is inhabited: the map itself, scaled, satisfies it. This keeps
`adam_exponent_lt_iff` from being vacuous. -/
theorem adamQuasiStatic_satisfiable (alpha eta eps : ℝ) :
    AdamQuasiStatic alpha eta eps (fun V => 2 * alpha / eta * adamVarianceMap eps V) :=
  ⟨fun _ _ => rfl⟩

end Adam

end IcnnLift
