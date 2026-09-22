import Mathlib
import TraceLemmas

/-!
# FW-3b: the slack channel is attenuated once, not zero times

The paper's mechanism section characterizes the shoulder by two scalars. The
**objective channel** is the variance of an already attenuated gradient: with
`σ_obj² ≡ tr Var_X[∇_θ L]` taken on the *constrained* coordinate, before the positivity-map prefactor
is applied, the objective noise entering the bias-channel diffusion is `s² σ_obj²`, where
`s = ψ'(θ̃)` is the shoulder prefactor. The **slack channel** is the trace of the
cross-covariance between the latent iterate jitter and the gradient fluctuation,
`σ_Jac² ≡ tr E_X[δθ̃ δgᵀ]` (`eq:cross-cov`). The shipped text calls the second of these
*unattenuated*, or *unmodulated* — that is, `Θ(1)` in the prefactor — and the authors' own
design note settles that this is wrong
(`docs/design/lemma1_rederivation.md` §11.2, "FW-3b — the power-of-s discrepancy"). The chain
rule `g = ψ'(θ̃) ⊙ ∇_θ L` already carries one factor of the prefactor — under the single-prefactor
idealization (A2) the fluctuation is `δg = s · δ(∇_θ L)`, and because `ψ'(θ̃)` is itself
batch-dependent that identity is an idealization whose defect is carried explicitly in the later
sections rather than dropped — while `δθ̃`, a re-evaluation of the latent weight rather than a gradient
step, carries none. Exactly one of the two slots of the bilinear expression `tr E_X[δθ̃ δgᵀ]`
passes through `ψ'`, so the slack channel is attenuated **once** where the objective channel is
attenuated **twice**: `σ_Jac² = Θ(s)`, not `Θ(1)` and not `Θ(s²)`.

This file machine-checks that count and its consequences. The counting is a theorem, not a
stipulation: the channels are *defined* from the chain rule as the two bilinear pairings
`tr E_X[δg δgᵀ]` and `tr E_X[δθ̃ δgᵀ]` of the fluctuation fields, the pairing is proved
homogeneous of degree one in each slot separately (`traceCrossMoment_chainRuleGrad_left`,
`traceCrossMoment_chainRuleGrad_right`), and the two powers then fall out of how many slots the
prefactor enters (`objChannel_eq`, `jacChannel_eq`). Under an explicit non-degeneracy
hypothesis on the prefactor-free coupling functional `q ≡ tr E_X[δθ̃ δ(∇_θ L)ᵀ]` — namely
`0 < q₁ ≤ q ≤ q₂` uniformly on the shoulder band — the count becomes a genuine two-sided bound
`q₁ s ≤ σ_Jac² ≤ q₂ s` with explicit positive constants, hence `σ_Jac² =Θ[𝓝[>] 0] s`.

The consequence that matters for the paper is that the correction is a wording fix and not a
substantive retraction. The dimensionless control `ϱ = σ_Jac²/(s² σ_obj²)` of the mechanism
section is proved to be `Θ(1/s)` and to diverge as the positivity map flattens
(`couplingRatio_isTheta_inv`, `couplingRatio_tendsto_atTop`), so every downstream conclusion
that needs only `ϱ → ∞` survives at the corrected power. The corrected power is in fact the
stronger reading: the shoulder penalty carried by the lift's effective diffusion is `Θ(1/s)`
while the direct baseline's is `Θ(1/s²)` (`liftExponent_isTheta_inv`,
`directExponent_isTheta_inv_sq`), and the ratio of the two exponents is exactly `1 + ϱ`, which
diverges (`exponentRatio_eq`, `exponentRatio_tendsto_atTop`). The contrast that explains the
slip is formalized side by side, and it is set up so as not to assume its own conclusion. The
latent iterate is `θ̃_s = b_s + h_φ(X)` with the slack *allowed* to move with the prefactor,
since different prefactor values are different locations on the shoulder and it is the slack that
puts the iterate there; only the body is required to be prefactor-free, and that is a statement
about the architecture, namely that the body is evaluated before `ψ` is applied. What then makes
`E_X‖δθ̃‖²` constant in `s` is the slack cancellation of `eq:lift-batch-fluct`, not the shape of
the hypothesis: the iterate itself does depend on the prefactor and only its fluctuation does
not. Both halves of `unattenuated_jitter_but_once_attenuated_channel` are stated about the same
field `θ̃_s`. The word is true of the jitter and false of the channel.

The single-prefactor idealization (A2) is **not** assumed away. The true coordinatewise
prefactor field `p = ψ'(θ̃)` is carried, the exact chain rule `positivityMapGrad p u` is decomposed as
the (A2) idealization plus its defect with no approximation
(`positivityMapGrad_eq_chainRuleGrad_add_defect`), the resulting remainder appears explicitly in the
conclusion (`jacChannelExact_split`), and it is bounded from (A2)'s own pointwise defect
`|ψ'(θ̃)_i - s| ≤ ε` in `abs_remainder_le_of_prefactor_defect`. Carrying that remainder into the
*asymptotic* statement, rather than only into a bound at one prefactor value, requires saying
what becomes of the prefactor field as the shoulder flattens, so the honest object there is a
family: `jacChannelExactFam` is the exact channel at prefactor level `s`, computed from the true
field `p s`. Two things about it have to be said plainly. First, (A2) must be read at *relative*
accuracy, `|ψ'(θ̃)_i - s| ≤ ε s`, and not at an absolute accuracy with `ε` held fixed as `s`
falls: an absolute defect contributes a remainder of constant order, which would swamp a `Θ(s)`
signal, and no asymptotic conclusion of this file would survive it. Second, the resulting
remainder constant `ε · ∑_i E_X|δθ̃_i (∇_θL)_i|` must be strictly smaller than the non-degeneracy
constant `q₁`. Under exactly those two readings and nothing more,
`exact_channel_corrected_power` delivers `Θ(s)`, `Θ(1/s)` for `ϱ`, and the divergence, for the
exact channel, with no idealization anywhere in the statement and the remainder never assumed to
vanish.

A note on the word "once", to forestall a mechanical mis-edit of the tex. This file counts powers
of the prefactor in the **channels**, which are second moments: the objective channel `s² σ_obj²`
carries two and the slack channel `s q` carries one. The shipped text's regime-scalar paragraph
uses the same adjective for the count on the **gradient**, calling `s² σ_obj²` the
"once-attenuated objective noise" because `∇_{θ̃}L` passes through `ψ'` exactly once before the
variance squares it. The two descriptions name the same quantity and differ only in what is being
counted, so that sentence is correct as written; the defect to be corrected is confined to
`σ_Jac²`.

The hypotheses are exhibited to be jointly satisfiable rather than merely consistent-looking.
`witMeasure` is a two-point batch law on one constrained coordinate for which `q = 2`,
`σ_obj² = 4` and `E_X‖δθ̃‖² = 1`; its constrained gradient has non-zero mean, so centring does
real work, its slack moves with the prefactor (`witIterate_depends_on_prefactor`), and its
positivity-map prefactor field deviates from the scalar `s` by a tenth at every batch, so the (A2)
remainder is genuinely non-zero there. `power_of_s_correction`,
`unattenuated_jitter_but_once_attenuated_channel`, `couplingCore_pos_of_forwardKL` and
`exact_channel_corrected_power` are each instantiated at it, so none of the four is vacuously
true.

**What this file does not prove.** It proves nothing about stochastic differential equations,
exit times, or the Kramers law. This mathlib has no Itô integral, no stochastic differential
equation, no Fokker–Planck equation and no first-passage theory, so `effDiffusion` is a
definition naming the paper's `σ_eff² = s² σ_obj² + σ_Jac²` and `shoulderExponent α σ² = 2α/σ²`
is a definition naming the Arrhenius exponent; the asymptotic `E[τ] ≍ exp(2α/σ_eff²)` that
connects them to an escape time is not established anywhere in this file, and the theorems
below are silent about escape times. What is proved is the `s`-scaling of the exponent, which
is the whole of the power-of-`s` question. Equally, that `σ_eff²` *is* the projected update
covariance of the lifted SGD iterate is not proved here: this file takes the two channels as
its starting definitions, in the form the paper gives them, and counts prefactors in them.

The occurrences to be corrected are, in `iclr/paper_icnn_lift_iclr.tex`, seven in number as of
this writing: the regime-scalar paragraph of the mechanism section, the intuition line and the
two in the scope remark attached to the first-passage corollary, the escape-diagnostic
paragraph, and the crossover remark of the appendix — six of the seven, each attaching the word
to `σ_Jac²` — together with one in the appendix walk-through's Step 3 that is **correct as
written**, because there the word describes the batch-resampling part of the diffusion, that is
the jitter itself, and not its cross-covariance. The same defect recurs in the longer
`paper_icnn_lift.tex`, where the word occurs nine times on nine lines: the Step 3 occurrence of
the appendix walk-through is again the correct one, the figure caption calling the slack channel
"the unmodulated escape route" is ambiguous rather than plainly wrong — read of `δθ̃` it is true,
read of `σ_Jac²` it is false — and the remaining seven attach the word to `σ_Jac²` and are wrong.
`unattenuated_jitter_but_once_attenuated_channel` below is the statement
that separates the two cases, and it is what makes the edit safe to perform mechanically: the
word survives where it qualifies `δθ̃` and must go where it qualifies `σ_Jac²`.

## Results
* `trace_crossMomentMatrix` — the trace of `E_X[a bᵀ]` is the diagonal pairing `∑ᵢ E_X[aᵢ bᵢ]`.
* `batchFluct_chainRuleGrad` — centring commutes with the positivity-map prefactor, `δ(s·u) = s·δu`.
* `traceCrossMoment_chainRuleGrad_left`, `traceCrossMoment_chainRuleGrad_right` — the trace
  pairing is homogeneous of degree one in each slot separately.
* `objChannel_eq` — the objective channel is `s² σ_obj²`: the prefactor enters both slots.
* `jacChannel_eq` — the slack channel is `s · q`: the prefactor enters one slot only.
* `prefactor_power_count` — the two counts side by side, from the same two lemmas.
* `jacChannel_two_sided`, `jacChannel_two_sided_uniform_on_band` — `q₁ s ≤ σ_Jac² ≤ q₂ s` under
  the non-degeneracy hypothesis, with constants uniform over the band.
* `jacChannel_isTheta_id` — `σ_Jac² =Θ[𝓝[>] 0] s`.
* `jacChannel_tendsto_nhds_zero`, `jacChannel_not_isTheta_one` — the slack channel vanishes as
  the positivity map flattens, so it is *not* `Θ(1)`: "unattenuated" is false of it.
* `couplingRatio_eq`, `couplingRatio_two_sided`, `couplingRatio_isTheta_inv` — `ϱ = q/(σ_obj² s)`
  and `ϱ =Θ[𝓝[>] 0] 1/s`.
* `couplingRatio_tendsto_atTop` — `ϱ → ∞` as `s → 0⁺`: the correction is a wording fix.
* `batchFluct_slack_plus_body` — `eq:lift-batch-fluct`: the slack cancels out of the fluctuation.
* `jitterPower_const`, `jitterPower_isTheta_one` — `E_X‖δθ̃‖²` does not depend on the prefactor
  and is `Θ(1)`: "unattenuated" is true of the jitter.
* `jacChannel_pre_readout` — the channel computed on the latent iterate `θ̃_s` itself agrees
  with the one computed on the body, because the slack cancels out of the fluctuation.
* `unattenuated_jitter_but_once_attenuated_channel` — the true and the false reading, together,
  both about the same field `θ̃_s`.
* `positivityMapGrad_eq_chainRuleGrad_add_defect`, `jacChannelExact_split` — the exact decomposition of
  the true chain rule, with the (A2) defect's contribution isolated as a named remainder.
* `abs_remainder_le_of_prefactor_defect` — that remainder is at most `ε ∑ᵢ E_X|δθ̃ᵢ (∇_θL)ᵢ|`
  when the prefactor field is within `ε` of `s` pointwise.
* `mixedAbsMoment`, `abs_remainder_le_of_relative_prefactor_defect` — (A2) read at *relative*
  accuracy `|ψ'(θ̃)_i - s| ≤ ε s` sizes the remainder as `(ε M) s`, which is the shape the
  asymptotic conclusion needs and which absolute accuracy does not supply.
* `jacChannelExact_two_sided` — `(q₁ - c) s ≤ σ_Jac² ≤ (q₂ + c) s` with the remainder carried.
* `corrected_power_survives_remainder` — `Θ(s)`, `Θ(1/s)` and the divergence, all with the
  remainder carried, provided the remainder constant `c` is smaller than `q₁`.
* `jacChannelExactFam`, `jacChannelExactFam_two_sided`, `exact_channel_corrected_power` — the
  same three conclusions for the exact channel as a *function of the prefactor*, computed from
  the true coordinatewise positivity-map field at each prefactor value, with no (A2) idealization
  anywhere in the statement.
* `effDiffusion_two_sided`, `liftExponent_isTheta_inv`, `directExponent_isTheta_inv_sq` — the
  shoulder penalty is `Θ(1/s)` for the lift and `Θ(1/s²)` for the direct baseline.
* `batchFluct_eq_zero_of_const`, `jacChannel_eq_zero_of_const`, `effDiffusion_direct`,
  `directExponent_effDiffusion_isTheta_inv_sq` — the direct parametrization's slack channel is
  identically zero, so the `Θ(1/s²)` statement is about the direct baseline's own effective
  diffusion and not merely about a term of it.
* `exponentRatio_eq`, `exponentRatio_tendsto_atTop` — the ratio of the two exponents is exactly
  `1 + ϱ`, and it diverges.
* `crossMomentMatrix_self_posSemidef` — `V = E_X[δθ̃ δθ̃ᵀ]` is positive semidefinite.
* `traceCrossMoment_mulVec`, `trace_crossMomentMatrix_mul_nonneg` — the forward-KL leading term
  of `q` is `tr(V H̄)`, and it is non-negative by self-duality of the PSD cone.
* `couplingCore_eq_trace_add_remainder`, `couplingCore_ge_neg_abs_remainder`,
  `couplingCore_pos_of_forwardKL` — the (A4) split of `q`, the non-negativity that
  positive-semidefiniteness of `H̄` buys on its own, and the strict positivity that the remainder
  gap buys on top of it.
* `power_of_s_correction` — FW-3b assembled into one statement.
* `witMeasure` and the `wit*` family, `power_of_s_correction_at_witness`,
  `unattenuated_jitter_but_once_attenuated_channel_at_witness`,
  `couplingCore_pos_of_forwardKL_at_witness`, `exact_channel_corrected_power_at_witness` — an
  explicit two-point batch law satisfying every hypothesis of the four headline statements, so
  that none of them is vacuously true.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses this file carries, and what each would take to discharge.

*Non-degeneracy of the coupling functional*, `0 < q₁ ≤ q ≤ q₂` on the shoulder band. This is the
hypothesis of the `Θ(s)` statement and it is stated, never assumed in the form of its own
conclusion: the theorems take the two-sided bound on `q` and produce the two-sided bound on
`σ_Jac²`. Its lower half is reduced, in the last section, to (A4)'s forward-KL chain
`δ(∇_θL) = H̄ δθ̃ + r` together with a gap condition `|tr E_X[δθ̃ rᵀ]| < tr(V H̄)`. Be exact about
what each input buys, because it is easy to overstate. Positive-semidefiniteness of `H̄` buys the
*non-negativity* of the leading term `tr(V H̄)` from the self-duality of the positive-semidefinite
cone, and that, and only that, is the content of `couplingCore_ge_neg_abs_remainder`; in
`couplingCore_pos_of_forwardKL` it is used only for the symmetry of `H̄`, and the strictness is
carried entirely by the gap. The gap condition is not a free lunch either: it itself forces
`tr(V H̄) > 0`, that is, it carries the requirement that the curvature act non-degenerately on
the jitter covariance. The upper bound `q ≤ q₂` and the uniformity of both constants over the
band are moment bounds of (A1) and remain hypotheses; discharging any of these needs a
quantitative model of the latent weight, which the paper does not supply and this development does
not invent.

*The single-prefactor idealization* (A2), `ψ'(θ̃) ≈ s I_d` on the band. This is carried
quantitatively as `|p ω i - s| ≤ ε`, never as an equality. The exact statements
(`jacChannelExact_split`) hold with no bound at all; the bound is used only to size the
remainder, and the remainder survives into the conclusion of every asymptotic statement that
uses it. The asymptotic statements need (A2) in its *relative* form, `|p ω i - s| ≤ ε s`, and
that is a real strengthening of the hypothesis rather than a restatement: the absolute form with
`ε` held fixed as the shoulder flattens is not enough for any of them, and this file does not
pretend otherwise. Discharging the relative form would need a uniform bound on `ψ''` over the
band together with a band whose width shrinks with the prefactor, neither of which the paper
supplies.

*The prefactor-freedom of the jitter.* This is not proved from nothing, and it could not be:
that the latent weight is evaluated before the positivity map is a fact about the lift's architecture rather
than a theorem. What is a hypothesis is exactly the decomposition `θ̃_s = b_s + h_φ(X)` with
the body `h_φ` independent of the prefactor, and the hypothesis is deliberately weaker than it
could have been — the slack `b_s` is allowed to depend on the prefactor, so the iterate itself
varies with `s` and only its fluctuation does not. That the fluctuation does not is then the
slack cancellation of `eq:lift-batch-fluct`, which is proved. Had `b` been held fixed as well,
`jitterPower_const` would have been a restatement of its own hypothesis.

*Integrability*, in the form of (A1)'s moment bounds. The scaling lemmas need none — mathlib's
Bochner integral is homogeneous unconditionally — but additivity of the integral, and hence the
exact split of the chain rule into idealization plus defect, does. Each such hypothesis is
stated coordinatewise and named at its use site. Discharging them needs a moment model of the
batch law, which is (A1).

*The SDE-of-SGD picture, the Kramers law, and the identification of `σ_eff²` with the update
covariance.* None of these is proved or assumed here; the exponent results are statements about
the real function `2α/σ_eff²(s)` and its `s`-scaling, and they are silent about exit times. See
"What this file does not prove" above.

The hypotheses are shown satisfiable by an explicit witness rather than argued to be plausible;
see the last section. The witness is minimal, and it establishes only non-vacuity — it is not
evidence that any particular trained network satisfies the non-degeneracy bounds.

Finally, the asymptotics are stated along `𝓝[>] 0` in the prefactor with every other quantity
held fixed. That is the paper's own idealization of the flattening shoulder — `q` and `σ_obj²`
are evaluated at a common reference iterate, as `FW-0` of the design note requires — and it is
not a statement about a limit taken along an actual training trajectory.
-/

open MeasureTheory Filter Asymptotics Matrix
open scoped Topology

namespace IcnnLift

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- The batch fluctuation `δf = f(X) - E_X[f(X)]` of a `d`-vector field, coordinatewise. -/
noncomputable def batchFluct (μ : Measure Ω) (f : Ω → Fin d → ℝ) : Ω → Fin d → ℝ :=
  fun ω i => f ω i - ∫ ω', f ω' i ∂μ

/-- The matrix second moment `E_X[a bᵀ]` of two `d`-vector fields. -/
noncomputable def crossMomentMatrix (μ : Measure Ω) (a b : Ω → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => ∫ ω, a ω i * b ω j ∂μ

/-- The trace `tr E_X[a bᵀ]` of the matrix second moment. -/
noncomputable def traceCrossMoment (μ : Measure Ω) (a b : Ω → Fin d → ℝ) : ℝ :=
  ∑ i, ∫ ω, a ω i * b ω i ∂μ

/-- The two readings of `tr E_X[a bᵀ]` agree: the trace of the matrix second moment is the sum
of the diagonal pairings `∑ᵢ E_X[aᵢ bᵢ]`. Every power count below is carried out on the second
form, and this lemma is what licenses reading the result as a statement about the matrix. -/
theorem trace_crossMomentMatrix (μ : Measure Ω) (a b : Ω → Fin d → ℝ) :
    (crossMomentMatrix μ a b).trace = traceCrossMoment μ a b := by
  simp [Matrix.trace, Matrix.diag, crossMomentMatrix, traceCrossMoment]

/-- The chain rule `∇_{θ̃}L = ψ'(θ̃) ⊙ ∇_θ L`, with `p` the positivity-map prefactor field. -/
def positivityMapGrad (p u : Ω → Fin d → ℝ) : Ω → Fin d → ℝ := fun ω i => p ω i * u ω i

/-- The chain rule under the single-prefactor idealization (A2), `ψ'(θ̃) = s I_d`. -/
def chainRuleGrad (s : ℝ) (u : Ω → Fin d → ℝ) : Ω → Fin d → ℝ := fun ω i => s * u ω i

omit [MeasurableSpace Ω] in
/-- (A2) is the special case of the chain rule in which the coordinatewise prefactor field is
the constant `s`. Nothing is approximated here; the defect of the idealization is introduced,
and carried, in `prefactorDefect` below. -/
theorem chainRuleGrad_eq_positivityMapGrad (s : ℝ) (u : Ω → Fin d → ℝ) :
    chainRuleGrad s u = positivityMapGrad (fun _ _ => s) u := rfl

/-! ### Scaling of the two slots -/

/-- Centring commutes with the positivity-map prefactor, `δ(s·u) = s·δu`. This is the step at which the
prefactor survives the passage from the gradient to its batch fluctuation, and it needs no
integrability hypothesis, because the Bochner integral is homogeneous unconditionally. -/
theorem batchFluct_chainRuleGrad (μ : Measure Ω) (s : ℝ) (u : Ω → Fin d → ℝ) :
    batchFluct μ (chainRuleGrad s u) = chainRuleGrad s (batchFluct μ u) := by
  funext ω i
  simp only [batchFluct, chainRuleGrad]
  rw [integral_const_mul]
  ring

/-- The trace pairing is homogeneous of degree one in its **second** slot. Together with the
companion lemma for the first slot, this is the entire mechanism of the power count: a channel
picks up one factor of the prefactor for each slot the prefactor enters. -/
theorem traceCrossMoment_chainRuleGrad_right (μ : Measure Ω) (s : ℝ) (a b : Ω → Fin d → ℝ) :
    traceCrossMoment μ a (chainRuleGrad s b) = s * traceCrossMoment μ a b := by
  simp only [traceCrossMoment, chainRuleGrad, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have h : ∀ ω, a ω i * (s * b ω i) = s * (a ω i * b ω i) := fun ω => by ring
  simp_rw [h]
  rw [integral_const_mul]

/-- The trace pairing is homogeneous of degree one in its **first** slot. -/
theorem traceCrossMoment_chainRuleGrad_left (μ : Measure Ω) (s : ℝ) (a b : Ω → Fin d → ℝ) :
    traceCrossMoment μ (chainRuleGrad s a) b = s * traceCrossMoment μ a b := by
  simp only [traceCrossMoment, chainRuleGrad, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have h : ∀ ω, s * a ω i * b ω i = s * (a ω i * b ω i) := fun ω => by ring
  simp_rw [h]
  rw [integral_const_mul]

/-! ### The two channels, and the power of `s` each carries -/

/-- `σ_obj² = tr Var_X[∇_θ L]`, the gradient-noise variance on the **constrained** coordinate,
before the positivity-map prefactor is applied. -/
noncomputable def objNoise (μ : Measure Ω) (u : Ω → Fin d → ℝ) : ℝ :=
  traceCrossMoment μ (batchFluct μ u) (batchFluct μ u)

/-- The coupling functional with the positivity-map prefactor stripped off,
`q = tr E_X[δθ̃ δ(∇_θ L)ᵀ]`. -/
noncomputable def couplingCore (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) : ℝ :=
  traceCrossMoment μ (batchFluct μ dth) (batchFluct μ u)

/-- The objective channel of the bias-channel diffusion, `tr Var_X[δg]`. -/
noncomputable def objChannel (μ : Measure Ω) (u : Ω → Fin d → ℝ) (s : ℝ) : ℝ :=
  traceCrossMoment μ (batchFluct μ (chainRuleGrad s u)) (batchFluct μ (chainRuleGrad s u))

/-- The slack channel `σ_Jac² = tr E_X[δθ̃ δgᵀ]`. -/
noncomputable def jacChannel (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) (s : ℝ) : ℝ :=
  traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (chainRuleGrad s u))

/-- **The objective channel is attenuated twice.** Both slots of `tr E_X[δg δgᵀ]` carry the
gradient, so both pass through the positivity-map prefactor, and the channel is `s² σ_obj²`. This is
the paper's own statement that `σ_obj²` is measured on the constrained coordinate *before* the
prefactor is applied. -/
theorem objChannel_eq (μ : Measure Ω) (u : Ω → Fin d → ℝ) (s : ℝ) :
    objChannel μ u s = s ^ 2 * objNoise μ u := by
  rw [objChannel, batchFluct_chainRuleGrad, traceCrossMoment_chainRuleGrad_left,
    traceCrossMoment_chainRuleGrad_right, objNoise]
  ring

/-- **The slack channel is attenuated once.** Only the second slot of `tr E_X[δθ̃ δgᵀ]` carries a
gradient; the first carries the latent iterate jitter, which is a re-evaluation of the
latent weight and passes through no `ψ'`. The channel is therefore `s · q` with
`q = tr E_X[δθ̃ δ(∇_θL)ᵀ]` the prefactor-free coupling functional. This is the correction of
`docs/design/lemma1_rederivation.md` §11.2. -/
theorem jacChannel_eq (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) (s : ℝ) :
    jacChannel μ dth u s = s * couplingCore μ dth u := by
  rw [jacChannel, batchFluct_chainRuleGrad, traceCrossMoment_chainRuleGrad_right, couplingCore]

/-- The two counts side by side, both derived from the same pair of homogeneity lemmas rather
than stipulated: two powers for the objective channel, one for the slack channel. -/
theorem prefactor_power_count (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) (s : ℝ) :
    objChannel μ u s = s ^ 2 * objNoise μ u ∧ jacChannel μ dth u s = s * couplingCore μ dth u :=
  ⟨objChannel_eq μ u s, jacChannel_eq μ dth u s⟩

/-! ### Two-sided bounds and the `Θ(s)` law -/

/-- A two-sided bound with positive constants on the right of `0` upgrades to `IsTheta`. -/
theorem isTheta_nhdsGT_zero_of_two_sided {f g : ℝ → ℝ} {c₁ c₂ : ℝ} (hc₁ : 0 < c₁)
    (hg : ∀ s : ℝ, 0 < s → 0 ≤ g s)
    (hlo : ∀ s : ℝ, 0 < s → c₁ * g s ≤ f s) (hhi : ∀ s : ℝ, 0 < s → f s ≤ c₂ * g s) :
    f =Θ[𝓝[>] (0:ℝ)] g := by
  have hmem : Set.Ioi (0:ℝ) ∈ 𝓝[>] (0:ℝ) := self_mem_nhdsWithin
  have hfnn : ∀ s : ℝ, 0 < s → 0 ≤ f s := fun s hs =>
    le_trans (mul_nonneg hc₁.le (hg s hs)) (hlo s hs)
  constructor
  · rw [isBigO_iff]
    refine ⟨c₂, Filter.eventually_of_mem hmem fun s hs => ?_⟩
    have hs' : (0:ℝ) < s := hs
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (hfnn s hs'),
      abs_of_nonneg (hg s hs')]
    exact hhi s hs'
  · rw [isBigO_iff]
    refine ⟨c₁⁻¹, Filter.eventually_of_mem hmem fun s hs => ?_⟩
    have hs' : (0:ℝ) < s := hs
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (hfnn s hs'),
      abs_of_nonneg (hg s hs')]
    have hle := hlo s hs'
    rw [le_inv_mul_iff₀ hc₁]
    linarith

/-- **The two-sided bound for the slack channel.** Under the non-degeneracy hypothesis
`q₁ ≤ q ≤ q₂` on the prefactor-free coupling functional, the slack channel is trapped between
`q₁ s` and `q₂ s` for every positive prefactor. The constants are explicit and the hypothesis is
about `q`, not about the conclusion. -/
theorem jacChannel_two_sided {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {q₁ q₂ : ℝ}
    (h1 : q₁ ≤ couplingCore μ dth u) (h2 : couplingCore μ dth u ≤ q₂) {s : ℝ} (hs : 0 < s) :
    q₁ * s ≤ jacChannel μ dth u s ∧ jacChannel μ dth u s ≤ q₂ * s := by
  rw [jacChannel_eq]
  constructor
  · rw [mul_comm s]
    exact mul_le_mul_of_nonneg_right h1 hs.le
  · rw [mul_comm s]
    exact mul_le_mul_of_nonneg_right h2 hs.le

/-- The objective channel's `s`-free factor is a sum of second moments, hence non-negative. -/
theorem objNoise_nonneg (μ : Measure Ω) (u : Ω → Fin d → ℝ) : 0 ≤ objNoise μ u := by
  rw [objNoise, traceCrossMoment]
  refine Finset.sum_nonneg fun i _ => ?_
  exact integral_nonneg (f := fun ω => batchFluct μ u ω i * batchFluct μ u ω i)
    fun ω => mul_self_nonneg _

/-- **`σ_Jac² = Θ(s)`**, in mathlib's own `IsTheta` along `𝓝[>] 0`. The only hypothesis is that
the prefactor-free coupling functional is strictly positive; the constants are `q` itself in both
directions. -/
theorem jacChannel_isTheta_id {μ : Measure Ω} {dth u : Ω → Fin d → ℝ}
    (hq : 0 < couplingCore μ dth u) :
    (fun s => jacChannel μ dth u s) =Θ[𝓝[>] (0:ℝ)] (fun s => s) :=
  isTheta_nhdsGT_zero_of_two_sided (c₂ := couplingCore μ dth u) hq (fun _s hs => hs.le)
    (fun _s hs => (jacChannel_two_sided le_rfl le_rfl hs).1)
    (fun _s hs => (jacChannel_two_sided le_rfl le_rfl hs).2)

/-- The slack channel vanishes in the limit of a flat positivity map. -/
theorem jacChannel_tendsto_nhds_zero (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) :
    Tendsto (fun s => jacChannel μ dth u s) (𝓝[>] (0:ℝ)) (𝓝 0) := by
  have h : Tendsto (fun s : ℝ => s * couplingCore μ dth u) (𝓝[>] (0:ℝ))
      (𝓝 (0 * couplingCore μ dth u)) :=
    (tendsto_id.mono_left nhdsWithin_le_nhds).mul_const _
  rw [zero_mul] at h
  exact h.congr fun s => (jacChannel_eq μ dth u s).symm

/-- **The slack channel is not unattenuated.** Since it carries one power of the prefactor it
vanishes in the limit of a flat positivity map, so it cannot be `Θ(1)`. This is the precise sense in which
the shipped text's word is wrong, and it holds with no hypothesis at all: the argument is that a
`Θ(1)` quantity cannot tend to zero along a filter that is not trivial. -/
theorem jacChannel_not_isTheta_one (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) :
    ¬ ((fun s => jacChannel μ dth u s) =Θ[𝓝[>] (0:ℝ)] (fun _ => (1:ℝ))) := by
  intro h
  have h0 : Tendsto (fun _ : ℝ => (1:ℝ)) (𝓝[>] (0:ℝ)) (𝓝 0) :=
    h.tendsto_zero_iff.mp (jacChannel_tendsto_nhds_zero μ dth u)
  exact one_ne_zero (tendsto_nhds_unique tendsto_const_nhds h0)

/-! ### The ratio still diverges -/

/-- The dimensionless control `ϱ = σ_Jac²/(s² σ_obj²)`. -/
noncomputable def couplingRatio (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) (s : ℝ) : ℝ :=
  jacChannel μ dth u s / (s ^ 2 * objNoise μ u)

/-- The closed form of the dimensionless control at the corrected power: one power of the
prefactor cancels between numerator and denominator and one survives, leaving `ϱ = q/(σ_obj² s)`.
Had the channel been unattenuated, `ϱ` would have been `Θ(1/s²)` instead; the correction changes
the rate of the divergence, not its existence. -/
theorem couplingRatio_eq {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {s : ℝ} (hs : s ≠ 0)
    (ho : objNoise μ u ≠ 0) :
    couplingRatio μ dth u s = couplingCore μ dth u / objNoise μ u * s⁻¹ := by
  rw [couplingRatio, jacChannel_eq]
  field_simp

/-- The two-sided bound on `ϱ` inherited from the two-sided bound on `q`, with explicit
constants. -/
theorem couplingRatio_two_sided {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {q₁ q₂ : ℝ}
    (h1 : q₁ ≤ couplingCore μ dth u) (h2 : couplingCore μ dth u ≤ q₂)
    (ho : 0 < objNoise μ u) {s : ℝ} (hs : 0 < s) :
    q₁ / objNoise μ u * s⁻¹ ≤ couplingRatio μ dth u s ∧
      couplingRatio μ dth u s ≤ q₂ / objNoise μ u * s⁻¹ := by
  rw [couplingRatio_eq hs.ne' ho.ne']
  have hinv : (0:ℝ) ≤ s⁻¹ := by positivity
  constructor
  · gcongr
  · gcongr

/-- **`ϱ = Θ(1/s)`.** The dimensionless control diverges at exactly the rate the corrected power
predicts. -/
theorem couplingRatio_isTheta_inv {μ : Measure Ω} {dth u : Ω → Fin d → ℝ}
    (hq : 0 < couplingCore μ dth u) (ho : 0 < objNoise μ u) :
    (fun s => couplingRatio μ dth u s) =Θ[𝓝[>] (0:ℝ)] (fun s => s⁻¹) :=
  isTheta_nhdsGT_zero_of_two_sided (c₂ := couplingCore μ dth u / objNoise μ u)
    (div_pos hq ho) (fun _s hs => by positivity)
    (fun _s hs => (couplingRatio_two_sided le_rfl le_rfl ho hs).1)
    (fun _s hs => (couplingRatio_two_sided le_rfl le_rfl ho hs).2)

/-- **The consequence that saves the paper.** Even at the corrected power the dimensionless
control diverges as the positivity map flattens, so every downstream conclusion that consumes only
`ϱ → ∞` — the crossover discussion, the ordering of the two channels, the reading of the
diffusive probe — survives the correction unchanged. This is the machine-checked statement that
the misplaced occurrences of "unattenuated" are a wording defect rather than a retraction. -/
theorem couplingRatio_tendsto_atTop {μ : Measure Ω} {dth u : Ω → Fin d → ℝ}
    (hq : 0 < couplingCore μ dth u) (ho : 0 < objNoise μ u) :
    Tendsto (fun s => couplingRatio μ dth u s) (𝓝[>] (0:ℝ)) atTop := by
  have hbase : Tendsto (fun s : ℝ => couplingCore μ dth u / objNoise μ u * s⁻¹)
      (𝓝[>] (0:ℝ)) atTop :=
    Filter.Tendsto.const_mul_atTop (div_pos hq ho) tendsto_inv_nhdsGT_zero
  refine hbase.congr' (Filter.eventually_of_mem self_mem_nhdsWithin fun s hs => ?_)
  exact (couplingRatio_eq (ne_of_gt (Set.mem_Ioi.mp hs)) ho.ne').symm

/-! ### The contrast: the jitter really is unattenuated -/

/-- `eq:lift-batch-fluct`: the slack cancels out of the fluctuation. The latent iterate is
`θ̃ = b + h_φ(X)` with the slack `b` constant across the window over which the fluctuation is
taken, so `δθ̃ = δh_φ(X)`, and the body carries the batch-induced fluctuation entirely. This is
the step from which the prefactor-freedom of the jitter follows: what survives the centring is
evaluated before the positivity map. -/
theorem batchFluct_slack_plus_body [IsProbabilityMeasure μ] {b : Fin d → ℝ}
    {h theta : Ω → Fin d → ℝ} (hint : ∀ i, Integrable (fun ω => h ω i) μ)
    (hdec : ∀ ω i, theta ω i = b i + h ω i) :
    batchFluct μ theta = batchFluct μ h := by
  funext ω i
  have hmean : ∫ ω', theta ω' i ∂μ = b i + ∫ ω', h ω' i ∂μ := by
    simp_rw [fun ω' => hdec ω' i]
    rw [integral_add (integrable_const (b i)) (hint i), integral_const]
    simp
  rw [batchFluct, batchFluct, hdec ω i, hmean]
  ring

/-- The mean-square size of the latent-weight jitter, `E_X‖δθ̃‖²`. -/
noncomputable def jitterPower (μ : Measure Ω) (P : ℝ → Ω → Fin d → ℝ) (s : ℝ) : ℝ :=
  traceCrossMoment μ (batchFluct μ (P s)) (batchFluct μ (P s))

/-- **The jitter carries no positivity-map prefactor.** The latent iterate at shoulder prefactor
`s` is `θ̃_s = b_s + h_φ(X)`. The slack is *not* required to be independent of the prefactor:
different prefactor values `s = ψ'(θ̃_s)` are different locations on the shoulder, and it is the
slack that moves the iterate there, so `b` carries an `s` and the iterate `P s` genuinely depends
on the prefactor. What is required of the architecture is only that the *body* is evaluated
before the positivity map, hence independent of `ψ'`. The mean-square fluctuation is nevertheless the
same at every value of `s`, and the whole content of that is the slack cancellation of
`batchFluct_slack_plus_body`: what survives the centring is the body's own fluctuation. -/
theorem jitterPower_const [IsProbabilityMeasure μ] {b : ℝ → Fin d → ℝ} {h : Ω → Fin d → ℝ}
    {P : ℝ → Ω → Fin d → ℝ} (hint : ∀ i, Integrable (fun ω => h ω i) μ)
    (hdec : ∀ s ω i, P s ω i = b s i + h ω i) (s s' : ℝ) :
    jitterPower μ P s = jitterPower μ P s' := by
  rw [jitterPower, jitterPower, batchFluct_slack_plus_body hint (hdec s),
    batchFluct_slack_plus_body hint (hdec s')]

/-- **The jitter is `Θ(1)`, so the word "unattenuated" is true of it.** The hypothesis is that
the latent weight genuinely jitters, `E_X‖δθ̃‖² > 0`; that is ingredient (ii) of the paper's
three-ingredient decomposition, and without it the lift has no channel at all. -/
theorem jitterPower_isTheta_one [IsProbabilityMeasure μ] {b : ℝ → Fin d → ℝ} {h : Ω → Fin d → ℝ}
    {P : ℝ → Ω → Fin d → ℝ} (hint : ∀ i, Integrable (fun ω => h ω i) μ)
    (hdec : ∀ s ω i, P s ω i = b s i + h ω i)
    (hpos : 0 < jitterPower μ P 1) :
    (fun s => jitterPower μ P s) =Θ[𝓝[>] (0:ℝ)] (fun _ => (1:ℝ)) := by
  refine isTheta_nhdsGT_zero_of_two_sided (c₁ := jitterPower μ P 1)
    (c₂ := jitterPower μ P 1) hpos (fun _s _hs => zero_le_one) (fun s _hs => ?_) (fun s _hs => ?_)
  · rw [mul_one, jitterPower_const hint hdec 1 s]
  · rw [mul_one, jitterPower_const hint hdec s 1]

/-- The slack channel computed on the **actual latent iterate** `θ̃_s = b_s + h_φ(X)`, rather
than on the body alone, is the same number. This is again the slack cancellation, and it is what
lets the contrast below be stated about the object the paper writes `σ_Jac²` for instead of about
a proxy for it. -/
theorem jacChannel_pre_readout [IsProbabilityMeasure μ] {b : ℝ → Fin d → ℝ}
    {h u : Ω → Fin d → ℝ} {P : ℝ → Ω → Fin d → ℝ}
    (hint : ∀ i, Integrable (fun ω => h ω i) μ)
    (hdec : ∀ s ω i, P s ω i = b s i + h ω i) (s s' : ℝ) :
    jacChannel μ (P s') u s = jacChannel μ h u s := by
  rw [jacChannel, jacChannel, batchFluct_slack_plus_body hint (hdec s')]

/-- **The distinction the correction turns on, side by side.** The jitter is `Θ(1)` — the word
"unattenuated" describes it correctly, which is why the shipped text's one occurrence about the
batch-resampling part is right — while the slack channel is `Θ(s)` and demonstrably not `Θ(1)`,
which is why the same word applied to `σ_Jac²` is wrong. Both halves are stated in a single
conclusion, and both are stated about the *same* latent iterate field `P`, so that the
distinction is visible in the Lean statement rather than only in the prose: the two occurrences
of `P s` in the conclusion are literally the same `θ̃_s`, read once through its own second moment
and once through its cross-moment with the gradient fluctuation. -/
theorem unattenuated_jitter_but_once_attenuated_channel [IsProbabilityMeasure μ]
    {b : ℝ → Fin d → ℝ} {h u : Ω → Fin d → ℝ} {P : ℝ → Ω → Fin d → ℝ}
    (hint : ∀ i, Integrable (fun ω => h ω i) μ)
    (hdec : ∀ s ω i, P s ω i = b s i + h ω i)
    (hjit : 0 < jitterPower μ P 1) (hq : 0 < couplingCore μ h u) :
    ((fun s => jitterPower μ P s) =Θ[𝓝[>] (0:ℝ)] fun _ => (1:ℝ)) ∧
      ((fun s => jacChannel μ (P s) u s) =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      ¬ ((fun s => jacChannel μ (P s) u s) =Θ[𝓝[>] (0:ℝ)] fun _ => (1:ℝ)) := by
  have hfun : (fun s => jacChannel μ (P s) u s) = fun s => jacChannel μ h u s :=
    funext fun s => jacChannel_pre_readout hint hdec s s
  rw [hfun]
  exact ⟨jitterPower_isTheta_one hint hdec hjit, jacChannel_isTheta_id hq,
    jacChannel_not_isTheta_one μ h u⟩

/-! ### The abstract analytic layer -/

/-- The two-sided bound only has to hold on a right neighbourhood of `0`. -/
theorem isTheta_nhdsGT_zero_of_two_sided_near {f g : ℝ → ℝ} {c₁ c₂ s₀ : ℝ} (hc₁ : 0 < c₁)
    (hs₀ : 0 < s₀) (hg : ∀ s : ℝ, 0 < s → s ≤ s₀ → 0 ≤ g s)
    (hlo : ∀ s : ℝ, 0 < s → s ≤ s₀ → c₁ * g s ≤ f s)
    (hhi : ∀ s : ℝ, 0 < s → s ≤ s₀ → f s ≤ c₂ * g s) :
    f =Θ[𝓝[>] (0:ℝ)] g := by
  have hmem : Set.Ioi (0:ℝ) ∩ Set.Iio s₀ ∈ 𝓝[>] (0:ℝ) :=
    Filter.inter_mem self_mem_nhdsWithin (nhdsWithin_le_nhds (isOpen_Iio.mem_nhds hs₀))
  have hfnn : ∀ s : ℝ, 0 < s → s ≤ s₀ → 0 ≤ f s := fun s hs hs' =>
    le_trans (mul_nonneg hc₁.le (hg s hs hs')) (hlo s hs hs')
  constructor
  · rw [isBigO_iff]
    refine ⟨c₂, Filter.eventually_of_mem hmem fun s hs => ?_⟩
    obtain ⟨hs1, hs2⟩ := hs
    have hs1' : (0:ℝ) < s := hs1
    have hs2' : s ≤ s₀ := le_of_lt hs2
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (hfnn s hs1' hs2'),
      abs_of_nonneg (hg s hs1' hs2')]
    exact hhi s hs1' hs2'
  · rw [isBigO_iff]
    refine ⟨c₁⁻¹, Filter.eventually_of_mem hmem fun s hs => ?_⟩
    obtain ⟨hs1, hs2⟩ := hs
    have hs1' : (0:ℝ) < s := hs1
    have hs2' : s ≤ s₀ := le_of_lt hs2
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (hfnn s hs1' hs2'),
      abs_of_nonneg (hg s hs1' hs2')]
    have hle := hlo s hs1' hs2'
    rw [le_inv_mul_iff₀ hc₁]
    linarith

/-- **The analytic content of the correction, isolated from every model assumption.** Any
quantity trapped between `c₁ s` and `c₂ s` near `0`, with `c₁ > 0`, is `Θ(s)`, its ratio to
`s² o` is `Θ(1/s)`, and that ratio diverges. Stating it for an abstract `J` is what lets the
same conclusion be drawn for the idealized channel and for the channel with the (A2) remainder
carried, without proving it twice. -/
theorem isTheta_and_ratio_tendsto_atTop_of_two_sided_near {J : ℝ → ℝ} {c₁ c₂ o s₀ : ℝ}
    (hc₁ : 0 < c₁) (ho : 0 < o) (hs₀ : 0 < s₀)
    (hlo : ∀ s : ℝ, 0 < s → s ≤ s₀ → c₁ * s ≤ J s)
    (hhi : ∀ s : ℝ, 0 < s → s ≤ s₀ → J s ≤ c₂ * s) :
    (J =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      ((fun s => J s / (s ^ 2 * o)) =Θ[𝓝[>] (0:ℝ)] fun s => s⁻¹) ∧
      Tendsto (fun s => J s / (s ^ 2 * o)) (𝓝[>] (0:ℝ)) atTop := by
  have hratio_lo : ∀ s : ℝ, 0 < s → s ≤ s₀ → c₁ / o * s⁻¹ ≤ J s / (s ^ 2 * o) := by
    intro s hs hs'
    have heq : c₁ / o * s⁻¹ = c₁ * s / (s ^ 2 * o) := by field_simp
    rw [heq]
    have hden : (0:ℝ) < s ^ 2 * o := by positivity
    exact div_le_div_of_nonneg_right (hlo s hs hs') hden.le
  have hratio_hi : ∀ s : ℝ, 0 < s → s ≤ s₀ → J s / (s ^ 2 * o) ≤ c₂ / o * s⁻¹ := by
    intro s hs hs'
    have heq : c₂ / o * s⁻¹ = c₂ * s / (s ^ 2 * o) := by field_simp
    rw [heq]
    have hden : (0:ℝ) < s ^ 2 * o := by positivity
    exact div_le_div_of_nonneg_right (hhi s hs hs') hden.le
  refine ⟨isTheta_nhdsGT_zero_of_two_sided_near hc₁ hs₀ (fun _s hs _ => hs.le) hlo hhi,
    isTheta_nhdsGT_zero_of_two_sided_near (div_pos hc₁ ho) hs₀
      (fun _s hs _ => by positivity) hratio_lo hratio_hi, ?_⟩
  have hbase : Tendsto (fun s : ℝ => c₁ / o * s⁻¹) (𝓝[>] (0:ℝ)) atTop :=
    Filter.Tendsto.const_mul_atTop (div_pos hc₁ ho) tendsto_inv_nhdsGT_zero
  refine Filter.tendsto_atTop_mono' _ ?_ hbase
  have hmem : Set.Ioi (0:ℝ) ∩ Set.Iio s₀ ∈ 𝓝[>] (0:ℝ) :=
    Filter.inter_mem self_mem_nhdsWithin (nhdsWithin_le_nhds (isOpen_Iio.mem_nhds hs₀))
  exact Filter.eventually_of_mem hmem fun s hs => hratio_lo s hs.1 (le_of_lt hs.2)

/-! ### The (A2) defect, carried explicitly -/

/-- The defining equation of the slack channel, kept as a rewriting lemma so that the exact
split below can fold the idealized part back into `jacChannel`. -/
theorem jacChannel_def (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) (s : ℝ) :
    jacChannel μ dth u s
      = traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (chainRuleGrad s u)) := rfl

/-- Centring is additive. Unlike homogeneity, this genuinely needs integrability of the two
summands, because `integral_add` does; the hypothesis is (A1)'s moment bound, stated
coordinatewise. -/
theorem batchFluct_add {μ : Measure Ω} {f g : Ω → Fin d → ℝ}
    (hf : ∀ i, Integrable (fun ω => f ω i) μ) (hg : ∀ i, Integrable (fun ω => g ω i) μ) :
    batchFluct μ (fun ω i => f ω i + g ω i)
      = fun ω i => batchFluct μ f ω i + batchFluct μ g ω i := by
  funext ω i
  simp only [batchFluct]
  rw [integral_add (hf i) (hg i)]
  ring

/-- The trace pairing is additive in its second slot, given integrability of the two products.
This is what splits the slack channel into its idealized part and the (A2) remainder. -/
theorem traceCrossMoment_add_right {μ : Measure Ω} {a b c : Ω → Fin d → ℝ}
    (hb : ∀ i, Integrable (fun ω => a ω i * b ω i) μ)
    (hc : ∀ i, Integrable (fun ω => a ω i * c ω i) μ) :
    traceCrossMoment μ a (fun ω i => b ω i + c ω i)
      = traceCrossMoment μ a b + traceCrossMoment μ a c := by
  simp only [traceCrossMoment, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have h : ∀ ω, a ω i * (b ω i + c ω i) = a ω i * b ω i + a ω i * c ω i := fun ω => by ring
  simp_rw [h]
  exact integral_add (hb i) (hc i)

/-- The defect of the single-prefactor idealization (A2). -/
def prefactorDefect (s : ℝ) (p u : Ω → Fin d → ℝ) : Ω → Fin d → ℝ :=
  fun ω i => (p ω i - s) * u ω i

omit [MeasurableSpace Ω] in
/-- The true chain rule is the (A2) idealization plus its defect, **exactly**: no approximation
is made at this step, and `prefactorDefect` is a definition rather than an assumption. -/
theorem positivityMapGrad_eq_chainRuleGrad_add_defect (s : ℝ) (p u : Ω → Fin d → ℝ) :
    positivityMapGrad p u = fun ω i => chainRuleGrad s u ω i + prefactorDefect s p u ω i := by
  funext ω i
  simp only [positivityMapGrad, chainRuleGrad, prefactorDefect]
  ring

/-- `σ_Jac²` computed from the true prefactor field, with no (A2) idealization. -/
noncomputable def jacChannelExact (μ : Measure Ω) (dth p u : Ω → Fin d → ℝ) : ℝ :=
  traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (positivityMapGrad p u))

/-- **The exact split.** The slack channel computed from the true prefactor field equals its
(A2) value `s · q` plus a remainder that is written down rather than discarded. This is an
identity: no bound on the defect is used, and the only hypotheses are the integrability
conditions that additivity of the integral requires. -/
theorem jacChannelExact_split {μ : Measure Ω} {dth p u : Ω → Fin d → ℝ} (s : ℝ)
    (hu : ∀ i, Integrable (fun ω => u ω i) μ)
    (hr : ∀ i, Integrable (fun ω => prefactorDefect s p u ω i) μ)
    (h1 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (chainRuleGrad s u) ω i) μ)
    (h2 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (prefactorDefect s p u) ω i) μ) :
    jacChannelExact μ dth p u
      = s * couplingCore μ dth u
        + traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (prefactorDefect s p u)) := by
  rw [jacChannelExact, positivityMapGrad_eq_chainRuleGrad_add_defect s,
    batchFluct_add (f := chainRuleGrad s u) (g := prefactorDefect s p u)
      (fun i => (hu i).const_mul s) hr,
    traceCrossMoment_add_right h1 h2, ← jacChannel_def, jacChannel_eq]

/-- The trace pairing is bounded by the sum of the absolute first moments of its diagonal
products. No integrability hypothesis is needed, because mathlib's `abs_integral_le_integral_abs`
is unconditional. -/
theorem abs_traceCrossMoment_le (μ : Measure Ω) (a b : Ω → Fin d → ℝ) :
    |traceCrossMoment μ a b| ≤ ∑ i, ∫ ω, |a ω i * b ω i| ∂μ := by
  rw [traceCrossMoment]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun i _ => ?_)
  exact abs_integral_le_integral_abs

/-- A pointwise domination `|bᵢ| ≤ ε |cᵢ|` transfers to the trace pairing. This is the step that
turns (A2)'s sup-norm defect into a bound on the remainder of the power count. -/
theorem abs_traceCrossMoment_le_of_pointwise {μ : Measure Ω} {a b c : Ω → Fin d → ℝ} {ε : ℝ}
    (hbc : ∀ ω i, |b ω i| ≤ ε * |c ω i|)
    (hab : ∀ i, Integrable (fun ω => a ω i * b ω i) μ)
    (hac : ∀ i, Integrable (fun ω => a ω i * c ω i) μ) :
    |traceCrossMoment μ a b| ≤ ε * ∑ i, ∫ ω, |a ω i * c ω i| ∂μ := by
  refine le_trans (abs_traceCrossMoment_le μ a b) ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hpt : ∀ ω, |a ω i * b ω i| ≤ ε * |a ω i * c ω i| := by
    intro ω
    rw [abs_mul, abs_mul, ← mul_assoc, mul_comm ε |a ω i|, mul_assoc]
    exact mul_le_mul_of_nonneg_left (hbc ω i) (abs_nonneg _)
  calc ∫ ω, |a ω i * b ω i| ∂μ
      ≤ ∫ ω, ε * |a ω i * c ω i| ∂μ :=
        integral_mono (hab i).abs (((hac i).abs).const_mul ε) hpt
    _ = ε * ∫ ω, |a ω i * c ω i| ∂μ := integral_const_mul _ _

/-- A centred field has zero mean. Stated for a probability measure, since `E_X` is an
expectation; used to discard the mean of the remainder in the bound below. -/
theorem integral_batchFluct_eq_zero [IsProbabilityMeasure μ] {f : Ω → Fin d → ℝ}
    (hf : ∀ i, Integrable (fun ω => f ω i) μ) (i : Fin d) :
    ∫ ω, batchFluct μ f ω i ∂μ = 0 := by
  simp only [batchFluct]
  rw [integral_sub (hf i) (integrable_const _), integral_const]
  simp

/-- Pairing against a centred field does not see the centring of the other argument. This is
what lets the remainder bound be stated in terms of the raw defect field rather than its
fluctuation, so that (A2)'s pointwise hypothesis applies directly. -/
theorem traceCrossMoment_batchFluct_right {μ : Measure Ω} {a r : Ω → Fin d → ℝ}
    (ha0 : ∀ i, ∫ ω, a ω i ∂μ = 0)
    (ha : ∀ i, Integrable (fun ω => a ω i) μ)
    (har : ∀ i, Integrable (fun ω => a ω i * r ω i) μ) :
    traceCrossMoment μ a (batchFluct μ r) = traceCrossMoment μ a r := by
  simp only [traceCrossMoment, batchFluct]
  refine Finset.sum_congr rfl fun i _ => ?_
  have h : ∀ ω, a ω i * (r ω i - ∫ ω', r ω' i ∂μ)
      = a ω i * r ω i - (∫ ω', r ω' i ∂μ) * a ω i := fun ω => by ring
  simp_rw [h]
  rw [integral_sub (har i) ((ha i).const_mul _), integral_const_mul, ha0 i, mul_zero, sub_zero]

/-- **From (A2)'s pointwise defect to a bound on the remainder.** If the true positivity-map prefactor
is within `ε` of the single value `s` at every coordinate and every batch — which is exactly what
the paper's (A2) asserts, stated quantitatively — then the remainder of the power count is at
most `ε` times the mixed absolute first moment of the jitter against the constrained gradient.
Nothing here assumes the remainder vanishes. -/
theorem abs_remainder_le_of_prefactor_defect [IsProbabilityMeasure μ]
    {dth p u : Ω → Fin d → ℝ} {s ε : ℝ}
    (hdefect : ∀ ω i, |p ω i - s| ≤ ε)
    (hdth : ∀ i, Integrable (fun ω => dth ω i) μ)
    (hfd : ∀ i, Integrable (fun ω => batchFluct μ dth ω i) μ)
    (hprod : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * prefactorDefect s p u ω i) μ)
    (hpu : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * u ω i) μ) :
    |traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (prefactorDefect s p u))|
      ≤ ε * ∑ i, ∫ ω, |batchFluct μ dth ω i * u ω i| ∂μ := by
  rw [traceCrossMoment_batchFluct_right (integral_batchFluct_eq_zero hdth) hfd hprod]
  refine abs_traceCrossMoment_le_of_pointwise (fun ω i => ?_) hprod hpu
  rw [prefactorDefect, abs_mul]
  exact mul_le_mul_of_nonneg_right (hdefect ω i) (abs_nonneg _)

/-- The mixed absolute first moment `∑ᵢ E_X|δθ̃ᵢ (∇_θ L)ᵢ|` of the jitter against the constrained
gradient. It is the size against which the (A2) defect is measured. -/
noncomputable def mixedAbsMoment (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) : ℝ :=
  ∑ i, ∫ ω, |batchFluct μ dth ω i * u ω i| ∂μ

/-- **(A2) at relative accuracy, and why the conclusion needs it.** The remainder bound that the
`Θ(s)` law consumes is of the form `|R| ≤ c·s`, and an *absolute* accuracy `|ψ'(θ̃)ᵢ - s| ≤ ε`
with `ε` fixed as the shoulder flattens does not supply one: it gives only `|R| ≤ ε M`, a
constant, which would swamp the `Θ(s)` signal. What supplies it is (A2) read as a *relative*
statement, `|ψ'(θ̃)ᵢ - s| ≤ ε s` — the prefactor field is uniform across coordinates to within a
fixed fraction of its own size, which is what "approximately constant across coordinates" means
for a quantity that is itself tending to zero. Under that reading the remainder is `(ε M)·s`,
with `M` the mixed absolute first moment. Nothing here assumes the remainder vanishes: this
statement converts one explicit bound into another. -/
theorem abs_remainder_le_of_relative_prefactor_defect [IsProbabilityMeasure μ]
    {dth p u : Ω → Fin d → ℝ} {s ε : ℝ}
    (hdefect : ∀ ω i, |p ω i - s| ≤ ε * s)
    (hdth : ∀ i, Integrable (fun ω => dth ω i) μ)
    (hfd : ∀ i, Integrable (fun ω => batchFluct μ dth ω i) μ)
    (hprod : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * prefactorDefect s p u ω i) μ)
    (hpu : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * u ω i) μ) :
    |traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (prefactorDefect s p u))|
      ≤ (ε * mixedAbsMoment μ dth u) * s := by
  have h := abs_remainder_le_of_prefactor_defect (ε := ε * s) hdefect hdth hfd hprod hpu
  have hrw : (ε * mixedAbsMoment μ dth u) * s
      = ε * s * ∑ i, ∫ ω, |batchFluct μ dth ω i * u ω i| ∂μ := by
    rw [mixedAbsMoment]; ring
  rw [hrw]
  exact h

/-- The exact slack channel as a **function of the shoulder prefactor**. At prefactor level `s`
the true coordinatewise positivity-map field is `p s`, so the channel that the asymptotic statements
must be about is `s ↦ tr E_X[δθ̃ δ(p s ⊙ ∇_θ L)ᵀ]`. Holding a single field `p` fixed while `s`
tends to zero would be incompatible with (A2), which pins `p` near `s`; the family is therefore
the honest object, and it is the one `exact_channel_corrected_power` is stated about. -/
noncomputable def jacChannelExactFam (μ : Measure Ω) (dth u : Ω → Fin d → ℝ)
    (p : ℝ → Ω → Fin d → ℝ) (s : ℝ) : ℝ :=
  jacChannelExact μ dth (p s) u

/-- **The two-sided bound with the (A2) defect carried through.** The remainder appears in the
hypothesis as an explicit bound `|R| ≤ c s` and in the conclusion as a widening of both
constants, from `[q₁, q₂]` to `[q₁ - c, q₂ + c]`. Taking `c = 0` recovers the idealized
statement; the point of the theorem is that `c` need not be zero. -/
theorem jacChannelExact_two_sided {μ : Measure Ω} {dth p u : Ω → Fin d → ℝ} {q₁ q₂ c s : ℝ}
    (hu : ∀ i, Integrable (fun ω => u ω i) μ)
    (hr : ∀ i, Integrable (fun ω => prefactorDefect s p u ω i) μ)
    (h1 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (chainRuleGrad s u) ω i) μ)
    (h2 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (prefactorDefect s p u) ω i) μ)
    (hq1 : q₁ ≤ couplingCore μ dth u) (hq2 : couplingCore μ dth u ≤ q₂)
    (hrem : |traceCrossMoment μ (batchFluct μ dth) (batchFluct μ (prefactorDefect s p u))|
      ≤ c * s)
    (hs : 0 < s) :
    (q₁ - c) * s ≤ jacChannelExact μ dth p u ∧ jacChannelExact μ dth p u ≤ (q₂ + c) * s := by
  rw [jacChannelExact_split s hu hr h1 h2]
  obtain ⟨hlo, hhi⟩ := abs_le.mp hrem
  have hA : q₁ * s ≤ s * couplingCore μ dth u := by
    rw [mul_comm q₁ s]
    exact mul_le_mul_of_nonneg_left hq1 hs.le
  have hB : s * couplingCore μ dth u ≤ q₂ * s := by
    rw [mul_comm q₂ s]
    exact mul_le_mul_of_nonneg_left hq2 hs.le
  have e1 : (q₁ - c) * s = q₁ * s - c * s := by ring
  have e2 : (q₂ + c) * s = q₂ * s + c * s := by ring
  rw [e1, e2]
  exact ⟨by linarith, by linarith⟩

/-- **The exact channel, as a function of the prefactor, is trapped between two multiples of
`s`.** The hypotheses are (A2) at relative accuracy on the band together with the non-degeneracy
bound on the prefactor-free coupling functional and the integrability that additivity of the
integral requires; the widening of the constants by `ε M` is the remainder, carried and not
discarded. -/
theorem jacChannelExactFam_two_sided [IsProbabilityMeasure μ]
    {dth u : Ω → Fin d → ℝ} {p : ℝ → Ω → Fin d → ℝ} {q₁ q₂ ε s₀ : ℝ}
    (hq1 : q₁ ≤ couplingCore μ dth u) (hq2 : couplingCore μ dth u ≤ q₂)
    (hdefect : ∀ s : ℝ, 0 < s → s ≤ s₀ → ∀ ω i, |p s ω i - s| ≤ ε * s)
    (hu : ∀ i, Integrable (fun ω => u ω i) μ)
    (hdth : ∀ i, Integrable (fun ω => dth ω i) μ)
    (hfd : ∀ i, Integrable (fun ω => batchFluct μ dth ω i) μ)
    (hpu : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * u ω i) μ)
    (hr : ∀ s : ℝ, ∀ i, Integrable (fun ω => prefactorDefect s (p s) u ω i) μ)
    (h1 : ∀ s : ℝ, ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (chainRuleGrad s u) ω i) μ)
    (h2 : ∀ s : ℝ, ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (prefactorDefect s (p s) u) ω i) μ)
    (hprod : ∀ s : ℝ, ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * prefactorDefect s (p s) u ω i) μ) :
    ∀ s : ℝ, 0 < s → s ≤ s₀ →
      (q₁ - ε * mixedAbsMoment μ dth u) * s ≤ jacChannelExactFam μ dth u p s ∧
        jacChannelExactFam μ dth u p s ≤ (q₂ + ε * mixedAbsMoment μ dth u) * s := by
  intro s hs hs0
  exact jacChannelExact_two_sided hu (hr s) (h1 s) (h2 s) hq1 hq2
    (abs_remainder_le_of_relative_prefactor_defect (hdefect s hs hs0) hdth hfd (hprod s) hpu) hs

/-- **The corrected power, and the divergence, survive the (A2) defect.** Provided the remainder
constant is strictly smaller than the non-degeneracy constant, `c < q₁`, the channel is still
`Θ(s)` and the dimensionless control still diverges. This is the honest form of the paper's
claim: the conclusion is stable under the idealization it was derived from, within an explicitly
quantified margin. -/
theorem corrected_power_survives_remainder {J : ℝ → ℝ} {q₁ q₂ c o s₀ : ℝ}
    (hcq : c < q₁) (ho : 0 < o) (hs₀ : 0 < s₀)
    (hlo : ∀ s : ℝ, 0 < s → s ≤ s₀ → (q₁ - c) * s ≤ J s)
    (hhi : ∀ s : ℝ, 0 < s → s ≤ s₀ → J s ≤ (q₂ + c) * s) :
    (J =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      ((fun s => J s / (s ^ 2 * o)) =Θ[𝓝[>] (0:ℝ)] fun s => s⁻¹) ∧
      Tendsto (fun s => J s / (s ^ 2 * o)) (𝓝[>] (0:ℝ)) atTop :=
  isTheta_and_ratio_tendsto_atTop_of_two_sided_near (by linarith) ho hs₀ hlo hhi

/-- **FW-3b for the exact channel: `Θ(s)`, `Θ(1/s)` and the divergence, with no (A2)
idealization anywhere in the statement.** The channel here is computed from the true
coordinatewise positivity-map field `p s` at every prefactor value, the chain rule is the exact one, and
the only quantitative input about the prefactor field is (A2) read at relative accuracy on the
band. The single smallness condition is that the remainder constant `ε M` be smaller than the
non-degeneracy constant `q₁`; the remainder is never assumed to vanish, and it appears in the
hypotheses of the theorem in the form the bound actually gives it. This is what makes the
correction robust: the `Θ(s)` law is not an artifact of replacing `ψ'(θ̃)` by the scalar `s`. -/
theorem exact_channel_corrected_power [IsProbabilityMeasure μ]
    {dth u : Ω → Fin d → ℝ} {p : ℝ → Ω → Fin d → ℝ} {q₁ q₂ ε s₀ : ℝ}
    (hq1 : q₁ ≤ couplingCore μ dth u) (hq2 : couplingCore μ dth u ≤ q₂)
    (hdefect : ∀ s : ℝ, 0 < s → s ≤ s₀ → ∀ ω i, |p s ω i - s| ≤ ε * s)
    (hu : ∀ i, Integrable (fun ω => u ω i) μ)
    (hdth : ∀ i, Integrable (fun ω => dth ω i) μ)
    (hfd : ∀ i, Integrable (fun ω => batchFluct μ dth ω i) μ)
    (hpu : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * u ω i) μ)
    (hr : ∀ s : ℝ, ∀ i, Integrable (fun ω => prefactorDefect s (p s) u ω i) μ)
    (h1 : ∀ s : ℝ, ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (chainRuleGrad s u) ω i) μ)
    (h2 : ∀ s : ℝ, ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * batchFluct μ (prefactorDefect s (p s) u) ω i) μ)
    (hprod : ∀ s : ℝ, ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * prefactorDefect s (p s) u ω i) μ)
    (hsmall : ε * mixedAbsMoment μ dth u < q₁) (ho : 0 < objNoise μ u) (hs₀ : 0 < s₀) :
    (jacChannelExactFam μ dth u p =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      ((fun s => jacChannelExactFam μ dth u p s / (s ^ 2 * objNoise μ u))
        =Θ[𝓝[>] (0:ℝ)] fun s => s⁻¹) ∧
      Tendsto (fun s => jacChannelExactFam μ dth u p s / (s ^ 2 * objNoise μ u))
        (𝓝[>] (0:ℝ)) atTop := by
  have hb := jacChannelExactFam_two_sided (q₂ := q₂) hq1 hq2 hdefect hu hdth hfd hpu hr h1 h2 hprod
  exact corrected_power_survives_remainder (q₂ := q₂) hsmall ho hs₀
    (fun s hs hs0 => (hb s hs hs0).1) (fun s hs hs0 => (hb s hs hs0).2)

/-! ### The shoulder penalty drops an order -/

/-- The effective bias-channel diffusion `σ_eff² = s²σ_obj² + σ_Jac²`. -/
noncomputable def effDiffusion (μ : Measure Ω) (dth u : Ω → Fin d → ℝ) (s : ℝ) : ℝ :=
  s ^ 2 * objNoise μ u + jacChannel μ dth u s

/-- The Kramers shoulder-penalty exponent `2α/σ²`. -/
noncomputable def shoulderExponent (α σsq : ℝ) : ℝ := 2 * α / σsq

/-- On a bounded band of prefactors the effective diffusion is bounded above and below by
multiples of `s`: the once-attenuated slack channel dominates the twice-attenuated objective
channel as the positivity map flattens, so `σ_eff²` inherits the slack channel's single power. -/
theorem effDiffusion_two_sided {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {q₁ q₂ s₀ s : ℝ}
    (hq1 : q₁ ≤ couplingCore μ dth u) (hq2 : couplingCore μ dth u ≤ q₂)
    (hs : 0 < s) (hs0 : s ≤ s₀) :
    q₁ * s ≤ effDiffusion μ dth u s ∧
      effDiffusion μ dth u s ≤ (s₀ * objNoise μ u + q₂) * s := by
  have ho : 0 ≤ objNoise μ u := objNoise_nonneg μ u
  have hsq : 0 ≤ s ^ 2 * objNoise μ u := mul_nonneg (sq_nonneg s) ho
  have hlow : s * q₁ ≤ s * couplingCore μ dth u := mul_le_mul_of_nonneg_left hq1 hs.le
  have hhigh : s * couplingCore μ dth u ≤ s * q₂ := mul_le_mul_of_nonneg_left hq2 hs.le
  have hquad : s * (s * objNoise μ u) ≤ s₀ * (s * objNoise μ u) :=
    mul_le_mul_of_nonneg_right hs0 (mul_nonneg hs.le ho)
  rw [effDiffusion, jacChannel_eq]
  constructor
  · linarith
  · linarith

/-- **The lift's shoulder penalty is `Θ(1/s)`.** Because the once-attenuated slack channel
dominates the twice-attenuated objective channel as the positivity map flattens, the Arrhenius exponent
`2α/σ_eff²` grows like `1/s` rather than like `1/s²`. Note that this is a statement about the
exponent as a function of the prefactor, not about an escape time: no exit-time theory is used
or proved. -/
theorem liftExponent_isTheta_inv {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {α q₁ q₂ s₀ : ℝ}
    (hα : 0 < α) (hq1 : 0 < q₁) (hq1' : q₁ ≤ couplingCore μ dth u)
    (hq2 : couplingCore μ dth u ≤ q₂) (hs₀ : 0 < s₀) :
    (fun s => shoulderExponent α (effDiffusion μ dth u s)) =Θ[𝓝[>] (0:ℝ)] (fun s => s⁻¹) := by
  set o := objNoise μ u with ho_def
  have ho : 0 ≤ o := objNoise_nonneg μ u
  set K := s₀ * o + q₂ with hK_def
  have hK : 0 < K := by
    have := le_trans hq1' hq2
    nlinarith
  refine isTheta_nhdsGT_zero_of_two_sided_near (c₁ := 2 * α / K) (c₂ := 2 * α / q₁)
    (by positivity) hs₀ (fun _s hs _ => by positivity) (fun s hs hs0 => ?_) (fun s hs hs0 => ?_)
  · obtain ⟨hlo, hhi⟩ := effDiffusion_two_sided hq1' hq2 hs hs0
    have hpos : 0 < effDiffusion μ dth u s := lt_of_lt_of_le (by positivity) hlo
    have heq : 2 * α / K * s⁻¹ = 2 * α / (K * s) := by field_simp
    rw [heq, shoulderExponent, div_le_div_iff₀ (by positivity) hpos]
    nlinarith
  · obtain ⟨hlo, hhi⟩ := effDiffusion_two_sided hq1' hq2 hs hs0
    have hpos : 0 < effDiffusion μ dth u s := lt_of_lt_of_le (by positivity) hlo
    have heq : 2 * α / q₁ * s⁻¹ = 2 * α / (q₁ * s) := by field_simp
    rw [heq, shoulderExponent, div_le_div_iff₀ hpos (by positivity)]
    nlinarith

/-- A batch-independent field has zero fluctuation. -/
theorem batchFluct_eq_zero_of_const [IsProbabilityMeasure μ] {f : Ω → Fin d → ℝ}
    (hconst : ∀ ω ω' i, f ω i = f ω' i) : batchFluct μ f = 0 := by
  funext ω i
  simp only [batchFluct]
  have hmean : ∫ ω', f ω' i ∂μ = f ω i := by
    rw [show (fun ω' => f ω' i) = (fun _ => f ω i) from funext fun ω' => hconst ω' ω i,
      integral_const]
    simp
  rw [hmean]
  simp

/-- The direct parametrization has no slack channel: its latent iterate is a free parameter
independent of the batch, so `δθ̃ ≡ 0` and `σ_Jac²` vanishes identically at every prefactor. -/
theorem jacChannel_eq_zero_of_const [IsProbabilityMeasure μ] {dth u : Ω → Fin d → ℝ}
    (hconst : ∀ ω ω' i, dth ω i = dth ω' i) (s : ℝ) : jacChannel μ dth u s = 0 := by
  rw [jacChannel, batchFluct_eq_zero_of_const hconst, traceCrossMoment]
  simp

/-- Hence the direct baseline's effective diffusion is the twice-attenuated objective channel
alone, with no `Θ(s)` term available to it. -/
theorem effDiffusion_direct [IsProbabilityMeasure μ] {dth u : Ω → Fin d → ℝ}
    (hconst : ∀ ω ω' i, dth ω i = dth ω' i) (s : ℝ) :
    effDiffusion μ dth u s = s ^ 2 * objNoise μ u := by
  rw [effDiffusion, jacChannel_eq_zero_of_const hconst, add_zero]

/-- **The direct baseline's shoulder penalty is `Θ(1/s²)`.** The direct parametrization has
`δθ̃ ≡ 0`, hence no slack channel at all, so its effective diffusion is the twice-attenuated
objective channel alone and its exponent grows one order faster. -/
theorem directExponent_isTheta_inv_sq {μ : Measure Ω} {u : Ω → Fin d → ℝ} {α : ℝ}
    (hα : 0 < α) (ho : 0 < objNoise μ u) :
    (fun s => shoulderExponent α (s ^ 2 * objNoise μ u)) =Θ[𝓝[>] (0:ℝ)]
      (fun s => (s ^ 2)⁻¹) := by
  refine isTheta_nhdsGT_zero_of_two_sided (c₁ := 2 * α / objNoise μ u)
    (c₂ := 2 * α / objNoise μ u) (by positivity) (fun _s _hs => by positivity)
    (fun s hs => ?_) (fun s hs => ?_)
  · rw [shoulderExponent]
    have h : 2 * α / objNoise μ u * (s ^ 2)⁻¹ = 2 * α / (s ^ 2 * objNoise μ u) := by field_simp
    rw [h]
  · rw [shoulderExponent]
    have h : 2 * α / objNoise μ u * (s ^ 2)⁻¹ = 2 * α / (s ^ 2 * objNoise μ u) := by field_simp
    rw [h]

/-- **The same statement, written about the direct baseline's own effective diffusion.** For a
batch-independent latent iterate the slack channel vanishes identically, so `σ_eff²` *is*
`s² σ_obj²` and the `Θ(1/s²)` law is a statement about the baseline itself rather than about an
expression that happens to coincide with it. -/
theorem directExponent_effDiffusion_isTheta_inv_sq [IsProbabilityMeasure μ]
    {dth u : Ω → Fin d → ℝ} {α : ℝ} (hconst : ∀ ω ω' i, dth ω i = dth ω' i)
    (hα : 0 < α) (ho : 0 < objNoise μ u) :
    (fun s => shoulderExponent α (effDiffusion μ dth u s)) =Θ[𝓝[>] (0:ℝ)]
      (fun s => (s ^ 2)⁻¹) := by
  have h : (fun s => shoulderExponent α (effDiffusion μ dth u s))
      = fun s => shoulderExponent α (s ^ 2 * objNoise μ u) :=
    funext fun s => by rw [effDiffusion_direct hconst]
  rw [h]
  exact directExponent_isTheta_inv_sq hα ho

/-- The ratio of the direct baseline's shoulder exponent to the lift's is exactly `1 + ϱ`. The
barrier `α` cancels, which is the formal content of the paper's convention that `α` is held
fixed across the comparison. -/
theorem exponentRatio_eq {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {α s : ℝ}
    (hα : α ≠ 0) (hs : s ≠ 0) (ho : objNoise μ u ≠ 0)
    (heff : effDiffusion μ dth u s ≠ 0) :
    shoulderExponent α (s ^ 2 * objNoise μ u) / shoulderExponent α (effDiffusion μ dth u s)
      = 1 + couplingRatio μ dth u s := by
  have hE : effDiffusion μ dth u s = s ^ 2 * objNoise μ u + jacChannel μ dth u s := rfl
  have hden : s ^ 2 * objNoise μ u ≠ 0 := mul_ne_zero (pow_ne_zero 2 hs) ho
  simp only [shoulderExponent, couplingRatio]
  rw [hE] at heff ⊢
  field_simp

/-- **The lift lowers the order of the shoulder penalty.** The ratio of the direct exponent to
the lifted exponent is exactly `1 + ϱ`, and it diverges as the positivity map flattens. So the lift does
not merely improve the constant in the exponent: at the corrected power it reduces the exponent's
order in the prefactor, from `Θ(1/s²)` to `Θ(1/s)`. This is the framing the design note
recommends for the tex, and it is stronger than the claim the word "unattenuated" was making. -/
theorem exponentRatio_tendsto_atTop {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {α : ℝ}
    (hα : α ≠ 0) (hq : 0 < couplingCore μ dth u) (ho : 0 < objNoise μ u) :
    Tendsto (fun s => shoulderExponent α (s ^ 2 * objNoise μ u) /
        shoulderExponent α (effDiffusion μ dth u s)) (𝓝[>] (0:ℝ)) atTop := by
  have hbase : Tendsto (fun s => 1 + couplingRatio μ dth u s) (𝓝[>] (0:ℝ)) atTop :=
    Filter.tendsto_atTop_add_const_left _ 1 (couplingRatio_tendsto_atTop hq ho)
  refine hbase.congr' (Filter.eventually_of_mem self_mem_nhdsWithin fun s hs => ?_)
  have hs' : (0:ℝ) < s := hs
  have heffpos : 0 < effDiffusion μ dth u s := by
    rw [effDiffusion, jacChannel_eq]
    have h1 : 0 < s * couplingCore μ dth u := mul_pos hs' hq
    have h2 : 0 ≤ s ^ 2 * objNoise μ u := mul_nonneg (sq_nonneg s) ho.le
    linarith
  exact (exponentRatio_eq hα hs'.ne' ho.ne' heffpos.ne').symm

/-! ### Where the non-degeneracy hypothesis comes from -/

/-- The iterate-fluctuation covariance `V = E_X[δθ̃ δθ̃ᵀ]` is positive semidefinite. -/
theorem crossMomentMatrix_self_posSemidef {μ : Measure Ω} (a : Ω → Fin d → ℝ)
    (hint : ∀ i j, Integrable (fun ω => a ω i * a ω j) μ) :
    (crossMomentMatrix μ a a).PosSemidef := by
  have hint2 : ∀ (x : Fin d → ℝ) (i j : Fin d),
      Integrable (fun ω => x i * a ω i * (x j * a ω j)) μ := by
    intro x i j
    have h := (hint i j).const_mul (x i * x j)
    refine h.congr (Filter.Eventually.of_forall fun ω => ?_)
    ring
  rw [Matrix.posSemidef_iff_dotProduct_mulVec]
  refine ⟨?_, fun x => ?_⟩
  · ext i j
    simp only [Matrix.conjTranspose_apply, star_trivial, crossMomentMatrix, Matrix.of_apply]
    exact integral_congr_ae (Filter.Eventually.of_forall fun ω => mul_comm _ _)
  · have hkey : star x ⬝ᵥ (crossMomentMatrix μ a a) *ᵥ x
        = ∫ ω, (∑ i, x i * a ω i) ^ 2 ∂μ := by
      have hstep : ∀ i : Fin d,
          x i * ∑ j, (∫ ω, a ω i * a ω j ∂μ) * x j
            = ∑ j, ∫ ω, x i * a ω i * (x j * a ω j) ∂μ := by
        intro i
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun j _ => ?_
        have h1 : x i * ((∫ ω, a ω i * a ω j ∂μ) * x j)
            = (x i * x j) * ∫ ω, a ω i * a ω j ∂μ := by ring
        rw [h1, ← integral_const_mul]
        exact integral_congr_ae (Filter.Eventually.of_forall fun ω => by ring)
      have hswap : ∫ ω, ∑ i, ∑ j, x i * a ω i * (x j * a ω j) ∂μ
          = ∑ i, ∑ j, ∫ ω, x i * a ω i * (x j * a ω j) ∂μ := by
        rw [integral_finsetSum _
          (fun i _ => integrable_finsetSum _ fun j _ => hint2 x i j)]
        exact Finset.sum_congr rfl fun i _ => integral_finsetSum _ fun j _ => hint2 x i j
      have hsq : ∀ ω, (∑ i, x i * a ω i) ^ 2 = ∑ i, ∑ j, x i * a ω i * (x j * a ω j) := by
        intro ω
        rw [sq, Finset.sum_mul_sum]
      simp only [dotProduct, Matrix.mulVec, crossMomentMatrix, Matrix.of_apply, star_trivial]
      simp_rw [hsq]
      rw [hswap]
      exact Finset.sum_congr rfl fun i _ => hstep i
    rw [hkey]
    exact integral_nonneg fun ω => sq_nonneg _

/-- The `s`-free coupling functional evaluated on (A4)'s forward-KL leading term:
`tr E_X[δθ̃ (H̄ δθ̃)ᵀ] = tr(V H̄)` with `V = E_X[δθ̃ δθ̃ᵀ]`, for symmetric `H̄`. -/
theorem traceCrossMoment_mulVec {μ : Measure Ω} {a : Ω → Fin d → ℝ}
    {H : Matrix (Fin d) (Fin d) ℝ} (hH : ∀ i j, H i j = H j i)
    (hint : ∀ i j, Integrable (fun ω => a ω i * a ω j) μ) :
    traceCrossMoment μ a (fun ω => H *ᵥ a ω)
      = ((crossMomentMatrix μ a a) * H).trace := by
  have hint3 : ∀ i j : Fin d, Integrable (fun ω => H i j * (a ω i * a ω j)) μ :=
    fun i j => (hint i j).const_mul _
  have hrow : ∀ i : Fin d,
      ∫ ω, a ω i * (H *ᵥ a ω) i ∂μ = ∑ j, H i j * ∫ ω, a ω i * a ω j ∂μ := by
    intro i
    have hpt : ∀ ω, a ω i * (H *ᵥ a ω) i = ∑ j, H i j * (a ω i * a ω j) := by
      intro ω
      simp only [Matrix.mulVec, dotProduct, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    simp_rw [hpt]
    rw [integral_finsetSum _ fun j _ => hint3 i j]
    exact Finset.sum_congr rfl fun j _ => integral_const_mul _ _
  rw [traceCrossMoment]
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, crossMomentMatrix, Matrix.of_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [hrow i]
  exact Finset.sum_congr rfl fun j _ => by rw [hH i j, mul_comm]

/-- The sign of the leading coupling term comes for free from the positive-semidefinite cone. -/
theorem trace_crossMomentMatrix_mul_nonneg {μ : Measure Ω} {a : Ω → Fin d → ℝ}
    {H : Matrix (Fin d) (Fin d) ℝ} (hH : H.PosSemidef)
    (hint : ∀ i j, Integrable (fun ω => a ω i * a ω j) μ) :
    0 ≤ ((crossMomentMatrix μ a a) * H).trace :=
  trace_mul_nonneg_of_posSemidef (crossMomentMatrix_self_posSemidef a hint) hH

/-- (A4)'s forward-KL chain `δ(∇_θL) = H̄ δθ̃ + r`, substituted into the `s`-free coupling
functional: `q = tr(V H̄) + tr E_X[δθ̃ rᵀ]`. The remainder is exhibited, not dropped. -/
theorem couplingCore_eq_trace_add_remainder {μ : Measure Ω} {dth u r : Ω → Fin d → ℝ}
    {H : Matrix (Fin d) (Fin d) ℝ} (hH : ∀ i j, H i j = H j i)
    (hchain : ∀ ω i, batchFluct μ u ω i = (H *ᵥ batchFluct μ dth ω) i + r ω i)
    (hint : ∀ i j, Integrable (fun ω => batchFluct μ dth ω i * batchFluct μ dth ω j) μ)
    (h1 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * (H *ᵥ batchFluct μ dth ω) i) μ)
    (h2 : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * r ω i) μ) :
    couplingCore μ dth u
      = (crossMomentMatrix μ (batchFluct μ dth) (batchFluct μ dth) * H).trace
        + traceCrossMoment μ (batchFluct μ dth) r := by
  have hb : batchFluct μ u = fun ω i => (H *ᵥ batchFluct μ dth ω) i + r ω i := by
    funext ω i
    exact hchain ω i
  rw [couplingCore, hb,
    traceCrossMoment_add_right (b := fun ω => H *ᵥ batchFluct μ dth ω) (c := r) h1 h2,
    traceCrossMoment_mulVec hH hint]

/-- **"The sign comes for free."** Under (A4)'s forward-KL chain with a positive-semidefinite
mean Hessian, the coupling functional splits as `tr(V H̄) + tr E_X[δθ̃ rᵀ]` and its leading term
is non-negative by self-duality of the positive-semidefinite cone, so the whole of `q` is bounded
below by minus the size of the linearization remainder — with no gap hypothesis, no lower bound
on the curvature, and no non-degeneracy assumed anywhere. This is the design note's observation
in the only form in which positive-semidefiniteness of `H̄` does mathematical work on its own;
strict positivity, in the next statement, is carried by the gap hypothesis instead. -/
theorem couplingCore_ge_neg_abs_remainder {μ : Measure Ω} {dth u r : Ω → Fin d → ℝ}
    {H : Matrix (Fin d) (Fin d) ℝ} (hHpsd : H.PosSemidef)
    (hchain : ∀ ω i, batchFluct μ u ω i = (H *ᵥ batchFluct μ dth ω) i + r ω i)
    (hint : ∀ i j, Integrable (fun ω => batchFluct μ dth ω i * batchFluct μ dth ω j) μ)
    (h1 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * (H *ᵥ batchFluct μ dth ω) i) μ)
    (h2 : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * r ω i) μ) :
    -|traceCrossMoment μ (batchFluct μ dth) r| ≤ couplingCore μ dth u := by
  have hHsym : ∀ i j, H i j = H j i := fun i j => by
    have h := hHpsd.isHermitian.apply i j
    simpa using h.symm
  have hnn := trace_crossMomentMatrix_mul_nonneg (a := batchFluct μ dth) hHpsd hint
  have hab := neg_abs_le (traceCrossMoment μ (batchFluct μ dth) r)
  rw [couplingCore_eq_trace_add_remainder hHsym hchain hint h1 h2]
  linarith

/-- **The positivity half of the non-degeneracy hypothesis, reduced to (A4).** Under (A4)'s
forward-KL chain `δ(∇_θL) = H̄ δθ̃ + r` the coupling functional splits as
`tr(V H̄) + tr E_X[δθ̃ rᵀ]`, and strict positivity of `q` follows from the gap hypothesis. Be
exact about the division of labour, because it is easy to overstate. The positive-semidefinite
structure of `H̄` buys the *non-negativity* of the leading term, and that is the content of
`couplingCore_ge_neg_abs_remainder` above; in the present proof it is used only for the symmetry
of `H̄`, and the strictness is carried entirely by `hgap`. The gap
`|tr E_X[δθ̃ rᵀ]| < tr(V H̄)` is a remainder-control clause in the shape of (A4'-q), and it
forces `tr(V H̄) > 0` — the curvature must act non-degenerately on the jitter covariance, which
is the paper's full-rank slack-Jacobian clause. That content is stated here, not hidden. The
upper bound `q ≤ q₂` and the uniformity of both constants over the band are separate moment
hypotheses and are not discharged here. -/
theorem couplingCore_pos_of_forwardKL {μ : Measure Ω} {dth u r : Ω → Fin d → ℝ}
    {H : Matrix (Fin d) (Fin d) ℝ} (hHpsd : H.PosSemidef)
    (hchain : ∀ ω i, batchFluct μ u ω i = (H *ᵥ batchFluct μ dth ω) i + r ω i)
    (hint : ∀ i j, Integrable (fun ω => batchFluct μ dth ω i * batchFluct μ dth ω j) μ)
    (h1 : ∀ i, Integrable
      (fun ω => batchFluct μ dth ω i * (H *ᵥ batchFluct μ dth ω) i) μ)
    (h2 : ∀ i, Integrable (fun ω => batchFluct μ dth ω i * r ω i) μ)
    (hgap : |traceCrossMoment μ (batchFluct μ dth) r|
      < (crossMomentMatrix μ (batchFluct μ dth) (batchFluct μ dth) * H).trace) :
    0 < couplingCore μ dth u := by
  have hHsym : ∀ i j, H i j = H j i := fun i j => by
    have h := hHpsd.isHermitian.apply i j
    simpa using h.symm
  rw [couplingCore_eq_trace_add_remainder hHsym hchain hint h1 h2]
  have hlt := (abs_lt.mp hgap).1
  linarith

/-! ### FW-3b, assembled -/

/-- **Uniformity over the shoulder band.** The constants `q₁` and `q₂` do not depend on the band
point, which is what "bounded away from zero on the band" means and what the downstream
Freidlin–Wentzell comparison consumes. The band is left abstract: no geometry of the shoulder is
used, only that the same two constants work at every point of it. -/
theorem jacChannel_two_sided_uniform_on_band {B : Type*} (band : Set B) {μ : Measure Ω}
    {dth u : B → Ω → Fin d → ℝ} {q₁ q₂ : ℝ}
    (hband : ∀ x ∈ band,
      q₁ ≤ couplingCore μ (dth x) (u x) ∧ couplingCore μ (dth x) (u x) ≤ q₂) :
    ∀ x ∈ band, ∀ s : ℝ, 0 < s →
      q₁ * s ≤ jacChannel μ (dth x) (u x) s ∧ jacChannel μ (dth x) (u x) s ≤ q₂ * s :=
  fun x hx _s hs => jacChannel_two_sided (hband x hx).1 (hband x hx).2 hs

/-- **The whole of FW-3b in one statement.** Under the non-degeneracy hypothesis and a positive
objective-noise variance: the objective channel carries the prefactor twice and the slack channel
once, the slack channel is trapped between `q₁ s` and `q₂ s` and is `Θ(s)` but not `Θ(1)`, and
the dimensionless control is `Θ(1/s)` and diverges. This is the statement the author needs in
hand before editing the misplaced occurrences of the word — the count is corrected and every
downstream conclusion is preserved. -/
theorem power_of_s_correction {μ : Measure Ω} {dth u : Ω → Fin d → ℝ} {q₁ q₂ : ℝ}
    (hq₁ : 0 < q₁) (hlo : q₁ ≤ couplingCore μ dth u) (hhi : couplingCore μ dth u ≤ q₂)
    (ho : 0 < objNoise μ u) :
    (∀ s : ℝ, objChannel μ u s = s * (s * objNoise μ u)) ∧
      (∀ s : ℝ, jacChannel μ dth u s = s * couplingCore μ dth u) ∧
      (∀ s : ℝ, 0 < s → q₁ * s ≤ jacChannel μ dth u s ∧ jacChannel μ dth u s ≤ q₂ * s) ∧
      ((fun s => jacChannel μ dth u s) =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      (¬ ((fun s => jacChannel μ dth u s) =Θ[𝓝[>] (0:ℝ)] fun _ => (1:ℝ))) ∧
      ((fun s => couplingRatio μ dth u s) =Θ[𝓝[>] (0:ℝ)] fun s => s⁻¹) ∧
      Tendsto (fun s => couplingRatio μ dth u s) (𝓝[>] (0:ℝ)) atTop := by
  have hq : 0 < couplingCore μ dth u := lt_of_lt_of_le hq₁ hlo
  refine ⟨fun s => by rw [objChannel_eq]; ring, fun s => jacChannel_eq μ dth u s,
    fun s hs => jacChannel_two_sided hlo hhi hs, jacChannel_isTheta_id hq,
    jacChannel_not_isTheta_one μ dth u, couplingRatio_isTheta_inv hq ho,
    couplingRatio_tendsto_atTop hq ho⟩

/-! ### The hypotheses are jointly satisfiable: an explicit witness

Every statement above is conditional, and several carry a dozen hypotheses at once. A referee is
entitled to ask whether any batch law satisfies all of them simultaneously, since a theorem whose
hypotheses are unsatisfiable is true and worthless. This section answers that by exhibiting one.
The witness is deliberately minimal — a two-point batch law on one constrained coordinate — but
it is not degenerate in any of the ways that would make the exhibition hollow: the latent weight
genuinely fluctuates, the constrained gradient has a non-zero mean so that centring does real
work, the coupling functional and the objective-noise variance take *different* positive values,
the slack genuinely moves with the prefactor so that `jitterPower_const` is not a tautology on
it, and the positivity-map prefactor field is genuinely non-constant across the two batches, so the (A2)
remainder carried through `exact_channel_corrected_power` is non-zero.
-/

/-- A two-point batch law with equiprobable draws. -/
noncomputable def witMeasure : Measure Bool := (PMF.uniformOfFintype Bool).toMeasure

instance isProbabilityMeasure_witMeasure : IsProbabilityMeasure witMeasure := by
  unfold witMeasure; infer_instance

/-- Expectation against the two-point law is the average of the two values. -/
theorem witMeasure_integral (f : Bool → ℝ) :
    ∫ ω, f ω ∂witMeasure = (f false + f true) / 2 := by
  rw [witMeasure, PMF.integral_eq_sum]
  simp [PMF.uniformOfFintype_apply]
  ring

/-- The latent weight body `h_φ(X)`, a genuinely batch-dependent field of mean zero. -/
def witBody : Bool → Fin 1 → ℝ := fun ω _ => if ω then 1 else -1

/-- The constrained gradient `∇_θ L`. Its mean is non-zero, so the centring in `batchFluct` is
not a no-op, and its fluctuation is twice the body's, so the coupling functional and the
objective-noise variance come out at different values. -/
def witGrad : Bool → Fin 1 → ℝ := fun ω _ => if ω then 3 else -1

/-- The slack `b_s`, which moves with the shoulder prefactor. -/
def witSlack : ℝ → Fin 1 → ℝ := fun s _ => s

/-- The latent iterate `θ̃_s = b_s + h_φ(X)`. -/
def witIterate : ℝ → Bool → Fin 1 → ℝ := fun s ω i => witSlack s i + witBody ω i

/-- The true coordinatewise positivity-map prefactor field at prefactor level `s`: within ten percent of
`s` at every coordinate and every batch, but not equal to it, so that the (A2) defect it induces
is genuinely non-zero. -/
noncomputable def witPrefactor : ℝ → Bool → Fin 1 → ℝ :=
  fun s ω _ => s + s * (if ω then 1 else -1) / 10

theorem witBody_integrable (i : Fin 1) : Integrable (fun ω => witBody ω i) witMeasure :=
  Integrable.of_finite

theorem batchFluct_witBody : batchFluct witMeasure witBody = witBody := by
  funext ω i
  have h0 : ∫ ω', witBody ω' i ∂witMeasure = 0 := by
    rw [witMeasure_integral]; simp [witBody]
  rw [batchFluct, h0, sub_zero]

theorem batchFluct_witGrad :
    batchFluct witMeasure witGrad = fun ω i => 2 * witBody ω i := by
  funext ω i
  have h0 : ∫ ω', witGrad ω' i ∂witMeasure = 1 := by
    rw [witMeasure_integral]; simp [witGrad]; norm_num
  rw [batchFluct, h0]
  cases ω <;> simp [witGrad, witBody] <;> norm_num

theorem batchFluct_witIterate (s : ℝ) :
    batchFluct witMeasure (witIterate s) = witBody :=
  (batchFluct_slack_plus_body witBody_integrable (fun _ _ => rfl)).trans batchFluct_witBody

/-- The prefactor-free coupling functional of the witness: `q = 2`. -/
theorem witMeasure_couplingCore : couplingCore witMeasure witBody witGrad = 2 := by
  rw [couplingCore, batchFluct_witBody, batchFluct_witGrad, traceCrossMoment]
  rw [Finset.sum_congr rfl
    (fun i _ => witMeasure_integral (fun ω => witBody ω i * (2 * witBody ω i)))]
  simp [witBody]

/-- The objective-noise variance of the witness: `σ_obj² = 4`, different from `q`. -/
theorem witMeasure_objNoise : objNoise witMeasure witGrad = 4 := by
  rw [objNoise, batchFluct_witGrad, traceCrossMoment]
  rw [Finset.sum_congr rfl
    (fun i _ => witMeasure_integral (fun ω => 2 * witBody ω i * (2 * witBody ω i)))]
  simp [witBody]
  norm_num

/-- The jitter power of the witness: `E_X‖δθ̃‖² = 1`, at every prefactor. -/
theorem witMeasure_jitterPower (s : ℝ) : jitterPower witMeasure witIterate s = 1 := by
  rw [jitterPower, batchFluct_witIterate, traceCrossMoment]
  rw [Finset.sum_congr rfl (fun i _ => witMeasure_integral (fun ω => witBody ω i * witBody ω i))]
  simp [witBody]

/-- The witness exercises the hypothesis of `jitterPower_const` rather than trivializing it: the
latent iterate really does depend on the prefactor, and it is only its fluctuation that does
not. -/
theorem witIterate_depends_on_prefactor : witIterate 0 ≠ witIterate 1 := by
  intro h
  have h' := congrFun (congrFun h true) 0
  simp [witIterate, witSlack, witBody] at h'

/-- The witness prefactor field satisfies (A2) at relative accuracy `ε = 1/10`. -/
theorem witPrefactor_defect (s : ℝ) (hs : 0 < s) (ω : Bool) (i : Fin 1) :
    |witPrefactor s ω i - s| ≤ (1/10) * s := by
  rw [abs_le]
  constructor <;> · simp only [witPrefactor]; split_ifs <;> linarith

/-- The mixed absolute first moment of the witness: `M = 2`. -/
theorem witMeasure_mixedAbsMoment : mixedAbsMoment witMeasure witBody witGrad = 2 := by
  rw [mixedAbsMoment, batchFluct_witBody]
  rw [Finset.sum_congr rfl
    (fun i _ => witMeasure_integral (fun ω => |witBody ω i * witGrad ω i|))]
  simp [witBody, witGrad]
  norm_num

/-- The witness's jitter covariance paired against the mean Hessian `H̄ = 2 I`: `tr(V H̄) = 2`. -/
theorem witMeasure_gramTrace :
    ((crossMomentMatrix witMeasure (batchFluct witMeasure witBody)
        (batchFluct witMeasure witBody)) * Matrix.diagonal (fun _ : Fin 1 => (2:ℝ))).trace
      = 2 := by
  rw [batchFluct_witBody]
  simp [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.diagonal, crossMomentMatrix]
  rw [witMeasure_integral (fun ω => witBody ω 0 * witBody ω 0)]
  simp [witBody]

/-- The witness satisfies (A4)'s forward-KL chain exactly, with mean Hessian `H̄ = 2 I` and zero
linearization remainder. -/
theorem witMeasure_chain (ω : Bool) (i : Fin 1) :
    batchFluct witMeasure witGrad ω i
      = (Matrix.diagonal (fun _ : Fin 1 => (2:ℝ)) *ᵥ batchFluct witMeasure witBody ω) i
        + (fun (_ : Bool) (_ : Fin 1) => (0:ℝ)) ω i := by
  rw [batchFluct_witBody, batchFluct_witGrad]
  simp [Matrix.mulVec_diagonal]

/-- **The forward-KL reduction is not vacuous.** Every hypothesis of
`couplingCore_pos_of_forwardKL` — the positive-semidefinite mean Hessian, the chain, the
integrability, and the remainder gap — holds at the witness. -/
theorem couplingCore_pos_of_forwardKL_at_witness :
    0 < couplingCore witMeasure witBody witGrad :=
  couplingCore_pos_of_forwardKL (H := Matrix.diagonal (fun _ : Fin 1 => (2:ℝ)))
    (r := fun _ _ => 0) (Matrix.PosSemidef.diagonal (by intro i; norm_num))
    witMeasure_chain (fun _ _ => Integrable.of_finite) (fun _ => Integrable.of_finite)
    (fun _ => Integrable.of_finite)
    (by rw [witMeasure_gramTrace]; simp [traceCrossMoment])

/-- **FW-3b is not vacuous.** `power_of_s_correction` instantiated at the witness, with
`q₁ = q₂ = q = 2` and `σ_obj² = 4`. -/
theorem power_of_s_correction_at_witness :
    (∀ s : ℝ, objChannel witMeasure witGrad s = s * (s * objNoise witMeasure witGrad)) ∧
      (∀ s : ℝ, jacChannel witMeasure witBody witGrad s
        = s * couplingCore witMeasure witBody witGrad) ∧
      (∀ s : ℝ, 0 < s → 2 * s ≤ jacChannel witMeasure witBody witGrad s ∧
        jacChannel witMeasure witBody witGrad s ≤ 2 * s) ∧
      ((fun s => jacChannel witMeasure witBody witGrad s) =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      (¬ ((fun s => jacChannel witMeasure witBody witGrad s)
        =Θ[𝓝[>] (0:ℝ)] fun _ => (1:ℝ))) ∧
      ((fun s => couplingRatio witMeasure witBody witGrad s) =Θ[𝓝[>] (0:ℝ)] fun s => s⁻¹) ∧
      Tendsto (fun s => couplingRatio witMeasure witBody witGrad s) (𝓝[>] (0:ℝ)) atTop :=
  power_of_s_correction (by norm_num) witMeasure_couplingCore.ge witMeasure_couplingCore.le
    (by rw [witMeasure_objNoise]; norm_num)

/-- **The side-by-side contrast is not vacuous.** -/
theorem unattenuated_jitter_but_once_attenuated_channel_at_witness :
    ((fun s => jitterPower witMeasure witIterate s) =Θ[𝓝[>] (0:ℝ)] fun _ => (1:ℝ)) ∧
      ((fun s => jacChannel witMeasure (witIterate s) witGrad s)
        =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      ¬ ((fun s => jacChannel witMeasure (witIterate s) witGrad s)
        =Θ[𝓝[>] (0:ℝ)] fun _ => (1:ℝ)) :=
  unattenuated_jitter_but_once_attenuated_channel (b := witSlack) (h := witBody)
    witBody_integrable (fun _ _ _ => rfl)
    (by rw [witMeasure_jitterPower]; norm_num)
    (by rw [witMeasure_couplingCore]; norm_num)

/-- **The exact-channel statement is not vacuous, and its (A2) remainder is not zero.** All
fourteen hypotheses of `exact_channel_corrected_power` hold at the witness, with a prefactor
field that deviates from the scalar `s` by a full ten percent at every batch. -/
theorem exact_channel_corrected_power_at_witness :
    (jacChannelExactFam witMeasure witBody witGrad witPrefactor =Θ[𝓝[>] (0:ℝ)] fun s => s) ∧
      ((fun s => jacChannelExactFam witMeasure witBody witGrad witPrefactor s /
          (s ^ 2 * objNoise witMeasure witGrad)) =Θ[𝓝[>] (0:ℝ)] fun s => s⁻¹) ∧
      Tendsto (fun s => jacChannelExactFam witMeasure witBody witGrad witPrefactor s /
          (s ^ 2 * objNoise witMeasure witGrad)) (𝓝[>] (0:ℝ)) atTop :=
  exact_channel_corrected_power (q₁ := 2) (q₂ := 2) (ε := 1/10) (s₀ := 1)
    witMeasure_couplingCore.ge witMeasure_couplingCore.le
    (fun s hs _ ω i => witPrefactor_defect s hs ω i)
    (fun _ => Integrable.of_finite) (fun _ => Integrable.of_finite)
    (fun _ => Integrable.of_finite) (fun _ => Integrable.of_finite)
    (fun _ _ => Integrable.of_finite) (fun _ _ => Integrable.of_finite)
    (fun _ _ => Integrable.of_finite) (fun _ _ => Integrable.of_finite)
    (by rw [witMeasure_mixedAbsMoment]; norm_num)
    (by rw [witMeasure_objNoise]; norm_num) one_pos

end IcnnLift
