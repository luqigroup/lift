import Mathlib
import TraceLemmas
import CrossCov

/-!
# The regularity assumptions (A1)-(A6), and the ledger of which result carries which

The appendix of the paper (`docs/paper/v4/A1_proofs.tex`, subsection "Regularity assumptions")
states five hypotheses (A1)-(A5) in one dense paragraph, and closes it with an attribution:
"\cref{thm:joint-necessity} uses only (A1); \cref{lem:moreau} uses (A1), (A3), (A4);
\cref{cor:fpt} uses (A1)--(A5)". The design note `docs/design/lemma1_rederivation.md`, which
replaces the refuted Hessian route of the shipped Lemma 1 by the state-dependent-diffusion route,
adds two further hypotheses in the course of the rederivation and labels them as such: the
remainder-control clause (A4'-q) of FW-1 (section 8) and the positive-alignment clause (A6) of
FW-2' (section 11.3). This module makes all seven machine-readable.

Each assumption becomes a Lean structure whose fields are the formalizable clauses, stated at the
strength the paper states them and not a notch stronger. Two choices are worth naming, because
they are where an assumption module can quietly cheat. First, **(A2) is carried with its
tolerance**: the paper says the positivity-map derivative is "approximately constant" on the operative
band, and `SinglePrefactor` says exactly that, `|psi'(w) - s| <= eps` for `w` on the band, with
`eps` a parameter that propagates into every consequence as the width of a two-sided bracket.
Replacing it by the exact identity `psi' = s` would not be a simplification but a different and
false hypothesis: `SinglePrefactor.subsingleton_of_tol_zero` shows that for a positivity map with
injective derivative -- softplus, whose derivative is the strictly increasing logistic, is one --
the exact version forces the band to be a single point. Second, **(A4) is carried with its
remainder**: `ForwardKLChain` asserts the forward-KL chain `dg = H dtheta + r` together with an
explicit quadratic bound `‖r‖ <= Cr ‖dtheta‖²`, and the remainder appears in the conclusion of
every consequence, as an explicit budget `Cr eps³` (`ForwardKLChain.remainder_budget`) or as an
explicit modulus `M Cr eps³` for (A4'-q) (`remainderControl_of_forwardKLChain`). At no point is a
remainder assumed to vanish.

A third discipline is worth naming because an assumption module is where vacuity hides. A
Lean structure whose fields are mutually incompatible makes every theorem that carries it true
and useless. Each of the assumptions whose fields interact is therefore accompanied by a witness.
`exists_singlePrefactor_band` builds (A2) around any regular point, and `SoftplusBand.lean`
builds it for the softplus positivity map itself; `forwardKLChain_collinear` builds (A4) with a nonzero
remainder that saturates the quadratic bound at `Cr = 1`; `singleBarrierPotential_quadratic` and
`smallEffectiveNoise_of_ratio` build the two clauses of (A5); `couplingAlignedOn_smul_one` builds
(A6)'s inequality; and `smallJitter_hypotheses_satisfiable` builds the one configuration that a
reader has real grounds to doubt, in which (A4)'s jitter bound and the non-degeneracy of the
magnitude channel *at the scale of that jitter* hold simultaneously with a nonzero remainder.

One clause of (A1) has no field, and the omission is deliberate rather than an oversight. "SDE of
SGD with `eta` small" asserts that a continuous-time Itô process is an adequate surrogate for the
discrete stochastic-gradient recursion. That is a modelling register: it relates two objects only
one of which exists in this development, and mathlib v4.31.0 has no Itô integral, no stochastic
differential equation and no weak-convergence theory for them. `SdeOfSgdModel` therefore carries
the learning rate, the i.i.d.-batch clause and the Lipschitz-covariance clause, and its docstring
says plainly that the surrogacy claim is proved nowhere in this development. Every module on the
diffusion side takes the diffusion as its primitive.

What this module proves, beyond bookkeeping, is the set of small bridges that let a referee check
the ledger rather than take it on trust: that (A1)'s i.i.d. clause supplies exactly two of the
four fields of `CrossCovSLLN.IsCentredIID` and not the other two; that (A1)'s Lipschitz clause
delivers, on the region where it is assumed, the continuity of the projected diffusion that
`BarrierRescaling.lean` and `QuasipotentialMonotone.lean` carry as hypotheses, and only in its
global form `D = univ` the stronger continuity that `StationaryDensity.lean` and
`LogDensityCurvature.lean` ask for; that (A2)'s tolerance delivers the `c_min`, `c_max`
bracket of `BarrierRescaling.quasipotential_gap_bounds`, and coordinatewise the defect clause
`PowerOfS.abs_remainder_le_of_prefactor_defect` consumes; that (A3) makes the slack-channel
reading of Theorem 1 the estimator itself; that (A4) delivers the remainder budget
`CouplingSign.lean` carries, the modulus (A4'-q) asks for, and -- in the small-jitter regime --
the positive alignment (A6) assumes; and that (A5)'s geometry delivers the window bounds
`KramersExitTime.meanExitTime_upper_bound` carries.

## The ledger

Each row names the paper clause, the Lean object here, and the modules that consume it. Names of
results in sibling modules are quoted as they stand in the tree; the ledger is a reading aid, and
the authority is always the sibling module's own hypothesis list.

| clause | Lean here | consumed by |
|---|---|---|
| (A1) i.i.d. batches | `IidBatchStream`, `IidBatchStream.indep_comp`, `.ident_comp` | `CrossCovSLLN.lean`: the `indep` and `ident` fields of `IsCentredIID` and `IsSquareCentredIID`, hence `IsCentredIID.tendsto_average_ae`, `crossCovEstimatorE_tendsto_zero_ae`, `crossCovEstimator_tendsto_zero_ae`, `variance_average_of_iid`, `crossCovEstimator_entry_l2_rate`, `crossCovEstimator_windowCentred_l2_bound` |
| (A1) Lipschitz gradient-noise covariance | `LipschitzGradientNoise`, `.continuousOn_quadForm` (on the region `D`), `.continuous_quadForm` (only when `D = univ`) | *from `.continuousOn_quadForm`:* `BarrierRescaling.lean` (`quasipotential_gap_bounds_of_continuousOn`), `QuasipotentialMonotone.lean` (the `ContinuousOn` hypotheses of `quasipotentialIncrement_strictAnti_in_diffusion`). *Only from `.continuous_quadForm`, because they carry a global `Continuous` hypothesis:* `StationaryDensity.lean` (`stationaryDensity_zeroFlux_of_continuous`, `hasDerivAt_stationaryDensity`), `LogDensityCurvature.lean` (`hasDerivAt_effectivePotential_integral`, whose hypothesis is continuity of the *ratio* `L̃'/σ_eff²` and so needs more than (A1) alone) |
| (A1) SDE of SGD, `eta` small | recorded in `SdeOfSgdModel`'s docstring; no field | nothing; no module discharges it |
| (A2) single prefactor, with tolerance | `SinglePrefactor`, `.deriv_pos`, `.diffusion_bracket`, `.diffusion_bracket_pos`, `.coordinate_defect`, `.jacobian_defect_le`, `exists_singlePrefactor_band` | `PowerOfS.lean` (`abs_remainder_le_of_prefactor_defect` carries the tolerance clause verbatim, and `jacChannelExact_two_sided`, `corrected_power_survives_remainder` carry its consequence), `ShoulderAttenuation.lean` (`crossCovEstimator_attenuated_once`, `_twice`), `UpdateCovariance.lean` (the scalar `s`), `LogDensityCurvature.lean` (`logDensityCurvature_of_const_diffusion`), `BarrierRescaling.lean` (`quasipotential_gap_bounds`), `QuasipotentialMonotone.lean` (ellipticity `0 < sdir`) |
| (A3) slack Jacobian is the identity | `SlackJacobianIdentity`, `.rank_eq`, `.mulVec_bijective`, `.mulVec_injective`, `.slackReading_eq` | `StructuralZeros.lean` (`slackReading_one` is (A3) itself, `J_b = 1`), `CrossCov.lean` (`slackReading_eq_zero_of_jacobian_eq_zero` is its *deletion*, `J_b = 0`, so it is Theorem 1's case (i) and not a consumer of (A3)). **Nothing else in the development consumes it**; in particular `DiffusionOrdering.magnitudeChannel_pos` does *not* -- see the discrepancy below |
| (A4) forward-KL chain with remainder | `ForwardKLChain`, `.remainder_budget`, `.trace_leading_nonneg`, `.pairing_bound`, `.outerMoment_chain`, `.coupling_quadForm_split`, `.abs_coupling_sub_leading_le`, `forwardKLChain_collinear` | `CouplingSign.lean` (`hchain` of `outerMoment_chain`, `crossTerm_eq_add_remainder`, `crossTerm_ge_leading_sub_of_remainder_bound`), `DiffusionOrdering.lean` (`couplingTerm_forwardKL`, `diffusionExcess_forwardKL`), `PowerOfS.lean` (`couplingCore_pos_of_forwardKL`), `TraceLemmas.lean` (`trace_mul_nonneg_of_posSemidef` consumes its PSD clause) |
| (A4'-q) remainder control | `RemainderControl`, `.excess_lower_bound`, `.excess_nonneg`, `remainderControl_of_forwardKLChain` | `DiffusionOrdering.lean` (`forwardKL_excess_lower_bound`, `forwardKL_excess_nonneg`), `CouplingSign.lean` (the `delta` budget of `crossTerm_ge_leading_sub_of_remainder_bound`) |
| (A5) single-barrier geometry | `SingleBarrierPotential`, `.isMaxOn_escape`, `.isMinOn_escape`, `.kramers_window` | `KramersExitTime.lean` (`meanExitTime_upper_bound`, `meanExitTime_window_lower_bound`, `mean_first_passage_arrhenius_of_dynkin`) |
| (A5) small effective noise | `SmallEffectiveNoise`, `.arrheniusExponent_ge`, `.lt_two_mul_barrier` | `KramersExitTime.lean` (`eventually_lower_bound_gt`, `meanExitTime_lt_of_gap`), `FreeDiffusion.lean` (`metastable_window`, and its failure regime `freeDiffusionTime_lt_arrheniusTime_of_small_variance`) |
| (A6) positive alignment on the band | `CouplingAlignedOn`, `.pairing_lower`, `couplingQuadForm_fin_one`, `couplingAlignedOn_of_small_jitter`, `smallJitter_hypotheses_satisfiable` | `CouplingSign.lean` (`PositiveAlignmentOn`, `projDiffusion_ge_of_positiveAlignment`, `projDiffusion_strict_increase_on_band`), `DiffusionOrdering.lean` (`updateCov_ordering_band`, `updateCov_ordering_strict`), `QuasipotentialMonotone.lean` (`liftQuasipotentialIncrement_lt_direct`, `fwQuasipotential_lt_of_bandOccupancy`), `BarrierRescaling.lean` (the `hcoupling` of `arrheniusExponent_lift_lt_direct`) |

## Discrepancies between the appendix's attribution and what the Lean statements need

These are reported because the appendix invites the comparison, and because each is a change a
referee would otherwise have to find.

**Theorem 1 does not use (A1), except in one of its three cases, and there in a weaker form.**
The appendix says `thm:joint-necessity` uses only (A1). Cases (i) and (ii) use nothing at all:
`joint_necessity_cases_assumption_free` below carries no assumption structure among its
hypotheses, only the two deletion hypotheses of the theorem itself. Case (iii) splits in two. Its
population half, `CrossCovIndep.condPopCrossCov_centered_ae_eq_zero`, needs conditional
independence of the pair together with integrability, and no clause of (A1). Its finite-sample
half, the unbiasedness, the almost-sure limit and the `L²` rate of `CrossCovSLLN.lean`, needs the
i.i.d. clause -- but only in the pairwise-independent form recorded in `IidBatchStream`, because
mathlib's strong law is Etemadi's. So the appendix's attribution is an over-statement for two of
the three cases and an over-strength for the third.

**The corrected Lemma 1 needs more than (A1), (A3), (A4).** The appendix's attribution is for the
shipped proof, whose Hessian decomposition `PullbackHessian.lean` refutes outright. Along the
corrected route the list changes in four ways. (A2) is needed and is not listed: the single
prefactor `s` appears in every term of the update-covariance decomposition of section 2 and in
the ellipticity of the projected diffusion. (A6) is needed and is not stated in the paper at all;
the evidence for that is the subject of the next discrepancy. (A4'-q) is needed and is not
stated in the paper at all. And (A3) is not needed at all, which is the subject of the one after.

**(A6) is not implied by (A4) together with (A4'-q), and the sharp evidence is not the
counterexample one would first reach for.** `CouplingSign.coupling_sign_not_automatic` exhibits a
configuration with symmetric positive-definite `H`, positive-definite `V` and a nonzero (A4)
remainder on which the coupling term is strictly negative -- but that configuration **violates
(A4'-q)**, as `CouplingSign.lean` itself records, so on its own it leaves open the possibility
that (A6) follows once the remainder-control clause of FW-1 is added. It does not:
`CouplingSign.coupling_sign_not_automatic_under_remainder_control` exhibits a second
configuration, on the same sample space, which *does* satisfy (A4'-q) with a nonzero remainder
and on which the coupling term is still strictly negative, and
`CouplingSign.positiveAlignment_fails_under_remainder_control` refutes (A6) there for every `κ`.
That pair, not the first counterexample alone, is what makes (A6) an irreducible assumption.
`couplingAlignedOn_of_small_jitter` below delimits the complementary regime in which it is
automatic.

**(A3)'s "full rank" clause is consumed by no theorem in this development, and (A3) itself by
only one.** The appendix lists (A3) among the hypotheses of `lem:moreau`. What the corrected route
uses it for is the remark that the added curvature is not confined to a proper subspace, and the
Lean statement of that remark, `DiffusionOrdering.magnitudeChannel_pos`, does **not** carry (A3):
its non-degeneracy hypotheses are `V.PosDef` and `Function.Injective H.mulVec`, injectivity of
the mean loss *Hessian*, and `DiffusionOrdering.lean` says in its own docstring that the note's
rank hypothesis on the slack Jacobian is not modelled there. (A3) can supply
`Function.Injective J_b.mulVec` (`SlackJacobianIdentity.mulVec_injective`) and nothing else in
that shape. Its only genuine consumer is `StructuralZeros.slackReading_one`, which is the
identity `J_b = 1` itself rather than a use of its rank; the rank and determinant consequences
proved here are the appendix's assertions discharged, not inputs to anything downstream.

**(A5)'s `C²` clause is not consumed anywhere in this development.** `KramersExitTime.lean` and
`FreeDiffusion.lean` use only continuity of the potential on the escape interval together with
the extremal structure and the two-sided window bounds that `SingleBarrierPotential.kramers_window`
supplies. The `C²` clause would be needed for the Kramers prefactor, which is set by the
curvatures at the basin minimum and the barrier top; no module addresses the prefactor, and none
should be read as doing so.

**(A1)'s Lipschitz clause is consumed only through continuity, and two of its four consumers
need it on the whole space.** The appendix justifies the clause as what "controls the Itô-Taylor
remainder". No Itô-Taylor remainder is formalized anywhere in this development, so that use is
not machine-checked. What the modules actually consume is the continuity of the projected
diffusion, and the Lipschitz constant itself is never used quantitatively downstream. The
consumers split, however. `BarrierRescaling.quasipotential_gap_bounds_of_continuousOn` and the
`ContinuousOn` hypotheses of `QuasipotentialMonotone.lean` are discharged by
`LipschitzGradientNoise.continuousOn_quadForm` on any region containing the escape interval.
`StationaryDensity.stationaryDensity_zeroFlux_of_continuous` and
`LogDensityCurvature.hasDerivAt_effectivePotential_integral` carry a *global* `Continuous`
hypothesis, so they are reachable from (A1) only when the clause is assumed on all of the
parameter space (`LipschitzGradientNoise.continuous_quadForm`, whose `D` is `univ`); and
`hasDerivAt_effectivePotential_integral`'s hypothesis is continuity of the ratio `L̃'/σ_eff²`,
which needs the drift's regularity in addition to (A1). Neither point is visible from the
appendix's one-line statement of the clause.

**(A4) is in tension with the scaling argument of the design note's section 11.0.** Section 11.0
argues that the coupling channel is *first* order in the jitter scale and the magnitude channel
second, on the ground that the gradient fluctuation "is set by the minibatch, NOT by the lift, so
it does not shrink with `epsilon`". Under (A4) it does shrink: `dg = H dtheta + r` with
`‖r‖ <= Cr ‖dtheta‖²` forces `‖dg‖ <= M eps + Cr eps²`, and `ForwardKLChain.abs_traceCrossCov_le`
proves the consequence, `|sigma_Jac²| <= M eps² + Cr eps³` -- the same second order as the
magnitude channel. The two readings cannot both hold. Nothing downstream of this module depends
on the first-order reading; it is used in the note only to estimate how far the matched-noise
bracket of E1 had to be inflated, and that paragraph should carry the caveat.

## Results
* `euclNorm`, `euclNorm_sq`, `euclNorm_smul`, `euclNorm_pos`, `euclNorm_const_fin_one`,
  `dotProduct_le_euclNorm_mul`, `abs_dotProduct_le_euclNorm_mul` — the Euclidean length on
  `n → ℝ` written through the dot product, and Cauchy-Schwarz.
* `quadForm_eq_sum`, `trace_vecMulVec`, `trace_weightedOuter`, `vecMulVec_mulVec_smul`,
  `dotProduct_symmPairing`, `transpose_eq_of_posSemidef` — the matrix bookkeeping the
  assumptions share.
* `IidBatchStream` — (A1)'s i.i.d.-batch clause; `indep_comp` and `ident_comp` carry it to any
  measurable per-step statistic, which is what the strong law consumes.
* `LipschitzGradientNoise` — (A1)'s Lipschitz-covariance clause; `abs_quadForm_sub_le` makes
  every quadratic form Lipschitz with an explicit constant, `continuousOn_quadForm` gives the
  continuity of the projected diffusion on the region where the clause is assumed, and
  `continuous_quadForm` gives the global continuity that two of the four consumers ask for.
* `SdeOfSgdModel` — (A1) assembled, with the modelling clause recorded and no field for it.
* `SinglePrefactor` — (A2) with its tolerance; `deriv_pos` is FW-3b's uniform ellipticity,
  `diffusion_bracket` and `diffusion_bracket_pos` are the two-sided band bracket and the strict
  positivity of its lower end, `coordinate_defect` and `jacobian_defect_le` are the appendix's
  "approximately constant across coordinates" and its matrix reading `ψ'(θ̃) ≈ σ_s I_d`,
  `exists_singlePrefactor_band` is the paper's own narrow-band justification proved from
  continuity, and `subsingleton_of_tol_zero` shows the tolerance cannot be set to zero.
* `SlackJacobianIdentity` — (A3); `rank_eq` proves the full rank the paper asserts,
  `mulVec_bijective` and `mulVec_injective` are the "not confined to a proper subspace" reading,
  and `slackReading_eq` identifies the slack-channel reading with the estimator.
* `ForwardKLChain` — (A4) with an explicit quadratic remainder bound; `remainder_budget` is the
  paper's dropped higher-order central moment with its constant kept, `trace_leading_nonneg` is
  the leading-order sign claim, `pairing_bound` supplies (A4'-q)'s modulus,
  `abs_traceCrossCov_le` is the second-order bound that contradicts section 11.0's scaling,
  `outerMoment_chain` is the substitution `Σ_slack = V H + E[δθ̃ rᵀ]` that (FW-1b) performs,
  `coupling_quadForm_split` is the exact decomposition of (A6)'s coupling term under (A4), and
  `abs_coupling_sub_leading_le` bounds the part of it the paper drops.
* `forwardKLChain_collinear`, `collinear_remainder_ne_zero` — (A4) is satisfiable with a nonzero
  remainder that saturates the quadratic bound, so the estimates in `Cr` are not vacuous.
* `coupling_pos_of_small_jitter` — the scalar core of the regime in which (A6) is automatic
  rather than assumed.
* `RemainderControl` — (A4'-q) in quadratic-form language; `excess_lower_bound` and
  `excess_nonneg` are FW-1b, and `remainderControl_of_forwardKLChain` derives the whole structure
  from (A4) plus a jitter bound and a non-degeneracy modulus.
* `barrierAction`, `SingleBarrierPotential` — (A5)'s single-barrier geometry; `isMaxOn_escape`,
  `isMinOn_escape` and `kramers_window` are the hypotheses the exit-time module carries, and
  `singleBarrierPotential_quadratic` witnesses that the geometry is realizable.
* `SmallEffectiveNoise` — (A5)'s smallness clause with an explicit ratio; `arrheniusExponent_ge`
  and `lt_two_mul_barrier` place the diffusion inside the metastable window, and
  `smallEffectiveNoise_of_ratio` witnesses the clause at every ratio.
* `CouplingAlignedOn` — (A6); `pairing_lower` is its inner-product reading,
  `couplingQuadForm_fin_one` is the one-dimensional form the design note calls measurable,
  `couplingAlignedOn_smul_one` witnesses that the inequality alone is satisfiable, and
  `couplingAlignedOn_of_small_jitter` derives it from (A4) in the small-jitter regime with the
  modulus written out, `smallJitter_hypotheses_satisfiable` showing that regime is nonempty.
* `crossCovEstimator_zero_window`, `joint_necessity_cases_assumption_free` — cases (i) and (ii)
  of Theorem 1, for **every** window length including the empty one, with no assumption
  structure among the hypotheses.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The assumptions themselves are hypotheses, by construction: this module's job is to state them,
not to discharge them, and each structure is exactly as strong as the clause it names. What
would be needed to discharge each is the following. (A1)'s i.i.d. clause is a modelling choice
about the sampler and is discharged by the training loop, not by a proof; its Lipschitz clause is
a regularity property of a particular network and loss and would be established, if at all, for a
specific architecture; its SDE clause needs a weak-convergence theory for stochastic differential
equations that mathlib does not have. (A2) is a property of the positivity map on a region, and
`exists_singlePrefactor_band` discharges it for any `C¹` non-decreasing positivity map on a small enough
band, so what remains genuinely assumed is that the *operative* band is small enough -- a claim
about where training goes, not about `psi`. (A3) is an architectural fact about the lift and is
true by construction of the parametrization. (A4) is a second-order Taylor statement about the
forward-KL gradient and would be discharged by a `C²` bound on the loss composed with the positivity map
together with a Taylor remainder estimate; the analytic input is available in principle, and it is
a hypothesis here because this module does not model the loss. (A4'-q) is discharged from (A4)
by `remainderControl_of_forwardKLChain` given a jitter bound and a non-degeneracy modulus for the
magnitude channel, so it is not independent of (A4) so much as a smallness regime for it. (A5) is
a property of the pullback landscape along the bias channel and is not derivable from anything
here; `singleBarrierPotential_quadratic` and `smallEffectiveNoise_of_ratio` show only that its
two clauses are realizable, not that the landscape realizes them. (A6) cannot be discharged:
`CouplingSign.positiveAlignment_fails_under_remainder_control` refutes it, for every `κ`, on a
configuration that satisfies (A4), (A4'-q) and a nonzero remainder, so it does not follow from
the assumption set FW-1 already carries. `couplingAlignedOn_of_small_jitter` delimits the
complementary regime -- jitter small compared with `lam / (2 M Cr)`, magnitude channel
non-degenerate at the scale of the jitter -- in which it does follow from (A4) instead of being
assumed, and `smallJitter_hypotheses_satisfiable` shows that regime is not empty.

Four further limits of this module should be stated plainly. It duplicates three small
definitions and identities that also occur in the mechanism modules -- `euclNorm` (which is
`CouplingSign.vecNorm`), the symmetrized pairing identity (which is
`DiffusionOrdering.dotProduct_symmetrizedPairing`) and `ForwardKLChain.outerMoment_chain` (which
is `CouplingSign.outerMoment_chain`) -- because the assumption module is upstream of the
mechanism modules by design (`SoftplusBand.lean` imports it) and importing them would invert the
dependency; the definitions agree, and the duplication is deliberate. It does not *enforce* the
ledger: nothing in Lean prevents a sibling module from carrying a hypothesis that differs from
the structure named here, and the ledger is checked by reading, not by the kernel. The bridges it
proves are one-directional: they show that each assumption supplies what the sibling module asks
for, not that the sibling module could not have made do with less. And the satisfiability
witnesses are witnesses only: `singleBarrierPotential_quadratic`, `smallEffectiveNoise_of_ratio`,
`forwardKLChain_collinear` and `smallJitter_hypotheses_satisfiable` show that the hypotheses of
the theorems that carry them are jointly realizable -- so that no conclusion below is vacuously
true -- and say nothing about whether the trained model realizes them.
-/

open MeasureTheory ProbabilityTheory Matrix Finset Set

namespace IcnnLift

/-! ## Vocabulary shared by several of the assumptions -/

section Vec

variable {n : Type*} [Fintype n]

/-- The Euclidean length of a vector of `n → ℝ`, written through the dot product so that it
composes with `Matrix.mulVec` without a change of carrier type. -/
noncomputable def euclNorm (v : n → ℝ) : ℝ := Real.sqrt (v ⬝ᵥ v)

theorem euclNorm_nonneg (v : n → ℝ) : 0 ≤ euclNorm v := Real.sqrt_nonneg _

theorem euclNorm_eq_sqrt_sum_sq (v : n → ℝ) : euclNorm v = Real.sqrt (∑ i, v i ^ 2) := by
  rw [euclNorm, dotProduct]
  congr 1
  exact Finset.sum_congr rfl fun i _ => by ring

/-- Cauchy-Schwarz for the dot product on `n → ℝ`. -/
theorem dotProduct_le_euclNorm_mul (u v : n → ℝ) : u ⬝ᵥ v ≤ euclNorm u * euclNorm v := by
  rw [euclNorm_eq_sqrt_sum_sq, euclNorm_eq_sqrt_sum_sq, dotProduct]
  exact Real.sum_mul_le_sqrt_mul_sqrt Finset.univ u v

theorem abs_dotProduct_le_euclNorm_mul (u v : n → ℝ) : |u ⬝ᵥ v| ≤ euclNorm u * euclNorm v := by
  refine abs_le.mpr ⟨?_, dotProduct_le_euclNorm_mul u v⟩
  have h := dotProduct_le_euclNorm_mul (-u) v
  have hn : euclNorm (-u) = euclNorm u := by
    rw [euclNorm, euclNorm]
    congr 1
    simp [dotProduct]
  rw [hn, neg_dotProduct] at h
  linarith

/-- The square of the Euclidean length is the dot product. -/
theorem euclNorm_sq (v : n → ℝ) : euclNorm v ^ 2 = v ⬝ᵥ v :=
  Real.sq_sqrt (Finset.sum_nonneg fun _ _ => mul_self_nonneg _)

/-- The Euclidean length is absolutely homogeneous. It is used to check that the collinear
remainder `r = ‖δθ̃‖ δθ̃` saturates (A4)'s quadratic bound, so that the witness below has a
remainder of exactly the order the assumption permits and not a smaller one. -/
theorem euclNorm_smul (c : ℝ) (v : n → ℝ) : euclNorm (c • v) = |c| * euclNorm v := by
  have hd : (c • v) ⬝ᵥ (c • v) = c ^ 2 * (v ⬝ᵥ v) := by
    simp only [dotProduct, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [euclNorm, hd, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs, euclNorm]

/-- The Euclidean length of a constant non-negative one-dimensional vector is that constant.
Used only by the one-dimensional satisfiability witness of (A6). -/
theorem euclNorm_const_fin_one {c : ℝ} (hc : 0 ≤ c) :
    euclNorm (fun _ : Fin 1 => c) = c := by
  rw [euclNorm, dotProduct, Fin.sum_univ_one, Real.sqrt_mul_self hc]

/-- A nonzero vector has positive Euclidean length. -/
theorem euclNorm_pos {v : n → ℝ} (hv : v ≠ 0) : 0 < euclNorm v := by
  have hnn : (0 : ℝ) ≤ v ⬝ᵥ v := Finset.sum_nonneg fun _ _ => mul_self_nonneg _
  have hne : v ⬝ᵥ v ≠ 0 := fun h => hv (dotProduct_self_eq_zero.mp h)
  exact Real.sqrt_pos.mpr (lt_of_le_of_ne hnn (Ne.symm hne))

/-- An outer product acting on a vector: `(u vᵀ) x = (v · x) u`. -/
theorem vecMulVec_mulVec_smul (u v x : n → ℝ) :
    Matrix.vecMulVec u v *ᵥ x = (v ⬝ᵥ x) • u := by
  ext i
  simp only [Matrix.mulVec, dotProduct, Matrix.vecMulVec_apply, Pi.smul_apply, smul_eq_mul,
    Finset.sum_mul]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- A finite sample space carrying weights that sum to one is nonempty. -/
theorem nonempty_of_sum_weights_eq_one {Ω : Type*} [Fintype Ω] {w : Ω → ℝ}
    (hsum : ∑ ω, w ω = 1) : Nonempty Ω := by
  rcases isEmpty_or_nonempty Ω with hE | hN
  · exfalso
    haveI := hE
    rw [Finset.univ_eq_empty, Finset.sum_empty] at hsum
    exact zero_ne_one hsum
  · exact hN

/-- A quadratic form written as a double sum over the entries. -/
theorem quadForm_eq_sum (a : n → ℝ) (M : Matrix n n ℝ) :
    a ⬝ᵥ (M *ᵥ a) = ∑ i, ∑ j, a i * M i j * a j := by
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- The trace of an outer product is the dot product: this is what makes `σ_Jac² = tr Σ_slack`
a scalar second moment rather than a matrix quantity. -/
theorem trace_vecMulVec (u v : n → ℝ) : (Matrix.vecMulVec u v).trace = u ⬝ᵥ v := by
  simp [Matrix.trace, Matrix.diag_apply, Matrix.vecMulVec_apply, dotProduct]

/-- The trace of a finite weighted second moment `E[u vᵀ] = ∑_ω w(ω) u(ω) v(ω)ᵀ` is the weighted
average of the dot products. The left-hand side is, verbatim, the definition of
`CouplingSign.outerMoment`, so this lemma reads `σ_Jac² = E[δθ̃ · δg]`. -/
theorem trace_weightedOuter {Ω : Type*} [Fintype Ω] (w : Ω → ℝ) (u v : Ω → n → ℝ) :
    (∑ ω, w ω • Matrix.vecMulVec (u ω) (v ω)).trace = ∑ ω, w ω * (u ω ⬝ᵥ v ω) := by
  rw [Matrix.trace_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Matrix.trace_smul, trace_vecMulVec, smul_eq_mul]

/-- The symmetrized pairing identity `eᵀ (H R + Rᵀ H) e = 2 ⟨H e, R e⟩` for symmetric `H`. It is
the identity behind the factor `2` in the remainder-control clause (A4'-q), and it is what turns
assumption (A6) from a matrix inequality into a statement about an inner product. -/
theorem dotProduct_symmPairing {H : Matrix n n ℝ} (hH : Hᵀ = H) (R : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ e) = 2 * ((H *ᵥ e) ⬝ᵥ (R *ᵥ e)) := by
  have hx : e ᵥ* H = H *ᵥ e := by
    conv_rhs => rw [← hH]
    rw [Matrix.mulVec_transpose]
  have h1 : e ⬝ᵥ ((H * R) *ᵥ e) = (H *ᵥ e) ⬝ᵥ (R *ᵥ e) := by
    rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, hx]
  have h2 : e ⬝ᵥ ((Rᵀ * H) *ᵥ e) = (H *ᵥ e) ⬝ᵥ (R *ᵥ e) := by
    rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
    exact dotProduct_comm _ _
  rw [Matrix.add_mulVec, dotProduct_add, h1, h2]
  ring

omit [Fintype n] in
/-- Over `ℝ` a positive-semidefinite matrix is symmetric. -/
theorem transpose_eq_of_posSemidef {H : Matrix n n ℝ} (h : H.PosSemidef) : Hᵀ = H := by
  ext i j
  have hij := congrFun (congrFun h.1.eq i) j
  simpa [Matrix.conjTranspose_apply] using hij

end Vec

/-! ## (A1) SDE-of-SGD, i.i.d. batches, Lipschitz gradient-noise covariance -/

section A1

variable {Ω X' : Type*} [MeasurableSpace Ω] [MeasurableSpace X']

/-- **(A1), the i.i.d.-batch clause.** The batch stream `X⁽⁰⁾, X⁽¹⁾, …` is pairwise independent
and identically distributed under the training measure `μ`.

Pairwise independence, rather than the full mutual independence the phrase "i.i.d." suggests, is
recorded here deliberately: it is all that the strong law of `CrossCovSLLN.lean` consumes
(mathlib's `ProbabilityTheory.strong_law_ae` is Etemadi's), so stating the assumption in the
weaker form makes visible that the development never uses mutual independence. A user who has
mutual independence can supply this structure from `ProbabilityTheory.iIndepFun.indepFun`. -/
structure IidBatchStream (μ : Measure Ω) (Xs : ℕ → Ω → X') : Prop where
  /-- distinct batches are independent -/
  indep : Pairwise fun i j => IndepFun (Xs i) (Xs j) μ
  /-- every batch has the law of the first -/
  ident : ∀ i, IdentDistrib (Xs i) (Xs 0) μ μ

/-- Any measurable per-step statistic of an i.i.d. batch stream is again pairwise independent
across steps. With `IidBatchStream.ident_comp` this supplies exactly two of the four fields of
`CrossCovSLLN.IsCentredIID`, namely `indep` and `ident`. -/
theorem IidBatchStream.indep_comp {μ : Measure Ω} {Xs : ℕ → Ω → X'} {E : Type*}
    [MeasurableSpace E] (h : IidBatchStream μ Xs) {F : X' → E} (hF : Measurable F) :
    Pairwise fun i j => IndepFun (fun ω => F (Xs i ω)) (fun ω => F (Xs j ω)) μ :=
  fun _ _ hij => (h.indep hij).comp hF hF

/-- Any measurable per-step statistic of an i.i.d. batch stream is identically distributed
across steps. -/
theorem IidBatchStream.ident_comp {μ : Measure Ω} {Xs : ℕ → Ω → X'} {E : Type*}
    [MeasurableSpace E] (h : IidBatchStream μ Xs) {F : X' → E} (hF : Measurable F) (i : ℕ) :
    IdentDistrib (fun ω => F (Xs i ω)) (fun ω => F (Xs 0 ω)) μ μ :=
  (h.ident i).comp hF

end A1

section A1cov

variable {P : Type*} [PseudoMetricSpace P] {n : Type*} [Fintype n]

/-- **(A1), the Lipschitz-covariance clause.** The gradient-noise covariance field
`Σ : φ ↦ Σ(φ)` is Lipschitz in the body parameters, with constant `K`, on the region `D`.

The clause is stated entrywise with a single constant. Over a finite index this is equivalent,
up to a factor depending only on the dimension, to Lipschitz continuity in any matrix norm; the
entrywise form is used because mathlib carries no global norm instance on `Matrix`, all of them
being scoped. What the downstream modules actually consume is the consequence
`LipschitzGradientNoise.continuousOn_quadForm`: continuity of the *projected* diffusion
`σ_eff²(φ) = e_bᵀ Σ(φ) e_b`, which is the standing continuity hypothesis of
`StationaryDensity.lean` and of `BarrierRescaling.quasipotential_gap_bounds_of_continuousOn`. -/
structure LipschitzGradientNoise (Sigma : P → Matrix n n ℝ) (K : ℝ) (D : Set P) : Prop where
  /-- the Lipschitz constant is non-negative -/
  const_nonneg : 0 ≤ K
  /-- every entry of the covariance is `K`-Lipschitz on `D` -/
  entry : ∀ i j : n, ∀ p ∈ D, ∀ q ∈ D, |Sigma p i j - Sigma q i j| ≤ K * dist p q

/-- Every quadratic form of a Lipschitz covariance field is Lipschitz, with the constant written
out: `|aᵀ Σ(p) a − aᵀ Σ(q) a| ≤ K (∑ᵢ |aᵢ|)² d(p, q)`. -/
theorem LipschitzGradientNoise.abs_quadForm_sub_le {Sigma : P → Matrix n n ℝ} {K : ℝ} {D : Set P}
    (h : LipschitzGradientNoise Sigma K D) (a : n → ℝ) {p q : P} (hp : p ∈ D) (hq : q ∈ D) :
    |a ⬝ᵥ (Sigma p *ᵥ a) - a ⬝ᵥ (Sigma q *ᵥ a)| ≤ K * dist p q * (∑ i, |a i|) ^ 2 := by
  have hKd : 0 ≤ K * dist p q := mul_nonneg h.const_nonneg dist_nonneg
  have key : a ⬝ᵥ (Sigma p *ᵥ a) - a ⬝ᵥ (Sigma q *ᵥ a)
      = ∑ i, ∑ j, a i * (Sigma p i j - Sigma q i j) * a j := by
    rw [quadForm_eq_sum, quadForm_eq_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  calc |a ⬝ᵥ (Sigma p *ᵥ a) - a ⬝ᵥ (Sigma q *ᵥ a)|
      = |∑ i, ∑ j, a i * (Sigma p i j - Sigma q i j) * a j| := by rw [key]
    _ ≤ ∑ i, |∑ j, a i * (Sigma p i j - Sigma q i j) * a j| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, |a i * (Sigma p i j - Sigma q i j) * a j| :=
        Finset.sum_le_sum fun i _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, (K * dist p q) * (|a i| * |a j|) := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
        have hb := h.entry i j p hp q hq
        have habs : |a i * (Sigma p i j - Sigma q i j) * a j|
            = |a i| * |Sigma p i j - Sigma q i j| * |a j| := by
          rw [abs_mul, abs_mul]
        rw [habs]
        have h1 : |a i| * |Sigma p i j - Sigma q i j| * |a j|
            ≤ |a i| * (K * dist p q) * |a j| := by
          have := mul_le_mul_of_nonneg_left hb (abs_nonneg (a i))
          exact mul_le_mul_of_nonneg_right this (abs_nonneg (a j))
        calc |a i| * |Sigma p i j - Sigma q i j| * |a j|
            ≤ |a i| * (K * dist p q) * |a j| := h1
          _ = (K * dist p q) * (|a i| * |a j|) := by ring
    _ = (K * dist p q) * ((∑ i, |a i|) * (∑ j, |a j|)) := by
        rw [Finset.sum_mul_sum, Finset.mul_sum]
        exact Finset.sum_congr rfl fun i _ => by rw [Finset.mul_sum]
    _ = K * dist p q * (∑ i, |a i|) ^ 2 := by ring

/-- The consequence the diffusion modules use: the projected diffusion `φ ↦ e_bᵀ Σ(φ) e_b` is
continuous on the region where (A1)'s Lipschitz clause holds. -/
theorem LipschitzGradientNoise.continuousOn_quadForm {Sigma : P → Matrix n n ℝ} {K : ℝ}
    {D : Set P} (h : LipschitzGradientNoise Sigma K D) (a : n → ℝ) :
    ContinuousOn (fun p => a ⬝ᵥ (Sigma p *ᵥ a)) D := by
  have hK : 0 ≤ K * (∑ i, |a i|) ^ 2 := mul_nonneg h.const_nonneg (sq_nonneg _)
  refine (LipschitzOnWith.of_dist_le_mul
    (K := ⟨K * (∑ i, |a i|) ^ 2, hK⟩) ?_).continuousOn
  intro p hp q hq
  rw [Real.dist_eq]
  calc |a ⬝ᵥ (Sigma p *ᵥ a) - a ⬝ᵥ (Sigma q *ᵥ a)|
      ≤ K * dist p q * (∑ i, |a i|) ^ 2 := h.abs_quadForm_sub_le a hp hq
    _ = (K * (∑ i, |a i|) ^ 2) * dist p q := by ring
    _ = ((⟨K * (∑ i, |a i|) ^ 2, hK⟩ : NNReal) : ℝ) * dist p q := by norm_num

/-- **The global form, which is what two of the four consumers actually ask for.** If the
Lipschitz clause is assumed on the whole parameter space rather than on a region, the projected
diffusion is continuous everywhere. This distinction is not cosmetic:
`BarrierRescaling.quasipotential_gap_bounds_of_continuousOn` and the `ContinuousOn` hypotheses of
`QuasipotentialMonotone.lean` are satisfied by `continuousOn_quadForm` on any region containing
the escape interval, whereas `StationaryDensity.stationaryDensity_zeroFlux_of_continuous` and
`LogDensityCurvature.hasDerivAt_effectivePotential_integral` carry a *global* `Continuous`
hypothesis and are therefore reachable from (A1) only in this form, with `D = univ`. -/
theorem LipschitzGradientNoise.continuous_quadForm {Sigma : P → Matrix n n ℝ} {K : ℝ}
    (h : LipschitzGradientNoise Sigma K Set.univ) (a : n → ℝ) :
    Continuous (fun p => a ⬝ᵥ (Sigma p *ᵥ a)) :=
  continuousOn_univ.mp (h.continuousOn_quadForm a)

end A1cov

/-- **(A1), assembled.** The learning rate, the i.i.d.-batch clause and the Lipschitz-covariance
clause, bundled.

The remaining clause of (A1) as the appendix states it -- "SDE-of-SGD with `η` small" -- has
**no field here, and deliberately so**. It is a modelling register: it asserts that the
continuous-time Itô process `dφ = −∇L dt + Σ^{1/2}(φ) dB` is an adequate surrogate for the
discrete stochastic-gradient recursion in the small-step limit. That is a statement relating two
objects only one of which exists in this development, and mathlib v4.31.0 has no Itô integral,
no stochastic differential equation and no weak-convergence theory for them, so there is nothing
to state it about. Every module downstream of this one therefore takes the *diffusion* as its
primitive and says so in its own docstring; the passage from SGD to the diffusion is not proved
anywhere in this development and must not be read as if it were. The learning rate is carried as
a positive real because the remainder terms of the ergodic expansion are `O(η)` and the modules
that quote them carry the remainder explicitly. The clause's own qualifier, "`η` small", has no
field either: smallness of `η` is meaningless without the comparison scale that only the
suppressed Itô-Taylor estimate would supply, and asserting an inequality `η < c` for an
unspecified `c` would record nothing. -/
structure SdeOfSgdModel {Ω X' P n : Type*} [MeasurableSpace Ω] [MeasurableSpace X']
    [PseudoMetricSpace P] [Fintype n] (μ : Measure Ω) (Xs : ℕ → Ω → X')
    (Sigma : P → Matrix n n ℝ) (D : Set P) where
  /-- the learning rate -/
  eta : ℝ
  /-- the learning rate is positive -/
  eta_pos : 0 < eta
  /-- the Lipschitz constant of the gradient-noise covariance -/
  lipConst : ℝ
  /-- the batches are independent and identically distributed -/
  batches : IidBatchStream μ Xs
  /-- the gradient-noise covariance is Lipschitz in the body parameters on `D` -/
  covariance : LipschitzGradientNoise Sigma lipConst D

/-! ## (A2) the single-prefactor idealization, with its tolerance -/

/-- **(A2).** The positivity map `ψ` is `C¹` and non-decreasing, and on the operative shoulder band `B`
its derivative is within `eps` of the single constant `s`.

The tolerance is the point of this structure. The appendix states (A2) as
`ψ'(θ̃) ≈ σ_s I_d` -- "approximately constant across coordinates" -- and justifies it by the band
being narrow. Turning that into the exact identity `ψ' = s` on `B` would be a different and
strictly stronger hypothesis: `SinglePrefactor.subsingleton_of_tol_zero` shows that for a positivity map
with injective derivative, softplus among them, the exact version forces the band to be a single
point, and a single-point band carries no information about the shoulder. The tolerance is
carried instead, and every consequence below is a two-sided bracket whose width is `eps`.
`SoftplusBand.lean`, which imports this module, discharges the structure for the softplus positivity map
on a closed ball of the shoulder, so the band is not merely hypothetical.

The two order clauses `0 < s` and `eps < s` are the ellipticity content that FW-3b of section
11.2 reads off (A2): they are what keeps the projected diffusion bounded away from zero on the
band, and without them the quasipotential comparison of `QuasipotentialMonotone.lean` has no
uniform lower bound to work with.

One further respect in which the structure is weaker than the appendix, deliberately: the
appendix pins the constant, `σ_s = ψ'(w̃_s)` at the shoulder point, whereas `s` is a free
parameter here. A free `s` is the weaker hypothesis and so the stronger theorem, and the
appendix's own choice is recovered by `exists_singlePrefactor_band`, which instantiates `s` as
the derivative at the centre of the band it produces. -/
structure SinglePrefactor (psi : ℝ → ℝ) (B : Set ℝ) (s eps : ℝ) : Prop where
  /-- the positivity map is continuously differentiable -/
  contDiff : ContDiff ℝ 1 psi
  /-- the positivity map is non-decreasing -/
  mono : Monotone psi
  /-- the single prefactor is positive -/
  prefactor_pos : 0 < s
  /-- the tolerance is non-negative -/
  tol_nonneg : 0 ≤ eps
  /-- the tolerance is smaller than the prefactor: the relative error is below one -/
  tol_lt : eps < s
  /-- on the band the derivative is within `eps` of the single prefactor -/
  approx : ∀ w ∈ B, |deriv psi w - s| ≤ eps

variable {psi : ℝ → ℝ} {B : Set ℝ} {s eps : ℝ}

/-- Monotonicity gives a non-negative derivative everywhere, band or not. -/
theorem SinglePrefactor.deriv_nonneg (h : SinglePrefactor psi B s eps) (w : ℝ) :
    0 ≤ deriv psi w := h.mono.deriv_nonneg

theorem SinglePrefactor.deriv_ge (h : SinglePrefactor psi B s eps) {w : ℝ} (hw : w ∈ B) :
    s - eps ≤ deriv psi w := by
  have := (abs_le.mp (h.approx w hw)).1
  linarith

theorem SinglePrefactor.deriv_le (h : SinglePrefactor psi B s eps) {w : ℝ} (hw : w ∈ B) :
    deriv psi w ≤ s + eps := by
  have := (abs_le.mp (h.approx w hw)).2
  linarith

/-- **Uniform ellipticity on the band, FW-3b.** The positivity-map prefactor is bounded away from zero
on the operative band, with the explicit constant `s − eps > 0`. -/
theorem SinglePrefactor.deriv_pos (h : SinglePrefactor psi B s eps) {w : ℝ} (hw : w ∈ B) :
    0 < deriv psi w := by
  have h1 := h.deriv_ge hw
  have h2 := h.tol_lt
  linarith

/-- **The band bracket (A2) actually delivers.** With the tolerance carried, the attenuated
objective diffusion `ψ'(w)² σ_obj²` on the band is trapped between `(s − eps)² σ_obj²` and
`(s + eps)² σ_obj²`. These two numbers are exactly the `c_min` and `c_max` that
`BarrierRescaling.quasipotential_gap_bounds` consumes, so the paper's phrase "over which `ψ'`
varies slowly" is discharged into an explicit interval rather than into an equality. Setting
`eps = 0` collapses the bracket to the shipped constant-diffusion identity. -/
theorem SinglePrefactor.diffusion_bracket (h : SinglePrefactor psi B s eps) {w : ℝ} (hw : w ∈ B)
    {sigmaObj2 : ℝ} (hobj : 0 ≤ sigmaObj2) :
    (s - eps) ^ 2 * sigmaObj2 ≤ deriv psi w ^ 2 * sigmaObj2 ∧
      deriv psi w ^ 2 * sigmaObj2 ≤ (s + eps) ^ 2 * sigmaObj2 := by
  have hge := h.deriv_ge hw
  have hle := h.deriv_le hw
  have hpos : 0 < s - eps := by have := h.tol_lt; linarith
  constructor
  · exact mul_le_mul_of_nonneg_right (by nlinarith) hobj
  · exact mul_le_mul_of_nonneg_right (by nlinarith) hobj

/-- **The `c_min` positivity the quasipotential comparison needs.** `diffusion_bracket` gives the
two-sided bracket but not the strict positivity of its lower end, which is a separate hypothesis
`0 < c_min` of `BarrierRescaling.quasipotential_gap_bounds`. It follows from (A2)'s order clause
`eps < s` as soon as the objective diffusion is itself positive, and it is recorded separately so
that the extra input -- positivity of `σ_obj²`, which (A2) does not assert -- is visible. -/
theorem SinglePrefactor.diffusion_bracket_pos (h : SinglePrefactor psi B s eps)
    {sigmaObj2 : ℝ} (hobj : 0 < sigmaObj2) : 0 < (s - eps) ^ 2 * sigmaObj2 := by
  have hpos : 0 < s - eps := by have := h.tol_lt; linarith
  exact mul_pos (pow_pos hpos 2) hobj

/-- **(A2) across coordinates, which is how the appendix states it.** The appendix writes the
clause on the *vector* iterate, `ψ'(θ̃) ≈ σ_s I_d`, "approximately constant across coordinates".
Because `ψ` is applied coordinatewise, that clause is the scalar band clause evaluated at each
coordinate, and this is the form `PowerOfS.abs_remainder_le_of_prefactor_defect` consumes
verbatim as its hypothesis `hdefect : ∀ ω i, |p ω i − s| ≤ ε`. Stating it here closes the gap
between the scalar band that `SinglePrefactor` carries and the matrix statement the paper makes. -/
theorem SinglePrefactor.coordinate_defect {ι : Type*} (h : SinglePrefactor psi B s eps)
    {th : ι → ℝ} (hth : ∀ i, th i ∈ B) (i : ι) : |deriv psi (th i) - s| ≤ eps :=
  h.approx (th i) (hth i)

/-- **The matrix reading of (A2), with the tolerance.** For an iterate all of whose coordinates
lie on the band, the positivity-map Jacobian `ψ'(θ̃) = diag(ψ'(θ̃ᵢ))` differs from `σ_s I_d` by a
quadratic form of size at most `eps ‖x‖²`. This is the appendix's `ψ'(θ̃) ≈ σ_s I_d` made
quantitative, and it is the sense in which the single-prefactor idealization is an
*approximation* rather than the substitution `ψ' = s` that the shipped proof performs. -/
theorem SinglePrefactor.jacobian_defect_le {ι : Type*} [Fintype ι] [DecidableEq ι]
    (h : SinglePrefactor psi B s eps) {th : ι → ℝ} (hth : ∀ i, th i ∈ B) (x : ι → ℝ) :
    |x ⬝ᵥ ((Matrix.diagonal (fun i => deriv psi (th i)) - s • (1 : Matrix ι ι ℝ)) *ᵥ x)|
      ≤ eps * (x ⬝ᵥ x) := by
  have hdiag : (Matrix.diagonal (fun i => deriv psi (th i)) - s • (1 : Matrix ι ι ℝ))
      = Matrix.diagonal (fun i => deriv psi (th i) - s) := by
    rw [Matrix.smul_one_eq_diagonal, Matrix.diagonal_sub]
  have hsum : x ⬝ᵥ (Matrix.diagonal (fun i => deriv psi (th i) - s) *ᵥ x)
      = ∑ i, (deriv psi (th i) - s) * x i ^ 2 := by
    simp only [dotProduct, Matrix.mulVec_diagonal]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hdot : x ⬝ᵥ x = ∑ i, x i ^ 2 := by
    simp only [dotProduct]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hdiag, hsum, hdot, Finset.mul_sum]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun i _ => ?_)
  rw [abs_mul, abs_of_nonneg (sq_nonneg (x i))]
  exact mul_le_mul_of_nonneg_right (h.approx (th i) (hth i)) (sq_nonneg _)

/-- **(A2) is satisfiable, and this is the paper's own justification made precise.** Every `C¹`
non-decreasing positivity map whose derivative at a point exceeds the tolerance satisfies the
single-prefactor idealization on some ball around that point, with that tolerance. This is the
appendix's "the operative region is a narrow band of the softplus shoulder over which `ψ'` varies
slowly", proved from continuity of `ψ'` rather than asserted -- and it makes visible that the
band's *width* is what the tolerance buys, so that (A2) constrains the band and the tolerance
jointly and neither alone. -/
theorem exists_singlePrefactor_band (hC : ContDiff ℝ 1 psi) (hmono : Monotone psi)
    {w0 : ℝ} (heps : 0 < eps) (hlt : eps < deriv psi w0) :
    ∃ delta > 0, SinglePrefactor psi (Metric.ball w0 delta) (deriv psi w0) eps := by
  have hcont : Continuous (deriv psi) := hC.continuous_deriv le_rfl
  obtain ⟨delta, hdelta, hball⟩ := Metric.continuousAt_iff.mp hcont.continuousAt eps heps
  refine ⟨delta, hdelta, ?_⟩
  refine
    { contDiff := hC
      mono := hmono
      prefactor_pos := heps.trans hlt
      tol_nonneg := heps.le
      tol_lt := hlt
      approx := ?_ }
  intro w hw
  have hd : dist w w0 < delta := Metric.mem_ball.mp hw
  have := hball hd
  rw [Real.dist_eq] at this
  exact this.le

/-- **Why the tolerance cannot be dropped.** If the tolerance is zero then the derivative is
literally constant on the band, so for a positivity map with injective derivative the band collapses to
a single point. The softplus positivity map of the paper is such a positivity map: its derivative is the
logistic (`ShoulderAttenuation.deriv_softplus_eq_logistic`), which is strictly increasing
(`ShoulderAttenuation.logistic_strictMono`) and hence injective. An exact single-prefactor
idealization is therefore not a mild strengthening of (A2): for the positivity map the paper uses it is
satisfiable only on a degenerate band, and a degenerate band says nothing about the shoulder. -/
theorem SinglePrefactor.subsingleton_of_tol_zero (h : SinglePrefactor psi B s 0)
    (hinj : Function.Injective (deriv psi)) : B.Subsingleton := by
  intro x hx y hy
  have hx' : deriv psi x = s := by
    have := h.approx x hx
    have h2 := abs_nonneg (deriv psi x - s)
    have : |deriv psi x - s| = 0 := le_antisymm this h2
    have := abs_eq_zero.mp this
    linarith
  have hy' : deriv psi y = s := by
    have := h.approx y hy
    have h2 := abs_nonneg (deriv psi y - s)
    have : |deriv psi y - s| = 0 := le_antisymm this h2
    have := abs_eq_zero.mp this
    linarith
  exact hinj (hx'.trans hy'.symm)

/-! ## (A3) the slack Jacobian is the identity -/

/-- **(A3).** The slack Jacobian `∂θ̃/∂b` is the `d × d` identity.

The appendix states (A3) as "the slack Jacobian `∂θ̃/∂b = I_d` has full rank". The identity is
the hypothesis and the full rank is its consequence, proved below rather than assumed. -/
structure SlackJacobianIdentity {d : ℕ} (Jb : Matrix (Fin d) (Fin d) ℝ) : Prop where
  /-- the slack Jacobian is the identity matrix -/
  eq_one : Jb = 1

/-- (A3) gives full rank: the rank of the slack Jacobian is the slack dimension. -/
theorem SlackJacobianIdentity.rank_eq {d : ℕ} {Jb : Matrix (Fin d) (Fin d) ℝ}
    (h : SlackJacobianIdentity Jb) : Jb.rank = d := by
  rw [h.eq_one, Matrix.rank_one, Fintype.card_fin]

theorem SlackJacobianIdentity.det_eq_one {d : ℕ} {Jb : Matrix (Fin d) (Fin d) ℝ}
    (h : SlackJacobianIdentity Jb) : Jb.det = 1 := by
  rw [h.eq_one, Matrix.det_one]

/-- The reading of full rank that the proof of Lemma 1 actually uses: the slack subspace is all
of `ℝ^d`, so the added curvature is not confined to a proper subspace. -/
theorem SlackJacobianIdentity.mulVec_bijective {d : ℕ} {Jb : Matrix (Fin d) (Fin d) ℝ}
    (h : SlackJacobianIdentity Jb) : Function.Bijective (fun v : Fin d → ℝ => Jb *ᵥ v) := by
  have hid : (fun v : Fin d → ℝ => Jb *ᵥ v) = id := by
    funext v
    rw [h.eq_one, Matrix.one_mulVec]
    rfl
  rw [hid]
  exact Function.bijective_id

/-- The injectivity form of full rank, stated in exactly the shape
`DiffusionOrdering.magnitudeMatrix_posDef` and `DiffusionOrdering.magnitudeChannel_pos` carry as
their non-degeneracy hypothesis -- *for a different matrix*. Those theorems ask for
`Function.Injective H.mulVec`, injectivity of the mean loss Hessian, and (A3) can supply that
predicate only for the slack Jacobian. The two are recorded side by side here because the
appendix's attribution suggests otherwise; see the ledger's discrepancy on (A3). -/
theorem SlackJacobianIdentity.mulVec_injective {d : ℕ} {Jb : Matrix (Fin d) (Fin d) ℝ}
    (h : SlackJacobianIdentity Jb) : Function.Injective Jb.mulVec :=
  h.mulVec_bijective.1

/-- **(A3) makes the slack-channel reading the estimator itself.** Under the identity slack
Jacobian the contraction of `CrossCov.crossCovEstimator` with `J_b` that Theorem 1 calls the
slack-channel reading is the estimator, so case (i) of Theorem 1 -- deletion of the slack
channel -- is a genuine deletion and not a change of coordinates. -/
theorem SlackJacobianIdentity.slackReading_eq {d : ℕ} {Jb : Matrix (Fin d) (Fin d) ℝ}
    (h : SlackJacobianIdentity Jb) (T t : ℕ) (dth dg : ℕ → Fin d → ℝ) :
    slackReading Jb T t dth dg = crossCovEstimator T t dth dg := by
  rw [slackReading, h.eq_one, Matrix.transpose_one, Matrix.one_mul]

/-! ## (A4) the forward-KL gradient-fluctuation chain, with its remainder -/

/-- **(A4).** On the operative region the forward-KL gradient fluctuation is the expected loss
Hessian acting on the iterate fluctuation, plus a remainder of order `‖δθ̃‖²`:

`δg(ω) = H δθ̃(ω) + r(ω)`,  `H = E_X[∇²_{θ̃} L] ⪰ 0` symmetric,  `‖r(ω)‖ ≤ Cr ‖δθ̃(ω)‖²`.

The remainder is a field, with an explicit quadratic bound and an explicit constant. The paper
writes `r` "a remainder of order `‖δθ̃‖²`" and then, in the proof of Lemma 1, drops it as "a
higher-order central moment"; `ForwardKLChain.remainder_budget` below is that step done with the
constant kept, and it is what discharges the hypothesis `hdelta` of
`CouplingSign.crossTerm_ge_leading_sub_of_remainder_bound`. -/
structure ForwardKLChain {n Ω : Type*} [Fintype n] (H : Matrix n n ℝ)
    (dth dg r : Ω → n → ℝ) (Cr : ℝ) : Prop where
  /-- the mean Hessian is symmetric positive semidefinite -/
  posSemidef : H.PosSemidef
  /-- the remainder constant is non-negative -/
  const_nonneg : 0 ≤ Cr
  /-- the chain, pointwise on the sample space -/
  chain : ∀ ω, dg ω = H *ᵥ dth ω + r ω
  /-- the remainder is quadratic in the iterate fluctuation -/
  remainder : ∀ ω, euclNorm (r ω) ≤ Cr * euclNorm (dth ω) ^ 2

variable {n Ω : Type*} [Fintype n] {H : Matrix n n ℝ} {dth dg r : Ω → n → ℝ} {Cr : ℝ}

theorem ForwardKLChain.transpose_eq (h : ForwardKLChain H dth dg r Cr) : Hᵀ = H :=
  transpose_eq_of_posSemidef h.posSemidef

/-- **(A4) is satisfiable with a remainder that is genuinely there and genuinely quadratic.**
For any positive-semidefinite mean Hessian and any iterate fluctuation, the collinear choice
`r(ω) = ‖δθ̃(ω)‖ δθ̃(ω)` satisfies the chain and saturates the quadratic bound with `Cr = 1`.
This is stated because every estimate below is an inequality in `Cr`, and an inequality in `Cr`
would be uninformative if the only configurations satisfying (A4) were those with `r = 0`;
`collinear_remainder_ne_zero` records that this remainder vanishes only where the fluctuation
does. The construction mirrors the collinear family of `CouplingSign.lean`, which is where the
same device is used to show that (A6) can fail under (A4'-q). -/
theorem forwardKLChain_collinear (hH : H.PosSemidef) (dth : Ω → n → ℝ) :
    ForwardKLChain H dth (fun ω => H *ᵥ dth ω + euclNorm (dth ω) • dth ω)
      (fun ω => euclNorm (dth ω) • dth ω) 1 where
  posSemidef := hH
  const_nonneg := zero_le_one
  chain := fun _ => rfl
  remainder := fun ω => by
    rw [euclNorm_smul, abs_of_nonneg (euclNorm_nonneg _), one_mul]
    exact le_of_eq (by ring)

/-- The collinear remainder of `forwardKLChain_collinear` is nonzero wherever the iterate
fluctuation is. -/
theorem collinear_remainder_ne_zero {ω : Ω} (hne : dth ω ≠ 0) :
    euclNorm (dth ω) • dth ω ≠ 0 :=
  smul_ne_zero (ne_of_gt (euclNorm_pos hne)) hne

/-- **(A4)'s remainder, budgeted.** On a finite weighted sample space with weights summing to
one, if the iterate fluctuation is bounded by `eps` then the remainder functional the coupling
estimate needs, `E[‖δθ̃‖ ‖r‖]`, is at most `Cr eps³`.

This is the paper's "the remainder `tr E[δθ̃ rᵀ] = O(E‖δθ̃‖³)` being a higher-order central
moment", proved with the constant kept rather than asserted, and it is exactly the number `δ`
that `CouplingSign.crossTerm_ge_leading_sub_of_remainder_bound` carries as a hypothesis. -/
theorem ForwardKLChain.remainder_budget [Fintype Ω] {w : Ω → ℝ} {eps : ℝ}
    (h : ForwardKLChain H dth dg r Cr) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (heps : ∀ ω, euclNorm (dth ω) ≤ eps) :
    ∑ ω, w ω * (euclNorm (dth ω) * euclNorm (r ω)) ≤ Cr * eps ^ 3 := by
  have hne : Nonempty Ω := nonempty_of_sum_weights_eq_one hsum
  have heps0 : 0 ≤ eps := le_trans (euclNorm_nonneg _) (heps hne.some)
  have hterm : ∀ ω : Ω, w ω * (euclNorm (dth ω) * euclNorm (r ω)) ≤ w ω * (Cr * eps ^ 3) := by
    intro ω
    refine mul_le_mul_of_nonneg_left ?_ (hw ω)
    have h1 : euclNorm (dth ω) ≤ eps := heps ω
    have h2 : euclNorm (r ω) ≤ Cr * eps ^ 2 := by
      refine le_trans (h.remainder ω) ?_
      have hsq : euclNorm (dth ω) ^ 2 ≤ eps ^ 2 := by
        have := euclNorm_nonneg (dth ω)
        nlinarith
      exact mul_le_mul_of_nonneg_left hsq h.const_nonneg
    have hn1 : 0 ≤ euclNorm (dth ω) := euclNorm_nonneg _
    have hn2 : 0 ≤ euclNorm (r ω) := euclNorm_nonneg _
    have hCe : 0 ≤ Cr * eps ^ 2 := mul_nonneg h.const_nonneg (sq_nonneg _)
    nlinarith
  calc ∑ ω, w ω * (euclNorm (dth ω) * euclNorm (r ω))
      ≤ ∑ ω, w ω * (Cr * eps ^ 3) := Finset.sum_le_sum fun ω _ => hterm ω
    _ = (∑ ω, w ω) * (Cr * eps ^ 3) := by rw [Finset.sum_mul]
    _ = Cr * eps ^ 3 := by rw [hsum, one_mul]

/-- **The paper's leading-order sign claim, in (A4)'s own terms.** With the remainder deleted the
slack-channel cross-covariance is `Σ_slack = V H`, and `σ_Jac² = tr(V H) ≥ 0` because the trace of
a product of two positive-semidefinite matrices is non-negative. The positive semidefiniteness of
`H` is (A4)'s own field; that of `V = E[δθ̃ δθ̃ᵀ]` is a theorem about second moments and is proved
in `CouplingSign.outerMoment_self_posSemidef`. This is the step the appendix calls "the inequality
holding because the trace of the product of the two PSD matrices `V` and `H` is non-negative", and
it is `TraceLemmas.trace_mul_nonneg_of_posSemidef`. -/
theorem ForwardKLChain.trace_leading_nonneg [DecidableEq n] {V : Matrix n n ℝ}
    (h : ForwardKLChain H dth dg r Cr) (hV : V.PosSemidef) : 0 ≤ (V * H).trace :=
  trace_mul_nonneg_of_posSemidef hV h.posSemidef

/-- **(A4) supplies the modulus that (A4'-q) asks for.** With the iterate fluctuation bounded by
`eps` and an operator bound `‖H v‖ ≤ M ‖v‖`, the remainder pairing of (A4'-q) obeys

`|⟨H x, E[δθ̃ rᵀ] x⟩| ≤ M Cr eps³ ‖x‖²`,

so the note's `rho` may be taken to be `M Cr eps³`. The left-hand side is written with the
second-moment matrix in the unfolded form `∑_ω w(ω) δθ̃(ω) r(ω)ᵀ`, which is the definition of
`CouplingSign.outerMoment w δθ̃ r`. -/
theorem ForwardKLChain.pairing_bound [Fintype Ω] {w : Ω → ℝ} {M eps : ℝ}
    (h : ForwardKLChain H dth dg r Cr) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hop : ∀ v : n → ℝ, euclNorm (H *ᵥ v) ≤ M * euclNorm v)
    (heps : ∀ ω, euclNorm (dth ω) ≤ eps) (hM : 0 ≤ M) (x : n → ℝ) :
    |(H *ᵥ x) ⬝ᵥ ((∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) *ᵥ x)|
      ≤ (M * Cr * eps ^ 3) * (x ⬝ᵥ x) := by
  have hexp : (H *ᵥ x) ⬝ᵥ ((∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) *ᵥ x)
      = ∑ ω, w ω * ((r ω ⬝ᵥ x) * ((H *ᵥ x) ⬝ᵥ dth ω)) := by
    rw [Matrix.sum_mulVec, dotProduct_sum]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [Matrix.smul_mulVec, vecMulVec_mulVec_smul, dotProduct_smul, dotProduct_smul,
      smul_eq_mul, smul_eq_mul]
  have hMx : 0 ≤ M * euclNorm x ^ 2 := mul_nonneg hM (sq_nonneg _)
  have hterm : ∀ ω : Ω, |w ω * ((r ω ⬝ᵥ x) * ((H *ᵥ x) ⬝ᵥ dth ω))|
      ≤ (M * euclNorm x ^ 2) * (w ω * (euclNorm (dth ω) * euclNorm (r ω))) := by
    intro ω
    have h1 : |r ω ⬝ᵥ x| ≤ euclNorm (r ω) * euclNorm x := abs_dotProduct_le_euclNorm_mul _ _
    have h2 : |(H *ᵥ x) ⬝ᵥ dth ω| ≤ euclNorm (H *ᵥ x) * euclNorm (dth ω) :=
      abs_dotProduct_le_euclNorm_mul _ _
    have habs : |w ω * ((r ω ⬝ᵥ x) * ((H *ᵥ x) ⬝ᵥ dth ω))|
        = w ω * (|r ω ⬝ᵥ x| * |(H *ᵥ x) ⬝ᵥ dth ω|) := by
      rw [abs_mul, abs_mul, abs_of_nonneg (hw ω)]
    have key : |r ω ⬝ᵥ x| * |(H *ᵥ x) ⬝ᵥ dth ω|
        ≤ (M * euclNorm x ^ 2) * (euclNorm (dth ω) * euclNorm (r ω)) :=
      calc |r ω ⬝ᵥ x| * |(H *ᵥ x) ⬝ᵥ dth ω|
          ≤ (euclNorm (r ω) * euclNorm x) * (euclNorm (H *ᵥ x) * euclNorm (dth ω)) :=
            mul_le_mul h1 h2 (abs_nonneg _)
              (mul_nonneg (euclNorm_nonneg _) (euclNorm_nonneg _))
        _ ≤ (euclNorm (r ω) * euclNorm x) * ((M * euclNorm x) * euclNorm (dth ω)) := by
            refine mul_le_mul_of_nonneg_left ?_
              (mul_nonneg (euclNorm_nonneg _) (euclNorm_nonneg _))
            exact mul_le_mul_of_nonneg_right (hop x) (euclNorm_nonneg _)
        _ = (M * euclNorm x ^ 2) * (euclNorm (dth ω) * euclNorm (r ω)) := by ring
    rw [habs]
    calc w ω * (|r ω ⬝ᵥ x| * |(H *ᵥ x) ⬝ᵥ dth ω|)
        ≤ w ω * ((M * euclNorm x ^ 2) * (euclNorm (dth ω) * euclNorm (r ω))) :=
          mul_le_mul_of_nonneg_left key (hw ω)
      _ = (M * euclNorm x ^ 2) * (w ω * (euclNorm (dth ω) * euclNorm (r ω))) := by ring
  calc |(H *ᵥ x) ⬝ᵥ ((∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) *ᵥ x)|
      = |∑ ω, w ω * ((r ω ⬝ᵥ x) * ((H *ᵥ x) ⬝ᵥ dth ω))| := by rw [hexp]
    _ ≤ ∑ ω, |w ω * ((r ω ⬝ᵥ x) * ((H *ᵥ x) ⬝ᵥ dth ω))| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ ω, (M * euclNorm x ^ 2) * (w ω * (euclNorm (dth ω) * euclNorm (r ω))) :=
        Finset.sum_le_sum fun ω _ => hterm ω
    _ = (M * euclNorm x ^ 2) * ∑ ω, w ω * (euclNorm (dth ω) * euclNorm (r ω)) := by
        rw [Finset.mul_sum]
    _ ≤ (M * euclNorm x ^ 2) * (Cr * eps ^ 3) :=
        mul_le_mul_of_nonneg_left (h.remainder_budget hw hsum heps) hMx
    _ = (M * Cr * eps ^ 3) * (x ⬝ᵥ x) := by rw [← euclNorm_sq x]; ring

/-- **(A4)'s chain, at the level of the second moments FW-1 works with.** The slack-channel
cross-covariance factors as `Σ_slack = V H + R` with `V = E[δθ̃ δθ̃ᵀ]` and `R = E[δθ̃ rᵀ]`. This
is the substitution the design note performs in the proof of (FW-1b), written on the finite
weighted sample space and proved from (A4) rather than asserted. It is the same statement as
`CouplingSign.outerMoment_chain`, restated here so that the bridges below do not have to import
the mechanism module. -/
theorem ForwardKLChain.outerMoment_chain [Fintype Ω] (h : ForwardKLChain H dth dg r Cr)
    (w : Ω → ℝ) :
    ∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω)
      = (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H
        + ∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω) := by
  have hHt : Hᵀ = H := h.transpose_eq
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun ω _ => ?_
  have hstep : Matrix.vecMulVec (dth ω) (dth ω) * H
      = Matrix.vecMulVec (dth ω) (H *ᵥ dth ω) := by
    rw [Matrix.vecMulVec_mul, ← hHt, Matrix.vecMul_transpose, hHt]
  rw [h.chain ω, Matrix.vecMulVec_add, smul_add, smul_mul_assoc, hstep]

/-- **The exact decomposition of the coupling term of (A6) under (A4).** The quadratic form of
`C = H Σ_slack + Σ_slackᵀ H` is twice the magnitude channel plus twice the remainder pairing:

`eᵀ C e = 2 eᵀ (H V H) e + 2 ⟨H e, R e⟩`.

Nothing is dropped: the second summand is the term the paper discards as higher order, and the
two bridges below bound it rather than delete it. -/
theorem ForwardKLChain.coupling_quadForm_split [Fintype Ω] (h : ForwardKLChain H dth dg r Cr)
    (w : Ω → ℝ) (e : n → ℝ) :
    e ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω))
            + (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω))ᵀ * H) *ᵥ e)
      = 2 * (e ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H) *ᵥ e))
        + 2 * ((H *ᵥ e) ⬝ᵥ ((∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) *ᵥ e)) := by
  set V := ∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω) with hV
  set R := ∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω) with hR
  set S := ∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω) with hS
  have hHt : Hᵀ = H := h.transpose_eq
  have hchain : S = V * H + R := h.outerMoment_chain w
  have hx : e ᵥ* H = H *ᵥ e := by
    conv_rhs => rw [← hHt]
    rw [Matrix.mulVec_transpose]
  have hpair : e ⬝ᵥ ((H * S + Sᵀ * H) *ᵥ e) = 2 * ((H *ᵥ e) ⬝ᵥ (S *ᵥ e)) :=
    dotProduct_symmPairing hHt S e
  have h1 : (V * H) *ᵥ e = V *ᵥ (H *ᵥ e) := by rw [Matrix.mulVec_mulVec]
  have h2 : (H * V * H) *ᵥ e = H *ᵥ (V *ᵥ (H *ᵥ e)) := by
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec]
  have key : e ⬝ᵥ (H *ᵥ (V *ᵥ (H *ᵥ e))) = (H *ᵥ e) ⬝ᵥ (V *ᵥ (H *ᵥ e)) := by
    rw [Matrix.dotProduct_mulVec, hx]
  have hVH : (H *ᵥ e) ⬝ᵥ ((V * H) *ᵥ e) = e ⬝ᵥ ((H * V * H) *ᵥ e) := by rw [h1, h2, key]
  rw [hpair, hchain, Matrix.add_mulVec, dotProduct_add, hVH]
  ring

/-- **The coupling term's distance from its leading part, with (A4)'s constant kept.** Under the
jitter bound `‖δθ̃‖ ≤ eps` and the operator bound `‖H v‖ ≤ M ‖v‖`,

`|eᵀ C e − 2 eᵀ (H V H) e| ≤ 2 M Cr eps³ ‖e‖²`.

This is the estimate that `coupling_pos_of_small_jitter` assumes abstractly, supplied from (A4)
instead of asserted in prose: the constant `c` of that lemma may be taken to be
`2 M Cr (e ⬝ᵥ e)`. -/
theorem ForwardKLChain.abs_coupling_sub_leading_le [Fintype Ω] {w : Ω → ℝ} {M eps : ℝ}
    (h : ForwardKLChain H dth dg r Cr) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hop : ∀ v : n → ℝ, euclNorm (H *ᵥ v) ≤ M * euclNorm v)
    (heps : ∀ ω, euclNorm (dth ω) ≤ eps) (hM : 0 ≤ M) (e : n → ℝ) :
    |e ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω))
              + (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω))ᵀ * H) *ᵥ e)
        - 2 * (e ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H) *ᵥ e))|
      ≤ 2 * (M * Cr * eps ^ 3) * (e ⬝ᵥ e) := by
  rw [h.coupling_quadForm_split w e]
  have hb := h.pairing_bound hw hsum hop heps hM e
  rw [show 2 * (e ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H) *ᵥ e))
        + 2 * ((H *ᵥ e) ⬝ᵥ ((∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) *ᵥ e))
        - 2 * (e ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H) *ᵥ e))
      = 2 * ((H *ᵥ e) ⬝ᵥ ((∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) *ᵥ e)) from by ring]
  rw [abs_mul, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 2)]
  linarith

/-- **(A4) forces the cross-covariance to be second order in the jitter scale.**

With `‖δθ̃‖ ≤ eps` on the sample space and an operator bound `‖H v‖ ≤ M ‖v‖`, the chain gives

`|σ_Jac²| = |E[δθ̃ · δg]| ≤ M eps² + Cr eps³`.

The left-hand side is the trace of the slack-channel cross-covariance (`trace_weightedOuter`).
The bound is stated here because it is a genuine consistency check on the design note, and it
fails one: section 11.0 of `docs/design/lemma1_rederivation.md` argues that the coupling channel
is **first** order in the jitter scale, `Σ_slack = O(ρ eps G)`, on the ground that "`δg` is set by
the minibatch, NOT by the lift, so it does not shrink with `eps`". Under (A4) it does shrink:
`δg = H δθ̃ + r`, so `‖δg‖ ≤ M eps + Cr eps²`, and the coupling channel is second order, the same
order as the magnitude channel `H V H`. The two readings cannot both hold. Nothing downstream of
this file depends on the first-order reading -- it is used in the note only to argue how far the
matched-noise bracket of E1 had to be inflated -- but the scaling argument in section 11.0 should
not be quoted alongside (A4) without this caveat. -/
theorem ForwardKLChain.abs_traceCrossCov_le [Fintype Ω] {w : Ω → ℝ} {M eps : ℝ}
    (h : ForwardKLChain H dth dg r Cr) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hop : ∀ v : n → ℝ, euclNorm (H *ᵥ v) ≤ M * euclNorm v)
    (heps : ∀ ω, euclNorm (dth ω) ≤ eps) (hM : 0 ≤ M) :
    |∑ ω, w ω * (dth ω ⬝ᵥ dg ω)| ≤ M * eps ^ 2 + Cr * eps ^ 3 := by
  have hne : Nonempty Ω := nonempty_of_sum_weights_eq_one hsum
  have heps0 : 0 ≤ eps := le_trans (euclNorm_nonneg _) (heps hne.some)
  have hpt : ∀ ω : Ω, |dth ω ⬝ᵥ dg ω| ≤ M * eps ^ 2 + Cr * eps ^ 3 := by
    intro ω
    have hsplit : dth ω ⬝ᵥ dg ω = dth ω ⬝ᵥ (H *ᵥ dth ω) + dth ω ⬝ᵥ r ω := by
      rw [h.chain ω, dotProduct_add]
    have hn1 : 0 ≤ euclNorm (dth ω) := euclNorm_nonneg _
    have hA : |dth ω ⬝ᵥ (H *ᵥ dth ω)| ≤ M * eps ^ 2 := by
      refine le_trans (abs_dotProduct_le_euclNorm_mul _ _) ?_
      have h1 : euclNorm (H *ᵥ dth ω) ≤ M * eps := by
        refine le_trans (hop _) ?_
        exact mul_le_mul_of_nonneg_left (heps ω) hM
      have h2 : euclNorm (dth ω) ≤ eps := heps ω
      have hMe : 0 ≤ M * eps := mul_nonneg hM heps0
      nlinarith
    have hB : |dth ω ⬝ᵥ r ω| ≤ Cr * eps ^ 3 := by
      refine le_trans (abs_dotProduct_le_euclNorm_mul _ _) ?_
      have h1 : euclNorm (r ω) ≤ Cr * eps ^ 2 := by
        refine le_trans (h.remainder ω) ?_
        have hsq : euclNorm (dth ω) ^ 2 ≤ eps ^ 2 := by nlinarith [heps ω]
        exact mul_le_mul_of_nonneg_left hsq h.const_nonneg
      have h2 : euclNorm (dth ω) ≤ eps := heps ω
      have hCe : 0 ≤ Cr * eps ^ 2 := mul_nonneg h.const_nonneg (sq_nonneg _)
      have hn2 : 0 ≤ euclNorm (r ω) := euclNorm_nonneg _
      nlinarith
    calc |dth ω ⬝ᵥ dg ω| = |dth ω ⬝ᵥ (H *ᵥ dth ω) + dth ω ⬝ᵥ r ω| := by rw [hsplit]
      _ ≤ |dth ω ⬝ᵥ (H *ᵥ dth ω)| + |dth ω ⬝ᵥ r ω| := abs_add_le _ _
      _ ≤ M * eps ^ 2 + Cr * eps ^ 3 := by linarith
  calc |∑ ω, w ω * (dth ω ⬝ᵥ dg ω)|
      ≤ ∑ ω, |w ω * (dth ω ⬝ᵥ dg ω)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ ω, w ω * (M * eps ^ 2 + Cr * eps ^ 3) := by
        refine Finset.sum_le_sum fun ω _ => ?_
        rw [abs_mul, abs_of_nonneg (hw ω)]
        exact mul_le_mul_of_nonneg_left (hpt ω) (hw ω)
    _ = (∑ ω, w ω) * (M * eps ^ 2 + Cr * eps ^ 3) := by rw [Finset.sum_mul]
    _ = M * eps ^ 2 + Cr * eps ^ 3 := by rw [hsum, one_mul]

/-- **Where (A6) is not needed: the small-jitter regime, in its scalar core.** If the magnitude
channel is non-degenerate at the scale of the jitter, `lam eps² ≤ 2 Q` with `Q = eᵀ H V H e`, and
the coupling term `X = eᵀ C e` differs from its leading part `2 Q` by at most `c eps³`, then
`X > 0` as soon as `c eps < lam`.

The second hypothesis is not an assumption about `C`: it is what (A4)'s quadratic remainder
delivers, and `ForwardKLChain.abs_coupling_sub_leading_le` proves it with `c = 2 M Cr (e ⬝ᵥ e)`,
so the conclusion is assembled into (A6) proper by `couplingAlignedOn_of_small_jitter` below.
This lemma is kept separate because it is the arithmetic of the regime and holds of any three
numbers standing in that relation. Positive alignment is therefore automatic for small enough
jitter under non-degeneracy, and assumption (A6) is doing work exactly in the regime where the
jitter is not small compared with `lam / c` -- which is where
`CouplingSign.coupling_sign_not_automatic_under_remainder_control` puts its counterexample. -/
theorem coupling_pos_of_small_jitter {lam c eps X Q : ℝ}
    (hQ : lam * eps ^ 2 ≤ 2 * Q) (hX : |X - 2 * Q| ≤ c * eps ^ 3)
    (heps : 0 < eps) (hlt : c * eps < lam) : 0 < X := by
  have h1 : -(c * eps ^ 3) ≤ X - 2 * Q := (abs_le.mp hX).1
  nlinarith [mul_pos (pow_pos heps 2) (sub_pos.mpr hlt)]

/-! ## (A4'-q) the remainder-control clause FW-1 adds -/

/-- **(A4'-q)**, the remainder-control clause of FW-1 (section 8 of the design note), in
quadratic-form language.

The note writes it with operator norms and a smallest eigenvalue,
`‖E[δθ̃ rᵀ]‖_op ≤ (3/2) λ_min(H̄ V H̄) / ‖H̄‖_op`. The quadratic-form version carried here is what
the proof of (FW-1b) actually uses: a lower bound `lam` for the magnitude channel `H V H`, an
upper bound `rho` for the pairing `⟨H x, R x⟩` with `R = E[δθ̃ rᵀ]`, and the control `2 rho ≤ 3
lam`. It is stated this way, rather than through mathlib's scoped operator norm and the
eigenvalue API, because the conclusion is a quadratic-form inequality and the translation through
norms would only lose constants. -/
structure RemainderControl {n : Type*} [Fintype n] (H V R : Matrix n n ℝ) (lam rho : ℝ) : Prop where
  /-- the mean Hessian is symmetric -/
  symm : Hᵀ = H
  /-- the magnitude channel's modulus is non-negative -/
  lam_nonneg : 0 ≤ lam
  /-- the remainder's modulus is non-negative -/
  rho_nonneg : 0 ≤ rho
  /-- `lam` is a lower bound for the magnitude channel -/
  magnitude_lower : ∀ x : n → ℝ, lam * (x ⬝ᵥ x) ≤ x ⬝ᵥ ((H * V * H) *ᵥ x)
  /-- `rho` is an upper bound for the remainder pairing -/
  pairing_upper : ∀ x : n → ℝ, |(H *ᵥ x) ⬝ᵥ (R *ᵥ x)| ≤ rho * (x ⬝ᵥ x)
  /-- the control clause itself -/
  control : 2 * rho ≤ 3 * lam

/-- **(FW-1b) under (A4'-q), quantitatively.** The substituted excess
`3 H V H + (H R + Rᵀ H)` has quadratic form at least `(3 lam − 2 rho) ‖x‖²`. -/
theorem RemainderControl.excess_lower_bound {V R : Matrix n n ℝ} {lam rho : ℝ}
    (h : RemainderControl H V R lam rho) (x : n → ℝ) :
    (3 * lam - 2 * rho) * (x ⬝ᵥ x)
      ≤ x ⬝ᵥ (((3 : ℝ) • (H * V * H) + (H * R + Rᵀ * H)) *ᵥ x) := by
  have hsplit : x ⬝ᵥ (((3 : ℝ) • (H * V * H) + (H * R + Rᵀ * H)) *ᵥ x)
      = 3 * (x ⬝ᵥ ((H * V * H) *ᵥ x)) + x ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ x) := by
    rw [Matrix.add_mulVec, dotProduct_add, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
  have hpair : x ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ x) = 2 * ((H *ᵥ x) ⬝ᵥ (R *ᵥ x)) :=
    dotProduct_symmPairing h.symm R x
  have hlow := h.magnitude_lower x
  have hupp := (abs_le.mp (h.pairing_upper x)).1
  rw [hsplit, hpair]
  linarith

/-- **(FW-1b) under (A4'-q).** The substituted excess is positive semidefinite: this is the
sentence "E is PSD whenever the remainder-control clause holds" of section 8, with the
quadratic-form witness. -/
theorem RemainderControl.excess_nonneg {V R : Matrix n n ℝ} {lam rho : ℝ}
    (h : RemainderControl H V R lam rho) (x : n → ℝ) :
    0 ≤ x ⬝ᵥ (((3 : ℝ) • (H * V * H) + (H * R + Rᵀ * H)) *ᵥ x) := by
  have hx : 0 ≤ x ⬝ᵥ x := Finset.sum_nonneg fun i _ => mul_self_nonneg _
  have hb := h.excess_lower_bound x
  have hc := h.control
  nlinarith

/-- **(A4) gives (A4'-q), with an explicit modulus.** If the iterate fluctuation is bounded by
`eps`, the mean Hessian obeys the operator bound `‖H v‖ ≤ M ‖v‖`, the magnitude channel is
non-degenerate with modulus `lam`, and the control clause holds at `rho = M Cr eps³`, then the
remainder-control structure of FW-1 holds for the second-moment remainder `R = E[δθ̃ rᵀ]`.

Two things about the statement should be read off it rather than assumed. The matrix `V` is a
free parameter: the theorem passes `hmag` through unchanged and never requires `V` to be the
second moment `E[δθ̃ δθ̃ᵀ]`, so it is more general than the note's clause and the intended
reading is supplied by the caller. And nothing here is asymptotic: the control clause is the
hypothesis `hcontrol`, carried and not taken in a limit. The informal scaling reading -- for the
intended `V`, `lam` is of order `eps²` while `rho` is of order `eps³`, so that `2 rho ≤ 3 lam`
holds once `eps³ ≤ 3 lam / (2 M Cr)` -- is a remark about the intended instance and is not
machine-checked here; `couplingAlignedOn_of_small_jitter` below is the version in which the
`eps`-dependence of the magnitude channel is written into the statement. -/
theorem remainderControl_of_forwardKLChain [Fintype Ω] {V : Matrix n n ℝ} {w : Ω → ℝ}
    {M eps lam : ℝ} (h : ForwardKLChain H dth dg r Cr) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hop : ∀ v : n → ℝ, euclNorm (H *ᵥ v) ≤ M * euclNorm v)
    (heps : ∀ ω, euclNorm (dth ω) ≤ eps) (hM : 0 ≤ M) (hlam : 0 ≤ lam)
    (hmag : ∀ x : n → ℝ, lam * (x ⬝ᵥ x) ≤ x ⬝ᵥ ((H * V * H) *ᵥ x))
    (hcontrol : 2 * (M * Cr * eps ^ 3) ≤ 3 * lam) :
    RemainderControl H V (∑ ω, w ω • Matrix.vecMulVec (dth ω) (r ω)) lam (M * Cr * eps ^ 3) where
  symm := h.transpose_eq
  lam_nonneg := hlam
  rho_nonneg := by
    have heps0 : 0 ≤ eps :=
      le_trans (euclNorm_nonneg _) (heps (nonempty_of_sum_weights_eq_one hsum).some)
    exact mul_nonneg (mul_nonneg hM h.const_nonneg) (pow_nonneg heps0 3)
  magnitude_lower := hmag
  pairing_upper := h.pairing_bound hw hsum hop heps hM
  control := hcontrol

/-! ## (A5) metastable single-barrier regularity -/

/-- The reference action of the barrier, `α = U(w_s) − U(w_b)`. -/
def barrierAction (U : ℝ → ℝ) (wb ws : ℝ) : ℝ := U ws - U wb

/-- **(A5), the geometry.** The bias-channel projection `U` of the pullback landscape is `C²` and
has a single-barrier profile on `[w_b, w_a]`: one interior maximum at `w_s` separating the
pre-barrier basin minimum `w_b` from the post-barrier minimum `w_a`.

What the fields record is the *extremal data* of "one interior maximum separating two minima":
that `w_s` maximizes `U` over the whole interval and that `w_b` and `w_a` minimize it over the
two halves. That is implied by, but weaker than, a literal single-barrier profile -- nothing here
forbids further interior critical points below the barrier top -- and the weaker form is
deliberate, because it is all the exit-time modules consume and a hypothesis should be no
stronger than its use. `singleBarrierPotential_quadratic` witnesses that the fields are jointly
realizable. The `C²` clause is the paper's; it is not consumed by the exit-time modules of this
development, which need only continuity of `U` on the escape interval and the extremal structure,
and this over-strength is recorded in the ledger above. -/
structure SingleBarrierPotential (U : ℝ → ℝ) (wb ws wa : ℝ) : Prop where
  /-- the potential is twice continuously differentiable -/
  contDiff : ContDiff ℝ 2 U
  /-- the basin minimum lies to the left of the barrier top -/
  well_lt_barrier : wb < ws
  /-- the barrier top lies to the left of the far minimum -/
  barrier_lt_exit : ws < wa
  /-- the barrier top is the maximum over the whole interval -/
  isMaxOn : IsMaxOn U (Set.Icc wb wa) ws
  /-- the near basin minimum -/
  isMinOn_left : IsMinOn U (Set.Icc wb ws) wb
  /-- the far basin minimum -/
  isMinOn_right : IsMinOn U (Set.Icc ws wa) wa
  /-- the barrier is genuine -/
  barrier_pos : U wb < U ws

variable {U : ℝ → ℝ} {wb ws wa : ℝ}

theorem SingleBarrierPotential.barrierAction_pos (h : SingleBarrierPotential U wb ws wa) :
    0 < barrierAction U wb ws := by
  have := h.barrier_pos
  rw [barrierAction]
  linarith

/-- The barrier top maximizes `U` on the escape interval `[w_b, w_s]`: this is the hypothesis
`hmax` of `KramersExitTime.meanExitTime_upper_bound`. -/
theorem SingleBarrierPotential.isMaxOn_escape (h : SingleBarrierPotential U wb ws wa) :
    IsMaxOn U (Set.Icc wb ws) ws :=
  h.isMaxOn.on_subset (Set.Icc_subset_Icc_right h.barrier_lt_exit.le)

/-- The basin minimum minimizes `U` on the escape interval: this is the hypothesis `hmin` of
`KramersExitTime.meanExitTime_upper_bound`. -/
theorem SingleBarrierPotential.isMinOn_escape (h : SingleBarrierPotential U wb ws wa) :
    IsMinOn U (Set.Icc wb ws) wb := h.isMinOn_left

/-- On the escape interval the potential is trapped between the basin value and the barrier top,
so the reference action is the full climb. -/
theorem SingleBarrierPotential.mem_bounds (h : SingleBarrierPotential U wb ws wa) {w : ℝ}
    (hw : w ∈ Set.Icc wb ws) : U wb ≤ U w ∧ U w ≤ U ws :=
  ⟨isMinOn_iff.mp h.isMinOn_escape w hw, isMaxOn_iff.mp h.isMaxOn_escape w hw⟩

/-- **(A5)'s geometry is satisfiable.** The downward parabola centred at the barrier top has
exactly the extremal structure `SingleBarrierPotential` records, on every interval straddling
its vertex. The witness is stated because every consequence of (A5) below is an implication out
of that structure, and because the structure is the weakest reading of "single barrier" -- it
constrains the extrema, not the number of critical points -- so a reader should be able to see
that the constraints are jointly realizable rather than take it on trust. -/
theorem singleBarrierPotential_quadratic (h1 : wb < ws) (h2 : ws < wa) :
    SingleBarrierPotential (fun w => -(w - ws) ^ 2) wb ws wa where
  contDiff := by fun_prop
  well_lt_barrier := h1
  barrier_lt_exit := h2
  isMaxOn := by
    intro w _
    simp only [Set.mem_setOf_eq]
    nlinarith [sq_nonneg (w - ws)]
  isMinOn_left := by
    intro w hw
    simp only [Set.mem_setOf_eq]
    obtain ⟨hw1, hw2⟩ := hw
    nlinarith
  isMinOn_right := by
    intro w hw
    simp only [Set.mem_setOf_eq]
    obtain ⟨hw1, hw2⟩ := hw
    nlinarith
  barrier_pos := by nlinarith [sq_nonneg (wb - ws)]

/-- **(A5), the smallness clause.** The effective noise is small relative to the barrier, with
the ratio quantified: `ratio · σ_eff² ≤ α` with `ratio ≥ 1`.

The paper says only "the effective noise is small relative to the barrier". Smallness is a limit
statement, so it is carried here as an explicit inequality with a named ratio, and every
consequence below is quantitative in that ratio; taking `ratio → ∞` recovers the asymptotic
reading. -/
structure SmallEffectiveNoise (sigma2 alpha ratio : ℝ) : Prop where
  /-- the effective diffusion is positive -/
  diffusion_pos : 0 < sigma2
  /-- the barrier action is positive -/
  barrier_pos : 0 < alpha
  /-- the ratio is at least one -/
  one_le_ratio : 1 ≤ ratio
  /-- the smallness clause -/
  small : ratio * sigma2 ≤ alpha

variable {sigma2 alpha ratio : ℝ}

/-- The Arrhenius exponent is at least `2 · ratio`: this is what the smallness clause buys, in
the form the exit-time modules use. -/
theorem SmallEffectiveNoise.arrheniusExponent_ge (h : SmallEffectiveNoise sigma2 alpha ratio) :
    2 * ratio ≤ 2 * alpha / sigma2 := by
  rw [le_div_iff₀ h.diffusion_pos]
  have := h.small
  nlinarith

/-- Hence the predicted escape factor is at least `exp(2 ratio)`. -/
theorem SmallEffectiveNoise.exp_le (h : SmallEffectiveNoise sigma2 alpha ratio) :
    Real.exp (2 * ratio) ≤ Real.exp (2 * alpha / sigma2) :=
  Real.exp_le_exp.mpr h.arrheniusExponent_ge

/-- The smallness clause places the effective diffusion inside the metastable window
`σ² ∈ (0, 2α)` of `FreeDiffusion.metastable_window`, the window on which the Arrhenius exponent
exceeds one and the escape is not free diffusion. -/
theorem SmallEffectiveNoise.lt_two_mul_barrier (h : SmallEffectiveNoise sigma2 alpha ratio) :
    sigma2 < 2 * alpha := by
  have h1 := h.small
  have h2 := h.one_le_ratio
  have h3 := h.diffusion_pos
  have h4 := h.barrier_pos
  nlinarith

/-- **(A5)'s smallness clause is satisfiable at every ratio.** Given any positive effective
diffusion and any ratio at least one, the barrier action `ratio · σ_eff²` realizes the clause
with equality. The witness makes visible that the clause constrains the *pair* `(σ_eff², α)` and
not either alone, which is why the ratio is carried as a parameter rather than hidden in an
asymptotic. -/
theorem smallEffectiveNoise_of_ratio (hs : 0 < sigma2) (hr : 1 ≤ ratio) :
    SmallEffectiveNoise sigma2 (ratio * sigma2) ratio where
  diffusion_pos := hs
  barrier_pos := by nlinarith
  one_le_ratio := hr
  small := le_rfl

/-- **(A5) supplies the exit-time modules' window hypotheses.** On the escape interval the
single-barrier geometry gives literally the two bounds that
`KramersExitTime.meanExitTime_upper_bound` carries as `hmax` and `hmin`, with the gap `M - m`
between them equal to the reference action. -/
theorem SingleBarrierPotential.kramers_window (h : SingleBarrierPotential U wb ws wa) :
    (∀ y ∈ Set.Icc wb ws, U y ≤ U ws) ∧ (∀ z ∈ Set.Icc wb ws, U wb ≤ U z) ∧
      U ws - U wb = barrierAction U wb ws :=
  ⟨fun _ hy => (h.mem_bounds hy).2, fun _ hz => (h.mem_bounds hz).1, rfl⟩

/-! ## (A6) positive alignment of the coupling term on the band -/

/-- **(A6)** of section 11.3 of the design note: on the shoulder band `B` the coupling term
`C(x) = H(x) Σ(x) + Σ(x)ᵀ H(x)` dominates `κ I` with `κ > 0`.

The design note is explicit that this is a *labelled assumption* and not a derived fact: the
magnitude channel `H V H` is positive semidefinite always, but `C` is symmetric and not positive
semidefinite in general. Two counterexamples in `CouplingSign.lean` say how far that goes, and
they are not interchangeable. `coupling_sign_not_automatic` exhibits a configuration with
symmetric positive-definite `H`, positive-definite `V` and a nonzero (A4) remainder on which the
coupling term is strictly negative, but that configuration violates (A4'-q), so it refutes only
"(A4) alone signs the coupling channel".
`coupling_sign_not_automatic_under_remainder_control` and
`positiveAlignment_fails_under_remainder_control` do the sharper job: a configuration satisfying
(A4) *and* (A4'-q) with a nonzero remainder, on which the coupling term is strictly negative and
(A6) fails for every `κ`. It is the second pair that makes (A6) irreducible. The statement here
is written in the same unfolded form as `CouplingSign.PositiveAlignmentOn`, so the two agree by
definition.

What (A6) asserts, in words, is that training drives the latent weight's batch dependence into
positive alignment with the curvature-weighted gradient fluctuation. Section 11.3 records that
this is directly measurable, reducing in one dimension to the sign of `H̄ σ_Jac²`
(`couplingQuadForm_fin_one` below). -/
def CouplingAlignedOn {ι n : Type*} [Fintype n] (B : Set ι) (Hf Sf : ι → Matrix n n ℝ)
    (kappa : ℝ) : Prop :=
  0 < kappa ∧ ∀ x ∈ B, ∀ e : n → ℝ,
    kappa * (e ⬝ᵥ e) ≤ e ⬝ᵥ ((Hf x * Sf x + (Sf x)ᵀ * Hf x) *ᵥ e)

/-- (A6) read through the symmetrized pairing: on the band the inner product of the
curvature-weighted direction with the cross-covariance-weighted direction is bounded below. -/
theorem CouplingAlignedOn.pairing_lower {ι : Type*} {B : Set ι} {Hf Sf : ι → Matrix n n ℝ}
    {kappa : ℝ} (h : CouplingAlignedOn B Hf Sf kappa) (hsym : ∀ x ∈ B, (Hf x)ᵀ = Hf x)
    {x : ι} (hx : x ∈ B) (e : n → ℝ) :
    kappa * (e ⬝ᵥ e) ≤ 2 * ((Hf x *ᵥ e) ⬝ᵥ (Sf x *ᵥ e)) := by
  have hb := h.2 x hx e
  rwa [dotProduct_symmPairing (hsym x hx)] at hb

/-- **(A6) in one dimension**, which is the form section 11.3 says is directly measurable: the
coupling term's quadratic form is `2 H̄ σ_Jac² e²`, so (A6) is a statement about the sign of the
product of the mean curvature and the cross-covariance on the band, and not about `σ_Jac²`
alone. -/
theorem couplingQuadForm_fin_one (Hm Sm : Matrix (Fin 1) (Fin 1) ℝ) (e : Fin 1 → ℝ) :
    e ⬝ᵥ ((Hm * Sm + Smᵀ * Hm) *ᵥ e) = 2 * (Hm 0 0 * Sm 0 0) * e 0 ^ 2 := by
  simp only [dotProduct, Matrix.mulVec, Matrix.add_apply, Matrix.mul_apply,
    Matrix.transpose_apply, Fin.sum_univ_one]
  ring

/-- (A6)'s inequality is satisfiable: an identity mean Hessian and a cross-covariance which is a
positive multiple of the identity give positive alignment with the corresponding modulus, in
every dimension and on every band. This is a witness for the *inequality* only -- it exhibits no
underlying chain, so it does not show that (A6) is compatible with (A4) and a nonzero remainder.
For that, see `couplingAlignedOn_of_small_jitter` together with
`smallJitter_hypotheses_satisfiable` here, and `CouplingSign.positiveAlignment_satisfiable`,
which carries a uniform `κ` over a continuum band in two dimensions. Together with the
refutations cited in `CouplingAlignedOn`'s docstring, these place (A6) where the note puts it --
a genuine, contingent hypothesis. -/
theorem couplingAlignedOn_smul_one {ι : Type*} [DecidableEq n] (B : Set ι) {c : ℝ} (hc : 0 < c) :
    CouplingAlignedOn B (fun _ : ι => (1 : Matrix n n ℝ))
      (fun _ : ι => (c / 2) • (1 : Matrix n n ℝ)) c := by
  refine ⟨hc, fun x _ e => ?_⟩
  have hstep : ((1 : Matrix n n ℝ) * ((c / 2) • (1 : Matrix n n ℝ))
      + ((c / 2) • (1 : Matrix n n ℝ))ᵀ * (1 : Matrix n n ℝ)) = c • (1 : Matrix n n ℝ) := by
    rw [Matrix.transpose_smul, Matrix.transpose_one, Matrix.one_mul, Matrix.mul_one]
    rw [← add_smul]
    congr 1
    ring
  rw [hstep, Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct_smul, smul_eq_mul]

/-- **(A6) derived from (A4) in the small-jitter regime, with every constant written out.**

If the mean Hessian obeys an operator bound `‖H v‖ ≤ M ‖v‖`, the iterate fluctuation is bounded
by `eps`, and the magnitude channel is non-degenerate *at the scale of the jitter*, meaning
`2 eᵀ (H V H) e ≥ lam eps² ‖e‖²` with `V = E[δθ̃ δθ̃ᵀ]`, then the coupling term satisfies (A6)
on any band, with the explicit modulus `κ = (lam − 2 M Cr eps) eps²`, as soon as
`2 M Cr eps < lam`.

This is the machine-checked form of the informal remark attached to
`coupling_pos_of_small_jitter`: the leading part of the coupling term is `2 eᵀ (H V H) e`, which
is second order in the jitter, and (A4)'s remainder perturbs it by at most `2 M Cr eps³ ‖e‖²`,
which is third. It does *not* discharge (A6): outside this regime (A6) fails, and it fails even
under the remainder-control clause (A4'-q) --
`CouplingSign.positiveAlignment_fails_under_remainder_control` refutes it for every `κ` on a
configuration that satisfies (A4'-q) with a nonzero remainder. What the theorem delimits is
where the assumption is doing work: at a jitter scale comparable with `lam / (2 M Cr)`. -/
theorem couplingAlignedOn_of_small_jitter [Fintype Ω] {ι : Type*} (Bnd : Set ι) {w : Ω → ℝ}
    {M jit lam : ℝ} (h : ForwardKLChain H dth dg r Cr) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hop : ∀ v : n → ℝ, euclNorm (H *ᵥ v) ≤ M * euclNorm v)
    (heps : ∀ ω, euclNorm (dth ω) ≤ jit) (hM : 0 ≤ M) (hjit : 0 < jit)
    (hmag : ∀ x : n → ℝ, lam * jit ^ 2 * (x ⬝ᵥ x)
      ≤ 2 * (x ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H) *ᵥ x)))
    (hlt : 2 * M * Cr * jit < lam) :
    CouplingAlignedOn Bnd (fun _ => H)
      (fun _ => ∑ ω, w ω • Matrix.vecMulVec (dth ω) (dg ω))
      ((lam - 2 * M * Cr * jit) * jit ^ 2) := by
  refine ⟨mul_pos (by linarith) (pow_pos hjit 2), fun _ _ e => ?_⟩
  have hb := h.abs_coupling_sub_leading_le hw hsum hop heps hM e
  have hlow := (abs_le.mp hb).1
  have hm := hmag e
  have hee : (0 : ℝ) ≤ e ⬝ᵥ e := Finset.sum_nonneg fun _ _ => mul_self_nonneg _
  nlinarith [hlow, hm, hee]

/-- **The hypotheses of `couplingAlignedOn_of_small_jitter` are jointly satisfiable, with a
nonzero remainder.** The non-degeneracy clause `2 eᵀ (H V H) e ≥ lam jit² ‖e‖²` constrains the
second moment of a fluctuation that the jitter bound simultaneously forces to be small, so a
reader is entitled to ask whether the two can hold at once; they can. The configuration is
one-dimensional -- which is the form section 11.3 of the design note calls directly measurable,
`C = 2 H̄ σ_Jac²` -- with `H = 1`, `δθ̃ ≡ 1/2`, `r ≡ 1/4` saturating (A4)'s bound at `Cr = 1`,
`jit = 1/2` and `lam = 2`. A band-indexed, two-dimensional witness with a uniform `κ` is
`CouplingSign.positiveAlignment_satisfiable`. -/
theorem smallJitter_hypotheses_satisfiable :
    ∃ (w : Fin 1 → ℝ) (dth dg r : Fin 1 → Fin 1 → ℝ) (H : Matrix (Fin 1) (Fin 1) ℝ)
      (Cr M jit lam : ℝ),
      ForwardKLChain H dth dg r Cr ∧ r ≠ 0 ∧
      (∀ ω, 0 ≤ w ω) ∧ (∑ ω, w ω = 1) ∧
      (∀ v : Fin 1 → ℝ, euclNorm (H *ᵥ v) ≤ M * euclNorm v) ∧
      (∀ ω, euclNorm (dth ω) ≤ jit) ∧ 0 ≤ M ∧ 0 < jit ∧
      (∀ x : Fin 1 → ℝ, lam * jit ^ 2 * (x ⬝ᵥ x)
        ≤ 2 * (x ⬝ᵥ ((H * (∑ ω, w ω • Matrix.vecMulVec (dth ω) (dth ω)) * H) *ᵥ x))) ∧
      2 * M * Cr * jit < lam := by
  refine ⟨fun _ => 1, fun _ _ => 1/2, fun _ _ => 3/4, fun _ _ => 1/4, 1, 1, 1, 1/2, 2, ?_, ?_,
    fun _ => by norm_num, by simp, ?_, ?_, by norm_num, by norm_num, ?_, by norm_num⟩
  · refine { posSemidef := ?_, const_nonneg := zero_le_one, chain := ?_, remainder := ?_ }
    · simpa using (Matrix.PosSemidef.one (n := Fin 1) (R := ℝ))
    · intro _
      funext i
      simp only [Matrix.one_mulVec, Pi.add_apply]
      norm_num
    · intro _
      rw [euclNorm_const_fin_one (by norm_num : (0:ℝ) ≤ 1/4),
        euclNorm_const_fin_one (by norm_num : (0:ℝ) ≤ 1/2)]
      norm_num
  · intro hcon
    have hval := congrFun (congrFun hcon 0) 0
    norm_num at hval
  · intro v
    rw [Matrix.one_mulVec, one_mul]
  · intro _
    rw [euclNorm_const_fin_one (by norm_num : (0:ℝ) ≤ 1/2)]
  · intro x
    have hV : ((1 : Matrix (Fin 1) (Fin 1) ℝ)
        * (∑ _ω : Fin 1, (1:ℝ) • Matrix.vecMulVec (fun _ : Fin 1 => (1/2:ℝ)) (fun _ => (1/2:ℝ)))
        * (1 : Matrix (Fin 1) (Fin 1) ℝ)) *ᵥ x
        = ((fun _ : Fin 1 => (1/2:ℝ)) ⬝ᵥ x) • (fun _ : Fin 1 => (1/2:ℝ)) := by
      rw [Matrix.one_mul, Matrix.mul_one, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, one_smul, one_smul, vecMulVec_mulVec_smul]
    rw [hV, dotProduct_smul, smul_eq_mul]
    simp only [dotProduct, Fin.sum_univ_one]
    nlinarith [sq_nonneg (x 0)]

/-! ## The attribution check the appendix invites -/

/-- **Cases (i) and (ii) of Theorem 1 carry no assumption at all.**

The appendix's assumption block says "\cref{thm:joint-necessity} uses only (A1)". The two
finite-sample deletions do not use even that: the statement below has no assumption structure
among its hypotheses, only the deletion hypotheses of the theorem itself, and it is proved from
`CrossCov.lean`'s identities. It is stated for every iteration `t` and every window length `T`,
the empty window included -- `CrossCov.crossCovEstimator_eq_zero_of_body_const` needs `T ≠ 0`
because the window mean of `bodyFluct` is undefined there, and `crossCovEstimator_zero_window`
supplies the missing case, so the quantifier here is the paper's own "for every `t` and every
window length `T`" without a side condition. (A1)'s i.i.d. clause is used by case (iii), and there only for the
estimator's almost-sure limit and its `L²` rate; the population zero of case (iii) needs
conditional independence and integrability, not the i.i.d. structure of the window. -/
theorem crossCovEstimator_zero_window {d : ℕ} (t : ℕ) (dth dg : ℕ → Fin d → ℝ) :
    crossCovEstimator 0 t dth dg = 0 := by
  simp [crossCovEstimator]

theorem joint_necessity_cases_assumption_free {d : ℕ} {Ω' : Type*}
    {Jb : Matrix (Fin d) (Fin d) ℝ} (hJ : Jb = 0) (T t : ℕ) (dth dg : ℕ → Fin d → ℝ)
    {hb : Ω' → Fin d → ℝ} (hconst : ∀ x y, hb x = hb y) (Xs : ℕ → Ω') :
    slackReading Jb T t dth dg = 0 ∧ crossCovEstimator T t (bodyFluct hb Xs T t) dg = 0 := by
  refine ⟨slackReading_eq_zero_of_jacobian_eq_zero hJ T t dth dg, ?_⟩
  rcases Nat.eq_zero_or_pos T with hT | hT
  · subst hT
    exact crossCovEstimator_zero_window t _ dg
  · exact crossCovEstimator_eq_zero_of_body_const hconst Xs dg T t hT.ne'

end IcnnLift
