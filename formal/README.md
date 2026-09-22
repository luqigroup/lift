# `formal/` — the Lean 4 development behind Section 3 of the ICLR paper

This directory is the machine-checked counterpart of the paper's theory section.
It exists so that every theorem, lemma, corollary and displayed identity that
Section 3 and Appendix A state can be traced to a kernel-verified declaration.

**The paper itself contains no Lean surface.** By the author's rule of
2026-09-03 the manuscript is English and mathematics only: no module names, no
theorem names, no `\lean{}` pointers, not in the body and not in the appendix.
The proofs in Appendix A follow the setup of the Lean proofs step for step —
same decomposition, same intermediate objects, same named hypotheses — but they
are written as ordinary mathematical prose. The formalization is acknowledged
in exactly one sentence, at the head of Section 3. **This file is
where the statement-by-statement correspondence lives.**

## Status

Last audited 2026-09-13, after the four steps the paper reported as outside the formalization were discharged or narrowed, and the reversal identity of `prop:clamp` and its passage to the limit were proved.

- `lake build` was last run green over the 31 modules of 2026-09-03 (8619
  jobs). The two modules added on 2026-09-13, `ShoulderDwell` and
  `ProjectionClamp`, were type-checked with `./check.sh` and their `.olean`s
  emitted directly, so that no build lock was taken while other agents were
  working; both are registered in `lakefile.toml` and the next `lake build`
  rebuilds them in the ordinary way.
- No `sorry`, no `admit`, no `native_decide`, no `axiom` anywhere in the tree
  (strict sweep with block comments, docstrings and line comments stripped,
  over all 33 files).
- **Exhaustive axiom sweep: 2692 declarations, of which 2310 are theorems,
  every one depending on exactly `propext`, `Classical.choice`, `Quot.sound`.
  Zero non-standard axioms, zero `sorryAx`, zero `axiom` declarations of our
  own.** The sweep enumerates the environment rather than parsing source, so
  it covers sub-namespace witnesses and wrapped declaration headers alike.
- Reproduce the sweep with `./check.sh AxiomAudit.lean`. `AxiomAudit.lean`
  imports all 33 modules, walks every constant whose defining module is one of
  ours, and runs the kernel's own axiom collection on each; it prints the four
  counts above. It is deliberately **not** in `lakefile.toml`'s
  `defaultTargets`, so `lake build` does not pay for it.
- `./check.sh <File>.lean` type-checks one file without taking the build lock.
  `./axioms.sh <File>.lean` remains as a per-file spot check; it parses source
  and cannot resolve names that wrap over three lines or sub-namespace
  witnesses it cannot reach by `open IcnnLift`, which is why `AxiomAudit.lean`
  is the audit of record.

## What is assumed rather than proved

Five modeling steps enter the development as explicit hypotheses rather than as
theorems, and the paper says so in the same words:

1. **The Itô surrogate of (A1).** That the SGD recursion may be read as an Itô
   process is the register of the stochastic modified equations of Li et al.;
   it is carried by the structure `SdeOfSgdModel`, not derived.
2. **Dynkin's formula for the exit-time boundary-value problem of `cor:fpt`.**
   The mean-exit-time theorems take the boundary-value problem as a hypothesis
   (`MeanExitTimeBVP`) and prove everything downstream of it.
3. **Dynkin's formula for the occupation-time boundary-value problem of
   `cor:dwell`** (added 2026-09-13). The dwell theorems take
   `ShoulderDwell.OccupationTimeBVP` at the band indicator as a hypothesis. It
   is the *same* modeling input as (2), applied to the occupation functional
   rather than the exit time, which is what the appendix proof says, and it is
   satisfiable: `ShoulderDwell.dwellTime_satisfies_bvp` exhibits a solution and
   `dwell_hypotheses_satisfiable` the whole stack.
4. **The quasi-static reading of Adam's second-moment estimate (`rem:adam`)**
   (added 2026-09-13). That the Adam escape exponent *is* `2α/η · (√V + ε)/V`
   packages the constancy of the second-moment estimate across the band and the
   identical action of the normalization on the two constructions. It is carried
   by `PgdBaseline.AdamQuasiStatic` and inhabited by
   `.adamQuasiStatic_satisfiable`; everything downstream of it — the strict
   monotonicity of the map and the equivalence of the two orderings — is proved.
4b. **Continuity of the exit and occupation times up to the closed interval**
   (added 2026-09-15 by the Section 3 load-bearing audit). `htau` / `hD` :
   `ContinuousOn (tau s) (Set.Icc a b)` is bound **separately from** `hDynkin`
   in every paper-facing declaration
   (`KramersExitTime.mean_first_passage_kramers_exponent_of_dynkin:1316`,
   `KramersBridge.eventually_mean_first_passage_lift_lt_direct_smallLearningRate:359-360`,
   `ShoulderDwell.mean_dwell_kramers_exponent_of_dynkin:1391`,
   `.mean_dwell_ratio_of_dynkin:1417-1418`). It is *not* implied by
   `MeanExitTimeBVP`, which gives differentiability on the **open** `Ioo a b`
   only; it is consumed by `eq_meanExitTime_of_bvp:471` as `hTcont`, whose
   docstring reads "any solution that is continuous up to the closed interval".
   Mathematically innocuous — the true mean exit time is continuous — but it is
   the same *kind* of object as the Dynkin clause, so the paper carries it in
   the same place: Theorem 1(iii), `cor:fpt` and `cor:dwell` all end their
   Dynkin clause with "and continuous up to the closed interval".
5. **The stationary law of the projected recursion (`prop:clamp`)** (added
   2026-09-13, narrowed three times the same day). What remains unformalized is
   the Spitzer ladder-epoch identity and the limit `0` as `ρ ↓ 0` that it
   supplies, together with the stationarity and uniqueness of the limit law,
   which no declaration states. Everything else the clause asserts is now
   proved: the reversal identity, `W_k → M` in distribution, and the clamped
   fraction is scale-free, non-decreasing in `ρ` and tends to `1` as `ρ → ∞`,
   at every finite step; and for the walk's supremum `M`, that it is finite
   almost surely, that it has a finite mean, and that that mean scales by `ησ`
   — the two facts the paper's proof takes from Asmussen, proved here by a
   Chernoff bound and Borel–Cantelli.

One further approximation is bracketed rather than assumed: reading the
effective diffusion as constant across the barrier band is discharged by the
two-sided bracket of Lemma 1(v), so the corollary's Arrhenius statement is
sandwiched rather than asserted.

## Convention

`(C0)`, the common reference iterate, is a convention and not a theorem: each
construction's mean update is exactly the gradient of its *own* pullback
landscape (`UpdateDrift.gradient_pullbackLandscape_finiteBatch` for a finite
batch law, unconditionally, and
`UpdateDrift.hasFDerivAt_pullbackLandscape_of_dominated` in general under
domination), and the two landscapes differ. The comparison fixes one reference iterate for both.

Throughout, the paper writes the latent-weight gradient, which already carries
`ψ'`; the corresponding Lean theorems are therefore instantiated at positivity-map
slope one wherever the paper's display has no explicit `σ_s`. The probe
estimator and the trailing-window estimator differ by the factor `T/(T−1)`
(`CrossCovProbe.probeCrossCov_eq_smul_crossCovEstimator`).

## The hypotheses

| Paper | Content | Lean object (`IcnnLift.*`) |
|---|---|---|
| (A1) | Itô surrogate and i.i.d. batches, with the Lipschitz clause (regularity left this label for the notation of §3.2 on 2026-09-15) | `Assumptions.SdeOfSgdModel` (**consumed by no theorem**: `grep -rn "SdeOfSgdModel" *.lean` returns four hits, all inside `Assumptions.lean`, the same standing as `SlackJacobianIdentity` below — it records that the surrogate is assumed and carries no field for the modelling clause), `Assumptions.IidBatchStream` (consumed through the pairwise-independence and identical-distribution binders of the `CrossCov*` theorems) |
| — | Regularity (square-integrable `g₀`, `δθ̃`, `r₂`; symmetric `H`; `σ²_obj > 0`), a sentence of the notation in §3.2 and no longer a clause of (A1) | the `SqIntegrableVec` and `Hᵀ = H` binders of `MinibatchNoise.projDiffusion_lift_eq_frozen_form` and `.projDiffusion_excess_small_jitter`, which are all that Theorem 1(i)–(ii) consume beyond (A2)'s first clause |
| (A2) | Band and barrier: single positivity-map prefactor within tolerance `ε` on the band, and a single barrier with the extremal pair on `[a,b]` | `Assumptions.SinglePrefactor`, `SoftplusBand.singlePrefactor_softplus_relative`; `Assumptions.SingleBarrierPotential` |
| `c > 0` (Thm 1(iii), not a numbered assumption) | The net coupling at the reference iterate along `e_b`, `c = 2 e_bᵀ H Σ_ξ e_b + ϱ(e_b)`, is positive. Read at the **one** direction `e_b` and the one iterate, which is what the proofs consume; the retired (A3) printed the band- and direction-uniform `κ‖e‖² ≤ eᵀ(HΣ_ξ + Σ_ξᵀH)e + ϱ(e)` and was therefore strictly stronger than the Lean binder | `MinibatchNoise.projDiffusion_direct_lt_lift_of_netAlignment` (the ordering under it) and `MinibatchNoise.projDiffusion_lift_ge_direct_add_of_netAlignment` (the quantitative excess `s²κ‖e‖²`), both of which bind `{kappa : ℝ} {e : n → ℝ}` and take `hnet` **at that one `e`**: instantiate `e := e_b`, `κ := c`, and with `‖e_b‖ = 1` the binder `hnet` reads `c ≤ c`. `MinibatchNoise.netAlignment_of_frozenAlignment_and_remainder` (a coupling margin `κ_C` with a remainder channel `τ < κ_C` gives `c ≥ κ_C − τ`) |
| — | Slack Jacobian is the identity (a fact of eq. (2), no longer a labeled hypothesis) | `Assumptions.SlackJacobianIdentity` (consumed by no theorem) |
| — | The chain `g = g₀ + H δθ̃ + r₂` (the definition of `r₂`, no longer a labeled hypothesis) | `MinibatchNoise.probeGrad_eq_frozen_add_hessian_add_remainder`, `MinibatchNoise.refinedChain_of_remainder` |
| — | Common reference iterate (a convention, now a sentence of the notation in §3.2; no longer numbered) | — (see above) |
| Dynkin, `cor:fpt` | the mean first-passage time solves the exit-time boundary-value problem | `KramersExitTime.MeanExitTimeBVP` (carried by every paper-facing exit-time theorem; satisfiable, `meanExitTime_satisfies_bvp`) |
| continuity, `cor:fpt` and `cor:dwell` | the exit time and the occupation time are continuous on the closed `[a,b]` | `htau` / `hD` : `ContinuousOn … (Set.Icc a b)`, bound separately from `hDynkin` in all four paper-facing declarations and consumed by `eq_meanExitTime_of_bvp` as `hTcont`. Printed in the paper as the trailing clause "and continuous up to the closed interval" |
| Dynkin, `cor:dwell` | the mean occupation time of the band solves the boundary-value problem with right-hand side `−1_{[a,w_s]}` | `ShoulderDwell.OccupationTimeBVP` (carried by `mean_dwell_kramers_exponent_of_dynkin`, `mean_dwell_ratio_of_dynkin`, `eventually_mean_dwell_lift_lt_direct_smallLearningRate`, `eventually_mean_dwell_additive_lift_lt_direct`; satisfiable, `ShoulderDwell.dwellTime_satisfies_bvp`, `.dwell_hypotheses_satisfiable`) |
| `rem:adam` quasi-static | the Adam escape exponent at directional variance `V` is `2α/η · (√V + ε)/V` | `PgdBaseline.AdamQuasiStatic` (carried by `adam_exponent_lt_iff`; inhabited by `adamQuasiStatic_satisfiable`) |
| `prop:clamp` noise | `ζ₁, ζ₂, …` independent standard normal | the binders `iIndepFun zeta P` and `∀ i, HasLaw (zeta i) (gaussianReal 0 1) P` of `ProjectionClamp.measure_visitEndsAt` and below (inhabited: `ProjectionClamp.clamp_hypotheses_satisfiable`, from mathlib's `exists_iid`) |
| `prop:clamp` stationary mean | that `E[W_k]` converges, and that `m(ρ)` is finite (the paper's Asmussen citation) | the hypothesis `hm` of `ProjectionClamp.tendsto_integral_clampIterate_homogeneous`; the conclusion is the `ησ` scaling only. Inhabited by `ProjectionClamp.homogeneity_hypotheses_satisfiable` (`m = 1`, instantiated conclusion `ησ`), a deterministic path on a Dirac space — a witness with i.i.d. Gaussian draws would *be* the Asmussen theorem |
| `prop:clamp` clamped fraction | the Spitzer ladder-epoch identity `P(W_∞=0) = exp(−∑ k⁻¹Φ(−ρ√k))`, its monotonicity and its limits | **not formalized**, and not assumed either: no declaration mentions it |

Notes that the paper's remarks record, and their witnesses:

- **(A2)'s tolerance is the substance.** `ε = 0` collapses the band to a point;
  softplus meets the assumption on every interval of half-width `log(1+ρ)` with
  `ε = ρσ_s`, at any depth of the shoulder
  (`SoftplusBand.exists_singlePrefactor_softplus_inside_shoulder`).
- **(A4) is an identity once `r₂` is the discrepancy**, and `ξ` is of order one
  in the jitter rather than a remainder: a quadratic bound on
  `δg − H δθ̃` along the family scaling the latent-weight fluctuation to zero forces
  `ξ ≡ 0` (`MinibatchNoise.noise_eq_zero_of_quadratic_bound`). This is why
  (A4′-q) was retired for (A4′-r).
- **(A4′-r) is satisfiable where (A4′-q) was not**: the remainder clause binds
  the remainder alone, and at small jitter scale no clause bounding the frozen
  coupling by the jitter term can hold, the coupling being first order in that
  scale and the jitter term second
  (`MinibatchNoise.remainderControl_fails_small_jitter`,
  `MinibatchNoise.remainderControl_caps_frozen_coupling`).
- **(A6) is a labeled assumption with empirical content, not a consequence.**
  A two-point batch law realizes `Σ_ξ` of either sign
  (`MinibatchNoise.LocationWitness.coupling_pos` / `.coupling_neg`); the chain
  of (A4) with a remainder bound does not sign it
  (`CouplingSign.coupling_sign_not_automatic_under_remainder_control`,
  `CouplingSign.positiveAlignment_fails_under_remainder_control`); it is
  satisfiable (`CouplingSign.positiveAlignment_satisfiable`) and a
  one-dimensional Gaussian instance meets it with `κ_C = 4`
  (`DiffusionOrdering.updateCov_ordering_strict_nonvacuous`).
  `WitnessInstance.hypothesis_stack_is_inhabited` is stated on the *earlier*
  chain's full `Σ_slack`, not on the refined (A6).

## Paper numbering (v5, 2026-09-15)

Section 3 of the body states two theorems. **Theorem 1 (the coupling channel
and the time on the shoulder)** has three parts: **(i) Channels** collects
Lemma 1(ii)–(iii) verbatim, **(ii) Orders** is Lemma 1(vi) with its two
coefficient *displays* abridged to a description and a by-number pointer (the
de Hoop compression pass of 2026-09-15; every clause of the conclusion,
"for a finite batch law" included, is still printed), and
**(iii) Escape and dwell, the classical law at the two diffusions** collects
`cor:fpt` with its explicit `L(δ)` bracket abbreviated to a pointer *and*
`cor:dwell`, which the body no longer prints as an environment of its own, and
**without their ordering clause**, which the body prints as the paragraph after
the theorem. Its proof, Appendix A.5, is the four pointers and it has no Lean
declaration of its own. Joint necessity is **Theorem 2**, stated in Section 3.4
and proved in Appendix A.2, as the **conditional-independence clause alone**;
the two substitution clauses are printed as a paragraph after it. The projection
results are Theorems 3 and 4 in Appendix A.6.

**The theory trim of 2026-09-15** (no Lean edit; every restructured statement
is an instantiation of a declaration that already existed at the same or a
weaker hypothesis): (A3) was deleted as a numbered assumption and its scalar
named `c` in the notation of Section 3.2, with `c > 0` the antecedent of
Theorem 1(iii) alone; the vacated Theorem 1(ii) now prints the order statement
that had been a trailing paragraph of Step 3 of the proof of Lemma 1, promoted
to Lemma 1(vi); and the shoulder-dwell corollary folded into Theorem 1(iii),
its full statement staying in Appendix A.4 under `cor:dwell`. The body prints
two assumptions, two theorems and one remark.

**The second pass of 2026-09-15** (again no Lean edit; each change makes the
printed hypothesis match the certified one or moves a classical step out of a
numbered statement):

- **(A1) lost its regularity clause to the notation of Section 3.2.** The
  declarations behind Theorem 1(i)–(ii) —
  `MinibatchNoise.projDiffusion_lift_eq_frozen_form` and
  `.projDiffusion_excess_small_jitter` — bind `(hH : Hᵀ = H)`, three
  `SqIntegrableVec` and the chain as a definition, and bind neither
  `SdeOfSgdModel` nor `IidBatchStream`. Theorem 1 now distributes its hypotheses
  per part rather than opening "Let Assumptions 1–2 hold", so parts (i)–(ii) are
  no longer printed under the Itô surrogate they do not use.
- **The ordering clause left Theorem 1(iii) for the paragraph after it.** Its
  declarations, `KramersBridge.eventually_mean_first_passage_lift_lt_direct_smallLearningRate`
  and `ShoulderDwell.eventually_mean_dwell_lift_lt_direct_smallLearningRate`,
  bind `(hsd : 0 < sd) (hsdl : sd < sl)` — two reals, no lift — and return the
  ordering of the two Kramers times. `cor:fpt` and `cor:dwell` keep the clause.
- **Theorem 2 lost its two substitution clauses to a paragraph.** `J_b ≡ 0 ⇒
  J_bᵀΣ̂ = 0` and `δθ̃_k ≡ 0 ⇒ Σ̂ = 0` are substitution into a definition; the
  `StructuralZeros.*` and `CrossCovProbe.probe*_eq_zero_*` declarations are
  unchanged and still carry them.
- **Three disclosures were added to the paper, none of them a Lean claim**: the
  horizon mismatch between the weak approximation of Li et al. (order `1/η`) and
  the exponentially longer times of `cor:fpt`; the one-landscape convention,
  moved from Appendix A.1 into the body as its own sentences; and, in the
  Section 3 opener, that what is machine-checked is the deduction from the
  hypotheses and not that they hold of a trained network.

**The hypotheses were cut from seven labels to three on 2026-09-11**, against
the binders of the declarations that consume them (the author's request:
"using Lean as your guide, can we reduce the list of assumptions"). The map
from the earlier labels:

| earlier | now | why |
|---|---|---|
| (A1) | (A1), plus a regularity clause | the square-integrability of old (A4) and the positivity `σ²_obj > 0` of old (A5) are binders of the identity and of the direct baseline, not hypotheses about training |
| (A2), (A5) | (A2) "Band and barrier" | both describe where the iterates are; the `c_α η σ²_eff ≤ α` clause of old (A5) is a binder of no clause of Theorem 1 (only of `FreeDiffusion.metastable_window`, the crossover remark) and is now a sentence in App. A.1 |
| (A3) | a sentence of notation | consumed by no theorem (`SlackJacobianIdentity`) |
| (A4) | the definition of `r₂` in the notation | `probeGrad_eq_frozen_add_hessian_add_remainder` is an identity once `r₂` is the discrepancy |
| (C0) | a sentence of notation | a convention, not a binder of any declaration; the theorem preambles fix the common iterate directly |
| (A4′-r), (A6) | (A3) "Net positive alignment", itself retired on 2026-09-15 for the scalar `c` and the sign condition `c > 0` | the pair `(κ_C, τ, τ < κ_C)` entered `projDiffusion_direct_lt_lift_of_frozenAlignment` only as `κ_C − τ`; the merged inequality is the weaker hypothesis (`netAlignment_of_frozenAlignment_and_remainder`) and the ordering under it is `projDiffusion_direct_lt_lift_of_netAlignment` |

Both are stated in Section 3.2 and discussed one by one in Appendix A.1,
together with the sign condition `c > 0`; Dynkin's formula stays inside the
statement of Theorem 1(iii), of `cor:fpt` and of `cor:dwell`, as before.

## Theorem 2 (joint necessity), Section 3.4 — proof in Appendix A.2

Stated on the probe estimator at a fixed iterate, for **every** probe count `T`
and **either** divisor (`T−1` or `T`).

Reordered 2026-09-15: the conditional-independence clause is printed first as
(i) and the two substitution clauses last as (ii). The clause families are
independent, so the print order is free and no Lean changed.

**Trimmed the same day (second pass):** the theorem printed in the body is the
conditional-independence clause **alone**, and the two substitution clauses are
printed as the paragraph after it, because each is substitution of a factor of
zero into the definition `eq:cross-cov` rather than a theorem. Appendix A.2
proves all four. The rows marked (ii) below are unchanged and still carry the
two substitutions; nothing left the development.

| Clause | Lean |
|---|---|
| (i) conditional independence ⇒ `Σ_ξ = 0` a.e. | `CrossCovIndep.popCrossCov_centered_eq_zero_of_indepFun`, `CrossCovIndep.condPopCrossCov_condExp_centered_ae_eq_zero` |
| (i) unbiasedness, either divisor | `CrossCovProbe.probeCrossCovWith_unbiased_of_indepFun`, `CrossCovProbe.probeCrossCov_centred_unbiased_of_iid`, `CrossCovProbe.probeCrossCovMean_centred_expectation_of_iid` |
| (i) `O(1/T)` mean-square error | `CrossCovProbe.probeCrossCov_entry_l2_rate`, `CrossCovSLLN.crossCovEstimator_raw_windowCentred_l2_bound`, `CrossCovSLLN.integral_sq_windowMean_mul` |
| (i) a.s. convergence to zero | `CrossCovProbe.probeCrossCov_centred_tendsto_zero_ae`, `CrossCovSLLN.crossCovEstimator_raw_tendsto_zero_ae` |
| (ii) no learnable slack ⇒ bias-channel reading ≡ 0 | `CrossCovProbe.probeSlackReading_eq_zero_of_jacobian_eq_zero`, `CrossCov.slackReading_eq_zero_of_jacobian_eq_zero` |
| (ii) body constant in the batch ⇒ estimator ≡ 0 | `CrossCovProbe.probeBodyFluct_eq_zero_of_const`, `CrossCovProbe.probeCrossCovWith_eq_zero_of_body_const`, `CrossCov.crossCovEstimator_eq_zero_of_body_const` |
| combined contrapositive | `CrossCovProbe.probe_joint_necessity_finite` |
| sample-cross-covariance identity | `CrossCovProbe.probe_sample_crossCov_identity`, `CrossCovSLLN.sample_crossCov_identity` |
| Bessel factor `T/(T−1)` | `CrossCovProbe.probeCrossCov_eq_smul_crossCovEstimator` |

**Which independence is used where.** Unbiasedness and the almost-sure limit
need the pairwise form; the mean-square rate needs the independence of the two
sample means (`CrossCovSLLN.integral_sq_windowMean_mul`). (A1) states i.i.d.
(mutual) and the appendix says which step consumes what. Non-degeneracy of the
conditional-independence clause — printed third before 2026-09-15 — is listed
below.

**Necessity, not sufficiency — the converse fails in both halves.**
A full lift can read zero
(`StructuralZeros.exists_fullLift_slackReading_eq_zero`) and a body-only model
can read nonzero without slack
(`StructuralZeros.exists_bodyOnly_estimator_ne_zero_of_no_slack`). Nothing in
the paper claims an unproven converse. Architectural zeros:
`StructuralZeros.crossCovEstimator_directSoftplus_eq_zero`,
`crossCovEstimator_anchored_eq_zero`, `crossCovEstimator_biasOnly_eq_zero`,
`crossCovEstimator_splitting_eq_zero`, `slackReading_bodyOnly_eq_zero`,
`mechanism_falls_silent_of_frozen_anchor`. Non-degeneracy of case (iii):
`CrossCovSLLN.exists_nondegenerate_case_iii_model`,
`CrossCovIndep.exists_centered_popCrossCov_ne_zero`,
`CrossCovProbe.probeCrossCov_centred_ne_zero`.

## Lemma 1 (diffusion and barrier), Appendix A.3 — parts (ii)–(iii) are Theorem 1(i), part (vi) is Theorem 1(ii)

The appendix proof runs in five steps; the body carries no sketch.

| Step / clause | Lean |
|---|---|
| Step 1, three-term split — (i) `Σ_slack = Σ_ξ + V H + Cov[δθ̃, r₂]` | `MinibatchNoise.covMat_slack_refined`, `MinibatchNoise.oldChain_remainder_eq_noise_add_refined` |
| Step 2, update covariance exactly — (ii) `eq:sigma-eff` | `MinibatchNoise.projDiffusion_lift_eq_frozen_form`, `UpdateCovariance.projDiffusion_decomposition_with_remainder_symmetric`, `UpdateCovariance.covMat_add_three_self` |
| Step 2, direct baseline — (iii) first clause | `UpdateCovariance.projDiffusion_direct_baseline` |
| Step 3, sign of the excess — (iii) `σ²_eff − σ_s²σ²_obj ≥ c`, unconditionally, strict exactly where `c > 0` | `MinibatchNoise.projDiffusion_lift_ge_direct_add_of_netAlignment` and `.projDiffusion_direct_lt_lift_of_netAlignment` at `e := e_b`, `κ := c`; `TraceLemmas.quadForm_conj_nonneg` (the jitter term `≥ 0`); `MinibatchNoise.projDiffusion_direct_lt_lift_of_frozenAlignment` (the earlier `κ_C − τ` form) |
| Step 3, the orders — **(vi)**, the exact `t`-polynomial, its linear coefficient `2 e_bᵀ H Cov[u,g₀] e_b + 2 e_bᵀ Cov[g₀,w₁] e_b`, and the jitter term `e_bᵀ H Cov[u] H e_b` in its quadratic coefficient | `MinibatchNoise.projDiffusion_excess_small_jitter` (the polynomial, exactly, with `crossTerm H Ξ e = eᵀ(HΞ + ΞᵀH)e = 2 eᵀHΞe` at symmetric `H`); `.projDiffusion_direct_lt_lift_small_jitter` (positive linear coefficient ⇒ the ordering at every small `t`); `.projDiffusion_lift_lt_direct_small_jitter` (negative ⇒ reversal); `.projDiffusion_excess_small_jitter_of_no_frozen_coupling` (the excess starts at `t²`) |
| (vi)'s independent-latent-weight sentence needs **two** zeros | `.projDiffusion_excess_small_jitter_of_no_frozen_coupling` binds `hΞ : Cov[u,g₀] = 0` **and** `hw : Cov[g₀,w₁] = 0`. The first is independence (`MinibatchNoise.covMat_eq_zero_of_indepFun`); the second is discharged only in the finite-batch model (`MinibatchNoise.outerMoment_prod_hessianFluct_eq_zero`, which binds `[Fintype Ω₁] [Fintype Ω₂]`). This is why the paper's part (ii) says "and, for a finite batch law, no first-order remainder pairing either" — writing it as one condition would be an overclaim |
| the net-coupling discussion's subtraction: the coupling form of `Σ_ξ` is that of `Σ_slack` less `2 eᵀ H V H e`, hence negative in the same direction | `CouplingSign.crossTerm_frozen_eq_slack_sub`, `.crossTerm_frozen_neg_of_slack_neg`, `.crossTerm_frozen_le_slack` |
| Step 4, stationary density — (iv) | `StationaryDensity.stationaryDensity_zeroFlux`, `.eq_stationaryDensity_of_zeroFlux`, `.boltzmann_zeroFlux`, `.hasDerivAt_effectivePotential` |
| Step 5, barrier bracket — (v) | `BarrierRescaling.quasipotential_gap_bounds`, `.quasipotential_gap_of_const_diffusion`, `QuasipotentialFTC.quasipotential_gap_bounds_fully_discharged` |

Supporting: log-density curvature and the `σ^{-2}_eff` prefactor
(`LogDensityCurvature.*`); non-degeneracy of the ordering
(`DiffusionOrdering.updateCov_ordering_strict_nonvacuous`,
`.magnitude_channel_ordering`); the directional reduction is a **lower** bound
off eigendirections (`FWReduction.fwActionScalar_le_fwActionMulti_alongLine`).

**The anti-overclaim is itself a theorem.** The lift does not add curvature to
the averaged loss: differentiation commutes with the batch average, so the
pullback Hessian has no cross-covariance term
(`PullbackHessian.hessianMatrix_pullbackLoss`,
`.not_hessianCrossCovDecomposition_of_trace_pos`,
`.symmPart_eq_zero_of_hessianCrossCovDecomposition`,
`.pullbackHessian_not_trace_lower_bound`), and each mean update is exactly the
gradient of its own pullback landscape
(`UpdateDrift.hasFDerivAt_pullbackLandscape_of_dominated`,
`.gradient_pullbackLandscape_finiteBatch`). "Implicit strong convexification"
and "landscape smoothing" are retired as mechanism names.

**The independent-latent-weight cell.** `Σ_ξ = 0` leaves the jitter term standing and
the excess begins at second order in the perturbation scale
(`MinibatchNoise.covMat_eq_zero_of_indepFun`,
`.projDiffusion_excess_small_jitter_of_no_frozen_coupling`, whose second
hypothesis `Cov[g₀,w₁] = 0` is discharged for such a perturbation only in the
finite-batch model, by `outerMoment_prod_hessianFluct_eq_zero`), against first
order at a positive linear coefficient — a positive net coupling is necessary
for the effect and not sufficient. This is the content the paper prints as
Theorem 1(ii) / Lemma 1(vi) since 2026-09-15.

**The trace is the wrong monitor for direction.** By (i) the reading contains
`tr(V H) ≥ 0`, so a positive trace does not sign `Σ_ξ`, and for `d > 1` a
positive trace is compatible with a negative direction of the coupling form
(`CouplingSign.counterexample_trace_slack_pos`, `.ce_crossTerm_neg`).

## `cor:fpt` (escape time), Appendix A.4 — the exit-time half of Theorem 1(iii)

| Clause | Lean |
|---|---|
| BVP and its unique solution | `KramersExitTime.MeanExitTimeBVP`, `.meanExitTime_satisfies_bvp`, `.eq_meanExitTime_of_bvp` |
| two-sided Arrhenius bracket | `KramersExitTime.meanExitTime_arrhenius_lower_bound`, `.meanExitTime_upper_bound`, `.mean_first_passage_arrhenius_of_dynkin` |
| exponent limit `ησ²_eff log τ → 2α` | `KramersExitTime.tendsto_mul_log_meanExitTime`, `.mean_first_passage_kramers_exponent_of_dynkin` |
| exponent strictly decreasing in `σ²_eff` | `BarrierRescaling.arrheniusExponent_strictAntiOn`, `.arrheniusExponent_lift_lt_direct` |
| lift's mean first-passage smaller for small `η` | `KramersBridge.eventually_mean_first_passage_lift_lt_direct_smallLearningRate`, `.meanExitTime_ratio_bracket`, `.kramers_hypothesis_sandwiched` |
| crossover past `ησ²_eff = 2α` | `FreeDiffusion.metastable_window`, `.isFreeExitTime_isTheta_inv`, `.arrheniusTime_not_isBigO_inv` |

No prefactor is claimed on either side; (A5) consumes continuity and the
extremal pair on the escape interval `[a,b]`, and the paper says so.

## Shoulder dwell (`cor:dwell`), Appendix A.4 — the dwell half of Theorem 1(iii)

Added 2026-09-13, in `ShoulderDwell.lean`. **Stated in Appendix A.4 only since
2026-09-15**: the body absorbed it into Theorem 1(iii), which prints its dwell
limit, its ratio limit and its subexponential-fraction clause, omits the
hypothesis pointer and the Dynkin qualifier "at the band indicator", and since
the second pass of that day carries its ordering clause in the paragraph after
the theorem rather than inside the statement. The corollary is proved the way the
appendix proves it: the exit-time development is **generalized** from the
constant right-hand side to a general one and then instantiated at the band
indicator `1_{[a,w_s]}`. Nothing in `KramersExitTime` is weakened — the
exit-time statements are the case `f ≡ 1` of the new ones, in both directions.

| Clause | Lean |
|---|---|
| the BVP `½σ²D″ − L̃′D′ = −1_{[a,w_s]}`, `D(b)=0`, `D′(a⁺)=0` | `ShoulderDwell.OccupationTimeBVP`, `.occupationTime_satisfies_bvp`, `.dwellTime_satisfies_bvp` |
| its unique solution | `ShoulderDwell.eq_of_occupationTimeBVP` (through `.constant_of_hasDerivAt_zero_off_point`) |
| the closed form `eq:pf-dwell` | `ShoulderDwell.occupationTime`, `.dwellTime` |
| `D(x) ≤ T(x)` (the upper bound, by monotonicity) | `ShoulderDwell.occupationTime_le_meanExitTime`, `.dwellTime_le_meanExitTime` |
| the windowed lower bound, inner window inside the band | `ShoulderDwell.occupationTime_window_lower_bound`, `.occupationTime_arrhenius_lower_bound_uniform` |
| `ησ²_eff log D(x) → 2α` | `ShoulderDwell.tendsto_mul_log_dwellTime`, `.mean_dwell_kramers_exponent_of_dynkin` |
| `ησ²_eff log (T(x)/D(x)) → 0` | `ShoulderDwell.tendsto_mul_log_meanExitTime_div_dwellTime`, `.mean_dwell_ratio_of_dynkin` |
| the dwell is the shorter under the lift for every sufficiently small **step size** — one shared `η → 0⁺`, two fixed variances `sd < sl`, the multiplicatively coupled pair of `thm:main`(iii) and Step 3 of `app:proof-fpt` | `ShoulderDwell.eventually_dwellTime_lift_lt_direct_smallLearningRate`, `.eventually_mean_dwell_lift_lt_direct_smallLearningRate`, `.eventually_occupationTime_lift_lt_direct_smallLearningRate`; witness `.cubicBarrier_dwell_lift_lt_direct_smallLearningRate` |
| *a companion, and not that clause*: the same ordering on the **additively** coupled family `σ²` against `σ² + σ_Jac²`, where the small parameter is the shared variance and the gap does not shrink with it | `ShoulderDwell.eventually_dwellTime_additive_lift_lt_direct`, `.eventually_mean_dwell_additive_lift_lt_direct` |
| the exit-time results re-derived as `f ≡ 1` | `ShoulderDwell.occupationTimeBVP_one_iff`, `.occupationTime_one`, `.meanExitTime_satisfies_bvp_of_occupation`, `.eq_meanExitTime_of_occupationTimeBVP` |
| non-vacuity on the cubic barrier, band `[−3/2, 1]` | `ShoulderDwell.dwell_hypotheses_satisfiable`, `.cubicBarrier_dwell_kramers_exponent` (limit `8/3 = 2α`), `.cubicBarrier_dwell_ratio` |

**The one place where the Lean statement is not the LaTeX one.** The appendix
says the two differentiations and the uniqueness argument "apply verbatim to a
bounded measurable right-hand side". The second differentiation does not, at
the jump: `D″(w_s)` does not exist, the closed form having a corner in `D′`
there (`ShoulderDwell.not_continuousAt_bandIndicator`). `OccupationTimeBVP`
therefore requires the interior equation only at the points where the
right-hand side is continuous — every point of `(a,b)` but `w_s`. That is the
form the closed form satisfies, so the Dynkin hypothesis is not vacuous, and
uniqueness still holds across the jump, so it still pins its solution. Read the
appendix sentence as the classical abuse it is; the Lean is the exact version.

**Two couplings, and which one is the corollary's clause.** "For every sufficiently small step
size" is one shared learning rate going to zero at two *fixed* effective variances — the pair
`(η·sd, η·sl)` with `sd < sl`, multiplicatively coupled, since the diffusion coefficient of the
boundary-value problem is `η σ²_eff`. That is
`eventually_dwellTime_lift_lt_direct_smallLearningRate`, added 2026-09-13 and mirroring
`KramersBridge.eventually_meanExitTime_lift_lt_direct_smallLearningRate`, which draws the same
distinction for the exit time. The additive theorem is a **different** statement — the small
parameter is itself the direct arm's diffusion, the lift's is that plus a fixed `σ_Jac² > 0`, and
the gap between them does not shrink — so it does not discharge the corollary's clause and
neither implies the other. It is kept as the companion for the variance-offset family of
`eq:sigma-eff`. (An earlier version of this file said the additive theorem gave "the paper's own
pair"; that was wrong, and is corrected here.)

**What is not claimed.** `D(x)/T(x) → 1` is nowhere stated (it needs
`inf_{(w_s,b]} L̃ > L̃(w_b)`, which (A2) does not supply, exactly as
`rem:dwell-band` says), and the general-band exponent of `rem:dwell-band` is
not formalized. The band containment that remark names as the extra condition
is the `one_le_on_band` clause of `ShoulderDwell.BandRHS`.

## One projected coordinate at the boundary (`prop:clamp`), Section 3.3 — proof in Appendix A.6

Added 2026-09-13, in `ProjectionClamp.lean`. **Two of the proposition's three
clauses are formalized, and the third is not.** The file's docstring says which
is which, and so does this table; the paper's machine-checking claim must be
read at this scope.

| Clause of `prop:clamp` | Lean |
|---|---|
| a visit ends at the first step whose gradient points into the cone (the paper's words; the inequality proved is `γ + σζ_k < 0`, i.e. the *update* points into the cone — correspondence report M7) | `ProjectionClamp.clampStep_pos_iff`, `.clampStep_eq_zero_iff`, `.VisitEndsAt`, `.clampRun`, `.clampRun_eq_of_visitEndsAt`, `.visitEndsAt_clampRun` |
| the visit is a visit **of the recursion**: `W` stays at the boundary through it and is released at its end | `ProjectionClamp.clampIterate_eq_zero_of_forall_le`, `.clampIterate_eq_zero_of_visitEndsAt`, `.clampIterate_pos_of_visitEndsAt` (added 2026-09-13). That the recursion *returns* to the boundary is **not** proved — it belongs to the stationary-law clause |
| it lasts a geometric number of steps | `ProjectionClamp.measure_visitEndsAt`, `.measureReal_visitEndsAt`, `.measureReal_visitEndsAt_eq_geometricMeasure` (against mathlib's own `geometricMeasure`), `.measureReal_clampRun_eq` |
| of mean `1/Φ(−ρ)` | `ProjectionClamp.tsum_visitLength`; `Φ` is `ProbabilityTheory.cdf (gaussianReal 0 1)` (`.clampExitProb`, `.clampExitProb_pos`, `.clampExitProb_lt_one`) |
| the recursion in the paper's letters, and the reduction to `ρ` | `ProjectionClamp.clampIterate`, `.clampIterate_succ_paper`, `.eta_mul_add_eq` |
| the clamped fraction is a function of `ρ` alone | `ProjectionClamp.clampIterate_eq_zero_iff_unit`, `.measureReal_clampIterate_eq_zero_scale_free` — at **every step**: the clamped event is literally the same event at every scale `ησ` |
| it increases in `ρ` | `ProjectionClamp.clampIterate_antitone_in_rho`, `.clampIterate_eq_zero_of_le_rho`, `.measureReal_clampIterate_eq_zero_mono_rho` — at every step, by a pathwise coupling on one and the same draws. **Non-decreasing**, not strictly increasing |
| it tends to `1` as `ρ → ∞` | `ProjectionClamp.measure_forall_le`, `.measureReal_clampIterate_eq_zero_ge`, `.tendsto_measureReal_clampIterate_eq_zero_atTop` — at every step, from the bound `(1 − Φ(−ρ))^k` |
| it tends to `0` as `ρ ↓ 0` | **not formalized**: false at any finite step, so it is a statement about the `k → ∞` limit alone, and its classical proofs are the Spitzer series or recurrence of the driftless walk. Mathlib v4.31.0 has neither (no `recurren*` in `Probability/`, no Spitzer/Sparre-Andersen/Wiener--Hopf anywhere) |
| `W_k = max_{0≤j≤k} S_j` in distribution (the reversal identity) | `ProjectionClamp.clampIterate_eq_sup'_sub` (the recursion unrolled, pathwise), `.clampIterate_eq_maxWalk_revDraws` (reversing the block, pathwise), `.map_revDraws_eq` (an i.i.d. block has the same law reversed), `.map_clampIterate_eq_map_maxWalk` (**the identity**) |
| `M = sup_j S_j` is finite a.s. (the paper's Asmussen citation) | `ProjectionClamp.hasSubgaussianMGF_of_hasLaw`, `.hasSubgaussianMGF_negSum`, `.measureReal_walk_pos_le`, `.ae_bddAbove_walkSum` — **proved** by Chernoff (`P(S_j>0) ≤ exp(−ρ²j/2)`) plus Borel–Cantelli; no strong law, no fluctuation theory |
| `M` has a finite mean (the paper's Asmussen citation) | `ProjectionClamp.le_inv_mul_exp`, `.lintegral_ofReal_walkSum_le`, `.lintegral_iSup_ofReal_walkSum_lt_top` — **proved**: `M ≤ ∑_j (S_j)^+` and the tilted bound makes that a geometric series |
| `W_k → M` in distribution | `ProjectionClamp.maxWalk_mono`, `.tendsto_maxWalk_atTop`, `.tendsto_integral_clampIterate` — **proved**, in the defining form (integrals against bounded continuous test functions converge) |
| `E[W_∞] = ησ·m(ρ)` with `m` finite and free of `η, σ` | `ProjectionClamp.lintegral_iSup_ofReal_walkSum_homogeneous` with the rows above — **proved for `M`**, which the two rows above tie to the iterates. Not proved: that the law of `M` is *stationary* for the recursion, or unique as such — nothing states invariance |
| the paper's `M = ησ·sup_j(−∑_{i≤j}(ρ+ζ_i))` and its scaling | `ProjectionClamp.walkSum`, `.iSup_walkSum_homogeneous`. **Correction (2026-09-13):** `iSup_clampIterate_homogeneous` scales the supremum of the *iterates*, which is a different pathwise object from the paper's `M` (a supremum of the walk's partial sums); the two agree only in distribution at a fixed time, and the docstring now says so |
| `E[W_∞] = ησ·m(ρ)`, the homogeneity | `ProjectionClamp.clampIterate_homogeneous`, `.integral_clampIterate_homogeneous`, `.tendsto_integral_clampIterate_homogeneous`, `.iSup_clampIterate_homogeneous`; the hypothesis of the third is inhabited by `.homogeneity_hypotheses_satisfiable` |
| the hypothesis stack is inhabited | `ProjectionClamp.clamp_hypotheses_satisfiable` (mathlib's `exists_iid`) |
| **what remains unformalized, in two parts**: the Spitzer series `exp(−∑ k⁻¹Φ(−ρ√k))` and the limit `0` as `ρ ↓ 0` that it supplies | **not formalized.** The series needs fluctuation theory mathlib does not carry. The `ρ ↓ 0` limit is false at any finite step, so it lives at the limit object; its non-Spitzer route (Kolmogorov zero-one at zero drift, then continuity in `ρ`) is supported in principle by mathlib but needs tail-σ-algebra scaffolding and a parametric continuity argument. Also not formalized, and not part of the paper's load-bearing use: the *stationarity* and uniqueness of the limit law, and the convergence of `P(W_k = 0)` itself (weak convergence does not give it — the limit law has an atom at `0`) |

Conventions to check against the paper: `VisitEndsAt ρ z n` means the visit
lasts `n + 1` steps — `n` clamped steps and the releasing one — which is why the
mean of `n + 1` is `1/Φ(−ρ)` and not `(1−Φ(−ρ))/Φ(−ρ)`; the draws are indexed
from `0`; `η` and `σ` enter the recursion only through `c = ησ`, with
`clampIterate_succ_paper` the machine-checked bridge to the paper's display; and
the mean visit length is a convergent series over the visit-length law, not an
integral of an `ℕ`-valued random variable. `γ > 0` is never used by the two
formalized clauses — negative drift is what the unformalized third one needs.

## The power of the positivity-map slope — the correction of 2026-09-03

**The earlier claim was wrong and is retired.** The additive form of `PowerOfS`
gave the lift an exponent of order `σ_s^{-1}` against direct softplus's
`σ_s^{-2}`. That reasoning does not apply to `eq:sigma-eff`, and
`PowerOfS.liftExponent_isTheta_inv` / `.exponentRatio_tendsto_atTop` must **not**
be cited for it. `CouplingPower` settles the question:

- By the chain rule (proved, not stipulated) the mean latent-weight Hessian is
  `H = diag(ψ') H_θ diag(ψ') + diag(ψ'' ⊙ ḡ)`
  (`hasFDerivAt_preReadoutGrad`, `fderiv_preReadoutGrad_apply`), and the frozen
  coupling `Σ_ξ = Ξ_θ diag(ψ')` is attenuated once (`covMat_hadamard_right`).
- Under (A2) read exactly the coupling term is `2σ_s³a + 2σ_s c₂ b`
  (`couplingTerm_single_prefactor`); for softplus `ψ'' = ψ'(1−ψ')`, giving
  `2σ_s³a + 2σ_s²(1−σ_s)b` (`couplingTerm_softplus`; exponential positivity map
  `couplingTerm_exp`).
- Hence the coupling term is `O(σ_s²)` always and **never** `Θ(σ_s)`
  (`couplingPoly_isBigO_sq`, `couplingPoly_not_isTheta_id`);
  `Θ(σ_s²)` when `b ≠ 0`, `Θ(σ_s³)` when `b = 0 ≠ a`
  (`couplingPoly_isTheta_sq_of_ne`, `couplingPoly_isTheta_cube_of_eq`).
  The same holds under (A2) at relative tolerance
  (`couplingTerm_isBigO_sq_of_relative`, `couplingTerm_not_isTheta_id_of_relative`).
- Assembled: `σ²_eff(σ_s) = σ_s²(σ²_obj + 2b + j₀) + σ_s³R + ρ`
  (`projDiffusion_lift_softplus_form`, `remainderScalars_once_attenuated`), so
  when `τ` falls below the leading coefficient, `σ²_eff = Θ(σ_s²)`
  (`sigmaEffPoly_isTheta_sq`) and **both** exponents are `Θ(σ_s^{-2})`
  (`liftExponent_sigmaEff_isTheta_inv_sq`, `directExponent_isTheta_inv_sq'`).
- The lift lowers the **constant**, not the order: the exponent ratio is
  eventually bounded, does not diverge, and converges to
  `(σ²_obj + 2ḡ_bΞ_bb + ḡ_b²V_bb + ϱ₀)/σ²_obj`
  (`exponentRatio_sigmaEff_eventually_bounded`,
  `exponentRatio_sigmaEff_not_tendsto_atTop`, `exponentRatio_sigmaEff_tendsto`).
  That leading coefficient is itself a variance,
  `Var[e_bᵀξ_θ + (ḡ ⊙ e_b)ᵀδθ̃]` (`leadingCoeff_eq_projDiffusion`,
  `leadingCoeff_nonneg`, `leadingCoeff_ge_direct_of_gradCoupling_nonneg`).

The paper states exactly this and nothing stronger. **The Adam recursion is not
formalized, and the reduction it licenses now is**: the paper states the
square-root reduction and does not derive it, and with it both exponents pass to
order `σ_s^{-1}`; there is no order separation to halve. What is machine-checked
since 2026-09-13 is the ordering consequence — the map `V ↦ (√V + ε)/V` is
strictly decreasing (`PgdBaseline.adamVarianceMap_strictAntiOn`), so under the
named hypothesis `PgdBaseline.AdamQuasiStatic` an inequality between two
exponents is equivalent to the reverse inequality between the two variances
(`PgdBaseline.adam_exponent_lt_iff`). The quasi-static reading itself stays a
modelling input.
The barrier convention (reading `α` in latent-weight units) rescales both
exponents alike, so the order equality and the bounded ratio are
convention-independent.

## Section 4.3 (projection) — statements collected in Appendix A.6

| Paper | Lean |
|---|---|
| Theorem 3, directional escape comparison (biconditional) | `PgdBaseline.arrheniusExponent_lift_lt_pgd_iff`, `.arrheniusExponent_lift_lt_pgd_iff_gate` |
| direct softplus strictly below projection when the positivity map attenuates | `PgdBaseline.directDiffusionAlong_lt_pgdDiffusionAlong`, `.directDiffusionAlong_le_liftDiffusionAlong` |
| no magnitude comparison settles lift vs. projection | `PgdBaseline.exists_liftDiffusionAlong_lt_pgdDiffusionAlong`, `.not_posSemidef_liftDiffusionMatrix_sub_pgdDiffusionMatrix` |
| Theorem 4, projection has no coupling term | `PgdBaseline.pgd_alignment_structurally_unavailable`, `StructuralZeros.exists_batchCoupled_sigmaJacSq_eq_zero` |
| the gate is sufficient, not necessary | `PgdBaseline.gateHolds_of_couplingOnly_gate`, `.exists_gateHolds_of_couplingOnly_gate_failure` |
| gate witnesses | `PgdBaseline.gateWitness_arrheniusExponent_lift_lt_pgd`, `.gateWitness_trace_lt`, `.gate_discriminates_on_slope` |
| lift's exit time below projection's under the gate | `PgdBaseline.eventually_mean_first_passage_lift_lt_pgd_of_gate`, `.meanExitTime_lift_lt_pgd_of_window` |
| the threshold read net of the remainder channel (`app:proof-pgd`'s opening) | `PgdBaseline.liftDiffusionAlong_add_remainder`, `.arrheniusExponent_lift_lt_pgd_iff_of_remainder` |
| `rem:adam`: the variance map is strictly decreasing and common, so the exponent ordering is the reverse variance ordering | `PgdBaseline.adamVarianceMap_strictAntiOn`, `.adam_exponent_lt_iff`; the quasi-static reading is the hypothesis `PgdBaseline.AdamQuasiStatic` (`.adamQuasiStatic_satisfiable`) |

The Lean biconditional is stated at zero remainder, and the with-remainder
version the paper uses is now proved as well
(`PgdBaseline.arrheniusExponent_lift_lt_pgd_iff_of_remainder`, added
2026-09-13): the remainder channel enters the lift's directional diffusion
additively, so the threshold is the same arithmetic with `vᵀ R v` added to the
excess, and `R = 0` returns the theorem as `thm:directional` states it.
The slope-below-one clause of Theorem 4 is one line of arithmetic on the Lean
statement at slope one, written out in the appendix. **The paper reports no
value for the gate ratio and does not claim the gate is met at our operating
point.** The `D = 1` PGD/lift tie is not claimed anywhere.

## Places where the paper is deliberately weaker than the Lean

Each is stated in the appendix; weaker in the paper is allowed, stronger is not.

- The `O(T^{-1/2})` rate is given in mean square plus Chebyshev; the Lean has
  the exact `L²` rate and the explicit sample-centered bound.
- `cor:fpt` states the two-sided bracket, the exponent limit and the small-`η`
  ordering; the Lean also has the window-explicit ordering and the additively
  coupled family.
- **Theorem 1(iii) states less than `cor:fpt` and `cor:dwell` do** (2026-09-15,
  second pass): the two exponent limits and the ratio limit, and not the
  ordering, which the body carries as prose with `c > 0` named in the open. The
  body is therefore weaker than the appendix, which is the allowed direction.
- **Theorem 2 states less than Appendix A.2 proves**: the conditional-
  independence clause alone, the two substitutions being printed as prose.
- (A5) consumes continuity, although the structure records `C²`.
- The directional reduction is a lower bound off eigendirections.
- Theorem 3 is stated at zero remainder in the paper; the Lean also carries the
  remainder-channel form (`arrheniusExponent_lift_lt_pgd_iff_of_remainder`).
- `cor:dwell` states the two limits and the small-step ordering; the Lean also
  has the windowed lower bound with explicit constants, the general
  right-hand-side forms of all three clauses (`ShoulderDwell.BandRHS`),
  `D(x) ≤ T(x)` at every diffusion, and the ordering in a second coupling (the
  additive family) that the corollary does not state.
- `prop:clamp`'s dwell is stated for the paper's Gaussian draws; the Lean proves
  the visit-length law from mutual independence and the standard-normal law
  alone, with no use of `γ > 0`, and identifies it with mathlib's own geometric
  distribution.

## Steps that are prose in the appendix, not in the development

The Itô surrogate and Dynkin's formula, for the exit time and for the
occupation time alike (above); the Spitzer ladder-epoch identity and the
existence of the projected recursion's stationary law, with the finiteness of
`m(ρ)` (`prop:clamp`, above); per-probe to collection-level
independence; the slope-below-one clause of Theorem 4; the clipping and
face-commitment propositions, which are numerical. Each is marked as such
where it is used.

**Three former entries of this list were discharged on 2026-09-13**, on the
instruction to close what was elementary and leave only what mathlib cannot
support. The `2eᵀHVHe` subtraction at the (A6) counterexample is
`CouplingSign.crossTerm_frozen_eq_slack_sub` with the sign transfer
`.crossTerm_frozen_neg_of_slack_neg`. Theorem A net of the remainder channel is
`PgdBaseline.arrheniusExponent_lift_lt_pgd_iff_of_remainder`, with the additive
identity `.liftDiffusionAlong_add_remainder`. The Adam reduction is split:
`PgdBaseline.adamVarianceMap_strictAntiOn` and `.adam_exponent_lt_iff` are
theorems, and the quasi-static reading that licenses the substitution is the
named hypothesis `PgdBaseline.AdamQuasiStatic`, carried in the register of the
Dynkin inputs and inhabited by `.adamQuasiStatic_satisfiable`.

## Modules

33 registered. Cited by the appendix: `Assumptions`, `BarrierRescaling`,
`CouplingPower`, `CouplingSign`, `CrossCov`, `CrossCovIndep`, `CrossCovProbe`,
`CrossCovSLLN`, `DiffusionOrdering`, `FreeDiffusion`, `FWReduction`,
`KramersBridge`, `KramersExitTime`, `LogDensityCurvature`, `MinibatchNoise`,
`PgdBaseline`, `PowerOfS`, `PullbackHessian`, `QuasipotentialFTC`,
`ShoulderAttenuation`, `SoftplusBand`, `StationaryDensity`, `StructuralZeros`,
`ProjectionClamp`, `ShoulderDwell`,
`TaylorChain`, `TraceLemmas`, `UpdateCovariance`, `UpdateDrift`,
`WitnessInstance`. Present but not cited: `BandOccupancy`,
`QuasipotentialMonotone`, `InversePSDAntitone`.
