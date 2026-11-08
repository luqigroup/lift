# `formal/` — the Lean 4 development behind Section 3 of the ICLR paper

This directory is the machine-checked counterpart of the paper's theory section.
It exists so that every theorem, lemma, corollary and displayed identity that
Section 3 and Appendix A state can be traced to a kernel-verified declaration.

**The paper itself contains no Lean surface.** The manuscript is English and
mathematics only: no module names, no theorem names, no `\lean{}` pointers,
not in the body and not in the appendix.
The proofs in Appendix A follow the setup of the Lean proofs step for step —
same decomposition, same intermediate objects, same named hypotheses — but they
are written as ordinary mathematical prose. The formalization is acknowledged
in one sentence at the head of Section 3, and the reproducibility statement
lists the machine-checked development among the contents of the submitted
archive. **This file is where the statement-by-statement correspondence lives.**

## Status

The stationary law of `prop:clamp` and Spitzer's ladder-epoch identity are
proved in five modules (`SpitzerCyclic`, `SpitzerIdentity`, `SpitzerGaussian`,
`LindleyStationary`, `LindleyClamped`), so that the appendix relies on no
unproved random-walk fact; its citations of Spitzer (1956) and Asmussen (2003)
are attributions.

- `lake build` runs green over all 38 modules (8633 jobs, no errors, no
  warnings).
- No `sorry`, no `admit`, no `native_decide`, no `axiom` anywhere in the tree
  (strict sweep with block comments, docstrings and line comments stripped,
  over all 38 files).
- **Exhaustive axiom sweep: 3078 declarations, of which 2670 are theorems,
  every one depending on no axiom beyond `propext`, `Classical.choice`, `Quot.sound`.
  Zero non-standard axioms, zero `sorryAx`, zero `axiom` declarations of our
  own.** The sweep
  enumerates the environment rather than parsing source, so it covers
  sub-namespace witnesses and wrapped declaration headers alike.
- Reproduce the sweep with `./check.sh AxiomAudit.lean`. `AxiomAudit.lean`
  imports all 38 modules, walks every constant whose defining module is one of
  ours, and runs the kernel's own axiom collection on each; it prints six
  counts (`AUDITED_DECLARATIONS`, `OF_WHICH_THEOREMS`, `DECLARED_AXIOMS_OF_OURS`,
  `INTERNAL_OF_THOSE`, `NONSTANDARD_AXIOM_HITS`, `SORRYAX_HITS`) and the two
  name lists behind the last two (`NONSTANDARD_NAMES`, `SORRY_NAMES`). It is deliberately **not** in `lakefile.toml`'s
  `defaultTargets`, so `lake build` does not pay for it.
- `./check.sh <File>.lean` type-checks one file without taking the build lock.
  `./axioms.sh <File>.lean` remains as a per-file spot check; it parses source
  and cannot resolve names that wrap over three lines or sub-namespace
  witnesses it cannot reach by `open IcnnLift`, which is why `AxiomAudit.lean`
  is the audit of record.

## What is assumed rather than proved

The modeling steps below enter the development as explicit hypotheses (or, for
item 6, a definition) rather than as theorems. Items 1–3 are the three modeling
inputs that the opening of Appendix A names; items 4a, 4b, 6 and 7 are rows of
`tab:hyp`. Items 4 and 5 are history: neither is an input of the current paper.

1. **The Itô surrogate of (A1).** That the SGD recursion may be read as an Itô
   process is the register of the stochastic modified equations of Li et al.;
   it is carried by the structure `SdeOfSgdModel`, not derived.
2. **Dynkin's formula for the exit-time boundary-value problem of `cor:fpt`.**
   The mean-exit-time theorems take the boundary-value problem as a hypothesis
   (`MeanExitTimeBVP`) and prove everything downstream of it.
3. **Dynkin's formula for the occupation-time boundary-value problem of
   `cor:dwell`**. The dwell theorems take
   `ShoulderDwell.OccupationTimeBVP` at the band indicator as a hypothesis. It
   is the *same* modeling input as (2), applied to the occupation functional
   rather than the exit time, which is what the appendix proof says, and it is
   satisfiable: `ShoulderDwell.dwellTime_satisfies_bvp` exhibits a solution and
   `dwell_hypotheses_satisfiable` the whole stack.
4. **The quasi-static reading of Adam's second-moment estimate (`rem:adam`)**
   (history: `rem:adam` is no longer in the paper, and
   `prop:precond-full` carries its content. The current paper names this
   reading nowhere and no paper result consumes `AdamQuasiStatic`). That the Adam escape exponent
   *is* `2α/η · (√V + ε)/V` packages the constancy of the second-moment estimate
   across the band and the identical action of the normalization on the two
   constructions. It is carried by `PgdBaseline.AdamQuasiStatic` and inhabited by
   `.adamQuasiStatic_satisfiable`; everything downstream of it — the strict
   monotonicity of the map and the equivalence of the two orderings — is proved.
   It is also *derived* from the preconditioned reading below, at
   `m = 0` (`PgdBaseline.adamQuasiStatic_precondExponent`).
4a. **The preconditioned reading of `prop:precond-full`.**
   Three modelling steps, none of them a theorem: that the preconditioned
   recursion, with `P_k` fixed by the batches before step `k` (Adam's
   second-moment estimate lagged by one step), is read as the Itô process with
   `P` in its drift and noise, the register of (A1); that the entry `p` of `P`
   along `e_b` is constant across the band; and, in part (iii), that it equals
   `χ(m² + σ²_eff)`. The first is carried by the Dynkin hypotheses of the
   preconditioned theorems, which are `MeanExitTimeBVP` and `OccupationTimeBVP`
   at slope `p U'` and diffusion coefficient `p² s` (`s = η σ²_eff`), the same
   structures as for (A1) and no new one. The second is built into those
   arguments, `p` being a real number, exactly as the constancy of `σ²_eff` is
   built into `meanExitTime`. The third is the definition of
   `PgdBaseline.precondExponent`, and the time comparison sees it only through
   the binder `hprod : pd * sd < pl * sl`.
4b. **Continuity of the exit and occupation times up to the closed interval.**
   `htau` / `hD` :
   `ContinuousOn (tau s) (Set.Icc a b)` is bound **separately from** `hDynkin`
   in every paper-facing declaration
   (`htau` of `KramersExitTime.mean_first_passage_kramers_exponent_of_dynkin`,
   `htaud`/`htaul` of `KramersBridge.eventually_mean_first_passage_lift_lt_direct_smallLearningRate`,
   `hD` of `ShoulderDwell.mean_dwell_kramers_exponent_of_dynkin`,
   `htau`/`hD` of `.mean_dwell_ratio_of_dynkin`). It is *not* implied by
   `MeanExitTimeBVP`, which gives differentiability on the **open** `Ioo a b`
   only; it is consumed by `eq_meanExitTime_of_bvp` as `hTcont`, whose
   docstring reads "any solution that is continuous up to the closed interval".
   Mathematically innocuous — the true mean exit time is continuous — but it is
   the same *kind* of object as the Dynkin clause, so the paper carries it in
   the same place: Theorem 1(iii), `cor:fpt` and `cor:dwell` all end their
   Dynkin clause with "and continuous up to the closed interval",
   `thm:directional` with "each continuous up to the closed interval", and the
   two Dynkin rows of `tab:hyp` print it as well.
5. **The stationary law of the projected recursion (`prop:clamp`)** is
   proved (section `prop:clamp` below): the law of the walk's supremum `M` is the
   recursion's unique stationary law (`LindleyStationary`), and its atom at `0`
   is the Spitzer series (`SpitzerCyclic`, `SpitzerIdentity`, `SpitzerGaussian`,
   joined in `LindleyClamped`). The paper relies on no unproved random-walk
   fact; its citations of Spitzer (1956) and Asmussen (2003) are attributions.
   Its proof of the finiteness of `M` follows the Lean route: the Gaussian
   tail `P(S_j > 0) ≤ e^{−ρ²j/2}` is summable, so by Borel–Cantelli only
   finitely many `S_j` are positive, and the mean is bounded by `∑_j E[S_j⁺]`.
6. **`Σ_PGD = Σ_obj` (Appendix A.7, `tab:hyp` "projection").** Projection's
   gradient noise is read into the diffusion undiminished, an input to every
   statement of A.7 that leaves out what the return step does to the update on
   the active set. In Lean it is the definition
   `PgdBaseline.pgdDiffusionMatrix Sobj := Sobj` (with `pgdDiffusionAlong` its
   reading along `v`); that this matrix is the projected recursion's diffusion
   is not proved, which is the paper's "input".
7. **A common mean latent weight for projection (Appendix A.7, `tab:hyp`).**
   Projection has no latent coordinate, so the common mean latent weight of §3.1
   does not apply to it and the comparison assumes it in addition. In Lean it
   is built into the statements, not bound: `pgdDiffusionMatrix`,
   `directDiffusionMatrix` and `liftDiffusionMatrix` are formed from one `Sobj`
   (and the lift's from one `C` and one `HVH`), that is, at one iterate.

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
| (A1) | Itô surrogate and i.i.d. batches, with the Lipschitz clause (regularity left this label for the notation of §3.1) | `Assumptions.SdeOfSgdModel` (**consumed by no theorem**: `grep -rn "SdeOfSgdModel" *.lean` returns four hits, all inside `Assumptions.lean`, the same standing as `SlackJacobianIdentity` below — it records that the surrogate is assumed and carries no field for the modelling clause), `Assumptions.IidBatchStream` (consumed through the pairwise-independence and identical-distribution binders of the `CrossCov*` theorems, and in its mutual form, `iIndepFun X μ` with `IdentDistrib (X k) (X 0)`, by `CrossCovSLLN.crossCovEstimator_raw_windowCentred_l2_bound_of_iidBatches`) |
| — | Regularity (square-integrable `g₀`, `δθ̃`, `r₂` and the Hessian of a batch; symmetric `H`; `σ²_obj > 0`), a sentence of the notation in §3.1 and no longer a clause of (A1) | the `SqIntegrableVec` and `Hᵀ = H` binders of `MinibatchNoise.projDiffusion_lift_eq_frozen_form` and `.projDiffusion_excess_small_jitter`, which are all that Theorem 1(i)–(ii) consume beyond (A2)'s first clause; the batch Hessian's (`hHb`) is consumed only by `.projDiffusion_excess_small_jitter_of_independent_perturbation`, Lemma 1(vi)'s independent-latent-weight sentence. `σ²_obj > 0` is consumed by the orderings, as `hsd : 0 < sd` of the `KramersBridge`, `ShoulderDwell` and preconditioned ordering theorems (its own row of `tab:hyp`), not by Lemma 1(i)–(iii) or (vi) |
| (A2) | "Slope and barrier": (a) `ψ` continuously differentiable and non-decreasing, `\|ψ'(w) − σ_s\| ≤ ε` on the interval `B`, `0 ≤ ε < σ_s`; (b) `L̃` continuous on `[a,b]` and differentiable inside, `a < w_b < w_s < b` with `w_s` the maximizer and `w_b` the minimizer of `L̃` on `[a,b]`, `L̃(w_b) < L̃(w_s)`, `L̃` non-decreasing on `[w_b,w_s]` | (a): `Assumptions.SinglePrefactor` (**consumed by no theorem outside `Assumptions` and `SoftplusBand`**: the declarations behind Theorem 1(i)–(ii) take the slope as a real `s`; its own API proves the bracket of A.1, `.diffusion_bracket`, and the `ε = 0` collapse, `.subsingleton_of_tol_zero`), `SoftplusBand.singlePrefactor_softplus_relative`. (b): the binders `hU`, `hU'`, `hawb`, `hwbws`, `hwsb`, `hbarrier`, `hmax`, `hmin` of the `KramersExitTime`, `ShoulderDwell` and `KramersBridge` theorems, and `hgrad` of `BarrierRescaling.quasipotential_gap_bounds` for the monotone climb. These binders are the paper's A2(b), not a weakening of it. `Assumptions.SingleBarrierPotential` is **consumed by no theorem** and is not A2(b): it adds `C²` and a third point `w_a` |
| `c > 0` (not a numbered assumption; it enters the ordering of `cor:fpt`, `cor:dwell` and `prop:precond-full`(iii), and in the paper's words "enters no part of" Theorem 1) | The net coupling at the reference iterate along `e_b`, `c = 2 e_bᵀ H Σ_ξ e_b + ϱ(e_b)`, is positive. Read at the **one** direction `e_b` and the one iterate, which is what the proofs consume; the retired (A3) printed the band- and direction-uniform `κ‖e‖² ≤ eᵀ(HΣ_ξ + Σ_ξᵀH)e + ϱ(e)` and was therefore strictly stronger than the Lean binder | `MinibatchNoise.projDiffusion_direct_lt_lift_of_netAlignment` (the ordering under it) and `MinibatchNoise.projDiffusion_lift_ge_direct_add_of_netAlignment` (the quantitative excess `s²κ‖e‖²`), both of which bind `{kappa : ℝ} {e : n → ℝ}` and take `hnet` **at that one `e`**: instantiate `e := e_b`, `κ := c`, and with `‖e_b‖ = 1` the binder `hnet` reads `c ≤ c`. `MinibatchNoise.netAlignment_of_frozenAlignment_and_remainder` (a coupling margin `κ_C` with a remainder channel `τ < κ_C` gives `c ≥ κ_C − τ`) |
| — | Slack Jacobian is the identity (a fact of eq. (2), no longer a labeled hypothesis) | `Assumptions.SlackJacobianIdentity` (consumed by no theorem) |
| — | The chain `g = g₀ + H δθ̃ + r₂` (the definition of `r₂`, no longer a labeled hypothesis) | `MinibatchNoise.probeGrad_eq_frozen_add_hessian_add_remainder`, `MinibatchNoise.refinedChain_of_remainder` |
| — | Common reference iterate (a convention, now the common mean latent weight, a sentence of the notation in §3.1; no longer numbered) | — (see above) |
| Dynkin, `cor:fpt` | the mean first-passage time solves the exit-time boundary-value problem | `KramersExitTime.MeanExitTimeBVP` (carried by every paper-facing exit-time theorem; satisfiable, `meanExitTime_satisfies_bvp`) |
| continuity, `cor:fpt` and `cor:dwell` | the exit time and the occupation time are continuous on the closed `[a,b]` | `htau` / `hD` : `ContinuousOn … (Set.Icc a b)`, bound separately from `hDynkin` in all four paper-facing declarations and consumed by `eq_meanExitTime_of_bvp` as `hTcont`. Printed in the paper as the trailing clause "and continuous up to the closed interval", and in the Dynkin rows of `tab:hyp` |
| Dynkin, `cor:dwell` | the mean occupation time of the band solves the boundary-value problem with right-hand side `−1_{[a,w_s]}` off `w_s`, and is continuously differentiable on `(a,b)` | `ShoulderDwell.OccupationTimeBVP`, whose clauses `hasDerivAt_fst` and `continuousOn_fst` are the paper's "continuously differentiable on `(a,b)`" (carried by `mean_dwell_kramers_exponent_of_dynkin`, `mean_dwell_ratio_of_dynkin`, `eventually_mean_dwell_lift_lt_direct_smallLearningRate`, `eventually_mean_dwell_additive_lift_lt_direct`; satisfiable, `ShoulderDwell.dwellTime_satisfies_bvp`, `.dwell_hypotheses_satisfiable`) |
| `rem:adam` quasi-static (remark retired; history, consumed by no current paper statement) | the Adam escape exponent at directional variance `V` is `2α/η · (√V + ε)/V` | `PgdBaseline.AdamQuasiStatic` (carried by `adam_exponent_lt_iff`; inhabited by `adamQuasiStatic_satisfiable`; met at `m = 0` by the exponent `prop:precond-full` derives, `adamQuasiStatic_precondExponent`) |
| Dynkin, `prop:precond-full`(ii) | the mean first passage and the mean time below the barrier top of `dw = -p L̃' dt + √η p σ_eff dB`, `p` constant, solve the problems of `cor:fpt` and `cor:dwell` for that diffusion | `KramersExitTime.MeanExitTimeBVP (p ^ 2 * s) (fun y => p * U' y)` and `ShoulderDwell.OccupationTimeBVP (p ^ 2 * s) (fun y => p * U' y) (bandIndicator a ws)`, `s = η σ²_eff`: the existing structures at the preconditioned slope and diffusion coefficient (satisfiable: the closed forms at `p s` divided by `p`, `KramersExitTime.meanExitTimeBVP_precond_div`, `ShoulderDwell.occupationTimeBVP_precond_div`) |
| `prop:clamp` noise | `ζ₀, ζ₁, …` independent standard normal | the binders `iIndepFun zeta P` and `∀ i, HasLaw (zeta i) (gaussianReal 0 1) P` of `ProjectionClamp.measure_visitEndsAt` and below; the later-visit theorems (`ProjectionClamp.measure_inter_visitFrom` and after) also bind `∀ i, Measurable (zeta i)` (inhabited: `ProjectionClamp.clamp_hypotheses_satisfiable`, from mathlib's `exists_iid`, which supplies all three) |
| `prop:clamp` means along the recursion | that `E[W_k]` converges; the paper does not assert it, its `E[W_∞]` being the mean of the stationary law, `E[M]`, whose finiteness and `ησ` scaling are proved | the hypothesis `hm` of `ProjectionClamp.tendsto_integral_clampIterate_homogeneous`; the conclusion is the `ησ` scaling only. Inhabited by `ProjectionClamp.homogeneity_hypotheses_satisfiable` (`m = 1`, instantiated conclusion `ησ`), a deterministic path on a Dirac space; a witness with i.i.d. Gaussian draws would need uniform integrability |

Notes that Appendix A.1, Step 3 of Lemma 1's proof and the paper's remarks
record, and their witnesses:

- **A2(a)'s tolerance is the substance.** The clause brackets the direct
  diffusion, `(σ_s − ε)²σ²_obj ≤ ψ'(w)²σ²_obj ≤ (σ_s + ε)²σ²_obj` on `B`
  (`Assumptions.SinglePrefactor.diffusion_bracket`), and at `ε = 0` the
  interval collapses to a point for any `ψ` with injective derivative
  (`Assumptions.SinglePrefactor.subsingleton_of_tol_zero`, and its softplus
  instance in `SoftplusBand`). For softplus, `ψ'' ≤ ψ'` gives
  `e^{−δ}ψ'(w) ≤ ψ'(u) ≤ e^{δ}ψ'(w)` for `|u − w| ≤ δ`
  (`SoftplusBand.logistic_relative_bracket`), so the clause holds on every
  interval of half-width `log(1+ρ)` with `ε = ρσ_s`, at any depth of the
  shoulder (`SoftplusBand.singlePrefactor_softplus_relative`,
  `.exists_singlePrefactor_softplus_inside_shoulder`).
- **The chain is an identity once `r₂` is the discrepancy**, and `ξ` is of order
  one in the fluctuation rather than a remainder: a quadratic bound on
  `δg − H δθ̃` along the family scaling the latent-weight fluctuation to zero forces
  `ξ ≡ 0` (`MinibatchNoise.noise_eq_zero_of_quadratic_bound`). (History: this
  is why the quadratic remainder clause (A4′-q) was retired for (A4′-r).)
- **`c > 0` is a sign condition with empirical content, not a consequence.**
  The paper says the ordering "rests on `c > 0`, a sign the method does not fix
  and the experiments do not measure". A two-point batch law realizes `Σ_ξ` of
  either sign (`MinibatchNoise.LocationWitness.coupling_pos` / `.coupling_neg`);
  the chain with a remainder bound does not sign it
  (`CouplingSign.coupling_sign_not_automatic_under_remainder_control`,
  `CouplingSign.positiveAlignment_fails_under_remainder_control`); it is
  satisfiable (`CouplingSign.positiveAlignment_satisfiable`), and the
  one-dimensional Gaussian instance of Step 3, with `r₂ ≡ 0`, has `c ≥ 4` and
  projected diffusions `1` and `9`
  (`DiffusionOrdering.updateCov_ordering_strict_nonvacuous`).
  `WitnessInstance.hypothesis_stack_is_inhabited` is stated on the *earlier*
  chain's full `Σ_slack`, not on `c`, and backs no current paper statement.

History of the retired remainder labels: **(A4′-r) was satisfiable where
(A4′-q) was not**. The remainder clause bound the remainder alone, and at small
jitter scale no clause bounding the frozen coupling by the jitter term can hold,
the coupling being first order in that scale and the jitter term second
(`MinibatchNoise.remainderControl_fails_small_jitter`,
`MinibatchNoise.remainderControl_caps_frozen_coupling`). Both labels left the
paper (table below); the current paper has neither.

## Paper wording after the readability revision

The revised paper renamed its labels for readability and moved some notation; **no statement changed and no Lean declaration changed**.
Lean identifiers keep the old words (`jitter`, `frozen`, `slack reading`,
`pullbackLandscape`, ...). The map from the paper's present wording to the
older wording used in the rest of this file:

| paper now | older paper wording (this file) |
|---|---|
| Theorem 1, "The coupling term and the escape time"; parts (i) Decomposition, (ii) Order in the fluctuation, (iii) Escape | "The coupling channel and the time on the shoulder"; (i) Channels, (ii) Orders, (iii) Escape and dwell |
| underbraces of `eq:sigma-eff`: direct softplus value, coupling term, fluctuation term, remainder | objective contribution, coupling term, jitter term, remainder channel |
| the coupling `Σ_ξ` | the frozen coupling |
| `c`, the coupling term plus the remainder | the net coupling |
| the mean latent weight `θ̄` (with `e_bᵀθ̄` in `B`), at which both methods are read | the reference iterate on the band (the common reference iterate, `(C0)`) |
| a coordinate direction `e_b` of the latent weight (equivalently of the slack) | the bias channel, a unit direction of the slack subspace |
| the expected loss `L̃` | the pullback landscape |
| the barrier height `α` | the reference action |
| the interval `[a,b]` of A2(b) | the escape interval |
| A1 "Independent batches and an Itô process"; A2 "Slope and barrier", parts (a) and (b) | A1 "Itô surrogate with i.i.d. batches"; A2 "Band and barrier", band clause and barrier clause |
| the interval `B` of A2(a) | the (operative) band |
| constant across the barrier | constant across the barrier band |
| `cor:dwell`, "Time below the barrier top" | "Time on the shoulder; the dwell half of Theorem 1(iii)" |
| Remarks 1–3 (`rem:matched-cell`, now titled "A perturbation independent of the batch"; `rem:power`; `rem:power-precond`) | Remarks 2–4; `rem:scope` (old Remark 1) was removed, its content being Theorem 1(ii) |

**The remainder.** The body now defines it in one line,
`ϱ = σ²_eff − e_bᵀ Cov[g₀ + H δθ̃] e_b`, and Appendix A's opening states the
three-term form the proofs and the Lean development use,
`ϱ(e) = 2 eᵀCov[g₀,r₂]e + 2 eᵀ H Cov[δθ̃,r₂] e + eᵀCov[r₂,r₂]e`, with the
identity `Cov[H δθ̃, r₂] = H Cov[δθ̃, r₂]` that links them (bilinearity of the
covariance for a deterministic `H`; symmetry of `H` is not needed for it). The
one-line form is the expansion of `Cov[a + r₂]` with `a = g₀ + H δθ̃`; the
three-term form is the one `UpdateCovariance.projDiffusion_decomposition_with_remainder`
and its neighbours state.

## Paper numbering

**Current numbering.** §2.1 the shoulder, §2.2 the lift; §3.1 the fluctuation and its coupling (the regularity sentences
and the common mean latent weight live here), §3.2 Hypotheses (A1, A2), §3.3
Escape time (Theorem 1, `thm:main`), §3.4 Joint necessity (Theorem 2), §3.5
Preconditioned steps (Proposition 1, `prop:precond`); §4.1 energy-based models,
§4.2 convex potential flows, §4.3 Darcy, §4.4 the image regularizer. Appendix A:
A.1 the hypotheses (`tab:hyp`), A.2 joint necessity, A.3 Lemma 1
(`lem:diffusion`) and Remark 1 (`rem:matched-cell`), A.4 escape time
(Corollary 1 `cor:fpt`, Remark 2 `rem:power`, Corollary 2 `cor:dwell`), A.5 the
proof of Theorem 1, A.6 preconditioned steps (Proposition 2 `prop:precond-full`,
Remark 3 `rem:power-precond`), A.7 projection (Theorem 3 `thm:directional`,
Theorem 4 `thm:pgd-zero`, Proposition 3 `prop:clamp`).

Theorem 1 of the body has three parts: (i) collects Lemma 1(ii)–(iii)
verbatim, (ii) is Lemma 1(vi) with its two coefficient *displays* abridged to
a description and a by-number pointer, every clause of the conclusion
printed, and (iii) collects `cor:fpt` with its explicit `L(δ)` bracket
abbreviated to a pointer *and* `cor:dwell`, which the body does not print as
an environment of its own, **without their ordering clause**, which the body
prints as the paragraph after the theorem. Its proof, Appendix A.5, is the
four pointers and it has no Lean declaration of its own. Joint necessity is
**Theorem 2**, stated in Section 3.4 and proved in Appendix A.2, as the
**conditional-independence clause alone**; the two substitution clauses are
printed as a paragraph after it. The projection results are Theorems 3 and 4
in Appendix A.7.

**The earlier hypothesis labels.** The module docstrings use earlier labels
for the hypotheses. They map onto the current ones, cut against the binders
of the declarations that consume them, as follows:

| earlier | now | why |
|---|---|---|
| (A1) | (A1), plus a regularity clause | the square-integrability of old (A4) and the positivity `σ²_obj > 0` of old (A5) are binders of the identity and of the direct baseline, not hypotheses about training |
| (A2), (A5) | (A2) "Band and barrier" | both describe where the iterates are; the `c_α η σ²_eff ≤ α` clause of old (A5) is a binder of no clause of Theorem 1 (only of `FreeDiffusion.metastable_window`, the crossover remark) and became a sentence in App. A.1, which the current paper no longer carries, together with the crossover remark |
| (A3) | a sentence of notation | consumed by no theorem (`SlackJacobianIdentity`) |
| (A4) | the definition of `r₂` in the notation | `probeGrad_eq_frozen_add_hessian_add_remainder` is an identity once `r₂` is the discrepancy |
| (C0) | a sentence of notation | a convention, not a binder of any declaration; the theorem preambles fix the common iterate directly |
| (A4′-r), (A6) | (A3) "Net positive alignment", itself retired for the scalar `c` and the sign condition `c > 0` | the pair `(κ_C, τ, τ < κ_C)` entered `projDiffusion_direct_lt_lift_of_frozenAlignment` only as `κ_C − τ`; the merged inequality is the weaker hypothesis (`netAlignment_of_frozenAlignment_and_remainder`) and the ordering under it is `projDiffusion_direct_lt_lift_of_netAlignment` |

The two current hypotheses, (A1) and (A2), are stated in Section 3.2 and
discussed one by one in Appendix A.1, together with the sign condition
`c > 0`; Dynkin's formula stays inside the statement of Theorem 1(iii), of
`cor:fpt` and of `cor:dwell`.

## Theorem 2 (joint necessity), Section 3.4 — proof in Appendix A.2

Stated on the probe estimator at a fixed iterate, for **every** probe count `T`
and **either** divisor (`T−1` or `T`).

The theorem printed in the body is the conditional-independence clause
**alone**, the rows marked (i) below, and the two substitution clauses, the rows
marked (ii), are printed as the paragraph after it, because each is
substitution of a factor of zero into the definition `eq:cross-cov` rather than
a theorem. Appendix A.2 proves all four.

| Clause | Lean |
|---|---|
| (i) conditional independence ⇒ `Σ_ξ = 0` a.e. | `CrossCovIndep.popCrossCov_centered_eq_zero_of_indepFun`, `CrossCovIndep.condPopCrossCov_condExp_centered_ae_eq_zero` |
| (i) unbiasedness, either divisor | `CrossCovProbe.probeCrossCovWith_unbiased_of_indepFun`, `CrossCovProbe.probeCrossCov_centred_unbiased_of_iid`, `CrossCovProbe.probeCrossCovMean_centred_expectation_of_iid` |
| (i) `O(1/T)` mean-square error | `CrossCovProbe.probeCrossCov_entry_l2_rate`, `CrossCovSLLN.crossCovEstimator_raw_windowCentred_l2_bound`, `CrossCovSLLN.integral_sq_windowMean_mul` |
| (i) `O(1/T)` mean-square error under the hypotheses the theorem prints: i.i.d. batches (A1), per-probe conditional independence of `θ̃` and `g₀`, finite moments | `CrossCovSLLN.crossCovEstimator_raw_windowCentred_l2_bound_of_iidBatches`: `E[(Σ̂_ξ)ᵢⱼ²] ≤ 2σᵢⱼ²/T + 2σ_u²σ_v²/T²` at divisor `T`, every binder of the bound above derived. It binds second moments of `θ̃ᵢ` and `g₀ⱼ` where the theorem prints fourth |
| (i) per-probe to collection-level independence (A.2: "the joint law of the `2T` vectors is the product over probes of the per-probe laws, each of which factorizes by the conditional independence, so the collections … are independent") | `CrossCovSLLN.indepFun_collections_of_iIndepFun` (mutually independent probes, independence within each probe ⇒ the two collections independent; any index type, so it gives the `Fin T` `hcoll` of `CrossCovProbe` as well as the `ℕ` one of `CrossCovSLLN`), `CrossCovSLLN.indepFun_comp_of_identDistrib` (identically distributed probes carry the per-probe independence from one probe to all). Proved on the generating π-system of finite cylinder events, not by writing the joint law as a product measure |
| (i) a.s. convergence to zero | `CrossCovProbe.probeCrossCov_centred_tendsto_zero_ae`, `CrossCovSLLN.crossCovEstimator_raw_tendsto_zero_ae` |
| (ii) no learnable slack ⇒ bias-channel reading ≡ 0 | `CrossCovProbe.probeSlackReading_eq_zero_of_jacobian_eq_zero`, `CrossCov.slackReading_eq_zero_of_jacobian_eq_zero` |
| (ii) body constant in the batch ⇒ estimator ≡ 0 | `CrossCovProbe.probeBodyFluct_eq_zero_of_const`, `CrossCovProbe.probeCrossCovWith_eq_zero_of_body_const`, `CrossCov.crossCovEstimator_eq_zero_of_body_const` |
| combined contrapositive | `CrossCovProbe.probe_joint_necessity_finite` |
| sample-cross-covariance identity | `CrossCovProbe.probe_sample_crossCov_identity`, `CrossCovSLLN.sample_crossCov_identity` |
| Bessel factor `T/(T−1)` | `CrossCovProbe.probeCrossCov_eq_smul_crossCovEstimator` |

**Which independence is used where.** Unbiasedness and the almost-sure limit
need the pairwise form; the mean-square rate needs the independence of the two
sample means (`CrossCovSLLN.integral_sq_windowMean_mul`), which comes from the
collection-level independence, which in turn comes from mutual independence
across probes and independence within each probe
(`CrossCovSLLN.indepFun_collections_of_iIndepFun`): the one step that uses the
i.i.d. clause beyond its pairwise form, as the appendix says. (A1) states i.i.d.
(mutual) and the appendix says which step consumes what. Non-degeneracy of the
conditional-independence clause is listed below.

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
| Step 1, the trace split of `σ²_Jac`, with `tr(V H) ≥ 0` for a positive semidefinite `H` | `MinibatchNoise.trace_covMat_slack_refined` |
| Step 2, update covariance exactly — (ii) `eq:sigma-eff` | `MinibatchNoise.projDiffusion_lift_eq_frozen_form`, `UpdateCovariance.projDiffusion_decomposition_with_remainder_symmetric`, `UpdateCovariance.covMat_add_three_self` |
| Step 2, direct baseline — (iii) first clause | `UpdateCovariance.projDiffusion_direct_baseline` |
| Step 2, the full-form display: written on `Σ_slack`, `σ²_eff = σ_s²σ²_obj + 2 e_bᵀ H Σ_slack e_b − e_bᵀ H V H e_b + 2 e_bᵀ Cov[g₀,r₂] e_b + e_bᵀ Cov[r₂] e_b`, the self-coupling inside `Σ_slack` counting the fluctuation term twice | `MinibatchNoise.projDiffusion_lift_eq_full_form` (with `crossTerm H S e = 2 eᵀ H S e` at symmetric `H`); the same subtraction on the coupling form, `CouplingSign.crossTerm_frozen_eq_slack_sub`, `.crossTerm_frozen_neg_of_slack_neg`, `.crossTerm_frozen_le_slack` |
| Step 3, sign of the excess — (iii) `σ²_eff − σ_s²σ²_obj ≥ c`, unconditionally, strict wherever `c > 0` | `MinibatchNoise.projDiffusion_lift_ge_direct_add_of_netAlignment` and `.projDiffusion_direct_lt_lift_of_netAlignment` at `e := e_b`, `κ := c`; `TraceLemmas.quadForm_conj_nonneg` (the jitter term `≥ 0`); `MinibatchNoise.projDiffusion_direct_lt_lift_of_frozenAlignment` (the earlier `κ_C − τ` form) |
| Step 3, the orders — **(vi)**, the exact `t`-polynomial, its linear coefficient `2 e_bᵀ H Cov[u,g₀] e_b + 2 e_bᵀ Cov[g₀,w₁] e_b`, and the jitter term `e_bᵀ H Cov[u] H e_b` in its quadratic coefficient | `MinibatchNoise.projDiffusion_excess_small_jitter` (the polynomial, exactly, with `crossTerm H Ξ e = eᵀ(HΞ + ΞᵀH)e = 2 eᵀHΞe` at symmetric `H`); `.projDiffusion_direct_lt_lift_small_jitter` (positive linear coefficient ⇒ the ordering at every small `t`); `.projDiffusion_lift_lt_direct_small_jitter` (negative ⇒ reversal); `.projDiffusion_excess_small_jitter_of_no_frozen_coupling` (the excess starts at `t²`) |
| (vi)'s independent-latent-weight sentence needs **two** zeros | `.projDiffusion_excess_small_jitter_of_no_frozen_coupling` binds `hΞ : Cov[u,g₀] = 0` **and** `hw : Cov[g₀,w₁] = 0`. Both are discharged for a centered perturbation independent of the batch, at any batch law, on the product space of batch and perturbation: `MinibatchNoise.covMat_prod_perturbation_batch_eq_zero` (`Σ_ξ = 0`), `.covMat_prod_batch_hessianFluct_eq_zero` (`Cov[g₀,w₁] = 0` for `w₁ = (H(X) − H)u`, the curvature-fluctuation part of the remainder), composed in `.projDiffusion_excess_small_jitter_of_independent_perturbation`, which binds square integrability of `g₀`, `u`, `w₂` and of the entries of the batch Hessian `H(X)` (`hHb`), and the zero mean of the perturbation; the paper's regularity names the Hessian of a batch for this reason. The paper's sentence, "has Σ_ξ = 0 and Cov_X[g₀,w₁] = 0 as well, w₁ being the curvature-fluctuation part of the remainder", is this theorem; its finite-sample companion is `.outerMoment_prod_hessianFluct_eq_zero` |
| Step 4, stationary density — (iv) | `StationaryDensity.stationaryDensity_zeroFlux`, `.eq_stationaryDensity_of_zeroFlux`, `.boltzmann_zeroFlux`, `.hasDerivAt_effectivePotential` |
| Step 4, the zero-flux condition follows from the equation: a non-negative solution with constant current `J` and `U_eff` bounded below has `J = 0` and is the stationary density; the lower bound cannot be dropped, a potential unbounded below admitting a transient solution of nonzero current | `StationaryDensity.eq_stationaryDensity_of_constant_current`; `StationaryDensity.exists_nonzero_constant_current` (the transient instance) |
| Step 4, the drift is the gradient of each method's own expected loss "whenever an integrable bound dominates the update, and for every finite batch law without one" | `UpdateDrift.hasFDerivAt_pullbackLandscape_of_dominated` (the dominated case), `UpdateDrift.gradient_pullbackLandscape_finiteBatch` (a finite batch law, differentiability alone) (see Convention above) |
| Step 5, barrier bracket — (v) | `BarrierRescaling.quasipotential_gap_bounds`, `.quasipotential_gap_of_const_diffusion`, `QuasipotentialFTC.quasipotential_gap_bounds_fully_discharged` |

Supporting: non-degeneracy of the ordering
(`DiffusionOrdering.updateCov_ordering_strict_nonvacuous`,
`.magnitude_channel_ordering`); the directional reduction is a **lower** bound
off eigendirections (`FWReduction.fwActionScalar_le_fwActionMulti_alongLine`).
The curvature of `log π` at a minimum (`LogDensityCurvature.*`) backed a
paragraph after Lemma 1's proof that the paper no longer carries, so nothing in
the current paper cites it.

**The anti-overclaim is itself a theorem.** (The current paper prints only the
second half of this paragraph, in Step 4 of Lemma 1's proof; the curvature half
backs no current paper statement and is kept as a record.) The lift does not add curvature to
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
hypotheses are discharged for such a perturbation at any batch law in
`.projDiffusion_excess_small_jitter_of_independent_perturbation`), against first
order at a positive linear coefficient — a positive net coupling is necessary
for the effect and not sufficient. This is the content the paper prints as
Theorem 1(ii) / Lemma 1(vi).

**The trace is the wrong monitor for direction.** By (i) the reading contains
`tr(V H) ≥ 0`, so a positive trace does not sign `Σ_ξ`, and for `d > 1` a
positive trace is compatible with a negative direction of the coupling form
(`CouplingSign.counterexample_trace_slack_pos`, `.ce_crossTerm_neg`).

## `cor:fpt` (escape time), Appendix A.4 — the exit-time half of Theorem 1(iii)

| Clause | Lean |
|---|---|
| BVP and its unique solution | `KramersExitTime.MeanExitTimeBVP`, `.meanExitTime_satisfies_bvp`, `.eq_meanExitTime_of_bvp` |
| two-sided Arrhenius bracket, with `L(δ)` depending on `δ`, `L̃`, `[a,b]` and `x` and not on `ησ²_eff` | `KramersExitTime.meanExitTime_arrhenius_lower_bound_uniform` (one `L` for every diffusion `σ² > 0`; carried to the Dynkin solution by `.eq_meanExitTime_of_bvp`), `.meanExitTime_upper_bound`. Not `.meanExitTime_arrhenius_lower_bound` or `.mean_first_passage_arrhenius_of_dynkin`, which fix one `σ²` and give a constant `c = 2L/σ²` that depends on it |
| exponent limit `ησ²_eff log τ → 2α` | `KramersExitTime.tendsto_mul_log_meanExitTime`, `.mean_first_passage_kramers_exponent_of_dynkin` |
| exponent strictly decreasing in `σ²_eff`, so "whenever `c > 0`" the lift's is strictly the smaller | `BarrierRescaling.arrheniusExponent_strictAntiOn` combined with `MinibatchNoise.projDiffusion_direct_lt_lift_of_netAlignment` (at `e := e_b`, `κ := c`, the strict ordering of Lemma 1(iii)). `BarrierRescaling.arrheniusExponent_lift_lt_direct` is an earlier form, with a non-negative coupling term and fluctuation term and no remainder |
| lift's mean first-passage smaller for every sufficiently small `η` | `KramersBridge.eventually_mean_first_passage_lift_lt_direct_smallLearningRate` |
| `η log(T_dir(x)/T_lift(x)) → 2α(1/σ²_dir − 1/σ²_lift)` | `KramersBridge.tendsto_mul_log_meanExitTime_ratio` (on the closed form; to the Dynkin solutions through `KramersExitTime.eq_meanExitTime_of_bvp`) |
| at fixed `η` the ratio pinned between the two displayed bounds | `KramersBridge.meanExitTime_ratio_bracket` |
| the sharp prefactor not addressed: `K e^{2α/σ²}` satisfies both bounds on a window of small diffusions | `KramersBridge.kramers_hypothesis_sandwiched` |

The crossover past `ησ²_eff = 2α` (`FreeDiffusion.metastable_window`,
`.isFreeExitTime_isTheta_inv`, `.arrheniusTime_not_isBigO_inv`) was a row of
this table; it is not in the paper, which has no free-diffusion regime and no
crossover remark.

No prefactor is claimed on either side. A2(b) asks for continuity on `[a,b]`,
differentiability inside, the extremal pair and the barrier; the binders `hU`,
`hU'`, `hawb`, `hwbws`, `hwsb`, `hbarrier`, `hmax`, `hmin` are exactly that, so
the Lean hypothesis is the paper's and not a weakening of it (the retired (A5)
was the earlier label for this clause).

## Shoulder dwell (`cor:dwell`), Appendix A.4 — the dwell half of Theorem 1(iii)

In `ShoulderDwell.lean`. **Stated in Appendix A.4 only**: the body absorbs it
into Theorem 1(iii), which prints its dwell limit, its ratio limit and its
subexponential-fraction clause, omits the hypothesis pointer and the Dynkin
qualifier "at the band indicator", and carries its ordering clause in the
paragraph after the theorem rather than inside the statement. The corollary is proved the way the
appendix proves it: the exit-time development is **generalized** from the
constant right-hand side to a general one and then instantiated at the band
indicator `1_{[a,w_s]}`. Nothing in `KramersExitTime` is weakened — the
exit-time statements are the case `f ≡ 1` of the new ones, in both directions.

| Clause | Lean |
|---|---|
| the BVP `½σ²D″ − L̃′D′ = −1_{[a,w_s]}` at every point of `(a,b)` other than `w_s`, `D(b)=0`, `D′(a⁺)=0`, with `D′` continuous across `w_s` | `ShoulderDwell.OccupationTimeBVP` (`generator` and `hasDerivAt_snd` off the jump, `continuousOn_fst` for `D′`), `.occupationTime_satisfies_bvp`, `.dwellTime_satisfies_bvp` |
| its unique solution | `ShoulderDwell.eq_of_occupationTimeBVP` (through `.constant_of_hasDerivAt_zero_off_point`) |
| the closed form `eq:pf-dwell` | `ShoulderDwell.occupationTime`, `.dwellTime` |
| `D(x) ≤ T(x)` (the upper bound, by monotonicity) | `ShoulderDwell.occupationTime_le_meanExitTime`, `.dwellTime_le_meanExitTime` |
| the windowed lower bound, inner window inside the band, with the `L(δ)` of `cor:fpt` | `ShoulderDwell.occupationTime_window_lower_bound`, `.occupationTime_arrhenius_lower_bound_uniform` (one `L` for every diffusion) |
| `ησ²_eff log D(x) → 2α` | `ShoulderDwell.tendsto_mul_log_dwellTime`, `.mean_dwell_kramers_exponent_of_dynkin` |
| `ησ²_eff log (T(x)/D(x)) → 0` | `ShoulderDwell.tendsto_mul_log_meanExitTime_div_dwellTime`, `.mean_dwell_ratio_of_dynkin` |
| the dwell is the shorter under the lift for every sufficiently small **step size** — one shared `η → 0⁺`, two fixed variances `sd < sl`, the multiplicatively coupled pair of `thm:main`(iii) and Step 3 of `app:proof-fpt` | `ShoulderDwell.eventually_dwellTime_lift_lt_direct_smallLearningRate`, `.eventually_mean_dwell_lift_lt_direct_smallLearningRate`, `.eventually_occupationTime_lift_lt_direct_smallLearningRate`; witness `.cubicBarrier_dwell_lift_lt_direct_smallLearningRate` |
| *a companion, and not that clause*: the same ordering on the **additively** coupled family `σ²` against `σ² + σ_Jac²`, where the small parameter is the shared variance and the gap does not shrink with it | `ShoulderDwell.eventually_dwellTime_additive_lift_lt_direct`, `.eventually_mean_dwell_additive_lift_lt_direct` |
| the exit-time results re-derived as `f ≡ 1` | `ShoulderDwell.occupationTimeBVP_one_iff`, `.occupationTime_one`, `.meanExitTime_satisfies_bvp_of_occupation`, `.eq_meanExitTime_of_occupationTimeBVP` |
| non-vacuity on the cubic barrier, band `[−3/2, 1]` | `ShoulderDwell.dwell_hypotheses_satisfiable`, `.cubicBarrier_dwell_kramers_exponent` (limit `8/3 = 2α`), `.cubicBarrier_dwell_ratio` |

**The jump at `w_s`.** The second differentiation does not apply at the
jump: `D″(w_s)` does not exist, the closed form having a corner in `D′` there
(`ShoulderDwell.not_continuousAt_bandIndicator`). The paper therefore states the
equation at every point of `(a,b)` other than `w_s`, with `D′` continuous
across it by the corollary's hypothesis that `D` is continuously differentiable
on `(a,b)`, and its proof says the argument applies "to this right-hand side,
bounded and continuous off `w_s`". That is `OccupationTimeBVP`, which asks for
the interior equation only where the right-hand side is continuous and carries
`continuousOn_fst` for `D′`, and the hypothesis `hfc` of
`eq_of_occupationTimeBVP`, a right-hand side continuous off one point. The
closed form satisfies this form, so the Dynkin hypothesis is not vacuous, and
uniqueness holds across the jump, so it pins its solution.

**Two couplings, and which one is the corollary's clause.** "For every sufficiently small step
size" is one shared learning rate going to zero at two *fixed* effective variances — the pair
`(η·sd, η·sl)` with `sd < sl`, multiplicatively coupled, since the diffusion coefficient of the
boundary-value problem is `η σ²_eff`. That is
`eventually_dwellTime_lift_lt_direct_smallLearningRate`, mirroring
`KramersBridge.eventually_meanExitTime_lift_lt_direct_smallLearningRate`, which draws the same
distinction for the exit time. The additive theorem is a **different** statement — the small
parameter is itself the direct arm's diffusion, the lift's is that plus a fixed `σ_Jac² > 0`, and
the gap between them does not shrink — so it does not discharge the corollary's clause and
neither implies the other. It is kept as the companion for the variance-offset family of
`eq:sigma-eff`.

**What is not claimed.** `D(x)/T(x) → 1` is nowhere stated: the paper claims
only that the fraction is subexponential, and the limit would need
`inf_{(w_s,b]} L̃ > L̃(w_b)`, which A2(b) does not supply. The paper states the
dwell for the band `[a,w_s]` only. The Lean's general right-hand side
(`ShoulderDwell.BandRHS`) carries the one property of the band the lower bound
uses, that the right-hand side is at least one on `[a,w_s]`, as its
`one_le_on_band` clause; the paper's counterpart is the sentence that the inner
window lies below the barrier top.

## Preconditioned steps (`prop:precond-full`, Proposition 2), Appendix A.6 — the body's Proposition 1 (`prop:precond`), Section 3.5

No new module: each statement sits beside the one it
extends, the exit time in `KramersExitTime` and `KramersBridge`, the dwell in
`ShoulderDwell`, the stationary density in `StationaryDensity`, and the
exponent map in the Adam section of `PgdBaseline`, which it generalizes. The
preconditioned diffusion is the diffusion of `MeanExitTimeBVP` at slope `p U'`
and diffusion coefficient `p² s` (`s = η σ²_eff`), so no new boundary-value
problem was introduced; the modelling inputs are item 4a above.

| Clause of `prop:precond-full` | Lean |
|---|---|
| (i) Lemma 1(i)–(iii), (v), (vi), Theorem 1(i)–(ii) and Theorem 2 with its substitutions hold unchanged | **no new declaration**: the declarations listed under Lemma 1 and Theorem 2 are statements about the batch law at one iterate (Theorem 2 also about i.i.d. batches) and bind no update rule |
| (i) Lemma 1(iv) at step `ηp`: `σ_eff⁻² exp(-(2/(ηp)) U_eff)` is zero-flux stationary for drift `-p L̃'` and diffusion coefficient `η p² σ²_eff`, and every zero-flux stationary density for them is of this form | `StationaryDensity.stationaryDensity_zeroFlux_precond` (existence) and `StationaryDensity.eq_stationaryDensity_of_zeroFlux_precond` (uniqueness, the constant identified; `eq_stationaryDensity_of_zeroFlux` at step `ηp²` for the slope `p L̃'` and potential `p U_eff`). Not a bare instantiation of `stationaryDensity_zeroFlux` at `η := ηp`, which gives the drift `-L̃'` and coefficient `ηp σ²_eff`, the preconditioned process slowed by `p`; the lemma multiplies that statement by `p` |
| (ii) the generator identity: `p T_p` solves `eq:pf-bvp` at step `ηp`, and the same for the indicator right-hand side | `KramersExitTime.meanExitTimeBVP_precond`, `ShoulderDwell.occupationTimeBVP_precond`; converses `.meanExitTimeBVP_precond_div`, `.occupationTimeBVP_precond_div` |
| (ii) `T_p = T^{(ηp)}/p`, `D_p = D^{(ηp)}/p` | `KramersExitTime.eq_meanExitTime_div_of_precond` (through `eq_meanExitTime_of_bvp`), `ShoulderDwell.eq_dwellTime_div_of_precond` (through `eq_of_occupationTimeBVP`) |
| (ii) at fixed `p`: `ηpσ²_eff log T_p(x) → 2α`, `ηpσ²_eff log D_p(x) → 2α`, `ηpσ²_eff log(T_p(x)/D_p(x)) → 0` as `ησ²_eff → 0⁺` | `KramersBridge.mean_first_passage_kramers_exponent_precond_of_dynkin`, `ShoulderDwell.mean_dwell_kramers_exponent_precond_of_dynkin`, `.mean_dwell_ratio_precond_of_dynkin`; the factor `1/p` through `KramersBridge.tendsto_mul_log_div_precond` |
| (iii) `Ψ(v) = 2α/(η v χ(m² + v))` is strictly decreasing | `PgdBaseline.precondExponent`, `.precondExponent_strictAntiOn` (binds `χ > 0` on `(0,∞)` and the strict increase of `v χ(m² + v)`; non-increase of `χ` is not used) |
| (iii) `eq:precond-ratio`: the middle identity, and `1 < Ψ(σ²_dir)/Ψ(σ²_lift) ≤ κ` | `PgdBaseline.precondExponent_ratio_eq`, `.precondExponent_ratio_bounds` (with `κ > 1` read as `sd < sl`) |
| (iii) the lift's mean first passage and mean time below the barrier top are the smaller for every sufficiently small `η` | `KramersBridge.eventually_mean_first_passage_precond_lift_lt_direct`, `ShoulderDwell.eventually_mean_dwell_precond_lift_lt_direct`, both through `KramersBridge.eventually_const_mul_lt_of_tendsto_mul_log`. They bind `hprod : pd * sd < pl * sl` and no `χ`; at `pd = χ(m² + sd)`, `pl = χ(m² + sl)` that binder is the strict increase of `v χ(m² + v)` at `sd < sl` |
| (iii) power family `χ(v) = (v + ε)^(-γ)`, `ε ≥ 0`, `γ ∈ [0,1)` or `γ = 1` with `m² + ε > 0`: meets the hypotheses, and the ratio is at least `κ^(1-γ)` | `PgdBaseline.powerChi`, `.powerChi_hypotheses` (positivity, non-increase, strict increase of `v χ(m² + v)`), `.powerChi_ratio_ge` |
| (iii) Adam's `χ(v) = (√v + ε)⁻¹`, `ε ≥ 0`: meets the hypotheses, and the ratio is at least `κ^(1/2)` | `PgdBaseline.adamChi`, `.adamChi_hypotheses`, `.adamChi_ratio_ge` (stated with `Real.sqrt (sl / sd)`) |
| the retired `rem:adam` as the case `m = 0` | `PgdBaseline.adamQuasiStatic_precondExponent`: the exponent (iii) derives meets `AdamQuasiStatic`, whose map `adamVarianceMap` and its monotonicity are therefore the `m = 0` case of `adamChi_hypotheses` |
| `rem:power-precond`, the direct exponent `2α(σ_s²μ² + ε)^γ/(ησ_s²σ²_obj)` and Adam's `2α(σ_sμ + ε)/(ησ_s²σ²_obj)` | `CouplingPower.directPrecondExponent_powerChi_eq`, `.directPrecondExponent_adamChi_eq` (at `σ_s > 0`, `σ²_obj > 0`, `ε ≥ 0`, with `μ = √(ḡ_θ² + σ²_obj)`) |
| `rem:power-precond`, the orders as `σ_s → 0`: for `χ(v) = (v + ε)^(-γ)`, `σ_s^(-2(1-γ))` at `ε = 0` and `σ_s^(-2)` at a fixed `ε > 0` (the law of `cor:fpt` at step `ηε^(-γ)`); for Adam's normalization, `σ_s^(-1)` at `ε = 0` and `σ_s^(-2)` at a fixed `ε > 0` (the law of `cor:fpt` at step `ηε^(-1)`); the lift's exponent of the same orders. The body's "from `σ_s^{-2}` to `σ_s^{-1}` at `ε = 0`" (§3.5) is the Adam case | Direct: `CouplingPower.directPrecondExponent_orders`; lift: `CouplingPower.liftPrecondExponent_orders`; both instances of `CouplingPower.precondExponent_orders`, stated for any diffusion `V = Θ(σ_s²)` positive near `0`, with `PgdBaseline.precondExponent`, `.powerChi`, `.adamChi`. Each is six conjuncts of `Asymptotics.IsTheta` (or a ratio limit) along `𝓝[>] 0`, with `α, η, σ²_obj > 0` and `ḡ_θ` fixed and the mean gradient `σ_s ḡ_θ` shared: at `ε = 0`, `Θ(σ_s^(-2(1-γ)))` (power) and `Θ(σ_s⁻¹)` (Adam, through `adamChi_zero_eq_powerChi`, Adam at `ε = 0` is the power family at `γ = 1/2`); at a fixed `ε > 0`, `Θ(σ_s⁻²)` for both, with the exponent over `arrheniusExponent α (η L) V` tending to one at `L = ε^(-γ)` (power) and `L = ε⁻¹` (Adam). **The remark and the Lean match.** The remark states the two regimes as the limits `ε = 0` and fixed `ε > 0` as `σ_s → 0`, as the Lean proves them, with no crossover stated by either, and gives Adam's fixed-`ε` law at step `ηε⁻¹`, which is the Lean's (Adam's `ε` is added to `√v`, so it is not the power family's `ηε^(-1/2)`). The one remaining difference is in the Lean's favor: its orders hold at every real `γ`, where the paper's power family takes `γ ∈ [0, 1]`. The lift's half binds the hypotheses of `CouplingPower.sigmaEffPoly_isTheta_sq` (`hc`: the remainder constant `τ` below the leading coefficient, `|ϱ(s)| ≤ τ s²` near `0`), which `rem:power` states and `rem:power-precond` inherits by "under the reading of `rem:power`". "The ratio of the two is bounded by `eq:precond-ratio`" is `PgdBaseline.precondExponent_ratio_bounds`, unchanged |

**Which statements are conditional.** The (ii) identities and limits, and the
(iii) time comparison, are conditional compositions: they carry the Dynkin
hypotheses of the preconditioned diffusion, exactly as `cor:fpt` and
`cor:dwell` carry theirs, together with continuity up to the closed interval.
The (iii) facts about `Ψ` and the two instances are unconditional
real-analysis facts; the reading of `p` as `χ(m² + σ²_eff)` is the definition
of `Ψ`, not a hypothesis of them.

**Where the Lean is weaker in hypothesis than the paper.** The time comparison
needs only two positive constants with `pd sd < pl sl`, no `χ` at all;
`powerChi_ratio_ge` needs only `γ ≥ 0`, `ε ≥ 0` and `sd ≤ sl`, and none of the
conditions that make the power family admissible; the strict decrease of `Ψ`
does not use the non-increase of `χ`, which only the upper bound `≤ κ` consumes.

**Where the Lean proof is not the appendix proof.** Step 3 of the appendix
transfers Step 3 of `app:proof-fpt` with a chosen `δ`; the Lean comparison goes
through the exponent limits of (ii) instead, at logarithmic scale
(`eventually_const_mul_lt_of_tendsto_mul_log`). The conclusion is the same, and
the window constant never appears.

## One projected coordinate at the boundary (`prop:clamp`, Proposition 3) — stated and proved in Appendix A.7

**All three clauses are formalized.** `ProjectionClamp` carries the dwell, the walk's
supremum and its mean, and the clamped fraction at every finite step. The
stationary law and its Spitzer series are in `LindleyStationary`,
`SpitzerCyclic`, `SpitzerIdentity`, `SpitzerGaussian` and `LindleyClamped`;
mathlib v4.31.0 has no fluctuation theory (no ladder epochs, no
Sparre-Andersen lemma, no Spitzer identity), so these build it.

| Clause of `prop:clamp` | Lean |
|---|---|
| a visit ends at the first step whose update points into the cone (the inequality proved is `γ + σζ_k < 0`) | `ProjectionClamp.clampStep_pos_iff`, `.clampStep_eq_zero_iff`, `.VisitEndsAt`, `.clampRun`, `.clampRun_eq_of_visitEndsAt`, `.visitEndsAt_clampRun` |
| the visit is a visit **of the recursion**: `W` stays at the boundary through it and is released at its end | `ProjectionClamp.clampIterate_eq_zero_of_forall_le`, `.clampIterate_eq_zero_of_visitEndsAt`, `.clampIterate_pos_of_visitEndsAt`. That the recursion *returns* to the boundary infinitely often is not proved and not asserted by the paper |
| it lasts a geometric number of steps | `ProjectionClamp.measure_visitEndsAt`, `.measureReal_visitEndsAt`, `.measureReal_visitEndsAt_eq_geometricMeasure` (against mathlib's own `geometricMeasure`), `.measureReal_clampRun_eq` |
| of mean `1/Φ(−ρ)` | `ProjectionClamp.tsum_visitLength`; `Φ` is `ProbabilityTheory.cdf (gaussianReal 0 1)` (`.clampExitProb`, `.clampExitProb_pos`, `.clampExitProb_lt_one`) |
| and so does every later visit, whatever the step at which it begins | `ProjectionClamp.measure_inter_visitFrom`: for every `k` and every event `E ∈ σ(ζ₀,…,ζ_{k−1})` on which `W_k = 0`, `P(E ∩ {the visit from k lasts n+1 steps}) = P(E)·(1−Φ(−ρ))ⁿ·Φ(−ρ)`. At the start of a visit, `E = {W_k > 0, W_{k+1} = 0}`: `.measurableSet_visitStart`, `.measure_visitStart_inter_visitFrom`. As a conditional law and mean `1/Φ(−ρ)` given `E` with `P(E) ≠ 0`: `.cond_measureReal_visitFrom`, `.tsum_cond_visitLength`. At a random start `τ` with `{τ = k} ∈ σ(ζ₀,…,ζ_{k−1})` and `W_τ = 0`: `.measure_iUnion_inter_visitFrom`, which covers every stopping time of the recursion's own filtration (`.iterateFiltration_le_pastDraws`). Supporting: `.VisitFrom`, `.visitFrom_iff` (the visit from `k` ends at the first releasing draw after `k`), `.clampIterate_congr`, `.pastDraws`, `.measurable_clampIterate_pastDraws` (`W_k` is a function of `ζ₀,…,ζ_{k−1}`), `.measure_inter_visitEndsAt_shift` (the later draws are independent of the past), `.tsum_succ_mul_geometric`. **Not through a strong Markov theorem**, which is not formalized: the start event is conditioned on at a fixed time and summed over its value. That a later visit occurs (recurrence) is not proved, so the conditional forms carry `P(E) ≠ 0` |
| the recursion in the paper's letters, and the reduction to `ρ` | `ProjectionClamp.clampIterate`, `.clampIterate_succ_paper`, `.eta_mul_add_eq` |
| the clamped fraction is a function of `ρ` alone | `ProjectionClamp.clampIterate_eq_zero_iff_unit`, `.measureReal_clampIterate_eq_zero_scale_free` — at **every step**: the clamped event is literally the same event at every scale `ησ` |
| it increases in `ρ` | `ProjectionClamp.clampIterate_antitone_in_rho`, `.clampIterate_eq_zero_of_le_rho`, `.measureReal_clampIterate_eq_zero_mono_rho` — at every step, by a pathwise coupling on one and the same draws. **Non-decreasing**, not strictly increasing |
| it tends to `1` as `ρ → ∞` | `ProjectionClamp.measure_forall_le`, `.measureReal_clampIterate_eq_zero_ge`, `.tendsto_measureReal_clampIterate_eq_zero_atTop` — at every step, from the bound `(1 − Φ(−ρ))^k` |
| it tends to `0` as `ρ ↓ 0` | false at any finite step, so it is stated for the limit law: `LindleyClamped.tendsto_lindleyLimitLaw_real_zero_nhdsGT`, through `SpitzerGaussian.tendsto_clampedFraction_nhdsGT` (the paper's harmonic bound) |
| `W_k = max_{0≤j≤k} S_j` in distribution (the reversal identity) | `ProjectionClamp.clampIterate_eq_sup'_sub` (the recursion unrolled, pathwise), `.clampIterate_eq_maxWalk_revDraws` (reversing the block, pathwise), `.map_revDraws_eq` (an i.i.d. block has the same law reversed), `.map_clampIterate_eq_map_maxWalk` (**the identity**) |
| `M = sup_j S_j` is finite a.s. (the paper's proof takes the same route, citing Asmussen as attribution) | `ProjectionClamp.hasSubgaussianMGF_of_hasLaw`, `.hasSubgaussianMGF_negSum`, `.measureReal_walk_pos_le`, `.ae_bddAbove_walkSum` — **proved** by Chernoff (`P(S_j>0) ≤ exp(−ρ²j/2)`) plus Borel–Cantelli; no strong law, no fluctuation theory |
| `M` has a finite mean, at most `∑_{j≥1} E[S_j⁺]` (the paper's proof takes the same route, citing Asmussen as attribution) | `ProjectionClamp.le_inv_mul_exp`, `.lintegral_ofReal_walkSum_le`, `.lintegral_iSup_ofReal_walkSum_lt_top` — **proved**: `M ≤ ∑_j (S_j)^+` and the tilted bound makes that a geometric series |
| `W_k → M` in distribution | `ProjectionClamp.maxWalk_mono`, `.tendsto_maxWalk_atTop`, `.tendsto_integral_clampIterate` — **proved**, in the defining form (integrals against bounded continuous test functions converge) |
| `E[W_∞] = ησ·m(ρ)` with `m` finite and free of `η, σ` | `ProjectionClamp.lintegral_iSup_ofReal_walkSum_homogeneous` with the rows above — **proved for `M`**, whose law is the stationary law (next row) |
| the paper's `M = ησ·sup_j(−∑_{i<j}(ρ+ζ_i))` and its scaling | `ProjectionClamp.walkSum`, `.iSup_walkSum_homogeneous`. `iSup_clampIterate_homogeneous` scales the supremum of the *iterates*, which is a different pathwise object from the paper's `M` (a supremum of the walk's partial sums); the two agree only in distribution at a fixed time, as its docstring says |
| `E[W_∞] = ησ·m(ρ)`, the homogeneity | `ProjectionClamp.clampIterate_homogeneous`, `.integral_clampIterate_homogeneous`, `.tendsto_integral_clampIterate_homogeneous`, `.iSup_clampIterate_homogeneous`; the hypothesis of the third is inhabited by `.homogeneity_hypotheses_satisfiable` |
| the hypothesis stack is inhabited | `ProjectionClamp.clamp_hypotheses_satisfiable` (mathlib's `exists_iid`) |
| the law of `M` is the recursion's unique stationary law | `LindleyStationary.lindleyStep_lindleyLimitLaw` (stationary), `.eq_lindleyLimitLaw_of_lindleyStep_eq` and `.lindleyStep_eq_self_iff` (unique among probability laws on `ℝ`); the recursion converges to it in distribution from every starting law, `.tendsto_integral_iterate_lindleyStep`; the bridges to the recursion, `.map_iSup_walkSum_eq_lindleyLimitLaw` and `.map_clampIterate_eq_iterate_lindleyStep`. Proved from the pathwise identity `M = max(0, X₀ + M')` and `S_k → -∞` almost surely; no Markov-chain theory |
| under it, `P(W_∞=0) = exp(−∑_{k≥1} k⁻¹Φ(−ρ√k))` | `LindleyClamped.lindleyLimitLaw_real_zero_eq_clampedFraction`; along the recursion `P(W_k=0)` decreases to it, `LindleyStationary.antitone_measureReal_clampIterate_eq_zero` and `LindleyClamped.tendsto_measureReal_clampIterate_eq_zero_clampedFraction` |
| the ladder-epoch identity `P(M=0) = exp(−∑_{j≥1} j⁻¹P(S_j>0))` (the paper's Spitzer citation) | `SpitzerIdentity.spitzer_ladder_identity`, for any real walk with i.i.d. increments under summability of the series, with the converse `.summable_of_measureReal_forall_le_pos` and the divergent case `.measureReal_forall_le_eq_zero_of_not_summable`. From the cyclic lemma `SpitzerCyclic.cyclic_ladder_identity`, exchangeability under rotations, ladder-epoch renewal and Tonelli; no generating functions |
| `P(S_j>0) = Φ(−ρ√j)`, and the series | `SpitzerGaussian.measureReal_walkSum_pos_eq` (the partial sum of `j` standard normals is `N(0, j)`), `.spitzerSeries`, `.summable_spitzerTerm`, `.spitzerSeries_strictAntiOn`, `.clampedFraction_strictMonoOn`, `.tendsto_clampedFraction_atTop`, `.tendsto_clampedFraction_nhdsGT` |
| the stationary clamped fraction increases in `ρ` | `LindleyClamped.lindleyLimitLaw_real_zero_strictMonoOn`, strictly |

Conventions to check against the paper: `VisitEndsAt ρ z n` means the visit
lasts `n + 1` steps — `n` clamped steps and the releasing one — which is why the
mean of `n + 1` is `1/Φ(−ρ)` and not `(1−Φ(−ρ))/Φ(−ρ)`; the draws are indexed
from `0`; `η` and `σ` enter the recursion only through `c = ησ`, with
`clampIterate_succ_paper` the machine-checked bridge to the paper's display; and
the mean visit length is a convergent series over the visit-length law, not an
integral of an `ℕ`-valued random variable. `γ > 0` is not used by the dwell
clause; the stationary-law results take `ρ > 0`.

## The power of the positivity-map slope

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

The paper states exactly this and nothing stronger. Under a preconditioned
step, Adam's normalization included, the paper now derives the exponent through
`prop:precond-full` and states the orders in `rem:power-precond`: at `ε = 0`
both exponents pass to order `σ_s^{-1}`, so there is no order separation to
halve. Both are machine-checked; see the preconditioned-steps section above.
(History: before `prop:precond-full`, the paper stated the square-root reduction
without deriving it, and what was machine-checked was the ordering consequence
under the named hypothesis `PgdBaseline.AdamQuasiStatic`,
`PgdBaseline.adamVarianceMap_strictAntiOn` and `.adam_exponent_lt_iff`; that is
now the `m = 0` case of `prop:precond-full`(iii) and backs no current paper
statement.) The barrier convention (reading `α` in latent-weight units) rescales both
exponents alike, so the order equality and the bounded ratio are
convention-independent.

## Projection (Theorems 3 and 4) — stated and proved in Appendix A.7

| Paper | Lean |
|---|---|
| A.7's opening: the three diffusion matrices at a common mean latent weight, `Σ_PGD = Σ_obj`, `Σ_dir = σ_s²Σ_obj`, `Σ_lift = σ_s²(Σ_obj + C + HVH)` at a vanishing remainder, read along `v` | `PgdBaseline.pgdDiffusionMatrix` (the definition `Sobj`, the input of item 6 above), `.directDiffusionMatrix`, `.liftDiffusionMatrix`, and their readings `.pgdDiffusionAlong`, `.directDiffusionAlong`, `.liftDiffusionAlong`, formed from one `Sobj`, `C`, `HVH` (item 7). That `vᵀΣ_lift v` is the variance of the update along `v` is `UpdateCovariance.projDiffusion_decomposition_with_remainder_symmetric` at `R = 0` |
| Theorem 3, directional escape comparison (biconditional) | `PgdBaseline.arrheniusExponent_lift_lt_pgd_iff`, `.arrheniusExponent_lift_lt_pgd_iff_gate` |
| after Theorem 3: the bar rises to `vᵀΣ_obj v` while the coupling and fluctuation terms are of order `σ_s²` (`rem:power`), so the threshold fails for every sufficiently small `σ_s` | `PgdBaseline.eventually_not_gateHolds_of_excess_isBigO_sq` (the paper's convention: the lift's excess over direct along `v` is `O(σ_s²)`), with the weaker-hypothesis forms `.eventually_not_gateHolds_of_isLittleO`, `.eventually_not_gateHolds_of_isBigO_one` and `.eventually_not_gateHolds_of_isBigO_sq`. The `O(σ_s²)` of the coupling term is `CouplingPower.couplingTerm_isBigO_sq_of_relative`; the composition is described in the `SmallSlope` section docstring, since `CouplingPower` imports `PgdBaseline`. The fluctuation term's `O(σ_s²)` is stated as no big-O theorem: `jitterTerm_softplus` gives it as a polynomial with no term below `s²`, at `ε = 0` |
| direct softplus strictly below projection when the positivity map attenuates | `PgdBaseline.directDiffusionAlong_lt_pgdDiffusionAlong`, `.directDiffusionAlong_le_liftDiffusionAlong` |
| no magnitude comparison settles lift vs. projection | `PgdBaseline.exists_liftDiffusionAlong_lt_pgdDiffusionAlong`, `.not_posSemidef_liftDiffusionMatrix_sub_pgdDiffusionMatrix` |
| Theorem 4, projection has no coupling term | The sample form: `PgdBaseline.pgd_alignment_structurally_unavailable_of_le_one` (with the coupling matrix built from the estimator of `eq:cross-cov`: `C = 0`, the lift's matrix at slope one equal to `Σ_PGD`, the ratio zero, and the threshold failing at every slope `s ∈ (0, 1]`; through `.gateThreshold_nonneg_of_le_one` and `.not_gateHolds_of_no_coupling_of_le_one`), `PgdBaseline.pgd_alignment_structurally_unavailable` (the same at slope one). The population form, for a latent weight constant almost surely: `StructuralZeros.popCrossCov_centered_eq_zero_of_deterministic` (`Σ_ξ = 0`, against every second factor), `.iterateFluctuationCovariance_eq_zero_of_deterministic` (`V = 0`), `.quadForm_conj_eq_zero_of_deterministic` (`eᵀHVHe = 0` in every direction). `StructuralZeros.exists_batchCoupled_sigmaJacSq_eq_zero` |
| the gate is sufficient, not necessary | `PgdBaseline.gateHolds_of_couplingOnly_gate`, `.exists_gateHolds_of_couplingOnly_gate_failure` |
| gate witnesses | `PgdBaseline.gateWitness_arrheniusExponent_lift_lt_pgd`, `.gateWitness_trace_lt`, `.gate_discriminates_on_slope` |
| lift's mean first passage below projection's under the gate, for every sufficiently small projection diffusion `σ²_PGD = η vᵀΣ_PGD v`, with the surplus `Δ = η vᵀ(Σ_lift − Σ_PGD)v` fixed | `KramersExitTime.eventually_mean_first_passage_additive_lift_lt_direct` at `jac2 := η · gateSurplus s v Sobj C HVH`, positive under the gate (`PgdBaseline.gateSurplus_pos_iff_gateHolds`, `η > 0`). `PgdBaseline.eventually_mean_first_passage_lift_lt_pgd_of_gate` is the case `η = 1`: its offset is `gateSurplus`, without the factor `η`. At a prescribed pair of diffusions, under the explicit gap inequality: `.meanExitTime_lift_lt_pgd_of_window` |
| the threshold read net of the remainder channel (`app:proof-pgd`'s opening) | `PgdBaseline.liftDiffusionAlong_add_remainder`, `.arrheniusExponent_lift_lt_pgd_iff_of_remainder` |
| `rem:adam` (retired; its content is now `prop:precond-full`(iii), section above): the variance map is strictly decreasing and common, so the exponent ordering is the reverse variance ordering | `PgdBaseline.adamVarianceMap_strictAntiOn`, `.adam_exponent_lt_iff`; the quasi-static reading is the hypothesis `PgdBaseline.AdamQuasiStatic` (`.adamQuasiStatic_satisfiable`) |

The Lean biconditional is stated at zero remainder, and the with-remainder
version the paper uses is now proved as well
(`PgdBaseline.arrheniusExponent_lift_lt_pgd_iff_of_remainder`): the remainder channel enters the lift's directional diffusion
additively, so the threshold is the same arithmetic with `vᵀ R v` added to the
excess, and `R = 0` returns the theorem as `thm:directional` states it.
The slope-below-one clause of Theorem 4 is machine-checked
(`PgdBaseline.pgd_alignment_structurally_unavailable_of_le_one`): the bar
`s⁻² − 1` is non-negative on `(0, 1]` and the ratio is zero. The slope is read
as positive; the Lean window needs only `T ≠ 0` where the paper says `T ≥ 2`. **The paper reports no
value for the gate ratio and does not claim the gate is met at our operating
point.** The `D = 1` PGD/lift tie is not claimed anywhere.

## Places where the paper is deliberately weaker than the Lean

Each is stated in the appendix; weaker in the paper is allowed, stronger is not.

- The `O(T^{-1/2})` rate is given in mean square plus Chebyshev; the Lean has
  the exact `L²` rate and the explicit sample-centered bound.
- `cor:fpt` states the two-sided bracket, the exponent limit and the small-`η`
  ordering; the Lean also has the window-explicit ordering and the additively
  coupled family.
- **Theorem 1(iii) states less than `cor:fpt` and `cor:dwell` do**: the two exponent limits and the ratio limit, and not the
  ordering, which the body carries as prose with `c > 0` named in the open. The
  body is therefore weaker than the appendix, which is the allowed direction.
- **Theorem 2 states less than Appendix A.2 proves**: the conditional-
  independence clause alone, the two substitutions being printed as prose.
- The directional reduction is a lower bound off eigendirections.
- Theorem 3 is stated at zero remainder in the paper; the Lean also carries the
  remainder-channel form (`arrheniusExponent_lift_lt_pgd_iff_of_remainder`).
- `cor:dwell` states the two limits and the small-step ordering; the Lean also
  has the windowed lower bound with explicit constants, the general
  right-hand-side forms of all three clauses (`ShoulderDwell.BandRHS`),
  `D(x) ≤ T(x)` at every diffusion, and the ordering in a second coupling (the
  additive family) that the corollary does not state.
- `prop:precond-full`(iii) states its time comparison and its power-family
  bound under the conditions on `χ`; the Lean comparison needs only
  `pd sd < pl sl`, and the bound `κ^(1-γ)` only `γ ≥ 0`, `ε ≥ 0`.
- `prop:clamp`'s dwell is stated for the paper's Gaussian draws; the Lean proves
  the visit-length law from mutual independence and the standard-normal law
  alone, with no use of `γ > 0`, and identifies it with mathlib's own geometric
  distribution.
- `prop:clamp`'s "so does every later visit" is proved conditional on any event
  of `σ(ζ₀,…,ζ_{k−1})` on which `W_k = 0`, not only on the start of a visit,
  and at every random start of that kind
  (`ProjectionClamp.measure_inter_visitFrom`, `.measure_iUnion_inter_visitFrom`).
- Theorem 2 assumes finite fourth moments of `θ̃` and `g₀`; the batch-level
  mean-square bound needs only second moments of the two coordinates it reads,
  the summand being square-integrable because its factors are independent
  (`CrossCovSLLN.crossCovEstimator_raw_windowCentred_l2_bound_of_iidBatches`).

## Steps that are prose in the appendix, not in the development

The Itô surrogate and Dynkin's formula, for the exit time and for the
occupation time alike, and the other modeling inputs listed above. Each is
marked as such where it is used.

Proved, and therefore not on this list:

- the Spitzer ladder-epoch identity and the stationary law of the projected
  recursion (section `prop:clamp`);
- the slope-below-one clause of Theorem 4,
  `PgdBaseline.pgd_alignment_structurally_unavailable_of_le_one`;
- per-probe to collection-level independence,
  `CrossCovSLLN.indepFun_collections_of_iIndepFun`, the sentence of Appendix A.2
  before `eq:pf-meanprod-l2`; with it Theorem 2's mean-square bound is stated
  under the batch-level hypotheses the theorem prints, as
  `CrossCovSLLN.crossCovEstimator_raw_windowCentred_l2_bound_of_iidBatches`;
- `prop:clamp`'s "so does every later visit, whatever the step at which it
  begins", `ProjectionClamp.measure_inter_visitFrom` and its corollaries (table
  above), proved from the independence of the later draws and without a strong
  Markov theorem;
- the `2eᵀHVHe` subtraction at the (A6) counterexample,
  `CouplingSign.crossTerm_frozen_eq_slack_sub` with the sign transfer
  `.crossTerm_frozen_neg_of_slack_neg`;
- Theorem 3 net of the remainder channel,
  `PgdBaseline.arrheniusExponent_lift_lt_pgd_iff_of_remainder`, with the
  additive identity `.liftDiffusionAlong_add_remainder`;
- the Adam reduction, split: `PgdBaseline.adamVarianceMap_strictAntiOn` and
  `.adam_exponent_lt_iff` are theorems, and the quasi-static reading that
  licenses the substitution is the named hypothesis `PgdBaseline.AdamQuasiStatic`,
  carried in the register of the Dynkin inputs and inhabited by
  `.adamQuasiStatic_satisfiable`.

## Modules

38 registered. Backing a current paper statement through a row of this file
(29): `Assumptions`, `BarrierRescaling`, `CouplingPower`, `CouplingSign`,
`CrossCov`, `CrossCovIndep`, `CrossCovProbe`, `CrossCovSLLN`,
`DiffusionOrdering`, `FWReduction`, `KramersBridge`, `KramersExitTime`,
`MinibatchNoise`, `PgdBaseline`, `QuasipotentialFTC`, `ShoulderAttenuation`
(through the shoulder and the logistic derivative in `SoftplusBand`'s
statements), `SoftplusBand`, `StationaryDensity`, `StructuralZeros`,
`ProjectionClamp`, `ShoulderDwell`, `LindleyStationary`, `LindleyClamped`,
`SpitzerCyclic`, `SpitzerIdentity`, `SpitzerGaussian`, `TraceLemmas`,
`UpdateCovariance`, `UpdateDrift`. Present but not cited (9): `BandOccupancy`,
`QuasipotentialMonotone`, `InversePSDAntitone`,
`FreeDiffusion` (the crossover, not in the paper), `LogDensityCurvature` (the
curvature paragraph after Lemma 1's proof, deleted), `PowerOfS` (retired, must
not be cited), `PullbackHessian` (the curvature half of the anti-overclaim,
which the paper does not state), `TaylorChain` (the retired (A4) clauses) and
`WitnessInstance` (stated on the earlier chain's `Σ_slack`, not on `c`).
