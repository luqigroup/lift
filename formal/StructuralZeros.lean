import CrossCov
import CrossCovIndep

/-!
# Three architectural structural zeros: the consensus splitting, the deterministic latent weight,
and the reverse-KL decoupling

The paper draws three architectural corollaries from Theorem 1 (`thm:joint-necessity`), each of
which says that a named alternative construction reads a *structural* zero of the slack-channel
cross-covariance rather than a measured one. This file machine-checks all three, together with the
falsifiability boundary that the paper attaches to them, and it does so by instantiating the
deletions already proved in `CrossCov.lean` and `CrossCovIndep.lean` rather than by reproving them.

The first corollary is the **ADMM / consensus-splitting reading** (`docs/paper/v4/04_mechanism.tex`,
subsection "ADMM reading"; ICLR appendix `app:admm-reading`). Read as a consensus splitting, the
lift's latent weight plays the auxiliary primal variable `z` of the augmented Lagrangian and the slack
`b` its own additive offset inside the latent iterate. The classical splitting has no
data-conditioned body: its auxiliary variable is a free optimization variable, fixed across
batches. The modelling assumption is therefore that the splitting's latent iterate is a
function of the conditioning batch that does not in fact depend on it, and everything else follows:
the resampling fluctuation is identically zero, the window sample covariance of the auxiliary
variable is zero, and the cross-covariance estimator and its slack-channel reading vanish for every
iteration and every window length. That modelling assumption is the whole mathematical content of
the paper's sentence "its auxiliary variable is fixed across batches", and it is named
`BatchIndependent` here so that a reader can see exactly what is being assumed.

The second is the **deterministic limit of the latent weight** (`rem:bcond` of the ICLR version).
Replacing the conditioning batch by a fixed learnable code leaves the pooling and the slack
unchanged but freezes the body's argument, so the latent weight is again batch-independent and the
reading is zero. The paper draws a sharper distinction inside the four-architecture ablation of
`fig:four-cell-cross-cov`, and the distinction survives formalization: the two batch-independent
cells (direct softplus, and direct softplus with a bias) read zero **by the theorem**, because the
estimator itself vanishes, whereas the body-only cell (hypernet without bias) reads zero
**definitionally**, because there is no slack channel to contract against — its zero holds for
every possible pair of fluctuation sequences, with no hypothesis on the data at all. The two are
separated here by a theorem, not only by a docstring: `exists_bodyOnly_estimator_ne_zero_of_no_slack`
exhibits a body-only configuration whose estimator is nonzero while its slack-channel reading is
zero, which is impossible for the two batch-independent cells.

The third is the **forward-versus-reverse-KL asymmetry** of `rem:moreau-scope`. In reverse KL the
latent weight and the loss gradient are driven by different draws and decouple; formalized as conditional
independence, this is case (iii) of Theorem 1 and is imported wholesale from `CrossCovIndep.lean`,
in both its unconditional and its conditional (disintegration-kernel) forms, and extended here to
the slack-channel *reading* `J_bᵀ Σ_slack`. That the hypothesis stack is satisfiable without
degeneracy — independent, non-constant channels and an identity slack Jacobian, and the reading
still zero — is itself a theorem, `exists_indepFun_popSlackReading_eq_zero`. Two honesty points then
govern this section. First, the paper's forward-KL clause — that `σ_Jac²` is "generically nonzero"
when both channels ride the same batch — **is not a theorem and is not proved here**. It is an
empirical claim; the four-cell ablation and the training-time traces are its evidence. What
independence licenses is one direction only, and it is stated as such: a nonzero reading certifies
that the latent weight and the gradient are *not* independent (`not_indepFun_of_sigmaJacSq_ne_zero`),
which is strictly weaker than certifying that they ride the same draw. Nothing in the converse
direction follows, and `exists_batchCoupled_sigmaJacSq_eq_zero` exhibits a pair of non-independent,
genuinely batch-coupled fluctuations whose cross-covariance is nevertheless zero — in that witness
the gradient fluctuation is a *deterministic function* of the iterate fluctuation, so the two ride
the same draw in the strongest sense available, and the cross-covariance is still exactly zero.
The scope of that counterexample is stated exactly, and not inflated: it refutes the implication
from batch coupling *alone*, and `not_hessianChain_gradWitness` proves that the witness lies outside
the paper's assumption (A4), under which the leading term of `σ_Jac²` is the non-negative `tr(V H)`
of `Assumptions.ForwardKLChain.trace_leading_nonneg`. This file takes no position on the (A4)
regime; it establishes only that batch coupling by itself does not force a nonzero reading.

Second, the paper writes `σ_Jac² ≈ 0` for reverse KL rather than `= 0`, because real reverse-KL
pipelines decouple only approximately. Rather than assume the approximation away,
`sigmaJacSq_eq_remainder_of_indepFun` carries the residual coupling as an **explicit remainder**
`δg = δg₀ + r` with `δg₀` independent of `δθ̃`, and concludes the exact identity
`σ_Jac² = tr E[δθ̃ rᵀ]`: under approximate decoupling the surviving channel is exactly the trace of
the cross-covariance with the coupling remainder alone. An identity is not yet a smallness claim,
which is what `≈` asserts, so `abs_sigmaJacSq_le_of_indepFun_of_remainder_bounded` supplies the
quantitative half as well: with the fluctuation and the remainder bounded entrywise by `M_u` and
`M_r` almost surely, `|σ_Jac²| ≤ d M_u M_r`, and exact decoupling is `M_r = 0`.

The fourth result is the paper's own **falsifiability boundary**, also in `rem:moreau-scope`: if an
external constraint freezes the conditioning batch to a fixed anchor, the batch-induced fluctuation
vanishes and the mechanism falls silent. This is proved at both levels. At the finite-window level
a frozen batch sequence zeroes the fluctuation even when the body is a genuinely batch-dependent
map, which is a different hypothesis from the batch-independent-body deletion and needs its own
argument; the hypothesis is stated on the trailing window the estimator actually reads, which is
sharper than the paper's global freezing and has the paper's form as a corollary. At the population
level a deterministic latent weight zeroes the iterate fluctuation, hence both `σ_Jac² = tr E[δθ̃ δgᵀ]`
and the iterate-fluctuation covariance `V = E[δθ̃ δθ̃ᵀ]`, and hence both jitter-sourced terms of the
corrected effective diffusion of `docs/design/lemma1_rederivation.md` §2, so that the lift's
effective diffusion collapses onto the direct baseline's `s² σ_obj²` and the Kramers exponent it
feeds is the baseline exponent.

The last step of that boundary — "and therefore `μ_eff → 0`" — is where a formalization is easiest
to fake, and the file is explicit about it. Written pointwise, the hypothesis "`q` is `Θ(σ_Jac²)`"
is, *at* `σ_Jac² = 0`, logically equivalent to the conclusion `q = 0`; `sigmaJacDominated_zero_iff`
proves that equivalence rather than leaving it for a referee to notice. The corollaries therefore
use `SigmaJacDominatedOn`, in which the implied constants are quantified before the configuration
varies, as `Θ` intends. That version constrains the modulus across the whole admissible range, is
satisfiable by a modulus which is not identically zero (`exists_sigmaJacDominatedOn_ne_zero`), and
supports the paper's arrow as an actual limit (`tendsto_zero_of_sigmaJacDominatedOn`) and not only
as an evaluation at a point.

A note on which version of the mechanism this file speaks for. The design note
`docs/design/lemma1_rederivation.md` §1 establishes that the shipped Lemma 1's Hessian
decomposition is false — differentiation commutes with the batch expectation, so the pullback
Hessian carries no cross-covariance term — and the corrected route puts `σ_Jac²` in the diffusion
instead. Nothing in this file asserts either route. The one place where a downstream modulus is
mentioned, `eq_zero_of_sigmaJacDominatedOn`, is deliberately route-neutral: it takes as an explicit
hypothesis only that some modulus is two-sidedly proportional to `σ_Jac²` with uniform constants —
the common shape of the shipped `μ_eff = Θ(σ_Jac²/(d κ²))` and of the corrected route's
`c₁ σ_Jac² |H_bb|` diffusion term — and concludes that this modulus vanishes when `σ_Jac²` does. It
asserts neither proportionality. The effective-diffusion statement is route-neutral in the same way:
its three coefficients are universally quantified, so the shipped additive form
`σ_eff² = σ_s² σ_obj² + σ_Jac²` is the case `c₁ = 1`, `|H_bb| = 1`, `(H V H)_bb = 0` of it.

## Results

Batch independence and the two window-level deletions it drives:
* `BatchIndependent` — a latent weight whose value does not depend on the conditioning batch.
* `bodyFluct_eq_zero_of_batchIndependent`, `crossCovEstimator_eq_zero_of_batchIndependent`,
  `slackReading_eq_zero_of_batchIndependent` — the deletion (ii) chain for such a latent weight.
* `liftIterate`, `bodyFluct_liftIterate` — the lift's own latent weight `θ̃ = b + h_φ(X)`, and the fact
  that the slack drops out of `eq:lift-batch-fluct`: it has the same resampling fluctuation as its
  body alone.
* `exists_bodyFluct_liftIterate_ne_of_empty_window` — and the `T ≠ 0` hypothesis of that identity
  is load-bearing: at `T = 0` the identity is false.
* `slackReading_one` — with the identity slack Jacobian of (A3) the reading is the estimator itself.

The consensus splitting:
* `splittingIterate`, `splittingIterate_batchIndependent` — the classical splitting's auxiliary
  variable, fixed across batches.
* `bodyFluct_splitting_eq_zero` — its resampling fluctuation is identically zero.
* `splittingIterate_resamplingCovariance_eq_zero` — its variance under resampling is zero.
* `crossCovEstimator_splitting_eq_zero`, `slackReading_splitting_eq_zero` — hence `eq:cross-cov`
  vanishes identically for it.

The deterministic latent weight and the four-cell ablation:
* `anchoredParameter`, `anchoredParameter_eq_liftIterate_const`,
  `anchoredParameter_batchIndependent` — the fixed-code limit of the same latent weight.
* `crossCovEstimator_anchored_eq_zero`, `slackReading_anchored_eq_zero` — `rem:bcond`'s zero.
* `crossCovEstimator_directSoftplus_eq_zero`, `crossCovEstimator_biasOnly_eq_zero` — the two
  batch-independent cells: zero by the theorem.
* `slackReading_bodyOnly_eq_zero` — the body-only cell: zero definitionally, for every fluctuation
  pair, with no hypothesis on the data.
* `exists_bodyOnly_estimator_ne_zero_of_no_slack` — the two are genuinely different: a body-only
  configuration with a nonzero estimator and a zero reading.
* `exists_fullLift_slackReading_eq_zero` — and the converse of Theorem 1 is unavailable: all three
  ingredients present, reading still zero.

Reverse KL, and what forward KL does and does not license:
* `popSlackReading`, `popSlackReading_eq_zero_of_popCrossCov_eq_zero` — the population reading.
* `reverseKL_popCrossCov_eq_zero`, `reverseKL_sigmaJacSq_eq_zero`,
  `reverseKL_popSlackReading_eq_zero` — the unconditional decoupling.
* `reverseKL_popCrossCov_ae_eq_zero`, `reverseKL_sigmaJacSq_ae_eq_zero`,
  `reverseKL_popSlackReading_ae_eq_zero` — the conditional (given the body parameters) decoupling.
* `exists_indepFun_popSlackReading_eq_zero` — the hypothesis stack is satisfiable with both channels
  non-constant and the slack Jacobian the identity, so the reverse-KL zeros are not vacuous.
* `popCrossCov_add_right`, `popCrossCov_zero_right`, `popCrossCov_eq_remainder_of_indepFun`,
  `sigmaJacSq_eq_remainder_of_indepFun` — the exact identity carrying the coupling remainder.
* `abs_sigmaJacSq_le_of_bounded`, `abs_sigmaJacSq_le_of_indepFun_of_remainder_bounded` — and the
  quantitative half of the paper's `≈`: `|σ_Jac²| ≤ d M_u M_r` under entrywise bounds on the
  fluctuation and on the coupling remainder.
* `not_indepFun_of_sigmaJacSq_ne_zero` — the only implication the forward-KL side licenses: a
  nonzero `σ_Jac²` certifies that the two channels are not independent, which is strictly weaker
  than certifying that they ride the same draw.
* `threePointUniform`, `integral_threePointUniform`, `fluctWitness`, `gradWitness` — the witness
  space and the two fluctuations carried on it.
* `exists_batchCoupled_sigmaJacSq_eq_zero` — and the converse fails: two centred, integrable,
  dependent fluctuations, the second a deterministic function of the first, with `σ_Jac² = 0`.
* `not_hessianChain_gradWitness` — the scope of that counterexample: the witness admits no
  representation of the form (A4) posits, so it refutes the implication from batch coupling alone
  and says nothing about the regime (A4) describes.

The frozen anchor:
* `bodyFluct_eq_zero_of_frozen_window`, `crossCovEstimator_eq_zero_of_frozen_window`,
  `slackReading_eq_zero_of_frozen_window` — freezing the batch across the trailing window the
  estimator reads already silences a batch-dependent body, with no constraint outside the window.
* `bodyFluct_eq_zero_of_frozen_batch`, `crossCovEstimator_eq_zero_of_frozen_batch`,
  `slackReading_eq_zero_of_frozen_batch` — the globally frozen batch the paper states.
* `popCrossCov_centered_eq_zero_of_deterministic`, `sigmaJacSq_eq_zero_of_deterministic` — and
  zeroes the population `σ_Jac²`.
* `iterateFluctuationCovariance_eq_zero_of_deterministic`,
  `quadForm_conj_eq_zero_of_deterministic` — and the iterate-fluctuation covariance with it.
* `effectiveDiffusion_collapses_of_frozen_anchor` — so the corrected effective diffusion collapses
  onto the direct baseline's, and `kramersExponent_eq_of_effectiveDiffusion_eq` carries that to the
  escape exponent.
* `SigmaJacDominated`, `sigmaJacDominated_zero_iff`, `eq_zero_of_sigmaJacDominated` — the pointwise
  reading of `Θ(·)`, together with the theorem that at `σ_Jac² = 0` it is *equivalent* to the
  conclusion it would be used to draw, and is therefore not usable there.
* `SigmaJacDominatedOn`, `sigmaJacDominated_of_sigmaJacDominatedOn`,
  `eq_zero_of_sigmaJacDominatedOn` — `Θ(·)` with the implied constants quantified before the
  configuration, which is what makes the corollary a deduction rather than a restatement.
* `tendsto_zero_of_sigmaJacDominatedOn` — the paper's `σ_Jac² → 0` implies `μ_eff → 0` as an actual
  limit, by a squeeze.
* `exists_sigmaJacDominatedOn_ne_zero` — the uniform relation is satisfiable by a modulus that is
  not identically zero, so those corollaries are not vacuous.
* `mechanism_falls_silent_of_frozen_anchor` — the three consequences in one statement: `σ_Jac² = 0`,
  the dominated modulus is zero, and the effective diffusion is the direct baseline's.

## Honesty

No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses this file carries, and why each is a hypothesis rather than a theorem:

* `BatchIndependent e`, in the splitting, deterministic-latent-weight and two batch-independent ablation
  cells. This is a **modelling assumption about the architecture**, not a mathematical gap: it says
  the construction in question has no data-conditioned body. It cannot be a theorem because it is
  the definition of the architecture being modelled. It is stated in the weakest usable form —
  equality of the latent weight on any two batches — and every zero in the file is derived from it.
* `T ≠ 0`, wherever a window mean is taken. With `T = 0` the estimator's `(T : ℝ)⁻¹` factor is `0`
  by mathlib's junk-value convention and the window mean subtracted in `bodyFluct` is the empty
  sum, so `bodyFluct` degenerates to the uncentred body and the "slack drops out" identity is false.
  That failure is not left as a claim in this docstring: it is the theorem
  `exists_bodyFluct_liftIterate_ne_of_empty_window`. The hypothesis is the honest one and is present
  in `CrossCov.lean` for the same reason.
* `IndepFun`, respectively `CondIndepFun`, for the reverse-KL statements. This is the formal content
  of the paper's "in reverse KL the two decouple": under reverse KL the loss gradient is evaluated
  on model samples drawn independently of the conditioning batch. It is an assumption about how the
  estimator is constructed, and it is exactly the hypothesis of case (iii) of Theorem 1.
* `Integrable` hypotheses, and `[IsProbabilityMeasure μ]` in the centring statements. These are the
  hypotheses of `ProbabilityTheory.IndepFun.integral_bilin` and of the centring step; without them
  the population objects `E[δθ̃ | φ]` and `E[δθ̃ δgᵀ]` are undefined. The entrywise integrability
  hypotheses of `popCrossCov_add_right` are what make the integral additive; nothing forces the
  product of the fluctuation with an arbitrary remainder to be integrable.
* `Measurable`, `[StandardBorelSpace Ω]`, `[IsFiniteMeasure μ]`, `m' ≤ mΩ` in the conditional
  statements. These are mathlib's standing hypotheses for `CondIndepFun` and `condExpKernel`; they
  are inherited unchanged from `CrossCovIndep.lean`.
* Entrywise almost-sure bounds `M_u` on the iterate fluctuation and `M_r` on the coupling
  remainder, in `abs_sigmaJacSq_le_of_bounded` and its corollary. These are what turn the exact
  remainder identity into the smallness claim the paper's `≈` makes; without a bound on the residual
  coupling nothing forces the surviving trace to be small, and assuming the residual away is exactly
  the corner this development does not cut.
* An almost-everywhere constant latent weight, in the population frozen-anchor statements. This is the
  frozen anchor itself: an external constraint that fixes the conditioning set makes the produced
  latent iterate a single vector. It is stated almost everywhere rather than pointwise, which
  is the weaker hypothesis, and it is the modelling content of `rem:moreau-scope`'s "if an external
  constraint freezes `X` to a fixed anchor".
* Frozen batches on the trailing window, in the finite-window frozen-anchor statements. The
  hypothesis constrains only the `T` batches the estimator at iteration `t` reads, which is weaker
  than the paper's global freezing; the paper's own form is recovered as
  `bodyFluct_eq_zero_of_frozen_batch`.
* `SigmaJacDominatedOn scale S q`, in `eq_zero_of_sigmaJacDominatedOn`,
  `tendsto_zero_of_sigmaJacDominatedOn` and the capstone. This is the paper's `Θ(·)` written out as
  the two-sided bound it abbreviates, with the implied constants named, quantified, and — this is
  the point — quantified **before** the configuration, since that uniformity is the whole content of
  a `Θ`. The pointwise alternative `SigmaJacDominated` is kept in the file only so that
  `sigmaJacDominated_zero_iff` can record why it must not be used here: at `σ_Jac² = 0` it is
  logically equivalent to the conclusion, so a falsifiability corollary drawn from it would assume
  what it proves. The uniform relation is a hypothesis because the file takes no position on which
  downstream modulus it applies to, and in particular asserts nothing about the shipped Lemma 1's
  `μ_eff`, whose Hessian derivation `docs/design/lemma1_rederivation.md` §1 refutes.

What this file does **not** prove, and why:

* It does not prove that the lift's own reading is nonzero. Theorem 1 is a necessity statement, the
  paper says so explicitly, and `exists_fullLift_slackReading_eq_zero` shows the converse fails as
  a general implication: a configuration can carry slack, a batch-dependent body and their coupling
  and still read zero. Whether a *trained* lift reads nonzero is measured, not proved.
* It does not prove the paper's "generically nonzero" clause for forward KL. That clause is
  empirical. See `not_indepFun_of_sigmaJacSq_ne_zero` for the one implication that does follow and
  `exists_batchCoupled_sigmaJacSq_eq_zero` for the counterexample to the converse. Neither does it
  disprove the clause: `not_hessianChain_gradWitness` records that the counterexample lies outside
  the paper's assumption (A4), so it refutes the implication from batch coupling alone and leaves
  the (A4) regime — where `Assumptions.ForwardKLChain.trace_leading_nonneg` signs the leading
  term — untouched.
* It does not prove that the `Θ` relation between `σ_Jac²` and any downstream modulus holds, on
  either route. `SigmaJacDominatedOn` is a hypothesis throughout, and
  `exists_sigmaJacDominatedOn_ne_zero` is a satisfiability witness, not an instance of the paper's
  claim.
* It does not prove that reverse-KL training in practice decouples the two channels; it proves that
  decoupling, once assumed, forces the population zero, and it quantifies the residual when
  decoupling is only approximate.
* It does not derive the corrected effective-diffusion decomposition itself, which belongs to
  `UpdateCovariance.lean`. `effectiveDiffusion_collapses_of_frozen_anchor` is a statement about the
  three scalars that decomposition produces, given that the two jitter-sourced ones vanish, and the
  vanishing is proved here from the frozen anchor.
* It says nothing about `μ_eff`, escape times, or the Kramers asymptotics beyond the purely
  algebraic `kramersExponent_eq_of_effectiveDiffusion_eq`; those live in `BarrierRescaling.lean`,
  `KramersExitTime.lean` and `FreeDiffusion.lean`.
-/

open MeasureTheory ProbabilityTheory Matrix Finset
open scoped ProbabilityTheory

namespace IcnnLift

/-! ### Batch-independent values of the latent weight, and the two deletions they drive -/

section Architecture

variable {d : ℕ} {Ω : Type*}

/-- A latent weight is **batch-independent** when its value does not depend on the conditioning batch.

This is the single modelling assumption behind every window-level structural zero in this file: the
classical consensus splitting's auxiliary variable is fixed across batches, the deterministic limit
of `rem:bcond` reads the body at a fixed code, and the two batch-independent cells of the four-cell
ablation have a latent iterate that no batch enters. In each case the architecture is modelled
as a map from batches to latent iterates that happens to be constant, so that the paper's
`δθ̃ ≡ 0` is *derived* from the architecture rather than assumed outright. -/
def BatchIndependent (e : Ω → Fin d → ℝ) : Prop := ∀ x y, e x = e y

/-- A batch-independent latent weight has zero resampling fluctuation: this is deletion (ii) of
Theorem 1 in the vocabulary the architectural corollaries use. -/
theorem bodyFluct_eq_zero_of_batchIndependent {e : Ω → Fin d → ℝ} (he : BatchIndependent e)
    (X : ℕ → Ω) (T t s : ℕ) (hT : T ≠ 0) : bodyFluct e X T t s = 0 :=
  bodyFluct_eq_zero_of_const he X T t s hT

/-- A batch-independent latent weight zeroes the cross-covariance estimator `eq:cross-cov` identically,
for every iteration `t` and every window length `T`. -/
theorem crossCovEstimator_eq_zero_of_batchIndependent {e : Ω → Fin d → ℝ}
    (he : BatchIndependent e) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct e X T t) dg = 0 :=
  crossCovEstimator_eq_zero_of_body_const he X dg T t hT

/-- Hence its slack-channel reading vanishes as well, whatever the slack Jacobian is. -/
theorem slackReading_eq_zero_of_batchIndependent (Jb : Matrix (Fin d) (Fin d) ℝ)
    {e : Ω → Fin d → ℝ} (he : BatchIndependent e) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ)
    (T t : ℕ) (hT : T ≠ 0) : slackReading Jb T t (bodyFluct e X T t) dg = 0 := by
  rw [slackReading, crossCovEstimator_eq_zero_of_batchIndependent he X dg T t hT, Matrix.mul_zero]

/-- The lift's own latent iterate `θ̃ = b + h_φ(X)` of `eq:lift`: an additive slack `b`,
constant across the batches of a window, plus a body `h` evaluated on the conditioning batch. -/
def liftIterate (b : Fin d → ℝ) (h : Ω → Fin d → ℝ) : Ω → Fin d → ℝ := fun x => b + h x

/-- **The slack drops out of `eq:lift-batch-fluct`.** The lift's latent weight and its body alone have
the same window-centred resampling fluctuation, because the slack is common to every step of the
window and cancels against the window mean. This is the paper's "the slack drops out, and the body
carries this variation on its own", derived rather than asserted, and it is why every structural
zero below is a statement about the body and not about the slack. -/
theorem bodyFluct_liftIterate (b : Fin d → ℝ) (h : Ω → Fin d → ℝ) (X : ℕ → Ω) (T t s : ℕ)
    (hT : T ≠ 0) : bodyFluct (liftIterate b h) X T t s = bodyFluct h X T t s := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  have hsum : ∑ s' ∈ range T, liftIterate b h (X (t - T + s'))
      = (T : ℝ) • b + ∑ s' ∈ range T, h (X (t - T + s')) := by
    simp only [liftIterate]
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, ← Nat.cast_smul_eq_nsmul ℝ]
  rw [bodyFluct, bodyFluct, hsum, smul_add, smul_smul, inv_mul_cancel₀ hTne, one_smul, liftIterate]
  abel

/-- **The `T ≠ 0` hypothesis of `bodyFluct_liftIterate` is load-bearing**, and the point is made
here as a theorem rather than as a remark in a docstring. At `T = 0` mathlib's junk-value
convention makes the window mean the empty sum scaled by `(0 : ℝ)⁻¹ = 0`, so `bodyFluct` degenerates
to the *uncentred* latent weight and the slack no longer cancels: the witness below has a latent weight with
slack `1` and body `0`, whose degenerate fluctuation is `1` while its body's is `0`. Every window
statement in this file therefore carries `T ≠ 0`, for the same reason and with the same convention
as `CrossCov.lean`. -/
theorem exists_bodyFluct_liftIterate_ne_of_empty_window :
    ∃ (b : Fin 1 → ℝ) (h : ℝ → Fin 1 → ℝ) (X : ℕ → ℝ) (t s : ℕ),
      bodyFluct (liftIterate b h) X 0 t s ≠ bodyFluct h X 0 t s := by
  refine ⟨fun _ => 1, fun _ _ => 0, fun _ => 0, 0, 0, ?_⟩
  intro hcon
  have hc := congrFun hcon 0
  norm_num [bodyFluct, liftIterate] at hc

/-- With the identity slack Jacobian that (A3) supplies, the slack-channel reading is the
cross-covariance estimator itself, with no further attenuating composition — the ICLR version's
`eq:cross-channel-lift`. -/
theorem slackReading_one (T t : ℕ) (dθ dg : ℕ → Fin d → ℝ) :
    slackReading (1 : Matrix (Fin d) (Fin d) ℝ) T t dθ dg = crossCovEstimator T t dθ dg := by
  rw [slackReading, Matrix.transpose_one, Matrix.one_mul]

end Architecture

/-! ### The ADMM / consensus-splitting reading -/

section Splitting

variable {d : ℕ} {Ω : Type*}

/-- The latent iterate of the **classical consensus splitting** of `app:admm-reading`, read in
the lift's coordinates: the auxiliary primal variable `z` of the augmented Lagrangian
`eq:lift-admm`, plus the slack `b` as its own additive offset.

The modelling assumption is in the type: `z` is a free optimization variable of the splitting, not
a function of the data, so the iterate it produces is a *constant* map on the conditioning batch.
That is the paper's "the classical splitting has no data-conditioned body — its auxiliary variable
is fixed across batches", and it is the only thing assumed about the splitting here. Nothing about
the alternating schedule, the multiplier `y`, or the penalty parameter `ρ` is used, because none of
it bears on the resampling fluctuation. -/
def splittingIterate (b z : Fin d → ℝ) : Ω → Fin d → ℝ := fun _ => b + z

/-- The splitting's auxiliary variable is fixed across batches. -/
theorem splittingIterate_batchIndependent (b z : Fin d → ℝ) :
    BatchIndependent (splittingIterate (Ω := Ω) b z) := fun _ _ => rfl

/-- The splitting's resampling fluctuation is identically zero, at every step of every window: this
is the fluctuation `eq:lift-batch-fluct` measures, and it is what the two zeros below are drawn
from. -/
theorem bodyFluct_splitting_eq_zero (b z : Fin d → ℝ) (X : ℕ → Ω) (T t s : ℕ) (hT : T ≠ 0) :
    bodyFluct (splittingIterate (Ω := Ω) b z) X T t s = 0 :=
  bodyFluct_eq_zero_of_batchIndependent (splittingIterate_batchIndependent b z) X T t s hT

/-- **The splitting's variance under resampling is zero**, in the paper's own words: the window
sample covariance of its latent iterate — the cross-covariance estimator of `eq:cross-cov`
with the iterate fluctuation in both slots — vanishes identically. -/
theorem splittingIterate_resamplingCovariance_eq_zero (b z : Fin d → ℝ) (X : ℕ → Ω) (T t : ℕ)
    (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct (splittingIterate (Ω := Ω) b z) X T t)
      (bodyFluct (splittingIterate (Ω := Ω) b z) X T t) = 0 :=
  crossCovEstimator_eq_zero_of_batchIndependent (splittingIterate_batchIndependent b z) X _ T t hT

/-- **The consensus splitting reads a structural zero.** Since the splitting's auxiliary variable is
fixed across batches, the cross-covariance estimator `eq:cross-cov` vanishes identically for it, for
every iteration `t` and every window length `T`, and for every gradient-fluctuation sequence
whatsoever. -/
theorem crossCovEstimator_splitting_eq_zero (b z : Fin d → ℝ) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ)
    (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct (splittingIterate (Ω := Ω) b z) X T t) dg = 0 :=
  crossCovEstimator_eq_zero_of_batchIndependent (splittingIterate_batchIndependent b z) X dg T t hT

/-- The splitting's slack-channel reading vanishes with it, whatever slack Jacobian the splitting is
credited with. In particular the splitting's slack offset `b`, which is genuinely present, buys it
nothing: the zero comes from the missing data-conditioned body alone. -/
theorem slackReading_splitting_eq_zero (Jb : Matrix (Fin d) (Fin d) ℝ) (b z : Fin d → ℝ)
    (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    slackReading Jb T t (bodyFluct (splittingIterate (Ω := Ω) b z) X T t) dg = 0 :=
  slackReading_eq_zero_of_batchIndependent Jb (splittingIterate_batchIndependent b z) X dg T t hT

end Splitting

/-! ### The deterministic limit of the latent weight, and the four-cell ablation -/

section FourCell

variable {d : ℕ} {Ω : Type*}

/-- The **deterministic limit of the latent weight** of `rem:bcond`: a learnable code `c` in place of the
batch summary. The pooling `h` and the slack `b` survive unchanged — the same body map, the same
additive slack — and only the body's argument is frozen. This is the reading of "a learnable code in
place of the batch summary" on which the pooling is retained and the code is what it pools; the
other reading, on which the code replaces the pooled summary outright, is `splittingIterate b c'`
with `c'` the learned vector, and it is covered by the splitting theorems above. Both are
batch-independent and both read the same structural zero, so the corollary does not turn on which
reading is intended. -/
def anchoredParameter (b : Fin d → ℝ) (h : Ω → Fin d → ℝ) (c : Ω) : Ω → Fin d → ℝ :=
  fun _ => b + h c

/-- The deterministic limit is literally the lift's own latent weight evaluated at the frozen code: the
construction keeps the lift's extra optimization variables, and changes only where the body is
read. -/
theorem anchoredParameter_eq_liftIterate_const (b : Fin d → ℝ) (h : Ω → Fin d → ℝ) (c : Ω) :
    anchoredParameter b h c = fun _ : Ω => liftIterate b h c := rfl

/-- Freezing the conditioning set to a code removes the body's batch dependence. -/
theorem anchoredParameter_batchIndependent (b : Fin d → ℝ) (h : Ω → Fin d → ℝ) (c : Ω) :
    BatchIndependent (anchoredParameter b h c) := fun _ _ => rfl

/-- **The deterministic latent weight reads a structural zero** (`rem:bcond`): the batch-induced
fluctuation of `eq:lift-batch-fluct` vanishes and takes the cross-covariance with it, for every
iteration and every window length. -/
theorem crossCovEstimator_anchored_eq_zero (b : Fin d → ℝ) (h : Ω → Fin d → ℝ) (c : Ω) (X : ℕ → Ω)
    (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct (anchoredParameter b h c) X T t) dg = 0 :=
  crossCovEstimator_eq_zero_of_batchIndependent (anchoredParameter_batchIndependent b h c) X dg T t hT

/-- Hence the deterministic latent weight forfeits the certificate that Theorem 1 attaches to the lift:
its slack-channel reading is identically zero even though its slack is intact. -/
theorem slackReading_anchored_eq_zero (Jb : Matrix (Fin d) (Fin d) ℝ) (b : Fin d → ℝ)
    (h : Ω → Fin d → ℝ) (c : Ω) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    slackReading Jb T t (bodyFluct (anchoredParameter b h c) X T t) dg = 0 :=
  slackReading_eq_zero_of_batchIndependent Jb (anchoredParameter_batchIndependent b h c) X dg T t hT

/-- Cell 1 of `fig:four-cell-cross-cov`: direct softplus `ψ(θ̃)`, a free latent weight with
neither slack nor body. Its zero is **by the theorem** — the estimator itself vanishes, because the
latent iterate carries no batch-induced fluctuation. -/
theorem crossCovEstimator_directSoftplus_eq_zero (w : Fin d → ℝ) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ)
    (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct (fun _ : Ω => w) X T t) dg = 0 :=
  crossCovEstimator_eq_zero_of_batchIndependent (fun _ _ => rfl) X dg T t hT

/-- Cell 2 of `fig:four-cell-cross-cov`: direct softplus with a bias, `ψ(b + θ̃)`. The slack is
present and its Jacobian is the identity, but the pre-activation carries no batch-induced variance,
so this cell's zero is again **by the theorem**: the estimator vanishes before any slack Jacobian is
applied. PGD, having neither slack nor data-conditioned body, is covered by the same
batch-independence argument, in the form `crossCovEstimator_directSoftplus_eq_zero`. -/
theorem crossCovEstimator_biasOnly_eq_zero (b w : Fin d → ℝ) (X : ℕ → Ω) (dg : ℕ → Fin d → ℝ)
    (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct (splittingIterate (Ω := Ω) b w) X T t) dg = 0 :=
  crossCovEstimator_splitting_eq_zero b w X dg T t hT

/-- Cell 3 of `fig:four-cell-cross-cov`: hypernet without bias, `ψ(h_φ(X))`. Its zero is
**definitional, not measured**: with no slack channel there is nothing to contract the estimator
against, and the reading is zero for *every* pair of fluctuation sequences, with no hypothesis on
the data, on the body, on the window, or on the batch. Contrast
`crossCovEstimator_directSoftplus_eq_zero` and `crossCovEstimator_biasOnly_eq_zero`, whose zeros
are conclusions about the estimator drawn from a hypothesis about the architecture's body. -/
theorem slackReading_bodyOnly_eq_zero (T t : ℕ) :
    ∀ dθ dg : ℕ → Fin d → ℝ, slackReading (0 : Matrix (Fin d) (Fin d) ℝ) T t dθ dg = 0 :=
  fun dθ dg => slackReading_eq_zero_of_jacobian_eq_zero rfl T t dθ dg

/-- **The two kinds of zero are genuinely different, and the distinction survives formalization.**
There is a body-only configuration whose cross-covariance estimator is nonzero — so its body does
depend on the batch and its gradient fluctuation does not vanish, since either failure would zero
the estimator — while its slack-channel reading is zero. No batch-independent cell can do this,
since for those the estimator itself vanishes. This is
the paper's "the body-only cell has no slack channel to contract against, so its zero is
definitional rather than measured, and the honest measurement for that cell, the body-latent-weight
cross-covariance, is a separate quantity". -/
theorem exists_bodyOnly_estimator_ne_zero_of_no_slack :
    ∃ (h : ℝ → Fin 1 → ℝ) (X : ℕ → ℝ) (dg : ℕ → Fin 1 → ℝ) (T t : ℕ),
      T ≠ 0 ∧ crossCovEstimator T t (bodyFluct h X T t) dg ≠ 0 ∧
        slackReading (0 : Matrix (Fin 1) (Fin 1) ℝ) T t (bodyFluct h X T t) dg = 0 := by
  refine ⟨fun x _ => x, fun n => (n : ℝ), fun s _ => if s = 1 then (1 : ℝ) else -1, 2, 2,
    two_ne_zero, ?_, slackReading_bodyOnly_eq_zero 2 2 _ _⟩
  intro hzero
  have h00 : crossCovEstimator 2 2
      (bodyFluct (fun (x : ℝ) (_ : Fin 1) => x) (fun n : ℕ => (n : ℝ)) 2 2)
      (fun (s : ℕ) (_ : Fin 1) => if s = 1 then (1 : ℝ) else -1) 0 0 = 0 := by
    rw [hzero]; rfl
  norm_num [crossCovEstimator, bodyFluct, Matrix.vecMulVec_apply, Finset.sum_range_succ] at h00

/-- **The converse of Theorem 1 is unavailable, and this is a theorem and not an omission.** There
is a configuration carrying all three ingredients — an identity slack Jacobian, a genuinely
batch-dependent body with a nonzero fluctuation, and a nonzero gradient fluctuation — whose
slack-channel reading is nevertheless zero. Theorem 1 is a necessity statement about the reading,
the paper says so explicitly, and nothing in this file upgrades it. -/
theorem exists_fullLift_slackReading_eq_zero :
    ∃ (h : ℝ → Fin 1 → ℝ) (X : ℕ → ℝ) (dg : ℕ → Fin 1 → ℝ) (T t : ℕ),
      T ≠ 0 ∧ (∃ s, bodyFluct h X T t s ≠ 0) ∧ (∃ s, dg s ≠ 0) ∧
        crossCovEstimator T t (bodyFluct h X T t) dg = 0 ∧
        slackReading (1 : Matrix (Fin 1) (Fin 1) ℝ) T t (bodyFluct h X T t) dg = 0 := by
  refine ⟨fun x _ => x, fun n => (n : ℝ), fun _ _ => (1 : ℝ), 2, 2, two_ne_zero, ⟨0, ?_⟩,
    ⟨0, ?_⟩, ?_, ?_⟩
  · intro hz
    have := congrFun hz 0
    norm_num [bodyFluct, Finset.sum_range_succ] at this
  · intro hz
    have := congrFun hz 0
    norm_num at this
  · ext i j
    fin_cases i
    fin_cases j
    norm_num [crossCovEstimator, bodyFluct, Matrix.vecMulVec_apply, Finset.sum_range_succ]
  · rw [slackReading_one]
    ext i j
    fin_cases i
    fin_cases j
    norm_num [crossCovEstimator, bodyFluct, Matrix.vecMulVec_apply, Finset.sum_range_succ]

end FourCell

/-! ### Forward versus reverse KL -/

section KL

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The **population** slack-channel reading `J_bᵀ Σ_slack`, the object of which the estimator's
reading is the sample version. -/
noncomputable def popSlackReading (Jb : Matrix (Fin d) (Fin d) ℝ) (μ : Measure Ω)
    (X Y : Ω → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ := Jbᵀ * popCrossCov μ X Y

/-- A vanishing population cross-covariance zeroes the population reading, for every slack
Jacobian. -/
theorem popSlackReading_eq_zero_of_popCrossCov_eq_zero (Jb : Matrix (Fin d) (Fin d) ℝ)
    {X Y : Ω → Fin d → ℝ} (h : popCrossCov μ X Y = 0) : popSlackReading Jb μ X Y = 0 := by
  rw [popSlackReading, h, Matrix.mul_zero]

/-- **Reverse KL, unconditional form.** Under reverse KL the latent weight and the loss gradient are
driven by different draws, and the formal content of that decoupling is independence of the two
random vectors. This is case (iii) of Theorem 1, imported from `CrossCovIndep.lean`
(`popCrossCov_centered_eq_zero_of_indepFun`): the population cross-covariance of the centred
fluctuations vanishes.

The hypothesis is a modelling assumption about how reverse-KL training is constructed, not a
derived fact; see `sigmaJacSq_eq_remainder_of_indepFun` for what survives when the decoupling is
only approximate, which is what the paper's `σ_Jac² ≈ 0` actually reports. -/
theorem reverseKL_popCrossCov_eq_zero [IsProbabilityMeasure μ] {θtil g : Ω → Fin d → ℝ}
    (hdec : IndepFun θtil g μ) (hθ : Integrable θtil μ) (hg : Integrable g μ) :
    popCrossCov μ (fun ω => θtil ω - popMean μ θtil) (fun ω => g ω - popMean μ g) = 0 :=
  popCrossCov_centered_eq_zero_of_indepFun hdec hθ hg

/-- Reverse KL: `σ_Jac² = tr Σ_slack = 0`, so the smoothing channel is inactive. -/
theorem reverseKL_sigmaJacSq_eq_zero [IsProbabilityMeasure μ] {θtil g : Ω → Fin d → ℝ}
    (hdec : IndepFun θtil g μ) (hθ : Integrable θtil μ) (hg : Integrable g μ) :
    sigmaJacSq μ (fun ω => θtil ω - popMean μ θtil) (fun ω => g ω - popMean μ g) = 0 :=
  sigmaJacSq_centered_eq_zero_of_indepFun hdec hθ hg

/-- Reverse KL: the population slack-channel reading vanishes as well. -/
theorem reverseKL_popSlackReading_eq_zero [IsProbabilityMeasure μ]
    (Jb : Matrix (Fin d) (Fin d) ℝ) {θtil g : Ω → Fin d → ℝ}
    (hdec : IndepFun θtil g μ) (hθ : Integrable θtil μ) (hg : Integrable g μ) :
    popSlackReading Jb μ (fun ω => θtil ω - popMean μ θtil) (fun ω => g ω - popMean μ g) = 0 :=
  popSlackReading_eq_zero_of_popCrossCov_eq_zero Jb (reverseKL_popCrossCov_eq_zero hdec hθ hg)

/-- **The reverse-KL hypothesis stack is satisfiable without degeneracy**, so the three theorems
above are not vacuously true and their zero is not obtained by switching off one of the objects it
is about. On two independent fair coins — the witness `coinPair` of `CrossCovIndep.lean` — the
latent weight and the loss gradient are independent, integrable, and each takes two different values, and
the slack Jacobian is the identity that (A3) supplies rather than the zero matrix of deletion (i);
the population slack-channel reading is nevertheless zero. Read together with
`CrossCovIndep.exists_centered_popCrossCov_ne_zero`, which exhibits a *non-vanishing* centred
population cross-covariance, this locates the zero where the paper locates it: in the independence
of the two channels, not in the definitions. -/
theorem exists_indepFun_popSlackReading_eq_zero (d : ℕ) (hd : 0 < d) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (θtil g : Ω → Fin d → ℝ),
      IsProbabilityMeasure μ ∧ Integrable θtil μ ∧ Integrable g μ ∧
        IndepFun θtil g μ ∧ (∃ ω ω', θtil ω ≠ θtil ω') ∧ (∃ ω ω', g ω ≠ g ω') ∧
        popSlackReading (1 : Matrix (Fin d) (Fin d) ℝ) μ
          (fun ω => θtil ω - popMean μ θtil) (fun ω => g ω - popMean μ g) = 0 := by
  refine ⟨Bool × Bool, inferInstance, coinPair, (fun ω => signVec d ω.1),
    (fun ω => signVec d ω.2), inferInstance, Integrable.of_finite, Integrable.of_finite,
    indepFun_coinPair, ⟨(true, true), (false, true), signVec_true_ne_signVec_false hd⟩,
    ⟨(true, true), (true, false), signVec_true_ne_signVec_false hd⟩, ?_⟩
  exact reverseKL_popSlackReading_eq_zero 1 indepFun_coinPair Integrable.of_finite
    Integrable.of_finite

/-- **What the forward-KL side of `rem:moreau-scope` licenses, and all it licenses.** A nonzero
`σ_Jac²` certifies that the latent weight and the loss gradient are *not* independent. That is strictly
less than the paper's "driven by the same target batch": common batch driving implies
non-independence, and non-independence does not imply common batch driving, so what the theorem
rules out is the fully decoupled sampling of reverse KL and nothing finer. The paper's converse
clause, that forward KL makes `σ_Jac²` "generically nonzero", is an **empirical** claim supported by
the four-cell ablation and the training-time traces; it is not a theorem, it is not proved here, and
`exists_batchCoupled_sigmaJacSq_eq_zero` shows that it cannot be proved from batch coupling alone.
-/
theorem not_indepFun_of_sigmaJacSq_ne_zero [IsProbabilityMeasure μ] {θtil g : Ω → Fin d → ℝ}
    (hθ : Integrable θtil μ) (hg : Integrable g μ)
    (hne : sigmaJacSq μ (fun ω => θtil ω - popMean μ θtil) (fun ω => g ω - popMean μ g) ≠ 0) :
    ¬ IndepFun θtil g μ := fun hdec => hne (reverseKL_sigmaJacSq_eq_zero hdec hθ hg)

/-- The population cross-covariance is additive in its second argument, given entrywise
integrability of the two products. This is the bookkeeping that lets the residual coupling of an
approximately decoupled pipeline be carried explicitly instead of assumed away. -/
theorem popCrossCov_add_right {X Y R : Ω → Fin d → ℝ}
    (hY : ∀ i j, Integrable (fun ω => X ω i * Y ω j) μ)
    (hR : ∀ i j, Integrable (fun ω => X ω i * R ω j) μ) :
    popCrossCov μ X (fun ω => Y ω + R ω) = popCrossCov μ X Y + popCrossCov μ X R := by
  ext i j
  simp only [popCrossCov, Matrix.of_apply, Matrix.add_apply, Pi.add_apply, mul_add]
  exact integral_add (hY i j) (hR i j)

/-- The population cross-covariance against a vanishing second factor is zero. Together with
`popCrossCov_eq_remainder_of_indepFun` this says that exact decoupling is the `r = 0` case of the
remainder identity, so the two statements are one statement with the remainder switched off. -/
theorem popCrossCov_zero_right (X : Ω → Fin d → ℝ) :
    popCrossCov μ X (fun _ => (0 : Fin d → ℝ)) = 0 := by
  ext i j
  simp [popCrossCov]

/-- **Approximate decoupling, with the remainder carried explicitly.** The paper writes
`σ_Jac² ≈ 0` for reverse KL rather than `= 0`, because a real reverse-KL pipeline decouples the two
channels only approximately. Rather than assume the approximation away, decompose the gradient
fluctuation as `δg = δg₀ + r` with `δg₀` independent of the centred iterate fluctuation and `r` the
residual coupling. Then the population cross-covariance is *exactly* the cross-covariance with the
remainder alone:
`E[δθ̃ δgᵀ] = E[δθ̃ rᵀ]`.
Exact decoupling is the case `r = 0`, where `popCrossCov_zero_right` returns the population zero. -/
theorem popCrossCov_eq_remainder_of_indepFun {u v₀ r : Ω → Fin d → ℝ}
    (hindep : IndepFun u v₀ μ) (hu : Integrable u μ) (hv₀ : Integrable v₀ μ)
    (hu0 : popMean μ u = 0)
    (hR : ∀ i j, Integrable (fun ω => u ω i * r ω j) μ) :
    popCrossCov μ u (fun ω => v₀ ω + r ω) = popCrossCov μ u r := by
  have hbil : Integrable (fun ω => outerProd d (u ω) (v₀ ω)) μ :=
    hindep.integrable_bilin hu hv₀ (outerProd d)
  have hY : ∀ i j, Integrable (fun ω => u ω i * v₀ ω j) μ := fun i j => by
    simpa using ((hbil.eval i).eval j)
  rw [popCrossCov_add_right hY hR,
    popCrossCov_eq_zero_of_indepFun_of_centered hindep hu hv₀ hu0, zero_add]

/-- The same statement on the trace: under approximate decoupling the surviving `σ_Jac²` is exactly
the trace of the cross-covariance of the iterate fluctuation with the coupling remainder. -/
theorem sigmaJacSq_eq_remainder_of_indepFun {u v₀ r : Ω → Fin d → ℝ}
    (hindep : IndepFun u v₀ μ) (hu : Integrable u μ) (hv₀ : Integrable v₀ μ)
    (hu0 : popMean μ u = 0)
    (hR : ∀ i j, Integrable (fun ω => u ω i * r ω j) μ) :
    sigmaJacSq μ u (fun ω => v₀ ω + r ω) = sigmaJacSq μ u r := by
  rw [sigmaJacSq, sigmaJacSq, popCrossCov_eq_remainder_of_indepFun hindep hu hv₀ hu0 hR]

/-- **A bound on the surviving channel, so that `σ_Jac² ≈ 0` is a quantitative statement.** The
identity above says what the surviving channel *is*; it does not by itself say that the channel is
small, and the paper's `≈` is a smallness claim. If the centred iterate fluctuation and the coupling
remainder are bounded entrywise by `Mu` and `Mr` almost surely, then
`|σ_Jac²| ≤ d · Mu · Mr`,
so the trace is controlled by the product of the fluctuation scale and the remainder scale, and it
tends to zero with the remainder at any fixed fluctuation scale. No integrability is assumed: on a
probability space an entrywise almost-sure bound is enough. -/
theorem abs_sigmaJacSq_le_of_bounded [IsProbabilityMeasure μ] {u r : Ω → Fin d → ℝ} {Mu Mr : ℝ}
    (hu : ∀ᵐ ω ∂μ, ∀ i, |u ω i| ≤ Mu) (hr : ∀ᵐ ω ∂μ, ∀ i, |r ω i| ≤ Mr) :
    |sigmaJacSq μ u r| ≤ d * (Mu * Mr) := by
  have hentry : ∀ i : Fin d, |∫ ω, u ω i * r ω i ∂μ| ≤ Mu * Mr := by
    intro i
    have hbd : ∀ᵐ ω ∂μ, ‖u ω i * r ω i‖ ≤ Mu * Mr := by
      filter_upwards [hu, hr] with ω h1 h2
      have h0 : (0 : ℝ) ≤ Mu := le_trans (abs_nonneg _) (h1 i)
      calc ‖u ω i * r ω i‖ = |u ω i| * |r ω i| := by rw [Real.norm_eq_abs, abs_mul]
        _ ≤ Mu * Mr := mul_le_mul (h1 i) (h2 i) (abs_nonneg _) h0
    simpa using norm_integral_le_of_norm_le_const hbd
  calc |sigmaJacSq μ u r| = |∑ i, ∫ ω, u ω i * r ω i ∂μ| := by rw [sigmaJacSq_eq_sum]
    _ ≤ ∑ i : Fin d, |∫ ω, u ω i * r ω i ∂μ| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin d, Mu * Mr := Finset.sum_le_sum fun i _ => hentry i
    _ = d * (Mu * Mr) := by simp [Finset.sum_const, nsmul_eq_mul]

/-- **The paper's `σ_Jac² ≈ 0` for reverse KL, with both halves supplied.** Decompose the gradient
fluctuation as `δg = δg₀ + r` with `δg₀` independent of the centred iterate fluctuation, and bound
the fluctuation and the coupling remainder entrywise. Then the whole slack-channel trace is bounded
by `d · Mu · Mr`: the independent part contributes exactly nothing, and what survives is controlled
by the size of the residual coupling alone. Exact decoupling is `Mr = 0`, where the bound returns
the exact zero of `reverseKL_sigmaJacSq_eq_zero`. -/
theorem abs_sigmaJacSq_le_of_indepFun_of_remainder_bounded [IsProbabilityMeasure μ]
    {u v₀ r : Ω → Fin d → ℝ} {Mu Mr : ℝ}
    (hindep : IndepFun u v₀ μ) (hu : Integrable u μ) (hv₀ : Integrable v₀ μ)
    (hu0 : popMean μ u = 0) (hR : ∀ i j, Integrable (fun ω => u ω i * r ω j) μ)
    (hub : ∀ᵐ ω ∂μ, ∀ i, |u ω i| ≤ Mu) (hrb : ∀ᵐ ω ∂μ, ∀ i, |r ω i| ≤ Mr) :
    |sigmaJacSq μ u (fun ω => v₀ ω + r ω)| ≤ d * (Mu * Mr) := by
  rw [sigmaJacSq_eq_remainder_of_indepFun hindep hu hv₀ hu0 hR]
  exact abs_sigmaJacSq_le_of_bounded hub hrb

end KL

/-! ### The converse of the forward-KL clause fails

The paper's forward-KL clause in `rem:moreau-scope` — that when the latent weight and the loss gradient
ride the same target batch, `σ_Jac²` is "generically nonzero" — is an empirical claim about trained
networks, not a theorem, and this section shows that it cannot be upgraded into one *from batch
coupling alone*. Batch coupling, however strong, does not force a nonzero cross-covariance: the
witness below has the loss-gradient fluctuation as a *deterministic function* of the iterate
fluctuation, which is as coupled as two random vectors can be, and its cross-covariance is exactly
zero.

The scope of that counterexample is stated precisely, because overstating it would be its own kind
of dishonesty. The paper does not assert the forward-KL clause in a vacuum: assumption (A4) of
`app:assumptions` posits that on the operative region the gradient fluctuation is the mean loss
Hessian acting on the iterate fluctuation plus a remainder of order `‖δθ̃‖²`, and under (A4) the
leading term of `σ_Jac²` is `tr(V H)` with `V` and `H` positive semidefinite, hence non-negative —
that is `Assumptions.ForwardKLChain` and `Assumptions.ForwardKLChain.trace_leading_nonneg`, and
nothing here contradicts it. What `not_hessianChain_gradWitness` establishes is that the witness
below is *outside* (A4): its deterministic coupling admits no such representation, for any Hessian
and any remainder obeying the quadratic budget. So the counterexample refutes the implication "the
two channels ride the same draw, therefore `σ_Jac² ≠ 0`", and it refutes nothing about what follows
once (A4) is in force. Neither statement is the paper's clause, which quantifies over trained
networks and is measured rather than proved.

The witness is carried on three atoms because two would not do — on a two-point space a centred pair
with vanishing cross-covariance has a constant factor, and is then independent — but that minimality
remark is motivation and is not itself formalized here. -/

section ForwardKLConverse

/-- The uniform probability measure on three atoms, the carrier of the witness below. -/
noncomputable def threePointUniform : Measure (Fin 3) :=
  (3 : ENNReal)⁻¹ • (Measure.dirac 0 + Measure.dirac 1 + Measure.dirac 2)

instance : IsProbabilityMeasure threePointUniform := by
  constructor
  simp only [threePointUniform, Measure.smul_apply, Measure.add_apply, smul_eq_mul]
  norm_num
  exact ENNReal.inv_mul_cancel (by norm_num) (by norm_num)

/-- Integration against the three-point uniform measure is the arithmetic mean of the three
values. -/
theorem integral_threePointUniform {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] (f : Fin 3 → E) :
    ∫ x, f x ∂threePointUniform = (3 : ℝ)⁻¹ • (f 0 + f 1 + f 2) := by
  rw [threePointUniform, integral_smul_measure,
    integral_add_measure (Integrable.of_finite) (Integrable.of_finite),
    integral_add_measure (Integrable.of_finite) (Integrable.of_finite),
    integral_dirac, integral_dirac, integral_dirac]
  simp [add_assoc]

/-- The iterate-fluctuation witness `δθ̃`, taking the three distinct values `1`, `-1`, `0` on the
three atoms. Because the three values are distinct, every random variable on this space is a
deterministic function of it; the theorem below exhibits the function for `gradWitness`. -/
noncomputable def fluctWitness : Fin 3 → Fin 1 → ℝ := fun ω _ => ![(1 : ℝ), -1, 0] ω

/-- The gradient-fluctuation witness `δg`, taking the three values `1`, `1`, `-2`. -/
noncomputable def gradWitness : Fin 3 → Fin 1 → ℝ := fun ω _ => ![(1 : ℝ), 1, -2] ω

/-- **Batch coupling does not force a nonzero reading.** The two fluctuations below are integrable
and centred, the gradient fluctuation is a deterministic function of the iterate fluctuation — so
the two are driven by the same draw in the strongest sense available — and they are not independent,
yet `σ_Jac² = tr E[δθ̃ δgᵀ] = 0` exactly.

Consequently the paper's forward-KL clause of `rem:moreau-scope` is not a theorem and is not proved
anywhere in this development: `not_indepFun_of_sigmaJacSq_ne_zero` gives the one implication that
does follow, and this witness shows that its converse fails as a general implication from batch
coupling alone. The clause remains what the paper's evidence makes it, an empirical regularity of
trained networks measured by the four-cell ablation and the training-time traces. The witness does
not satisfy the paper's assumption (A4), and `not_hessianChain_gradWitness` below proves that it
cannot be made to; the counterexample is therefore about batch coupling in general and not about the
regime (A4) describes. -/
theorem exists_batchCoupled_sigmaJacSq_eq_zero :
    ∃ u v : Fin 3 → Fin 1 → ℝ,
      Integrable u threePointUniform ∧ Integrable v threePointUniform ∧
        popMean threePointUniform u = 0 ∧ popMean threePointUniform v = 0 ∧
        (∃ q : ℝ → Fin 1 → ℝ, v = fun ω => q (u ω 0)) ∧
        ¬ IndepFun u v threePointUniform ∧
        sigmaJacSq threePointUniform u v = 0 := by
  refine ⟨fluctWitness, gradWitness, Integrable.of_finite, Integrable.of_finite, ?_, ?_,
    ⟨fun x => if x = 0 then ![(-2 : ℝ)] else ![(1 : ℝ)], ?_⟩, ?_, ?_⟩
  · rw [popMean, integral_threePointUniform]
    funext i
    simp [fluctWitness]
  · rw [popMean, integral_threePointUniform]
    funext i
    simp [gradWitness]
    norm_num
  · funext ω i
    fin_cases ω <;> fin_cases i <;> norm_num [fluctWitness, gradWitness]
  · intro hind
    have h2 : IndepFun (fun ω => (fluctWitness ω 0) ^ 2) (fun ω => gradWitness ω 0)
        threePointUniform :=
      hind.comp (φ := fun w : Fin 1 → ℝ => (w 0) ^ 2) (ψ := fun w : Fin 1 → ℝ => w 0)
        (by fun_prop) (by fun_prop)
    have hfac := h2.integral_mul_eq_mul_integral (by fun_prop) (by fun_prop)
    simp only [Pi.mul_apply] at hfac
    rw [integral_threePointUniform, integral_threePointUniform, integral_threePointUniform] at hfac
    simp [fluctWitness, gradWitness] at hfac
    norm_num at hfac
  · rw [sigmaJacSq_eq_sum, Fin.sum_univ_one, integral_threePointUniform]
    simp [fluctWitness, gradWitness]

/-- **The witness lies outside (A4), and this bounds exactly what the counterexample refutes.**
Assumption (A4) of `app:assumptions` writes the forward-KL gradient fluctuation as the mean loss
Hessian acting on the iterate fluctuation plus a remainder of order `‖δθ̃‖²`,
`δg = H δθ̃ + r` with `‖r(ω)‖ ≤ C_r ‖δθ̃(ω)‖²`.
No such representation exists for the pair above, for *any* Hessian `H`, *any* remainder field `r`
obeying that quadratic budget entrywise, and *any* constant `C_r`: on the third atom the iterate
fluctuation vanishes, so the budget forces the remainder to vanish there as well and (A4) predicts a
vanishing gradient fluctuation, whereas the witness's is `-2`.

The reason to record this is scope. `exists_batchCoupled_sigmaJacSq_eq_zero` refutes the implication
"the two channels ride the same draw, therefore `σ_Jac² ≠ 0`". It does not refute, and this file
does not investigate, what follows under (A4), where `Assumptions.ForwardKLChain.trace_leading_nonneg`
signs the leading term. Both statements leave the paper's clause where the paper leaves it: an
empirical regularity, evidenced by the four-cell ablation and the training-time traces. -/
theorem not_hessianChain_gradWitness (H : Matrix (Fin 1) (Fin 1) ℝ) (rem : Fin 3 → Fin 1 → ℝ)
    (Cr : ℝ) (hchain : ∀ ω, gradWitness ω = H *ᵥ fluctWitness ω + rem ω)
    (hrem : ∀ ω i, |rem ω i| ≤ Cr * ∑ j, (fluctWitness ω j) ^ 2) : False := by
  have hfl : fluctWitness 2 = 0 := rfl
  have hsum : ∑ j, (fluctWitness 2 j) ^ 2 = 0 := by simp [hfl]
  have hr : rem 2 0 = 0 := by
    have hb := hrem 2 0
    rw [hsum, mul_zero] at hb
    exact abs_nonpos_iff.mp hb
  have hcomp := congrFun (hchain 2) 0
  simp only [hfl, Matrix.mulVec_zero, Pi.add_apply, Pi.zero_apply, zero_add, hr] at hcomp
  have hval : (-2 : ℝ) = 0 := hcomp
  norm_num at hval

end ForwardKLConverse

/-! ### Reverse KL, conditional on the body parameters -/

section ConditionalKL

variable {d : ℕ} {Ω : Type*} {m' : MeasurableSpace Ω} [mΩ : MeasurableSpace Ω]
  [StandardBorelSpace Ω] {μ : Measure Ω} [IsFiniteMeasure μ]

/-- **Reverse KL, in the conditional form Theorem 1 states.** Conditioning on the body parameters
`φ` is modelled by a sub-σ-algebra `m'` and the disintegration kernel `condExpKernel μ m'`, exactly
as in `CrossCovIndep.lean`. If reverse-KL training makes the latent weight and the loss gradient
conditionally independent given `φ`, then `Σ_slack = E[δθ̃ δgᵀ | φ] = 0` almost surely. -/
theorem reverseKL_popCrossCov_ae_eq_zero (hm' : m' ≤ mΩ) {θtil g : Ω → Fin d → ℝ}
    (hθ : Measurable θtil) (hg : Measurable g) (hθi : Integrable θtil μ) (hgi : Integrable g μ)
    (hdec : CondIndepFun m' hm' θtil g μ) :
    ∀ᵐ ω ∂μ, popCrossCov (condExpKernel μ m' ω)
        (fun x => θtil x - popMean (condExpKernel μ m' ω) θtil)
        (fun x => g x - popMean (condExpKernel μ m' ω) g) = 0 :=
  condPopCrossCov_centered_ae_eq_zero hm' hθ hg hθi hgi hdec

/-- Hence `σ_Jac² = 0` almost surely under reverse KL, and the smoothing channel is inactive. -/
theorem reverseKL_sigmaJacSq_ae_eq_zero (hm' : m' ≤ mΩ) {θtil g : Ω → Fin d → ℝ}
    (hθ : Measurable θtil) (hg : Measurable g) (hθi : Integrable θtil μ) (hgi : Integrable g μ)
    (hdec : CondIndepFun m' hm' θtil g μ) :
    ∀ᵐ ω ∂μ, sigmaJacSq (condExpKernel μ m' ω)
        (fun x => θtil x - popMean (condExpKernel μ m' ω) θtil)
        (fun x => g x - popMean (condExpKernel μ m' ω) g) = 0 :=
  condSigmaJacSq_centered_ae_eq_zero hm' hθ hg hθi hgi hdec

/-- And the conditional population slack-channel reading vanishes almost surely. -/
theorem reverseKL_popSlackReading_ae_eq_zero (Jb : Matrix (Fin d) (Fin d) ℝ) (hm' : m' ≤ mΩ)
    {θtil g : Ω → Fin d → ℝ} (hθ : Measurable θtil) (hg : Measurable g)
    (hθi : Integrable θtil μ) (hgi : Integrable g μ) (hdec : CondIndepFun m' hm' θtil g μ) :
    ∀ᵐ ω ∂μ, popSlackReading Jb (condExpKernel μ m' ω)
        (fun x => θtil x - popMean (condExpKernel μ m' ω) θtil)
        (fun x => g x - popMean (condExpKernel μ m' ω) g) = 0 := by
  filter_upwards [reverseKL_popCrossCov_ae_eq_zero hm' hθ hg hθi hgi hdec] with ω hω
  exact popSlackReading_eq_zero_of_popCrossCov_eq_zero Jb hω

end ConditionalKL

/-! ### The frozen anchor: the paper's own falsifiability boundary -/

section FrozenAnchor

variable {d : ℕ} {Ω : Type*}

/-- **A frozen conditioning batch silences even a genuinely batch-dependent body**, and the
freezing is only ever needed on the window the estimator reads. This is a different hypothesis from
the batch-independent-body deletion: the body `h` here is arbitrary, and it is the *batch sequence*
that an external constraint has frozen. The window mean then coincides with every window value and
the fluctuation of `eq:lift-batch-fluct` vanishes. Stating the hypothesis on the trailing window
rather than on all of time is the sharper statement, and it is what makes the frozen anchor a
property of the measurement window rather than of the whole trajectory;
`bodyFluct_eq_zero_of_frozen_batch` below is the globally frozen case the paper states. -/
theorem bodyFluct_eq_zero_of_frozen_window {h : Ω → Fin d → ℝ} {X : ℕ → Ω} {T t s : ℕ}
    (hT : T ≠ 0) (hs : ∀ s' ∈ range T, X (t - T + s') = X s) : bodyFluct h X T t s = 0 := by
  have hmean : ∑ s' ∈ range T, h (X (t - T + s')) = (T : ℝ) • h (X s) := by
    rw [Finset.sum_congr rfl (fun s' hs' => by rw [hs s' hs']), Finset.sum_const,
      Finset.card_range, ← Nat.cast_smul_eq_nsmul ℝ]
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT
  ext i
  simp [bodyFluct, hmean, smul_smul, inv_mul_cancel₀ hTne]

/-- Hence the estimator `eq:cross-cov` at iteration `t` vanishes as soon as the batch is frozen
across that one trailing window, whatever the body and whatever the gradient fluctuations are, and
with no constraint on the batches outside the window. -/
theorem crossCovEstimator_eq_zero_of_frozen_window {h : Ω → Fin d → ℝ} {X : ℕ → Ω} {T t : ℕ}
    (hT : T ≠ 0) (hX : ∀ s' ∈ range T, X (t - T + s') = X (t - T)) (dg : ℕ → Fin d → ℝ) :
    crossCovEstimator T t (bodyFluct h X T t) dg = 0 := by
  have hz : ∀ s ∈ range T,
      Matrix.vecMulVec (bodyFluct h X T t (t - T + s)) (dg (t - T + s))
        = (0 : Matrix (Fin d) (Fin d) ℝ) := by
    intro s hs
    have hb : bodyFluct h X T t (t - T + s) = 0 :=
      bodyFluct_eq_zero_of_frozen_window hT (fun s' hs' => by rw [hX s' hs', ← hX s hs])
    rw [hb]
    ext i j
    simp [Matrix.vecMulVec_apply]
  simp [crossCovEstimator, Finset.sum_congr rfl hz]

/-- And so does its slack-channel reading, however large the slack Jacobian. -/
theorem slackReading_eq_zero_of_frozen_window (Jb : Matrix (Fin d) (Fin d) ℝ) {h : Ω → Fin d → ℝ}
    {X : ℕ → Ω} {T t : ℕ} (hT : T ≠ 0) (hX : ∀ s' ∈ range T, X (t - T + s') = X (t - T))
    (dg : ℕ → Fin d → ℝ) : slackReading Jb T t (bodyFluct h X T t) dg = 0 := by
  rw [slackReading, crossCovEstimator_eq_zero_of_frozen_window hT hX dg, Matrix.mul_zero]

/-- The paper's own form of the falsifiability hypothesis: an external constraint freezes the
conditioning set for the whole run, and the fluctuation vanishes at every step of every window. It
is the globally frozen case of `bodyFluct_eq_zero_of_frozen_window`. -/
theorem bodyFluct_eq_zero_of_frozen_batch {h : Ω → Fin d → ℝ} {X : ℕ → Ω}
    (hX : ∀ s s', X s = X s') (T t s : ℕ) (hT : T ≠ 0) : bodyFluct h X T t s = 0 :=
  bodyFluct_eq_zero_of_frozen_window hT (fun _ _ => hX _ _)

/-- Hence the cross-covariance estimator vanishes identically under a frozen anchor: the mechanism
falls silent, at the window level, until batch stochasticity is restored. -/
theorem crossCovEstimator_eq_zero_of_frozen_batch {h : Ω → Fin d → ℝ} {X : ℕ → Ω}
    (hX : ∀ s s', X s = X s') (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    crossCovEstimator T t (bodyFluct h X T t) dg = 0 :=
  crossCovEstimator_eq_zero_of_fluct_eq_zero
    (fun s => bodyFluct_eq_zero_of_frozen_batch hX T t s hT) T t

/-- And so does the slack-channel reading, however large the slack Jacobian. -/
theorem slackReading_eq_zero_of_frozen_batch (Jb : Matrix (Fin d) (Fin d) ℝ) {h : Ω → Fin d → ℝ}
    {X : ℕ → Ω} (hX : ∀ s s', X s = X s') (dg : ℕ → Fin d → ℝ) (T t : ℕ) (hT : T ≠ 0) :
    slackReading Jb T t (bodyFluct h X T t) dg = 0 := by
  rw [slackReading, crossCovEstimator_eq_zero_of_frozen_batch hX dg T t hT, Matrix.mul_zero]

end FrozenAnchor

section FrozenAnchorPopulation

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **A deterministic latent weight zeroes the population cross-covariance.** If the latent iterate
takes a single value `c` almost surely — the frozen-anchor limit at the population level — then its
centred fluctuation vanishes almost surely and `Σ_slack = E[δθ̃ Yᵀ] = 0` for *every* second factor
`Y`, in particular for the gradient fluctuation and for the iterate fluctuation itself. The
hypothesis is stated almost everywhere rather than pointwise, which is weaker and is what a
population statement should ask for; a genuinely frozen anchor gives it pointwise. -/
theorem popCrossCov_centered_eq_zero_of_deterministic [IsProbabilityMeasure μ]
    {θtil : Ω → Fin d → ℝ} (c : Fin d → ℝ) (hc : ∀ᵐ ω ∂μ, θtil ω = c) (Y : Ω → Fin d → ℝ) :
    popCrossCov μ (fun ω => θtil ω - popMean μ θtil) Y = 0 := by
  have hmean : popMean μ θtil = c := by
    rw [popMean, integral_congr_ae hc, integral_const]
    simp
  ext i j
  have hae : (fun ω => (θtil ω - popMean μ θtil) i * Y ω j) =ᵐ[μ] fun _ => 0 := by
    filter_upwards [hc] with ω hω
    simp [hω, hmean]
  simp only [popCrossCov, Matrix.of_apply, Matrix.zero_apply]
  rw [integral_congr_ae hae, integral_zero]

/-- Hence `σ_Jac² = 0` exactly: the paper's `σ_Jac² → 0` under a frozen anchor. -/
theorem sigmaJacSq_eq_zero_of_deterministic [IsProbabilityMeasure μ] {θtil : Ω → Fin d → ℝ}
    (c : Fin d → ℝ) (hc : ∀ᵐ ω ∂μ, θtil ω = c) (Y : Ω → Fin d → ℝ) :
    sigmaJacSq μ (fun ω => θtil ω - popMean μ θtil) Y = 0 := by
  rw [sigmaJacSq, popCrossCov_centered_eq_zero_of_deterministic c hc Y, Matrix.trace_zero]

/-- The iterate-fluctuation covariance `V = E[δθ̃ δθ̃ᵀ]` vanishes with it. This is the second
jitter-sourced ingredient of the corrected update-covariance decomposition of
`docs/design/lemma1_rederivation.md` §2, and it has to be zeroed separately from `σ_Jac²`. -/
theorem iterateFluctuationCovariance_eq_zero_of_deterministic [IsProbabilityMeasure μ]
    {θtil : Ω → Fin d → ℝ} (c : Fin d → ℝ) (hc : ∀ᵐ ω ∂μ, θtil ω = c) :
    popCrossCov μ (fun ω => θtil ω - popMean μ θtil) (fun ω => θtil ω - popMean μ θtil) = 0 :=
  popCrossCov_centered_eq_zero_of_deterministic c hc _

/-- Consequently the corrected decomposition's quadratic term `eᵀ H V H e` vanishes in every
direction and for every mean Hessian. -/
theorem quadForm_conj_eq_zero_of_deterministic [IsProbabilityMeasure μ] {θtil : Ω → Fin d → ℝ}
    (c : Fin d → ℝ) (hc : ∀ᵐ ω ∂μ, θtil ω = c) (H : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) :
    e ⬝ᵥ ((H * popCrossCov μ (fun ω => θtil ω - popMean μ θtil)
        (fun ω => θtil ω - popMean μ θtil) * H) *ᵥ e) = 0 := by
  rw [iterateFluctuationCovariance_eq_zero_of_deterministic c hc]
  simp

end FrozenAnchorPopulation

section MechanismSilent

/-- **The corrected effective diffusion collapses onto the direct baseline's.** In the corrected
route of `docs/design/lemma1_rederivation.md` §2 the bias-channel diffusion decomposes as
`σ_eff² = s² σ_obj² + c₁ σ_Jac² |H_bb| + s² (H V H)_bb`,
with the second and third terms sourced by the batch-induced jitter. Under a frozen anchor both of
those vanish — `sigmaJacSq_eq_zero_of_deterministic` and `quadForm_conj_eq_zero_of_deterministic`
supply exactly the two hypotheses — and the lift's effective diffusion is the direct baseline's
`s² σ_obj²`. This is the paper's "the lift falls silent until batch stochasticity is restored",
stated on the object that the corrected route actually puts `σ_Jac²` into.

The decomposition itself is not derived here; it belongs to `UpdateCovariance.lean`. What is proved
here is that its two jitter-sourced terms drop out, which is the whole content of the corollary. The
three coefficients are universally quantified, so the shipped paper's own additive form
`σ_eff² = σ_s² σ_obj² + σ_Jac²` is the case `c₁ = 1`, `|H_bb| = 1`, `(H V H)_bb = 0` of the same
statement: the corollary is neutral between the two routes rather than committed to the corrected
one. -/
theorem effectiveDiffusion_collapses_of_frozen_anchor {s sObj c₁ sJac Hbb quad : ℝ}
    (hJac : sJac = 0) (hquad : quad = 0) :
    s ^ 2 * sObj ^ 2 + c₁ * sJac * Hbb + s ^ 2 * quad = s ^ 2 * sObj ^ 2 := by
  rw [hJac, hquad]; ring

/-- With the effective diffusion back at the baseline value, the Kramers exponent `2α/(η σ_eff²)`
that it feeds is the baseline exponent: whatever escape-time advantage the lift had is gone. This is
pure algebra — the Arrhenius law itself is not available in this mathlib and is treated in
`KramersExitTime.lean` — but it is the step the falsifiability claim needs. -/
theorem kramersExponent_eq_of_effectiveDiffusion_eq {α η sEffLift sEffDirect : ℝ}
    (h : sEffLift = sEffDirect) :
    2 * α / (η * sEffLift) = 2 * α / (η * sEffDirect) := by rw [h]

/-- The two-sided bound that `Θ(·)` abbreviates, **at a single configuration**: a scalar `q` is
squeezed between two positive multiples of `σ_Jac² · scale`.

This is the shape of two different claims in two different versions of the mechanism — the shipped
Lemma 1's `μ_eff = Θ(σ_Jac²/(d κ²))`, whose Hessian derivation
`docs/design/lemma1_rederivation.md` §1 refutes, and the corrected route's diffusion term
`c₁ σ_Jac² |H_bb|` — and this file asserts neither of them.

A warning about this pointwise form, which `sigmaJacDominated_zero_iff` turns into a theorem so that
no reader has to take it on trust: **at `σ_Jac² = 0` it is logically equivalent to `q = 0`**. Both
bounds collapse to `0` and the hypothesis says exactly what the conclusion says. A pointwise
`Θ`-hypothesis is therefore worthless at precisely the point the falsifiability corollary is about,
because `Θ` is not a statement about one triple of numbers at all: it is a statement about a family,
with the implied constants fixed *before* the configuration varies. `SigmaJacDominatedOn` below is
that statement, it is what the corollaries in this file use, and it is genuinely stronger than its
conclusion. -/
def SigmaJacDominated (sJac scale q : ℝ) : Prop :=
  ∃ c₁ c₂ : ℝ, 0 < c₁ ∧ 0 < c₂ ∧ c₁ * (sJac * scale) ≤ q ∧ q ≤ c₂ * (sJac * scale)

/-- **The pointwise `Θ`-hypothesis is empty where it would be used.** At `σ_Jac² = 0` the
two-sided bound of `SigmaJacDominated` holds if and only if `q = 0`, so a corollary drawn from it
there assumes its own conclusion. This is stated as an `↔` rather than left implicit, because the
temptation to write the falsifiability corollary in the pointwise form and call the `Θ` relation a
carried hypothesis is exactly the corner this development is meant not to cut. -/
theorem sigmaJacDominated_zero_iff {scale q : ℝ} : SigmaJacDominated 0 scale q ↔ q = 0 := by
  constructor
  · rintro ⟨c₁, c₂, _, _, hlow, hhigh⟩
    rw [zero_mul, mul_zero] at hlow hhigh
    exact le_antisymm hhigh hlow
  · rintro rfl
    exact ⟨1, 1, one_pos, one_pos, by norm_num, by norm_num⟩

/-- The pointwise reading, kept only so that `sigmaJacDominated_zero_iff` has something to be about
and so that the family form below can be seen to specialize to it. It carries no content beyond
antisymmetry of `≤`; the falsifiability corollary uses `eq_zero_of_sigmaJacDominatedOn`. -/
theorem eq_zero_of_sigmaJacDominated {sJac scale q : ℝ} (h : SigmaJacDominated sJac scale q)
    (hJac : sJac = 0) : q = 0 := by
  rw [hJac] at h
  exact sigmaJacDominated_zero_iff.mp h

/-- **The paper's `Θ(·)`, with the implied constants quantified where `Θ` quantifies them.** A
modulus `q` — read as a function of `σ_Jac²`, which is how `μ_eff = Θ(σ_Jac²/(d κ²))` and the
corrected route's `c₁ σ_Jac² |H_bb|` are both meant — is **two-sidedly proportional to `σ_Jac²`** on
a set `S` of admissible values when there are two positive constants, *fixed once for the whole
family*, between whose multiples of `σ_Jac² · scale` it is squeezed at every value in `S`.

The uniformity is the entire content of the `Θ`. It is what makes the hypothesis a genuine
constraint on the modulus rather than a restatement of the corollary at the one point where the
corollary is applied, and it is what supports the paper's arrow `σ_Jac² → 0` implies `μ_eff → 0`,
which `tendsto_zero_of_sigmaJacDominatedOn` proves as an actual limit rather than as an evaluation
at a point. This file still asserts neither of the two proportionality claims: the relation is a
hypothesis, and `exists_sigmaJacDominatedOn_ne_zero` records that it is satisfiable by a modulus
that is not identically zero, so the corollaries below are not vacuous. -/
def SigmaJacDominatedOn (scale : ℝ) (S : Set ℝ) (q : ℝ → ℝ) : Prop :=
  ∃ c₁ c₂ : ℝ, 0 < c₁ ∧ 0 < c₂ ∧ ∀ x ∈ S, c₁ * (x * scale) ≤ q x ∧ q x ≤ c₂ * (x * scale)

/-- The family relation specializes to the pointwise one at each admissible value, so nothing is
lost by stating the hypothesis uniformly. -/
theorem sigmaJacDominated_of_sigmaJacDominatedOn {scale : ℝ} {S : Set ℝ} {q : ℝ → ℝ}
    (h : SigmaJacDominatedOn scale S q) {x : ℝ} (hx : x ∈ S) :
    SigmaJacDominated x scale (q x) := by
  obtain ⟨c₁, c₂, hc₁, hc₂, hb⟩ := h
  exact ⟨c₁, c₂, hc₁, hc₂, (hb x hx).1, (hb x hx).2⟩

/-- **The mechanism falls silent.** A modulus two-sidedly proportional to `σ_Jac²`, with constants
uniform over the admissible range, vanishes at `σ_Jac² = 0`. Unlike the pointwise version this is
not a restatement of its own hypothesis: the hypothesis constrains the modulus at every admissible
value, and the conclusion reads it off at one of them. -/
theorem eq_zero_of_sigmaJacDominatedOn {scale : ℝ} {S : Set ℝ} {q : ℝ → ℝ}
    (h : SigmaJacDominatedOn scale S q) (h0 : (0 : ℝ) ∈ S) : q 0 = 0 := by
  obtain ⟨c₁, c₂, _, _, hb⟩ := h
  obtain ⟨hlow, hhigh⟩ := hb 0 h0
  rw [zero_mul, mul_zero] at hlow hhigh
  exact le_antisymm hhigh hlow

/-- **The paper's arrow, as an arrow.** `rem:moreau-scope` writes "`σ_Jac² → 0` and `μ_eff → 0`",
which is a limit statement and not an evaluation at a point. With the `Θ` constants uniform, it is a
squeeze: along any net of configurations whose slack-channel trace tends to zero within the
admissible range, the modulus tends to zero as well. -/
theorem tendsto_zero_of_sigmaJacDominatedOn {scale : ℝ} {S : Set ℝ} {q : ℝ → ℝ}
    (h : SigmaJacDominatedOn scale S q) {ι : Type*} {l : Filter ι} {x : ι → ℝ}
    (hmem : ∀ᶠ a in l, x a ∈ S) (hx : Filter.Tendsto x l (nhds 0)) :
    Filter.Tendsto (fun a => q (x a)) l (nhds 0) := by
  obtain ⟨c₁, c₂, _, _, hb⟩ := h
  have hxs : Filter.Tendsto (fun a => x a * scale) l (nhds 0) := by
    simpa using hx.mul_const scale
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le'
    (g := fun a => c₁ * (x a * scale)) (h := fun a => c₂ * (x a * scale))
    (by simpa using hxs.const_mul c₁) (by simpa using hxs.const_mul c₂) ?_ ?_
  · filter_upwards [hmem] with a ha using (hb (x a) ha).1
  · filter_upwards [hmem] with a ha using (hb (x a) ha).2

/-- **The uniform `Θ`-hypothesis is satisfiable by a modulus that is not identically zero**, so the
two corollaries above are not vacuous and their conclusion is not forced by the hypothesis being
unsatisfiable. The witness is the identity modulus at unit scale on the non-negative half-line,
which is squeezed between half and twice itself. -/
theorem exists_sigmaJacDominatedOn_ne_zero :
    ∃ (scale : ℝ) (q : ℝ → ℝ), SigmaJacDominatedOn scale (Set.Ici 0) q ∧ q 0 = 0 ∧ q 1 ≠ 0 := by
  refine ⟨1, id, ⟨2⁻¹, 2, by norm_num, by norm_num, ?_⟩, rfl, one_ne_zero⟩
  intro x hx
  simp only [Set.mem_Ici] at hx
  constructor <;> simp <;> linarith

end MechanismSilent

section FrozenAnchorSummary

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **The falsifiability boundary of `rem:moreau-scope`, end to end.** Suppose an external
constraint freezes the conditioning set, so that the produced latent iterate `θ̃` is almost
surely a single vector `c`. Then, simultaneously and with no further assumption:

* the slack-channel cross-covariance trace vanishes, `σ_Jac² = 0`;
* any modulus two-sidedly proportional to `σ_Jac²` — read, as `Θ` requires, as a function of
  `σ_Jac²` with the implied constants uniform over the non-negative admissible range — vanishes at
  that value, on either the shipped or the corrected route;
* the corrected effective diffusion of `docs/design/lemma1_rederivation.md` §2 loses both of its
  jitter-sourced terms and collapses onto the direct baseline's `s² σ_obj²`, for every positivity-map
  prefactor, every objective-noise scale, every mean Hessian and every direction.

The lift then carries no channel the direct softplus lacks, which is exactly the boundary the paper
draws for its own claim.

The `Θ` relation is carried as the named hypothesis `hΘ` and is not asserted. It is deliberately the
*uniform* relation `SigmaJacDominatedOn` and not the pointwise `SigmaJacDominated`: by
`sigmaJacDominated_zero_iff` the pointwise relation at `σ_Jac² = 0` is logically equivalent to the
second conclusion, so a version of this theorem stated with it would be assuming what it proves. The
uniform relation constrains the modulus across the whole admissible range, and the theorem reads it
off at the value the frozen anchor produces. Everything else is derived from the frozen anchor. -/
theorem mechanism_falls_silent_of_frozen_anchor [IsProbabilityMeasure μ]
    {θtil : Ω → Fin d → ℝ} (c : Fin d → ℝ) (hc : ∀ᵐ ω ∂μ, θtil ω = c) (g : Ω → Fin d → ℝ)
    (H : Matrix (Fin d) (Fin d) ℝ) (e : Fin d → ℝ) {scale s sObj c₁ Hbb : ℝ} {q : ℝ → ℝ}
    (hΘ : SigmaJacDominatedOn scale (Set.Ici 0) q) :
    sigmaJacSq μ (fun ω => θtil ω - popMean μ θtil) g = 0 ∧
      q (sigmaJacSq μ (fun ω => θtil ω - popMean μ θtil) g) = 0 ∧
      s ^ 2 * sObj ^ 2
          + c₁ * sigmaJacSq μ (fun ω => θtil ω - popMean μ θtil) g * Hbb
          + s ^ 2 * (e ⬝ᵥ ((H * popCrossCov μ (fun ω => θtil ω - popMean μ θtil)
              (fun ω => θtil ω - popMean μ θtil) * H) *ᵥ e))
        = s ^ 2 * sObj ^ 2 := by
  have hJac := sigmaJacSq_eq_zero_of_deterministic c hc g
  refine ⟨hJac, ?_, effectiveDiffusion_collapses_of_frozen_anchor hJac
    (quadForm_conj_eq_zero_of_deterministic c hc H e)⟩
  rw [hJac]
  exact eq_zero_of_sigmaJacDominatedOn hΘ Set.self_mem_Ici

end FrozenAnchorSummary

end IcnnLift
