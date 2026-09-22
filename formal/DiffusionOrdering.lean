import Mathlib
import TraceLemmas
import UpdateCovariance
import CouplingSign

/-!
# (FW-1) Diffusion ordering on the shoulder band, and the coupling/magnitude excess split

This file machine-checks FW-1 of `docs/design/lemma1_rederivation.md` §8 ("FW-1 — diffusion
ordering on the shoulder band"), the excess split of §10 (`eq:excess-split`), and the
direct-baseline endpoint of §2. It belongs to the **corrected** mechanism. The shipped Lemma 1
put a cross-covariance term inside the pullback Hessian; §1 of the same note establishes that
this is false, because differentiation commutes with the batch expectation. Nothing here uses
that refuted decomposition. The cross-covariance enters where it actually lives, in the
covariance of the per-step update, hence in the diffusion coefficient.

The file is deliberately built on the two modules that already exist rather than restating them.
`UpdateCovariance.lean` supplies the **exact** decomposition of `Cov[G]` for an update expanding
as `G = s g + s H δθ̃ + R` (`updateCov_decomposition`), with the six remainder terms displayed
rather than discarded, together with the covariance matrix `covMat` and the projected diffusion
`projDiffusion`. `CouplingSign.lean` supplies the coupling term `couplingTerm H Σ = H Σ + Σᵀ H`,
its quadratic form `crossTerm`, assumption (A6) as `PositiveAlignmentOn`, and the ordering
`projDiffusionDirect_lt_projDiffusionLift` for its *postulated* decomposition
`projDiffusionLift`. What this file adds is the bridge and what the bridge buys:

* `updateCov_excess_eq` **derives** FW-1a — `Σ_lift = Σ_dir + s²(C + H V H) + T3` — from
  `updateCov_decomposition` at a common reference iterate (FW-0's ceteris-paribus convention),
  with `V = Cov[δθ̃] = covMat μ δθ̃ δθ̃`, `Σ_slack = E[δθ̃ δgᵀ] = covMat μ δθ̃ g`, and `T3` the
  explicit remainder group `updateRemainderMatrix`. FW-1a is therefore a theorem here, not a
  hypothesis.
* `projDiffusion_eq_projDiffusionLift` shows that `CouplingSign.projDiffusionLift`, which is a
  definition there, is **realized** by the actual projected update covariance. Everything
  `CouplingSign` proves about that definition consequently holds of the real object; that
  transfer is `updateCov_ordering_band` and `updateCov_ordering_strict`.
* `updateCov_posSemidef_ordering` gives the positive semidefinite ordering `Σ_lift ⪰ Σ_dir` of
  the full matrices, not merely of one projection, under §11.3's (A6); and
  `updateCov_posSemidef_ordering_of_forwardKL` gives the same ordering under §8's own hypotheses,
  the forward-KL chain of (A4') with the remainder control (A4'-q), which do not mention (A6) at
  all. The distinction is not cosmetic. (A6) is the empirical assumption §11.3 introduces for
  FW-2', and §8's FW-1 does not use it: presenting the (A6) route as FW-1 would misstate which
  hypothesis the ordering rests on. Both routes are proved here, separately and for the real
  update covariances.
* `updateCov_posSemidef_ordering_on_band` supplies the quantifier §11.3 writes and §8's pointwise
  statement does not have: (A6) as `CouplingSign.PositiveAlignmentOn` on a band `B`, one `κ` and
  one `τ` for the whole band, and the conclusion `Σ_lift ⪰ Σ_dir + (s²κ − τ) I` at every iterate
  of `B`. Every other ordering theorem in this file is at a *single* reference iterate.
* `covMat_self_posSemidef` discharges the hypothesis `V.PosSemidef` for the real covariance
  rather than assuming it.

**Which coefficient is formalized, and why.** §7's sketch writes the substituted excess as
`2 H V H + sym(H E[δθ̃ rᵀ]) + …`; §8's FW-1b writes `3 H V H + H E[δθ̃ rᵀ] + E[r δθ̃ᵀ] H`. This
file formalizes the `3`, and the reason is not a preference: `couplingTerm_forwardKL` **proves**
that substituting the forward-KL chain `δg = H δθ̃ + r`, whose second-moment form is
`Σ_slack = V H + E[δθ̃ rᵀ]`, into the coupling term yields exactly `2 H V H` plus the remainder
pairing, and `diffusionExcess_forwardKL` adds the magnitude channel's own `H V H` for a total of
`3 H V H`. §7's sketch dropped the magnitude term when substituting; §8's coefficient is the
arithmetically correct one, and the proofs below are a machine-checked certificate of that.
The second-moment form itself is not assumed either: `covMat_slack_of_chain` derives
`Σ_slack = V H + E[δθ̃ rᵀ]` for the real covariances from the pointwise chain, and
`diffusionExcess_forwardKL_of_chain` is FW-1b at the true `V` and `Σ_slack`.

**A second correction to the note, found by the arithmetic.** FW-1's parenthetical asserts
"in particular `E ≥ s² H̄ V H̄`" under (A4'-q). That comparison does **not** follow from
(A4'-q)'s constant `3/2`: the clause guarantees `eᵀ E e ≥ (3λ − 2ρ)‖e‖²`, which drops below
`λ‖e‖²` as soon as `ρ > λ`, and `forwardKL_excess_ge_magnitude_needs_tight_constant` exhibits a
realizable one-dimensional configuration — Rademacher jitter, `r = −(5/4) δθ̃` — satisfying
(A4'-q) on which `E` is PSD yet strictly below `H V H`. The comparison holds under the tighter
clause `ρ ≤ λ` (operator-norm coefficient `1` in place of `3/2`), which is
`forwardKL_excess_ge_magnitude`. FW-1's PSD conclusion is untouched; only the "in particular"
comparison overreaches.

**What the ordering does and does not say.** The magnitude channel `s² eᵀ H V H e` is
non-negative for every `e` and every `s`, given only `V ⪰ 0` and `Hᵀ = H`
(`IcnnLift.quadForm_conj_nonneg` of `TraceLemmas.lean`), and both of those are discharged for the
real objects — `V ⪰ 0` by `covMat_self_posSemidef`, `Hᵀ = H` by Clairaut. The coupling channel is
not sign-definite — `CouplingSign.coupling_sign_not_automatic` exhibits a positive definite `H`, a
positive definite `V` and a bounded remainder for which it is strictly negative — so its sign has
to be bought, and there are exactly two ways to buy it here: §11.3's (A6), which asserts it
outright, and §8's route through the forward-KL chain with (A4'-q), which controls it by bounding
the chain's remainder. Neither is free and the file keeps them distinct. The third-and-higher
remainder is carried explicitly and appears in every conclusion: the ordering degrades the
constant from `s²κ` to `s²κ − τ` rather than dropping the remainder.

**The limitation §10 records.** `magnitude_channel_ordering` states the corollary that the
magnitude channel alone — the coupling channel set to zero, which is exactly what
matched-marginal isotropic injected noise gives, since `δθ̃` independent of `δg` makes
`Σ_slack = 0` — still orders the two diffusions. The E1 experiment (30 seeds, and the σ bracket
of §11.0 at 1×/3×/10× injected scale) **refuted** the claim that this ordering suffices for the
observed effect: matched noise is statistically indistinguishable from no noise at every injected
scale, while the coupled latent weight wins. So the ordering proved here is necessary for the
mechanism and demonstrably not sufficient, and §11.0's scaling reading — the coupling channel
first order in the perturbation scale, the magnitude channel second — is the reason. That is a
genuine limitation of what the theory delivers, and it is why the two channels are kept as
separately named objects throughout instead of being summed away.

## Results
* `magnitudeMatrix`, `diffusionExcess` — the magnitude channel `H V H` and the FW-1a excess
  `s² (C + H V H) + T3`, with `C = CouplingSign.couplingTerm`.
* `excess_channel_split` — `eq:excess-split`: the projected excess is the sum of the coupling
  channel, the magnitude channel and the remainder channel.
* `magnitudeChannel_nonneg` — the magnitude channel is non-negative in every direction, given
  only `V ⪰ 0` and `Hᵀ = H`;
  `magnitudeMatrix_posDef`, `magnitudeChannel_pos` — and strictly positive when `V` is
  nondegenerate and `H` is injective, which is FW-1's `s² H̄ V H̄ > 0` clause.
* `covMat_self_posSemidef` — a covariance matrix is positive semidefinite.
* `covMat_direct_update` — the direct baseline's update covariance is `s² Cov[g]`.
* `updateCov_excess_eq` — **FW-1a, derived**: `Σ_lift = Σ_dir + diffusionExcess …`.
* `projDiffusion_excess_split` — its projection: `σ²_lift = σ²_dir + coupling + magnitude +
  remainder`.
* `projDiffusion_eq_projDiffusionLift` — the real projected diffusion realizes
  `CouplingSign.projDiffusionLift`.
* `updateCov_ordering_band`, `updateCov_ordering_strict` — under (A6) and a remainder bound `τ`,
  `σ²_dir + (s²κ − τ)‖e‖² ≤ σ²_lift`, strictly when `τ < s²κ` and `e ≠ 0`.
* `updateCov_posSemidef_ordering` — the positive semidefinite order `Σ_lift ⪰ Σ_dir`, under
  §11.3's (A6), at one reference iterate.
* `projDiffusion_ordering_of_forwardKL`, `updateCov_posSemidef_ordering_of_forwardKL` — the same
  ordering on §8's own hypotheses, the forward-KL chain and (A4'-q), with (A6) unused.
* `forwardKL_excess_posSemidef` — FW-1's conclusion "`E` is PSD" as a statement about the matrix
  rather than about one direction at a time.
* `updateCov_posSemidef_ordering_on_band` — the ordering with its margin, quantified over the
  band the way §11.3 quantifies it.
* `updateCov_posSemidef_ordering_margin` — the same with the margin kept,
  `Σ_lift ⪰ Σ_dir + (s²κ − τ) I`, which is the matrix inequality FW-2' (§11.3) actually uses.
* `dotProduct_symmetrizedPairing` — `eᵀ(H R + Rᵀ H)e = 2⟨H e, R e⟩`, the identity behind the
  factor `2` in (A4'-q).
* `couplingTerm_forwardKL`, `diffusionExcess_forwardKL` — (A4') substituted, giving FW-1b with
  the coefficient `3 H V H`.
* `covMat_slack_of_chain`, `diffusionExcess_forwardKL_of_chain` — the chain's second-moment form
  `Σ_slack = V H + E[δθ̃ rᵀ]` derived for the real covariances, and FW-1b at the true moments.
* `forwardKL_excess_lower_bound`, `forwardKL_excess_nonneg` — (A4'-q): the substituted excess is
  bounded below by `(3λ − 2ρ)‖e‖²`, hence positive semidefinite when `2ρ ≤ 3λ`.
* `forwardKL_excess_ge_magnitude`, `forwardKL_excess_ge_magnitude_needs_tight_constant` — the
  note's further claim `E ⪰ s² H V H` holds under `ρ ≤ λ` and provably **not** under (A4'-q)
  alone; the second is the machine-checked counterexample.
* `magnitude_channel_ordering`, `magnitude_channel_ordering_of_dominated` — the §10 corollary,
  which E1 shows is necessary but not sufficient for the observed effect.
* `updateCov_ordering_of_slack_zero` — the §10 corollary derived for the real update
  covariances: `Cov[δθ̃, g] = 0` (E1's matched-marginal cell) still orders the diffusions,
  provided the third-order remainder channel is dominated by the magnitude channel it perturbs.
* `channels_eq_zero_of_no_jitter`, `updateCov_eq_of_no_jitter` — the direct-baseline endpoint:
  with `δθ̃ = 0` both named channels vanish identically and (with `R = 0`) the two diffusions
  coincide.
* `jitter_necessary_for_strict_ordering` — the contrapositive, in the remainder-free model
  (`R ≡ 0`): a strict ordering forces a genuinely fluctuating latent weight.
* `updateCov_ordering_strict_nonvacuous` — the full hypothesis set of the strict ordering,
  measure-theoretic covariances included, is jointly satisfied on the standard-Gaussian witness
  of `UpdateCovariance.lean`, with (A6) at `κ = 4` and the ordering reading `1 < 9`.
* `updateCov_posSemidef_ordering_of_forwardKL_nonvacuous` — the same for the (A4'-q) hypothesis
  set, which is a different set and needs its own witness: on the same instance the chain forces
  a remainder as large as the jitter itself, `λ = 4` and `ρ = 2` satisfy (A4'-q), and the margin
  the theorem guarantees is exactly the excess the instance has.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The explicit hypotheses this file carries, and what it would take to discharge each:

* `hlift : ∀ ω, G_lift ω = s • g ω + s • (H *ᵥ δθ̃ ω) + R ω` and
  `hdir : ∀ ω, G_dir ω = s • g ω`. The first is `UpdateCovariance`'s own hypothesis and is not an
  approximation, because `R` is an arbitrary square-integrable random vector defined by the
  equation; the analytic work of showing `R` small is the second-order Taylor argument named in
  that module's honesty section. The second is FW-0's ceteris-paribus convention: the direct
  baseline is the same update with the latent-weight fluctuation removed, evaluated at the **same**
  reference iterate. §8 labels FW-0 a modelling convention rather than a theorem, and it is
  carried here as the hypothesis `hdir`, not smuggled in.
* `hH : Hᵀ = H`, symmetry of the mean Hessian at the reference iterate. Clairaut's theorem for a
  `C²` loss; a hypothesis only because no module here constructs `H` from a loss.
* `halign`, assumption (A6) of §11.3: `κ ‖e‖² ≤ eᵀ C e`. This is a genuine empirical assumption
  and the note says so — it asserts that training drives the latent weight's batch dependence into
  positive alignment with the curvature-weighted gradient fluctuation. It is not a theorem and
  cannot be made one: `CouplingSign.coupling_sign_not_automatic` exhibits a configuration on
  which it fails. In one dimension it reduces to the sign of `H̄ σ_Jac²`, which is measurable.
* `hT3`, the remainder bound `|eᵀ T3 e| ≤ τ ‖e‖²`. §8's FW-1 introduces a remainder-control
  clause and §11.3's proof says "T3 absorbed by (A4'-q)". That absorption is **not** literally
  implied by (A4'-q), which constrains only the second-moment remainder `E[δθ̃ rᵀ]` and says
  nothing about third moments; this file therefore carries the `T3` control as its own named
  hypothesis with its own constant `τ`, and reports `τ` in the conclusion.
  `UpdateCovariance.projDiffusion_remainder_bound` is what would supply it quantitatively from
  an `L²` bound on `R`; making that bound uniform on the band is the step §8 itself flags as
  "STATUS: needs the remainder bound made uniform on the band".
* `hlam`, `hrho`, `hA4q`: (A4'-q) itself, in quadratic-form language. The note states it with
  operator norms, `‖E[δθ̃ rᵀ]‖_op ≤ (3/2) λ_min(H V H)/‖H‖_op`. The quadratic-form hypotheses
  used here are implied by that clause — via `dotProduct_symmetrizedPairing` and
  `|2⟨He,Re⟩| ≤ 2‖H‖_op‖R‖_op‖e‖²` — so nothing stronger than the note is assumed. They are
  stated in quadratic-form terms because matrix operator norms are scoped in this mathlib and
  because the weaker hypothesis is the better one.
* The scope of every ordering theorem except `updateCov_posSemidef_ordering_on_band` is a
  *single* reference iterate. §8's FW-1 is itself pointwise — "Fix `θ̄` in the shoulder band `B`"
  — so this matches the note, and the occurrences of the word "band" in the names of
  `updateCov_ordering_band` and its relatives record the intended reading rather than a
  quantifier in the statement. §11.3's FW-2', by contrast, consumes the ordering uniformly on
  `B`, and `updateCov_posSemidef_ordering_on_band` is that statement: one `κ` and one `τ` at
  every iterate of the band. Its content beyond the pointwise theorem is exactly the uniformity
  of the two constants, which it *assumes*; producing a uniform `τ` from an `L²` bound on the
  remainder is the analytic step §8 flags with "STATUS: needs the remainder bound made uniform on
  the band", and it is not done here.
* (A6) and (A4'-q) are different assumptions and this file does not pass one off as the other.
  §8's FW-1 concludes the ordering from the forward-KL chain with (A4'-q)
  (`updateCov_posSemidef_ordering_of_forwardKL`); §11.3's FW-2' concludes it, with the margin
  `s²κ I` the quasipotential estimate needs, from (A6) (`updateCov_posSemidef_ordering`).
  Neither implies the other: the chain with `r = 0` gives the coupling term `2 H V H`, positive
  semidefinite but with no positive lower bound when `H V H` is degenerate, so (A4'-q) does not
  give (A6); and (A6) constrains only the coupling term and never mentions the remainder
  `E[δθ̃ rᵀ]` that (A4'-q) bounds, so it does not give (A4'-q).

What is **not** proved here. This file proves the ordering of the diffusion coefficients, not the
quasipotential comparison FW-2 draws from it and not the exit-time asymptotics: this mathlib has
no Itô calculus, no stochastic differential equations, no Freidlin–Wentzell theory and no exit
times, so those live in other modules with their own hypotheses. FW-1's further remark that the
excess is *strictly* positive on the slack subspace under (A3) is formalized only in the form
`magnitudeMatrix_posDef` — nondegenerate `V` and injective `H` — since the rank hypothesis on the
slack Jacobian is not modelled in this file.
-/

open Matrix MeasureTheory ProbabilityTheory

namespace IcnnLift

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Type*} [Fintype n]

/-! ## Elementary facts used throughout -/

omit [Fintype n] in
/-- Over `ℝ` a symmetric matrix is a Hermitian one; the converse direction of
`IcnnLift.transpose_eq_of_isHermitian`. `UpdateCovariance` states Hessian symmetry as `Hᵀ = H`
and `CouplingSign` states it as `H.IsHermitian`; this is the bridge between the two conventions,
which the ordering theorems below need because they use results from both. -/
theorem isHermitian_of_transpose_eq {H : Matrix n n ℝ} (hH : Hᵀ = H) : H.IsHermitian :=
  Matrix.isHermitian_iff_isSymm.mpr hH

/-! `0 ≤ ‖e‖²` and its strict form are stated in `CouplingSign.lean`
(`dotProduct_self_nonneg`, `dotProduct_self_pos`, namespace `IcnnLift`); they are not
re-declared here, because a second declaration of the same name in the same namespace would
not build. Where the file needs them they are inlined as `have` steps. -/

/-! ## The two channels of the FW-1a excess -/

/-- The **magnitude channel** `H V H` of `eq:excess-split` (§10): the part of the excess that
depends only on the marginal law of the latent-weight fluctuation, through `V = Cov[δθ̃]`. It is
positive semidefinite always, and §11.0 finds it empirically inert. Its partner is the coupling
channel `IcnnLift.couplingTerm H Σ_slack` of `CouplingSign.lean`, which sees the *joint* law. -/
def magnitudeMatrix (H V : Matrix n n ℝ) : Matrix n n ℝ := H * V * H

/-- The FW-1a excess `s² (C + H V H) + T3`, with `C = couplingTerm H Σ_slack` the coupling
channel and `T3` the third-and-higher mixed moments, carried explicitly rather than discarded. -/
def diffusionExcess (s : ℝ) (H V Sslack T3 : Matrix n n ℝ) : Matrix n n ℝ :=
  s ^ 2 • (couplingTerm H Sslack + magnitudeMatrix H V) + T3

/-- The coupling channel projected on a direction, `s² eᵀ (H Σ_slack + Σ_slackᵀ H) e`. In the
scalar reading of §2 this is the `c₁ σ_Jac² |H|` term. -/
def couplingChannel (s : ℝ) (H Sslack : Matrix n n ℝ) (e : n → ℝ) : ℝ :=
  s ^ 2 * crossTerm H Sslack e

/-- The magnitude channel projected on a direction, `s² eᵀ H V H e`; the quadratic term of §2. -/
def magnitudeChannel (s : ℝ) (H V : Matrix n n ℝ) (e : n → ℝ) : ℝ :=
  s ^ 2 * (e ⬝ᵥ (magnitudeMatrix H V *ᵥ e))

/-- The remainder channel `eᵀ T3 e`, the projection of the third-and-higher mixed moments. -/
def remainderChannel (T3 : Matrix n n ℝ) (e : n → ℝ) : ℝ := e ⬝ᵥ (T3 *ᵥ e)

/-- **`eq:excess-split` (§10).** The projected FW-1a excess splits into the coupling channel, the
magnitude channel and the remainder channel. §10's whole point is that the empirical evidence
distinguishes the first two, so they are kept as separate named terms. -/
theorem excess_channel_split (s : ℝ) (H V Sslack T3 : Matrix n n ℝ) (e : n → ℝ) :
    e ⬝ᵥ (diffusionExcess s H V Sslack T3 *ᵥ e)
      = couplingChannel s H Sslack e + magnitudeChannel s H V e + remainderChannel T3 e := by
  rw [diffusionExcess, quadForm_add, quadForm_smul, quadForm_add]
  simp only [couplingChannel, magnitudeChannel, remainderChannel, crossTerm]
  ring

/-- The magnitude channel is non-negative in every direction, with no assumption beyond the
positive semidefiniteness of `V` and the symmetry of `H`. This is
`IcnnLift.quadForm_conj_nonneg`; it is the reason the quadratic term of §2 can only add
diffusion. -/
theorem magnitudeChannel_nonneg {H V : Matrix n n ℝ} (hV : V.PosSemidef) (hH : H.IsHermitian)
    (s : ℝ) (e : n → ℝ) : 0 ≤ magnitudeChannel s H V e :=
  mul_nonneg (sq_nonneg s) (quadForm_conj_nonneg hV hH e)

/-- The magnitude channel matrix is positive **definite**, not merely semidefinite, when the
latent-weight-fluctuation covariance is nondegenerate and the curvature is injective. This is the
`s² H̄ V H̄ > 0` half of FW-1's parenthetical "`E ≥ s² H̄ V H̄ > 0` on the slack subspace whenever
the slack Jacobian has full rank (A3) and `V` is nondegenerate there", and only that half: the
comparison `E ⪰ s² H̄ V H̄` in the same parenthetical does *not* follow from (A4'-q), as
`forwardKL_excess_ge_magnitude_needs_tight_constant` shows. Nor is the restriction to the slack
subspace modelled — the statement here is positivity in every direction, under a hypothesis on
`V` and `H` in place of the note's rank hypothesis on the slack Jacobian, which no object in
this file carries. -/
theorem magnitudeMatrix_posDef {H V : Matrix n n ℝ} (hV : V.PosDef) (hH : H.IsHermitian)
    (hinj : Function.Injective H.mulVec) : (magnitudeMatrix H V).PosDef := by
  have h := hV.conjTranspose_mul_mul_same hinj
  rw [hH.eq] at h
  exact h

/-- Hence the projected magnitude channel is strictly positive in every nonzero direction, for
every nonzero positivity-map prefactor. This is where the shoulder's `s ≠ 0` clause of (A2) enters. -/
theorem magnitudeChannel_pos {H V : Matrix n n ℝ} (hV : V.PosDef) (hH : H.IsHermitian)
    (hinj : Function.Injective H.mulVec) {s : ℝ} (hs : s ≠ 0) {e : n → ℝ} (he : e ≠ 0) :
    0 < magnitudeChannel s H V e := by
  have h := (magnitudeMatrix_posDef hV hH hinj).dotProduct_mulVec_pos he
  have hstar : (star e : n → ℝ) = e := rfl
  rw [hstar] at h
  exact mul_pos (by positivity) h

/-! ## FW-1a, derived from the exact update-covariance decomposition -/

/-- A covariance matrix is positive semidefinite: its quadratic form in the direction `e` is the
variance of the projection `⟨e, U⟩`. This discharges the hypothesis `V.PosSemidef` of the
ordering theorems for the actual latent-weight-fluctuation covariance, rather than assuming it. -/
theorem covMat_self_posSemidef [IsFiniteMeasure μ] {U : Ω → n → ℝ} (hU : SqIntegrableVec μ U) :
    (covMat μ U U).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
    (Matrix.isHermitian_iff_isSymm.mpr (covMat_transpose μ U U)) fun e => ?_
  have hstar : (star e : n → ℝ) = e := rfl
  rw [hstar, covMat_quadForm_eq_covariance e hU hU,
    covariance_self (hU.projection e).aestronglyMeasurable.aemeasurable]
  exact variance_nonneg _ _

omit [Fintype n] in
/-- The **direct baseline's** update covariance at the common reference iterate of FW-0: with no
latent-weight fluctuation the update is `s g`, so its covariance is the once-attenuated objective
noise `s² Cov[g]`. -/
theorem covMat_direct_update (s : ℝ) {g Gdir : Ω → n → ℝ} (hdir : ∀ ω, Gdir ω = s • g ω) :
    covMat μ Gdir Gdir = s ^ 2 • covMat μ g g := by
  have h : Gdir = fun ω => s • g ω := funext hdir
  subst h
  rw [covMat_smul_left, covMat_smul_right, smul_smul, sq]

/-- The **remainder group** of the exact decomposition: the six terms of
`UpdateCovariance.updateCov_decomposition` that carry the remainder `R`. This is the note's `T3`,
"third-and-higher mixed moments", named rather than abbreviated. -/
noncomputable def updateRemainderMatrix (μ : Measure Ω) (s : ℝ) (H : Matrix n n ℝ)
    (g dth R : Ω → n → ℝ) : Matrix n n ℝ :=
  s • (covMat μ g R + covMat μ R g) + s • (H * covMat μ dth R + covMat μ R dth * Hᵀ)
    + covMat μ R R

/-- **FW-1a, derived.** At the common reference iterate of FW-0, with the lift's update expanding
as `G = s g + s H δθ̃ + R` and the direct baseline's as `G = s g`, the lifted update covariance is
the direct one plus the FW-1a excess

  `Σ_lift − Σ_dir = s² (H Σ_slack + Σ_slackᵀ H + H V H) + T3`,

with `V = Cov[δθ̃]`, `Σ_slack = E[δθ̃ δgᵀ]` and `T3` the explicit remainder group. This is an
identity, not an approximation: it is `UpdateCovariance.updateCov_decomposition` rearranged, and
every term the note writes as `+ h.o.t.` is inside `updateRemainderMatrix`. -/
theorem updateCov_excess_eq [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) :
    covMat μ Glift Glift
      = covMat μ Gdir Gdir
        + diffusionExcess s H (covMat μ dth dth) (covMat μ dth g)
            (updateRemainderMatrix μ s H g dth R) := by
  have hc : couplingTerm H (covMat μ dth g) = covMat μ g dth * Hᵀ + H * covMat μ dth g := by
    rw [couplingTerm, covMat_transpose, hH]
    exact add_comm _ _
  have hm : magnitudeMatrix H (covMat μ dth dth) = H * covMat μ dth dth * Hᵀ := by
    rw [magnitudeMatrix, hH]
  rw [updateCov_decomposition s H hg hθ hR hlift, covMat_direct_update s hdir, diffusionExcess,
    hc, hm, updateRemainderMatrix]
  simp only [smul_add]
  abel

/-- The projection of FW-1a on a direction: `σ²_lift = σ²_dir + coupling + magnitude + remainder`,
with no term dropped. This is `eq:excess-split` applied to the real update covariances. -/
theorem projDiffusion_excess_split [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) (e : n → ℝ) :
    projDiffusion μ e Glift
      = projDiffusion μ e Gdir + couplingChannel s H (covMat μ dth g) e
        + magnitudeChannel s H (covMat μ dth dth) e
        + remainderChannel (updateRemainderMatrix μ s H g dth R) e := by
  rw [projDiffusion, projDiffusion, updateCov_excess_eq s hH hg hθ hR hlift hdir, quadForm_add,
    excess_channel_split]
  ring

/-- The direct baseline's projected diffusion is `CouplingSign.projDiffusionDirect`. -/
theorem projDiffusion_direct_eq (s : ℝ) {g Gdir : Ω → n → ℝ} (hdir : ∀ ω, Gdir ω = s • g ω)
    (e : n → ℝ) : projDiffusion μ e Gdir = projDiffusionDirect s (covMat μ g g) e := by
  rw [projDiffusion, covMat_direct_update s hdir, quadForm_smul, projDiffusionDirect]

/-- **The bridge to `CouplingSign`.** `CouplingSign.projDiffusionLift` is a *definition* there —
it postulates the three-channel shape of the lift's projected diffusion. Here that shape is
shown to be **realized** by the actual projected update covariance, with the objective noise,
the jitter covariance and the slack cross-covariance instantiated at their true values and the
third-order term at the projected remainder group. Every consequence `CouplingSign` draws from
its definition therefore holds of the real object. -/
theorem projDiffusion_eq_projDiffusionLift [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dth R Glift : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dth) (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω) (e : n → ℝ) :
    projDiffusion μ e Glift
      = projDiffusionLift s (covMat μ g g) (covMat μ dth dth) H (covMat μ dth g)
          (remainderChannel (updateRemainderMatrix μ s H g dth R) e) e := by
  have hdir : ∀ ω, (fun ω => s • g ω) ω = s • g ω := fun _ => rfl
  rw [projDiffusion_excess_split s hH hg hθ hR hlift hdir e,
    projDiffusion_direct_eq s hdir e, projDiffusionLift, couplingChannel, magnitudeChannel,
    magnitudeMatrix]
  ring

/-! ## The ordering on the shoulder band -/

/-- **FW-2', the ordering, for the real update covariances.** Under (A6) — the coupling term
dominates `κ I` in the direction `e` — and an explicit bound `τ` on the third-and-higher
remainder, the lift's projected diffusion exceeds the direct baseline's with the explicit margin
`(s² κ − τ) ‖e‖²`. The remainder degrades the constant; it is never discarded.

Two scope warnings, because the name says "band" and the statement does not. First, this is a
statement at a *single* reference iterate; no band appears in it, and the uniform-on-the-band
form that §11.3 consumes is `updateCov_posSemidef_ordering_on_band`. Second, the hypothesis is
(A6), which is §11.3's and not §8's: FW-1 itself derives the ordering from the forward-KL chain
and (A4'-q), which is `projDiffusion_ordering_of_forwardKL`.

The proof is the bridge `projDiffusion_eq_projDiffusionLift` composed with
`CouplingSign.projDiffusion_ge_of_positiveAlignment`; the positive semidefiniteness of the
latent-weight-fluctuation covariance is supplied by `covMat_self_posSemidef` rather than assumed. -/
theorem updateCov_ordering_band [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) {kappa tau : ℝ} {e : n → ℝ}
    (halign : kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g) e)
    (hT3 : |remainderChannel (updateRemainderMatrix μ s H g dth R) e| ≤ tau * (e ⬝ᵥ e)) :
    projDiffusion μ e Gdir + (s ^ 2 * kappa - tau) * (e ⬝ᵥ e) ≤ projDiffusion μ e Glift := by
  rw [projDiffusion_eq_projDiffusionLift s hH hg hθ hR hlift e, projDiffusion_direct_eq s hdir e]
  exact projDiffusion_ge_of_positiveAlignment (covMat_self_posSemidef hθ)
    (isHermitian_of_transpose_eq hH) halign hT3

/-- **The strict ordering.** When the remainder bound is smaller than the alignment gain the band
ordering is strict in every nonzero direction: `σ²_dir < σ²_lift`. This is the conclusion FW-2'
feeds into the quasipotential comparison. -/
theorem updateCov_ordering_strict [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) {kappa tau : ℝ} {e : n → ℝ}
    (halign : kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g) e)
    (hT3 : |remainderChannel (updateRemainderMatrix μ s H g dth R) e| ≤ tau * (e ⬝ᵥ e))
    (htau : tau < s ^ 2 * kappa) (he : e ≠ 0) :
    projDiffusion μ e Gdir < projDiffusion μ e Glift := by
  have hband := updateCov_ordering_band s hH hg hθ hR hlift hdir halign hT3
  have hpos : 0 < (s ^ 2 * kappa - tau) * (e ⬝ᵥ e) :=
    mul_pos (by linarith) (dotProduct_self_pos he)
  linarith

/-- **The positive semidefinite ordering `Σ_lift(θ̄) ⪰ Σ_dir(θ̄)`**, as matrices rather than in
one projection, which is the shape §8 states FW-1 in. The hypotheses, however, are §11.3's and
not §8's: (A6), and the remainder bound holding uniformly in the direction with the remainder
bound not exceeding the alignment gain. FW-1's own route to the same conclusion — the forward-KL
chain and (A4'-q), with no appeal to (A6) — is `updateCov_posSemidef_ordering_of_forwardKL`.
This statement is at one reference iterate; the band-quantified form is
`updateCov_posSemidef_ordering_on_band`. -/
theorem updateCov_posSemidef_ordering [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) {kappa tau : ℝ}
    (halign : ∀ e : n → ℝ, kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g) e)
    (hT3 : ∀ e : n → ℝ,
      |remainderChannel (updateRemainderMatrix μ s H g dth R) e| ≤ tau * (e ⬝ᵥ e))
    (htau : tau ≤ s ^ 2 * kappa) :
    (covMat μ Glift Glift - covMat μ Gdir Gdir).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun e => ?_
  · refine Matrix.isHermitian_iff_isSymm.mpr ?_
    show (covMat μ Glift Glift - covMat μ Gdir Gdir)ᵀ
      = covMat μ Glift Glift - covMat μ Gdir Gdir
    rw [Matrix.transpose_sub, covMat_transpose, covMat_transpose]
  · have hstar : (star e : n → ℝ) = e := rfl
    have hsub : e ⬝ᵥ ((covMat μ Glift Glift - covMat μ Gdir Gdir) *ᵥ e)
        = projDiffusion μ e Glift - projDiffusion μ e Gdir := by
      rw [projDiffusion, projDiffusion, Matrix.sub_mulVec, dotProduct_sub]
    have hband := updateCov_ordering_band s hH hg hθ hR hlift hdir (halign e) (hT3 e)
    have hgain : 0 ≤ (s ^ 2 * kappa - tau) * (e ⬝ᵥ e) :=
      mul_nonneg (by linarith) (dotProduct_self_nonneg e)
    rw [hstar, hsub]
    linarith

/-- **The ordering with its margin kept, in the positive semidefinite order.** §11.3's FW-2'
does not consume the bare ordering `Σ_lift ⪰ Σ_dir`; its proof uses `Σ_lift ⪰ Σ_dir + s²κ I`
on the band, since the quasipotential gap `δ = s²κ / (Λ(Λ + s²κ))` is computed from that
margin. This is that statement for the real update covariances, with the remainder degrading
the margin to `s²κ − τ` rather than being dropped: `Σ_lift − Σ_dir − (s²κ − τ) I ⪰ 0`. No
constraint between `τ` and `s²κ` is needed — when `τ > s²κ` the statement is weaker than
`updateCov_posSemidef_ordering` but still true — and `updateCov_posSemidef_ordering` is the
special case obtained by discarding a non-negative margin. -/
theorem updateCov_posSemidef_ordering_margin [DecidableEq n] [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) {kappa tau : ℝ}
    (halign : ∀ e : n → ℝ, kappa * (e ⬝ᵥ e) ≤ crossTerm H (covMat μ dth g) e)
    (hT3 : ∀ e : n → ℝ,
      |remainderChannel (updateRemainderMatrix μ s H g dth R) e| ≤ tau * (e ⬝ᵥ e)) :
    (covMat μ Glift Glift - covMat μ Gdir Gdir
        - (s ^ 2 * kappa - tau) • (1 : Matrix n n ℝ)).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun e => ?_
  · refine Matrix.isHermitian_iff_isSymm.mpr ?_
    show (covMat μ Glift Glift - covMat μ Gdir Gdir
        - (s ^ 2 * kappa - tau) • (1 : Matrix n n ℝ))ᵀ
      = covMat μ Glift Glift - covMat μ Gdir Gdir - (s ^ 2 * kappa - tau) • (1 : Matrix n n ℝ)
    rw [Matrix.transpose_sub, Matrix.transpose_sub, covMat_transpose, covMat_transpose,
      Matrix.transpose_smul, Matrix.transpose_one]
  · have hstar : (star e : n → ℝ) = e := rfl
    have hq : e ⬝ᵥ ((covMat μ Glift Glift - covMat μ Gdir Gdir
          - (s ^ 2 * kappa - tau) • (1 : Matrix n n ℝ)) *ᵥ e)
        = projDiffusion μ e Glift - projDiffusion μ e Gdir
          - (s ^ 2 * kappa - tau) * (e ⬝ᵥ e) := by
      rw [projDiffusion, projDiffusion, Matrix.sub_mulVec, Matrix.sub_mulVec, dotProduct_sub,
        dotProduct_sub, quadForm_smul, Matrix.one_mulVec]
    have hband := updateCov_ordering_band s hH hg hθ hR hlift hdir (halign e) (hT3 e)
    rw [hstar, hq]
    linarith

/-- **The ordering uniformly on the band, with (A6) as §11.3 literally states it.** Every
ordering theorem above is pointwise at a single reference iterate: there is no band in its
statement, and the word "band" in the names records the intended reading rather than a
quantifier. §11.3 states (A6) as `C(θ̄) ⪰ κ I` *for all* `θ̄ ∈ B` and concludes
`Σ_lift ⪰ Σ_dir + s² κ I` *on* `B`, so the quantifier over the band belongs in the statement.
Here it is: the reference iterate ranges over an abstract index set, every datum of the
comparison is a family indexed by it, (A6) is `CouplingSign.PositiveAlignmentOn` — the note's
assumption verbatim, `κ > 0` included — and the remainder bound `τ` is uniform in the iterate
and the direction, which is what §8's "STATUS: needs the remainder bound made uniform on the
band" asks for and what this theorem consequently *assumes* rather than proves.

The mathematical content beyond `updateCov_posSemidef_ordering_margin` is exactly the
uniformity of the two constants: the theorem is that one `κ` and one `τ` suffice at every
iterate of the band. Producing such a pair from an `L²` bound on the remainder is the analytic
step §8 flags as open; `UpdateCovariance.projDiffusion_remainder_bound` is its quantitative
input. -/
theorem updateCov_posSemidef_ordering_on_band {ι : Type*} [DecidableEq n] [IsFiniteMeasure μ]
    {B : Set ι} {sf : ι → ℝ} {Hf : ι → Matrix n n ℝ}
    {gf dthf Rf Gliftf Gdirf : ι → Ω → n → ℝ}
    (hH : ∀ x ∈ B, (Hf x)ᵀ = Hf x)
    (hg : ∀ x ∈ B, SqIntegrableVec μ (gf x)) (hθ : ∀ x ∈ B, SqIntegrableVec μ (dthf x))
    (hR : ∀ x ∈ B, SqIntegrableVec μ (Rf x))
    (hlift : ∀ x ∈ B, ∀ ω, Gliftf x ω = sf x • gf x ω + sf x • (Hf x *ᵥ dthf x ω) + Rf x ω)
    (hdir : ∀ x ∈ B, ∀ ω, Gdirf x ω = sf x • gf x ω) {kappa tau : ℝ}
    (hA6 : PositiveAlignmentOn B Hf (fun x => covMat μ (dthf x) (gf x)) kappa)
    (hT3 : ∀ x ∈ B, ∀ e : n → ℝ,
      |remainderChannel (updateRemainderMatrix μ (sf x) (Hf x) (gf x) (dthf x) (Rf x)) e|
        ≤ tau * (e ⬝ᵥ e))
    {x : ι} (hx : x ∈ B) :
    (covMat μ (Gliftf x) (Gliftf x) - covMat μ (Gdirf x) (Gdirf x)
        - (sf x ^ 2 * kappa - tau) • (1 : Matrix n n ℝ)).PosSemidef :=
  updateCov_posSemidef_ordering_margin (sf x) (hH x hx) (hg x hx) (hθ x hx) (hR x hx)
    (hlift x hx) (hdir x hx) (fun e => hA6.2 x hx e) (hT3 x hx)

/-! ## (A4'): substituting the forward-KL chain, and the coefficient `3 H V H` -/

/-- The symmetrized pairing identity `eᵀ (H R + Rᵀ H) e = 2 ⟨H e, R e⟩` for symmetric `H`. This
is where the factor `2` in the remainder-control clause (A4'-q) comes from: bounding the left
side by `2 ‖H‖_op ‖R‖_op ‖e‖²` is Cauchy–Schwarz applied to the right side. -/
theorem dotProduct_symmetrizedPairing {H : Matrix n n ℝ} (hH : Hᵀ = H) (R : Matrix n n ℝ)
    (e : n → ℝ) :
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

/-- **(A4') substituted into the coupling channel.** Under the forward-KL chain `δg = H δθ̃ + r`,
whose second-moment form is `Σ_slack = V H + E[δθ̃ rᵀ]`, the coupling term is
`2 H V H + (H R + Rᵀ H)` with `R = E[δθ̃ rᵀ]`. The factor `2` is the note's, and it is the whole
of the coupling channel's contribution: the magnitude channel's own `H V H` is not part of it.
This is the matrix-level companion of `CouplingSign.crossTerm_eq_add_remainder`, which states the
same substitution for the quadratic form. -/
theorem couplingTerm_forwardKL {H V Sslack R : Matrix n n ℝ}
    (hH : Hᵀ = H) (hV : Vᵀ = V) (hchain : Sslack = V * H + R) :
    couplingTerm H Sslack = (2 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H) := by
  have ht : (V * H + R)ᵀ = H * V + Rᵀ := by
    rw [Matrix.transpose_add, Matrix.transpose_mul, hH, hV]
  have key : couplingTerm H Sslack
      = magnitudeMatrix H V + magnitudeMatrix H V + (H * R + Rᵀ * H) := by
    simp only [couplingTerm, magnitudeMatrix, hchain, ht, mul_add, add_mul, ← mul_assoc]
    abel
  rw [key, two_smul]

/-- **FW-1b, with the `3 H V H` of §8.** Substituting the forward-KL chain into FW-1a gives
`Σ_lift − Σ_dir = s² (3 H V H + H R + Rᵀ H) + T3`. The coefficient is `3`, not the `2` of the §7
sketch: `2` comes from the coupling channel (`couplingTerm_forwardKL`) and `1` from the magnitude
channel, which the §7 sketch dropped when substituting. -/
theorem diffusionExcess_forwardKL {s : ℝ} {H V Sslack R T3 : Matrix n n ℝ}
    (hH : Hᵀ = H) (hV : Vᵀ = V) (hchain : Sslack = V * H + R) :
    diffusionExcess s H V Sslack T3
      = s ^ 2 • ((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)) + T3 := by
  rw [diffusionExcess, couplingTerm_forwardKL hH hV hchain]
  module

/-- **The second-moment form of the forward-KL chain, derived rather than assumed.** The two
substitution theorems above take `Σ_slack = V H + R` as a hypothesis; the note asserts that
identity as the "second-moment form" of the pointwise chain `δg = H δθ̃ + r`, and this theorem
derives it for the real covariances, from bilinearity: if `g = H δθ̃ + r` pointwise then
`Cov[δθ̃, g] = Cov[δθ̃] H + Cov[δθ̃, r]`. The covariance ignores means, so neither `g` nor `r`
need be centered here — for the note's centered remainder `r`, `covMat μ dth r` **is** its
`E[δθ̃ rᵀ]` — and the hypothesis `hchain` rules out no configuration, because `r` can always be
defined as `g − H δθ̃`. This is the measure-theoretic companion of
`CouplingSign.outerMoment_chain`, which does the same for the finite-sample vocabulary. -/
theorem covMat_slack_of_chain [IsFiniteMeasure μ] {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth r : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r)
    (hchain : ∀ ω, g ω = H *ᵥ dth ω + r ω) :
    covMat μ dth g = covMat μ dth dth * H + covMat μ dth r := by
  have hg : g = fun ω => H *ᵥ dth ω + r ω := funext hchain
  subst hg
  rw [covMat_add_right hθ (hθ.mulVec H) hr, covMat_mulVec_right H hθ hθ, hH]

/-- **FW-1b for the real objects.** Under the pointwise forward-KL chain `g = H δθ̃ + r`, the
FW-1a excess at the true covariances — `V = Cov[δθ̃]`, `Σ_slack = Cov[δθ̃, g]` — equals
`s² (3 H V H + H E[δθ̃ rᵀ] + E[r δθ̃ᵀ] H) + T3`, with `E[δθ̃ rᵀ] = covMat μ dth r`. This closes
the gap between the abstract substitution `diffusionExcess_forwardKL` and the derived
decomposition `updateCov_excess_eq`: nothing in FW-1b is assumed at the matrix level any more. -/
theorem diffusionExcess_forwardKL_of_chain [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dth r : Ω → n → ℝ} (hθ : SqIntegrableVec μ dth)
    (hr : SqIntegrableVec μ r) (hchain : ∀ ω, g ω = H *ᵥ dth ω + r ω) (T3 : Matrix n n ℝ) :
    diffusionExcess s H (covMat μ dth dth) (covMat μ dth g) T3
      = s ^ 2 • ((3 : ℝ) • magnitudeMatrix H (covMat μ dth dth)
          + (H * covMat μ dth r + (covMat μ dth r)ᵀ * H)) + T3 :=
  diffusionExcess_forwardKL hH (covMat_transpose μ dth dth)
    (covMat_slack_of_chain hH hθ hr hchain)

/-- **(A4'-q), the remainder-control clause of FW-1, in quadratic-form language.** If the
magnitude channel is bounded below by `lam` and the symmetrized remainder pairing is bounded in
modulus by `2 rho`, then the substituted excess of FW-1b is bounded below by
`(3 lam − 2 rho) ‖e‖²`. The note's operator-norm clause
`‖E[δθ̃ rᵀ]‖_op ≤ (3/2) λ_min(H V H) / ‖H‖_op` implies both hypotheses, with
`rho = ‖H‖_op ‖E[δθ̃ rᵀ]‖_op`, via `dotProduct_symmetrizedPairing` and Cauchy–Schwarz. -/
theorem forwardKL_excess_lower_bound {H V R : Matrix n n ℝ} {lam rho : ℝ}
    (hlam : ∀ e : n → ℝ, lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H V *ᵥ e))
    (hrho : ∀ e : n → ℝ, |e ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    (e : n → ℝ) :
    (3 * lam - 2 * rho) * (e ⬝ᵥ e)
      ≤ e ⬝ᵥ (((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)) *ᵥ e) := by
  rw [quadForm_add, quadForm_smul]
  have h1 := hlam e
  have h2 := (abs_le.mp (hrho e)).1
  linarith

/-- Under (A4'-q) proper, `2 rho ≤ 3 lam`, the substituted excess of FW-1b is positive
semidefinite. This is the note's PSD conclusion, with the arithmetic done. -/
theorem forwardKL_excess_nonneg {H V R : Matrix n n ℝ} {lam rho : ℝ}
    (hlam : ∀ e : n → ℝ, lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H V *ᵥ e))
    (hrho : ∀ e : n → ℝ, |e ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    (hA4q : 2 * rho ≤ 3 * lam) (e : n → ℝ) :
    0 ≤ e ⬝ᵥ (((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)) *ᵥ e) := by
  have hlb := forwardKL_excess_lower_bound hlam hrho e
  have hnn : 0 ≤ (3 * lam - 2 * rho) * (e ⬝ᵥ e) :=
    mul_nonneg (by linarith) (dotProduct_self_nonneg e)
  linarith

/-- **FW-1's positive-semidefiniteness conclusion at the matrix level.**
`forwardKL_excess_nonneg` gives the non-negativity of the substituted excess in each direction
separately; §8 states the conclusion as a property of the matrix — "`E` is PSD whenever the
remainder-control clause (A4'-q) holds" — and this theorem assembles it. The symmetry of the
substituted excess is not assumed: it follows from `Hᵀ = H` and `Vᵀ = V`, the pairing
`H R + Rᵀ H` being symmetric for every `R` whatsoever. -/
theorem forwardKL_excess_posSemidef {H V R : Matrix n n ℝ} {lam rho : ℝ}
    (hH : Hᵀ = H) (hV : Vᵀ = V)
    (hlam : ∀ e : n → ℝ, lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H V *ᵥ e))
    (hrho : ∀ e : n → ℝ, |e ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    (hA4q : 2 * rho ≤ 3 * lam) :
    ((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)).PosSemidef := by
  have hmag : (magnitudeMatrix H V)ᵀ = magnitudeMatrix H V := by
    rw [magnitudeMatrix, Matrix.transpose_mul, Matrix.transpose_mul, hH, hV, ← Matrix.mul_assoc]
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
    (Matrix.isHermitian_iff_isSymm.mpr ?_) fun e => ?_
  · show ((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H))ᵀ
      = (3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)
    simp only [Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_mul,
      Matrix.transpose_transpose, hmag, hH]
    abel
  · have hstar : (star e : n → ℝ) = e := rfl
    rw [hstar]
    exact forwardKL_excess_nonneg hlam hrho hA4q e

/-- **The comparison `E ⪰ H V H` needs a tighter constant than (A4'-q), and this is a
correction to the note.** FW-1's parenthetical asserts "in particular `E ≥ s² H̄ V H̄`" in the
same breath as (A4'-q), but the arithmetic does not go through: (A4'-q) guarantees only
`eᵀ E e ≥ (3λ − 2ρ)‖e‖²`, and `3λ − 2ρ < λ` as soon as `ρ > λ`, which (A4'-q)'s constant
`3/2` permits. The comparison holds under the strictly tighter clause `ρ ≤ λ` — in operator
norms, `‖E[δθ̃ rᵀ]‖_op ≤ λ_min(H V H)/‖H̄‖_op`, coefficient `1` in place of the note's `3/2` —
which is what this theorem carries. `forwardKL_excess_ge_magnitude_needs_tight_constant` below
is the machine-checked witness that (A4'-q) alone is not enough. -/
theorem forwardKL_excess_ge_magnitude {H V R : Matrix n n ℝ} {lam rho : ℝ}
    (hlam : ∀ e : n → ℝ, lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H V *ᵥ e))
    (hrho : ∀ e : n → ℝ, |e ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    (hstrong : rho ≤ lam) (e : n → ℝ) :
    e ⬝ᵥ (magnitudeMatrix H V *ᵥ e)
      ≤ e ⬝ᵥ (((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)) *ᵥ e) := by
  rw [quadForm_add, quadForm_smul]
  have h1 := hlam e
  have h2 := (abs_le.mp (hrho e)).1
  have h3 : 0 ≤ (lam - rho) * (e ⬝ᵥ e) :=
    mul_nonneg (by linarith) (dotProduct_self_nonneg e)
  nlinarith

/-- **(A4'-q) does not imply `E ⪰ s² H V H`: a one-dimensional counterexample.** With `H = 1`,
`V = 1`, `R = −(5/4)`, the hypotheses of `forwardKL_excess_lower_bound` hold with `λ = 1` and
`ρ = 5/4`, and (A4'-q) holds — `2ρ = 5/2 ≤ 3 = 3λ`, so the substituted excess is positive
semidefinite (`eᵀ E e = ‖e‖²/2 ≥ 0`) — yet the excess is strictly **below** the magnitude
matrix in the unit direction: `1/2 < 1`. The configuration is realizable, not merely formal:
`δθ̃` a Rademacher sign and `r = −(5/4) δθ̃` give exactly these second moments. The note's
"in particular `E ≥ s² H̄ V H̄`" therefore does not follow from (A4'-q) as stated; it needs
`ρ ≤ λ`, which is `forwardKL_excess_ge_magnitude`. The PSD conclusion of FW-1 itself is
untouched — this counterexample satisfies it. -/
theorem forwardKL_excess_ge_magnitude_needs_tight_constant :
    ∃ (H V R : Matrix (Fin 1) (Fin 1) ℝ) (lam rho : ℝ),
      Hᵀ = H ∧ Vᵀ = V ∧ 0 < lam ∧
      (∀ e : Fin 1 → ℝ, lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H V *ᵥ e)) ∧
      (∀ e : Fin 1 → ℝ, |e ⬝ᵥ ((H * R + Rᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e)) ∧
      2 * rho ≤ 3 * lam ∧
      ¬ (∀ e : Fin 1 → ℝ, e ⬝ᵥ (magnitudeMatrix H V *ᵥ e)
            ≤ e ⬝ᵥ (((3 : ℝ) • magnitudeMatrix H V + (H * R + Rᵀ * H)) *ᵥ e)) := by
  have hmm : magnitudeMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 = 1 := by
    simp [magnitudeMatrix]
  have hmat : (1 : Matrix (Fin 1) (Fin 1) ℝ) * ((-(5 / 4) : ℝ) • 1)
      + ((-(5 / 4) : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℝ))ᵀ * 1 = (-(5 / 2) : ℝ) • 1 := by
    rw [Matrix.one_mul, Matrix.transpose_smul, Matrix.transpose_one, Matrix.mul_one, ← add_smul]
    norm_num
  refine ⟨1, 1, (-(5 / 4) : ℝ) • 1, 1, 5 / 4, Matrix.transpose_one, Matrix.transpose_one,
    one_pos, ?_, ?_, by norm_num, ?_⟩
  · intro e
    rw [hmm, Matrix.one_mulVec, one_mul]
  · intro e
    have habs : |(-(5 / 2) : ℝ)| = 5 / 2 := by
      rw [abs_neg]
      exact abs_of_pos (by norm_num)
    rw [hmat, quadForm_smul, Matrix.one_mulVec, abs_mul, habs,
      abs_of_nonneg (dotProduct_self_nonneg e)]
    norm_num
  · intro hall
    have h := hall (fun _ => 1)
    rw [hmm, hmat, quadForm_add, quadForm_smul, quadForm_smul] at h
    simp only [Matrix.one_mulVec] at h
    have hd : (fun _ : Fin 1 => (1 : ℝ)) ⬝ᵥ (fun _ : Fin 1 => (1 : ℝ)) = 1 := by
      simp [dotProduct]
    rw [hd] at h
    norm_num at h

/-! ## FW-1's own conclusion: the ordering under (A4'-q), with no appeal to (A6)

The ordering theorems of the previous section carry (A6) as a hypothesis, and (A6) is *not*
FW-1's hypothesis. §8 states FW-1 under the forward-KL chain of (A4') together with the
remainder-control clause (A4'-q); §2 says so in as many words — "under the forward-KL structure
`δg = H δθ̃ + r` … (A4) reduces to the remainder-control clause". Assumption (A6) enters only in
§11.3, where FW-2' needs the *margin* `s² κ I` rather than the bare ordering, and §11.3 labels
it a genuine empirical assumption. The two theorems below are therefore FW-1 on FW-1's own
hypotheses; `updateCov_ordering_band`, `updateCov_ordering_strict` and
`updateCov_posSemidef_ordering` above are FW-2''s statement, on §11.3's.

Neither hypothesis set implies the other. (A4'-q) does not give (A6): the chain with `r = 0`
makes the coupling term `2 H V H`, which is positive semidefinite but has no positive lower
bound when `H V H` is degenerate. (A6) does not give (A4'-q): (A6) constrains the coupling term
and says nothing at all about the remainder `E[δθ̃ rᵀ]` that (A4'-q) bounds. Both are stated,
and the file does not pass one off as the other.
-/

/-- **FW-1, projected, on FW-1's own hypotheses.** Under the pointwise forward-KL chain
`g = H δθ̃ + r`, the (A4'-q) bounds `λ` below on the magnitude channel and `ρ` on the symmetrized
remainder pairing, and the explicit third-order bound `τ`, the lift's projected diffusion exceeds
the direct baseline's by at least `(s² (3λ − 2ρ) − τ) ‖e‖²`. Assumption (A6) is not used: the
whole of the gain comes from the substituted excess of FW-1b, whose coefficient `3` is
`diffusionExcess_forwardKL_of_chain`. -/
theorem projDiffusion_ordering_of_forwardKL [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dth r R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r) (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω)
    (hchain : ∀ ω, g ω = H *ᵥ dth ω + r ω) {lam rho tau : ℝ}
    (hlam : ∀ e : n → ℝ,
      lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
    (hrho : ∀ e : n → ℝ,
      |e ⬝ᵥ ((H * covMat μ dth r + (covMat μ dth r)ᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    {e : n → ℝ}
    (hT3 : |remainderChannel (updateRemainderMatrix μ s H g dth R) e| ≤ tau * (e ⬝ᵥ e)) :
    projDiffusion μ e Gdir + (s ^ 2 * (3 * lam - 2 * rho) - tau) * (e ⬝ᵥ e)
      ≤ projDiffusion μ e Glift := by
  have hsplit : projDiffusion μ e Glift
      = projDiffusion μ e Gdir
        + e ⬝ᵥ (diffusionExcess s H (covMat μ dth dth) (covMat μ dth g)
            (updateRemainderMatrix μ s H g dth R) *ᵥ e) := by
    rw [projDiffusion, projDiffusion, updateCov_excess_eq s hH hg hθ hR hlift hdir, quadForm_add]
  have hfk : e ⬝ᵥ (diffusionExcess s H (covMat μ dth dth) (covMat μ dth g)
        (updateRemainderMatrix μ s H g dth R) *ᵥ e)
      = s ^ 2 * (e ⬝ᵥ (((3 : ℝ) • magnitudeMatrix H (covMat μ dth dth)
            + (H * covMat μ dth r + (covMat μ dth r)ᵀ * H)) *ᵥ e))
        + remainderChannel (updateRemainderMatrix μ s H g dth R) e := by
    rw [diffusionExcess_forwardKL_of_chain s hH hθ hr hchain, quadForm_add, quadForm_smul,
      remainderChannel]
  have hmul : s ^ 2 * ((3 * lam - 2 * rho) * (e ⬝ᵥ e))
      ≤ s ^ 2 * (e ⬝ᵥ (((3 : ℝ) • magnitudeMatrix H (covMat μ dth dth)
          + (H * covMat μ dth r + (covMat μ dth r)ᵀ * H)) *ᵥ e)) :=
    mul_le_mul_of_nonneg_left (forwardKL_excess_lower_bound hlam hrho e) (sq_nonneg s)
  have hT3' : -(tau * (e ⬝ᵥ e))
      ≤ remainderChannel (updateRemainderMatrix μ s H g dth R) e := (abs_le.mp hT3).1
  have hexp : (s ^ 2 * (3 * lam - 2 * rho) - tau) * (e ⬝ᵥ e)
      = s ^ 2 * ((3 * lam - 2 * rho) * (e ⬝ᵥ e)) - tau * (e ⬝ᵥ e) := by ring
  rw [hsplit, hfk, hexp]
  linarith

/-- **FW-1 in the positive semidefinite order, on FW-1's own hypotheses**: `Σ_lift ⪰ Σ_dir` for
the real update covariances, under the forward-KL chain, (A4'-q) in quadratic-form language, and
a third-order remainder bound `τ` not exceeding the (A4'-q) gain `s² (3λ − 2ρ)`. This is the
conclusion §8 draws, from the hypotheses §8 draws it from; nothing here appeals to (A6).

The extra clause `hA4q` relative to the note is the third-order remainder. §11.3's proof says
"T3 absorbed by (A4'-q)", which is not literally true — (A4'-q) bounds the second-moment
remainder `E[δθ̃ rᵀ]` and says nothing about third moments — so `τ` is carried with its own name
and appears in the hypothesis rather than being dropped. Setting `τ = 0` recovers the note's
statement exactly. -/
theorem updateCov_posSemidef_ordering_of_forwardKL [IsFiniteMeasure μ] (s : ℝ)
    {H : Matrix n n ℝ} (hH : Hᵀ = H) {g dth r R Glift Gdir : Ω → n → ℝ}
    (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth) (hr : SqIntegrableVec μ r)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω)
    (hchain : ∀ ω, g ω = H *ᵥ dth ω + r ω) {lam rho tau : ℝ}
    (hlam : ∀ e : n → ℝ,
      lam * (e ⬝ᵥ e) ≤ e ⬝ᵥ (magnitudeMatrix H (covMat μ dth dth) *ᵥ e))
    (hrho : ∀ e : n → ℝ,
      |e ⬝ᵥ ((H * covMat μ dth r + (covMat μ dth r)ᵀ * H) *ᵥ e)| ≤ 2 * rho * (e ⬝ᵥ e))
    (hT3 : ∀ e : n → ℝ,
      |remainderChannel (updateRemainderMatrix μ s H g dth R) e| ≤ tau * (e ⬝ᵥ e))
    (hA4q : tau ≤ s ^ 2 * (3 * lam - 2 * rho)) :
    (covMat μ Glift Glift - covMat μ Gdir Gdir).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun e => ?_
  · refine Matrix.isHermitian_iff_isSymm.mpr ?_
    show (covMat μ Glift Glift - covMat μ Gdir Gdir)ᵀ
      = covMat μ Glift Glift - covMat μ Gdir Gdir
    rw [Matrix.transpose_sub, covMat_transpose, covMat_transpose]
  · have hstar : (star e : n → ℝ) = e := rfl
    have hsub : e ⬝ᵥ ((covMat μ Glift Glift - covMat μ Gdir Gdir) *ᵥ e)
        = projDiffusion μ e Glift - projDiffusion μ e Gdir := by
      rw [projDiffusion, projDiffusion, Matrix.sub_mulVec, dotProduct_sub]
    have hbound := projDiffusion_ordering_of_forwardKL s hH hg hθ hr hR hlift hdir hchain
      hlam hrho (hT3 e)
    have hgain : 0 ≤ (s ^ 2 * (3 * lam - 2 * rho) - tau) * (e ⬝ᵥ e) :=
      mul_nonneg (by linarith) (dotProduct_self_nonneg e)
    rw [hstar, hsub]
    linarith

/-! ## The §10 corollary: the magnitude channel alone still orders the diffusions -/

/-- With no cross-covariance the coupling term is identically zero. This is exactly the
matched-marginal cell of E1: injected noise independent of the batch makes
`Σ_slack = E[δθ̃ δgᵀ] = 0`. -/
@[simp]
theorem couplingTerm_zero (H : Matrix n n ℝ) :
    couplingTerm H (0 : Matrix n n ℝ) = 0 := by
  simp [couplingTerm]

/-- **The §10 corollary.** With the coupling channel absent and the magnitude channel kept, the
two diffusions are still ordered: the magnitude channel alone can only add diffusion.

The empirical caveat is not a footnote. E1 (30 seeds, and the σ bracket of §11.0 at 1×/3×/10×
injected scale) **refuted** the claim that this ordering suffices for the observed effect.
Matched-marginal noise realizes exactly this hypothesis — `Σ_slack = 0`, `V` matched — and is
statistically indistinguishable from no noise at every scale tested, while the coupled latent weight
wins. The ordering here is therefore necessary for the mechanism and demonstrably not sufficient
for it. §11.0 gives the reason: the coupling channel is first order in the perturbation scale
and the magnitude channel second, so a second-order channel cannot reproduce a first-order
effect without being inflated to the square root of the lift's scale. -/
theorem magnitude_channel_ordering {Slift Sdir : Matrix n n ℝ} {s : ℝ} {H V : Matrix n n ℝ}
    (hsplit : Slift = Sdir + diffusionExcess s H V 0 0)
    (hV : V.PosSemidef) (hH : H.IsHermitian) (e : n → ℝ) :
    e ⬝ᵥ (Sdir *ᵥ e) ≤ e ⬝ᵥ (Slift *ᵥ e) := by
  have hsplit' : e ⬝ᵥ (Slift *ᵥ e)
      = e ⬝ᵥ (Sdir *ᵥ e) + couplingChannel s H 0 e + magnitudeChannel s H V e
        + remainderChannel (0 : Matrix n n ℝ) e := by
    rw [hsplit, quadForm_add, excess_channel_split]
    ring
  have hzero : couplingChannel s H 0 e = 0 := by simp [couplingChannel, crossTerm]
  have hmag : 0 ≤ magnitudeChannel s H V e := magnitudeChannel_nonneg hV hH s e
  have hrem : remainderChannel (0 : Matrix n n ℝ) e = 0 := by simp [remainderChannel]
  linarith [hsplit', hzero, hmag, hrem]

/-- The §10 corollary with the third-order remainder carried explicitly: if it is dominated in
modulus by the magnitude channel it corrects, the ordering survives. -/
theorem magnitude_channel_ordering_of_dominated {Slift Sdir : Matrix n n ℝ} {s : ℝ}
    {H V T3 : Matrix n n ℝ} (hsplit : Slift = Sdir + diffusionExcess s H V 0 T3) {e : n → ℝ}
    (hT3 : |remainderChannel T3 e| ≤ magnitudeChannel s H V e) :
    e ⬝ᵥ (Sdir *ᵥ e) ≤ e ⬝ᵥ (Slift *ᵥ e) := by
  have hsplit' : e ⬝ᵥ (Slift *ᵥ e)
      = e ⬝ᵥ (Sdir *ᵥ e) + couplingChannel s H 0 e + magnitudeChannel s H V e
        + remainderChannel T3 e := by
    rw [hsplit, quadForm_add, excess_channel_split]
    ring
  have hzero : couplingChannel s H 0 e = 0 := by simp [couplingChannel, crossTerm]
  have hrem : -magnitudeChannel s H V e ≤ remainderChannel T3 e := (abs_le.mp hT3).1
  linarith [hsplit', hzero, hrem]

/-- **The §10 corollary for the real update covariances.** The two abstract statements above
take the split as a hypothesis; here it is derived. If the injected perturbation is uncorrelated
with the gradient — `Cov[δθ̃, g] = 0`, which is exactly E1's matched-marginal cell, where the
noise is drawn independently of the batch — then the coupling channel of
`projDiffusion_excess_split` vanishes and the lifted diffusion still dominates the direct one,
by the magnitude channel alone, whenever the remainder channel is dominated by the magnitude
channel it perturbs. This is the ordering that E1 measured and found insufficient for the
observed effect: it holds, and matched noise still does nothing, which is §11.0's finding that
the magnitude channel is inert. -/
theorem updateCov_ordering_of_slack_zero [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dth) (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) (hslack : covMat μ dth g = 0) {e : n → ℝ}
    (hT3 : |remainderChannel (updateRemainderMatrix μ s H g dth R) e|
      ≤ magnitudeChannel s H (covMat μ dth dth) e) :
    projDiffusion μ e Gdir ≤ projDiffusion μ e Glift := by
  have hsplit := projDiffusion_excess_split s hH hg hθ hR hlift hdir e
  have hzero : couplingChannel s H (covMat μ dth g) e = 0 := by
    rw [hslack]
    simp [couplingChannel, crossTerm]
  have hrem := (abs_le.mp hT3).1
  linarith [hsplit, hzero, hrem]

/-! ## The direct-baseline endpoint: no latent-weight fluctuation, no excess -/

/-- **The direct-baseline endpoint, channel form (§2).** The direct baseline has `δθ̃ = 0`; both
named channels then vanish identically, for every direction, every positivity-map prefactor and every
curvature. The structural asymmetry between the lift and the direct parameterization is
therefore not an artifact of the ordering hypotheses. -/
theorem channels_eq_zero_of_no_jitter {dth g : Ω → n → ℝ} (hθ0 : ∀ ω, dth ω = 0) (s : ℝ)
    (H : Matrix n n ℝ) (e : n → ℝ) :
    couplingChannel s H (covMat μ dth g) e = 0 ∧
      magnitudeChannel s H (covMat μ dth dth) e = 0 := by
  have h : dth = fun _ : Ω => (0 : n → ℝ) := funext hθ0
  subst h
  rw [covMat_zero_left μ g, covMat_zero_left μ (fun _ : Ω => (0 : n → ℝ))]
  exact ⟨by simp [couplingChannel, crossTerm],
    by simp [magnitudeChannel, magnitudeMatrix]⟩

/-- **The direct-baseline endpoint.** With `δθ̃ = 0` and no remainder, the lifted and the direct
update covariances coincide exactly. The ordering of `updateCov_ordering_strict` is therefore not
vacuous: it is strict only when the latent weight genuinely fluctuates. -/
theorem updateCov_eq_of_no_jitter [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ} (hH : Hᵀ = H)
    {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g) (hθ : SqIntegrableVec μ dth)
    (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) (hθ0 : ∀ ω, dth ω = 0) (hR0 : ∀ ω, R ω = 0) :
    covMat μ Glift Glift = covMat μ Gdir Gdir := by
  have hd : dth = fun _ : Ω => (0 : n → ℝ) := funext hθ0
  have hr : R = fun _ : Ω => (0 : n → ℝ) := funext hR0
  subst hd
  subst hr
  rw [updateCov_excess_eq s hH hg hθ hR hlift hdir]
  simp [diffusionExcess, updateRemainderMatrix, couplingTerm, magnitudeMatrix,
    covMat_zero_left, covMat_zero_right]

/-- **Strictness forces a fluctuating latent weight — in the remainder-free model.** The
contrapositive of the endpoint: under the exact first-order expansion (`R ≡ 0`, hypothesis
`hR0`), if the lifted diffusion strictly exceeds the direct one in some direction, the latent weight
must fluctuate with the batch somewhere. The hypothesis `hR0` is not decorative: with a nonzero
Taylor remainder the strict ordering could be produced by `R` alone, with `δθ̃ ≡ 0`, so the
necessity claim is genuinely a claim about the remainder-free model. This is the sense in which
the ordering is "strict exactly when the latent weight genuinely fluctuates", and it mirrors the
finite-sample necessity statement of `CrossCov.lean`. -/
theorem jitter_necessary_for_strict_ordering [IsFiniteMeasure μ] (s : ℝ) {H : Matrix n n ℝ}
    (hH : Hᵀ = H) {g dth R Glift Gdir : Ω → n → ℝ} (hg : SqIntegrableVec μ g)
    (hθ : SqIntegrableVec μ dth) (hR : SqIntegrableVec μ R)
    (hlift : ∀ ω, Glift ω = s • g ω + s • (H *ᵥ dth ω) + R ω)
    (hdir : ∀ ω, Gdir ω = s • g ω) (hR0 : ∀ ω, R ω = 0) {e : n → ℝ}
    (hstrict : projDiffusion μ e Gdir < projDiffusion μ e Glift) :
    ∃ ω, dth ω ≠ 0 := by
  by_contra hcon
  have hθ0 : ∀ ω, dth ω = 0 := by
    intro ω
    by_contra hne
    exact hcon ⟨ω, hne⟩
  have heq := updateCov_eq_of_no_jitter s hH hg hθ hR hlift hdir hθ0 hR0
  rw [projDiffusion, projDiffusion, heq] at hstrict
  exact lt_irrefl _ hstrict

/-! ## The ordering hypotheses are jointly satisfiable: a non-degenerate instance

The ordering theorems above are conditional on (A6) for the *real* cross-covariance
`covMat μ δθ̃ g` together with a remainder bound, and `CouplingSign.coupling_sign_not_automatic`
shows that (A6) can fail; a referee is entitled to ask whether the full hypothesis set of
`updateCov_ordering_strict` — measure-theoretic covariances included — is ever satisfied at
all. `UpdateCovariance`'s standard-Gaussian witness answers it: the batch law is the standard
Gaussian on `ℝ`, gradient and latent-weight fluctuation are both the coordinate (perfect positive
coupling), the curvature is `H = 2`, the prefactor `s = 1` and the remainder is identically
zero. There (A6) holds with `κ = 4` and equality, the remainder bound holds with `τ = 0`, and
the strict ordering delivered by the theorem is `1 < 9`, consistent with the band margin
`1 + (s²κ − τ)‖e‖² = 5 ≤ 9`. -/

open StdGaussianWitness in
/-- **`updateCov_ordering_strict` is not vacuous.** On the standard-Gaussian witness instance
of `UpdateCovariance.lean`, every hypothesis of the strict ordering theorem is discharged with
nothing degenerate: (A6) holds for the real cross-covariance with `κ = 4` (first conjunct), the
remainder group vanishes identically so `τ = 0` works (second conjunct), the two projected
diffusions take the nonzero values `1` and `9` (third and fourth conjuncts, computed
independently of the ordering theorems), and the strict inequality between them is obtained by
*applying* `updateCov_ordering_strict`, not by comparing the computed values.

The instance's one limitation is stated rather than implied: the Taylor remainder is identically
zero, so this is a witness at `τ = 0` and it does not exercise a nonzero third-order bound. It
does exercise (A6) non-trivially, at `κ = 4` and with equality. -/
theorem updateCov_ordering_strict_nonvacuous :
    ((4 : ℝ) * (dir ⬝ᵥ dir) ≤ crossTerm hessian (covMat batchLaw jitter grad) dir)
    ∧ updateRemainderMatrix batchLaw 1 hessian grad jitter (fun _ => (0 : Fin 1 → ℝ)) = 0
    ∧ projDiffusion batchLaw dir grad = 1
    ∧ projDiffusion batchLaw dir update = 9
    ∧ projDiffusion batchLaw dir grad < projDiffusion batchLaw dir update := by
  have hcov : covMat batchLaw jitter grad = fun _ _ => (1 : ℝ) := by
    ext i j
    exact stdGaussianWitness_covariance_coord
  have hcross : crossTerm hessian (covMat batchLaw jitter grad) dir = 4 := by
    rw [crossTerm, couplingTerm, hcov]
    simp [dotProduct, Matrix.mulVec, Matrix.mul_apply, Matrix.transpose_apply, dir, hessian]
    norm_num
  have halign : (4 : ℝ) * (dir ⬝ᵥ dir)
      ≤ crossTerm hessian (covMat batchLaw jitter grad) dir := by
    rw [stdGaussianWitness_dir_unit, hcross]
    norm_num
  have hrem0 : updateRemainderMatrix batchLaw 1 hessian grad jitter
      (fun _ => (0 : Fin 1 → ℝ)) = 0 := by
    rw [updateRemainderMatrix]
    simp only [covMat_zero_right, covMat_zero_left]
    simp
  have hT3 : |remainderChannel (updateRemainderMatrix batchLaw 1 hessian grad jitter
      (fun _ => (0 : Fin 1 → ℝ))) dir| ≤ 0 * (dir ⬝ᵥ dir) := by
    rw [hrem0]
    simp [remainderChannel]
  have hR0 : SqIntegrableVec batchLaw (fun _ : ℝ => (0 : Fin 1 → ℝ)) :=
    fun _ => memLp_const 0
  have hlift : ∀ ω : ℝ, update ω = (1 : ℝ) • grad ω + (1 : ℝ) • (hessian *ᵥ jitter ω)
      + (fun _ : ℝ => (0 : Fin 1 → ℝ)) ω := by
    intro ω
    rw [stdGaussianWitness_expansion ω]
    simp
  have hdir : ∀ ω : ℝ, grad ω = (1 : ℝ) • grad ω := fun ω => (one_smul ℝ (grad ω)).symm
  have hdirval : projDiffusion batchLaw dir grad = 1 := by
    rw [projDiffusion]
    exact stdGaussianWitness_objective_term
  exact ⟨halign, hrem0, hdirval, stdGaussianWitness_projDiffusion,
    updateCov_ordering_strict 1 stdGaussianWitness_hessian_symm
      stdGaussianWitness_grad_sqIntegrable stdGaussianWitness_jitter_sqIntegrable hR0
      hlift hdir halign hT3 (by norm_num)
      (fun hzero => one_ne_zero (congrFun hzero 0))⟩

open StdGaussianWitness in
/-- **`updateCov_posSemidef_ordering_of_forwardKL` is not vacuous, and its bound is attained.**
FW-1's own hypothesis set — the pointwise forward-KL chain together with (A4'-q) — is a
different set from (A6), so its satisfiability has to be checked separately, and with a
*nonzero* remainder: were `r ≡ 0` the pairing bound would hold at `ρ = 0` and (A4'-q) would
carry no content. On the standard-Gaussian witness of `UpdateCovariance.lean` the chain
`g = H δθ̃ + r` forces `r = −δθ̃`, which is as large as `δθ̃` itself; the second-moment remainder
is then `E[δθ̃ rᵀ] = −1` (third conjunct), and the (A4'-q) constants come out `λ = 4` (second
conjunct, an equality, so `λ` is sharp) and `ρ = 2`, satisfying `2ρ = 4 ≤ 12 = 3λ` with room to
spare. The theorem's conclusion is the fourth conjunct.

The fifth conjunct is the sharpness check: the margin the theorem guarantees,
`s²(3λ − 2ρ) − τ = 8`, is exactly the excess the witness actually has, `9 − 1 = 8`. So on this
instance the (A4'-q) route loses nothing at all — the substituted coefficient `3` of §8, and not
the `2` of the §7 sketch, is what makes the two sides agree.

One limitation of the instance is stated rather than implied: like
`updateCov_ordering_strict_nonvacuous`, it takes the Taylor remainder `R ≡ 0`, so it is a
witness at `τ = 0` and does not exercise a nonzero third-order bound. -/
theorem updateCov_posSemidef_ordering_of_forwardKL_nonvacuous :
    (∀ ω : ℝ, grad ω = hessian *ᵥ jitter ω + (-1 : ℝ) • grad ω)
    ∧ (∀ e : Fin 1 → ℝ,
        e ⬝ᵥ (magnitudeMatrix hessian (covMat batchLaw jitter jitter) *ᵥ e) = 4 * (e ⬝ᵥ e))
    ∧ (covMat batchLaw jitter (fun ω : ℝ => (-1 : ℝ) • grad ω) = fun _ _ => (-1 : ℝ))
    ∧ (covMat batchLaw update update - covMat batchLaw grad grad).PosSemidef
    ∧ projDiffusion batchLaw dir grad
          + ((1 : ℝ) ^ 2 * (3 * 4 - 2 * 2) - 0) * (dir ⬝ᵥ dir)
        = projDiffusion batchLaw dir update := by
  have hchain : ∀ ω : ℝ, grad ω = hessian *ᵥ jitter ω + (-1 : ℝ) • grad ω := by
    intro ω
    funext i
    simp [grad, jitter, hessian, Matrix.mulVec, dotProduct]
    ring
  have hcovjg : covMat batchLaw jitter grad = fun _ _ => (1 : ℝ) := by
    ext i j
    exact stdGaussianWitness_covariance_coord
  have hcovjj : covMat batchLaw jitter jitter = fun _ _ => (1 : ℝ) := by
    ext i j
    exact stdGaussianWitness_covariance_coord
  have hcovjr : covMat batchLaw jitter (fun ω : ℝ => (-1 : ℝ) • grad ω)
      = fun _ _ => (-1 : ℝ) := by
    rw [covMat_smul_right, hcovjg]
    ext i j
    simp
  have hq : ∀ e : Fin 1 → ℝ,
      e ⬝ᵥ (magnitudeMatrix hessian (covMat batchLaw jitter jitter) *ᵥ e) = 4 * (e ⬝ᵥ e) := by
    intro e
    rw [magnitudeMatrix, hcovjj]
    simp [dotProduct, Matrix.mulVec, Matrix.mul_apply, hessian]
    ring
  have hp : ∀ e : Fin 1 → ℝ,
      e ⬝ᵥ ((hessian * covMat batchLaw jitter (fun ω : ℝ => (-1 : ℝ) • grad ω)
          + (covMat batchLaw jitter (fun ω : ℝ => (-1 : ℝ) • grad ω))ᵀ * hessian) *ᵥ e)
        = -4 * (e ⬝ᵥ e) := by
    intro e
    rw [hcovjr]
    simp [dotProduct, Matrix.mulVec, Matrix.mul_apply, Matrix.transpose_apply, hessian]
    ring
  have hrho : ∀ e : Fin 1 → ℝ,
      |e ⬝ᵥ ((hessian * covMat batchLaw jitter (fun ω : ℝ => (-1 : ℝ) • grad ω)
          + (covMat batchLaw jitter (fun ω : ℝ => (-1 : ℝ) • grad ω))ᵀ * hessian) *ᵥ e)|
        ≤ 2 * 2 * (e ⬝ᵥ e) := by
    intro e
    rw [hp e, abs_mul, abs_of_nonneg (dotProduct_self_nonneg e)]
    have habs : |(-4 : ℝ)| = 4 := by norm_num
    rw [habs]
    linarith [dotProduct_self_nonneg e]
  have hR0 : SqIntegrableVec batchLaw (fun _ : ℝ => (0 : Fin 1 → ℝ)) :=
    fun _ => memLp_const 0
  have hlift : ∀ ω : ℝ, update ω = (1 : ℝ) • grad ω + (1 : ℝ) • (hessian *ᵥ jitter ω)
      + (fun _ : ℝ => (0 : Fin 1 → ℝ)) ω := by
    intro ω
    rw [stdGaussianWitness_expansion ω]
    simp
  have hdir : ∀ ω : ℝ, grad ω = (1 : ℝ) • grad ω := fun ω => (one_smul ℝ (grad ω)).symm
  have hrem0 : updateRemainderMatrix batchLaw 1 hessian grad jitter
      (fun _ => (0 : Fin 1 → ℝ)) = 0 := by
    rw [updateRemainderMatrix]
    simp only [covMat_zero_right, covMat_zero_left]
    simp
  have hT3 : ∀ e : Fin 1 → ℝ,
      |remainderChannel (updateRemainderMatrix batchLaw 1 hessian grad jitter
        (fun _ => (0 : Fin 1 → ℝ))) e| ≤ 0 * (e ⬝ᵥ e) := by
    intro e
    rw [hrem0]
    simp [remainderChannel]
  have hpsd := updateCov_posSemidef_ordering_of_forwardKL (μ := batchLaw) 1
    stdGaussianWitness_hessian_symm stdGaussianWitness_grad_sqIntegrable
    stdGaussianWitness_jitter_sqIntegrable
    (SqIntegrableVec.smul (-1) stdGaussianWitness_grad_sqIntegrable) hR0 hlift hdir hchain
    (fun e => (hq e).ge) hrho hT3 (by norm_num)
  refine ⟨hchain, hq, hcovjr, hpsd, ?_⟩
  have hdirval : projDiffusion batchLaw dir grad = 1 := by
    rw [projDiffusion]
    exact stdGaussianWitness_objective_term
  rw [hdirval, stdGaussianWitness_projDiffusion, stdGaussianWitness_dir_unit]
  norm_num

end IcnnLift
