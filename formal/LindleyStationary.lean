import Mathlib
import ProjectionClamp

/-!
# The stationary law of the Lindley recursion

Proposition `prop:clamp` of the paper models one constrained
coordinate under projected stochastic gradient descent, held at the boundary of the cone:

    W_{k+1} = max(0, W_k - η(γ + σ ζ_k)),   W_0 = 0,   ζ_0, ζ_1, … i.i.d. N(0, 1),   ρ = γ/σ.

This is `clampIterate c ρ` at scale `c = ησ` (`clampIterate_succ_paper` in
`ProjectionClamp.lean`). That file proves the dwell, the clamped fraction at each step, the
reversal identity `W_k = max_{j≤k} S_j` in distribution with `S_j = -c ∑_{i<j}(ρ + ζ_i)`
(`walkSum`), that `M = sup_j S_j` is finite with finite mean, and that `W_k → M` in
distribution from `W_0 = 0`. It left one sentence of the proposition as a citation:

> The law of `M` is the recursion's unique stationary law.

This file proves that sentence, for every `c > 0` and `ρ > 0`, and with it two facts the
sentence leans on: the iterates converge to the law of `M` from every starting law, and the
clamped fraction `P(W_k = 0)` decreases to the mass that law puts at `0`.

## How the statement is read

A law is stationary when one step of the recursion, applied to an iterate with that law and a
fresh independent draw, gives back the same law. `lindleyStep c ρ ν` is that step on laws: the
law of `max(0, W - c(ρ + ζ))` with `W ~ ν` independent of `ζ ~ N(0, 1)`, written as the image of
`ν ⊗ N(0, 1)`. `lindleyLimitLaw c ρ` is the law of `M`, built on the canonical space of i.i.d.
standard normal sequences (`Measure.infinitePi`, abbreviated `lindleyDraws`). "Unique" is among
probability measures on `ℝ`; nothing is assumed about where a candidate law puts its mass.

Two bridges tie these objects to the recursion as `ProjectionClamp.lean` states it, on any
probability space that carries i.i.d. standard normal draws, with the hypotheses written exactly
as there. The law of `M` on that space is `lindleyLimitLaw` (`map_iSup_walkSum_eq_lindleyLimitLaw`),
and the law of `W_k` is `lindleyStep` applied `k` times to the point mass at `0`
(`map_clampIterate_eq_iterate_lindleyStep`).

## The proofs

**Stationarity** (`lindleyStep_lindleyLimitLaw`). Splitting off the first draw gives
`S_{j+1}(z) = S_j(z ∘ succ) - c(ρ + z_0)`, so wherever the walk is bounded above,
`M(z) = max(0, M(z ∘ succ) - c(ρ + z_0))` (`iSup_walkSum_eq_lindley`), and the walk is bounded
above almost surely (`ae_bddAbove_walkSum`). Under the i.i.d. law the first draw is independent
of the rest, and the rest are again i.i.d. standard normal (`map_head_tail_lindleyDraws`), so the
right-hand side has law `lindleyStep (law of M)`.

**Uniqueness** (`eq_lindleyLimitLaw_of_lindleyStep_eq`). It goes through convergence from every
starting law, which is stronger and is stated as a theorem of its own
(`tendsto_integral_iterate_lindleyStep`).
* With `W_0 ~ ν` independent of the draws, the law of `W_k` is `lindleyStep` applied `k` times
  to `ν` (`map_lindleyFrom_eq_iterate`), by taking the first step first.
* From a start `x ≥ 0`, pathwise, `W_k = max(x + S_k, W_k⁰)` with `W_k⁰` the iterate from `0`
  (`lindleyFrom_eq_max`). Reversing the first `k` draws, which does not change their law, turns
  this into `max(x + S_k, max_{j≤k} S_j)` (`lindleyFrom_revDraws`).
* The walk drifts to `-∞` almost surely, since it is the walk at drift `ρ/2`, which is bounded
  above, minus `c(ρ/2)k` (`ae_lindleyDraws_eventually_add_walkSum_nonpos`). The first term is
  therefore eventually below the second, the running maximum increases to `M`, and dominated
  convergence gives convergence of the integrals of every bounded continuous function.
* A starting law with mass below `0` is handled by one step, after which the law sits on
  `[0, ∞)`.

A stationary `ν` has the same integrals at every step, so they equal the integrals against the
law of `M`, and two finite measures on `ℝ` with the same integrals of bounded continuous functions
are equal (`ext_of_forall_integral_eq_of_IsFiniteMeasure`). No strong law, no Markov-chain
theory and no fluctuation theory is used: the probabilistic inputs are the Chernoff and
Borel--Cantelli bound `ae_bddAbove_walkSum` and the reversal identity, both from
`ProjectionClamp.lean`, and the independence of the first draw from the rest.

**The clamped fraction at the limit.** `ProjectionClamp.lean` records that convergence of
`P(W_k = 0)` does not follow from convergence in distribution, because the limit law has an atom
at `0`. It is proved here with monotone events. `W_k = 0` has the probability of the event
`{S_j ≤ 0 for all j ≤ k}` (`measureReal_clampIterate_eq_zero_eq_forall_walkSum_le`), and these
events decrease in `k`, so `P(W_k = 0)` is non-increasing
(`antitone_measureReal_clampIterate_eq_zero`) and tends, by continuity from above, to
`P(S_j ≤ 0 for all j)` (`tendsto_measureReal_clampIterate_eq_zero`). That limit is the mass of
the stationary law at `0` (`lindleyLimitLaw_real_zero`).

## What is not here

The paper's value of that mass, `exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))`, is Spitzer's identity and is not
proved in this file. `lindleyLimitLaw_real_zero` is the bridge such a proof needs: any theorem
that `P(S_j ≤ 0 for all j)` equals the series becomes, through it, a theorem about the
stationary law and, through `tendsto_measureReal_clampIterate_eq_zero`, about the limit of the
clamped fraction.

## Results
* `lindleyStep`, `lindleyLimitLaw`, `lindleyDraws` — one step on laws, the law of `M`, and the
  i.i.d. standard normal law on sequences.
* `map_draws_eq_lindleyDraws`, `lindleyDraws_indepFun_head_tail`, `map_head_tail_lindleyDraws` —
  the draws of any i.i.d. sequence have the canonical law, and under it the first draw is
  independent of the rest, which are again i.i.d.
* `lindleyFrom`, `map_firstStep_lindleyDraws`, `map_lindleyFrom_eq_iterate` — the recursion from
  any start, and its law after `k` steps.
* `iSup_walkSum_eq_lindley`, `lindleyStep_lindleyLimitLaw` — **stationarity**.
* `map_iSup_walkSum_eq_lindleyLimitLaw`, `map_clampIterate_eq_iterate_lindleyStep`,
  `lindleyStep_map_iSup_walkSum` — the bridges to the recursion on any probability space.
* `lindleyFrom_eq_max`, `lindleyFrom_revDraws`, `ae_lindleyDraws_eventually_add_walkSum_nonpos`,
  `tendsto_integral_iterate_lindleyStep_of_nonneg`, `tendsto_integral_iterate_lindleyStep` —
  **`W_k → M` in distribution from every starting law**.
* `eq_lindleyLimitLaw_of_lindleyStep_eq`, `lindleyStep_eq_self_iff` — **uniqueness**.
* `lindleyLimitLaw_real_zero`, `measureReal_clampIterate_eq_zero_eq_forall_walkSum_le`,
  `antitone_measureReal_clampIterate_eq_zero`, `tendsto_measureReal_clampIterate_eq_zero` — **the
  clamped fraction** decreases to the stationary mass at `0`, which is `P(S_j ≤ 0 for all j)`;
  `iterate_lindleyStep_real_zero_eq`, `antitone_iterate_lindleyStep_real_zero`,
  `tendsto_iterate_lindleyStep_real_zero` state the same on the laws alone.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each main result lists only
`propext`, `Classical.choice`, `Quot.sound`. Convergence in distribution is stated, as in
`ProjectionClamp.lean`, in its defining form: integrals against bounded continuous test functions
converge. The hypotheses `c > 0` and `ρ > 0` are used throughout. That the recursion has no
stationary law at `ρ ≤ 0`, where the walk is not bounded above, is the classical counterpart and
is not proved here.
-/

open MeasureTheory Filter Set ProbabilityTheory
open scoped Topology ENNReal NNReal

namespace IcnnLift

/-- One step of the recursion, acting on laws: the law of `max(0, W - c(ρ + ζ))` when `W ~ ν`
is independent of `ζ ~ N(0, 1)`. A law `ν` is stationary when `lindleyStep c ρ ν = ν`. -/
noncomputable def lindleyStep (c rho : ℝ) (ν : Measure ℝ) : Measure ℝ :=
  (ν.prod (gaussianReal 0 1)).map (fun p => max 0 (p.1 - c * (rho + p.2)))

/-- The law of `M = sup_j S_j` on the canonical space of i.i.d. standard normal draws. -/
noncomputable def lindleyLimitLaw (c rho : ℝ) : Measure ℝ :=
  (Measure.infinitePi (fun _ : ℕ => gaussianReal 0 1)).map (fun z => ⨆ j, walkSum c rho z j)

/-- The canonical law of the draws: i.i.d. standard normal, on sequences `ℕ → ℝ`. -/
noncomputable abbrev lindleyDraws : Measure (ℕ → ℝ) :=
  Measure.infinitePi (fun _ : ℕ => gaussianReal 0 1)

/-- The one-step map `(w, ζ) ↦ max(0, w - c(ρ + ζ))` is measurable. -/
theorem measurable_lindleyMap (c rho : ℝ) :
    Measurable (fun p : ℝ × ℝ => max 0 (p.1 - c * (rho + p.2))) := by
  fun_prop

private theorem measurable_iSup_walkSum (c rho : ℝ) :
    Measurable (fun z : ℕ → ℝ => ⨆ j, walkSum c rho z j) :=
  Measurable.iSup fun j => measurable_walkSum_apply c rho j

/-- Under the canonical law the coordinates are independent. -/
theorem lindleyDraws_iIndepFun : iIndepFun (fun i (z : ℕ → ℝ) => z i) lindleyDraws :=
  iIndepFun_infinitePi (X := fun _ x => x) (fun _ => measurable_id)

/-- Under the canonical law each coordinate is standard normal. -/
theorem lindleyDraws_hasLaw (i : ℕ) :
    HasLaw (fun z : ℕ → ℝ => z i) (gaussianReal 0 1) lindleyDraws :=
  ⟨(measurable_pi_apply i).aemeasurable, Measure.infinitePi_map_eval _ i⟩

/-- One step maps probability laws to probability laws. -/
theorem isProbabilityMeasure_lindleyStep (c rho : ℝ) (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    IsProbabilityMeasure (lindleyStep c rho ν) :=
  Measure.isProbabilityMeasure_map (measurable_lindleyMap c rho).aemeasurable

/-- The law of `M` is a probability law: `M` is measurable, `⨆` taking the junk value `0` on
paths where the walk is unbounded. -/
theorem isProbabilityMeasure_lindleyLimitLaw (c rho : ℝ) :
    IsProbabilityMeasure (lindleyLimitLaw c rho) :=
  Measure.isProbabilityMeasure_map (measurable_iSup_walkSum c rho).aemeasurable

/-- The shifted draws `z ∘ succ` are again i.i.d. standard normal. -/
theorem map_shift_lindleyDraws :
    lindleyDraws.map (fun z n => z (n + 1)) = lindleyDraws :=
  Measure.map_infinitePi_infinitePi_of_inj (P := fun _ : ℕ => gaussianReal 0 1)
    (f := fun n => n + 1) (fun _ _ h => Nat.succ_injective h)

/-- **The first draw is independent of the rest.** -/
theorem lindleyDraws_indepFun_head_tail :
    IndepFun (fun z : ℕ → ℝ => z 0) (fun z n => z (n + 1)) lindleyDraws := by
  have hiI : iIndep (fun i => MeasurableSpace.comap (fun z : ℕ → ℝ => z i) inferInstance)
      lindleyDraws :=
    (iIndepFun_iff_iIndep _ _ _).1 lindleyDraws_iIndepFun
  have hle : ∀ i, MeasurableSpace.comap (fun z : ℕ → ℝ => z i) (inferInstance : MeasurableSpace ℝ)
      ≤ (inferInstance : MeasurableSpace (ℕ → ℝ)) :=
    fun i => measurable_iff_comap_le.1 (measurable_pi_apply i)
  have hdisj : Disjoint ({0} : Set ℕ) (Set.Ici 1) := by
    rw [Set.disjoint_singleton_left]
    simp
  have hInd := indep_iSup_of_disjoint hle hiI hdisj
  rw [IndepFun_iff_Indep]
  refine indep_of_indep_of_le hInd ?_ ?_
  · exact le_iSup₂_of_le (f := fun i (_ : i ∈ ({0} : Set ℕ)) =>
      MeasurableSpace.comap (fun z : ℕ → ℝ => z i) inferInstance) 0 rfl le_rfl
  · rw [← measurable_iff_comap_le]
    refine @measurable_pi_lambda _ _ _ (⨆ i ∈ Set.Ici 1,
      MeasurableSpace.comap (fun z : ℕ → ℝ => z i) inferInstance) _ _ fun n => ?_
    rw [measurable_iff_comap_le]
    exact le_iSup₂_of_le (f := fun i (_ : i ∈ Set.Ici 1) =>
      MeasurableSpace.comap (fun z : ℕ → ℝ => z i) inferInstance) (n + 1)
      (by simp) le_rfl

/-- **The split of the draws into the first and the rest**: under the i.i.d. law, the pair
`(z₀, z ∘ succ)` has law `N(0,1) ⊗ (i.i.d. law)`. -/
theorem map_head_tail_lindleyDraws :
    lindleyDraws.map (fun z : ℕ → ℝ => (z 0, fun n => z (n + 1)))
      = (gaussianReal 0 1).prod lindleyDraws := by
  rw [lindleyDraws_indepFun_head_tail.map_prod_eq_prod_map_map (measurable_pi_apply 0).aemeasurable
    (measurable_pi_lambda _ fun n => measurable_pi_apply (n + 1)).aemeasurable,
    map_shift_lindleyDraws, Measure.infinitePi_map_eval _ 0]

/-- The block of draws of any i.i.d. standard-normal sequence has the canonical law. -/
theorem map_draws_eq_lindleyDraws {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) :
    P.map (fun ω i => zeta i ω) = lindleyDraws := by
  rw [(iIndepFun_iff_map_fun_eq_infinitePi_map hmeas).mp hindep]
  congr 1
  funext i
  exact (hlaw i).map_eq

/-! ### The recursion from an arbitrary starting point -/

/-- The recursion started at `W₀ = x`. -/
noncomputable def lindleyFrom (c rho x : ℝ) (z : ℕ → ℝ) : ℕ → ℝ
  | 0 => x
  | k + 1 => max 0 (lindleyFrom c rho x z k - c * (rho + z k))

/-- From `W₀ = 0` this is the recursion of `ProjectionClamp.lean`. -/
theorem lindleyFrom_zero_eq_clampIterate (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    lindleyFrom c rho 0 z k = clampIterate c rho z k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [lindleyFrom, clampIterate_succ, ih]

/-- The first step taken first: `k + 1` steps from `x` are `k` steps from `W₁`, driven by the
remaining draws. -/
theorem lindleyFrom_succ_shift (c rho x : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    lindleyFrom c rho x z (k + 1)
      = lindleyFrom c rho (max 0 (x - c * (rho + z 0))) (fun n => z (n + 1)) k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [lindleyFrom, ih]; rfl

/-- The iterate is a measurable function of the start and the draws. -/
theorem measurable_lindleyFrom (c rho : ℝ) (k : ℕ) :
    Measurable (fun p : ℝ × (ℕ → ℝ) => lindleyFrom c rho p.1 p.2 k) := by
  induction k with
  | zero => exact measurable_fst
  | succ k ih =>
    simp only [lindleyFrom]
    have h2 : Measurable (fun p : ℝ × (ℕ → ℝ) => c * (rho + p.2 k)) := by fun_prop
    exact measurable_const.max (ih.sub h2)

/-- **One step on laws.** If `W₀ ~ ν` is independent of the draws, then `(W₁, z ∘ succ)` has law
`lindleyStep ν ⊗ (i.i.d. law)`. -/
theorem map_firstStep_lindleyDraws (c rho : ℝ) (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    (ν.prod lindleyDraws).map
        (fun p : ℝ × (ℕ → ℝ) => (max 0 (p.1 - c * (rho + p.2 0)), fun n => p.2 (n + 1)))
      = (lindleyStep c rho ν).prod lindleyDraws := by
  have hsplit : Measurable (fun z : ℕ → ℝ => (z 0, fun n => z (n + 1))) := by fun_prop
  have h1 : (ν.prod lindleyDraws).map (Prod.map id (fun z : ℕ → ℝ => (z 0, fun n => z (n + 1))))
      = ν.prod ((gaussianReal 0 1).prod lindleyDraws) := by
    rw [← Measure.map_prod_map _ _ measurable_id hsplit, Measure.map_id, map_head_tail_lindleyDraws]
  have h2 : (ν.prod ((gaussianReal 0 1).prod lindleyDraws)).map
      (MeasurableEquiv.prodAssoc (α := ℝ) (β := ℝ) (γ := ℕ → ℝ)).symm
      = (ν.prod (gaussianReal 0 1)).prod lindleyDraws := by
    rw [← Measure.prodAssoc_prod, MeasurableEquiv.map_symm_map]
  have h3 : ((ν.prod (gaussianReal 0 1)).prod lindleyDraws).map
      (Prod.map (fun p : ℝ × ℝ => max 0 (p.1 - c * (rho + p.2))) id)
      = (lindleyStep c rho ν).prod lindleyDraws := by
    rw [lindleyStep, ← Measure.map_prod_map _ _ (measurable_lindleyMap c rho) measurable_id,
      Measure.map_id]
  rw [← h3, ← h2, ← h1, Measure.map_map (by fun_prop) (by fun_prop),
    Measure.map_map (by fun_prop) (by fun_prop)]
  rfl

/-- **The `k`-step law from any starting law.** With `W₀ ~ ν` independent of the draws, the law
of `W_k` is `lindleyStep` applied `k` times to `ν`. -/
theorem map_lindleyFrom_eq_iterate (c rho : ℝ) (k : ℕ) (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    (ν.prod lindleyDraws).map (fun p : ℝ × (ℕ → ℝ) => lindleyFrom c rho p.1 p.2 k)
      = (lindleyStep c rho)^[k] ν := by
  induction k generalizing ν with
  | zero =>
    simp only [lindleyFrom, Function.iterate_zero, id]
    rw [Measure.map_fst_prod]
    simp
  | succ k ih =>
    haveI := isProbabilityMeasure_lindleyStep c rho ν
    rw [Function.iterate_succ_apply, ← ih (lindleyStep c rho ν),
      ← map_firstStep_lindleyDraws c rho ν,
      Measure.map_map (measurable_lindleyFrom c rho k) (by fun_prop)]
    congr 1
    funext p
    exact lindleyFrom_succ_shift c rho p.1 p.2 k

/-! ### Stationarity -/

private theorem walkSum_zero' (c rho : ℝ) (z : ℕ → ℝ) : walkSum c rho z 0 = 0 := by
  simp [walkSum]

/-- The walk after its first step is the walk of the remaining draws, shifted by the first
increment. -/
private theorem walkSum_succ_shift (c rho : ℝ) (z : ℕ → ℝ) (j : ℕ) :
    walkSum c rho z (j + 1) = walkSum c rho (fun n => z (n + 1)) j - c * (rho + z 0) := by
  simp only [walkSum, Finset.sum_range_succ']
  ring

/-- **The fixed-point identity, pathwise.** Wherever the walk is bounded above, its supremum is
one Lindley step applied to the supremum of the walk of the remaining draws:
`M(z) = max(0, M(z ∘ succ) - c(ρ + z₀))`. -/
theorem iSup_walkSum_eq_lindley {c rho : ℝ} {z : ℕ → ℝ}
    (hbdd : BddAbove (Set.range (fun j => walkSum c rho z j))) :
    ⨆ j, walkSum c rho z j
      = max 0 ((⨆ j, walkSum c rho (fun n => z (n + 1)) j) - c * (rho + z 0)) := by
  obtain ⟨b, hb⟩ := id hbdd
  have hbdd' : BddAbove (Set.range (fun j => walkSum c rho (fun n => z (n + 1)) j)) := by
    refine ⟨b + c * (rho + z 0), ?_⟩
    rintro _ ⟨j, rfl⟩
    have h : walkSum c rho z (j + 1) ≤ b := hb ⟨j + 1, rfl⟩
    rw [walkSum_succ_shift] at h
    show walkSum c rho (fun n => z (n + 1)) j ≤ b + c * (rho + z 0)
    linarith
  refine le_antisymm (ciSup_le fun j => ?_) (max_le ?_ ?_)
  · cases j with
    | zero => rw [walkSum_zero']; exact le_max_left _ _
    | succ j =>
      rw [walkSum_succ_shift]
      exact le_max_of_le_right (sub_le_sub_right (le_ciSup hbdd' j) _)
  · have h := le_ciSup hbdd 0
    rwa [walkSum_zero'] at h
  · rw [sub_le_iff_le_add]
    refine ciSup_le fun j => ?_
    have h := le_ciSup hbdd (j + 1)
    rw [walkSum_succ_shift] at h
    linarith

/-- `ae_bddAbove_walkSum` on the canonical space. -/
theorem ae_bddAbove_walkSum_lindleyDraws {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    ∀ᵐ z ∂lindleyDraws, BddAbove (Set.range (fun j => walkSum c rho z j)) :=
  ae_bddAbove_walkSum lindleyDraws_iIndepFun (fun i => measurable_pi_apply i)
    lindleyDraws_hasLaw hc hrho

/-- **The law of `M` is stationary for the recursion.** If `W ~ law(M)` is independent of a
standard normal `ζ`, then `max(0, W - c(ρ + ζ))` again has the law of `M`. -/
theorem lindleyStep_lindleyLimitLaw {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    lindleyStep c rho (lindleyLimitLaw c rho) = lindleyLimitLaw c rho := by
  set M : (ℕ → ℝ) → ℝ := fun z => ⨆ j, walkSum c rho z j with hM
  set g : ℝ × ℝ → ℝ := fun p => max 0 (p.1 - c * (rho + p.2)) with hg
  have hMm : Measurable M := measurable_iSup_walkSum c rho
  have hgm : Measurable g := measurable_lindleyMap c rho
  have hsplit : Measurable (fun z : ℕ → ℝ => (z 0, fun n => z (n + 1))) := by fun_prop
  have hshift : Measurable (fun z : ℕ → ℝ => fun n => z (n + 1)) :=
    measurable_pi_lambda _ fun n => measurable_pi_apply (n + 1)
  have hpairm : Measurable (fun z : ℕ → ℝ => (M (fun n => z (n + 1)), z 0)) :=
    (hMm.comp hshift).prodMk (measurable_pi_apply 0)
  have hae : M =ᵐ[lindleyDraws] fun z => g (M (fun n => z (n + 1)), z 0) := by
    filter_upwards [ae_bddAbove_walkSum_lindleyDraws hc hrho] with z hz
    exact iSup_walkSum_eq_lindley hz
  have hpair : lindleyDraws.map (fun z => (M (fun n => z (n + 1)), z 0))
      = (lindleyLimitLaw c rho).prod (gaussianReal 0 1) := by
    have e1 : (fun z : ℕ → ℝ => (M (fun n => z (n + 1)), z 0))
        = ((Prod.map M id) ∘ Prod.swap) ∘ (fun z : ℕ → ℝ => (z 0, fun n => z (n + 1))) := rfl
    rw [e1, ← Measure.map_map ((hMm.prodMap measurable_id).comp measurable_swap) hsplit,
      map_head_tail_lindleyDraws, ← Measure.map_map (hMm.prodMap measurable_id) measurable_swap,
      Measure.prod_swap, ← Measure.map_prod_map _ _ hMm measurable_id, Measure.map_id]
    rfl
  calc lindleyStep c rho (lindleyLimitLaw c rho)
      = (lindleyDraws.map (fun z => (M (fun n => z (n + 1)), z 0))).map g := by
        rw [hpair]; rfl
    _ = lindleyDraws.map (fun z => g (M (fun n => z (n + 1)), z 0)) :=
        Measure.map_map hgm hpairm
    _ = lindleyDraws.map M := (Measure.map_congr hae).symm
    _ = lindleyLimitLaw c rho := rfl

/-! ### Bridges to the recursion on any probability space -/

/-- The supremum of the walk of any i.i.d. standard-normal sequence has law `lindleyLimitLaw`. -/
theorem map_iSup_walkSum_eq_lindleyLimitLaw {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (c rho : ℝ) :
    P.map (fun ω => ⨆ j, walkSum c rho (fun i => zeta i ω) j) = lindleyLimitLaw c rho := by
  show _ = lindleyDraws.map (fun z => ⨆ j, walkSum c rho z j)
  rw [← map_draws_eq_lindleyDraws hindep hmeas hlaw,
    Measure.map_map (measurable_iSup_walkSum c rho) (measurable_pi_lambda _ hmeas)]
  rfl

/-- The iterate `W_k` of the recursion from `W₀ = 0` has law `lindleyStep` applied `k` times to
the point mass at `0`. -/
theorem map_clampIterate_eq_iterate_lindleyStep {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (c rho : ℝ) (k : ℕ) :
    P.map (fun ω => clampIterate c rho (fun i => zeta i ω) k)
      = (lindleyStep c rho)^[k] (Measure.dirac 0) := by
  rw [← map_lindleyFrom_eq_iterate c rho k (Measure.dirac 0), Measure.dirac_prod,
    Measure.map_map (measurable_lindleyFrom c rho k) measurable_prodMk_left,
    ← map_draws_eq_lindleyDraws hindep hmeas hlaw,
    Measure.map_map ((measurable_lindleyFrom c rho k).comp measurable_prodMk_left)
      (measurable_pi_lambda _ hmeas)]
  congr 1
  funext ω
  exact (lindleyFrom_zero_eq_clampIterate c rho _ k).symm

/-! ### Convergence from every starting law, and uniqueness -/

/-- From a nonnegative start the recursion is the larger of the start carried by the walk and
the recursion from `0`: `W_k = max(x + S_k, W_k⁰)`. Pathwise. -/
theorem lindleyFrom_eq_max {c rho x : ℝ} (hx : 0 ≤ x) (z : ℕ → ℝ) (k : ℕ) :
    lindleyFrom c rho x z k = max (x + walkSum c rho z k) (clampIterate c rho z k) := by
  induction k with
  | zero =>
    rw [walkSum_zero', add_zero, clampIterate_zero]
    exact (max_eq_left hx).symm
  | succ k ih =>
    rw [lindleyFrom, ih, clampIterate_succ, walkSum_succ]
    have h : x + (walkSum c rho z k - c * (rho + z k))
        = x + walkSum c rho z k - c * (rho + z k) := by ring
    rw [h, ← max_sub_sub_right, max_left_comm]

private theorem revDraws_revDraws (k : ℕ) (z : ℕ → ℝ) : revDraws k (revDraws k z) = z := by
  funext i
  simp only [revDraws_eq_comp]
  exact congrArg z (revIndex_involutive k i)

/-- The same, with the first `k` draws reversed: `max(x + S_k, max_{j≤k} S_j)`. -/
theorem lindleyFrom_revDraws {c rho x : ℝ} (hx : 0 ≤ x) (z : ℕ → ℝ) (k : ℕ) :
    lindleyFrom c rho x (revDraws k z) k = max (x + walkSum c rho z k) (maxWalk c rho z k) := by
  rw [lindleyFrom_eq_max hx, clampIterate_eq_maxWalk_revDraws, revDraws_revDraws,
    walkSum_revDraws c rho z le_rfl, Nat.sub_self, walkSum_zero', sub_zero]

/-- `map_revDraws_eq` on the canonical space: reversing the first `k` draws preserves the law. -/
theorem map_revDraws_lindleyDraws (k : ℕ) : lindleyDraws.map (revDraws k) = lindleyDraws := by
  have h := map_revDraws_eq lindleyDraws_iIndepFun (fun i => measurable_pi_apply i)
    lindleyDraws_hasLaw k
  have hid : (fun (ω : ℕ → ℝ) (i : ℕ) => ω i) = id := rfl
  rw [hid, Measure.map_id] at h
  exact h

private theorem walkSum_half (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    walkSum c rho z k = walkSum c (rho / 2) z k - c * (rho / 2) * k := by
  simp only [walkSum, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  ring

/-- **The walk drifts to `-∞`**, almost surely: it is the walk at drift `ρ/2`, which is bounded
above, minus `c(ρ/2)k`. So any fixed start `x` is eventually overtaken. -/
theorem ae_lindleyDraws_eventually_add_walkSum_nonpos {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    ∀ᵐ z ∂lindleyDraws, ∀ x : ℝ, ∀ᶠ k in atTop, x + walkSum c rho z k ≤ 0 := by
  filter_upwards [ae_bddAbove_walkSum_lindleyDraws hc (half_pos hrho)] with z hz x
  obtain ⟨b, hb⟩ := hz
  have hpos : 0 < c * (rho / 2) := by positivity
  obtain ⟨N, hN⟩ := exists_nat_ge ((x + b) / (c * (rho / 2)))
  filter_upwards [eventually_ge_atTop N] with k hk
  have h1 : walkSum c (rho / 2) z k ≤ b := hb ⟨k, rfl⟩
  rw [walkSum_half c rho z k]
  have hk' : (x + b) / (c * (rho / 2)) ≤ k := le_trans hN (by exact_mod_cast hk)
  have h2 : x + b ≤ c * (rho / 2) * k := by
    rw [div_le_iff₀ hpos] at hk'
    linarith
  linarith

private theorem maxWalk_nonneg (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) : 0 ≤ maxWalk c rho z k := by
  have h := walkSum_le_maxWalk c rho z (Nat.zero_le k)
  rwa [walkSum_zero'] at h

/-- Convergence of the laws, from a starting law carried by `[0, ∞)`. -/
theorem tendsto_integral_iterate_lindleyStep_of_nonneg {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hν : ∀ᵐ x ∂ν, 0 ≤ x)
    {f : ℝ → ℝ} (hfc : Continuous f) {C : ℝ} (hfb : ∀ x, |f x| ≤ C) :
    Tendsto (fun k => ∫ x, f x ∂((lindleyStep c rho)^[k] ν)) atTop
      (𝓝 (∫ x, f x ∂(lindleyLimitLaw c rho))) := by
  set Q : Measure (ℝ × (ℕ → ℝ)) := ν.prod lindleyDraws with hQ
  have hF : ∀ k, Measurable
      (fun p : ℝ × (ℕ → ℝ) => f (max (p.1 + walkSum c rho p.2 k) (maxWalk c rho p.2 k))) :=
    fun k => hfc.measurable.comp ((measurable_fst.add
      ((measurable_walkSum_apply c rho k).comp measurable_snd)).max
      ((measurable_maxWalk_apply c rho k).comp measurable_snd))
  have hfst : ∀ᵐ p ∂Q, 0 ≤ p.1 := by
    refine ae_of_ae_map measurable_fst.aemeasurable ?_
    rw [hQ, Measure.map_fst_prod]
    simpa using hν
  -- each term, read on the reversed block
  have hterm : ∀ k, ∫ x, f x ∂((lindleyStep c rho)^[k] ν)
      = ∫ p, f (max (p.1 + walkSum c rho p.2 k) (maxWalk c rho p.2 k)) ∂Q := by
    intro k
    rw [← map_lindleyFrom_eq_iterate c rho k ν,
      integral_map (measurable_lindleyFrom c rho k).aemeasurable hfc.aestronglyMeasurable]
    have hpres : Q.map (Prod.map id (revDraws k)) = Q := by
      rw [hQ, ← Measure.map_prod_map _ _ measurable_id (measurable_revDraws k), Measure.map_id,
        map_revDraws_lindleyDraws]
    have hG : Measurable (fun p : ℝ × (ℕ → ℝ) => f (lindleyFrom c rho p.1 p.2 k)) :=
      hfc.measurable.comp (measurable_lindleyFrom c rho k)
    rw [← hQ]
    conv_lhs => rw [← hpres]
    rw [integral_map (measurable_id.prodMap (measurable_revDraws k)).aemeasurable
      hG.aestronglyMeasurable]
    refine integral_congr_ae ?_
    filter_upwards [hfst] with p hp
    simp only [Prod.map_fst, Prod.map_snd, id]
    rw [lindleyFrom_revDraws hp]
  -- pointwise convergence on the reversed block
  have hsnd : ∀ᵐ p ∂Q, BddAbove (Set.range (fun j => walkSum c rho p.2 j))
      ∧ ∀ x : ℝ, ∀ᶠ k in atTop, x + walkSum c rho p.2 k ≤ 0 := by
    refine ae_of_ae_map (f := (Prod.snd : ℝ × (ℕ → ℝ) → ℕ → ℝ)) measurable_snd.aemeasurable
      (p := fun z : ℕ → ℝ => BddAbove (Set.range (fun j => walkSum c rho z j))
        ∧ ∀ x : ℝ, ∀ᶠ k in atTop, x + walkSum c rho z k ≤ 0) ?_
    rw [hQ, Measure.map_snd_prod]
    simp only [measure_univ, one_smul]
    exact (ae_bddAbove_walkSum_lindleyDraws hc hrho).and
      (ae_lindleyDraws_eventually_add_walkSum_nonpos hc hrho)
  have hlim : ∀ᵐ p ∂Q, Tendsto
      (fun k => f (max (p.1 + walkSum c rho p.2 k) (maxWalk c rho p.2 k))) atTop
      (𝓝 (f (⨆ j, walkSum c rho p.2 j))) := by
    filter_upwards [hsnd] with p hp
    refine (hfc.tendsto _).comp ?_
    refine (tendsto_maxWalk_atTop hp.1).congr' ?_
    filter_upwards [hp.2 p.1] with k hk
    exact (max_eq_right (hk.trans (maxWalk_nonneg c rho p.2 k))).symm
  have hdom := tendsto_integral_of_dominated_convergence (fun _ => C)
    (fun k => (hF k).aestronglyMeasurable) (integrable_const C)
    (fun k => ae_of_all _ fun p => by simpa [Real.norm_eq_abs] using hfb _) hlim
  have hrhs : ∫ p, f (⨆ j, walkSum c rho p.2 j) ∂Q = ∫ x, f x ∂(lindleyLimitLaw c rho) := by
    show _ = ∫ x, f x ∂(lindleyDraws.map (fun z => ⨆ j, walkSum c rho z j))
    rw [integral_map (measurable_iSup_walkSum c rho).aemeasurable hfc.aestronglyMeasurable, hQ,
      integral_fun_snd (fun z => f (⨆ j, walkSum c rho z j))]
    simp
  rw [← hrhs]
  exact hdom.congr fun k => (hterm k).symm

/-- **`W_k → M` in distribution from every starting law.** Started from `W₀ ~ ν`, independent of
the draws, the laws of the iterates converge to the law of `M`, tested against bounded continuous
functions. -/
theorem tendsto_integral_iterate_lindleyStep {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    {f : ℝ → ℝ} (hfc : Continuous f) {C : ℝ} (hfb : ∀ x, |f x| ≤ C) :
    Tendsto (fun k => ∫ x, f x ∂((lindleyStep c rho)^[k] ν)) atTop
      (𝓝 (∫ x, f x ∂(lindleyLimitLaw c rho))) := by
  haveI := isProbabilityMeasure_lindleyStep c rho ν
  have hν : ∀ᵐ x ∂(lindleyStep c rho ν), 0 ≤ x := by
    rw [lindleyStep, ae_map_iff (measurable_lindleyMap c rho).aemeasurable measurableSet_Ici]
    exact ae_of_all _ fun p => le_max_left _ _
  have h := tendsto_integral_iterate_lindleyStep_of_nonneg hc hrho (lindleyStep c rho ν) hν hfc hfb
  rw [← tendsto_add_atTop_iff_nat 1]
  refine h.congr fun k => ?_
  rw [Function.iterate_succ_apply]

/-- **The stationary law is unique.** Every probability law `ν` on `ℝ` that the recursion leaves
invariant is the law of `M`. -/
theorem eq_lindleyLimitLaw_of_lindleyStep_eq {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hν : lindleyStep c rho ν = ν) :
    ν = lindleyLimitLaw c rho := by
  haveI := isProbabilityMeasure_lindleyLimitLaw c rho
  refine ext_of_forall_integral_eq_of_IsFiniteMeasure fun f => ?_
  have h := tendsto_integral_iterate_lindleyStep hc hrho ν f.continuous (C := ‖f‖)
    (fun x => by simpa [Real.norm_eq_abs] using f.norm_coe_le_norm x)
  simp only [Function.iterate_fixed hν] at h
  exact tendsto_nhds_unique tendsto_const_nhds h

/-- **The recursion has exactly one stationary law, the law of `M`.** -/
theorem lindleyStep_eq_self_iff {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    lindleyStep c rho ν = ν ↔ ν = lindleyLimitLaw c rho :=
  ⟨eq_lindleyLimitLaw_of_lindleyStep_eq hc hrho ν,
    fun h => h ▸ lindleyStep_lindleyLimitLaw hc hrho⟩

/-! ### The clamped fraction at the limit -/

/-- Where the walk is bounded above, `M = 0` exactly when no partial sum is positive. -/
private theorem iSup_walkSum_eq_zero_iff {c rho : ℝ} {z : ℕ → ℝ}
    (hbdd : BddAbove (Set.range (fun j => walkSum c rho z j))) :
    ⨆ j, walkSum c rho z j = 0 ↔ ∀ j, walkSum c rho z j ≤ 0 := by
  constructor
  · intro h j
    rw [← h]
    exact le_ciSup hbdd j
  · intro h
    refine le_antisymm (ciSup_le h) ?_
    have h0 := le_ciSup hbdd 0
    rwa [walkSum_zero'] at h0

/-- `max_{j≤k} S_j = 0` exactly when none of `S_0, …, S_k` is positive. -/
private theorem maxWalk_eq_zero_iff (c rho : ℝ) (z : ℕ → ℝ) (k : ℕ) :
    maxWalk c rho z k = 0 ↔ ∀ j ≤ k, walkSum c rho z j ≤ 0 := by
  constructor
  · intro h j hj
    rw [← h]
    exact walkSum_le_maxWalk c rho z hj
  · intro h
    refine le_antisymm ?_ (maxWalk_nonneg c rho z k)
    rw [maxWalk_eq_sup']
    exact Finset.sup'_le _ _ fun j hj => h j (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj))

/-- **The atom of the stationary law at `0`** is the probability that the walk never rises
above its start: `P(W_∞ = 0) = P(S_j ≤ 0 for all j)`. -/
theorem lindleyLimitLaw_real_zero {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    (lindleyLimitLaw c rho).real {0}
      = P.real {ω | ∀ j, walkSum c rho (fun i => zeta i ω) j ≤ 0} := by
  have hM : Measurable (fun ω => ⨆ j, walkSum c rho (fun i => zeta i ω) j) :=
    (measurable_iSup_walkSum c rho).comp (measurable_pi_lambda _ hmeas)
  rw [← map_iSup_walkSum_eq_lindleyLimitLaw hindep hmeas hlaw c rho,
    map_measureReal_apply hM (measurableSet_singleton 0)]
  refine measureReal_congr (Filter.eventuallyEq_set.mpr ?_)
  filter_upwards [ae_bddAbove_walkSum hindep hmeas hlaw hc hrho] with ω hω
  simp only [Set.mem_preimage, Set.mem_singleton_iff]
  exact iSup_walkSum_eq_zero_iff hω

/-- At every step, the clamped fraction is the probability that none of `S_0, …, S_k` is
positive. -/
theorem measureReal_clampIterate_eq_zero_eq_forall_walkSum_le {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (c rho : ℝ) (k : ℕ) :
    P.real {ω | clampIterate c rho (fun i => zeta i ω) k = 0}
      = P.real {ω | ∀ j ≤ k, walkSum c rho (fun i => zeta i ω) j ≤ 0} := by
  have hW : Measurable (fun ω => clampIterate c rho (fun i => zeta i ω) k) :=
    (measurable_clampIterate_apply c rho k).comp (measurable_pi_lambda _ hmeas)
  have hMW : Measurable (fun ω => maxWalk c rho (fun i => zeta i ω) k) :=
    (measurable_maxWalk_apply c rho k).comp (measurable_pi_lambda _ hmeas)
  have e1 : {ω | clampIterate c rho (fun i => zeta i ω) k = 0}
      = (fun ω => clampIterate c rho (fun i => zeta i ω) k) ⁻¹' {0} := rfl
  rw [e1, ← map_measureReal_apply hW (measurableSet_singleton 0),
    map_clampIterate_eq_map_maxWalk hindep hmeas hlaw c rho k,
    map_measureReal_apply hMW (measurableSet_singleton 0)]
  congr 1
  ext ω
  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_setOf_eq]
  exact maxWalk_eq_zero_iff c rho _ k

/-- **The clamped fraction is non-increasing in the step.** -/
theorem antitone_measureReal_clampIterate_eq_zero {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) (c rho : ℝ) :
    Antitone (fun k => P.real {ω | clampIterate c rho (fun i => zeta i ω) k = 0}) := by
  intro k l hkl
  simp only [measureReal_clampIterate_eq_zero_eq_forall_walkSum_le hindep hmeas hlaw c rho]
  exact measureReal_mono fun ω hω j hj => hω j (hj.trans hkl)

/-- **The clamped fraction converges to the atom of the stationary law at `0`.** The events
`{S_j ≤ 0 for all j ≤ k}` decrease to `{S_j ≤ 0 for all j}`, and continuity from above does the
rest; the convergence in distribution alone would not give this, the limit law having an atom
at `0`. -/
theorem tendsto_measureReal_clampIterate_eq_zero {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    Tendsto (fun k => P.real {ω | clampIterate c rho (fun i => zeta i ω) k = 0}) atTop
      (𝓝 ((lindleyLimitLaw c rho).real {0})) := by
  set A : ℕ → Set Ω := fun k => {ω | ∀ j ≤ k, walkSum c rho (fun i => zeta i ω) j ≤ 0} with hA
  have hS : ∀ j, Measurable (fun ω => walkSum c rho (fun i => zeta i ω) j) :=
    fun j => (measurable_walkSum_apply c rho j).comp (measurable_pi_lambda _ hmeas)
  have hAm : ∀ k, MeasurableSet (A k) := by
    intro k
    have : A k = ⋂ j ∈ Finset.range (k + 1), {ω | walkSum c rho (fun i => zeta i ω) j ≤ 0} := by
      ext ω
      simp [hA]
    rw [this]
    exact Finset.measurableSet_biInter _ fun j _ => measurableSet_le (hS j) measurable_const
  have hanti : Antitone A := fun k l hkl ω hω j hj => hω j (hj.trans hkl)
  have hinter : (⋂ k, A k) = {ω | ∀ j, walkSum c rho (fun i => zeta i ω) j ≤ 0} := by
    ext ω
    simp only [hA, Set.mem_iInter, Set.mem_setOf_eq]
    exact ⟨fun h j => h j j le_rfl, fun h k j _ => h j⟩
  have hlim := tendsto_measure_iInter_atTop (μ := P) (fun k => (hAm k).nullMeasurableSet) hanti
    ⟨0, measure_ne_top P _⟩
  have hreal := (ENNReal.tendsto_toReal (measure_ne_top P (⋂ k, A k))).comp hlim
  rw [lindleyLimitLaw_real_zero hindep hmeas hlaw hc hrho, ← hinter]
  refine hreal.congr fun k => ?_
  simp only [Function.comp_apply]
  rw [measureReal_clampIterate_eq_zero_eq_forall_walkSum_le hindep hmeas hlaw c rho k]
  rfl

/-! ### The same statements on laws alone -/

/-- The law of `M` built from any i.i.d. standard-normal sequence is stationary. -/
theorem lindleyStep_map_iSup_walkSum {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ}
    (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    lindleyStep c rho (P.map (fun ω => ⨆ j, walkSum c rho (fun i => zeta i ω) j))
      = P.map (fun ω => ⨆ j, walkSum c rho (fun i => zeta i ω) j) := by
  rw [map_iSup_walkSum_eq_lindleyLimitLaw hindep hmeas hlaw c rho]
  exact lindleyStep_lindleyLimitLaw hc hrho

/-- The mass that the law of `W_k` puts at `0` is the clamped fraction `P(W_k = 0)`. -/
theorem iterate_lindleyStep_real_zero_eq (c rho : ℝ) (k : ℕ) :
    ((lindleyStep c rho)^[k] (Measure.dirac 0)).real {0}
      = lindleyDraws.real {z | clampIterate c rho z k = 0} := by
  rw [← map_clampIterate_eq_iterate_lindleyStep lindleyDraws_iIndepFun
      (fun i => measurable_pi_apply i) lindleyDraws_hasLaw c rho k,
    map_measureReal_apply (measurable_clampIterate_apply c rho k) (measurableSet_singleton 0)]
  rfl

/-- The clamped fraction read on the laws of the iterates from `W₀ = 0` is non-increasing in the
step. -/
theorem antitone_iterate_lindleyStep_real_zero (c rho : ℝ) :
    Antitone (fun k => ((lindleyStep c rho)^[k] (Measure.dirac 0)).real {0}) := by
  simp only [iterate_lindleyStep_real_zero_eq]
  exact antitone_measureReal_clampIterate_eq_zero lindleyDraws_iIndepFun
    (fun i => measurable_pi_apply i) lindleyDraws_hasLaw c rho

/-- The clamped fraction read on the laws of the iterates from `W₀ = 0` converges to the mass of
the stationary law at `0`. -/
theorem tendsto_iterate_lindleyStep_real_zero {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    Tendsto (fun k => ((lindleyStep c rho)^[k] (Measure.dirac 0)).real {0}) atTop
      (𝓝 ((lindleyLimitLaw c rho).real {0})) := by
  simp only [iterate_lindleyStep_real_zero_eq]
  exact tendsto_measureReal_clampIterate_eq_zero lindleyDraws_iIndepFun
    (fun i => measurable_pi_apply i) lindleyDraws_hasLaw hc hrho

end IcnnLift
