import SpitzerCyclic
import SpitzerIdentity
import SpitzerGaussian
import LindleyStationary

/-!
# The clamped fraction of the projected recursion at its limit (`prop:clamp`)

The last clause of the paper's Proposition `prop:clamp` (Appendix, the Lindley recursion): for
the projected recursion `W_{k+1} = max(0, W_k - c(ρ + ζ_k))` with i.i.d. standard normal draws,
scale `c = ησ > 0` and drift ratio `ρ = γ/σ > 0`, the probability that the walk
`S_j = -c ∑_{i<j} (ρ + ζ_i)` never rises above its start, which is the clamped fraction
`P(W_∞ = 0)` of the limit law, is

  `P(S_j ≤ 0 for all j) = exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))`,

free of `η` and `σ`, strictly increasing in `ρ`, tending to `0` as `ρ ↓ 0` and to `1` as
`ρ → ∞`. The law of `M = sup_j S_j` is the recursion's unique stationary law
(`LindleyStationary.lindleyStep_eq_self_iff`), so this is its atom at `0`
(`lindleyLimitLaw_real_zero_eq_clampedFraction`), and the clamped fraction `P(W_k = 0)` of the
recursion started at `W_0 = 0` decreases to it (`tendsto_measureReal_clampIterate_eq_zero_clampedFraction`).

This file only joins four others:

* `SpitzerCyclic.cyclic_ladder_identity`: the deterministic cyclic lemma;
* `SpitzerIdentity.spitzer_ladder_identity`: Spitzer's ladder-epoch identity
  `P(S_n ≤ 0 ∀ n) = exp(-∑_{n≥1} n⁻¹ P(S_n > 0))` for an i.i.d. real walk, with the cyclic lemma
  as a hypothesis, discharged here;
* `SpitzerGaussian`: `P(S_n > 0) = Φ(-ρ√n)` for the Gaussian walk, the summability of the series
  and its limits in `ρ` (`clampedFraction ρ = exp(-spitzerSeries ρ)`);
* `LindleyStationary`: the law of `M` is the unique stationary law of the recursion, the
  recursion converges to it in distribution from every starting law, and its atom at `0` is
  `P(S_j ≤ 0 for all j)`.

No `sorry`/`admit`/`native_decide`/`axiom`.
-/

open MeasureTheory ProbabilityTheory Filter Set
open scoped Topology

namespace IcnnLift

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Spitzer's identity for the projected recursion's walk.** With i.i.d. standard normal draws,
`c > 0` and `ρ > 0`, the walk `S_j = -c ∑_{i<j}(ρ + ζ_i)` stays at or below its start forever
with probability `exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))`. -/
theorem measureReal_forall_walkSum_nonpos {P : Measure Ω} [IsProbabilityMeasure P]
    {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P) (hmeas : ∀ i, Measurable (zeta i))
    (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P) {c rho : ℝ} (hc : 0 < c)
    (hrho : 0 < rho) :
    P.real {ω | ∀ j, walkSum c rho (fun i => zeta i ω) j ≤ 0} = clampedFraction rho := by
  have h := spitzer_ladder_identity (X := fun i ω => -(c * (rho + zeta i ω)))
    (iIndepFun_walkIncrements hindep c rho) (fun i => measurable_walkIncrements hmeas c rho i)
    (fun i => identDistrib_walkIncrements hlaw c rho i)
    (fun n x hn => cyclic_ladder_identity n x hn)
    (summable_measureReal_walkIncrements_pos hindep hmeas hlaw hc hrho)
  rw [tsum_measureReal_walkIncrements_pos hindep hmeas hlaw hc rho] at h
  simp only [sum_walkIncrements_eq_walkSum] at h
  exact h

/-- **The clamped fraction of the stationary law.** At every scale `c > 0`, the recursion's unique
stationary law puts mass `exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))` on `0`, a function of `ρ` alone. -/
theorem lindleyLimitLaw_real_zero_eq_clampedFraction {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    (lindleyLimitLaw c rho).real {0} = clampedFraction rho := by
  rw [lindleyLimitLaw_real_zero lindleyDraws_iIndepFun (fun i => measurable_pi_apply i)
      lindleyDraws_hasLaw hc hrho]
  exact measureReal_forall_walkSum_nonpos lindleyDraws_iIndepFun
    (fun i => measurable_pi_apply i) lindleyDraws_hasLaw hc hrho

/-- **The clamped fraction along the recursion.** From `W_0 = 0`, `P(W_k = 0)` converges to
`exp(-∑_{k≥1} k⁻¹ Φ(-ρ√k))` (non-increasingly: `antitone_measureReal_clampIterate_eq_zero`). -/
theorem tendsto_measureReal_clampIterate_eq_zero_clampedFraction {P : Measure Ω}
    [IsProbabilityMeasure P] {zeta : ℕ → Ω → ℝ} (hindep : iIndepFun zeta P)
    (hmeas : ∀ i, Measurable (zeta i)) (hlaw : ∀ i, HasLaw (zeta i) (gaussianReal 0 1) P)
    {c rho : ℝ} (hc : 0 < c) (hrho : 0 < rho) :
    Tendsto (fun k => P.real {ω | clampIterate c rho (fun i => zeta i ω) k = 0}) atTop
      (𝓝 (clampedFraction rho)) := by
  rw [← lindleyLimitLaw_real_zero_eq_clampedFraction hc hrho]
  exact tendsto_measureReal_clampIterate_eq_zero hindep hmeas hlaw hc hrho

/-- **The stationary clamped fraction decreases to `0` as `ρ ↓ 0`**, at every scale `c > 0`. -/
theorem tendsto_lindleyLimitLaw_real_zero_nhdsGT {c : ℝ} (hc : 0 < c) :
    Tendsto (fun rho => (lindleyLimitLaw c rho).real {0}) (𝓝[>] 0) (𝓝 0) := by
  refine tendsto_clampedFraction_nhdsGT.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with rho hrho
  exact (lindleyLimitLaw_real_zero_eq_clampedFraction hc hrho).symm

/-- **The stationary clamped fraction is strictly increasing in `ρ`.** -/
theorem lindleyLimitLaw_real_zero_strictMonoOn {c : ℝ} (hc : 0 < c) :
    StrictMonoOn (fun rho => (lindleyLimitLaw c rho).real {0}) (Ioi 0) := by
  intro a ha b hb hab
  simp only
  rw [lindleyLimitLaw_real_zero_eq_clampedFraction hc ha,
    lindleyLimitLaw_real_zero_eq_clampedFraction hc hb]
  exact clampedFraction_strictMonoOn ha hb hab

end IcnnLift
