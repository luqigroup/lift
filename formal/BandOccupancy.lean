import Mathlib

/-!
# (FW-3a) Band occupancy: the deterministic transit-time core

The corrected mechanism of the paper is developed in `docs/design/lemma1_rederivation.md`. Its
Freidlin–Wentzell step (FW-2) compares the quasipotential of two diffusions that differ only on
the shoulder band, and its **strictness clause** needs one geometric input: that a path cannot
evade the band. Section 11.1 of that note, labelled **FW-3a (band occupancy)**, supplies the
input. In the note's own words: if the drift is bounded, `‖∇U‖ ≤ M` on the region, if the diffusion
satisfies `Σ ≤ Λ I`, and if the shoulder band `B` contains a slab of width `w > 0` that separates
the well from the exit set, then every path `γ` whose Freidlin–Wentzell action satisfies
`S[γ] ≤ S̄` spends total time at least

    τ(w, Λ, S̄, M) = (√((4 Λ S̄)² + 2 M² w²) − 4 Λ S̄) / (2 M²)  >  0

inside `B`. The point of the statement is that `τ` is positive for **every** `w > 0` and requires
no assumption about where the minimising path goes: the diffusion advantage the band carries
cannot be routed around.

This file machine-checks two things, and it matters that they are kept apart. The first is a
**deterministic transit-time theorem** about paths whose own speed is bounded, proved here in full
from the intermediate value theorem and the Lipschitz condition; it is a theorem of real analysis,
and its hypothesis is not the note's action bound. The second is the note's own **algebra**, from
its energy inequality to the positive root `τ`, also proved here in full. The link between them,
that an action bound produces the note's energy inequality, is a property of the
Freidlin–Wentzell rate functional and is carried as named hypotheses.

*Transit.* A continuous path that starts at or below `a` at time `0` and is at or above `b` at
time `T` attains **every** value of the band `[a, b]`, at a time inside the window
(`transit_of_crossing`, by `intermediate_value_Icc`), so the occupancy set `bandTimes` is
nonempty. More is true, and it is worth separating out: for `a < b` the occupancy set contains a
nonempty *open* set, hence already has strictly positive Lebesgue measure. That is
`bandTime_pos_of_crossing`, whose only hypothesis on the path is continuity on the window.
Positivity of the occupancy time is therefore free; what a speed bound buys is a *rate*.

*Occupancy.* If in addition the path is Lipschitz on the window with constant `M`, the occupancy
set contains an open interval of length at least `(b − a) / M`, so the occupancy time is at least
`(b − a) / M`. The interval is produced by the "last entry before the first exit" construction:
`t₂` is the *first* time the path reaches `b`, `t₁` is the *last* time at or before `t₂` at which
the path is still at or below `a`, and the *open* interval `(t₁, t₂)` is then trapped strictly
inside the band by the extremality of `t₁` and `t₂` alone. Taking the interval open is what makes
the statement true without any further limit argument at the endpoints, and the Lipschitz bound
applied to the two endpoint values gives `b − a ≤ M (t₂ − t₁)`. The statement is then transported
to the note's geometry in three steps: `slabTime_ge_width_div_speed` reads a path in an arbitrary
metric space through a `1`-Lipschitz scalar `φ`; `lipschitzWith_one_inner_left` proves that the
readout the note actually uses, `x ↦ ⟪n, x⟫` for a unit normal `n`, is such a scalar, so that
`slabTime_ge_width_div_speed_inner` is the note's slab statement in `ℝ^d` verbatim; and
`bandOccupancy_ge_width_div_speed` passes from the slab to any band `B` containing it, which is
the set whose occupancy the note's conclusion is about.

**The constant `M` means two different things in the two halves of this file, and the difference
is the whole of the gap.** In the note, `M` bounds the *drift*: `‖∇U‖ ≤ M`. The note never bounds
the speed of a finite-action path pointwise; it bounds it only in mean square, through
`∫_B ‖γ̇‖² ≤ 8 Λ S̄ + 2 M² t`. In the transit theorems of this file, `M` is a bound on the *speed
of the path itself*, `LipschitzOnWith M.toNNReal γ`. That is strictly stronger than an action
bound and is not implied by one: a path of finite action need not be Lipschitz at all, since
finite action controls `∫ ‖γ̇‖²` and not `sup ‖γ̇‖` — on `[0, 1]` the path `t ↦ t^(3/4)` has
`∫ ‖γ̇‖² = 9/8` and unbounded speed at the origin. The transit theorems are therefore not a
formalisation of the note's proof; they are an independent argument reaching a conclusion of the
same shape under a stronger hypothesis. The two constants are compared exactly, and the
comparison is proved: `bandOccupancyTime_lt_width_div_speed` shows that at equal `M` the note's
`τ` is *strictly smaller* than the transit theorems' `w / M`, which is the precise sense in which
the Lipschitz route assumes more and concludes more.

*The note's `τ`.* The passage from the note's energy inequality `w² ≤ t (8 Λ S̄ + 2 M² t)` to the
positive root `τ` is pure algebra, and it is proved here in full, together with `τ > 0`. Two forms
are given. In `bandOccupancyTime_le_of_energy_bound` and `occupancy_time_of_action_bound` the
occupancy time `t` is an uninterpreted real, and those are theorems about the note's arithmetic;
in `bandOccupancyTime_le_bandTime` the same `t` is `(bandTime γ T a b).toReal`, the measure of the
set of times the path really is in the band, and that is the form to read as the note's
conclusion about a path.

## What this file does NOT prove, and why

The note's FW-3a is a statement about action-minimising paths of a diffusion. Its analytic links
lie outside this development, and each is carried as an **explicit, named hypothesis** rather than
assumed away or axiomatised.

1. *From bounded action to the energy inequality.* The Freidlin–Wentzell rate functional
   `S[γ] = ¼ ∫ (γ̇ + ∇U(γ))ᵀ Σ(γ)⁻¹ (γ̇ + ∇U(γ)) dt` is not definable in this mathlib: the audit
   of `v4.31.0` records zero declarations for Itô integrals, stochastic differential equations,
   diffusion generators, Fokker–Planck, exit times or large deviations, so there is no object
   `S[γ]` to quantify over and no theorem relating it to `γ̇`. The note's derivation — `Σ ≤ Λ I`
   gives `Σ⁻¹ ≥ Λ⁻¹ I`, hence `∫_B ‖u‖² ≤ 4 Λ S̄` for `u = γ̇ + ∇U(γ)`, and then
   `‖γ̇‖² ≤ 2‖u‖² + 2‖∇U‖²` with `‖∇U‖ ≤ M` gives `∫_B ‖γ̇‖² ≤ 8 Λ S̄ + 2 M² t` — is therefore
   carried as the hypothesis `hE` of `occupancy_time_of_action_bound`. Discharging it needs a
   formal theory of the rate functional, hence a formal Itô calculus.
2. *Transversality, and Cauchy–Schwarz on the occupancy set.* That the path covers Euclidean
   distance at least `w` inside the slab (`hwL`) is the note's separation hypothesis on the
   geometry of the band, and `L² ≤ t · E` (`hCS`) is Cauchy–Schwarz for the speed on the
   occupancy set, whose formal statement needs the path derivative `γ̇` as an integrable function
   — again an object the missing calculus would supply. In the transit theorems of this file both
   are *replaced by proof*: `transit_of_crossing` derives the crossing from the intermediate value
   theorem, and `exists_crossing_interval` derives the time bound from the Lipschitz constant
   directly, with no integral inequality at all.
3. *The Lipschitz hypothesis is not a stand-in for the action bound.* It would be wrong to read
   `LipschitzOnWith M.toNNReal γ (Icc 0 T)` as "the note's action bound, modulo unformalised
   stochastic analysis". No further formalisation derives it from an action bound, because that
   implication is false, as the paragraph on the two meanings of `M` above records. The Lipschitz
   theorems are offered as an independent and fully proved route to a positive occupancy time,
   under a hypothesis a referee can check directly for the path at hand.

So `bandTime_ge_width_div_speed`, `slabTime_ge_width_div_speed` and their corollaries are complete
theorems about paths of bounded speed; `bandTime_pos_of_crossing` is a complete theorem about
merely continuous paths; and `bandOccupancyTime_le_bandTime` together with
`occupancy_time_of_action_bound` are complete theorems about the note's arithmetic. None of them
is, and none is presented as, the whole of FW-3a. The unproved part of FW-3a is exactly the
passage from `S[γ] ≤ S̄` to the note's energy inequality, together with the transversality input:
items 1 and 2 above.

## Results
* `bandTimes`, `bandTime` — the occupancy set of the band `[a, b]` inside the window `[0, T]`,
  and its Lebesgue measure.
* `measurableSet_bandTimes` — the occupancy set of a continuous path is closed, hence measurable.
* `bandTime_ne_top` — the occupancy time is finite, so it has a real value.
* `transit_of_crossing` — intermediate value theorem: a path crossing the band attains every
  value of it.
* `band_subset_image_bandTimes`, `bandTimes_nonempty` — every band value is attained *at a time
  in the occupancy set*, which is therefore nonempty.
* `bandTime_pos_of_crossing` — the occupancy time of a merely continuous crossing path is
  already strictly positive: positivity needs no bound on the speed.
* `exists_crossing_interval` — the occupancy set of a crossing path whose own speed is bounded by
  `M` contains an open interval of length at least `(b − a) / M`.
* `bandTime_ge_width_div_speed` — hence the occupancy time is at least `(b − a) / M`.
* `bandTime_pos_of_bounded_speed` — the quantitative form of positivity for such a path.
* `bandTime_ge_width_div_speed_of_descending` — the same for a path crossing downwards, which is
  the exit direction.
* `slabTime_ge_width_div_speed` — the slab form: a path in a metric space read through a
  `1`-Lipschitz scalar `φ` spends time at least `(b − a) / M` in the slab `φ⁻¹[a, b]`.
* `slabTime_le_bandOccupancy` — a band containing the slab is occupied at least as long as the
  slab, so the bounds transfer to the set the note's conclusion names.
* `bandOccupancy_ge_width_div_speed` — hence the time spent in a band `B ⊇ φ⁻¹[a, b]` is at
  least `(b − a) / M`.
* `lipschitzWith_one_inner_left` — the note's own readout `x ↦ ⟪n, x⟫` for a unit normal `n` is
  `1`-Lipschitz.
* `slabTime_ge_width_div_speed_inner` — the note's `ℝ^d` slab statement: at least `w / M` inside
  `{x : a ≤ ⟪n, x⟫ ≤ b}`, of width `w = b − a`.
* `bandOccupancyTime` — the note's `τ(w, Λ, S̄, M)`.
* `bandOccupancyTime_pos` — `τ > 0` for every `w > 0` and `M > 0`.
* `bandOccupancyTime_lt_width_div_speed` — at equal `M`, `τ < w / M` strictly: the note's
  constant is smaller than the transit theorems'.
* `quadratic_positive_root_le` — the algebraic core: `w² ≤ 2At + 2M²t²` and `t ≥ 0` force
  `t ≥ (√(A² + 2M²w²) − A)/(2M²)`.
* `bandOccupancyTime_le_of_energy_bound` — the note's inequality `w² ≤ t (8ΛS̄ + 2M²t)` forces
  `t ≥ τ`, for an uninterpreted real `t`.
* `bandOccupancyTime_le_bandTime` — the same with `t` instantiated to the occupancy time of an
  actual path, which is the note's conclusion as a statement about paths.
* `occupancy_time_of_action_bound` — the note's FW-3a conclusion `0 < τ ≤ t` from its three
  named analytic inputs.
* `bandTime_unitCrossing_eq_one`, `bandTime_ge_width_div_speed_sharp`, `slabTime_straightPath`,
  `slabTime_straightPath_euclidean`, `bandOccupancyTime_le_bandTime_nonvacuous`,
  `occupancy_time_of_action_bound_nonvacuous` — witnesses:
  the hypotheses of each main theorem are satisfied by an explicit instance, and the constant
  `(b − a) / M` is attained there, so no result above is vacuously true or improvable.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`. The explicit hypotheses carried, and what would discharge each,
are these. The speed bound `LipschitzOnWith M.toNNReal γ (Icc 0 T)` is a hypothesis about the
path, is not the note's action bound, and is not implied by it; nothing short of replacing it
discharges it, and item 3 above says why. The three analytic inputs of
`occupancy_time_of_action_bound` — the transversality bound `w ≤ L`, the Cauchy–Schwarz step
`L² ≤ t E`, and the action-plus-drift energy bound `E ≤ 8 Λ S̄ + 2 M² t` — require a formalised
Freidlin–Wentzell rate functional and the path derivative as an integrable function, hence a
formal Itô calculus, which mathlib `v4.31.0` does not contain. The sign conditions `0 ≤ Λ` and
`0 ≤ S̄` are not decoration: for `4 Λ S̄ < 0` the root inequality is false, since `t = w = 0`
satisfies the quadratic hypothesis while `τ > 0`. Finally, in
`occupancy_time_of_action_bound` the reals `t`, `L` and `E` are uninterpreted, and nothing in that
statement ties `t` to the occupancy time of any path; `bandOccupancyTime_le_bandTime` is the
version in which `t` is `(bandTime γ T a b).toReal`, and it is the one that speaks about a path.
-/

open MeasureTheory Set
open scoped ENNReal

namespace IcnnLift

/-! ### The occupancy set of a band -/

section Band

variable {γ : ℝ → ℝ} {T a b M : ℝ}

/-- The **occupancy set**: the times of the window `[0, T]` at which the scalar path `γ` lies in
the shoulder band `[a, b]`. In the note's notation this is `{t ∈ [0,T] : γ(t) ∈ B}` for a band
`B` presented as a slab of width `b - a`. -/
def bandTimes (γ : ℝ → ℝ) (T a b : ℝ) : Set ℝ := Set.Icc 0 T ∩ γ ⁻¹' Set.Icc a b

/-- Membership in the occupancy set, unfolded. -/
theorem mem_bandTimes {t : ℝ} :
    t ∈ bandTimes γ T a b ↔ t ∈ Set.Icc 0 T ∧ γ t ∈ Set.Icc a b := Iff.rfl

/-- The **occupancy time**: the Lebesgue measure of the occupancy set. This is the quantity the
note calls `t` and bounds below by `τ`. -/
noncomputable def bandTime (γ : ℝ → ℝ) (T a b : ℝ) : ℝ≥0∞ := volume (bandTimes γ T a b)

/-- The occupancy set of a path that is continuous on the window is closed, hence measurable, so
`bandTime` is the measure of a genuine measurable set rather than an outer measure of a wild
set. -/
theorem measurableSet_bandTimes (hcont : ContinuousOn γ (Set.Icc 0 T)) :
    MeasurableSet (bandTimes γ T a b) :=
  (hcont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Icc).measurableSet

/-- The occupancy time is **finite**, because the occupancy set is contained in the window. Hence
`(bandTime γ T a b).toReal` is a genuine real number and may be used as the note's `t`; this is
what `bandOccupancyTime_le_bandTime` needs. No continuity is required. -/
theorem bandTime_ne_top : bandTime γ T a b ≠ ∞ := by
  have hle : bandTime γ T a b ≤ volume (Set.Icc (0 : ℝ) T) := measure_mono Set.inter_subset_left
  rw [Real.volume_Icc] at hle
  exact ne_top_of_le_ne_top ENNReal.ofReal_ne_top hle

/-! ### Part 1: transit -/

/-- **Transit (intermediate value theorem).** A path that is continuous on the window `[0, T]`,
starts at or below the bottom `a` of the band and ends at or above its top `b` attains **every**
value of the band at some time of the window. This is the formal content of the note's phrase
"every path from `x₀` to the exit set crosses `B`": a path cannot jump the band. -/
theorem transit_of_crossing (hT : 0 ≤ T) (hcont : ContinuousOn γ (Set.Icc 0 T))
    (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) : Set.Icc a b ⊆ γ '' Set.Icc 0 T := by
  intro y hy
  exact intermediate_value_Icc hT hcont ⟨h0.trans hy.1, hy.2.trans hTb⟩

/-- Every value of the band is attained at a time that itself lies in the occupancy set: the
crossing time witnesses occupancy, not merely attainment. -/
theorem band_subset_image_bandTimes (hT : 0 ≤ T) (hcont : ContinuousOn γ (Set.Icc 0 T))
    (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) : Set.Icc a b ⊆ γ '' bandTimes γ T a b := by
  intro y hy
  obtain ⟨t, ht, hty⟩ := transit_of_crossing hT hcont h0 hTb hy
  refine ⟨t, ⟨ht, ?_⟩, hty⟩
  show γ t ∈ Set.Icc a b
  rw [hty]
  exact hy

/-- The occupancy set of a crossing path is nonempty. This is the qualitative half of FW-3a: the
band is visited. That it is visited for a positive *length of time* follows already from
continuity, in `bandTime_pos_of_crossing` immediately below; a bound on the speed is needed only
for an explicit rate, in `bandTime_ge_width_div_speed`. -/
theorem bandTimes_nonempty (hT : 0 ≤ T) (hab : a ≤ b) (hcont : ContinuousOn γ (Set.Icc 0 T))
    (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) : (bandTimes γ T a b).Nonempty := by
  obtain ⟨t, ht, -⟩ := band_subset_image_bandTimes hT hcont h0 hTb (Set.left_mem_Icc.2 hab)
  exact ⟨t, ht⟩

/-- **Positivity of the occupancy time needs no bound on the speed.** A path merely continuous on
the window, starting at or below `a` and ending at or above `b` with `a < b`, already spends a
set of times of **strictly positive Lebesgue measure** inside the band.

The reason is that the occupancy set contains the *open* set `(0, T) ∩ γ⁻¹((a, b))`, which is
nonempty: by the intermediate value theorem the path takes the midpoint value `(a + b) / 2` at
some time of the window, and that time is interior to the window because `γ 0 ≤ a` and `b ≤ γ T`
put the endpoints strictly outside the open band. A nonempty open subset of `ℝ` has positive
Lebesgue measure. No speed hypothesis enters, which is why the Lipschitz constant `M` of the
next section should be read as buying a *rate* rather than positivity; the note's `τ` is likewise
a rate, and the qualitative statement that the band cannot be evaded in zero time is this
theorem. -/
theorem bandTime_pos_of_crossing (hT : 0 ≤ T) (hab : a < b)
    (hcont : ContinuousOn γ (Set.Icc 0 T)) (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) :
    0 < bandTime γ T a b := by
  have hmem : (a + b) / 2 ∈ Set.Icc (γ 0) (γ T) := ⟨by linarith, by linarith⟩
  obtain ⟨t₀, ht₀mem, ht₀⟩ := intermediate_value_Icc hT hcont hmem
  -- the crossing time is interior to the window, because the endpoints are outside the open band
  have h0t : 0 < t₀ := by
    rcases lt_or_eq_of_le ht₀mem.1 with h | h
    · exact h
    · exfalso; rw [← h] at ht₀; linarith
  have htT : t₀ < T := by
    rcases lt_or_eq_of_le ht₀mem.2 with h | h
    · exact h
    · exfalso; rw [h] at ht₀; linarith
  -- the occupancy set contains a nonempty open set
  have hopen : IsOpen (Set.Ioo (0 : ℝ) T ∩ γ ⁻¹' Set.Ioo a b) :=
    (hcont.mono Set.Ioo_subset_Icc_self).isOpen_inter_preimage isOpen_Ioo isOpen_Ioo
  have hne : (Set.Ioo (0 : ℝ) T ∩ γ ⁻¹' Set.Ioo a b).Nonempty :=
    ⟨t₀, ⟨h0t, htT⟩, by rw [Set.mem_preimage, ht₀]; constructor <;> linarith⟩
  have hsub : Set.Ioo (0 : ℝ) T ∩ γ ⁻¹' Set.Ioo a b ⊆ bandTimes γ T a b := fun t ht =>
    ⟨Set.Ioo_subset_Icc_self ht.1, Set.Ioo_subset_Icc_self ht.2⟩
  exact lt_of_lt_of_le (hopen.measure_pos volume hne) (measure_mono hsub)

/-! ### Part 2: occupancy -/

/-- **The crossing interval.** For a path that crosses the band and whose speed is bounded by
`M > 0` in the Lipschitz sense, the occupancy set contains an *open* interval `(t₁, t₂)` of
length at least `(b - a) / M`. Here and in everything that follows from it, `M` bounds the speed
of the path itself, which is not the note's `M`: the note's `M` bounds the drift `‖∇U‖`, and the
note bounds the speed of a finite-action path only in mean square. The module docstring records
the difference and why it cannot be removed.

The construction is the note's "last entry before the first exit", ordered so that no limit
argument at the endpoints is needed. Let `t₂` be the **least** time of the window at which
`γ ≥ b`; it exists because `{t ∈ [0,T] : γ t ≥ b}` is a nonempty compact set (`T` belongs to it),
compactness coming from continuity of `γ` on the window. Let `t₁` be the **greatest** time in
`[0, t₂]` at which `γ ≤ a`; it exists for the same reason (`0` belongs to that set). Then for
`t₁ < t < t₂` minimality of `t₂` gives `γ t < b` and maximality of `t₁` gives `γ t > a`, so the
whole open interval lies in the band, with no information needed about `γ t₁` or `γ t₂` beyond
`γ t₁ ≤ a` and `γ t₂ ≥ b`. Those two inequalities and the Lipschitz bound give
`b - a ≤ γ t₂ - γ t₁ ≤ M (t₂ - t₁)`: the path cannot traverse the band faster than its speed
allows. -/
theorem exists_crossing_interval (hT : 0 ≤ T) (hM : 0 < M)
    (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T)) (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) :
    ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ t₂ ∧ t₂ ≤ T ∧ (b - a) / M ≤ t₂ - t₁ ∧
      Set.Ioo t₁ t₂ ⊆ bandTimes γ T a b := by
  have hcont : ContinuousOn γ (Set.Icc 0 T) := hlip.continuousOn
  -- the first time at which the path has reached the top of the band
  have hBclosed : IsClosed (Set.Icc 0 T ∩ γ ⁻¹' Set.Ici b) :=
    hcont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Ici
  have hBcompact : IsCompact (Set.Icc 0 T ∩ γ ⁻¹' Set.Ici b) :=
    IsCompact.of_isClosed_subset isCompact_Icc hBclosed Set.inter_subset_left
  have hBne : (Set.Icc 0 T ∩ γ ⁻¹' Set.Ici b).Nonempty := ⟨T, Set.right_mem_Icc.2 hT, hTb⟩
  obtain ⟨t₂, ht₂⟩ := hBcompact.exists_isLeast hBne
  have ht₂mem : t₂ ∈ Set.Icc (0 : ℝ) T := ht₂.1.1
  have hγt₂ : b ≤ γ t₂ := ht₂.1.2
  -- the last time at or before it at which the path is still at or below the bottom of the band
  have hsub : Set.Icc (0 : ℝ) t₂ ⊆ Set.Icc 0 T := Set.Icc_subset_Icc le_rfl ht₂mem.2
  have hAclosed : IsClosed (Set.Icc 0 t₂ ∩ γ ⁻¹' Set.Iic a) :=
    (hcont.mono hsub).preimage_isClosed_of_isClosed isClosed_Icc isClosed_Iic
  have hAcompact : IsCompact (Set.Icc 0 t₂ ∩ γ ⁻¹' Set.Iic a) :=
    IsCompact.of_isClosed_subset isCompact_Icc hAclosed Set.inter_subset_left
  have hAne : (Set.Icc 0 t₂ ∩ γ ⁻¹' Set.Iic a).Nonempty :=
    ⟨0, Set.left_mem_Icc.2 ht₂mem.1, h0⟩
  obtain ⟨t₁, ht₁⟩ := hAcompact.exists_isGreatest hAne
  have ht₁mem : t₁ ∈ Set.Icc (0 : ℝ) t₂ := ht₁.1.1
  have hγt₁ : γ t₁ ≤ a := ht₁.1.2
  -- the speed bound turns the band width into a time
  have hdist : dist (γ t₂) (γ t₁) ≤ (M.toNNReal : ℝ) * dist t₂ t₁ :=
    hlip.dist_le_mul t₂ ht₂mem t₁ (hsub ht₁mem)
  rw [Real.dist_eq, Real.dist_eq, Real.coe_toNNReal M hM.le] at hdist
  have hwidth : b - a ≤ M * (t₂ - t₁) := by
    have h1 : γ t₂ - γ t₁ ≤ |γ t₂ - γ t₁| := le_abs_self _
    have h2 : |t₂ - t₁| = t₂ - t₁ := abs_of_nonneg (by linarith [ht₁mem.2])
    rw [h2] at hdist
    linarith
  refine ⟨t₁, t₂, ht₁mem.1, ht₁mem.2, ht₂mem.2, ?_, ?_⟩
  · rw [div_le_iff₀ hM]
    linarith [hwidth]
  · intro t ht
    have htT : t ∈ Set.Icc (0 : ℝ) T :=
      ⟨le_trans ht₁mem.1 ht.1.le, le_trans ht.2.le ht₂mem.2⟩
    refine ⟨htT, ?_, ?_⟩
    · by_contra hcon
      have hlt : γ t < a := not_le.1 hcon
      exact absurd (ht₁.2 ⟨⟨le_trans ht₁mem.1 ht.1.le, ht.2.le⟩, hlt.le⟩) (not_le.2 ht.1)
    · by_contra hcon
      have hgt : b < γ t := not_le.1 hcon
      exact absurd (ht₂.2 ⟨htT, hgt.le⟩) (not_le.2 ht.2)

/-- **Occupancy.** A path of speed at most `M` that starts at or below the band and ends at or
above it spends time at least `(b - a) / M` inside the band: the width of the band divided by the
largest speed available to the path. This is the deterministic core of FW-3a. -/
theorem bandTime_ge_width_div_speed (hT : 0 ≤ T) (hM : 0 < M)
    (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T)) (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) :
    ENNReal.ofReal ((b - a) / M) ≤ bandTime γ T a b := by
  obtain ⟨t₁, t₂, -, -, -, hlen, hIoo⟩ := exists_crossing_interval hT hM hlip h0 hTb
  calc ENNReal.ofReal ((b - a) / M)
      ≤ ENNReal.ofReal (t₂ - t₁) := ENNReal.ofReal_le_ofReal hlen
    _ = volume (Set.Ioo t₁ t₂) := Real.volume_Ioo.symm
    _ ≤ bandTime γ T a b := measure_mono hIoo

/-- The occupancy time of a bounded-speed crossing path is **strictly positive** whenever the
band has positive width, with the explicit rate `(b - a) / M`. This is the conclusion FW-2's
strictness clause consumes: the diffusion advantage carried by the band is not something a path
of bounded speed can route around. The bare positivity here is weaker than
`bandTime_pos_of_crossing`, which needs no speed bound at all; what the speed bound contributes
is the rate, and it is the rate that FW-2 uses quantitatively. -/
theorem bandTime_pos_of_bounded_speed (hT : 0 ≤ T) (hab : a < b) (hM : 0 < M)
    (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T)) (h0 : γ 0 ≤ a) (hTb : b ≤ γ T) :
    0 < bandTime γ T a b :=
  lt_of_lt_of_le (ENNReal.ofReal_pos.2 (div_pos (by linarith) hM))
    (bandTime_ge_width_div_speed hT hM hlip h0 hTb)

/-- The same bound for a path crossing the band **downwards**, which is the direction of an exit
from the well in the note's geometry. It follows from the ascending case applied to `-γ` on the
reflected band `[-b, -a]`. -/
theorem bandTime_ge_width_div_speed_of_descending (hT : 0 ≤ T) (hM : 0 < M)
    (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T)) (h0 : b ≤ γ 0) (hTb : γ T ≤ a) :
    ENNReal.ofReal ((b - a) / M) ≤ bandTime γ T a b := by
  have hneg : LipschitzOnWith M.toNNReal (fun t => -γ t) (Set.Icc 0 T) := by
    refine LipschitzOnWith.of_dist_le_mul fun x hx y hy => ?_
    have h := hlip.dist_le_mul x hx y hy
    show dist (-γ x) (-γ y) ≤ (M.toNNReal : ℝ) * dist x y
    rw [Real.dist_eq] at h ⊢
    rw [show -γ x - -γ y = -(γ x - γ y) by ring, abs_neg]
    exact h
  have hbands : bandTimes (fun t => -γ t) T (-b) (-a) = bandTimes γ T a b := by
    ext t
    constructor
    · rintro ⟨htw, h1, h2⟩
      exact ⟨htw, by linarith, by linarith⟩
    · rintro ⟨htw, h1, h2⟩
      exact ⟨htw, by linarith, by linarith⟩
  have heq : bandTime (fun t => -γ t) T (-b) (-a) = bandTime γ T a b := by
    unfold bandTime
    rw [hbands]
  have h := bandTime_ge_width_div_speed (γ := fun t => -γ t) (a := -b) (b := -a) hT hM hneg
    (neg_le_neg h0) (neg_le_neg hTb)
  rw [heq, show (-a : ℝ) - -b = b - a from by ring] at h
  exact h

end Band

/-! ### The slab form: a path in a metric space, read through a `1`-Lipschitz scalar -/

/-- **The slab statement of the note, in an arbitrary metric space.** The note's FW-3a concerns a
path in `ℝ^d` crossing a slab of width `w` that separates the well from the exit set. Present the
slab through a `1`-Lipschitz scalar readout `φ`, so that the slab is `φ⁻¹[a, b]` and its width is
`b - a`. Then an `M`-Lipschitz path that starts on one side and ends on the other spends time at
least `(b - a) / M` inside the slab.

This is the one-dimensional theorem transported along `φ`, and the transport is exactly the
observation that the composite `φ ∘ γ` is again `M`-Lipschitz. The readout the note itself uses,
the unit linear functional `x ↦ ⟪n, x⟫` normal to the slab, is shown to be `1`-Lipschitz in
`lipschitzWith_one_inner_left`, and the resulting `ℝ^d` statement is
`slabTime_ge_width_div_speed_inner`; `slabTime_le_bandOccupancy` then passes from the slab to a
band containing it. -/
theorem slabTime_ge_width_div_speed {E : Type*} [PseudoMetricSpace E] {γ : ℝ → E} {φ : E → ℝ}
    {T a b M : ℝ} (hT : 0 ≤ T) (hM : 0 < M) (hφ : LipschitzWith 1 φ)
    (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T)) (h0 : φ (γ 0) ≤ a) (hTb : b ≤ φ (γ T)) :
    ENNReal.ofReal ((b - a) / M) ≤ bandTime (φ ∘ γ) T a b := by
  have hcomp : LipschitzOnWith M.toNNReal (φ ∘ γ) (Set.Icc 0 T) := by
    have := hφ.comp_lipschitzOnWith hlip
    rwa [one_mul] at this
  exact bandTime_ge_width_div_speed hT hM hcomp h0 hTb

/-- **From the slab to the band.** The note assumes only that the shoulder band `B` *contains* a
slab of width `w`, and its conclusion is about the time spent in `B`. Since occupancy is monotone
in the target set, every lower bound proved for the slab is a lower bound for `B`. -/
theorem slabTime_le_bandOccupancy {E : Type*} {γ : ℝ → E} {φ : E → ℝ} {B : Set E} {T a b : ℝ}
    (hsub : φ ⁻¹' Set.Icc a b ⊆ B) :
    bandTime (φ ∘ γ) T a b ≤ volume (Set.Icc 0 T ∩ γ ⁻¹' B) :=
  measure_mono (Set.inter_subset_inter (le_refl _) (Set.preimage_mono hsub))

/-- **Occupancy of the band itself.** Combining the slab bound with the inclusion of the slab in
the band: a path of speed at most `M` that crosses a slab `φ⁻¹[a, b]` contained in a band `B`
spends time at least `(b - a) / M` inside `B`. This is the shape of the note's conclusion, whose
subject is the band and not the slab. -/
theorem bandOccupancy_ge_width_div_speed {E : Type*} [PseudoMetricSpace E] {γ : ℝ → E}
    {φ : E → ℝ} {B : Set E} {T a b M : ℝ} (hT : 0 ≤ T) (hM : 0 < M) (hφ : LipschitzWith 1 φ)
    (hslab : φ ⁻¹' Set.Icc a b ⊆ B) (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T))
    (h0 : φ (γ 0) ≤ a) (hTb : b ≤ φ (γ T)) :
    ENNReal.ofReal ((b - a) / M) ≤ volume (Set.Icc 0 T ∩ γ ⁻¹' B) :=
  le_trans (slabTime_ge_width_div_speed hT hM hφ hlip h0 hTb) (slabTime_le_bandOccupancy hslab)

/-- The note presents its slab as the region between two parallel hyperplanes, that is as
`{x : a ≤ ⟪n, x⟫ ≤ b}` for a **unit** normal `n`, the width of the slab being `b - a`. The readout
`x ↦ ⟪n, x⟫` is `1`-Lipschitz, by Cauchy–Schwarz and `‖n‖ = 1`. This is the fact that makes
`slabTime_ge_width_div_speed` apply to the note's geometry rather than to an abstract readout. -/
theorem lipschitzWith_one_inner_left {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {n : E} (hn : ‖n‖ = 1) : LipschitzWith 1 (fun x : E => (inner ℝ n x : ℝ)) := by
  refine LipschitzWith.of_dist_le_mul fun x y => ?_
  have hsub : (inner ℝ n x : ℝ) - inner ℝ n y = inner ℝ n (x - y) := (inner_sub_right ..).symm
  rw [Real.dist_eq, hsub, NNReal.coe_one, one_mul, dist_eq_norm]
  calc |(inner ℝ n (x - y) : ℝ)| ≤ ‖n‖ * ‖x - y‖ := abs_real_inner_le_norm n (x - y)
    _ = ‖x - y‖ := by rw [hn, one_mul]

/-- **The note's slab statement in `ℝ^d`.** Let `n` be a unit vector and let the slab be
`{x : a ≤ ⟪n, x⟫ ≤ b}`, of width `w = b - a`. A path of speed at most `M` that begins on the
`⟪n, ·⟫ ≤ a` side and ends on the `⟪n, ·⟫ ≥ b` side spends time at least `w / M` inside the slab.
The hypothesis on the path is the speed bound, not an action bound; see the module docstring for
the difference, which is the unformalised half of FW-3a. -/
theorem slabTime_ge_width_div_speed_inner {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {γ : ℝ → E} {n : E} {T a b M : ℝ} (hT : 0 ≤ T) (hM : 0 < M)
    (hn : ‖n‖ = 1) (hlip : LipschitzOnWith M.toNNReal γ (Set.Icc 0 T))
    (h0 : (inner ℝ n (γ 0) : ℝ) ≤ a) (hTb : b ≤ (inner ℝ n (γ T) : ℝ)) :
    ENNReal.ofReal ((b - a) / M) ≤ bandTime (fun t => (inner ℝ n (γ t) : ℝ)) T a b :=
  slabTime_ge_width_div_speed hT hM (lipschitzWith_one_inner_left hn) hlip h0 hTb

/-! ### Part 3: the note's own `τ`, and the corollary in the note's terms -/

/-- The note's **band occupancy bound**
`τ(w, Λ, S̄, M) = (√((4 Λ S̄)² + 2 M² w²) − 4 Λ S̄) / (2 M²)`,
the positive root of `M² t² + 4 Λ S̄ t − w²/2 = 0`. Here `w` is the width of the slab, `Λ` the
upper bound on the diffusion (`Σ ≤ Λ I`), `S̄` the admissible action, and `M` the bound on the
drift. -/
noncomputable def bandOccupancyTime (Λ Sbar M w : ℝ) : ℝ :=
  (Real.sqrt ((4 * Λ * Sbar) ^ 2 + 2 * M ^ 2 * w ^ 2) - 4 * Λ * Sbar) / (2 * M ^ 2)

/-- The note's `τ` is **strictly positive for every positive slab width**, with no sign condition
on `Λ S̄` at all: `√(A² + 2M²w²) > |A| ≥ A`. This is the remark the note makes after the proof —
`τ` degrades as the admissible action grows or the band narrows, but never reaches zero. -/
theorem bandOccupancyTime_pos {Λ Sbar M w : ℝ} (hM : 0 < M) (hw : 0 < w) :
    0 < bandOccupancyTime Λ Sbar M w := by
  have hM2 : (0 : ℝ) < 2 * M ^ 2 := by positivity
  have hpos : (0 : ℝ) < 2 * M ^ 2 * w ^ 2 := by positivity
  have h1 : (4 * Λ * Sbar) ^ 2 < (4 * Λ * Sbar) ^ 2 + 2 * M ^ 2 * w ^ 2 := by linarith
  have h2 : |4 * Λ * Sbar| < Real.sqrt ((4 * Λ * Sbar) ^ 2 + 2 * M ^ 2 * w ^ 2) := by
    rw [← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_lt_sqrt (sq_nonneg _) h1
  have h3 : 4 * Λ * Sbar < Real.sqrt ((4 * Λ * Sbar) ^ 2 + 2 * M ^ 2 * w ^ 2) :=
    lt_of_le_of_lt (le_abs_self _) h2
  unfold bandOccupancyTime
  exact div_pos (by linarith) hM2

/-- **The note's `τ` is strictly smaller than the transit theorems' `w / M`**, for the same
numerical `M` and any admissible `Λ S̄ ≥ 0`. This is the precise comparison between the two halves
of this file. The transit theorems assume a bound on the speed of the path and conclude an
occupancy time of at least `w / M`; the note assumes only an action bound, from which the speed is
controlled in mean square, and concludes the smaller `τ`. The inequality is therefore not a
competition between two proofs of the same statement: it records that the Lipschitz hypothesis is
strictly stronger and buys a strictly better constant, and it forbids the mistake of reading
`w / M` as a formalisation of `τ`. Equality is never attained for `w > 0`, and already at
`Λ S̄ = 0` one has `τ = w / (M √2)`. -/
theorem bandOccupancyTime_lt_width_div_speed {Λ Sbar M w : ℝ} (hM : 0 < M) (hΛ : 0 ≤ Λ)
    (hS : 0 ≤ Sbar) (hw : 0 < w) : bandOccupancyTime Λ Sbar M w < w / M := by
  have hA : (0 : ℝ) ≤ 4 * Λ * Sbar := by positivity
  have hM2 : (0 : ℝ) < 2 * M ^ 2 := by positivity
  have hR : (0 : ℝ) ≤ 4 * Λ * Sbar + 2 * M * w := by nlinarith
  have key : (4 * Λ * Sbar) ^ 2 + 2 * M ^ 2 * w ^ 2 < (4 * Λ * Sbar + 2 * M * w) ^ 2 := by
    nlinarith [mul_nonneg (mul_nonneg hA hM.le) hw.le, mul_pos (mul_pos hM hM) (mul_pos hw hw)]
  have hsq : Real.sqrt ((4 * Λ * Sbar) ^ 2 + 2 * M ^ 2 * w ^ 2) < 4 * Λ * Sbar + 2 * M * w := by
    have h := Real.sqrt_lt_sqrt (by positivity) key
    rwa [Real.sqrt_sq hR] at h
  unfold bandOccupancyTime
  rw [div_lt_div_iff₀ hM2 hM]
  nlinarith [hsq]

/-- The algebraic core of the note's proof: a non-negative `t` satisfying the quadratic
inequality `w² ≤ 2 A t + 2 M² t²` is at least the positive root `(√(A² + 2M²w²) − A) / (2M²)`.

The proof is the one the note gives, run backwards: it suffices that
`√(A² + 2M²w²) ≤ 2M²t + A`, and squaring the right-hand side turns that into exactly `2M²`
times the hypothesis. The non-negativity `0 ≤ A` is needed and is not decoration: for `A < 0`,
`t = w = 0` satisfies the hypothesis while the root is strictly positive. -/
theorem quadratic_positive_root_le {A M w t : ℝ} (hM : 0 < M) (hA : 0 ≤ A) (ht : 0 ≤ t)
    (h : w ^ 2 ≤ 2 * A * t + 2 * M ^ 2 * t ^ 2) :
    (Real.sqrt (A ^ 2 + 2 * M ^ 2 * w ^ 2) - A) / (2 * M ^ 2) ≤ t := by
  have hM2 : (0 : ℝ) < 2 * M ^ 2 := by positivity
  have hR : (0 : ℝ) ≤ 2 * M ^ 2 * t + A := by positivity
  have h2 : 2 * M ^ 2 * w ^ 2 ≤ 2 * M ^ 2 * (2 * A * t + 2 * M ^ 2 * t ^ 2) :=
    mul_le_mul_of_nonneg_left h hM2.le
  have h3 : 2 * M ^ 2 * (2 * A * t + 2 * M ^ 2 * t ^ 2) = 4 * M ^ 4 * t ^ 2 + 4 * M ^ 2 * A * t :=
    by ring
  rw [h3] at h2
  have hexp : (2 * M ^ 2 * t + A) ^ 2 = 4 * M ^ 4 * t ^ 2 + 4 * M ^ 2 * A * t + A ^ 2 := by ring
  have key : A ^ 2 + 2 * M ^ 2 * w ^ 2 ≤ (2 * M ^ 2 * t + A) ^ 2 := by rw [hexp]; linarith
  have hsqrt : Real.sqrt (A ^ 2 + 2 * M ^ 2 * w ^ 2) ≤ 2 * M ^ 2 * t + A :=
    calc Real.sqrt (A ^ 2 + 2 * M ^ 2 * w ^ 2)
        ≤ Real.sqrt ((2 * M ^ 2 * t + A) ^ 2) := Real.sqrt_le_sqrt key
      _ = 2 * M ^ 2 * t + A := Real.sqrt_sq hR
  rw [div_le_iff₀ hM2]
  linarith

/-- The note's inequality `w² ≤ t (8 Λ S̄ + 2 M² t)` forces `t ≥ τ(w, Λ, S̄, M)`. This is the last
line of the note's proof of FW-3a, and it is proved here in full. The real `t` is uninterpreted
here; `bandOccupancyTime_le_bandTime` is the version in which it is the occupancy time of a
path. -/
theorem bandOccupancyTime_le_of_energy_bound {Λ Sbar M w t : ℝ} (hM : 0 < M) (hΛ : 0 ≤ Λ)
    (hS : 0 ≤ Sbar) (ht : 0 ≤ t) (hquad : w ^ 2 ≤ t * (8 * Λ * Sbar + 2 * M ^ 2 * t)) :
    bandOccupancyTime Λ Sbar M w ≤ t := by
  have hA : (0 : ℝ) ≤ 4 * Λ * Sbar := by positivity
  have h : w ^ 2 ≤ 2 * (4 * Λ * Sbar) * t + 2 * M ^ 2 * t ^ 2 := by nlinarith [hquad]
  exact quadratic_positive_root_le hM hA ht h

/-- **The note's conclusion as a statement about a path.** The occupancy time of a path is finite
(`bandTime_ne_top`), so it has a real value `(bandTime γ T a b).toReal`; if that value satisfies
the note's energy inequality, then the note's `τ` is a lower bound for the occupancy time itself,
as an extended-real measure. This is the only statement in the file in which the note's `τ` and
the measure `bandTime` of an actual path appear together, and it is the one to read as FW-3a's
conclusion; `bandOccupancyTime_le_of_energy_bound` and `occupancy_time_of_action_bound` quantify
instead over an uninterpreted real `t`. The energy inequality remains a hypothesis, for the reason
given in the module docstring: the rate functional that would supply it is not definable in this
mathlib. -/
theorem bandOccupancyTime_le_bandTime {γ : ℝ → ℝ} {T a b Λ Sbar M w : ℝ} (hM : 0 < M)
    (hΛ : 0 ≤ Λ) (hS : 0 ≤ Sbar)
    (hquad : w ^ 2 ≤ (bandTime γ T a b).toReal *
      (8 * Λ * Sbar + 2 * M ^ 2 * (bandTime γ T a b).toReal)) :
    ENNReal.ofReal (bandOccupancyTime Λ Sbar M w) ≤ bandTime γ T a b :=
  calc ENNReal.ofReal (bandOccupancyTime Λ Sbar M w)
      ≤ ENNReal.ofReal ((bandTime γ T a b).toReal) :=
        ENNReal.ofReal_le_ofReal
          (bandOccupancyTime_le_of_energy_bound hM hΛ hS ENNReal.toReal_nonneg hquad)
    _ = bandTime γ T a b := ENNReal.ofReal_toReal bandTime_ne_top

/-- **FW-3a in the note's terms.** Let `t` be the time a path spends inside the slab, let `L` be
the length of the piece of path inside the slab, and let `E` be the integral of the squared speed
over that piece. Then the note's three analytic inputs

* `hwL : w ≤ L`   — *transversality*: the path covers Euclidean distance at least the slab width
  `w` while inside the slab, which is the note's hypothesis that the slab separates the well from
  the exit set;
* `hCS : L² ≤ t * E`   — *Cauchy–Schwarz* for the speed over the occupancy set, of measure `t`;
* `hE : E ≤ 8 Λ S̄ + 2 M² t`   — the *action-plus-drift energy bound*, obtained in the note from
  `Σ ≤ Λ I` (so `∫ ‖γ̇ + ∇U‖² ≤ 4 Λ S̄`), the elementary bound `‖γ̇‖² ≤ 2‖γ̇ + ∇U‖² + 2‖∇U‖²`, and
  `‖∇U‖ ≤ M`;

together force `t ≥ τ(w, Λ, S̄, M) > 0`.

All three are hypotheses, not theorems, and each names a step of the note's derivation that
requires the Freidlin–Wentzell rate functional and the path derivative `γ̇` as formal objects;
mathlib `v4.31.0` supplies neither (no Itô integral, no stochastic differential equations, no
large-deviation rate functionals). What is proved here is that, granted the three, the note's `τ`
is correct and positive.

Two things about the statement should be read literally rather than through the prose above. The
reals `t`, `L` and `E` are **uninterpreted**: nothing here ties `t` to `bandTime γ T a b`, so as
it stands this is a fact of arithmetic. `bandOccupancyTime_le_bandTime` is the version whose `t`
is the occupancy time of a path. And the transit theorems
`bandTime_ge_width_div_speed` and `slabTime_ge_width_div_speed` are **not** a substitute for this
one: they reach a conclusion of the same shape, with the different and larger constant `w / M`
(see `bandOccupancyTime_lt_width_div_speed`), from a bound on the speed of the path, which is a
strictly stronger hypothesis than an action bound and is not implied by one. -/
theorem occupancy_time_of_action_bound {Λ Sbar M w t L E : ℝ} (hM : 0 < M) (hw : 0 < w)
    (hΛ : 0 ≤ Λ) (hS : 0 ≤ Sbar) (ht : 0 ≤ t) (hwL : w ≤ L) (hCS : L ^ 2 ≤ t * E)
    (hE : E ≤ 8 * Λ * Sbar + 2 * M ^ 2 * t) :
    0 < bandOccupancyTime Λ Sbar M w ∧ bandOccupancyTime Λ Sbar M w ≤ t := by
  refine ⟨bandOccupancyTime_pos hM hw, ?_⟩
  have hwL2 : w ^ 2 ≤ L ^ 2 := by nlinarith
  have hstep : t * E ≤ t * (8 * Λ * Sbar + 2 * M ^ 2 * t) := mul_le_mul_of_nonneg_left hE ht
  exact bandOccupancyTime_le_of_energy_bound hM hΛ hS ht (by linarith)

/-! ### Witnesses: the hypotheses above are satisfiable, and the constants are attained

A theorem whose hypotheses no instance satisfies is vacuously true and says nothing. The
following declarations exhibit explicit instances of each family of hypotheses used above, so
that the results are checked to be non-vacuous inside this file rather than by inspection. They
also show that the transit constant `(b - a) / M` cannot be improved. -/

/-- The unit-speed path `t ↦ t` crossing the band `[0, 1]` over the window `[0, 1]` occupies the
whole window: its occupancy time is `1`. -/
theorem bandTime_unitCrossing_eq_one : bandTime (fun t : ℝ => t) 1 0 1 = 1 := by
  have h : bandTimes (fun t : ℝ => t) 1 0 1 = Set.Icc (0 : ℝ) 1 := by
    ext t; simp [bandTimes]
  rw [bandTime, h, Real.volume_Icc]
  norm_num

/-- **`bandTime_ge_width_div_speed` is neither vacuous nor improvable.** Its hypotheses are
satisfied by the unit-speed path `t ↦ t` on `[0, 1]` crossing the band `[0, 1]`, and for that
path the bound `(b - a) / M` is attained exactly, so no larger constant is available. -/
theorem bandTime_ge_width_div_speed_sharp :
    ENNReal.ofReal (((1 : ℝ) - 0) / 1) ≤ bandTime (fun t : ℝ => t) 1 0 1 ∧
      bandTime (fun t : ℝ => t) 1 0 1 = ENNReal.ofReal (((1 : ℝ) - 0) / 1) := by
  refine ⟨bandTime_ge_width_div_speed zero_le_one one_pos ?_ le_rfl le_rfl, ?_⟩
  · rw [Real.toNNReal_one]
    exact LipschitzWith.id.lipschitzOnWith
  · rw [bandTime_unitCrossing_eq_one]; norm_num

/-- **`slabTime_ge_width_div_speed_inner` is not vacuous, in any dimension.** In any real inner
product space, the straight unit-speed path `t ↦ t • n` along a unit normal `n` satisfies every
hypothesis of the slab theorem for the slab `{x : 0 ≤ ⟪n, x⟫ ≤ 1}` of width `1` over the window
`[0, 1]`, and spends time at least `1` inside it. This is the note's own picture of a path
crossing a slab transversally. -/
theorem slabTime_straightPath {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {n : E}
    (hn : ‖n‖ = 1) : (1 : ℝ≥0∞) ≤ bandTime (fun t : ℝ => (inner ℝ n (t • n) : ℝ)) 1 0 1 := by
  have hlip : LipschitzOnWith (1 : ℝ).toNNReal (fun t : ℝ => t • n) (Set.Icc 0 1) := by
    rw [Real.toNNReal_one]
    refine LipschitzWith.lipschitzOnWith ?_
    refine LipschitzWith.of_dist_le_mul fun x y => ?_
    rw [dist_eq_norm, ← sub_smul, norm_smul, hn, mul_one, Real.norm_eq_abs, ← Real.dist_eq,
      NNReal.coe_one, one_mul]
  have h0 : (inner ℝ n ((0 : ℝ) • n) : ℝ) ≤ 0 := by rw [zero_smul, inner_zero_right]
  have h1 : (1 : ℝ) ≤ (inner ℝ n ((1 : ℝ) • n) : ℝ) := by
    rw [one_smul, real_inner_self_eq_norm_sq, hn, one_pow]
  have h := slabTime_ge_width_div_speed_inner (n := n) (T := 1) (a := 0) (b := 1) (M := 1)
    zero_le_one one_pos hn hlip h0 h1
  norm_num at h
  exact h

/-- The previous witness instantiated in `ℝ²`, so that the slab theorem is checked to be
non-vacuous in a dimension greater than one and is not a one-dimensional statement in disguise. -/
theorem slabTime_straightPath_euclidean :
    (1 : ℝ≥0∞) ≤ bandTime (fun t : ℝ => (inner ℝ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (t • EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) : ℝ)) 1 0 1 :=
  slabTime_straightPath (by rw [PiLp.norm_single, norm_one])

/-- **`bandOccupancyTime_le_bandTime` is not vacuous.** For the unit-speed path `t ↦ t` on the
window `[0, 1]` crossing the band `[0, 1]`, the occupancy time is `1`, the note's energy
inequality holds with `Λ = S̄ = 0` and `M = w = 1`, and the conclusion is the true bound
`√2 / 2 ≤ 1`. So the statement that joins the two halves of this file has an instance in which
every hypothesis is met by an actual path. -/
theorem bandOccupancyTime_le_bandTime_nonvacuous :
    ENNReal.ofReal (bandOccupancyTime 0 0 1 1) ≤ bandTime (fun t : ℝ => t) 1 0 1 := by
  refine bandOccupancyTime_le_bandTime one_pos le_rfl le_rfl ?_
  rw [bandTime_unitCrossing_eq_one, ENNReal.toReal_one]
  norm_num

/-- **`occupancy_time_of_action_bound` is not vacuous.** Its three analytic hypotheses are
simultaneously satisfiable: with `Λ = S̄ = M = w = L = E = t = 1` the transversality bound reads
`1 ≤ 1`, Cauchy–Schwarz reads `1 ≤ 1`, and the energy bound reads `1 ≤ 10`, so the conclusion
applies and yields the true inequality `0 < (√18 - 4)/2 ≤ 1`. -/
theorem occupancy_time_of_action_bound_nonvacuous :
    0 < bandOccupancyTime 1 1 1 1 ∧ bandOccupancyTime 1 1 1 1 ≤ 1 :=
  occupancy_time_of_action_bound (L := 1) (E := 1) one_pos one_pos zero_le_one zero_le_one
    zero_le_one le_rfl (by norm_num) (by norm_num)

end IcnnLift
