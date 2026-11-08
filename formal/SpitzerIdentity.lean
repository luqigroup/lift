import Mathlib

/-!
# Spitzer's ladder-epoch identity for a random walk with i.i.d. increments

Let `X_0, X_1, …` be i.i.d. real random variables and `S_n = X_0 + ⋯ + X_{n-1}`, so `S_0 = 0`.
This file proves

    P(S_n ≤ 0 for all n) = exp(-∑_{n≥1} n⁻¹ P(S_n > 0))

whenever the series converges (`spitzer_ladder_identity`), and that the series converges whenever
the left side is positive (`summable_of_measureReal_forall_le_pos`). Mathlib has no fluctuation
theory, so the ladder epochs, their renewal structure and the exchangeability argument are built
here. The one deterministic input, the cyclic lemma, is taken as the hypothesis `hcyc`, stated
verbatim in each theorem (`CyclicLemma` is the same proposition under a name). It is a statement
about finite real sequences, with no probability in it.

## The argument, step by step

A *ladder epoch* of a path `x` is an index `j ≥ 1` with `s_j > s_i` for every `i < j`, where
`s_j = x_0 + ⋯ + x_{j-1}` and `s_0 = 0`. Write `N_n` for the number of ladder epochs in `[1, n]`,
`{τ_r = n}` for the event that `n` is a ladder epoch and `N_n = r`, and `q = P(τ_1 < ∞)`.

1. **Ladder epochs, pathwise** (`IsLadderEpoch`, `ladderEpochs`, `IsLadderTime`,
   `isLadderTime_unique`, `hasLadderEpoch_iff`). The `r`-th ladder epoch is unique, and a path
   has a ladder epoch exactly when some `s_n > 0`.
2. **Exchangeability** (`map_reindex_eq`). The law of an i.i.d. sequence is the infinite product
   of its marginal, and so is the law of the sequence read through any injective index map. The
   rotation of the first `n` coordinates and the shift by `m` are both such maps.
3. **The cyclic lemma in expectation** (`sum_ladderWeight_rotPath`, `measure_pathSum_pos`).
   Summed over the `n` rotations of `(x_0, …, x_{n-1})`, the weight `1{n is a ladder epoch}/N_n`
   adds up to `1{s_n > 0}`. In expectation every rotation contributes the same amount, so
   `P(S_n > 0) = n ∑_{r=1}^n r⁻¹ P(τ_r = n)`.
4. **Renewal** (`exists_isLadderTime_succ_iff`, `measure_isLadderTime_inter_shift`,
   `measure_exists_isLadderTime`). There is an `(r+1)`-th ladder epoch exactly when there is an
   `r`-th one, at some `m`, and the path restarted at `m` has a ladder epoch of its own. The first
   event is determined by `X_0, …, X_{m-1}` and the second by `X_m, X_{m+1}, …`, so the two are
   independent, and the second has probability `q` by step 2. Summing over `m` gives
   `P(τ_r < ∞) = q^r`.
5. **Tonelli at `s = 1`** (`tsum_measure_pathSum_pos_div`). Every term is non-negative, so in
   `ℝ≥0∞` the double series may be summed in either order:
   `∑_{n≥1} n⁻¹ P(S_n > 0) = ∑_{r≥1} r⁻¹ P(τ_r < ∞) = ∑_{r≥1} r⁻¹ q^r`. No generating function
   is used.
6. **Conclusion** (`tsum_inv_succ_eq_top`, `tsum_inv_succ_mul_pow`, `measureReal_forall_le`).
   A finite left side forces `q < 1`, because the harmonic series diverges, and then
   `∑_{r≥1} r⁻¹ q^r = -log(1 - q)`. Since `1 - q = P(S_n ≤ 0 for all n)`, exponentiating gives
   the identity.

## Hypotheses

* `iIndepFun X P` (mutual independence), `∀ i, Measurable (X i)`, and
  `∀ i, IdentDistrib (X i) (X 0) P P`. No moment condition is assumed.
* `hcyc`, the cyclic lemma. A ladder epoch of the rotated sequence `y_l = x_{(l + r) mod n}` is
  computed from the partial sums of `y`, with `s_0 = 0` included.
* `hsum`, the convergence of `∑_{n≥0} P(S_{n+1} > 0)/(n + 1)`. It cannot be dropped from the
  statement as written: for a divergent series Lean's `tsum` is `0` by convention, which would
  make the right side `1`. The true statement in that case, `P(S_n ≤ 0 for all n) = 0`, is
  `measureReal_forall_le_eq_zero_of_not_summable`.

The increments are indexed from `0`, so the `n`-th step of the walk is `X_{n-1}`, and the series
is written over `n + 1` with `n : ℕ`.
-/

open MeasureTheory ProbabilityTheory Finset
open scoped ENNReal

namespace IcnnLift
namespace SpitzerLadder

/-! ### Step 1: ladder epochs, pathwise -/
/-- The partial sum `s_j = x_0 + ⋯ + x_{j-1}` of a path, with `s_0 = 0`. -/
noncomputable def pathSum (x : ℕ → ℝ) (j : ℕ) : ℝ := ∑ l ∈ range j, x l

/-- `j` is a (strict ascending) ladder epoch: `s_j` exceeds every earlier partial sum. -/
def IsLadderEpoch (x : ℕ → ℝ) (j : ℕ) : Prop := ∀ i < j, pathSum x i < pathSum x j

/-- The ladder epochs of `x` in `[1, n]`. -/
noncomputable def ladderEpochs (x : ℕ → ℝ) (n : ℕ) : Finset ℕ := by
  classical exact (Icc 1 n).filter (fun j => IsLadderEpoch x j)

/-- `τ_r = n`: `n` is a ladder epoch and it is the `r`-th one. -/
def IsLadderTime (x : ℕ → ℝ) (r n : ℕ) : Prop :=
  IsLadderEpoch x n ∧ (ladderEpochs x n).card = r

/-- The path has at least one ladder epoch. -/
def HasLadderEpoch (x : ℕ → ℝ) : Prop := ∃ j, 1 ≤ j ∧ IsLadderEpoch x j

theorem pathSum_zero (x : ℕ → ℝ) : pathSum x 0 = 0 := by simp [pathSum]

theorem mem_ladderEpochs {x : ℕ → ℝ} {n j : ℕ} :
    j ∈ ladderEpochs x n ↔ (1 ≤ j ∧ j ≤ n) ∧ IsLadderEpoch x j := by
  classical
  simp [ladderEpochs]

theorem pathSum_shift (x : ℕ → ℝ) (m j : ℕ) :
    pathSum (fun i => x (m + i)) j = pathSum x (m + j) - pathSum x m := by
  simp only [pathSum, Finset.sum_range_add]
  ring

theorem pathSum_congr {x y : ℕ → ℝ} {j : ℕ} (h : ∀ l < j, x l = y l) :
    pathSum x j = pathSum y j :=
  Finset.sum_congr rfl fun l hl => h l (Finset.mem_range.mp hl)

theorem isLadderEpoch_congr {x y : ℕ → ℝ} {j : ℕ} (h : ∀ l < j, x l = y l) :
    IsLadderEpoch x j ↔ IsLadderEpoch y j := by
  unfold IsLadderEpoch
  refine forall₂_congr fun i hi => ?_
  rw [pathSum_congr h, pathSum_congr fun l hl => h l (lt_trans hl hi)]

theorem ladderEpochs_congr {x y : ℕ → ℝ} {n : ℕ} (h : ∀ l < n, x l = y l) :
    ladderEpochs x n = ladderEpochs y n := by
  ext j
  simp only [mem_ladderEpochs]
  constructor
  · rintro ⟨hj, hl⟩
    exact ⟨hj, (isLadderEpoch_congr fun l hl' => h l (by omega)).1 hl⟩
  · rintro ⟨hj, hl⟩
    exact ⟨hj, (isLadderEpoch_congr fun l hl' => h l (by omega)).2 hl⟩

theorem isLadderTime_congr {x y : ℕ → ℝ} {r n : ℕ} (h : ∀ l < n, x l = y l) :
    IsLadderTime x r n ↔ IsLadderTime y r n := by
  unfold IsLadderTime
  rw [isLadderEpoch_congr h, ladderEpochs_congr h]

theorem ladderEpochs_zero (x : ℕ → ℝ) : ladderEpochs x 0 = ∅ := by
  ext j
  simp only [mem_ladderEpochs, Finset.notMem_empty, iff_false]
  omega

theorem card_ladderEpochs_le (x : ℕ → ℝ) (n : ℕ) : (ladderEpochs x n).card ≤ n := by
  classical
  calc (ladderEpochs x n).card ≤ (Icc 1 n).card := by
        unfold ladderEpochs; exact Finset.card_filter_le _ _
    _ = n := by simp

theorem ladderEpochs_mono (x : ℕ → ℝ) {n n' : ℕ} (h : n ≤ n') :
    ladderEpochs x n ⊆ ladderEpochs x n' := by
  intro j hj
  rw [mem_ladderEpochs] at hj ⊢
  exact ⟨⟨hj.1.1, hj.1.2.trans h⟩, hj.2⟩

theorem self_mem_ladderEpochs {x : ℕ → ℝ} {n : ℕ} (hn : 1 ≤ n) (h : IsLadderEpoch x n) :
    n ∈ ladderEpochs x n :=
  mem_ladderEpochs.2 ⟨⟨hn, le_rfl⟩, h⟩

theorem isLadderTime_zero_zero (x : ℕ → ℝ) : IsLadderTime x 0 0 :=
  ⟨fun i hi => absurd hi (Nat.not_lt_zero i), by rw [ladderEpochs_zero]; rfl⟩

theorem not_isLadderTime_succ_zero (x : ℕ → ℝ) (r : ℕ) : ¬ IsLadderTime x (r + 1) 0 := by
  rintro ⟨-, h⟩
  rw [ladderEpochs_zero] at h
  simp at h

theorem lt_of_isLadderTime {x : ℕ → ℝ} {r n : ℕ} (h : IsLadderTime x (r + 1) n) : r < n := by
  have := card_ladderEpochs_le x n
  rw [h.2] at this
  omega

/-- **The `r`-th ladder epoch is unique.** Two ladder epochs that are both the `r`-th coincide,
because a later ladder epoch has strictly more ladder epochs before it. -/
theorem isLadderTime_unique {x : ℕ → ℝ} {r n n' : ℕ} (h : IsLadderTime x r n)
    (h' : IsLadderTime x r n') : n = n' := by
  have key : ∀ {a b : ℕ}, IsLadderTime x r a → IsLadderTime x r b → a < b → False := by
    intro a b ha hb hab
    have hsub : ladderEpochs x a ⊂ ladderEpochs x b := by
      refine Finset.ssubset_iff_subset_ne.2 ⟨ladderEpochs_mono x hab.le, fun heq => ?_⟩
      have hb' : b ∈ ladderEpochs x b := self_mem_ladderEpochs (by omega) hb.1
      rw [← heq, mem_ladderEpochs] at hb'
      omega
    have := Finset.card_lt_card hsub
    rw [ha.2, hb.2] at this
    exact lt_irrefl _ this
  rcases lt_trichotomy n n' with hlt | heq | hgt
  · exact (key h h' hlt).elim
  · exact heq
  · exact (key h' h hgt).elim

/-- A path has a ladder epoch exactly when some partial sum is positive: the first positive
partial sum is a strict record. -/
theorem hasLadderEpoch_iff (x : ℕ → ℝ) : HasLadderEpoch x ↔ ∃ n, 0 < pathSum x n := by
  classical
  constructor
  · rintro ⟨j, hj, hl⟩
    exact ⟨j, by simpa [pathSum_zero] using hl 0 (by omega)⟩
  · intro h
    refine ⟨Nat.find h, ?_, fun i hi => ?_⟩
    · by_contra h0
      have : Nat.find h = 0 := by omega
      have hs := Nat.find_spec h
      rw [this, pathSum_zero] at hs
      exact lt_irrefl _ hs
    · have := Nat.find_min h hi
      push Not at this
      exact lt_of_le_of_lt this (Nat.find_spec h)

/-- **The renewal step, pathwise.** There is an `(r+1)`-th ladder epoch exactly when there is an
`r`-th one, at some `m`, and the path restarted at `m` has a ladder epoch of its own: after a
strict record at `m`, the next strict record is the first strict record of `x(m + ·)`. -/
theorem exists_isLadderTime_succ_iff (x : ℕ → ℝ) (r : ℕ) :
    (∃ n, IsLadderTime x (r + 1) n)
      ↔ ∃ m, IsLadderTime x r m ∧ HasLadderEpoch (fun i => x (m + i)) := by
  classical
  constructor
  · rintro ⟨n, hn⟩
    have hn1 : 1 ≤ n := by have := lt_of_isLadderTime hn; omega
    set m := Nat.findGreatest (fun j => IsLadderEpoch x j) (n - 1) with hm
    have hmle : m ≤ n - 1 := Nat.findGreatest_le _
    have hmL : IsLadderEpoch x m :=
      Nat.findGreatest_spec (P := fun j => IsLadderEpoch x j) (Nat.zero_le _)
        (fun i hi => absurd hi (Nat.not_lt_zero i))
    have hset : ladderEpochs x m = (ladderEpochs x n).erase n := by
      ext j
      rw [Finset.mem_erase, mem_ladderEpochs, mem_ladderEpochs]
      constructor
      · rintro ⟨⟨h1, h2⟩, hl⟩
        exact ⟨by omega, ⟨h1, by omega⟩, hl⟩
      · rintro ⟨hne, ⟨h1, h2⟩, hl⟩
        refine ⟨⟨h1, ?_⟩, hl⟩
        by_contra hgt
        exact Nat.findGreatest_is_greatest (P := fun j => IsLadderEpoch x j)
          (Nat.lt_of_not_le hgt) (by omega) hl
    refine ⟨m, ⟨hmL, ?_⟩, n - m, by omega, fun i hi => ?_⟩
    · rw [hset, Finset.card_erase_of_mem (self_mem_ladderEpochs hn1 hn.1), hn.2]
      rfl
    · rw [pathSum_shift, pathSum_shift, show m + (n - m) = n by omega]
      have := hn.1 (m + i) (by omega)
      linarith
  · rintro ⟨m, hm, hH⟩
    let j₀ := Nat.find hH
    have hj₀ : 1 ≤ j₀ ∧ IsLadderEpoch (fun i => x (m + i)) j₀ := Nat.find_spec hH
    have hrise : ∀ i < j₀, pathSum x (m + i) < pathSum x (m + j₀) := by
      intro i hi
      have := hj₀.2 i hi
      rw [pathSum_shift, pathSum_shift] at this
      linarith
    have hL : IsLadderEpoch x (m + j₀) := by
      intro i hi
      by_cases him : m ≤ i
      · have := hrise (i - m) (by omega)
        rwa [show m + (i - m) = i by omega] at this
      · have h1 := hm.1 i (by omega)
        have h2 := hrise 0 (by omega)
        rw [add_zero] at h2
        linarith
    refine ⟨m + j₀, hL, ?_⟩
    have hset : ladderEpochs x (m + j₀) = insert (m + j₀) (ladderEpochs x m) := by
      ext j
      rw [Finset.mem_insert, mem_ladderEpochs, mem_ladderEpochs]
      constructor
      · rintro ⟨⟨h1, h2⟩, hl⟩
        by_cases hjm : j ≤ m
        · exact Or.inr ⟨⟨h1, hjm⟩, hl⟩
        by_cases hjn : j = m + j₀
        · exact Or.inl hjn
        exfalso
        have hmin := Nat.find_min hH (show j - m < j₀ by omega)
        refine hmin ⟨by omega, fun i hi => ?_⟩
        rw [pathSum_shift, pathSum_shift, show m + (j - m) = j by omega]
        have := hl (m + i) (by omega)
        linarith
      · rintro (h | ⟨⟨h1, h2⟩, hl⟩)
        · subst h; exact ⟨⟨by omega, le_rfl⟩, hL⟩
        · exact ⟨⟨h1, by omega⟩, hl⟩
    rw [hset, Finset.card_insert_of_notMem, hm.2]
    rw [mem_ladderEpochs]
    omega


/-! ### Rotations of the first `n` coordinates, and the cyclic lemma in weighted form -/

/-- The index map that rotates the first `n` coordinates by `k` and fixes the rest. -/
def rotIdx (n k : ℕ) (i : ℕ) : ℕ := if i < n then (i + k) % n else i

/-- The path with its first `n` coordinates rotated by `k`. -/
def rotPath (n k : ℕ) (x : ℕ → ℝ) : ℕ → ℝ := fun i => x (rotIdx n k i)

theorem rotIdx_injective (n k : ℕ) : Function.Injective (rotIdx n k) := by
  intro a b hab
  unfold rotIdx at hab
  by_cases ha : a < n <;> by_cases hb : b < n <;> simp only [ha, hb, if_true, if_false] at hab
  · have hmod : a % n = b % n := Nat.ModEq.add_right_cancel' k hab
    rwa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at hmod
  · have : (a + k) % n < n := Nat.mod_lt _ (by omega)
    omega
  · have : (b + k) % n < n := Nat.mod_lt _ (by omega)
    omega
  · exact hab

/-- The statement of the deterministic cyclic lemma, taken as a hypothesis. -/
def CyclicLemma : Prop :=
  ∀ (n : ℕ) (x : ℕ → ℝ), 0 < n →
    (∑ r ∈ Finset.range n,
      if (∀ i < n, ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range n, x ((l + r) % n))
      then (((Finset.Icc 1 n).filter (fun j => ∀ i < j,
              ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range j, x ((l + r) % n))).card : ℝ)⁻¹
      else 0)
    = if 0 < ∑ l ∈ Finset.range n, x l then 1 else 0

/-- The weight `1/N_n` carried by a path on which `n` is a ladder epoch, `N_n` being the number of
ladder epochs in `[1, n]`, written as `∑_{r ≥ 1} r⁻¹ 1{τ_r = n}`. -/
noncomputable def ladderWeight (n : ℕ) (y : ℕ → ℝ) : ℝ≥0∞ :=
  ∑ r ∈ range n, ((r : ℝ≥0∞) + 1)⁻¹ * {z | IsLadderTime z (r + 1) n}.indicator 1 y

theorem pathSum_rotPath (n k : ℕ) (x : ℕ → ℝ) {i : ℕ} (hi : i ≤ n) :
    ∑ l ∈ range i, x ((l + k) % n) = pathSum (rotPath n k x) i := by
  unfold pathSum rotPath rotIdx
  refine Finset.sum_congr rfl fun l hl => ?_
  rw [if_pos (by have := Finset.mem_range.mp hl; omega)]

/-- One summand of the cyclic lemma is the weight of the rotated path. -/
theorem cyclic_summand_eq (n k : ℕ) (x : ℕ → ℝ) (hn : 0 < n) :
    ENNReal.ofReal
      (if (∀ i < n, ∑ l ∈ Finset.range i, x ((l + k) % n) < ∑ l ∈ Finset.range n, x ((l + k) % n))
      then (((Finset.Icc 1 n).filter (fun j => ∀ i < j,
              ∑ l ∈ Finset.range i, x ((l + k) % n) < ∑ l ∈ Finset.range j, x ((l + k) % n))).card : ℝ)⁻¹
      else 0) = ladderWeight n (rotPath n k x) := by
  classical
  set y := rotPath n k x with hy
  have hC : (∀ i < n, ∑ l ∈ Finset.range i, x ((l + k) % n) < ∑ l ∈ Finset.range n, x ((l + k) % n))
      ↔ IsLadderEpoch y n := by
    unfold IsLadderEpoch
    refine forall₂_congr fun i hi => ?_
    rw [pathSum_rotPath n k x hi.le, pathSum_rotPath n k x le_rfl]
  have hF : (Finset.Icc 1 n).filter (fun j => ∀ i < j,
      ∑ l ∈ Finset.range i, x ((l + k) % n) < ∑ l ∈ Finset.range j, x ((l + k) % n))
      = ladderEpochs y n := by
    unfold ladderEpochs
    refine Finset.filter_congr fun j hj => ?_
    have hjn : j ≤ n := (Finset.mem_Icc.mp hj).2
    unfold IsLadderEpoch
    refine forall₂_congr fun i hi => ?_
    rw [pathSum_rotPath n k x (by omega), pathSum_rotPath n k x hjn]
  unfold ladderWeight
  by_cases hL : IsLadderEpoch y n
  · rw [if_pos (hC.2 hL), hF]
    set c := (ladderEpochs y n).card with hc
    have hc1 : 1 ≤ c := Finset.card_pos.2 ⟨n, self_mem_ladderEpochs hn hL⟩
    have hcn : c ≤ n := card_ladderEpochs_le y n
    rw [Finset.sum_eq_single (c - 1)]
    · have hmem : y ∈ {z | IsLadderTime z (c - 1 + 1) n} := ⟨hL, by omega⟩
      rw [Set.indicator_of_mem hmem, Pi.one_apply, mul_one,
        ENNReal.ofReal_inv_of_pos (by exact_mod_cast hc1), ENNReal.ofReal_natCast]
      congr 1
      rw [show ((c - 1 : ℕ) : ℝ≥0∞) + 1 = ((c - 1 + 1 : ℕ) : ℝ≥0∞) by push_cast; ring,
        Nat.sub_add_cancel hc1]
    · intro r _ hr
      have hnot : y ∉ {z | IsLadderTime z (r + 1) n} := by
        rintro ⟨-, h⟩
        omega
      rw [Set.indicator_of_notMem hnot, mul_zero]
    · intro h
      exact absurd (Finset.mem_range.2 (by omega)) h
  · rw [if_neg (mt hC.1 hL), ENNReal.ofReal_zero]
    symm
    refine Finset.sum_eq_zero fun r _ => ?_
    have hnot : y ∉ {z | IsLadderTime z (r + 1) n} := fun h => hL h.1
    rw [Set.indicator_of_notMem hnot, mul_zero]

/-- **The cyclic lemma in weighted form.** Summed over the `n` rotations of the first `n`
coordinates, the weights `1/N_n` add up to the indicator of `s_n > 0`. -/
theorem sum_ladderWeight_rotPath (hcyc : CyclicLemma) {n : ℕ} (hn : 0 < n) (x : ℕ → ℝ) :
    ∑ k ∈ range n, ladderWeight n (rotPath n k x) = {z | 0 < pathSum z n}.indicator 1 x := by
  classical
  have h := hcyc n x hn
  have h2 := congrArg ENNReal.ofReal h
  rw [ENNReal.ofReal_sum_of_nonneg] at h2
  · simp_rw [cyclic_summand_eq n _ x hn] at h2
    rw [h2]
    by_cases hp : 0 < pathSum x n
    · rw [Set.indicator_of_mem (show x ∈ {z | 0 < pathSum z n} from hp), Pi.one_apply]
      rw [if_pos (show 0 < ∑ l ∈ range n, x l from hp), ENNReal.ofReal_one]
    · rw [Set.indicator_of_notMem (show x ∉ {z | 0 < pathSum z n} from hp)]
      rw [if_neg (show ¬ 0 < ∑ l ∈ range n, x l from hp), ENNReal.ofReal_zero]
  · intro k _
    split_ifs
    · exact inv_nonneg.2 (Nat.cast_nonneg _)
    · exact le_rfl


/-! ### Measurability on path space -/

theorem measurable_pathSum (j : ℕ) : Measurable (fun x : ℕ → ℝ => pathSum x j) := by
  unfold pathSum
  fun_prop

theorem measurableSet_isLadderEpoch (j : ℕ) : MeasurableSet {x : ℕ → ℝ | IsLadderEpoch x j} := by
  have : {x : ℕ → ℝ | IsLadderEpoch x j} = ⋂ i ∈ range j, {x | pathSum x i < pathSum x j} := by
    ext x
    simp [IsLadderEpoch]
  rw [this]
  exact Finset.measurableSet_biInter _ fun i _ =>
    measurableSet_lt (measurable_pathSum i) (measurable_pathSum j)

theorem measurable_card_ladderEpochs (n : ℕ) :
    Measurable (fun x : ℕ → ℝ => ((ladderEpochs x n).card : ℝ)) := by
  classical
  have : (fun x : ℕ → ℝ => ((ladderEpochs x n).card : ℝ))
      = fun x => ∑ j ∈ Icc 1 n, {z : ℕ → ℝ | IsLadderEpoch z j}.indicator (fun _ => (1 : ℝ)) x := by
    funext x
    unfold ladderEpochs
    rw [Finset.natCast_card_filter]
    refine Finset.sum_congr rfl fun j _ => ?_
    by_cases h : IsLadderEpoch x j
    · rw [if_pos h, Set.indicator_of_mem (show x ∈ {z | IsLadderEpoch z j} from h)]
    · rw [if_neg h, Set.indicator_of_notMem (show x ∉ {z | IsLadderEpoch z j} from h)]
  rw [this]
  exact Finset.measurable_sum _ fun j _ =>
    measurable_const.indicator (measurableSet_isLadderEpoch j)

theorem measurableSet_isLadderTime (r n : ℕ) : MeasurableSet {x : ℕ → ℝ | IsLadderTime x r n} := by
  have : {x : ℕ → ℝ | IsLadderTime x r n}
      = {x | IsLadderEpoch x n} ∩ (fun x : ℕ → ℝ => ((ladderEpochs x n).card : ℝ)) ⁻¹' {(r : ℝ)} := by
    ext x
    simp [IsLadderTime]
  rw [this]
  exact (measurableSet_isLadderEpoch n).inter
    (measurable_card_ladderEpochs n (measurableSet_singleton _))

theorem measurableSet_hasLadderEpoch : MeasurableSet {x : ℕ → ℝ | HasLadderEpoch x} := by
  have : {x : ℕ → ℝ | HasLadderEpoch x} = ⋃ j, ⋃ (_ : 1 ≤ j), {x | IsLadderEpoch x j} := by
    ext x
    simp [HasLadderEpoch]
  rw [this]
  exact MeasurableSet.iUnion fun j => MeasurableSet.iUnion fun _ => measurableSet_isLadderEpoch j

theorem measurableSet_pathSum_pos (n : ℕ) : MeasurableSet {x : ℕ → ℝ | 0 < pathSum x n} :=
  measurableSet_lt measurable_const (measurable_pathSum n)

theorem measurable_rotPath (n k : ℕ) : Measurable (rotPath n k) :=
  measurable_pi_lambda _ fun i => measurable_pi_apply (rotIdx n k i)

theorem measurable_ladderWeight (n : ℕ) : Measurable (ladderWeight n) := by
  unfold ladderWeight
  exact Finset.measurable_sum _ fun r _ =>
    (measurable_one.indicator (measurableSet_isLadderTime (r + 1) n)).const_mul _


/-! ### Exchangeability: an injective reindexing of an i.i.d. sequence keeps its law -/

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {X : ℕ → Ω → ℝ}

/-- The law of an i.i.d. sequence is the infinite product of its marginal. -/
theorem map_seq_eq_infinitePi {Y : ℕ → Ω → ℝ} (hindep : iIndepFun Y P)
    (hmeas : ∀ i, Measurable (Y i)) {ν : Measure ℝ} (hlaw : ∀ i, P.map (Y i) = ν) :
    P.map (fun ω i => Y i ω) = Measure.infinitePi (fun _ : ℕ => ν) := by
  have hblock : Measurable (fun ω i => Y i ω) := measurable_pi_lambda _ hmeas
  rw [(iIndepFun_iff_map_fun_eq_infinitePi_map₀ hblock.aemeasurable).mp hindep]
  congr 1
  funext i
  exact hlaw i

/-- **Exchangeability.** Reading an i.i.d. sequence through an injective index map (a rotation
of the first `n` coordinates, or a shift) does not change its joint law. -/
theorem map_reindex_eq (hindep : iIndepFun X P) (hmeas : ∀ i, Measurable (X i))
    (hident : ∀ i, IdentDistrib (X i) (X 0) P P) {σ : ℕ → ℕ} (hσ : Function.Injective σ) :
    P.map (fun ω i => X (σ i) ω) = P.map (fun ω i => X i ω) := by
  rw [map_seq_eq_infinitePi (hindep.precomp hσ) (fun i => hmeas (σ i))
      (fun i => (hident (σ i)).map_eq),
    map_seq_eq_infinitePi hindep hmeas (fun i => (hident i).map_eq)]

theorem measure_reindex_eq (hindep : iIndepFun X P) (hmeas : ∀ i, Measurable (X i))
    (hident : ∀ i, IdentDistrib (X i) (X 0) P P) {σ : ℕ → ℕ} (hσ : Function.Injective σ)
    {A : Set (ℕ → ℝ)} (hA : MeasurableSet A) :
    P {ω | (fun i => X (σ i) ω) ∈ A} = P {ω | (fun i => X i ω) ∈ A} := by
  have h := congrArg (fun μ : Measure (ℕ → ℝ) => μ A) (map_reindex_eq hindep hmeas hident hσ)
  rwa [Measure.map_apply (measurable_pi_lambda _ fun i => hmeas (σ i)) hA,
    Measure.map_apply (measurable_pi_lambda _ hmeas) hA] at h

/-! ### Step 3: `n⁻¹ P(S_n > 0) = ∑_{r ≥ 1} r⁻¹ P(τ_r = n)` -/

/-- **The cyclic lemma in expectation.** For `n ≥ 1`,
`P(S_n > 0) = n ∑_{r=1}^{n} r⁻¹ P(τ_r = n)`: take expectations in the weighted cyclic lemma, and
use that each of the `n` rotations has the law of the unrotated sequence. -/
theorem measure_pathSum_pos (hcyc : CyclicLemma) (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P)
    {n : ℕ} (hn : 0 < n) :
    P {ω | 0 < pathSum (fun i => X i ω) n}
      = n * ∑ r ∈ range n, ((r : ℝ≥0∞) + 1)⁻¹ * P {ω | IsLadderTime (fun i => X i ω) (r + 1) n} := by
  set μ := P.map (fun ω i => X i ω) with hμ
  have hblock : Measurable (fun ω i => X i ω) := measurable_pi_lambda _ hmeas
  have hrot : ∀ k, μ.map (rotPath n k) = μ := by
    intro k
    rw [hμ, Measure.map_map (measurable_rotPath n k) hblock]
    exact map_reindex_eq hindep hmeas hident (rotIdx_injective n k)
  have hE : ∀ {A : Set (ℕ → ℝ)}, MeasurableSet A → μ A = P {ω | (fun i => X i ω) ∈ A} :=
    fun hA => Measure.map_apply hblock hA
  have hW : ∫⁻ z, ladderWeight n z ∂μ
      = ∑ r ∈ range n, ((r : ℝ≥0∞) + 1)⁻¹ * μ {z | IsLadderTime z (r + 1) n} := by
    unfold ladderWeight
    rw [lintegral_finsetSum _ fun r _ =>
      (measurable_one.indicator (measurableSet_isLadderTime (r + 1) n)).const_mul _]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [lintegral_const_mul _ (measurable_one.indicator (measurableSet_isLadderTime (r + 1) n)),
      lintegral_indicator_one (measurableSet_isLadderTime (r + 1) n)]
  calc P {ω | 0 < pathSum (fun i => X i ω) n}
      = μ {z | 0 < pathSum z n} := (hE (measurableSet_pathSum_pos n)).symm
    _ = ∫⁻ z, {z : ℕ → ℝ | 0 < pathSum z n}.indicator 1 z ∂μ :=
        (lintegral_indicator_one (measurableSet_pathSum_pos n)).symm
    _ = ∫⁻ z, ∑ k ∈ range n, ladderWeight n (rotPath n k z) ∂μ := by
        congr 1
        funext z
        exact (sum_ladderWeight_rotPath hcyc hn z).symm
    _ = ∑ k ∈ range n, ∫⁻ z, ladderWeight n (rotPath n k z) ∂μ :=
        lintegral_finsetSum _ fun k _ => (measurable_ladderWeight n).comp (measurable_rotPath n k)
    _ = ∑ k ∈ range n, ∫⁻ z, ladderWeight n z ∂μ := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [← lintegral_map (measurable_ladderWeight n) (measurable_rotPath n k), hrot k]
    _ = n * ∑ r ∈ range n, ((r : ℝ≥0∞) + 1)⁻¹ * P {ω | IsLadderTime (fun i => X i ω) (r + 1) n} := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hW]
        congr 1
        refine Finset.sum_congr rfl fun r _ => ?_
        rw [hE (measurableSet_isLadderTime (r + 1) n)]
        rfl


/-! ### Step 4: the renewal at a ladder epoch -/

omit [MeasurableSpace Ω] [IsProbabilityMeasure P] in
/-- **`{τ_r = m}` is determined by the first `m` increments.** -/
theorem measurableSet_isLadderTime_first (X : ℕ → Ω → ℝ) (r m : ℕ) :
    MeasurableSet[⨆ i ∈ Set.Iio m, MeasurableSpace.comap (X i) inferInstance]
      {ω | IsLadderTime (fun i => X i ω) r m} := by
  have heq : {ω | IsLadderTime (fun i => X i ω) r m}
      = (fun ω (i : ℕ) => if i < m then X i ω else 0) ⁻¹' {x | IsLadderTime x r m} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_preimage]
    exact isLadderTime_congr fun l hl => by simp [hl]
  have htr : Measurable[⨆ i ∈ Set.Iio m, MeasurableSpace.comap (X i) inferInstance]
      (fun ω (i : ℕ) => if i < m then X i ω else 0) := by
    rw [measurable_iff_comap_le]
    simp only [MeasurableSpace.pi, MeasurableSpace.comap_iSup, MeasurableSpace.comap_comp]
    refine iSup_le fun i => ?_
    by_cases hi : i < m
    · simp only [Function.comp_def, hi, if_true]
      exact le_iSup₂_of_le (f := fun i (_ : i ∈ Set.Iio m) =>
        MeasurableSpace.comap (X i) inferInstance) i hi le_rfl
    · simp only [Function.comp_def, hi, if_false, MeasurableSpace.comap_const]
      exact bot_le
  rw [heq]
  exact htr (measurableSet_isLadderTime r m)

omit [MeasurableSpace Ω] [IsProbabilityMeasure P] in
/-- The event that the path restarted at `m` has a ladder epoch is determined by the increments
from `m` on. -/
theorem measurableSet_hasLadderEpoch_shift (X : ℕ → Ω → ℝ) (m : ℕ) :
    MeasurableSet[⨆ i ∈ Set.Ici m, MeasurableSpace.comap (X i) inferInstance]
      {ω | HasLadderEpoch (fun i => X (m + i) ω)} := by
  have hsh : Measurable[⨆ i ∈ Set.Ici m, MeasurableSpace.comap (X i) inferInstance]
      (fun ω (i : ℕ) => X (m + i) ω) := by
    refine @measurable_pi_lambda _ _ _ (_) _ _ fun i => ?_
    exact Measurable.mono (comap_measurable (X (m + i)))
      (le_iSup₂_of_le (f := fun i (_ : i ∈ Set.Ici m) =>
        MeasurableSpace.comap (X i) inferInstance) (m + i) (Nat.le_add_right m i) le_rfl) le_rfl
  exact hsh measurableSet_hasLadderEpoch

/-- **The renewal step, in measure.** The event `{τ_r = m}` and the event that the path restarted
at `m` has a ladder epoch are independent, and the second has the probability `q = P(τ_1 < ∞)`
of the unshifted path. -/
theorem measure_isLadderTime_inter_shift (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P) (r m : ℕ) :
    P ({ω | IsLadderTime (fun i => X i ω) r m} ∩ {ω | HasLadderEpoch (fun i => X (m + i) ω)})
      = P {ω | IsLadderTime (fun i => X i ω) r m} * P {ω | HasLadderEpoch (fun i => X i ω)} := by
  have hiI : iIndep (fun i => MeasurableSpace.comap (X i) inferInstance) P :=
    (iIndepFun_iff_iIndep _ _ _).1 hindep
  have hle : ∀ i, MeasurableSpace.comap (X i) (inferInstance : MeasurableSpace ℝ)
      ≤ ‹MeasurableSpace Ω› := fun i => measurable_iff_comap_le.1 (hmeas i)
  have hInd := indep_iSup_of_disjoint hle hiI (Set.Iio_disjoint_Ici (le_refl m))
  rw [(Indep_iff _ _ _).1 hInd _ _ (measurableSet_isLadderTime_first X r m)
    (measurableSet_hasLadderEpoch_shift X m)]
  congr 1
  exact measure_reindex_eq hindep hmeas hident (add_right_injective m) measurableSet_hasLadderEpoch

theorem measurableSet_isLadderTime_seq (hmeas : ∀ i, Measurable (X i)) (r n : ℕ) :
    MeasurableSet {ω | IsLadderTime (fun i => X i ω) r n} :=
  measurable_pi_lambda _ hmeas (measurableSet_isLadderTime r n)

omit [IsProbabilityMeasure P] in
/-- For each `r`, the events `{τ_r = n}` are disjoint in `n`, so their probabilities add up to
`P(τ_r < ∞)`. -/
theorem tsum_measure_isLadderTime (hmeas : ∀ i, Measurable (X i)) (r : ℕ) :
    ∑' n, P {ω | IsLadderTime (fun i => X i ω) r n}
      = P {ω | ∃ n, IsLadderTime (fun i => X i ω) r n} := by
  rw [show {ω | ∃ n, IsLadderTime (fun i => X i ω) r n}
      = ⋃ n, {ω | IsLadderTime (fun i => X i ω) r n} by ext ω; simp]
  refine (measure_iUnion (fun a b hab => ?_) fun n => measurableSet_isLadderTime_seq hmeas r n).symm
  exact Set.disjoint_left.2 fun ω ha hb => hab (isLadderTime_unique ha hb)

/-- **`P(τ_r < ∞) = q^r`**, by induction on `r` through the renewal step. -/
theorem measure_exists_isLadderTime (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P) (r : ℕ) :
    P {ω | ∃ n, IsLadderTime (fun i => X i ω) r n}
      = P {ω | HasLadderEpoch (fun i => X i ω)} ^ r := by
  induction r with
  | zero =>
    rw [pow_zero, show {ω | ∃ n, IsLadderTime (fun i => X i ω) 0 n} = Set.univ from
      Set.eq_univ_of_forall fun ω => ⟨0, isLadderTime_zero_zero _⟩, measure_univ]
  | succ r ih =>
    have hset : {ω | ∃ n, IsLadderTime (fun i => X i ω) (r + 1) n}
        = ⋃ m, ({ω | IsLadderTime (fun i => X i ω) r m}
            ∩ {ω | HasLadderEpoch (fun i => X (m + i) ω)}) := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
      exact exists_isLadderTime_succ_iff _ r
    have hmeasH : ∀ m, MeasurableSet {ω | HasLadderEpoch (fun i => X (m + i) ω)} := fun m =>
      measurable_pi_lambda _ (fun i => hmeas (m + i)) measurableSet_hasLadderEpoch
    rw [hset, measure_iUnion (fun a b hab => Set.disjoint_left.2 fun ω ha hb =>
        hab (isLadderTime_unique ha.1 hb.1))
      fun m => (measurableSet_isLadderTime_seq hmeas r m).inter (hmeasH m)]
    simp_rw [measure_isLadderTime_inter_shift hindep hmeas hident r]
    rw [ENNReal.tsum_mul_right, tsum_measure_isLadderTime hmeas r, ih, pow_succ]


/-! ### Step 5: Tonelli at `s = 1` -/

/-- **The key series identity**, in `ℝ≥0∞`:
`∑_{n≥1} n⁻¹ P(S_n > 0) = ∑_{r≥1} r⁻¹ q^r`. Every term is non-negative, so the double series
`∑_n ∑_r r⁻¹ P(τ_r = n)` may be summed in either order. -/
theorem tsum_measure_pathSum_pos_div (hcyc : CyclicLemma) (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P) :
    ∑' n : ℕ, P {ω | 0 < pathSum (fun i => X i ω) (n + 1)} / ((n : ℝ≥0∞) + 1)
      = ∑' r : ℕ, ((r : ℝ≥0∞) + 1)⁻¹ * P {ω | HasLadderEpoch (fun i => X i ω)} ^ (r + 1) := by
  set f : ℕ → ℕ → ℝ≥0∞ := fun r n =>
    ((r : ℝ≥0∞) + 1)⁻¹ * P {ω | IsLadderTime (fun i => X i ω) (r + 1) n} with hf
  have hterm : ∀ n : ℕ, P {ω | 0 < pathSum (fun i => X i ω) (n + 1)} / ((n : ℝ≥0∞) + 1)
      = ∑' r, f r (n + 1) := by
    intro n
    rw [measure_pathSum_pos hcyc hindep hmeas hident (Nat.succ_pos n), mul_comm,
      show (((n + 1 : ℕ) : ℝ≥0∞)) = (n : ℝ≥0∞) + 1 by push_cast; ring,
      ENNReal.mul_div_cancel_right (by positivity) (by simp)]
    refine (tsum_eq_sum fun r hr => ?_).symm
    have hempty : {ω | IsLadderTime (fun i => X i ω) (r + 1) (n + 1)} = ∅ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      intro h
      exact hr (Finset.mem_range.2 (lt_of_isLadderTime h))
    simp [hempty]
  have hzero : ∀ r, f r 0 = 0 := by
    intro r
    have hempty : {ω | IsLadderTime (fun i => X i ω) (r + 1) 0} = ∅ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      exact not_isLadderTime_succ_zero _ r
    simp [hf, hempty]
  simp_rw [hterm]
  rw [ENNReal.tsum_comm]
  refine tsum_congr fun r => ?_
  rw [← zero_add (∑' n, f r (n + 1)), ← hzero r, ← tsum_eq_zero_add' ENNReal.summable]
  simp only [hf]
  rw [ENNReal.tsum_mul_left, tsum_measure_isLadderTime hmeas,
    measure_exists_isLadderTime hindep hmeas hident]

/-- `∑_{r≥1} r⁻¹ = ∞`, read in `ℝ≥0∞`. -/
theorem tsum_inv_succ_eq_top : ∑' r : ℕ, ((r : ℝ≥0∞) + 1)⁻¹ = ⊤ := by
  by_contra h
  have hs := ENNReal.summable_toReal h
  have heq : (fun r : ℕ => (((r : ℝ≥0∞) + 1)⁻¹).toReal) = fun r : ℕ => 1 / ((r + 1 : ℕ) : ℝ) := by
    funext r
    rw [ENNReal.toReal_inv, one_div, show ((r : ℝ≥0∞) + 1) = ((r + 1 : ℕ) : ℝ≥0∞) by push_cast; ring,
      ENNReal.toReal_natCast]
  rw [heq] at hs
  exact Real.not_summable_one_div_natCast ((summable_nat_add_iff 1).1 hs)

/-- `∑_{r≥1} r⁻¹ q^r = -log(1 - q)` for `q < 1`, read in `ℝ≥0∞`. -/
theorem tsum_inv_succ_mul_pow (q : ℝ≥0∞) (hq : q < 1) :
    ∑' r : ℕ, ((r : ℝ≥0∞) + 1)⁻¹ * q ^ (r + 1) = ENNReal.ofReal (-Real.log (1 - q.toReal)) := by
  have hqtop : q ≠ ⊤ := ne_top_of_lt hq
  set p := q.toReal with hp
  have hp0 : 0 ≤ p := ENNReal.toReal_nonneg
  have hp1 : p < 1 := ENNReal.toReal_lt_of_lt_ofReal (by rwa [ENNReal.ofReal_one])
  have hsum := Real.hasSum_pow_div_log_of_abs_lt_one (show |p| < 1 by rwa [abs_of_nonneg hp0])
  have hterm : ∀ r : ℕ, ((r : ℝ≥0∞) + 1)⁻¹ * q ^ (r + 1)
      = ENNReal.ofReal (p ^ (r + 1) / ((r : ℝ) + 1)) := by
    intro r
    rw [ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_pow hp0, hp,
      ENNReal.ofReal_toReal hqtop, div_eq_mul_inv, mul_comm]
    congr 2
    rw [ENNReal.ofReal_add (Nat.cast_nonneg _) zero_le_one, ENNReal.ofReal_natCast,
      ENNReal.ofReal_one]
  simp_rw [hterm]
  rw [← ENNReal.ofReal_tsum_of_nonneg (fun r => by positivity) hsum.summable, hsum.tsum_eq]

/-! ### Step 6: from the series to the probability of never rising -/

omit [IsProbabilityMeasure P] in
/-- One term of the series, read in `ℝ≥0∞`. -/
theorem ofReal_measureReal_div (A : Set Ω) (hA : P A ≠ ⊤) (n : ℕ) :
    ENNReal.ofReal (P.real A / (n + 1)) = P A / ((n : ℝ≥0∞) + 1) := by
  rw [ENNReal.ofReal_div_of_pos (by positivity), ofReal_measureReal hA,
    ENNReal.ofReal_add (Nat.cast_nonneg _) zero_le_one, ENNReal.ofReal_natCast, ENNReal.ofReal_one]

/-- The series of the theorem, as stated there with real terms, summed in `ℝ≥0∞`. -/
theorem tsum_ofReal_measureReal_div (hcyc : CyclicLemma) (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P) :
    ∑' n : ℕ, ENNReal.ofReal (P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), X i ω} / (n + 1))
      = ∑' r : ℕ, ((r : ℝ≥0∞) + 1)⁻¹ * P {ω | HasLadderEpoch (fun i => X i ω)} ^ (r + 1) := by
  rw [← tsum_measure_pathSum_pos_div hcyc hindep hmeas hident]
  refine tsum_congr fun n => ?_
  rw [ofReal_measureReal_div _ (measure_ne_top P _)]
  rfl

/-- **Never rising is the complement of having a ladder epoch**:
`P(S_n ≤ 0 for all n) = 1 - q`. -/
theorem measureReal_forall_le (hmeas : ∀ i, Measurable (X i)) :
    P.real {ω | ∀ n, ∑ i ∈ Finset.range n, X i ω ≤ 0}
      = 1 - (P {ω | HasLadderEpoch (fun i => X i ω)}).toReal := by
  have hset : {ω | ∀ n, ∑ i ∈ Finset.range n, X i ω ≤ 0}
      = {ω | HasLadderEpoch (fun i => X i ω)}ᶜ := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, hasLadderEpoch_iff, not_exists, not_lt]
    rfl
  have hHm : MeasurableSet {ω | HasLadderEpoch (fun i => X i ω)} :=
    measurable_pi_lambda _ hmeas measurableSet_hasLadderEpoch
  rw [hset, probReal_compl_eq_one_sub hHm, measureReal_def]

end SpitzerLadder

open SpitzerLadder

/-! ### Spitzer's identity -/

/-- **Spitzer's ladder-epoch identity.** For a real random walk `S_n = X_0 + ⋯ + X_{n-1}` with
i.i.d. increments, `P(S_n ≤ 0 for all n) = exp(-∑_{n≥1} n⁻¹ P(S_n > 0))` whenever the series
converges. The deterministic cyclic lemma enters as the hypothesis `hcyc`. -/
theorem spitzer_ladder_identity {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {X : ℕ → Ω → ℝ} (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P)
    (hcyc : ∀ (n : ℕ) (x : ℕ → ℝ), 0 < n →
      (∑ r ∈ Finset.range n,
        if (∀ i < n, ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range n, x ((l + r) % n))
        then (((Finset.Icc 1 n).filter (fun j => ∀ i < j,
                ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range j, x ((l + r) % n))).card : ℝ)⁻¹
        else 0)
      = if 0 < ∑ l ∈ Finset.range n, x l then 1 else 0)
    (hsum : Summable (fun n : ℕ => P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), X i ω} / (n + 1))) :
    P.real {ω | ∀ n, ∑ i ∈ Finset.range n, X i ω ≤ 0}
      = Real.exp (-∑' n : ℕ, P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), X i ω} / (n + 1)) := by
  set q := P {ω | HasLadderEpoch (fun i => X i ω)} with hq
  have hT := tsum_ofReal_measureReal_div hcyc hindep hmeas hident
  rw [← ENNReal.ofReal_tsum_of_nonneg (fun n => by positivity) hsum] at hT
  have hq1 : q < 1 := by
    refine lt_of_le_of_ne prob_le_one fun h1 => ?_
    rw [← hq, h1] at hT
    simp only [one_pow, mul_one] at hT
    rw [tsum_inv_succ_eq_top] at hT
    exact ENNReal.ofReal_ne_top hT
  rw [← hq, tsum_inv_succ_mul_pow q hq1] at hT
  have hp0 : 0 ≤ q.toReal := ENNReal.toReal_nonneg
  have hp1 : q.toReal < 1 := ENNReal.toReal_lt_of_lt_ofReal (by rwa [ENNReal.ofReal_one])
  have hlog : Real.log (1 - q.toReal) ≤ 0 := Real.log_nonpos (by linarith) (by linarith)
  rw [ENNReal.ofReal_eq_ofReal_iff (tsum_nonneg fun n => by positivity) (by linarith)] at hT
  rw [hT, neg_neg, Real.exp_log (by linarith), measureReal_forall_le hmeas]

/-- **The converse direction.** If the walk stays at or below zero forever with positive
probability, the series `∑_{n≥1} n⁻¹ P(S_n > 0)` converges. -/
theorem summable_of_measureReal_forall_le_pos {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {X : ℕ → Ω → ℝ} (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P)
    (hcyc : ∀ (n : ℕ) (x : ℕ → ℝ), 0 < n →
      (∑ r ∈ Finset.range n,
        if (∀ i < n, ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range n, x ((l + r) % n))
        then (((Finset.Icc 1 n).filter (fun j => ∀ i < j,
                ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range j, x ((l + r) % n))).card : ℝ)⁻¹
        else 0)
      = if 0 < ∑ l ∈ Finset.range n, x l then 1 else 0)
    (hpos : 0 < P.real {ω | ∀ n, ∑ i ∈ Finset.range n, X i ω ≤ 0}) :
    Summable (fun n : ℕ => P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), X i ω} / (n + 1)) := by
  set q := P {ω | HasLadderEpoch (fun i => X i ω)} with hq
  rw [measureReal_forall_le hmeas, ← hq] at hpos
  have hqtop : q ≠ ⊤ := measure_ne_top P _
  have hq1 : q < 1 :=
    (ENNReal.toReal_lt_toReal hqtop ENNReal.one_ne_top).1 (by rw [ENNReal.toReal_one]; linarith)
  have hT := tsum_ofReal_measureReal_div hcyc hindep hmeas hident
  rw [← hq, tsum_inv_succ_mul_pow q hq1] at hT
  have hs := ENNReal.summable_toReal (hT ▸ ENNReal.ofReal_ne_top)
  refine hs.congr fun n => ?_
  exact ENNReal.toReal_ofReal (by positivity)

/-- Contrapositive of the converse, the identity's degenerate case: if the series diverges, the
walk rises above zero almost surely, `P(S_n ≤ 0 for all n) = 0 = exp(-∞)`. -/
theorem measureReal_forall_le_eq_zero_of_not_summable {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {X : ℕ → Ω → ℝ} (hindep : iIndepFun X P)
    (hmeas : ∀ i, Measurable (X i)) (hident : ∀ i, IdentDistrib (X i) (X 0) P P)
    (hcyc : ∀ (n : ℕ) (x : ℕ → ℝ), 0 < n →
      (∑ r ∈ Finset.range n,
        if (∀ i < n, ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range n, x ((l + r) % n))
        then (((Finset.Icc 1 n).filter (fun j => ∀ i < j,
                ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range j, x ((l + r) % n))).card : ℝ)⁻¹
        else 0)
      = if 0 < ∑ l ∈ Finset.range n, x l then 1 else 0)
    (hdiv : ¬ Summable (fun n : ℕ => P.real {ω | 0 < ∑ i ∈ Finset.range (n + 1), X i ω} / (n + 1))) :
    P.real {ω | ∀ n, ∑ i ∈ Finset.range n, X i ω ≤ 0} = 0 :=
  le_antisymm (not_lt.1 fun hpos =>
    hdiv (summable_of_measureReal_forall_le_pos hindep hmeas hident hcyc hpos)) measureReal_nonneg

end IcnnLift
