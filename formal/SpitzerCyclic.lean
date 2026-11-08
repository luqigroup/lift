import Mathlib

/-!
# The cyclic lemma behind Spitzer's identity

This file proves the deterministic, combinatorial step in the proof of Spitzer's identity
(Feller, *An Introduction to Probability Theory and Its Applications*, vol. II, ch. XII.6).

Fix `n > 0` and real numbers `x 0, …, x (n - 1)`. For `r < n`, the rotation by `r` is the
sequence `l ↦ x ((l + r) % n)`, with partial sums `s_0 = 0, s_1, …, s_n`. An index `j ∈ [1, n]`
is a strict ascending ladder epoch of the rotation if `s_j > s_i` for every `i < j`. The cyclic
lemma `cyclic_ladder_identity` states that

    ∑_{r < n, n is a ladder epoch of rotation r}  1 / #{ladder epochs of rotation r in [1, n]}
      = 1 if x 0 + ⋯ + x (n - 1) > 0, and 0 otherwise.

Ties are allowed; the ladder epochs are strict.

## How the proof goes

The helpers live in the namespace `IcnnLift.SpitzerCyclic`; the theorem itself is
`IcnnLift.cyclic_ladder_identity`.

* All rotations are read through one sequence, the partial sums `P m = cycPsum n x m` of the
  `n`-periodic extension of `x`. A rotation sum is a difference of these
  (`rot_sum_eq`), and they satisfy `P (m + n) = P m + P n` (`cycPsum_add_n`), where
  `P n` is the total. Ladder epochs then become records of `P` over a window:
  `IsRec n x a j` says that `P (a + j) > P (a + i)` for every `i < j`. It depends on `a`
  only through `a % n` (`isRec_mod`).
* If the total is not positive, `n` is never a ladder epoch, since it would have to beat
  `s_0 = 0`, and every term of the sum vanishes.
* Key lemma (`isRec_add_iff`): if `n` is a ladder epoch of the rotation by `r`, then for
  `k ∈ [1, n]`, `n` is a ladder epoch of the rotation by `r + k` exactly when `k` is a ladder
  epoch of the rotation by `r`.
* Existence (`exists_isRec`): if the total is positive, the first index in `[0, n)` at which
  `P` attains its maximum over `[0, n)` is a rotation for which `n` is a ladder epoch.
* Counting (`card_ladder_eq`): by the key lemma, for every rotation `r` in the set `G` of
  rotations having `n` as a ladder epoch, the number of ladder epochs of the rotation by `r`
  equals `#G`. The sum is therefore `#G · (#G)⁻¹ = 1`.
-/

namespace IcnnLift

namespace SpitzerCyclic

/-- Partial sums `P m = x (0 % n) + ⋯ + x ((m - 1) % n)` of the `n`-periodic extension of `x`. -/
noncomputable def cycPsum (n : ℕ) (x : ℕ → ℝ) (m : ℕ) : ℝ :=
  ∑ l ∈ Finset.range m, x (l % n)

/-- One period adds the total: `P (m + n) = P m + P n`. -/
lemma cycPsum_add_n (n : ℕ) (x : ℕ → ℝ) (m : ℕ) :
    cycPsum n x (m + n) = cycPsum n x m + cycPsum n x n := by
  unfold cycPsum
  rw [add_comm m n, Finset.sum_range_add, add_comm]
  congr 1
  exact Finset.sum_congr rfl (fun l _ => by rw [Nat.add_mod_left])

/-- A partial sum of the rotation by `r` is a difference of periodic partial sums. -/
lemma rot_sum_eq (n : ℕ) (x : ℕ → ℝ) (r i : ℕ) :
    ∑ l ∈ Finset.range i, x ((l + r) % n) = cycPsum n x (r + i) - cycPsum n x r := by
  unfold cycPsum
  rw [Finset.sum_range_add, add_sub_cancel_left]
  exact Finset.sum_congr rfl (fun l _ => by rw [add_comm])

/-- Over the first period, the periodic partial sum is the plain sum. -/
lemma cycPsum_self (n : ℕ) (x : ℕ → ℝ) :
    ∑ l ∈ Finset.range n, x l = cycPsum n x n := by
  unfold cycPsum
  exact Finset.sum_congr rfl (fun l hl => by rw [Nat.mod_eq_of_lt (Finset.mem_range.1 hl)])

/-- `a + j` is a strict record of the periodic partial sums over the window `[a, a + j)`:
`j` is a strict ascending ladder epoch of the rotation by `a`. -/
def IsRec (n : ℕ) (x : ℕ → ℝ) (a j : ℕ) : Prop :=
  ∀ i < j, cycPsum n x (a + i) < cycPsum n x (a + j)

/-- Shifting the window by one period does not change the record property. -/
lemma isRec_add_n (n : ℕ) (x : ℕ → ℝ) (a j : ℕ) : IsRec n x (a + n) j ↔ IsRec n x a j := by
  unfold IsRec
  have h : ∀ i, cycPsum n x (a + n + i) = cycPsum n x (a + i) + cycPsum n x n := by
    intro i
    rw [show a + n + i = (a + i) + n by omega, cycPsum_add_n]
  simp only [h, add_lt_add_iff_right]

/-- Shifting the window by any number of periods does not change the record property. -/
lemma isRec_add_mul (n : ℕ) (x : ℕ → ℝ) (b j : ℕ) :
    ∀ q, IsRec n x (b + n * q) j ↔ IsRec n x b j := by
  intro q
  induction q with
  | zero => simp
  | succ q ih => rw [Nat.mul_succ, ← add_assoc, isRec_add_n, ih]

/-- The record property depends on the start of the window only modulo `n`. -/
lemma isRec_mod (n : ℕ) (x : ℕ → ℝ) (a j : ℕ) : IsRec n x (a % n) j ↔ IsRec n x a j := by
  conv_rhs => rw [← Nat.mod_add_div a n]
  exact (isRec_add_mul n x (a % n) j (a / n)).symm

/-- Key lemma. If `n` is a ladder epoch of the rotation by `r`, then for `k ∈ [1, n]`, `n` is a
ladder epoch of the rotation by `r + k` iff `k` is a ladder epoch of the rotation by `r`. -/
lemma isRec_add_iff (n : ℕ) (x : ℕ → ℝ) (r k : ℕ) (hr : IsRec n x r n) (hk1 : 1 ≤ k)
    (hkn : k ≤ n) : IsRec n x (r + k) n ↔ IsRec n x r k := by
  unfold IsRec at *
  have per : ∀ m, cycPsum n x (m + n) = cycPsum n x m + cycPsum n x n := cycPsum_add_n n x
  constructor
  · intro h i hi
    have h1 := h (n - k + i) (by omega)
    rw [show r + k + (n - k + i) = (r + i) + n by omega, per (r + i), per (r + k)] at h1
    linarith
  · intro h i hi
    have h0 : cycPsum n x r < cycPsum n x (r + k) := by simpa using h 0 (by omega)
    rw [per (r + k)]
    by_cases hm : k + i ≤ n
    · have h2 : cycPsum n x (r + (k + i)) ≤ cycPsum n x (r + n) := by
        rcases lt_or_eq_of_le hm with hlt | heq
        · exact le_of_lt (hr _ hlt)
        · rw [heq]
      rw [per r] at h2
      rw [show r + k + i = r + (k + i) by omega]
      linarith
    · have h2 := h (k + i - n) (by omega)
      rw [show r + k + i = (r + (k + i - n)) + n by omega, per]
      linarith

/-- Existence. If the total is positive, some rotation `r < n` has `n` as a ladder epoch: the
first index in `[0, n)` at which the periodic partial sums attain their maximum over `[0, n)`. -/
lemma exists_isRec (n : ℕ) (x : ℕ → ℝ) (hn : 0 < n) (hT : 0 < cycPsum n x n) :
    ∃ r < n, IsRec n x r n := by
  classical
  have per : ∀ m, cycPsum n x (m + n) = cycPsum n x m + cycPsum n x n := cycPsum_add_n n x
  have hex : ∃ m, m < n ∧ ∀ m' < n, cycPsum n x m' ≤ cycPsum n x m := by
    obtain ⟨m, hm, hmax⟩ :=
      Finset.exists_max_image (Finset.range n) (cycPsum n x) ⟨0, Finset.mem_range.2 hn⟩
    exact ⟨m, Finset.mem_range.1 hm, fun m' hm' => hmax m' (Finset.mem_range.2 hm')⟩
  have hr : Nat.find hex < n ∧ ∀ m' < n, cycPsum n x m' ≤ cycPsum n x (Nat.find hex) :=
    Nat.find_spec hex
  have hfirst : ∀ m < Nat.find hex, cycPsum n x m < cycPsum n x (Nat.find hex) := by
    intro m hm
    have h1 := Nat.find_min hex hm
    push Not at h1
    obtain ⟨m'', hm''n, hlt⟩ := h1 (by omega)
    exact lt_of_lt_of_le hlt (hr.2 m'' hm''n)
  refine ⟨Nat.find hex, hr.1, fun i hi => ?_⟩
  rw [per]
  by_cases h : Nat.find hex + i < n
  · have := hr.2 _ h
    linarith
  · have h1 := hfirst (Nat.find hex + i - n) (by omega)
    have h2 : cycPsum n x (Nat.find hex + i) = cycPsum n x (Nat.find hex + i - n) + cycPsum n x n := by
      rw [← per]
      congr 1
      omega
    linarith

/-- Counting. If `n` is a ladder epoch of the rotation by `r`, the number of ladder epochs of that
rotation in `[1, n]` equals the number of rotations `a < n` having `n` as a ladder epoch. -/
lemma card_ladder_eq (n : ℕ) (x : ℕ → ℝ) (r : ℕ) (hr : IsRec n x r n)
    [DecidablePred (IsRec n x r)] [DecidablePred (fun a => IsRec n x a n)] :
    ((Finset.Icc 1 n).filter (IsRec n x r)).card
      = ((Finset.range n).filter (fun a => IsRec n x a n)).card := by
  have hper : Function.Periodic (fun a => IsRec n x a n) n :=
    fun a => propext (isRec_add_n n x a n)
  rw [← Nat.count_eq_card_filter_range, ← Nat.filter_Ico_card_eq_of_periodic (r + 1) n _ hper]
  refine Finset.card_nbij' (fun k => r + k) (fun a => a - r) ?_ ?_ ?_ ?_
  · intro k hk
    simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_Icc, Finset.mem_Ico] at hk ⊢
    exact ⟨⟨by omega, by omega⟩, (isRec_add_iff n x r k hr hk.1.1 hk.1.2).2 hk.2⟩
  · intro a ha
    simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_Icc, Finset.mem_Ico] at ha ⊢
    have h1 : r + (a - r) = a := by omega
    refine ⟨⟨by omega, by omega⟩, (isRec_add_iff n x r (a - r) hr (by omega) (by omega)).1 ?_⟩
    rw [h1]
    exact ha.2
  · intro k _
    simp
  · intro a ha
    simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_Ico] at ha
    simp only
    omega

open Classical in
/-- The cyclic identity in terms of the periodic partial sums. -/
lemma cyclic_identity_cycPsum (n : ℕ) (x : ℕ → ℝ) (hn : 0 < n) :
    (∑ r ∈ Finset.range n,
      if IsRec n x r n then (((Finset.Icc 1 n).filter (IsRec n x r)).card : ℝ)⁻¹ else 0)
    = if 0 < cycPsum n x n then 1 else 0 := by
  split_ifs with hT
  · set G := (Finset.range n).filter (fun a => IsRec n x a n) with hG
    have hne : G.Nonempty := by
      obtain ⟨r, hrn, hr⟩ := exists_isRec n x hn hT
      exact ⟨r, Finset.mem_filter.2 ⟨Finset.mem_range.2 hrn, hr⟩⟩
    calc (∑ r ∈ Finset.range n,
          if IsRec n x r n then (((Finset.Icc 1 n).filter (IsRec n x r)).card : ℝ)⁻¹ else 0)
        = ∑ r ∈ Finset.range n, (if IsRec n x r n then (G.card : ℝ)⁻¹ else 0) := by
          refine Finset.sum_congr rfl (fun r _ => ?_)
          split_ifs with h
          · rw [card_ladder_eq n x r h]
          · rfl
      _ = ∑ r ∈ G, (G.card : ℝ)⁻¹ := by rw [Finset.sum_filter]
      _ = 1 := by
          rw [Finset.sum_const, nsmul_eq_mul]
          exact mul_inv_cancel₀ (by exact_mod_cast hne.card_pos.ne')
  · refine Finset.sum_eq_zero (fun r _ => ?_)
    rw [if_neg]
    intro h
    have h1 := h 0 hn
    rw [add_zero, cycPsum_add_n] at h1
    exact hT (by linarith)

end SpitzerCyclic

open SpitzerCyclic in
/-- The cyclic lemma behind Spitzer's identity. Summing, over the rotations of `x 0, …, x (n - 1)`
for which `n` is a strict ascending ladder epoch, the reciprocal of the number of strict
ascending ladder epochs in `[1, n]` gives `1` if the total is positive and `0` otherwise. -/
theorem cyclic_ladder_identity (n : ℕ) (x : ℕ → ℝ) (hn : 0 < n) :
    (∑ r ∈ Finset.range n,
      if (∀ i < n, ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range n, x ((l + r) % n))
      then (((Finset.Icc 1 n).filter (fun j => ∀ i < j,
              ∑ l ∈ Finset.range i, x ((l + r) % n) < ∑ l ∈ Finset.range j, x ((l + r) % n))).card : ℝ)⁻¹
      else 0)
    = if 0 < ∑ l ∈ Finset.range n, x l then 1 else 0 := by
  classical
  simp only [rot_sum_eq n x, sub_lt_sub_iff_right]
  rw [cycPsum_self n x]
  convert cyclic_identity_cycPsum n x hn <;> exact Iff.rfl

end IcnnLift
