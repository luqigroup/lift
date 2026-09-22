import Mathlib

/-!
# (A4) is first-order Taylor of the gradient map, not an assumption about it

Assumption (A4) of the paper is carried in this development as a hypothesis. It is stated in
`Assumptions.lean` as the structure `ForwardKLChain`, and it is consumed in `CouplingSign.lean`
(`hchain` of `outerMoment_chain` and of `crossTerm_ge_leading_sub_of_remainder_bound`), in
`UpdateCovariance.lean` (the sign of the cross term), in `DiffusionOrdering.lean` and in
`PowerOfS.lean`. It has four clauses: the mean Hessian `H` is symmetric positive semidefinite;
the remainder constant `Cr` is non-negative; the chain

  `δg(ω) = H δθ̃(ω) + r(ω)`                                                              (A4-chain)

holds pointwise on the sample space; and the remainder is quadratic in the iterate fluctuation,

  `‖r(ω)‖ ≤ Cr ‖δθ̃(ω)‖²`.                                                              (A4-rem)

This file discharges (A4-chain) and (A4-rem) from a smoothness hypothesis on the loss, and
discharges the positive-semidefiniteness clause from a second-order necessary condition at an
interior minimum. It is a standalone module: it imports only `Mathlib`. Two pieces of vocabulary
it shares with the existing modules are therefore restated rather than imported, and under
different names so that this module can be imported alongside them: `l2Norm` is
`Assumptions.euclNorm`, which is in turn `CouplingSign.vecNorm`; and `sigmoid` is
`ShoulderAttenuation.logistic`, the derivative of the softplus positivity map that that module names
`softplus`. The definitions agree in each case; a later consolidation should keep one copy of
each and delete the others.

**What is actually assumed, and what is not.** (A4-chain) is not an assumption at all once `r` is
read as what the paper's own derivation makes it: the discrepancy between the gradient
fluctuation and its linearization. `gradFluct_eq_mulVec_add_remainder` proves the chain outright
for the remainder field `taylorRemainder`, which is *defined* as that discrepancy. All the
content of (A4) is therefore in (A4-rem) — the assertion that the discrepancy is `O(‖δθ̃‖²)` with
a uniform constant — and in the positive semidefiniteness of `H`.

**(A4-rem) is the mean-value inequality.** If the gradient map `G = ∇_θ E_X[L]` is differentiable
on a convex region `s` with derivative `DG` (the Hessian), and `DG` is Lipschitz at `θ̄` with
constant `K` on `s` — the standard `C²`-with-Lipschitz-Hessian hypothesis — then

  `‖G(θ̄ + v) − G(θ̄) − DG(θ̄) v‖ ≤ (K/2) ‖v‖²`

for every `v` with `θ̄ + v ∈ s`. This is `norm_taylorRemainder_le_of_lipschitz_fderiv`, proved from
the fencing form of the mean value inequality
(`image_norm_le_of_norm_deriv_right_le_deriv_boundary`) applied to `t ↦ G(θ̄ + t v) − G(θ̄) − t
DG(θ̄) v` against the boundary function `t ↦ (K/2) t² ‖v‖²`. The factor `1/2` is genuine and is not
obtained by weakening: the cruder route through
`Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le'` gives only `K ‖v‖²`. So (A4-rem) holds with
`Cr = K/2`, where `K` is the Hessian's Lipschitz modulus on the operative region. That reduces
(A4) to a smoothness statement about the loss, and `abs_sigmoid_taylorRemainder_le` below
verifies that statement for the softplus positivity map with the explicit constant `K = 1/4`.

**Positive semidefiniteness is a separate matter, and it does not follow from smoothness.**
`H = E_X[∇²L]` is positive semidefinite when `θ̄` is an interior local minimum of the expected
loss — the second-order necessary condition — and this is proved here in one dimension
(`secondDeriv_nonneg_of_isLocalMin`), for a general normed space by restriction to lines
(`quadForm_nonneg_of_isLocalMin`), and in matrix form (`posSemidef_of_isLocalMin`), the last using
mathlib's `second_derivative_symmetric` for the Hermitian clause. What is not proved, and is not
provable, is that the mean iterate of a stochastic-gradient trajectory *is* such a minimum: during
training `θ̄` is where the iterate happens to be, and at a saddle or on a plateau of the lifted
landscape `H` has negative directions. The honest reading of (A4)'s PSD clause is therefore that it
restricts the paper's claims to the neighbourhood of a minimum of the pullback landscape, or to a
loss convex on the operative region; it is not a consequence of the smoothness that discharges
(A4-rem). This file separates the two so that the distinction is visible.

## Results
* `norm_increment_sub_le_half_of_norm_deriv_sub_le` — the fencing form of the mean value
  inequality: if `‖h'(t) − c‖ ≤ K t` on `[0,1]` then `‖h(1) − h(0) − c‖ ≤ K/2`.
* `taylorRemainder`, `taylorRemainder_chain` — the remainder of the first-order expansion of a map,
  defined as the discrepancy, and the chain it satisfies by construction.
* `norm_taylorRemainder_le_of_lipschitz_fderiv_segment`,
  `norm_taylorRemainder_le_of_lipschitz_fderiv` — `‖r‖ ≤ (K/2)‖v‖²` from a Lipschitz derivative,
  along a segment and on a convex set.
* `abs_taylorRemainder_le_of_lipschitz_deriv` — the one-dimensional statement,
  `|f'(x+v) − f'(x) − f''(x) v| ≤ (K/2) v²`.
* `l2Norm`, `l2Norm_eq_norm_toLp` — the paper's Euclidean vector norm and its identification
  with the norm of `EuclideanSpace`.
* `gradFluct`, `gradRemainder`, `gradFluct_eq_mulVec_add_remainder`, `l2Norm_gradRemainder_le` —
  (A4-chain) and (A4-rem) in the paper's own matrix-and-vector vocabulary, the first by
  construction and the second from the Lipschitz-Hessian hypothesis with `Cr = K/2`.
* `forwardKL_chain_of_lipschitz_hessian` — the bridge: both dynamical clauses of (A4) at once,
  from differentiability of the gradient map on the operative region together with a Lipschitz
  Hessian at the mean iterate.
* `secondDeriv_nonneg_of_isLocalMin`, `quadForm_nonneg_of_isLocalMin`, `posSemidef_of_isLocalMin` —
  the second-order necessary condition, in one dimension, in a normed space, and as
  `Matrix.PosSemidef`.
* `sigmoid`, `sigmoidDeriv`, `hasDerivAt_logOneAddExp`, `hasDerivAt_sigmoid`,
  `hasDerivAt_sigmoidDeriv`, `abs_sigmoidDeriv_deriv_le` — the softplus derivative tower, computed.
* `abs_sigmoidDeriv_sub_le` — the softplus second derivative is `(1/4)`-Lipschitz on the whole
  line; `abs_sigmoid_taylorRemainder_le` — hence (A4-rem) holds for the softplus positivity map with
  `Cr = 1/8`; `sigmoidDeriv_pos` — and the scalar Hessian is strictly positive.
* `taylorRemainder_eq_zero_of_affine` — for an affine gradient map, that is a quadratic loss, the
  remainder vanishes identically and (A4) holds with `Cr = 0`.
* `forwardKLChain_softplus_witness` — a complete witness: on a one-dimensional lift with softplus
  loss all four clauses of (A4) hold simultaneously, with `Cr = 1/8`, for an arbitrary iterate
  fluctuation field. (A4) is therefore not vacuous.

## Honesty
No `sorry`/`admit`/`native_decide`/`axiom`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.

The hypotheses this file carries, and what would discharge each. The Lipschitz-Hessian bound
`hLip` is a regularity property of the particular network, positivity map and loss; it is verified here
for softplus and for quadratics, and for a deep network it would follow from a bound on the third
derivative of the composed loss over a bounded parameter region, which is a computation about an
architecture and not a theorem about `EuclideanSpace`. The membership hypothesis `hmem`, that the
iterate fluctuation stays inside the region on which the Hessian is Lipschitz, is the paper's "on
the operative region" and is a claim about where training goes. The local-minimality hypothesis
`hmin` of the positive-semidefiniteness results is discussed above and is not discharged: nothing
in this development shows that the mean iterate is a local minimum. The presentation hypothesis
`hH`, that `H` is the matrix of the second derivative in the standard basis, is a definition of
`H` and carries no content, but it is a hypothesis rather than a construction and is listed here
for completeness.

Two limits of the bridge should be stated plainly. First, this file linearizes the gradient of a
*fixed* function, so it delivers (A4) for the expected loss `E_X[L]`; the paper's `H` is
`E_X[∇²L]`, which is the same object only because differentiation commutes with the batch
expectation — that interchange is `UpdateCovariance.lean`'s business and is not re-proved here.
The per-batch Hessian fluctuation `H(X) − H` is absorbed into the remainder there, and the
absorbed remainder is not covered by the constant `K/2` proved here. Second, the bridge is stated
for the gradient map of a loss on a Euclidean parameter space, with the Hessian presented as a
matrix through the hypothesis `hH`; it does not construct that matrix from the loss, because
`Matrix` and `EuclideanSpace` are different carriers in mathlib and the identification is
bookkeeping rather than mathematics. Third, the softplus verification is of the positivity-map
nonlinearity alone, not of a network loss composed with it: it establishes that the class of
functions satisfying the Lipschitz-Hessian hypothesis contains the paper's positivity map and is not
empty, which is what a satisfiability check is for, and it does not establish the hypothesis for
any particular trained model.
-/

open Set Filter Matrix
open scoped Topology

namespace IcnnLift

/-! ## The mean value inequality in fencing form

The one input from mathlib is `image_norm_le_of_norm_deriv_right_le_deriv_boundary`: a function
whose derivative is dominated by the derivative of a scalar boundary function is itself dominated
by that boundary function. Taking the boundary function to be `t ↦ (K/2) t²` converts a linearly
growing bound on the derivative into a quadratic bound on the increment, which is exactly the
factor `1/2` that a Taylor remainder carries and that a uniform mean-value bound loses. -/

/-- **The fencing form of the mean value inequality.** If `h` is continuous on `[0,1]`, has right
derivative `h'` there, and `h'` differs from a fixed vector `c` by at most `K t` at time `t`, then
the increment `h(1) − h(0)` differs from `c` by at most `K/2`.

Read with `h t = G(x + t v)` and `c = DG(x) v` this is the first-order Taylor bound with its
sharp constant: the bound on the derivative is linear in `t`, so integrating it costs a factor
`1/2` that the uniform mean-value inequality does not see. -/
theorem norm_increment_sub_le_half_of_norm_deriv_sub_le {F : Type*} [NormedAddCommGroup F]
    [NormedSpace ℝ F] {h h' : ℝ → F} {c : F} {K : ℝ}
    (hcont : ContinuousOn h (Icc 0 1))
    (hderiv : ∀ t ∈ Ico (0 : ℝ) 1, HasDerivWithinAt h (h' t) (Ici t) t)
    (hbound : ∀ t ∈ Ico (0 : ℝ) 1, ‖h' t - c‖ ≤ K * t) :
    ‖h 1 - h 0 - c‖ ≤ K / 2 := by
  set φ : ℝ → F := fun t => h t - h 0 - t • c with hφ
  have hsmul : ContinuousOn (fun t : ℝ => t • c) (Icc 0 1) :=
    (continuous_id.smul continuous_const).continuousOn
  have hφcont : ContinuousOn φ (Icc 0 1) := (hcont.sub continuousOn_const).sub hsmul
  have hφderiv : ∀ t ∈ Ico (0 : ℝ) 1, HasDerivWithinAt φ (h' t - c) (Ici t) t := by
    intro t ht
    have h1 : HasDerivWithinAt (fun t => h t - h 0) (h' t) (Ici t) t :=
      (hderiv t ht).sub_const (h 0)
    have h2 : HasDerivWithinAt (fun t : ℝ => t • c) c (Ici t) t := by
      simpa using ((hasDerivAt_id t).smul_const c).hasDerivWithinAt
    exact h1.sub h2
  have hB : ∀ t : ℝ, HasDerivAt (fun t : ℝ => K / 2 * t ^ 2) (K * t) t := by
    intro t
    have h1 : HasDerivAt (fun y : ℝ => y ^ 2) (2 * t) t := by
      simpa using hasDerivAt_pow 2 t
    have h2 := h1.const_mul (K / 2)
    have hrw : K / 2 * (2 * t) = K * t := by ring
    rwa [hrw] at h2
  have ha : ‖φ 0‖ ≤ K / 2 * (0 : ℝ) ^ 2 := by simp [hφ]
  have hmain := image_norm_le_of_norm_deriv_right_le_deriv_boundary hφcont hφderiv ha hB hbound
    (right_mem_Icc.2 zero_le_one)
  simpa [hφ] using hmain

/-! ## The first-order Taylor remainder of a map, and its quadratic bound -/

/-- **(A4)'s remainder field, defined rather than assumed.** For a map `G` — the gradient of the
loss, in the paper's application — and a candidate linearization `A` at `x`, the remainder is the
discrepancy `G(x + v) − G(x) − A v`. Nothing is asserted about its size by this definition; the
size is the content of `norm_taylorRemainder_le_of_lipschitz_fderiv`. -/
noncomputable def taylorRemainder {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] (G : E → F) (A : E →L[ℝ] F) (x v : E) : F :=
  G (x + v) - G x - A v

/-- **(A4)'s chain holds by construction.** With the remainder read as the discrepancy, the chain
`δg = H δθ̃ + r` of assumption (A4) is an identity, not a hypothesis. What (A4) genuinely asserts
is the bound on `r`, and that is the subject of the next theorem. -/
theorem taylorRemainder_chain {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] (G : E → F) (A : E →L[ℝ] F) (x v : E) :
    G (x + v) - G x = A v + taylorRemainder G A x v := by
  rw [taylorRemainder]
  abel

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- **The Taylor remainder bound, along a segment.** If `G` is differentiable at every point of
the segment `[x, x + v]` with derivative `DG`, and `DG` deviates from `DG x` by at most `K t ‖v‖`
at the point `x + t v`, then the first-order remainder at `v` is at most `(K/2) ‖v‖²`.

This is (A4-rem) with `Cr = K/2`. The hypotheses are exactly `C¹` of the gradient map — that is,
`C²` of the loss — together with a Lipschitz modulus for the Hessian along the segment. -/
theorem norm_taylorRemainder_le_of_lipschitz_fderiv_segment
    {G : E → F} {DG : E → E →L[ℝ] F} {x v : E} {K : ℝ}
    (hG : ∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt G (DG (x + t • v)) (x + t • v))
    (hK : ∀ t ∈ Icc (0 : ℝ) 1, ‖DG (x + t • v) - DG x‖ ≤ K * t * ‖v‖) :
    ‖taylorRemainder G (DG x) x v‖ ≤ K / 2 * ‖v‖ ^ 2 := by
  set h : ℝ → F := fun t => G (x + t • v) with hh
  set h' : ℝ → F := fun t => DG (x + t • v) v with hh'
  have hline : ∀ t : ℝ, HasDerivAt (fun t : ℝ => x + t • v) v t := by
    intro t
    simpa using ((hasDerivAt_id t).smul_const v).const_add x
  have hderiv : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt h (h' t) t := fun t ht =>
    (hG t ht).comp_hasDerivAt t (hline t)
  have hcont : ContinuousOn h (Icc 0 1) := fun t ht =>
    (hderiv t ht).continuousAt.continuousWithinAt
  have hbound : ∀ t ∈ Ico (0 : ℝ) 1, ‖h' t - DG x v‖ ≤ K * ‖v‖ ^ 2 * t := by
    intro t ht
    have hmem : t ∈ Icc (0 : ℝ) 1 := Ico_subset_Icc_self ht
    have hsub : h' t - DG x v = (DG (x + t • v) - DG x) v := by simp [hh']
    have h1 : ‖(DG (x + t • v) - DG x) v‖ ≤ ‖DG (x + t • v) - DG x‖ * ‖v‖ :=
      ContinuousLinearMap.le_opNorm _ _
    have h2 : ‖DG (x + t • v) - DG x‖ * ‖v‖ ≤ K * t * ‖v‖ * ‖v‖ :=
      mul_le_mul_of_nonneg_right (hK t hmem) (norm_nonneg v)
    rw [hsub]
    calc ‖(DG (x + t • v) - DG x) v‖ ≤ K * t * ‖v‖ * ‖v‖ := le_trans h1 h2
      _ = K * ‖v‖ ^ 2 * t := by ring
  have hmain := norm_increment_sub_le_half_of_norm_deriv_sub_le hcont
    (fun t ht => (hderiv t (Ico_subset_Icc_self ht)).hasDerivWithinAt) hbound
  have h1 : h 1 = G (x + v) := by simp [hh]
  have h0 : h 0 = G x := by simp [hh]
  rw [h1, h0] at hmain
  calc ‖taylorRemainder G (DG x) x v‖ ≤ K * ‖v‖ ^ 2 / 2 := hmain
    _ = K / 2 * ‖v‖ ^ 2 := by ring

/-- **The Taylor remainder bound on a convex region.** The form in which the paper's hypothesis is
naturally stated: `G` is differentiable on a convex region `s` and its derivative is Lipschitz at
`x` with modulus `K` on `s`. Then for every `v` whose endpoint `x + v` lies in `s`, the remainder
is at most `(K/2) ‖v‖²`.

This is the theorem that reduces assumption (A4) to a smoothness assumption on the loss: the
region `s` is the paper's operative region, `x` is the mean iterate `θ̄`, `v` is the iterate
fluctuation `δθ̃`, and `K` is the Lipschitz modulus of the Hessian there. -/
theorem norm_taylorRemainder_le_of_lipschitz_fderiv
    {G : E → F} {DG : E → E →L[ℝ] F} {s : Set E} {x v : E} {K : ℝ}
    (hs : Convex ℝ s) (hx : x ∈ s) (hxv : x + v ∈ s)
    (hG : ∀ y ∈ s, HasFDerivAt G (DG y) y)
    (hK : ∀ y ∈ s, ‖DG y - DG x‖ ≤ K * ‖y - x‖) :
    ‖taylorRemainder G (DG x) x v‖ ≤ K / 2 * ‖v‖ ^ 2 := by
  have hmem : ∀ t ∈ Icc (0 : ℝ) 1, x + t • v ∈ s := by
    intro t ht
    have h := hs hx hxv (by linarith [ht.2] : (0 : ℝ) ≤ 1 - t) ht.1 (by ring)
    have e : (1 - t) • x + t • (x + v) = x + t • v := by module
    rwa [e] at h
  refine norm_taylorRemainder_le_of_lipschitz_fderiv_segment
    (fun t ht => hG _ (hmem t ht)) (fun t ht => ?_)
  have hnorm : ‖x + t • v - x‖ = t * ‖v‖ := by
    rw [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1]
  have := hK _ (hmem t ht)
  rw [hnorm] at this
  calc ‖DG (x + t • v) - DG x‖ ≤ K * (t * ‖v‖) := this
    _ = K * t * ‖v‖ := by ring

/-- Reading the norm of a difference of one-dimensional derivatives, needed to specialise the
Fréchet statement to the `deriv` statement. -/
theorem norm_smulRight_sub_smulRight (a b : ℝ) :
    ‖ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) a
      - ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) b‖ = |a - b| := by
  have e : ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) a
      - ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) b
      = ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (a - b) := by
    ext; simp [mul_sub]
  rw [e, ContinuousLinearMap.norm_smulRight_apply]
  simp

/-- **The one-dimensional statement.** If `f'` has derivative `f''` along the segment from `x` to
`x + v` and `f''` is Lipschitz at `x` with constant `K` there, then

`|f'(x + v) − f'(x) − f''(x) v| ≤ (K/2) v²`.

This is (A4-rem) in the scalar case, and it is the statement the positivity-map computation of the next
section instantiates. -/
theorem abs_taylorRemainder_le_of_lipschitz_deriv {f' f'' : ℝ → ℝ} {x v K : ℝ}
    (hf : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt f' (f'' (x + t * v)) (x + t * v))
    (hK : ∀ t ∈ Icc (0 : ℝ) 1, |f'' (x + t * v) - f'' x| ≤ K * t * |v|) :
    |f' (x + v) - f' x - f'' x * v| ≤ K / 2 * v ^ 2 := by
  have hG : ∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt f'
      (ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (f'' (x + t • v))) (x + t • v) := by
    intro t ht
    simp only [smul_eq_mul]
    exact (hf t ht).hasFDerivAt
  have hK' : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (f'' (x + t • v))
        - ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (f'' x)‖ ≤ K * t * ‖v‖ := by
    intro t ht
    rw [norm_smulRight_sub_smulRight, Real.norm_eq_abs, smul_eq_mul]
    exact hK t ht
  have hmain := norm_taylorRemainder_le_of_lipschitz_fderiv_segment (G := f')
    (DG := fun y => ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (f'' y))
    (x := x) (v := v) (K := K) hG hK'
  simpa [taylorRemainder, Real.norm_eq_abs, mul_comm] using hmain

/-! ## (A4) in the paper's own vocabulary

The assumption modules state (A4) for coordinate vectors `Ω → n → ℝ` with the Euclidean norm
(`Assumptions.euclNorm`, restated here as `l2Norm`), and for a mean Hessian presented as a
`Matrix n n ℝ` acting by `Matrix.mulVec`. The
analysis above lives on `EuclideanSpace`, whose norm is the same Euclidean norm but whose carrier
is `WithLp 2`, a distinct type in this version of mathlib. This section carries the two statements
across, so that the bridge lands in the vocabulary the mechanism modules actually consume. The
Hessian's presentation as a matrix is a hypothesis (`hH`), not a construction: it says that `H` is
the matrix of the derivative `DG θ̄` in the standard basis. -/

/-- The Euclidean norm of a coordinate vector, exactly as `Assumptions.euclNorm` and
`CouplingSign.vecNorm` define it. It is restated, under a third name, rather than imported,
because this module is standalone and must remain importable alongside those two; the three
definitions agree. -/
noncomputable def l2Norm {n : Type*} [Fintype n] (v : n → ℝ) : ℝ := Real.sqrt (v ⬝ᵥ v)

/-- The paper's Euclidean vector norm is the norm of `EuclideanSpace`. -/
theorem l2Norm_eq_norm_toLp {n : Type*} [Fintype n] (v : n → ℝ) :
    l2Norm v = ‖(WithLp.toLp 2 v : EuclideanSpace ℝ n)‖ := by
  rw [EuclideanSpace.norm_eq, l2Norm, dotProduct]
  simp [Real.norm_eq_abs, pow_two]

/-- The same identification, read in the other direction. -/
theorem l2Norm_ofLp {n : Type*} [Fintype n] (z : EuclideanSpace ℝ n) :
    l2Norm (WithLp.ofLp z) = ‖z‖ := by
  rw [l2Norm_eq_norm_toLp]

/-- **The forward-KL gradient fluctuation of (A4)**, in coordinates:
`δg(ω) = ∇L(θ̄ + δθ̃(ω)) − ∇L(θ̄)`, where `G` is the gradient map of the expected loss. -/
noncomputable def gradFluct {d : ℕ} {Ω : Type*}
    (G : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (thb : EuclideanSpace ℝ (Fin d)) (dth : Ω → Fin d → ℝ) (ω : Ω) : Fin d → ℝ :=
  WithLp.ofLp (G (thb + WithLp.toLp 2 (dth ω)) - G thb)

/-- **(A4)'s remainder field `r`**, in coordinates, defined as the discrepancy between the
gradient fluctuation and its linearization by `A`. -/
noncomputable def gradRemainder {d : ℕ} {Ω : Type*}
    (G : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (A : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (thb : EuclideanSpace ℝ (Fin d)) (dth : Ω → Fin d → ℝ) (ω : Ω) : Fin d → ℝ :=
  WithLp.ofLp (taylorRemainder G A thb (WithLp.toLp 2 (dth ω)))

/-- **(A4-chain), proved.** If `H` is the matrix of the derivative `A = DG(θ̄)` in the standard
basis, then the chain `δg(ω) = H δθ̃(ω) + r(ω)` holds for every `ω`, with no hypothesis on the
loss whatsoever: it is the definition of `r` rearranged. The clause `chain` of
`Assumptions.ForwardKLChain` therefore carries no analytic content, and the whole content of (A4)
is the remainder bound proved next together with the positive semidefiniteness of `H`. -/
theorem gradFluct_eq_mulVec_add_remainder {d : ℕ} {Ω : Type*}
    (G : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (A : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (thb : EuclideanSpace ℝ (Fin d)) (H : Matrix (Fin d) (Fin d) ℝ) (dth : Ω → Fin d → ℝ)
    (hH : ∀ w, A w = WithLp.toLp 2 (H *ᵥ WithLp.ofLp w)) (ω : Ω) :
    gradFluct G thb dth ω = H *ᵥ dth ω + gradRemainder G A thb dth ω := by
  have hAv : WithLp.ofLp (A (WithLp.toLp 2 (dth ω))) = H *ᵥ dth ω := by
    rw [hH]
  rw [gradFluct, gradRemainder, taylorRemainder, ← hAv]
  simp only [WithLp.ofLp_sub]
  abel

/-- **(A4-rem), proved.** With the gradient map differentiable on a convex operative region `s`
and its derivative Lipschitz at `θ̄` with modulus `K` there, the remainder obeys
`‖r(ω)‖ ≤ (K/2) ‖δθ̃(ω)‖²` for every `ω` whose iterate fluctuation stays in `s`. So (A4)'s
remainder constant may be taken to be `Cr = K/2`. -/
theorem l2Norm_gradRemainder_le {d : ℕ} {Ω : Type*}
    {G : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d)}
    {DG : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)}
    {thb : EuclideanSpace ℝ (Fin d)} {s : Set (EuclideanSpace ℝ (Fin d))}
    {dth : Ω → Fin d → ℝ} {K : ℝ}
    (hs : Convex ℝ s) (hthb : thb ∈ s)
    (hG : ∀ y ∈ s, HasFDerivAt G (DG y) y)
    (hK : ∀ y ∈ s, ‖DG y - DG thb‖ ≤ K * ‖y - thb‖)
    (hmem : ∀ ω, thb + WithLp.toLp 2 (dth ω) ∈ s) (ω : Ω) :
    l2Norm (gradRemainder G (DG thb) thb dth ω) ≤ K / 2 * l2Norm (dth ω) ^ 2 := by
  rw [gradRemainder, l2Norm_ofLp, l2Norm_eq_norm_toLp]
  exact norm_taylorRemainder_le_of_lipschitz_fderiv hs hthb (hmem ω) hG hK

/-- **The bridge.** Assumption (A4)'s two dynamical clauses — the forward-KL chain and its
quadratic remainder bound — follow from one smoothness hypothesis: that the gradient map of the
expected loss is differentiable on the operative region with a Hessian Lipschitz at the mean
iterate. The remainder constant is `Cr = K/2` where `K` is the Hessian's Lipschitz modulus.

What remains of (A4) after this theorem is the positive semidefiniteness of `H`, which is the
subject of the next section and is not a smoothness property. -/
theorem forwardKL_chain_of_lipschitz_hessian {d : ℕ} {Ω : Type*}
    {G : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d)}
    {DG : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)}
    {thb : EuclideanSpace ℝ (Fin d)} {s : Set (EuclideanSpace ℝ (Fin d))}
    {dth : Ω → Fin d → ℝ} {H : Matrix (Fin d) (Fin d) ℝ} {K : ℝ}
    (hs : Convex ℝ s) (hthb : thb ∈ s)
    (hG : ∀ y ∈ s, HasFDerivAt G (DG y) y)
    (hK : ∀ y ∈ s, ‖DG y - DG thb‖ ≤ K * ‖y - thb‖)
    (hH : ∀ w, DG thb w = WithLp.toLp 2 (H *ᵥ WithLp.ofLp w))
    (hmem : ∀ ω, thb + WithLp.toLp 2 (dth ω) ∈ s) :
    (∀ ω, gradFluct G thb dth ω = H *ᵥ dth ω + gradRemainder G (DG thb) thb dth ω)
      ∧ (∀ ω, l2Norm (gradRemainder G (DG thb) thb dth ω)
          ≤ K / 2 * l2Norm (dth ω) ^ 2) :=
  ⟨fun ω => gradFluct_eq_mulVec_add_remainder G (DG thb) thb H dth hH ω,
   fun ω => l2Norm_gradRemainder_le hs hthb hG hK hmem ω⟩

/-! ## The positive-semidefiniteness clause of (A4)

Smoothness gives the chain and the remainder bound; it says nothing about the sign of the mean
Hessian. What does give the sign is minimality: at an interior local minimum of a twice
differentiable function the second derivative is positive semidefinite. That is the second-order
necessary condition, and it is proved here in three forms, the last being `Matrix.PosSemidef` as
`Assumptions.ForwardKLChain.posSemidef` states it.

mathlib has the first-order condition (`IsLocalMin.hasDerivAt_eq_zero`,
`IsLocalMin.fderiv_eq_zero`) but not the second-order one, so it is proved from the mean value
theorem: if the second derivative at the minimum were negative, the first derivative would be
strictly negative just to the right of it, and the mean value theorem on a short interval to the
right would produce a strictly smaller value of the function, contradicting minimality.

The hypothesis of local minimality is not discharged anywhere in this development, and it should
not be read as innocuous. Along a stochastic-gradient trajectory the mean iterate is wherever
training has reached; at a saddle of the lifted landscape the mean Hessian has negative
directions and (A4)'s PSD clause fails. -/

/-- **The second-order necessary condition in one dimension.** If `f` is differentiable with
derivative `f'`, `f'` is differentiable at a local minimum `x` with derivative `c`, then `0 ≤ c`.

The proof is by contradiction through the mean value theorem, as described above; mathlib does not
carry this statement. -/
theorem secondDeriv_nonneg_of_isLocalMin {f f' : ℝ → ℝ} {x c : ℝ}
    (hf : ∀ y, HasDerivAt f (f' y) y) (hf' : HasDerivAt f' c x) (hmin : IsLocalMin f x) :
    0 ≤ c := by
  by_contra hcon
  rw [not_le] at hcon
  have hfx : f' x = 0 := hmin.hasDerivAt_eq_zero (hf x)
  have hslope : Tendsto (slope f' x) (𝓝[≠] x) (𝓝 c) := hasDerivAt_iff_tendsto_slope.mp hf'
  have hmono : (𝓝[>] x) ≤ (𝓝[≠] x) :=
    nhdsWithin_mono _ fun y hy => Set.mem_compl_singleton_iff.mpr (Set.mem_Ioi.mp hy).ne'
  have hev1 : ∀ᶠ y in 𝓝[>] x, slope f' x y < 0 :=
    Filter.Eventually.filter_mono hmono (hslope.eventually (gt_mem_nhds hcon))
  have hev2 : ∀ᶠ y in 𝓝[>] x, f x ≤ f y := nhdsWithin_le_nhds hmin
  obtain ⟨u, hu, hsub⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp (hev1.and hev2)
  have hxu : x < u := Set.mem_Ioi.mp hu
  set b : ℝ := (x + u) / 2 with hb
  have hxb : x < b := by rw [hb]; linarith
  have hbu : b < u := by rw [hb]; linarith
  have hbmem : b ∈ Ioo x u := ⟨hxb, hbu⟩
  obtain ⟨xi, hxi, hxieq⟩ := exists_hasDerivAt_eq_slope f f' hxb
    (fun y _ => (hf y).continuousAt.continuousWithinAt) (fun y _ => hf y)
  have hxi' : xi ∈ Ioo x u := ⟨hxi.1, lt_trans hxi.2 hbu⟩
  have hneg : slope f' x xi < 0 := (hsub hxi').1
  have hfxi : f' xi < 0 := by
    have hs : slope f' x xi = (f' xi - f' x) / (xi - x) := slope_def_field f' x xi
    rw [hs, hfx, sub_zero] at hneg
    have hpos : 0 < xi - x := by linarith [hxi.1]
    by_contra hnn
    rw [not_lt] at hnn
    have : 0 ≤ f' xi / (xi - x) := div_nonneg hnn (le_of_lt hpos)
    linarith
  have hfb : f x ≤ f b := (hsub hbmem).2
  have hquot : 0 ≤ (f b - f x) / (b - x) := div_nonneg (by linarith) (by linarith)
  rw [← hxieq] at hquot
  linarith

/-- **The second-order necessary condition in a normed space.** At an interior local minimum the
second derivative is a positive semidefinite quadratic form. The proof restricts to the line
`t ↦ f(x + t v)` and applies the one-dimensional statement; the second derivative of the
restriction at `0` is `A v v` by the chain rule. -/
theorem quadForm_nonneg_of_isLocalMin {f : E → ℝ} {f' : E → E →L[ℝ] ℝ} {A : E →L[ℝ] E →L[ℝ] ℝ}
    {x : E} (hf : ∀ y, HasFDerivAt f (f' y) y) (hA : HasFDerivAt f' A x)
    (hmin : IsLocalMin f x) (v : E) : 0 ≤ A v v := by
  have hline : ∀ t : ℝ, HasDerivAt (fun t : ℝ => x + t • v) v t := by
    intro t
    simpa using ((hasDerivAt_id t).smul_const v).const_add x
  have hg0 : (fun t : ℝ => x + t • v) 0 = x := by simp
  have hphi : ∀ t : ℝ, HasDerivAt (fun t : ℝ => f (x + t • v)) (f' (x + t • v) v) t := fun t =>
    (hf (x + t • v)).comp_hasDerivAt t (hline t)
  have hAline : HasDerivAt (fun t : ℝ => f' (x + t • v)) (A v) 0 := by
    have h0 : HasFDerivAt f' A ((fun t : ℝ => x + t • v) 0) := by rw [hg0]; exact hA
    exact h0.comp_hasDerivAt 0 (hline 0)
  have hphi' : HasDerivAt (fun t : ℝ => f' (x + t • v) v) (A v v) 0 := by
    have h := hAline.clm_apply (hasDerivAt_const (0 : ℝ) v)
    simpa using h
  have hminphi : IsLocalMin (fun t : ℝ => f (x + t • v)) 0 := by
    have h1 : IsLocalMin (f ∘ fun t : ℝ => x + t • v) 0 := by
      refine IsLocalMin.comp_continuous ?_ (hline 0).continuousAt
      simpa using hmin
    exact h1
  exact secondDeriv_nonneg_of_isLocalMin hphi hphi' hminphi

/-- **(A4)'s positive-semidefiniteness clause, proved at an interior minimum.** If the expected
loss on a Euclidean parameter space is differentiable everywhere, twice differentiable at `θ̄`
with second derivative represented by the matrix `H`, and `θ̄` is a local minimum, then `H` is
positive semidefinite in the sense `Matrix.PosSemidef` that `Assumptions.ForwardKLChain` asks for.

The Hermitian clause comes from mathlib's `second_derivative_symmetric` — Clairaut's theorem —
and the quadratic-form clause from `quadForm_nonneg_of_isLocalMin`. -/
theorem posSemidef_of_isLocalMin {d : ℕ} {f : EuclideanSpace ℝ (Fin d) → ℝ}
    {f' : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ}
    {A : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ}
    {thb : EuclideanSpace ℝ (Fin d)} {H : Matrix (Fin d) (Fin d) ℝ}
    (hf : ∀ y, HasFDerivAt f (f' y) y) (hA : HasFDerivAt f' A thb)
    (hmin : IsLocalMin f thb)
    (hH : ∀ u w : EuclideanSpace ℝ (Fin d), A u w = WithLp.ofLp w ⬝ᵥ (H *ᵥ WithLp.ofLp u)) :
    H.PosSemidef := by
  have hpi : ∀ p q : Fin d → ℝ, q ⬝ᵥ (H *ᵥ p) = p ⬝ᵥ (H *ᵥ q) := by
    intro p q
    have h := second_derivative_symmetric hf hA (WithLp.toLp 2 p) (WithLp.toLp 2 q)
    rw [hH, hH] at h
    simpa using h
  refine Matrix.posSemidef_iff_dotProduct_mulVec.mpr ⟨?_, ?_⟩
  · ext i j
    have h := hpi (Pi.single j 1) (Pi.single i 1)
    simpa [Matrix.conjTranspose_apply] using h.symm
  · intro p
    have h := quadForm_nonneg_of_isLocalMin hf hA hmin (WithLp.toLp 2 p)
    rw [hH] at h
    simpa using h

/-! ## Satisfiability: the softplus positivity map, and a quadratic

A hypothesis that no object satisfies discharges nothing, so this section exhibits the objects.
The paper's positivity map is a softplus, and the softplus is the interesting case because its Hessian is
genuinely non-constant: the Lipschitz modulus is a real number, not zero, and it is computed here.
The quadratic case is recorded too because it is the degenerate one, where the remainder vanishes
identically and (A4) holds with `Cr = 0`.

The softplus computation is exact. With `zeta x = log (1 + exp x)` the first derivative is the
sigmoid function `sigma`, the second is `sigma (1 - sigma)`, and the third is
`sigma (1 - sigma) (1 - 2 sigma)`, whose absolute value is bounded by `1/4` because
`0 <= sigma (1 - sigma) <= 1/4` and `|1 - 2 sigma| <= 1`. Hence the softplus second derivative is
`(1/4)`-Lipschitz on all of the line -- not merely on a band -- and (A4-rem) holds for it with
`Cr = 1/8`. The bound `1/4` is not the sharp one, the true supremum of the third derivative being
about `0.0962`, but it is a bound, it is explicit, and it is proved. -/

/-- The sigmoid function, the derivative of the softplus positivity map. The positivity map itself is not
named here: `ShoulderAttenuation.lean` already carries it as `softplus`, together with the
identification of its derivative as the logistic function, and duplicating either name would make
the two modules unimportable together. -/
noncomputable def sigmoid (x : ℝ) : ℝ := Real.exp x / (1 + Real.exp x)

/-- The second derivative of the softplus positivity map, `sigma (1 - sigma)`. -/
noncomputable def sigmoidDeriv (x : ℝ) : ℝ := sigmoid x * (1 - sigmoid x)

/-- The denominator of the sigmoid is positive, so the sigmoid and all of the derivative
computations below are well defined. -/
theorem zero_lt_one_add_exp (x : ℝ) : (0 : ℝ) < 1 + Real.exp x := by positivity

/-- The sigmoid is positive everywhere. -/
theorem sigmoid_pos (x : ℝ) : 0 < sigmoid x :=
  div_pos (Real.exp_pos x) (zero_lt_one_add_exp x)

/-- The sigmoid is bounded above by one. -/
theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  rw [sigmoid, div_lt_one (zero_lt_one_add_exp x)]
  linarith [Real.exp_pos x]

/-- The derivative of the softplus positivity map is the sigmoid function, restated here for the positivity map
written out rather than named. -/
theorem hasDerivAt_logOneAddExp (x : ℝ) :
    HasDerivAt (fun y : ℝ => Real.log (1 + Real.exp y)) (sigmoid x) x := by
  have h1 : HasDerivAt (fun y : ℝ => 1 + Real.exp y) (Real.exp x) x := by
    simpa using (Real.hasDerivAt_exp x).const_add (1 : ℝ)
  exact h1.log (ne_of_gt (zero_lt_one_add_exp x))

/-- The derivative of the sigmoid function is `sigma (1 - sigma)`, the softplus second
derivative. -/
theorem hasDerivAt_sigmoid (x : ℝ) : HasDerivAt sigmoid (sigmoidDeriv x) x := by
  have h1 : HasDerivAt (fun y : ℝ => 1 + Real.exp y) (Real.exp x) x := by
    simpa using (Real.hasDerivAt_exp x).const_add (1 : ℝ)
  have h2 := (Real.hasDerivAt_exp x).div h1 (ne_of_gt (zero_lt_one_add_exp x))
  have heq : (Real.exp x * (1 + Real.exp x) - Real.exp x * Real.exp x) / (1 + Real.exp x) ^ 2
      = sigmoidDeriv x := by
    rw [sigmoidDeriv, sigmoid]
    field_simp
  rwa [heq] at h2

/-- The softplus is strictly convex: its second derivative is positive everywhere. In one
dimension this is (A4)'s positive-semidefiniteness clause, satisfied strictly and without any
appeal to minimality. -/
theorem sigmoidDeriv_pos (x : ℝ) : 0 < sigmoidDeriv x := by
  rw [sigmoidDeriv]
  exact mul_pos (sigmoid_pos x) (by linarith [sigmoid_lt_one x])

/-- The third derivative of the softplus, computed. -/
theorem hasDerivAt_sigmoidDeriv (x : ℝ) :
    HasDerivAt sigmoidDeriv (sigmoidDeriv x * (1 - 2 * sigmoid x)) x := by
  have hd : HasDerivAt (fun y : ℝ => 1 - sigmoid y) (-sigmoidDeriv x) x :=
    (hasDerivAt_sigmoid x).const_sub 1
  have h := (hasDerivAt_sigmoid x).mul hd
  have heq : sigmoidDeriv x * (1 - sigmoid x) + sigmoid x * -sigmoidDeriv x
      = sigmoidDeriv x * (1 - 2 * sigmoid x) := by
    rw [sigmoidDeriv]; ring
  rwa [heq] at h

/-- The third derivative of the softplus is bounded by `1/4` in absolute value, everywhere. -/
theorem abs_sigmoidDeriv_deriv_le (x : ℝ) :
    |sigmoidDeriv x * (1 - 2 * sigmoid x)| ≤ 1 / 4 := by
  have ht0 := sigmoid_pos x
  have ht1 := sigmoid_lt_one x
  have hq0 : 0 ≤ sigmoidDeriv x := le_of_lt (sigmoidDeriv_pos x)
  have hq4 : sigmoidDeriv x ≤ 1 / 4 := by
    rw [sigmoidDeriv]
    nlinarith [sq_nonneg (2 * sigmoid x - 1)]
  have hfac : |1 - 2 * sigmoid x| ≤ 1 := by
    rw [abs_le]
    constructor <;> linarith
  rw [abs_mul, abs_of_nonneg hq0]
  calc sigmoidDeriv x * |1 - 2 * sigmoid x| ≤ (1 / 4) * 1 :=
        mul_le_mul hq4 hfac (abs_nonneg _) (by norm_num)
    _ = 1 / 4 := by norm_num

/-- **The softplus Hessian is `(1/4)`-Lipschitz on the whole line.** This is the analytic input
that (A4) needs, verified for the paper's positivity map with an explicit constant. -/
theorem abs_sigmoidDeriv_sub_le (x y : ℝ) :
    |sigmoidDeriv y - sigmoidDeriv x| ≤ (1 / 4) * |y - x| := by
  have h := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := sigmoidDeriv)
    (f' := fun z => sigmoidDeriv z * (1 - 2 * sigmoid z)) (s := Set.univ) (C := 1 / 4)
    (fun z _ => (hasDerivAt_sigmoidDeriv z).hasDerivWithinAt)
    (fun z _ => by rw [Real.norm_eq_abs]; exact abs_sigmoidDeriv_deriv_le z)
    convex_univ (Set.mem_univ x) (Set.mem_univ y)
  simpa [Real.norm_eq_abs] using h

/-- **(A4-rem) holds for the softplus positivity map with `Cr = 1/8`.** The first-order Taylor remainder
of the softplus gradient is bounded by `(1/8) v²` uniformly in the base point, so the hypothesis
the assumption modules carry is satisfied by the object the paper trains. -/
theorem abs_sigmoid_taylorRemainder_le (x v : ℝ) :
    |sigmoid (x + v) - sigmoid x - sigmoidDeriv x * v| ≤ (1 / 8) * v ^ 2 := by
  have hbound : ∀ t ∈ Icc (0 : ℝ) 1,
      |sigmoidDeriv (x + t * v) - sigmoidDeriv x| ≤ 1 / 4 * t * |v| := by
    intro t ht
    have h := abs_sigmoidDeriv_sub_le x (x + t * v)
    have hrw : |x + t * v - x| = t * |v| := by
      rw [add_sub_cancel_left, abs_mul, abs_of_nonneg ht.1]
    rw [hrw] at h
    calc |sigmoidDeriv (x + t * v) - sigmoidDeriv x| ≤ 1 / 4 * (t * |v|) := h
      _ = 1 / 4 * t * |v| := by ring
  have hmain := abs_taylorRemainder_le_of_lipschitz_deriv (f' := sigmoid)
    (f'' := sigmoidDeriv) (x := x) (v := v) (K := 1 / 4)
    (fun t _ => hasDerivAt_sigmoid _) hbound
  calc |sigmoid (x + v) - sigmoid x - sigmoidDeriv x * v| ≤ 1 / 4 / 2 * v ^ 2 := hmain
    _ = 1 / 8 * v ^ 2 := by ring

/-- **The degenerate case.** For an affine gradient map -- that is, for a quadratic loss -- the
first-order remainder vanishes identically, so (A4) holds exactly, with `Cr = 0` and no error
term at all. -/
theorem taylorRemainder_eq_zero_of_affine {G : E → F} {A : E →L[ℝ] F} {b : F}
    (hG : ∀ y, G y = A y + b) (x v : E) : taylorRemainder G A x v = 0 := by
  rw [taylorRemainder, hG, hG, map_add]
  abel

/-! ### A complete witness for all four clauses of (A4)

The pieces above are assembled into a single one-dimensional instance in which every clause of
`Assumptions.ForwardKLChain` holds simultaneously, for an arbitrary iterate-fluctuation field.
The loss is the softplus of the scalar parameter, the mean Hessian is the one-by-one matrix of the
softplus second derivative at the mean iterate, and the remainder constant is `1/8`. Nothing here
is asymptotic and no constant is left implicit, so (A4) is exhibited as a satisfiable, non-vacuous
hypothesis. -/

/-- The Euclidean norm of a one-component vector is the absolute value of its entry. -/
theorem l2Norm_fin_one (v : Fin 1 → ℝ) : l2Norm v = |v 0| := by
  rw [l2Norm]
  simp [dotProduct, Real.sqrt_mul_self_eq_abs]

/-- The gradient fluctuation of the scalar softplus witness. -/
noncomputable def sigmoidFluct {Ω : Type*} (thb : ℝ) (dth : Ω → Fin 1 → ℝ) (ω : Ω) :
    Fin 1 → ℝ := fun i => sigmoid (thb + dth ω i) - sigmoid thb

/-- The mean Hessian of the scalar softplus witness. -/
noncomputable def sigmoidHessian (thb : ℝ) : Matrix (Fin 1) (Fin 1) ℝ :=
  Matrix.diagonal fun _ => sigmoidDeriv thb

/-- The (A4) remainder of the scalar softplus witness, defined as the discrepancy. -/
noncomputable def sigmoidRemainder {Ω : Type*} (thb : ℝ) (dth : Ω → Fin 1 → ℝ) (ω : Ω) :
    Fin 1 → ℝ := sigmoidFluct thb dth ω - sigmoidHessian thb *ᵥ dth ω

/-- **(A4) is satisfiable.** On a one-dimensional lift with softplus loss, an arbitrary sample
space and an arbitrary iterate-fluctuation field, all four clauses of assumption (A4) hold at
once: the mean Hessian is positive semidefinite, the remainder constant `1/8` is non-negative,
the chain holds pointwise, and the remainder is quadratic in the fluctuation with that constant.

These are precisely the four fields of `Assumptions.ForwardKLChain`, in the same order. -/
theorem forwardKLChain_softplus_witness {Ω : Type*} (thb : ℝ) (dth : Ω → Fin 1 → ℝ) :
    (sigmoidHessian thb).PosSemidef
      ∧ (0 : ℝ) ≤ 1 / 8
      ∧ (∀ ω, sigmoidFluct thb dth ω
          = sigmoidHessian thb *ᵥ dth ω + sigmoidRemainder thb dth ω)
      ∧ (∀ ω, l2Norm (sigmoidRemainder thb dth ω) ≤ 1 / 8 * l2Norm (dth ω) ^ 2) := by
  refine ⟨?_, by norm_num, ?_, ?_⟩
  · rw [sigmoidHessian, Matrix.posSemidef_diagonal_iff]
    exact fun _ => le_of_lt (sigmoidDeriv_pos thb)
  · intro ω
    rw [sigmoidRemainder]
    abel
  · intro ω
    have hentry : sigmoidRemainder thb dth ω 0
        = sigmoid (thb + dth ω 0) - sigmoid thb - sigmoidDeriv thb * dth ω 0 := by
      simp only [sigmoidRemainder, sigmoidFluct, sigmoidHessian, Pi.sub_apply,
        Matrix.mulVec_diagonal]
    rw [l2Norm_fin_one, l2Norm_fin_one, hentry, sq_abs]
    exact abs_sigmoid_taylorRemainder_le thb (dth ω 0)

end IcnnLift
