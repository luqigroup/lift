"""PCP-Map on the Darcy inverse problem: PGD, direct softplus and the lift.

PCP-Map (Wang, Baptista, Marzouk, Ruthotto, Verma, SISC 2024) learns an
amortized conditional transport map through a partially input-convex potential
``F(x, y)``, convex in the unknown and free in the observation, and scores

    log p(x | y) = log p_prior(grad_x F(x, y)) + log det H_F(x, y).

The positivity constraint on the convex path's inter-layer weights is built
three ways: projection onto the cone, the direct softplus readout, and the lift,
which emits those weights from a permutation-invariant summary of the minibatch.

Two conditionings, kept separate
--------------------------------
``F(x, y)`` conditions the potential on the observation, which is PCP-Map's
amortization and stays live at test time: a new ``y`` gives a new posterior.
Separately, the lift emits the convex-path weights from a DeepSets summary of
the minibatch, a training-time reparametrization of the constrained weights that
has nothing to do with ``y``.

The summary is taken on a detached copy of the batch, and that is not cosmetic.
``PCPMap.gxinv`` differentiates the potential through ``x`` to build the
transport map, so if the emitted weights were a live function of the same
tensor, the derivative would pick up the path through the batch mean, every
sample's map value would depend on every other sample in the batch, and the
log-prior term would differentiate a potential the log-determinant term does not
see: ``gxinv_grad`` runs under ``torch.func`` and treats the cached weights as
constants. Detaching the summary leaves every gradient path into the hypernet's
own parameters intact.

Evaluation freezes the emitted weights against a fixed anchor drawn from the
train split, because a density needs a fixed model: with the readout live, the
score of a point would depend on which other points shared its batch and would
not integrate to one against any single model. With the weights frozen the
potential is strongly convex in ``x`` on all of ``R^D``, so its gradient is a
bijection and the scored density is proper. The frozen score depends on the
anchor, so the anchor is the whole train split, the minimum-variance choice for
a mean-pooled summary, and the spread over resampled anchors is recorded beside
it. Training itself runs with batch stochasticity live.

Matched initialization
----------------------
``PICNN.__init__`` rectifies its convex-path weights at construction in every
positivity mode, which leaves the three recipes at very different effective
weights before a single step: projection starts with half its coordinates
exactly at zero, while softplus starts at ``softplus(relu(w))``, bounded below
by ``softplus(0)``. Comparing from there measures the initializer, so every
recipe is re-initialized to a common effective convex body and the emitter's
output bias is set so the lift starts at the same scale.

What is measured
----------------
* Held-out negative log-likelihood on the full evaluation split, the headline
  three-way comparison.
* A per-direction contraction diagnostic: the curvature of the scored
  log-density along each probe direction, by central differences with a step
  scaled to that direction's reference spread, compared against the linearized
  Gaussian conditional precision the archive implies. This separates the
  directions the data informs from those where the posterior should revert
  toward the prior; without it the total likelihood is dominated by directions
  on which all three agree.
* Shoulder occupancy, reported as a change from iteration zero because its
  absolute level is an artifact of the initializer. Under projection the
  analogue of an attenuated coordinate is the active set the projection pins,
  namely a coordinate at zero whose gradient points out of the cone.

Which archive a run trains on
-----------------------------
``--data_path`` names the HDF5 archive; when it is empty, ``--data_tag``
resolves it (``50k`` for the 50,000/2,000 archive of
``scripts/prepare_darcy_kl.py``, ``5k`` for ``darcy_laundering.h5``, the
4,000/1,000 archive shipped with ``priorlaundermat``). Since ``data_path`` sits
outside the run identity, each record also carries ``data_path_resolved``,
``n_train`` and ``n_eval``, so a directory cannot be misread later.

Knobs whose defaults reproduce the shipped archives, each part of the run
identity: ``pool_norm`` and ``pool_scale`` standardize or pin the emitter's
readout input (see ``PicnnHypernetYConditional._readout``), ``readout_lr_mult``
sets the Adam step of the readout weights relative to the shared one, and
``init_pre_shift`` adds a constant to every pre-readout target, the colder start
that puts the softplus recipes inside the shoulder. ``mirror_figure`` only
decides whether ``visualize`` overwrites the paper's figure.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from projorg import checkpointsdir, plotsdir, gitdir

from lift.baselines.picnn import PICNN
from lift.baselines.picnn_hypernet import (
    PicnnHypernet,
    PicnnHypernetYConditional,
)
from lift.baselines.pcpmap import PCPMap
from lift.dataset.darcy_kl import DarcyKL
from lift.utils.paper_style import PALETTE, apply_paper_style

_REPO = gitdir()
# "hypernet_nocond" is the y-independent emitter: the same parameter count as
# the lift with no batch conditioning, which isolates raw overparametrization of
# the constrained weights from the lift's structure.
BACKENDS = ("pgd", "direct", "hypernet", "hypernet_nocond")
_HYPER_BACKENDS = ("hypernet", "hypernet_nocond")

# ``data_tag`` names an archive under ``data/datasets/darcy`` when ``data_path``
# is empty. ``5k`` is ``DarcyKL``'s own default.
_DATA_FILES = {
    "50k": (
        "darcy_kl_reference_path-_N-40_K-10_alpha-0.0_s-1.1_sigma-1.0_"
        "sigma_y-0.01_n_sensors-33_maxiter-40_n_train-50000_n_eval-2000_"
        "map_for_eval-1_seed_offset-100000_seed-0.h5"
    ),
    "5k": "darcy_laundering.h5",
}

# The palette carries no "pgd" key, so PGD's green is written out here rather
# than re-picked; the same value is used by every other renderer.
_COLOR = {
    "hypernet": PALETTE["hypernet"],
    "hypernet_nocond": "#9467bd",
    "direct": PALETTE["direct_softplus"],
    "pgd": "#2ca02c",
}

_LABEL = {
    "hypernet": "lift",
    "hypernet_nocond": "lift, no conditioning",
    "direct": "direct softplus",
    "pgd": "PGD",
}

# softplus'(u) = sigmoid(u); the shoulder is where that prefactor has
# collapsed below ``sigma_s``, i.e. u < logit(sigma_s).
_SHOULDER_SIGMA = 0.05


# The three initializations, with the alias spelling the cluster array and the
# local driver may each use. ``harmonized`` is the default.
_INIT_MODES = ("published", "pytorch", "harmonized")
_INIT_SCHEMES = {
    "prescribed": "published",
    "torch": "pytorch",
    "harmonized": "harmonized",
}


def _shoulder_threshold(sigma_s: float = _SHOULDER_SIGMA) -> float:
    return float(np.log(sigma_s / (1.0 - sigma_s)))


def _softplus_inv(w: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus, for w > 0."""
    return w + torch.log(-torch.expm1(-w))


class PCPMapDarcy:
    """Train and compare the positivity recipes on the Darcy problem."""

    def __init__(self, args) -> None:
        self.args = args
        self.device = (
            f"cuda:{args.gpu_id}"
            if torch.cuda.is_available() and int(args.gpu_id) >= 0
            else "cpu"
        )
        self._target: Optional[DarcyKL] = None

        unknown = [b for b in list(args.backends) if b not in BACKENDS]
        if unknown:
            raise ValueError(
                f"unknown backend(s) {unknown!r}; must be a subset of "
                f"{BACKENDS}.",
            )
        self.backends = list(args.backends)
        if not self.backends:
            raise ValueError("backends must be non-empty.")
        self.seeds = [int(s) for s in list(args.seeds)]
        self.results: Dict[str, List[dict]] = {}

    @property
    def target(self) -> DarcyKL:
        """Load the archive on first use.

        Kept lazy so ``--phase visualization`` renders from cached
        results without needing the data file present.
        """
        if self._target is None:
            self._target = DarcyKL(
                data_path=self._resolve_data_path(), device=self.device,
            )
            print(f"Target: {self._target}")
        return self._target

    def _resolve_data_path(self) -> str:
        """An explicit ``data_path`` wins; otherwise ``data_tag`` names it.

        A tag without a known file, or whose file is missing, is an error
        rather than a silent fall-through to the 5k archive.
        """
        explicit = str(getattr(self.args, "data_path", "") or "")
        if explicit:
            return explicit
        tag = str(getattr(self.args, "data_tag", "") or "")
        if tag not in _DATA_FILES:
            raise ValueError(
                f"data_tag {tag!r} names no archive (known: "
                f"{sorted(_DATA_FILES)}); pass --data_path.",
            )
        path = os.path.join(_REPO, "data", "datasets", "darcy", _DATA_FILES[tag])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"data_tag {tag!r} resolves to {path!r}, which is missing; "
                "pass --data_path.",
            )
        return path

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _init_mode(self) -> str:
        """Which initialization the run uses; ``harmonized`` when unset.

        Two spellings reach the same three settings: ``--init_mode`` takes
        ``published`` / ``pytorch`` / ``harmonized``, and ``--init_scheme``
        takes ``prescribed`` / ``torch`` / ``harmonized``. Either alone is fine,
        both together only if they agree, and an unrecognized value raises
        rather than falling back to the default.

        The setting is outside the run identity, so two settings are told apart
        by ``--experiment_name``, and the resolved mode is written into every
        record so a directory cannot be misread later.
        """
        mode = str(getattr(self.args, "init_mode", "") or "")
        scheme = str(getattr(self.args, "init_scheme", "") or "")
        if mode and mode not in _INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {sorted(_INIT_MODES)}; got "
                f"{mode!r}.",
            )
        if scheme:
            if scheme not in _INIT_SCHEMES:
                raise ValueError(
                    f"init_scheme must be one of {sorted(_INIT_SCHEMES)}; "
                    f"got {scheme!r}.",
                )
            mapped = _INIT_SCHEMES[scheme]
            if mode and mode != mapped:
                raise ValueError(
                    f"init_scheme {scheme!r} means init_mode {mapped!r}, "
                    f"but init_mode {mode!r} was also passed; give one.",
                )
            return mapped
        return mode or "harmonized"

    def _harmonize_init(
        self,
        picnn: PICNN,
        hypernet: Optional[nn.Module],
        seed: int,
    ) -> None:
        """Place the constrained parameters, three ways.

        The hook ``_build`` calls after construction; ``init_mode`` picks which
        of the three runs, and defaults to ``harmonized``.

        ``"published"`` -- the initialization the baselines themselves ship.
        Every recipe is left where its own constructor put it, so this method
        does nothing:

        * PGD and direct softplus: ``PICNN.__init__`` stores ``relu(U(-b, b))``
          with ``b = 1/sqrt(fan_in)``, half the coordinates exactly at zero,
          which is PCP-Map's own initial state, since their ``src/icnn.py``
          applies ``nonneg = F.relu`` to every convex-path weight at
          construction in both positivity modes.
        * Lift: ``PicnnHypernetYConditional.__init__`` fills the readout bias
          with -2.0 and, under ``readout_fanin_scale``, draws the readout
          weights from ``N(0, 1)``. The -2.0 is the emitter's shipped slack.

        ``"pytorch"`` -- plain default PyTorch initialization, which shows the
        reading does not depend on a particular start. Every constrained layer
        is handed back to ``nn.Linear.reset_parameters``
        (``kaiming_uniform_(a=sqrt(5))`` on the weight,
        ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` on the bias), undoing both fixups
        above: the convex path loses its rectification, so half its weights
        start negative, and the emitter loses both its ``N(0, 1)`` readout draw
        and its -2.0 slack, so the emission starts near zero. Only the
        initialization changes; the forward path, including ``pool_norm`` and
        the ``1/d_h`` readout scaling, is architecture and is untouched.

        ``"harmonized"`` -- see ``_apply_harmonized_init``.
        """
        mode = self._init_mode()
        if mode == "harmonized":
            self._apply_harmonized_init(picnn, hypernet, seed)
            return
        if mode == "published":
            return
        if mode != "pytorch":
            raise ValueError(
                f"init_mode must be one of 'published', 'pytorch', "
                f"'harmonized'; got {mode!r}.",
            )
        torch.manual_seed(seed + 1_000_003)
        with torch.no_grad():
            for lw in picnn.Lw:
                lw.reset_parameters()
            if hypernet is not None:
                hypernet.mlp[-1].reset_parameters()

    def _apply_harmonized_init(
        self,
        picnn: PICNN,
        hypernet: Optional[nn.Module],
        seed: int,
    ) -> None:
        """Put every recipe at the same effective convex body at step zero.

        ``PICNN.__init__`` rectifies its convex-path weights in every positivity
        mode, which leaves projection with half its coordinates at exactly zero
        while softplus reads ``softplus(relu(w))``. Comparing from there
        measures the initializer.

        The common target is softplus's own natural init,
        ``softplus(relu(U(-b, b)))``, the scale the network was designed around.
        Projection stores that effective weight directly, the softplus recipes
        store its softplus inverse, and the emitter's output bias is placed at
        the mean of that inverse.

        The choice of target matters beyond fairness. A target drawn at the raw
        initializer's own scale would have a mean near ``0.05``, whose softplus
        inverse sits near ``-3``, so every softplus recipe would begin inside
        the readout shoulder with the chain-rule prefactor already collapsed and
        nothing would move. A pre-readout near zero instead leaves every recipe
        at an unattenuated operating point, so any later drift into the shoulder
        is a property of the training dynamics rather than of the start.

        The emitted spread is matched in level only, not coordinate by
        coordinate.
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        # ``init_pre_shift`` adds a constant to every pre-readout target. At 0
        # nothing changes; at -3.5 every softplus recipe starts inside the
        # readout shoulder (threshold logit(0.05) = -2.94, effective weight
        # softplus(-3.5) = 0.03 against the default's ~0.7). Projection
        # receives the same effective weights and has no shoulder, so it is the
        # control.
        shift = float(getattr(self.args, "init_pre_shift", 0.0) or 0.0)
        pre_targets, targets = [], []
        for lw in picnn.Lw:
            fan_in = lw.weight.shape[1]
            bound = 1.0 / float(np.sqrt(fan_in))
            raw = torch.empty(lw.weight.shape).uniform_(
                -bound, bound, generator=generator,
            ).relu()
            if shift != 0.0:
                raw = raw + shift
            pre_targets.append(raw.to(lw.weight.device))
            targets.append(F.softplus(raw).to(lw.weight.device))

        with torch.no_grad():
            for lw, w, raw in zip(picnn.Lw, targets, pre_targets):
                if picnn.pos_constraint_mode == "cone":
                    lw.weight.data.copy_(w)
                else:
                    lw.weight.data.copy_(raw)

            if hypernet is not None:
                pre = torch.cat([r.reshape(-1) for r in pre_targets])
                linears = [m for m in hypernet.mlp if isinstance(m, nn.Linear)]
                head = linears[-1]
                # Shrink the head so the emitted vector starts near a
                # constant, then place that constant at the common scale.
                head.weight.data.mul_(0.1)
                head.bias.data.fill_(float(pre.mean()))

    def _build(self, backend: str, seed: int) -> dict:
        """Instantiate one recipe, seeding locally and explicitly."""
        args = self.args
        torch.manual_seed(seed)

        pos_mode = "cone" if backend == "pgd" else "softplus_reparam"
        picnn = PICNN(
            input_x_dim=self.target.D,
            input_y_dim=self.target.D_y,
            feature_dim=int(args.feature_dim),
            feature_y_dim=int(args.feature_y_dim),
            out_dim=1,
            num_layers=int(args.num_layers),
            pos_constraint_mode=pos_mode,
        ).to(self.device)

        hypernet: Optional[nn.Module] = None
        supplier = None
        if backend in _HYPER_BACKENDS:
            hidden = ",".join(str(h) for h in args.hyper_hidden_sizes)
            # The emitter's positivity mode must match the host PICNN and must
            # never be "cone": in cone mode the override is passed through a
            # ReLU, so every negative emitted coordinate would receive exactly
            # zero gradient into the emitter.
            if backend == "hypernet":
                hypernet = PicnnHypernetYConditional(
                    picnn.lw_weight_shapes,
                    input_x_dim=self.target.D,
                    input_y_dim=self.target.D_y,
                    cond_dim=int(args.cond_dim),
                    summary_dim=int(args.summary_dim),
                    phi_hidden=int(args.phi_hidden),
                    phi_out=int(args.phi_out),
                    rho_hidden=int(args.rho_hidden),
                    hidden_sizes=hidden,
                    pos_constraint_mode="softplus_reparam",
                    readout_fanin_scale=bool(int(getattr(args, "readout_fanin_scale", 0))),
                    pool_norm=bool(int(getattr(args, "pool_norm", 0) or 0)),
                    pool_scale=float(getattr(args, "pool_scale", 0.0) or 0.0),
                ).to(self.device)
            else:
                hypernet = PicnnHypernet(
                    picnn.lw_weight_shapes,
                    cond_dim=int(args.cond_dim),
                    hidden_sizes=hidden,
                    pos_constraint_mode="softplus_reparam",
                    readout_fanin_scale=bool(int(getattr(args, "readout_fanin_scale", 0))),
                    pool_norm=bool(int(getattr(args, "pool_norm", 0) or 0)),
                    pool_scale=float(getattr(args, "pool_scale", 0.0) or 0.0),
                ).to(self.device)
            supplier = hypernet.emit

        self._harmonize_init(picnn, hypernet, seed)

        prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(self.target.D, device=self.device),
                torch.ones(self.target.D, device=self.device),
            ),
            1,
        )
        model = PCPMap(prior, picnn, pos_weights_supplier=supplier).to(self.device)

        # The host PICNN's stored convex weights are dead under an
        # override, so they are excluded from the optimized set and the
        # clip groups rather than sitting there with a null gradient.
        if hypernet is None:
            picnn_params = list(model.parameters())
        else:
            dead = {id(lw.weight) for lw in picnn.Lw}
            picnn_params = [p for p in model.parameters() if id(p) not in dead]
        hyper_params = list(hypernet.parameters()) if hypernet is not None else []

        # Adam's default eps of 1e-8 is the same order as the constrained
        # weights' gradient RMS here (2.4e-8 per coordinate at initialization,
        # against 2e-4 for the unconstrained skip and observation paths), so at
        # that setting the denominator floor rather than the gradient sets their
        # step, and the optimizer would impose its own attenuation on exactly
        # the coordinates whose attenuation this experiment measures. Hence
        # ``adam_eps``.
        # ``readout_lr_mult`` gives the emitter's readout weights
        # ``lr * readout_lr_mult``, while its bias, the trunk, the DeepSets body
        # and every PICNN parameter keep the shared step. The emitted weight is
        # a sum over the d_h readout inputs, so under Adam one step moves it by
        # about ``lr * ||h||_1`` through the weights against ``lr`` through the
        # bias; the multiplier brings the former back to O(lr) without changing
        # the initialization. At 1.0 the optimizer is built as usual.
        readout_mult = float(getattr(self.args, "readout_lr_mult", 1.0) or 1.0)
        if hypernet is not None and readout_mult != 1.0:
            head_w = hypernet.mlp[-1].weight
            rest = [p for p in hyper_params if p is not head_w]
            param_groups = [
                {"params": picnn_params + rest},
                {"params": [head_w], "lr": float(self.args.lr) * readout_mult},
            ]
        else:
            param_groups = picnn_params + hyper_params
        optimizer = torch.optim.Adam(
            param_groups,
            lr=float(self.args.lr),
            eps=float(self.args.adam_eps),
            weight_decay=float(getattr(self.args, "weight_decay", 0.0)),
        )
        return {
            "model": model,
            "picnn": picnn,
            "hypernet": hypernet,
            "optimizer": optimizer,
            "groups": [g for g in (picnn_params, hyper_params) if g],
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _loglik(
        self,
        arm: dict,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample ``log p(x | y)``, evaluated in chunks.

        Runs under ``enable_grad`` because the map ``z = grad_x F``
        needs autograd through ``x`` even at evaluation. Parameter
        gradients are switched off for the duration so no
        double-backward graph is built and discarded.
        """
        chunk = int(self.args.eval_chunk)
        params = [p for g in arm["groups"] for p in g]
        flags = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        try:
            out = []
            with torch.enable_grad():
                for i in range(0, len(x), chunk):
                    xb = x[i:i + chunk].detach().clone().requires_grad_(True)
                    out.append(
                        arm["model"].loglik_picnn(xb, y[i:i + chunk]).detach()
                    )
        finally:
            for p, f in zip(params, flags):
                p.requires_grad_(f)
        return torch.cat(out)

    def _freeze_weights(self, arm: dict, anchor: int = 0) -> None:
        """Pin the emitted weights to a fixed anchor batch.

        ``anchor`` indexes ``DarcyKL.anchor_batches``; index zero is the whole
        train split. A no-op when there is no emitter.
        """
        hypernet = arm["hypernet"]
        if hypernet is None:
            return
        batches = self._anchors()
        ax, ay = batches[min(anchor, len(batches) - 1)]
        with torch.no_grad():
            if isinstance(hypernet, PicnnHypernetYConditional):
                hypernet.prepare(ax, ay)

    def _anchors(self) -> List[tuple]:
        if not hasattr(self, "_anchor_cache"):
            g = torch.Generator(device=self.target.x_train.device)
            g.manual_seed(12345)
            self._anchor_cache = self.target.anchor_batches(
                int(self.args.n_ref_batch), int(self.args.n_anchor_probes), g,
            )
        return self._anchor_cache

    def _eval_nll(self, arm: dict, n_eval: int = 0) -> float:
        """Mean held-out NLL over the eval split with weights frozen."""
        self._freeze_weights(arm)
        x = self.target.x_eval if n_eval <= 0 else self.target.x_eval[:n_eval]
        y = self.target.y_eval if n_eval <= 0 else self.target.y_eval[:n_eval]
        return float(-self._loglik(arm, x, y).mean().item())

    def _anchor_spread(self, arm: dict) -> Optional[dict]:
        """Sensitivity of the frozen score to the anchor batch."""
        if arm["hypernet"] is None:
            return None
        vals = []
        for k in range(len(self._anchors())):
            self._freeze_weights(arm, anchor=k)
            vals.append(
                float(-self._loglik(
                    arm, self.target.x_eval, self.target.y_eval,
                ).mean().item())
            )
        self._freeze_weights(arm)
        return {
            "full_split": vals[0],
            "resampled_mean": float(np.mean(vals[1:])) if len(vals) > 1 else None,
            "resampled_std": float(np.std(vals[1:])) if len(vals) > 1 else None,
        }

    def _readout_input_norm(self, arm: dict) -> float:
        """``||h||`` of the emitter's readout input from its last forward."""
        hypernet = arm["hypernet"]
        h = getattr(hypernet, "_last_readout_input", None) if hypernet is not None else None
        if h is None:
            return float("nan")
        return float(h.norm().item())

    def _pre_readout_values(self, arm: dict) -> torch.Tensor:
        """The values the positivity readout is applied to.

        For direct softplus and PGD these are the stored convex weights; for
        the lift they are the emitted vector. This is the coordinate whose sign
        and magnitude decide whether the chain-rule prefactor has collapsed.
        """
        if arm["hypernet"] is not None:
            self._freeze_weights(arm)
            return torch.cat([
                w.reshape(-1) for w in arm["hypernet"].emit().values()
            ]).detach()
        return torch.cat([
            lw.weight.reshape(-1) for lw in arm["picnn"].Lw
        ]).detach()

    def _pre_readout_percentiles(self, arm: dict) -> list:
        """[min, 1st, 5th, 50th, 95th] percentiles of the pre-readout."""
        with torch.no_grad():
            v = self._pre_readout_values(arm).float()
            qs = torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95], device=v.device)
            return torch.quantile(v, qs).cpu().tolist()

    def _constrained_grad_rms(self, arm: dict) -> float:
        """Per-coordinate gradient RMS on the constrained parameters.

        For direct softplus and PGD these are the convex-path weights
        themselves; for the lift they are the emitter's parameters, which stand
        in the same place in the chain.
        """
        if arm["hypernet"] is not None:
            params = list(arm["hypernet"].parameters())
        else:
            params = [lw.weight for lw in arm["picnn"].Lw]
        total, n = 0.0, 0
        for p in params:
            if p.grad is None:
                continue
            total += float((p.grad ** 2).sum())
            n += p.numel()
        return float((total / n) ** 0.5) if n else float("nan")

    def _shoulder_occupancy(self, arm: dict) -> float:
        """Fraction of constrained coordinates with an attenuated update.

        Under softplus this is the fraction of pre-readout values below the
        shoulder threshold. Under projection the analogue is the active set:
        coordinates pinned at zero whose gradient points out of the cone. A
        coordinate merely sitting at zero is not attenuated, since the next
        step's gradient is the ordinary one and it leaves zero if that gradient
        points inward.
        """
        picnn = arm["picnn"]
        with torch.no_grad():
            if arm["hypernet"] is not None:
                self._freeze_weights(arm)
                vals = torch.cat([
                    w.reshape(-1) for w in arm["hypernet"].emit().values()
                ])
                return float((vals < _shoulder_threshold()).float().mean())
            if picnn.pos_constraint_mode == "softplus_reparam":
                raw = torch.cat([lw.weight.reshape(-1) for lw in picnn.Lw])
                return float((raw < _shoulder_threshold()).float().mean())

        # Projection needs a gradient, so it is taken outside ``no_grad`` on a
        # fresh batch.
        x, y = self.target.train_batch(int(self.args.batch_size))
        x = x.detach().clone().requires_grad_(True)
        arm["model"].zero_grad(set_to_none=True)
        arm["model"].loss(x, y).backward()
        active = []
        for lw in picnn.Lw:
            if lw.weight.grad is None:
                continue
            active.append(
                ((lw.weight.data <= 0) & (lw.weight.grad >= 0)).reshape(-1)
            )
        arm["model"].zero_grad(set_to_none=True)
        if not active:
            return float("nan")
        return float(torch.cat(active).float().mean())

    def _directional_precision(self, arm: dict) -> Dict[str, object]:
        """Learned conditional precision along the probe directions.

        Central second difference of the scored log-density along each
        direction. The step is scaled to that direction's reference spread
        rather than held fixed: a fixed step spans a large fraction of a sharply
        contracted direction's posterior width while barely moving in an
        uninformed one, so truncation error would be worst exactly where the
        signal lives.
        """
        args = self.args
        self._freeze_weights(arm)
        n = int(args.n_precision_pairs)
        x = self.target.x_eval[:n]
        y = self.target.y_eval[:n]

        n_blind = int(args.n_blind_probe)
        dirs = self.target.probe_directions(n_blind)
        ref = self.target.conditional_precision_reference(dirs.shape[1])
        steps = float(args.fd_rel) / ref.sqrt()

        base = self._loglik(arm, x, y)
        prec = []
        for j in range(dirs.shape[1]):
            u = dirs[:, j].unsqueeze(0)
            h = float(steps[j])
            plus = self._loglik(arm, x + h * u, y)
            minus = self._loglik(arm, x - h * u, y)
            prec.append(float((-(plus - 2.0 * base + minus) / (h * h)).mean()))

        prec_arr = np.asarray(prec, dtype=np.float64)
        return {
            "precision": prec_arr.tolist(),
            "reference_conditional": ref.cpu().numpy().tolist(),
            "reference_residual": self.target.residual_precision_reference(
                n_blind,
            ).cpu().numpy().tolist(),
            "steps": steps.cpu().numpy().tolist(),
            "n_resolved": int(self.target.resolved.shape[1]),
            "n_blind": int(min(n_blind, self.target.blind.shape[1])),
            "n_informed": self.target.n_informed_directions(),
            "n_nonpositive": int((prec_arr <= 0).sum()),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _arm_path(self, backend: str, seed: int) -> str:
        return os.path.join(
            checkpointsdir(self.args.experiment), f"{backend}_seed{seed}.pth",
        )

    def _train_arm(self, backend: str, seed: int) -> dict:
        args = self.args
        arm = self._build(backend, seed)
        model, hypernet = arm["model"], arm["hypernet"]
        optimizer, picnn = arm["optimizer"], arm["picnn"]

        generator = torch.Generator(device=self.target.x_train.device)
        generator.manual_seed(seed)

        shoulder0 = self._shoulder_occupancy(arm)
        history = {"iters": [], "train_loss": [], "eval_nll": []}
        clip_hits = 0
        n_iters = int(args.n_iters)
        t_start = time.time()

        snap_dir = os.path.join(
            checkpointsdir(args.experiment), f"snaps_{backend}_seed{seed}",
        )
        if int(args.snap_every) > 0:
            os.makedirs(snap_dir, exist_ok=True)

        pbar = tqdm(range(n_iters), desc=f"{backend} seed{seed}", unit="it")
        for it in pbar:
            x_raw, y = self.target.train_batch(int(args.batch_size), generator)
            # The summary is taken on a detached copy. Without this the
            # emitted weights enter the transport map's own Jacobian and couple
            # samples across the batch.
            if hypernet is not None:
                if isinstance(hypernet, PicnnHypernetYConditional):
                    # The DeepSets summary mean-pools its conditioning set, so
                    # Var[summary] scales as 1/n_cond and the magnitude of the
                    # mechanism scales with it. The whole loss batch maximizes
                    # the correlation between the emission and the gradient but
                    # minimizes that variance; a strict subset trades the other
                    # way. n_cond = 0 means the full batch.
                    n_c = int(getattr(args, "n_cond", 0) or 0)
                    if 0 < n_c < x_raw.shape[0]:
                        hypernet.prepare(x_raw[:n_c].detach(), y[:n_c])
                    else:
                        hypernet.prepare(x_raw.detach(), y)
            x = x_raw.detach().clone().requires_grad_(True)
            loss = model.loss(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clipped per group: the emitter carries far more parameters than
            # the network it parametrizes, so one global norm would bind on the
            # lift and not the others, silently giving them different effective
            # learning rates.
            hit = False
            for group in arm["groups"]:
                total = torch.nn.utils.clip_grad_norm_(
                    group, float(args.grad_clip),
                )
                hit = hit or bool(total > float(args.grad_clip))
            clip_hits += int(hit)
            optimizer.step()
            # Projection onto the non-negative cone. Without it the convex
            # path is not non-negative and every log-det below is meaningless.
            if backend == "pgd":
                picnn.project_positive()

            if (it + 1) % int(args.eval_every) == 0 or it == 0:
                nll = self._eval_nll(arm)
                history["iters"].append(it + 1)
                history["train_loss"].append(float(loss.item()))
                history["eval_nll"].append(nll)
                # Recorded so a stall can be attributed: a constrained-weight
                # gradient that collapses relative to the unconstrained paths
                # is the readout doing its work, whereas one that never moved
                # at all points at the optimizer or the parametrization.
                history.setdefault("grad_rms_constrained", []).append(
                    self._constrained_grad_rms(arm)
                )
                # The shoulder begins where softplus' falls below sigma_s, so
                # a run whose lower percentiles never approach that threshold
                # never puts the readout under test, however long it trains.
                history.setdefault("theta_tail", []).append(
                    self._pre_readout_percentiles(arm)
                )
                # ``||h||`` of the emitter's readout input at the whole-train
                # anchor, which the ``_eval_nll`` above sets; the quantity the
                # 1/d_h rule and ``pool_norm`` are about. NaN without an
                # emitter.
                history.setdefault("readout_input_norm", []).append(
                    self._readout_input_norm(arm)
                )
                pbar.set_postfix(train=f"{loss.item():.2f}", eval=f"{nll:.2f}")

            if int(args.snap_every) > 0 and (it + 1) % int(args.snap_every) == 0:
                # The emitted convex weights are what the constrained-space
                # landscape consumes. The emitter's own state dict is two
                # orders of magnitude larger and is stored only under
                # ``--save_hyper_snaps 1``: it is the (phi, b) trajectory the
                # lifted-space landscape panel takes its two principal
                # directions from.
                snap = {"iter": it + 1, "picnn": picnn.state_dict()}
                if hypernet is not None:
                    self._freeze_weights(arm)
                    snap["emitted"] = {
                        k: v.detach().cpu()
                        for k, v in hypernet.emit().items()
                    }
                    if int(getattr(args, "save_hyper_snaps", 0) or 0):
                        snap["hypernet"] = {
                            k: v.detach().cpu().clone()
                            for k, v in hypernet.state_dict().items()
                        }
                torch.save(snap, os.path.join(snap_dir, f"snap_{it + 1:06d}.pth"))

        wall_train = time.time() - t_start
        record = {
            "backend": backend,
            "seed": seed,
            "history": history,
            "final_eval_nll": self._eval_nll(arm),
            "anchor_spread": self._anchor_spread(arm),
            "clip_fraction": clip_hits / max(1, n_iters),
            "shoulder_occupancy": self._shoulder_occupancy(arm),
            "shoulder_occupancy_init": shoulder0,
            "directional": self._directional_precision(arm),
            # Provenance and cost, for the summary tables only.
            "data_path_resolved": self._resolve_data_path(),
            "init_mode": self._init_mode(),
            "n_train": int(len(self.target.x_train)),
            "n_eval": int(len(self.target.x_eval)),
            "wall_seconds_train": float(wall_train),
            "wall_seconds_total": float(time.time() - t_start),
            "n_params_trainable": int(sum(
                p.numel() for g in arm["groups"] for p in g
            )),
        }
        torch.save(
            {
                "model": model.state_dict(),
                "hypernet": (
                    hypernet.state_dict() if hypernet is not None else None
                ),
                "record": record,
            },
            self._arm_path(backend, seed),
        )
        return record

    def train(self) -> None:
        records = []
        for backend in self.backends:
            for seed in self.seeds:
                path = self._arm_path(backend, seed)
                if os.path.isfile(path) and not bool(int(self.args.overwrite)):
                    print(f"Skipping {backend} seed{seed}: {path} exists.")
                    records.append(
                        torch.load(path, weights_only=False)["record"]
                    )
                    continue
                records.append(self._train_arm(backend, seed))
                self._dump(records)
        self._dump(records)

    def _dump(self, records: List[dict]) -> None:
        path = os.path.join(
            checkpointsdir(self.args.experiment), "results.json",
        )
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved to {path}")

    # ------------------------------------------------------------------
    # Reload and render
    # ------------------------------------------------------------------

    def load_checkpoint(self) -> None:
        ckpt_dir = checkpointsdir(self.args.experiment)
        path = os.path.join(ckpt_dir, "results.json")
        records: List[dict] = []
        if os.path.isfile(path):
            with open(path) as f:
                records = json.load(f)
        else:
            # Fall back to whatever per-run checkpoints survived, so an
            # interrupted sweep still renders.
            for p in sorted(glob.glob(os.path.join(ckpt_dir, "*_seed*.pth"))):
                records.append(torch.load(p, weights_only=False)["record"])
        if not records:
            raise FileNotFoundError(
                f"no results under {ckpt_dir!r}; run with --phase train first.",
            )
        self.results = {}
        for rec in records:
            self.results.setdefault(rec["backend"], []).append(rec)

    def visualize(self) -> None:
        if not self.results:
            raise RuntimeError("call load_checkpoint() before visualize().")
        apply_paper_style()
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
        order = [b for b in BACKENDS if b in self.results]

        # (a) held-out NLL versus iteration, every seed drawn.
        ax = axes[0]
        for backend in order:
            color = _COLOR[backend]
            for j, rec in enumerate(self.results[backend]):
                h = rec["history"]
                ax.plot(
                    h["iters"], h["eval_nll"], color=color, lw=1.4,
                    alpha=0.9 if j == 0 else 0.45,
                    label=_LABEL[backend] if j == 0 else None,
                )
        ax.set_xlabel("iteration")
        ax.set_ylabel("held-out negative log-likelihood")
        ax.set_title("(a) held-out loss")
        ax.legend(frameon=False)

        # (b) per-direction learned precision against the linearized
        # conditional reference, every seed drawn, on a symmetric-log axis so
        # a negative curvature is visible rather than silently dropped.
        ax = axes[1]
        ref_drawn = False
        for backend in order:
            for j, rec in enumerate(self.results[backend]):
                d = rec["directional"]
                prec = np.asarray(d["precision"])
                ax.plot(
                    np.arange(len(prec)), prec, color=_COLOR[backend], lw=1.4,
                    alpha=0.9 if j == 0 else 0.45,
                    label=_LABEL[backend] if j == 0 else None,
                )
                if not ref_drawn:
                    ref = np.asarray(d["reference_conditional"])
                    ax.plot(
                        np.arange(len(ref)), ref, color=PALETTE["target"],
                        lw=1.2, ls="--", label="linearized reference",
                    )
                    ax.axvline(
                        d["n_resolved"] - 0.5, color="#555555", lw=0.8, ls=":",
                    )
                    n_inf = d.get("n_informed", 0)
                    if 0 < n_inf < len(ref):
                        ax.axvline(
                            n_inf - 0.5, color="#555555", lw=0.8, ls="-.",
                        )
                    ref_drawn = True
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xlabel("direction, ordered by singular value")
        ax.set_ylabel("learned conditional precision")
        ax.set_title("(b) per-direction contraction")
        ax.legend(frameon=False)

        # (c) final held-out NLL relative to the weakest recipe, with the
        # per-seed spread, since the absolute value is O(100) nats while the
        # differences are a few nats.
        ax = axes[2]
        means, stds, names, colors = [], [], [], []
        for backend in order:
            vals = [r["final_eval_nll"] for r in self.results[backend]]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            names.append(_LABEL[backend])
            colors.append(_COLOR[backend])
        baseline = max(means)
        ax.barh(
            names, [baseline - m for m in means], xerr=stds,
            color=colors, height=0.55, error_kw={"lw": 1.0},
        )
        ax.set_xlabel("nats below the weakest arm")
        ax.set_title("(c) final loss")
        ax.grid(axis="y", visible=False)

        paths = [os.path.join(plotsdir(self.args.experiment), "pcpmap_darcy.pdf")]
        # The paper's figure is overwritten only when ``mirror_figure`` is set,
        # so a probe run cannot replace it by accident.
        if int(getattr(self.args, "mirror_figure", 1) or 0):
            paths.append(
                os.path.join(_REPO, "figures", "figures", "pcpmap_darcy.pdf"),
            )
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved to {path}")
        plt.close(fig)
