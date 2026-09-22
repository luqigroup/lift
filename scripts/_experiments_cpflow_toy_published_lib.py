r"""The published CP-Flow toy-density recipe, and the same recipe lifted.

A reproduction, not a variant: every hyperparameter is Huang+ ICLR 2021's,
taken from the upstream repository (https://github.com/CW-Huang/CP-Flow,
``train_toy.py``) and from the paper's appendix. Nothing is tuned here.

Two published configurations, selectable via ``recipe``:

* ``A`` -- the repo defaults, which are also the README command:
  ``--nblocks=1 --depth=20 --dimh=32``, 10 epochs. Upstream applies them to
  whatever ``--data`` is passed, so A is a repo default and not the published
  configuration for any particular target; every A run records that as
  ``recipe_note``.
* ``B`` -- the appendix table, one row per target, resolved by
  ``lift.baselines.cpflow_toy.resolve_recipe`` against the single copy of the
  table in ``PAPER_APPENDIX_TABLE``; 50 epochs. 2-spirals has no row there
  and borrows the Eight-Gaussians one, also recorded as ``recipe_note``.

Where the repo and the paper disagree the repo wins and the paper's value
stays reachable from the config: ``num_epochs`` is 10 in A and 50 in B, and
``logdet_mode="exact2x2"`` (default) is the closed-form 2x2 Hessian
determinant -- the appendix's brute force, and the zero-variance limit of the
repo's CG / Lanczos estimator at ``d = 2``, which
``logdet_mode="upstream_stochastic"`` selects instead.

``backend`` selects the positivity construction and is the only thing that
differs; the training, evaluation and data code is shared:

* ``published`` -- constrained weights as free raw parameters under the
  positivity map.
* ``pgd``       -- the same weights projected onto the non-negative orthant
  after every optimizer step.
* ``lifted``    -- the same ICNN3 with the constrained latent weight emitted
  as ``Theta_E h_phi(X) + b_h`` (learnable slack ``b_h`` plus an
  unconstrained DeepSets emitter on the batch), with the positivity map and
  the ``1 / fan_in`` gain applied to it by the unmodified ``PosLinear``.

Two properties of the lifted construction that the code does not show at a
glance. The emitted weights depend on a conditioning batch, so a test NLL has
to say which batch conditions the model that scores it:
``eval_condition_on="train_batch"`` (the default) pins the emission on a
fixed batch from the training set, so the scored model is one normalized
density, while ``"test_batch"`` lets each test batch condition its own
scoring, which is transductive and is not a normalized likelihood; both are
recorded on every run. And the map has to stay monotone for the log-det to
mean anything, so every Hessian determinant is audited and non-positive ones
are counted and reported (``hessian_det_audit_*``) rather than clamped.

Targets are the four upstream ``data/toy_data.py`` generators.
``eight_gaussians`` and ``two_spirals`` exist in both conventions
(``convention="cpflow"``, the upstream generators; ``convention="lift_f1"``,
the radius-2.0 / sigma-0.1 convention of
``scripts/experiments_cpflow_demo_2d_f1.py``); ``rings`` and ``one_moon``
exist in the CP-Flow convention only, and asking for ``lift_f1`` on either
fails at config resolution. Absolute nats are not comparable across
conventions, so the convention in use is named in the metrics file and on
every plot. Each target also carries an entropy floor and a moment-matched
single-Gaussian NLL, so a run reads as at-the-floor or collapsed.

Outputs, in the projorg per-config directories: ``metrics.json``,
``seed<k>_flow.pt`` (final state dict plus history), ``density_grid.pdf``
and ``test_nll_vs_iter.pdf``.

Config: ``configs/experiments_cpflow_toy_published.json``.
Driver:  ``scripts/experiments_cpflow_toy_published.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import ExitStack
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from projorg import checkpointsdir, plotsdir
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lift.baselines.cpflow_toy import (  # noqa: E402
    PAPER_APPENDIX_TABLE,
    PUBLISHED_RECIPES,
    build_toy_cpflow,
    check_target_convention,
    constrained_latent_weights,
    lifted_bodies,
    mark_actnorms_initialized,
    parameter_counts,
    pinned_emission,
    prime_actnorms,
    project_positive_all,
    read_det_audit,
    reset_det_audit,
    resolve_plogv,
    resolve_recipe,
    sample_target,
    softplus_shoulder_fraction,
    target_reference_stats,
    toy_logp,
)
from lift.utils.paper_style import apply_paper_style, palette  # noqa: E402

# ``CPFlowConfig`` records a CP-Flow model shape; reused here so the resolved
# architecture lands in metrics.json in the form the other CP-Flow
# experiments use. Its sibling ``build_cpflow_direct`` is not reusable: it
# builds ``ICNN2`` where the toy recipe needs the input-augmented ``ICNN3``,
# and it always emits ``nblocks + 1`` log-det ActNorms where ``train_toy.py``
# emits exactly one when ``nblocks == 1``.
from _experiments_cpflow_uci_lib import CPFlowConfig  # noqa: E402


DIM = 2


def _merge_audit(acc: Dict, a: Dict) -> None:
    """Fold one determinant-audit reading into a running accumulator."""
    acc["n_evaluated"] += int(a["n_evaluated"])
    acc["n_nonpositive"] += int(a["n_nonpositive"])
    acc["min_det"] = min(float(acc["min_det"]), float(a["min_det"]))


class CPFlowToyPublished:
    """Run the published CP-Flow toy recipe over a list of seeds."""

    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        # ``train_toy.py`` sets this at module scope, before any module is
        # constructed, so the whole stack including the parameters is double
        # precision. It must happen before the first build.
        torch.set_default_dtype(torch.float64)
        self.resolved = self._resolve()
        self.results: Dict = {}

    # ------------------------------------------------------------------
    # Config resolution.
    # ------------------------------------------------------------------
    def _resolve(self) -> Dict:
        a = self.args
        recipe = str(getattr(a, "recipe", "A")).upper()
        if recipe not in PUBLISHED_RECIPES and recipe != "CUSTOM":
            raise ValueError(
                f"recipe must be one of {sorted(PUBLISHED_RECIPES)} or "
                f"'custom'; got {recipe!r}",
            )
        target = str(a.target)
        convention = str(a.convention)
        # Fail before anything is built if the target does not have the
        # requested convention: rings and one_moon exist in the CP-Flow
        # convention only.
        check_target_convention(target, convention)
        # Recipe B is the appendix table, one row per target.
        # ``resolve_recipe`` does that lookup against the single copy of the
        # table in ``cpflow_toy``, so no architecture is written down twice.
        base = resolve_recipe(recipe, target)
        recipe_note = str(base["recipe_note"])
        recipe_arch_source = str(base["arch_source"])

        def _pick(name: str) -> int:
            # -1 in the config JSON means "take the recipe's value".
            v = int(getattr(a, name))
            return base[name] if v < 0 else v

        nblocks = _pick("nblocks")
        depth = _pick("depth")
        dimh = _pick("dimh")
        num_epochs = _pick("num_epochs")

        trainable_w0_cfg = int(getattr(a, "trainable_w0", -1))
        trainable_w0 = (
            (nblocks == 1) if trainable_w0_cfg < 0 else bool(trainable_w0_cfg)
        )
        # ``plogv`` is a string field so that "auto" (upstream's rule)
        # and an explicit negative log-variance are both expressible.
        plogv_cfg = str(getattr(a, "plogv", "auto")).strip().lower()
        plogv = (
            resolve_plogv(nblocks) if plogv_cfg in ("auto", "")
            else float(plogv_cfg)
        )

        backend = str(getattr(a, "backend", "published"))
        if backend not in ("published", "lifted", "pgd"):
            raise ValueError(
                f"backend must be 'published', 'lifted' or 'pgd', got {backend!r}",
            )

        cfg = CPFlowConfig(
            D=DIM,
            dimh=dimh,
            nhidden=depth,
            nblocks=nblocks,
            softplus_type=str(a.softplus_type),
            zero_softplus=bool(int(a.zero_softplus)),
            m1=int(a.m1),
            atol=float(a.atol),
            rtol=float(a.rtol),
            trainable_w0=trainable_w0,
        )
        n_train = int(a.n_train)
        batch = int(a.batch_size_train)
        iters_per_epoch = int(np.ceil(n_train / batch))
        hyper_hidden = [int(v) for v in getattr(a, "hyper_hidden_sizes")]
        return {
            "backend": backend,
            "recipe": recipe,
            "recipe_note": recipe_note,
            "recipe_arch_source": recipe_arch_source,
            "recipe_source": {
                "A": "CP-Flow train_toy.py argparse defaults == README "
                     "command: --nblocks=1 --depth=20 --dimh=32, 10 epochs; "
                     "upstream applies them to whatever --data is passed, so "
                     "A is a repo default and not the published "
                     "configuration for any particular target",
                "B": "CP-Flow appendix 'Architectural details for toy "
                     "density estimation', one row per target (One moon "
                     "5/3/32, Eight Gaussians 5/3/32, Rings 5/5/256); 50 "
                     "epochs, batch 128, lr 0.005",
            },
            "paper_appendix_table": PAPER_APPENDIX_TABLE,
            "model": {
                "arch": "ICNN3 (input-augmented ICNN)",
                "D": DIM,
                "nblocks": nblocks,
                "depth": depth,
                "dimh": dimh,
                "symm_act_first": bool(int(a.symm_act_first)),
                "softplus_type": str(a.softplus_type),
                "zero_softplus": bool(int(a.zero_softplus)),
                "bias_w1": float(a.bias_w1),
                "trainable_w0": trainable_w0,
                "plogv": plogv,
                "prior": f"N(0, exp({plogv}) I)",
                "logdet_mode": str(a.logdet_mode),
                "cpflow_config": vars(cfg),
            },
            "lift": {
                "backend": backend,
                "emitter": "lift.models.HyperNetwork (DeepSets)",
                "emitted": "PosLinear.weight of every ICNN3 block "
                           "(the constrained weights only)",
                "latent_weight": "Theta_E h_phi(X) + b_h",
                "slack": "the readout bias b_h, learnable, one free "
                         "entry per constrained weight",
                "hyper_hidden_sizes": hyper_hidden,
                "pool_norm": bool(int(a.pool_norm)),
                "pool_scale": float(a.pool_scale),
                "use_outer_net": bool(int(a.use_outer_net)),
                "readout_bias_init": float(a.readout_bias_init),
                "readout_weight_init_scale": float(
                    a.readout_weight_init_scale,
                ),
                "readout_bias_init_note": (
                    "0.0, NOT the HyperNetwork default -2.0: the "
                    "published init is zero-centred, and the two arms "
                    "must start at the same effective weight"
                ),
                "condition_on": "the batch entering each block, detached",
                "eval_condition_on": str(a.eval_condition_on),
                "eval_cond_batch_size": int(a.eval_cond_batch_size),
                "eval_cond_seed": int(a.eval_cond_seed),
                "eval_condition_on_note": (
                    "train_batch: emission pinned on a fixed 128-point "
                    "training batch; train_all: emission pinned ONCE on "
                    "the whole training set (each block conditioning on "
                    "its own input of that set); test_batch: transductive, "
                    "not a likelihood. All three are recorded at every "
                    "evaluation; eval_condition_on picks the headline."
                ),
                # Lift-only training knobs; the defaults reproduce the
                # shipped behavior exactly. See HyperNetwork for the decay
                # semantics.
                "emission_decay_frac": float(
                    getattr(a, "emission_decay_frac", 0.0),
                ),
                "emission_decay_end": float(
                    getattr(a, "emission_decay_end", 1.0),
                ),
                "emission_decay_floor": float(
                    getattr(a, "emission_decay_floor", 0.0),
                ),
                "emission_decay_ema": float(
                    getattr(a, "emission_decay_ema", 0.99),
                ),
                "body_layernorm": bool(int(
                    getattr(a, "hyper_body_layernorm", 0),
                )),
                "readout_fanin_scale": bool(int(
                    getattr(a, "readout_fanin_scale", 0),
                )),
                "body_lr_mult": float(getattr(a, "body_lr_mult", 1.0)),
                "readout_lr_mult": float(
                    getattr(a, "readout_lr_mult", 1.0),
                ),
                "lr_mult_note": (
                    "body_lr_mult scales the Adam lr of the DeepSets "
                    "encoder h_phi (inner_net, outer_net); readout_lr_mult "
                    "scales the lr of the readout WEIGHTS Theta_E. The "
                    "slack b_h and every ICNN parameter stay at the shared "
                    "lr and schedule (StepLR scales every group by gamma, "
                    "so the multipliers are preserved). At 1.0 / 1.0 the "
                    "optimiser is a single param group."
                ),
            } if backend == "lifted" else {"backend": backend},
            "optim": {
                "optimizer": "Adam",
                "lr": float(a.lr),
                "betas": [float(a.beta1), float(a.beta2)],
                "weight_decay": float(a.weight_decay),
                "lr_scheduler": "StepLR",
                "lr_step_size": int(a.lr_step_size),
                "lr_gamma": float(a.lr_gamma),
                "lr_step_every_iteration": True,
                "clip_grad": float(a.clip_grad),
                "clip_grad_note": (
                    "0 == upstream's CP-Flow setting; train_toy.py forces "
                    "clip_grad=10 only for its MAF/NAF (iaf/naf) baselines"
                ),
            },
            "data": {
                "target": target,
                "convention": convention,
                "n_train": n_train,
                "n_test": int(a.n_test),
                "batch_size_train": batch,
                "batch_size_test": int(a.batch_size_test),
                "data_seed": int(a.data_seed),
                "fixed_pre_drawn": True,
            },
            "schedule": {
                "num_epochs": num_epochs,
                "iters_per_epoch": iters_per_epoch,
                "n_iters": num_epochs * iters_per_epoch,
                "eval_every": int(a.eval_every),
                "log_every": int(a.log_every),
            },
            "dtype": "float64",
            "seeds": [int(s) for s in a.seeds],
        }

    # ------------------------------------------------------------------
    # Data.
    # ------------------------------------------------------------------
    def _data(self):
        """Fixed, pre-sampled train / test sets, shared across seeds.

        Upstream re-samples the toy set from the global numpy RNG on every
        launch. Pinning it to ``data_seed`` is what makes the per-seed test
        NLLs comparable: the seed then varies only the model init and the
        shuffling.
        """
        d = self.resolved["data"]
        rng = np.random.default_rng(int(d["data_seed"]))
        x_train = sample_target(
            d["target"], d["convention"], int(d["n_train"]), rng,
        )
        x_test = sample_target(
            d["target"], d["convention"], int(d["n_test"]), rng,
        )
        return x_train, x_test

    # ------------------------------------------------------------------
    # Model.
    # ------------------------------------------------------------------
    def _build(self):
        """Build the flow. The backend is the only thing that differs.

        Every other argument is read from the same resolved config whatever
        the backend, so an accidental divergence would have to be typed
        twice.
        """
        m = self.resolved["model"]
        lift = self.resolved["lift"]
        kw = {}
        if lift["backend"] == "lifted":
            kw = dict(
                hyper_hidden=list(lift["hyper_hidden_sizes"]),
                pool_norm=bool(lift["pool_norm"]),
                pool_scale=float(lift["pool_scale"]),
                use_outer_net=bool(lift["use_outer_net"]),
                readout_bias_init=float(lift["readout_bias_init"]),
                readout_weight_init_scale=float(
                    lift["readout_weight_init_scale"],
                ),
                # ``.get``: blobs saved before these keys existed resolve
                # to the shipped behavior.
                emission_decay_frac=float(
                    lift.get("emission_decay_frac", 0.0),
                ),
                emission_decay_end=float(lift.get("emission_decay_end", 1.0)),
                emission_decay_floor=float(
                    lift.get("emission_decay_floor", 0.0),
                ),
                emission_decay_ema=float(lift.get("emission_decay_ema", 0.99)),
                body_layernorm=bool(lift.get("body_layernorm", False)),
                readout_fanin_scale=bool(lift.get("readout_fanin_scale", False)),
            )
        return build_toy_cpflow(
            dim=DIM,
            dimh=int(m["dimh"]),
            depth=int(m["depth"]),
            nblocks=int(m["nblocks"]),
            symm_act_first=bool(m["symm_act_first"]),
            softplus_type=str(m["softplus_type"]),
            zero_softplus=bool(m["zero_softplus"]),
            bias_w1=float(m["bias_w1"]),
            trainable_w0=bool(m["trainable_w0"]),
            logdet_mode=str(m["logdet_mode"]),
            m1=int(m["cpflow_config"]["m1"]),
            atol=float(m["cpflow_config"]["atol"]),
            rtol=float(m["cpflow_config"]["rtol"]),
            backend=str(lift["backend"]),
            **kw,
        )

    def _eval_cond_batch(self, x_train) -> "torch.Tensor":
        """The fixed conditioning batch used at evaluation time.

        Sampled once from the training set with its own seed, so the emission
        that scores the test set carries no test information and is identical
        across evaluations and seeds. ``None`` for the published recipe,
        which emits nothing.
        """
        lift = self.resolved["lift"]
        if lift["backend"] != "lifted":
            return None
        n = int(lift["eval_cond_batch_size"])
        rng = np.random.default_rng(int(lift["eval_cond_seed"]))
        idx = rng.choice(x_train.shape[0], size=n, replace=False)
        return torch.from_numpy(x_train[idx]).to(self.device).view(-1, DIM)

    def _eval_cond_all(self, x_train) -> "torch.Tensor":
        """The whole training set as the evaluation conditioning set.

        The latent weight is then computed once from all the training data,
        each block conditioning on its own input of that set as in training,
        which gives the model's own density rather than one 128-point batch's
        sample of it. No test information is involved. ``None`` for the
        published recipe.
        """
        if self.resolved["lift"]["backend"] != "lifted":
            return None
        return torch.from_numpy(np.asarray(x_train)).to(self.device).view(
            -1, DIM,
        )

    @staticmethod
    def _frozen_emas(flow) -> ExitStack:
        """Context in which no diagnostic forward advances the decay EMA.

        A no-op unless ``emission_decay_frac > 0`` (see
        ``HyperNetwork.frozen_pooled_ema``).
        """
        st = ExitStack()
        for b in lifted_bodies(flow):
            st.enter_context(b.hypernet.frozen_pooled_ema())
        return st

    def _latent_snapshot(self, flow, x_cond) -> "torch.Tensor":
        """Flat copy of every constrained latent weight, for step logging.

        Published recipe: the raw ``PosLinear`` weights. Lifted: the emission
        pinned on the fixed 128-point training batch, read in eval mode with
        the decay EMA and the determinant audit restored afterwards, so the
        reading changes nothing about the run. The mean absolute difference
        between consecutive snapshots is the per-``log_every`` step of the
        latent weight.
        """
        was_training = flow.training
        flow.eval()
        saved = [
            (f, f._audit_n_seen, f._audit_n_nonpos.clone(),
             f._audit_min.clone())
            for f in flow.flows
            if getattr(f, "_audit_n_seen", None) is not None
        ]
        with self._frozen_emas(flow):
            with torch.no_grad():
                w = constrained_latent_weights(flow, x_cond=x_cond)
        for f, n_seen, n_nonpos, mn in saved:
            f._audit_n_seen = n_seen
            f._audit_n_nonpos = n_nonpos
            f._audit_min = mn
        if was_training:
            flow.train()
        return torch.cat([v.detach().flatten() for v in w.values()])

    @staticmethod
    def _slack_snapshot(flow):
        """Flat copy of every readout bias ``b_h`` (the lift's slack).

        The slack is the lift's counterpart of the published recipe's free
        latent weight: it enters the emission additively and is the one
        emitted-weight channel Adam moves by at most one nominal step size
        per iteration. Reading it beside the emitted weight makes the step
        probe a ratio measured inside one run. ``None`` for the published
        recipe, which has no readout.
        """
        bodies = lifted_bodies(flow)
        if not bodies:
            return None
        with torch.no_grad():
            return torch.cat([
                layer.bias.detach().flatten()
                for b in bodies
                for layer in b.hypernet.weight_predictors
            ]).clone()

    # ------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------
    def _test_nll(self, flow, test_loader, x_cond=None):
        """Mean test NLL in nats per 2-D sample, on the fixed test set.

        ``x_cond`` pins the lift's emission on a fixed batch, so the scored
        model is one density and the number is a normalized log-likelihood.
        ``x_cond=None`` lets each test batch condition the weights that score
        it: for the published recipe that is the same thing, since it emits
        nothing, and for the lift it is a transductive quantity that is not
        the likelihood of any single density, reported alongside and never as
        the headline.

        ``no_grad`` is safe: every second-derivative path inside the convex
        block opens its own ``enable_grad`` context.
        """
        plogv = float(self.resolved["model"]["plogv"])
        was_training = flow.training
        flow.eval()
        total = 0.0
        n = 0
        with pinned_emission(flow, x_cond):
            # Reset after pinning so the conditioning pass itself is not
            # counted; the audit then covers exactly the test set.
            reset_det_audit(flow)
            with torch.no_grad():
                for x in test_loader:
                    x = x.to(self.device).view(-1, DIM)
                    lp = toy_logp(flow, x, plogv)
                    total += float((-lp).sum().item())
                    n += int(x.shape[0])
            audit = read_det_audit(flow)
        if was_training:
            flow.train()
        return total / max(n, 1), audit

    def _evaluate(
        self, flow, test_loader, x_cond, is_lifted, x_cond_all=None,
    ) -> Dict:
        """Test NLL under every conditioning choice.

        The published recipe emits nothing, so the numbers coincide and are
        computed once. The lift reports all three:

        * ``cond_train`` -- emission pinned on a fixed training batch. The
          scored model is one fixed density, the number is a normalized
          log-likelihood, and no test information reaches the parameters.
        * ``cond_self``  -- each test batch conditions the weights that score
          it. This is what the model does at training time, but it is
          transductive, and the resulting quantity is not the log-density of
          any single distribution.
        * ``cond_all``   -- emission pinned once on the whole training set
          (``x_cond_all``). Also a normalized log-likelihood of one fixed
          density, with no test information. Computed whenever
          ``x_cond_all`` is given.

        ``eval_condition_on`` picks which one is the headline; the others are
        recorded next to it either way.
        """
        if not is_lifted:
            nll, audit = self._test_nll(flow, test_loader, None)
            return {
                "nll": nll, "nll_cond_train": nll, "nll_cond_all": nll,
                "nll_cond_self": nll, "audit": audit,
            }
        nll_train, audit_train = self._test_nll(flow, test_loader, x_cond)
        nll_self, audit_self = self._test_nll(flow, test_loader, None)
        audits = [audit_train, audit_self]
        if x_cond_all is not None:
            nll_all, audit_all = self._test_nll(
                flow, test_loader, x_cond_all,
            )
            audits.append(audit_all)
        else:
            nll_all, audit_all = float("nan"), None
        mode = str(self.resolved["lift"].get(
            "eval_condition_on", "train_batch",
        ))
        if mode == "test_batch":
            primary, audit = nll_self, audit_self
        elif mode == "train_batch":
            primary, audit = nll_train, audit_train
        elif mode == "train_all":
            if audit_all is None:
                raise ValueError(
                    "eval_condition_on='train_all' needs the training set "
                    "as x_cond_all",
                )
            primary, audit = nll_all, audit_all
        else:
            raise ValueError(
                "eval_condition_on must be 'train_batch', 'train_all' or "
                f"'test_batch', got {mode!r}",
            )
        # Every pass is audited; report the worst so a monotonicity
        # violation cannot hide in an unreported pass.
        merged = dict(audit)
        merged["n_nonpositive"] = max(a["n_nonpositive"] for a in audits)
        merged["min_det"] = min(a["min_det"] for a in audits)
        return {
            "nll": primary,
            "nll_cond_train": nll_train,
            "nll_cond_all": nll_all,
            "nll_cond_self": nll_self,
            "audit": merged,
        }

    # ------------------------------------------------------------------
    # One seed.
    # ------------------------------------------------------------------
    def _train_one_seed(self, seed: int, x_train, x_test) -> Dict:
        a = self.args
        res = self.resolved
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        xt = torch.from_numpy(x_train)
        xv = torch.from_numpy(x_test)
        # A Tensor is a valid map-style dataset (``__getitem__`` gives a
        # row), which is what upstream's ``ToyDataset`` yields. The shuffling
        # generator is pinned to the seed rather than left on the global RNG
        # as upstream leaves it: the backends consume different amounts of
        # global randomness while building their models, so without the pin
        # they would see different batch orders.
        g_shuffle = torch.Generator()
        g_shuffle.manual_seed(int(seed))
        train_loader = DataLoader(
            xt, batch_size=int(res["data"]["batch_size_train"]), shuffle=True,
            generator=g_shuffle,
        )
        # Upstream's test loader is ``shuffle=True``; the order cannot change
        # a mean, and ``shuffle=False`` keeps the evaluation deterministic.
        test_loader = DataLoader(
            xv, batch_size=int(res["data"]["batch_size_test"]), shuffle=False,
        )

        flow = self._build().to(self.device)
        x_cond = self._eval_cond_batch(x_train)
        x_cond_all = self._eval_cond_all(x_train)
        param_counts = parameter_counts(flow)
        is_lifted = bool(lifted_bodies(flow))
        lift = res["lift"]

        # Lift-only learning-rate multipliers. The slack b_h and every ICNN
        # parameter stay in the base group at the shared learning rate, and
        # at 1.0 / 1.0 the optimizer is a single parameter group.
        base_lr = float(res["optim"]["lr"])
        body_mult = float(lift.get("body_lr_mult", 1.0))
        readout_mult = float(lift.get("readout_lr_mult", 1.0))
        if is_lifted and (body_mult != 1.0 or readout_mult != 1.0):
            body_params, readout_weights = [], []
            for b in lifted_bodies(flow):
                body_params += list(b.hypernet.inner_net.parameters())
                if b.hypernet.outer_net is not None:
                    body_params += list(b.hypernet.outer_net.parameters())
                readout_weights += [
                    layer.weight for layer in b.hypernet.weight_predictors
                ]
            special = {id(p) for p in body_params + readout_weights}
            rest = [p for p in flow.parameters() if id(p) not in special]
            param_groups = [
                {"params": rest},
                {"params": body_params, "lr": base_lr * body_mult},
                {"params": readout_weights, "lr": base_lr * readout_mult},
            ]
        else:
            param_groups = flow.parameters()
        decay_on = is_lifted and float(lift.get("emission_decay_frac", 0.0)) > 0

        optim = torch.optim.Adam(
            param_groups,
            lr=base_lr,
            betas=(float(res["optim"]["betas"][0]),
                   float(res["optim"]["betas"][1])),
            weight_decay=float(res["optim"]["weight_decay"]),
        )
        sch = torch.optim.lr_scheduler.StepLR(
            optim,
            int(res["optim"]["lr_step_size"]),
            float(res["optim"]["lr_gamma"]),
        )

        # One priming forward pass on the first batch for the data-dependent
        # ActNorm init, before any optimizer step, as upstream does.
        for x in train_loader:
            prime_actnorms(flow, x.to(self.device).view(-1, DIM))
            break

        thr = float(a.shoulder_deriv_threshold)
        # Measured on the latent weight under every backend: the raw
        # parameter for the published recipe, the emitted
        # Theta_E h_phi(X) + b_h for the lift. If the two differ at init the
        # comparison is measuring the initialization, not the construction.
        with self._frozen_emas(flow):
            shoulder_init = softplus_shoulder_fraction(
                flow, thr, x_cond=x_cond,
            )
        print(
            f"[seed {seed}] init shoulder fraction "
            f"{shoulder_init['fraction']:.4g} "
            f"(source: {shoulder_init['source']}; "
            f"mean w {shoulder_init['mean_w']:.4g}, "
            f"std w {shoulder_init['std_w']:.4g}, "
            f"mean softplus(w) {shoulder_init['mean_softplus_w']:.4g})",
            flush=True,
        )
        reset_det_audit(flow)
        prev_snapshot = self._latent_snapshot(flow, x_cond)

        n_iters = int(res["schedule"]["n_iters"])
        eval_every = int(res["schedule"]["eval_every"])
        log_every = int(res["schedule"]["log_every"])
        clip_grad = float(res["optim"]["clip_grad"])

        history = {
            "iter": [], "train_nll": [], "grad_norm": [],
            "latent_step_mean_abs": [],
            "eval_iter": [], "test_nll": [],
            "test_nll_cond_train": [], "test_nll_cond_all": [],
            "test_nll_cond_self": [], "emission_gain": [],
            "test_nonpositive_det": [], "eval_wall_seconds": [],
        }
        # Opt-in per-iteration step probe, off unless the config carries
        # ``step_probe_iters > 0``: the shipped config has no such key, so
        # ``getattr`` returns 0 and neither the history keys nor the hot loop
        # change. When on, the emitted latent weight is snapshotted after
        # each of the first ``step_probe_iters`` optimizer steps and the
        # per-coordinate |delta| is summarized, which measures how far one
        # Adam step moves an emitted weight in units of the nominal step
        # size.
        probe_n = int(getattr(a, "step_probe_iters", 0) or 0)
        probe_prev = None
        probe_slack = None
        if probe_n > 0:
            history["step_probe"] = {
                "iter": [], "lr": [], "median_abs": [], "mean_abs": [],
                "p90_abs": [], "slack_median_abs": [],
            }
            probe_prev = prev_snapshot.clone()
            probe_slack = self._slack_snapshot(flow)

        best = {"test_nll": float("inf"), "iter": -1}
        train_audit = {
            "n_evaluated": 0, "n_nonpositive": 0, "min_det": float("inf"),
        }
        loss_acc = 0.0
        t = 0
        grad_norm = float("nan")
        t0 = time.time()
        flow.train()
        for epoch in range(int(res["schedule"]["num_epochs"])):
            for x in train_loader:
                x = x.to(self.device).view(-1, DIM)
                if decay_on:
                    # Emission-decay progress: gain 1 until the window opens,
                    # then linear to the floor (see HyperNetwork).
                    for b in lifted_bodies(flow):
                        b.hypernet.set_train_progress(t / n_iters)
                loss = -toy_logp(flow, x, float(res["model"]["plogv"])).mean()
                optim.zero_grad()
                loss.backward()
                if clip_grad == 0:
                    ps = [p for p in flow.parameters() if p.grad is not None]
                    grad_norm = float(torch.norm(torch.stack(
                        [torch.norm(p.grad.detach(), 2.0) for p in ps]), 2.0,
                    ).item())
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        flow.parameters(), clip_grad,
                    ).item())
                lr_used = (
                    float(optim.param_groups[0]["lr"]) if probe_n else 0.0
                )
                optim.step()
                if res["backend"] == "pgd":
                    project_positive_all(flow)      # the projection of the PGD step
                # StepLR is stepped every iteration upstream, not every
                # epoch: with step_size=2000 the learning rate halves every
                # 2000 iterations.
                sch.step()

                loss_acc += float(loss.item())
                t += 1
                if probe_n and t <= probe_n:
                    snap = self._latent_snapshot(flow, x_cond)
                    dlt = (snap - probe_prev).abs().double()
                    probe_prev = snap
                    sp = history["step_probe"]
                    sp["iter"].append(t)
                    sp["lr"].append(lr_used)
                    sp["median_abs"].append(float(dlt.median().item()))
                    sp["mean_abs"].append(float(dlt.mean().item()))
                    sp["p90_abs"].append(float(
                        torch.quantile(dlt, 0.9).item(),
                    ))
                    if probe_slack is not None:
                        sl = self._slack_snapshot(flow)
                        sp["slack_median_abs"].append(float(
                            (sl - probe_slack).abs().double().median().item(),
                        ))
                        probe_slack = sl
                if t % log_every == 0:
                    snapshot = self._latent_snapshot(flow, x_cond)
                    history["latent_step_mean_abs"].append(float(
                        (snapshot - prev_snapshot).abs().mean().item(),
                    ))
                    prev_snapshot = snapshot
                    history["iter"].append(t)
                    history["train_nll"].append(loss_acc / log_every)
                    history["grad_norm"].append(grad_norm)
                    print(
                        f"[seed {seed}] epoch {epoch} iter {t}/{n_iters} "
                        f"train nll {loss_acc / log_every:.4f} "
                        f"grad norm {grad_norm:.3g} "
                        f"lr {optim.param_groups[0]['lr']:.3g}",
                        flush=True,
                    )
                    loss_acc = 0.0
                if (eval_every > 0 and t % eval_every == 0) or t == n_iters:
                    te = time.time()
                    # Fold the training segment's determinant audit away
                    # before the test pass resets the counters.
                    _merge_audit(train_audit, read_det_audit(flow))
                    ev = self._evaluate(
                        flow, test_loader, x_cond, is_lifted, x_cond_all,
                    )
                    reset_det_audit(flow)
                    nll = ev["nll"]
                    history["eval_iter"].append(t)
                    history["test_nll"].append(nll)
                    history["test_nll_cond_train"].append(
                        ev["nll_cond_train"],
                    )
                    history["test_nll_cond_all"].append(ev["nll_cond_all"])
                    history["test_nll_cond_self"].append(ev["nll_cond_self"])
                    history["emission_gain"].append(float(
                        lifted_bodies(flow)[0].hypernet.emission_gain
                        if is_lifted else 1.0
                    ))
                    history["test_nonpositive_det"].append(
                        int(ev["audit"]["n_nonpositive"]),
                    )
                    history["eval_wall_seconds"].append(time.time() - te)
                    if nll < best["test_nll"]:
                        best = {"test_nll": nll, "iter": t}
                    if ev["audit"]["n_nonpositive"]:
                        print(
                            f"[seed {seed}] WARNING iter {t}: "
                            f"{ev['audit']['n_nonpositive']} of "
                            f"{ev['audit']['n_evaluated']} test Hessian "
                            f"determinants were non-positive (min "
                            f"{ev['audit']['min_det']:.3g}) -- the map is "
                            "not monotone there, so the log-det is not a "
                            "log-det and the NLL is not a likelihood",
                            flush=True,
                        )
                    extra = (
                        f" | pinned-128 {ev['nll_cond_train']:.4f}"
                        f" | train-all {ev['nll_cond_all']:.4f}"
                        f" | self-conditioned {ev['nll_cond_self']:.4f}"
                        if is_lifted else ""
                    )
                    print(
                        f"[seed {seed}] iter {t} TEST NLL {nll:.4f} nats "
                        f"(best {best['test_nll']:.4f} @ {best['iter']})"
                        + extra,
                        flush=True,
                    )
        wall = time.time() - t0

        _merge_audit(train_audit, read_det_audit(flow))
        if history["test_nll"]:
            final_nll = history["test_nll"][-1]
            final_cond_train = history["test_nll_cond_train"][-1]
            final_cond_all = history["test_nll_cond_all"][-1]
            final_cond_self = history["test_nll_cond_self"][-1]
            final_audit = {
                "n_nonpositive": int(history["test_nonpositive_det"][-1]),
            }
        else:
            ev = self._evaluate(
                flow, test_loader, x_cond, is_lifted, x_cond_all,
            )
            final_nll = ev["nll"]
            final_cond_train = ev["nll_cond_train"]
            final_cond_all = ev["nll_cond_all"]
            final_cond_self = ev["nll_cond_self"]
            final_audit = ev["audit"]
        # Read the end-of-training shoulder after the train audit is folded
        # away: the lift's reading runs a conditioning forward pass, which
        # would otherwise land in the audit. The shoulder share is a function
        # of the conditioning set, so it is read under both the 128-point
        # batch and the whole training set.
        with self._frozen_emas(flow):
            shoulder_final = softplus_shoulder_fraction(
                flow, thr, x_cond=x_cond,
            )
            shoulder_final_all = (
                softplus_shoulder_fraction(flow, thr, x_cond=x_cond_all)
                if is_lifted else shoulder_final
            )
        peak_mem_mb = (
            float(torch.cuda.max_memory_allocated(self.device)) / 2**20
            if self.device.type == "cuda" else float("nan")
        )

        step_probe_summary = None
        if probe_n > 0 and history["step_probe"]["iter"]:
            sp = history["step_probe"]
            lrs = np.asarray(sp["lr"], dtype=float)
            med = np.asarray(sp["median_abs"], dtype=float)
            mea = np.asarray(sp["mean_abs"], dtype=float)
            ratio = med / lrs
            step_probe_summary = {
                "n_iters": int(len(sp["iter"])),
                "quantity": (
                    "per-iteration |delta| of the emitted (pre-positivity) "
                    "constrained weight, pinned on the fixed 128-point "
                    "training conditioning batch, divided by the nominal "
                    "Adam step size in force at that iteration"
                ),
                "median_step_ratio": float(np.median(ratio)),
                "mean_step_ratio": float(np.mean(ratio)),
                "p10_step_ratio": float(np.percentile(ratio, 10)),
                "p90_step_ratio": float(np.percentile(ratio, 90)),
                "mean_abs_step_ratio": float(np.mean(mea / lrs)),
                "first_iter_step_ratio": float(ratio[0]),
                "lr_nominal": float(lrs[0]),
            }
            if sp["slack_median_abs"]:
                sl = np.asarray(sp["slack_median_abs"], dtype=float)
                step_probe_summary["slack_median_step_ratio"] = float(
                    np.median(sl / lrs[:len(sl)]),
                )
            print(
                f"[seed {seed}] step probe over {step_probe_summary['n_iters']}"
                f" iterations: median |d theta| / lr = "
                f"{step_probe_summary['median_step_ratio']:.4g} "
                f"(mean {step_probe_summary['mean_step_ratio']:.4g}, "
                f"slack "
                f"{step_probe_summary.get('slack_median_step_ratio', float('nan')):.4g})",
                flush=True,
            )

        ckpt_dir = checkpointsdir(self.args.experiment)
        torch.save(
            {
                "state_dict": flow.state_dict(),
                "resolved": self.resolved,
                "seed": int(seed),
                "history": history,
            },
            os.path.join(ckpt_dir, f"seed{int(seed)}_flow.pt"),
        )
        return {
            "seed": int(seed),
            "backend": str(self.resolved["backend"]),
            "final_test_nll_nats": float(final_nll),
            "final_test_nll_cond_train_nats": float(final_cond_train),
            "final_test_nll_cond_all_nats": float(final_cond_all),
            "final_test_nll_cond_self_nats": float(final_cond_self),
            "eval_condition_on": str(
                self.resolved["lift"].get("eval_condition_on", "n/a")
            ),
            "best_test_nll_nats": float(best["test_nll"]),
            "best_iter": int(best["iter"]),
            "final_iter": int(t),
            "wall_seconds": float(wall),
            "n_parameters": param_counts,
            "softplus_shoulder_at_init": shoulder_init,
            "softplus_shoulder_at_end": shoulder_final,
            "softplus_shoulder_at_end_cond_all": shoulder_final_all,
            "peak_cuda_mem_mb": peak_mem_mb,
            "step_probe_summary": step_probe_summary,
            "hessian_det_audit_train": train_audit,
            "hessian_det_audit_test_final": final_audit,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Phases.
    # ------------------------------------------------------------------
    def train(self) -> None:
        ckpt_dir = checkpointsdir(self.args.experiment)
        os.makedirs(ckpt_dir, exist_ok=True)
        x_train, x_test = self._data()
        d = self.resolved["data"]
        ref = target_reference_stats(
            d["target"], d["convention"], x_train, x_test,
            n_latents=int(self.args.ref_n_latents),
            n_eval=int(self.args.ref_n_eval),
        )
        print(
            f"[target] {d['target']} / convention {d['convention']}: "
            f"entropy floor {ref['entropy_floor_nats']:.4f} nats, "
            f"moment-matched Gaussian "
            f"{ref['gaussian_moment_matched_nll_nats']:.4f} nats",
            flush=True,
        )

        results = {
            "arm": f"cpflow_toy_{self.resolved['backend']}",
            "backend": self.resolved["backend"],
            "provenance": (
                "Huang, Chen, Tsirigotis, Courville, 'Convex Potential "
                "Flows', ICLR 2021 -- train_toy.py + appendix; reproduced, "
                "not tuned"
            ),
            "resolved_config": self.resolved,
            "convention": d["convention"],
            "target": d["target"],
            "target_reference": ref,
            "device": str(self.device),
            "per_seed": {},
        }
        for s in self.resolved["seeds"]:
            print(f"\n=== seed {s} ===", flush=True)
            results["per_seed"][str(int(s))] = self._train_one_seed(
                int(s), x_train, x_test,
            )
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump(results, f, indent=2)

        ps = list(results["per_seed"].values())
        fin = [v["final_test_nll_nats"] for v in ps]
        bst = [v["best_test_nll_nats"] for v in ps]
        results["summary"] = {
            "backend": self.resolved["backend"],
            "final_test_nll_mean": float(np.mean(fin)),
            "final_test_nll_std": float(np.std(fin, ddof=1)) if len(fin) > 1
            else 0.0,
            "best_test_nll_mean": float(np.mean(bst)),
            "best_test_nll_std": float(np.std(bst, ddof=1)) if len(bst) > 1
            else 0.0,
            "final_test_nll_cond_train_mean": float(np.mean(
                [v["final_test_nll_cond_train_nats"] for v in ps],
            )),
            "final_test_nll_cond_all_mean": float(np.mean(
                [v["final_test_nll_cond_all_nats"] for v in ps],
            )),
            "final_test_nll_cond_self_mean": float(np.mean(
                [v["final_test_nll_cond_self_nats"] for v in ps],
            )),
            "shoulder_fraction_at_init": float(np.mean(
                [v["softplus_shoulder_at_init"]["fraction"] for v in ps],
            )),
            "shoulder_fraction_at_end": float(np.mean(
                [v["softplus_shoulder_at_end"]["fraction"] for v in ps],
            )),
            "shoulder_fraction_at_end_cond_all": float(np.mean(
                [v["softplus_shoulder_at_end_cond_all"]["fraction"]
                 for v in ps],
            )),
            "n_parameters_trainable": ps[0]["n_parameters"]["trainable"],
            "n_nonpositive_det_test_total": int(sum(
                sum(v["history"]["test_nonpositive_det"]) for v in ps
            )),
            "n_nonpositive_det_train_total": int(sum(
                v["hessian_det_audit_train"]["n_nonpositive"] for v in ps
            )),
            "wall_seconds_total": float(sum(
                v["wall_seconds"] for v in ps
            )),
        }
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        self.results = results
        sm = results["summary"]
        print(
            f"\n[summary] backend={sm['backend']} | final test NLL "
            f"{sm['final_test_nll_mean']:.4f} +/- "
            f"{sm['final_test_nll_std']:.4f} nats "
            f"(cond_train {sm['final_test_nll_cond_train_mean']:.4f}, "
            f"cond_all {sm['final_test_nll_cond_all_mean']:.4f}, "
            f"cond_self {sm['final_test_nll_cond_self_mean']:.4f}) | "
            f"floor {ref['entropy_floor_nats']:.4f} | Gaussian "
            f"{ref['gaussian_moment_matched_nll_nats']:.4f} | "
            f"shoulder init {sm['shoulder_fraction_at_init']:.4g} -> end "
            f"{sm['shoulder_fraction_at_end']:.4g} | trainable params "
            f"{sm['n_parameters_trainable']} | non-positive dets "
            f"{sm['n_nonpositive_det_test_total']} test / "
            f"{sm['n_nonpositive_det_train_total']} train",
            flush=True,
        )

    def load_checkpoint(self) -> None:
        path = os.path.join(
            checkpointsdir(self.args.experiment), "metrics.json",
        )
        if os.path.exists(path):
            with open(path) as f:
                self.results = json.load(f)

    def _load_flow(self, seed: int):
        path = os.path.join(
            checkpointsdir(self.args.experiment), f"seed{int(seed)}_flow.pt",
        )
        if not os.path.exists(path):
            return None, None
        blob = torch.load(path, map_location=self.device, weights_only=False)
        flow = self._build().to(self.device)
        flow.load_state_dict(blob["state_dict"])
        # ``ActNorm.initialized`` is not part of the state dict; without this
        # the first forward pass would re-run the data-dependent init and
        # overwrite the trained parameters.
        mark_actnorms_initialized(flow)
        flow.eval()
        return flow, blob

    def _logp_grid(
        self, flow, b: float = 4.0, n: int = 100, x_cond=None,
    ) -> np.ndarray:
        """Log-density on a grid.

        The emission is pinned on the conditioning batch: without the pin the
        lift would emit its weights from the plot grid, which is not a batch
        of data and would render a density the model never defines. A no-op
        for the published recipe.
        """
        plogv = float(self.resolved["model"]["plogv"])
        xs = torch.linspace(-b, b, n, device=self.device)
        ys = torch.linspace(-b, b, n, device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        out = []
        with pinned_emission(flow, x_cond):
            with torch.no_grad():
                for i in range(0, grid.shape[0], 2000):
                    out.append(toy_logp(flow, grid[i:i + 2000], plogv).cpu())
        return torch.cat(out).reshape(n, n).numpy()

    def visualize(self) -> None:
        apply_paper_style()
        plot_dir = plotsdir(self.args.experiment)
        os.makedirs(plot_dir, exist_ok=True)
        seeds = self.resolved["seeds"]
        d = self.resolved["data"]
        ref = (self.results or {}).get("target_reference", {})

        x_train, _ = self._data()
        mode = str(self.resolved["lift"].get("eval_condition_on", "train_batch"))
        x_cond = (
            self._eval_cond_all(x_train) if mode == "train_all"
            else self._eval_cond_batch(x_train)
        )
        b = 4.0
        panels = []
        for s in seeds:
            flow, _ = self._load_flow(int(s))
            if flow is None:
                continue
            panels.append(
                (int(s), np.exp(self._logp_grid(flow, b=b, x_cond=x_cond))),
            )

        ncol = 1 + len(panels)
        fig, axes = plt.subplots(1, ncol, figsize=(2.4 * ncol, 2.6))
        axes = np.atleast_1d(axes)
        H, _, _ = np.histogram2d(
            x_train[:, 0], x_train[:, 1], 200, range=[[-b, b], [-b, b]],
        )
        axes[0].imshow(
            H.T, cmap="BuPu", origin="lower", extent=[-b, b, -b, b],
        )
        axes[0].set_title(f"data ({d['target']}, {d['convention']})")
        for ax, (s, p) in zip(axes[1:], panels):
            ax.imshow(p, cmap="BuPu", origin="lower", extent=[-b, b, -b, b])
            ax.set_title(
                f"CP-Flow {self.resolved['backend']} (seed {s})",
            )
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        fig.suptitle(
            f"CP-Flow recipe {self.resolved['recipe']} "
            f"[{self.resolved['backend']}]: "
            f"nblocks={self.resolved['model']['nblocks']}, "
            f"depth={self.resolved['model']['depth']}, "
            f"dimh={self.resolved['model']['dimh']}",
            fontsize=9,
        )
        fig.savefig(os.path.join(plot_dir, "density_grid.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for s in seeds:
            ps = (self.results or {}).get("per_seed", {}).get(str(int(s)))
            if not ps:
                continue
            h = ps["history"]
            # Palette roles: the published recipe is the direct-softplus
            # backend, the lifted one is the hypernet backend. Colors come
            # from paper_style, never picked by hand.
            role = (
                "hypernet" if self.resolved["backend"] == "lifted"
                else "direct_softplus"
            )
            ax.plot(
                h["eval_iter"], h["test_nll"],
                color=palette(role), alpha=0.85, lw=1.2,
                label=(
                    f"CP-Flow ({self.resolved['backend']})"
                    if s == seeds[0] else None
                ),
            )
        if ref:
            ax.axhline(
                ref["entropy_floor_nats"], color=palette("target"),
                ls="--", lw=1.0, label="entropy floor",
            )
            ax.axhline(
                ref["gaussian_moment_matched_nll_nats"], color=palette("gmm"),
                ls=":", lw=1.0, label="moment-matched Gaussian",
            )
        ax.set_xlabel("iteration")
        ax.set_ylabel("test NLL (nats / 2-D sample)")
        ax.set_title(
            f"{d['target']} -- {d['convention']} convention", fontsize=9,
        )
        ax.legend(frameon=False)
        fig.savefig(os.path.join(plot_dir, "test_nll_vs_iter.pdf"))
        plt.close(fig)
        print(f"[visualize] wrote {plot_dir}", flush=True)
