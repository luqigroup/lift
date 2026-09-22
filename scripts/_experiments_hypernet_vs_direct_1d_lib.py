"""Hypernet-vs-direct ICNN comparison on 1-D and 2-D generative targets.

Both backends train a log-concave EBM on the same target under the same
objective and are scored by TV / Hellinger / KL against the target. The
``hypernet`` backend predicts the ICNN's effective weights from the input
batch; the ``direct`` backend stores those weights and applies softplus to
the positivity-tagged ones.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from projorg import checkpointsdir, plotsdir
from torch import nn

from torch.func import functional_call

from lift.dataset import (
    AsymmetricGaussian1D,
    Beta1D,
    BimodalGumbel,
    Gamma1D,
    Gumbel1D,
    HalfNormal1D,
    Laplace1D,
    SkewNormal1D,
)
from lift.dataset.toy2d import (
    AsymmetricLaplace2D,
    GammaMode2D,
    GumbelMixture2D,
    SkewedMode,
)
from lift.dataset.uci import get_uci_target
from lift.models import LogConcaveEBM, icnn_pos_param_names
from lift.objectives.builders import build_hypernet_ebm
from lift.utils.baselines import grid_metrics_2d
from lift.objectives import train_ebm
from lift.objectives._batched import aug_energy_batched
from lift.objectives._param_utils import detach_dict, stack_per_k_params
from lift.objectives.eval import build_eval_grid, hypernet_ebm_logp
from lift.objectives.partition import log_Z_via_grid_1d
from lift.utils.baselines import (
    gmm_fit_logp, gmm_fit_logp_md, grid_metrics_1d,
)


INIT_MODES = ("ours", "hk", "he", "cpflow", "torch")

#: The :data:`INIT_MODES` that give the first (unconstrained) layer LeCun
#: initialization, as Hoedt and Klambauer's own runner does.
_LECUN_FIRST_LAYER_MODES = ("hk", "he")


def apply_init_mode(inner, pos_names, mode: str, latent: bool) -> None:
    """Re-initialize the constrained path of a 1-D ICNN-EBM in place.

    ``latent`` is True for the softplus recipes, whose stored parameter is
    the softplus inverse of the constrained weight, and False for PGD, which
    stores the constrained weight itself.

    * ``ours``   -- leave the constructor's own draw: a folded normal on the
      constrained weight with mean ``1/sqrt(n_in)``, biases zeroed.
    * ``hk``     -- Hoedt and Klambauer (arXiv:2312.12474). With
      ``D = 6(pi-1) + (N-1)(3*sqrt(3) + 2*pi - 6)`` at fan-in ``N`` and their
      rho* = 1/2, their Eq. (11)-(12) give ``mu_w = sqrt(6*pi / (N*D))`` and
      ``sigma_w^2 = 1/N``, and the non-negative weight is drawn log-normal at
      that mean and variance. Their Eq. (8) makes the bias mean negative,
      with magnitude ``sqrt(3N / D)``, to cancel the positive pre-activation
      mean that non-negative weights force. The first layer, which may carry
      negative weights, takes LeCun initialization. Two caveats stated in
      their paper carry over: the constants are derived for a ReLU kernel and
      are applied here to a softplus network, and their analysis excludes the
      skip connections this architecture has.
    * ``he``     -- their ``TraditionalInitialiser(gain = 2.)``: the
      constrained layer's raw weight from ``N(0, sqrt(2 / n_in))``, its bias
      zeroed, and the first layer LeCun as in ``hk``. This is not PyTorch's
      default (``kaiming_uniform_(a = sqrt(5))``, the ``torch`` mode here),
      which is about 2.5 times smaller in standard deviation.
    * ``cpflow`` -- the initialization half of the CP-Flow recipe (Huang et
      al., 2021, ``lib/icnn.py``): the latent weight at PyTorch's default
      ``U(-1/sqrt(n_in), 1/sqrt(n_in))``, biases zeroed, so ``psi'`` starts
      at about one half. CP-Flow pairs that draw with an explicit ``1/fan_in``
      gain on the convex path, which this architecture does not have, so this
      mode reproduces the initialization and not the gain.
    * ``torch``  -- plain ``nn.Linear.reset_parameters`` on every layer,
      constrained and unconstrained alike: no positivity-aware
      initialization at all.
    """
    import math
    if mode == "ours":
        return
    if mode not in INIT_MODES:
        raise ValueError(f"unknown init_mode {mode!r}; expected one of {INIT_MODES}")

    if mode == "torch":
        for m in inner.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        return

    def _hk_moments(n_in: int) -> tuple[float, float, float]:
        """Hoedt-Klambauer Eq. (11)-(12) at rho* = 1/2, and Eq. (8)."""
        d = 6.0 * (math.pi - 1.0) + (n_in - 1) * (
            3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0)
        mu_w = math.sqrt(6.0 * math.pi / (n_in * d))
        var_w = 1.0 / n_in
        mu_b = -math.sqrt(3.0 * n_in / d)            # Eq. (8): negative
        return mu_w, var_w, mu_b

    with torch.no_grad():
        for name, p_ in inner.named_parameters():
            if name not in pos_names:
                continue
            n_in = int(p_.shape[-1])
            if mode == "hk":
                mu_w, var_w, _ = _hk_moments(n_in)
                second = var_w + mu_w * mu_w
                mu_t = math.log(mu_w * mu_w) - 0.5 * math.log(second)
                sd_t = math.sqrt(math.log(second) - math.log(mu_w * mu_w))
                w = torch.exp(mu_t + sd_t * torch.randn_like(p_))
            elif mode == "he":
                # The raw weight is what PGD stores and what softplus
                # takes as its latent, so the draw is the pre-positivity
                # variable either way and is copied in unchanged.
                p_.normal_(0.0, math.sqrt(2.0 / n_in))
                continue
            else:                                    # cpflow: zero-centered latent
                bound = 1.0 / math.sqrt(n_in)
                lat = torch.empty_like(p_).uniform_(-bound, bound)
                if latent:
                    p_.copy_(lat)
                    continue
                w = F.softplus(lat)                  # PGD stores the weight itself
            p_.copy_(torch.log(torch.expm1(w.clamp(min=1e-8))) if latent else w)

        # Biases of the constrained layers. The Hoedt-Klambauer bias is
        # negative and fan-in dependent, so it is read off the same moments
        # rather than filled with a constant.
        for name, p_ in inner.named_parameters():
            if not name.endswith(".bias"):
                continue
            stem = name[: -len("bias")] + "weight"
            if stem not in pos_names:
                continue
            if mode == "hk":
                w_par = dict(inner.named_parameters())[stem]
                p_.fill_(_hk_moments(int(w_par.shape[-1]))[2])
            else:
                p_.zero_()

        # The first layer may carry negative weights and takes LeCun.
        for m in inner.modules():
            if isinstance(m, nn.Linear) \
                    and mode in _LECUN_FIRST_LAYER_MODES \
                    and not any(m.weight is inner.get_parameter(n)
                                for n in pos_names):
                fan_in = int(m.weight.shape[-1])
                nn.init.normal_(m.weight, 0.0, math.sqrt(1.0 / fan_in))
                if m.bias is not None:
                    m.bias.zero_()


class DirectParamModule(nn.Module):
    """Drop-in replacement for ``hyper_E`` that exposes the EBM's own params.

    Holds a ``LogConcaveEBM`` whose positivity-tagged weights are
    materialized in one of four modes set by ``pos_constraint_mode``:

    * ``"softplus"`` (default): raw weights are unconstrained and
      ``forward(x)`` emits ``F.softplus(p)`` for the positivity-tagged
      names. This is the smooth-autodiff direct baseline.
    * ``"pgd"``: raw weights are unconstrained and ``forward(x)`` emits the
      raw ``p``. Positivity is enforced out of band by
      ``project_positive()``, which the trainer calls after each
      ``opt_E.step()``.
    * ``"pgd_floor"``: as ``"pgd"``, but the projection target is
      ``max(w, pos_floor)`` for a small positive ``pos_floor``, so a
      coordinate the projection catches keeps a live gradient instead of
      being absorbed at zero.
    * ``"noise"``: softplus plus isotropic noise on the positivity-tagged
      pre-readout parameters, the matched-noise control.

    ``forward(x)`` returns the same dict shape ``hyper_E`` produces
    (keys = the EBM template's ``named_parameters()``); the ``x``
    argument is ignored.
    """

    def __init__(
        self,
        K: int,
        D: int,
        hidden_dim: int,
        nlayers: int,
        strong_convexity: float = 0.02,
        activation: str = "softplus",
        pos_constraint_mode: str = "softplus",
        component_kind: str = "dense",
        image_shape=None,
        conv_hidden_channels: int = 32,
        conv_kernel_size: int = 3,
        noise_sigma: float = 0.0,
        pos_floor: float = 0.0,
        init_mode: str = "ours",
        inner_module: Optional[nn.Module] = None,
        pos_names: Optional[set[str]] = None,
    ) -> None:
        super().__init__()
        if pos_constraint_mode not in ("softplus", "pgd", "pgd_floor",
                                       "noise"):
            raise ValueError(
                f"DirectParamModule pos_constraint_mode must be "
                f"'softplus', 'pgd', 'pgd_floor' or 'noise', got "
                f"{pos_constraint_mode!r}.",
            )
        # Projection target for ``pgd_floor``; unused in every other mode,
        # and 0.0 there degenerates ``pgd_floor`` back onto plain ``pgd``.
        self.pos_floor = float(pos_floor)
        # Per-iteration N(0, noise_sigma^2) for ``noise`` mode. It is
        # injected only while gradients are enabled, so every no-grad path
        # (TV/loss eval, snapshots) sees the mean iterate.
        self.noise_sigma = float(noise_sigma)
        self.K = K
        self.nlayers = nlayers
        self.pos_constraint_mode = pos_constraint_mode
        # In PGD mode the inner EBM stores raw weights and non-negativity
        # comes from ``project_positive()`` after each Adam step.
        inner_pos_mode = (
            "clamp" if pos_constraint_mode in ("pgd", "pgd_floor")
            else "softplus"
        )
        _kwargs = dict(
            K=K,
            D=D,
            hidden_dim=hidden_dim,
            nlayers=nlayers,
            activation=activation,
            strong_convexity=strong_convexity,
            pos_constraint_mode=inner_pos_mode,
            component_kind=component_kind,
        )
        if component_kind == "conv":
            _kwargs["image_shape"] = image_shape
            _kwargs["conv_hidden_channels"] = int(conv_hidden_channels)
            _kwargs["conv_kernel_size"] = int(conv_kernel_size)
        # ``inner_module`` lets a caller supply the target network instead
        # of having one built as a ``LogConcaveEBM``, so softplus and
        # projection are the same code on any ICNN. It must be given
        # together with ``pos_names``.
        if (inner_module is None) != (pos_names is None):
            raise ValueError(
                "inner_module and pos_names must be given together; got "
                f"inner_module={type(inner_module).__name__}, "
                f"pos_names={pos_names!r}",
            )
        if inner_module is not None:
            self._inner: nn.Module = inner_module
            self._pos_names: set[str] = set(pos_names)
            missing = self._pos_names - {
                n for n, _ in self._inner.named_parameters()
            }
            if missing:
                raise ValueError(
                    f"pos_names absent from inner_module: {sorted(missing)}",
                )
        else:
            self._inner = LogConcaveEBM(**_kwargs)
            self._pos_names = set()
            for k in range(K):
                self._pos_names |= icnn_pos_param_names(
                    nlayers, prefix=f"components.{k}."
                )
        self.init_mode = str(init_mode)
        apply_init_mode(self._inner, self._pos_names, self.init_mode,
                        latent=(pos_constraint_mode != "pgd"
                                and pos_constraint_mode != "pgd_floor"))

    def forward(self, x: torch.Tensor):
        out = {}
        if self.pos_constraint_mode in ("pgd", "pgd_floor"):
            # PGD emits raw weights; ``project_positive()`` enforces
            # non-negativity after the optimizer step.
            for name, p in self._inner.named_parameters():
                out[name] = p
            return out
        inject = (
            self.pos_constraint_mode == "noise"
            and self.noise_sigma > 0
            and torch.is_grad_enabled()
        )
        for name, p in self._inner.named_parameters():
            if name in self._pos_names:
                pre = p
                if inject:
                    pre = p + torch.randn_like(p) * self.noise_sigma
                out[name] = F.softplus(pre)
            else:
                out[name] = p
        return out

    def param_groups(self, lr: float):
        """Adam parameter groups: constrained latents at ``lr * lr_mult``,
        every other weight at ``lr``. The trainer uses this when the module
        exposes it and falls back to a single group otherwise.
        """
        mult = float(getattr(self, "lr_mult", 1.0))
        pos, rest = [], []
        for name, p in self._inner.named_parameters():
            (pos if name in self._pos_names else rest).append(p)
        return [{"params": pos, "lr": float(lr) * mult},
                {"params": rest, "lr": float(lr)}]

    def project_positive(self) -> None:
        """In-place projection of the positivity-tagged raw parameters onto
        the non-negative orthant; a no-op outside the PGD modes.

        Called by the trainer after each ``opt_E.step()``. In ``pgd_floor``
        mode the clamp target is ``self.pos_floor`` rather than zero.
        """
        if self.pos_constraint_mode not in ("pgd", "pgd_floor"):
            return
        lo = 0.0 if self.pos_constraint_mode == "pgd" else self.pos_floor
        with torch.no_grad():
            for name, p in self._inner.named_parameters():
                if name in self._pos_names:
                    p.data.clamp_(min=lo)


class _ConstCodeLift(nn.Module):
    """A lift whose pooled code is pinned to a constant.

    The readout-only rung of the minimal-lift ladder: the readouts, the
    parameter count and the fan-in of the lift it wraps, with no data
    conditioning. ``h_bar`` is the wrapped lift's own pooled summary of one
    batch, taken once at construction; the per-sample encoder is frozen, so
    only the emitter trains.
    """

    def __init__(self, hyper: nn.Module, cond: torch.Tensor) -> None:
        super().__init__()
        self.hyper = hyper
        with torch.no_grad():
            z = hyper.inner_net(cond.reshape(-1, hyper.input_size))
            self.register_buffer("h_bar", z.mean(dim=0, keepdim=True).clone())
        hyper.inner_net.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        del x                       # the control: no data conditioning
        return self.hyper.emit_from_pooled(self.h_bar)

    # The trainer duck-types these on the emitter.
    def set_noise_scale(self, scale: float) -> None:
        self.hyper.set_noise_scale(scale)

    def capture_pre_positivity(self):
        return self.hyper.capture_pre_positivity()


_FLOOR_BACKEND_RE = re.compile(r"^pgd_floor(\d+)$")


def _floor_of(name: str) -> float | None:
    """``pgd_floor<E>`` -> the floor ``10^-E``; anything else -> ``None``.

    Under the floored projection ``max(w, eps)`` no coordinate reaches an
    exact zero, so every unit keeps a live gradient. The exponent rides in
    the backend name rather than in a config key.
    """
    m = _FLOOR_BACKEND_RE.match(str(name))
    return None if m is None else 10.0 ** (-int(m.group(1)))


_DIRECT_MULT_RE = re.compile(r"^direct_m([0-9.]+)$")


def _direct_mult(name: str) -> float | None:
    """``direct_m<M>`` -> the per-group multiplier ``M``; else ``None``.

    Direct softplus whose constrained latents alone take an Adam step of
    ``M * lr`` while every unconstrained weight keeps ``lr``. It matches the
    lift, where for a fixed code the emitted latent moves by
    ``lr * (1 + ||z||_1)`` per step and that multiplier reaches the emitted,
    hence constrained, weights only. Raising the global learning rate would
    also over-step the unconstrained body.
    """
    m = _DIRECT_MULT_RE.match(str(name))
    return None if m is None else float(m.group(1))


class _GainCodeLift(nn.Module):
    """The moments lift with its batch fluctuation as a dial.

    The code is ``z = z_bar + gamma * (phi(X) - z_bar) / s``. The population
    moments ``z_bar`` and their batch-to-batch spread ``s`` are taken once
    from a large draw, so the drift multiplier ``1 + ||z_bar||_1`` and the
    initial emitted weights stay fixed across the dial and ``gamma`` sets the
    standardized size of the batch fluctuation alone. ``gamma = 0`` is the
    readout-only control at the moments' own width; ``gamma = s`` is the raw
    moments rung.
    """

    def __init__(self, hyper: nn.Module, cond_big: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.hyper = hyper
        self.gamma = float(gamma)
        n_batch = None
        with torch.no_grad():
            z = hyper.inner_net(cond_big.reshape(-1, hyper.input_size))
            self.register_buffer("z_bar", z.mean(dim=0, keepdim=True).clone())
            # the spread of a 256-sample batch mean, per feature
            self.register_buffer("z_sd", (z.std(dim=0, keepdim=True) / 16.0).clone() + 1e-8)
        hyper.inner_net.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        z = self.hyper.inner_net(x.reshape(-1, self.hyper.input_size)).mean(dim=0, keepdim=True)
        return self.hyper.emit_from_pooled(self.z_bar + self.gamma * (z - self.z_bar) / self.z_sd)

    def set_noise_scale(self, scale: float) -> None:
        self.hyper.set_noise_scale(scale)

    def capture_pre_positivity(self):
        return self.hyper.capture_pre_positivity()


_LIFT_BACKEND_RE = re.compile(
    r"^lift_(?:(moments|pn|constpn|constmom)|(tiny|const)(\d+)?|(momg)(\d+e\d+|0))(_fs)?"
    r"(?:_(c|ind)(\d+))?$"
)


def _parse_lift_backend(name: str) -> tuple[str, float, bool] | None:
    """The ladder's rung names. Returns ``(kind, param, fanin_scale)``.

    ``lift_moments``: ``b + A[m1(X), m2(X)]``, the parameter-free summary.
    ``lift_tiny<K>`` / ``lift_const<K>``: one hidden layer of width K,
    mean-pooled, live or frozen to one batch (default K = 8).
    ``lift_pn`` / ``lift_constpn``: ``pool_norm`` on at 64,64,96, live or
    frozen; note that the plain ``hypernet`` backend of this driver is the
    unstandardized lift. ``lift_constmom``: the moments readout with the
    code fixed at the population moments. ``lift_momg<G>``: the moments lift
    with a standardized fluctuation of size G (``momg2e3`` = 2e-3,
    ``momg0`` = constmom). ``_fs`` sets ``readout_fanin_scale``. The width
    and the gain ride in the name rather than in a config key.
    """
    m = _LIFT_BACKEND_RE.match(str(name))
    if m is None:
        return None
    if m.group(1):
        return m.group(1), 0.0, bool(m.group(6))
    if m.group(2):
        return m.group(2), float(m.group(3) or 8), bool(m.group(6))
    g = m.group(5)
    gamma = 0.0 if g == "0" else float(g.replace("e", "e-"))
    return "momg", gamma, bool(m.group(6))


def _cond_of(name: str):
    """The conditioning-batch sweep: ``_c<N>`` / ``_ind<N>``.

    ``_c<N>``: the body reads the first N samples of the loss batch, so the
    code's batch-to-batch fluctuation scales as 1/sqrt(N) while the gradient
    noise is untouched; the frozen rung is the N -> infinity limit.
    ``_ind<N>``: N samples drawn independently of the loss batch, which is
    jitter without coupling. Returns ``(cond_source, n_batch_cond)`` or
    ``None``.
    """
    m = _LIFT_BACKEND_RE.match(str(name))
    if m is None or m.group(7) is None:
        return None
    return ("subbatch" if m.group(7) == "c" else "independent"), int(m.group(8))


def _train_kwargs(args, *, exclude_strong_conv: bool = False) -> dict:
    base = dict(
        n_iters=int(args.n_iters),
        lr_ebm=float(args.lr_ebm),
        lr_sampler=float(args.lr_sampler),
        n_batch_hyper=int(args.n_batch_hyper),
        n_batch_val=int(args.n_batch_val),
        eval_every=int(args.eval_every),
        grad_clip=float(args.grad_clip),
        tv_grid_n=int(args.tv_grid_n),
        hidden_dim=int(args.hidden_dim),
        nlayers=int(args.nlayers),
        strong_convexity=float(args.strong_convexity),
        hyper_hidden_sizes=[int(h) for h in args.hyper_hidden_sizes],
        sampler_hyper_hidden_sizes=[
            int(h) for h in args.sampler_hyper_hidden_sizes
        ],
        flow_D_aug=int(args.flow_D_aug),
        flow_n_hidden=int(args.flow_n_hidden),
        flow_n_layers=int(args.flow_n_layers),
        flow_n_mlp_layers=int(args.flow_n_mlp_layers),
        T_sampler=int(args.T_sampler),
        eta_svgd=float(args.eta_svgd),
        M_particles=int(args.M_particles),
        M_logZ=int(args.M_logZ),
    )
    for k in ("T_sampler_max", "sampler_kl_threshold", "M_kl_check",
              "M_logZ_eval", "sampler_objective",
              "n_sampler_warmup_iters", "n_ebm_warmup_iters",
              "n_cooldown_iters", "cond_dim",
              "n_batch_cond", "cond_source",
              "abort_loss_abs", "abort_patience", "abort_after",
              "sampler_lr_ramp", "ramp_sampler_T_max"):
        v = getattr(args, k, None)
        if v is None:
            continue
        if k in ("sampler_objective", "cond_source"):
            base[k] = str(v)
        elif k in ("sampler_kl_threshold", "abort_loss_abs"):
            base[k] = float(v)
        else:
            base[k] = int(v)
    # Extra HyperNetwork kwargs and the trainer's activation-noise knob,
    # passed only when the driver set them.
    extra = getattr(args, "hyper_extra_kwargs", None)
    if extra:
        base["hyper_extra_kwargs"] = dict(extra)
    noise = float(getattr(args, "hyper_noise_std", 0.0) or 0.0)
    if noise > 0.0:
        base["hyper_activation_noise_std"] = noise
    return base


def _build_2d_grid(target, n_per_axis, device):
    (lo_x, hi_x), (lo_y, hi_y) = target.xlim
    xs = torch.linspace(lo_x, hi_x, n_per_axis, device=device)
    ys = torch.linspace(lo_y, hi_y, n_per_axis, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    dvol = float((hi_x - lo_x) / (n_per_axis - 1)
                 * (hi_y - lo_y) / (n_per_axis - 1))
    return grid, dvol


class HypernetVsDirect1D:
    def __init__(self, args, target_kind: str = "gumbel"):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.target_kind = target_kind
        self.K = int(getattr(args, "K", 1))
        self.objective = str(getattr(args, "objective", "fkl-direct"))
        self.results: dict = {}

    def _torch_logp_to_numpy(self, target):
        device = self.device

        def _np_logp(x_np):
            x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float64)).to(
                device=device, dtype=torch.float32,
            )
            with torch.no_grad():
                return target.log_prob(x_t).detach().cpu().numpy().astype(
                    np.float64
                )

        return _np_logp

    def _eval_density(self, ebm, hyper_E, flow, target):
        K = int(ebm.K)
        D = int(getattr(target, "D", 1))
        D_aug = int(self.args.flow_D_aug)
        n_grid = int(self.args.tv_grid_n)
        with torch.no_grad():
            x_h = target.sample(
                int(self.args.n_batch_hyper), device=self.device
            ).detach()
            params_E_eval = hyper_E(x_h)
            log_Z = None
            if D == 1:
                xs, _, dvol = build_eval_grid(target, n_grid, self.device)
                log_Z = log_Z_via_grid_1d(
                    ebm, params_E_eval, K=K,
                    grid=xs, dvol=float(dvol),
                )
            elif D == 2:
                n_axis = max(int(np.sqrt(n_grid)), 96)
                xs, dvol = _build_2d_grid(target, n_axis, self.device)
                log_Z = log_Z_via_grid_1d(
                    ebm, params_E_eval, K=K,
                    grid=xs, dvol=float(dvol),
                )
            # D >= 3: leave log_Z=None so hypernet_ebm_logp uses flow-IS.
        M_logZ_eval = int(
            getattr(self.args, "M_logZ_eval", None) or self.args.M_logZ
        )
        return hypernet_ebm_logp(
            ebm, hyper_E, flow, K, D, D_aug,
            target_sample=x_h, M_logZ=M_logZ_eval, device=self.device,
            log_Z_override=log_Z,
        )

    def _target_label(self):
        m = {
            "gumbel": "1-D Gumbel",
            "bimodal_gumbel": "1-D bimodal Gumbel",
            "laplace": "1-D Laplace",
            "laplace_wide": "1-D Laplace (b=5)",
            "gamma_wide": "1-D Gamma (rate 0.2)",
            "gamma": "1-D Gamma",
            "beta": "1-D Beta",
            "halfnormal": "1-D half-normal",
            "skewnormal": "1-D skew-normal",
            "asym-gaussian": "1-D asymmetric Gaussian",
            "gumbel_mix_2d": "2-D Gumbel-Gauss mixture",
            "skewed_2d": "2-D skewed-mode mixture",
            "gamma_mode_2d": "2-D Gamma-Gauss single mode",
            "asymmetric_laplace_2d": "2-D asymmetric-Laplace",
        }
        return f"K={self.K} {m.get(self.target_kind, self.target_kind)}"

    def _build_target(self):
        if self.target_kind == "gumbel":
            return Gumbel1D(mu=0.0)
        if self.target_kind == "laplace":
            return Laplace1D(
                mu=float(getattr(self.args, "laplace_mu", 0.0)),
                b=float(getattr(self.args, "laplace_b", 1.0)),
            )
        # Wide variants: a target wide relative to the initial model pushes
        # every output weight down.
        if self.target_kind == "laplace_wide":
            return Laplace1D(mu=0.0, b=5.0)
        if self.target_kind == "gamma_wide":
            return Gamma1D(alpha=2.0, beta=0.2)
        if self.target_kind == "gamma":
            return Gamma1D(
                alpha=float(getattr(self.args, "gamma_alpha", 2.0)),
                beta=float(getattr(self.args, "gamma_beta", 1.0)),
            )
        if self.target_kind == "beta":
            return Beta1D(
                alpha=float(getattr(self.args, "beta_alpha", 2.0)),
                beta=float(getattr(self.args, "beta_beta", 5.0)),
            )
        if self.target_kind == "halfnormal":
            return HalfNormal1D(
                sigma=float(getattr(self.args, "halfnormal_sigma", 1.0)),
            )
        if self.target_kind == "skewnormal":
            return SkewNormal1D(
                xi=float(getattr(self.args, "skewnormal_xi", 0.0)),
                omega=float(getattr(self.args, "skewnormal_omega", 1.0)),
                alpha=float(getattr(self.args, "skewnormal_alpha", 4.0)),
            )
        if self.target_kind == "asym-gaussian":
            # Piecewise-quadratic log-density with left/right curvature
            # ratio ``delta = c_- / c_+``; ``delta = 1`` recovers the
            # standard normal.
            return AsymmetricGaussian1D(
                delta=float(getattr(self.args, "asym_delta", 1.0)),
                c_plus=float(getattr(self.args, "asym_c_plus", 1.0)),
            )
        if self.target_kind == "bimodal_gumbel":
            return BimodalGumbel(
                centre=float(getattr(self.args, "gumbel_centre", 4.0)),
                weight=float(getattr(self.args, "gumbel_weight", 0.5)),
                xlim=(-5.0, 12.0),
            )
        if self.target_kind == "gumbel_mix_2d":
            return GumbelMixture2D(
                separation=float(getattr(self.args, "separation_2d", 3.0)),
                sigma_y=float(getattr(self.args, "sigma_y", 0.4)),
                weight=float(getattr(self.args, "gumbel_weight", 0.5)),
            )
        if self.target_kind == "skewed_2d":
            return SkewedMode(
                separation=float(getattr(self.args, "separation_2d", 3.0)),
                sigma=float(getattr(self.args, "sigma_skewed_2d", 0.6)),
            )
        if self.target_kind == "gamma_mode_2d":
            return GammaMode2D(
                shape=float(getattr(self.args, "gamma_shape", 2.0)),
                rate=float(getattr(self.args, "gamma_rate", 1.0)),
                origin_shift=float(getattr(self.args, "gamma_shift", -3.0)),
                sigma_y=float(getattr(self.args, "sigma_y", 0.5)),
            )
        if self.target_kind == "asymmetric_laplace_2d":
            # ``separation=0`` collapses the two-mode mixture to a single
            # log-concave mode at the origin: the x-marginal is an
            # asymmetric Laplace (different exponential decay rates for
            # x < 0 and x >= 0) and the y-marginal is Gaussian.
            return AsymmetricLaplace2D(
                alpha=float(getattr(self.args, "al_alpha", 4.0)),
                beta=float(getattr(self.args, "al_beta", 1.0)),
                separation=float(getattr(self.args, "separation_2d", 0.0)),
                sigma_y=float(getattr(self.args, "sigma_y", 0.4)),
            )
        if self.target_kind.startswith("uci_"):
            ds = self.target_kind[len("uci_"):]
            data_path = str(getattr(self.args, "data_path", "")) or None
            return get_uci_target(
                ds,
                path=data_path,
                max_train_rows=int(getattr(self.args, "max_train_rows", 0)),
            )
        if self.target_kind == "mnist_latent":
            from lift.dataset.mnist_latent import MNISTLatent
            ae_path = getattr(self.args, "ae_path", "") or None
            return MNISTLatent(
                ae_path=ae_path,
                mnist_root=str(getattr(
                    self.args, "mnist_root", "data/datasets/mnist",
                )),
                latent_dim=int(getattr(self.args, "latent_dim", 32)),
                hidden=int(getattr(self.args, "ae_hidden", 64)),
                device=self.device,
            )
        if self.target_kind == "mnist_latent_class":
            # Per-digit-class MNIST AE-latent: ``args.digit_class`` in
            # 0..9 selects the class-conditional latent target.
            from lift.dataset.mnist_latent import (
                MNISTLatentClassConditional,
            )
            ae_path = getattr(self.args, "ae_path", "") or None
            digit_class = int(getattr(self.args, "digit_class", 0))
            return MNISTLatentClassConditional(
                digit_class=digit_class,
                ae_path=ae_path,
                latent_dim=int(getattr(self.args, "latent_dim", 32)),
                hidden=int(getattr(self.args, "ae_hidden", 64)),
                device=self.device,
            )
        raise ValueError(f"unknown target_kind {self.target_kind!r}")

    def _target_D(self) -> int:
        return int(getattr(self._build_target(), "D", 1))

    def _train_one(self, backend: str, target, seed: int):
        kwargs = _train_kwargs(self.args)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if (backend in ("direct", "pgd", "direct_noise")
                or _floor_of(backend) is not None
                or _direct_mult(backend) is not None):
            _mode = {"pgd": "pgd", "direct_noise": "noise"}.get(
                backend, "softplus")
            if _floor_of(backend) is not None:
                _mode = "pgd_floor"
            direct_hyper_E = DirectParamModule(
                K=self.K, D=int(getattr(target, "D", 1)),
                hidden_dim=int(self.args.hidden_dim),
                nlayers=int(self.args.nlayers),
                strong_convexity=float(self.args.strong_convexity),
                pos_constraint_mode=_mode,
                noise_sigma=float(getattr(self.args, "noise_sigma", 0.0)),
                pos_floor=float(_floor_of(backend) or 0.0),
                init_mode=str(getattr(self.args, "init_mode", "ours")),
            )
            tmpl_ebm, _hyper_unused = build_hypernet_ebm(
                K=self.K, D=int(getattr(target, "D", 1)),
                hidden_dim=int(self.args.hidden_dim),
                nlayers=int(self.args.nlayers),
                strong_convexity=float(self.args.strong_convexity),
            )
            _mult = _direct_mult(backend)
            if _mult is not None:
                direct_hyper_E.lr_mult = float(_mult)
            kwargs["ebm_and_hyper_E"] = (tmpl_ebm, direct_hyper_E)
        elif (lift := _parse_lift_backend(backend)) is not None:
            # The minimal-lift ladder. Every rung is the same
            # HyperNetwork with a different summary, handed to the trainer
            # the way the direct backend is, so the loop, the optimizer and
            # the snapshots are shared.
            kind, param, fanin_scale = lift
            D = int(getattr(target, "D", 1))
            moments = kind in ("moments", "constmom", "momg")
            paper = kind in ("pn", "constpn")
            extra = {}
            if fanin_scale:
                extra["readout_fanin_scale"] = True
            if paper:
                extra["pool_norm"] = True     # pooled-code standardization
            tmpl_ebm, mini_hyper = build_hypernet_ebm(
                K=self.K, D=D,
                hidden_dim=int(self.args.hidden_dim),
                nlayers=int(self.args.nlayers),
                strong_convexity=float(self.args.strong_convexity),
                hyper_hidden_sizes=(
                    [2 * D] if moments
                    else [int(h) for h in self.args.hyper_hidden_sizes] if paper
                    else [int(param)]
                ),
                hyper_summary_kind="moments" if moments else "mlp",
                hyper_extra_kwargs=extra or None,
            )
            mini_hyper = mini_hyper.to(self.device)
            if kind in ("const", "constpn"):
                # Frozen to one batch's summary: the readout-only control
                # of the rung above it at the same width. The batch is
                # ``snap_x`` below, so the frozen and live rungs emit the
                # same weights at iterate zero.
                self._pending_const = ("batch", 0.0)
            elif kind in ("constmom", "momg"):
                # Population moments from a large draw, plus the
                # fluctuation dial (gamma = 0 for constmom).
                self._pending_const = ("population", float(param))
            else:
                self._pending_const = None
            kwargs["ebm_and_hyper_E"] = (tmpl_ebm, mini_hyper)
            cond = _cond_of(backend)
            if cond is not None:
                kwargs["cond_source"], kwargs["n_batch_cond"] = cond

        # One data stream for every backend. Construction consumes a
        # backend-dependent amount of RNG, so without these reseeds the
        # snapshot batch, the flow initialization, the validation set and
        # every training batch differ across backends at the same seed and
        # paired seeds are not paired in the data.
        torch.manual_seed(int(seed) + 1_000_003)
        snap_x = target.sample(int(self.args.n_batch_hyper),
                               device=self.device).detach()
        pending = getattr(self, "_pending_const", None)
        if pending is not None:
            mode, gamma = pending
            tmpl_ebm, mini_hyper = kwargs["ebm_and_hyper_E"]
            if mode == "batch":
                mini_hyper = _ConstCodeLift(mini_hyper, snap_x)
            else:
                cond_big = target.sample(1 << 16, device=self.device).detach()
                mini_hyper = _GainCodeLift(mini_hyper, cond_big, gamma)
            kwargs["ebm_and_hyper_E"] = (tmpl_ebm, mini_hyper.to(self.device))
            self._pending_const = None
        torch.manual_seed(int(seed) + 2_000_003)
        theta_snaps: list = []
        snap_iters: list = []
        hyper_E_snaps: list = []

        # ``--snap_every`` subsamples the per-eval snapshot stride; 0 snaps
        # at every eval. The iteration-0 and final snapshots are kept either
        # way, so the landscape renderer always has both endpoints.
        snap_every = int(getattr(self.args, "snap_every", 0) or 0)

        def _snap(it, ebm, hyper_E, flow, **_kw):
            it_int = int(it)
            # Keep iteration 0 so the trajectory's start point is defined.
            if snap_every > 0 and it_int > 0 and (it_int % snap_every) != 0:
                return
            with torch.no_grad():
                params_E = hyper_E(snap_x)
            theta_snaps.append(
                {k: v.detach().cpu().clone() for k, v in params_E.items()}
            )
            # The emitter's full state dict per snapshot is what the
            # lifted-space landscape renderers consume, and it is enormous
            # whenever the emitted parameter vector is large: on a 512-wide
            # five-layer ICNN the emitter maps its last hidden layer onto
            # roughly a million constrained weights, which is hundreds of
            # megabytes per snapshot. Callers that need only the test loss
            # and ``theta_snaps`` should pass ``--save_hyper_snaps 0``.
            if bool(int(getattr(self.args, "save_hyper_snaps", 1))):
                hyper_E_snaps.append(
                    {
                        k: v.detach().cpu().clone()
                        for k, v in hyper_E.state_dict().items()
                    }
                )
            snap_iters.append(it_int)

        kwargs["eval_callback"] = _snap
        objective = str(getattr(self.args, "objective", "fkl-direct"))
        out = train_ebm(
            target=target, K=self.K, objective=objective,
            device=self.device, progress=True,
            **kwargs,
        )
        ebm_out, hyper_E_out, _flow_out, _hist_out = out
        with torch.no_grad():
            params_E_final = hyper_E_out(snap_x)
        theta_snaps.append(
            {k: v.detach().cpu().clone() for k, v in params_E_final.items()}
        )
        hyper_E_snaps.append(
            {
                k: v.detach().cpu().clone()
                for k, v in hyper_E_out.state_dict().items()
            }
        )
        snap_iters.append(int(self.args.n_iters))
        return out + (theta_snaps, snap_iters, snap_x, hyper_E_snaps)

    def train(self):
        ckpt_dir = checkpointsdir(self.args.experiment)
        os.makedirs(ckpt_dir, exist_ok=True)
        target = self._build_target()
        D = int(getattr(target, "D", 1))
        self.D = D
        n_per_axis = int(self.args.tv_grid_n)
        if D == 1:
            n_grid = n_per_axis
            x_grid_np = np.linspace(target.xlim[0], target.xlim[1], n_grid)
            target_lp_grid = target.log_prob_np(x_grid_np)
            target_logp_np = target.log_prob_np
        elif D == 2:
            n_grid = max(int(np.sqrt(n_per_axis)), 96)
            (lo_x, hi_x), (lo_y, hi_y) = target.xlim
            xs = np.linspace(lo_x, hi_x, n_grid)
            ys = np.linspace(lo_y, hi_y, n_grid)
            gx, gy = np.meshgrid(xs, ys, indexing="xy")
            x_grid_np = np.stack([gx.flatten(), gy.flatten()], axis=-1)
            target_logp_np = self._torch_logp_to_numpy(target)
            target_lp_grid = target_logp_np(x_grid_np)
        else:
            # D >= 3: no density-grid, evaluate via held-out test NLL.
            x_grid_np = np.zeros((1, D), dtype=np.float32)  # placeholder
            target_lp_grid = np.zeros(1, dtype=np.float32)
            target_logp_np = None
            n_grid = 1
        np.save(os.path.join(ckpt_dir, "x_grid.npy"), x_grid_np)
        np.save(os.path.join(ckpt_dir, "target_lp_grid.npy"), target_lp_grid)

        # GMM-EM matched-K reference baseline (seed-independent).
        x_train_gmm = target.sample(
            8192, device=self.device
        ).detach().cpu().numpy()
        if D == 1:
            gmm_logp_fn = gmm_fit_logp(x_train_gmm.squeeze(), K=self.K)
            gmm_metrics = grid_metrics_1d(
                target.log_prob_np, gmm_logp_fn, *target.xlim, n=n_grid,
            )
        elif D == 2:
            gmm_logp_fn = gmm_fit_logp_md(x_train_gmm, K=self.K)
            gmm_metrics = grid_metrics_2d(
                target_logp_np, gmm_logp_fn, target.xlim,
                n_per_axis=n_grid,
            )
        else:
            cov_type = "full" if D <= 16 else "diag"
            gmm_logp_fn = gmm_fit_logp_md(
                x_train_gmm, K=self.K, covariance_type=cov_type,
            )
            test_data = np.asarray(target.test_data, dtype=np.float64)
            gmm_test_nll = float(-np.asarray(gmm_logp_fn(test_data)).mean())
            gmm_metrics = {"test_nll": gmm_test_nll}
        gmm_lp_grid = gmm_logp_fn(x_grid_np) if D <= 2 else np.zeros(1)
        np.save(os.path.join(ckpt_dir, "gmm_lp_grid.npy"), gmm_lp_grid)
        gmm_key = f"gmm_em_K{self.K}"
        all_results = {
            gmm_key: {k: float(v) for k, v in gmm_metrics.items()},
            "per_seed": {},
        }

        seeds = [int(s) for s in getattr(self.args, "seeds", [self.args.seed])]
        for s in seeds:
            print(f"\n############ SEED {s} ############")
            torch.manual_seed(s + 17)
            x_val_landscape = target.sample(
                int(self.args.n_batch_val), device=self.device
            ).detach()
            np.save(
                os.path.join(ckpt_dir, f"seed{s}_x_val_landscape.npy"),
                x_val_landscape.detach().cpu().numpy(),
            )
            backend_results = {}
            seed_metrics = {}
            # ``backends`` restricts the per-seed run to a subset of
            # {hypernet, direct, pgd}; the default is hypernet + direct.
            backends = tuple(
                str(b) for b in getattr(self.args, "backends",
                                        ["hypernet", "direct"])
            )
            for backend in backends:
                if (
                    backend not in ("hypernet", "direct", "pgd",
                                    "direct_noise")
                    and _parse_lift_backend(backend) is None
                    and _floor_of(backend) is None
                    and _direct_mult(backend) is None
                ):
                    raise ValueError(
                        f"unknown backend {backend!r}; expected one of "
                        "{'hypernet', 'direct', 'pgd', 'direct_noise'}, "
                        "pgd_floor<E>, or lift_moments | lift_tiny<K> | "
                        "lift_const<K> (optional _fs suffix)",
                    )
                # ``--resume 1`` skips any backend whose checkpoint is
                # already on disk. Checkpoints are saved one backend at a
                # time, so an interrupted sweep loses only the backend that
                # was in flight. The shared landscape figure needs the live
                # hypernet and direct objects, so a seed with any resumed
                # backend skips that figure; metrics and the emitted-weight
                # trajectory are read back from the checkpoint either way.
                arm_path = os.path.join(ckpt_dir, f"seed{s}_{backend}.pt")
                if (
                    bool(int(getattr(self.args, "resume", 0)))
                    and os.path.isfile(arm_path)
                ):
                    prev = torch.load(
                        arm_path, map_location="cpu", weights_only=False,
                    )
                    prev_metrics = prev.get("metrics", {}) or {}
                    seed_metrics[backend] = {
                        k: float(v) for k, v in prev_metrics.items()
                    }
                    # Older checkpoints carry no divergence field, so
                    # recompute it from the saved history; otherwise a
                    # resumed sweep reports every run as having survived.
                    _le = np.asarray(
                        (prev.get("history", {}) or {}).get(
                            "loss_E_per_iter", []),
                        dtype=float,
                    )
                    _bad = np.flatnonzero(~np.isfinite(_le))
                    seed_metrics[backend]["diverged_at_iter"] = (
                        float(_bad[0] + 1) if _bad.size else float("nan")
                    )
                    print(
                        f"[seed {s} | {backend}] resumed from {arm_path}: "
                        + "  ".join(
                            f"{k}={v:.4g}"
                            for k, v in seed_metrics[backend].items()
                        )
                    )
                    del prev
                    continue

                print(f"\n=== seed {s} | backend={backend} ===")
                (
                    ebm, hyper_E, flow, history,
                    theta_snaps, snap_iters, _snap_x, hyper_E_snaps,
                ) = self._train_one(backend, target, seed=s)
                backend_results[backend] = {
                    "ebm": ebm,
                    "hyper_E": hyper_E,
                    "flow": flow,
                    "theta_snaps": theta_snaps,
                    "snap_iters": snap_iters,
                    "hyper_E_snaps": hyper_E_snaps,
                    "snap_x": _snap_x,
                }
                logp_fn = self._eval_density(ebm, hyper_E, flow, target)
                if D == 1:
                    metrics = grid_metrics_1d(
                        target.log_prob_np, logp_fn,
                        *target.xlim, n=n_grid,
                    )
                elif D == 2:
                    metrics = grid_metrics_2d(
                        target_logp_np, logp_fn, target.xlim,
                        n_per_axis=n_grid,
                    )
                else:
                    test_data = np.asarray(target.test_data, dtype=np.float64)
                    test_nll = float(-np.asarray(logp_fn(test_data)).mean())
                    metrics = {"test_nll": test_nll}
                log_p_grid = (
                    logp_fn(x_grid_np) if D <= 2
                    else np.zeros(1, dtype=np.float32)
                )
                torch.save(
                    {
                        "ebm_state": ebm.state_dict(),
                        "hyper_E_state": hyper_E.state_dict(),
                        "flow_state": flow.state_dict(),
                        "history": history,
                        "log_p_grid": log_p_grid,
                        "metrics": metrics,
                        "theta_snaps": theta_snaps,
                        "snap_iters": snap_iters,
                        "hyper_E_snaps": hyper_E_snaps,
                        "snap_x": _snap_x.detach().cpu().clone(),
                        "seed": s,
                    },
                    os.path.join(ckpt_dir, f"seed{s}_{backend}.pt"),
                )
                seed_metrics[backend] = {
                    k: float(v) for k, v in metrics.items()
                }
                # The trainer skips the update on a non-finite energy loss
                # and records a NaN for that iteration, so a run that blows
                # up finishes quietly with a frozen model rather than
                # raising. Carry the first non-finite iteration into
                # metrics.json, which travels when the checkpoint does not.
                _le = np.asarray(
                    history.get("loss_E_per_iter", []), dtype=float,
                )
                _bad = np.flatnonzero(~np.isfinite(_le))
                # The trainer's dead-run guard counts as divergence too:
                # finite but absurd loss, frozen weights.
                _ab = history.get("aborted_at_iter")
                _div = [float(_bad[0] + 1)] if _bad.size else []
                if _ab is not None:
                    _div.append(float(_ab))
                seed_metrics[backend]["diverged_at_iter"] = (
                    min(_div) if _div else float("nan")
                )
                print(f"[seed {s} | {backend}] " + "  ".join(
                    f"{k}={v:.4g}" for k, v in metrics.items()
                ) + (
                    f"  DIVERGED at it={int(min(_div))}"
                    + ("  (ABORTED)" if _ab is not None else "")
                    if _div else ""
                ))
            # The shared landscape figure needs both the hypernet and the
            # direct trajectory, so skip it when the backend selection omits
            # either, or when the caller sets ``landscape_grid_n <= 1`` or
            # ``skip_landscape``.
            _skip_ls = bool(getattr(self.args, "skip_landscape", 0)) or (
                int(getattr(self.args, "landscape_grid_n", 41)) <= 1
            ) or any(                       # a diverged run has no landscape worth the GPU time
                np.isfinite(float(seed_metrics.get(b, {}).get("diverged_at_iter", float("nan"))))
                for b in ("hypernet", "direct")
            )
            if (
                "hypernet" in backend_results
                and "direct" in backend_results
                and not _skip_ls
            ):
                print(f"\n=== seed {s} | building shared landscape ===")
                self._build_landscape_1d(
                    ckpt_dir, backend_results, target, x_val_landscape,
                    seed=s,
                )
            all_results["per_seed"][str(s)] = seed_metrics
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        self.results = all_results
        print(f"\n[GMM-EM K={self.K}] " + "  ".join(
            f"{k}={v:.4g}" for k, v in gmm_metrics.items()
        ))

    # ----------------------------------------------------- landscape (1-D)
    def _filter_norm_dir(self, theta_dict):
        d = {}
        for name, w in theta_dict.items():
            r = torch.randn_like(w)
            if w.ndim >= 2 and w.size(0) > 1:
                n = w.size(0)
                rf = r.view(n, -1)
                wf = w.view(n, -1)
                rn = rf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                wn = wf.norm(dim=-1, keepdim=True)
                rf = rf * (wn / rn)
                d[name] = rf.view_as(w)
            else:
                norm_r = r.norm().clamp(min=1e-8)
                d[name] = r * (w.norm() / norm_r)
        return d

    def _diff_dict(self, a, b):
        return {k: a[k] - b[k] for k in a}

    def _scale_dict(self, a, s: float):
        return {k: v * s for k, v in a.items()}

    def _add_dict(self, a, b, sb: float = 1.0):
        return {k: a[k] + sb * b[k] for k in a}

    def _dot_dict(self, a, b):
        return float(sum((a[k] * b[k]).sum() for k in a))

    def _gs_orthogonalise(self, a, basis_dirs):
        for d_b in basis_dirs:
            denom = max(self._dot_dict(d_b, d_b), 1e-20)
            coef = self._dot_dict(a, d_b) / denom
            a = self._add_dict(a, d_b, sb=-coef)
        return a

    def _trajectory_pca_dirs(self, traj_h, traj_d, center, device):
        """Return (d1, d2) so direct's final lands at (1, 0) and the
        second axis captures the trajectories' largest motion orthogonal
        to d1.

        d1 = (direct_final - hypernet_final) normalized so that
             direct_final's projection coordinate equals 1.
        d2 = top SVD direction of all (theta_t - center) snapshots after
             Gram-Schmidt removal of d1, normalized similarly.
        """
        device = device
        keys = list(center.keys())

        def _to_dev(d):
            return {k: v.to(device) for k, v in d.items()}

        traj_h = [_to_dev(t) for t in traj_h]
        traj_d = [_to_dev(t) for t in traj_d]
        d_final = traj_d[-1]
        # d1 = direct_final - center, which makes the projection
        # coefficient of direct's final snapshot exactly 1.
        d1 = self._diff_dict(d_final, center)
        # Flatten the snapshots, remove d1 by Gram-Schmidt, SVD the rest.
        all_snaps = traj_h + traj_d
        diffs = [self._diff_dict(t, center) for t in all_snaps]
        diffs_perp = [self._gs_orthogonalise(d, [d1]) for d in diffs]
        flat = torch.stack([self._flatten(d, keys) for d in diffs_perp])  # (T, |theta|)
        try:
            _, S, Vh = torch.linalg.svd(flat, full_matrices=False)
            v_top = Vh[0]  # top right-singular vector
            d2 = {}
            o = 0
            for k in keys:
                n = center[k].numel()
                d2[k] = v_top[o : o + n].view_as(center[k])
                o += n
            # Rescale so the largest projection across snapshots is 1.
            projs = [self._dot_dict(d, d2) / max(self._dot_dict(d2, d2), 1e-20)
                     for d in diffs_perp]
            scale = max(abs(min(projs)), abs(max(projs)), 1e-12)
            d2 = self._scale_dict(d2, scale)
        except Exception:
            d2 = self._filter_norm_dir(center)
            d2 = self._gs_orthogonalise(d2, [d1])
        return d1, d2

    def _flatten(self, dct, keys):
        return torch.cat([dct[k].reshape(-1) for k in keys])

    def _project_traj(self, snaps, theta_star, d1, d2, device):
        keys = list(theta_star.keys())
        v_star = self._flatten(theta_star, keys).to(device)
        v_d1 = self._flatten(d1, keys).to(device)
        v_d2 = self._flatten(d2, keys).to(device)
        D_mat = torch.stack([v_d1, v_d2], dim=1)
        A = D_mat.T @ D_mat
        coords = []
        for theta in snaps:
            v = self._flatten(theta, keys).to(device)
            rhs = D_mat.T @ (v - v_star)
            ab = torch.linalg.solve(A, rhs)
            coords.append(ab.cpu().numpy())
        return np.stack(coords)

    def _nll_at_theta(self, ebm_template, theta, x_val, grid, dvol):
        """K-component NLL via grid log-Z (used at D <= 2)."""
        K = int(ebm_template.K)
        D = int(x_val.shape[-1])
        log_Z = log_Z_via_grid_1d(
            ebm_template, theta, K=K, grid=grid, dvol=float(dvol),
        )
        log_pi = torch.log_softmax(theta["log_pi"], dim=-1)
        ebm_stacked = detach_dict(
            stack_per_k_params(theta, K=K, prefix="components.")
        )
        E_val = aug_energy_batched(
            ebm_template.components[0], ebm_stacked, x_val,
            K=K, D=D, D_aug=D,
        )
        log_q = torch.logsumexp(
            log_pi[:, None] - E_val - log_Z[:, None], dim=0
        )
        return (-log_q).mean().item()

    def _nll_at_theta_flow_is(
        self, ebm_template, theta, x_val, x_prop_aug, log_q_prop,
        D_aug,
    ):
        """K-component NLL via fixed flow-IS log-Z (used at D >= 3).

        ``x_prop_aug`` (M, D_aug) and ``log_q_prop`` (M,) are pre-drawn
        once from the hypernet's converged flow. Re-using them across
        all (alpha, beta) grid points keeps the NLL surface
        deterministic and decoupled from the IS variance.
        """
        K = int(ebm_template.K)
        D = int(x_val.shape[-1])
        ebm_stacked = detach_dict(
            stack_per_k_params(theta, K=K, prefix="components.")
        )
        # log Z_k = logsumexp_i(-E_aug,k(x_i) - log q_phi(x_i)) - log M
        E_prop = aug_energy_batched(
            ebm_template.components[0], ebm_stacked, x_prop_aug,
            K=K, D=D, D_aug=int(D_aug),
        )  # (K, M)
        M = int(x_prop_aug.shape[0])
        log_Z = torch.logsumexp(
            -E_prop - log_q_prop[None, :], dim=1
        ) - math.log(M)
        log_pi = torch.log_softmax(theta["log_pi"], dim=-1)
        E_val = aug_energy_batched(
            ebm_template.components[0], ebm_stacked, x_val,
            K=K, D=D, D_aug=D,
        )
        log_q = torch.logsumexp(
            log_pi[:, None] - E_val - log_Z[:, None], dim=0
        )
        return (-log_q).mean().item()

    def _build_landscape_1d(self, ckpt_dir, backend_results, target,
                            x_val_landscape, seed: int):
        device = self.device
        # Center on hypernet's final emitted theta.
        center_theta = {
            k: v.detach().to(device).clone()
            for k, v in backend_results["hypernet"]["theta_snaps"][-1].items()
        }
        torch.manual_seed(int(seed) + 31)
        direction_mode = str(getattr(self.args, "direction_mode", "trajectory"))
        if direction_mode == "trajectory":
            d1, d2 = self._trajectory_pca_dirs(
                backend_results["hypernet"]["theta_snaps"],
                backend_results["direct"]["theta_snaps"],
                center_theta, device,
            )
        else:
            d1 = self._filter_norm_dir(center_theta)
            d2 = self._filter_norm_dir(center_theta)
        # The EBM template is frozen and in clamp mode, so the offset
        # weights are used as given rather than passed through softplus.
        from lift.objectives.builders import build_hypernet_ebm
        D_tgt = int(getattr(target, "D", 1))
        ebm_template, _ = build_hypernet_ebm(
            K=self.K, D=D_tgt,
            hidden_dim=int(self.args.hidden_dim),
            nlayers=int(self.args.nlayers),
            strong_convexity=float(self.args.strong_convexity),
            device=device,
        )
        for p in ebm_template.parameters():
            p.requires_grad_(False)

        # Eval grid for log_Z: Riemann quadrature at D <= 2, and at D >= 3
        # the converged hypernet's flow as a fixed importance-sampling
        # proposal, one draw reused at every (alpha, beta) grid point.
        n_grid_arg = int(self.args.tv_grid_n)
        if D_tgt == 1:
            xs, _, dvol = build_eval_grid(target, n_grid_arg, device)
            x_prop_aug = None
            log_q_prop = None
        elif D_tgt == 2:
            n_axis = max(int(np.sqrt(n_grid_arg)), 96)
            xs, dvol = _build_2d_grid(target, n_axis, device)
            x_prop_aug = None
            log_q_prop = None
        else:
            from lift.objectives._batched import conditional_flow_sample
            xs = None
            dvol = None
            flow_proposal = backend_results["hypernet"]["flow"]
            D_aug = int(self.args.flow_D_aug)
            M_prop = int(getattr(self.args, "M_logZ_eval", 4096) or 4096)
            with torch.no_grad():
                eps = torch.randn(
                    1, M_prop, D_aug, device=device,
                )
                x_aug, log_q_aug = conditional_flow_sample(
                    flow_proposal, eps, K=1, with_logprob=True,
                )
            x_prop_aug = x_aug[0].detach()
            log_q_prop = log_q_aug[0].detach()

        # Auto-pick a radius that contains both trajectories with margin.
        coords_h_pre = self._project_traj(
            backend_results["hypernet"]["theta_snaps"],
            center_theta, d1, d2, device,
        )
        coords_d_pre = self._project_traj(
            backend_results["direct"]["theta_snaps"],
            center_theta, d1, d2, device,
        )
        max_abs = float(max(
            np.max(np.abs(coords_h_pre)),
            np.max(np.abs(coords_d_pre)),
            1.0,
        ))
        margin = float(getattr(self.args, "landscape_margin", 1.4))
        radius_cfg = float(getattr(self.args, "landscape_radius", 0.0))
        radius = max_abs * margin if radius_cfg <= 0.0 else radius_cfg
        gn = int(getattr(self.args, "landscape_grid_n", 41))
        alphas = np.linspace(-radius, radius, gn)
        betas = np.linspace(-radius, radius, gn)
        print(
            f"[landscape] direction_mode={direction_mode} radius={radius:.3f} "
            f"(max-abs proj={max_abs:.3f}, margin={margin})"
        )

        # Apply pos-clamp on positivity-tagged params after offsetting.
        pos_names = set()
        for k in range(self.K):
            pos_names |= icnn_pos_param_names(
                int(self.args.nlayers), prefix=f"components.{k}."
            )

        Z = np.zeros((gn, gn))
        x_val_dev = x_val_landscape.to(device)
        with torch.no_grad():
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    theta_off = {}
                    for k, v_star in center_theta.items():
                        t = v_star + a * d1[k] + b * d2[k]
                        if k in pos_names:
                            t = t.clamp(min=0.0)
                        theta_off[k] = t
                    if D_tgt <= 2:
                        Z[i, j] = self._nll_at_theta(
                            ebm_template, theta_off, x_val_dev, xs, dvol,
                        )
                    else:
                        Z[i, j] = self._nll_at_theta_flow_is(
                            ebm_template, theta_off, x_val_dev,
                            x_prop_aug, log_q_prop,
                            int(self.args.flow_D_aug),
                        )

        coords_h = self._project_traj(
            backend_results["hypernet"]["theta_snaps"],
            center_theta, d1, d2, device,
        )
        coords_d = self._project_traj(
            backend_results["direct"]["theta_snaps"],
            center_theta, d1, d2, device,
        )
        np.savez(
            os.path.join(ckpt_dir, f"seed{seed}_landscape.npz"),
            alphas=alphas, betas=betas, Z=Z,
            coords_hyper=coords_h, coords_direct=coords_d,
            iters_hyper=np.asarray(
                backend_results["hypernet"]["snap_iters"], dtype=np.int64
            ),
            iters_direct=np.asarray(
                backend_results["direct"]["snap_iters"], dtype=np.int64
            ),
        )
        print(
            f"[landscape] NLL grid {gn}x{gn} r={radius} | "
            f"Z range [{Z.min():.3g}, {Z.max():.3g}] | "
            f"hyper traj {coords_h.shape[0]} pts, direct traj "
            f"{coords_d.shape[0]} pts"
        )

    def load_checkpoint(self):
        ckpt_dir = checkpointsdir(self.args.experiment)
        path = os.path.join(ckpt_dir, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                self.results = json.load(f)

    def visualize(self):
        # The multi-panel density and landscape overlays scale poorly past
        # a few seeds, so large sweeps can short-circuit them.
        if bool(getattr(self.args, "skip_visualize", 0)):
            print("[visualize] skip_visualize=1 -- skipping multi-panel plots")
            return
        ckpt_dir = checkpointsdir(self.args.experiment)
        plot_dir = plotsdir(self.args.experiment)
        os.makedirs(plot_dir, exist_ok=True)
        seeds = [int(s) for s in getattr(self.args, "seeds", [self.args.seed])]
        gmm_lp = np.load(os.path.join(ckpt_dir, "gmm_lp_grid.npy"))
        x_grid = np.load(os.path.join(ckpt_dir, "x_grid.npy"))
        target_lp = np.load(os.path.join(ckpt_dir, "target_lp_grid.npy"))
        gmm_key = next(
            (k for k in self.results if k.startswith("gmm_em_K")),
            f"gmm_em_K{self.K}",
        )
        tv_g = self.results.get(gmm_key, {}).get("tv", float("nan"))
        target_label = self._target_label()

        seed_data = []
        for s in seeds:
            try:
                dh = torch.load(
                    os.path.join(ckpt_dir, f"seed{s}_hypernet.pt"),
                    map_location="cpu", weights_only=False,
                )
                dd = torch.load(
                    os.path.join(ckpt_dir, f"seed{s}_direct.pt"),
                    map_location="cpu", weights_only=False,
                )
                ls = np.load(os.path.join(ckpt_dir, f"seed{s}_landscape.npz"))
            except FileNotFoundError:
                # A backend subset without both hypernet and direct has no
                # landscape triple; skip the seed.
                print(f"seed {s} missing hypernet/direct/landscape -- "
                      "run --phase train with backends={hypernet,direct} "
                      "to populate the shared overlay figure")
                continue
            seed_data.append((s, dh, dd, ls))
        if not seed_data:
            return

        # Multi-panel density overlays. Only meaningful at D <= 2.
        n = len(seed_data)
        D = 1 if x_grid.ndim == 1 else x_grid.shape[-1]
        if D > 2:
            fig = None  # density figure undefined at D >= 3
        elif D == 1:
            fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), squeeze=False)
            for idx, (s, dh, dd, _ls) in enumerate(seed_data):
                ax = axes[0, idx]
                tv_h = dh["metrics"].get("tv", float("nan"))
                tv_d = dd["metrics"].get("tv", float("nan"))
                ax.plot(x_grid, np.exp(target_lp), color="k", lw=1.8, label="target")
                ax.plot(x_grid, np.exp(dh["log_p_grid"]),
                        color="#00b4d8", lw=1.5,
                        label=f"hypernet TV={tv_h:.3g}")
                ax.plot(x_grid, np.exp(dd["log_p_grid"]),
                        color="#e41a1c", lw=1.5,
                        label=f"direct TV={tv_d:.3g}")
                ax.plot(x_grid, np.exp(gmm_lp), color="green", lw=1.0, ls="--",
                        label=f"GMM-K{self.K} TV={tv_g:.3g}")
                ax.set_xlabel("x")
                if idx == 0:
                    ax.set_ylabel("density")
                ax.set_title(f"seed {s}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
            fig.suptitle(
                f"{target_label}: hypernet vs direct ICNN (multi-seed)",
                y=1.02,
            )
        else:
            fig, axes = plt.subplots(
                3, n, figsize=(3.4 * n, 9.5), squeeze=False,
            )
            n_axis = int(np.sqrt(x_grid.shape[0]))
            xs = x_grid[:, 0].reshape(n_axis, n_axis)
            ys = x_grid[:, 1].reshape(n_axis, n_axis)
            target_grid = np.exp(target_lp).reshape(n_axis, n_axis)
            vmax = float(target_grid.max())
            for idx, (s, dh, dd, _ls) in enumerate(seed_data):
                tv_h = dh["metrics"].get("tv", float("nan"))
                tv_d = dd["metrics"].get("tv", float("nan"))
                hyper_grid = np.exp(dh["log_p_grid"]).reshape(n_axis, n_axis)
                direct_grid = np.exp(dd["log_p_grid"]).reshape(n_axis, n_axis)
                for row, (g, label, col) in enumerate([
                    (target_grid, "target", "k"),
                    (hyper_grid, f"hyper TV={tv_h:.3g}", "#00b4d8"),
                    (direct_grid, f"direct TV={tv_d:.3g}", "#e41a1c"),
                ]):
                    ax = axes[row, idx]
                    cs = ax.contourf(xs, ys, g, levels=20, cmap="viridis",
                                     vmin=0, vmax=vmax)
                    if row == 0:
                        ax.set_title(f"seed {s}", fontsize=10)
                    ax.set_xlabel("x" if row == 2 else "")
                    if idx == 0:
                        ax.set_ylabel(f"{label}\ny", fontsize=8)
                    else:
                        ax.set_ylabel("")
                    ax.tick_params(labelsize=7)
                    ax.text(
                        0.02, 0.95, label, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.6, lw=0,
                                   pad=1.0),
                    )
            fig.suptitle(
                f"{target_label}: hypernet vs direct ICNN (multi-seed)",
                y=1.0,
            )
        if fig is not None:
            fig.tight_layout()
            fig.savefig(
                os.path.join(plot_dir, "density_overlay_multiseed.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

        # Multi-panel landscapes.
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
        for idx, (s, _dh, _dd, ls) in enumerate(seed_data):
            ax = axes[0, idx]
            self._plot_landscape_panel(ax, ls, title=f"seed {s}")
        fig.suptitle(
            "Shared $\\theta$-space NLL landscape, multi-seed "
            f"({target_label})", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "landscape_multiseed.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

        # Per-seed standalone landscape PDFs.
        for s, _dh, _dd, ls in seed_data:
            fig, ax = plt.subplots(figsize=(5.6, 4.4))
            self._plot_landscape_panel(ax, ls, title=f"seed {s}",
                                       show_full_legend=True)
            fig.tight_layout()
            fig.savefig(
                os.path.join(plot_dir, f"landscape_seed{s}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

        # Curve versus iteration: tv_per_eval at D <= 2, and the held-out
        # NLL val_loss_per_eval at D >= 3.
        metric_key, ylabel, title_metric = (
            ("tv_per_eval", r"TV$(q, p)$", "TV")
            if D <= 2 else
            ("val_loss_per_eval", r"held-out NLL", "NLL")
        )
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.2), squeeze=False)
        for idx, (s, dh, dd, _ls) in enumerate(seed_data):
            ax = axes[0, idx]
            for backend, d, c in [
                ("hypernet", dh, "#00b4d8"),
                ("direct", dd, "#e41a1c"),
            ]:
                h = d.get("history", {})
                arr = None
                for key in (metric_key, "tv", "TV", "metric_tv"):
                    if key in h and len(h[key]) > 0:
                        arr = np.asarray(h[key], dtype=float)
                        break
                if arr is None:
                    continue
                iters_key = "eval_iters" if "eval_iters" in h else "eval_iter"
                iters = np.asarray(h.get(
                    iters_key,
                    np.linspace(0, int(self.args.n_iters), len(arr)),
                ))[: len(arr)]
                # tv_per_eval is NaN at D >= 3.
                m = np.isfinite(arr)
                if not m.any():
                    continue
                ax.plot(iters[m], arr[m], color=c, lw=1.6, label=backend)
                if (arr[m] > 0).all():
                    ax.set_yscale("log")
            ax.set_xlabel("iter")
            if idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"seed {s}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{target_label}: {title_metric} vs iter (multi-seed)",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "tv_vs_iter_multiseed.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    def _plot_landscape_panel(self, ax, npz_data, title="",
                              show_full_legend: bool = False):
        d = npz_data
        alphas, betas = d["alphas"], d["betas"]
        Z = d["Z"]
        cd = d["coords_direct"]
        ch = d["coords_hyper"]
        A, B = np.meshgrid(alphas, betas, indexing="ij")
        Z_safe = Z - Z.min() + 1e-8
        cs = ax.contourf(A, B, np.log10(Z_safe), levels=25, cmap="viridis")
        ax.contour(A, B, np.log10(Z_safe), levels=15, colors="white",
                   linewidths=0.4, alpha=0.5)
        cb = ax.figure.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(r"$\log_{10}(\mathrm{loss}-\min\mathrm{loss})$",
                     fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.plot(ch[:, 0], ch[:, 1], "-s", color="#00b4d8", lw=1.4, ms=2.6,
                mec="black", mew=0.25,
                label="hypernet traj", zorder=4, alpha=0.95)
        ax.plot(cd[:, 0], cd[:, 1], "-o", color="#e41a1c", lw=1.4, ms=2.6,
                mec="black", mew=0.25,
                label="direct traj", zorder=4, alpha=0.95)
        ax.plot(ch[0, 0], ch[0, 1], "X", color="#00b4d8", ms=11,
                mec="black", mew=0.7, zorder=5,
                label="hyper init" if show_full_legend else None)
        ax.plot(cd[0, 0], cd[0, 1], "X", color="#e41a1c", ms=11,
                mec="black", mew=0.7, zorder=5,
                label="direct init" if show_full_legend else None)
        ax.plot([0], [0], "*", color="gold", ms=18, mec="black", mew=0.7,
                label=r"hyper final $\theta^\star$" if show_full_legend else None,
                zorder=6)
        ax.plot(cd[-1, 0], cd[-1, 1], "P", color="#ff7b00", ms=13,
                mec="black", mew=0.7, zorder=6,
                label="direct final $\\theta$" if show_full_legend else None)
        ax.set_xlabel(r"$\alpha$ along $\theta_{\rm direct,fin}-\theta^\star$",
                      fontsize=9)
        ax.set_ylabel(r"$\beta$ orth (traj PC)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        if show_full_legend:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
