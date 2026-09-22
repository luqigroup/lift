"""FFHQ ICNN regularizer trained three ways: the lift, direct softplus, PGD.

Re-implements the adversarial regularizer training of Ehrhardt, Mukherjee and
Wong (Zenodo DOI 10.5281/zenodo.17426033, their ``example_ar_training.py``) on
the vendored ``simple_ICNN``: the critic objective ``E[g(clean)] - E[g(noisy)]``
with a one-sided gradient penalty, scored by PDHG reconstruction PSNR. Each
(backend, seed) cell trains into its own subdirectory and writes
``RESULTS_COMPLETE.json`` as its last act, so a relaunch resumes the sweep.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from projorg import checkpointsdir, plotsdir
from torch import nn
from torch.func import functional_call

from lift.baselines.ehrhardt_vendor import (
    InpaintingPhysics,
    L2Fidelity,
    PDHG_ICNN,
    PDHG_ICNN_L1,
    compare_psnr,
    power,
    simple_ICNN,
    simple_ICNNPrior,
)
from lift.dataset.ffhq import FFHQColorPairs, FFHQPairs
from lift.models import HyperNetwork
from lift.utils.paper_style import PALETTE, apply_paper_style


BACKENDS = ("pgd", "direct", "hypernet", "hypernet_const",
            "direct_lr2", "direct_lr5", "direct_lr8")


def _direct_lr_mult(name: str) -> float | None:
    """``direct_lr<M>`` -> M; anything else -> None.

    Selects direct softplus with the Adam step size of its constrained latents
    alone multiplied by M. The multiplier is carried in the backend name.
    """
    m = re.fullmatch(r"direct_lr(\d+)", str(name))
    return None if m is None else float(m.group(1))

# PGD is green. PALETTE carries no "pgd" key, so the literal is used.
PGD_COLOR = "#2ca02c"
BACKEND_COLORS = {
    "hypernet": PALETTE["hypernet"],
    "hypernet_const": "#9467bd",   # the lift's frozen-code control
    "direct": PALETTE["direct_softplus"],
    # direct softplus with a larger step on its constrained latents:
    # darker shades of direct's red as the multiplier grows
    "direct_lr2": "#b22222",
    "direct_lr5": "#8b0000",
    "direct_lr8": "#5a0000",
    "pgd": PGD_COLOR,
}

# Shoulder edge in effective-weight units: softplus(-2.944) ~ 0.0513.
SHOULDER_EDGE = float(F.softplus(torch.tensor(-2.944)).item())

# The constrained tensors of the vendored simple_ICNN template.
CONSTRAINED_NAMES = ("fc1.weight", "fc2.weight")

# The validation images of their bilevel trainer, as indices into the
# sorted 1,000-image training set.
HOLDOUT_LO, HOLDOUT_HI = 960, 1000

_MARKER_NAME = "RESULTS_COMPLETE.json"


def _read_json_or_none(path: str) -> dict | None:
    """Load a JSON file, returning ``None`` for a missing or torn file.

    A crash between marker creation and the completed write must read as an
    incomplete cell, never as a crash of the relaunch.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_json_atomic(obj: dict, path: str) -> None:
    """Durably write JSON: temp file, flush and fsync, then ``os.replace``.

    ``os.replace`` is atomic on a POSIX filesystem, so a reader sees either
    the previous state or the complete document, never a torn marker.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _torch_save_atomic(obj, path: str) -> None:
    """Durably write a torch checkpoint: temp file, fsync, ``os.replace``.

    ``icnn_best.pt`` is overwritten at every validation improvement, and a
    kill mid-write must leave the previous best readable.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _serializable_args(args) -> dict:
    """``vars(args)`` with any non-JSON value stringified."""
    out = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: ``y + log(-expm1(-y))`` (stable for y>0)."""
    return y + torch.log(-torch.expm1(-y))


def _constrained_stats(w: torch.Tensor) -> dict:
    """Screen statistics over a flat vector of effective weights."""
    w = w.detach().float()
    pos = w[w > 0]
    if pos.numel() > 0:
        med_pre = float(softplus_inv(pos).median().item())
    else:
        med_pre = float("nan")
    return {
        "median": float(w.median().item()),
        "frac_zero": float((w == 0).float().mean().item()),
        "frac_shoulder": float((w <= SHOULDER_EDGE).float().mean().item()),
        "median_pre_softplus_pos": med_pre,
    }


def _panel_img(a: np.ndarray) -> np.ndarray:
    """``(C, H, W)`` float array -> imshow-ready image, clipped to [0, 1].

    One channel gives ``(H, W)``, rendered through the gray colormap; three
    give ``(H, W, C)`` RGB, for which imshow ignores cmap, vmin and vmax, so
    the call site stays uniform.
    """
    a = np.clip(np.asarray(a), 0.0, 1.0)
    if a.ndim != 3:
        raise ValueError(f"expected (C, H, W); got shape {a.shape}")
    return a[0] if a.shape[0] == 1 else np.transpose(a, (1, 2, 0))


def _gradient_penalty(g_fn, real: torch.Tensor, fake: torch.Tensor):
    """Their one-sided WGAN gradient penalty, on a generic critic.

    Mirrors ``example_ar_training.py::compute_gradient_penalty`` composed with
    ``simple_ICNNPrior.grad``: uniform per-sample alpha, the gradient of
    ``sum g`` at the interpolates with ``create_graph=True`` so the penalty
    backpropagates into the critic's parameters, emitted or owned, then
    ``relu(||grad|| - 1)^2`` averaged.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=real.device)
    interpolates = real + alpha * (fake - real)
    interpolates.requires_grad_(True)
    z = torch.sum(g_fn(interpolates))
    grad = torch.autograd.grad(z, interpolates, create_graph=True)[0]
    grad_norm = grad.flatten(1).norm(2, dim=1)
    return F.relu(grad_norm - 1).square().mean()


# --------------------------------------------------------------------------
# Backend adapters. All three expose the same narrow surface to the
# trainer: ``set_train_progress(t_frac)``, ``begin_step(cond)``, ``g(x)``,
# ``post_step()``, ``effective_param_dict(cond)`` and
# ``export_param_dict(...)``. The template stays a frozen vendored
# ``simple_ICNN``; direct and hypernet drive it through
# ``torch.func.functional_call``. Its clamp-inside-forward is a numerical
# no-op under softplus-emitted weights, which are strictly positive and
# clamped in place before any op saves them for backward, so the template
# can stay exactly as shipped.
# --------------------------------------------------------------------------


FFHQ_INIT_MODES = ("own", "zero_centred", "hk")


def apply_ffhq_init(module, init_mode: str) -> None:
    """Rewrite the constrained weights of a ``simple_ICNN`` in place.

    ``DirectBackend`` and ``PGDBackend`` take no initialization argument and
    ``--head_init_pos_bias`` never reaches them, so this supplies the
    initialization axis for both. It operates on the constrained weight, not
    on a latent: the caller takes the softplus inverse afterwards where it
    needs one.

    * ``own``           leave the published ``U[0, 0.01]`` draw of Ehrhardt
      et al., whose softplus inverse has median -5.30.
    * ``zero_centred``  the latent at PyTorch's default, so the weight is
      ``softplus(U(-1/sqrt(N), 1/sqrt(N)))`` with median about 0.693 and
      nothing on the shoulder. The shoulder is ``w <= 0.0513`` and the
      published mean is 0.005, so moving off the shoulder necessarily
      raises the effective weight by two orders of magnitude: this mode
      moves the shoulder position and the strength of the regularizer
      together, and does not separate them.
    * ``hk``            Hoedt and Klambauer (arXiv:2312.12474) at their
      rho* = 1/2 solution, log-normal at mean ``sqrt(6 pi / (N D))`` and
      variance ``1/N`` with ``D = 6(pi-1) + (N-1)(3 sqrt 3 + 2 pi - 6)``,
      bias ``-sqrt(3N/D)`` by their Eq. (8). Their constants are derived
      for a ReLU kernel and for ICNNs without skip connections.
    """
    import math
    if init_mode == "own":
        return
    if init_mode not in FFHQ_INIT_MODES:
        raise ValueError(
            f"unknown init_mode {init_mode!r}; expected one of {FFHQ_INIT_MODES}")
    with torch.no_grad():
        for name, p_ in module.named_parameters():
            if name not in CONSTRAINED_NAMES:
                continue
            n_in = int(p_[0].numel()) if p_.dim() > 1 else int(p_.shape[-1])
            if init_mode == "zero_centred":
                bound = 1.0 / math.sqrt(max(n_in, 1))
                p_.copy_(F.softplus(
                    torch.empty_like(p_).uniform_(-bound, bound)))
            else:
                d = 6.0 * (math.pi - 1.0) + (n_in - 1) * (
                    3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0)
                mu_w = math.sqrt(6.0 * math.pi / (n_in * d))
                second = 1.0 / n_in + mu_w * mu_w
                mu_t = math.log(mu_w * mu_w) - 0.5 * math.log(second)
                sd_t = math.sqrt(math.log(second) - math.log(mu_w * mu_w))
                p_.copy_(torch.exp(mu_t + sd_t * torch.randn_like(p_)))


class PGDBackend(nn.Module):
    """Their recipe, unchanged: the vendored prior trained in place."""

    def __init__(self, n_channels, n_filters, kernel_size, img_size,
                 smoothed, device, init_mode: str = "own"):
        super().__init__()
        # simple_ICNNPrior's constructor applies their U[0, 0.01]
        # constrained-weight initialization.
        self.prior = simple_ICNNPrior(
            n_channels, n_filters, kernel_size, img_size, smoothed, device,
        )
        apply_ffhq_init(self.prior.icnn, init_mode)

    def set_train_progress(self, t_frac: float) -> None:  # noqa: ARG002
        pass

    def begin_step(self, cond: torch.Tensor) -> None:  # noqa: ARG002
        pass

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior.g(x)

    def post_step(self) -> None:
        # Their example_ar_training.py clamps after every Adam step.
        self.prior.icnn.zero_clip_weights()

    def effective_param_dict(self, cond=None) -> dict:  # noqa: ARG002
        with torch.no_grad():
            return {
                name: p.detach().clone()
                for name, p in self.prior.icnn.named_parameters()
            }

    def export_param_dict(self, pairs=None, device=None) -> dict:
        return self.effective_param_dict()


class DirectBackend(nn.Module):
    """Direct softplus: ``W = softplus(V)``, with no hypernetwork body.

    ``V`` starts at the softplus inverse of the same U[0, 0.01] draw PGD
    uses, median pre-readout about -5.30; the unconstrained tensors keep the
    template's default initialization, as their recipe leaves them.
    """

    def __init__(self, n_channels, n_filters, kernel_size, img_size,
                 smoothed, device, init_mode: str = "own"):
        super().__init__()
        self.template = simple_ICNN(
            n_channels, n_filters, kernel_size, img_size, smoothed, device,
        ).to(device)
        self.template.initialize_weights()  # the same U[0, 0.01] recipe
        apply_ffhq_init(self.template, init_mode)
        self._names = [n for n, _ in self.template.named_parameters()]
        raw = {}
        for name, p in self.template.named_parameters():
            v = p.detach().clone()
            if name in CONSTRAINED_NAMES:
                # torch.rand can return exactly 0, whose softplus inverse
                # is -inf; the clamp floors the draw at 1e-8, pre-readout
                # about -18.4 and still far inside the shoulder.
                v = softplus_inv(v.clamp(min=1e-8))
            raw[name.replace(".", "__")] = nn.Parameter(v)
        self.raw = nn.ParameterDict(raw)
        for p in self.template.parameters():
            p.requires_grad_(False)
        self._params: dict | None = None

    def _materialize(self) -> dict:
        out = {}
        for name in self._names:
            v = self.raw[name.replace(".", "__")]
            out[name] = F.softplus(v) if name in CONSTRAINED_NAMES else v
        return out

    def set_train_progress(self, t_frac: float) -> None:  # noqa: ARG002
        pass

    def begin_step(self, cond: torch.Tensor) -> None:  # noqa: ARG002
        self._params = self._materialize()

    def g(self, x: torch.Tensor) -> torch.Tensor:
        if self._params is None:
            self._params = self._materialize()
        return functional_call(self.template, self._params, (x,))

    def post_step(self) -> None:
        self._params = None

    def effective_param_dict(self, cond=None) -> dict:  # noqa: ARG002
        with torch.no_grad():
            return {k: v.detach().clone()
                    for k, v in self._materialize().items()}

    def export_param_dict(self, pairs=None, device=None) -> dict:
        return self.effective_param_dict()



class GroupLRDirectBackend(DirectBackend):
    """Direct softplus with a larger Adam step on the constrained latents.

    Identical to :class:`DirectBackend` except that :meth:`param_groups`
    hands the optimizer the constrained latents at ``lr * lr_mult`` and every
    other parameter at ``lr``. The trainer uses it when present.
    """

    def __init__(self, *args, lr_mult: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr_mult = float(lr_mult)

    def param_groups(self, lr: float) -> list[dict]:
        con, rest = [], []
        for name in self._names:
            p = self.raw[name.replace(".", "__")]
            (con if name in CONSTRAINED_NAMES else rest).append(p)
        return [{"params": con, "lr": float(lr) * self.lr_mult},
                {"params": rest, "lr": float(lr)}]

class HypernetBackend(nn.Module):
    """The lift: conv-summary ``HyperNetwork`` emitting the template."""

    def __init__(self, n_channels, n_filters, kernel_size, img_size,
                 smoothed, device, hyper_hidden_sizes, conv_channels,
                 head_init_pos_bias, emission_decay_frac=0.0,
                 emission_decay_end=1.0, emission_decay_floor=0.0,
                 emission_decay_ema=0.99, pool_scale=0.0,
                 readout_fanin_scale=False):
        super().__init__()
        self.template = simple_ICNN(
            n_channels, n_filters, kernel_size, img_size, smoothed, device,
        ).to(device)
        for p in self.template.parameters():
            p.requires_grad_(False)
        self.hyper = HyperNetwork(
            input_size=int(n_channels) * int(img_size) * int(img_size),
            hidden_sizes=[int(h) for h in hyper_hidden_sizes],
            downstream_network=self.template,
            pos_param_names=set(CONSTRAINED_NAMES),
            pos_constraint_mode="softplus",
            head_init_pos_bias=float(head_init_pos_bias),
            summary_kind="conv",
            image_shape=(int(n_channels), int(img_size), int(img_size)),
            conv_channels=[int(c) for c in conv_channels],
            emission_decay_frac=float(emission_decay_frac),
            emission_decay_end=float(emission_decay_end),
            emission_decay_floor=float(emission_decay_floor),
            emission_decay_ema=float(emission_decay_ema),
            # Norm pin on the pooled summary (0 = off). Bounds the
            # batch-dependent part of the emission,
            # |Theta_E (u(X) - u(X'))| <= 2 s |Theta_E|, independently of
            # how large the encoder's ReLU features grow.
            pool_scale=float(pool_scale),
            readout_fanin_scale=bool(readout_fanin_scale),
        ).to(device)
        self._params: dict | None = None

    def set_train_progress(self, t_frac: float) -> None:
        self.hyper.set_train_progress(float(t_frac))

    def begin_step(self, cond: torch.Tensor) -> None:
        # One emission per optimizer step: the critic must be a single
        # fixed function within a step, so the clean, noisy and interpolated
        # evaluations all consume this same emitted weight set.
        self._params = self.hyper(cond)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        if self._params is None:
            raise RuntimeError(
                "HypernetBackend.g called before begin_step(cond); the "
                "emission must be conditioned explicitly.",
            )
        return functional_call(self.template, self._params, (x,))

    def post_step(self) -> None:
        self._params = None

    def effective_param_dict(self, cond: torch.Tensor = None) -> dict:
        if cond is None:
            raise ValueError(
                "HypernetBackend needs a conditioning batch to "
                "materialize effective weights.",
            )
        was_training = self.hyper.training
        self.hyper.eval()
        with torch.no_grad():
            out = {k: v.detach().clone() for k, v in self.hyper(cond).items()}
        self.hyper.train(was_training)
        return out

    def export_param_dict(self, pairs, device, chunk: int = 64) -> dict:
        """Deployment freeze: the emission pooled over the full training set.

        The DeepSets summary is a mean of per-sample encodings, so pooling
        over all images, in chunks, is the canonical summary and a reference
        batch is its subsample estimate. Only the pooling happens here: the
        map from a pooled summary to the emitted weights stays
        ``HyperNetwork.emit_from_pooled``, so the freeze cannot drift from
        what training computed. Eval mode, so no activation noise and no
        running-mean update.
        """
        was_training = self.hyper.training
        self.hyper.eval()
        with torch.no_grad():
            acc, total = None, 0
            for imgs in pairs.clean.split(chunk):
                z = self.hyper.inner_net(imgs.to(device).clamp(0, 1))
                s = z.sum(dim=0, keepdim=True)
                acc = s if acc is None else acc + s
                total += int(z.shape[0])
            out = {
                k: v.detach().clone()
                for k, v in self.hyper.emit_from_pooled(
                    acc / float(total),
                ).items()
            }
        self.hyper.train(was_training)
        return out


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------



class FrozenCodeHypernetBackend(HypernetBackend):
    """The lift with its pooled code pinned: the readout-only control.

    Identical readouts, parameter count, fan-in, norm pin and initialization
    to the live lift, and no batch dependence: at the first ``begin_step``
    the encoder's pooled summary of that batch is stored, the encoder is
    frozen, and every later step emits from the stored code. Only
    ``Theta_E`` and ``b_h`` train, so a speed-up reproduced here is the
    readout's step size under Adam rather than the batch coupling.
    """

    def begin_step(self, cond: torch.Tensor) -> None:
        h = self.hyper
        if not hasattr(self, "h_bar"):
            with torch.no_grad():
                x = cond if cond.dim() == 4 else cond.reshape(-1, *h.image_shape)
                z = h.inner_net(x).mean(dim=0, keepdim=True)
            self.register_buffer("h_bar", z.detach().clone())
            h.inner_net.requires_grad_(False)
        self._params = h.emit_from_pooled(self.h_bar)

    def _emit_frozen(self) -> dict:
        if not hasattr(self, "h_bar"):
            raise RuntimeError("frozen-code lift: no step has pinned the code yet")
        was_training = self.hyper.training
        self.hyper.eval()
        with torch.no_grad():
            out = {k: v.detach().clone()
                   for k, v in self.hyper.emit_from_pooled(self.h_bar).items()}
        self.hyper.train(was_training)
        return out

    def effective_param_dict(self, cond: torch.Tensor) -> dict:
        del cond                    # the code is pinned; the batch is not read
        return self._emit_frozen()

    def export_param_dict(self, pairs=None, device=None, chunk: int = 64) -> dict:
        """Deployment freeze: the pinned code this recipe trained with.

        The live lift pools its summary over the whole training set here.
        This recipe never read a second batch, so the only consistent export
        is the emission from the stored code.
        """
        del pairs, device, chunk
        return self._emit_frozen()

class FFHQICNNRegularizer:
    """Driver-facing experiment: train / load_checkpoint / visualize."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if int(args.img_size) % 16 != 0:
            raise ValueError(
                f"img_size must be divisible by 16 (the template's fixed "
                f"16x16/stride-16 average pool); got {args.img_size}",
            )
        if str(args.task) not in ("inpaint", "snp"):
            raise ValueError(f"task must be 'inpaint' or 'snp', "
                             f"got {args.task!r}")
        self.task = str(args.task)
        self.results: dict = {}
        self._pairs_cache: dict = {}

    # ------------------------------------------------------------- data
    def _data_root(self):
        root = str(getattr(self.args, "data_root", "") or "")
        return root if root else None

    def _build_pairs(self, split: str, degradation_seed: int):
        """Channel-dispatched pairs: 1 selects the grayscale Ehrhardt
        subset, 3 the prepared color FFHQ root. Both classes expose the
        identical surface the trainer consumes: clean, noisy, negatives,
        mask, img_size and len."""
        a = self.args
        n_ch = int(a.n_channels)
        common = dict(
            split=split, task=self.task,
            degradation_seed=int(degradation_seed),
            data_root=self._data_root(),
            noise_sigma=float(a.noise_sigma),
            snp_prob=float(a.snp_prob),
            # DC-shortcut fix for the critic's negatives, off by default;
            # see lift/dataset/ffhq.py.
            mean_match_negatives=bool(
                int(getattr(a, "mean_match_negatives", 0))),
        )
        if n_ch == 1:
            pairs = FFHQPairs(**common)
        elif n_ch == 3:
            pairs = FFHQColorPairs(**common)
        else:
            raise ValueError(
                f"n_channels must be 1 (grayscale Ehrhardt subset) or "
                f"3 (prepared color FFHQ); got {n_ch}",
            )
        # The template's fc1 in_features derives from args.img_size, so a
        # mismatch with the data extent would break the flatten silently
        # downstream. Fail here instead.
        if int(pairs.img_size) != int(a.img_size):
            raise ValueError(
                f"--img_size {a.img_size} does not match the loaded "
                f"data extent {pairs.img_size} "
                f"(data_root={self._data_root()!r}).",
            )
        return pairs

    def _train_pairs(self, seed: int):
        key = ("train", int(seed))
        if key not in self._pairs_cache:
            print(f"[data] loading FFHQ train pairs "
                  f"(task={self.task}, degradation_seed={seed}, "
                  f"n_channels={self.args.n_channels})...",
                  flush=True)
            self._pairs_cache[key] = self._build_pairs("train", seed)
        return self._pairs_cache[key]

    def _test_pairs(self):
        key = ("test", int(self.args.eval_noise_seed))
        if key not in self._pairs_cache:
            print(f"[data] loading FFHQ test pairs "
                  f"(task={self.task}, "
                  f"degradation_seed={self.args.eval_noise_seed}, "
                  f"n_channels={self.args.n_channels})...",
                  flush=True)
            self._pairs_cache[key] = self._build_pairs(
                "test", int(self.args.eval_noise_seed),
            )
        return self._pairs_cache[key]

    # --------------------------------------------------------- builders
    def _template_args(self):
        a = self.args
        return (int(a.n_channels), int(a.n_filters), int(a.kernel_size),
                int(a.img_size), bool(int(a.smoothed)), self.device)

    def _build_backend(self, backend: str):
        a = self.args
        _im = str(getattr(a, "init_mode", "own"))
        if backend == "pgd":
            return PGDBackend(*self._template_args(), init_mode=_im)
        if backend == "direct":
            return DirectBackend(*self._template_args(), init_mode=_im)
        if _direct_lr_mult(backend) is not None:
            return GroupLRDirectBackend(
                *self._template_args(), init_mode=_im,
                lr_mult=_direct_lr_mult(backend),
            )
        if backend in ("hypernet", "hypernet_const"):
            cls = HypernetBackend if backend == "hypernet" else FrozenCodeHypernetBackend
            return cls(
                *self._template_args(),
                hyper_hidden_sizes=list(a.hyper_hidden_sizes),
                conv_channels=list(a.conv_channels),
                head_init_pos_bias=float(a.head_init_pos_bias),
                emission_decay_frac=float(
                    getattr(a, "emission_decay_frac", 0.0)),
                emission_decay_end=float(
                    getattr(a, "emission_decay_end", 1.0)),
                emission_decay_floor=float(
                    getattr(a, "emission_decay_floor", 0.0)),
                emission_decay_ema=float(
                    getattr(a, "emission_decay_ema", 0.99)),
                pool_scale=float(getattr(a, "pool_scale", 0.0)),
                readout_fanin_scale=bool(int(
                    getattr(a, "readout_fanin_scale", 0))),
            )
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {BACKENDS}",
        )

    def _scratch_prior(self) -> simple_ICNNPrior:
        return simple_ICNNPrior(*self._template_args())

    @staticmethod
    def _load_effective(prior: simple_ICNNPrior, eff: dict) -> None:
        with torch.no_grad():
            for name, t in eff.items():
                prior.icnn.get_parameter(name).data.copy_(t)

    # ------------------------------------------------------------- PDHG
    def _pdhg_step_sizes(self, prior: simple_ICNNPrior):
        a = self.args
        gp = torch.Generator().manual_seed(0)
        H = int(a.img_size)
        x_power = torch.randn(
            1, int(a.n_channels), H, H, generator=gp,
        ).to(self.device)
        z_power = torch.randn(
            1, int(a.n_filters), H, H, generator=gp,
        ).to(self.device)
        W0norm = power(prior.icnn.W0, prior.icnn.W0T, x_power)
        WAnorm = power(prior.icnn.WA, prior.icnn.WAT, z_power)
        c1, c2 = float(a.pdhg_c1), float(a.pdhg_c2)
        sigma0 = c1 / W0norm ** 2
        sigma1 = c2 / WAnorm ** 2
        taux = 1 / (sigma0 * W0norm ** 2)
        tauz = 1 / (sigma0 + sigma1 * WAnorm ** 2)
        return sigma0, sigma1, taux, tauz

    def _solve_psnr(self, prior, cleans, noisys, mask, max_iter,
                    return_recons=False):
        """Per-image reconstruction PSNR through the vendored solvers.

        Batch 1 throughout, which is their evaluation-script protocol and
        also sidesteps the batched ``v2`` aliasing quirk recorded in the
        vendored solver. The inpainting defaults follow their shipped
        bilevel-validation protocol: smoothed prior, lambda 0.1, c1 = 0.01,
        c2 = 0.001, max_iter 500, tol 0.1. The dataset's ``mask*img + noise``
        differs from ``example_inpaint.py``'s ``(img + noise)*mask`` only
        outside the mask, and every solver-touching path masks ``y``, so the
        two conventions are equivalent here.
        """
        a = self.args
        psnrs, recons = [], []
        if self.task == "inpaint":
            physics = InpaintingPhysics(mask.to(self.device))
            fid = L2Fidelity()
            sigma0, sigma1, taux, tauz = self._pdhg_step_sizes(prior)
        for i in range(int(cleans.shape[0])):
            img = cleans[i:i + 1].to(self.device)
            y = noisys[i:i + 1].to(self.device)
            if self.task == "inpaint":
                x0 = physics.A_adjoint(y)
                x, _, _ = PDHG_ICNN(
                    x0, y, img, physics, fid, prior,
                    float(a.pdhg_lambda), sigma0, sigma1, taux, tauz,
                    max_iter=int(max_iter), tol=float(a.pdhg_tol),
                    device=self.device,
                )
            else:
                x = PDHG_ICNN_L1(
                    y, prior, float(a.snp_reg_param),
                    float(a.snp_c1), float(a.snp_c2),
                    max_iter=int(max_iter), device=self.device,
                )
            psnrs.append(compare_psnr(img.cpu().numpy(), x.cpu().numpy()))
            if return_recons:
                recons.append(x.detach().cpu())
        if return_recons:
            return psnrs, recons
        return psnrs

    # ---------------------------------------------------------- training
    def train(self):
        a = self.args
        backends = [str(b) for b in a.backends]
        for b in backends:
            if b not in BACKENDS:
                raise ValueError(
                    f"unknown backend {b!r}; expected one of {BACKENDS}",
                )
        seeds = [int(s) for s in a.seeds]
        ckpt_dir = checkpointsdir(a.experiment)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[train] device={self.device}  backends={backends}  "
              f"seeds={seeds}  task={self.task}", flush=True)
        for backend in backends:
            for seed in seeds:
                arm_dir = os.path.join(ckpt_dir, f"{backend}_seed{seed}")
                os.makedirs(arm_dir, exist_ok=True)
                marker = os.path.join(arm_dir, _MARKER_NAME)
                done = _read_json_or_none(marker)
                if done is not None:
                    print(f"[skip] {backend} seed {seed} already complete "
                          f"(test PSNR "
                          f"{done.get('test_psnr_mean', float('nan')):.2f} dB, "
                          f"finished {done.get('completed', '?')}); "
                          f"delete {marker} to retrain.", flush=True)
                    continue
                if os.path.isfile(marker):
                    # A torn marker means the cell is not complete.
                    print(f"[resume] {backend} seed {seed}: unreadable "
                          f"marker at {marker}; treating the arm as "
                          f"incomplete and retraining.", flush=True)
                try:
                    self._train_one_arm(backend, seed, arm_dir)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:  # noqa: BLE001 -- isolate cells
                    # A sweep cell that breaks one recipe must not take the
                    # other backends in the same invocation down with it.
                    # No completion marker is written, so a resubmission
                    # retries this cell.
                    failed = os.path.join(arm_dir, "RESULTS_FAILED.json")
                    with open(failed, "w") as f:
                        json.dump({
                            "backend": backend, "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "failed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }, f, indent=2)
                    print(f"[FAILED] {backend} seed {seed}: "
                          f"{type(exc).__name__}: {exc}\n"
                          f"         recorded in {failed}; continuing with "
                          f"the next arm.", flush=True)
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                # Refresh the run-root summary after every cell, so a
                # killed sweep task still leaves an args-bearing
                # metrics.json for the aggregator to identify it by.
                self._write_summary()
        self._write_summary()

    def _train_one_arm(self, backend: str, seed: int, arm_dir: str):
        a = self.args
        t_start = time.time()
        pairs = self._train_pairs(seed)
        N = len(pairs)
        batch = int(a.batch_size)
        n_epochs = int(a.n_epochs)
        steps_per_epoch = int(math.ceil(N / batch))
        total_steps = n_epochs * steps_per_epoch
        print(f"\n=== [{backend} | seed {seed}] {n_epochs} epochs x "
              f"{steps_per_epoch} steps = {total_steps} steps "
              f"(N={N}, batch={batch}) ===", flush=True)

        # Seed right before model construction. The data noise realization
        # and the batch order come from explicit generators, so they are
        # paired across backends at equal seed.
        torch.manual_seed(seed)
        adapter = self._build_backend(backend)
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        n_trainable = sum(int(p.numel()) for p in trainable)
        print(f"[{backend} | seed {seed}] trainable parameters: "
              f"{n_trainable:,}", flush=True)
        # A backend may split its parameters into groups with their own
        # step sizes (direct_lr<M>); the others hand over one group.
        groups = (adapter.param_groups(float(a.lr))
                  if hasattr(adapter, "param_groups") else trainable)
        opt = torch.optim.Adam(
            groups, lr=float(a.lr),
            betas=(float(a.adam_beta1), float(a.adam_beta2)),
        )
        gshuffle = torch.Generator().manual_seed(seed + 101)

        # Held-out diagnostic split: their bilevel validation images, which
        # sit inside the training set.
        if N > HOLDOUT_LO:
            hold_idx = list(range(HOLDOUT_LO, min(HOLDOUT_HI, N)))
        else:  # tiny synthetic sets in tests
            hold_idx = list(range(max(0, N - 4), N))
        hold_clean = pairs.clean[hold_idx]
        hold_noisy = pairs.noisy[hold_idx]
        n_val = min(int(a.val_n_images), len(hold_idx))
        # Fixed conditioning batch for the hypernet's diagnostic emission,
        # stable across the run and disjoint from the batch order.
        diag_cond = hold_clean[:batch].to(self.device).clamp(0, 1)

        val_prior = self._scratch_prior()
        # Best-validation tracker; None turns selection off. ``vp > -inf``
        # is False for NaN, so an empty validation split can never record a
        # best.
        best = ({"val_psnr": -float("inf"), "step": None}
                if int(getattr(a, "best_val_ckpt", 1)) else None)
        # A killed earlier attempt can leave a stale icnn_best.pt or a torn
        # .tmp behind. Results never consume it, but remove both so the
        # directory cannot mislead inspection when this attempt records no
        # finite validation PSNR.
        for stale in ("icnn_best.pt", "icnn_best.pt.tmp"):
            stale_path = os.path.join(arm_dir, stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
        history = {
            "steps": [], "loss": [], "diff_loss": [], "gp_loss": [],
            "eval_steps": [], "holdout_diff": [], "val_psnr": [],
            "w_median": [], "w_frac_zero": [], "w_frac_shoulder": [],
            "w_median_pre_softplus_pos": [],
        }
        ckpt_every = int(a.checkpoint_every)
        snap_every = int(getattr(a, "snap_every", 0))
        # Conditioning sub-batch; 0, the default, is the whole batch. Only
        # the lift reads it: pgd and direct ignore the argument to
        # ``begin_step``, so the driver refuses that combination rather than
        # accept a knob that does nothing.
        n_cond = int(getattr(a, "n_cond", 0))
        if n_cond > batch:
            raise ValueError(
                f"--n_cond {n_cond} exceeds the training batch {batch}; "
                "the emission conditions on a SUB-batch of the batch "
                "the gradient is taken on",
            )
        step = 0
        for epoch in range(n_epochs):
            perm = torch.randperm(N, generator=gshuffle)
            for start in range(0, N, batch):
                idx = perm[start:start + batch]
                clean = pairs.clean[idx].to(self.device).clamp(0, 1)
                # The critic's negative class: ``pairs.noisy`` under their
                # released recipe, the DC-matched copy under
                # --mean_match_negatives 1. Validation and test keep solving
                # from ``pairs.noisy``, the measurement y.
                noisy = pairs.negatives[idx].to(self.device).clamp(0, 1)

                # Emission-decay schedule: telling the emitter where
                # training is lets it anneal the batch-dependent component
                # of the emission toward its running mean, so the escape
                # from the attenuated region stays early and the iterate
                # can settle late. A no-op unless the run sets
                # ``--emission_decay_frac``, and always a no-op for pgd
                # and direct.
                adapter.set_train_progress(
                    float(step + 1) / float(total_steps),
                )
                # The emission is conditioned on the first ``n_cond``
                # samples of the batch the gradient is taken on, or the
                # whole batch when ``n_cond`` is 0. The coupling between
                # emission and gradient is a per-sample correlation,
                # unchanged by how many samples the emission reads, while
                # the jitter carries the ratio of the gradient batch to the
                # conditioning sub-batch, so a smaller sub-batch raises the
                # jitter alone. The sub-batch must come from the gradient
                # batch: an emission read off an independent batch would
                # zero the coupling. Taking the first rows is as good as a
                # random subset, since the batch is a fresh shuffle every
                # epoch.
                adapter.begin_step(
                    clean if n_cond <= 0 else clean[:n_cond]
                )
                diff_loss = (adapter.g(clean).mean()
                             - adapter.g(noisy).mean())
                gp_loss = _gradient_penalty(
                    adapter.g, clean.detach(), noisy.detach(),
                )
                loss = diff_loss + float(a.lambda_gp) * gp_loss
                if not torch.isfinite(loss):
                    # Stop on divergence rather than spend the budget and
                    # 500 PDHG iterations on NaN weights. train() records
                    # the failure and continues with the next cell.
                    raise RuntimeError(
                        f"non-finite loss at step {step + 1}/{total_steps} "
                        f"(diff={diff_loss.item()}, gp={gp_loss.item()}); "
                        f"lr={a.lr} diverged for backend {backend!r}",
                    )

                opt.zero_grad()
                loss.backward()
                opt.step()
                adapter.post_step()
                step += 1

                if step == 1 or step % int(a.log_every) == 0:
                    history["steps"].append(step)
                    history["loss"].append(float(loss.item()))
                    history["diff_loss"].append(float(diff_loss.item()))
                    history["gp_loss"].append(float(gp_loss.item()))
                    print(f"[{backend} | seed {seed}] step {step}/"
                          f"{total_steps}  loss={loss.item():.6f}  "
                          f"diff={diff_loss.item():.6f}  "
                          f"gp={gp_loss.item():.6f}", flush=True)

                if step % int(a.eval_every) == 0 or step == total_steps:
                    self._record_eval(
                        adapter, backend, seed, step, diag_cond,
                        hold_clean, hold_noisy, n_val, pairs,
                        val_prior, history, arm_dir, best,
                    )
                    hyper = getattr(adapter, "hyper", None)
                    if (hyper is not None
                            and hyper.emission_decay_frac > 0.0):
                        # Only written by a decay run.
                        history.setdefault("emission_gain", []).append(
                            float(hyper.emission_gain),
                        )

                if ckpt_every > 0 and step % ckpt_every == 0:
                    eff = adapter.effective_param_dict(cond=diag_cond)
                    self._load_effective(val_prior, eff)
                    torch.save(
                        {"step": step, "backend": backend, "seed": seed,
                         "prior_state": {
                             k: v.cpu()
                             for k, v in val_prior.state_dict().items()
                         }},
                        os.path.join(arm_dir, "checkpoint_last.pt"),
                    )
                    print(f"[{backend} | seed {seed}] checkpoint at step "
                          f"{step} -> checkpoint_last.pt", flush=True)

                # Trajectory snapshots for the landscape renderers. The
                # house frame's second direction is the top SVD loading
                # across the union of snapshots, so these are
                # non-overwriting, unlike checkpoint_last.pt, and store the
                # effective-weight prior state every landscape reduction
                # starts from. Same-step filenames make a retrain after a
                # kill self-healing.
                if snap_every > 0 and (step % snap_every == 0 or step == 1):
                    eff = adapter.effective_param_dict(cond=diag_cond)
                    self._load_effective(val_prior, eff)
                    snap_dir = os.path.join(arm_dir, "snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    _torch_save_atomic(
                        {"step": step,
                         "prior_state": {
                             k: v.cpu()
                             for k, v in val_prior.state_dict().items()
                         }},
                        os.path.join(snap_dir, f"snap_{step:05d}.pt"),
                    )

        # ---- completion: export, final eval, metrics, marker (last) ----
        export_extra = {}
        if backend in ("hypernet", "hypernet_const"):
            full_set = adapter.export_param_dict(pairs, self.device)
            ref_batch = adapter.effective_param_dict(cond=diag_cond)
            freeze_diag = {}
            for name in full_set:
                d = (full_set[name] - ref_batch[name]).abs()
                denom = full_set[name].abs().max().clamp(min=1e-12)
                freeze_diag[name] = {
                    "max_abs_diff": float(d.max().item()),
                    "rel_max_abs_diff": float((d.max() / denom).item()),
                }
            export_extra["emission_freeze_diag"] = freeze_diag
            export = full_set
            print(f"[{backend} | seed {seed}] emission freeze: full-set "
                  f"vs reference-batch max rel diff = "
                  + "  ".join(f"{k}:{v['rel_max_abs_diff']:.3e}"
                              for k, v in freeze_diag.items()),
                  flush=True)
        else:
            export = adapter.export_param_dict()

        self._load_effective(val_prior, export)
        torch.save(
            {k: v.cpu() for k, v in val_prior.state_dict().items()},
            os.path.join(arm_dir, "icnn_final.pt"),
        )
        if backend in ("hypernet", "hypernet_const") and int(getattr(a, "save_emitter", 1)):
            # Provenance-only artifact of roughly 0.5-1 GB. Nothing
            # downstream reads it, so a sweep passes --save_emitter 0.
            torch.save(
                {k: v.cpu() for k, v in adapter.hyper.state_dict().items()},
                os.path.join(arm_dir, "emitter_final.pt"),
            )

        test = self._test_pairs()
        print(f"[{backend} | seed {seed}] final eval: PDHG over the "
              f"{len(test)} test images...", flush=True)
        n_ex = min(int(a.n_recon_examples), len(test))
        max_iter = (int(a.eval_pdhg_max_iter) if self.task == "inpaint"
                    else int(a.snp_eval_max_iter))
        test_psnrs, recons = self._solve_psnr(
            val_prior, test.clean, test.noisy, test.mask, max_iter,
            return_recons=True,
        )
        test_mean = float(np.mean(test_psnrs))
        test_std = float(np.std(test_psnrs))

        # Best-validation checkpoint: the same full test protocol at the
        # weights captured when the held-out PSNR peaked. The last-iterate
        # keys above stay untouched.
        bestval_extra: dict = {}
        if best is not None and best["step"] is not None:
            best_path = os.path.join(arm_dir, "icnn_best.pt")
            best_ckpt = torch.load(
                best_path, map_location="cpu", weights_only=True,
            )
            best_prior = self._scratch_prior()
            best_prior.load_state_dict(best_ckpt["prior_state"])
            print(f"[{backend} | seed {seed}] best-val eval: PDHG over "
                  f"the {len(test)} test images at step "
                  f"{best_ckpt['step']} weights "
                  f"(val PSNR {best_ckpt['val_psnr']:.2f} dB)...",
                  flush=True)
            best_psnrs = self._solve_psnr(
                best_prior, test.clean, test.noisy, test.mask, max_iter,
            )
            bestval_extra = {
                "test_psnr_bestval_per_image": [float(p)
                                                for p in best_psnrs],
                "test_psnr_bestval_mean": float(np.mean(best_psnrs)),
                "test_psnr_bestval_std": float(np.std(best_psnrs)),
                "bestval_step": int(best["step"]),
                "bestval_val_psnr": float(best["val_psnr"]),
            }
        elif best is not None:
            print(f"[{backend} | seed {seed}] WARNING: best_val_ckpt on "
                  f"but no finite validation PSNR was ever recorded; "
                  f"skipping the best-val eval.", flush=True)

        np.savez(
            os.path.join(arm_dir, "recon_examples.npz"),
            clean=test.clean[:n_ex].numpy(),
            noisy=test.noisy[:n_ex].numpy(),
            recon=torch.cat(recons[:n_ex], dim=0).numpy(),
        )
        wall = time.time() - t_start
        metrics = {
            "backend": backend,
            "seed": int(seed),
            "task": self.task,
            "history": history,
            "test_psnr_per_image": [float(p) for p in test_psnrs],
            "test_psnr_mean": test_mean,
            "test_psnr_std": test_std,
            "n_trainable": int(n_trainable),
            "wall_time_s": float(wall),
            **bestval_extra,
            **export_extra,
        }
        _write_json_atomic(metrics, os.path.join(arm_dir, "metrics.json"))
        marker_bestval = {
            k: bestval_extra[k]
            for k in ("test_psnr_bestval_mean", "test_psnr_bestval_std",
                      "bestval_step", "bestval_val_psnr")
            if k in bestval_extra
        }
        # The marker is written last, and atomically, so an interrupted
        # cell reruns and a torn file can never shadow a complete one.
        _write_json_atomic(
            {"backend": backend, "seed": int(seed),
             "completed": datetime.datetime.now(
                 datetime.timezone.utc).isoformat(),
             "test_psnr_mean": test_mean,
             "test_psnr_std": test_std,
             **marker_bestval,
             "wall_time_s": float(wall)},
            os.path.join(arm_dir, _MARKER_NAME),
        )
        done_msg = (f"[{backend} | seed {seed}] DONE in "
                    f"{wall / 60:.1f} min; test PSNR "
                    f"{test_mean:.2f} +/- {test_std:.2f} dB")
        if "test_psnr_bestval_mean" in bestval_extra:
            done_msg += (f" (best-val @ step "
                         f"{bestval_extra['bestval_step']}: "
                         f"{bestval_extra['test_psnr_bestval_mean']:.2f}"
                         f" +/- "
                         f"{bestval_extra['test_psnr_bestval_std']:.2f}"
                         f" dB)")
        print(done_msg, flush=True)

    def _record_eval(self, adapter, backend, seed, step, diag_cond,
                     hold_clean, hold_noisy, n_val, pairs, val_prior,
                     history, arm_dir, best):
        a = self.args
        mask = pairs.mask
        with torch.no_grad():
            eff = adapter.effective_param_dict(cond=diag_cond)
            w = torch.cat([eff[n].flatten() for n in CONSTRAINED_NAMES])
            stats = _constrained_stats(w)
            self._load_effective(val_prior, eff)
            # Held-out critic difference under the same effective weights
            # the validation solve uses.
            n_hold = int(hold_clean.shape[0])
            bs = int(a.batch_size)
            g_c, g_n, seen = 0.0, 0.0, 0
            for s0 in range(0, n_hold, bs):
                c = hold_clean[s0:s0 + bs].to(self.device).clamp(0, 1)
                n_ = hold_noisy[s0:s0 + bs].to(self.device).clamp(0, 1)
                g_c += float(val_prior.g(c).sum().item())
                g_n += float(val_prior.g(n_).sum().item())
                seen += int(c.shape[0])
            hd = (g_c - g_n) / max(seen, 1)
        val_psnrs = self._solve_psnr(
            val_prior, hold_clean[:n_val], hold_noisy[:n_val], mask,
            max_iter=int(a.val_pdhg_max_iter),
        )
        vp = float(np.mean(val_psnrs)) if val_psnrs else float("nan")
        history["eval_steps"].append(step)
        history["holdout_diff"].append(float(hd))
        history["val_psnr"].append(vp)
        history["w_median"].append(stats["median"])
        history["w_frac_zero"].append(stats["frac_zero"])
        history["w_frac_shoulder"].append(stats["frac_shoulder"])
        history["w_median_pre_softplus_pos"].append(
            stats["median_pre_softplus_pos"],
        )
        if best is not None and vp > best["val_psnr"]:
            # New best held-out PSNR: capture the deployment weights at
            # this step. For the hypernet that is the same chunked
            # full-training-set emission as the final export, never the
            # diagnostic-batch emission. No RNG is consumed here, so the
            # capture cannot perturb the training trajectory.
            if backend in ("hypernet", "hypernet_const"):
                export = adapter.export_param_dict(pairs, self.device)
            else:
                export = adapter.export_param_dict()
            self._load_effective(val_prior, export)
            _torch_save_atomic(
                {"step": int(step), "backend": backend,
                 "seed": int(seed), "val_psnr": float(vp),
                 "prior_state": {
                     k: v.cpu()
                     for k, v in val_prior.state_dict().items()
                 }},
                os.path.join(arm_dir, "icnn_best.pt"),
            )
            best["val_psnr"] = float(vp)
            best["step"] = int(step)
            print(f"[{backend} | seed {seed}] new best val PSNR "
                  f"{vp:.2f} dB at step {step} -> icnn_best.pt",
                  flush=True)
        print(f"[{backend} | seed {seed}] eval @ step {step}: "
              f"val_psnr={vp:.2f} dB  holdout_diff={hd:.6f}  "
              f"w_median={stats['median']:.5f}  "
              f"frac_zero={stats['frac_zero']:.3f}  "
              f"frac_shoulder={stats['frac_shoulder']:.3f}  "
              f"pre_median={stats['median_pre_softplus_pos']:.3f}",
              flush=True)

    # ------------------------------------------------------- aggregation
    def _completed_arms(self):
        ckpt_dir = checkpointsdir(self.args.experiment)
        out = {}
        for backend in BACKENDS:
            for seed in [int(s) for s in self.args.seeds]:
                arm_dir = os.path.join(ckpt_dir, f"{backend}_seed{seed}")
                m = _read_json_or_none(os.path.join(arm_dir, "metrics.json"))
                if m is not None:
                    out[(backend, seed)] = m
        return out

    def _write_summary(self):
        ckpt_dir = checkpointsdir(self.args.experiment)
        arms = self._completed_arms()
        summary: dict = {
            "task": self.task,
            # The resolved run arguments identify this hash directory's
            # sweep cell for the post-hoc aggregator.
            "args": _serializable_args(self.args),
            "per_arm": {},
            "per_backend": {},
        }
        for (backend, seed), m in arms.items():
            entry = {
                "test_psnr_mean": m["test_psnr_mean"],
                "test_psnr_std": m["test_psnr_std"],
            }
            # Present only when best-val selection was on.
            if "test_psnr_bestval_mean" in m:
                entry["test_psnr_bestval_mean"] = m["test_psnr_bestval_mean"]
                entry["test_psnr_bestval_std"] = m["test_psnr_bestval_std"]
                entry["bestval_step"] = m["bestval_step"]
            summary["per_arm"][f"{backend}_seed{seed}"] = entry
        for backend in BACKENDS:
            means = [m["test_psnr_mean"] for (b, _), m in arms.items()
                     if b == backend]
            if means:
                summary["per_backend"][backend] = {
                    "test_psnr_mean_over_seeds": float(np.mean(means)),
                    "test_psnr_std_over_seeds": float(np.std(means)),
                    "n_seeds": len(means),
                }
            bmeans = [m["test_psnr_bestval_mean"]
                      for (b, _), m in arms.items()
                      if b == backend and "test_psnr_bestval_mean" in m]
            if bmeans:
                summary["per_backend"][backend].update({
                    "test_psnr_bestval_mean_over_seeds":
                        float(np.mean(bmeans)),
                    "test_psnr_bestval_std_over_seeds":
                        float(np.std(bmeans)),
                    "n_seeds_bestval": len(bmeans),
                })
        _write_json_atomic(summary, os.path.join(ckpt_dir, "metrics.json"))
        for backend, s in summary["per_backend"].items():
            line = (f"[summary] {backend}: test PSNR "
                    f"{s['test_psnr_mean_over_seeds']:.2f} +/- "
                    f"{s['test_psnr_std_over_seeds']:.2f} dB over "
                    f"{s['n_seeds']} seed(s)")
            if "test_psnr_bestval_mean_over_seeds" in s:
                line += (f"; best-val "
                         f"{s['test_psnr_bestval_mean_over_seeds']:.2f}"
                         f" +/- "
                         f"{s['test_psnr_bestval_std_over_seeds']:.2f}"
                         f" dB")
            print(line, flush=True)

    # ---------------------------------------------------------- phases
    def load_checkpoint(self):
        self.results = self._completed_arms()
        if not self.results:
            raise FileNotFoundError(
                f"no completed (backend, seed) arms under "
                f"{checkpointsdir(self.args.experiment)}; run with "
                f"--phase train first.",
            )

    def visualize(self):
        apply_paper_style()
        plot_dir = plotsdir(self.args.experiment)
        os.makedirs(plot_dir, exist_ok=True)
        arms = self.results or self._completed_arms()
        if not arms:
            print("[visualize] nothing to plot yet.", flush=True)
            return
        self._write_summary()

        # -- training / validation curves --------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for (backend, seed), m in sorted(arms.items()):
            h = m["history"]
            c = BACKEND_COLORS[backend]
            axes[0].plot(h["steps"], h["loss"], color=c, lw=0.9, alpha=0.7,
                         label=backend if seed == min(
                             s for (b, s) in arms if b == backend) else None)
            axes[1].plot(h["eval_steps"], h["holdout_diff"], color=c,
                         lw=0.9, alpha=0.7)
            axes[2].plot(h["eval_steps"], h["val_psnr"], color=c,
                         lw=0.9, alpha=0.7)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("training loss")
        axes[0].legend(frameon=False)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("held-out critic difference")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("validation PSNR (dB)")
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        path = os.path.join(plot_dir, "training_curves.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {path}", flush=True)

        # -- shoulder screen ---------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for (backend, seed), m in sorted(arms.items()):
            h = m["history"]
            c = BACKEND_COLORS[backend]
            lab = backend if seed == min(
                s for (b, s) in arms if b == backend) else None
            axes[0].semilogy(h["eval_steps"], h["w_median"], color=c,
                             lw=0.9, alpha=0.7, label=lab)
            axes[1].plot(h["eval_steps"], h["w_frac_shoulder"], color=c,
                         lw=0.9, alpha=0.7)
            axes[2].plot(h["eval_steps"], h["w_frac_zero"], color=c,
                         lw=0.9, alpha=0.7)
        axes[0].axhline(SHOULDER_EDGE, color="k", ls="--", lw=0.8)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("median effective weight")
        axes[0].legend(frameon=False)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("fraction at/below shoulder edge")
        axes[1].set_ylim(-0.02, 1.02)
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("fraction exactly zero")
        axes[2].set_ylim(-0.02, 1.02)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        path = os.path.join(plot_dir, "shoulder_screen.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {path}", flush=True)

        # -- reconstruction examples -------------------------------------
        ckpt_dir = checkpointsdir(self.args.experiment)
        by_backend = {}
        for (backend, seed) in sorted(arms):
            if backend in by_backend:
                continue
            npz = os.path.join(
                ckpt_dir, f"{backend}_seed{seed}", "recon_examples.npz",
            )
            if os.path.isfile(npz):
                by_backend[backend] = np.load(npz)
        if by_backend:
            any_npz = next(iter(by_backend.values()))
            n_ex = int(any_npz["clean"].shape[0])
            n_rows = 2 + len(by_backend)
            fig, axes = plt.subplots(
                n_rows, n_ex, figsize=(2.2 * n_ex, 2.2 * n_rows),
                squeeze=False,
            )
            row_labels = ["clean", "degraded"] + list(by_backend)
            for j in range(n_ex):
                axes[0][j].imshow(_panel_img(any_npz["clean"][j]),
                                  cmap="gray", vmin=0, vmax=1)
                axes[1][j].imshow(_panel_img(any_npz["noisy"][j]),
                                  cmap="gray", vmin=0, vmax=1)
                for r, backend in enumerate(by_backend):
                    axes[2 + r][j].imshow(
                        _panel_img(by_backend[backend]["recon"][j]),
                        cmap="gray", vmin=0, vmax=1,
                    )
            for r, lab in enumerate(row_labels):
                axes[r][0].set_ylabel(lab)
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            path = os.path.join(plot_dir, "reconstructions.pdf")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved to {path}", flush=True)
