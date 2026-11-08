r"""Shoulder occupancy, entry and exit hazards, and dwell, tracked per step.

A constrained coordinate is on the shoulder when the softplus derivative has
collapsed, ``psi'(w) = sigmoid(w) <= 0.05``, i.e. latent
``w <= logit(0.05) = -2.9444``; on the emitted weight the same condition reads
``theta <= softplus(-2.9444) = 0.0512933``. The constrained set is
``z_layers[i].weight`` for every hidden layer plus ``output_layer.weight``;
input skips and biases are excluded.

The tracker records, on a grid of thresholds ``tau`` and identically for
every construction, the occupancy and the entry and exit hazards at every
gradient step, so ``tau = 0`` (exact zeros) and the softplus shoulder are two
points of one curve. The dwell is reported as ``1 / exit_hazard``; the mean
observed episode length, which saturates at the window, is kept beside it
with the censored fraction. The per-step displacement of the emitted weight
is recorded as well.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

THR_LATENT = -2.9444389791664403                            # logit(0.05)
THR_EMITTED = float(F.softplus(torch.tensor(THR_LATENT)))   # 0.0512933

DEFAULT_TAUS: tuple[float, ...] = (
    0.0, 1e-4, 1e-3, 1e-2, THR_EMITTED, 0.1, 0.2,
)


def constrained_names(params: Iterable[str]) -> list[str]:
    """The constrained (non-negative) weight names, sorted.

    Accepts prefixed names (``components.0.z_layers.0.weight``) and the bare
    names of a single :class:`~lift.models.icnn.ICNN`.
    """
    return sorted(
        k for k in params
        if k.endswith(".weight")
        and ("z_layers." in k or k.endswith("output_layer.weight"))
    )


def constrained_vector(theta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten the constrained weights of one emitted-parameter dict."""
    keys = constrained_names(theta.keys())
    if not keys:
        raise KeyError(f"no constrained weights among {sorted(theta)[:6]}...")
    return torch.cat([theta[k].reshape(-1) for k in keys]).detach()


class _ThresholdState:
    """Per-threshold counters, kept on device until ``summary()``."""

    __slots__ = ("tau", "series", "prev", "counts", "episodes",
                 "inside_steps")

    def __init__(self, tau: float) -> None:
        self.tau = float(tau)
        self.series: list[torch.Tensor] = []
        self.prev: Optional[torch.Tensor] = None
        self.counts: Optional[torch.Tensor] = None   # [n_in, left, out, in]
        self.episodes: Optional[torch.Tensor] = None
        self.inside_steps: Optional[torch.Tensor] = None

    def update(self, w: torch.Tensor, inside: torch.Tensor) -> None:
        self.series.append(inside.double().mean())
        if self.episodes is None:
            self.episodes = inside.long().clone()
            self.inside_steps = torch.zeros_like(inside, dtype=torch.long)
            self.counts = torch.zeros(4, dtype=torch.long, device=w.device)
        else:
            entered = inside & ~self.prev
            self.episodes += entered.long()
            self.counts += torch.stack([
                self.prev.sum(),
                (self.prev & ~inside).sum(),
                (~self.prev).sum(),
                entered.sum(),
            ]).long()
        self.inside_steps += inside.long()
        self.prev = inside

    def summary(self, n_constrained: int) -> dict:
        s = torch.stack(self.series).cpu().numpy()
        n_in, n_left, n_out, n_entered = (
            int(v) for v in self.counts.cpu().tolist()
        )
        ever = int((self.episodes > 0).sum())
        total_inside = int(self.inside_steps.sum())
        total_ep = int(self.episodes.sum())
        open_at_end = int(self.prev.sum())
        nan = float("nan")
        exit_hazard = n_left / n_in if n_in else nan
        return {
            "tau": self.tau,
            "occupancy_init": float(s[0]),
            "occupancy_final": float(s[-1]),
            "occupancy_max": float(s.max()),
            "occupancy_mean": float(s.mean()),
            "exit_hazard": exit_hazard,
            "entry_hazard": n_entered / n_out if n_out else nan,
            "hazard_dwell_steps": (
                1.0 / exit_hazard
                if (exit_hazard == exit_hazard and exit_hazard > 0)
                else float("inf") if exit_hazard == 0.0 else nan
            ),
            "mean_observed_episode_steps": (
                total_inside / total_ep if total_ep else nan
            ),
            "censored_frac": open_at_end / total_ep if total_ep else nan,
            "survival_per_step": (
                1.0 - n_left / n_in if n_in else nan
            ),
            "episodes_per_visiting_coord": total_ep / ever if ever else nan,
            "frac_coords_ever_inside": ever / n_constrained
            if n_constrained else 0.0,
            "monotone_nondecreasing": bool(np.all(np.diff(s) >= -1e-12)),
            "n_inside_steps_observed": n_in,
            "n_exits_observed": n_left,
            "series": [float(v) for v in s],
        }


class ShoulderTracker:
    """Per-step occupancy, entry and exit hazards, dwell and displacement.

    ``update`` is called once per gradient step with the emitted parameter
    dict. ``summary()`` returns the primary threshold's numbers as flat keys
    and the full grid under ``"hazard_curve"``.
    """

    def __init__(
        self,
        taus: Sequence[float] = DEFAULT_TAUS,
        primary_tau: float = THR_EMITTED,
    ) -> None:
        taus = tuple(sorted({float(t) for t in taus} | {float(primary_tau)}))
        self.primary_tau = float(primary_tau)
        self._states = [_ThresholdState(t) for t in taus]
        self._tau_vec: Optional[torch.Tensor] = None
        self.median: list[torch.Tensor] = []
        self.disp_median: list[torch.Tensor] = []
        self.disp_on_shoulder: list[torch.Tensor] = []
        self.frac_zero_disp: list[torch.Tensor] = []
        self._prev_w: Optional[torch.Tensor] = None
        self.n_constrained = 0
        keys = [f"{st.tau:.6g}" for st in self._states]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"threshold grid collides under 6-digit formatting: {keys}",
            )

    def update(self, theta: Dict[str, torch.Tensor]) -> None:
        w = constrained_vector(theta).double()
        self.n_constrained = int(w.numel())
        self.median.append(w.median())
        if self._tau_vec is None:
            self._tau_vec = torch.tensor(
                [st.tau for st in self._states],
                dtype=w.dtype, device=w.device,
            )
        inside_all = w.unsqueeze(0) <= self._tau_vec.unsqueeze(1)
        prev_primary = None
        for i, st in enumerate(self._states):
            if abs(st.tau - self.primary_tau) < 1e-12:
                prev_primary = st.prev
            st.update(w, inside_all[i])

        nan = torch.tensor(float("nan"), dtype=w.dtype, device=w.device)
        if self._prev_w is None:
            self.disp_median.append(nan)
            self.disp_on_shoulder.append(nan)
            self.frac_zero_disp.append(nan)
        else:
            d = (w - self._prev_w).abs()
            self.disp_median.append(d.median())
            # A masked mean without boolean indexing keeps the update on
            # device, with no host synchronization per step.
            if prev_primary is None:
                self.disp_on_shoulder.append(nan)
            else:
                m = prev_primary.to(d.dtype)
                cnt = m.sum()
                self.disp_on_shoulder.append(
                    torch.where(cnt > 0, (d * m).sum() / cnt.clamp(min=1), nan),
                )
            self.frac_zero_disp.append((d == 0).double().mean())
        self._prev_w = w

    def summary(self) -> dict:
        if self.n_constrained == 0:
            return {"n_steps": 0}
        curve = {
            f"{st.tau:.6g}": st.summary(self.n_constrained)
            for st in self._states
        }
        primary = curve[f"{self.primary_tau:.6g}"]

        def _list(v):
            return [float(x) for x in torch.stack(v).cpu()] if v else []

        def late(v):
            a = np.asarray(v[-100:] if v else [np.nan], dtype=np.float64)
            a = a[np.isfinite(a)]
            return float(np.median(a)) if a.size else float("nan")

        med = _list(self.median)
        disp_med = _list(self.disp_median)
        disp_sh = _list(self.disp_on_shoulder)
        zero_disp = _list(self.frac_zero_disp)
        out = {
            "n_steps": len(med),
            "n_constrained": self.n_constrained,
            "primary_tau": self.primary_tau,
            "hazard_curve": curve,
            "median_series": med,
            "disp_median_series": disp_med,
            "disp_on_shoulder_series": disp_sh,
            "frac_zero_disp_series": zero_disp,
            "disp_median_late": late(disp_med),
            "disp_on_shoulder_late": late(disp_sh),
            "frac_zero_disp_late": late(zero_disp),
        }
        for k in ("occupancy_init", "occupancy_final", "occupancy_max",
                  "occupancy_mean", "exit_hazard", "entry_hazard",
                  "hazard_dwell_steps", "mean_observed_episode_steps",
                  "censored_frac", "survival_per_step",
                  "episodes_per_visiting_coord", "frac_coords_ever_inside",
                  "monotone_nondecreasing", "n_inside_steps_observed",
                  "n_exits_observed"):
            out[k] = primary[k]
        out["leave_rate"] = primary["exit_hazard"]
        out["occupancy_series"] = primary["series"]
        zero_key = "0"
        if zero_key in curve:
            out["frac_exact_zero_init"] = curve[zero_key]["occupancy_init"]
            out["frac_exact_zero_final"] = curve[zero_key]["occupancy_final"]
            out["frac_exact_zero_series"] = curve[zero_key]["series"]
        # Renewal consistency: entry hazard times the hazard dwell should
        # reproduce the mean occupancy.
        h_in = primary["entry_hazard"]
        d = primary["hazard_dwell_steps"]
        if np.isfinite(h_in) and np.isfinite(d):
            pred = h_in * d / (1.0 + h_in * d)
            out["renewal_occupancy_pred"] = float(pred)
            out["renewal_residual"] = float(pred - primary["occupancy_mean"])
        return out
