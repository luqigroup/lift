"""Shared out-of-distribution detection utilities.

AUROC, the Nalisnick median-violation diagnostic, histogram plotting,
and the certificate ceiling, in one place so every OOD experiment
produces interchangeable artifacts.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .paper_style import palette


def compute_ood_metrics(
    logp_id: np.ndarray,
    logp_ood: np.ndarray,
) -> Dict[str, float]:
    """Standard OOD scoring + Nalisnick median-violation diagnostic.

    Convention: higher log-likelihood = more in-distribution. The
    label is 1 for ID and 0 for OOD; the score is the log-likelihood
    itself, so AUROC = P(score_ID > score_OOD).
    """
    from sklearn.metrics import roc_auc_score

    logp_id = np.asarray(logp_id).ravel()
    logp_ood = np.asarray(logp_ood).ravel()
    y = np.concatenate(
        [np.ones_like(logp_id), np.zeros_like(logp_ood)],
    )
    s = np.concatenate([logp_id, logp_ood])
    auroc = float(roc_auc_score(y, s))
    id_median = float(np.median(logp_id))
    ood_median = float(np.median(logp_ood))
    median_violation = float((logp_ood >= id_median).mean())
    return {
        "auroc": auroc,
        "id_median": id_median,
        "ood_median": ood_median,
        "id_mean": float(logp_id.mean()),
        "ood_mean": float(logp_ood.mean()),
        "median_violation": median_violation,
        "n_id": int(logp_id.size),
        "n_ood": int(logp_ood.size),
    }


def plot_id_ood_histogram(
    logp_id: np.ndarray,
    logp_ood: np.ndarray,
    *,
    out_path: str,
    id_label: str = "ID",
    ood_label: str = "OOD",
    model_label: str = "model",
    certificate_line: Optional[float] = None,
    n_bins: int = 60,
    figsize: Tuple[float, float] = (5.0, 3.5),
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Save an ID/OOD log-likelihood histogram with median lines and
    AUROC + median-violation in the title.

    ``certificate_line`` is the ceiling :math:`L^*` of
    :func:`certificate_ceiling`. A finite value adds a dashed vertical
    line, which is meaningful on EBM panels only; elsewhere pass
    ``None``.
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = compute_ood_metrics(logp_id, logp_ood)

    lo = float(min(np.percentile(logp_id, 0.5),
                   np.percentile(logp_ood, 0.5)))
    hi = float(max(np.percentile(logp_id, 99.5),
                   np.percentile(logp_ood, 99.5)))
    if certificate_line is not None and math.isfinite(certificate_line):
        lo = min(lo, float(certificate_line) - 1.0)
    bins = np.linspace(lo, hi, int(n_bins))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        logp_id, bins=bins, alpha=0.55, color=palette("id"),
        label=f"{id_label}\nmedian {metrics['id_median']:.1f}",
        density=True,
    )
    ax.hist(
        logp_ood, bins=bins, alpha=0.55, color=palette("ood"),
        label=f"{ood_label}\nmedian {metrics['ood_median']:.1f}",
        density=True,
    )
    ax.axvline(metrics["id_median"], color=palette("id"),
               linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(metrics["ood_median"], color=palette("ood"),
               linestyle=":", linewidth=1.0, alpha=0.7)
    if certificate_line is not None and math.isfinite(certificate_line):
        ax.axvline(
            float(certificate_line),
            color=palette("certificate"),
            linestyle="--",
            linewidth=1.5,
            label=fr"$L^* = {float(certificate_line):.1f}$",
        )

    ax.set_xlabel(r"$\log q(\mathbf{x})$ (higher $=$ more likely)")
    ax.set_ylabel("density")
    title = (
        f"{model_label}: {id_label} vs {ood_label}\n"
        f"AUROC $= {metrics['auroc']:.3f}$, "
        f"median violation $= {100 * metrics['median_violation']:.1f}\\%$"
    )
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def evaluate_ood_pair(
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    latents_id: np.ndarray,
    latents_ood: np.ndarray,
    *,
    out_dir: str,
    name_id: str = "id",
    name_ood: str = "ood",
    model_label: str = "model",
    certificate_line: Optional[float] = None,
    chunk: int = 2048,
    suffix: str = "",
) -> Dict[str, float]:
    """End-to-end OOD evaluation: chunked log-prob -> metrics ->
    histogram -> cached .npz.

    The cached arrays land at ``<out_dir>/ood_logp{suffix}.npz``
    with keys ``name_id`` and ``name_ood``; the histogram lands at
    ``<out_dir>/hist_ood_{name_ood}{suffix}.pdf``.
    """
    os.makedirs(out_dir, exist_ok=True)
    logp_id = _chunked_log_prob(log_prob_fn, latents_id, chunk=chunk)
    logp_ood = _chunked_log_prob(log_prob_fn, latents_ood, chunk=chunk)
    metrics = compute_ood_metrics(logp_id, logp_ood)

    npz_path = os.path.join(out_dir, f"ood_logp{suffix}.npz")
    np.savez(npz_path, **{name_id: logp_id, name_ood: logp_ood})

    hist_path = os.path.join(out_dir, f"hist_ood_{name_ood}{suffix}.pdf")
    plot_id_ood_histogram(
        logp_id, logp_ood,
        out_path=hist_path,
        id_label=name_id.upper(),
        ood_label=name_ood.upper(),
        model_label=model_label,
        certificate_line=certificate_line,
        metrics=metrics,
    )
    return metrics


def plot_id_two_ood_joint(
    logp_id: np.ndarray,
    logp_ood_a: np.ndarray,
    logp_ood_b: np.ndarray,
    *,
    out_path: str,
    id_label: str = "ID",
    ood_a_label: str = "OOD-A",
    ood_b_label: str = "OOD-B",
    model_label: str = "model",
    certificate_line: Optional[float] = None,
    n_bins: int = 60,
) -> Dict[str, Dict[str, float]]:
    """Two-panel histogram: ID against OOD-A on the left, ID against
    OOD-B on the right.
    """
    import matplotlib.pyplot as plt

    metrics_a = compute_ood_metrics(logp_id, logp_ood_a)
    metrics_b = compute_ood_metrics(logp_id, logp_ood_b)

    arrays = (logp_id, logp_ood_a, logp_ood_b)
    lo = float(min(np.percentile(a, 0.5) for a in arrays))
    hi = float(max(np.percentile(a, 99.5) for a in arrays))
    if certificate_line is not None and math.isfinite(certificate_line):
        lo = min(lo, float(certificate_line) - 1.0)
    bins = np.linspace(lo, hi, int(n_bins))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, logp_ood, lab, mets in (
        (axes[0], logp_ood_a, ood_a_label, metrics_a),
        (axes[1], logp_ood_b, ood_b_label, metrics_b),
    ):
        ax.hist(logp_id, bins=bins, alpha=0.55, color=palette("id"),
                label=f"{id_label}\nmedian {mets['id_median']:.1f}",
                density=True)
        ax.hist(logp_ood, bins=bins, alpha=0.55, color=palette("ood"),
                label=f"{lab}\nmedian {mets['ood_median']:.1f}",
                density=True)
        ax.axvline(mets["id_median"], color=palette("id"),
                   linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axvline(mets["ood_median"], color=palette("ood"),
                   linestyle=":", linewidth=1.0, alpha=0.7)
        if certificate_line is not None and math.isfinite(certificate_line):
            ax.axvline(float(certificate_line),
                       color=palette("certificate"),
                       linestyle="--", linewidth=1.5,
                       label=fr"$L^* = {float(certificate_line):.1f}$")
        ax.set_xlabel(r"$\log q(\mathbf{x})$")
        ax.set_title(
            f"{lab}: AUROC $= {mets['auroc']:.3f}$, "
            f"viol. $= {100 * mets['median_violation']:.1f}\\%$"
        )
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle(model_label, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {ood_a_label: metrics_a, ood_b_label: metrics_b}


def certificate_ceiling(c: float, D: int, r_star: float) -> float:
    r"""Certificate ceiling.

    The deterministic upper bound on per-input log-likelihood at radial
    distance ``r_star`` from the nearest EBM component center, under
    strong-convexity coefficient ``c``::

        L^* = (D / 2) * log(c / (2 pi)) - c * r_star**2 / 2

    The first term is the log-density of the Gaussian floor at its mode,
    constant in :math:`\bx` once the floor is fixed; the second is the
    strong-convexity decay.
    """
    if c <= 0.0:
        return float("-inf")
    return (
        0.5 * float(D) * math.log(float(c) / (2.0 * math.pi))
        - 0.5 * float(c) * float(r_star) ** 2
    )


def median_radial_distance(
    latents: np.ndarray,
    centres: np.ndarray,
) -> float:
    """Median over points of the Euclidean distance from a latent to its
    nearest EBM component center.

    This is the ``r_star`` that :func:`certificate_ceiling` takes.
    """
    latents = np.asarray(latents)
    centres = np.asarray(centres)
    diff = latents[:, None, :] - centres[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    nearest = np.sqrt(d2.min(axis=-1))
    return float(np.median(nearest))


def _chunked_log_prob(
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    arr: np.ndarray,
    *,
    chunk: int = 2048,
) -> np.ndarray:
    arr = np.asarray(arr)
    n = arr.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, int(chunk)):
        out[i : i + chunk] = np.asarray(log_prob_fn(arr[i : i + chunk]))
    return out
