r"""Composite loss-landscape figure: columns = problems, rows = spaces.

One figure carries the landscape grammar repeated across the paper (CPFlow
2-D, 1-D Gumbel EBM, MiniBooNE EBM, HEPMASS EBM, Darcy PCP-Map): row 1 is
the constrained :math:`\theta` space shared by every method, row 2 the
lifted :math:`(\phi, b)` hypernetwork space, which only the lift inhabits.
The reading across columns is a plateau wherever the positivity constraint
binds and one clean basin where it does not (Darcy, the control column).

Every panel is a cached product of the repo's one landscape renderer
(``lift/utils/landscape/render_generic.py`` and the scripts built on it) in
the paper-wide hybrid frame, :math:`\Delta`\ fin + PCA-orth and
direct-anchored: the converged lift sits at the origin (gold star) and the
converged direct model at :math:`\alpha = 1`. Lifted panels are centered on
the converged hypernetwork with top-2 trajectory-PCA directions. This
script composes those NPZs and never evaluates a loss surface itself,
except under ``--precompute``, which builds a named cache under
``checkpoints/landscape_composite/`` and exits.

Provenance and frame coordinates for every panel are printed at compose
time: source NPZ path and mtime, and the final coordinates, which must sit
at (0,0) for the lift and (1,0) for direct softplus in the constrained
frame.

The figure is authored 1:1 for the page at 5.5 in wide, the paper's
single-column text width, and saved without a tight bbox so
``\includegraphics[width=\linewidth]`` neither shrinks nor stretches it.
That is also why there are no per-panel colorbars; see ``_paint``.

Usage::

    # one-time cache builds for the lifted panels
    python scripts/render_landscape_composite.py --precompute gumbel_lifted
    python scripts/render_landscape_composite.py --precompute ebm2d_lifted
    python scripts/render_landscape_composite.py --precompute miniboone_lifted
    python scripts/render_landscape_composite.py --precompute darcy_lifted

    # the HEPMASS lifted panel's cache, if absent. As written it needs
    # about 35 GB of host RAM (107 M lifted parameters x 22 snapshots in
    # float64):
    #   python scripts/compute_lifted_landscape_hepmass.py

    # compose
    python scripts/render_landscape_composite.py

Outputs ``figures/landscape_composite.pdf`` and a ``.png`` preview next to
it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for p in (_REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

# Palette source of truth: lift/utils/paper_style.PALETTE, which has no PGD
# key; #2ca02c is the paper-wide PGD color.
HYPER_COLOR = PALETTE["hypernet"]            # "#ff7f0e"
DIRECT_COLOR = PALETTE["direct_softplus"]    # "#d62728"
PGD_COLOR = "#2ca02c"

_CKPT = os.path.join(_REPO_ROOT, "data", "checkpoints")
_CACHE = os.path.join(_REPO_ROOT, "checkpoints", "landscape_composite")

# The 2-D gamma-mode EBM checkpoints were never copied into this repo; they
# live in the sibling ``lcmm`` repo and are read from there, read-only.
_FALLBACK_CKPT_ROOTS = [
    os.path.join(os.path.expanduser("~"), "Codes", "lcmm", "data",
                 "checkpoints"),
]


def _resolve_ckpt_dir(prefix: str, hash_substr: str = "") -> str:
    """Newest ckpt dir matching ``prefix`` across the repo + fallback roots.

    Mirrors ``render_landscape_two_spaces._find_ckpt_dir`` so the composite
    resolves the same directory the shipped figure did.  Returns a
    non-existent placeholder path (rather than raising) when nothing
    matches, so a missing column degrades to an annotated empty cell.
    """
    best: tuple[float, str] | None = None
    for root in [_CKPT, *_FALLBACK_CKPT_ROOTS]:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isdir(full) or not name.startswith(prefix):
                continue
            if hash_substr and hash_substr not in name:
                continue
            m = os.path.getmtime(full)
            if best is None or m > best[0]:
                best = (m, full)
    return best[1] if best is not None else os.path.join(_CKPT, prefix)


_CPFLOW2D_DIR = os.path.join(_CKPT, "experiments_cpflow_demo_2d_f1_symmetric")
# The fkl-direct run, the gallery objective. Seeds 0, 1 and 2 were trained
# in one invocation with trajectory snapshots and share the landscape NPZ.
_GUMBEL_DIR = os.path.join(
    _CKPT,
    "experiments_hypernet_vs_direct_1d_target_kind-gumbel_objective-"
    "fkl-direct_n_iters-3000_lr_ebm-0.001_lr_sampler-0.005_hidden_dim-128_"
    "nlayers-3_hyper_hidden_sizes-64-64-96_sampler_hyper_hidden_sizes-"
    "64-64-96_strong_c.893a04ae9b0c157d2c4a68d6006142ba1ad47166",
)
# 2-D gamma-mode ICNN-EBM (D=2). The hash pin is required: ``f036aa7d`` is
# the run carrying the Delta-fin + PCA-orth constrained NPZ, and the newer
# re-run does not have it.
_EBM2D_DIR = _resolve_ckpt_dir(
    "experiments_hypernet_vs_direct_2d_k1_target_kind-gamma_mode_2d_"
    "objective-fkl-direct_K-1_",
    hash_substr="f036aa7d",
)

# MiniBooNE (D = 43), a single-seed line, so this column comes from seed 0.
# Its ``..._direct_anchored_121.npz`` carries the lift and direct
# trajectories only, unlike HEPMASS's file of the same name, which also
# carries PGD. The ``seed0_landscape_pgd_anchored.npz`` beside it is a
# 31 x 31 grid in a PGD-anchored frame, not the delta_fin + PCA-orth
# convention, so reading it here would break comparability across columns.
_MINIBOONE_DIR = _resolve_ckpt_dir(
    "pgd_uci_miniboone_target_kind-uci_miniboone_K-1_",
    hash_substr="0abf822e",
)
# The lifted panel's source: the same configuration and seed as the lift in
# ``_MINIBOONE_DIR``, retrained with ``--save_hyper_snaps 1 --snap_every
# 200`` so the emitter's (phi, b) state is stored at 21 iterates instead of
# once. The pinned 0abf822e checkpoint holds one emitter snapshot and so has
# no (phi, b) trajectory to take principal directions of.
_MINIBOONE_LIFT_DIR = _resolve_ckpt_dir(
    "landscape_miniboone_lift_target_kind-uci_miniboone_K-1_",
)

_HEPMASS_DIR = os.path.join(
    _CKPT,
    "experiments_pgd_uci_hepmass_target_kind-uci_hepmass_K-1_objective-"
    "fkl-direct_data_path-_max_train_rows-0_n_iters-4000_lr_ebm-0.001_"
    "lr_sampler-0.005_hidden_dim-512_nlayers-5_hyper_hidden_sizes-64-64-96_"
    "sampler_hyper.b0929293841311db47eff4ffed48c9fb67bf0bca",
)

# The Darcy lift of both rows: the fan-in lift (``--pool_norm 1
# --readout_fanin_scale 1`` on the 50k recipe), retrained at seed 0 with
# emitter and constrained snapshots. Direct and PGD stay in the 84a09bf7
# archive, and the constrained cache ``darcy_seed0_landscape.npz`` was
# rebuilt with this lift at the origin.
_DARCY_LIFT_DIR = _resolve_ckpt_dir("landscape_darcy_fanin_lift_")

_OUT_PDF = os.path.join(
    _REPO_ROOT, "figures", "figures", "landscape_composite.pdf",
)

# See the seed-choice comment on the 1-D Gumbel column below.
_GUMBEL_SEED = 0

# ------------------------------------------------------------------ columns
#
# Each entry: header label; per-row {path, method -> traj key(s)}. A traj
# value is either one NPZ key (the full path, chronological) or a
# (traj_key, final_key) pair, since Darcy stores the converged point apart.
# ``lifted: None`` marks a column that is single-row by design.

COLUMNS = [
    dict(
        label="CPFlow 2-D\n(8-Gaussians)",
        constrained=dict(
            path=os.path.join(
                _CPFLOW2D_DIR, "seed0_landscape_constrained_121.npz"),
            arms={"hypernet": "hyper_traj", "direct": "direct_traj"},
        ),
        lifted=dict(
            path=os.path.join(
                _CPFLOW2D_DIR, "seed0_landscape_lifted_121.npz"),
            arms={"hypernet": "hyper_traj"},
        ),
    ),
    dict(
        # Seed 0, the default: its lift has converged by the last iterate
        # and its frame is the canonical r = 1.35, with both endpoints in
        # place.
        label="1-D Gumbel\n(ICNN-EBM)",
        constrained=dict(
            path=os.path.join(_GUMBEL_DIR, f"seed{_GUMBEL_SEED}_landscape.npz"),
            arms={"hypernet": "coords_hyper", "direct": "coords_direct"},
        ),
        lifted=dict(
            path=os.path.join(
                _CACHE, f"gumbel1d_seed{_GUMBEL_SEED}_landscape_lifted.npz"),
            arms={"hypernet": "traj_coords"},
            missing_note=("cache absent -- build with\n"
                          "--precompute gumbel_lifted"),
        ),
    ),
    dict(
        label="MiniBooNE\n($D=43$)",
        constrained=dict(
            path=os.path.join(
                _MINIBOONE_DIR, "seed0_landscape_direct_anchored_121.npz"),
            arms={"hypernet": "coords_hyper", "direct": "coords_direct"},
        ),
        # The lifted panel comes from the seed-0 lift retrain in
        # ``_MINIBOONE_LIFT_DIR``, not from the run the constrained row
        # reads, whose ``seed0_hypernet.pt`` stores one emitter snapshot
        # against 42 constrained ones. The frame is the HEPMASS column's
        # lifted convention.
        lifted=dict(
            path=os.path.join(_CACHE, "miniboone_seed0_landscape_lifted.npz"),
            arms={"hypernet": "coords_hyper"},
            missing_note=("cache absent -- build with\n"
                          "--precompute miniboone_lifted"),
        ),
    ),
    dict(
        label="HEPMASS\n($D=21$)",
        constrained=dict(
            path=os.path.join(
                _HEPMASS_DIR, "seed0_landscape_direct_anchored_121.npz"),
            arms={"hypernet": "coords_hyper", "direct": "coords_direct",
                  "pgd": "coords_pgd"},
        ),
        lifted=dict(
            path=os.path.join(_HEPMASS_DIR, "seed0_landscape_lifted.npz"),
            arms={"hypernet": "coords_hyper"},
            missing_note=("cache absent -- build with\n"
                          "compute_lifted_landscape_hepmass.py"),
        ),
    ),
    dict(
        label="Darcy\n($D=100$)",
        constrained=dict(
            path=os.path.join(_CACHE, "darcy_seed0_landscape.npz"),
            arms={"hypernet": ("traj_hypernet", "final_hypernet"),
                  "direct": ("traj_direct", "final_direct"),
                  "pgd": ("traj_pgd", "final_pgd")},
        ),
        # The control column's lifted panel comes from the seed-0 retrain
        # of the fan-in lift in ``_DARCY_LIFT_DIR``, the same run the
        # constrained row's lift comes from, on the same batch
        # (x_eval[:256]). The earlier archives store only the emitted
        # weights per snapshot, so they hold no (phi, b) trajectory.
        lifted=dict(
            path=os.path.join(_CACHE, "darcy_seed0_landscape_lifted.npz"),
            arms={"hypernet": "coords_hyper"},
            missing_note=("cache absent -- build with\n"
                          "--precompute darcy_lifted"),
        ),
    ),
]

_ROW_LABELS = [
    "constrained $\\boldsymbol{\\theta}$ space",
    "lifted $(\\boldsymbol{\\phi},\\,b)$ space",
]

_ARM_STYLE = {
    # method -> (color, linestyle, marker, mfc, final marker)
    "hypernet": (HYPER_COLOR, "-", "s", HYPER_COLOR, None),
    "direct": (DIRECT_COLOR, "--", "o", "none", "P"),
    "pgd": (PGD_COLOR, "-.", "^", "none", "X"),
}


# ------------------------------------------------------------------ loading

def _load_panel(spec: dict) -> dict | None:
    """Load one row-cell spec into {alphas, betas, Z, trajs, path, mtime}."""
    path = spec["path"]
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as f:
        files = set(f.files)
        out = dict(
            path=path,
            mtime=time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))),
            alphas=np.asarray(f["alphas"], dtype=np.float64),
            betas=np.asarray(f["betas"], dtype=np.float64),
            Z=np.asarray(f["Z"], dtype=np.float64),
            trajs={},
        )
        for arm, key in spec["arms"].items():
            if isinstance(key, tuple):
                traj_key, final_key = key
                if traj_key not in files or final_key not in files:
                    continue
                traj = np.asarray(f[traj_key], dtype=np.float64)
                final = np.asarray(f[final_key], dtype=np.float64)[None, :]
                out["trajs"][arm] = np.concatenate([traj, final], axis=0)
            else:
                if key not in files:
                    continue
                out["trajs"][arm] = np.asarray(f[key], dtype=np.float64)
    return out


# ----------------------------------------------------------------- painting

def _paint(ax, panel: dict, *, letter: str, show_xlabel: bool,
           show_ylabel: bool) -> None:
    """One landscape cell: filled log-contours + trajectory overlays.

    Each panel is shifted to its own minimum and shown on a log10 axis with
    vmax = min(p75, 3) and a 3.5-decade floor, so a lifted panel whose
    corners diverge does not drown its basin.

    No per-panel colorbar: every panel carries its own shift and scale, so
    one colorbar per panel would encode mutually incomparable axes and eat a
    third of the figure width. The caveat is stated once, in the figure note.

    Axis labels are drawn only on the outer frame (``show_xlabel`` /
    ``show_ylabel``): the frames are per-problem but the axis meaning is
    shared, so the leftmost column and the bottom row carry it.
    """
    # Imported here, not at module scope, so the backend selection in
    # ``main`` still runs before matplotlib resolves one.
    from matplotlib.ticker import MaxNLocator

    A, B = np.meshgrid(panel["alphas"], panel["betas"], indexing="ij")
    Z = np.array(panel["Z"], dtype=np.float64)
    # Non-finite grid points, from a Cholesky or eigvalsh failure off the
    # feasible set, are very bad rather than missing: paint them at the
    # finite maximum so the log-axis cap saturates them like their neighbors
    # instead of contourf leaving white holes.
    bad = ~np.isfinite(Z)
    if bad.any() and (~bad).any():
        Z[bad] = np.max(Z[~bad])
    log_Z = np.log10(Z - np.nanmin(Z) + 1e-8)
    finite = log_Z[np.isfinite(log_Z)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(min(np.percentile(finite, 75.0), 3.0)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    vmin = max(vmin, vmax - 3.5)
    levels = np.linspace(vmin, vmax, 25)
    ax.contourf(
        A, B, log_Z, levels=levels, cmap="viridis", extend="both",
        vmin=vmin, vmax=vmax,
    )

    for arm in ("pgd", "direct", "hypernet"):   # lift painted on top
        if arm not in panel["trajs"]:
            continue
        coords = panel["trajs"][arm]
        color, ls, marker, mfc, final_marker = _ARM_STYLE[arm]
        ax.plot(
            coords[:, 0], coords[:, 1], linestyle=ls, marker=marker,
            color=color, lw=1.0, ms=1.9, mfc=mfc,
            mec=("black" if mfc == color else color), mew=0.4,
            zorder=4, alpha=0.95,
        )
        if final_marker is not None:
            ax.plot(
                coords[-1, 0], coords[-1, 1], final_marker,
                color=color, ms=6.0, mec="black", mew=0.5, zorder=6,
            )
    # The converged lift is the frame origin, in every panel.
    ax.plot([0], [0], "*", color="gold", ms=8.5, mec="black", mew=0.5,
            zorder=7)

    ax.annotate(
        f"({letter})", xy=(0.04, 0.955), xycoords="axes fraction",
        ha="left", va="top", fontsize=7.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                  alpha=0.8),
    )
    if show_xlabel:
        ax.set_xlabel(r"$\alpha$", fontsize=8, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel(r"$\beta$", fontsize=8, labelpad=1.0)
    # Three ticks per axis is all a ~0.8 in panel can carry without the
    # labels running into one another.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(labelsize=7, length=2.0, width=0.6, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def _empty_cell(ax, note: str) -> None:
    # DejaVu, not the paper serif: cmr10 has no underscore glyph and
    # these notes quote script/flag names.
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=6.5,
            color="0.45", transform=ax.transAxes, family="DejaVu Sans")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_alpha(0.3)
        s.set_linewidth(0.6)
    ax.grid(False)


# ----------------------------------------------------------- sanity report

def _report(panel: dict | None, col_label: str, row: int) -> None:
    tag = f"[{col_label.splitlines()[0]:<16s} | row {row + 1}]"
    if panel is None:
        print(f"{tag} MISSING")
        return
    print(f"{tag} {os.path.relpath(panel['path'], _REPO_ROOT)}  "
          f"(mtime {panel['mtime']})")
    print(f"{tag}   grid {panel['Z'].shape[0]}x{panel['Z'].shape[1]}, "
          f"alpha [{panel['alphas'].min():+.3f}, {panel['alphas'].max():+.3f}]")
    for arm, coords in panel["trajs"].items():
        print(f"{tag}   {arm:<8s} start ({coords[0, 0]:+.4f}, "
              f"{coords[0, 1]:+.4f})  final ({coords[-1, 0]:+.4f}, "
              f"{coords[-1, 1]:+.4f})  [{len(coords)} pts]")
    if row == 0 and "hypernet" in panel["trajs"]:
        h = panel["trajs"]["hypernet"][-1]
        if not np.allclose(h, 0.0, atol=5e-3):
            print(f"{tag}   WARNING hypernet final off-origin: {h}")
    if row == 0 and "direct" in panel["trajs"]:
        d = panel["trajs"]["direct"][-1]
        if abs(d[0] - 1.0) > 2e-2 or abs(d[1]) > 2e-2:
            print(f"{tag}   WARNING direct final off (1,0): {d}")


# --------------------------------------------------------------- precompute

def _precompute_lifted_ebm(
    tag: str, ckpt_dir: str, seed: int, config_name: str, D: int,
    out_name: str, grid_n: int,
) -> str:
    """Build one lifted-space NPZ cache for an ICNN-EBM column.

    Delegates the computation to
    ``render_landscape_two_spaces._compute_lifted_landscape_ebm``, then
    stores its arrays under ``checkpoints/landscape_composite/``.
    """
    import torch
    from render_landscape_two_spaces import _compute_lifted_landscape_ebm

    with open(os.path.join(_REPO_ROOT, "configs", config_name)) as f:
        args_dict = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precompute {tag}] ckpt dir: {ckpt_dir}")
    print(f"[precompute {tag}] seed={seed} D={D} device={device} "
          f"grid_n={grid_n}")
    out = _compute_lifted_landscape_ebm(
        ckpt_dir, seed=seed, args_dict=args_dict, D=D, device=device,
        grid_n=grid_n,
    )
    os.makedirs(_CACHE, exist_ok=True)
    out_path = os.path.join(_CACHE, out_name)
    np.savez(
        out_path,
        alphas=out["alphas"], betas=out["betas"], Z=out["Z"],
        traj_coords=out["traj_coords"], snap_iters=out["snap_iters"],
        method=np.array("lifted_pca_trajectory"),
        source=np.array(os.path.basename(ckpt_dir)),
    )
    print(f"[precompute {tag}] Z range "
          f"[{out['Z'].min():.4g}, {out['Z'].max():.4g}]; "
          f"traj {out['traj_coords'].shape}")
    print(f"[precompute {tag}] wrote {out_path}")
    return out_path


def _precompute_gumbel_lifted(grid_n: int) -> str:
    """1-D Gumbel lifted-space NPZ cache (one-time, ~minutes)."""
    return _precompute_lifted_ebm(
        "gumbel_lifted", _GUMBEL_DIR, _GUMBEL_SEED,
        "experiments_hypernet_vs_direct_1d.json", D=1,
        out_name=f"gumbel1d_seed{_GUMBEL_SEED}_landscape_lifted.npz",
        grid_n=grid_n,
    )


def _precompute_miniboone_lifted(grid_n: int, max_snaps: int = 12) -> str:
    """MiniBooNE lifted-space NPZ cache (one-time, ~20 min at 121^2).

    ``_compute_lifted_landscape_ebm`` needs a grid log-Z and stops at D = 2,
    so this column goes through ``compute_lifted_landscape_hepmass.py``:
    flow-IS log-Z with the run's own converged sampler as proposal, centered
    on the converged emitter, top-2 trajectory PCA over the kept
    ``(phi, b)`` snapshots, Li-2018 filter normalization, and the run's own
    ``seed0_x_val_landscape.npy`` as the fixed evaluation batch. It runs in
    a subprocess: the checkpoint is about 10 GB, and the builder memory-maps
    it and materializes ``max_snaps`` snapshots.
    """
    import subprocess
    ckpt = os.path.join(_MINIBOONE_LIFT_DIR, "seed0_hypernet.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"{ckpt} missing -- the seed-0 lift retrain with emitter "
            "snapshots has not run")
    os.makedirs(_CACHE, exist_ok=True)
    out_path = os.path.join(_CACHE, "miniboone_seed0_landscape_lifted.npz")
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "compute_lifted_landscape_hepmass.py"),
        "--run_dir", _MINIBOONE_LIFT_DIR,
        "--config", os.path.join(
            _REPO_ROOT, "configs", "experiments_pgd_uci_miniboone.json"),
        "--target_kind", "uci_miniboone",
        "--seed", "0",
        "--grid_n", str(int(grid_n)),
        "--max_snaps", str(int(max_snaps)),
        "--out_path", out_path,
    ]
    print(f"[precompute miniboone_lifted] ckpt dir: {_MINIBOONE_LIFT_DIR}")
    print("[precompute miniboone_lifted] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    print(f"[precompute miniboone_lifted] wrote {out_path}")
    return out_path


def _precompute_darcy_lifted(grid_n: int) -> str:
    """Darcy lifted-space NPZ cache (one-time, ~15 min at 121^2).

    Runs ``compute_lifted_landscape_darcy.py`` in a subprocess on the newest
    ``landscape_darcy_lift_*`` archive: centered on the converged
    hypernetwork, top-2 trajectory PCA over the stored (phi, b) snapshots,
    Li-2018 filter normalization, PCP-Map NLL on the constrained panel's
    256-point held-out batch.
    """
    import subprocess
    os.makedirs(_CACHE, exist_ok=True)
    out_path = os.path.join(_CACHE, "darcy_seed0_landscape_lifted.npz")
    if not os.path.isfile(os.path.join(_DARCY_LIFT_DIR, "hypernet_seed0.pth")):
        raise FileNotFoundError(
            f"{_DARCY_LIFT_DIR}/hypernet_seed0.pth missing -- the seed-0 "
            "fan-in lift retrain has not run")
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "compute_lifted_landscape_darcy.py"),
        "--run_dir", _DARCY_LIFT_DIR,
        "--seed", "0", "--grid_n", str(int(grid_n)), "--out_path", out_path,
    ]
    print("[precompute darcy_lifted] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    print(f"[precompute darcy_lifted] wrote {out_path}")
    return out_path


def _precompute_ebm2d_lifted(grid_n: int) -> str:
    """2-D gamma-mode lifted-space NPZ cache (one-time, ~minutes)."""
    return _precompute_lifted_ebm(
        "ebm2d_lifted", _EBM2D_DIR, 0,
        "experiments_hypernet_vs_direct_2d_k1.json", D=2,
        out_name="ebm2d_seed0_landscape_lifted.npz",
        grid_n=grid_n,
    )


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_pdf", default=_OUT_PDF)
    parser.add_argument(
        "--out_png", default="",
        help="preview PNG; defaults to out_pdf with a .png suffix")
    parser.add_argument(
        "--precompute",
        choices=["gumbel_lifted", "ebm2d_lifted", "miniboone_lifted",
                 "darcy_lifted"],
        default="",
        help="build the named NPZ cache and exit (no figure)")
    parser.add_argument(
        "--gumbel_lifted_grid_n", type=int, default=25,
        help="grid size for --precompute gumbel_lifted (default 25, "
             "matching the two-spaces figure)")
    parser.add_argument(
        "--ebm2d_lifted_grid_n", type=int, default=21,
        help="grid size for --precompute ebm2d_lifted (default 21, "
             "matching the two-spaces figure's 2-D lifted panel)")
    parser.add_argument(
        "--miniboone_lifted_grid_n", type=int, default=121,
        help="grid size for --precompute miniboone_lifted (default 121, "
             "the constrained row's grid; each point is one emitter "
             "forward plus a flow-IS log-Z at D = 43, about 20 min in "
             "total on a shared GPU)")
    parser.add_argument(
        "--miniboone_lifted_max_snaps", type=int, default=12,
        help="emitter snapshots kept for the trajectory PCA of "
             "--precompute miniboone_lifted (default 12 = the HEPMASS "
             "lifted panel's count)")
    parser.add_argument(
        "--darcy_lifted_grid_n", type=int, default=121,
        help="grid size for --precompute darcy_lifted (default 121)")
    parser.add_argument(
        "--fig_width", type=float, default=5.5,
        help="figure width in inches. Default 5.5 = the ICLR "
             "single-column text width the paper includes this at, so "
             "the figure is authored 1:1 and the point sizes below are "
             "the sizes on the printed page.")
    parser.add_argument(
        "--panel_aspect", type=float, default=1.28,
        help="panel height / panel width. >1 gives the portrait panels "
             "the contours need once five columns share 5.5 in.")
    parser.add_argument(
        "--png_dpi", type=float, default=165.0,
        help="preview PNG dpi; 165 renders the 5.5 in figure at ~900 px, "
             "i.e. roughly the size it occupies on the page.")
    args = parser.parse_args()
    if not args.out_png:
        args.out_png = os.path.splitext(args.out_pdf)[0] + ".png"

    if args.precompute == "gumbel_lifted":
        _precompute_gumbel_lifted(args.gumbel_lifted_grid_n)
        return
    if args.precompute == "ebm2d_lifted":
        _precompute_ebm2d_lifted(args.ebm2d_lifted_grid_n)
        return
    if args.precompute == "miniboone_lifted":
        _precompute_miniboone_lifted(
            args.miniboone_lifted_grid_n, args.miniboone_lifted_max_snaps)
        return
    if args.precompute == "darcy_lifted":
        _precompute_darcy_lifted(args.darcy_lifted_grid_n)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_paper_style()
    # The paper style saves with a tight bbox, which would crop the canvas,
    # or expand it where an annotation overhangs, and so silently rescale the
    # figure at ``\includegraphics[width=\linewidth]``. "standard" is
    # matplotlib's no-crop setting, so the PDF is emitted at exactly
    # ``fig_width`` inches and lands on the page at 1:1.
    matplotlib.rcParams["savefig.bbox"] = "standard"

    # Geometry, in inches, at the size the figure is used. The paper
    # includes this at ``width=\linewidth`` in a single-column 5.5 in text
    # block, so the figure is authored at that width and the font sizes
    # below are the sizes a reader sees on the page.
    W = args.fig_width
    n_col = len(COLUMNS)
    # Room for the row label, the y tick labels and the beta axis label.
    left_in = 0.74
    right_in = 0.05
    gap_in = 0.235      # inter-column gap: two half tick labels
    panel_w = (W - left_in - right_in - (n_col - 1) * gap_in) / n_col
    panel_h = args.panel_aspect * panel_w   # modest portrait aspect
    top_in = 0.36       # two-line column headers
    row_gap_in = 0.30   # x tick labels of the top row
    bot_in = 0.98       # x label + two-row legend + two-line note
    H = top_in + 2 * panel_h + row_gap_in + bot_in

    fig, axes = plt.subplots(
        2, n_col, figsize=(W, H),
        gridspec_kw=dict(
            hspace=row_gap_in / panel_h, wspace=gap_in / panel_w,
            left=left_in / W, right=1.0 - right_in / W,
            top=1.0 - top_in / H, bottom=bot_in / H,
        ),
    )
    print(f"layout: figure {W:.2f} x {H:.2f} in, per-panel drawing area "
          f"{panel_w:.3f} x {panel_h:.3f} in (used at 1:1 on the page)")

    # Two spare letters, so adding a column does not exhaust the iterator.
    letters = iter("abcdefghijklm")
    print("=" * 72)
    print("landscape composite: panel provenance + frame sanity")
    print("=" * 72)
    # Load first, paint second: whether a column's constrained panel is the
    # bottom-most one it draws, and so carries the x label, depends on
    # whether the lifted cell below it resolved.
    loaded: dict[tuple[int, int], dict | None] = {}
    for row in range(2):
        for col, colspec in enumerate(COLUMNS):
            spec = colspec["constrained"] if row == 0 else colspec["lifted"]
            panel = None if spec is None else _load_panel(spec)
            if spec is not None:
                _report(panel, colspec["label"], row)
            loaded[(row, col)] = panel

    for row in range(2):
        for col, colspec in enumerate(COLUMNS):
            ax = axes[row, col]
            spec = colspec["constrained"] if row == 0 else colspec["lifted"]
            if spec is None:
                # No column is single-row at present; this keeps a
                # ``lifted=None`` column degrading to a per-column note
                # rather than an error.
                _empty_cell(ax, colspec.get(
                    "no_lifted_note", "constrained row only"))
                continue
            panel = loaded[(row, col)]
            if panel is None:
                _empty_cell(ax, spec.get(
                    "missing_note", "cache absent"))
                continue
            # Outer frame only: the leftmost column carries the y label,
            # the bottom-most drawn panel of each column the x label.
            _paint(ax, panel, letter=next(letters),
                   show_xlabel=(row == 1 or loaded[(1, col)] is None),
                   show_ylabel=(col == 0))

    # Column headers + row labels.
    for col, colspec in enumerate(COLUMNS):
        axes[0, col].annotate(
            colspec["label"], xy=(0.5, 1.04), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=7, annotation_clip=False,
        )
    for row in range(2):
        axes[row, 0].annotate(
            _ROW_LABELS[row], xy=(-0.52, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90, fontsize=8,
            annotation_clip=False,
        )

    # One shared legend and the per-panel-scale note, in two rows of three:
    # the six entries do not fit across 5.5 in on one line at a readable
    # size. Column-major fill puts each method's trajectory above its
    # converged marker, so the entries read as three methods rather than six
    # unrelated glyphs.
    handles = [
        Line2D([], [], color=HYPER_COLOR, ls="-", marker="s", ms=3.4,
               mec="black", mew=0.4, label="lift"),
        Line2D([], [], color="gold", ls="", marker="*", ms=8.5,
               mec="black", mew=0.5, label="converged lift (origin)"),
        Line2D([], [], color=DIRECT_COLOR, ls="--", marker="o", ms=3.4,
               mfc="none", label="direct softplus"),
        Line2D([], [], color=DIRECT_COLOR, ls="", marker="P", ms=6.0,
               mec="black", mew=0.5,
               label=r"converged direct ($\alpha=1$)"),
        Line2D([], [], color=PGD_COLOR, ls="-.", marker="^", ms=3.4,
               mfc="none", label="PGD"),
        Line2D([], [], color=PGD_COLOR, ls="", marker="X", ms=6.0,
               mec="black", mew=0.5, label="converged PGD"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, handlelength=2.0, columnspacing=1.4,
               handletextpad=0.5, labelspacing=0.35,
               bbox_to_anchor=(0.5, 0.30 / H))
    # Wrapped by hand: an auto-wrapped line would overflow the 5.5 in canvas.
    fig.text(
        0.5, 0.055 / H,
        "Axes are the slice coordinates $(\\alpha,\\beta)$ in per-problem "
        "frame units. Each panel is shifted to its own minimum and\ncolored "
        "on its own log$_{10}$ scale, so shade reads within a panel and "
        "never across panels; per-panel colorbars are omitted.",
        ha="center", va="bottom", fontsize=7, color="0.35",
        linespacing=1.35,
    )

    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches=None)
    fig.savefig(args.out_png, dpi=args.png_dpi, bbox_inches=None)
    plt.close(fig)
    print("=" * 72)
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
