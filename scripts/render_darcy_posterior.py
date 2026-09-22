"""Darcy PCP-Map posterior figure: truth, mean, std, credible band, landscape.

Renders from the npz cached by ``darcy_posterior_cache.py``; no GPU, no
re-inversion. KL coefficients are pushed through the same Stuart prior
the archive was generated with, so the fields are on the generator's own
basis rather than a re-derived one.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from projorg import gitdir
sys.path.insert(0, gitdir())
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402

ARMS = [("hypernet", "hypernet (lift)", PALETTE["hypernet"]),
        ("direct", "direct softplus", PALETTE["direct_softplus"]),
        ("pgd", "PGD on cone", "#2ca02c")]


def _load_kl(grid=40, K=10):
    """The Karhunen-Loeve prior the groundwater archive was generated against.

    Vendored into the package, so this resolves without a sibling checkout.
    ``DARCY_KL_PRIOR`` points at a file path instead, for an archive generated
    with a different copy.
    """
    override = os.environ.get("DARCY_KL_PRIOR", "")
    if override:
        spec = importlib.util.spec_from_file_location("kl_prior", override)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.StuartKLPrior(grid, K=K, alpha=0.0, s=1.1, sigma=1.0)
    from lift.dataset.kl_prior import StuartKLPrior
    return StuartKLPrior(grid, K=K, alpha=0.0, s=1.1, sigma=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--landscape_npz", default="")
    p.add_argument("--obs", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    z = np.load(a.cache)
    kl = _load_kl()
    fld = lambda xi: np.asarray(kl.reconstruct(np.asarray(xi, dtype=np.float64)))

    truth = fld(z["xi_true"][a.obs])
    fields = {k: np.stack([fld(x) for x in z[f"xi_{k}"][a.obs]])
              for k, _, _ in ARMS}

    apply_paper_style()
    fig = plt.figure(figsize=(11.2, 9.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.95],
                          width_ratios=[1, 1, 1, 1],
                          hspace=0.34, wspace=0.28)

    vmax = np.abs(truth).max()
    kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(truth, **kw); ax.set_title("(a) truth", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    for j, (key, lab, _) in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, j + 1])
        im = ax.imshow(fields[key].mean(0), **kw)
        ax.set_title(f"({chr(98+j)}) {lab}\nposterior mean", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, ax=fig.axes[-4:], fraction=0.030, pad=0.015)
    cb.set_label("log-permeability", fontsize=9)

    # posterior std, on one scale across the three methods
    stds = {k: fields[k].std(0) for k, _, _ in ARMS}
    smax = max(s.max() for s in stds.values())
    for j, (key, lab, _) in enumerate(ARMS):
        ax = fig.add_subplot(gs[1, j])
        im2 = ax.imshow(stds[key], cmap="magma", vmin=0.0, vmax=smax)
        ax.set_title(f"({chr(101+j)}) {lab}\nposterior std", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb2 = fig.colorbar(im2, ax=fig.axes[-3:], fraction=0.030, pad=0.015)
    cb2.set_label("posterior std (shared)", fontsize=8.5)

    # transect with credible bands
    ax = fig.add_subplot(gs[2, 0:2])
    row = truth.shape[0] // 2
    xs = np.linspace(0, 1, truth.shape[1])
    ax.plot(xs, truth[row], color="black", lw=1.6, label="truth", zorder=6)
    # The three 90 percent bands are visually indistinguishable, and filling
    # all three stacks them into a muddy color, so only the lift is filled and
    # the other two are drawn as dashed band edges over it.
    for key, lab, col in ARMS:
        s = fields[key][:, row, :]
        lo, hi = np.percentile(s, [5, 95], axis=0)
        if key == "hypernet":
            ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0,
                            label="90% band (lift)", zorder=1)
        else:
            for e in (lo, hi):
                ax.plot(xs, e, color=col, lw=0.8, ls="--", alpha=0.85,
                        zorder=2)
        ax.plot(xs, s.mean(0), color=col, lw=1.3, label=lab, zorder=5)
    ax.set_title("(h) mid-domain transect, 90% credible band", fontsize=9.5)
    ax.set_xlabel("$x_2$", fontsize=9)
    ax.set_ylabel("log-permeability", fontsize=9)
    ax.legend(fontsize=6.8, frameon=False, loc="lower center",
              ncol=3, columnspacing=1.0, handlelength=1.6,
              bbox_to_anchor=(0.5, -0.44))

    # landscape
    ax = fig.add_subplot(gs[2, 2:4])
    ls = a.landscape_npz or ""
    if ls and os.path.isfile(ls):
        n = np.load(ls)
        Z, al, be = n["Z"], n["alphas"], n["betas"]
        # The mapping the tabular landscapes use: log10 of the loss above the
        # in-plane minimum, so the basin structure survives a surface whose
        # edges run to hundreds of nats.
        S = np.log10(Z - Z.min() + 1e-3)
        cf = ax.contourf(al, be, S, levels=28, cmap="viridis")
        ax.contour(al, be, S, levels=28, colors="white",
                   linewidths=0.25, alpha=0.55)
        # Optimization trajectories, one per method, drawn over the surface,
        # each with its converged solution marked at the end. An npz written
        # without --snap_every carries no trajectory, and only the markers
        # are drawn.
        marks = {"hypernet": "*", "direct": "o", "pgd": "^"}
        for key, lab, col in ARMS:
            if f"traj_{key}" in n:
                t = np.atleast_2d(n[f"traj_{key}"])
                ax.plot(t[:, 0], t[:, 1], "-", color=col, lw=1.5, alpha=0.95,
                        zorder=6, solid_capstyle="round")
                ax.plot(t[0, 0], t[0, 1], "o", color="white", ms=4.5,
                        mec=col, mew=1.2, zorder=7)
            c = (n[f"final_{key}"] if f"final_{key}" in n
                 else np.atleast_2d(n["traj_coords"])[
                     [k for k, _, _ in ARMS].index(key)])
            # direct and PGD converge to the same point; direct is drawn
            # larger and underneath so both remain visible.
            ms = {"hypernet": 17, "direct": 13, "pgd": 7}[key]
            zo = {"hypernet": 10, "direct": 8, "pgd": 9}[key]
            ax.plot([c[0]], [c[1]], marks[key], color=col, ms=ms,
                    mec="black", mew=0.7, zorder=zo, label=lab)
        ax.set_title(r"(i) direct-anchored slice ($D=100$)", fontsize=9.5)
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        ax.set_ylabel(r"$\beta$", fontsize=9)
        ax.legend(fontsize=6.8, frameon=False, loc="lower center", ncol=3,
                  columnspacing=1.0, handlelength=1.4,
                  bbox_to_anchor=(0.5, -0.36))
        cb3 = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        cb3.set_label(r"$\log_{10}(\mathrm{loss}-\mathrm{min\,loss})$",
                      fontsize=8)

        # Zoomed inset over the direct and PGD cluster, on the same plane
        # with a mesh about 50 times finer. Without it the two baselines are
        # one marker and their trajectories a single pixel.
        if "inset_Z" in n:
            iZ, ia, ib = n["inset_Z"], n["inset_alphas"], n["inset_betas"]
            axi = ax.inset_axes([0.605, 0.055, 0.355, 0.355])
            axi.contourf(ia, ib, np.log10(iZ - iZ.min() + 1e-6),
                         levels=22, cmap="viridis")
            for key, _, col in ARMS:
                if f"traj_{key}" in n:
                    t = np.atleast_2d(n[f"traj_{key}"])
                    axi.plot(t[:, 0], t[:, 1], "-", color=col, lw=1.1,
                             alpha=0.95, zorder=6)
                c = n[f"final_{key}"]
                ms = {"hypernet": 11, "direct": 8, "pgd": 5}[key]
                zo = {"hypernet": 10, "direct": 8, "pgd": 9}[key]
                axi.plot([c[0]], [c[1]], marks[key], color=col, ms=ms,
                         mec="black", mew=0.6, zorder=zo)
            axi.set_xlim(ia.min(), ia.max()); axi.set_ylim(ib.min(), ib.max())
            axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
            for sp in axi.spines.values():
                sp.set_edgecolor("black"); sp.set_linewidth(0.9)
            axi.set_title("zoom: direct / PGD (own scale)", fontsize=6.2,
                          pad=2.0,
                          bbox=dict(facecolor="white", alpha=0.88,
                                    edgecolor="none",
                                    boxstyle="round,pad=0.18"))
            ax.indicate_inset_zoom(axi, edgecolor="black", alpha=0.85,
                                   linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "landscape NPZ\nnot supplied", ha="center",
                va="center", fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()

    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print("Saved to", a.out)
    for key, _, _ in ARMS:
        err = np.linalg.norm(fields[key].mean(0) - truth) / np.linalg.norm(truth)
        print("  %-9s posterior-mean rel err %.3f   mean std %.4f"
              % (key, err, stds[key].mean()))


if __name__ == "__main__":
    main()
