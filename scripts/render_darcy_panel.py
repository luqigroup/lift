"""The Darcy transport map as one full-width row of four cells.

  1. the true log-permeability field of one held-out observation;
  2. the lift's posterior mean over the amortized samples, with the
     posterior standard deviation as a corner inset on its own scale;
  3. the mid-domain transect: truth in black, the lift's posterior mean and
     its 90% credible band;
  4. the training objective, or the held-out NLL under --curve val, for the
     three constructions at one seed, with a framed table reading each at
     its best held-out value and direct softplus's shoulder occupancy as
     the subtitle.

Truth and mean share one symmetric color scale, set by the truth field.

Everything is read from caches, so there is no training and no GPU here.
``--cache`` is the posterior-sample archive, and its ``source_run`` and
``source_lift_run`` fields are asserted against ``--archive`` (direct and
PGD) and ``--lift_archive`` (the lift). The figure is written to
``--out_dir`` as ``darcy_panel.{pdf,png}`` and mirrored to ``--mirror_to``.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys
import numpy as np, matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402
from render_darcy_posterior import _load_kl  # noqa: E402
from render_tabular_val_loss_multiseed import _draw_holdout, best_val, holdout_order  # noqa: E402  one implementation of the framed held-out table and the best held-out read

ARMS = [("direct", "direct softplus"), ("pgd", "PGD"), ("hypernet", "lift")]


def _text_box_rot(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs, rotation=90); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=f"{_REPO}/data/analysis/darcy_post_seed0_fanin.npz")
    p.add_argument("--archive", default=(sorted(glob.glob(f"{_REPO}/data/checkpoints/experiments_pcpmap_darcy_*785a2e5d*")) or [""])[-1])
    p.add_argument("--lift_archive", default=(sorted(glob.glob(f"{_REPO}/data/checkpoints/darcy_improve_*6641e617*")) or [""])[-1],
                   help="archive holding the lift (fan-in readout and standardized summary); empty = same as --archive")
    p.add_argument("--obs", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=f"{_REPO}/figures")
    p.add_argument("--mirror_to", default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5)
    p.add_argument("--tile", type=float, default=0.95, help="side of the square image cells, inches")
    p.add_argument("--curve", choices=["train", "val"], default="train",
                   help="cell 4: the training objective (default) or the held-out NLL along training")
    p.add_argument("--holdout", choices=["table", "block", "ends"], default="table", help="where the final held-out NLL of each construction goes")
    p.add_argument("--skip_frac", type=float, default=0.25, help="--curve train: the y-range is taken over the run after this fraction of it (a quarter: the objective is logged every 100 iterations and its opening descent is over by 1,000)")
    p.add_argument("--std_inset", type=float, default=0.40, help="side of the std inset as a fraction of the tile (0 = no inset, no std shown)")
    a = p.parse_args()
    apply_paper_style(); matplotlib.rcParams["savefig.bbox"] = "standard"; matplotlib.rcParams["axes.grid"] = False
    col = {"hypernet": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c", "grey": "0.35"}
    dash = {"hypernet": (0, ()), "direct": (0, (5, 3.0)), "pgd": (3.0, (1.6, 6.4))}
    zo = {"hypernet": 3.0, "direct": 3.6, "pgd": 4.0}
    lwk = {"hypernet": 2.4, "direct": 1.0, "pgd": 1.0}

    # ---- data ---------------------------------------------------------------
    z = np.load(a.cache); kl = _load_kl()
    fld = lambda xi: np.asarray(kl.reconstruct(np.asarray(xi, dtype=np.float64)))
    truth = fld(z["xi_true"][a.obs])
    samples = np.stack([fld(x) for x in z["xi_hypernet"][a.obs]])            # (n_samples, 40, 40)
    mean_l, std_l = samples.mean(0), samples.std(0)
    R = json.load(open(a.archive + "/results.json"))
    RL = json.load(open(a.lift_archive + "/results.json")) if a.lift_archive else R
    for _extra in sorted(glob.glob(f"{_REPO}/data/checkpoints/darcy_improve_*770fbc2a*")):
        if os.path.isfile(_extra + "/results.json"):
            _seen = {int(r["seed"]) for r in RL if r["backend"] == "hypernet"}
            RL = list(RL) + [r for r in json.load(open(_extra + "/results.json"))
                             if r["backend"] == "hypernet" and int(r["seed"]) not in _seen]
    rec = {k: next(r for r in (RL if k == "hypernet" else R) if r["backend"] == k and int(r["seed"]) == a.seed) for k, _ in ARMS}

    # ---- provenance: every number this figure draws, with its archive -------
    src_run, src_lift = str(z["source_run"]), str(z["source_lift_run"]) if "source_lift_run" in z.files else ""
    assert os.path.basename(a.archive) == src_run, f"cache was drawn from {src_run}, not {os.path.basename(a.archive)}"
    if src_lift: assert os.path.basename(a.lift_archive) == src_lift, f"cache lift arm is {src_lift}, not {os.path.basename(a.lift_archive)}"
    print("PROVENANCE")
    print(f"  posterior cache  {os.path.relpath(a.cache, _REPO)}  obs {a.obs} of {z['xi_true'].shape[0]}, {samples.shape[0]} amortized draws, field {truth.shape[0]}x{truth.shape[1]}")
    print(f"  direct / PGD     {os.path.basename(a.archive)[-40:]}  (hash {os.path.basename(a.archive).split('.')[-1][:8]})")
    print(f"  lift             {os.path.basename(a.lift_archive)[-40:]}  (hash {os.path.basename(a.lift_archive).split('.')[-1][:8]})")
    for k, lab in ARMS:
        h = rec[k]["history"]
        print(f"  {lab:16s} seed {a.seed}: final held-out NLL {rec[k]['final_eval_nll']:.3f}  (curve {len(h['iters'])} evals, iters {h['iters'][0]}..{h['iters'][-1]}, "
              f"first {h['eval_nll'][0]:.2f}, min {min(h['eval_nll']):.3f}), shoulder occupancy {float(rec[k]['shoulder_occupancy']):.4f}")
    for src, name in ((R, "direct/PGD archive"), (RL, "lift archive")):
        per = {}
        for r in src: per.setdefault(r["backend"], []).append(float(r["final_eval_nll"]))
        print(f"  {name} seeds: " + "; ".join(f"{b} n={len(v)} {np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}" for b, v in sorted(per.items())))

    # ---- layout -------------------------------------------------------------
    W = a.fig_width
    strip_entries = [("lift", col["hypernet"], 1.2, dash["hypernet"]), ("PGD", col["pgd"], 1.2, dash["pgd"]), ("direct softplus", col["direct"], 1.2, dash["direct"])]   # bottom to top: the order of the held-out table, read downward
    STRIP_FS = 6.6; strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.44, 0.30, 0.045, 0.04, 0.06
    fig = plt.figure(figsize=(W, 2.0)); strip_text_in = max(_text_box_rot(fig, t, STRIP_FS)[0] for t, _, _, _ in strip_entries); plt.close(fig)
    tail_in = 0.0   # the framed table sits inside the loss panel, whose upper right is free once the opening descent leaves the axis
    right_in = tail_in + strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in   # strip_gap_in also holds the NLL end labels
    img_left_in, tile_gap, tr_left_in, nll_left_in = 0.05, 0.05, 0.42, 0.46          # y-tick room before the transect / before the NLL panel
    tile = a.tile
    pw = (W - right_in - img_left_in - 2 * tile - tile_gap - tr_left_in - nll_left_in) / 2
    top_in, title_in, bot_in = 0.03, 0.14, 0.33
    panel_h = tile
    H = top_in + title_in + panel_h + bot_in
    fig = plt.figure(figsize=(W, H)); y0 = bot_in / H

    # cells 1-2: truth and the lift's posterior mean on one symmetric scale
    vmax = float(np.abs(truth).max()); kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    for j, (title, img, c) in enumerate([("truth", truth, col["grey"]), ("lift, posterior mean", mean_l, col["hypernet"])]):
        ax = fig.add_axes([(img_left_in + j * (tile + tile_gap)) / W, y0, tile / W, tile / H])
        ax.imshow(img, **kw); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_edgecolor(c); s_.set_linewidth(1.1)
        ax.set_title(title, fontsize=7.0, pad=2.0, color=c if c != col["grey"] else "0.25")
        if j == 1 and a.std_inset > 0:                                              # posterior sd as a corner inset, own scale
            f = a.std_inset; pad = 0.035
            axi = ax.inset_axes([1 - f - pad, pad, f, f])
            axi.imshow(std_l, cmap="magma", vmin=0.0, origin="lower"); axi.set_xticks([]); axi.set_yticks([])
            for s_ in axi.spines.values(): s_.set_edgecolor("white"); s_.set_linewidth(1.1)
            axi.text(0.5, 0.985, "sd", transform=axi.transAxes, ha="center", va="top", fontsize=5.8, color="white",
                     bbox=dict(facecolor="black", alpha=0.45, pad=0.8, lw=0))
    print(f"  colour scales: log-permeability +-{vmax:.2f} (truth and mean share it); sd inset 0 to {float(std_l.max()):.2f} (magma)")

    # cell 3: the mid-domain transect
    x_tr = img_left_in + 2 * tile + tile_gap + tr_left_in
    axt = fig.add_axes([x_tr / W, y0, pw / W, panel_h / H]); row = truth.shape[0] // 2; xg = np.linspace(0, 1, truth.shape[1])
    sec = samples[:, row, :]; lo, hi = np.percentile(sec, [5, 95], axis=0)
    axt.fill_between(xg, lo, hi, color=col["hypernet"], alpha=0.22, lw=0, zorder=1)
    axt.plot(xg, sec.mean(0), color=col["hypernet"], lw=1.2, zorder=3); axt.plot(xg, truth[row], color="black", lw=1.0, zorder=4)
    axt.set_xlim(0, 1); axt.set_xticks([0, 0.5, 1]); axt.set_xticklabels(["0", "0.5", "1"])
    tlo, thi = float(min(lo.min(), truth[row].min())), float(max(hi.max(), truth[row].max())); tpad = 0.07 * (thi - tlo)
    axt.set_ylim(tlo - tpad, thi + tpad)                                            # the band must not sit on the bottom spine
    axt.tick_params(labelsize=7.0, length=2.2, pad=1.2); axt.set_xlabel("$x_2$", fontsize=8.0, labelpad=0.5)
    axt.set_ylabel("log-perm.", fontsize=8.0, labelpad=2)
    axt.set_title("mid-domain transect", fontsize=7.0, pad=2.0, color="0.25")
    for s_ in ("top", "right"): axt.spines[s_].set_visible(False)
    cov = float(np.mean((truth[row] >= lo) & (truth[row] <= hi)))
    print(f"  transect: row {row} of {truth.shape[0]}, truth inside the 90% band on {cov:.2f} of the row; band width median {float(np.median(hi - lo)):.3f}")
    print(f"  posterior shrinkage at obs {a.obs}: corr(mean, truth) {float(np.corrcoef(mean_l.ravel(), truth.ravel())[0, 1]):.3f}, "
          f"sd of the mean field {float(mean_l.std()):.3f} against sd of the truth {float(truth.std()):.3f} and of a single draw {float(samples.std(axis=(1, 2)).mean()):.3f} "
          f"(33 observations of a 100-coefficient field: the mean is the shrunk estimate, the draws carry the prior amplitude)")

    # cell 4: the objective along training (or the held-out NLL, --curve val), the three constructions at one seed
    x_nll = x_tr + pw + nll_left_in
    ax = fig.add_axes([x_nll / W, y0, pw / W, panel_h / H]); ends = []; ymin, ymax = np.inf, -np.inf
    for key, _ in ARMS:
        h = rec[key]["history"]; x = np.asarray(h["iters"], float); y = np.asarray(h["train_loss" if a.curve == "train" else "eval_nll"], float)
        ax.plot(x, y, color=col[key], lw=lwk[key], linestyle=dash[key], zorder=zo[key])
        t = y[int(a.skip_frac * len(y)):] if a.curve == "train" else y     # the y-range skips the opening descent
        ymin, ymax = min(ymin, t.min()), max(ymax, t.max())
        src = RL if key == "hypernet" else R
        # Each seed enters at its lowest held-out NLL, not its last one.
        vals = [best_val(r["history"]["eval_nll"], r["history"]["iters"]) for r in src
                if r["backend"] == key and np.isfinite(float(r["final_eval_nll"]))]
        vals = [v for v in vals if np.isfinite(v)]
        ends.append((float(np.median(vals)), key))
        print(f"  {key:9s} framed table: median best held-out {np.median(vals):.3f} over {len(vals)} seeds "
                f"(drawn seed {a.seed}: {best_val(rec[key]['history']['eval_nll'], rec[key]['history']['iters']):.3f}, "
                f"final {float(rec[key]['final_eval_nll']):.3f})")
    pad = 0.04 * (ymax - ymin); ymin, ymax = ymin - pad, ymax + pad + (0.85 if a.holdout in ("block", "table") else 0.0) * (ymax - ymin)   # the block or table clears the opening descent
    ax.set_ylim(ymin, ymax); ax.set_xlim(0, float(x.max()))
    ax.set_xticks([0, 2000, 4000]); ax.set_xticklabels(["0", "2k", "4k"])
    if a.holdout == "ends":
        ends = sorted(ends); ys = [e[0] for e in ends]; step = 0.13 * (ymax - ymin)      # three labels must clear each other
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < step: ys[k] = ys[k - 1] + step
        for (fin, key), yy in zip(ends, ys):
            ax.annotate(f"{fin:.2f}", xy=(x.max(), fin), xytext=(1.04, yy), textcoords=("axes fraction", "data"), xycoords="data",
                        fontsize=7.0, color=col[key], ha="left", va="center", clip_on=False,
                        arrowprops=dict(arrowstyle="-", color=col[key], lw=0.5, alpha=0.6, shrinkA=0, shrinkB=1))
    elif a.holdout == "table":
        # a framed legend of each construction's best held-out NLL
        _draw_holdout(fig, ax, "table", holdout_order(ends), col, dash, lambda v: f"{v:.2f}", (x_nll, y0, pw, panel_h), (W, H), inside=True)
    else:
        ax.text(0.97, 0.96, "held-out NLL", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.35")
        for kk, (fin, key) in enumerate(reversed(ends)):
            ax.text(0.97 - 0.29 * kk, 0.82, f"{fin:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=col[key])
    sh = float(rec["direct"]["shoulder_occupancy"])
    ax.text(0.5, 1.02, f"{sh:.2f} on shoulder", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=col["direct"])
    ax.tick_params(labelsize=7.0, length=2.2, pad=1.2); ax.set_xlabel("iteration", fontsize=8.0, labelpad=0.5)
    ax.set_ylabel("training loss" if a.curve == "train" else "held-out NLL", fontsize=8.0, labelpad=2)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

    # legend strip
    sx0 = W - right_in + strip_gap_in; sw = W - edge_in - sx0
    ax_s = fig.add_axes([sx0 / W, edge_in / H, sw / W, (H - 2 * edge_in) / H]); ax_s.set_axis_off(); ax_s.set_xlim(0, sw); ax_s.set_ylim(0, H - 2 * edge_in)
    foot = [max(strip_handle_in, _text_box_rot(fig, t, STRIP_FS)[1]) for t, _, _, _ in strip_entries]; sgap = (H - 2 * edge_in - sum(foot)) / len(foot)
    xh = strip_lw_in / 2; xt = strip_lw_in + strip_pad_in + strip_text_in / 2; yc = sgap / 2
    for (label, c, lw, ls), f in zip(strip_entries, foot):
        yc += f / 2
        ax_s.add_line(Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2], color=c, lw=lw, linestyle=ls, clip_on=False))
        ax_s.text(xt, yc, label, rotation=90, ha="center", va="center", fontsize=STRIP_FS, clip_on=False); yc += f / 2 + sgap
    os.makedirs(a.out_dir, exist_ok=True); stem = f"{a.out_dir}/darcy_panel" + ("_train" if a.curve == "train" else "")
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to): shutil.copy2(stem + ".pdf", a.mirror_to)
    print(f"wrote {stem}.pdf ({W:.2f} x {H:.2f} in): tiles {tile:.2f} in square, panels {pw:.2f} x {panel_h:.2f} in")


if __name__ == "__main__":
    main()
