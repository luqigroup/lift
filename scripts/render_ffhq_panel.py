"""The FFHQ image-regularizer panel.

Image rows: held-out FFHQ test images at 256 x 256 -- ground truth, the
inpainting measurement (Gaussian noise 0.03 plus 25% salt-and-pepper), and
the primal-dual reconstruction under the regularizer trained with direct
softplus, PGD and the lift, at step size 5e-4 on one paired seed, each tile
carrying its PSNR. Bottom row: the critic objective along training, or the
validation PSNR beside it, each curve ending at the test PSNR at its
lowest-validation-loss iterate.

Renders from cached runs only: ``data/checkpoints/*meanmatch*/{backend}_seed0/
{recon_examples.npz, metrics.json}``, selected on the same driver arguments
the other FFHQ renderers use (n_filters 32, n_epochs 20, n_channels 1, lr
5e-4; lift: pool_scale 0.5, n_cond 0, head_init_pos_bias -5.3,
hyper_hidden_sizes [64], conv_channels [32, 64, 128]; direct and PGD:
head_init_pos_bias -2.0).
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, sys
import numpy as np, matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402
from render_tabular_val_loss_multiseed import _draw_holdout, holdout_order  # noqa: E402  the framed held-out table shared by the main-text figures

LR = 5e-4
ARMS = [("direct", "direct softplus"), ("pgd", "PGD"), ("hypernet", "lift")]


def _psnr_median(backend, initmatch=False):
    """Median test PSNR at the reported iterate over every seed of the cell.

    The framed table carries this median; the image tiles keep the drawn
    seed's own value.
    """
    # The matched start exists in two cell families of the same configuration:
    # the small initmatch cells the panel draws from, and the larger initaxis
    # cells at init_mode "own". Take both, preferring initaxis where a seed is
    # in each.
    vals = {}
    for path in sorted(glob.glob(f"{_REPO}/data/checkpoints/*meanmatch*/metrics.json")):
        try: rec = json.load(open(path))
        except (OSError, json.JSONDecodeError): continue
        a = rec.get("args") or {}
        name = str(a.get("experiment_name", ""))
        axis = "initaxis" in name and str(a.get("init_mode", "")) == "own"
        if not (axis or ("initmatch" in name) == bool(initmatch)): continue
        if (a.get("n_filters"), a.get("n_epochs"), a.get("n_channels")) != (32, 20, 1) or a.get("lr") != LR: continue
        if int(a.get("readout_fanin_scale", 0) or 0) != 0: continue
        for key, arm in (rec.get("per_arm") or {}).items():
            m = re.match(r"(.+)_seed(\d+)$", key)
            if not m or m.group(1) != backend: continue
            v = arm.get("test_psnr_bestval_mean")
            if v is None: continue
            seed = int(m.group(2))
            if axis or seed not in vals: vals[seed] = float(v)
    if not vals:
        raise SystemExit(f"no seeds for {backend} at lr {LR}")
    return float(np.median(list(vals.values()))), len(vals)


def _cell(backend, seed, initmatch=False):
    """The directory of one construction at the published step size.

    ``initmatch`` selects the control in which every method starts at
    ``head_init_pos_bias -5.3``, an emitted constrained median of
    softplus(-5.3) = 0.005. Those cells carry the same pool_scale, body and
    encoder as the published lift, so without the experiment-name test below
    the two are indistinguishable and either could be picked silently.
    """
    for path in sorted(glob.glob(f"{_REPO}/data/checkpoints/*meanmatch*/metrics.json")):
        try: rec = json.load(open(path))
        except (OSError, json.JSONDecodeError): continue
        a = rec.get("args") or {}
        if ("initmatch" in str(a.get("experiment_name", ""))) != bool(initmatch): continue
        if (a.get("n_filters"), a.get("n_epochs"), a.get("n_channels")) != (32, 20, 1) or a.get("lr") != LR: continue
        if int(a.get("readout_fanin_scale", 0) or 0) != 0: continue  # fan-in readout cells are not the published lift
        if backend == "hypernet" or initmatch:
            if a.get("pool_scale") != 0.5 or int(a.get("n_cond", 0) or 0) != 0 or a.get("head_init_pos_bias") != -5.3: continue
            if a.get("hyper_hidden_sizes") != [64] or a.get("conv_channels") != [32, 64, 128]: continue
        elif a.get("head_init_pos_bias") != -2.0: continue
        d = os.path.join(os.path.dirname(path), f"{backend}_seed{seed}")
        if os.path.isfile(d + "/recon_examples.npz") and os.path.isfile(d + "/metrics.json"): return d
    raise SystemExit(f"no cell for {backend} seed {seed} at lr {LR}"
                     + (" (init-matched)" if initmatch else ""))


def _psnr(a, b):
    return float(10.0 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12)))


def _text_box_rot(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs, rotation=90); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=f"{_REPO}/figures")
    p.add_argument("--mirror_to", default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5);     p.add_argument("--loss_h", type=float, default=0.54)
    p.add_argument("--tile_scale", type=float, default=1.0,
                   help="image-row tile size as a fraction of the full five-across width")
    p.add_argument("--initmatch", action="store_true", help="draw the init-matched control: every construction starts inside the shoulder")
    p.add_argument("--out_name", default=None, help="default: ffhq_panel (--curve val) or ffhq_panel_train")
    p.add_argument("--curve", choices=["train", "val"], default="train",
                   help="bottom row: the critic objective alone with the test PSNR at the reported iterate in a framed table (default; "
                        "the paper is about optimization), or the critic objective beside the validation PSNR along training")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--examples", default="1", help="which stored test images to show")
    a = p.parse_args()
    if a.out_name is None:
        a.out_name = "ffhq_panel" + ("_train" if a.curve == "train" else "")
    apply_paper_style(); matplotlib.rcParams["savefig.bbox"] = "standard"; matplotlib.rcParams["axes.grid"] = False
    col = {"hypernet": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c", "grey": "0.35"}
    dash = {"hypernet": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2))}
    zo = {"hypernet": 3.0, "direct": 3.6, "pgd": 4.0}

    cells = {k: _cell(k, a.seed, a.initmatch) for k, _ in ARMS}
    # Prefer the best-validation reconstructions: the driver stores the final
    # iterate, while every reported PSNR is at the lowest-validation-loss one.
    def _recon(d):
        bv = d + "/recon_examples_bestval.npz"
        return np.load(bv if os.path.isfile(bv) else d + "/recon_examples.npz")
    npz = {k: _recon(d) for k, d in cells.items()}
    for k, d in cells.items():
        print(f"  {k:9s} images from {'best-val' if os.path.isfile(d + '/recon_examples_bestval.npz') else 'FINAL iterate'}")
    met = {k: json.load(open(d + "/metrics.json")) for k, d in cells.items()}
    clean, noisy = npz["direct"]["clean"], npz["direct"]["noisy"]
    strip_entries = [("lift", col["hypernet"], 1.2, dash["hypernet"]), ("PGD", col["pgd"], 1.2, dash["pgd"]), ("direct softplus", col["direct"], 1.2, dash["direct"])]   # bottom to top: the order of the held-out table, read downward
    ex = [int(s) for s in a.examples.split(",")]
    for k, d in cells.items():
        print(f"{k:9s} {os.path.relpath(d, _REPO)}  test PSNR at reported iterate {met[k]['test_psnr_bestval_mean']:.2f} (step {met[k]['bestval_step']}), shoulder share at end {met[k]['history']['w_frac_shoulder'][-1]:.3f}")

    W = a.fig_width
    strip_entries = [("lift", col["hypernet"], 1.2, dash["hypernet"]), ("direct softplus", col["direct"], 1.2, dash["direct"]), ("PGD", col["pgd"], 1.2, dash["pgd"])]
    STRIP_FS = 6.6; strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.44, 0.30, 0.045, 0.04, 0.06
    fig = plt.figure(figsize=(W, 2.0)); strip_text_in = max(_text_box_rot(fig, t, STRIP_FS)[0] for t, _, _, _ in strip_entries); plt.close(fig)
    right_in = strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in
    tile_gap = 0.04
    left_in = 0.50                       # room for the critic panel's y-label
    img_left_in = left_in
    tab_in = 0.82                        # the framed held-out table, outside the critic panel
    span_in = W - right_in - left_in - tab_in          # the critic panel: leaves tab_in for the table
    # The tiles run the full width up to the legend strip; only the critic
    # panel stops short of it, to leave room for the table.
    img_span_in = W - right_in - img_left_in
    tile = a.tile_scale * (img_span_in - 4 * tile_gap) / 5
    top_in, title_in, mid_in, bot_in = 0.03, 0.13, 0.15, 0.29
    n_rows = len(ex)
    H = top_in + title_in + n_rows * tile + (n_rows - 1) * tile_gap + mid_in + a.loss_h + bot_in
    fig = plt.figure(figsize=(W, H))
    # rows 1-2: images
    columns = [("ground truth", None, col["grey"]), ("measurement", "noisy", col["grey"])] + [(lab, k, col[k]) for k, lab in ARMS]
    for r, i in enumerate(ex):
        y0 = (bot_in + a.loss_h + mid_in + (n_rows - 1 - r) * (tile + tile_gap)) / H
        for j, (lab, key, c) in enumerate(columns):
            ax = fig.add_axes([(img_left_in + j * (tile + tile_gap)) / W, y0, tile / W, tile / H])
            if key is None: img, val = clean[i, 0], None
            elif key == "noisy": img, val = noisy[i, 0], _psnr(noisy[i], clean[i])
            else: img, val = npz[key]["recon"][i, 0], _psnr(npz[key]["recon"][i], clean[i])
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_edgecolor(c); s_.set_linewidth(1.2)
            if val is not None:
                ax.text(0.04, 0.04, f"{val:.1f} dB", transform=ax.transAxes, color="white", fontsize=6.3, ha="left", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1.4, lw=0))
            if r == 0: ax.set_title(lab, fontsize=7.0, pad=2.0, color=c if key not in (None, "noisy") else "0.25")
    # Bottom row: left, the critic objective on the training batches; right,
    # PSNR on the validation images, ending at the test PSNR at the reported
    # iterate, with direct softplus's shoulder share in red.
    gap_in = 0.55
    pw = (W - left_in - right_in - gap_in) / 2
    if a.curve == "train":
        # One panel, the critic objective, centered under the tiles with the
        # framed held-out table outside its right edge.
        pw_full = span_in
        x_l = left_in
        ax_l = fig.add_axes([x_l / W, bot_in / H, pw_full / W, a.loss_h / H]); ax = None
    else:
        ax_l = fig.add_axes([left_in / W, bot_in / H, pw / W, a.loss_h / H])
        ax = fig.add_axes([(left_in + pw + gap_in) / W, bot_in / H, pw / W, a.loss_h / H])
    ymin, ymax = np.inf, -np.inf; ends = []
    for key, _ in ARMS:
        h = met[key]["history"]
        xs = np.asarray(h["steps"], float); L = np.asarray(h["diff_loss"], float)
        ax_l.plot(xs, L, color=col[key], lw=1.2, linestyle=dash[key], zorder=zo[key])
        _m, _n = _psnr_median(key, a.initmatch)
        ends.append((_m, key))      # median over seeds of the test PSNR at the reported iterate, the number the text reports
        print(f"  {key:9s} framed table: median {_m:.2f} dB over {_n} seeds "
              f"(drawn seed {a.seed}: {float(met[key]['test_psnr_bestval_mean']):.2f})")
        x = np.asarray(h["eval_steps"], float); y = np.asarray(h["val_psnr"], float)
        if ax is None:
            continue
        # Each curve stops at its own reported iterate, which is where the
        # marker sits. Direct softplus and PGD peak at the end of the budget,
        # so only the lift is truncated.
        bs = float(met[key].get("bestval_step") or x.max())
        m = x <= bs
        if m.sum() >= 2: x, y = x[m], y[m]
        ax.plot(x, y, color=col[key], lw=1.2, linestyle=dash[key], zorder=zo[key])
        ax.plot([x[-1]], [y[-1]], marker="o", ms=2.6, color=col[key], zorder=zo[key] + 0.1)
        ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
    if ax is not None:
        ymin, ymax = ymin - 1.0, ymax + 1.0; ax.set_ylim(ymin, ymax); ax.set_xlim(0, float(max(np.asarray(met[k]["history"]["eval_steps"], float).max() for k, _ in ARMS)))
    ax_l.set_xlim(0, float(xs.max()))
    for a_ in ((ax_l, ax) if ax is not None else (ax_l,)):
        a_.set_xticks([0, 1000, 2000, 2500]); a_.set_xticklabels(["0", "1k", "2k", "2.5k"])
        a_.tick_params(labelsize=7.5, length=2.5, pad=1.5); a_.set_xlabel("iteration", fontsize=8.5, labelpad=0.5)
        for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
    ax_l.set_ylabel("critic objective", fontsize=8.5, labelpad=2)
    sh = float(met["direct"]["history"]["w_frac_shoulder"][-1])
    if ax is not None:
        ax.text(0.5, 1.015, f"{sh:.2f} on shoulder", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=col["direct"])
        ax.set_ylabel("val. PSNR (dB)", fontsize=8.5, labelpad=2)
    else:
        ax_l.text(0.5, 1.015, f"{sh:.2f} on shoulder", transform=ax_l.transAxes, ha="center", va="bottom", fontsize=7.0, color=col["direct"])
        # The held-out read: test PSNR at each construction's reported
        # iterate, higher is better. The frame is right-aligned on the tile
        # row's right edge, so it clears the critic panel's spine on the left
        # and ends where the images end on the right.
        leg = _draw_holdout(fig, ax_l, "table", holdout_order(ends), col, dash, lambda v: f"{v:.1f} dB",
                            (x_l, bot_in, pw_full, a.loss_h), (W, H), inside=True, anchor_x=img_span_in / pw_full)
        ax_l.get_legend().set_title("test PSNR (median)"); ax_l.get_legend().get_title().set_color("0.35")
    # legend strip
    sx0 = W - right_in + strip_gap_in; sw = W - edge_in - sx0
    ax_s = fig.add_axes([sx0 / W, edge_in / H, sw / W, (H - 2 * edge_in) / H]); ax_s.set_axis_off(); ax_s.set_xlim(0, sw); ax_s.set_ylim(0, H - 2 * edge_in)
    foot = [max(strip_handle_in, _text_box_rot(fig, t, STRIP_FS)[1]) for t, _, _, _ in strip_entries]; sgap = (H - 2 * edge_in - sum(foot)) / len(foot)
    xh = strip_lw_in / 2; xt = strip_lw_in + strip_pad_in + strip_text_in / 2; yc = sgap / 2
    for (label, c, lw, ls), f in zip(strip_entries, foot):
        yc += f / 2
        ax_s.add_line(Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2], color=c, lw=lw, linestyle=ls, clip_on=False))
        ax_s.text(xt, yc, label, rotation=90, ha="center", va="center", fontsize=STRIP_FS, clip_on=False); yc += f / 2 + sgap
    os.makedirs(a.out_dir, exist_ok=True); stem = f"{a.out_dir}/{a.out_name}"
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to): shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
