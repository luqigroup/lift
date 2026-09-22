"""The two 2-D convex-potential-flow toys at the published recipe.

Row 1: density on a grid -- the data, direct softplus (CP-Flow's own
construction and initialization), the lift -- each model at one seed
(``--seed_pick``), evaluated exactly as the driver does, with the lift's
emission pinned on a training conditioning set; without the pin it would emit
its weights from the plot grid. Row 2: the loss along training for the same
seed, with the entropy floor and the baseline's end-of-training shoulder
share.

Direct softplus never enters the shoulder on these targets, so the escape
account does not apply and no gap is expected.

Reuses the driver library's own methods (``_data``, ``_build``,
``_load_flow``, ``_eval_cond_batch``, ``_logp_grid``) on the stored resolved
config, so the density here is the density the run defines.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys, types
import numpy as np, matplotlib, matplotlib.pyplot as plt, torch
from matplotlib.lines import Line2D
_HERE = os.path.dirname(os.path.abspath(__file__))
from projorg import gitdir
_REPO = gitdir()
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
from lift.utils.paper_style import PALETTE, apply_paper_style  # noqa: E402
from _experiments_cpflow_toy_published_lib import CPFlowToyPublished  # noqa: E402
from render_tabular_val_loss_multiseed import _draw_holdout, holdout_order  # noqa: E402

COLS = [("8-Gaussians", "eight_gaussians"), ("2-spirals", "two_spirals"), ("One moon", "one_moon")]
_ALL_ARMS = [("published", "direct", "direct softplus"), ("pgd", "pgd", "PGD"), ("lifted", "hypernet", "lift")]
ARMS = _ALL_ARMS   # narrowed by --arms in main()
# Seed rank to show per (target, construction): 0 = best by the reported number.
SEED_RANK = {}


def _archives(backend, target, variant="fanin"):
    """seed -> (archive dir, per_seed record) over one backend's recipe-B archives.

    Runs may be split across archives (seeds are part of the config hash), so
    the seeds are merged. For the lift, ``variant`` selects the construction:
    "fanin" = readout_fanin_scale (the 1/d_h readout), "default" = the shipped
    lift, "ro0.02" = the readout_lr_mult 0.02 runs; each excludes the others.
    Duplicate seeds prefer the archive with more seeds.
    """
    out = {}
    for d in sorted(glob.glob(f"{_REPO}/data/checkpoints/experiments_cpflow_toy_published_backend-{backend}_recipe-B_target-{target}_*"),
                    key=lambda d: -len(json.load(open(d + "/metrics.json"))["per_seed"]) if os.path.exists(d + "/metrics.json") else 0):
        if not os.path.exists(d + "/metrics.json"): continue
        m = json.load(open(d + "/metrics.json")); lift = m["resolved_config"].get("lift", {})
        if backend == "lifted":
            mult = float(lift.get("readout_lr_mult", 1.0)); fanin = bool(lift.get("readout_fanin_scale", False))
            if variant == "ro0.02" and not (mult == 0.02 and not fanin): continue
            if variant == "fanin" and not (fanin and mult == 1.0): continue
            if variant == "default" and not (mult == 1.0 and not fanin): continue
        for s_, rec in m["per_seed"].items():
            out.setdefault(s_, (d, rec, m))
    return out


def _stub(archive, blob):
    """A CPFlowToyPublished with only what its evaluation methods read."""
    o = CPFlowToyPublished.__new__(CPFlowToyPublished)
    o.resolved = blob["resolved"]; o.device = torch.device("cpu"); o.results = None
    o.args = types.SimpleNamespace(experiment=os.path.basename(archive))
    return o


def _load(archive, seed, cond="all"):
    """The trained flow of one seed, its data, and the emission's conditioning set.

    cond="all": the lift's latent weight is computed once from the whole
    training set. cond="batch" is the driver's pinned 128-point conditioning
    batch, which is what metrics.json records. ``None`` for the published
    recipe, which emits nothing.
    """
    blob = torch.load(f"{archive}/seed{seed}_flow.pt", map_location="cpu", weights_only=False)
    o = _stub(archive, blob)
    torch.set_default_dtype(torch.float64)
    flow = o._build().to(o.device); flow.load_state_dict(blob["state_dict"])
    from lift.baselines.cpflow_toy import mark_actnorms_initialized
    mark_actnorms_initialized(flow); flow.eval()
    x_train, x_test = o._data()
    if o.resolved["lift"]["backend"] != "lifted": x_cond = None
    elif cond == "all": x_cond = torch.from_numpy(np.asarray(x_train)).to(o.device).view(-1, 2)
    else: x_cond = o._eval_cond_batch(x_train)
    return o, flow, x_train, x_test, x_cond


def _final_nll(archive, seed, cond="all"):
    """Final-iterate test NLL under the chosen conditioning set."""
    o, flow, x_train, x_test, x_cond = _load(archive, seed, cond)
    loader = torch.utils.data.DataLoader(torch.from_numpy(np.asarray(x_test)), batch_size=1000)
    return float(o._test_nll(flow, loader, x_cond)[0])


def _density(archive, seed, b=4.0, n=100, cond="all"):
    """Density on the grid under the chosen conditioning set."""
    o, flow, x_train, _, x_cond = _load(archive, seed, cond)
    return np.exp(o._logp_grid(flow, b=b, n=n, x_cond=x_cond)), x_train


def _text_box_rot(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs, rotation=90); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=f"{_REPO}/figures")
    p.add_argument("--mirror_to", default="", help="optional second directory to copy the figure into")
    p.add_argument("--fig_width", type=float, default=5.5); p.add_argument("--loss_h", type=float, default=0.86)
    p.add_argument("--dens_layout", choices=["row", "grid"], default="row",
                   help="'row' puts the nine densities on one line, three per target, which is the "
                        "compact form; 'grid' gives "
                        "one row per arm with larger squares at three times the height")
    p.add_argument("--dens_sq", type=float, default=0.58,
                   help="side of one density square in inches ("
                        "rows data/direct/lift and columns the targets, so they read at reading size)")
    p.add_argument("--band", type=float, default=10.0); p.add_argument("--grid_n", type=int, default=100)
    p.add_argument("--cond", choices=["all", "batch"], default="all", help="conditioning set for the plotted density: the whole training set, or the driver's pinned 128-point batch")
    p.add_argument("--lift_variant", choices=["fanin", "default", "ro0.02"], default="fanin", help="which lift construction to show (the 1/d_h readout is the paper's)")
    p.add_argument("--seed_pick", choices=["median", "best"], default="median", help="which seed carries the density and the bold curve")
    p.add_argument("--curve", choices=["train", "val"], default="train",
                   help="loss row: the training objective (default) or the held-out NLL along training")
    p.add_argument("--holdout", choices=["table", "block", "ends"], default="table",
                   help="where the final held-out NLL of each construction goes: a block in the panel's headroom, or labels at the curve ends")
    p.add_argument("--skip_frac", type=float, default=0.1, help="--curve train: the y-range is taken over the run after this fraction of it")
    p.add_argument("--arms", default="direct,hypernet", help="which constructions to draw (direct, pgd, hypernet); the default is the paper's two-way comparison (PGD is run where the published work uses it, and CP-Flow ships with the positivity map)")
    a = p.parse_args()
    global ARMS
    ARMS = [t for t in _ALL_ARMS if t[1] in a.arms.split(",")]
    if not ARMS:
        raise ValueError(f"--arms selected nothing, got {a.arms!r}")
    apply_paper_style(); matplotlib.rcParams["savefig.bbox"] = "standard"; matplotlib.rcParams["axes.grid"] = False
    col = {"hypernet": PALETTE["hypernet"], "direct": PALETTE["direct_softplus"], "pgd": "#2ca02c", "floor": "0.45", "data": "0.35"}
    dash = {"hypernet": (0, ()), "direct": (0, (4, 1.6)), "pgd": (0, (3, 1.2, 1, 1.2)), "floor": (0, (1.2, 1.2))}

    # ---- data: per-seed archives, the shown seed, densities ---------------
    M, dens, x_trains, floors, sh_direct, nll_final, nlls_all, arch_of = {}, {}, {}, {}, {}, {}, {}, {}
    global COLS
    COLS = [c for c in COLS if all(_archives(b, c[1], a.lift_variant) for b, _, _ in ARMS)]
    print("targets with archives for variant", a.lift_variant, ":", [c[1] for c in COLS], flush=True)
    for _, tgt in COLS:
        common = set.intersection(*[set(_archives(b, tgt, a.lift_variant)) for b, _, _ in ARMS])   # paired seeds only
        for backend, key, _ in ARMS:
            seeds = {s_: v for s_, v in _archives(backend, tgt, a.lift_variant).items() if s_ in common}   # seed -> (archive, record, metrics)
            nlls = {s_: _final_nll(d, int(s_), a.cond) for s_, (d, _, _) in seeds.items()}
            order = sorted(nlls, key=lambda s_: nlls[s_])
            shown = order[0] if a.seed_pick == "best" else order[(len(order) - 1) // 2]   # median = the middle of an odd count, lower middle of an even one
            d_shown, rec_shown, m_shown = seeds[shown]
            M[(tgt, key)] = (m_shown, {s_: seeds[s_][1] for s_ in seeds}, shown); nll_final[(tgt, key)] = nlls[shown]; nlls_all[(tgt, key)] = nlls
            arch_of[(tgt, key)] = {s_: seeds[s_][0] for s_ in seeds}
            if key != "pgd":                                                  # densities: data | direct | lift only
                dens[(tgt, key)], x_trains[tgt] = _density(d_shown, int(shown), n=a.grid_n, cond=a.cond)
            floors[tgt] = float(m_shown["target_reference"].get("entropy_floor_nats") or float("nan"))
            if key == "direct": sh_direct[tgt] = float(rec_shown["softplus_shoulder_at_end"]["fraction"])
            print(f"{tgt:16s} {key:8s} {a.seed_pick} seed {shown}: {nlls[shown]:.4f} (cond={a.cond})  seeds {sorted(nlls)}: {[round(nlls[s_], 4) for s_ in sorted(nlls)]}", flush=True)

    # ---- layout: no target titles, the density row spanning the full width
    # up to the legend strip, and little air between it and the "on shoulder"
    # line ---------------------------------------------------------------
    W = a.fig_width
    _drawn = {k for _, k, _ in ARMS}
    strip_entries = [(lab, col[k], 1.2, dash[k]) for k, lab in (("hypernet", "lift"), ("direct", "direct softplus"), ("pgd", "PGD")) if k in _drawn] + ([("floor", col["floor"], 1.0, dash["floor"])] if a.curve == "val" else [])
    STRIP_FS = 6.6; strip_gap_in, strip_handle_in, strip_pad_in, strip_lw_in, edge_in = 0.44, 0.30, 0.045, 0.04, 0.06
    fig = plt.figure(figsize=(W, 2.0)); strip_text_in = max(_text_box_rot(fig, t, STRIP_FS)[0] for t, _, _, _ in strip_entries); plt.close(fig)
    tail_in = 0.42 if a.holdout == "table" else 0.0   # the framed held-out table sits outside each panel's right edge
    right_in = tail_in + strip_gap_in + strip_lw_in + strip_pad_in + strip_text_in + edge_in
    # loss row: its own left margin for the y-label
    G = len(COLS)
    left_in, gap_in = 0.55, (0.72 if a.holdout == "table" else 0.62)   # room for the table / three end labels before the next panel
    pw = (W - left_in - right_in - (G - 1) * gap_in) / G
    # Density block: a 3 x G grid, one row per construction, each column
    # centered on the loss panel below it, squares as large as the column
    # pitch allows.
    row_gap = 0.06
    dens_rows = [("data", None, col["data"]), ("direct softplus", "direct", col["direct"]), ("lift", "hypernet", col["hypernet"])]
    ROW = a.dens_layout == "row"
    sq_gap, grp_gap, dens_left_in = 0.035, 0.16, 0.06
    if ROW:   # three squares per target on one line, filling the width
        sq = (W - right_in - dens_left_in - 2 * G * sq_gap - (G - 1) * grp_gap) / (3 * G)
    else:
        sq = min(a.dens_sq, pw + gap_in - 0.14)
    top_in, dens_t_in, mid_in, bot_in = 0.03, (0.13 if ROW else 0.02), 0.22, 0.33
    dens_block = sq if ROW else (len(dens_rows) * sq + (len(dens_rows) - 1) * row_gap)
    H = top_in + dens_t_in + dens_block + mid_in + a.loss_h + bot_in
    fig = plt.figure(figsize=(W, H))
    y_loss = bot_in / H; y_sq = (bot_in + a.loss_h + mid_in) / H
    dens_lab_in = 0.11                       # rotated row label sits left of the grid
    b = 4.0
    # y-axes are per panel, not shared across targets
    for i, (title, tgt) in enumerate(COLS):
        # densities: one line of three per target ("row"), or one row per construction ("grid")
        cx = left_in + i * (pw + gap_in) + pw / 2.0
        x0d = dens_left_in + i * (3 * sq + 2 * sq_gap + grp_gap)
        for j, (lab, key, c) in enumerate(dens_rows):
            if ROW:
                x_cell, y_row = x0d + j * (sq + sq_gap), y_sq
            else:
                x_cell = cx - sq / 2.0
                y_row = (bot_in + a.loss_h + mid_in + dens_block - (j + 1) * sq - j * row_gap) / H
            ax = fig.add_axes([x_cell / W, y_row, sq / W, sq / H])
            if key is None:
                Hh, _, _ = np.histogram2d(x_trains[tgt][:, 0], x_trains[tgt][:, 1], 200, range=[[-b, b], [-b, b]])
                ax.imshow(Hh.T, cmap="viridis", origin="lower", extent=[-b, b, -b, b], aspect="equal")
            else:
                ax.imshow(dens[(tgt, key)], cmap="viridis", origin="lower", extent=[-b, b, -b, b], aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_edgecolor(c); s_.set_linewidth(1.2)
            if ROW:
                ax.set_title(lab, fontsize=7.0, pad=2.0, color=c if key else "0.25")
            elif i == 0:
                ax.text(-dens_lab_in / sq, 0.5, lab, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=7.0, color=c if key else "0.25")
        # row 2: the objective along training (or the held-out NLL, --curve val), the shown seed bold
        def series(rec):
            """(iterations, values) of the plotted curve for one seed's record."""
            h = rec["history"]
            if a.curve == "val":
                return np.asarray(h["eval_iter"], float), np.asarray(h["test_nll"], float)
            v = np.asarray(h["train_nll"], float); w = 3                 # already a mean over log_every steps; a short running median
            return np.asarray(h["iter"], float), np.array([np.nanmedian(v[max(0, k - w):k + 1]) for k in range(len(v))])
        x0 = left_in + i * (pw + gap_in)
        ax = fig.add_axes([x0 / W, y_loss, pw / W, a.loss_h / H]); ends = []; tails = []
        for backend, key, _ in ARMS:
            m, ps, shown = M[(tgt, key)]
            it, y_shown = series(ps[shown])
            for s_ in ps:                                                   # every seed, thin
                if s_ == shown: continue
                xs, ys_ = series(ps[s_])
                ax.plot(xs, ys_, color=col[key], lw=0.55, alpha=0.38, linestyle=dash[key], zorder=2); tails.append(ys_[int(a.skip_frac * len(ys_)):])
            ax.plot(it, y_shown, color=col[key], lw=1.4, linestyle=dash[key], zorder={"hypernet": 3.0, "direct": 3.6, "pgd": 4.0}[key]); tails.append(y_shown[int(a.skip_frac * len(y_shown)):])
            ends.append((nll_final[(tgt, key)], key))                     # the shown seed's final held-out NLL under --cond
        floor = floors[tgt]
        if floor == floor and a.curve == "val":
            ax.axhline(floor, color=col["floor"], lw=1.0, linestyle=dash["floor"], zorder=1)
        if a.curve == "val":
            ymax = max(float(t.max()) for t in tails) + 0.03
            ymin = (floor if floor == floor else min(nll_final[(tgt, k)] for _, k, _ in ARMS)) - 0.05
        else:
            # the range from the run after its first tenth, the floor kept in view, with headroom for the held-out block
            lo = min(float(t.min()) for t in tails); hi = max(float(t.max()) for t in tails)
            if floor == floor: lo = min(lo, floor)
            pad = 0.06 * (hi - lo); ymin, ymax = lo - pad, hi + pad + (0.55 if a.holdout == "block" else 0.0) * (hi - lo)   # the table needs no headroom
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, float(it.max())); ax.set_xticks([0, 10000, 19550]); ax.set_xticklabels(["0", "10k", "19.5k"], fontsize=7)
        if a.holdout == "ends":
            ends = sorted(ends); ys = [e[0] for e in ends]; step = 0.135 * (ymax - ymin)   # three labels must clear each other in a 0.86 in panel
            for k in range(1, len(ys)):
                if ys[k] - ys[k - 1] < step: ys[k] = ys[k - 1] + step
            for (fin, key), y in zip(ends, ys):
                ax.annotate(f"{fin:.3f}", xy=(it.max(), fin), xytext=(1.03, y), textcoords=("axes fraction", "data"), xycoords="data",
                            fontsize=7.5, color=col[key], ha="left", va="center", clip_on=False,
                            arrowprops=dict(arrowstyle="-", color=col[key], lw=0.5, alpha=0.6, shrinkA=0, shrinkB=1))
        elif a.holdout == "table":
            # A framed legend whose entries are the shown seed's final
            # held-out NLL of each construction, outside the right edge; bare
            # numbers in the corner read as the curve's last value.
            _draw_holdout(fig, ax, "table", holdout_order(ends), col, dash, lambda v: f"{v:.3f}", (x0, 0, pw, a.loss_h), (W, H))
        else:
            ax.text(0.97, 0.96, "held-out NLL", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.35")
            for kk, (fin, key) in enumerate(reversed(ends)):
                ax.text(0.97 - 0.30 * kk, 0.81, f"{fin:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=col[key])
        ax.text(0.5, 1.015, f"{sh_direct[tgt]:.2f} on shoulder", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=col["direct"])
        ax.tick_params(labelsize=7.5, length=2.5, pad=1.5); ax.set_xlabel("iteration", fontsize=8.5, labelpad=0.5)
        if i == 0: ax.set_ylabel("training NLL" if a.curve == "train" else "held-out NLL", fontsize=8.5, labelpad=2)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    # legend strip
    # The strip names the curve styles, so it sits beside the loss row only;
    # a full-height strip would run up beside the densities it does not
    # describe.
    sx0 = W - right_in + strip_gap_in; sw = W - edge_in - sx0
    strip_y0, strip_h = bot_in, (H - bot_in - top_in) if ROW else (a.loss_h + mid_in + sq + row_gap)
    ax_s = fig.add_axes([sx0 / W, strip_y0 / H, sw / W, strip_h / H]); ax_s.set_axis_off(); ax_s.set_xlim(0, sw); ax_s.set_ylim(0, strip_h)
    foot = [max(strip_handle_in, _text_box_rot(fig, t, STRIP_FS)[1]) for t, _, _, _ in strip_entries]; sgap = (strip_h - sum(foot)) / len(foot)
    xh = strip_lw_in / 2; xt = strip_lw_in + strip_pad_in + strip_text_in / 2; yc = sgap / 2
    for (label, c, lw, ls), f in zip(strip_entries, foot):
        yc += f / 2
        ax_s.add_line(Line2D([xh, xh], [yc - strip_handle_in / 2, yc + strip_handle_in / 2], color=c, lw=lw, linestyle=ls, clip_on=False))
        ax_s.text(xt, yc, label, rotation=90, ha="center", va="center", fontsize=STRIP_FS, clip_on=False); yc += f / 2 + sgap
    os.makedirs(a.out_dir, exist_ok=True); stem = f"{a.out_dir}/cpflow_toy_panel" + ("_train" if a.curve == "train" else "")
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); plt.close(fig)
    if a.mirror_to and os.path.isdir(a.mirror_to): shutil.copy2(stem + ".pdf", a.mirror_to)
    print("wrote", stem + ".pdf", f"({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
