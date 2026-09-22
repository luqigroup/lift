"""FFHQ ICNN-regularizer three-way comparison (Ehrhardt pipeline).

Trains the Ehrhardt/Mukherjee/Wong adversarial ICNN regularizer on the
shipped FFHQ inpainting pairs under three backends -- pgd (their recipe
verbatim), direct (softplus reparametrization), hypernet (the lift) -- and
evaluates each by downstream reconstruction PSNR through their vendored
primal-dual solver on the 20 test images (their Table 2 protocol).

Usage (the exact CLI the committed slurm jobs run):

    python scripts/experiments_ffhq_icnn_regularizer.py \\
        --backends pgd --seeds 0 1 2 --phase train --gpu_id 0
    python scripts/experiments_ffhq_icnn_regularizer.py \\
        --backends hypernet --seeds 0 1 2 --phase train --gpu_id 0

Re-plot from cached results (skips training; completed runs are also skipped
on any retrain relaunch):

    python scripts/experiments_ffhq_icnn_regularizer.py \\
        --backends pgd --seeds 0 1 2 --phase visualization

Color / full-FFHQ variant (CLI overrides only; grayscale stays the config
default). This leaves the Ehrhardt anchor -- their published numbers exist
only for the 1,000-image grayscale subset -- and runs the internal three-way
benchmark at 3 channels x all 70,000 images (data prepared by
``scripts/prepare_ffhq_color.py``):

    python scripts/experiments_ffhq_icnn_regularizer.py \\
        --backends pgd,direct,hypernet --seeds 0 1 2 \\
        --n_channels 3 --img_size 256 \\
        --data_root data/datasets/ffhq_color \\
        --n_epochs 1 --log_every 50 --eval_every 250 \\
        --checkpoint_every 2000 --snap_every 250 \\
        --phase train --gpu_id 0
"""

from __future__ import annotations

import sys

from projorg import setup_environment

from _experiments_ffhq_icnn_regularizer_lib import FFHQICNNRegularizer

CONFIG_FILE = "experiments_ffhq_icnn_regularizer.json"

_SEQUENCE_FLAGS = (
    "--seeds", "--backends", "--hyper_hidden_sizes", "--conv_channels",
)


def _coalesce_sequence_argv(argv: list[str]) -> list[str]:
    """Join space-separated sequence-flag values into comma-strings.

    The committed slurm jobs pass ``--seeds 0 1 2`` (argparse ``nargs``
    style), but projorg's generated parser takes exactly one token per
    flag and would reject the extras. Rewriting ``--seeds 0 1 2`` into
    ``--seeds 0,1,2`` BEFORE projorg parses keeps the fixed CLI
    contract working; the comma form (and ``--seeds=0,1,2``) passes
    through unchanged, and both spellings resolve to the same projorg
    experiment name (commas become dashes there either way).
    """
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        out.append(tok)
        i += 1
        if tok in _SEQUENCE_FLAGS:
            vals = []
            while i < len(argv) and not argv[i].startswith("--"):
                vals.append(argv[i])
                i += 1
            if vals:
                out.append(",".join(vals))
    return out


if __name__ == "__main__":
    sys.argv = _coalesce_sequence_argv(sys.argv)
    args = setup_environment(
        CONFIG_FILE,
        # ``log_every`` / ``checkpoint_every`` / ``n_recon_examples`` /
        # ``save_emitter`` govern what is printed or written to disk, not
        # what is computed, so they stay out of the run identity.
        #
        # The ``emission_decay_*`` keys, ``mean_match_negatives``,
        # ``pool_scale`` and ``n_cond`` are also ignored although they do
        # change what is computed, because each is an exact no-op at its
        # config default and ignoring them keeps every completed run
        # resolving to the directory it already occupies. The cost is that
        # the hash no longer separates such a run from its baseline, so a
        # run that turns one on must claim its own directory through
        # ``--experiment_name`` (the run name's prefix, not a hashed field).
        # The guards below enforce that; without them the run would land on
        # top of the default-valued one and be skipped as complete.
        ignore_arg_list=[
            "experiment_name", "gpu_id", "phase",
            "log_every", "checkpoint_every", "n_recon_examples",
            "save_emitter",
            "emission_decay_frac", "emission_decay_end",
            "emission_decay_floor", "emission_decay_ema",
            "mean_match_negatives",
            "pool_scale",
            "n_cond",
        ],
        sequence_args_and_types=[
            ("seeds", int),
            ("backends", str),
            ("hyper_hidden_sizes", int),
            ("conv_channels", int),
        ],
    )
    if int(getattr(args, "mean_match_negatives", 0)) and \
            "meanmatch" not in str(args.experiment_name):
        raise SystemExit(
            "--mean_match_negatives 1 is not part of the run hash; pass "
            "--experiment_name experiments_ffhq_icnn_regularizer_meanmatch "
            "so the run does not land on top of a shortcut-negatives run "
            "and get skipped as complete.")
    _ps = float(getattr(args, "pool_scale", 0.0))
    if _ps > 0.0 and f"ps{_ps:g}" not in str(args.experiment_name):
        raise SystemExit(
            f"--pool_scale {_ps:g} is not part of the run hash; pass an "
            f"--experiment_name containing 'ps{_ps:g}' (e.g. "
            f"experiments_ffhq_icnn_regularizer_meanmatch_ps{_ps:g}) so "
            "the pinned run does not land on top of the unpinned one and "
            "get skipped as complete.")
    _nc = int(getattr(args, "n_cond", 0))
    if _nc > 0:
        if f"ncond{_nc}" not in str(args.experiment_name):
            raise SystemExit(
                f"--n_cond {_nc} is not part of the run hash; pass an "
                f"--experiment_name containing 'ncond{_nc}' (e.g. "
                f"experiments_ffhq_icnn_regularizer_meanmatch_ps1_"
                f"ncond{_nc}) so the sub-batch run does not land on top "
                "of the full-batch one and get skipped as complete.")
        # Only the lift reads the conditioning batch; pgd and direct ignore
        # it, so running them under an n_cond name would fill a separate
        # directory with runs identical to the full-batch ones.
        # ``backends`` is parsed by ``sequence_args_and_types`` into a list,
        # but a CLI-free run leaves the config's comma string, so accept
        # both rather than stringifying the list.
        _raw = args.backends
        _backends = ([str(b).strip() for b in _raw]
                     if isinstance(_raw, (list, tuple))
                     else [b.strip() for b in str(_raw).split(",")])
        _backends = [b for b in _backends if b]
        if [b for b in _backends if b != "hypernet"]:
            raise SystemExit(
                f"--n_cond {_nc} only changes the lift; run it with "
                "--backends hypernet (the pgd and direct baselines are "
                "unaffected by the conditioning batch and already "
                "measured at n_cond 0).")
    if float(getattr(args, "emission_decay_frac", 0.0)) > 0.0 and \
            "decay" not in str(args.experiment_name):
        raise SystemExit(
            "--emission_decay_frac > 0 is not part of the run hash; pass an "
            "--experiment_name containing 'decay' (e.g. "
            "experiments_ffhq_icnn_regularizer_meanmatch_decay) so the "
            "annealed run does not land on top of the un-annealed one and "
            "get skipped as complete.")
    experiment = FFHQICNNRegularizer(args)
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
