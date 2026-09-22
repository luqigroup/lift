"""Shared plumbing for the Darcy PCP-Map figure.

The Darcy rendering entrypoints all need the same two things: a
`PCPMapDarcy` instance configured exactly as the training run was, and the
checkpoint directory that run wrote into.

The architecture fields below must match
`configs/experiments_pcpmap_darcy.json`. They are repeated here rather than
re-parsed through `projorg.setup_environment` because these are post-hoc
renderers, not experiment drivers, and must not mint a new config hash,
which would resolve to an empty output directory.
"""
from __future__ import annotations

import glob
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Architecture of the paper's Darcy run. Keep in sync with
# configs/experiments_pcpmap_darcy.json.
ARCH = dict(
    feature_dim=128, feature_y_dim=64, num_layers=4, cond_dim=8,
    summary_dim=16, phi_hidden=64, phi_out=32, rho_hidden=16,
    hyper_hidden_sizes=[128, 128, 192], lr=1e-3, adam_eps=1e-12,
    weight_decay=0.0, batch_size=256, grad_clip=100.0,
    n_ref_batch=256, n_anchor_probes=2, fd_rel=0.3,
    n_precision_pairs=32, n_blind_probe=5,
)

ARMS = ("hypernet", "direct", "pgd")


def find_data(data_path: str = "", tag: str = "n_train-50000") -> str:
    """Locate the Darcy archive prepared by `scripts/prepare_darcy_kl.py`."""
    if data_path:
        return data_path
    hits = [f for f in glob.glob("data/datasets/darcy/*.h5") if tag in f]
    if not hits:
        raise FileNotFoundError(
            "no Darcy archive matching %r under data/datasets/darcy/; "
            "run scripts/prepare_darcy_kl.py first" % tag)
    return sorted(hits)[0]


def find_run(run_dir: str = "", tag: str = "data_tag-50k") -> str:
    """Locate the checkpoint dir projorg hashed the training config into."""
    if run_dir:
        return run_dir
    hits = [f for f in glob.glob("data/checkpoints/experiments_pcpmap_darcy_*")
            if tag in f]
    if not hits:
        raise FileNotFoundError(
            "no Darcy checkpoint dir matching %r; run "
            "scripts/experiments_pcpmap_darcy.py --phase train first" % tag)
    return sorted(hits)[0]


# Emitter flags that change the forward pass and so must match the run a
# checkpoint came from; all default off. A checkpoint of the fan-in lift
# (``readout_fanin_scale=1, pool_norm=1``) loaded into a default-flag
# hypernet emits the wrong weights without raising.
ARM_FLAGS = ("readout_fanin_scale", "pool_norm", "pool_scale")


def run_arm_flags(run_dir: str) -> dict:
    """The emitter flags a run was trained with, from its projorg log.

    Reads ``logs/<basename(run_dir)>/experiment_info.json``, written at
    launch, and returns the subset of ``ARM_FLAGS`` recorded there. An empty
    dict when the log is absent, in which case the defaults apply.
    """
    import json
    name = os.path.basename(os.path.normpath(run_dir))
    p = os.path.join("logs", name, "experiment_info.json")
    if not os.path.isfile(p):
        return {}
    a = json.load(open(p)).get("arguments", {})
    return {k: a[k] for k in ARM_FLAGS if k in a}


def build_experiment(data_path: str, backends=ARMS, gpu_id: int = 0,
                     eval_chunk: int = 128, **arm_flags):
    """A `PCPMapDarcy` wired to the trained architecture, without training.

    ``arm_flags`` (see ``ARM_FLAGS``) override the defaults for a checkpoint
    of the fan-in lift; pass ``**run_arm_flags(run_dir)``.
    """
    from _experiments_pcpmap_darcy_lib import PCPMapDarcy  # noqa: E402

    unknown = set(arm_flags) - set(ARM_FLAGS)
    if unknown:
        raise TypeError(f"unknown arm flags {sorted(unknown)}")
    args = types.SimpleNamespace(
        gpu_id=gpu_id, data_path=data_path, backends=list(backends),
        seeds=[0], eval_chunk=eval_chunk, experiment="probe", **ARCH,
        **arm_flags)
    if arm_flags:
        print("  arm flags:", arm_flags)
    return PCPMapDarcy(args)
