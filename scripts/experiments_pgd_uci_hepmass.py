"""UCI HEPMASS PGD baseline (D=21).

Thin driver over the lift-vs-direct UCI K=1 harness with
``backends="pgd"``. It mirrors the widened-direct sizing so the PGD
test NLL is directly comparable to the direct-softplus and lift numbers.
"""
from __future__ import annotations

from projorg import setup_environment
from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D

CONFIG_FILE = "experiments_pgd_uci_hepmass.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        # ``save_hyper_snaps`` and ``snap_every`` govern how much is
        # written to disk, not what is computed, so they stay out of the
        # run identity. ``hyper_readout_fanin_scale``, ``sampler_lr_ramp``
        # and ``ramp_sampler_T_max`` do change what is trained, but are
        # excluded so that adding them leaves every existing archive's
        # hash unchanged; a run that turns one on must pass its own
        # ``--experiment_name``.
        ignore_arg_list=[
            "experiment_name", "gpu_id", "phase",
            "save_hyper_snaps", "snap_every", "resume",
            "hyper_readout_fanin_scale", "sampler_lr_ramp",
            "ramp_sampler_T_max",
        ],
        sequence_args_and_types=[
            ("hyper_hidden_sizes", int),
            ("sampler_hyper_hidden_sizes", int),
            ("seeds", int),
            ("backends", str),
        ],
    )

    # Architecture knobs, forwarded to HyperNetwork only when set so that
    # an invocation that leaves them off builds what it built before. Both
    # keys default to 0 and enter the run identity, since they change the
    # model: pool_norm standardizes the pooled summary, body_layernorm
    # layer-normalizes the summary body.
    extra = {}
    if int(getattr(args, "hyper_pool_norm", 0)):
        extra["pool_norm"] = True
    if int(getattr(args, "hyper_body_layernorm", 0)):
        extra["body_layernorm"] = True
    # muP-style readout scaling: the emitted latent weight moves by ~lr per
    # Adam step instead of ~d_h * lr.
    if int(getattr(args, "hyper_readout_fanin_scale", 0)):
        extra["readout_fanin_scale"] = True
    if extra:
        args.hyper_extra_kwargs = extra
    experiment = HypernetVsDirect1D(args, target_kind=str(args.target_kind))
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
