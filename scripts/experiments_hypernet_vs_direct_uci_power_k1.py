"""K=1 UCI POWER (D=6) hypernet-vs-direct ICNN comparison.

Extends the single-component comparison from the toy 1-D and 2-D targets to
a real-data $D = 6$ tabular setting. The shared $\\theta$-space NLL landscape
takes log Z from flow importance sampling with the converged hypernet flow as
a fixed proposal, so the landscape is deterministic and decoupled from
per-cell importance-sampling variance. K is fixed to 1, which isolates the
single-component parameterization question from any K-component coordination.
"""

from __future__ import annotations

from projorg import setup_environment

from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D


CONFIG_FILE = "experiments_hypernet_vs_direct_uci_power_k1.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        # The three trailing knobs are opt-in and stay out of the run identity.
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "save_hyper_snaps", "resume",
                         "hyper_readout_fanin_scale", "sampler_lr_ramp", "ramp_sampler_T_max"],
        sequence_args_and_types=[
            ("hyper_hidden_sizes", int),
            ("sampler_hyper_hidden_sizes", int),
            ("seeds", int),
            ("backends", str),
        ],
    )

    # Architecture knobs, forwarded to HyperNetwork only when set.
    # ``pool_norm`` is the standardized pooled summary and
    # ``body_layernorm`` the alternative. Both keys live in the config json,
    # default to 0, and enter the run identity because they change the
    # model.
    extra = {}
    if int(getattr(args, "hyper_pool_norm", 0)):
        extra["pool_norm"] = True
    if int(getattr(args, "hyper_body_layernorm", 0)):
        extra["body_layernorm"] = True
    if int(getattr(args, "hyper_readout_fanin_scale", 0)):
        extra["readout_fanin_scale"] = True
    if extra:
        args.hyper_extra_kwargs = extra
    experiment = HypernetVsDirect1D(args, target_kind=str(args.target_kind))
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
