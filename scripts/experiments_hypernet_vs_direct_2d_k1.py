"""K=1 2-D Gamma-Gauss single-mode hypernet-against-direct ICNN ablation.

One 2-D log-concave mode, a Gamma marginal on x and a Gaussian on y, which
has an exact log_prob and is log-concave for Gamma shape at least 1. The
single-component question is the 1-D setting's, asked where the parameter
space is larger and the optimization can fail in more ways.
"""

from __future__ import annotations

from projorg import setup_environment

from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D


CONFIG_FILE = "experiments_hypernet_vs_direct_2d_k1.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "save_hyper_snaps"],
        sequence_args_and_types=[
            ("hyper_hidden_sizes", int),
            ("sampler_hyper_hidden_sizes", int),
            ("seeds", int),
        ],
    )
    experiment = HypernetVsDirect1D(args, target_kind=str(args.target_kind))
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
