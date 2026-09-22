"""1-D generative ablation: hypernet against direct-softplus ICNN.

Trains a log-concave EBM on a 1-D Gumbel target under FKL-direct
(samples known) for two backends:

  * **hypernet** (the default): the DeepSets ``HyperNetwork`` predicting the
    ICNN's effective weights from each batch.
  * **direct**: ``DirectParamModule``, an EBM in
    ``pos_constraint_mode="softplus"`` exposing its own raw parameters, with
    the same ``forward(x) -> dict`` API the trainer expects.

Both are scored on one task-level metric, the TV, Hellinger and KL distance
between the learned density and the target, which avoids comparing
optimization landscapes that live in different parameter spaces.

Config: ``configs/experiments_hypernet_vs_direct_1d.json``.
Lib:    ``_experiments_hypernet_vs_direct_1d_lib.py``.
"""

from __future__ import annotations

from projorg import setup_environment

from _experiments_hypernet_vs_direct_1d_lib import HypernetVsDirect1D


CONFIG_FILE = "experiments_hypernet_vs_direct_1d.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "save_hyper_snaps"],
        sequence_args_and_types=[
            ("hyper_hidden_sizes", int),
            ("sampler_hyper_hidden_sizes", int),
            ("seeds", int),
            ("backends", str),
        ],
    )
    experiment = HypernetVsDirect1D(args, target_kind=str(args.target_kind))
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
