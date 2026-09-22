"""PCP-Map on the Darcy inverse problem: PGD vs direct softplus vs the lift.

Thin driver. The experiment lives in
``scripts/_experiments_pcpmap_darcy_lib.py`` and the defaults in
``configs/experiments_pcpmap_darcy.json``.

Sequence arguments are comma-separated, not space-separated::

    python scripts/experiments_pcpmap_darcy.py --seeds 0,1,2
    python scripts/experiments_pcpmap_darcy.py --seeds 0,1,2 --phase visualization
    python scripts/experiments_pcpmap_darcy.py --backends pgd,direct,hypernet,hypernet_nocond

Finished runs are skipped on a re-run unless ``--overwrite 1``, so an
interrupted sweep resumes where it stopped.
"""
from __future__ import annotations

from projorg import setup_environment
from _experiments_pcpmap_darcy_lib import PCPMapDarcy

CONFIG_FILE = "experiments_pcpmap_darcy.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        # ``data_path`` stays out of the run identity: an absolute path
        # would inject separators into the experiment name, and a train run
        # that passed it would not resolve from a later plain ``--phase
        # visualization``. It therefore does not enter the config hash, so
        # pass it explicitly on every call.
        # ``init_mode`` / ``init_scheme`` do change what is computed but are
        # also kept out of the identity, so two initializations share one
        # directory unless they are separated by ``--experiment_name``;
        # every record carries ``init_mode`` so a directory can be checked.
        # ``save_hyper_snaps`` and ``mirror_figure`` govern only what is
        # written to disk.
        ignore_arg_list=[
            "experiment_name", "gpu_id", "phase",
            "data_path", "overwrite", "save_hyper_snaps", "mirror_figure",
            "init_mode", "init_scheme",
        ],
        sequence_args_and_types=[
            ("hyper_hidden_sizes", int),
            ("backends", str),
            ("seeds", int),
        ],
    )
    experiment = PCPMapDarcy(args)
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
