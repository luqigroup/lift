"""The lift on the CIFAR-10 input-convex classifier of Sivaprasad et al. (2021).

Trains their CIFAR-10 architecture, three hidden layers of 800 units with ELU,
on our ICNN under six positivity constructions on one pipeline: their
post-update rule (``exppost``, after every step a negative constrained weight
``w`` becomes ``exp(w - 5)``), its scale-matched control (``exppost_matched``,
``eps = 8.80``), direct softplus (``direct``), projected gradient descent
(``pgd``), the lift (``lift``) and no positivity (``nonconvex``). Same ``hk``
initialization, batch 64, mean cross entropy, Adam, a 40-epoch cap with early
stopping on validation accuracy at patience 8, and the learning rate selected
per construction from {1e-4, 3e-4, 1e-3} on validation accuracy of the
held-out 10%. Test accuracy at the epoch of best validation accuracy and final
test accuracy are recorded per cell, with the training curves and the dwell
statistics beside them.

Phases: ``preflight`` checks that every setting takes effect and trains
nothing; ``cell`` trains the requested cells and exits; ``train`` trains and
then plots; ``visualization`` plots from the cells on disk.

    python scripts/experiments_sivaprasad_cifar.py --phase preflight --lr 0.0003
    python scripts/experiments_sivaprasad_cifar.py --phase cell \\
        --constructions lift --lr 0.0003 --seeds 0

CPU smoke, a handful of steps:

    CUDA_VISIBLE_DEVICES="" python scripts/experiments_sivaprasad_cifar.py \\
        --phase cell --gpu_id -1 --data_on_device 0 --epochs 2 \\
        --max_steps_per_epoch 5 --constructions exppost --seeds 0

The learning rate is in the run identity, so each rate is its own directory;
the six grid axes are not, so every cell at one rate shares a directory.
``scripts/collect_sivaprasad_cifar.py`` reads them together.
"""

from __future__ import annotations

from projorg import setup_environment

from _experiments_sivaprasad_cifar_lib import SivaprasadCIFAR


CONFIG_FILE = "experiments_sivaprasad_cifar.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        # Run control and fields that change no recorded number.
        ignore_arg_list=["experiment_name", "gpu_id", "phase",
                         "resume", "data_on_device", "eval_chunk",
                         "max_lift_params", "ce0_match_rtol",
                         "draw_match_median_rtol", "draw_match_occ_atol",
                         "lift_cond_spread_batches",
                         "checkpoint_every_epochs", "save_weights",
                         "gpu_memory_fraction",
                         "datasets", "constructions", "init_modes",
                         "depths", "activations", "seeds"],
        sequence_args_and_types=[
            ("datasets", str),
            ("constructions", str),
            ("init_modes", str),
            ("depths", int),
            ("activations", str),
            ("seeds", int),
            ("hyper_hidden_sizes", int),
        ],
    )
    experiment = SivaprasadCIFAR(args)
    if args.phase in ("train", "preflight", "cell"):
        experiment.train()
    if args.phase in ("preflight", "cell"):
        raise SystemExit(0)
    experiment.load_checkpoint()
    experiment.visualize()
