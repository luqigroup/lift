"""Published CP-Flow toy-density recipe, and the same model lifted.

Reproduces Huang et al. ICLR 2021's ``train_toy.py`` field for field on the
2-D toy densities under this repo's harness, so the CP-Flow numbers have a
like-for-like reference.

``--backend published`` (the default) is that recipe. ``--backend lifted``
runs the same model with one expression changed: the ICNN's constrained latent
weight becomes ``Theta_E h_phi(X) + b_h``, a learnable slack plus an
unconstrained DeepSets emitter over the conditioning batch, with the
positivity map and the ``1 / fan_in`` gain applied to it exactly as they are
applied to the raw parameter. Everything else is shared code, so a difference
between the two can only come from the positivity construction.

* ``--recipe A`` -- the upstream repository defaults and README command:
  ``nblocks=1, depth=20, dimh=32``, 10 epochs. Upstream applies these to
  whatever ``--data`` is passed, so A is target-independent here too, and is
  a repository default rather than the published configuration for any
  particular target.
* ``--recipe B`` -- the paper's appendix table, one row per target: One moon
  and Eight Gaussians at ``nblocks=5, depth=3, dimh=32``, Rings at
  ``nblocks=5, depth=5, dimh=256``, 50 epochs. 2-spirals has no row and
  borrows the Eight-Gaussians one, recorded as ``recipe_note``.

Targets ``--target {eight_gaussians,two_spirals,rings,one_moon}``.
``eight_gaussians`` and ``two_spirals`` run in either
``--convention {cpflow,lift_f1}``; ``rings`` and ``one_moon`` are
upstream's own generators and run in ``cpflow`` only (this repo's
``lift_f1`` convention has no ring / moon analogue). The absolute
nats are not comparable across conventions, so the one in use is
recorded in ``metrics.json`` and printed on the plots.

Examples::

    # published recipe A on 8-Gaussians, three seeds
    python scripts/experiments_cpflow_toy_published.py \\
        --recipe A --target eight_gaussians --convention cpflow \\
        --seeds 0,1,2

    # the lifted variant, on the appendix-table architecture
    python scripts/experiments_cpflow_toy_published.py \\
        --backend lifted --recipe B --target one_moon \\
        --convention cpflow --seeds 0

Config: ``configs/experiments_cpflow_toy_published.json``.
Lib:    ``_experiments_cpflow_toy_published_lib.py``.
"""

from __future__ import annotations

from projorg import setup_environment

from _experiments_cpflow_toy_published_lib import CPFlowToyPublished


CONFIG_FILE = "experiments_cpflow_toy_published.json"


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase"],
        sequence_args_and_types=[("seeds", int),
                                 ("hyper_hidden_sizes", int)],
    )
    experiment = CPFlowToyPublished(args)
    if args.phase == "train":
        experiment.train()
    experiment.load_checkpoint()
    experiment.visualize()
