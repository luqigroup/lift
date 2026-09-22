"""Generate a larger Darcy Karhunen-Loeve archive for the PCP-Map experiment.

The 5000-pair archive shipped with ``priorlaundermat`` is too thin for a
100-dimensional conditional transport map: at 40 samples per dimension the
``+ log det H`` term in the likelihood rewards sharpening the potential with
too few points to pin it between the training points, so the map steepens and
held-out points are thrown far from the origin in reference space.

This script reuses ``priorlaundermat``'s forward model, Karhunen-Loeve prior,
and sensor layout rather than reimplementing them, and it never writes into
that repository: the output goes to ``data/datasets/darcy/<experiment>.h5``
under this repo, with the name supplied by ``projorg`` so the archive's
identity is its configuration.

Two economies make a large archive cheap. The MAP reconstruction, which
dominates the generator's runtime, is computed for the evaluation rows only.
And the operator's singular values and resolved/blind bases are carried over
from a reference archive rather than recomputed, which saves the
finite-difference pass and guarantees the new samples share a basis with the
old ones. The carry-over is refused unless every operator-defining field of
the reference archive matches this configuration.

Usage::

    python scripts/prepare_darcy_kl.py --n_train 50000 --n_eval 2000
    python scripts/prepare_darcy_kl.py --n_train 2000 --n_eval 500 --map_for_eval 0
"""

from __future__ import annotations

import os
import time

import h5py
import numpy as np

from projorg import setup_environment

try:
    from priorlaundermat.groundwater import (
        DarcyForward,
        StuartKLPrior,
        beskos_sensors,
        map_reconstruct,
    )
except ImportError as exc:
    raise SystemExit(
        "the groundwater archive is generated with priorlaundermat, which is "
        "not installed. Install it with\n"
        "    pip install git+https://github.com/alisiahkoohi/priorlaundermat\n"
        "and run this script again."
    ) from exc

CONFIG_FILE = "prepare_darcy_kl.json"

# Fields that define the forward operator. The reference archive's
# subspace bases may only be carried over when all of them agree.
_OPERATOR_FIELDS = ("N", "K", "alpha", "s", "sigma", "sigma_y", "n_sensors")

_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "datasets", "darcy",
)


def _load_operator(reference_path: str, args) -> dict:
    """Carry the operator's spectrum and bases over from a reference archive."""
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            f"reference archive not found at {reference_path!r}. It supplies "
            "the operator's singular values and resolved/blind bases; point "
            "--reference_path at an existing darcy_laundering.h5.",
        )
    with h5py.File(reference_path, "r") as h:
        mismatched = {
            k: (float(h.attrs[k]), float(getattr(args, k)))
            for k in _OPERATOR_FIELDS
            if k in h.attrs and float(h.attrs[k]) != float(getattr(args, k))
        }
        if mismatched:
            raise ValueError(
                "reference archive was generated under a different operator, "
                f"so its subspace bases do not apply here: {mismatched!r} "
                "(reference value, requested value).",
            )
        return {
            "sv": np.asarray(h["sv"][:], dtype=np.float32),
            "resolved": np.asarray(h["resolved"][:], dtype=np.float32),
            "blind": np.asarray(h["blind"][:], dtype=np.float32),
            "sensor_ix": np.asarray(h["sensor_ix"][:]),
            "sensor_iy": np.asarray(h["sensor_iy"][:]),
            "rank": int(h.attrs.get("rank", 0)),
        }


def main() -> None:
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase"],
        sequence_args_and_types=[],
    )

    reference_path = str(args.reference_path) or os.path.join(
        _OUT_DIR, "darcy_laundering.h5",
    )
    operator = _load_operator(reference_path, args)

    kl = StuartKLPrior(
        int(args.N), K=int(args.K), alpha=float(args.alpha),
        s=float(args.s), sigma=float(args.sigma),
    )
    fwd = DarcyForward(int(args.N))
    sensors, _ = beskos_sensors(int(args.N), int(args.n_sensors))

    n_train, n_eval = int(args.n_train), int(args.n_eval)
    n = n_train + n_eval
    # A seed offset distinct from the reference archive's, so the new truths
    # are fresh samples rather than a superset containing it.
    rng = np.random.default_rng(int(args.seed) + int(args.seed_offset))
    xi_true = rng.standard_normal((n, kl.d))
    y_obs = np.empty((n, int(args.n_sensors)))
    xi_map = np.zeros_like(xi_true)

    print(
        f"Generating {n} pairs (train {n_train} / eval {n_eval}) at "
        f"d={kl.d}, N={args.N}; MAP for eval rows: "
        f"{bool(int(args.map_for_eval))}",
    )
    t0 = time.time()
    for i in range(n):
        u = kl.reconstruct(xi_true[i])
        y = fwd.observe(fwd.solve(u), sensors)
        y = y + float(args.sigma_y) * rng.standard_normal(int(args.n_sensors))
        y_obs[i] = y
        # The MAP reconstruction feeds only the secondary residual
        # reference curve, which lives on the evaluation split, and it is two
        # orders of magnitude more expensive than the forward solve.
        if i >= n_train and int(args.map_for_eval):
            xi_map[i] = map_reconstruct(
                fwd, kl, sensors, y, float(args.sigma_y),
                maxiter=int(args.maxiter),
            )
        if (i + 1) % max(1, n // 20) == 0:
            print(f"  {i + 1}/{n}  ({time.time() - t0:.0f}s)", flush=True)

    split = np.zeros(n, dtype=np.int8)
    split[n_train:] = 1

    os.makedirs(_OUT_DIR, exist_ok=True)
    out = os.path.join(_OUT_DIR, f"{args.experiment}.h5")
    if os.path.exists(out):
        raise FileExistsError(
            f"{out!r} already exists. The configuration hash names the "
            "archive, so an identical configuration would reproduce it; "
            "remove the file explicitly if a rebuild is intended.",
        )
    with h5py.File(out, "w") as f:
        for k in _OPERATOR_FIELDS + ("maxiter", "seed"):
            f.attrs[k] = getattr(args, k)
        f.attrs["n_train"] = n_train
        f.attrs["n_eval"] = n_eval
        f.attrs["rank"] = operator["rank"]
        f.attrs["generator"] = "scripts/prepare_darcy_kl.py"
        f.attrs["reference_archive"] = reference_path
        f.attrs["map_for_eval"] = int(args.map_for_eval)
        f.create_dataset("xi_true", data=xi_true.astype(np.float32))
        f.create_dataset("xi_map", data=xi_map.astype(np.float32))
        f.create_dataset("y_obs", data=y_obs.astype(np.float32))
        f.create_dataset("split", data=split)
        for k in ("sv", "resolved", "blind", "sensor_ix", "sensor_iy"):
            f.create_dataset(k, data=operator[k])
    print(f"Saved to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
