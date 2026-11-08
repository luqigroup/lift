"""The step-size selection of ``scripts/collect_exp_map_1d.py``."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from collect_exp_map_1d import run_summary, select  # noqa: E402


def _run(val_200: float, val_400: float, loss_nan: bool = False) -> dict:
    loss = [1.0] * 400
    if loss_nan:
        loss[300] = math.nan
    return run_summary({"loss_E_per_iter": loss, "eval_iters": [0, 200, 399],
                        "val_loss_per_eval": [-10.0, val_200, val_400]})


def test_a_difference_below_0002_nats_goes_to_1e_3_and_iteration_0_is_not_read():
    runs = {"a": {"5e-4": _run(1.0, 1.5), "1e-3": _run(1.6, 1.0019), "2e-3": _run(1.01, 1.2)}}
    sel = select(runs)["a"]
    assert sel["best_val"] == {"5e-4": 1.0, "1e-3": 1.0019, "2e-3": 1.01}
    assert sel["lr"] == "1e-3" and sel["edge"] is None
    runs["a"]["1e-3"] = _run(1.6, 1.0021)
    sel = select(runs)["a"]
    assert sel["lr"] == "5e-4" and sel["edge"] == "lower"


def test_a_selection_at_2e_3_is_labeled_the_upper_end():
    sel = select({"a": {"5e-4": _run(1.2, 1.3), "1e-3": _run(1.1, 1.2), "2e-3": _run(1.0, 1.1)}})["a"]
    assert sel["lr"] == "2e-3" and sel["edge"] == "upper"


def test_a_step_where_any_method_diverges_is_eligible_for_none():
    runs = {"a": {"5e-4": _run(1.2, 1.3), "1e-3": _run(1.1, 1.2), "2e-3": _run(1.0, 1.1)},
            "b": {"5e-4": _run(1.2, 1.3), "1e-3": _run(1.1, 1.2), "2e-3": _run(1.0, 1.1, loss_nan=True)}}
    assert not runs["b"]["2e-3"]["finite"]
    sel = select(runs)
    assert sel["a"]["eligible"] == {"5e-4": True, "1e-3": True, "2e-3": False}
    assert sel["a"]["lr"] == sel["b"]["lr"] == "1e-3"
    nan_val = run_summary({"loss_E_per_iter": [1.0] * 400, "eval_iters": [0, 200, 399],
                           "val_loss_per_eval": [1.0, math.nan, 1.0]})
    assert not nan_val["finite"]
