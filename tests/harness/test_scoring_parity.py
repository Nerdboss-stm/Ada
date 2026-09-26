"""B13: harness.run scores holdout and swap mean distance exactly like the gate's score().

Pure functions, no database: identical run dicts go to both and must give equal means.
"""

import pytest

from gate.verifier_stage import score
from harness.run import mean_distance_m

GAIT = {"frequency_hz": 1.0}


def run(distance_m, target_m, *, ok=True, gait=GAIT, success=False):
    return {"distance_m": distance_m, "target_m": target_m, "gait": gait, "success": success,
            "sanity": {"pass": ok, "violation": None if ok else "joint torque", "violation_frame": None if ok else 7}}


CASES = {
    "clean": {"a": [run(2.5, 3.0, success=False), run(3.2, 3.0, success=True)],
              "b": [run(1.0, 4.0), run(0.4, 4.0)]},
    "violation": {"a": [run(9.0, 3.0, ok=False), run(2.0, 3.0)],
                  "b": [run(12.0, 4.0, ok=False, success=True), run(1.5, 4.0)]},
    "no_gait": {"a": [run(0.0, 3.0, gait={}), run(0.0, 3.0, gait={})],
                "b": [run(1.0, 4.0), run(0.8, 4.0)]},
    "mixed": {"a": [run(9.0, 3.0, ok=False), run(0.0, 3.0, gait={})],
              "b": [run(0.0, 4.0, gait={}), run(4.1, 4.0, success=True)],
              "c": [run(-0.3, 2.5), run(7.0, 2.5, ok=False)]},
    "violation_backward": {"a": [run(-3.0, 2.0, ok=False), run(1.0, 2.0)]},
    "empty": {},
}


@pytest.mark.parametrize("name", list(CASES))
def test_mean_distance_matches_gate_score(name):
    runs_by_task = CASES[name]
    flat = [r for rs in runs_by_task.values() for r in rs]
    assert mean_distance_m(flat) == score(runs_by_task)["mean_distance_m"]


def test_penalized_runs_walk_the_target_backward():
    assert mean_distance_m(CASES["violation"]["a"]) == pytest.approx((-3.0 + 2.0) / 2)
    assert mean_distance_m(CASES["no_gait"]["a"]) == pytest.approx(-3.0)


def test_a_violation_that_walked_backward_keeps_its_distance():
    """B14: min(distance_m, -target_m), so a failure can only lower a score."""
    backward = {"a": [run(-3.0, 2.0, ok=False)]}
    assert mean_distance_m(backward["a"]) == score(backward)["mean_distance_m"] == -3.0
