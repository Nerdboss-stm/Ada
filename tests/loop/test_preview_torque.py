"""A14: preview_run reports max_torque_ratio from its own rollout, as telemetry only."""

import math

from core.contracts import Task
from harness.preview import run_preview
from sim import model as M
from sim.gait import Gait

TASK = Task.model_validate({"_id": "t-a14-preview", "split": "train", "slope_deg": 0.0,
                            "friction": 1.0, "target_m": 5.0, "eval_seeds": [101],
                            "practice_seeds": [7]})


def gait(power=1.0):
    return Gait.model_validate({
        "frequency_hz": 1.0, "power": power, "kp": 5.0, "kd": 0.1, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": 0.0, "phase": 0.0} for n in M.FORCE_JOINTS],
    })


def test_preview_returns_max_torque_ratio_without_verdict():
    result, _ = run_preview(gait(), TASK, seconds=1.0)
    ratio = result["max_torque_ratio"]
    assert isinstance(ratio, float) and math.isfinite(ratio) and ratio > 0
    assert not {"sanity", "success", "violation"} & set(result)
