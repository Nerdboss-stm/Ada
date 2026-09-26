"""Preview (SPEC §3 preview_run): a short sim of one gait on the task's first practice seed.

    run_preview(gait, task) -> (result, contact_log)

Rolls out sim.model directly for PREVIEW_S seconds; sim.verifier.evaluate and every
module default are left alone, so a full verification running at the same time is
unaffected. The result holds only distance, fell, max tilt, the contact rhythm and max_torque_ratio
(the highest per-joint peak torque over rated, from the same rollout) as plain telemetry:
no sanity check, no success, and evaluation seeds are never used.
"""

from __future__ import annotations

import math

from core.contracts import Task
from sim import model as M
from sim.controller import make_ctrl_fn
from sim.gait import Gait
from sim.verifier import TORSO_GEOM, TorqueLog, fell

PREVIEW_S = 3.0
LOG_BINS = 10
LEGS = ("leg_1", "leg_2", "leg_3", "leg_4")  # M.CONTACT_GEOMS order; the legs of hip_1..hip_4


def preview_seed(task: Task) -> int:
    return task.practice_seeds[0]


def _tilt_deg(frame: dict) -> float:
    """Angle between the torso's up axis and world z."""
    _, qx, qy, _ = frame["geoms"][TORSO_GEOM][3:]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))


def _leg(frames: list[dict], leg: int) -> list[bool]:
    return [bool(f["contacts"][leg]) for f in frames]


def rhythm(frames: list[dict]) -> dict[str, dict[str, float | int]]:
    """Per leg: share of frames in ground contact and number of touchdowns (rising edges)."""
    out = {}
    for i, name in enumerate(LEGS):
        c = _leg(frames, i)
        touchdowns = sum(1 for a, b in zip(c, c[1:]) if b and not a)
        out[name] = {"duty": round(sum(c) / len(c), 2) if c else 0.0, "touchdowns": touchdowns}
    return out


def contact_log(frames: list[dict], bins: int = LOG_BINS) -> dict[str, list[float]]:
    """Per leg: share of frames in ground contact in each of `bins` equal time bins."""
    n = len(frames)
    edges = [b * n // bins for b in range(bins + 1)]
    out = {}
    for i, name in enumerate(LEGS):
        c = _leg(frames, i)
        out[name] = [
            round(sum(c[lo:hi]) / (hi - lo), 2) if hi > lo else 0.0 for lo, hi in zip(edges, edges[1:])
        ]
    return out


def run_preview(gait: Gait, task: Task, seconds: float = PREVIEW_S) -> tuple[dict, dict]:
    seed = preview_seed(task)
    model, data = M.load(gait.power, task.slope_deg, task.friction)
    M.reset(model, data, seed)
    x0 = float(data.xpos[model.body(M.TORSO).id][0])
    torque = TorqueLog(model)
    frames = M.rollout(model, data, seed, make_ctrl_fn(model, gait), steps=round(seconds / M.TIMESTEP),
                       on_step=torque)
    result = {
        "seconds": seconds,
        "distance_m": round(frames[-1]["torso"][0] - x0, 4),
        "fell": fell(frames),
        "max_tilt_deg": round(max(_tilt_deg(f) for f in frames), 1),
        "rhythm": rhythm(frames),
        "max_torque_ratio": max(torque.peak_torque().values()),
    }
    return result, {"seconds": seconds, "bins": LOG_BINS, "legs": contact_log(frames)}
