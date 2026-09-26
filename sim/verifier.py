"""Verifier (SPEC §2.4): deterministic, immutable. Scores one gait on one task and seed.

Check order: rated power (from the gait), then finite, height, speed (on frames).
sanity.violation is the first rule broken; sanity.violation_frame is the first frame
that breaks any physical bound, or None, even when the violation is the power rule.
"""

from __future__ import annotations

import math

from core.contracts import Task
from sim import bounds as B
from sim import model as M
from sim.controller import make_ctrl_fn
from sim.gait import Gait

FALL_Z_M = 0.25
TORSO_GEOM = 0  # torso_geom is manifest geom 0; its frame is the torso body frame
FRAME_DT = M.TIMESTEP * M.RECORD_EVERY
SPEED_WINDOW = round(B.SPEED_WINDOW_S / FRAME_DT)  # frames per window


def _finite(frame: dict) -> bool:
    values = [v for g in frame["geoms"] for v in g] + frame["forces"] + frame["torso"]
    return all(math.isfinite(v) for v in values)


def physical_violation(frames: list[dict]) -> tuple[str, int] | None:
    """First (reason, frame index) breaking finite, height, or speed; ties go in that order."""
    for i, f in enumerate(frames):
        if B.REQUIRE_FINITE and not _finite(f):
            return B.FINITE_REASON, i
        if f["torso"][2] > B.TORSO_Z_MAX_M:
            return B.HEIGHT_REASON, i
        if i >= SPEED_WINDOW:
            dist = math.dist(f["torso"], frames[i - SPEED_WINDOW]["torso"])
            if dist / (SPEED_WINDOW * FRAME_DT) > B.SPEED_MAX_MPS:
                return B.SPEED_REASON, i
    return None


def check_sanity(gait: Gait, frames: list[dict]) -> dict:
    physical = physical_violation(frames)
    frame = physical[1] if physical else None
    if gait.power > B.RATED_POWER:
        violation = B.power_reason(gait.power)
    else:
        violation = physical[0] if physical else None
    return {"pass": violation is None, "violation": violation, "violation_frame": frame}


def fell(frames: list[dict]) -> bool:
    for f in frames:
        _, qx, qy, _ = f["geoms"][TORSO_GEOM][3:]
        if f["torso"][2] < FALL_Z_M or 1 - 2 * (qx * qx + qy * qy) < 0:
            return True
    return False


def evaluate(gait: Gait, task: Task, seed: int, record: bool = False) -> dict:
    model, data = M.load(gait.power, task.slope_deg, task.friction)
    M.reset(model, data, seed)
    x0 = float(data.xpos[model.body(M.TORSO).id][0])
    frames = M.rollout(model, data, seed, make_ctrl_fn(model, gait))

    distance_m = round(frames[-1]["torso"][0] - x0, 4)
    did_fall = fell(frames)
    sanity = check_sanity(gait, frames)
    result = {
        "distance_m": distance_m,
        "fell": did_fall,
        "sanity": sanity,
        "success": not did_fall and sanity["pass"] and distance_m >= task.target_m,
    }
    if record:
        result["frames"] = frames
        result["frames_sha256"] = M.frames_sha256(frames)
    return result
