"""Verifier (SPEC §2.4): deterministic, immutable. Scores one gait on one task and seed.

Every simulation step records each joint's |qfrc_actuator| / 150 (rated gear); its peak,
with the step it happened at, is returned per joint as peak_torque.
Check order: rated torque, then finite, height, speed (on frames). sanity.violation is the
first rule broken; the torque reason cites the joint with the highest peak ratio at the
time of that peak. sanity.violation_frame is the first frame that breaks a physical bound,
else the first frame holding an over-torque step, else None.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

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


class TorqueLog:
    """rollout on_step hook: per joint peak |torque| / rated gear and its step; first over-rated step."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.dofs = np.array([int(model.jnt_dofadr[model.joint(n).id]) for n in M.FORCE_JOINTS])
        self.peak = np.zeros(len(M.FORCE_JOINTS))
        self.peak_step = np.zeros(len(M.FORCE_JOINTS), dtype=int)
        self.first_over_step: int | None = None

    def __call__(self, step: int, data: mujoco.MjData) -> None:
        ratio = np.abs(data.qfrc_actuator[self.dofs]) / B.RATED_GEAR
        higher = ratio > self.peak  # strict: the first step reaching a peak keeps it
        self.peak[higher] = ratio[higher]
        self.peak_step[higher] = step
        if self.first_over_step is None and bool((ratio > B.TORQUE_RATIO_MAX).any()):
            self.first_over_step = step

    def peak_torque(self) -> dict[str, float]:
        return {n: round(float(r), 4) for n, r in zip(M.FORCE_JOINTS, self.peak)}

    def violation(self) -> str | None:
        """The torque reason for the joint with the highest peak (first in FORCE_JOINTS on ties)."""
        if self.first_over_step is None:
            return None
        j = int(np.argmax(self.peak))
        t = (int(self.peak_step[j]) + 1) * M.TIMESTEP  # data.time after that step
        return B.torque_reason(M.FORCE_JOINTS[j], float(self.peak[j]), t)


def step_frame(step: int, n_frames: int) -> int:
    """The first recorded frame at or after simulation step `step`."""
    return min(step // M.RECORD_EVERY, n_frames - 1)


def check_sanity(frames: list[dict], torque: TorqueLog) -> dict:
    physical = physical_violation(frames)
    over = torque.violation()
    if physical:
        frame = physical[1]
    elif over is not None:
        frame = step_frame(torque.first_over_step, len(frames))
    else:
        frame = None
    violation = over if over is not None else (physical[0] if physical else None)
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
    torque = TorqueLog(model)
    frames = M.rollout(model, data, seed, make_ctrl_fn(model, gait), on_step=torque)

    distance_m = round(frames[-1]["torso"][0] - x0, 4)
    did_fall = fell(frames)
    sanity = check_sanity(frames, torque)
    result = {
        "distance_m": distance_m,
        "fell": did_fall,
        "sanity": sanity,
        "success": not did_fall and sanity["pass"] and distance_m >= task.target_m,
        "peak_torque": torque.peak_torque(),
    }
    if record:
        result["frames"] = frames
        result["frames_sha256"] = M.frames_sha256(frames)
    return result
