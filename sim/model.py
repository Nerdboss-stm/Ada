"""Engine: load the ant, apply task physics, roll out deterministic frames (SPEC §2.1, §2.5).

Frame shape matches CONTRACTS.md §3 frames.frames[]:
  {"geoms": [[x, y, z, qw, qx, qy, qz], ...],  # non-floor geoms, manifest order
   "contacts": [bool] * 4,                      # CONTACT_GEOMS touching the floor
   "forces": [float] * 8,                       # joint torque / gear, clipped to [-1, 1], FORCE_JOINTS order
   "torso": [x, y, z]}
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
ANT_XML = ASSETS / "ant.xml"
MANIFEST_PATH = ASSETS / "manifest.json"

STOCK_GEAR = 150.0
GRAVITY = 9.81
TIMESTEP = 0.01
EPISODE_STEPS = 1000
RECORD_EVERY = 3
INIT_NOISE = 0.02
ROUND = 4

FLOOR = "floor"
TORSO = "torso"
CONTACT_GEOMS = ("left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom")
FORCE_JOINTS = ("hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4")
GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}

CtrlFn = Callable[[float, mujoco.MjData], np.ndarray]
StepFn = Callable[[int, mujoco.MjData], None]


def load(power: float = 1.0, slope_deg: float = 0.0, friction: float = 1.0) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load ant.xml with gear = 150 × power, gravity tilted so +x is uphill, and floor friction."""
    model = mujoco.MjModel.from_xml_path(str(ANT_XML))
    model.actuator_gear[:, 0] = STOCK_GEAR * power
    theta = math.radians(slope_deg)
    model.opt.gravity[:] = (-GRAVITY * math.sin(theta), 0.0, -GRAVITY * math.cos(theta))
    floor = model.geom(FLOOR).id
    model.geom_friction[floor, 0] = friction
    # Equal priorities mix friction by max; priority makes the floor's friction win.
    model.geom_priority[floor] = 1
    return model, mujoco.MjData(model)


def reset(model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> None:
    """Start from the file's init_qpos plus uniform ±0.02 qpos/qvel noise from default_rng(seed)."""
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(seed)
    init_qpos = np.asarray(model.numeric("init_qpos").data, dtype=np.float64)
    data.qpos[:] = init_qpos + rng.uniform(-INIT_NOISE, INIT_NOISE, size=model.nq)
    data.qvel[:] = rng.uniform(-INIT_NOISE, INIT_NOISE, size=model.nv)
    mujoco.mj_forward(model, data)


class _Indices:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.floor = model.geom(FLOOR).id
        self.geoms = [g for g in range(model.ngeom) if g != self.floor]
        self.contact_geoms = [model.geom(n).id for n in CONTACT_GEOMS]
        joint_ids = [model.joint(n).id for n in FORCE_JOINTS]
        act_by_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self.force_actuators = [act_by_joint[j] for j in joint_ids]
        self.force_dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        self.torso = model.body(TORSO).id


def _frame(model: mujoco.MjModel, data: mujoco.MjData, idx: _Indices) -> dict:
    quat = np.zeros(4)
    geoms = []
    for g in idx.geoms:
        mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
        geoms.append([round(float(v), ROUND) for v in (*data.geom_xpos[g], *quat)])

    touching = set()
    for c in range(data.ncon):
        g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
        if g1 == idx.floor:
            touching.add(g2)
        elif g2 == idx.floor:
            touching.add(g1)
    contacts = [g in touching for g in idx.contact_geoms]

    torque = data.qfrc_actuator[idx.force_dofs]
    gear = model.actuator_gear[idx.force_actuators, 0]
    forces = [round(float(v), ROUND) for v in np.clip(torque / gear, -1.0, 1.0)]

    torso = [round(float(v), ROUND) for v in data.xpos[idx.torso]]
    return {"geoms": geoms, "contacts": contacts, "forces": forces, "torso": torso}


def rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: int,
    ctrl_fn: CtrlFn,
    steps: int = EPISODE_STEPS,
    every: int = RECORD_EVERY,
    on_step: StepFn | None = None,
) -> list[dict]:
    """Reset with seed, step `steps` times with ctrl = ctrl_fn(t, data), record a frame every `every` steps.

    on_step(step, data), if given, runs after every mj_step; it must not change `data`.
    """
    reset(model, data, seed)
    idx = _Indices(model)
    frames = []
    for step in range(steps):
        data.ctrl[:] = np.clip(ctrl_fn(float(data.time), data), -1.0, 1.0)
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(step, data)
        if (step + 1) % every == 0:
            frames.append(_frame(model, data, idx))
    return frames


def frames_sha256(frames: list[dict]) -> str:
    blob = json.dumps(frames, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def build_manifest() -> dict:
    model = mujoco.MjModel.from_xml_path(str(ANT_XML))
    idx = _Indices(model)
    geoms = []
    for g in idx.geoms:
        body_id = int(model.geom_bodyid[g])
        geoms.append({
            "id": g,
            "name": model.geom(g).name,
            "type": GEOM_TYPES[int(model.geom_type[g])],
            "size": [round(float(v), 6) for v in model.geom_size[g]],
            "body": model.body(body_id).name or f"body_{body_id}",
        })
    return {
        "manifest_version": hashlib.sha256(ANT_XML.read_bytes()).hexdigest()[:12],
        "geoms": geoms,
        "contact_geoms": list(CONTACT_GEOMS),
        "force_joints": list(FORCE_JOINTS),
    }


def write_manifest(path: Path = MANIFEST_PATH) -> None:
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--manifest"]:
        write_manifest()
        print(MANIFEST_PATH)
    else:
        sys.exit("usage: python -m sim.model --manifest")
