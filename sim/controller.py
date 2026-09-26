"""Controller: PD tracking of a sine per joint plus hip tilt correction (SPEC §2.2).

target_i(t) = offset_i + amplitude_i * sin(2*pi*f*t + phase_i)
            - tilt_gain * (pitch * sx_i + roll * sy_i)      # hips only
ctrl_i      = clip(kp * (target_i - q_i) - kd * qdot_i, -1, 1)

(sx_i, sy_i) are the signs of leg i's direction in the torso frame. `power` is not
applied here: pass gait.power to sim.model.load, which sets gear = 150 * power.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from sim import model as M
from sim.gait import Gait

# Leg direction signs (x, y) in the torso frame, from ant.xml's leg bodies.
HIP_SIGNS = {"hip_1": (1.0, 1.0), "hip_2": (-1.0, 1.0), "hip_3": (-1.0, -1.0), "hip_4": (1.0, -1.0)}


def torso_roll_pitch(data: mujoco.MjData) -> tuple[float, float]:
    """Roll and pitch (rad) of the free joint's quaternion qpos[3:7] = (w, x, y, z)."""
    w, x, y, z = data.qpos[3:7]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def make_ctrl_fn(model: mujoco.MjModel, gait: Gait) -> M.CtrlFn:
    """Return ctrl_fn(t, data) -> ctrl in actuator order, for sim.model.rollout."""
    joint_ids = [model.joint(n).id for n in M.FORCE_JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    dadr = np.array([model.jnt_dofadr[j] for j in joint_ids])
    act_by_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    acts = np.array([act_by_joint[j] for j in joint_ids])

    amp = np.array([j.amplitude for j in gait.joints])
    off = np.array([j.offset for j in gait.joints])
    phase = np.array([j.phase for j in gait.joints])
    sx = np.array([HIP_SIGNS.get(n, (0.0, 0.0))[0] for n in M.FORCE_JOINTS])
    sy = np.array([HIP_SIGNS.get(n, (0.0, 0.0))[1] for n in M.FORCE_JOINTS])
    omega = 2 * math.pi * gait.frequency_hz

    def ctrl_fn(t: float, data: mujoco.MjData) -> np.ndarray:
        target = off + amp * np.sin(omega * t + phase)
        if gait.tilt_gain:
            roll, pitch = torso_roll_pitch(data)
            target = target - gait.tilt_gain * (pitch * sx + roll * sy)
        u = gait.kp * (target - data.qpos[qadr]) - gait.kd * data.qvel[dadr]
        ctrl = np.zeros(model.nu)
        ctrl[acts] = np.clip(u, -1.0, 1.0)
        return ctrl

    return ctrl_fn
