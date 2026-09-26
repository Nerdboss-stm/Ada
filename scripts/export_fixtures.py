"""Write three fixture frames documents to ada.frames for Lane C and print their ids.

    uv run python -m scripts.export_fixtures

All on the showcase task (slope 0, friction 1), seed 0:
  zero   neutral offsets, no motion
  trot   diagonal trot: hip 0.5, ankle 0.4, 1.5 Hz, kp 5, kd 0.5, power 1
  flail  power 5.0, amplitude 1.2 on all joints, 3.0 Hz, kp 20
"""

from __future__ import annotations

import math

from core.contracts import Task
from core.db import ADA
from sim import model as M
from sim import tasks as T
from sim.gait import Gait, neutral_offsets
from sim.record import write_frames
from sim.verifier import evaluate

SEED = 0
FIXTURE_VERSION = "fixture"

# Hip axes all point up, so hip_1/hip_2 swing -x on positive rotation and hip_3/hip_4 +x.
# Diagonal pairs (1,3) and (2,4) swing together in x, the pairs half a cycle apart.
HIP_PHASE = {"hip_1": 0.0, "hip_2": math.pi, "hip_3": math.pi, "hip_4": 0.0}
ANKLE_LEAD = math.pi / 2  # ankle phase = its leg's hip phase + pi/2


def _gait(hip: float, ankle: float, frequency_hz: float, kp: float, kd: float, power: float) -> Gait:
    neutral = neutral_offsets()
    joints = []
    for name in M.FORCE_JOINTS:
        hip_phase = HIP_PHASE["hip_" + name.split("_")[1]]
        is_hip = name.startswith("hip")
        joints.append({
            "name": name,
            "amplitude": hip if is_hip else ankle,
            "offset": neutral[name],
            "phase": hip_phase if is_hip else hip_phase + ANKLE_LEAD,
        })
    return Gait.model_validate({
        "frequency_hz": frequency_hz, "power": power, "kp": kp, "kd": kd, "tilt_gain": 0.0, "joints": joints,
    })


def zero_gait() -> Gait:
    return _gait(0.0, 0.0, 1.5, kp=5.0, kd=0.5, power=1.0)


def trot_gait(kd: float = 0.5) -> Gait:
    return _gait(0.5, 0.4, 1.5, kp=5.0, kd=kd, power=1.0)


def flail_gait() -> Gait:
    return _gait(1.2, 1.2, 3.0, kp=20.0, kd=0.1, power=5.0)


FIXTURES = {"zero": (zero_gait, "showcase"), "trot": (trot_gait, "showcase"), "flail": (flail_gait, "rejected")}


def showcase_task() -> Task:
    return next(t for t in T.build_tasks() if t.split == "showcase")


def export(db_name: str = ADA) -> dict[str, dict]:
    task = showcase_task()
    out = {}
    for name, (make, kind) in FIXTURES.items():
        r = evaluate(make(), task, SEED, record=True)
        vf = r["sanity"]["violation_frame"]
        frames_id = write_frames(r["frames"], f"fixture-{name}", FIXTURE_VERSION, kind, vf, db_name=db_name)
        out[name] = {"frames_id": frames_id, "distance_m": r["distance_m"], "fell": r["fell"],
                     "success": r["success"], "violation": r["sanity"]["violation"], "violation_frame": vf}
    return out


if __name__ == "__main__":
    for name, info in export().items():
        print(f"{name:6} {info['frames_id']}  distance_m={info['distance_m']:.2f} fell={info['fell']} "
              f"success={info['success']} violation={info['violation']!r} violation_frame={info['violation_frame']}")
