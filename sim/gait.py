"""Gait schema: the JSON the cheap model writes (SPEC §2.2).

Joints are listed in the manifest's force_joints order (sim.model.FORCE_JOINTS).
Field descriptions are what the model reads; they state the neutral standing pose.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import mujoco
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sim import model as M


@lru_cache(maxsize=1)
def _neutral() -> tuple[tuple[str, float], ...]:
    model = mujoco.MjModel.from_xml_path(str(M.ANT_XML))
    init_qpos = model.numeric("init_qpos").data
    return tuple((n, float(init_qpos[model.jnt_qposadr[model.joint(n).id]])) for n in M.FORCE_JOINTS)


def neutral_offsets() -> dict[str, float]:
    """Joint angles (rad) of the standing pose in ant.xml's init_qpos, in FORCE_JOINTS order."""
    return dict(_neutral())


NEUTRAL_TEXT = ", ".join(f"{n} {v:+.2f}" for n, v in _neutral())

OFFSET_DESC = (
    "Center angle in radians the joint oscillates around. "
    f"Neutral standing offsets (rest pose): {NEUTRAL_TEXT}. "
    "An offset of 0 on an ankle is not the rest pose."
)


class JointGait(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description=f"Joint name; the 8 joints in this order: {', '.join(M.FORCE_JOINTS)}.")
    amplitude: Annotated[float, Field(ge=0.0, le=1.2, description="Sine amplitude in radians, 0 to 1.2.")]
    offset: Annotated[float, Field(ge=-1.0, le=1.0, description=OFFSET_DESC + " Range -1.0 to 1.0.")]
    phase: Annotated[float, Field(ge=0.0, le=6.2832, description="Sine phase in radians, 0 to 6.2832.")]


class Gait(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequency_hz: Annotated[float, Field(ge=0.5, le=3.0, description="Stride frequency in Hz, 0.5 to 3.0.")]
    power: Annotated[float, Field(ge=0.2, le=5.0, description="Scales actuator gear (1.0 = stock Ada), 0.2 to 5.0.")]
    kp: Annotated[float, Field(ge=0.5, le=20.0, description="Proportional gain toward the target angle, 0.5 to 20.")]
    kd: Annotated[float, Field(ge=0.0, le=2.0, description="Damping gain on joint velocity, 0.0 to 2.0.")]
    joints: list[JointGait] = Field(
        description=(
            f"Exactly 8 joints in this order: {', '.join(M.FORCE_JOINTS)}. "
            "Target angle = offset + amplitude * sin(2*pi*frequency_hz*t + phase). "
            f"Neutral standing offsets (rest pose): {NEUTRAL_TEXT}."
        ),
    )
    tilt_gain: Annotated[
        float,
        Field(ge=0.0, le=2.0, description="Hip correction from torso roll and pitch, 0.0 to 2.0; 0 disables it."),
    ]

    @field_validator("joints")
    @classmethod
    def _joint_order(cls, v: list[JointGait]) -> list[JointGait]:
        names = tuple(j.name for j in v)
        if names != M.FORCE_JOINTS:
            raise ValueError(f"joints must be {list(M.FORCE_JOINTS)} in order; got {list(names)}")
        return v
