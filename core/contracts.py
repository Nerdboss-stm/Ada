"""Pydantic v2 models for CONTRACTS.md §3 (collections in `ada`) and §4 (events).

The only cross-directory interface. Dump with `model_dump(by_alias=True)` so `_id`,
`pass` and `schema` keep their document names.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, field_validator

# --- shared types -----------------------------------------------------------

# Raw Mongo documents carry bson.ObjectId; store every _id as str.
DocId = Annotated[str, BeforeValidator(str)]
# USD floats rounded to 6 places in documents.
Usd = Annotated[float, AfterValidator(lambda v: round(v, 6))]

Split = Literal["train", "holdout", "showcase"]
RoundStatus = Literal["open", "closed"]
VersionStatus = Literal["baseline", "frontier", "accepted", "rejected"]
EditOrigin = Literal["model", "probe", "cli"]
Verdict = Literal["accepted", "rejected"]
FrameKind = Literal["showcase", "frontier", "rejected"]
Stage = Literal[
    "harness", "baseline", "sensor", "controller",
    "gate.verifier", "gate.gpa", "gate.meta", "gate.constraints", "actuator",
]
EventStatus = Literal["start", "pass", "fail", "info"]

# Card = one edit_id; four cells in this order.
CARD_STAGES: tuple[Stage, ...] = ("gate.verifier", "gate.gpa", "gate.meta", "gate.constraints")
EVENT_PAYLOAD_MAX_BYTES = 2048


class _Doc(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --- §3 collections ---------------------------------------------------------

class Task(_Doc):
    id: DocId = Field(alias="_id")
    split: Split
    slope_deg: float
    friction: float
    target_m: float
    eval_seeds: list[int]
    practice_seeds: list[int]


class Event(_Doc):
    id: DocId | None = Field(default=None, alias="_id")
    ts: datetime
    round_id: str | None = None
    version_id: str | None = None
    edit_id: str | None = None
    stage: Stage
    status: EventStatus
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _payload_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        size = len(json.dumps(v, separators=(",", ":"), default=str).encode())
        if size > EVENT_PAYLOAD_MAX_BYTES:
            raise ValueError(f"payload is {size} bytes; limit {EVENT_PAYLOAD_MAX_BYTES}")
        return v


class Round(_Doc):
    id: DocId = Field(alias="_id")
    status: RoundStatus
    opened_at: datetime
    closed_at: datetime | None = None
    budget_usd: Usd
    budget_s: float
    open_lock: bool | None = None


class Harness(_Doc):
    rules: list[str] = Field(default_factory=list)
    context_policy: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    model_per_step: dict[str, str] = Field(default_factory=dict)
    engine: dict[str, Any] = Field(default_factory=dict)


class Metrics(_Doc):
    train_reliability: float
    holdout_reliability_80: float
    mean_distance_m: float
    cost_per_run_usd: Usd
    n: int


class Version(_Doc):
    id: DocId = Field(alias="_id")
    parent: str | None = None
    status: VersionStatus
    harness: Harness
    metrics: Metrics | None = None
    showcase_frames_id: str | None = None
    created_at: datetime


class Edit(_Doc):
    id: DocId = Field(alias="_id")
    round_id: str | None = None
    from_version: str
    to_version: str | None = None
    origin: EditOrigin
    primitive: str
    old: Any = None
    new: Any = None
    rationale: str = ""
    predicted_delta: float | None = None
    actual_delta: float | None = None
    verdict: Verdict | None = None
    reason: str | None = None
    violation_frame: int | None = None
    frames_id: str | None = None
    created_at: datetime


class Sanity(_Doc):
    pass_: bool = Field(alias="pass")
    violation: str | None = None
    violation_frame: int | None = None


class Run(_Doc):
    id: DocId | None = Field(default=None, alias="_id")
    task_id: str
    seed: int
    split: Split
    version_id: str
    model_id: str
    gait: dict[str, Any]
    distance_m: float
    fell: bool
    sanity: Sanity
    success: bool
    cost_usd: Usd
    tokens: int
    trace_id: str | None = None


class Trace(_Doc):
    id: DocId | None = Field(default=None, alias="_id")
    trace_id: str
    raw_steps: list[dict[str, Any]] = Field(default_factory=list)
    compressed_steps: list[dict[str, Any]] = Field(default_factory=list)
    schema_: Literal["gait-v1"] = Field(default="gait-v1", alias="schema")


GeomPose = Annotated[list[float], Field(min_length=7, max_length=7)]  # x,y,z,qw,qx,qy,qz


class Frame(_Doc):
    geoms: list[GeomPose]
    contacts: Annotated[list[bool], Field(min_length=4, max_length=4)]
    forces: Annotated[list[float], Field(min_length=8, max_length=8)]
    torso: Annotated[list[float], Field(min_length=3, max_length=3)]


class FramesDoc(_Doc):
    id: DocId = Field(alias="_id")
    run_id: str
    version_id: str
    kind: FrameKind
    manifest_version: int | str
    fps: float
    frames: list[Frame]
    sha256: str


class Skill(_Doc):
    id: DocId = Field(alias="_id")
    version_id: str
    task_id: str
    summary: str
    gait: dict[str, Any]
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime


class GuardrailContent(_Doc):
    tool_whitelist: list[str]
    sanity_bounds: dict[str, Any]
    verifier_sha: str
    compressor_sha: str


class HarnessGuardrails(_Doc):
    id: DocId = Field(alias="_id")  # sha256(content)
    content: GuardrailContent
