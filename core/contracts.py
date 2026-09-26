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
# [B11] runs only: "swap" runs use holdout tasks but never feed a version's metrics.
RunSplit = Literal["train", "holdout", "showcase", "swap"]
RoundStatus = Literal["open", "closed"]
VersionStatus = Literal["baseline", "frontier", "candidate", "accepted", "rejected"]
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
    stop_reason: str | None = None
    spent_usd: float = 0.0


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
    # gate-subset train mean; None when a gate task had no gait. Holdout never writes it.
    train_mean_distance_m: float | None = None


class Version(_Doc):
    id: DocId = Field(alias="_id")
    parent: str | None = None
    status: VersionStatus
    harness: Harness
    metrics: Metrics | None = None
    showcase_frames_id: str | None = None
    created_at: datetime


class Baseline(_Doc):
    """One frozen `baselines` document per version (scripts/baseline.py, SPEC §5)."""

    id: DocId = Field(alias="_id")  # = version_id
    version_id: str
    split: Split
    k: int
    metrics: Metrics
    sha256: str  # over the version's sorted run results on `split`
    created_at: datetime


class Edit(_Doc):
    id: DocId = Field(alias="_id")
    round_id: str | None = None
    from_version: str
    to_version: str | None = None
    origin: EditOrigin
    primitive: str
    op: str | None = None
    path: str | None = None
    old: Any = None
    new: Any = None
    rationale: str = ""
    evidence_trace_ids: list[str] = Field(default_factory=list)
    predicted_delta: float | None = None
    predicted_delta_m: float | None = None  # supervisor's bet: change in train mean distance, meters
    actual_delta: float | None = None
    actual_delta_m: float | None = None  # candidate minus parent train mean distance, meters
    attempt_frames_id: str | None = None  # candidate on train-s0-f1, first eval seed (kind showcase)
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
    split: RunSplit
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
    peak_torque: dict[str, float] = Field(default_factory=dict)  # joint -> peak |torque| / rated


class Trace(_Doc):
    id: DocId | None = Field(default=None, alias="_id")
    trace_id: str
    raw_steps: list[dict[str, Any]] = Field(default_factory=list)
    compressed_steps: list[dict[str, Any]] = Field(default_factory=list)
    schema_: Literal["gait-v1"] = Field(default="gait-v1", alias="schema")
    end_reason: str | None = None  # submitted|attempts_exhausted|previews_exhausted|step_limit|no_tool_call
    previews: int | None = None  # preview_run calls in the episode


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
    violation_frame: int | None = None


class Swap(_Doc):
    """One `swaps` document (scripts/capture_swap.py): v0 (left) and a best version (right)
    re-run back to back on holdout, frames on one fixed holdout task. Flat names for the UI."""

    id: DocId = Field(alias="_id")
    captured_at: datetime
    task_id: str  # the fixed holdout task shown in the lanes
    k: int
    left_version: str
    right_version: str
    left_holdout: int  # holdout tasks passed
    right_holdout: int
    left_frames_id: str | None = None
    right_frames_id: str | None = None
    model_id: str  # v0's agent model id
    right_model_id: str
    verifier_sha: str
    mujoco_version: str
    manifest_version: int | str
    harness_diff: list[str] = Field(default_factory=list)
    left_mean_distance_m: float
    right_mean_distance_m: float
    left_cost_per_run_usd: Usd
    right_cost_per_run_usd: Usd
    n: int  # holdout runs per side (tasks x k)
    fresh_calls: bool = False  # [B11] every agent model call was live (cached_chat fresh=True)
    model_calls: int = 0  # [B11] agent model calls across both sides
    left_violations: int = 0  # [B13] physics-violating runs per side
    right_violations: int = 0


class CostPerGait(_Doc):
    v0: Usd | None = None
    frontier: Usd | None = None
    best: Usd | None = None


class Scoreboard(_Doc):
    """One `scoreboard` document (scripts/scoreboard.py)."""

    id: DocId = Field(alias="_id")
    created_at: datetime
    best_version: str | None = None
    rewrite_cost_usd: Usd
    cost_per_gait: CostPerGait
    overrated_attempts: int
    proposals_total: int
    physics_rejected: int
    physics_rejected_judge_passed: int


class SnapshotGhost(_Doc):
    version_id: str
    frames_id: str | None = None  # the version's showcase_frames_id


class Snapshot(_Doc):
    """One `snapshots` document (scripts/pin_snapshot.py): the ids the demo scenes play,
    frozen at pin time while the loop keeps writing."""

    id: DocId = Field(alias="_id")
    pinned_at: datetime
    best_version: str | None = None  # loop.run.pick_leader
    v0: str | None = None
    frontier: str | None = None
    attempt_edit_ids: list[str] = Field(default_factory=list)  # ui/lib/attempts.ts orderAttempts
    cheat_edit_id: str | None = None  # what api/cheat shows as the card
    swaps_id: str | None = None
    scoreboard_id: str | None = None
    ghosts: list[SnapshotGhost] = Field(default_factory=list)  # oldest first


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
    rated_power: float
    verifier_sha: str
    compressor_sha: str | None = None  # None until compress/ exists


class HarnessGuardrails(_Doc):
    id: DocId = Field(alias="_id")  # sha256(content)
    content: GuardrailContent


# --- §2 ada_cache and core/llm.py -------------------------------------------

LlmRole = Literal["agent_v0", "frontier", "controller", "judge"]  # keys of harness/models.json (§7)


class LlmCacheEntry(_Doc):
    """One `ada_cache.llm_cache` document; unique (model_id, prompt_hash)."""

    id: DocId | None = Field(default=None, alias="_id")
    model_id: str
    prompt_hash: str
    request: dict[str, Any]
    response: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class ChatResult(_Doc):
    """What core.llm.cached_chat returns."""

    role: LlmRole
    model_id: str
    prompt_hash: str
    message: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    cost_usd: Usd
    cached: bool
