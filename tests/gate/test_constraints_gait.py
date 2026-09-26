"""A10: gate.constraints rejects harness text that states literal gait values."""

import uuid
from datetime import datetime, timezone

import pytest

import core.llm as llm
import gate.pipeline as pipeline
import harness.agent as agent
from core.contracts import CARD_STAGES, Edit, Harness, Version
from core.db import ADA_TEST, db
from gate import constraints
from harness.guardrails import COLLECTION, seed_guardrails
from loop.edits import GAIT_VALUE_REASON

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def version(vid, **over):
    base = {"tools": ["read_task", "submit_gait"], "model_per_step": {"agent": "agent_v0"},
            "engine": {"temperature": 0, "max_attempts": 3}}
    return Version(_id=vid, status="candidate", harness=Harness.model_validate({**base, **over}),
                   created_at=NOW)


@pytest.fixture
def adb():
    database = db(ADA_TEST)
    existing = {d["_id"] for d in database[COLLECTION].find({}, {"_id": 1})}
    gid = seed_guardrails(ADA_TEST)
    yield database
    if gid not in existing:
        database[COLLECTION].delete_one({"_id": gid})


def test_reason_text():
    assert GAIT_VALUE_REASON == "states literal gait values"


@pytest.mark.parametrize("over", [
    {"rules": ["Keep the torso level.", "Use kp 5 on slopes."]},
    {"rules": ["Try 1.5Hz strides."]},
    {"context_policy": {"hint": {"frequency_hz": 1.5}}},
    {"context_policy": {"amplitude": 0.8}},
    {"context_policy": {"note": "offsets of 0.3 on the ankles"}},
])
def test_gait_values_reject(adb, over):
    assert constraints.check(version("t-a10-c", **over), ADA_TEST) == GAIT_VALUE_REASON


@pytest.mark.parametrize("over", [
    {},
    {"rules": ["Keep the torso level.", "Use at most 3 previews before submitting."]},
    {"rules": ["Lower the stride frequency on steep slopes."]},
    {"context_policy": {"past_attempts": 2, "telemetry": ["tilt"]}},
])
def test_plain_harness_passes(adb, over):
    assert constraints.check(version("t-a10-c", **over), ADA_TEST) is None


def test_gate_rejects_at_precheck_without_rolling(adb, monkeypatch):
    tag = uuid.uuid4().hex[:8]
    parent_id, cand_id, edit_id, round_id = (f"t-a10-{tag}-{s}" for s in "pcer")
    calls = []
    monkeypatch.setattr(agent, "run_episode", lambda *a, **k: calls.append("episode"))
    monkeypatch.setattr(llm, "cached_chat", lambda *a, **k: calls.append("llm"))

    parent = version(parent_id).model_copy(update={"status": "accepted"})
    cand = version(cand_id, rules=["Use kp 5 on slopes."])
    edit = Edit(_id=edit_id, round_id=round_id, from_version=parent_id, to_version=cand_id,
                origin="model", primitive="rules", op="add", new="Use kp 5 on slopes.",
                predicted_delta=0.1, predicted_delta_m=0.2, created_at=NOW)
    try:
        res = pipeline.run_gate(parent, cand, edit, round_id, db_name=ADA_TEST)
        events = [(e["stage"], e["status"], e["payload"])
                  for e in adb.events.find({"edit_id": edit_id}).sort("_id", 1)]
    finally:
        adb.events.delete_many({"edit_id": edit_id})

    assert res["verdict"] == "rejected" and res["reason"] == GAIT_VALUE_REASON
    assert res["stages"] == {s: ("fail" if s == "gate.constraints" else "skipped") for s in CARD_STAGES}
    assert res["train"]["n"] == 0 and calls == []
    assert events[-1][:2] == ("gate.constraints", "fail")
    assert events[-1][2]["reason"] == GAIT_VALUE_REASON
