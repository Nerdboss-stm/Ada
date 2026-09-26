import importlib
import json
import uuid
from datetime import datetime, timezone

import pytest

import core.llm as llm
import gate.pipeline as pipeline
from compress import provider as compress
from core.contracts import CARD_STAGES, ChatResult, Edit
from core.db import ADA_TEST
from gate import judge, meta

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
COST = 0.0012
BANNED = ("power", "glitch", "exploit", "cheat")
RAW = [  # one harness step pair: a model call to submit_gait, then its tool reply
    {"node": "model", "message": {"role": "assistant", "content": "Short steps, level torso.",
                                  "tool_calls": [{"id": "c1", "type": "function", "function": {
                                      "name": "submit_gait", "arguments": "{\"frequency_hz\": 1.0}"}}]}},
    {"node": "tools", "name": "submit_gait", "arguments": {"frequency_hz": 1.0},
     "result": {"ok": True, "distance_m": 1.4}},
]


def reply(obj):
    return obj if isinstance(obj, str) else json.dumps(obj)


GOOD = {"goal": 4, "plan": 4, "action": 5, "justification": "Walked forward with one steady plan."}
YES = {"supported": "yes", "reason": "The trace shows steady forward progress."}


class FakeChat:
    """Queue of reply contents; records every call's role and messages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, role, messages, tools=None, **params):
        self.calls.append((role, messages))
        content = reply(self.replies.pop(0))
        return ChatResult(role=role, model_id="test/judge", prompt_hash="h" * 64,
                          message={"role": "assistant", "content": content}, cost_usd=COST, cached=False)


@pytest.fixture
def traces(adb):
    ids = [f"t-a9-{uuid.uuid4().hex[:8]}-{i}" for i in range(2)]
    adb.traces.insert_many([{"trace_id": t, "raw_steps": RAW, "compressed_steps": [], "schema": "gait-v1"}
                            for t in ids])
    yield ids
    adb.traces.delete_many({"trace_id": {"$in": ids}})


def edit():
    return Edit(_id="t-a9-edit", from_version="t-a9-p", origin="model", primitive="rules", op="add",
                new="Keep the torso level.", rationale="Falls come from pitch.", created_at=NOW)


def use(monkeypatch, fake):
    monkeypatch.setattr(llm, "cached_chat", fake)
    return fake


def run_judge(trace_ids):
    return judge.run(edit=edit(), db_name=ADA_TEST, verifier={"trace_ids": trace_ids})


def test_templates_never_mention_banned_words():
    for template in (judge.SYSTEM, judge.USER, meta.SYSTEM, meta.USER):
        assert not [w for w in BANNED if w in template.lower()]


def test_trulens_provider_registered_at_import():
    assert compress._registered is True


def test_judge_reads_compressed_traces_only(monkeypatch, traces):
    fake = use(monkeypatch, FakeChat(GOOD))
    res = run_judge(traces)

    assert res["pass"] and res["reason"] is None
    (role, messages), = fake.calls
    assert role == "judge"
    budget = compress.TOKEN_BUDGET // len(traces)
    expected = [{"trace_id": t, "steps": compress.compress(RAW, budget=budget)} for t in traces]
    assert messages[1]["content"].endswith(judge.dumps(expected))
    assert "\"node\"" not in messages[1]["content"]  # raw step records never reach the model
    assert res["payload"] == {"scores": {"goal": 4, "plan": 4, "action": 5},
                              "justification": GOOD["justification"], "cost_usd": COST,
                              "model_id": "test/judge"}


def test_low_score_fails_with_reason(monkeypatch, traces):
    use(monkeypatch, FakeChat({**GOOD, "plan": 2, "justification": "Changed approach every attempt."}))
    res = run_judge(traces)

    assert not res["pass"]
    assert res["reason"] == "judge scored Plan 2 of 5: Changed approach every attempt."
    assert res["payload"]["scores"]["plan"] == 2 and res["payload"]["cost_usd"] == COST


@pytest.mark.parametrize("content", ["not json", "```json\n{\"goal\": 4}\n```",
                                     json.dumps({**GOOD, "action": 7}), "[1, 2, 3]"])
def test_malformed_judge_json_fails_closed(monkeypatch, traces, content):
    use(monkeypatch, FakeChat(content))
    res = run_judge(traces)

    assert not res["pass"] and res["reason"] == "judge output unreadable"
    assert res["payload"]["scores"] is None and res["payload"]["cost_usd"] == COST


def test_fenced_judge_json_is_read(monkeypatch, traces):
    use(monkeypatch, FakeChat(f"```json\n{json.dumps(GOOD)}\n```"))
    assert run_judge(traces)["pass"]


def test_no_traces_fails_without_a_call(monkeypatch):
    fake = use(monkeypatch, FakeChat())
    res = run_judge(["t-a9-missing"])
    assert not res["pass"] and res["reason"] == "no compressed trace to judge" and fake.calls == []


def test_meta_veto_fails(monkeypatch, traces):
    fake = use(monkeypatch, FakeChat(GOOD, {"supported": "no", "reason": "Plan score ignores two resets."}))
    verifier = {"trace_ids": traces}
    judged = judge.run(edit=edit(), db_name=ADA_TEST, verifier=verifier)
    res = meta.run(db_name=ADA_TEST, verifier=verifier, judge=judged)

    assert not res["pass"]
    assert res["reason"] == "meta-verifier veto: Plan score ignores two resets."
    assert res["payload"]["supported"] == "no" and res["payload"]["cost_usd"] == COST
    role, messages = fake.calls[1]
    assert role == "judge" and judge.dumps({"scores": judged["payload"]["scores"],
                                            "justification": GOOD["justification"]}) in messages[1]["content"]


def test_meta_unreadable_and_missing_scores_fail_closed(monkeypatch, traces):
    fake = use(monkeypatch, FakeChat("maybe"))
    verifier = {"trace_ids": traces}
    judged = {"pass": True, "reason": None, "payload": {"scores": {"goal": 4, "plan": 4, "action": 4},
                                                        "justification": "ok"}}
    res = meta.run(db_name=ADA_TEST, verifier=verifier, judge=judged)
    assert not res["pass"] and res["reason"] == "meta output unreadable" and res["payload"]["cost_usd"] == COST

    res = meta.run(db_name=ADA_TEST, verifier=verifier, judge={"pass": False, "payload": {"scores": None}})
    assert not res["pass"] and res["reason"] == "no judge scores to check" and len(fake.calls) == 1


# --- through the pipeline ---------------------------------------------------

@pytest.fixture
def gate_world(world, adb, monkeypatch):
    """The shared world with the real judge and meta, and a trace document per episode."""
    monkeypatch.setattr(pipeline, "_stage_module", importlib.import_module)
    adb.traces.insert_many([{"trace_id": world.trace_id(t.id), "raw_steps": RAW,
                             "compressed_steps": [], "schema": "gait-v1"} for t in world.tasks])
    return world


def half(tasks, distance=1.5):
    return {t.id: (i < 3, distance) for i, t in enumerate(tasks)}


def test_verifier_failure_still_runs_judge_and_meta(gate_world, monkeypatch):
    world = gate_world
    world.parent_runs(half(world.tasks))
    world.power = 5.0
    fake = use(monkeypatch, FakeChat(GOOD, YES))
    res = world.gate()

    assert res["verdict"] == "rejected"
    assert res["reason"] == world.violation  # the verifier stage's own reason, first in card order
    assert res["stages"] == {"gate.verifier": "fail", "gate.gpa": "pass",
                             "gate.meta": "pass", "gate.constraints": "pass"}
    assert [(s, st) for s, st, _ in world.events()] == [
        (s, st) for s in CARD_STAGES for st in ("start", "fail" if s == "gate.verifier" else "pass")]
    assert [role for role, _ in fake.calls] == ["judge", "judge"]
    fail = next(p for s, st, p in world.events() if st == "fail")
    assert fail["reason"] == world.violation


def test_costs_appear_in_stage_payloads(gate_world, monkeypatch):
    world = gate_world
    world.parent_runs(half(world.tasks))
    world.outcome = {t.id: (True, 2.1) for t in world.tasks}
    use(monkeypatch, FakeChat(GOOD, YES))
    res = world.gate()

    assert res["verdict"] == "accepted" and res["stages"] == {s: "pass" for s in CARD_STAGES}
    payloads = {s: p for s, st, p in world.events() if st == "pass"}
    assert payloads["gate.gpa"]["cost_usd"] == COST and payloads["gate.gpa"]["scores"]["action"] == 5
    assert payloads["gate.meta"]["cost_usd"] == COST and payloads["gate.meta"]["supported"] == "yes"


def test_judge_failure_rejects_with_cost_on_fail_event(gate_world, monkeypatch):
    world = gate_world
    world.parent_runs(half(world.tasks))
    world.outcome = {t.id: (True, 2.1) for t in world.tasks}
    use(monkeypatch, FakeChat({**GOOD, "goal": 1}, YES))
    res = world.gate()

    assert res["verdict"] == "rejected" and res["reason"].startswith("judge scored Goal 1 of 5")
    assert res["stages"] == {"gate.verifier": "pass", "gate.gpa": "fail",
                             "gate.meta": "pass", "gate.constraints": "pass"}
    fail = next(p for s, st, p in world.events() if s == "gate.gpa" and st == "fail")
    assert fail["cost_usd"] == COST and fail["reason"] == res["reason"]
