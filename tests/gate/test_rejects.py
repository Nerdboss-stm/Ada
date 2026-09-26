import uuid
from datetime import datetime, timezone

import pytest

import core.llm as llm
import gate.pipeline as pipeline
import harness.agent as agent
import sim.verifier as verifier
from core.contracts import CARD_STAGES, Edit, Harness, Version
from core.db import ADA_TEST, db
from gate import verifier_stage
from harness.guardrails import COLLECTION, seed_guardrails
from sim import model as M
from sim.bounds import power_reason
from sim.gait import Gait

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
FRAME = {"geoms": [[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], "contacts": [True] * 4,
         "forces": [0.0] * 8, "torso": [0.0, 0.0, 0.5]}


def gait(power=1.0):
    return Gait.model_validate({
        "frequency_hz": 1.0, "power": power, "kp": 5.0, "kd": 0.1, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": 0.0, "phase": 0.0} for n in M.FORCE_JOINTS],
    })


def harness(**over):
    base = {"tools": ["read_task", "submit_gait"], "model_per_step": {"agent": "agent_v0"},
            "engine": {"temperature": 0, "max_attempts": 3}}
    return Harness.model_validate({**base, **over})


@pytest.fixture
def adb():
    return db(ADA_TEST)


@pytest.fixture
def world(monkeypatch, adb):
    """Own parent/candidate/edit ids; episodes, verifier, LLM and stages 2-3 patched."""
    tag = uuid.uuid4().hex[:8]
    existing = {d["_id"] for d in adb[COLLECTION].find({}, {"_id": 1})}
    gid = seed_guardrails(ADA_TEST)
    tasks = verifier_stage.gate_tasks(ADA_TEST)

    class W:
        parent_id, cand_id = f"t-a8-{tag}-p", f"t-a8-{tag}-c"
        edit_id, round_id = f"t-a8-{tag}-e", f"t-a8-{tag}-r"
        episodes: list[str] = []
        llm_calls: list = []
        power = 1.0
        outcome: dict[str, tuple[bool, float]] = {}  # candidate task_id -> (success, distance)

        @classmethod
        def parent_runs(cls, outcome):
            docs = [{"task_id": t.id, "seed": s, "split": "train", "version_id": cls.parent_id,
                     "success": outcome[t.id][0], "distance_m": outcome[t.id][1]}
                    for t in tasks for s in t.eval_seeds[:verifier_stage.K]]
            adb.runs.insert_many(docs)

        @classmethod
        def gate(cls, **over):
            parent = Version(_id=cls.parent_id, status="accepted", harness=harness(), created_at=NOW)
            cand = Version(_id=cls.cand_id, parent=cls.parent_id, status="candidate",
                           harness=harness(**over), created_at=NOW)
            edit = Edit(_id=cls.edit_id, round_id=cls.round_id, from_version=cls.parent_id,
                        to_version=cls.cand_id, origin="model", primitive="rules", op="add",
                        new="Keep the torso level.", predicted_delta=0.1, created_at=NOW)
            return pipeline.run_gate(parent, cand, edit, cls.round_id, db_name=ADA_TEST)

        @classmethod
        def events(cls):
            return [(e["stage"], e["status"], e["payload"])
                    for e in adb.events.find({"edit_id": cls.edit_id}).sort("_id", 1)]

    W.tasks = tasks

    def run_episode(task, version_id, h, *, db_name, **_):
        W.episodes.append(task.id)
        return agent.EpisodeResult(gait=gait(W.power), attempts=1, cost_usd=0.002, tokens=100,
                                   model_id="test/agent", trace_id=f"{version_id}-{task.id}")

    def evaluate(g, task, seed, record=False):
        if g.power > 1.0:
            out = {"distance_m": 6.0, "fell": False, "success": False,
                   "sanity": {"pass": False, "violation": power_reason(g.power), "violation_frame": 1}}
        else:
            ok, dist = W.outcome[task.id]
            out = {"distance_m": dist, "fell": False, "success": ok,
                   "sanity": {"pass": True, "violation": None, "violation_frame": None}}
        if record:
            out["frames"] = [FRAME] * 3
        return out

    def cached_chat(*a, **k):
        W.llm_calls.append((a, k))
        raise AssertionError("the gate made a model call")

    monkeypatch.setattr(agent, "run_episode", run_episode)
    monkeypatch.setattr(verifier, "evaluate", evaluate)
    monkeypatch.setattr(llm, "cached_chat", cached_chat)
    monkeypatch.setattr(pipeline, "_stage_module", lambda name: None)
    yield W

    ids = [W.parent_id, W.cand_id]
    adb.runs.delete_many({"version_id": {"$in": ids}})
    adb.frames.delete_many({"version_id": W.cand_id})
    adb.events.delete_many({"edit_id": W.edit_id})
    if gid not in existing:
        adb[COLLECTION].delete_one({"_id": gid})


def half(tasks, distance=1.5):
    """First three gate tasks succeed on every seed, the rest fail: reliability 0.5."""
    return {t.id: (i < 3, distance) for i, t in enumerate(tasks)}


def assert_precheck_reject(world, adb, res):
    assert res["verdict"] == "rejected"
    assert res["stages"] == {"gate.verifier": "skipped", "gate.gpa": "skipped",
                             "gate.meta": "skipped", "gate.constraints": "fail"}
    assert world.episodes == [] and world.llm_calls == []
    assert res["train"]["n"] == 0 and res["frames_id"] is None
    assert adb.runs.count_documents({"version_id": world.cand_id}) == 0
    assert [(s, st) for s, st, _ in world.events()] == [
        ("gate.verifier", "info"), ("gate.gpa", "info"), ("gate.meta", "info"),
        ("gate.constraints", "start"), ("gate.constraints", "fail")]
    assert all(p == {"skipped": True} for _, st, p in world.events() if st == "info")


def test_tool_outside_whitelist_rejects_at_constraints(world, adb):
    world.parent_runs(half(world.tasks))
    res = world.gate(tools=["read_task", "submit_gait", "edit_physics"])
    assert_precheck_reject(world, adb, res)
    assert res["reason"] == "tools outside the whitelist: edit_physics"
    assert res["parent_train_reliability"] == 0.5
    assert world.events()[-1][2]["reason"] == res["reason"]


def test_frontier_model_rejects_at_constraints(world, adb):
    world.parent_runs(half(world.tasks))
    res = world.gate(model_per_step={"agent": "frontier"})
    assert_precheck_reject(world, adb, res)
    assert res["reason"] == "model outside the cheap tier"


def test_power_5_rejects_at_verifier_with_frames(world, adb):
    world.parent_runs(half(world.tasks))
    world.power = 5.0
    res = world.gate()

    assert res["verdict"] == "rejected" and res["reason"] == power_reason(5.0)
    assert res["stages"] == {"gate.verifier": "fail", "gate.gpa": "skipped",
                             "gate.meta": "skipped", "gate.constraints": "skipped"}
    assert world.episodes == [world.tasks[0].id]  # stops at the first violation
    assert res["train"]["n"] == 1
    frames = adb.frames.find_one({"_id": res["frames_id"]})
    assert (frames["kind"], frames["violation_frame"], frames["version_id"]) == \
        ("rejected", 1, world.cand_id)
    run = adb.runs.find_one({"version_id": world.cand_id})
    assert frames["run_id"] == str(run["_id"]) and run["sanity"]["violation"] == power_reason(5.0)
    fail = [p for s, st, p in world.events() if st == "fail"]
    assert fail[0]["reason"] == power_reason(5.0) and fail[0]["frames_id"] == res["frames_id"]


def test_no_improvement_rejects_with_both_numbers(world, adb):
    world.parent_runs(half(world.tasks, distance=1.5))
    world.outcome = half(world.tasks, distance=1.55)  # same reliability, +0.05 m
    res = world.gate()

    assert res["verdict"] == "rejected"
    assert res["reason"] == ("no strict improvement: train reliability 0.5000 vs parent 0.5000, "
                             "mean distance 1.55 m vs parent 1.50 m")
    assert res["stages"]["gate.verifier"] == "fail" and res["frames_id"] is None
    assert res["train"]["n"] == 12 and len(world.episodes) == 6


@pytest.mark.parametrize("outcome,rel", [
    (lambda ts: {t.id: (True, 1.5) for t in ts}, 1.0),   # higher reliability
    (lambda ts: half(ts, distance=1.6), 0.5),             # equal reliability, +0.1 m
])
def test_strict_improvement_accepts(world, adb, outcome, rel):
    world.parent_runs(half(world.tasks, distance=1.5))
    world.outcome = outcome(world.tasks)
    res = world.gate()

    assert res["verdict"] == "accepted" and res["reason"] is None
    assert res["stages"] == {s: "pass" for s in CARD_STAGES}
    assert res["train"]["reliability"] == rel and res["train"]["n"] == 12
    assert res["train"]["cost_per_run_usd"] == 0.002
    assert res["parent_train_reliability"] == 0.5 and world.llm_calls == []
    runs = list(adb.runs.find({"version_id": world.cand_id}))
    assert len(runs) == 12 and {r["split"] for r in runs} == {"train"}


def test_events_arrive_in_card_order(world, adb):
    world.parent_runs(half(world.tasks))
    world.outcome = {t.id: (True, 2.1) for t in world.tasks}
    world.gate()

    assert [(s, st) for s, st, _ in world.events()] == [
        (s, st) for s in CARD_STAGES for st in ("start", "pass")]
    docs = list(adb.events.find({"edit_id": world.edit_id}))
    assert {(d["round_id"], d["version_id"]) for d in docs} == {(world.round_id, world.cand_id)}
    payloads = {s: p for s, st, p in world.events() if st == "pass"}
    assert payloads["gate.gpa"] == payloads["gate.meta"] == {"stub": True}
    assert payloads["gate.verifier"]["predicted_delta"] == 0.1
    assert payloads["gate.verifier"]["actual_delta"] == 0.5
