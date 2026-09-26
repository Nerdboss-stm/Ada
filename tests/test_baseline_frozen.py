import json
import sys
import types
from datetime import datetime, timezone

import pytest

from core import llm
from core.contracts import ChatResult, Version
from core.db import ADA_TEST, db
from harness.agent import AGENT_STEP, V0_HARNESS
from harness.guardrails import seed_guardrails
from harness.run import load_harness
from scripts import baseline as B
from sim.gait import neutral_offsets
from sim.tasks import build_tasks

IDS = list(B.VERSION_IDS)
OWN_THREADS = {"$regex": "^(frontier|v0)-"}
CALL_COST = 0.001
HOLDOUT = [t for t in build_tasks() if t.split == "holdout"][:2]
SHOWCASE = [t for t in build_tasks() if t.split == "showcase"]


def valid_gait() -> dict:
    return {
        "frequency_hz": 1.0, "power": 1.0, "kp": 5.0, "kd": 0.5, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": v, "phase": 0.0}
                   for n, v in neutral_offsets().items()],
    }


def fake_model():
    """read_task on the first turn, a valid gait on the next; model_id names the role."""
    calls = []

    def fake(role, messages, tools=None, **params):
        calls.append(role)
        turn = sum(m["role"] == "assistant" for m in messages)
        name, args = ("read_task", {}) if turn == 0 else ("submit_gait", valid_gait())
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{turn}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}
        return ChatResult(role=role, model_id=f"test/{role}", prompt_hash=f"h{len(calls)}",
                          message=msg, usage={"total_tokens": 120}, cost_usd=CALL_COST, cached=False)

    fake.calls = calls
    return fake


def cleanup(database):
    database.versions.delete_many({"_id": {"$in": IDS}})
    database.baselines.delete_many({"_id": {"$in": IDS}})
    database.runs.delete_many({"version_id": {"$in": IDS}})
    database.events.delete_many({"version_id": {"$in": IDS}})
    database.traces.delete_many({"trace_id": OWN_THREADS})
    database.checkpoints.delete_many({"thread_id": OWN_THREADS})
    database.checkpoint_writes.delete_many({"thread_id": OWN_THREADS})


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    database = db(ADA_TEST)
    cleanup(database)
    seed_guardrails(ADA_TEST)
    written = []

    def evaluate(gait, task, seed, record=False):
        out = {"distance_m": 2.5, "fell": False,
               "sanity": {"pass": True, "violation": None, "violation_frame": None}, "success": True}
        if record:
            out.update(frames=[{"torso": [0.0, 0.0, 0.5]}], frames_sha256="abc")
        return out

    def write_frames(frames, run_id, version_id, kind, violation_frame=None, db_name="ada"):
        written.append({"version_id": version_id, "kind": kind, "db_name": db_name})
        return f"frames-{version_id}-{run_id}"

    monkeypatch.setitem(sys.modules, "sim.verifier", types.SimpleNamespace(evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "sim.record", types.SimpleNamespace(write_frames=write_frames))
    model = fake_model()
    monkeypatch.setattr(llm, "cached_chat", model)
    yield types.SimpleNamespace(db=database, written=written, model=model)
    cleanup(database)


def seed_v0_showcase(database, frames_id="frames-existing"):
    v0 = Version(_id="v0", status="baseline", harness=V0_HARNESS, showcase_frames_id=frames_id,
                 created_at=datetime.now(timezone.utc))
    database.versions.insert_one(v0.model_dump(by_alias=True))


def baseline(force=False):
    return B.run_baseline(force, holdout_tasks=HOLDOUT, showcase_tasks=SHOWCASE,
                          db_name=ADA_TEST, ckpt_db=ADA_TEST)


def test_frontier_version_uses_frontier_role_and_frontier_frames(env):
    seed_v0_showcase(env.db)
    baseline()

    doc = env.db.versions.find_one({"_id": "frontier"})
    assert doc["status"] == "frontier"
    assert doc["harness"] == {**V0_HARNESS.model_dump(), "model_per_step": {AGENT_STEP: "frontier"}}
    assert load_harness("frontier", ADA_TEST).model_per_step == {AGENT_STEP: "frontier"}
    assert {r["model_id"] for r in env.db.runs.find({"version_id": "frontier"})} == {"test/frontier"}
    assert {r["model_id"] for r in env.db.runs.find({"version_id": "v0"})} == {"test/agent_v0"}
    assert set(env.model.calls) == {"frontier", "agent_v0"}

    # frontier showcase recorded as kind frontier; v0 already had frames, so no v0 showcase run
    assert [(w["version_id"], w["kind"]) for w in env.written] == [("frontier", "frontier")]
    assert doc["showcase_frames_id"].startswith("frames-frontier-")
    assert env.db.versions.find_one({"_id": "v0"})["showcase_frames_id"] == "frames-existing"
    assert env.db.runs.count_documents({"version_id": "v0", "split": "showcase"}) == 0


def test_v0_showcase_runs_only_when_missing(env):
    baseline()
    assert [(w["version_id"], w["kind"]) for w in env.written] == [
        ("frontier", "frontier"), ("v0", "showcase")]


def test_n_is_tasks_times_k_and_metrics_are_holdout(env):
    seed_v0_showcase(env.db)
    out = baseline()
    n = len(HOLDOUT) * B.K

    assert [b.version_id for b in out] == ["frontier", "v0"]
    for vid in IDS:
        m = env.db.versions.find_one({"_id": vid})["metrics"]
        b = env.db.baselines.find_one({"_id": vid})
        assert m["n"] == n and b["metrics"] == m
        assert m["holdout_reliability_80"] == 1.0
        assert m["mean_distance_m"] == 2.5
        assert m["cost_per_run_usd"] == pytest.approx(2 * CALL_COST)  # per episode
        assert b["split"] == "holdout" and b["k"] == 3
        assert env.db.runs.count_documents({"version_id": vid, "split": "holdout"}) == n
    lines = [B.summary_line(b) for b in out]
    assert lines[0].startswith("frontier") and f"n={n}" in lines[0] and "$0.0020" in lines[0]


def test_second_run_without_force_refuses(env):
    seed_v0_showcase(env.db)
    baseline()
    calls = len(env.model.calls)
    frozen = list(env.db.baselines.find({"_id": {"$in": IDS}}).sort("_id", 1))
    run_ids = sorted(r["_id"] for r in env.db.runs.find({"version_id": {"$in": IDS}}))

    with pytest.raises(B.BaselineFrozen):
        baseline()
    assert len(env.model.calls) == calls
    assert list(env.db.baselines.find({"_id": {"$in": IDS}}).sort("_id", 1)) == frozen
    assert sorted(r["_id"] for r in env.db.runs.find({"version_id": {"$in": IDS}})) == run_ids

    assert len(baseline(force=True)) == 2


def test_sha_is_stable_and_covers_run_results(env):
    seed_v0_showcase(env.db)
    first = {b.version_id: b.sha256 for b in baseline()}
    assert first == {vid: B.runs_sha256(vid, "holdout", ADA_TEST) for vid in IDS}

    second = {b.version_id: b.sha256 for b in baseline(force=True)}
    assert second == first

    env.db.runs.update_one({"version_id": "frontier", "split": "holdout"}, {"$set": {"distance_m": 9.0}})
    assert B.runs_sha256("frontier", "holdout", ADA_TEST) != first["frontier"]


def test_load_harness_v0_is_always_v0_harness(env):
    other = V0_HARNESS.model_copy(deep=True, update={"tools": ["submit_gait"]})
    env.db.versions.insert_one(Version(_id="v0", status="baseline", harness=other,
                                       created_at=datetime.now(timezone.utc)).model_dump(by_alias=True))
    assert load_harness("v0", ADA_TEST) == V0_HARNESS
    with pytest.raises(LookupError):
        load_harness("frontier", ADA_TEST)
