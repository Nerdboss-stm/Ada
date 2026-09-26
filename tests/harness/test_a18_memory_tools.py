"""A18: preview_all_seeds (every practice seed, one preview toward the cap) and recall_best_gait
(honest train gaits re-run on practice seeds; evaluation-seed distances never returned)."""

import json
import uuid

import pytest
from langgraph.checkpoint.mongodb import MongoDBSaver

from core import llm
from core.contracts import ChatResult, Harness, Task
from core.db import ADA_TEST, client, db
from harness import agent, preview
from sim import model as M
from sim.gait import Gait, neutral_offsets

SLOPE, FRICTION = 7.3, 0.37  # far from every real train task, so nearest-task ranking sees ours first
PRACTICE = [10000, 10001, 10002]
EVAL_DISTANCE = 987.654  # an evaluation-seed distance that must never reach the agent


def gait(freq=1.0) -> dict:
    return {
        "frequency_hz": freq, "power": 1.0, "kp": 5.0, "kd": 0.5, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": v, "phase": 0.0}
                   for n, v in neutral_offsets().items()],
    }


def turns(script: list[tuple[str, dict]]):
    """A fake model: the i-th assistant turn calls script[i] (the last entry repeats)."""
    def fake(role, messages, tools=None, **params):
        turn = sum(m["role"] == "assistant" for m in messages)
        name, args = script[min(turn, len(script) - 1)]
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{turn}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}
        return ChatResult(role=role, model_id="test/agent", prompt_hash=f"h{turn}", message=msg,
                          usage={"total_tokens": 10}, cost_usd=0.0, cached=False)
    return fake


@pytest.fixture
def own(monkeypatch):
    """This file's tasks and runs in ada_test, all tagged a18-<tag>; only those are deleted."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    tag = f"a18-{uuid.uuid4().hex[:8]}"
    database = db(ADA_TEST)

    def task(name="t", slope=SLOPE, friction=FRICTION, insert=True) -> Task:
        t = Task.model_validate({"_id": f"{tag}-{name}", "split": "train", "slope_deg": slope,
                                 "friction": friction, "target_m": 5.0, "eval_seeds": [0, 1, 2],
                                 "practice_seeds": PRACTICE})
        if insert:
            database.tasks.insert_one(t.model_dump(by_alias=True))
        return t

    def run(task_id, g, split="train", passed=True, distance=EVAL_DISTANCE):
        database.runs.insert_one({
            "task_id": task_id, "seed": 0, "split": split, "version_id": tag, "model_id": "test/agent",
            "gait": g, "distance_m": distance, "fell": False,
            "sanity": {"pass": passed, "violation": None if passed else "test", "violation_frame": None},
            "success": passed, "cost_usd": 0.0, "tokens": 0, "trace_id": f"{tag}-trace",
        })

    ns = type("Own", (), {"tag": tag, "db": database, "task": staticmethod(task), "run": staticmethod(run)})
    yield ns
    database.runs.delete_many({"version_id": tag})
    database.tasks.delete_many({"_id": {"$regex": f"^{tag}-"}})
    database.traces.delete_many({"trace_id": {"$regex": f"^{tag}-"}})


def fake_preview_all(calls, torque=None):
    """Practice mean = 10 x frequency_hz, so a leaked high-frequency gait would rank first."""
    def fake(g, task, seconds=preview.PREVIEW_S):
        calls.append(g.frequency_hz)
        mean = round(10 * g.frequency_hz, 4)
        seeds = [{"seed": s, "distance_m": mean, "fell": False} for s in task.practice_seeds]
        return {"seconds": seconds, "seeds": seeds, "mean_distance_m": mean, "min_distance_m": mean,
                "max_torque_ratio": (torque or {}).get(g.frequency_hz, 0.5)}, []
    return fake


def recall(task) -> list[dict]:
    out = agent.TOOLS["recall_best_gait"].run(agent.ToolCtx(task, [], ADA_TEST), {})
    assert out.result["ok"] and out.gait is None and out.preview is None
    return out.result["gaits"]


def test_recall_never_reads_holdout_swap_showcase_or_physics_failing_runs(own, monkeypatch):
    t = own.task()
    own.run(t.id, gait(1.1))                        # the one honest run
    own.run(t.id, gait(1.2), split="holdout", distance=50)
    own.run(t.id, gait(1.3), split="swap", distance=50)
    own.run(t.id, gait(1.4), split="showcase", distance=50)
    own.run(t.id, gait(1.5), passed=False, distance=60)
    own.run(t.id, {})
    own.run(t.id, None)
    calls = []
    monkeypatch.setattr(preview, "run_preview_all", fake_preview_all(calls))

    candidates = agent.recall_candidates(t, ADA_TEST)
    assert candidates[0] == (t.id, Gait.model_validate(gait(1.1)))
    freqs = {g.frequency_hz for _, g in candidates}
    assert not freqs & {1.2, 1.3, 1.4, 1.5}
    assert not set(calls) & {1.2, 1.3, 1.4, 1.5}  # never even re-run

    returned = recall(t)
    assert 1.1 in {g["gait"]["frequency_hz"] for g in returned if g["task_id"] == t.id}
    assert not {g["gait"]["frequency_hz"] for g in returned} & {1.2, 1.3, 1.4, 1.5}


def test_recall_prefers_same_task_then_nearest(own, monkeypatch):
    t, near, far = own.task(), own.task("near", friction=FRICTION + 0.01), own.task("far", slope=SLOPE + 30)
    own.run(far.id, gait(1.3))
    own.run(near.id, gait(1.2))
    own.run(t.id, gait(1.1))
    own.run(t.id, gait(1.1))  # a duplicate gait is re-run once
    candidates = agent.recall_candidates(t, ADA_TEST)
    assert [(tid, g.frequency_hz) for tid, g in candidates[:2]] == [(t.id, 1.1), (near.id, 1.2)]
    assert (far.id, 1.3) in [(tid, g.frequency_hz) for tid, g in candidates]
    assert agent.recall_candidates(t, ADA_TEST, pool=1) == candidates[:1]

    calls = []
    monkeypatch.setattr(preview, "run_preview_all", fake_preview_all(calls, torque={1.3: 1.5}))
    returned = recall(t)
    assert calls.count(1.1) == 1
    assert len(returned) <= agent.RECALL_TOP
    assert all(g["gait"]["frequency_hz"] != 1.3 for g in returned)  # over the torque ratio: dropped
    means = [g["mean_distance_m"] for g in returned]
    assert means == sorted(means, reverse=True)


def test_recall_returns_practice_seed_numbers_only(own, monkeypatch):
    t = own.task()
    own.run(t.id, gait(1.0), distance=EVAL_DISTANCE)
    seeds, real_reset = [], M.reset

    def reset(model, data, seed):
        seeds.append(seed)
        return real_reset(model, data, seed)

    monkeypatch.setattr(M, "reset", reset)
    returned = recall(t)
    ours = next(g for g in returned if g["task_id"] == t.id)
    expected, _ = preview.run_preview_all(Gait.model_validate(gait(1.0)), t)
    assert {k: ours[k] for k in expected} == expected
    assert [s["seed"] for s in ours["seeds"]] == PRACTICE
    assert set(seeds) <= set(PRACTICE)
    assert str(EVAL_DISTANCE) not in json.dumps(returned)
    assert "distance_m" not in ours and "sanity" not in json.dumps(returned)


def test_preview_all_seeds_covers_every_practice_seed_and_shares_the_cap(own, monkeypatch):
    t = own.task(insert=False)
    rolled, real_rollout = [], M.rollout

    def rollout(model, data, seed, *a, **k):
        rolled.append(seed)
        return real_rollout(model, data, seed, *a, **k)

    monkeypatch.setattr(M, "rollout", rollout)
    script = ([("preview_all_seeds", gait())] * 3 + [("preview_run", gait())] * 3
              + [("preview_all_seeds", gait()), ("get_contact_log", {}), ("submit_gait", gait())])
    monkeypatch.setattr(llm, "cached_chat", turns(script))
    harness = Harness(tools=["read_task", "submit_gait", "preview_run", "preview_all_seeds", "get_contact_log"],
                      model_per_step={"agent": "agent_v0"}, engine={"max_attempts": 2})
    tid = agent.thread_id(own.tag, t)
    try:
        ep = agent.run_episode(t, own.tag, harness, db_name=ADA_TEST, ckpt_db=ADA_TEST)
    finally:
        MongoDBSaver(client(), db_name=ADA_TEST).delete_thread(tid)

    assert ep.end_reason == "submitted"
    assert rolled == PRACTICE * 3 + [PRACTICE[0]] * 3  # the 7th preview never rolls out
    trace = own.db.traces.find_one({"trace_id": tid})
    assert trace["previews"] == agent.MAX_PREVIEWS
    results = [json.loads(s["result"]) for s in trace["raw_steps"] if s["node"] == "tools"]
    for r in results[:3]:
        assert r["ok"] and [s["seed"] for s in r["seeds"]] == PRACTICE
        assert r["mean_distance_m"] == round(sum(s["distance_m"] for s in r["seeds"]) / 3, 4)
        assert r["min_distance_m"] == min(s["distance_m"] for s in r["seeds"])
        assert 0 < r["max_torque_ratio"]
    assert results[6] == {"ok": False, "error": agent.PREVIEW_LIMIT_MSG}
    assert results[7]["ok"] and results[7]["bins"] == preview.LOG_BINS
