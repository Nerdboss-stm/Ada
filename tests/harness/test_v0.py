import json
import sys
import types

import pytest

from core import llm
from core.contracts import ChatResult, Harness
from core.db import ADA_TEST, db
from harness import agent
from harness.guardrails import seed_guardrails
from harness.run import run_split
from sim.gait import neutral_offsets
from sim.tasks import build_tasks

COLLECTIONS = ("runs", "versions", "traces", "events", "checkpoints", "checkpoint_writes", "harness_guardrails")
CALL_COST = 0.001
OWN = {"version_id": "v0"}  # [B4] ada_test is shared: query only this file's documents
OWN_TRACE = {"trace_id": {"$regex": "^v0-"}}


def valid_gait() -> dict:
    return {
        "frequency_hz": 1.0, "power": 1.0, "kp": 5.0, "kd": 0.5, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": v, "phase": 0.0}
                   for n, v in neutral_offsets().items()],
    }


def invalid_gait() -> dict:
    return {**valid_gait(), "frequency_hz": 10.0}


def call(name: str, args: dict, i: int) -> dict:
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(submissions: list[dict]):
    """A fake model: read_task first, then submissions[i] on the i-th submit turn."""
    calls = []

    def fake(role, messages, tools=None, **params):
        calls.append({"role": role, "tools": [t["function"]["name"] for t in tools or []], **params})
        turn = sum(m["role"] == "assistant" for m in messages)
        if turn == 0:
            msg = {"role": "assistant", "content": None, "tool_calls": [call("read_task", {}, 0)]}
        else:
            gait = submissions[min(turn - 1, len(submissions) - 1)]
            msg = {"role": "assistant", "content": None, "tool_calls": [call("submit_gait", gait, turn)]}
        return ChatResult(
            role=role, model_id="test/agent", prompt_hash=f"h{len(calls)}", message=msg,
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            cost_usd=CALL_COST, cached=False,
        )

    fake.calls = calls
    return fake


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    database = db(ADA_TEST)
    for c in COLLECTIONS:
        database[c].drop()
    seed_guardrails(ADA_TEST)

    evaluated, written = [], []

    def evaluate(gait, task, seed, record=False):
        evaluated.append({"task": task.id, "seed": seed, "record": record, "gait": gait})
        out = {"distance_m": 2.5, "fell": False,
               "sanity": {"pass": True, "violation": None, "violation_frame": None}, "success": True}
        if record:
            out.update(frames=[{"torso": [0.0, 0.0, 0.5]}], frames_sha256="abc")
        return out

    def write_frames(frames, run_id, version_id, kind, violation_frame=None, db_name="ada"):
        written.append({"frames": frames, "run_id": run_id, "version_id": version_id,
                        "kind": kind, "violation_frame": violation_frame, "db_name": db_name})
        return f"frames-{run_id}"

    monkeypatch.setitem(sys.modules, "sim.verifier", types.SimpleNamespace(evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "sim.record", types.SimpleNamespace(write_frames=write_frames))
    yield types.SimpleNamespace(db=database, evaluated=evaluated, written=written, monkeypatch=monkeypatch)
    for c in COLLECTIONS:
        database[c].drop()


def train_tasks(n=2):
    return [t for t in build_tasks() if t.split == "train"][:n]


def run(env, model, split="train", tasks=None, k=2, **kw):
    env.monkeypatch.setattr(llm, "cached_chat", model)
    return run_split("v0", split, k, tasks=tasks if tasks is not None else train_tasks(),
                     db_name=ADA_TEST, ckpt_db=ADA_TEST, **kw)


def test_valid_first_try_one_run_per_seed_and_v0(env):
    model = scripted([valid_gait()])
    tasks = train_tasks()
    version = run(env, model, tasks=tasks)

    runs = list(env.db.runs.find(OWN).sort([("task_id", 1), ("seed", 1)]))
    assert len(runs) == 4
    assert {(r["task_id"], r["seed"]) for r in runs} == {(t.id, s) for t in tasks for s in t.eval_seeds[:2]}
    assert [(e["task"], e["seed"]) for e in env.evaluated] == [(t.id, s) for t in tasks for s in t.eval_seeds[:2]]
    assert all(not e["record"] for e in env.evaluated)
    # one episode = read_task turn + submit turn = 2 calls; split over k=2 seeds
    assert all(r["cost_usd"] == pytest.approx(CALL_COST) and r["tokens"] == 120 for r in runs)
    assert all(r["success"] and r["model_id"] == "test/agent" and r["gait"] == valid_gait() for r in runs)
    tid = f"v0-{tasks[0].id}-{tasks[0].practice_seeds[0]}"
    assert {r["trace_id"] for r in runs if r["task_id"] == tasks[0].id} == {tid}

    doc = env.db.versions.find_one({"_id": "v0"})
    assert doc["status"] == "baseline"
    assert doc["harness"] == agent.V0_HARNESS.model_dump()
    assert doc["metrics"]["train_reliability"] == 1.0
    assert doc["metrics"]["mean_distance_m"] == 2.5
    assert doc["metrics"]["cost_per_run_usd"] == pytest.approx(2 * CALL_COST)  # per episode, not per seed
    assert doc["metrics"]["n"] == 4
    assert version.metrics.n == 4

    assert all(c["role"] == "agent_v0" and c["tools"] == ["read_task", "submit_gait"] for c in model.calls)
    assert all(c["temperature"] == 0 for c in model.calls)

    trace = env.db.traces.find_one({"trace_id": tid})
    nodes = [s["node"] for s in trace["raw_steps"]]
    assert nodes == ["model", "tools", "model", "tools"]
    read = json.loads(trace["raw_steps"][1]["result"])
    assert read["slope_deg"] == tasks[0].slope_deg and read["target_m"] == tasks[0].target_m
    assert read["neutral_offsets"] == neutral_offsets()
    assert "joints" in read["gait_schema"]["properties"]
    assert "eval_seeds" not in json.dumps(read)

    events = list(env.db.events.find({**OWN, "stage": "harness"}).sort("ts", 1))
    assert [e["status"] for e in events] == ["start", "pass", "start", "pass"]
    assert env.db.checkpoints.count_documents({"thread_id": tid}) > 0


def test_invalid_then_valid_uses_two_attempts(env):
    run(env, scripted([invalid_gait(), valid_gait()]), tasks=train_tasks(1))

    runs = list(env.db.runs.find(OWN))
    assert len(runs) == 2 and all(r["success"] for r in runs)
    trace = env.db.traces.find_one({"trace_id": runs[0]["trace_id"]})
    submits = [s for s in trace["raw_steps"] if s.get("name") == "submit_gait"]
    assert [(s["attempt"], s["valid"]) for s in submits] == [(1, False), (2, True)]
    assert "frequency_hz" in json.loads(submits[0]["result"])["error"]
    end = env.db.events.find_one({**OWN, "stage": "harness", "status": "pass"})
    assert end["payload"]["attempts"] == 2


def test_max_attempts_exceeded_is_a_failed_run(env):
    model = scripted([invalid_gait()])
    run(env, model, tasks=train_tasks(1))

    runs = list(env.db.runs.find(OWN))
    assert len(runs) == 2
    assert all(not r["success"] and r["distance_m"] == 0.0 and r["gait"] == {} for r in runs)
    assert env.evaluated == []
    assert len(model.calls) == 1 + 3  # read_task, then 3 attempts
    end = env.db.events.find_one({**OWN, "stage": "harness", "status": "fail"})
    assert end["payload"]["attempts"] == 3 and end["payload"]["valid_gait"] is False
    assert env.db.versions.find_one({"_id": "v0"})["metrics"]["train_reliability"] == 0.0


def test_text_reply_without_tool_call_ends_episode(env):
    def talker(role, messages, tools=None, **params):
        return ChatResult(role=role, model_id="test/agent", prompt_hash="t",
                          message={"role": "assistant", "content": "hello"}, usage={},
                          cost_usd=0.0, cached=False)

    run(env, talker, tasks=train_tasks(1))
    assert [r["success"] for r in env.db.runs.find(OWN)] == [False, False]


def test_only_bound_tools_are_sent_and_run(env):
    harness = Harness(tools=["submit_gait"], model_per_step={"agent": "agent_v0"},
                      engine={"max_attempts": 2})
    model = scripted([valid_gait()])  # its first turn calls read_task, which is not bound
    run(env, model, tasks=train_tasks(1), harness=harness)

    assert all(c["tools"] == ["submit_gait"] for c in model.calls)
    trace = env.db.traces.find_one(OWN_TRACE)
    first = trace["raw_steps"][1]
    assert first["name"] == "read_task" and "unknown tool" in json.loads(first["result"])["error"]
    assert all(r["success"] for r in env.db.runs.find(OWN))


def test_showcase_records_frames_through_sim_record(env):
    showcase = [t for t in build_tasks() if t.split == "showcase"]
    run(env, scripted([valid_gait()]), split="showcase", tasks=showcase)

    runs = list(env.db.runs.find(OWN))
    assert len(runs) == 1 and runs[0]["seed"] == showcase[0].eval_seeds[0]
    assert env.evaluated[0]["record"] is True
    assert len(env.written) == 1
    w = env.written[0]
    assert w == {"frames": [{"torso": [0.0, 0.0, 0.5]}], "run_id": str(runs[0]["_id"]),
                 "version_id": "v0", "kind": "showcase", "violation_frame": None, "db_name": ADA_TEST}
    assert env.db.versions.find_one({"_id": "v0"})["showcase_frames_id"] == f"frames-{runs[0]['_id']}"


def test_rerun_replaces_runs_for_same_split(env):
    run(env, scripted([valid_gait()]), tasks=train_tasks(1))
    run(env, scripted([valid_gait()]), tasks=train_tasks(1))
    assert env.db.runs.count_documents(OWN) == 2
    assert env.db.versions.find_one({"_id": "v0"})["metrics"]["n"] == 2


def holdout_tasks(n=1):
    return [t for t in build_tasks() if t.split == "holdout"][:n]


def test_holdout_counts_a_run_without_gait_as_minus_target(env):
    task = holdout_tasks()[0]
    run(env, scripted([invalid_gait()]), split="holdout", tasks=[task])
    assert env.db.versions.find_one({"_id": "v0"})["metrics"]["mean_distance_m"] == -task.target_m


def test_holdout_counts_a_violating_run_as_minus_target(env):
    task = holdout_tasks()[0]

    def evaluate(gait, task, seed, record=False):
        ok = seed == task.eval_seeds[0]  # the second seed breaks a physical bound
        return {"distance_m": 9.0 if not ok else 2.5, "fell": False, "success": ok,
                "sanity": {"pass": ok, "violation": None if ok else "joint torque",
                           "violation_frame": None if ok else 3}}

    env.monkeypatch.setitem(sys.modules, "sim.verifier", types.SimpleNamespace(evaluate=evaluate))
    run(env, scripted([valid_gait()]), split="holdout", tasks=[task])
    assert env.db.versions.find_one({"_id": "v0"})["metrics"]["mean_distance_m"] == round((2.5 - task.target_m) / 2, 4)


def test_train_mean_distance_stays_raw(env):
    run(env, scripted([invalid_gait()]), tasks=train_tasks(1))
    assert env.db.versions.find_one({"_id": "v0"})["metrics"]["mean_distance_m"] == 0.0
