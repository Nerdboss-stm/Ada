"""B10: previews never exhaust the episode; attempts count only submissions; holdout runs in parallel."""

import json
import threading
import time

import pytest

from core import llm
from core.contracts import ChatResult, Harness
from core.db import ADA_TEST, db
from harness import agent, preview
from harness import run as run_mod
from sim.gait import neutral_offsets
from sim.tasks import build_tasks

PREFIX = "b10"  # [B4] ada_test is shared: this file touches only documents whose ids start with b10
OWN_THREAD = {"$regex": f"^{PREFIX}"}
MAX_ATTEMPTS = 3
HARNESS = Harness(tools=["read_task", "submit_gait", "preview_run"], model_per_step={"agent": "agent_v0"},
                  engine={"temperature": 0, "max_attempts": MAX_ATTEMPTS})


def valid_gait() -> dict:
    return {
        "frequency_hz": 1.0, "power": 1.0, "kp": 5.0, "kd": 0.5, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": v, "phase": 0.0}
                   for n, v in neutral_offsets().items()],
    }


def invalid_gait() -> dict:
    return {**valid_gait(), "frequency_hz": 10.0}


READ = [("read_task", {})]
PREVIEW = [("preview_run", valid_gait())]
SUBMIT = [("submit_gait", valid_gait())]
BAD_SUBMIT = [("submit_gait", invalid_gait())]


def turns(script: list[list[tuple[str, dict]]]):
    """A fake model: the i-th assistant turn makes the tool calls in script[i] (the last entry repeats)."""
    calls = []

    def fake(role, messages, tools=None, **params):
        turn = sum(m["role"] == "assistant" for m in messages)
        calls.append(turn)
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{turn}_{j}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for j, (name, args) in enumerate(script[min(turn, len(script) - 1)])]}
        return ChatResult(role=role, model_id="test/agent", prompt_hash=f"b10h{len(calls)}", message=msg,
                          usage={"total_tokens": 10}, cost_usd=0.0, cached=False)

    fake.calls = calls
    return fake


def _cleanup():
    database = db(ADA_TEST)
    database.traces.delete_many({"trace_id": OWN_THREAD})
    database.checkpoints.delete_many({"thread_id": OWN_THREAD})
    database.checkpoint_writes.delete_many({"thread_id": OWN_THREAD})


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    _cleanup()
    previews = []

    def fake_preview(gait, task, seconds=preview.PREVIEW_S):
        previews.append(gait)
        return ({"seconds": seconds, "distance_m": 0.1, "fell": False, "max_tilt_deg": 1.0, "rhythm": {}},
                {"seconds": seconds, "bins": preview.LOG_BINS, "legs": {}})

    monkeypatch.setattr(preview, "run_preview", fake_preview)
    yield type("Env", (), {"previews": previews, "monkeypatch": monkeypatch})
    _cleanup()


def task():
    return next(t for t in build_tasks() if t.split == "train")


def episode(env, model, version_id):
    env.monkeypatch.setattr(llm, "cached_chat", model)
    ep = agent.run_episode(task(), version_id, HARNESS, db_name=ADA_TEST, ckpt_db=ADA_TEST)
    trace = db(ADA_TEST).traces.find_one({"trace_id": ep.trace_id})
    return ep, trace


def tool_steps(trace, name):
    return [s for s in trace["raw_steps"] if s["node"] == "tools" and s["name"] == name]


def test_six_previews_then_submit_is_accepted(env):
    ep, trace = episode(env, turns([READ, *[PREVIEW] * agent.MAX_PREVIEWS, SUBMIT]), f"{PREFIX}a")

    assert ep.gait is not None and ep.attempts == 1 and ep.end_reason == "submitted"
    assert len(env.previews) == agent.MAX_PREVIEWS
    assert all(json.loads(s["result"])["ok"] for s in tool_steps(trace, "preview_run"))
    assert trace["end_reason"] == "submitted" and trace["previews"] == agent.MAX_PREVIEWS


def test_every_preview_and_every_attempt_still_reaches_submit(env):
    script = [READ, *[PREVIEW] * agent.MAX_PREVIEWS, *[BAD_SUBMIT] * (MAX_ATTEMPTS - 1), SUBMIT]
    ep, trace = episode(env, turns(script), f"{PREFIX}b")

    assert ep.gait is not None and ep.attempts == MAX_ATTEMPTS and ep.end_reason == "submitted"
    assert [(s["attempt"], s["valid"]) for s in tool_steps(trace, "submit_gait")] == [(1, False), (2, False), (3, True)]
    assert agent.recursion_limit(HARNESS) == 2 * (agent.MAX_PREVIEWS + MAX_ATTEMPTS + agent.EXTRA_TURNS) + 1


def test_seventh_preview_returns_the_limit_message(env):
    script = [READ, *[PREVIEW] * (agent.MAX_PREVIEWS + 1), SUBMIT]
    ep, trace = episode(env, turns(script), f"{PREFIX}c")

    results = [json.loads(s["result"]) for s in tool_steps(trace, "preview_run")]
    assert len(results) == agent.MAX_PREVIEWS + 1
    assert all(r["ok"] for r in results[:-1])
    assert results[-1] == {"ok": False, "error": "preview limit reached; submit your gait now"}
    assert len(env.previews) == agent.MAX_PREVIEWS  # the seventh never reached the sim
    assert ep.gait is not None and ep.end_reason == "submitted"


def test_preview_cap_counts_calls_within_one_turn(env):
    ep, trace = episode(env, turns([READ, PREVIEW * (agent.MAX_PREVIEWS + 2), SUBMIT]), f"{PREFIX}d")

    results = [json.loads(s["result"]) for s in tool_steps(trace, "preview_run")]
    assert [r["ok"] for r in results] == [True] * agent.MAX_PREVIEWS + [False, False]
    assert len(env.previews) == agent.MAX_PREVIEWS and ep.end_reason == "submitted"


def test_attempts_count_only_submissions(env):
    ep, trace = episode(env, turns([READ, PREVIEW]), f"{PREFIX}e")  # previews forever, never submits

    assert ep.attempts == 0 and ep.gait is None
    assert all(s["attempt"] == 0 for s in trace["raw_steps"] if s["node"] == "tools")
    assert ep.end_reason == "previews_exhausted" and trace["previews"] == agent.MAX_PREVIEWS

    ep, trace = episode(env, turns([READ, BAD_SUBMIT]), f"{PREFIX}f")
    assert ep.attempts == MAX_ATTEMPTS and ep.end_reason == "attempts_exhausted"
    assert trace["previews"] == 0


def test_parallel_holdout_keeps_task_order(monkeypatch):
    tasks = [t for t in build_tasks() if t.split == "holdout"]
    assert len(tasks) > 1
    threads = set()

    def fake_run_task(t, version_id, harness, split, k, **kw):
        i = tasks.index(t)
        time.sleep(0.05 * (len(tasks) - i))  # the first task finishes last
        threads.add(threading.current_thread().name)
        return {"task_id": t.id}

    monkeypatch.setattr(run_mod, "load_guardrails", lambda db_name: None)
    monkeypatch.setattr(run_mod, "run_task", fake_run_task)
    monkeypatch.setattr(run_mod, "_write_version", lambda vid, h, split, results, db_name: results)

    results = run_mod.run_split(f"{PREFIX}p", "holdout", 3, tasks=tasks, harness=HARNESS, db_name=ADA_TEST)

    assert [r["task_id"] for r in results] == [t.id for t in tasks]
    assert len(threads) > 1
