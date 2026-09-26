import hashlib
import json
import sys
import types

import pytest

from core import llm
from core.contracts import ChatResult, Harness
from core.db import ADA_TEST, db
from harness import agent, guardrails, preview
from harness.guardrails import GuardrailError, content_sha256, load_guardrails, seed_guardrails
from harness.run import run_split
from sim import bounds
from sim import model as M
from sim.gait import neutral_offsets
from sim.tasks import build_tasks

COLLECTIONS = ("runs", "versions", "traces", "events", "checkpoints", "checkpoint_writes", "harness_guardrails")
ALL_TOOLS = ["read_task", "submit_gait", "preview_run", "get_contact_log", "list_my_attempts"]
OWN = {"version_id": "v1"}  # [B4] ada_test is shared: query only this file's documents
OWN_TRACE = {"trace_id": {"$regex": "^v1-"}}
PREVIEW_KEYS = {"ok", "seconds", "distance_m", "fell", "max_tilt_deg", "rhythm", "max_torque_ratio"}


def valid_gait() -> dict:
    return {
        "frequency_hz": 1.0, "power": 1.0, "kp": 5.0, "kd": 0.5, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": v, "phase": 0.0}
                   for n, v in neutral_offsets().items()],
    }


def turns(script: list[tuple[str, dict]]):
    """A fake model: the i-th assistant turn calls script[i] (the last entry repeats)."""
    calls = []

    def fake(role, messages, tools=None, **params):
        calls.append([t["function"]["name"] for t in tools or []])
        turn = sum(m["role"] == "assistant" for m in messages)
        name, args = script[min(turn, len(script) - 1)]
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{turn}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}
        return ChatResult(role=role, model_id="test/agent", prompt_hash=f"h{len(calls)}", message=msg,
                          usage={"total_tokens": 10}, cost_usd=0.0, cached=False)

    fake.calls = calls
    return fake


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    database = db(ADA_TEST)
    for c in COLLECTIONS:
        database[c].drop()
    seed_guardrails(ADA_TEST)

    evaluated = []

    def evaluate(gait, task, seed, record=False):  # scoring stub; previews never go through it
        evaluated.append(seed)
        return {"distance_m": 1.0, "fell": False,
                "sanity": {"pass": True, "violation": None, "violation_frame": None}, "success": True}

    monkeypatch.setitem(sys.modules, "sim.verifier", types.SimpleNamespace(evaluate=evaluate))
    yield types.SimpleNamespace(db=database, evaluated=evaluated, monkeypatch=monkeypatch)
    for c in COLLECTIONS:
        database[c].drop()


def task():
    return next(t for t in build_tasks() if t.split == "train")


def run(env, model, tools, tasks=None):
    env.monkeypatch.setattr(llm, "cached_chat", model)
    harness = Harness(tools=tools, model_per_step={"agent": "agent_v0"}, engine={"max_attempts": 2})
    return run_split("v1", "train", 1, tasks=tasks or [task()], harness=harness,
                     db_name=ADA_TEST, ckpt_db=ADA_TEST)


def tool_results(env) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in env.db.traces.find_one(OWN_TRACE)["raw_steps"]:
        if s["node"] == "tools":
            out.setdefault(s["name"], []).append(json.loads(s["result"]))
    return out


# --- whitelist ---------------------------------------------------------------

def test_tools_not_in_version_are_refused(env):
    ran = []
    env.monkeypatch.setattr(preview, "run_preview", lambda *a, **k: ran.append(a))
    model = turns([("preview_run", valid_gait()), ("get_contact_log", {}),
                   ("list_my_attempts", {}), ("submit_gait", valid_gait())])
    run(env, model, ["read_task", "submit_gait"])

    assert all(c == ["read_task", "submit_gait"] for c in model.calls)
    results = tool_results(env)
    for name in ("preview_run", "get_contact_log", "list_my_attempts"):
        assert "unknown tool" in results[name][0]["error"]
    assert ran == []
    assert results["submit_gait"] == [{"ok": True}]


def test_tool_outside_whitelist_is_an_error():
    with pytest.raises(ValueError, match="outside the whitelist"):
        agent.bound_tools(Harness(tools=["read_task", "set_power"], model_per_step={"agent": "agent_v0"}))
    assert agent.TOOL_WHITELIST == (
        "read_task", "submit_gait", "preview_run", "get_contact_log", "list_my_attempts",
        "preview_all_seeds", "recall_best_gait")


# --- preview -----------------------------------------------------------------

def test_preview_uses_first_practice_seed_only(env):
    seeds, real_reset, real_rollout = [], M.reset, M.rollout

    def reset(model, data, seed):
        seeds.append(("reset", seed))
        return real_reset(model, data, seed)

    def rollout(model, data, seed, ctrl_fn, steps=M.EPISODE_STEPS, every=M.RECORD_EVERY, on_step=None):
        seeds.append(("rollout", seed, steps))
        return real_rollout(model, data, seed, ctrl_fn, steps=steps, every=every, on_step=on_step)

    env.monkeypatch.setattr(M, "reset", reset)
    env.monkeypatch.setattr(M, "rollout", rollout)
    t = task()
    other = valid_gait() | {"frequency_hz": 1.5}
    model = turns([("get_contact_log", {}), ("preview_run", valid_gait()), ("preview_run", other),
                   ("get_contact_log", {}), ("list_my_attempts", {}), ("submit_gait", valid_gait())])
    run(env, model, ALL_TOOLS, tasks=[t])

    practice = t.practice_seeds[0]
    assert [s for s in seeds if s[0] == "rollout"] == [("rollout", practice, 300)] * 2
    assert {s[1] for s in seeds if s[0] == "reset"} == {practice}  # rollout resets again inside
    assert not {s[1] for s in seeds} & set(t.eval_seeds)
    assert env.evaluated == t.eval_seeds[:1]  # scoring alone sees evaluation seeds

    results = tool_results(env)
    assert results["get_contact_log"][0] == {"ok": False, "error": "no preview in this episode yet"}
    first, second = results["preview_run"]
    for p in (first, second):
        assert set(p) == PREVIEW_KEYS and p["ok"] and p["seconds"] == 3.0
        assert isinstance(p["fell"], bool) and 0.0 <= p["max_tilt_deg"] <= 180.0
        assert list(p["rhythm"]) == list(preview.LEGS)
        assert all(0.0 <= r["duty"] <= 1.0 and r["touchdowns"] >= 0 for r in p["rhythm"].values())
    text = json.dumps(results).lower()
    assert "sanity" not in text and "success" not in text and "violation" not in text

    log = results["get_contact_log"][1]
    assert log["bins"] == 10 and list(log["legs"]) == list(preview.LEGS)
    assert all(len(b) == 10 and all(0.0 <= x <= 1.0 for x in b) for b in log["legs"].values())

    listed = results["list_my_attempts"][0]["attempts"]
    assert [a["n"] for a in listed] == [1, 2]
    assert listed[0]["gait"] == valid_gait() and listed[1]["gait"] == other
    assert [a["preview"] for a in listed] == [{k: v for k, v in p.items() if k != "ok"} for p in (first, second)]


def test_preview_summaries_on_known_frames():
    def frame(c):
        return {"contacts": c}
    frames = [frame([i % 4 < 2, True, False, i >= 10]) for i in range(20)]
    r = preview.rhythm(frames)
    assert r["leg_1"] == {"duty": 0.5, "touchdowns": 4}
    assert r["leg_2"] == {"duty": 1.0, "touchdowns": 0}
    assert r["leg_3"] == {"duty": 0.0, "touchdowns": 0}
    assert r["leg_4"] == {"duty": 0.5, "touchdowns": 1}
    log = preview.contact_log(frames)
    assert log["leg_2"] == [1.0] * 10 and log["leg_4"] == [0.0] * 5 + [1.0] * 5


# --- guardrails --------------------------------------------------------------

def test_seed_is_content_addressed_and_idempotent(env):
    doc_id = seed_guardrails(ADA_TEST)
    assert env.db.harness_guardrails.count_documents({}) == 1
    doc = env.db.harness_guardrails.find_one()
    assert doc["_id"] == doc_id == content_sha256(doc["content"])
    c = doc["content"]
    assert c["tool_whitelist"] == list(agent.TOOL_WHITELIST)
    assert c["sanity_bounds"] == bounds.SANITY_BOUNDS and c["rated_power"] == bounds.RATED_POWER
    assert c["verifier_sha"] == hashlib.sha256(guardrails.VERIFIER_PATH.read_bytes()).hexdigest()
    assert (c["compressor_sha"] is None) == (not guardrails.COMPRESS_DIR.is_dir())
    assert [g.id for g in load_guardrails(ADA_TEST)] == [doc_id]


def _flip_last_hex(s: str) -> str:
    return s[:-1] + ("0" if s[-1] != "0" else "1")


@pytest.mark.parametrize("tamper", [
    lambda c: {"content.verifier_sha": _flip_last_hex(c["verifier_sha"])},
    lambda c: {"content.rated_power": 5.0},
    lambda c: {"content.tool_whitelist": c["tool_whitelist"] + ["set_power"]},
], ids=["verifier_sha", "rated_power", "tool_whitelist"])
def test_tampered_guardrail_refuses_to_start(env, tamper):
    doc = env.db.harness_guardrails.find_one()
    env.db.harness_guardrails.update_one({"_id": doc["_id"]}, {"$set": tamper(doc["content"])})
    model = turns([("submit_gait", valid_gait())])

    with pytest.raises(GuardrailError, match="does not match"):
        run(env, model, ALL_TOOLS)
    assert model.calls == [] and env.evaluated == []
    assert env.db.runs.count_documents(OWN) == 0 and env.db.events.count_documents(OWN) == 0


def test_missing_guardrails_refuse_to_start(env):
    env.db.harness_guardrails.drop()
    model = turns([("submit_gait", valid_gait())])
    with pytest.raises(GuardrailError, match="no harness_guardrails"):
        run(env, model, ALL_TOOLS)
    assert model.calls == []
