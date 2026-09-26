"""A4: compress/provider.py — deterministic, ordered, within budget, tail kept whole."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from compress import provider
from compress.provider import TOKEN_BUDGET, compress, compressor_sha256, estimate_tokens

JOINTS = ("hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4")
GAIT = {
    "frequency_hz": 1.4, "power": 1.0, "kp": 6.0, "kd": 0.3, "tilt_gain": 0.5,
    "joints": [{"name": n, "amplitude": 0.35, "offset": 0.0 if n.startswith("hip") else 0.8,
                "phase": round(1.5708 * k, 4)} for k, n in enumerate(JOINTS)],
}


# --- trace builders -----------------------------------------------------------

def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _model(content: str, calls: list[dict]) -> dict:
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"node": "model", "role": "agent", "model_id": "m", "prompt_hash": "h", "cached": True,
            "cost_usd": 0.0, "usage": {}, "message": msg}


def _tool(call: dict, result: dict, attempt: int = 0, valid: bool | None = None) -> dict:
    fn = call["function"]
    return {"node": "tools", "name": fn["name"], "arguments": fn["arguments"],
            "result": json.dumps(result, separators=(",", ":")), "attempt": attempt, "valid": valid}


def harness_trace(submit_ok: bool = True) -> list[dict]:
    """read_task, one invalid submit, then a valid (or invalid) submit — harness/agent.py format."""
    c0, c1, c2 = _call(0, "read_task", {}), _call(1, "submit_gait", {"power": 9}), _call(2, "submit_gait", GAIT)
    last = {"ok": True} if submit_ok else {"ok": False, "error": "bad"}
    return [
        _model("Let me read the task first.", [c0]),
        _tool(c0, {"slope_deg": 5.0, "friction": 0.8, "target_m": 3.0, "gait_schema": {"x": "y" * 2000}}),
        _model("Trying a gait.", [c1]),
        _tool(c1, {"ok": False, "error": "power out of range"}, attempt=1, valid=False),
        _model("Fixing power.", [c2]),
        _tool(c2, last, attempt=2, valid=submit_ok),
    ]


def openai_trace() -> list[dict]:
    c0, c1 = _call(0, "read_task", {}), _call(1, "submit_gait", GAIT)
    return [
        {"role": "system", "content": "Write a gait for Ada for this task."},
        {"role": "user", "content": "Task t1."},
        {"role": "assistant", "content": "Reading.", "tool_calls": [c0]},
        {"role": "tool", "tool_call_id": "call_0", "content": json.dumps({"slope_deg": 0.0})},
        {"role": "assistant", "content": "Submitting.", "tool_calls": [c1]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True})},
    ]


def oversized_trace(n: int = 60) -> list[dict]:
    steps = []
    for k in range(n):
        c = _call(k, "read_task", {})
        steps += [_model(f"step {k} " + "why " * 300, [c]), _tool(c, {"note": f"r{k} " + "z" * 1000})]
    c = _call(n, "submit_gait", GAIT)
    steps += [_model("final " + "because " * 100, [c]), _tool(c, {"ok": True}, attempt=1, valid=True)]
    return steps


# --- tests --------------------------------------------------------------------

def test_same_input_gives_identical_output():
    for raw in (harness_trace(), openai_trace(), oversized_trace()):
        before = copy.deepcopy(raw)
        a, b = compress(raw), compress(copy.deepcopy(raw))
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert raw == before  # input untouched


def test_order_and_fields_preserved():
    out = compress(harness_trace())
    assert [s["i"] for s in out] == [0, 1, 2]
    assert [s["tool"] for s in out] == ["read_task", "submit_gait", "submit_gait"]
    assert [s["reasoning"] for s in out] == ["Let me read the task first.", "Trying a gait.", "Fixing power."]
    assert [s["result"]["status"] for s in out] == ["ok", "error", "ok"]
    assert [s["decision"] for s in out] == ["continue", "continue", "submitted"]
    assert set(out[0]) == {"i", "reasoning", "tool", "args", "result", "decision"}


def test_openai_message_format():
    out = compress(openai_trace())
    assert [s["tool"] for s in out] == ["read_task", "submit_gait"]
    assert [s["decision"] for s in out] == ["continue", "submitted"]
    assert out[1]["args"] == GAIT


def test_truncation_limits():
    out = compress(harness_trace())
    assert len(out[0]["result"]["summary"]) == 300
    assert out[0]["result"]["summary"].endswith("…")
    long = [_model("x" * 1000, [])]
    step = compress(long)[0]
    assert len(step["reasoning"]) == 400 and step["tool"] is None and step["result"] is None


def test_gave_up_when_nothing_submitted():
    assert compress(harness_trace(submit_ok=False))[-1]["decision"] == "gave_up"
    assert compress([_model("I cannot do this.", [])])[-1]["decision"] == "gave_up"


def test_oversized_trace_stays_under_budget_and_keeps_ends_whole():
    raw = oversized_trace()
    full = compress(raw, budget=None)
    assert estimate_tokens(full) > TOKEN_BUDGET
    out = compress(raw)
    assert estimate_tokens(out) <= TOKEN_BUDGET
    assert len(out) == len(full)  # middle shortened, not dropped
    assert out[-3:] == full[-3:]
    assert out[:2] == full[:2]
    assert out[-1]["tool"] == "submit_gait" and out[-1]["decision"] == "submitted"
    mid_lengths = {len(s["reasoning"]) for s in out[2:-3]}
    assert len(mid_lengths) == 1  # shortened evenly
    assert [s["i"] for s in out] == list(range(len(full)))


def test_gait_args_survive_intact():
    for raw in (harness_trace(), openai_trace(), oversized_trace()):
        submits = [s for s in compress(raw) if s["decision"] == "submitted"]
        assert len(submits) == 1
        assert submits[0]["args"] == GAIT
        assert len(submits[0]["args"]["joints"]) == 8


def test_hash_is_stable():
    h = compressor_sha256()
    assert h == compressor_sha256()
    assert len(h) == 64 and int(h, 16) >= 0
    assert h == hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()


def test_trulens_provider_registers():
    from trulens.core.utils.trace_provider import get_trace_provider

    assert provider.register_trulens_provider() is True
    trace = {"trace_id": "v0-t1-1", "schema": "gait-v1", "raw_steps": harness_trace()}
    p = get_trace_provider(trace)
    out = p.compress_trace(trace)
    assert out["compressed_steps"] == compress(trace["raw_steps"])
    assert out["compressor_sha256"] == compressor_sha256()
