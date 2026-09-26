"""Trace compressor (SPEC §4 stage 2): what the GPA judge reads.

    compress(raw_steps) -> list[dict]      deterministic, ordered, <= TOKEN_BUDGET
    compressor_sha256() -> str             sha256 of this file; checked by gate.constraints

raw_steps are either harness step records as stored by harness/agent.py
({"node": "model", "message": {...}} / {"node": "tools", "name", "arguments", "result"})
or plain OpenAI chat messages (assistant tool_calls, then role "tool" replies).
Each compressed step: {i, reasoning, tool, args, result {status, summary} | None, decision}.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

TOKEN_BUDGET = 6000  # estimated as characters / 4
REASONING_CHARS = 400
RESULT_CHARS = 300
KEEP_HEAD = 2
KEEP_TAIL = 3
SUBMIT_TOOL = "submit_gait"
SCHEMA = "gait-v1"


def compressor_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def estimate_tokens(steps: list[dict[str, Any]]) -> int:
    return math.ceil(len(_dumps(steps)) / 4)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…" if limit > 0 else ""


def _text(content: Any) -> str:
    """Assistant content as plain text; list content keeps its text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# --- normalize raw steps into turns ------------------------------------------

def _new_call(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    return {"id": call.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments"),
            "result": None, "answered": False}


def _turns(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[{reasoning, calls: [{name, arguments, result, answered}]}] in trace order."""
    turns: list[dict[str, Any]] = []

    def open_turn(message: dict[str, Any]) -> None:
        turns.append({"reasoning": _text(message.get("content")),
                      "calls": [_new_call(c) for c in message.get("tool_calls") or []]})

    def answer(call_id: Any, name: Any, arguments: Any, result: Any, by_position: bool) -> None:
        if not turns:
            turns.append({"reasoning": "", "calls": []})
        calls = turns[-1]["calls"]
        pending = [c for c in calls if not c["answered"]]
        match = next((c for c in pending if call_id is not None and c["id"] == call_id), None)
        if match is None and (by_position or call_id is None) and pending:
            match = pending[0]
        if match is None:
            match = {"id": call_id, "name": name, "arguments": arguments, "result": None, "answered": False}
            calls.append(match)
        if by_position:  # harness tool records hold what actually ran
            match["name"], match["arguments"] = name, arguments
        match["name"] = match["name"] or name
        match["result"], match["answered"] = result, True

    for step in raw_steps:
        node = step.get("node")
        if node == "model":
            open_turn(step.get("message") or {})
        elif node == "tools":
            answer(step.get("tool_call_id"), step.get("name"), step.get("arguments"), step.get("result"), True)
        elif step.get("role") == "assistant":
            open_turn(step)
        elif step.get("role") == "tool":
            answer(step.get("tool_call_id"), step.get("name"), None, step.get("content"), False)
    return turns


# --- one compressed step per tool call ---------------------------------------

def _args(name: str | None, arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return arguments if name == SUBMIT_TOOL else _clip(arguments, RESULT_CHARS)
        return parsed
    return arguments


def _result(call: dict[str, Any]) -> dict[str, str]:
    if not call["answered"]:
        return {"status": "error", "summary": "no result"}
    raw = call["result"]
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
    failed = isinstance(parsed, dict) and (parsed.get("ok") is False or "error" in parsed)
    text = raw if isinstance(raw, str) else json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    return {"status": "error" if failed else "ok", "summary": _clip(text, RESULT_CHARS)}


def _steps(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for turn in _turns(raw_steps):
        reasoning = _clip(turn["reasoning"], REASONING_CHARS)
        if not turn["calls"]:
            steps.append({"reasoning": reasoning, "tool": None, "args": None, "result": None,
                          "decision": "continue"})
            continue
        for k, call in enumerate(turn["calls"]):
            result = _result(call)
            submitted = call["name"] == SUBMIT_TOOL and result["status"] == "ok"
            steps.append({"reasoning": reasoning if k == 0 else "", "tool": call["name"],
                          "args": _args(call["name"], call["arguments"]), "result": result,
                          "decision": "submitted" if submitted else "continue"})
    if steps and not any(s["decision"] == "submitted" for s in steps):
        steps[-1]["decision"] = "gave_up"
    return [{"i": i, **s} for i, s in enumerate(steps)]


# --- budget -------------------------------------------------------------------

def _shorten(step: dict[str, Any], cap: int) -> dict[str, Any]:
    out = {**step, "reasoning": _clip(step["reasoning"], min(cap, REASONING_CHARS))}
    if step["result"] is not None:
        out["result"] = {**step["result"], "summary": _clip(step["result"]["summary"], min(cap, RESULT_CHARS))}
    return out


def _fit(steps: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Keep the first KEEP_HEAD and last KEEP_TAIL steps whole; shorten the middle evenly."""
    if estimate_tokens(steps) <= budget or len(steps) <= KEEP_HEAD + KEEP_TAIL:
        return steps
    head, mid, tail = steps[:KEEP_HEAD], steps[KEEP_HEAD:-KEEP_TAIL], steps[-KEEP_TAIL:]

    def with_cap(cap: int) -> list[dict[str, Any]]:
        return head + [_shorten(s, cap) for s in mid] + tail

    lo, hi = 0, max(REASONING_CHARS, RESULT_CHARS)
    if estimate_tokens(with_cap(lo)) <= budget:
        while lo < hi:  # largest cap that fits
            cap = (lo + hi + 1) // 2
            if estimate_tokens(with_cap(cap)) <= budget:
                lo = cap
            else:
                hi = cap - 1
        return with_cap(lo)
    # Last resort: the middle collapses into one marker; the tail is never dropped.
    marker = {"i": mid[0]["i"], "reasoning": "", "tool": None, "args": None, "result": None,
              "decision": "continue", "elided_steps": len(mid)}
    return head + [marker] + tail


def compress(raw_steps: list[dict[str, Any]], budget: int | None = TOKEN_BUDGET) -> list[dict[str, Any]]:
    steps = _steps(raw_steps)
    return steps if budget is None else _fit(steps, budget)


# --- TruLens trace provider ---------------------------------------------------

_registered = False


def register_trulens_provider() -> bool:
    """Register a TruLens TraceProvider whose compress_trace calls compress(). False if unavailable."""
    global _registered
    if _registered:
        return True
    try:
        from trulens.core.utils.trace_provider import TraceProvider, register_trace_provider
    except ImportError:
        return False

    class GaitTraceProvider(TraceProvider):
        def can_handle(self, trace_data: dict[str, Any]) -> bool:
            return trace_data.get("schema") == SCHEMA and "raw_steps" in trace_data

        def extract_plan(self, trace_data: dict[str, Any]) -> Any:
            return None

        def extract_execution_flow(self, trace_data: dict[str, Any]) -> list[dict[str, Any]]:
            return compress(trace_data.get("raw_steps") or [])

        def extract_agent_interactions(self, trace_data: dict[str, Any]) -> list[dict[str, Any]]:
            return []

        def compress_trace(self, trace_data: dict[str, Any]) -> dict[str, Any]:
            return {"trace_id": trace_data.get("trace_id"), "schema": SCHEMA,
                    "compressed_steps": compress(trace_data.get("raw_steps") or []),
                    "compressor_sha256": compressor_sha256()}

    register_trace_provider(GaitTraceProvider())
    _registered = True
    return True
