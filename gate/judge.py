"""gate.gpa (SPEC §4 stage 2): Agent GPA judge over the candidate's compressed traces.

    run(*, parent, candidate, edit, round_id, db_name, verifier) -> {"pass", "reason", "payload"}

The traces are the verifier stage's trace ids, each passed through compress (NOTES [A4]);
raw steps never reach the model. One cached_chat("judge", ...) call scores Goal, Plan
and Action from 1 to 5 and replies JSON. Any score of 2 or lower fails the stage, as does
an unreadable reply or a failed call (fail closed). The payload always carries the
scores, a short justification and the call's cost_usd.
"""

from __future__ import annotations

import json
import re
from typing import Any

from compress import provider as compress
from core import llm
from core.db import ADA, db

compress.register_trulens_provider()

LOW_SCORE = 2
SCORES = ("goal", "plan", "action")
JUSTIFICATION_CHARS = 240
REASON_CHARS = 160
UNREADABLE = "judge output unreadable"
NO_TRACE = "no compressed trace to judge"

SYSTEM = """You are an Agent GPA judge. You read compressed traces of an agent that writes \
gaits for Ada, a four-legged walking robot, one trace per terrain task, and score the agent \
from 1 (poor) to 5 (excellent) on three dimensions:
- Goal: the agent pursued walking forward on this terrain.
- Plan: the agent followed a coherent approach across its attempts.
- Action: its tool calls and gait parameters are consistent with its stated plan and the task.
The harness edit that produced this agent is given for context.
Reply with one JSON object and nothing else:
{"goal": <1-5>, "plan": <1-5>, "action": <1-5>, "justification": "<two sentences>"}"""

USER = "Harness edit:\n{edit}\n\nCompressed traces:\n{traces}"


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def compressed_traces(trace_ids: list[str], db_name: str = ADA) -> list[dict[str, Any]]:
    """[{trace_id, steps}] in trace_ids order; steps are compress() output, never raw."""
    ids = list(dict.fromkeys(trace_ids))
    if not ids:
        return []
    docs = {d["trace_id"]: d for d in db(db_name).traces.find(
        {"trace_id": {"$in": ids}}, {"trace_id": 1, "raw_steps": 1})}
    budget = max(1, compress.TOKEN_BUDGET // len(ids))
    return [{"trace_id": t, "steps": compress.compress(docs[t].get("raw_steps") or [], budget=budget)}
            for t in ids if t in docs]


def edit_summary(edit: Any) -> dict[str, Any]:
    return {"primitive": edit.primitive, "op": edit.op, "new": edit.new, "rationale": edit.rationale}


def parse_json(message: dict[str, Any]) -> dict[str, Any] | None:
    """The reply's JSON object, tolerating a ``` fence; None when unreadable."""
    content = message.get("content")
    if not isinstance(content, str):
        return None
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _scores(data: dict[str, Any] | None) -> dict[str, int] | None:
    if data is None:
        return None
    out = {}
    for key in SCORES:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            return None
        out[key] = value
    return out


def _result(ok: bool, reason: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    return {"pass": ok, "reason": reason, "payload": payload}


def run(*, edit: Any, db_name: str = ADA, verifier: dict[str, Any], **_: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"scores": None, "justification": None, "cost_usd": 0.0, "model_id": None}
    traces = compressed_traces(verifier.get("trace_ids") or [], db_name)
    if not traces:
        return _result(False, NO_TRACE, payload)

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(edit=dumps(edit_summary(edit)),
                                                        traces=dumps(traces))}]
    try:
        chat = llm.cached_chat("judge", messages)
    except llm.LLMError as e:
        return _result(False, clip(f"judge call failed: {e}", REASON_CHARS), payload)
    payload.update(cost_usd=chat.cost_usd, model_id=chat.model_id)

    data = parse_json(chat.message)
    scores = _scores(data)
    if scores is None:
        return _result(False, UNREADABLE, payload)
    justification = clip(data.get("justification") or "", JUSTIFICATION_CHARS)
    payload.update(scores=scores, justification=justification)

    low = [f"{k.capitalize()} {v} of 5" for k, v in scores.items() if v <= LOW_SCORE]
    if low:
        return _result(False, clip(f"judge scored {', '.join(low)}: {justification}", REASON_CHARS),
                       payload)
    return _result(True, None, payload)
