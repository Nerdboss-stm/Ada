"""gate.meta (SPEC §4 stage 3): meta-verifier over the judge's output.

    run(*, parent, candidate, edit, round_id, db_name, verifier, judge) -> {"pass", "reason", "payload"}

One cached_chat("judge", ...) call reads the judge's JSON (scores and justification) and
the same compressed traces, and answers only whether the scores are supported by the
trace. "no" vetoes with its reason; an unreadable reply, a failed call, or a judge
result without scores fails closed. The payload carries the call's cost_usd.
"""

from __future__ import annotations

from typing import Any

from core import llm
from core.db import ADA
from gate import judge as judge_stage

REASON_CHARS = 200
UNREADABLE = "meta output unreadable"
NO_SCORES = "no judge scores to check"

SYSTEM = """You check another judge's work. You are given that judge's JSON scores \
(Goal, Plan, Action, 1 to 5) with its justification, and the compressed traces it read. \
Answer only whether the scores are supported by the trace. Do not rescore.
Reply with one JSON object and nothing else:
{"supported": "yes" | "no", "reason": "<one sentence>"}"""

USER = "Judge output:\n{judge}\n\nCompressed traces:\n{traces}"


def _result(ok: bool, reason: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    return {"pass": ok, "reason": reason, "payload": payload}


def run(*, db_name: str = ADA, verifier: dict[str, Any], judge: dict[str, Any] | None,
        **_: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"supported": None, "reason": None, "cost_usd": 0.0, "model_id": None}
    judged = (judge or {}).get("payload") or {}
    if not judged.get("scores"):
        return _result(False, NO_SCORES, payload)

    traces = judge_stage.compressed_traces(verifier.get("trace_ids") or [], db_name)
    judge_json = judge_stage.dumps({"scores": judged["scores"], "justification": judged.get("justification")})
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(judge=judge_json,
                                                        traces=judge_stage.dumps(traces))}]
    try:
        chat = llm.cached_chat("judge", messages)
    except llm.LLMError as e:
        return _result(False, judge_stage.clip(f"meta call failed: {e}", REASON_CHARS), payload)
    payload.update(cost_usd=chat.cost_usd, model_id=chat.model_id)

    data = judge_stage.parse_json(chat.message)
    answer = str((data or {}).get("supported", "")).strip().lower()
    if answer not in ("yes", "no"):
        return _result(False, UNREADABLE, payload)
    why = judge_stage.clip((data or {}).get("reason") or "", REASON_CHARS)
    payload.update(supported=answer, reason=why)
    if answer == "no":
        return _result(False, judge_stage.clip(f"meta-verifier veto: {why}", REASON_CHARS), payload)
    return _result(True, None, payload)
