"""Controller (SPEC §5): one supervisor-model call proposes at most 6 harness edits.

    propose(head_version, sensor_summary, recent_edits, round_id=None) -> list[EditProposal]

The prompt carries the head harness (never its metrics: holdout stays unseen), the tool
whitelist, the sensor summary with its example trace ids, the last 10 edits with verdict and
reason, and the objective. Dynamic content (sensor readings, gate reasons) goes in verbatim: it is the gate's real feedback. Only the static
template we wrote is checked against BANNED. The cost is estimated before the call and the call
is refused above BUDGET_USD. The reply is JSON; when the strict parse fails (a reply cut off at
max_tokens), every complete edit object is salvaged; invalid (including a missing predicted_delta_m
or literal gait values), stale and duplicate proposals are dropped, the rest sorted smallest
change first and cut to MAX_EDITS. One controller event.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import ValidationError

from core.contracts import Edit, Harness, Version
from core.db import ADA
from core.events import emit
from core.llm import cached_chat, models
from loop.edits import (
    CHEAP_ROLES, ENGINE_BOUNDS, MAX_RATIONALE_WORDS, TOOL_WHITELIST, EditError, EditProposal,
    apply_edit,
)

ROLE = "controller"
OBJECTIVE = "0.7 × train reliability + 0.3 × normalized mean distance"
MAX_EDITS = 6
N_RECENT = 10
BUDGET_USD = 3.0
MAX_TOKENS = 16000  # the reasoning model thinks before it answers; 4096 truncated the JSON
CHARS_PER_TOKEN = 3  # deliberately low, so the estimate errs high
BANNED = ("power", "glitch", "exploit", "cheat")
EDIT_FIELDS = ("_id", "primitive", "op", "path", "old", "new", "predicted_delta", "predicted_delta_m",
               "actual_delta", "verdict", "reason")

SYSTEM = (
    "You tune the harness of a small, cheap agent that writes gaits for Ada, a four-legged robot. "
    "You change the harness only; the agent writes the gaits. Reply with JSON only."
)

TEMPLATE = """Objective, to maximize: {objective}

## Head version {version_id}: its harness
{harness}

## Harness primitives and how to edit them
- rules: list of strings added to the agent's system prompt. add `new`; remove `old`; set replaces `old` with `new`.
- tools: list of tool names the agent may call, from the whitelist below only. add `new`; remove `old`; set replaces `old` with `new`.
- model_per_step: object; `path` is the step name ("agent"); `new` must be one of {roles}. add or set only.
- engine: object; set only; `path` is {engine}; temperature in [0, 1], max_attempts an integer in [1, 6].
For every edit, `old` must equal the head's current value (null when adding). Each edit is tested alone against the head version.
Rule text must not state literal gait values (a gait field name next to a number); the agent chooses the numbers. Such edits are dropped.

## Tool whitelist
{tools}

## Sensor summary: the head version on the train subset
{sensor}

## Last {n_recent} edits, oldest first, with verdict and reason
{edits}

## Reply
A JSON object {{"edits": [...]}} with at most {max_edits} edits, smallest change first. Each edit:
{{"primitive": "rules|tools|model_per_step|engine", "op": "add|remove|set", "path": "", "old": null, "new": null,
 "predicted_delta": <expected change in train reliability, -1 to 1>,
 "predicted_delta_m": <expected change in train mean distance in meters, a number; required>, "rationale": "<one sentence, at most {max_words} words>",
 "evidence_trace_ids": ["<trace ids from the sensor summary that motivate the edit>"]}}"""


class ControllerBudgetError(RuntimeError):
    """The estimated cost of the controller call exceeds BUDGET_USD; nothing was called."""


def _tool_lines() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_WHITELIST.items())


def _assert_clean(*texts: str) -> None:
    """The static prompt we wrote must never name the four banned words."""
    for text in texts:
        found = [w for w in BANNED if w in text.lower()]
        if found:
            raise AssertionError(f"controller template mentions {found}")


def _static_text() -> tuple[str, ...]:
    return (SYSTEM, TEMPLATE, OBJECTIVE, _tool_lines())


_assert_clean(*_static_text())


# --- prompt -----------------------------------------------------------------

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _version(head: Version | dict[str, Any]) -> Version:
    return head if isinstance(head, Version) else Version.model_validate(head)


def _edit_row(edit: Edit | dict[str, Any]) -> dict[str, Any]:
    doc = edit.model_dump(by_alias=True) if isinstance(edit, Edit) else edit
    return {k: doc.get(k) for k in EDIT_FIELDS}


def render_messages(
    head: Version, sensor_summary: dict[str, Any], recent_edits: list[Edit | dict[str, Any]],
) -> list[dict[str, str]]:
    recent = recent_edits[-N_RECENT:]
    edits = "\n".join(_json(_edit_row(e)) for e in recent) or "(none yet)"
    user = TEMPLATE.format(
        objective=OBJECTIVE,
        version_id=head.id,
        harness=_json(head.harness.model_dump()),
        roles=" or ".join(CHEAP_ROLES),
        engine=" or ".join(ENGINE_BOUNDS),
        tools=_tool_lines(),
        sensor=_json(sensor_summary),
        n_recent=N_RECENT,
        edits=edits,
        max_edits=MAX_EDITS,
        max_words=MAX_RATIONALE_WORDS,
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def estimate_usd(messages: list[dict[str, str]], max_tokens: int = MAX_TOKENS) -> float:
    """Upper-bound cost: input tokens from characters, output at max_tokens, the pricier of primary/alt."""
    spec = models()[ROLE]
    specs = [spec] + ([spec["alt"]] if spec.get("alt") else [])
    tokens_in = math.ceil(len(_json(messages)) / CHARS_PER_TOKEN)
    usd_in = max(s["usd_per_mtok_in"] for s in specs)
    usd_out = max(s["usd_per_mtok_out"] for s in specs)
    return round((tokens_in * usd_in + max_tokens * usd_out) / 1e6, 6)


# --- reply ------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_reply(text: str) -> list[Any] | None:
    """The list of raw edits, or None when the reply holds no usable JSON."""
    text = (text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = data.get("edits")
    return data if isinstance(data, list) else None


_OPEN_FENCE = re.compile(r"^```(?:json)?\s*")


def salvage_reply(text: str) -> list[Any] | None:
    """Every complete element of the edits array, read one by one, stopping at the first that is
    cut off or malformed. None when the reply has no array at all."""
    text = _OPEN_FENCE.sub("", (text or "").strip())
    key = text.find('"edits"')
    start = text.find("[", key if key >= 0 else 0)
    if start < 0:
        return None
    decoder, items, i = json.JSONDecoder(), [], start + 1
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] == "]":
            return items
        try:
            item, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return items
        items.append(item)


def truncated(result: Any) -> bool:
    """The reply hit max_tokens. ChatResult has no finish_reason yet, so the completion tokens tell."""
    if getattr(result, "finish_reason", None) == "length":
        return True
    return (result.usage or {}).get("completion_tokens", 0) >= MAX_TOKENS


def select(items: list[Any], harness: Harness) -> list[EditProposal]:
    """Valid proposals that apply to `harness`, deduplicated, smallest change first, at most MAX_EDITS."""
    kept: dict[tuple[str, ...], EditProposal] = {}
    for item in items:
        try:
            proposal = EditProposal.model_validate(item)
            apply_edit(harness, proposal)
        except (ValidationError, EditError):
            continue
        kept.setdefault(proposal.key(), proposal)
    ordered = sorted(kept.values(), key=EditProposal.change_size)  # stable: ties keep reply order
    return ordered[:MAX_EDITS]


# --- public -----------------------------------------------------------------

def propose(
    head_version: Version | dict[str, Any],
    sensor_summary: dict[str, Any],
    recent_edits: list[Edit | dict[str, Any]],
    round_id: str | None = None,
    *,
    db_name: str = ADA,
) -> list[EditProposal]:
    head = _version(head_version)
    messages = render_messages(head, sensor_summary, recent_edits)
    estimate = estimate_usd(messages)
    if estimate > BUDGET_USD:
        emit(ROLE, "fail", {"count": 0, "reason": "over budget", "estimate_usd": estimate,
                            "budget_usd": BUDGET_USD},
             round_id=round_id, version_id=head.id, db_name=db_name)
        raise ControllerBudgetError(f"controller call estimated at ${estimate:.4f} > ${BUDGET_USD}")

    result = cached_chat(ROLE, messages, max_tokens=MAX_TOKENS)
    content = result.message.get("content") or ""
    items = parse_reply(content)
    if items is None:
        items = salvage_reply(content)
    proposals = select(items or [], head.harness)
    emit(ROLE, "info" if items is not None else "fail",
         {"count": len(proposals), "received": len(items or []), "estimate_usd": estimate,
          "cost_usd": result.cost_usd, "cached": result.cached, "truncated": truncated(result)},
         round_id=round_id, version_id=head.id, db_name=db_name)
    return proposals
