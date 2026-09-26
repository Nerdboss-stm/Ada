"""Edit proposals (SPEC §5) and a pure apply_edit over the five harness primitives (SPEC §3).

    EditProposal   one edit: primitive, op, path, old, new, predicted_delta, predicted_delta_m,
                   rationale, evidence
    apply_edit(harness, proposal) -> Harness   a new harness; the input is never mutated
    states_gait_values(text) -> bool           a gait field name within 12 characters of a number

Semantics per primitive:
    rules, tools             lists; path is always "". add appends `new`, remove drops `old`,
                             set replaces the item equal to `old` with `new`.
    context_policy,          dicts; path is the key. add needs the key absent, remove needs
    model_per_step, engine   `old` to equal the current value, set upserts after checking that
                             `old` equals the current value (None when the key is absent).
Static checks (no harness needed) live on the model; checks against the current harness raise
EditError from apply_edit.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.contracts import Harness

# CONTRACTS §6, exactly 6, with the one-line descriptions the controller sees.
TOOL_WHITELIST: dict[str, str] = {
    "read_task": "read the task: slope, friction, target distance, gait schema and neutral joint offsets",
    "submit_gait": "submit a gait for scoring; each call counts as one attempt",
    "preview_run": "try a gait for 3 s on a practice seed: distance, fell, max torso tilt, contact rhythm",
    "get_contact_log": "each leg's ground contact during the last preview, in equal time bins",
    "list_my_attempts": "the gaits previewed earlier in this episode, with their preview results",
    "lookup_skill": "gait summaries stored from earlier accepted versions on similar tasks",
}
CHEAP_ROLES = ("agent_v0", "agent_v0.alt")  # NOTES [A6] RULE
ENGINE_BOUNDS: dict[str, tuple[int, int]] = {"temperature": (0, 1), "max_attempts": (1, 6)}
LIST_PRIMITIVES = ("rules", "tools")
MAX_RATIONALE_WORDS = 25

Primitive = Literal["rules", "context_policy", "tools", "model_per_step", "engine"]
Op = Literal["add", "remove", "set"]

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

# Gait fields (sim.gait.Gait) the supervisor may never pin to a number; the agent picks the numbers.
GAIT_VALUE_REASON = "states literal gait values"
GAIT_VALUE_GAP = 12  # characters allowed between a field name and a number
_GAIT_NAME = re.compile(
    r"(?<![a-z])(?:frequenc(?:y|ies)|power|kp|kd|amplitudes?|offsets?|phases?|tilt_gain|hz)(?![a-z])",
    re.I,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")


class EditError(ValueError):
    """The proposal does not fit the harness it is applied to."""


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def states_gait_values(text: str) -> bool:
    """True when a gait field name and a number sit within GAIT_VALUE_GAP characters, either order."""
    numbers = [m.span() for m in _NUMBER.finditer(text)]
    for name in _GAIT_NAME.finditer(text):
        a, b = name.span()
        for c, d in numbers:
            gap = c - b if c >= b else a - d
            if gap <= GAIT_VALUE_GAP:
                return True
    return False


class EditProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: Primitive
    op: Op
    path: str = ""
    old: Any = None
    new: Any = None
    predicted_delta: float  # expected change in train reliability
    predicted_delta_m: float  # expected change in train mean distance, meters
    rationale: str
    evidence_trace_ids: list[str] = Field(default_factory=list)

    @field_validator("predicted_delta_m", mode="before")
    @classmethod
    def _meters_is_number(cls, v: Any) -> Any:
        if not _is_number(v):
            raise ValueError(f"predicted_delta_m must be a finite number, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check(self) -> EditProposal:
        if not math.isfinite(self.predicted_delta) or not -1 <= self.predicted_delta <= 1:
            raise ValueError(f"predicted_delta {self.predicted_delta} outside [-1, 1]")
        self._check_rationale()
        if states_gait_values(self.added_text()):
            raise ValueError(GAIT_VALUE_REASON)
        if self.primitive in LIST_PRIMITIVES:
            self.path = ""
            self._check_list()
        else:
            if not self.path:
                raise ValueError(f"{self.primitive} needs a path (the key)")
            getattr(self, f"_check_{self.primitive}")()
        return self

    def _check_rationale(self) -> None:
        text = self.rationale.strip()
        words = len(text.split())
        if not 1 <= words <= MAX_RATIONALE_WORDS:
            raise ValueError(f"rationale has {words} words; 1 to {MAX_RATIONALE_WORDS} allowed")
        if "\n" in text or len(_SENTENCE_END.findall(text)) > 1:
            raise ValueError("rationale must be one sentence")

    def _check_list(self) -> None:
        needs_old = self.op in ("remove", "set")
        needs_new = self.op in ("add", "set")
        for name, needed in (("old", needs_old), ("new", needs_new)):
            value = getattr(self, name)
            if needed and not (isinstance(value, str) and value.strip()):
                raise ValueError(f"{self.primitive} {self.op} needs a non-empty string {name}")
            if not needed and value is not None:
                raise ValueError(f"{self.primitive} {self.op} takes no {name}")
            if needed and self.primitive == "tools" and value not in TOOL_WHITELIST:
                raise ValueError(f"tool {value!r} is not in the whitelist")

    def _check_context_policy(self) -> None:
        if self.op == "add" and self.old is not None:
            raise ValueError("add takes no old")
        if self.op == "remove" and self.new is not None:
            raise ValueError("remove takes no new")

    def _check_model_per_step(self) -> None:
        if self.op == "remove":
            raise ValueError("model_per_step edits may only add or set")
        if self.op == "add" and self.old is not None:
            raise ValueError("add takes no old")
        if self.new not in CHEAP_ROLES:
            raise ValueError(f"model_per_step may only use {CHEAP_ROLES}, got {self.new!r}")

    def _check_engine(self) -> None:
        if self.op != "set":
            raise ValueError("engine edits may only set")
        if self.path not in ENGINE_BOUNDS:
            raise ValueError(f"engine may only set {sorted(ENGINE_BOUNDS)}, got {self.path!r}")
        lo, hi = ENGINE_BOUNDS[self.path]
        if not _is_number(self.new) or not lo <= self.new <= hi:
            raise ValueError(f"engine.{self.path} must be a number in [{lo}, {hi}], got {self.new!r}")
        if self.path == "max_attempts" and not isinstance(self.new, int):
            raise ValueError(f"engine.max_attempts must be an integer, got {self.new!r}")

    def added_text(self) -> str:
        """The text this edit adds or changes in the harness: `new`, with the key for dict primitives."""
        if self.op == "remove" or self.new is None:
            return ""
        new = self.new if isinstance(self.new, str) else canon(self.new)
        return new if self.primitive in LIST_PRIMITIVES else f"{self.path} {new}"

    def change_size(self) -> int:
        """Characters of old plus new as canonical JSON; the controller sorts smallest first."""
        return sum(len(canon(v)) for v in (self.old, self.new) if v is not None)

    def key(self) -> tuple[str, str, str, str, str]:
        """Two proposals with the same key are duplicates, whatever their rationale."""
        return (self.primitive, self.op, self.path, canon(self.old), canon(self.new))


def _apply_list(items: list[Any], p: EditProposal) -> None:
    if p.op == "add":
        if p.new in items:
            raise EditError(f"{p.primitive} already contains {p.new!r}")
        items.append(copy.deepcopy(p.new))
        return
    if p.old not in items:
        raise EditError(f"{p.primitive} has no {p.old!r}")
    if p.op == "remove":
        items.remove(p.old)
        return
    if p.new == p.old:
        raise EditError(f"{p.primitive} set changes nothing")
    if p.new in items:
        raise EditError(f"{p.primitive} already contains {p.new!r}")
    items[items.index(p.old)] = copy.deepcopy(p.new)


def _apply_dict(d: dict[str, Any], p: EditProposal) -> None:
    present = p.path in d
    current = d.get(p.path)
    if p.op == "add":
        if present:
            raise EditError(f"{p.primitive}.{p.path} already exists")
        d[p.path] = copy.deepcopy(p.new)
        return
    if p.op == "remove" and not present:
        raise EditError(f"{p.primitive}.{p.path} does not exist")
    if canon(current) != canon(p.old):
        raise EditError(f"{p.primitive}.{p.path} is {current!r}, not {p.old!r}")
    if p.op == "remove":
        del d[p.path]
        return
    if present and canon(current) == canon(p.new):
        raise EditError(f"{p.primitive}.{p.path} set changes nothing")
    d[p.path] = copy.deepcopy(p.new)


def apply_edit(harness: Harness, proposal: EditProposal) -> Harness:
    """Return a new harness with the edit applied; raises EditError if it does not fit."""
    data = copy.deepcopy(harness.model_dump())
    target = data[proposal.primitive]
    if proposal.primitive in LIST_PRIMITIVES:
        _apply_list(target, proposal)
    else:
        _apply_dict(target, proposal)
    return Harness.model_validate(data)
