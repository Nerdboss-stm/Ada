"""gate.constraints (SPEC §4 stage 4): pure checks on the candidate harness and the guardrails.

    check(candidate, db_name) -> str | None     the first broken rule, or None

No model calls. The pipeline runs this silently before stage 1 and again as the
gate.constraints cell. Order: guardrail documents verify against their content hash,
the live sim/verifier.py and compress/ hashes match a guardrail document, tools are a
subset of the stored whitelist, model_per_step uses only the cheap roles (NOTES [A6]),
engine keys and values sit inside loop.edits.ENGINE_BOUNDS, and no rule or context_policy
entry states a literal gait value (loop.edits.states_gait_values, the controller's own check).
"""

from __future__ import annotations

import math

from core.contracts import Version
from core.db import ADA
from harness import guardrails
from loop.edits import CHEAP_ROLES, ENGINE_BOUNDS, GAIT_VALUE_REASON, canon, states_gait_values

CHEAP_TIER_REASON = "model outside the cheap tier"


def _harness_text(candidate: Version) -> list[str]:
    """Free text the supervisor can write: each rule, and each context_policy key with its value."""
    h = candidate.harness
    return [*h.rules, *(f"{k} {v if isinstance(v, str) else canon(v)}" for k, v in h.context_policy.items())]


def _live_hashes() -> tuple[str, str | None]:
    verifier = guardrails.file_sha256(guardrails.VERIFIER_PATH)
    compress_dir = guardrails.COMPRESS_DIR
    compressor = guardrails.dir_sha256(compress_dir) if compress_dir.is_dir() else None
    return verifier, compressor


def check(candidate: Version, db_name: str = ADA) -> str | None:
    try:
        docs = guardrails.load_guardrails(db_name)
    except guardrails.GuardrailError as e:
        return f"guardrails failed to load: {e}"

    verifier_sha, compressor_sha = _live_hashes()
    matching = [d for d in docs if d.content.verifier_sha == verifier_sha]
    if not matching:
        return "sim/verifier.py changed: its sha256 matches no guardrail document"
    matching = [d for d in matching if d.content.compressor_sha == compressor_sha]
    if not matching:
        return "compress/ changed: its sha256 matches no guardrail document"

    harness = candidate.harness
    whitelist = set(matching[0].content.tool_whitelist)
    outside = [t for t in harness.tools if t not in whitelist]
    if outside:
        return f"tools outside the whitelist: {', '.join(outside)}"

    if any(role not in CHEAP_ROLES for role in harness.model_per_step.values()):
        return CHEAP_TIER_REASON

    for key, value in harness.engine.items():
        if key not in ENGINE_BOUNDS:
            return f"engine.{key} is not an allowed engine setting"
        lo, hi = ENGINE_BOUNDS[key]
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not number or not math.isfinite(value) or not lo <= value <= hi:
            return f"engine.{key} = {value!r} outside [{lo}, {hi}]"

    if any(states_gait_values(text) for text in _harness_text(candidate)):
        return GAIT_VALUE_REASON
    return None
