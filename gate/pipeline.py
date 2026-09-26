"""The gate (SPEC §4, NOTES [A7]): four stages in card order, each an event.

    run_gate(parent, candidate, edit, round_id, db_name="ada") -> dict
        {verdict, reason, stages{gate.verifier, gate.gpa, gate.meta, gate.constraints},
         train{reliability, mean_distance_m, cost_per_run_usd, n},
         parent_train_reliability, frames_id,
         attempt_frames_id, train_mean_distance_m, parent_train_mean_distance_m}

attempt_frames_id, train_mean_distance_m and parent_train_mean_distance_m (NOTES [A11])
come from the verifier stage; all three stay None when the pre-check short-circuits.

A silent constraints pre-check runs first. If it fails, the first three cells are
emitted as info {"skipped": true}, gate.constraints emits start then fail, and the
edit is rejected with zero model calls. Otherwise all four stages run in card order,
even after an earlier failure: gate.verifier, gate.gpa, gate.meta and gate.constraints
(checked again) each emit start then pass or fail, and `stages` shows every result.
The verdict is accepted only when all four pass; `reason` is the first failure's reason
in card order. This is what counts edits the physics rejected that the judge passed.

Stages 2 and 3 live in gate.judge and gate.meta (A9), loaded through importlib. Each
exposes run(*, parent, candidate, edit, round_id, db_name, verifier[, judge]) ->
{"pass": bool, "reason": str | None, "payload": dict}; gate.meta also receives the
judge's result as `judge`. A missing module passes with payload {"stub": true}.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from core.contracts import CARD_STAGES, Edit, Version
from core.db import ADA
from core.events import emit
from gate import constraints, verifier_stage

VERIFIER, GPA, META, CONSTRAINTS = CARD_STAGES
STAGE_MODULES = {GPA: "gate.judge", META: "gate.meta"}
SKIPPED = {"skipped": True}


def _stage_module(name: str) -> ModuleType | None:
    """The stage's module, or None when it has not been written yet."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            return None
        raise


def _stub(**_: Any) -> dict[str, Any]:
    return {"pass": True, "reason": None, "payload": {"stub": True}}


def _verifier_payload(res: dict[str, Any], edit: Edit) -> dict[str, Any]:
    par = res["parent"] or {}
    delta = (round(res["train"]["reliability"] - par["reliability"], 4)
             if par and res["train"]["n"] else None)
    delta_m = (round(res["train"]["mean_distance_m"] - par["mean_distance_m"], 4)
               if par and res["train"]["n"] else None)
    return {
        "reason": res["reason"], "train": res["train"],
        "parent": {k: par.get(k) for k in ("reliability", "mean_distance_m", "n")} if par else None,
        "predicted_delta": edit.predicted_delta, "actual_delta": delta,
        "predicted_delta_m": edit.predicted_delta_m, "actual_delta_m": delta_m,
        "gaits": res.get("gaits"), "attempt_frames_id": res.get("attempt_frames_id"),
        "frames_id": res["frames_id"], "violation_frame": res["violation_frame"],
    }


def run_gate(parent: Version, candidate: Version, edit: Edit, round_id: str,
             db_name: str = ADA) -> dict[str, Any]:
    ids = {"round_id": round_id, "version_id": candidate.id, "edit_id": edit.id, "db_name": db_name}
    stages = {s: "skipped" for s in CARD_STAGES}
    out: dict[str, Any] = {
        "verdict": "rejected", "reason": None, "stages": stages,
        "train": {"reliability": 0.0, "mean_distance_m": 0.0, "cost_per_run_usd": 0.0, "n": 0},
        "parent_train_reliability": 0.0, "frames_id": None,
        "attempt_frames_id": None, "train_mean_distance_m": None,
        "parent_train_mean_distance_m": None,
    }

    def fail(stage: str, reason: str, payload: dict[str, Any] | None = None) -> None:
        stages[stage] = "fail"
        if out["reason"] is None:
            out["reason"] = reason
        emit(stage, "fail", {"reason": reason, **(payload or {})}, **ids)

    def passed(stage: str, payload: dict[str, Any] | None = None) -> None:
        stages[stage] = "pass"
        emit(stage, "pass", payload or {}, **ids)

    reason = constraints.check(candidate, db_name)
    if reason is not None:
        tasks = verifier_stage.gate_tasks(db_name)
        par = verifier_stage.parent_score(parent, tasks, db_name)
        out["parent_train_reliability"] = par["reliability"] if par else 0.0
        for stage in (VERIFIER, GPA, META):
            emit(stage, "info", SKIPPED, **ids)
        emit(CONSTRAINTS, "start", {}, **ids)
        fail(CONSTRAINTS, reason)
        return out

    # 1. verifier
    emit(VERIFIER, "start", {"tasks": list(verifier_stage.GATE_TASKS), "k": verifier_stage.K}, **ids)
    res = verifier_stage.run(parent, candidate, db_name)
    out["train"] = {k: res["train"][k] for k in ("reliability", "mean_distance_m", "cost_per_run_usd", "n")}
    out["parent_train_reliability"] = res["parent"]["reliability"] if res["parent"] else 0.0
    out["frames_id"] = res["frames_id"]
    for key in ("attempt_frames_id", "train_mean_distance_m", "parent_train_mean_distance_m"):
        out[key] = res.get(key)
    payload = _verifier_payload(res, edit)
    if res["pass"]:
        passed(VERIFIER, payload)
    else:
        fail(VERIFIER, res["reason"], payload)

    # 2. judge, 3. meta: run even when the verifier failed
    common = {"parent": parent, "candidate": candidate, "edit": edit, "round_id": round_id,
              "db_name": db_name, "verifier": res}
    judged: dict[str, Any] | None = None
    for stage in (GPA, META):
        module = _stage_module(STAGE_MODULES[stage])
        runner = module.run if module is not None else _stub
        emit(stage, "start", {}, **ids)
        extra = {"judge": judged} if stage == META else {}
        result = runner(**common, **extra)
        if stage == GPA:
            judged = result
        if result["pass"]:
            passed(stage, result.get("payload"))
        else:
            fail(stage, result["reason"] or f"{stage} failed", result.get("payload"))

    # 4. constraints, checked again at the end
    emit(CONSTRAINTS, "start", {}, **ids)
    reason = constraints.check(candidate, db_name)
    if reason is None:
        passed(CONSTRAINTS)
    else:
        fail(CONSTRAINTS, reason)

    if all(v == "pass" for v in stages.values()):
        out["verdict"] = "accepted"
    return out
