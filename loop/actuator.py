"""Actuator (SPEC §5): one round of child version -> gate -> `versions`, `edits`, `events`.

    run_round(head_version_id, db_name="ada", budget_usd=3.0, budget_s=600) -> dict

A round is a `rounds` document with open_lock: true under a unique partial index, so a second
open round is refused by the database; closing $unsets open_lock (NOTES [B0]: never null).
The head is sensed, the controller proposes, and each proposal in order is re-applied to the
current head (stale ones are skipped), given the next id v{N}, written to `edits` with verdict
null, sent to gate.pipeline.run_gate (NOTES [A7]), then the child `versions` document is
written and the edit gets its verdict (actual_delta null when the gate rolled nothing,
train.n 0). An accepted child becomes the head and is scored on
holdout (k 3) with its showcase run recorded. No new proposal starts once the round's spend
or time has reached its budget; the reason lands in the round document.
"""

from __future__ import annotations

import importlib
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

from core.contracts import Edit, Metrics, Round, Version
from core.db import ADA, db
from core.events import emit
from harness import run as harness_run
from loop import controller, sensor
from loop.edits import EditError, EditProposal, apply_edit

STAGE = "actuator"
HOLDOUT_K = 3
SHOWCASE_K = 1
REASON_CHARS = 300
LOCK_INDEX = "one_open_round"
VERDICTS = ("accepted", "rejected")
_VERSION_ID = re.compile(r"^v(\d+)$")

_clock = time.monotonic  # tests patch this to move time


class RoundOpenError(RuntimeError):
    """Another round still holds open_lock; nothing was written."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clip(text: str | None) -> str | None:
    return None if text is None else text[:REASON_CHARS]


def new_round_id() -> str:
    return f"r-{_now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


# --- rounds -----------------------------------------------------------------

def ensure_lock_index(db_name: str = ADA) -> None:
    db(db_name).rounds.create_index(
        "open_lock", unique=True, name=LOCK_INDEX,
        partialFilterExpression={"open_lock": {"$exists": True}},
    )


def open_round(budget_usd: float, budget_s: float, db_name: str = ADA) -> Round:
    ensure_lock_index(db_name)
    rnd = Round(_id=new_round_id(), status="open", opened_at=_now(),
                budget_usd=budget_usd, budget_s=budget_s, open_lock=True)
    try:
        db(db_name).rounds.insert_one(rnd.model_dump(by_alias=True, exclude_none=True))
    except DuplicateKeyError as e:
        held = db(db_name).rounds.find_one({"open_lock": {"$exists": True}}, {"_id": 1})
        raise RoundOpenError(f"round {held['_id'] if held else '?'} is still open") from e
    return rnd


def close_round(round_id: str, stop_reason: str | None, spent_usd: float, db_name: str = ADA) -> None:
    db(db_name).rounds.update_one({"_id": round_id}, {
        "$set": {"status": "closed", "closed_at": _now(), "stop_reason": stop_reason,
                 "spent_usd": round(spent_usd, 6)},
        "$unset": {"open_lock": ""},
    })


# --- reads ------------------------------------------------------------------

def load_version(version_id: str, db_name: str = ADA) -> Version:
    doc = db(db_name).versions.find_one({"_id": version_id})
    if doc is None:
        raise LookupError(f"no versions document {version_id!r}")
    return Version.model_validate(doc)


def next_version_id(db_name: str = ADA) -> str:
    ids = db(db_name).versions.find({"_id": {"$regex": r"^v[0-9]+$"}}, {"_id": 1})
    numbers = [int(m.group(1)) for d in ids if (m := _VERSION_ID.match(str(d["_id"])))]
    return f"v{max(numbers, default=0) + 1}"


def recent_edits(db_name: str = ADA) -> list[dict[str, Any]]:
    docs = db(db_name).edits.find({}).sort("created_at", -1).limit(controller.N_RECENT)
    return list(docs)[::-1]


def _controller_cost(round_id: str, db_name: str) -> float:
    ev = db(db_name).events.find_one({"stage": "controller", "round_id": round_id}, sort=[("ts", -1)])
    return float(((ev or {}).get("payload") or {}).get("cost_usd") or 0.0)


def _violation_frame(frames_id: str | None, db_name: str) -> int | None:
    if not frames_id:
        return None
    doc = db(db_name).frames.find_one({"_id": frames_id}, {"violation_frame": 1})
    return (doc or {}).get("violation_frame")


def _scoring_cost(version_id: str, db_name: str) -> float:
    rows = db(db_name).runs.aggregate([
        {"$match": {"version_id": version_id, "split": {"$in": ["holdout", "showcase"]}}},
        {"$group": {"_id": None, "usd": {"$sum": "$cost_usd"}}},
    ])
    return float(next(rows, {}).get("usd") or 0.0)


# --- one proposal -----------------------------------------------------------

def _run_gate(**kwargs: Any) -> dict[str, Any]:
    return importlib.import_module("gate.pipeline").run_gate(**kwargs)


def _score_accepted(child: Version, db_name: str) -> float:
    """Holdout k 3 and the showcase run through harness.run; returns their cost."""
    harness_run.run_split(child.id, "holdout", HOLDOUT_K, harness=child.harness, db_name=db_name)
    harness_run.run_split(child.id, "showcase", SHOWCASE_K, harness=child.harness, db_name=db_name)
    n = db(db_name).runs.count_documents({"version_id": child.id, "split": "holdout"})
    if n:  # NOTES [B3]: n is the holdout run count once holdout is written
        db(db_name).versions.update_one({"_id": child.id}, {"$set": {"metrics.n": n}})
    return _scoring_cost(child.id, db_name)


def _stale(p: EditProposal, head: Version, index: int, round_id: str, err: EditError,
           db_name: str) -> dict[str, Any]:
    decision = {"index": index, "decision": "stale", "edit_id": None, "version_id": None,
                "reason": _clip(str(err)), "actual_delta": None}
    emit(STAGE, "info", {"event": "decision", "decision": "stale", "index": index,
                         "primitive": p.primitive, "op": p.op, "path": p.path,
                         "from_version": head.id, "reason": decision["reason"]},
         round_id=round_id, version_id=head.id, db_name=db_name)
    return decision


def actuate(p: EditProposal, head: Version, index: int, round_id: str, db_name: str = ADA,
            ) -> tuple[dict[str, Any], Version | None, float]:
    """Gate one proposal against `head`. Returns (decision, new head or None, spend)."""
    try:
        harness = apply_edit(head.harness, p)
    except EditError as err:
        return _stale(p, head, index, round_id, err, db_name), None, 0.0

    child_id = next_version_id(db_name)
    edit = Edit(
        _id=f"{round_id}.e{index}", round_id=round_id, from_version=head.id, to_version=child_id,
        origin="model", primitive=p.primitive, op=p.op, path=p.path, old=p.old, new=p.new,
        rationale=p.rationale, evidence_trace_ids=p.evidence_trace_ids,
        predicted_delta=p.predicted_delta, created_at=_now(),
    )
    db(db_name).edits.insert_one(edit.model_dump(by_alias=True))
    candidate = Version(_id=child_id, parent=head.id, status="candidate", harness=harness,
                        created_at=_now())

    result = _run_gate(parent=head, candidate=candidate, edit=edit, round_id=round_id, db_name=db_name)
    verdict = result["verdict"]
    if verdict not in VERDICTS:
        raise ValueError(f"gate returned verdict {verdict!r}; expected one of {VERDICTS}")
    train = result["train"]
    metrics = Metrics(train_reliability=train["reliability"], holdout_reliability_80=0.0,
                      mean_distance_m=train["mean_distance_m"],
                      cost_per_run_usd=train["cost_per_run_usd"], n=train["n"])
    child = Version.model_validate({**candidate.model_dump(by_alias=True), "status": verdict,
                                    "metrics": metrics.model_dump()})
    db(db_name).versions.insert_one(child.model_dump(by_alias=True))

    # n == 0: rejected at the constraints pre-check, nothing rolled, so there is no delta
    actual = (round(train["reliability"] - result["parent_train_reliability"], 4)
              if train["n"] else None)
    frames_id = result.get("frames_id")
    db(db_name).edits.update_one({"_id": edit.id}, {"$set": {
        "verdict": verdict, "reason": result.get("reason"), "actual_delta": actual,
        "frames_id": frames_id, "violation_frame": _violation_frame(frames_id, db_name),
    }})
    accepted = verdict == "accepted"
    emit(STAGE, "pass" if accepted else "fail", {
        "event": "decision", "decision": verdict, "index": index, "from_version": head.id,
        "to_version": child_id, "reason": _clip(result.get("reason")), "actual_delta": actual,
        "predicted_delta": p.predicted_delta, "frames_id": frames_id, "stages": result.get("stages"),
    }, round_id=round_id, version_id=child_id, edit_id=edit.id, db_name=db_name)

    spend = float(train["cost_per_run_usd"]) * int(train["n"])  # per episode x runs: errs high
    new_head = None
    if accepted:
        spend += _score_accepted(child, db_name)
        new_head = load_version(child_id, db_name)
    decision = {"index": index, "decision": verdict, "edit_id": edit.id, "version_id": child_id,
                "reason": result.get("reason"), "actual_delta": actual}
    return decision, new_head, spend


# --- round ------------------------------------------------------------------

def _budget_stop(spent: float, elapsed: float, budget_usd: float, budget_s: float) -> str | None:
    if spent >= budget_usd:
        return "budget_usd"
    if elapsed >= budget_s:
        return "budget_s"
    return None


def _run(state: dict[str, Any], started: float, budget_usd: float, budget_s: float,
         db_name: str) -> None:
    rid = state["round_id"]
    head = load_version(state["head"], db_name)
    reading = sensor.sense(head.id, "train", db_name=db_name, round_id=rid)
    summary = {k: v for k, v in reading.items() if k != "per_task"}
    try:
        proposals = controller.propose(head, summary, recent_edits(db_name), round_id=rid,
                                       db_name=db_name)
    except controller.ControllerBudgetError:
        state["stop_reason"] = "controller over budget"
        return
    finally:
        state["spent_usd"] += _controller_cost(rid, db_name)

    for i, p in enumerate(proposals):
        stop = _budget_stop(state["spent_usd"], _clock() - started, budget_usd, budget_s)
        if stop:
            state["stop_reason"], state["skipped"] = stop, len(proposals) - i
            return
        decision, new_head, spend = actuate(p, head, i, rid, db_name)
        state["spent_usd"] += spend
        state["decisions"].append(decision)
        if decision["decision"] == "stale":
            state["stale"] += 1
        else:
            state[decision["decision"]].append(decision["version_id"])
        if new_head is not None:
            head = new_head
            state["head"] = head.id
    state["stop_reason"] = "proposals exhausted" if proposals else "no proposals"


def run_round(head_version_id: str, db_name: str = ADA, budget_usd: float = 3.0,
              budget_s: float = 600) -> dict[str, Any]:
    """One round from `head_version_id`; raises RoundOpenError if another round is open."""
    rnd = open_round(budget_usd, budget_s, db_name)
    started = _clock()
    state: dict[str, Any] = {
        "round_id": rnd.id, "start_head": head_version_id, "head": head_version_id,
        "decisions": [], "accepted": [], "rejected": [], "stale": 0, "skipped": 0,
        "stop_reason": None, "spent_usd": 0.0, "elapsed_s": 0.0,
    }
    emit(STAGE, "start", {"event": "round_open", "head": head_version_id,
                          "budget_usd": budget_usd, "budget_s": budget_s},
         round_id=rnd.id, version_id=head_version_id, db_name=db_name)
    try:
        _run(state, started, budget_usd, budget_s, db_name)
    except BaseException as e:
        state["stop_reason"] = _clip(f"error: {type(e).__name__}: {e}")
        raise
    finally:
        state["spent_usd"] = round(state["spent_usd"], 6)
        state["elapsed_s"] = round(_clock() - started, 3)
        close_round(rnd.id, state["stop_reason"], state["spent_usd"], db_name)
        emit(STAGE, "info", {
            "event": "round_close", "start_head": head_version_id, "head": state["head"],
            "stop_reason": state["stop_reason"], "spent_usd": state["spent_usd"],
            "elapsed_s": state["elapsed_s"], "accepted": state["accepted"],
            "rejected": state["rejected"], "stale": state["stale"], "skipped": state["skipped"],
        }, round_id=rnd.id, version_id=state["head"], db_name=db_name)
    return state
