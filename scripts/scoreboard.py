"""The scoreboard (NOTES [B9]): one document summing what the rewrite loop cost and caught.

    uv run python -m scripts.scoreboard [--best <version_id>] [--round <round_id> ...]

- rewrite_cost_usd: payload.cost_usd on controller, gate.gpa (judge) and gate.meta events,
  plus the gate verifier's episode costs (train runs of the proposals' candidate versions).
- cost_per_gait: versions.metrics.cost_per_run_usd (mean episode cost) of v0, frontier, best.
- proposals_total: edits proposed by the loop (origin model or probe; ./verify rows excluded).
- overrated_attempts: proposals rejected with a torque or power reason ("rated torque" or
  "rated motors"; both wordings exist in today's data).
- physics_rejected: edits whose gate.verifier failed on a sanity violation, i.e. the fail
  event's payload.reason matches SANITY ("no strict improvement" and gait-count failures
  are not physics) [B11]; physics_rejected_judge_passed: of those, the ones whose
  gate.gpa passed.

--best defaults to the right_version of the latest swaps document. --round scopes events
and edits to those rounds; without it every round counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any

from core.contracts import CostPerGait, Scoreboard
from core.db import ADA, db

V0 = "v0"
FRONTIER = "frontier"
COST_STAGES = ("controller", "gate.gpa", "gate.meta")
OVERRATED = re.compile(r"rated torque|rated motors", re.IGNORECASE)
SANITY = re.compile(r"rated torque|rated motors|exploits the simulator|rose above|diverged", re.IGNORECASE)


def _now_ms() -> datetime:
    """UTC now at Mongo's millisecond precision, so the stored document equals the returned one."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def _scope(round_ids: list[str] | None) -> dict[str, Any]:
    return {"round_id": {"$in": list(round_ids)}} if round_ids is not None else {}


def _cost_per_gait(version_id: str | None, db_name: str) -> float | None:
    if not version_id:
        return None
    doc = db(db_name).versions.find_one({"_id": version_id}, {"metrics.cost_per_run_usd": 1})
    return ((doc or {}).get("metrics") or {}).get("cost_per_run_usd")


def latest_best(db_name: str = ADA) -> str | None:
    doc = db(db_name).swaps.find_one({}, {"right_version": 1}, sort=[("captured_at", -1)])
    return (doc or {}).get("right_version")


def _edit_ids_with(stage: str, status: str, scope: dict[str, Any], db_name: str) -> set[str]:
    return {e for e in db(db_name).events.distinct("edit_id", {**scope, "stage": stage, "status": status}) if e}


def _sanity_failed(scope: dict[str, Any], db_name: str) -> set[str]:
    """Edit ids with a gate.verifier fail event whose reason is a sanity violation."""
    events = db(db_name).events.find({**scope, "stage": "gate.verifier", "status": "fail"},
                                     {"edit_id": 1, "payload.reason": 1})
    return {ev["edit_id"] for ev in events
            if ev.get("edit_id") and SANITY.search(str((ev.get("payload") or {}).get("reason") or ""))}


def build_scoreboard(
    *, best: str | None = None, v0: str = V0, frontier: str = FRONTIER,
    round_ids: list[str] | None = None, db_name: str = ADA,
) -> Scoreboard:
    d = db(db_name)
    scope = _scope(round_ids)
    best = best or latest_best(db_name)

    proposals = list(d.edits.find({**scope, "origin": {"$ne": "cli"}},
                                  {"verdict": 1, "reason": 1, "to_version": 1}))
    overrated = sum(e.get("verdict") == "rejected" and bool(OVERRATED.search(e.get("reason") or ""))
                    for e in proposals)

    physics = {str(e["_id"]) for e in proposals} & _sanity_failed(scope, db_name)
    judge_passed = physics & _edit_ids_with("gate.gpa", "pass", scope, db_name)

    event_usd = sum(float((ev.get("payload") or {}).get("cost_usd") or 0.0)
                    for ev in d.events.find({**scope, "stage": {"$in": list(COST_STAGES)}}, {"payload": 1}))
    candidates = sorted({e["to_version"] for e in proposals if e.get("to_version")})
    rows = d.runs.aggregate([
        {"$match": {"split": "train", "version_id": {"$in": candidates}}},
        {"$group": {"_id": None, "usd": {"$sum": "$cost_usd"}}},
    ])
    gate_usd = float(next(rows, {}).get("usd") or 0.0)

    created_at = _now_ms()
    board = Scoreboard(
        _id=f"scoreboard-{created_at:%Y%m%dT%H%M%S%f}", created_at=created_at, best_version=best,
        rewrite_cost_usd=event_usd + gate_usd,
        cost_per_gait=CostPerGait(v0=_cost_per_gait(v0, db_name), frontier=_cost_per_gait(frontier, db_name),
                                  best=_cost_per_gait(best, db_name)),
        overrated_attempts=overrated, proposals_total=len(proposals),
        physics_rejected=len(physics), physics_rejected_judge_passed=len(judge_passed),
    )
    d.scoreboard.insert_one(board.model_dump(by_alias=True))
    return board


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.scoreboard")
    parser.add_argument("--best", default=None, help="best version (default: latest swap's right_version)")
    parser.add_argument("--round", action="append", dest="rounds", default=None, help="limit to a round")
    args = parser.parse_args(argv)
    board = build_scoreboard(best=args.best, round_ids=args.rounds)
    print(json.dumps(board.model_dump(by_alias=True, mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
