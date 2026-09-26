"""Sensor (SPEC §5): verifier results for one version and split, failure codes by `$group`.

Each failed run gets exactly one code, first match wins:
    sanity   sanity check failed (carries the violation text)
    fell     the robot fell
    backward distance_m < 0
    short    0 <= distance_m < the task's target_m
"""

from __future__ import annotations

from typing import Any

from core.db import ADA, db
from core.events import emit
from sim.tasks import TARGET_M

CODES = ("sanity", "fell", "backward", "short")
N_EXAMPLES = 3
N_WORST = 3
N_VIOLATIONS = 3
VIOLATION_CHARS = 80


def _rates(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "n": {"$sum": 1},
        "success_rate": {"$avg": {"$cond": ["$success", 1, 0]}},
        "mean_distance_m": {"$avg": "$distance_m"},
        **(extra or {}),
    }


def _pipeline(version_id: str, split: str) -> list[dict[str, Any]]:
    target = {"$ifNull": [{"$first": "$task.target_m"}, TARGET_M]}
    code = {
        "$switch": {
            "branches": [
                {"case": {"$not": ["$sanity.pass"]}, "then": "sanity"},
                {"case": "$fell", "then": "fell"},
                {"case": {"$lt": ["$distance_m", 0]}, "then": "backward"},
                {"case": {"$lt": ["$distance_m", target]}, "then": "short"},
            ],
            "default": None,  # contradicts sim.verifier's success rule; _code_row raises
        }
    }
    return [
        {"$match": {"version_id": version_id, "split": split}},
        {"$lookup": {"from": "tasks", "localField": "task_id", "foreignField": "_id", "as": "task"}},
        {"$addFields": {
            "code": {"$cond": ["$success", None, code]},
            # runs with a trace_id sort first so examples cite real evidence
            "no_trace": {"$eq": [{"$ifNull": ["$trace_id", None]}, None]},
        }},
        {"$project": {"task": 0, "gait": 0}},
        {"$sort": {"no_trace": 1, "task_id": 1, "seed": 1, "_id": 1}},
        {"$facet": {
            "codes": [
                {"$match": {"success": False}},
                {"$group": {
                    "_id": "$code",
                    "count": {"$sum": 1},
                    "examples": {"$firstN": {"input": "$trace_id", "n": N_EXAMPLES}},
                    "violations": {"$addToSet": "$sanity.violation"},
                }},
            ],
            "per_task": [{"$group": {"_id": "$task_id", **_rates()}}],
            "totals": [{"$group": {"_id": None, **_rates({"cost_usd": {"$sum": "$cost_usd"}})}}],
        }},
    ]


def _task_row(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": doc["_id"],
        "n": doc["n"],
        "success_rate": round(doc["success_rate"], 4),
        "mean_distance_m": round(doc["mean_distance_m"], 2),
    }


def _code_row(doc: dict[str, Any]) -> dict[str, Any]:
    if doc["_id"] is None:
        raise ValueError(
            f"{doc['count']} runs have success=False yet passed sanity, did not fall and reached "
            f"target_m; the runs disagree with sim.verifier (e.g. trace_ids {doc['examples']})"
        )
    row = {
        "code": doc["_id"],
        "count": doc["count"],
        "examples": [t for t in doc["examples"] if t is not None],
    }
    if doc["_id"] == "sanity":
        row["violations"] = sorted(v for v in doc["violations"] if v)
    return row


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    """The event payload: everything but per_task, violation texts capped, well under 2 KB."""
    codes = []
    for row in result["failure_codes"]:
        row = dict(row)
        if "violations" in row:
            row["violations"] = [v[:VIOLATION_CHARS] for v in row["violations"][:N_VIOLATIONS]]
        codes.append(row)
    return {k: v for k, v in result.items() if k != "per_task"} | {"failure_codes": codes}


def sense(
    version_id: str,
    split: str = "train",
    db_name: str = ADA,
    round_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate one version's runs on one split, emit one sensor event, return the full reading."""
    facets = next(db(db_name).runs.aggregate(_pipeline(version_id, split)))
    totals = facets["totals"][0] if facets["totals"] else {
        "n": 0, "success_rate": 0.0, "mean_distance_m": 0.0, "cost_usd": 0.0,
    }
    per_task = sorted((_task_row(d) for d in facets["per_task"]), key=lambda r: r["task_id"])
    worst = sorted(per_task, key=lambda r: (r["success_rate"], r["mean_distance_m"], r["task_id"]))
    codes = sorted((_code_row(d) for d in facets["codes"]), key=lambda r: CODES.index(r["code"]))

    result = {
        "version_id": version_id,
        "split": split,
        "n": totals["n"],
        "success_rate": round(totals["success_rate"], 4),
        "mean_distance_m": round(totals["mean_distance_m"], 2),
        "cost_usd": round(totals["cost_usd"], 6),
        "failure_codes": codes,
        "worst_tasks": [{k: r[k] for k in ("task_id", "success_rate", "mean_distance_m")}
                        for r in worst[:N_WORST]],
        "per_task": per_task,
    }
    emit("sensor", "info", _summary(result), round_id=round_id, version_id=version_id, db_name=db_name)
    return result
