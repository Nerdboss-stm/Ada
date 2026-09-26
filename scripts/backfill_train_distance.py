"""Backfill metrics.train_mean_distance_m on existing versions (NOTES [A11] FIELDS).

    uv run python -m scripts.backfill_train_distance [--db ada] [--dry-run]

For every versions document with metrics: its train runs on the six gate tasks and the first
K evaluation seeds, the same subset the gate scores children on, so the leader compares like
with like. The value is their mean distance, or None when any gate task has no run carrying a
submitted gait (a version that never submitted cannot lead on distance). A version missing
train runs on any gate task is skipped and left unchanged. Holdout runs are never read.
Prints one line per version.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from core.db import ADA, db
from gate.verifier_stage import gait_count, gate_tasks, score, train_mean_distance, train_runs


def backfill(db_name: str = ADA, ids: list[str] | None = None, dry_run: bool = False,
             ) -> list[dict[str, Any]]:
    """[{version_id, status "set"|"skipped", train_mean_distance_m, gaits}] in _id order."""
    tasks = gate_tasks(db_name)
    query: dict[str, Any] = {"metrics": {"$type": "object"}}
    if ids is not None:
        query["_id"] = {"$in": ids}
    out = []
    for doc in db(db_name).versions.find(query, {"_id": 1}).sort("_id", 1):
        vid = str(doc["_id"])
        runs_by_task = train_runs(vid, tasks, db_name)
        if runs_by_task is None:
            out.append({"version_id": vid, "status": "skipped", "train_mean_distance_m": None,
                        "gaits": None})
            continue
        gaits = gait_count(runs_by_task)
        value = train_mean_distance(score(runs_by_task)["mean_distance_m"], gaits, len(tasks))
        if not dry_run:
            db(db_name).versions.update_one(
                {"_id": vid}, {"$set": {"metrics.train_mean_distance_m": value}})
        out.append({"version_id": vid, "status": "set", "train_mean_distance_m": value,
                    "gaits": gaits})
    return out


def line(row: dict[str, Any], n_tasks: int) -> str:
    if row["status"] == "skipped":
        return f"{row['version_id']}  skipped: no train runs on every gate task"
    dist = row["train_mean_distance_m"]
    shown = "None" if dist is None else f"{dist:.2f} m"
    return f"{row['version_id']}  train_mean_distance_m {shown}  gaits {row['gaits']}/{n_tasks}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.backfill_train_distance")
    parser.add_argument("--db", default=ADA)
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args(argv)
    n_tasks = len(gate_tasks(args.db))
    for row in backfill(args.db, dry_run=args.dry_run):
        print(line(row, n_tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
