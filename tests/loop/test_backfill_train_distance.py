"""scripts.backfill_train_distance: gate-subset train mean, None when a task had no gait."""

import uuid
from datetime import datetime, timezone

import pytest

from core.contracts import Version
from core.db import ADA_TEST, db
from gate.verifier_stage import K, gate_tasks
from scripts.backfill_train_distance import backfill

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
METRICS = {"train_reliability": 0.5, "holdout_reliability_80": 0.2, "mean_distance_m": 7.0,
           "cost_per_run_usd": 0.001, "n": 18}


@pytest.fixture
def world():
    adb = db(ADA_TEST)
    tag = uuid.uuid4().hex[:8]
    tasks = gate_tasks(ADA_TEST)
    ids = {k: f"t-a11-{tag}-{k}" for k in ("full", "gap", "none", "viol")}

    def version(vid):
        adb.versions.insert_one({"_id": vid, "parent": "v0", "status": "accepted",
                                 "harness": {}, "metrics": dict(METRICS), "created_at": NOW})

    def runs(vid, split, dist, no_gait=(), violate=()):
        adb.runs.insert_many([
            {"task_id": t.id, "seed": s, "split": split, "version_id": vid, "success": True,
             "distance_m": dist + i, "gait": {} if t.id in no_gait else {"power": 1.0},
             "sanity": {"pass": (t.id, s) not in violate}}
            for i, t in enumerate(tasks) for s in t.eval_seeds[:K]])

    for vid in ids.values():
        version(vid)
    runs(ids["full"], "train", 1.0)            # mean of 1..6 = 3.5
    runs(ids["full"], "holdout", 50.0)         # never read
    runs(ids["gap"], "train", 1.0, no_gait={tasks[2].id})
    runs(ids["viol"], "train", 1.0, violate={(tasks[0].id, tasks[0].eval_seeds[0])})
    yield ids, tasks, adb

    adb.versions.delete_many({"_id": {"$in": list(ids.values())}})
    adb.runs.delete_many({"version_id": {"$in": list(ids.values())}})


def test_backfill_sets_gate_mean_or_none(world):
    ids, tasks, adb = world
    rows = {r["version_id"]: r for r in backfill(ADA_TEST, ids=list(ids.values()))}

    assert (rows[ids["full"]]["status"], rows[ids["full"]]["train_mean_distance_m"]) == ("set", 3.5)
    assert (rows[ids["gap"]]["status"], rows[ids["gap"]]["train_mean_distance_m"]) == ("set", None)
    assert rows[ids["gap"]]["gaits"] == len(tasks) - 1
    assert rows[ids["none"]]["status"] == "skipped"

    full = Version.model_validate(adb.versions.find_one({"_id": ids["full"]}))
    assert full.metrics.train_mean_distance_m == 3.5 and full.metrics.mean_distance_m == 7.0
    gap = adb.versions.find_one({"_id": ids["gap"]})
    assert "train_mean_distance_m" in gap["metrics"] and gap["metrics"]["train_mean_distance_m"] is None
    assert "train_mean_distance_m" not in adb.versions.find_one({"_id": ids["none"]})["metrics"]


def test_backfill_counts_a_violating_run_as_minus_target(world):
    """A14: the backfill scores like the gate; one violating run walks its target backward."""
    ids, tasks, _ = world
    [row] = backfill(ADA_TEST, ids=[ids["viol"]])
    total = 2 * sum(1.0 + i for i in range(len(tasks))) - 1.0 - tasks[0].target_m
    assert row["train_mean_distance_m"] == round(total / (2 * len(tasks)), 4)


def test_dry_run_writes_nothing(world):
    ids, _, adb = world
    rows = backfill(ADA_TEST, ids=[ids["full"]], dry_run=True)
    assert rows[0]["train_mean_distance_m"] == 3.5
    assert "train_mean_distance_m" not in adb.versions.find_one({"_id": ids["full"]})["metrics"]
