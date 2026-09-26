"""Frozen baselines (SPEC §5 "Frontier baseline", PLAN B5).

    uv run python -m scripts.baseline [--force]

1. Upsert versions `frontier` (status frontier): V0_HARNESS with the agent step on the
   `frontier` model role.
2. frontier on holdout (k = 3), then its showcase run, recorded as frames kind "frontier".
3. v0 on holdout (k = 3) again, so its n is the holdout run count ([B3]); v0's showcase
   run only if its versions document has no showcase_frames_id.
4. One `baselines` document per version: split holdout, k 3, the holdout metrics, and a
   sha256 over the version's sorted holdout run results.

Once either baselines document exists the script refuses to run unless --force, so the
frontier line stays frozen. Prints one summary line per version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

from core.contracts import Baseline, Harness, Run, Split, Task, Version
from core.db import ADA, ADA_CKPT, db
from core.events import emit
from harness.agent import AGENT_STEP, V0_HARNESS
from harness.guardrails import GuardrailError
from harness.run import run_split

FRONTIER_ID = "frontier"
V0_ID = "v0"
VERSION_IDS = (FRONTIER_ID, V0_ID)
SPLIT: Split = "holdout"
K = 3
SHOWCASE_K = 1


class BaselineFrozen(RuntimeError):
    """A baselines document already exists and --force was not given."""


def frontier_harness() -> Harness:
    return V0_HARNESS.model_copy(deep=True, update={"model_per_step": {AGENT_STEP: "frontier"}})


def upsert_frontier(db_name: str = ADA) -> Version:
    coll = db(db_name).versions
    existing = coll.find_one({"_id": FRONTIER_ID}) or {}
    version = Version(
        _id=FRONTIER_ID, parent=None, status="frontier", harness=frontier_harness(),
        metrics=existing.get("metrics"), showcase_frames_id=existing.get("showcase_frames_id"),
        created_at=existing.get("created_at") or datetime.now(timezone.utc),
    )
    coll.replace_one({"_id": FRONTIER_ID}, version.model_dump(by_alias=True), upsert=True)
    return version


def runs_sha256(version_id: str, split: Split = SPLIT, db_name: str = ADA) -> str:
    """sha256 over the version's run results on split, sorted by (task_id, seed), without _id."""
    runs = [
        Run.model_validate(d).model_dump(by_alias=True, exclude={"id"}, mode="json")
        for d in db(db_name).runs.find({"version_id": version_id, "split": split})
    ]
    runs.sort(key=lambda r: (r["task_id"], r["seed"]))
    blob = json.dumps(runs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def write_baseline(version: Version, db_name: str = ADA) -> Baseline:
    baseline = Baseline(
        _id=version.id, version_id=version.id, split=SPLIT, k=K, metrics=version.metrics,
        sha256=runs_sha256(version.id, SPLIT, db_name), created_at=datetime.now(timezone.utc),
    )
    db(db_name).baselines.replace_one({"_id": baseline.id}, baseline.model_dump(by_alias=True), upsert=True)
    return baseline


def run_baseline(
    force: bool = False, *, holdout_tasks: list[Task] | None = None,
    showcase_tasks: list[Task] | None = None, db_name: str = ADA, ckpt_db: str = ADA_CKPT,
) -> list[Baseline]:
    frozen = [d["_id"] for d in db(db_name).baselines.find({"_id": {"$in": list(VERSION_IDS)}}, {"_id": 1})]
    if frozen and not force:
        raise BaselineFrozen(f"baselines already frozen for {sorted(frozen)}; pass --force to redo")

    upsert_frontier(db_name)
    kw = {"db_name": db_name, "ckpt_db": ckpt_db}
    for vid in VERSION_IDS:
        emit("baseline", "start", {"split": SPLIT, "k": K}, version_id=vid, db_name=db_name)
    frontier = run_split(FRONTIER_ID, SPLIT, K, tasks=holdout_tasks, **kw)
    run_split(FRONTIER_ID, "showcase", SHOWCASE_K, tasks=showcase_tasks, frames_kind="frontier", **kw)
    v0 = run_split(V0_ID, SPLIT, K, tasks=holdout_tasks, **kw)
    if not v0.showcase_frames_id:
        run_split(V0_ID, "showcase", SHOWCASE_K, tasks=showcase_tasks, **kw)

    baselines = [write_baseline(v, db_name) for v in (frontier, v0)]
    for b in baselines:
        emit("baseline", "pass", {
            "split": b.split, "k": b.k, "n": b.metrics.n,
            "holdout_reliability_80": b.metrics.holdout_reliability_80,
            "mean_distance_m": b.metrics.mean_distance_m,
            "cost_per_run_usd": b.metrics.cost_per_run_usd, "sha256": b.sha256,
        }, version_id=b.version_id, db_name=db_name)
    return baselines


def summary_line(b: Baseline) -> str:
    m = b.metrics
    return (f"{b.version_id:<8} holdout reliability {m.holdout_reliability_80:.2f}  n={m.n}  "
            f"mean distance {m.mean_distance_m:.2f} m  cost/run ${m.cost_per_run_usd:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.baseline")
    parser.add_argument("--force", action="store_true", help="redo frozen baselines")
    args = parser.parse_args(argv)
    try:
        baselines = run_baseline(force=args.force)
    except (BaselineFrozen, GuardrailError) as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    for b in baselines:
        print(summary_line(b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
