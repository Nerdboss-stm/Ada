"""The harness swap (NOTES [PLAN] RECAST, [B9]): same model, two harness documents.

    uv run python -m scripts.capture_swap --best <version_id> [--task <holdout task id>]

1. v0 (left), then the best version (right), back to back through harness.run.run_split
   on every holdout task, k = 3. Each side's holdout runs and metrics are replaced.
2. Showcase frames for both on one fixed holdout task (default: the first holdout task by
   _id): the side's stored run on the task's first eval seed is re-evaluated with
   record=True, must reproduce the stored distance, and is written as kind "showcase".
3. One `swaps` document: holdout tasks passed per side, frames ids, both agent model ids
   (from the runs), harness_diff (changed lines between the two harness documents; a
   model line first when the models differ), and verifier_sha, mujoco_version,
   manifest_version read from the live code.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import mujoco

from core.contracts import Harness, Swap, Task, Version
from core.db import ADA, ADA_CKPT, db
from harness.guardrails import VERIFIER_PATH, GuardrailError, file_sha256
from harness.run import SUCCESS_SHARE, load_harness, load_tasks, run_split
from sim import record, verifier
from sim.gait import Gait

LEFT = "v0"
SPLIT = "holdout"
K = 3


def _now_ms() -> datetime:
    """UTC now at Mongo's millisecond precision, so the stored document equals the returned one."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def harness_lines(harness: Harness) -> list[str]:
    return json.dumps(harness.model_dump(), indent=2, sort_keys=True).splitlines()


def harness_diff(left: Harness, right: Harness, model_id: str, right_model_id: str) -> list[str]:
    """Changed lines only ("-" left, "+" right); the model line first when the models differ."""
    lines = [
        line for line in difflib.unified_diff(harness_lines(left), harness_lines(right), n=0, lineterm="")
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    ]
    if model_id != right_model_id:
        lines.insert(0, f"model: {model_id} → {right_model_id} (chosen by the harness; neither was trained)")
    return lines


def _runs(version_id: str, db_name: str) -> list[dict]:
    return list(db(db_name).runs.find({"version_id": version_id, "split": SPLIT}).sort([("task_id", 1), ("seed", 1)]))


def holdout_passed(runs: list[dict]) -> int:
    """Tasks whose runs meet harness.run.SUCCESS_SHARE."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in runs:
        by_task[r["task_id"]].append(bool(r["success"]))
    return sum(sum(s) / len(s) >= SUCCESS_SHARE for s in by_task.values())


def model_id_of(version_id: str, runs: list[dict]) -> str:
    counts = Counter(r["model_id"] for r in runs)
    if not counts:
        raise LookupError(f"{version_id} has no {SPLIT} runs")
    return counts.most_common(1)[0][0]


def record_frames(version_id: str, task: Task, runs: list[dict], db_name: str) -> str | None:
    """Re-record the stored run on task's first eval seed; None when that run had no gait."""
    seed = task.eval_seeds[0]
    run = next((r for r in runs if r["task_id"] == task.id and r["seed"] == seed), None)
    if run is None:
        raise LookupError(f"{version_id} has no {SPLIT} run on {task.id} seed {seed}")
    if not run["gait"]:
        return None
    result = verifier.evaluate(Gait.model_validate(run["gait"]), task, seed, record=True)
    if round(float(result["distance_m"]), 4) != round(float(run["distance_m"]), 4):
        raise RuntimeError(f"{version_id} {task.id} seed {seed}: recorded distance "
                           f"{result['distance_m']} != stored {run['distance_m']}")
    return record.write_frames(result["frames"], str(run["_id"]), version_id, "showcase",
                               violation_frame=result["sanity"]["violation_frame"], db_name=db_name)


def capture_swap(
    best: str, *, left: str = LEFT, task_id: str | None = None, tasks: list[Task] | None = None,
    db_name: str = ADA, ckpt_db: str = ADA_CKPT,
) -> Swap:
    tasks = tasks if tasks is not None else load_tasks(SPLIT, db_name)
    if not tasks:
        raise LookupError(f"no {SPLIT} tasks")
    shown = next((t for t in tasks if t.id == task_id), None) if task_id else min(tasks, key=lambda t: t.id)
    if shown is None:
        raise LookupError(f"{task_id!r} is not a {SPLIT} task")
    harnesses = {vid: load_harness(vid, db_name) for vid in (left, best)}  # fail before any episode

    sides: dict[str, dict] = {}
    for vid in (left, best):  # back to back, left first
        version: Version = run_split(vid, SPLIT, K, tasks=tasks, harness=harnesses[vid],
                                     db_name=db_name, ckpt_db=ckpt_db)
        sides[vid] = {"version": version, "runs": _runs(vid, db_name)}
    for vid, side in sides.items():
        side["frames_id"] = record_frames(vid, shown, side["runs"], db_name)

    lm, rm = sides[left]["version"].metrics, sides[best]["version"].metrics
    model_id, right_model_id = (model_id_of(v, sides[v]["runs"]) for v in (left, best))
    captured_at = _now_ms()
    swap = Swap(
        _id=f"swap-{left}-{best}-{captured_at:%Y%m%dT%H%M%S%f}", captured_at=captured_at,
        task_id=shown.id, k=K, left_version=left, right_version=best,
        left_holdout=holdout_passed(sides[left]["runs"]), right_holdout=holdout_passed(sides[best]["runs"]),
        left_frames_id=sides[left]["frames_id"], right_frames_id=sides[best]["frames_id"],
        model_id=model_id, right_model_id=right_model_id,
        verifier_sha=file_sha256(VERIFIER_PATH), mujoco_version=mujoco.__version__,
        manifest_version=record.manifest_version(),
        harness_diff=harness_diff(harnesses[left], harnesses[best], model_id, right_model_id),
        left_mean_distance_m=lm.mean_distance_m, right_mean_distance_m=rm.mean_distance_m,
        left_cost_per_run_usd=lm.cost_per_run_usd, right_cost_per_run_usd=rm.cost_per_run_usd,
        n=len(sides[left]["runs"]),
    )
    db(db_name).swaps.insert_one(swap.model_dump(by_alias=True))
    return swap


def summary_line(s: Swap, n_tasks: int) -> str:
    return (f"{s.id}  {s.left_version} {s.left_holdout}/{n_tasks} vs {s.right_version} "
            f"{s.right_holdout}/{n_tasks} holdout (k {s.k}, {s.model_id} / {s.right_model_id})  "
            f"cost/gait ${s.left_cost_per_run_usd:.4f} vs ${s.right_cost_per_run_usd:.4f}  "
            f"{len(s.harness_diff)} changed lines")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.capture_swap")
    parser.add_argument("--best", required=True, help="version id for the right lane")
    parser.add_argument("--task", default=None, help="holdout task to record (default: first by _id)")
    args = parser.parse_args(argv)
    try:
        swap = capture_swap(args.best, task_id=args.task)
    except (GuardrailError, LookupError) as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    print(summary_line(swap, len(load_tasks(SPLIT))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
