"""gate.verifier (SPEC §4 stage 1): roll the candidate on loop.subset.GATE_TASKS, k = 2.

    run(parent, candidate, db_name) -> dict

One harness.agent episode per task, sim.verifier.evaluate on the task's first k
evaluation seeds, one `runs` document per seed under the candidate's version id.
The parent's numbers come from its existing train runs on the same tasks and seeds;
without them the stage fails before any episode runs.

The first sanity violation stops the rollout and fails the stage with the verifier's
reason word for word; that run is re-evaluated with record=True and its frames written
through sim.record.write_frames (kind "rejected", with violation_frame).
Otherwise the stage passes only on strict improvement: candidate reliability above the
parent's, or equal reliability with mean distance at least MIN_DISTANCE_GAIN_M greater.
Reliability follows harness.run: a task is reliable when SUCCESS_SHARE of its seeds
succeed; reliability is the mean over tasks.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from core.contracts import Run, Sanity, Task, Version
from core.db import ADA, db
from harness import agent
from harness.run import NO_GAIT_RESULT, SUCCESS_SHARE, load_tasks
from loop.subset import GATE_TASKS
from sim import record, verifier

K = 2
MIN_DISTANCE_GAIN_M = 0.1
_EPS = 1e-9


def gate_tasks(db_name: str = ADA) -> list[Task]:
    by_id = {t.id: t for t in load_tasks("train", db_name)}
    missing = [t for t in GATE_TASKS if t not in by_id]
    if missing:
        raise LookupError(f"gate tasks not found: {missing}")
    return [by_id[t] for t in GATE_TASKS]


def score(runs_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """{reliability, mean_distance_m, n} over per-task lists of run dicts."""
    runs = [r for rs in runs_by_task.values() for r in rs]
    oks = [sum(bool(r["success"]) for r in rs) / len(rs) >= SUCCESS_SHARE
           for rs in runs_by_task.values() if rs]
    return {
        "reliability": round(mean(oks), 4) if oks else 0.0,
        "mean_distance_m": round(mean(float(r["distance_m"]) for r in runs), 4) if runs else 0.0,
        "n": len(runs),
    }


def parent_score(parent: Version, tasks: list[Task], db_name: str = ADA) -> dict[str, Any] | None:
    """The parent's train numbers on these tasks and first K seeds; None if any task has no runs."""
    runs_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        runs = list(db(db_name).runs.find(
            {"version_id": parent.id, "split": "train", "task_id": task.id,
             "seed": {"$in": task.eval_seeds[:K]}},
            {"success": 1, "distance_m": 1},
        ))
        if not runs:
            return None
        runs_by_task[task.id] = runs
    return score(runs_by_task)


def _record_violation(gait, task: Task, seed: int, run_id: str, version_id: str, db_name: str,
                      ) -> tuple[str, int | None]:
    result = verifier.evaluate(gait, task, seed, record=True)
    frame = result["sanity"]["violation_frame"]
    frames_id = record.write_frames(result["frames"], run_id, version_id, "rejected",
                                    violation_frame=frame, db_name=db_name)
    return frames_id, frame


def _fmt(s: dict[str, Any]) -> str:
    return f"{s['reliability']:.4f}", f"{s['mean_distance_m']:.2f} m"


def improvement_reason(cand: dict[str, Any], par: dict[str, Any]) -> str | None:
    """None on strict improvement, else a reason stating both sides' numbers."""
    if cand["reliability"] > par["reliability"] + _EPS:
        return None
    same = abs(cand["reliability"] - par["reliability"]) <= _EPS
    if same and cand["mean_distance_m"] - par["mean_distance_m"] >= MIN_DISTANCE_GAIN_M - _EPS:
        return None
    (cr, cd), (pr, pd) = _fmt(cand), _fmt(par)
    return (f"no strict improvement: train reliability {cr} vs parent {pr}, "
            f"mean distance {cd} vs parent {pd}")


def run(parent: Version, candidate: Version, db_name: str = ADA) -> dict[str, Any]:
    tasks = gate_tasks(db_name)
    out: dict[str, Any] = {
        "pass": False, "reason": None, "frames_id": None, "violation_frame": None,
        "train": {"reliability": 0.0, "mean_distance_m": 0.0, "cost_per_run_usd": 0.0, "n": 0},
        "parent": None, "trace_ids": [],
    }
    par = parent_score(parent, tasks, db_name)
    if par is None:
        out["reason"] = f"parent {parent.id} has no train runs on the gate tasks"
        return out
    out["parent"] = par

    db(db_name).runs.delete_many({"version_id": candidate.id, "split": "train",
                                  "task_id": {"$in": list(GATE_TASKS)}})  # reruns replace
    runs_by_task: dict[str, list[dict[str, Any]]] = {}
    costs: list[float] = []
    for task in tasks:
        ep = agent.run_episode(task, candidate.id, candidate.harness, db_name=db_name)
        costs.append(ep.cost_usd)
        out["trace_ids"].append(ep.trace_id)
        seeds = task.eval_seeds[:K]
        runs_by_task[task.id] = []
        for seed in seeds:
            result = (verifier.evaluate(ep.gait, task, seed) if ep.gait is not None
                      else dict(NO_GAIT_RESULT))
            run_doc = Run(
                task_id=task.id, seed=seed, split="train", version_id=candidate.id,
                model_id=ep.model_id, gait=ep.gait.model_dump() if ep.gait is not None else {},
                distance_m=float(result["distance_m"]), fell=bool(result["fell"]),
                sanity=Sanity.model_validate(result["sanity"]), success=bool(result["success"]),
                cost_usd=round(ep.cost_usd / len(seeds), 6), tokens=ep.tokens // len(seeds),
                trace_id=ep.trace_id, peak_torque=result.get("peak_torque") or {},
            )
            run_id = str(db(db_name).runs.insert_one(
                run_doc.model_dump(by_alias=True, exclude={"id"})).inserted_id)
            runs_by_task[task.id].append(result)
            if not run_doc.sanity.pass_:
                out["reason"] = run_doc.sanity.violation
                out["frames_id"], out["violation_frame"] = _record_violation(
                    ep.gait, task, seed, run_id, candidate.id, db_name)
                break
        if out["reason"] is not None:
            break

    cand = score(runs_by_task)
    out["train"] = {**cand, "cost_per_run_usd": round(mean(costs), 6) if costs else 0.0}
    if out["reason"] is None:
        out["reason"] = improvement_reason(cand, par)
        out["pass"] = out["reason"] is None
    return out
