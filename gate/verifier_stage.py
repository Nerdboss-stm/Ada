"""gate.verifier (SPEC §4 stage 1): roll the candidate on loop.subset.GATE_TASKS, k = 2.

    run(parent, candidate, db_name) -> dict

One harness.agent episode per task, sim.verifier.evaluate on the task's first k
evaluation seeds, one `runs` document per seed under the candidate's version id.
The parent's numbers come from its existing train runs on the same tasks and seeds;
without them the stage fails before any episode runs.

The six episodes and their evaluations run in parallel (MAX_WORKERS threads); results,
`runs` documents and every decision below follow task order, as if run one by one.

Judged by rates (NOTES [A14]): every run rolls, and a failed run counts as -target_m, walking
the target backward, in every mean distance: a physics-violating run (sanity.pass false) and a
run with no submitted gait (gait {}) alike (A15), so neither cheating nor submitting nothing can
raise a score. The candidate is rejected with the verifier's reason word for word only when it
has more violating runs than the parent on the same tasks and seeds; the first violating run in
task order is then re-evaluated with record=True and its frames written through
sim.record.write_frames (kind "rejected", with violation_frame). Otherwise the stage passes
only on strict improvement: candidate reliability above the parent's, or equal reliability
with penalized mean distance at least MIN_DISTANCE_GAIN_M greater. Reliability follows
harness.run: a task is reliable when SUCCESS_SHARE of its seeds succeed; reliability is the mean
over tasks. Gait counts {candidate, parent} are kept in the result for the record only.

Every candidate that rolls, accepted or rejected, is re-evaluated on ATTEMPT_TASK's first
evaluation seed with record=True (its gate gait, no model call); the frames are written
with kind "showcase" and returned as attempt_frames_id. train_mean_distance_m is the
candidate's penalized train mean distance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any

from core.contracts import FrameKind, Run, Sanity, Task, Version
from core.db import ADA, db
from harness import agent
from harness.agent import EpisodeResult
from harness.run import NO_GAIT_RESULT, SUCCESS_SHARE, load_tasks
from loop.subset import GATE_TASKS
from sim import record, verifier

K = 2
MIN_DISTANCE_GAIN_M = 0.1
MAX_WORKERS = 6
ATTEMPT_TASK = "train-s0-f1"  # GATE_TASKS[0]: flat ground, full friction
_EPS = 1e-9


def gate_tasks(db_name: str = ADA) -> list[Task]:
    by_id = {t.id: t for t in load_tasks("train", db_name)}
    missing = [t for t in GATE_TASKS if t not in by_id]
    if missing:
        raise LookupError(f"gate tasks not found: {missing}")
    return [by_id[t] for t in GATE_TASKS]


def violated(run: dict[str, Any]) -> bool:
    """A run that broke a physical bound; runs stored before sanity existed count as passing."""
    return (run.get("sanity") or {}).get("pass", True) is False


def no_gait(run: dict[str, Any]) -> bool:
    """A run whose episode submitted nothing (stored gait {}); a run without the key counts as
    having one."""
    return "gait" in run and not run["gait"]


def failed(run: dict[str, Any]) -> bool:
    return violated(run) or no_gait(run)


def scored_distance(run: dict[str, Any]) -> float:
    """The run's distance, or -target_m (the target walked backward) when it violated physics
    or submitted no gait."""
    return -float(run["target_m"]) if failed(run) else float(run["distance_m"])


def score(runs_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """{reliability, mean_distance_m, n, violations} over per-task lists of run dicts; each run
    carries its task's target_m so a failed run (violation or no gait) can count as -target_m."""
    runs = [r for rs in runs_by_task.values() for r in rs]
    oks = [sum(bool(r["success"]) and not failed(r) for r in rs) / len(rs) >= SUCCESS_SHARE
           for rs in runs_by_task.values() if rs]
    return {
        "reliability": round(mean(oks), 4) if oks else 0.0,
        "mean_distance_m": round(mean(scored_distance(r) for r in runs), 4) if runs else 0.0,
        "n": len(runs),
        "violations": sum(violated(r) for r in runs),
    }


def train_runs(version_id: str, tasks: list[Task], db_name: str = ADA,
               ) -> dict[str, list[dict[str, Any]]] | None:
    """A version's train runs on these tasks and first K seeds, each with its task's target_m;
    None if any task has none."""
    runs_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        runs = list(db(db_name).runs.find(
            {"version_id": version_id, "split": "train", "task_id": task.id,
             "seed": {"$in": task.eval_seeds[:K]}},
            {"success": 1, "distance_m": 1, "gait": 1, "sanity": 1},
        ))
        if not runs:
            return None
        runs_by_task[task.id] = [{**r, "target_m": task.target_m} for r in runs]
    return runs_by_task


def gait_count(runs_by_task: dict[str, list[dict[str, Any]]]) -> int:
    """Tasks on which a run carries a submitted gait (a run without one stores gait {})."""
    return sum(any(r.get("gait") for r in rs) for rs in runs_by_task.values())


def parent_score(parent: Version, tasks: list[Task], db_name: str = ADA) -> dict[str, Any] | None:
    """The parent's train numbers (with violations) on these tasks and first K seeds, plus
    `gaits`; None if any task has no runs."""
    runs_by_task = train_runs(parent.id, tasks, db_name)
    if runs_by_task is None:
        return None
    return {**score(runs_by_task), "gaits": gait_count(runs_by_task)}


def _roll(task: Task, candidate: Version, db_name: str) -> tuple[EpisodeResult, list[dict[str, Any]]]:
    """One episode and its K evaluations; runs in a worker thread, writes no runs."""
    ep = agent.run_episode(task, candidate.id, candidate.harness, db_name=db_name)
    results = [verifier.evaluate(ep.gait, task, seed) if ep.gait is not None else dict(NO_GAIT_RESULT)
               for seed in task.eval_seeds[:K]]
    return ep, results


def roll_all(tasks: list[Task], candidate: Version, db_name: str = ADA,
             ) -> list[tuple[EpisodeResult, list[dict[str, Any]]]]:
    """_roll for every task in parallel; the list is in task order."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return list(pool.map(lambda t: _roll(t, candidate, db_name), tasks))


def _write_recorded(gait, task: Task, seed: int, run_id: str, version_id: str, kind: FrameKind,
                    db_name: str) -> tuple[str, int | None]:
    result = verifier.evaluate(gait, task, seed, record=True)
    frame = result["sanity"]["violation_frame"]
    frames_id = record.write_frames(result["frames"], run_id, version_id, kind,
                                    violation_frame=frame, db_name=db_name)
    return frames_id, frame


def first_violation(runs_by_task: dict[str, list[dict[str, Any]]]) -> tuple[str, int] | None:
    """(task_id, seed index) of the first violating run in task order, or None."""
    return next(((tid, i) for tid, rs in runs_by_task.items() for i, r in enumerate(rs)
                 if violated(r)), None)


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
        "parent": None, "trace_ids": [], "gaits": None, "attempt_frames_id": None,
        "train_mean_distance_m": None, "parent_train_mean_distance_m": None, "violations": None,
    }
    par = parent_score(parent, tasks, db_name)
    if par is None:
        out["reason"] = f"parent {parent.id} has no train runs on the gate tasks"
        return out
    out["parent"] = par
    out["parent_train_mean_distance_m"] = par["mean_distance_m"]

    db(db_name).runs.delete_many({"version_id": candidate.id, "split": "train",
                                  "task_id": {"$in": list(GATE_TASKS)}})  # reruns replace
    rolled = roll_all(tasks, candidate, db_name)
    runs_by_task: dict[str, list[dict[str, Any]]] = {}
    run_ids: dict[tuple[str, int], str] = {}
    costs = [ep.cost_usd for ep, _ in rolled]
    out["trace_ids"] = [ep.trace_id for ep, _ in rolled]
    cand_gaits = sum(ep.gait is not None for ep, _ in rolled)
    out["gaits"] = {"candidate": cand_gaits, "parent": par["gaits"]}
    for task, (ep, results) in zip(tasks, rolled):
        seeds = task.eval_seeds[:K]
        gait = ep.gait.model_dump() if ep.gait is not None else {}
        runs_by_task[task.id] = []
        for seed, result in zip(seeds, results):
            run_doc = Run(
                task_id=task.id, seed=seed, split="train", version_id=candidate.id,
                model_id=ep.model_id, gait=gait,
                distance_m=float(result["distance_m"]), fell=bool(result["fell"]),
                sanity=Sanity.model_validate(result["sanity"]), success=bool(result["success"]),
                cost_usd=round(ep.cost_usd / len(seeds), 6), tokens=ep.tokens // len(seeds),
                trace_id=ep.trace_id, peak_torque=result.get("peak_torque") or {},
            )
            run_id = str(db(db_name).runs.insert_one(
                run_doc.model_dump(by_alias=True, exclude={"id"})).inserted_id)
            run_ids[(task.id, seed)] = run_id
            runs_by_task[task.id].append({**result, "target_m": task.target_m, "gait": gait})

    for task, (ep, _) in zip(tasks, rolled):  # attempt frames: every candidate, any verdict
        key = (task.id, task.eval_seeds[0])
        if task.id == ATTEMPT_TASK and ep.gait is not None and key in run_ids:
            out["attempt_frames_id"], _ = _write_recorded(
                ep.gait, task, key[1], run_ids[key], candidate.id, "showcase", db_name)

    cand = score(runs_by_task)
    out["violations"] = {"candidate": cand["violations"], "parent": par["violations"]}
    out["train"] = {**cand, "cost_per_run_usd": round(mean(costs), 6) if costs else 0.0}
    out["train_mean_distance_m"] = cand["mean_distance_m"]
    if cand["violations"] > par["violations"]:
        task_id, i = first_violation(runs_by_task)
        task = next(t for t in tasks if t.id == task_id)
        seed = task.eval_seeds[i]
        ep = rolled[tasks.index(task)][0]
        out["reason"] = runs_by_task[task_id][i]["sanity"]["violation"]
        out["frames_id"], out["violation_frame"] = _write_recorded(
            ep.gait, task, seed, run_ids[(task_id, seed)], candidate.id, "rejected", db_name)
    else:
        out["reason"] = improvement_reason(cand, par)
    out["pass"] = out["reason"] is None
    return out
