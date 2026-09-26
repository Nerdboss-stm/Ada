"""Run a harness version over one split and score it (SPEC §3).

    python -m harness.run --version v0 --split train|holdout|showcase --k 2

One agent episode per task; the gait is evaluated by sim.verifier on the task's first
k evaluation seeds, one `runs` document per seed. Then the `versions` document is
upserted: train writes train_reliability, holdout writes holdout_reliability_80; both
also write mean_distance_m, cost_per_run_usd (mean cost per episode) and n (tasks x k)
from that split's runs. Showcase records frames through sim.record.write_frames
(kind "showcase", or "frontier" for the frontier baseline) and sets showcase_frames_id.

Holdout (and the swap, which runs the holdout split) scores mean_distance_m exactly like the
gate's score() (NOTES [B13]): a physics-violating run (sanity.pass false) or a run with no gait
counts as -target_m, the target walked backward. Train keeps the raw mean; the gate re-scores
train runs itself.

Holdout runs its per-task episodes in parallel (HOLDOUT_WORKERS threads); results keep task order.

v0 always runs V0_HARNESS; every other version loads its harness from `versions`.

store_as runs one split's tasks but stores the runs (and harness events) under another split
name, in their own episode threads and traces (tag = store_as). Such runs never touch
`versions`: run_split returns a Version whose metrics describe those runs, unsaved (the
swap uses store_as="swap"). fresh=True makes every model call live (core.llm fresh).

Before the first episode, harness.guardrails.load_guardrails() must pass; on a
mismatch the run refuses to start.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from core.contracts import FrameKind, Harness, Metrics, Run, RunSplit, Sanity, Split, Task, Version
from core.db import ADA, ADA_CKPT, db
from core.events import emit
from harness.agent import V0_HARNESS, EpisodeResult, run_episode
from harness.guardrails import GuardrailError, load_guardrails

SPLITS: tuple[Split, ...] = ("train", "holdout", "showcase")
SUCCESS_SHARE = 0.8  # a task is reliable when at least 80% of its seeds succeed
HOLDOUT_WORKERS = 6  # [A11] holdout episodes run in parallel; cached_chat caps model concurrency itself
NO_GAIT_RESULT: dict[str, Any] = {
    "distance_m": 0.0, "fell": False,
    "sanity": {"pass": True, "violation": None, "violation_frame": None}, "success": False,
}


def penalized(run: dict[str, Any]) -> bool:
    """A run that broke a physical bound, or one without a submitted gait (stored as gait {});
    a run without the gait key counts as having one, as in the gate."""
    return (run.get("sanity") or {}).get("pass", True) is False or ("gait" in run and not run["gait"])


def scored_distance(run: dict[str, Any]) -> float:
    """The run's distance, or -target_m when penalized; the run dict carries its task's target_m."""
    return -float(run["target_m"]) if penalized(run) else float(run["distance_m"])


def mean_distance_m(runs: list[dict[str, Any]]) -> float:
    """Mean scored distance over run dicts (each with target_m), rounded to 4 places."""
    return round(mean(scored_distance(r) for r in runs), 4) if runs else 0.0


def load_tasks(split: Split, db_name: str = ADA) -> list[Task]:
    docs = list(db(db_name).tasks.find({"split": split}).sort("_id", 1))
    if docs:
        return [Task.model_validate(d) for d in docs]
    from sim.tasks import build_tasks

    return [t for t in build_tasks() if t.split == split]


def load_harness(version_id: str, db_name: str = ADA) -> Harness:
    """v0 is always V0_HARNESS; every other version's harness comes from its versions document."""
    if version_id == "v0":
        return V0_HARNESS
    doc = db(db_name).versions.find_one({"_id": version_id}, {"harness": 1})
    if doc and doc.get("harness"):
        return Harness.model_validate(doc["harness"])
    raise LookupError(f"no versions document {version_id!r} with a harness")


def _evaluate(ep: EpisodeResult, task: Task, seed: int, record: bool) -> dict[str, Any]:
    if ep.gait is None:
        return dict(NO_GAIT_RESULT)
    verifier = importlib.import_module("sim.verifier")
    return verifier.evaluate(ep.gait, task, seed, record=record)


def run_task(
    task: Task, version_id: str, harness: Harness, split: Split, k: int, *,
    db_name: str = ADA, ckpt_db: str = ADA_CKPT, frames_kind: FrameKind = "showcase",
    store_as: RunSplit | None = None, fresh: bool = False,
) -> dict[str, Any]:
    """One episode, k evaluated runs. Returns {episode, runs[], frames_id}."""
    label = store_as or split
    emit("harness", "start", {"task_id": task.id, "split": label, "k": k},
         version_id=version_id, db_name=db_name)
    ep = run_episode(task, version_id, harness, db_name=db_name, ckpt_db=ckpt_db, fresh=fresh,
                     tag=label if label != split else None)
    seeds = task.eval_seeds[:k]
    record = split == "showcase"
    cost_share = round(ep.cost_usd / len(seeds), 6) if seeds else 0.0
    token_share = ep.tokens // len(seeds) if seeds else 0
    runs, frames_id = [], None
    for seed in seeds:
        result = _evaluate(ep, task, seed, record)
        run = Run(
            task_id=task.id, seed=seed, split=label, version_id=version_id, model_id=ep.model_id,
            gait=ep.gait.model_dump() if ep.gait is not None else {},
            distance_m=float(result["distance_m"]), fell=bool(result["fell"]),
            sanity=Sanity.model_validate(result["sanity"]), success=bool(result["success"]),
            cost_usd=cost_share, tokens=token_share, trace_id=ep.trace_id,
            peak_torque=result.get("peak_torque") or {},
        )
        run_id = str(db(db_name).runs.insert_one(run.model_dump(by_alias=True, exclude={"id"})).inserted_id)
        runs.append(run)
        if record and ep.gait is not None:
            writer = importlib.import_module("sim.record")
            frames_id = writer.write_frames(
                result["frames"], run_id, version_id, frames_kind,
                violation_frame=run.sanity.violation_frame, db_name=db_name,
            )
    successes = sum(r.success for r in runs)
    ok = bool(runs) and successes / len(runs) >= SUCCESS_SHARE
    emit("harness", "pass" if ok else "fail", {
        "task_id": task.id, "attempts": ep.attempts, "valid_gait": ep.gait is not None,
        "end_reason": ep.end_reason,
        "successes": successes, "n": len(runs),
        "mean_distance_m": round(mean(r.distance_m for r in runs), 4) if runs else 0.0,
        "cost_usd": ep.cost_usd,
    }, version_id=version_id, db_name=db_name)
    return {"episode": ep, "runs": runs, "ok": ok, "frames_id": frames_id, "target_m": task.target_m}


def _build_version(
    version_id: str, harness: Harness, split: Split, results: list[dict[str, Any]], db_name: str,
) -> Version:
    """The versions document these results would produce; nothing is written."""
    existing = db(db_name).versions.find_one({"_id": version_id}) or {}
    runs = [r for res in results for r in res["runs"]]
    reliability = round(mean(res["ok"] for res in results), 4) if results else 0.0
    metrics = dict(existing.get("metrics") or {
        "train_reliability": 0.0, "holdout_reliability_80": 0.0,
        "mean_distance_m": 0.0, "cost_per_run_usd": 0.0, "n": 0,
    })
    if split in ("train", "holdout"):
        # [B3] n, distance and cost always describe the last scored split; holdout is scored
        # last, so every spine row (v0, frontier, accepted) reads the same holdout numbers.
        scored = [{**r.model_dump(by_alias=True), "target_m": res["target_m"]}
                  for res in results for r in res["runs"]]
        metrics.update(
            mean_distance_m=(mean_distance_m(scored) if split == "holdout"  # [B13] gate parity
                             else round(mean(r.distance_m for r in runs), 4) if runs else 0.0),
            # cost to produce one gait: mean over episodes (one per task), not per seed
            cost_per_run_usd=round(mean(res["episode"].cost_usd for res in results), 6) if results else 0.0,
            n=len(runs),
        )
        metrics["train_reliability" if split == "train" else "holdout_reliability_80"] = reliability
    frames_ids = [res["frames_id"] for res in results if res["frames_id"]]
    return Version(
        _id=version_id,
        parent=existing.get("parent"),
        status=existing.get("status") or "baseline",
        harness=harness,
        metrics=Metrics.model_validate(metrics) if (existing.get("metrics") or split != "showcase") else None,
        showcase_frames_id=frames_ids[-1] if frames_ids else existing.get("showcase_frames_id"),
        created_at=existing.get("created_at") or datetime.now(timezone.utc),
    )


def _write_version(
    version_id: str, harness: Harness, split: Split, results: list[dict[str, Any]], db_name: str,
) -> Version:
    version = _build_version(version_id, harness, split, results, db_name)
    db(db_name).versions.replace_one({"_id": version_id}, version.model_dump(by_alias=True), upsert=True)
    return version


def run_split(
    version_id: str, split: Split, k: int, *, tasks: list[Task] | None = None,
    harness: Harness | None = None, db_name: str = ADA, ckpt_db: str = ADA_CKPT,
    frames_kind: FrameKind = "showcase", store_as: RunSplit | None = None, fresh: bool = False,
) -> Version:
    if split not in SPLITS:
        raise ValueError(f"split {split!r}; expected one of {SPLITS}")
    if k < 1:
        raise ValueError("k must be >= 1")
    load_guardrails(db_name)  # raises GuardrailError: the run refuses to start
    harness = harness or load_harness(version_id, db_name)
    tasks = tasks if tasks is not None else load_tasks(split, db_name)
    label = store_as or split
    db(db_name).runs.delete_many({"version_id": version_id, "split": label})  # reruns replace
    def one(t: Task) -> dict[str, Any]:
        return run_task(t, version_id, harness, split, k, db_name=db_name, ckpt_db=ckpt_db,
                        frames_kind=frames_kind, store_as=store_as, fresh=fresh)

    if split == "holdout":
        with ThreadPoolExecutor(max_workers=HOLDOUT_WORKERS) as pool:
            results = list(pool.map(one, tasks))  # map keeps task order
    else:
        results = [one(t) for t in tasks]
    if label != split:  # [B11] relabelled runs (the swap) never update a version's metrics
        return _build_version(version_id, harness, split, results, db_name)
    return _write_version(version_id, harness, split, results, db_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.run")
    parser.add_argument("--version", required=True)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        version = run_split(args.version, args.split, args.k)
    except GuardrailError as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    print(json.dumps(version.model_dump(by_alias=True, mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
