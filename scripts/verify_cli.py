"""The ./verify CLI (PLAN B8, SPEC §2.6 test 5 and §9): two Q&A proofs.

    ./verify --edit <edit_id> --clip-power 1.0
        edit -> frames_id -> frames doc -> run -> (gait, task, seed). Re-evaluates the
        same gait, task and seed twice with sim.verifier.evaluate: as stored, and with
        only power set to the given value. Prints the two results side by side, writes
        one edits document with origin "cli" and emits its four card cells
        (gate.verifier start then pass/fail; the other three info {"skipped": true}).

    ./verify --determinism <frames_id>
        frames doc -> run -> (gait, task, seed). Re-runs with record=True and prints
        whether frames_sha256 matches the stored sha256. Writes nothing.

Exit codes: 0 clipped run passes sanity / hashes match, 1 they do not, 2 lookup error.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from core.contracts import CARD_STAGES, Edit, Task
from core.db import ADA, db
from core.events import emit
from sim import verifier
from sim.gait import Gait

VERIFIER = CARD_STAGES[0]
SKIPPED = {"skipped": True}
SHA_SHOWN = 16


class LookupFailed(Exception):
    pass


def _one(coll: str, _id: str, db_name: str) -> dict[str, Any]:
    keys: list[Any] = [_id] + ([ObjectId(_id)] if ObjectId.is_valid(_id) else [])
    doc = db(db_name)[coll].find_one({"_id": {"$in": keys}})
    if doc is None:
        raise LookupFailed(f"no {coll} document {_id!r} in {db_name}")
    return doc


def _task(task_id: str, db_name: str) -> Task:
    doc = db(db_name).tasks.find_one({"_id": task_id})
    if doc is not None:
        return Task.model_validate(doc)
    from sim.tasks import build_tasks

    for t in build_tasks():
        if t.id == task_id:
            return t
    raise LookupFailed(f"no task {task_id!r}")


def _replay_inputs(frames_id: str, db_name: str) -> tuple[dict, dict, Task]:
    """(frames doc, run doc, task) behind one frames document."""
    frames = _one("frames", frames_id, db_name)
    run = _one("runs", frames["run_id"], db_name)
    return frames, run, _task(run["task_id"], db_name)


def _side(gait: Gait, result: dict[str, Any]) -> dict[str, Any]:
    s = result["sanity"]
    return {"power": gait.power, "sanity": "pass" if s["pass"] else "fail",
            "violation": s["violation"], "distance_m": round(float(result["distance_m"]), 4),
            "fell": bool(result["fell"])}


def _row(label: str, side: dict[str, Any]) -> str:
    sanity = "pass" if side["sanity"] == "pass" else f"FAIL {side['violation']}"
    return (f"{label:<9} {side['power']:<6.2f} {sanity:<52} "
            f"{side['distance_m']:>7.2f} m  {'yes' if side['fell'] else 'no'}")


def clip_power(edit_id: str, power: float, db_name: str = ADA) -> dict[str, Any]:
    """Re-evaluate a rejected edit's gait at `power`; write the cli edit and its events."""
    orig = _one("edits", edit_id, db_name)
    if not orig.get("frames_id"):
        raise LookupFailed(f"edit {edit_id!r} has no rejected frames")
    _, run, task = _replay_inputs(orig["frames_id"], db_name)
    gait = Gait.model_validate(run["gait"])
    clipped_gait = Gait.model_validate({**gait.model_dump(), "power": power})
    seed = int(run["seed"])

    original = _side(gait, verifier.evaluate(gait, task, seed))
    clipped = _side(clipped_gait, verifier.evaluate(clipped_gait, task, seed))
    passed = clipped["sanity"] == "pass"
    reason = (f"verification: same gait at power {power:g} passes sanity" if passed else
              f"verification: same gait at power {power:g} still fails: {clipped['violation']}")

    now = datetime.now(timezone.utc)
    edit = Edit(
        _id=f"{edit_id}.cli.{now:%Y%m%dT%H%M%S%f}", round_id=orig.get("round_id"),
        from_version=run["version_id"], to_version=None, origin="cli", primitive="verify",
        op="clip_power", path="gait.power", old=gait.power, new=clipped_gait.power,
        rationale=f"Same gait, task {task.id} and seed {seed} as edit {edit_id}; only power changed.",
        verdict="accepted" if passed else "rejected", reason=reason,
        violation_frame=orig.get("violation_frame"), frames_id=orig["frames_id"], created_at=now,
    )
    db(db_name).edits.insert_one(edit.model_dump(by_alias=True))

    ids = {"round_id": edit.round_id, "version_id": edit.from_version, "edit_id": edit.id,
           "db_name": db_name}
    emit(VERIFIER, "start", {"origin": "cli", "checks_edit": edit_id, "task_id": task.id,
                             "seed": seed, "clip_power": clipped_gait.power}, **ids)
    emit(VERIFIER, "pass" if passed else "fail",
         {"reason": reason, "original": original, "clipped": clipped}, **ids)
    for stage in CARD_STAGES[1:]:
        emit(stage, "info", SKIPPED, **ids)

    return {"edit_id": edit.id, "checks_edit": edit_id, "task_id": task.id, "seed": seed,
            "original": original, "clipped": clipped, "pass": passed, "reason": reason}


def determinism(frames_id: str, db_name: str = ADA) -> dict[str, Any]:
    """Re-run a frames document's gait, task and seed with record=True; compare sha256."""
    frames, run, task = _replay_inputs(frames_id, db_name)
    result = verifier.evaluate(Gait.model_validate(run["gait"]), task, int(run["seed"]), record=True)
    return {"frames_id": frames_id, "run_id": str(run["_id"]), "task_id": task.id,
            "seed": int(run["seed"]), "stored": frames["sha256"], "rerun": result["frames_sha256"],
            "n_frames": len(result["frames"]), "match": frames["sha256"] == result["frames_sha256"]}


def main(argv: list[str] | None = None, db_name: str = ADA) -> int:
    p = argparse.ArgumentParser(prog="verify", description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--edit", metavar="EDIT_ID", help="rejected edit to re-run")
    mode.add_argument("--determinism", metavar="FRAMES_ID", help="frames document to re-run")
    p.add_argument("--clip-power", type=float, default=1.0, help="power for --edit (default 1.0)")
    args = p.parse_args(argv)

    try:
        if args.edit:
            r = clip_power(args.edit, args.clip_power, db_name)
            print(f"edit {r['checks_edit']}   task {r['task_id']}   seed {r['seed']}")
            print(f"{'':<9} {'power':<6} {'sanity':<52} {'distance':>9}  fell")
            print(_row("original", r["original"]))
            print(_row("clipped", r["clipped"]))
            print(f"-> {r['reason']}   (written as edit {r['edit_id']})")
            return 0 if r["pass"] else 1
        r = determinism(args.determinism, db_name)
        print(f"frames {r['frames_id']}   run {r['run_id']}   task {r['task_id']}   seed {r['seed']}")
        print(f"stored  sha256 {r['stored'][:SHA_SHOWN]}   rerun  sha256 {r['rerun'][:SHA_SHOWN]}")
        print(f"{'MATCH' if r['match'] else 'MISMATCH'} ({r['n_frames']} frames)")
        return 0 if r["match"] else 1
    except (LookupFailed, ValueError) as e:  # ValueError: pydantic rejects an out-of-range power
        print(f"verify: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
