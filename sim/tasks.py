"""Tasks (SPEC §2.3): 12 train, 6 holdout, 1 showcase, written to `tasks`.

Seed blocks: train/holdout evaluation seeds count up from 0, the showcase evaluation
seed is 9000, practice seeds count up from 10000. Every seed belongs to one task.

    python -m sim.tasks            # print the tasks
    python -m sim.tasks --write    # upsert them into db("ada").tasks
"""

from __future__ import annotations

import itertools
import json
import sys

from pymongo.database import Database

from core.contracts import Task

TARGET_M = 2.0
N_EVAL = 5
N_PRACTICE = 3
EVAL_START = 0
SHOWCASE_SEED = 9000
PRACTICE_START = 10_000

TRAIN = list(itertools.product((0.0, 2.0, 4.0, 6.0), (1.0, 0.6, 0.35)))
HOLDOUT = list(itertools.product((1.0, 3.0, 5.0), (0.8, 0.45)))
SHOWCASE = (0.0, 1.0)


def task_id(split: str, slope_deg: float, friction: float) -> str:
    return "showcase" if split == "showcase" else f"{split}-s{slope_deg:g}-f{friction:g}"


def build_tasks() -> list[Task]:
    eval_next, practice_next = EVAL_START, PRACTICE_START
    tasks = []
    specs = [("train", c) for c in TRAIN] + [("holdout", c) for c in HOLDOUT] + [("showcase", SHOWCASE)]
    for split, (slope, friction) in specs:
        if split == "showcase":
            eval_seeds = [SHOWCASE_SEED]
        else:
            eval_seeds = list(range(eval_next, eval_next + N_EVAL))
            eval_next += N_EVAL
        practice_seeds = list(range(practice_next, practice_next + N_PRACTICE))
        practice_next += N_PRACTICE
        tasks.append(Task(
            _id=task_id(split, slope, friction), split=split, slope_deg=slope, friction=friction,
            target_m=TARGET_M, eval_seeds=eval_seeds, practice_seeds=practice_seeds,
        ))
    return tasks


def write_tasks(database: Database) -> int:
    """Upsert every task into database.tasks by _id; returns the number written."""
    tasks = build_tasks()
    for t in tasks:
        doc = t.model_dump(by_alias=True)
        database.tasks.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    return len(tasks)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--write"]:
        from core.db import db

        print(f"wrote {write_tasks(db())} tasks to ada.tasks")
    elif not args:
        print(json.dumps([t.model_dump(by_alias=True) for t in build_tasks()], indent=2))
    else:
        sys.exit("usage: python -m sim.tasks [--write]")
