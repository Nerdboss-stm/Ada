"""A11: gait-count rejection, attempt frames for every candidate, parallel rollout order."""

import time

import pytest

from core.db import ADA_TEST
from gate import verifier_stage


def outcome(tasks, ok=True, distance=2.0):
    return {t.id: (ok, distance) for t in tasks}


def half(tasks, distance=1.5):
    return {t.id: (i < 3, distance) for i, t in enumerate(tasks)}


@pytest.mark.parametrize("parent_outcome,no_gait,c", [
    # the live v2 case: nothing submitted, 0.0 m beat a parent walking backward
    (lambda ts: outcome(ts, ok=False, distance=-0.4), lambda ts: {t.id for t in ts}, 0),
    # four gaits at 3 m and 4/6 reliable would clear the improvement bar on its own
    (half, lambda ts: {ts[4].id, ts[5].id}, 4),
])
def test_fewer_gaits_than_parent_is_rejected(world, adb, parent_outcome, no_gait, c):
    world.parent_runs(parent_outcome(world.tasks))
    world.no_gait = no_gait(world.tasks)
    world.outcome = outcome(world.tasks, distance=3.0)
    res = world.gate()

    assert res["verdict"] == "rejected"
    assert res["reason"] == f"submitted a gait on {c}/6 tasks vs parent 6/6"
    assert res["stages"]["gate.verifier"] == "fail"
    assert res["train_mean_distance_m"] is None
    fail = [p for s, st, p in world.events() if s == "gate.verifier" and st == "fail"][0]
    assert fail["reason"] == res["reason"] and fail["gaits"] == {"candidate": c, "parent": 6}
    assert adb.runs.count_documents({"version_id": world.cand_id, "gait": {}}) == 2 * (6 - c)


def test_equal_gait_count_falls_through_to_improvement(world, adb):
    """Parent and candidate both miss one task: the comparison runs as before, and the
    candidate's train mean distance is None because a gate task got no gait."""
    world.parent_runs(half(world.tasks), gaits=[t.id for t in world.tasks[:5]])
    world.no_gait = {world.tasks[5].id}
    world.outcome = outcome(world.tasks)  # 5/6 reliable vs 0.5
    res = world.gate()

    assert res["verdict"] == "accepted" and res["reason"] is None
    assert res["train_mean_distance_m"] is None
    assert res["parent_train_mean_distance_m"] == 1.5


def attempt_frames(adb, world, res):
    frames = adb.frames.find_one({"_id": res["attempt_frames_id"]})
    task = world.tasks[0]
    run = adb.runs.find_one({"version_id": world.cand_id, "task_id": task.id,
                             "seed": task.eval_seeds[0]})
    return frames, run


def test_rejected_candidate_gets_attempt_frames(world, adb):
    world.parent_runs(half(world.tasks, distance=1.5))
    world.outcome = half(world.tasks, distance=1.55)  # no strict improvement
    res = world.gate()

    assert res["verdict"] == "rejected" and res["reason"].startswith("no strict improvement")
    assert world.tasks[0].id == verifier_stage.ATTEMPT_TASK == "train-s0-f1"
    frames, run = attempt_frames(adb, world, res)
    assert (frames["kind"], frames["version_id"], frames["run_id"]) == \
        ("showcase", world.cand_id, str(run["_id"]))
    assert res["train_mean_distance_m"] == 1.55 and res["parent_train_mean_distance_m"] == 1.5
    payload = [p for s, st, p in world.events() if s == "gate.verifier"][-1]
    assert payload["attempt_frames_id"] == res["attempt_frames_id"]
    assert payload["actual_delta_m"] == 0.05
    assert world.llm_calls == []


def test_violating_candidate_gets_attempt_and_rejected_frames(world, adb):
    world.parent_runs(half(world.tasks))
    world.power = 5.0
    res = world.gate()

    assert res["frames_id"] and res["attempt_frames_id"]
    assert res["frames_id"] != res["attempt_frames_id"]
    frames, run = attempt_frames(adb, world, res)
    assert frames["kind"] == "showcase" and frames["run_id"] == str(run["_id"])
    assert res["train_mean_distance_m"] is None  # cut short: mean covers one run


def test_no_gait_on_attempt_task_writes_no_attempt_frames(world, adb):
    world.parent_runs(half(world.tasks), gaits=[])
    world.no_gait = {world.tasks[0].id}
    world.outcome = outcome(world.tasks)
    res = world.gate()

    assert res["attempt_frames_id"] is None
    assert adb.frames.count_documents({"version_id": world.cand_id}) == 0


def test_parallel_rollout_keeps_task_order(world, adb):
    world.parent_runs(half(world.tasks))
    world.outcome = {t.id: (True, float(i)) for i, t in enumerate(world.tasks)}
    world.delay = {t.id: 0.05 * (6 - i) for i, t in enumerate(world.tasks)}  # first finishes last
    parent, cand = world.versions()

    ids = [t.id for t in world.tasks]
    started = time.monotonic()
    rolled = verifier_stage.roll_all(world.tasks, cand, ADA_TEST)  # no DB writes: timed alone
    elapsed = time.monotonic() - started
    assert world.episodes == ids[::-1]  # completed in reverse ...
    assert elapsed < sum(world.delay.values())  # ... concurrently
    assert [ep.trace_id for ep, _ in rolled] == [world.trace_id(t) for t in ids]  # ... in task order

    world.episodes.clear()
    res = verifier_stage.run(parent, cand, ADA_TEST)
    assert world.episodes == ids[::-1]
    assert res["trace_ids"] == [world.trace_id(t) for t in ids]
    runs = list(adb.runs.find({"version_id": world.cand_id}).sort("_id", 1))
    assert [(r["task_id"], r["distance_m"]) for r in runs] == \
        [(t, float(i)) for i, t in enumerate(ids) for _ in range(verifier_stage.K)]
