"""A11/A15: a missing gait is a failed run (-target_m), attempt frames for every candidate,
parallel rollout order."""

import time

import pytest

from core.db import ADA_TEST
from gate import verifier_stage


def outcome(tasks, ok=True, distance=2.0):
    return {t.id: (ok, distance) for t in tasks}


def half(tasks, distance=1.5):
    return {t.id: (i < 3, distance) for i, t in enumerate(tasks)}


def both_seeds(*tasks):
    return {(t.id, s) for t in tasks for s in t.eval_seeds[:verifier_stage.K]}


def verifier_payload(world, status):
    return [p for s, st, p in world.events() if s == "gate.verifier" and st == status][0]


@pytest.mark.parametrize("distance,verdict", [(1.8, "accepted"), (1.6, "rejected")])
def test_missing_gait_is_a_failed_run_not_a_veto(world, adb, distance, verdict):
    """A15, the r-20260926T182513 case: one task without a gait and half the parent's
    violations. The missing gait counts as -target_m; the 0.1 m bar decides."""
    ts = world.tasks
    world.parent_runs(outcome(ts, ok=False, distance=1.5), violations=both_seeds(ts[0], ts[1]))
    world.violate = both_seeds(ts[0])
    world.no_gait = {ts[5].id}
    world.outcome = outcome(ts, ok=False, distance=distance)
    res = world.gate()

    cand = round((-2 * ts[0].target_m - 2 * ts[5].target_m + 8 * distance) / 12, 4)
    par = round((-2 * ts[0].target_m - 2 * ts[1].target_m + 8 * 1.5) / 12, 4)
    assert res["train_mean_distance_m"] == res["train"]["mean_distance_m"] == cand
    assert res["parent_train_mean_distance_m"] == par
    assert res["verdict"] == verdict
    if verdict == "accepted":
        assert cand - par >= verifier_stage.MIN_DISTANCE_GAIN_M and res["reason"] is None
    else:
        assert cand - par < verifier_stage.MIN_DISTANCE_GAIN_M
        assert res["reason"].startswith("no strict improvement")
    payload = verifier_payload(world, "pass" if verdict == "accepted" else "fail")
    assert payload["gaits"] == {"candidate": 5, "parent": 6}
    assert payload["violations"] == {"candidate": 2, "parent": 4}
    assert adb.runs.count_documents({"version_id": world.cand_id, "gait": {}}) == 2


@pytest.mark.parametrize("parent_distance", [
    1.5,
    -0.4,  # the live v2 case: nothing submitted against a parent walking backward
])
def test_submitting_nothing_scores_minus_target_and_is_rejected(world, adb, parent_distance):
    ts = world.tasks
    world.parent_runs(outcome(ts, ok=False, distance=parent_distance))
    world.no_gait = {t.id for t in ts}
    world.outcome = outcome(ts, distance=3.0)
    res = world.gate()

    assert res["verdict"] == "rejected" and res["stages"]["gate.verifier"] == "fail"
    assert res["reason"].startswith("no strict improvement")
    assert res["train_mean_distance_m"] == round(sum(-t.target_m for t in ts) / 6, 4)
    assert res["train"]["reliability"] == 0.0 and res["frames_id"] is None
    fail = verifier_payload(world, "fail")
    assert fail["reason"] == res["reason"] and fail["gaits"] == {"candidate": 0, "parent": 6}
    assert fail["violations"] == {"candidate": 0, "parent": 0}
    assert adb.runs.count_documents({"version_id": world.cand_id, "gait": {}}) == 12


def test_missing_gaits_score_as_minus_target_on_both_sides(world, adb):
    """Parent and candidate both miss one task: each side's missing runs count as -target_m."""
    ts = world.tasks
    world.parent_runs(half(ts), gaits=[t.id for t in ts[:5]])
    world.no_gait = {ts[5].id}
    world.outcome = outcome(ts)  # 5/6 reliable vs 0.5
    res = world.gate()

    assert res["verdict"] == "accepted" and res["reason"] is None
    assert res["train_mean_distance_m"] == round((10 * 2.0 - 2 * ts[5].target_m) / 12, 4)
    assert res["parent_train_mean_distance_m"] == round((10 * 1.5 - 2 * ts[5].target_m) / 12, 4)


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
    # A14: every run rolls; each violating run counts as -target_m
    assert res["train_mean_distance_m"] == round(sum(-t.target_m for t in world.tasks) / 6, 4)


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
