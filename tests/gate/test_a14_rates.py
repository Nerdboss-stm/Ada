"""A14: judge by rates. A violating run counts as -target_m; reject on physics only when the
candidate has more violating runs than the parent on the same tasks and seeds."""

from gate import verifier_stage
from sim import bounds


def outcome(tasks, ok=False, distance=1.5):
    return {t.id: (ok, distance) for t in tasks}


def both_seeds(*tasks):
    return {(t.id, s) for t in tasks for s in t.eval_seeds[:verifier_stage.K]}


def verifier_payload(world, status):
    return [p for s, st, p in world.events() if s == "gate.verifier" and st == status][0]


def test_violating_run_counts_as_minus_target():
    runs = {"a": [{"success": True, "distance_m": 2.0, "target_m": 5.0, "sanity": {"pass": True}},
                  {"success": True, "distance_m": 9.0, "target_m": 5.0, "sanity": {"pass": False}}],
            "b": [{"success": False, "distance_m": 1.0, "target_m": 3.0}]}  # stored before sanity
    s = verifier_stage.score(runs)
    assert s["mean_distance_m"] == round((2.0 - 5.0 + 1.0) / 3, 4)
    assert s["violations"] == 1 and s["n"] == 3


def test_halving_parent_violations_with_distance_gain_is_accepted(world, adb):
    ts = world.tasks
    world.parent_runs(outcome(ts), violations=both_seeds(ts[0], ts[1]))  # 4 of 12
    world.outcome = outcome(ts)
    world.violate = both_seeds(ts[0])  # 2 of 12
    res = world.gate()

    assert res["verdict"] == "accepted" and res["reason"] is None and res["frames_id"] is None
    cand = round((-2 * ts[0].target_m + 10 * 1.5) / 12, 4)
    par = round((-2 * ts[0].target_m - 2 * ts[1].target_m + 8 * 1.5) / 12, 4)
    assert res["train"]["mean_distance_m"] == res["train_mean_distance_m"] == cand
    assert res["parent_train_mean_distance_m"] == par
    assert verifier_payload(world, "pass")["violations"] == {"candidate": 2, "parent": 4}


def test_one_extra_violation_is_rejected_with_torque_reason_and_frames(world, adb):
    ts = world.tasks
    world.violation = bounds.torque_reason("hip_2", 1.9, 2.1)
    world.parent_runs(outcome(ts), violations=both_seeds(ts[1]))  # 2 of 12
    world.outcome = outcome(ts, ok=True, distance=50.0)  # a big distance gain does not help
    world.violate = both_seeds(ts[2]) | {(ts[3].id, ts[3].eval_seeds[0])}  # 3 of 12
    res = world.gate()

    assert res["verdict"] == "rejected" and res["reason"] == world.violation
    assert res["stages"]["gate.verifier"] == "fail"
    frames = adb.frames.find_one({"_id": res["frames_id"]})
    run = adb.runs.find_one({"version_id": world.cand_id, "task_id": ts[2].id,
                             "seed": ts[2].eval_seeds[0]})  # first violation in task order
    assert (frames["kind"], frames["run_id"], frames["violation_frame"]) == \
        ("rejected", str(run["_id"]), 1)
    assert adb.runs.count_documents({"version_id": world.cand_id}) == 12
    fail = verifier_payload(world, "fail")
    assert fail["reason"] == world.violation and fail["frames_id"] == res["frames_id"]
    assert fail["violation_frame"] == 1
    assert fail["violations"] == {"candidate": 3, "parent": 2}
