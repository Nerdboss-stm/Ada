import pytest

from core.contracts import CARD_STAGES


def half(tasks, distance=1.5):
    """First three gate tasks succeed on every seed, the rest fail: reliability 0.5."""
    return {t.id: (i < 3, distance) for i, t in enumerate(tasks)}


def assert_precheck_reject(world, adb, res):
    assert res["verdict"] == "rejected"
    assert res["stages"] == {"gate.verifier": "skipped", "gate.gpa": "skipped",
                             "gate.meta": "skipped", "gate.constraints": "fail"}
    assert world.episodes == [] and world.llm_calls == []
    assert res["train"]["n"] == 0 and res["frames_id"] is None
    assert adb.runs.count_documents({"version_id": world.cand_id}) == 0
    assert [(s, st) for s, st, _ in world.events()] == [
        ("gate.verifier", "info"), ("gate.gpa", "info"), ("gate.meta", "info"),
        ("gate.constraints", "start"), ("gate.constraints", "fail")]
    assert all(p == {"skipped": True} for _, st, p in world.events() if st == "info")


def test_tool_outside_whitelist_rejects_at_constraints(world, adb):
    world.parent_runs(half(world.tasks))
    res = world.gate(tools=["read_task", "submit_gait", "edit_physics"])
    assert_precheck_reject(world, adb, res)
    assert res["reason"] == "tools outside the whitelist: edit_physics"
    assert res["parent_train_reliability"] == 0.5
    assert world.events()[-1][2]["reason"] == res["reason"]


def test_changed_context_policy_rejects_at_constraints(world, adb):
    """A16: the agent does not read context_policy, so a candidate may not change it."""
    world.parent_runs(half(world.tasks))
    res = world.gate(context_policy={"past_attempts": 2})
    assert_precheck_reject(world, adb, res)
    assert res["reason"] == "context policy is not implemented"
    assert world.events()[-1][2]["reason"] == res["reason"]


@pytest.mark.parametrize("role", ["frontier", "agent_v0.alt"])  # A14: agent_v0 is the only cheap role
def test_non_cheap_model_rejects_at_constraints(world, adb, role):
    world.parent_runs(half(world.tasks))
    res = world.gate(model_per_step={"agent": role})
    assert_precheck_reject(world, adb, res)
    assert res["reason"] == "model outside the cheap tier"


def test_power_5_rejects_at_verifier_with_frames(world, adb):
    world.parent_runs(half(world.tasks))
    world.power = 5.0
    res = world.gate()

    assert res["verdict"] == "rejected" and res["reason"] == world.violation
    # stages 2-4 still run after a physics failure (stubs pass here)
    assert res["stages"] == {"gate.verifier": "fail", "gate.gpa": "pass",
                             "gate.meta": "pass", "gate.constraints": "pass"}
    assert [(s, st) for s, st, _ in world.events()] == [
        (s, st) for s in CARD_STAGES for st in ("start", "fail" if s == "gate.verifier" else "pass")]
    # all six episodes roll in parallel; every run is kept, the frames are the first violation's
    assert sorted(world.episodes) == sorted(t.id for t in world.tasks)
    assert res["train"]["n"] == 12
    frames = adb.frames.find_one({"_id": res["frames_id"]})
    assert (frames["kind"], frames["violation_frame"], frames["version_id"]) == \
        ("rejected", 1, world.cand_id)
    first = world.tasks[0]
    run = adb.runs.find_one({"version_id": world.cand_id, "task_id": first.id,
                             "seed": first.eval_seeds[0]})
    assert frames["run_id"] == str(run["_id"]) and run["sanity"]["violation"] == world.violation
    fail = [p for s, st, p in world.events() if st == "fail"]
    assert fail[0]["reason"] == world.violation and fail[0]["frames_id"] == res["frames_id"]


def test_no_improvement_rejects_with_both_numbers(world, adb):
    world.parent_runs(half(world.tasks, distance=1.5))
    world.outcome = half(world.tasks, distance=1.55)  # same reliability, +0.05 m
    res = world.gate()

    assert res["verdict"] == "rejected"
    assert res["reason"] == ("no strict improvement: train reliability 0.5000 vs parent 0.5000, "
                             "mean distance 1.55 m vs parent 1.50 m")
    assert res["stages"] == {"gate.verifier": "fail", "gate.gpa": "pass",
                             "gate.meta": "pass", "gate.constraints": "pass"}
    assert res["frames_id"] is None
    assert res["train"]["n"] == 12 and len(world.episodes) == 6


@pytest.mark.parametrize("outcome,rel", [
    (lambda ts: {t.id: (True, 1.5) for t in ts}, 1.0),   # higher reliability
    (lambda ts: half(ts, distance=1.6), 0.5),             # equal reliability, +0.1 m
])
def test_strict_improvement_accepts(world, adb, outcome, rel):
    world.parent_runs(half(world.tasks, distance=1.5))
    world.outcome = outcome(world.tasks)
    res = world.gate()

    assert res["verdict"] == "accepted" and res["reason"] is None
    assert res["stages"] == {s: "pass" for s in CARD_STAGES}
    assert res["train"]["reliability"] == rel and res["train"]["n"] == 12
    assert res["train"]["cost_per_run_usd"] == 0.002
    assert res["parent_train_reliability"] == 0.5 and world.llm_calls == []
    runs = list(adb.runs.find({"version_id": world.cand_id}))
    assert len(runs) == 12 and {r["split"] for r in runs} == {"train"}


def test_events_arrive_in_card_order(world, adb):
    world.parent_runs(half(world.tasks))
    world.outcome = {t.id: (True, 2.1) for t in world.tasks}
    world.gate()

    assert [(s, st) for s, st, _ in world.events()] == [
        (s, st) for s in CARD_STAGES for st in ("start", "pass")]
    docs = list(adb.events.find({"edit_id": world.edit_id}))
    assert {(d["round_id"], d["version_id"]) for d in docs} == {(world.round_id, world.cand_id)}
    payloads = {s: p for s, st, p in world.events() if st == "pass"}
    assert payloads["gate.gpa"] == payloads["gate.meta"] == {"stub": True}
    assert payloads["gate.verifier"]["predicted_delta"] == 0.1
    assert payloads["gate.verifier"]["actual_delta"] == 0.5
