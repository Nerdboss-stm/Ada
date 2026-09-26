"""B9: scripts/capture_swap.py and scripts/scoreboard.py on ada_test.

The swap uses the real verifier and recorder with a fake model; every document is tagged
t-b9-{tag} and teardown deletes only those.
"""

import json
import types
import uuid
from datetime import datetime, timezone

import mujoco
import pytest
from bson import ObjectId

from core import llm
from core.contracts import ChatResult, Edit, Metrics, Run, Sanity, Swap, Version
from core.db import ADA_TEST, db
from core.events import emit
from harness import run as HR
from harness.agent import AGENT_STEP, V0_HARNESS
from harness.agent import thread_id as CS_thread
from harness.guardrails import VERIFIER_PATH, file_sha256, seed_guardrails
from scripts import capture_swap as CS
from scripts import scoreboard as SB
from sim.gait import neutral_offsets
from sim.tasks import build_tasks

HOLDOUT = sorted((t for t in build_tasks() if t.split == "holdout"), key=lambda t: t.id)[:2]
CALL_COST = 0.001
SWAP_FIELDS = {
    "_id", "captured_at", "task_id", "k", "left_version", "right_version", "left_holdout",
    "right_holdout", "left_frames_id", "right_frames_id", "model_id", "right_model_id",
    "verifier_sha", "mujoco_version", "manifest_version", "harness_diff",
    "left_mean_distance_m", "right_mean_distance_m", "left_cost_per_run_usd",
    "right_cost_per_run_usd", "n", "fresh_calls", "model_calls", "left_violations", "right_violations",
}
FRESH_SEEN: list = []  # the fresh flag of every fake model call


def valid_gait() -> dict:
    return {
        "frequency_hz": 1.0, "power": 1.0, "kp": 5.0, "kd": 0.5, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": v, "phase": 0.0}
                   for n, v in neutral_offsets().items()],
    }


def fake_model(role, messages, tools=None, **params):
    """read_task on the first turn, a valid gait on the next; model_id names the role."""
    FRESH_SEEN.append(params.get("fresh"))
    turn = sum(m["role"] == "assistant" for m in messages)
    name, args = ("read_task", {}) if turn == 0 else ("submit_gait", valid_gait())
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{turn}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}
    return ChatResult(role=role, model_id=f"test/{role}", prompt_hash=f"h{turn}", message=msg,
                      usage={"total_tokens": 100}, cost_usd=CALL_COST, cached=False)


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(llm, "cached_chat", fake_model)
    FRESH_SEEN.clear()
    seed_guardrails(ADA_TEST)
    d = db(ADA_TEST)
    tag = f"t-b9-{uuid.uuid4().hex[:8]}"
    own = {"$regex": f"^{tag}"}
    made = {"swaps": [], "scoreboard": []}

    def version(suffix, harness=V0_HARNESS, cost=None):
        vid = f"{tag}-{suffix}"
        metrics = None if cost is None else Metrics(
            train_reliability=0.0, holdout_reliability_80=0.0, mean_distance_m=0.0,
            cost_per_run_usd=cost, n=0)
        d.versions.insert_one(Version(_id=vid, status="accepted", harness=harness, metrics=metrics,
                                      created_at=datetime.now(timezone.utc)).model_dump(by_alias=True))
        return vid

    yield types.SimpleNamespace(db=d, tag=tag, made=made, version=version)

    d.swaps.delete_many({"_id": {"$in": made["swaps"]}})
    d.scoreboard.delete_many({"_id": {"$in": made["scoreboard"]}})
    for coll in ("versions", "edits"):
        d[coll].delete_many({"_id": own})
    for coll in ("runs", "frames"):
        d[coll].delete_many({"version_id": own})
    d.events.delete_many({"$or": [{"version_id": own}, {"round_id": own}]})
    d.traces.delete_many({"trace_id": own})
    d.checkpoints.delete_many({"thread_id": own})
    d.checkpoint_writes.delete_many({"thread_id": own})


def capture(w, left, right):
    swap = CS.capture_swap(right, left=left, tasks=HOLDOUT, db_name=ADA_TEST, ckpt_db=ADA_TEST)
    w.made["swaps"].append(swap.id)
    return swap


def test_swap_document_fields_runs_and_frames(world):
    left = world.version("l")
    rule = "Keep the torso level on slopes."
    right = world.version("r", V0_HARNESS.model_copy(deep=True, update={"rules": [*V0_HARNESS.rules, rule]}))

    before = {v: world.db.versions.find_one({"_id": v}) for v in (left, right)}
    swap = capture(world, left, right)
    doc = world.db.swaps.find_one({"_id": swap.id})
    assert set(doc) == SWAP_FIELDS
    assert Swap.model_validate(doc) == swap

    assert (swap.task_id, swap.k) == (HOLDOUT[0].id, 3)
    assert (swap.left_version, swap.right_version) == (left, right)
    assert swap.n == len(HOLDOUT) * 3
    for vid, passed in ((left, swap.left_holdout), (right, swap.right_holdout)):
        runs = list(world.db.runs.find({"version_id": vid, "split": "swap"}))
        assert len(runs) == swap.n
        assert world.db.runs.count_documents({"version_id": vid, "split": "holdout"}) == 0
        assert world.db.versions.find_one({"_id": vid}) == before[vid]  # metrics untouched
        assert passed == CS.holdout_passed(runs) and 0 <= passed <= len(HOLDOUT)
    assert swap.left_holdout == swap.right_holdout  # same gait from the same fake model
    assert swap.left_cost_per_run_usd == pytest.approx(2 * CALL_COST)  # per episode

    for vid, fid in ((left, swap.left_frames_id), (right, swap.right_frames_id)):
        frames = world.db.frames.find_one({"_id": fid})
        assert frames["kind"] == "showcase" and frames["version_id"] == vid
        run = world.db.runs.find_one({"_id": ObjectId(frames["run_id"])})
        assert (run["task_id"], run["seed"], run["split"]) == (HOLDOUT[0].id, HOLDOUT[0].eval_seeds[0], "swap")

    assert swap.model_id == swap.right_model_id == "test/agent_v0"
    assert all(line[:1] in "+-" for line in swap.harness_diff)
    assert [line for line in swap.harness_diff if line.startswith("+") and rule in line]
    assert not [line for line in swap.harness_diff if line.startswith("model:")]
    assert swap.verifier_sha == file_sha256(VERIFIER_PATH)
    assert swap.mujoco_version == mujoco.__version__
    assert swap.manifest_version == CS.record.manifest_version()
    assert swap.model_calls == 2 * len(HOLDOUT) * 2  # two sides, two turns per episode
    assert swap.fresh_calls and FRESH_SEEN == [True] * swap.model_calls
    assert (swap.left_violations, swap.right_violations) == (0, 0)


def test_swap_violations_per_side_and_penalized_means(world, monkeypatch):
    """[B13, B14] a physics-violating run counts on its side and as min(distance_m, -target_m)
    in that side's mean."""
    left, right = world.version("l"), world.version("r")
    real = HR._evaluate
    bad = HOLDOUT[1]  # not the recorded task, so the showcase re-record still reproduces

    def evaluate(ep, task, seed, record):
        result = real(ep, task, seed, record)
        if task.id == bad.id and seed == bad.eval_seeds[0]:
            result = {**result, "success": False,
                      "sanity": {"pass": False, "violation": "joint torque", "violation_frame": 3}}
        return result

    monkeypatch.setattr(HR, "_evaluate", evaluate)
    swap = capture(world, left, right)

    assert (swap.left_violations, swap.right_violations) == (1, 1)
    for vid, mean_m in ((left, swap.left_mean_distance_m), (right, swap.right_mean_distance_m)):
        runs = list(world.db.runs.find({"version_id": vid, "split": "swap"}))
        assert CS.violations(runs) == 1
        targets = {t.id: t.target_m for t in HOLDOUT}
        scored = [min(r["distance_m"], -targets[r["task_id"]]) if not r["sanity"]["pass"] else r["distance_m"]
                  for r in runs]
        assert mean_m == round(sum(scored) / len(scored), 4)


def test_swap_leaves_frozen_holdout_runs_metrics_and_traces_alone(world):
    left = world.version("l", cost=0.0002)
    right = world.version("r", cost=0.0003)
    task = HOLDOUT[0]
    frozen = Run(task_id=task.id, seed=task.eval_seeds[0], split="holdout", version_id=left, model_id="m",
                 gait={}, distance_m=1.23, fell=False, sanity=Sanity.model_validate({"pass": True}),
                 success=True, cost_usd=0.1, tokens=7, trace_id=CS_thread(left, task))
    world.db.runs.insert_one(frozen.model_dump(by_alias=True, exclude={"id"}))
    world.db.traces.insert_one({"trace_id": CS_thread(left, task), "raw_steps": [{"node": "baseline"}]})
    before = {v: world.db.versions.find_one({"_id": v}) for v in (left, right)}

    swap = capture(world, left, right)

    held = list(world.db.runs.find({"version_id": left, "split": "holdout"}, {"_id": 0}))
    assert held == [frozen.model_dump(by_alias=True, exclude={"id"})]
    assert world.db.traces.find_one({"trace_id": CS_thread(left, task)})["raw_steps"] == [{"node": "baseline"}]
    for vid in (left, right):
        assert world.db.versions.find_one({"_id": vid}) == before[vid]
        swap_runs = list(world.db.runs.find({"version_id": vid, "split": "swap"}))
        assert len(swap_runs) == swap.n and all(r["trace_id"].endswith("-swap") for r in swap_runs)
    assert swap.left_holdout == CS.holdout_passed(list(world.db.runs.find({"version_id": left, "split": "swap"})))
    assert swap.left_cost_per_run_usd == pytest.approx(2 * CALL_COST)  # from the swap episodes, not v0's 0.0002


def test_different_model_goes_first_in_harness_diff(world):
    left = world.version("l")
    right = world.version("r", V0_HARNESS.model_copy(deep=True, update={"model_per_step": {AGENT_STEP: "frontier"}}))

    swap = capture(world, left, right)
    assert (swap.model_id, swap.right_model_id) == ("test/agent_v0", "test/frontier")
    assert swap.harness_diff[0] == (
        "model: test/agent_v0 → test/frontier (chosen by the harness; neither was trained)")
    assert any('"frontier"' in line for line in swap.harness_diff[1:] if line.startswith("+"))


def test_identical_harnesses_have_no_diff():
    assert CS.harness_diff(V0_HARNESS, V0_HARNESS, "m", "m") == []


def test_unknown_task_refuses_before_any_episode(world):
    left, right = world.version("l"), world.version("r")
    with pytest.raises(LookupError):
        CS.capture_swap(right, left=left, task_id="nope", tasks=HOLDOUT, db_name=ADA_TEST, ckpt_db=ADA_TEST)
    assert world.db.runs.count_documents({"version_id": {"$in": [left, right]}}) == 0


def test_scoreboard_counts_and_costs(world):
    t = world.tag
    rid = f"{t}-round"
    now = datetime.now(timezone.utc)

    def edit(n, origin, verdict, reason, to_version):
        e = Edit(_id=f"{t}-e{n}", round_id=rid, from_version="v0", to_version=to_version, origin=origin,
                 primitive="rule", verdict=verdict, reason=reason, created_at=now)
        world.db.edits.insert_one(e.model_dump(by_alias=True))
        return e.id

    e1 = edit(1, "model", "rejected", "hip_2 peak torque 1.90x rated torque (150) at 2.10 s", f"{t}-c1")
    e2 = edit(2, "model", "rejected", "power 5.0 exceeds Ada's rated motors (1.0)", f"{t}-c2")
    e3 = edit(3, "probe", "rejected", "no strict improvement: train reliability 0.5 vs parent 0.5", f"{t}-c3")
    e4 = edit(4, "model", "accepted", None, f"{t}-c4")
    e5 = edit(5, "cli", "rejected", "verification: still fails: power 5.0 exceeds Ada's rated motors", None)
    e6 = edit(6, "model", "rejected", "submitted a gait on 1/3 tasks vs parent 3/3", f"{t}-c6")
    e7 = edit(7, "model", "rejected", "torso height rose above 0.9 m at 1.20 s", f"{t}-c7")

    def ev(stage, status, edit_id=None, **payload):
        emit(stage, status, payload, round_id=rid, edit_id=edit_id, db_name=ADA_TEST)

    ev("controller", "info", count=4, cost_usd=0.01)
    ev("gate.verifier", "fail", e1, reason="hip_2 peak torque 1.90x rated torque (150) at 2.10 s")
    ev("gate.gpa", "start", e1)
    ev("gate.gpa", "pass", e1, cost_usd=0.002)
    ev("gate.meta", "pass", e1, cost_usd=0.004)
    ev("gate.verifier", "fail", e2, reason="power 5.0 exceeds Ada's rated motors (1.0)")
    ev("gate.gpa", "fail", e2, cost_usd=0.003)
    ev("gate.verifier", "fail", e3, reason="no strict improvement: train reliability 0.5 vs parent 0.5")
    ev("gate.gpa", "pass", e3)  # not physics: excluded even though the judge passed it
    ev("gate.verifier", "fail", e6, reason="submitted a gait on 1/3 tasks vs parent 3/3")
    ev("gate.gpa", "pass", e6)
    ev("gate.verifier", "fail", e7, reason="torso height rose above 0.9 m at 1.20 s")
    ev("gate.gpa", "fail", e7)
    ev("gate.verifier", "pass", e4)
    ev("gate.gpa", "pass", e4)
    ev("gate.verifier", "fail", e5, reason="power 5.0 exceeds Ada's rated motors")  # ./verify row: excluded
    ev("gate.gpa", "pass", e5)

    def run(vid, split, cost):
        r = Run(task_id="t", seed=1, split=split, version_id=vid, model_id="m", gait={}, distance_m=0.0,
                fell=False, sanity=Sanity.model_validate({"pass": True}), success=False, cost_usd=cost, tokens=0)
        world.db.runs.insert_one(r.model_dump(by_alias=True, exclude={"id"}))

    run(f"{t}-c1", "train", 0.0005)
    run(f"{t}-c1", "train", 0.0005)
    run(f"{t}-c4", "train", 0.001)
    run(f"{t}-c4", "holdout", 0.5)  # scoring an accepted version is not rewrite cost
    run(f"{t}-other", "train", 9.0)  # not a proposal's candidate

    v0, fr, best = world.version("v0", cost=0.0002), world.version("fr", cost=0.02), world.version("best", cost=0.0003)
    board = SB.build_scoreboard(best=best, v0=v0, frontier=fr, round_ids=[rid], db_name=ADA_TEST)
    world.made["scoreboard"].append(board.id)

    assert board.proposals_total == 6
    assert board.overrated_attempts == 2
    assert board.physics_rejected == 3  # e1, e2, e7; not e3 (improvement) or e6 (gait count)
    assert board.physics_rejected_judge_passed == 1  # e1
    assert board.rewrite_cost_usd == pytest.approx(0.01 + 0.002 + 0.004 + 0.003 + 0.0005 * 2 + 0.001)
    assert board.cost_per_gait.model_dump() == {"v0": 0.0002, "frontier": 0.02, "best": 0.0003}
    assert board.best_version == best
    assert world.db.scoreboard.find_one({"_id": board.id})["overrated_attempts"] == 2
    assert world.db.edits.find_one({"_id": e3})["origin"] == "probe"  # counted as a proposal


def test_scoreboard_missing_version_cost_is_none(world):
    board = SB.build_scoreboard(best=f"{world.tag}-absent", v0=f"{world.tag}-absent",
                                frontier=f"{world.tag}-absent", round_ids=[f"{world.tag}-empty"], db_name=ADA_TEST)
    world.made["scoreboard"].append(board.id)
    assert board.cost_per_gait.model_dump() == {"v0": None, "frontier": None, "best": None}
    assert (board.proposals_total, board.rewrite_cost_usd, board.physics_rejected) == (0, 0.0, 0)
