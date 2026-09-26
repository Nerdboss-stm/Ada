import json

import pytest

from core.contracts import EVENT_PAYLOAD_MAX_BYTES
from core.db import ADA_TEST, db
from loop.sensor import sense
from loop.subset import GATE_TASKS
from sim.tasks import TRAIN, task_id

# ada_test is shared with other sessions: touch only documents with these ids.
TASKS = ("sensor-t-a", "sensor-t-b", "sensor-t-c", "sensor-t-d")
V1, V2 = "sensor-v1", "sensor-v2"
TARGET = 2.0
SPEED = "body speed exceeds physical bound; exploits the simulator"
POWER = "exceeds Ada rated motors"


@pytest.fixture
def adb():
    d = db(ADA_TEST)
    cleanup(d)
    d.tasks.insert_many([
        {"_id": t, "split": "train", "slope_deg": 0.0, "friction": 1.0, "target_m": TARGET,
         "eval_seeds": [], "practice_seeds": []}
        for t in TASKS
    ])
    yield d
    cleanup(d)


def cleanup(d):
    d.runs.delete_many({"task_id": {"$in": TASKS}})
    d.tasks.delete_many({"_id": {"$in": TASKS}})
    d.events.delete_many({"stage": "sensor", "version_id": {"$in": [V1, V2]}})


def run(task, seed, distance=2.5, fell=False, sanity_pass=True, violation=None,
        version=V1, split="train", trace="auto", cost=0.001):
    success = sanity_pass and not fell and distance >= TARGET
    return {
        "task_id": task, "seed": seed, "split": split, "version_id": version, "model_id": "m",
        "gait": {}, "distance_m": distance, "fell": fell,
        "sanity": {"pass": sanity_pass, "violation": violation, "violation_frame": None},
        "success": success, "cost_usd": cost, "tokens": 10,
        "trace_id": f"{task}-{seed}" if trace == "auto" else trace,
    }


def codes_by_name(result):
    return {c["code"]: c for c in result["failure_codes"]}


def test_each_code_and_priority(adb):
    adb.runs.insert_many([
        run("sensor-t-a", 0),                                                     # success
        run("sensor-t-a", 1, fell=True, sanity_pass=False, violation=POWER),       # sanity beats fell
        run("sensor-t-a", 2, distance=-1.0, fell=True),                            # fell beats backward
        run("sensor-t-b", 0, distance=-0.5),                                       # backward
        run("sensor-t-b", 1, distance=0.0),                                        # short (0 is not backward)
        run("sensor-t-b", 2, distance=1.99),                                       # short
        run("sensor-t-c", 0, distance=-3.0, sanity_pass=False, violation=SPEED),   # sanity beats backward
    ])
    r = sense(V1, db_name=ADA_TEST)
    codes = codes_by_name(r)
    assert [c["code"] for c in r["failure_codes"]] == ["sanity", "fell", "backward", "short"]
    assert {k: c["count"] for k, c in codes.items()} == {"sanity": 2, "fell": 1, "backward": 1, "short": 2}
    assert sum(c["count"] for c in r["failure_codes"]) == r["n"] - 1
    assert codes["sanity"]["violations"] == sorted([POWER, SPEED])
    assert "violations" not in codes["fell"]
    assert codes["fell"]["examples"] == ["sensor-t-a-2"]
    assert codes["short"]["examples"] == ["sensor-t-b-1", "sensor-t-b-2"]


def test_examples_capped_and_skip_missing_trace(adb):
    adb.runs.insert_many(
        [run("sensor-t-a", 0, fell=True, trace=None)] + [run("sensor-t-a", s, fell=True) for s in range(1, 6)]
    )
    ex = codes_by_name(sense(V1, db_name=ADA_TEST))["fell"]
    assert ex["count"] == 6
    assert ex["examples"] == ["sensor-t-a-1", "sensor-t-a-2", "sensor-t-a-3"]


def test_filters_version_and_split_and_totals(adb):
    adb.runs.insert_many([
        run("sensor-t-a", 0, cost=0.0012345),
        run("sensor-t-a", 1, distance=1.0, cost=0.002),
        run("sensor-t-a", 2, fell=True, version=V2),
        run("sensor-t-a", 3, fell=True, split="holdout"),
    ])
    r = sense(V1, db_name=ADA_TEST)
    assert r["n"] == 2
    assert r["success_rate"] == 0.5
    assert r["mean_distance_m"] == 1.75
    assert r["cost_usd"] == round(0.0032345, 6)
    assert [c["code"] for c in r["failure_codes"]] == ["short"]


def test_short_uses_task_target(adb):
    adb.tasks.update_one({"_id": "sensor-t-a"}, {"$set": {"target_m": 3.0}})
    doc = run("sensor-t-a", 0, distance=2.5)
    doc["success"] = False  # the verifier judged it against 3.0 m
    adb.runs.insert_one(doc)
    assert codes_by_name(sense(V1, db_name=ADA_TEST))["short"]["count"] == 1


def test_worst_tasks_and_per_task(adb):
    adb.runs.insert_many([
        run("sensor-t-a", 0), run("sensor-t-a", 1),                             # 1.0
        run("sensor-t-b", 0, distance=1.0), run("sensor-t-b", 1),               # 0.5
        run("sensor-t-c", 0, distance=0.5), run("sensor-t-c", 1, distance=1.5),  # 0.0, mean 1.0
        run("sensor-t-d", 0, distance=0.2), run("sensor-t-d", 1, distance=0.4),  # 0.0, mean 0.3
    ])
    r = sense(V1, db_name=ADA_TEST)
    assert [w["task_id"] for w in r["worst_tasks"]] == ["sensor-t-d", "sensor-t-c", "sensor-t-b"]
    assert r["worst_tasks"][0] == {"task_id": "sensor-t-d", "success_rate": 0.0, "mean_distance_m": 0.3}
    assert {p["task_id"]: p["success_rate"] for p in r["per_task"]} == {
        "sensor-t-a": 1.0, "sensor-t-b": 0.5, "sensor-t-c": 0.0, "sensor-t-d": 0.0,
    }


def test_one_sensor_event_under_2kb_worst_case(adb):
    long_violations = [f"violation {i} " + "x" * 300 for i in range(6)]
    docs = []
    for i in range(40):
        kw = [
            {"sanity_pass": False, "violation": long_violations[(i // 4) % 6]},
            {"fell": True},
            {"distance": -1.0},
            {"distance": 0.5},
        ][i % 4]
        docs.append(run(f"sensor-t-{'abcd'[i % 4]}", i, trace="f" * 64 + str(i), **kw))
    adb.runs.insert_many(docs)
    r = sense(V1, db_name=ADA_TEST, round_id="r7")

    events = list(adb.events.find({"stage": "sensor", "version_id": V1}))
    assert len(events) == 1
    ev = events[0]
    assert (ev["stage"], ev["status"], ev["version_id"], ev["round_id"]) == ("sensor", "info", V1, "r7")
    size = len(json.dumps(ev["payload"], separators=(",", ":"), default=str).encode())
    assert size < EVENT_PAYLOAD_MAX_BYTES
    assert "per_task" not in ev["payload"]
    assert len(codes_by_name(r)["sanity"]["violations"]) == 6  # full reading keeps them all
    assert all(len(c["examples"]) == 3 for c in ev["payload"]["failure_codes"])
    sent = codes_by_name(ev["payload"])["sanity"]["violations"]
    assert len(sent) == 3 and all(len(v) <= 80 for v in sent)


def test_run_contradicting_verifier_raises(adb):
    doc = run("sensor-t-a", 0, distance=2.5)
    doc["success"] = False  # target is 2.0, so sim.verifier would have said success
    adb.runs.insert_one(doc)
    with pytest.raises(ValueError, match="disagree with sim.verifier"):
        sense(V1, db_name=ADA_TEST)


def test_gate_tasks_cover_every_slope_and_friction():
    train_ids = {task_id("train", s, f): (s, f) for s, f in TRAIN}
    assert len(GATE_TASKS) == 6 == len(set(GATE_TASKS))
    assert set(GATE_TASKS) <= set(train_ids)
    covered = [train_ids[t] for t in GATE_TASKS]
    assert {s for s, _ in covered} == {s for s, _ in TRAIN}
    assert {f for _, f in covered} == {f for _, f in TRAIN}
