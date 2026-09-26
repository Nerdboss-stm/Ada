import itertools
import random
import sys
import types
import uuid
from datetime import datetime, timezone

import pytest
from pymongo.errors import DuplicateKeyError

import harness.run as harness_run
import loop.actuator as actuator
import loop.controller as controller
from core.contracts import Harness, Round, Version
from core.db import ADA_TEST, db
from loop.edits import EditProposal

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
REASON = "exceeds Ada rated motors"


def prop(**over):
    base = {"primitive": "engine", "op": "set", "path": "temperature", "old": 0, "new": 0.2,
            "predicted_delta": 0.05, "rationale": "Some variety may help the agent escape a bad gait.",
            "evidence_trace_ids": ["tr-fell-1", "tr-fell-2"]}
    return EditProposal.model_validate({**base, **over})


P_TEMP = prop()
P_STALE = prop(new=0.4)  # still expects temperature 0
P_RULE = prop(primitive="rules", op="add", path="", old=None, new="Lift one leg at a time.",
              evidence_trace_ids=["tr-short-1"])


def gate_result(verdict="accepted", rel=0.6, parent=0.5, reason=None, frames_id=None,
                cost=0.001, n=12):
    stages = {s: "pass" for s in ("gate.verifier", "gate.gpa", "gate.meta", "gate.constraints")}
    if verdict == "rejected":
        stages = {"gate.verifier": "fail", "gate.gpa": "skipped", "gate.meta": "skipped",
                  "gate.constraints": "skipped"}
    return {"verdict": verdict, "reason": reason, "stages": stages,
            "train": {"reliability": rel, "mean_distance_m": 1.9, "cost_per_run_usd": cost, "n": n},
            "parent_train_reliability": parent, "frames_id": frames_id}


@pytest.fixture
def adb():
    return db(ADA_TEST)


@pytest.fixture
def world(monkeypatch, adb):
    """Own head version, own round ids; controller, gate and harness.run patched."""
    tag = uuid.uuid4().hex[:8]

    class W:
        head_id = f"v{9_000_000_000 + random.randrange(1_000_000)}"
        round_ids: list[str] = []
        frames_ids: list[str] = []
        proposals: list[EditProposal] = []
        results: list[dict] = []
        propose_calls: list[dict] = []
        gate_calls: list[dict] = []
        split_calls: list[tuple] = []

    counter = itertools.count()

    def new_round_id():
        rid = f"t-a7-{tag}-{next(counter)}"
        W.round_ids.append(rid)
        return rid

    def propose(head, summary, recent, round_id=None, *, db_name):
        W.propose_calls.append({"head": head.id, "summary": summary, "round_id": round_id})
        return list(W.proposals)

    def run_gate(*, parent, candidate, edit, round_id, db_name):
        W.gate_calls.append({
            "parent": parent, "candidate": candidate, "edit": edit, "round_id": round_id,
            "edit_doc": adb.edits.find_one({"_id": edit.id}),
            "candidate_doc": adb.versions.find_one({"_id": candidate.id}),
        })
        return W.results.pop(0)

    def run_split(version_id, split, k, *, harness=None, db_name):
        W.split_calls.append((version_id, split, k))

    gate = types.ModuleType("gate.pipeline")
    gate.run_gate = run_gate
    monkeypatch.setitem(sys.modules, "gate.pipeline", gate)
    monkeypatch.setattr(actuator, "new_round_id", new_round_id)
    monkeypatch.setattr(controller, "propose", propose)
    monkeypatch.setattr(harness_run, "run_split", run_split)

    head = Version(
        _id=W.head_id, parent="v0", status="accepted", created_at=NOW,
        harness=Harness(rules=["Keep the torso level."], tools=["read_task", "submit_gait"],
                        model_per_step={"agent": "agent_v0"},
                        engine={"temperature": 0, "max_attempts": 3}),
        metrics={"train_reliability": 0.5, "holdout_reliability_80": 0.4, "mean_distance_m": 1.5,
                 "cost_per_run_usd": 0.001, "n": 18},
    )
    adb.versions.insert_one(head.model_dump(by_alias=True))
    W.frame = lambda doc: (W.frames_ids.append(doc["_id"]), adb.frames.insert_one(doc))
    yield W

    rids = W.round_ids
    children = [e["to_version"] for e in adb.edits.find({"round_id": {"$in": rids}})]
    adb.versions.delete_many({"_id": {"$in": [W.head_id, *children]}})
    adb.edits.delete_many({"round_id": {"$in": rids}})
    adb.events.delete_many({"round_id": {"$in": rids}})
    adb.rounds.delete_many({"_id": {"$in": rids}})
    adb.frames.delete_many({"_id": {"$in": W.frames_ids}})


def round_doc(adb, rid):
    return adb.rounds.find_one({"_id": rid})


def actuator_events(adb, rid):
    return list(adb.events.find({"stage": "actuator", "round_id": rid}).sort("ts", 1))


def test_accept_advances_head(world, adb):
    world.proposals = [P_TEMP, P_RULE]
    world.results = [gate_result(rel=0.6, parent=0.5), gate_result(rel=0.7, parent=0.6)]
    r = actuator.run_round(world.head_id, db_name=ADA_TEST)

    c1, c2 = r["accepted"]
    assert r["head"] == c2 and r["start_head"] == world.head_id and r["rejected"] == []
    assert int(c1[1:]) > int(world.head_id[1:]) and int(c2[1:]) == int(c1[1:]) + 1
    assert world.propose_calls[0]["head"] == world.head_id
    assert "per_task" not in world.propose_calls[0]["summary"]

    g1, g2 = world.gate_calls
    assert g1["parent"].id == world.head_id and g2["parent"].id == c1
    assert g1["candidate"].status == g2["candidate"].status == "candidate"
    assert g1["candidate"].harness.engine["temperature"] == 0.2
    assert g2["candidate"].harness.engine["temperature"] == 0.2  # built on the new head
    assert g2["candidate"].harness.rules[-1] == "Lift one leg at a time."
    assert g1["edit_doc"]["verdict"] is None and g1["candidate_doc"] is None  # edit first

    v1, v2 = adb.versions.find_one({"_id": c1}), adb.versions.find_one({"_id": c2})
    assert (v1["status"], v1["parent"], v2["parent"]) == ("accepted", world.head_id, c1)
    assert v1["metrics"]["train_reliability"] == 0.6 and v1["metrics"]["n"] == 12
    assert world.split_calls == [(c1, "holdout", 3), (c1, "showcase", 1),
                                 (c2, "holdout", 3), (c2, "showcase", 1)]

    e1 = adb.edits.find_one({"to_version": c1})
    assert (e1["verdict"], e1["actual_delta"], e1["origin"]) == ("accepted", 0.1, "model")
    assert (e1["from_version"], e1["op"], e1["path"], e1["old"], e1["new"]) == \
        (world.head_id, "set", "temperature", 0, 0.2)
    assert e1["evidence_trace_ids"] == ["tr-fell-1", "tr-fell-2"]
    assert e1["rationale"] == P_TEMP.rationale and e1["predicted_delta"] == 0.05

    ev = actuator_events(adb, r["round_id"])
    assert [(e["status"], e["payload"]["event"]) for e in ev] == [
        ("start", "round_open"), ("pass", "decision"), ("pass", "decision"), ("info", "round_close")]
    assert ev[1]["edit_id"] == e1["_id"] and ev[1]["version_id"] == c1


def test_reject_keeps_head(world, adb):
    frames_id = f"t-a7-frames-{uuid.uuid4().hex[:8]}"
    world.frame({"_id": frames_id, "kind": "rejected", "violation_frame": 42})
    world.proposals = [P_TEMP, P_RULE]
    world.results = [gate_result("rejected", rel=0.3, parent=0.5, reason=REASON, frames_id=frames_id),
                     gate_result(rel=0.55, parent=0.5)]
    r = actuator.run_round(world.head_id, db_name=ADA_TEST)

    (bad,), (good,) = r["rejected"], r["accepted"]
    assert r["head"] == good
    assert world.gate_calls[1]["parent"].id == world.head_id
    assert world.gate_calls[1]["candidate"].harness.engine["temperature"] == 0  # reject not applied
    assert (bad, "holdout", 3) not in world.split_calls

    v = adb.versions.find_one({"_id": bad})
    assert (v["status"], v["parent"]) == ("rejected", world.head_id)
    e = adb.edits.find_one({"to_version": bad})
    assert (e["verdict"], e["reason"], e["frames_id"], e["violation_frame"]) == \
        ("rejected", REASON, frames_id, 42)
    assert e["actual_delta"] == -0.2
    ev = actuator_events(adb, r["round_id"])
    assert [e["status"] for e in ev] == ["start", "fail", "pass", "info"]
    assert ev[1]["payload"]["reason"] == REASON


def test_precheck_reject_stores_null_delta(world, adb):
    res = gate_result("rejected", rel=0.0, parent=0.5, reason="model outside the cheap tier",
                      cost=0.0, n=0)
    res["train"]["mean_distance_m"] = 0.0
    res["stages"] = {"gate.verifier": "skipped", "gate.gpa": "skipped", "gate.meta": "skipped",
                     "gate.constraints": "fail"}
    world.proposals = [P_TEMP]
    world.results = [res]
    r = actuator.run_round(world.head_id, db_name=ADA_TEST)

    (bad,) = r["rejected"]
    assert r["decisions"][0]["actual_delta"] is None
    e = adb.edits.find_one({"to_version": bad})
    assert (e["verdict"], e["reason"], e["actual_delta"]) == \
        ("rejected", "model outside the cheap tier", None)
    decision = actuator_events(adb, r["round_id"])[1]
    assert decision["status"] == "fail" and decision["payload"]["actual_delta"] is None


def test_stale_proposal_is_skipped(world, adb):
    world.proposals = [P_TEMP, P_STALE, P_RULE]
    world.results = [gate_result(), gate_result(rel=0.65, parent=0.6)]
    r = actuator.run_round(world.head_id, db_name=ADA_TEST)

    assert r["stale"] == 1 and len(r["accepted"]) == 2 and len(world.gate_calls) == 2
    assert [d["decision"] for d in r["decisions"]] == ["accepted", "stale", "accepted"]
    assert r["decisions"][1]["edit_id"] is None
    assert adb.edits.count_documents({"round_id": r["round_id"]}) == 2
    stale = [e for e in actuator_events(adb, r["round_id"]) if e["payload"].get("decision") == "stale"]
    assert len(stale) == 1 and stale[0]["status"] == "info"


def test_second_open_round_is_refused(world, adb):
    held = actuator.new_round_id()
    actuator.ensure_lock_index(ADA_TEST)
    adb.rounds.insert_one({"_id": held, "status": "open", "opened_at": NOW, "budget_usd": 3.0,
                           "budget_s": 600, "open_lock": True})
    with pytest.raises(DuplicateKeyError):
        adb.rounds.insert_one({"_id": f"{held}-x", "status": "open", "opened_at": NOW,
                               "budget_usd": 3.0, "budget_s": 600, "open_lock": True})
    with pytest.raises(actuator.RoundOpenError, match=held):
        actuator.run_round(world.head_id, db_name=ADA_TEST)

    assert world.propose_calls == [] and world.gate_calls == []
    refused = world.round_ids[-1]
    assert round_doc(adb, refused) is None and actuator_events(adb, refused) == []


def test_closing_unsets_open_lock(world, adb):
    world.proposals = []
    r1 = actuator.run_round(world.head_id, db_name=ADA_TEST)
    doc = round_doc(adb, r1["round_id"])
    assert "open_lock" not in doc
    assert (doc["status"], doc["stop_reason"]) == ("closed", "no proposals")
    assert doc["closed_at"] is not None
    Round.model_validate(doc)

    r2 = actuator.run_round(world.head_id, db_name=ADA_TEST)  # the lock was released
    assert "open_lock" not in round_doc(adb, r2["round_id"])


def test_error_still_closes_round(world, adb):
    world.proposals = [P_TEMP]
    world.results = [gate_result(verdict="maybe")]
    with pytest.raises(ValueError):
        actuator.run_round(world.head_id, db_name=ADA_TEST)
    doc = round_doc(adb, world.round_ids[-1])
    assert "open_lock" not in doc and doc["status"] == "closed"
    assert doc["stop_reason"].startswith("error: ValueError")


def test_budget_usd_stop_is_recorded(world, adb):
    world.proposals = [P_TEMP, P_RULE]
    world.results = [gate_result(cost=0.01, n=12)]
    r = actuator.run_round(world.head_id, db_name=ADA_TEST, budget_usd=0.05)

    assert len(world.gate_calls) == 1
    assert (r["stop_reason"], r["skipped"], r["spent_usd"]) == ("budget_usd", 1, 0.12)
    doc = round_doc(adb, r["round_id"])
    assert (doc["stop_reason"], doc["spent_usd"], doc["budget_usd"]) == ("budget_usd", 0.12, 0.05)
    close = actuator_events(adb, r["round_id"])[-1]
    assert close["payload"]["stop_reason"] == "budget_usd" and close["payload"]["skipped"] == 1


def test_budget_s_stop_is_recorded(world, adb, monkeypatch):
    ticks = iter([0.0])
    monkeypatch.setattr(actuator, "_clock", lambda: next(ticks, 700.0))
    world.proposals = [P_TEMP, P_RULE]
    r = actuator.run_round(world.head_id, db_name=ADA_TEST, budget_s=600)

    assert world.gate_calls == []
    assert (r["stop_reason"], r["skipped"]) == ("budget_s", 2)
    assert round_doc(adb, r["round_id"])["stop_reason"] == "budget_s"
