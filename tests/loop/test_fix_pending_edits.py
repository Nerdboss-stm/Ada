"""scripts.fix_pending_edits: only null-verdict edits in closed rounds change; stubs keep ids unique."""

import random
import uuid
from datetime import datetime, timezone

import pytest

from core.contracts import Harness, Version
from core.db import ADA_TEST, db
from scripts.fix_pending_edits import REASON, fix

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


@pytest.fixture
def world():
    adb = db(ADA_TEST)
    tag = uuid.uuid4().hex[:8]
    base = 8_000_000_000 + random.randrange(1_000_000) * 10
    vid = {k: f"v{base + i}" for i, k in enumerate(("head", "closed", "open", "done", "taken"))}
    rid = {k: f"t-a13-{tag}-{k}" for k in ("closed", "open")}
    head = Version(_id=vid["head"], parent="v0", status="accepted", created_at=NOW,
                   harness=Harness(rules=["Keep the torso level."], tools=["read_task", "submit_gait"],
                                   model_per_step={"agent": "agent_v0"}))
    adb.versions.insert_one(head.model_dump(by_alias=True))
    adb.versions.insert_one({**head.model_dump(by_alias=True), "_id": vid["taken"],
                             "parent": vid["head"], "status": "rejected"})
    adb.rounds.insert_many([
        {"_id": rid["closed"], "status": "closed", "opened_at": NOW, "closed_at": NOW,
         "budget_usd": 3.0, "budget_s": 600},
        {"_id": rid["open"], "status": "open", "opened_at": NOW, "budget_usd": 3.0, "budget_s": 600},
    ])

    def edit(eid, round_id, to_version, verdict=None):
        return {"_id": eid, "round_id": round_id, "from_version": vid["head"],
                "to_version": to_version, "origin": "model", "primitive": "tools", "op": "add",
                "path": "", "old": None, "new": "preview_run", "verdict": verdict,
                "created_at": NOW}

    eids = {k: f"{rid['closed']}.{k}" for k in ("pending", "done", "taken")}
    eids["open"] = f"{rid['open']}.pending"
    adb.edits.insert_many([
        edit(eids["pending"], rid["closed"], vid["closed"]),
        edit(eids["done"], rid["closed"], vid["done"], verdict="accepted"),
        edit(eids["taken"], rid["closed"], vid["taken"]),
        edit(eids["open"], rid["open"], vid["open"]),
    ])
    yield vid, eids, adb, list(rid.values())

    adb.edits.delete_many({"_id": {"$in": list(eids.values())}})
    adb.rounds.delete_many({"_id": {"$in": list(rid.values())}})
    adb.versions.delete_many({"_id": {"$in": list(vid.values())}})


def by_edit(rows):
    return {r["edit_id"]: r for r in rows}


def test_only_closed_round_edits_change(world):
    vid, eids, adb, rids = world
    rows = by_edit(fix(ADA_TEST, round_ids=rids))

    assert set(rows) == {eids["pending"], eids["taken"]}
    for k in ("pending", "taken"):
        e = adb.edits.find_one({"_id": eids[k]})
        assert (e["verdict"], e["reason"]) == ("rejected", REASON)
    assert adb.edits.find_one({"_id": eids["open"]})["verdict"] is None
    assert adb.edits.find_one({"_id": eids["done"]})["verdict"] == "accepted"
    assert adb.edits.find_one({"_id": eids["done"]}).get("reason") is None

    assert rows[eids["pending"]]["stub"] == "written" and rows[eids["taken"]]["stub"] == "exists"
    stub = Version.model_validate(adb.versions.find_one({"_id": vid["closed"]}))
    head = Version.model_validate(adb.versions.find_one({"_id": vid["head"]}))
    assert (stub.status, stub.parent, stub.metrics) == ("rejected", vid["head"], None)
    assert stub.harness == head.harness
    assert adb.versions.find_one({"_id": vid["open"]}) is None  # open round: no stub

    assert by_edit(fix(ADA_TEST, round_ids=rids)) == {}  # idempotent


def test_dry_run_writes_nothing(world):
    vid, eids, adb, rids = world
    rows = by_edit(fix(ADA_TEST, dry_run=True, round_ids=rids))
    assert set(rows) == {eids["pending"], eids["taken"]}
    assert adb.edits.find_one({"_id": eids["pending"]})["verdict"] is None
    assert adb.versions.find_one({"_id": vid["closed"]}) is None
