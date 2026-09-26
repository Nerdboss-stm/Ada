"""B8: ./verify on ada_test with the real verifier; each test deletes only its own documents."""

import uuid
from datetime import datetime, timezone

import pytest

from core.contracts import CARD_STAGES, Edit, Run, Sanity
from core.db import ADA_TEST, db
from scripts import verify_cli
from sim import model as M
from sim import record, verifier
from sim.gait import Gait
from sim.tasks import build_tasks

TASK = next(t for t in build_tasks() if t.split == "train" and t.slope_deg == 0 and t.friction == 1.0)
SEED = TASK.eval_seeds[0]


def gait(power, amplitude, frequency_hz):
    return Gait.model_validate({
        "frequency_hz": frequency_hz, "power": power, "kp": 20.0, "kd": 0.0, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": amplitude, "offset": 0.0, "phase": i * 0.785}
                   for i, n in enumerate(M.FORCE_JOINTS)],
    })


FLAIL = gait(5.0, 1.2, 3.0)
WALK = gait(1.0, 0.3, 1.0)


@pytest.fixture
def world():
    """Records runs/frames/edits under its own tag; teardown deletes exactly those."""
    adb = db(ADA_TEST)
    tag = uuid.uuid4().hex[:8]
    made = {"runs": [], "frames": [], "edits": []}

    def recorded(g, kind, violation=True):
        result = verifier.evaluate(g, TASK, SEED, record=True)
        run = Run(task_id=TASK.id, seed=SEED, split="train", version_id=f"t-b8-{tag}",
                  model_id="test", gait=g.model_dump(), distance_m=result["distance_m"],
                  fell=result["fell"], sanity=Sanity.model_validate(result["sanity"]),
                  success=result["success"], cost_usd=0.0, tokens=0)
        run_id = adb.runs.insert_one(run.model_dump(by_alias=True, exclude={"id"})).inserted_id
        made["runs"].append(run_id)
        frame = result["sanity"]["violation_frame"] if violation else None
        fid = record.write_frames(result["frames"], str(run_id), f"t-b8-{tag}", kind,
                                  violation_frame=frame, db_name=ADA_TEST)
        made["frames"].append(fid)
        return fid, result

    def rejected_edit():
        fid, result = recorded(FLAIL, "rejected")
        edit = Edit(_id=f"t-b8-{tag}.e0", round_id=f"t-b8-{tag}", from_version="v0",
                    to_version=f"t-b8-{tag}", origin="model", primitive="rule", verdict="rejected",
                    reason=result["sanity"]["violation"], frames_id=fid,
                    violation_frame=result["sanity"]["violation_frame"],
                    created_at=datetime.now(timezone.utc))
        adb.edits.insert_one(edit.model_dump(by_alias=True))
        made["edits"].append(edit.id)
        return edit.id

    class W:
        pass

    w = W()
    w.recorded, w.rejected_edit, w.adb, w.made = recorded, rejected_edit, adb, made
    yield w

    edit_ids = made["edits"] + [d["_id"] for d in adb.edits.find(
        {"_id": {"$regex": f"^t-b8-{tag}\\."}}, {"_id": 1})]
    adb.events.delete_many({"edit_id": {"$in": edit_ids}})
    adb.edits.delete_many({"_id": {"$in": edit_ids}})
    adb.frames.delete_many({"_id": {"$in": made["frames"]}})
    adb.runs.delete_many({"_id": {"$in": made["runs"]}})


def test_clip_power_flail_passes_sanity(world, capsys):
    edit_id = world.rejected_edit()
    rc = verify_cli.main(["--edit", edit_id, "--clip-power", "1.0"], db_name=ADA_TEST)
    out = capsys.readouterr().out
    assert rc == 0, out

    lines = out.splitlines()
    original = next(line for line in lines if line.startswith("original"))
    clipped = next(line for line in lines if line.startswith("clipped"))
    assert "5.00" in original and "FAIL" in original
    assert "1.00" in clipped and "pass" in clipped and "FAIL" not in clipped

    cli = list(world.adb.edits.find({"_id": {"$regex": f"^{edit_id}\\.cli\\."}}))
    assert len(cli) == 1
    doc = Edit.model_validate(cli[0])
    assert (doc.origin, doc.old, doc.new, doc.verdict) == ("cli", 5.0, 1.0, "accepted")
    assert doc.reason == "verification: same gait at power 1 passes sanity"
    assert doc.frames_id and doc.frames_id in world.made["frames"]

    events = list(world.adb.events.find({"edit_id": doc.id}).sort("ts", 1))
    cells = [(e["stage"], e["status"]) for e in events]
    assert cells[:2] == [("gate.verifier", "start"), ("gate.verifier", "pass")]
    assert {(s, "info") for s in CARD_STAGES[1:]} <= set(cells)
    verdict = events[1]["payload"]
    assert verdict["clipped"]["sanity"] == "pass" and verdict["original"]["sanity"] == "fail"


def test_determinism_matches_recorded_run(world, capsys):
    fid, _ = world.recorded(WALK, "showcase", violation=False)
    rc = verify_cli.main(["--determinism", fid], db_name=ADA_TEST)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "MATCH" in out and "MISMATCH" not in out


def test_determinism_reports_tampered_sha(world, capsys):
    fid, _ = world.recorded(WALK, "showcase", violation=False)
    world.adb.frames.update_one({"_id": fid}, {"$set": {"sha256": "0" * 64}})
    rc = verify_cli.main(["--determinism", fid], db_name=ADA_TEST)
    assert rc == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_unknown_edit_exits_2(world, capsys):
    assert verify_cli.main(["--edit", "t-b8-missing"], db_name=ADA_TEST) == 2
    assert "no edits document" in capsys.readouterr().err
