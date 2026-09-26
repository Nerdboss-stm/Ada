"""A12: scripts/pin_snapshot.py on ada_test.

Every document is tagged t-a12-{uuid}; build_snapshot runs with that prefix so other lanes'
documents on the shared database never enter the pin, and teardown deletes only our ids.
Prefixed ids carry no version number, so version order falls back to created_at.
"""

import types
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.contracts import Edit, Metrics, Snapshot, Version
from core.db import ADA_TEST, db
from harness.agent import V0_HARNESS
from scripts import pin_snapshot as PS

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def at(s: int) -> datetime:
    return T0 + timedelta(seconds=s)


@pytest.fixture
def world():
    d = db(ADA_TEST)
    tag = f"t-a12-{uuid.uuid4().hex[:8]}"
    snaps: list[str] = []

    def version(suffix, status, s, rel=None, mean=None, frames=None):
        metrics = None if rel is None else Metrics(
            train_reliability=rel, holdout_reliability_80=0.0, mean_distance_m=0.0,
            cost_per_run_usd=0.001, n=6, train_mean_distance_m=mean)
        d.versions.insert_one(Version(_id=f"{tag}-{suffix}", status=status, harness=V0_HARNESS,
                                      metrics=metrics, showcase_frames_id=frames,
                                      created_at=at(s)).model_dump(by_alias=True))

    def edit(suffix, s, src, dst, verdict, *, bet=None, actual=None, attempt=None,
             reason=None, frames=None, origin="model"):
        d.edits.insert_one(Edit(
            _id=f"{tag}-{suffix}", from_version=f"{tag}-{src}", to_version=f"{tag}-{dst}",
            origin=origin, primitive="rules", verdict=verdict, predicted_delta_m=bet,
            actual_delta_m=actual, attempt_frames_id=attempt, reason=reason, frames_id=frames,
            created_at=at(s)).model_dump(by_alias=True))

    def pin():
        snap, ordered = PS.build_snapshot(v0=f"{tag}-v0", frontier=f"{tag}-frontier",
                                          prefix=tag, db_name=ADA_TEST)
        snaps.append(snap.id)
        return snap, ordered

    yield types.SimpleNamespace(db=d, tag=tag, id=lambda s: f"{tag}-{s}", version=version,
                                edit=edit, pin=pin)

    own = {"$regex": f"^{tag}"}
    for coll in ("versions", "edits", "swaps", "scoreboard"):
        d[coll].delete_many({"_id": own})
    d.snapshots.delete_many({"_id": {"$in": snaps}})


def seed(w):
    w.version("v0", "baseline", 0, rel=0.5, mean=1.0, frames="fv0")
    w.version("frontier", "frontier", 1, rel=0.5, mean=2.0, frames="ffr")
    w.version("v1", "accepted", 2, rel=0.5, mean=1.4, frames="fv1")
    w.version("v2", "rejected", 3, rel=0.3, mean=0.6)
    w.version("v3", "accepted", 4, rel=0.67, mean=1.2, frames="fv3")  # the train leader
    w.version("v4", "rejected", 5, rel=0.5, mean=0.9, frames="fv4")
    w.version("v6", "accepted", 6, rel=0.5, mean=1.5)  # kept, but no showcase frames

    w.edit("e1", 10, "v0", "v1", "accepted", bet=0.2, actual=0.4, attempt="a1")   # largest gain
    w.edit("e2", 11, "v1", "v2", "rejected", bet=0.5, actual=-0.8, attempt="a2")
    w.edit("e3", 12, "v1", "v3", "accepted", bet=0.1, attempt="a3")  # measured from means: -0.2
    w.edit("e4", 13, "v3", "v4", "rejected", bet=0.9, actual=-0.3, attempt="a4")  # worst bet
    w.edit("e5", 14, "v3", "v6", "accepted", bet=0.3, actual=0.3, attempt="a5")
    w.edit("e6", 15, "v3", "v6", "accepted", bet=0.3, actual=0.3)  # no attempt frames: skipped
    w.edit("e7", 16, "v3", "v5", "rejected", reason="peak hip torque 1.8x rated torque at 2.1 s",
           frames="fc7")
    w.edit("e8", 17, "v3", "v5", "rejected", reason="exceeds Ada rated motors", frames="fc8")  # cheat
    w.edit("e9", 18, "v3", "v5", "rejected", reason="exceeds rated torque")  # no frames
    w.edit("e10", 19, "v3", "v5", "rejected", reason="Rated Torque exceeded", frames="fc10",
           origin="cli")
    w.edit("e11", 20, "v3", "v5", "rejected", reason="exceeds Ada rated motors", frames="")

    w.db.swaps.insert_many([{"_id": w.id("swap-old"), "captured_at": at(30)},
                            {"_id": w.id("swap-new"), "captured_at": at(31)}])
    w.db.scoreboard.insert_many([{"_id": w.id("board-new"), "created_at": at(33)},
                                 {"_id": w.id("board-old"), "created_at": at(32)}])


def test_pinned_ids_match_seeded_dataset(world):
    seed(world)
    snap, ordered = world.pin()
    i = world.id

    assert snap.best_version == i("v3")
    assert (snap.v0, snap.frontier) == (i("v0"), i("frontier"))
    assert snap.attempt_edit_ids == [i("e4"), i("e1"), i("e3"), i("e5")]
    assert [a["slot"] for a in ordered] == ["worst bet", "largest gain", "kept", "kept"]
    assert ordered[2]["measured"] == pytest.approx(-0.2)
    assert snap.cheat_edit_id == i("e8")
    assert (snap.swaps_id, snap.scoreboard_id) == (i("swap-new"), i("board-new"))
    assert [(g.version_id, g.frames_id) for g in snap.ghosts] == [
        (i("v0"), "fv0"), (i("frontier"), "ffr"), (i("v1"), "fv1"), (i("v3"), "fv3")]

    stored = world.db.snapshots.find_one({"_id": snap.id})
    assert Snapshot.model_validate(stored) == snap

    text = PS.summary(snap, ordered, ADA_TEST)
    for needle in (snap.id, i("v3"), i("e4"), i("e8"), "exceeds Ada rated motors", "fv3"):
        assert needle in text


def test_newer_edit_after_pin_does_not_change_stored_snapshot(world):
    seed(world)
    snap, _ = world.pin()
    before = world.db.snapshots.find_one({"_id": snap.id})

    world.edit("e12", 40, "v3", "v5", "rejected", reason="exceeds Ada rated motors", frames="fc12")
    world.edit("e13", 41, "v3", "v6", "accepted", bet=0.1, actual=2.0, attempt="a13")

    assert world.db.snapshots.find_one({"_id": snap.id}) == before
    assert Snapshot.model_validate(before) == snap

    # the new edits would have changed a fresh pin, so the frozen one is what held
    fresh, _ = world.pin()
    assert fresh.cheat_edit_id == world.id("e12")
    assert fresh.attempt_edit_ids[:2] == [world.id("e4"), world.id("e13")]


def test_ghosts_stop_at_newest_accepted_plus_v0_and_frontier(world):
    world.version("v0", "baseline", 0, rel=0.1, mean=0.5, frames="fv0")
    world.version("frontier", "frontier", 1, rel=0.1, mean=0.5)
    n = PS.GHOST_LIMIT + 2
    for k in range(1, n + 1):
        world.version(f"a{k:02d}", "accepted", 10 + k, rel=0.2, mean=float(k), frames=f"f{k:02d}")

    snap, _ = world.pin()
    ids = [g.version_id for g in snap.ghosts]
    assert len(ids) == PS.GHOST_LIMIT + 2
    assert ids[:2] == [world.id("v0"), world.id("frontier")]
    assert ids[2:] == [world.id(f"a{k:02d}") for k in range(3, n + 1)]
    assert snap.ghosts[1].frames_id is None
    assert snap.attempt_edit_ids == [] and snap.cheat_edit_id is None
    assert (snap.swaps_id, snap.scoreboard_id) == (None, None)
