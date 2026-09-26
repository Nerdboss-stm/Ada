"""A3: verifier and recorder (SPEC §2.4-2.6 test 3, frames sha stable)."""

import math
import re

import numpy as np
import pytest

from core.contracts import FramesDoc
from core.db import ADA_TEST, db
from scripts.export_fixtures import flail_gait, showcase_task, trot_gait, zero_gait
from sim import bounds as B
from sim import model as M
from sim import verifier as V
from sim.record import write_frames

TASK = showcase_task()
PHYSICAL_REASONS = {B.FINITE_REASON, B.HEIGHT_REASON, B.SPEED_REASON}


@pytest.fixture(scope="module")
def trot_run():
    return V.evaluate(trot_gait(), TASK, 0, record=True)


@pytest.fixture(scope="module")
def flail_run():
    return V.evaluate(flail_gait(), TASK, 0)


# --- bounds -----------------------------------------------------------------

def test_bounds_values_and_sha():
    assert (B.RATED_GEAR, B.TORQUE_RATIO_MAX, B.SPEED_MAX_MPS, B.SPEED_WINDOW_S, B.TORSO_Z_MAX_M,
            B.REQUIRE_FINITE) == (150.0, 1.0, 8.0, 0.3, 3.0, True)
    assert B.SANITY_BOUNDS["rated_torque_ratio_max"] == 1.0 and B.SANITY_BOUNDS["rated_gear"] == 150.0
    assert B.bounds_sha256() == B.bounds_sha256() and len(B.bounds_sha256()) == 64
    assert V.SPEED_WINDOW == 10
    assert B.torque_reason("hip_1", 4.96, 0.021) == "hip_1 at 5.0× rated torque at t=0.02 s"
    assert B.HEIGHT_REASON == "torso rose above the 3.0 m bound"
    assert B.FINITE_REASON == "simulation diverged"
    assert B.SPEED_REASON == "body speed exceeds physical bound; exploits the simulator"


# --- evaluate ---------------------------------------------------------------

def test_result_keys_follow_contract(trot_run):
    plain = V.evaluate(zero_gait(), TASK, 0)
    assert set(plain) == {"distance_m", "fell", "sanity", "success", "peak_torque"}
    assert set(plain["sanity"]) == {"pass", "violation", "violation_frame"}
    assert set(trot_run) == {"distance_m", "fell", "sanity", "success", "peak_torque", "frames",
                             "frames_sha256"}
    assert list(plain["peak_torque"]) == list(M.FORCE_JOINTS)


def test_zero_gait_passes_sanity_and_does_not_succeed():
    r = V.evaluate(zero_gait(), TASK, 0)
    assert r["sanity"] == {"pass": True, "violation": None, "violation_frame": None}
    assert not r["fell"]
    assert abs(r["distance_m"]) < 0.1  # measured from the reset pose; settling slides ~0.05 m
    assert not r["success"]


def test_trot_walks_and_succeeds(trot_run):
    assert trot_run["distance_m"] > 2.0
    assert not trot_run["fell"]
    assert trot_run["sanity"]["pass"]
    assert trot_run["success"]
    # stock gear with ctrl clipped to 1 peaks at exactly rated torque, never above
    assert max(trot_run["peak_torque"].values()) == 1.0


def test_honest_stock_power_gait_passes_sanity_even_if_it_falls():
    # kd 0 at stock power reaches 3-4 m/s and flips: honest, just bad.
    r = V.evaluate(trot_gait(kd=0.0), TASK, 0)
    assert r["sanity"] == {"pass": True, "violation": None, "violation_frame": None}
    assert r["fell"]
    assert not r["success"]


def test_flail_fails_on_torque_with_frame(flail_run):
    s = flail_run["sanity"]
    assert s["pass"] is False
    assert s["violation"] not in PHYSICAL_REASONS
    assert re.fullmatch(r"(hip|ankle)_[1-4] at 5\.0× rated torque at t=\d+\.\d\d s", s["violation"])
    assert isinstance(s["violation_frame"], int) and s["violation_frame"] >= 0
    assert max(flail_run["peak_torque"].values()) > B.TORQUE_RATIO_MAX
    assert not flail_run["success"]


def test_over_torque_without_physical_breach_uses_first_over_torque_frame():
    # the trot at power 1.5: over rated torque but no launch
    r = V.evaluate(trot_gait().model_copy(update={"power": 1.5}), TASK, 0, record=True)
    s = r["sanity"]
    assert V.physical_violation(r["frames"]) is None
    peak_joint = max(r["peak_torque"], key=r["peak_torque"].get)
    assert s["violation"].startswith(f"{peak_joint} at 1.5× rated torque at t=")
    assert isinstance(s["violation_frame"], int) and 0 <= s["violation_frame"] < 50


def test_torque_log_peaks_and_first_over_step():
    m, _ = M.load()
    log = V.TorqueLog(m)

    class D:
        qfrc_actuator = np.zeros(m.nv)

    for step, dof_ratio in enumerate([0.5, 1.2, 3.0, 2.0]):
        D.qfrc_actuator[:] = 0.0
        D.qfrc_actuator[log.dofs[2]] = -dof_ratio * 150.0  # hip_2, sign ignored
        log(step, D)
    assert log.first_over_step == 1
    assert log.peak_torque()["hip_2"] == 3.0 and log.peak_step[2] == 2
    assert log.violation() == "hip_2 at 3.0× rated torque at t=0.03 s"
    assert V.step_frame(1, 333) == 0 and V.step_frame(999, 333) == 332


def test_record_twice_same_sha(trot_run):
    again = V.evaluate(trot_gait(), TASK, 0, record=True)
    assert again["frames_sha256"] == trot_run["frames_sha256"]
    assert again["frames"] == trot_run["frames"]


# --- physical bounds on synthetic frames ------------------------------------

def _still(n=20, z=0.5):
    return [{"geoms": [[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]], "forces": [0.0] * 8, "torso": [0.0, 0.0, z]}
            for _ in range(n)]


def test_physical_checks_each_trigger_their_reason():
    assert V.physical_violation(_still()) is None

    nan = _still()
    nan[7]["forces"][3] = math.nan
    assert V.physical_violation(nan) == (B.FINITE_REASON, 7)

    high = _still()
    high[5]["torso"][2] = 3.01
    assert V.physical_violation(high) == (B.HEIGHT_REASON, 5)

    fast = _still()
    for f in fast[12:]:
        f["torso"][0] = 2.5  # 2.5 m in one window of 0.3 s = 8.33 m/s
    assert V.physical_violation(fast) == (B.SPEED_REASON, 12)

    slow = _still()
    for f in slow[12:]:
        f["torso"][0] = 2.3  # 7.67 m/s: honest
    assert V.physical_violation(slow) is None


# --- record -----------------------------------------------------------------

@pytest.fixture
def test_frames_coll():
    database = db(ADA_TEST)
    database.drop_collection("frames")
    yield database
    database.drop_collection("frames")


def test_write_frames_round_trip(test_frames_coll, trot_run):
    frames = trot_run["frames"]
    a = write_frames(frames, "run-1", "v0", "showcase", db_name=ADA_TEST)
    b = write_frames(frames, "run-1", "v0", "showcase", db_name=ADA_TEST)
    c = write_frames(frames, "run-2", "v0", "rejected", violation_frame=42, db_name=ADA_TEST)
    assert a == b != c
    assert test_frames_coll.frames.count_documents({}) == 2
    doc = FramesDoc.model_validate(test_frames_coll.frames.find_one({"_id": a}))
    assert doc.sha256 == trot_run["frames_sha256"]
    assert len(doc.frames) == len(frames) and doc.violation_frame is None
    assert FramesDoc.model_validate(test_frames_coll.frames.find_one({"_id": c})).violation_frame == 42
    with pytest.raises(ValueError):
        write_frames(frames, "run-3", "v0", "rejected", violation_frame=len(frames), db_name=ADA_TEST)
