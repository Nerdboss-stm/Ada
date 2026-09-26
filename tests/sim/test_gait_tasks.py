"""A2: gait schema, controller, tasks (SPEC §2.2, §2.3, §2.6 tests 2 and 4)."""

import math

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from core.contracts import Task
from core.db import ADA_TEST, db
from sim import model as M
from sim import tasks as T
from sim.controller import make_ctrl_fn
from sim.gait import Gait, neutral_offsets

FRAME_DT = M.TIMESTEP * M.RECORD_EVERY


def gait_dict(**over):
    neutral = neutral_offsets()
    g = {
        "frequency_hz": 1.0, "power": 1.0, "kp": 2.0, "kd": 0.1, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.0, "offset": neutral[n], "phase": 0.0} for n in M.FORCE_JOINTS],
    }
    g.update(over)
    return g


# --- schema -----------------------------------------------------------------

def test_neutral_offsets_match_init_qpos():
    assert list(neutral_offsets()) == list(M.FORCE_JOINTS)
    assert neutral_offsets() == {
        "hip_1": 0.0, "ankle_1": 1.0, "hip_2": 0.0, "ankle_2": -1.0,
        "hip_3": 0.0, "ankle_3": -1.0, "hip_4": 0.0, "ankle_4": 1.0,
    }


def test_descriptions_state_neutral_offsets():
    schema = str(Gait.model_json_schema())
    for name, v in neutral_offsets().items():
        assert f"{name} {v:+.2f}" in schema


def test_boundary_values_accepted():
    Gait.model_validate(gait_dict(frequency_hz=0.5, power=5.0, kp=20, kd=0.0, tilt_gain=2.0))
    g = gait_dict(frequency_hz=3.0, power=0.2, kp=0.5, kd=2.0)
    g["joints"][0].update(amplitude=1.2, offset=-1.0, phase=6.2832)
    Gait.model_validate(g)


@pytest.mark.parametrize("field,value", [
    ("frequency_hz", 0.49), ("frequency_hz", 3.01), ("power", 0.19), ("power", 5.01),
    ("kp", 0.49), ("kp", 20.01), ("kd", -0.01), ("kd", 2.01), ("tilt_gain", -0.01), ("tilt_gain", 2.01),
])
def test_top_level_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        Gait.model_validate(gait_dict(**{field: value}))


@pytest.mark.parametrize("field,value", [
    ("amplitude", -0.01), ("amplitude", 1.21), ("offset", -1.01), ("offset", 1.01),
    ("phase", -0.01), ("phase", 6.2833),
])
def test_joint_out_of_range_rejected(field, value):
    g = gait_dict()
    g["joints"][3][field] = value
    with pytest.raises(ValidationError):
        Gait.model_validate(g)


def test_joint_names_order_count_and_extras_rejected():
    swapped = gait_dict()
    swapped["joints"][0], swapped["joints"][1] = swapped["joints"][1], swapped["joints"][0]
    renamed = gait_dict()
    renamed["joints"][0]["name"] = "hip_9"
    short = gait_dict()
    short["joints"] = short["joints"][:7]
    extra_joint_key = gait_dict()
    extra_joint_key["joints"][0]["gain"] = 1.0
    for bad in (swapped, renamed, short, extra_joint_key, gait_dict(glitch=1)):
        with pytest.raises(ValidationError):
            Gait.model_validate(bad)


# --- controller -------------------------------------------------------------

def _at_neutral():
    m, d = M.load()
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.numeric("init_qpos").data
    mujoco.mj_forward(m, d)
    return m, d


def test_ctrl_maps_joints_to_actuators():
    m, d = _at_neutral()
    assert np.allclose(make_ctrl_fn(m, Gait.model_validate(gait_dict()))(0.0, d), 0.0)

    g = gait_dict(kp=5.0)
    g["joints"][0]["offset"] += 0.1  # hip_1
    ctrl = make_ctrl_fn(m, Gait.model_validate(g))(0.0, d)
    hip_1_act = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == m.joint("hip_1").id)
    expected = np.zeros(m.nu)
    expected[hip_1_act] = 0.5
    assert np.allclose(ctrl, expected)


def test_tilt_correction_moves_hips_only():
    m, d = _at_neutral()
    half = math.radians(5) / 2
    d.qpos[3:7] = (math.cos(half), 0.0, math.sin(half), 0.0)  # pitch 5 deg
    assert np.allclose(make_ctrl_fn(m, Gait.model_validate(gait_dict()))(0.0, d), 0.0)
    ctrl = make_ctrl_fn(m, Gait.model_validate(gait_dict(tilt_gain=1.0)))(0.0, d)
    names = [m.joint(int(m.actuator_trnid[a, 0])).name for a in range(m.nu)]
    for a, n in enumerate(names):
        assert (ctrl[a] != 0.0) == n.startswith("hip"), n


def test_neutral_gait_through_controller_stands_still():
    gait = Gait.model_validate(gait_dict())
    m, d = M.load(power=gait.power)
    frames = M.rollout(m, d, 0, make_ctrl_fn(m, gait))
    torso = np.array([f["torso"] for f in frames])
    quat = np.array([f["geoms"][0][3:] for f in frames])  # torso_geom is manifest geom 0
    up_z = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    w = round(0.3 / FRAME_DT)
    window_speed = np.linalg.norm(torso[w:] - torso[:-w], axis=1) / (w * FRAME_DT)

    assert abs(torso[-1, 0] - torso[0, 0]) < 0.05
    assert torso[:, 2].min() >= 0.25 and up_z.min() >= 0  # no fall
    assert np.isfinite(torso).all() and np.isfinite(quat).all()  # sanity: finite
    assert torso[:, 2].max() <= 1.5  # sanity: height
    assert window_speed.max() <= 2.5  # sanity: speed


# --- tasks ------------------------------------------------------------------

def test_task_splits_and_grid():
    tasks = T.build_tasks()
    by_split = {s: [t for t in tasks if t.split == s] for s in ("train", "holdout", "showcase")}
    assert [len(by_split[s]) for s in ("train", "holdout", "showcase")] == [12, 6, 1]
    train = {(t.slope_deg, t.friction) for t in by_split["train"]}
    assert train == {(s, f) for s in (0.0, 2.0, 4.0, 6.0) for f in (1.0, 0.6, 0.35)}
    holdout = {(t.slope_deg, t.friction) for t in by_split["holdout"]}
    assert len(holdout) == 6 and not holdout & train
    (show,) = by_split["showcase"]
    assert (show.slope_deg, show.friction, show.eval_seeds) == (0.0, 1.0, [T.SHOWCASE_SEED])
    assert all(t.target_m == 2.0 for t in tasks)
    assert len({t.id for t in tasks}) == len(tasks)
    for t in tasks:
        Task.model_validate(t.model_dump(by_alias=True))


def test_practice_and_eval_seeds_disjoint():
    tasks = T.build_tasks()
    practice = [s for t in tasks for s in t.practice_seeds]
    evals = [s for t in tasks for s in t.eval_seeds]
    assert practice and evals
    assert not set(practice) & set(evals)
    assert len(set(practice + evals)) == len(practice) + len(evals)  # every seed unique across tasks


@pytest.fixture
def test_tasks_coll():
    database = db(ADA_TEST)
    database.drop_collection("tasks")
    yield database
    database.drop_collection("tasks")


def test_write_tasks_idempotent(test_tasks_coll):
    assert T.write_tasks(test_tasks_coll) == 19
    assert T.write_tasks(test_tasks_coll) == 19
    docs = list(test_tasks_coll.tasks.find())
    assert len(docs) == 19
    stored = {Task.model_validate(d).id: Task.model_validate(d) for d in docs}
    assert stored == {t.id: t for t in T.build_tasks()}
