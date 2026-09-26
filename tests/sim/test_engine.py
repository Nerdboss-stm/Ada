"""A1: engine determinism and zero gait (SPEC §2.6 tests 1-2)."""

import json
import math

import mujoco
import numpy as np
import pytest

from sim import model as M

FRAME_DT = M.TIMESTEP * M.RECORD_EVERY


def zero_ctrl(t, data):
    return np.zeros(8)


def noisy_ctrl(seed):
    table = np.random.default_rng(seed).uniform(-1, 1, size=(M.EPISODE_STEPS + 1, 8))
    return lambda t, data: table[int(round(t / M.TIMESTEP))]


def run(seed, ctrl_fn, **task):
    m, d = M.load(**task)
    return M.rollout(m, d, seed, ctrl_fn)


def test_frames_match_contract_shape():
    frames = run(0, zero_ctrl)
    assert len(frames) == M.EPISODE_STEPS // M.RECORD_EVERY
    f = frames[0]
    assert set(f) == {"geoms", "contacts", "forces", "torso"}
    assert len(f["geoms"]) == 13 and all(len(g) == 7 for g in f["geoms"])
    assert len(f["contacts"]) == 4 and all(isinstance(c, bool) for c in f["contacts"])
    assert len(f["forces"]) == 8 and all(-1.0 <= v <= 1.0 for v in f["forces"])
    assert len(f["torso"]) == 3


def test_determinism_same_inputs_same_sha():
    task = dict(power=1.5, slope_deg=4.0, friction=0.6)
    a = run(7, noisy_ctrl(1), **task)
    b = run(7, noisy_ctrl(1), **task)
    assert M.frames_sha256(a) == M.frames_sha256(b)
    assert any(any(f["contacts"]) for f in a)
    assert any(any(v != 0.0 for v in f["forces"]) for f in a)
    assert M.frames_sha256(run(8, noisy_ctrl(1), **task)) != M.frames_sha256(a)


def test_zero_gait_stands_still():
    frames = run(0, zero_ctrl, power=1.0)
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


def test_load_applies_task_physics():
    m, _ = M.load(power=2.0, slope_deg=4.0, friction=0.35)
    assert np.all(m.actuator_gear[:, 0] == 300.0)
    th = math.radians(4.0)
    assert np.allclose(m.opt.gravity, [-9.81 * math.sin(th), 0, -9.81 * math.cos(th)])
    floor = m.geom("floor").id
    assert m.geom_friction[floor, 0] == pytest.approx(0.35)
    assert m.geom_priority[floor] > m.geom_priority[np.arange(m.ngeom) != floor].max()


def test_floor_friction_governs_contact():
    m, d = M.load(friction=0.35)
    M.reset(m, d, 0)
    for _ in range(20):
        mujoco.mj_step(m, d)
    assert d.ncon > 0
    assert all(d.contact[i].friction[0] == pytest.approx(0.35) for i in range(d.ncon))


def test_manifest_file_is_current():
    on_disk = json.loads(M.MANIFEST_PATH.read_text())
    assert on_disk == M.build_manifest()
    names = [g["name"] for g in on_disk["geoms"]]
    assert "floor" not in names and len(names) == 13
    assert {g["type"] for g in on_disk["geoms"]} == {"sphere", "capsule"}
