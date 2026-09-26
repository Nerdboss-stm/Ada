"""Shared gate fixtures: own parent/candidate/edit ids in ada_test, episodes, verifier,
LLM and stages 2-3 patched. The fake verifier's violation reason is `world.violation`;
tests compare the gate's reason to it, never to a hard-coded verifier string."""

import time
import uuid
from datetime import datetime, timezone

import pytest

import core.llm as llm
import gate.pipeline as pipeline
import harness.agent as agent
import sim.verifier as verifier
from core.contracts import Edit, Harness, Version
from core.db import ADA_TEST, db
from gate import verifier_stage
from harness.guardrails import COLLECTION, seed_guardrails
from sim import model as M
from sim.gait import Gait

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
FRAME = {"geoms": [[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], "contacts": [True] * 4,
         "forces": [0.0] * 8, "torso": [0.0, 0.0, 0.5]}


def gait(power=1.0):
    return Gait.model_validate({
        "frequency_hz": 1.0, "power": power, "kp": 5.0, "kd": 0.1, "tilt_gain": 0.0,
        "joints": [{"name": n, "amplitude": 0.3, "offset": 0.0, "phase": 0.0} for n in M.FORCE_JOINTS],
    })


def harness(**over):
    base = {"tools": ["read_task", "submit_gait"], "model_per_step": {"agent": "agent_v0"},
            "engine": {"temperature": 0, "max_attempts": 3}}
    return Harness.model_validate({**base, **over})


@pytest.fixture
def adb():
    return db(ADA_TEST)


@pytest.fixture
def world(monkeypatch, adb):
    tag = uuid.uuid4().hex[:8]
    existing = {d["_id"] for d in adb[COLLECTION].find({}, {"_id": 1})}
    gid = seed_guardrails(ADA_TEST)
    tasks = verifier_stage.gate_tasks(ADA_TEST)

    class W:
        parent_id, cand_id = f"t-a8-{tag}-p", f"t-a8-{tag}-c"
        edit_id, round_id = f"t-a8-{tag}-e", f"t-a8-{tag}-r"
        episodes: list[str] = []  # task ids in completion order (episodes run in parallel)
        no_gait: set[str] = set()  # candidate task ids whose episode submits nothing
        delay: dict[str, float] = {}  # candidate task id -> seconds the episode takes
        llm_calls: list = []
        power = 1.0
        violation = "test double: sanity bound tripped"
        outcome: dict[str, tuple[bool, float]] = {}  # candidate task_id -> (success, distance)
        violate: set[tuple[str, int]] = set()  # candidate (task_id, seed) runs that break physics

        @classmethod
        def parent_runs(cls, outcome, gaits=None, violations=()):
            """gaits: parent task ids that carry a submitted gait; default every task.
            violations: parent (task_id, seed) runs stored with a failed sanity check."""
            gaits = {t.id for t in tasks} if gaits is None else set(gaits)
            docs = [{"task_id": t.id, "seed": s, "split": "train", "version_id": cls.parent_id,
                     "success": outcome[t.id][0] and (t.id, s) not in violations,
                     "distance_m": outcome[t.id][1],
                     "gait": {"power": 1.0} if t.id in gaits else {},
                     "sanity": {"pass": (t.id, s) not in violations,
                                "violation": cls.violation if (t.id, s) in violations else None,
                                "violation_frame": 1 if (t.id, s) in violations else None}}
                    for t in tasks for s in t.eval_seeds[:verifier_stage.K]]
            adb.runs.insert_many(docs)

        @classmethod
        def trace_id(cls, task_id):
            return f"{cls.cand_id}-{task_id}"

        @classmethod
        def versions(cls, **over):
            parent = Version(_id=cls.parent_id, status="accepted", harness=harness(), created_at=NOW)
            cand = Version(_id=cls.cand_id, parent=cls.parent_id, status="candidate",
                           harness=harness(**over), created_at=NOW)
            return parent, cand

        @classmethod
        def gate(cls, **over):
            parent, cand = cls.versions(**over)
            edit = Edit(_id=cls.edit_id, round_id=cls.round_id, from_version=cls.parent_id,
                        to_version=cls.cand_id, origin="model", primitive="rules", op="add",
                        new="Keep the torso level.", predicted_delta=0.1, created_at=NOW)
            return pipeline.run_gate(parent, cand, edit, cls.round_id, db_name=ADA_TEST)

        @classmethod
        def events(cls):
            return [(e["stage"], e["status"], e["payload"])
                    for e in adb.events.find({"edit_id": cls.edit_id}).sort("_id", 1)]

    W.tasks = tasks

    def run_episode(task, version_id, h, *, db_name, **_):
        time.sleep(W.delay.get(task.id, 0.0))
        W.episodes.append(task.id)
        g = None if task.id in W.no_gait else gait(W.power)
        return agent.EpisodeResult(gait=g, attempts=1, cost_usd=0.002, tokens=100,
                                   model_id="test/agent", trace_id=f"{version_id}-{task.id}")

    def evaluate(g, task, seed, record=False):
        if g.power > 1.0 or (task.id, seed) in W.violate:
            out = {"distance_m": 6.0, "fell": False, "success": False,
                   "sanity": {"pass": False, "violation": W.violation, "violation_frame": 1}}
        else:
            ok, dist = W.outcome[task.id]
            out = {"distance_m": dist, "fell": False, "success": ok,
                   "sanity": {"pass": True, "violation": None, "violation_frame": None}}
        if record:
            out["frames"] = [FRAME] * 3
        return out

    def cached_chat(*a, **k):
        W.llm_calls.append((a, k))
        raise AssertionError("the gate made a model call")

    monkeypatch.setattr(agent, "run_episode", run_episode)
    monkeypatch.setattr(verifier, "evaluate", evaluate)
    monkeypatch.setattr(llm, "cached_chat", cached_chat)
    monkeypatch.setattr(pipeline, "_stage_module", lambda name: None)
    yield W

    ids = [W.parent_id, W.cand_id]
    adb.runs.delete_many({"version_id": {"$in": ids}})
    adb.frames.delete_many({"version_id": W.cand_id})
    adb.events.delete_many({"edit_id": W.edit_id})
    adb.traces.delete_many({"trace_id": {"$in": [W.trace_id(t.id) for t in tasks]}})
    if gid not in existing:
        adb[COLLECTION].delete_one({"_id": gid})
