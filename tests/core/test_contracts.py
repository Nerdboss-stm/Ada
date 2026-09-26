from datetime import datetime, timezone

import pytest
from bson import ObjectId
from pydantic import BaseModel, ValidationError

from core import contracts as c
from core import db as core_db

NOW = datetime(2026, 9, 26, 11, 5, 0, tzinfo=timezone.utc)
FRAME = {
    "geoms": [[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.75, 1.0, 0.0, 0.0, 0.0]],
    "contacts": [True, False, True, False],
    "forces": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    "torso": [0.0, 0.0, 0.75],
}

SAMPLES: list[BaseModel] = [
    c.Task.model_validate({
        "_id": "t_train_0", "split": "train", "slope_deg": 5.0, "friction": 0.8,
        "target_m": 3.0, "eval_seeds": [101, 102], "practice_seeds": [1, 2, 3],
    }),
    c.Event.model_validate({
        "ts": NOW, "round_id": "r1", "version_id": "v1", "edit_id": "e1",
        "stage": "gate.verifier", "status": "pass", "payload": {"distance_m": 2.41},
    }),
    c.Event.model_validate({
        "_id": "ev1", "ts": NOW, "round_id": None, "version_id": None, "edit_id": None,
        "stage": "harness", "status": "info", "payload": {},
    }),
    c.Round.model_validate({
        "_id": "r1", "status": "open", "opened_at": NOW, "closed_at": None,
        "budget_usd": 1.5, "budget_s": 600.0, "open_lock": True,
    }),
    c.Version.model_validate({
        "_id": "v1", "parent": "v0", "status": "frontier",
        "harness": {
            "rules": ["keep the torso level"], "context_policy": {"history": 3},
            "tools": ["read_task", "submit_gait"], "model_per_step": {"plan": "agent_v0"},
            "engine": {"max_steps": 4},
        },
        "metrics": {
            "train_reliability": 0.7, "holdout_reliability_80": 0.6, "mean_distance_m": 2.5,
            "cost_per_run_usd": 0.0012345, "n": 20,
        },
        "showcase_frames_id": "f1", "created_at": NOW,
    }),
    c.Edit.model_validate({
        "_id": "e1", "round_id": "r1", "from_version": "v0", "to_version": "v1",
        "origin": "model", "primitive": "add_rule", "old": None, "new": "keep the torso level",
        "rationale": "falls on slopes", "predicted_delta": 0.1, "actual_delta": 0.08,
        "verdict": "accepted", "reason": "all gates pass", "violation_frame": None,
        "frames_id": "f1", "created_at": NOW,
    }),
    c.Run.model_validate({
        "task_id": "t_train_0", "seed": 1, "split": "train", "version_id": "v1",
        "model_id": "some/model", "gait": {"freq_hz": 1.5, "amp": [0.3, 0.4]},
        "distance_m": 2.41, "fell": False,
        "sanity": {"pass": True, "violation": None, "violation_frame": None},
        "success": True, "cost_usd": 0.000321, "tokens": 1234, "trace_id": "tr1",
    }),
    c.Trace.model_validate({
        "trace_id": "tr1", "raw_steps": [{"step": 0, "tool": "read_task"}],
        "compressed_steps": [{"s": 0}], "schema": "gait-v1",
    }),
    c.FramesDoc.model_validate({
        "_id": "f1", "run_id": "run1", "version_id": "v1", "kind": "showcase",
        "manifest_version": 1, "fps": 33.3, "frames": [FRAME, FRAME], "sha256": "ab" * 32,
    }),
    c.Skill.model_validate({
        "_id": "s1", "version_id": "v1", "task_id": "t_train_0", "summary": "trot",
        "gait": {"freq_hz": 1.5}, "embedding": [0.1, 0.2, 0.3], "created_at": NOW,
    }),
    c.HarnessGuardrails.model_validate({
        "_id": "cd" * 32,
        "content": {
            "tool_whitelist": ["read_task", "submit_gait"], "sanity_bounds": {"max_speed": 3.0},
            "rated_power": 1.0, "verifier_sha": "11" * 32, "compressor_sha": "22" * 32,
        },
    }),
    c.HarnessGuardrails.model_validate({
        "_id": "ef" * 32,
        "content": {
            "tool_whitelist": ["read_task"], "sanity_bounds": {}, "rated_power": 1.0,
            "verifier_sha": "11" * 32,
        },
    }),
]


@pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
def test_json_round_trip(model: BaseModel) -> None:
    dumped = model.model_dump_json(by_alias=True)
    assert type(model).model_validate_json(dumped) == model


def test_every_collection_model_is_sampled() -> None:
    expected = {
        c.Task, c.Event, c.Round, c.Version, c.Edit, c.Run,
        c.Trace, c.FramesDoc, c.Skill, c.HarnessGuardrails,
    }
    assert expected <= {type(m) for m in SAMPLES}


def test_aliases_in_dump() -> None:
    run = next(m for m in SAMPLES if isinstance(m, c.Run))
    trace = next(m for m in SAMPLES if isinstance(m, c.Trace))
    assert "pass" in run.model_dump(by_alias=True)["sanity"]
    assert trace.model_dump(by_alias=True)["schema"] == "gait-v1"


def test_object_id_coerced_to_str() -> None:
    oid = ObjectId()
    ev = c.Event.model_validate({"_id": oid, "ts": NOW, "stage": "sensor", "status": "start"})
    assert ev.id == str(oid)
    assert c.Event.model_validate({"ts": NOW, "stage": "sensor", "status": "start"}).id is None


def test_usd_rounded_to_six_places() -> None:
    rnd = next(m for m in SAMPLES if isinstance(m, c.Round))
    assert c.Round.model_validate({**rnd.model_dump(by_alias=True), "budget_usd": 0.12345678}).budget_usd == 0.123457


def test_payload_over_2kb_rejected() -> None:
    with pytest.raises(ValidationError, match="payload"):
        c.Event.model_validate({"ts": NOW, "stage": "sensor", "status": "info", "payload": {"x": "a" * 2100}})


def test_unknown_stage_rejected() -> None:
    with pytest.raises(ValidationError):
        c.Event.model_validate({"ts": NOW, "stage": "gate.other", "status": "info"})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        c.Event.model_validate({"ts": NOW, "stage": "sensor", "status": "info", "surprise": 1})


def test_frame_contacts_length() -> None:
    with pytest.raises(ValidationError):
        c.Frame.model_validate({**FRAME, "contacts": [True, False, True]})


def test_card_stage_order() -> None:
    assert c.CARD_STAGES == ("gate.verifier", "gate.gpa", "gate.meta", "gate.constraints")


def test_client_cached_and_db_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    core_db.client.cache_clear()
    try:
        assert core_db.client() is core_db.client()
        assert core_db.db(core_db.ADA_TEST).name == "ada_test"
        with pytest.raises(ValueError):
            core_db.db("tax")
    finally:
        core_db.client().close()
        core_db.client.cache_clear()
