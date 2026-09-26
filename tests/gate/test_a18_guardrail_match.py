"""A18: gate.constraints uses the guardrail document matching all live values, never _id order."""

from datetime import datetime, timezone

import pytest

from core.contracts import Harness, HarnessGuardrails, Version
from core.db import ADA_TEST, db
from gate import constraints
from harness import guardrails
from harness.guardrails import COLLECTION, content_sha256, seed_guardrails

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
OLD_WHITELIST = ["read_task", "submit_gait", "preview_run", "get_contact_log", "list_my_attempts"]


def version(tools):
    return Version(_id="t-a18-c", status="candidate", created_at=NOW, harness=Harness(
        tools=tools, model_per_step={"agent": "agent_v0"}, engine={"temperature": 0, "max_attempts": 3}))


def stale_doc() -> dict:
    content = guardrails.current_content().model_dump(mode="json") | {"tool_whitelist": OLD_WHITELIST}
    return HarnessGuardrails(_id=content_sha256(content), content=content).model_dump(by_alias=True, mode="json")


@pytest.fixture
def adb():
    database = db(ADA_TEST)
    existing = {d["_id"] for d in database[COLLECTION].find({}, {"_id": 1})}
    stale = stale_doc()
    added = [seed_guardrails(ADA_TEST)]
    database[COLLECTION].replace_one({"_id": stale["_id"]}, stale, upsert=True)
    added.append(stale["_id"])
    yield database
    for gid in added:
        if gid not in existing:
            database[COLLECTION].delete_one({"_id": gid})


def test_new_tools_pass_with_a_stale_document_present(adb):
    assert constraints.check(version(["read_task", "submit_gait", "recall_best_gait", "preview_all_seeds"]),
                             ADA_TEST) is None


def test_only_a_stale_whitelist_fails(adb, monkeypatch):
    stale = HarnessGuardrails.model_validate(stale_doc())
    monkeypatch.setattr(guardrails, "load_guardrails", lambda db_name: [stale])
    assert constraints.check(version(["read_task", "submit_gait"]), ADA_TEST) == \
        "tool whitelist changed: it matches no guardrail document"
