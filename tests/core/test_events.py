import threading
import time

import pytest
from pydantic import ValidationError

from core.db import ADA_TEST, db
from core.events import emit

COLL = "events"


@pytest.fixture
def coll():
    c = db(ADA_TEST)[COLL]
    c.drop()
    db(ADA_TEST).create_collection(COLL)
    yield c
    c.drop()


def test_emit_arrives_on_change_stream(coll):
    ids = {"round_id": "r1", "version_id": "v1", "edit_id": "e1"}
    emitted: list[str] = []
    with coll.watch([{"$match": {"operationType": "insert"}}], max_await_time_ms=200) as stream:
        worker = threading.Thread(
            target=lambda: emitted.append(
                emit("gate.verifier", "pass", {"k": 1}, db_name=ADA_TEST, **ids)
            )
        )
        worker.start()
        deadline = time.monotonic() + 3
        change = None
        while change is None and time.monotonic() < deadline:
            change = stream.try_next()
        worker.join(timeout=3)

    assert change is not None, "event did not arrive within 3 s"
    doc = change["fullDocument"]
    assert doc["stage"] == "gate.verifier"
    assert doc["status"] == "pass"
    assert {k: doc[k] for k in ids} == ids
    assert doc["payload"] == {"k": 1}
    assert emitted == [str(doc["_id"])]


def test_oversized_payload_raises_before_insert(coll):
    with pytest.raises(ValidationError, match="limit 2048"):
        emit("harness", "info", {"x": "a" * 3000}, db_name=ADA_TEST)
    assert coll.count_documents({}) == 0


def test_unknown_stage_raises_before_insert(coll):
    with pytest.raises(ValidationError):
        emit("nope", "info", db_name=ADA_TEST)
    assert coll.count_documents({}) == 0
