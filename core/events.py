"""core.events.emit(): the only writer of `events` (CONTRACTS.md §3–4)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.contracts import Event
from core.db import ADA, db


def emit(
    stage: str,
    status: str,
    payload: dict[str, Any] | None = None,
    round_id: str | None = None,
    version_id: str | None = None,
    edit_id: str | None = None,
    db_name: str = ADA,
) -> str:
    """Validate one Event (stage list, 2 KB payload cap), insert it, return its _id as str."""
    event = Event(
        ts=datetime.now(timezone.utc),
        round_id=round_id,
        version_id=version_id,
        edit_id=edit_id,
        stage=stage,
        status=status,
        payload=payload or {},
    )
    result = db(db_name).events.insert_one(event.model_dump(by_alias=True, exclude={"id"}))
    return str(result.inserted_id)
