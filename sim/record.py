"""Recorder (SPEC §2.5): writes one FramesDoc to the `frames` collection.

The _id is derived from (run_id, version_id, kind, frames sha256), so writing the same
recording again upserts the same document.
"""

from __future__ import annotations

import hashlib
import json

from core.contracts import FrameKind, FramesDoc
from core.db import ADA, db
from sim import model as M

FPS = round(1.0 / (M.TIMESTEP * M.RECORD_EVERY), 4)


def manifest_version() -> str:
    return json.loads(M.MANIFEST_PATH.read_text())["manifest_version"]


def write_frames(
    frames: list[dict],
    run_id: str,
    version_id: str,
    kind: FrameKind,
    violation_frame: int | None = None,
    db_name: str = ADA,
) -> str:
    """Validate and upsert a frames document; returns its _id."""
    if violation_frame is not None and not 0 <= violation_frame < len(frames):
        raise ValueError(f"violation_frame {violation_frame} outside 0..{len(frames) - 1}")
    sha = M.frames_sha256(frames)
    doc_id = hashlib.sha256(f"{run_id}|{version_id}|{kind}|{sha}".encode()).hexdigest()[:24]
    doc = FramesDoc(
        _id=doc_id, run_id=run_id, version_id=version_id, kind=kind,
        manifest_version=manifest_version(), fps=FPS, frames=frames, sha256=sha,
        violation_frame=violation_frame,
    ).model_dump(by_alias=True)
    db(db_name).frames.replace_one({"_id": doc_id}, doc, upsert=True)
    return doc_id
