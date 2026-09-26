"""harness_guardrails (CONTRACTS §3): content-addressed, verified at load.

    python -m harness.guardrails          # seed into `ada`, print the _id

seed_guardrails() writes one document whose _id is sha256 of its content (canonical
JSON: sorted keys, compact separators). load_guardrails() recomputes that hash for
every document and raises GuardrailError on any mismatch or when none exist, so a
tampered or deleted collection stops harness.run before the first episode.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.contracts import GuardrailContent, HarnessGuardrails
from core.db import ADA, db
from harness.agent import TOOL_WHITELIST
from sim import bounds

COLLECTION = "harness_guardrails"
ROOT = Path(__file__).resolve().parent.parent
VERIFIER_PATH = ROOT / "sim" / "verifier.py"
COMPRESS_DIR = ROOT / "compress"


class GuardrailError(RuntimeError):
    pass


def content_sha256(content: dict[str, Any]) -> str:
    blob = json.dumps(content, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_sha256(path: Path) -> str:
    """sha256 over (relative path, sha256 of bytes) of every file, sorted; caches skipped."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f"{f.relative_to(path).as_posix()}\0{file_sha256(f)}\n".encode())
    return h.hexdigest()


def current_content() -> GuardrailContent:
    return GuardrailContent(
        tool_whitelist=list(TOOL_WHITELIST),
        sanity_bounds=dict(bounds.SANITY_BOUNDS),
        rated_power=bounds.RATED_POWER,
        verifier_sha=file_sha256(VERIFIER_PATH),
        compressor_sha=dir_sha256(COMPRESS_DIR) if COMPRESS_DIR.is_dir() else None,
    )


def seed_guardrails(db_name: str = ADA) -> str:
    """Upsert the current guardrails document; returns its _id. Idempotent."""
    content = current_content().model_dump(mode="json")
    doc = HarnessGuardrails(_id=content_sha256(content), content=content)
    db(db_name)[COLLECTION].replace_one({"_id": doc.id}, doc.model_dump(by_alias=True, mode="json"), upsert=True)
    return doc.id


def load_guardrails(db_name: str = ADA) -> list[HarnessGuardrails]:
    docs = list(db(db_name)[COLLECTION].find().sort("_id", 1))
    if not docs:
        raise GuardrailError(f"no {COLLECTION} documents; seed with `python -m harness.guardrails`")
    out = []
    for d in docs:
        content = d.get("content")
        if not isinstance(content, dict) or content_sha256(content) != d["_id"]:
            raise GuardrailError(f"{COLLECTION} {d['_id']}: content does not match its sha256 _id")
        try:
            out.append(HarnessGuardrails.model_validate(d))
        except ValidationError as e:
            raise GuardrailError(f"{COLLECTION} {d['_id']}: {e}") from e
    return out


def main() -> int:
    print(seed_guardrails())
    return 0


if __name__ == "__main__":
    sys.exit(main())
