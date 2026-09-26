"""Reject edits left without a verdict by a round that stopped (NOTES [A13]).

    uv run python -m scripts.fix_pending_edits [--db ada] [--dry-run]

Every edits document with a null verdict whose round is closed gets verdict "rejected" and
reason REASON. Edits in open rounds, or in no known round, are left alone. When the edit's
to_version has no versions document, a stub is written (status rejected, the parent's
harness, parent = from_version, no metrics) so next_version_id never hands that id out
again and no two edits point at one version. Prints one line per edit.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from core.contracts import Version
from core.db import ADA, db

REASON = "harness error: round stopped before a verdict"


def _stub(edit: dict[str, Any], db_name: str, dry_run: bool) -> str:
    """'exists', 'written' or 'no parent' for the edit's to_version."""
    d = db(db_name)
    vid = edit.get("to_version")
    if not vid or d.versions.find_one({"_id": vid}, {"_id": 1}):
        return "exists"
    parent = d.versions.find_one({"_id": edit.get("from_version")}, {"harness": 1})
    if parent is None:
        return "no parent"
    stub = Version(_id=vid, parent=edit["from_version"], status="rejected",
                   harness=parent["harness"], created_at=datetime.now(timezone.utc))
    if not dry_run:
        d.versions.insert_one(stub.model_dump(by_alias=True))
    return "written"


def fix(db_name: str = ADA, dry_run: bool = False, round_ids: list[str] | None = None,
        ) -> list[dict[str, Any]]:
    """[{edit_id, round_id, to_version, stub}] for every edit rejected, in _id order.
    round_ids limits the scan to those rounds (tests); None scans every round."""
    d = db(db_name)
    query: dict[str, Any] = {"verdict": None}
    if round_ids is not None:
        query["round_id"] = {"$in": round_ids}
    pending = list(d.edits.find(query).sort("_id", 1))
    seen = list({e.get("round_id") for e in pending} - {None})
    closed = {r["_id"] for r in d.rounds.find({"_id": {"$in": seen}, "status": "closed"}, {"_id": 1})}
    out = []
    for e in pending:
        if e.get("round_id") not in closed:
            continue
        if not dry_run:
            d.edits.update_one({"_id": e["_id"], "verdict": None},
                               {"$set": {"verdict": "rejected", "reason": REASON}})
        out.append({"edit_id": e["_id"], "round_id": e["round_id"],
                    "to_version": e.get("to_version"), "stub": _stub(e, db_name, dry_run)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.fix_pending_edits")
    parser.add_argument("--db", default=ADA)
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args(argv)
    rows = fix(args.db, dry_run=args.dry_run)
    for row in rows:
        print(f"{row['edit_id']}  rejected  to_version {row['to_version']}  stub {row['stub']}")
    print(f"{len(rows)} edit(s) rejected{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
