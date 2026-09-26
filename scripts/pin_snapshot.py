"""Pin the demo scenes (NOTES [A12]): one `snapshots` document naming exactly what plays.

    uv run python -m scripts.pin_snapshot [--db ada]

The loop keeps writing after the pin; the scenes read these ids, so only live counters move.

- best_version: loop.run.pick_leader over baseline and accepted versions (train only).
- attempt_edit_ids: ui/lib/attempts.ts, ported line for line (toAttempt, attemptsOf,
  orderAttempts): the worst bet, the largest measured gain, then every other kept edit.
- cheat_edit_id: what api/cheat shows as the card: the newest rejected non-cli edit whose
  reason names "rated torque" or "rated motors" and that has a frames_id.
- swaps_id / scoreboard_id: the newest `swaps` (captured_at) and `scoreboard` (created_at).
- ghosts: the GHOST_LIMIT newest accepted versions with showcase frames, plus v0 and the
  frontier, oldest first (ui/lib/ghosts.ts sortVersions).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from typing import Any

from core.contracts import Snapshot, SnapshotGhost
from core.db import ADA, db
from loop.run import pick_leader

V0 = "v0"
FRONTIER = "frontier"
GHOST_LIMIT = 20
OVERRATED_REASON = {"$regex": "rated torque|rated motors", "$options": "i"}  # as api/cheat
_FRAMES_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # as ui/lib/attempts.ts FRAMES_ID
_VERSION_NUM = re.compile(r"^v(\d+)$")
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _now_ms() -> datetime:
    """UTC now at Mongo's millisecond precision, so the stored document equals the returned one."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _scope(prefix: str | None) -> dict[str, Any]:
    """Tests pin on the shared ada_test: restrict every query to their own _ids."""
    return {"_id": {"$regex": f"^{re.escape(prefix)}"}} if prefix else {}


def _created(doc: dict[str, Any], key: str = "created_at") -> datetime:
    t = doc.get(key)
    return t if isinstance(t, datetime) else _EPOCH


def _version_order(doc: dict[str, Any]) -> tuple[int, datetime, str]:
    """ui/lib/ghosts.ts sortVersions: version number, then created_at, then _id."""
    m = _VERSION_NUM.match(str(doc["_id"]))
    return (int(m.group(1)) if m else sys.maxsize, _created(doc), str(doc["_id"]))


# --- attempts: ui/lib/attempts.ts ------------------------------------------------

def _train_mean(v: dict[str, Any] | None) -> float | None:
    m = ((v or {}).get("metrics") or {}).get("train_mean_distance_m")
    return float(m) if _finite(m) else None


def to_attempt(edit: dict[str, Any], versions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """toAttempt: None unless the attempt can be drawn from stored numbers alone."""
    frames_id, bet = edit.get("attempt_frames_id"), edit.get("predicted_delta_m")
    if not isinstance(frames_id, str) or not _FRAMES_ID.match(frames_id) or not _finite(bet):
        return None
    if edit.get("verdict") not in ("accepted", "rejected"):
        return None
    parent = _train_mean(versions.get(edit.get("from_version") or ""))
    if parent is None:
        return None
    candidate = _train_mean(versions.get(edit.get("to_version") or ""))
    actual = edit.get("actual_delta_m")
    measured = float(actual) if _finite(actual) else candidate - parent if candidate is not None else None
    if measured is None:
        return None
    return {"edit_id": str(edit["_id"]), "created_at": _created(edit), "bet": float(bet),
            "measured": measured, "kept": edit.get("verdict") == "accepted",
            "verdict": edit.get("verdict")}


def attempts_of(edits: list[dict[str, Any]], versions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """attemptsOf: every drawable attempt, oldest first (created_at, then _id)."""
    out = [a for a in (to_attempt(e, versions) for e in edits) if a is not None]
    return sorted(out, key=lambda a: (a["created_at"], a["edit_id"]))


def order_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """orderAttempts: worst bet, largest measured gain, every other kept edit. Ties keep the older."""
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    worst = None
    for a in attempts:
        if a["measured"] < 0 and (worst is None or a["bet"] > worst["bet"]):
            worst = a
    if worst:
        out.append({**worst, "slot": "worst bet"})
        used.add(worst["edit_id"])
    gain = None
    for a in attempts:
        if a["edit_id"] not in used and a["measured"] > 0 and (gain is None or a["measured"] > gain["measured"]):
            gain = a
    if gain:
        out.append({**gain, "slot": "largest gain"})
        used.add(gain["edit_id"])
    out += [{**a, "slot": "kept"} for a in attempts if a["kept"] and a["edit_id"] not in used]
    return out


# --- the snapshot ------------------------------------------------------------------

def _ghosts(versions: list[dict[str, Any]], v0: str, frontier: str) -> list[SnapshotGhost]:
    accepted = sorted((v for v in versions if v.get("status") == "accepted"
                       and isinstance(v.get("showcase_frames_id"), str) and v["showcase_frames_id"]),
                      key=_version_order)[-GHOST_LIMIT:]
    ends = [v for v in versions if v["_id"] in (v0, frontier) and v not in accepted]
    return [SnapshotGhost(version_id=str(v["_id"]), frames_id=v.get("showcase_frames_id") or None)
            for v in sorted(accepted + ends, key=_version_order)]


def _newest_id(coll: Any, key: str, scope: dict[str, Any]) -> str | None:
    doc = coll.find_one(scope, {"_id": 1}, sort=[(key, -1)])
    return str(doc["_id"]) if doc else None


def build_snapshot(*, v0: str = V0, frontier: str = FRONTIER, prefix: str | None = None,
                   db_name: str = ADA) -> tuple[Snapshot, list[dict[str, Any]]]:
    """Insert one snapshots document; return it and the ordered attempts behind attempt_edit_ids."""
    d = db(db_name)
    scope = _scope(prefix)
    versions = list(d.versions.find(scope, {"status": 1, "metrics": 1, "showcase_frames_id": 1,
                                            "created_at": 1}))
    try:
        best = pick_leader(versions)
    except LookupError:
        best = None

    edits = list(d.edits.find(scope, {"from_version": 1, "to_version": 1, "predicted_delta_m": 1,
                                      "actual_delta_m": 1, "attempt_frames_id": 1, "verdict": 1,
                                      "created_at": 1}))
    ordered = order_attempts(attempts_of(edits, {str(v["_id"]): v for v in versions}))

    cheat = d.edits.find_one(
        {**scope, "verdict": "rejected", "origin": {"$ne": "cli"}, "reason": OVERRATED_REASON,
         "frames_id": {"$type": "string", "$ne": ""}},
        {"_id": 1}, sort=[("created_at", -1)])

    ids = {str(v["_id"]) for v in versions}
    pinned_at = _now_ms()
    snap = Snapshot(
        _id=f"snapshot-{pinned_at:%Y%m%dT%H%M%S%f}", pinned_at=pinned_at, best_version=best,
        v0=v0 if v0 in ids else None, frontier=frontier if frontier in ids else None,
        attempt_edit_ids=[a["edit_id"] for a in ordered],
        cheat_edit_id=str(cheat["_id"]) if cheat else None,
        swaps_id=_newest_id(d.swaps, "captured_at", scope),
        scoreboard_id=_newest_id(d.scoreboard, "created_at", scope),
        ghosts=_ghosts(versions, v0, frontier),
    )
    d.snapshots.insert_one(snap.model_dump(by_alias=True))
    return snap, ordered


def _signed(x: float) -> str:
    return f"{x:+.2f} m"


def summary(snap: Snapshot, ordered: list[dict[str, Any]], db_name: str = ADA) -> str:
    lines = [f"pinned {snap.id}  at {snap.pinned_at:%Y-%m-%d %H:%M:%S} UTC",
             f"  best      {snap.best_version or '-'}   (train leader)",
             f"  v0        {snap.v0 or '-'}",
             f"  frontier  {snap.frontier or '-'}",
             f"  attempts  {len(ordered)} in play order"]
    for i, a in enumerate(ordered, 1):
        lines.append(f"    {i:>2}. {a['edit_id']}  bets {_signed(a['bet'])}  "
                     f"measured {_signed(a['measured'])}  {a['verdict']}  ({a['slot']})")
    reason = ""
    if snap.cheat_edit_id:
        doc = db(db_name).edits.find_one({"_id": snap.cheat_edit_id}, {"reason": 1}) or {}
        reason = f"   \"{doc.get('reason', '')}\""
    lines += [f"  cheat     {snap.cheat_edit_id or '-'}{reason}",
              f"  swaps     {snap.swaps_id or '-'}",
              f"  scoreboard {snap.scoreboard_id or '-'}",
              f"  ghosts    {len(snap.ghosts)}"]
    lines += [f"    {g.version_id:<12} {g.frames_id or '(no showcase frames)'}" for g in snap.ghosts]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.pin_snapshot")
    parser.add_argument("--db", default=ADA)
    args = parser.parse_args(argv)
    snap, ordered = build_snapshot(db_name=args.db)
    print(summary(snap, ordered, args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
