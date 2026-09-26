"""Run N rounds, each from the current leader (NOTES [A11]).

    python -m loop.run --rounds N [--budget-usd 3.0] [--budget-s 600] [--db ada]

The leader is the baseline or accepted version with the highest train_reliability, then the
highest metrics.train_mean_distance_m (None, as for a version that left a gate task without a
gait, ranks below every distance), then the lower version number. Holdout numbers are never
read. Each round prints one line: round id, head in -> head out, accepted, rejected, spend,
seconds. An exception stops the run; loop.actuator.run_round has already closed that round,
and its line shows the stop reason.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from typing import Any

from core.db import ADA, db
from loop import actuator

LEADER_STATUSES = ("baseline", "accepted")
_VERSION_NUM = re.compile(r"^v(\d+)$")


def _rank(doc: dict[str, Any]) -> tuple[float, float, int]:
    m = doc.get("metrics") or {}
    dist = m.get("train_mean_distance_m")
    num = _VERSION_NUM.match(str(doc["_id"]))
    return (float(m.get("train_reliability") or 0.0),
            float("-inf") if dist is None else float(dist),
            -int(num.group(1)) if num else -sys.maxsize)


def pick_leader(docs: Iterable[dict[str, Any]]) -> str:
    """The leader's _id among versions documents; LookupError when none qualifies."""
    eligible = [d for d in docs if d.get("status") in LEADER_STATUSES and d.get("metrics")]
    if not eligible:
        raise LookupError("no baseline or accepted version with metrics")
    return str(max(eligible, key=_rank)["_id"])


def _versions(db_name: str) -> list[dict[str, Any]]:
    return list(db(db_name).versions.find(
        {"status": {"$in": list(LEADER_STATUSES)}},
        {"status": 1, "metrics.train_reliability": 1, "metrics.train_mean_distance_m": 1},
    ))


def leader(db_name: str = ADA) -> str:
    return pick_leader(_versions(db_name))


def _last_round(db_name: str) -> dict[str, Any]:
    return db(db_name).rounds.find_one({}, sort=[("opened_at", -1)]) or {}


def _ids(ids: list[str]) -> str:
    return "[" + ",".join(ids) + "]"


def round_line(state: dict[str, Any]) -> str:
    return (f"{state['round_id']}  {state['start_head']} → {state['head']}  "
            f"accepted {_ids(state['accepted'])}  rejected {_ids(state['rejected'])}  "
            f"spend ${state['spent_usd']:.4f}  {state['elapsed_s']:.1f} s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="loop.run", description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--budget-s", type=float, default=600)
    ap.add_argument("--db", default=ADA)
    args = ap.parse_args(argv)
    if args.rounds < 1:
        ap.error("--rounds must be >= 1")

    for i in range(args.rounds):
        head = leader(args.db)
        try:
            state = actuator.run_round(head, db_name=args.db, budget_usd=args.budget_usd,
                                       budget_s=args.budget_s)
        except actuator.RoundOpenError as e:
            print(f"round {i + 1}/{args.rounds} refused: {e}", flush=True)
            return 1
        except Exception as e:  # run_round closed the round before re-raising
            last = _last_round(args.db)
            print(f"{last.get('_id', '?')}  {head} → stopped  "
                  f"{last.get('stop_reason') or f'{type(e).__name__}: {e}'}", flush=True)
            return 1
        print(round_line(state), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
