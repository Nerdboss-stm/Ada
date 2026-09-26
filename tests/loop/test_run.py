"""loop.run: leader by train metrics only, one line per round, early stop on error."""

import pytest

from core.db import ADA_TEST
from loop import actuator
from loop import run as loop_run


def vdoc(vid, status="accepted", rel=0.5, dist=None, holdout=0.0, mean=0.0):
    return {"_id": vid, "status": status, "metrics": {
        "train_reliability": rel, "train_mean_distance_m": dist,
        "holdout_reliability_80": holdout, "mean_distance_m": mean}}


def test_leader_uses_train_metrics_never_holdout():
    docs = [
        vdoc("v0", "baseline", rel=0.5, dist=1.0),
        vdoc("v3", rel=0.5, dist=2.0),
        vdoc("v4", rel=0.5, dist=None, holdout=1.0, mean=9.0),  # best holdout, no train distance
        vdoc("v6", rel=0.5, dist=2.0),                          # ties v3: the older wins
        vdoc("v5", "rejected", rel=1.0, dist=5.0),
        vdoc("frontier", "frontier", rel=0.9, dist=4.0),
        vdoc("v7", "candidate", rel=1.0, dist=5.0),
    ]
    assert loop_run.pick_leader(docs) == "v3"


def test_reliability_outranks_distance_and_none_never_leads_on_distance():
    assert loop_run.pick_leader([vdoc("v1", rel=0.5, dist=3.0), vdoc("v2", rel=0.67, dist=0.1)]) == "v2"
    # v2 never submitted on some gate task: None ranks below any distance, even a negative one
    assert loop_run.pick_leader([vdoc("v1", rel=0.0, dist=-0.4), vdoc("v2", rel=0.0, dist=None)]) == "v1"
    assert loop_run.pick_leader([vdoc("v9", dist=None), vdoc("v2", dist=None)]) == "v2"


def test_no_eligible_version_raises():
    with pytest.raises(LookupError):
        loop_run.pick_leader([vdoc("v5", "rejected"), {"_id": "v0", "status": "baseline"}])


@pytest.fixture
def rounds(monkeypatch):
    """In-memory versions; run_round accepts one better child per round, or raises."""

    class R:
        docs = [vdoc("v0", "baseline", rel=0.5, dist=1.0), vdoc("v1", "rejected", rel=1.0, dist=9.0)]
        heads: list[str] = []
        kwargs: list[dict] = []
        fail_on: int | None = None

    def run_round(head, db_name, budget_usd, budget_s):
        R.heads.append(head)
        R.kwargs.append({"db_name": db_name, "budget_usd": budget_usd, "budget_s": budget_s})
        i = len(R.heads)
        if R.fail_on == i:
            raise RuntimeError("boom")
        child = f"v{10 + i}"
        R.docs.append(vdoc(child, rel=0.5, dist=1.0 + i))
        return {"round_id": f"r-{i}", "start_head": head, "head": child, "accepted": [child],
                "rejected": [f"v{20 + i}"], "spent_usd": 0.12346, "elapsed_s": 41.26}

    monkeypatch.setattr(loop_run, "_versions", lambda db_name: list(R.docs))
    monkeypatch.setattr(actuator, "run_round", run_round)
    monkeypatch.setattr(loop_run, "_last_round",
                        lambda db_name: {"_id": f"r-{len(R.heads)}", "stop_reason": "error: RuntimeError: boom"})
    return R


def test_each_round_starts_from_the_leader_and_prints_one_line(rounds, capsys):
    assert loop_run.main(["--rounds", "3", "--db", ADA_TEST, "--budget-usd", "1.5"]) == 0

    assert rounds.heads == ["v0", "v11", "v12"]
    assert {k["db_name"] for k in rounds.kwargs} == {ADA_TEST}
    assert {k["budget_usd"] for k in rounds.kwargs} == {1.5}
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "r-1  v0 → v11  accepted [v11]  rejected [v21]  spend $0.1235  41.3 s",
        "r-2  v11 → v12  accepted [v12]  rejected [v22]  spend $0.1235  41.3 s",
        "r-3  v12 → v13  accepted [v13]  rejected [v23]  spend $0.1235  41.3 s",
    ]


def test_exception_stops_early_with_the_round_line(rounds, capsys):
    rounds.fail_on = 2
    assert loop_run.main(["--rounds", "5", "--db", ADA_TEST]) == 1

    assert rounds.heads == ["v0", "v11"]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[1] == "r-2  v11 → stopped  error: RuntimeError: boom"
