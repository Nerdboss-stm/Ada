import copy
import json
from pathlib import Path
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import loop.controller as controller
from core.contracts import ChatResult, Harness
from loop.edits import (
    GAIT_VALUE_REASON, TOOL_WHITELIST, EditError, EditProposal, apply_edit, states_gait_values,
)

BANNED = ("power", "glitch", "exploit", "cheat")
SPEED = "body speed exceeds physical bound; exploits the simulator"
MOTORS = "exceeds Ada rated motors"
NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def harness():
    return Harness(
        rules=["Keep the torso level."],
        context_policy={"past_attempts": 2, "telemetry": ["tilt"]},
        tools=["read_task", "submit_gait"],
        model_per_step={"agent": "agent_v0"},
        engine={"temperature": 0, "max_attempts": 3},
    )


def head():
    return {"_id": "v3", "parent": "v2", "status": "accepted",
            "harness": harness().model_dump(), "created_at": NOW}


def sensor(sanity=False):
    codes = [
        {"code": "fell", "count": 4, "examples": ["tr-fell-1", "tr-fell-2"]},
        {"code": "short", "count": 2, "examples": ["tr-short-1"]},
    ]
    if sanity:
        codes.insert(0, {"code": "sanity", "count": 1, "examples": ["tr-sanity-1"],
                         "violations": [MOTORS]})
    return {"version_id": "v3", "split": "train", "n": 12, "success_rate": 0.5,
            "mean_distance_m": 1.8, "cost_usd": 0.01, "failure_codes": codes,
            "worst_tasks": [{"task_id": "t1", "success_rate": 0.0, "mean_distance_m": 0.4}]}


def edit(i, verdict="accepted", reason=None):
    return {"_id": f"edit-{i:02d}", "round_id": "r1", "from_version": "v2", "origin": "model",
            "primitive": "rules", "op": "add", "path": "", "old": None, "new": f"Rule {i}.",
            "rationale": "Try it.", "predicted_delta": 0.05, "predicted_delta_m": 0.2,
            "actual_delta": 0.01,
            "verdict": verdict, "reason": reason, "created_at": NOW}


def prop(**over):
    base = {"primitive": "engine", "op": "set", "path": "temperature", "old": 0, "new": 0.2,
            "predicted_delta": 0.05, "predicted_delta_m": 0.15,
            "rationale": "Some variety may help the agent escape a bad gait.",
            "evidence_trace_ids": ["tr-fell-1"]}
    return {**base, **over}


@pytest.fixture
def fake(monkeypatch):
    """Patch cached_chat and emit; set `.reply` to control the model's answer."""
    class Fake:
        reply = json.dumps({"edits": []})
        calls = []
        events = []

    def chat(role, messages, tools=None, **params):
        Fake.calls.append({"role": role, "messages": messages, "params": params})
        return ChatResult(role=role, model_id="m", prompt_hash="h",
                          message={"role": "assistant", "content": Fake.reply},
                          usage={}, cost_usd=0.02, cached=False)

    def emit(stage, status, payload=None, **kw):
        Fake.events.append({"stage": stage, "status": status, "payload": payload, **kw})
        return "id"

    Fake.calls, Fake.events = [], []
    monkeypatch.setattr(controller, "cached_chat", chat)
    monkeypatch.setattr(controller, "emit", emit)
    return Fake


def prompt_text(fake):
    return "\n".join(m["content"] for m in fake.calls[-1]["messages"])


# --- validation -------------------------------------------------------------

def test_valid_proposal_passes():
    p = EditProposal.model_validate(prop())
    assert p.primitive == "engine" and p.new == 0.2


@pytest.mark.parametrize("over", [
    {"primitive": "tools", "op": "add", "path": "", "old": None, "new": "shell_exec"},
    {"primitive": "tools", "op": "remove", "path": "", "old": "run_anything", "new": None},
    {"primitive": "tools", "op": "set", "path": "", "old": "read_task", "new": "web_search"},
    {"primitive": "model_per_step", "op": "set", "path": "agent", "old": "agent_v0", "new": "frontier"},
    {"primitive": "model_per_step", "op": "set", "path": "agent", "old": "agent_v0", "new": "controller"},
    {"primitive": "model_per_step", "op": "remove", "path": "agent", "old": "agent_v0", "new": None},
    {"primitive": "engine", "op": "set", "path": "top_p", "old": None, "new": 0.9},
    {"primitive": "engine", "op": "set", "path": "temperature", "old": 0, "new": -0.1},
    {"primitive": "engine", "op": "set", "path": "temperature", "old": 0, "new": 1.5},
    {"primitive": "engine", "op": "set", "path": "temperature", "old": 0, "new": True},
    {"primitive": "engine", "op": "set", "path": "max_attempts", "old": 3, "new": 0},
    {"primitive": "engine", "op": "set", "path": "max_attempts", "old": 3, "new": 7},
    {"primitive": "engine", "op": "set", "path": "max_attempts", "old": 3, "new": 4.5},
    {"primitive": "engine", "op": "add", "path": "temperature", "old": None, "new": 0.5},
    {"primitive": "engine", "op": "remove", "path": "temperature", "old": 0, "new": None},
    {"primitive": "context_policy", "op": "set", "path": "", "old": 2, "new": 3},
    {"primitive": "judge", "op": "set", "path": "x", "old": None, "new": 1},
    {"primitive": "rules", "op": "replace", "path": "", "old": "a", "new": "b"},
    {"primitive": "rules", "op": "add", "path": "", "old": None, "new": ""},
    {"rationale": " ".join(["word"] * 26)},
    {"rationale": "Raise temperature. It helps."},
    {"rationale": ""},
    {"predicted_delta": 1.5},
    {"predicted_delta": float("nan")},
    {"predicted_delta_m": None},
    {"predicted_delta_m": "0.3"},
    {"predicted_delta_m": True},
    {"predicted_delta_m": float("inf")},
    {"surprise": 1},
])
def test_validation_rejects(over):
    with pytest.raises(ValidationError):
        EditProposal.model_validate(prop(**over))


def test_predicted_delta_m_is_required():
    base = prop()
    del base["predicted_delta_m"]
    with pytest.raises(ValidationError):
        EditProposal.model_validate(base)
    assert EditProposal.model_validate(prop(predicted_delta_m=-1)).predicted_delta_m == -1


# --- gait-number guardrail ----------------------------------------------------

@pytest.mark.parametrize("text", [
    "Use kp 5 on slopes.", "Try 1.5Hz strides.", "Set the stride frequency to 2.", "Lower the offsets to 0.3",
    "KD of 0.4 steadies the legs.", "0.8 amplitude on the hips.", "phase=3.14 on leg four",
    '{"frequency_hz":1.5}', "tilt_gain 1",
])
def test_states_gait_values_catches(text):
    assert states_gait_values(text)


@pytest.mark.parametrize("text", [
    "Keep the torso level.", "Use at most 3 attempts.", "Lower the stride frequency on steep slopes.",
    "Prefer a slower stride frequency on slopes steeper than 4 degrees", "Preview twice, then submit.",
    "backward 3 times",
])
def test_states_gait_values_allows(text):
    assert not states_gait_values(text)


@pytest.mark.parametrize("over", [
    {"primitive": "rules", "op": "add", "path": "", "old": None, "new": "Use kp 5 on slopes."},
    {"primitive": "rules", "op": "set", "path": "", "old": "Keep the torso level.", "new": "Keep kd near 0.4."},
])
def test_gait_values_in_added_text_rejected(over):
    with pytest.raises(ValidationError, match=GAIT_VALUE_REASON):
        EditProposal.model_validate(prop(**over))


def test_gait_values_only_in_removed_text_allowed():
    p = EditProposal.model_validate(prop(primitive="rules", op="remove", path="", old="Use kp 5.", new=None))
    assert p.added_text() == ""


def test_rationale_with_decimal_is_one_sentence():
    EditProposal.model_validate(prop(rationale="Moving temperature to 0.2 adds variety."))


def test_list_primitive_path_is_normalized():
    p = EditProposal.model_validate(prop(primitive="tools", op="add", path="tools", old=None,
                                         new="preview_run"))
    assert p.path == ""


# --- apply_edit -------------------------------------------------------------

@pytest.mark.parametrize("over, check", [
    (dict(primitive="rules", op="add", path="", old=None, new="Take short steps."),
     lambda h: h.rules == ["Keep the torso level.", "Take short steps."]),
    (dict(primitive="rules", op="remove", path="", old="Keep the torso level.", new=None),
     lambda h: h.rules == []),
    (dict(primitive="rules", op="set", path="", old="Keep the torso level.", new="Keep it low."),
     lambda h: h.rules == ["Keep it low."]),
    (dict(primitive="tools", op="add", path="", old=None, new="preview_run"),
     lambda h: h.tools == ["read_task", "submit_gait", "preview_run"]),
    (dict(primitive="tools", op="remove", path="", old="read_task", new=None),
     lambda h: h.tools == ["submit_gait"]),
    (dict(primitive="model_per_step", op="add", path="judge", old=None, new="agent_v0"),
     lambda h: h.model_per_step == {"agent": "agent_v0", "judge": "agent_v0"}),
    (dict(primitive="engine", op="set", path="max_attempts", old=3, new=5),
     lambda h: h.engine == {"temperature": 0, "max_attempts": 5}),
])
def test_apply_each_op(over, check):
    assert check(apply_edit(harness(), EditProposal.model_validate(prop(**over))))


@pytest.mark.parametrize("over", [
    dict(primitive="rules", op="add", path="", old=None, new="Keep the torso level."),
    dict(primitive="rules", op="remove", path="", old="Not a rule.", new=None),
    dict(primitive="tools", op="add", path="", old=None, new="read_task"),
    dict(primitive="tools", op="set", path="", old="read_task", new="submit_gait"),
    dict(primitive="model_per_step", op="add", path="agent", old=None, new="agent_v0"),
    dict(primitive="model_per_step", op="set", path="agent", old="agent_x", new="agent_v0"),
    dict(primitive="engine", op="set", path="max_attempts", old=5, new=4),
    dict(primitive="engine", op="set", path="temperature", old=0, new=0),
])
def test_apply_rejects_stale_or_noop(over):
    with pytest.raises(EditError):
        apply_edit(harness(), EditProposal.model_validate(prop(**over)))


def test_apply_edit_is_pure():
    h = harness()
    before = copy.deepcopy(h.model_dump())
    p = EditProposal.model_validate(prop(primitive="rules", op="set", path="",
                                         old="Keep the torso level.", new="Keep it low."))
    new_before = copy.deepcopy(p.new)
    out = apply_edit(h, p)
    assert h.model_dump() == before
    assert out.rules == ["Keep it low."]
    out.context_policy["telemetry"].append("x")
    out.rules.append("y")
    out.engine["temperature"] = 1
    assert h.model_dump() == before
    assert p.new == new_before


# --- propose ----------------------------------------------------------------

def test_duplicates_invalid_stale_and_extras_dropped(fake):
    long_rule = "Before submitting, preview the gait twice on practice seeds and keep the one with the lower tilt."
    reply = [
        prop(),                                                                          # valid
        prop(rationale="Same edit, other words."),                                        # duplicate
        prop(primitive="rules", op="add", path="", old=None, new=long_rule),              # valid, large
        prop(primitive="tools", op="add", path="", old=None, new="preview_run"),          # valid
        prop(primitive="tools", op="add", path="", old=None, new="preview_run"),          # duplicate
        prop(primitive="model_per_step", op="add", path="judge", old=None, new="agent_v0"),  # valid
        prop(primitive="model_per_step", op="set", path="agent", old="agent_v0", new="frontier"),  # invalid
        prop(primitive="engine", op="set", path="max_attempts", old=5, new=4),            # stale
        prop(primitive="rules", op="remove", path="", old="Keep the torso level.", new=None),  # valid
        prop(primitive="tools", op="add", path="", old=None, new="get_contact_log"),      # valid
        prop(primitive="tools", op="add", path="", old=None, new="list_my_attempts"),     # valid
        prop(primitive="engine", op="set", path="max_attempts", old=3, new=4),            # valid
        prop(primitive="model_per_step", op="set", path="agent", old="agent_v0", new="agent_v0.alt"),  # invalid: not cheap
        "not an edit",                                                                    # invalid
        {k: v for k, v in prop(new=0.3).items() if k != "predicted_delta_m"},             # no meters
        prop(primitive="rules", op="add", path="", old=None, new="Use kp 5."),            # gait value
    ]
    fake.reply = "```json\n" + json.dumps({"edits": reply}) + "\n```"
    out = controller.propose(head(), sensor(), [], round_id="r1")

    valid = [EditProposal.model_validate(r) for i, r in enumerate(reply) if i in (0, 2, 3, 5, 8, 9, 10, 11)]
    assert len(out) == controller.MAX_EDITS
    assert len({p.key() for p in out}) == len(out)
    sizes = [p.change_size() for p in out]
    assert sizes == sorted(sizes)
    assert sizes == sorted(p.change_size() for p in valid)[:controller.MAX_EDITS]
    assert all(p.new != long_rule for p in out)
    assert len(fake.calls) == 1 and fake.calls[0]["role"] == "controller"
    assert len(fake.events) == 1
    ev = fake.events[0]
    assert ev["stage"] == "controller" and ev["status"] == "info"
    assert ev["payload"]["count"] == 6 and ev["payload"]["received"] == len(reply)
    assert ev["round_id"] == "r1" and ev["version_id"] == "v3"


@pytest.mark.parametrize("over", [
    dict(primitive="context_policy", op="set", path="past_attempts", old=2, new=3),
    dict(primitive="context_policy", op="add", path="history", old=None, new=["tilt"]),
    dict(primitive="context_policy", op="remove", path="past_attempts", old=2, new=None),
])
def test_context_policy_proposal_dropped(fake, over):
    """A16: harness/agent.py does not read context_policy, so edits to it are dropped."""
    with pytest.raises(ValidationError):
        EditProposal.model_validate(prop(**over))
    fake.reply = json.dumps({"edits": [prop(**over), prop()]})
    out = controller.propose(head(), sensor(), [])
    assert [p.primitive for p in out] == ["engine"]
    assert fake.events[0]["payload"]["count"] == 1


def test_bare_list_reply_accepted(fake):
    fake.reply = "Here you go: " + json.dumps([prop()])
    out = controller.propose(head(), sensor(), [])
    assert [p.path for p in out] == ["temperature"]


def test_unparseable_reply_gives_no_edits(fake):
    fake.reply = "I would raise the temperature."
    assert controller.propose(head(), sensor(), []) == []
    assert [(e["status"], e["payload"]["count"]) for e in fake.events] == [("fail", 0)]


def test_over_budget_aborts_before_calling(fake, monkeypatch):
    pricey = {"id": "x", "usd_per_mtok_in": 1e6, "usd_per_mtok_out": 1e6}
    monkeypatch.setattr(controller, "models", lambda: {"controller": pricey})
    with pytest.raises(controller.ControllerBudgetError):
        controller.propose(head(), sensor(), [])
    assert fake.calls == []
    assert [e["status"] for e in fake.events] == ["fail"]
    assert fake.events[0]["payload"]["estimate_usd"] > controller.BUDGET_USD


def test_real_prices_fit_budget(fake):
    msgs = controller.render_messages(controller._version(head()), sensor(True),
                                      [edit(i) for i in range(10)])
    assert 0 < controller.estimate_usd(msgs) < controller.BUDGET_USD


# --- prompt -----------------------------------------------------------------

def test_static_prompt_never_names_banned_words(fake):
    controller.propose(head(), sensor(sanity=False), [])
    text = prompt_text(fake).lower()
    for word in BANNED:
        assert word not in text
    for part in controller._static_text():
        for word in BANNED:
            assert word not in part.lower()


def test_prompt_contents(fake):
    history = [edit(i) for i in range(10)] + [edit(10, verdict="rejected", reason=SPEED)]
    controller.propose(head(), sensor(sanity=True), history)
    text = prompt_text(fake)
    assert "0.7 × train reliability + 0.3 × normalized mean distance" in text
    assert "predicted_delta_m" in text and "meters" in text
    assert '"predicted_delta_m": 0.2' in text  # past bets are shown
    for tool in TOOL_WHITELIST:
        assert tool in text
    for trace in ("tr-fell-1", "tr-fell-2", "tr-short-1", "tr-sanity-1"):
        assert trace in text
    assert json.dumps(harness().model_dump()) in text
    assert "edit-00" not in text
    for i in range(1, 11):
        assert f"edit-{i:02d}" in text
    assert SPEED in text     # gate feedback goes in verbatim
    assert MOTORS in text
    assert '"verdict": "rejected"' in text


def test_prompt_names_four_editable_parts(fake):
    controller.propose(head(), sensor(), [])
    text = prompt_text(fake)
    section = text.split("## Harness primitives and how to edit them\n", 1)[1].split("\n## ", 1)[0]
    bullets = [line[2:].split(":", 1)[0] for line in section.splitlines() if line.startswith("- ")]
    assert bullets == ["rules", "tools", "model_per_step", "engine"]
    assert "context_policy" not in section
    assert '"primitive": "rules|tools|model_per_step|engine"' in text


def test_assert_clean_catches_template_words():
    with pytest.raises(AssertionError):
        controller._assert_clean("Use more POWER.")


# --- holdout isolation --------------------------------------------------------

HOLDOUT = 0.8333


def test_prompt_never_shows_holdout(fake):
    h = head() | {"metrics": {"train_reliability": 0.4167, "holdout_reliability_80": HOLDOUT,
                              "mean_distance_m": 1.73, "cost_per_run_usd": 0.0012, "n": 18}}
    history = [edit(i) for i in range(3)]
    controller.propose(h, sensor(), history)
    text = prompt_text(fake)
    assert str(HOLDOUT) not in text and "83.33" not in text
    assert "holdout" not in text.lower()
    assert "18" not in text  # the holdout run count


def test_loop_selection_never_reads_holdout():
    loop_dir = Path(controller.__file__).parent
    for name in ("controller.py", "sensor.py", "edits.py", "subset.py"):
        code = (loop_dir / name).read_text()
        assert "holdout_reliability_80" not in code, name
        assert '"holdout"' not in code and "'holdout'" not in code, name
