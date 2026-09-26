"""Harness agent (SPEC §3): a LangGraph StateGraph checkpointed by MongoDBSaver.

    run_episode(task, version_id, harness) -> EpisodeResult

The model node calls core.llm.cached_chat with the role from harness.model_per_step;
the tool node runs only tools listed in harness.tools, which must be a subset of
TOOL_WHITELIST (CONTRACTS §6). Previews live in the checkpointed state for one episode. Every submit_gait call counts
as an attempt; the episode ends on the first valid gait or at engine.max_attempts. preview_run calls never count
as attempts; at most MAX_PREVIEWS run per episode, later ones return PREVIEW_LIMIT_MSG. The recursion limit leaves
room for every preview and every attempt. Each step is stored in `traces.raw_steps` (trace_id = thread id), with
the episode's end_reason and preview count.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from core import llm
from core.contracts import Harness, Task, Trace
from core.db import ADA, ADA_CKPT, client, db
from harness import preview
from sim.gait import Gait, neutral_offsets

SYSTEM_PROMPT = "Write a gait for Ada for this task."
AGENT_STEP = "agent"  # key of harness.model_per_step used by the model node
DEFAULT_MAX_ATTEMPTS = 3
MAX_PREVIEWS = 6  # preview_run calls per episode, valid or not
PREVIEW_LIMIT_MSG = "preview limit reached; submit your gait now"
EXTRA_TURNS = 2  # read_task turn + one spare turn, on top of one turn per preview and per attempt
TOOL_WHITELIST = (  # CONTRACTS §6, exactly 6; immutable
    "read_task", "submit_gait", "preview_run", "get_contact_log", "list_my_attempts", "lookup_skill",
)

V0_HARNESS = Harness(
    rules=[],
    context_policy={},
    tools=["read_task", "submit_gait"],
    model_per_step={AGENT_STEP: "agent_v0"},
    engine={"temperature": 0, "max_attempts": DEFAULT_MAX_ATTEMPTS},
)


# --- tools ------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCtx:
    task: Task
    previews: list[dict[str, Any]]  # this episode's previews, oldest first: {gait, result, contact_log}


@dataclass(frozen=True)
class ToolOut:
    result: dict[str, Any]
    gait: Gait | None = None            # a valid submission
    preview: dict[str, Any] | None = None  # a new preview record


@dataclass(frozen=True)
class ToolDef:
    spec: dict[str, Any]
    run: Callable[[ToolCtx, dict[str, Any]], ToolOut]
    is_attempt: bool


def _read_task(ctx: ToolCtx, args: dict[str, Any]) -> ToolOut:
    task = ctx.task
    return ToolOut({
        "slope_deg": task.slope_deg,
        "friction": task.friction,
        "target_m": task.target_m,
        "gait_schema": Gait.model_json_schema(),
        "neutral_offsets": neutral_offsets(),
    })


def _submit_gait(ctx: ToolCtx, args: dict[str, Any]) -> ToolOut:
    try:
        gait = Gait.model_validate(args)
    except ValidationError as e:
        return ToolOut({"ok": False, "error": str(e)})
    return ToolOut({"ok": True}, gait=gait)


def _preview_run(ctx: ToolCtx, args: dict[str, Any]) -> ToolOut:
    try:
        gait = Gait.model_validate(args)
    except ValidationError as e:
        return ToolOut({"ok": False, "error": str(e)})
    result, log = preview.run_preview(gait, ctx.task)
    record = {"gait": gait.model_dump(), "result": result, "contact_log": log}
    return ToolOut({"ok": True, **result}, preview=record)


def _get_contact_log(ctx: ToolCtx, args: dict[str, Any]) -> ToolOut:
    if not ctx.previews:
        return ToolOut({"ok": False, "error": "no preview in this episode yet"})
    return ToolOut({"ok": True, **ctx.previews[-1]["contact_log"]})


def _list_my_attempts(ctx: ToolCtx, args: dict[str, Any]) -> ToolOut:
    return ToolOut({"ok": True, "attempts": [
        {"n": i, "gait": p["gait"], "preview": p["result"]} for i, p in enumerate(ctx.previews, 1)
    ]})


def _spec(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}


TOOLS: dict[str, ToolDef] = {
    "read_task": ToolDef(
        spec=_spec(
            "read_task",
            "Read this task: slope, friction, target distance, the gait JSON schema and neutral joint offsets.",
            NO_ARGS,
        ),
        run=_read_task,
        is_attempt=False,
    ),
    "submit_gait": ToolDef(
        spec=_spec(
            "submit_gait",
            "Submit a gait for this task. The arguments are the gait JSON, validated against the schema.",
            Gait.model_json_schema(),
        ),
        run=_submit_gait,
        is_attempt=True,
    ),
    "preview_run": ToolDef(
        spec=_spec(
            "preview_run",
            f"Try a gait for {preview.PREVIEW_S:.0f} s on a practice run of this task without submitting it. "
            "Returns distance, whether Ada fell, max torso tilt and each leg's contact rhythm.",
            Gait.model_json_schema(),
        ),
        run=_preview_run,
        is_attempt=False,
    ),
    "get_contact_log": ToolDef(
        spec=_spec(
            "get_contact_log",
            f"Each leg's ground contact during the last preview: share of time in contact "
            f"in each of {preview.LOG_BINS} equal time bins.",
            NO_ARGS,
        ),
        run=_get_contact_log,
        is_attempt=False,
    ),
    "list_my_attempts": ToolDef(
        spec=_spec(
            "list_my_attempts",
            "The gaits you previewed earlier in this episode, with their preview results.",
            NO_ARGS,
        ),
        run=_list_my_attempts,
        is_attempt=False,
    ),
}
assert set(TOOLS) <= set(TOOL_WHITELIST)


def bound_tools(harness: Harness) -> dict[str, ToolDef]:
    """The tools this harness binds, in harness order; names off the whitelist or unbuilt are an error."""
    outside = [n for n in harness.tools if n not in TOOL_WHITELIST]
    if outside:
        raise ValueError(f"harness lists tools outside the whitelist: {outside}")
    missing = [n for n in harness.tools if n not in TOOLS]
    if missing:
        raise NotImplementedError(f"harness lists tools with no implementation: {missing}")
    return {n: TOOLS[n] for n in harness.tools}


# --- graph ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    steps: Annotated[list[dict[str, Any]], operator.add]
    attempts: int
    gait: dict[str, Any] | None
    previews: Annotated[list[dict[str, Any]], operator.add]
    preview_calls: int
    cost_usd: float
    tokens: int
    model_id: str | None


@dataclass
class EpisodeResult:
    gait: Gait | None
    attempts: int
    cost_usd: float
    tokens: int
    model_id: str
    trace_id: str
    end_reason: str | None = None  # submitted|attempts_exhausted|previews_exhausted|step_limit|no_tool_call


def max_attempts(harness: Harness) -> int:
    return int(harness.engine.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)


def recursion_limit(harness: Harness) -> int:
    """Supersteps for one turn (model + tools) per preview, per attempt and EXTRA_TURNS, plus a final model step."""
    return 2 * (MAX_PREVIEWS + max_attempts(harness) + EXTRA_TURNS) + 1


def end_reason(state: AgentState, harness: Harness, hit_step_limit: bool) -> str:
    if state["gait"] is not None:
        return "submitted"
    if state["attempts"] >= max_attempts(harness):
        return "attempts_exhausted"
    if hit_step_limit:
        return "previews_exhausted" if state["preview_calls"] >= MAX_PREVIEWS else "step_limit"
    return "no_tool_call"


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def _tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if total is None:
        total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    return int(total)


def build_graph(task: Task, harness: Harness, checkpointer: MongoDBSaver | None = None):
    tools = bound_tools(harness)
    specs = [t.spec for t in tools.values()] or None
    role = harness.model_per_step[AGENT_STEP]
    limit = max_attempts(harness)
    params = {k: harness.engine[k] for k in ("temperature",) if harness.engine.get(k) is not None}

    def model_node(state: AgentState) -> dict[str, Any]:
        res = llm.cached_chat(role, state["messages"], tools=specs, **params)
        msg = _assistant_message(res.message)
        step = {
            "node": "model", "role": role, "model_id": res.model_id, "prompt_hash": res.prompt_hash,
            "cached": res.cached, "cost_usd": res.cost_usd, "usage": res.usage, "message": msg,
        }
        return {
            "messages": [msg], "steps": [step], "model_id": res.model_id,
            "cost_usd": round(state["cost_usd"] + res.cost_usd, 6),
            "tokens": state["tokens"] + _tokens(res.usage),
        }

    def tool_node(state: AgentState) -> dict[str, Any]:
        attempts, gait, preview_calls = state["attempts"], state["gait"], state["preview_calls"]
        messages, steps, previews = [], [], []
        for call in state["messages"][-1].get("tool_calls") or []:
            name = call.get("function", {}).get("name", "")
            raw_args = call.get("function", {}).get("arguments") or "{}"
            tool = tools.get(name)
            valid = None
            is_preview = name == "preview_run" and tool is not None
            if tool is None:
                result = {"ok": False, "error": f"unknown tool {name!r}; available: {list(tools)}"}
            elif tool.is_attempt and (gait is not None or attempts >= limit):
                result = {"ok": False, "error": "no attempts left"}
            elif is_preview and preview_calls >= MAX_PREVIEWS:
                result = {"ok": False, "error": PREVIEW_LIMIT_MSG}
            else:
                if is_preview:
                    preview_calls += 1
                if tool.is_attempt:
                    attempts += 1
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as e:
                    out = ToolOut({"ok": False, "error": f"invalid arguments: {e}"})
                else:
                    out = tool.run(ToolCtx(task, state["previews"] + previews), args)
                result, submitted = out.result, out.gait
                if out.preview is not None:
                    previews.append(out.preview)
                if tool.is_attempt:
                    valid = submitted is not None
                    if submitted is not None:
                        gait = submitted.model_dump()
            content = json.dumps(result, separators=(",", ":"))
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
            steps.append({
                "node": "tools", "name": name, "arguments": raw_args if isinstance(raw_args, str)
                else json.dumps(raw_args), "result": content, "attempt": attempts, "valid": valid,
                "preview": preview_calls,
            })
        return {"messages": messages, "steps": steps, "attempts": attempts, "gait": gait, "previews": previews,
                "preview_calls": preview_calls}

    def after_model(state: AgentState) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") else END

    def after_tools(state: AgentState) -> str:
        return END if state["gait"] is not None or state["attempts"] >= limit else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    return graph.compile(checkpointer=checkpointer)


def thread_id(version_id: str, task: Task) -> str:
    """{version}-{task}-{seed}; one episode per task, keyed by its first practice seed."""
    return f"{version_id}-{task.id}-{task.practice_seeds[0]}"


def run_episode(
    task: Task, version_id: str, harness: Harness, *, db_name: str = ADA, ckpt_db: str = ADA_CKPT,
) -> EpisodeResult:
    tid = thread_id(version_id, task)
    saver = MongoDBSaver(client(), db_name=ckpt_db)
    saver.delete_thread(tid)  # a rerun starts fresh instead of resuming
    graph = build_graph(task, harness, saver)
    system = "\n".join([SYSTEM_PROMPT, *harness.rules])
    initial: AgentState = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"Task {task.id}."}],
        "steps": [], "attempts": 0, "gait": None, "previews": [], "preview_calls": 0, "cost_usd": 0.0, "tokens": 0, "model_id": None,
    }
    config = {"configurable": {"thread_id": tid}, "recursion_limit": recursion_limit(harness)}
    hit_step_limit = False
    try:
        state = graph.invoke(initial, config, durability="sync")
    except GraphRecursionError:
        state = graph.get_state(config).values  # model kept calling tools without finishing
        hit_step_limit = True

    reason = end_reason(state, harness, hit_step_limit)
    trace = Trace(trace_id=tid, raw_steps=state["steps"], end_reason=reason, previews=state["preview_calls"])
    db(db_name).traces.replace_one(
        {"trace_id": tid}, trace.model_dump(by_alias=True, exclude={"id"}), upsert=True,
    )
    role = harness.model_per_step[AGENT_STEP]
    return EpisodeResult(
        gait=Gait.model_validate(state["gait"]) if state["gait"] is not None else None,
        attempts=state["attempts"],
        cost_usd=round(state["cost_usd"], 6),
        tokens=state["tokens"],
        model_id=state["model_id"] or llm.models()[role]["id"],
        trace_id=tid,
        end_reason=reason,
    )
