"""Harness agent (SPEC §3): a LangGraph StateGraph checkpointed by MongoDBSaver.

    run_episode(task, version_id, harness) -> EpisodeResult

The model node calls core.llm.cached_chat with the role from harness.model_per_step;
the tool node runs only tools listed in harness.tools. Every submit_gait call counts
as an attempt; the episode ends on the first valid gait or at engine.max_attempts.
Each step is stored in `traces.raw_steps` (trace_id = thread id).
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
from sim.gait import Gait, neutral_offsets

SYSTEM_PROMPT = "Write a gait for Ada for this task."
AGENT_STEP = "agent"  # key of harness.model_per_step used by the model node
DEFAULT_MAX_ATTEMPTS = 3
STEPS_PER_ATTEMPT = 6  # recursion budget per attempt (model + tools nodes, with read_task turns)

V0_HARNESS = Harness(
    rules=[],
    context_policy={},
    tools=["read_task", "submit_gait"],
    model_per_step={AGENT_STEP: "agent_v0"},
    engine={"temperature": 0, "max_attempts": DEFAULT_MAX_ATTEMPTS},
)


# --- tools ------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDef:
    spec: dict[str, Any]
    run: Callable[[Task, dict[str, Any]], tuple[dict[str, Any], Gait | None]]
    is_attempt: bool


def _read_task(task: Task, args: dict[str, Any]) -> tuple[dict[str, Any], Gait | None]:
    return {
        "slope_deg": task.slope_deg,
        "friction": task.friction,
        "target_m": task.target_m,
        "gait_schema": Gait.model_json_schema(),
        "neutral_offsets": neutral_offsets(),
    }, None


def _submit_gait(task: Task, args: dict[str, Any]) -> tuple[dict[str, Any], Gait | None]:
    try:
        gait = Gait.model_validate(args)
    except ValidationError as e:
        return {"ok": False, "error": str(e)}, None
    return {"ok": True}, gait


def _spec(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS: dict[str, ToolDef] = {
    "read_task": ToolDef(
        spec=_spec(
            "read_task",
            "Read this task: slope, friction, target distance, the gait JSON schema and neutral joint offsets.",
            {"type": "object", "properties": {}, "additionalProperties": False},
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
}


def bound_tools(harness: Harness) -> dict[str, ToolDef]:
    """The tools this harness binds, in harness order; unknown names are an error."""
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


def max_attempts(harness: Harness) -> int:
    return int(harness.engine.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)


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
        attempts, gait = state["attempts"], state["gait"]
        messages, steps = [], []
        for call in state["messages"][-1].get("tool_calls") or []:
            name = call.get("function", {}).get("name", "")
            raw_args = call.get("function", {}).get("arguments") or "{}"
            tool = tools.get(name)
            valid = None
            if tool is None:
                result = {"ok": False, "error": f"unknown tool {name!r}; available: {list(tools)}"}
            elif tool.is_attempt and (gait is not None or attempts >= limit):
                result = {"ok": False, "error": "no attempts left"}
            else:
                if tool.is_attempt:
                    attempts += 1
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as e:
                    result, submitted = {"ok": False, "error": f"invalid arguments: {e}"}, None
                else:
                    result, submitted = tool.run(task, args)
                if tool.is_attempt:
                    valid = submitted is not None
                    if submitted is not None:
                        gait = submitted.model_dump()
            content = json.dumps(result, separators=(",", ":"))
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
            steps.append({
                "node": "tools", "name": name, "arguments": raw_args if isinstance(raw_args, str)
                else json.dumps(raw_args), "result": content, "attempt": attempts, "valid": valid,
            })
        return {"messages": messages, "steps": steps, "attempts": attempts, "gait": gait}

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
        "steps": [], "attempts": 0, "gait": None, "cost_usd": 0.0, "tokens": 0, "model_id": None,
    }
    config = {"configurable": {"thread_id": tid}, "recursion_limit": STEPS_PER_ATTEMPT * max_attempts(harness)}
    try:
        state = graph.invoke(initial, config, durability="sync")
    except GraphRecursionError:
        state = graph.get_state(config).values  # model kept calling tools without finishing

    trace = Trace(trace_id=tid, raw_steps=state["steps"])
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
    )
