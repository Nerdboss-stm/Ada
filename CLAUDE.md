# CLAUDE.md — rules for every session

Read CONTRACTS.md first, then only the PLAN.md task named in your opening message, then the SPEC.md section it cites.

## Hard rules
- One task per session. Plan first (files you will touch, the exact ./check command, ≤ 30 min). Wait for approval.
- Never run `git commit`, `git add`, `git push`. No `Co-Authored-By` or "Generated with" text anywhere. The human commits.
- Stay inside your session's directories (CONTRACTS.md §8). Need something outside them: append one line to NOTES.md (`- [TASK] BLOCKED/IDEA/CONTRACT-CHANGE: ...`) and stop.
- No new dependencies. Need one: append `- [TASK] DEP: <pkg> <why>` to NOTES.md and stop.
- Never edit `sim/verifier.py` sanity bounds, `harness_guardrails`, or `compress/` after they are committed.
- Never script, force, or hint the reward hack. The `power` range and the distance term stay as written in SPEC.md. No prompt under `harness/` or `loop/` may mention glitches, exploits, or power limits.
- The agent's tools use practice seeds only. Evaluation seeds are read only by `sim/verifier.py` and the gate.
- The UI draws only recorded frames and stored documents. Replay speed may change; frames may not be invented, interpolated beyond linear between recorded frames, or smoothed into values that never existed.
- Only `core/llm.py` calls OpenRouter, always through the (model_id, prompt_hash) cache in `ada_cache`.
- Tests use database `ada_test` and drop only collections they created.
- Unsure of a library API (mujoco, langgraph, trulens, drei): a subagent reads the installed source first.
- Kill any run that goes 15 minutes without a green ./check or a visible checkpoint; write 3 lines to NOTES.md (tried / failing / next) and stop.

## Stack
Python 3.11, uv, pydantic v2, pytest via `./check <path>`. mujoco. LangGraph + MongoDBSaver. OpenRouter. TruLens GPA scorers (or hand-written GPA prompts if setup exceeds 20 minutes). Next.js App Router in `ui/`, React Three Fiber, @react-three/drei, @react-three/postprocessing, IBM Plex via next/font, qrcode.react. Vercel.

## Conventions
- Names only from CONTRACTS.md. Pydantic models in `core/contracts.py` are the only cross-directory interface.
- Money in USD floats rounded to 6 places in documents; displayed to 4.
- Distances in meters, 2 decimals on screen.
- Cache every model output; `LLM_CACHE_MODE=record|replay_only`.

## End every task with 5 lines
files changed / command / CHECKPOINT line / known gaps / next task.
