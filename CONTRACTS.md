# CONTRACTS.md (frozen at commit; changes only via a NOTES.md line applied by the human)

## 1. Interface
Python types only in `core/contracts.py` (pydantic v2), generated from this file. The UI reads JSON only.

## 2. Databases (one cluster, MONGODB_URI)
| db | contents |
|---|---|
| ada | app data below; the only db the UI watches |
| ada_ckpt | LangGraph MongoDBSaver(client, db_name="ada_ckpt") |
| ada_cache | llm_cache {model_id, prompt_hash, request, response, usage, ts}, unique (model_id, prompt_hash) |
| ada_test | tests only; each test drops only collections it created |

## 3. Collections in `ada`
| collection | writer | fields |
|---|---|---|
| tasks | sim/tasks.py | _id, split train/holdout/showcase, slope_deg, friction, target_m, eval_seeds[], practice_seeds[] |
| events | core/events.emit() | ts, round_id, version_id, edit_id, stage, status, payload (<= 2 KB) |
| rounds | loop/actuator | _id, status open/closed, opened_at, closed_at, budget_usd, budget_s, open_lock |
| versions | loop/actuator | _id v0,v1,..., parent, status baseline/frontier/accepted/rejected, harness {rules[], context_policy{}, tools[], model_per_step{}, engine{}}, metrics {train_reliability, holdout_reliability_80, mean_distance_m, cost_per_run_usd, n}, showcase_frames_id, created_at |
| edits | loop/controller, loop/actuator, ./verify | _id, round_id, from_version, to_version, origin model/probe/cli, primitive, old, new, rationale, predicted_delta, actual_delta, verdict accepted/rejected, reason, violation_frame, frames_id, created_at |
| runs | harness + sim | task_id, seed, split train/holdout/showcase/swap (swap: written only by scripts/capture_swap.py, never updates versions metrics), version_id, model_id, gait{}, distance_m, fell, sanity{pass, violation, violation_frame}, success, cost_usd, tokens, trace_id |
| traces | harness, compress | trace_id, raw_steps[], compressed_steps[], schema "gait-v1" |
| frames | sim/record.py | _id, run_id, version_id, kind showcase/frontier/rejected, manifest_version, fps, frames[{geoms[[x,y,z,qw,qx,qy,qz]...], contacts[4], forces[8], torso[3]}], sha256 |
| skills | cut; not built (§7 skill memory was cut; no code writes or reads it) | _id, version_id, task_id, summary, gait{}, embedding[] (Voyage), created_at |
| harness_guardrails | seeded once | _id = sha256(content), content {tool_whitelist, sanity_bounds, rated_power, verifier_sha, compressor_sha}; verified at load |
| baselines | scripts/baseline.py | _id = version_id, version_id, split, k, metrics, sha256, created_at |
| swaps | scripts/capture_swap.py | _id, captured_at, task_id, k, left_version, right_version, left_holdout, right_holdout, left_frames_id, right_frames_id, model_id, right_model_id, verifier_sha, mujoco_version, manifest_version, harness_diff[], left_mean_distance_m, right_mean_distance_m, left_cost_per_run_usd, right_cost_per_run_usd, n, fresh_calls, model_calls |
| scoreboard | scripts/scoreboard.py | _id, created_at, best_version, rewrite_cost_usd, cost_per_gait{v0, frontier, best}, overrated_attempts, proposals_total, physics_rejected, physics_rejected_judge_passed |
| snapshots | scripts/pin_snapshot.py | _id, pinned_at, best_version, v0, frontier, attempt_edit_ids[], cheat_edit_id, swaps_id, scoreboard_id, ghosts[{version_id, frames_id}] |

## 4. Event schema (the UI contract)
{"ts": ISO8601, "round_id": str|null, "version_id": str|null, "edit_id": str|null,
 "stage": "harness|baseline|sensor|controller|gate.verifier|gate.gpa|gate.meta|gate.constraints|actuator",
 "status": "start|pass|fail|info", "payload": {...}}
Card = one edit_id; four cells in order gate.verifier, gate.gpa, gate.meta, gate.constraints.

## 5. API routes (ui/app/api/)
- `stream/route.ts`: runtime "nodejs", dynamic "force-dynamic", maxDuration 300. On connect send a snapshot {frontier version, latest 30 versions, latest edit, showcase frames ids}. Then `db("ada").watch([{ $match: { operationType: {$in:["insert","update"]}, "ns.coll": {$in:["events","versions","edits"]} } }], {fullDocument:"updateLookup"})`. `: ping` every 15 s. On reconnect resend the snapshot. Client dedupes by _id.
- `frames/[id]/route.ts`: returns one frames document.
- `manifest/route.ts`: returns sim/assets/manifest.json (copied into ui/public/ at build is fine).

## 6. Tool whitelist (exactly 5; v0 binds the first 2)
1. read_task  2. submit_gait  3. preview_run  4. get_contact_log  5. list_my_attempts

## 7. models.json (human-owned)
Keys agent_v0, frontier, controller, judge; each {id, family, usd_per_mtok_in, usd_per_mtok_out, temperature, alt{...}}. judge.family ≠ agent_v0.family and ≠ controller.family.

## 8. Ownership
Session A: sim/ tests/sim/ compress/ loop/ tests/loop/
Session B: core/ harness/ gate/ scripts/ verify tests/core/ tests/harness/ tests/gate/
Session C: ui/
Human only: SPEC.md PLAN.md CLAUDE.md CONTRACTS.md NOTES.md c check pyproject.toml uv.lock .env
