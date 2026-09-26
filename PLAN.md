# PLAN.md — Ada, 3D walker

Times assume the pivot starts at 11:00. Shift everything by your actual start. Every task: fresh session, Plan mode, approve, switch to Auto, `./check` green, human commits with `./c`.
Effort: A and B on Opus 5.5 high, C on medium, any retry on xhigh.

## P0 Pivot (human, 10 min)
First stop every running Claude Code session (let a session finish its current edit, then close it). Download the four files from this chat into `~/hack-prep/walk/`. Then:
```
cd ~/taxloop
./c TAX "tax wip before pivot" env tests data 2>/dev/null || true
git branch -f tax-archive
git rm -r -q --ignore-unmatch env tests/env tests/test_known_answers.py data/irs && rm -rf env tests/env data/irs
cp ~/hack-prep/walk/{SPEC,PLAN,CLAUDE,CONTRACTS}.md .
uv add mujoco gymnasium voyageai
uv run python - <<'EOF2'
import gymnasium, os, shutil
src=os.path.join(os.path.dirname(gymnasium.__file__),"envs/mujoco/assets/ant.xml")
os.makedirs("sim/assets",exist_ok=True); shutil.copy(src,"sim/assets/ant.xml"); print("copied",src)
EOF2
echo 'LANGSMITH_TRACING=true' >> .env
./c P0 "pivot: 3D walker docs, mujoco, ant asset" SPEC.md PLAN.md CLAUDE.md CONTRACTS.md sim pyproject.toml uv.lock
```
If `ui/` already exists from the tax C0 task, keep it; the next C session changes the database name to `ada`. If `core/` exists from B0, the next B session regenerates `core/contracts.py` from the new CONTRACTS.md.
Add `LANGSMITH_API_KEY` and `VOYAGE_API_KEY` to `.env` when you have them. After C0: generate three ElevenLabs sound effects (soft metallic robot footstep; heavy thud; rising electronic glitch whine) and save as `ui/public/sfx/step.mp3`, `fall.mp3`, `glitch.mp3`.

## Lane A — sim, compressor, loop
- **A1 11:05 — engine and determinism.** `sim/model.py` loads ant.xml with `mujoco`, sets gear = 150 × power, gravity tilt for slope, floor friction; `sim/assets/manifest.json` export (non-floor geoms: name, type, size, body). Tests: determinism sha (SPEC §2.6 test 1), zero gait (test 2). `./check tests/sim/test_engine.py`
- **A2 11:35 — gait, controller, tasks.** `sim/gait.py` (pydantic schema with ranges per SPEC §2.2), `sim/controller.py` (PD + sine + tilt correction), `sim/tasks.py` (12 train, 6 holdout, 1 showcase; disjoint practice/eval seeds) writes `tasks`. Test: seeds disjoint; schema rejects out-of-range. `./check tests/sim/test_gait_tasks.py`
- **A3 12:00 — verifier and recorder.** `sim/verifier.py` (distance, fell, sanity with violation_frame, success) and `sim/record.py` (frames per SPEC §2.5). `scripts/export_fixtures.py` writes three fixture frames docs (zero, a hand-written diagonal trot, the power-5 flail) into `ada` and prints their ids for Lane C. Tests: SPEC §2.6 test 3; frames sha stable. `./check tests/sim/test_verifier.py`
- **A4 12:30 — compressor.** `compress/provider.py`: deterministic, ordered steps (reasoning summary, tool, args, result summary, decision), ≤ 6k tokens, never tail-trimmed; hash exported. `./check tests/compress`
- **A5 13:00 — sensor.** `loop/sensor.py`: aggregate verifier results on the train subset, `$group` by failure code. `./check tests/loop/test_sensor.py`
- **A6 13:25 — controller.** `loop/controller.py`: ≤ 6 edits, smallest first, primitive/old/new/predicted_delta/rationale; objective `0.7 × train reliability + 0.3 × normalized distance`; cost estimate before the call, abort above $3. `./check tests/loop/test_controller.py`
- **A7 13:50 — actuator.** `loop/actuator.py`: child version → gate → write versions/edits/events; accepted ⇒ holdout k=3 + showcase frames; rejected with sanity violation ⇒ record frames; one open round via unique partial index. `./check tests/loop/test_actuator.py`
- **A8 14:15 — round end to end.** `loop/run.py --rounds N`. Test: one round ≤ 6 edits, ≤ $3, events in Atlas, one open round. `./check tests/loop/test_round_e2e.py`. Then start the loop running and leave it.

## Lane B — core, harness, gate
- **B0 11:05 — contracts and db.** `core/contracts.py` from CONTRACTS.md §3–4 exactly; `core/db.py` one cached MongoClient; `tests/core/test_contracts.py` round-trip. `./check tests/core`
- **B1 11:30 — LLM cache.** `core/llm.py`: OpenRouter client, cache keyed on sha256(canonical JSON of model_id, messages, tools, params), store in `ada_cache.llm_cache`, modes record/replay_only, cost from models.json prices, semaphore 4, backoff honoring Retry-After. `./check tests/core/test_llm_cache.py` (second identical call = 0 HTTP).
- **B2 11:55 — events.** `core/events.py` emit(); Python change-stream test on `ada_test`. `./check tests/core/test_events.py`
- **B3 12:10 — harness v0.** `harness/agent.py`: LangGraph + MongoDBSaver(db `ada_ckpt`), tools `read_task`, `submit_gait`; `harness/run.py --version v0 --split train --k 2` runs the agent per task, evaluates each gait with `sim.verifier`, writes `runs`, `versions` v0, showcase frames. **12:30 kill check: v0 scored and recorded.** `./check tests/harness/test_v0.py`
- **B4 12:40 — whitelist and guardrails.** Tools `preview_run`, `get_contact_log`, `list_my_attempts` (practice seeds only); `harness_guardrails` seeded with content-addressed `_id`; load-time verification; tamper test refuses start. `./check tests/harness/test_tools_guardrails.py`
- **B5 13:10 — baselines.** `scripts/baseline.py`: v0 and frontier on holdout k=3, showcase frames for both, frozen with sha. Start the frontier run and let it finish. `./check tests/test_baseline_frozen.py`
- **B6 13:30 — gate stage 1 and 4.** `gate/verifier_stage.py` (train subset, sanity reject with reason + violation_frame), `gate/constraints.py` (whitelist, guardrail hashes, verifier/compressor hashes). Test: power-5 flail gait ⇒ rejected at gate.verifier; non-whitelisted tool ⇒ rejected at gate.constraints with 0 LLM calls. `./check tests/gate/test_rejects.py`
- **B7 14:00 — gate stages 2 and 3.** `gate/judge.py` Agent GPA on the compressed trace via core/llm (judge family ≠ agent family); `gate/meta.py` veto. `./check tests/gate/test_judge.py`
- **B8 14:30 — verify CLI.** `./verify --edit {id} --clip-power 1.0` re-runs a rejected gait at stock power and prints sanity + distance; writes an `edits` doc with origin cli. `./check tests/gate/test_verify_cli.py`
- **B9 15:00 — skill memory (only if A8 is green).** Voyage embedding on accepted versions into `skills`; Atlas vector index; `lookup_skill` tool. `./check tests/harness/test_skills.py`
- **B10 15:30 — demo check.** `./demo_check 10`: replay_only, 0 cache misses, 10/10.

## Lane C — the screen (Opus 5.5 medium)
- **C0 11:05 — scaffold.** `npx create-next-app@latest ui --yes --disable-git`; add `three @react-three/fiber @react-three/drei @react-three/postprocessing mongodb qrcode.react`; IBM Plex via next/font; `api/stream`, `api/frames/[id]` per CONTRACTS §5; page shows raw events. Deploy to Vercel; set MONGODB_URI in Vercel env. Check: an inserted event appears on the prod URL.
- **C1 11:40 — the stage.** Scene per SPEC §6: background, fog, ACES tone mapping, key + rim light, contact shadows, floor with meter lines and numbers, Ada built from manifest in the physical material, replaying one fixture frames doc at 33.3 fps. Check: fixture trot plays; it already looks cinematic.
- **C2 12:30 — ghosts and camera.** Load showcase frames for all versions; ghosts per SPEC §6; follow cam with lag on the leader. Check: 3 fixture runs play together as ghosts.
- **C3 13:10 — card and spine overlays.** Glass card with four stage cells filling per event, reason; top-right frontier and current rows with n; claim line rule; top-left "Ada · v{N}" and live distance at 72 pt; QR bottom-right. Check: a fake edit's four events fill the cells in order.
- **C4 13:50 — footprints and joint glow.** Contact rising edges ⇒ fading emissive discs; leg emissive from recorded forces; bloom on these only.
- **C5 14:30 — bullet time.** On a frames doc with `violation_frame`: 0.1× from 0.5 s before, 4 s orbit, red wireframe, verdict text. Check on the flail fixture.
- **C6 15:00 — race and sound.** Frontier vs best in two lanes with cost under each; step/fall/glitch sounds per SPEC §6.
- **C7 15:30 — booth mode and phone.** `?mode=booth` 60 s loop with the "Replay of runs recorded today" label; phone layout with version slider. Projector check from 3 m.

## Milestones
- 12:30 kill check: v0 scored and recorded (else back to tax-archive).
- 13:30 3D replay check (else 2D side view from the same frames).
- 14:30 loop running continuously; every model output cached.
- 15:15 count rejected edits with a sanity violation. Zero ⇒ let it run; never write one by hand. At 15:45 with still zero: the demo's cheat beat uses `./verify` on the flail probe, labeled "probe" on the card and aloud.
- 16:00 UI freeze. Best version on holdout k=3 final.
- T−90 feature freeze. T−80 record functionality (two clips, 20 s each: opening ghosts, the cheat). T−75 record code (loop, gate, recorder, 15 s). T−70 Trupeer + ElevenLabs voice, 60 s export. T−60 submit: public repo, video, description. Check the submission deadline now and write it here: ______.
- T−45 rehearse SPEC §8 twice on the prod URL.

## Cut order
SPEC §10.
