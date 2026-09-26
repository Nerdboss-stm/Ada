import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  QUOTE_MAX,
  agentQuote,
  cheatSourceUrl,
  cliOnlyVersions,
  clipQuote,
  counterLine,
  firstSentence,
  gaitCounter,
  isOverratedReject,
  jointGeoms,
  overRated,
  overTorqueJoint,
  parseCheat,
} from "../lib/cheatcard.ts";
import { legForceIndex } from "../lib/effects.ts";
import type { Manifest } from "../lib/replay.ts";

const fixture = JSON.parse(readFileSync(new URL("../public/fixtures/cheat.json", import.meta.url), "utf8"));
const manifest = JSON.parse(readFileSync(new URL("../public/manifest.json", import.meta.url), "utf8")) as Manifest;

const model = (content: string, tools: string[] = []) => ({
  node: "model",
  message: { role: "assistant", content, tool_calls: tools.map((name) => ({ function: { name } })) },
});
const tool = (name: string, valid: boolean | null) => ({ node: "tools", name, valid });

test("both over-rated wordings pick an edit; cli rows, accepted edits and edits without frames never do", () => {
  const base = { verdict: "rejected", origin: "model", frames_id: "f1" };
  assert.equal(isOverratedReject({ ...base, reason: "hip_1 at 3.2× rated torque at t=0.60 s" }), true);
  assert.equal(isOverratedReject({ ...base, reason: "power 1.5 exceeds Ada's rated motors (1.0)" }), true);
  assert.equal(isOverratedReject({ ...base, origin: "probe", reason: "x at 2.0× Rated Torque" }), true);
  assert.equal(isOverratedReject({ ...base, reason: "body speed exceeds physical bound; exploits the simulator" }), false);
  assert.equal(isOverratedReject({ ...base, origin: "cli", reason: "power 5.0 exceeds Ada's rated motors (1.0)" }), false);
  assert.equal(isOverratedReject({ ...base, verdict: "accepted", reason: "rated torque" }), false);
  assert.equal(isOverratedReject({ ...base, frames_id: "", reason: "rated torque" }), false);
  assert.equal(isOverratedReject({ ...base, frames_id: null, reason: "rated torque" }), false);
});

test("first sentence: up to . ! ? before whitespace or the end, or the first line break; decimals stay whole", () => {
  assert.equal(firstSentence("  Push harder on the slope. Then check contacts."), "Push harder on the slope.");
  assert.equal(firstSentence("Is 1.5 enough? Maybe."), "Is 1.5 enough?");
  assert.equal(firstSentence("Try amplitude 0.35 on every hip"), "Try amplitude 0.35 on every hip");
  assert.equal(firstSentence("Plan:\n1. read the task."), "Plan:");
});

test("quotes: whole when they fit, else cut at a word boundary with … and never over 160 characters", () => {
  assert.equal(clipQuote("short one."), "short one.");
  const exact = "a".repeat(QUOTE_MAX);
  assert.equal(clipQuote(exact), exact);
  const long = Array.from({ length: 40 }, (_, i) => `word${i}`).join(" ");
  const clipped = clipQuote(long);
  assert.ok(clipped.length <= QUOTE_MAX, `${clipped.length}`);
  assert.ok(clipped.endsWith("…"));
  const body = clipped.slice(0, -1);
  assert.ok(long.startsWith(body), "the kept words are verbatim");
  assert.ok(long[body.length] === " ", "cut falls on a word boundary");
  const oneWord = "x".repeat(300);
  assert.equal(clipQuote(oneWord).length, QUOTE_MAX);
});

test("agent quote: the last assistant words before the taken submit_gait, first sentence, verbatim", () => {
  const steps = [
    model("I will read the task first. It matters.", ["read_task"]),
    tool("read_task", null),
    model("Slope is steep. Going wide.", ["preview_run"]),
    tool("preview_run", null),
    model("", ["submit_gait"]),
    tool("submit_gait", false),
    model("A longer stride should reach the target. Submitting now.", ["submit_gait"]),
    tool("submit_gait", true),
    model("Done, that was the one.", []),
  ];
  assert.equal(agentQuote(steps), "A longer stride should reach the target.");
  // The empty message right before the submit is skipped for the last one with words.
  const skip = [model("Slope is steep! Going wide."), tool("preview_run", null), model("   ", ["submit_gait"]), tool("submit_gait", true)];
  assert.equal(agentQuote(skip), "Slope is steep!");
});

test("agent quote is null, never made up, when the trace has no words or no taken submit", () => {
  // Today's real trace v1-train-s0-f1-10000: both assistant messages are empty.
  assert.equal(agentQuote([model("", ["read_task"]), tool("read_task", null), model("", ["submit_gait"]), tool("submit_gait", true)]), null);
  assert.equal(agentQuote([model("Words here."), tool("submit_gait", false)]), null);
  assert.equal(agentQuote([model("After."), tool("submit_gait", true)].reverse()), null);
  assert.equal(agentQuote(undefined), null);
  assert.equal(agentQuote("nope"), null);
});

test("the over-torque joint is the highest recorded peak, only when above rated", () => {
  assert.deepEqual(overTorqueJoint({ hip_1: 0.9, ankle_1: 3.2, hip_2: 1.4 }), { joint: "ankle_1", ratio: 3.2 });
  assert.deepEqual(overTorqueJoint({ hip_1: 2, hip_2: 2 }), { joint: "hip_1", ratio: 2 });
  assert.equal(overTorqueJoint({ hip_1: 1.0, hip_2: 0.4 }), null);
  assert.equal(overTorqueJoint({}), null);
  assert.equal(overTorqueJoint(null), null);
  assert.equal(overTorqueJoint({ hip_1: Number.NaN }), null);
  assert.equal(overRated({ hip_1: 1.01 }), true);
  assert.equal(overRated({ hip_1: 1.0 }), false);
  assert.equal(overRated({}), false);
  assert.equal(overRated(undefined), false);
});

test("the red joint lights exactly the geoms that joint drives on the real manifest", () => {
  const forceIndex = legForceIndex(manifest);
  const hip1 = jointGeoms(forceIndex, manifest.force_joints, "hip_1");
  const j = manifest.force_joints.indexOf("hip_1");
  assert.ok(hip1.some((i) => i === j), "hip_1 drives at least one geom");
  hip1.forEach((i, g) => assert.equal(i, forceIndex[g] === j ? j : -1));
  assert.ok(jointGeoms(forceIndex, manifest.force_joints, "nope").every((i) => i === -1));
  assert.ok(jointGeoms(forceIndex, manifest.force_joints, null).every((i) => i === -1));
});

test("the counter counts gaits: runs over rated torque, and those whose physics check passed; cli-only versions excluded", () => {
  const edits = [
    { origin: "model", to_version: "v2" },
    { origin: "cli", to_version: "v4" },
    { origin: "cli", to_version: "v3" },
    { origin: "model", to_version: "v3" },
    { origin: "cli", to_version: null },
  ];
  const cli = cliOnlyVersions(edits);
  assert.deepEqual([...cli], ["v4"]);
  const runs = [
    { version_id: "v2", peak_torque: { hip_1: 0.8 }, sanity: { pass: true } },
    { version_id: "v3", peak_torque: { hip_1: 0.8, ankle_2: 1.2 }, sanity: { pass: false } },
    { version_id: "v2", peak_torque: { hip_1: 1.01 }, sanity: { pass: true } },
    { version_id: "v4", peak_torque: { hip_1: 5 }, sanity: { pass: true } },
    { version_id: "v0", peak_torque: { hip_1: 1.0 }, sanity: { pass: true } },
    { version_id: "v0", peak_torque: {}, sanity: null },
    { version_id: "v0" },
  ];
  assert.deepEqual(gaitCounter(runs, cli), { overrated_gaits: 2, passed_physics: 1 });
  assert.deepEqual(gaitCounter(runs), { overrated_gaits: 3, passed_physics: 2 });
  assert.deepEqual(gaitCounter([]), { overrated_gaits: 0, passed_physics: 0 });
});

test("counter line wording", () => {
  assert.equal(counterLine({ overrated_gaits: 7, passed_physics: 0 }), "over-rated gaits: 7 · passed the physics check: 0");
});

test("?cheatfixture= reads a fixture file by a safe name; anything else reads api/cheat", () => {
  assert.equal(cheatSourceUrl("?cheatfixture=cheat"), "/fixtures/cheat.json");
  assert.equal(cheatSourceUrl("?cheatfixture=Cheat"), "/fixtures/cheat.json");
  assert.equal(cheatSourceUrl("?cheatfixture=../x"), "/api/cheat");
  assert.equal(cheatSourceUrl("?cheatfixture="), "/api/cheat");
  assert.equal(cheatSourceUrl(""), "/api/cheat");
});

test("the fixture: marked, on the flail frames id, a card with a quote and an over-torque joint", () => {
  assert.equal(fixture.fixture, true);
  const doc = parseCheat(fixture);
  assert.ok(doc && doc.card);
  assert.equal(doc.fixture, true);
  assert.equal(doc.card.frames_id, "ffc64b4bfe41b185e9157074");
  assert.equal(isOverratedReject({ verdict: "rejected", origin: "model", reason: doc.card.reason, frames_id: doc.card.frames_id }), true);
  assert.ok(doc.card.quote && doc.card.quote.length <= QUOTE_MAX);
  assert.ok(overTorqueJoint(doc.card.peak_torque));
  assert.equal(counterLine(doc.counter), "over-rated gaits: 22 · passed the physics check: 0");
  assert.ok(Object.keys(doc.spine.models).length > 0);
});

test("parseCheat keeps a null card, drops a bad frames id, and rejects a malformed counter", () => {
  const ok = { card: null, counter: { overrated_gaits: 0, passed_physics: 0 }, spine: { models: {}, rewrite_cost_usd: null, edits: [] } };
  assert.deepEqual(parseCheat(ok), { fixture: false, ...ok });
  const badFrames = { ...ok, card: { ...fixture.card, frames_id: "../etc" } };
  assert.equal(parseCheat(badFrames)?.card, null);
  assert.equal(parseCheat({ ...ok, counter: { overrated_gaits: -1, passed_physics: 0 } }), null);
  assert.equal(parseCheat({ ...ok, counter: { overrated_gaits: 1, passed_physics: 2 } }), null);
  assert.equal(parseCheat({ ...ok, counter: { overrated: 1, accepted: 0, source: "count" } }), null);
  assert.equal(parseCheat({ ...ok, spine: { models: {} } }), null);
  assert.equal(parseCheat(null), null);
  const emptyQuote = parseCheat({ ...ok, card: { ...fixture.card, quote: "  " } });
  assert.equal(emptyQuote?.card?.quote, null);
});
