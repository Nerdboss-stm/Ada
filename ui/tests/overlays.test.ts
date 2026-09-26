import { test } from "node:test";
import assert from "node:assert/strict";
import {
  GATE_STAGES,
  HOLDOUT_TASKS,
  claim,
  claimLine,
  formatDelta,
  formatMeters,
  gateCells,
  latestEdit,
  mergeEdits,
  oldToNew,
  pickBest,
  pickFrontier,
  toRow,
  type EditDoc,
} from "../lib/overlays.ts";
import type { VersionDoc } from "../lib/ghosts.ts";
import type { AdaEvent } from "../lib/stream.ts";

let seq = 0;
const ev = (edit_id: string | null, stage: string, status: AdaEvent["status"], sec: number): AdaEvent => ({
  _id: `e${seq++}`,
  ts: new Date(Date.UTC(2026, 8, 26, 13, 0, sec)).toISOString(),
  round_id: null,
  version_id: null,
  edit_id,
  stage,
  status,
  payload: {},
});

const states = (events: AdaEvent[], id: string | null) => gateCells(events, id).map((c) => c.state);

test("cells are the four gate stages in fixed order", () => {
  assert.deepEqual(
    gateCells([], "x").map((c) => c.stage),
    ["gate.verifier", "gate.gpa", "gate.meta", "gate.constraints"],
  );
  assert.deepEqual(GATE_STAGES, ["gate.verifier", "gate.gpa", "gate.meta", "gate.constraints"]);
});

test("cells fill one event at a time: start, then pass or fail", () => {
  const all = [
    ev("x", "gate.verifier", "start", 0),
    ev("x", "gate.verifier", "pass", 1),
    ev("x", "gate.gpa", "start", 2),
    ev("x", "gate.gpa", "pass", 3),
    ev("x", "gate.meta", "start", 4),
    ev("x", "gate.meta", "fail", 5),
  ];
  const seen = [0, 1, 2, 3, 4, 5, 6].map((k) => states(all.slice(0, k), "x"));
  assert.deepEqual(seen, [
    ["waiting", "waiting", "waiting", "waiting"],
    ["running", "waiting", "waiting", "waiting"],
    ["passed", "waiting", "waiting", "waiting"],
    ["passed", "running", "waiting", "waiting"],
    ["passed", "passed", "waiting", "waiting"],
    ["passed", "passed", "running", "waiting"],
    ["passed", "passed", "failed", "waiting"],
  ]);
});

test("cells ignore other edits, other stages, info, and a late start", () => {
  const events = [
    ev("y", "gate.verifier", "fail", 0),
    ev("x", "actuator", "pass", 1),
    ev("x", "gate.gpa", "info", 2),
    ev("x", "gate.verifier", "pass", 3),
    ev("x", "gate.verifier", "start", 1), // arrived out of order; the result still stands
  ];
  assert.deepEqual(states(events, "x"), ["passed", "waiting", "waiting", "waiting"]);
  assert.deepEqual(states(events, null), ["waiting", "waiting", "waiting", "waiting"]);
});

test("latest edit is newest created_at; merge dedupes with incoming winning", () => {
  const a: EditDoc = { _id: "a", created_at: "2026-09-26T13:00:00Z" };
  const b: EditDoc = { _id: "b", created_at: "2026-09-26T13:05:00Z" };
  assert.equal(latestEdit([a, b])?._id, "b");
  assert.equal(latestEdit([]), null);
  const merged = mergeEdits([a, b], [{ ...a, verdict: "rejected" }]);
  assert.equal(merged.length, 2);
  assert.equal(merged.find((e) => e._id === "a")?.verdict, "rejected");
});

test("old → new is one line and clipped", () => {
  assert.equal(oldToNew("a\n  b", ["c"]), 'a b → ["c"]');
  assert.equal(oldToNew(null, "x"), "– → x");
  const line = oldToNew("x".repeat(100), "y", 10);
  assert.equal(line, `${"x".repeat(9)}… → y`);
  assert.ok(!line.includes("\n"));
});

test("deltas and meters format for the screen", () => {
  assert.equal(formatDelta(0.123), "+0.12");
  assert.equal(formatDelta(-0.5), "-0.50");
  assert.equal(formatDelta(-0.001), "0.00");
  assert.equal(formatDelta(null), null);
  assert.equal(formatMeters(3.456), "3.46 m");
});

// r is both train and holdout reliability unless `train` is given: the leader reads train only.
const ver = (_id: string, status: string, r: number, c: number, n = 30, d = 1, train = r): VersionDoc => ({
  _id,
  status,
  metrics: { train_reliability: train, holdout_reliability_80: r, cost_per_run_usd: c, n, mean_distance_m: d },
});

test("rows show whole-percent reliability, terrains passed of 6, n, and 4-place cost", () => {
  const row = toRow(ver("v3", "accepted", 0.834, 0.00123, 30));
  assert.deepEqual(
    row && { r: row.reliability, t: row.tasksPassed, c: row.cost, n: row.n },
    { r: "83%", t: 5, c: "$0.0012", n: 30 },
  );
  assert.equal(HOLDOUT_TASKS, 6);
  assert.equal(toRow({ _id: "v4", status: "accepted", metrics: null }), null);
});

test("best is the train-reliability-first leader; frontier is the frozen version", () => {
  const vs = [
    ver("v0", "baseline", 0.5, 0.001),
    ver("v1", "accepted", 0.9, 0.001, 30, 2),
    ver("v2", "accepted", 0.6, 0.001, 30, 40), // walks far, unreliable
    ver("vF", "frontier", 0.9, 0.01),
  ];
  assert.equal(pickBest(vs)?._id, "v1");
  assert.equal(pickFrontier(vs)?._id, "vF");
  assert.equal(pickBest([ver("vF", "frontier", 1, 0.001)]), null);
  // Better holdout does not make a version best.
  assert.equal(pickBest([ver("v1", "accepted", 0.2, 0.001, 30, 1, 0.9), ver("v2", "accepted", 1, 0.001, 30, 1, 0.5)])?._id, "v1");
});

const r6 = (tasks: number) => tasks / 6;

test("claim: [B5] rule on displayed values", () => {
  const f = toRow(ver("vF", "frontier", r6(1), 0.0100));
  // 3 of 6 vs 1 of 6, 0.0100 / 0.0030 = 3.33 -> floored to 3.3.
  const c = claim(f, toRow(ver("v1", "accepted", r6(3), 0.00296)));
  assert.deepEqual(c, { b: 3, f: 1, k: "3.3" });
  assert.equal(
    claimLine(c!),
    "3 of 6 unseen terrains vs the frontier model at 1 of 6, at 3.3x lower cost per gait",
  );
  // Equal terrains: not strictly greater, no claim.
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(1), 0.001))), null);
  // Fewer terrains: no claim.
  assert.equal(claim(toRow(ver("vF", "frontier", r6(4), 0.01)), toRow(ver("v1", "accepted", r6(3), 0.001))), null);
  // No frontier or no best: no claim.
  assert.equal(claim(null, toRow(ver("v1", "accepted", r6(3), 0.001))), null);
  assert.equal(claim(f, null), null);
});

test("claim: both at 0 of 6 shows no claim (today's frontier and v0)", () => {
  const frontier = toRow(ver("frontier", "frontier", 0, 0.031978, 18));
  const v0 = toRow(ver("v0", "baseline", 0, 0.000521, 18));
  assert.equal(frontier?.tasksPassed, 0);
  assert.equal(v0?.tasksPassed, 0);
  assert.equal(claim(frontier, v0), null);
});

test("claim: best must pass more than 0 terrains even if the frontier is lower", () => {
  // A negative or rounding-to-0 frontier cannot make 0 of 6 a claim.
  assert.equal(claim(toRow(ver("vF", "frontier", 0, 0.01)), toRow(ver("v1", "accepted", 0.05, 0.001))), null);
});

test("claim: cost must display above $0.0000 and k above 1.0; no 1.5 minimum", () => {
  const f = toRow(ver("vF", "frontier", r6(1), 0.0100));
  // Best cost displays as $0.0000: no unbounded ratio.
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(3), 0.00004))), null);
  // Cost not lower at 4 places: 0.01004 shows $0.0100, k = 1.0.
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(3), 0.01004))), null);
  // Costlier than the frontier: k < 1.
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(3), 0.02))), null);
  // 0.0100 / 0.0099 = 1.01 floors to "1.0": not above 1.0.
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(3), 0.0099))), null);
  // 0.0100 / 0.0090 = 1.11 floors to "1.1": earned (the old 1.5 bar is gone).
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(3), 0.009)))?.k, "1.1");
  // 0.0100 / 0.0070 = 1.43 floors to "1.4": earned.
  assert.equal(claim(f, toRow(ver("v1", "accepted", r6(3), 0.007)))?.k, "1.4");
});

test("claim line never says matches, beats, or smarter", () => {
  const line = claimLine({ b: 2, f: 0, k: "12.5" });
  for (const word of ["matches", "beats", "smarter", "×"]) assert.ok(!line.includes(word), word);
});
