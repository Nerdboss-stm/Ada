import { test } from "node:test";
import assert from "node:assert/strict";
import {
  GATE_STAGES,
  claimK,
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

const ver = (_id: string, status: string, r: number, c: number, n = 30, d = 1): VersionDoc => ({
  _id,
  status,
  metrics: { holdout_reliability_80: r, cost_per_run_usd: c, n, mean_distance_m: d },
});

test("rows show whole-percent reliability, n, and 4-place cost", () => {
  const row = toRow(ver("v3", "accepted", 0.834, 0.00123, 30));
  assert.deepEqual(
    row && { r: row.reliability, c: row.cost, n: row.n },
    { r: "83%", c: "$0.0012", n: 30 },
  );
  assert.equal(toRow({ _id: "v4", status: "accepted", metrics: null }), null);
});

test("best is the reliability-first leader; frontier is the frozen version", () => {
  const vs = [
    ver("v0", "baseline", 0.5, 0.001),
    ver("v1", "accepted", 0.9, 0.001, 30, 2),
    ver("v2", "accepted", 0.6, 0.001, 30, 40), // walks far, unreliable
    ver("vF", "frontier", 0.9, 0.01),
  ];
  assert.equal(pickBest(vs)?._id, "v1");
  assert.equal(pickFrontier(vs)?._id, "vF");
  assert.equal(pickBest([ver("vF", "frontier", 1, 0.001)]), null);
});

test("claim holds only on displayed values (SPEC §0)", () => {
  const f = toRow(ver("vF", "frontier", 0.9, 0.0100));
  // Equal at displayed precision: 0.904 and 0.896 both show 90%.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.896, 0.0020))), "5.0");
  // 0.834 shows 83%, 0.835 shows 84%: below the frontier on screen, no claim.
  assert.equal(
    claimK(toRow(ver("vF", "frontier", 0.835, 0.01)), toRow(ver("v1", "accepted", 0.834, 0.001))),
    null,
  );
  // Cost not lower at 4 places: 0.01004 shows $0.0100.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.01004))), null);
  // Cost that displays as $0.0000: no unbounded ratio.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.00004))), null);
  // k from the rounded costs, floored to one decimal: 0.0100 / 0.0030 = 3.33 -> 3.3.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.00296))), "3.3");
  // Displayed k must be at least 1.5: 0.0100 / 0.0099 = 1.01 -> "1.0", 0.0100 / 0.0070 = 1.43 -> "1.4".
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.0099))), null);
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.0070))), null);
  // 0.0100 / 0.0066 = 1.515 -> "1.5": exactly at the bar, earned.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.0066))), "1.5");
  // 0.0100 / 0.0067 = 1.4925 would round to 1.5 but floors to 1.4: not earned.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.0067))), null);
  // n must be printed and positive.
  assert.equal(claimK(f, toRow(ver("v1", "accepted", 0.95, 0.001, 0))), null);
  assert.equal(claimK(null, toRow(ver("v1", "accepted", 0.95, 0.001))), null);
  assert.equal(claimLine("3.3"), "matches frontier reliability at 3.3× lower cost");
});
