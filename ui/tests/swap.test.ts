import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  PHYSICS_HASH_CHARS,
  SWAP_LANE_Y,
  clockTime,
  laneTorso,
  parseSwap,
  swapCamera,
  swapDiff,
  swapHeadline,
  swapSourceUrl,
  swapSubline,
  type SwapDoc,
} from "../lib/swap.ts";

const fixture = JSON.parse(readFileSync(new URL("../public/fixtures/swap.json", import.meta.url), "utf8"));

const good: SwapDoc = {
  left_version: "v0",
  right_version: "v7",
  task_id: "holdout-s3-f0.45",
  left_frames_id: "92267354eb2aa3c7652eaa1c",
  right_frames_id: "6b8e6ec9ad5fffd33ba308d1",
  left_holdout: 0,
  right_holdout: 3,
  model_id: "openai/gpt-6-luna",
  verifier_sha: "5e2fe18c3cd6816003f11383f61cb07e35e594b02e09b4920d5db0def1fe3b84",
  mujoco_version: "3.14.0",
  manifest_version: "cd5f83ef0ea3",
  harness_diff: ["+ a", "- b"],
  captured_at: "2026-09-26T12:40:00Z",
};

test("the swap fixture parses, uses the named frames ids, and is marked fixture", () => {
  const d = parseSwap(fixture);
  assert.ok(d);
  assert.equal(d.fixture, true);
  assert.equal(d.left_frames_id, "92267354eb2aa3c7652eaa1c");
  assert.equal(d.right_frames_id, "6b8e6ec9ad5fffd33ba308d1");
  assert.equal(d.harness_diff.length, 3);
  assert.ok(d.harness_diff.every((l) => l.includes("fixture")));
});

test("parseSwap rejects missing fields, bad ids, and out-of-range holdout counts", () => {
  assert.deepEqual(parseSwap(good), good);
  assert.equal(parseSwap(null), null);
  assert.equal(parseSwap({ ...good, model_id: undefined }), null);
  assert.equal(parseSwap({ ...good, left_frames_id: "../etc" }), null);
  assert.equal(parseSwap({ ...good, right_holdout: 7 }), null);
  assert.equal(parseSwap({ ...good, right_holdout: 2.5 }), null);
  assert.equal(parseSwap({ ...good, harness_diff: "+ a" }), null);
  assert.equal(parseSwap({ ...good, captured_at: "not a time" }), null);
});

test("source: ?swapfixture=name loads the fixture file, otherwise the newest swaps document", () => {
  assert.equal(swapSourceUrl("?mode=swap&swapfixture=swap"), "/fixtures/swap.json");
  assert.equal(swapSourceUrl("?mode=swap"), "/api/swaps/latest");
  assert.equal(swapSourceUrl("?mode=swap&swapfixture=../../secret"), "/api/swaps/latest");
});

test("headline and subline read the document word for word", () => {
  assert.equal(swapHeadline(good), "holdout 0/6 → 3/6");
  assert.equal(clockTime("2026-09-26T12:40:00Z", "UTC"), "12:40");
  assert.equal(clockTime("2026-09-26T00:05:00Z", "UTC"), "00:05");
  assert.equal(swapSubline(good.captured_at, "UTC"), "tasks the loop never saw · recorded back-to-back at 12:40");
});

test("diff: model, task, physics hash, sim version identical; harness expands into its lines", () => {
  const rows = swapDiff(good);
  assert.deepEqual(
    rows.map((r) => [r.label, r.identical]),
    [["model", true], ["task", true], ["physics hash", true], ["sim version", true], ["harness", false]],
  );
  assert.equal(rows[2].left, "5e2fe18c3cd6");
  assert.equal(rows[2].left.length, PHYSICS_HASH_CHARS);
  assert.equal(rows[3].left, "mujoco 3.14.0 · manifest cd5f83ef0ea3");
  assert.deepEqual([rows[4].left, rows[4].right, rows[4].lines], ["v0", "v7", ["+ a", "- b"]]);
});

test("lanes: same start line, v0 on screen left (MuJoCo +y), best on the right", () => {
  assert.ok(SWAP_LANE_Y.left > 0 && SWAP_LANE_Y.right < 0);
  // Same recorded torso, different lanes: same x (start line), mirrored three z.
  const l = laneTorso([0, 0, 0.5], SWAP_LANE_Y.left);
  const r = laneTorso([0, 0, 0.5], SWAP_LANE_Y.right);
  assert.equal(l[0], r[0]);
  assert.equal(l[2], -SWAP_LANE_Y.left);
  assert.equal(r[2], -SWAP_LANE_Y.right);
});

test("camera frames both torsos: centered between them, pulling back as they spread", () => {
  const near = swapCamera([0, 0.5, -0.9], [0, 0.5, 0.9]);
  assert.equal(near.pos[2], 0);
  assert.ok(near.pos[0] < 0 && near.pos[1] > 0.5);
  const far = swapCamera([0, 0.5, -0.9], [6, 0.5, 0.9]);
  // Behind the rear torso, and farther back relative to the midpoint than when level.
  assert.ok(far.pos[0] < 0);
  assert.ok(3 - far.pos[0] > 0 - near.pos[0]);
  assert.ok(far.look[0] > 3);
});
