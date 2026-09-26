import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  BOOTH,
  OFF_SEQUENCE_S,
  SCENE_IDS,
  boothNext,
  boothSeconds,
  directorParams,
  parsePinned,
  parseScript,
  pinnedFramesIds,
  pinnedGhostSources,
  pinnedSlots,
  replayLabel,
  sceneForKey,
  v0Line,
} from "../lib/director.ts";

const cheatFixture = JSON.parse(readFileSync(new URL("../public/fixtures/cheat.json", import.meta.url), "utf8"));
const swapFixture = JSON.parse(readFileSync(new URL("../public/fixtures/swap.json", import.meta.url), "utf8"));
const script = JSON.parse(readFileSync(new URL("../public/script.json", import.meta.url), "utf8"));

// The shape api/snapshot/latest returned for today's pin (snapshot-20260926T175028149000).
const raw = {
  snapshot: {
    _id: "snapshot-20260926T175028149000",
    pinned_at: "2026-09-26T17:50:28.149Z",
    best_version: "v2",
    v0: "v0",
    frontier: "frontier",
    attempt_edit_ids: ["e1", 7, ""],
    cheat_edit_id: "r-20260926T163653-41e812.e0",
    swaps_id: null,
    scoreboard_id: null,
    ghosts: [
      { version_id: "v0", frames_id: "92267354eb2aa3c7652eaa1c" },
      { version_id: "v2", frames_id: "8b456c3c8a6efa5b35cb30a3" },
      { version_id: "v9", frames_id: null },
      { version_id: "vX", frames_id: "../etc" },
    ],
  },
  versions: [
    { _id: "v0", status: "baseline", metrics: { holdout_reliability_80: 0, cost_per_run_usd: 0.000521, n: 18 } },
    { _id: "v2", status: "accepted", metrics: { holdout_reliability_80: 0, cost_per_run_usd: 0.003994, n: 18 } },
  ],
  v0_walk: { frames_id: "92267354eb2aa3c7652eaa1c", distance_m: -0.0681, model_id: "openai/gpt-6-luna" },
  v0_holdout: { passed: 0, runs: 18 },
  rewrite_cost_usd: null,
  swap: swapFixture,
  card: cheatFixture.card,
  replay: { first: "2026-09-26T15:20:08.000Z", last: "2026-09-26T17:36:27.000Z" },
};

test("number keys 1..7 pick a scene; modifiers and other keys never do", () => {
  assert.deepEqual(
    SCENE_IDS.map((n) => sceneForKey(String(n))),
    [1, 2, 3, 4, 5, 6, 7],
  );
  for (const k of ["0", "8", "a", "Enter", "11", ""]) assert.equal(sceneForKey(k), null);
  assert.equal(sceneForKey("3", true), null);
});

test("booth cycles 1, 3, 4, 5, 7 with fixed durations; a jump continues from the jumped-to scene", () => {
  assert.deepEqual(
    BOOTH.map((b) => b.scene),
    [1, 3, 4, 5, 7],
  );
  assert.ok(BOOTH.every((b) => b.seconds > 0));
  let s = BOOTH[0].scene;
  const seen = [s];
  for (let i = 0; i < 5; i++) seen.push((s = boothNext(s)));
  assert.deepEqual(seen, [1, 3, 4, 5, 7, 1]);
  // Off-sequence jumps hold OFF_SEQUENCE_S, then continue with the next scene by number.
  assert.equal(boothNext(2), 3);
  assert.equal(boothNext(6), 7);
  assert.equal(boothSeconds(6), OFF_SEQUENCE_S);
  assert.equal(boothSeconds(3), BOOTH[1].seconds);
});

test("?mode=booth implies pinned; ?pinned=1 alone is the director; nothing else is", () => {
  assert.deepEqual(directorParams("?mode=booth"), { booth: true, pinned: true });
  assert.deepEqual(directorParams("?pinned=1"), { booth: false, pinned: true });
  assert.deepEqual(directorParams("?pinned=0"), { booth: false, pinned: false });
  assert.deepEqual(directorParams("?mode=swap"), { booth: false, pinned: false });
});

test("script.json has a caption for every booth scene; the parser drops bad keys and empties", () => {
  const parsed = parseScript(script);
  for (const b of BOOTH) assert.ok(parsed[b.scene], `caption for scene ${b.scene}`);
  assert.deepEqual(parseScript({ "1": " hi ", "8": "no", x: "no", "2": "  ", "3": 4 }), { 1: "hi" });
  assert.deepEqual(parseScript(null), {});
  assert.deepEqual(parseScript(["a"]), {});
});

test("parsePinned keeps what the scenes may use and nulls what they may not", () => {
  const p = parsePinned(raw);
  assert.ok(p);
  assert.deepEqual(p.snapshot.attempt_edit_ids, ["e1"]);
  assert.deepEqual(
    pinnedGhostSources(p).map((g) => g.versionId),
    ["v0", "v2"],
  );
  assert.equal(p.swap?.left_version, swapFixture.left_version);
  assert.equal(p.card?.frames_id, cheatFixture.card.frames_id);
  const want = new Set(["8b456c3c8a6efa5b35cb30a3", "92267354eb2aa3c7652eaa1c", cheatFixture.card.frames_id, swapFixture.left_frames_id, swapFixture.right_frames_id]);
  assert.deepEqual(pinnedFramesIds(p).sort(), [...want].sort());
  assert.equal(parsePinned({ ...raw, swap: { bogus: 1 }, card: { frames_id: "../x" } })?.swap, null);
  assert.equal(parsePinned({ ...raw, card: { frames_id: "../x" } })?.card, null);
  assert.equal(parsePinned({ snapshot: { _id: "x" } }), null);
  assert.equal(parsePinned(null), null);
});

test("scene 1's line is read from the pin: showcase distance and model, metrics cost, holdout runs passed", () => {
  const p = parsePinned(raw);
  assert.ok(p);
  assert.equal(v0Line(p), "-0.07 m · openai/gpt-6-luna · $0.0005 a gait · harness v0 · 0/18");
  assert.equal(v0Line({ ...p, v0_walk: null }), null);
  assert.equal(v0Line({ ...p, v0_holdout: { passed: 0, runs: 0 } }), null);
  assert.equal(v0Line({ ...p, versions: [] }), null);
});

test("the replay label reads the runs' own times and never says today", () => {
  assert.equal(replayLabel(raw.replay.first, raw.replay.last, "UTC"), "Replay of runs recorded Sep 26, 15:20–17:36");
  assert.equal(
    replayLabel("2026-09-26T23:50:00Z", "2026-09-27T00:10:00Z", "UTC"),
    "Replay of runs recorded Sep 26 23:50–Sep 27 00:10",
  );
  assert.equal(replayLabel(null, raw.replay.last), null);
});

test("scene 3 plays the pinned attempts in the pinned order; the worst bet and the largest gain stay at 1x", () => {
  const a = (edit_id: string, bet: number, measured: number, kept: boolean) =>
    ({ edit_id, bet, measured, kept }) as unknown as Parameters<typeof pinnedSlots>[0][number];
  const all = [a("e1", 0.5, 0.2, true), a("e2", 0.9, -0.3, false), a("e3", 0.1, 0.8, true), a("e4", 0.2, 0.1, true), a("e5", 0.1, 0.1, true)];
  const slots = pinnedSlots(all, ["e2", "e3", "e1", "e4", "gone"]);
  assert.deepEqual(
    slots.map((s) => [s.attempt.edit_id, s.fast]),
    [
      ["e2", false],
      ["e3", false],
      ["e1", true],
      ["e4", true],
    ],
  );
  assert.deepEqual(pinnedSlots(all, []), []);
});
