import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  FAST,
  HOLD_S,
  LOST,
  NORMAL,
  STAMP_NOTE,
  WON,
  attemptFrame,
  attemptsOf,
  attemptsSourceUrl,
  attemptsView,
  betText,
  earnedText,
  locate,
  orderAttempts,
  parseAttemptsDoc,
  phaseAt,
  primitiveTag,
  recordedTime,
  rejectedTotal,
  schedule,
  terrainLabel,
  toAttempt,
  typedChars,
  type Attempt,
  type AttemptsDoc,
} from "../lib/attempts.ts";
import type { VersionDoc } from "../lib/ghosts.ts";
import type { EditDoc } from "../lib/overlays.ts";

const fixture = JSON.parse(readFileSync(new URL("../public/fixtures/attempts.json", import.meta.url), "utf8"));
const REAL_FRAMES = new Set([
  "5ceaf7462ef2f8687170efb0",
  "6b8e6ec9ad5fffd33ba308d1",
  "92267354eb2aa3c7652eaa1c",
  "f5a45893d4ed0094c154a159",
  "0626564266f23349a20a7980",
]);

const v = (id: string, mean: number | null): VersionDoc => ({ _id: id, metrics: { train_mean_distance_m: mean } });
const e = (id: string, over: Partial<EditDoc> = {}): EditDoc => ({
  _id: id,
  from_version: "p",
  to_version: `c-${id}`,
  primitive: "rules",
  rationale: `rationale ${id}`,
  predicted_delta_m: 0.3,
  actual_delta_m: 0.1,
  attempt_frames_id: `f${id}`,
  verdict: "accepted",
  created_at: `2026-09-26T16:${id.padStart(2, "0")}:00Z`,
  ...over,
});
const attempt = (id: string, bet: number, measured: number, kept = true): Attempt => ({
  edit_id: id,
  created_at: `2026-09-26T16:${id.padStart(2, "0")}:00Z`,
  tag: "RULES",
  rationale: id,
  framesId: `f${id}`,
  terrain: null,
  bet,
  parentMean: 1,
  candidateMean: 1 + measured,
  measured,
  line: 1 + bet,
  reached: measured >= bet,
  kept,
});

test("primitives are tagged RULES, CONTEXT, TOOLS, MODEL, ENGINE", () => {
  assert.deepEqual(
    ["rules", "context_policy", "tools", "model_per_step", "engine"].map(primitiveTag),
    ["RULES", "CONTEXT", "TOOLS", "MODEL", "ENGINE"],
  );
});

test("an attempt reads its numbers from the edit and the two versions' train means", () => {
  const versions = new Map([["p", v("p", 0.5)], ["c-1", v("c-1", 0.9)]]);
  const a = toAttempt(e("1", { predicted_delta_m: 0.3, actual_delta_m: 0.4 }), versions, {});
  assert.ok(a);
  assert.equal(a.parentMean, 0.5);
  assert.equal(a.candidateMean, 0.9);
  assert.equal(a.line, 0.8);
  assert.equal(a.measured, 0.4);
  assert.equal(a.reached, true);
  assert.equal(a.kept, true);
  // No actual_delta_m: candidate minus parent.
  const b = toAttempt(e("1", { actual_delta_m: null }), versions, {});
  assert.ok(b);
  assert.ok(Math.abs(b.measured - 0.4) < 1e-12);
  // No candidate version: parent plus the measured change.
  const c = toAttempt(e("2", { actual_delta_m: -0.2 }), versions, {});
  assert.ok(c);
  assert.ok(Math.abs(c.candidateMean - 0.3) < 1e-12);
  assert.equal(c.reached, false);
});

test("edits without attempt_frames_id or predicted_delta_m are skipped; nothing is filled in", () => {
  const versions = new Map([["p", v("p", 0.5)]]);
  assert.equal(toAttempt(e("1", { attempt_frames_id: null }), versions, {}), null);
  assert.equal(toAttempt(e("1", { predicted_delta_m: null }), versions, {}), null);
  assert.equal(toAttempt(e("1", { attempt_frames_id: "../x" }), versions, {}), null);
  assert.equal(toAttempt(e("1", { verdict: null }), versions, {}), null);
  assert.equal(toAttempt(e("1", { from_version: "missing" }), versions, {}), null);
  assert.equal(toAttempt(e("1", { actual_delta_m: null }), versions, {}), null);
});

test("stamp is green only when the candidate reached the bet line", () => {
  assert.equal(WON, "#85BB65");
  assert.equal(LOST, "#FF4A3D");
  const versions = new Map([["p", v("p", 1)], ["c-1", v("c-1", 1.3)]]);
  assert.equal(toAttempt(e("1", { predicted_delta_m: 0.3, actual_delta_m: 0.3 }), versions, {})?.reached, true);
  assert.equal(toAttempt(e("1", { predicted_delta_m: 0.31, actual_delta_m: 0.3 }), versions, {})?.reached, false);
});

test("order: worst bet, then largest measured gain, then every other kept edit (oldest first, fast)", () => {
  const as = [
    attempt("1", 0.2, 0.3), // kept
    attempt("2", 0.9, -0.1, false), // worst bet
    attempt("3", 0.5, -0.4, false), // lost, smaller bet: not played
    attempt("4", 0.1, 0.8), // largest gain
    attempt("5", 0.3, 0.05), // kept
    attempt("6", 0.3, 0.2, false), // rejected, not the largest gain: not played
  ];
  const order = orderAttempts(as);
  assert.deepEqual(
    order.map((s) => [s.attempt.edit_id, s.fast]),
    [
      ["2", false],
      ["4", false],
      ["1", true],
      ["5", true],
    ],
  );
  // No lost bet: the largest gain opens; ties keep the older attempt.
  assert.deepEqual(
    orderAttempts([attempt("1", 0.1, 0.5), attempt("2", 0.1, 0.5)]).map((s) => s.attempt.edit_id),
    ["1", "2"],
  );
  assert.deepEqual(orderAttempts([]), []);
});

test("one attempt is 13 s: type 0-3, walk 3-10 at 1x, stamp 10-11.5, file 11.5-13; fast ones 1.5 s at 4x", () => {
  assert.equal(NORMAL.intro, 3);
  assert.equal(NORMAL.walkEnd, 10);
  assert.equal(NORMAL.stampEnd, 11.5);
  assert.equal(NORMAL.end, 13);
  assert.equal(NORMAL.speed, 1);
  assert.equal(FAST.end, 1.5);
  assert.equal(FAST.speed, 4);
  assert.deepEqual([0, 2.9, 3, 9.99, 10, 11.49, 11.5, 12.99].map((t) => phaseAt(NORMAL, t)), [
    "intro",
    "intro",
    "walk",
    "walk",
    "stamp",
    "stamp",
    "file",
    "file",
  ]);
});

test("the walk shows recorded frames only: frame 0 on the start mark, then 1x (or 4x), held after", () => {
  const fps = 10;
  const n = 100;
  assert.equal(attemptFrame(NORMAL, 0, fps, n), 0);
  assert.equal(attemptFrame(NORMAL, 2.99, fps, n), 0);
  assert.equal(attemptFrame(NORMAL, 4, fps, n), 10);
  assert.equal(attemptFrame(NORMAL, 10, fps, n), 70);
  assert.equal(attemptFrame(NORMAL, 12.5, fps, n), 70);
  assert.equal(recordedTime(FAST, FAST.intro + 0.5), 2);
  assert.equal(attemptFrame(FAST, FAST.intro + 0.5, fps, n), 20);
  // Never past the recording.
  assert.equal(attemptFrame(NORMAL, 10, fps, 30), 29);
  for (let t = 0; t < 13; t += 0.07) assert.ok(Number.isInteger(attemptFrame(NORMAL, t, fps, n)));
});

test("the rationale types in word for word by 2.6 s and stays whole", () => {
  const text = "Read the contact log after every preview.";
  assert.equal(typedChars(text, NORMAL, 0), 0);
  const mid = typedChars(text, NORMAL, 1.3);
  assert.ok(mid > 0 && mid < text.length);
  assert.equal(typedChars(text, NORMAL, NORMAL.typeEnd), text.length);
  assert.equal(typedChars(text, NORMAL, 12), text.length);
});

test("schedule and view: notebook fills with kept edits as each finishes filing, then holds and loops", () => {
  const sc = schedule(orderAttempts([attempt("1", 0.9, -0.1, false), attempt("2", 0.1, 0.5), attempt("3", 0.1, 0.2)]));
  assert.deepEqual(sc.slots.map((s) => s.start), [0, 13, 26]);
  assert.equal(sc.total, 26 + 1.5 + HOLD_S);
  assert.deepEqual(locate(sc, 14), { index: 1, local: 1, holding: false });
  assert.equal(attemptsView(sc, 12.9)?.notebook.length, 0);
  assert.equal(attemptsView(sc, 25.9)?.notebook.length, 0); // the rejected one never enters
  assert.deepEqual(attemptsView(sc, 26.1)?.notebook.map((a) => a.edit_id), ["2"]);
  const hold = attemptsView(sc, 28);
  assert.equal(hold?.holding, true);
  assert.deepEqual(hold?.notebook.map((a) => a.edit_id), ["2", "3"]);
  // Loops back to an empty notebook.
  assert.deepEqual(attemptsView(sc, sc.total + 0.5)?.notebook, []);
  assert.equal(attemptsView(schedule([]), 3), null);
});

test("labels: signed bet, earned meters, single walk vs the mean", () => {
  assert.equal(betText(0.4), "bets +0.40 m");
  assert.equal(betText(-0.1), "bets -0.10 m");
  assert.equal(earnedText(0.62), "+0.62 m");
  assert.equal(STAMP_NOTE, "mean over 6 terrains");
  assert.equal(terrainLabel({ task_id: "train-s0-f1", slope_deg: 0, friction: 1 }), "one walk · train-s0-f1 · 0° slope · friction 1");
  assert.equal(terrainLabel(null), "one walk · terrain not recorded");
});

test("?attemptsfixture= picks a fixture file; anything else is the live route", () => {
  assert.equal(attemptsSourceUrl("?mode=attempts&attemptsfixture=attempts"), "/fixtures/attempts.json");
  assert.equal(attemptsSourceUrl("?mode=attempts&attemptsfixture=../../etc"), "/api/attempts");
  assert.equal(attemptsSourceUrl("?mode=attempts"), "/api/attempts");
});

test("the tray counts every rejected edit in the data, played or not", () => {
  assert.equal(rejectedTotal([e("1", { verdict: "rejected" }), e("2"), e("3", { verdict: "rejected", attempt_frames_id: null })]), 2);
});

test("the fixture plays a lost bet, a won bet, then fills the notebook", () => {
  const doc = parseAttemptsDoc(fixture) as AttemptsDoc;
  assert.ok(doc);
  assert.equal(doc.fixture, true);
  assert.equal(parseAttemptsDoc({ versions: [], edits: {} }), null);
  const attempts = attemptsOf(doc);
  // e7 (no bet) and e8 (no walk) are skipped.
  assert.deepEqual(attempts.map((a) => a.edit_id), ["fixture.e1", "fixture.e2", "fixture.e3", "fixture.e4", "fixture.e5", "fixture.e6"]);
  for (const a of attempts) assert.ok(REAL_FRAMES.has(a.framesId), a.framesId);
  const sc = schedule(orderAttempts(attempts));
  const [lost, won, ...rest] = sc.slots;
  assert.equal(lost.attempt.edit_id, "fixture.e1");
  assert.equal(lost.attempt.measured < 0 && !lost.attempt.reached && !lost.attempt.kept, true);
  assert.equal(lost.attempt.tag, "ENGINE");
  assert.equal(lost.attempt.terrain?.task_id, "train-s0-f1");
  assert.equal(won.attempt.edit_id, "fixture.e2");
  assert.equal(won.attempt.reached && won.attempt.kept, true);
  assert.equal(won.fast, false);
  assert.deepEqual(rest.map((s) => [s.attempt.edit_id, s.fast]), [
    ["fixture.e4", true],
    ["fixture.e5", true],
    ["fixture.e6", true],
  ]);
  // Every number is the stored one.
  assert.equal(won.attempt.bet, 0.4);
  assert.ok(Math.abs(won.attempt.line - 0.6) < 1e-12);
  assert.equal(won.attempt.candidateMean, 0.82);
  // The notebook fills one kept edit at a time, ending with all four.
  const counts = [12, 14, 26.1, 27.6, 29.1, 30.6].map((t) => attemptsView(sc, t)?.notebook.length);
  assert.deepEqual(counts, [0, 0, 1, 2, 3, 4]);
  assert.equal(attemptsView(sc, 31)?.holding, true);
  assert.equal(rejectedTotal(doc.edits), 3);
});
