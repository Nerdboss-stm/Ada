import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LANE_Y,
  RETURN_S,
  SLOW,
  cheatState,
  cheatTimeline,
  cheatTorso,
  isCheatDoc,
  latestRejectedEdit,
  orbitCamera,
  pickCheatDoc,
  recordedTime,
  verdictText,
  type CheatTimeline,
} from "../lib/cheat.ts";
import type { FramesDoc } from "../lib/replay.ts";

// The flail fixture: 333 frames at 33.3333 fps, violation at frame 19.
const FPS = 33.3333;
const N = 333;
const V = 19;
const close = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);

function tl(): CheatTimeline {
  const t = cheatTimeline(FPS, N, V);
  assert.ok(t);
  return t;
}

test("bullet time covers 0.3 s before to 0.2 s after the violation frame, over 5 s of wall time", () => {
  const t = tl();
  close(t.slowFrom, V / FPS - 0.3);
  close(t.slowTo, V / FPS + 0.2);
  close(t.slowWall, 5);
  close(t.easeIn, 1);
  // One pass: the run at 1× plus the extra 4.5 s bullet time adds.
  close(t.loop, N / FPS + 4.5);
});

test("recorded time: 1× before, 0.1× during, 1× after bullet time, continuous at both joins", () => {
  const t = tl();
  close(recordedTime(t, 0.1), 0.1);
  close(recordedTime(t, t.slowFrom), t.slowFrom);
  close(recordedTime(t, t.slowFrom + 2), t.slowFrom + 2 * SLOW);
  close(recordedTime(t, t.slowFrom + t.slowWall), t.slowTo);
  close(recordedTime(t, t.slowFrom + t.slowWall + 1), t.slowTo + 1);
});

test("the cheat shows every recorded frame once, in order, never skipping or inventing one", () => {
  const t = tl();
  const seen: number[] = [];
  let last = -1;
  for (let w = 0; w < t.loop; w += 1 / 240) {
    const { frame } = cheatState(t, w);
    assert.ok(Number.isInteger(frame) && frame >= 0 && frame < N);
    assert.ok(frame >= last, `frame went back at w=${w}`);
    assert.ok(frame - last <= 1 || last === -1, `skipped a frame at w=${w}`);
    if (frame !== last) seen.push(frame);
    last = frame;
  }
  assert.deepEqual(seen, Array.from({ length: N }, (_, k) => k));
  // During bullet time each frame is held 10× longer: 1 / fps / 0.1 = 0.3 s.
  const holdStart = t.slowFrom + (V / FPS - t.slowFrom) / SLOW;
  assert.equal(cheatState(t, holdStart + 0.01).frame, V);
  assert.equal(cheatState(t, holdStart + 0.29).frame, V);
  assert.equal(cheatState(t, holdStart + 0.31).frame, V + 1);
});

test("the verdict appears exactly when the violation frame is shown and leaves when the camera is back", () => {
  const t = tl();
  const slowEnd = t.slowFrom + t.slowWall;
  let first = -1;
  for (let w = 0; w < t.loop; w += 1 / 1000) {
    const s = cheatState(t, w);
    if (s.verdict && first < 0) first = w;
    if (first < 0) assert.ok(s.frame < V, `frame ${s.frame} without a verdict`);
    else if (w < slowEnd + RETURN_S - 1e-3) assert.ok(s.verdict, `verdict dropped at w=${w}`);
  }
  assert.equal(cheatState(t, first).frame, V);
  assert.equal(cheatState(t, first - 1 / 1000).frame, V - 1);
  // 0.3 s of recorded lead at 0.1× puts the violation 3 s into bullet time.
  close(first, t.slowFrom + 3, 2e-3);
  assert.equal(cheatState(t, slowEnd + RETURN_S + 1e-3).verdict, false);
});

test("camera: on the leader, 1 s ease to the flail, 4 s orbit, 1.5 s back, on the leader again", () => {
  const t = tl();
  const a = t.slowFrom;
  assert.equal(cheatState(t, a - 0.01).camWeight, 0);
  assert.equal(cheatState(t, a).camWeight, 0);
  const mid = cheatState(t, a + 0.5).camWeight;
  assert.ok(mid > 0 && mid < 1);
  assert.equal(cheatState(t, a + 0.5).orbit, 0);
  assert.equal(cheatState(t, a + 1).camWeight, 1);
  close(cheatState(t, a + 1).orbit, 0);
  close(cheatState(t, a + 3).orbit, 0.5);
  assert.equal(cheatState(t, a + 4.99).camWeight, 1);
  const back = cheatState(t, a + 5 + RETURN_S / 2).camWeight;
  assert.ok(back > 0 && back < 1);
  assert.equal(cheatState(t, a + 5 + RETURN_S).camWeight, 0);
  assert.equal(cheatState(t, t.loop - 0.01).camWeight, 0);
});

test("the pass loops: the next pass starts over at frame 0 with the camera on the leader", () => {
  const t = tl();
  const s = cheatState(t, t.loop + 0.001);
  assert.equal(s.frame, 0);
  assert.equal(s.camWeight, 0);
  assert.equal(s.verdict, false);
  assert.equal(cheatState(t, -1).frame, 0);
  assert.equal(cheatState(t, Number.NaN).frame, 0);
});

test("a violation at the edges clamps bullet time to the run and keeps the camera return", () => {
  const early = cheatTimeline(FPS, N, 0);
  assert.ok(early);
  assert.equal(early.slowFrom, 0);
  close(early.slowWall, 2);
  assert.equal(cheatState(early, 0).frame, 0);
  assert.equal(cheatState(early, 0).verdict, true);
  const late = cheatTimeline(FPS, N, N - 1);
  assert.ok(late);
  close(late.slowTo, N / FPS);
  assert.ok(late.loop >= late.slowFrom + late.slowWall + RETURN_S);
  assert.equal(cheatState(late, late.loop - 0.01).frame, N - 1);
  assert.equal(cheatTimeline(FPS, N, N), null);
  assert.equal(cheatTimeline(FPS, N, -1), null);
  assert.equal(cheatTimeline(FPS, N, 1.5), null);
  assert.equal(cheatTimeline(0, N, V), null);
  assert.equal(cheatTimeline(FPS, 0, 0), null);
});

function doc(kind: FramesDoc["kind"], violation_frame: number | null | undefined, n = 30): FramesDoc {
  const frame = { geoms: [], contacts: [false, false, false, false], forces: [], torso: [0, 0, 0.6] as [number, number, number] };
  return {
    _id: "x",
    run_id: "r",
    version_id: "v",
    kind,
    manifest_version: "m",
    fps: FPS,
    frames: Array.from({ length: n }, () => frame),
    sha256: "s",
    violation_frame,
  };
}

test("a cheat is a rejected frames doc with an integer violation frame inside the run", () => {
  assert.equal(isCheatDoc(doc("rejected", 19)), true);
  assert.equal(isCheatDoc(doc("rejected", 0)), true);
  assert.equal(isCheatDoc(doc("rejected", null)), false);
  assert.equal(isCheatDoc(doc("rejected", undefined)), false);
  assert.equal(isCheatDoc(doc("rejected", 19.5)), false);
  assert.equal(isCheatDoc(doc("rejected", 30)), false);
  assert.equal(isCheatDoc(doc("showcase", 19)), false);
  assert.equal(
    pickCheatDoc([
      { key: "trot", doc: doc("showcase", null) },
      { key: "bad", doc: doc("rejected", null) },
      { key: "flail", doc: doc("rejected", 19) },
      { key: "flail2", doc: doc("rejected", 5) },
    ]),
    "flail",
  );
  assert.equal(pickCheatDoc([{ key: "trot", doc: doc("showcase", null) }]), null);
});

test("stream mode: the newest rejected edit with frames", () => {
  const edits = [
    { _id: "e1", verdict: "rejected" as const, frames_id: "f1", reason: "old", created_at: "2026-09-26T10:00:00Z" },
    { _id: "e2", verdict: "accepted" as const, frames_id: "f2", created_at: "2026-09-26T12:00:00Z" },
    { _id: "e3", verdict: "rejected" as const, frames_id: null, reason: "no frames", created_at: "2026-09-26T13:00:00Z" },
    { _id: "e4", verdict: "rejected" as const, frames_id: "f4", reason: "exceeds Ada rated motors", created_at: "2026-09-26T11:00:00Z" },
  ];
  assert.equal(latestRejectedEdit(edits)?._id, "e4");
  assert.equal(latestRejectedEdit(edits.slice(1, 3)), null);
});

test("verdict: the edit's reason word for word, else 'rejected at frame {v}'", () => {
  assert.equal(verdictText("exceeds Ada rated motors", 19), "exceeds Ada rated motors");
  assert.equal(verdictText("  body speed exceeds physical bound; exploits the simulator ", 3), "  body speed exceeds physical bound; exploits the simulator ");
  assert.equal(verdictText(null, 19), "rejected at frame 19");
  assert.equal(verdictText(undefined, 19), "rejected at frame 19");
  assert.equal(verdictText("   ", 7), "rejected at frame 7");
});

test("the cheat's lane and orbit: beside the leader, once around the torso, looking at it", () => {
  const c = cheatTorso([2, 0, 3]);
  assert.deepEqual(c, [2, 3, -LANE_Y]);
  const start = orbitCamera(c, 0);
  const end = orbitCamera(c, 1);
  const half = orbitCamera(c, 0.5);
  assert.deepEqual(start.look, c);
  for (let k = 0; k < 3; k++) close(start.pos[k], end.pos[k]);
  const r = (p: number[]) => Math.hypot(p[0] - c[0], p[2] - c[2]);
  close(r(start.pos), r(half.pos));
  // Opposite side of the torso halfway round.
  close(start.pos[0] - c[0], -(half.pos[0] - c[0]));
  assert.ok(start.pos[1] > c[1]);
});
