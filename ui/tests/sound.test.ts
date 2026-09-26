import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { FALL_Z_M, MAX_FALLS, SFX, VoiceLimit, crossedFrames, fallFrame } from "../lib/sound.ts";
import type { Frame, GeomPose, Manifest } from "../lib/replay.ts";

const manifest = JSON.parse(readFileSync(new URL("../public/manifest.json", import.meta.url), "utf8")) as Manifest;

test("clip lengths live in one table: step 0.5 s, fall 1.0 s, glitch 2.0 s; files exist", () => {
  assert.deepEqual(
    Object.fromEntries(Object.entries(SFX).map(([k, v]) => [k, v.clipS])),
    { step: 0.5, fall: 1.0, glitch: 2.0 },
  );
  for (const { url } of Object.values(SFX)) assert.ok(existsSync(new URL(`../public${url}`, import.meta.url)), url);
  assert.equal(MAX_FALLS, 6);
});

test("crossedFrames fires each cue frame once per pass, including across a loop restart", () => {
  const cues = [3, 10, 19];
  assert.deepEqual(crossedFrames(cues, -1, 0), []);
  assert.deepEqual(crossedFrames(cues, -1, 3), [3]);
  assert.deepEqual(crossedFrames(cues, 3, 3), []); // held frame: no repeat
  assert.deepEqual(crossedFrames(cues, 3, 12), [10]); // a skipped-over frame still fires
  assert.deepEqual(crossedFrames(cues, 18, 19), [19]);
  // Loop restart: the rest of the old pass, then the start of the new one.
  assert.deepEqual(crossedFrames(cues, 15, 4), [3, 19]);
  assert.deepEqual(crossedFrames(cues, 19, 0), []);
});

test("bullet time holds frame 19 many rendered frames; the glitch fires once", () => {
  const shown = [17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 21];
  let prev = -1;
  let fired = 0;
  for (const cur of shown) {
    fired += crossedFrames([19], prev, cur).length;
    prev = cur;
  }
  assert.equal(fired, 1);
});

const pose = (z: number, qx = 0, qy = 0): GeomPose => [0, 0, z, 1, qx, qy, 0];
const frame = (z: number, qx = 0, qy = 0): Frame => ({
  geoms: manifest.geoms.map(() => pose(z, qx, qy)),
  contacts: [false, false, false, false],
  forces: [],
  torso: [0, 0, z],
});

test("fallFrame uses the verifier's rule: torso low, or upside down", () => {
  assert.equal(fallFrame({ frames: [frame(0.55), frame(0.5), frame(FALL_Z_M - 0.01)] }, manifest), 2);
  assert.equal(fallFrame({ frames: [frame(0.55), frame(0.55)] }, manifest), null);
  // Flipped: 1 - 2(qx² + qy²) < 0 at qx = 0.8.
  assert.equal(fallFrame({ frames: [frame(0.55), frame(0.6, 0.8)] }, manifest), 1);
  // Exactly at the bound is not a fall (strictly below, as in sim/verifier.py).
  assert.equal(fallFrame({ frames: [frame(FALL_Z_M)] }, manifest), null);
});

test("fall voices: at most six at once, a slot frees when one ends", () => {
  const v = new VoiceLimit(MAX_FALLS);
  const got = Array.from({ length: 8 }, () => v.acquire());
  assert.deepEqual(got, [true, true, true, true, true, true, false, false]);
  v.release();
  assert.equal(v.acquire(), true);
  assert.equal(v.acquire(), false);
  for (let i = 0; i < 10; i++) v.release();
  assert.equal(v.count, 0);
});
