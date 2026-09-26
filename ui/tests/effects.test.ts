import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  CHATTER_S,
  FOOTPRINT_FADE_S,
  GLOW_MAX,
  activeFootprints,
  footprintEdges,
  glowIntensity,
  legForceIndex,
  HOT_EMISSIVE_LINEAR,
  HOT_GLOW_MAX,
  HOT_GLOW_MIN,
  hotGlowIntensity,
} from "../lib/effects.ts";
import type { Frame, GeomPose, Manifest } from "../lib/replay.ts";

const manifest = JSON.parse(readFileSync(new URL("../public/manifest.json", import.meta.url), "utf8")) as Manifest;
const at = (name: string) => manifest.geoms.findIndex((g) => g.name === name);
const force = (joint: string) => manifest.force_joints.indexOf(joint);

test("legs map to the joint that drives their body; torso and hip stubs never glow", () => {
  const map = legForceIndex(manifest);
  assert.equal(map.length, manifest.geoms.length);
  assert.equal(map[at("left_leg_geom")], force("hip_1"));
  assert.equal(map[at("left_ankle_geom")], force("ankle_1"));
  assert.equal(map[at("right_leg_geom")], force("hip_2"));
  assert.equal(map[at("right_ankle_geom")], force("ankle_2"));
  assert.equal(map[at("back_leg_geom")], force("hip_3"));
  assert.equal(map[at("third_ankle_geom")], force("ankle_3"));
  assert.equal(map[at("rightback_leg_geom")], force("hip_4"));
  assert.equal(map[at("fourth_ankle_geom")], force("ankle_4"));
  for (const name of ["torso_geom", "aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom"]) assert.equal(map[at(name)], -1, name);
  // Every force joint drives exactly one geom.
  assert.deepEqual(
    map.filter((k) => k >= 0).sort((a, b) => a - b),
    manifest.force_joints.map((_, k) => k),
  );
});

test("a leg whose lower body holds no contact geom gets no glow, not a guess", () => {
  const broken: Manifest = { ...manifest, contact_geoms: manifest.contact_geoms.filter((n) => n !== "left_ankle_geom") };
  const map = legForceIndex(broken);
  assert.equal(map[at("left_leg_geom")], -1);
  assert.equal(map[at("left_ankle_geom")], -1);
  assert.equal(map[at("right_ankle_geom")], force("ankle_2"));
});

test("glow follows |force|; zero force, zero glow", () => {
  assert.equal(glowIntensity(0), 0);
  assert.equal(glowIntensity(-0), 0);
  assert.equal(glowIntensity(0.5), GLOW_MAX * 0.5);
  assert.equal(glowIntensity(-0.5), GLOW_MAX * 0.5);
  assert.equal(glowIntensity(1), GLOW_MAX);
  assert.equal(glowIntensity(undefined), 0);
  assert.equal(glowIntensity(Number.NaN), 0);
});

// Frames where only the contacts and the ankle geoms' positions matter.
function frame(contacts: boolean[], x: number): Frame {
  const geoms: GeomPose[] = manifest.geoms.map((_, g) => [x + g, 10 + g, 0.3, 1, 0, 0, 0]);
  return { geoms, contacts, forces: new Array(8).fill(0), torso: [x, 0, 0.6] };
}

test("footprints are contact rising edges only, at that ankle geom's recorded position", () => {
  const F = false;
  const T = true;
  const frames = [
    frame([T, F, F, F], 0), // frame 0 is never an edge
    frame([T, T, F, F], 1), // leg 1 rises
    frame([T, T, F, F], 2), // held: nothing
    frame([F, T, T, F], 3), // leg 0 lifts, leg 2 rises
    frame([T, F, T, F], 4), // leg 0 rises again
  ];
  const edges = footprintEdges({ frames, fps: 10 }, manifest);
  const ankle = (leg: number) => at(manifest.contact_geoms[leg]);
  assert.deepEqual(edges, [
    { frame: 1, leg: 1, x: 1 + ankle(1), y: 10 + ankle(1) },
    { frame: 3, leg: 2, x: 3 + ankle(2), y: 10 + ankle(2) },
    { frame: 4, leg: 0, x: 4 + ankle(0), y: 10 + ankle(0) },
  ]);
});

test("contact chatter: a rising edge under 0.15 s after the same foot's previous rising edge leaves no footprint", () => {
  const fps = 20; // 0.05 s a frame: 3 frames = 0.15 s, the first gap that counts as a new step
  assert.equal(CHATTER_S, 0.15);
  const F = false;
  const T = true;
  // Leg 0 contact per frame; legs 1-3 stay down (no edges).
  const leg0 = [F, T, F, T, F, F, T, F, F, F, F, T, F, T];
  //            0  1  2  3  4  5  6  7  8  9 10 11 12 13
  // rises at 1 (kept), 3 (0.10 s after 1: chatter), 6 (0.15 s after 3: kept),
  // 11 (kept), 13 (0.10 s after 11: chatter)
  const frames = leg0.map((c, k) => frame([c, T, T, T], k));
  const edges = footprintEdges({ frames, fps }, manifest);
  assert.deepEqual(
    edges.map((e) => [e.frame, e.leg]),
    [
      [1, 0],
      [6, 0],
      [11, 0],
    ],
  );
  // Chatter is per foot: leg 1 rising right after leg 0 still counts.
  const both = [
    frame([F, F, T, T], 0),
    frame([T, F, T, T], 1),
    frame([T, T, T, T], 2),
  ];
  assert.deepEqual(
    footprintEdges({ frames: both, fps }, manifest).map((e) => [e.frame, e.leg]),
    [
      [1, 0],
      [2, 1],
    ],
  );
  // A chain of chatter stays one footprint: each ignored edge still resets the clock.
  const chain = [F, T, F, T, F, T, F, T].map((c, k) => frame([c, T, T, T], k));
  assert.deepEqual(footprintEdges({ frames: chain, fps }, manifest).map((e) => e.frame), [1]);
});

test("footprints appear on their frame, fade linearly over 2 s, and clear when the loop restarts", () => {
  const fps = 10;
  const edges = [
    { frame: 5, leg: 0, x: 0, y: 0 },
    { frame: 20, leg: 1, x: 1, y: 0 },
  ];
  assert.deepEqual(activeFootprints(edges, fps, 0.49), []); // frame 5 not shown yet
  assert.deepEqual(activeFootprints(edges, fps, 0.5).map((f) => f.alpha), [1]);
  const mid = activeFootprints(edges, fps, 0.5 + FOOTPRINT_FADE_S / 2);
  assert.equal(mid.length, 1);
  assert.ok(Math.abs(mid[0].alpha - 0.5) < 1e-9);
  const later = activeFootprints(edges, fps, 2.2);
  assert.deepEqual(later.map((f) => f.edge.frame), [5, 20]);
  assert.ok(Math.abs(later[0].alpha - 0.15) < 1e-9);
  assert.ok(Math.abs(later[1].alpha - 0.9) < 1e-9);
  assert.deepEqual(activeFootprints(edges, fps, 0.5 + FOOTPRINT_FADE_S).map((f) => f.edge.frame), [20]);
  // Loop time restarts at 0: the trail is empty again.
  assert.deepEqual(activeFootprints(edges, fps, 0), []);
  assert.deepEqual(activeFootprints(edges, 0, 1), []);
  assert.deepEqual(activeFootprints(edges, fps, -1), []);
});

test("the cheat's hot joint: lit at rest, blooms at full recorded force, and stays red", () => {
  assert.equal(hotGlowIntensity(0), HOT_GLOW_MIN);
  assert.equal(hotGlowIntensity(undefined), HOT_GLOW_MIN);
  assert.equal(hotGlowIntensity(Number.NaN), HOT_GLOW_MIN);
  assert.equal(hotGlowIntensity(-1), HOT_GLOW_MAX);
  assert.equal(hotGlowIntensity(3), HOT_GLOW_MAX);
  const [r, g, b] = HOT_EMISSIVE_LINEAR.map((c) => c * HOT_GLOW_MAX);
  // Rec. 709 luminance of the linear emissive clears the scene's 1.4 bloom threshold...
  assert.ok(0.2126 * r + 0.7152 * g + 0.0722 * b > 1.4);
  // ...while green and blue stay far below red, so tone mapping cannot wash it to white.
  assert.ok(g < 0.25 && b < 0.25);
});
