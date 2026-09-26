import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  frameIndexAt,
  geomShape,
  mjQuatToXYZW,
  poseParts,
  type FramesDoc,
  type Manifest,
} from "../lib/replay.ts";

const readJSON = <T>(rel: string): T => JSON.parse(readFileSync(new URL(rel, import.meta.url), "utf8")) as T;

test("frameIndexAt steps at fps with no interpolation and loops", () => {
  const fps = 33.333;
  assert.equal(frameIndexAt(0, fps, 10), 0);
  assert.equal(frameIndexAt(0.029, fps, 10), 0);
  assert.equal(frameIndexAt(0.031, fps, 10), 1);
  assert.equal(frameIndexAt(0.3, fps, 10), 9);
  assert.equal(frameIndexAt(0.31, fps, 10), 0);
  assert.equal(frameIndexAt(-1, fps, 10), 0);
  assert.equal(frameIndexAt(1, fps, 0), 0);
  assert.equal(frameIndexAt(Number.NaN, fps, 10), 0);
});

test("MuJoCo w-first quat becomes three.js x,y,z,w", () => {
  assert.deepEqual(mjQuatToXYZW(1, 0, 0, 0), [0, 0, 0, 1]);
  assert.deepEqual(poseParts([1, 2, 3, 0.9, 0.1, 0.2, 0.3]), { pos: [1, 2, 3], quat: [0.1, 0.2, 0.3, 0.9] });
});

test("geomShape maps sphere radius and capsule half-length to cylinder length", () => {
  assert.deepEqual(geomShape({ id: 1, name: "t", type: "sphere", size: [0.25, 0, 0], body: "torso" }), {
    kind: "sphere",
    radius: 0.25,
  });
  assert.deepEqual(geomShape({ id: 2, name: "c", type: "capsule", size: [0.08, 0.141421, 0], body: "b" }), {
    kind: "capsule",
    radius: 0.08,
    length: 0.282842,
  });
});

test("wiggle fixture matches the manifest and the frames contract", () => {
  const manifest = readJSON<Manifest>("../public/manifest.json");
  const doc = readJSON<FramesDoc>("../public/fixtures/wiggle.json");
  assert.equal(doc.manifest_version, manifest.manifest_version);
  assert.ok(Math.abs(doc.fps - 100 / 3) < 0.01);
  assert.ok(doc.frames.length > 100);
  assert.match(doc.sha256, /^[0-9a-f]{64}$/);
  for (const f of doc.frames) {
    assert.equal(f.geoms.length, manifest.geoms.length);
    for (const g of f.geoms) assert.equal(g.length, 7);
    assert.equal(f.contacts.length, 4);
    assert.equal(f.forces.length, 8);
    assert.equal(f.torso.length, 3);
  }
});
