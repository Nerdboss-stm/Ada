import { test } from "node:test";
import assert from "node:assert/strict";
import {
  cycleSeconds,
  followStep,
  ghostColor,
  ghostSources,
  mergeVersions,
  mjToThree,
  parseFixtures,
  parseFrameIds,
  pickFixtureLeader,
  pickLeader,
  sharedFrameIndex,
  sortVersions,
  type VersionDoc,
} from "../lib/ghosts.ts";
import type { FramesDoc } from "../lib/replay.ts";

test("ghost colors run from #3a6bff (oldest) to #e8eefc (newest)", () => {
  assert.equal(ghostColor(0, 3), "#3a6bff");
  assert.equal(ghostColor(2, 3), "#e8eefc");
  assert.equal(ghostColor(1, 3), "#91adfe");
  assert.equal(ghostColor(0, 1), "#e8eefc");
});

test("versions sort numerically and merge by _id with incoming winning", () => {
  const vs: VersionDoc[] = [{ _id: "v10" }, { _id: "v2" }, { _id: "v0" }];
  assert.deepEqual(sortVersions(vs).map((v) => v._id), ["v0", "v2", "v10"]);
  const merged = mergeVersions(vs, [{ _id: "v2", status: "accepted" }, { _id: "v3" }]);
  assert.deepEqual(merged.map((v) => v._id), ["v0", "v2", "v3", "v10"]);
  assert.equal(merged[1].status, "accepted");
});

test("ghost sources are versions with showcase_frames_id, oldest first", () => {
  const vs: VersionDoc[] = [
    { _id: "v2", showcase_frames_id: "f2" },
    { _id: "v1", showcase_frames_id: null },
    { _id: "v0", showcase_frames_id: "f0" },
  ];
  assert.deepEqual(ghostSources(vs), [
    { versionId: "v0", framesId: "f0" },
    { versionId: "v2", framesId: "f2" },
  ]);
});

test("leader is train-reliability-first among accepted or baseline, distance breaks ties", () => {
  const m = (r: number | null, d: number) => ({ train_reliability: r, mean_distance_m: d });
  const vs: VersionDoc[] = [
    { _id: "v0", status: "baseline", metrics: m(0.4, 1.2) },
    { _id: "v1", status: "rejected", metrics: m(1.0, 9.9) },
    { _id: "v2", status: "frontier", metrics: m(1.0, 8.0) },
    { _id: "v3", status: "accepted", metrics: m(0.8, 3.4) },
    { _id: "v4", status: "accepted", metrics: null },
    { _id: "v5", status: "accepted", metrics: m(0.6, 20) }, // far but unreliable: never the leader
  ];
  assert.equal(pickLeader(vs), "v3");
  // Tie on reliability: farther wins; full tie: oldest keeps it.
  assert.equal(pickLeader([...vs, { _id: "v6", status: "accepted", metrics: m(0.8, 3.5) }]), "v6");
  assert.equal(pickLeader([...vs, { _id: "v7", status: "accepted", metrics: m(0.8, 3.4) }]), "v3");
  // No train reliability: not eligible.
  assert.equal(pickLeader([{ _id: "v8", status: "accepted", metrics: m(null, 30) }]), null);
  assert.equal(pickLeader([{ _id: "v1", status: "rejected", metrics: m(1, 5) }]), null);
});

test("leader never looks at holdout: holdout is evidence only", () => {
  const vs: VersionDoc[] = [
    { _id: "v1", status: "accepted", metrics: { train_reliability: 0.9, holdout_reliability_80: 0.1, mean_distance_m: 2 } },
    { _id: "v2", status: "accepted", metrics: { train_reliability: 0.5, holdout_reliability_80: 1.0, mean_distance_m: 2 } },
  ];
  assert.equal(pickLeader(vs), "v1");
  // Holdout alone does not make a version eligible.
  assert.equal(pickLeader([{ _id: "v3", status: "accepted", metrics: { holdout_reliability_80: 1, mean_distance_m: 9 } }]), null);
});

const doc = (xs: number[], fps = 10, kind: FramesDoc["kind"] = "showcase"): FramesDoc => ({
  _id: "f",
  run_id: "r",
  version_id: "v",
  kind,
  manifest_version: "m",
  fps,
  sha256: "",
  frames: xs.map((x) => ({ geoms: [], contacts: [], forces: [], torso: [x, 0, 0.5] })),
});

test("fixture leader is the largest recorded final torso x", () => {
  assert.equal(
    pickFixtureLeader([
      { key: "a", doc: doc([0, 1]) },
      { key: "b", doc: doc([0, 3]) },
      { key: "c", doc: doc([0, -2]) },
    ]),
    "b",
  );
  assert.equal(pickFixtureLeader([]), null);
});

test("a run recorded as rejected never leads, however far it went", () => {
  assert.equal(
    pickFixtureLeader([
      { key: "trot", doc: doc([0, 3.01]) },
      { key: "flail", doc: doc([0, 40], 10, "rejected") },
    ]),
    "trot",
  );
  assert.equal(pickFixtureLeader([{ key: "flail", doc: doc([0, 40], 10, "rejected") }]), null);
});

test("frames ids map to /api/frames/[id] and are sanitized", () => {
  assert.deepEqual(parseFrameIds("6b8e6ec9ad5fffd33ba308d1, ../x,6b8e6ec9ad5fffd33ba308d1,a?b=1,fixture-wiggle"), [
    { name: "6b8e6ec9ad5fffd33ba308d1", url: "/api/frames/6b8e6ec9ad5fffd33ba308d1" },
    { name: "fixture-wiggle", url: "/api/frames/fixture-wiggle" },
  ]);
  assert.deepEqual(parseFrameIds(null), []);
});

test("fixture names are sanitized against traversal", () => {
  assert.deepEqual(parseFixtures("wiggle, Sway,../etc/passwd,sway,a/b,,x%2e"), [
    { name: "wiggle", url: "/fixtures/wiggle.json" },
    { name: "sway", url: "/fixtures/sway.json" },
  ]);
  assert.deepEqual(parseFixtures(null), []);
});

test("shared clock: all start at 0, short runs hold last frame, all loop on the longest", () => {
  const cycle = cycleSeconds([doc(new Array(10).fill(0)), doc(new Array(5).fill(0))]);
  assert.equal(cycle, 1);
  assert.equal(sharedFrameIndex(0, 10, 5, cycle), 0);
  assert.equal(sharedFrameIndex(0.35, 10, 5, cycle), 3);
  assert.equal(sharedFrameIndex(0.75, 10, 5, cycle), 4);
  assert.equal(sharedFrameIndex(0.95, 10, 10, cycle), 9);
  assert.equal(sharedFrameIndex(1.05, 10, 10, cycle), 0);
  assert.equal(sharedFrameIndex(1.05, 10, 5, cycle), 0);
  assert.equal(sharedFrameIndex(-1, 10, 5, cycle), 0);
  assert.equal(sharedFrameIndex(Number.NaN, 10, 5, cycle), 0);
});

test("mjToThree maps z-up to y-up once", () => {
  assert.deepEqual(mjToThree([1, 2, 3]), [1, 3, -2]);
});

test("followStep lerps 0.05 per 60 Hz frame, independent of frame rate", () => {
  const one = followStep([0, 0, 0], [1, 0, 0], 0.05, 1 / 60);
  assert.ok(Math.abs(one[0] - 0.05) < 1e-12);
  const twoHalf = followStep(followStep([0, 0, 0], [1, 0, 0], 0.05, 1 / 120), [1, 0, 0], 0.05, 1 / 120);
  assert.ok(Math.abs(twoHalf[0] - 0.05) < 1e-12);
  assert.deepEqual(followStep([2, 2, 2], [4, 4, 4], 0.05, 0), [2, 2, 2]);
});
