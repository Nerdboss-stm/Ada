import { test } from "node:test";
import assert from "node:assert/strict";
import type { Db } from "mongodb";
import { REJECTED_WITH_FRAMES, buildSnapshot } from "../lib/snapshot.ts";
import { latestRejectedEdit } from "../lib/cheat.ts";
import { latestEdit, mergeEdits, type EditDoc } from "../lib/overlays.ts";

type Call = { coll: string; op: string; filter: unknown; opts: unknown };

// Mock Db: records every query, answers findOne from a table keyed by collection + filter.
function fakeDb(answers: Record<string, unknown>): { db: Db; calls: Call[] } {
  const calls: Call[] = [];
  const db = {
    collection: (coll: string) => ({
      findOne: async (filter: unknown, opts: unknown) => {
        calls.push({ coll, op: "findOne", filter, opts });
        return answers[`${coll}:${JSON.stringify(filter)}`] ?? null;
      },
      find: (filter: unknown, opts: unknown) => {
        calls.push({ coll, op: "find", filter, opts });
        const cursor = { sort: () => cursor, limit: () => cursor, toArray: async () => [] };
        return cursor;
      },
    }),
  } as unknown as Db;
  return { db, calls };
}

const rejected: EditDoc = { _id: "e7", verdict: "rejected", frames_id: "ffc64b4bfe41b185e9157074", reason: "exceeds Ada rated motors", created_at: "2026-09-26T12:00:00Z" };
const newer: EditDoc = { _id: "e9", verdict: "accepted", frames_id: null, created_at: "2026-09-26T12:05:00Z" };

test("snapshot sends the latest rejected edit that has a frames_id", async () => {
  const { db, calls } = fakeDb({
    [`edits:${JSON.stringify({})}`]: newer,
    [`edits:${JSON.stringify(REJECTED_WITH_FRAMES)}`]: rejected,
  });
  const snap = await buildSnapshot(db);
  assert.equal((snap.latest_edit as EditDoc)._id, "e9");
  assert.equal((snap.latest_rejected_edit as EditDoc)._id, "e7");
  const q = calls.find((c) => c.coll === "edits" && JSON.stringify(c.filter) === JSON.stringify(REJECTED_WITH_FRAMES));
  assert.deepEqual(q?.opts, { sort: { created_at: -1 } });
  assert.deepEqual(REJECTED_WITH_FRAMES, { verdict: "rejected", frames_id: { $type: "string", $ne: "" } });
});

test("snapshot sends null when no rejected edit has frames", async () => {
  const { db } = fakeDb({ [`edits:${JSON.stringify({})}`]: newer });
  const snap = await buildSnapshot(db);
  assert.equal(snap.latest_rejected_edit, null);
});

test("after a reload the client still finds the cheat, and the card still shows the newest edit", () => {
  const edits = mergeEdits([], [newer, rejected]);
  assert.equal(latestRejectedEdit(edits)?._id, "e7");
  assert.equal(latestEdit(edits)?._id, "e9");
});
