import { test } from "node:test";
import assert from "node:assert/strict";
import { ObjectId } from "mongodb";
import {
  WATCH_OPTIONS,
  WATCH_PIPELINE,
  mergeEvents,
  sseData,
  toJSONSafe,
  type AdaEvent,
} from "../lib/stream.ts";

test("watch pipeline and options match CONTRACTS.md §5", () => {
  assert.deepEqual(WATCH_PIPELINE, [
    {
      $match: {
        operationType: { $in: ["insert", "update"] },
        "ns.coll": { $in: ["events", "versions", "edits"] },
      },
    },
  ]);
  assert.deepEqual(WATCH_OPTIONS, { fullDocument: "updateLookup" });
});

test("toJSONSafe turns ObjectId into hex and Date into ISO 8601, recursively", () => {
  const oid = new ObjectId("65f000000000000000000abc");
  const when = new Date("2026-09-26T11:05:00.000Z");
  const out = toJSONSafe({
    _id: oid,
    created_at: when,
    parent: null,
    n: 3,
    nested: { ids: [oid, "v1"], at: [when] },
  });
  assert.deepEqual(out, {
    _id: "65f000000000000000000abc",
    created_at: "2026-09-26T11:05:00.000Z",
    parent: null,
    n: 3,
    nested: { ids: ["65f000000000000000000abc", "v1"], at: ["2026-09-26T11:05:00.000Z"] },
  });
  const frame = JSON.parse(sseData({ type: "change", coll: "events", op: "insert", doc: { _id: oid, ts: when } }).slice(6));
  assert.equal(typeof frame.doc._id, "string");
  assert.equal(frame.doc._id, oid.toHexString());
  assert.equal(frame.doc.ts, when.toISOString());
});

test("sseData frames one data line terminated by a blank line", () => {
  const s = sseData({ type: "error", message: "x" });
  assert.match(s, /^data: \{.*\}\n\n$/);
});

const ev = (id: string, ts: string, stage = "harness"): AdaEvent => ({
  _id: id,
  ts,
  round_id: null,
  version_id: null,
  edit_id: null,
  stage,
  status: "info",
  payload: {},
});

test("mergeEvents dedupes by _id and orders newest first", () => {
  const a = ev("a", "2026-09-26T11:00:00Z");
  const b = ev("b", "2026-09-26T11:02:00Z");
  const c = ev("c", "2026-09-26T11:01:00Z");
  const merged = mergeEvents([a, b], [c, ev("a", "2026-09-26T11:00:00Z", "baseline")]);
  assert.deepEqual(merged.map((e) => e._id), ["b", "c", "a"]);
  assert.equal(merged.find((e) => e._id === "a")?.stage, "baseline");
  assert.equal(mergeEvents([a, b, c], [], 2).length, 2);
});
