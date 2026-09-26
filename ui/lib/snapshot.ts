import type { Db } from "mongodb";
import { SNAPSHOT_EVENTS, SNAPSHOT_VERSIONS, type AdaEvent, type StreamMessage } from "./stream.ts";

/** The newest rejected edit with recorded frames: a reload never loses the cheat. */
export const REJECTED_WITH_FRAMES = { verdict: "rejected", frames_id: { $type: "string", $ne: "" } } as const;

/**
 * CONTRACTS.md §5 snapshot, plus the latest raw events for the event list and the latest
 * rejected edit that has a frames_id (NOTES.md [C6]).
 */
export async function buildSnapshot(ada: Db): Promise<Extract<StreamMessage, { type: "snapshot" }>> {
  const [frontier, versions, latestEdit, latestRejected, showcase, events] = await Promise.all([
    ada.collection("versions").findOne({ status: "frontier" }, { sort: { created_at: -1 } }),
    ada.collection("versions").find().sort({ created_at: -1 }).limit(SNAPSHOT_VERSIONS).toArray(),
    ada.collection("edits").findOne({}, { sort: { created_at: -1 } }),
    ada.collection("edits").findOne(REJECTED_WITH_FRAMES, { sort: { created_at: -1 } }),
    ada.collection("frames").find({ kind: "showcase" }, { projection: { _id: 1 } }).toArray(),
    ada.collection("events").find().sort({ ts: -1 }).limit(SNAPSHOT_EVENTS).toArray(),
  ]);
  return {
    type: "snapshot",
    frontier,
    versions,
    latest_edit: latestEdit,
    latest_rejected_edit: latestRejected,
    showcase_frames_ids: showcase.map((f) => String(f._id)),
    events: events as unknown as AdaEvent[],
  };
}
