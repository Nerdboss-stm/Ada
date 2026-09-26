import type { Db } from "mongodb";
import { SNAPSHOT_EVENTS, SNAPSHOT_VERSIONS, type AdaEvent, type StreamMessage } from "./stream";

/** CONTRACTS.md §5 snapshot, plus the latest raw events for the event list. */
export async function buildSnapshot(ada: Db): Promise<Extract<StreamMessage, { type: "snapshot" }>> {
  const [frontier, versions, latestEdit, showcase, events] = await Promise.all([
    ada.collection("versions").findOne({ status: "frontier" }, { sort: { created_at: -1 } }),
    ada.collection("versions").find().sort({ created_at: -1 }).limit(SNAPSHOT_VERSIONS).toArray(),
    ada.collection("edits").findOne({}, { sort: { created_at: -1 } }),
    ada.collection("frames").find({ kind: "showcase" }, { projection: { _id: 1 } }).toArray(),
    ada.collection("events").find().sort({ ts: -1 }).limit(SNAPSHOT_EVENTS).toArray(),
  ]);
  return {
    type: "snapshot",
    frontier,
    versions,
    latest_edit: latestEdit,
    showcase_frames_ids: showcase.map((f) => String(f._id)),
    events: events as unknown as AdaEvent[],
  };
}
