// Pure helpers shared by the API routes and the client. No runtime imports from
// "mongodb" here so this file is safe in the client bundle and in node --test.
import type { Document } from "mongodb";

/** CONTRACTS.md §5: the change stream on db "ada". */
export const WATCH_PIPELINE: Document[] = [
  {
    $match: {
      operationType: { $in: ["insert", "update"] },
      "ns.coll": { $in: ["events", "versions", "edits"] },
    },
  },
];
export const WATCH_OPTIONS = { fullDocument: "updateLookup" } as const;

export const PING_MS = 15_000;
export const SNAPSHOT_VERSIONS = 30;
export const SNAPSHOT_EVENTS = 50;

/** CONTRACTS.md §4, as the client sees it after toJSONSafe. */
export type AdaEvent = {
  _id: string;
  ts: string;
  round_id: string | null;
  version_id: string | null;
  edit_id: string | null;
  stage: string;
  status: "start" | "pass" | "fail" | "info";
  payload: Record<string, unknown>;
};

export type StreamMessage =
  | {
      type: "snapshot";
      frontier: Document | null;
      versions: Document[];
      latest_edit: Document | null;
      latest_rejected_edit: Document | null;
      showcase_frames_ids: string[];
      events: AdaEvent[];
    }
  | { type: "change"; coll: string; op: string; doc: Document }
  | { type: "error"; message: string };

/**
 * Recursively converts BSON values into plain JSON: ObjectId -> 24-char hex,
 * Date -> ISO 8601. Everything else passes through (objects and arrays copied).
 */
export function toJSONSafe(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (value instanceof Date) return value.toISOString();
  const bsontype = (value as { _bsontype?: string })._bsontype;
  if (bsontype === "ObjectId" || bsontype === "ObjectID") {
    return (value as { toHexString(): string }).toHexString();
  }
  if (Array.isArray(value)) return value.map(toJSONSafe);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) out[k] = toJSONSafe(v);
  return out;
}

/** One SSE `data:` frame; always goes through toJSONSafe. */
export function sseData(message: StreamMessage): string {
  return `data: ${JSON.stringify(toJSONSafe(message))}\n\n`;
}

export const SSE_PING = ": ping\n\n";

function tsValue(e: AdaEvent): number {
  const t = Date.parse(e.ts);
  return Number.isNaN(t) ? 0 : t;
}

/** Dedupe by string _id (incoming wins), newest ts first, capped at limit. */
export function mergeEvents(current: AdaEvent[], incoming: AdaEvent[], limit = 500): AdaEvent[] {
  const byId = new Map<string, AdaEvent>();
  for (const e of current) byId.set(e._id, e);
  for (const e of incoming) byId.set(e._id, e);
  return [...byId.values()]
    .sort((a, b) => tsValue(b) - tsValue(a) || (a._id < b._id ? 1 : a._id > b._id ? -1 : 0))
    .slice(0, limit);
}
