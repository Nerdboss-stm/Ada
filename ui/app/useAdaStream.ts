"use client";

import { useEffect, useState } from "react";
import { mergeVersions, type VersionDoc } from "@/lib/ghosts";
import { mergeEdits, type EditDoc } from "@/lib/overlays";
import { mergeEvents, type AdaEvent, type StreamMessage } from "@/lib/stream";

export type AdaStream = {
  versions: VersionDoc[];
  edits: EditDoc[];
  events: AdaEvent[];
  error: string | null;
};

/** The page's one /api/stream connection: snapshot plus change events, deduped by _id. */
export function useAdaStream(): AdaStream {
  const [state, setState] = useState<AdaStream>({ versions: [], edits: [], events: [], error: null });

  useEffect(() => {
    const source = new EventSource("/api/stream");
    source.onmessage = (msg) => {
      const data = JSON.parse(msg.data) as StreamMessage;
      if (data.type === "snapshot") {
        const versions = [...(data.versions as VersionDoc[]), ...(data.frontier ? [data.frontier as VersionDoc] : [])];
        setState((s) => ({
          versions: mergeVersions(s.versions, versions),
          // The latest rejected edit rides along so the cheat survives a reload; the card still
          // shows the newest edit (latestEdit), which is never older than it.
          edits: mergeEdits(
            s.edits,
            [data.latest_edit, data.latest_rejected_edit].filter((e) => e !== null && e !== undefined) as EditDoc[],
          ),
          events: mergeEvents(s.events, data.events),
          error: null,
        }));
      } else if (data.type === "change") {
        setState((s) => {
          if (data.coll === "versions") return { ...s, versions: mergeVersions(s.versions, [data.doc as VersionDoc]) };
          if (data.coll === "edits") return { ...s, edits: mergeEdits(s.edits, [data.doc as EditDoc]) };
          if (data.coll === "events") return { ...s, events: mergeEvents(s.events, [data.doc as AdaEvent]) };
          return s;
        });
      } else if (data.type === "error") {
        setState((s) => ({ ...s, error: data.message }));
      }
    };
    return () => source.close();
  }, []);

  return state;
}
