"use client";

import { useEffect, useState } from "react";
import { mergeEvents, type AdaEvent, type StreamMessage } from "@/lib/stream";

type Conn = "connecting" | "live" | "error";

export default function EventList() {
  const [events, setEvents] = useState<AdaEvent[]>([]);
  const [conn, setConn] = useState<Conn>("connecting");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const source = new EventSource("/api/stream");
    source.onopen = () => setConn("live");
    source.onerror = () => setConn("connecting"); // EventSource reconnects; server resends the snapshot
    source.onmessage = (msg) => {
      const data = JSON.parse(msg.data) as StreamMessage;
      if (data.type === "snapshot") {
        setError(null);
        setEvents((prev) => mergeEvents(prev, data.events));
      } else if (data.type === "change" && data.coll === "events") {
        setEvents((prev) => mergeEvents(prev, [data.doc as AdaEvent]));
      } else if (data.type === "error") {
        setConn("error");
        setError(data.message);
      }
    };
    return () => source.close();
  }, []);

  return (
    <section>
      <header className="mb-4 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold">Ada · raw events</h1>
        <span className="text-sm opacity-70">
          {conn} · {events.length}
        </span>
      </header>
      {error && <p className="mb-4 text-sm text-red-600">{error}</p>}
      <ol className="flex flex-col gap-2">
        {events.map((e) => (
          <li key={e._id} data-id={e._id} className="rounded border border-black/10 p-3 text-sm dark:border-white/15">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span className="tabular-nums">{e.ts}</span>
              <span className="font-medium">{e.stage}</span>
              <span>{e.status}</span>
              <span className="opacity-60">round {e.round_id ?? "–"}</span>
              <span className="opacity-60">version {e.version_id ?? "–"}</span>
              <span className="opacity-60">edit {e.edit_id ?? "–"}</span>
              <span className="opacity-40">{e._id}</span>
            </div>
            <pre className="mt-2 whitespace-pre-wrap break-words opacity-80">{JSON.stringify(e.payload, null, 2)}</pre>
          </li>
        ))}
      </ol>
    </section>
  );
}
