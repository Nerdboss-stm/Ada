import type { ChangeStream } from "mongodb";
import { db } from "@/lib/mongo";
import { buildSnapshot } from "@/lib/snapshot";
import { PING_MS, SSE_PING, WATCH_OPTIONS, WATCH_PIPELINE, sseData } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function GET(req: Request) {
  const encoder = new TextEncoder();
  let cleanup = () => {};

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let closed = false;
      let changes: ChangeStream | undefined;
      let ping: ReturnType<typeof setInterval> | undefined;
      const send = (chunk: string) => {
        if (!closed) controller.enqueue(encoder.encode(chunk));
      };
      cleanup = () => {
        if (closed) return;
        closed = true;
        clearInterval(ping);
        changes?.close().catch(() => {});
        try {
          controller.close();
        } catch {}
      };
      req.signal.addEventListener("abort", cleanup);

      try {
        const ada = await db("ada");
        // Open the watch before reading the snapshot so nothing falls between them;
        // duplicates are fine because the client dedupes by _id.
        changes = ada.watch(WATCH_PIPELINE, WATCH_OPTIONS);
        changes.on("change", (change) => {
          if (!("fullDocument" in change) || !change.fullDocument) return;
          send(sseData({ type: "change", coll: change.ns.coll, op: change.operationType, doc: change.fullDocument }));
        });
        changes.on("error", (err) => {
          send(sseData({ type: "error", message: err.message }));
          cleanup();
        });
        send(sseData(await buildSnapshot(ada)));
        ping = setInterval(() => send(SSE_PING), PING_MS);
      } catch (err) {
        send(sseData({ type: "error", message: (err as Error).message }));
        cleanup();
      }
    },
    cancel() {
      cleanup();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
