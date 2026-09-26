import { db } from "@/lib/mongo";
import { toJSONSafe } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** The newest `swaps` document (NOTES.md [C6]); 404 until Lane B writes one. */
export async function GET() {
  const ada = await db("ada");
  const doc = await ada.collection("swaps").findOne({}, { sort: { captured_at: -1 } });
  if (!doc) return Response.json({ error: "no swaps document yet" }, { status: 404 });
  return Response.json(toJSONSafe(doc));
}
