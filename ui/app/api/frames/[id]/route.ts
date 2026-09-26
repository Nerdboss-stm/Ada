import { ObjectId } from "mongodb";
import { db } from "@/lib/mongo";
import { toJSONSafe } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_req: Request, ctx: RouteContext<"/api/frames/[id]">) {
  const { id } = await ctx.params;
  const ids: (string | ObjectId)[] = [id];
  if (/^[0-9a-f]{24}$/i.test(id)) ids.push(new ObjectId(id));
  const ada = await db("ada");
  const doc = await ada.collection("frames").findOne({ _id: { $in: ids } as never });
  if (!doc) return Response.json({ error: `frames ${id} not found` }, { status: 404 });
  return Response.json(toJSONSafe(doc));
}
