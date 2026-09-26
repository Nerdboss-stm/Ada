import type { CheatCard, CheatDoc } from "@/lib/cheatcard";
import { NOT_CLI, OVERRATED_REASON, buildCard, countGaits } from "@/lib/cheatserver";
import { db } from "@/lib/mongo";
import { toJSONSafe } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * The cheat card and the spine additions (NOTES.md [C8], [C9]). Read-only.
 * card: the newest rejected non-cli edit whose reason names the rated torque or motors and that
 * has recorded frames (lib/cheatserver.ts buildCard). counter: every non-cli run with a
 * peak_torque ratio above 1.0, and how many of those passed the physics check. spine: agent
 * model ids per version from runs, the newest scoreboard's rewrite cost, and every non-cli
 * edit's verdict.
 */
export async function GET() {
  const ada = await db("ada");
  const [edit, scoreboard, edits, modelRows, counter] = await Promise.all([
    ada.collection("edits").findOne(
      { verdict: "rejected", ...NOT_CLI, reason: OVERRATED_REASON, frames_id: { $type: "string", $ne: "" } },
      { sort: { created_at: -1 } },
    ),
    ada.collection("scoreboard").findOne({}, { sort: { created_at: -1 } }),
    ada
      .collection<{ _id: string; verdict?: string | null; origin?: string | null }>("edits")
      .find(NOT_CLI, { projection: { verdict: 1, origin: 1 } })
      .toArray(),
    ada.collection("runs").aggregate([{ $group: { _id: "$version_id", models: { $addToSet: "$model_id" } } }]).toArray(),
    countGaits(ada),
  ]);

  const card: CheatCard | null = edit ? await buildCard(ada, edit) : null;

  const models: Record<string, string[]> = {};
  for (const r of modelRows) {
    if (typeof r._id === "string") models[r._id] = (r.models as unknown[]).filter((m): m is string => typeof m === "string").sort();
  }

  const doc: CheatDoc = {
    card,
    counter,
    spine: {
      models,
      rewrite_cost_usd: typeof scoreboard?.rewrite_cost_usd === "number" ? scoreboard.rewrite_cost_usd : null,
      edits: edits.map((e) => ({ _id: String(e._id), verdict: e.verdict ?? null, origin: e.origin ?? null })),
    },
  };
  return Response.json(toJSONSafe(doc));
}
