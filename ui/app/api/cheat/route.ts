import { ObjectId } from "mongodb";
import { acceptedOverRated, agentQuote, type CheatCard, type CheatDoc } from "@/lib/cheatcard";
import { db } from "@/lib/mongo";
import { toJSONSafe } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const OVERRATED_REASON = { $regex: "rated torque|rated motors", $options: "i" };
const NOT_CLI = { origin: { $ne: "cli" } };

/** A string id, plus its ObjectId when it looks like one (frames and runs use either). */
const idsOf = (id: string): (string | ObjectId)[] => (/^[0-9a-f]{24}$/i.test(id) ? [id, new ObjectId(id)] : [id]);

/**
 * The cheat card and the spine additions (NOTES.md [C8]). Read-only.
 * card: the newest rejected non-cli edit whose reason names the rated torque or motors and that
 * has recorded frames; its run via frames.run_id -> runs; the agent's words via runs.trace_id ->
 * traces.raw_steps. counter: the newest scoreboard's overrated_attempts, else a count of those
 * edits; accepted is counted from runs' peak_torque. spine: agent model ids per version from
 * runs, the newest scoreboard's rewrite cost, and every non-cli edit's verdict.
 */
export async function GET() {
  const ada = await db("ada");
  const [edit, scoreboard, edits, modelRows] = await Promise.all([
    ada.collection("edits").findOne(
      { verdict: "rejected", ...NOT_CLI, reason: OVERRATED_REASON, frames_id: { $type: "string", $ne: "" } },
      { sort: { created_at: -1 } },
    ),
    ada.collection("scoreboard").findOne({}, { sort: { created_at: -1 } }),
    ada
      .collection<{ _id: string; verdict?: string | null; origin?: string | null; to_version?: string | null }>("edits")
      .find(NOT_CLI, { projection: { verdict: 1, origin: 1, to_version: 1 } })
      .toArray(),
    ada.collection("runs").aggregate([{ $group: { _id: "$version_id", models: { $addToSet: "$model_id" } } }]).toArray(),
  ]);

  let card: CheatCard | null = null;
  if (edit) {
    const frames = await ada.collection("frames").findOne({ _id: { $in: idsOf(String(edit.frames_id)) } as never }, { projection: { run_id: 1 } });
    const run =
      typeof frames?.run_id === "string"
        ? await ada.collection("runs").findOne({ _id: { $in: idsOf(frames.run_id) } as never }, { projection: { trace_id: 1, peak_torque: 1, model_id: 1 } })
        : null;
    const trace =
      typeof run?.trace_id === "string"
        ? await ada.collection("traces").findOne({ trace_id: run.trace_id }, { projection: { raw_steps: 1 } })
        : null;
    card = {
      edit_id: String(edit._id),
      rationale: typeof edit.rationale === "string" ? edit.rationale : "",
      reason: String(edit.reason),
      frames_id: String(edit.frames_id),
      quote: agentQuote(trace?.raw_steps),
      peak_torque: (run?.peak_torque as Record<string, number> | undefined) ?? {},
      model_id: typeof run?.model_id === "string" ? run.model_id : null,
    };
  }

  const accepted = edits.filter((e) => e.verdict === "accepted" && typeof e.to_version === "string");
  const hotRuns = accepted.length
    ? await ada
        .collection<{ version_id?: string; peak_torque?: Record<string, number> }>("runs")
        .find({ version_id: { $in: accepted.map((e) => e.to_version as string) } }, { projection: { version_id: 1, peak_torque: 1 } })
        .toArray()
    : [];
  const overrated =
    typeof scoreboard?.overrated_attempts === "number"
      ? scoreboard.overrated_attempts
      : await ada.collection("edits").countDocuments({ verdict: "rejected", ...NOT_CLI, reason: OVERRATED_REASON });

  const models: Record<string, string[]> = {};
  for (const r of modelRows) {
    if (typeof r._id === "string") models[r._id] = (r.models as unknown[]).filter((m): m is string => typeof m === "string").sort();
  }

  const doc: CheatDoc = {
    card,
    counter: {
      overrated,
      accepted: acceptedOverRated(edits, hotRuns),
      source: typeof scoreboard?.overrated_attempts === "number" ? "scoreboard" : "count",
    },
    spine: {
      models,
      rewrite_cost_usd: typeof scoreboard?.rewrite_cost_usd === "number" ? scoreboard.rewrite_cost_usd : null,
      edits: edits.map((e) => ({ _id: String(e._id), verdict: e.verdict ?? null, origin: e.origin ?? null })),
    },
  };
  return Response.json(toJSONSafe(doc));
}
