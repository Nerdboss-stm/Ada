// Server-only reads shared by api/cheat and api/snapshot/latest (NOTES.md [C8], [C9]).
// Read-only: nothing here writes a document.
import { ObjectId, type Db, type Document } from "mongodb";
import { agentQuote, cliOnlyVersions, gaitCounter, type CheatCard, type CheatCounter } from "./cheatcard.ts";

export const OVERRATED_REASON = { $regex: "rated torque|rated motors", $options: "i" };
export const NOT_CLI = { origin: { $ne: "cli" } };

/** A string id, plus its ObjectId when it looks like one (frames and runs use either). */
export const idsOf = (ids: string[]): (string | ObjectId)[] =>
  ids.flatMap((id) => (/^[0-9a-f]{24}$/i.test(id) ? [id, new ObjectId(id)] : [id]));

/**
 * The cheat card for one rejected edit: its rationale and reason word for word, its run via
 * frames.run_id -> runs, and the agent's words via runs.trace_id -> traces.raw_steps.
 */
export async function buildCard(ada: Db, edit: Document): Promise<CheatCard> {
  const frames = await ada.collection("frames").findOne({ _id: { $in: idsOf([String(edit.frames_id)]) } as never }, { projection: { run_id: 1 } });
  const run =
    typeof frames?.run_id === "string"
      ? await ada.collection("runs").findOne({ _id: { $in: idsOf([frames.run_id]) } as never }, { projection: { trace_id: 1, peak_torque: 1, model_id: 1 } })
      : null;
  const trace =
    typeof run?.trace_id === "string" ? await ada.collection("traces").findOne({ trace_id: run.trace_id }, { projection: { raw_steps: 1 } }) : null;
  return {
    edit_id: String(edit._id),
    rationale: typeof edit.rationale === "string" ? edit.rationale : "",
    reason: String(edit.reason),
    frames_id: String(edit.frames_id),
    quote: agentQuote(trace?.raw_steps),
    peak_torque: (run?.peak_torque as Record<string, number> | undefined) ?? {},
    model_id: typeof run?.model_id === "string" ? run.model_id : null,
  };
}

/** Every non-cli run in `ada`: over-rated gaits and how many of them passed the physics check. */
export async function countGaits(ada: Db): Promise<CheatCounter> {
  const [runs, edits] = await Promise.all([
    ada
      .collection<{ version_id?: string; peak_torque?: Record<string, number>; sanity?: { pass?: boolean } }>("runs")
      .find({}, { projection: { version_id: 1, peak_torque: 1, "sanity.pass": 1 } })
      .toArray(),
    ada
      .collection<{ origin?: string; to_version?: string }>("edits")
      .find({ to_version: { $type: "string" } }, { projection: { origin: 1, to_version: 1 } })
      .toArray(),
  ]);
  return gaitCounter(runs, cliOnlyVersions(edits));
}
