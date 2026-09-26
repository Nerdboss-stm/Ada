import { ObjectId } from "mongodb";
import { db } from "@/lib/mongo";
import { toJSONSafe } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const EDIT_FIELDS = {
  from_version: 1,
  to_version: 1,
  primitive: 1,
  rationale: 1,
  predicted_delta_m: 1,
  actual_delta_m: 1,
  attempt_frames_id: 1,
  verdict: 1,
  created_at: 1,
};

/** A string id, plus its ObjectId when it looks like one (frames and runs use either). */
const idsOf = (ids: string[]): (string | ObjectId)[] =>
  ids.flatMap((id) => (/^[0-9a-f]{24}$/i.test(id) ? [id, new ObjectId(id)] : [id]));

/**
 * ?mode=attempts (NOTES.md [C7]): every edit (the tray counts all rejected ones), the versions
 * the attempt edits name (their train means), and each attempt walk's terrain via
 * frames.run_id -> runs.task_id -> tasks. Read-only.
 */
export async function GET() {
  const ada = await db("ada");
  const edits = await ada.collection("edits").find({}, { projection: EDIT_FIELDS }).sort({ created_at: 1 }).toArray();
  const attempts = edits.filter((e) => typeof e.attempt_frames_id === "string" && typeof e.predicted_delta_m === "number");

  const versionIds = [...new Set(attempts.flatMap((e) => [e.from_version, e.to_version]).filter((v): v is string => typeof v === "string"))];
  const versions = await ada
    .collection("versions")
    .find({ _id: { $in: versionIds } as never }, { projection: { status: 1, metrics: 1 } })
    .toArray();

  const framesIds = [...new Set(attempts.map((e) => e.attempt_frames_id as string))];
  const frames = await ada.collection("frames").find({ _id: { $in: idsOf(framesIds) } as never }, { projection: { run_id: 1 } }).toArray();
  const runIds = [...new Set(frames.map((f) => f.run_id).filter((r): r is string => typeof r === "string"))];
  const runs = await ada.collection("runs").find({ _id: { $in: idsOf(runIds) } as never }, { projection: { task_id: 1 } }).toArray();
  const taskIds = [...new Set(runs.map((r) => r.task_id).filter((t): t is string => typeof t === "string"))];
  const tasks = await ada.collection("tasks").find({ _id: { $in: taskIds } as never }, { projection: { slope_deg: 1, friction: 1 } }).toArray();

  const taskOfRun = new Map(runs.map((r) => [String(r._id), r.task_id as string]));
  const taskById = new Map(tasks.map((t) => [String(t._id), t]));
  const terrains: Record<string, unknown> = {};
  for (const id of framesIds) {
    const f = frames.find((x) => String(x._id) === id);
    const taskId = f ? taskOfRun.get(String(f.run_id)) : undefined;
    const task = taskId ? taskById.get(taskId) : undefined;
    terrains[id] = taskId ? { task_id: taskId, slope_deg: task?.slope_deg ?? null, friction: task?.friction ?? null } : null;
  }

  return Response.json(toJSONSafe({ versions, edits, terrains }));
}
