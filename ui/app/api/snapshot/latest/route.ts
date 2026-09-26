import { ObjectId, type Document } from "mongodb";
import type { CheatCard } from "@/lib/cheatcard";
import { buildCard, idsOf } from "@/lib/cheatserver";
import { db } from "@/lib/mongo";
import { toJSONSafe } from "@/lib/stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VERSION_FIELDS = { status: 1, metrics: 1, showcase_frames_id: 1, created_at: 1 };

/** When a run was written: its ObjectId's timestamp (runs get Mongo's default _id). */
function runTime(id: unknown): number | null {
  if (id instanceof ObjectId) return id.getTimestamp().getTime();
  return typeof id === "string" && ObjectId.isValid(id) && /^[0-9a-f]{24}$/i.test(id) ? new ObjectId(id).getTimestamp().getTime() : null;
}

/**
 * ?pinned=1 and ?mode=booth (NOTES.md [C9]). Read-only. The newest `snapshots` document by
 * pinned_at (404 if none) and every document the scenes read through it: the versions it names,
 * v0's showcase run (distance, model), its train and holdout runs (mean distance over them and
 * the distinct tasks they cover; holdout passed of run count), the pinned
 * scoreboard's rewrite cost, the pinned swap, the pinned cheat edit's card, and the earliest and
 * latest write time of the runs behind every frames document the scenes play.
 */
export async function GET() {
  const ada = await db("ada");
  const snapshot = await ada.collection("snapshots").findOne({}, { sort: { pinned_at: -1 } });
  if (!snapshot) return Response.json({ error: "no snapshots document yet" }, { status: 404 });

  const str = (x: unknown): string | null => (typeof x === "string" && x.length > 0 ? x : null);
  const ghosts = (Array.isArray(snapshot.ghosts) ? snapshot.ghosts : []) as { version_id?: unknown; frames_id?: unknown }[];
  const attemptIds = (Array.isArray(snapshot.attempt_edit_ids) ? snapshot.attempt_edit_ids : []).filter((x: unknown): x is string => typeof x === "string");
  const v0 = str(snapshot.v0);
  const versionIds = [
    ...new Set([snapshot.best_version, snapshot.v0, snapshot.frontier, ...ghosts.map((g) => g.version_id)].map(str).filter((x): x is string => !!x)),
  ];

  const [versions, scoreboard, swap, cheatEdit, attemptEdits, v0Runs] = await Promise.all([
    ada.collection("versions").find({ _id: { $in: versionIds } as never }, { projection: VERSION_FIELDS }).toArray(),
    str(snapshot.scoreboard_id) ? ada.collection("scoreboard").findOne({ _id: snapshot.scoreboard_id }) : null,
    str(snapshot.swaps_id) ? ada.collection("swaps").findOne({ _id: { $in: idsOf([snapshot.swaps_id]) } as never }) : null,
    str(snapshot.cheat_edit_id) ? ada.collection("edits").findOne({ _id: snapshot.cheat_edit_id }) : null,
    ada.collection("edits").find({ _id: { $in: attemptIds } as never }, { projection: { attempt_frames_id: 1 } }).toArray(),
    v0
      ? ada
          .collection("runs")
          .find({ version_id: v0, split: { $in: ["train", "holdout"] } }, { projection: { split: 1, success: 1, distance_m: 1, task_id: 1 } })
          .toArray()
      : Promise.resolve([] as Document[]),
  ]);

  const v0Holdout = v0Runs.filter((r) => r.split === "holdout");
  const v0Measured = v0Runs.filter((r) => typeof r.distance_m === "number" && Number.isFinite(r.distance_m));
  const v0Tasks = new Set(v0Measured.map((r) => str(r.task_id)).filter((x): x is string => !!x)).size;

  const card: CheatCard | null = cheatEdit && str(cheatEdit.frames_id) ? await buildCard(ada, cheatEdit) : null;
  const v0Frames = str(versions.find((v) => String(v._id) === v0)?.showcase_frames_id);

  // Every frames document a scene plays, then the runs behind them.
  const framesIds = [
    ...new Set(
      [
        ...ghosts.map((g) => g.frames_id),
        swap?.left_frames_id,
        swap?.right_frames_id,
        card?.frames_id,
        v0Frames,
        ...attemptEdits.map((e) => e.attempt_frames_id),
      ]
        .map(str)
        .filter((x): x is string => !!x),
    ),
  ];
  const frames = framesIds.length
    ? await ada.collection("frames").find({ _id: { $in: idsOf(framesIds) } as never }, { projection: { run_id: 1 } }).toArray()
    : [];
  const runIds = [...new Set(frames.map((f) => str(f.run_id)).filter((x): x is string => !!x))];
  const runs = runIds.length
    ? await ada.collection("runs").find({ _id: { $in: idsOf(runIds) } as never }, { projection: { distance_m: 1, model_id: 1 } }).toArray()
    : [];
  const times = runs.map((r) => runTime(r._id)).filter((t): t is number => t !== null);
  const v0RunId = str(frames.find((f) => String(f._id) === v0Frames)?.run_id);
  const v0Run = v0RunId ? runs.find((r) => String(r._id) === v0RunId) : undefined;

  return Response.json(
    toJSONSafe({
      snapshot,
      versions,
      v0_walk:
        v0Frames && v0Run && typeof v0Run.distance_m === "number"
          ? { frames_id: v0Frames, distance_m: v0Run.distance_m, model_id: str(v0Run.model_id) }
          : null,
      v0_mean:
        v0Measured.length && v0Tasks
          ? { distance_m: v0Measured.reduce((a, r) => a + (r.distance_m as number), 0) / v0Measured.length, tasks: v0Tasks }
          : null,
      v0_holdout: { passed: v0Holdout.filter((r) => r.success === true).length, runs: v0Holdout.length },
      rewrite_cost_usd: typeof scoreboard?.rewrite_cost_usd === "number" ? scoreboard.rewrite_cost_usd : null,
      swap,
      card,
      replay: {
        first: times.length ? new Date(Math.min(...times)).toISOString() : null,
        last: times.length ? new Date(Math.max(...times)).toISOString() : null,
      },
    }),
  );
}
