// Pure helpers for the scene director (`?pinned=1`) and booth attract mode (`?mode=booth`),
// NOTES.md [C9]. Every scene reads its ids from the newest `snapshots` document through
// api/snapshot/latest; only the live counters keep moving. Nothing here invents a number: each
// line below formats a value read from a stored document, or returns null.
// No three/React/mongodb imports so this runs under node --test.
import { orderAttempts, type Attempt, type Slot } from "./attempts.ts";
import { parseCheat, type CheatCard } from "./cheatcard.ts";
import type { VersionDoc } from "./ghosts.ts";
import { formatMeters, formatUsd } from "./overlays.ts";
import { clockTime, parseSwap, type SwapDoc } from "./swap.ts";

export const SCENE_IDS = [1, 2, 3, 4, 5, 6, 7] as const;
export type SceneId = (typeof SCENE_IDS)[number];

export const SCENE_NAMES: Record<SceneId, string> = {
  1: "v0 alone",
  2: "the spine, frontier row",
  3: "attempts",
  4: "the cheat, bullet time",
  5: "the swap",
  6: "the spine comparison",
  7: "the finale",
};

/** Booth attract mode: these scenes, in this order, each for a fixed number of seconds. */
export const BOOTH: { scene: SceneId; seconds: number }[] = [
  { scene: 1, seconds: 12 },
  { scene: 3, seconds: 45 },
  { scene: 4, seconds: 20 },
  { scene: 5, seconds: 20 },
  { scene: 7, seconds: 20 },
];
/** A scene the booth staff jumps to that is not in the sequence (2 or 6) holds this long. */
export const OFF_SEQUENCE_S = 20;

export const FINALE_CORNER = "Atlas change stream · live";

/** "1".."7" -> that scene; anything else (or a key with a modifier) -> null. */
export function sceneForKey(key: string, modified = false): SceneId | null {
  if (modified || !/^[1-7]$/.test(key)) return null;
  return Number(key) as SceneId;
}

/** Seconds the booth stays on `scene`. */
export function boothSeconds(scene: SceneId): number {
  return BOOTH.find((b) => b.scene === scene)?.seconds ?? OFF_SEQUENCE_S;
}

/** The booth scene after `scene`: the next one in the sequence by number, wrapping. A jump continues from there. */
export function boothNext(scene: SceneId): SceneId {
  return (BOOTH.find((b) => b.scene > scene) ?? BOOTH[0]).scene;
}

/** `?mode=booth` implies the pinned snapshot: the booth only replays what was pinned. */
export function directorParams(search: string): { pinned: boolean; booth: boolean } {
  const q = new URLSearchParams(search);
  const booth = q.get("mode") === "booth";
  return { booth, pinned: booth || q.get("pinned") === "1" };
}

/** ui/public/script.json: scene number -> caption. Non-scene keys and empty captions are dropped. */
export function parseScript(raw: unknown): Partial<Record<SceneId, string>> {
  const out: Partial<Record<SceneId, string>> = {};
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return out;
  for (const [k, v] of Object.entries(raw)) {
    const id = sceneForKey(k.trim());
    if (id !== null && typeof v === "string" && v.trim().length > 0) out[id] = v.trim();
  }
  return out;
}

export type SnapshotGhost = { version_id: string; frames_id: string | null };

/** A `snapshots` document (NOTES.md [A12]) as the UI reads it. */
export type Snapshot = {
  _id: string;
  pinned_at: string;
  best_version: string | null;
  v0: string | null;
  frontier: string | null;
  attempt_edit_ids: string[];
  cheat_edit_id: string | null;
  swaps_id: string | null;
  scoreboard_id: string | null;
  /** Oldest first. */
  ghosts: SnapshotGhost[];
};

/** What api/snapshot/latest returns: the snapshot and every document the scenes read through it. */
export type Pinned = {
  snapshot: Snapshot;
  versions: VersionDoc[];
  /** v0's showcase run: what scene 1 plays. */
  v0_walk: { frames_id: string; distance_m: number; model_id: string | null } | null;
  /** v0's train and holdout runs: mean distance over them, and how many distinct tasks they cover. */
  v0_mean: { distance_m: number; tasks: number } | null;
  /** v0's holdout runs: how many succeeded, of how many. */
  v0_holdout: { passed: number; runs: number };
  /** The pinned scoreboard's rewrite cost, or null. */
  rewrite_cost_usd: number | null;
  swap: SwapDoc | null;
  card: CheatCard | null;
  /** Earliest and latest write time of the runs behind every frames doc the scenes play. */
  replay: { first: string | null; last: string | null };
};

// Frames ids go into /api/frames/[id] URLs: the same charset parseFrameIds accepts.
const FRAMES_ID = /^[A-Za-z0-9_-]{1,64}$/;
const str = (x: unknown): string | null => (typeof x === "string" && x.length > 0 ? x : null);
const framesId = (x: unknown): string | null => (typeof x === "string" && FRAMES_ID.test(x) ? x : null);
const count = (x: unknown): x is number => typeof x === "number" && Number.isInteger(x) && x >= 0;
const isoOrNull = (x: unknown): string | null => (typeof x === "string" && !Number.isNaN(Date.parse(x)) ? x : null);

/** Validates api/snapshot/latest; null when the snapshot itself is malformed. Bad parts become null. */
export function parsePinned(raw: unknown): Pinned | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Record<string, unknown>;
  const s = d.snapshot as Record<string, unknown> | undefined;
  if (!s || !str(s._id) || !isoOrNull(s.pinned_at)) return null;
  const ghosts = (Array.isArray(s.ghosts) ? s.ghosts : [])
    .filter((g): g is Record<string, unknown> => !!g && typeof g === "object" && !!str((g as Record<string, unknown>).version_id))
    .map((g) => ({ version_id: g.version_id as string, frames_id: framesId(g.frames_id) }));
  const snapshot: Snapshot = {
    _id: s._id as string,
    pinned_at: s.pinned_at as string,
    best_version: str(s.best_version),
    v0: str(s.v0),
    frontier: str(s.frontier),
    attempt_edit_ids: (Array.isArray(s.attempt_edit_ids) ? s.attempt_edit_ids : []).filter((x): x is string => !!str(x)),
    cheat_edit_id: str(s.cheat_edit_id),
    swaps_id: str(s.swaps_id),
    scoreboard_id: str(s.scoreboard_id),
    ghosts,
  };
  const versions = (Array.isArray(d.versions) ? d.versions : []).filter(
    (v): v is VersionDoc => !!v && typeof v === "object" && !!str((v as VersionDoc)._id),
  );
  const w = d.v0_walk as Record<string, unknown> | null | undefined;
  const v0Walk =
    w && framesId(w.frames_id) && typeof w.distance_m === "number" && Number.isFinite(w.distance_m)
      ? { frames_id: w.frames_id as string, distance_m: w.distance_m, model_id: str(w.model_id) }
      : null;
  const m = d.v0_mean as Record<string, unknown> | null | undefined;
  const v0Mean =
    m && typeof m.distance_m === "number" && Number.isFinite(m.distance_m) && count(m.tasks) && m.tasks > 0
      ? { distance_m: m.distance_m, tasks: m.tasks }
      : null;
  const h = d.v0_holdout as Record<string, unknown> | undefined;
  const v0Holdout = h && count(h.passed) && count(h.runs) && h.passed <= h.runs ? { passed: h.passed, runs: h.runs } : { passed: 0, runs: 0 };
  const cost = d.rewrite_cost_usd;
  // The card rides in api/cheat's shape; parseCheat validates it with a stand-in counter and spine.
  const card = parseCheat({ card: d.card, counter: { overrated_gaits: 0, passed_physics: 0 }, spine: { models: {}, edits: [] } })?.card ?? null;
  const r = d.replay as Record<string, unknown> | undefined;
  return {
    snapshot,
    versions,
    v0_walk: v0Walk,
    v0_mean: v0Mean,
    v0_holdout: v0Holdout,
    rewrite_cost_usd: typeof cost === "number" && Number.isFinite(cost) ? cost : null,
    swap: parseSwap(d.swap),
    card,
    replay: { first: isoOrNull(r?.first), last: isoOrNull(r?.last) },
  };
}

/** A version the snapshot names, or null. */
export function pinnedVersion(p: Pinned, id: string | null): VersionDoc | null {
  return id === null ? null : (p.versions.find((v) => v._id === id) ?? null);
}

/** The snapshot's ghosts that have frames, oldest first, as the ghost loader reads them. */
export function pinnedGhostSources(p: Pinned): { versionId: string; framesId: string }[] {
  return p.snapshot.ghosts.filter((g) => g.frames_id !== null).map((g) => ({ versionId: g.version_id, framesId: g.frames_id as string }));
}

/**
 * Scene 3 plays exactly the pinned attempts, in the pinned order (scripts/pin_snapshot.py ran
 * orderAttempts at pin time). Which ones replay at speed still comes from orderAttempts on that
 * set, so the worst bet and the largest gain play at 1×. Ids with no drawable attempt are skipped.
 */
export function pinnedSlots(attempts: Attempt[], ids: string[]): Slot[] {
  const rank = new Map(ids.map((id, i) => [id, i]));
  return orderAttempts(attempts.filter((a) => rank.has(a.edit_id))).sort(
    (a, b) => (rank.get(a.attempt.edit_id) as number) - (rank.get(b.attempt.edit_id) as number),
  );
}

/** Every frames doc a scene plays: fetched once when the director loads, so a key switch is instant. */
export function pinnedFramesIds(p: Pinned): string[] {
  const ids = [
    ...pinnedGhostSources(p).map((g) => g.framesId),
    p.v0_walk?.frames_id,
    p.card?.frames_id,
    p.swap?.left_frames_id,
    p.swap?.right_frames_id,
  ];
  return [...new Set(ids.filter((x): x is string => typeof x === "string"))];
}

/** Scene 1's caption for the walk it plays, so the single showcase walk is never read as the mean. */
export const V0_WALK_LABEL = "one walk · showcase terrain";

/**
 * Scene 1's one line: "{v0 mean} m mean over {tasks} tasks · {model} · ${cost} a gait · harness v0 · {passed}/{runs}".
 * Mean distance and task count from v0's train and holdout runs, model from its showcase run,
 * cost from versions.metrics.cost_per_run_usd (the spine's source), passed of the holdout runs
 * counted. Null when any of them is missing.
 */
export function v0Line(p: Pinned): string | null {
  const v0 = pinnedVersion(p, p.snapshot.v0);
  const cost = formatUsd(v0?.metrics?.cost_per_run_usd);
  if (!v0 || !p.v0_mean || !p.v0_walk || !p.v0_walk.model_id || !cost || p.v0_holdout.runs === 0) return null;
  const mean = `${formatMeters(p.v0_mean.distance_m)} mean over ${p.v0_mean.tasks} tasks`;
  return `${mean} · ${p.v0_walk.model_id} · ${cost} a gait · harness ${v0._id} · ${p.v0_holdout.passed}/${p.v0_holdout.runs}`;
}

/** "Sep 26" in `timeZone` (the viewer's zone when omitted). */
export function dayLabel(iso: string, timeZone?: string): string {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone }).format(new Date(iso));
}

/**
 * The booth label, from the runs' own write times: "Replay of runs recorded Sep 26, 15:20–17:36",
 * or with both dates when they differ. It never says "today": the booth plays days later.
 */
export function replayLabel(first: string | null, last: string | null, timeZone?: string): string | null {
  if (!first || !last) return null;
  const d1 = dayLabel(first, timeZone);
  const d2 = dayLabel(last, timeZone);
  const t1 = clockTime(first, timeZone);
  const t2 = clockTime(last, timeZone);
  return d1 === d2 ? `Replay of runs recorded ${d1}, ${t1}–${t2}` : `Replay of runs recorded ${d1} ${t1}–${d2} ${t2}`;
}
