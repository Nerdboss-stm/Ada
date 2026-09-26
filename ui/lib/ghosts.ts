// Pure ghost helpers (SPEC §6 "Ghosts"): ordering, colors, leader, shared clock, follow cam.
// No three/React imports so this runs under node --test.
import type { FramesDoc } from "./replay";

export const GHOST_OLDEST = "#3a6bff";
export const GHOST_NEWEST = "#e8eefc";
export const GHOST_OPACITY = 0.18;
/** Follow-cam lerp per 60 Hz frame. */
export const FOLLOW_ALPHA = 0.05;

/** The slice of a `versions` document (CONTRACTS.md §3) the ghosts read. */
export type VersionDoc = {
  _id: string;
  status?: string;
  metrics?: {
    holdout_reliability_80?: number | null;
    mean_distance_m?: number | null;
    cost_per_run_usd?: number | null;
    n?: number | null;
  } | null;
  showcase_frames_id?: string | null;
  created_at?: string;
};

export type Vec3 = [number, number, number];

/** "v12" -> 12; anything else sorts last. */
export function versionRank(id: string): number {
  const m = /^v(\d+)$/.exec(id);
  return m ? Number(m[1]) : Number.MAX_SAFE_INTEGER;
}

/** Oldest first: by version number, then created_at, then _id. */
export function sortVersions<T extends VersionDoc>(versions: T[]): T[] {
  return [...versions].sort(
    (a, b) =>
      versionRank(a._id) - versionRank(b._id) ||
      (a.created_at ?? "").localeCompare(b.created_at ?? "") ||
      a._id.localeCompare(b._id),
  );
}

/** Dedupe by _id, incoming wins. */
export function mergeVersions<T extends VersionDoc>(current: T[], incoming: T[]): T[] {
  const byId = new Map<string, T>();
  for (const v of current) byId.set(v._id, v);
  for (const v of incoming) byId.set(v._id, v);
  return sortVersions([...byId.values()]);
}

/** Versions that have showcase frames, oldest first. */
export function ghostSources(versions: VersionDoc[]): { versionId: string; framesId: string }[] {
  return sortVersions(versions)
    .filter((v) => typeof v.showcase_frames_id === "string" && v.showcase_frames_id.length > 0)
    .map((v) => ({ versionId: v._id, framesId: v.showcase_frames_id as string }));
}

const finite = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

/**
 * The leader, which is also the "current best" the claim is about: among accepted or baseline
 * versions with a holdout reliability, the highest metrics.holdout_reliability_80, ties broken
 * by the highest metrics.mean_distance_m, then the oldest. Null if none.
 */
export function pickLeader(versions: VersionDoc[]): string | null {
  let best: string | null = null;
  let bestR = -Infinity;
  let bestD = -Infinity;
  for (const v of sortVersions(versions)) {
    if (v.status !== "accepted" && v.status !== "baseline") continue;
    const r = v.metrics?.holdout_reliability_80;
    if (!finite(r)) continue;
    const d = finite(v.metrics?.mean_distance_m) ? (v.metrics?.mean_distance_m as number) : -Infinity;
    if (r > bestR || (r === bestR && d > bestD)) {
      bestR = r;
      bestD = d;
      best = v._id;
    }
  }
  return best;
}

/**
 * `?fixtures=` / `?frames=` have no versions document: the leader is the largest recorded
 * final torso x among runs not recorded as rejected.
 */
export function pickFixtureLeader(docs: { key: string; doc: FramesDoc }[]): string | null {
  let best: string | null = null;
  let bestX = -Infinity;
  for (const { key, doc } of docs) {
    if (doc.kind === "rejected") continue;
    const last = doc.frames[doc.frames.length - 1];
    if (last && last.torso[0] > bestX) {
      bestX = last.torso[0];
      best = key;
    }
  }
  return best;
}

/** `?fixtures=a,b,c` -> public fixture URLs. Only [a-z0-9_-] names; anything else is dropped. */
export function parseFixtures(param: string | null): { name: string; url: string }[] {
  if (!param) return [];
  const seen = new Set<string>();
  const out: { name: string; url: string }[] = [];
  for (const raw of param.split(",")) {
    const name = raw.trim().toLowerCase();
    if (!/^[a-z0-9_-]{1,64}$/.test(name) || seen.has(name)) continue;
    seen.add(name);
    out.push({ name, url: `/fixtures/${name}.json` });
  }
  return out;
}

/** `?frames=id1,id2` -> /api/frames/[id] URLs. Only [A-Za-z0-9_-] ids; anything else is dropped. */
export function parseFrameIds(param: string | null): { name: string; url: string }[] {
  if (!param) return [];
  const seen = new Set<string>();
  const out: { name: string; url: string }[] = [];
  for (const raw of param.split(",")) {
    const id = raw.trim();
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(id) || seen.has(id)) continue;
    seen.add(id);
    out.push({ name: id, url: `/api/frames/${id}` });
  }
  return out;
}

function hexToRgb(hex: string): Vec3 {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Ghost i of n, oldest (#3a6bff) to newest (#e8eefc). */
export function ghostColor(i: number, n: number): string {
  const t = n <= 1 ? 1 : Math.min(1, Math.max(0, i / (n - 1)));
  const a = hexToRgb(GHOST_OLDEST);
  const b = hexToRgb(GHOST_NEWEST);
  return (
    "#" +
    a
      .map((c, k) => Math.round(c + (b[k] - c) * t))
      .map((c) => c.toString(16).padStart(2, "0"))
      .join("")
  );
}

/** Seconds of one shared loop: the longest run. */
export function cycleSeconds(docs: Pick<FramesDoc, "fps" | "frames">[]): number {
  let s = 0;
  for (const d of docs) if (d.fps > 0) s = Math.max(s, d.frames.length / d.fps);
  return s;
}

/**
 * Frame shown at shared time t: every ghost starts at t = 0, holds its last recorded
 * frame once it runs out, and all loop together after `cycle` seconds. No interpolation.
 */
export function sharedFrameIndex(t: number, fps: number, n: number, cycle: number): number {
  if (n <= 0 || !(fps > 0) || !Number.isFinite(t) || t < 0) return 0;
  const tc = cycle > 0 ? t % cycle : t;
  return Math.min(Math.floor(tc * fps), n - 1);
}

/** MuJoCo z-up (x, y, z) -> three y-up (x, z, -y). For the camera only; poses stay in the rotated group. */
export function mjToThree(p: Vec3): Vec3 {
  return [p[0], p[2], -p[1]];
}

/** Frame-rate-independent lerp: `alpha` per 60 Hz frame. */
export function followStep(prev: Vec3, target: Vec3, alpha: number, dt: number): Vec3 {
  const k = 1 - Math.pow(1 - alpha, Math.max(0, dt) * 60);
  return [prev[0] + (target[0] - prev[0]) * k, prev[1] + (target[1] - prev[1]) * k, prev[2] + (target[2] - prev[2]) * k];
}
