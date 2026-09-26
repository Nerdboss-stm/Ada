// Pure helpers for ?mode=attempts: the harness's own bets, one attempt at a time. Each attempt
// reads one edits document (rationale, primitive, predicted_delta_m, actual_delta_m,
// attempt_frames_id, verdict) and the train means of its parent and candidate versions
// (NOTES.md [A11]). Nothing is hand-picked: orderAttempts is the only thing that decides what
// plays, and every number on screen comes from a stored document.
// No three/React imports so this runs under node --test.
import type { VersionDoc } from "./ghosts.ts";
import { formatDelta, type EditDoc } from "./overlays.ts";

/** The gate's train terrains (loop/subset.py GATE_TASKS): what a train mean is averaged over. */
export const GATE_TERRAINS = 6;
export const WON = "#85BB65";
export const LOST = "#FF4A3D";

/** The terrain one attempt walked, from frames.run_id -> runs.task_id -> tasks. */
export type Terrain = { task_id: string; slope_deg?: number | null; friction?: number | null };

/** What /api/attempts returns and what an attempts fixture holds. */
export type AttemptsDoc = {
  fixture?: boolean;
  versions: VersionDoc[];
  edits: EditDoc[];
  /** attempt_frames_id -> terrain, null when the run is not recorded. */
  terrains: Record<string, Terrain | null>;
};

export type Attempt = {
  edit_id: string;
  created_at: string;
  tag: string;
  rationale: string;
  framesId: string;
  terrain: Terrain | null;
  /** predicted_delta_m. */
  bet: number;
  parentMean: number;
  candidateMean: number;
  /** actual_delta_m (or candidate minus parent when absent). */
  measured: number;
  /** Where the dashed line drops: parent train mean plus the bet. */
  line: number;
  reached: boolean;
  kept: boolean;
};

const TAGS: Record<string, string> = {
  rules: "RULES",
  context_policy: "CONTEXT",
  tools: "TOOLS",
  model_per_step: "MODEL",
  engine: "ENGINE",
};

/** SPEC §3 primitive -> the tag on screen. */
export function primitiveTag(primitive: string | undefined): string {
  return TAGS[primitive ?? ""] ?? (primitive ?? "").toUpperCase();
}

// Frames ids go into /api/frames/[id] URLs: the same charset parseFrameIds accepts.
const FRAMES_ID = /^[A-Za-z0-9_-]{1,64}$/;
const FIXTURE_NAME = /^[a-z0-9_-]{1,64}$/;
const EPS = 1e-9;

const finite = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

/** `?attemptsfixture=name` -> the fixture file; otherwise the live route. Bad names fall back to the API. */
export function attemptsSourceUrl(search: string): string {
  const name = new URLSearchParams(search).get("attemptsfixture")?.trim().toLowerCase() ?? "";
  return FIXTURE_NAME.test(name) ? `/fixtures/${name}.json` : "/api/attempts";
}

/** The document if its three collections are well-formed, else null. */
export function parseAttemptsDoc(raw: unknown): AttemptsDoc | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Record<string, unknown>;
  if (!Array.isArray(d.versions) || !Array.isArray(d.edits)) return null;
  if (!d.terrains || typeof d.terrains !== "object" || Array.isArray(d.terrains)) return null;
  const hasId = (x: unknown) => !!x && typeof x === "object" && typeof (x as { _id?: unknown })._id === "string";
  if (!d.versions.every(hasId) || !d.edits.every(hasId)) return null;
  return d as unknown as AttemptsDoc;
}

const trainMean = (v: VersionDoc | undefined): number | null => {
  const m = v?.metrics?.train_mean_distance_m;
  return finite(m) ? m : null;
};

/**
 * One attempt, or null when it cannot be drawn from stored numbers alone: no attempt_frames_id
 * or predicted_delta_m (the brief's skip rule), no verdict yet, no parent train mean, or no
 * measured change. Nothing is filled in.
 */
export function toAttempt(edit: EditDoc, versions: Map<string, VersionDoc>, terrains: AttemptsDoc["terrains"]): Attempt | null {
  const framesId = edit.attempt_frames_id;
  const bet = edit.predicted_delta_m;
  if (typeof framesId !== "string" || !FRAMES_ID.test(framesId) || !finite(bet)) return null;
  if (edit.verdict !== "accepted" && edit.verdict !== "rejected") return null;
  const parentMean = trainMean(versions.get(edit.from_version ?? ""));
  if (parentMean === null) return null;
  const candidate = trainMean(versions.get(edit.to_version ?? ""));
  const measured = finite(edit.actual_delta_m) ? edit.actual_delta_m : candidate !== null ? candidate - parentMean : null;
  if (measured === null) return null;
  const candidateMean = candidate ?? parentMean + measured;
  const line = parentMean + bet;
  return {
    edit_id: edit._id,
    created_at: edit.created_at ?? "",
    tag: primitiveTag(edit.primitive),
    rationale: edit.rationale ?? "",
    framesId,
    terrain: terrains[framesId] ?? null,
    bet,
    parentMean,
    candidateMean,
    measured,
    line,
    reached: candidateMean >= line - EPS,
    kept: edit.verdict === "accepted",
  };
}

/** Every drawable attempt, oldest first (created_at, then _id). */
export function attemptsOf(doc: AttemptsDoc): Attempt[] {
  const versions = new Map(doc.versions.map((v) => [v._id, v]));
  return doc.edits
    .map((e) => toAttempt(e, versions, doc.terrains))
    .filter((a): a is Attempt => a !== null)
    .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.edit_id.localeCompare(b.edit_id));
}

/** Every rejected edit in the data, played or not: the lessons thrown out. */
export function rejectedTotal(edits: EditDoc[]): number {
  return edits.filter((e) => e.verdict === "rejected").length;
}

export type Slot = { attempt: Attempt; fast: boolean };

/**
 * The play order. First the worst bet (largest predicted gain whose measured change is
 * negative), then the largest measured gain, then every other kept edit, oldest first, fast.
 * Ties keep the older attempt. Input must be oldest first (attemptsOf).
 */
export function orderAttempts(attempts: Attempt[]): Slot[] {
  const out: Slot[] = [];
  const used = new Set<string>();
  let worst: Attempt | null = null;
  for (const a of attempts) if (a.measured < 0 && (!worst || a.bet > worst.bet)) worst = a;
  if (worst) {
    out.push({ attempt: worst, fast: false });
    used.add(worst.edit_id);
  }
  let gain: Attempt | null = null;
  for (const a of attempts) if (!used.has(a.edit_id) && a.measured > 0 && (!gain || a.measured > gain.measured)) gain = a;
  if (gain) {
    out.push({ attempt: gain, fast: false });
    used.add(gain.edit_id);
  }
  for (const a of attempts) if (a.kept && !used.has(a.edit_id)) out.push({ attempt: a, fast: true });
  return out;
}

/** Seconds within one attempt: type until typeEnd, walk intro..walkEnd at `speed`, stamp, file. */
export type Timeline = { typeEnd: number; intro: number; walkEnd: number; stampEnd: number; end: number; speed: number };
export const NORMAL: Timeline = { typeEnd: 2.6, intro: 3, walkEnd: 10, stampEnd: 11.5, end: 13, speed: 1 };
export const FAST: Timeline = { typeEnd: 0.2, intro: 0.25, walkEnd: 1, stampEnd: 1.15, end: 1.5, speed: 4 };
/** The complete notebook holds this long before the loop restarts. */
export const HOLD_S = 4;

export type Scheduled = Slot & { start: number; timeline: Timeline };
export type Schedule = { slots: Scheduled[]; total: number };

export function schedule(slots: Slot[]): Schedule {
  let start = 0;
  const out = slots.map((s) => {
    const timeline = s.fast ? FAST : NORMAL;
    const sc = { ...s, start, timeline };
    start += timeline.end;
    return sc;
  });
  return { slots: out, total: out.length ? start + HOLD_S : 0 };
}

/** Which slot is on stage at loop time t, and its local time. During the hold, the last slot at its end. */
export function locate(sc: Schedule, t: number): { index: number; local: number; holding: boolean } {
  if (sc.slots.length === 0 || !(sc.total > 0)) return { index: -1, local: 0, holding: false };
  const tt = Number.isFinite(t) && t > 0 ? t % sc.total : 0;
  for (let i = 0; i < sc.slots.length; i++) {
    const s = sc.slots[i];
    if (tt < s.start + s.timeline.end) return { index: i, local: tt - s.start, holding: false };
  }
  const last = sc.slots.length - 1;
  return { index: last, local: sc.slots[last].timeline.end, holding: true };
}

/** Recorded seconds shown at local time: 0 before the walk, held at the walk's last recorded instant after it. */
export function recordedTime(tl: Timeline, local: number): number {
  const walk = Math.min(Math.max(local, tl.intro), tl.walkEnd) - tl.intro;
  return walk * tl.speed;
}

/** The recorded frame shown at local time. No interpolation. */
export function attemptFrame(tl: Timeline, local: number, fps: number, n: number): number {
  if (n <= 0 || !(fps > 0)) return 0;
  return Math.min(Math.floor(recordedTime(tl, local) * fps + EPS), n - 1);
}

export type Phase = "intro" | "walk" | "stamp" | "file";

export function phaseAt(tl: Timeline, local: number): Phase {
  if (local < tl.intro) return "intro";
  if (local < tl.walkEnd) return "walk";
  if (local < tl.stampEnd) return "stamp";
  return "file";
}

/** Characters of the rationale shown: typed evenly until typeEnd, whole after. */
export function typedChars(text: string, tl: Timeline, local: number): number {
  if (!(local > 0)) return 0;
  return Math.min(text.length, Math.floor((text.length * local) / tl.typeEnd));
}

/** 0..1 through a window, clamped. */
export function progress(local: number, from: number, to: number): number {
  if (!(to > from)) return local >= to ? 1 : 0;
  return Math.min(1, Math.max(0, (local - from) / (to - from)));
}

export type AttemptsView = {
  index: number;
  phase: Phase;
  local: number;
  typed: number;
  holding: boolean;
  /** Kept attempts whose filing has finished, in play order. */
  notebook: Attempt[];
};

/** Everything the overlay shows at loop time t. */
export function attemptsView(sc: Schedule, t: number): AttemptsView | null {
  const at = locate(sc, t);
  if (at.index < 0) return null;
  const slot = sc.slots[at.index];
  const done = at.holding ? sc.slots.length : at.index;
  return {
    index: at.index,
    phase: at.holding ? "file" : phaseAt(slot.timeline, at.local),
    local: at.local,
    typed: typedChars(slot.attempt.rationale, slot.timeline, at.local),
    holding: at.holding,
    notebook: sc.slots.slice(0, done).filter((s) => s.attempt.kept).map((s) => s.attempt),
  };
}

/** "bets +0.40 m": the supervisor's number, signed, 2 decimals. */
export function betText(bet: number): string {
  return `bets ${formatDelta(bet) ?? bet.toFixed(2)} m`;
}

/** What a kept edit earned in the notebook: its measured change. */
export function earnedText(measured: number): string {
  return `${formatDelta(measured) ?? measured.toFixed(2)} m`;
}

export const STAMP_NOTE = `mean over ${GATE_TERRAINS} terrains`;

/** The single walk's caption, so it is never mistaken for the mean. */
export function terrainLabel(t: Terrain | null): string {
  if (!t) return "one walk · terrain not recorded";
  const parts = [`one walk · ${t.task_id}`];
  if (finite(t.slope_deg)) parts.push(`${t.slope_deg}° slope`);
  if (finite(t.friction)) parts.push(`friction ${t.friction}`);
  return parts.join(" · ");
}
