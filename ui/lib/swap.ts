// Pure helpers for ?mode=swap: the same model and the same holdout task, harness v0 in the left
// lane and the best harness in the right, from the newest `swaps` document (NOTES.md [C6]).
// Every number on screen is read from that document; the bodies play recorded frames only.
// No three/React imports so this runs under node --test.
import { mjToThree, type Vec3 } from "./ghosts.ts";
import { HOLDOUT_TASKS } from "./overlays.ts";

/** A `swaps` document as the UI reads it. `fixture` is set only in ui/public/fixtures files. */
export type SwapDoc = {
  left_version: string;
  right_version: string;
  task_id: string;
  left_frames_id: string;
  right_frames_id: string;
  /** Holdout terrains passed, of HOLDOUT_TASKS. */
  left_holdout: number;
  right_holdout: number;
  model_id: string;
  verifier_sha: string;
  mujoco_version: string;
  manifest_version: string;
  /** Changed harness lines, v0 → best. */
  harness_diff: string[];
  captured_at: string;
  fixture?: boolean;
};

const STRING_FIELDS = [
  "left_version",
  "right_version",
  "task_id",
  "left_frames_id",
  "right_frames_id",
  "model_id",
  "verifier_sha",
  "mujoco_version",
  "manifest_version",
  "captured_at",
] as const;

// Frames ids go into /api/frames/[id] URLs: the same charset parseFrameIds accepts.
const FRAMES_ID = /^[A-Za-z0-9_-]{1,64}$/;
const FIXTURE_NAME = /^[a-z0-9_-]{1,64}$/;

const isTasks = (x: unknown): x is number => Number.isInteger(x) && (x as number) >= 0 && (x as number) <= HOLDOUT_TASKS;

/** The document if every field is present and well-formed, else null (nothing is guessed). */
export function parseSwap(raw: unknown): SwapDoc | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Record<string, unknown>;
  for (const k of STRING_FIELDS) if (typeof d[k] !== "string" || (d[k] as string).length === 0) return null;
  if (!FRAMES_ID.test(d.left_frames_id as string) || !FRAMES_ID.test(d.right_frames_id as string)) return null;
  if (!isTasks(d.left_holdout) || !isTasks(d.right_holdout)) return null;
  if (!Array.isArray(d.harness_diff) || !d.harness_diff.every((l) => typeof l === "string")) return null;
  if (Number.isNaN(Date.parse(d.captured_at as string))) return null;
  return d as unknown as SwapDoc;
}

/** `?swapfixture=name` -> the fixture file; otherwise the newest swaps document. Bad names fall back to the API. */
export function swapSourceUrl(search: string): string {
  const name = new URLSearchParams(search).get("swapfixture")?.trim().toLowerCase() ?? "";
  return FIXTURE_NAME.test(name) ? `/fixtures/${name}.json` : "/api/swaps/latest";
}

/** Top center, 72 pt. */
export function swapHeadline(d: Pick<SwapDoc, "left_holdout" | "right_holdout">): string {
  return `holdout ${d.left_holdout}/${HOLDOUT_TASKS} → ${d.right_holdout}/${HOLDOUT_TASKS}`;
}

/** 24-hour HH:MM of captured_at, in `timeZone` (the viewer's zone when omitted). */
export function clockTime(iso: string, timeZone?: string): string {
  const parts = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone }).formatToParts(
    new Date(iso),
  );
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "00";
  return `${get("hour")}:${get("minute")}`;
}

/** Beneath the headline, 24 pt. */
export function swapSubline(capturedAt: string, timeZone?: string): string {
  return `tasks the loop never saw · recorded back-to-back at ${clockTime(capturedAt, timeZone)}`;
}

export const PHYSICS_HASH_CHARS = 12;

export type DiffRow = { label: string; left: string; right: string; identical: boolean; lines?: string[] };

/**
 * The two-column diff under the lanes. One document describes both runs, so model, task, physics
 * and sim are the same value in both columns by construction and marked identical; the harness is
 * the only row that differs, and it expands into the changed lines.
 */
export function swapDiff(d: SwapDoc): DiffRow[] {
  const same = (label: string, value: string): DiffRow => ({ label, left: value, right: value, identical: true });
  return [
    same("model", d.model_id),
    same("task", d.task_id),
    same("physics hash", d.verifier_sha.slice(0, PHYSICS_HASH_CHARS)),
    same("sim version", `mujoco ${d.mujoco_version} · manifest ${d.manifest_version}`),
    { label: "harness", left: d.left_version, right: d.right_version, identical: false, lines: d.harness_diff },
  ];
}

/**
 * Lane offsets, MuJoCo +y. The swap camera sits behind the start line looking down +x, where
 * MuJoCo +y is screen left: v0 left, best right, same start line.
 */
export const SWAP_LANE_Y = { left: 0.9, right: -0.9 } as const;

/** A recorded torso (MuJoCo) moved into its lane, in three's y-up world. */
export function laneTorso(torso: Vec3, laneY: number): Vec3 {
  return mjToThree([torso[0], torso[1] + laneY, torso[2]]);
}

// Behind and above the midpoint of the two torsos, looking a little ahead of it.
const BACK = 6.8;
const RISE = 4.2;
const AHEAD = 0.4;
/** Extra pull-back per meter the two torsos are apart along the track, so both stay framed. */
const SPREAD_BACK = 0.9;
const SPREAD_RISE = 0.5;

/** Camera framing both lane torsos (three coords in, three coords out). */
export function swapCamera(a: Vec3, b: Vec3): { pos: Vec3; look: Vec3 } {
  const mid: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  const spread = Math.abs(a[0] - b[0]);
  return {
    pos: [mid[0] - BACK - SPREAD_BACK * spread, mid[1] + RISE + SPREAD_RISE * spread, mid[2]],
    look: [mid[0] + AHEAD, mid[1] * 0.5, mid[2]],
  };
}
