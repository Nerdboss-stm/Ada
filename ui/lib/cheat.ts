// Pure helpers for SPEC §6 "The cheat, bullet time". The cheat body shows only recorded frames:
// slow motion holds each frame longer, it never makes new ones. No three/React imports.
import { mjToThree, type Vec3 } from "./ghosts.ts";
import { latestEdit, type EditDoc } from "./overlays.ts";
import type { FramesDoc } from "./replay.ts";

export const CHEAT_RED = "#FF4A3D";
/** Playback speed during bullet time. */
export const SLOW = 0.1;
/** Bullet time covers recorded time from this long before the violation frame... */
export const SLOW_BEFORE_S = 0.3;
/** ...to this long after it. */
export const SLOW_AFTER_S = 0.2;
/** Wall seconds the camera takes to reach the flail; it orbits for the rest of bullet time. */
export const EASE_IN_S = 1;
/** Wall seconds the camera takes to get back to the leader after bullet time. */
export const RETURN_S = 1.5;
/** The cheat plays in its own lane, this many meters to the side (MuJoCo +y, away from the camera). */
export const LANE_Y = 1.5;
const ORBIT_RADIUS = 4.5;
const ORBIT_RISE = 0.8;
// Start where the follow cam sits (behind and to the camera side), then go once around.
const ORBIT_START = Math.atan2(3, -4);

/** A frames doc the UI may play as a cheat: kind rejected with an integer violation_frame inside the run. */
export function isCheatDoc(doc: Pick<FramesDoc, "kind" | "frames" | "violation_frame">): boolean {
  const v = doc.violation_frame;
  return doc.kind === "rejected" && Number.isInteger(v) && (v as number) >= 0 && (v as number) < doc.frames.length;
}

/** `?frames=` mode: the first listed doc that is a cheat, or null. */
export function pickCheatDoc(docs: { key: string; doc: FramesDoc }[]): string | null {
  return docs.find((d) => isCheatDoc(d.doc))?.key ?? null;
}

/** Stream mode: the newest rejected edit that has recorded frames. */
export function latestRejectedEdit(edits: EditDoc[]): EditDoc | null {
  return latestEdit(edits.filter((e) => e.verdict === "rejected" && typeof e.frames_id === "string" && e.frames_id.length > 0));
}

/** Stream mode: the edit's reason, word for word. `?frames=` mode (or no reason): "rejected at frame {v}". */
export function verdictText(reason: string | null | undefined, violationFrame: number): string {
  return typeof reason === "string" && reason.trim().length > 0 ? reason : `rejected at frame ${violationFrame}`;
}

export type CheatTimeline = {
  fps: number;
  n: number;
  violationFrame: number;
  /** Recorded time where bullet time starts and ends. */
  slowFrom: number;
  slowTo: number;
  /** Wall seconds bullet time lasts: (slowTo - slowFrom) / SLOW. */
  slowWall: number;
  easeIn: number;
  /** Wall seconds of one pass; the cheat loops on its own. */
  loop: number;
};

/** Null when the doc cannot be played (no fps, no frames, violation frame outside the run). */
export function cheatTimeline(fps: number, n: number, violationFrame: number): CheatTimeline | null {
  if (!(fps > 0) || n <= 0 || !Number.isInteger(violationFrame) || violationFrame < 0 || violationFrame >= n) return null;
  const duration = n / fps;
  const tv = violationFrame / fps;
  const slowFrom = Math.max(0, tv - SLOW_BEFORE_S);
  const slowTo = Math.min(duration, tv + SLOW_AFTER_S);
  const slowWall = (slowTo - slowFrom) / SLOW;
  const easeIn = Math.min(EASE_IN_S, slowWall / 5);
  const played = duration + (slowWall - (slowTo - slowFrom));
  // A violation near the end still gets its camera return before the pass restarts.
  const loop = Math.max(played, slowFrom + slowWall + RETURN_S);
  return { fps, n, violationFrame, slowFrom, slowTo, slowWall, easeIn, loop };
}

/** Recorded time shown `w` wall seconds into a pass: 1×, then 0.1× through bullet time, then 1×. */
export function recordedTime(tl: CheatTimeline, w: number): number {
  if (w < tl.slowFrom) return w;
  if (w < tl.slowFrom + tl.slowWall) return tl.slowFrom + SLOW * (w - tl.slowFrom);
  return tl.slowTo + (w - tl.slowFrom - tl.slowWall);
}

const smooth = (x: number): number => {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
};

export type CheatState = {
  /** Wall seconds into the current pass. */
  w: number;
  /** The recorded frame to show; holds the last one if the pass outlasts the run. */
  frame: number;
  /** 0 = camera on the leader, 1 = camera on the orbit. */
  camWeight: number;
  /** Orbit progress 0..1. */
  orbit: number;
  verdict: boolean;
  /** Recorded time is playing at 0.1× (the red joint glows only now). */
  bulletTime: boolean;
  /** The cheat card is up: from the start of bullet time until the camera is back on the leader. */
  card: boolean;
};

/** Everything the cheat body, camera, and verdict need at shared playback time t. */
export function cheatState(tl: CheatTimeline, t: number): CheatState {
  const w = Number.isFinite(t) && t > 0 ? t % tl.loop : 0;
  // The epsilon keeps a frame boundary that float math lands a hair short of on the right frame.
  const frame = Math.min(Math.max(0, Math.floor(recordedTime(tl, w) * tl.fps + 1e-9)), tl.n - 1);
  const slowEnd = tl.slowFrom + tl.slowWall;
  const back = slowEnd + RETURN_S;
  let camWeight = 0;
  if (w >= tl.slowFrom && w < tl.slowFrom + tl.easeIn) camWeight = smooth((w - tl.slowFrom) / tl.easeIn);
  else if (w >= tl.slowFrom + tl.easeIn && w < slowEnd) camWeight = 1;
  else if (w >= slowEnd && w < back) camWeight = 1 - smooth((w - slowEnd) / RETURN_S);
  const orbitWall = tl.slowWall - tl.easeIn;
  const orbit = orbitWall > 0 ? Math.min(1, Math.max(0, (w - tl.slowFrom - tl.easeIn) / orbitWall)) : 1;
  // Appears when the violation frame is on screen, stays until the camera is back on the leader.
  const verdict = frame >= tl.violationFrame && w >= tl.slowFrom && w < back;
  const bulletTime = w >= tl.slowFrom && w < slowEnd;
  const card = w >= tl.slowFrom && w < back;
  return { w, frame, camWeight, orbit, verdict, bulletTime, card };
}

/** The cheat's recorded torso (MuJoCo) in its lane, in three's y-up world. */
export function cheatTorso(torso: Vec3): Vec3 {
  return mjToThree([torso[0], torso[1] + LANE_Y, torso[2]]);
}

/** Camera on the orbit around `center` (three coords), once around over progress 0..1. */
export function orbitCamera(center: Vec3, progress: number): { pos: Vec3; look: Vec3 } {
  const a = ORBIT_START + 2 * Math.PI * smooth(progress);
  return {
    pos: [center[0] + ORBIT_RADIUS * Math.cos(a), center[1] + ORBIT_RISE, center[2] + ORBIT_RADIUS * Math.sin(a)],
    look: center,
  };
}
