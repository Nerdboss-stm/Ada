// Pure helpers for SPEC §6 "Sound": which recorded frames fire a cue, and how many falls may
// sound at once. Every cue is read from recorded frames. No Web Audio/React imports.
import type { FramesDoc, Manifest } from "./replay.ts";

export type Cue = "step" | "fall" | "glitch";

/**
 * The one place the clips are defined. Each file is padded with silence after its sound (which
 * starts at 0 s), so only the first `clipS` seconds are played.
 */
export const SFX: Record<Cue, { url: string; clipS: number }> = {
  step: { url: "/sfx/step.mp3", clipS: 0.5 },
  fall: { url: "/sfx/fall.mp3", clipS: 1.0 },
  glitch: { url: "/sfx/glitch.mp3", clipS: 2.0 },
};

/** At most this many fall sounds at once (SPEC §6). */
export const MAX_FALLS = 6;

/** sim/verifier.py fell(): torso below this height, or the torso upside down. */
export const FALL_Z_M = 0.25;
const TORSO_GEOM_NAME = "torso_geom";

/**
 * Cue frames reached for the first time since the last rendered frame. `prev` is the frame shown
 * last (-1 before the first), `cur` the frame shown now. A lower `cur` means the loop restarted:
 * the rest of the old pass, then the start of the new one. Nothing fires twice for one pass.
 */
export function crossedFrames(frames: number[], prev: number, cur: number): number[] {
  if (cur === prev) return [];
  if (cur > prev) return frames.filter((f) => f > prev && f <= cur);
  return frames.filter((f) => f > prev || f <= cur);
}

/**
 * The first recorded frame where the run has fallen, by the verifier's own rule (torso z under
 * FALL_Z_M, or its up axis pointing down), or null if it never falls.
 */
export function fallFrame(doc: Pick<FramesDoc, "frames">, manifest: Pick<Manifest, "geoms">): number | null {
  const g = manifest.geoms.findIndex((m) => m.name === TORSO_GEOM_NAME);
  for (let k = 0; k < doc.frames.length; k++) {
    const f = doc.frames[k];
    if (f.torso[2] < FALL_Z_M) return k;
    const pose = g >= 0 ? f.geoms[g] : undefined;
    if (pose && 1 - 2 * (pose[4] * pose[4] + pose[5] * pose[5]) < 0) return k;
  }
  return null;
}

/** Counts voices of one cue; a new one starts only while fewer than `max` are playing. */
export class VoiceLimit {
  private playing = 0;
  private readonly max: number;
  constructor(max: number) {
    this.max = max;
  }
  acquire(): boolean {
    if (this.playing >= this.max) return false;
    this.playing++;
    return true;
  }
  release(): void {
    this.playing = Math.max(0, this.playing - 1);
  }
  get count(): number {
    return this.playing;
  }
}
