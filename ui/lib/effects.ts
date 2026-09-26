// Pure helpers for SPEC §6 "Footprints" and "Joint glow" (leader only). Everything here is read
// from recorded frames and the manifest; nothing is interpolated. No three/React imports.
import type { FramesDoc, Manifest } from "./replay";

export const WARM = "#ff8a3d";
/** Footprints fade linearly to nothing over this much playback time. */
export const FOOTPRINT_FADE_S = 2;
/** Emissive intensity at |force| = 1 (recorded forces are torque / gear, clipped to [-1, 1]). */
export const GLOW_MAX = 4;

/**
 * For each manifest geom, the index into `forces` of the joint that drives its body, or -1.
 * By body, per sim/assets/ant.xml: `hip_N` lives in body `aux_N` (the upper leg), and `ankle_N`
 * in the next body down, which is the one holding a contact geom. Manifest geoms are in body
 * tree order, so that child is the next distinct body after `aux_N`. A leg that does not fit
 * this shape gets no glow rather than a guess.
 */
export function legForceIndex(manifest: Manifest): number[] {
  const bodies = [...new Set(manifest.geoms.map((g) => g.body))];
  const contactBodies = new Set(manifest.geoms.filter((g) => manifest.contact_geoms.includes(g.name)).map((g) => g.body));
  const forceOfBody = new Map<string, number>();
  manifest.force_joints.forEach((joint, hip) => {
    const m = /^hip_(\d+)$/.exec(joint);
    if (!m) return;
    const upper = `aux_${m[1]}`;
    const b = bodies.indexOf(upper);
    const lower = b >= 0 ? bodies[b + 1] : undefined;
    const ankle = manifest.force_joints.indexOf(`ankle_${m[1]}`);
    if (lower === undefined || !contactBodies.has(lower) || ankle < 0) return;
    forceOfBody.set(upper, hip);
    forceOfBody.set(lower, ankle);
  });
  return manifest.geoms.map((g) => forceOfBody.get(g.body) ?? -1);
}

/** Emissive intensity for one recorded force; zero force, no glow. */
export function glowIntensity(force: number | undefined): number {
  if (typeof force !== "number" || !Number.isFinite(force)) return 0;
  return GLOW_MAX * Math.min(1, Math.abs(force));
}

/** A contact rising edge: frame `frame` is the first touching frame; x, y are that ankle geom's recorded position. */
export type FootprintEdge = { frame: number; leg: number; x: number; y: number };

/** Every contact false -> true between frames k-1 and k, in frame order. Frame 0 is never an edge. */
export function footprintEdges(doc: Pick<FramesDoc, "frames">, manifest: Manifest): FootprintEdge[] {
  const geomOfLeg = manifest.contact_geoms.map((name) => manifest.geoms.findIndex((g) => g.name === name));
  const out: FootprintEdge[] = [];
  for (let k = 1; k < doc.frames.length; k++) {
    const prev = doc.frames[k - 1].contacts;
    const cur = doc.frames[k].contacts;
    for (let leg = 0; leg < geomOfLeg.length; leg++) {
      if (!cur[leg] || prev[leg]) continue;
      const pose = doc.frames[k].geoms[geomOfLeg[leg]];
      if (geomOfLeg[leg] < 0 || !pose) continue;
      out.push({ frame: k, leg, x: pose[0], y: pose[1] });
    }
  }
  return out;
}

/**
 * Footprints visible at loop time `tc` (seconds since the shared loop last restarted): edges whose
 * frame has been shown, younger than the fade, with alpha 1 -> 0 linearly. Because `tc` restarts
 * with the loop, the trail clears when the loop does.
 */
export function activeFootprints(edges: FootprintEdge[], fps: number, tc: number): { edge: FootprintEdge; alpha: number }[] {
  if (!(fps > 0) || !Number.isFinite(tc) || tc < 0) return [];
  const shown = Math.floor(tc * fps);
  const out: { edge: FootprintEdge; alpha: number }[] = [];
  for (const edge of edges) {
    if (edge.frame > shown) break;
    const alpha = 1 - (tc - edge.frame / fps) / FOOTPRINT_FADE_S;
    if (alpha > 0) out.push({ edge, alpha: Math.min(1, alpha) });
  }
  return out;
}
