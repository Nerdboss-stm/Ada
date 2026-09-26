// Pure replay helpers: frame indexing, MuJoCo -> three conversions, manifest shapes.
// No three/React imports so this runs under node --test.

/** One geom pose as recorded: position then w-first quaternion (CONTRACTS.md §3). */
export type GeomPose = [number, number, number, number, number, number, number];

export type Frame = {
  geoms: GeomPose[];
  contacts: boolean[];
  forces: number[];
  torso: [number, number, number];
};

/** A `frames` document (CONTRACTS.md §3). */
export type FramesDoc = {
  _id: string;
  run_id: string;
  version_id: string;
  kind: "showcase" | "frontier" | "rejected";
  manifest_version: string;
  fps: number;
  frames: Frame[];
  sha256: string;
  /** First frame that broke a sanity bound; set only on kind "rejected". */
  violation_frame?: number | null;
};

export type ManifestGeom = {
  id: number;
  name: string;
  type: "sphere" | "capsule";
  size: [number, number, number];
  body: string;
};

export type Manifest = {
  manifest_version: string;
  geoms: ManifestGeom[];
  contact_geoms: string[];
  force_joints: string[];
};

export type GeomShape =
  | { kind: "sphere"; radius: number }
  | { kind: "capsule"; radius: number; length: number };

/** Index of the recorded frame shown at playback time t (seconds), looping. No interpolation. */
export function frameIndexAt(t: number, fps: number, n: number): number {
  if (n <= 0 || !(fps > 0) || !Number.isFinite(t) || t < 0) return 0;
  return Math.floor(t * fps) % n;
}

/** MuJoCo quat [w, x, y, z] -> three.js Quaternion order [x, y, z, w]. */
export function mjQuatToXYZW(w: number, x: number, y: number, z: number): [number, number, number, number] {
  return [x, y, z, w];
}

/** Split a recorded pose into position and three-order quaternion. */
export function poseParts(p: GeomPose): { pos: [number, number, number]; quat: [number, number, number, number] } {
  return { pos: [p[0], p[1], p[2]], quat: mjQuatToXYZW(p[3], p[4], p[5], p[6]) };
}

/**
 * MuJoCo sizes: sphere [radius], capsule [radius, half-length] along local z.
 * `length` is the cylinder section only (three's CapsuleGeometry `height`), i.e. 2 × half-length.
 */
export function geomShape(g: ManifestGeom): GeomShape {
  if (g.type === "sphere") return { kind: "sphere", radius: g.size[0] };
  return { kind: "capsule", radius: g.size[0], length: 2 * g.size[1] };
}
