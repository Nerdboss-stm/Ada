"use client";

import { useEffect, useMemo, useRef, type RefObject } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  FOLLOW_ALPHA,
  GHOST_OPACITY,
  cycleSeconds,
  followStep,
  ghostColor,
  mjToThree,
  sharedFrameIndex,
  type Vec3,
} from "@/lib/ghosts";
import { cheatState, cheatTorso, orbitCamera, type CheatTimeline } from "@/lib/cheat";
import { HOT_EMISSIVE_LINEAR, HOT_GLOW_MIN, WARM, glowIntensity, hotGlowIntensity, legForceIndex } from "@/lib/effects";
import { formatMeters } from "@/lib/overlays";
import { geomShape, poseParts, type Manifest } from "@/lib/replay";
import type { Cheat, Ghost } from "./useGhosts";

export const ADA = "#c9ced6";
// Follow cam, three y-up world: behind (-x), above, a little to the side; look slightly ahead.
const CAM_OFFSET: Vec3 = [-4, 2.2, 3];
const LOOK_AHEAD: Vec3 = [1.5, 0.3, 0];
const LEADER_RENDER_ORDER = 10;

export type Epoch = RefObject<number | null>;

/** Seconds since the first ghost appeared; every ghost, the camera, and the footprints share it. */
export function playbackTime(epoch: Epoch, elapsed: number): number {
  if (epoch.current === null) epoch.current = elapsed;
  return elapsed - epoch.current;
}

export function useGeometries(manifest: Manifest): THREE.BufferGeometry[] {
  const geometries = useMemo(
    () =>
      manifest.geoms.map((g) => {
        const shape = geomShape(g);
        if (shape.kind === "sphere") return new THREE.SphereGeometry(shape.radius, 48, 32);
        // three's capsule runs along +Y; MuJoCo's along local +Z.
        return new THREE.CapsuleGeometry(shape.radius, shape.length, 12, 24).rotateX(Math.PI / 2);
      }),
    [manifest],
  );
  useEffect(() => () => geometries.forEach((g) => g.dispose()), [geometries]);
  return geometries;
}

function makeMaterial(leader: boolean, color: string): THREE.Material {
  if (leader) {
    // Solid; in the transparent list only so renderOrder can draw it after the ghosts.
    return new THREE.MeshPhysicalMaterial({
      color: ADA,
      metalness: 0.9,
      roughness: 0.25,
      clearcoat: 1,
      transparent: true,
      opacity: 1,
      depthWrite: true,
    });
  }
  return new THREE.MeshStandardMaterial({
    color,
    metalness: 0.3,
    roughness: 0.4,
    transparent: true,
    opacity: GHOST_OPACITY,
    depthWrite: false,
  });
}

/**
 * SPEC §6 "Joint glow": the leader's metal plus a warm emissive whose intensity is set each
 * frame from the recorded force. Emissive above the bloom threshold is what blooms.
 */
function makeGlowMaterial(): THREE.MeshPhysicalMaterial {
  const m = makeMaterial(true, ADA) as THREE.MeshPhysicalMaterial;
  m.emissive = new THREE.Color(WARM);
  m.emissiveIntensity = 0;
  m.toneMapped = false;
  return m;
}

/** The over-torque joint during bullet time: solid, red emissive from the recorded force (lib/effects.ts). */
function makeHotMaterial(): THREE.MeshStandardMaterial {
  const m = new THREE.MeshStandardMaterial({ color: "#000000", metalness: 0, roughness: 1, toneMapped: false });
  m.emissive.setRGB(...HOT_EMISSIVE_LINEAR);
  m.emissiveIntensity = HOT_GLOW_MIN;
  return m;
}

/** Geoms driven by one joint (`index` per geom: force index or -1) glow red while `active(t)`. */
export type HotJoint = { index: number[]; active: (t: number) => boolean };

/**
 * One recorded run. Poses are raw MuJoCo z-up; the parent group does the only axis rotation.
 * `frameAt` maps shared playback time to the recorded frame shown; `material`, when given,
 * replaces the ghost/leader material (the cheat's red wireframe). `hot`, when given, swaps the
 * over-torque joint's geoms to a red glow while it is active.
 */
export function Body({
  ghost,
  geometries,
  color,
  leader,
  forceIndex,
  frameAt,
  epoch,
  material: override,
  hot,
}: {
  ghost: Ghost;
  geometries: THREE.BufferGeometry[];
  color: string;
  leader: boolean;
  forceIndex: number[];
  frameAt: (t: number) => number;
  epoch: Epoch;
  material?: THREE.Material;
  hot?: HotJoint;
}) {
  const refs = useRef<(THREE.Mesh | null)[]>([]);
  const own = useMemo(() => (override ? null : makeMaterial(leader, color)), [override, leader, color]);
  useEffect(() => () => own?.dispose(), [own]);
  const material = override ?? (own as THREE.Material);
  // Leader only, never ghosts: one glow material per driven leg geom, null elsewhere.
  const glow = useMemo(
    () => geometries.map((_, k) => (leader && (forceIndex[k] ?? -1) >= 0 ? makeGlowMaterial() : null)),
    [geometries, leader, forceIndex],
  );
  useEffect(() => () => glow.forEach((m) => m?.dispose()), [glow]);
  const hotMats = useMemo(() => geometries.map((_, k) => (hot && (hot.index[k] ?? -1) >= 0 ? makeHotMaterial() : null)), [geometries, hot]);
  useEffect(() => () => hotMats.forEach((m) => m?.dispose()), [hotMats]);

  useFrame(({ clock }) => {
    const t = playbackTime(epoch, clock.elapsedTime);
    const frame = ghost.doc.frames[frameAt(t)];
    const hotNow = hot ? hot.active(t) : false;
    const geoms = frame?.geoms ?? [];
    for (let g = 0; g < geoms.length; g++) {
      const mesh = refs.current[g];
      if (!mesh) continue;
      if (leader && forceIndex[g] >= 0) {
        (mesh.material as THREE.MeshPhysicalMaterial).emissiveIntensity = glowIntensity(frame.forces[forceIndex[g]]);
      }
      const hm = hotMats[g];
      if (hm && hot) {
        mesh.material = hotNow ? hm : material;
        if (hotNow) (mesh.material as THREE.MeshStandardMaterial).emissiveIntensity = hotGlowIntensity(frame.forces[hot.index[g]]);
      }
      const { pos, quat } = poseParts(geoms[g]);
      mesh.position.set(pos[0], pos[1], pos[2]);
      mesh.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
    }
  });

  return (
    <group>
      {geometries.map((geometry, k) => (
        <mesh
          key={k}
          ref={(el) => void (refs.current[k] = el)}
          geometry={geometry}
          material={glow[k] ?? material}
          castShadow={leader}
          receiveShadow={leader}
          renderOrder={leader ? LEADER_RENDER_ORDER : 0}
        />
      ))}
    </group>
  );
}

/** Orbit center lag per 60 Hz frame: bullet time holds each recorded torso for 0.3 s; the camera glides between them. */
const ORBIT_CENTER_ALPHA = 0.08;

const mix = (a: Vec3, b: Vec3, k: number): Vec3 => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];

/**
 * Tracks the leader's torso with lag. During the cheat's bullet time it blends onto an orbit
 * around the cheat's torso and back (lib/cheat.ts cheatState.camWeight); the leader follow keeps
 * running underneath, so weight 0 is exactly back on the leader.
 */
function FollowCam({
  ghost,
  cycle,
  epoch,
  cheat,
}: {
  ghost: Ghost;
  cycle: number;
  epoch: Epoch;
  cheat: { cheat: Cheat; timeline: CheatTimeline } | null;
}) {
  const pos = useRef<Vec3 | null>(null);
  const look = useRef<Vec3 | null>(null);
  const lastIndex = useRef(-1);
  const center = useRef<Vec3 | null>(null);

  useFrame(({ camera, clock }, delta) => {
    const { doc } = ghost;
    const t = playbackTime(epoch, clock.elapsedTime);
    const i = sharedFrameIndex(t, doc.fps, doc.frames.length, cycle);
    const frame = doc.frames[i];
    if (!frame) return;
    const torso = mjToThree(frame.torso);
    const wantPos: Vec3 = [torso[0] + CAM_OFFSET[0], torso[1] + CAM_OFFSET[1], torso[2] + CAM_OFFSET[2]];
    const wantLook: Vec3 = [torso[0] + LOOK_AHEAD[0], torso[1] + LOOK_AHEAD[1], torso[2] + LOOK_AHEAD[2]];
    // Cut, don't swoop, when the shared loop restarts.
    const cut = pos.current === null || look.current === null || i < lastIndex.current;
    lastIndex.current = i;
    pos.current = cut ? wantPos : followStep(pos.current as Vec3, wantPos, FOLLOW_ALPHA, delta);
    look.current = cut ? wantLook : followStep(look.current as Vec3, wantLook, FOLLOW_ALPHA, delta);
    let camPos = pos.current;
    let camLook = look.current;
    const s = cheat ? cheatState(cheat.timeline, t) : null;
    const cheatAt = s && cheat ? cheat.cheat.doc.frames[s.frame]?.torso : undefined;
    if (s && cheatAt && s.camWeight > 0) {
      const want = cheatTorso(cheatAt);
      center.current = center.current === null ? want : followStep(center.current, want, ORBIT_CENTER_ALPHA, delta);
      const orbit = orbitCamera(center.current, s.orbit);
      camPos = mix(camPos, orbit.pos, s.camWeight);
      camLook = mix(camLook, orbit.look, s.camWeight);
    } else {
      center.current = null;
    }
    camera.position.set(...camPos);
    camera.lookAt(...camLook);
  });
  return null;
}

/** Bodies go inside the z-up -> y-up group; the camera goes outside it. */
export function GhostBodies({
  manifest,
  ghosts,
  leader,
  epoch,
}: {
  manifest: Manifest;
  ghosts: Ghost[];
  leader: string | null;
  epoch: Epoch;
}) {
  const geometries = useGeometries(manifest);
  const forceIndex = useMemo(() => legForceIndex(manifest), [manifest]);
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  // Oldest first; the leader is drawn last.
  const ordered = [...ghosts.filter((g) => g.key !== leader), ...ghosts.filter((g) => g.key === leader)];
  const colorOf = new Map(ghosts.map((g, i) => [g.key, ghostColor(i, ghosts.length)]));
  return (
    <group>
      {ordered.map((g) => (
        <Body
          key={g.key}
          ghost={g}
          geometries={geometries}
          color={colorOf.get(g.key) ?? ADA}
          leader={g.key === leader}
          forceIndex={forceIndex}
          frameAt={(t) => sharedFrameIndex(t, g.doc.fps, g.doc.frames.length, cycle)}
          epoch={epoch}
        />
      ))}
    </group>
  );
}

/**
 * Reports the leader's recorded torso x at the shown frame (no React re-render, no
 * interpolation): the same shared clock and frame index the bodies use. Called on change only.
 */
export function LeaderReadout({
  ghosts,
  leader,
  epoch,
  onText,
}: {
  ghosts: Ghost[];
  leader: string | null;
  epoch: Epoch;
  onText: (text: string) => void;
}) {
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  const ghost = ghosts.find((g) => g.key === leader);
  const last = useRef<string | null>(null);
  useFrame(({ clock }) => {
    if (!ghost) return;
    const { doc } = ghost;
    const i = sharedFrameIndex(playbackTime(epoch, clock.elapsedTime), doc.fps, doc.frames.length, cycle);
    const frame = doc.frames[i];
    if (!frame) return;
    const text = formatMeters(frame.torso[0]);
    if (text === last.current) return;
    last.current = text;
    onText(text);
  });
  return null;
}

export function GhostCamera({
  ghosts,
  leader,
  epoch,
  cheat,
}: {
  ghosts: Ghost[];
  leader: string | null;
  epoch: Epoch;
  cheat: { cheat: Cheat; timeline: CheatTimeline } | null;
}) {
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  // No solid leader yet: follow the newest ghost.
  const target = ghosts.find((g) => g.key === leader) ?? ghosts[ghosts.length - 1];
  return target ? <FollowCam ghost={target} cycle={cycle} epoch={epoch} cheat={cheat} /> : null;
}

