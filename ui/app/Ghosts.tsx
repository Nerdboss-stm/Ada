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
import { geomShape, poseParts, type Manifest } from "@/lib/replay";
import type { Ghost } from "./useGhosts";

export const ADA = "#c9ced6";
// Follow cam, three y-up world: behind (-x), above, a little to the side; look slightly ahead.
const CAM_OFFSET: Vec3 = [-4, 2.2, 3];
const LOOK_AHEAD: Vec3 = [1.5, 0.3, 0];
const LEADER_RENDER_ORDER = 10;

export type Epoch = RefObject<number | null>;

/** Seconds since the first ghost appeared; every ghost and the camera share it. */
function playbackTime(epoch: Epoch, elapsed: number): number {
  if (epoch.current === null) epoch.current = elapsed;
  return elapsed - epoch.current;
}

function useGeometries(manifest: Manifest): THREE.BufferGeometry[] {
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

/** One recorded run. Poses are raw MuJoCo z-up; the parent group does the only axis rotation. */
function Body({
  ghost,
  geometries,
  color,
  leader,
  cycle,
  epoch,
}: {
  ghost: Ghost;
  geometries: THREE.BufferGeometry[];
  color: string;
  leader: boolean;
  cycle: number;
  epoch: Epoch;
}) {
  const refs = useRef<(THREE.Mesh | null)[]>([]);
  const material = useMemo(() => makeMaterial(leader, color), [leader, color]);
  useEffect(() => () => material.dispose(), [material]);

  useFrame(({ clock }) => {
    const { doc } = ghost;
    const i = sharedFrameIndex(playbackTime(epoch, clock.elapsedTime), doc.fps, doc.frames.length, cycle);
    const geoms = doc.frames[i]?.geoms ?? [];
    for (let g = 0; g < geoms.length; g++) {
      const mesh = refs.current[g];
      if (!mesh) continue;
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
          material={material}
          castShadow={leader}
          receiveShadow={leader}
          renderOrder={leader ? LEADER_RENDER_ORDER : 0}
        />
      ))}
    </group>
  );
}

/** Tracks the leader's torso with lag. The only place mjToThree is applied. */
function FollowCam({ ghost, cycle, epoch }: { ghost: Ghost; cycle: number; epoch: Epoch }) {
  const pos = useRef<Vec3 | null>(null);
  const look = useRef<Vec3 | null>(null);
  const lastIndex = useRef(-1);

  useFrame(({ camera, clock }, delta) => {
    const { doc } = ghost;
    const i = sharedFrameIndex(playbackTime(epoch, clock.elapsedTime), doc.fps, doc.frames.length, cycle);
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
    camera.position.set(...pos.current);
    camera.lookAt(...look.current);
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
          cycle={cycle}
          epoch={epoch}
        />
      ))}
    </group>
  );
}

export function GhostCamera({ ghosts, leader, epoch }: { ghosts: Ghost[]; leader: string | null; epoch: Epoch }) {
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  // No solid leader yet: follow the newest ghost.
  const target = ghosts.find((g) => g.key === leader) ?? ghosts[ghosts.length - 1];
  return target ? <FollowCam ghost={target} cycle={cycle} epoch={epoch} /> : null;
}

