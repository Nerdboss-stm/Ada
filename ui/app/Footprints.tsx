"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { WARM, activeFootprints, footprintEdges } from "@/lib/effects";
import type { Manifest } from "@/lib/replay";
import { playbackTime, type Epoch } from "./Ghosts";
import type { Ghost } from "./useGhosts";

// More than can be live in a 2 s fade at any recorded gait.
const MAX_PRINTS = 64;
const DISC_RADIUS = 0.1;
// Above the meter lines (0.001) and numbers (0.002), in the z-up group.
const DISC_Z = 0.004;
// Peak disc radiance (linear, unclipped): well above the bloom threshold at full alpha.
const PEAK = 6;

/**
 * SPEC §6 "Footprints", leader only: a flat emissive disc under the ankle geom at each recorded
 * contact rising edge, fading linearly over 2 s of the shared loop clock. Additive, so a fade
 * to black is a fade to nothing. Goes inside the z-up group. `recordedTime`, when given, maps
 * playback time to recorded seconds instead of the shared loop (?mode=attempts: 1× or 4×).
 */
export function Footprints({
  ghost,
  manifest,
  cycle,
  epoch,
  recordedTime,
}: {
  ghost: Ghost;
  manifest: Manifest;
  cycle: number;
  epoch: Epoch;
  recordedTime?: (t: number) => number;
}) {
  const mesh = useRef<THREE.InstancedMesh | null>(null);
  const edges = useMemo(() => footprintEdges(ghost.doc, manifest), [ghost, manifest]);
  const geometry = useMemo(() => new THREE.CircleGeometry(DISC_RADIUS, 40), []);
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: "#ffffff",
        toneMapped: false,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );
  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );
  const scratch = useMemo(() => ({ m: new THREE.Matrix4(), c: new THREE.Color(), warm: new THREE.Color(WARM) }), []);

  useFrame(({ clock }) => {
    const im = mesh.current;
    if (!im) return;
    const t = playbackTime(epoch, clock.elapsedTime);
    const tc = recordedTime ? recordedTime(t) : cycle > 0 ? t % cycle : t;
    const live = activeFootprints(edges, ghost.doc.fps, tc).slice(-MAX_PRINTS);
    for (let k = 0; k < live.length; k++) {
      const { edge, alpha } = live[k];
      scratch.m.makeTranslation(edge.x, edge.y, DISC_Z);
      im.setMatrixAt(k, scratch.m);
      im.setColorAt(k, scratch.c.copy(scratch.warm).multiplyScalar(PEAK * alpha));
    }
    im.count = live.length;
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={(el) => {
        mesh.current = el;
        // Allocate instanceColor once; hidden until a contact edge is shown.
        if (el && !el.instanceColor) {
          el.setColorAt(0, new THREE.Color(0, 0, 0));
          el.count = 0;
        }
      }}
      args={[geometry, material, MAX_PRINTS]}
      frustumCulled={false}
      renderOrder={5}
    />
  );
}
