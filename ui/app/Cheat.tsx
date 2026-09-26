"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { CHEAT_RED, LANE_Y, cheatState, cheatTorso, type CheatTimeline } from "@/lib/cheat";
import type { Manifest } from "@/lib/replay";
import { Body, playbackTime, useGeometries, type Epoch } from "./Ghosts";
import type { Cheat } from "./useGhosts";

const NO_FORCES: number[] = [];
// Beside the torso: to its right, or to its left once the torso is on the right of the screen.
const RIGHT_OF = "translate(64px, -50%)";
const LEFT_OF = "translate(calc(-100% - 64px), -50%)";

/**
 * SPEC §6 "The cheat": the rejected run as its own body in red wireframe, in its own lane beside
 * the leader, on its own clock (0.1× through bullet time; lib/cheat.ts). Only recorded frames are
 * shown. Basic material well under the bloom threshold: the cheat never blooms. Goes inside the
 * z-up group.
 */
export function CheatBody({
  manifest,
  cheat,
  timeline,
  epoch,
}: {
  manifest: Manifest;
  cheat: Cheat;
  timeline: CheatTimeline;
  epoch: Epoch;
}) {
  const geometries = useGeometries(manifest);
  const material = useMemo(() => new THREE.MeshBasicMaterial({ color: CHEAT_RED, wireframe: true }), []);
  useEffect(() => () => material.dispose(), [material]);
  const ghost = useMemo(() => ({ key: cheat.key, doc: cheat.doc }), [cheat]);
  return (
    <group position={[0, LANE_Y, 0]}>
      <Body
        ghost={ghost}
        geometries={geometries}
        color={CHEAT_RED}
        leader={false}
        forceIndex={NO_FORCES}
        frameAt={(t) => cheatState(timeline, t).frame}
        epoch={epoch}
        material={material}
      />
    </group>
  );
}

/**
 * The verdict at 36 pt beside the cheat's torso, shown from the violation frame until the camera
 * is back on the leader. Positioned and toggled per frame without a React re-render. Goes outside
 * the z-up group (three y-up world).
 */
export function CheatVerdict({ cheat, timeline, epoch }: { cheat: Cheat; timeline: CheatTimeline; epoch: Epoch }) {
  const anchor = useRef<THREE.Group | null>(null);
  const box = useRef<HTMLDivElement | null>(null);
  const ndc = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ clock, camera }) => {
    const s = cheatState(timeline, playbackTime(epoch, clock.elapsedTime));
    const torso = cheat.doc.frames[s.frame]?.torso;
    if (anchor.current && torso) anchor.current.position.set(...cheatTorso(torso));
    if (!box.current) return;
    box.current.style.visibility = s.verdict && torso ? "visible" : "hidden";
    if (anchor.current && torso) box.current.style.transform = ndc.copy(anchor.current.position).project(camera).x > 0 ? LEFT_OF : RIGHT_OF;
  });
  return (
    <group ref={anchor}>
      <Html zIndexRange={[20, 10]} pointerEvents="none">
        <div
          ref={box}
          data-testid="cheat-verdict"
          className="font-sans font-semibold leading-tight"
          style={{
            visibility: "hidden",
            color: CHEAT_RED,
            fontSize: "36pt",
            width: "max-content",
            maxWidth: "14em",
            transform: RIGHT_OF,
            textShadow: "0 2px 18px rgba(5,6,10,0.9)",
          }}
        >
          {cheat.verdict}
        </div>
      </Html>
    </group>
  );
}
