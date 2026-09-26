"use client";

import { Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer, Text } from "@react-three/drei";
import { Bloom, EffectComposer, ToneMapping } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import * as THREE from "three";
import type { Manifest } from "@/lib/replay";
import { ADA } from "./Ghosts";

// SPEC §6 "Scene".
export const BG = "#05060a";
const FLOOR = "#0b0d12";
const KEY = "#ffd9a8";
const RIM = "#7fb7ff";
const METER_LINE = "#1b2130";
const METER_TEXT = "#7fb7ff";
// Plex Sans Medium for drei Text (troika cannot read next/font's woff2 subsets). Local: no CDN on stage.
const PLEX_WOFF = "/fonts/ibm-plex-sans-500.woff";

// Linear HDR luminance a pixel needs to bloom: above the metal's brightest reflected highlight,
// below the footprints and a pushing leg's glow (lib/effects.ts, Footprints.tsx).
const BLOOM_THRESHOLD = 1.4;

const METERS_FROM = -5;
const METERS_TO = 40;
// MuJoCo is z-up; three is y-up. Everything recorded lives under this group.
const Z_UP_TO_Y_UP: [number, number, number] = [-Math.PI / 2, 0, 0];

function Floor() {
  const meters = useMemo(() => {
    const out: number[] = [];
    for (let m = METERS_FROM; m <= METERS_TO; m++) out.push(m);
    return out;
  }, []);
  return (
    <group>
      <mesh receiveShadow position={[10, 0, 0]}>
        <planeGeometry args={[200, 200]} />
        <meshStandardMaterial color={FLOOR} roughness={0.85} metalness={0.1} />
      </mesh>
      {meters.map((m) => (
        <mesh key={`l${m}`} position={[m, 0, 0.001]}>
          <planeGeometry args={[m % 5 === 0 ? 0.02 : 0.01, 60]} />
          <meshBasicMaterial color={METER_LINE} toneMapped={false} />
        </mesh>
      ))}
      {meters
        .filter((m) => m >= 0)
        .map((m) => (
          <Text
            key={`t${m}`}
            font={PLEX_WOFF}
            position={[m, -1.6, 0.002]}
            fontSize={0.28}
            anchorX="center"
            anchorY="middle"
            characters="0123456789m"
          >
            {`${m} m`}
            <meshBasicMaterial color={METER_TEXT} toneMapped={false} transparent opacity={0.85} />
          </Text>
        ))}
    </group>
  );
}

// Reflections for the metal: key and rim softboxes rendered once into a cube map (no HDR fetch).
function Reflections() {
  return (
    <Environment resolution={256} frames={1}>
      <Lightformer form="rect" color={KEY} intensity={2.5} scale={[6, 3]} position={[4, 6, 5]} target={[0, 0, 0]} />
      <Lightformer form="rect" color={RIM} intensity={2} scale={[8, 1.5]} position={[-5, 2.5, -6]} target={[0, 0, 0]} />
      <Lightformer form="rect" color={ADA} intensity={0.35} scale={[20, 20]} position={[0, 10, 0]} rotation={[Math.PI / 2, 0, 0]} />
    </Environment>
  );
}

/**
 * The composer renders the scene into a HalfFloat target (three applies no tone mapping there)
 * and sets gl.toneMapping to NoToneMapping, so ACES Filmic is applied exactly once, here, after
 * Bloom has added onto the linear HDR image.
 */
function Post() {
  return (
    <EffectComposer>
      <Bloom mipmapBlur luminanceThreshold={BLOOM_THRESHOLD} luminanceSmoothing={0.3} intensity={1.1} />
      <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
    </EffectComposer>
  );
}

function Lights() {
  return (
    <>
      <hemisphereLight args={[RIM, FLOOR, 0.25]} />
      <directionalLight
        color={KEY}
        intensity={3.2}
        position={[4, 7, 5]}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-radius={6}
        shadow-bias={-0.0004}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
        shadow-camera-near={0.5}
        shadow-camera-far={30}
      />
      <directionalLight color={RIM} intensity={2.4} position={[-5, 3, -6]} />
    </>
  );
}

/** manifest.json, loaded once. */
export function useManifest(): { manifest: Manifest | null; error: string | null } {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch("/manifest.json")
      .then(async (m) => {
        if (!m.ok) throw new Error(`load failed: manifest ${m.status}`);
        const mj = (await m.json()) as Manifest;
        if (live) setManifest(mj);
      })
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, []);
  return { manifest, error };
}

/**
 * SPEC §6 "Scene", shared by every mode. `zUp` goes inside the MuJoCo z-up group (recorded
 * poses); `children` go in three's y-up world (cameras, Html anchored to torsos).
 */
export function Scene({ zUp, children }: { zUp: ReactNode; children: ReactNode }) {
  return (
    <Canvas
      shadows="percentage"
      dpr={[1, 2]}
      // ACES comes from the ToneMapping effect in <Post>, never from the renderer.
      gl={{ toneMapping: THREE.NoToneMapping, antialias: true }}
      camera={{ position: [1.2, 2.4, 6.5], fov: 40, near: 0.1, far: 200 }}
    >
      <color attach="background" args={[BG]} />
      <fog attach="fog" args={[BG, 6, 38]} />
      <Lights />
      <Reflections />
      <ContactShadows position={[0, 0.002, 0]} scale={40} resolution={1024} blur={2.4} far={3} opacity={0.75} />
      <group rotation={Z_UP_TO_Y_UP}>
        <Suspense fallback={null}>
          <Floor />
        </Suspense>
        {zUp}
      </group>
      {children}
      <Post />
    </Canvas>
  );
}
