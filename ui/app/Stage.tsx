"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import type { Manifest } from "@/lib/replay";
import { ADA, GhostBodies, GhostCamera } from "./Ghosts";
import { useGhosts } from "./useGhosts";

// SPEC §6 "Scene".
const BG = "#05060a";
const FLOOR = "#0b0d12";
const KEY = "#ffd9a8";
const RIM = "#7fb7ff";
const METER_LINE = "#1b2130";
const METER_TEXT = "#7fb7ff";
// Plex Sans Medium for drei Text (troika cannot read next/font's woff2 subsets). Local: no CDN on stage.
const PLEX_WOFF = "/fonts/ibm-plex-sans-500.woff";

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

export default function Stage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const { ghosts: loaded, leader, error: ghostError } = useGhosts();
  const epoch = useRef<number | null>(null);
  const orbit = useMemo(() => new URLSearchParams(window.location.search).has("orbit"), []);

  useEffect(() => {
    let live = true;
    fetch("/manifest.json")
      .then(async (m) => {
        if (!m.ok) throw new Error(`load failed: manifest ${m.status}`);
        const mj = (await m.json()) as Manifest;
        if (live) setManifest(mj);
      })
      .catch((e: unknown) => live && setLoadError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, []);

  // A frames doc recorded against another manifest cannot be posed; leave it out and say so.
  const ghosts = useMemo(
    () => (manifest ? loaded.filter((g) => g.doc.manifest_version === manifest.manifest_version) : []),
    [loaded, manifest],
  );
  const mismatched = manifest ? loaded.length - ghosts.length : 0;
  const error =
    loadError ?? ghostError ?? (mismatched > 0 ? `${mismatched} frames doc(s) skipped: manifest_version mismatch` : null);

  return (
    <div className="relative h-dvh w-full" style={{ background: BG }}>
      <Canvas
        shadows="percentage"
        dpr={[1, 2]}
        gl={{ toneMapping: THREE.ACESFilmicToneMapping, antialias: true }}
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
          {manifest && <GhostBodies manifest={manifest} ghosts={ghosts} leader={leader} epoch={epoch} />}
        </group>
        {orbit ? (
          <OrbitControls makeDefault target={[1.2, 0.4, -1.5]} enableDamping />
        ) : (
          <GhostCamera ghosts={ghosts} leader={leader} epoch={epoch} />
        )}
      </Canvas>
      {ghosts.length === 0 && !error && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center font-sans text-lg font-medium tracking-wide text-[#7fb7ff]/80">
          Waiting for Ada&rsquo;s first recorded run
        </p>
      )}
      {error && <p className="absolute left-4 top-4 text-sm text-red-400">{error}</p>}
    </div>
  );
}
