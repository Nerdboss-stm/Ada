"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { frameIndexAt, geomShape, poseParts, type FramesDoc, type Manifest } from "@/lib/replay";

// SPEC §6 "Scene".
const BG = "#05060a";
const FLOOR = "#0b0d12";
const KEY = "#ffd9a8";
const RIM = "#7fb7ff";
const ADA = "#c9ced6";
const METER_LINE = "#1b2130";
const METER_TEXT = "#7fb7ff";
// Plex for drei Text (troika cannot read next/font's woff2 subsets).
const PLEX_WOFF = "https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-sans@5/files/ibm-plex-sans-latin-500-normal.woff";

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

function Ada({ manifest, doc }: { manifest: Manifest; doc: FramesDoc }) {
  const refs = useRef<(THREE.Group | null)[]>([]);
  const clock = useRef(0);
  const shown = useRef(-1);

  useFrame((_, delta) => {
    clock.current += delta;
    const i = frameIndexAt(clock.current, doc.fps, doc.frames.length);
    if (i === shown.current) return;
    shown.current = i;
    const geoms = doc.frames[i].geoms;
    for (let g = 0; g < geoms.length; g++) {
      const obj = refs.current[g];
      if (!obj) continue;
      const { pos, quat } = poseParts(geoms[g]);
      obj.position.set(pos[0], pos[1], pos[2]);
      obj.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
    }
  });

  return (
    <group>
      {manifest.geoms.map((g, k) => {
        const shape = geomShape(g);
        return (
          <group key={g.id} ref={(el) => void (refs.current[k] = el)}>
            {/* three's capsule runs along +Y; MuJoCo's along local +Z. */}
            <mesh castShadow receiveShadow rotation={shape.kind === "capsule" ? [Math.PI / 2, 0, 0] : [0, 0, 0]}>
              {shape.kind === "sphere" ? (
                <sphereGeometry args={[shape.radius, 48, 32]} />
              ) : (
                <capsuleGeometry args={[shape.radius, shape.length, 12, 24]} />
              )}
              <meshPhysicalMaterial color={ADA} metalness={0.9} roughness={0.25} clearcoat={1} />
            </mesh>
          </group>
        );
      })}
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

export default function Stage({ framesUrl = "/fixtures/wiggle.json" }: { framesUrl?: string }) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [doc, setDoc] = useState<FramesDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([fetch("/manifest.json"), fetch(framesUrl)])
      .then(async ([m, f]) => {
        if (!m.ok || !f.ok) throw new Error(`load failed: manifest ${m.status}, frames ${f.status}`);
        const [mj, fj] = (await Promise.all([m.json(), f.json()])) as [Manifest, FramesDoc];
        if (fj.manifest_version !== mj.manifest_version) throw new Error("frames manifest_version does not match manifest");
        if (live) {
          setManifest(mj);
          setDoc(fj);
        }
      })
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [framesUrl]);

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
          {manifest && doc && <Ada manifest={manifest} doc={doc} />}
        </group>
        <OrbitControls makeDefault target={[1.2, 0.4, -1.5]} enableDamping />
      </Canvas>
      {error && <p className="absolute left-4 top-4 text-sm text-red-400">{error}</p>}
    </div>
  );
}
