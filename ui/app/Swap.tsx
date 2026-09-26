"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { cycleSeconds, followStep, FOLLOW_ALPHA, sharedFrameIndex, type Vec3 } from "@/lib/ghosts";
import { legForceIndex } from "@/lib/effects";
import type { FramesDoc, Manifest } from "@/lib/replay";
import {
  SWAP_LANE_Y,
  laneTorso,
  parseSwap,
  swapCamera,
  swapDiff,
  swapHeadline,
  swapSourceUrl,
  swapSubline,
  type SwapDoc,
} from "@/lib/swap";
import { Footprints } from "./Footprints";
import { ADA, Body, playbackTime, useGeometries, type Epoch } from "./Ghosts";
import { BG, Scene, useManifest } from "./Scene";
import { SoundCues } from "./SoundCues";
import { fetchFrames, type Ghost } from "./useGhosts";

const NO_GHOSTS: Ghost[] = [];
const INK = "#e6e8ec";
const MUTED = "#8a909b";
const FAIL = "#FF4A3D";
const glass = "rounded-xl border border-white/10 bg-white/[0.06] backdrop-blur-md";

/** One lane: a recorded run, its side, and its MuJoCo +y offset. */
export type Lane = { side: "left" | "right"; ghost: Ghost; laneY: number };

type SwapState =
  | { status: "loading" }
  | { status: "waiting" }
  | { status: "error"; message: string }
  | { status: "ready"; swap: SwapDoc; left: FramesDoc; right: FramesDoc };

/** The newest swaps document (or `?swapfixture=`), then both recorded runs it names. */
function useSwap(): SwapState {
  const [state, setState] = useState<SwapState>({ status: "loading" });
  useEffect(() => {
    let live = true;
    const url = swapSourceUrl(window.location.search);
    (async () => {
      const res = await fetch(url);
      if (res.status === 404 && url.startsWith("/api/")) return live && setState({ status: "waiting" });
      if (!res.ok) throw new Error(`swap ${url}: ${res.status}`);
      const swap = parseSwap(await res.json());
      if (!swap) throw new Error(`swap ${url}: missing or malformed fields`);
      const [left, right] = await Promise.all([
        fetchFrames(`/api/frames/${encodeURIComponent(swap.left_frames_id)}`),
        fetchFrames(`/api/frames/${encodeURIComponent(swap.right_frames_id)}`),
      ]);
      if (live) setState({ status: "ready", swap, left, right });
    })().catch((e: unknown) => live && setState({ status: "error", message: e instanceof Error ? e.message : String(e) }));
    return () => {
      live = false;
    };
  }, []);
  return state;
}

/** Both lanes solid (the leader's metal and joint glow), same start line, same shared clock. Inside the z-up group. */
function SwapBodies({ manifest, lanes, cycle, epoch }: { manifest: Manifest; lanes: Lane[]; cycle: number; epoch: Epoch }) {
  const geometries = useGeometries(manifest);
  const forceIndex = useMemo(() => legForceIndex(manifest), [manifest]);
  return (
    <>
      {lanes.map(({ side, ghost, laneY }) => (
        <group key={side} position={[0, laneY, 0]}>
          <Body
            ghost={ghost}
            geometries={geometries}
            color={ADA}
            leader
            forceIndex={forceIndex}
            frameAt={(t) => sharedFrameIndex(t, ghost.doc.fps, ghost.doc.frames.length, cycle)}
            epoch={epoch}
          />
          <Footprints ghost={ghost} manifest={manifest} cycle={cycle} epoch={epoch} />
        </group>
      ))}
    </>
  );
}

/** One camera framing both recorded torsos, with the follow cam's lag; cuts when the loop restarts. */
function SwapCamera({ lanes, cycle, epoch }: { lanes: Lane[]; cycle: number; epoch: Epoch }) {
  const pos = useRef<Vec3 | null>(null);
  const look = useRef<Vec3 | null>(null);
  const lastT = useRef(-1);
  useFrame(({ camera, clock }, delta) => {
    const t = playbackTime(epoch, clock.elapsedTime);
    const torsos = lanes.map(({ ghost, laneY }) => {
      const { fps, frames } = ghost.doc;
      const frame = frames[sharedFrameIndex(t, fps, frames.length, cycle)];
      return frame ? laneTorso(frame.torso, laneY) : null;
    });
    if (!torsos[0] || !torsos[1]) return;
    const want = swapCamera(torsos[0], torsos[1]);
    const tc = cycle > 0 ? t % cycle : t;
    const cut = pos.current === null || look.current === null || tc < lastT.current;
    lastT.current = tc;
    pos.current = cut ? want.pos : followStep(pos.current as Vec3, want.pos, FOLLOW_ALPHA, delta);
    look.current = cut ? want.look : followStep(look.current as Vec3, want.look, FOLLOW_ALPHA, delta);
    camera.position.set(...pos.current);
    camera.lookAt(...look.current);
  });
  return null;
}

function SwapOverlay({ swap }: { swap: SwapDoc }) {
  const [open, setOpen] = useState(false);
  const rows = useMemo(() => swapDiff(swap), [swap]);
  return (
    <div className="pointer-events-none absolute inset-0 font-sans" style={{ color: INK }}>
      <header className="absolute inset-x-0 top-5 flex flex-col items-center px-4 text-center">
        <h1 data-testid="swap-headline" className="font-semibold leading-none tabular-nums" style={{ fontSize: "72pt" }}>
          {swapHeadline(swap)}
        </h1>
        <p data-testid="swap-subline" className="mt-3" style={{ fontSize: "24pt", color: MUTED }}>
          {swapSubline(swap.captured_at)}
        </p>
        {swap.fixture && (
          <span className="mt-2 rounded border px-2 py-0.5 text-xs uppercase tracking-widest" style={{ borderColor: MUTED, color: MUTED }}>
            fixture
          </span>
        )}
      </header>
      <section
        data-testid="swap-diff"
        className={`pointer-events-auto absolute bottom-5 left-1/2 w-[min(780px,calc(100vw-2rem))] -translate-x-1/2 px-5 py-4 text-sm ${glass}`}
      >
        <div className="grid grid-cols-[8rem_1fr_1fr_5.5rem] gap-x-4 gap-y-1.5">
          <span />
          <span className="text-xs" style={{ color: MUTED }}>
            left · {swap.left_version}
          </span>
          <span className="text-xs" style={{ color: MUTED }}>
            right · {swap.right_version}
          </span>
          <span />
          {rows.map((r) =>
            r.identical ? (
              <div key={r.label} className="contents" data-row={r.label}>
                <span style={{ color: MUTED }}>{r.label}</span>
                <span className="truncate tabular-nums" title={r.left}>
                  {r.left}
                </span>
                <span className="truncate tabular-nums" title={r.right}>
                  {r.right}
                </span>
                <span style={{ color: MUTED }}>identical</span>
              </div>
            ) : (
              <div key={r.label} className="contents" data-row={r.label}>
                <button
                  type="button"
                  className="cursor-pointer text-left"
                  style={{ color: MUTED }}
                  aria-expanded={open}
                  onClick={() => setOpen((o) => !o)}
                >
                  {open ? "▾" : "▸"} {r.label}
                </button>
                <span className="truncate">{r.left}</span>
                <span className="truncate">{r.right}</span>
                <span style={{ color: INK }}>{r.lines?.length ?? 0} changed</span>
                {open && (
                  <ul data-testid="harness-diff" className="col-span-4 mt-1 space-y-0.5 font-mono text-[13px] leading-snug">
                    {(r.lines ?? []).map((line, i) => (
                      <li key={i} className="whitespace-pre-wrap" style={{ color: line.startsWith("-") ? MUTED : INK }}>
                        {line}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ),
          )}
        </div>
      </section>
    </div>
  );
}

/** `?mode=swap`: same model, same holdout task; harness v0 left, the best harness right. */
export default function SwapStage() {
  const { manifest, error: manifestError } = useManifest();
  const state = useSwap();
  const epoch = useRef<number | null>(null);

  const lanes = useMemo<Lane[]>(() => {
    if (state.status !== "ready") return [];
    return [
      { side: "left", ghost: { key: `left:${state.swap.left_frames_id}`, doc: state.left }, laneY: SWAP_LANE_Y.left },
      { side: "right", ghost: { key: `right:${state.swap.right_frames_id}`, doc: state.right }, laneY: SWAP_LANE_Y.right },
    ];
  }, [state]);
  const mismatch = manifest && lanes.some((l) => l.ghost.doc.manifest_version !== manifest.manifest_version);
  const cycle = useMemo(() => cycleSeconds(lanes.map((l) => l.ghost.doc)), [lanes]);
  const show = manifest && lanes.length === 2 && !mismatch;
  const error =
    manifestError ?? (state.status === "error" ? state.message : null) ?? (mismatch ? "swap frames skipped: manifest_version mismatch" : null);

  return (
    <div className="relative h-dvh w-full" style={{ background: BG }}>
      <Scene zUp={show && <SwapBodies manifest={manifest} lanes={lanes} cycle={cycle} epoch={epoch} />}>
        {show && <SwapCamera lanes={lanes} cycle={cycle} epoch={epoch} />}
        {/* No ghosts and no cheat in the swap: steps follow the best harness (right lane). */}
        {show && <SoundCues manifest={manifest} leader={lanes[1].ghost} ghosts={NO_GHOSTS} cheat={null} cycle={cycle} epoch={epoch} />}
      </Scene>
      {state.status === "ready" && <SwapOverlay swap={state.swap} />}
      {state.status === "waiting" && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center font-sans text-lg font-medium tracking-wide text-[#7fb7ff]/80">
          Waiting for the first recorded swap
        </p>
      )}
      {error && (
        <p className="absolute left-1/2 top-4 -translate-x-1/2 text-sm" style={{ color: FAIL }}>
          failed: {error}
        </p>
      )}
    </div>
  );
}
