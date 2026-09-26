"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import * as THREE from "three";
import {
  LOST,
  STAMP_NOTE,
  WON,
  attemptFrame,
  attemptsOf,
  attemptsSourceUrl,
  attemptsView,
  betText,
  earnedText,
  locate,
  orderAttempts,
  parseAttemptsDoc,
  progress,
  recordedTime,
  rejectedTotal,
  schedule,
  terrainLabel,
  type AttemptsDoc,
  type AttemptsView,
  type Schedule,
  type Scheduled,
} from "@/lib/attempts";
import { pinnedSlots } from "@/lib/director";
import { legForceIndex } from "@/lib/effects";
import type { Vec3 } from "@/lib/ghosts";
import { formatMeters } from "@/lib/overlays";
import type { FramesDoc, Manifest } from "@/lib/replay";
import { Footprints } from "./Footprints";
import { ADA, Body, playbackTime, useGeometries, type Epoch } from "./Ghosts";
import { BG, Scene, useManifest, useSharedCanvas } from "./Scene";
import { fetchFrames } from "./useGhosts";

const INK = "#e6e8ec";
const MUTED = "#8a909b";
const glass = "rounded-xl border border-white/10 bg-white/[0.06] backdrop-blur-md";
const PLEX_WOFF = "/fonts/ibm-plex-sans-500.woff";
// Fixed side camera, three y-up: off the MuJoCo -y side of the track, looking across the first meters.
// Identical for every attempt; it never follows the body.
const CAM_POS: Vec3 = [2.2, 3.6, 10.5];
const CAM_LOOK: Vec3 = [2.2, 0.2, -1.5];
// Floor marks, z-up group: dashes span the track's width; labels sit on the camera side.
const HALF_WIDTH = 6;
const DASH = 0.3;
const DASH_GAP = 0.2;
const MARK_Z = 0.005;
const DROP_FROM = 1.4;
const BET_LABEL_Y = -2.3;
const STAMP_LABEL_Y = -2.9;

type Ready = { status: "ready"; doc: AttemptsDoc; sched: Schedule; frames: Record<string, FramesDoc>; rejected: number };
type LoadState = { status: "loading" } | { status: "waiting" } | { status: "error"; message: string } | Ready;

/**
 * The attempts document (or `?attemptsfixture=`), its play order, and every frames doc that order
 * names. Fetched once. `editIds` (C9 director) are the pinned attempts, played in that order.
 */
function useAttempts(editIds: string[] | undefined): LoadState {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  useEffect(() => {
    let live = true;
    const url = attemptsSourceUrl(window.location.search);
    (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`attempts ${url}: ${res.status}`);
      const doc = parseAttemptsDoc(await res.json());
      if (!doc) throw new Error(`attempts ${url}: missing or malformed fields`);
      const sched = schedule(editIds ? pinnedSlots(attemptsOf(doc), editIds) : orderAttempts(attemptsOf(doc)));
      if (sched.slots.length === 0) return live && setState({ status: "waiting" });
      const ids = [...new Set(sched.slots.map((s) => s.attempt.framesId))];
      const docs = await Promise.all(ids.map((id) => fetchFrames(`/api/frames/${encodeURIComponent(id)}`)));
      const frames = Object.fromEntries(ids.map((id, k) => [id, docs[k]]));
      if (live) setState({ status: "ready", doc, sched, frames, rejected: rejectedTotal(doc.edits) });
    })().catch((e: unknown) => live && setState({ status: "error", message: e instanceof Error ? e.message : String(e) }));
    return () => {
      live = false;
    };
  }, [editIds]);
  return state;
}

function SideCamera() {
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    camera.position.set(...CAM_POS);
    camera.lookAt(...CAM_LOOK);
  }, [camera]);
  return null;
}

/** Local time of slot `index` at playback time t, or null while another slot is on stage. */
function localOf(sched: Schedule, index: number, t: number): number | null {
  const at = locate(sched, t);
  return at.index === index ? at.local : null;
}

/** One Ada on the start mark, walking only her recorded frames at the slot's speed. Inside the z-up group. */
function AttemptWalk({ sched, index, doc, manifest, epoch }: { sched: Schedule; index: number; doc: FramesDoc; manifest: Manifest; epoch: Epoch }) {
  const geometries = useGeometries(manifest);
  const forceIndex = useMemo(() => legForceIndex(manifest), [manifest]);
  const slot = sched.slots[index];
  const ghost = useMemo(() => ({ key: `attempt:${index}:${slot.attempt.framesId}`, doc }), [index, slot, doc]);
  const frameAt = (t: number) => {
    const local = localOf(sched, index, t);
    return local === null ? 0 : attemptFrame(slot.timeline, local, doc.fps, doc.frames.length);
  };
  const recorded = (t: number) => {
    const local = localOf(sched, index, t);
    return local === null ? 0 : recordedTime(slot.timeline, local);
  };
  return (
    <>
      <Body ghost={ghost} geometries={geometries} color={ADA} leader forceIndex={forceIndex} frameAt={frameAt} epoch={epoch} />
      <Footprints ghost={ghost} manifest={manifest} cycle={0} epoch={epoch} recordedTime={recorded} />
    </>
  );
}

function StartMark() {
  return (
    <mesh position={[0, 0, MARK_Z]}>
      <planeGeometry args={[0.05, 3]} />
      <meshBasicMaterial color={INK} toneMapped={false} transparent opacity={0.7} />
    </mesh>
  );
}

function FloorLabel({ x, y, color, text }: { x: number; y: number; color: string; text: string }) {
  return (
    <Text font={PLEX_WOFF} position={[x, y, MARK_Z]} fontSize={0.26} anchorX="center" anchorY="middle">
      {text}
      <meshBasicMaterial color={color} toneMapped={false} />
    </Text>
  );
}

/**
 * The bet (dashed, parent train mean plus predicted_delta_m) drops onto the floor as the typing
 * ends; the candidate's train mean (solid, green or red) appears with the stamp.
 */
function FloorMarks({ sched, index, epoch }: { sched: Schedule; index: number; epoch: Epoch }) {
  const { attempt, timeline: tl } = sched.slots[index];
  const bet = useRef<THREE.Group | null>(null);
  const stamp = useRef<THREE.Group | null>(null);
  const dashes = useMemo(() => {
    const ys: number[] = [];
    for (let y = -HALF_WIDTH; y <= HALF_WIDTH; y += DASH + DASH_GAP) ys.push(y);
    return ys;
  }, []);
  const stampColor = attempt.reached ? WON : LOST;
  useFrame(({ clock }) => {
    const local = localOf(sched, index, playbackTime(epoch, clock.elapsedTime));
    if (bet.current) {
      const shown = local !== null && local >= tl.typeEnd;
      bet.current.visible = shown;
      if (shown) bet.current.position.z = DROP_FROM * (1 - progress(local, tl.typeEnd, tl.intro));
    }
    if (stamp.current) stamp.current.visible = local !== null && local >= tl.walkEnd;
  });
  return (
    <>
      <group ref={bet} position={[attempt.line, 0, DROP_FROM]} visible={false}>
        {dashes.map((y) => (
          <mesh key={y} position={[0, y + DASH / 2, MARK_Z]}>
            <planeGeometry args={[0.05, DASH]} />
            <meshBasicMaterial color={INK} toneMapped={false} />
          </mesh>
        ))}
        <FloorLabel x={0} y={BET_LABEL_Y} color={INK} text={`bet ${formatMeters(attempt.line)}`} />
      </group>
      <group ref={stamp} position={[attempt.candidateMean, 0, 0]} visible={false}>
        <mesh position={[0, 0, MARK_Z]}>
          <planeGeometry args={[0.07, 2 * HALF_WIDTH]} />
          <meshBasicMaterial color={stampColor} toneMapped={false} />
        </mesh>
        <FloorLabel x={0} y={STAMP_LABEL_Y} color={stampColor} text={`${formatMeters(attempt.candidateMean)} ${STAMP_NOTE}`} />
      </group>
    </>
  );
}

/** Pushes the overlay's view to React only when something it shows changes (30 steps/s while animating). */
function AttemptClock({ sched, epoch, onView }: { sched: Schedule; epoch: Epoch; onView: (v: AttemptsView) => void }) {
  const last = useRef("");
  useFrame(({ clock }) => {
    const v = attemptsView(sched, playbackTime(epoch, clock.elapsedTime));
    if (!v) return;
    const animating = !v.holding && (v.phase === "stamp" || v.phase === "file");
    const key = `${v.index}|${v.phase}|${v.typed}|${v.holding}|${v.notebook.length}|${animating ? Math.floor(v.local * 30) : -1}`;
    if (key === last.current) return;
    last.current = key;
    onView(v);
  });
  return null;
}

/** The card's flight at the end of an attempt: kept edits up-left into the notebook, rejected ones down into the tray. */
function filingStyle(kept: boolean, p: number): CSSProperties {
  if (p <= 0) return {};
  const e = p * p;
  return kept
    ? { transform: `translate(${-42 * e}vw, ${18 * e}vh) scale(${1 - 0.55 * e})`, opacity: 1 - e }
    : { transform: `translate(${24 * e}vw, ${62 * e}vh) rotate(${8 * e}deg) scale(${1 - 0.4 * e})`, opacity: 1 - 0.9 * e };
}

function Tag({ tag }: { tag: string }) {
  return (
    <span className="rounded border px-1.5 py-0.5 text-[11px] font-semibold tracking-widest" style={{ borderColor: MUTED, color: INK }}>
      {tag}
    </span>
  );
}

function AttemptCard({ slot, view, count }: { slot: Scheduled; view: AttemptsView; count: number }) {
  const { attempt: a, timeline: tl } = slot;
  const betShown = view.local >= tl.typeEnd;
  const stampP = progress(view.local, tl.walkEnd, tl.walkEnd + Math.min(0.3, tl.stampEnd - tl.walkEnd));
  const fileP = view.phase === "file" ? progress(view.local, tl.stampEnd, tl.end) : 0;
  const color = a.reached ? WON : LOST;
  return (
    <section
      data-testid="attempt-card"
      data-edit={a.edit_id}
      className={`w-[min(760px,100%)] px-6 py-5 max-[800px]:w-full max-[800px]:px-4 max-[800px]:py-3 ${glass}`}
      style={filingStyle(a.kept, fileP)}
    >
      <div className="flex items-center gap-3 text-xs" style={{ color: MUTED }}>
        <Tag tag={a.tag} />
        <span>
          attempt {view.index + 1} of {count}
        </span>
        {slot.fast && <span>replay {tl.speed}×</span>}
      </div>
      <p
        data-testid="attempt-rationale"
        className="mt-3 min-h-[3.2em] text-[22px] leading-snug max-[800px]:text-[20px]"
        style={{ color: INK, textDecorationLine: !a.kept && fileP > 0 ? "line-through" : "none", textDecorationColor: LOST }}
      >
        {a.rationale.slice(0, view.typed)}
        {view.typed < a.rationale.length && <span style={{ color: MUTED }}>▍</span>}
      </p>
      <div className="mt-4 flex flex-wrap items-baseline gap-x-6 gap-y-2 tabular-nums">
        <span data-testid="attempt-bet" className="text-[26px] font-semibold" style={{ color: INK, visibility: betShown ? "visible" : "hidden" }}>
          {betText(a.bet)}
          <span className="ml-2 text-sm font-normal" style={{ color: MUTED }}>
            line at {formatMeters(a.line)}
          </span>
        </span>
        {stampP > 0 && (
          <span
            data-testid="attempt-stamp"
            data-reached={a.reached}
            className="inline-block rounded-lg border-2 px-3 py-1 text-[26px] font-semibold"
            style={{ color, borderColor: color, opacity: stampP, transform: `scale(${1.6 - 0.6 * stampP}) rotate(-3deg)` }}
          >
            {formatMeters(a.candidateMean)}
            <span className="ml-2 text-sm font-normal">{STAMP_NOTE}</span>
          </span>
        )}
      </div>
    </section>
  );
}

function AttemptsOverlay({ sched, view, rejected, fixture }: { sched: Schedule; view: AttemptsView; rejected: number; fixture: boolean }) {
  const slot = sched.slots[view.index];
  const walking = !view.holding && view.phase !== "intro";
  const landing = !view.holding && view.phase === "file" && !slot.attempt.kept;
  return (
    // Below 800 px everything stacks in one column: the card, the walk label, the tray, the notebook.
    <div
      className="pointer-events-none absolute inset-0 font-sans max-[800px]:flex max-[800px]:flex-col max-[800px]:gap-3 max-[800px]:overflow-hidden max-[800px]:p-4"
      style={{ color: INK }}
    >
      <aside
        data-testid="notebook"
        className={`absolute bottom-5 left-5 top-5 flex w-[320px] flex-col px-4 py-4 max-[800px]:static max-[800px]:order-4 max-[800px]:mt-auto max-[800px]:max-h-[30vh] max-[800px]:w-full ${glass}`}
      >
        <h2 className="text-xs uppercase tracking-widest" style={{ color: MUTED }}>
          notebook · kept edits
        </h2>
        <ol className="mt-3 space-y-3 overflow-hidden">
          {view.notebook.map((a) => (
            <li key={a.edit_id} data-edit={a.edit_id} className="text-sm">
              <div className="flex items-center justify-between gap-2">
                <Tag tag={a.tag} />
                <span className="font-semibold tabular-nums">{earnedText(a.measured)}</span>
              </div>
              <p className="mt-1 line-clamp-2 text-xs leading-snug" style={{ color: MUTED }}>
                {a.rationale}
              </p>
            </li>
          ))}
        </ol>
      </aside>
      <div className="absolute left-[360px] right-6 top-6 flex justify-center max-[800px]:static max-[800px]:order-1">
        {!view.holding && <AttemptCard key={`${view.index}:${slot.attempt.edit_id}`} slot={slot} view={view} count={sched.slots.length} />}
        {view.holding && (
          <p className="text-[26px] font-semibold" style={{ color: INK }}>
            notebook complete · {view.notebook.length} kept edits
          </p>
        )}
      </div>
      {walking && (
        <p
          data-testid="walk-label"
          className="absolute bottom-6 left-[360px] right-[240px] text-center text-base max-[800px]:static max-[800px]:order-2"
          style={{ color: MUTED }}
        >
          {terrainLabel(slot.attempt.terrain)} · replay {slot.timeline.speed}×
        </p>
      )}
      <div
        data-testid="tray"
        className={`absolute bottom-5 right-5 w-[200px] px-4 py-3 text-center max-[800px]:static max-[800px]:order-3 max-[800px]:w-full max-[800px]:py-2 ${glass}`}
        style={{ borderColor: landing ? LOST : undefined }}
      >
        <div className="text-[40px] font-semibold leading-none tabular-nums">{rejected}</div>
        <div className="mt-1 text-xs uppercase tracking-widest" style={{ color: MUTED }}>
          rejected edits
        </div>
      </div>
      {fixture && (
        <span className="absolute right-5 top-5 rounded border px-2 py-0.5 text-xs uppercase tracking-widest" style={{ borderColor: MUTED, color: MUTED }}>
          fixture
        </span>
      )}
    </div>
  );
}

/**
 * `?mode=attempts`: the harness's bets, one attempt at a time, in the order orderAttempts
 * computes; the C9 director passes the pinned edit ids instead.
 */
export default function AttemptsStage({ editIds }: { editIds?: string[] } = {}) {
  const { manifest, error: manifestError } = useManifest();
  const state = useAttempts(editIds);
  const shared = useSharedCanvas();
  const epoch = useRef<number | null>(null);
  const [view, setView] = useState<AttemptsView | null>(null);

  const ready = state.status === "ready" ? state : null;
  const mismatch =
    manifest && ready ? Object.values(ready.frames).filter((d) => d.manifest_version !== manifest.manifest_version).length : 0;
  const show = manifest && ready && mismatch === 0 ? { manifest, ready } : null;
  const index = view?.index ?? 0;
  const slot = show ? show.ready.sched.slots[index] : null;
  const error =
    manifestError ??
    (state.status === "error" ? state.message : null) ??
    (mismatch > 0 ? `${mismatch} attempt frames doc(s) skipped: manifest_version mismatch` : null);

  return (
    <div className="relative h-full w-full" style={shared ? undefined : { background: BG }}>
      <Scene
        zUp={
          show &&
          slot && (
            <>
              <StartMark />
              <AttemptWalk
                key={`walk:${index}`}
                sched={show.ready.sched}
                index={index}
                doc={show.ready.frames[slot.attempt.framesId]}
                manifest={show.manifest}
                epoch={epoch}
              />
              <FloorMarks key={`marks:${index}`} sched={show.ready.sched} index={index} epoch={epoch} />
            </>
          )
        }
      >
        <SideCamera />
        {show && <AttemptClock sched={show.ready.sched} epoch={epoch} onView={setView} />}
      </Scene>
      {show && view && <AttemptsOverlay sched={show.ready.sched} view={view} rejected={show.ready.rejected} fixture={!!show.ready.doc.fixture} />}
      {state.status === "waiting" && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center font-sans text-lg font-medium tracking-wide text-[#7fb7ff]/80">
          Waiting for the first edit with a recorded attempt and a bet
        </p>
      )}
      {error && (
        <p className="absolute left-1/2 top-4 -translate-x-1/2 text-sm" style={{ color: LOST }}>
          failed: {error}
        </p>
      )}
    </div>
  );
}
