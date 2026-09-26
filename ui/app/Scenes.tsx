"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cheatTimeline, isCheatDoc } from "@/lib/cheat";
import { overTorqueJoint, type CheatCounter, type CheatSpine } from "@/lib/cheatcard";
import { FINALE_CORNER, V0_WALK_LABEL, pinnedGhostSources, pinnedVersion, v0Line, type Pinned } from "@/lib/director";
import { cycleSeconds } from "@/lib/ghosts";
import type { EditDoc } from "@/lib/overlays";
import type { Manifest } from "@/lib/replay";
import { CheatBody, CheatCardDriver, CheatVerdict } from "./Cheat";
import CheatCard from "./CheatCard";
import { Footprints } from "./Footprints";
import { GhostBodies, GhostCamera, type Epoch } from "./Ghosts";
import { Spine } from "./Overlays";
import { Scene } from "./Scene";
import { SoundCues } from "./SoundCues";
import { fetchFrames, useTargetCheat, type CheatTarget, type Ghost } from "./useGhosts";

// The pinned scenes of the C9 director (lib/director.ts). Every id comes from the snapshot; only
// the counters passed in as `live` keep moving. Bodies play recorded frames only.

const INK = "#e6e8ec";
const glass = "rounded-xl border border-white/10 bg-white/[0.06] backdrop-blur-md";

/** The live pieces: api/cheat's spine and gait counter, and the streamed edits. */
export type Live = { spine: CheatSpine | null; counter: CheatCounter | null; edits: EditDoc[] };

/** Recorded frames docs for these (versionId, framesId) pairs, in order; ones from another manifest are left out. */
function usePinnedGhosts(sources: { versionId: string; framesId: string }[], manifest: Manifest): { ghosts: Ghost[]; error: string | null } {
  const [loaded, setLoaded] = useState<Ghost[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    Promise.all(sources.map(async (s) => ({ key: s.versionId, doc: await fetchFrames(`/api/frames/${encodeURIComponent(s.framesId)}`) })))
      .then((all) => live && setLoaded(all))
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [sources]);
  const ghosts = useMemo(() => loaded.filter((g) => g.doc.manifest_version === manifest.manifest_version), [loaded, manifest]);
  const skipped = loaded.length - ghosts.length;
  return { ghosts, error: error ?? (skipped > 0 ? `${skipped} frames doc(s) skipped: manifest_version mismatch` : null) };
}

/** Recorded ghosts on the shared clock, the leader solid with footprints, the follow cam, and step sounds. */
function Walkers({ manifest, ghosts, leader, epoch }: { manifest: Manifest; ghosts: Ghost[]; leader: string | null; epoch: Epoch }) {
  const lead = ghosts.find((g) => g.key === leader);
  const others = useMemo(() => ghosts.filter((g) => g.key !== leader), [ghosts, leader]);
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  return (
    <Scene
      zUp={
        <>
          <GhostBodies manifest={manifest} ghosts={ghosts} leader={leader} epoch={epoch} />
          {lead && <Footprints ghost={lead} manifest={manifest} cycle={cycle} epoch={epoch} />}
        </>
      }
    >
      <GhostCamera ghosts={ghosts} leader={leader} epoch={epoch} cheat={null} />
      <SoundCues manifest={manifest} leader={lead} ghosts={others} cheat={null} cycle={cycle} epoch={epoch} />
    </Scene>
  );
}

function Missing({ text }: { text: string }) {
  return (
    <p className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center font-sans text-lg font-medium tracking-wide text-[#7fb7ff]/80">
      {text}
    </p>
  );
}

export function SceneError({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p className="absolute left-1/2 top-4 z-10 -translate-x-1/2 px-4 text-center text-sm" style={{ color: "#FF4A3D" }}>
      failed: {error}
    </p>
  );
}

/** The best version's showcase walk alone, solid: the backdrop of the spine scenes. */
function useBestSource(p: Pinned): { versionId: string; framesId: string }[] {
  return useMemo(() => pinnedGhostSources(p).filter((g) => g.versionId === p.snapshot.best_version), [p]);
}

/** Scene 1: v0 alone, walking its recorded showcase run, captioned as one walk, with one line (its mean) read from the pin. */
export function V0Scene({ pinned, manifest }: { pinned: Pinned; manifest: Manifest }) {
  const epoch = useRef<number | null>(null);
  const v0 = pinned.snapshot.v0;
  const sources = useMemo(
    () => (v0 && pinned.v0_walk ? [{ versionId: v0, framesId: pinned.v0_walk.frames_id }] : []),
    [v0, pinned.v0_walk],
  );
  const { ghosts, error } = usePinnedGhosts(sources, manifest);
  const line = v0Line(pinned);
  return (
    <div className="relative h-full w-full">
      <Walkers manifest={manifest} ghosts={ghosts} leader={v0} epoch={epoch} />
      {line ? (
        <p
          data-testid="v0-line"
          className="pointer-events-none absolute inset-x-4 top-8 text-center font-sans text-[40px] font-semibold leading-tight tabular-nums max-[800px]:top-4 max-[800px]:text-[22px]"
          style={{ color: INK, textShadow: "0 2px 18px rgba(5,6,10,0.9)" }}
        >
          {line}
          <span data-testid="v0-walk-label" className="mt-2 block text-[18px] font-medium tracking-wide text-[#9aa3b2] max-[800px]:text-[13px]">
            {V0_WALK_LABEL}
          </span>
        </p>
      ) : (
        <Missing text="The pin has no recorded v0 showcase run with train and holdout runs" />
      )}
      <SceneError error={error} />
    </div>
  );
}

const CENTERED =
  "absolute left-1/2 top-[45%] -translate-x-1/2 -translate-y-1/2 min-[1100px]:scale-125 max-[800px]:left-4 max-[800px]:right-4 max-[800px]:top-4 max-[800px]:translate-x-0 max-[800px]:translate-y-0";

/**
 * Scenes 2 and 6: the spine over the best version's walk. 2 outlines the frontier row. 6 is the
 * comparison: harness v0, the frontier and the best, each at its metrics.cost_per_run_usd per
 * gait, the pinned rewrite cost, and the live edit and gait counters.
 */
export function SpineScene({
  pinned,
  manifest,
  live,
  comparison,
}: {
  pinned: Pinned;
  manifest: Manifest;
  live: Live;
  comparison: boolean;
}) {
  const epoch = useRef<number | null>(null);
  const { ghosts, error } = usePinnedGhosts(useBestSource(pinned), manifest);
  const rows = useMemo(
    () => ({
      frontier: pinnedVersion(pinned, pinned.snapshot.frontier),
      best: pinnedVersion(pinned, pinned.snapshot.best_version),
      v0: comparison ? pinnedVersion(pinned, pinned.snapshot.v0) : null,
      rewrite_cost_usd: pinned.rewrite_cost_usd,
    }),
    [pinned, comparison],
  );
  return (
    <div className="relative h-full w-full">
      <Walkers manifest={manifest} ghosts={ghosts} leader={pinned.snapshot.best_version} epoch={epoch} />
      <div className="pointer-events-none absolute inset-0 font-sans">
        <Spine
          versions={[]}
          edits={live.edits}
          spine={live.spine}
          pinned={comparison ? rows : { ...rows, rewrite_cost_usd: null }}
          showCounts={comparison}
          emphasize={comparison ? undefined : "frontier"}
          counter={comparison ? live.counter : null}
          className={CENTERED}
        />
      </div>
      {!rows.frontier && !rows.best && <Missing text="The pin names no frontier or best version with holdout metrics" />}
      <SceneError error={error} />
    </div>
  );
}

/** Scene 4: the pinned cheat in bullet time beside the best version, with its card and the live gait counter. */
export function CheatScene({ pinned, manifest, counter }: { pinned: Pinned; manifest: Manifest; counter: CheatCounter | null }) {
  const epoch = useRef<number | null>(null);
  const { ghosts, error: ghostError } = usePinnedGhosts(useBestSource(pinned), manifest);
  const card = pinned.card;
  const target = useMemo(
    (): CheatTarget | null =>
      card ? { framesId: card.frames_id, reason: card.reason, hotJoint: overTorqueJoint(card.peak_torque)?.joint ?? null } : null,
    [card],
  );
  const { cheat: loaded, error: cheatError } = useTargetCheat(target);
  // A rejected run recorded without a violation frame has no moment to slow down on; say so.
  const [unplayable, setUnplayable] = useState(false);
  useEffect(() => {
    if (!card) return;
    let live = true;
    fetchFrames(`/api/frames/${encodeURIComponent(card.frames_id)}`)
      .then((doc) => live && setUnplayable(!isCheatDoc(doc)))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [card]);
  const cheat = useMemo(() => {
    if (!loaded || loaded.doc.manifest_version !== manifest.manifest_version) return null;
    const timeline = cheatTimeline(loaded.doc.fps, loaded.doc.frames.length, loaded.doc.violation_frame);
    return timeline ? { cheat: loaded, timeline } : null;
  }, [loaded, manifest]);
  const cardRef = useRef<HTMLElement | null>(null);
  const reasonRef = useRef<HTMLElement | null>(null);
  const showCardState = useCallback((cardUp: boolean, reasonUp: boolean) => {
    if (cardRef.current) cardRef.current.style.visibility = cardUp ? "visible" : "hidden";
    if (reasonRef.current) reasonRef.current.style.visibility = reasonUp ? "visible" : "hidden";
  }, []);
  const leader = pinned.snapshot.best_version;
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  const lead = ghosts.find((g) => g.key === leader);
  return (
    <div className="relative h-full w-full">
      <Scene
        zUp={
          <>
            <GhostBodies manifest={manifest} ghosts={ghosts} leader={leader} epoch={epoch} />
            {lead && <Footprints ghost={lead} manifest={manifest} cycle={cycle} epoch={epoch} />}
            {cheat && <CheatBody manifest={manifest} cheat={cheat.cheat} timeline={cheat.timeline} epoch={epoch} />}
          </>
        }
      >
        {cheat && card && <CheatCardDriver timeline={cheat.timeline} epoch={epoch} onState={showCardState} />}
        {cheat && !card && <CheatVerdict cheat={cheat.cheat} timeline={cheat.timeline} epoch={epoch} />}
        <GhostCamera ghosts={ghosts} leader={leader} epoch={epoch} cheat={cheat} />
        <SoundCues manifest={manifest} leader={lead} ghosts={[]} cheat={cheat?.timeline ?? null} cycle={cycle} epoch={epoch} />
      </Scene>
      {cheat && card && <CheatCard card={card} counter={counter} fixture={false} cardRef={cardRef} reasonRef={reasonRef} />}
      {!card && <Missing text="The pin names no over-rated rejected edit with recorded frames" />}
      {card && unplayable && <Missing text={`Edit ${card.edit_id}: its recorded frames have no violation frame, so there is no bullet time to play`} />}
      <SceneError error={ghostError ?? cheatError} />
    </div>
  );
}

/** Scene 7: every pinned ghost, the best solid in front, and the change stream in the corner. */
export function FinaleScene({ pinned, manifest }: { pinned: Pinned; manifest: Manifest }) {
  const epoch = useRef<number | null>(null);
  const sources = useMemo(() => pinnedGhostSources(pinned), [pinned]);
  const { ghosts, error } = usePinnedGhosts(sources, manifest);
  return (
    <div className="relative h-full w-full">
      <Walkers manifest={manifest} ghosts={ghosts} leader={pinned.snapshot.best_version} epoch={epoch} />
      <p
        data-testid="finale-corner"
        className={`pointer-events-none absolute bottom-5 right-6 flex items-center gap-2 px-3 py-1.5 font-sans text-base max-[800px]:bottom-4 max-[800px]:right-4 ${glass}`}
        style={{ color: INK }}
      >
        <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#85BB65" }} />
        {FINALE_CORNER}
      </p>
      {sources.length === 0 && <Missing text="The pin names no ghosts with recorded frames" />}
      <SceneError error={error} />
    </div>
  );
}
