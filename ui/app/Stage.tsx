"use client";

import { useCallback, useMemo, useRef } from "react";
import { OrbitControls } from "@react-three/drei";
import { cheatTimeline } from "@/lib/cheat";
import { cycleSeconds } from "@/lib/ghosts";
import { CheatBody, CheatCardDriver, CheatVerdict } from "./Cheat";
import CheatCard from "./CheatCard";
import { Footprints } from "./Footprints";
import { GhostBodies, GhostCamera, LeaderReadout } from "./Ghosts";
import Overlays from "./Overlays";
import { BG, Scene, useManifest } from "./Scene";
import { SoundCues } from "./SoundCues";
import AttemptsStage from "./Attempts";
import SwapStage from "./Swap";
import { useAdaStream } from "./useAdaStream";
import { useCheat } from "./useCheat";
import { useGhosts } from "./useGhosts";

/** One route (SPEC §6): `?mode=swap` is v0 vs the best harness; `?mode=attempts` is the harness's bets; everything else is the demo. */
export default function Stage() {
  const mode = useMemo(() => new URLSearchParams(window.location.search).get("mode"), []);
  if (mode === "swap") return <SwapStage />;
  if (mode === "attempts") return <AttemptsStage />;
  return <DemoStage />;
}

function DemoStage() {
  const { manifest, error: loadError } = useManifest();
  const stream = useAdaStream();
  const cheatData = useCheat(stream.edits);
  const card = cheatData.doc?.card ?? null;
  const { ghosts: loaded, leader, cheat: loadedCheat, error: ghostError } = useGhosts(stream.versions, stream.edits, stream.error, card);
  const cardRef = useRef<HTMLElement | null>(null);
  const reasonRef = useRef<HTMLElement | null>(null);
  // Written straight to the DOM when bullet time starts and ends; React never re-renders for it.
  const showCardState = useCallback((cardUp: boolean, reasonUp: boolean) => {
    if (cardRef.current) cardRef.current.style.visibility = cardUp ? "visible" : "hidden";
    if (reasonRef.current) reasonRef.current.style.visibility = reasonUp ? "visible" : "hidden";
  }, []);
  const epoch = useRef<number | null>(null);
  const distanceRef = useRef<HTMLSpanElement | null>(null);
  // Written straight to the DOM every recorded frame; React never re-renders for it.
  const showDistance = useCallback((text: string) => {
    if (distanceRef.current) distanceRef.current.textContent = text;
  }, []);
  const orbit = useMemo(() => new URLSearchParams(window.location.search).has("orbit"), []);

  // A frames doc recorded against another manifest cannot be posed; leave it out and say so.
  const ghosts = useMemo(
    () => (manifest ? loaded.filter((g) => g.doc.manifest_version === manifest.manifest_version) : []),
    [loaded, manifest],
  );
  // The cheat plays on its own clock; the ghosts' shared loop never waits for it.
  const cheat = useMemo(() => {
    if (!manifest || !loadedCheat || loadedCheat.doc.manifest_version !== manifest.manifest_version) return null;
    const { fps, frames, violation_frame } = loadedCheat.doc;
    const timeline = cheatTimeline(fps, frames.length, violation_frame);
    return timeline ? { cheat: loadedCheat, timeline } : null;
  }, [manifest, loadedCheat]);
  const leaderGhost = ghosts.find((g) => g.key === leader);
  const cycle = useMemo(() => cycleSeconds(ghosts.map((g) => g.doc)), [ghosts]);
  const others = useMemo(() => ghosts.filter((g) => g.key !== leader), [ghosts, leader]);
  const mismatched = manifest ? loaded.length - ghosts.length + (loadedCheat && !cheat ? 1 : 0) : 0;
  // The card belongs to the cheat that is playing: its edit's frames, loaded and playable.
  const showCard = card !== null && cheat !== null && cheat.cheat.key === `cheat:${card.frames_id}`;
  const error =
    loadError ?? ghostError ?? stream.error ?? cheatData.error ?? (mismatched > 0 ? `${mismatched} frames doc(s) skipped: manifest_version mismatch` : null);

  return (
    <div className="relative h-dvh w-full" style={{ background: BG }}>
      <Scene
        zUp={
          <>
            {manifest && <GhostBodies manifest={manifest} ghosts={ghosts} leader={leader} epoch={epoch} />}
            {manifest && leaderGhost && <Footprints ghost={leaderGhost} manifest={manifest} cycle={cycle} epoch={epoch} />}
            {manifest && cheat && <CheatBody manifest={manifest} cheat={cheat.cheat} timeline={cheat.timeline} epoch={epoch} />}
          </>
        }
      >
        {cheat && !showCard && <CheatVerdict cheat={cheat.cheat} timeline={cheat.timeline} epoch={epoch} />}
        {cheat && showCard && <CheatCardDriver timeline={cheat.timeline} epoch={epoch} onState={showCardState} />}
        {orbit ? (
          <OrbitControls makeDefault target={[1.2, 0.4, -1.5]} enableDamping />
        ) : (
          <GhostCamera ghosts={ghosts} leader={leader} epoch={epoch} cheat={cheat} />
        )}
        <LeaderReadout ghosts={ghosts} leader={leader} epoch={epoch} onText={showDistance} />
        {manifest && (
          <SoundCues manifest={manifest} leader={leaderGhost} ghosts={others} cheat={cheat?.timeline ?? null} cycle={cycle} epoch={epoch} />
        )}
      </Scene>
      <Overlays
        leaderVersion={leaderGhost?.doc.version_id ?? null}
        distanceRef={distanceRef}
        versions={stream.versions}
        edits={stream.edits}
        events={stream.events}
        spine={cheatData.doc?.spine ?? null}
      />
      {showCard && card && cheatData.doc && (
        <CheatCard card={card} counter={cheatData.doc.counter} fixture={!!cheatData.doc.fixture} cardRef={cardRef} reasonRef={reasonRef} />
      )}
      {ghosts.length === 0 && !error && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center font-sans text-lg font-medium tracking-wide text-[#7fb7ff]/80">
          Waiting for Ada&rsquo;s first recorded run
        </p>
      )}
      {error && (
        <p className="absolute left-1/2 top-4 -translate-x-1/2 text-sm" style={{ color: "#FF4A3D" }}>
          failed: {error}
        </p>
      )}
    </div>
  );
}
