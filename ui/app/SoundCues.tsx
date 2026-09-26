"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { cheatState, type CheatTimeline } from "@/lib/cheat";
import { footprintEdges } from "@/lib/effects";
import { sharedFrameIndex } from "@/lib/ghosts";
import type { Manifest } from "@/lib/replay";
import { crossedFrames, fallFrame } from "@/lib/sound";
import { playbackTime, type Epoch } from "./Ghosts";
import { play, startSound } from "./sound";
import type { Ghost } from "./useGhosts";

/**
 * SPEC §6 "Sound", on the same shared clock and frame index as the bodies: a step on each of the
 * leader's footprints (the same debounced contact edges the discs use), a fall when a ghost
 * reaches its fall frame, the glitch when the cheat reaches its violation frame. Renders nothing;
 * goes inside the Canvas.
 */
export function SoundCues({
  manifest,
  leader,
  ghosts,
  cheat,
  cycle,
  epoch,
}: {
  manifest: Manifest;
  leader: Ghost | undefined;
  ghosts: Ghost[];
  cheat: CheatTimeline | null;
  cycle: number;
  epoch: Epoch;
}) {
  useEffect(() => startSound(), []);
  const steps = useMemo(() => (leader ? [...new Set(footprintEdges(leader.doc, manifest).map((e) => e.frame))] : []), [leader, manifest]);
  const falls = useMemo(
    () => ghosts.map((g) => ({ key: g.key, doc: g.doc, frame: fallFrame(g.doc, manifest) })).filter((g) => g.frame !== null),
    [ghosts, manifest],
  );
  // Last frame shown per source; -1 before the first. Kept while muted so unlocking never replays a backlog.
  const last = useRef(new Map<string, number>());

  useFrame(({ clock }) => {
    const t = playbackTime(epoch, clock.elapsedTime);
    const seen = last.current;
    const advance = (key: string, cur: number, cues: number[]): boolean => {
      const hit = crossedFrames(cues, seen.get(key) ?? -1, cur).length > 0;
      seen.set(key, cur);
      return hit;
    };
    if (leader) {
      const { fps, frames } = leader.doc;
      // Feet landing in the same rendered frame are one step.
      if (advance(`step:${leader.key}`, sharedFrameIndex(t, fps, frames.length, cycle), steps)) play("step");
    }
    for (const g of falls) {
      if (advance(`fall:${g.key}`, sharedFrameIndex(t, g.doc.fps, g.doc.frames.length, cycle), [g.frame as number])) play("fall");
    }
    if (cheat && advance("glitch", cheatState(cheat, t).frame, [cheat.violationFrame])) play("glitch");
  });
  return null;
}
