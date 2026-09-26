"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BOOTH,
  boothNext,
  boothSeconds,
  parsePinned,
  parseScript,
  pinnedFramesIds,
  replayLabel,
  sceneForKey,
  type Pinned,
  type SceneId,
} from "@/lib/director";
import type { Manifest } from "@/lib/replay";
import AttemptsStage from "./Attempts";
import { BG, SharedCanvas, useManifest } from "./Scene";
import { CheatScene, FinaleScene, SceneError, SpineScene, V0Scene, type Live } from "./Scenes";
import SwapStage from "./Swap";
import { useAdaStream } from "./useAdaStream";
import { useCheat } from "./useCheat";
import { fetchFrames } from "./useGhosts";

const INK = "#e6e8ec";
const MUTED = "#8a909b";

type PinState = { status: "loading" } | { status: "waiting" } | { status: "error"; message: string } | { status: "ready"; pinned: Pinned };

/** api/snapshot/latest, read once: the ids every scene plays. The newest pin wins on reload. */
function usePinned(): PinState {
  const [state, setState] = useState<PinState>({ status: "loading" });
  useEffect(() => {
    let live = true;
    (async () => {
      const res = await fetch("/api/snapshot/latest");
      if (res.status === 404) return live && setState({ status: "waiting" });
      if (!res.ok) throw new Error(`snapshot: ${res.status}`);
      const pinned = parsePinned(await res.json());
      if (!pinned) throw new Error("snapshot: missing or malformed fields");
      // Every frames doc a scene plays, fetched once now so a key switch never waits on the network.
      for (const id of pinnedFramesIds(pinned)) fetchFrames(`/api/frames/${encodeURIComponent(id)}`).catch(() => {});
      if (live) setState({ status: "ready", pinned });
    })().catch((e: unknown) => live && setState({ status: "error", message: e instanceof Error ? e.message : String(e) }));
    return () => {
      live = false;
    };
  }, []);
  return state;
}

/** ui/public/script.json captions (booth only); a missing file just means no captions. */
function useScript(booth: boolean): Partial<Record<SceneId, string>> {
  const [script, setScript] = useState<Partial<Record<SceneId, string>>>({});
  useEffect(() => {
    if (!booth) return;
    let live = true;
    fetch("/script.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((raw) => live && setScript(parseScript(raw)))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [booth]);
  return script;
}

/** Keys 1..7 switch scenes (booth too: staff can steer, the cycle continues from there). */
function useSceneKeys(onScene: (scene: SceneId) => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
      const scene = sceneForKey(e.key, e.metaKey || e.ctrlKey || e.altKey);
      if (scene !== null) onScene(scene);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onScene]);
}

function SceneView({ scene, pinned, manifest, live }: { scene: SceneId; pinned: Pinned; manifest: Manifest; live: Live }) {
  switch (scene) {
    case 1:
      return <V0Scene pinned={pinned} manifest={manifest} />;
    case 2:
      return <SpineScene pinned={pinned} manifest={manifest} live={live} comparison={false} />;
    case 3:
      return <AttemptsStage editIds={pinned.snapshot.attempt_edit_ids} />;
    case 4:
      return <CheatScene pinned={pinned} manifest={manifest} counter={live.counter} />;
    case 5:
      return <SwapStage pinned={pinned.swap} />;
    case 6:
      return <SpineScene pinned={pinned} manifest={manifest} live={live} comparison />;
    case 7:
      return <FinaleScene pinned={pinned} manifest={manifest} />;
  }
}

/**
 * `?pinned=1` (C9): every scene reads its ids from the newest snapshots document; number keys
 * switch scenes instantly (one shared canvas; frames prefetched); only the live counters (api/cheat's gait counter and the streamed edit
 * counts) keep updating. `booth` (`?mode=booth`) auto-advances 1, 3, 4, 5, 7 with fixed
 * durations, a caption per scene from script.json, and the replay label.
 */
export default function Director({ booth }: { booth: boolean }) {
  const { manifest, error: manifestError } = useManifest();
  const pin = usePinned();
  const script = useScript(booth);
  const stream = useAdaStream();
  const cheat = useCheat(stream.edits);
  // `n` remounts the scene, so pressing its key again restarts it from its first frame.
  const [at, setAt] = useState<{ scene: SceneId; n: number }>({ scene: BOOTH[0].scene, n: 0 });
  const jump = useCallback((scene: SceneId) => setAt((s) => ({ scene, n: s.n + 1 })), []);
  useSceneKeys(jump);

  useEffect(() => {
    if (!booth) return;
    const timer = setTimeout(() => setAt((s) => ({ scene: boothNext(s.scene), n: s.n + 1 })), boothSeconds(at.scene) * 1000);
    return () => clearTimeout(timer);
  }, [booth, at]);

  const live = useMemo<Live>(
    () => ({ spine: cheat.doc?.spine ?? null, counter: cheat.doc?.counter ?? null, edits: stream.edits }),
    [cheat.doc, stream.edits],
  );
  const pinned = pin.status === "ready" ? pin.pinned : null;
  const caption = booth ? script[at.scene] : undefined;
  const label = booth && pinned ? replayLabel(pinned.replay.first, pinned.replay.last) : null;
  const error = manifestError ?? (pin.status === "error" ? pin.message : null) ?? stream.error ?? cheat.error;

  return (
    <div className="flex h-dvh w-full flex-col font-sans" style={{ background: BG }} data-scene={at.scene}>
      <div className="relative min-h-0 flex-1">
        <SharedCanvas>
          {pinned && manifest && <SceneView key={`${at.scene}:${at.n}`} scene={at.scene} pinned={pinned} manifest={manifest} live={live} />}
        </SharedCanvas>
        {pin.status === "waiting" && (
          <p className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center text-lg font-medium tracking-wide text-[#7fb7ff]/80">
            Waiting for the first pinned snapshot
          </p>
        )}
        <SceneError error={error} />
      </div>
      {booth && (caption || label) && (
        <footer className="shrink-0 border-t border-white/10 px-8 py-4 text-center max-[800px]:px-4 max-[800px]:py-3">
          {caption && (
            <p data-testid="caption" className="text-[28px] font-medium leading-snug max-[800px]:text-[20px]" style={{ color: INK }}>
              {caption}
            </p>
          )}
          {label && (
            <p data-testid="replay-label" className="mt-1 text-base tabular-nums max-[800px]:text-sm" style={{ color: MUTED }}>
              {label}
            </p>
          )}
        </footer>
      )}
    </div>
  );
}
