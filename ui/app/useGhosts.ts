"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ghostSources,
  parseFixtures,
  parseFrameIds,
  pickFixtureLeader,
  pickLeader,
  type VersionDoc,
} from "@/lib/ghosts";
import { isCheatDoc, latestRejectedEdit, pickCheatDoc, verdictText } from "@/lib/cheat";
import { overTorqueJoint, type CheatCard } from "@/lib/cheatcard";
import type { EditDoc } from "@/lib/overlays";
import type { FramesDoc } from "@/lib/replay";

/** One ghost: a recorded showcase frames doc, keyed by version (or fixture name), oldest first. */
export type Ghost = { key: string; doc: FramesDoc };

/**
 * A rejected run with a violation frame, played as its own body in bullet time, its verdict text,
 * and the joint over the rated torque in that run's recorded peak_torque (null if none known).
 */
export type Cheat = { key: string; doc: FramesDoc & { violation_frame: number }; verdict: string; hotJoint: string | null };

export type GhostState = { ghosts: Ghost[]; leader: string | null; cheat: Cheat | null; error: string | null };

export async function fetchFrames(url: string): Promise<FramesDoc> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`frames ${url}: ${res.status}`);
  return (await res.json()) as FramesDoc;
}

function asCheat(key: string, doc: FramesDoc, reason: string | null | undefined, hotJoint: string | null = null): Cheat | null {
  if (!isCheatDoc(doc)) return null;
  const v = doc.violation_frame as number;
  return { key, doc: { ...doc, violation_frame: v }, verdict: verdictText(reason, v), hotJoint };
}

/** Which rejected run plays as the cheat, and what it says. */
type CheatTarget = { framesId: string; reason: string | null | undefined; hotJoint: string | null };

/**
 * `?fixtures=` files and `?frames=` documents, in query order; leader by recorded final torso x.
 * The first rejected doc with a violation frame is the cheat, not a ghost.
 */
function useLocalGhosts(sources: { name: string; url: string }[] | null): GhostState {
  const [state, setState] = useState<GhostState>({ ghosts: [], leader: null, cheat: null, error: null });
  useEffect(() => {
    if (!sources) return;
    let live = true;
    Promise.all(sources.map(async (f) => ({ key: f.name, doc: await fetchFrames(f.url) })))
      .then((all) => {
        if (!live) return;
        const cheatKey = pickCheatDoc(all);
        const found = all.find((g) => g.key === cheatKey);
        const ghosts = all.filter((g) => g.key !== cheatKey);
        setState({ ghosts, leader: pickFixtureLeader(ghosts), cheat: found ? asCheat(found.key, found.doc, null) : null, error: null });
      })
      .catch((e: unknown) => live && setState((s) => ({ ...s, error: e instanceof Error ? e.message : String(e) })));
    return () => {
      live = false;
    };
  }, [sources]);
  return state;
}

/** The target's frames, if they are a cheat; verdict is the target's reason. */
function useTargetCheat(target: CheatTarget | null): { cheat: Cheat | null; error: string | null } {
  const framesId = target?.framesId ?? null;
  const [loaded, setLoaded] = useState<{ id: string; doc: FramesDoc } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!framesId) return;
    let live = true;
    fetchFrames(`/api/frames/${encodeURIComponent(framesId)}`)
      .then((doc) => live && setLoaded({ id: framesId, doc }))
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [framesId]);
  const reason = target?.reason;
  const hotJoint = target?.hotJoint ?? null;
  const cheat = useMemo(
    () => (framesId && loaded && loaded.id === framesId ? asCheat(`cheat:${loaded.id}`, loaded.doc, reason, hotJoint) : null),
    [loaded, framesId, reason, hotJoint],
  );
  return useMemo(() => ({ cheat, error }), [cheat, error]);
}

/** Stream mode: versions from the page's one stream; each showcase frames doc fetched once. */
function useStreamGhosts(versions: VersionDoc[], streamError: string | null): Omit<GhostState, "cheat"> {
  const [docs, setDocs] = useState<Record<string, FramesDoc>>({});
  const [fetchError, setFetchError] = useState<string | null>(null);
  const requested = useRef(new Set<string>());
  const error = streamError ?? fetchError;

  const sources = useMemo(() => ghostSources(versions), [versions]);

  useEffect(() => {
    for (const { framesId } of sources) {
      if (requested.current.has(framesId)) continue;
      requested.current.add(framesId);
      fetchFrames(`/api/frames/${encodeURIComponent(framesId)}`)
        .then((doc) => setDocs((prev) => ({ ...prev, [framesId]: doc })))
        .catch((e: unknown) => {
          requested.current.delete(framesId);
          setFetchError(e instanceof Error ? e.message : String(e));
        });
    }
  }, [sources]);

  return useMemo(() => {
    const ghosts = sources.filter((s) => docs[s.framesId]).map((s) => ({ key: s.versionId, doc: docs[s.framesId] }));
    const leader = pickLeader(versions);
    return { ghosts, leader: ghosts.some((g) => g.key === leader) ? leader : null, error };
  }, [sources, docs, versions, error]);
}

const NO_VERSIONS: VersionDoc[] = [];

/**
 * The cheat card's edit (api/cheat or `?cheatfixture=`) plays as the cheat in every mode. Without
 * a card: stream mode plays the latest rejected edit with frames; `?fixtures=`/`?frames=` mode
 * plays the first rejected doc it lists.
 */
export function useGhosts(versions: VersionDoc[], edits: EditDoc[], streamError: string | null, card: CheatCard | null = null): GhostState {
  // Stage is client-only (ssr: false), so the query string is available on first render.
  const local = useMemo(() => {
    const q = new URLSearchParams(window.location.search);
    if (!q.has("fixtures") && !q.has("frames")) return null;
    return [...parseFixtures(q.get("fixtures")), ...parseFrameIds(q.get("frames"))];
  }, []);
  const fromLocal = useLocalGhosts(local);
  const fromStream = useStreamGhosts(local === null ? versions : NO_VERSIONS, local === null ? streamError : null);
  const target = useMemo((): CheatTarget | null => {
    if (card) return { framesId: card.frames_id, reason: card.reason, hotJoint: overTorqueJoint(card.peak_torque)?.joint ?? null };
    if (local) return null;
    const e = latestRejectedEdit(edits);
    return e?.frames_id ? { framesId: e.frames_id, reason: e.reason, hotJoint: null } : null;
  }, [card, local, edits]);
  const targetCheat = useTargetCheat(target);
  return useMemo(() => {
    if (local && !card) return fromLocal;
    const base = local ? fromLocal : fromStream;
    return { ...base, cheat: targetCheat.cheat, error: base.error ?? targetCheat.error };
  }, [local, card, fromLocal, fromStream, targetCheat]);
}
