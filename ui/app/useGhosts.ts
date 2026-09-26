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
import type { FramesDoc } from "@/lib/replay";

/** One ghost: a recorded showcase frames doc, keyed by version (or fixture name), oldest first. */
export type Ghost = { key: string; doc: FramesDoc };

export type GhostState = { ghosts: Ghost[]; leader: string | null; error: string | null };

async function fetchFrames(url: string): Promise<FramesDoc> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`frames ${url}: ${res.status}`);
  return (await res.json()) as FramesDoc;
}

/** `?fixtures=` files and `?frames=` documents, in query order; leader by recorded final torso x. */
function useLocalGhosts(sources: { name: string; url: string }[] | null): GhostState {
  const [state, setState] = useState<GhostState>({ ghosts: [], leader: null, error: null });
  useEffect(() => {
    if (!sources) return;
    let live = true;
    Promise.all(sources.map(async (f) => ({ key: f.name, doc: await fetchFrames(f.url) })))
      .then((ghosts) => live && setState({ ghosts, leader: pickFixtureLeader(ghosts), error: null }))
      .catch((e: unknown) => live && setState((s) => ({ ...s, error: e instanceof Error ? e.message : String(e) })));
    return () => {
      live = false;
    };
  }, [sources]);
  return state;
}

/** Stream mode: versions from the page's one stream; each showcase frames doc fetched once. */
function useStreamGhosts(versions: VersionDoc[], streamError: string | null): GhostState {
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

export function useGhosts(versions: VersionDoc[], streamError: string | null): GhostState {
  // Stage is client-only (ssr: false), so the query string is available on first render.
  const local = useMemo(() => {
    const q = new URLSearchParams(window.location.search);
    if (!q.has("fixtures") && !q.has("frames")) return null;
    return [...parseFixtures(q.get("fixtures")), ...parseFrameIds(q.get("frames"))];
  }, []);
  const fromLocal = useLocalGhosts(local);
  const fromStream = useStreamGhosts(local === null ? versions : NO_VERSIONS, local === null ? streamError : null);
  return local ? fromLocal : fromStream;
}
