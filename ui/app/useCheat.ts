"use client";

import { useEffect, useMemo, useState } from "react";
import { cheatSourceUrl, parseCheat, type CheatDoc } from "@/lib/cheatcard";
import type { EditDoc } from "@/lib/overlays";

/** Wait this long after the last verdict change before re-reading api/cheat. */
const REFETCH_MS = 1000;

/**
 * api/cheat (or `?cheatfixture=`, read once). The API is re-read whenever a streamed edit gets a
 * verdict, so a new over-rated rejection, its count, and new versions' model ids show up live.
 */
export function useCheat(edits: EditDoc[]): { doc: CheatDoc | null; error: string | null } {
  const url = useMemo(() => cheatSourceUrl(window.location.search), []);
  const live = url === "/api/cheat";
  const verdicts = useMemo(
    () =>
      live
        ? edits
            .filter((e) => e.verdict)
            .map((e) => `${e._id}:${e.verdict}`)
            .sort()
            .join("|")
        : "",
    [edits, live],
  );
  const [doc, setDoc] = useState<CheatDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const timer = setTimeout(
      () =>
        fetch(url)
          .then(async (res) => {
            if (!res.ok) throw new Error(`cheat ${url}: ${res.status}`);
            const parsed = parseCheat(await res.json());
            if (!parsed) throw new Error(`cheat ${url}: malformed`);
            if (alive) {
              setDoc(parsed);
              setError(null);
            }
          })
          .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e))),
      doc === null ? 0 : REFETCH_MS,
    );
    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // `doc` only picks the delay; re-reading is driven by the verdicts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, verdicts]);

  return { doc, error };
}
