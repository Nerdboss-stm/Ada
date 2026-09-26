"use client";

import type { RefObject } from "react";
import { CHEAT_RED } from "@/lib/cheat";
import { counterLine, type CheatCard as Card, type CheatCounter } from "@/lib/cheatcard";

const INK = "#e6e8ec";
const MUTED = "#8a909b";

/**
 * The cheat card (C8), shown with bullet time (CheatCardDriver toggles `cardRef` and `reasonRef`):
 * the edit's rationale word for word in Plex Serif, the agent's own words from the rejected run's
 * trace, the verifier reason word for word at 36 pt, and the counted over-rated gaits.
 */
export default function CheatCard({
  card,
  counter,
  fixture,
  cardRef,
  reasonRef,
}: {
  card: Card;
  /** The live gait counter; null until api/cheat answers. */
  counter: CheatCounter | null;
  fixture: boolean;
  cardRef: RefObject<HTMLElement | null>;
  reasonRef: RefObject<HTMLElement | null>;
}) {
  return (
    <section
      ref={cardRef}
      data-testid="cheat-card"
      className="pointer-events-none absolute right-6 top-1/2 w-[min(560px,calc(100vw-3rem))] -translate-y-1/2 rounded-xl border bg-[#05060a]/70 p-5 font-sans backdrop-blur-md max-[800px]:inset-x-4 max-[800px]:bottom-4 max-[800px]:top-auto max-[800px]:w-auto max-[800px]:translate-y-0 max-[800px]:p-4"
      style={{ visibility: "hidden", borderColor: `${CHEAT_RED}66`, color: INK }}
    >
      <header className="flex items-baseline justify-between gap-3 text-xs" style={{ color: MUTED }}>
        <span>the edit</span>
        {fixture && (
          <span className="rounded border px-1.5 py-0.5" style={{ borderColor: MUTED }} data-testid="cheat-fixture">
            fixture
          </span>
        )}
      </header>
      <p className="mt-1 font-serif text-lg leading-snug" data-testid="cheat-rationale">
        {card.rationale}
      </p>
      {card.quote && (
        <figure className="mt-4">
          <figcaption className="text-xs" style={{ color: MUTED }}>
            the agent
          </figcaption>
          <blockquote className="mt-1 text-base leading-snug" data-testid="cheat-quote">
            &ldquo;{card.quote}&rdquo;
          </blockquote>
        </figure>
      )}
      <p
        ref={reasonRef as RefObject<HTMLParagraphElement | null>}
        data-testid="cheat-reason"
        // 36 pt; smaller on a phone so the card fits one column.
        className="mt-4 text-[48px] font-semibold leading-tight max-[800px]:text-[28px]"
        style={{ visibility: "hidden", color: CHEAT_RED }}
      >
        {card.reason}
      </p>
      {counter && (
        <p className="mt-4 text-sm tabular-nums" style={{ color: MUTED }} data-testid="cheat-counter">
          {counterLine(counter)}
        </p>
      )}
    </section>
  );
}
