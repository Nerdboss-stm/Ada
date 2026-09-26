"use client";

import { useMemo, type CSSProperties, type RefObject } from "react";
import { QRCodeSVG } from "qrcode.react";
import { counterLine, type CheatCounter, type CheatSpine } from "@/lib/cheatcard";
import type { VersionDoc } from "@/lib/ghosts";
import {
  claim,
  claimLine,
  countsLine,
  editCounts,
  formatDelta,
  formatUsd,
  gateCells,
  latestEdit,
  modelLabel,
  oldToNew,
  pickBest,
  pickFrontier,
  stageLabel,
  toRow,
  type Cell,
  type EditDoc,
  type Row,
} from "@/lib/overlays";
import type { AdaEvent } from "@/lib/stream";

// SPEC §6 "Overlays". Red only for fail and rejected; green only for the claim line.
const FAIL = "#FF4A3D";
const CLAIM = "#85BB65";
const INK = "#e6e8ec";
const MUTED = "#8a909b";
const FROZEN = "#5d636d";

const glass = "rounded-xl border border-white/10 bg-white/[0.06] backdrop-blur-md";

function TopLeft({ leaderVersion, distanceRef }: { leaderVersion: string | null; distanceRef: RefObject<HTMLSpanElement | null> }) {
  if (!leaderVersion) return null;
  return (
    <div className="absolute left-6 top-5" style={{ color: INK }}>
      <div className="text-2xl font-medium tracking-wide">Ada · {leaderVersion}</div>
      <span
        ref={distanceRef}
        data-testid="distance"
        className="block font-semibold leading-none tabular-nums"
        style={{ fontSize: "72pt" }}
      />
    </div>
  );
}

const CELL_STYLE: Record<Cell["state"], CSSProperties> = {
  waiting: { borderColor: "rgba(255,255,255,0.18)", color: MUTED },
  running: { borderColor: INK, color: INK, background: "rgba(230,232,236,0.12)" },
  passed: { borderColor: INK, color: "#05060a", background: INK },
  failed: { borderColor: FAIL, color: "#05060a", background: FAIL },
};

function EditCard({ edit, events }: { edit: EditDoc | null; events: AdaEvent[] }) {
  const cells = useMemo(() => gateCells(events, edit?._id ?? null), [events, edit]);
  if (!edit) return null;
  const predicted = formatDelta(edit.predicted_delta);
  const actual = formatDelta(edit.actual_delta);
  const verdict = edit.verdict ?? "pending";
  return (
    <section
      data-testid="edit-card"
      className={`absolute bottom-5 left-6 w-[min(440px,calc(100vw-3rem))] p-4 text-sm ${glass}`}
      style={{ color: INK }}
    >
      <header className="flex items-baseline justify-between gap-3">
        <span className="truncate font-medium">{edit._id}</span>
        <span style={{ color: edit.verdict === "rejected" ? FAIL : MUTED }} data-testid="verdict">
          {verdict}
        </span>
      </header>
      <div className="mt-1 flex gap-3" style={{ color: MUTED }}>
        <span>
          origin <span style={{ color: INK }}>{edit.origin ?? "unknown"}</span>
        </span>
        <span>
          primitive <span style={{ color: INK }}>{edit.primitive ?? "–"}</span>
        </span>
      </div>
      <p className="mt-2 truncate" title={oldToNew(edit.old, edit.new, 400)}>
        {oldToNew(edit.old, edit.new)}
      </p>
      {edit.rationale && <p className="mt-2 line-clamp-3 font-serif text-[15px] leading-snug">{edit.rationale}</p>}
      <div className="mt-2 flex gap-4 tabular-nums" style={{ color: MUTED }}>
        <span>
          predicted <span style={{ color: INK }}>{predicted ?? "none"}</span>
        </span>
        <span>
          actual <span style={{ color: INK }}>{actual ?? "pending"}</span>
        </span>
      </div>
      <ol className="mt-3 grid grid-cols-4 gap-2">
        {cells.map((c) => (
          <li
            key={c.stage}
            data-stage={c.stage}
            data-state={c.state}
            className="rounded-md border px-2 py-1.5 transition-colors duration-300"
            style={CELL_STYLE[c.state]}
          >
            <div className="text-xs opacity-80">{stageLabel(c.stage)}</div>
            <div className="font-medium">{c.state}</div>
          </li>
        ))}
      </ol>
      {edit.verdict === "rejected" && edit.reason && (
        <p className="mt-2" style={{ color: FAIL }}>
          {edit.reason}
        </p>
      )}
    </section>
  );
}

// Below 800 px the table stacks: no header, each row's label on its own line, its values wrapped under it.
const TR_STACK = "max-[800px]:flex max-[800px]:flex-wrap max-[800px]:gap-x-3 max-[800px]:pb-2";
const TD_LABEL = "pr-4 max-[800px]:w-full max-[800px]:pr-0 max-[800px]:font-medium";
const TD = "pr-4 max-[800px]:pr-0";

function RowLine({
  label,
  row,
  model,
  color,
  emphasized = false,
}: {
  label: string;
  row: Row | null;
  model: string;
  color: string;
  emphasized?: boolean;
}) {
  return (
    <tr
      style={{ color, outline: emphasized ? `1px solid ${INK}` : undefined, outlineOffset: 4 }}
      className={`${TR_STACK} ${emphasized ? "font-semibold" : ""}`}
      data-emphasized={emphasized || undefined}
    >
      <td className={TD_LABEL}>{label}</td>
      <td className={TD} data-testid="spine-model">
        {row ? model : "–"}
      </td>
      <td className={`${TD} text-right`}>{row ? row.holdout : "–"}</td>
      <td className={`${TD} text-right`}>{row ? `n ${row.n}` : "–"}</td>
      <td className="text-right">{row ? `${row.cost} per gait` : "–"}</td>
    </tr>
  );
}

const NO_MODELS: Record<string, string[]> = {};

/** The pinned scenes (C9): the rows the snapshot names instead of the streamed versions. */
export type PinnedRows = {
  frontier: VersionDoc | null;
  best: VersionDoc | null;
  /** Scene 6 adds harness v0 as the first row. */
  v0?: VersionDoc | null;
  rewrite_cost_usd: number | null;
};

/**
 * The spine: frontier and best rows (holdout, runs, cost per gait from metrics.cost_per_run_usd),
 * the rewrite cost, the live edit counts, and the claim when it is earned. `pinned` replaces the
 * streamed versions (C9); `emphasize` outlines the frontier row; `counter` adds the live cheat
 * counter line; `showCounts` false drops the edit counts; `className` replaces the corner placement.
 */
export function Spine({
  versions,
  edits,
  spine,
  pinned,
  emphasize,
  counter,
  showCounts = true,
  className = "absolute right-6 top-5",
}: {
  versions: VersionDoc[];
  edits: EditDoc[];
  spine: CheatSpine | null;
  pinned?: PinnedRows;
  emphasize?: "frontier";
  counter?: CheatCounter | null;
  showCounts?: boolean;
  className?: string;
}) {
  const frontier = useMemo(() => toRow(pinned ? pinned.frontier : pickFrontier(versions)), [pinned, versions]);
  const best = useMemo(() => toRow(pinned ? pinned.best : pickBest(versions)), [pinned, versions]);
  const v0 = useMemo(() => (pinned?.v0 ? toRow(pinned.v0) : null), [pinned]);
  const counts = useMemo(() => (spine && showCounts ? editCounts(spine.edits, edits) : null), [spine, edits, showCounts]);
  if (!frontier && !best) return null;
  const earned = claim(frontier, best);
  const models = spine?.models ?? NO_MODELS;
  const rewrite = formatUsd(pinned ? pinned.rewrite_cost_usd : spine?.rewrite_cost_usd);
  const dim = emphasize ? MUTED : undefined;
  return (
    <section className={`${className} px-4 py-3 text-base tabular-nums max-[800px]:text-sm ${glass}`} data-testid="spine">
      <table className="max-[800px]:block">
        <thead className="max-[800px]:hidden">
          <tr className="text-xs" style={{ color: MUTED }}>
            <th className="pr-4 text-left font-normal">version</th>
            <th className="pr-4 text-left font-normal">agent model</th>
            <th className="pr-4 text-right font-normal">holdout</th>
            <th className="pr-4 text-right font-normal">runs</th>
            <th className="text-right font-normal">cost per gait</th>
          </tr>
        </thead>
        <tbody className="max-[800px]:block">
          {v0 && <RowLine label={`harness ${v0.versionId}`} row={v0} model={modelLabel(models, v0.versionId)} color={dim ?? MUTED} />}
          <RowLine
            label={frontier ? `frontier ${frontier.versionId} · frozen` : "frontier · none"}
            row={frontier}
            model={frontier ? modelLabel(models, frontier.versionId) : "–"}
            color={emphasize === "frontier" ? INK : FROZEN}
            emphasized={emphasize === "frontier"}
          />
          <RowLine
            label={best ? `best ${best.versionId}` : "best · none"}
            row={best}
            model={best ? modelLabel(models, best.versionId) : "–"}
            color={dim ?? INK}
          />
        </tbody>
      </table>
      {(rewrite || counts || counter) && (
        <div className="mt-2 flex flex-wrap gap-x-4 text-sm" style={{ color: MUTED }}>
          {rewrite && <span data-testid="rewrite-cost">rewrite cost {rewrite}</span>}
          {counts && <span data-testid="edit-counts">{countsLine(counts)}</span>}
          {counter && <span data-testid="gait-counter">{counterLine(counter)}</span>}
        </div>
      )}
      {earned && (
        <p className="mt-2 font-medium" style={{ color: CLAIM }} data-testid="claim">
          {claimLine(earned)}
        </p>
      )}
    </section>
  );
}

function Qr() {
  const origin = window.location.origin;
  return (
    <div className="absolute bottom-5 right-6 flex flex-col items-center gap-1" style={{ color: MUTED }}>
      <div className="rounded-lg bg-[#e6e8ec] p-1.5">
        <QRCodeSVG value={origin} size={104} bgColor="#e6e8ec" fgColor="#05060a" marginSize={1} title={origin} />
      </div>
      <span className="text-xs">{window.location.host}</span>
    </div>
  );
}

export default function Overlays({
  leaderVersion,
  distanceRef,
  versions,
  edits,
  events,
  spine,
}: {
  leaderVersion: string | null;
  distanceRef: RefObject<HTMLSpanElement | null>;
  versions: VersionDoc[];
  edits: EditDoc[];
  events: AdaEvent[];
  spine: CheatSpine | null;
}) {
  const edit = useMemo(() => latestEdit(edits), [edits]);
  return (
    <div className="pointer-events-none absolute inset-0 font-sans">
      <TopLeft leaderVersion={leaderVersion} distanceRef={distanceRef} />
      <Spine versions={versions} edits={edits} spine={spine} />
      <EditCard edit={edit} events={events} />
      <Qr />
    </div>
  );
}
