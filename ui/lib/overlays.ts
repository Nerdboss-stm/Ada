// Pure overlay helpers (SPEC §6 "Overlays", SPEC §0 claim rule). No React imports so this
// runs under node --test.
import { pickLeader, type VersionDoc } from "./ghosts.ts";
import type { AdaEvent } from "./stream.ts";

/** CONTRACTS.md §4: one card, four cells, this order. */
export const GATE_STAGES = ["gate.verifier", "gate.gpa", "gate.meta", "gate.constraints"] as const;
export type GateStage = (typeof GATE_STAGES)[number];

export type CellState = "waiting" | "running" | "passed" | "failed";
export type Cell = { stage: GateStage; state: CellState };

/** The slice of an `edits` document (CONTRACTS.md §3) the card reads. */
export type EditDoc = {
  _id: string;
  origin?: "model" | "probe" | "cli" | string;
  primitive?: string;
  old?: unknown;
  new?: unknown;
  rationale?: string;
  predicted_delta?: number | null;
  actual_delta?: number | null;
  verdict?: "accepted" | "rejected" | null;
  reason?: string | null;
  created_at?: string;
};

const time = (s: string | undefined): number => {
  const t = Date.parse(s ?? "");
  return Number.isNaN(t) ? 0 : t;
};

/** Newest by created_at, then _id; null if none. */
export function latestEdit(edits: EditDoc[]): EditDoc | null {
  let best: EditDoc | null = null;
  for (const e of edits) {
    if (!best || time(e.created_at) > time(best.created_at) || (time(e.created_at) === time(best.created_at) && e._id > best._id)) {
      best = e;
    }
  }
  return best;
}

/** Dedupe by _id, incoming wins. */
export function mergeEdits(current: EditDoc[], incoming: EditDoc[]): EditDoc[] {
  const byId = new Map<string, EditDoc>();
  for (const e of current) byId.set(e._id, e);
  for (const e of incoming) byId.set(e._id, e);
  return [...byId.values()];
}

/**
 * The four cells for one edit, fixed order. A cell is running once its `start` arrives and
 * passed or failed once its `pass` or `fail` arrives (a result wins over a start regardless of
 * arrival order); `info` events do not change a cell.
 */
export function gateCells(events: AdaEvent[], editId: string | null): Cell[] {
  return GATE_STAGES.map((stage) => {
    let state: CellState = "waiting";
    let resultTs = -Infinity;
    if (editId !== null) {
      for (const e of events) {
        if (e.edit_id !== editId || e.stage !== stage) continue;
        if (e.status === "pass" || e.status === "fail") {
          const t = time(e.ts);
          if (t >= resultTs) {
            resultTs = t;
            state = e.status === "pass" ? "passed" : "failed";
          }
        } else if (e.status === "start" && state === "waiting") {
          state = "running";
        }
      }
    }
    return { stage, state };
  });
}

/** "gate.constraints" -> "constraints". */
export function stageLabel(stage: GateStage): string {
  return stage.slice("gate.".length);
}

function flat(value: unknown): string {
  if (value === undefined || value === null) return "–";
  const s = typeof value === "string" ? value : JSON.stringify(value);
  return s.replace(/\s+/g, " ").trim() || "–";
}

function clip(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

/** One line: old → new, each side clipped. */
export function oldToNew(old: unknown, next: unknown, max = 44): string {
  return `${clip(flat(old), max)} → ${clip(flat(next), max)}`;
}

/** Signed, 2 decimals; null -> null (the caller prints a word). */
export function formatDelta(d: number | null | undefined): string | null {
  if (typeof d !== "number" || !Number.isFinite(d)) return null;
  const s = d.toFixed(2);
  return d > 0 ? `+${s}` : s === "-0.00" ? "0.00" : s;
}

/** Meters, 2 decimals (CLAUDE.md). */
export function formatMeters(x: number): string {
  return `${x.toFixed(2)} m`;
}

// Displayed precision. The claim is judged on these integers, never on the raw floats.
const RELIABILITY_SCALE = 100; // whole percent
const COST_SCALE = 10_000; // USD to 4 places

export type Row = {
  versionId: string;
  reliabilityPct: number; // integer percent as displayed
  n: number;
  costUnits: number; // integer 1/10000 USD as displayed
  reliability: string;
  cost: string;
};

/** A version as it is displayed; null unless it has reliability, n, and cost. */
export function toRow(v: VersionDoc | null | undefined): Row | null {
  const m = v?.metrics;
  const r = m?.holdout_reliability_80;
  const c = m?.cost_per_run_usd;
  const n = m?.n;
  if (!v || typeof r !== "number" || typeof c !== "number" || typeof n !== "number") return null;
  if (![r, c, n].every(Number.isFinite)) return null;
  const reliabilityPct = Math.round(r * RELIABILITY_SCALE);
  const costUnits = Math.round(c * COST_SCALE);
  return {
    versionId: v._id,
    reliabilityPct,
    n,
    costUnits,
    reliability: `${reliabilityPct}%`,
    cost: `$${(costUnits / COST_SCALE).toFixed(4)}`,
  };
}

/** The frozen frontier: newest version with status "frontier". */
export function pickFrontier(versions: VersionDoc[]): VersionDoc | null {
  let best: VersionDoc | null = null;
  for (const v of versions) {
    if (v.status !== "frontier") continue;
    if (!best || (v.created_at ?? "") > (best.created_at ?? "")) best = v;
  }
  return best;
}

/** The current best is the leader (lib/ghosts.ts pickLeader), so the claim and the solid Ada agree. */
export function pickBest(versions: VersionDoc[]): VersionDoc | null {
  const id = pickLeader(versions);
  return versions.find((v) => v._id === id) ?? null;
}

/**
 * SPEC §0 on the displayed rounded values: best reliability ≥ frontier's, best cost < frontier's,
 * both with n > 0. Returns the cost ratio k as displayed (floored to one decimal so it never
 * overstates), or null when the claim is not earned. A best cost that displays as $0.0000 earns
 * nothing: the ratio would be unbounded.
 */
export function claimK(frontier: Row | null, best: Row | null): string | null {
  if (!frontier || !best) return null;
  if (frontier.n <= 0 || best.n <= 0) return null;
  if (best.reliabilityPct < frontier.reliabilityPct) return null;
  if (best.costUnits <= 0 || best.costUnits >= frontier.costUnits) return null;
  const k = Math.floor((frontier.costUnits / best.costUnits) * 10) / 10;
  return k.toFixed(1);
}

export function claimLine(k: string): string {
  return `matches frontier reliability at ${k}× lower cost`;
}
