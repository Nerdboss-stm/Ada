// Pure helpers for the cheat card and the spine additions (C8), shared by api/cheat and the
// client. Every word on the card is read from a stored document; nothing here writes one.
// No three/React/mongodb imports so this runs under node --test.

/** NOTES.md [B9]: the over-rated reasons, both wordings in today's data. */
export const OVERRATED = /rated torque|rated motors/i;
/** The agent's words on the card: one sentence, at most this many characters. */
export const QUOTE_MAX = 160;
/** A peak torque ratio above this is over the rated torque (sim/bounds.py TORQUE_RATIO_MAX). */
export const RATED_RATIO_MAX = 1.0;

const FIXTURE_NAME = /^[a-z0-9_-]{1,64}$/;
const FRAMES_ID = /^[A-Za-z0-9_-]{1,64}$/;

export type CheatCard = {
  edit_id: string;
  /** The edit's rationale, word for word. */
  rationale: string;
  /** The verifier reason, word for word. */
  reason: string;
  frames_id: string;
  /** The agent's own words from the rejected run's trace, or null when it wrote none. */
  quote: string | null;
  /** The rejected run's recorded joint -> peak |torque| / rated. */
  peak_torque: Record<string, number>;
  model_id: string | null;
};

export type CheatCounter = {
  overrated: number;
  /** Accepted non-cli edits whose version has a run over the rated torque. */
  accepted: number;
  source: "scoreboard" | "count";
};

export type EditVerdict = { _id: string; verdict?: string | null; origin?: string | null };

export type CheatSpine = {
  /** version_id -> distinct agent model ids of that version's runs. */
  models: Record<string, string[]>;
  /** From the newest scoreboard document, or null when there is none. */
  rewrite_cost_usd: number | null;
  /** Every non-cli edit, for the live counter's starting point. */
  edits: EditVerdict[];
};

export type CheatDoc = { fixture?: boolean; card: CheatCard | null; counter: CheatCounter; spine: CheatSpine };

/** A rejected edit the card may show: over-rated reason, recorded frames, not a ./verify row. */
export function isOverratedReject(e: { verdict?: string | null; reason?: string | null; origin?: string | null; frames_id?: string | null }): boolean {
  return (
    e.verdict === "rejected" &&
    e.origin !== "cli" &&
    typeof e.reason === "string" &&
    OVERRATED.test(e.reason) &&
    typeof e.frames_id === "string" &&
    e.frames_id.length > 0
  );
}

/** The first sentence of `text`: up to the first . ! ? followed by whitespace, or the first line break. */
export function firstSentence(text: string): string {
  const s = text.trim();
  const end = /[.!?](?=\s|$)|\n/.exec(s);
  if (!end) return s;
  return s.slice(0, end[0] === "\n" ? end.index : end.index + 1).trim();
}

/** At most `max` characters: whole if it fits, else cut at a word boundary with "…". */
export function clipQuote(s: string, max = QUOTE_MAX): string {
  if (s.length <= max) return s;
  const cut = s.slice(0, max - 1);
  const space = cut.lastIndexOf(" ");
  return `${(space > 0 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

type Step = { node?: unknown; name?: unknown; valid?: unknown; message?: { content?: unknown } | null };

/**
 * harness/agent.py raw_steps: the last assistant message with words before the submit_gait
 * whose gait was taken (valid true); its first sentence, verbatim, clipped. Null if none.
 */
export function agentQuote(rawSteps: unknown): string | null {
  if (!Array.isArray(rawSteps)) return null;
  const steps = rawSteps as Step[];
  const submit = steps.findIndex((s) => s?.node === "tools" && s.name === "submit_gait" && s.valid === true);
  for (let i = submit - 1; i >= 0; i--) {
    const s = steps[i];
    if (s?.node !== "model") continue;
    const content = s.message?.content;
    if (typeof content === "string" && content.trim().length > 0) {
      const sentence = firstSentence(content);
      return sentence.length > 0 ? clipQuote(sentence) : null;
    }
  }
  return null;
}

/** Any recorded joint over the rated torque. */
export function overRated(peak: unknown): boolean {
  if (!peak || typeof peak !== "object") return false;
  return Object.values(peak).some((r) => typeof r === "number" && Number.isFinite(r) && r > RATED_RATIO_MAX);
}

/** The joint with the highest recorded peak, if it is over the rated torque; first on ties. */
export function overTorqueJoint(peak: Record<string, number> | null | undefined): { joint: string; ratio: number } | null {
  let best: { joint: string; ratio: number } | null = null;
  for (const [joint, ratio] of Object.entries(peak ?? {})) {
    if (typeof ratio !== "number" || !Number.isFinite(ratio)) continue;
    if (!best || ratio > best.ratio) best = { joint, ratio };
  }
  return best && best.ratio > RATED_RATIO_MAX ? best : null;
}

/** Per geom: the force index when the geom is driven by `joint` (lib/effects.ts legForceIndex), else -1. */
export function jointGeoms(forceIndex: number[], forceJoints: string[], joint: string | null): number[] {
  const j = joint === null ? -1 : forceJoints.indexOf(joint);
  return forceIndex.map((f) => (j >= 0 && f === j ? j : -1));
}

/** Accepted non-cli edits whose to_version has at least one run over the rated torque. */
export function acceptedOverRated(
  edits: { verdict?: string | null; origin?: string | null; to_version?: string | null }[],
  runs: { version_id?: unknown; peak_torque?: unknown }[],
): number {
  const hot = new Set(runs.filter((r) => overRated(r.peak_torque)).map((r) => String(r.version_id)));
  return edits.filter((e) => e.verdict === "accepted" && e.origin !== "cli" && typeof e.to_version === "string" && hot.has(e.to_version)).length;
}

export function counterLine(c: Pick<CheatCounter, "overrated" | "accepted">): string {
  return `over-rated attempts today: ${c.overrated} · accepted: ${c.accepted}`;
}

/** `?cheatfixture=name` -> the fixture file; otherwise api/cheat. Bad names fall back to the API. */
export function cheatSourceUrl(search: string): string {
  const name = new URLSearchParams(search).get("cheatfixture")?.trim().toLowerCase() ?? "";
  return FIXTURE_NAME.test(name) ? `/fixtures/${name}.json` : "/api/cheat";
}

const isStr = (x: unknown): x is string => typeof x === "string";
const isCount = (x: unknown): x is number => typeof x === "number" && Number.isInteger(x) && x >= 0;

function parseCard(c: unknown): CheatCard | null {
  if (!c || typeof c !== "object") return null;
  const d = c as Record<string, unknown>;
  if (!isStr(d.edit_id) || !isStr(d.rationale) || !isStr(d.reason) || !isStr(d.frames_id) || !FRAMES_ID.test(d.frames_id)) return null;
  const peak: Record<string, number> = {};
  if (d.peak_torque && typeof d.peak_torque === "object") {
    for (const [k, v] of Object.entries(d.peak_torque)) if (typeof v === "number" && Number.isFinite(v)) peak[k] = v;
  }
  return {
    edit_id: d.edit_id,
    rationale: d.rationale,
    reason: d.reason,
    frames_id: d.frames_id,
    quote: isStr(d.quote) && d.quote.trim().length > 0 ? d.quote : null,
    peak_torque: peak,
    model_id: isStr(d.model_id) ? d.model_id : null,
  };
}

/** Validates an api/cheat response or fixture; null when the counter or spine is malformed. */
export function parseCheat(x: unknown): CheatDoc | null {
  if (!x || typeof x !== "object") return null;
  const d = x as Record<string, unknown>;
  const c = d.counter as Record<string, unknown> | undefined;
  const s = d.spine as Record<string, unknown> | undefined;
  if (!c || !isCount(c.overrated) || !isCount(c.accepted) || (c.source !== "scoreboard" && c.source !== "count")) return null;
  if (!s || !s.models || typeof s.models !== "object" || !Array.isArray(s.edits)) return null;
  const models: Record<string, string[]> = {};
  for (const [v, ids] of Object.entries(s.models as Record<string, unknown>)) {
    if (Array.isArray(ids)) models[v] = ids.filter(isStr);
  }
  const edits = (s.edits as unknown[]).filter((e): e is EditVerdict => !!e && typeof e === "object" && isStr((e as EditVerdict)._id));
  const cost = s.rewrite_cost_usd;
  return {
    fixture: d.fixture === true,
    card: parseCard(d.card),
    counter: { overrated: c.overrated, accepted: c.accepted, source: c.source },
    spine: { models, rewrite_cost_usd: typeof cost === "number" && Number.isFinite(cost) ? cost : null, edits },
  };
}
