"use client";

// SPEC §6 "Sound": one Web Audio context for the page. Created and decoded at startup (it stays
// suspended), resumed on the first click or key press, since browsers block audio before a
// gesture. A missing or undecodable file leaves its cue silent; nothing here ever throws.
import { MAX_FALLS, SFX, VoiceLimit, type Cue } from "@/lib/sound";

type Engine = { ctx: AudioContext; buffers: Partial<Record<Cue, AudioBuffer>>; falls: VoiceLimit };

let engine: Engine | null = null;

/** Idempotent: the first call creates the context, decodes each file once, and arms the unlock. */
export function startSound(): void {
  if (engine || typeof window === "undefined") return;
  let ctx: AudioContext;
  try {
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    ctx = new Ctor();
  } catch {
    return;
  }
  const e: Engine = { ctx, buffers: {}, falls: new VoiceLimit(MAX_FALLS) };
  engine = e;
  for (const cue of Object.keys(SFX) as Cue[]) {
    fetch(SFX[cue].url)
      .then((res) => (res.ok ? res.arrayBuffer() : null))
      .then((data) => (data ? ctx.decodeAudioData(data) : null))
      .then((buffer) => {
        if (buffer) e.buffers[cue] = buffer;
      })
      .catch(() => {});
  }
  const unlock = () => {
    ctx
      .resume()
      .then(() => {
        if (ctx.state !== "running") return;
        window.removeEventListener("pointerdown", unlock, true);
        window.removeEventListener("keydown", unlock, true);
      })
      .catch(() => {});
  };
  window.addEventListener("pointerdown", unlock, true);
  window.addEventListener("keydown", unlock, true);
}

/**
 * Plays the audible part of one clip on a fresh source node, so overlapping steps never cut each
 * other off. Before the unlock (or with its file missing) a cue is dropped, never queued: nothing
 * bursts out when sound starts.
 */
export function play(cue: Cue): void {
  const e = engine;
  if (!e || e.ctx.state !== "running") return;
  const buffer = e.buffers[cue];
  if (!buffer) return;
  if (cue === "fall" && !e.falls.acquire()) return;
  try {
    const src = e.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(e.ctx.destination);
    if (cue === "fall") src.onended = () => e.falls.release();
    src.start(e.ctx.currentTime, 0, SFX[cue].clipS);
  } catch {
    if (cue === "fall") e.falls.release();
  }
}
