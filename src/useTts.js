import { useCallback, useMemo, useRef } from "react";
import { nextChunk } from "./chunk.js";

function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Speaks streamed assistant text, and tracks exactly how much of it has actually
// been spoken.
//
// Each synthesised clip carries ElevenLabs character-level timestamps and is
// scheduled at a known point on the Web Audio timeline, so at any moment the
// audio clock can be mapped back to a character offset in the model's text.
// That's what makes it possible to dim text that hasn't been voiced yet, and on
// interruption to report what the assistant actually got to say.
//
// onSpeakingChange(bool) fires when the pipeline goes busy/idle — the caller uses
// it to gate the microphone.
export function useTts(enabled, { onSpeakingChange, onAudibleChange } = {}) {
  const refs = useRef({
    ctx: null,
    buffer: "", // text not yet cut into a chunk
    raw: "", // everything the model has emitted this turn
    consumed: 0, // how much of `raw` has been cut into chunks
    clips: [], // { rawStart, rawEnd, startAt, endAt, ends[] }
    nextStart: 0,
    chain: Promise.resolve(),
    gen: 0,
    inflight: 0, // synthesis requests in progress
    playing: 0, // scheduled clips not finished
    speaking: false,
    audible: false, // sound is actually coming out, not merely being fetched
    spoken: false, // has anything been spoken this turn
    stopped: false, // interrupted: stay silent for the rest of this turn
  });

  const cbRef = useRef(onSpeakingChange);
  cbRef.current = onSpeakingChange;
  const audibleCbRef = useRef(onAudibleChange);
  audibleCbRef.current = onAudibleChange;

  const sync = useCallback(() => {
    const r = refs.current;
    const speaking = r.inflight > 0 || r.playing > 0;
    if (speaking !== r.speaking) {
      r.speaking = speaking;
      cbRef.current?.(speaking);
    }
    // Distinct from `speaking`, which is true during synthesis too. Barge-in must
    // key off this: armed while merely fetching, an interruption lands before any
    // clip is scheduled, so playback position reads 0 and the whole reply looks
    // unspoken.
    const audible = r.playing > 0;
    if (audible !== r.audible) {
      r.audible = audible;
      audibleCbRef.current?.(audible);
    }
  }, []);

  const context = () => {
    const r = refs.current;
    if (!r.ctx || r.ctx.state === "closed") r.ctx = new AudioContext();
    if (r.ctx.state === "suspended") r.ctx.resume();
    return r.ctx;
  };

  // How many characters of `raw` have been voiced by now.
  const progress = useCallback(() => {
    const r = refs.current;
    if (!r.ctx || r.ctx.state === "closed") return 0;
    const now = r.ctx.currentTime;
    let cursor = 0;

    for (const clip of r.clips) {
      if (clip.standalone) continue; // announcement, not part of this turn's text
      if (now >= clip.endAt) {
        cursor = clip.rawEnd; // fully spoken
        continue;
      }
      if (now < clip.startAt) break; // not started, so nothing later has either

      // Mid-clip: count the characters whose audio has already finished.
      const t = now - clip.startAt;
      let n = 0;
      while (n < clip.ends.length && clip.ends[n] <= t) n++;
      cursor = Math.min(clip.rawEnd, clip.rawStart + n);
      break;
    }
    return cursor;
  }, []);

  const play = useCallback(
    (text, rawStart, rawEnd, standalone = false) => {
      const r = refs.current;
      const generation = r.gen;
      r.inflight += 1;
      sync();

      // Serialize so clips queue in the order they were spoken.
      r.chain = r.chain.then(async () => {
        try {
          if (generation !== r.gen) return; // cancelled while queued

          const res = await fetch("/api/tts", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ text }),
          });
          if (!res.ok || generation !== r.gen) return;

          const { audio, ends } = await res.json();
          const ctx = context();
          const decoded = await ctx.decodeAudioData(base64ToBytes(audio).buffer);
          if (generation !== r.gen) return;

          const src = ctx.createBufferSource();
          src.buffer = decoded;
          src.connect(ctx.destination);
          const startAt = Math.max(ctx.currentTime + 0.02, r.nextStart);

          r.playing += 1;
          src.onended = () => {
            r.playing = Math.max(0, r.playing - 1);
            sync();
          };
          src.start(startAt);
          r.nextStart = startAt + decoded.duration;

          r.clips.push({
            standalone,
            rawStart,
            rawEnd,
            startAt,
            endAt: startAt + decoded.duration,
            // Even spacing is the fallback if alignment didn't come back.
            ends:
              ends?.length === text.length
                ? ends
                : Array.from(
                    { length: text.length },
                    (_, i) => ((i + 1) / text.length) * decoded.duration
                  ),
          });
        } catch (err) {
          console.warn("tts playback failed:", err);
        } finally {
          r.inflight = Math.max(0, r.inflight - 1);
          sync();
        }
      });
    },
    [sync]
  );

  // Cut whatever chunks are ready out of the buffer, tracking where each one
  // sits in `raw` so playback position maps back to the displayed text.
  const drain = useCallback(
    (final) => {
      const r = refs.current;
      let cut;
      while (final ? r.buffer.trim() : (cut = nextChunk(r.buffer, r.spoken))) {
        let chunk;
        let rest;
        if (final) {
          chunk = r.buffer.trim();
          rest = "";
        } else {
          [chunk, rest] = cut;
        }

        const spanStart = r.consumed;
        r.consumed += r.buffer.length - rest.length;
        r.buffer = rest;

        if (chunk) {
          // The chunk was trimmed out of its raw span, so skip the leading
          // whitespace to keep character offsets aligned with the text on screen.
          const lead = r.raw.slice(spanStart, r.consumed).search(/\S/);
          const rawStart = spanStart + Math.max(0, lead);
          play(chunk, rawStart, rawStart + chunk.length);
          r.spoken = true;
        }
        if (final) break;
      }
    },
    [play]
  );

  const feed = useCallback(
    (delta) => {
      const r = refs.current;
      r.raw += delta; // tracked even when disabled so offsets stay meaningful
      if (!enabled || r.stopped) return;
      r.buffer += delta;
      drain(false);
    },
    [enabled, drain]
  );

  const flush = useCallback(() => {
    const r = refs.current;
    if (!enabled || r.stopped) return;
    drain(true);
  }, [enabled, drain]);

  const speak = useCallback(
    (text) => {
      // Standalone announcement — not part of the streamed turn's offsets.
      if (enabled && !refs.current.stopped && text?.trim()) play(text.trim(), 0, 0, true);
    },
    [enabled, play]
  );

  const reset = useCallback(() => {
    const r = refs.current;
    r.gen += 1;
    r.buffer = "";
    r.raw = "";
    r.consumed = 0;
    r.clips = [];
    r.spoken = false;
    r.nextStart = 0;
    r.inflight = 0;
    r.playing = 0;
    if (r.ctx && r.ctx.state !== "closed") {
      r.ctx.close();
      r.ctx = null;
    }
    sync();
  }, [sync]);

  // Start-of-turn clear: silence whatever is queued, then accept new text.
  const cancel = useCallback(() => {
    reset();
    refs.current.stopped = false;
  }, [reset]);

  // Interruption. Returns where playback actually got to, so the caller can trim
  // the assistant's message to what the user really heard, and ignores the rest
  // of this turn's text — otherwise later chunks would start speaking again.
  const stopTurn = useCallback(() => {
    const r = refs.current;
    const index = progress();
    const cut = { index, spoken: r.raw.slice(0, index), full: r.raw };
    reset();
    r.stopped = true;
    return cut;
  }, [progress, reset]);

  // Stable identity — App keeps this in effect dependency lists.
  return useMemo(
    () => ({ feed, flush, speak, cancel, stopTurn, progress }),
    [feed, flush, speak, cancel, stopTurn, progress]
  );
}
