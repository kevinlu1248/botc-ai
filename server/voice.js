import { WebSocketServer, WebSocket } from "ws";
import fs from "node:fs";
import path from "node:path";
import { resetVoiceBinding } from "./vision.js";
import { getSettings } from "./settings.js";

// Writes a playable WAV rather than headerless PCM, so dumps can be opened in
// Finder / QuickTime directly. The data length isn't known until the stream ends,
// so the header is written with placeholders and patched on close.
function createWavWriter(file, sampleRate) {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36, 4); // patched
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28); // byte rate
  header.writeUInt16LE(2, 32); // block align
  header.writeUInt16LE(16, 34); // bits per sample
  header.write("data", 36);
  header.writeUInt32LE(0, 40); // patched

  const stream = fs.createWriteStream(file);
  stream.write(header);
  let bytes = 0;

  return {
    file,
    write(chunk) {
      bytes += chunk.length;
      stream.write(chunk);
    },
    end() {
      stream.end(() => {
        // Patch RIFF size and data size now that the length is known.
        fs.open(file, "r+", (err, fd) => {
          if (err) return;
          const sizes = Buffer.alloc(4);
          sizes.writeUInt32LE(36 + bytes, 0);
          fs.write(fd, sizes, 0, 4, 4, () => {
            sizes.writeUInt32LE(bytes, 0);
            fs.write(fd, sizes, 0, 4, 40, () => fs.close(fd, () => {}));
          });
        });
      });
      return bytes / 2 / sampleRate;
    },
  };
}

// ---------------------------------------------------------------------------
// Speech-to-text: browser mic (16 kHz PCM over WebSocket) -> Deepgram Nova-3.
// The browser never sees the Deepgram key; this process is the only holder.
// ---------------------------------------------------------------------------

const ms = (name, fallback) => {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v > 0 ? Math.round(v) : fallback;
};

// How long you may pause before the turn is considered over.
// Balance: too short → "um" / mid-thought cuts; too long → painful wait after done.
// Deepgram utterance_end_ms minimum is 1000.
const ENDPOINTING_MS = ms("STT_ENDPOINTING_MS", 1100);
const UTTERANCE_END_MS = Math.max(1000, ms("STT_UTTERANCE_END_MS", 1000));
// Extra wait when the transcript looks incomplete ("…and then", "um", "Read").
const CONTINUATION_GRACE_MS = ms("STT_CONTINUATION_GRACE_MS", 450);
// Backstop when Deepgram reports no boundary at all.
const IDLE_FLUSH_MS = ms("STT_IDLE_FLUSH_MS", 1600);

// Words that essentially never end a sentence. If speech stops right after one,
// the speaker is pausing to think, not finished — "come up with a hard math
// problem, and then …" should not be submitted as a complete request.
const DANGLING_WORD =
  /(?:^|\s)(?:and|or|but|so|then|because|since|although|though|while|whereas|if|when|whenever|where|which|who|whom|whose|that|than|with|without|within|to|too|for|from|of|off|in|into|on|onto|at|by|as|about|after|before|over|under|between|among|the|a|an|my|our|your|his|her|its|their|this|these|those|is|are|was|were|am|be|been|being|will|would|can|could|should|shall|may|might|must|do|does|did|have|has|had|also|plus|versus|vs|like|such|um|uh|er|ah|hmm|need|let'?s|gonna|wanna|kinda|sorta)\s*$/i;

// Pure hesitation / ack with no ask — never submit as a user turn.
const FILLER_ONLY =
  /^(um+|uh+|er+|ah+|oh+|hmm+|mm+|mhm+|huh|like|so|well|yeah|yep|yup|nah|right|okay|ok)([.?!…,]*)$/i;

// Single-word openers that almost always continue ("Read …", "Need …").
const OPENER =
  /^(read|write|tell|give|make|say|show|get|put|do|can|could|would|will|i|we|you|the|a|an|need|let'?s|please|just|also|and|but|so)\b/i;

const COMMAND =
  /^(stop|wait|cancel|quiet|enough|pause|resume|go|yes|no|thanks|thank you)([.?!…]*)$/i;

export function isCommand(text) {
  return COMMAND.test(text.trim());
}

export function isFillerOnly(text) {
  return FILLER_ONLY.test(text.trim());
}

export function looksUnfinished(text) {
  const t = text.trim();
  if (!t) return false;
  if (isCommand(t)) return false; // "stop" is a complete turn
  if (/[,;:—–]$/.test(t)) return true; // trailing clause punctuation
  if (DANGLING_WORD.test(t)) return true;
  if (isFillerOnly(t)) return true; // wait — more speech usually follows "um"
  const words = t.replace(/[.?!…,]+$/g, "").split(/\s+/).filter(Boolean);
  // One short token is rarely a finished request ("Read", "Need").
  if (words.length === 1 && words[0].length <= 14) return true;
  // Very short openers only ("Okay so", "I need") — not "tell me a joke".
  if (words.length <= 2 && OPENER.test(t) && !/[.?!]$/.test(t)) return true;
  return false;
}

// Selectable from the UI so accuracy can be A/B'd live. Benchmarked on clean
// speech (14s of Deepgram's own sample, WER vs the published transcript):
// nova-3 3.4%, nova-2 6.9%, enhanced 10.3%, base 10.3% — and nova-3's single
// "error" is actually correct (the speaker really does say "as, as"), so it is
// effectively perfect there. If nova-3 is transcribing badly, the audio is the
// problem, not the model.
export const STT_MODELS = ["nova-3", "nova-2", "enhanced", "base"];
// Kept as a named export for /api/state; the live value comes from settings.
export const DEFAULT_STT_MODEL = getSettings().stt;

// Vocabulary the recogniser should expect. Nova-3's `keyterm` prompting biases
// decoding toward these. Always boost barge-in commands; env adds more.
// (Nova-2 used `keywords`; nova-3 uses `keyterm`.) Optional `:boost` per term.
const DEFAULT_KEYTERMS = ["stop:2", "wait:1.5", "cancel:1.5", "quiet", "enough"];
const KEYTERMS = [
  ...DEFAULT_KEYTERMS,
  ...(process.env.STT_KEYTERMS || "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean),
];

// The browser reports the sample rate it actually got from the audio hardware —
// requesting 16 kHz is only a hint, so don't hardcode it here.
function deepgramParams(sampleRate, model) {
  const params = new URLSearchParams({
    model,
    encoding: "linear16",
    sample_rate: String(sampleRate),
    channels: "1",
    interim_results: "true", // also required for UtteranceEnd to be emitted
    smart_format: "true",
    punctuate: "true",
    vad_events: "true", // SpeechStarted, used for barge-in
    // Per-word speaker labels. Deepgram clusters voices but has no idea which
    // one is the user, so the index is bound to whoever is looking at the camera
    // (server/vision.js). Indices are per-connection and reset on reconnect.
    diarize: String(process.env.STT_DIARIZE !== "false"),
    endpointing: String(ENDPOINTING_MS),
    // speech_final is not guaranteed to fire on continuous speech, so
    // UtteranceEnd is the reliable turn boundary. Minimum accepted is 1000.
    utterance_end_ms: String(UTTERANCE_END_MS),
  });
  // nova-3 renamed this; older models still take `keywords`.
  const vocabParam = model.startsWith("nova-3") ? "keyterm" : "keywords";
  for (const term of KEYTERMS) params.append(vocabParam, term);
  return params;
}

export function attachSpeechSocket(server) {
  const wss = new WebSocketServer({ server, path: "/ws/stt" });

  wss.on("connection", (browser, req) => {
    // Diarization indices are only meaningful within one Deepgram connection.
    resetVoiceBinding();
    const key = process.env.DEEPGRAM_API_KEY;
    if (!key) {
      browser.send(
        JSON.stringify({
          type: "error",
          message: "DEEPGRAM_API_KEY is not set on the server — voice input is unavailable.",
        })
      );
      browser.close();
      return;
    }

    const query = new URL(req.url, "http://localhost").searchParams;
    const requested = Number(query.get("rate"));
    const sampleRate = Number.isFinite(requested) && requested >= 8000 && requested <= 96000
      ? Math.round(requested)
      : 16000;
    const wanted = query.get("model");
    const model = STT_MODELS.includes(wanted) ? wanted : getSettings().stt;

    // STT_DUMP=1 writes the exact PCM sent upstream to disk. This is the only
    // way to answer "is the audio bad or is the recogniser wrong?" — re-transcribe
    // the dump offline and compare against what the live session produced.
    let dump = null;
    if (process.env.STT_DUMP) {
      const dir =
        process.env.STT_DUMP_DIR || path.join(process.cwd(), "recordings");
      fs.mkdirSync(dir, { recursive: true });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
      dump = createWavWriter(path.join(dir, `${stamp}-${model}.wav`), sampleRate);
      console.log(`[stt] ${model} @ ${sampleRate} Hz — recording to ${dump.file}`);
    }

    const dg = new WebSocket(`wss://api.deepgram.com/v1/listen?${deepgramParams(sampleRate, model)}`, {
      headers: { Authorization: `Token ${key}` },
    });

    // Buffer mic audio that arrives before the upstream socket is ready.
    const pending = [];
    let dgReady = false;

    dg.on("open", () => {
      dgReady = true;
      for (const chunk of pending) dg.send(chunk);
      pending.length = 0;
      browser.send(JSON.stringify({ type: "ready", model, sampleRate }));
    });

    // Deepgram finalizes speech in segments (`is_final`). A whole spoken turn
    // may span several, so segments are accumulated and flushed on a turn
    // boundary: `speech_final` when it fires, otherwise `UtteranceEnd`.
    let settled = [];
    // Turn-level speaker tally. A turn spans several is_final segments, so the
    // reported speaker is whichever index owns the most words in it.
    let speakerWords = new Map();
    let idleTimer = null;
    let graceTimer = null;

    const flushTurn = () => {
      clearTimeout(idleTimer);
      clearTimeout(graceTimer);
      idleTimer = graceTimer = null;
      const text = settled.join(" ").replace(/\s+/g, " ").trim();
      settled = [];
      let speaker = null;
      let topWords = 0;
      for (const [idx, count] of speakerWords) {
        if (count > topWords) {
          topWords = count;
          speaker = idx;
        }
      }
      speakerWords = new Map();
      if (!text) return;
      // "um" / "uh" alone are not user turns — drop them.
      if (isFillerOnly(text) && !isCommand(text)) {
        if (process.env.STT_DUMP) console.log(`[stt] drop filler: ${text}`);
        return;
      }
      if (process.env.STT_DUMP) console.log(`[stt] speaker=${speaker} final: ${text}`);
      // `speaker` has to actually reach the client: it is forwarded to
      // /api/vision/gate, which skips voice binding entirely when it is null. It was
      // tallied and logged here but never sent, so the `[stt] speaker=N` line made
      // diarization look wired up while every gate decision saw null and let other
      // voices (a TV, a phone) through as if they were the bound speaker.
      browser.send(JSON.stringify({ type: "final", text, speaker }));
    };

    // Deepgram thinks the turn ended. If the transcript trails off mid-thought
    // or is a filler/opener, hold so more speech can append to `settled`.
    const endTurn = () => {
      if (!settled.length) return;
      const joined = settled.join(" ");
      if (!graceTimer && looksUnfinished(joined)) {
        graceTimer = setTimeout(flushTurn, CONTINUATION_GRACE_MS);
        return;
      }
      flushTurn();
    };

    // Any new words mean the speaker is still going: drop a pending hold and
    // re-arm the backstop for when Deepgram reports no boundary at all.
    const onTranscript = () => {
      clearTimeout(graceTimer);
      graceTimer = null;
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (!settled.length) return;
        // Prefer grace for unfinished text even on the idle backstop.
        if (looksUnfinished(settled.join(" "))) {
          endTurn();
        } else {
          flushTurn();
        }
      }, IDLE_FLUSH_MS);
    };

    dg.on("message", (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return;
      }

      if (msg.type === "Results") {
        const transcript = msg.channel?.alternatives?.[0]?.transcript ?? "";
        if (transcript) onTranscript();
        if (msg.is_final) {
          if (transcript) settled.push(transcript);
          for (const w of msg.channel?.alternatives?.[0]?.words || []) {
            if (w.speaker === undefined) continue;
            speakerWords.set(w.speaker, (speakerWords.get(w.speaker) || 0) + 1);
          }
          if (msg.speech_final) endTurn();
          else if (settled.length) {
            browser.send(JSON.stringify({ type: "partial", text: settled.join(" ") }));
          }
        } else if (transcript) {
          // Show settled text plus the in-flight guess.
          browser.send(
            JSON.stringify({ type: "partial", text: [...settled, transcript].join(" ") })
          );
        }
      } else if (msg.type === "UtteranceEnd") {
        endTurn();
      } else if (msg.type === "SpeechStarted") {
        browser.send(JSON.stringify({ type: "speech_started" }));
      }
    });

    dg.on("error", (err) => {
      browser.send(JSON.stringify({ type: "error", message: `Deepgram: ${err.message}` }));
    });
    dg.on("close", () => {
      flushTurn(); // don't drop a trailing segment that never hit a boundary
      if (browser.readyState === WebSocket.OPEN) {
        browser.send(JSON.stringify({ type: "closed" }));
        browser.close();
      }
    });

    browser.on("message", (chunk, isBinary) => {
      if (isBinary) {
        dump?.write(chunk);
        if (dgReady) dg.send(chunk);
        else if (pending.length < 200) pending.push(chunk);
        return;
      }

      // Control frame. "finish" means the user stopped talking: ask Deepgram to
      // flush, but keep this socket open so the trailing transcript can still be
      // delivered. Deepgram emits its last speech_final only after CloseStream,
      // so closing here would drop the final utterance.
      let control;
      try {
        control = JSON.parse(chunk.toString());
      } catch {
        return;
      }
      if (control?.type === "finish" && dg.readyState === WebSocket.OPEN) {
        dg.send(JSON.stringify({ type: "CloseStream" }));
      } else if (control?.type === "finalize" && dg.readyState === WebSocket.OPEN) {
        // The client's local VAD heard the user stop. Force Deepgram to finalize
        // buffered audio now instead of waiting out `endpointing` — that is what
        // made turn ends feel like a multi-second pause. Deepgram's own Finalize
        // message exists for exactly this.
        dg.send(JSON.stringify({ type: "Finalize" }));
      }
    });

    browser.on("close", () => {
      if (dump) {
        const secs = dump.end();
        console.log(`[stt] saved ${secs.toFixed(1)}s -> ${dump.file}`);
      }
      if (dg.readyState === WebSocket.OPEN) dg.close();
      else dg.terminate();
    });
  });

  return wss;
}

// ---------------------------------------------------------------------------
// Text-to-speech: ElevenLabs Flash v2.5, streamed straight through to the browser.
// ---------------------------------------------------------------------------


export function registerTtsRoute(app) {
  app.post("/api/tts", async (req, res) => {
    const text = (req.body?.text || "").trim();
    if (!text) return res.status(400).json({ error: "text required" });

    const key = process.env.ELEVENLABS_API_KEY;
    if (!key) return res.status(503).json({ error: "ELEVENLABS_API_KEY is not set" });

    try {
      // The with-timestamps variant returns NDJSON: each line carries a slice of
      // base64 audio and, on some lines, character-level alignment. Those times
      // are absolute across the whole response, so the arrays just concatenate.
      // They're what lets the client know exactly how far playback got when the
      // user interrupts.
      const upstream = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${getSettings().voice}/stream/with-timestamps` +
          `?output_format=mp3_44100_128`,
        {
          method: "POST",
          headers: { "xi-api-key": key, "content-type": "application/json" },
          body: JSON.stringify({
            text,
            model_id: getSettings().tts,
            voice_settings: { stability: 0.4, similarity_boost: 0.75, speed: 1.05 },
          }),
        }
      );

      if (!upstream.ok) {
        const detail = await upstream.text();
        console.error("ElevenLabs error:", upstream.status, detail);
        return res.status(502).json({ error: `TTS failed (${upstream.status})` });
      }

      const audio = [];
      const characters = [];
      const starts = [];
      const ends = [];

      for (const line of (await upstream.text()).split("\n")) {
        if (!line.trim()) continue;
        let part;
        try {
          part = JSON.parse(line);
        } catch {
          continue; // partial line at a chunk boundary
        }
        if (part.audio_base64) audio.push(Buffer.from(part.audio_base64, "base64"));
        // `alignment` maps to the characters we sent; `normalized_alignment`
        // maps to ElevenLabs' normalised text, whose indices wouldn't line up.
        const a = part.alignment;
        if (a?.characters?.length) {
          characters.push(...a.characters);
          starts.push(...a.character_start_times_seconds);
          ends.push(...a.character_end_times_seconds);
        }
      }

      res.setHeader("Cache-Control", "no-store");
      res.json({
        audio: Buffer.concat(audio).toString("base64"),
        characters,
        starts,
        ends,
      });
    } catch (err) {
      console.error("tts error:", err);
      if (!res.headersSent) res.status(500).json({ error: String(err) });
      else res.end();
    }
  });
}
