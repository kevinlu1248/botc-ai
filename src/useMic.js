import { useCallback, useEffect, useRef, useState } from "react";
import { VAD, frameFeatures, isProfileFrame } from "./vad.js";
import { createProfiler } from "./profile.js";
import { createSileroVad, SILERO } from "./silero.js";
import { createBargeInGate } from "./bargein.js";
import { reportEvent } from "./report.js";

// Once Silero has heard silence for this long after speech, the turn is over. The
// client says so explicitly (Deepgram `Finalize`) instead of waiting out
// server-side `endpointing`, which is what made turn ends feel like a long pause.
const VAD_END_MS = 480;

// Auto-gain is OFF by default. It reacts to whatever is loudest, so intermittent
// background bangs (a door, a pool table) make it duck — which pushes quiet
// speech further down exactly when you need it up. The conditioning chain below
// does the levelling instead, with a fast limiter that tames transients without
// turning speech down. Noise suppression stays on (it helps with steady room
// hum). Echo cancellation must stay on — barge-in depends on it.
// Override either with VITE_MIC_NOISE_SUPPRESSION / VITE_MIC_AGC.
const flag = (name, fallback) => {
  const v = import.meta.env?.[name];
  return v === undefined ? fallback : v !== "false" && v !== false;
};

// Every one of these is an `ideal` hint. A bare value (channelCount: 1) is a
// *required* constraint, and devices that can't honour it reject the whole
// request with OverconstrainedError — many interfaces won't do mono.
const AUDIO_HINTS = {
  channelCount: { ideal: 1 },
  echoCancellation: { ideal: true },
  noiseSuppression: { ideal: flag("VITE_MIC_NOISE_SUPPRESSION", true) },
  autoGainControl: { ideal: flag("VITE_MIC_AGC", false) },
};

// Barge-in runs on Silero VAD over the raw tap (see silero.js). Deepgram's own
// VAD can't be trusted during playback — it hears the playback too.

// Makeup gain applied after compression. 4x suits a laptop mic at arm's length;
// lower it for a headset, where the raw level is already healthy.
const MIC_GAIN = Number(import.meta.env?.VITE_MIC_GAIN) || 4;
// Set VITE_MIC_CONDITIONING=false to send the raw mic instead, for comparison.
const CONDITIONING = flag("VITE_MIC_CONDITIONING", true);

// A final that lands while the gate is shut is usually TTS echo — but it can also
// be the front half of an interruption, because the server completes command
// words ("stop") immediately as their own turn. Discarding it outright loses the
// "stop" from "stop here". So hold it briefly, and keep it only if a real barge-in
// follows: that is what distinguishes the user's voice from our own playback.
const HELD_STITCH_MS = 2500; // max age to prepend onto the next final
const HELD_ALONE_MS = 1200; // deliver held text alone if nothing follows

// Acoustic profiling is per-utterance, not per-session. A gap this long between
// qualifying frames means a new source is talking, and the cap stops one long
// stretch of background audio from dominating the average even without a gap.
const PROFILE_GAP_MS = 1200;
const PROFILE_MAX_FRAMES = 200; // ~5s of qualifying frames at the 25ms poll

async function openStream(deviceId) {
  const attempts = [
    { audio: deviceId ? { ...AUDIO_HINTS, deviceId: { exact: deviceId } } : AUDIO_HINTS },
    // Fall back to the loosest request that still respects the chosen device.
    { audio: deviceId ? { deviceId: { ideal: deviceId } } : true },
  ];
  let lastError;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      lastError = err;
      if (err?.name !== "OverconstrainedError" && err?.name !== "NotReadableError") throw err;
    }
  }
  throw lastError;
}

// Captures the mic as 16 kHz PCM, streams it to /ws/stt, and surfaces Deepgram
// transcripts. Also exposes an AnalyserNode for the level meter, and the list of
// available input devices.
export function useMic({ onFinal, onSpeechStart }) {
  const [state, setState] = useState("idle"); // idle | connecting | live | error
  const [partial, setPartial] = useState("");
  const [error, setError] = useState(null);
  const [muted, setMutedState] = useState(false);
  // True once the audio graph is running, independent of the upstream socket —
  // so the level meter still proves the mic works if the relay is unreachable.
  const [capturing, setCapturing] = useState(false);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(""); // "" = system default
  const [sttModel, setSttModel] = useState(""); // "" = whatever the server defaults to
  const [vadBackend, setVadBackend] = useState("loading"); // silero once loaded

  const refs = useRef({
    ws: null,
    ctx: null,
    stream: null,
    node: null,
    source: null,
    chain: null,
    muted: false,
    playing: false,
    held: null, // { text, at } captured while gated
    profiler: null, // per-utterance acoustic profile (raw mic)
    vad: null, // Silero session; null until loaded, or if loading failed
    vadNode: null, // raw-tap worklet feeding it
    vadBusy: false, // inference in flight — drop frames rather than queue
    bargeIn: createBargeInGate(), // sole authority for interrupting the assistant
    vadSpeaking: false,
    vadSpokeAt: 0, // last frame that contained speech
    vadFinalized: true, // already asked Deepgram to finalize this turn
    heldTimer: null,
    interruptedAt: 0, // assistant audio is actually coming out of the speakers
  });
  // Read directly by the visualizer's animation loop (conditioned signal).
  const analyserRef = useRef(null);
  // Raw mic, for acoustic profiling — pre-compression, because the dynamics that
  // separate a person from a loudspeaker are exactly what the compressor flattens.
  const vadAnalyserRef = useRef(null);

  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;
  const onSpeechStartRef = useRef(onSpeechStart);
  onSpeechStartRef.current = onSpeechStart;

  // Every barge-in path goes through here so the timestamp is always recorded —
  // held text is only trusted when a real interruption backs it up.
  const fireInterrupt = useCallback(() => {
    refs.current.interruptedAt = Date.now();
    onSpeechStartRef.current?.();
  }, []);

  // Deliver a final, stitching on anything held from behind the gate.
  const deliverFinal = useCallback((text, speaker) => {
    const r = refs.current;
    clearTimeout(r.heldTimer);
    r.heldTimer = null;

    let out = (text || "").trim();
    const held = r.held;
    r.held = null;
    if (held) {
      const fresh = Date.now() - held.at < HELD_STITCH_MS;
      // Slack: the VAD needs ~150ms of speech to confirm, so the interrupt is
      // stamped slightly *after* the words that triggered it were finalized.
      const backedByInterrupt = r.interruptedAt >= held.at - 1000;
      if (fresh && backedByInterrupt && held.text && !out.startsWith(held.text)) {
        out = `${held.text} ${out}`.replace(/\s+/g, " ").trim();
        // The held half carries the speaker label when the tail didn't.
        if (speaker === null || speaker === undefined) speaker = held.speaker;
      }
    }
    const profile = r.profiler?.take() ?? null;
    if (out) {
      // Makes a starved profiler visible instead of it looking like silence.
      const stats = r.profStats;
      reportEvent("profile", {
        speaker: speaker ?? null,
        profileFrames: profile?.frames ?? 0,
        qualifyingFrames: stats?.ok ?? 0,
        framesSeen: stats?.seen ?? 0,
        maxRawRms: Number((stats?.maxRms ?? 0).toFixed(4)),
        rmsFloor: VAD.PROFILE_RMS_MIN,
      });
      r.profStats = { seen: 0, ok: 0, maxRms: 0 };
      onFinalRef.current?.(out, { speaker: speaker ?? null, profile });
    }
  }, []);

  // The mic is never closed or silenced — audio always flows upstream, so
  // nothing is clipped when you start talking over the assistant. What this
  // gates is *delivery*: while the assistant is responding, transcripts are
  // dropped rather than submitted, so playback leaking through the speakers
  // can't become the next turn. Relies on the browser's echo canceller;
  // headphones make it airtight.
  const setMuted = useCallback((value) => {
    refs.current.muted = value;
    setMutedState(value);
  }, []);

  // Barge-in is only meaningful while audio is playing. Arming it during the
  // thinking phase let room noise "interrupt" a reply that hadn't started, which
  // truncated the turn to nothing.
  const setPlaying = useCallback((value) => {
    refs.current.playing = value;
  }, []);

  const refreshDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      // Before permission is granted browsers return placeholder entries with
      // blank ids and labels; those aren't selectable, so drop them.
      setDevices(all.filter((d) => d.kind === "audioinput" && d.deviceId));
    } catch {
      /* enumeration can fail before any permission is granted */
    }
  }, []);

  useEffect(() => {
    refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshDevices);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", refreshDevices);
  }, [refreshDevices]);

  // hard=true tears the socket down immediately (device switch / unmount).
  // hard=false asks the server to flush so a trailing transcript still arrives.
  const teardown = useCallback((hard) => {
    const r = refs.current;

    r.node?.disconnect();
    r.source?.disconnect();
    analyserRef.current?.disconnect();
    analyserRef.current = null;
    vadAnalyserRef.current?.disconnect();
    vadAnalyserRef.current = null;
    r.chain?.forEach((n) => n.disconnect());
    r.chain = null;
    r.stream?.getTracks().forEach((t) => t.stop());
    if (r.ctx && r.ctx.state !== "closed") r.ctx.close();
    r.node = r.source = r.stream = r.ctx = null;

    if (r.ws) {
      if (hard) {
        if (r.ws.readyState <= WebSocket.OPEN) r.ws.close();
        r.ws = null;
      } else if (r.ws.readyState === WebSocket.OPEN) {
        r.ws.send(JSON.stringify({ type: "finish" }));
        const ws = r.ws;
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) ws.close();
        }, 3000);
      } else {
        r.ws = null;
      }
    }

    clearTimeout(r.heldTimer);
    r.heldTimer = null;
    r.held = null;
    r.vadNode?.disconnect();
    r.vadNode = null;
    r.bargeIn.reset();
    r.vadSpeaking = false;
    r.vadSpokeAt = 0;
    r.vadFinalized = true;
    r.profiler = null;

    setPartial("");
    setCapturing(false);
    setState("idle");
  }, []);

  const stop = useCallback(() => teardown(false), [teardown]);

  const start = useCallback(
    async (overrideDeviceId, overrideModel) => {
      const wanted = overrideDeviceId ?? deviceId;
      setError(null);
      setMuted(false);
      setState("connecting");

      try {
        const stream = await openStream(wanted);
        refs.current.stream = stream;
        refreshDevices(); // labels are only populated once permission is granted

        // 16 kHz avoids resampling, but not every browser/device will grant it.
        // Whatever rate we actually get is what we tell Deepgram.
        let ctx;
        try {
          ctx = new AudioContext({ sampleRate: 16000 });
        } catch {
          ctx = new AudioContext();
        }
        refs.current.ctx = ctx;
        await ctx.audioWorklet.addModule("/pcm-worklet.js");

        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const query = new URLSearchParams({ rate: String(Math.round(ctx.sampleRate)) });
        if (overrideModel ?? sttModel) query.set("model", overrideModel ?? sttModel);
        const ws = new WebSocket(`${proto}//${location.host}/ws/stt?${query}`);
        ws.binaryType = "arraybuffer";
        refs.current.ws = ws;

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          // While the assistant is responding, transcripts are most likely its
          // own voice leaking through the speakers — drop them. Real user speech
          // trips the local barge-in detector below, which lifts this gate.
          const gated = refs.current.muted;
          if (msg.type === "partial") {
            setPartial(gated ? "" : msg.text);
          } else if (msg.type === "final") {
            setPartial("");
            const text = (msg.text || "").trim();
            if (!text) {
              // nothing to do
            } else if (gated) {
              // Hold it: this may be the leading words of an interruption.
              const r = refs.current;
              r.held = {
                text: r.held ? `${r.held.text} ${text}` : text,
                speaker: msg.speaker ?? r.held?.speaker ?? null,
                at: Date.now(),
              };
              clearTimeout(r.heldTimer);
              // If the user interrupted and then said nothing more, the held
              // words *were* the whole utterance — deliver them rather than
              // silently dropping. If no interrupt backed them, they were echo.
              r.heldTimer = setTimeout(() => {
                const h = refs.current.held;
                refs.current.held = null;
                if (h && refs.current.interruptedAt >= h.at - 1000) {
                  onFinalRef.current?.(h.text, {
                    speaker: h.speaker ?? null,
                    profile: refs.current.profiler?.take() ?? null,
                  });
                }
              }, HELD_ALONE_MS);
            } else {
              deliverFinal(text, msg.speaker ?? null);
            }
          } else if (msg.type === "speech_started") {
            // Deliberately does NOT interrupt. This is Deepgram's energy-based VAD:
            // it fires on a tap on the laptop, a door, a siren. It used to call
            // fireInterrupt() directly, which bypassed Silero completely and was
            // the real cause of the "overly sensitive VAD" false barge-ins —
            // Silero scores those same transients at 0.01-0.05.
            //
            // Barge-in has one authority now (src/bargein.js). Logged, not acted on,
            // so the value of this signal stays visible.
            if (refs.current.playing) {
              reportEvent("bargein-ignored", {
                source: "deepgram-vad",
                trail: refs.current.bargeIn?.trail() ?? [],
              });
            }
          } else if (msg.type === "ready") {
            setState("live");
          } else if (msg.type === "error") {
            setError(msg.message);
            setState("error");
          }
        };
        ws.onerror = () => {
          setError(
            "Can't reach the speech service. Check that the API server is running on port 3001 " +
              "and that Vite is proxying /ws (needs `ws: true` in vite.config.js)."
          );
          setState("error");
        };
        ws.onclose = () => {
          if (refs.current.ws === ws) refs.current.ws = null;
          setState((s) => (s === "error" ? s : "idle"));
        };

        const source = ctx.createMediaStreamSource(stream);
        refs.current.source = source;

        // --- conditioning chain: quiet speech up, background transients down ---
        // Everything downstream of `tail` is what Deepgram actually receives.
        let tail = source;
        if (CONDITIONING) {
          // Rumble and thump energy sits below speech; a pool ball or a desk
          // knock is mostly here, and none of it carries information.
          const highpass = ctx.createBiquadFilter();
          highpass.type = "highpass";
          highpass.frequency.value = 90;
          highpass.Q.value = 0.7;

          // Trim above the speech band: hiss and the crack of a transient.
          const lowpass = ctx.createBiquadFilter();
          lowpass.type = "lowpass";
          lowpass.frequency.value = 7500;

          // Levelling. A low threshold with a fast attack pulls quiet speech up
          // and catches spikes before they get through, unlike browser AGC which
          // reacts by turning everything down.
          const comp = ctx.createDynamicsCompressor();
          comp.threshold.value = -34;
          comp.knee.value = 12;
          comp.ratio.value = 5;
          comp.attack.value = 0.003;
          comp.release.value = 0.22;

          const makeup = ctx.createGain();
          makeup.gain.value = MIC_GAIN;

          // Brick wall after makeup gain so nothing clips into the recogniser.
          const limiter = ctx.createDynamicsCompressor();
          limiter.threshold.value = -3;
          limiter.knee.value = 0;
          limiter.ratio.value = 20;
          limiter.attack.value = 0.001;
          limiter.release.value = 0.05;

          source.connect(highpass);
          highpass.connect(lowpass);
          lowpass.connect(comp);
          comp.connect(makeup);
          makeup.connect(limiter);
          tail = limiter;
          refs.current.chain = [highpass, lowpass, comp, makeup, limiter];
        }

        // Meter taps the conditioned signal, so what you see is what is sent.
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048; // finer bins for the low mel bands
        analyser.smoothingTimeConstant = 0.5; // meter does its own ballistics
        tail.connect(analyser);
        analyserRef.current = analyser;

        // Acoustic *profiling* taps the raw mic: speaker identity should be measured
        // before compression flattens the dynamics it keys on. This is no longer a
        // barge-in tap — vad.js is profiling-only (see its header).
        const vadAnalyser = ctx.createAnalyser();
        vadAnalyser.fftSize = 2048;
        vadAnalyser.smoothingTimeConstant = 0.5;
        source.connect(vadAnalyser);
        vadAnalyserRef.current = vadAnalyser;

        const node = new AudioWorkletNode(ctx, "pcm-worklet");
        node.port.onmessage = (event) => {
          // Always stream, even while the assistant is talking — muting here
          // would clip the start of an interrupting sentence.
          if (ws.readyState === WebSocket.OPEN) ws.send(event.data);
        };
        tail.connect(node);
        // Worklets are only pulled if they reach a destination; a zeroed gain
        // keeps that path silent so the mic isn't echoed to the speakers.
        const mute = ctx.createGain();
        mute.gain.value = 0;
        node.connect(mute).connect(ctx.destination);
        refs.current.node = node;
        refs.current.profiler = createProfiler();

        // --- Silero VAD on the raw tap ---
        // Deliberately NOT fault-tolerant. An earlier version caught load failures
        // and quietly fell back to the spectral heuristics, which meant a broken
        // wasm path degraded barge-in for an unknown length of time with nothing
        // but a console warning. If the model cannot load, the mic fails loudly.
        {
          await ctx.audioWorklet.addModule("/vad-worklet.js");
          const vadNode = new AudioWorkletNode(ctx, "vad-worklet", {
            processorOptions: { frameSize: 512 },
          });
          // Feed Silero the CONDITIONED signal — the same audio Deepgram gets.
          //
          // This was `source` (raw), which made barge-in impossible during playback:
          // the raw mic is un-amplified and browser echo cancellation ducks it while
          // TTS plays, so the tap measured 0.0003 RMS / p=0.00 while the user was
          // saying "stop" clearly enough for Deepgram to transcribe it. That silent
          // failure is why a second, energy-based detector had been bolted on.
          //
          // Measured on a real captured session (scripts/analyse-recording.mjs over
          // the conditioned stream that STT_DUMP records):
          //   user speech          p=0.73-1.00, 4-16 speech frames per bucket -> fires
          //   364 loud transients  p=0.00 even at RMS 0.36                    -> ignored
          //   assistant's own TTS  no speech buckets at all                   -> no self-interrupt
          // Amplification does not fool a trained classifier the way it fools an
          // energy threshold, which is the whole reason this can be one detector.
          tail.connect(vadNode);
          const vadMute = ctx.createGain();
          vadMute.gain.value = 0;
          vadNode.connect(vadMute).connect(ctx.destination);
          refs.current.vadNode = vadNode;

          const vad = refs.current.vad ?? (await createSileroVad());
          refs.current.vad = vad;
          vad.reset();
          setVadBackend("silero");

          vadNode.port.onmessage = async (event) => {
            const r = refs.current;
            if (!r.vad || r.vadBusy) return; // never queue: stale frames are worse
            r.vadBusy = true;
            try {
              const prob = await r.vad.process(event.data);
              let sum = 0;
              for (let i = 0; i < event.data.length; i++) sum += event.data[i] * event.data[i];
              const frameRms = Math.sqrt(sum / event.data.length);

              // Hysteresis, so a wavering score can't flap mid-syllable.
              r.vadSpeaking = r.vadSpeaking
                ? prob >= SILERO.EXIT
                : prob >= SILERO.ENTER;

              // End-of-turn: local silence after speech finalizes immediately.
              const now = performance.now();
              if (r.vadSpeaking) {
                r.vadSpokeAt = now;
                r.vadFinalized = false;
              } else if (
                !r.vadFinalized &&
                r.vadSpokeAt &&
                now - r.vadSpokeAt >= VAD_END_MS &&
                !r.muted
              ) {
                r.vadFinalized = true;
                if (r.ws?.readyState === WebSocket.OPEN) {
                  r.ws.send(JSON.stringify({ type: "finalize" }));
                }
              }

              // The single barge-in authority. Every frame goes in; the gate
              // decides, so there is one place to reason about and one place to log.
              const hit = r.bargeIn.consider({
                speaking: r.vadSpeaking,
                prob,
                rms: frameRms,
                armed: r.playing,
              });
              if (hit) {
                reportEvent("bargein", hit);
                fireInterrupt();
              }
            } finally {
              r.vadBusy = false;
            }
          };
        }

        setCapturing(true);
      } catch (err) {
        setError(
          err?.name === "NotAllowedError"
            ? "Microphone permission denied."
            : `Mic failed to start: ${err?.message || err}`
        );
        setState("error");
        teardown(true);
      }
    },
    [deviceId, sttModel, refreshDevices, setMuted, teardown, deliverFinal, fireInterrupt]
  );

  // Switching device mid-session needs a hard restart of the audio graph.
  const selectDevice = useCallback(
    async (id) => {
      setDeviceId(id);
      if (refs.current.stream) {
        teardown(true);
        await start(id);
      }
    },
    [start, teardown]
  );

  // The model is fixed for the life of a Deepgram connection, so changing it
  // means reconnecting.
  const selectSttModel = useCallback(
    async (model) => {
      setSttModel(model);
      if (refs.current.stream) {
        teardown(true);
        await start(undefined, model);
      }
    },
    [start, teardown]
  );

  // Builds the per-utterance speaker profile from the RAW mic (see profile.js):
  // the conditioning chain compresses dynamics and band-limits, destroying the
  // very features that distinguish a person from a phone speaker. Skipped while
  // the assistant is audible so its own voice never enters the profile.
  useEffect(() => {
    if (!capturing) return;
    let buf = null;
    let spectrumDb = null;
    let power = null;

    const tick = () => {
      const analyser = vadAnalyserRef.current;
      if (!analyser) return;
      // Only profile while nothing is playing, so our own TTS never contaminates it.
      if (refs.current.playing) return;
      if (!buf || buf.length !== analyser.fftSize) {
        buf = new Uint8Array(analyser.fftSize);
        spectrumDb = new Float32Array(analyser.frequencyBinCount);
        power = new Float32Array(analyser.frequencyBinCount);
      }
      analyser.getByteTimeDomainData(buf);
      analyser.getFloatFrequencyData(spectrumDb);
      for (let i = 0; i < spectrumDb.length; i++) {
        power[i] = Number.isFinite(spectrumDb[i]) ? 10 ** (spectrumDb[i] / 10) : 0;
      }

      const features = frameFeatures(buf, power, analyser.context.sampleRate, analyser.fftSize);

      // Why this is measured: the profiler needs 25 qualifying frames before
      // take() returns anything, and it qualifies frames off the RAW tap against
      // VAD.RMS_MIN. The raw mic is quiet on this hardware, so "profile is null"
      // and "the user did not speak" look identical without these counters.
      const stats = (refs.current.profStats ??= { seen: 0, ok: 0, maxRms: 0 });
      stats.seen++;
      stats.maxRms = Math.max(stats.maxRms, features.rms);

      if (isProfileFrame(features)) {
        // Keep the profile about the CURRENT utterance. take() used to be the only
        // reset, so everything heard since the last final was averaged together —
        // measured 895 profile frames drawn from 30,359 polls (~12 minutes), which
        // blended a TV playing in the room into "the user's" voice and collapsed the
        // distance between them (TV 1.68 vs user 1.19-1.70: no separation at all).
        const now = performance.now();
        const gap = now - (refs.current.profLastAt || 0);
        if (refs.current.profLastAt && (gap > PROFILE_GAP_MS || stats.ok >= PROFILE_MAX_FRAMES)) {
          refs.current.profiler?.reset();
          stats.ok = 0;
        }
        refs.current.profLastAt = now;
        stats.ok++;
        refs.current.profiler?.push(
          features.rms,
          power,
          analyser.context.sampleRate,
          analyser.fftSize
        );
      }
    };

    // 25ms is ample: this only accumulates a per-utterance average, and barge-in
    // (which did need a tight loop) now runs off the Silero worklet instead.
    const id = setInterval(tick, 25);
    return () => clearInterval(id);
  }, [capturing]);

  useEffect(() => () => teardown(true), [teardown]);

  return {
    state,
    partial,
    error,
    muted,
    capturing,
    devices,
    deviceId,
    sttModel,
    vadBackend,
    analyserRef,
    start,
    stop,
    setMuted,
    setPlaying,
    selectDevice,
    selectSttModel,
    // Based on the audio graph, not the socket, so the stop button still works
    // (and the mic gets released) when the speech service is unreachable.
    listening: capturing || state === "connecting",
  };
}
