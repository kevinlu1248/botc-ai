import { useCallback, useEffect, useRef, useState } from "react";
import { VAD, frameFeatures, isSpeechFrame, isHardBargeIn } from "./vad.js";

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

// Barge-in while the assistant is speaking. Deepgram's own VAD can't be trusted
// here — it's hearing the playback too — so this is measured locally, after the
// browser's echo canceller, and classified as speech-or-not by src/vad.js.
// Loudness alone used to be the test, which meant a door slam or keyboard
// clatter would "interrupt" the assistant.

// Makeup gain applied after compression. 4x suits a laptop mic at arm's length;
// lower it for a headset, where the raw level is already healthy.
const MIC_GAIN = Number(import.meta.env?.VITE_MIC_GAIN) || 4;
// Set VITE_MIC_CONDITIONING=false to send the raw mic instead, for comparison.
const CONDITIONING = flag("VITE_MIC_CONDITIONING", true);

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

  const refs = useRef({
    ws: null,
    ctx: null,
    stream: null,
    node: null,
    source: null,
    chain: null,
    muted: false,
    playing: false, // assistant audio is actually coming out of the speakers
  });
  // Read directly by the visualizer's animation loop (conditioned signal).
  const analyserRef = useRef(null);
  // Raw mic, for barge-in classification (see vad.js threshold calibration).
  const vadAnalyserRef = useRef(null);

  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;
  const onSpeechStartRef = useRef(onSpeechStart);
  onSpeechStartRef.current = onSpeechStart;

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
            if (!gated && msg.text.trim()) onFinalRef.current?.(msg.text.trim());
          } else if (msg.type === "speech_started") {
            // Only barge-in while the assistant is *audibly* playing. During
            // thinking / TTS fetch the mic is unmuted; treating room noise or
            // residual VAD as speech_started was cutting off every new reply
            // (stopped=true → feed() drops → "interrupted" with no audio).
            if (refs.current.playing && !refs.current.muted) {
              onSpeechStartRef.current?.();
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

        // Barge-in taps the *raw* mic instead: src/vad.js thresholds are
        // calibrated against unprocessed levels, and reading them off the
        // compressed signal would fire on amplified background.
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
        setCapturing(true);
      } catch (err) {
        setError(err?.name === "NotAllowedError" ? "Microphone permission denied." : String(err?.message || err));
        setState("error");
        teardown(true);
      }
    },
    [deviceId, sttModel, refreshDevices, setMuted, teardown]
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

  // While the assistant is speaking, watch the raw mic for barge-in. Deepgram
  // is muted then, so this is the only path. Poll on a short timer (not only
  // rAF) so a busy main thread can't delay detection by a full frame budget.
  useEffect(() => {
    if (!capturing) return;
    let hits = 0;
    let buf = null;
    let spectrumDb = null;
    let power = null;
    let fired = false;

    const tick = () => {
      const analyser = vadAnalyserRef.current;
      if (!analyser || !refs.current.playing) {
        hits = 0;
        fired = false;
        return;
      }
      if (fired) return; // one interrupt per playback arming
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
      // Hard path: one strong frame → cut immediately (~0–16ms).
      if (isHardBargeIn(features)) {
        hits = 0;
        fired = true;
        onSpeechStartRef.current?.();
        return;
      }
      // Soft path: two consecutive speech-like frames.
      if (isSpeechFrame(features)) {
        if (++hits >= VAD.FRAMES) {
          hits = 0;
          fired = true;
          onSpeechStartRef.current?.();
        }
      } else if (hits > 0) {
        hits = 0; // require consecutive, don't slowly decay
      }
    };

    // ~8ms polling while armed — much snappier than rAF alone under load.
    const id = setInterval(tick, 8);
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
