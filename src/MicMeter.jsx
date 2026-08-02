import { useEffect, useRef, useState } from "react";
import { buildMelBank, bandDb } from "./mel.js";

const BANDS = 28;
const PAD_X = 10; // inset so bars don't run into the rounded border
const PAD_Y = 8;

// Display range in dB. Below the floor is drawn as silence; the ceiling is where
// bars top out. Speech through the browser's auto-gain typically peaks near -35.
const DB_FLOOR = -85;
const DB_CEIL = -35;

// Meter ballistics, borrowed from hardware VU meters: fast attack so onsets are
// visible, slow release so the motion reads as smooth rather than strobing.
const ATTACK = 0.5;
const RELEASE = 0.12;

// Level readout (time domain RMS, the standard way to measure loudness).
// Two thresholds, because "quiet room" and "dead microphone" differ: a quiet
// room still shows ambient RMS around 0.005–0.04, a dead device sits at zero.
const DEAD_FLOOR = 0.003;
const VOICE_FLOOR = 0.02;

// Live mel-spectrum meter. Reads the AnalyserNode from useMic's ref inside a rAF
// loop, so audio frames never trigger a React render.
export default function MicMeter({ analyserRef, active, muted, status }) {
  const canvasRef = useRef(null);
  const [level, setLevel] = useState("quiet"); // voice | quiet | dead

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let raf = 0;
    let spectrumDb = null;
    let waveform = null;
    let bank = null;
    let bankKey = "";
    let decayingPeak = 0;
    let frames = 0;
    const heights = new Float32Array(BANDS);

    const draw = () => {
      raf = requestAnimationFrame(draw);

      // Size the backing store to the CSS box, accounting for DPR.
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const styles = getComputedStyle(canvas);
      const idleColor = styles.getPropertyValue("--meter-idle").trim() || "#30363d";
      const liveColor = styles.getPropertyValue("--meter-live").trim() || "#3fb950";

      const innerW = Math.max(1, w - PAD_X * 2);
      const maxH = Math.max(2, h - PAD_Y * 2);
      const gap = 3;
      const barW = Math.max(2, (innerW - gap * (BANDS - 1)) / BANDS);
      const radius = Math.min(barW / 2, 2);
      const bar = (i, height) => {
        const x = PAD_X + i * (barW + gap);
        ctx.beginPath();
        ctx.roundRect(x, (h - height) / 2, barW, height, radius);
        ctx.fill();
      };

      const analyser = analyserRef.current;
      if (!analyser || !active) {
        heights.fill(0);
        ctx.fillStyle = idleColor;
        for (let i = 0; i < BANDS; i++) bar(i, 2);
        return;
      }

      const { fftSize, frequencyBinCount } = analyser;
      const sampleRate = analyser.context.sampleRate;
      const key = `${sampleRate}:${fftSize}`;
      if (bankKey !== key) {
        bank = buildMelBank({ sampleRate, fftSize, bands: BANDS });
        bankKey = key;
      }
      if (!spectrumDb || spectrumDb.length !== frequencyBinCount) {
        spectrumDb = new Float32Array(frequencyBinCount);
        waveform = new Uint8Array(fftSize);
      }

      // Float data is already in dB — no byte quantisation to undo.
      analyser.getFloatFrequencyData(spectrumDb);
      analyser.getByteTimeDomainData(waveform);

      let sum = 0;
      for (let i = 0; i < waveform.length; i++) {
        const v = (waveform[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / waveform.length);

      frames++;
      decayingPeak = Math.max(rms, decayingPeak * 0.985);
      if (frames > 120 && decayingPeak < DEAD_FLOOR) setLevel("dead");
      else if (decayingPeak > VOICE_FLOOR) setLevel("voice");
      else setLevel("quiet");

      ctx.fillStyle = liveColor;
      for (let i = 0; i < BANDS; i++) {
        const db = bandDb(spectrumDb, bank[i]);
        // Normalise the dB range to 0..1; the floor doubles as the noise gate.
        let mag = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR);
        mag = Math.min(1, Math.max(0, Number.isFinite(mag) ? mag : 0));
        if (muted) mag *= 0.4;

        const k = mag > heights[i] ? ATTACK : RELEASE;
        heights[i] += (mag - heights[i]) * k;

        ctx.globalAlpha = muted ? 0.45 : 0.6 + heights[i] * 0.4;
        bar(i, Math.max(2, heights[i] * maxH));
      }
      ctx.globalAlpha = 1;
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [analyserRef, active, muted]);

  const hint =
    status === "thinking"
      ? "thinking…"
      : status === "speaking"
        ? "speaking…"
        : !active
          ? "mic off"
          : level === "voice"
            ? "hearing you"
            : level === "dead"
              ? "no signal — check device"
              : "listening";

  const tone = status || (active && !muted ? level : "");

  return (
    <div className="meter">
      <canvas ref={canvasRef} />
      <span className={`meter-hint ${tone}`}>{hint}</span>
    </div>
  );
}
