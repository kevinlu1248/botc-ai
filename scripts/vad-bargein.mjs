// Measures the barge-in gate offline, so thresholds come from numbers instead of
// from guessing in a live session.
//
// Runs the real Silero model over real recordings and over synthesised sirens and
// mouse clicks — the two false triggers reported — and reports, per source, what
// the CURRENT gate does (2 consecutive frames >= 0.5) alongside the features a
// better gate could use.
//
//   node scripts/vad-bargein.mjs [extra.wav ...]
import fs from "node:fs";
import path from "node:path";
// Import the real gate rather than reimplementing it, so this harness can never
// certify a threshold the app isn't actually using.
import { BARGE_IN, createBargeInGate } from "../src/bargein.js";

const SR = 16000;
const FRAME = 512;
const CONTEXT = 64;

// ---------------------------------------------------------------- silero (node)
async function loadSilero() {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = "error";
  const bytes = new Uint8Array(fs.readFileSync("public/models/silero_vad.onnx"));
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });

  let state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
  let context = new Float32Array(CONTEXT);
  const input = new Float32Array(CONTEXT + FRAME);
  const sr = new ort.Tensor("int64", BigInt64Array.from([BigInt(SR)]), []);

  return {
    async process(frame) {
      input.set(context, 0);
      input.set(frame, CONTEXT);
      const out = await session.run({
        input: new ort.Tensor("float32", input, [1, input.length]),
        state,
        sr,
      });
      state = out.stateN;
      context = frame.slice(-CONTEXT);
      return out.output.data[0];
    },
    reset() {
      state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
      context = new Float32Array(CONTEXT);
    },
  };
}

// ------------------------------------------------------------------ wav reading
function readWav(file) {
  const buf = fs.readFileSync(file);
  // Walk chunks rather than assuming a 44-byte header.
  let pos = 12;
  let fmt = null;
  let data = null;
  while (pos + 8 <= buf.length) {
    const id = buf.toString("ascii", pos, pos + 4);
    const size = buf.readUInt32LE(pos + 4);
    if (id === "fmt ") {
      fmt = {
        channels: buf.readUInt16LE(pos + 10),
        rate: buf.readUInt32LE(pos + 12),
        bits: buf.readUInt16LE(pos + 22),
      };
    } else if (id === "data") {
      // A still-open recording reports size 0; fall back to the rest of the file.
      const avail = buf.length - (pos + 8);
      data = buf.subarray(pos + 8, pos + 8 + (size > 0 && size <= avail ? size : avail));
      break;
    }
    pos += 8 + size + (size % 2);
  }
  if (!fmt || !data) throw new Error(`${file}: no fmt/data chunk`);
  if (fmt.bits !== 16) throw new Error(`${file}: expected 16-bit, got ${fmt.bits}`);

  const n = Math.floor(data.length / 2 / fmt.channels);
  const mono = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let c = 0; c < fmt.channels; c++) acc += data.readInt16LE((i * fmt.channels + c) * 2);
    mono[i] = acc / fmt.channels / 32768;
  }
  if (fmt.rate === SR) return mono;
  // Linear resample — adequate for VAD-rate analysis.
  const out = new Float32Array(Math.floor((n * SR) / fmt.rate));
  for (let i = 0; i < out.length; i++) {
    const t = (i * fmt.rate) / SR;
    const a = Math.floor(t);
    const frac = t - a;
    out[i] = (mono[a] ?? 0) * (1 - frac) + (mono[a + 1] ?? 0) * frac;
  }
  return out;
}

// ------------------------------------------------------------- synthetic noises
// Deterministic PRNG: workflows/tests must not depend on Math.random.
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;

function sirenTwoTone(seconds = 4, amp = 0.25) {
  // Emergency-vehicle style: alternating tones, the classic hard case for a VAD —
  // tonal, harmonic-ish, and sitting squarely in the speech band.
  const out = new Float32Array(SR * seconds);
  for (let i = 0; i < out.length; i++) {
    const t = i / SR;
    const hz = Math.floor(t * 2) % 2 ? 960 : 740;
    out[i] = amp * Math.sin(2 * Math.PI * hz * t) + amp * 0.25 * Math.sin(4 * Math.PI * hz * t);
  }
  return out;
}

function sirenSweep(seconds = 4, amp = 0.25) {
  // Wail: continuous sweep, so the spectrum is never stationary — the harder of
  // the two for any stationarity-based rejection.
  const out = new Float32Array(SR * seconds);
  let phase = 0;
  for (let i = 0; i < out.length; i++) {
    const t = i / SR;
    const hz = 700 + 500 * Math.sin(2 * Math.PI * 0.25 * t);
    phase += (2 * Math.PI * hz) / SR;
    out[i] = amp * Math.sin(phase) + amp * 0.2 * Math.sin(2 * phase);
  }
  return out;
}

function mouseClicks(seconds = 4, amp = 0.5) {
  // Broadband transients ~700ms apart, each a few ms long with a fast decay.
  const out = new Float32Array(SR * seconds);
  for (let c = 0; c < Math.floor(seconds / 0.7); c++) {
    const at = Math.floor((0.35 + c * 0.7) * SR);
    const len = Math.floor(0.004 * SR);
    for (let i = 0; i < len && at + i < out.length; i++) {
      out[at + i] = amp * rnd() * Math.exp(-i / (len / 3));
    }
  }
  return out;
}

function keyboard(seconds = 4, amp = 0.35) {
  const out = new Float32Array(SR * seconds);
  for (let c = 0; c < Math.floor(seconds / 0.18); c++) {
    const at = Math.floor((0.1 + c * 0.18) * SR);
    const len = Math.floor(0.008 * SR);
    for (let i = 0; i < len && at + i < out.length; i++) {
      out[at + i] = amp * rnd() * Math.exp(-i / (len / 4));
    }
  }
  return out;
}

// ---------------------------------------------------------------- frame features
function frameStats(frame) {
  let sum = 0;
  let crossings = 0;
  for (let i = 0; i < frame.length; i++) {
    sum += frame[i] * frame[i];
    if (i > 0 && (frame[i] >= 0) !== (frame[i - 1] >= 0)) crossings++;
  }
  return { rms: Math.sqrt(sum / frame.length), zcr: crossings / frame.length };
}

const std = (xs) => {
  if (xs.length < 2) return 0;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length);
};
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

// ------------------------------------------------------------------------- gates
const ENTER = 0.5;
const EXIT = 0.35;

/** Current shipped gate: 2 consecutive speech frames, with hysteresis. */
function currentGate(frames) {
  let speaking = false;
  let hits = 0;
  for (let i = 0; i < frames.length; i++) {
    speaking = speaking ? frames[i].prob >= EXIT : frames[i].prob >= ENTER;
    if (speaking) {
      if (++hits >= 2) return i;
    } else hits = 0;
  }
  return -1;
}

/**
 * Candidate gates. The point of comparing them on real speech is that any
 * tightening costs barge-in latency, and that cost has to be a measured number
 * rather than a guess.
 */
const GATES = {
  "old (2 consecutive)": (frames) => currentGate(frames),
  [`SHIPPED bargein.js (${BARGE_IN.MIN_FRAMES})`]: (frames) => {
    const gate = createBargeInGate();
    let speaking = false;
    for (let i = 0; i < frames.length; i++) {
      speaking = speaking ? frames[i].prob >= EXIT : frames[i].prob >= ENTER;
      if (gate.consider({ speaking, prob: frames[i].prob, rms: frames[i].rms, armed: true })) {
        return i;
      }
    }
    return -1;
  },
  "4 consecutive": (frames) => runGate(frames, { consecutive: 4 }),
  "6 consecutive": (frames) => runGate(frames, { consecutive: 6 }),
  "6 of 10 window": (frames) => runGate(frames, { window: 10, hits: 6 }),
  "6 of 10 + meanP>=0.6": (frames) => runGate(frames, { window: 10, hits: 6, meanProb: 0.6 }),
};

function runGate(frames, { consecutive = 0, window = 0, hits = 0, meanProb = 0 }) {
  let speaking = false;
  let run = 0;
  const ring = [];
  for (let i = 0; i < frames.length; i++) {
    speaking = speaking ? frames[i].prob >= EXIT : frames[i].prob >= ENTER;
    ring.push({ speaking, prob: frames[i].prob });
    if (window && ring.length > window) ring.shift();

    if (consecutive) {
      run = speaking ? run + 1 : 0;
      if (run >= consecutive) return i;
      continue;
    }
    const n = ring.filter((f) => f.speaking).length;
    if (n >= hits && (!meanProb || mean(ring.map((f) => f.prob)) >= meanProb)) return i;
  }
  return -1;
}

async function analyse(label, samples, vad) {
  vad.reset();
  const frames = [];
  for (let off = 0; off + FRAME <= samples.length; off += FRAME) {
    const frame = samples.subarray(off, off + FRAME);
    const prob = await vad.process(frame);
    frames.push({ prob, ...frameStats(frame) });
  }

  const fired = currentGate(frames);
  const probs = frames.map((f) => f.prob);
  const speechIdx = frames.map((f, i) => (f.prob >= ENTER ? i : -1)).filter((i) => i >= 0);

  // Feature spread across the 10 frames (320ms) around the moment it first fires —
  // this is what a smarter gate would get to look at.
  const W = 10;
  const at = fired >= 0 ? fired : 0;
  const win = frames.slice(Math.max(0, at - W + 1), at + 1);
  const zcrStd = std(win.map((f) => f.zcr));
  const logRms = win.map((f) => Math.log10(Math.max(1e-6, f.rms)));

  const gateResults = {};
  for (const [name, fn] of Object.entries(GATES)) {
    const idx = fn(frames);
    gateResults[name] = idx >= 0 ? Math.round((idx * FRAME * 1000) / SR) : null;
  }

  return {
    label,
    frames: frames.length,
    gateResults,
    firedFrame: fired,
    firedMs: fired >= 0 ? Math.round((fired * FRAME * 1000) / SR) : null,
    speechFrac: speechIdx.length / frames.length,
    maxProb: Math.max(...probs),
    meanProbWin: mean(win.map((f) => f.prob)),
    zcrStd,
    logRmsStd: std(logRms),
    meanRms: mean(frames.map((f) => f.rms)),
  };
}

// --------------------------------------------------------------------------- run
const vad = await loadSilero();

const SPEECH_FILES = [
  "recordings/2026-08-03_02-40-33-nova-3.wav",
  "recordings/2026-08-02_04-11-34-nova-3.wav",
  "recordings/botc-stt-2026-08-02T03-52-10-504Z-16000hz.wav",
];

const sources = [];
for (const f of [...SPEECH_FILES, ...process.argv.slice(2)]) {
  if (!fs.existsSync(f)) {
    console.log(`(missing, skipped) ${f}`);
    continue;
  }
  // Cap length so a 5-minute file doesn't dominate runtime.
  const all = readWav(f);
  sources.push({ label: `SPEECH  ${path.basename(f)}`, samples: all.subarray(0, SR * 25), speech: true });
}
// Real captured audio from the session where barge-in failed, taken from the
// CONDITIONED stream (what STT_DUMP records, and what Silero now listens to).
// These two slices are the regression test that matters: the gate must fire on the
// first and stay silent on the second.
const REAL = [
  {
    file: "recordings/2026-08-03_23-01-18-nova-3.wav",
    from: 158,
    to: 180,
    label: "SPEECH  real 'stop' during playback (conditioned)",
    speech: true,
  },
  {
    file: "recordings/2026-08-03_23-01-18-nova-3.wav",
    from: 250,
    to: 286,
    label: "NOISE   real laptop taps (conditioned)",
    speech: false,
  },
];
for (const c of REAL) {
  if (!fs.existsSync(c.file)) {
    console.log(`(missing, skipped) ${c.file}`);
    continue;
  }
  const all = readWav(c.file);
  sources.push({
    label: c.label,
    samples: all.subarray(c.from * SR, c.to * SR),
    speech: c.speech,
  });
}

sources.push({ label: "NOISE   siren (two-tone)", samples: sirenTwoTone(), speech: false });
sources.push({ label: "NOISE   siren (wail/sweep)", samples: sirenSweep(), speech: false });
sources.push({ label: "NOISE   mouse clicks", samples: mouseClicks(), speech: false });
sources.push({ label: "NOISE   keyboard typing", samples: keyboard(), speech: false });

const rows = [];
for (const s of sources) rows.push({ ...(await analyse(s.label, s.samples, vad)), speech: s.speech });

console.log(
  `\n${"source".padEnd(46)} ${"current".padEnd(10)} speech%  maxP  meanP(win)  zcrStd  logRmsStd  meanRms`
);
for (const r of rows) {
  console.log(
    `${r.label.padEnd(46)} ${(r.firedFrame >= 0 ? `FIRES ${r.firedMs}ms` : "silent").padEnd(10)} ` +
      `${(r.speechFrac * 100).toFixed(0).padStart(6)}  ${r.maxProb.toFixed(2)}  ` +
      `${r.meanProbWin.toFixed(2).padStart(10)}  ${r.zcrStd.toFixed(4)}  ${r.logRmsStd.toFixed(3).padStart(9)}  ${r.meanRms.toFixed(4)}`
  );
}

const falseFires = rows.filter((r) => !r.speech && r.firedFrame >= 0);
const missed = rows.filter((r) => r.speech && r.firedFrame < 0);
console.log(
  `\ncurrent gate: ${falseFires.length} false trigger(s) on noise, ${missed.length} missed on speech`
);
// Latency cost of each candidate, measured against the current gate on real speech.
console.log("\ngate comparison — first fire, ms (lower is more responsive):");
const names = Object.keys(GATES);
console.log(`${"source".padEnd(46)} ${names.map((n) => n.padStart(22)).join("")}`);
for (const r of rows) {
  const cells = names.map((n) => {
    const v = r.gateResults[n];
    return (v === null ? "silent" : String(v)).padStart(22);
  });
  console.log(`${r.label.padEnd(46)} ${cells.join("")}`);
}
console.log("\nadded latency vs current gate, on speech only:");
for (const n of names.slice(1)) {
  const deltas = rows
    .filter((r) => r.speech && r.gateResults[n] !== null && r.gateResults["old (2 consecutive)"] !== null)
    .map((r) => r.gateResults[n] - r.gateResults["old (2 consecutive)"]);
  const missed = rows.filter((r) => r.speech && r.gateResults[n] === null).length;
  console.log(
    `  ${n.padEnd(24)} +${Math.min(...deltas)}..${Math.max(...deltas)}ms` +
      `${missed ? `  MISSES ${missed} speech source(s)` : ""}`
  );
}

console.log("\nfeature separation (what a better gate can key on):");
for (const key of ["zcrStd", "logRmsStd", "meanProbWin"]) {
  const sp = rows.filter((r) => r.speech).map((r) => r[key]);
  const no = rows.filter((r) => !r.speech && r.firedFrame >= 0).map((r) => r[key]);
  const fmt = (xs) => (xs.length ? `${Math.min(...xs).toFixed(4)}–${Math.max(...xs).toFixed(4)}` : "n/a");
  console.log(`  ${key.padEnd(12)} speech ${fmt(sp).padEnd(20)} false-firing noise ${fmt(no)}`);
}
