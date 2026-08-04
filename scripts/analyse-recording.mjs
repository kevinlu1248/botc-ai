// Timeline of a captured session: what Silero thought, half-second by half-second.
// Used to locate the physical taps in a real recording and see how the trained VAD
// scored them — as opposed to how Deepgram's energy VAD did.
//
//   node scripts/analyse-recording.mjs recordings/foo.wav [startSec] [endSec]
import fs from "node:fs";

const SR = 16000;
const FRAME = 512;
const CONTEXT = 64;

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
  return async (frame) => {
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
  };
}

function readWav(file) {
  const buf = fs.readFileSync(file);
  let pos = 12;
  let fmt = null;
  let data = null;
  while (pos + 8 <= buf.length) {
    const id = buf.toString("ascii", pos, pos + 4);
    const size = buf.readUInt32LE(pos + 4);
    if (id === "fmt ") fmt = { channels: buf.readUInt16LE(pos + 10), rate: buf.readUInt32LE(pos + 12) };
    else if (id === "data") {
      // A recording that is still open (or was never closed cleanly) reports a
      // data size of 0, which silently yields an empty file. Use the remainder.
      const avail = buf.length - (pos + 8);
      data = buf.subarray(pos + 8, pos + 8 + (size > 0 && size <= avail ? size : avail));
      break;
    }
    pos += 8 + size + (size % 2);
  }
  if (!fmt || !data) throw new Error(`${file}: no fmt/data chunk`);
  const n = Math.floor(data.length / 2 / fmt.channels);
  const mono = new Float32Array(n);
  for (let i = 0; i < n; i++) mono[i] = data.readInt16LE(i * fmt.channels * 2) / 32768;
  return { mono, rate: fmt.rate };
}

const file = process.argv[2];
const startSec = Number(process.argv[3] ?? 0);
const endSec = Number(process.argv[4] ?? Infinity);

const { mono, rate } = readWav(file);
if (rate !== SR) throw new Error(`expected ${SR}Hz, got ${rate}`);
const from = Math.floor(startSec * SR);
const to = Math.min(mono.length, endSec === Infinity ? mono.length : Math.floor(endSec * SR));
const samples = mono.subarray(from, to);
console.log(`${file}  ${(samples.length / SR).toFixed(1)}s from ${startSec}s\n`);

const process_ = await loadSilero();
const frames = [];
for (let off = 0; off + FRAME <= samples.length; off += FRAME) {
  const f = samples.subarray(off, off + FRAME);
  let sum = 0;
  for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
  frames.push({ prob: await process_(f), rms: Math.sqrt(sum / f.length) });
}

// Half-second buckets: ~15.6 frames each.
const PER = Math.round(0.5 / (FRAME / SR));
const buckets = [];
for (let i = 0; i < frames.length; i += PER) {
  const b = frames.slice(i, i + PER);
  buckets.push({
    t: startSec + (i * FRAME) / SR,
    maxRms: Math.max(...b.map((f) => f.rms)),
    maxProb: Math.max(...b.map((f) => f.prob)),
    speechFrames: b.filter((f) => f.prob >= 0.5).length,
  });
}

// Only print buckets with something in them, so a long session stays readable.
const LOUD = 0.02;
console.log("   time   maxRms  maxProb  speechFrames  verdict");
for (const b of buckets) {
  if (b.maxRms < LOUD && b.maxProb < 0.5) continue;
  const transient = b.maxRms >= LOUD && b.maxProb < 0.5;
  const verdict = transient
    ? "LOUD but Silero says not speech  <-- transient (tap/click/bang)"
    : b.speechFrames >= 4
      ? "speech (would fire barge-in)"
      : "brief speech-ish";
  console.log(
    `${b.t.toFixed(1).padStart(7)}s  ${b.maxRms.toFixed(4)}   ${b.maxProb.toFixed(2)}   ` +
      `${String(b.speechFrames).padStart(10)}  ${verdict}`
  );
}

const transients = buckets.filter((b) => b.maxRms >= LOUD && b.maxProb < 0.5);
const speech = buckets.filter((b) => b.speechFrames >= 4);
console.log(
  `\n${transients.length} loud-but-not-speech bucket(s), ${speech.length} speech bucket(s).\n` +
    `Silero correctly rejects the transients; anything that interrupted the assistant\n` +
    `during them came from a detector other than Silero.`
);
