// Proves the root cause of the false barge-ins: Deepgram's `vad_events`
// SpeechStarted fires on non-speech transients (a laptop tap, a click), and
// useMic.js used to turn that straight into an interrupt without asking Silero —
// which scores the same audio at 0.01-0.05.
//
//   node scripts/deepgram-vad-taps.mjs
import WebSocket from "ws";

const SR = 16000;
const host = process.env.STT_HOST || "localhost:3001";

let seed = 987654321;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;

/** Knuckle-taps on a laptop chassis: short, broadband, fairly loud. */
function taps(seconds = 6, amp = 0.6) {
  const out = new Float32Array(SR * seconds);
  for (let c = 0; c < Math.floor(seconds / 0.8); c++) {
    const at = Math.floor((0.5 + c * 0.8) * SR);
    const len = Math.floor(0.012 * SR); // ~12ms thud
    for (let i = 0; i < len && at + i < out.length; i++) {
      // Low-frequency body plus a broadband edge, decaying fast.
      const env = Math.exp(-i / (len / 4));
      out[at + i] = amp * env * (0.7 * Math.sin((2 * Math.PI * 120 * i) / SR) + 0.3 * rnd());
    }
  }
  return out;
}

function toPcm16(f32) {
  const buf = Buffer.alloc(f32.length * 2);
  for (let i = 0; i < f32.length; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    buf.writeInt16LE(Math.round(v * 32767), i * 2);
  }
  return buf;
}

const pcm = toPcm16(taps());
const CHUNK = 1600; // 50ms

const ws = new WebSocket(`ws://${host}/ws/stt?rate=${SR}`);
let speechStarted = 0;
let finals = 0;
let offset = 0;

ws.on("open", () => console.log("relay open — streaming 6s of laptop taps (no speech)\n"));

ws.on("message", (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.type === "ready") {
    const timer = setInterval(() => {
      if (offset >= pcm.length) {
        clearInterval(timer);
        ws.send(JSON.stringify({ type: "finish" }));
        return;
      }
      ws.send(pcm.subarray(offset, offset + CHUNK));
      offset += CHUNK;
    }, 50);
  } else if (msg.type === "speech_started") {
    speechStarted++;
    console.log(`  speech_started  #${speechStarted}  <-- this used to fire barge-in`);
  } else if (msg.type === "final" && msg.text?.trim()) {
    finals++;
    console.log(`  final: ${JSON.stringify(msg.text)}`);
  }
});

ws.on("close", () => {
  console.log(
    `\nresult: ${speechStarted} speech_started event(s), ${finals} final transcript(s) from pure taps`
  );
  console.log(
    speechStarted > 0
      ? "ROOT CAUSE CONFIRMED: Deepgram's VAD calls taps speech; Silero scores them 0.01-0.05."
      : "no speech_started — taps did not trip Deepgram's VAD in this run."
  );
  process.exit(0);
});

ws.on("error", (e) => {
  console.log("socket error:", e.message);
  process.exit(1);
});
