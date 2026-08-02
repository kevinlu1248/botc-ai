// Re-transcribes a captured PCM dump under several Deepgram configurations, so
// you can tell whether a bad transcript came from the audio or from the settings.
//
//   STT_DUMP=1 npm run server        # then talk; note the file it logs
//   node scripts/stt-compare.mjs /tmp/botc-stt-….raw
//
// If every configuration produces the same wrong words, the audio is the problem
// (mic, distance, room, browser DSP). If some are right, the settings are.
import fs from "node:fs";
import WebSocket from "ws";

const file = process.argv[2];
if (!file) {
  console.error("usage: node scripts/stt-compare.mjs <dump.raw> [sampleRate]");
  process.exit(1);
}
// The dump filename records the rate the browser negotiated.
const rate = Number(process.argv[3]) || Number(file.match(/-(\d+)hz\.raw$/)?.[1]) || 16000;
const pcm = fs.readFileSync(file);
const seconds = pcm.length / 2 / rate;

const KEYTERMS = (process.env.STT_KEYTERMS || "").split(",").map((t) => t.trim()).filter(Boolean);

const CONFIGS = [
  { label: "nova-3 (as configured)", model: "nova-3", keyterms: KEYTERMS },
  { label: "nova-3, no keyterms", model: "nova-3", keyterms: [] },
  { label: "nova-3, no smart_format", model: "nova-3", keyterms: KEYTERMS, smart_format: false },
  { label: "nova-2 (older model)", model: "nova-2", keyterms: [] },
];

function transcribe(cfg) {
  return new Promise((resolve) => {
    const params = new URLSearchParams({
      model: cfg.model,
      encoding: "linear16",
      sample_rate: String(rate),
      channels: "1",
      smart_format: String(cfg.smart_format !== false),
      punctuate: "true",
    });
    // keyterm is nova-3 only; nova-2 used `keywords`.
    for (const k of cfg.keyterms) params.append(cfg.model.startsWith("nova-3") ? "keyterm" : "keywords", k);

    const ws = new WebSocket(`wss://api.deepgram.com/v1/listen?${params}`, {
      headers: { Authorization: `Token ${process.env.DEEPGRAM_API_KEY}` },
    });
    const out = [];
    let off = 0;
    ws.on("open", () => {
      // Faster than real time is fine for a batch comparison.
      const timer = setInterval(() => {
        if (off >= pcm.length) {
          clearInterval(timer);
          ws.send(JSON.stringify({ type: "CloseStream" }));
          return;
        }
        ws.send(pcm.subarray(off, off + 8000));
        off += 8000;
      }, 10);
    });
    ws.on("message", (m) => {
      const d = JSON.parse(m.toString());
      if (d.type === "Results" && d.is_final) {
        const t = d.channel?.alternatives?.[0]?.transcript;
        if (t) out.push(t);
      }
    });
    ws.on("close", () => resolve(out.join(" ")));
    ws.on("error", (e) => resolve(`<error: ${e.message}>`));
  });
}

console.log(`${file}\n${rate} Hz, ${seconds.toFixed(1)}s\n`);
for (const cfg of CONFIGS) {
  const text = await transcribe(cfg);
  console.log(`${cfg.label}:\n  ${text || "(nothing)"}\n`);
}
