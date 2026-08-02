// Streams raw 16 kHz PCM through the app's own /ws/stt relay, exactly as the
// browser's AudioWorklet would, and prints the transcripts Deepgram returns.
import fs from "node:fs";
import WebSocket from "ws";

const pcm = fs.readFileSync(process.argv[2]);
const CHUNK = 1600; // 50 ms of 16 kHz 16-bit mono

const rate = process.argv[3] || "16000";
// STT_HOST lets you test through the Vite proxy (5181) or the API directly (3001).
const host = process.env.STT_HOST || "localhost:3001";
const ws = new WebSocket(`ws://${host}/ws/stt?rate=${rate}`);
let offset = 0;
let finals = 0;

ws.on("open", () => console.log("relay socket open"));

ws.on("message", (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.type === "ready") {
    console.log("deepgram ready — streaming audio\n");
    const timer = setInterval(() => {
      if (offset >= pcm.length) {
        clearInterval(timer);
        // Mirror the real client: ask the server to flush, let it close us.
        ws.send(JSON.stringify({ type: "finish" }));
        return;
      }
      ws.send(pcm.subarray(offset, offset + CHUNK));
      offset += CHUNK;
    }, 50);
  } else if (msg.type === "partial") {
    process.stdout.write(`\r  [partial] ${msg.text.slice(-95).padEnd(96)}`);
  } else if (msg.type === "final") {
    finals++;
    process.stdout.write(`\r${" ".repeat(100)}\r`);
    console.log(`  [FINAL] ${msg.text}`);
  } else if (msg.type === "error") {
    console.log(`  [ERROR] ${msg.message}`);
  }
});

ws.on("close", () => {
  console.log(`\ndone — ${finals} final transcript(s)`);
  process.exit(finals > 0 ? 0 : 1);
});
ws.on("error", (err) => {
  console.log("socket error:", err.message);
  process.exit(1);
});
