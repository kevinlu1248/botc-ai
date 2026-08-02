import "./env.js"; // must stay first — populates process.env for the imports below
import express from "express";
import http from "node:http";
import { bus, shared } from "./context.js";
import { runFastChat, MODELS, truncateLastReply } from "./agents.js";
import { attachSpeechSocket, registerTtsRoute, STT_MODELS, DEFAULT_STT_MODEL } from "./voice.js";
import { registerVisionRoutes, visionSnapshot } from "./vision.js";

for (const [name, purpose] of [
  ["ANTHROPIC_API_KEY", "both models"],
  ["DEEPGRAM_API_KEY", "speech-to-text"],
  ["ELEVENLABS_API_KEY", "text-to-speech"],
]) {
  if (!process.env[name] && !(name === "ANTHROPIC_API_KEY" && process.env.ANTHROPIC_AUTH_TOKEN)) {
    console.warn(`⚠️  ${name} not set — ${purpose} will not work. See .env.example.`);
  }
}

const app = express();
app.use(express.json());

// --- Chat: one user turn through the fast model, streamed back as NDJSON ---
app.post("/api/chat", async (req, res) => {
  const text = (req.body?.text || "").trim();
  if (!text) return res.status(400).json({ error: "text required" });

  res.setHeader("Content-Type", "application/x-ndjson");
  res.setHeader("Cache-Control", "no-cache");
  const send = (obj) => res.write(JSON.stringify(obj) + "\n");

  try {
    await runFastChat(text, send);
  } catch (err) {
    console.error("chat error:", err);
    send({ type: "error", message: err?.message || String(err) });
  }
  res.end();
});

// --- Events: SSE channel for the slow model's announcements and job updates ---
app.get("/api/events", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const onEvent = (evt) => res.write(`data: ${JSON.stringify(evt)}\n\n`);
  bus.on("event", onEvent);
  const heartbeat = setInterval(() => res.write(": ping\n\n"), 25000);

  req.on("close", () => {
    bus.off("event", onEvent);
    clearInterval(heartbeat);
  });
});

// --- State snapshot for page load ---
app.get("/api/state", (req, res) => {
  res.json({
    models: MODELS,
    stt: { models: STT_MODELS, current: DEFAULT_STT_MODEL },
    voice: {
      stt: Boolean(process.env.DEEPGRAM_API_KEY),
      tts: Boolean(process.env.ELEVENLABS_API_KEY),
    },
    vision: visionSnapshot(),
    insights: shared.insights,
    jobs: [...shared.jobs.values()],
  });
});

registerVisionRoutes(app);

// The client reports how far playback actually got when the user cut in.
app.post("/api/interrupted", (req, res) => {
  const spoken = typeof req.body?.spoken === "string" ? req.body.spoken : "";
  if (shared.turnInFlight) {
    // The current assistant message isn't in history yet — it's pushed when its
    // stream finishes. Truncating now would rewrite the *previous* reply and
    // leave this one intact. runFastChat applies this once the message exists.
    shared.pendingTruncation = spoken;
    return res.json({ deferred: true });
  }
  res.json(truncateLastReply(spoken));
});

registerTtsRoute(app);

const server = http.createServer(app);
attachSpeechSocket(server); // ws://.../ws/stt

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`fast (voice):     ${MODELS.fast}`);
  console.log(`slow (reasoning): ${MODELS.slow}`);
  console.log(`BOTC AI server on http://localhost:${PORT}`);
});
