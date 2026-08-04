import "./env.js"; // must stay first — populates process.env for the imports below
import express from "express";
import http from "node:http";
import { bus, shared } from "./context.js";
import { runFastChat, MODELS, truncateLastReply } from "./agents.js";
import { attachSpeechSocket, registerTtsRoute, STT_MODELS, DEFAULT_STT_MODEL } from "./voice.js";
import { getSettings, updateSettings, OPTIONS } from "./settings.js";
import { registerVisionRoutes, visionSnapshot, voiceBinding } from "./vision.js";

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
    // A 400 about message shape is unfixable without seeing the exact payload, and
    // history is in-memory so a restart destroys the evidence. Dump it.
    try {
      const { writeFileSync, mkdirSync } = await import("node:fs");
      mkdirSync(".run", { recursive: true });
      writeFileSync(
        ".run/last-bad-history.json",
        JSON.stringify({ error: err?.message, history: shared.history }, null, 2)
      );
      console.error("chat error: history dumped to .run/last-bad-history.json");
    } catch {
      /* dumping must never mask the original error */
    }
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
    stt: { models: STT_MODELS, current: DEFAULT_STT_MODEL, voice: voiceBinding() },
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

// Model configuration, read and written by the settings modal. Changes apply to
// the next turn; the STT model additionally needs the mic to reconnect, which the
// client handles.
app.get("/api/settings", (_req, res) => {
  res.json({ settings: getSettings(), options: OPTIONS });
});

app.post("/api/settings", (req, res) => {
  const changed = updateSettings(req.body || {});
  if (Object.keys(changed).length) console.log("[settings]", changed);
  res.json({ ok: true, changed, settings: getSettings() });
});

// Frontend errors, so browser-only failures show up in the server log.
const clientErrors = [];
app.post("/api/client-error", (req, res) => {
  const { message, kind, stack, at, url } = req.body || {};
  const entry = {
    ts: new Date().toISOString(),
    kind: kind || "error",
    message: String(message || "").slice(0, 2000),
    at,
    url,
    stack: stack ? String(stack).slice(0, 1200) : undefined,
  };
  clientErrors.push(entry);
  if (clientErrors.length > 100) clientErrors.shift();
  console.error(`[client:${entry.kind}] ${entry.message}${entry.at ? ` (${entry.at})` : ""}`);
  res.json({ ok: true });
});

app.get("/api/client-errors", (_req, res) => {
  res.json({ errors: clientErrors.slice(-50).reverse() });
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
