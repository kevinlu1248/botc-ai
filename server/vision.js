// Proxies the Python vision sidecar and exposes looking-at-camera state for
// the STT gate and the live video/gallery UI.

const VISION_URL = process.env.VISION_URL || "http://127.0.0.1:8766";

let cache = {
  running: false,
  fps: 0,
  people: [],
  looking: [],
  error: "vision offline",
  ts: 0,
};
let lastFetch = 0;

async function refresh(force = false) {
  const now = Date.now();
  if (!force && now - lastFetch < 200) return cache;
  lastFetch = now;
  try {
    const res = await fetch(`${VISION_URL}/api/state`, {
      signal: AbortSignal.timeout(800),
    });
    if (!res.ok) throw new Error(`vision ${res.status}`);
    cache = await res.json();
    cache.error = cache.error || null;
  } catch (err) {
    cache = {
      ...cache,
      running: false,
      error: err?.message || "vision offline",
      looking: [],
    };
  }
  return cache;
}

/** People currently looking at the camera. */
export async function lookingNow() {
  const state = await refresh(true);
  return (state.looking || state.people || []).filter((p) => p.looking && p.present);
}

/**
 * Build the user-turn text the model should see.
 * Returns null when nobody is looking — STT should be dropped.
 */
export function formatUtterance(text, looking, allPeople = []) {
  const clean = (text || "").trim();
  if (!clean) return null;
  if (!looking?.length) return null;

  const who = looking.map((p) => p.label || p.pid).join(", ");
  const scores = looking
    .map((p) => `${p.label || p.pid} (look ${Number(p.looking_score || 0).toFixed(2)})`)
    .join(", ");
  const room = (allPeople || [])
    .filter((p) => p.present)
    .map((p) => {
      const tag = p.looking ? "looking" : "not looking";
      return `${p.label || p.pid} [${tag}]`;
    })
    .join("; ");

  // Structured prefix so the voice model can report who said what.
  return (
    `[Room] ${room || "unknown"}\n` +
    `[Speaker] ${who} — looking at the camera (${scores})\n` +
    `[Said] ${clean}`
  );
}

export function registerVisionRoutes(app) {
  app.get("/api/vision/state", async (_req, res) => {
    res.json(await refresh(true));
  });

  app.get("/api/vision/frame.jpg", async (_req, res) => {
    try {
      const upstream = await fetch(`${VISION_URL}/api/frame.jpg`, {
        signal: AbortSignal.timeout(1500),
      });
      if (!upstream.ok) {
        return res.status(upstream.status).type("text").send("no frame");
      }
      const buf = Buffer.from(await upstream.arrayBuffer());
      res.setHeader("Content-Type", "image/jpeg");
      res.setHeader("Cache-Control", "no-store");
      res.send(buf);
    } catch {
      res.status(503).type("text").send("vision offline");
    }
  });

  app.get("/api/vision/faces/:pid.jpg", async (req, res) => {
    const pid = String(req.params.pid || "").replace(/[^a-zA-Z0-9_#]/g, "");
    if (!pid) return res.status(400).end();
    try {
      const upstream = await fetch(`${VISION_URL}/api/vision/faces/${pid}.jpg`, {
        signal: AbortSignal.timeout(1500),
      });
      if (!upstream.ok) return res.status(404).end();
      const buf = Buffer.from(await upstream.arrayBuffer());
      res.setHeader("Content-Type", "image/jpeg");
      res.setHeader("Cache-Control", "no-store");
      res.send(buf);
    } catch {
      res.status(503).end();
    }
  });

  // Called by the client after a final transcript: gate + enrich.
  app.post("/api/vision/gate", async (req, res) => {
    const text = (req.body?.text || "").trim();
    if (!text) return res.status(400).json({ ok: false, reason: "empty" });

    const state = await refresh(true);
    const looking = (state.looking || []).filter((p) => p.looking && p.present);
    if (!looking.length) {
      return res.json({
        ok: false,
        reason: "not_looking",
        message: "Nobody is looking at the camera — transcript ignored.",
        state,
      });
    }
    const enriched = formatUtterance(text, looking, state.people || []);
    res.json({
      ok: true,
      text: enriched,
      looking,
      people: state.people,
      display: {
        who: looking.map((p) => p.label || p.pid).join(", "),
        said: text,
        looking: true,
      },
    });
  });
}

// Background poll so /api/state can include a cheap snapshot.
setInterval(() => refresh(false).catch(() => {}), 500);

export function visionSnapshot() {
  return cache;
}
