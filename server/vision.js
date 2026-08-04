// Proxies the Python vision sidecar and exposes looking-at-camera state for
// the STT gate and the live video/gallery UI.

import fs from "node:fs";
import path from "node:path";
import { profileDistance, mergeProfile } from "./profile.js";

const VISION_URL = process.env.VISION_URL || "http://127.0.0.1:8766";

// How far a voice may sit from the enrolled profile before it is treated as a
// different source. Roughly "noticeable differences" — see src/profile.js. Kept
// deliberately loose to start: rejecting the user's own speech is a worse failure
// than letting a phone through, and every decision is logged so this can be
// tightened from real data (VOICE_PROFILE_MAX_DIST).
const PROFILE_MAX_DIST = Number(process.env.VOICE_PROFILE_MAX_DIST) || 2.2;
const PROFILE_ENROLL_MIN = 2; // utterances before the profile is trusted to reject

let cache = {
  running: false,
  fps: 0,
  people: [],
  looking: [],
  error: "vision offline",
  ts: 0,
};
let lastFetch = 0;

// Deepgram diarization separates voices but has no idea which is the user, so the
// first index heard while someone is looking at the camera becomes "the speaker"
// and every other voice is rejected. That is what stops a video playing on a
// phone from being treated as a turn. Indices are per-connection, so this resets
// whenever the STT socket reconnects.
const BIND_VOICE = process.env.VOICE_BIND !== "false";
let binding = { speaker: null, label: null, at: 0 };
let enrolled = null; // running acoustic profile of the person who talks to it

export function resetVoiceBinding() {
  binding = { speaker: null, label: null, at: 0 };
  // The acoustic profile deliberately survives reconnects — unlike diarization
  // indices it is not tied to a Deepgram session.
}

export function resetVoiceProfile() {
  enrolled = null;
}

export function voiceBinding() {
  return {
    ...binding,
    enabled: BIND_VOICE,
    profile: enrolled,
    maxDistance: PROFILE_MAX_DIST,
  };
}

// Human-chosen names for the faces the sidecar tracks, keyed by its persistent
// person id. Kept here rather than in the sidecar so renaming does not depend on the
// Python process, and on disk so names survive a restart — the sidecar's own labels
// are "Person 1", "Person 2", assigned in the order it happens to see people.
const NAMES_FILE = path.join(process.cwd(), ".run", "people-names.json");
let names = {};

function loadNames() {
  try {
    names = JSON.parse(fs.readFileSync(NAMES_FILE, "utf8"));
  } catch {
    names = {}; // absent or unreadable: start empty, not fatal
  }
}

function saveNames() {
  fs.mkdirSync(path.dirname(NAMES_FILE), { recursive: true });
  fs.writeFileSync(NAMES_FILE, JSON.stringify(names, null, 2));
}

loadNames();

/** Overlays custom names so every consumer — gate, prompt, UI — sees the same one. */
function applyNames(state) {
  const rename = (p) => (p && names[p.pid] ? { ...p, label: names[p.pid] } : p);
  return {
    ...state,
    people: (state.people || []).map(rename),
    looking: (state.looking || []).map(rename),
  };
}

async function refresh(force = false) {
  const now = Date.now();
  if (!force && now - lastFetch < 200) return cache;
  lastFetch = now;
  try {
    const res = await fetch(`${VISION_URL}/api/state`, {
      signal: AbortSignal.timeout(800),
    });
    if (!res.ok) throw new Error(`vision ${res.status}`);
    cache = applyNames(await res.json());
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
export function formatUtterance(text, looking, allPeople = [], attribution = null) {
  const clean = (text || "").trim();
  if (!clean) return null;

  // Nobody looking is no longer a reason to discard the utterance — it only means the
  // camera cannot say which face it came from.
  const present = (allPeople || []).filter((p) => p.present);
  const who = looking?.length
    ? looking.map((p) => p.label || p.pid).join(", ")
    : present.length === 1
      ? `${present[0].label || present[0].pid} (not looking)`
      : "Unidentified speaker";
  const scores = (looking || [])
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
  if (attribution) {
    // Explicitly not the person at the camera: ambient audio, a phone, a TV, or
    // someone else in the room. Spelled out rather than hinted at, because the model
    // has to be able to tell "the user asked me something" from "a voice happened".
    return (
      `[Room] ${room || "unknown"}\n` +
      `[Speaker] ${attribution.label} — NOT ${who} and not addressing you: ` +
      `${attribution.detail}. Background audio or another person; context only.\n` +
      `[Said] ${clean}`
    );
  }
  return (
    `[Room] ${room || "unknown"}\n` +
    `[Speaker] ${who}` +
    (looking?.length
      ? ` — looking at the camera (${scores})`
      : ` — nobody is looking at the camera, so this is who most likely spoke`) +
    `\n[Said] ${clean}`
  );
}

export function registerVisionRoutes(app) {
  app.get("/api/vision/state", async (_req, res) => {
    res.json(await refresh(true));
  });

  // Rename a tracked face. An empty name clears the override and falls back to the
  // sidecar's own label.
  app.post("/api/vision/people/:pid/name", async (req, res) => {
    const pid = String(req.params.pid || "").trim();
    if (!pid) return res.status(400).json({ ok: false, error: "pid required" });

    const name = String(req.body?.name ?? "").trim().slice(0, 40);
    if (name) names[pid] = name;
    else delete names[pid];

    try {
      saveNames();
    } catch (err) {
      // Loud: a rename that silently fails to persist looks like it worked until
      // the next restart.
      console.error(`[vision] could not save names: ${err.message}`);
      return res.status(500).json({ ok: false, error: "could not save name" });
    }
    console.log(`[vision] ${pid} renamed to ${name || "(cleared)"}`);

    // Names appear in the voice binding label too, so refresh it for consistency.
    if (binding.label && binding.speaker !== null) {
      const state = await refresh(true);
      const bound = (state.people || []).find((p) => p.present && p.looking);
      if (bound) binding = { ...binding, label: bound.label || bound.pid };
    }
    res.json({ ok: true, pid, name: name || null, names });
  });

  app.get("/api/vision/people/names", (_req, res) => res.json({ names }));

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
    // Prefer explicit looking flag; fall back to a decent look score so a brief
    // flicker at the end of an utterance does not drop a real turn (esp. close-up).
    let looking = (state.looking || []).filter((p) => p.looking && p.present);
    if (!looking.length) {
      looking = (state.people || []).filter(
        (p) => p.present && (p.looking || Number(p.looking_score || 0) >= 0.35)
      );
    }
    // Looking is used to ATTRIBUTE an utterance to a face, not to gate it. It used
    // to drop the transcript outright, which threw away real speech whenever someone
    // glanced away mid-sentence. Whose voice it is is decided by diarization and the
    // acoustic profile below; the camera only answers "which person in frame".
    const nobodyLooking = !looking.length;
    // Voice binding. Only meaningful once diarization has actually labelled a
    // speaker; without a label, fall through and behave as before.
    // Another voice is no longer dropped. It goes into the transcript attributed to
    // a different speaker, so the model has the context and can decide for itself
    // whether any of it was addressed to it. Dropping it lost information the model
    // sometimes needs ("what was that on the TV?").
    //
    // `attribution` is either the bound user or a labelled other voice.
    let attribution = null; // null = the person looking at the camera
    let indexSuspicion = null; // diarization disagrees, pending the acoustic check

    const speaker = Number.isInteger(req.body?.speaker) ? req.body.speaker : null;
    if (BIND_VOICE && speaker !== null) {
      if (binding.speaker === null) {
        // Only ever bind to a voice heard while someone is actually looking, or the
        // first thing the TV says becomes "the user" and the real user gets labelled
        // as the other voice. Looking no longer gates delivery, but it is still what
        // makes the initial binding trustworthy.
        if (!nobodyLooking) {
          binding = {
            speaker,
            label: looking.map((p) => p.label || p.pid).join(", "),
            at: Date.now(),
          };
        }
      } else if (speaker !== binding.speaker) {
        // A SUSPICION, not a verdict. Deepgram has been observed putting the user's
        // own speech at index 1 and a TV at index 1 in the same session, so this is
        // only acted on when the acoustic profile agrees or is unavailable.
        indexSuspicion = {
          // Indices are 0-based and the bound user is normally 0, so +1 reads
          // naturally: index 1 becomes "Speaker 2".
          label: `Speaker ${speaker + 1}`,
          reason: "different_voice_index",
          detail: `diarized as a different voice from ${binding.label || "the speaker"}`,
          speaker,
        };
      }
    }

    // Acoustic check. Diarization indices proved unreliable on short utterances,
    // so the load-bearing test is the profile: a phone speaker cannot reproduce a
    // human fundamental (lowRatio) and broadcast audio is compression-flattened
    // (rmsStd), which separates the two cleanly without any model.
    const profile = req.body?.profile || null;
    if (profile) {
      const dist = enrolled ? profileDistance(enrolled, profile) : 0;
      const trusted = (enrolled?.enrolledFrom || 0) >= PROFILE_ENROLL_MIN;
      console.log(
        `[voice] dist=${dist.toFixed(2)} enrolled=${enrolled?.enrolledFrom || 0} ` +
          `low=${profile.lowRatio} rmsStd=${profile.rmsStd} cent=${profile.centroidMean} ` +
          `frames=${profile.frames} :: ${text.slice(0, 48)}`
      );
      // The profile OVERRULES the diarization index, in both directions.
      //
      // Measured in one session: Deepgram labelled the user's own "Teach me something
      // new." as index 1 while the profile put it at 1.70 (inside the 2.2 threshold,
      // i.e. the same person) — and because the index was checked first and
      // short-circuited, the user got labelled Speaker 2. The unreliable signal was
      // beating the reliable one, so it no longer gets to.
      if (trusted && dist > PROFILE_MAX_DIST) {
        attribution = {
          label:
            speaker !== null && binding.speaker !== null && speaker !== binding.speaker
              ? `Speaker ${speaker + 1}`
              : "Speaker 2",
          reason: "other_voice_acoustic",
          detail: `sounds acoustically different from ${binding.label || "the speaker"} (distance ${dist.toFixed(2)}, threshold ${PROFILE_MAX_DIST})`,
          distance: +dist.toFixed(2),
          speaker,
        };
      } else if (trusted && indexSuspicion) {
        console.log(
          `[voice] index said other voice but profile says same person ` +
            `(dist=${dist.toFixed(2)}) — keeping it attributed to the user`
        );
      } else if (!trusted && indexSuspicion) {
        // Still enrolling, so there is nothing to overrule the index with yet. Also
        // keeps this voice out of the enrollment below, which matters most early on:
        // a contaminated reference is what makes every later comparison useless.
        attribution = indexSuspicion;
      }
      // Only reinforce with the bound user's voice. Merging an unattributed voice
      // would drag the reference toward whatever else is in the room, and after a
      // few TV lines the user would start failing their own check.
      if (!attribution) enrolled = mergeProfile(enrolled, profile);
    } else {
      // No profile: the index is all that is left, so act on the suspicion. Logged
      // loudly, because the load-bearing half of the gate did not run.
      attribution = indexSuspicion;
      console.log(
        `[voice] NO PROFILE — falling back to diarization index ` +
          `(${indexSuspicion ? `attributing to ${indexSuspicion.label}` : "no mismatch"}) ` +
          `:: ${text.slice(0, 48)}`
      );
    }

    if (attribution) {
      console.log(
        `[voice] attributed to ${attribution.label} (${attribution.reason}) :: ${text.slice(0, 48)}`
      );
    }

    const enriched = formatUtterance(text, looking, state.people || [], attribution);
    res.json({
      ok: true,
      text: enriched,
      looking,
      people: state.people,
      attribution,
      display: {
        who: attribution ? attribution.label : looking.map((p) => p.label || p.pid).join(", "),
        said: text,
        looking: !attribution,
        attributed: !attribution,
      },
    });
  });
}

// Background poll so /api/state can include a cheap snapshot.
setInterval(() => refresh(false).catch(() => {}), 500);

export function visionSnapshot() {
  return cache;
}
