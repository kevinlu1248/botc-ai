// Runtime-mutable model configuration.
//
// Every model used to be a module-level `const` read from env at import time,
// which meant changing one required editing .env and restarting three processes.
// These live here instead so the settings modal can change them between turns.
// Env still supplies the defaults, so .env remains the way to set startup state.

const pick = (value, allowed, fallback) => (allowed.includes(value) ? value : fallback);

// Options are curated rather than fetched: the ElevenLabs key is scoped to
// text-to-speech only (no models_read), and every entry below was verified by
// actually calling the API with it.
export const OPTIONS = {
  // Gemini entries are only listed because server/gemini.js can actually drive
  // them; an option the code cannot execute is worse than no option.
  fast: [
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4-5",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
  ],
  slow: ["claude-opus-5", "claude-fable-5", "claude-sonnet-5"],
  stt: ["nova-3", "nova-2", "enhanced", "base"],
  tts: [
    "eleven_flash_v2_5",
    "eleven_flash_v2",
    "eleven_turbo_v2_5",
    "eleven_turbo_v2",
    "eleven_multilingual_v2",
    "eleven_v3",
  ],
  // Library voices (Rachel, Aria, …) return 402 on the free tier; these are
  // default voices confirmed working on this key.
  voice: [
    { id: "JBFqnCBsd6RMkjVDRZzb", label: "George" },
    { id: "EXAVITQu4vr4xnSDxMaL", label: "Sarah" },
    { id: "nPczCjzI2devNBz1zQrb", label: "Brian" },
  ],
};

const settings = {
  fast: pick(process.env.FAST_MODEL, OPTIONS.fast, "claude-sonnet-5"),
  slow: pick(process.env.SLOW_MODEL, OPTIONS.slow, "claude-fable-5"),
  stt: pick(process.env.STT_MODEL, OPTIONS.stt, "nova-3"),
  tts: pick(process.env.TTS_MODEL, OPTIONS.tts, "eleven_flash_v2_5"),
  voice: pick(
    process.env.ELEVENLABS_VOICE_ID,
    OPTIONS.voice.map((v) => v.id),
    "JBFqnCBsd6RMkjVDRZzb" // George
  ),
};

export function getSettings() {
  return { ...settings };
}

/** Applies only recognised keys with allowed values; returns what changed. */
export function updateSettings(patch = {}) {
  const changed = {};
  for (const key of ["fast", "slow", "stt", "tts"]) {
    if (patch[key] && OPTIONS[key].includes(patch[key]) && patch[key] !== settings[key]) {
      settings[key] = patch[key];
      changed[key] = patch[key];
    }
  }
  const voiceIds = OPTIONS.voice.map((v) => v.id);
  if (patch.voice && voiceIds.includes(patch.voice) && patch.voice !== settings.voice) {
    settings.voice = patch.voice;
    changed.voice = patch.voice;
  }
  return changed;
}
