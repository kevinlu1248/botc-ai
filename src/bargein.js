// The ONE place that decides whether the user has started talking over the
// assistant.
//
// This module exists because the decision used to be made in two places at once,
// and the weaker one won:
//
//   1. Silero VAD over the raw mic tap — a trained speech/no-speech model.
//   2. Deepgram's `vad_events` SpeechStarted, relayed as `speech_started`, which
//      called the interrupt directly with no corroboration.
//
// (2) is energy-based. It fires on a knuckle-tap on the laptop, a door, a siren —
// and because it bypassed (1) entirely, every threshold in Silero was irrelevant to
// the false interrupts users actually hit. Debugging the model while a second
// detector was doing the firing cost a lot of wasted effort, so the rule is now
// structural rather than a matter of care:
//
//   **Barge-in has exactly one authority: Silero, through this module.**
//
// Deepgram's VAD is still useful for *end of turn* (see useMic's finalize path),
// where it cannot cut anyone off. It is never allowed to start an interrupt.
// If you are adding a new signal, it goes through `consider()` — do not call the
// interrupt from a message handler.

export const BARGE_IN = {
  // Frames are 32ms, so four is ~128ms of sustained speech.
  //
  // Measured by scripts/vad-bargein.mjs against real recordings rather than
  // guessed — each candidate's cost in added latency, and whether it ever misses:
  //   2 consecutive   baseline (was), a single short burst clears it
  //   4 consecutive   +64ms flat, misses nothing            <- chosen
  //   6 consecutive   +128..608ms, worst on quiet speech
  //   6-of-10 window  +128..608ms, no better than 6 consecutive
  // Over half a second of extra latency on quiet speech is a worse failure than
  // the one being fixed, so the window variants were rejected.
  MIN_FRAMES: 4,

  // Frames of history kept for diagnostics. A false barge-in is otherwise
  // unreproducible: it happens in the browser, from audio nobody kept, and the
  // report arrives minutes later.
  TRAIL: 16,
};

/**
 * Accumulates VAD frames and reports when the assistant should be cut off.
 *
 * Every frame goes in, whether or not the assistant is speaking, so the trail
 * still explains a firing that happened right after playback started.
 */
export function createBargeInGate() {
  let hits = 0;
  const trail = [];

  return {
    /**
     * @param speaking Silero's hysteresis-smoothed speech decision for this frame
     * @param prob     raw Silero probability, for the diagnostic trail
     * @param rms      frame level, for the diagnostic trail
     * @param armed    true only while assistant audio is actually playing
     * @returns a diagnostic object when the assistant should be interrupted, else null
     */
    consider({ speaking, prob, rms, armed }) {
      trail.push({ p: Number(prob.toFixed(3)), r: Number(rms.toFixed(4)) });
      if (trail.length > BARGE_IN.TRAIL) trail.shift();

      // Not armed, or not speech: no evidence accumulates.
      if (!armed || !speaking) {
        hits = 0;
        return null;
      }
      if (++hits < BARGE_IN.MIN_FRAMES) return null;

      hits = 0;
      return { source: "silero", frames: BARGE_IN.MIN_FRAMES, trail: trail.slice(-8) };
    },

    /** Recent frames, for logging a signal this gate chose NOT to act on. */
    trail() {
      return trail.slice(-8);
    },

    reset() {
      hits = 0;
      trail.length = 0;
    },
  };
}
