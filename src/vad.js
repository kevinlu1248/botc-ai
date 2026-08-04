// Frame-level spectral features, used **only** to decide which frames are worth
// feeding into the per-utterance acoustic profile (see profile.js).
//
// This is NOT the barge-in detector. Barge-in is Silero, via src/bargein.js, and
// it has exactly one authority — see the comment at the top of that file for why
// that rule is now structural. The hand-tuned thresholds below are kept because
// profiling uses `isProfileFrame`, but they no longer decide anything the user can
// feel, so do not re-tune them for interruption behaviour.
//
// Two cheap spectral features separate speech from noise, and both come free from
// the AnalyserNode we already run.
//
//   speechRatio — fraction of power inside 300–3400 Hz (the band voices occupy).
//                 Slams and desk thumps put most of their energy below that;
//                 keyboard clicks and clatter put it above.
//   flatness    — spectral flatness, the geometric mean of the power spectrum
//                 over its arithmetic mean. Noise-like spectra are flat (→1);
//                 voiced speech is peaky, with harmonics and formants (→0).
//   dominance   — share of band energy in the single loudest bin. This is what
//                 separates a beep from a vowel: both are peaky, but a tone puts
//                 nearly all its energy in one bin, while speech spreads it
//                 across a harmonic series and formants. (Flatness cannot tell
//                 them apart — measured 0.0000 for a 1kHz tone vs 0.0005 for
//                 voiced speech.)
//
// Barge-in is intentionally aggressive: cutting the assistant a bit early is
// far better UX than lagging half a second after the user starts talking.

export const VAD = {
  // Soft path: one consecutive speech-like frame is enough.
  RMS_MIN: 0.028,
  // Level floor for profiling only — see isProfileFrame for the measurements.
  PROFILE_RMS_MIN: 0.006,
  SPEECH_RATIO_MIN: 0.28,
  FLATNESS_MAX: 0.48, // was 0.55: hair/clothing noise sits just under that
  DOMINANCE_MAX: 0.8,
  FRAMES: 1, // kept for reference; the loop confirms by duration (MIN_MS) instead
  // Poll windows overlap heavily (6ms interval, 128ms FFT), so counting frames is
  // a weak time filter. Requiring qualifying audio to persist for a real duration
  // rejects clicks and taps while staying far below perceptible latency.
  MIN_MS: 90,
  // Hard path: loud energy → interrupt on the first sample (AEC can dull speech
  // features, so pure loudness still counts while the assistant is talking).
  HARD_RMS: 0.04,
  HARD_SPEECH_RATIO: 0.45, // was 0.25: noise measures 28-41%, speech 61-98%
  HARD_FLATNESS_MAX: 0.35, // was 0.6: noise measures 0.54-0.58, voiced speech 0.002-0.04
  // Ultra path: just loud — user said "stop" four times because spectral gates
  // rejected near-field speech under echo cancellation.
  LOUD_RMS: 0.07,
  LOUD_FLATNESS_MAX: 0.35, // reject broadband noise: scratching, inhaling, taps
  LOUD_SPEECH_RATIO: 0.3, // and require the energy to be in the voice band
};

const SPEECH_LO = 300;
const SPEECH_HI = 3400;
const BAND_LO = 80;
const BAND_HI = 8000;
const EPS = 1e-12;

// waveform: Uint8Array from getByteTimeDomainData (128 = silence)
// power:    Float32Array of *linear* power per bin (convert dB with 10^(dB/10))
export function frameFeatures(waveform, power, sampleRate, fftSize) {
  let sum = 0;
  for (let i = 0; i < waveform.length; i++) {
    const v = (waveform[i] - 128) / 128;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / waveform.length);

  const binHz = sampleRate / fftSize;
  const bin = (hz) => Math.min(power.length - 1, Math.max(0, Math.round(hz / binHz)));
  const lo = bin(BAND_LO);
  const hi = bin(Math.min(BAND_HI, sampleRate / 2));
  const sLo = bin(SPEECH_LO);
  const sHi = bin(Math.min(SPEECH_HI, sampleRate / 2));

  let total = 0;
  let speech = 0;
  let logSum = 0;
  let count = 0;
  let peak = 0;
  for (let k = lo; k <= hi; k++) {
    const p = Math.max(EPS, power[k]);
    total += p;
    logSum += Math.log(p);
    count++;
    if (p > peak) peak = p;
    if (k >= sLo && k <= sHi) speech += p;
  }

  const speechRatio = total > EPS ? speech / total : 0;
  // Geometric mean / arithmetic mean.
  const flatness = count > 0 && total > EPS ? Math.exp(logSum / count) / (total / count) : 1;
  const dominance = total > EPS ? peak / total : 1;

  return { rms, speechRatio, flatness, dominance };
}

export function isSpeechFrame({ rms, speechRatio, flatness, dominance }) {
  return (
    rms >= VAD.RMS_MIN &&
    speechRatio >= VAD.SPEECH_RATIO_MIN &&
    flatness <= VAD.FLATNESS_MAX &&
    dominance <= VAD.DOMINANCE_MAX
  );
}

/**
 * Frame selector for acoustic *profiling*, which needs a much lower level floor
 * than the old barge-in test it used to share.
 *
 * Measured on the raw tap, which is what profiling reads (levels are pre-makeup-gain
 * and browser echo cancellation ducks it further):
 *   user speech   10 of 100 frames cleared 0.028, peak 0.0446
 *   user "Stop."   0 of 104 frames cleared 0.028, peak 0.0186
 *   a TV playing   0 of 449 frames cleared 0.028, peak 0.0107
 * So at RMS_MIN the profiler never reached the 25 frames it needs and every single
 * utterance was gated with no profile at all. The spectral tests are kept — they are
 * what reject non-speech — and only the level floor is lowered.
 *
 * The floor deliberately admits the TV too: a voice must be *characterised* before
 * it can be rejected, and the discrimination is profileDistance's job, not the
 * floor's.
 */
export function isProfileFrame({ rms, speechRatio, flatness, dominance }) {
  return (
    rms >= VAD.PROFILE_RMS_MIN &&
    speechRatio >= VAD.SPEECH_RATIO_MIN &&
    flatness <= VAD.FLATNESS_MAX &&
    dominance <= VAD.DOMINANCE_MAX
  );
}

// `isHardBargeIn` used to live here — a loudness-first interrupt test. It is gone
// on purpose. It had no callers left, and keeping a plausible-looking
// "should I interrupt?" helper next to the real detector is how this subsystem
// ended up with two competing authorities in the first place. Interruption logic
// belongs in src/bargein.js, and nowhere else.
