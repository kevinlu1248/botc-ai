// Frame-level "is this speech?" test, used to decide whether the user is talking
// over the assistant.
//
// Loudness alone can't answer that: a door slam, a keyboard, or music is loud
// too. Two cheap spectral features separate speech from noise well enough for a
// barge-in gate, and both come free from the AnalyserNode we already run.
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
// No single feature suffices, so a frame must clear all four, and enough
// consecutive frames must clear them before it counts as barge-in.

export const VAD = {
  // Loudness separates "talking to the assistant" from a conversation across
  // the room. Slightly lower than before so barge-in feels snappier; spectral
  // gates still reject slams/clatter.
  RMS_MIN: 0.05,
  SPEECH_RATIO_MIN: 0.45,
  FLATNESS_MAX: 0.35,
  DOMINANCE_MAX: 0.65,
  // ~50ms at 60fps — was 9 frames (~150ms) and felt sluggish.
  FRAMES: 3,
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
