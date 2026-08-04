// Per-utterance acoustic profile, used to tell one voice from another.
//
// This is not speaker recognition — it is a cheap physical discriminator, which is
// enough for the case that actually bites: a phone or laptop speaker playing audio
// versus a person talking a foot from the mic. Those differ in ways you can measure
// with a handful of numbers:
//
//   lowRatio  — share of energy in 80–300 Hz, where a human fundamental lives. A
//               small speaker physically cannot reproduce it, so this is the single
//               most decisive feature.
//   rmsStd    — dynamic range across the utterance. Broadcast/game audio is
//               loudness-compressed and sits flat; a live voice swells and drops.
//   centroid  — spectral brightness, mean and spread.
//   flatness  — tonal vs noise-like.
//
// IMPORTANT: feed this the *raw* microphone tap. The conditioning chain in
// useMic.js compresses and band-limits deliberately, which flattens rmsStd and
// scrubs lowRatio — the very things being measured here.

const LOW_LO = 80;
const LOW_HI = 300;
const BAND_LO = 80;
const BAND_HI = 4000;

export function createProfiler() {
  let n = 0;
  let rmsSum = 0;
  let rmsSqSum = 0;
  let lowSum = 0;
  let centroidSum = 0;
  let centroidSqSum = 0;
  let flatSum = 0;

  return {
    /** Accumulate one speech-ish frame. `power` is linear power per FFT bin. */
    push(rms, power, sampleRate, fftSize) {
      const binHz = sampleRate / fftSize;
      const bin = (hz) => Math.min(power.length - 1, Math.max(0, Math.round(hz / binHz)));
      const lo = bin(BAND_LO);
      const hi = bin(Math.min(BAND_HI, sampleRate / 2));
      const lowLo = bin(LOW_LO);
      const lowHi = bin(LOW_HI);

      let total = 0;
      let low = 0;
      let weighted = 0;
      let logSum = 0;
      let count = 0;
      for (let k = lo; k <= hi; k++) {
        const p = Math.max(1e-12, power[k]);
        total += p;
        weighted += p * k * binHz;
        logSum += Math.log(p);
        count++;
        if (k >= lowLo && k <= lowHi) low += p;
      }
      if (total <= 1e-11) return;

      const centroid = weighted / total;
      const flatness = Math.exp(logSum / count) / (total / count);

      n++;
      rmsSum += rms;
      rmsSqSum += rms * rms;
      lowSum += low / total;
      centroidSum += centroid;
      centroidSqSum += centroid * centroid;
      flatSum += flatness;
    },

    frames() {
      return n;
    },

    /** Snapshot and reset. Returns null if too little speech to characterise. */
    take(minFrames = 25) {
      if (n < minFrames) {
        this.reset();
        return null;
      }
      const mean = (sum) => sum / n;
      const std = (sum, sqSum) => {
        const m = sum / n;
        return Math.sqrt(Math.max(0, sqSum / n - m * m));
      };
      const out = {
        frames: n,
        rmsMean: +mean(rmsSum).toFixed(5),
        rmsStd: +std(rmsSum, rmsSqSum).toFixed(5),
        lowRatio: +mean(lowSum).toFixed(4),
        centroidMean: Math.round(mean(centroidSum)),
        centroidStd: Math.round(std(centroidSum, centroidSqSum)),
        flatness: +mean(flatSum).toFixed(5),
      };
      this.reset();
      return out;
    },

    reset() {
      n = 0;
      rmsSum = rmsSqSum = lowSum = centroidSum = centroidSqSum = flatSum = 0;
    },
  };
}

// Rough per-feature scales, so one dimension can't dominate the distance purely
// because of its units. Tuned to be about "one noticeable difference" each.
const SCALE = {
  lowRatio: 0.09,
  rmsStd: 0.02,
  centroidMean: 500,
  flatness: 0.05,
};

/**
 * Normalised distance between two profiles. Roughly "how many noticeable
 * differences apart", so a threshold near 1.5–2 means clearly a different source.
 * lowRatio is weighted hardest because it is the physically grounded one.
 */
export function profileDistance(a, b) {
  if (!a || !b) return 0;
  const d = (k) => Math.abs((a[k] ?? 0) - (b[k] ?? 0)) / SCALE[k];
  return Math.sqrt(
    2.0 * d("lowRatio") ** 2 +
      1.0 * d("rmsStd") ** 2 +
      0.7 * d("centroidMean") ** 2 +
      0.5 * d("flatness") ** 2
  );
}

/** Running average of enrolled profiles, so the reference firms up with use. */
export function mergeProfile(base, next, weight = 0.25) {
  if (!base) return { ...next, enrolledFrom: 1 };
  const lerp = (k) => base[k] + (next[k] - base[k]) * weight;
  return {
    frames: next.frames,
    rmsMean: +lerp("rmsMean").toFixed(5),
    rmsStd: +lerp("rmsStd").toFixed(5),
    lowRatio: +lerp("lowRatio").toFixed(4),
    centroidMean: Math.round(lerp("centroidMean")),
    centroidStd: Math.round(lerp("centroidStd")),
    flatness: +lerp("flatness").toFixed(5),
    enrolledFrom: (base.enrolledFrom || 1) + 1,
  };
}
