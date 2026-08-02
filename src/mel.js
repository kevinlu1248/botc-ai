// Standard mel-scale filterbank, the usual front end for speech visualisation.
//
// Two conventions matter here and both were missing from the first version of
// the meter: frequency bands should be spaced on the *mel* scale (roughly
// logarithmic, matching how hearing resolves pitch) rather than linearly or by
// an arbitrary power law, and magnitudes should be read in *decibels*, because
// perceived loudness is logarithmic in amplitude.

export const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
export const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

// Triangular filters, evenly spaced in mel between fMin and fMax. Each filter
// spans its neighbours' centres, so adjacent bands overlap by half — the
// textbook shape (as in MFCC front ends).
export function buildMelBank({ sampleRate, fftSize, bands, fMin = 80, fMax = 8000 }) {
  const binCount = fftSize / 2;
  const binHz = sampleRate / fftSize;
  const top = Math.min(fMax, sampleRate / 2);

  const melLo = hzToMel(fMin);
  const melHi = hzToMel(top);
  // bands + 2 edges gives every band a left edge, centre and right edge.
  const edgeBins = Array.from(
    { length: bands + 2 },
    (_, i) => melToHz(melLo + ((melHi - melLo) * i) / (bands + 1)) / binHz
  );

  const filters = [];
  for (let b = 0; b < bands; b++) {
    const lo = edgeBins[b];
    const mid = edgeBins[b + 1];
    const hi = edgeBins[b + 2];
    const start = Math.max(0, Math.floor(lo));
    const end = Math.min(binCount - 1, Math.ceil(hi));
    const weights = new Float32Array(Math.max(0, end - start + 1));

    for (let k = start; k <= end; k++) {
      // Rising edge up to the centre, falling edge after it.
      const w = k <= mid ? (k - lo) / Math.max(1e-9, mid - lo) : (hi - k) / Math.max(1e-9, hi - mid);
      weights[k - start] = Math.min(1, Math.max(0, w));
    }
    filters.push({ start, end, weights });
  }
  return filters;
}

// Weighted mean power in a band, returned in dB.
// getFloatFrequencyData gives magnitude in dB (20·log10|X|), so linear power is
// 10^(dB/10). Averaging power and converting back is correct; averaging the dB
// values directly is not, since dB is logarithmic.
export function bandDb(spectrumDb, filter) {
  let power = 0;
  let weight = 0;
  for (let k = filter.start; k <= filter.end; k++) {
    const db = spectrumDb[k];
    if (!Number.isFinite(db)) continue; // -Infinity on digital silence
    const w = filter.weights[k - filter.start];
    power += 10 ** (db / 10) * w;
    weight += w;
  }
  if (weight <= 0) return -Infinity;
  return 10 * Math.log10(power / weight + 1e-12);
}
