"""Applies the browser's mic conditioning chain offline, for A/B testing.

Mirrors src/useMic.js: high-pass 90 Hz, low-pass 7.5 kHz, compressor
(-34 dB / 5:1, 3 ms attack), makeup gain, then a brick-wall limiter. Use it to
check whether conditioning improves a transcript before trusting it live:

    python3 scripts/condition.py in.wav out.wav [gain]
"""
import math
import struct
import sys
import wave


def biquad(samples, b0, b1, b2, a1, a2):
    out = [0.0] * len(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(samples):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out


def highpass(samples, sr, freq, q=0.7):
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / (2 * q)
    cos = math.cos(w0)
    b0, b1, b2 = (1 + cos) / 2, -(1 + cos), (1 + cos) / 2
    a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    return biquad(samples, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def lowpass(samples, sr, freq, q=0.7):
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / (2 * q)
    cos = math.cos(w0)
    b0, b1, b2 = (1 - cos) / 2, 1 - cos, (1 - cos) / 2
    a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    return biquad(samples, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def compress(samples, sr, threshold_db=-34.0, ratio=5.0, attack=0.003, release=0.22):
    at = math.exp(-1.0 / (sr * attack))
    rt = math.exp(-1.0 / (sr * release))
    env = 0.0
    out = [0.0] * len(samples)
    for i, x in enumerate(samples):
        a = abs(x)
        env = a + (at if a > env else rt) * (env - a)
        db = 20 * math.log10(env) if env > 1e-9 else -120.0
        over = db - threshold_db
        gain_db = -over * (1 - 1 / ratio) if over > 0 else 0.0
        out[i] = x * (10 ** (gain_db / 20))
    return out


def limit(samples, ceiling_db=-3.0):
    ceiling = 10 ** (ceiling_db / 20)
    return [max(-ceiling, min(ceiling, x)) for x in samples]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    gain = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

    with wave.open(src) as w:
        sr, n = w.getframerate(), w.getnframes()
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected 16-bit mono"
        samples = [v / 32768 for v in struct.unpack(f"<{n}h", w.readframes(n))]

    def rms(xs):
        return math.sqrt(sum(x * x for x in xs) / max(1, len(xs)))

    before = rms(samples)
    s = highpass(samples, sr, 90)
    s = lowpass(s, sr, 7500)
    s = compress(s, sr)
    s = [x * gain for x in s]
    s = limit(s)
    after = rms(s)

    with wave.open(dst, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, x)) * 32767)) for x in s))

    print(f"{src} -> {dst}")
    print(f"  rms {before:.4f} -> {after:.4f}  ({20 * math.log10(after / max(before, 1e-9)):+.1f} dB)")


if __name__ == "__main__":
    main()
