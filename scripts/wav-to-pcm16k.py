"""Resample a 16-bit mono WAV to 16 kHz raw PCM (linear interpolation).

Only used to feed the STT relay a realistic test signal — audioop was removed
in Python 3.13+, and ffmpeg isn't installed on this machine.
"""
import struct
import sys
import wave

src, dst, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
TARGET = 16000

with wave.open(src) as w:
    assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected 16-bit mono"
    rate = w.getframerate()
    n = min(w.getnframes(), int(rate * seconds))
    samples = struct.unpack(f"<{n}h", w.readframes(n))

ratio = rate / TARGET
out_len = int(len(samples) / ratio)
out = bytearray()
for i in range(out_len):
    pos = i * ratio
    lo = int(pos)
    hi = min(lo + 1, len(samples) - 1)
    frac = pos - lo
    out += struct.pack("<h", int(samples[lo] * (1 - frac) + samples[hi] * frac))

with open(dst, "wb") as f:
    f.write(out)

print(f"{src} {rate}Hz -> {dst} {TARGET}Hz, {out_len} samples ({out_len/TARGET:.1f}s)")
