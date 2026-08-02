#!/usr/bin/env python3
"""Measure how well voices separate here, and pick the threshold from data.

Record a few people, one at a time:

    python calibrate_voice.py --speaker sam
    python calibrate_voice.py --speaker alex
    python calibrate_voice.py --report

The default thresholds in `VoiceConfig` are a guess. This replaces the guess
with numbers from your microphone, your room, and the people who actually use
it - which is the only version that means anything, because a speaker embedding
model's separation depends heavily on recording conditions.

The report prints two distributions: how similar a person is to *themselves*
across different utterances, and how similar they are to *other people*. The
threshold belongs in the gap. If there is no gap, the recordings are too short
or too noisy, and it will say so rather than inventing a number.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from botcirl.audio import AudioListener
from botcirl.config import DATA_DIR, AudioConfig, VoiceConfig
from botcirl.voices import build_voice_backend, voiceprint

CALIB_DIR = DATA_DIR / "voice_calib"


def record(speaker: str, count: int, min_s: float) -> None:
    cfg = VoiceConfig()
    backend = build_voice_backend(cfg)
    listener = AudioListener(AudioConfig()).start()

    print(f"\nrecording {count} utterances for {speaker!r}")
    print(f"say a full sentence each time - anything under {min_s:.1f}s is ignored")
    print("vary it: normal, quieter, turned away from the laptop. ctrl-c to stop early.\n")

    embeddings, durations = [], []
    try:
        while len(embeddings) < count:
            for seg in listener.poll():
                if seg.duration < min_s:
                    print(f"  (skipped {seg.duration:.1f}s - too short)")
                    continue
                vp = voiceprint(backend, seg)
                if vp is None:
                    continue
                embeddings.append(vp.embedding)
                durations.append(seg.duration)
                print(f"  [{len(embeddings)}/{count}] {seg.duration:.1f}s  "
                      f"quality {vp.quality():.2f}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        listener.stop()

    if not embeddings:
        sys.exit("nothing recorded")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIB_DIR / f"{speaker}.npz"
    np.savez(path, embeddings=np.stack(embeddings), durations=np.array(durations))
    print(f"\nsaved {len(embeddings)} utterances to {path}")

    if len(embeddings) >= 2:
        E = np.stack(embeddings)
        sims = [float(E[i] @ E[j]) for i, j in combinations(range(len(E)), 2)]
        print(f"self-similarity for {speaker}: min {min(sims):.3f}  "
              f"median {np.median(sims):.3f}  max {max(sims):.3f}")


def report() -> None:
    files = sorted(CALIB_DIR.glob("*.npz"))
    if not files:
        sys.exit(f"no recordings in {CALIB_DIR} - run with --speaker NAME first")

    data = {f.stem: np.load(f)["embeddings"] for f in files}
    print(f"\n{len(data)} speaker(s): " + ", ".join(f"{k} ({len(v)})" for k, v in data.items()))

    same, diff = [], []
    for name, E in data.items():
        same += [float(E[i] @ E[j]) for i, j in combinations(range(len(E)), 2)]
    for a, b in combinations(data, 2):
        Ea, Eb = data[a], data[b]
        diff += [float(x @ y) for x in Ea for y in Eb]

    def describe(label, xs):
        if not xs:
            print(f"  {label}: no pairs")
            return None
        xs = np.array(xs)
        print(f"  {label}: n={len(xs)}  min {xs.min():.3f}  p5 {np.percentile(xs,5):.3f}  "
              f"median {np.median(xs):.3f}  p95 {np.percentile(xs,95):.3f}  max {xs.max():.3f}")
        return xs

    print("\ncosine similarity")
    s = describe("same person ", same)
    d = describe("different   ", diff)

    if s is None or len(s) < 3:
        print("\nnot enough utterances per speaker - record more, they are the "
              "'same person' pairs the threshold depends on")
        return
    if d is None:
        print("\nonly one speaker recorded. Record a second person - without "
              "'different speaker' pairs there is no way to know what score means "
              "'not you', and a threshold from self-similarity alone is a guess.")
        print(f"\nprovisional, from same-person spread only:")
        print(f"  match_threshold  ~ {np.percentile(s, 5) - 0.05:.2f}  (below the 5th "
              "percentile of your own utterances)")
        return

    lo, hi = float(np.percentile(s, 5)), float(np.percentile(d, 95))
    print(f"\nsame-person 5th percentile : {lo:.3f}")
    print(f"different   95th percentile: {hi:.3f}")

    if lo > hi:
        match = (lo + hi) / 2
        print(f"\nclean separation - gap of {lo-hi:.3f}")
        print(f"  voice.match_threshold  = {match:.2f}")
        print(f"  voice.enroll_threshold = {max(0.05, hi - 0.05):.2f}")
        print("\nput those in botcirl/config.py under VoiceConfig.")
    else:
        overlap = hi - lo
        print(f"\nWARNING: the distributions overlap by {overlap:.3f} - these voices "
              "are not cleanly separable under these conditions.")
        print("  likely causes: utterances too short, background noise, or the mic "
              "picking up room reverb.")
        print(f"  a threshold of {hi:.2f} avoids merging people but will miss real "
              "matches;")
        print(f"  {lo:.2f} catches matches but will sometimes merge two people.")
        print("  record longer utterances (3s+) before trusting either.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", help="name to record under")
    ap.add_argument("--count", type=int, default=8, help="utterances to collect")
    ap.add_argument("--min-seconds", type=float, default=1.5)
    ap.add_argument("--report", action="store_true", help="analyse what is recorded")
    args = ap.parse_args()

    if args.report or not args.speaker:
        report()
    else:
        record(args.speaker, args.count, args.min_seconds)


if __name__ == "__main__":
    main()
