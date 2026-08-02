#!/usr/bin/env python3
"""Check the microphone and voice detection, without any video.

    python listen_test.py            # listen for 20 seconds
    python listen_test.py --seconds 45

Talk, pause, talk again. Each utterance should print as one segment. Silence and
background noise should print nothing - Silero is a voice model, not a loudness
gate, so a noisy room is not the same as a talking room.

Segments are written to data/speech/ as wav files so you can listen back and
check the boundaries are where you expect.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from botcirl.audio import AudioListener
from botcirl.config import DATA_DIR, AudioConfig


def bar(level: float, width: int = 28) -> str:
    filled = int(min(1.0, level / 0.15) * width)
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--device", default=None, help="input device index or name")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = AudioConfig()
    if args.device is not None:
        cfg.device = int(args.device) if args.device.isdigit() else args.device

    out_dir = DATA_DIR / "speech"
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        listener = AudioListener(cfg).start()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Could not open the microphone: {exc}\n"
            "On macOS the terminal app needs microphone access:\n"
            "  System Settings > Privacy & Security > Microphone > enable your terminal,\n"
            "then fully quit and reopen it."
        )

    print(f"listening for {args.seconds:.0f}s - say something, pause, say something else")
    print("(ctrl-c to stop early)\n")

    t0 = time.time()
    segments = []
    try:
        while time.time() - t0 < args.seconds:
            for seg in listener.poll():
                segments.append(seg)
                rel = seg.start - t0
                print(f"  [{rel:5.1f}s] speech for {seg.duration:5.2f}s  "
                      f"level {bar(seg.rms())} {seg.rms():.4f}")
                if not args.no_save:
                    import soundfile as sf
                    path = out_dir / f"seg_{len(segments):03d}_{seg.duration:.1f}s.wav"
                    sf.write(path, seg.audio, seg.samplerate)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        listener.stop()

    print(f"\n{len(segments)} speech segments in {time.time()-t0:.0f}s")
    if segments:
        total = sum(s.duration for s in segments)
        print(f"  total speech {total:.1f}s ({total/(time.time()-t0)*100:.0f}% of the time)")
        print(f"  shortest {min(s.duration for s in segments):.2f}s, "
              f"longest {max(s.duration for s in segments):.2f}s")
        if not args.no_save:
            print(f"  wavs written to {out_dir}")
    else:
        print("  nothing detected. if you were definitely talking, the mic may be muted,\n"
              "  or the terminal may lack microphone permission.")


if __name__ == "__main__":
    main()
