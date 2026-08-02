#!/usr/bin/env python3
"""Reopen a recorded session in the dashboard. No camera, no microphone.

    python review.py              # the most recent session
    python review.py --list
    python review.py data/sessions/20260730-142233
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from botcirl.config import DATA_DIR
from botcirl.dashboard import serve_session

SESSIONS = DATA_DIR / "sessions"


def sessions() -> list[Path]:
    if not SESSIONS.exists():
        return []
    return sorted((d for d in SESSIONS.iterdir() if (d / "session.json").exists()))


def describe(d: Path) -> str:
    blob = json.loads((d / "session.json").read_text())
    dur = blob.get("ended_at", blob["now"]) - blob["started_at"]
    clips = len(list((d / "audio").glob("*.wav"))) if (d / "audio").exists() else 0
    return (f"{d.name}  {dur/60:5.1f} min  {len(blob['labels']):2d} people  "
            f"{len(blob['speech']):3d} utterances  {clips} clips"
            f"{'  [floor calibrated]' if blob.get('calibrated') else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="session directory")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    found = sessions()
    if args.list:
        if not found:
            sys.exit(f"no sessions in {SESSIONS}")
        for d in found:
            print(" ", describe(d))
        return

    target = Path(args.session) if args.session else (found[-1] if found else None)
    if target is None:
        sys.exit(f"no sessions recorded yet - run run_vision.py first ({SESSIONS})")
    if not (target / "session.json").exists():
        sys.exit(f"{target} does not look like a session directory")

    print(describe(target))
    serve_session(target, port=args.port)


if __name__ == "__main__":
    main()
