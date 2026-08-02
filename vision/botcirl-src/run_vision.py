#!/usr/bin/env python3
"""Live room perception on the laptop webcam.

    python run_vision.py                 # webcam → Chrome dashboard only
    python run_vision.py --source clip.mp4
    python run_vision.py --window        # also show the OpenCV preview window

UI is the Chrome dashboard by default (no separate camera window).
With --window, while the OpenCV preview is focused:
    1-9   name the Nth person (left to right), then type and press enter
    s     save the gallery now
    q     quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2

from botcirl import Config, Pipeline
from botcirl.config import EVENT_LOG
from botcirl.dashboard import Dashboard
from botcirl.background import BackgroundPlate
from botcirl.session import SessionRecorder
from botcirl.viz import render

WINDOW = "botcirl - room perception"


def open_capture(source: str, cfg: Config) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(
            f"Could not open source {source!r}.\n"
            "If this is the webcam on macOS, the *terminal app* needs camera access:\n"
            "  System Settings > Privacy & Security > Camera > enable your terminal,\n"
            "then fully quit and reopen the terminal (not just this shell)."
        )
    return cap


def list_devices() -> None:
    """Show what cameras and microphones are attached, with their indices."""
    print("microphones:")
    try:
        import sounddevice as sd

        default_in = sd.default.device[0]
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] < 1:
                continue
            mark = " (default)" if i == default_in else ""
            print(f"  {i:2d}  {d['name']}  {d['max_input_channels']}ch "
                  f"@{int(d['default_samplerate'])}Hz{mark}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not enumerate audio devices: {exc}")

    print("\ncameras (index -> resolution it opened at):")
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"  {i:2d}  {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    print("\nnetwork cameras: pass the stream URL to --source, e.g.")
    print("  --source http://192.168.1.42:8080/video      (phone running IP Webcam)")
    print("  --source rtsp://192.168.1.50:554/stream1     (wifi camera)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--face-backend", default="auto", choices=["auto", "insightface", "opencv"])
    ap.add_argument("--window", action="store_true",
                    help="also open the native OpenCV camera window (off by default)")
    ap.add_argument("--headless", action="store_true",
                    help="deprecated alias: same as default (no OpenCV window)")
    ap.add_argument("--no-flip", action="store_true", help="do not mirror the preview")
    ap.add_argument("--max-fps", type=float, default=30.0)
    ap.add_argument("--no-audio", action="store_true", help="vision only, do not open the mic")
    ap.add_argument("--audio-device", default=None,
                    help="microphone index or name (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true",
                    help="show available cameras and microphones, then exit")
    ap.add_argument("--no-dashboard", action="store_true", help="do not serve the web dashboard")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    cfg = Config()
    cfg.camera.flip = not args.no_flip and args.source.isdigit()
    # Chrome dashboard is the UI. OpenCV window is opt-in via --window.
    cfg.show_window = bool(args.window) and not args.headless
    # Independent of where the video comes from. A robot streaming its camera
    # over the network still has a microphone, and tying the two together
    # silently dropped the audio half of the system.
    cfg.audio.enabled = not args.no_audio
    if args.audio_device is not None:
        cfg.audio.device = (int(args.audio_device) if args.audio_device.isdigit()
                            else args.audio_device)

    # Session first so the face gallery lives under this recording's directory
    # (Person 1 / Person 2 are scoped to this run, not a global lifetime list).
    recorder = SessionRecorder()
    pipeline = Pipeline(
        cfg,
        face_backend=args.face_backend,
        gallery_path=recorder.dir / "gallery",
    )
    cap = open_capture(args.source, cfg)

    speech = None
    if cfg.audio.enabled:
        try:
            from botcirl.speech import SpeechTracker

            speech = SpeechTracker(cfg, pipeline).start()
        except Exception as exc:  # noqa: BLE001
            # Losing the microphone must not take the camera down with it.
            print(f"[speech] disabled ({type(exc).__name__}: {exc})")
            speech = None

    plate = BackgroundPlate()
    last_plate_write = 0.0
    dashboard = None
    if not args.no_dashboard:
        try:
            dashboard = Dashboard(recorder, port=args.port).start()
        except Exception as exc:  # noqa: BLE001
            print(f"[dashboard] disabled ({type(exc).__name__}: {exc})")

    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = EVENT_LOG.open("a", buffering=1)

    mode, buffer, naming_pid = "idle", "", None
    min_dt = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    if dashboard is not None:
        print(f"[run] streaming in Chrome at {dashboard.url} — ctrl-c here to stop")
    elif cfg.show_window:
        print("[run] streaming - press q in the window (or ctrl-c here) to stop")
    else:
        print("[run] streaming headless - ctrl-c to stop")

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[run] source ended")
                break
            if cfg.camera.flip:
                frame = cv2.flip(frame, 1)

            scene = pipeline.process(frame)

            events = pipeline.drain_events()
            if speech is not None:
                for att in speech.poll():
                    recorder.add_speech(att, speech.speaking.label if speech.speaking
                                        else "unknown", cfg.audio.samplerate)
                events.extend(speech.drain_events())
            recorder.sample(pipeline)
            # Live dashboard preview. Push the camera frame (not the empty-room
            # plate) so the browser tab tracks motion in real time.
            recorder.push_frame(frame)
            # Per-session people gallery + face thumbnails for the sidebar.
            recorder.sync_people(pipeline, frame)

            # Grow the empty-room plate. Throttled: the average moves slowly by
            # design, so touching it every frame just burns cycles.
            if scene.frame_idx % 5 == 0:
                plate.update(frame, [t.bbox for t in scene.people])
                recorder.frame_size = pipeline.frame_size
                recorder.background_coverage = plate.coverage()
            if time.time() - last_plate_write > 4.0 and plate.updates > 8:
                plate.save(recorder.dir / "background.jpg")
                recorder.background = "/background.jpg"
                last_plate_write = time.time()

            for event in events:
                log.write(json.dumps(event) + "\n")
                print(f"[event] {event['event']}: "
                      + ", ".join(f"{k}={v}" for k, v in event.items() if k not in ("t", "event")))

            if cfg.show_window:
                canvas = render(frame, scene, pipeline, mode, buffer, speech)
                cv2.imshow(WINDOW, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    pass
                elif mode == "naming":
                    if key in (13, 10):  # enter
                        if buffer.strip() and naming_pid:
                            pipeline.gallery.rename(naming_pid, buffer.strip())
                            print(f"[run] {naming_pid} is now {buffer.strip()!r}")
                        mode, buffer, naming_pid = "idle", "", None
                    elif key == 27:  # esc
                        mode, buffer, naming_pid = "idle", "", None
                    elif key in (8, 127):  # backspace
                        buffer = buffer[:-1]
                    elif 32 <= key < 127:
                        buffer += chr(key)
                elif key == ord("q") or key == 27:
                    break
                elif key == ord("s"):
                    pipeline.gallery.save()
                    print("[run] gallery saved")
                elif ord("1") <= key <= ord("9"):
                    ordered = sorted(scene.people, key=lambda t: t.bbox[0])
                    idx = key - ord("1")
                    if idx < len(ordered):
                        trk = ordered[idx]
                        if trk.pid is None:
                            print("[run] that person has no face-based identity yet - "
                                  "get their face in frame first")
                        else:
                            mode, naming_pid = "naming", trk.pid
                            buffer = pipeline.gallery.label_of(trk.pid)
                            if buffer.startswith("Person "):
                                buffer = ""

            if min_dt:
                slack = min_dt - (time.time() - loop_start)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        if speech is not None:
            speech.stop()
        plate.save(recorder.dir / "background.jpg")
        recorder.background = "/background.jpg"
        recorder.frame_size = pipeline.frame_size
        session_dir = recorder.save()
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()
        log.close()
        print(f"[run] gallery: {len(pipeline.gallery.identities)} identities")
        for pid, idt in pipeline.gallery.identities.items():
            print(f"  {pid:>4}  {idt.label:<20} {len(idt.embeddings)} views, {idt.sightings} sightings")
        if speech is not None:
            print("[run] " + speech.summary().replace("\n", "\n[run] "))
        if pipeline.votes:
            from collections import Counter
            tally = Counter(v["label"] for v in pipeline.votes)
            total = sum(v["duration_s"] for v in pipeline.votes)
            print(f"[run] votes: {len(pipeline.votes)} raised hands, {total:.0f}s of hands up")
            for who, n in tally.most_common():
                held = sum(v["duration_s"] for v in pipeline.votes if v["label"] == who)
                print(f"         {who:<20} {n:2d}  ({held:.0f}s)")
        print(f"[run] events appended to {EVENT_LOG}")
        print(f"[run] session saved to {session_dir}")
        if dashboard is not None:
            dashboard.mark_finished()
            print(f"[run] dashboard still up at {dashboard.url} - ctrl-c to close")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                dashboard.stop()


if __name__ == "__main__":
    main()
