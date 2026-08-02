"""Local web dashboard for a running (or finished) session.

A browser rather than another OpenCV window, for two reasons that are not about
taste: scrubbing a timeline and playing back audio clips are things a browser
does natively and OpenCV does not do at all.

Serves on localhost only. The data is video and audio of people in a room, and
binding to 0.0.0.0 would put it on the network for anyone who asks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dashboard_page import PAGE


def open_in_chrome(url: str) -> None:
    """Open the dashboard in Google Chrome when available.

    Falls back to the system default browser only if Chrome is not installed.
    """
    if sys.platform == "darwin":
        for app in ("Google Chrome", "Chromium", "Google Chrome Canary"):
            try:
                subprocess.run(
                    ["open", "-a", app, url],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[dashboard] opened in {app}")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    elif sys.platform.startswith("linux"):
        for bin_name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            path = shutil.which(bin_name)
            if path:
                subprocess.Popen(
                    [path, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[dashboard] opened in {bin_name}")
                return
    elif sys.platform == "win32":
        chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        if not chrome.is_file():
            chrome = Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
        if chrome.is_file():
            subprocess.Popen([str(chrome), url])
            print("[dashboard] opened in Google Chrome")
            return

    # Last resort: whatever the OS thinks is the default browser.
    print("[dashboard] Chrome not found — opening system default browser")
    webbrowser.open(url)


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, recorder=None, is_running=None, **kwargs):
        self.recorder = recorder
        self.is_running = is_running
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:
        pass  # the console belongs to the event log, not to HTTP chatter

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab was closed mid-response; nothing to do about it

    def do_GET(self) -> None:  # noqa: N802 - http.server's interface
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return

        if path == "/api/session":
            blob = self.recorder.snapshot()
            blob["running"] = bool(self.is_running and self.is_running())
            self._send(200, json.dumps(blob).encode(), "application/json")
            return

        if path == "/api/frame.jpg":
            # Live camera preview. Only present while a session is recording.
            jpeg = None
            if hasattr(self.recorder, "latest_jpeg"):
                jpeg = self.recorder.latest_jpeg()
            if not jpeg:
                self._send(404, b"no frame yet", "text/plain")
                return
            self._send(200, jpeg, "image/jpeg")
            return

        if path == "/background.jpg":
            plate = self.recorder.dir / "background.jpg"
            if not plate.is_file():
                self._send(404, b"no plate yet", "text/plain")
                return
            self._send(200, plate.read_bytes(), "image/jpeg")
            return

        if path.startswith("/faces/"):
            # Per-session face thumbnails for the people gallery sidebar.
            name = Path(path).name
            pid = Path(name).stem  # p1.jpg -> p1
            jpeg = None
            if hasattr(self.recorder, "face_jpeg"):
                jpeg = self.recorder.face_jpeg(pid)
            if not jpeg:
                # Fall back to reading from the session faces dir directly
                # (review mode after rehydrate).
                faces_dir = getattr(self.recorder, "faces_dir", None)
                if faces_dir is not None:
                    try:
                        target = (Path(faces_dir) / f"{pid}.jpg").resolve()
                        target.relative_to(Path(faces_dir).resolve())
                        if target.is_file():
                            jpeg = target.read_bytes()
                    except (ValueError, OSError):
                        jpeg = None
            if not jpeg:
                self._send(404, b"no face", "text/plain")
                return
            self._send(200, jpeg, "image/jpeg")
            return

        if path.startswith("/audio/"):
            name = Path(path).name
            # Resolve and confirm containment: a crafted path must not be able
            # to read outside the session's audio directory.
            audio_dir = self.recorder.audio_dir.resolve()
            try:
                target = (audio_dir / name).resolve()
                target.relative_to(audio_dir)
            except (ValueError, OSError):
                self._send(403, b"forbidden", "text/plain")
                return
            if not target.is_file():
                self._send(404, b"no such clip", "text/plain")
                return
            self._send(200, target.read_bytes(), "audio/wav")
            return

        self._send(404, b"not found", "text/plain")


class Dashboard:
    def __init__(self, recorder, port: int = 8765, open_browser: bool = True):
        self.recorder = recorder
        self.port = port
        self.open_browser = open_browser
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> Dashboard:
        handler = partial(_Handler, recorder=self.recorder,
                          is_running=lambda: self._running)
        last_error: OSError | None = None
        for port in range(self.port, self.port + 10):
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
                self.port = port
                break
            except OSError as exc:  # port in use, try the next one
                last_error = exc
        if self._server is None:
            raise RuntimeError(f"no free port near {self.port}: {last_error}")

        self._thread = threading.Thread(
            target=self._server.serve_forever, name="dashboard", daemon=True
        )
        self._thread.start()
        print(f"[dashboard] {self.url}")
        if self.open_browser:
            open_in_chrome(self.url)
        return self

    def mark_finished(self) -> None:
        """Session over, but keep serving so it stays scrubbable."""
        self._running = False

    def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def serve_session(session_dir: Path, port: int = 8765) -> None:
    """Reopen a saved session for review, with no camera or microphone."""
    import time

    from .session import GestureRecord, SessionRecorder, SpeechRecord

    blob = json.loads((session_dir / "session.json").read_text())

    rec = SessionRecorder.__new__(SessionRecorder)  # rehydrate without a new dir
    rec.dir = session_dir
    rec.audio_dir = session_dir / "audio"
    rec.faces_dir = session_dir / "faces"
    rec.started_at = blob["started_at"]
    rec.ended_at = blob.get("ended_at")
    rec._lock = threading.Lock()
    rec.positions = [tuple(p) for p in blob["positions"]]
    rec.labels = blob["labels"]
    rec.calibrated = blob["calibrated"]
    rec.speech = [
        SpeechRecord(id=s["id"], start=s["start"], end=s["end"], pid=s["pid"],
                     label=s["label"], voice_pid=s.get("voice_pid"),
                     confidence=s["confidence"], basis=s["basis"],
                     audio_file=s.get("audio"))
        for s in blob["speech"]
    ]
    rec.gestures = [
        GestureRecord(pid=g["pid"], label=g["label"], start=g["start"], end=g["end"])
        for g in blob["gestures"]
    ]
    # Restore the people gallery for the sidebar. Photos live as files under
    # faces/; meta is either in session.json or rebuilt from labels.
    rec.people = {}
    for person in blob.get("people") or []:
        pid = person.get("pid")
        if not pid:
            continue
        photo = person.get("photo")
        if not photo and (rec.faces_dir / f"{pid}.jpg").is_file():
            photo = f"/faces/{pid}.jpg"
        rec.people[pid] = {
            "pid": pid,
            "label": person.get("label") or pid,
            "first_seen": person.get("first_seen") or blob["started_at"],
            "last_seen": person.get("last_seen") or blob.get("ended_at") or blob["started_at"],
            "sightings": person.get("sightings") or 0,
            "photo": photo,
            "photo_quality": 1.0 if photo else -1.0,
            "present": False,
        }
    if not rec.people:
        for pid, label in (blob.get("labels") or {}).items():
            if pid.startswith("#"):
                continue
            photo = f"/faces/{pid}.jpg" if (rec.faces_dir / f"{pid}.jpg").is_file() else None
            rec.people[pid] = {
                "pid": pid, "label": label, "first_seen": blob["started_at"],
                "last_seen": blob.get("ended_at") or blob["started_at"],
                "sightings": 0, "photo": photo,
                "photo_quality": 1.0 if photo else -1.0, "present": False,
            }
    rec._latest_jpeg = None
    rec.hud = blob.get("hud") or {
        "fps": 0.0, "people": 0, "named": 0,
        "gallery": len(rec.people),
        "floor": "metres" if rec.calibrated else "uncalibrated",
        "voting": [],
    }
    rec._saved = True

    dash = Dashboard(rec, port=port).start()
    dash.mark_finished()
    print("reviewing saved session - ctrl-c to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dash.stop()
