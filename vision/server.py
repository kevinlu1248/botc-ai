#!/usr/bin/env python3
"""Room-perception sidecar for BOTC AI.

Runs the botcirl vision pipeline (people, face gallery, looking-at-camera) and
serves a live annotated JPEG + JSON state for the React UI and Node STT gate.

Low-latency design:
  - camera buffer drained every tick (never process a queued-up old frame)
  - 640×360 capture, small YOLO imgsz, faces every 3rd frame
  - JPEG encode always uses the *latest* annotated frame only
  - browser must fetch frames one-at-a-time (no request pile-up)

    vision/.venv/bin/python vision/server.py
    → http://127.0.0.1:8766/api/state
    → http://127.0.0.1:8766/api/frame.jpg
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "botcirl-src"
sys.path.insert(0, str(SRC))

from botcirl import Config, Pipeline  # noqa: E402
from botcirl.session import SessionRecorder  # noqa: E402
from botcirl.viz import draw_person  # noqa: E402


PORT = int(os.environ.get("VISION_PORT", "8766"))
FACE_BACKEND = os.environ.get("VISION_FACE_BACKEND", "opencv")
CAMERA = os.environ.get("VISION_CAMERA", "0")
# Preview encode width — smaller = less lag through the browser.
PREVIEW_W = int(os.environ.get("VISION_PREVIEW_W", "640"))
JPEG_QUALITY = int(os.environ.get("VISION_JPEG_QUALITY", "55"))


class VisionRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.jpeg: bytes | None = None
        self.frame_seq = 0
        self.state: dict = {
            "running": False,
            "fps": 0.0,
            "people": [],
            "looking": [],
            "error": None,
            "seq": 0,
        }
        self.recorder = SessionRecorder(root=ROOT / "data" / "sessions")
        cfg = Config()
        cfg.show_window = False
        cfg.audio.enabled = False  # botc-ai owns the mic / STT path
        cfg.camera.flip = True
        # Prefer low latency over resolution — 640×360 is enough for looking + ID.
        # 960×540: better close-up faces than 360p, still light enough for low lag.
        cfg.camera.width = int(os.environ.get("VISION_WIDTH", "960"))
        cfg.camera.height = int(os.environ.get("VISION_HEIGHT", "540"))
        cfg.person.imgsz = int(os.environ.get("VISION_IMGSZ", "480"))
        cfg.face.every_n_frames = int(os.environ.get("VISION_FACE_EVERY", "2"))
        cfg.face.det_size = int(os.environ.get("VISION_FACE_DET", "480"))
        cfg.face.min_face_px = int(os.environ.get("VISION_MIN_FACE", "28"))
        cfg.face.looking_enter = float(os.environ.get("VISION_LOOK_ENTER", "0.38"))
        cfg.face.looking_exit = float(os.environ.get("VISION_LOOK_EXIT", "0.22"))
        cfg.face.looking_hold_s = float(os.environ.get("VISION_LOOK_HOLD", "0.15"))
        cfg.face.looking_release_s = float(os.environ.get("VISION_LOOK_RELEASE", "1.6"))
        cfg.face.looking_sticky_s = float(os.environ.get("VISION_LOOK_STICKY", "2.5"))
        # Prefer a local weight file if present; else let Ultralytics download
        # yolo11n-pose.pt on first run.
        pose = SRC / "yolo11n-pose.pt"
        if pose.is_file() or pose.is_symlink():
            cfg.person.model = str(pose.resolve())
        else:
            cfg.person.model = "yolo11n-pose.pt"
        self.cfg = cfg
        self.pipeline = Pipeline(
            cfg,
            face_backend=FACE_BACKEND,
            gallery_path=self.recorder.dir / "gallery",
        )
        self.cap = None

    def open_camera(self) -> None:
        source = CAMERA
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
            # Critical for lag: don't buffer stale frames while we process.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {source!r}. Grant camera access to the "
                "terminal app (System Settings → Privacy & Security → Camera)."
            )

    def grab_latest(self):
        """Read the freshest frame; discard anything queued in the driver."""
        if not self.cap:
            return False, None
        # grab() is cheap (no decode); retrieve() decodes only the last one.
        ok = False
        for _ in range(4):
            grabbed = self.cap.grab()
            if not grabbed:
                break
            ok = True
        if not ok:
            return self.cap.read()
        return self.cap.retrieve()

    def annotate(self, frame, scene):
        img = frame  # mutate in place — we don't need the raw copy after this
        ordered = sorted(scene.people, key=lambda t: t.bbox[0])
        for i, trk in enumerate(ordered, start=1):
            label = self.pipeline.label_for(trk)
            draw_person(img, trk, label, i, known=trk.pid is not None)
        looking_n = sum(1 for t in scene.people if t.looking)
        named = sum(1 for t in scene.people if t.pid)
        h, w = img.shape[:2]
        strip = 28
        band = img[h - strip : h]
        band[:] = (band * 0.3).astype(band.dtype)
        line = (
            f"{scene.fps:4.1f} fps  people {len(scene.people)} ({named} named)  "
            f"looking {looking_n}  gallery {len(self.pipeline.gallery.identities)}"
        )
        cv2.putText(
            img, line, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
            cv2.LINE_AA,
        )
        return img

    def people_payload(self) -> list[dict]:
        out = []
        for pid, idt in self.pipeline.gallery.identities.items():
            present = looking = False
            score = 0.0
            box = None
            for trk in self.pipeline.tracks.values():
                if trk.pid == pid and trk.announced:
                    present = True
                    looking = bool(trk.looking)
                    score = float(trk.looking_score or 0.0)
                    box = [int(v) for v in trk.bbox]
                    break
            photo = None
            face_path = self.recorder.faces_dir / f"{pid}.jpg"
            if face_path.is_file():
                photo = f"/api/vision/faces/{pid}.jpg"
            out.append({
                "pid": pid,
                "label": idt.label,
                "present": present,
                "looking": looking,
                "looking_score": round(score, 2),
                "sightings": idt.sightings,
                "photo": photo,
                "box": box,
            })
        for trk in self.pipeline.tracks.values():
            if trk.announced and not trk.pid:
                out.append({
                    "pid": f"#{trk.track_id}",
                    "label": f"#{trk.track_id}",
                    "present": True,
                    "looking": bool(trk.looking),
                    "looking_score": round(float(trk.looking_score or 0.0), 2),
                    "sightings": 0,
                    "photo": None,
                    "box": [int(v) for v in trk.bbox],
                })
        out.sort(key=lambda p: (0 if str(p["pid"]).startswith("p") else 1, str(p["pid"])))
        return out

    def publish(self, annotated, scene, now: float) -> None:
        h, w = annotated.shape[:2]
        if w > PREVIEW_W:
            scale = PREVIEW_W / w
            small = cv2.resize(
                annotated, (PREVIEW_W, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = annotated
        ok_j, buf = cv2.imencode(
            ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        people = self.people_payload()
        looking = [p for p in people if p["looking"] and p["present"]]
        with self.lock:
            self.frame_seq += 1
            if ok_j:
                self.jpeg = buf.tobytes()
            self.state = {
                "running": True,
                "fps": round(float(scene.fps or 0.0), 1),
                "people": people,
                "looking": looking,
                "gallery_size": len(self.pipeline.gallery.identities),
                "error": None,
                "ts": now,
                "seq": self.frame_seq,
            }

    def loop(self) -> None:
        try:
            self.open_camera()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.state = {**self.state, "running": False, "error": str(exc)}
            print(f"[vision] {exc}")
            return

        print(
            f"[vision] camera open backend={FACE_BACKEND} "
            f"{self.cfg.camera.width}x{self.cfg.camera.height} "
            f"imgsz={self.cfg.person.imgsz} face_every={self.cfg.face.every_n_frames} "
            f"port={PORT}"
        )
        tick = 0
        while self.running:
            ok, frame = self.grab_latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if self.cfg.camera.flip:
                frame = cv2.flip(frame, 1)

            scene = self.pipeline.process(frame)
            # Bookkeeping is cheaper than it looks; still throttle face crops.
            self.recorder.sample(self.pipeline)
            tick += 1
            if tick % 2 == 0:
                self.recorder.sync_people(self.pipeline, frame)

            annotated = self.annotate(frame, scene)
            self.publish(annotated, scene, time.time())

        if self.cap:
            self.cap.release()
        self.pipeline.close()
        try:
            self.recorder.save()
        except Exception:  # noqa: BLE001
            pass
        print("[vision] stopped")

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def frame_jpeg(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def face_jpeg(self, pid: str) -> bytes | None:
        if not pid or not pid.replace("_", "").isalnum():
            return None
        path = self.recorder.faces_dir / f"{pid}.jpg"
        if not path.is_file():
            return None
        return path.read_bytes()


RUNTIME = VisionRuntime()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        if path in ("/", "/api/state", "/api/vision/state"):
            body = json.dumps(RUNTIME.snapshot()).encode()
            self._send(200, body, "application/json")
            return

        if path in ("/api/frame.jpg", "/api/vision/frame.jpg"):
            jpeg = RUNTIME.frame_jpeg()
            if not jpeg:
                self._send(404, b"no frame yet", "text/plain")
                return
            self._send(200, jpeg, "image/jpeg")
            return

        if path.startswith("/api/vision/faces/") or path.startswith("/faces/"):
            name = Path(path).name
            pid = Path(name).stem
            jpeg = RUNTIME.face_jpeg(pid)
            if not jpeg:
                self._send(404, b"no face", "text/plain")
                return
            self._send(200, jpeg, "image/jpeg")
            return

        if path == "/health":
            self._send(200, b"ok", "text/plain")
            return

        self._send(404, b"not found", "text/plain")


def main() -> None:
    thread = threading.Thread(target=RUNTIME.loop, name="vision-loop", daemon=True)
    thread.start()
    # ThreadingHTTPServer so a slow client download can't stall the next frame.
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[vision] serving http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[vision] shutting down")
    finally:
        RUNTIME.running = False
        server.shutdown()


if __name__ == "__main__":
    main()
