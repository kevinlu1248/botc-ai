"""Recording a session: where everyone walked, and who spoke when.

Kept separate from the live pipeline on purpose. The pipeline is about the
present moment - what is in this frame - and deliberately forgets things. A
session is the opposite: an append-only record you can scrub back through.

Everything lands in `data/sessions/<timestamp>/`:

    session.json   positions, speech, gestures, labels
    audio/NNN.wav  one file per utterance, so you can play back what was said

Positions are stored in metres when the floor is calibrated and in normalised
image coordinates when it is not. `calibrated` says which, and the dashboard
draws a real plan view or an approximation accordingly - never silently passing
one off as the other.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR


@dataclass
class SpeechRecord:
    id: int
    start: float
    end: float
    pid: str | None
    label: str
    voice_pid: str | None
    confidence: float
    basis: str
    audio_file: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "pid": self.pid,
            "label": self.label,
            "voice_pid": self.voice_pid,
            "confidence": round(self.confidence, 2),
            "basis": self.basis,
            "audio": f"/audio/{self.id:04d}.wav" if self.audio_file else None,
        }


@dataclass
class GestureRecord:
    pid: str | None
    label: str
    start: float
    end: float | None = None

    def as_dict(self) -> dict:
        return {
            "pid": self.pid,
            "label": self.label,
            "start": round(self.start, 3),
            "end": round(self.end, 3) if self.end else None,
        }


class SessionRecorder:
    """Accumulates a scrubbable history of a run."""

    def __init__(self, root: Path | None = None, sample_hz: float = 5.0):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = (root or DATA_DIR / "sessions") / stamp
        self.audio_dir = self.dir / "audio"
        self.started_at = time.time()
        self.sample_period = 1.0 / sample_hz
        self._last_sample = 0.0
        self._lock = threading.Lock()

        # Flat arrays keep the JSON small: one row is [t, track_id, x, y].
        # Stored by *track*, resolved to an identity only when read - see
        # `_key_for`. A person is usually seen before they are recognised, and
        # keying on identity at write time would split their trail in two.
        self.positions: list[tuple[float, int, float, float]] = []
        self.identity_of: dict[int, str] = {}
        # Image-space pose per sample, for the camera-view reconstruction. With
        # a fixed camera these coordinates *are* the correct perspective - they
        # were measured, not modelled - so replaying them needs no reprojection
        # and does not depend on the shaky focal-length estimate.
        self.poses: list[dict] = []
        self.frame_size: tuple[int, int] | None = None
        self.background: str | None = None
        self.background_coverage: float = 0.0
        self.speech: list[SpeechRecord] = []
        self.gestures: list[GestureRecord] = []
        self._open_gestures: dict[str, GestureRecord] = {}
        self.labels: dict[str, str] = {}
        self.calibrated = False
        self.camera: dict | None = None  # where the camera stands, in room metres
        self.fov: list | None = None     # the patch of floor it can see
        self.ended_at: float | None = None
        self._saved = False
        # Latest camera frame as JPEG for the live dashboard preview. Not part of
        # the saved session - only kept in memory while recording.
        self._latest_jpeg: bytes | None = None
        self._last_jpeg_at = 0.0
        self._jpeg_period = 1.0 / 12.0  # ~12 fps is plenty for a web preview
        # Per-session people gallery: who has been identified this run, with a
        # face crop for the dashboard sidebar. Scoped to this session dir so
        # Person 1 in one recording is not mixed with Person 1 from another.
        self.faces_dir = self.dir / "faces"
        self.people: dict[str, dict] = {}  # pid -> meta for the sidebar
        # Live HUD line mirrored from the OpenCV overlay (fps, counts, floor).
        self.hud: dict = {
            "fps": 0.0,
            "people": 0,
            "named": 0,
            "gallery": 0,
            "floor": "uncalibrated",
            "voting": [],
        }

    # ---------- recording ----------

    def push_frame(self, frame_bgr: np.ndarray, now: float | None = None) -> None:
        """Publish the current camera frame for the live dashboard preview.

        Throttled and JPEG-encoded so the video loop is not blocked by serving
        full-rate raw frames over HTTP. Scrubbing still uses the pose history;
        this is only for "what does the camera see right now".
        """
        import cv2

        now = now or time.time()
        if now - self._last_jpeg_at < self._jpeg_period:
            return
        self._last_jpeg_at = now
        # Encode a downscaled copy - dashboard canvas is ~900px wide, full 720p
        # is wasted bandwidth and makes the poll feel sticky.
        h, w = frame_bgr.shape[:2]
        if w > 960:
            scale = 960.0 / w
            small = cv2.resize(frame_bgr, (960, max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = buf.tobytes()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def sync_people(self, pipeline, frame_bgr: np.ndarray,
                    now: float | None = None) -> None:
        """Update the session people gallery from the live pipeline.

        Every identity the face gallery knows about appears in the sidebar.
        When a track has a decent face crop, we keep the best-quality one as
        their thumbnail for this session.
        """
        import cv2

        now = now or time.time()
        h, w = frame_bgr.shape[:2]
        gallery = getattr(pipeline, "gallery", None)
        if gallery is None:
            return

        with self._lock:
            # Ensure every enrolled identity is listed, even before a crop lands.
            for pid, idt in gallery.identities.items():
                rec = self.people.get(pid)
                if rec is None:
                    rec = {
                        "pid": pid,
                        "label": idt.label,
                        "first_seen": idt.first_seen,
                        "last_seen": idt.last_seen,
                        "sightings": idt.sightings,
                        "photo": None,
                        "photo_quality": -1.0,
                        "present": False,
                        "looking": False,
                        "looking_score": 0.0,
                    }
                    self.people[pid] = rec
                else:
                    rec["label"] = idt.label
                    rec["last_seen"] = idt.last_seen
                    rec["sightings"] = idt.sightings
                rec["present"] = False
                rec["looking"] = False

            # Mark who is in frame right now and refresh face crops when better.
            for trk in pipeline.tracks.values():
                if not trk.pid or trk.pid not in self.people:
                    continue
                rec = self.people[trk.pid]
                if trk.announced:
                    rec["present"] = True
                    rec["last_seen"] = now
                rec["looking"] = bool(getattr(trk, "looking", False))
                rec["looking_score"] = round(float(getattr(trk, "looking_score", 0.0)), 2)
                face = getattr(trk, "face", None)
                if face is None:
                    continue
                quality = float(face.quality())
                # Only replace the stored crop when this view is clearly better.
                if quality < 0.4 or quality < rec.get("photo_quality", -1.0) + 0.05:
                    continue
                x1, y1, x2, y2 = face.bbox
                # Pad a little so the crop is a head, not a tight face box.
                fw, fh = x2 - x1, y2 - y1
                pad_x, pad_y = int(fw * 0.25), int(fh * 0.35)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                if x2 - x1 < 24 or y2 - y1 < 24:
                    continue
                crop = frame_bgr[y1:y2, x1:x2]
                # Square-ish thumbnail for the sidebar.
                side = max(crop.shape[0], crop.shape[1])
                thumb = np.zeros((side, side, 3), dtype=np.uint8)
                oy = (side - crop.shape[0]) // 2
                ox = (side - crop.shape[1]) // 2
                thumb[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
                if side > 160:
                    thumb = cv2.resize(thumb, (160, 160), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", thumb,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue
                self.faces_dir.mkdir(parents=True, exist_ok=True)
                path = self.faces_dir / f"{trk.pid}.jpg"
                path.write_bytes(buf.tobytes())
                rec["photo"] = f"/faces/{trk.pid}.jpg"
                rec["photo_quality"] = quality

    def face_jpeg(self, pid: str) -> bytes | None:
        """Read a saved face thumbnail for the dashboard."""
        # Only allow simple identity ids (p1, p2, …) - never path segments.
        if not pid or not pid.replace("_", "").isalnum():
            return None
        try:
            root = self.faces_dir.resolve()
            path = (self.faces_dir / f"{pid}.jpg").resolve()
            path.relative_to(root)
        except (ValueError, OSError):
            return None
        if not path.is_file():
            return None
        return path.read_bytes()

    def sample(self, pipeline, now: float | None = None) -> None:
        """Take a positional snapshot, at most `sample_hz` times a second.

        Sampling rather than storing every frame: at 11 fps a ten-minute session
        is thousands of near-identical points per person, which makes the
        timeline slow to draw and no more informative.
        """
        now = now or time.time()
        if now - self._last_sample < self.sample_period:
            return
        self._last_sample = now
        self.calibrated = pipeline.floor.valid
        if self.calibrated and self.camera is None:
            self.camera = pipeline.floor.load_camera()
            frame_size = getattr(pipeline, "frame_size", None)
            if frame_size:
                self.fov = [list(p) for p in pipeline.floor.visible_ground(frame_size)]

        # Same stats the OpenCV HUD prints, so the web app can show them too.
        live = list(getattr(pipeline, "tracks", {}).values())
        named = sum(1 for t in live if getattr(t, "pid", None))
        gallery = getattr(pipeline, "gallery", None)
        voting_labels = []
        if hasattr(pipeline, "voting"):
            try:
                voting_labels = [pipeline.label_for(t) for t in pipeline.voting()]
            except Exception:  # noqa: BLE001
                voting_labels = []
        looking_n = sum(1 for t in live if getattr(t, "looking", False))
        self.hud = {
            "fps": round(float(getattr(pipeline, "_fps", 0.0) or 0.0), 1),
            "people": len(live),
            "named": named,
            "looking": looking_n,
            "gallery": len(gallery.identities) if gallery is not None else 0,
            "floor": "metres" if self.calibrated else "uncalibrated",
            "voting": voting_labels,
        }

        with self._lock:
            for trk in pipeline.tracks.values():
                if not trk.announced:
                    continue
                if trk.pid:
                    # Retroactive: every earlier sample from this track now
                    # resolves to the identity too, so the trail is continuous
                    # across the moment of recognition - and across a track id
                    # break, which is the same person either way.
                    self.identity_of[trk.track_id] = trk.pid
                key = self._key_for(trk.track_id)
                self.labels[key] = pipeline.label_for(trk)

                if trk.room is not None:
                    x, y = trk.room
                else:
                    # No floor calibration: fall back to image space, normalised
                    # so the dashboard can still draw something meaningful.
                    fx, fy = trk.foot
                    x, y = fx, fy
                self.positions.append((now, trk.track_id, float(x), float(y)))

                pose = {"t": round(now, 2), "id": trk.track_id,
                        "box": [int(v) for v in trk.bbox]}
                if trk.kps is not None and trk.kp_conf is not None:
                    # Rounded to whole pixels and confidence to one decimal:
                    # sub-pixel precision is meaningless here and triples the
                    # size of the session file.
                    pose["kp"] = [[int(x), int(y), round(float(c), 1)]
                                  for (x, y), c in zip(trk.kps.tolist(), trk.kp_conf.tolist())]
                if trk.hand_raised:
                    pose["up"] = 1
                if getattr(trk, "looking", False):
                    pose["looking"] = 1
                    pose["look"] = round(float(getattr(trk, "looking_score", 0.0)), 2)
                self.poses.append(pose)

                # Hand raises become spans, not points, so they can be shown as
                # bars on the timeline alongside speech.
                if trk.hand_raised and key not in self._open_gestures:
                    rec = GestureRecord(pid=key, label=pipeline.label_for(trk),
                                        start=trk.raised_since or now)
                    self._open_gestures[key] = rec
                    self.gestures.append(rec)
                elif not trk.hand_raised and key in self._open_gestures:
                    self._open_gestures.pop(key).end = now

            # Close gestures for anyone who vanished mid-raise.
            live = {self._key_for(t.track_id) for t in pipeline.tracks.values()}
            for pid in list(self._open_gestures):
                if pid not in live:
                    self._open_gestures.pop(pid).end = now

    def add_speech(self, attribution, label: str, samplerate: int = 16000) -> SpeechRecord:
        seg = attribution.segment
        with self._lock:
            rec = SpeechRecord(
                id=len(self.speech),
                start=seg.start,
                end=seg.end,
                pid=attribution.speaker_pid,
                label=label,
                voice_pid=attribution.voice_pid,
                confidence=attribution.confidence,
                basis=attribution.basis,
            )
            self.speech.append(rec)

        # Written outside the lock: encoding a wav should not stall the video loop.
        try:
            import soundfile as sf

            self.audio_dir.mkdir(parents=True, exist_ok=True)
            path = self.audio_dir / f"{rec.id:04d}.wav"
            sf.write(path, seg.audio, samplerate)
            rec.audio_file = str(path)
        except Exception as exc:  # noqa: BLE001 - playback is a nicety, not the point
            print(f"[session] could not write audio for utterance {rec.id}: {exc}")
        return rec

    def _key_for(self, track_id: int) -> str:
        """The identity this track belongs to, or a placeholder until known."""
        return self.identity_of.get(track_id, f"#{track_id}")

    # ---------- reading ----------

    def bounds(self) -> dict:
        """Extent of the recorded positions, padded a little."""
        with self._lock:
            if not self.positions:
                return {"xmin": -2.0, "xmax": 2.0, "ymin": 0.0, "ymax": 4.0}
            xs = [p[2] for p in self.positions]
            ys = [p[3] for p in self.positions]
        # Include the camera, or its marker sits outside the drawn area.
        if self.camera:
            xs = xs + [self.camera["ground"][0]]
            ys = ys + [self.camera["ground"][1]]
        pad = 0.5 if self.calibrated else 40.0
        return {
            "xmin": min(xs) - pad, "xmax": max(xs) + pad,
            "ymin": min(ys) - pad, "ymax": max(ys) + pad,
        }

    def snapshot(self) -> dict:
        with self._lock:
            positions = [[round(t, 2), self._key_for(tid), round(x, 3), round(y, 3)]
                         for t, tid, x, y in self.positions]
            # Drop placeholder keys that later resolved to a real identity.
            labels = {k: v for k, v in self.labels.items()
                      if not (k.startswith("#") and k[1:].isdigit()
                              and int(k[1:]) in self.identity_of)}
            speech = [s.as_dict() for s in self.speech]
            gestures = [g.as_dict() for g in self.gestures]
            poses = [dict(p, pid=self._key_for(p["id"])) for p in self.poses]
            # People sidebar: stable order by identity number (Person 1, 2, …).
            def _pid_key(pid: str) -> tuple:
                num = "".join(ch for ch in pid if ch.isdigit())
                return (0, int(num)) if num else (1, pid)

            people = [
                {
                    "pid": rec["pid"],
                    "label": rec["label"],
                    "first_seen": round(rec["first_seen"], 2),
                    "last_seen": round(rec["last_seen"], 2),
                    "sightings": rec["sightings"],
                    "photo": rec.get("photo"),
                    "present": bool(rec.get("present")),
                    "looking": bool(rec.get("looking")),
                    "looking_score": float(rec.get("looking_score") or 0.0),
                }
                for pid, rec in sorted(self.people.items(), key=lambda kv: _pid_key(kv[0]))
            ]
        return {
            "started_at": self.started_at,
            # A finished session must report when it ended, not the present
            # moment - otherwise reviewing one an hour later stretches the
            # timeline to now and squashes the recording into a sliver.
            "ended_at": self.ended_at,
            "now": time.time(),
            "calibrated": self.calibrated,
            "camera": self.camera,
            "fov": self.fov,
            "frame_size": self.frame_size,
            "background": self.background,
            "background_coverage": round(self.background_coverage, 3),
            "poses": poses,
            "units": "metres" if self.calibrated else "pixels",
            "bounds": self.bounds(),
            "labels": labels,
            "positions": positions,
            "speech": speech,
            "gestures": gestures,
            "people": people,
            "hud": dict(self.hud),
        }

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ended_at = time.time()
        blob = self.snapshot()
        (self.dir / "session.json").write_text(json.dumps(blob))
        self._saved = True
        return self.dir
