"""Frame in, scene out.

The pipeline runs three things and stitches them together:

  1. person detection + tracking (YOLO11 + ByteTrack) -> where bodies are
  2. face detection + embedding                       -> who is in frame
  3. the gallery                                      -> is that somebody new

Why both a body tracker and a face gallery: the tracker gives smooth,
frame-to-frame continuity (and works on a back turned to the camera, or a torso
behind a table) but its ids break whenever somebody is occluded or leaves and
comes back. The face gallery is the opposite - it cannot follow motion, but it
is durable across minutes, rooms, and restarts. Bind them together and a track
keeps its name through a head turn, while a returning person reclaims the name
they had an hour ago.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .faces import Face, build_face_backend
from .floor import FloorMap
from .gestures import SweepTracker, detect_clasped_point, detect_hand_raise
from .identity import IdentityGallery


@dataclass
class PersonTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    score: float
    first_seen: float
    last_seen: float
    votes: dict[str, float] = field(default_factory=dict)
    pid: str | None = None  # committed identity
    face: Face | None = None
    face_age: float = 0.0  # seconds since we last had a face for this track
    unmatched_streak: int = 0
    blocked: dict[str, float] = field(default_factory=dict)  # pid -> barred until
    frames: int = 0
    announced: bool = False  # has this track survived long enough to be reported
    room: tuple[float, float] | None = None  # floor position in metres, if calibrated
    trail: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)

    # Head toward camera (landmark yaw). Not true eye gaze.
    looking: bool = False
    looking_score: float = 0.0  # smoothed frontal 0..1
    looking_since: float | None = None
    looking_last_good: float | None = None  # last time score cleared enter threshold
    _looking_seen_at: float | None = None
    _looking_lost_at: float | None = None
    _looking_ema: float = 0.0

    # Pose + gesture state
    kps: np.ndarray | None = None  # (17, 2) COCO keypoints, image pixels
    kp_conf: np.ndarray | None = None  # (17,)
    hand_raised: bool = False  # debounced: this person is voting
    raise_sides: tuple[str, ...] = ()
    raised_since: float | None = None  # when the confirmed raise began
    _raise_seen_at: float | None = None  # first frame of the current candidate
    _lower_seen_at: float | None = None

    # Vote sweep (storyteller pointing round the circle)
    sweeping: bool = False
    sweep_bearing: float | None = None  # degrees; 0 = at camera, +90 = frame right
    sweep_started_at: float | None = None
    _sweep: SweepTracker = field(default_factory=SweepTracker)
    _sweep_seen_at: float | None = None
    _sweep_lost_at: float | None = None
    _sweep_logged: float | None = None

    @property
    def raised_for(self) -> float:
        """Seconds the hand has been confirmed up. 0 when it is not."""
        return 0.0 if self.raised_since is None else time.time() - self.raised_since

    @property
    def foot(self) -> tuple[float, float]:
        """Ground contact point, in pixels. Bottom-centre of the body box.

        For an above-the-waist detection this is the cut-off edge rather than
        real feet, so treat it as "direction from the camera" until the floor
        homography is calibrated.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

    @property
    def height_px(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Scene:
    frame_idx: int
    timestamp: float
    people: list[PersonTrack]
    faces: list[Face]
    fps: float = 0.0


def _containment(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of `inner`'s area that falls inside `outer`."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1)
    return inter / area


class Pipeline:
    def __init__(
        self,
        cfg: Config | None = None,
        face_backend: str = "auto",
        gallery_path: Path | None = None,
    ):
        from ultralytics import YOLO  # imported late; it pulls in torch

        self.cfg = cfg or Config()
        self.detector = YOLO(self.cfg.person.model)
        self.faces = build_face_backend(self.cfg.face, prefer=face_backend)
        if self.cfg.face.match_threshold is None:
            self.cfg.face.match_threshold = self.faces.default_match_threshold
        if self.cfg.face.enroll_threshold is None:
            self.cfg.face.enroll_threshold = self.faces.default_enroll_threshold
        # Default is the durable global gallery. Pass a session directory to
        # keep identities (Person 1, Person 2, …) scoped to one recording.
        self.gallery = IdentityGallery(
            path=gallery_path,
            max_per_identity=self.cfg.face.embeddings_per_identity,
        )
        self.gallery.assert_compatible(self.faces.embedding_dim)
        self.floor = FloorMap()
        if self.floor.valid:
            print("[floor] calibrated - positions reported in metres")
        self.tracks: dict[int, PersonTrack] = {}
        # Who was visible, and when. Speech arrives after the fact - a segment
        # is only complete once the speaker stops - so attributing it means
        # looking back at who was in the room while the words were said.
        self.presence: deque[tuple[float, frozenset[str]]] = deque(maxlen=2000)
        self.frame_idx = 0
        self.frame_size: tuple[int, int] | None = None  # (w, h), set on first frame
        self._last_t = time.time()
        self._fps = 0.0
        self.events: list[dict] = []  # drained by the caller each frame
        self.votes: list[dict] = []  # completed raises, for the closing tally

    # ---------- main entry ----------

    def process(self, frame_bgr: np.ndarray) -> Scene:
        now = time.time()
        self.frame_idx += 1
        if self.frame_size is None:
            self.frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

        self._update_people(frame_bgr, now)

        faces: list[Face] = []
        if self.frame_idx % self.cfg.face.every_n_frames == 0:
            faces = self.faces.detect(frame_bgr)
            self._bind_faces(faces, now)
        else:
            # Face pass skipped this frame; still advance looking debounce so a
            # turn-away releases on wall-clock time rather than face-frame count.
            for trk in self.tracks.values():
                self._update_looking(trk, now)

        self._retire(now)
        self.presence.append(
            (now, frozenset(t.pid for t in self.tracks.values() if t.pid and t.announced))
        )
        return Scene(
            frame_idx=self.frame_idx,
            timestamp=now,
            people=list(self.tracks.values()),
            faces=faces,
            fps=self._fps,
        )

    # ---------- stage 1: bodies ----------

    def _update_people(self, frame_bgr: np.ndarray, now: float) -> None:
        kwargs = {}
        if self.cfg.person.device:
            kwargs["device"] = self.cfg.person.device
        res = self.detector.track(
            frame_bgr,
            persist=True,
            classes=[0],  # COCO person
            conf=self.cfg.person.conf,
            iou=self.cfg.person.iou,
            imgsz=self.cfg.person.imgsz,
            tracker=self.cfg.person.tracker,
            verbose=False,
            **kwargs,
        )[0]

        if res.boxes is None or res.boxes.id is None:
            return

        ids = res.boxes.id.int().cpu().tolist()
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        # Present only when running a -pose model; everything downstream treats
        # missing keypoints as "gestures unavailable" rather than an error.
        kp_xy = kp_conf = None
        if getattr(res, "keypoints", None) is not None and res.keypoints.conf is not None:
            kp_xy = res.keypoints.xy.cpu().numpy()
            kp_conf = res.keypoints.conf.cpu().numpy()

        for i, (tid, box, conf) in enumerate(zip(ids, xyxy, confs)):
            bbox = tuple(int(v) for v in box)  # type: ignore[assignment]
            trk = self.tracks.get(tid)
            if trk is None:
                trk = PersonTrack(
                    track_id=tid, bbox=bbox, score=float(conf), first_seen=now, last_seen=now
                )
                self.tracks[tid] = trk
            trk.bbox = bbox
            trk.score = float(conf)
            trk.last_seen = now
            trk.frames += 1
            # Detectors throw off single-frame ghosts. Do not report a person
            # until the track has actually held together for a moment, or the
            # event log fills with arrivals that never happened.
            if not trk.announced and trk.frames >= self.cfg.track.frames_to_announce:
                trk.announced = True
                self._emit("person_appeared", track_id=tid, bbox=bbox, room=trk.room)
            if kp_xy is not None and i < len(kp_xy):
                trk.kps, trk.kp_conf = kp_xy[i], kp_conf[i]
                self._update_gesture(trk, now)
                self._update_sweep(trk, now)
            trk.face_age = now - getattr(trk, "_face_t", trk.first_seen)
            fx, fy = trk.foot
            trk.room = self.floor.to_room(fx, fy)
            if not trk.trail or now - trk.trail[-1][0] > 0.2:
                trk.trail.append((now, fx, fy))
                del trk.trail[:-150]

    def _update_gesture(self, trk: PersonTrack, now: float) -> None:
        """Debounce the per-frame hand-raise verdict into a stable state.

        Raw per-frame detection flickers - a wrist crossing the threshold, an
        arm briefly occluded - and a box that strobes between voting and not is
        useless to look at and worse to act on. A hand must hold up for
        `hold_s` to count, and stay down for the longer `release_s` to stop
        counting.
        """
        gcfg = self.cfg.gesture
        if not gcfg.enabled or trk.kps is None:
            return

        verdict = detect_hand_raise(
            trk.kps, trk.kp_conf,
            min_conf=gcfg.min_kp_conf,
            face_level_frac=gcfg.face_level_frac,
        )

        if verdict.raised:
            trk._lower_seen_at = None  # noqa: SLF001
            if trk._raise_seen_at is None:  # noqa: SLF001
                trk._raise_seen_at = now  # noqa: SLF001
            trk.raise_sides = verdict.sides
            if not trk.hand_raised and now - trk._raise_seen_at >= gcfg.hold_s:  # noqa: SLF001
                trk.hand_raised = True
                trk.raised_since = trk._raise_seen_at  # noqa: SLF001
                self._emit(
                    "hand_raised",
                    track_id=trk.track_id,
                    pid=trk.pid,
                    label=self.gallery.label_of(trk.pid) if trk.pid else f"#{trk.track_id}",
                    sides=list(verdict.sides),
                    room=trk.room,
                )
        else:
            trk._raise_seen_at = None  # noqa: SLF001
            if trk.hand_raised:
                if trk._lower_seen_at is None:  # noqa: SLF001
                    trk._lower_seen_at = now  # noqa: SLF001
                if now - trk._lower_seen_at >= gcfg.release_s:  # noqa: SLF001
                    self._close_vote(trk, trk._lower_seen_at, "lowered")  # noqa: SLF001

    def _update_sweep(self, trk: PersonTrack, now: float) -> None:
        """Track the storyteller's clasped-hands pointing sweep."""
        gcfg = self.cfg.gesture
        if not gcfg.sweep_enabled or trk.kps is None:
            return

        verdict = detect_clasped_point(
            trk.kps, trk.kp_conf,
            min_conf=gcfg.sweep_min_kp_conf,
            max_gap=gcfg.sweep_max_gap,
            max_drop=gcfg.sweep_max_drop,
            reach=gcfg.sweep_reach,
        )
        bearing = trk._sweep.update(verdict, now)  # noqa: SLF001

        if verdict.detected:
            trk._sweep_lost_at = None  # noqa: SLF001
            if trk._sweep_seen_at is None:  # noqa: SLF001
                trk._sweep_seen_at = now  # noqa: SLF001
            trk.sweep_bearing = bearing
            if not trk.sweeping and now - trk._sweep_seen_at >= gcfg.sweep_hold_s:  # noqa: SLF001
                trk.sweeping = True
                trk.sweep_started_at = trk._sweep_seen_at  # noqa: SLF001
                self._emit("sweep_started", track_id=trk.track_id, pid=trk.pid,
                           label=self.gallery.label_of(trk.pid) if trk.pid
                           else f"#{trk.track_id}",
                           bearing_deg=round(bearing, 1) if bearing is not None else None)
        else:
            trk._sweep_seen_at = None  # noqa: SLF001
            if trk.sweeping:
                if trk._sweep_lost_at is None:  # noqa: SLF001
                    trk._sweep_lost_at = now  # noqa: SLF001
                # Generous: pointing straight away hides the hands for a moment
                # every revolution, and that is not the sweep ending.
                if now - trk._sweep_lost_at >= gcfg.sweep_release_s:  # noqa: SLF001
                    self._end_sweep(trk, trk._sweep_lost_at, "pose released")  # noqa: SLF001

    def _end_sweep(self, trk: PersonTrack, ended_at: float, reason: str) -> None:
        if not trk.sweeping:
            return
        started = trk.sweep_started_at if trk.sweep_started_at is not None else ended_at
        self._emit(
            "sweep_ended",
            track_id=trk.track_id,
            pid=trk.pid,
            label=self.gallery.label_of(trk.pid) if trk.pid else f"#{trk.track_id}",
            held_s=round(max(0.0, ended_at - started), 2),
            revolutions=round(trk._sweep.revolutions, 2),  # noqa: SLF001
            reason=reason,
        )
        trk.sweeping = False
        trk.sweep_started_at = None
        trk.sweep_bearing = None
        trk._sweep.reset()  # noqa: SLF001
        trk._sweep_seen_at = trk._sweep_lost_at = trk._sweep_logged = None  # noqa: SLF001

    def sweeping_now(self) -> list[PersonTrack]:
        """Anyone currently running a vote sweep."""
        return [t for t in self.tracks.values() if t.sweeping and t.announced]

    def _close_vote(self, trk: PersonTrack, ended_at: float, reason: str) -> None:
        """Finish a raised hand and write the completed vote to the log.

        `ended_at` is when the hand actually came down, not when the debounce
        confirmed it - otherwise every vote is reported `release_s` longer than
        it was, which quietly inflates any duration you later count.

        The log gets one mark when a hand goes up and one when it comes down -
        `hand_raised` and `hand_lowered` - and nothing else. `reason` says
        whether the hand was actually lowered or the person simply left.
        """
        if not trk.hand_raised:
            return
        # `or` would be wrong here: a raised_since of 0.0 is a valid timestamp
        # but falsy, which silently zeroes the duration.
        started = trk.raised_since if trk.raised_since is not None else ended_at
        held = max(0.0, ended_at - started)
        label = self.gallery.label_of(trk.pid) if trk.pid else f"#{trk.track_id}"
        sides = list(trk.raise_sides)

        trk.hand_raised = False
        trk.raised_since = None
        trk.raise_sides = ()
        trk._lower_seen_at = None  # noqa: SLF001
        trk._raise_seen_at = None  # noqa: SLF001

        self._emit(
            "hand_lowered",
            track_id=trk.track_id,
            pid=trk.pid,
            label=label,
            identified=trk.pid is not None,
            raised_at=round(started, 3),
            lowered_at=round(ended_at, 3),
            held_s=round(held, 2),
            sides=sides,
            reason=reason,
        )
        self.votes.append({
            "pid": trk.pid, "label": label, "started_at": started,
            "duration_s": held, "reason": reason,
        })

    def present_during(self, start: float, end: float, min_fraction: float = 0.5) -> set[str]:
        """Face identities visible for at least `min_fraction` of a time window.

        Not "seen at any point": somebody who crossed the frame for one frame
        while a sentence was spoken is not a candidate for having spoken it.
        """
        samples = [pids for t, pids in self.presence if start <= t <= end]
        if not samples:
            # The window may predate the first frame, or fall between frames at
            # low fps. Fall back to the nearest sample rather than claiming the
            # room was empty.
            nearest = min(
                self.presence,
                key=lambda s: min(abs(s[0] - start), abs(s[0] - end)),
                default=None,
            )
            return set(nearest[1]) if nearest else set()

        counts: dict[str, int] = {}
        for pids in samples:
            for pid in pids:
                counts[pid] = counts.get(pid, 0) + 1
        need = max(1, int(len(samples) * min_fraction))
        return {pid for pid, n in counts.items() if n >= need}

    def voting(self) -> list[PersonTrack]:
        """Everyone currently holding a hand up."""
        return [t for t in self.tracks.values() if t.hand_raised and t.announced]

    # ---------- stage 2: faces -> tracks -> identities ----------

    def _bind_faces(self, faces: list[Face], now: float) -> None:
        for trk in self.tracks.values():
            trk.face = None

        # Bind each face to the body it belongs to. Containment alone is not
        # enough once people overlap: in a crowd a face sits inside three or four
        # boxes at once, and picking the tightest can hand somebody's face to the
        # person standing in front of them - which then hands them their name.
        #
        # So the candidate also has to be anatomically possible: the face must be
        # near the top of the body box, and it must be a plausible fraction of
        # that body's width. Those two checks cost nothing and throw out most of
        # the wrong answers.
        tcfg = self.cfg.track
        frame_w = self.frame_size[0] if self.frame_size else 1280
        pairs: list[tuple[float, Face, PersonTrack]] = []
        for face in faces:
            close_up = face.width / max(frame_w, 1) >= 0.22
            for trk in self.tracks.values():
                if now - trk.last_seen > 0.5:
                    continue
                c = _containment(face.bbox, trk.bbox)
                # Close-up: face often barely fits the body box (or the body is
                # only a torso crop). Accept weaker containment.
                if c < (0.25 if close_up else 0.6):
                    # Also try: body centre near face centre (same person).
                    if not close_up:
                        continue
                    tx = (trk.bbox[0] + trk.bbox[2]) / 2.0
                    ty = (trk.bbox[1] + trk.bbox[3]) / 2.0
                    if abs(tx - face.center[0]) > face.width * 1.2:
                        continue
                    if abs(ty - face.center[1]) > face.height * 2.0:
                        continue
                    c = max(c, 0.3)
                x1, y1, x2, y2 = trk.bbox
                bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

                # Where down the body does the face sit? Heads are at the top.
                depth = (face.center[1] - y1) / bh
                if not close_up and depth > tcfg.max_face_depth:
                    continue
                # A head is roughly a fifth to a half of a visible body's width.
                # Way outside that and this box belongs to somebody else.
                # Close-up: face can be almost as wide as the body box.
                ratio = face.width / bw
                lo = tcfg.min_face_body_ratio
                hi = 1.15 if close_up else tcfg.max_face_body_ratio
                if not (lo <= ratio <= hi) and not close_up:
                    continue

                score = c * (1.0 - min(depth, 1.0)) * min(bw / max(face.width, 1), 8.0) ** -0.25
                if close_up:
                    score += 0.5  # prefer binding large faces rather than dropping them
                pairs.append((score, face, trk))

        # Greedy best-first, one face per body and one body per face, so two
        # tracks cannot end up voting on the same face.
        taken_faces: set[int] = set()
        for score, face, trk in sorted(pairs, key=lambda p: p[0], reverse=True):
            if id(face) in taken_faces or trk.face is not None:
                continue
            taken_faces.add(id(face))
            trk.face = face
            trk._face_t = now  # noqa: SLF001 - internal bookkeeping

        # Orphan close-up faces: if only one announced track, hand it the face
        # so looking still works when the body detector is a mess at short range.
        unbound = [f for f in faces if id(f) not in taken_faces]
        announced = [t for t in self.tracks.values() if t.announced and t.face is None]
        if len(unbound) == 1 and len(announced) == 1:
            face = unbound[0]
            if face.width / max(frame_w, 1) >= 0.18:
                announced[0].face = face
                announced[0]._face_t = now  # noqa: SLF001

        for trk in self.tracks.values():
            if trk.face is not None:
                self._vote(trk, trk.face, now)
            self._update_looking(trk, now)

        self._resolve_conflicts(now)

    def _update_looking(self, trk: PersonTrack, now: float) -> None:
        """Debounced + hysteresis head-toward-camera from face landmarks.

        Face detection only runs every N frames, so a missing face this pass is
        not "looked away" — we keep the EMA score and only release after
        looking_release_s of consistently non-frontal evidence.

        Close-up faces (filling much of the frame) are treated more leniently:
        if a face is detected at all and reasonably large, that is strong
        evidence the person is addressing the camera.
        """
        fcfg = self.cfg.face
        enter = float(getattr(fcfg, "looking_enter", fcfg.looking_threshold))
        exit_thr = float(getattr(fcfg, "looking_exit", max(0.0, enter - 0.15)))

        raw = None
        if trk.face is not None:
            face = trk.face
            if face.kps is not None:
                raw = float(face.frontal())
            else:
                # No landmarks: a confident face is a weak "toward" prior.
                raw = 0.6 if face.score >= fcfg.det_conf else 0.0
            # Close-up boost: a large face almost always means addressing the cam.
            if self.frame_size is not None:
                fw = max(face.width, 1)
                frame_w = max(self.frame_size[0], 1)
                if fw / frame_w >= 0.22:
                    raw = max(raw, 0.72)
                if fw / frame_w >= 0.35:
                    raw = max(raw, 0.85)

        if raw is not None:
            alpha = 0.35
            trk._looking_ema = (1.0 - alpha) * float(trk._looking_ema) + alpha * raw
            score = trk._looking_ema
        else:
            # No face this frame — hold the previous score (do not slam to 0).
            score = float(trk._looking_ema) if trk._looking_ema > 0 else float(trk.looking_score)

        trk.looking_score = float(score)

        # Hysteresis: higher bar to enter looking, lower bar to leave.
        sticky = float(getattr(fcfg, "looking_sticky_s", 2.0))
        if trk.looking:
            if raw is not None:
                toward = score >= exit_thr
            elif trk.looking_last_good is not None:
                toward = (now - trk.looking_last_good) <= sticky
            else:
                toward = False
        else:
            toward = raw is not None and score >= enter

        if toward and raw is not None and score >= enter:
            trk.looking_last_good = now

        if toward:
            if trk._looking_seen_at is None:
                trk._looking_seen_at = now
            trk._looking_lost_at = None
            if not trk.looking and now - trk._looking_seen_at >= fcfg.looking_hold_s:
                trk.looking = True
                trk.looking_since = trk._looking_seen_at
                if trk.announced:
                    self._emit(
                        "looking_at_camera",
                        track_id=trk.track_id,
                        pid=trk.pid,
                        label=self.gallery.label_of(trk.pid) if trk.pid else f"#{trk.track_id}",
                        score=round(score, 2),
                    )
            return

        # Not frontal enough.
        trk._looking_seen_at = None
        if not trk.looking:
            trk.looking_since = None
            return
        if trk._looking_lost_at is None:
            trk._looking_lost_at = now
        if now - trk._looking_lost_at >= fcfg.looking_release_s:
            held = (trk._looking_lost_at - (trk.looking_since or trk._looking_lost_at))
            trk.looking = False
            trk.looking_since = None
            trk._looking_lost_at = None
            if trk.announced:
                self._emit(
                    "looked_away",
                    track_id=trk.track_id,
                    pid=trk.pid,
                    label=self.gallery.label_of(trk.pid) if trk.pid else f"#{trk.track_id}",
                    duration_s=round(max(held, 0.0), 2),
                )

    def looking_now(self, sticky: bool = True) -> list[PersonTrack]:
        """Everyone facing the camera (optionally including sticky recent looks)."""
        now = time.time()
        sticky_s = float(getattr(self.cfg.face, "looking_sticky_s", 2.0))
        out = []
        for t in self.tracks.values():
            if not t.announced:
                continue
            if t.looking:
                out.append(t)
            elif (
                sticky
                and t.looking_last_good is not None
                and now - t.looking_last_good <= sticky_s
            ):
                out.append(t)
        return out

    def _vote(self, trk: PersonTrack, face: Face, now: float) -> None:
        fcfg = self.cfg.face
        tcfg = self.cfg.track
        quality = face.quality()
        pid, sim = self.gallery.match(face.embedding)
        raw_sim = sim  # before blocking, so a barred track still counts as "known"

        # A track that just lost this identity to a better-supported track must
        # not immediately re-claim it, or the two ping-pong every frame and the
        # event log fills with identical assignments.
        if pid is not None and trk.blocked.get(pid, 0.0) > now:
            pid, sim = None, -1.0

        for k in list(trk.votes):
            trk.votes[k] *= tcfg.vote_decay

        if pid is not None and sim >= fcfg.match_threshold:
            trk.votes[pid] = trk.votes.get(pid, 0.0) + max(quality, 0.25)
            trk.unmatched_streak = 0
        elif raw_sim < fcfg.enroll_threshold and quality >= 0.55:
            # Nothing in the gallery is close and this is a good look at a face.
            # Wait for a couple of frames of agreement before minting a person,
            # so a motion-blurred stranger does not become a permanent ghost.
            trk.unmatched_streak += 1
            if trk.unmatched_streak >= 3 and trk.pid is None:
                idt = self.gallery.enroll(face.embedding)
                trk.votes[idt.pid] = float(tcfg.votes_to_commit)
                trk.unmatched_streak = 0
                self.gallery.save()
                self._emit(
                    "identity_created", track_id=trk.track_id, pid=idt.pid, label=idt.label
                )
        else:
            trk.unmatched_streak = 0

        best_pid = max(trk.votes, key=trk.votes.get, default=None)
        # Claiming an *unclaimed* track on a confident single look is fine - there
        # is nothing to overwrite. Taking a name away from a track that already
        # has one has to clear the slower accumulated-evidence bar, so a bad
        # frame cannot relabel somebody mid-conversation.
        confident_first_look = (
            trk.pid is None and sim >= tcfg.instant_claim_sim and quality >= 0.45
        )
        committed = best_pid is not None and (
            trk.votes[best_pid] >= tcfg.votes_to_commit
            or (confident_first_look and best_pid == pid)
        )
        if committed and best_pid != trk.pid:
            was = trk.pid
            trk.pid = best_pid
            self._emit(
                "identity_assigned",
                track_id=trk.track_id,
                pid=best_pid,
                label=self.gallery.label_of(best_pid),
                previous=was,
                similarity=round(sim, 3),
                room=trk.room,
            )
        if trk.pid:
            self.gallery.reinforce(trk.pid, face.embedding, quality)

    def _resolve_conflicts(self, now: float) -> None:
        """One person cannot be two tracks. Strongest vote keeps the identity.

        The loser is barred from re-claiming that identity for a few seconds.
        Without the bar it simply wins the name back on the next frame and the
        two tracks trade it indefinitely.
        """
        by_pid: dict[str, list[PersonTrack]] = {}
        for trk in self.tracks.values():
            if trk.pid:
                by_pid.setdefault(trk.pid, []).append(trk)
        for pid, trks in by_pid.items():
            if len(trks) < 2:
                continue
            trks.sort(key=lambda t: t.votes.get(pid, 0.0), reverse=True)
            for loser in trks[1:]:
                loser.votes.pop(pid, None)
                loser.pid = None
                loser.blocked[pid] = now + self.cfg.track.conflict_block_s
                self._emit("identity_contested", track_id=loser.track_id, pid=pid,
                           kept_by=trks[0].track_id)

    # ---------- housekeeping ----------

    def _retire(self, now: float) -> None:
        gone = [
            tid
            for tid, trk in self.tracks.items()
            if now - trk.last_seen > self.cfg.track.forget_after_s
        ]
        for tid in gone:
            trk = self.tracks.pop(tid)
            # A hand still up when its owner disappears is a real vote that
            # would otherwise never be written - it just vanishes from the log.
            if trk.hand_raised:
                self._close_vote(trk, trk.last_seen, "person left")
            if trk.sweeping:
                self._end_sweep(trk, trk.last_seen, "person left")
            if not trk.announced:
                continue
            self._emit(
                "person_left",
                track_id=tid,
                pid=trk.pid,
                label=self.gallery.label_of(trk.pid),
                duration_s=round(trk.last_seen - trk.first_seen, 2),
            )

    def _emit(self, kind: str, **payload) -> None:
        self.events.append({"t": time.time(), "event": kind, **payload})

    def drain_events(self) -> list[dict]:
        out, self.events = self.events, []
        return out

    def pairs_within(self, metres: float) -> list[tuple[PersonTrack, PersonTrack, float]]:
        """People standing close enough to plausibly be in conversation.

        Proximity alone is not conversation - two people can share a sofa in
        silence, and a table can seat people who are talking past each other.
        This is the geometric half of the answer; the audio half comes later.
        Requires a calibrated floor.
        """
        out = []
        people = [t for t in self.tracks.values() if t.room is not None and t.announced]
        for i, a in enumerate(people):
            for b in people[i + 1 :]:
                d = float(np.hypot(a.room[0] - b.room[0], a.room[1] - b.room[1]))
                if d <= metres:
                    out.append((a, b, d))
        return sorted(out, key=lambda p: p[2])

    def label_for(self, trk: PersonTrack) -> str:
        if trk.pid:
            return self.gallery.label_of(trk.pid)
        return f"#{trk.track_id}"

    def close(self) -> None:
        # Same again at shutdown: whoever still has a hand up gets their vote
        # recorded rather than dropped on the floor.
        for trk in self.tracks.values():
            if trk.hand_raised:
                self._close_vote(trk, trk.last_seen, "session ended")
            if trk.sweeping:
                self._end_sweep(trk, trk.last_seen, "session ended")
        self.gallery.save()
