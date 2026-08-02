"""Reading gestures off pose keypoints. Currently: is this person's hand up.

Kept as pure functions over arrays so the decision can be tested against
fabricated skeletons - waiting for footage of somebody actually voting is a bad
way to find out the threshold is wrong.

COCO-17 keypoint order, which is what YOLO pose emits. Note "left" is the
*person's* left, so it appears on the right of the image when they face you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class HandRaise:
    """A single frame's verdict. Temporal smoothing happens in the pipeline."""

    raised: bool
    sides: tuple[str, ...] = ()
    threshold_y: float | None = None  # image y at/above which a wrist counts
    reason: str = ""

    def __bool__(self) -> bool:
        return self.raised


def _pt(kps: np.ndarray, conf: np.ndarray, idx: int, min_conf: float):
    """Keypoint if the model is confident about it, else None.

    Low-confidence keypoints are not merely noisy - YOLO parks unseen joints at
    plausible-looking guessed positions, so using one without checking gives a
    wrist that is quietly invented.
    """
    if idx >= len(conf) or conf[idx] < min_conf:
        return None
    return kps[idx]


def head_scale(kps: np.ndarray, conf: np.ndarray, min_conf: float) -> float | None:
    """Vertical nose-to-shoulders distance, used as this person's body scale.

    Scales with distance from the camera, so thresholds expressed as a fraction
    of it mean the same thing across the room. Falls back to the eye-to-nose
    span, which is smaller and noisier but better than a pixel constant.
    """
    nose = _pt(kps, conf, NOSE, min_conf)
    if nose is None:
        return None
    shoulders = [
        p for p in (_pt(kps, conf, LEFT_SHOULDER, min_conf),
                    _pt(kps, conf, RIGHT_SHOULDER, min_conf)) if p is not None
    ]
    if shoulders:
        shoulder_y = float(np.mean([p[1] for p in shoulders]))
        span = shoulder_y - float(nose[1])
        if span > 1.0:
            return span
    eyes = [
        p for p in (_pt(kps, conf, LEFT_EYE, min_conf), _pt(kps, conf, RIGHT_EYE, min_conf))
        if p is not None
    ]
    if eyes:
        eye_y = float(np.mean([p[1] for p in eyes]))
        span = float(nose[1]) - eye_y
        if span > 1.0:
            return span * 3.0  # eyes-to-nose is roughly a third of nose-to-shoulder
    return None


def detect_hand_raise(
    kps: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.5,
    face_level_frac: float = 0.45,
) -> HandRaise:
    """Is a wrist level with the face or above it.

    The threshold is `face_level_frac` of the way from the nose down to the
    shoulders, which lands at about chin level - measured against a real
    detection, the chin sat at 0.46 of that span. Expressing it as a fraction
    rather than pixels is what makes it hold at any distance from the camera.

    Lower `face_level_frac` towards 0 to require the hand higher (0 = at the
    nose, which cuts hand-near-the-chin false positives).
    """
    kps = np.asarray(kps, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    nose = _pt(kps, conf, NOSE, min_conf)
    if nose is None:
        return HandRaise(False, reason="no confident head keypoint")

    scale = head_scale(kps, conf, min_conf)
    if scale is None:
        return HandRaise(False, reason="cannot estimate body scale")

    threshold_y = float(nose[1]) + face_level_frac * scale

    sides = []
    for idx, name in ((LEFT_WRIST, "left"), (RIGHT_WRIST, "right")):
        wrist = _pt(kps, conf, idx, min_conf)
        # An unseen wrist is not a lowered wrist. Staying silent is right: the
        # hand may be up and simply occluded, and guessing either way is worse
        # than reporting nothing.
        if wrist is None:
            continue
        if float(wrist[1]) <= threshold_y:
            sides.append(name)

    return HandRaise(
        raised=bool(sides),
        sides=tuple(sides),
        threshold_y=threshold_y,
        reason="wrist at or above face level" if sides else "wrists below face level",
    )


@dataclass
class ClaspedPoint:
    """The storyteller's vote sweep: hands clasped, arms out, turning on the spot.

    `bearing_deg` is where they are pointing, seen from the camera:
        0    straight at the camera
        +90  to the right of frame
        ±180 directly away
        -90  to the left of frame
    """

    detected: bool
    bearing_deg: float | None = None
    facing: str = "unknown"  # toward | away | unknown
    confidence: float = 0.0
    gap: float | None = None  # hand separation, in torso lengths
    drop: float | None = None  # how far below the shoulders the hands sit
    reason: str = ""

    def __bool__(self) -> bool:
        return self.detected


def torso_length(kps: np.ndarray, conf: np.ndarray, min_conf: float) -> float | None:
    """Shoulder midpoint to hip midpoint - the one body scale that survives yaw.

    Shoulder *width* is the obvious choice and the wrong one: it collapses from
    174px to 24px as somebody turns side-on, so any ratio built on it swings by
    an order of magnitude while the pose is unchanged.
    """
    pts = [_pt(kps, conf, i, min_conf)
           for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)]
    if any(p is None for p in pts):
        return None
    shoulders = (pts[0] + pts[1]) / 2.0
    hips = (pts[2] + pts[3]) / 2.0
    length = float(np.linalg.norm(shoulders - hips))
    return length if length > 5.0 else None


def _facing(kps: np.ndarray, conf: np.ndarray, min_conf: float) -> tuple[str, float]:
    """Are we looking at their front or their back, and how sure are we.

    Face keypoints first: if the nose and eyes are visible, they are facing us.
    Shoulder left/right ordering is the fallback - it flips when somebody turns
    around - but it is unreliable near profile, where the shoulders nearly
    coincide and their order is noise.
    """
    face = float(np.mean([conf[NOSE], conf[LEFT_EYE], conf[RIGHT_EYE]]))
    if face > 0.5:
        return "toward", min(1.0, face)
    if face < 0.15:
        return "away", min(1.0, 1.0 - face)

    lsh = _pt(kps, conf, LEFT_SHOULDER, min_conf)
    rsh = _pt(kps, conf, RIGHT_SHOULDER, min_conf)
    if lsh is None or rsh is None:
        return "unknown", 0.0
    # A person's own left shoulder appears on the right of frame when they face us.
    return ("toward" if lsh[0] > rsh[0] else "away"), 0.4


def detect_clasped_point(
    kps: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.15,
    max_gap: float = 0.45,
    max_drop: float = 0.75,
    reach: float = 0.90,
) -> ClaspedPoint:
    """Detect hands-clasped-arms-extended, and which way it points.

    Deliberately does NOT test how long the arms look. Projected arm length is
    ~0.9 torso lengths pointing across the camera and ~0.2 pointing straight at
    it - indistinguishable from arms at rest - because an arm aimed at the lens
    foreshortens to nothing. What survives every direction is that the hands
    stay *together* and stay *up*.

    `min_conf` is low on purpose: pointing directly away puts both wrists behind
    the body, where the detector's confidence falls to ~0.15 while the position
    estimate stays usable. Rejecting those frames would blind the system to a
    quarter of every turn.
    """
    kps = np.asarray(kps, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    torso = torso_length(kps, conf, 0.4)
    if torso is None:
        return ClaspedPoint(False, reason="no confident torso")

    lwr = _pt(kps, conf, LEFT_WRIST, min_conf)
    rwr = _pt(kps, conf, RIGHT_WRIST, min_conf)
    if lwr is None or rwr is None:
        return ClaspedPoint(False, reason="wrists not visible")

    lsh, rsh = kps[LEFT_SHOULDER], kps[RIGHT_SHOULDER]
    shoulders = (lsh + rsh) / 2.0
    hands = (lwr + rwr) / 2.0

    gap = float(np.linalg.norm(lwr - rwr)) / torso
    drop = float(hands[1] - shoulders[1]) / torso

    if gap > max_gap:
        return ClaspedPoint(False, gap=gap, drop=drop, reason="hands apart")
    if drop > max_drop:
        return ClaspedPoint(False, gap=gap, drop=drop, reason="arms hanging")

    facing, facing_conf = _facing(kps, conf, min_conf)
    if facing == "unknown":
        return ClaspedPoint(False, gap=gap, drop=drop, reason="cannot tell front from back")

    # Sideways component straight off the image; depth component from whether we
    # are looking at their face or their back. Together that is the full circle.
    # Normalised by torso: the offset is in pixels and `reach` is in torso
    # lengths, so dividing one by the other directly saturates the clip and
    # pins every bearing to +/-90.
    across = float(np.clip((hands[0] - shoulders[0]) / (reach * torso), -1.0, 1.0))
    depth = float(np.sqrt(max(0.0, 1.0 - across * across)))
    if facing == "away":
        depth = -depth
    bearing = float(np.degrees(np.arctan2(across, depth)))

    tightness = float(np.clip(1.0 - gap / max_gap, 0.0, 1.0))
    quality = float(np.clip(min(conf[LEFT_WRIST], conf[RIGHT_WRIST]) / 0.6, 0.0, 1.0))
    return ClaspedPoint(
        detected=True,
        bearing_deg=bearing,
        facing=facing,
        confidence=float(0.5 * tightness + 0.3 * quality + 0.2 * facing_conf),
        gap=gap,
        drop=drop,
        reason="hands clasped and raised",
    )


class SweepTracker:
    """Keeps a rotating bearing continuous through the profile crossings.

    A single frame cannot always tell front from back. Side-on, the depth
    component of the pointing direction is near zero and its sign is decided by
    whether a nose happened to be detected - so the bearing folds back on itself
    exactly when somebody turns through profile, which for a storyteller walking
    a circle is twice per revolution.

    Rotation, though, is continuous. Every reading has a mirror solution
    (180 - bearing) that is equally consistent with the image, so pick whichever
    of the two continues the turn already in progress. That resolves the
    ambiguity with information the single frame does not have.
    """

    def __init__(self, reset_after_s: float = 1.0, velocity_smoothing: float = 0.6):
        self.reset_after_s = reset_after_s
        self.velocity_smoothing = velocity_smoothing
        self.bearing: float | None = None  # unwrapped, may run beyond +/-180
        self.last_t: float | None = None
        self.velocity: float = 0.0  # degrees per second, signed
        self.revolutions: float = 0.0
        self._start: float | None = None

    @staticmethod
    def _delta(a: float, b: float) -> float:
        """Shortest signed angle from b to a, in degrees."""
        return (a - b + 180.0) % 360.0 - 180.0

    def update(self, verdict: ClaspedPoint | None, now: float) -> float | None:
        """Fold one frame's reading into a continuous bearing.

        Each reading has a mirror twin (180 - bearing) that fits the image just
        as well, so the choice is scored on two things at once:

          * continuity  - how far it is from where the turn already was
          * facing      - whether it agrees with what the face keypoints say

        Weighted by how much the facing evidence is worth. Square-on, a visible
        face is decisive and continuity should not override it. Side-on, the
        face tells you nothing and continuity carries the decision. Using
        continuity alone locks onto the wrong branch and never recovers; using
        facing alone folds the bearing back on itself at every profile crossing.
        """
        if verdict is None or not verdict.detected or verdict.bearing_deg is None:
            return self.wrapped
        gap = 0.0 if self.last_t is None else now - self.last_t
        if self.last_t is not None and gap > self.reset_after_s:
            self.bearing = None  # too long a gap to assume the turn continued
            self.velocity = 0.0
        self.last_t = now

        if self.bearing is None:
            self.bearing = verdict.bearing_deg
            self._start = verdict.bearing_deg
            self.velocity = 0.0
            return self.wrapped

        # Where the turn was heading, not merely where it was. A storyteller
        # walking the circle rotates steadily, so the previous bearing plus the
        # rotation already in progress is a better prediction than the previous
        # bearing alone - and it is what stops the reading folding back on
        # itself as they pass through profile.
        dt = max(0.0, gap)
        predicted = self.wrapped + self.velocity * min(dt, 0.5)

        face_weight = float(np.clip(verdict.confidence, 0.0, 1.0)) * 0.6
        best, best_cost = None, None
        for cand in (verdict.bearing_deg, 180.0 - verdict.bearing_deg):
            continuity = abs(self._delta(cand, predicted)) / 180.0
            implied = "toward" if abs(((cand + 180) % 360) - 180) < 90 else "away"
            disagreement = 0.0 if implied == verdict.facing else 1.0
            cost = (1.0 - face_weight) * continuity + face_weight * disagreement
            if best_cost is None or cost < best_cost:
                best, best_cost = cand, cost

        step = self._delta(best, self.wrapped)
        self.bearing += step
        if dt > 1e-3:
            a = self.velocity_smoothing
            self.velocity = a * self.velocity + (1 - a) * (step / dt)
        if self._start is not None:
            self.revolutions = (self.bearing - self._start) / 360.0
        return self.wrapped

    @property
    def wrapped(self) -> float | None:
        """Bearing in -180..180."""
        if self.bearing is None:
            return None
        return (self.bearing + 180.0) % 360.0 - 180.0

    def reset(self) -> None:
        self.bearing = self.last_t = self._start = None
        self.velocity = 0.0
        self.revolutions = 0.0


@dataclass
class Pointing:
    """Where a storyteller is aiming, whichever pose they use to do it."""

    detected: bool
    bearing_deg: float | None = None
    facing: str = "unknown"
    style: str = ""  # clasped | single-arm
    side: str = ""  # which arm, when it is a single-arm point
    confidence: float = 0.0
    reason: str = ""

    def __bool__(self) -> bool:
        return self.detected


def detect_pointing(
    kps: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.15,
    max_gap: float = 0.45,
    max_drop: float = 0.75,
    reach: float = 0.90,
    single_reach: float = 1.15,
    min_single_reach: float = 0.55,
) -> Pointing:
    """Storyteller pointing, by either convention.

    Two poses are in use and both mean the same thing:

      clasped     - both hands together, arms out, turning on the spot
      single-arm  - one arm extended, sweeping round the circle

    They need different tests. Clasped is recognised by the hands staying
    *together*; a single arm has nothing to compare against, so it is
    recognised by one hand being far from the body and clearly above the hip
    line, with the other hand nowhere near it.

    Bearing convention matches `detect_clasped_point`: 0 straight at the
    camera, +90 to the right of frame, ±180 directly away.
    """
    kps = np.asarray(kps, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    clasped = detect_clasped_point(kps, conf, min_conf, max_gap, max_drop, reach)
    if clasped.detected:
        return Pointing(True, clasped.bearing_deg, clasped.facing, "clasped",
                        confidence=clasped.confidence, reason=clasped.reason)

    torso = torso_length(kps, conf, 0.4)
    if torso is None:
        return Pointing(False, reason="no confident torso")

    shoulders = (kps[LEFT_SHOULDER] + kps[RIGHT_SHOULDER]) / 2.0
    best = None
    for wrist_i, name in ((LEFT_WRIST, "left"), (RIGHT_WRIST, "right")):
        wrist = _pt(kps, conf, wrist_i, min_conf)
        if wrist is None:
            continue
        out = float(abs(wrist[0] - shoulders[0])) / torso  # sideways extension
        drop = float(wrist[1] - shoulders[1]) / torso
        if out < min_single_reach or drop > max_drop:
            continue
        if best is None or out > best[1]:
            best = (wrist, out, name, float(conf[wrist_i]))
    if best is None:
        return Pointing(False, reason="no arm extended")

    wrist, out, side, wconf = best
    facing, facing_conf = _facing(kps, conf, min_conf)
    if facing == "unknown":
        return Pointing(False, reason="cannot tell front from back")

    # A single outstretched arm reaches further than clasped hands, which stay on
    # the midline: measured 1.25 torso lengths at full extension versus 0.90.
    # Reusing the clasped figure clips every wide point to exactly +/-90 and
    # throws away the ends of the sweep, which is where the circle wraps.
    across = float(np.clip((wrist[0] - shoulders[0]) / (single_reach * torso), -1.0, 1.0))
    depth = float(np.sqrt(max(0.0, 1.0 - across * across)))
    if facing == "away":
        depth = -depth
    bearing = float(np.degrees(np.arctan2(across, depth)))
    return Pointing(
        detected=True,
        bearing_deg=bearing,
        facing=facing,
        style="single-arm",
        side=side,
        confidence=float(0.5 * min(1.0, out) + 0.3 * min(1.0, wconf / 0.6)
                         + 0.2 * facing_conf),
        reason="one arm extended",
    )
