"""The storyteller's vote sweep: hands clasped, arms out, turning on the spot.

Synthetic skeletons for the edge cases, plus a run over the real clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from botcirl.gestures import (LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_HIP,
                              RIGHT_SHOULDER, RIGHT_WRIST, ClaspedPoint, SweepTracker,
                              detect_clasped_point, torso_length)

CLIP = Path("/private/tmp/claude-501/-Users-sam-Desktop-botcirl/"
            "e8fb913f-0632-4fc5-b2c2-1f6cf69cc329/scratchpad/vote.mov")


def pose(hands_dx=0.0, hands_dy=0.25, gap=0.10, torso=400.0, facing="toward",
         conf=0.9, wrist_conf=0.9, x0=800.0, y0=300.0):
    """Build a skeleton with the hands where you ask, in torso units."""
    k = np.zeros((17, 2), np.float32)
    c = np.full(17, conf, np.float32)
    sx = 1.0 if facing == "toward" else -1.0  # left shoulder sits right of frame when facing us

    k[LEFT_SHOULDER] = (x0 + 90 * sx, y0)
    k[RIGHT_SHOULDER] = (x0 - 90 * sx, y0)
    k[LEFT_HIP] = (x0 + 60 * sx, y0 + torso)
    k[RIGHT_HIP] = (x0 - 60 * sx, y0 + torso)
    k[NOSE] = (x0, y0 - 120)
    k[1] = (x0 - 20, y0 - 140)
    k[2] = (x0 + 20, y0 - 140)
    if facing == "away":
        c[NOSE] = c[1] = c[2] = 0.05

    hx, hy = x0 + hands_dx * torso, y0 + hands_dy * torso
    k[LEFT_WRIST] = (hx + gap * torso / 2, hy)
    k[RIGHT_WRIST] = (hx - gap * torso / 2, hy)
    c[LEFT_WRIST] = c[RIGHT_WRIST] = wrist_conf
    return k, c


# ---------- detection ----------


def test_clasped_and_raised_is_detected():
    assert detect_clasped_point(*pose()).detected


def test_hands_apart_is_not_the_pose():
    v = detect_clasped_point(*pose(gap=0.9))
    assert not v.detected and v.reason == "hands apart"


def test_arms_hanging_is_not_the_pose():
    """Arms down by the sides, hands near each other, must still be rejected."""
    v = detect_clasped_point(*pose(hands_dy=0.95, gap=0.3))
    assert not v.detected and v.reason == "arms hanging"


def test_pointing_at_the_camera_is_detected():
    """The case that breaks any test based on arm length.

    An arm aimed at the lens foreshortens to almost nothing, so its projected
    length is indistinguishable from arms at rest. Measured on the real clip:
    0.29 torso lengths pointing at the camera vs 0.9 pointing across it.
    """
    v = detect_clasped_point(*pose(hands_dx=0.0, hands_dy=0.25))
    assert v.detected
    assert abs(v.bearing_deg) < 15, f"should read as straight ahead, got {v.bearing_deg}"


def test_pointing_across_frame_reads_ninety_degrees():
    right = detect_clasped_point(*pose(hands_dx=0.90))
    left = detect_clasped_point(*pose(hands_dx=-0.90))
    assert right.bearing_deg == pytest.approx(90, abs=12)
    assert left.bearing_deg == pytest.approx(-90, abs=12)


def test_facing_away_flips_the_bearing_behind():
    v = detect_clasped_point(*pose(hands_dx=0.0, facing="away"))
    assert v.detected and v.facing == "away"
    assert abs(v.bearing_deg) > 165, f"pointing away should be near 180, got {v.bearing_deg}"


def test_occluded_wrists_are_still_usable():
    """Pointing straight away hides both wrists behind the body.

    Confidence there falls to ~0.15 on the real clip while the positions stay
    usable. Rejecting those frames blinds the system for a quarter of every turn.
    """
    v = detect_clasped_point(*pose(facing="away", wrist_conf=0.18))
    assert v.detected


def test_missing_wrists_report_honestly():
    k, c = pose()
    c[LEFT_WRIST] = c[RIGHT_WRIST] = 0.0
    v = detect_clasped_point(k, c)
    assert not v.detected and v.reason == "wrists not visible"


def test_torso_is_the_scale_not_shoulder_width():
    """Shoulder width collapses 174px -> 24px through a turn; torso does not."""
    k, c = pose()
    wide = torso_length(k, c, 0.4)
    # Squash the shoulders together as if turned side-on.
    k[LEFT_SHOULDER][0] = k[RIGHT_SHOULDER][0] = 800.0
    narrow = torso_length(k, c, 0.4)
    assert narrow == pytest.approx(wide, rel=0.05), "torso must survive yaw"


def test_scale_invariance():
    """Same gesture near and far must read the same."""
    near = detect_clasped_point(*pose(hands_dx=0.5, torso=600))
    far = detect_clasped_point(*pose(hands_dx=0.5, torso=120))
    assert near.detected and far.detected
    assert near.bearing_deg == pytest.approx(far.bearing_deg, abs=2)


# ---------- continuity ----------


def test_tracker_prefers_the_face_when_it_can_see_one():
    t = SweepTracker()
    t.update(ClaspedPoint(True, bearing_deg=0.0, facing="toward", confidence=0.9), 0.0)
    v = ClaspedPoint(True, bearing_deg=30.0, facing="toward", confidence=0.9)
    assert t.update(v, 0.1) == pytest.approx(30, abs=1)


def test_tracker_carries_through_a_profile_crossing():
    """Side-on, the face says nothing and continuity has to decide.

    Without this the bearing folds back on itself twice per revolution.
    """
    t = SweepTracker()
    for i, b in enumerate([0, 30, 60, 85]):
        t.update(ClaspedPoint(True, bearing_deg=b, facing="toward", confidence=0.8), i * 0.1)
    # Ambiguous reading with no facing evidence; the turn was heading positive.
    v = ClaspedPoint(True, bearing_deg=80.0, facing="away", confidence=0.05)
    out = t.update(v, 0.5)
    assert out > 90, f"should continue the turn past profile, got {out}"


def test_tracker_resets_after_a_long_gap():
    t = SweepTracker(reset_after_s=1.0)
    t.update(ClaspedPoint(True, bearing_deg=170.0, facing="away", confidence=0.9), 0.0)
    out = t.update(ClaspedPoint(True, bearing_deg=10.0, facing="toward", confidence=0.9), 5.0)
    assert out == pytest.approx(10, abs=1), "a long gap must not be bridged"


def test_tracker_ignores_frames_with_no_detection():
    t = SweepTracker()
    t.update(ClaspedPoint(True, bearing_deg=45.0, facing="toward", confidence=0.9), 0.0)
    assert t.update(None, 0.1) == pytest.approx(45, abs=1)
    assert t.update(ClaspedPoint(False), 0.2) == pytest.approx(45, abs=1)


# ---------- the real clip ----------


@pytest.mark.skipif(not CLIP.exists(), reason="reference clip not available")
def test_against_the_real_sweep():
    import cv2
    from ultralytics import YOLO

    m = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(str(CLIP))
    tracker = SweepTracker()
    detected, total, bearings, arms_down_hits = 0, 0, [], 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = m.predict(frame, classes=[0], verbose=False)[0]
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            v = detect_clasped_point(r.keypoints.xy[0].cpu().numpy(),
                                     r.keypoints.conf[0].cpu().numpy())
            total += 1
            if v.detected:
                detected += 1
                bearings.append(tracker.update(v, i / 30.0))
                if 222 <= i <= 250:  # the section where the arms are down
                    arms_down_hits += 1
        i += 1
    cap.release()

    assert detected / total > 0.5, f"only detected in {detected}/{total} frames"
    assert arms_down_hits == 0, f"{arms_down_hits} false positives while arms were down"
    # A full turn: the sweep must reach both sides and behind.
    assert min(bearings) < -80, f"never swept left, min {min(bearings)}"
    assert max(bearings) > 80, f"never swept right, max {max(bearings)}"
    assert any(abs(b) > 150 for b in bearings), "never pointed away from the camera"
    assert any(abs(b) < 20 for b in bearings), "never pointed at the camera"


# ---------- pipeline integration ----------


def test_sweep_is_debounced_and_survives_the_hands_vanishing(tmp_path, monkeypatch):
    """Pointing straight away hides the hands once per revolution.

    That gap must not be read as the sweep ending, or every turn produces a
    burst of spurious start/stop events.
    """
    import botcirl.config as config_mod

    monkeypatch.setattr(config_mod, "GALLERY_DIR", tmp_path / "g")
    monkeypatch.setattr(config_mod, "FLOOR_HOMOGRAPHY", tmp_path / "f.npy")

    from botcirl import Config, Pipeline
    from botcirl.pipeline import PersonTrack

    p = Pipeline(Config())
    trk = PersonTrack(track_id=1, bbox=(0, 0, 300, 800), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)

    def tick(t, **kw):
        trk.kps, trk.kp_conf = pose(**kw)
        trk.last_seen = t
        p._update_sweep(trk, t)

    tick(0.0, hands_dx=0.0)
    assert not trk.sweeping, "must not fire on a single frame"
    tick(0.6, hands_dx=0.2)
    assert trk.sweeping
    assert sum(1 for e in p.events if e["event"] == "sweep_started") == 1

    # Hands hidden behind the body for half a second mid-turn.
    tick(0.9, gap=1.5)
    tick(1.2, gap=1.5)
    assert trk.sweeping, "a brief occlusion is not the end of the sweep"

    # Genuinely released.
    tick(3.0, gap=1.5)
    assert not trk.sweeping
    ended = [e for e in p.events if e["event"] == "sweep_ended"]
    assert len(ended) == 1 and ended[0]["reason"] == "pose released"


def test_sweep_closes_when_the_storyteller_leaves(tmp_path, monkeypatch):
    import botcirl.config as config_mod

    monkeypatch.setattr(config_mod, "GALLERY_DIR", tmp_path / "g")
    monkeypatch.setattr(config_mod, "FLOOR_HOMOGRAPHY", tmp_path / "f.npy")

    from botcirl import Config, Pipeline
    from botcirl.pipeline import PersonTrack

    p = Pipeline(Config())
    trk = PersonTrack(track_id=1, bbox=(0, 0, 300, 800), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)
    p.tracks[1] = trk
    for t in (0.0, 0.6):
        trk.kps, trk.kp_conf = pose()
        p._update_sweep(trk, t)
    assert trk.sweeping

    trk.last_seen = 5.0
    p._retire(now=5.0 + p.cfg.track.forget_after_s + 1)
    ended = [e for e in p.events if e["event"] == "sweep_ended"]
    assert len(ended) == 1 and ended[0]["reason"] == "person left"
