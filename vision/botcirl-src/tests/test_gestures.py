"""Hand-raise detection, driven by fabricated skeletons.

Building poses by hand means the edge cases (occluded wrist, person far away,
head tilted) are testable without filming somebody voting for each one.
"""

from __future__ import annotations

import numpy as np
import pytest

from botcirl.gestures import LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_SHOULDER, detect_hand_raise, head_scale
from helpers import skeleton


def test_hands_down_is_not_voting():
    kps, conf = skeleton(left_wrist_y=400, right_wrist_y=400)
    assert not detect_hand_raise(kps, conf)


def test_hand_well_above_head_is_voting():
    kps, conf = skeleton(left_wrist_y=20)
    v = detect_hand_raise(kps, conf)
    assert v.raised and v.sides == ("left",)


def test_hand_level_with_the_face_counts():
    """The stated rule is level-with-or-above, not strictly above."""
    kps, conf = skeleton(nose_y=100, shoulder_y=200, left_wrist_y=144)
    # threshold sits at nose + 0.45 * (shoulder - nose) = 145
    assert detect_hand_raise(kps, conf).raised


def test_hand_just_below_face_level_does_not_count():
    kps, conf = skeleton(nose_y=100, shoulder_y=200, left_wrist_y=160)
    assert not detect_hand_raise(kps, conf)


def test_both_hands_reported():
    kps, conf = skeleton(left_wrist_y=30, right_wrist_y=40)
    assert set(detect_hand_raise(kps, conf).sides) == {"left", "right"}


def test_threshold_scales_with_distance():
    """The same gesture must read the same whether near or far from the camera.

    A pixel threshold would silently stop working across the room, which is
    exactly the failure a room robot would hit.
    """
    near = detect_hand_raise(*skeleton(left_wrist_y=120, scale=1.0))
    far = detect_hand_raise(*skeleton(left_wrist_y=120, scale=0.3))
    assert near.raised == far.raised is True

    near_down = detect_hand_raise(*skeleton(left_wrist_y=300, scale=1.0))
    far_down = detect_hand_raise(*skeleton(left_wrist_y=300, scale=0.3))
    assert near_down.raised == far_down.raised is False


def test_unseen_wrist_is_not_a_lowered_wrist():
    """An occluded arm must not be read as "hand down" - it is simply unknown."""
    kps, conf = skeleton(left_wrist_y=20, missing=(LEFT_WRIST,))
    v = detect_hand_raise(kps, conf)
    assert not v.raised  # the raised wrist is unseen, so no claim either way
    assert "left" not in v.sides


def test_low_confidence_keypoints_are_not_trusted():
    """YOLO parks unseen joints at guessed positions; using them invents gestures."""
    kps, conf = skeleton(left_wrist_y=20, conf=0.2)
    assert not detect_hand_raise(kps, conf, min_conf=0.5).raised


def test_no_head_keypoint_means_no_verdict():
    kps, conf = skeleton(left_wrist_y=20, missing=(NOSE,))
    v = detect_hand_raise(kps, conf)
    assert not v.raised and "head" in v.reason


def test_falls_back_to_eyes_when_shoulders_are_hidden():
    """A person behind a table still has a usable body scale."""
    kps, conf = skeleton(left_wrist_y=20, missing=(LEFT_SHOULDER, RIGHT_SHOULDER))
    assert head_scale(kps, conf, 0.5) is not None
    assert detect_hand_raise(kps, conf).raised


def test_stricter_fraction_demands_a_higher_hand():
    kps, conf = skeleton(nose_y=100, shoulder_y=200, left_wrist_y=130)
    assert detect_hand_raise(kps, conf, face_level_frac=0.45).raised
    assert not detect_hand_raise(kps, conf, face_level_frac=0.1).raised


@pytest.mark.parametrize("frac", [0.0, 0.45, 1.0])
def test_threshold_stays_between_nose_and_shoulders(frac):
    kps, conf = skeleton(nose_y=100, shoulder_y=200)
    v = detect_hand_raise(kps, conf, face_level_frac=frac)
    assert 100 <= v.threshold_y <= 200
