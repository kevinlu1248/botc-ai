"""Shared test fixtures: hand-built skeletons.

Fabricating poses means edge cases (occluded wrist, distant person, hidden
shoulders) are testable without filming somebody performing each one.
"""

from __future__ import annotations

import numpy as np

from botcirl.gestures import LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_SHOULDER, RIGHT_WRIST


def skeleton(nose_y=100.0, shoulder_y=200.0, left_wrist_y=400.0, right_wrist_y=400.0,
             conf=0.9, missing=(), scale=1.0, x0=300.0):
    """A minimal upright person. `scale` shrinks them as if further away."""
    kps = np.zeros((17, 2), np.float32)
    confs = np.full(17, conf, np.float32)

    def place(idx, x, y):
        kps[idx] = (x0 + x * scale, y * scale)

    place(NOSE, 0, nose_y)
    place(1, -10, nose_y - 15)   # left eye
    place(2, 10, nose_y - 15)    # right eye
    place(LEFT_SHOULDER, -40, shoulder_y)
    place(RIGHT_SHOULDER, 40, shoulder_y)
    place(7, -55, (shoulder_y + left_wrist_y) / 2)   # left elbow
    place(8, 55, (shoulder_y + right_wrist_y) / 2)   # right elbow
    place(LEFT_WRIST, -60, left_wrist_y)
    place(RIGHT_WRIST, 60, right_wrist_y)
    for idx in missing:
        confs[idx] = 0.05
    return kps, confs
