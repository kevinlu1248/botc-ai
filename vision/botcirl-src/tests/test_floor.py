"""Floor homography maths. Fast - no models involved."""

from __future__ import annotations

import numpy as np
import pytest

from botcirl.floor import FloorMap

# A plausible calibration: four floor marks seen by a camera looking across a
# room. Far points sit higher in the frame and closer together, as perspective
# requires.
IMAGE_PTS = [(300, 700), (980, 700), (860, 430), (420, 430)]
ROOM_PTS = [(-1.0, 1.0), (1.0, 1.0), (1.0, 4.0), (-1.0, 4.0)]


@pytest.fixture
def fm(tmp_path):
    m = FloorMap(tmp_path / "H.npy")
    m.save(FloorMap.solve(IMAGE_PTS, ROOM_PTS))
    return m


def test_calibration_points_map_back_to_themselves(fm):
    for (px, py), (rx, ry) in zip(IMAGE_PTS, ROOM_PTS):
        got = fm.to_room(px, py)
        assert got == pytest.approx((rx, ry), abs=1e-3)


def test_midpoint_lands_between(fm):
    """A point halfway along the near edge should read as roughly (0, 1)."""
    x, y = fm.to_room(640, 700)
    assert x == pytest.approx(0.0, abs=0.05)
    assert y == pytest.approx(1.0, abs=0.05)


def test_distance_matches_the_room(fm):
    d = fm.distance(IMAGE_PTS[0], IMAGE_PTS[1])
    assert d == pytest.approx(2.0, abs=1e-2)


def test_uncalibrated_map_is_honest_about_it(tmp_path):
    m = FloorMap(tmp_path / "missing.npy")
    assert not m.valid
    assert m.to_room(100, 100) is None
    assert m.distance((0, 0), (1, 1)) is None


def test_persists(tmp_path, fm):
    reloaded = FloorMap(tmp_path / "H.npy")
    assert reloaded.valid
    assert reloaded.to_room(*IMAGE_PTS[2]) == pytest.approx(ROOM_PTS[2], abs=1e-3)


def test_collinear_image_points_are_rejected():
    """cv2 returns a matrix for these; we must not."""
    line = [(0, 0), (10, 10), (20, 20), (30, 30)]
    with pytest.raises(ValueError, match="straight line"):
        FloorMap.solve(line, ROOM_PTS)


def test_collinear_room_points_are_rejected():
    line = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
    with pytest.raises(ValueError, match="straight line"):
        FloorMap.solve(IMAGE_PTS, line)


def test_four_points_cannot_be_validated(fm):
    """Documents a real limit, so nobody trusts the check more than it deserves.

    Four correspondences are an exact solve: they reproduce themselves however
    wrong they are. Only a 5th point makes the residual check mean anything.
    """
    swapped = [ROOM_PTS[0], ROOM_PTS[2], ROOM_PTS[1], ROOM_PTS[3]]
    assert FloorMap.solve(IMAGE_PTS, swapped) is not None  # accepted, and wrong


def _consistent_fifth(fm, px, py):
    return fm.to_room(px, py)


def test_a_grossly_mistyped_fifth_point_is_rejected(fm):
    fifth_img = (640, 520)
    fifth_room = _consistent_fifth(fm, *fifth_img)
    good_img = IMAGE_PTS + [fifth_img]

    # Consistent -> fits fine.
    assert FloorMap.solve(good_img, ROOM_PTS + [fifth_room]) is not None

    # Off by most of a metre -> caught.
    wrong = (fifth_room[0], fifth_room[1] + 0.8)
    with pytest.raises(ValueError, match="does not reproduce"):
        FloorMap.solve(good_img, ROOM_PTS + [wrong])


def test_small_measurement_error_is_tolerated(fm):
    """Tape-measure and click error must not block an otherwise fine calibration."""
    fifth_img = (640, 520)
    fifth_room = _consistent_fifth(fm, *fifth_img)
    slightly_off = (fifth_room[0] + 0.05, fifth_room[1] - 0.08)
    assert FloorMap.solve(IMAGE_PTS + [fifth_img], ROOM_PTS + [slightly_off]) is not None


def test_too_few_points():
    with pytest.raises(ValueError, match="at least 4"):
        FloorMap.solve(IMAGE_PTS[:3], ROOM_PTS[:3])


# ---------- camera recovery ----------


def _synthetic_camera(f=900.0, height=1.2, tilt_deg=20.0, cam=(0.0, 0.0), size=(1280, 720)):
    """Ground->image for a camera `height` up, tilted down, looking along +y."""
    w, h = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    td = np.deg2rad(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(td), -np.sin(td)], [0, np.sin(td), np.cos(td)]])
    R = Rx @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    t = -R @ np.array([cam[0], cam[1], height])
    G = K @ np.column_stack([R[:, 0], R[:, 1], t])
    m = FloorMap.__new__(FloorMap)
    m.H, m.valid = np.linalg.inv(G), True
    return m, size


def test_recovers_camera_position_and_height():
    for cam, height in [((0.0, 0.0), 1.2), ((0.8, -0.5), 1.2), ((-1.2, 0.3), 2.4)]:
        m, size = _synthetic_camera(height=height, cam=cam)
        pose = m.camera_pose(size)
        assert pose is not None
        assert pose["ground"] == pytest.approx(cam, abs=0.01)
        assert pose["height"] == pytest.approx(height, abs=0.01)


def test_camera_pose_is_scale_invariant():
    """The homography's overall scale is arbitrary and must not change the answer."""
    m, size = _synthetic_camera()
    a = m.camera_pose(size)
    m.H = m.H * 7.3
    b = m.camera_pose(size)
    assert a["ground"] == pytest.approx(b["ground"], abs=1e-6)
    assert a["height"] == pytest.approx(b["height"], abs=1e-6)


def test_uncalibrated_has_no_camera_pose(tmp_path):
    m = FloorMap(tmp_path / "none.npy")
    assert m.camera_pose((1280, 720)) is None
    assert m.visible_ground((1280, 720)) == []


def test_visible_ground_drops_points_beyond_the_horizon():
    """Above the horizon maps behind the camera; drawing it gives a wrong shape."""
    m, size = _synthetic_camera()
    poly = m.visible_ground(size)
    assert len(poly) > 8
    # Everything kept must be in front of the camera and at a sane distance.
    assert all(abs(x) <= 60 and abs(y) <= 60 for x, y in poly)
    assert min(y for _, y in poly) > 0, "visible floor should be in front of the camera"


def test_camera_position_persists(tmp_path):
    m = FloorMap(tmp_path / "H.npy")
    m.save(FloorMap.solve(IMAGE_PTS, ROOM_PTS))
    m.save_camera((0.3, -0.4), 1.15, source="measured")

    reloaded = FloorMap(tmp_path / "H.npy")
    cam = reloaded.load_camera()
    assert cam["ground"] == [0.3, -0.4]
    assert cam["height"] == pytest.approx(1.15)
    assert cam["source"] == "measured"


def test_no_camera_recorded_reads_as_none(tmp_path):
    m = FloorMap(tmp_path / "H.npy")
    m.save(FloorMap.solve(IMAGE_PTS, ROOM_PTS))
    assert m.load_camera() is None
