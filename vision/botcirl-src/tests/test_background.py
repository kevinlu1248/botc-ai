"""Recovering the empty room from a fixed camera."""

from __future__ import annotations

import numpy as np
import pytest

from botcirl.background import BackgroundPlate

H, W = 120, 200


def room(seed=0):
    """A fixed 'room' with structure, so a person standing in it is detectable."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = (60, 90, 130)
    img[:40, :] = (30, 40, 60)  # a wall
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    return img


def with_person(base, x0, x1, colour=(240, 20, 240)):
    f = base.copy()
    f[40:H, x0:x1] = colour
    return f


def test_recovers_the_room_where_nobody_stood():
    p = BackgroundPlate(alpha=0.5, dilate_px=0)
    base = room()
    for _ in range(6):
        p.update(base, [])
    got = p.image()
    assert np.abs(got.astype(int) - base.astype(int)).mean() < 3


def test_a_person_who_moves_is_not_baked_in():
    p = BackgroundPlate(alpha=0.5, dilate_px=2)
    base = room()
    # Somebody crosses the room; every column is eventually seen empty.
    for x in range(0, 160, 20):
        p.update(with_person(base, x, x + 30), [(x, 40, x + 30, H)])
    for _ in range(6):
        p.update(base, [])

    got = p.image().astype(int)
    # No trace of the bright person colour should survive.
    assert np.abs(got - base.astype(int)).mean() < 6
    assert p.coverage() > 0.95


def test_a_motionless_person_is_never_baked_in():
    """The bug this exists to prevent.

    Seeding the plate from the first frame put whoever was already in the room
    into the 'empty room' image permanently, because those pixels were never
    observed without them. It has to come out as obviously-unknown instead.
    """
    p = BackgroundPlate(alpha=0.5, dilate_px=0)
    base = room()
    person = (240, 20, 240)
    for _ in range(12):
        p.update(with_person(base, 60, 100, person), [(60, 40, 100, H)])

    got = p.image().astype(int)
    occupied = got[40:H, 60:100]
    # Whatever is there, it must not be the person.
    assert np.abs(occupied - np.array(person)).mean() > 40, "person baked into the plate"

    mask = p.unobserved_mask()
    assert mask[40:H, 60:100].all(), "their pixels must be flagged as never observed"
    assert not mask[0:30, 0:30].any(), "the rest of the room was seen and is fine"
    assert p.coverage() < 0.95, "coverage must admit the hidden patch"


def test_coverage_starts_at_zero_and_grows():
    p = BackgroundPlate(alpha=0.5, dilate_px=0)
    assert p.coverage() == 0.0
    for _ in range(5):
        p.update(room(), [])
    assert p.coverage() == pytest.approx(1.0)


def test_boxes_are_dilated_so_outlines_do_not_bleed():
    """Detection boxes clip hair and shoulders; a hard crop leaves a halo."""
    p = BackgroundPlate(alpha=0.5, dilate_px=10)
    base = room()
    p.update(with_person(base, 60, 100), [(60, 40, 100, H)])
    mask = p.unobserved_mask(min_observations=1)
    assert mask[45, 55], "should exclude a margin outside the box"
    assert not mask[45, 20], "far from the person must still be recovered"


def test_no_frames_yet_is_honest():
    p = BackgroundPlate()
    assert p.image() is None
    assert p.coverage() == 0.0
    assert p.unobserved_mask() is None


def test_saves_a_jpeg(tmp_path):
    p = BackgroundPlate(alpha=0.5)
    p.update(room(), [])
    out = p.save(tmp_path / "plate.jpg")
    assert out.exists() and out.read_bytes()[:2] == b"\xff\xd8"
