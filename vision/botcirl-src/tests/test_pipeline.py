"""End-to-end checks against a real group photo.

Slow (they run the actual models) but they cover the parts that are easy to get
quietly wrong: face-to-body binding, when a new identity is minted, and whether
somebody is recognised again after a restart.

    .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from botcirl import Config, Pipeline
from botcirl.identity import IdentityGallery

GROUP_PHOTO = (
    Path(__file__).resolve().parent.parent
    / ".venv/lib/python3.13/site-packages/insightface/data/images/t1.jpg"
)


@pytest.fixture(scope="module")
def photo():
    if not GROUP_PHOTO.exists():
        pytest.skip(f"test image not found at {GROUP_PHOTO}")
    return cv2.imread(str(GROUP_PHOTO))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A Config whose gallery and floor map live in a throwaway directory."""
    import botcirl.config as config_mod

    monkeypatch.setattr(config_mod, "GALLERY_DIR", tmp_path / "gallery")
    monkeypatch.setattr(config_mod, "FLOOR_HOMOGRAPHY", tmp_path / "floor.npy")
    c = Config()
    c.face.every_n_frames = 1  # tests should not wait on frame parity
    return c


def test_tests_do_not_touch_the_real_gallery(cfg, tmp_path, photo):
    """Guards the isolation the other tests depend on.

    Regression: the gallery path was a default argument, so it bound the real
    GALLERY_DIR at import and every test run enrolled strangers into the user's
    actual gallery while appearing to use a tmp dir.
    """
    # REPO_ROOT is not patched, so this is the genuine on-disk location
    # regardless of what the fixture redirected.
    from botcirl.config import REPO_ROOT

    real_gallery = REPO_ROOT / "data" / "gallery"
    existed_before = real_gallery.exists()

    p = Pipeline(cfg)
    assert p.gallery.path == tmp_path / "gallery"
    assert p.floor.path == tmp_path / "floor.npy"

    _run(p, photo, 8)
    p.close()

    assert p.gallery.identities, "the test should have enrolled somebody"
    assert (tmp_path / "gallery" / "gallery.json").exists()
    if not existed_before:
        assert not real_gallery.exists(), "the test wrote into the real gallery"


def _run(pipeline, frame, n):
    scene = None
    for _ in range(n):
        scene = pipeline.process(frame.copy())
    return scene


def test_detects_and_identifies_the_group(cfg, photo):
    p = Pipeline(cfg)
    scene = _run(p, photo, 8)

    assert len(scene.people) >= 5, "should find the bodies in the group photo"
    assert len(scene.faces) >= 5, "should find the faces too"

    identified = [t for t in scene.people if t.pid]
    assert len(identified) >= 4, "most visible faces should get an identity"

    # No two tracks may hold the same identity at once.
    pids = [t.pid for t in identified]
    assert len(pids) == len(set(pids)), f"duplicate identity across tracks: {pids}"

    # Every bound face must actually sit inside its person's box.
    for trk in scene.people:
        if trk.face is None:
            continue
        fx, fy = trk.face.center
        x1, y1, x2, y2 = trk.bbox
        assert x1 <= fx <= x2 and y1 <= fy <= y2, f"face bound to wrong body: track {trk.track_id}"


def test_does_not_enroll_low_quality_faces(cfg, photo):
    """A hard profile shot should be refused, not turned into a phantom person."""
    p = Pipeline(cfg)
    scene = _run(p, photo, 8)
    for trk in scene.people:
        if trk.face is not None and trk.face.quality() < 0.35:
            assert trk.pid is None, "a low-quality face should not mint an identity"


def test_reidentifies_after_restart(cfg, photo):
    """The whole point: a person who leaves and comes back keeps their name."""
    p1 = Pipeline(cfg)
    _run(p1, photo, 8)
    enrolled = {pid: idt.label for pid, idt in p1.gallery.identities.items()}
    p1.close()
    assert len(enrolled) >= 4

    # A different sighting: dimmer, softer, smaller in frame.
    alt = cv2.resize(
        cv2.convertScaleAbs(cv2.GaussianBlur(photo, (3, 3), 0), alpha=0.78, beta=28),
        None,
        fx=0.85,
        fy=0.85,
    )

    p2 = Pipeline(cfg)  # fresh tracker state, gallery reloaded from disk
    assert set(p2.gallery.identities) == set(enrolled), "gallery should survive restart"
    scene = _run(p2, alt, 8)

    matched = [t for t in scene.people if t.pid in enrolled]
    assert len(matched) >= 4, "known people should be recognised, not re-enrolled"
    assert len(p2.gallery.identities) <= len(enrolled) + 1, (
        f"re-enrolled strangers: {len(enrolled)} -> {len(p2.gallery.identities)}"
    )


def test_rename_persists(cfg, photo):
    p = Pipeline(cfg)
    _run(p, photo, 8)
    pid = next(iter(p.gallery.identities))
    p.gallery.rename(pid, "Sam")
    p.close()

    reloaded = IdentityGallery(p.gallery.path)
    assert reloaded.identities[pid].label == "Sam"


def test_one_identity_per_track_and_no_event_spam(cfg, photo):
    """Two tracks must not trade one identity back and forth every frame.

    Regression: conflict resolution stripped the identity from the loser, which
    then instantly re-claimed it, emitting an identical assignment forever.
    """
    p = Pipeline(cfg)
    events = []
    for _ in range(30):
        scene = p.process(photo.copy())
        events.extend(p.drain_events())

    live = [t for t in scene.people if t.pid]
    assert len({t.pid for t in live}) == len(live), "an identity is held by two tracks at once"

    # Each (track, identity) pair should be announced once, not once per frame.
    assignments = [e for e in events if e["event"] == "identity_assigned"]
    seen = [(e["track_id"], e["pid"]) for e in assignments]
    assert len(seen) == len(set(seen)), f"repeated identity_assigned events: {seen}"


def test_faces_bind_to_plausible_bodies(cfg, photo):
    """A face must be near the top of its body box and a sane size for it.

    Regression: in a crowd a face sits inside several boxes, and binding on
    containment alone handed one person's face to whoever stood in front.
    """
    p = Pipeline(cfg)
    scene = _run(p, photo, 6)
    bound = [t for t in scene.people if t.face is not None]
    assert bound, "expected at least some faces bound to bodies"

    for trk in bound:
        x1, y1, x2, y2 = trk.bbox
        bw, bh = x2 - x1, y2 - y1
        depth = (trk.face.center[1] - y1) / bh
        assert depth <= cfg.track.max_face_depth + 1e-6, (
            f"face bound below the plausible head region (depth {depth:.2f})"
        )
        ratio = trk.face.width / bw
        assert cfg.track.min_face_body_ratio <= ratio <= cfg.track.max_face_body_ratio, (
            f"face is an implausible fraction of the body width ({ratio:.2f})"
        )

    # No face may be bound to two different tracks.
    ids = [id(t.face) for t in bound]
    assert len(ids) == len(set(ids)), "the same face was bound to more than one body"


def test_backend_supplies_its_own_thresholds(cfg):
    """ArcFace and SFace separate people at different scales.

    Reusing one model's thresholds for the other either merges strangers or
    never recognises anyone, so an unset threshold must come from the backend.
    """
    assert cfg.face.match_threshold is None  # unset by default
    p = Pipeline(cfg)
    assert p.cfg.face.match_threshold == p.faces.default_match_threshold
    assert p.cfg.face.enroll_threshold == p.faces.default_enroll_threshold
    assert p.cfg.face.enroll_threshold < p.cfg.face.match_threshold


def test_hand_raise_is_debounced(cfg, photo):
    """A flickering wrist must not strobe the box between voting and not.

    Driven by injecting poses and a controlled clock rather than real frames,
    so the timing behaviour is actually pinned down.
    """
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    up = skeleton(left_wrist_y=20)
    down = skeleton(left_wrist_y=400)

    trk = PersonTrack(track_id=1, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)

    def tick(t, pose):
        trk.kps, trk.kp_conf = pose
        p._update_gesture(trk, t)

    tick(0.0, up)
    assert not trk.hand_raised, "should not fire instantly - hold_s has not elapsed"

    tick(0.2, up)
    assert not trk.hand_raised

    tick(0.5, up)  # past hold_s = 0.4
    assert trk.hand_raised and trk.raise_sides == ("left",)
    assert any(e["event"] == "hand_raised" for e in p.events)

    # A brief drop must not retract the vote.
    tick(0.7, down)
    assert trk.hand_raised, "released too eagerly - release_s has not elapsed"
    tick(0.9, up)
    assert trk.hand_raised

    # A sustained drop does.
    tick(1.0, down)
    tick(2.0, down)  # past release_s = 0.8
    assert not trk.hand_raised
    lowered = [e for e in p.events if e["event"] == "hand_lowered"]
    assert len(lowered) == 1 and lowered[0]["held_s"] > 0

    # And exactly one of each event, not one per frame.
    assert len([e for e in p.events if e["event"] == "hand_raised"]) == 1


def test_voting_list_reports_raised_hands(cfg, photo):
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    trk = PersonTrack(track_id=7, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)
    p.tracks[7] = trk
    assert p.voting() == []

    trk.kps, trk.kp_conf = skeleton(left_wrist_y=20)
    p._update_gesture(trk, 0.0)
    p._update_gesture(trk, 1.0)
    assert p.voting() == [trk]


def test_gallery_rejects_mismatched_embedding_size(cfg, tmp_path):
    g = IdentityGallery(tmp_path / "g")
    g.enroll(np.ones(512, np.float32) / np.sqrt(512))
    g.assert_compatible(512)  # fine
    with pytest.raises(SystemExit, match="cannot be compared"):
        g.assert_compatible(128)


def test_vote_duration_excludes_the_release_debounce(cfg, photo):
    """A vote must be as long as the hand was up, not up-plus-debounce.

    Regression: held_s measured to when the debounce *confirmed* the drop, so
    every vote was reported up to release_s (0.8s) longer than it really was.
    """
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    up, down = skeleton(left_wrist_y=20), skeleton(left_wrist_y=400)
    trk = PersonTrack(track_id=4, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)

    def tick(t, pose):
        trk.kps, trk.kp_conf = pose
        p._update_gesture(trk, t)

    tick(0.0, up)
    tick(0.5, up)          # confirmed raised, raised_since = 0.0
    tick(3.2, up)
    tick(3.4, down)        # hand actually comes down here
    tick(4.5, down)        # debounce confirms it

    down = next(e for e in p.events if e["event"] == "hand_lowered")
    assert down["held_s"] == pytest.approx(3.4, abs=0.01), (
        f"vote inflated by the debounce: {down['held_s']}"
    )
    assert down["lowered_at"] == pytest.approx(3.4, abs=0.01)


def test_hand_up_and_hand_down_are_both_marked(cfg, photo):
    """One mark when the hand goes up, one when it comes down. Nothing else."""
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    trk = PersonTrack(track_id=4, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)
    trk.pid = None
    for t, pose in [(0.0, skeleton(left_wrist_y=20)), (0.6, skeleton(left_wrist_y=20)),
                    (2.0, skeleton(left_wrist_y=400)), (3.5, skeleton(left_wrist_y=400))]:
        trk.kps, trk.kp_conf = pose
        p._update_gesture(trk, t)

    kinds = [e["event"] for e in p.events if e["event"].startswith("hand_")]
    assert kinds == ["hand_raised", "hand_lowered"], f"expected two marks, got {kinds}"

    up = next(e for e in p.events if e["event"] == "hand_raised")
    down = next(e for e in p.events if e["event"] == "hand_lowered")
    assert up["sides"] == ["left"]
    for key in ("label", "identified", "raised_at", "lowered_at", "held_s", "reason"):
        assert key in down, f"hand_lowered is missing {key}"
    assert down["identified"] is False, "an unnamed voter must be flagged, not hidden"


def test_a_vote_survives_the_voter_walking_out(cfg, photo):
    """Regression: a hand still up when the track died was never logged at all."""
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    trk = PersonTrack(track_id=4, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=0.0, announced=True)
    p.tracks[4] = trk
    for t in (0.0, 0.6):
        trk.kps, trk.kp_conf = skeleton(left_wrist_y=20)
        p._update_gesture(trk, t)
    assert trk.hand_raised

    trk.last_seen = 5.0
    p._retire(now=5.0 + cfg.track.forget_after_s + 1)

    down = next(e for e in p.events if e["event"] == "hand_lowered")
    assert down["reason"] == "person left"
    assert down["held_s"] == pytest.approx(5.0, abs=0.01)
    assert len(p.votes) == 1


def test_a_vote_survives_shutdown(cfg, photo):
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    trk = PersonTrack(track_id=4, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=9.0, announced=True)
    p.tracks[4] = trk
    for t in (0.0, 0.6):
        trk.kps, trk.kp_conf = skeleton(left_wrist_y=20)
        p._update_gesture(trk, t)

    p.close()
    down = next(e for e in p.events if e["event"] == "hand_lowered")
    assert down["reason"] == "session ended"


def test_closing_a_vote_twice_does_nothing(cfg, photo):
    """Retire-then-close must not write the same vote to the log twice."""
    from botcirl.pipeline import PersonTrack
    from helpers import skeleton

    p = Pipeline(cfg)
    trk = PersonTrack(track_id=4, bbox=(0, 0, 200, 400), score=0.9,
                      first_seen=0.0, last_seen=2.0, announced=True)
    for t in (0.0, 0.6):
        trk.kps, trk.kp_conf = skeleton(left_wrist_y=20)
        p._update_gesture(trk, t)

    p._close_vote(trk, 2.0, "lowered")
    p._close_vote(trk, 2.0, "lowered")
    assert len([e for e in p.events if e["event"] == "hand_lowered"]) == 1
