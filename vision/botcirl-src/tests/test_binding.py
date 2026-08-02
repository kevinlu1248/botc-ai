"""Voice-to-face linking. Pure logic, no models, fast."""

from __future__ import annotations

import numpy as np
import pytest

from botcirl.audio import SpeechSegment
from botcirl.binding import LinkStats, SpeakerBinder

KNOWN = {"p1", "p2", "p3"}


def seg(duration=2.0):
    return SpeechSegment(start=0.0, end=duration,
                         audio=np.zeros(int(duration * 16000), np.float32))


def test_alone_in_the_room_settles_it():
    b = SpeakerBinder()
    att = b.attribute(seg(), voice_pid="v1", present={"p1"}, known_faces=KNOWN)
    assert att.speaker_pid == "p1"
    assert att.basis == "only person visible"


def test_a_crowd_without_a_link_is_honest_about_not_knowing():
    b = SpeakerBinder()
    att = b.attribute(seg(), voice_pid="v1", present={"p1", "p2"}, known_faces=KNOWN)
    assert att.speaker_pid is None
    assert not att.attributed
    assert "cannot tell them apart" in att.basis


def test_nobody_visible_is_not_an_attribution():
    b = SpeakerBinder()
    att = b.attribute(seg(), voice_pid="v1", present=set(), known_faces=KNOWN)
    assert att.speaker_pid is None
    assert att.basis == "nobody visible"


def test_link_is_learned_from_one_person_moments_then_used_in_a_crowd():
    """The whole point: bootstrap alone, then work when it is busy."""
    b = SpeakerBinder(min_support=4, min_strength=0.6)

    for _ in range(8):  # v1 speaks while only p1 is around
        b.attribute(seg(), "v1", present={"p1"}, known_faces=KNOWN)

    face, strength = b.best_link("v1")
    assert face == "p1" and strength >= 0.6

    # Now three people are present - presence alone could not choose.
    att = b.attribute(seg(), "v1", present={"p1", "p2", "p3"}, known_faces=KNOWN)
    assert att.speaker_pid == "p1"
    assert att.basis == "learned voice-face link"


def test_absence_is_evidence_against():
    """Without negative evidence a voice links to whoever is around most."""
    b = SpeakerBinder(min_support=4, min_strength=0.6)
    # p2 is present every single time, but so is p1 - and later p1 leaves.
    for _ in range(6):
        b.observe("v1", present={"p1", "p2"}, known_faces=KNOWN)
    # v1 keeps speaking while p2 is gone, so p2 cannot be v1.
    for _ in range(6):
        b.observe("v1", present={"p1"}, known_faces=KNOWN)

    face, _ = b.best_link("v1")
    assert face == "p1"
    assert b.links["v1"]["p2"].apart == 6


def test_weak_evidence_does_not_become_a_link():
    b = SpeakerBinder(min_support=4, min_strength=0.6)
    b.observe("v1", present={"p1"}, known_faces=KNOWN)  # a single observation
    face, strength = b.best_link("v1")
    assert face is None, "one co-occurrence must not become a certainty"


def test_strength_rewards_evidence_not_just_ratio():
    """Twenty observations at 95% should beat two at 100%."""
    many = LinkStats(together=19, apart=1)
    few = LinkStats(together=2, apart=0)
    assert many.strength() > few.strength()


def test_conflicting_evidence_weakens_the_link():
    b = SpeakerBinder(min_support=4, min_strength=0.6)
    for _ in range(10):
        b.observe("v1", present={"p1"}, known_faces=KNOWN)
    strong = b.best_link("v1")[1]
    for _ in range(10):  # now v1 speaks with p1 nowhere to be seen
        b.observe("v1", present={"p2"}, known_faces=KNOWN)
    weaker = b.best_link("v1")[1]
    assert weaker < strong


def test_a_wrong_link_cannot_keep_confirming_itself():
    """Reinforcement requires the face to actually be present."""
    b = SpeakerBinder(min_support=4, min_strength=0.6)
    for _ in range(8):
        b.attribute(seg(), "v1", present={"p1"}, known_faces=KNOWN)
    before = b.links["v1"]["p1"].together

    # v1 speaks while p1 is absent; the link is used but must not be reinforced.
    b.attribute(seg(), "v1", present={"p2"}, known_faces=KNOWN)
    assert b.links["v1"]["p1"].together == before


def test_links_persist(tmp_path):
    path = tmp_path / "links.json"
    b = SpeakerBinder(path, min_support=4)
    for _ in range(8):
        b.observe("v1", present={"p1"}, known_faces=KNOWN)
    b.save()

    reloaded = SpeakerBinder(path, min_support=4)
    assert reloaded.best_link("v1")[0] == "p1"
    assert reloaded.linked_pairs()[0][:2] == ("v1", "p1")


def test_no_voice_identity_still_falls_back_to_presence():
    b = SpeakerBinder()
    att = b.attribute(seg(), voice_pid=None, present={"p1"}, known_faces=KNOWN)
    assert att.speaker_pid == "p1"


def test_voice_and_face_ids_live_in_separate_namespaces(tmp_path):
    """Both galleries used to mint "p1", so a link could point at itself.

    A voice->face link of "p1 -> p1" is two unrelated people sharing a string,
    and nothing downstream could tell the difference.
    """
    import numpy as np

    from botcirl.identity import IdentityGallery

    faces = IdentityGallery(tmp_path / "faces", prefix="p", label_word="Person")
    voices = IdentityGallery(tmp_path / "voices", prefix="v", label_word="Speaker")

    f = faces.enroll(np.ones(8, np.float32) / np.sqrt(8))
    v = voices.enroll(np.ones(8, np.float32) / np.sqrt(8))

    assert f.pid == "p1" and v.pid == "v1"
    assert f.pid != v.pid, "face and voice ids must not collide"
    assert f.label == "Person 1" and v.label == "Speaker 1"
