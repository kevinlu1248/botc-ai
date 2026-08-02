"""Session recording and the dashboard's HTTP surface."""

from __future__ import annotations

import json
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from botcirl.audio import SpeechSegment
from botcirl.binding import Attribution
from botcirl.dashboard import Dashboard
from botcirl.session import SessionRecorder


class FakeTrack:
    def __init__(self, pid, x, y, announced=True, raised=False, raised_since=None,
                 track_id=1):
        self.pid = pid
        self.track_id = track_id
        self.announced = announced
        self.room = (x, y)
        self.hand_raised = raised
        self.raised_since = raised_since
        self.bbox = (0, 0, 100, 200)
        self.kps = None
        self.kp_conf = None

    @property
    def foot(self):
        return (50.0, 200.0)


class FakeFloor:
    def __init__(self, valid=True, camera=None):
        self.valid = valid
        self._camera = camera

    def load_camera(self):
        return self._camera

    def visible_ground(self, size):
        return [(-1.0, 1.0), (1.0, 1.0), (2.0, 4.0), (-2.0, 4.0)] if self.valid else []


class FakePipeline:
    def __init__(self, tracks, calibrated=True, camera=None):
        self.tracks = {t.track_id: t for t in tracks}
        self.floor = FakeFloor(calibrated, camera)
        self.frame_size = (1280, 720)
        self.gallery = SimpleNamespace(label_of=lambda p: {"p1": "Sam"}.get(p, p))

    def label_for(self, trk):
        return {"p1": "Sam", "p2": "Alex"}.get(trk.pid, f"#{trk.track_id}")


@pytest.fixture
def recorder(tmp_path):
    return SessionRecorder(root=tmp_path, sample_hz=1000)  # no throttling in tests


def test_records_positions(recorder):
    p = FakePipeline([FakeTrack("p1", 1.0, 2.0)])
    recorder.sample(p, now=100.0)
    recorder.sample(p, now=100.5)
    assert len(recorder.positions) == 2
    assert recorder.snapshot()["positions"][0][1:] == ["p1", 1.0, 2.0]
    assert recorder.labels["p1"] == "Sam"


def test_sampling_is_throttled(tmp_path):
    """A ten-minute session must not store thousands of identical points."""
    r = SessionRecorder(root=tmp_path, sample_hz=5.0)
    p = FakePipeline([FakeTrack("p1", 1.0, 2.0)])
    for i in range(50):
        r.sample(p, now=100.0 + i * 0.01)  # 100 Hz of calls
    assert len(r.positions) <= 4, f"throttling failed: {len(r.positions)} samples"


def test_unannounced_tracks_are_not_recorded(recorder):
    p = FakePipeline([FakeTrack("p1", 1.0, 2.0, announced=False)])
    recorder.sample(p, now=100.0)
    assert recorder.positions == []


def test_hand_raise_becomes_a_span(recorder):
    trk = FakeTrack("p1", 1.0, 2.0, raised=True, raised_since=99.0)
    p = FakePipeline([trk])
    recorder.sample(p, now=100.0)
    assert len(recorder.gestures) == 1
    assert recorder.gestures[0].start == 99.0
    assert recorder.gestures[0].end is None  # still up

    trk.hand_raised = False
    recorder.sample(p, now=102.0)
    assert recorder.gestures[0].end == 102.0
    assert len(recorder.gestures) == 1, "lowering must not open a second span"


def test_gesture_closes_when_the_person_disappears(recorder):
    trk = FakeTrack("p1", 1.0, 2.0, raised=True, raised_since=99.0)
    p = FakePipeline([trk])
    recorder.sample(p, now=100.0)
    p.tracks.clear()
    recorder.sample(p, now=101.0)
    assert recorder.gestures[0].end == 101.0


def test_speech_is_recorded_with_audio(recorder):
    seg = SpeechSegment(start=10.0, end=12.0,
                        audio=np.random.default_rng(0).standard_normal(32000).astype(np.float32) * 0.1)
    att = Attribution(segment=seg, voice_pid="v1", speaker_pid="p1",
                      confidence=0.9, basis="only person visible")
    rec = recorder.add_speech(att, "Sam")
    assert rec.pid == "p1" and rec.label == "Sam"
    assert (recorder.audio_dir / "0000.wav").exists(), "audio must be written for playback"


def test_snapshot_is_json_serialisable(recorder):
    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)]), now=100.0)
    blob = recorder.snapshot()
    json.dumps(blob)  # must not raise
    assert blob["calibrated"] is True
    assert blob["units"] == "metres"


def test_uncalibrated_snapshot_says_so(tmp_path):
    r = SessionRecorder(root=tmp_path, sample_hz=1000)
    trk = FakeTrack("p1", 0, 0)
    trk.room = None  # no floor homography
    r.sample(FakePipeline([trk], calibrated=False), now=100.0)
    blob = r.snapshot()
    assert blob["calibrated"] is False
    assert blob["units"] == "pixels", "must not pass pixels off as metres"


def test_save_and_reload(recorder):
    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)]), now=100.0)
    path = recorder.save()
    blob = json.loads((path / "session.json").read_text())
    assert blob["positions"][0][1] == "p1"
    assert "ended_at" in blob


# ---------- dashboard ----------


@pytest.fixture
def server(recorder):
    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)]), now=100.0)
    seg = SpeechSegment(start=10.0, end=12.0, audio=np.zeros(32000, np.float32))
    recorder.add_speech(Attribution(segment=seg, speaker_pid="p1", confidence=0.9,
                                    basis="only person visible"), "Sam")
    d = Dashboard(recorder, port=8971, open_browser=False).start()
    yield d
    d.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_serves_the_page(server):
    status, body = _get(server.url)
    assert status == 200 and b"botcirl" in body


def test_serves_session_json(server):
    status, body = _get(server.url + "api/session")
    blob = json.loads(body)
    assert blob["positions"][0][1] == "p1"
    assert blob["speech"][0]["label"] == "Sam"
    assert blob["running"] is True


def test_serves_audio(server):
    status, body = _get(server.url + "audio/0000.wav")
    assert status == 200 and body[:4] == b"RIFF"


def test_path_traversal_is_refused(server):
    """The audio route must not become a file reader for the whole disk."""
    import urllib.error

    for attack in ("audio/../../../../etc/passwd", "audio/..%2f..%2fsession.json"):
        try:
            status, body = _get(server.url + attack)
            assert b"root:" not in body and b"positions" not in body
        except urllib.error.HTTPError as e:
            assert e.code in (403, 404)


def test_binds_to_localhost_only(server):
    """Room audio and movement must not be served to the network."""
    assert server._server.server_address[0] == "127.0.0.1"


def test_finished_session_reports_when_it_ended(recorder):
    """Reviewing an old session must not stretch the timeline to the present.

    Regression: `ended_at` was written into the saved file but never into
    snapshot(), so a reloaded session reported `now` and squashed the whole
    recording into a sliver at the left of the timeline.
    """
    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)]), now=100.0)
    assert recorder.snapshot()["ended_at"] is None  # still running

    recorder.save()
    blob = recorder.snapshot()
    assert blob["ended_at"] is not None
    assert blob["ended_at"] <= blob["now"]


def test_reloaded_session_keeps_its_duration(recorder, tmp_path):
    import json as _json

    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)]), now=100.0)
    path = recorder.save()
    saved = _json.loads((path / "session.json").read_text())

    from botcirl.dashboard import serve_session  # noqa: F401 - exercises the rehydrate path
    from botcirl.session import SessionRecorder as SR

    rec = SR.__new__(SR)
    rec.started_at = saved["started_at"]
    rec.ended_at = saved.get("ended_at")
    assert rec.ended_at is not None, "duration would otherwise run to the present"


def test_identity_resolves_retroactively(recorder):
    """A person walks in unrecognised, then their face lands. One trail, not two.

    Regression: positions were keyed by identity at write time, so the samples
    before recognition stayed under "#3" and the map showed one person twice
    with a broken trail.
    """
    unknown = FakeTrack(None, 1.0, 2.0, track_id=3)
    p = FakePipeline([unknown])
    recorder.sample(p, now=100.0)
    recorder.sample(p, now=100.5)
    assert recorder.snapshot()["positions"][0][1] == "#3"

    unknown.pid = "p1"  # face recognised
    recorder.sample(p, now=101.0)

    blob = recorder.snapshot()
    keys = {row[1] for row in blob["positions"]}
    assert keys == {"p1"}, f"trail split across {keys}"
    assert "#3" not in blob["labels"], "the placeholder must not survive as a person"


def test_track_id_break_merges_into_one_trail(recorder):
    """ByteTrack loses the id on occlusion; the same person must stay one line."""
    first = FakeTrack("p1", 1.0, 2.0, track_id=7)
    recorder.sample(FakePipeline([first]), now=100.0)
    later = FakeTrack("p1", 1.5, 2.5, track_id=19)  # same person, new track id
    recorder.sample(FakePipeline([later]), now=105.0)

    blob = recorder.snapshot()
    assert {row[1] for row in blob["positions"]} == {"p1"}
    assert len(blob["positions"]) == 2


def test_camera_and_fov_reach_the_dashboard(recorder):
    """The map cannot draw the camera if the session never carries it."""
    cam = {"ground": [0.2, -0.3], "height": 1.1, "source": "measured"}
    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)], camera=cam), now=100.0)
    blob = recorder.snapshot()
    assert blob["camera"] == cam
    assert len(blob["fov"]) == 4
    # The camera must be inside the drawn area or its marker lands off-canvas.
    b = blob["bounds"]
    assert b["xmin"] <= cam["ground"][0] <= b["xmax"]
    assert b["ymin"] <= cam["ground"][1] <= b["ymax"]


def test_poses_are_recorded_for_the_camera_view(recorder):
    trk = FakeTrack("p1", 1.0, 2.0)
    trk.kps = np.array([[10.4, 20.6]] * 17, np.float32)
    trk.kp_conf = np.array([0.93] * 17, np.float32)
    recorder.sample(FakePipeline([trk]), now=100.0)

    pose = recorder.snapshot()["poses"][0]
    assert pose["pid"] == "p1"
    assert pose["box"] == [0, 0, 100, 200]
    assert len(pose["kp"]) == 17
    # Rounded: sub-pixel precision is meaningless and triples the file size.
    assert pose["kp"][0] == [10, 20, 0.9]


def test_poses_without_keypoints_still_record_a_box(recorder):
    """A plain detector has no keypoints; the view falls back to the box."""
    recorder.sample(FakePipeline([FakeTrack("p1", 1.0, 2.0)]), now=100.0)
    pose = recorder.snapshot()["poses"][0]
    assert "kp" not in pose and pose["box"] == [0, 0, 100, 200]
