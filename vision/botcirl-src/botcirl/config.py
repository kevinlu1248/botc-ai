"""Tunables for the vision prototype. Everything the pipeline reads lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GALLERY_DIR = DATA_DIR / "gallery"
VOICE_GALLERY_DIR = DATA_DIR / "voices"
EVENT_LOG = DATA_DIR / "events.jsonl"
FLOOR_HOMOGRAPHY = DATA_DIR / "floor_homography.npy"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    # Mirror the preview so it reads like a mirror. Detection is unaffected.
    flip: bool = True


@dataclass
class PersonConfig:
    # The -pose variant returns boxes *and* 17 COCO keypoints from one pass, so
    # hand-raise detection costs no extra model. Plain "yolo11n.pt" is a little
    # faster if you never need gestures. Step up to yolo11s-pose.pt if the
    # laptop keeps up. COCO class 0 is "person", trained on plenty of
    # half-occluded torsos, so above-the-waist people still land.
    model: str = "yolo11n-pose.pt"
    conf: float = 0.35
    iou: float = 0.5
    # ByteTrack keeps an id alive across a few missed frames. Ids are cheap and
    # disposable: the face gallery is what actually carries identity.
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    # None lets ultralytics choose. "mps" measured no faster than "cpu" for
    # yolo11n on Apple silicon - the model is too small to pay for the transfer.
    device: str | None = None


@dataclass
class FaceConfig:
    # Run detection on every Nth frame. Faces move slower than the frame rate and
    # the embedder is the expensive part of the loop.
    every_n_frames: int = 2
    det_size: int = 640
    min_face_px: int = 44
    # Use the Apple Neural Engine / GPU for the ONNX face models when present.
    use_coreml: bool = True
    det_conf: float = 0.6
    # Cosine similarity against the gallery. Above `match_threshold` we call it a
    # known person; below `enroll_threshold` we are confident enough that this is
    # somebody new. The gap in between is "unsure", and we do nothing.
    #
    # None means "take the backend's own defaults", which is what you want:
    # ArcFace and SFace separate people at very different scales, and reusing
    # one model's numbers for the other either merges strangers or never
    # recognises anyone. Set a number here to override.
    match_threshold: float | None = None
    enroll_threshold: float | None = None
    # Embeddings kept per identity. More angles/lighting = more robust matching.
    embeddings_per_identity: int = 12
    # Head-toward-camera (landmark yaw proxy, not true eye gaze).
    # Hysteresis: easier to *stay* looking than to *enter*, so the flag does not
    # flicker when the nose landmark jitters around the threshold.
    looking_enter: float = 0.42
    looking_exit: float = 0.28
    looking_hold_s: float = 0.2
    looking_release_s: float = 1.4
    # Keep "looking" sticky this long after the last good frontal face, so a
    # missed face frame mid-sentence does not drop the STT gate.
    looking_sticky_s: float = 2.0
    # Back-compat alias used by older code paths.
    looking_threshold: float = 0.42


@dataclass
class AudioConfig:
    enabled: bool = True
    device: int | str | None = None  # None = system default input
    # Silero VAD is trained at 16 kHz and wants exactly 512-sample blocks (32 ms).
    samplerate: int = 16000
    block_size: int = 512
    vad_threshold: float = 0.5
    # Silence this long ends an utterance. Too short and one sentence becomes
    # five segments; too long and two people's turns merge into one.
    min_silence_ms: int = 350
    speech_pad_ms: int = 120  # keep a little either side, speech onsets get clipped
    min_speech_s: float = 0.4  # below this there is not enough voice to identify
    max_segment_s: float = 15.0  # cut monologues so they still produce output
    buffer_s: float = 30.0  # rolling audio kept in memory


@dataclass
class VoiceConfig:
    enabled: bool = True
    model_dir: Path = DATA_DIR / "models" / "ecapa"
    # Below this there is not enough voice for a meaningful embedding at all.
    min_embed_s: float = 0.6
    # Minting a new speaker needs more evidence than recognising a known one,
    # for the same reason it does with faces: a bad enrolment is permanent.
    min_enroll_s: float = 1.5
    min_enroll_quality: float = 0.55
    embeddings_per_identity: int = 12
    # None = take the backend's own thresholds.
    match_threshold: float | None = None
    enroll_threshold: float | None = None


@dataclass
class GestureConfig:
    enabled: bool = True
    # Keypoints below this confidence are treated as unseen rather than trusted.
    min_kp_conf: float = 0.5
    # How far down from the nose towards the shoulders the "face level" line
    # sits. 0.45 lands at about the chin; drop towards 0 to demand the hand be
    # higher and cut hand-resting-on-chin false positives.
    face_level_frac: float = 0.45
    # A hand must stay up this long before it counts as a vote, and stay down
    # this long before the vote is withdrawn. The asymmetry is deliberate:
    # brief occlusion of a wrist should not retract a raised hand.
    hold_s: float = 0.4
    release_s: float = 0.8

    # --- the storyteller's vote sweep: hands clasped, arms out, turning ---
    sweep_enabled: bool = True
    # Hand separation and wrist drop, both in torso lengths. Measured on real
    # footage: clasped runs 0.05-0.40 and arms-down 0.72-1.06, so 0.45 sits in
    # the gap. Torso is the scale because shoulder width collapses through a turn.
    sweep_max_gap: float = 0.45
    sweep_max_drop: float = 0.75
    # Sideways hand offset when pointing square across the camera, in torso
    # lengths. Sets what counts as 90 degrees.
    sweep_reach: float = 0.90
    sweep_min_kp_conf: float = 0.15  # low: pointing away hides both wrists
    sweep_hold_s: float = 0.5
    sweep_release_s: float = 1.2


@dataclass
class TrackConfig:
    # Accumulated quality-weighted evidence needed before a track takes a name
    # away from one it already holds. Each face pass contributes roughly its
    # quality score, so ~3 decent looks.
    votes_to_commit: float = 2.0
    # ...but an unclaimed track will take a name from a single look this similar.
    instant_claim_sim: float = 0.5
    # Votes decay so an old wrong guess does not outweigh what we see now.
    vote_decay: float = 0.92
    # Drop a track this long after its last sighting.
    forget_after_s: float = 3.0
    # Consecutive frames before a track is reported as a real person.
    frames_to_announce: int = 4
    # After losing an identity to a better-supported track, how long before this
    # track may claim it again.
    conflict_block_s: float = 3.0

    # --- face-to-body binding sanity, all as fractions of the body box ---
    # How far down the body a face may sit and still be that body's head. Kept
    # loose because a seated person behind a table is mostly head.
    max_face_depth: float = 0.55
    min_face_body_ratio: float = 0.12
    max_face_body_ratio: float = 0.65


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    # OpenCV native window is opt-in; the Chrome dashboard is the default UI.
    show_window: bool = False
    log_events: bool = True
