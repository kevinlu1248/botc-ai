"""Face detection + embedding.

Two interchangeable backends behind one interface:

  insightface (buffalo_l)  - SCRFD detector + ArcFace r50 embeddings. Best
                             recognition accuracy; needs the `insightface` pip
                             package (which compiles a Cython extension).
  opencv (yunet + sface)   - ships with opencv-python, models are two small
                             ONNX downloads. Noticeably weaker than ArcFace but
                             fast and dependency-free.

Both return L2-normalised embeddings, so `identity.py` does not care which one
is running: cosine similarity is just a dot product either way.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR, FaceConfig

MODEL_DIR = DATA_DIR / "models"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


@dataclass
class Face:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    score: float
    embedding: np.ndarray  # L2-normalised, float32
    kps: np.ndarray | None = None  # (5, 2) landmarks when the backend has them

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def frontal(self) -> float:
        """0..1 how much the head is turned toward the camera.

        Uses the 5 face landmarks (eyes + nose). A frontal face has the nose
        near the midpoint of the eyes; a profile pushes it off to one side.
        This is head orientation, not true eye gaze - looking at the screen
        vs the lens with a still head will still read as "looking".
        Returns 0 when landmarks are missing (cannot tell).
        """
        if self.kps is None or len(self.kps) < 5:
            return 0.0
        # Landmarks are [left eye, right eye, nose, left mouth, right mouth].
        left_eye, right_eye, nose = self.kps[0], self.kps[1], self.kps[2]
        eye_span = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
        eye_mid = (left_eye + right_eye) / 2.0
        # A turned head pushes the nose away from the midpoint of the eyes.
        yaw = abs(float(nose[0] - eye_mid[0])) / eye_span
        return float(np.clip(1.0 - (yaw - 0.15) / 0.45, 0.0, 1.0))

    def quality(self) -> float:
        """Rough 0..1 "is this face worth enrolling" score.

        Big, confident, and roughly frontal wins. Enrolling a blurry profile as a
        brand new person is the main way a gallery goes bad, so we gate on this.
        """
        # Saturates around 110px, which is roughly a face at conversational
        # distance on a 720p webcam. Below ~45px there is not enough detail for
        # ArcFace to be worth trusting with a new identity.
        size = min(self.width, self.height)
        size_score = float(np.clip((size - 45) / 65.0, 0.0, 1.0))
        return float(size_score * self.frontal() * np.clip(self.score, 0.0, 1.0))


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"[faces] downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


class InsightFaceBackend:
    name = "insightface"
    embedding_dim = 512
    # Measured on a 6-person group photo: different people peaked at 0.21 cosine,
    # the same face across lighting/scale/rotation held 0.98. Wide empty gap.
    default_match_threshold = 0.38
    default_enroll_threshold = 0.28

    def __init__(self, cfg: FaceConfig):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        self.cfg = cfg
        providers = ["CPUExecutionProvider"]
        if cfg.use_coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            # Roughly 2.4x on Apple silicon, which is the difference between
            # "sluggish" and "usable" for the face pass.
            providers.insert(0, "CoreMLExecutionProvider")
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=(cfg.det_size, cfg.det_size))
        self.providers = providers

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        out = []
        for f in self.app.get(frame_bgr):
            if f.det_score < self.cfg.det_conf:
                continue
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            if min(x2 - x1, y2 - y1) < self.cfg.min_face_px:
                continue
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            out.append(
                Face(
                    bbox=(x1, y1, x2, y2),
                    score=float(f.det_score),
                    embedding=emb,
                    kps=np.asarray(f.kps, dtype=np.float32) if f.kps is not None else None,
                )
            )
        return out


class OpenCVBackend:
    name = "opencv-yunet-sface"
    embedding_dim = 128
    # SFace separates people far less cleanly than ArcFace: on the same group
    # photo, different people reached 0.31 cosine. OpenCV's own recommended
    # same/different boundary is 0.363, so these sit deliberately higher than
    # the insightface numbers - reusing those would merge strangers.
    default_match_threshold = 0.42
    default_enroll_threshold = 0.32

    def __init__(self, cfg: FaceConfig):
        self.cfg = cfg
        det_path = _download(YUNET_URL, MODEL_DIR / "face_detection_yunet_2023mar.onnx")
        rec_path = _download(SFACE_URL, MODEL_DIR / "face_recognition_sface_2021dec.onnx")
        self.detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (cfg.det_size, cfg.det_size), cfg.det_conf, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, raw = self.detector.detect(frame_bgr)
        if raw is None:
            return []
        out = []
        for row in raw:
            x, y, fw, fh = (int(v) for v in row[:4])
            score = float(row[-1])
            if min(fw, fh) < self.cfg.min_face_px:
                continue
            aligned = self.recognizer.alignCrop(frame_bgr, row)
            feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
            feat /= np.linalg.norm(feat) + 1e-9
            kps = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            out.append(
                Face(
                    bbox=(x, y, x + fw, y + fh),
                    score=score,
                    embedding=feat,
                    kps=kps,
                )
            )
        return out


def build_face_backend(cfg: FaceConfig, prefer: str = "auto"):
    """Pick a backend. `prefer` is one of auto | insightface | opencv."""
    if prefer in ("auto", "insightface"):
        try:
            backend = InsightFaceBackend(cfg)
            print(
                f"[faces] backend: {backend.name} (buffalo_l / ArcFace) "
                f"via {backend.providers[0]}"
            )
            return backend
        except Exception as exc:  # noqa: BLE001 - any import/model failure falls back
            if prefer == "insightface":
                raise
            print(f"[faces] insightface unavailable ({exc.__class__.__name__}: {exc})")
    backend = OpenCVBackend(cfg)
    print(f"[faces] backend: {backend.name}")
    return backend
