"""Recovering a clean photo of the empty room from a camera that never moves.

If the camera is fixed, every pixel is *usually* background - people occlude any
given spot only some of the time. So keep a slow running average of each pixel,
updated only when nobody is standing in front of it, and after a minute or two
you have the room as it looks with nobody in it.

The alternative, taking one frame at startup, breaks the moment somebody is
already in the room when you press go. This does not: it just takes longer to
fill in the parts of the room they were standing in.

`coverage()` is the honest part. If somebody stands still in one spot for the
whole session, the pixels behind them were never observed empty and that patch
of the plate is whatever was there at the start. Better to say so than to
present a plate with a person baked into it as if it were the empty room.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class BackgroundPlate:
    def __init__(self, alpha: float = 0.02, dilate_px: int = 24):
        # alpha is per-update, and updates are throttled by the caller. Slow
        # enough that a person pausing briefly does not smear into the plate,
        # fast enough to follow the light changing over an afternoon.
        self.alpha = alpha
        self.dilate_px = dilate_px
        self.plate: np.ndarray | None = None  # float32 accumulator
        self.seen: np.ndarray | None = None  # how many times each pixel was background
        self.updates = 0

    def update(self, frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
        h, w = frame_bgr.shape[:2]
        if self.plate is None or self.plate.shape[:2] != (h, w):
            # Neutral grey, NOT the first frame. Seeding from a frame bakes
            # whoever was already in the room into the "empty room" plate, and
            # any pixel they never move away from keeps them there forever - a
            # motionless ghost that reads as real furniture.
            self.plate = np.full((h, w, 3), 110.0, np.float32)
            self.seen = np.zeros((h, w), np.uint16)

        # Mask out people, generously. Detection boxes clip shoulders and hair,
        # and a few pixels of somebody's outline bleeding into the plate is very
        # visible - a faint ghost standing in the room.
        free = np.ones((h, w), bool)
        for x1, y1, x2, y2 in boxes:
            d = self.dilate_px
            free[max(0, y1 - d):min(h, y2 + d), max(0, x1 - d):min(w, x2 + d)] = False

        # First sight of a pixel takes it outright; grey has no information to
        # preserve, and blending from it would leave the room washed out for
        # minutes. After that, average slowly.
        fresh = free & (self.seen == 0)
        settled = free & (self.seen > 0)
        self.plate[fresh] = frame_bgr[fresh].astype(np.float32)
        a = self.alpha
        self.plate[settled] = ((1 - a) * self.plate[settled]
                               + a * frame_bgr[settled].astype(np.float32))
        self.seen[free] = np.minimum(self.seen[free] + 1, 65535)
        self.updates += 1

    def coverage(self, min_observations: int = 3) -> float:
        """Fraction of the frame genuinely observed without anybody in it."""
        if self.seen is None:
            return 0.0
        return float((self.seen >= min_observations).mean())

    def image(self) -> np.ndarray | None:
        if self.plate is None:
            return None
        return np.clip(self.plate, 0, 255).astype(np.uint8)

    def unobserved_mask(self, min_observations: int = 3) -> np.ndarray | None:
        """Where the plate is not trustworthy, for shading on the dashboard."""
        if self.seen is None:
            return None
        return self.seen < min_observations

    def save(self, path: Path | str, quality: int = 82) -> Path | None:
        img = self.image()
        if img is None:
            return None
        import cv2

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return path
