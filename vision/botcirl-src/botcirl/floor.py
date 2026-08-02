"""Camera pixels -> floor coordinates.

A person's foot point and the floor they stand on are related by a homography:
one 3x3 matrix maps the image plane onto the ground plane, as long as the camera
does not move. Four known floor points are enough to solve for it.

Once calibrated, "person A is at (2.1m, 0.4m)" is a real statement about the
room instead of a statement about where they happen to land in the frame. That
is what makes distance between two people, and eventually "who is facing whom",
mean anything.

Caveat that matters for this robot: the moment somebody picks it up, the
homography is void. Treat `FloorMap.valid` as something the robot's motion
sensing should be allowed to switch off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


class FloorMap:
    def __init__(self, path: Path | None = None):
        # Resolved at call time, not bound as a default - see IdentityGallery.
        self.path = Path(path) if path is not None else Path(config.FLOOR_HOMOGRAPHY)
        self.H: np.ndarray | None = None
        self.valid = False
        if self.path.exists():
            self.H = np.load(self.path)
            self.valid = True

    @staticmethod
    def _spread(pts: np.ndarray) -> float:
        """How two-dimensional a point set is: 0 = a straight line, 1 = a square.

        Ratio of the smaller to the larger principal axis. cv2.findHomography
        does *not* reject collinear input - it returns a matrix that looks fine
        and maps everything to nonsense - so this has to be checked up front.
        """
        centred = pts - pts.mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        return float(sv[1] / sv[0]) if sv[0] > 1e-9 else 0.0

    @staticmethod
    def solve(image_pts: list[tuple[float, float]], room_pts: list[tuple[float, float]]):
        """Least-squares homography from >=4 image/floor correspondences.

        Raises ValueError rather than returning a plausible-looking matrix when
        the points cannot support a solve. A bad calibration is worse than none:
        it reports confident positions that are quietly wrong.
        """
        if len(image_pts) < 4 or len(image_pts) != len(room_pts):
            raise ValueError("need at least 4 matching image/room points")
        import cv2

        src = np.asarray(image_pts, np.float64)
        dst = np.asarray(room_pts, np.float64)
        for name, pts in (("image", src), ("floor", dst)):
            if FloorMap._spread(pts) < 0.02:
                raise ValueError(
                    f"the {name} points are almost in a straight line - pick four spots "
                    "that form a wide quadrilateral, not a row along one wall"
                )

        H, _ = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=0)
        if H is None:
            raise ValueError("could not fit a homography to those points")

        # Trust but verify: push the inputs back through and demand they land.
        proj = (H @ np.hstack([src, np.ones((len(src), 1))]).T).T
        w = proj[:, 2:3]
        if np.any(np.abs(w) < 1e-9):
            raise ValueError("degenerate homography - one of the points maps to infinity")
        residual = float(np.max(np.linalg.norm(proj[:, :2] / w - dst, axis=1)))
        scale = float(np.max(np.linalg.norm(dst - dst.mean(axis=0), axis=1))) or 1.0
        # Generous, because clicking a floor mark and measuring it with a tape
        # both carry real error. This is here to catch a mistyped or mismatched
        # point, not to grade your handiwork - calibrate_floor.py prints the
        # per-point residuals so you can judge the fit yourself.
        #
        # Note this can only fire with 5+ points: four correspondences are an
        # exact solve, so they reproduce themselves no matter how wrong they are.
        if residual > 0.15 * scale:
            raise ValueError(
                f"the fit does not reproduce your own points (off by {residual:.2f} m) - "
                "check the measurements, or that each click matches the point you typed"
            )
        return H

    def save(self, H: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, H)
        self.H, self.valid = H, True

    def to_room(self, x: float, y: float) -> tuple[float, float] | None:
        """Map an image-space ground point to room metres. None if uncalibrated."""
        if not self.valid or self.H is None:
            return None
        v = self.H @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    # ---------- where the camera is ----------

    @property
    def _camera_file(self) -> Path:
        return self.path.with_name("floor_camera.json")

    def load_camera(self) -> dict | None:
        """Where the camera stands, if it has been recorded."""
        import json

        if not self._camera_file.exists():
            return None
        try:
            return json.loads(self._camera_file.read_text())
        except (ValueError, OSError):
            return None

    def save_camera(self, ground: tuple[float, float], height: float,
                    source: str = "measured") -> None:
        """Record the camera's floor position. `source` is measured|estimated."""
        import json

        self._camera_file.parent.mkdir(parents=True, exist_ok=True)
        self._camera_file.write_text(json.dumps({
            "ground": [float(ground[0]), float(ground[1])],
            "height": float(height),
            "source": source,
        }, indent=2))

    def camera_pose(self, image_size: tuple[int, int]) -> dict | None:
        """Recover where the camera stands on the floor, and how high up it is.

        The homography already encodes this; it just has to be unpacked. With
        the ground plane at Z=0, the room->image homography is G = K[r1 r2 t].
        Assuming square pixels and the principal point at the image centre, the
        only unknown in K is the focal length, and the two facts that r1 and r2
        are orthogonal and the same length each give an estimate of it. From
        there R and t fall out, and the camera centre is C = -R'ᵀt.

        Returns metres: `ground` is the point on the floor directly beneath the
        camera, `height` is how far above it. None when the solve is degenerate.

        TREAT THE RESULT AS A HINT, NOT A MEASUREMENT. Recovering focal length
        from a single homography is badly ill-conditioned: on synthetic data, a
        homography differing from the true one by one part in a *million* gave a
        focal length 15% out and a position 14 cm off. Under realistic click
        noise the position lands within roughly 5-25 cm, which is fine for
        drawing a marker on a map and useless for anything that needs precision.

        `calibrate_floor.py` offers this as a default and lets you type a
        measured position instead - you already have a tape measure out.
        """
        if not self.valid or self.H is None:
            return None
        w, h = image_size
        cx, cy = w / 2.0, h / 2.0

        G = np.linalg.inv(self.H)  # room -> image
        g1, g2, g3 = G[:, 0], G[:, 1], G[:, 2]

        def centred(g):
            return np.array([g[0] - cx * g[2], g[1] - cy * g[2]])

        c1, c2 = centred(g1), centred(g2)

        # f from orthogonality of r1 and r2, and from their equal length.
        estimates = []
        denom = g1[2] * g2[2]
        if abs(denom) > 1e-12:
            f_sq = -float(c1 @ c2) / float(denom)
            if f_sq > 0:
                estimates.append(f_sq)
        denom2 = g2[2] ** 2 - g1[2] ** 2
        if abs(denom2) > 1e-12:
            f_sq = float(c1 @ c1 - c2 @ c2) / float(denom2)
            if f_sq > 0:
                estimates.append(f_sq)
        if not estimates:
            return None
        f = float(np.sqrt(np.mean(estimates)))
        if not np.isfinite(f) or f <= 1.0:
            return None

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        M = np.linalg.inv(K) @ G
        n1, n2 = np.linalg.norm(M[:, 0]), np.linalg.norm(M[:, 1])
        if n1 < 1e-12 or n2 < 1e-12:
            return None
        lam = 2.0 / (n1 + n2)  # average the two scale estimates
        r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam

        # The sign of the homography is arbitrary; pick the solution with the
        # camera above the floor rather than buried beneath it.
        if t[2] < 0:
            r1, r2, t = -r1, -r2, -t

        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        # Nearest true rotation - the columns are only approximately orthonormal.
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt

        C = -R.T @ t
        if not np.all(np.isfinite(C)):
            return None
        return {
            "ground": (float(C[0]), float(C[1])),
            "height": float(C[2]),
            "focal_px": f,
        }

    def visible_ground(self, image_size: tuple[int, int], steps: int = 24
                       ) -> list[tuple[float, float]]:
        """The patch of floor the camera can actually see, as a polygon.

        Walks the image border and maps each point down onto the floor. Points
        above the horizon are dropped: they map to somewhere behind the camera
        or infinitely far away, and drawing them produces a wildly wrong shape
        rather than an obviously broken one.
        """
        if not self.valid or self.H is None:
            return []
        w, h = image_size
        border = []
        for i in range(steps + 1):
            f = i / steps
            border.append((f * w, float(h)))       # bottom edge, nearest the camera
        for i in range(steps + 1):
            f = i / steps
            border.append((float(w), h * (1 - f)))  # right edge going up
        for i in range(steps + 1):
            f = i / steps
            border.append((w * (1 - f), 0.0))       # top edge
        for i in range(steps + 1):
            f = i / steps
            border.append((0.0, h * f))             # left edge coming down

        out = []
        for px, py in border:
            v = self.H @ np.array([px, py, 1.0])
            if v[2] <= 1e-9:
                continue  # at or beyond the horizon
            pt = (float(v[0] / v[2]), float(v[1] / v[2]))
            if abs(pt[0]) > 60 or abs(pt[1]) > 60:
                continue  # effectively at infinity; no room is 60 m across
            out.append(pt)
        return out

    def distance(self, a: tuple[float, float], b: tuple[float, float]) -> float | None:
        """Floor distance in metres between two image-space ground points."""
        ra, rb = self.to_room(*a), self.to_room(*b)
        if ra is None or rb is None:
            return None
        return float(np.hypot(ra[0] - rb[0], ra[1] - rb[1]))
