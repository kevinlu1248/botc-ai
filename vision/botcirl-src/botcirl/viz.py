"""Debug overlay. Purely for the laptop stage - the robot will not draw any of this."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .pipeline import PersonTrack, Scene

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(key: str | int) -> tuple[int, int, int]:
    """Stable, well-separated colour per identity (golden-ratio hue hopping)."""
    h = (abs(hash(str(key))) % 997) * 0.618033988749895 % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _label_box(img, text: str, org: tuple[int, int], color, scale=0.55, thickness=1) -> None:
    x, y = org
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 10, y + base), color, -1)
    cv2.putText(img, text, (x + 5, y - 2), FONT, scale, (16, 16, 16), thickness, cv2.LINE_AA)


VOTE_COLOR = (60, 240, 255)  # amber, deliberately unlike any identity colour


def draw_person(img, trk: PersonTrack, label: str, index: int, known: bool) -> None:
    x1, y1, x2, y2 = trk.bbox
    color = color_for(trk.pid or f"t{trk.track_id}")
    thickness = 2 if known else 1

    if trk.hand_raised:
        # A raised hand has to be readable across the room at a glance, so it
        # gets its own colour and a heavier box rather than a subtle marker.
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), VOTE_COLOR, 3)
        _draw_raised_arms(img, trk)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    tag = f"[{index}] {label}"
    if not known:
        tag += " ?"
    if trk.looking:
        tag += "  LOOKING"
    if trk.hand_raised:
        tag += f"  VOTING {trk.raised_for:.0f}s"
    if trk.sweeping and trk.sweep_bearing is not None:
        tag += f"  SWEEP {trk.sweep_bearing:+.0f}\u00b0"
    # Tuck the label inside the box when the person is cropped by the top edge,
    # otherwise every tall track piles its name into the same strip of sky.
    label_y = y1 - 6 if y1 > 26 else min(y1 + 24, y2 - 4)
    _label_box(img, tag, (x1, label_y), VOTE_COLOR if trk.hand_raised else color)

    # Ground point + recent path.
    fx, fy = trk.foot
    cv2.circle(img, (int(fx), int(fy)), 4, color, -1)
    if len(trk.trail) > 1:
        pts = np.array([[int(x), int(y)] for _, x, y in trk.trail[-40:]], np.int32)
        cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)

    if trk.sweeping and trk.sweep_bearing is not None:
        _draw_sweep_dial(img, trk)

    if trk.face is not None:
        fx1, fy1, fx2, fy2 = trk.face.bbox
        # Bright face box when looking at the camera; dimmer when turned away.
        face_color = (120, 255, 120) if trk.looking else color
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), face_color, 2 if trk.looking else 1)
        q = trk.face.quality()
        look = trk.looking_score
        cv2.putText(
            img, f"q{q:.2f} look{look:.2f}", (fx1, fy2 + 14), FONT, 0.42,
            face_color, 1, cv2.LINE_AA,
        )


def _draw_raised_arms(img, trk: PersonTrack) -> None:
    """Show the arm the call was made on, and the face-level line it crossed.

    Without this a wrong verdict is a mystery; with it you can see immediately
    whether the pose was misread or the threshold is simply set too low.
    """
    from .gestures import (LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST,
                           detect_hand_raise)

    if trk.kps is None or trk.kp_conf is None:
        return
    verdict = detect_hand_raise(trk.kps, trk.kp_conf)
    if verdict.threshold_y is not None:
        x1, _, x2, _ = trk.bbox
        y = int(verdict.threshold_y)
        cv2.line(img, (x1, y), (x2, y), VOTE_COLOR, 1, cv2.LINE_AA)

    for side, elbow, wrist in (("left", LEFT_ELBOW, LEFT_WRIST),
                               ("right", RIGHT_ELBOW, RIGHT_WRIST)):
        if side not in trk.raise_sides:
            continue
        w = trk.kps[wrist]
        if trk.kp_conf[elbow] >= 0.5:
            e = trk.kps[elbow]
            cv2.line(img, (int(e[0]), int(e[1])), (int(w[0]), int(w[1])),
                     VOTE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(img, (int(w[0]), int(w[1])), 7, VOTE_COLOR, -1)


SWEEP_COLOR = (255, 150, 90)


def _draw_sweep_dial(img, trk: PersonTrack) -> None:
    """A compass over the storyteller showing where they are pointing.

    Drawn as a top-down dial rather than an arrow in the image, because the
    bearing is a direction in the room - the camera sits at the bottom of the
    dial, so "up" is away from it.
    """
    x1, y1, x2, _ = trk.bbox
    cx, cy, r = (x1 + x2) // 2, max(y1 - 46, 40), 26
    cv2.circle(img, (cx, cy), r, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + r), 3, (200, 200, 200), -1)  # the camera

    a = np.radians(trk.sweep_bearing)
    # bearing 0 points at the camera, which is downward on this dial
    tip = (int(cx + np.sin(a) * r), int(cy + np.cos(a) * r))
    cv2.arrowedLine(img, (cx, cy), tip, SWEEP_COLOR, 2, cv2.LINE_AA, tipLength=0.35)


def draw_orphan_faces(img, scene: Scene) -> None:
    """Faces that no body box claimed - usually a detector disagreement."""
    claimed = {id(t.face) for t in scene.people if t.face is not None}
    for face in scene.faces:
        if id(face) in claimed:
            continue
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)


def draw_minimap(img, scene: Scene, pipeline, size: int = 190, extent_m: float = 5.0) -> None:
    """Top-down sketch of where people are.

    Once the floor is calibrated this is a real plan view in metres. Before
    that it falls back to raw image space, where x is position across the view
    and y is how far down the frame the feet are - a crude stand-in for
    distance. The panel says which one you are looking at, because the two are
    easy to confuse and only one of them means anything.
    """
    h, w = img.shape[:2]
    pad = 12
    x0, y0 = w - size - pad, pad
    panel = img[y0 : y0 + size, x0 : x0 + size]
    panel[:] = (panel * 0.35).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (90, 90, 90), 1)

    calibrated = pipeline.floor.valid
    title = f"room +/-{extent_m:.0f}m" if calibrated else "room (uncalibrated)"
    cv2.putText(img, title, (x0 + 6, y0 + 16), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Camera sits at the bottom-centre of the sketch.
    cam = (x0 + size // 2, y0 + size - 8)
    cv2.circle(img, cam, 4, (255, 255, 255), -1)

    if calibrated:
        for ring in (1, 2, 3, 4):  # metre rings, so distance is readable at a glance
            r = int(ring / extent_m * (size - 16) / 2)
            if r < size // 2:
                cv2.circle(img, cam, r, (70, 70, 70), 1)

    def place(trk) -> tuple[int, int]:
        if calibrated and trk.room is not None:
            rx, ry = trk.room
            px = int(cam[0] + rx / extent_m * (size - 16) / 2)
            py = int(cam[1] - ry / extent_m * (size - 16))
        else:
            fx, fy = trk.foot
            px = int(x0 + (fx / w) * size)
            py = int(y0 + size - (fy / h) * size * 0.9)
        return (int(np.clip(px, x0 + 3, x0 + size - 3)), int(np.clip(py, y0 + 3, y0 + size - 3)))

    # Draw plausible conversation pairs before the dots so lines sit underneath.
    # Only the closest few: in a crowd every pair is "within 1.8m" and the panel
    # turns into a mesh that says nothing.
    if calibrated:
        for a, b, dist in pipeline.pairs_within(1.8)[:4]:
            pa, pb = place(a), place(b)
            cv2.line(img, pa, pb, (90, 200, 250), 1, cv2.LINE_AA)
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            cv2.putText(img, f"{dist:.1f}", mid, FONT, 0.32, (90, 200, 250), 1, cv2.LINE_AA)

    for trk in scene.people:
        px, py = place(trk)
        color = color_for(trk.pid or f"t{trk.track_id}")
        cv2.circle(img, (px, py), 5, color, -1)
        if not calibrated:
            cv2.line(img, cam, (px, py), color, 1, cv2.LINE_AA)


def draw_hud(img, scene: Scene, pipeline, mode: str, buffer: str) -> None:
    """Status strip along the bottom.

    Deliberately not top-left: that is where labels for anyone standing close to
    the camera end up, and the two were unreadable on top of each other.
    """
    h, w = img.shape[:2]
    known = sum(1 for t in scene.people if t.pid)
    strip = 52
    band = img[h - strip : h, 0:w]
    band[:] = (band * 0.25).astype(np.uint8)

    if mode == "naming":
        prompt = f"name: {buffer}_"
        cv2.putText(img, prompt, (14, h - 18), FONT, 0.7, (90, 230, 160), 2, cv2.LINE_AA)
        cv2.putText(img, "enter = save   esc = cancel", (w - 300, h - 20), FONT, 0.5,
                    (170, 170, 170), 1, cv2.LINE_AA)
        return

    floor = "floor: metres" if pipeline.floor.valid else "floor: uncalibrated"
    looking_n = sum(1 for t in scene.people if t.looking)
    left = (
        f"{scene.fps:4.1f} fps   people {len(scene.people)} ({known} named)"
        f"   looking {looking_n}   gallery {len(pipeline.gallery.identities)}   {floor}"
    )
    cv2.putText(img, left, (14, h - 30), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

    voting = pipeline.voting()
    if voting:
        names = ", ".join(pipeline.label_for(t) for t in voting)
        tally = f"VOTING ({len(voting)}): {names}"
        (tw, _), _ = cv2.getTextSize(tally, FONT, 0.55, 2)
        cv2.putText(img, tally, (w - tw - 16, h - 30), FONT, 0.55, VOTE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(img, "[1-9] name person    [s] save    [q] quit", (14, h - 11), FONT, 0.45,
                (150, 150, 150), 1, cv2.LINE_AA)


SPEAK_COLOR = (120, 255, 120)


def draw_speaking(img, scene: Scene, pipeline, speech) -> None:
    """Mark whoever the last utterance was attributed to.

    Shown with its basis, because "only person visible" and "learned voice-face
    link" are very different claims and the difference matters when you are
    deciding whether to trust it.
    """
    if speech is None or speech.speaking is None:
        return
    s = speech.speaking
    for trk in scene.people:
        if trk.pid and trk.pid == s.pid:
            x1, y1, x2, y2 = trk.bbox
            cv2.rectangle(img, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), SPEAK_COLOR, 2)
            _label_box(img, "SPEAKING", (x1, min(y2 + 24, img.shape[0] - 4)), SPEAK_COLOR)

    h, w = img.shape[:2]
    text = f"heard: {s.label}"
    detail = s.basis if s.pid else f"{s.basis} - not attributed"
    cv2.putText(img, text, (14, h - 76), FONT, 0.6, SPEAK_COLOR, 2, cv2.LINE_AA)
    cv2.putText(img, detail, (14, h - 58), FONT, 0.42, (170, 200, 170), 1, cv2.LINE_AA)


def render(img, scene: Scene, pipeline, mode: str = "idle", buffer: str = "",
           speech=None) -> np.ndarray:
    ordered = sorted(scene.people, key=lambda t: t.bbox[0])
    for i, trk in enumerate(ordered, start=1):
        draw_person(img, trk, pipeline.label_for(trk), i, known=trk.pid is not None)
    draw_orphan_faces(img, scene)
    draw_speaking(img, scene, pipeline, speech)
    draw_minimap(img, scene, pipeline)
    draw_hud(img, scene, pipeline, mode, buffer)
    return img
