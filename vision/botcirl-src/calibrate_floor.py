#!/usr/bin/env python3
"""Teach the camera where the floor is, so positions come out in metres.

    python calibrate_floor.py                  # grab a frame from the webcam
    python calibrate_floor.py --source room.jpg

Pick spots on the floor that you can measure - the corners of a rug, table legs,
tape on the carpet. Click each one in the image, then type its position in
metres. Use any origin and axes you like, as long as you are consistent; the
camera itself is a sensible (0, 0).

Four points is the minimum, but give it five or six. Four is an exact solve: it
will reproduce your own numbers perfectly even if one of them is wrong, so there
is no way to tell a good calibration from a typo. The extra points are what make
the accuracy readout at the end mean something.

Spread them out and do not put them in a line, or the solve is degenerate.
Points near the far wall matter most: that is where a few pixels of click error
turns into a metre of position error.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from botcirl.floor import FloorMap

clicks: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))


def grab_frame(source: str) -> np.ndarray:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            sys.exit("Could not open the camera. Grant camera access to your terminal app.")
        for _ in range(10):  # let exposure settle
            ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("Camera opened but returned no frame.")
        return frame
    frame = cv2.imread(source) if not source.endswith((".mp4", ".mov")) else None
    if frame is None:
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Could not read a frame from {source!r}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, image, or video")
    args = ap.parse_args()

    frame = grab_frame(args.source)
    win = "click floor points (u = undo, enter = done, esc = quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(canvas, (x, y), 6, (60, 220, 255), -1)
            cv2.putText(canvas, str(i + 1), (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 255), 2, cv2.LINE_AA)
        if len(clicks) >= 2:
            cv2.polylines(canvas, [np.array(clicks, np.int32)], len(clicks) > 3,
                          (60, 220, 255), 1, cv2.LINE_AA)
        msg = f"{len(clicks)} floor points (4 minimum, 5+ recommended)"
        cv2.putText(canvas, msg, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, msg, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            sys.exit("cancelled")
        if key == ord("u") and clicks:
            clicks.pop()
        if key in (13, 10) and len(clicks) >= 4:
            break
    cv2.destroyAllWindows()

    print("\nNow give each point's real position on the floor, in metres.")
    print("x = right, y = away from the camera, origin wherever you like.\n")
    room: list[tuple[float, float]] = []
    for i, (px, py) in enumerate(clicks, start=1):
        while True:
            raw = input(f"  point {i} at pixel ({px},{py})  ->  x y : ").replace(",", " ").split()
            try:
                room.append((float(raw[0]), float(raw[1])))
                break
            except (ValueError, IndexError):
                print("    please type two numbers, e.g.  1.5 2.0")

    H = FloorMap.solve([(float(x), float(y)) for x, y in clicks], room)
    fm = FloorMap()
    fm.save(H)
    print(f"\nsaved homography to {fm.path}")

    print("\ncheck - your calibration points mapped back through the solve:")
    worst = 0.0
    for (px, py), (rx, ry) in zip(clicks, room):
        got = fm.to_room(px, py)
        err = float(np.hypot(got[0] - rx, got[1] - ry))
        worst = max(worst, err)
        print(f"  ({px:4d},{py:4d}) -> ({got[0]:6.2f}, {got[1]:6.2f}) m   want ({rx:5.2f}, {ry:5.2f})   off by {err*100:4.1f} cm")
    # ---- where is the camera itself ----
    h_img, w_img = frame.shape[:2]
    est = fm.camera_pose((w_img, h_img))
    print("\nwhere is the camera, in the same coordinates?")
    if est:
        print(f"  estimated from the calibration: "
              f"({est['ground'][0]:.2f}, {est['ground'][1]:.2f}) at {est['height']:.2f} m high")
        print("  that estimate is rough - recovering focal length from one homography")
        print("  is ill-conditioned, so it is typically 5-25 cm out. Measuring beats it.")
    else:
        print("  (could not estimate it from the calibration)")

    default = est["ground"] if est else (0.0, 0.0)
    default_h = est["height"] if est else 1.2
    raw = input(f"  camera x y  [enter for {default[0]:.2f} {default[1]:.2f}]: "
                ).replace(",", " ").split()
    try:
        ground = (float(raw[0]), float(raw[1]))
        source = "measured"
    except (ValueError, IndexError):
        ground, source = default, ("estimated" if est else "assumed")

    raw = input(f"  height above the floor in metres [enter for {default_h:.2f}]: ").strip()
    try:
        height = float(raw)
        if source != "measured":
            source = "partly measured"
    except ValueError:
        height = default_h

    fm.save_camera(ground, height, source)
    print(f"  camera recorded at ({ground[0]:.2f}, {ground[1]:.2f}), "
          f"{height:.2f} m high [{source}]")

    seen = fm.visible_ground((w_img, h_img))
    if seen:
        ys = [p[1] for p in seen]
        print(f"  visible floor reaches from {min(ys):.1f} m to {max(ys):.1f} m away")

    if len(clicks) == 4:
        print("\nnote: with exactly 4 points those residuals are meaningless - the solve "
              "is exact, so it reproduces your numbers even if one is wrong.\n"
              "re-run with 5 or more points if you want the accuracy readout to mean "
              "anything.")
    elif worst > 0.15:
        print(f"\nworst point is off by {worst*100:.0f} cm - re-measure or re-click, "
              "the fit is only as good as the points you gave it.")
    else:
        print(f"\nlooks good - worst point off by {worst*100:.1f} cm.")


if __name__ == "__main__":
    main()
