#!/usr/bin/env python3
"""Work out who voted, from a video of a Blood on the Clocktower vote.

    python analyze_vote.py testvid.mov
    python analyze_vote.py testvid.mov --names Aidan,Mars,Keith,...
    python analyze_vote.py testvid.mov --annotate out.jpg

Players sit still during a vote, so seats are the stable thing to reason about -
not track ids, which fragment every time somebody is occluded. Seats are built
by taking each track's median hip position and merging tracks that sit in the
same place; whoever moves is the storyteller and drops out.

A vote is a *sustained* raise, not any raise. On the reference clip that
separates cleanly: every real voter held for at least 0.6s, and the three
non-voters produced literally zero raised frames.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from botcirl.gestures import SweepTracker, detect_hand_raise, detect_pointing

LEFT_HIP, RIGHT_HIP = 11, 12


@dataclass
class Seat:
    index: int
    x: float
    y: float
    track_ids: list[int] = field(default_factory=list)
    samples: list[tuple[int, bool]] = field(default_factory=list)  # (frame, raised)
    name: str | None = None

    def runs(self, max_gap_frames: int = 4) -> list[tuple[int, int]]:
        """Contiguous stretches where the hand was up."""
        raised = sorted(f for f, r in self.samples if r)
        if not raised:
            return []
        out, start, prev = [], raised[0], raised[0]
        for f in raised[1:]:
            if f - prev > max_gap_frames:
                out.append((start, prev))
                start = f
            prev = f
        out.append((start, prev))
        return out

    def longest_hold(self, fps: float) -> float:
        return max(((b - a) / fps for a, b in self.runs()), default=0.0)


def collect_poses(path: str, model_name: str, imgsz: int, stride: int,
                  start: int, end: int | None) -> tuple[list[dict], float]:
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= start and (end is None or i <= end) and i % stride == 0:
            r = model.track(frame, classes=[0], imgsz=imgsz, conf=0.25,
                            persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            if r.boxes is not None and r.boxes.id is not None:
                frames.append({
                    "frame": i,
                    "ids": r.boxes.id.int().cpu().numpy(),
                    "box": r.boxes.xyxy.cpu().numpy(),
                    "kp": r.keypoints.xy.cpu().numpy(),
                    "kc": r.keypoints.conf.cpu().numpy(),
                })
        i += 1
    cap.release()
    return frames, fps


def build_seats(records: list[dict], move_threshold: float = 120.0,
                merge_x: float = 95.0, merge_y: float = 140.0) -> list[Seat]:
    """Group detections into seats, dropping anyone who walks about.

    The hip midpoint is the anchor because it barely moves while seated, even
    when the arms are waving - unlike the box centre, which jumps every time a
    hand goes up.
    """
    per_track: dict[int, list[tuple[float, float]]] = {}
    for rec in records:
        for j, tid in enumerate(rec["ids"]):
            kc = rec["kc"][j]
            if min(kc[LEFT_HIP], kc[RIGHT_HIP]) < 0.5:
                continue
            kp = rec["kp"][j]
            hip = ((kp[LEFT_HIP][0] + kp[RIGHT_HIP][0]) / 2,
                   (kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) / 2)
            per_track.setdefault(int(tid), []).append(hip)

    stable = []
    for tid, pts in per_track.items():
        if len(pts) < 12:
            continue  # too fleeting to be a seated player
        xs = np.array([p[0] for p in pts])
        if xs.std() > move_threshold:
            continue  # this one walks - the storyteller
        stable.append((tid, float(np.median(xs)),
                       float(np.median([p[1] for p in pts])), len(pts)))

    stable.sort(key=lambda s: s[1])
    seats: list[Seat] = []
    for tid, x, y, _n in stable:
        # One person often picks up several track ids across a clip; anything
        # sitting in the same spot is the same seat.
        if seats and abs(x - seats[-1].x) < merge_x and abs(y - seats[-1].y) < merge_y:
            seats[-1].track_ids.append(tid)
            seats[-1].x = (seats[-1].x + x) / 2
        else:
            seats.append(Seat(index=len(seats) + 1, x=x, y=y, track_ids=[tid]))
    for i, s in enumerate(seats, start=1):
        s.index = i
    return seats


def score(seats: list[Seat], records: list[dict], min_kp_conf: float,
          face_level_frac: float) -> None:
    lookup = {tid: s for s in seats for tid in s.track_ids}
    for rec in records:
        for j, tid in enumerate(rec["ids"]):
            seat = lookup.get(int(tid))
            if seat is None:
                continue
            verdict = detect_hand_raise(rec["kp"][j], rec["kc"][j],
                                        min_conf=min_kp_conf,
                                        face_level_frac=face_level_frac)
            seat.samples.append((int(rec["frame"]), verdict.raised))


def storyteller_sweep(records: list[dict], seats: list[Seat], fps: float
                      ) -> list[tuple[float, float]]:
    """Track the storyteller's pointing bearing over the vote.

    Whoever is not in a seat is the storyteller. `detect_pointing` accepts both
    conventions - hands clasped and turning, or one arm sweeping round - because
    both are in use and both mean the same thing.
    """
    seated = {tid for s in seats for tid in s.track_ids}
    tracker = SweepTracker(reset_after_s=0.6)
    out = []
    for rec in records:
        for j, tid in enumerate(rec["ids"]):
            if int(tid) in seated:
                continue
            p = detect_pointing(rec["kp"][j], rec["kc"][j])
            t = rec["frame"] / fps
            b = tracker.update(p if p.detected else None, t)
            if p.detected and b is not None:
                out.append((t, b))
            break
    return out


def pass_times(sweep: list[tuple[float, float]], order: list[int]
               ) -> dict[int, float]:
    """When the sweep reached each seat.

    The vote runs clockwise and never doubles back, so the bearing is forced
    monotonic before inverting it - that discards the jitter without discarding
    the shape.

    Seats are then spread evenly in ANGLE across the sweep, which assumes the
    chairs are evenly spaced round the circle. That assumption is the weak point
    and it is only needed because the seats cannot be located geometrically from
    one uncalibrated camera: they project to a nearly straight line, so the
    circle they sit on is not recoverable. Calibrate the floor and each seat's
    true bearing is known outright, no assumption required.
    """
    if len(sweep) < 4:
        return {}
    ts = np.array([t for t, _ in sweep])
    bs = np.minimum.accumulate(np.array([b for _, b in sweep]))
    angles = np.linspace(bs[0], bs[-1], len(order))
    return {seat: float(ts[int(np.argmin(np.abs(bs - a)))])
            for seat, a in zip(order, angles)}


def annotate(path: str, records: list[dict], seats: list[Seat], out: str,
             min_kp_conf: float) -> None:
    lookup = {tid: s for s in seats for tid in s.track_ids}
    best = max(records, key=lambda r: sum(
        detect_hand_raise(r["kp"][j], r["kc"][j], min_conf=min_kp_conf).raised
        for j in range(len(r["ids"]))))
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    links = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12)]
    for j, tid in enumerate(best["ids"]):
        seat = lookup.get(int(tid))
        kp, kc = best["kp"][j], best["kc"][j]
        col = (90, 220, 255) if seat else (130, 130, 130)
        for a, b in links:
            if kc[a] > 0.4 and kc[b] > 0.4:
                cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), col, 4)
        if seat is None:
            continue
        v = detect_hand_raise(kp, kc, min_conf=min_kp_conf)
        hip = ((kp[LEFT_HIP] + kp[RIGHT_HIP]) / 2).astype(int)
        label = seat.name or f"seat {seat.index}"
        cv2.putText(frame, label, (hip[0] - 40, hip[1] + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        if v.raised:
            cv2.putText(frame, "HAND UP", (hip[0] - 40, hip[1] + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
    cv2.imwrite(out, frame)
    print(f"annotated frame {best['frame']} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--stride", type=int, default=2, help="analyse every Nth frame")
    ap.add_argument("--start", type=int, default=0, help="first frame of the vote")
    ap.add_argument("--end", type=int, default=None, help="last frame of the vote")
    ap.add_argument("--hold", type=float, default=0.5,
                    help="seconds a hand must stay up to count as a vote")
    ap.add_argument("--min-kp-conf", type=float, default=0.4)
    ap.add_argument("--face-level", type=float, default=0.45)
    ap.add_argument("--names", default=None, help="comma-separated, left to right")
    ap.add_argument("--rule", default="pass", choices=["pass", "any"],
                    help="pass: hand up when the storyteller reached them (correct); "
                         "any: hand up at any point (wrong - counts pre-vote hands)")
    ap.add_argument("--order", default=None,
                    help="seat numbers in voting order, clockwise, nominee last")
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="seconds either side of the pass to look for a raised hand")
    ap.add_argument("--annotate", default=None, help="write a labelled frame here")
    args = ap.parse_args()

    print(f"reading {args.video} ...")
    records, fps = collect_poses(args.video, args.model, args.imgsz,
                                 args.stride, args.start, args.end)
    if not records:
        sys.exit("no people detected")
    print(f"{len(records)} frames analysed at {fps:.0f} fps")

    seats = build_seats(records)
    score(seats, records, args.min_kp_conf, args.face_level)

    names = [n.strip() for n in args.names.split(",")] if args.names else []
    for i, s in enumerate(seats):
        if i < len(names):
            s.name = names[i]
    if names and len(names) != len(seats):
        print(f"warning: {len(names)} names given but {len(seats)} seats found - "
              "the mapping below is probably misaligned")

    voters = []
    if args.rule == "any":
        print(f"\n{len(seats)} seats. Counting any sustained raise >= {args.hold}s.")
        print("WARNING: hands raised before the vote began are counted too.\n")
        print(f"{'seat':>4} {'who':<12} {'x':>6} {'longest':>9}  verdict")
        for s in seats:
            hold = s.longest_hold(fps)
            if hold >= args.hold:
                voters.append(s)
            print(f"{s.index:>4} {(s.name or ''):<12} {s.x:6.0f} {hold:8.2f}s  "
                  f"{'VOTED' if hold >= args.hold else '-'}")
    else:
        sweep = storyteller_sweep(records, seats, fps)
        if not sweep:
            sys.exit("no storyteller sweep found - try --rule any, or check the clip "
                     "covers the vote")
        print(f"\nstoryteller swept {sweep[0][1]:+.0f} to {sweep[-1][1]:+.0f} degrees "
              f"over {sweep[0][0]:.2f}-{sweep[-1][0]:.2f}s")
        order = ([int(v) for v in args.order.split(",")] if args.order
                 else list(range(1, len(seats) + 1)))
        times = pass_times(sweep, order)
        print(f"\n{len(seats)} seats. A vote is a hand up when the storyteller "
              f"reached them (+/-{args.tolerance}s).\n")
        print(f"{'seat':>4} {'who':<12} {'passed':>8} {'hand up':>8}  verdict")
        for seat in order:
            s = seats[seat - 1]
            tp = times.get(seat)
            near = [r for t, r in ((f / fps, r) for f, r in s.samples)
                    if abs(t - tp) <= args.tolerance] if tp is not None else []
            voted = any(near)
            if voted:
                voters.append(s)
            print(f"{seat:>4} {(s.name or ''):<12} "
                  f"{(f'{tp:.2f}s' if tp is not None else '-'):>8} "
                  f"{('yes' if voted else 'no'):>8}  {'VOTED' if voted else '-'}")

    print(f"\n{len(voters)} votes: " +
          ", ".join(v.name or f"seat {v.index}" for v in voters))

    if args.annotate:
        annotate(args.video, records, seats, args.annotate, args.min_kp_conf)


if __name__ == "__main__":
    main()
