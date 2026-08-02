# botcirl — room perception

Laptop prototype for the room robot. It answers **who is in the room, where they
are, who is who, who has a hand up, and who is speaking** — from one webcam and
one microphone.

The honest state of each piece:

| | status |
|---|---|
| people, tracking, positions | working, verified on live camera |
| face identity, re-identification | working, thresholds measured |
| raised hands / voting | logic verified against synthetic poses; **not yet confirmed on a real arm** |
| speech detection | working, verified live |
| speaker identity | working; **thresholds are a guess until `calibrate_voice.py` is run** |
| voice → face binding | logic verified; needs real multi-person use to build links |
| dashboard: map, timeline, scrub, playback | working, verified live |
| direction of arrival | not possible on this laptop — needs a mic array |

## Run it

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_vision.py                  # webcam
.venv/bin/python run_vision.py --source clip.mp4
.venv/bin/python run_vision.py --headless       # event log only, no window
```

First run downloads ~290 MB of weights (YOLO11n, then insightface `buffalo_l`).

**macOS camera permission**: a terminal cannot open the webcam until you grant
it. *System Settings → Privacy & Security → Camera*, enable your terminal app,
then fully quit and reopen it — restarting just the shell is not enough. Until
then you get `not authorized to capture video`.

While the window is focused:

| key | |
|---|---|
| `1`–`9` | name the Nth person (left to right), type, `enter` |
| `s` | save the gallery |
| `q` | quit |

## What it does

```
frame ──┬─→ YOLO11n + ByteTrack ──→ person boxes with track ids
        └─→ SCRFD ──→ face boxes ──→ ArcFace ──→ 512-d embedding
                                          │
                          bind face to the body box containing it
                                          │
                        cosine-match against the identity gallery
                                          │
                    known → vote for that name    new → enrol as "Person N"
```

**Why both a body tracker and a face gallery.** They fail in opposite
directions. The tracker is smooth frame to frame and works on a back turned to
the camera or a torso behind a table — but its ids break on every occlusion, so
the same person becomes track 2, then 13, then 27. The gallery cannot follow
motion at all, but it is durable across minutes, rooms, and restarts. Bound
together, a track keeps its name through a head turn, and somebody who walks
out and comes back reclaims the name they had an hour ago.

You can see that happening in the event log — track 2 leaves, track 13 arrives
and is recognised as the same person:

```
person_left      track_id=2   pid=p1  label=Person 1  duration_s=1.78
person_appeared  track_id=13
identity_assigned track_id=13 pid=p1  label=Person 1  similarity=0.85
```

Above-the-waist people are handled: YOLO's person class is trained on plenty of
occluded torsos, and nothing in the pipeline assumes a full body is visible.

### Deciding somebody is new

The risk is phantom people — a blurry profile becomes "Person 7" forever. Three
gates, all of which must pass:

- the face must not match anything in the gallery (cosine < `enroll_threshold`)
- `quality()` ≥ 0.55 — a blend of face size, detector confidence, and how
  frontal the head is, judged from the eye/nose landmarks
- it has to hold for 3 consecutive face passes

Measured on a test group photo: frontal faces score 0.80–0.82, a hard profile
scores 0.33 and is correctly refused rather than enrolled.

Thresholds live in `botcirl/config.py`, but leave them as `None` and each
backend supplies its own. They are not interchangeable: on the same test photo
different people reached **0.21** cosine similarity under ArcFace and **0.31**
under SFace, so ArcFace's 0.38 match threshold would merge strangers if reused
for SFace. Same face across lighting, scale and rotation changes: **0.98**.

An unclaimed track takes a name on one confident look (`instant_claim_sim`);
*changing* a name a track already holds needs accumulated evidence
(`votes_to_commit`). That asymmetry is deliberate — fast to identify, slow to
relabel somebody mid-conversation.

### Binding faces to bodies

The subtle failure here is not detection, it is attribution. In a crowd a face
sits inside three or four person boxes at once, and binding it to the tightest
one hands somebody's face — and therefore their name — to whoever is standing in
front of them. Two extra constraints, both just anatomy, remove most of it:

- the face must be in the upper `max_face_depth` of the body box
- the face must be a plausible fraction of that body's width

Candidates are then matched greedily best-first, one face per body, so two
tracks can never vote on the same face. And if two tracks still end up claiming
one identity, the better-supported one keeps it and the loser is barred from
re-claiming it for `conflict_block_s` — otherwise the two trade the name every
frame forever. That case is logged as `identity_contested` rather than
suppressed, because it usually means something upstream is wrong.

### Where people are

Raw pixel positions are almost useless for reasoning about a room. Calibrate the
floor once and positions come out in metres:

```bash
.venv/bin/python calibrate_floor.py
```

Click floor marks you can measure — rug corners, table legs, tape — and type
each one's position in metres. **Give it 5+ points, not 4.** Four
correspondences are an exact solve: they reproduce your own numbers perfectly
even if one is wrong, so the accuracy readout is meaningless. The fifth point is
what makes it a real check. The solver refuses near-collinear points and fits
that fail to reproduce their own input, because a confidently wrong calibration
is worse than none.

After that, `PersonTrack.room` is `(x, y)` in metres, `pipeline.pairs_within(m)`
lists people close enough to plausibly be talking, and the minimap becomes a
real plan view.

## Voting (raised hands)

The person detector is the `-pose` YOLO variant, so it returns 17 COCO keypoints
alongside each box at no extra model cost. A hand counts as raised when a wrist
is at or above face level.

"Face level" is not a pixel constant — it is `face_level_frac` of the way from
the nose down to the shoulders, which lands at about chin height (measured
against a real detection, the chin sat at 0.46 of that span). Expressed as a
fraction of the person's own body, the same gesture reads the same whether they
are a metre away or across the room. A pixel threshold would silently stop
working at distance, which is exactly the failure a room robot would hit.

Two things stop it flickering:

- **an unseen wrist is not a lowered wrist.** Low-confidence keypoints are
  treated as unknown, because YOLO parks unseen joints at plausible guessed
  positions — using one gives you a wrist that was never observed.
- **asymmetric debounce.** A hand must hold up for `hold_s` (0.4s) to count, and
  stay down for the longer `release_s` (0.8s) to stop counting, so a brief
  occlusion does not retract a vote.

Voting people get an amber box and a `VOTING` tag, with the threshold drawn as a
line so a wrong call is diagnosable at a glance. `pipeline.voting()` returns
them; `hand_raised` / `hand_lowered` events carry the identity and duration.

## Speech

```
mic ─→ Silero VAD ─→ speech segments ─→ ECAPA-TDNN ─→ voice gallery
                            │                              │
                            └──── who was visible? ────────┴─→ attribution
```

Audio runs on its own thread. It has to: frames arrive at ~11 fps, and blocking
the microphone callback for 90 ms drops samples, which VAD then reads as a gap
in speech. Timestamps are the contract between the two — every segment carries
wall-clock start and end, so it lines up against the video events after the fact.

The voice gallery is *the same class* as the face gallery. Speaker embeddings
are unit vectors matched by cosine similarity, exactly like faces, so enrol /
match / reinforce / persist are all reused. Ids are namespaced (`v1` vs `p1`) —
without that both galleries mint "p1" and a voice→face link points at itself.

### Who said it

Three sources of evidence:

1. **presence** — only people visible while the words were said are candidates.
   When one person is in the room this settles it, and that is the case worth
   exploiting because it bootstraps everything else.
2. **learned link** — having watched enough one-person moments, "voice 3 belongs
   to face 2" becomes a fact that survives a crowded room and a turned back.
   Nobody labels anything; the robot works it out by sitting there.
3. **direction** — not available on this laptop. macOS hands over one beamformed
   channel, so there is no angle to compare against where people are standing.
   Arrives with a USB mic array.

The learned link uses **negative evidence**, which is what makes it sound:
co-occurrence alone would link a voice to whoever is simply in the room most
often. Tracking when a voice speaks and a face is *absent* is what distinguishes
"these are the same person" from "these two are often in the same room".
Strength is discounted by how little evidence there is, so twenty observations
at 95% beat two at 100%.

### Calibrating the voice threshold

The defaults in `VoiceConfig` are a guess and should be treated as one. A live
run showed the same speaker scoring 0.68 on one utterance and 0.10 on another,
which is exactly the kind of spread that decides whether the system works.

```bash
python calibrate_voice.py --speaker sam
python calibrate_voice.py --speaker alex     # a second person is essential
python calibrate_voice.py --report
```

It prints how similar a person is to themselves across utterances versus how
similar they are to other people, and puts the threshold in the gap. With only
one speaker recorded there are no "different speaker" pairs, so it says so
rather than inventing a number. If the distributions overlap it reports that
too — that means these voices are not separable under these conditions, and no
threshold will fix it.

## Dashboard

A browser tab opens alongside the camera window, at `http://127.0.0.1:8765/`.

### Camera view — the reconstruction

The default tab replays the room from the camera's own viewpoint: a photo of the
empty room, with articulated figures showing what people actually did, coloured
per identity.

Both halves lean on the camera never moving.

**The room** is recovered, not captured. Every pixel is *usually* background —
people block any given spot only some of the time — so a slow running average
that skips pixels covered by a person converges on the room with nobody in it.
Taking one frame at startup would fail the moment somebody is already there.

Pixels never seen without a person in front of them stay flat grey, and the
header reports coverage (`room 96% recovered`). That matters: seeding the plate
from the first frame instead bakes whoever was already sitting there into the
"empty room" permanently — a motionless ghost that reads as furniture. It is
better to show an honest hole that fills in when they move.

**The people** are drawn from the 17 pose keypoints recorded per person per
sample, as a skeleton with a filled torso and a head scaled from shoulder width.
Because the camera is fixed, the recorded image coordinates *are* the correct
perspective — measured, not modelled — so replay needs no reprojection and does
not inherit the focal-length estimate's error. Raised hands show `VOTING`,
speaking shows a green dot.

### Top-down map

- Plan view with a fading trail per person so direction of travel reads at a
  glance. A green halo means speaking, a dashed amber ring means hand up.
- **The camera itself**, plus the patch of floor it can actually see — so blind
  spots are visible. Distances from the camera are labelled on each person.

  The field-of-view polygon comes straight from the homography and is as good as
  the calibration. The camera *marker* is shakier: recovering focal length from
  a single homography is badly ill-conditioned — on synthetic data a homography
  differing from truth by one part in a million gave a focal length 15% out and
  a position 14 cm off, and realistic click noise lands within 5–25 cm. So
  `calibrate_floor.py` offers the estimate as a default and lets you type a
  measured position instead, and the marker is drawn with a dashed ring until it
  has been measured.
- **Timeline** — one lane per person, green blocks for speech and amber bars for
  raised hands. Unattributed speech gets its own lane rather than being hidden,
  because "something was said and we don't know by whom" is information.
- **Scrub** — drag the slider or click the timeline to move through the session
  and watch people move. `▶ Play` animates it; `Live` snaps back to the present.
- **Playback** — click any utterance in the speech list to hear it.

A browser rather than another OpenCV window for a concrete reason: scrubbing a
timeline and playing audio clips are things a browser does natively and OpenCV
does not do at all.

It binds to `127.0.0.1` only. The data is movement and audio of people in a
room; `0.0.0.0` would hand that to anyone on the network. The audio route
resolves and containment-checks every path, so it cannot be turned into a
general file reader. Both have tests.

Sessions are saved to `data/sessions/<timestamp>/` — a `session.json` plus one
wav per utterance — and can be reopened later with no camera or microphone:

```bash
python review.py            # most recent session
python review.py --list
```

**Top-down needs the floor calibrated.** With a homography, the map is a true
plan view in metres with a metre grid. Without one it falls back to raw image
space and the header says *approximate: floor not calibrated* — a fake plan view
is worse than an obviously approximate one, because you would trust it.

Two details that matter more than they look:

- **Identity resolves retroactively.** People are usually seen before they are
  recognised. Positions are stored by *track* and resolved to an identity only
  when read, so the samples from before their face landed join the same trail —
  and so do samples from after a track id break, which is the same person again.
- **Positions are sampled at 5 Hz, not per frame.** A ten-minute session at
  11 fps is thousands of near-identical points per person: slower to draw and no
  more informative.

## Performance

MacBook, Apple silicon, 960×540, ~5 people in frame:

| | fps |
|---|---|
| person detect + track | 48 |
| face detect + embed (CPU) | 8 |
| face detect + embed (CoreML) | 19 |
| **full pipeline** | **18.5** (54 ms/frame) |

CoreML is used automatically when available — it is 2.4× on the face models,
which is the whole difference between sluggish and usable. MPS measured *no*
faster than CPU for YOLO11n; the model is too small to pay for the transfer.

Faces run every 2nd frame (`face.every_n_frames`); the face pass is the
bottleneck, and faces move slower than the frame rate. Knobs if you need more:
raise `every_n_frames`, drop `face.det_size` from 640, or feed a smaller frame.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

66 tests. The pipeline ones run the real models against a group photo and cover
face-to-body binding, refusing low-quality enrolments, re-identification after a
simulated restart, and the attribution bugs above. Gesture, binding and floor
tests are pure logic and run in under a tenth of a second — the hand-raise ones
drive fabricated skeletons and a controlled clock, so edge cases like an
occluded wrist are pinned down without filming each one.

One of them checks that the suite does not write into your real
`data/gallery/`. It earned its place: the gallery path used to be a default
argument, which binds `GALLERY_DIR` once at import — so redirecting it to a tmp
dir looked like it worked while every test quietly enrolled strangers into the
live gallery. Paths are resolved at call time now.

## Layout

```
run_vision.py         live loop, window, key handling, dashboard
review.py             reopen a saved session in the dashboard
listen_test.py        microphone + VAD check, no video
calibrate_floor.py    click floor points → homography
calibrate_voice.py    measure voice separability, pick thresholds from data
botcirl/
  config.py           every tunable, with the reasoning
  faces.py            face detect + embed; insightface, opencv fallback
  identity.py         the gallery: enrol, match, persist (faces and voices)
  pipeline.py         binds bodies to faces to identities, plus gestures
  gestures.py         pose keypoints → is a hand up
  audio.py            mic capture + voice activity detection (own thread)
  voices.py           speaker embeddings (ECAPA-TDNN)
  binding.py          which voice belongs to which face
  speech.py           ties the mic to the people in the room
  session.py          records positions, speech and gestures for scrubbing
  dashboard.py        localhost web server for the dashboard
  dashboard_page.py   the dashboard page (self-contained HTML/JS)
  floor.py            pixels → metres
  viz.py              debug overlay
data/                 galleries, sessions, events.jsonl, homography  (gitignored)
```

## Things worth knowing before this becomes a product

**Ultralytics is AGPL-3.0.** Fine for prototyping, a real problem for shipping a
product unless you buy their commercial licence. Swappable: `pipeline.py` only
needs boxes and track ids, so torchvision detectors (BSD) or RT-DETR are drop-in
replacements when it matters.

**The gallery is biometric data.** `data/gallery/` holds face embeddings of
everyone the robot has seen, and it is gitignored for that reason. A room robot
that silently enrols every visitor has consent and retention questions attached —
worth deciding deliberately, e.g. auto-expiring unnamed identities, and being
able to show and delete what it holds.

**Picking the robot up voids the floor calibration**, and probably breaks the
tracker too. `FloorMap.valid` is the flag to switch off when the robot's motion
sensing says it is moving. The private-conversation mode you described is
essentially "camera has moved, so trust faces and ignore geometry".

**`quality()` is a proxy, not a real pose estimate.** It infers yaw from eye/nose
landmark spacing. Good enough to gate enrolment; not good enough for "is this
person facing that person", which is what "who is talking to who" will need.

## Where speech fits

Everything is already timestamped into `data/events.jsonl`, which is the seam
audio plugs into. The shape of the rest:

1. **Voice activity + diarisation** — who is speaking, when, as time segments.
   A voice gallery mirroring the face gallery gives you speaker identity that
   survives the person walking out of frame.
2. **Binding voice to face** — with a mic array, direction-of-arrival matched
   against the room positions this already produces. With one mic, correlate
   speech segments against mouth movement instead. This is the hard part and it
   is worth prototyping on its own before integrating.
3. **Who is talking to whom** — needs head orientation, not just position.
   `pairs_within()` is the geometric half; proximity alone will call two people
   sitting silently on a sofa a conversation.

A useful next step on the vision side alone: log a position snapshot every N
frames rather than only on arrival and departure, so there is a continuous
trajectory to correlate audio against later.
