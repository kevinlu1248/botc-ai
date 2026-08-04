# BOTC AI — Agent Handoff

**Repo:** https://github.com/kevinlu1248/botc-ai  
**Local path:** `/Users/kevinlu/botc-ai`  
**Branch:** `main` (synced with `origin/main` as of handoff)  
**Last commit:** `fcb4f1c` Fix keepup vision env prefix (use env cmd)  
**Date:** 2026-08-02  

This document is for a successor agent taking over after a long session that moved **room vision into the web app**, then polished looking-gated STT, interrupts, thinking-vs-spoken, endpointing, UI restyle, and process keep-alive.

Read this first, then `README.md` for product narrative. Prefer code + this file over re-discovering decisions from chat history.

---

## What this product is

Low-latency **voice + room vision** assistant:

| Piece | Role |
| --- | --- |
| Fast model | Claude Sonnet 5 — live conversation, spoken replies |
| Slow model | Claude Opus 5 (fallback; Fable 5 preferred if org has 30-day retention) — background reasoning |
| STT | Deepgram Nova-3 via Node WebSocket proxy (`/ws/stt`) |
| TTS | ElevenLabs Flash streaming |
| Vision | Python sidecar: YOLO11n-pose + OpenCV YuNet/SFace looking-at-camera |

**Speaker attribution is vision-looking, not audio diarization.** Deepgram is single-stream. “Who said it” = people looking at the camera when the final transcript is accepted.

```
camera ─► vision :8766 ──looking gate──┐
                                       ▼
mic ─► Deepgram ─► Sonnet 5 ──tool──► Opus 5 (background)
                     ▲                    │
                     └──── shared context ◄┘
                              │
                 ElevenLabs ◄─┘──► browser (SSE + live video)
```

---

## How to run / stop (do this, not ad-hoc bg jobs)

Vite/API/vision **die when started as agent background jobs** (stdin EOF kills Vite). Use the keepup watchdog:

```sh
cd /Users/kevinlu/botc-ai
npm run up        # start watchdog + ensure vision/api/ui
npm run status
npm run down      # stop everything
```

| Service | Port | Health |
| --- | --- | --- |
| vision | 8766 | `http://127.0.0.1:8766/api/state` |
| api | 3001 | `http://127.0.0.1:3001/api/state` |
| ui | 5181 | `http://127.0.0.1:5181/` |

- State/logs/pids: `.run/` (gitignored)  
- Implementation: `scripts/keepup.sh`  
- Vision must start with `env VISION_FACE_BACKEND=opencv …` — plain `exec VISION_FACE_BACKEND=…` **fails** under bash `exec`. Keepup already does this correctly (`fcb4f1c`).  
- **Reboot:** no LaunchAgent yet — user must `npm run up` again.  
- Open **http://localhost:5181** (Chrome preferred for mic/camera). Mic+camera need localhost/HTTPS. Grant Camera to the **terminal app** (vision process) and browser (mic).

Manual three-terminal alternative: `npm run vision` / `npm run server` / `npm run dev` (Vite needs stdin detached if backgrounded: `… < /dev/null`).

---

## Key files (edit map)

### Server (Node)

| File | Responsibility |
| --- | --- |
| `server/index.js` | Express: `/api/chat`, `/api/events`, `/api/state`, `/api/interrupted`, wires STT+TTS+vision |
| `server/agents.js` | Dual Claude: `runFastChat`, `truncateLastReply`, system prompt (thinking vs spoken, silence rules), slow job tool |
| `server/voice.js` | Deepgram WS, endpointing/grace/fillers/keyterms, TTS route, STT dump WAV |
| `server/vision.js` | Proxy vision state; **looking gate** + `[Room]/[Speaker]/[Said]` formatting |
| `server/context.js` | Shared history, jobs, insights, `pendingTruncation`, bus |
| `server/env.js` | Load `.env` first |

### Client (React / Vite)

| File | Responsibility |
| --- | --- |
| `src/App.jsx` | Chat UI, looking gate call, interrupt freeze, mute-while-busy, Thinking label, room card |
| `src/useMic.js` | Mic → PCM worklet → `/ws/stt`; mute drops finals but **not** `speech_started` when playing |
| `src/useTts.js` | Streaming TTS, `stopTurn()` spoken-char cut, cancel on new turn |
| `src/vad.js` | Local barge-in VAD (RMS + speech band + flatness + dominance + loud path) |
| `src/VisionPanel.jsx` | Live frame + people chips |
| `src/App.css` | Compact room vision card/chips restyle |
| `src/MicMeter.jsx` | Level + thinking/speaking status |

### Vision (Python)

| File | Responsibility |
| --- | --- |
| `vision/server.py` | HTTP state + JPEG frame; looking hysteresis/sticky knobs |
| `vision/botcirl-src/` | Vendored botcirl pipeline (faces, pose, gallery). **Audio ECAPA path not wired** in botc-ai live loop |
| `vision/.venv` | Project venv (recreate if broken/zipped) |

### Ops

| File | Responsibility |
| --- | --- |
| `scripts/keepup.sh` | Watchdog every 5s |
| `.env` / `.env.example` | Keys + STT/vision knobs |
| `README.md` | User-facing docs (attribution, config, pipeline) |
| `recordings/` | Optional STT dumps (`STT_DUMP=1`) |

---

## Core product rules (do not regress)

### 1. Looking gate

- On each **final** voice transcript, client → server vision gate (`server/vision.js`).
- **Nobody looking → drop transcript** (do not chat, do not interrupt).
- Looking people credited; model gets:

```
[Room] Person 1 [looking]; Person 2 [not looking]
[Speaker] Person 1 — looking at the camera (look 0.92)
[Said] what's the weather
```

- UI shows e.g. `Person 1 (looking): …`
- **Typed text always sends**, skips looking gate, **always interrupts** TTS if audible.
- Looking is **head-toward-camera / face frontal score**, not iris gaze. Close-up flakiness was mitigated with hysteresis + sticky + Node fallback `looking_score >= 0.35`.

Vision defaults (`vision/server.py` env):

| Env | Default | Meaning |
| --- | --- | --- |
| `VISION_LOOK_ENTER` | 0.38 | Enter looking |
| `VISION_LOOK_EXIT` | 0.22 | Exit looking |
| `VISION_LOOK_HOLD` | 0.15s | Hold before enter |
| `VISION_LOOK_RELEASE` | 1.6s | Release delay |
| `VISION_LOOK_STICKY` | 2.5s | Sticky looking |
| `VISION_FACE_BACKEND` | opencv | YuNet/SFace |
| `VISION_WIDTH`×`HEIGHT` | 960×540 | Capture |
| `VISION_PREVIEW_W` | 640 | JPEG preview width |

### 2. Thinking vs spoken (critical)

- Model uses Anthropic **adaptive thinking**; thought deltas → UI only (`type: "thought"`).
- **Never TTS thoughts.** Only `type: "delta"` text goes to ElevenLabs.
- Model may **think and produce zero spoken text** (mic checks, fragments, nothing useful). Prompt in `fastSystem()` enforces: do not narrate silence (“I’ll stay quiet”).
- UI shows a **Thinking** label while busy without spoken stream.

### 3. Half-duplex + barge-in

While assistant turn is active with voice out:

- **Mute Deepgram finals** for the whole busy turn (not only while audio is playing) so TTS echo doesn’t re-enter as user speech.
- Mic still streams PCM; **`speech_started` must fire even when muted** if playback is armed — otherwise “stop” never interrupts.
- Local VAD (`src/vad.js`) + Deepgram `SpeechStarted` both call `interrupt()`.
- Typed keystrokes interrupt only while assistant is speaking/audible.

### 4. Interrupt truncation

Flow:

1. Client `tts.stopTurn()` → `{ index, spoken, full }`.
2. Freeze UI: stop appending model deltas (`streamLiveRef`); show spoken vs dim unspoken.
3. Always `POST /api/interrupted` with `{ spoken }` even if index is 0.
4. If turn still streaming: set `shared.pendingTruncation`; applied when assistant message is pushed.
5. `truncateLastReply(spoken)` rewrites last assistant content to spoken + interrupt note.

**Extended thinking API constraint:** Anthropic returns **400** if you partially edit a message that still has `thinking` / `redacted_thinking` blocks. On interrupt, **replace entire assistant content** with plain text note (optionally keep identical `tool_use` blocks). See `truncateLastReply` in `server/agents.js`.

### 5. TTS lifecycle

- `stopped=true` must only be set on real barge-in / cancel — **not** left true across a new turn (was a silent-TTS bug).
- New send must re-arm TTS (`cancel` then fresh feed).
- Interrupt only when playing/armed, not spuriously mid-stream before first audio.

### 6. STT turn end (current defaults in code)

Code defaults in `server/voice.js` (`.env.example` comments may be older/snappier — **trust code**):

| Knob | Default | Role |
| --- | --- | --- |
| `STT_ENDPOINTING_MS` | **650** | Silence before Deepgram ends utterance |
| `STT_UTTERANCE_END_MS` | **1000** (DG min) | UtteranceEnd fallback |
| `STT_CONTINUATION_GRACE_MS` | **750** | Extra wait if `looksUnfinished` |
| `STT_IDLE_FLUSH_MS` | **1600** | Backstop if no boundary |

Logic:

- Accumulate `is_final` segments; flush on `speech_final` / `UtteranceEnd` / idle.
- **Drop pure fillers** (`um`, `uh`, `okay`, …) via `isFillerOnly` unless `isCommand`.
- **Hold unfinished**: dangling conjunctions/prepositions, trailing `,;:`, short openers (`Okay so`, `I need`), fillers mid-thought → grace timer.
- **Commands complete immediately:** `stop|wait|cancel|quiet|enough|…` (`isCommand`).
- Fixed false unfinished for phrases like **“tell me a joke”** (`4f0d48b`).
- Keyterms always boost barge-in: `stop:2`, `wait:1.5`, `cancel:1.5`, … plus `STT_KEYTERMS` env.

**Tradeoff:** snappier endpointing fragments speech and hurts accuracy; longer feels sluggish. User still feels Deepgram Voice Agent product is smoother — this is a custom stack, not that product.

### 7. “Stop” reliability

Mostly **ASR**, not gate logic. Mitigations already in:

- Keyterm boost for stop/wait/cancel
- Loud VAD path (`LOUD_RMS: 0.07`) when AEC dulls spectral features
- `speech_started` while muted + playing
- Unfinished hold does **not** apply to bare “stop”

Still imperfect: “stop” can mis-hear as e.g. “Okay. So” — treat further work as ASR/keyterm/audio chain, not looking gate.

---

## API surface (quick)

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/api/chat` | NDJSON stream: `thought`, `delta`, `job_*`, `error`, done |
| GET | `/api/events` | SSE slow-model updates |
| GET | `/api/state` | models, stt, voice flags, vision snapshot, jobs |
| POST | `/api/interrupted` | `{ spoken }` → truncate or defer |
| WS | `/ws/stt` | Binary PCM + JSON control; query `rate`, `model` |
| proxy | `/api/vision/*` | Frame, faces, state via Node → 8766 |
| vision | `:8766/api/state`, `/api/frame.jpg` | Direct sidecar |

---

## Models & keys

From `.env` / startup logs / UI header:

- `FAST_MODEL` default `claude-sonnet-5`
- `SLOW_MODEL` currently **`claude-opus-5`** — Fable 5 needs 30-day Anthropic data retention (not enabled on this org)
- `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`
- ElevenLabs free tier: **library** voices 402; use default voices (Sarah `EXAVITQu4vr4xnSDxMaL`, etc.)
- Missing keys degrade (type-only / text-only), don’t crash

---

## Recent commit history (vision → keepup)

```
c525f43 Add room vision: looking-gated STT, live camera gallery, dual-model voice bot
b0ede17 Fix TTS streaming and close-up looking detection
ca19a29 Fix reply TTS: stop false interrupts during streaming
7836f3b Restore proper interrupt truncation and faster barge-in
8351d8e Restore think-without-speaking for fragments and filler
79fc448 Make barge-in interrupt much faster
93b80b5 Fix thinking UI and stop silence/TTS echo loops
919ac10 Restyle room vision to match original sidebar
caa06a3 Fix 400 on interrupt with extended thinking blocks
472587e Snappier end-of-speech and barge-in
ae82e20 Smarter STT turn ends: hold um/uh/openers, drop pure fillers
4f0d48b Fix tell me a joke false unfinished
b8fca92 Add keepup watchdog so UI/API/vision stay running
fcb4f1c Fix keepup vision env prefix (use env cmd)
```

---

## Session 2 (successor agent) — what changed

1. **Bare stop commands no longer get answered.** Saying "stop" barge-in-cancelled the reply *and*
   forwarded "Stop." as a chat turn, so the model answered "Okay, stopping." — a reply to an
   instruction whose point was to stop replying. `src/App.jsx` now swallows a *bare* interrupt command
   (`stop|wait|hold on|cancel|quiet|shut up|enough|never mind|pause`, optionally punctuated) when it
   lands within 4s of an interrupt or while audio is audible. The turn is still shown, tagged
   `stopped playback`, so an unanswered turn doesn't read as a hang. Phrases that merely *contain* a
   command word ("stop the deploy", "can you stop the timer?") still go through — 14/14 on a
   boundary test.
2. **Room video enlarged.** Sidebar is now `clamp(380px, 32vw, 620px)` and `.vision-preview` uses
   `aspect-ratio: 16/9` instead of a fixed `height: 132px`. Measured 486×273 (was ~356×132). The old
   fixed height plus `object-fit: cover` was cropping heads off; capture is 960×540, so 16:9 shows the
   whole frame.
3. **Voice binding — cheap fix for "my phone's video gets treated as me"** (part of open issue 3).
   Deepgram `diarize=true` is now on (`STT_DIARIZE=false` to disable), per-word speaker labels are
   tallied per turn in `server/voice.js`, and the dominant index rides the `final` message to the
   client and on to `/api/vision/gate`. `server/vision.js` binds the **first index heard while someone
   is looking** and rejects every other voice with `reason: "other_voice"` (toast in the UI).
   Binding resets on each `/ws/stt` connection because indices are per-connection. Exposed at
   `/api/state → stt.voice`; `VOICE_BIND=false` keeps labels but accepts all voices.
   *Known limits:* unsupervised clustering is shaky on 1–2s utterances, and the binding resets on
   reconnect. If it proves unreliable, the ECAPA path is the real fix — `vision/botcirl-src/botcirl/`
   already has `voices.py` (`SpeechBrainECAPA.embed`), `identity.py` (`IdentityGallery.match/enroll`)
   and `calibrate_voice.py` (derives the threshold from your room), with torch 2.13 + speechbrain
   1.1.0 already installed in `vision/.venv`. Roughly a day: tee PCM from Node to the sidecar,
   embed per utterance, verify against the gallery.

4. **Interruptions no longer lose their first words.** "stop here" transcribed as just "here".
   Cause: `looksUnfinished` returns false for commands, so "stop" flushed as its own final *while the
   mic gate was still shut*, and `useMic` discarded gated finals outright. Gated finals are now
   **held** (`HELD_STITCH_MS` 2500) and prepended to the next final — but only when a real barge-in
   was stamped (`fireInterrupt`), which is what distinguishes the user's voice from TTS echo. If
   nothing follows within `HELD_ALONE_MS` (1200) the held words are delivered alone.

5. **Acoustic voice profile — this, not diarization, is what rejects the phone.** Diarization was
   unreliable (`stt.voice.speaker` often stayed `null`), so `src/profile.js` builds a per-utterance
   profile from the **raw** mic tap — `lowRatio` (80–300 Hz share; a phone speaker cannot make a human
   fundamental), `rmsStd` (broadcast audio is compression-flattened), `centroidMean`, `flatness`. The
   gate enrolls the first looking-gated utterance and rejects past `VOICE_PROFILE_MAX_DIST` (2.2);
   phone-like profiles measure ~4.3, real speech ~0. `POST /api/voice/reset` re-enrolls.
   *Gotchas, both found by testing:* profile from the RAW tap (the conditioning chain flattens the
   very features used), and reinforce only on close matches (dist < half threshold) or a borderline
   foreign voice drags the reference until it matches. Diarization is still on and still binds, but is
   now the weaker of the two signals.

6. **Barge-in no longer fires on inhaling / hair scratching / taps.** `472587e` had added
   `if (rms >= VAD.LOUD_RMS) return true;` — pure loudness, no spectral check — and `FRAMES: 1`.
   Any loud broadband transient interrupted. The loud path now also requires `flatness <= 0.35` and
   `speechRatio >= 0.3`; `HARD_FLATNESS_MAX` went 0.6 → 0.35 and `HARD_SPEECH_RATIO` 0.25 → 0.45,
   which sit inside a measured gap (voiced speech flatness 0.002–0.04, noise 0.50–0.58). Frame
   counting was replaced with duration confirmation (`VAD.MIN_MS` 90) because the 6ms poll reads
   heavily overlapping FFT windows, so counting frames barely filters time at all. Verified 8/8 on
   synthesized scratch / inhale / tap versus real speech at two levels. **Shouting still cuts through
   instantly** — that was the point of the loud path and it is preserved.

7. **Gemini as a conversation-model option.** `server/gemini.js` translates
   Anthropic-shaped `shared.history` to Gemini `contents` and back, streams SSE, and
   drives function calling; `runFastTool` in agents.js is now shared so a tool behaves
   identically on either provider. Verified: plain reply, tool call to the reasoning
   model, and switching provider **mid-conversation both ways** with shared history.
   *Two traps:* Gemini needs `thoughtSignature` echoed back with its `functionCall`,
   but storing it on the history block makes Anthropic 400 on switch-back
   ("Extra inputs are not permitted") — signatures live in a Map beside the history
   (`rememberSignature`). And `thinkingConfig.includeThoughts` returns a *signature*,
   not readable thought text, so the Thinking panel stays empty on Gemini.

8. **Settings modal** (gear right of the model pills). `server/settings.js` holds
   runtime-mutable config with env defaults; `GET/POST /api/settings`. Conversation
   model, reasoning model, STT, TTS model, spoken voice and microphone all switch
   without a restart (STT reconnects the mic, since the model is fixed per Deepgram
   connection). All six TTS models and 17 voices were verified by calling the API —
   the key is TTS-scoped so `/v1/models` and `/v1/voices` both 401.

9. **Silero VAD replaces the hand-tuned heuristics for barge-in.** See README →
   Models. 9/9 on synthetic cases and, on Kevin's own 327s recording, it rejects all
   332 frames that were loud enough to trip the old `rms >= LOUD_RMS` shortcut.
   **Hard-fails by design** — no fallback (per Kevin's instruction), so the mic
   starting is itself proof the model loaded. Call contract: **576 samples** per
   inference (64 carried context + 512 new); a bare 512 returns ~0.003 on clear
   speech and looks like a working model that hears nothing.

10. **Turn end is client-driven.** Silero detects ~480ms of silence and sends a
   `finalize` control frame; the server forwards Deepgram's own `Finalize`. Because
   latency no longer depends on server endpointing, `STT_ENDPOINTING_MS` went *up*
   650 → 1100 (longer segments = more context = better accuracy) and grace 750 → 450.

11. **Interrupt truncation, again.** `caa06a3`'s guard dropped thinking blocks but
   kept `tool_use`, which Anthropic also rejects — thinking must accompany tool_use
   in the same turn. Now the interrupted assistant turn collapses to a plain string
   and **all following messages are dropped** (they belong to the aborted tool
   round-trip, and an orphaned `tool_result` is its own 400). Reproduced the 400 and
   confirmed it is gone.

12. **Frontend errors reach the server log.** `src/report.js` reports
   `window.onerror`, unhandled rejections and every error toast to
   `POST /api/client-error`; `GET /api/client-errors` returns the last 50. The
   extended-thinking 400 had been sitting in a toast unnoticed.

13. **Queued utterances expire.** Anything spoken while a turn was in flight was
   replayed unconditionally, so an old "continue" fired after the thing it referred
   to had been interrupted. Entries are stamped, expire after 3.5s, and the queue is
   cleared on interrupt.

14. **Diarization confirmed working** (`speaker=0` tallied per turn and carried on
   the `final`). The earlier `stt.voice.speaker: null` was simply no looking-gated
   turn having reached the gate since a restart. It remains the *weaker* signal; the
   acoustic profile is what actually rejects a phone.

15. **Doc drift fixed (was open issue 8).** README and `.env.example` now match code defaults
   650 / 1000 / 750 / 1600.

Not done, deliberately: no commits (not asked), no LaunchAgent (persistent user config — needs the
user's go-ahead), no GitHub invite for Radilx (outward-facing — needs confirmation).

## Open issues / good next work

Prioritized by user pain observed in the session:

1. **Endpointing UX vs accuracy** — still not as good as Deepgram Voice Agent. Options: tune grace/endpointing further; experiment with Deepgram Flux / Voice Agent API; smarter unfinished heuristics; optional “push-to-talk” mode.
2. **“Stop” ASR errors** — more keyterms, phonetic variants, or command classifier on partials; consider always treating short high-confidence barge-in as interrupt even if text is wrong.
3. **Looking ≠ eye gaze** — expect flakiness at extreme angles / multi-person both looking. Optional: re-attach botcirl ECAPA voice↔face binding from `vision/botcirl-src` for true diarization.
4. **No auto-start on reboot** — optional macOS LaunchAgent / login item for `scripts/keepup.sh start`.
5. **Raised hands** — available in vision pose; **not wired** into chat.
6. **Lag** — vision FPS ~6 on last status; capture/preview/JPEG knobs already exposed; avoid reprocessing old frames (server already drains buffer).
7. **Invite collaborator** — user previously wanted GitHub invite for **Radilx** (confirm if done).
8. **README vs code** — some README/`.env.example` endpointing numbers lag code defaults (650/750/1600). Sync docs if you touch STT.
9. **Do not re-fix monologue/silence/TTS-echo bugs** unless user reports regression — those were deliberately fixed (`93b80b5`, `8351d8e`, prompt restore).

---

## Debugging cheatsheet

```sh
npm run status
curl -s http://127.0.0.1:3001/api/state | jq .
curl -s http://127.0.0.1:8766/api/state | jq .
tail -f .run/logs/{vision,api,ui,watchdog}.log
```

STT dumps (WAVs in `recordings/`):

```sh
# in .env or process env
STT_DUMP=1
# optional STT_DUMP_DIR
```

Scripts: `scripts/stt-smoke.mjs`, `scripts/stt-compare.mjs`, `scripts/condition.py`.

Barge-in not working checklist:

1. Is `mic.setPlaying(true)` while speaking?
2. Does muted path still forward `speech_started`? (`useMic.js`)
3. Is VAD too strict under AEC? (try loud path / room noise)
4. Is TTS `stopped` stuck true from previous turn?

Interrupt 400 thinking:

- Ensure `truncateLastReply` strips thinking blocks entirely.

Vision won’t start under keepup:

- Must use `env VISION_FACE_BACKEND=opencv python …`, not `exec VAR=…`.

---

## What was explicitly out of scope / not claimed

- This is **not** multi-speaker audio diarization.
- Not lip-sync “who is talking.”
- Not full Deepgram Voice Agent product parity.
- botcirl voice embeddings exist in vendor tree but are **disabled** (`cfg.audio.enabled = False`).

---

## Suggested first actions for the next agent

1. `cd /Users/kevinlu/botc-ai && npm run status` — confirm three services + watchdog.
2. Open http://localhost:5181 — look at camera, speak, try type-interrupt and “stop.”
3. Skim `server/agents.js` (`fastSystem`, `truncateLastReply`) and `server/voice.js` (turn end) before changing behavior.
4. Ask the user which pain is next: endpointing, stop ASR, looking accuracy, or process auto-start — don’t thrash all knobs at once.

---

## Session origin note

Work began in a related “botcirl / who’s looking” dashboard, then landed and continued in **botc-ai**. Workspace of the chat that produced this handoff may have been `diff-viewer`; **all product code is under `/Users/kevinlu/botc-ai`**. Ignore unrelated untracked PNGs in other workspaces.

End of handoff.

## Session 3

16. **`thinking` blocks 400 — properly diagnosed this time.** The previous fix was wrong about the
    mechanism: it assumed a partial edit to an assistant message was the cause, but the only mutation
    in the codebase (`truncateLastReply`) replaces content with a plain **string**, which cannot
    produce the reported `messages.N.content.1` path. Reproduced the real constraints against the API
    with `scripts/thinking-*.mjs`; results are tabulated in README → *History hygiene*. Headline: with
    `display: "summarized"` neither Sonnet 5 nor Opus 5 enforces thinking-block integrity, so the exact
    reported string is only reachable on the **raw** thinking path — `runReasoningJob`, which omitted
    `display` and additionally uses `fallbacks: "default"` (a substitute model answers, so its blocks
    don't match the model named next turn). Fixes: per-request payload sanitising via
    `messagesForClaude(model)` with a `WeakMap` recording which model produced each assistant message;
    explicit `display: "summarized"` on the reasoning job; and **retry-once-with-thinking-stripped** in
    both loops, so a stale block can no longer kill every subsequent turn (previously only a server
    restart recovered — the reported failure shows the same error twice back to back).
17. **`claude-haiku-4-5` was broken in the dropdown.** It rejects adaptive thinking *and*
    `output_config.effort`. `fastRequest` omits both for it. Every turn used to fail with it selected.
18. **Default voice is George** (`JBFqnCBsd6RMkjVDRZzb`) in `settings.js` and `.env`; verified through
    `/api/tts` (55 KB audio, per-character alignment intact).
19. **Test tooling added** — all runnable without a microphone:
    - `scripts/test-history-sanitizer.mjs` — 10 assertions against the live API, incl. reproducing a
      rejected payload and proving the sanitised version is accepted.
    - `scripts/repro-matrix.mjs [scenario]` — drives the running server through `single-interrupt`,
      `concurrent`, `concurrent-interrupt`, `double-interrupt`, `haiku-selected`, `model-switch-storm`,
      `gemini-tool-then-claude`. Restarts the server per scenario (history is in memory).
    - `scripts/thinking-{roundtrip,sysprompt,lowffort,foreign-shape,cross-model,tool-continuation}.mjs`
      — the characterisation probes behind the README table.
    - `/api/chat` now dumps history to `.run/last-bad-history.json` on error, because a restart
      destroys the only copy of the payload that failed.
20. **Verified with real audio**, not just synthetic turns: `recordings/2026-08-03_02-40-33-nova-3.wav`
    through `scripts/stt-smoke.mjs` returns a clean final transcript. Note
    `botc-stt-2026-08-02T03-48-26-643Z` yields **0 finals** — it is the quiet pool-noise recording, and
    rejecting it is the gate working as intended, not a regression.

### Still open
- Combined diarization + acoustic-profile gate needs the user at the camera; it short-circuits on
  `not_looking` while they're away.
- Gemini returns a thought *signature*, not readable summaries, so the Thinking panel stays empty on
  Gemini. Provider difference, not a bug.
- Uncommitted work: nothing has been committed this session or last; ~25 files are dirty.

## Session 4 — the false barge-ins

21. **Root cause: two barge-in authorities, and the wrong one was firing.**
    `useMic.js` handled Deepgram's `speech_started` (energy-based `vad_events`) by
    calling `fireInterrupt()` directly, bypassing Silero entirely. Taps, sirens and
    door bangs are exactly what an energy VAD trips on. Every Silero threshold was
    therefore irrelevant to the false interrupts being reported — and the file's own
    comment already said Deepgram's VAD "can't be trusted during playback".
    Proven on the user's real captured audio (`scripts/analyse-recording.mjs`):
    **124 loud-but-not-speech buckets (RMS to 0.365) scoring 0.00–0.03 on Silero**,
    against 2 genuine speech buckets, while the assistant was being cut off.
    Diagnostic absence was the tell: the instrumented Silero path logged nothing.
22. **Fix is structural, not a threshold.** New `src/bargein.js` is the single
    authority; `speech_started` is logged as `bargein-ignored` and acted on by
    nothing; dead `isHardBargeIn` deleted from `vad.js` (a loudness-first interrupt
    helper with no callers — the trap that created the second authority);
    `vad.js` demoted in its header to profiling-only.
23. **MIN_FRAMES 2 → 4** (~64ms → ~128ms sustained). Measured, not guessed:
    4 consecutive costs +64ms flat and misses nothing; 6-consecutive and 6-of-10
    both cost +128..608ms, worst on quiet speech, so they were rejected.
24. **Tooling** — all offline, no microphone needed. Silero now runs under Node
    (`onnxruntime-web/wasm` with the model passed as bytes; `wasmPaths` must NOT be
    overridden or paths double up):
    - `scripts/vad-bargein.mjs` — real model over real recordings + synthetic
      sirens/clicks/typing; compares candidate gates and imports `BARGE_IN` from
      `src/bargein.js` so it always tests the shipped value.
    - `scripts/analyse-recording.mjs <wav> [start] [end]` — per-half-second timeline
      of RMS vs Silero probability; this is what identified the taps.
    - `scripts/deepgram-vad-taps.mjs` — streams synthetic taps at the relay.
      Note: synthetic taps did *not* trip Deepgram, so real audio was required.
    - WAV reader gotcha fixed in both: a still-open recording reports a `data`
      chunk size of **0**, which silently yields an empty file.
25. **`reportEvent(kind, data)`** added to `src/report.js` for non-error diagnostics;
    the server already logs any `kind` as `[client:<kind>]`. `STT_DUMP=1` is now on.

### Caveats
- Synthetic sirens/clicks/taps score 0.01–0.05 on Silero but did not reproduce the
  Deepgram trigger, so noise characterisation from synthesis alone is not conclusive.
  Real captured audio settled it; keep `STT_DUMP=1` on for future reports.
- One `[bargein]` entry at 22:51:50Z in the client-error log is a synthetic curl test,
  not a real event.
26. **The other half of the same bug: Silero was tapping the wrong signal.**
    Removing the energy detector (item 21) exposed why it had existed. Silero read
    `source` (raw mic) while Deepgram read `tail` (conditioned: highpass, compressor,
    4x makeup gain, limiter). Raw mic is un-amplified *and* ducked by browser echo
    cancellation during playback, so barge-in could not fire while the assistant
    spoke — the trail captured **0.0003 RMS / p=0.00** at the exact moment the user
    said "Stop. Stop. Stop. Stop." and Deepgram transcribed it correctly.
    Fix: `tail.connect(vadNode)`. Measured on the captured session's conditioned
    stream (which is what STT_DUMP records, so this is reproducible offline):
    user speech p=0.73-1.00 (fires), 364 loud transients p=0.00 at RMS to 0.36
    (ignored), assistant's own TTS produced no speech buckets (no self-interrupt).
    The stale rationale was the comment "vad.js thresholds are calibrated against
    unprocessed levels" — true for the hand-tuned heuristics, meaningless for a model,
    and never revisited when Silero replaced them.
    `scripts/vad-bargein.mjs` now includes two slices of that real recording as the
    regression test for both directions.

### Margin to watch
- The real tap slice peaks at **p=0.29** against an ENTER threshold of 0.5. That is a
  genuine margin but narrower than the synthetic noises (0.01-0.05), so if tap-like
  false triggers ever return, that number is the first thing to re-measure rather
  than a reason to add a second detector.

## Session 5 — explicit non-response

27. **`[NO RESPONSE]` sentinel replaces empty-turn silence.** `NO_RESPONSE` is exported
    from `server/agents.js` and taught in `fastSystem()`. An empty assistant turn was
    the old convention and it is ambiguous — indistinguishable from a dropped stream,
    a refusal, or a bug. The model now declines explicitly, the decision is visible in
    the transcript and in history, and the UI shows a `stayed silent` badge driven by a
    `{type:"no_response"}` event (`turn.declined`) rather than by inferring from
    "no text but some thinking".
28. **The sentinel can never reach text-to-speech.** `noResponseFilter()` wraps `send`
    for the whole turn, so both providers are covered by one code path. Deltas stream
    token-by-token and TTS starts mid-stream, so the filter holds output *only* while
    what has arrived could still become the sentinel — a normal reply diverges on its
    first character and pays no latency. `finish()` releases a partial hold so a
    malformed near-sentinel is never swallowed. Test asserts 0 leaks into deltas.
29. **Client-side swallowing of "stop" removed.** The model now hears everything the
    user says, which was the explicit request. Two bugs died with it: the model never
    learned it had been told to stop, and the pattern only matched a *single* word, so
    "Stop. Stop." (the common shape when the first stop is ignored) was forwarded and
    answered out loud anyway. `lastInterruptRef` and `audibleRef` became write-only
    with the swallow gone and were removed.
30. **`scripts/test-silence.mjs`** — 12 cases on a fresh server each, asserting
    declined-vs-speaks, and that the sentinel never appears in a delta. 12/12.

### Observation, not a bug
- 6 of 8 declined cases produce **no thinking at all**: adaptive thinking at
  `effort: "low"` skips reasoning for trivial utterances. The decline is still
  explicit and the badge still shows, but the Thinking panel is empty. Forcing
  thinking would add latency to every turn, so it was left alone.
