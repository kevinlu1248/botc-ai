# BOTC AI

A low-latency **voice + room vision** assistant. Two language models share one context; a laptop
camera decides *whether* speech is accepted and *who* is credited.

- **Fast model — Claude Sonnet 5.** Live conversation. Streams at low effort; replies are spoken.
  One tool, `start_deep_reasoning`, hands hard questions to the slow model without blocking.
- **Slow model — Claude Fable 5** (or Opus 5 fallback). Background high-effort reasoning with
  `announce` for interim findings.
- **Room vision** (Python sidecar, vendored from botcirl). People tracking, face gallery,
  looking-at-camera. Live annotated video + sidebar gallery in the UI.

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

## Quick start

```sh
cd botc-ai
cp .env.example .env      # fill in ANTHROPIC / DEEPGRAM / ELEVENLABS keys
npm install

# three processes (or use separate terminals)
npm run vision            # camera + gallery     → http://localhost:8766
npm run server            # chat + STT + proxy   → http://localhost:3001
npm run dev               # UI                   → http://localhost:5181
```

Open http://localhost:5181.

- **Look at the camera and talk** — speech is only transcribed while someone is looking.
- **Type anytime** — typed text always sends and always interrupts TTS.
- Hard questions go to the reasoning model; watch the right-hand panel.

Mic and camera need `localhost` or HTTPS. Grant **Camera** (and mic) to your terminal app for the
vision process, and to the browser for STT.

> **Vite must be started with stdin detached** if you background it (`npx vite … < /dev/null`).
> Vite watches stdin for shortcuts and exits on EOF.

---

## Who is talking? (speaker attribution — not audio diarization)

**Short answer: it is vision-based, not classic audio diarization.**

There is **no** multi-speaker audio clustering (no pyannote, no ECAPA voice gallery on the live
path, no mic-array direction-of-arrival). Deepgram produces a single transcript stream. “Who said
it” is inferred from the **camera at the moment the utterance is accepted**.

### Rules in this app

1. **Speech-to-text** — Deepgram Nova-3 streams one transcript (browser mic → Node → Deepgram).
2. **Looking gate** — On each *final* transcript, Node calls the vision sidecar. If **nobody** is looking at the camera, the transcript is **dropped** (not sent to the model).
3. **Speaker label** — If one or more people are looking, they are credited as the speaker(s). The model receives a structured user message:

```
[Room] Person 1 [looking]; Person 2 [not looking]
[Speaker] Person 1 — looking at the camera (look 0.92)
[Said] what's the weather
```

4. **Display** — The chat UI shows e.g. `Person 1 (looking): what's the weather`.

So attribution is:

- **Video:** face identity gallery (YuNet + SFace by default) + head-toward-camera score (face landmarks).
- **Not audio:** we do *not* match voiceprint to face on this path. If two people both look at the camera while one speaks, both may be listed as speakers.

### What vision *does* provide

| Signal | Source | Used for |
| --- | --- | --- |
| People in frame | YOLO11n-pose | Presence, boxes, pose |
| Who is who | Face embeddings (session gallery) | Labels Person 1, Person 2, … |
| Looking at camera | Face landmark frontal score | STT gate + “looking” badge |
| Raised hands | Pose keypoints | Available in vision; not wired into chat yet |

### What it does *not* do

- Audio-only speaker diarization (“this segment is speaker A”)
- Lip-sync / mouth-motion “who is moving their lips”
- Direction of arrival (single laptop mic)
- Transcription when looking away (by design)

Typed messages skip the looking gate and are treated as direct user input.

### Related: full botcirl stack

The original `botcirl` prototype also had **ECAPA voice embeddings + voice↔face co-occurrence
binding**. That audio half is **not** active in botc-ai’s live loop (botc-ai uses Deepgram for STT
and vision only for gate/identity). The vision package under `vision/botcirl-src/` still contains
that code if you want to re-attach it later.

---

## Configuration

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Both models |
| `DEEPGRAM_API_KEY` | Speech-to-text |
| `ELEVENLABS_API_KEY` | Text-to-speech |
| `ELEVENLABS_VOICE_ID` | Optional; defaults to Sarah |
| `FAST_MODEL` / `SLOW_MODEL` | Model overrides |
| `STT_MODEL` | Deepgram model, default `nova-3` |
| `STT_KEYTERMS` | Vocabulary to bias decoding toward |
| `STT_DUMP` / `STT_DUMP_DIR` | Record the PCM sent upstream, for diagnosis |
| `STT_ENDPOINTING_MS` etc. | Turn detection, see below |
| `VITE_MIC_NOISE_SUPPRESSION` / `VITE_MIC_AGC` | Browser audio processing (AGC off by default) |
| `VITE_MIC_GAIN` / `VITE_MIC_CONDITIONING` | Makeup gain (4×) and the conditioning chain |

Missing keys degrade rather than crash: without Deepgram the mic reports the problem and you can
still type; without ElevenLabs replies are text-only. The server prints which models it resolved at
startup and the UI header shows them, so this can't silently drift.

### Model availability

**Fable 5 requires 30-day data retention on your Anthropic org.** Without it every request fails
with `400 … must have data retention enabled` — Fable 5 is not available under zero data retention.
This org doesn't have it, so `.env` sets `SLOW_MODEL=claude-opus-5`, the next-strongest reasoning
model available. Delete that line once retention is enabled.

### ElevenLabs free tier

*Library* voices (Rachel, Aria, …) return `402 "Free users cannot use library voices via the API"`.
Default voices work — Sarah (`EXAVITQu4vr4xnSDxMaL`), George (`JBFqnCBsd6RMkjVDRZzb`) and Brian
(`nPczCjzI2devNBz1zQrb`) are confirmed. The key in use is scoped to text-to-speech only, so
`/v1/voices` and `/v1/user` return 401 — that's expected, not a broken key.

---

## Voice pipeline

### Speech to text

Deepgram **Nova-3** over a WebSocket. The browser captures mic audio, converts it to 16 kHz PCM in an
`AudioWorklet` (off the main thread), and streams it to the Node server, which relays to Deepgram.
The Deepgram key never reaches the browser.

**Sample rate is negotiated, not assumed.** `new AudioContext({sampleRate: 16000})` is a *request*;
the browser may ignore it. The client reports the rate it actually got via `?rate=` and the server
passes that to Deepgram, so a mismatch can't silently produce garbage transcripts.

**Turn detection** — when is your sentence over? Three layers, all tunable:

| Variable | Default | Role |
| --- | --- | --- |
| `STT_ENDPOINTING_MS` | 1000 | Silence before Deepgram ends the utterance |
| `STT_UTTERANCE_END_MS` | 1000 | Fallback boundary (Deepgram minimum is 1000) |
| `STT_CONTINUATION_GRACE_MS` | 900 | Extra wait when speech trails off mid-thought |
| `STT_IDLE_FLUSH_MS` | 2500 | Backstop when no boundary is reported at all |

Three things are worth knowing here:

- **Deepgram's example value of 300 ms cuts people off constantly.** An ordinary mid-thought pause is
  around a second. That was the cause of `"Come up with a hard math problem, and then"` being
  submitted as a complete request.
- **`speech_final` alone is not enough.** On continuous speech Deepgram finalizes in *segments*
  (`is_final`) and may never emit `speech_final`, so the relay accumulates segments and flushes on
  `speech_final` **or** `UtteranceEnd`. Relying on `speech_final` alone means voice input never
  submits anything at all.
- **Short endpointing costs accuracy.** More segments means each one gives Deepgram's language model
  less context, so errors cluster around pauses. The duplicated word in `"Yeah. As as much as,"` is
  that artifact. It's a genuine latency/accuracy dial.

**Trailing-off detection.** On top of the timers, `looksUnfinished()` in `server/voice.js` checks
whether the transcript ends on a dangling word or clause punctuation ("…problem, and then"). If so
the turn is held for `STT_CONTINUATION_GRACE_MS` rather than submitted, and resumed speech cancels
the hold. This is a **regex heuristic, not a model** — free and instant, but it matches surface forms
rather than understanding meaning. The upgrade path is a fast LLM completeness check (as Pipecat's
smart-turn model does), at ~200–400 ms and one extra call per pause.

**Stopping the mic** sends a `{"type":"finish"}` control frame rather than closing the socket,
because Deepgram emits its last transcript only *after* `CloseStream`. The server closes the socket
once that final has been forwarded — hanging up first silently loses the last utterance.

### Text to speech

ElevenLabs **Flash v2.5**, fed incrementally so speech starts while the model is still generating.

`src/chunk.js` cuts the **first** chunk early, at a clause boundary, because time-to-first-audio is
what makes an assistant feel responsive; later chunks prefer sentence boundaries, which sound better.
Measured on real replies, audio starts around 20–25% into the stream instead of at the end. Clips are
scheduled back-to-back on a Web Audio timeline so there are no gaps between them.

> The original sentence regex required a terminator **followed by whitespace**, so the final sentence
> never matched mid-stream and a short one-sentence reply never matched at all — TTS only fired on
> the closing flush. That's the bug that made it feel like "generate everything, then speak".

### Private reasoning, and the option to stay silent

The fast model runs with `thinking: {type: "adaptive", display: "summarized"}`. Thinking deltas are
streamed to the client as `type: "thought"` and rendered dimmed and italic — they are **never sent to
text-to-speech**, so the model can deliberate without narrating out loud. Adaptive means it skips
thinking entirely when there's nothing to think about ("What is the capital of Japan?" → no reasoning,
just "Tokyo.").

It is also told it **does not have to reply**. Speech recognition regularly ends a turn mid-sentence,
and answering half a question is worse than waiting. If the transcript looks cut off, the model
reasons about why and ends its turn with no spoken text; the UI shows the reasoning and a
`stayed silent` badge. Verified: `"Come up with a hard math problem, and then"` produces
*"They're repeating themselves without finishing the thought. I should wait…"* and says nothing.

**The silence rule is scoped to genuine fragments only**, and says so explicitly. Left vague, it made
the model treat terseness as incompleteness: short complete instructions like "read a paragraph" or
"stop" got clarifying questions instead of action.

Three other prompt rules exist because their absence produced concrete failures:

- **Brevity is a default, not a limit.** "Keep replies to one to three sentences" directly contradicts
  "read me a long paragraph", and the model resolved the conflict by asking what was meant.
- **Compose, don't ask for source material.** "Read a paragraph" was interpreted as reading from a
  document that had been supplied, so it replied *"I don't have a text loaded to read from yet"*.
  The prompt now states there is nothing to load and that asking the user to paste text is wrong.
- **Expect speech-recognition errors and read through them.** "just any generic text" arrives as
  "Sending generic text.", and objecting to the literal wording is a worse failure than guessing the
  obvious intent.

The slow model gets the opposite instruction — it must **always** produce a conclusion, since it isn't
part of the live conversation and an empty turn throws the work away.

Because thinking shares the token budget with the reply, `max_tokens` is 4096 rather than 2048.
Thinking blocks are passed back to the API unchanged (they're part of `msg.content`, which is pushed
to history wholesale), and the truncation rewrite preserves them along with `tool_use` blocks.

### Interruption

The mic is **always live**. It is never closed and never fed silence — audio streams upstream
continuously, so the start of an interrupting sentence isn't clipped. (An earlier version muted the
upstream, which swallowed the first ~120 ms of every barge-in.) What's gated while the assistant
responds is *transcript delivery*: those transcripts are dropped rather than submitted, so playback
leaking through the speakers can't become the next turn.

**Barge-in requires speech, not just volume.** Deepgram's VAD can't be trusted during playback (it
hears the playback too), so this is measured locally, after the browser's echo canceller, and
classified by `src/vad.js` using four features per frame:

| Feature | Rejects |
| --- | --- |
| `rms` ≥ 0.06 | Distant conversation not aimed at the mic |
| `speechRatio` ≥ 0.5 — power inside 300–3400 Hz | Low thumps (door, desk), high-frequency hiss |
| `flatness` ≤ 0.3 — geometric/arithmetic mean of the spectrum | Broadband noise: clatter, key clicks |
| `dominance` ≤ 0.6 — share of energy in the loudest bin | Pure tones: beeps, alarms, feedback |

Nine consecutive qualifying frames (~150 ms) are required. Verified against real signals: 11/11
classified correctly, with close-mic speech accepted and every non-speech source rejected for a
different reason.

Flatness alone **cannot** separate a beep from a vowel — measured 0.0000 for a 1 kHz tone versus
0.0005 for voiced speech. That's why `dominance` exists.

Barge-in arms **only while audio is actually playing**. Armed during the thinking phase, room noise
"interrupts" a reply that hasn't started and truncates the turn to nothing.

**Interruption is exact, not estimated.** `/api/tts` uses ElevenLabs' `/stream/with-timestamps`,
which returns per-character start/end times (absolute across the response, so the arrays simply
concatenate; some lines carry audio only). Every clip is scheduled at a known point on the Web Audio
timeline, so `useTts.progress()` maps the audio clock back to a character offset in the model's text.
On interruption that offset does two things:

1. The visible reply is trimmed to what was heard and tagged `interrupted`.
2. `POST /api/interrupted` rewrites the assistant message in `shared.history` to the same prefix plus
   a marker. **Without this the model believes it said things you never heard** and refers back to
   them. `tool_use` and `thinking` blocks are preserved — dropping a `tool_use` breaks
   tool_use/tool_result pairing and 400s the next request.

**The truncation has to be deferred.** The assistant message is only pushed to history once its
stream finishes, so an interrupt arriving mid-stream has nothing to rewrite yet — rewriting "the last
assistant message" at that moment clobbers the **previous** reply and leaves the current one intact,
corrupting history in both directions. `shared.turnInFlight` and `shared.pendingTruncation` hold the
cut until `runFastChat`'s `finally` block, which applies it to the message that actually exists.

Two related traps on the client: barge-in must key off *audible* (`playing > 0`), not *speaking*
(`inflight > 0 || playing > 0`) — armed during synthesis, an interruption lands before any clip is
scheduled, playback position reads 0, and the whole reply looks unspoken. And the interrupt handler
freezes the cursor rather than replacing the text, so the spoken part stays on screen with the unheard
remainder dimmed. Deleting it made the entire reply vanish, including what you had already heard.

The cut uses the moment playback was cancelled, not the moment speech was detected ~150 ms earlier:
audio keeps playing until cancellation, so that's genuinely what reached your ears.

**Text is synced to audio.** The spoken prefix renders normally and the rest is dimmed, so reading
and hearing stay in step while you can still read ahead. On interruption the dimmed remainder
disappears, showing exactly what went unsaid. Driven by a 20 Hz poll of the audio clock rather than
per-frame, since it only needs to update text.

### Level meter

A proper **mel-scale spectrum**, not ad-hoc binning. `src/mel.js` builds a triangular mel filterbank:
bands evenly spaced on the mel scale (`2595·log10(1+f/700)`), each spanning its neighbours' centres so
adjacent filters overlap by half — the textbook MFCC front-end shape. Magnitudes come from
`getFloatFrequencyData`, which is already in dB, and band energy is averaged in **linear power**
before converting back to dB (averaging dB directly is wrong, since dB is logarithmic). Normalised
over a −85→−35 dB window with VU-style fast-attack/slow-release ballistics.

Validated numerically rather than by eye: Hz↔mel round-trips exactly, bands widen from 20 to 171 bins
as frequency rises, centres are monotonic, and a synthetic 1 kHz tone lights exactly two adjacent
bands.

The status word beside it distinguishes a **quiet room** from a **dead device** using two thresholds —
ambient in a quiet room still reads 0.005–0.04 RMS, while a disconnected input sits at zero. Saying
"check the device" when the device is fine is worse than saying nothing.

### Microphone conditioning

Raw mic audio goes through a conditioning chain in `src/useMic.js` before it reaches the PCM worklet:
high-pass at 90 Hz → low-pass at 7.5 kHz → compressor (−34 dB, 5:1, 3 ms attack) → makeup gain
(`VITE_MIC_GAIN`, default 4×) → brick-wall limiter at −3 dB. The worklet then applies a gentle,
smoothed noise gate, because lifting quiet speech also lifts the room floor.

**Browser auto-gain is off by default, deliberately.** AGC reacts to whatever is loudest, so
intermittent background bangs — a door, a pool table — make it duck, pushing quiet speech further down
exactly when it needs to come up. The compressor levels instead, with a fast limiter that flattens
transients without turning speech down.

Verified on a real recording (quiet speech, pool balls in the background). Overall RMS actually falls
~4 dB, because the loudest thing in the room was the noise, not the speaker — but the transcript
improves concretely:

| Raw | Conditioned |
| --- | --- |
| "Guess, make someone's **good data**" | "I guess it **makes sense**" |
| "I made it always **list**" | "I made it always **listen**" |

`scripts/condition.py` applies the same chain offline so you can A/B a recording before trusting it
live. Set `VITE_MIC_CONDITIONING=false` to bypass the chain entirely.

Two analyser taps exist for a reason: the **meter** reads the conditioned signal (what you see is what
is sent), while **barge-in** reads the raw mic, because `src/vad.js` thresholds are calibrated against
unprocessed levels and would false-fire on amplified background.

### Transcription accuracy

If transcription is poor, in rough order of impact:

**The model is not the problem.** Benchmarked on 14 s of clean speech, WER against the published
transcript: **nova-3 3.4%**, nova-2 6.9%, enhanced 10.3%, base 10.3% — and nova-3's single "error" is
actually correct (the speaker really does say "as, as"), so it is effectively perfect there.
ElevenLabs Scribe was also tested: acoustically comparable, but batch-only (no interim results) and it
transcribes filler words verbatim. nova-3 is the right choice; a UI dropdown lets you A/B the others
live. If nova-3 is transcribing badly, the audio is at fault.

1. **`STT_KEYTERMS`** — Nova-3 keyterm prompting biases decoding toward supplied vocabulary at no
   latency cost. The cheapest win for jargon, product names and acronyms. (Nova-2 called this
   `keywords`; the parameter was renamed.)
2. **`STT_ENDPOINTING_MS`** — raise it; short endpointing fragments speech and costs accuracy.
3. **`VITE_MIC_NOISE_SUPPRESSION` / `VITE_MIC_AGC`** — Chrome's noise suppression and auto-gain are
   tuned for humans on phone calls, not recognisers, and can smear phonemes. On by default; worth
   A/B-ing. **Never disable echo cancellation** — barge-in depends on it.
4. **The microphone itself is usually the ceiling.** A laptop mic picking up the room at a distance
   defeats any parameter tuning. A headset also makes barge-in reliable, since AEC leakage stops
   mattering.

---

## Layout

```
server/
  env.js       Loads .env — MUST be imported first (see Gotchas)
  index.js     Express + HTTP server, routes, WebSocket attach
  agents.js    Both model loops, their tools, and history truncation
  context.js   Shared context (history, insights, jobs) + event bus
  voice.js     Deepgram relay, turn detection, ElevenLabs proxy
src/
  App.jsx      UI: transcript, voice bar, shared-context panel
  useMic.js    Mic capture → PCM → /ws/stt, transcripts, barge-in
  useTts.js    Chunked synthesis, playback timeline, spoken-position tracking
  vad.js       Speech-vs-noise classifier for barge-in
  mel.js       Mel filterbank for the level meter
  chunk.js     Splits streaming text into speakable chunks
  MicMeter.jsx Live mel-spectrum canvas
  Toasts.jsx   Error toasts
public/
  pcm-worklet.js   Float32 → Int16 PCM conversion off the main thread
scripts/
  wav-to-pcm16k.py Resample a 16-bit mono WAV to 16 kHz raw PCM
  stt-smoke.mjs    Stream that PCM through /ws/stt and print transcripts
  stt-compare.mjs  Re-transcribe a recording under several Deepgram configs
  condition.py     Apply the mic conditioning chain offline, for A/B testing
```

### HTTP surface

| Route | Purpose |
| --- | --- |
| `POST /api/chat` | One user turn through the fast model; NDJSON stream back |
| `GET /api/events` | SSE: announcements and reasoning-job updates |
| `GET /api/state` | Snapshot: models, voice flags, insights, jobs |
| `POST /api/tts` | ElevenLabs proxy; returns base64 audio + character alignment |
| `POST /api/interrupted` | Reports what was heard; truncation is deferred if a turn is streaming |
| `ws /ws/stt` | Mic PCM in, transcripts out |

### Diagnosing a bad transcript

Don't guess whether the audio or the recogniser is at fault — capture what Deepgram actually
received and re-run it:

Recordings are written as playable WAV to `recordings/` (`STT_DUMP_DIR` to change).

```sh
STT_DUMP=1 npm run server      # logs the .wav path per connection, and each final transcript
# …talk, note the file, then:
export $(grep '^DEEPGRAM_API_KEY=' .env | xargs)
node scripts/stt-compare.mjs /tmp/botc-stt-….raw
```

`stt-compare.mjs` re-transcribes the same audio under several configurations (with/without keyterms,
with/without smart_format, nova-2 for comparison). **If every configuration returns the same wrong
words, the audio is the problem** — mic, distance, room, or browser DSP. If some are right, the
settings are. The dump is raw `s16le` mono at the rate in its filename, so you can also listen to it:

```sh
ffplay -f s16le -ar 16000 -ac 1 /tmp/botc-stt-….raw   # or import into any editor
```

### Testing without a microphone

```sh
curl -sL -o /tmp/sample.wav https://dpgr.am/spacewalk.wav
python3 scripts/wav-to-pcm16k.py /tmp/sample.wav /tmp/sample16k.raw 14
node scripts/stt-smoke.mjs /tmp/sample16k.raw                            # direct to API server
STT_HOST=localhost:5181 node scripts/stt-smoke.mjs /tmp/sample16k.raw    # through Vite's proxy
```

Test through the **proxy** when debugging the browser path — connecting straight to port 3001 bypasses
Vite entirely and hides proxy misconfiguration.

---

## Gotchas worth keeping

Each of these was a real bug, and each is silent rather than loud.

- **Vite must proxy `/ws` with `ws: true`.** Without it the browser's speech socket fails while the
  server is perfectly healthy, reporting only "Can't reach the speech service".
- **`server/env.js` must be the first import.** ES module imports are evaluated *before* the
  importing module's body, so an inline `.env` loader in `index.js` runs *after* `agents.js` has
  already read `process.env.SLOW_MODEL`.
- **`getUserMedia` constraints are mandatory unless wrapped in `ideal`.** A bare `channelCount: 1`
  rejects the whole request with `OverconstrainedError` on any device that won't do mono.
- **Truncating an assistant message must preserve `tool_use` and `thinking` blocks**, or the next
  request 400s on unpaired tool_use/tool_result.
- **The assistant message doesn't exist in history until its stream ends.** Anything that rewrites
  "the last assistant message" mid-stream hits the previous turn instead.
- **`useTts` must return a stable object identity** (it's in effect dependency lists) or the
  `EventSource` reconnects on every render.
- **Grid columns need `minmax(0, 1fr)`.** A wide flex child floors the column at its min-content width
  and pushes the sidebar off-screen.
- **No API key is bundled into the client.** Verified: `grep -r sk-ant dist/` is clean. Vite only
  exposes `VITE_`-prefixed variables.

---

## Known limitations

- **Shared context lives in memory** (`server/context.js`) — one conversation per server process.
  Swap for Redis or a database for multiple users or restart durability.
- **An always-live mic transcribes the whole room**, including conversations not directed at the
  assistant. Deepgram's `diarize=true` (speaker labels, then ignore everyone but the primary speaker)
  is the natural fix; a wake word or push-to-talk are the alternatives.
- **Barge-in classification is frame-local.** It rejects noise well, but cannot tell *your* speech
  from someone else's at similar volume. A stronger gate would additionally require a non-empty
  Deepgram transcript during playback. Headphones sidestep it.
- **No React error boundary** — an uncaught render error blanks the page rather than showing anything.
- **`speak()` for announcements** is deliberately excluded from the spoken-position cursor, so an
  announcement arriving mid-turn can't corrupt interruption offsets.
- **Utterances spoken while a turn is in flight are queued, not dropped.** Transcript delivery is
  gated only while audio is *audible* — gating on "busy" as well silently discarded anything said
  while the model was thinking, and `busyRef` then ignored it a second time.
- The slow model runs with `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`), so a
  request its safety classifiers decline is re-run on Anthropic's recommended substitute rather than
  failing. Fable 5's thinking blocks are passed back unchanged across tool turns, as the API requires.
