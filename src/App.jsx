import { useCallback, useEffect, useRef, useState } from "react";
import { useMic } from "./useMic.js";
import { useTts } from "./useTts.js";
import MicMeter from "./MicMeter.jsx";
import Toasts from "./Toasts.jsx";
import VisionPanel from "./VisionPanel.jsx";

const TOAST_MS = 7000;

// "claude-opus-5" -> "Opus 5"; "claude-haiku-4-5" -> "Haiku 4.5"
function shortModel(id) {
  if (!id) return "…";
  const m = id.match(/^claude-([a-z]+)-([\d-]+?)(?:-\d{8})?$/);
  if (!m) return id;
  const name = m[1][0].toUpperCase() + m[1].slice(1);
  return `${name} ${m[2].replace(/-/g, ".")}`;
}

export default function App() {
  const [turns, setTurns] = useState([]); // { role: 'user'|'assistant', text }
  const [busy, setBusy] = useState(false); // a chat request is in flight
  const [speaking, setSpeaking] = useState(false); // TTS is synthesizing or playing
  const [audible, setAudible] = useState(false); // sound is actually coming out
  const [interrupted, setInterrupted] = useState(false); // user talked over the assistant
  const [spokenChars, setSpokenChars] = useState(null); // chars of the last reply actually voiced
  const [voiceOut, setVoiceOut] = useState(true);
  const [jobs, setJobs] = useState([]);
  const [feed, setFeed] = useState([]);
  const [typed, setTyped] = useState("");
  const [toasts, setToasts] = useState([]); // { id, text, kind }
  const [meta, setMeta] = useState(null); // { models, voice, stt } from the server

  const tts = useTts(voiceOut, { onSpeakingChange: setSpeaking, onAudibleChange: setAudible });
  const busyRef = useRef(false);
  // Utterances that arrive while a turn is in flight. Without this they were
  // silently discarded: anything said while the model was thinking was lost.
  const queueRef = useRef([]);
  const sendRef = useRef(null);
  const interruptRef = useRef(null);
  // False once the user barges in: stop appending model deltas to the UI /
  // TTS. The HTTP stream is still drained so the server can finish the turn
  // and apply pendingTruncation to the right assistant message.
  const streamLiveRef = useRef(true);
  const transcriptRef = useRef(null);
  const toastId = useRef(0);

  const dismissToast = useCallback((id) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const pushToast = useCallback(
    (text, kind = "error") => {
      const id = ++toastId.current;
      setToasts((t) => {
        if (t.some((x) => x.text === text)) return t; // don't stack duplicates
        return [...t, { id, text, kind }];
      });
      setTimeout(() => dismissToast(id), TOAST_MS);
    },
    [dismissToast]
  );

  // Barge-in only (while audio is playing). Trims UI + server history to what
  // was actually heard. Do NOT call at the start of a normal send — that left
  // TTS stopped=true and dropped the whole next reply.
  const interrupt = useCallback(() => {
    if (!streamLiveRef.current) return; // already cut this stream
    streamLiveRef.current = false;

    const cut = tts.stopTurn();
    setInterrupted(true);

    const spoken = cut?.spoken ?? "";
    const index = cut?.index ?? 0;

    // Freeze the on-screen transcript to what was heard — not the full draft.
    setSpokenChars(index);
    setTurns((t) => {
      const next = [...t];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          text: spoken || (index > 0 ? last.text.slice(0, index) : last.text),
          cutOff: true,
        };
      }
      return next;
    });

    // Always notify the server so model history is truncated when the stream
    // finishes (pendingTruncation), even if index is 0.
    fetch("/api/interrupted", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ spoken }),
    }).catch(() => {});
  }, [tts]);
  interruptRef.current = interrupt;

  const send = useCallback(
    async (text, meta = {}) => {
      if (!text.trim()) return;
      // Re-arm streaming TTS for this turn. (Barge-in already stopTurn'd if needed.)
      tts.cancel();

      if (busyRef.current) {
        queueRef.current.push({ text, meta });
        return;
      }
      busyRef.current = true;
      setBusy(true);
      setInterrupted(false);
      setSpokenChars(null);
      streamLiveRef.current = true;

      const display = meta.displayText || text;
      const modelText = meta.modelText || text;
      setTurns((t) => [
        ...t,
        {
          role: "user",
          text: display,
          who: meta.who,
          looking: meta.looking,
        },
        { role: "assistant", text: "" },
      ]);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text: modelText }),
        });
        if (!res.ok || !res.body) throw new Error(`chat failed (${res.status})`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // Always drain the stream so the server can complete and apply
        // pendingTruncation to the assistant message it just pushed.
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.trim()) continue;
            let msg;
            try {
              msg = JSON.parse(line);
            } catch {
              continue;
            }
            if (msg.type === "delta") {
              if (!streamLiveRef.current) continue; // barge-in: freeze UI + TTS
              setTurns((t) => {
                const next = [...t];
                const last = next[next.length - 1];
                if (last?.cutOff) return t;
                next[next.length - 1] = { ...last, text: last.text + msg.text };
                return next;
              });
              // Spoken reply only — never feed thoughts (those are type "thought").
              tts.feed(msg.text);
            } else if (msg.type === "thought") {
              // Private reasoning: UI only, never TTS.
              if (!streamLiveRef.current) continue;
              setTurns((t) => {
                const next = [...t];
                const last = next[next.length - 1];
                if (last?.cutOff) return t;
                next[next.length - 1] = {
                  ...last,
                  thought: (last.thought || "") + msg.text,
                };
                return next;
              });
            } else if (msg.type === "job_started") {
              setJobs((j) => [...j, { ...msg.job, status: "running" }]);
            } else if (msg.type === "error") {
              pushToast(msg.message);
            }
          }
        }
        // Only flush TTS if this turn produced spoken text. A think-and-stay-silent
        // turn has empty buffer — flush would be a no-op, but be explicit.
        if (streamLiveRef.current) tts.flush();
      } catch (err) {
        pushToast(
          err instanceof TypeError
            ? "Can't reach the API server. Is it running on port 3001? (npm run server)"
            : err.message
        );
      } finally {
        busyRef.current = false;
        setBusy(false);
        const next = queueRef.current.shift();
        if (next) {
          if (typeof next === "string") sendRef.current?.(next);
          else sendRef.current?.(next.text, next.meta);
        }
      }
    },
    [tts, pushToast]
  );
  sendRef.current = send;

  // Drop pure hesitation so "um"/"uh" never become user turns (server also drops).
  const isFillerOnly = (t) =>
    /^(um+|uh+|er+|ah+|oh+|hmm+|mm+|mhm+|huh|like|so|well|yeah|yep|yup|nah|right|okay|ok)([.?!…,]*)$/i.test(
      t.trim()
    );

  // Voice finals are gated on looking-at-camera via the vision sidecar.
  const onVoiceFinal = useCallback(
    async (raw) => {
      const text = (raw || "").trim();
      if (!text) return;
      if (isFillerOnly(text)) return;
      // Do not interrupt here: a rejected (not looking) transcript must not
      // kill a playing reply. Barge-in is handled by onSpeechStart while audible.
      try {
        const res = await fetch("/api/vision/gate", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const gate = await res.json();
        if (!gate.ok) {
          if (gate.reason === "not_looking") {
            pushToast("Look at the camera to talk — transcript ignored", "warn");
          }
          return;
        }
        const who = gate.display?.who || "Someone";
        send(gate.text, {
          modelText: gate.text,
          displayText: `${who} (looking): ${gate.display?.said || text}`,
          who,
          looking: true,
        });
      } catch {
        // Vision offline — fall back to un-gated text so the app still works.
        send(text, { displayText: text });
      }
    },
    [send, pushToast]
  );

  const mic = useMic({
    onFinal: onVoiceFinal,
    // Only while the assistant is actually playing audio (useMic also gates this).
    onSpeechStart: () => interruptRef.current?.(),
  });

  // Half-duplex: while the assistant turn is in flight and speak-replies is on,
  // mute Deepgram for the whole turn — not only while audio is audible. That
  // stops (a) TTS echo and (b) accidental speech of "I'll stay quiet" reasoning
  // from being re-transcribed as the next user message. Local VAD still does
  // barge-in when playback is armed.
  useEffect(() => {
    const gate = voiceOut && busy && !interrupted;
    mic.setMuted(gate || (audible && !interrupted));
  }, [voiceOut, busy, audible, interrupted, mic.setMuted]);

  // Arm barge-in as soon as TTS is active (fetching or playing).
  useEffect(() => {
    mic.setPlaying((speaking || audible) && !interrupted);
  }, [speaking, audible, interrupted, mic.setPlaying]);

  useEffect(() => {
    if (mic.error) pushToast(mic.error);
  }, [mic.error, pushToast]);

  // Follow the audio clock so un-spoken text can be dimmed. Polled rather than
  // driven per-frame: 20Hz is plenty for text and costs one render each.
  useEffect(() => {
    if (!speaking) return;
    const id = setInterval(() => setSpokenChars(tts.progress()), 50);
    return () => clearInterval(id);
  }, [speaking, tts]);

  // Once playback finishes, everything was spoken. Clearing the cursor avoids
  // leaving the last few characters dimmed because the final poll landed early.
  useEffect(() => {
    if (!speaking && !interrupted) setSpokenChars(null);
  }, [speaking, interrupted]);

  // Voice-first: open the mic on load so you can just start talking. If the
  // permission prompt is declined, useMic reports it and we stay in text mode.
  const autoStarted = useRef(false);
  useEffect(() => {
    if (autoStarted.current) return;
    autoStarted.current = true;
    mic.start();
  }, [mic.start]);

  // Live updates pushed by the background reasoning model.
  useEffect(() => {
    const es = new EventSource("/api/events");
    es.onmessage = (event) => {
      const evt = JSON.parse(event.data);
      if (evt.type === "insight") {
        setFeed((f) => (f.some((x) => x.id === evt.insight.id) ? f : [evt.insight, ...f]));
        if (evt.speak) tts.speak(evt.insight.text);
      } else if (evt.type === "job_started") {
        setJobs((j) => (j.some((x) => x.id === evt.job.id) ? j : [...j, evt.job]));
      } else if (evt.type === "job_update") {
        setJobs((j) => j.map((x) => (x.id === evt.job.id ? evt.job : x)));
      }
    };
    return () => es.close();
  }, [tts]);

  useEffect(() => {
    fetch("/api/state")
      .then((r) => r.json())
      .then((s) => {
        setJobs(s.jobs);
        setFeed(s.insights.slice().reverse()); // newest first
        setMeta({ models: s.models, voice: s.voice, stt: s.stt });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [turns, busy]);

  const running = jobs.filter((j) => j.status === "running").length;

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="dot" />
          BOTC AI
        </div>
        <div className="models">
          <span className="tag fast">{shortModel(meta?.models?.fast)} · voice</span>
          <span className="tag slow">
            {shortModel(meta?.models?.slow)} · reasoning
            {running > 0 && <em>{running} running</em>}
          </span>
        </div>
      </header>

      <main>
        <section className="pane conversation">
          <div className="transcript" ref={transcriptRef}>
            {turns.length === 0 && (
              <p className="empty">
                The mic is live — look at the camera and talk, or type below. Hard questions go to
                the reasoning model on the right.
              </p>
            )}
            {turns.map((turn, i) => {
              const isLast = i === turns.length - 1;
              const dim =
                turn.role === "assistant" &&
                isLast &&
                spokenChars != null &&
                spokenChars < turn.text.length;
              const silent =
                turn.role === "assistant" &&
                !turn.text?.trim() &&
                !!turn.thought &&
                !(busy && isLast) &&
                !turn.cutOff;
              return (
                <div key={i} className={`turn ${turn.role}${silent ? " silent-turn" : ""}`}>
                  <span className="who">
                    {turn.role === "user"
                      ? turn.who
                        ? `${turn.who}${turn.looking ? " · looking" : ""}`
                        : "You"
                      : "Assistant"}
                  </span>
                  {turn.thought && (
                    <div className="thought-block">
                      <span className="thought-label">Thinking</span>
                      <p className="thought">{turn.thought}</p>
                    </div>
                  )}
                  {(turn.text?.trim() || (busy && isLast && turn.role === "assistant") || turn.cutOff) && (
                    <p className="speech">
                      {dim ? (
                        <>
                          {turn.text.slice(0, spokenChars)}
                          <span className="unspoken">{turn.text.slice(spokenChars)}</span>
                        </>
                      ) : (
                        turn.text || (busy && isLast ? "…" : "")
                      )}
                      {turn.cutOff && <span className="cutoff"> interrupted</span>}
                    </p>
                  )}
                  {silent && <span className="silent">stayed silent</span>}
                </div>
              );
            })}
            {mic.partial && (
              <div className="turn user partial">
                <span className="who">Listening…</span>
                <p>{mic.partial}</p>
              </div>
            )}
          </div>

          <div className="voicebar">
            <button
              className={`mic ${mic.listening ? "on" : ""}`}
              onClick={mic.listening ? mic.stop : () => mic.start()}
              title={mic.listening ? "Stop listening" : "Start listening"}
            >
              {mic.state === "connecting" ? "…" : mic.listening ? "◼" : "🎤"}
            </button>

            <MicMeter
              analyserRef={mic.analyserRef}
              active={mic.capturing}
              muted={mic.muted}
              status={busy ? "thinking" : speaking ? "speaking" : null}
            />

            {meta?.stt?.models?.length > 1 && (
              <select
                className="device"
                value={mic.sttModel || meta.stt.current}
                onChange={(e) => mic.selectSttModel(e.target.value)}
                title="Speech-recognition model — nova-3 benchmarks best; the others are for comparison"
              >
                {meta.stt.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            )}

            <select
              className="device"
              value={mic.deviceId}
              onChange={(e) => mic.selectDevice(e.target.value)}
              title={`Microphone: ${
                mic.devices.find((d) => d.deviceId === mic.deviceId)?.label || "system default"
              }`}
            >
              <option value="">Default mic</option>
              {mic.devices.map((d, i) => (
                <option key={d.deviceId || i} value={d.deviceId}>
                  {d.label || `Input ${i + 1}`}
                </option>
              ))}
            </select>
          </div>

          <div className="composer">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                // Typed text always submits (no looking gate). send() re-arms TTS.
                if (audible) interrupt();
                send(typed, { displayText: typed, modelText: typed });
                setTyped("");
              }}
            >
              <input
                value={typed}
                onChange={(e) => {
                  const v = e.target.value;
                  // Keystrokes only barge-in while the assistant is speaking.
                  if (audible && v.length > typed.length) interrupt();
                  setTyped(v);
                }}
                placeholder={
                  mic.listening
                    ? "Listening (look at camera) — or type anytime"
                    : "Type a message"
                }
              />
              <button type="submit" disabled={!typed.trim()}>
                Send
              </button>
            </form>
            <label className="speaker">
              <input
                type="checkbox"
                checked={voiceOut}
                onChange={(e) => {
                  setVoiceOut(e.target.checked);
                  if (!e.target.checked) tts.cancel();
                }}
              />
              Speak replies
            </label>
          </div>

        </section>

        <aside className="pane shared">
          <VisionPanel />

          <div className="shared-body">
            <h2>Shared context</h2>

            <div className="jobs">
              {jobs.length === 0 && <p className="empty small">No reasoning tasks yet.</p>}
              {jobs.map((job) => (
                <div key={job.id} className={`job ${job.status}`}>
                  <span className="status">{job.status}</span>
                  <p>{job.question}</p>
                </div>
              ))}
            </div>

            <div className="feed">
              {feed.map((item) => (
                <div key={item.id} className={`item ${item.source}`}>
                  <span className="kind">
                    {item.source} · {item.jobId}
                  </span>
                  <p>{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>

      <Toasts items={toasts} onDismiss={dismissToast} />
    </div>
  );
}
