import { useEffect, useRef, useState } from "react";

const STATE_MS = 400;

/**
 * Live room camera. Critical for lag: never pile up frame requests.
 * Only fetch the next JPEG after the previous one has loaded (or failed).
 */
export default function VisionPanel() {
  const [state, setState] = useState(null);
  const imgRef = useRef(null);
  const inflight = useRef(false);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let stateTimer;

    const tickState = async () => {
      try {
        const r = await fetch("/api/vision/state", { cache: "no-store" });
        if (!r.ok) throw new Error("offline");
        const j = await r.json();
        if (alive.current) setState(j);
      } catch {
        if (alive.current) {
          setState((s) => ({ ...(s || {}), running: false, error: "vision offline" }));
        }
      }
    };

    const pullFrame = () => {
      if (!alive.current || inflight.current) return;
      const img = imgRef.current;
      if (!img) return;
      inflight.current = true;
      const url = `/api/vision/frame.jpg?t=${Date.now()}`;
      const probe = new Image();
      probe.onload = () => {
        if (alive.current && imgRef.current) imgRef.current.src = url;
        inflight.current = false;
        // Immediately request the next frame — no fixed interval backlog.
        if (alive.current) requestAnimationFrame(pullFrame);
      };
      probe.onerror = () => {
        inflight.current = false;
        if (alive.current) setTimeout(pullFrame, 200);
      };
      probe.src = url;
    };

    tickState();
    stateTimer = setInterval(tickState, STATE_MS);
    pullFrame();

    return () => {
      alive.current = false;
      clearInterval(stateTimer);
    };
  }, []);

  const people = state?.people || [];
  const looking = (state?.looking || []).length;
  const live = Boolean(state?.running);

  return (
    <div className="vision-panel">
      <div className="vision-head">
        <h2>Room</h2>
        <span className={`vision-status ${live ? "on" : "off"}`}>
          {live ? `● ${state.fps || 0} fps · looking ${looking}` : "○ offline"}
        </span>
      </div>

      <div className="vision-frame-wrap">
        <img
          ref={imgRef}
          className="vision-frame"
          alt="Live room camera"
          style={{ display: live ? "block" : "none" }}
        />
        {!live && (
          <div className="vision-placeholder">
            {state?.error || "Start the vision server (npm run vision)"}
          </div>
        )}
      </div>

      <h3 className="vision-sub">People this session</h3>
      <div className="vision-people">
        {people.length === 0 && (
          <p className="empty small">No one identified yet — face the camera.</p>
        )}
        {people.map((p) => {
          const initial = (p.label || p.pid || "?").trim().slice(0, 1).toUpperCase();
          return (
            <div
              key={p.pid}
              className={`vision-person ${p.present ? "here" : ""} ${p.looking ? "looking" : ""}`}
            >
              {p.photo ? (
                <img className="vision-face" src={p.photo} alt="" />
              ) : (
                <div className="vision-face placeholder">{initial}</div>
              )}
              <div className="vision-meta">
                <div className="vision-name">{p.label || p.pid}</div>
                <div className="vision-id">
                  {p.pid}
                  {p.looking_score != null && p.present
                    ? ` · look ${Number(p.looking_score).toFixed(2)}`
                    : ""}
                </div>
              </div>
              <div className="vision-badges">
                <span className="vbadge">{p.present ? "in room" : "away"}</span>
                {p.present && (
                  <span className={`vbadge ${p.looking ? "look" : ""}`}>
                    {p.looking ? "looking" : "not looking"}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <p className="vision-hint">
        Speech is only transcribed when someone is looking at the camera. Attribution is
        vision-based (who is looking), not audio diarization. Any text interrupts the assistant.
      </p>
    </div>
  );
}
