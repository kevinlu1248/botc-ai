import { useEffect, useRef, useState } from "react";

const STATE_MS = 400;

/**
 * Compact room strip — matches the original shared-context sidebar language.
 * Live JPEG is pulled one-at-a-time to avoid request pile-up / lag.
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
          setState((s) => ({ ...(s || {}), running: false, error: "offline" }));
        }
      }
    };

    const pullFrame = () => {
      if (!alive.current || inflight.current) return;
      if (!imgRef.current) return;
      inflight.current = true;
      const url = `/api/vision/frame.jpg?t=${Date.now()}`;
      const probe = new Image();
      probe.onload = () => {
        if (alive.current && imgRef.current) imgRef.current.src = url;
        inflight.current = false;
        if (alive.current) requestAnimationFrame(pullFrame);
      };
      probe.onerror = () => {
        inflight.current = false;
        if (alive.current) setTimeout(pullFrame, 250);
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
  const present = people.filter((p) => p.present);
  const looking = present.filter((p) => p.looking).length;
  const live = Boolean(state?.running);

  return (
    <div className="vision-card">
      <div className="vision-card-head">
        <span className="vision-kicker">Room</span>
        <span className={`vision-dot ${live ? "on" : "off"}`}>
          {live
            ? `${state.fps || 0} fps${looking ? ` · ${looking} looking` : ""}`
            : state?.error || "offline"}
        </span>
      </div>

      <div className={`vision-preview ${live ? "" : "empty"}`}>
        <img
          ref={imgRef}
          className="vision-frame"
          alt=""
          style={{ visibility: live ? "visible" : "hidden" }}
        />
        {!live && <span>{state?.error || "camera offline"}</span>}
      </div>

      {people.length > 0 && (
        <div className="vision-roster" title="People seen this session">
          {people.map((p) => {
            const initial = (p.label || p.pid || "?").trim().slice(0, 1).toUpperCase();
            return (
              <div
                key={p.pid}
                className={`vision-chip ${p.present ? "here" : ""} ${p.looking ? "looking" : ""}`}
                title={`${p.label || p.pid}${p.looking ? " · looking" : p.present ? " · in room" : " · away"}`}
              >
                {p.photo ? (
                  <img src={p.photo} alt="" />
                ) : (
                  <span className="vision-initial">{initial}</span>
                )}
                <span className="vision-chip-name">{p.label || p.pid}</span>
                {p.looking && <span className="vision-chip-mark" />}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
