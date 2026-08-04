import { useCallback, useEffect, useState } from "react";

// Every model in one place. The server keeps these in server/settings.js rather
// than as module constants, so a change here applies to the next turn without
// restarting anything. The STT model is the exception: it is fixed for the life of
// a Deepgram connection, so changing it reconnects the mic.
export default function Settings({
  open,
  onClose,
  onSttChange,
  vadBackend,
  devices = [],
  deviceId = "",
  onDeviceChange,
}) {
  const [data, setData] = useState(null);
  const [saving, setSaving] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!open) return;
    fetch("/api/settings")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message));
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const change = useCallback(
    async (key, value) => {
      setSaving(key);
      setError(null);
      // Optimistic: the select should not snap back while the POST is in flight.
      setData((d) => (d ? { ...d, settings: { ...d.settings, [key]: value } } : d));
      try {
        const res = await fetch("/api/settings", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ [key]: value }),
        });
        const out = await res.json();
        setData((d) => (d ? { ...d, settings: out.settings } : d));
        if (key === "stt") onSttChange?.(value);
      } catch (e) {
        setError(e.message);
      } finally {
        setSaving(null);
      }
    },
    [onSttChange]
  );

  if (!open) return null;
  const s = data?.settings;
  const o = data?.options;

  const row = (key, label, options, note) => (
    <label className="setting" key={key}>
      <span className="setting-label">
        {label}
        {saving === key && <em> saving…</em>}
      </span>
      <select value={s[key]} onChange={(e) => change(key, e.target.value)}>
        {options.map((opt) =>
          typeof opt === "string" ? (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ) : (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          )
        )}
      </select>
      {note && <span className="setting-note">{note}</span>}
    </label>
  );

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Models</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {error && <p className="setting-error">{error}</p>}
        {!s && !error && <p className="setting-note">Loading…</p>}

        {s && o && (
          <div className="settings-grid">
            {row("fast", "Conversation model", o.fast, "Replies in the live conversation")}
            {row("slow", "Reasoning model", o.slow, "Background deep reasoning")}
            {row("stt", "Speech to text", o.stt, "Reconnects the mic; nova-3 benchmarks best")}
            {row("tts", "Speech synthesis", o.tts, "flash = lowest latency")}
            {row("voice", "Spoken voice", o.voice, "Default voices only — library voices need a paid plan")}

            <label className="setting">
              <span className="setting-label">Microphone</span>
              <select value={deviceId} onChange={(e) => onDeviceChange?.(e.target.value)}>
                <option value="">System default</option>
                {devices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
                    {d.label || `Input ${i + 1}`}
                  </option>
                ))}
              </select>
              <span className="setting-note">Switching reconnects the audio graph</span>
            </label>

            <div className="setting">
              <span className="setting-label">Voice activity detection</span>
              <span className="setting-static">
                {vadBackend === "silero" ? "Silero VAD v5 (2.2 MB, local)" : vadBackend}
              </span>
              <span className="setting-note">
                Fixed — barge-in hard-fails rather than falling back
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
