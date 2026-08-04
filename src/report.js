// Ships frontend errors to the server so they land in .run/logs/api.log next to
// everything else. Browser-only errors were previously invisible unless someone
// happened to have devtools open — the extended-thinking 400 sat in a toast for a
// while before anyone noticed it.

const ENDPOINT = "/api/client-error";
const seen = new Map(); // message -> last sent, to avoid flooding on a render loop
const DEDUPE_MS = 10000;

let installed = false;

export function reportError(message, extra = {}) {
  const text = String(message || "").slice(0, 2000);
  if (!text) return;

  const now = Date.now();
  const last = seen.get(text);
  if (last && now - last < DEDUPE_MS) return;
  seen.set(text, now);

  // Deliberately fire-and-forget, and deliberately swallowing: a failing error
  // reporter must never surface a second error and recurse.
  try {
    fetch(ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: text, url: location.pathname, ...extra }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* ignore */
  }
}

/**
 * Reports a non-error diagnostic to the server log (`[client:<kind>]`).
 *
 * Barge-in is the motivating case: it fires in the browser, from audio nobody
 * kept, and by the time the user says "it interrupted me for no reason" the
 * evidence is gone. Deliberately bypasses the dedupe window — each event carries
 * its own measurements and they are all worth keeping.
 */
export function reportEvent(kind, data = {}) {
  try {
    fetch(ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        kind,
        message: typeof data === "string" ? data : JSON.stringify(data),
        url: location.pathname,
      }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* ignore */
  }
}

export function installErrorReporting() {
  if (installed) return;
  installed = true;

  window.addEventListener("error", (e) => {
    reportError(e.message, {
      kind: "error",
      stack: e.error?.stack?.slice(0, 1200),
      at: e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : undefined,
    });
  });

  window.addEventListener("unhandledrejection", (e) => {
    const reason = e.reason;
    reportError(reason?.message || String(reason), {
      kind: "unhandledrejection",
      stack: reason?.stack?.slice(0, 1200),
    });
  });
}
