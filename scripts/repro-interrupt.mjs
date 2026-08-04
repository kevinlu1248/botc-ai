// Drives the real server through the interrupt path that produced
// "thinking blocks ... cannot be modified", then sends a follow-up turn — the
// follow-up is where the poisoned history gets replayed and rejected.
//
//   node scripts/repro-interrupt.mjs
const HOST = process.env.API_HOST || "http://localhost:3001";

const shapes = (history) =>
  history.map((m, i) => {
    const c = m.content;
    const body =
      typeof c === "string"
        ? `str(${JSON.stringify(c.slice(0, 40))})`
        : `[${c.map((b, j) => `${j}:${b.type}`).join(" ")}]`;
    return `  ${i} ${m.role} ${body}`;
  }).join("\n");

async function state() {
  const r = await fetch(`${HOST}/api/state`).then((r) => r.json());
  return r;
}

/** Streams /api/chat, optionally firing an interrupt after `interruptAfterMs`. */
async function chat(text, { interruptAfterMs = null, spoken = "" } = {}) {
  const res = await fetch(`${HOST}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });

  let fired = false;
  const started = Date.now();
  let out = "";
  let error = null;

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const evt = JSON.parse(line);
      if (evt.type === "delta") out += evt.text;
      if (evt.type === "error") error = evt.message;
    }
    if (interruptAfterMs !== null && !fired && Date.now() - started > interruptAfterMs) {
      fired = true;
      const r = await fetch(`${HOST}/api/interrupted`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ spoken: spoken || out.slice(0, 40) }),
      }).then((r) => r.json());
      console.log(`  [interrupt sent mid-stream] -> ${JSON.stringify(r)}`);
    }
  }
  return { out, error };
}

console.log(`host=${HOST}`);
const s0 = await state();
console.log(`fast=${s0.models.fast} slow=${s0.models.slow}\n`);

// Turn 1: a question hard enough to trigger the reasoning tool, interrupted while
// it is still speaking — this is the exact sequence from the user's session.
console.log("turn 1: tool-triggering question, interrupted mid-stream");
const t1 = await chat(
  "Should we shard Postgres or move to DynamoDB for 50TB of multi-tenant analytics? Think it through properly.",
  { interruptAfterMs: 900 }
);
console.log(`  spoken: ${JSON.stringify(t1.out.slice(0, 90))}`);
console.log(`  error: ${t1.error || "none"}\n`);

// Turn 2: the follow-up that replays the truncated turn.
console.log("turn 2: follow-up (replays the truncated turn)");
const t2 = await chat("What did you just say?");
console.log(`  reply: ${JSON.stringify(t2.out.slice(0, 90))}`);
console.log(`  error: ${t2.error || "none"}\n`);

// Turn 3: one more, in case poisoning only shows up once the bad turn is buried.
console.log("turn 3: another follow-up");
const t3 = await chat("Okay, and briefly why?");
console.log(`  reply: ${JSON.stringify(t3.out.slice(0, 90))}`);
console.log(`  error: ${t3.error || "none"}\n`);

const bad = [t1.error, t2.error, t3.error].filter(Boolean);
console.log(bad.length ? `RESULT: ${bad.length} turn(s) failed` : "RESULT: all turns OK");
