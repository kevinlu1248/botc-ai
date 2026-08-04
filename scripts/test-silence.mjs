// The model hears everything and decides for itself whether to answer, declining
// with the NO_RESPONSE sentinel. This checks that contract:
//
//   - declining emits a `no_response` event, not speech
//   - the sentinel NEVER appears in a delta (deltas are what reach text-to-speech)
//   - a real question still gets a spoken answer
//   - stop-like utterances now reach the model rather than being swallowed
//
// Each case runs on a freshly restarted server: shared.history is in memory, and a
// run of declined turns in the transcript biases whatever comes next.
//
//   node scripts/test-silence.mjs
import { spawn, execSync } from "node:child_process";
import { NO_RESPONSE } from "../server/agents.js";

const HOST = "http://localhost:3001";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function startServer() {
  try {
    execSync("lsof -ti:3001 | xargs kill -9", { stdio: "ignore" });
  } catch {
    /* nothing listening */
  }
  const proc = spawn("node", ["server/index.js"], { stdio: "ignore" });
  for (let i = 0; i < 60; i++) {
    await sleep(250);
    try {
      await fetch(`${HOST}/api/state`);
      return proc;
    } catch {
      /* not up yet */
    }
  }
  throw new Error("server never came up");
}

async function chat(text) {
  const res = await fetch(`${HOST}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let spoken = "";
  let thought = "";
  let declined = false;
  let error = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const l of lines) {
      if (!l.trim()) continue;
      let e;
      try {
        e = JSON.parse(l);
      } catch {
        continue;
      }
      if (e.type === "delta") spoken += e.text;
      if (e.type === "thought") thought += e.text;
      if (e.type === "no_response") declined = true;
      if (e.type === "error") error = e.message;
    }
  }
  return { spoken: spoken.trim(), thought: thought.trim(), declined, error };
}

const CASES = [
  // Reach the model now — nothing is swallowed client-side.
  { text: "Stop.", want: "declined" },
  { text: "Stop. Stop.", want: "declined" },
  { text: "Stop talking.", want: "declined" },
  { text: "wait", want: "declined" },
  { text: "ok cool", want: "declined" },
  { text: "nice", want: "declined" },
  { text: "thanks", want: "declined" },
  { text: "yeah, I see what you mean", want: "declined" },
  { text: "Tell me a joke.", want: "speaks" },
  { text: "Say something.", want: "speaks" },
  { text: "What's the capital of Japan?", want: "speaks" },
  { text: "Read me a short paragraph about the sea.", want: "speaks" },
];

console.log(`sentinel = ${NO_RESPONSE}\n`);
let fails = 0;
let leaked = 0;

for (const c of CASES) {
  const proc = await startServer();
  const r = await chat(c.text);
  proc.kill("SIGKILL");
  await sleep(250);

  // The sentinel reaching a delta would mean it reached the speakers.
  const sentinelLeak = r.spoken.toUpperCase().includes(NO_RESPONSE);
  if (sentinelLeak) leaked++;

  const ok =
    (c.want === "declined" ? r.declined && !r.spoken : !r.declined && r.spoken.length > 0) &&
    !sentinelLeak;
  if (!ok) fails++;

  console.log(
    `${ok ? "PASS" : "FAIL"}  want=${c.want.padEnd(8)} ${JSON.stringify(c.text).padEnd(46)}` +
      ` declined=${String(r.declined).padEnd(5)} spoken=${JSON.stringify(r.spoken.slice(0, 40))}` +
      `${sentinelLeak ? "  <-- SENTINEL LEAKED TO TTS" : ""}${r.error ? `  ERROR ${r.error.slice(0, 50)}` : ""}`
  );
  if (c.want === "declined" && r.declined && !r.thought) {
    console.log("        note: declined without producing thinking (adaptive thinking skipped it)");
  }
}

console.log(`\n${CASES.length - fails}/${CASES.length} passed; sentinel leaks to TTS: ${leaked}`);
process.exit(fails ? 1 : 0);
