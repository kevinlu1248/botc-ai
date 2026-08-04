// Scenario matrix for the "thinking blocks cannot be modified" 400.
//
// shared.history is global mutable state and lives in memory, so each scenario
// gets a freshly restarted server — otherwise scenario N inherits N-1's history
// and a pass means nothing.
//
//   node scripts/repro-matrix.mjs [scenarioName]
import { spawn, execSync } from "node:child_process";
import { readFileSync, existsSync, rmSync } from "node:fs";

const HOST = "http://localhost:3001";
const only = process.argv[2];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function startServer() {
  try {
    execSync("lsof -ti:3001 | xargs kill -9", { stdio: "ignore" });
  } catch {
    /* nothing listening */
  }
  rmSync(".run/last-bad-history.json", { force: true });
  const proc = spawn("node", ["server/index.js"], {
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });
  let log = "";
  proc.stdout.on("data", (d) => (log += d));
  proc.stderr.on("data", (d) => (log += d));

  for (let i = 0; i < 60; i++) {
    await sleep(250);
    try {
      await fetch(`${HOST}/api/state`);
      return { proc, log: () => log };
    } catch {
      /* not up yet */
    }
  }
  throw new Error("server never came up");
}

async function setFast(model) {
  await fetch(`${HOST}/api/settings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ fast: model }),
  });
}

/** Streams /api/chat. onDelta lets a scenario react mid-stream. */
async function chat(text, { onFirstDelta = null } = {}) {
  const res = await fetch(`${HOST}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let out = "";
  let error = null;
  let fired = false;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      let evt;
      try {
        evt = JSON.parse(line);
      } catch {
        continue;
      }
      if (evt.type === "delta") out += evt.text;
      if (evt.type === "error") error = evt.message;
    }
    if (onFirstDelta && !fired && out.length > 20) {
      fired = true;
      await onFirstDelta(out);
    }
  }
  return { out, error };
}

const interrupt = (spoken) =>
  fetch(`${HOST}/api/interrupted`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ spoken }),
  }).then((r) => r.json());

const HARD =
  "Should we shard Postgres or move to DynamoDB for 50TB of multi-tenant analytics? Think it through properly.";

const SCENARIOS = {
  // Baseline: one interrupt during a tool-calling turn, then follow-ups.
  "single-interrupt": async () => {
    const a = await chat(HARD, { onFirstDelta: (s) => interrupt(s.slice(0, 40)) });
    const b = await chat("What did you just say?");
    return [a, b];
  },

  // Two turns in flight at once. The queued-utterance path can do this: a second
  // final arrives while the first turn is still streaming.
  concurrent: async () => {
    const a = chat(HARD);
    await sleep(700);
    const b = chat("Actually wait, what about Clickhouse?");
    const [ra, rb] = await Promise.all([a, b]);
    const c = await chat("Summarise where we landed.");
    return [ra, rb, c];
  },

  // Concurrency plus an interrupt — the queue-flush-during-speech case.
  "concurrent-interrupt": async () => {
    const a = chat(HARD, { onFirstDelta: (s) => interrupt(s.slice(0, 40)) });
    await sleep(700);
    const b = chat("What about Clickhouse?");
    const [ra, rb] = await Promise.all([a, b]);
    const c = await chat("Summarise where we landed.");
    return [ra, rb, c];
  },

  // Interrupt twice in one turn (user keeps talking over it).
  "double-interrupt": async () => {
    const a = await chat(HARD, {
      onFirstDelta: async (s) => {
        await interrupt(s.slice(0, 20));
        await interrupt(s.slice(0, 30));
      },
    });
    const b = await chat("What did you just say?");
    return [a, b];
  },

  // Haiku is offered in the settings dropdown but rejects adaptive thinking.
  "haiku-selected": async () => {
    await setFast("claude-haiku-4-5");
    const a = await chat("Hey, quick question — what's 2+2?");
    return [a];
  },

  // Mirrors the user's session: several turns, the conversation model switched
  // between turns, interrupts landing mid-tool-call.
  "model-switch-storm": async () => {
    const out = [];
    out.push(await chat(HARD, { onFirstDelta: (s) => interrupt(s.slice(0, 30)) }));
    await setFast("gemini-3.6-flash");
    out.push(await chat("What about Clickhouse instead?"));
    await setFast("claude-sonnet-5");
    out.push(await chat("Okay, so which one would you pick?", {
      onFirstDelta: (s) => interrupt(s.slice(0, 25)),
    }));
    await setFast("gemini-3.6-flash");
    out.push(await chat("And what did you say before that?"));
    await setFast("claude-sonnet-5");
    out.push(await chat("Summarise the whole conversation."));
    return out;
  },

  // Gemini makes a tool call (assistant message has NO thinking block), then the
  // user switches to Claude, which replays that turn under thinking.
  "gemini-tool-then-claude": async () => {
    await setFast("gemini-3.6-flash");
    const a = await chat(HARD);
    await setFast("claude-sonnet-5");
    const b = await chat("What did you just say?");
    const c = await chat("And why?");
    return [a, b, c];
  },
};

const names = only ? [only] : Object.keys(SCENARIOS);
const summary = [];

for (const name of names) {
  const { proc } = await startServer();
  process.stdout.write(`\n=== ${name} ===\n`);
  let results = [];
  let thrown = null;
  try {
    results = await SCENARIOS[name]();
  } catch (e) {
    thrown = e.message;
  }
  results.forEach((r, i) => {
    console.log(`  turn ${i + 1}: ${r.error ? `ERROR ${r.error.slice(0, 150)}` : `ok ${JSON.stringify(r.out.slice(0, 60))}`}`);
  });
  if (thrown) console.log(`  threw: ${thrown}`);

  const failed = results.filter((r) => r.error).length;
  let dump = null;
  if (existsSync(".run/last-bad-history.json")) {
    const d = JSON.parse(readFileSync(".run/last-bad-history.json", "utf8"));
    dump = d.history.map((m, i) => {
      const c = m.content;
      return `    ${i} ${m.role} ${
        typeof c === "string" ? `str(${JSON.stringify(c.slice(0, 30))})` : `[${c.map((b, j) => `${j}:${b.type}`).join(" ")}]`
      }`;
    }).join("\n");
    console.log(`  history at failure:\n${dump}`);
  }
  summary.push({ name, failed, thrown, dump: Boolean(dump) });
  proc.kill("SIGKILL");
  await sleep(400);
}

console.log("\n===== SUMMARY =====");
for (const s of summary) {
  console.log(`${s.failed || s.thrown ? "FAIL" : "pass"}  ${s.name}${s.thrown ? ` (threw: ${s.thrown})` : ""}`);
}
