// Isolates the "thinking blocks cannot be modified" 400 on a tool round-trip.
//
// Replicates exactly what runFastChat does: adaptive+summarized thinking, low
// effort, one tool, then feed the tool_result back with the assistant message
// passed through verbatim. Run variants to find which knob causes the reject.
//
//   node scripts/thinking-roundtrip.mjs
import "../server/env.js";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const MODEL = process.env.FAST_MODEL || "claude-sonnet-5";

const TOOLS = [
  {
    name: "start_deep_reasoning",
    description: "Hand a hard question to a background deep-reasoning model.",
    input_schema: {
      type: "object",
      properties: { question: { type: "string" } },
      required: ["question"],
    },
  },
];

const PROMPT =
  "Compare Postgres and DynamoDB for multi-tenant analytics at 50TB. " +
  "This needs careful reasoning — use start_deep_reasoning.";

async function variant(label, { display, effort, stream: useStream }) {
  const thinking = { type: "adaptive" };
  if (display) thinking.display = display;

  const req = {
    model: MODEL,
    max_tokens: 4096,
    thinking,
    system: "You are a voice assistant. Use your tool for hard questions.",
    tools: TOOLS,
    messages: [{ role: "user", content: PROMPT }],
  };
  if (effort) req.output_config = { effort };

  let msg;
  if (useStream) {
    const s = client.messages.stream(req);
    msg = await s.finalMessage();
  } else {
    msg = await client.messages.create(req);
  }

  const shape = msg.content.map((b, i) => `${i}:${b.type}`).join(" ");
  const thinkBlocks = msg.content.filter((b) => b.type === "thinking");
  const sig = thinkBlocks[0]?.signature;

  if (msg.stop_reason !== "tool_use") {
    console.log(`${label}\n  shape=[${shape}] stop=${msg.stop_reason} — no tool call, skipping round-trip`);
    return;
  }

  // Round-trip: assistant message verbatim + tool_result, exactly like agents.js.
  const results = msg.content
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "Job started (id 1)." }));

  const messages = [
    { role: "user", content: PROMPT },
    { role: "assistant", content: msg.content },
    { role: "user", content: results },
  ];

  try {
    const s2 = client.messages.stream({ ...req, messages });
    const m2 = await s2.finalMessage();
    console.log(
      `${label}\n  shape=[${shape}] sig=${sig ? `${sig.length}ch` : "MISSING"} ` +
        `thinking_blocks=${thinkBlocks.length}\n  round-trip OK -> ${m2.stop_reason}`
    );
  } catch (e) {
    console.log(
      `${label}\n  shape=[${shape}] sig=${sig ? `${sig.length}ch` : "MISSING"} ` +
        `thinking_blocks=${thinkBlocks.length}\n  ROUND-TRIP FAILED: ${e.status} ${e.error?.error?.message || e.message}`
    );
  }
}

console.log(`model=${MODEL}\n`);
await variant("A. summarized + effort:low + stream  (what the app does)", {
  display: "summarized",
  effort: "low",
  stream: true,
});
await variant("B. summarized + effort:low + non-stream", {
  display: "summarized",
  effort: "low",
  stream: false,
});
await variant("C. summarized, no effort + stream", { display: "summarized", stream: true });
await variant("D. no display (raw), effort:low + stream", { effort: "low", stream: true });
