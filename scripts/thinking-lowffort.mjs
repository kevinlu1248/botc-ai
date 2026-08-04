// The app runs adaptive+summarized thinking at effort:"low". Earlier probes passed
// only because low effort emitted NO thinking block, making the round-trip trivially
// valid. This forces thinking to appear at low effort and then replays it.
//
//   node scripts/thinking-lowffort.mjs
import "../server/env.js";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const MODEL = process.env.FAST_MODEL || "claude-sonnet-5";

const TOOLS = [
  {
    name: "start_deep_reasoning",
    description:
      "Hand a hard question to the background deep-reasoning model. Call this when the user " +
      "asks something that needs careful multi-step reasoning.",
    input_schema: {
      type: "object",
      properties: { question: { type: "string" } },
      required: ["question"],
    },
  },
];

const SYSTEM =
  "You are the voice of a low-latency assistant. You always think before answering. " +
  "Keep spoken replies to a sentence or two. Use start_deep_reasoning for hard questions.";

const PROMPTS = [
  "Should we shard Postgres or move to DynamoDB for 50TB of multi-tenant analytics?",
  "Is it better to run our own Kubernetes or use ECS Fargate for a 12-service backend?",
  "How should we design idempotency for a payments API that retries across regions?",
  "What's the right consistency model for a collaborative editor with offline support?",
];

let fails = 0;
let withThinking = 0;

for (const [i, prompt] of PROMPTS.entries()) {
  const req = {
    model: MODEL,
    max_tokens: 4096,
    thinking: { type: "adaptive", display: "summarized" },
    output_config: { effort: "low" },
    system: SYSTEM,
    tools: TOOLS,
    messages: [{ role: "user", content: prompt }],
  };

  const msg = await client.messages.stream(req).finalMessage();
  const shape = msg.content.map((b, j) => `${j}:${b.type}`).join(" ");
  const nThink = msg.content.filter((b) => b.type === "thinking").length;
  if (nThink) withThinking++;

  if (msg.stop_reason !== "tool_use") {
    console.log(`${i + 1}. shape=[${shape}] stop=${msg.stop_reason} — no tool call`);
    continue;
  }

  const results = msg.content
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "Job 1 started." }));

  try {
    await client.messages.stream({
      ...req,
      messages: [
        { role: "user", content: prompt },
        { role: "assistant", content: msg.content },
        { role: "user", content: results },
      ],
    }).finalMessage();
    console.log(`${i + 1}. shape=[${shape}] thinking=${nThink} -> round-trip OK`);
  } catch (e) {
    fails++;
    console.log(
      `${i + 1}. shape=[${shape}] thinking=${nThink} -> FAILED ${e.status}: ${
        e.error?.error?.message || e.message
      }`
    );
  }
}

console.log(`\n${withThinking}/${PROMPTS.length} responses contained thinking blocks; ${fails} round-trip failure(s)`);
