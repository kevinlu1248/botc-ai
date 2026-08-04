// Pins down the exact trigger for
//   "`thinking` ... blocks in the latest assistant message cannot be modified"
// The reported error pointed at a tool_use block (content.1), so validation is
// strict only when the assistant turn ends in tool_use and is continued by a
// tool_result. Mutate a real [thinking, tool_use] turn every plausible way and see
// which one the API rejects with that message.
//
//   node scripts/thinking-tool-continuation.mjs
import "../server/env.js";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const MODEL = process.argv[2] || "claude-sonnet-5";

const TOOLS = [
  {
    name: "start_deep_reasoning",
    description:
      "Hand a hard question to the background deep-reasoning model. Use for anything " +
      "needing careful multi-step reasoning.",
    input_schema: {
      type: "object",
      properties: { question: { type: "string" } },
      required: ["question"],
    },
  },
];

const PROMPT =
  "Should we shard Postgres or move to DynamoDB for 50TB of multi-tenant analytics? " +
  "Hand it to the reasoning model.";

const send = (messages) =>
  client.messages
    .stream({
      model: MODEL,
      max_tokens: 4096,
      thinking: { type: "adaptive", display: "summarized" },
      // No effort cap here: the goal is to characterise the API's validation rule,
      // and low effort usually skips thinking entirely on a tool-calling turn.
      system:
        "You are a voice assistant. Think carefully first, then use " +
        "start_deep_reasoning for hard questions.",
      tools: TOOLS,
      messages,
    })
    .finalMessage();

// Need a turn that has BOTH a thinking block and a tool_use block.
let origin = null;
for (let i = 0; i < 8 && !origin; i++) {
  const msg = await send([{ role: "user", content: PROMPT }]);
  const hasThinking = msg.content.some((b) => b.type === "thinking");
  if (msg.stop_reason === "tool_use" && hasThinking) origin = msg;
}
if (!origin) {
  console.log("could not obtain a [thinking, tool_use] turn; aborting");
  process.exit(1);
}

console.log(`origin shape=[${origin.content.map((b, i) => `${i}:${b.type}`).join(" ")}]\n`);

const toolResult = {
  role: "user",
  content: origin.content
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "Job 1 started." })),
};

const think = origin.content.find((b) => b.type === "thinking");
const other = await send([{ role: "user", content: "Name one tradeoff of microservices. Think first." }]);
const foreign = other.content.find((b) => b.type === "thinking");

const MUTATIONS = {
  "unmodified (control)": origin.content,
  "thinking block removed entirely": origin.content.filter((b) => b.type !== "thinking"),
  "thinking text altered, signature kept": origin.content.map((b) =>
    b.type === "thinking" ? { ...b, thinking: `${b.thinking} (edited)` } : b
  ),
  "thinking text emptied": origin.content.map((b) => (b.type === "thinking" ? { ...b, thinking: "" } : b)),
  "signature stripped": origin.content.map((b) => {
    if (b.type !== "thinking") return b;
    const { signature, ...rest } = b;
    return rest;
  }),
  "thinking block from another turn": foreign
    ? origin.content.map((b) => (b.type === "thinking" ? foreign : b))
    : null,
  "blocks reordered (tool_use first)": [...origin.content].reverse(),
  "extra text block inserted before tool_use": think
    ? [think, { type: "text", text: "Let me look into that." }, ...origin.content.filter((b) => b.type === "tool_use")]
    : null,
};

for (const [label, content] of Object.entries(MUTATIONS)) {
  if (!content) continue;
  let detail = "";
  try {
    await send([{ role: "user", content: PROMPT }, { role: "assistant", content }, toolResult]);
  } catch (e) {
    detail = e.error?.error?.message || e.message;
  }
  const exact = /cannot be modified/.test(detail);
  const tag = exact ? ">>> REPORTED ERROR" : detail ? "other 400        " : "accepted         ";
  console.log(`${tag}  ${label}${detail ? `\n                     ${detail.slice(0, 130)}` : ""}`);
}
