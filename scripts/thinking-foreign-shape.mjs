// Sends Anthropic an assistant turn it never produced: [text, tool_use] with no
// thinking block — exactly what server/gemini.js writes into shared.history — and
// checks whether that is the "thinking blocks cannot be modified" 400.
//
//   node scripts/thinking-foreign-shape.mjs
import "../server/env.js";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const MODEL = process.env.FAST_MODEL || "claude-sonnet-5";

const TOOLS = [
  {
    name: "start_deep_reasoning",
    description: "Hand a hard question to the background deep-reasoning model.",
    input_schema: {
      type: "object",
      properties: { question: { type: "string" } },
      required: ["question"],
    },
  },
];

const ID = "gem_1754000000000_0";
const PROMPT = "Should we shard Postgres or move to DynamoDB for 50TB of analytics?";

// The shape runGeminiChat pushes: optional text, then tool_use. No thinking block.
const GEMINI_TURN = [
  { type: "text", text: "Good question — let me get the reasoning model on it." },
  { type: "tool_use", id: ID, name: "start_deep_reasoning", input: { question: PROMPT } },
];

const CASES = {
  "gemini turn is the LATEST assistant message (followed by tool_result)": [
    { role: "user", content: PROMPT },
    { role: "assistant", content: GEMINI_TURN },
    { role: "user", content: [{ type: "tool_result", tool_use_id: ID, content: "Job 1 started." }] },
  ],
  "gemini turn followed by tool_result + a new user message": [
    { role: "user", content: PROMPT },
    { role: "assistant", content: GEMINI_TURN },
    { role: "user", content: [{ type: "tool_result", tool_use_id: ID, content: "Job 1 started." }] },
    { role: "user", content: "What did you just say?" },
  ],
  "gemini turn buried behind a later plain-text assistant turn": [
    { role: "user", content: PROMPT },
    { role: "assistant", content: GEMINI_TURN },
    { role: "user", content: [{ type: "tool_result", tool_use_id: ID, content: "Job 1 started." }] },
    { role: "assistant", content: "I've kicked off deeper analysis on that." },
    { role: "user", content: "What did you just say?" },
  ],
  "tool_use only (no text block), latest assistant message": [
    { role: "user", content: PROMPT },
    { role: "assistant", content: [GEMINI_TURN[1]] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: ID, content: "Job 1 started." }] },
  ],
};

for (const [label, messages] of Object.entries(CASES)) {
  try {
    await client.messages.stream({
      model: MODEL,
      max_tokens: 1024,
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "low" },
      system: "You are a voice assistant.",
      tools: TOOLS,
      messages,
    }).finalMessage();
    console.log(`OK    ${label}`);
  } catch (e) {
    console.log(`400   ${label}\n        ${e.error?.error?.message || e.message}`);
  }
}
