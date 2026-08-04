// Thinking-block signatures are bound to the model that produced them. The settings
// modal lets the conversation model change mid-conversation, which replays one
// model's thinking blocks to a different model.
//
//   node scripts/thinking-cross-model.mjs
import "../server/env.js";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

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

const SYSTEM = "You are the voice of a low-latency assistant. Use your tool for hard questions.";
const PROMPT = "Is it better to run our own Kubernetes or use ECS Fargate for a 12-service backend?";

const base = (model) => ({
  model,
  max_tokens: 4096,
  thinking: { type: "adaptive", display: "summarized" },
  output_config: { effort: "low" },
  system: SYSTEM,
  tools: TOOLS,
});

// Produce a real assistant turn that contains a thinking block.
async function produce(model) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const msg = await client.messages
      .stream({ ...base(model), messages: [{ role: "user", content: PROMPT }] })
      .finalMessage();
    if (msg.content.some((b) => b.type === "thinking")) return msg;
  }
  return null;
}

const PRODUCER = "claude-sonnet-5";
const origin = await produce(PRODUCER);
if (!origin) {
  console.log(`could not get a thinking block out of ${PRODUCER}; aborting`);
  process.exit(1);
}
const shape = origin.content.map((b, i) => `${i}:${b.type}`).join(" ");
console.log(`produced by ${PRODUCER}: shape=[${shape}] stop=${origin.stop_reason}\n`);

const followUp =
  origin.stop_reason === "tool_use"
    ? {
        role: "user",
        content: origin.content
          .filter((b) => b.type === "tool_use")
          .map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "Job 1 started." })),
      }
    : { role: "user", content: "What did you just say?" };

const messages = [
  { role: "user", content: PROMPT },
  { role: "assistant", content: origin.content },
  followUp,
];

for (const consumer of ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"]) {
  try {
    await client.messages.stream({ ...base(consumer), messages }).finalMessage();
    console.log(`OK    replay ${PRODUCER} thinking -> ${consumer}`);
  } catch (e) {
    console.log(`${e.status}   replay ${PRODUCER} thinking -> ${consumer}\n        ${e.error?.error?.message || e.message}`);
  }
}
