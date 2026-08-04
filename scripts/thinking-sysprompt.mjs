// Hypothesis: a thinking signature is bound to the request it was produced in.
// agents.js builds `system: fastSystem()` fresh per request, and fastSystem()
// embeds contextBlock() — which CHANGES mid-turn because runFastTool creates a
// job. So the tool round-trip replays thinking blocks under a different system
// prompt, and Anthropic rejects them as modified.
//
//   node scripts/thinking-sysprompt.mjs
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

const SYS_BEFORE = (jobs) =>
  "You are a voice assistant. Use start_deep_reasoning for hard questions.\n\n" +
  `<shared_context>\n  Background jobs: ${jobs}\n</shared_context>`;

const PROMPT =
  "Design a globally consistent multi-region ledger with sub-100ms writes. " +
  "Reason carefully, then hand it to start_deep_reasoning.";

async function run(label, { systemOnRoundTrip }) {
  const base = {
    model: MODEL,
    max_tokens: 8000,
    thinking: { type: "adaptive", display: "summarized" },
    tools: TOOLS,
  };

  const msg = await client.messages.stream({
    ...base,
    system: SYS_BEFORE("none"),
    messages: [{ role: "user", content: PROMPT }],
  }).finalMessage();

  const shape = msg.content.map((b, i) => `${i}:${b.type}`).join(" ");
  const nThink = msg.content.filter((b) => b.type === "thinking").length;

  if (msg.stop_reason !== "tool_use" || nThink === 0) {
    console.log(`${label}\n  shape=[${shape}] stop=${msg.stop_reason} — need thinking+tool_use, retrying not attempted`);
    return null;
  }

  const results = msg.content
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "Job 1 started." }));

  try {
    await client.messages.stream({
      ...base,
      system: systemOnRoundTrip,
      messages: [
        { role: "user", content: PROMPT },
        { role: "assistant", content: msg.content },
        { role: "user", content: results },
      ],
    }).finalMessage();
    console.log(`${label}\n  shape=[${shape}] thinking=${nThink}\n  OK`);
    return true;
  } catch (e) {
    console.log(
      `${label}\n  shape=[${shape}] thinking=${nThink}\n  FAILED ${e.status}: ${
        e.error?.error?.message || e.message
      }`
    );
    return false;
  }
}

console.log(`model=${MODEL}\n`);
// Identical system prompt on the round-trip — the correct behaviour.
await run("E. same system prompt on round-trip", { systemOnRoundTrip: SYS_BEFORE("none") });
// System prompt mutated by the job the tool just created — what the app does.
await run("F. system prompt changed mid-turn (job added)", {
  systemOnRoundTrip: SYS_BEFORE("#1 running: multi-region ledger"),
});
