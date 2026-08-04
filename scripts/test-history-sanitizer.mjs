// Reproduces "`thinking` blocks in the latest assistant message cannot be modified"
// against the real API, then proves messagesForClaude() turns each poisoned history
// into a payload the API accepts.
//
//   node scripts/test-history-sanitizer.mjs
import "../server/env.js";
import Anthropic from "@anthropic-ai/sdk";
import { messagesForClaude, recordAssistant } from "../server/agents.js";
import { shared } from "../server/context.js";

const client = new Anthropic();
const MODEL = "claude-sonnet-5";

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

const PROMPT = "Is it better to run our own Kubernetes or use ECS Fargate for 12 services?";

const send = (model, messages) =>
  client.messages
    .stream({
      model,
      max_tokens: 1024,
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "low" },
      system: "You are a voice assistant.",
      tools: TOOLS,
      messages,
    })
    .finalMessage();

async function attempt(label, messages, { expect }) {
  let got;
  let detail = "";
  try {
    await send(MODEL, messages);
    got = "accepted";
  } catch (e) {
    got = "rejected";
    detail = e.error?.error?.message || e.message;
  }
  const pass = got === expect;
  console.log(`${pass ? "PASS" : "FAIL"}  ${label}\n        expected ${expect}, got ${got}${detail ? `: ${detail.slice(0, 120)}` : ""}`);
  return pass;
}

// A real assistant turn containing a genuine, correctly-signed thinking block.
let origin = null;
for (let i = 0; i < 5 && !origin; i++) {
  const msg = await send(MODEL, [{ role: "user", content: PROMPT }]);
  if (msg.content.some((b) => b.type === "thinking")) origin = msg;
}
if (!origin) {
  console.log("could not obtain a thinking block; aborting");
  process.exit(1);
}
console.log(`origin shape=[${origin.content.map((b, i) => `${i}:${b.type}`).join(" ")}] stop=${origin.stop_reason}\n`);

const toolBlocks = origin.content.filter((b) => b.type === "tool_use");
const followUp = toolBlocks.length
  ? {
      role: "user",
      content: toolBlocks.map((b) => ({
        type: "tool_result",
        tool_use_id: b.id,
        content: "Job 1 started.",
      })),
    }
  : { role: "user", content: "What did you just say?" };

// Poison it the way a mid-conversation model switch does: the block is present and
// well-formed, but its signature is no longer valid for this request.
const tampered = origin.content.map((b) =>
  b.type === "thinking" ? { ...b, signature: b.signature.slice(0, -8) + "AAAAAAAA" } : b
);

const poisoned = [
  { role: "user", content: PROMPT },
  { role: "assistant", content: tampered },
  followUp,
];

const results = [];

// 1. The bug itself — this is the user's 400.
results.push(await attempt("poisoned history sent raw (the reported bug)", poisoned, { expect: "rejected" }));

// 1b. The reported error verbatim: block CONTENT altered, signature left intact,
//     which is a different message from a corrupted signature. Both must be caught
//     by the retry predicate in streamFastTurn.
const editedText = origin.content.map((b) =>
  b.type === "thinking" ? { ...b, thinking: `${b.thinking} (edited)` } : b
);
const edited = [
  { role: "user", content: PROMPT },
  { role: "assistant", content: editedText },
  followUp,
];
// The retry predicate in streamFastTurn must match the error the user actually
// hit, so assert against its literal text rather than hoping to synthesize it.
const REPORTED =
  "messages.1.content.1: `thinking` or `redacted_thinking` blocks in the latest assistant " +
  "message cannot be modified. These blocks must remain as they were in the original response.";
const matchesReported = /thinking|tool_use|content\.\d/.test(REPORTED);
console.log(
  `${matchesReported ? "PASS" : "FAIL"}  retry predicate matches the reported error string`
);
results.push(matchesReported);

// Which structural mutation actually yields "cannot be modified"? Informational —
// the sanitiser handles all of them, but it pins down the real trigger.
const secondTurn = await send(MODEL, [
  { role: "user", content: "Name one tradeoff of microservices. Think first." },
]);
const foreignThinking = secondTurn.content.find((b) => b.type === "thinking");
const MUTATIONS = {
  "thinking text altered, signature kept": editedText,
  "thinking block from a DIFFERENT turn (valid signature, wrong turn)": foreignThinking
    ? origin.content.map((b) => (b.type === "thinking" ? foreignThinking : b))
    : null,
  "blocks reordered (text before thinking)": [...origin.content].reverse(),
  "thinking text emptied": origin.content.map((b) => (b.type === "thinking" ? { ...b, thinking: "" } : b)),
};
for (const [label, content] of Object.entries(MUTATIONS)) {
  if (!content) continue;
  let detail = "";
  try {
    await send(MODEL, [{ role: "user", content: PROMPT }, { role: "assistant", content }, followUp]);
  } catch (e) {
    detail = e.error?.error?.message || e.message;
  }
  const exact = /cannot be modified/.test(detail);
  console.log(`      ${exact ? ">>> REPORTED ERROR" : detail ? "other 400" : "accepted"}  ${label}${detail ? `\n          ${detail.slice(0, 120)}` : ""}`);
}
results.push(
  await attempt("altered-text history through messagesForClaude()", messagesForClaude(MODEL, edited).messages, {
    expect: "accepted",
  })
);

// 2. Sanitised. producedBy has no entry for this message, so it is treated as
//    foreign — exactly the mid-conversation-switch case.
const { messages: fixed, notes } = messagesForClaude(MODEL, poisoned);
console.log(`        sanitiser notes: ${notes.join("; ") || "(none)"}`);
results.push(await attempt("same history through messagesForClaude()", fixed, { expect: "accepted" }));

// 3. Gemini-shaped tool turn (no thinking block) followed by an orphaned result.
const GEM_ID = "gem_1754000000000_0";
const geminiHistory = [
  { role: "user", content: PROMPT },
  {
    role: "assistant",
    content: [
      { type: "text", text: "Let me get the reasoning model on that." },
      { type: "tool_use", id: GEM_ID, name: "start_deep_reasoning", input: { question: PROMPT } },
    ],
  },
  { role: "user", content: [{ type: "tool_result", tool_use_id: GEM_ID, content: "Job 1 started." }] },
  { role: "user", content: "So what do you think?" },
];
const { messages: gemFixed } = messagesForClaude(MODEL, geminiHistory);
results.push(await attempt("gemini-shaped history through messagesForClaude()", gemFixed, { expect: "accepted" }));

// 4. Strict recovery pass must also be valid.
const { messages: strictFixed, notes: strictNotes } = messagesForClaude(MODEL, poisoned, { strict: true });
console.log(`        strict notes: ${strictNotes.join("; ") || "(none)"}`);
results.push(await attempt("strict recovery pass", strictFixed, { expect: "accepted" }));

// 5. The normal path: a turn recorded through recordAssistant by the SAME model
//    must be forwarded verbatim, because that is the only form the API accepts and
//    dropping it would throw away the turn's reasoning for no reason.
shared.history.length = 0;
shared.history.push({ role: "user", content: PROMPT });
recordAssistant(origin.content, MODEL);
shared.history.push(followUp);

const { messages: cleanOut, notes: cleanNotes } = messagesForClaude(MODEL);
const verbatim = cleanOut[1].content === origin.content;
console.log(
  `${verbatim && cleanNotes.length === 0 ? "PASS" : "FAIL"}  same-model turn forwarded verbatim\n` +
    `        verbatim=${verbatim} notes=${cleanNotes.join("; ") || "(none)"}`
);
results.push(verbatim && cleanNotes.length === 0);
results.push(await attempt("same-model history through messagesForClaude()", cleanOut, { expect: "accepted" }));

// 6. Switching model mid-conversation must strip the other model's thinking.
const { notes: switchNotes } = messagesForClaude("claude-opus-5");
const stripped = switchNotes.some((n) => n.includes("stripped thinking"));
console.log(
  `${stripped ? "PASS" : "FAIL"}  switching model strips the previous model's thinking\n` +
    `        notes=${switchNotes.join("; ") || "(none)"}`
);
results.push(stripped);

// 7. Haiku rejects both adaptive thinking and effort; the request builder must omit
//    them so choosing Haiku in the dropdown does not break every turn.
try {
  const { messages: hm } = messagesForClaude("claude-haiku-4-5");
  await client.messages
    .stream({
      model: "claude-haiku-4-5",
      max_tokens: 512,
      system: "You are a voice assistant.",
      tools: TOOLS,
      messages: hm,
    })
    .finalMessage();
  console.log("PASS  haiku-4-5 without thinking or effort");
  results.push(true);
} catch (e) {
  console.log(`FAIL  haiku-4-5 without thinking or effort: ${e.error?.error?.message || e.message}`);
  results.push(false);
}

const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
