import Anthropic from "@anthropic-ai/sdk";
import { shared, bus, addInsight, createJob, contextBlock } from "./context.js";

const FAST_MODEL = process.env.FAST_MODEL || "claude-sonnet-5";
// Fable 5 is the intended deep-reasoning model, but it requires 30-day data
// retention on the org — SLOW_MODEL lets you fall back to Opus 5 without it.
const SLOW_MODEL = process.env.SLOW_MODEL || "claude-fable-5";

export const MODELS = { fast: FAST_MODEL, slow: SLOW_MODEL };

// The user talked over the assistant, so most of the last reply was never heard.
// Rewrite it to what actually reached them — otherwise the model carries on
// believing it said things the user has no knowledge of, and refers back to them.
export function truncateLastReply(spoken) {
  for (let i = shared.history.length - 1; i >= 0; i--) {
    const msg = shared.history[i];
    if (msg.role !== "assistant") continue;

    const said = spoken.trim();
    const note = said
      ? `${said} —[interrupted by the user here; they did not hear the rest]`
      : "[interrupted by the user before this could be spoken]";

    if (typeof msg.content === "string") {
      msg.content = note;
    } else if (Array.isArray(msg.content)) {
      let replaced = false;
      // Keep tool_use blocks intact; only the spoken text is rewritten.
      msg.content = msg.content.filter((block) => {
        if (block.type !== "text") return true;
        if (replaced) return false; // collapse multiple text blocks into one
        block.text = note;
        replaced = true;
        return true;
      });
      if (!replaced) msg.content.push({ type: "text", text: note });
    }
    return { truncated: true, text: note };
  }
  return { truncated: false };
}

let _client = null;
function client() {
  if (!_client) _client = new Anthropic(); // reads ANTHROPIC_API_KEY / auth profile
  return _client;
}

// ---------------------------------------------------------------------------
// Fast model (voice-facing): Sonnet 5, streaming, low effort for latency.
// Has one tool: hand a hard question to the slow model in the background.
// ---------------------------------------------------------------------------

const FAST_TOOLS = [
  {
    name: "start_deep_reasoning",
    description:
      "Hand a hard question to the background deep-reasoning model (Claude Fable 5). " +
      "Call this when the user asks something that needs careful multi-step reasoning, " +
      "analysis, planning, or research-grade thinking — anything you can't answer well " +
      "instantly. It runs in the background; its interim announcements and final " +
      "conclusion are added to the shared context, which you see on every turn. " +
      "After calling it, tell the user you've started thinking on it and keep chatting.",
    input_schema: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description:
            "A self-contained statement of the problem for the reasoning model, " +
            "including any relevant details from the conversation.",
        },
      },
      required: ["question"],
    },
  },
];

function fastSystem() {
  return (
    `You are the fast, voice-facing half of a two-model assistant. The user talks to you ` +
    `by voice; your replies are read aloud, so keep them short, natural, and conversational — ` +
    `one to three sentences by default. No markdown, no bullet lists.\n\n` +
    `That default is a default, not a limit. When the user asks for something long — a ` +
    `paragraph, a story, an explanation, "keep talking" — give them exactly that, at the length ` +
    `they asked for. Brevity does not override an explicit request.\n\n` +
    `Just produce what is asked for. If the user says "read a paragraph", "say something", ` +
    `"give me an example" or similar without supplying source material, they want you to ` +
    `compose it — write the paragraph and say it. You are not reading from a document and there ` +
    `is nothing to load; asking them to paste or provide text is wrong. Only ask a clarifying ` +
    `question when the answer genuinely depends on something you cannot reasonably choose ` +
    `yourself; otherwise pick something sensible and go.\n\n` +
    `The user's words reach you through speech recognition, so expect small errors: dropped or ` +
    `swapped words, odd phrasing, missing punctuation. Interpret them charitably and act on the ` +
    `obvious intent instead of objecting to the literal wording. "Sending generic text" almost ` +
    `certainly means "any generic text is fine".\n\n` +
    `ROOM VISION: A camera watches the room. The server already drops voice when nobody is ` +
    `looking — you never re-check looking yourself, and you never stay silent *because of* ` +
    `camera state. Messages may include [Room] / [Speaker] / [Said] metadata; the real user ` +
    `utterance is the [Said] line (or the whole message if there is no prefix). When you do ` +
    `speak, you may briefly name who spoke if useful. Typed messages have no vision prefix.\n\n` +
    `You work alongside a slow deep-reasoning model. For hard questions, use the ` +
    `start_deep_reasoning tool and keep the conversation going while it works. Don't attempt ` +
    `deep multi-step reasoning yourself. When the shared context below contains announcements ` +
    `or completed reasoning relevant to what the user asked, relay the substance conversationally.\n\n` +
    `You always think before answering. Your thinking is private — it is shown to the ` +
    `user on screen but never spoken aloud, so put reasoning there, not in your reply.\n\n` +
    `You do not have to produce spoken text. Thinking without a spoken reply is a normal, ` +
    `correct outcome. Stay silent (think, then end the turn with no assistant text at all) when:\n` +
    `- the utterance is a genuine fragment — trails off, ends on a conjunction/preposition, or ` +
    `  breaks mid-word (speech recognition often cuts people off);\n` +
    `- it is only a mic check / filler with no ask ("test test", "um", "hello?" with nothing after);\n` +
    `- there is nothing useful to say and speaking would just fill silence.\n` +
    `Do not narrate the silence ("I'll wait", "go ahead", "staying quiet") — say nothing.\n` +
    `This does NOT apply to short complete instructions ("tell me a joke", "stop", "read a ` +
    `paragraph", "do it") — those you carry out. Never treat terseness as incompleteness, and ` +
    `never use silence as a way to ask what they meant.\n\n` +
    `<shared_context>\n${contextBlock()}\n</shared_context>`
  );
}

// Runs one user turn through the fast model, streaming deltas via `send`.
// send(obj) writes one event to the HTTP response stream.
export async function runFastChat(userText, send) {
  shared.history.push({ role: "user", content: userText });

  shared.turnInFlight = true;
  try {
  while (true) {
    const stream = client().messages.stream({
      model: FAST_MODEL,
      // Thinking shares this budget with the reply, so leave headroom.
      max_tokens: 4096,
      // Reasoning is private: it reaches the UI but never text-to-speech, which
      // is what lets the model deliberate without narrating out loud.
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "low" }, // latency-sensitive voice path
      system: fastSystem(),
      tools: FAST_TOOLS,
      messages: shared.history,
    });

    stream.on("thinking", (delta) => send({ type: "thought", text: delta }));
    stream.on("text", (delta) => send({ type: "delta", text: delta }));
    const msg = await stream.finalMessage();
    shared.history.push({ role: "assistant", content: msg.content });

    if (msg.stop_reason !== "tool_use") break;

    const results = [];
    for (const block of msg.content) {
      if (block.type !== "tool_use") continue;
      if (block.name === "start_deep_reasoning") {
        const job = createJob(block.input.question);
        send({ type: "job_started", job: { id: job.id, question: job.question } });
        bus.emit("event", { type: "job_started", job: { id: job.id, question: job.question, status: job.status } });
        runReasoningJob(job).catch((err) => {
          job.status = "error";
          job.result = String(err);
          bus.emit("event", { type: "job_update", job });
        });
        results.push({
          type: "tool_result",
          tool_use_id: block.id,
          content:
            `Deep reasoning started (${job.id}). Announcements and the final conclusion ` +
            `will appear in shared context. Let the user know and continue the conversation.`,
        });
      } else {
        results.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: `Unknown tool: ${block.name}`,
          is_error: true,
        });
      }
    }
    shared.history.push({ role: "user", content: results });

    // The model often speaks before and after a tool call; keep those segments
    // from running together in the transcript.
    if (msg.content.some((b) => b.type === "text" && b.text.trim())) {
      send({ type: "delta", text: "\n\n" });
    }
  }
  } finally {
    shared.turnInFlight = false;
    // An interruption that arrived mid-stream deferred its truncation until the
    // assistant message actually existed. Apply it now, to the right message.
    if (shared.pendingTruncation !== null) {
      truncateLastReply(shared.pendingTruncation);
      shared.pendingTruncation = null;
    }
  }

  send({ type: "done" });
}

// ---------------------------------------------------------------------------
// Slow model (background): Fable 5, thinking always on, high effort.
// Has one tool: announce an interim finding through the fast model / UI.
// ---------------------------------------------------------------------------

const SLOW_TOOLS = [
  {
    name: "announce",
    description:
      "Push an interim finding, milestone, or important intermediate conclusion to the user " +
      "right now, via the voice assistant. Use it sparingly — only for genuinely useful " +
      "progress updates, not routine narration. One or two sentences, spoken-style.",
    input_schema: {
      type: "object",
      properties: {
        message: { type: "string", description: "The update to deliver to the user." },
      },
      required: ["message"],
    },
  },
];

function slowSystem() {
  return (
    `You are the deep-reasoning half of a two-model assistant. A fast voice model handles the ` +
    `live conversation; you work on hard questions in the background. Reason as carefully as ` +
    `the problem deserves. Use the announce tool for significant interim findings the user ` +
    `would want to hear before you finish. Your final message is stored in shared context and ` +
    `relayed by the voice model, so end with a clear, self-contained conclusion — lead with ` +
    `the answer, then the key supporting reasoning, in plain prose.\n\n` +
    `Always produce a conclusion — you are not part of the live conversation and must never end ` +
    `a turn empty. The voice model is the only thing the user hears; if you say nothing, your ` +
    `work is lost.\n\n` +
    `<shared_context>\n${contextBlock()}\n</shared_context>`
  );
}

async function runReasoningJob(job) {
  const messages = [{ role: "user", content: job.question }];

  while (true) {
    const stream = client().beta.messages.stream({
      model: SLOW_MODEL,
      max_tokens: 64000,
      output_config: { effort: "high" },
      // Fable 5's safety classifiers can decline a request (stop_reason: "refusal").
      // Server-side fallback re-runs declined requests on Anthropic's recommended
      // substitute model automatically.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: slowSystem(),
      tools: SLOW_TOOLS,
      messages,
    });
    const msg = await stream.finalMessage();

    if (msg.stop_reason === "refusal") {
      job.status = "refused";
      job.result = "The reasoning model declined this request.";
      bus.emit("event", { type: "job_update", job });
      return;
    }

    // Pass content (incl. thinking blocks) back unchanged when continuing the turn.
    messages.push({ role: "assistant", content: msg.content });

    if (msg.stop_reason !== "tool_use") {
      const text = msg.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n");
      job.status = "done";
      job.result = text;
      const insight = addInsight({
        source: "conclusion",
        jobId: job.id,
        question: job.question,
        text,
      });
      bus.emit("event", { type: "job_update", job });
      bus.emit("event", { type: "insight", insight });
      return;
    }

    const results = [];
    for (const block of msg.content) {
      if (block.type !== "tool_use") continue;
      if (block.name === "announce") {
        const insight = addInsight({
          source: "announcement",
          jobId: job.id,
          text: block.input.message,
        });
        bus.emit("event", { type: "insight", insight, speak: true });
        results.push({ type: "tool_result", tool_use_id: block.id, content: "Announced to the user." });
      } else {
        results.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: `Unknown tool: ${block.name}`,
          is_error: true,
        });
      }
    }
    messages.push({ role: "user", content: results });
  }
}
