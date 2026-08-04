import Anthropic from "@anthropic-ai/sdk";
import { shared, bus, addInsight, createJob, contextBlock } from "./context.js";
import { getSettings } from "./settings.js";
import { isGeminiModel, streamGeminiTurn, rememberSignature } from "./gemini.js";


// Read per request, so the settings modal takes effect on the next turn without
// a restart. Fable 5 needs 30-day org data retention; see settings.js defaults.
export const MODELS = {
  get fast() {
    return getSettings().fast;
  },
  get slow() {
    return getSettings().slow;
  },
};

// The user talked over the assistant, so most of the last reply was never heard.
// Rewrite it to what actually reached them — otherwise the model carries on
// believing it said things the user has no knowledge of, and refers back to them.
//
// CRITICAL with extended thinking: Anthropic rejects any partial edit to an
// assistant message that still contains `thinking` / `redacted_thinking` blocks
// ("cannot be modified"). On interrupt we replace the *entire* content with
// plain text (dropping thinking). That turn's private reasoning is abandoned;
// only what was spoken stays in the transcript.
export function truncateLastReply(spoken) {
  for (let i = shared.history.length - 1; i >= 0; i--) {
    const msg = shared.history[i];
    if (msg.role !== "assistant") continue;

    const said = spoken.trim();
    const note = said
      ? `${said} —[interrupted by the user here; they did not hear the rest]`
      : "[interrupted by the user before this could be spoken]";

    // Extended thinking makes this stricter than it looks. Anthropic rejects any
    // edit to an assistant message that carries `thinking`/`redacted_thinking`
    // ("cannot be modified"), and it also rejects keeping `tool_use` while the
    // accompanying thinking is gone — that combination produced
    // "messages.N.content.1: thinking ... cannot be modified" in practice.
    //
    // So: collapse the whole turn to a plain string, and drop everything after
    // it. Anything later belongs to the tool round-trip of the turn that was just
    // interrupted, and leaving a `tool_result` whose `tool_use` no longer exists
    // is its own 400.
    msg.content = note;
    shared.history.length = i + 1;
    return { truncated: true, text: note };
  }
  return { truncated: false };
}

// The model's explicit "I chose not to answer" token.
//
// An empty turn was the previous convention, and it is ambiguous: indistinguishable
// from a dropped stream, a refusal, or a bug. A sentinel makes the decision visible
// in the transcript and in history, and the user always gets heard either way —
// nothing is filtered out before the model sees it.
export const NO_RESPONSE = "[NO RESPONSE]";

/**
 * Wraps `send` so the sentinel is never spoken.
 *
 * Deltas stream token by token and text-to-speech starts mid-stream, so by the time
 * the full token is visible it would already be on its way to the speakers. This
 * holds output only while what has arrived so far could still *become* the sentinel
 * — a normal reply diverges on its first character and pays nothing.
 */
function noResponseFilter(send) {
  let held = "";
  let passthrough = false; // this turn is speech; stop inspecting
  let declined = false;

  return {
    send(evt) {
      if (passthrough || evt.type !== "delta") return send(evt);
      if (declined) return; // trailing junk after the token

      held += evt.text;
      const seen = held.trim().toUpperCase();

      // Lenient on trailing punctuation/newlines the model may append.
      if (seen.startsWith(NO_RESPONSE)) {
        declined = true;
        return send({ type: "no_response" });
      }
      if (NO_RESPONSE.startsWith(seen)) return; // still a possible prefix — hold
      passthrough = true;
      return send({ type: "delta", text: held });
    },

    /** Releases a partial hold, so a malformed near-sentinel is never swallowed. */
    finish() {
      if (!declined && !passthrough && held) {
        passthrough = true;
        send({ type: "delta", text: held });
      }
    },

    declined: () => declined,
  };
}

let _client = null;
function client() {
  if (!_client) _client = new Anthropic(); // reads ANTHROPIC_API_KEY / auth profile
  return _client;
}

// ---------------------------------------------------------------------------
// History hygiene.
//
// Anthropic validates thinking-block integrity on the LAST assistant message and
// rejects any block it did not itself produce in that exact form:
//   "`thinking` or `redacted_thinking` blocks in the latest assistant message
//    cannot be modified."
// A thinking signature is bound to the model that produced it, and the settings
// modal makes switching the conversation model mid-conversation a normal thing to
// do — Sonnet -> Gemini -> Sonnet. Gemini turns carry no thinking blocks at all.
// So shared.history is NOT guaranteed to be a valid payload for whichever model is
// selected now; it has to be sanitised per request.
//
// Which model produced a message is request metadata, not content: Anthropic
// rejects unknown fields on blocks ("tool_use.thoughtSignature: Extra inputs are
// not permitted"), so it is kept beside the history in a WeakMap.
const producedBy = new WeakMap();

/** Pushes an assistant message, remembering which model produced it. */
export function recordAssistant(content, model) {
  const msg = { role: "assistant", content };
  producedBy.set(msg, model);
  shared.history.push(msg);
  return msg;
}

const isThinking = (b) => b.type === "thinking" || b.type === "redacted_thinking";

/**
 * Builds the messages payload for `model` from history.
 *
 * Verified against the API rather than assumed (scripts/thinking-*.mjs):
 *  - thinking blocks on NON-final assistant messages are ignored, so dropping
 *    them is safe and keeps the payload provider-agnostic;
 *  - an assistant turn with tool_use and no thinking block is accepted;
 *  - only the final assistant message's thinking blocks are integrity-checked.
 *
 * `strict` is the recovery pass: strip every thinking block, including the final
 * message's, and collapse a tool round we can no longer prove is replayable.
 */
export function messagesForClaude(model, history = shared.history, { strict = false } = {}) {
  let lastAssistant = -1;
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === "assistant") {
      lastAssistant = i;
      break;
    }
  }

  const out = [];
  const notes = [];
  let orphanedResults = false;

  for (let i = 0; i < history.length; i++) {
    const msg = history[i];
    const blocks = Array.isArray(msg.content) ? msg.content : null;

    if (msg.role !== "assistant") {
      // A tool_result whose tool_use was just collapsed away is its own 400.
      if (orphanedResults && blocks?.some((b) => b.type === "tool_result")) {
        notes.push(`dropped orphaned tool_result at ${i}`);
        continue;
      }
      out.push(msg);
      continue;
    }

    // Already collapsed to plain text (an interrupted turn) — nothing to validate.
    if (!blocks) {
      out.push(msg);
      continue;
    }

    const hasThinking = blocks.some(isThinking);
    const isLast = i === lastAssistant;

    if (isLast && !strict) {
      // Safe verbatim when this same model produced it, or when there is no
      // thinking block whose signature could fail to validate.
      if (!hasThinking || producedBy.get(msg) === model) {
        out.push(msg);
        continue;
      }
    }

    const kept = blocks.filter((b) => !isThinking(b));

    if (isLast && kept.some((b) => b.type === "tool_use")) {
      // Stripping thinking while keeping tool_use in the final message is rejected
      // too, so collapse the turn to what it said and drop its tool round-trip.
      const text = kept
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join(" ")
        .trim();
      out.push({ role: "assistant", content: text || "[handed that to the reasoning model]" });
      orphanedResults = true;
      notes.push(`collapsed tool turn at ${i} from ${producedBy.get(msg) ?? "another provider"}`);
      continue;
    }

    if (hasThinking) notes.push(`stripped thinking at ${i}`);
    out.push({ role: "assistant", content: kept.length ? kept : "" });
  }

  return { messages: out, notes };
}

// Not every model the settings modal offers accepts the tuning the voice path
// wants. claude-haiku-4-5 rejects both, one at a time:
//   "adaptive thinking is not supported on this model"
//   "This model does not support the effort parameter."
// Sending either would fail every turn, so selecting Haiku has to drop both.
const NO_THINKING_OR_EFFORT = new Set(["claude-haiku-4-5"]);

function fastRequest(model, messages) {
  const req = {
    model,
    // Thinking shares this budget with the reply, so leave headroom.
    max_tokens: 4096,
    system: fastSystem(),
    tools: FAST_TOOLS,
    messages,
  };
  if (!NO_THINKING_OR_EFFORT.has(model)) {
    req.output_config = { effort: "low" }; // latency-sensitive voice path
    // Reasoning is private: it reaches the UI but never text-to-speech, which is
    // what lets the model deliberate without narrating out loud.
    req.thinking = { type: "adaptive", display: "summarized" };
  }
  return req;
}

/**
 * One streamed Claude turn. Retries once with a stricter payload if the API
 * rejects the message shape, so a single stale block cannot kill the whole
 * conversation — before this, every following turn failed too and the only way
 * out was restarting the server.
 */
async function streamFastTurn(model, send) {
  for (const strict of [false, true]) {
    const { messages, notes } = messagesForClaude(model, shared.history, { strict });
    if (notes.length) console.warn(`[history] ${notes.join("; ")}`);

    try {
      const stream = client().messages.stream(fastRequest(model, messages));
      stream.on("thinking", (delta) => send({ type: "thought", text: delta }));
      stream.on("text", (delta) => send({ type: "delta", text: delta }));
      return await stream.finalMessage();
    } catch (err) {
      const detail = err?.error?.error?.message || err?.message || "";
      const shapeRejected = err?.status === 400 && /thinking|tool_use|content\.\d/.test(detail);
      if (strict || !shapeRejected) throw err;
      console.error(`[history] payload rejected: ${detail}\n[history] retrying with thinking stripped`);
    }
  }
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
    `ROOM VISION: A camera watches the room. Messages may include [Room] / [Speaker] / [Said] ` +
    `metadata; the actual utterance is the [Said] line (or the whole message if there is no ` +
    `prefix). The camera tells you *who* spoke — you never decide whether to answer based on ` +
    `camera state. When you do speak, you may briefly name who spoke if useful. Typed messages ` +
    `have no vision prefix.\n\n` +
    `NOT EVERY UTTERANCE IS FOR YOU. When [Speaker] says the voice is NOT the person at the ` +
    `camera — a labelled "Speaker 2", background audio, a TV, a phone, someone else in the room ` +
    `— treat it as context you overheard, not as a request. Decline it. Keep it in mind, because ` +
    `the user may ask about it later ("what did that just say?"), and answer *then*. Only the ` +
    `person at the camera is talking to you.\n\n` +
    `You work alongside a slow deep-reasoning model. For hard questions, use the ` +
    `start_deep_reasoning tool and keep the conversation going while it works. Don't attempt ` +
    `deep multi-step reasoning yourself. When the shared context below contains announcements ` +
    `or completed reasoning relevant to what the user asked, relay the substance conversationally.\n\n` +
    `You always think before answering. Your thinking is private — it is shown on screen in a ` +
    `separate "Thinking" panel and is NEVER spoken aloud. Put all reasoning, mic-check ` +
    `judgements, and "should I answer?" decisions ONLY in thinking — never in the spoken reply.\n\n` +
    `You do not have to answer. Choosing not to respond is a normal, correct outcome — but it ` +
    `must be an explicit choice, not an empty turn. To decline, reply with exactly:\n` +
    `${NO_RESPONSE}\n` +
    `That token and nothing else — no punctuation, no explanation around it. It is never spoken ` +
    `aloud and never shown as speech; it tells the app you deliberately said nothing. Reason in ` +
    `thinking first, then emit it. Decline when:\n` +
    `- the utterance is a genuine fragment — trails off, ends on a conjunction/preposition, or ` +
    `  breaks mid-word (speech recognition often cuts people off);\n` +
    `- it is only a mic check / filler with no ask ("test test", "testing", "um", "hello?" alone);\n` +
    `- they told you to stop, wait, be quiet, or hold on. Your audio has ALREADY been cut ` +
    `off before you see this — the stop happened. "Okay, stopping." is a reply to an ` +
    `instruction whose entire point was for you to stop talking, so it is exactly wrong. ` +
    `Note in thinking that you were stopped, then decline;\n` +
    `- the utterance contains no request. Acknowledgements and reactions ("ok", "cool", ` +
    `"nice", "thanks", "got it", "yeah", "I see", "makes sense"), the user thinking out loud, ` +
    `or talk that is plainly not addressed to you all fall here. Answer what was asked; when ` +
    `nothing was asked, stay quiet rather than manufacturing a response to be polite;\n` +
    `- there is nothing useful to say and speaking would just fill silence.\n` +
    `CRITICAL: never narrate the silence as speech. Wrong: "I'll stay quiet" / "this is just a ` +
    `mic check" / "${NO_RESPONSE} — just a mic check". Right: thinking explains why, and the ` +
    `entire reply is the bare token.\n` +
    `This does NOT apply to short complete instructions ("tell me a joke", ` +
    `"read a paragraph", "do it", "say something") — those you carry out, and a question always ` +
    `gets an answer. Never treat terseness as incompleteness, and never use silence as a way to ` +
    `ask what they meant. The test is whether anything was actually asked of you, not how short ` +
    `the utterance was.\n\n` +
    `<shared_context>\n${contextBlock()}\n</shared_context>`
  );
}


// Executes one fast-model tool call. Shared by both providers so a tool never
// behaves differently depending on which model asked for it.
function runFastTool(name, input, send) {
  if (name === "start_deep_reasoning") {
    const job = createJob(input?.question || "");
    send({ type: "job_started", job: { id: job.id, question: job.question } });
    bus.emit("event", {
      type: "job_started",
      job: { id: job.id, question: job.question, status: job.status },
    });
    runReasoningJob(job).catch((err) => {
      job.status = "error";
      job.result = String(err);
      bus.emit("event", { type: "job_update", job });
    });
    return {
      content:
        `Deep reasoning started (${job.id}). Announcements and the final conclusion ` +
        `will appear in shared context. Let the user know and continue the conversation.`,
    };
  }
  return { content: `Unknown tool: ${name}`, is_error: true };
}

// One turn on Gemini. Mirrors the Claude loop: stream, run any tool calls, feed
// results back, repeat. History stays Anthropic-shaped (see gemini.js).
async function runGeminiChat(send) {
  const model = getSettings().fast;
  let guard = 0;

  while (guard++ < 6) {
    const { text, toolCalls } = await streamGeminiTurn({
      model,
      system: fastSystem(),
      tools: FAST_TOOLS,
      history: shared.history,
      send,
    });

    if (!toolCalls.length) {
      // A turn with no speech is legitimate — the prompt allows staying silent.
      recordAssistant(text || "", model);
      return;
    }

    const blocks = [];
    if (text.trim()) blocks.push({ type: "text", text });
    const results = [];
    toolCalls.forEach((call, i) => {
      const id = `gem_${Date.now()}_${i}`;
      blocks.push({ type: "tool_use", id, name: call.name, input: call.input });
      // Kept out of the block itself — see rememberSignature in gemini.js.
      rememberSignature(id, call.thoughtSignature);
      const out = runFastTool(call.name, call.input, send);
      results.push({ type: "tool_result", tool_use_id: id, ...out });
    });

    recordAssistant(blocks, model);
    shared.history.push({ role: "user", content: results });
    if (text.trim()) send({ type: "delta", text: "\n\n" });
  }
}

// Runs one user turn through the fast model, streaming deltas via `send`.
// send(obj) writes one event to the HTTP response stream.
export async function runFastChat(userText, rawSend) {
  shared.history.push({ role: "user", content: userText });

  // Everything the model emits goes through here, so a declined turn cannot reach
  // text-to-speech by any path — both providers included.
  const filter = noResponseFilter(rawSend);
  const send = (evt) => filter.send(evt);

  shared.turnInFlight = true;
  try {
  if (isGeminiModel(getSettings().fast)) {
    await runGeminiChat(send);
  } else
  while (true) {
    // Re-read per iteration: the model can change between tool round-trips.
    const model = getSettings().fast;
    const msg = await streamFastTurn(model, send);
    recordAssistant(msg.content, model);

    if (msg.stop_reason !== "tool_use") break;

    const results = [];
    for (const block of msg.content) {
      if (block.type !== "tool_use") continue;
      results.push({
        type: "tool_result",
        tool_use_id: block.id,
        ...runFastTool(block.name, block.input, send),
      });
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

  filter.finish();
  rawSend({ type: "done" });
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

/** Drops thinking blocks so a rejected payload can be retried. */
function withoutThinking(messages) {
  return messages.map((m) =>
    m.role === "assistant" && Array.isArray(m.content)
      ? { role: "assistant", content: m.content.filter((b) => !isThinking(b)) }
      : m
  );
}

async function runReasoningJob(job) {
  const messages = [{ role: "user", content: job.question }];

  while (true) {
    // display:"summarized" is deliberate, not cosmetic. Omitting it yields raw
    // thinking blocks, whose replay on the announce round-trip is validated
    // strictly — and `fallbacks` can answer on a substitute model, so the blocks
    // in `messages` may not even be from the model named here.
    const request = {
      model: getSettings().slow,
      max_tokens: 64000,
      output_config: { effort: "high" },
      thinking: { type: "adaptive", display: "summarized" },
      // Fable 5's safety classifiers can decline a request (stop_reason: "refusal").
      // Server-side fallback re-runs declined requests on Anthropic's recommended
      // substitute model automatically.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: slowSystem(),
      tools: SLOW_TOOLS,
    };

    let msg;
    try {
      msg = await client().beta.messages.stream({ ...request, messages }).finalMessage();
    } catch (err) {
      const detail = err?.error?.error?.message || err?.message || "";
      if (!(err?.status === 400 && /thinking|tool_use|content\.\d/.test(detail))) throw err;
      // Don't lose a long reasoning job to a stale block on the announce round-trip.
      console.error(`[job ${job.id}] payload rejected: ${detail}\n[job ${job.id}] retrying without thinking`);
      msg = await client()
        .beta.messages.stream({ ...request, messages: withoutThinking(messages) })
        .finalMessage();
    }

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
