// Gemini provider for the conversational model.
//
// shared.history stays in Anthropic message shape — it is the app's native format
// and truncateLastReply, the context block and the transcript all assume it — so
// this module translates on the way out and translates back on the way in.
//
// Shapes confirmed against gemini-3.6-flash rather than assumed:
//   - streamed parts carry `text`, `functionCall`, and `thoughtSignature`
//   - `thinkingConfig.includeThoughts` yields a *signature*, not readable thought
//     text, so the Thinking panel stays empty on Gemini (Claude does return
//     summaries). Not a bug, a provider difference.
//   - thoughtSignature must be echoed back alongside the functionCall it came
//     with, or multi-turn function calling on a thinking model is rejected.

const BASE = "https://generativelanguage.googleapis.com/v1beta";

// Gemini's thought signatures must be echoed back with the functionCall they came
// with, but they cannot be stored *on* the history block: history is serialised
// verbatim to Anthropic when the model is switched back, and Anthropic rejects
// unknown fields ("tool_use.thoughtSignature: Extra inputs are not permitted").
// So they live beside the history, keyed by tool_use id.
const signatures = new Map();

export function rememberSignature(toolUseId, signature) {
  if (!toolUseId || !signature) return;
  signatures.set(toolUseId, signature);
  // Bounded: only the current conversation's calls can still be referenced.
  if (signatures.size > 200) signatures.delete(signatures.keys().next().value);
}

export function isGeminiModel(model) {
  return typeof model === "string" && model.startsWith("gemini");
}

export function hasGeminiKey() {
  return Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY);
}

function apiKey() {
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) throw new Error("GEMINI_API_KEY is not set");
  return key;
}

/** Anthropic-shaped history -> Gemini `contents`. */
export function toGeminiContents(history) {
  const out = [];
  // functionResponse needs the name of the function it answers, which the
  // Anthropic tool_result block doesn't carry — so remember it by id.
  const nameById = new Map();

  for (const msg of history) {
    const role = msg.role === "assistant" ? "model" : "user";
    const parts = [];

    if (typeof msg.content === "string") {
      if (msg.content.trim()) parts.push({ text: msg.content });
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "text") {
          if (block.text?.trim()) parts.push({ text: block.text });
        } else if (block.type === "tool_use") {
          nameById.set(block.id, block.name);
          const part = { functionCall: { name: block.name, args: block.input || {} } };
          const sig = signatures.get(block.id);
          if (sig) part.thoughtSignature = sig;
          parts.push(part);
        } else if (block.type === "tool_result") {
          parts.push({
            functionResponse: {
              name: nameById.get(block.tool_use_id) || "tool",
              response: { output: String(block.content ?? "") },
            },
          });
        }
        // thinking / redacted_thinking are Claude-only and are dropped.
      }
    }

    if (parts.length) out.push({ role, parts });
  }
  return out;
}

/**
 * One streaming request. Emits deltas through `send` and returns
 * { text, toolCalls } so the caller can drive its own tool loop.
 */
export async function streamGeminiTurn({ model, system, tools, history, send, maxTokens = 4096 }) {
  const body = {
    contents: toGeminiContents(history),
    generationConfig: {
      maxOutputTokens: maxTokens,
      thinkingConfig: { includeThoughts: true },
    },
  };
  if (system) body.systemInstruction = { parts: [{ text: system }] };
  if (tools?.length) {
    body.tools = [
      {
        functionDeclarations: tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.input_schema,
        })),
      },
    ];
  }

  const res = await fetch(`${BASE}/models/${model}:streamGenerateContent?alt=sse`, {
    method: "POST",
    headers: { "x-goog-api-key": apiKey(), "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    // Fail loudly with the provider's own message rather than a generic error.
    throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 400)}`);
  }

  let text = "";
  const toolCalls = [];
  let buffer = "";

  for await (const chunk of res.body) {
    buffer += Buffer.from(chunk).toString("utf8");
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;

      let event;
      try {
        event = JSON.parse(payload);
      } catch {
        continue; // a split JSON object; the remainder stays in `buffer`
      }

      for (const cand of event.candidates || []) {
        for (const part of cand.content?.parts || []) {
          if (part.functionCall) {
            toolCalls.push({
              name: part.functionCall.name,
              input: part.functionCall.args || {},
              thoughtSignature: part.thoughtSignature,
            });
          } else if (part.text) {
            // `thought: true` marks reasoning rather than speech. Gemini has not
            // been observed returning it, but honour it if it appears — it must
            // never reach text-to-speech.
            if (part.thought) send({ type: "thought", text: part.text });
            else {
              text += part.text;
              send({ type: "delta", text: part.text });
            }
          }
        }
      }
    }
  }

  return { text, toolCalls };
}
