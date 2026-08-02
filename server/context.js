import { EventEmitter } from "node:events";

// Event bus for pushing announcements / job updates to connected browsers (SSE)
// and to anything else that wants to observe the shared context.
export const bus = new EventEmitter();

// The shared context both models can see.
export const shared = {
  history: [], // fast-model conversation (Anthropic MessageParam[], includes tool turns)
  // The assistant message is only pushed to history once its stream finishes, so
  // an interruption that arrives mid-stream has nothing to rewrite yet. These
  // let the truncation wait for the right message instead of clobbering the
  // previous reply.
  turnInFlight: false,
  pendingTruncation: null,
  insights: [], // { id, source: 'announcement'|'deep-reasoning', text, ts }
  jobs: new Map(), // jobId -> { id, question, status: 'running'|'done'|'refused'|'error', result }
};

let insightCounter = 0;
let jobCounter = 0;

// fields: { source: 'announcement'|'conclusion', jobId, text, question? }
export function addInsight(fields) {
  const insight = { id: ++insightCounter, ts: Date.now(), ...fields };
  shared.insights.push(insight);
  return insight;
}

export function createJob(question) {
  const job = { id: `job-${++jobCounter}`, question, status: "running", result: null };
  shared.jobs.set(job.id, job);
  return job;
}

// Rendered into both models' prompts so they work from the same picture.
// Questions live only in the job list and findings only in the findings list,
// so neither model sees the same text twice.
export function contextBlock() {
  if (shared.jobs.size === 0 && shared.insights.length === 0) {
    return "(empty — no reasoning tasks yet)";
  }

  const out = [];
  if (shared.jobs.size > 0) {
    out.push("Reasoning tasks:");
    for (const job of shared.jobs.values()) {
      out.push(`  ${job.id} [${job.status}] ${job.question}`);
    }
  }
  if (shared.insights.length > 0) {
    out.push("Findings:");
    // Conclusions can run to many thousands of characters. This block is
    // re-sent on every turn of both models, so cap each finding here; the UI
    // and /api/state still serve the full text from shared.insights.
    for (const ins of shared.insights.slice(-MAX_FINDINGS)) {
      out.push(`  ${ins.jobId} [${ins.source}] ${clip(ins.text, MAX_FINDING_CHARS)}`);
    }
  }
  return out.join("\n");
}

const MAX_FINDINGS = 10;
const MAX_FINDING_CHARS = 4000;

function clip(text, limit) {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}\n  […truncated for context; ${text.length - limit} more characters available in the UI]`;
}
