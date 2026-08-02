// Splits streaming model text into speakable chunks.
//
// Time-to-first-audio dominates how responsive the assistant feels, so the
// first chunk is cut early — at a clause boundary — rather than waiting for a
// full sentence. Later chunks prefer sentence boundaries, which sound better.

const FIRST_CHUNK_MIN = 32; // start talking once we have at least this much
const CLAUSE_MIN = 24; // don't cut a clause shorter than this
const MAX_CHUNK = 180; // never buffer more than this before speaking

// Returns [chunk, rest] or null if nothing is ready to speak yet.
export function nextChunk(buffer, spokenAnything) {
  // A complete sentence, confirmed by trailing whitespace. Requiring the
  // whitespace is what keeps "3.14" or "Dr. Chen" from being cut mid-number.
  const sentence = buffer.match(/^([\s\S]*?[.!?…]+["'”’)\]]?)(\s+)([\s\S]*)$/);
  if (sentence) return [sentence[1].trim(), sentence[3]];

  // Nothing spoken yet: cut at the first clause boundary so audio starts
  // without waiting for the sentence to finish.
  if (!spokenAnything && buffer.length >= FIRST_CHUNK_MIN) {
    const clause = buffer.match(
      new RegExp(`^([\\s\\S]{${CLAUSE_MIN},}?[,;:—–])(\\s+)([\\s\\S]*)$`)
    );
    if (clause) return [clause[1].trim(), clause[3]];
  }

  // Hard cap for text that never punctuates.
  if (buffer.length > MAX_CHUNK) {
    const cut = buffer.lastIndexOf(" ", MAX_CHUNK);
    if (cut > CLAUSE_MIN) return [buffer.slice(0, cut).trim(), buffer.slice(cut + 1)];
  }

  return null;
}
