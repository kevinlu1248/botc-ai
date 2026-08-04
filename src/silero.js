// Silero VAD v5 — a trained speech/no-speech model, replacing the hand-tuned
// spectral heuristics in vad.js for barge-in detection.
//
// Why a model: every constant in vad.js (flatness ceiling, band ratio, RMS floor)
// is a number tuned against one room, and they have already been loosened once for
// speed and re-tightened afterwards. Breath, hair rustle and keyboard noise are
// exactly what a VAD model is trained to reject. Benchmarked on the same cases the
// heuristics were: 9/9, speech scoring 0.999–1.000 against 0.021–0.137 for every
// noise — a margin wide enough that the threshold needs no per-room tuning. It
// also fires on speech far quieter than the old RMS floor allowed.
//
// Call contract (easy to get wrong): v5 wants **576** samples per inference —
// 64 samples of context carried from the previous frame followed by 512 new ones.
// Feeding a bare 512 returns ~0.003 for clear speech, which looks exactly like a
// working model that simply hears nothing.

const SAMPLE_RATE = 16000;
const FRAME = 512; // new samples per inference
const CONTEXT = 64; // carried from the previous frame
const STATE_SHAPE = [2, 1, 128];

let ortPromise = null;

async function loadOrt() {
  if (!ortPromise) {
    // The bare "onnxruntime-web" entry drags in the WebGPU/jsep build — a 26MB
    // wasm asset for a 2MB model we run on CPU. The /wasm subpath is the plain
    // CPU backend only.
    ortPromise = import("onnxruntime-web/wasm").then((ort) => {
      // Point ONLY the binary at our own copy, using the object form.
      // Two failure modes this threads between, both hit already:
      //   wasmPaths = "/models/"  -> ORT also imports its .mjs glue from /public,
      //     which Vite refuses to serve ("should not be imported from source code")
      //   no override at all      -> in dev the glue looks for the .wasm beside the
      //     pre-bundled dep in /node_modules/.vite/deps/, which does not exist, so
      //     the fetch 404s to index.html and WebAssembly reports "found 3c 21 64 6f"
      //     (that is the ASCII for "<!do")
      ort.env.wasm.wasmPaths = { wasm: "/models/ort-wasm-simd-threaded.wasm" };
      ort.env.wasm.numThreads = 1; // no COOP/COEP headers in dev, so no threads
      ort.env.logLevel = "error";
      return ort;
    });
  }
  return ortPromise;
}

export async function createSileroVad({ modelUrl = "/models/silero_vad.onnx" } = {}) {
  const ort = await loadOrt();
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  let state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), STATE_SHAPE);
  let context = new Float32Array(CONTEXT);
  const input = new Float32Array(CONTEXT + FRAME);
  const sr = new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), []);

  return {
    frameSize: FRAME,

    /** frame: Float32Array of exactly FRAME samples. Returns speech probability. */
    async process(frame) {
      if (frame.length !== FRAME) return 0;
      input.set(context, 0);
      input.set(frame, CONTEXT);

      const out = await session.run({
        input: new ort.Tensor("float32", input, [1, input.length]),
        state,
        sr,
      });
      state = out.stateN;
      context = frame.slice(-CONTEXT);
      return out.output.data[0];
    },

    /** Forget conversational state — call when the mic restarts. */
    reset() {
      state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), STATE_SHAPE);
      context = new Float32Array(CONTEXT);
    },

    close() {
      session.release?.();
    },
  };
}

// Hysteresis around the probability. Entering high and leaving lower stops a
// wavering score from flapping mid-syllable.
// Per-frame hysteresis only. How many such frames it takes to interrupt the
// assistant is policy, not model plumbing, and lives in src/bargein.js.
export const SILERO = {
  ENTER: 0.5,
  EXIT: 0.35,
};
