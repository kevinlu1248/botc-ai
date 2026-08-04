// Taps the RAW microphone and emits fixed 512-sample Float32 frames, which is
// what Silero VAD consumes at 16 kHz. Separate from pcm-worklet.js on purpose:
// that one carries the *conditioned* signal to Deepgram, and the conditioning
// chain compresses dynamics — fine for a recogniser, wrong for voice detection.
class VadWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options?.processorOptions?.frameSize ?? 512;
    this.buf = new Float32Array(this.size);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this.buf[this.filled++] = channel[i];
      if (this.filled === this.size) {
        // Copy: the receiver holds it past this callback.
        this.port.postMessage(this.buf.slice(0));
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("vad-worklet", VadWorklet);
