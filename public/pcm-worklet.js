// Converts mic Float32 frames to 16-bit PCM and posts them to the main thread.
// The AudioContext is created at 16 kHz, so no resampling is needed here.
//
// It also applies a gentle noise gate. The conditioning chain ahead of this node
// (high-pass, compressor, makeup gain) necessarily lifts the room floor along
// with speech, so quiet stretches are pushed back down rather than handed to the
// recogniser as amplified background. Attenuation is partial and smoothed — hard
// gating clips the soft onset of a word, which costs more than it saves.
class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options?.processorOptions ?? {};
    this.openRms = o.openRms ?? 0.02; // frame level that opens the gate
    this.closeRms = o.closeRms ?? 0.008; // hysteresis, so it can't chatter
    this.closedGain = o.closedGain ?? 0.2; // ~-14dB when closed, not silence
    this.holdFrames = o.holdFrames ?? 30; // ~240ms at 128 frames / 16kHz
    this.enabled = o.gate !== false;

    this.gain = 1;
    this.open = false;
    this.hold = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    let target = 1;
    if (this.enabled) {
      let sum = 0;
      for (let i = 0; i < channel.length; i++) sum += channel[i] * channel[i];
      const rms = Math.sqrt(sum / channel.length);

      if (rms > this.openRms) {
        this.open = true;
        this.hold = this.holdFrames;
      } else if (this.open && rms < this.closeRms && --this.hold <= 0) {
        this.open = false;
      }
      target = this.open ? 1 : this.closedGain;
    }

    const pcm = new Int16Array(channel.length);
    for (let i = 0; i < channel.length; i++) {
      // Ramp the gain per sample so gating is inaudible to the recogniser.
      this.gain += (target - this.gain) * 0.01;
      const s = Math.max(-1, Math.min(1, channel[i] * this.gain));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
