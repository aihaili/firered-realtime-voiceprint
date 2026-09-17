// Mic capture worklet: resample to 16kHz, pack PCM16, post chunks.
class MicWorklet extends AudioWorkletProcessor {
  constructor(options = {}) {
    super();
    this._srcRate = (options && options.sampleRate) || 48000;
    this._buf = [];
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    // 关键：必须立即拷贝！引擎会复用/覆写这块输入缓冲区，
    // 存引用攒批会导致整段音频变成同一片段的重复
    this._buf.push(new Float32Array(ch));
    // flush when we have >= 0.5s of source samples
    while (this._buf.length * ch.length >= this._srcRate * 0.5) {
      const total = this._buf.reduce((n, b) => n + b.length, 0);
      const src = new Float32Array(total);
      let off = 0;
      for (const b of this._buf) { src.set(b, off); off += b.length; }
      this._buf = [];
      const out = this._resample(src, 16000);
      const pcm = new Int16Array(out.length);
      for (let i = 0; i < out.length; i++) {
        let v = Math.max(-1, Math.min(1, out[i]));
        pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
  _resample(x, target) {
    const sr = this._srcRate;
    if (sr === target) return x;
    const ratio = sr / target;
    const n = Math.floor(x.length / ratio);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, x.length - 1);
      const f = idx - i0;
      out[i] = x[i0] * (1 - f) + x[i1] * f;
    }
    return out;
  }
}
registerProcessor("mic-capture", MicWorklet);
