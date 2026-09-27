// Lecteur : file de morceaux PCM (float32, 24 kHz) étiquetés (tour, segment).
// Signale le début de chaque segment (pour savoir ce que l'élève a réellement entendu)
// et la fin de lecture ; "flush" vide tout instantanément (interruption du prof).
class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.current = null;
    this.playing = false;
    this.port.onmessage = ({ data }) => {
      if (data.type === "push") {
        this.queue.push(data);
      } else if (data.type === "flush") {
        this.queue = [];
        this.offset = 0;
        this.current = null;
        this.playing = false;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let i = 0;
    while (i < out.length && this.queue.length) {
      const chunk = this.queue[0];
      const key = chunk.turn + ":" + chunk.seg;
      if (key !== this.current) {
        this.current = key;
        this.port.postMessage({ type: "seg_start", turn: chunk.turn, seg: chunk.seg });
      }
      this.playing = true;
      const n = Math.min(out.length - i, chunk.samples.length - this.offset);
      out.set(chunk.samples.subarray(this.offset, this.offset + n), i);
      i += n;
      this.offset += n;
      if (this.offset >= chunk.samples.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    out.fill(0, i);
    if (this.playing && !this.queue.length) {
      this.playing = false;
      this.port.postMessage({ type: "drained" });
    }
    return true;
  }
}

registerProcessor("player-processor", PlayerProcessor);
