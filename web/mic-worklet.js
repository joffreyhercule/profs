// Capture micro : rééchantillonne la fréquence native (souvent 48 kHz) vers 16 kHz et envoie
// des blocs PCM16 de 512 échantillons (32 ms, la taille attendue par Silero), avec le niveau RMS.
const TARGET_RATE = 16000;
const BLOCK = 512;

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / TARGET_RATE; // échantillons d'entrée par échantillon de sortie
    this.pos = 0;     // position fractionnaire du prochain échantillon de sortie
    this.acc = 0;     // somme de la fenêtre courante (filtre moyenneur anti-repliement)
    this.accN = 0;
    this.buf = new Int16Array(BLOCK);
    this.len = 0;
    this.sumSq = 0;
  }

  push(s) {
    s = Math.max(-1, Math.min(1, s));
    this.sumSq += s * s;
    this.buf[this.len++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.len === BLOCK) {
      const rms = Math.sqrt(this.sumSq / BLOCK);
      this.port.postMessage({ pcm: this.buf.buffer, rms }, [this.buf.buffer]);
      this.buf = new Int16Array(BLOCK);
      this.len = 0;
      this.sumSq = 0;
    }
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.acc += input[i];
      this.accN++;
      this.pos += 1;
      if (this.pos >= this.step) {
        this.pos -= this.step;
        this.push(this.acc / this.accN);
        this.acc = 0;
        this.accN = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
