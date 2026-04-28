export class WakewordsListener extends EventTarget {
  constructor(options) {
    super();

    if (!options || typeof options !== "object") {
      throw new TypeError("createListener() requires an options object.");
    }
    if (!options.wakewords) {
      throw new TypeError("createListener() requires a wakewords instance.");
    }
    if (!(options.stream instanceof MediaStream)) {
      throw new TypeError("createListener() requires a MediaStream.");
    }

    this.wakewords = options.wakewords;
    this.stream = options.stream;
    this.intervalMs = options.intervalMs ?? 350;
    this.windowSeconds = options.windowSeconds ?? 1.0;
    this.eventName = options.eventName ?? "wakewords:prediction";
    this.running = false;
    this.context = null;
    this.source = null;
    this.processor = null;
    this.buffer = new Float32Array(0);
    this.maxSamples = 0;
    this.loopPromise = null;
  }

  async start() {
    if (this.running) return;

    const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("AudioContext is not available in this environment.");
    }

    this.context = new AudioContextClass();
    if (this.context.state === "suspended") {
      await this.context.resume();
    }

    this.maxSamples = Math.max(1, Math.round(this.context.sampleRate * this.windowSeconds));
    this.buffer = new Float32Array(0);
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => {
      const input = new Float32Array(event.inputBuffer.getChannelData(0));
      const next = new Float32Array(Math.min(this.maxSamples, this.buffer.length + input.length));
      const combined = new Float32Array(this.buffer.length + input.length);
      combined.set(this.buffer, 0);
      combined.set(input, this.buffer.length);
      next.set(combined.slice(Math.max(0, combined.length - this.maxSamples)));
      this.buffer = next;
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
    this.running = true;
    this.dispatchEvent(new Event("start"));
    this.loopPromise = this.runLoop();
    await Promise.resolve();
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    this.processor?.disconnect();
    this.source?.disconnect();
    if (this.context && this.context.state !== "closed") {
      void this.context.close();
    }
    this.processor = null;
    this.source = null;
    this.context = null;
    this.buffer = new Float32Array(0);
    this.maxSamples = 0;
    this.dispatchEvent(new Event("stop"));
  }

  destroy() {
    this.stop();
  }

  async runLoop() {
    while (this.running) {
      await sleep(this.intervalMs);
      if (!this.running || !this.context || !this.ready()) continue;

      try {
        const result = await this.wakewords.predict({
          samples: this.samples(),
          sampleRate: this.context.sampleRate,
        });
        this.dispatchPrediction(result);
      } catch (error) {
        this.dispatchEvent(new ErrorEvent("error", { error }));
      }
    }
  }

  ready() {
    return this.buffer.length >= this.maxSamples * 0.75;
  }

  samples() {
    return new Float32Array(this.buffer);
  }

  dispatchPrediction(result) {
    const event = new CustomEvent("prediction", { detail: result });
    this.dispatchEvent(event);
    if (this.eventName !== "prediction") {
      this.dispatchEvent(new CustomEvent(this.eventName, { detail: result }));
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms));
}
