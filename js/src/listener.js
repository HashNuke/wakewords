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
  }

  async start() {
    if (this.running) return;
    throw new Error("WakewordsListener.start() is not implemented yet.");
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    this.dispatchEvent(new Event("stop"));
  }

  destroy() {
    this.stop();
  }
}
