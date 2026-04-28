import { WakewordsListener } from "./listener.js";

export class Wakewords {
  static async load(options) {
    const resolved = await resolveLoadOptions(options);
    return new Wakewords(resolved);
  }

  constructor(options) {
    this.modelUrl = options.modelUrl;
    this.labels = options.labels;
  }

  async predict(input) {
    validateAudioInput(input);
    throw new Error("Wakewords.predict() is not implemented yet.");
  }

  createListener(options) {
    return new WakewordsListener({ wakewords: this, ...options });
  }
}

async function resolveLoadOptions(options) {
  if (!options || typeof options !== "object") {
    throw new TypeError("Wakewords.load() requires an options object.");
  }
  if (typeof options.modelUrl !== "string" || !options.modelUrl) {
    throw new TypeError("Wakewords.load() requires a non-empty modelUrl.");
  }

  if (Array.isArray(options.labels)) {
    return { modelUrl: options.modelUrl, labels: options.labels.slice() };
  }

  if (typeof options.labelsUrl === "string" && options.labelsUrl) {
    const response = await fetch(options.labelsUrl);
    if (!response.ok) {
      throw new Error(`labels failed: ${response.status}`);
    }
    const labels = await response.json();
    if (!Array.isArray(labels) || !labels.every((label) => typeof label === "string")) {
      throw new TypeError("labels response must be a JSON array of strings.");
    }
    return { modelUrl: options.modelUrl, labels };
  }

  return { modelUrl: options.modelUrl, labels: [] };
}

function validateAudioInput(input) {
  if (!input || typeof input !== "object") {
    throw new TypeError("predict() requires an input object.");
  }
  if (!(input.samples instanceof Float32Array)) {
    throw new TypeError("predict() requires samples to be a Float32Array.");
  }
  if (!Number.isFinite(input.sampleRate) || input.sampleRate <= 0) {
    throw new TypeError("predict() requires a positive sampleRate.");
  }
}
