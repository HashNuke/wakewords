import * as ort from "onnxruntime-web";

import { WakewordsListener } from "./listener.js";
import { mfccFeatures, resample } from "./preprocessing.js";

const MODEL_SAMPLE_RATE = 16000;
const N_MFCC = 64;
const TOP_PROBABILITY_COUNT = 5;

export class Wakewords {
  static async load(options) {
    const resolved = await resolveLoadOptions(options);
    return new Wakewords(resolved);
  }

  constructor(options) {
    this.modelUrl = options.modelUrl;
    this.labels = options.labels;
    this.session = options.session;
  }

  async predict(input) {
    validateAudioInput(input);
    const samples =
      input.sampleRate === MODEL_SAMPLE_RATE
        ? new Float32Array(input.samples)
        : resample(input.samples, input.sampleRate, MODEL_SAMPLE_RATE);
    const feeds = buildFeeds(this.session, samples);
    const outputs = await this.session.run(feeds);
    const output = outputs[this.session.outputNames[0]];
    const probabilities = normalizeProbabilities(Array.from(output.data));
    const resultLabels =
      this.labels.length === probabilities.length
        ? this.labels
        : probabilities.map((_, index) => `class_${index}`);

    let bestIndex = 0;
    for (let index = 1; index < probabilities.length; index += 1) {
      if (probabilities[index] > probabilities[bestIndex]) bestIndex = index;
    }

    return {
      label: resultLabels[bestIndex],
      probability: probabilities[bestIndex],
      probabilities: Object.fromEntries(resultLabels.map((label, index) => [label, probabilities[index]])),
      topProbabilities: topProbabilities(resultLabels, probabilities),
    };
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
    return {
      modelUrl: options.modelUrl,
      labels: options.labels.slice(),
      session: await ort.InferenceSession.create(options.modelUrl, { executionProviders: ["wasm"] }),
    };
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
    return {
      modelUrl: options.modelUrl,
      labels,
      session: await ort.InferenceSession.create(options.modelUrl, { executionProviders: ["wasm"] }),
    };
  }

  return {
    modelUrl: options.modelUrl,
    labels: [],
    session: await ort.InferenceSession.create(options.modelUrl, { executionProviders: ["wasm"] }),
  };
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

function buildFeeds(session, samples) {
  const inputShapes = inputShapeMap(session);
  const audioShape = audioInputShape(session, inputShapes);
  const audioInput = expectsMfcc(audioShape) ? mfccFeatures(samples) : new Float32Array(samples);
  const feeds = {};
  for (const name of session.inputNames) {
    const normalized = name.toLowerCase();
    if (normalized.includes("length")) {
      feeds[name] = lengthTensor(inputShapes.get(name), audioInput);
    } else if (normalized.includes("audio") || normalized.includes("signal") || session.inputNames.length === 1) {
      feeds[name] = audioTensor(audioInput, audioShape);
    } else {
      throw new Error(`unsupported ONNX input: ${name}`);
    }
  }
  return feeds;
}

function inputShapeMap(session) {
  const result = new Map();
  const metadata = session.inputMetadata;
  if (!metadata) return result;
  if (typeof metadata.get === "function") {
    for (const name of session.inputNames) {
      const input = metadata.get(name);
      const shape = input?.dimensions || input?.dims || input?.shape;
      if (Array.isArray(shape)) result.set(name, shape);
    }
    return result;
  }
  if (Array.isArray(metadata)) {
    for (const input of metadata) {
      const shape = input?.dimensions || input?.dims || input?.shape;
      if (input?.name && Array.isArray(shape)) result.set(input.name, shape);
    }
    return result;
  }
  for (const name of session.inputNames) {
    const input = metadata[name];
    const shape = input?.dimensions || input?.dims || input?.shape;
    if (Array.isArray(shape)) result.set(name, shape);
  }
  return result;
}

function audioInputShape(session, inputShapes) {
  for (const name of session.inputNames) {
    const normalized = name.toLowerCase();
    if (normalized.includes("audio") || normalized.includes("signal")) {
      return inputShapes.get(name) || [1, N_MFCC, "time"];
    }
  }
  return inputShapes.get(session.inputNames[0]) || [1, N_MFCC, "time"];
}

function expectsMfcc(shape) {
  return shape.length === 3 && shape[1] === N_MFCC;
}

function audioTensor(input, shape) {
  const rank = shape.length || 2;
  if (rank === 1) {
    return new ort.Tensor("float32", input, [input.length]);
  }
  if (rank === 2) {
    return new ort.Tensor("float32", input, [1, input.length]);
  }
  if (rank === 3 && input.features === N_MFCC) {
    return new ort.Tensor("float32", input.data, [1, input.features, input.frames]);
  }
  if (rank === 3) {
    return new ort.Tensor("float32", input, [1, 1, input.length]);
  }
  throw new Error(`unsupported ONNX audio input rank: ${rank}`);
}

function lengthTensor(shape, audioInput) {
  const rank = shape?.length || 1;
  const length = audioInput.frames || audioInput.length;
  const data = BigInt64Array.from([BigInt(length)]);
  if (rank === 0) {
    return new ort.Tensor("int64", data, []);
  }
  if (rank === 2) {
    return new ort.Tensor("int64", data, [1, 1]);
  }
  return new ort.Tensor("int64", data, [1]);
}

function normalizeProbabilities(values) {
  const sum = values.reduce((total, value) => total + value, 0);
  const alreadyProbabilities =
    values.length > 0 && values.every((value) => value >= 0 && value <= 1) && sum >= 0.99 && sum <= 1.01;
  if (alreadyProbabilities) {
    return values.map((value) => value / sum);
  }
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const total = exp.reduce((partial, value) => partial + value, 0);
  return exp.map((value) => value / total);
}

function topProbabilities(labels, probabilities) {
  return labels
    .map((label, index) => ({ label, probability: probabilities[index] }))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, TOP_PROBABILITY_COUNT);
}
