let sharedStream = null;
let sharedContext = null;
let modelPromise = null;
let labelsPromise = null;
let mfccCache = null;

const MODEL_SAMPLE_RATE = 16000;
const N_FFT = 512;
const WIN_LENGTH = 400;
const HOP_LENGTH = 160;
const N_MELS = 64;
const N_MFCC = 64;
const LOG_MEL_EPSILON = 1e-6;
const TOP_PROBABILITY_COUNT = 5;

export async function enableMic() {
  if (!sharedStream) {
    sharedStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  }
  if (!sharedContext) {
    sharedContext = new AudioContext();
  }
  if (sharedContext.state === "suspended") {
    await sharedContext.resume();
  }
  return { stream: sharedStream, context: sharedContext };
}

export async function loadLabels() {
  if (labelsPromise) return labelsPromise;
  labelsPromise = fetchLabels();
  return labelsPromise;
}

async function fetchLabels() {
  const response = await fetch("/api/labels");
  if (!response.ok) {
    throw new Error(`labels failed: ${response.status}`);
  }
  return response.json();
}

export async function loadLabelMetadata() {
  const response = await fetch("/api/labels/metadata");
  if (!response.ok) {
    throw new Error(`labels failed: ${response.status}`);
  }
  return response.json();
}

export async function inferWav(wavBlob) {
  const [{ session, labels }, samples] = await Promise.all([loadModel(), decodeWav(wavBlob)]);
  return predictSamples(session, labels, samples);
}

export async function inferWavWindows(wavBlob, { windowMs = 1000, stepMs = 100 } = {}) {
  if (windowMs < 1) throw new Error("windowMs must be at least 1");
  if (stepMs < 1) throw new Error("stepMs must be at least 1");
  const [{ session, labels }, samples] = await Promise.all([loadModel(), decodeWav(wavBlob)]);
  const windowSamples = Math.max(1, Math.round((MODEL_SAMPLE_RATE * windowMs) / 1000));
  const stepSamples = Math.max(1, Math.round((MODEL_SAMPLE_RATE * stepMs) / 1000));
  const totalSamples = Math.max(1, samples.length);
  const results = [];

  for (let start = 0; start < totalSamples; start += stepSamples) {
    const windowSamplesBuffer = new Float32Array(windowSamples);
    windowSamplesBuffer.set(samples.slice(start, Math.min(start + windowSamples, samples.length)));
    const result = await predictSamples(session, labels, windowSamplesBuffer);
    results.push({
      ...result,
      startMs: Math.round((start / MODEL_SAMPLE_RATE) * 1000),
      endMs: Math.round(((start + windowSamples) / MODEL_SAMPLE_RATE) * 1000),
    });
  }

  return results;
}

function predictSamples(session, labels, samples) {
  const feeds = buildFeeds(session, samples);
  return session.run(feeds).then((outputs) => predictionFromOutputs(session, labels, outputs));
}

function predictionFromOutputs(session, labels, outputs) {
  const output = outputs[session.outputNames[0]];
  const probabilities = normalizeProbabilities(Array.from(output.data));
  const resultLabels =
    labels.length === probabilities.length
      ? labels
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

export function createRecorder(stream, context) {
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  let chunks = [];
  let recording = false;

  processor.onaudioprocess = (event) => {
    if (!recording) return;
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(context.destination);

  return {
    start() {
      chunks = [];
      recording = true;
    },
    stop() {
      recording = false;
      return encodeWav(merge(chunks), context.sampleRate);
    },
  };
}

export function createWindowSampler(stream, context, seconds = 1.0) {
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const maxSamples = Math.max(1, Math.round(context.sampleRate * seconds));
  let buffer = new Float32Array(0);

  processor.onaudioprocess = (event) => {
    const input = new Float32Array(event.inputBuffer.getChannelData(0));
    const next = new Float32Array(Math.min(maxSamples, buffer.length + input.length));
    const combined = new Float32Array(buffer.length + input.length);
    combined.set(buffer, 0);
    combined.set(input, buffer.length);
    next.set(combined.slice(Math.max(0, combined.length - maxSamples)));
    buffer = next;
  };
  source.connect(processor);
  processor.connect(context.destination);

  return {
    wav() {
      return encodeWav(buffer, context.sampleRate);
    },
    ready() {
      return buffer.length >= maxSamples;
    },
    stop() {
      processor.disconnect();
      source.disconnect();
    },
  };
}

export function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  floatTo16BitPCM(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function merge(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function writeString(view, offset, string) {
  for (let index = 0; index < string.length; index += 1) {
    view.setUint8(offset + index, string.charCodeAt(index));
  }
}

function floatTo16BitPCM(view, offset, input) {
  for (let index = 0; index < input.length; index += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

async function loadModel() {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    if (!window.ort) {
      throw new Error("onnxruntime-web failed to load");
    }
    window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    const [session, labels] = await Promise.all([
      window.ort.InferenceSession.create("/model.onnx", { executionProviders: ["wasm"] }),
      loadLabels(),
    ]);
    return { session, labels };
  })();
  return modelPromise;
}

async function decodeWav(wavBlob) {
  const context = sharedContext || new AudioContext();
  const audioBuffer = await context.decodeAudioData(await wavBlob.arrayBuffer());
  let samples = audioBuffer.getChannelData(0);
  if (audioBuffer.numberOfChannels > 1) {
    const mixed = new Float32Array(audioBuffer.length);
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
      const channelData = audioBuffer.getChannelData(channel);
      for (let index = 0; index < mixed.length; index += 1) {
        mixed[index] += channelData[index] / audioBuffer.numberOfChannels;
      }
    }
    samples = mixed;
  }
  if (audioBuffer.sampleRate !== MODEL_SAMPLE_RATE) {
    return resample(samples, audioBuffer.sampleRate, MODEL_SAMPLE_RATE);
  }
  return new Float32Array(samples);
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
    return new window.ort.Tensor("float32", input, [input.length]);
  }
  if (rank === 2) {
    return new window.ort.Tensor("float32", input, [1, input.length]);
  }
  if (rank === 3 && input.features === N_MFCC) {
    return new window.ort.Tensor("float32", input.data, [1, input.features, input.frames]);
  }
  if (rank === 3) {
    return new window.ort.Tensor("float32", input, [1, 1, input.length]);
  }
  throw new Error(`unsupported ONNX audio input rank: ${rank}`);
}

function lengthTensor(shape, audioInput) {
  const rank = shape?.length || 1;
  const length = audioInput.frames || audioInput.length;
  const data = BigInt64Array.from([BigInt(length)]);
  if (rank === 0) {
    return new window.ort.Tensor("int64", data, []);
  }
  if (rank === 2) {
    return new window.ort.Tensor("int64", data, [1, 1]);
  }
  return new window.ort.Tensor("int64", data, [1]);
}

export function mfccFeatures(samples) {
  const cache = getMfccCache();
  const padded = reflectPad(samples, N_FFT / 2);
  const frames = Math.floor((padded.length - N_FFT) / HOP_LENGTH) + 1;
  const mfcc = new Float32Array(N_MFCC * frames);
  const spectrum = new Array(N_FFT / 2 + 1).fill(0);
  const mel = new Array(N_MELS).fill(0);
  const frameSamples = new Array(N_FFT).fill(0);

  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * HOP_LENGTH;
    for (let index = 0; index < N_FFT; index += 1) {
      frameSamples[index] = padded[offset + index] * cache.window[index];
    }
    for (let bin = 0; bin <= N_FFT / 2; bin += 1) {
      let real = 0;
      let imag = 0;
      const cosRow = cache.cos[bin];
      const sinRow = cache.sin[bin];
      for (let index = 0; index < N_FFT; index += 1) {
        const value = frameSamples[index];
        real += value * cosRow[index];
        imag -= value * sinRow[index];
      }
      spectrum[bin] = real * real + imag * imag;
    }

    for (let melIndex = 0; melIndex < N_MELS; melIndex += 1) {
      let energy = 0;
      const filter = cache.melFilters[melIndex];
      for (const [bin, weight] of filter) {
        energy += spectrum[bin] * weight;
      }
      mel[melIndex] = Math.log(energy + LOG_MEL_EPSILON);
    }

    for (let coeff = 0; coeff < N_MFCC; coeff += 1) {
      let value = 0;
      for (let melIndex = 0; melIndex < N_MELS; melIndex += 1) {
        value += mel[melIndex] * cache.dct[melIndex][coeff];
      }
      mfcc[coeff * frames + frame] = value;
    }
  }

  return { data: mfcc, features: N_MFCC, frames };
}

function getMfccCache() {
  if (mfccCache) return mfccCache;
  mfccCache = {
    window: paddedHannWindow(),
    cos: dftTable(Math.cos),
    sin: dftTable(Math.sin),
    melFilters: melFilterbank(),
    dct: dctMatrix(),
  };
  return mfccCache;
}

function paddedHannWindow() {
  const window = new Float32Array(N_FFT);
  const start = Math.floor((N_FFT - WIN_LENGTH) / 2);
  for (let index = 0; index < WIN_LENGTH; index += 1) {
    window[start + index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / WIN_LENGTH);
  }
  return window;
}

function dftTable(fn) {
  const rows = [];
  for (let bin = 0; bin <= N_FFT / 2; bin += 1) {
    const row = new Float32Array(N_FFT);
    for (let index = 0; index < N_FFT; index += 1) {
      row[index] = fn((2 * Math.PI * bin * index) / N_FFT);
    }
    rows.push(row);
  }
  return rows;
}

function melFilterbank() {
  const filters = Array.from({ length: N_MELS }, () => []);
  const fMin = 0;
  const fMax = MODEL_SAMPLE_RATE / 2;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const points = [];
  for (let index = 0; index < N_MELS + 2; index += 1) {
    const mel = melMin + ((melMax - melMin) * index) / (N_MELS + 1);
    points.push(melToHz(mel));
  }
  const fftFreqs = [];
  for (let bin = 0; bin <= N_FFT / 2; bin += 1) {
    fftFreqs.push((bin * MODEL_SAMPLE_RATE) / N_FFT);
  }
  for (let melIndex = 0; melIndex < N_MELS; melIndex += 1) {
    const lower = points[melIndex];
    const center = points[melIndex + 1];
    const upper = points[melIndex + 2];
    for (let bin = 0; bin < fftFreqs.length; bin += 1) {
      const freq = fftFreqs[bin];
      let weight = 0;
      if (freq >= lower && freq <= center) {
        weight = (freq - lower) / (center - lower);
      } else if (freq > center && freq <= upper) {
        weight = (upper - freq) / (upper - center);
      }
      if (weight > 0) filters[melIndex].push([bin, weight]);
    }
  }
  return filters;
}

function hzToMel(freq) {
  return 2595 * Math.log10(1 + freq / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function dctMatrix() {
  const matrix = [];
  for (let mel = 0; mel < N_MELS; mel += 1) {
    const row = new Float32Array(N_MFCC);
    for (let coeff = 0; coeff < N_MFCC; coeff += 1) {
      let value = Math.cos((Math.PI / N_MELS) * (mel + 0.5) * coeff);
      if (coeff === 0) value *= 1 / Math.sqrt(2);
      row[coeff] = value * Math.sqrt(2 / N_MELS);
    }
    matrix.push(row);
  }
  return matrix;
}

function reflectPad(samples, pad) {
  const result = new Float32Array(samples.length + pad * 2);
  for (let index = 0; index < pad; index += 1) {
    result[index] = samples[pad - index];
    result[pad + samples.length + index] = samples[samples.length - 2 - index];
  }
  result.set(samples, pad);
  return result;
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

function resample(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate) return new Float32Array(samples);
  const targetLength = Math.max(1, Math.round((samples.length * targetRate) / sourceRate));
  const result = new Float32Array(targetLength);
  const scale = (samples.length - 1) / Math.max(1, targetLength - 1);
  for (let index = 0; index < targetLength; index += 1) {
    const position = index * scale;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const weight = position - left;
    result[index] = samples[left] * (1 - weight) + samples[right] * weight;
  }
  return result;
}
