let sharedStream = null;
let sharedContext = null;
let modelPromise = null;
let labelsPromise = null;

const MODEL_SAMPLE_RATE = 16000;

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
  const feeds = buildFeeds(session, samples);
  const outputs = await session.run(feeds);
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
      return buffer.length >= maxSamples * 0.75;
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
  const signal = new Float32Array(samples);
  const feeds = {};
  for (const name of session.inputNames) {
    const normalized = name.toLowerCase();
    if (normalized.includes("length")) {
      feeds[name] = new window.ort.Tensor("int64", BigInt64Array.from([BigInt(signal.length)]), [1]);
    } else if (normalized.includes("audio") || normalized.includes("signal") || session.inputNames.length === 1) {
      feeds[name] = new window.ort.Tensor("float32", signal, [1, signal.length]);
    } else {
      throw new Error(`unsupported ONNX input: ${name}`);
    }
  }
  return feeds;
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
