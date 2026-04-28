import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const packageRoot = await import("wakewords");
const packageNode = await import("wakewords/node");
const packageBrowser = await import("wakewords/browser");
const fixturesUrl = new URL("../../tests/fixtures/", import.meta.url);
const defaultModelPath = fileURLToPath(import.meta.resolve("wakewords/model.onnx"));
const defaultLabelsPath = fileURLToPath(import.meta.resolve("wakewords/labels.json"));
const labels = JSON.parse(await readFile(new URL("labels.json", fixturesUrl), "utf8"));

assert.equal(await readFile(defaultModelPath).then((data) => data.length > 0), true, "default model is exported");
assert.deepEqual(JSON.parse(await readFile(defaultLabelsPath, "utf8")), labels, "default labels are exported");

assert.equal("Wakewords" in packageRoot, true, "root exports Wakewords");
assert.equal("mfccFeatures" in packageRoot, true, "root exports mfccFeatures");
assert.equal("resample" in packageRoot, true, "root exports resample");
assert.equal("WakewordsListener" in packageRoot, false, "root should not export WakewordsListener");

assert.equal("Wakewords" in packageNode, true, "node entry exports Wakewords");
assert.equal("WakewordsListener" in packageNode, false, "node entry should not export WakewordsListener");

assert.equal("Wakewords" in packageBrowser, true, "browser entry exports Wakewords");
assert.equal("WakewordsListener" in packageBrowser, true, "browser entry exports WakewordsListener");

const browserWakewords = await packageBrowser.Wakewords.load({ session: {}, labels: [] });
assert.equal(typeof browserWakewords.createListener, "function", "browser load returns listener-capable instance");

const modelUrl = fileURLToPath(new URL("model.onnx", fixturesUrl));
const labelsUrl = fileURLToPath(new URL("labels.json", fixturesUrl));
const wav = await readWav(new URL("speech-commands/backward/017c4098_nohash_0.wav", fixturesUrl));

for (const entry of [packageRoot, packageNode]) {
  const wakewords = await entry.Wakewords.load({ modelUrl, labels });
  const result = await wakewords.predict(wav);
  assert.equal(result.label, "backward");
  assert.ok(result.probability > 0.4);
}

for (const entry of [packageRoot, packageNode]) {
  const wakewords = await entry.Wakewords.load({ modelUrl, labelsUrl });
  const result = await wakewords.predict(wav);
  assert.equal(result.label, "backward");
  assert.ok(result.probability > 0.4);
}

for (const entry of [packageRoot, packageNode]) {
  const wakewords = await entry.Wakewords.load();
  const result = await wakewords.predict(wav);
  assert.equal(result.label, "backward");
  assert.ok(result.probability > 0.4);
}

async function readWav(url) {
  const buffer = await readFile(url);
  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`Expected WAV file: ${url}`);
  }

  let offset = 12;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataStart = 0;
  let dataLength = 0;

  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkStart = offset + 8;

    if (chunkId === "fmt ") {
      const audioFormat = buffer.readUInt16LE(chunkStart);
      channels = buffer.readUInt16LE(chunkStart + 2);
      sampleRate = buffer.readUInt32LE(chunkStart + 4);
      bitsPerSample = buffer.readUInt16LE(chunkStart + 14);
      if (audioFormat !== 1) throw new Error(`Expected PCM WAV: ${url}`);
    } else if (chunkId === "data") {
      dataStart = chunkStart;
      dataLength = chunkSize;
    }

    offset = chunkStart + chunkSize + (chunkSize % 2);
  }

  if (!channels || !sampleRate || bitsPerSample !== 16 || !dataStart || !dataLength) {
    throw new Error(`Unsupported WAV file: ${url}`);
  }

  const frameCount = Math.floor(dataLength / 2 / channels);
  const samples = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sample = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const byteOffset = dataStart + (frame * channels + channel) * 2;
      sample += buffer.readInt16LE(byteOffset) / 32768;
    }
    samples[frame] = sample / channels;
  }

  return { samples, sampleRate };
}
