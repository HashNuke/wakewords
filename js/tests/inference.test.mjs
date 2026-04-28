import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { Wakewords } from "../src/index.js";

const fixturesUrl = new URL("../../tests/fixtures/", import.meta.url);
const labels = JSON.parse(await readFile(new URL("labels.json", fixturesUrl), "utf8"));
const modelUrl = fileURLToPath(new URL("model.onnx", fixturesUrl));
const wakewords = await Wakewords.load({ modelUrl, labels });

const cases = [
  ["backward", "017c4098_nohash_0.wav"],
  ["backward", "017c4098_nohash_1.wav"],
  ["forward", "012187a4_nohash_0.wav"],
  ["forward", "017c4098_nohash_1.wav"],
  ["happy", "00970ce1_nohash_1.wav"],
  ["happy", "012187a4_nohash_0.wav"],
];

for (const [label, filename] of cases) {
  const wav = await readWav(new URL(`speech-commands/${label}/${filename}`, fixturesUrl));
  const result = await wakewords.predict(wav);
  assert.equal(result.label, label, `${filename}: expected ${label}, got ${result.label}`);
  assert.ok(result.probability > 0.4, `${filename}: expected confidence over 40%, got ${result.probability}`);
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
