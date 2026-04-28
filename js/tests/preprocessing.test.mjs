import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { mfccFeatures } from "../src/preprocessing.js";

const fixture = JSON.parse(await readFile(new URL("../../tests/fixtures/mfcc-preprocessor.json", import.meta.url), "utf8"));

const tolerance = 3e-4;

for (const testCase of fixture.cases) {
  const actual = mfccFeatures(Float32Array.from(testCase.samples));
  assert.equal(actual.features, testCase.mfcc_shape[0], `${testCase.name}: feature count`);
  assert.equal(actual.frames, testCase.mfcc_shape[1], `${testCase.name}: frame count`);
  assert.equal(actual.data.length, testCase.mfcc.length, `${testCase.name}: flattened MFCC length`);

  let maxDelta = 0;
  for (let index = 0; index < testCase.mfcc.length; index += 1) {
    const delta = Math.abs(actual.data[index] - testCase.mfcc[index]);
    if (delta > maxDelta) maxDelta = delta;
  }
  assert.ok(maxDelta <= tolerance, `${testCase.name}: max MFCC delta ${maxDelta} exceeds ${tolerance}`);
}
