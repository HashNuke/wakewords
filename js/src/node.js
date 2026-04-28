import { readFile } from "node:fs/promises";

import * as ort from "onnxruntime-node";

import { mfccFeatures, resample } from "./preprocessing.js";
import { createWakewordsClass } from "./wakewords-core.js";

const runtime = {
  ort,
  loadText: async (source) => {
    if (source instanceof URL) {
      if (source.protocol === "file:") {
        return readFile(source, "utf8");
      }
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`labels failed: ${response.status}`);
      }
      return response.text();
    }
    if (typeof source === "string" && /^https?:\/\//.test(source)) {
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`labels failed: ${response.status}`);
      }
      return response.text();
    }
    return readFile(source, "utf8");
  },
  onnxOptions: (options) => options.onnxOptions ?? {},
};

export const Wakewords = createWakewordsClass(runtime);
export { mfccFeatures, resample };
