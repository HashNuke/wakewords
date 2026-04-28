import * as ort from "onnxruntime-web";

import { WakewordsListener } from "./listener.js";
import { mfccFeatures, resample } from "./preprocessing.js";
import { createWakewordsClass } from "./wakewords-core.js";

const BaseWakewords = createWakewordsClass({
  ort,
  defaultModelUrl: new URL("../assets/model.onnx", import.meta.url).href,
  defaultLabelsUrl: new URL("../assets/labels.json", import.meta.url).href,
  loadText: async (source) => {
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`labels failed: ${response.status}`);
    }
    return response.text();
  },
  onnxOptions: (options) => options.onnxOptions ?? { executionProviders: ["wasm"] },
});

export class Wakewords extends BaseWakewords {
  createListener(options) {
    return new WakewordsListener({ wakewords: this, ...options });
  }
}

export { WakewordsListener, mfccFeatures, resample };
