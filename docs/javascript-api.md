# JS API

## Goals

- Keep one-time inference separate from mic-driven continuous inference.
- Require the app to provide a `MediaStream` for continuous inference.
- Reuse the same preprocessing and ONNX inference path for both APIs.
- Ship one npm package with runtime-specific entrypoints for browser and Node.js.

## Entrypoints

```ts
import { Wakewords } from "wakewords";
import { Wakewords as BrowserWakewords, WakewordsListener } from "wakewords/browser";
import { Wakewords as NodeWakewords } from "wakewords/node";
```

- `wakewords` exports inference-only APIs and resolves to the best runtime entrypoint through package exports.
- `wakewords/browser` exports browser inference plus `WakewordsListener` and `createListener()`.
- `wakewords/node` exports Node.js inference with local filesystem support for model and label paths.
- `wakewords/model.onnx` and `wakewords/labels.json` expose the bundled default assets for tooling that needs direct asset paths.

## Load Model

Load the bundled default model and labels:

```ts
import { Wakewords } from "wakewords";

const wakewords = await Wakewords.load();
```

The default model and labels are shipped in the npm package. In Node.js they are loaded from package files. In browser builds they are referenced with `new URL(..., import.meta.url)` so bundlers can emit them as static assets.

Pass custom model and label locations when you want to use your own trained model.

```ts
import { Wakewords } from "wakewords";

const wakewords = await Wakewords.load({
  modelUrl: "/custom-model.onnx",
  labelsUrl: "/custom-labels.json",
});
```

`labelsUrl` is optional if labels are provided inline. If `modelUrl` or `modelData` is provided without labels, predictions fall back to `class_0`, `class_1`, and so on.

```ts
const wakewords = await Wakewords.load({
  modelUrl: "/custom-model.onnx",
  labels: ["hey_computer", "background"],
});
```

In Node.js, `labelsUrl` can be a local path, a `file:` URL, or an HTTP(S) URL.

```ts
import { Wakewords } from "wakewords/node";

const wakewords = await Wakewords.load({
  modelUrl: "./models/model.onnx",
  labelsUrl: "./models/labels.json",
});
```

If the app already has model bytes or a created ONNX session, pass those directly.

```ts
const wakewords = await Wakewords.load({
  modelData: modelBytes,
  labels: ["hey_computer", "background"],
});
```

## Bundled Assets

Most apps should call `Wakewords.load()` and let the library resolve the bundled assets. If tooling needs direct access to the files, use the exported package subpaths:

```ts
const modelUrl = import.meta.resolve("wakewords/model.onnx");
const labelsUrl = import.meta.resolve("wakewords/labels.json");
```

The direct asset exports are primarily for build tooling and diagnostics. They are not required for normal inference.

## One-Time Inference

```ts
const result = await wakewords.predict({
  samples: float32Samples,
  sampleRate: 16000,
});
```

Expected input:

```ts
{
  samples: Float32Array;
  sampleRate: number;
}
```

Result shape:

```ts
type Prediction = {
  label: string;
  probability: number;
  probabilities: Record<string, number>;
  topProbabilities: Array<{ label: string; probability: number }>;
};
```

## Continuous Inference

Continuous inference is browser-only. Import from `wakewords/browser` so the listener APIs are available. The library does not request mic permission. The app must provide a `MediaStream`.

```ts
import { Wakewords } from "wakewords/browser";

const wakewords = await Wakewords.load();

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

const listener = wakewords.createListener({
  stream,
  intervalMs: 350,
  windowSeconds: 1.0,
  eventName: "wakewords:prediction",
});

listener.addEventListener("prediction", (event) => {
  console.log(event.detail.label, event.detail.probability);
});

await listener.start();
listener.stop();
listener.destroy();
```

The listener should dispatch `prediction` events with `Prediction` as `event.detail`.

## Public API

```ts
class Wakewords {
  static load(options: {
    modelUrl?: string | URL;
    modelData?: ArrayBuffer | Uint8Array;
    session?: InferenceSession;
    labelsUrl?: string | URL;
    labels?: string[];
    onnxOptions?: Record<string, unknown>;
  }): Promise<Wakewords>;

  predict(input: {
    samples: Float32Array;
    sampleRate: number;
  }): Promise<Prediction>;
}

class BrowserWakewords extends Wakewords {
  static load(options: WakewordsLoadOptions): Promise<BrowserWakewords>;

  createListener(options: {
    stream: MediaStream;
    intervalMs?: number;
    windowSeconds?: number;
    eventName?: string;
  }): WakewordsListener;
}

type WakewordsListener = EventTarget & {
  start(): Promise<void>;
  stop(): void;
  destroy(): void;
  readonly running: boolean;
};
```

## Notes

- `wakewords` and `wakewords/node` do not export `WakewordsListener`.
- `wakewords/browser` uses `onnxruntime-web` and defaults to the WASM execution provider.
- `wakewords/node` uses `onnxruntime-node` and supports local label file paths.
- `stream` is required for continuous inference.
- `predict()` is the shared inference primitive.
- `createListener()` samples from the provided stream and repeatedly calls `predict()`.
