# JS API

## Goals

- Keep one-time inference separate from mic-driven continuous inference.
- Require the app to provide a `MediaStream` for continuous inference.
- Reuse the same preprocessing and ONNX inference path for both APIs.

## Load Model

```ts
import { Wakewords } from "wakewords";

const wakewords = await Wakewords.load({
  modelUrl: "/model.onnx",
  labelsUrl: "/labels.json",
});
```

`labelsUrl` is optional if labels are provided inline.

```ts
const wakewords = await Wakewords.load({
  modelUrl: "/model.onnx",
  labels: ["hey_computer", "background"],
});
```

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

The library does not request mic permission. The app must provide a `MediaStream`.

```ts
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

## Proposed Public API

```ts
class Wakewords {
  static load(options: {
    modelUrl: string;
    labelsUrl?: string;
    labels?: string[];
  }): Promise<Wakewords>;

  predict(input: {
    samples: Float32Array;
    sampleRate: number;
  }): Promise<Prediction>;

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

- `stream` is required for continuous inference.
- `predict()` is the shared inference primitive.
- `createListener()` should sample from the provided stream and repeatedly call `predict()`.
