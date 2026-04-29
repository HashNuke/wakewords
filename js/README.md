# wakewords (JavaScript)

Detect wakewords in browsers and Node.js.

## Install

```sh
npm install wakewords
```

## Browser microphone detection

```js
import { Wakewords } from "wakewords/browser";

const wakewords = await Wakewords.load();
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const listener = wakewords.createListener({ stream });

listener.addEventListener("prediction", (event) => {
  console.log(event.detail.label, event.detail.probability);
});

await listener.start();
```

## One-shot inference

```js
import { Wakewords } from "wakewords";

const wakewords = await Wakewords.load();

const result = await wakewords.predict({
  samples: float32Samples,
  sampleRate: 16000,
});
```

## Custom model

```js
import { Wakewords } from "wakewords";

const wakewords = await Wakewords.load({
  modelUrl: "/models/model.onnx",
  labelsUrl: "/models/labels.json",
});
```

## Node.js

```js
import { Wakewords } from "wakewords/node";

const wakewords = await Wakewords.load({
  modelUrl: "./models/model.onnx",
  labelsUrl: "./models/labels.json",
});
```

## More Details

See [`../docs/javascript-api.md`](../docs/javascript-api.md) for entrypoints, listener options, custom labels, bundled assets, and result shapes.

## License

Copyright &copy; 2026 Akash Manohar John, under MIT License (See LICENSE file at root of git repo).
