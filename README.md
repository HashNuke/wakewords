# `wakewords` detection for your apps & webpages

![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![NodeJS](https://img.shields.io/badge/node.js-6DA55F.svg?style=for-the-badge&logo=node.js&logoColor=white)
![Swift](https://img.shields.io/badge/swift-F54A2A?style=for-the-badge&logo=swift&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

> Add your wakewords like *"alexa"*, *"hey siri"* in your apps.

* Works in the browser with Javascript, Node.js, Swift and Python.
* Train your own custom wakewords detection model in 30 min (Yes, you can do "Jarvis", "Computer", etc)

## Quick Start: Detect wakewords in your web pages

```sh
npm install wakewords
```

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

**Default wake words**
The default model is trained for wakewords from Google Speech Commands dataset v2. List of words below.

|  |  |  |  |  |  |
|---|---|---|---|---|---|
| backward | bed | bird | cat | dog | down |
| eight | five | follow | forward | four | go |
| happy | house | learn | left | marvin | nine |
| no | off | on | one | right | seven |
| sheila | six | stop | three | tree | two |
| up | visual | wow | yes | zero | |

> It does include 12 other words mentioned in the labels.json in this repo. That shouldnt matter if you only want the words from the Speech Commands dataset.

## Quick Start: Train for your own wakewords

```sh
pip install wakewords
```

> Use `uv run` for below commands if applicable to you.

```sh
wakewords init

# Edit your custom wake words
$EDITOR config.json

export CARTESIA_API_KEY=your-api-key

# Generates synthetic dataset
wakewords generate --lang en --all-voices
wakewords augment

# Train your model
wakewords download
wakewords manifest
wakewords train

# Export to onnx and use whereever
wakewords export
```

## Docs

* [JavaScript API](docs/javascript-api.md)
* [Python training](python/README.md)
* [Swift package](swift/README.md)

## License

Copyright &copy; 2026 Akash Manohar John, under MIT License (See LICENSE file).
