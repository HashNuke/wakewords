import { createWindowSampler, enableMic, inferWav, loadLabelMetadata } from "./audio.js";

const enableButton = document.querySelector("#enableMic");
const statusEl = document.querySelector("#status");
const gridEl = document.querySelector("#labelGrid");
const lamps = new Map();

let sampler = null;
let running = false;
let googleSpeechCommands = new Set();

loadLabelMetadata()
  .then((metadata) => {
    googleSpeechCommands = new Set(metadata.google_speech_commands || []);
    const customLabels = metadata.custom || [];
    const otherLabels = metadata.other || [];
    const lampsToShow = [...customLabels, ...otherLabels].map((label) => createLamp(label));
    if (googleSpeechCommands.size > 0) {
      lampsToShow.push(createLamp("google speech", { key: "google-speech", className: "google-lamp" }));
    }
    gridEl.replaceChildren(...lampsToShow);
  })
  .catch((error) => {
    statusEl.textContent = error.message;
  });

enableButton.addEventListener("click", async () => {
  try {
    const { stream, context } = await enableMic();
    sampler = createWindowSampler(stream, context, 1.0);
    running = true;
    enableButton.disabled = true;
    enableButton.textContent = "👍 Mic enabled!";
    statusEl.textContent = "listening";
    loop();
  } catch (error) {
    statusEl.textContent = error.message;
  }
});

function createLamp(label, options = {}) {
  const element = document.createElement("div");
  element.className = options.className ? `lamp ${options.className}` : "lamp";
  element.dataset.label = label;
  element.textContent = label;
  lamps.set(options.key || label, element);
  return element;
}

async function loop() {
  while (running) {
    await sleep(350);
    if (!sampler || !sampler.ready()) continue;
    try {
      const result = await inferWav(sampler.wav());
      console.log("wakewords inference", result);
      if (result.probability >= 0.7) {
        if (googleSpeechCommands.has(result.label)) {
          activate("google-speech", result.label);
        } else {
          activate(result.label);
        }
      }
    } catch (error) {
      console.log("wakewords inference error", error);
    }
  }
}

function activate(key, label = key) {
  const lamp = lamps.get(key);
  if (!lamp) return;
  lamp.textContent = label;
  lamp.classList.remove("active");
  window.requestAnimationFrame(() => {
    lamp.classList.add("active");
    window.setTimeout(() => lamp.classList.remove("active"), 120);
  });
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
