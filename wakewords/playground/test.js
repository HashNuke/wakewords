import { createWindowSampler, enableMic, inferWav, loadLabelMetadata } from "./audio.js";

const enableButton = document.querySelector("#enableMic");
const statusEl = document.querySelector("#status");
const gridEl = document.querySelector("#labelGrid");
const thresholdInput = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const lamps = new Map();

let sampler = null;
let running = false;
let googleSpeechCommands = new Set();
let customWordLabels = new Set();
let threshold = Number(thresholdInput.value) / 100;
let googleSpeechReset = null;

loadLabelMetadata()
  .then((metadata) => {
    googleSpeechCommands = new Set(metadata.google_speech_commands || []);
    const customLabels = metadata.custom || [];
    customWordLabels = new Set(customLabels);
    const otherLabels = metadata.other || [];
    const lampsToShow = [...customLabels, ...otherLabels].map((label) => createLamp(label));
    if (googleSpeechCommands.size > 0) {
      lampsToShow.push(createLamp("....", { key: "google-speech", className: "google-lamp" }));
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

thresholdInput.addEventListener("input", () => {
  threshold = Number(thresholdInput.value) / 100;
  thresholdValue.textContent = `${thresholdInput.value}%`;
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
      if (result.probability >= threshold) {
        if (googleSpeechCommands.has(result.label) && !customWordLabels.has(result.label)) {
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
    window.setTimeout(() => {
      lamp.classList.remove("active");
      if (key === "google-speech") resetGoogleSpeechLamp();
    }, 120);
  });
}

function resetGoogleSpeechLamp() {
  window.clearTimeout(googleSpeechReset);
  googleSpeechReset = window.setTimeout(() => {
    const lamp = lamps.get("google-speech");
    if (lamp && !lamp.classList.contains("active")) lamp.textContent = "....";
  }, 2000);
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
