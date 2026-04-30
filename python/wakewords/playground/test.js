import { createWindowSampler, enableMic, inferWav, loadLabelMetadata } from "./audio.js";

const enableButton = document.querySelector("#enableMic");
const statusEl = document.querySelector("#status");
const gridEl = document.querySelector("#labelGrid");
const thresholdInput = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const windowConfigForm = document.querySelector("#windowConfig");
const windowSecondsInput = document.querySelector("#windowSeconds");
const stepMsInput = document.querySelector("#stepMs");
const lamps = new Map();

let sampler = null;
let micStream = null;
let audioContext = null;
let running = false;
let googleSpeechCommands = new Set();
let customWordLabels = new Set();
let threshold = Number(thresholdInput.value) / 100;
let windowSeconds = Number(windowSecondsInput.value);
let stepMs = Number(stepMsInput.value);
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
    micStream = stream;
    audioContext = context;
    resetSampler();
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

windowConfigForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const nextWindowSeconds = Number(windowSecondsInput.value);
  const nextStepMs = Number(stepMsInput.value);
  if (!Number.isFinite(nextWindowSeconds) || nextWindowSeconds <= 0) {
    statusEl.textContent = "window seconds must be greater than 0";
    return;
  }
  if (!Number.isFinite(nextStepMs) || nextStepMs < 1) {
    statusEl.textContent = "step ms must be at least 1";
    return;
  }
  windowSeconds = nextWindowSeconds;
  stepMs = nextStepMs;
  resetSampler();
  statusEl.textContent = `window ${windowSeconds}s, step ${stepMs}ms`;
});

function resetSampler() {
  if (sampler) sampler.stop();
  if (micStream && audioContext) sampler = createWindowSampler(micStream, audioContext, windowSeconds);
}

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
    await sleep(stepMs);
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
      statusEl.textContent = error.message;
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
