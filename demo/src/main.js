import { Wakewords } from "wakewords";

import "./style.css";

const modelUrlInput = document.querySelector("#modelUrl");
const labelsUrlInput = document.querySelector("#labelsUrl");
const loadButton = document.querySelector("#loadModel");
const enableButton = document.querySelector("#enableMic");
const thresholdInput = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const statusEl = document.querySelector("#status");
const gridEl = document.querySelector("#labelGrid");

const lamps = new Map();

let wakewords = null;
let listener = null;
let stream = null;
let threshold = Number(thresholdInput.value) / 100;

loadButton.addEventListener("click", async () => {
  const modelUrl = modelUrlInput.value.trim();
  const labelsUrl = labelsUrlInput.value.trim();

  if (!modelUrl) {
    statusEl.textContent = "enter a model URL";
    return;
  }

  if (!labelsUrl) {
    statusEl.textContent = "enter a labels URL";
    return;
  }

  loadButton.disabled = true;
  enableButton.disabled = true;
  statusEl.textContent = "loading model";

  try {
    wakewords = await Wakewords.load({
      modelUrl,
      labelsUrl,
    });
    renderLamps(wakewords.labels);
    enableButton.disabled = false;
    statusEl.textContent = "model ready";
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    loadButton.disabled = false;
  }
});

enableButton.addEventListener("click", async () => {
  if (listener?.running) {
    stopListening();
    statusEl.textContent = "stopped";
    return;
  }

  if (!wakewords) {
    statusEl.textContent = "load a model first";
    return;
  }

  enableButton.disabled = true;
  statusEl.textContent = "requesting mic";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    listener = wakewords.createListener({
      stream,
      intervalMs: 350,
      windowSeconds: 1.0,
    });
    listener.addEventListener("prediction", handlePrediction);
    listener.addEventListener("error", handleError);
    await listener.start();
    enableButton.textContent = "Stop mic";
    statusEl.textContent = "listening";
  } catch (error) {
    statusEl.textContent = error.message;
    stopListening();
  } finally {
    enableButton.disabled = false;
  }
});

thresholdInput.addEventListener("input", () => {
  threshold = Number(thresholdInput.value) / 100;
  thresholdValue.textContent = `${thresholdInput.value}%`;
});

window.addEventListener("beforeunload", () => {
  stopListening();
});

function renderLamps(labels) {
  lamps.clear();
  const lampNodes = labels.map((label) => createLamp(label));
  gridEl.replaceChildren(...lampNodes);
}

function createLamp(label) {
  const element = document.createElement("div");
  element.className = "lamp";
  element.dataset.label = label;
  element.textContent = label;
  lamps.set(label, element);
  return element;
}

function handlePrediction(event) {
  const result = event.detail;
  statusEl.textContent = `${result.label} ${Math.round(result.probability * 100)}%`;
  if (result.probability < threshold) return;
  activate(result.label);
}

function handleError(event) {
  statusEl.textContent = event.error?.message || "inference failed";
}

function activate(label) {
  const lamp = lamps.get(label);
  if (!lamp) return;
  lamp.classList.remove("active");
  window.requestAnimationFrame(() => {
    lamp.classList.add("active");
    window.setTimeout(() => {
      lamp.classList.remove("active");
    }, 120);
  });
}

function stopListening() {
  listener?.destroy();
  listener = null;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  stream = null;
  enableButton.textContent = "Enable mic";
}
