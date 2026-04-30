import { createRecorder, enableMic, inferWavWindows, loadLabelMetadata } from "./audio.js";

const INFERENCE_WINDOW_MS = 1000;
const INFERENCE_STEP_MS = 100;

const enableButton = document.querySelector("#enableMic");
const labelSelect = document.querySelector("#labelSelect");
const statusEl = document.querySelector("#status");
const panelEl = document.querySelector("#recordPanel");
const readoutEl = document.querySelector("#readout");
const probabilityEl = document.querySelector("#probability");

let recorder = null;
let isRecording = false;

loadLabelMetadata()
  .then((metadata) => {
    const labels = metadata.custom || [];
    labelSelect.replaceChildren(...labels.map((label) => new Option(label, label)));
    if (labels.length === 0) {
      statusEl.textContent = "no custom labels";
    }
  })
  .catch((error) => {
    statusEl.textContent = error.message;
  });

enableButton.addEventListener("click", async () => {
  try {
    const { stream, context } = await enableMic();
    recorder = createRecorder(stream, context);
    enableButton.disabled = true;
    enableButton.textContent = "👍 Mic enabled!";
    statusEl.textContent = "ready";
  } catch (error) {
    statusEl.textContent = error.message;
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() !== "z" || event.repeat || !recorder || isRecording) return;
  isRecording = true;
  panelEl.classList.add("recording");
  readoutEl.textContent = "recording";
  probabilityEl.textContent = "";
  recorder.start();
});

window.addEventListener("keyup", async (event) => {
  if (event.key.toLowerCase() !== "z" || !recorder || !isRecording) return;
  isRecording = false;
  panelEl.classList.remove("recording");
  statusEl.textContent = "running inference";
  const wav = recorder.stop();
  try {
    const results = await inferWavWindows(wav, { windowMs: INFERENCE_WINDOW_MS, stepMs: INFERENCE_STEP_MS });
    const result = strongestPrediction(results);
    readoutEl.textContent = result.label;
    probabilityEl.textContent = `${Math.round(result.probability * 100)}% (${result.startMs}-${result.endMs}ms)`;
    statusEl.textContent = `ready (${results.length} windows)`;
    saveDiagnostic(wav, labelSelect.value, result.label);
  } catch (error) {
    readoutEl.textContent = "error";
    probabilityEl.textContent = error.message;
    statusEl.textContent = "ready";
  }
});

function strongestPrediction(results) {
  if (results.length === 0) {
    throw new Error("no inference windows produced");
  }
  return results.reduce((best, result) => (result.probability > best.probability ? result : best));
}

async function saveDiagnostic(wav, truthLabel, prediction) {
  const form = new FormData();
  form.append("truth_label", truthLabel);
  form.append("prediction", prediction);
  form.append("audio", wav, "sample.wav");
  try {
    const response = await fetch("/api/diagnostics/sample", { method: "POST", body: form });
    if (!response.ok) {
      console.log("wakewords diagnostics failed", await response.text());
    }
  } catch (error) {
    console.log("wakewords diagnostics failed", error);
  }
}
