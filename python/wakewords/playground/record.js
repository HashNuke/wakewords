import { createRecorder, enableMic, inferWav, loadLabelMetadata } from "./audio.js";

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
    const result = await inferWav(wav);
    readoutEl.textContent = result.label;
    probabilityEl.textContent = `${Math.round(result.probability * 100)}%`;
    statusEl.textContent = "ready";
    saveDiagnostic(wav, labelSelect.value, result.label);
  } catch (error) {
    readoutEl.textContent = "error";
    probabilityEl.textContent = error.message;
    statusEl.textContent = "ready";
  }
});

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
