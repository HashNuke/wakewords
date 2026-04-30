from __future__ import annotations

import io
import json
import math
import re
import threading
import wave
import webbrowser
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from wakewords.export import ExportBundle, export_model

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ServeConfig:
    project_dir: Path
    runs_dir: Path
    output_dir: Path
    host: str
    port: int
    open_browser: bool


def serve_playground(
    *,
    project_dir: Path,
    runs_dir: Path,
    output_dir: Path,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Export the latest model and serve the local playground."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI standard dependencies are not installed. Install wakewords with serve dependencies.") from exc

    config = ServeConfig(
        project_dir=project_dir.resolve(),
        runs_dir=runs_dir,
        output_dir=output_dir,
        host=host,
        port=port,
        open_browser=open_browser,
    )
    bundle = export_model(
        project_dir=config.project_dir,
        runs_dir=config.runs_dir,
        run_dir=None,
        model_path=None,
        checkpoint_path=None,
        output_dir=config.output_dir,
        format="onnx",
        overwrite=True,
    )
    app = create_app(config=config, bundle=bundle)
    if open_browser:
        threading.Timer(0.75, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port)


def _existing_export_bundle(*, project_dir: Path, output_dir: Path) -> ExportBundle | None:
    resolved_output_dir = output_dir if output_dir.is_absolute() else project_dir / output_dir
    model_path = resolved_output_dir / "model.onnx"
    if not model_path.is_file():
        return None

    checkpoint_dir = resolved_output_dir / "last_checkpoint"
    checkpoint_path = checkpoint_dir / "last.ckpt"
    train_config_path = checkpoint_dir / "train_config.json"
    labels_path = resolved_output_dir / "labels.json"
    config_path = resolved_output_dir / "export_config.json"
    return ExportBundle(
        output_dir=resolved_output_dir,
        model_path=model_path,
        checkpoint_dir=checkpoint_dir if checkpoint_dir.is_dir() else None,
        checkpoint_path=checkpoint_path if checkpoint_path.is_file() else None,
        train_config_path=train_config_path if train_config_path.is_file() else None,
        labels_path=labels_path if labels_path.is_file() else None,
        config_path=config_path,
    )


def create_app(*, config: ServeConfig, bundle: ExportBundle) -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install wakewords with serve dependencies.") from exc
    globals()["UploadFile"] = UploadFile

    app = FastAPI(title="wakewords playground")
    model_labels = _load_labels(bundle.labels_path)
    inference_cache: dict[str, OnnxWakewordModel] = {}
    static_dir = resources.files("wakewords.playground")

    @app.middleware("http")
    async def no_cache_playground_assets(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    with resources.as_file(static_dir) as playground_dir:
        app.mount("/static", StaticFiles(directory=str(playground_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def test_page() -> str:
        return _read_asset("test.html")

    @app.get("/record", response_class=HTMLResponse)
    def record_page() -> str:
        return _read_asset("record.html")

    @app.get("/test-report", response_class=HTMLResponse)
    def test_report_page() -> str:
        return _test_report_html()

    @app.get("/model.onnx")
    def model() -> FileResponse:
        return FileResponse(bundle.model_path)

    @app.get("/labels.json")
    @app.get("/api/labels")
    def labels() -> list[str]:
        return model_labels

    @app.get("/api/labels/metadata")
    def labels_metadata() -> dict[str, list[str]]:
        return _labels_metadata(project_dir=config.project_dir, model_labels=model_labels)

    @app.post("/api/infer")
    async def infer(audio: UploadFile = File(...)) -> dict[str, Any]:
        try:
            audio_bytes = await audio.read()
            inference = inference_cache.get("model")
            if inference is None:
                inference = OnnxWakewordModel(model_path=bundle.model_path, labels_path=bundle.labels_path)
                inference_cache["model"] = inference
            return inference.predict(audio_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/diagnostics/sample")
    async def diagnostics_sample(
        truth_label: str = Form(...),
        prediction: str = Form(...),
        audio: UploadFile = File(...),
    ) -> dict[str, str]:
        audio_bytes = await audio.read()
        diagnostics_dir = config.project_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = _next_diagnostic_path(
            diagnostics_dir,
            truth_label=_safe_filename_part(truth_label),
            prediction=_safe_filename_part(prediction),
        )
        path.write_bytes(audio_bytes)
        return {"path": str(path)}

    @app.get("/api/test-report")
    def test_report(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        report = _latest_test_report(config.project_dir, config.runs_dir)
        if report is None:
            raise HTTPException(status_code=404, detail="No test report found. Run `wakewords test` first.")
        rows = _read_report_rows(report.rows_path, offset=offset, limit=limit)
        return {
            "summary": report.summary,
            "offset": offset,
            "limit": limit,
            "total": report.total,
            "rows": [_report_row_response(row) for row in rows],
        }

    @app.get("/api/test-report/audio/{index}")
    def test_report_audio(index: int) -> FileResponse:
        report = _latest_test_report(config.project_dir, config.runs_dir)
        if report is None:
            raise HTTPException(status_code=404, detail="No test report found. Run `wakewords test` first.")
        row = _read_report_row(report.rows_path, index=index)
        if row is None:
            raise HTTPException(status_code=404, detail="Report row not found")
        audio_filepath = row.get("audio_filepath")
        if not isinstance(audio_filepath, str):
            raise HTTPException(status_code=404, detail="Report row has no audio file")
        audio_path = Path(audio_filepath)
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(audio_path, media_type="audio/wav")

    return app


@dataclass(frozen=True)
class _TestReport:
    summary_path: Path
    rows_path: Path
    summary: dict[str, Any]
    total: int


class OnnxWakewordModel:
    def __init__(self, *, model_path: Path, labels_path: Path | None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed. Install wakewords with serve dependencies.") from exc

        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_shapes = {input_meta.name: list(input_meta.shape or []) for input_meta in self._session.get_inputs()}
        self.labels = _load_labels(labels_path)

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        samples = _read_wav_as_float32_mono(audio_bytes, target_sample_rate=SAMPLE_RATE)
        feeds = self._feeds(samples)
        outputs = self._session.run(None, feeds)
        probabilities = _probabilities(outputs[0])
        if len(self.labels) != len(probabilities):
            labels = [f"class_{index}" for index in range(len(probabilities))]
        else:
            labels = self.labels
        indexed = list(zip(labels, probabilities, strict=False))
        label, probability = max(indexed, key=lambda item: item[1])
        return {
            "label": label,
            "probability": probability,
            "probabilities": {item_label: item_probability for item_label, item_probability in indexed},
        }

    def _feeds(self, samples: Any) -> dict[str, Any]:
        import numpy as np

        waveform = np.asarray(samples, dtype=np.float32)
        feeds: dict[str, Any] = {}
        for name, shape in self._input_shapes.items():
            normalized = name.lower()
            if "length" in normalized:
                feeds[name] = np.asarray([_feature_length(waveform, self._input_shapes)], dtype=np.int64)
            elif "audio" in normalized or "signal" in normalized or len(self._input_shapes) == 1:
                feeds[name] = _shape_audio_signal(_audio_input(waveform, shape=shape), shape=shape)
            else:
                raise ValueError(
                    f"Unsupported ONNX input {name!r}. Expected audio/signal and optional length inputs."
                )
        return feeds


def _audio_input(waveform: Any, *, shape: list[Any]) -> Any:
    if _expects_mfcc_features(shape):
        return _nemo_mfcc_features(waveform)
    return waveform


def _expects_mfcc_features(shape: list[Any]) -> bool:
    return len(shape) == 3 and shape[1] == 64


def _feature_length(waveform: Any, input_shapes: dict[str, list[Any]]) -> int:
    if any(_expects_mfcc_features(shape) for shape in input_shapes.values()):
        return _nemo_mfcc_features(waveform).shape[-1]
    return int(waveform.shape[0])


def _shape_audio_signal(signal: Any, *, shape: list[Any]) -> Any:
    rank = len(shape)
    if rank == 0:
        rank = 2
    if rank == 1:
        return signal
    if rank == 2:
        return signal[None, :]
    if rank == 3:
        if len(signal.shape) == 2:
            return signal[None, :, :]
        return signal[None, None, :]
    raise ValueError(f"Unsupported ONNX audio input rank {rank}. Expected rank 1, 2, or 3.")


def _nemo_mfcc_features(waveform: Any) -> Any:
    import numpy as np
    import torch
    from nemo.collections.asr.modules.audio_preprocessing import AudioToMFCCPreprocessor

    preprocessor = _nemo_mfcc_preprocessor()
    signal = torch.as_tensor(np.asarray(waveform, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
    length = torch.as_tensor([signal.shape[1]], dtype=torch.long)
    with torch.no_grad():
        features, _ = preprocessor(input_signal=signal, length=length)
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def _nemo_mfcc_preprocessor() -> Any:
    if not hasattr(_nemo_mfcc_preprocessor, "_instance"):
        from nemo.collections.asr.modules.audio_preprocessing import AudioToMFCCPreprocessor

        preprocessor = AudioToMFCCPreprocessor(
            sample_rate=SAMPLE_RATE,
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            n_fft=512,
            n_mels=64,
            n_mfcc=64,
            log=True,
        )
        preprocessor.eval()
        _nemo_mfcc_preprocessor._instance = preprocessor
    return _nemo_mfcc_preprocessor._instance


def _read_asset(name: str) -> str:
    return resources.files("wakewords.playground").joinpath(name).read_text(encoding="utf-8")


def _test_report_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>wakewords test report</title>
    <link rel="stylesheet" href="/static/styles.css" />
  </head>
  <body>
    <main class="shell">
      <header class="topbar">
        <a class="brand" href="/">wakewords</a>
        <nav class="nav" aria-label="playground">
          <a href="/">playground</a>
          <a href="/record">record</a>
          <a href="/test-report" aria-current="page">test report</a>
        </nav>
      </header>
      <section class="panel report-panel">
        <h1>Test report</h1>
        <div id="reportSummary" class="hint">Loading...</div>
        <div class="controls">
          <button id="prevPage" type="button">Prev</button>
          <button id="nextPage" type="button">Next</button>
          <div id="pageStatus" class="status"></div>
        </div>
        <div class="report-table-wrap">
          <table class="report-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Actual</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Audio</th>
              </tr>
            </thead>
            <tbody id="reportRows"></tbody>
          </table>
        </div>
      </section>
    </main>
    <script>
      const limit = 100;
      let offset = 0;
      let total = 0;
      const summaryEl = document.getElementById('reportSummary');
      const rowsEl = document.getElementById('reportRows');
      const pageStatus = document.getElementById('pageStatus');
      const prevPage = document.getElementById('prevPage');
      const nextPage = document.getElementById('nextPage');

      function pct(value) {
        return `${(Number(value || 0) * 100).toFixed(2)}%`;
      }

      function metric(summary, key) {
        const first = (summary.metrics || [])[0] || {};
        return first[key];
      }

      async function loadPage(nextOffset) {
        const response = await fetch(`/api/test-report?offset=${nextOffset}&limit=${limit}`);
        if (!response.ok) {
          summaryEl.textContent = await response.text();
          return;
        }
        const report = await response.json();
        offset = report.offset;
        total = report.total;
        const summary = report.summary || {};
        summaryEl.textContent = `samples ${summary.sample_count || total}, correct ${summary.correct_count || 0}, accuracy ${pct(metric(summary, 'test_acc_micro_top_1'))}, macro ${pct(metric(summary, 'test_acc_macro'))}, loss ${Number(metric(summary, 'test_loss') || 0).toFixed(4)}`;
        rowsEl.replaceChildren(...report.rows.map((row) => {
          const tr = document.createElement('tr');
          if (!row.correct) tr.className = 'incorrect';
          tr.innerHTML = `
            <td>${row.index}</td>
            <td>${row.actual_label || ''}</td>
            <td>${row.predicted_label || ''}</td>
            <td>${pct(row.probability)}</td>
            <td><audio controls preload="none" src="${row.audio_url}"></audio></td>
          `;
          return tr;
        }));
        pageStatus.textContent = `${offset + 1}-${Math.min(offset + limit, total)} of ${total}`;
        prevPage.disabled = offset <= 0;
        nextPage.disabled = offset + limit >= total;
      }

      prevPage.addEventListener('click', () => loadPage(Math.max(0, offset - limit)));
      nextPage.addEventListener('click', () => loadPage(offset + limit));
      loadPage(0);
    </script>
  </body>
</html>"""


def _latest_test_report(project_dir: Path, runs_dir: Path) -> _TestReport | None:
    resolved_runs_dir = runs_dir if runs_dir.is_absolute() else project_dir / runs_dir
    if not resolved_runs_dir.is_dir():
        return None
    candidates = sorted(
        resolved_runs_dir.glob("*/test_report_summary.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for summary_path in candidates:
        rows_path = summary_path.with_name("test_report.jsonl")
        if not rows_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        total = summary.get("sample_count")
        if not isinstance(total, int):
            total = _count_jsonl_rows(rows_path)
        return _TestReport(summary_path=summary_path, rows_path=rows_path, summary=summary, total=total)
    return None


def _read_report_rows(path: Path, *, offset: int, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index < offset:
                continue
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_report_row(path: Path, *, index: int) -> dict[str, Any] | None:
    if index < 0:
        return None
    with path.open(encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            if row_index == index and line.strip():
                return json.loads(line)
    return None


def _report_row_response(row: dict[str, Any]) -> dict[str, Any]:
    index = row.get("index")
    return {**row, "audio_url": f"/api/test-report/audio/{index}"}


def _count_jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _load_labels(labels_path: Path | None) -> list[str]:
    if labels_path is None or not labels_path.is_file():
        return []
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return []
    return labels


def _labels_metadata(*, project_dir: Path, model_labels: list[str]) -> dict[str, list[str]]:
    configured_custom, configured_google = _configured_label_groups(project_dir / "config.json")
    custom = [label for label in model_labels if label in set(configured_custom)]
    google = [label for label in model_labels if label in set(configured_google)]
    grouped = set(custom) | set(google)
    other = [label for label in model_labels if label not in grouped]
    return {
        "all": model_labels,
        "custom": custom,
        "google_speech_commands": google,
        "other": other,
    }


def _configured_label_groups(config_path: Path) -> tuple[list[str], list[str]]:
    if not config_path.is_file():
        return [], []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return _custom_word_labels(config.get("custom_words")), _string_list(config.get("google_speech_commands"))


def _custom_word_labels(custom_words: object) -> list[str]:
    labels: list[str] = []
    if not isinstance(custom_words, list):
        return labels
    for word in custom_words:
        if isinstance(word, str) and word:
            if word not in labels:
                labels.append(word)
            continue
        if isinstance(word, dict):
            label = word.get("label")
            if isinstance(label, str) and label and label not in labels:
                labels.append(label)
    return labels


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    labels: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in labels:
            labels.append(value)
    return labels


def _read_wav_as_float32_mono(audio_bytes: bytes, *, target_sample_rate: int) -> list[float]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            payload = wav_file.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError("Expected WAV audio upload.") from exc

    if channels < 1:
        raise ValueError("WAV audio must have at least one channel.")
    if sample_width != 2:
        raise ValueError("WAV audio must be 16-bit PCM.")

    import numpy as np

    data = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    data = data / 32768.0
    if sample_rate != target_sample_rate:
        data = _resample(data, source_rate=sample_rate, target_rate=target_sample_rate)
    if data.size == 0:
        raise ValueError("WAV audio contains no samples.")
    return data.astype(np.float32).tolist()


def _resample(samples: Any, *, source_rate: int, target_rate: int) -> Any:
    import numpy as np

    if source_rate <= 0:
        raise ValueError("WAV audio has an invalid sample rate.")
    if source_rate == target_rate:
        return samples
    duration = samples.shape[0] / source_rate
    target_count = max(1, round(duration * target_rate))
    source_positions = np.linspace(0, samples.shape[0] - 1, num=samples.shape[0], dtype=np.float32)
    target_positions = np.linspace(0, samples.shape[0] - 1, num=target_count, dtype=np.float32)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _probabilities(output: Any) -> list[float]:
    import numpy as np

    values = np.asarray(output, dtype=np.float32).reshape(-1)
    values_sum = float(np.sum(values))
    if values.size > 0 and np.all(values >= 0) and np.all(values <= 1) and 0.99 <= values_sum <= 1.01:
        return [float(value / values_sum) for value in values]

    logits = values
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    total = float(np.sum(exp))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Model returned invalid logits.")
    return [float(value / total) for value in exp]


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return safe or "unknown"


def _next_diagnostic_path(diagnostics_dir: Path, *, truth_label: str, prediction: str) -> Path:
    base = diagnostics_dir / f"{truth_label}-{prediction}.wav"
    if not base.exists():
        return base
    for index in range(1, 10_000):
        candidate = diagnostics_dir / f"{truth_label}-{prediction}-{index:04d}.wav"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Too many diagnostics samples for {truth_label}-{prediction}")
