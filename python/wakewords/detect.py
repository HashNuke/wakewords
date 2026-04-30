from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

from wakewords.server import OnnxWakewordModel


DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class DetectionResult:
    audio_path: Path
    label: str
    probability: float
    top_probabilities: list[dict[str, float | str]]
    start_ms: int | None = None
    end_ms: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "audio_path": str(self.audio_path),
            "label": self.label,
            "probability": self.probability,
            "top_probabilities": self.top_probabilities,
        }
        if self.start_ms is not None and self.end_ms is not None:
            result["start_ms"] = self.start_ms
            result["end_ms"] = self.end_ms
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)


def detect_wakeword(
    audio_path: Path,
    *,
    model_path: Path | None = None,
    labels_path: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> DetectionResult:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    with _model_paths(model_path=model_path, labels_path=labels_path) as paths:
        model = OnnxWakewordModel(model_path=paths.model_path, labels_path=paths.labels_path)
        return _prediction_result(audio_path=audio_path, audio_bytes=audio_path.read_bytes(), model=model, top_k=top_k)


def detect_wakeword_windows(
    audio_path: Path,
    *,
    window_ms: int,
    step_ms: int,
    max_duration_ms: int | None = None,
    model_path: Path | None = None,
    labels_path: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[DetectionResult]:
    if window_ms < 1:
        raise ValueError("window_ms must be >= 1")
    if step_ms < 1:
        raise ValueError("step_ms must be >= 1")
    if max_duration_ms is not None and max_duration_ms < 1:
        raise ValueError("max_duration_ms must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    wav = _read_wav(audio_path.read_bytes())
    limit_ms = min(wav.duration_ms, max_duration_ms) if max_duration_ms is not None else wav.duration_ms
    if limit_ms <= 0:
        return []

    results: list[DetectionResult] = []
    with _model_paths(model_path=model_path, labels_path=labels_path) as paths:
        model = OnnxWakewordModel(model_path=paths.model_path, labels_path=paths.labels_path)
        start_ms = 0
        while start_ms < limit_ms:
            end_ms = min(start_ms + window_ms, limit_ms)
            window_bytes = _slice_wav(wav, start_ms=start_ms, end_ms=end_ms)
            results.append(
                _prediction_result(
                    audio_path=audio_path,
                    audio_bytes=window_bytes,
                    model=model,
                    top_k=top_k,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )
            start_ms += step_ms
    return results


@dataclass(frozen=True)
class _ModelPaths:
    model_path: Path
    labels_path: Path | None


class _model_paths:
    def __init__(self, *, model_path: Path | None, labels_path: Path | None) -> None:
        self._model_path = model_path
        self._labels_path = labels_path
        self._model_context: Any | None = None
        self._labels_context: Any | None = None

    def __enter__(self) -> _ModelPaths:
        if self._model_path is not None:
            return _ModelPaths(model_path=self._model_path, labels_path=self._labels_path)
        model_resource = resources.files("wakewords.assets").joinpath("model.onnx")
        labels_resource = resources.files("wakewords.assets").joinpath("labels.json")
        self._model_context = resources.as_file(model_resource)
        self._labels_context = resources.as_file(labels_resource)
        model_path = self._model_context.__enter__()
        labels_path = self._labels_context.__enter__()
        return _ModelPaths(model_path=model_path, labels_path=self._labels_path or labels_path)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._labels_context is not None:
            self._labels_context.__exit__(exc_type, exc, traceback)
        if self._model_context is not None:
            self._model_context.__exit__(exc_type, exc, traceback)


def _prediction_result(
    *,
    audio_path: Path,
    audio_bytes: bytes,
    model: OnnxWakewordModel,
    top_k: int,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> DetectionResult:
    prediction = model.predict(audio_bytes)
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = {}
    top_probabilities = _top_probabilities(probabilities, top_k=top_k)
    return DetectionResult(
        audio_path=audio_path,
        label=_require_str(prediction, "label"),
        probability=_require_float(prediction, "probability"),
        top_probabilities=top_probabilities,
        start_ms=start_ms,
        end_ms=end_ms,
    )


@dataclass(frozen=True)
class _WavAudio:
    channels: int
    sample_width: int
    sample_rate: int
    frames: bytes
    duration_ms: int


def _read_wav(audio_bytes: bytes) -> _WavAudio:
    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    duration_ms = round(len(frames) / (sample_width * channels) / sample_rate * 1000)
    return _WavAudio(
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
        frames=frames,
        duration_ms=duration_ms,
    )


def _slice_wav(wav: _WavAudio, *, start_ms: int, end_ms: int) -> bytes:
    bytes_per_frame = wav.sample_width * wav.channels
    start_sample = round(wav.sample_rate * start_ms / 1000)
    end_sample = round(wav.sample_rate * end_ms / 1000)
    frames = wav.frames[start_sample * bytes_per_frame : end_sample * bytes_per_frame]
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(wav.channels)
            wav_file.setsampwidth(wav.sample_width)
            wav_file.setframerate(wav.sample_rate)
            wav_file.writeframes(frames)
        return buffer.getvalue()


def _top_probabilities(probabilities: dict[Any, Any], *, top_k: int) -> list[dict[str, float | str]]:
    items = [
        {"label": str(label), "probability": float(probability)}
        for label, probability in probabilities.items()
        if isinstance(probability, int | float)
    ]
    return sorted(items, key=lambda item: float(item["probability"]), reverse=True)[:top_k]


def _require_str(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Prediction result missing string {key!r}")
    return value


def _require_float(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Prediction result missing float {key!r}")
    return float(value)
