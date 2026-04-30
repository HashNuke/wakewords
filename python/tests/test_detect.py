from __future__ import annotations

import io
import json
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakewords import cli
from wakewords.detect import detect_wakeword, detect_wakeword_windows


class DetectTests(unittest.TestCase):
    def test_detect_wakeword_uses_packaged_assets_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_bytes(_wav_bytes())

            with mock.patch("wakewords.detect.OnnxWakewordModel", _FakeWakewordModel) as model:
                result = detect_wakeword(audio_path, top_k=2)

            self.assertEqual(result.label, "tokyo")
            self.assertAlmostEqual(result.probability, 0.9)
            self.assertEqual(
                result.top_probabilities,
                [
                    {"label": "tokyo", "probability": 0.9},
                    {"label": "boston", "probability": 0.08},
                ],
            )
            self.assertEqual(model.model_path.name, "model.onnx")
            self.assertEqual(model.labels_path.name, "labels.json")

    def test_detect_cli_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_bytes(_wav_bytes())

            output = io.StringIO()
            with mock.patch("wakewords.cli.detect_wakeword", return_value=_FakeDetectionResult(audio_path)):
                with redirect_stdout(output):
                    cli.DataTools().detect(str(audio_path))

            self.assertEqual(json.loads(output.getvalue()), {"audio_path": str(audio_path), "label": "boston"})

    def test_detect_wakeword_windows_runs_over_sliding_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_bytes(_wav_bytes(duration_ms=3000))

            with mock.patch("wakewords.detect.OnnxWakewordModel", _FakeWakewordModel):
                results = detect_wakeword_windows(audio_path, window_ms=1000, step_ms=200, max_duration_ms=1400)

            self.assertEqual([(result.start_ms, result.end_ms) for result in results], [(0, 1000), (200, 1200), (400, 1400), (600, 1400), (800, 1400), (1000, 1400), (1200, 1400)])
            self.assertTrue(all(result.label == "tokyo" for result in results))

    def test_detect_cli_prints_one_json_line_per_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_bytes(_wav_bytes(duration_ms=3000))
            results = [_FakeDetectionResult(audio_path, start_ms=0, end_ms=1000), _FakeDetectionResult(audio_path, start_ms=200, end_ms=1200)]

            output = io.StringIO()
            with mock.patch("wakewords.cli.detect_wakeword_windows", return_value=results):
                with redirect_stdout(output):
                    cli.DataTools().detect(str(audio_path), window_ms=1000, step_ms=200, max_duration_ms=1200)

            lines = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([line["start_ms"] for line in lines], [0, 200])


class _FakeWakewordModel:
    model_path: Path
    labels_path: Path | None

    def __init__(self, *, model_path: Path, labels_path: Path | None) -> None:
        type(self).model_path = model_path
        type(self).labels_path = labels_path

    def predict(self, audio_bytes: bytes) -> dict[str, object]:
        return {
            "label": "tokyo",
            "probability": 0.9,
            "probabilities": {"boston": 0.08, "tokyo": 0.9, "unknown": 0.02},
        }


class _FakeDetectionResult:
    def __init__(self, audio_path: Path, start_ms: int | None = None, end_ms: int | None = None) -> None:
        self.audio_path = audio_path
        self.start_ms = start_ms
        self.end_ms = end_ms

    def to_json(self) -> str:
        result = {"audio_path": str(self.audio_path), "label": "boston"}
        if self.start_ms is not None and self.end_ms is not None:
            result["start_ms"] = self.start_ms
            result["end_ms"] = self.end_ms
        return json.dumps(result)


def _wav_bytes(*, duration_ms: int = 100) -> bytes:
    sample_rate = 16_000
    frames = sample_rate * duration_ms // 1000
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frames)
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
