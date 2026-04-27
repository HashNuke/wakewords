from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from wakewords.export import ExportBundle
from wakewords.server import (
    OnnxWakewordModel,
    ServeConfig,
    _existing_export_bundle,
    _labels_metadata,
    _next_diagnostic_path,
    _probabilities,
    _read_wav_as_float32_mono,
    serve_playground,
)


class ServerTests(unittest.TestCase):
    def test_serve_exports_latest_model_before_starting_uvicorn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            bundle = ExportBundle(
                output_dir=project_dir / "models",
                model_path=project_dir / "models" / "model.onnx",
                checkpoint_dir=None,
                checkpoint_path=None,
                train_config_path=None,
                labels_path=project_dir / "models" / "labels.json",
                config_path=project_dir / "models" / "export_config.json",
            )

            uvicorn_run = mock.Mock()
            with (
                mock.patch.dict(sys.modules, {"uvicorn": SimpleNamespace(run=uvicorn_run)}),
                mock.patch("wakewords.server.export_model", return_value=bundle) as export_model,
                mock.patch("wakewords.server.create_app", return_value=object()) as create_app,
            ):
                serve_playground(
                    project_dir=project_dir,
                    runs_dir=Path("runs"),
                    output_dir=Path("models"),
                    host="127.0.0.1",
                    port=9000,
                    open_browser=False,
                )

            self.assertEqual(export_model.call_args.kwargs["project_dir"], project_dir.resolve())
            self.assertEqual(export_model.call_args.kwargs["runs_dir"], Path("runs"))
            self.assertEqual(export_model.call_args.kwargs["output_dir"], Path("models"))
            self.assertTrue(export_model.call_args.kwargs["overwrite"])
            self.assertIsInstance(create_app.call_args.kwargs["config"], ServeConfig)
            uvicorn_run.assert_called_once()

    def test_serve_reuses_existing_model_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            output_dir = project_dir / "models"
            output_dir.mkdir()
            (output_dir / "model.onnx").write_bytes(b"onnx")
            (output_dir / "labels.json").write_text(json.dumps(["yes"]) + "\n", encoding="utf-8")

            uvicorn_run = mock.Mock()
            with (
                mock.patch.dict(sys.modules, {"uvicorn": SimpleNamespace(run=uvicorn_run)}),
                mock.patch("wakewords.server.export_model") as export_model,
                mock.patch("wakewords.server.create_app", return_value=object()) as create_app,
            ):
                serve_playground(
                    project_dir=project_dir,
                    runs_dir=Path("runs"),
                    output_dir=Path("models"),
                    host="127.0.0.1",
                    port=9000,
                    open_browser=False,
                )

            export_model.assert_not_called()
            bundle = create_app.call_args.kwargs["bundle"]
            self.assertEqual(bundle.model_path, output_dir / "model.onnx")
            self.assertEqual(bundle.labels_path, output_dir / "labels.json")
            uvicorn_run.assert_called_once()

    def test_existing_export_bundle_returns_none_without_model_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertIsNone(_existing_export_bundle(project_dir=Path(tmp_dir), output_dir=Path("models")))

    def test_diagnostic_path_avoids_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics_dir = Path(tmp_dir)
            (diagnostics_dir / "yes-no.wav").write_bytes(b"existing")

            path = _next_diagnostic_path(diagnostics_dir, truth_label="yes", prediction="no")

            self.assertEqual(path, diagnostics_dir / "yes-no-0001.wav")

    def test_read_wav_as_float32_mono_resamples_to_16khz(self) -> None:
        samples = _wav_bytes(sample_rate=8000, values=[0, 1000, -1000, 0])

        decoded = _read_wav_as_float32_mono(samples, target_sample_rate=16000)

        self.assertGreaterEqual(len(decoded), 7)
        self.assertLessEqual(len(decoded), 9)

    def test_onnx_predict_maps_probabilities_to_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.onnx"
            labels_path = Path(tmp_dir) / "labels.json"
            model_path.write_bytes(b"onnx")
            labels_path.write_text(json.dumps(["yes", "no"]) + "\n", encoding="utf-8")

            fake_ort = SimpleNamespace(InferenceSession=mock.Mock(return_value=_FakeSession()))
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                model = OnnxWakewordModel(model_path=model_path, labels_path=labels_path)

            result = model.predict(_wav_bytes(sample_rate=16000, values=[0] * 1600))

            self.assertEqual(result["label"], "no")
            self.assertGreater(result["probability"], 0.5)
            self.assertEqual(set(result["probabilities"]), {"yes", "no"})

    def test_probabilities_preserves_normalized_output(self) -> None:
        probabilities = _probabilities([[0.8, 0.2]])

        self.assertAlmostEqual(probabilities[0], 0.8)
        self.assertAlmostEqual(probabilities[1], 0.2)

    def test_labels_metadata_groups_custom_and_google_speech_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps(
                    {
                        "custom_words": [{"tts_input": "Hey Astra", "label": "astra"}],
                        "google_speech_commands": ["yes", "no"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            metadata = _labels_metadata(project_dir=project_dir, model_labels=["astra", "yes", "unknown"])

            self.assertEqual(metadata["custom"], ["astra"])
            self.assertEqual(metadata["google_speech_commands"], ["yes"])
            self.assertEqual(metadata["other"], ["unknown"])


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def get_inputs(self) -> list[_Input]:
        return [_Input("audio_signal"), _Input("length")]

    def run(self, _: object, feeds: dict[str, object]) -> list[list[list[float]]]:
        self.feeds = feeds
        return [[[0.1, 2.0]]]


def _wav_bytes(*, sample_rate: int, values: list[int]) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
