from __future__ import annotations

import io
import json
import os
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
    _audio_input,
    create_app,
    _existing_export_bundle,
    _labels_metadata,
    _latest_test_report,
    _next_diagnostic_path,
    _probabilities,
    _read_wav_as_float32_mono,
    _shape_audio_signal,
    serve_playground,
)


class ServerTests(unittest.TestCase):
    def test_serve_exports_missing_bundle_before_starting_uvicorn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)

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

            self.assertEqual(export_model.call_args.kwargs["project_dir"], project_dir.resolve())
            self.assertEqual(export_model.call_args.kwargs["runs_dir"], Path("runs"))
            self.assertEqual(export_model.call_args.kwargs["output_dir"], Path("models"))
            self.assertTrue(export_model.call_args.kwargs["overwrite"])
            self.assertIsInstance(create_app.call_args.kwargs["config"], ServeConfig)
            self.assertNotIn("bundle", create_app.call_args.kwargs)
            uvicorn_run.assert_called_once()

    def test_serve_skips_startup_export_when_bundle_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            bundle = _create_export_bundle(project_dir, "models", ["yes"])
            source_model = _create_source_model(project_dir, "latest")
            os.utime(source_model, (100, 100))
            os.utime(bundle.model_path, (200, 200))

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
            self.assertNotIn("bundle", create_app.call_args.kwargs)
            uvicorn_run.assert_called_once()

    def test_existing_export_bundle_returns_none_without_model_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertIsNone(_existing_export_bundle(project_dir=Path(tmp_dir), output_dir=Path("models")))

    def test_model_and_labels_routes_read_existing_bundle_per_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            bundle = _create_export_bundle(project_dir, "models", ["old"])
            fake_app = _FakeFastAPI()
            fastapi = SimpleNamespace(
                FastAPI=mock.Mock(return_value=fake_app),
                File=mock.Mock(),
                Form=mock.Mock(),
                HTTPException=Exception,
                Query=mock.Mock(side_effect=lambda default, **_: default),
                UploadFile=object,
            )

            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "fastapi": fastapi,
                        "fastapi.responses": SimpleNamespace(FileResponse=_FakeFileResponse, HTMLResponse=object),
                        "fastapi.staticfiles": SimpleNamespace(StaticFiles=_FakeStaticFiles),
                    },
                ),
                mock.patch("wakewords.server.export_model") as export_model,
            ):
                app = create_app(
                    config=ServeConfig(
                        project_dir=project_dir,
                        runs_dir=Path("runs"),
                        output_dir=Path("models"),
                        host="127.0.0.1",
                        port=8000,
                        open_browser=False,
                    )
                )

                model_response = app.routes["/model.onnx"]()
                bundle.labels_path.write_text(json.dumps(["new"]) + "\n", encoding="utf-8")
                labels_response = app.routes["/api/labels"]()

            self.assertEqual(model_response.path, bundle.model_path)
            self.assertEqual(labels_response, ["new"])
            export_model.assert_not_called()

    def test_latest_test_report_uses_latest_available_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            old_run = project_dir / "runs" / "zz-latest-run-name"
            latest_report_run = project_dir / "runs" / "aa-latest-report"
            no_report_run = project_dir / "runs" / "newer-without-report"
            old_run.mkdir(parents=True)
            latest_report_run.mkdir(parents=True)
            no_report_run.mkdir(parents=True)
            _write_report(old_run, sample_count=1, mtime=100)
            _write_report(latest_report_run, sample_count=2, mtime=200)
            os.utime(no_report_run, (300, 300))

            report = _latest_test_report(project_dir, Path("runs"))

            self.assertIsNotNone(report)
            self.assertEqual(report.summary_path, latest_report_run / "test_report_summary.json")
            self.assertEqual(report.total, 2)

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

            session = _FakeSession()
            fake_ort = SimpleNamespace(InferenceSession=mock.Mock(return_value=session))
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                model = OnnxWakewordModel(model_path=model_path, labels_path=labels_path)

            result = model.predict(_wav_bytes(sample_rate=16000, values=[0] * 1600))

            self.assertEqual(result["label"], "no")
            self.assertGreater(result["probability"], 0.5)
            self.assertEqual(set(result["probabilities"]), {"yes", "no"})
            self.assertEqual(session.audio_shape, (1, 1600))

    def test_probabilities_preserves_normalized_output(self) -> None:
        probabilities = _probabilities([[0.8, 0.2]])

        self.assertAlmostEqual(probabilities[0], 0.8)
        self.assertAlmostEqual(probabilities[1], 0.2)

    def test_shape_audio_signal_uses_model_input_rank(self) -> None:
        import numpy as np

        signal = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

        self.assertEqual(_shape_audio_signal(signal, shape=[3]).shape, (3,))
        self.assertEqual(_shape_audio_signal(signal, shape=[1, 3]).shape, (1, 3))
        self.assertEqual(_shape_audio_signal(signal, shape=[1, 1, 3]).shape, (1, 1, 3))

    def test_audio_input_uses_nemo_mfcc_for_feature_model(self) -> None:
        import numpy as np

        features = np.zeros((64, 101), dtype=np.float32)
        with mock.patch("wakewords.server._nemo_mfcc_features", return_value=features) as mfcc:
            audio_input = _audio_input(np.zeros(16000, dtype=np.float32), shape=[1, 64, "time"])

        mfcc.assert_called_once()
        self.assertEqual(audio_input.shape, (64, 101))

    def test_labels_metadata_groups_model_labels_by_project_config(self) -> None:
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

    def test_labels_metadata_omits_configured_custom_words_missing_from_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": [{"tts_input": "New Word", "label": "new_word"}]}) + "\n",
                encoding="utf-8",
            )

            metadata = _labels_metadata(project_dir=project_dir, model_labels=["old_word"])

            self.assertEqual(metadata["custom"], [])
            self.assertEqual(metadata["other"], ["old_word"])


class _Input:
    def __init__(self, name: str, shape: list[int | str] | None = None) -> None:
        self.name = name
        self.shape = shape or []


class _FakeSession:
    def get_inputs(self) -> list[_Input]:
        return [_Input("audio_signal", [1, "time"]), _Input("length", [1])]

    def run(self, _: object, feeds: dict[str, object]) -> list[list[list[float]]]:
        self.feeds = feeds
        self.audio_shape = feeds["audio_signal"].shape
        return [[[0.1, 2.0]]]


class _FakeFastAPI:
    def __init__(self) -> None:
        self.routes: dict[str, object] = {}

    def middleware(self, _: str) -> object:
        def decorator(func: object) -> object:
            return func

        return decorator

    def mount(self, *_: object, **__: object) -> None:
        return None

    def get(self, path: str, **_: object) -> object:
        def decorator(func: object) -> object:
            self.routes[path] = func
            return func

        return decorator

    def post(self, path: str, **_: object) -> object:
        def decorator(func: object) -> object:
            self.routes[path] = func
            return func

        return decorator


class _FakeFileResponse:
    def __init__(self, path: Path, **_: object) -> None:
        self.path = path


class _FakeStaticFiles:
    def __init__(self, **_: object) -> None:
        return None


def _wav_bytes(*, sample_rate: int, values: list[int]) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))
        return buffer.getvalue()


def _create_export_bundle(project_dir: Path, name: str, labels: list[str]) -> ExportBundle:
    output_dir = project_dir / name
    output_dir.mkdir()
    model_path = output_dir / "model.onnx"
    labels_path = output_dir / "labels.json"
    config_path = output_dir / "export_config.json"
    model_path.write_bytes(name.encode())
    labels_path.write_text(json.dumps(labels) + "\n", encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")
    return ExportBundle(
        output_dir=output_dir,
        model_path=model_path,
        checkpoint_dir=None,
        checkpoint_path=None,
        train_config_path=None,
        labels_path=labels_path,
        config_path=config_path,
    )


def _create_source_model(project_dir: Path, run_name: str) -> Path:
    run_dir = project_dir / "runs" / run_name
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    (run_dir / "train_config.json").write_text("{}\n", encoding="utf-8")
    model_path = models_dir / "model.nemo"
    model_path.write_bytes(b"nemo")
    return model_path


def _write_report(run_dir: Path, *, sample_count: int, mtime: int) -> None:
    summary_path = run_dir / "test_report_summary.json"
    rows_path = run_dir / "test_report.jsonl"
    summary_path.write_text(json.dumps({"sample_count": sample_count}) + "\n", encoding="utf-8")
    rows_path.write_text("".join(json.dumps({"index": index}) + "\n" for index in range(sample_count)), encoding="utf-8")
    os.utime(summary_path, (mtime, mtime))
    os.utime(rows_path, (mtime, mtime))


if __name__ == "__main__":
    unittest.main()
