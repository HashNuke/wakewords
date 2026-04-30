from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakewords import cli
from wakewords.export import export_model
from wakewords.train import DEFAULT_MODEL_NAME


class ExportTests(unittest.TestCase):
    def test_export_model_writes_project_bundle_from_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            run_dir = _create_run(project_dir, "20260427-054037-commandrecognition")
            output_dir = project_dir / "models"

            with mock.patch("wakewords.export._export_onnx", side_effect=_fake_export_onnx):
                bundle = export_model(
                    project_dir=project_dir,
                    runs_dir=Path("runs"),
                    run_dir=None,
                    model_path=None,
                    checkpoint_path=None,
                    output_dir=Path("models"),
                    format="onnx",
                    overwrite=False,
                )

            self.assertEqual(bundle.output_dir, output_dir)
            self.assertEqual(bundle.model_path, output_dir / "model.onnx")
            self.assertEqual(bundle.checkpoint_dir, output_dir / "last_checkpoint")
            self.assertEqual(bundle.checkpoint_path, output_dir / "last_checkpoint" / "last.ckpt")
            self.assertEqual(bundle.train_config_path, output_dir / "last_checkpoint" / "train_config.json")
            self.assertEqual(bundle.labels_path, output_dir / "labels.json")
            self.assertEqual(bundle.config_path, output_dir / "export_config.json")
            self.assertEqual((output_dir / "model.onnx").read_text(encoding="utf-8"), "onnx")
            self.assertEqual((output_dir / "last_checkpoint" / "last.ckpt").read_text(encoding="utf-8"), "checkpoint")
            train_config = json.loads((output_dir / "last_checkpoint" / "train_config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                train_config,
                {
                    "model_name": DEFAULT_MODEL_NAME,
                    "labels": ["yes", "unknown"],
                    "training": {
                        "max_epochs": 3,
                        "batch_size": 16,
                        "num_workers": 2,
                        "accelerator": "cpu",
                        "devices": 1,
                        "learning_rate": 0.001,
                        "tensorboard": False,
                    },
                },
            )
            self.assertEqual(json.loads((output_dir / "labels.json").read_text(encoding="utf-8")), ["unknown", "yes"])

            export_config = json.loads((output_dir / "export_config.json").read_text(encoding="utf-8"))
            self.assertEqual(export_config["format"], "onnx")
            self.assertEqual(export_config["model_path"], "model.onnx")
            self.assertEqual(export_config["checkpoint_dir"], "last_checkpoint")
            self.assertEqual(export_config["checkpoint_path"], "last_checkpoint/last.ckpt")
            self.assertEqual(export_config["train_config_path"], "last_checkpoint/train_config.json")
            self.assertEqual(export_config["labels_path"], "labels.json")
            self.assertNotIn("source", export_config)
            self.assertNotIn(str(project_dir), json.dumps(export_config))
            self.assertNotIn(str(project_dir), json.dumps(train_config))

    def test_export_model_uses_latest_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            old_run = _create_run(project_dir, "zz-custom-name")
            latest_run = _create_run(project_dir, "aa-custom-name")
            old_model = next((old_run / "models").glob("*.nemo"))
            latest_model = next((latest_run / "models").glob("*.nemo"))
            os.utime(old_model, (100, 100))
            os.utime(latest_model, (200, 200))
            os.utime(old_run, (300, 300))
            os.utime(latest_run, (50, 50))

            with mock.patch("wakewords.export._export_onnx", side_effect=_fake_export_onnx) as export_onnx:
                export_model(
                    project_dir=project_dir,
                    runs_dir=Path("runs"),
                    run_dir=None,
                    model_path=None,
                    checkpoint_path=None,
                    output_dir=Path("models"),
                    format="onnx",
                    overwrite=False,
                )

            self.assertEqual(export_onnx.call_args.args[0], latest_model)

    def test_export_model_refuses_to_overwrite_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            _create_run(project_dir, "existing")
            output_dir = project_dir / "models"
            output_dir.mkdir()
            (output_dir / "model.onnx").write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                export_model(
                    project_dir=project_dir,
                    runs_dir=Path("runs"),
                    run_dir=None,
                    model_path=None,
                    checkpoint_path=None,
                    output_dir=Path("models"),
                    format="onnx",
                    overwrite=False,
                )

    def test_cli_export_prints_bundle_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            _create_run(project_dir, "cli-run")
            stdout = io.StringIO()
            argv = [
                "wakewords",
                "export",
                "--project-dir",
                tmp_dir,
                "--format",
                "onnx",
            ]

            with mock.patch.object(cli.sys, "argv", argv):
                with mock.patch("wakewords.export._export_onnx", side_effect=_fake_export_onnx):
                    with redirect_stdout(stdout):
                        cli.main()

            output_dir = project_dir / "models"
            self.assertEqual(
                stdout.getvalue().strip().splitlines(),
                [
                    str(output_dir),
                    str(output_dir / "model.onnx"),
                    str(output_dir / "last_checkpoint"),
                    str(output_dir / "last_checkpoint" / "last.ckpt"),
                    str(output_dir / "last_checkpoint" / "train_config.json"),
                    str(output_dir / "labels.json"),
                    str(output_dir / "export_config.json"),
                ],
            )

    def test_cli_export_overwrites_existing_bundle_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            _create_run(project_dir, "cli-run")
            output_dir = project_dir / "models"
            checkpoint_dir = output_dir / "last_checkpoint"
            checkpoint_dir.mkdir(parents=True)
            (output_dir / "model.onnx").write_text("old onnx", encoding="utf-8")
            (checkpoint_dir / "last.ckpt").write_text("old checkpoint", encoding="utf-8")
            (checkpoint_dir / "train_config.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "labels.json").write_text("[]\n", encoding="utf-8")
            (output_dir / "export_config.json").write_text("{}\n", encoding="utf-8")
            argv = [
                "wakewords",
                "export",
                "--project-dir",
                tmp_dir,
            ]

            with mock.patch.object(cli.sys, "argv", argv):
                with mock.patch("wakewords.export._export_onnx", side_effect=_fake_export_onnx):
                    with redirect_stdout(io.StringIO()):
                        cli.main()

            self.assertEqual((output_dir / "model.onnx").read_text(encoding="utf-8"), "onnx")
            self.assertEqual((checkpoint_dir / "last.ckpt").read_text(encoding="utf-8"), "checkpoint")
            self.assertEqual(
                json.loads((checkpoint_dir / "train_config.json").read_text(encoding="utf-8")),
                {
                    "model_name": DEFAULT_MODEL_NAME,
                    "labels": ["yes", "unknown"],
                    "training": {
                        "max_epochs": 3,
                        "batch_size": 16,
                        "num_workers": 2,
                        "accelerator": "cpu",
                        "devices": 1,
                        "learning_rate": 0.001,
                        "tensorboard": False,
                    },
                },
            )

    def test_export_model_sanitizes_checkpoint_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            run_dir = _create_run(project_dir, "leaky-run")
            (run_dir / "checkpoints" / "last.ckpt").write_text(f"checkpoint from {project_dir}\n", encoding="utf-8")

            with mock.patch("wakewords.export._export_onnx", side_effect=_fake_export_onnx):
                export_model(
                    project_dir=project_dir,
                    runs_dir=Path("runs"),
                    run_dir=None,
                    model_path=None,
                    checkpoint_path=None,
                    output_dir=Path("models"),
                    format="onnx",
                    overwrite=False,
                )

            checkpoint = (project_dir / "models" / "last_checkpoint" / "last.ckpt").read_text(encoding="utf-8")
            self.assertNotIn(str(project_dir), checkpoint)

    def test_cli_serve_starts_playground(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "wakewords",
                "serve",
                "--project-dir",
                tmp_dir,
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--open-browser",
                "False",
            ]

            with mock.patch.object(cli.sys, "argv", argv):
                with mock.patch("wakewords.server.serve_playground") as serve_playground:
                    cli.main()

            self.assertEqual(serve_playground.call_args.kwargs["project_dir"], Path(tmp_dir))
            self.assertEqual(serve_playground.call_args.kwargs["runs_dir"], Path("runs"))
            self.assertEqual(serve_playground.call_args.kwargs["output_dir"], Path("models"))
            self.assertEqual(serve_playground.call_args.kwargs["host"], "0.0.0.0")
            self.assertEqual(serve_playground.call_args.kwargs["port"], 9000)
            self.assertFalse(serve_playground.call_args.kwargs["open_browser"])

    def test_cli_serve_defaults_to_port_4491(self) -> None:
        argv = ["wakewords", "serve"]

        with mock.patch.object(cli.sys, "argv", argv):
            with mock.patch("wakewords.server.serve_playground") as serve_playground:
                cli.main()

        self.assertEqual(serve_playground.call_args.kwargs["port"], 4491)


def _create_run(project_dir: Path, run_name: str) -> Path:
    run_dir = project_dir / "runs" / run_name
    models_dir = run_dir / "models"
    checkpoints_dir = run_dir / "checkpoints"
    models_dir.mkdir(parents=True)
    checkpoints_dir.mkdir()
    (models_dir / f"{DEFAULT_MODEL_NAME}.nemo").write_text("nemo", encoding="utf-8")
    (checkpoints_dir / "last.ckpt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model_name": DEFAULT_MODEL_NAME,
                "base_model_path": str(project_dir / "local" / "base.nemo"),
                "from_checkpoint": str(checkpoints_dir / "last.ckpt"),
                "resume_source": {"checkpoint_path": str(project_dir / "other" / "last.ckpt")},
                "labels": ["yes", "unknown"],
                "manifests": {
                    "train": str(project_dir / "data" / "manifests" / "train.jsonl"),
                    "validation": str(project_dir / "data" / "manifests" / "validation.jsonl"),
                    "test": str(project_dir / "data" / "manifests" / "test.jsonl"),
                },
                "training": {
                    "max_epochs": 3,
                    "batch_size": 16,
                    "num_workers": 2,
                    "accelerator": "cpu",
                    "devices": 1,
                    "learning_rate": 0.001,
                    "tensorboard": False,
                },
                "outputs": {
                    "run_dir": str(run_dir),
                    "checkpoints_dir": str(checkpoints_dir),
                    "models_dir": str(models_dir),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _fake_export_onnx(source_model_path: Path, destination_model_path: Path) -> None:
    destination_model_path.write_text("onnx", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
