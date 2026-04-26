from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakewords import cli
from wakewords.train import DEFAULT_MODEL_NAME, train_model


class TrainTests(unittest.TestCase):
    def test_train_model_dry_run_creates_project_local_run_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(data_dir / filename)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": ["dexa"], "google_speech_commands": ["yes"]}) + "\n",
                encoding="utf-8",
            )

            run = train_model(
                project_dir=project_dir,
                data_dir=Path("data"),
                runs_dir=Path("runs"),
                run_name="smoke run",
                model_name=DEFAULT_MODEL_NAME,
                train_manifest="train_manifest.jsonl",
                validation_manifest="validation_manifest.jsonl",
                test_manifest="test_manifest.jsonl",
                words_file=None,
                max_epochs=1,
                batch_size=2,
                num_workers=0,
                accelerator="cpu",
                devices=1,
                learning_rate=None,
                tensorboard=True,
                dry_run=True,
            )

            self.assertEqual(run.run_dir, project_dir / "runs" / "smoke-run")
            self.assertTrue(run.checkpoints_dir.is_dir())
            self.assertTrue(run.logs_dir.is_dir())
            self.assertTrue(run.models_dir.is_dir())
            self.assertEqual(run.final_model_path, run.models_dir / f"{DEFAULT_MODEL_NAME}.nemo")

            train_config = json.loads(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(train_config["labels"], ["dexa", "yes", "unknown"])
            self.assertEqual(train_config["training"]["tensorboard"], True)
            self.assertEqual(train_config["outputs"]["checkpoints_dir"], str(run.checkpoints_dir))

    def test_cli_train_dry_run_prints_run_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(data_dir / filename)

            stdout = io.StringIO()
            argv = [
                "wakewords",
                "train",
                "--project-dir",
                tmp_dir,
                "--run-name",
                "cli smoke",
                "--max-epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--dry-run",
            ]

            with mock.patch.object(cli.sys, "argv", argv):
                with redirect_stdout(stdout):
                    cli.main()

            run_dir = project_dir / "runs" / "cli-smoke"
            self.assertEqual(
                stdout.getvalue().strip().splitlines(),
                [
                    str(run_dir),
                    str(run_dir / "train_config.json"),
                    str(run_dir / "checkpoints"),
                    str(run_dir / "logs"),
                    str(run_dir / "models"),
                    str(run_dir / "models" / f"{DEFAULT_MODEL_NAME}.nemo"),
                ],
            )


def _write_manifest(path: Path) -> None:
    entries = [
        {"audio_filepath": str(path.parent / "yes.wav"), "duration": 1.0, "label": "yes"},
        {"audio_filepath": str(path.parent / "unknown.wav"), "duration": 1.0, "label": "unknown"},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
