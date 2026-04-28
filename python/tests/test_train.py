from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakewords import cli
from wakewords.lfs import GitLfsPointerError
from wakewords.train import DEFAULT_MODEL_NAME, train_model


class TrainTests(unittest.TestCase):
    def test_train_model_dry_run_creates_project_local_run_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(project_dir / filename)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": [{"tts_input": "Dexa", "label": "dexa"}], "google_speech_commands": ["yes"]}) + "\n",
                encoding="utf-8",
            )

            run = train_model(
                project_dir=project_dir,
                data_dir=Path("data"),
                runs_dir=Path("runs"),
                run_name="smoke run",
                model_name=DEFAULT_MODEL_NAME,
                base_model_path=None,
                from_checkpoint=None,
                train_manifest="train_manifest.jsonl",
                validation_manifest="validation_manifest.jsonl",
                test_manifest="test_manifest.jsonl",
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
            self.assertIsNone(train_config["base_model_path"])
            self.assertEqual(train_config["training"]["tensorboard"], True)
            self.assertEqual(train_config["outputs"]["checkpoints_dir"], str(run.checkpoints_dir))

    def test_cli_train_dry_run_prints_run_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            manifests_dir = data_dir / "manifests"
            manifests_dir.mkdir(parents=True)
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(manifests_dir / filename)

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

    def test_train_model_dry_run_records_explicit_base_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(project_dir / filename)

            run = train_model(
                project_dir=project_dir,
                data_dir=Path("data"),
                runs_dir=Path("runs"),
                run_name="local base",
                model_name=DEFAULT_MODEL_NAME,
                base_model_path=Path("models/custom.nemo"),
                from_checkpoint=None,
                train_manifest="train_manifest.jsonl",
                validation_manifest="validation_manifest.jsonl",
                test_manifest="test_manifest.jsonl",
                max_epochs=1,
                batch_size=2,
                num_workers=0,
                accelerator="cpu",
                devices=1,
                learning_rate=None,
                tensorboard=True,
                dry_run=True,
            )

            train_config = json.loads(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(train_config["base_model_path"], str(project_dir / "models" / "custom.nemo"))

    def test_train_model_dry_run_reuses_run_dir_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(project_dir / filename)
            checkpoint = project_dir / "runs" / "existing-run" / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")

            run = train_model(
                project_dir=project_dir,
                data_dir=Path("data"),
                runs_dir=Path("runs"),
                run_name=None,
                model_name=DEFAULT_MODEL_NAME,
                base_model_path=None,
                from_checkpoint=Path("runs/existing-run/checkpoints/last.ckpt"),
                train_manifest="train_manifest.jsonl",
                validation_manifest="validation_manifest.jsonl",
                test_manifest="test_manifest.jsonl",
                max_epochs=20,
                batch_size=2,
                num_workers=0,
                accelerator="cpu",
                devices=1,
                learning_rate=None,
                tensorboard=True,
                dry_run=True,
            )

            self.assertEqual(run.run_dir, project_dir / "runs" / "existing-run")
            self.assertEqual(run.checkpoints_dir, checkpoint.parent)
            self.assertTrue(run.logs_dir.is_dir())
            self.assertTrue(run.models_dir.is_dir())
            train_config = json.loads(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(train_config["from_checkpoint"], str(checkpoint))
            self.assertEqual(train_config["training"]["max_epochs"], 20)

    def test_train_model_dry_run_imports_exported_checkpoint_into_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest(project_dir / filename)
            exported_checkpoint_dir = project_dir / "models" / "last_checkpoint"
            exported_checkpoint_dir.mkdir(parents=True)
            exported_checkpoint = exported_checkpoint_dir / "last.ckpt"
            exported_train_config = exported_checkpoint_dir / "train_config.json"
            exported_checkpoint.write_text("checkpoint", encoding="utf-8")
            exported_train_config.write_text(json.dumps({"labels": ["yes", "unknown"], "training": {"max_epochs": 10}}) + "\n", encoding="utf-8")

            run = train_model(
                project_dir=project_dir,
                data_dir=Path("data"),
                runs_dir=Path("runs"),
                run_name="continued export",
                model_name=DEFAULT_MODEL_NAME,
                base_model_path=None,
                from_checkpoint=Path("models/last_checkpoint/last.ckpt"),
                train_manifest="train_manifest.jsonl",
                validation_manifest="validation_manifest.jsonl",
                test_manifest="test_manifest.jsonl",
                max_epochs=20,
                batch_size=2,
                num_workers=0,
                accelerator="cpu",
                devices=1,
                learning_rate=None,
                tensorboard=True,
                dry_run=True,
            )

            imported_checkpoint = project_dir / "runs" / "continued-export" / "checkpoints" / "last.ckpt"
            self.assertEqual(run.run_dir, project_dir / "runs" / "continued-export")
            self.assertEqual(run.checkpoints_dir, imported_checkpoint.parent)
            self.assertEqual(imported_checkpoint.read_text(encoding="utf-8"), "checkpoint")
            self.assertEqual(
                json.loads((run.run_dir / "source_train_config.json").read_text(encoding="utf-8")),
                {"labels": ["yes", "unknown"], "training": {"max_epochs": 10}},
            )
            train_config = json.loads(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(train_config["from_checkpoint"], str(imported_checkpoint))
            self.assertEqual(
                train_config["resume_source"],
                {
                    "checkpoint_path": str(exported_checkpoint),
                    "train_config_path": str(exported_train_config),
                },
            )
            self.assertEqual(train_config["training"]["max_epochs"], 20)

    def test_train_model_rejects_run_name_with_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "run_name"):
                train_model(
                    project_dir=project_dir,
                    data_dir=Path("data"),
                    runs_dir=Path("runs"),
                    run_name="new-run",
                    model_name=DEFAULT_MODEL_NAME,
                    base_model_path=None,
                    from_checkpoint=Path("runs/existing-run/checkpoints/last.ckpt"),
                    train_manifest="train_manifest.jsonl",
                    validation_manifest="validation_manifest.jsonl",
                    test_manifest="test_manifest.jsonl",
                    max_epochs=20,
                    batch_size=2,
                    num_workers=0,
                    accelerator="cpu",
                    devices=1,
                    learning_rate=None,
                    tensorboard=True,
                    dry_run=True,
                )

    def test_train_model_reports_lfs_manifest_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            data_dir.mkdir()
            audio_path = project_dir / "yes.wav"
            audio_path.write_bytes(_lfs_pointer_bytes(size=32000))
            for filename in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                _write_manifest_with_audio(project_dir / filename, audio_path)

            with self.assertRaisesRegex(GitLfsPointerError, "git lfs pull"):
                train_model(
                    project_dir=project_dir,
                    data_dir=Path("data"),
                    runs_dir=Path("runs"),
                    run_name="lfs audio",
                    model_name=DEFAULT_MODEL_NAME,
                    base_model_path=None,
                    from_checkpoint=None,
                    train_manifest="train_manifest.jsonl",
                    validation_manifest="validation_manifest.jsonl",
                    test_manifest="test_manifest.jsonl",
                    max_epochs=1,
                    batch_size=2,
                    num_workers=0,
                    accelerator="cpu",
                    devices=1,
                    learning_rate=None,
                    tensorboard=True,
                    dry_run=True,
                )


def _write_manifest(path: Path) -> None:
    entries = [
        {"audio_filepath": str(path.parent / "yes.wav"), "duration": 1.0, "label": "yes"},
        {"audio_filepath": str(path.parent / "unknown.wav"), "duration": 1.0, "label": "unknown"},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _write_manifest_with_audio(path: Path, audio_path: Path) -> None:
    entries = [
        {"audio_filepath": str(audio_path), "duration": 1.0, "label": "yes"},
        {"audio_filepath": str(path.parent / "unknown.wav"), "duration": 1.0, "label": "unknown"},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _lfs_pointer_bytes(*, size: int) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        + f"size {size}\n".encode("ascii")
    )


if __name__ == "__main__":
    unittest.main()
