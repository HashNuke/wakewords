from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "commandrecognition_en_matchboxnet3x2x64_v2"


@dataclass(frozen=True)
class TrainingRun:
    run_dir: Path
    config_path: Path
    checkpoints_dir: Path
    logs_dir: Path
    models_dir: Path
    final_model_path: Path

    def paths(self) -> list[Path]:
        return [
            self.run_dir,
            self.config_path,
            self.checkpoints_dir,
            self.logs_dir,
            self.models_dir,
            self.final_model_path,
        ]


def train_model(
    *,
    project_dir: Path,
    data_dir: Path,
    runs_dir: Path,
    run_name: str | None,
    model_name: str,
    base_model_path: Path | None,
    from_checkpoint: Path | None,
    train_manifest: str,
    validation_manifest: str,
    test_manifest: str,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: int | str,
    learning_rate: float | None,
    tensorboard: bool,
    dry_run: bool,
) -> TrainingRun:
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    project_dir = project_dir.resolve()
    data_dir = _resolve_project_path(project_dir, data_dir)
    runs_dir = _resolve_project_path(project_dir, runs_dir)
    if base_model_path is not None:
        base_model_path = _resolve_project_path(project_dir, base_model_path)
    if from_checkpoint is not None:
        from_checkpoint = _resolve_project_path(project_dir, from_checkpoint)
        if run_name is not None:
            raise ValueError("run_name cannot be used with from_checkpoint; the run directory is inferred from the checkpoint")

    train_manifest_path = _resolve_project_path(project_dir, Path(train_manifest))
    validation_manifest_path = _resolve_project_path(project_dir, Path(validation_manifest))
    test_manifest_path = _resolve_project_path(project_dir, Path(test_manifest))

    _require_file(train_manifest_path, "train manifest")
    _require_file(validation_manifest_path, "validation manifest")
    _require_file(test_manifest_path, "test manifest")

    labels = _load_labels(
        project_dir=project_dir,
        manifest_paths=[train_manifest_path, validation_manifest_path, test_manifest_path],
    )
    if len(labels) < 2:
        raise ValueError("Training requires at least two labels.")

    if from_checkpoint is None:
        run = _create_training_run(
            runs_dir=runs_dir,
            run_name=run_name,
            model_name=model_name,
        )
    else:
        run = _load_training_run_from_checkpoint(from_checkpoint=from_checkpoint, model_name=model_name)

    train_config = {
        "model_name": model_name,
        "base_model_path": str(base_model_path) if base_model_path else None,
        "from_checkpoint": str(from_checkpoint) if from_checkpoint else None,
        "labels": labels,
        "manifests": {
            "train": str(train_manifest_path),
            "validation": str(validation_manifest_path),
            "test": str(test_manifest_path),
        },
        "training": {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "accelerator": accelerator,
            "devices": devices,
            "learning_rate": learning_rate,
            "tensorboard": tensorboard,
        },
        "outputs": {
            "run_dir": str(run.run_dir),
            "checkpoints_dir": str(run.checkpoints_dir),
            "logs_dir": str(run.logs_dir),
            "models_dir": str(run.models_dir),
            "final_model_path": str(run.final_model_path),
        },
    }
    run.config_path.write_text(json.dumps(train_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if dry_run:
        return run

    _run_nemo_training(
        run=run,
        model_name=model_name,
        base_model_path=base_model_path,
        from_checkpoint=from_checkpoint,
        labels=labels,
        train_manifest_path=train_manifest_path,
        validation_manifest_path=validation_manifest_path,
        max_epochs=max_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
        devices=devices,
        learning_rate=learning_rate,
        tensorboard=tensorboard,
    )
    return run


def _run_nemo_training(
    *,
    run: TrainingRun,
    model_name: str,
    base_model_path: Path | None,
    from_checkpoint: Path | None,
    labels: list[str],
    train_manifest_path: Path,
    validation_manifest_path: Path,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: int | str,
    learning_rate: float | None,
    tensorboard: bool,
) -> None:
    if sys.platform == "darwin":
        raise RuntimeError(
            "Training is not supported on macOS because NeMo's ASR dependency chain does not publish macOS wheels. "
            "Prepare datasets and manifests on macOS, then run training on Linux."
        )
    if importlib.util.find_spec("nemo") is None:
        raise RuntimeError("NeMo is not installed. Install package dependencies with: uv sync")
    if tensorboard and importlib.util.find_spec("tensorboard") is None:
        raise RuntimeError("TensorBoard is not installed. Install package dependencies with: uv sync")
    if base_model_path is not None and not base_model_path.is_file():
        raise FileNotFoundError(f"Missing base model: {base_model_path}")
    if from_checkpoint is not None and not from_checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {from_checkpoint}")

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from nemo.collections.asr.models import EncDecClassificationModel

    if base_model_path is None:
        model = EncDecClassificationModel.from_pretrained(model_name=model_name)
    else:
        model = EncDecClassificationModel.restore_from(restore_path=str(base_model_path))
    model.change_labels(labels)
    if learning_rate is not None:
        model.cfg.optim.lr = learning_rate

    train_data_config = _dataset_config(
        manifest_path=train_manifest_path,
        labels=labels,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    validation_data_config = _dataset_config(
        manifest_path=validation_manifest_path,
        labels=labels,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    model.setup_training_data(train_data_config)
    model.setup_validation_data(validation_data_config)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(run.checkpoints_dir),
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=3,
        )
    ]
    logger = False
    if tensorboard:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=str(run.logs_dir), name="tensorboard")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        default_root_dir=str(run.run_dir),
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, ckpt_path=str(from_checkpoint) if from_checkpoint else None)
    model.save_to(str(run.final_model_path))


def _dataset_config(
    *,
    manifest_path: Path,
    labels: list[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> dict[str, Any]:
    return {
        "manifest_filepath": str(manifest_path),
        "sample_rate": 16000,
        "labels": labels,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }


def _create_training_run(*, runs_dir: Path, run_name: str | None, model_name: str) -> TrainingRun:
    safe_model_name = _safe_path_name(model_name)
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{safe_model_name}"
    run_dir = runs_dir / _safe_path_name(run_name)
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    models_dir = run_dir / "models"
    for path in (checkpoints_dir, logs_dir, models_dir):
        path.mkdir(parents=True, exist_ok=False)
    return TrainingRun(
        run_dir=run_dir,
        config_path=run_dir / "train_config.json",
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        models_dir=models_dir,
        final_model_path=models_dir / f"{safe_model_name}.nemo",
    )


def _load_training_run_from_checkpoint(*, from_checkpoint: Path, model_name: str) -> TrainingRun:
    if from_checkpoint.parent.name != "checkpoints":
        raise ValueError("from_checkpoint must point to a .ckpt file inside a checkpoints directory")
    if from_checkpoint.suffix != ".ckpt":
        raise ValueError("from_checkpoint must point to a .ckpt file")
    if not from_checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {from_checkpoint}")
    run_dir = from_checkpoint.parent.parent
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    models_dir = run_dir / "models"
    for path in (checkpoints_dir, logs_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)
    safe_model_name = _safe_path_name(model_name)
    return TrainingRun(
        run_dir=run_dir,
        config_path=run_dir / "train_config.json",
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        models_dir=models_dir,
        final_model_path=models_dir / f"{safe_model_name}.nemo",
    )


def _load_labels(*, project_dir: Path, manifest_paths: list[Path]) -> list[str]:
    configured_labels = _load_configured_labels(project_dir=project_dir)
    manifest_labels = _load_manifest_labels(manifest_paths)
    labels = list(configured_labels)
    for label in manifest_labels:
        if label not in labels:
            labels.append(label)
    return labels


def _load_configured_labels(*, project_dir: Path) -> list[str]:
    candidate_words_file = project_dir / "words.json"
    if candidate_words_file.exists():
        words = json.loads(candidate_words_file.read_text(encoding="utf-8"))
        return _dedupe(word["word"] for word in words if isinstance(word, dict) and isinstance(word.get("word"), str))

    config_path = project_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return _dedupe([*config.get("custom_words", []), *config.get("google_speech_commands", [])])
    return []


def _load_manifest_labels(manifest_paths: list[Path]) -> list[str]:
    labels: list[str] = []
    for manifest_path in manifest_paths:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            label = entry.get("label")
            if isinstance(label, str) and label not in labels:
                labels.append(label)
    return labels


def _dedupe(values: Any) -> list[str]:
    labels: list[str] = []
    for value in values:
        if isinstance(value, str) and value not in labels:
            labels.append(value)
    return labels


def _resolve_project_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not safe:
        raise ValueError("run_name must contain at least one path-safe character")
    return safe
