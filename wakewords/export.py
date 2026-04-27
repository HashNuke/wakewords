from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_FORMATS = {"onnx"}


@dataclass(frozen=True)
class ExportBundle:
    output_dir: Path
    model_path: Path
    checkpoint_dir: Path | None
    checkpoint_path: Path | None
    train_config_path: Path | None
    labels_path: Path | None
    config_path: Path

    def paths(self) -> list[Path]:
        paths = [self.output_dir, self.model_path]
        if self.checkpoint_dir is not None:
            paths.append(self.checkpoint_dir)
        if self.checkpoint_path is not None:
            paths.append(self.checkpoint_path)
        if self.train_config_path is not None:
            paths.append(self.train_config_path)
        if self.labels_path is not None:
            paths.append(self.labels_path)
        paths.append(self.config_path)
        return paths


def export_model(
    *,
    project_dir: Path,
    runs_dir: Path,
    run_dir: Path | None,
    model_path: Path | None,
    checkpoint_path: Path | None,
    output_dir: Path,
    format: str,
    overwrite: bool,
) -> ExportBundle:
    export_format = format.lower()
    if export_format not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(f"Unsupported export format: {format}. Supported formats: {supported}")

    project_dir = project_dir.resolve()
    runs_dir = _resolve_project_path(project_dir, runs_dir)
    output_dir = _resolve_project_path(project_dir, output_dir)
    if run_dir is not None:
        run_dir = _resolve_project_path(project_dir, run_dir)
    if model_path is not None:
        model_path = _resolve_project_path(project_dir, model_path)
    if checkpoint_path is not None:
        checkpoint_path = _resolve_project_path(project_dir, checkpoint_path)

    source = _resolve_export_source(
        runs_dir=runs_dir,
        run_dir=run_dir,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
    )
    _require_file(source.model_path, "source model")
    if source.checkpoint_path is not None:
        _require_file(source.checkpoint_path, "checkpoint")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination_model = output_dir / f"model.{export_format}"
    checkpoint_dir = output_dir / "last_checkpoint" if source.checkpoint_path is not None else None
    destination_checkpoint = checkpoint_dir / "last.ckpt" if checkpoint_dir is not None else None
    destination_train_config = checkpoint_dir / "train_config.json" if checkpoint_dir is not None and source.train_config_path is not None else None
    labels = _load_labels(source.train_config_path)
    labels_path = output_dir / "labels.json" if labels is not None else None
    config_path = output_dir / "export_config.json"

    _check_writable(destination_model, overwrite=overwrite)
    if checkpoint_dir is not None:
        _check_writable_dir(checkpoint_dir, overwrite=overwrite)
    if destination_checkpoint is not None:
        _check_writable(destination_checkpoint, overwrite=overwrite)
    if destination_train_config is not None:
        _check_writable(destination_train_config, overwrite=overwrite)
    if labels_path is not None:
        _check_writable(labels_path, overwrite=overwrite)
    _check_writable(config_path, overwrite=overwrite)

    if export_format == "onnx":
        _export_onnx(source.model_path, destination_model)

    if destination_checkpoint is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.checkpoint_path, destination_checkpoint)

    if destination_train_config is not None:
        shutil.copyfile(source.train_config_path, destination_train_config)

    if labels_path is not None:
        labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    export_config = {
        "format": export_format,
        "model_path": str(destination_model),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_path": str(destination_checkpoint) if destination_checkpoint is not None else None,
        "train_config_path": str(destination_train_config) if destination_train_config is not None else None,
        "labels_path": str(labels_path) if labels_path is not None else None,
        "source": {
            "run_dir": str(source.run_dir) if source.run_dir is not None else None,
            "model_path": str(source.model_path),
            "checkpoint_path": str(source.checkpoint_path) if source.checkpoint_path is not None else None,
            "train_config_path": str(source.train_config_path) if source.train_config_path is not None else None,
        },
    }
    config_path.write_text(json.dumps(export_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    return ExportBundle(
        output_dir=output_dir,
        model_path=destination_model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=destination_checkpoint,
        train_config_path=destination_train_config,
        labels_path=labels_path,
        config_path=config_path,
    )


@dataclass(frozen=True)
class _ExportSource:
    run_dir: Path | None
    model_path: Path
    checkpoint_path: Path | None
    train_config_path: Path | None


def _resolve_export_source(
    *,
    runs_dir: Path,
    run_dir: Path | None,
    model_path: Path | None,
    checkpoint_path: Path | None,
) -> _ExportSource:
    if run_dir is None and model_path is None:
        run_dir = _latest_run_dir(runs_dir)
    if run_dir is not None:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
        model_path = model_path or _single_nemo_model(run_dir / "models")
        checkpoint_path = checkpoint_path or _last_checkpoint(run_dir / "checkpoints")
        train_config_path = run_dir / "train_config.json"
        if not train_config_path.is_file():
            train_config_path = None
        return _ExportSource(
            run_dir=run_dir,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            train_config_path=train_config_path,
        )
    if model_path is None:
        raise ValueError("model_path is required when run_dir is not provided")
    return _ExportSource(
        run_dir=None,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        train_config_path=None,
    )


def _latest_run_dir(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "models").is_dir() and any((path / "models").glob("*.nemo"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed training runs found in: {runs_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _single_nemo_model(models_dir: Path) -> Path:
    models = sorted(models_dir.glob("*.nemo"))
    if not models:
        raise FileNotFoundError(f"No .nemo model found in: {models_dir}")
    if len(models) > 1:
        names = ", ".join(path.name for path in models)
        raise ValueError(f"Multiple .nemo models found in {models_dir}: {names}. Pass --model-path explicitly.")
    return models[0]


def _last_checkpoint(checkpoints_dir: Path) -> Path | None:
    checkpoint = checkpoints_dir / "last.ckpt"
    if checkpoint.is_file():
        return checkpoint
    return None


def _load_labels(train_config_path: Path | None) -> list[str] | None:
    if train_config_path is None or not train_config_path.is_file():
        return None
    config = json.loads(train_config_path.read_text(encoding="utf-8"))
    labels = config.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return None
    return labels


def _export_onnx(source_model_path: Path, destination_model_path: Path) -> None:
    if sys.platform == "darwin":
        raise RuntimeError(
            "ONNX export is not supported on macOS because NeMo's ASR dependency chain does not publish macOS wheels. "
            "Run export on Linux with NeMo installed."
        )
    if importlib.util.find_spec("nemo") is None:
        raise RuntimeError("NeMo is not installed. Install package dependencies with: uv sync")

    from nemo.collections.asr.models import EncDecClassificationModel

    model = EncDecClassificationModel.restore_from(str(source_model_path), map_location="cpu")
    model.export(str(destination_model_path))


def _resolve_project_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _check_writable(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Pass --overwrite to replace it.")


def _check_writable_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Refusing to overwrite existing non-directory path: {path}")
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing directory: {path}. Pass --overwrite to replace it.")
