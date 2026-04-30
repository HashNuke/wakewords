from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    forbidden_paths = _forbidden_path_strings(project_dir=project_dir, source=source)

    if destination_checkpoint is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_portable_checkpoint(source.checkpoint_path, destination_checkpoint, forbidden_paths=forbidden_paths)

    if destination_train_config is not None:
        _write_portable_train_config(source.train_config_path, destination_train_config)

    if labels_path is not None:
        labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    export_config = {
        "format": export_format,
        "model_path": _export_relative_path(destination_model, output_dir),
        "checkpoint_dir": _export_relative_path(checkpoint_dir, output_dir)
        if checkpoint_dir is not None
        else None,
        "checkpoint_path": _export_relative_path(destination_checkpoint, output_dir)
        if destination_checkpoint is not None
        else None,
        "train_config_path": _export_relative_path(destination_train_config, output_dir)
        if destination_train_config is not None
        else None,
        "labels_path": _export_relative_path(labels_path, output_dir) if labels_path is not None else None,
    }
    config_path.write_text(json.dumps(export_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _validate_export_privacy(
        paths=[
            path
            for path in (destination_model, destination_checkpoint, destination_train_config, labels_path, config_path)
            if path
        ],
        source=source,
        project_dir=project_dir,
    )

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
        model_path = _latest_run_model(runs_dir)
        run_dir = model_path.parent.parent
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


def _latest_run_model(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")
    candidates = [
        model
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "train_config.json").is_file()
        for model in (path / "models").glob("*.nemo")
        if model.is_file()
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
    return sorted(labels)


def _write_portable_train_config(source_path: Path, destination_path: Path) -> None:
    source_config = json.loads(source_path.read_text(encoding="utf-8"))
    portable_config: dict[str, Any] = {}
    model_name = source_config.get("model_name")
    if isinstance(model_name, str):
        portable_config["model_name"] = model_name
    labels = source_config.get("labels")
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        portable_config["labels"] = labels
    training = source_config.get("training")
    if isinstance(training, dict):
        portable_training_keys = {
            "max_epochs",
            "batch_size",
            "num_workers",
            "accelerator",
            "devices",
            "learning_rate",
            "tensorboard",
        }
        portable_config["training"] = {
            key: value for key, value in training.items() if key in portable_training_keys
        }
    destination_path.write_text(json.dumps(portable_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_portable_checkpoint(source_path: Path, destination_path: Path, *, forbidden_paths: set[str]) -> None:
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
            checkpoint = _sanitize_checkpoint_value(checkpoint, forbidden_paths=forbidden_paths, memo={})
            torch.save(checkpoint, destination_path)
            return
        except Exception:
            if _looks_like_text_file(source_path):
                _write_sanitized_bytes(source_path, destination_path, forbidden_paths=forbidden_paths)
                return
            raise

    _write_sanitized_bytes(source_path, destination_path, forbidden_paths=forbidden_paths)


def _sanitize_checkpoint_value(value: Any, *, forbidden_paths: set[str], memo: dict[int, Any]) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value, forbidden_paths=forbidden_paths)
    if isinstance(value, Path):
        return Path(_sanitize_text(str(value), forbidden_paths=forbidden_paths))
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]
        sanitized: dict[Any, Any] = {}
        memo[value_id] = sanitized
        sanitized.update(
            {
                _sanitize_checkpoint_value(key, forbidden_paths=forbidden_paths, memo=memo): _sanitize_checkpoint_value(
                    item, forbidden_paths=forbidden_paths, memo=memo
                )
                for key, item in value.items()
            }
        )
        return sanitized
    if isinstance(value, list):
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]
        sanitized_list: list[Any] = []
        memo[value_id] = sanitized_list
        sanitized_list.extend(_sanitize_checkpoint_value(item, forbidden_paths=forbidden_paths, memo=memo) for item in value)
        return sanitized_list
    if isinstance(value, tuple):
        return tuple(_sanitize_checkpoint_value(item, forbidden_paths=forbidden_paths, memo=memo) for item in value)
    if isinstance(value, set):
        return {_sanitize_checkpoint_value(item, forbidden_paths=forbidden_paths, memo=memo) for item in value}
    return value


def _write_sanitized_bytes(source_path: Path, destination_path: Path, *, forbidden_paths: set[str]) -> None:
    data = source_path.read_bytes()
    for forbidden_path in sorted(forbidden_paths, key=len, reverse=True):
        forbidden_bytes = forbidden_path.encode()
        data = data.replace(forbidden_bytes, b"x" * len(forbidden_bytes))
    destination_path.write_bytes(data)


def _sanitize_text(value: str, *, forbidden_paths: set[str]) -> str:
    for forbidden_path in sorted(forbidden_paths, key=len, reverse=True):
        value = value.replace(forbidden_path, "<local-path>")
    return value


def _looks_like_text_file(path: Path) -> bool:
    sample = path.read_bytes()[:4096]
    return b"\0" not in sample


def _export_relative_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _validate_export_privacy(*, paths: list[Path], source: _ExportSource, project_dir: Path) -> None:
    forbidden = _forbidden_path_strings(project_dir=project_dir, source=source)
    leaked: list[str] = []
    for path in paths:
        data = path.read_bytes()
        for value in forbidden:
            if value.encode() in data:
                leaked.append(f"{path.name}: {value}")
    if leaked:
        details = "; ".join(leaked)
        raise ValueError(f"Export contains local path information: {details}")


def _forbidden_path_strings(*, project_dir: Path, source: _ExportSource) -> set[str]:
    paths = {project_dir, source.model_path}
    if source.run_dir is not None:
        paths.add(source.run_dir)
    if source.checkpoint_path is not None:
        paths.add(source.checkpoint_path)
    if source.train_config_path is not None:
        paths.add(source.train_config_path)
    home = Path.home()
    if home != Path("/"):
        paths.add(home)
    return {str(path.resolve()) for path in paths if str(path.resolve()) != "/"}


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
