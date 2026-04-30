from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wakewords.lfs import manifest_audio_paths, require_materialized_files


@dataclass(frozen=True)
class TestEvaluation:
    run_dir: Path
    train_config_path: Path
    checkpoint_path: Path
    test_manifest_path: Path
    report_summary_path: Path
    report_rows_path: Path
    metrics: list[dict[str, Any]] | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_dir": str(self.run_dir),
                "train_config_path": str(self.train_config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "test_manifest_path": str(self.test_manifest_path),
                "report_summary_path": str(self.report_summary_path),
                "report_rows_path": str(self.report_rows_path),
                "metrics": _json_safe(self.metrics),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )


def test_model(
    *,
    project_dir: Path,
    runs_dir: Path,
    run_dir: Path | None,
    checkpoint_path: Path | None,
    test_manifest: str | None,
    batch_size: int | None,
    num_workers: int | None,
    accelerator: str | None,
    devices: int | str | None,
    dry_run: bool,
) -> TestEvaluation:
    project_dir = project_dir.resolve()
    runs_dir = _resolve_project_path(project_dir, runs_dir)
    if run_dir is not None:
        run_dir = _resolve_project_path(project_dir, run_dir)
    if checkpoint_path is not None:
        checkpoint_path = _resolve_project_path(project_dir, checkpoint_path)

    source = _resolve_test_source(
        runs_dir=runs_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )
    train_config = _load_train_config(source.train_config_path)
    labels = _load_labels(train_config)
    training = _load_training_config(train_config)
    model_name = _load_model_name(train_config)
    base_model_path = _load_base_model_path(project_dir=project_dir, train_config=train_config)
    manifest_path = _resolve_test_manifest(
        project_dir=project_dir,
        train_config=train_config,
        test_manifest=test_manifest,
    )

    _require_file(source.checkpoint_path, "checkpoint")
    _require_file(manifest_path, "test manifest")
    require_materialized_files(
        [source.checkpoint_path, manifest_path, *([base_model_path] if base_model_path is not None else [])],
        context="test evaluation inputs",
        include_hint="runs/**,models/**,data/manifests/*.jsonl",
    )
    require_materialized_files(
        manifest_audio_paths([manifest_path]),
        context="test evaluation audio",
        include_hint="data/**/*.wav",
    )

    report_summary_path = source.run_dir / "test_report_summary.json"
    report_rows_path = source.run_dir / "test_report.jsonl"
    metrics = None
    if not dry_run:
        metrics, samples = _run_nemo_test(
            model_name=model_name,
            base_model_path=base_model_path,
            checkpoint_path=source.checkpoint_path,
            test_manifest_path=manifest_path,
            labels=labels,
            batch_size=batch_size if batch_size is not None else int(training.get("batch_size", 32)),
            num_workers=num_workers if num_workers is not None else int(training.get("num_workers", 4)),
            accelerator=accelerator if accelerator is not None else str(training.get("accelerator", "auto")),
            devices=devices if devices is not None else training.get("devices", "auto"),
        )
        _write_test_report(
            report_summary_path=report_summary_path,
            report_rows_path=report_rows_path,
            run_dir=source.run_dir,
            train_config_path=source.train_config_path,
            checkpoint_path=source.checkpoint_path,
            test_manifest_path=manifest_path,
            metrics=metrics,
            samples=samples,
        )

    return TestEvaluation(
        run_dir=source.run_dir,
        train_config_path=source.train_config_path,
        checkpoint_path=source.checkpoint_path,
        test_manifest_path=manifest_path,
        report_summary_path=report_summary_path,
        report_rows_path=report_rows_path,
        metrics=metrics,
    )


@dataclass(frozen=True)
class _TestSource:
    run_dir: Path
    train_config_path: Path
    checkpoint_path: Path


def _resolve_test_source(*, runs_dir: Path, run_dir: Path | None, checkpoint_path: Path | None) -> _TestSource:
    if run_dir is None and checkpoint_path is None:
        run_dir = _latest_run_dir(runs_dir)
    if checkpoint_path is not None and run_dir is None:
        if checkpoint_path.parent.name != "checkpoints":
            raise ValueError("run_dir is required when checkpoint_path is not inside a checkpoints directory")
        run_dir = checkpoint_path.parent.parent
    if run_dir is None:
        raise ValueError("run_dir could not be resolved")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    train_config_path = run_dir / "train_config.json"
    _require_file(train_config_path, "train config")
    checkpoint_path = checkpoint_path or _last_checkpoint(run_dir / "checkpoints")
    _require_file(checkpoint_path, "checkpoint")
    return _TestSource(run_dir=run_dir, train_config_path=train_config_path, checkpoint_path=checkpoint_path)


def _run_nemo_test(
    *,
    model_name: str,
    base_model_path: Path | None,
    checkpoint_path: Path,
    test_manifest_path: Path,
    labels: list[str],
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: int | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sys.platform == "darwin":
        raise RuntimeError(
            "Test evaluation is not supported on macOS because NeMo's ASR dependency chain does not publish macOS wheels. "
            "Run test evaluation on Linux with NeMo installed."
        )
    if importlib.util.find_spec("nemo") is None:
        raise RuntimeError("NeMo is not installed. Install package dependencies with: uv sync")

    from lightning.pytorch import Trainer
    from nemo.collections.asr.models import EncDecClassificationModel

    if base_model_path is None:
        model = EncDecClassificationModel.from_pretrained(model_name=model_name)
    else:
        model = EncDecClassificationModel.restore_from(restore_path=str(base_model_path))
    model.change_labels(labels)
    model.setup_test_data(
        {
            "manifest_filepath": str(test_manifest_path),
            "sample_rate": 16000,
            "labels": labels,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": True,
        }
    )
    trainer = Trainer(accelerator=accelerator, devices=devices, logger=False)
    results = trainer.test(model, ckpt_path=str(checkpoint_path), verbose=True)
    return [dict(result) for result in results], _predict_test_samples(model=model, labels=labels, manifest_path=test_manifest_path)


def _predict_test_samples(*, model: Any, labels: list[str], manifest_path: Path) -> list[dict[str, Any]]:
    import torch

    manifest_entries = _read_manifest_entries(manifest_path)
    samples: list[dict[str, Any]] = []
    model.eval()
    device = next(model.parameters()).device
    offset = 0
    with torch.no_grad():
        for batch in model.test_dataloader():
            audio_signal = batch[0].to(device)
            audio_signal_length = batch[1].to(device)
            actual_ids = batch[2].detach().cpu().view(-1).tolist()
            logits = model(input_signal=audio_signal, input_signal_length=audio_signal_length)
            if isinstance(logits, tuple):
                logits = logits[0]
            probabilities = torch.softmax(logits.detach().cpu(), dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1).view(-1).tolist()
            for row_index, (actual_id, predicted_id) in enumerate(zip(actual_ids, predicted_ids, strict=False)):
                manifest_entry = manifest_entries[offset + row_index]
                actual_label = _label_at(labels, int(actual_id))
                predicted_label = _label_at(labels, int(predicted_id))
                probability = float(probabilities[row_index, int(predicted_id)].item())
                samples.append(
                    {
                        "index": offset + row_index,
                        "audio_filepath": manifest_entry["audio_filepath"],
                        "actual_label": actual_label,
                        "manifest_label": manifest_entry["label"],
                        "predicted_label": predicted_label,
                        "probability": probability,
                        "correct": actual_label == predicted_label,
                    }
                )
            offset += len(actual_ids)
    return samples


def _read_manifest_entries(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        audio_filepath = item.get("audio_filepath")
        label = item.get("label")
        if not isinstance(audio_filepath, str) or not isinstance(label, str):
            raise ValueError(f"Invalid manifest entry in {path}:{line_number}")
        entries.append({"audio_filepath": audio_filepath, "label": label})
    return entries


def _label_at(labels: list[str], index: int) -> str:
    if 0 <= index < len(labels):
        return labels[index]
    return f"class_{index}"


def _write_test_report(
    *,
    report_summary_path: Path,
    report_rows_path: Path,
    run_dir: Path,
    train_config_path: Path,
    checkpoint_path: Path,
    test_manifest_path: Path,
    metrics: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> None:
    report_summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "train_config_path": str(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "test_manifest_path": str(test_manifest_path),
        "report_rows_path": str(report_rows_path),
        "metrics": _json_safe(metrics),
        "sample_count": len(samples),
        "correct_count": sum(1 for sample in samples if sample.get("correct") is True),
    }
    report_summary_path.write_text(json.dumps(report_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_rows_path.write_text(
        "".join(json.dumps(_json_safe(sample), ensure_ascii=True, sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )


def _latest_run_dir(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")
    candidates = [path for path in runs_dir.iterdir() if path.is_dir() and (path / "train_config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No training runs found in: {runs_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _last_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoint = checkpoints_dir / "last.ckpt"
    if checkpoint.is_file():
        return checkpoint
    checkpoints = sorted(checkpoints_dir.glob("*.ckpt")) if checkpoints_dir.is_dir() else []
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoints_dir}")
    return max(checkpoints, key=lambda path: (path.stat().st_mtime, path.name))


def _load_train_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid train config: {path}")
    return config


def _load_labels(train_config: dict[str, Any]) -> list[str]:
    labels = train_config.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("train_config.json must contain a non-empty string list at labels")
    return labels


def _load_training_config(train_config: dict[str, Any]) -> dict[str, Any]:
    training = train_config.get("training")
    if not isinstance(training, dict):
        return {}
    return training


def _load_model_name(train_config: dict[str, Any]) -> str:
    model_name = train_config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("train_config.json must contain model_name")
    return model_name


def _load_base_model_path(*, project_dir: Path, train_config: dict[str, Any]) -> Path | None:
    raw_path = train_config.get("base_model_path")
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("train_config.json base_model_path must be null or a string")
    return _resolve_project_path(project_dir, Path(raw_path))


def _resolve_test_manifest(*, project_dir: Path, train_config: dict[str, Any], test_manifest: str | None) -> Path:
    if test_manifest is not None:
        return _resolve_project_path(project_dir, Path(test_manifest))
    manifests = train_config.get("manifests")
    if not isinstance(manifests, dict) or not isinstance(manifests.get("test"), str):
        raise ValueError("train_config.json must contain manifests.test, or pass --test-manifest")
    return _resolve_project_path(project_dir, Path(manifests["test"]))


def _resolve_project_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
