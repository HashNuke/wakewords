from __future__ import annotations

import json
from pathlib import Path

from wakewords.parquet_store import CustomWordStore


_BASE_SUFFIX = "-t100-clean-nonoise-nosnr.wav"
_SPLIT_MANIFESTS = ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl")


def clean_dataset(
    *,
    data_dir: Path,
    generated: bool = False,
    augmented: bool = False,
    all: bool = False,
) -> list[Path]:
    if all and (generated or augmented):
        raise ValueError("Use only one clean mode: --generated, --augmented, or --all.")
    if not all and not generated and not augmented:
        raise ValueError("Choose what to clean with --generated, --augmented, or --all.")

    parquet_path = data_dir / "custom_words.parquet"
    deleted: list[Path] = []
    removed_rows = _remove_parquet_rows(parquet_path, generated=generated, augmented=augmented, all=all)
    if removed_rows:
        deleted.append(parquet_path)
        deleted.extend(_remove_compatibility_outputs(data_dir, removed_rows))
    else:
        deleted.extend(_remove_legacy_outputs(data_dir, generated=generated, augmented=augmented, all=all))

    if deleted:
        deleted.extend(_remove_split_manifests(data_dir))
        deleted.extend(_remove_split_manifests(data_dir.parent))
    return deleted


def _should_delete(path: Path, *, generated: bool, augmented: bool, all: bool) -> bool:
    if all:
        return True
    is_generated = path.name.endswith(_BASE_SUFFIX)
    return (generated and is_generated) or (augmented and not is_generated)


def _remove_manifest_entries(manifest_path: Path, deleted_names: set[str]) -> None:
    if not manifest_path.exists():
        return

    remaining: list[dict[str, object]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        audio_filepath = entry.get("audio_filepath")
        if not isinstance(audio_filepath, str):
            continue
        if Path(audio_filepath).name not in deleted_names:
            remaining.append(entry)

    if remaining:
        lines = [json.dumps(entry, sort_keys=True) for entry in remaining]
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        manifest_path.unlink()


def _remove_split_manifests(manifests_root: Path) -> list[Path]:
    deleted: list[Path] = []
    for filename in _SPLIT_MANIFESTS:
        manifest_path = manifests_root / "manifests" / filename
        if not manifest_path.exists():
            manifest_path = manifests_root / filename
        if manifest_path.exists():
            manifest_path.unlink()
            deleted.append(manifest_path)
    return deleted


def _remove_parquet_rows(
    parquet_path: Path,
    *,
    generated: bool,
    augmented: bool,
    all: bool,
) -> list[dict[str, object]]:
    if not parquet_path.exists():
        return []
    store = CustomWordStore(parquet_path)
    return store.delete_matching(lambda row: _should_delete_row(row, generated=generated, augmented=augmented, all=all))


def _should_delete_row(row: dict[str, object], *, generated: bool, augmented: bool, all: bool) -> bool:
    if all:
        return True
    source_type = row.get("source_type")
    if not isinstance(source_type, str):
        return False
    return (generated and source_type == "generated") or (augmented and source_type == "augmented")


def _remove_compatibility_outputs(data_dir: Path, removed_rows: list[dict[str, object]]) -> list[Path]:
    deleted: list[Path] = []
    deleted_names_by_label: dict[str, set[str]] = {}
    for row in removed_rows:
        label = row.get("label")
        filename = row.get("filename")
        sample_id = row.get("sample_id")
        if not isinstance(label, str):
            continue
        if isinstance(filename, str):
            compatibility_path = data_dir / label / filename
            if compatibility_path.exists():
                compatibility_path.unlink()
                deleted.append(compatibility_path)
            deleted_names_by_label.setdefault(label, set()).add(filename)
        if isinstance(sample_id, str):
            materialized_path = data_dir / "custom-words" / label / f"{sample_id}.wav"
            if materialized_path.exists():
                materialized_path.unlink()
                deleted.append(materialized_path)

    for label, deleted_names in deleted_names_by_label.items():
        _remove_manifest_entries(data_dir / label / "manifest.jsonl", deleted_names)
    return deleted


def _remove_legacy_outputs(data_dir: Path, *, generated: bool, augmented: bool, all: bool) -> list[Path]:
    deleted: list[Path] = []
    for word_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        deleted_names: set[str] = set()
        for wav_path in sorted(word_dir.glob("*.wav")):
            if _should_delete(wav_path, generated=generated, augmented=augmented, all=all):
                wav_path.unlink()
                deleted.append(wav_path)
                deleted_names.add(wav_path.name)
        if deleted_names:
            _remove_manifest_entries(word_dir / "manifest.jsonl", deleted_names)
    return deleted
