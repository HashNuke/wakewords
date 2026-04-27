from __future__ import annotations

from pathlib import Path

from wakewords.parquet_store import CustomWordStore


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
        deleted.extend(_remove_materialized_outputs(data_dir, removed_rows))

    if deleted:
        deleted.extend(_remove_split_manifests(data_dir))
        deleted.extend(_remove_split_manifests(data_dir.parent))
    return deleted


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


def _remove_materialized_outputs(data_dir: Path, removed_rows: list[dict[str, object]]) -> list[Path]:
    deleted: list[Path] = []
    for row in removed_rows:
        label = row.get("label")
        sample_id = row.get("sample_id")
        if not isinstance(label, str) or not isinstance(sample_id, str):
            continue
        materialized_path = data_dir / "custom-words" / label / f"{sample_id}.wav"
        if materialized_path.exists():
            materialized_path.unlink()
            deleted.append(materialized_path)
    return deleted
