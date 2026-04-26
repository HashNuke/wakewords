from __future__ import annotations

import json
from pathlib import Path


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

    if deleted:
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


def _remove_split_manifests(project_dir: Path) -> list[Path]:
    deleted: list[Path] = []
    for filename in _SPLIT_MANIFESTS:
        manifest_path = project_dir / filename
        if manifest_path.exists():
            manifest_path.unlink()
            deleted.append(manifest_path)
    return deleted
