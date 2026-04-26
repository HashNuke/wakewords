from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wakewords.manifest import load_word_manifest_entries


def build_split_manifests(
    *,
    data_dir: Path,
    train_ratio: int,
    validate_ratio: int,
    test_ratio: int,
    train_filename: str = "train_manifest.jsonl",
    validate_filename: str = "validation_manifest.jsonl",
    test_filename: str = "test_manifest.jsonl",
) -> dict[str, Path]:
    ratios = {
        "train": train_ratio,
        "validate": validate_ratio,
        "test": test_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be >= 0.")
    if sum(ratios.values()) <= 0:
        raise ValueError("At least one split ratio must be > 0.")

    grouped_entries = _load_grouped_entries(data_dir)
    split_entries = {"train": [], "validate": [], "test": []}

    for entries in grouped_entries.values():
        ordered_entries = sorted(entries, key=_stable_entry_key)
        counts = _split_counts(len(ordered_entries), ratios)
        start = 0
        for split_name in ("train", "validate", "test"):
            end = start + counts[split_name]
            split_entries[split_name].extend(ordered_entries[start:end])
            start = end

    output_paths = {
        "train": data_dir.parent / train_filename,
        "validate": data_dir.parent / validate_filename,
        "test": data_dir.parent / test_filename,
    }
    for split_name, output_path in output_paths.items():
        _write_manifest(output_path, split_entries[split_name])
    return output_paths


def _load_grouped_entries(data_dir: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for word_dir in sorted(path for path in data_dir.iterdir() if path.is_dir() and path.name != "_noises_"):
        for entry in load_word_manifest_entries(word_dir):
            label = entry.get("label")
            if not isinstance(label, str):
                continue
            grouped.setdefault(label, []).append(entry)
    return grouped


def _stable_entry_key(entry: dict[str, object]) -> str:
    audio_filepath = entry.get("audio_filepath")
    if not isinstance(audio_filepath, str):
        audio_filepath = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(audio_filepath.encode("utf-8")).hexdigest()


def _split_counts(total: int, ratios: dict[str, int]) -> dict[str, int]:
    ratio_sum = sum(ratios.values())
    raw_counts = {name: total * value / ratio_sum for name, value in ratios.items()}
    counts = {name: int(raw_counts[name]) for name in ratios}
    remaining = total - sum(counts.values())

    priorities = sorted(
        ratios,
        key=lambda name: (raw_counts[name] - counts[name], ratios[name], _split_order(name)),
        reverse=True,
    )
    for split_name in priorities[:remaining]:
        counts[split_name] += 1
    return counts


def _split_order(name: str) -> int:
    return {"train": 2, "validate": 1, "test": 0}[name]


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    lines = []
    for entry in entries:
        lines.append(
            json.dumps(
                {
                    "audio_filepath": entry["audio_filepath"],
                    "duration": entry["duration"],
                    "label": entry["label"],
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
