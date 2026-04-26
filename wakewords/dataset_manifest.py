from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wakewords.download import GOOGLE_SPEECH_COMMANDS_DIR
from wakewords.manifest import load_word_manifest_entries
from wakewords.manifest import probe_wav_duration


def build_split_manifests(
    *,
    data_dir: Path,
    train_ratio: int,
    validate_ratio: int,
    test_ratio: int,
    google_data_dir: Path | None = None,
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

    grouped_entries = _load_grouped_entries(data_dir=data_dir, google_data_dir=google_data_dir)
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


def _load_grouped_entries(*, data_dir: Path, google_data_dir: Path | None) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for word_dir in sorted(path for path in data_dir.iterdir() if path.is_dir() and path.name != "_noises_"):
        for entry in load_word_manifest_entries(word_dir):
            label = entry.get("label")
            if not isinstance(label, str):
                continue
            grouped.setdefault(label, []).append(entry)

    configured_google_words = _load_configured_google_words(data_dir.parent / "config.json")
    speech_commands_dir = google_data_dir or data_dir.parent / GOOGLE_SPEECH_COMMANDS_DIR
    if configured_google_words and speech_commands_dir.exists():
        for entry in _load_google_speech_command_entries(speech_commands_dir, configured_google_words):
            grouped.setdefault(str(entry["label"]), []).append(entry)
    return grouped


def _load_configured_google_words(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    google_words = config.get("google_speech_commands")
    if not isinstance(google_words, list):
        return []
    return [word for word in google_words if isinstance(word, str) and word]


def _load_google_speech_command_entries(dataset_dir: Path, words: list[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for word in words:
        if word == "_background_noise_":
            continue
        word_dir = dataset_dir / word
        if not word_dir.is_dir():
            continue
        for audio_path in sorted(word_dir.glob("*.wav")):
            duration_seconds, duration_ms = probe_wav_duration(audio_path)
            entries.append(
                {
                    "audio_filepath": str(audio_path.resolve()),
                    "duration": duration_seconds,
                    "duration_ms": duration_ms,
                    "label": word,
                    "split_key": f"{GOOGLE_SPEECH_COMMANDS_DIR}/{word}/{audio_path.name}",
                }
            )
    return entries


def _stable_entry_key(entry: dict[str, object]) -> str:
    split_key = entry.get("split_key")
    if isinstance(split_key, str):
        return split_key
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
