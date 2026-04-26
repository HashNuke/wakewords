from __future__ import annotations

import json
import threading
import wave
from pathlib import Path


class ManifestStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._manifests: dict[Path, _WordManifest] = {}

    def for_word_dir(self, word_dir: Path) -> "_WordManifest":
        manifest_path = word_dir / "manifest.jsonl"
        with self._lock:
            manifest = self._manifests.get(manifest_path)
            if manifest is None:
                manifest = _WordManifest(manifest_path)
                self._manifests[manifest_path] = manifest
            return manifest


class _WordManifest:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, object]] = {}
        self._load()

    def get(self, audio_path: Path) -> dict[str, object] | None:
        key = _local_audio_key(self._path.parent, audio_path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            return dict(entry)

    def record(self, *, audio_path: Path, label: str) -> dict[str, object]:
        entry = build_manifest_entry(audio_path=audio_path, label=label)
        with self._lock:
            self._entries[entry["audio_filepath"]] = entry
            self._write()
            return dict(entry)

    def _load(self) -> None:
        if not self._path.exists():
            return
        word_dir = self._path.parent
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_filepath = entry.get("audio_filepath")
            if isinstance(audio_filepath, str):
                normalized_entry = {
                    **entry,
                    "audio_filepath": _local_audio_key(word_dir, resolve_audio_path(word_dir, audio_filepath)),
                }
                self._entries[normalized_entry["audio_filepath"]] = normalized_entry

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self._entries[key], sort_keys=True) for key in sorted(self._entries)]
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_manifest_entry(*, audio_path: Path, label: str) -> dict[str, object]:
    duration_seconds, duration_ms = probe_wav_duration(audio_path)
    return {
        "audio_filepath": audio_path.name,
        "duration": duration_seconds,
        "duration_ms": duration_ms,
        "label": label,
    }


def load_word_manifest_entries(word_dir: Path) -> list[dict[str, object]]:
    manifest_path = word_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        audio_filepath = entry.get("audio_filepath")
        if not isinstance(audio_filepath, str):
            continue
        resolved_audio_path = resolve_audio_path(word_dir, audio_filepath)
        entries.append(
            {
                **entry,
                "audio_filepath": str(resolved_audio_path.resolve()),
            }
        )
    return entries


def resolve_audio_path(word_dir: Path, audio_filepath: str) -> Path:
    audio_path = Path(audio_filepath)
    if audio_path.is_absolute():
        return audio_path
    return word_dir / audio_path


def probe_wav_duration(audio_path: Path) -> tuple[float, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
    duration_seconds = frame_count / frame_rate
    duration_ms = round(duration_seconds * 1000)
    return duration_seconds, duration_ms


def _local_audio_key(word_dir: Path, audio_path: Path) -> str:
    try:
        return str(audio_path.resolve().relative_to(word_dir.resolve()))
    except ValueError:
        return audio_path.name
