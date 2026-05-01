from __future__ import annotations

from collections.abc import Iterable
import json
import secrets
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from wakewords.augment import (
    DEFAULT_CONTEXT_MAX_GAP_MS,
    DEFAULT_CONTEXT_MIN_GAP_MS,
    DEFAULT_CONTEXT_TARGET_DURATION_MS,
    DEFAULT_PARQUET_WRITE_BATCH_SIZE,
)


GOOGLE_SPEECH_COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]

CUSTOM_WORDS_SAMPLE = [
    {"tts_input": "Astra", "label": "astra"},
    {"tts_input": "Boston", "label": "boston"},
    {"tts_input": "Tokyo", "label": "tokyo"}
]

BACKGROUND_AUDIO_URL = (
    "https://github.com/HashNuke/wakewords/releases/download/"
    "background-audio-r2/background_audio.zip"
)


def init_project(project_dir: Path) -> list[Path]:
    data_dir = project_dir / "data"
    background_audio_dir = project_dir / "background_audio"
    config_path = project_dir / "config.json"
    gitignore_path = project_dir / ".gitignore"

    if config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {config_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    background_audio_dir.mkdir(parents=True, exist_ok=True)
    _download_background_audio(background_audio_dir)

    config = {
        "custom_words": CUSTOM_WORDS_SAMPLE,
        "google_speech_commands": GOOGLE_SPEECH_COMMANDS,
        "augment": {
            "seed": secrets.randbits(32),
            "parquet_writes_batch_size": DEFAULT_PARQUET_WRITE_BATCH_SIZE,
            "speech_context": {
                "enabled": False,
                "target_duration_ms": DEFAULT_CONTEXT_TARGET_DURATION_MS,
                "gap_ms": [DEFAULT_CONTEXT_MIN_GAP_MS, DEFAULT_CONTEXT_MAX_GAP_MS],
                "reverse_donor": True,
            },
        },
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _ensure_gitignore_entry(gitignore_path, "google-speech-commands/")
    _ensure_gitignore_entry(gitignore_path, "diagnostics/")

    return [data_dir, background_audio_dir, config_path, gitignore_path]


def _download_background_audio(background_audio_dir: Path, *, url: str = BACKGROUND_AUDIO_URL) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip") as archive_file:
        with urllib.request.urlopen(url, timeout=60) as response:
            archive_file.write(response.read())
        archive_file.flush()
        _extract_background_audio_archive(Path(archive_file.name), background_audio_dir)


def _extract_background_audio_archive(archive_path: Path, background_audio_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        file_infos = [info for info in archive.infolist() if not info.is_dir()]
        common_prefix = _zip_common_directory_prefix(info.filename for info in file_infos)
        for info in file_infos:
            member_path = Path(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe background audio archive member: {info.filename}")

            output_parts = member_path.parts[len(common_prefix) :]
            if not output_parts:
                continue
            output_path = background_audio_dir.joinpath(*output_parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output_path.open("wb") as destination:
                destination.write(source.read())


def _zip_common_directory_prefix(filenames: Iterable[str]) -> tuple[str, ...]:
    paths = [Path(str(filename)).parts for filename in filenames]
    if not paths:
        return ()
    first_parts = paths[0][:-1]
    prefix: list[str] = []
    for index, part in enumerate(first_parts):
        if all(len(path) > index + 1 and path[index] == part for path in paths):
            prefix.append(part)
        else:
            break
    return tuple(prefix)


def _ensure_gitignore_entry(gitignore_path: Path, entry: str) -> None:
    if gitignore_path.exists():
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    if entry in lines:
        return

    text = "\n".join(lines)
    if text:
        text += "\n"
    text += entry + "\n"
    gitignore_path.write_text(text, encoding="utf-8")
