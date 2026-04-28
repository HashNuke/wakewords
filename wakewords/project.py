from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

from wakewords.augment import DEFAULT_PARQUET_WRITE_BATCH_SIZE


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


def init_project(project_dir: Path) -> list[Path]:
    data_dir = project_dir / "data"
    background_audio_dir = project_dir / "background_audio"
    config_path = project_dir / "config.json"
    gitignore_path = project_dir / ".gitignore"

    data_dir.mkdir(parents=True, exist_ok=True)
    background_audio_dir.mkdir(parents=True, exist_ok=True)

    background_noise_files = resources.files("wakewords.google_scd_background_noise")
    for resource in background_noise_files.iterdir():
        if not resource.is_file() or resource.name == "__init__.py":
            continue
        destination = background_audio_dir / resource.name
        with resources.as_file(resource) as source:
            shutil.copyfile(source, destination)

    if config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {config_path}")

    config = {
        "custom_words": CUSTOM_WORDS_SAMPLE,
        "google_speech_commands": GOOGLE_SPEECH_COMMANDS,
        "augment": {
            "parquet_writes_batch_size": DEFAULT_PARQUET_WRITE_BATCH_SIZE,
        },
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _ensure_gitignore_entry(gitignore_path, "google-speech-commands/")
    _ensure_gitignore_entry(gitignore_path, "diagnostics/")

    return [data_dir, background_audio_dir, config_path, gitignore_path]


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
