from __future__ import annotations

import json
from pathlib import Path


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


def init_project(project_dir: Path) -> list[Path]:
    data_dir = project_dir / "data"
    config_path = project_dir / "config.json"

    data_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {config_path}")

    config = {
        "custom_words": ["dexa"],
        "google_speech_commands": GOOGLE_SPEECH_COMMANDS,
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return [data_dir, config_path]
