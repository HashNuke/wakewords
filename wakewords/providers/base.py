from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Voice:
    id: str
    name: str | None = None
    language: str | None = None


class TTSProvider(Protocol):
    name: str

    def list_voices(
        self,
        pages: int = 1,
        all: bool = False,
        lang: str | None = None,
    ) -> list[Voice]:
        ...

    def generate(
        self,
        *,
        prompts: list[str],
        data_dir: Path,
        parquet_path: Path,
        voice: str | None,
        voices: int | None,
        all_voices: bool,
        lang: str | None,
        concurrency: int,
        model_id: str,
        sample_rate: int,
        encoding: str,
        overwrite: bool,
    ) -> list[Path]:
        ...
