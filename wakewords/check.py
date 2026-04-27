from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from wakewords.audio import wav_has_speech
from wakewords.parquet_store import CustomWordStore


@dataclass(frozen=True)
class CheckStats:
    parquet_path: Path
    no_speech_path: Path
    source_types: tuple[str, ...]
    sample_count: int
    median_duration_ms: float | None
    longest_duration_ms: int | None
    longest_sample_id: str | None
    no_speech_count: int

    def lines(self) -> list[str]:
        source_types = ", ".join(self.source_types)
        median_duration = "n/a" if self.median_duration_ms is None else _format_duration(self.median_duration_ms)
        longest_duration = "n/a" if self.longest_duration_ms is None else _format_duration(self.longest_duration_ms)
        longest_sample_id = self.longest_sample_id or "n/a"
        return [
            f"parquet: {self.parquet_path}",
            f"sources: {source_types}",
            f"samples: {self.sample_count}",
            f"median_duration_ms: {median_duration}",
            f"longest_duration_ms: {longest_duration}",
            f"longest_sample_id: {longest_sample_id}",
            f"no_speech_count: {self.no_speech_count}",
            f"no_speech_file: {self.no_speech_path}",
        ]


def check_dataset(
    *,
    project_dir: Path,
    data_dir: Path,
    all: bool,
    generated: bool,
    augmented: bool,
) -> CheckStats:
    if all and (generated or augmented):
        raise ValueError("Use only one check mode: --all, --generated, or --augmented.")

    source_types = _source_types(all=all, generated=generated, augmented=augmented)
    project_dir = project_dir.resolve()
    data_dir = _resolve_project_path(project_dir, data_dir)
    parquet_path = data_dir / "custom_words.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing custom words Parquet file: {parquet_path}")

    rows = [row for row in CustomWordStore(parquet_path).rows() if row.get("source_type") in source_types]
    durations = [_duration_ms(row) for row in rows]
    longest_row = max(rows, key=_duration_ms, default=None)
    no_speech_sample_ids = [
        sample_id
        for row in rows
        if (sample_id := _sample_id(row)) is not None and not wav_has_speech(_audio_bytes(row))
    ]

    no_speech_path = project_dir / "no-speech.txt"
    no_speech_path.write_text("".join(f"{sample_id}\n" for sample_id in no_speech_sample_ids), encoding="utf-8")

    return CheckStats(
        parquet_path=parquet_path,
        no_speech_path=no_speech_path,
        source_types=source_types,
        sample_count=len(rows),
        median_duration_ms=median(durations) if durations else None,
        longest_duration_ms=_duration_ms(longest_row) if longest_row is not None else None,
        longest_sample_id=_sample_id(longest_row) if longest_row is not None else None,
        no_speech_count=len(no_speech_sample_ids),
    )


def _source_types(*, all: bool, generated: bool, augmented: bool) -> tuple[str, ...]:
    if all or not (generated or augmented):
        return ("generated", "augmented")
    if generated and augmented:
        return ("generated", "augmented")
    if generated:
        return ("generated",)
    return ("augmented",)


def _duration_ms(row: dict[str, object] | None) -> int:
    if row is None:
        return 0
    value = row.get("duration_ms")
    if not isinstance(value, int):
        raise ValueError(f"Missing integer duration_ms for sample_id={row.get('sample_id')}")
    return value


def _sample_id(row: dict[str, object] | None) -> str | None:
    if row is None:
        return None
    value = row.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Missing sample_id in Parquet row")
    return value


def _audio_bytes(row: dict[str, object]) -> bytes:
    value = row.get("audio_bytes")
    if not isinstance(value, bytes):
        raise ValueError(f"Missing audio_bytes for sample_id={row.get('sample_id')}")
    return value


def _resolve_project_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _format_duration(duration_ms: float) -> str:
    if duration_ms == int(duration_ms):
        return str(int(duration_ms))
    return f"{duration_ms:.1f}"
