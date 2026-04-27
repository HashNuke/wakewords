from __future__ import annotations

import hashlib
import io
import re
import threading
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised in runtime environments without pyarrow
    raise ImportError("PyArrow is required for Parquet-backed custom word storage. Install 'pyarrow'.") from exc


DEFAULT_PARQUET_PATH = Path("data/custom_words.parquet")

_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("label", pa.string()),
        pa.field("audio_bytes", pa.binary()),
        pa.field("voice_id", pa.string()),
        pa.field("voice_code", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("duration_ms", pa.int64()),
        pa.field("lang", pa.string()),
        pa.field("sample_rate", pa.int64()),
        pa.field("channels", pa.int64()),
        pa.field("source_type", pa.string()),
        pa.field("parent_sample_id", pa.string()),
        pa.field("tempo", pa.float64()),
        pa.field("noise_type", pa.string()),
        pa.field("snr", pa.int64()),
        pa.field("created_at", pa.string()),
        pa.field("sha256", pa.string()),
    ]
)


class CustomWordStore:
    def __init__(self, path: Path = DEFAULT_PARQUET_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._rows: dict[str, dict[str, object]] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._voice_codes: dict[tuple[str, str], str] = {}
        self._provider_counters: dict[str, int] = {}
        self._load()

    def contains(self, *, label: str, filename: str) -> bool:
        with self._lock:
            return (label, filename) in self._keys

    def get(self, *, label: str, filename: str) -> dict[str, object] | None:
        with self._lock:
            sample_id = self._keys.get((label, filename))
            if sample_id is None:
                return None
            return dict(self._rows[sample_id])

    def voice_code(self, *, provider: str, voice_id: str) -> str:
        key = (provider, voice_id)
        with self._lock:
            code = self._voice_codes.get(key)
            if code is not None:
                return code
            idx = self._provider_counters.get(provider, 0) + 1
            code = f"{provider}{idx}"
            self._voice_codes[key] = code
            self._provider_counters[provider] = idx
            return code

    def upsert(self, row: dict[str, object], *, overwrite: bool) -> bool:
        key = _row_key(row)
        sample_id = _require_str(row, "sample_id")
        normalized = _normalize_row(row)
        with self._lock:
            existing_id = self._keys.get(key)
            if existing_id is not None and not overwrite:
                return False
            if existing_id is not None:
                self._rows.pop(existing_id, None)
            self._rows[sample_id] = normalized
            self._keys[key] = sample_id
            self._track_voice_code(normalized)
            self._write()
            return True

    def delete_matching(self, predicate: "Callable[[dict[str, object]], bool]") -> list[dict[str, object]]:
        removed: list[dict[str, object]] = []
        with self._lock:
            remaining_rows: dict[str, dict[str, object]] = {}
            remaining_keys: dict[tuple[str, str], str] = {}
            for sample_id, row in self._rows.items():
                if predicate(row):
                    removed.append(dict(row))
                    continue
                remaining_rows[sample_id] = row
                remaining_keys[_row_key(row)] = sample_id
            if not removed:
                return []
            self._rows = remaining_rows
            self._keys = remaining_keys
            self._rebuild_voice_codes()
            self._write()
            return removed

    def rows(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(row) for row in sorted(self._rows.values(), key=lambda row: (_require_str(row, "label"), _require_str(row, "filename")))]

    def _load(self) -> None:
        if not self.path.exists():
            return
        table = pq.read_table(self.path, schema=_SCHEMA)
        for row in table.to_pylist():
            normalized = _normalize_row(row)
            sample_id = _require_str(normalized, "sample_id")
            self._rows[sample_id] = normalized
            self._keys[_row_key(normalized)] = sample_id
            self._track_voice_code(normalized)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        table = pa.Table.from_pylist(rows, schema=_SCHEMA)
        pq.write_table(table, self.path)

    def _track_voice_code(self, row: dict[str, object]) -> None:
        provider = _require_str(row, "provider")
        voice_id = _require_str(row, "voice_id")
        voice_code = _require_str(row, "voice_code")
        self._voice_codes[(provider, voice_id)] = voice_code
        match = re.fullmatch(rf"{re.escape(provider)}(\d+)", voice_code)
        if match is not None:
            self._provider_counters[provider] = max(self._provider_counters.get(provider, 0), int(match.group(1)))

    def _rebuild_voice_codes(self) -> None:
        self._voice_codes = {}
        self._provider_counters = {}
        for row in self._rows.values():
            self._track_voice_code(row)


def build_generated_row(
    *,
    audio_bytes: bytes,
    filename: str,
    label: str,
    voice_id: str,
    voice_code: str,
    provider: str,
    lang: str | None,
) -> dict[str, object]:
    sample_rate, channels, duration_ms = probe_wav_bytes(audio_bytes)
    sha256 = hashlib.sha256(audio_bytes).hexdigest()
    sample_id = hashlib.sha256(f"{label}\0{filename}\0{sha256}".encode("utf-8")).hexdigest()
    return {
        "sample_id": sample_id,
        "filename": filename,
        "label": label,
        "audio_bytes": audio_bytes,
        "voice_id": voice_id,
        "voice_code": voice_code,
        "provider": provider,
        "duration_ms": duration_ms,
        "lang": lang,
        "sample_rate": sample_rate,
        "channels": channels,
        "source_type": "generated",
        "parent_sample_id": None,
        "tempo": None,
        "noise_type": None,
        "snr": None,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": sha256,
    }


def build_augmented_row(
    *,
    audio_bytes: bytes,
    filename: str,
    source_row: dict[str, object],
    tempo: float,
    noise_type: str | None,
    snr: int | None,
) -> dict[str, object]:
    sample_rate, channels, duration_ms = probe_wav_bytes(audio_bytes)
    sha256 = hashlib.sha256(audio_bytes).hexdigest()
    label = _require_str(source_row, "label")
    parent_sample_id = _require_str(source_row, "sample_id")
    sample_id = hashlib.sha256(f"{parent_sample_id}\0{filename}\0{sha256}".encode("utf-8")).hexdigest()
    return {
        "sample_id": sample_id,
        "filename": filename,
        "label": label,
        "audio_bytes": audio_bytes,
        "voice_id": _require_str(source_row, "voice_id"),
        "voice_code": _require_str(source_row, "voice_code"),
        "provider": _require_str(source_row, "provider"),
        "duration_ms": duration_ms,
        "lang": source_row.get("lang"),
        "sample_rate": sample_rate,
        "channels": channels,
        "source_type": "augmented",
        "parent_sample_id": parent_sample_id,
        "tempo": tempo,
        "noise_type": noise_type,
        "snr": snr,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": sha256,
    }


def probe_wav_bytes(audio_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
    duration_ms = round(frame_count / sample_rate * 1000)
    return sample_rate, channels, duration_ms


def _normalize_row(row: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for field in _SCHEMA.names:
        value = row.get(field)
        if isinstance(value, memoryview):
            value = value.tobytes()
        normalized[field] = value
    return normalized


def _row_key(row: dict[str, object]) -> tuple[str, str]:
    return (_require_str(row, "label"), _require_str(row, "filename"))


def _require_str(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string field '{field}' in Parquet row")
    return value
