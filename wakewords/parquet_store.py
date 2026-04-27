from __future__ import annotations

import hashlib
import re
import struct
import threading
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
        self._voice_codes: dict[tuple[str, str], str] = {}
        self._provider_counters: dict[str, int] = {}
        self._load()

    def get_by_sample_id(self, sample_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._rows.get(sample_id)
            if row is None:
                return None
            return dict(row)

    def find_augmented(
        self,
        *,
        parent_sample_id: str,
        tempo: float,
        noise_type: str | None,
        snr: int | None,
    ) -> dict[str, object] | None:
        with self._lock:
            for row in self._rows.values():
                if row.get("source_type") != "augmented":
                    continue
                if row.get("parent_sample_id") != parent_sample_id:
                    continue
                if row.get("tempo") == tempo and row.get("noise_type") == noise_type and row.get("snr") == snr:
                    return dict(row)
            return None

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
        sample_id = _require_str(row, "sample_id")
        normalized = _normalize_row(row)
        with self._lock:
            if sample_id in self._rows and not overwrite:
                return False
            self._rows[sample_id] = normalized
            self._track_voice_code(normalized)
            self._write()
            return True

    def delete_matching(self, predicate: "Callable[[dict[str, object]], bool]") -> list[dict[str, object]]:
        removed: list[dict[str, object]] = []
        with self._lock:
            remaining_rows: dict[str, dict[str, object]] = {}
            for sample_id, row in self._rows.items():
                if predicate(row):
                    removed.append(dict(row))
                    continue
                remaining_rows[sample_id] = row
            if not removed:
                return []
            self._rows = remaining_rows
            self._rebuild_voice_codes()
            self._write()
            return removed

    def rows(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(row) for row in sorted(self._rows.values(), key=lambda row: (_require_str(row, "label"), _require_str(row, "sample_id")))]

    def _load(self) -> None:
        if not self.path.exists():
            return
        table = pq.read_table(self.path, schema=_SCHEMA)
        for row in table.to_pylist():
            normalized = _normalize_row(row)
            sample_id = _require_str(normalized, "sample_id")
            self._rows[sample_id] = normalized
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
    label: str,
    voice_id: str,
    voice_code: str,
    provider: str,
    lang: str | None,
) -> dict[str, object]:
    sample_rate, channels, duration_ms = probe_wav_bytes(audio_bytes)
    sha256 = hashlib.sha256(audio_bytes).hexdigest()
    sample_id = hashlib.sha256(f"generated\0{label}\0{provider}\0{voice_id}".encode("utf-8")).hexdigest()
    return {
        "sample_id": sample_id,
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
    source_row: dict[str, object],
    tempo: float,
    noise_type: str | None,
    snr: int | None,
) -> dict[str, object]:
    sample_rate, channels, duration_ms = probe_wav_bytes(audio_bytes)
    sha256 = hashlib.sha256(audio_bytes).hexdigest()
    label = _require_str(source_row, "label")
    parent_sample_id = _require_str(source_row, "sample_id")
    sample_id = hashlib.sha256(f"augmented\0{parent_sample_id}\0{tempo}\0{noise_type}\0{snr}".encode("utf-8")).hexdigest()
    return {
        "sample_id": sample_id,
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
    sample_rate, channels, byte_rate, data_start, declared_data_size = _parse_wav_chunks(audio_bytes)
    available_data_size = max(len(audio_bytes) - data_start, 0)
    if declared_data_size == 0xFFFFFFFF or data_start + declared_data_size > len(audio_bytes):
        data_size = available_data_size
    else:
        data_size = declared_data_size
    duration_ms = round(data_size / byte_rate * 1000)
    return sample_rate, channels, duration_ms


def _parse_wav_chunks(audio_bytes: bytes) -> tuple[int, int, int, int, int]:
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE audio bytes")

    sample_rate: int | None = None
    channels: int | None = None
    byte_rate: int | None = None
    data_start: int | None = None
    data_size: int | None = None
    offset = 12

    while offset + 8 <= len(audio_bytes):
        chunk_id = audio_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio_bytes, offset + 4)[0]
        payload_start = offset + 8

        if chunk_id == b"fmt ":
            if payload_start + 16 > len(audio_bytes):
                raise ValueError("WAV fmt chunk is truncated")
            channels = struct.unpack_from("<H", audio_bytes, payload_start + 2)[0]
            sample_rate = struct.unpack_from("<I", audio_bytes, payload_start + 4)[0]
            byte_rate = struct.unpack_from("<I", audio_bytes, payload_start + 8)[0]
            if channels < 1 or sample_rate < 1 or byte_rate < 1:
                raise ValueError("WAV fmt chunk has invalid audio parameters")
        elif chunk_id == b"data":
            data_start = payload_start
            data_size = chunk_size
            break

        next_offset = payload_start + chunk_size + (chunk_size % 2)
        if next_offset <= offset:
            raise ValueError("WAV chunk parser did not advance")
        offset = next_offset

    if sample_rate is None or channels is None or byte_rate is None:
        raise ValueError("WAV bytes are missing a fmt chunk")
    if data_start is None or data_size is None:
        raise ValueError("WAV bytes are missing a data chunk")
    return sample_rate, channels, byte_rate, data_start, data_size


def _normalize_row(row: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for field in _SCHEMA.names:
        value = row.get(field)
        if isinstance(value, memoryview):
            value = value.tobytes()
        normalized[field] = value
    return normalized


def _require_str(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string field '{field}' in Parquet row")
    return value
