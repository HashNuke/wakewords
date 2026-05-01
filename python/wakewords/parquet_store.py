from __future__ import annotations

import hashlib
import struct
import tempfile
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
        pa.field("donor_sample_id", pa.string()),
        pa.field("donor_offset_ms", pa.int64()),
        pa.field("donor_duration_ms", pa.int64()),
        pa.field("context_position", pa.string()),
        pa.field("context_gap_ms", pa.int64()),
        pa.field("created_at", pa.string()),
        pa.field("sha256", pa.string()),
        pa.field("speech_rms_dbfs", pa.float64()),
    ]
)


class CustomWordStore:
    def __init__(self, path: Path = DEFAULT_PARQUET_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._rows: dict[str, dict[str, object]] = {}
        self._augmented_rows: dict[tuple[str, float, str | None, int | None], str] = {}
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
        donor_sample_id: str | None = None,
        donor_offset_ms: int | None = None,
        donor_duration_ms: int | None = None,
        context_position: str | None = None,
        context_gap_ms: int | None = None,
    ) -> dict[str, object] | None:
        with self._lock:
            sample_id = self._augmented_rows.get(
                (
                    parent_sample_id,
                    tempo,
                    noise_type,
                    snr,
                    donor_sample_id,
                    donor_offset_ms,
                    donor_duration_ms,
                    context_position,
                    context_gap_ms,
                )
            )
            if sample_id is None:
                return None
            row = self._rows.get(sample_id)
            if row is None:
                return None
            return dict(row)

    def upsert(self, row: dict[str, object], *, overwrite: bool) -> bool:
        with self._lock:
            if not self._upsert_locked(row, overwrite=overwrite):
                return False
            self._write()
            return True

    def upsert_many(self, rows: list[dict[str, object]], *, overwrite: bool) -> int:
        with self._lock:
            changed = 0
            for row in rows:
                if self._upsert_locked(row, overwrite=overwrite):
                    changed += 1
            if changed == 0:
                return 0
            self._write()
            return changed

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
            self._rebuild_augmented_rows()
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
            self._track_augmented_row(normalized)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        table = pa.Table.from_pylist(rows, schema=_SCHEMA)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                suffix=".parquet.tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            pq.write_table(table, temp_path)
            temp_path.replace(self.path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _track_augmented_row(self, row: dict[str, object]) -> None:
        if row.get("source_type") != "augmented":
            return
        parent_sample_id = row.get("parent_sample_id")
        tempo = row.get("tempo")
        noise_type = row.get("noise_type")
        snr = row.get("snr")
        if not isinstance(parent_sample_id, str) or not isinstance(tempo, float):
            return
        donor_sample_id = row.get("donor_sample_id")
        donor_offset_ms = row.get("donor_offset_ms")
        donor_duration_ms = row.get("donor_duration_ms")
        context_position = row.get("context_position")
        context_gap_ms = row.get("context_gap_ms")
        self._augmented_rows[
            (
                parent_sample_id,
                tempo,
                noise_type if isinstance(noise_type, str) else None,
                snr if isinstance(snr, int) else None,
                donor_sample_id if isinstance(donor_sample_id, str) else None,
                donor_offset_ms if isinstance(donor_offset_ms, int) else None,
                donor_duration_ms if isinstance(donor_duration_ms, int) else None,
                context_position if isinstance(context_position, str) else None,
                context_gap_ms if isinstance(context_gap_ms, int) else None,
            )
        ] = _require_str(row, "sample_id")

    def _rebuild_augmented_rows(self) -> None:
        self._augmented_rows = {}
        for row in self._rows.values():
            self._track_augmented_row(row)

    def _upsert_locked(self, row: dict[str, object], *, overwrite: bool) -> bool:
        sample_id = _require_str(row, "sample_id")
        normalized = _normalize_row(row)
        existing = self._rows.get(sample_id)
        if existing is not None and not overwrite:
            return False
        self._rows[sample_id] = normalized
        if existing is not None:
            self._rebuild_augmented_rows()
        else:
            self._track_augmented_row(normalized)
        return True


def build_generated_row(
    *,
    audio_bytes: bytes,
    label: str,
    voice_id: str,
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
        "donor_sample_id": None,
        "donor_offset_ms": None,
        "donor_duration_ms": None,
        "context_position": None,
        "context_gap_ms": None,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": sha256,
        "speech_rms_dbfs": None,
    }


def build_augmented_row(
    *,
    audio_bytes: bytes,
    source_row: dict[str, object],
    tempo: float,
    noise_type: str | None,
    snr: int | None,
    donor_sample_id: str | None = None,
    donor_offset_ms: int | None = None,
    donor_duration_ms: int | None = None,
    context_position: str | None = None,
    context_gap_ms: int | None = None,
) -> dict[str, object]:
    sample_rate, channels, duration_ms = probe_wav_bytes(audio_bytes)
    sha256 = hashlib.sha256(audio_bytes).hexdigest()
    label = _require_str(source_row, "label")
    parent_sample_id = _require_str(source_row, "sample_id")
    sample_id = hashlib.sha256(
        f"augmented\0{parent_sample_id}\0{tempo}\0{noise_type}\0{snr}\0{donor_sample_id}\0{donor_offset_ms}\0{donor_duration_ms}\0{context_position}\0{context_gap_ms}".encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "sample_id": sample_id,
        "label": label,
        "audio_bytes": audio_bytes,
        "voice_id": _require_str(source_row, "voice_id"),
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
        "donor_sample_id": donor_sample_id,
        "donor_offset_ms": donor_offset_ms,
        "donor_duration_ms": donor_duration_ms,
        "context_position": context_position,
        "context_gap_ms": context_gap_ms,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": sha256,
        "speech_rms_dbfs": None,
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
