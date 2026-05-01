from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wakewords.audio import speech_rms_dbfs
from wakewords.parquet_store import _SCHEMA

logger = logging.getLogger(__name__)


def backfill_speech_rms_dbfs(*, parquet_path: Path, overwrite: bool = False) -> int:
    parquet_file = pq.ParquetFile(parquet_path)
    generated_rms_by_sample_id: dict[str, float] = {}
    generated_sample_ids_needing_rms: set[str] = set()
    for batch in parquet_file.iter_batches(
        batch_size=2048,
        columns=["sample_id", "source_type", "speech_rms_dbfs"],
    ):
        sample_ids = batch.column("sample_id").to_pylist()
        source_types = batch.column("source_type").to_pylist()
        speech_rms_values = batch.column("speech_rms_dbfs").to_pylist()
        for sample_id, source_type, rms_dbfs in zip(sample_ids, source_types, speech_rms_values, strict=True):
            if source_type != "generated" or not isinstance(sample_id, str):
                continue
            if isinstance(rms_dbfs, float) and not overwrite:
                generated_rms_by_sample_id[sample_id] = rms_dbfs
            else:
                generated_sample_ids_needing_rms.add(sample_id)

    if generated_sample_ids_needing_rms:
        _compute_generated_speech_rms_values(
            parquet_file=parquet_file,
            sample_ids_needing_rms=generated_sample_ids_needing_rms,
            generated_rms_by_sample_id=generated_rms_by_sample_id,
        )

    temp_path = parquet_path.with_name(f".{parquet_path.name}.speech-rms.tmp")
    updated_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(temp_path, _SCHEMA)
        for batch in parquet_file.iter_batches(batch_size=512):
            table = pa.Table.from_batches([batch]).cast(_SCHEMA)
            speech_rms_values, changed = _updated_speech_rms_values(
                table=table,
                generated_rms_by_sample_id=generated_rms_by_sample_id,
                overwrite=overwrite,
            )
            if changed:
                table = table.set_column(
                    _SCHEMA.get_field_index("speech_rms_dbfs"),
                    _SCHEMA.field("speech_rms_dbfs"),
                    pa.array(speech_rms_values, type=pa.float64()),
                )
                updated_rows += changed
            writer.write_table(table)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    if updated_rows == 0:
        temp_path.unlink(missing_ok=True)
        return 0

    temp_path.replace(parquet_path)
    return updated_rows


def _updated_speech_rms_values(
    *,
    table: pa.Table,
    generated_rms_by_sample_id: dict[str, float],
    overwrite: bool,
) -> tuple[list[float | None], int]:
    sample_ids = table.column("sample_id").to_pylist()
    source_types = table.column("source_type").to_pylist()
    parent_sample_ids = table.column("parent_sample_id").to_pylist()
    speech_rms_values = table.column("speech_rms_dbfs").to_pylist()

    changed = 0
    for index, source_type in enumerate(source_types):
        existing_rms = speech_rms_values[index]
        if source_type == "generated":
            if isinstance(existing_rms, float) and not overwrite:
                continue
            sample_id = sample_ids[index]
            rms_dbfs = generated_rms_by_sample_id.get(sample_id) if isinstance(sample_id, str) else None
            if rms_dbfs is None:
                continue
            if existing_rms != rms_dbfs:
                speech_rms_values[index] = rms_dbfs
                changed += 1
            continue

        if source_type != "augmented":
            continue
        parent_sample_id = parent_sample_ids[index]
        if not isinstance(parent_sample_id, str):
            continue
        parent_rms = generated_rms_by_sample_id.get(parent_sample_id)
        if parent_rms is None or existing_rms == parent_rms:
            continue
        speech_rms_values[index] = parent_rms
        changed += 1

    return speech_rms_values, changed


def _compute_generated_speech_rms_values(
    *,
    parquet_file: pq.ParquetFile,
    sample_ids_needing_rms: set[str],
    generated_rms_by_sample_id: dict[str, float],
) -> None:
    for batch in parquet_file.iter_batches(
        batch_size=256,
        columns=["sample_id", "source_type", "audio_bytes"],
    ):
        sample_ids = batch.column("sample_id").to_pylist()
        source_types = batch.column("source_type").to_pylist()
        audio_bytes_values = batch.column("audio_bytes").to_pylist()
        for sample_id, source_type, audio_bytes in zip(sample_ids, source_types, audio_bytes_values, strict=True):
            if source_type != "generated" or sample_id not in sample_ids_needing_rms or not isinstance(audio_bytes, bytes):
                continue
            rms_dbfs = _compute_speech_rms_dbfs(audio_bytes=audio_bytes, sample_id=sample_id)
            if rms_dbfs is not None:
                generated_rms_by_sample_id[sample_id] = rms_dbfs


def _compute_speech_rms_dbfs(*, audio_bytes: bytes, sample_id: str) -> float | None:
    try:
        return speech_rms_dbfs(audio_bytes)
    except Exception as exc:
        logger.warning("Could not compute speech_rms_dbfs for generated sample %s: %s", sample_id, exc)
        return None
