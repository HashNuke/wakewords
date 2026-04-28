from __future__ import annotations

import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from wakewords.parquet_store import CustomWordStore, build_augmented_row, build_generated_row, probe_wav_bytes


class CustomWordStoreTests(unittest.TestCase):
    def test_upsert_writes_and_reads_generated_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            row = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="dexa",
                voice_id="voice-123",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )

            changed = store.upsert(row, overwrite=False)

            self.assertTrue(changed)
            reloaded = CustomWordStore(store.path)
            rows = reloaded.rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "dexa")
            self.assertEqual(rows[0]["voice_id"], "voice-123")
            self.assertEqual(rows[0]["provider"], "cr")
            self.assertEqual(rows[0]["duration_ms"], 250)

    def test_upsert_skips_duplicate_sample_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            first = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="dexa",
                voice_id="voice-123",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            second = build_generated_row(
                audio_bytes=_wav_bytes(duration_ms=500),
                label="dexa",
                voice_id="voice-123",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )

            self.assertTrue(store.upsert(first, overwrite=False))
            self.assertFalse(store.upsert(second, overwrite=False))

            rows = CustomWordStore(store.path).rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["duration_ms"], 250)

    def test_voice_code_reuses_existing_provider_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            store.upsert(
                build_generated_row(
                    audio_bytes=_wav_bytes(),
                    label="dexa",
                    voice_id="voice-123",
                    voice_code="cr1",
                    provider="cr",
                    lang="en",
                ),
                overwrite=False,
            )

            self.assertEqual(store.voice_code(provider="cr", voice_id="voice-123"), "cr1")
            self.assertEqual(store.voice_code(provider="cr", voice_id="voice-456"), "cr2")

    def test_upsert_many_writes_once_for_multiple_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            rows = [
                build_generated_row(
                    audio_bytes=_wav_bytes(),
                    label="dexa",
                    voice_id="voice-123",
                    voice_code="cr1",
                    provider="cr",
                    lang="en",
                ),
                build_generated_row(
                    audio_bytes=_wav_bytes(duration_ms=500),
                    label="alexa",
                    voice_id="voice-456",
                    voice_code="cr2",
                    provider="cr",
                    lang="en",
                ),
            ]

            with unittest.mock.patch.object(store, "_write", wraps=store._write) as write_mock:
                changed = store.upsert_many(rows, overwrite=False)

            self.assertEqual(changed, 2)
            self.assertEqual(write_mock.call_count, 1)
            self.assertEqual(len(CustomWordStore(store.path).rows()), 2)

    def test_find_augmented_returns_indexed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            source = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="dexa",
                voice_id="voice-123",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            augmented = build_augmented_row(
                audio_bytes=_wav_bytes(duration_ms=500),
                source_row=source,
                tempo=1.05,
                noise_type="rain",
                snr=10,
            )

            store.upsert(source, overwrite=False)
            store.upsert(augmented, overwrite=False)

            found = store.find_augmented(
                parent_sample_id=source["sample_id"],
                tempo=1.05,
                noise_type="rain",
                snr=10,
            )

            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found["sample_id"], augmented["sample_id"])

    def test_find_augmented_updates_after_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            source = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="dexa",
                voice_id="voice-123",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            augmented = build_augmented_row(
                audio_bytes=_wav_bytes(duration_ms=500),
                source_row=source,
                tempo=1.05,
                noise_type="rain",
                snr=10,
            )

            store.upsert(source, overwrite=False)
            store.upsert(augmented, overwrite=False)
            store.delete_matching(lambda row: row.get("sample_id") == augmented["sample_id"])

            found = store.find_augmented(
                parent_sample_id=source["sample_id"],
                tempo=1.05,
                noise_type="rain",
                snr=10,
            )

            self.assertIsNone(found)

    def test_probe_wav_bytes_uses_actual_data_payload_for_placeholder_sizes(self) -> None:
        wav_bytes = _wav_bytes(duration_ms=250)
        data_offset = wav_bytes.index(b"data")
        riff_size_offset = 4
        data_size_offset = data_offset + 4
        wav_bytes = (
            wav_bytes[:riff_size_offset]
            + struct.pack("<I", 0xFFFFFFFF)
            + wav_bytes[riff_size_offset + 4 : data_size_offset]
            + struct.pack("<I", 0xFFFFFFFF)
            + wav_bytes[data_size_offset + 4 :]
        )

        sample_rate, channels, duration_ms = probe_wav_bytes(wav_bytes)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 1)
        self.assertEqual(duration_ms, 250)

    def test_probe_wav_bytes_uses_actual_data_payload_when_declared_size_exceeds_file(self) -> None:
        wav_bytes = _wav_bytes(duration_ms=250)
        data_size_offset = wav_bytes.index(b"data") + 4
        wav_bytes = wav_bytes[:data_size_offset] + struct.pack("<I", 999_999) + wav_bytes[data_size_offset + 4 :]

        sample_rate, channels, duration_ms = probe_wav_bytes(wav_bytes)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 1)
        self.assertEqual(duration_ms, 250)


def _wav_bytes(*, duration_ms: int = 250) -> bytes:
    sample_rate = 16000
    frame_count = sample_rate * duration_ms // 1000
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
