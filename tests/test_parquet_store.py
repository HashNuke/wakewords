from __future__ import annotations

import io
import tempfile
import unittest
import wave
from pathlib import Path

from wakewords.parquet_store import CustomWordStore, build_generated_row


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
