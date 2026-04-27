from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakewords import cli
from wakewords.check import check_dataset
from wakewords.parquet_store import CustomWordStore, build_augmented_row, build_generated_row


class CheckTests(unittest.TestCase):
    def test_check_dataset_prints_stats_and_writes_no_speech_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            store = CustomWordStore(data_dir / "custom_words.parquet")
            generated = build_generated_row(
                audio_bytes=_wav_bytes(duration_ms=500),
                label="dexa",
                voice_id="voice-1",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            augmented = build_augmented_row(
                audio_bytes=_wav_bytes(duration_ms=1500),
                source_row=generated,
                tempo=1.0,
                noise_type="white",
                snr=10,
            )
            store.upsert(generated, overwrite=False)
            store.upsert(augmented, overwrite=False)

            with mock.patch("wakewords.check.wav_has_speech", side_effect=lambda audio_bytes: audio_bytes == generated["audio_bytes"]):
                stats = check_dataset(
                    project_dir=project_dir,
                    data_dir=Path("data"),
                    all=False,
                    generated=False,
                    augmented=False,
                )

            self.assertEqual(stats.sample_count, 2)
            self.assertEqual(stats.median_duration_ms, 1000)
            self.assertEqual(stats.longest_duration_ms, 1500)
            self.assertEqual(stats.longest_sample_id, augmented["sample_id"])
            self.assertEqual(stats.no_speech_count, 1)
            self.assertEqual((project_dir / "no-speech.txt").read_text(encoding="utf-8"), f"{augmented['sample_id']}\n")

    def test_check_dataset_filters_generated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            store = CustomWordStore(data_dir / "custom_words.parquet")
            generated = build_generated_row(
                audio_bytes=_wav_bytes(duration_ms=500),
                label="dexa",
                voice_id="voice-1",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            augmented = build_augmented_row(
                audio_bytes=_wav_bytes(duration_ms=1500),
                source_row=generated,
                tempo=1.0,
                noise_type="white",
                snr=10,
            )
            store.upsert(generated, overwrite=False)
            store.upsert(augmented, overwrite=False)

            with mock.patch("wakewords.check.wav_has_speech", return_value=True) as has_speech:
                stats = check_dataset(
                    project_dir=project_dir,
                    data_dir=Path("data"),
                    all=False,
                    generated=True,
                    augmented=False,
                )

            self.assertEqual(stats.source_types, ("generated",))
            self.assertEqual(stats.sample_count, 1)
            self.assertEqual(stats.longest_sample_id, generated["sample_id"])
            has_speech.assert_called_once()

    def test_cli_check_prints_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir).resolve()
            data_dir = project_dir / "data"
            store = CustomWordStore(data_dir / "custom_words.parquet")
            row = build_generated_row(
                audio_bytes=_wav_bytes(duration_ms=500),
                label="dexa",
                voice_id="voice-1",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            store.upsert(row, overwrite=False)
            stdout = io.StringIO()
            argv = ["wakewords", "check", "--project-dir", tmp_dir, "--generated"]

            with mock.patch.object(cli.sys, "argv", argv):
                with mock.patch("wakewords.check.wav_has_speech", return_value=True):
                    with redirect_stdout(stdout):
                        cli.main()

            lines = stdout.getvalue().strip().splitlines()
            self.assertIn(f"parquet: {data_dir / 'custom_words.parquet'}", lines)
            self.assertIn("sources: generated", lines)
            self.assertIn("samples: 1", lines)
            self.assertIn("median_duration_ms: 500", lines)
            self.assertIn(f"longest_sample_id: {row['sample_id']}", lines)
            self.assertIn("no_speech_count: 0", lines)
            self.assertIn(f"no_speech_file: {project_dir / 'no-speech.txt'}", lines)


def _wav_bytes(*, duration_ms: int) -> bytes:
    import wave

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
