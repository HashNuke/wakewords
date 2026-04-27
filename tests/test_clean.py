from __future__ import annotations

import json
import io
import tempfile
import unittest
import wave
from pathlib import Path

from wakewords.clean import clean_dataset
from wakewords.parquet_store import CustomWordStore, build_augmented_row, build_generated_row


class CleanTests(unittest.TestCase):
    def test_clean_ignores_legacy_word_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            word_dir = project_dir / "data" / "yes"
            word_dir.mkdir(parents=True)
            clean_path = word_dir / "yes-cr1-t100-clean-nonoise-nosnr.wav"
            augmented_path = word_dir / "yes-cr1-t095-rain-snr10.wav"
            clean_path.write_bytes(b"clean")
            augmented_path.write_bytes(b"augmented")
            _write_word_manifest(word_dir, [clean_path.name, augmented_path.name])
            split_manifest = project_dir / "train_manifest.jsonl"
            split_manifest.write_text("{}\n", encoding="utf-8")

            deleted = clean_dataset(data_dir=project_dir / "data", augmented=True)

            self.assertEqual(deleted, [])
            self.assertTrue(clean_path.exists())
            self.assertTrue(augmented_path.exists())
            self.assertTrue(split_manifest.exists())
            entries = _read_word_manifest(word_dir)
            self.assertEqual([entry["audio_filepath"] for entry in entries], [clean_path.name, augmented_path.name])

    def test_clean_requires_one_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                clean_dataset(data_dir=Path(tmp_dir), generated=True, all=True)
            with self.assertRaises(ValueError):
                clean_dataset(data_dir=Path(tmp_dir))

    def test_clean_generated_removes_matching_parquet_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            word_dir = data_dir / "yes"
            word_dir.mkdir(parents=True)
            clean_path = word_dir / "yes-cr1-t100-clean-nonoise-nosnr.wav"
            clean_path.write_bytes(b"clean")
            _write_word_manifest(word_dir, [clean_path.name])

            store = CustomWordStore(data_dir / "custom_words.parquet")
            generated_row = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="yes",
                voice_id="voice-1",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            store.upsert(generated_row, overwrite=False)
            materialized_dir = data_dir / "custom-words" / "yes"
            materialized_dir.mkdir(parents=True)
            materialized_path = materialized_dir / f"{generated_row['sample_id']}.wav"
            materialized_path.write_bytes(b"materialized")

            deleted = clean_dataset(data_dir=data_dir, generated=True)

            self.assertEqual(deleted, [data_dir / "custom_words.parquet", materialized_path])
            self.assertEqual(CustomWordStore(data_dir / "custom_words.parquet").rows(), [])
            self.assertTrue(clean_path.exists())

    def test_clean_augmented_uses_parquet_source_type_not_filename_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            word_dir = data_dir / "yes"
            word_dir.mkdir(parents=True)
            generated_path = word_dir / "yes-cr1-t100-clean-nonoise-nosnr.wav"
            augmented_path = word_dir / "yes-cr1-t100-rain-snr10.wav"
            generated_path.write_bytes(b"generated")
            augmented_path.write_bytes(b"augmented")
            _write_word_manifest(word_dir, [generated_path.name, augmented_path.name])

            store = CustomWordStore(data_dir / "custom_words.parquet")
            generated_row = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="yes",
                voice_id="voice-1",
                voice_code="cr1",
                provider="cr",
                lang="en",
            )
            store.upsert(generated_row, overwrite=False)
            augmented_row = build_augmented_row(
                audio_bytes=_wav_bytes(),
                source_row=generated_row,
                tempo=1.0,
                noise_type="rain",
                snr=10,
            )
            store.upsert(augmented_row, overwrite=False)
            materialized_dir = data_dir / "custom-words" / "yes"
            materialized_dir.mkdir(parents=True)
            materialized_generated = materialized_dir / f"{generated_row['sample_id']}.wav"
            materialized_augmented = materialized_dir / f"{augmented_row['sample_id']}.wav"
            materialized_generated.write_bytes(b"generated")
            materialized_augmented.write_bytes(b"augmented")

            deleted = clean_dataset(data_dir=data_dir, augmented=True)

            self.assertEqual(
                deleted,
                [data_dir / "custom_words.parquet", materialized_augmented],
            )
            remaining_rows = CustomWordStore(data_dir / "custom_words.parquet").rows()
            self.assertEqual([row["source_type"] for row in remaining_rows], ["generated"])
            self.assertTrue(generated_path.exists())
            self.assertTrue(materialized_generated.exists())
            self.assertTrue(augmented_path.exists())
            self.assertFalse(materialized_augmented.exists())


def _write_word_manifest(word_dir: Path, filenames: list[str]) -> None:
    entries = [
        {"audio_filepath": filename, "duration": 1.0, "duration_ms": 1000, "label": word_dir.name}
        for filename in filenames
    ]
    (word_dir / "manifest.jsonl").write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _read_word_manifest(word_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (word_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _wav_bytes() -> bytes:
    sample_rate = 16000
    frame_count = sample_rate // 4
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
