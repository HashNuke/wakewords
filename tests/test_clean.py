from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wakewords.clean import clean_dataset


class CleanTests(unittest.TestCase):
    def test_clean_augmented_removes_only_augmented_files_and_stale_split_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            word_dir = project_dir / "data" / "yes"
            word_dir.mkdir(parents=True)
            clean_path = word_dir / "yes-cr1-t100-clean-nonoise-nosnr.wav"
            augmented_path = word_dir / "yes-cr1-t095-rain-rain-snr10.wav"
            clean_path.write_bytes(b"clean")
            augmented_path.write_bytes(b"augmented")
            _write_word_manifest(word_dir, [clean_path.name, augmented_path.name])
            split_manifest = project_dir / "train_manifest.jsonl"
            split_manifest.write_text("{}\n", encoding="utf-8")

            deleted = clean_dataset(data_dir=project_dir / "data", augmented=True)

            self.assertEqual(deleted, [augmented_path, split_manifest])
            self.assertTrue(clean_path.exists())
            self.assertFalse(augmented_path.exists())
            self.assertFalse(split_manifest.exists())
            entries = _read_word_manifest(word_dir)
            self.assertEqual([entry["audio_filepath"] for entry in entries], [clean_path.name])

    def test_clean_generated_removes_only_clean_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            word_dir = project_dir / "data" / "yes"
            word_dir.mkdir(parents=True)
            clean_path = word_dir / "yes-cr1-t100-clean-nonoise-nosnr.wav"
            augmented_path = word_dir / "yes-cr1-t095-rain-rain-snr10.wav"
            clean_path.write_bytes(b"clean")
            augmented_path.write_bytes(b"augmented")
            _write_word_manifest(word_dir, [clean_path.name, augmented_path.name])

            deleted = clean_dataset(data_dir=project_dir / "data", generated=True)

            self.assertEqual(deleted, [clean_path])
            self.assertFalse(clean_path.exists())
            self.assertTrue(augmented_path.exists())
            entries = _read_word_manifest(word_dir)
            self.assertEqual([entry["audio_filepath"] for entry in entries], [augmented_path.name])

    def test_clean_all_removes_all_wavs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            word_dir = project_dir / "data" / "yes"
            word_dir.mkdir(parents=True)
            clean_path = word_dir / "yes-cr1-t100-clean-nonoise-nosnr.wav"
            augmented_path = word_dir / "yes-cr1-t095-rain-rain-snr10.wav"
            clean_path.write_bytes(b"clean")
            augmented_path.write_bytes(b"augmented")
            _write_word_manifest(word_dir, [clean_path.name, augmented_path.name])

            deleted = clean_dataset(data_dir=project_dir / "data", all=True)

            self.assertEqual(set(deleted), {clean_path, augmented_path})
            self.assertFalse(clean_path.exists())
            self.assertFalse(augmented_path.exists())
            self.assertFalse((word_dir / "manifest.jsonl").exists())

    def test_clean_requires_one_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                clean_dataset(data_dir=Path(tmp_dir), generated=True, all=True)
            with self.assertRaises(ValueError):
                clean_dataset(data_dir=Path(tmp_dir))


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


if __name__ == "__main__":
    unittest.main()
