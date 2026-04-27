from __future__ import annotations

import json
import io
import tempfile
import unittest
import wave
from pathlib import Path

from wakewords.dataset_manifest import build_split_manifests
from wakewords.parquet_store import CustomWordStore, build_generated_row


class DatasetManifestTests(unittest.TestCase):
    def test_build_split_manifests_materializes_custom_words_from_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            store = CustomWordStore(data_dir / "custom_words.parquet")
            store.upsert(
                build_generated_row(
                    audio_bytes=_wav_bytes(),
                    label="yes",
                    voice_id="voice-1",
                    voice_code="cr1",
                    provider="cr",
                    lang="en",
                ),
                overwrite=False,
            )

            outputs = build_split_manifests(
                data_dir=data_dir,
                train_ratio=1,
                validate_ratio=0,
                test_ratio=0,
            )

            row = store.rows()[0]
            sample_id = row["sample_id"]

            self.assertEqual(
                outputs,
                {
                    "train": data_dir / "manifests" / "train_manifest.jsonl",
                    "validate": data_dir / "manifests" / "validation_manifest.jsonl",
                    "test": data_dir / "manifests" / "test_manifest.jsonl",
                },
            )
            materialized = data_dir / "custom-words" / "yes" / f"{sample_id}.wav"
            self.assertTrue(materialized.is_file())
            entries = _read_jsonl(data_dir / "manifests" / "train_manifest.jsonl")
            self.assertEqual(entries[0]["audio_filepath"], str(materialized.resolve()))

    def test_build_split_manifests_includes_configured_google_words_except_background_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"google_speech_commands": ["yes", "no", "_background_noise_"]}) + "\n",
                encoding="utf-8",
            )
            data_dir = project_dir / "data"
            data_dir.mkdir()
            google_dir = project_dir / "google-speech-commands"
            yes_dir = google_dir / "yes"
            no_dir = google_dir / "no"
            up_dir = google_dir / "up"
            noise_dir = google_dir / "_background_noise_"
            yes_dir.mkdir(parents=True)
            no_dir.mkdir(parents=True)
            up_dir.mkdir(parents=True)
            noise_dir.mkdir(parents=True)
            _write_wav(yes_dir / "sample.wav")
            _write_wav(no_dir / "sample.wav")
            _write_wav(up_dir / "sample.wav")
            _write_wav(noise_dir / "noise.wav")

            build_split_manifests(
                data_dir=data_dir,
                train_ratio=1,
                validate_ratio=0,
                test_ratio=0,
            )

            entries = _read_jsonl(data_dir / "manifests" / "train_manifest.jsonl")
            self.assertEqual(sorted(entry["label"] for entry in entries), ["no", "yes"])
            self.assertEqual(
                sorted(entry["audio_filepath"] for entry in entries),
                sorted([str((yes_dir / "sample.wav").resolve()), str((no_dir / "sample.wav").resolve())]),
            )
            self.assertNotIn("up", "\n".join(str(entry["audio_filepath"]) for entry in entries))
            self.assertNotIn("_background_noise_", "\n".join(str(entry["audio_filepath"]) for entry in entries))

    def test_build_split_manifests_skips_google_when_config_list_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"google_speech_commands": []}) + "\n",
                encoding="utf-8",
            )
            data_dir = project_dir / "data"
            data_dir.mkdir()
            yes_dir = project_dir / "google-speech-commands" / "yes"
            yes_dir.mkdir(parents=True)
            _write_wav(yes_dir / "sample.wav")

            build_split_manifests(
                data_dir=data_dir,
                train_ratio=1,
                validate_ratio=0,
                test_ratio=0,
            )

            self.assertEqual(_read_jsonl(data_dir / "manifests" / "train_manifest.jsonl"), [])

    def test_build_split_manifests_splits_google_words_by_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"google_speech_commands": ["yes"]}) + "\n",
                encoding="utf-8",
            )
            data_dir = project_dir / "data"
            data_dir.mkdir()
            yes_dir = project_dir / "google-speech-commands" / "yes"
            yes_dir.mkdir(parents=True)
            for index in range(10):
                _write_wav(yes_dir / f"sample-{index}.wav")

            build_split_manifests(
                data_dir=data_dir,
                train_ratio=70,
                validate_ratio=20,
                test_ratio=10,
            )

            manifests_dir = data_dir / "manifests"
            self.assertEqual(len(_read_jsonl(manifests_dir / "train_manifest.jsonl")), 7)
            self.assertEqual(len(_read_jsonl(manifests_dir / "validation_manifest.jsonl")), 2)
            self.assertEqual(len(_read_jsonl(manifests_dir / "test_manifest.jsonl")), 1)
            self.assertEqual(_manifest_basenames(manifests_dir / "train_manifest.jsonl"), [f"sample-{index}.wav" for index in range(7)])
            self.assertEqual(_manifest_basenames(manifests_dir / "validation_manifest.jsonl"), ["sample-7.wav", "sample-8.wav"])
            self.assertEqual(_manifest_basenames(manifests_dir / "test_manifest.jsonl"), ["sample-9.wav"])

    def test_build_split_manifests_splits_google_words_independent_of_project_path(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp_dir:
            with tempfile.TemporaryDirectory() as second_tmp_dir:
                first_project_dir = Path(first_tmp_dir)
                second_project_dir = Path(second_tmp_dir)
                _write_google_project(first_project_dir, word="yes", sample_count=10)
                _write_google_project(second_project_dir, word="yes", sample_count=10)

                build_split_manifests(
                    data_dir=first_project_dir / "data",
                    train_ratio=70,
                    validate_ratio=20,
                    test_ratio=10,
                )
                build_split_manifests(
                    data_dir=second_project_dir / "data",
                    train_ratio=70,
                    validate_ratio=20,
                    test_ratio=10,
                )

                for manifest_name in ("train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl"):
                    self.assertEqual(
                        _manifest_basenames(first_project_dir / "data" / "manifests" / manifest_name),
                        _manifest_basenames(second_project_dir / "data" / "manifests" / manifest_name),
                    )


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


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\0\0" * 16_000)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_google_project(project_dir: Path, *, word: str, sample_count: int) -> None:
    (project_dir / "config.json").write_text(
        json.dumps({"google_speech_commands": [word]}) + "\n",
        encoding="utf-8",
    )
    data_dir = project_dir / "data"
    word_dir = project_dir / "google-speech-commands" / word
    data_dir.mkdir()
    word_dir.mkdir(parents=True)
    for index in range(sample_count):
        _write_wav(word_dir / f"sample-{index}.wav")


def _manifest_basenames(path: Path) -> list[str]:
    return sorted(Path(str(entry["audio_filepath"])).name for entry in _read_jsonl(path))


if __name__ == "__main__":
    unittest.main()
