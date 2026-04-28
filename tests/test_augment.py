from __future__ import annotations

import json
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from wakewords.augment import (
    AugmentTask,
    DEFAULT_PARQUET_WRITE_BATCH_SIZE,
    NoiseSample,
    SourceSample,
    _build_tasks,
    _collect_noises,
    _collect_sources,
    _combo_shape,
    _mix_to_snr,
    _select_subset,
    augment_dataset,
)
from wakewords.lfs import GitLfsPointerError
from wakewords import cli
from wakewords.parquet_store import CustomWordStore, build_generated_row


class AugmentTests(unittest.TestCase):
    def test_collect_noises_uses_manifest_durations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            noises_dir = Path(tmp_dir)
            (noises_dir / "rain.wav").write_bytes(b"not a real wav")
            (noises_dir / "manifest.jsonl").write_text(
                json.dumps({"audio": "rain.wav", "duration_ms": 1234}) + "\n",
                encoding="utf-8",
            )

            with mock.patch("wakewords.augment._media_duration_seconds") as probe:
                noises = _collect_noises(noises_dir)

            probe.assert_not_called()
            self.assertEqual(len(noises), 1)
            self.assertEqual(noises[0].path, noises_dir / "rain.wav")
            self.assertEqual(noises[0].duration_ms, 1234)
            self.assertEqual(noises[0].duration, 1.234)

    def test_build_tasks_uses_subset_combo_counts_per_voice(self) -> None:
        source = SourceSample(
            sample_id="sample-yes-1",
            word="yes",
            provider="cr",
            voice_id="voice-1",
            duration=1.0,
            duration_ms=1000,
            audio_bytes=b"wav",
            row={},
        )
        noises = [
            NoiseSample(path=Path(f"background_audio/noise-{index}.wav"), duration=60.0, duration_ms=60000)
            for index in range(6)
        ]

        tasks = _build_tasks(
            sources=[source],
            noises=noises,
            tempos=(0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15),
            snrs=(20, 10, 5),
            target_samples_per_word=11,
        )

        self.assertEqual(len(tasks), 10)
        self.assertTrue(all(task.noise is not None for task in tasks))
        self.assertEqual(len({task.tempo for task in tasks}), 5)
        self.assertEqual(len({task.noise for task in tasks}), 2)
        self.assertEqual(len({task.snr for task in tasks}), 1)

    def test_collect_sources_keeps_directory_label_with_hyphens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CustomWordStore(Path(tmp_dir) / "custom_words.parquet")
            row = build_generated_row(
                audio_bytes=_wav_bytes(),
                label="hey-astra-now",
                voice_id="voice-107",
                voice_code="cr107",
                provider="cr",
                lang="en",
            )
            store.upsert(row, overwrite=False)

            sources = _collect_sources(store)

        self.assertEqual([source.word for source in sources], ["hey-astra-now"])
        self.assertEqual([source.sample_id for source in sources], [row["sample_id"]])
        self.assertEqual([source.voice_id for source in sources], ["voice-107"])

    def test_augment_dataset_appends_parquet_rows_without_wav_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            noises_dir = project_dir / "background_audio"
            noises_dir.mkdir()
            (noises_dir / "rain.wav").write_bytes(b"not read because manifest exists")
            (noises_dir / "manifest.jsonl").write_text(
                json.dumps({"audio": "rain.wav", "duration_ms": 1000}) + "\n",
                encoding="utf-8",
            )
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

            with (
                mock.patch("wakewords.augment._tempo_adjust", side_effect=_copy_to_temp_wav),
                mock.patch("wakewords.augment._extract_noise_segment", side_effect=lambda **_: _temp_wav(_wav_bytes())),
                mock.patch("wakewords.augment._mix_to_snr", side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(_wav_bytes())),
            ):
                outputs = augment_dataset(
                    data_dir=data_dir,
                    noises_dir=noises_dir,
                    concurrency=1,
                    overwrite=False,
                    tempos=(1.0,),
                    snrs=(10,),
                    target_samples_per_word=2,
                )

            self.assertEqual(outputs, [data_dir / "custom_words.parquet"])
            self.assertFalse((data_dir / "yes").exists())
            rows = CustomWordStore(data_dir / "custom_words.parquet").rows()
            self.assertEqual(sorted(row["source_type"] for row in rows), ["augmented", "generated"])
            generated = next(row for row in rows if row["source_type"] == "generated")
            augmented = next(row for row in rows if row["source_type"] == "augmented")
            self.assertEqual(augmented["label"], "yes")
            self.assertNotIn("filename", generated)
            self.assertNotIn("filename", augmented)
            self.assertEqual(augmented["tempo"], 1.0)
            self.assertEqual(augmented["noise_type"], "rain")
            self.assertEqual(augmented["snr"], 10)

    def test_augment_dataset_reports_lfs_background_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            noises_dir = project_dir / "background_audio"
            noises_dir.mkdir()
            (noises_dir / "rain.wav").write_bytes(_lfs_pointer_bytes(size=1234))
            (noises_dir / "manifest.jsonl").write_text(
                json.dumps({"audio": "rain.wav", "duration_ms": 1000}) + "\n",
                encoding="utf-8",
            )
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

            with self.assertRaisesRegex(GitLfsPointerError, "git lfs pull"):
                augment_dataset(
                    data_dir=data_dir,
                    noises_dir=noises_dir,
                    concurrency=1,
                    overwrite=False,
                    tempos=(1.0,),
                    snrs=(10,),
                    target_samples_per_word=2,
                )

    def test_augment_dataset_uses_configured_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            noises_dir = project_dir / "background_audio"
            noises_dir.mkdir()
            (noises_dir / "rain.wav").write_bytes(b"not read because manifest exists")
            (noises_dir / "manifest.jsonl").write_text(
                json.dumps({"audio": "rain.wav", "duration_ms": 1000}) + "\n",
                encoding="utf-8",
            )
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

            original_upsert_many = CustomWordStore.upsert_many
            with (
                mock.patch("wakewords.augment._tempo_adjust", side_effect=_copy_to_temp_wav),
                mock.patch("wakewords.augment._extract_noise_segment", side_effect=lambda **_: _temp_wav(_wav_bytes())),
                mock.patch("wakewords.augment._mix_to_snr", side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(_wav_bytes())),
                mock.patch.object(CustomWordStore, "upsert_many", autospec=True, wraps=original_upsert_many) as upsert_many,
            ):
                augment_dataset(
                    data_dir=data_dir,
                    noises_dir=noises_dir,
                    concurrency=1,
                    overwrite=False,
                    tempos=(0.95, 1.0, 1.05),
                    snrs=(10,),
                    target_samples_per_word=4,
                    parquet_writes_batch_size=2,
                )

            self.assertEqual(upsert_many.call_count, 2)

    def test_load_augment_batch_size_reads_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"augment": {"parquet_writes_batch_size": 17}}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                cli._load_augment_parquet_writes_batch_size(config_path=config_path),
                17,
            )

    def test_load_augment_batch_size_defaults_when_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps({"custom_words": []}) + "\n", encoding="utf-8")

            self.assertEqual(
                cli._load_augment_parquet_writes_batch_size(config_path=config_path),
                DEFAULT_PARQUET_WRITE_BATCH_SIZE,
            )

    def test_load_augment_batch_size_rejects_invalid_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"augment": {"parquet_writes_batch_size": 0}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "augment.parquet_writes_batch_size must be >= 1"):
                cli._load_augment_parquet_writes_batch_size(config_path=config_path)

    def test_combo_shape_tracks_target_as_voice_count_changes(self) -> None:
        self.assertEqual(
            _combo_shape(
                voice_count=373,
                target_samples_per_word=4000,
                tempos_available=7,
                noises_available=6,
                snrs_available=3,
            ),
            (5, 2, 1),
        )
        self.assertEqual(
            _combo_shape(
                voice_count=1000,
                target_samples_per_word=4000,
                tempos_available=7,
                noises_available=6,
                snrs_available=3,
            ),
            (3, 1, 1),
        )

    def test_select_subset_shuffles_deterministically_per_voice_and_category(self) -> None:
        source = SourceSample(
            sample_id="sample-yes-1",
            word="yes",
            provider="cr",
            voice_id="voice-1",
            duration=1.0,
            duration_ms=1000,
            audio_bytes=b"wav",
            row={},
        )
        values = (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)

        first = _select_subset(values, 5, source=source, category="tempo")
        second = _select_subset(values, 5, source=source, category="tempo")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertNotEqual(first, values[:5])

    def test_mix_to_snr_uses_ffmpeg_filter_with_double_precision_and_length_lock(self) -> None:
        speech_path = Path("speech.wav")
        noise_path = Path("noise.wav")
        output_path = Path("output.wav")

        with (
            mock.patch(
                "wakewords.augment._read_wav_samples",
                side_effect=[
                    ((1, 2, 16000), [1000, -1000, 500, -500]),
                    ((1, 2, 16000), [250, -250]),
                ],
            ),
            mock.patch("wakewords.augment._run_ffmpeg") as run_ffmpeg,
        ):
            _mix_to_snr(
                speech_path=speech_path,
                noise_path=noise_path,
                output_path=output_path,
                snr_db=10,
            )

        run_ffmpeg.assert_called_once()
        command = run_ffmpeg.call_args.args[0]
        self.assertEqual(command[:6], ["ffmpeg", "-y", "-i", str(speech_path), "-i", str(noise_path)])
        self.assertEqual(command[-3:], ["-c:a", "pcm_s16le", str(output_path)])
        filter_complex = command[command.index("-filter_complex") + 1]
        self.assertIn("[0:a]atrim=end_sample=4[speech]", filter_complex)
        self.assertIn("precision=double", filter_complex)
        self.assertIn("apad,atrim=end_sample=4[scaled]", filter_complex)
        self.assertIn("amix=inputs=2:normalize=0:duration=first", filter_complex)


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


def _temp_wav(audio_bytes: bytes) -> Path:
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    temp_path.write_bytes(audio_bytes)
    return temp_path


def _copy_to_temp_wav(source_path: Path, tempo: float) -> Path:
    return _temp_wav(source_path.read_bytes())


def _lfs_pointer_bytes(*, size: int) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        + f"size {size}\n".encode("ascii")
    )


if __name__ == "__main__":
    unittest.main()
