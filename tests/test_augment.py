from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wakewords.augment import NoiseSample, SourceSample, _build_tasks, _collect_noises, _combo_shape, _select_subset


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
            path=Path("data/yes/yes-cr1-t100-clean-nonoise-nosnr.wav"),
            word="yes",
            voice_code="cr1",
            duration=1.0,
            duration_ms=1000,
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
            path=Path("data/yes/yes-cr1-t100-clean-nonoise-nosnr.wav"),
            word="yes",
            voice_code="cr1",
            duration=1.0,
            duration_ms=1000,
        )
        values = (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)

        first = _select_subset(values, 5, source=source, category="tempo")
        second = _select_subset(values, 5, source=source, category="tempo")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertNotEqual(first, values[:5])


if __name__ == "__main__":
    unittest.main()
