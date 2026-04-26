from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wakeword.augment import _collect_noises


class AugmentTests(unittest.TestCase):
    def test_collect_noises_uses_manifest_durations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            noises_dir = Path(tmp_dir)
            (noises_dir / "rain.wav").write_bytes(b"not a real wav")
            (noises_dir / "manifest.jsonl").write_text(
                json.dumps({"audio": "rain.wav", "duration_ms": 1234}) + "\n",
                encoding="utf-8",
            )

            with mock.patch("wakeword.augment._media_duration_seconds") as probe:
                noises = _collect_noises(noises_dir)

            probe.assert_not_called()
            self.assertEqual(len(noises), 1)
            self.assertEqual(noises[0].path, noises_dir / "rain.wav")
            self.assertEqual(noises[0].duration_ms, 1234)
            self.assertEqual(noises[0].duration, 1.234)


if __name__ == "__main__":
    unittest.main()
