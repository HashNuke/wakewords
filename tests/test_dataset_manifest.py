from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wakewords.dataset_manifest import build_split_manifests


class DatasetManifestTests(unittest.TestCase):
    def test_build_split_manifests_writes_project_root_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            word_dir = project_dir / "data" / "yes"
            word_dir.mkdir(parents=True)
            (word_dir / "manifest.jsonl").write_text(
                json.dumps({"audio_filepath": "yes.wav", "duration": 1.0, "label": "yes"}) + "\n",
                encoding="utf-8",
            )

            outputs = build_split_manifests(
                data_dir=project_dir / "data",
                train_ratio=1,
                validate_ratio=0,
                test_ratio=0,
            )

            self.assertEqual(
                outputs,
                {
                    "train": project_dir / "train_manifest.jsonl",
                    "validate": project_dir / "validation_manifest.jsonl",
                    "test": project_dir / "test_manifest.jsonl",
                },
            )
            self.assertTrue((project_dir / "train_manifest.jsonl").is_file())
            self.assertFalse((project_dir / "data" / "train_manifest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
