from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wakewords.download import MATCHBOXNET_MODEL_FILENAME, download_datasets


class DownloadTests(unittest.TestCase):
    def test_download_datasets_downloads_google_data_and_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            downloads_dir = project_dir / "downloads"
            models_dir = project_dir / "models" / "base"

            with mock.patch("wakewords.download._download_file") as download_file:
                with mock.patch("wakewords.download._extract_tar") as extract_tar:
                    outputs = download_datasets(
                        downloads_dir=downloads_dir,
                        data_dir=data_dir,
                        models_dir=models_dir,
                    )

            self.assertEqual(
                outputs,
                [
                    data_dir / "google-speech-commands",
                    models_dir / MATCHBOXNET_MODEL_FILENAME,
                ],
            )
            self.assertEqual(download_file.call_count, 2)
            extract_tar.assert_called_once_with(
                downloads_dir / "speech_commands_v0.02.tar.gz",
                data_dir / "google-speech-commands",
                description="Extract Google Speech Commands",
            )


if __name__ == "__main__":
    unittest.main()
