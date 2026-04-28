from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wakewords.download import download_datasets


class DownloadTests(unittest.TestCase):
    def test_download_datasets_downloads_google_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            downloads_dir = project_dir / "downloads"

            with mock.patch("wakewords.download._download_file") as download_file:
                with mock.patch("wakewords.download._extract_tar") as extract_tar:
                    outputs = download_datasets(
                        downloads_dir=downloads_dir,
                        data_dir=project_dir,
                    )

            self.assertEqual(outputs, [project_dir / "google-speech-commands"])
            download_file.assert_called_once()
            extract_tar.assert_called_once_with(
                downloads_dir / "speech_commands_v0.02.tar.gz",
                project_dir / "google-speech-commands",
                description="Extract Google Speech Commands",
            )


if __name__ == "__main__":
    unittest.main()
