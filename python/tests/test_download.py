from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wakewords.project import BACKGROUND_AUDIO_URL
from wakewords.download import download_datasets


class DownloadTests(unittest.TestCase):
    def test_download_datasets_downloads_google_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            downloads_dir = project_dir / "downloads"

            with mock.patch("wakewords.download._download_file") as download_file:
                with mock.patch("wakewords.download._extract_tar") as extract_tar:
                    with mock.patch("wakewords.download._extract_zip") as extract_zip:
                        outputs = download_datasets(
                            downloads_dir=downloads_dir,
                            data_dir=project_dir,
                        )

            self.assertEqual(
                outputs,
                [
                    project_dir / "google-speech-commands",
                    project_dir / "background_audio",
                ],
            )
            self.assertEqual(download_file.call_count, 2)
            download_file.assert_has_calls(
                [
                    mock.call(
                        "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                        downloads_dir / "speech_commands_v0.02.tar.gz",
                        description="Google Speech Commands",
                    ),
                    mock.call(
                        BACKGROUND_AUDIO_URL,
                        downloads_dir / "background-noise-r1.zip",
                        description="Background Audio",
                    ),
                ],
            )
            extract_tar.assert_called_once_with(
                downloads_dir / "speech_commands_v0.02.tar.gz",
                project_dir / "google-speech-commands",
                description="Extract Google Speech Commands",
            )
            extract_zip.assert_called_once_with(
                downloads_dir / "background-noise-r1.zip",
                project_dir / "background_audio",
            )


if __name__ == "__main__":
    unittest.main()
