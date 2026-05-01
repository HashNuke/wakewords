from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakewords import cli
from wakewords.project import (
    BACKGROUND_AUDIO_URL,
    CUSTOM_WORDS_SAMPLE,
    GOOGLE_SPEECH_COMMANDS,
    _extract_background_audio_archive,
    init_project,
)


class InitProjectTests(unittest.TestCase):
    def test_init_project_creates_expected_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)

            with (
                mock.patch("wakewords.project.secrets.randbits", return_value=123456789),
                mock.patch("wakewords.project._download_background_audio") as download_background_audio,
            ):
                download_background_audio.side_effect = _write_background_audio_fixture
                outputs = init_project(project_dir)

            download_background_audio.assert_called_once_with(project_dir / "background_audio")
            self.assertEqual(
                outputs,
                [
                    project_dir / "data",
                    project_dir / "background_audio",
                    project_dir / "config.json",
                    project_dir / ".gitignore",
                ],
            )
            self.assertTrue((project_dir / "data").is_dir())
            self.assertEqual(
                sorted(path.name for path in (project_dir / "background_audio").iterdir()),
                [
                    "README.md",
                    "manifest.jsonl",
                    "noise.wav",
                ],
            )
            manifest_entries = [
                json.loads(line)
                for line in (project_dir / "background_audio" / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                manifest_entries,
                [
                    {"audio": "noise.wav", "duration_ms": 60000},
                ],
            )

            config_text = (project_dir / "config.json").read_text(encoding="utf-8")
            config = json.loads(config_text)
            self.assertEqual(config["custom_words"], CUSTOM_WORDS_SAMPLE)
            self.assertEqual(config["google_speech_commands"], GOOGLE_SPEECH_COMMANDS)
            self.assertEqual(
                config["augment"],
                {
                    "seed": 123456789,
                    "parquet_writes_batch_size": 128,
                    "speech_context": {
                        "enabled": False,
                        "target_duration_ms": 1000,
                        "gap_ms": [10, 100],
                        "reverse_donor": True,
                    },
                },
            )
            self.assertEqual(
                config_text,
                json.dumps(
                    {
                        "custom_words": CUSTOM_WORDS_SAMPLE,
                        "google_speech_commands": GOOGLE_SPEECH_COMMANDS,
                        "augment": {
                            "seed": 123456789,
                            "parquet_writes_batch_size": 128,
                            "speech_context": {
                                "enabled": False,
                                "target_duration_ms": 1000,
                                "gap_ms": [10, 100],
                                "reverse_donor": True,
                            },
                        },
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
            )
            self.assertEqual(
                (project_dir / ".gitignore").read_text(encoding="utf-8"),
                "google-speech-commands/\ndiagnostics/\n",
            )

    def test_init_project_refuses_to_overwrite_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text("{}\n", encoding="utf-8")

            with (
                self.assertRaises(FileExistsError),
                mock.patch("wakewords.project._download_background_audio") as download_background_audio,
            ):
                init_project(project_dir)

            download_background_audio.assert_not_called()

    def test_extract_background_audio_archive_strips_single_root_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            archive_path = tmp_path / "background.zip"
            output_dir = tmp_path / "background_audio"
            output_dir.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("background_audio/manifest.jsonl", "{}\n")
                archive.writestr("background_audio/noise.wav", b"wav")
                archive.writestr("background_audio/README.md", "credits\n")
                archive.writestr("__MACOSX/background_audio/._noise.wav", b"metadata")
                archive.writestr(".DS_Store", b"metadata")

            _extract_background_audio_archive(archive_path, output_dir)

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["README.md", "manifest.jsonl", "noise.wav"],
            )
            self.assertFalse((output_dir / "background_audio").exists())
            self.assertFalse((output_dir / "__MACOSX").exists())

    def test_background_audio_url_points_to_release_asset(self) -> None:
        self.assertEqual(
            BACKGROUND_AUDIO_URL,
            "https://github.com/HashNuke/wakewords/releases/download/background-audio-r2/background_audio.zip",
        )

    def test_cli_init_prints_created_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()

            with (
                mock.patch.object(cli.sys, "argv", ["wakewords", "init", tmp_dir]),
                mock.patch("wakewords.project._download_background_audio", side_effect=_write_background_audio_fixture),
            ):
                with redirect_stdout(stdout):
                    cli.main()

            output_lines = stdout.getvalue().strip().splitlines()
            project_dir = Path(tmp_dir)
            self.assertEqual(
                output_lines,
                [
                    str(project_dir / "data"),
                    str(project_dir / "background_audio"),
                    str(project_dir / "config.json"),
                    str(project_dir / ".gitignore"),
                ],
            )

    def test_cli_version_prints_project_version(self) -> None:
        for version_flag in ("--version", "—version"):
            with self.subTest(version_flag=version_flag):
                stdout = io.StringIO()

                with mock.patch.object(cli.sys, "argv", ["wakewords", version_flag]):
                    with mock.patch.object(cli, "_package_version", return_value="0.2.1"):
                        with redirect_stdout(stdout):
                            cli.main()

                self.assertEqual(stdout.getvalue(), "0.2.1\n")


def _write_background_audio_fixture(background_audio_dir: Path) -> None:
    (background_audio_dir / "README.md").write_text("credits\n", encoding="utf-8")
    (background_audio_dir / "manifest.jsonl").write_text(
        json.dumps({"audio": "noise.wav", "duration_ms": 60000}) + "\n",
        encoding="utf-8",
    )
    (background_audio_dir / "noise.wav").write_bytes(b"wav")


if __name__ == "__main__":
    unittest.main()
