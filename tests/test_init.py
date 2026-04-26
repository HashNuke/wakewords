from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from wakeword import cli
from wakeword.project import GOOGLE_SPEECH_COMMANDS, init_project


class InitProjectTests(unittest.TestCase):
    def test_init_project_creates_expected_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)

            outputs = init_project(project_dir)

            self.assertEqual(
                outputs,
                [
                    project_dir / "data",
                    project_dir / "config.json",
                ],
            )
            self.assertTrue((project_dir / "data").is_dir())
            self.assertFalse((project_dir / "background_audio").exists())

            config_text = (project_dir / "config.json").read_text(encoding="utf-8")
            config = json.loads(config_text)
            self.assertEqual(config["custom_words"], ["dexa"])
            self.assertEqual(config["google_speech_commands"], GOOGLE_SPEECH_COMMANDS)
            self.assertEqual(
                config_text,
                json.dumps(
                    {
                        "custom_words": ["dexa"],
                        "google_speech_commands": GOOGLE_SPEECH_COMMANDS,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
            )

    def test_init_project_refuses_to_overwrite_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                init_project(project_dir)

    def test_cli_init_prints_created_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()

            with mock.patch.object(cli.sys, "argv", ["wakeword", "init", tmp_dir]):
                with redirect_stdout(stdout):
                    cli.main()

            output_lines = stdout.getvalue().strip().splitlines()
            project_dir = Path(tmp_dir)
            self.assertEqual(
                output_lines,
                [
                    str(project_dir / "data"),
                    str(project_dir / "config.json"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
