from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wakewords.cli import DataTools
from wakewords.providers.base import Voice
from wakewords.providers.cartesia import CartesiaProvider


class GenerateVoiceSelectionTests(unittest.TestCase):
    def test_select_voices_limits_first_matching_language_page(self) -> None:
        provider = CartesiaProvider()
        calls = []

        def list_voices(*, pages: int = 1, all: bool = False, lang: str | None = None) -> list[Voice]:
            calls.append({"pages": pages, "all": all, "lang": lang})
            return [
                Voice(id="v1", name="Voice 1", language=lang),
                Voice(id="v2", name="Voice 2", language=lang),
                Voice(id="v3", name="Voice 3", language=lang),
            ]

        provider.list_voices = list_voices  # type: ignore[method-assign]

        selected = provider._select_voices(voice=None, voices=2, all_voices=False, lang="en")

        self.assertEqual([voice.id for voice in selected], ["v1", "v2"])
        self.assertEqual(calls, [{"pages": 1, "all": False, "lang": "en"}])

    def test_select_voices_rejects_specific_voice_with_limit(self) -> None:
        provider = CartesiaProvider()

        with self.assertRaises(ValueError):
            provider._select_voices(voice="v1", voices=2, all_voices=False, lang=None)


class GenerateCommandTests(unittest.TestCase):
    def test_generate_reads_custom_words_from_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": ["dexa", "tincan"], "google_speech_commands": ["yes"]}) + "\n",
                encoding="utf-8",
            )
            provider = _FakeProvider()

            with mock.patch("wakewords.cli.get_provider", return_value=provider) as get_provider:
                DataTools().generate(project_dir=str(project_dir))

            get_provider.assert_called_once_with("cartesia", config_path=project_dir / "config.json")
            self.assertEqual(provider.prompts, ["dexa", "tincan"])
            self.assertEqual(provider.data_dir, project_dir / "data")
            self.assertEqual(provider.parquet_path, project_dir / "data" / "custom_words.parquet")

    def test_generate_text_overrides_project_config_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": ["dexa"]}) + "\n",
                encoding="utf-8",
            )
            provider = _FakeProvider()

            with mock.patch("wakewords.cli.get_provider", return_value=provider):
                DataTools().generate(project_dir=str(project_dir), text="single")

            self.assertEqual(provider.prompts, ["single"])

    def test_generate_rejects_empty_custom_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(json.dumps({"custom_words": []}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "custom_words"):
                DataTools().generate(project_dir=str(project_dir))


class _FakeProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.data_dir: Path | None = None
        self.parquet_path: Path | None = None

    def generate(self, **kwargs: object) -> list[Path]:
        self.prompts = list(kwargs["prompts"])
        self.data_dir = kwargs["data_dir"]
        self.parquet_path = kwargs["parquet_path"]
        return []


if __name__ == "__main__":
    unittest.main()
