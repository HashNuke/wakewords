from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from wakewords.providers import get_provider


class ProviderRegistryTests(unittest.TestCase):
    def test_get_provider_loads_custom_provider_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "custom_provider.py").write_text(
                "from wakewords.providers.base import Voice\n"
                "class CustomProvider:\n"
                "    name = 'custom'\n"
                "    def list_voices(self, pages=1, all=False, lang=None):\n"
                "        return [Voice(id='custom-1', name='Custom Voice', language=lang)]\n"
                "    def generate(self, **kwargs):\n"
                "        return []\n",
                encoding="utf-8",
            )
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"providers": {"custom": "custom_provider:CustomProvider"}}) + "\n",
                encoding="utf-8",
            )

            sys.path.insert(0, str(project_dir))
            try:
                provider = get_provider("custom", config_path=config_path)
            finally:
                sys.path.remove(str(project_dir))
                sys.modules.pop("custom_provider", None)

            voices = provider.list_voices(lang="en")
            self.assertEqual(provider.name, "custom")
            self.assertEqual(voices[0].id, "custom-1")
            self.assertEqual(voices[0].language, "en")

    def test_config_providers_are_merged_with_builtins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"providers": {"custom": "custom_provider:CustomProvider"}}) + "\n",
                encoding="utf-8",
            )

            provider = get_provider("cartesia", config_path=config_path)

            self.assertEqual(provider.name, "cartesia")

    def test_get_provider_rejects_invalid_custom_provider_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "bad_provider.py").write_text("class BadProvider:\n    pass\n", encoding="utf-8")
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"providers": {"bad": "bad_provider:BadProvider"}}) + "\n",
                encoding="utf-8",
            )

            sys.path.insert(0, str(project_dir))
            try:
                with self.assertRaises(TypeError):
                    get_provider("bad", config_path=config_path)
            finally:
                sys.path.remove(str(project_dir))
                sys.modules.pop("bad_provider", None)


if __name__ == "__main__":
    unittest.main()
