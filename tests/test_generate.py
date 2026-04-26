from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
