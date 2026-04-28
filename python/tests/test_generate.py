from __future__ import annotations

import json
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from wakewords.cli import DataTools
from wakewords.generate import build_tasks, generate_audio, select_voices
from wakewords.parquet_store import CustomWordStore
from wakewords.providers.base import GeneratedAudioContext, GenerationPrompt, Voice, VoiceSelectionConfig, prepare_generated_audio
from wakewords.providers.cartesia import CartesiaProvider, _cartesia_gender, _generic_gender


class GenerateVoiceSelectionTests(unittest.TestCase):
    def test_select_voices_limits_first_matching_language_page(self) -> None:
        provider = CartesiaProvider()
        calls = []

        def list_voices(
            *, pages: int = 1, all: bool = False, lang: str | None = None, gender: str | None = None
        ) -> list[Voice]:
            calls.append({"pages": pages, "all": all, "lang": lang, "gender": gender})
            return [
                Voice(id="v1", name="Voice 1", language=lang),
                Voice(id="v2", name="Voice 2", language=lang),
                Voice(id="v3", name="Voice 3", language=lang),
            ]

        provider.list_voices = list_voices  # type: ignore[method-assign]

        selected = select_voices(provider=provider, voice=None, voices=2, all_voices=False, lang="en")

        self.assertEqual([voice.id for voice in selected], ["v1", "v2"])
        self.assertEqual(calls, [{"pages": 1, "all": False, "lang": "en", "gender": None}])

    def test_select_voices_rejects_specific_voice_with_limit(self) -> None:
        provider = CartesiaProvider()

        with self.assertRaises(ValueError):
            select_voices(provider=provider, voice="v1", voices=2, all_voices=False, lang=None)

    def test_select_grouped_voices_picks_limit_per_gender_for_all_languages(self) -> None:
        provider = CartesiaProvider()
        calls = []

        def list_voices(
            *, pages: int = 1, all: bool = False, lang: str | None = None, gender: str | None = None
        ) -> list[Voice]:
            calls.append({"pages": pages, "all": all, "lang": lang, "gender": gender})
            voices = [
                Voice(id="en-m-1", language="en", gender="masculine"),
                Voice(id="en-m-2", language="en", gender="masculine"),
                Voice(id="en-m-3", language="en", gender="masculine"),
                Voice(id="en-f-1", language="en", gender="feminine"),
                Voice(id="en-f-2", language="en", gender="feminine"),
                Voice(id="es-m-1", language="es", gender="masculine"),
                Voice(id="es-m-2", language="es", gender="masculine"),
                Voice(id="es-f-1", language="es", gender="feminine"),
                Voice(id="es-x-1", language="es", gender="nonbinary"),
                Voice(id="missing-gender", language="es"),
            ]
            if gender:
                voices = [voice for voice in voices if voice.gender == gender]
            return voices

        provider.list_voices = list_voices  # type: ignore[method-assign]

        selected = select_voices(
            provider=provider,
            voice=None,
            voices=None,
            all_voices=False,
            lang=None,
            voice_selection=VoiceSelectionConfig(
                group_by=("language", "gender"),
                languages="all",
                genders=("masculine", "feminine"),
                limit_per_group=2,
            ),
        )

        self.assertEqual(
            [voice.id for voice in selected],
            ["en-m-1", "en-m-2", "en-f-1", "en-f-2", "es-m-1", "es-m-2", "es-f-1"],
        )
        self.assertEqual(
            calls,
            [
                {"pages": 1, "all": True, "lang": None, "gender": "masculine"},
                {"pages": 1, "all": True, "lang": None, "gender": "feminine"},
            ],
        )

    def test_cartesia_gender_mapping_uses_sdk_values(self) -> None:
        self.assertEqual(_cartesia_gender("masculine"), "masculine")
        self.assertEqual(_cartesia_gender("feminine"), "feminine")
        self.assertEqual(_cartesia_gender("neutral"), "gender_neutral")
        self.assertEqual(_generic_gender("masculine"), "masculine")
        self.assertEqual(_generic_gender("feminine"), "feminine")
        self.assertEqual(_generic_gender("gender_neutral"), "neutral")


class GenerateCommandTests(unittest.TestCase):
    def test_generate_reads_custom_words_from_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": ["dexa", "tincan"], "google_speech_commands": ["yes"]}) + "\n",
                encoding="utf-8",
            )
            provider = _FakeProvider()

            with (
                mock.patch("wakewords.cli.get_provider", return_value=provider) as get_provider,
                mock.patch("wakewords.cli.generate_audio", return_value=[]) as generate_audio_mock,
            ):
                DataTools().generate(project_dir=str(project_dir))

            get_provider.assert_called_once_with("cartesia", config_path=project_dir / "config.json")
            self.assertEqual(
                generate_audio_mock.call_args.kwargs["prompts"],
                [GenerationPrompt(tts_input="dexa", label="dexa"), GenerationPrompt(tts_input="tincan", label="tincan")],
            )
            self.assertEqual(generate_audio_mock.call_args.kwargs["parquet_path"], project_dir / "data" / "custom_words.parquet")

    def test_generate_reads_object_custom_words_from_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": [{"tts_input": "Hey Astra", "label": "astra"}]}) + "\n",
                encoding="utf-8",
            )
            provider = _FakeProvider()

            with (
                mock.patch("wakewords.cli.get_provider", return_value=provider),
                mock.patch("wakewords.cli.generate_audio", return_value=[]) as generate_audio_mock,
            ):
                DataTools().generate(project_dir=str(project_dir))

            self.assertEqual(generate_audio_mock.call_args.kwargs["prompts"], [GenerationPrompt(tts_input="Hey Astra", label="astra")])

    def test_generate_text_overrides_project_config_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps({"custom_words": ["dexa"]}) + "\n",
                encoding="utf-8",
            )
            provider = _FakeProvider()

            with (
                mock.patch("wakewords.cli.get_provider", return_value=provider),
                mock.patch("wakewords.cli.generate_audio", return_value=[]) as generate_audio_mock,
            ):
                DataTools().generate(project_dir=str(project_dir), text="single")

            self.assertEqual(generate_audio_mock.call_args.kwargs["prompts"], [GenerationPrompt(tts_input="single", label="single")])

    def test_generate_rejects_empty_custom_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(json.dumps({"custom_words": []}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "custom_words"):
                DataTools().generate(project_dir=str(project_dir))

    def test_generate_reads_voice_selection_from_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "config.json").write_text(
                json.dumps(
                    {
                        "custom_words": [{"tts_input": "Hey Astra", "label": "astra"}],
                        "generate": {
                            "voice_selection": {
                                "group_by": ["language", "gender"],
                                "languages": "all",
                                "genders": ["masculine", "feminine"],
                                "limit_per_group": 3,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider = _FakeProvider()

            with (
                mock.patch("wakewords.cli.get_provider", return_value=provider),
                mock.patch("wakewords.cli.generate_audio", return_value=[]) as generate_audio_mock,
            ):
                DataTools().generate(project_dir=str(project_dir))

            self.assertEqual(
                generate_audio_mock.call_args.kwargs["voice_selection"],
                VoiceSelectionConfig(
                    group_by=("language", "gender"),
                    languages="all",
                    genders=("masculine", "feminine"),
                    limit_per_group=3,
                ),
            )

    def test_build_tasks_deduplicates_prompt_collisions(self) -> None:
        voice = Voice(id="voice-1", name="Voice 1", language="en")

        tasks = build_tasks(
            prompts=[
                GenerationPrompt(tts_input="Hey Astra", label="astra"),
                GenerationPrompt(tts_input="Astra", label="astra"),
            ],
            voices=[voice],
            voice_codes={voice.id: "cr1"},
            provider="cr",
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].word_slug, "astra")
        self.assertTrue(tasks[0].sample_id)

    def test_cartesia_generate_writes_only_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            provider = CartesiaProvider()
            provider.list_voices = lambda **_: [  # type: ignore[method-assign]
                Voice(id="voice-1", name="Voice 1", language="en")
            ]
            client = _FakeCartesiaClient()

            with (
                mock.patch("wakewords.providers.cartesia._client", return_value=client),
                mock.patch("wakewords.generate.prepare_generated_audio", side_effect=lambda audio_bytes, *, context: audio_bytes),
            ):
                outputs = generate_audio(
                    provider=provider,
                    prompts=[GenerationPrompt(tts_input="Hey Astra", label="astra")],
                    parquet_path=data_dir / "custom_words.parquet",
                    voice=None,
                    voices=None,
                    all_voices=False,
                    lang=None,
                    voice_selection=None,
                    concurrency=1,
                    model_id="sonic-3",
                    sample_rate=16000,
                    encoding="pcm_s16le",
                    overwrite=False,
                )

            self.assertEqual(outputs, [data_dir / "custom_words.parquet"])
            self.assertEqual(client.transcripts, ["Hey Astra"])
            self.assertFalse((data_dir / "astra").exists())
            self.assertFalse((data_dir / "astra" / "manifest.jsonl").exists())
            rows = CustomWordStore(data_dir / "custom_words.parquet").rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "astra")
            self.assertNotIn("filename", rows[0])
            self.assertIsInstance(rows[0]["audio_bytes"], bytes)

    def test_cartesia_generate_skips_parquet_row_when_vad_detects_no_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            provider = CartesiaProvider()
            provider.list_voices = lambda **_: [  # type: ignore[method-assign]
                Voice(id="voice-1", name="Voice 1", language="en")
            ]

            with (
                mock.patch("wakewords.providers.cartesia._client", return_value=_FakeCartesiaClient()),
                mock.patch("wakewords.generate.prepare_generated_audio", return_value=None),
            ):
                outputs = generate_audio(
                    provider=provider,
                    prompts=[GenerationPrompt(tts_input="Hey Astra", label="astra")],
                    parquet_path=data_dir / "custom_words.parquet",
                    voice=None,
                    voices=None,
                    all_voices=False,
                    lang=None,
                    voice_selection=None,
                    concurrency=1,
                    model_id="sonic-3",
                    sample_rate=16000,
                    encoding="pcm_s16le",
                    overwrite=False,
                )

            self.assertEqual(outputs, [])
            self.assertFalse((data_dir / "custom_words.parquet").exists())

    def test_prepare_generated_audio_logs_context_when_vad_detects_no_speech(self) -> None:
        context = GeneratedAudioContext(
            prompt="Hey Astra",
            label="hey-astra",
            provider="cr",
            voice_id="voice-1",
            voice_code="cr1",
            sample_id="sample-1",
        )

        with (
            mock.patch("wakewords.providers.base.trim_wav_to_speech", return_value=None),
            mock.patch("wakewords.providers.base.logger") as logger,
        ):
            audio_bytes = prepare_generated_audio(b"wav", context=context)

        self.assertIsNone(audio_bytes)
        logger.warning.assert_called_once()
        log_args = logger.warning.call_args.args
        self.assertIn("VAD detected no speech", log_args[0])
        self.assertEqual(log_args[1:], ("Hey Astra", "hey-astra", "cr", "voice-1", "cr1", "sample-1"))


class _FakeProvider:
    name = "fake"

    def list_voices(self, **kwargs: object) -> list[Voice]:
        return [Voice(id="voice-1", name="Voice 1", language="en")]

    def generate(self, **kwargs: object) -> bytes:
        return _wav_bytes()


class _FakeCartesiaClient:
    def __init__(self) -> None:
        self.tts = self
        self.transcripts: list[str] = []

    def __enter__(self) -> "_FakeCartesiaClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def generate(self, **kwargs: object) -> "_FakeTtsResponse":
        self.transcripts.append(str(kwargs["transcript"]))
        return _FakeTtsResponse()


class _FakeTtsResponse:
    def read(self) -> bytes:
        return _wav_bytes()


def _wav_bytes() -> bytes:
    sample_rate = 16000
    frame_count = sample_rate // 4
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
