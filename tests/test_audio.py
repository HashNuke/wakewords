from __future__ import annotations

import io
import unittest
import wave
from unittest import mock

from wakewords.audio import trim_wav_to_speech
from wakewords.parquet_store import probe_wav_bytes


class AudioTrimTests(unittest.TestCase):
    def test_trim_wav_to_speech_keeps_available_leading_and_200ms_trailing_padding(self) -> None:
        audio_bytes = _wav_bytes(
            _silence(100)
            + _tone(2000)
            + _silence(3000)
        )

        with mock.patch(
            "wakewords.audio._speech_timestamps",
            return_value=[{"start": _frames(100), "end": _frames(2100)}],
        ):
            trimmed = trim_wav_to_speech(audio_bytes)

        self.assertEqual(probe_wav_bytes(trimmed), (16000, 1, 2300))

    def test_trim_wav_to_speech_keeps_200ms_padding_on_each_side_when_available(self) -> None:
        audio_bytes = _wav_bytes(
            _silence(500)
            + _tone(1000)
            + _silence(500)
        )

        with mock.patch(
            "wakewords.audio._speech_timestamps",
            return_value=[{"start": _frames(500), "end": _frames(1500)}],
        ):
            trimmed = trim_wav_to_speech(audio_bytes)

        self.assertEqual(probe_wav_bytes(trimmed), (16000, 1, 1400))

    def test_trim_wav_to_speech_validates_but_does_not_trim_audio_at_or_below_threshold(self) -> None:
        audio_bytes = _wav_bytes(_silence(100) + _tone(900) + _silence(200))

        with mock.patch(
            "wakewords.audio._speech_timestamps",
            return_value=[{"start": _frames(100), "end": _frames(1000)}],
        ) as speech_timestamps:
            trimmed = trim_wav_to_speech(audio_bytes)

        self.assertEqual(trimmed, audio_bytes)
        speech_timestamps.assert_called_once()

    def test_trim_wav_to_speech_returns_none_when_vad_finds_no_speech(self) -> None:
        audio_bytes = _wav_bytes(_silence(1500))

        with (
            mock.patch("wakewords.audio._speech_timestamps", return_value=[]),
            mock.patch("wakewords.audio.logger"),
        ):
            trimmed = trim_wav_to_speech(audio_bytes)

        self.assertIsNone(trimmed)

    def test_trim_wav_to_speech_keeps_original_when_vad_fails(self) -> None:
        audio_bytes = _wav_bytes(_silence(1500))

        with (
            mock.patch("wakewords.audio._speech_timestamps", side_effect=RuntimeError("model error")),
            mock.patch("wakewords.audio.logger"),
        ):
            trimmed = trim_wav_to_speech(audio_bytes)

        self.assertEqual(trimmed, audio_bytes)


def _silence(duration_ms: int) -> list[int]:
    return [0] * _frames(duration_ms)


def _tone(duration_ms: int) -> list[int]:
    return [10_000] * _frames(duration_ms)


def _frames(duration_ms: int) -> int:
    return 16000 * duration_ms // 1000


def _wav_bytes(samples: list[int]) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
