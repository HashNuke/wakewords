from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from wakewords.audio import trim_wav_to_speech

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    id: str
    name: str | None = None
    language: str | None = None
    gender: str | None = None


@dataclass(frozen=True)
class VoiceSelectionConfig:
    group_by: tuple[Literal["language", "gender"], Literal["language", "gender"]]
    languages: tuple[str, ...] | Literal["all"]
    genders: tuple[str, ...]
    limit_per_group: int


@dataclass(frozen=True)
class GeneratedAudioContext:
    prompt: str
    label: str
    provider: str
    voice_id: str
    voice_code: str
    sample_id: str


@dataclass(frozen=True)
class GenerationPrompt:
    tts_input: str
    label: str


def prepare_generated_audio(audio_bytes: bytes, *, context: GeneratedAudioContext) -> bytes | None:
    processed_audio = trim_wav_to_speech(audio_bytes)
    if processed_audio is None:
        logger.warning(
            "Skipping generated sample because VAD detected no speech: prompt=%r label=%s provider=%s voice_id=%s voice_code=%s sample_id=%s",
            context.prompt,
            context.label,
            context.provider,
            context.voice_id,
            context.voice_code,
            context.sample_id,
        )
    return processed_audio


class TTSProvider(Protocol):
    name: str

    def list_voices(
        self,
        pages: int = 1,
        all: bool = False,
        lang: str | None = None,
        gender: str | None = None,
    ) -> list[Voice]:
        ...

    def generate(
        self,
        *,
        prompt: str,
        voice: Voice,
        lang: str | None,
        model_id: str,
        sample_rate: int,
        encoding: str,
    ) -> bytes:
        ...
