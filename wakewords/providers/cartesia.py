from __future__ import annotations

import logging
import os

from cartesia import Cartesia

from wakewords.providers.base import Voice

logger = logging.getLogger(__name__)


class CartesiaProvider:
    name = "cartesia"
    short_code = "cr"

    def list_voices(
        self,
        pages: int = 1,
        all: bool = False,
        lang: str | None = None,
        gender: str | None = None,
    ) -> list[Voice]:
        with _client() as client:
            list_kwargs = {"limit": 100}
            if gender:
                list_kwargs["gender"] = _cartesia_gender(gender)
            if lang:
                list_kwargs["extra_query"] = {"language": lang}

            voices: list[Voice] = []
            page = 1
            logger.info("Fetching Cartesia voices page %s with params %s", page, list_kwargs or "{}")
            voice_page = client.voices.list(**list_kwargs)

            while True:
                for raw_voice in voice_page.data:
                    voices.append(_voice_from_response(raw_voice))

                if not all and page >= pages:
                    return voices

                if not voice_page.has_next_page():
                    return voices

                page += 1
                page_info = voice_page.next_page_info()
                logger.info(
                    "Fetching Cartesia voices page %s with params %s",
                    page,
                    page_info.params if page_info else "{}",
                )
                voice_page = voice_page.get_next_page()

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
        with _client() as client:
            generate_kwargs = {}
            if lang:
                generate_kwargs["language"] = lang

            response = client.tts.generate(
                model_id=model_id,
                output_format={
                    "container": "wav",
                    "encoding": encoding,
                    "sample_rate": sample_rate,
                },
                transcript=prompt,
                voice={
                    "mode": "id",
                    "id": voice.id,
                },
                **generate_kwargs,
            )
            return response.read()


def _voice_from_response(raw_voice: object) -> Voice:
    return Voice(
        id=str(getattr(raw_voice, "id")),
        name=_optional_str(getattr(raw_voice, "name", None)),
        language=_optional_str(getattr(raw_voice, "language", None)),
        gender=_generic_gender(_optional_str(getattr(raw_voice, "gender", None))),
    )


def _client() -> Cartesia:
    return Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _cartesia_gender(gender: str) -> str:
    normalized = gender.strip().lower()
    if normalized == "masculine":
        return "masculine"
    if normalized == "feminine":
        return "feminine"
    if normalized in {"neutral", "nonbinary", "non-binary", "gender_neutral"}:
        return "gender_neutral"
    raise ValueError(f"Unsupported Cartesia gender: {gender}")


def _generic_gender(gender: str | None) -> str | None:
    if gender is None:
        return None
    normalized = gender.strip().lower()
    if normalized == "gender_neutral":
        return "neutral"
    return normalized


def _slug(value: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled"

