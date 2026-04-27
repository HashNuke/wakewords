from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path

from cartesia import Cartesia

from wakewords.parquet_store import CustomWordStore, build_generated_row
from wakewords.providers.base import GeneratedAudioContext, GenerationPrompt, Voice, prepare_generated_audio

logger = logging.getLogger(__name__)


class CartesiaProvider:
    name = "cartesia"
    short_code = "cr"

    def list_voices(
        self,
        pages: int = 1,
        all: bool = False,
        lang: str | None = None,
    ) -> list[Voice]:
        with _client() as client:
            list_kwargs = {"limit": 100}
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
        prompts: list[GenerationPrompt],
        data_dir: Path,
        parquet_path: Path,
        voice: str | None,
        voices: int | None,
        all_voices: bool,
        lang: str | None,
        concurrency: int,
        model_id: str,
        sample_rate: int,
        encoding: str,
        overwrite: bool,
    ) -> list[Path]:
        voices = self._select_voices(voice=voice, voices=voices, all_voices=all_voices, lang=lang)
        store = CustomWordStore(parquet_path)
        voice_codes = {
            selected_voice.id: store.voice_code(provider=self.short_code, voice_id=selected_voice.id)
            for selected_voice in voices
        }
        tasks = _build_tasks(prompts=prompts, voices=voices, voice_codes=voice_codes, provider=self.short_code)

        wrote_rows = False

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self._generate_one,
                    task=task,
                    store=store,
                    lang=lang,
                    model_id=model_id,
                    sample_rate=sample_rate,
                    encoding=encoding,
                    overwrite=overwrite,
                )
                for task in tasks
            ]

            with tqdm(total=len(futures), unit="sample") as bar:
                for future in as_completed(futures):
                    wrote_rows = future.result() or wrote_rows
                    bar.update(1)

        return [parquet_path] if wrote_rows else []

    def _select_voices(
        self,
        *,
        voice: str | None,
        voices: int | None,
        all_voices: bool,
        lang: str | None,
    ) -> list[Voice]:
        if voices is not None and voices < 1:
            raise ValueError("voices must be >= 1")
        if voice and voices is not None:
            raise ValueError("Use --voice for one specific voice or --voices to limit selected voices, not both.")

        if all_voices:
            selected_voices = self.list_voices(all=True, lang=lang)
            if not selected_voices:
                raise RuntimeError("Cartesia returned no voices.")
            return selected_voices[:voices]

        if voice:
            selected_voices = self.list_voices(all=True, lang=lang)
            if not selected_voices:
                raise RuntimeError("Cartesia returned no voices.")
            normalized = voice.strip().lower()
            for candidate in selected_voices:
                if candidate.id.lower() == normalized:
                    return [candidate]
                if candidate.name and candidate.name.lower() == normalized:
                    return [candidate]
            raise ValueError(f"Could not find Cartesia voice by id or name: {voice}")

        selected_voices = self.list_voices(pages=1, all=False, lang=lang)
        if not selected_voices:
            raise RuntimeError("Cartesia returned no voices.")
        return selected_voices[: voices or 1]

    def _generate_one(
        self,
        *,
        task: "_GenerationTask",
        store: CustomWordStore,
        lang: str | None,
        model_id: str,
        sample_rate: int,
        encoding: str,
        overwrite: bool,
    ) -> bool:
        existing = store.get_by_sample_id(task.sample_id)
        if existing is not None and not overwrite:
            return False

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
                transcript=task.prompt,
                voice={
                    "mode": "id",
                    "id": task.voice.id,
                },
                **generate_kwargs,
            )
            audio_bytes = response.read()
            audio_bytes = prepare_generated_audio(
                audio_bytes,
                context=GeneratedAudioContext(
                    prompt=task.prompt,
                    label=task.word_slug,
                    provider=self.short_code,
                    voice_id=task.voice.id,
                    voice_code=task.voice_code,
                    sample_id=task.sample_id,
                ),
            )
            if audio_bytes is None:
                return False

        store.upsert(
            build_generated_row(
                audio_bytes=audio_bytes,
                label=task.word_slug,
                voice_id=task.voice.id,
                voice_code=task.voice_code,
                provider=self.short_code,
                lang=lang or task.voice.language,
            ),
            overwrite=overwrite,
        )

        return True


class _GenerationTask:
    def __init__(self, *, prompt: str, voice: Voice, voice_code: str, word_slug: str, sample_id: str) -> None:
        self.prompt = prompt
        self.voice = voice
        self.voice_code = voice_code
        self.word_slug = word_slug
        self.sample_id = sample_id


def _build_tasks(*, prompts: list[GenerationPrompt], voices: list[Voice], voice_codes: dict[str, str], provider: str) -> list[_GenerationTask]:
    tasks: list[_GenerationTask] = []
    seen: set[str] = set()
    for selected_voice in voices:
        voice_code = voice_codes[selected_voice.id]
        for prompt in prompts:
            word_slug = prompt.label
            sample_id = _generated_sample_id(label=word_slug, provider=provider, voice_id=selected_voice.id)
            if sample_id in seen:
                continue
            seen.add(sample_id)
            tasks.append(
                _GenerationTask(
                    prompt=prompt.tts_input,
                    voice=selected_voice,
                    voice_code=voice_code,
                    word_slug=word_slug,
                    sample_id=sample_id,
                )
            )
    return tasks


def _voice_from_response(raw_voice: object) -> Voice:
    return Voice(
        id=str(getattr(raw_voice, "id")),
        name=_optional_str(getattr(raw_voice, "name", None)),
        language=_optional_str(getattr(raw_voice, "language", None)),
    )


def _client() -> Cartesia:
    return Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _slug(value: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled"


def _generated_sample_id(*, label: str, provider: str, voice_id: str) -> str:
    import hashlib

    return hashlib.sha256(f"generated\0{label}\0{provider}\0{voice_id}".encode("utf-8")).hexdigest()
