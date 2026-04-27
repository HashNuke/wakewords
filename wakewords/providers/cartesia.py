from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path

from cartesia import Cartesia

from wakewords.manifest import ManifestStore
from wakewords.parquet_store import CustomWordStore, build_generated_row
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
        prompts: list[str],
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
        manifests = ManifestStore()
        tasks = [
            _GenerationTask(prompt=prompt, voice=selected_voice)
            for selected_voice in voices
            for prompt in prompts
        ]

        outputs: list[Path] = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self._generate_one,
                    task=task,
                    data_dir=data_dir,
                    store=store,
                    manifests=manifests,
                    lang=lang,
                    model_id=model_id,
                    sample_rate=sample_rate,
                    encoding=encoding,
                    overwrite=overwrite,
                )
                for task in tasks
            ]

            with tqdm(total=len(futures), unit="file") as bar:
                for future in as_completed(futures):
                    outputs.append(future.result())
                    bar.update(1)

        return sorted(set(outputs))

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
        data_dir: Path,
        store: CustomWordStore,
        manifests: ManifestStore,
        lang: str | None,
        model_id: str,
        sample_rate: int,
        encoding: str,
        overwrite: bool,
    ) -> Path:
        voice_code = store.voice_code(provider=self.short_code, voice_id=task.voice.id)
        word_slug = _slug(task.prompt)
        filename_word = _filename_token(task.prompt)
        word_dir = data_dir / word_slug
        word_dir.mkdir(parents=True, exist_ok=True)
        output_path = word_dir / f"{filename_word}-{voice_code}-t100-clean-nonoise-nosnr.wav"

        existing = store.get(label=word_slug, filename=output_path.name)
        if existing is not None and not overwrite:
            audio_bytes = existing.get("audio_bytes")
            if isinstance(audio_bytes, bytes):
                output_path.write_bytes(audio_bytes)
            manifests.for_word_dir(word_dir).record(audio_path=output_path, label=word_slug)
            return output_path

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
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
            try:
                response.write_to_file(str(temp_path))
                audio_bytes = temp_path.read_bytes()
            finally:
                temp_path.unlink(missing_ok=True)

        store.upsert(
            build_generated_row(
                audio_bytes=audio_bytes,
                filename=output_path.name,
                label=word_slug,
                voice_id=task.voice.id,
                voice_code=voice_code,
                provider=self.short_code,
                lang=lang or task.voice.language,
            ),
            overwrite=overwrite,
        )

        output_path.write_bytes(audio_bytes)
        manifests.for_word_dir(word_dir).record(audio_path=output_path, label=word_slug)

        return output_path


class _GenerationTask:
    def __init__(self, *, prompt: str, voice: Voice) -> None:
        self.prompt = prompt
        self.voice = voice


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


def _filename_token(value: str) -> str:
    slug = "".join(ch.lower() for ch in value if ch.isalnum())
    return slug or "untitled"
