from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from wakewords.parquet_store import CustomWordStore, build_generated_row
from wakewords.providers.base import GeneratedAudioContext, GenerationPrompt, TTSProvider, Voice, VoiceSelectionConfig, prepare_generated_audio


def generate_audio(
    *,
    provider: TTSProvider,
    prompts: list[GenerationPrompt],
    parquet_path: Path,
    voice: str | None,
    voices: int | None,
    all_voices: bool,
    lang: str | None,
    voice_selection: VoiceSelectionConfig | None,
    concurrency: int,
    model_id: str,
    sample_rate: int,
    encoding: str,
    overwrite: bool,
) -> list[Path]:
    selected_voices = select_voices(
        provider=provider,
        voice=voice,
        voices=voices,
        all_voices=all_voices,
        lang=lang,
        voice_selection=voice_selection,
    )
    provider_code = _provider_code(provider)
    store = CustomWordStore(parquet_path)
    voice_codes = {
        selected_voice.id: store.voice_code(provider=provider_code, voice_id=selected_voice.id)
        for selected_voice in selected_voices
    }
    tasks = build_tasks(prompts=prompts, voices=selected_voices, voice_codes=voice_codes, provider=provider_code)

    wrote_rows = False

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _generate_one,
                provider=provider,
                provider_code=provider_code,
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


def select_voices(
    *,
    provider: TTSProvider,
    voice: str | None,
    voices: int | None,
    all_voices: bool,
    lang: str | None,
    voice_selection: VoiceSelectionConfig | None = None,
) -> list[Voice]:
    if voices is not None and voices < 1:
        raise ValueError("voices must be >= 1")
    if voice and voices is not None:
        raise ValueError("Use --voice for one specific voice or --voices to limit selected voices, not both.")
    if voice_selection is not None:
        if voice or voices is not None or all_voices or lang:
            raise ValueError("Use CLI voice options or generate.voice_selection, not both.")
        return _select_grouped_voices(provider=provider, voice_selection=voice_selection)

    if all_voices:
        selected_voices = provider.list_voices(all=True, lang=lang)
        if not selected_voices:
            raise RuntimeError(f"{provider.name} returned no voices.")
        return selected_voices[:voices]

    if voice:
        selected_voices = provider.list_voices(all=True, lang=lang)
        if not selected_voices:
            raise RuntimeError(f"{provider.name} returned no voices.")
        normalized = voice.strip().lower()
        for candidate in selected_voices:
            if candidate.id.lower() == normalized:
                return [candidate]
            if candidate.name and candidate.name.lower() == normalized:
                return [candidate]
        raise ValueError(f"Could not find {provider.name} voice by id or name: {voice}")

    selected_voices = provider.list_voices(pages=1, all=False, lang=lang)
    if not selected_voices:
        raise RuntimeError(f"{provider.name} returned no voices.")
    return selected_voices[: voices or 1]


def build_tasks(*, prompts: list[GenerationPrompt], voices: list[Voice], voice_codes: dict[str, str], provider: str) -> list["GenerationTask"]:
    tasks: list[GenerationTask] = []
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
                GenerationTask(
                    prompt=prompt.tts_input,
                    voice=selected_voice,
                    voice_code=voice_code,
                    word_slug=word_slug,
                    sample_id=sample_id,
                )
            )
    return tasks


class GenerationTask:
    def __init__(self, *, prompt: str, voice: Voice, voice_code: str, word_slug: str, sample_id: str) -> None:
        self.prompt = prompt
        self.voice = voice
        self.voice_code = voice_code
        self.word_slug = word_slug
        self.sample_id = sample_id


def _select_grouped_voices(*, provider: TTSProvider, voice_selection: VoiceSelectionConfig) -> list[Voice]:
    if set(voice_selection.group_by) != {"language", "gender"}:
        raise ValueError('Only grouping by ["language", "gender"] is supported.')

    selected_voices: list[Voice] = []
    genders = tuple(gender.lower() for gender in voice_selection.genders)
    gender_set = set(genders)

    if voice_selection.languages == "all":
        candidates: list[Voice] = []
        for gender in genders:
            candidates.extend(provider.list_voices(all=True, gender=gender))
        language_groups: dict[str, list[Voice]] = {}
        for candidate in candidates:
            if candidate.language:
                language_groups.setdefault(candidate.language, []).append(candidate)
        language_voices = language_groups.values()
    else:
        language_voices = []
        for language in voice_selection.languages:
            candidates = []
            for gender in genders:
                candidates.extend(provider.list_voices(all=True, lang=language, gender=gender))
            language_voices.append(candidates)

    for candidates in language_voices:
        group_counts: dict[tuple[str, str], int] = {}
        for candidate in candidates:
            if not candidate.language or not candidate.gender:
                continue
            gender = candidate.gender.lower()
            if gender not in gender_set:
                continue
            group_key = (candidate.language, gender)
            count = group_counts.get(group_key, 0)
            if count >= voice_selection.limit_per_group:
                continue
            selected_voices.append(candidate)
            group_counts[group_key] = count + 1

    if not selected_voices:
        raise RuntimeError(f"{provider.name} returned no voices matching generate.voice_selection.")
    return selected_voices


def _generate_one(
    *,
    provider: TTSProvider,
    provider_code: str,
    task: GenerationTask,
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

    generation_language = lang or task.voice.language
    audio_bytes = provider.generate(
        prompt=task.prompt,
        voice=task.voice,
        lang=generation_language,
        model_id=model_id,
        sample_rate=sample_rate,
        encoding=encoding,
    )
    audio_bytes = prepare_generated_audio(
        audio_bytes,
        context=GeneratedAudioContext(
            prompt=task.prompt,
            label=task.word_slug,
            provider=provider_code,
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
            provider=provider_code,
            lang=generation_language,
        ),
        overwrite=overwrite,
    )

    return True


def _provider_code(provider: TTSProvider) -> str:
    return str(getattr(provider, "short_code", provider.name))


def _generated_sample_id(*, label: str, provider: str, voice_id: str) -> str:
    import hashlib

    return hashlib.sha256(f"generated\0{label}\0{provider}\0{voice_id}".encode("utf-8")).hexdigest()
