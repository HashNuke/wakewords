from __future__ import annotations

import hashlib
import io
import json
import math
import random
import struct
import subprocess
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from wakewords.lfs import require_materialized_files
from wakewords.parquet_store import CustomWordStore, build_augmented_row


DEFAULT_TEMPOS = (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)
DEFAULT_SNRS = (20, 10, 5)
DEFAULT_TARGET_SAMPLES_PER_WORD = 4000
DEFAULT_PARQUET_WRITE_BATCH_SIZE = 128
DEFAULT_CONTEXT_TARGET_DURATION_MS = 1000
DEFAULT_CONTEXT_MIN_GAP_MS = 10
DEFAULT_CONTEXT_MAX_GAP_MS = 100
DEFAULT_AUGMENT_SEED = 0


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    word: str
    provider: str
    voice_id: str
    duration: float
    duration_ms: int
    audio_bytes: bytes
    row: dict[str, object]


@dataclass(frozen=True)
class NoiseSample:
    path: Path
    duration: float
    duration_ms: int


@dataclass(frozen=True)
class DonorSample:
    sample_id: str
    word: str
    audio_bytes: bytes
    speech_start_ms: int
    speech_end_ms: int


@dataclass(frozen=True)
class ContextSlot:
    donor: DonorSample
    donor_offset_ms: int
    donor_duration_ms: int


@dataclass(frozen=True)
class ContextPlan:
    position: str
    before: ContextSlot | None
    after: ContextSlot | None
    before_gap_ms: int
    after_gap_ms: int
    reverse_donor: bool

    @property
    def primary_slot(self) -> ContextSlot:
        slot = self.before or self.after
        if slot is None:
            raise ValueError("Context plan has no donor slot")
        return slot

    @property
    def donor(self) -> DonorSample:
        return self.primary_slot.donor

    @property
    def donor_offset_ms(self) -> int:
        return self.primary_slot.donor_offset_ms

    @property
    def donor_duration_ms(self) -> int:
        return self.primary_slot.donor_duration_ms

    @property
    def context_gap_ms(self) -> int:
        return self.before_gap_ms + self.after_gap_ms


@dataclass(frozen=True)
class SpeechContextConfig:
    enabled: bool = False
    seed: int = DEFAULT_AUGMENT_SEED
    target_duration_ms: int = DEFAULT_CONTEXT_TARGET_DURATION_MS
    min_gap_ms: int = DEFAULT_CONTEXT_MIN_GAP_MS
    max_gap_ms: int = DEFAULT_CONTEXT_MAX_GAP_MS
    reverse_donor: bool = True


@dataclass(frozen=True)
class AugmentTask:
    source: SourceSample
    tempo: float
    noise: NoiseSample | None
    snr: int | None
    donors: tuple[DonorSample, ...] = ()
    speech_context: SpeechContextConfig = SpeechContextConfig()


def augment_dataset(
    *,
    data_dir: Path,
    noises_dir: Path,
    concurrency: int,
    overwrite: bool,
    tempos: tuple[float, ...] = DEFAULT_TEMPOS,
    snrs: tuple[int, ...] = DEFAULT_SNRS,
    target_samples_per_word: int = DEFAULT_TARGET_SAMPLES_PER_WORD,
    parquet_writes_batch_size: int = DEFAULT_PARQUET_WRITE_BATCH_SIZE,
) -> list[Path]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if parquet_writes_batch_size < 1:
        raise ValueError("parquet_writes_batch_size must be >= 1")

    parquet_path = data_dir / "custom_words.parquet"
    require_materialized_files(
        [parquet_path],
        context="augmenting custom words",
        include_hint="data/custom_words.parquet",
    )
    require_materialized_files(
        sorted(noises_dir.glob("*.wav")),
        context="augmenting with background audio",
        include_hint=f"{noises_dir}/*.wav",
    )
    store = CustomWordStore(parquet_path)
    config_path = data_dir.parent / "config.json"
    all_sources = _collect_sources(store)
    configured_labels = _load_configured_custom_labels(config_path)
    speech_context = _load_speech_context_config(config_path)
    sources = [source for source in all_sources if not configured_labels or source.word in configured_labels]
    if not sources:
        raise ValueError(f"No clean generated rows found in {parquet_path}.")
    donors = _collect_donors(all_sources) if speech_context.enabled and len({source.word for source in all_sources}) > 1 else []

    noises = _collect_noises(noises_dir)
    if not noises:
        raise ValueError(f"No noise wav files found under {noises_dir}.")

    tasks = _build_tasks(
        sources=sources,
        donors=donors,
        speech_context=speech_context,
        noises=noises,
        tempos=tempos,
        snrs=snrs,
        target_samples_per_word=target_samples_per_word,
    )
    outputs: list[Path] = []
    pending_rows: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_run_task, task=task, store=store, overwrite=overwrite)
            for task in tasks
        ]
        with tqdm(total=len(futures), unit="file") as bar:
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    pending_rows.append(row)
                    if _flush_pending_rows(
                        store,
                        pending_rows,
                        overwrite=overwrite,
                        batch_size=parquet_writes_batch_size,
                    ):
                        outputs.append(parquet_path)
                bar.update(1)

    if pending_rows and store.upsert_many(pending_rows, overwrite=overwrite):
        pending_rows.clear()
        outputs.append(parquet_path)

    return sorted(set(outputs))


def _flush_pending_rows(
    store: CustomWordStore,
    pending_rows: list[dict[str, object]],
    *,
    overwrite: bool,
    batch_size: int = DEFAULT_PARQUET_WRITE_BATCH_SIZE,
) -> bool:
    if len(pending_rows) < batch_size:
        return False
    changed = store.upsert_many(pending_rows, overwrite=overwrite)
    pending_rows.clear()
    return changed > 0


def _collect_sources(store: CustomWordStore) -> list[SourceSample]:
    sources: list[SourceSample] = []
    for row in store.rows():
        if row.get("source_type") != "generated":
            continue
        sample_id = row.get("sample_id")
        label = row.get("label")
        audio_bytes = row.get("audio_bytes")
        provider = row.get("provider")
        voice_id = row.get("voice_id")
        duration_ms = row.get("duration_ms")
        if not isinstance(sample_id, str) or not isinstance(label, str):
            continue
        if not isinstance(audio_bytes, bytes) or not isinstance(provider, str) or not isinstance(voice_id, str) or not isinstance(duration_ms, int):
            continue
        sources.append(
            SourceSample(
                sample_id=sample_id,
                word=label,
                provider=provider,
                voice_id=voice_id,
                duration=duration_ms / 1000,
                duration_ms=duration_ms,
                audio_bytes=audio_bytes,
                row=row,
            )
        )
    return sources


def _collect_noises(noises_dir: Path) -> list[NoiseSample]:
    manifest_durations = _load_noise_manifest(noises_dir / "manifest.jsonl")
    noises: list[NoiseSample] = []
    for wav_path in sorted(noises_dir.glob("*.wav")):
        duration_ms = manifest_durations.get(wav_path.name)
        if duration_ms is None:
            duration = _media_duration_seconds(wav_path)
            duration_ms = round(duration * 1000)
        else:
            duration = duration_ms / 1000
        noises.append(NoiseSample(path=wav_path, duration=duration, duration_ms=duration_ms))
    return noises


def _load_configured_custom_labels(config_path: Path) -> set[str]:
    if not config_path.exists():
        return set()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    custom_words = config.get("custom_words")
    if not isinstance(custom_words, list):
        return set()
    labels: set[str] = set()
    for word in custom_words:
        if isinstance(word, dict) and isinstance(word.get("label"), str):
            labels.add(word["label"])
    return labels


def _load_speech_context_config(config_path: Path) -> SpeechContextConfig:
    if not config_path.exists():
        return SpeechContextConfig()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    augment_config = config.get("augment")
    if not isinstance(augment_config, dict):
        return SpeechContextConfig()
    seed = augment_config.get("seed", DEFAULT_AUGMENT_SEED)
    if not isinstance(seed, int):
        raise ValueError("augment.seed must be an integer")
    context_config = augment_config.get("speech_context")
    if not isinstance(context_config, dict):
        return SpeechContextConfig(seed=seed)
    enabled = context_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("augment.speech_context.enabled must be a boolean")
    target_duration_ms = context_config.get("target_duration_ms", DEFAULT_CONTEXT_TARGET_DURATION_MS)
    if not isinstance(target_duration_ms, int) or target_duration_ms < 1:
        raise ValueError("augment.speech_context.target_duration_ms must be >= 1")
    gap_ms = context_config.get("gap_ms", [DEFAULT_CONTEXT_MIN_GAP_MS, DEFAULT_CONTEXT_MAX_GAP_MS])
    if not isinstance(gap_ms, list) or len(gap_ms) != 2 or not all(isinstance(value, int) for value in gap_ms):
        raise ValueError("augment.speech_context.gap_ms must be [min_ms, max_ms]")
    min_gap_ms, max_gap_ms = gap_ms
    if min_gap_ms < 0 or max_gap_ms < min_gap_ms:
        raise ValueError("augment.speech_context.gap_ms must be an increasing non-negative range")
    reverse_donor = context_config.get("reverse_donor", True)
    if not isinstance(reverse_donor, bool):
        raise ValueError("augment.speech_context.reverse_donor must be a boolean")
    return SpeechContextConfig(
        enabled=enabled,
        seed=seed,
        target_duration_ms=target_duration_ms,
        min_gap_ms=min_gap_ms,
        max_gap_ms=max_gap_ms,
        reverse_donor=reverse_donor,
    )


def _collect_donors(sources: list[SourceSample]) -> list[DonorSample]:
    donors: list[DonorSample] = []
    for source in sources:
        segment = _speech_segment_ms(source.audio_bytes)
        if segment is None:
            continue
        speech_start_ms, speech_end_ms = segment
        if speech_end_ms <= speech_start_ms:
            continue
        donors.append(
            DonorSample(
                sample_id=source.sample_id,
                word=source.word,
                audio_bytes=source.audio_bytes,
                speech_start_ms=speech_start_ms,
                speech_end_ms=speech_end_ms,
            )
        )
    return donors


def _load_noise_manifest(manifest_path: Path) -> dict[str, int]:
    if not manifest_path.exists():
        return {}
    durations: dict[str, int] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        audio = entry.get("audio")
        duration_ms = entry.get("duration_ms")
        if not isinstance(audio, str) or not isinstance(duration_ms, int):
            raise ValueError(f"Invalid noise manifest entry in {manifest_path}: {line}")
        durations[audio] = duration_ms
    return durations


def _build_tasks(
    *,
    sources: list[SourceSample],
    noises: list[NoiseSample],
    tempos: tuple[float, ...],
    snrs: tuple[int, ...],
    target_samples_per_word: int,
    donors: list[DonorSample] | None = None,
    speech_context: SpeechContextConfig = SpeechContextConfig(),
) -> list[AugmentTask]:
    if target_samples_per_word < 1:
        raise ValueError("target_samples_per_word must be >= 1")

    tasks: list[AugmentTask] = []
    sources_by_word: dict[str, list[SourceSample]] = {}
    for source in sources:
        sources_by_word.setdefault(source.word, []).append(source)

    for word_sources in sources_by_word.values():
        tempo_count, noise_count, snr_count = _combo_shape(
            voice_count=len(word_sources),
            target_samples_per_word=target_samples_per_word,
            tempos_available=len(tempos),
            noises_available=len(noises),
            snrs_available=len(snrs),
        )
        for source in word_sources:
            source_tempos = _select_subset(tempos, tempo_count, source=source, category="tempo")
            source_noises = _select_subset(tuple(noises), noise_count, source=source, category="noise")
            source_snrs = _select_subset(snrs, snr_count, source=source, category="snr")
            tasks.extend(_build_source_tasks(source, source_tempos, source_noises, source_snrs, tuple(donors or ()), speech_context))
    return tasks


def _build_source_tasks(
    source: SourceSample,
    source_tempos: tuple[float, ...],
    source_noises: tuple[NoiseSample, ...],
    source_snrs: tuple[int, ...],
    donors: tuple[DonorSample, ...] = (),
    speech_context: SpeechContextConfig = SpeechContextConfig(),
) -> list[AugmentTask]:
    tasks: list[AugmentTask] = []
    for tempo in source_tempos:
        for noise in source_noises:
            for snr in source_snrs:
                tasks.append(AugmentTask(source=source, tempo=tempo, noise=noise, snr=snr, donors=donors, speech_context=speech_context))
    return tasks


def _combo_shape(
    *,
    voice_count: int,
    target_samples_per_word: int,
    tempos_available: int,
    noises_available: int,
    snrs_available: int,
) -> tuple[int, int, int]:
    if voice_count < 1:
        raise ValueError("voice_count must be >= 1")

    target_combos_per_voice = max(1.0, (target_samples_per_word - voice_count) / voice_count)
    best_shape = (1, 1, 1)
    best_distance = abs(1 - target_combos_per_voice)

    for tempo_count in range(1, tempos_available + 1):
        for noise_count in range(1, noises_available + 1):
            for snr_count in range(1, snrs_available + 1):
                total = tempo_count * noise_count * snr_count
                distance = abs(total - target_combos_per_voice)
                shape = (tempo_count, noise_count, snr_count)
                if distance < best_distance or (distance == best_distance and _shape_score(shape) > _shape_score(best_shape)):
                    best_shape = shape
                    best_distance = distance
    return best_shape


def _shape_score(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    tempo_count, noise_count, snr_count = shape
    return (tempo_count, noise_count, snr_count)


def _select_subset[T](values: tuple[T, ...], count: int, *, source: SourceSample, category: str) -> tuple[T, ...]:
    rng = random.Random(_selection_seed(source=source, category=category))
    shuffled = list(values)
    rng.shuffle(shuffled)
    return tuple(shuffled[:count])


def _selection_seed(*, source: SourceSample, category: str) -> int:
    key = "|".join((source.word, source.provider, source.voice_id, category))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _run_task(*, task: AugmentTask, store: CustomWordStore, overwrite: bool) -> dict[str, object] | None:
    source_temp_path: Path | None = None
    speech_temp_path: Path | None = None
    context_temp_path: Path | None = None
    noise_temp_path: Path | None = None
    output_temp_path: Path | None = None
    try:
        source_temp_path = _new_temp_wav_path()
        source_temp_path.write_bytes(task.source.audio_bytes)
        speech_temp_path = _tempo_adjust(source_temp_path, task.tempo)
        context_plan = _context_plan(task=task, speech_path=speech_temp_path)
        speech_input_path = speech_temp_path
        if context_plan is not None:
            context_temp_path = _new_temp_wav_path()
            _compose_context_audio_ffmpeg(speech_path=speech_temp_path, context_plan=context_plan, output_path=context_temp_path)
            speech_input_path = context_temp_path

        existing = store.find_augmented(
            parent_sample_id=task.source.sample_id,
            tempo=task.tempo,
            noise_type=task.noise.path.stem if task.noise is not None else None,
            snr=task.snr,
            donor_sample_id=context_plan.donor.sample_id if context_plan is not None else None,
            donor_offset_ms=context_plan.donor_offset_ms if context_plan is not None else None,
            donor_duration_ms=context_plan.donor_duration_ms if context_plan is not None else None,
            context_position=context_plan.position if context_plan is not None else None,
            context_gap_ms=context_plan.context_gap_ms if context_plan is not None else None,
        )
        if existing is not None and not overwrite:
            return None

        if task.noise is None:
            audio_bytes = speech_input_path.read_bytes()
        else:
            duration = _media_duration_seconds(speech_input_path)
            noise_temp_path = _extract_noise_segment(
                source_key=f"{task.source.word}/{task.source.sample_id}",
                noise=task.noise,
                duration=duration,
                tempo=task.tempo,
                snr=task.snr or 0,
            )
            output_temp_path = _new_temp_wav_path()
            _mix_to_snr(
                speech_path=speech_input_path,
                noise_path=noise_temp_path,
                output_path=output_temp_path,
                snr_db=task.snr or 0,
            )
            audio_bytes = output_temp_path.read_bytes()

        return build_augmented_row(
            audio_bytes=audio_bytes,
            source_row=task.source.row,
            tempo=task.tempo,
            noise_type=task.noise.path.stem if task.noise is not None else None,
            snr=task.snr,
            donor_sample_id=context_plan.donor.sample_id if context_plan is not None else None,
            donor_offset_ms=context_plan.donor_offset_ms if context_plan is not None else None,
            donor_duration_ms=context_plan.donor_duration_ms if context_plan is not None else None,
            context_position=context_plan.position if context_plan is not None else None,
            context_gap_ms=context_plan.context_gap_ms if context_plan is not None else None,
        )
    finally:
        if source_temp_path is not None:
            source_temp_path.unlink(missing_ok=True)
        if speech_temp_path is not None:
            speech_temp_path.unlink(missing_ok=True)
        if context_temp_path is not None:
            context_temp_path.unlink(missing_ok=True)
        if noise_temp_path is not None:
            noise_temp_path.unlink(missing_ok=True)
        if output_temp_path is not None:
            output_temp_path.unlink(missing_ok=True)


def _context_plan(*, task: AugmentTask, speech_path: Path) -> ContextPlan | None:
    if not task.speech_context.enabled:
        return None
    donors = [donor for donor in task.donors if donor.word != task.source.word]
    if not donors:
        return None
    speech_duration_ms = round(_media_duration_seconds(speech_path) * 1000)
    rng = random.Random(_task_seed(task=task, category="speech-context"))
    remaining_ms = task.speech_context.target_duration_ms - speech_duration_ms
    if remaining_ms < 60:
        return None
    source_start_ms = rng.randint(0, remaining_ms)
    before_budget_ms = source_start_ms
    after_budget_ms = remaining_ms - source_start_ms
    before_gap_ms = _budgeted_gap_ms(rng, before_budget_ms, task.speech_context)
    after_gap_ms = _budgeted_gap_ms(rng, after_budget_ms, task.speech_context)
    before_duration_ms = before_budget_ms - before_gap_ms
    after_duration_ms = after_budget_ms - after_gap_ms
    before = _context_slot(donors=donors, duration_ms=before_duration_ms, rng=rng)
    after = _context_slot(donors=donors, duration_ms=after_duration_ms, rng=rng, exclude_sample_id=before.donor.sample_id if before else None)
    if before is None and after is None:
        return None
    if before is None:
        before_gap_ms = 0
    if after is None:
        after_gap_ms = 0
    position = "both" if before is not None and after is not None else "prepend" if before is not None else "append"
    return ContextPlan(
        position=position,
        before=before,
        after=after,
        before_gap_ms=before_gap_ms,
        after_gap_ms=after_gap_ms,
        reverse_donor=task.speech_context.reverse_donor,
    )


def _budgeted_gap_ms(rng: random.Random, budget_ms: int, config: SpeechContextConfig) -> int:
    if budget_ms <= config.min_gap_ms + 60:
        return 0
    return rng.randint(config.min_gap_ms, min(config.max_gap_ms, budget_ms - 60))


def _context_slot(
    *,
    donors: list[DonorSample],
    duration_ms: int,
    rng: random.Random,
    exclude_sample_id: str | None = None,
) -> ContextSlot | None:
    if duration_ms < 60:
        return None
    shuffled = [donor for donor in donors if donor.sample_id != exclude_sample_id]
    rng.shuffle(shuffled)
    for donor in shuffled:
        speech_duration = donor.speech_end_ms - donor.speech_start_ms
        if speech_duration < duration_ms:
            continue
        max_offset = speech_duration - duration_ms
        offset_within_speech = 0 if max_offset == 0 else rng.randint(0, max_offset)
        return ContextSlot(
            donor=donor,
            donor_offset_ms=donor.speech_start_ms + offset_within_speech,
            donor_duration_ms=duration_ms,
        )
    return None


def _task_seed(*, task: AugmentTask, category: str) -> int:
    key = "|".join(
        (
            task.source.word,
            task.source.sample_id,
            str(task.speech_context.seed),
            _tempo_label(task.tempo),
            task.noise.path.stem if task.noise is not None else "nonoise",
            f"snr{task.snr}" if task.snr is not None else "nosnr",
            category,
        )
    )
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _compose_context_audio(*, speech_path: Path, context_plan: ContextPlan, output_path: Path) -> None:
    speech_params, speech_samples = _read_wav_samples(speech_path)
    sample_rate = speech_params[2]
    samples: list[int] = []
    if context_plan.before is not None:
        samples.extend(_donor_slot_samples(context_plan.before, speech_params, reverse=context_plan.reverse_donor))
    if context_plan.before_gap_ms:
        samples.extend([0] * round(context_plan.before_gap_ms * sample_rate / 1000))
    samples.extend(speech_samples)
    if context_plan.after_gap_ms:
        samples.extend([0] * round(context_plan.after_gap_ms * sample_rate / 1000))
    if context_plan.after is not None:
        samples.extend(_donor_slot_samples(context_plan.after, speech_params, reverse=context_plan.reverse_donor))
    _write_wav_samples(output_path, speech_params, samples)


def _donor_slot_samples(slot: ContextSlot, params: tuple[int, int, int], *, reverse: bool) -> list[int]:
    donor_params, donor_samples = _read_wav_bytes_samples(slot.donor.audio_bytes)
    if params != donor_params:
        raise ValueError("Expected mono 16kHz 16-bit WAV source and donor audio for context augmentation.")
    sample_rate = params[2]
    donor_start = round(slot.donor_offset_ms * sample_rate / 1000)
    donor_count = round(slot.donor_duration_ms * sample_rate / 1000)
    samples = donor_samples[donor_start : donor_start + donor_count]
    return list(reversed(samples)) if reverse else samples


def _compose_context_audio_ffmpeg(*, speech_path: Path, context_plan: ContextPlan, output_path: Path) -> None:
    donor_paths: list[Path] = []
    try:
        inputs = ["-i", str(speech_path)]
        filters: list[str] = []
        concat_labels: list[str] = []
        next_input = 1

        def add_gap(duration_ms: int, label: str) -> None:
            if duration_ms <= 0:
                return
            filters.append(f"anullsrc=r=16000:cl=mono,atrim=duration={duration_ms / 1000:.6f},asetpts=PTS-STARTPTS[{label}]")
            concat_labels.append(f"[{label}]")

        def add_donor(slot: ContextSlot, label: str) -> None:
            nonlocal next_input
            path = _new_temp_wav_path()
            path.write_bytes(slot.donor.audio_bytes)
            donor_paths.append(path)
            inputs.extend(["-i", str(path)])
            reverse = ",areverse" if context_plan.reverse_donor else ""
            filters.append(
                f"[{next_input}:a]atrim=start={slot.donor_offset_ms / 1000:.6f}:duration={slot.donor_duration_ms / 1000:.6f},asetpts=PTS-STARTPTS{reverse},aformat=sample_rates=16000:sample_fmts=s16:channel_layouts=mono[{label}]"
            )
            concat_labels.append(f"[{label}]")
            next_input += 1

        if context_plan.before is not None:
            add_donor(context_plan.before, "before")
        add_gap(context_plan.before_gap_ms, "beforegap")
        filters.append("[0:a]aformat=sample_rates=16000:sample_fmts=s16:channel_layouts=mono[source]")
        concat_labels.append("[source]")
        add_gap(context_plan.after_gap_ms, "aftergap")
        if context_plan.after is not None:
            add_donor(context_plan.after, "after")
        filters.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=0:a=1,aformat=sample_rates=16000:sample_fmts=s16:channel_layouts=mono[out]")
        _run_ffmpeg([
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ])
    finally:
        for path in donor_paths:
            path.unlink(missing_ok=True)


def _read_wav_bytes_samples(audio_bytes: bytes) -> tuple[tuple[int, int, int], list[int]]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        params = (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        )
        if params[0] != 1 or params[1] != 2 or wav_file.getcomptype() != "NONE":
            raise ValueError("Unsupported donor wav format; expected mono PCM16 WAV.")
        frames = wav_file.readframes(wav_file.getnframes())
    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
    return params, samples


def _write_wav_samples(path: Path, params: tuple[int, int, int], samples: list[int]) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(params[0])
        wav_file.setsampwidth(params[1])
        wav_file.setframerate(params[2])
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))


_SILERO_MODEL = None
_SILERO_LOCK = threading.Lock()


def _speech_segment_ms(audio_bytes: bytes) -> tuple[int, int] | None:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:  # pragma: no cover - dependency is installed with package deps
        raise RuntimeError("silero-vad and torch are required for donor speech context augmentation.") from exc

    global _SILERO_MODEL
    with _SILERO_LOCK:
        if _SILERO_MODEL is None:
            _SILERO_MODEL = load_silero_vad()
        model = _SILERO_MODEL

    params, samples = _read_wav_bytes_samples(audio_bytes)
    waveform = torch.asarray(samples, dtype=torch.float32) / 32768.0
    with _SILERO_LOCK:
        timestamps = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=params[2],
            threshold=0.5,
            min_speech_duration_ms=60,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
        )
    if not timestamps:
        return None
    return round(timestamps[0]["start"] / params[2] * 1000), round(timestamps[-1]["end"] / params[2] * 1000)


def _tempo_adjust(source_path: Path, tempo: float) -> Path:
    temp_path = _new_temp_wav_path()
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-filter:a",
        f"atempo={tempo}",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ])
    return temp_path


def _extract_noise_segment(
    *,
    source_key: str,
    noise: NoiseSample,
    duration: float,
    tempo: float,
    snr: int,
) -> Path:
    max_start = max(noise.duration - duration, 0.0)
    start = 0.0 if max_start == 0 else _deterministic_fraction(source_key, noise.path, tempo, snr) * max_start

    temp_path = _new_temp_wav_path()
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(noise.path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ])
    return temp_path


def _mix_to_snr(*, speech_path: Path, noise_path: Path, output_path: Path, snr_db: int) -> None:
    speech_params, speech_samples = _read_wav_samples(speech_path)
    noise_params, noise_samples = _read_wav_samples(noise_path)
    if speech_params != noise_params:
        raise ValueError("Expected mono 16kHz 16-bit WAV inputs for augmentation.")

    speech_rms = _rms(speech_samples)
    noise_rms = _rms(noise_samples)
    target_noise_rms = 0.0 if speech_rms == 0 else speech_rms / (10 ** (snr_db / 20))
    scale = 0.0 if noise_rms == 0 else target_noise_rms / noise_rms

    speech_frame_count = len(speech_samples)
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(speech_path),
        "-i",
        str(noise_path),
        "-filter_complex",
        (
            f"[0:a]atrim=end_sample={speech_frame_count}[speech];"
            f"[1:a]volume={scale:.12f}:precision=double,apad,atrim=end_sample={speech_frame_count}[scaled];"
            "[speech][scaled]amix=inputs=2:normalize=0:duration=first,"
            "aformat=sample_rates=16000:sample_fmts=s16:channel_layouts=mono[out]"
        ),
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ])


def _read_wav_samples(path: Path) -> tuple[tuple[int, int, int], list[int]]:
    with wave.open(str(path), "rb") as wav_file:
        params = (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        )
        if params[0] != 1 or params[1] != 2 or wav_file.getcomptype() != "NONE":
            raise ValueError(f"Unsupported wav format for {path}; expected mono PCM16 WAV.")
        frames = wav_file.readframes(wav_file.getnframes())
    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
    return params, samples


def _media_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _tempo_label(tempo: float) -> str:
    return f"t{int(round(tempo * 100)):03d}"


def _deterministic_fraction(source_key: str, noise_path: Path, tempo: float, snr: int) -> float:
    key = "|".join((source_key, str(noise_path), _tempo_label(tempo), f"snr{snr:02d}"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _new_temp_wav_path() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        return Path(temp_file.name)
