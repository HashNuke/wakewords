from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import subprocess
import tempfile
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
class AugmentTask:
    source: SourceSample
    tempo: float
    noise: NoiseSample | None
    snr: int | None


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
    sources = _collect_sources(store)
    if not sources:
        raise ValueError(f"No clean generated rows found in {parquet_path}.")

    noises = _collect_noises(noises_dir)
    if not noises:
        raise ValueError(f"No noise wav files found under {noises_dir}.")

    tasks = _build_tasks(
        sources=sources,
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
            tasks.extend(_build_source_tasks(source, source_tempos, source_noises, source_snrs))
    return tasks


def _build_source_tasks(
    source: SourceSample,
    source_tempos: tuple[float, ...],
    source_noises: tuple[NoiseSample, ...],
    source_snrs: tuple[int, ...],
) -> list[AugmentTask]:
    tasks: list[AugmentTask] = []
    for tempo in source_tempos:
        for noise in source_noises:
            for snr in source_snrs:
                tasks.append(AugmentTask(source=source, tempo=tempo, noise=noise, snr=snr))
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
    existing = store.find_augmented(
        parent_sample_id=task.source.sample_id,
        tempo=task.tempo,
        noise_type=task.noise.path.stem if task.noise is not None else None,
        snr=task.snr,
    )
    if existing is not None and not overwrite:
        return None

    source_temp_path: Path | None = None
    speech_temp_path: Path | None = None
    noise_temp_path: Path | None = None
    output_temp_path: Path | None = None
    try:
        source_temp_path = _new_temp_wav_path()
        source_temp_path.write_bytes(task.source.audio_bytes)
        speech_temp_path = _tempo_adjust(source_temp_path, task.tempo)
        if task.noise is None:
            audio_bytes = speech_temp_path.read_bytes()
        else:
            duration = task.source.duration / task.tempo
            noise_temp_path = _extract_noise_segment(
                source_key=f"{task.source.word}/{task.source.sample_id}",
                noise=task.noise,
                duration=duration,
                tempo=task.tempo,
                snr=task.snr or 0,
            )
            output_temp_path = _new_temp_wav_path()
            _mix_to_snr(
                speech_path=speech_temp_path,
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
        )
    finally:
        if source_temp_path is not None:
            source_temp_path.unlink(missing_ok=True)
        if speech_temp_path is not None:
            speech_temp_path.unlink(missing_ok=True)
        if noise_temp_path is not None:
            noise_temp_path.unlink(missing_ok=True)
        if output_temp_path is not None:
            output_temp_path.unlink(missing_ok=True)
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
