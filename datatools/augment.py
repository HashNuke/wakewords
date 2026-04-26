from __future__ import annotations

import hashlib
import math
import struct
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from datatools.manifest import ManifestStore
from tqdm import tqdm


DEFAULT_TEMPOS = (0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.25)
DEFAULT_SNRS = (20, 10, 5)
_BASE_SUFFIX = "-t100-clean-nonoise-nosnr.wav"


@dataclass(frozen=True)
class SourceSample:
    path: Path
    word: str
    voice_code: str
    duration: float
    duration_ms: int


@dataclass(frozen=True)
class AugmentTask:
    source: SourceSample
    tempo: float
    noise: Path | None
    snr: int | None

    @property
    def output_path(self) -> Path:
        tempo_label = _tempo_label(self.tempo)
        if self.noise is None:
            filename = f"{self.source.word}-{self.source.voice_code}-{tempo_label}-clean-nonoise-nosnr.wav"
        else:
            noise_name = _slug(self.noise.stem)
            filename = f"{self.source.word}-{self.source.voice_code}-{tempo_label}-{noise_name}-{noise_name}-snr{self.snr:02d}.wav"
        return self.source.path.parent / filename


def augment_dataset(
    *,
    data_dir: Path,
    noises_dir: Path,
    concurrency: int,
    overwrite: bool,
    tempos: tuple[float, ...] = DEFAULT_TEMPOS,
    snrs: tuple[int, ...] = DEFAULT_SNRS,
) -> list[Path]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    manifests = ManifestStore()
    sources = _collect_sources(data_dir, manifests)
    if not sources:
        raise ValueError(f"No clean generated files found under {data_dir}.")

    noises = sorted(noises_dir.glob("*.wav"))
    if not noises:
        raise ValueError(f"No noise wav files found under {noises_dir}.")

    tasks = _build_tasks(sources=sources, noises=noises, tempos=tempos, snrs=snrs)
    outputs: list[Path] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_run_task, task=task, manifests=manifests, overwrite=overwrite)
            for task in tasks
        ]
        with tqdm(total=len(futures), unit="file") as bar:
            for future in as_completed(futures):
                outputs.append(future.result())
                bar.update(1)

    return sorted(outputs)


def _collect_sources(data_dir: Path, manifests: ManifestStore) -> list[SourceSample]:
    sources: list[SourceSample] = []
    for wav_path in sorted(data_dir.glob("*/*.wav")):
        if wav_path.parent.name == "_noises_":
            continue
        parsed = _parse_source_filename(wav_path.name)
        if parsed is None:
            continue
        word, voice_code = parsed
        manifest = manifests.for_word_dir(wav_path.parent)
        entry = manifest.get(wav_path)
        if entry is None:
            entry = manifest.record(audio_path=wav_path, label=word)
        sources.append(
            SourceSample(
                path=wav_path,
                word=word,
                voice_code=voice_code,
                duration=float(entry["duration"]),
                duration_ms=int(entry["duration_ms"]),
            )
        )
    return sources


def _build_tasks(
    *,
    sources: list[SourceSample],
    noises: list[Path],
    tempos: tuple[float, ...],
    snrs: tuple[int, ...],
) -> list[AugmentTask]:
    tasks: list[AugmentTask] = []
    for source in sources:
        for tempo in tempos:
            tasks.append(AugmentTask(source=source, tempo=tempo, noise=None, snr=None))
        for tempo in (1.0, *tempos):
            for noise in noises:
                for snr in snrs:
                    tasks.append(AugmentTask(source=source, tempo=tempo, noise=noise, snr=snr))
    return tasks


def _run_task(*, task: AugmentTask, manifests: ManifestStore, overwrite: bool) -> Path:
    output_path = task.output_path
    manifest = manifests.for_word_dir(output_path.parent)
    if output_path.exists() and not overwrite:
        manifest.record(audio_path=output_path, label=task.source.word)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    speech_temp_path: Path | None = None
    noise_temp_path: Path | None = None
    try:
        speech_temp_path = _tempo_adjust(task.source.path, task.tempo)
        if task.noise is None:
            output_path.write_bytes(speech_temp_path.read_bytes())
            manifest.record(audio_path=output_path, label=task.source.word)
            return output_path

        duration = task.source.duration / task.tempo
        noise_temp_path = _extract_noise_segment(
            source_path=task.source.path,
            noise_path=task.noise,
            duration=duration,
            tempo=task.tempo,
            snr=task.snr or 0,
        )
        _mix_to_snr(
            speech_path=speech_temp_path,
            noise_path=noise_temp_path,
            output_path=output_path,
            snr_db=task.snr or 0,
        )
        manifest.record(audio_path=output_path, label=task.source.word)
        return output_path
    finally:
        if speech_temp_path is not None:
            speech_temp_path.unlink(missing_ok=True)
        if noise_temp_path is not None:
            noise_temp_path.unlink(missing_ok=True)


def _parse_source_filename(filename: str) -> tuple[str, str] | None:
    if not filename.endswith(_BASE_SUFFIX):
        return None
    stem = filename[: -len(_BASE_SUFFIX)]
    if "-" not in stem:
        return None
    word, voice_code = stem.rsplit("-", 1)
    if not word or not voice_code:
        return None
    return word, voice_code


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
    source_path: Path,
    noise_path: Path,
    duration: float,
    tempo: float,
    snr: int,
) -> Path:
    noise_duration = _media_duration_seconds(noise_path)
    max_start = max(noise_duration - duration, 0.0)
    start = 0.0 if max_start == 0 else _deterministic_fraction(source_path, noise_path, tempo, snr) * max_start

    temp_path = _new_temp_wav_path()
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(noise_path),
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

    if len(noise_samples) < len(speech_samples):
        noise_samples.extend([0] * (len(speech_samples) - len(noise_samples)))
    elif len(noise_samples) > len(speech_samples):
        noise_samples = noise_samples[: len(speech_samples)]

    speech_rms = _rms(speech_samples)
    noise_rms = _rms(noise_samples)
    target_noise_rms = 0.0 if speech_rms == 0 else speech_rms / (10 ** (snr_db / 20))
    scale = 0.0 if noise_rms == 0 else target_noise_rms / noise_rms

    mixed = [_clamp_int16(s + int(round(n * scale))) for s, n in zip(speech_samples, noise_samples, strict=False)]
    _write_wav_samples(output_path, speech_params, mixed)


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


def _write_wav_samples(path: Path, params: tuple[int, int, int], samples: list[int]) -> None:
    channels, sample_width, frame_rate = params
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))


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


def _slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "untitled"


def _deterministic_fraction(source_path: Path, noise_path: Path, tempo: float, snr: int) -> float:
    key = "|".join((str(source_path), str(noise_path), _tempo_label(tempo), f"snr{snr:02d}"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))


def _new_temp_wav_path() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        return Path(temp_file.name)
