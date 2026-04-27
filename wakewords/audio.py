from __future__ import annotations

import io
import logging
import struct
import threading
import wave
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_SPEECH_PADDING_MS = 200
DEFAULT_MIN_TRIM_DURATION_MS = 1200
_MODEL_LOCK = threading.Lock()


def trim_wav_to_speech(
    audio_bytes: bytes,
    *,
    padding_ms: int = DEFAULT_SPEECH_PADDING_MS,
    min_duration_ms: int = DEFAULT_MIN_TRIM_DURATION_MS,
) -> bytes:
    try:
        wav_data = _read_pcm16_mono_wav(audio_bytes)
    except ValueError as exc:
        logger.warning("Generated audio is not compatible with Silero VAD trimming: %s", exc)
        return audio_bytes

    duration_ms = round(len(wav_data.samples) / wav_data.sample_rate * 1000)
    if duration_ms <= min_duration_ms:
        return audio_bytes

    try:
        timestamps = _speech_timestamps(wav_data.samples, sample_rate=wav_data.sample_rate)
    except Exception:
        logger.exception("Silero VAD trimming failed; keeping original generated audio.")
        return audio_bytes

    if not timestamps:
        logger.warning("Silero VAD detected no speech in generated audio; keeping original audio.")
        return audio_bytes

    try:
        speech_start = min(_timestamp_frame(timestamp, "start") for timestamp in timestamps)
        speech_end = max(_timestamp_frame(timestamp, "end") for timestamp in timestamps)
    except ValueError:
        logger.exception("Silero VAD returned invalid speech timestamps; keeping original audio.")
        return audio_bytes

    if speech_start < 0 or speech_end <= speech_start:
        logger.warning("Silero VAD returned invalid speech timestamps; keeping original audio.")
        return audio_bytes

    padding_frames = round(wav_data.sample_rate * padding_ms / 1000)
    trim_start = max(speech_start - padding_frames, 0)
    trim_end = min(speech_end + padding_frames, len(wav_data.samples))
    if trim_start == 0 and trim_end == len(wav_data.samples):
        return audio_bytes

    return _write_pcm16_mono_wav(
        wav_data.samples[trim_start:trim_end],
        sample_rate=wav_data.sample_rate,
        sample_width=wav_data.sample_width,
    )


class _WavData:
    def __init__(self, *, samples: list[int], sample_rate: int, sample_width: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.sample_width = sample_width


def _read_pcm16_mono_wav(audio_bytes: bytes) -> _WavData:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        compression = wav_file.getcomptype()
    if channels != 1 or sample_width != 2 or compression != "NONE":
        raise ValueError("Generated audio must be mono PCM16 WAV for Silero VAD trimming.")
    if sample_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports generated audio at 8000 Hz or 16000 Hz.")

    payload = _wav_data_payload(audio_bytes)
    if len(payload) % sample_width != 0:
        payload = payload[: -(len(payload) % sample_width)]
    samples = list(struct.unpack(f"<{len(payload) // sample_width}h", payload))
    return _WavData(samples=samples, sample_rate=sample_rate, sample_width=sample_width)


def _wav_data_payload(audio_bytes: bytes) -> bytes:
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE audio bytes")

    offset = 12
    while offset + 8 <= len(audio_bytes):
        chunk_id = audio_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio_bytes, offset + 4)[0]
        payload_start = offset + 8
        if chunk_id == b"data":
            if chunk_size == 0xFFFFFFFF or payload_start + chunk_size > len(audio_bytes):
                return audio_bytes[payload_start:]
            return audio_bytes[payload_start : payload_start + chunk_size]
        offset = payload_start + chunk_size + (chunk_size % 2)

    raise ValueError("WAV bytes are missing a data chunk")


def _write_pcm16_mono_wav(samples: list[int], *, sample_rate: int, sample_width: int) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return buffer.getvalue()


def _speech_timestamps(samples: list[int], *, sample_rate: int) -> list[dict[str, int]]:
    import torch
    from silero_vad import get_speech_timestamps

    waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
    with _MODEL_LOCK:
        model = _silero_model()
        timestamps = get_speech_timestamps(waveform, model, sampling_rate=sample_rate)
    return [timestamp for timestamp in timestamps if isinstance(timestamp, dict)]


@lru_cache(maxsize=1)
def _silero_model() -> object:
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _timestamp_frame(timestamp: dict[str, int], key: str) -> int:
    value = timestamp.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Silero VAD timestamp missing integer {key!r}")
    return value
