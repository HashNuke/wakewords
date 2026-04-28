from __future__ import annotations

import importlib
import io
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _load_augment_module():
    original_augment = sys.modules.get("wakewords.augment")
    original_parquet_store = sys.modules.get("wakewords.parquet_store")
    original_tqdm = sys.modules.get("tqdm")

    fake_parquet_store = types.ModuleType("wakewords.parquet_store")
    fake_parquet_store.CustomWordStore = object
    fake_parquet_store.build_augmented_row = lambda *args, **kwargs: None

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable

    sys.modules["wakewords.parquet_store"] = fake_parquet_store
    sys.modules["tqdm"] = fake_tqdm
    sys.modules.pop("wakewords.augment", None)

    try:
        yield importlib.import_module("wakewords.augment")
    finally:
        _restore_module("wakewords.augment", original_augment)
        _restore_module("wakewords.parquet_store", original_parquet_store)
        _restore_module("tqdm", original_tqdm)


def _restore_module(name: str, module: types.ModuleType | None) -> None:
    if module is None:
        sys.modules.pop(name, None)
        return
    sys.modules[name] = module


class AugmentMixEquivalenceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for mix equivalence test")
    def test_ffmpeg_filter_mix_matches_python_mix(self) -> None:
        with _load_augment_module() as augment, tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            speech_path = tmp_path / "speech.wav"
            noise_path = tmp_path / "noise.wav"
            python_output = tmp_path / "python-mix.wav"
            ffmpeg_output = tmp_path / "ffmpeg-mix.wav"

            speech_path.write_bytes(_wav_bytes_from_samples(_speech_samples()))
            noise_path.write_bytes(_wav_bytes_from_samples(_noise_samples()))

            augment._mix_to_snr(
                speech_path=speech_path,
                noise_path=noise_path,
                output_path=python_output,
                snr_db=10,
            )
            _mix_to_snr_with_ffmpeg(
                augment,
                speech_path=speech_path,
                noise_path=noise_path,
                output_path=ffmpeg_output,
                snr_db=10,
            )

            python_params, python_samples = augment._read_wav_samples(python_output)
            ffmpeg_params, ffmpeg_samples = augment._read_wav_samples(ffmpeg_output)

            self.assertEqual(ffmpeg_params, python_params)
            self.assertEqual(
                ffmpeg_samples,
                python_samples,
                msg=_sample_mismatch_message(python_samples, ffmpeg_samples),
            )


def _wav_bytes_from_samples(samples: list[int], *, sample_rate: int = 16000) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return buffer.getvalue()


def _speech_samples(*, sample_rate: int = 16000, duration_s: float = 0.5) -> list[int]:
    frame_count = int(sample_rate * duration_s)
    return [
        int(round(9000 * math.sin(2 * math.pi * 440 * index / sample_rate)))
        for index in range(frame_count)
    ]


def _noise_samples(*, sample_rate: int = 16000, duration_s: float = 0.5) -> list[int]:
    frame_count = int(sample_rate * duration_s)
    samples: list[int] = []
    for index in range(frame_count):
        sine = 3500 * math.sin(2 * math.pi * 113 * index / sample_rate)
        square = 700 if (index // 31) % 2 == 0 else -700
        saw = ((index * 97) % 2000) - 1000
        samples.append(int(round(sine + square + saw / 2)))
    return samples


def _mix_to_snr_with_ffmpeg(
    augment_module,
    *,
    speech_path: Path,
    noise_path: Path,
    output_path: Path,
    snr_db: int,
) -> None:
    speech_params, speech_samples = augment_module._read_wav_samples(speech_path)
    noise_params, noise_samples = augment_module._read_wav_samples(noise_path)
    if speech_params != noise_params:
        raise ValueError("Expected matching WAV params in ffmpeg mix test helper.")

    speech_rms = augment_module._rms(speech_samples)
    noise_rms = augment_module._rms(noise_samples)
    target_noise_rms = 0.0 if speech_rms == 0 else speech_rms / (10 ** (snr_db / 20))
    scale = 0.0 if noise_rms == 0 else target_noise_rms / noise_rms
    speech_frame_count = len(speech_samples)

    subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _sample_mismatch_message(expected: list[int], actual: list[int]) -> str:
    if len(expected) != len(actual):
        return f"Length mismatch: expected {len(expected)} samples, got {len(actual)}"

    differing = [
        (index, left, right)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        if left != right
    ]
    if not differing:
        return "No mismatch"

    first_index, expected_sample, actual_sample = differing[0]
    max_delta = max(abs(left - right) for _, left, right in differing)
    return (
        f"Found {len(differing)} differing samples; first mismatch at index {first_index}: "
        f"expected {expected_sample}, got {actual_sample}; max abs delta {max_delta}"
    )


if __name__ == "__main__":
    unittest.main()
