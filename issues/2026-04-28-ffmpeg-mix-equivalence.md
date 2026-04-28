Status: DONE

# FFMPEG Mix Equivalence

## Problem

We wanted to move augmentation mixing from Python into a single `ffmpeg -filter_complex` pipeline to reduce temporary files and avoid Python-side sample loops.

Before changing production code, we needed an isolated verification that the proposed `ffmpeg` path could generate the same PCM output as the original `_mix_to_snr` implementation.

An initial direct `volume + amix` comparison was not bit-identical. It produced only a few sample differences, but that was enough to show the naive filter setup was not an exact replacement.

## Solution

Added an isolated test in `tests/test_augment_mix_equivalence.py` that:

1. Stubs unrelated Parquet and progress-bar dependencies so the test can import `wakewords.augment` without the full data stack.
2. Generates deterministic speech and noise WAV fixtures.
3. Mixes them once with the current Python `_mix_to_snr` path.
4. Mixes them again with `ffmpeg -filter_complex` using `volume` and `amix`.
5. Compares decoded PCM samples for exact equality.

The key finding was that `ffmpeg` matches the Python implementation sample-for-sample when the noise scaling step uses:

`volume=<scale>:precision=double`

without that precision override, the comparison drifted by a few 1-LSB samples.

After verifying equivalence, production augmentation mixing was updated in `wakewords/augment.py` to replace the Python sample loop with the same `ffmpeg` approach. The production path now:

1. Computes RMS and the target noise scale in Python, as before.
2. Uses a single `ffmpeg -filter_complex` command for the actual audio mix.
3. Forces `volume` to `precision=double`.
4. Uses `apad` plus `atrim=end_sample=<speech_len>` so the noise input matches the speech length exactly.
5. Mixes with `amix=inputs=2:normalize=0:duration=first` and writes PCM16 output.

The old Python-side sample addition, clamping, and WAV rewrite path was removed.

## Result

The isolated equivalence test passes, the augmentation unit tests pass, and production augmentation now uses the verified `ffmpeg` mixing path.
