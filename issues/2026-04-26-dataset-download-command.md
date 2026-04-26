# Dataset Download Command

Status: DONE

## Problem

Dataset expansion required a support command for downloading external datasets.
The project had dataset notes for Google Speech Commands and the Common Voice
7.0 single-word target segment, but no CLI command to download and extract them
into the expected local `data/` layout.

## Solution

Added `uv run datatools download` with dataset switches for
`--google-speech-commands`, `--common-voice-sw`, and `--all`. Downloads show a
progress bar, extraction shows a separate progress bar, default downloads use a
temporary local directory that is deleted after extraction, and
`--downloads-dir` can retain downloaded archives while still extracting to the
fixed dataset directories.

Usage is documented in `USAGE.md`.
