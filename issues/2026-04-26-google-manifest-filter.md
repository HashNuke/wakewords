Status: DONE

# Google Manifest Filter

## Problem

The `wakewords manifest` command only consumed per-word manifests from `data/`, so downloaded Google Speech Commands word files were not included in generated train, validation, and test manifests. The command also needed to avoid treating files under the Google dataset's `_background_noise_` directory as labeled speech-command examples.

## Solution

Manifest generation now reads `config.json` and includes Google Speech Commands audio only when `google_speech_commands` is defined with at least one word. It scans only the configured word directories under `google-speech-commands/`, skips `_background_noise_` even if it is configured, and sends those entries through the same per-label split logic used for other samples so each word follows the requested train/validation/test ratio.
