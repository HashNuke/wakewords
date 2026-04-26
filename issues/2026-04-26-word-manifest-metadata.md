Status: DONE

Problem

Generated and augmented files did not have a per-word manifest that captured NeMo-style metadata plus millisecond duration. That forced later workflows to re-probe files for basic details and gave augmentation no structured source metadata to reuse.

Solution

Added per-word `manifest.jsonl` files under `data/<word>/`. Generation now probes each saved WAV and records `audio_filepath`, `duration`, `duration_ms`, and `label`. Augmentation now reads clean-source metadata from that manifest when available, falls back to probing older files if needed, and separately probes each augmented output before recording it so any tempo-related duration drift is reflected accurately.
