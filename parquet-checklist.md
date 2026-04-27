# Parquet Migration Checklist

- [ ] Add a new Parquet storage module, likely `wakewords/parquet_store.py`.
  Details: Define the schema for `data/custom_words.parquet`, row validation, file read/write helpers, full-file rewrite helpers, row append helpers, lookup helpers by `label` and `source_type`, and WAV byte serialization helpers. Keep the first version simple and single-file based.

- [ ] Define the canonical row schema for custom-word samples.
  Details: Include `sample_id`, `filename`, `label`, `audio_bytes`, `voice_id`, `provider`, `duration_ms`, `lang`, `sample_rate`, `channels`, `source_type`, `parent_sample_id`, `tempo`, `noise_type`, `snr`, `created_at`, and optionally `sha256`. Decide whether `status` is needed in v1 or whether clean should physically rewrite the file and remove rows.

- [ ] Remove the old assumption that generated custom-word audio is primarily stored as files under `data/<word>/`.
  Details: Update code paths so generated and augmented custom-word samples are treated as Parquet-first data. Materialized WAVs under `data/custom-words/` should be derived artifacts created by `manifest`, not the source of truth.

- [ ] Rework `generate` to write rows into `data/custom_words.parquet`.
  Details: Remove the need for `output_dir` semantics. Keep provider selection, prompt loading, and concurrency behavior. Capture the generated WAV bytes, derive the deterministic filename using the current naming scheme, compute duration metadata, assign a stable `sample_id`, and write one row per sample to Parquet. Do not write per-word manifests.

- [ ] Remove the `voices.<provider>.txt` dependency from the Parquet path.
  Details: Stop treating the text registry as canonical for generated samples. Persist `voice_id` directly in Parquet and make later commands rely on that field. If voice code generation is still needed for deterministic filenames, compute it once at generation time and persist it in the row.

- [ ] Update provider generation code to return or expose WAV bytes cleanly.
  Details: The current provider implementation writes files directly. Refactor it so the generation path can store audio bytes in Parquet without first treating a directory as the destination. Keep the resulting filename logic identical to today.

- [ ] Rework `augment` to read from Parquet instead of scanning `data/*/*.wav`.
  Details: Load active generated rows from `data/custom_words.parquet`, decode `audio_bytes` to temp WAV files only when tooling like `ffmpeg` needs filesystem paths, run the existing tempo/noise augmentation logic, then capture the augmented WAV bytes and append derived rows back to Parquet. Store `source_type=augmented`, `parent_sample_id`, `tempo`, `noise_type`, and `snr`. Do not write word-level manifests.

- [ ] Preserve the current augmentation combinatorics and deterministic sampling behavior.
  Details: Keep the existing target-samples-per-word logic, subset selection logic, tempo set, SNR set, and filename conventions so the data distribution does not accidentally change during the migration.

- [ ] Rework `clean` to operate on `data/custom_words.parquet`.
  Details: Replace file deletion and per-word manifest edits with Parquet filtering and rewrite logic. Match the existing `--generated`, `--augmented`, and `--all` behavior by filtering rows using filename/source metadata. After rewriting Parquet, remove generated artifacts under `data/custom-words/` and remove split manifests under `data/manifests/` so `manifest` rebuilds everything from canonical data.

- [ ] Rework `manifest` to materialize custom-word WAVs from Parquet.
  Details: Load rows from `data/custom_words.parquet`, group them by `label`, recreate `data/custom-words/<word>/`, and write WAV files using `filename` and `audio_bytes`. Treat this directory as a fully rebuildable cache. Do not read custom-word data from per-word manifests anymore.

- [ ] Keep Google Speech Commands external and merge it only during manifest building.
  Details: Continue using the existing download/extract path for Google Speech Commands. During `manifest`, load configured Google labels from project config, scan the extracted Google dataset, and combine those entries with the materialized custom-word entries to build NeMo-compatible train/validation/test manifests.

- [ ] Keep `train` behavior unchanged.
  Details: Do not change NeMo integration or `train_model()` flow. Continue to train from JSONL manifests. The only upstream change is that `manifest` now produces those JSONL files from Parquet-backed custom-word data plus Google Speech Commands data.

- [ ] Decide and standardize the output location for split manifests.
  Details: Prefer `data/manifests/train_manifest.jsonl`, `data/manifests/validation_manifest.jsonl`, and `data/manifests/test_manifest.jsonl`. Update CLI defaults and any path resolution logic if needed, but do not change the training mechanics beyond manifest location.

- [ ] Remove old per-word manifest infrastructure once the Parquet path is complete.
  Details: Delete or retire `wakewords/manifest.py` usage for custom-word generation and augmentation after all callers have moved. Update docs and tests so they no longer describe or depend on `data/<word>/manifest.jsonl` for custom-word samples.

- [ ] Update CLI docs and repo docs for the new lifecycle.
  Details: Document that `generate` and `augment` write to `data/custom_words.parquet`, `manifest` materializes `data/custom-words/` and writes JSONL manifests, and `train` remains NeMo-manifest-based. Remove references that imply custom-word WAVs or per-word manifests are the canonical store.

- [ ] Add tests for the Parquet-backed workflow.
  Details: Cover row serialization, generate-to-Parquet writes, augment-from-Parquet reads and append behavior, clean rewrite behavior, manifest materialization, and train dry-run compatibility with the generated JSONL manifests. Replace tests that were only validating `ManifestStore` behavior.

- [ ] Validate the migration end to end.
  Details: Run `generate`, `augment`, `clean`, `manifest`, and `train --dry-run` against a small project fixture. Confirm that labels still resolve correctly, materialized WAVs are valid, generated manifests are NeMo-compatible, and no command still depends on per-word custom manifests.
