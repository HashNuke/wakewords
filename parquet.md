# Parquet Migration Plan

## Goal

Move custom-word dataset storage from per-word WAV directories plus per-word `manifest.jsonl` files to a single Parquet-backed source of truth, while continuing to train with NeMo through generated JSONL manifests.

Google Speech Commands stays external and is not stored in Parquet.

## Non-goals

- Do not patch NeMo.
- Do not switch MatchboxNet training to Lhotse.
- Do not store Google Speech Commands rows in our Parquet dataset.
- Do not keep word-level manifests for custom-word audio.

## Proposed Layout

```text
data/
  custom_words.parquet
  custom-words/
    <word>/
      *.wav
  google-speech-commands/
    <word>/
      *.wav
  manifests/
    train_manifest.jsonl
    validation_manifest.jsonl
    test_manifest.jsonl
```

Notes:

- `data/custom_words.parquet` is the canonical store for custom-word audio and metadata.
- `data/custom-words/` is materialized output generated from Parquet during `manifest`.
- `data/google-speech-commands/` remains downloaded and extracted external data.
- `data/manifests/*.jsonl` are generated compatibility files for NeMo training.

## Parquet Schema

Use one row per custom-word sample.

Required columns:

- `sample_id`: stable unique ID
- `filename`: deterministic output filename, such as `astra-cr1-t100-clean-nonoise-nosnr.wav`
- `label`: normalized word label, such as `astra`
- `audio_bytes`: WAV bytes
- `voice_id`: stable voice code, such as `cr1`
- `provider`: provider short code, such as `cr`
- `duration_ms`: integer duration in milliseconds
- `lang`: language code or null

Recommended additional columns:

- `sample_rate`: expected to be `16000`
- `channels`: expected to be `1`
- `source_type`: `generated` or `augmented`
- `parent_sample_id`: null for generated rows, source row ID for augmentations
- `tempo`: float or null
- `noise_type`: string or null
- `snr`: int or null
- `created_at`: timestamp string
- `status`: `active` or `deleted`
- `sha256`: checksum of `audio_bytes`

## Command Behavior Changes

### `generate`

Current behavior:

- writes WAV files under `data/<word>/`
- writes per-word `manifest.jsonl`

Target behavior:

- generate WAV bytes from provider response
- build one Parquet row per generated sample
- append or rewrite `data/custom_words.parquet`
- do not write per-word manifests
- do not require `data/<word>/` output as the canonical store

Implementation notes:

- continue using the current filename scheme
- continue using the current voice selection behavior
- do not maintain `voices.<provider>.txt`; `voice_id` in Parquet becomes the canonical stable voice identifier used by later commands
- preserve `output_dir` semantics only if still needed; otherwise narrow the command around the project `data/` directory

### `augment`

Current behavior:

- reads clean WAV files from `data/*/*.wav`
- writes augmented WAV files beside them
- updates per-word `manifest.jsonl`

Target behavior:

- load active generated custom-word rows from Parquet
- decode `audio_bytes` to temp WAV files only when ffmpeg or other file-based tooling needs paths
- create augmented WAV bytes
- append augmented rows to Parquet
- do not write word-level manifests

Implementation notes:

- keep the current deterministic augmentation selection logic
- keep filename conventions for tempo/noise/SNR variants
- record `parent_sample_id`, `tempo`, `noise_type`, and `snr`

### `clean`

Current behavior:

- deletes WAV files from `data/<word>/`
- edits per-word manifests
- deletes split manifests

Target behavior:

- operate on Parquet rows instead of per-word directories
- mark matching rows as `deleted`, or rewrite the file without them
- delete `data/manifests/*.jsonl`
- optionally remove `data/custom-words/` materialized files so the next `manifest` rebuilds them cleanly

Recommended first implementation:

- rewrite `data/custom_words.parquet` and physically remove cleaned rows
- remove generated artifacts under `data/custom-words/`
- remove `data/manifests/*.jsonl`

This is simpler than introducing tombstones on the first pass.

### `manifest`

Current behavior:

- builds train/validate/test manifests from per-word manifests and Google Speech Commands directories

Target behavior:

1. ensure Google Speech Commands is downloaded and extracted
2. load active custom-word rows from `data/custom_words.parquet`
3. materialize custom rows into `data/custom-words/<word>/filename.wav`
4. load Google Speech Commands entries only for labels configured in `config.json`
5. create train/validate/test JSONL manifests in `data/manifests/`

Important change:

- custom-word rows no longer come from per-word manifests
- custom-word durations and labels come from Parquet metadata
- `audio_filepath` in generated NeMo manifests should point to the materialized WAV files in `data/custom-words/`

### `train`

Current behavior:

- consumes local JSONL manifest paths

Target behavior:

- no behavioral change to training
- keep NeMo usage unchanged
- keep the current `train_model()` flow and current NeMo manifest-based inputs
- only the upstream producer of those JSONL manifests changes from per-word manifests to Parquet-backed materialization

Optional improvement:

- if manifests are missing, fail with a clear message telling the user to run `wakewords manifest`

## Materialization Rules

`manifest` should treat `data/custom-words/` as a rebuildable cache.

Recommended rules:

- remove and recreate `data/custom-words/<word>/` for labels being materialized
- write WAV files from `audio_bytes` using stored `filename`
- never treat materialized custom-word WAVs as the source of truth

This keeps the repo model simple:

- Parquet is source of truth
- `custom-words/` is derived output
- NeMo manifests are derived output

## Migration Steps

### Phase 1: Add Parquet storage primitives

- add a new module for Parquet row read/write helpers
- define schema validation and row serialization
- support reading rows by label, by source type, and all active rows

### Phase 2: Move `generate` to Parquet

- replace `ManifestStore` writes with Parquet row writes
- keep filename generation unchanged
- store audio bytes and metadata in Parquet

### Phase 3: Move `augment` to Parquet

- read source rows from Parquet instead of scanning `data/*/*.wav`
- produce augmented rows and append them to Parquet
- remove dependence on per-word manifests

### Phase 4: Rework `manifest`

- materialize custom-word WAVs from Parquet into `data/custom-words/`
- keep Google Speech Commands loading as a separate source
- write split JSONL manifests into `data/manifests/`

### Phase 5: Rework `clean`

- clean Parquet rows instead of directory files
- clear materialized custom-word files and split manifests

### Phase 6: Remove old manifest assumptions

- delete `wakewords/manifest.py` if no longer needed
- remove references to per-word `manifest.jsonl` from docs and code
- update tests to cover Parquet-based flows

## Files Likely To Change

- `wakewords/cli.py`
- `wakewords/providers/cartesia.py`
- `wakewords/augment.py`
- `wakewords/clean.py`
- `wakewords/dataset_manifest.py`
- `wakewords/train.py`
- `DATASET.md`
- new Parquet helper module, likely `wakewords/parquet_store.py`

## Testing Plan

- unit test Parquet row creation and rewrite behavior
- unit test `generate` adds rows with expected metadata
- unit test `augment` reads from Parquet and appends derived rows
- unit test `clean` removes or rewrites expected rows
- unit test `manifest` materializes custom words and writes valid NeMo manifests
- regression test `train --dry-run` still resolves labels and manifest paths correctly

## Risks

- single-file Parquet rewrites may become slow as the dataset grows
- augment code currently assumes filesystem-first audio processing and will need temporary-file bridging
- materialized custom-word WAVs can drift if users edit them manually; code should always treat them as disposable
- existing tests around `ManifestStore` will need replacement, not just updates

## Recommended First Cut

Keep the first implementation intentionally simple:

- one `data/custom_words.parquet` file
- full-file rewrite on add/remove
- `manifest` fully rebuilds `data/custom-words/` and `data/manifests/`
- no incremental compaction logic
- no word-level manifests anywhere

That gives us the Parquet-backed workflow with the smallest change in behavior and keeps NeMo integration unchanged.
