# Usage

## Project Init

Initialize a wakewords dataset project in the current directory:

```sh
uv run wakewords init
```

This creates:

- `data/`
- `background_audio/`
- `config.json`
- `.gitignore`

The generated `config.json` contains an editable `custom_words` list at the top of the file and the Google Speech Commands v0.02 word list below it.

## Audio Generation

The default TTS provider is Cartesia. Set `CARTESIA_API_KEY` before using it:

```sh
export CARTESIA_API_KEY=your-api-key
```

List voices from the default TTS provider:

```sh
uv run wakewords voices
```

List every available voice:

```sh
uv run wakewords voices --all
```

Fetch more voice pages:

```sh
uv run wakewords voices --pages 3
```

Filter voices by language or locale:

```sh
uv run wakewords voices --lang en
uv run wakewords voices --lang en_GB
```

Generate one WAV per word in the project `config.json` `custom_words` list with
the provider's first voice:

```sh
uv run wakewords generate
```

Run generation for a project directory other than the current directory:

```sh
uv run wakewords generate --project-dir path/to/project
```

Generate one explicit text prompt instead of `custom_words`:

```sh
uv run wakewords generate --text dexa
```

Generate with a specific voice:

```sh
uv run wakewords generate --voice <voice-id-or-name>
```

Generate with the first matching voices for a language or locale:

```sh
uv run wakewords generate --lang en --voices 5
```

Generate with every matching voice:

```sh
uv run wakewords generate --lang en --all-voices
```

Generate with a configured grouped voice selection policy:

```json
{
  "custom_words": [
    { "tts_input": "Astra", "label": "astra" },
    { "tts_input": "Boston", "label": "boston" },
    { "tts_input": "Tokyo", "label": "tokyo" }
  ],
  "generate": {
    "voice_selection": {
      "group_by": ["language", "gender"],
      "languages": "all",
      "genders": ["masculine", "feminine"],
      "limit_per_group": 3
    }
  },
  "google_speech_commands": ["yes", "no", "up"]
}
```

`languages` can be `"all"` or a non-empty list such as `["en", "es"]`. `genders` uses voice presentation names such as `"masculine"` and `"feminine"`; each provider maps them to its own API values. CLI voice options like `--voice`, `--voices`, `--all-voices`, or `--lang` override `generate.voice_selection` for that run.

Generate with the first five voices returned by the provider:

```sh
uv run wakewords generate --voices 5
```

Increase request concurrency:

```sh
uv run wakewords generate --concurrency 4
```

Custom TTS providers can be registered from `config.json`. See
`docs/custom-providers.md`.

By default, generated clean audio and metadata are written to the project's
`data/custom_words.parquet`. The Parquet rows include the WAV bytes, label,
voice metadata, duration, sample rate, source type, and content hash.

```json
{"label": "astra", "voice_code": "cr1", "source_type": "generated", "duration_ms": 920}
```

## Dataset Downloads

Download Google Speech Commands:

```sh
uv run wakewords download
```

This writes the extracted dataset to:

```text
google-speech-commands
```

By default, archives are downloaded into a temporary directory in the current
directory, extracted, and then deleted. To keep the downloaded archives, pass
`--downloads-dir`:

```sh
uv run wakewords download --downloads-dir data/downloads
```

`--downloads-dir` only controls where downloaded archives are retained. Extracted
datasets still use the fixed paths shown above.

The download command shows separate progress bars for archive download and
archive extraction.

## Augmentation

By default, augmentation reads background-noise clips from `background_audio/`.
If that directory has a `manifest.jsonl` with `{ "audio": "<filename>",
"duration_ms": <milliseconds> }` entries, augment uses those durations instead
of probing the background files. The basename is used in augmented filenames.

Generate noisy tempo variants from clean generated word audio:

```sh
uv run wakewords augment
```

By default, augmentation targets about `4000` total samples per word by reducing
the number of tempo, background-noise, and SNR choices per voice as voice count
increases. For `373` voices, this selects `5 tempos x 2 noises x 1 SNR = 10`
augmented variants per voice, for about `4103` total samples including the clean
originals.

The augment command reads clean generated rows from `data/custom_words.parquet`,
keeps the existing voice metadata, picks deterministic subsets of tempo,
background noise, and SNR values for each voice, and writes augmented rows back
to the same Parquet file. Each augmented row stores the derived WAV bytes,
parent sample ID, tempo, noise type, SNR, and probed duration.

To change the target:

```sh
uv run wakewords augment --target-samples-per-word 4000
```

Clean generated clean audio, augmented audio, or both:

```sh
uv run wakewords clean --generated
uv run wakewords clean --augmented
uv run wakewords clean --all
```

Cleaning removes matching rows from `data/custom_words.parquet`, deletes any
materialized `data/custom-words/<label>/<sample-id>.wav` files, and deletes
split manifests so they can be regenerated with `wakewords manifest`.

## Data Checks

Run a sanity check over generated and augmented custom-word rows:

```sh
uv run wakewords checkdata
```

The checkdata command reads `data/custom_words.parquet`, prints the sample count,
median duration, longest duration, longest sample ID, and no-speech count, then
writes the no-speech sample IDs to `no-speech.txt` in the project root.

Check only generated or augmented rows:

```sh
uv run wakewords checkdata --generated
uv run wakewords checkdata --augmented
```

`--all` is the default and includes both generated and augmented rows.

## Training

Build split manifests first:

```sh
uv run wakewords manifest
```

To change split ratios:

```sh
uv run wakewords manifest --train-ratio 70 --validate-ratio 20 --test-ratio 10
```

This command reads `data/custom_words.parquet`, materializes custom-word WAVs
under `data/custom-words/<label>/`, performs a deterministic per-label split,
and writes manifests under `data/manifests/`:

- `train_manifest.jsonl`
- `validation_manifest.jsonl`
- `test_manifest.jsonl`

Install package dependencies:

```sh
uv sync
```

TensorBoard installs on every platform. NeMo installs automatically on non-macOS platforms; training is not supported on macOS because NeMo's ASR dependency chain does not publish macOS wheels. Use macOS for dataset preparation and a Linux machine for training.

Finetune the model from `DATASET.md`:

```sh
uv run wakewords train
```

Training uses NeMo's `from_pretrained()` by default. To train from a local `.nemo`
file instead, pass `--base-model-path`.

Resume an interrupted or earlier training run from its last checkpoint:

```sh
uv run wakewords train --from-checkpoint runs/<run-name>/checkpoints/last.ckpt --max-epochs 20
```

When resuming, the run directory is inferred from the checkpoint path and training
continues writing checkpoints, logs, and the final model under that same run.
`--max-epochs` is the total epoch target, not the number of additional epochs.

By default, a training run writes only inside the current project directory:

- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/train_config.json`
- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/checkpoints/`
- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/logs/`
- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/models/`

Use `--dry-run` to create the run directory and inspect `train_config.json` without starting training.

## Export

Export the latest completed training run to ONNX:

```sh
uv run wakewords export --format onnx
```

By default, the command selects the most recently modified run under `runs/`
that contains a `.nemo` file, then writes a project-level bundle under
`models/`:

- `model.onnx`
- `last_checkpoint/last.ckpt`
- `last_checkpoint/train_config.json`
- `labels.json`
- `export_config.json`

`model.onnx` is the deployable inference artifact. `last_checkpoint/` is the
ready resume folder; its checkpoint contains training state, while its config
records the labels, manifests, model name, and training settings needed to
recreate the training setup.

Resume from the exported checkpoint folder:

```sh
uv run wakewords train --from-checkpoint models/last_checkpoint/last.ckpt
```

Unlike resuming from `runs/<run-name>/checkpoints/last.ckpt`, this creates a new
run under `runs/`, copies the exported checkpoint to
`runs/<new-run>/checkpoints/last.ckpt`, and stores the exported training config
as `runs/<new-run>/source_train_config.json`. The new run's
`train_config.json` records both the imported checkpoint path and the original
exported checkpoint source.

To export a specific run:

```sh
uv run wakewords export --format onnx --run-dir runs/<run-name>
```

Pass `--overwrite` to replace an existing bundle.
