# wakewords

Tools and notes for building an expanded speech-command dataset from Google Speech Commands plus custom wake words.

## Data Tools

The `wakewords` CLI is installed through the `uv` project.

Initialize a dataset project in the current directory:

```sh
uv run wakewords init
```

This creates an empty `data/` directory, a `background_audio/` directory populated with Google Speech Commands background-noise clips and a `manifest.jsonl` duration manifest, and a prettified `config.json` file containing the Google Speech Commands v0.02 word list and an example custom word.

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

Generate one WAV per line in `extended-words.txt` using the first available Cartesia voice:

```sh
uv run wakewords generate
```

Generate with a specific voice id or exact voice name:

```sh
uv run wakewords generate --voice <voice-id-or-name>
```

Generate with every available voice:

```sh
uv run wakewords generate --all-voices
```

Generate with the first five voices returned by the provider:

```sh
uv run wakewords generate --voices 5
```

Generate using voices filtered by language or locale:

```sh
uv run wakewords generate --lang en
uv run wakewords generate --lang en --voices 5
```

Custom TTS providers can be registered from `config.json`. See
`docs/custom-providers.md`.

Increase request concurrency:

```sh
uv run wakewords generate --concurrency 4
```

By default, generated files are written under `data/<word>/`.

Each word directory also gets a `manifest.jsonl` file. Entries use the NeMo manifest shape with an extra `duration_ms` field:

```json
{"audio_filepath": "astra-cr1-t100-clean-nonoise-nosnr.wav", "duration": 0.92, "duration_ms": 920, "label": "astra"}
```

By default, augmentation reads background-noise clips from `background_audio/`. If that directory has a `manifest.jsonl` with `{ "audio": "<filename>", "duration_ms": <milliseconds> }` entries, augment uses those durations instead of probing the background files. The basename is used in augmented filenames.

Generate tempo-only and tempo+noise variants in place:

```sh
uv run wakewords augment
```

The augment command scans `data/<word>/` for clean files named like `astra-cr1-t100-clean-nonoise-nosnr.wav`, keeps the existing voice code, picks a deterministic stretch from each noise clip, and writes derived files back into the same word directory.
It reuses the clean source metadata from that word directory's `manifest.jsonl` and probes each augmented output separately before recording its final duration.

Build dataset-level train, validation, and test manifests from the per-word manifests:

```sh
uv run wakewords manifest --train-ratio 70 --validate-ratio 20 --test-ratio 10
```

This command reads `data/<word>/manifest.jsonl`, resolves the local filenames to full paths, performs a deterministic per-label split, and writes:

- `data/train_manifest.jsonl`
- `data/validation_manifest.jsonl`
- `data/test_manifest.jsonl`

Download the external training assets:

```sh
uv run wakewords download
```

This writes Google Speech Commands under `data/google-speech-commands` and the
base MatchboxNet model to
`models/base/commandrecognition_en_matchboxnet3x2x64_v2.nemo`.

Finetune the downloaded NeMo command-recognition model from those manifests:

```sh
uv run wakewords train
```

Training artifacts stay inside the initialized project directory under `runs/<run-name>/`:

- `train_config.json`
- `checkpoints/`
- `logs/`
- `models/`

TensorBoard is enabled by default for training runs and writes logs under the run's `logs/` directory. TensorBoard is installed on every platform; NeMo is installed as a package dependency on non-macOS platforms because its ASR dependency chain does not publish macOS wheels. Prepare datasets on macOS, then train on Linux.

Preview the resolved manifests, labels, and output layout without importing NeMo or starting training:

```sh
uv run wakewords train --dry-run
```

## License

Copyright (c) 2026 Akash Manohar John under MIT License (See LICENSE file).

**Background Sound:** The background audio embedded in this pypi package comes from the Google Speech Commands dataset and ships with this library for convenience. This is licensed under the same license as the dataset. The details are in the `README.md` file inside of the `wakewords/google_scd_background_noise` dir.
