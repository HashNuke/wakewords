# wakewords

Tools and notes for building an expanded speech-command dataset from Google Speech Commands plus custom wake words.

## Data Tools

The `wakewords` CLI is installed through the `uv` project.

Initialize a dataset project in the current directory:

```sh
uv run wakewords init
```

This creates an empty `data/` directory, a `background_audio/` directory populated with Google Speech Commands background-noise clips and a `manifest.jsonl` duration manifest, a prettified `config.json` file containing the Google Speech Commands v0.02 word list and an example custom word, and a `.gitignore` entry for the extracted Google Speech Commands dataset.

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

Generate tempo, background-noise, and SNR combo variants in place:

```sh
uv run wakewords augment
```

The augment command scans `data/<word>/` for clean files named like `astra-cr1-t100-clean-nonoise-nosnr.wav`, keeps the existing voice code, picks deterministic subsets of tempo, background noise, and SNR values for each voice, and writes derived files back into the same word directory. By default, it targets about `4000` total samples per word; for `373` voices that selects `5 tempos x 2 noises x 1 SNR = 10` augmented files per voice, or about `4103` total samples including the clean originals.
It reuses the clean source metadata from that word directory's `manifest.jsonl` and probes each augmented output separately before recording its final duration.

Change the per-word target with:

```sh
uv run wakewords augment --target-samples-per-word 4000
```

Delete generated clean audio, augmented audio, or both:

```sh
uv run wakewords clean --generated
uv run wakewords clean --augmented
uv run wakewords clean --all
```

Cleaning also updates each word directory's `manifest.jsonl` and removes root split manifests because they may contain stale audio paths.

Build dataset-level train, validation, and test manifests from the per-word manifests:

```sh
uv run wakewords manifest --train-ratio 70 --validate-ratio 20 --test-ratio 10
```

This command reads `data/<word>/manifest.jsonl`, resolves the local filenames to full paths, performs a deterministic per-label split, and writes project-root manifests:

- `train_manifest.jsonl`
- `validation_manifest.jsonl`
- `test_manifest.jsonl`

Download Google Speech Commands:

```sh
uv run wakewords download
```

This writes Google Speech Commands under `google-speech-commands/` in the project root.

Finetune the NeMo command-recognition model from those manifests:

```sh
uv run wakewords train
```

Training uses NeMo's `from_pretrained()` by default. To train from a local `.nemo`
file instead, pass `--base-model-path`.

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
