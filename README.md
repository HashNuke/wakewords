# wakeword

Tools and notes for building an expanded speech-command dataset from Google Speech Commands plus custom wake words.

## Data Tools

The `wakeword` CLI is installed through the `uv` project.

Initialize a dataset project in the current directory:

```sh
uv run wakeword init
```

This creates an empty `data/` directory plus a prettified `config.json` file containing the Google Speech Commands v0.02 word list and an example custom word.

List voices from the default TTS provider:

```sh
uv run wakeword voices
```

List every available voice:

```sh
uv run wakeword voices --all
```

Fetch more voice pages:

```sh
uv run wakeword voices --pages 3
```

Filter voices by language or locale:

```sh
uv run wakeword voices --lang en
uv run wakeword voices --lang en_GB
```

Generate one WAV per line in `extended-words.txt` using the first available Cartesia voice:

```sh
uv run wakeword generate
```

Generate with a specific voice id or exact voice name:

```sh
uv run wakeword generate --voice <voice-id-or-name>
```

Generate with every available voice:

```sh
uv run wakeword generate --all-voices
```

Generate using voices filtered by language or locale:

```sh
uv run wakeword generate --lang en
```

Increase request concurrency:

```sh
uv run wakeword generate --concurrency 4
```

By default, generated files are written under `data/<word>/`.

Each word directory also gets a `manifest.jsonl` file. Entries use the NeMo manifest shape with an extra `duration_ms` field:

```json
{"audio_filepath": "astra-cr1-t100-clean-nonoise-nosnr.wav", "duration": 0.92, "duration_ms": 920, "label": "astra"}
```

Place background-noise clips under `data/_noises_/` as `.wav` files such as `cafe.wav`, `car.wav`, or `office.wav`. The basename is used in augmented filenames.

Generate tempo-only and tempo+noise variants in place:

```sh
uv run wakeword augment
```

The augment command scans `data/<word>/` for clean files named like `astra-cr1-t100-clean-nonoise-nosnr.wav`, keeps the existing voice code, picks a deterministic stretch from each noise clip, and writes derived files back into the same word directory.
It reuses the clean source metadata from that word directory's `manifest.jsonl` and probes each augmented output separately before recording its final duration.

Build dataset-level train, validation, and test manifests from the per-word manifests:

```sh
uv run wakeword manifest --train-ratio 70 --validate-ratio 20 --test-ratio 10
```

This command reads `data/<word>/manifest.jsonl`, resolves the local filenames to full paths, performs a deterministic per-label split, and writes:

- `data/train_manifest.jsonl`
- `data/validation_manifest.jsonl`
- `data/test_manifest.jsonl`

## Background Sound Credits

- Sea waves by Loredenii: https://freesound.org/people/Loredenii/sounds/851298/
