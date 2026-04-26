# tincan-wakewords

Tools and notes for building an expanded speech-command dataset from Google Speech Commands plus custom wake words.

## Data Tools

The `datatools` CLI is installed through the `uv` project.

List voices from the default TTS provider:

```sh
uv run datatools voices
```

List every available voice:

```sh
uv run datatools voices --all
```

Fetch more voice pages:

```sh
uv run datatools voices --pages 3
```

Filter voices by language or locale:

```sh
uv run datatools voices --lang en
uv run datatools voices --lang en_GB
```

Generate one WAV per line in `extended-words.txt` using the first available Cartesia voice:

```sh
uv run datatools generate
```

Generate with a specific voice id or exact voice name:

```sh
uv run datatools generate --voice <voice-id-or-name>
```

Generate with every available voice:

```sh
uv run datatools generate --all-voices
```

Generate using voices filtered by language or locale:

```sh
uv run datatools generate --lang en
```

Increase request concurrency:

```sh
uv run datatools generate --concurrency 4
```

By default, generated files are written under `data/<word>/`.

Place background-noise clips under `data/_noises_/` as `.wav` files such as `cafe.wav`, `car.wav`, or `office.wav`. The basename is used in augmented filenames.

Generate tempo-only and tempo+noise variants in place:

```sh
uv run datatools augment
```

The augment command scans `data/<word>/` for clean files named like `astra-cr1-t100-clean-nonoise-nosnr.wav`, keeps the existing voice code, picks a deterministic stretch from each noise clip, and writes derived files back into the same word directory.
