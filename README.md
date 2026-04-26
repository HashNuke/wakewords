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

By default, generated files are written under `data/generated/cartesia/<voice>/`.
