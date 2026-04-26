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

The generated `config.json` contains an editable `custom_words` list at the top of the file and the Google Speech Commands v0.02 word list below it.

## Dataset Downloads

Download and extract Google Speech Commands:

```sh
uv run wakewords download --google-speech-commands
```

This writes the extracted dataset to:

```text
data/google-speech-commands
```

Download and extract the Common Voice 7.0 single-word target segment:

```sh
COMMONVOICE_API_KEY=<api-key> uv run wakewords download --common-voice-sw
```

This writes the extracted dataset to:

```text
data/common-voice-7-single-word
```

Download and extract all supported external datasets:

```sh
COMMONVOICE_API_KEY=<api-key> uv run wakewords download --all
```

By default, archives are downloaded into a temporary directory in the current
directory, extracted, and then deleted. To keep the downloaded archives, pass
`--downloads-dir`:

```sh
COMMONVOICE_API_KEY=<api-key> uv run wakewords download --all --downloads-dir data/downloads
```

`--downloads-dir` only controls where archives are retained. Extracted datasets
still use the fixed paths under `data/` shown above.

The download command shows separate progress bars for archive download and
archive extraction.
