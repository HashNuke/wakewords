# Usage

## Dataset Downloads

Download and extract Google Speech Commands:

```sh
uv run datatools download --google-speech-commands
```

This writes the extracted dataset to:

```text
data/google-speech-commands
```

Download and extract the Common Voice 7.0 single-word target segment:

```sh
COMMONVOICE_API_KEY=<api-key> uv run datatools download --common-voice-sw
```

This writes the extracted dataset to:

```text
data/common-voice-7-single-word
```

Download and extract all supported external datasets:

```sh
COMMONVOICE_API_KEY=<api-key> uv run datatools download --all
```

By default, archives are downloaded into a temporary directory in the current
directory, extracted, and then deleted. To keep the downloaded archives, pass
`--downloads-dir`:

```sh
COMMONVOICE_API_KEY=<api-key> uv run datatools download --all --downloads-dir data/downloads
```

`--downloads-dir` only controls where archives are retained. Extracted datasets
still use the fixed paths under `data/` shown above.

The download command shows separate progress bars for archive download and
archive extraction.
