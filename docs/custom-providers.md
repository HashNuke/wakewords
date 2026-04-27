# Custom Providers

`wakewords` ships with the built-in `cartesia` provider. You can add project
specific providers from `config.json` without modifying the package.

## Register a Provider

Add a `providers` object to the project `config.json` created by
`wakewords init`:

```json
{
  "custom_words": [{ "tts_input": "Dexa", "label": "dexa" }],
  "google_speech_commands": ["yes", "no"],
  "providers": {
    "my-provider": "my_project.providers:MyProvider"
  }
}
```

The provider map is merged with built-in providers, so `cartesia` remains
available unless you intentionally register another provider with the same name.

Use the provider by name:

```sh
uv run wakewords voices --provider my-provider
uv run wakewords generate --provider my-provider --lang en --voices 5
```

## Provider Shape

Providers use Python import paths in the form:

```text
module:attribute
```

The attribute can be a class, factory, or already-created provider object. A
provider must expose `list_voices` and `generate` methods compatible with the
package's `TTSProvider` protocol.

Minimal example:

```python
from wakewords.providers.base import Voice


class MyProvider:
    name = "my-provider"

    def list_voices(
        self,
        pages: int = 1,
        all: bool = False,
        lang: str | None = None,
        gender: str | None = None,
    ) -> list[Voice]:
        return [Voice(id="voice-1", name="Voice 1", language=lang, gender="feminine")]

    def generate(
        self,
        *,
        prompt: str,
        voice: Voice,
        lang: str | None,
        model_id: str,
        sample_rate: int,
        encoding: str,
    ) -> bytes:
        # Return one generated WAV as bytes. The CLI handles selection,
        # concurrency, VAD trimming, and Parquet writes.
        return b"..."
```

The CLI writes generated audio and metadata to `data/custom_words.parquet` and
uses `wakewords.providers.base.Voice` for voice metadata. Custom providers only
need to list voices and return one generated WAV byte string per `generate` call.

## Import Requirements

The module in `config.json` must be importable from the Python environment where
the CLI runs. Common approaches:

- Put provider code in the project directory and run commands from that directory.
- Put provider code in an installed Python package.
- Set `PYTHONPATH` to include the directory containing the provider module.

Provider-specific credentials should stay inside the provider implementation,
usually by reading environment variables.
