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

## Audio Generation

Generate one WAV per word with the provider's first voice:

```sh
uv run wakewords generate
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

## Dataset Downloads

Download the external training assets:

```sh
uv run wakewords download
```

This writes the extracted Google Speech Commands dataset and base MatchboxNet
model to:

```text
data/google-speech-commands
models/base/commandrecognition_en_matchboxnet3x2x64_v2.nemo
```

By default, archives are downloaded into a temporary directory in the current
directory, extracted, and then deleted. To keep the downloaded archives, pass
`--downloads-dir`:

```sh
uv run wakewords download --downloads-dir data/downloads
```

`--downloads-dir` only controls where downloaded archives are retained. Extracted
datasets and models still use the fixed paths shown above.

The download command shows separate progress bars for archive download and
archive extraction.

## Training

Build split manifests first:

```sh
uv run wakewords manifest
```

Install package dependencies:

```sh
uv sync
```

TensorBoard installs on every platform. NeMo installs automatically on non-macOS platforms; training is not supported on macOS because NeMo's ASR dependency chain does not publish macOS wheels. Use macOS for dataset preparation and a Linux machine for training.

Finetune the downloaded base model from `DATASET.md`:

```sh
uv run wakewords train
```

By default, a training run writes only inside the current project directory:

- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/train_config.json`
- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/checkpoints/`
- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/logs/`
- `runs/<timestamp>-commandrecognition_en_matchboxnet3x2x64_v2/models/`

Use `--dry-run` to create the run directory and inspect `train_config.json` without starting training.
