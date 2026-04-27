# wakewords

Build custom wakeword and command-word datasets from TTS-generated words plus Google Speech Commands.

## Quick Start

### Create A Project

Initialize the project layout:

```sh
uv run wakewords init
```

This creates `data/`, `background_audio/`, `config.json`, and a project `.gitignore` entry for downloaded Google Speech Commands data.

Edit `config.json` and put your wake words in `custom_words`.

### Set Up TTS

The default TTS provider is Cartesia. Set your API key before generating audio:

```sh
export CARTESIA_API_KEY=your-api-key
```

Custom TTS providers can be registered from `config.json`. See [`docs/custom-providers.md`](docs/custom-providers.md).

### Generate English Data

Generate clean samples for the `custom_words` in the project `config.json` using every available English voice:

```sh
uv run wakewords generate --lang en --all-voices
```

Generated audio and metadata are written to the project's
`data/custom_words.parquet`.

### Augment The Dataset

Create noisy tempo variants for the generated clean samples:

```sh
uv run wakewords augment
```

By default, augmentation targets about `4000` total samples per word.

### Check Data

Print duration and no-speech stats for generated and augmented rows:

```sh
uv run wakewords check
```

Use `--generated` or `--augmented` to check only one source type. No-speech
sample IDs are written to `no-speech.txt` in the project root.

### Train

Download Google Speech Commands, build manifests, and preview the training run:

```sh
uv run wakewords download
uv run wakewords manifest
uv run wakewords train --dry-run
```

Run training on Linux with NeMo installed:

```sh
uv run wakewords train
```

Training uses NeMo's `from_pretrained()` by default. To train from a local `.nemo` file instead, pass `--base-model-path`.

### Export

Export the latest completed training run into a project-level model bundle:

```sh
uv run wakewords export --format onnx
```

This writes `models/model.onnx` for inference, plus
`models/last_checkpoint/last.ckpt`,
`models/last_checkpoint/train_config.json`, `models/labels.json`, and
`models/export_config.json` when those source files are available. The
checkpoint directory is kept ready for continued training with the original
training settings.

Resume from an exported checkpoint bundle with:

```sh
uv run wakewords train --from-checkpoint models/last_checkpoint/last.ckpt
```

That imports the checkpoint into a new `runs/<run-name>/` directory before
training continues.

### Find Outputs

Training artifacts are written under `runs/<run-name>/`:

- `train_config.json`
- `checkpoints/`
- `logs/`
- `models/`

The final exported model is written under the `models/` directory of that specific training run.

## More Details

See [`docs/USAGE.md`](docs/USAGE.md) for command options, split ratios, augmentation details, cleaning commands, and training notes.

## License

Copyright &copy; 2026 Akash Manohar John, under MIT License (See LICENSE file).

**Background sounds:** The background audio embedded in this pypi package comes from the Google Speech Commands dataset and ships with this library for convenience. This is licensed under the same license as the dataset. The details are in the `README.md` file inside of the `wakewords/google_scd_background_noise` dir.
