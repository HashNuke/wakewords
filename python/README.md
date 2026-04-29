# wakewords (python)

Build custom wakewords detection datasets and models from synthetic data (TTS-generated).

> This python library is for training wakewords models. For using the models trained using this tool, please check the `wakewords` javascript and swift libraries.

## Install

```
# For pip
pip install wakewords

# For uv
uv add wakewords
```

## How to train a wakewords model for your custom words

### 1. Create a project

`cd` into any directory and run the init command to create the project scaffolding and a `config.json`.

```sh
uv run wakewords init
```

### 2. Define your wake words

Edit `config.json` and put your wake words in `custom_words`. Define as many as you like.

* `tts_input` is what is sent to the TTS to generate the audio.
* `label` is for the dataset.

```json
{
  "custom": [
    {"tts_input": "Atlas", label: "atlas}
  ]
}
```

The default config includes words from the google speech commands dataset by default. Feel free to review them and retain what you like.

### 3. Set Up TTS provider credentials

The default TTS provider is Cartesia. Set your API key to generate audio:

```sh
export CARTESIA_API_KEY=your-api-key
```

Custom TTS providers can be registered from `config.json`. See [`../docs/custom-providers.md`](../docs/custom-providers.md).

### 3. Generate audio samples

Generate clean samples for the `custom_words` in the project `config.json` using every available voice:

```sh
uv run wakewords generate --lang en --all-voices
```

The `lang` option is for a specfic language. If you skip it, it'll use the voices you specify. Check docs for more options like selecting by gender, language, etc. You can specify config to stay selective like "3 voices per gender per language".

Generated audio and metadata are written to the project's `data/custom_words.parquet`.

### 4. Augment the dataset

This command will augment the dataset to create more samples using variations with background noise, tempo and snr. The target sample size post-augmentation is around 4k samples per word.

```sh
uv run wakewords augment
```

The default target for total number of samples per word is 4000 (approximated). This is to match the number of samples for each word in the google speech commands dataset (helps if you retain the words for that dataset too). Helps balance the samples available for each word.

So the command calculates (`targetSamplesPerWord = 4000 - generatedSamples`) and then generates enough variations to (approximately) match that target sample count.

### 5. Train your model

Download Google Speech Commands, build manifests, and preview the training run:

```sh
# Download google speech commands dataset
uv run wakewords download

# Create manifest files for 70-20-10 split (train-validate-test)
uv run wakewords manifest

# Start training (linux only).
# Default 15 epochs. Use --max-epochs to change.
uv run wakewords train
```

### Export

Export the latest completed training run into a project-level model bundle:

```sh
# defaults to onnx format
uv run wakewords export
```

* This writes `models/model.onnx` and `models/labels.json` for inference.
* And also saves `models/last_checkpoint/last.ckpt`.

You can now use `model.onnx` and `labels.json` for inference with the `wakewords` javascript or swift libraries.

## More Details

We have other commands like `checkdata`. See [`../docs/USAGE.md`](../docs/USAGE.md) for command options, split ratios, augmentation details, cleaning commands, and training notes.

## License

Copyright &copy; 2026 Akash Manohar John, under MIT License (See LICENSE file at root of git repo).

**Background sounds:** The background audio embedded in this PyPI package comes from the Google Speech Commands dataset and ships with this library for convenience. This is licensed under the same license as the dataset. The details are in the `README.md` file inside of the `wakewords/google_scd_background_noise` dir.
