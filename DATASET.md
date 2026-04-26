# Dataset Notes

This project is targeting wake-word finetuning for the labels defined in `words.json`.

`words.json` is the canonical word list for this repo. It contains an array of objects
with two fields:

```json
{"word": "yes", "source": "google-speech-commands"}
```

The file includes a collection of most words from the Google Speech Commands
dataset and the words for Tincan.

Regenerate it after editing either source list:

```sh
uv run python merge_words.py
```

## `google/speech_commands` on Hugging Face

Reference:

- <https://huggingface.co/datasets/google/speech_commands>
- <https://huggingface.co/datasets/google/speech_commands/blob/main/speech_commands.py>

The Hugging Face dataset is a packaged version of the Google Speech Commands corpus.

Key points:

- It exposes fixed `train`, `validation`, and `test` splits.
- Audio is decoded at `16_000` Hz.
- Each sample is a single `.wav` clip.
- File names follow the original structure: `<word>/<speaker>_nohash_<utterance>.wav`.
- Hugging Face parses `speaker_id` and `utterance_id` from the filename.

Example fields per sample:

- `audio`
- `label`
- `file`
- `speaker_id`
- `utterance_id`
- `is_unknown`

Useful implications for this repo:

- The dataset is already close to the format needed for wake-word classification.
- Non-target spoken words can be used as hard negatives.
- `speaker_id` is available, which makes speaker-disjoint validation possible.
- `_silence_` exists as a class, but silence/background examples may still need custom windowing depending on training setup.

### Word labels

For `v0.02`, Hugging Face keeps explicit Speech Commands word labels rather than
collapsing them into a single unknown bucket.

This repo tracks the selected Speech Commands labels and custom Tincan words in
`words.json`. Each entry records the label text and its source, so training and
data-generation scripts can consume the same word list without duplicating labels
in documentation.

## NeMo finetuning requirements

Reference:

- <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/commandrecognition_en_matchboxnet3x1x64_v1?version=1.0.0rc1>
- <https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/speech_classification/datasets.html>
- <https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/speech_classification/speech_to_label.py>
- <https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/conf/matchboxnet/matchboxnet_3x1x64_v1.yaml>

The target model is a NeMo speech classification model. For finetuning, the important requirement is not a folder layout but a manifest layout.

### Expected training data format

NeMo expects line-delimited JSON manifests, usually:

- `train_manifest.json`
- `validation_manifest.json`
- `test_manifest.json`

Each line should contain at least:

- `audio_filepath`
- `duration`
- `label`

Optional field:

- `offset`

Minimal example:

```json
{"audio_filepath": "/abs/path/audio/mika_001.wav", "duration": 0.92, "label": "mika"}
{"audio_filepath": "/abs/path/audio/lyra_014.wav", "duration": 1.03, "label": "lyra"}
{"audio_filepath": "/abs/path/audio/noise_020.wav", "duration": 1.00, "label": "unknown"}
{"audio_filepath": "/abs/path/audio/sil_003.wav", "duration": 1.00, "label": "silence"}
```

### Audio expectations

The safest choice for compatibility with the pretrained checkpoint is:

- mono audio
- `16_000` Hz sample rate
- `.wav` files
- short command-length utterances

### Label expectations

The manifest `label` strings must exactly match the labels configured in the model.

For this project, derive model labels from `words.json` instead of copying a
separate hard-coded list into training configs or scripts. Entries with
`"source": "tincan-words"` are the custom Tincan labels, and entries with
`"source": "google-speech-commands"` can be used for Speech Commands examples,
negative speech, or any stock command classes included in an experiment.

It is also likely useful to include rejection classes such as:

- `unknown`
- `silence`

Without explicit negative classes, the classifier is forced to choose one of the wake words even for non-wake audio.

## Recommended dataset organization for this repo

The model does not require a strict directory structure, but a practical local layout would be:

```text
data/
  raw/
    <word-from-words-json>/
    unknown/
    silence/
  manifests/
    train_manifest.json
    validation_manifest.json
    test_manifest.json
```

Recommended conventions:

- Keep all training audio at `16 kHz` mono WAV.
- Put wake-word positives in per-word folders during collection and curation.
- Build manifests from curated audio rather than relying on directory names at training time.
- Use `google/speech_commands` words as negative speech where useful.
- Keep validation and test speaker-disjoint from training where possible.
