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

## Tincan generated dataset

Generate original Tincan word audio first and place those files directly in the
local dataset tree with deterministic filenames. Generated audio and derived
augmentations should live under `data/` and should not be committed.

Use this filename schema for generated and augmented WAV files:

```text
<word>-<provider>-<voice>-t<tempo>-<noise>-<noiseid>-<snr>.wav
```

Fields:

- `word`: normalized label from `words.json`, such as `astra`
- `provider`: short provider key, such as `cr` for Cartesia or `gk` for Grok
- `voice`: stable short voice code from `data/voices.<provider>.txt`, such as `cr1` or `gk12`
- `tempo`: tempo multiplier without the decimal point, such as `t100` or `t085`
- `noise`: noise label, such as `clean`, `cafe`, `car`, `office`, or `street`
- `noiseid`: background-noise clip id, or `nonoise` for clean samples. If you use one file per noise source such as `data/_noises_/cafe.wav`, the current tools use the basename for both `noise` and `noiseid`.
- `snr`: signal-to-noise ratio label, or `nosnr` for clean samples

Clean examples:

```text
astra-cr1-t100-clean-nonoise-nosnr.wav
astra-cr1-t085-clean-nonoise-nosnr.wav
```

Noisy examples:

```text
astra-cr1-t100-cafe-cafe-snr20.wav
astra-cr1-t110-car-car-snr10.wav
```

Each `data/<word>/` directory can also carry a local `manifest.jsonl` for generated and augmented files. The format matches the NeMo manifest shape with one extra field:

- `duration_ms`: duration in milliseconds

Example:

```json
{"audio_filepath": "astra-cr1-t100-clean-nonoise-nosnr.wav", "duration": 0.92, "duration_ms": 920, "label": "astra"}
```

Because these manifests live inside each word directory, `audio_filepath` is stored as a local filename there. When building dataset-level manifests for training, resolve those local filenames back to full paths.

### Speed augmentation with ffmpeg

Use `ffmpeg`'s `atempo` audio filter to make generated speech faster or slower
without shifting pitch:

```sh
ffmpeg -i input.wav -filter:a "atempo=1.25" -ar 16000 -ac 1 output_fast.wav
ffmpeg -i input.wav -filter:a "atempo=0.9" -ar 16000 -ac 1 output_slow.wav
```

Generate these tempo variants for each original sample:

- `0.85`
- `0.90`
- `0.95`
- `1.05`
- `1.10`
- `1.15`
- `1.25`

Keep the original generated samples as `t100` and write speed-augmented files as
separate derived variants. These fixed tempo values are enough to teach the
model tolerance for speaking-rate variation; the model does not need an example
at every possible in-between tempo.

### Background noise augmentation

Use real background-noise clips for noisy variants rather than generating noise
with TTS. Keep curated or downloaded noise files outside git, then mix them with
clean speech at controlled SNR levels.

SNR means signal-to-noise ratio:

- signal: the spoken Tincan word
- noise: background sound such as cafe, car, office, street, fan, or kitchen

Higher SNR means cleaner speech. Start with these noisy variants:

- `snr20` for light background noise
- `snr10` for noticeable background noise
- `snr05` for hard noisy examples

Clean samples should use `clean-nonoise-nosnr`.

## Common Voice 7.0 - Single Word Target Segment

```
Dataset ID: cmkzhp64p00wlno07elrmt20y
Dataset Slug: common-voice-7-0-single-word-target-segm-2ba5e0aa
```

Script download dataset is below.

```shell
# Script requirements
# Export `COMMONVOICE_API_KEY`

# Get download URL
RESPONSE=$(curl -s -X POST "https://mozilladatacollective.com/api/datasets/cmkzhp64p00wlno07elrmt20y/download" \
  -H "Authorization: Bearer $COMMONVOICE_API_KEY" \
  -H "Content-Type: application/json")

# Extract download URL and download file
DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')
curl -L -o "Common Voice 7.0 - Single Word Target Segment.tar.gz" "$DOWNLOAD_URL"
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

You can generate these combined manifests from the per-word manifests with:

```sh
uv run datatools manifest --train-ratio 70 --validate-ratio 20 --test-ratio 10
```

The split is deterministic and performed per label, with small rounding adjustments when an exact integer split is not possible.

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
