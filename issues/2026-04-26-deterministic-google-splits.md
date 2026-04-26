Status: DONE

# Deterministic Google Splits

## Problem

Google Speech Commands examples were sorted for split assignment by the manifest entry's absolute `audio_filepath`. Recreating the same dataset under a different project directory could therefore change the stable hash used for train, validation, and test assignment, even when the Google word files themselves were identical.

## Solution

Google Speech Commands entries now carry an internal `split_key` based on `google-speech-commands/<word>/<filename>`. The split key is used directly for alphabetical ordering before ratio splitting and is not written into the generated manifest files.
