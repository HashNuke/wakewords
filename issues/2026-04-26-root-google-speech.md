# Root Google Speech

Status: DONE

## Problem

Google Speech Commands extraction was organized under `data/google-speech-commands/`, which mixed the external source dataset with generated and augmented project audio under `data/`.

## Solution

The download default now extracts Google Speech Commands into the project root as `google-speech-commands/`. Project initialization also ensures `google-speech-commands/` is present in `.gitignore` so the large extracted dataset is not committed.
