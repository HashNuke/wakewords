# Root Split Manifests

Status: DONE

## Problem

Dataset-level `train_manifest.jsonl`, `validation_manifest.jsonl`, and `test_manifest.jsonl` were generated inside `data/`, even though they describe the whole project dataset rather than a single word directory.

## Solution

Split manifest generation now writes the train, validation, and test manifests to the project root. Training defaults were updated to resolve those manifest filenames from the project root so `wakewords manifest` followed by `wakewords train` continues to work without extra flags.
