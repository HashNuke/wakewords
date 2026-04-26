# Download Base Model

Status: PARTIAL

## Problem

The download command only fetched external dataset assets. We considered also
downloading the MatchboxNet base checkpoint into the project directory.

## Solution

Kept `wakewords download` focused on Google Speech Commands. Training uses
NeMo's `from_pretrained()` by default, and `--base-model-path` can be used when a
caller intentionally wants to train from a local `.nemo` file.
