# Clean Dataset Command

Status: DONE

## Problem

Generated clean WAV files and augmented WAV files could accumulate under `data/<word>/` with no project command to remove one category without manually deleting files and repairing manifests.

## Solution

Added `wakewords clean` with `--generated`, `--augmented`, and `--all` modes. The command deletes matching WAV files, removes stale entries from each word directory's `manifest.jsonl`, and deletes root split manifests so they can be regenerated without stale audio paths.
