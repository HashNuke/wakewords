# Remove Common Voice

Status: DONE

## Problem

The `wakewords download` command still exposed Common Voice download support even
though this project no longer needs Common Voice as an executable dataset source.
That added extra CLI surface area and required a `COMMONVOICE_API_KEY` for paths
that should only download supported datasets.

## Solution

Removed Common Voice from the download implementation, CLI flags, and user-facing
usage docs. Common Voice research notes remain in `DATASET.md`.
