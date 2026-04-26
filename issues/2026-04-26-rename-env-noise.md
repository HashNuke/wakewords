Status: DONE

Problem

The augmentation filename schema and dataset docs used the field name `env` for the background-noise label. That clashes with the common meaning of `env` as an environment variable and makes the augmentation terminology less clear.

Solution

Renamed the documented filename field from `env` to `noise` and updated the surrounding explanation in `DATASET.md` so the schema consistently refers to background noise rather than environment variables.
