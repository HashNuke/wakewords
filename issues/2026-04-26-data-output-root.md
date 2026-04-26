Status: DONE

Problem

Generated audio defaulted to `data/generated/...`, which no longer matched the intended dataset layout of storing files directly under `data/<word>/`.

Solution

Changed the `datatools generate` default `output_dir` from `data/generated` to `data` and updated the README to reflect the current output structure.
