Status: DONE

Problem

The project had generation support but no augmentation command that could derive tempo and background-noise variants directly from the generated `data/<word>/` layout. There was also no defined convention for consuming `data/_noises_/` clips or for keeping augmentation resumable and reproducible across runs.

Solution

Added a `datatools augment` command that:

- scans `data/<word>/` for clean baseline files named like `<word>-<voicecode>-t100-clean-nonoise-nosnr.wav`
- reads noise clips from `data/_noises_/*.wav`
- creates tempo-only clean variants and tempo-plus-noise variants in the same word directories
- keeps the existing short voice code from `data/voices.<provider>.txt`
- chooses a deterministic stretch from each longer noise file so reruns reproduce the same slice
- skips files that already exist unless `--overwrite` is set

The docs were also updated to reflect the current short voice-code naming and the noise-basename convention used in augmented filenames.
