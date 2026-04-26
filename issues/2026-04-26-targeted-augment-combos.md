# Targeted Augment Combos

Status: DONE

## Problem

Augmentation used the full Cartesian product of tempo, background-noise, and SNR values for every voice. With hundreds of voices, this produced far more custom-word samples than the Google Speech Commands per-word scale of roughly a few thousand samples.

## Solution

Augmentation now uses a `target_samples_per_word` setting, defaulting to `4000`. For each word, it counts clean voice samples, calculates the combo count needed to stay near the target, then deterministically picks smaller per-voice subsets of tempo, background-noise, and SNR values before creating all combinations for that voice.

For `373` voices, the default chooses `5 tempos x 2 noises x 1 SNR = 10` augmented variants per voice, producing about `4103` total samples per word including the clean originals.
