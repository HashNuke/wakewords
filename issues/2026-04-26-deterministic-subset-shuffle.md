# Deterministic Subset Shuffle

Status: DONE

## Problem

Augmentation subset selection ranked every available tempo, noise, or SNR value by hash and then took the first values. That was reproducible, but it did not match the intended behavior of shuffling the list and taking a random subset.

## Solution

Changed subset selection to create a deterministic random generator seeded by word, voice code, and category. The command now shuffles the available values and takes the requested count, giving random-like subsets without losing reproducibility across runs.
