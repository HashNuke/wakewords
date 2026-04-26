Status: DONE

# Generate Config Words

## Problem

The `wakewords generate` command defaulted to `extended-words.txt`, which was outdated and did not match the project-oriented workflow where commands operate in the context of a project directory.

## Solution

`generate` now accepts `project_dir` and reads prompts from the project's `config.json` `custom_words` key by default. Explicit `--text` still generates a single prompt, and `--words-file` remains available as an override. Provider lookup now uses the same project config path, and relative output directories are resolved inside the project directory.

The README and usage docs now describe the project-based generation behavior.
