from __future__ import annotations

import logging
import sys
from pathlib import Path

import fire

from datatools.augment import augment_dataset
from datatools.dataset_manifest import build_split_manifests
from datatools.providers import get_provider

logger = logging.getLogger(__name__)


class DataTools:
    def voices(
        self,
        provider: str = "cartesia",
        pages: int = 1,
        lang: str | None = None,
        all: bool = False,
        verbose: bool = False,
    ) -> None:
        """List available voices for a TTS provider."""
        _configure_logging(verbose=verbose)
        if pages < 1:
            raise ValueError("pages must be >= 1")

        tts = get_provider(provider)
        voices = tts.list_voices(pages=pages, all=all, lang=lang)

        for voice in voices:
            description = f"{voice.id}\t{voice.name or ''}".rstrip()
            if voice.language:
                description = f"{description}\t{voice.language}"
            print(description)

    def generate(
        self,
        provider: str = "cartesia",
        words_file: str = "extended-words.txt",
        text: str | None = None,
        output_dir: str = "data",
        voice: str | None = None,
        all_voices: bool = False,
        lang: str | None = None,
        concurrency: int = 1,
        model_id: str = "sonic-3",
        sample_rate: int = 16000,
        encoding: str = "pcm_s16le",
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """Generate TTS audio for one text string or each line in a words file."""
        _configure_logging(verbose=verbose)
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        prompts = [text] if text else _read_words(Path(words_file))
        if not prompts:
            raise ValueError("No text to generate. Provide --text or a non-empty --words-file.")

        tts = get_provider(provider)
        outputs = tts.generate(
            prompts=prompts,
            output_dir=Path(output_dir),
            voice=voice,
            all_voices=all_voices,
            lang=lang,
            concurrency=concurrency,
            model_id=model_id,
            sample_rate=sample_rate,
            encoding=encoding,
            overwrite=overwrite,
        )

        for output in outputs:
            print(output)

    def augment(
        self,
        data_dir: str = "data",
        noises_dir: str = "data/_noises_",
        concurrency: int = 1,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """Augment clean generated wav files with tempo and background-noise variants."""
        _configure_logging(verbose=verbose)
        outputs = augment_dataset(
            data_dir=Path(data_dir),
            noises_dir=Path(noises_dir),
            concurrency=concurrency,
            overwrite=overwrite,
        )

        for output in outputs:
            print(output)

    def manifest(
        self,
        data_dir: str = "data",
        train_ratio: int = 70,
        validate_ratio: int = 20,
        test_ratio: int = 10,
        train_filename: str = "train_manifest.jsonl",
        validate_filename: str = "validation_manifest.jsonl",
        test_filename: str = "test_manifest.jsonl",
        verbose: bool = False,
    ) -> None:
        """Build train/validation/test manifests from per-word manifests."""
        _configure_logging(verbose=verbose)
        outputs = build_split_manifests(
            data_dir=Path(data_dir),
            train_ratio=train_ratio,
            validate_ratio=validate_ratio,
            test_ratio=test_ratio,
            train_filename=train_filename,
            validate_filename=validate_filename,
            test_filename=test_filename,
        )

        for output in outputs.values():
            print(output)


def _read_words(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> None:
    _normalize_cli_flags()
    fire.Fire(DataTools())


def _normalize_cli_flags() -> None:
    flag_aliases = {
        "--all-voices": "--all_voices",
        "--words-file": "--words_file",
        "--output-dir": "--output_dir",
        "--data-dir": "--data_dir",
        "--noises-dir": "--noises_dir",
        "--train-ratio": "--train_ratio",
        "--validate-ratio": "--validate_ratio",
        "--test-ratio": "--test_ratio",
        "--train-filename": "--train_filename",
        "--validate-filename": "--validate_filename",
        "--test-filename": "--test_filename",
        "--model-id": "--model_id",
        "--sample-rate": "--sample_rate",
    }
    sys.argv = [flag_aliases.get(arg, arg) for arg in sys.argv]


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    main()
