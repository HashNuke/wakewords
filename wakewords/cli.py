from __future__ import annotations

import logging
import sys
from pathlib import Path

import fire

from wakewords.augment import augment_dataset
from wakewords.dataset_manifest import build_split_manifests
from wakewords.download import download_datasets
from wakewords.project import init_project
from wakewords.providers import get_provider
from wakewords.train import DEFAULT_MODEL_NAME, train_model

logger = logging.getLogger(__name__)


class DataTools:
    def init(self, directory: str = ".") -> None:
        """Initialize a wakewords dataset project directory."""
        outputs = init_project(Path(directory))
        for output in outputs:
            print(output)

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
        voices: int | None = None,
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
            voices=voices,
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
        noises_dir: str = "background_audio",
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

    def download(
        self,
        google_speech_commands: bool = False,
        common_voice_sw: bool = False,
        all: bool = False,
        downloads_dir: str | None = None,
        data_dir: str = "data",
        verbose: bool = False,
    ) -> None:
        """Download and extract external speech datasets."""
        _configure_logging(verbose=verbose)
        outputs = download_datasets(
            google_speech_commands=google_speech_commands,
            common_voice_sw=common_voice_sw,
            all=all,
            downloads_dir=Path(downloads_dir) if downloads_dir else None,
            data_dir=Path(data_dir),
        )

        for output in outputs:
            print(output)

    def train(
        self,
        project_dir: str = ".",
        data_dir: str = "data",
        runs_dir: str = "runs",
        run_name: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        train_manifest: str = "train_manifest.jsonl",
        validation_manifest: str = "validation_manifest.jsonl",
        test_manifest: str = "test_manifest.jsonl",
        words_file: str | None = None,
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
        accelerator: str = "auto",
        devices: int | str = "auto",
        learning_rate: float | None = None,
        tensorboard: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        """Finetune the NeMo command-recognition model from local manifests."""
        _configure_logging(verbose=verbose)
        run = train_model(
            project_dir=Path(project_dir),
            data_dir=Path(data_dir),
            runs_dir=Path(runs_dir),
            run_name=run_name,
            model_name=model_name,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            test_manifest=test_manifest,
            words_file=Path(words_file) if words_file else None,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator=accelerator,
            devices=devices,
            learning_rate=learning_rate,
            tensorboard=tensorboard,
            dry_run=dry_run,
        )

        for output in run.paths():
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
        "--downloads-dir": "--downloads_dir",
        "--google-speech-commands": "--google_speech_commands",
        "--common-voice-sw": "--common_voice_sw",
        "--noises-dir": "--noises_dir",
        "--train-ratio": "--train_ratio",
        "--validate-ratio": "--validate_ratio",
        "--test-ratio": "--test_ratio",
        "--train-filename": "--train_filename",
        "--validate-filename": "--validate_filename",
        "--test-filename": "--test_filename",
        "--model-id": "--model_id",
        "--model-name": "--model_name",
        "--sample-rate": "--sample_rate",
        "--project-dir": "--project_dir",
        "--runs-dir": "--runs_dir",
        "--run-name": "--run_name",
        "--train-manifest": "--train_manifest",
        "--validation-manifest": "--validation_manifest",
        "--test-manifest": "--test_manifest",
        "--words-file": "--words_file",
        "--max-epochs": "--max_epochs",
        "--batch-size": "--batch_size",
        "--num-workers": "--num_workers",
        "--learning-rate": "--learning_rate",
        "--dry-run": "--dry_run",
    }
    sys.argv = [_normalize_flag(arg, flag_aliases) for arg in sys.argv]


def _normalize_flag(arg: str, flag_aliases: dict[str, str]) -> str:
    if arg.startswith("—"):
        arg = f"--{arg[1:]}"
    flag, separator, value = arg.partition("=")
    return f"{flag_aliases.get(flag, flag)}{separator}{value}"


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    main()
