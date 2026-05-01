from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

from wakewords.project import BACKGROUND_AUDIO_URL, _extract_background_audio_archive

GOOGLE_SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GOOGLE_SPEECH_COMMANDS_ARCHIVE = "speech_commands_v0.02.tar.gz"
GOOGLE_SPEECH_COMMANDS_DIR = "google-speech-commands"
BACKGROUND_AUDIO_ARCHIVE = "background_audio.zip"
BACKGROUND_AUDIO_DIR = "background_audio"


def download_datasets(
    *,
    downloads_dir: Path | None = None,
    data_dir: Path = Path("."),
) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if downloads_dir is None:
        with tempfile.TemporaryDirectory(prefix=".wakewords-download-", dir=Path.cwd()) as tmp_dir:
            outputs.extend(
                _download_selected(
                    downloads_dir=Path(tmp_dir),
                    data_dir=data_dir,
                )
            )
    else:
        downloads_dir.mkdir(parents=True, exist_ok=True)
        outputs.extend(
            _download_selected(
                downloads_dir=downloads_dir,
                data_dir=data_dir,
            )
        )

    return outputs


def _download_selected(
    *,
    downloads_dir: Path,
    data_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []

    if _should_download_google_speech_commands(data_dir / "config.json"):
        archive_path = downloads_dir / GOOGLE_SPEECH_COMMANDS_ARCHIVE
        _download_file(GOOGLE_SPEECH_COMMANDS_URL, archive_path, description="Google Speech Commands")
        dataset_dir = data_dir / GOOGLE_SPEECH_COMMANDS_DIR
        _extract_tar(archive_path, dataset_dir, description="Extract Google Speech Commands")
        outputs.append(dataset_dir)

    background_archive_path = downloads_dir / BACKGROUND_AUDIO_ARCHIVE
    _download_file(BACKGROUND_AUDIO_URL, background_archive_path, description="Background Audio")
    background_audio_dir = data_dir / BACKGROUND_AUDIO_DIR
    _extract_zip(background_archive_path, background_audio_dir)
    outputs.append(background_audio_dir)

    return outputs


def _should_download_google_speech_commands(config_path: Path) -> bool:
    if not config_path.exists():
        return True

    config = json.loads(config_path.read_text(encoding="utf-8"))
    google_speech_commands = config.get("google_speech_commands")
    if isinstance(google_speech_commands, list):
        return bool(google_speech_commands)
    return True


def _download_file(url: str, output_path: Path, *, description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "wakewords/0.1"})
    with urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with output_path.open("wb") as output_file:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=description,
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    progress.update(len(chunk))


def _extract_tar(archive_path: Path, output_dir: Path, *, description: str) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in tqdm(members, desc=description, unit="file"):
            _safe_extract_member(archive, member, output_dir)


def _safe_extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, output_dir: Path) -> None:
    if member.issym() or member.islnk():
        raise RuntimeError(f"Archive member uses a link, which is not allowed: {member.name}")

    target = (output_dir / member.name).resolve()
    output_root = output_dir.resolve()
    if output_root != target and output_root not in target.parents:
        raise RuntimeError(f"Archive member escapes output directory: {member.name}")
    archive.extract(member, output_dir)


def _extract_zip(archive_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _extract_background_audio_archive(archive_path, output_dir)
