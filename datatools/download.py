from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tqdm import tqdm

GOOGLE_SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GOOGLE_SPEECH_COMMANDS_ARCHIVE = "speech_commands_v0.02.tar.gz"
GOOGLE_SPEECH_COMMANDS_DIR = "google-speech-commands"

COMMON_VOICE_DATASET_ID = "cmkzhp64p00wlno07elrmt20y"
COMMON_VOICE_DOWNLOAD_API = (
    f"https://mozilladatacollective.com/api/datasets/{COMMON_VOICE_DATASET_ID}/download"
)
COMMON_VOICE_ARCHIVE = "common-voice-7-single-word.tar.gz"
COMMON_VOICE_DIR = "common-voice-7-single-word"


def download_datasets(
    *,
    google_speech_commands: bool = False,
    common_voice_sw: bool = False,
    all: bool = False,
    downloads_dir: Path | None = None,
    data_dir: Path = Path("data"),
) -> list[Path]:
    if all:
        google_speech_commands = True
        common_voice_sw = True
    if not google_speech_commands and not common_voice_sw:
        raise ValueError(
            "Select at least one dataset with --google-speech-commands, --common-voice-sw, or --all."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if downloads_dir is None:
        with tempfile.TemporaryDirectory(prefix=".datatools-download-", dir=Path.cwd()) as tmp_dir:
            outputs.extend(
                _download_selected(
                    google_speech_commands=google_speech_commands,
                    common_voice_sw=common_voice_sw,
                    downloads_dir=Path(tmp_dir),
                    data_dir=data_dir,
                )
            )
    else:
        downloads_dir.mkdir(parents=True, exist_ok=True)
        outputs.extend(
            _download_selected(
                google_speech_commands=google_speech_commands,
                common_voice_sw=common_voice_sw,
                downloads_dir=downloads_dir,
                data_dir=data_dir,
            )
        )

    return outputs


def _download_selected(
    *,
    google_speech_commands: bool,
    common_voice_sw: bool,
    downloads_dir: Path,
    data_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    if google_speech_commands:
        archive_path = downloads_dir / GOOGLE_SPEECH_COMMANDS_ARCHIVE
        _download_file(GOOGLE_SPEECH_COMMANDS_URL, archive_path, description="Google Speech Commands")
        output_dir = data_dir / GOOGLE_SPEECH_COMMANDS_DIR
        _extract_tar(archive_path, output_dir, description="Extract Google Speech Commands")
        outputs.append(output_dir)

    if common_voice_sw:
        archive_path = downloads_dir / COMMON_VOICE_ARCHIVE
        download_url = _common_voice_download_url()
        _download_file(download_url, archive_path, description="Common Voice 7.0 single word")
        output_dir = data_dir / COMMON_VOICE_DIR
        _extract_tar(archive_path, output_dir, description="Extract Common Voice 7.0 single word")
        outputs.append(output_dir)

    return outputs


def _common_voice_download_url() -> str:
    api_key = os.environ.get("COMMONVOICE_API_KEY")
    if not api_key:
        raise ValueError("COMMONVOICE_API_KEY must be set to download Common Voice.")

    request = Request(
        COMMON_VOICE_DOWNLOAD_API,
        data=b"{}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Common Voice download URL request failed: {message}") from error

    download_url = payload.get("downloadUrl")
    if not isinstance(download_url, str) or not download_url:
        raise RuntimeError("Common Voice response did not include downloadUrl.")
    return download_url


def _download_file(url: str, output_path: Path, *, description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "tincan-wakewords-datatools/0.1"})
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
