from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


class GitLfsPointerError(RuntimeError):
    pass


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def require_materialized_files(
    paths: Iterable[Path],
    *,
    context: str,
    include_hint: str | None = None,
    max_reported: int = 8,
) -> None:
    pointer_paths: list[Path] = []
    extra_count = 0
    for path in paths:
        if not is_git_lfs_pointer(path):
            continue
        if len(pointer_paths) < max_reported:
            pointer_paths.append(path)
        else:
            extra_count += 1

    if not pointer_paths:
        return

    lines = [f"Git LFS files need to be pulled before {context}:"]
    lines.extend(f"  - {path}" for path in pointer_paths)
    if extra_count:
        lines.append(f"  - ... and {extra_count} more")
    lines.append("Run `git lfs pull` from the project repository, then rerun the command.")
    if include_hint:
        lines.append(f"For just these inputs, try: `git lfs pull --include=\"{include_hint}\"`")
    raise GitLfsPointerError("\n".join(lines))


def manifest_audio_paths(manifest_paths: Iterable[Path]) -> list[Path]:
    audio_paths: list[Path] = []
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in manifest {manifest_path}:{line_number}") from exc
            audio_filepath = entry.get("audio_filepath")
            if not isinstance(audio_filepath, str) or not audio_filepath:
                continue
            audio_path = Path(audio_filepath)
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            audio_paths.append(audio_path)
    return audio_paths
