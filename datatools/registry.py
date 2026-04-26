"""Voice registry that maps provider+voice_id to short stable codes like cr1, cr2.

File format (`voices.<provider>.txt`) — one entry per line, append-only:
    cr1 a1b2-c3d4-e5f6
    cr2 c3d4-e5f6-a1b2
"""
from __future__ import annotations

import threading
from pathlib import Path


_DEFAULT_PATH = Path("data/voices.provider.txt")


class VoiceRegistry:
    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], str] = {}  # (provider, voice_id) -> short_code
        self._counters: dict[str, int] = {}  # provider -> current max index
        self._load()

    def short_code(self, provider: str, voice_id: str) -> str:
        """Return the short code for a voice, registering it if not already known."""
        key = (provider, voice_id)
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            idx = self._counters.get(provider, 0) + 1
            code = f"{provider}{idx}"
            self._entries[key] = code
            self._counters[provider] = idx
            self._append(code=code, voice_id=voice_id)
            return code

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            code, voice_id = line.split(None, 1)
            # derive provider prefix (everything before the trailing digits)
            provider = code.rstrip("0123456789")
            self._entries[(provider, voice_id)] = code
            suffix = code[len(provider):]
            if suffix.isdigit():
                self._counters[provider] = max(self._counters.get(provider, 0), int(suffix))

    def _append(self, *, code: str, voice_id: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(f"{code} {voice_id}\n")
