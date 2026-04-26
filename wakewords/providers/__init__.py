from __future__ import annotations

from wakeword.providers.base import TTSProvider
from wakeword.providers.cartesia import CartesiaProvider


def get_provider(name: str) -> TTSProvider:
    normalized = name.strip().lower()

    if normalized == "cartesia":
        return CartesiaProvider()

    raise ValueError(f"Unsupported TTS provider: {name}")
