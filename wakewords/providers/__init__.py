from __future__ import annotations

from wakewords.providers.base import TTSProvider
from wakewords.providers.cartesia import CartesiaProvider


def get_provider(name: str) -> TTSProvider:
    normalized = name.strip().lower()

    if normalized == "cartesia":
        return CartesiaProvider()

    raise ValueError(f"Unsupported TTS provider: {name}")
