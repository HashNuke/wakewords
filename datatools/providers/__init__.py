from __future__ import annotations

from datatools.providers.base import TTSProvider
from datatools.providers.cartesia import CartesiaProvider


def get_provider(name: str) -> TTSProvider:
    normalized = name.strip().lower()

    if normalized == "cartesia":
        return CartesiaProvider()

    raise ValueError(f"Unsupported TTS provider: {name}")
