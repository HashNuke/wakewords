from __future__ import annotations

import importlib
import json
from pathlib import Path

from wakewords.providers.base import TTSProvider


BUILTIN_PROVIDERS = {
    "cartesia": "wakewords.providers.cartesia:CartesiaProvider",
}


def get_provider(name: str, *, config_path: Path = Path("config.json")) -> TTSProvider:
    normalized = name.strip().lower()
    providers = _provider_specs(config_path)

    provider_spec = providers.get(normalized)
    if provider_spec is None:
        supported = ", ".join(sorted(providers))
        raise ValueError(f"Unsupported TTS provider: {name}. Supported providers: {supported}")

    provider = _load_provider(provider_spec)
    _validate_provider(provider, normalized)
    return provider


def _provider_specs(config_path: Path) -> dict[str, str]:
    providers = dict(BUILTIN_PROVIDERS)
    if not config_path.exists():
        return providers

    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_providers = config.get("providers", {})
    if configured_providers is None:
        return providers
    if not isinstance(configured_providers, dict):
        raise ValueError("config.json providers must be an object mapping names to import paths")

    for provider_name, provider_spec in configured_providers.items():
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("Provider names in config.json must be non-empty strings")
        if not isinstance(provider_spec, str) or not provider_spec.strip():
            raise ValueError(f"Provider import path for {provider_name!r} must be a non-empty string")
        providers[provider_name.strip().lower()] = provider_spec.strip()
    return providers


def _load_provider(provider_spec: str) -> TTSProvider:
    module_name, separator, attribute_name = provider_spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"Provider import path must use 'module:attribute': {provider_spec}")

    module = importlib.import_module(module_name)
    provider_factory = getattr(module, attribute_name)
    provider = provider_factory() if callable(provider_factory) else provider_factory
    return provider


def _validate_provider(provider: TTSProvider, provider_name: str) -> None:
    missing = [name for name in ("list_voices", "generate") if not callable(getattr(provider, name, None))]
    if missing:
        missing_methods = ", ".join(missing)
        raise TypeError(f"Provider {provider_name!r} is missing required methods: {missing_methods}")
