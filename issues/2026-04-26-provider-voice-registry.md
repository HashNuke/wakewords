Status: DONE

Problem

The voice registry was stored in a shared `data/voices.txt` file. That mixed mappings across providers and made the filename less explicit about which provider the registry belonged to.

Solution

Changed provider-specific generation to write and read registry entries from `data/voices.<provider>.txt`, for example `data/voices.cartesia.txt`. Documentation was updated to reflect the provider-specific registry naming.
