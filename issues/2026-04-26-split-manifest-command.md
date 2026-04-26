Status: DONE

Problem

The project had per-word manifests but no command to assemble dataset-level train, validation, and test manifests for NeMo training. There was also a mismatch between local manifest placement and output manifest path requirements: local manifests live inside `data/<word>/`, so they should store local filenames, while training manifests need resolved dataset paths.

Solution

Added `datatools manifest`, which reads `data/<word>/manifest.jsonl`, resolves local filenames to full paths, and writes split manifests under `data/` using configurable train, validation, and test ratios. The split is deterministic and applied per label with small rounding adjustments when exact integer splits are impossible. Per-word manifests now store local filenames, and older absolute-path entries are normalized on load.
