# Local Base Model Override

Status: DONE

## Problem

Training briefly assumed the base MatchboxNet model should be downloaded into the
project by `wakewords download`. That made the download command responsible for
model artifacts even though NeMo already supports loading the target model by
name with `from_pretrained()`.

## Solution

Removed model downloading from `wakewords download`. Training now uses
`from_pretrained(model_name=...)` unless `--base-model-path` is explicitly
provided, in which case it restores from that local `.nemo` file.
