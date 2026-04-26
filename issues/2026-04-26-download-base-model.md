# Download Base Model

Status: DONE

## Problem

The download command only fetched external dataset assets. Training still loaded
the MatchboxNet base checkpoint by model name at training time, which meant the
project directory did not contain the base model artifact that would be used for
finetuning.

## Solution

Changed `wakewords download` to require no dataset/model selection flags and to
download both Google Speech Commands and the NGC MatchboxNet base `.nemo` file.
The base model is stored at
`models/base/commandrecognition_en_matchboxnet3x2x64_v2.nemo`. Updated training
to restore from that local model path, with a clear error asking the user to run
`wakewords download` if it is missing.
