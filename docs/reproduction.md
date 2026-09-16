# Reproducing runs and inspecting the upstream checkpoint

## Repeat a reference run

The README commands train a small instance of the architecture. Run them in a second, unused output directory with the same CSV, configuration, source revision, and locked environment to repeat the experiment.

Each run writes:

| File | Contents |
| --- | --- |
| `run.json` | Input/config checksums, seed, requested and resolved configuration, package versions, device, source revision, and parameter counts |
| `splits.json` | Canonical compound identities assigned to each partition |
| `properties.json` | Ordered property identifiers corresponding to model indices |
| `history.json` | Training loss and structure-only validation metrics for each epoch |
| `best/` | Checkpoint and fitted tokenizer from the epoch with lowest validation loss |

The new trainer uses AdamW, shuffled observed-property sequences, gradient clipping, and a fixed learning rate. It sets Python, NumPy, and PyTorch seeds, uses one CPU thread, and requests deterministic PyTorch operations. Unsupported deterministic GPU operations fail explicitly. Reproducibility across different PyTorch versions, devices, or hardware is not guaranteed. The tested claim is identical tensors and epoch metrics for two runs in the same CPU environment.

The CSV reader and trainer hold the grouped dataset in memory. This implementation is intended as an inspectable reference and is not validated for the complete historical corpus. Full-scale work may need a streaming input pipeline, distributed training, and a deliberate property sampling strategy.

## Train on a GPU

`uv sync` uses the CPU lockfile. For GPU work, create a separate environment and install a PyTorch 2.8.0 build suitable for the target GPU. For a CUDA 12.8 environment:

```bash
uv venv --python 3.11 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu/bin/python -e .
.venv-gpu/bin/toxtransformer train \
  --data data/observations.csv --config configs/research.json \
  --device cuda --output runs/research
```

The CUDA commands have not been executed in this repository's CPU validation. Pin and record the complete GPU environment for a published experiment. `run.json` records the core scientific package versions and GPU identity; the CPU lockfile is not a lock for the separate GPU environment.

`configs/research.json` is a starting point for new experiments. Its optimizer settings, batch size, epochs, data splitting, and sampling are not asserted to match the historical checkpoint's training run.

## Load the inspected upstream checkpoint

Run `git lfs pull` to download the weights. The complete inference bundle is versioned with the repository:

```text
models/toxtransformer/
  multitask_encoder.pt
  properties.json
  spvt_tokenizer/
    selfies_property_val_tokenizer.json
    selfies_tokenizer.json
```

Verify hashes, load every parameter strictly, and inspect the architecture:

```bash
uv run --locked python -m toxtransformer_research.checkpoint \
  --checkpoint models/toxtransformer --manifest artifacts/upstream-checkpoint.json
```

Run one structure-only prediction by zero-based property index:

```bash
uv run --locked python -m toxtransformer_research.checkpoint \
  --checkpoint models/toxtransformer --manifest artifacts/upstream-checkpoint.json \
  --smiles CCO --property-index 0
```

`properties.json` maps each index to its source identifier and available title. Endpoint definitions and label thresholds still require the original data sources. The local loader was tested against this stored checkpoint; its equivalence to every currently deployed service image has not been established. The [local reproduction guide](local-reproduction.md) explains the unpadded upstream inference convention and provides a single numerical verification command.

## Historical training recipe

Additional historical model-training source was recovered at upstream commit `cc51ae0d1c96fc80e3726bf50dfe968c2f97a213` and placed in `reference/historical_training/`. It includes the full-data and bootstrap training entry points, schedulers, trainers, evaluators, and dataset sampling classes. It is an archival reference, not an installed or tested command in this package.

The full-data trainer uses the V3 architecture, custom property sampling, gradient accumulation, and warmup/cosine scheduling. Its comments state that validation reuses a bootstrap test set while training on final tensors containing all data. That validation is a training monitor, not evidence of held-out generalization. The bootstrap trainer has a separate role. Preserve these distinctions when comparing results.

## Remaining requirements for exact historical reproduction

The following items are not yet established as a single verified release:

1. Immutable training, validation, and benchmark data snapshots, including the augmented activities and final tensors, with endpoint definitions and label thresholds.
2. The specific source revision, effective optimizer/sampler/scheduler settings, seeds, and environment that produced the inspected weights. Recovered historical code is evidence, but does not prove that association.
3. Benchmark predictions and evaluation commands tied to those artifacts, with explicit split and context definitions.

The weights, tokenizer, property-index mapping, and numerical inference examples are now distributed in this repository. The independent research workflow can also be reproduced. Exact retraining of the historical weights and benchmark scores remains unverified until the remaining artifacts and their lineage are supplied.
