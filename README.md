# ToxTransformer research

ToxTransformer predicts binary molecular properties from a SELFIES representation of a molecule and, optionally, other observed property values. This repository contains its PyTorch architecture, tokenizers, a local training and evaluation workflow, and historical preprocessing and training source.

This is a research derivative of [toxindex/toxtransformer](https://github.com/toxindex/toxtransformer), with independent Git history. The upstream service, cloud infrastructure, partner-specific analyses, and other model projects are outside this repository.

New readers and integrators: [start here](docs/getting-started.md) for a reading path, prediction meanings, public links, and the distinction between local research and hosted access.

## What can be reproduced

| Task | Status |
| --- | --- |
| Study the architecture used by an inspected upstream checkpoint | Included; strict checkpoint loading verified |
| Train, save, evaluate, and query a small model locally | Tested on CPU with the included artificial dataset |
| Repeat the small run with the same seed and environment | Tested; learned tensors and epoch metrics match exactly |
| Load the inspected 6,647-property checkpoint | Weights in Git LFS; tokenizers, property mapping, checksums, and numerical reproduction examples included |
| Recreate the deployed weights or historical benchmark results from scratch | Not yet verified; training snapshots, label definitions, splits, and run provenance remain to be released |

The released checkpoint has **59,218,946 parameters**, 16 causal transformer layers, hidden width 512, 8 attention heads, and 6,647 property embeddings. These dimensions come from the checkpoint. See [architecture](docs/architecture.md) and [artifact checksums](artifacts/upstream-checkpoint.json).

## Reproduce pretrained predictions locally

Install [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/). The tested setup is Linux, Python 3.11, and CPU PyTorch. The weights download is approximately 237 MB; allow additional space for the Python environment.

```bash
git lfs install
git clone https://github.com/toxindex/toxtransformer-research.git
cd toxtransformer-research
git lfs pull
uv sync --locked --extra dev --python 3.11

# Verify every artifact and reproduce four recorded CPU predictions.
uv run --locked python -m toxtransformer_research.reproduce

# Predict a property locally; output includes its identifier, source, and title.
uv run --locked python -m toxtransformer_research.checkpoint \
  --manifest artifacts/upstream-checkpoint.json --smiles CCO --property-index 0

# Find indices by source, identifier, or available title.
uv run --locked python -m toxtransformer_research.checkpoint \
  --list-properties --search tox21
```

The verification command reports `checksums: passed` and `strict_load: passed`. The structure-only ethanol prediction for property index 0 is approximately `0.08650372` in the tested CPU environment. These examples check numerical reproduction, not scientific accuracy. The [local setup guide](docs/local-reproduction.md) covers Python use, known property context, troubleshooting, and the distinction between inference reproduction and retraining.

## Run the research workflow

After the setup above, the reference workflow can train a small model from scratch. The lockfile selects a CPU build of PyTorch; no GPU, cloud account, or service credentials are required.

```bash
uv run --locked pytest -q

uv run --locked toxtransformer train \
  --data examples/toy.csv --config configs/smoke.json --output runs/demo

uv run --locked toxtransformer evaluate \
  --run runs/demo --data examples/toy.csv --split test --output runs/demo/test.json

uv run --locked toxtransformer predict \
  --run runs/demo --smiles CCO --property-id toy_oxygen
```

`examples/toy.csv` contains 24 simple molecules and three artificial labels derived from their SMILES strings. It exercises the software. Its metrics do not measure toxicology prediction quality. Training refuses to overwrite an existing run directory.

For new data, provide a CSV with `smiles,property_id,value,split` columns. Values must be `0` or `1`, and splits must be `train`, `validation`, or `test`. See the [data contract](docs/data.md) before running an experiment.

## Read the model

| Path | Purpose |
| --- | --- |
| `src/cvae/multitask_encoder.py` | Causal attention, embeddings, RMSNorm, SwiGLU, rotary positions, span masking, and checkpoint I/O |
| `src/cvae/tokenizer/` | SELFIES vocabulary and property/value tokenizer formats |
| `src/toxtransformer_research/` | Dataset validation, deterministic training, evaluation, prediction, and checkpoint inspection |
| `models/toxtransformer/` | Git LFS weights, portable tokenizers, and the 6,647-entry property mapping |
| `configs/upstream-architecture.json` | Architecture configuration read from the inspected checkpoint |
| `configs/research.json` | Starting configuration for new experiments; it is not the historical training recipe |
| `reference/` | Historical preprocessing, sampling, optimization, evaluation, and optional CUDA code |
| `tests/` | Causal isolation, padding, checkpoint compatibility, split validation, and repeatability checks |

Start with [architecture](docs/architecture.md), then [reproduction](docs/reproduction.md). [Provenance](docs/provenance.md) describes copied source and deliberate changes. The [model card](MODEL_CARD.md) states the limits of the available evidence.

## License and artifacts

Code and the released model weights use the [MIT license](LICENSE). The property catalog preserves upstream property identifiers, descriptive titles where available, and source attribution. No measured activity records, production credentials, prediction cache, or historical training dataset are included. Consult the original data sources for endpoint definitions and source-specific data terms.
