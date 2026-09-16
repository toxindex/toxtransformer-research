# ToxTransformer research

ToxTransformer predicts binary molecular properties from a SELFIES representation of a molecule and, optionally, other observed property values. This repository contains its PyTorch architecture, tokenizers, a local training and evaluation workflow, and historical preprocessing and training source.

This is a research derivative of [toxindex/toxtransformer](https://github.com/toxindex/toxtransformer), with independent Git history. The upstream service, cloud infrastructure, partner-specific analyses, and other model projects are outside this repository.

## What can be reproduced

| Task | Status |
| --- | --- |
| Study the architecture used by an inspected upstream checkpoint | Included; strict checkpoint loading verified |
| Train, save, evaluate, and query a small model locally | Tested on CPU with the included artificial dataset |
| Repeat the small run with the same seed and environment | Tested; learned tensors and epoch metrics match exactly |
| Load the inspected 6,647-property checkpoint | Loader and artifact checksums included; weights must be obtained separately |
| Recreate the deployed weights or historical benchmark results from scratch | Not yet verified; training snapshots, mappings, splits, and run provenance remain to be released |

The inspected checkpoint has **59,218,946 parameters**, 16 causal transformer layers, hidden width 512, 8 attention heads, and 6,647 property embeddings. These dimensions come from the checkpoint. They are not inferred from an advertised model name. See [architecture](docs/architecture.md) and [artifact checksums](artifacts/upstream-checkpoint.json).

## Run the research workflow

Use Python 3.11 and `uv`. The lockfile selects a CPU build of PyTorch so this example needs no GPU, cloud account, or service credentials.

```bash
git clone https://github.com/toxindex/toxtransformer-research.git
cd toxtransformer-research
uv sync --locked --extra dev --python 3.11
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
| `configs/upstream-architecture.json` | Architecture configuration read from the inspected checkpoint |
| `configs/research.json` | Starting configuration for new experiments; it is not the historical training recipe |
| `reference/` | Historical preprocessing, sampling, optimization, evaluation, and optional CUDA code |
| `tests/` | Causal isolation, padding, checkpoint compatibility, split validation, and repeatability checks |

Start with [architecture](docs/architecture.md), then [reproduction](docs/reproduction.md). [Provenance](docs/provenance.md) describes copied source and deliberate changes. The [model card](MODEL_CARD.md) states the limits of the available evidence.

## License and artifacts

Code retains the upstream [MIT license](LICENSE). No production credentials, deployed prediction cache, historical training dataset, or pretrained weights are committed here. Dataset and weight distribution terms must be supplied with their eventual releases; the code license alone does not document those terms.
