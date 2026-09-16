# ToxTransformer research model card

## Intended research use

Study molecular property prediction, conditioning on observed assay results, and the behavior of a causal transformer over SELFIES and property/value sequences. The package supports local training on user-supplied binary observations and explicit evaluation partitions.

## Architecture and inspected artifact

The released upstream checkpoint contains a causal transformer with 16 layers, hidden width 512, 8 heads, and 59,218,946 parameters. Its property vocabulary has 6,647 entries and its output classes are binary. `artifacts/upstream-checkpoint.json` identifies all bundle files by SHA-256. Weights are distributed through Git LFS under `models/toxtransformer/`, together with tokenizers and the complete property-index catalog. No activity records are bundled.

The new training workflow uses the same architecture implementation but a separately documented optimization and data protocol. A model trained with `configs/research.json` is a new experiment, not a reproduced historical model by default.

## Training data evidence

Historical preprocessing code reads ChemHarmony through BioBricks and constructs binary property observations. The exact immutable dataset snapshot and all transformations used for the inspected weights remain unverified. The bundled catalog preserves the original property mapping; endpoint definitions and binary-label thresholds require the original data sources.

The bundled toy data has artificial labels such as presence of oxygen in a simple SMILES string. It is not measured biological activity and must not be used to report toxicological performance.

## Evaluation evidence

Verified software properties include strict checkpoint loading, recorded CPU inference examples, finite forward/backward passes, save/load prediction equality, causal exclusion of target/future labels, and repeatable small CPU training runs. The pretrained inference helper matches the upstream single-molecule convention with no padding gap before property queries. These checks do not validate prediction accuracy on a scientific benchmark.

The reference evaluator reports per-property and pooled metrics, class counts, macro AUC coverage, and the amount of observed context. No historical benchmark score is asserted here. Evidence from observation-level splits, novel-compound splits, scaffold splits, and full-data training monitors must be distinguished.

## Limitations

- The workflow supports binary properties only.
- A property value's meaning depends on the endpoint definition and label threshold used to build the dataset.
- Unknown molecular vocabulary symbols and untrained properties are rejected.
- Predictions depend on the supplied property context and its ordering.
- Calibration, uncertainty estimates, applicability domain, and external validation have not been established by this extraction.
- The in-memory reference trainer is not validated at the scale of the full historical corpus.

## Release status

Code and released weights are available under the included MIT license. Property metadata retains source attribution. Training data, exact historical run provenance, and scientific benchmark validation remain separate release work.
