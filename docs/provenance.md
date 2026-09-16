# Source provenance and deliberate changes

The main architecture extraction comes from `toxindex/toxtransformer` commit `bdd340eb4d69a913f251504d55f9d2aca5b29c40`. The historical training reference comes from commit `cc51ae0d1c96fc80e3726bf50dfe968c2f97a213`. `provenance.json` lists source paths, destination paths, and hashes of the original files. Hashes describe the source before the modifications below. Extracted text files have trailing whitespace and line endings normalized.

The repository begins with an independent root commit. It does not import upstream Git history, pull-request references, other branches, deployment configuration, partner analyses, cached prediction data, or the earlier credential-bearing file. The MIT copyright and license text are preserved.

## Architecture extraction

The active architecture remains in the `cvae` package to preserve the upstream naming. Changes to the copied implementation are limited to:

- Importing `torch.utils.checkpoint` explicitly so gradient checkpointing works independently of incidental imports.
- Loading checkpoints with `weights_only=True` and strict state-dictionary matching.
- Correcting comments that called property-query positions value positions.
- Removing two unverified benchmark annotations from architecture presets.
- Making Spark imports lazy in the SELFIES tokenizer, so model use does not require Spark.
- Sorting new Spark vocabulary entries. Existing checkpoint vocabularies are loaded unchanged.
- Replacing the broad upstream utility module with the directory helper used by checkpoint saving.

The embedding, attention, decoder, classifier, and state-dictionary parameter names are retained. Strict loading of the inspected 59,218,946-parameter checkpoint verifies parameter compatibility. Tests also cover causal isolation, padding, attention implementation agreement, gradients, and save/load prediction equality.

## New research workflow

`src/toxtransformer_research/`, the runnable configurations, artificial fixture, and tests are new. They provide an explicit research baseline and do not claim to recreate the historical training procedure. The workflow fits vocabularies on training data, checks canonical compound overlap, rejects invalid labels and silent truncation, records run provenance, and evaluates structure-only and observed-context predictions separately.

## Distributed checkpoint and catalog

The Git LFS checkpoint is byte-identical to the inspected upstream weights; its original SHA-256 is retained in the artifact manifest. The state dictionary contains tensors only and the checkpoint metadata is empty. The tokenizer's stored file path was changed from a training-directory path to the portable filename `selfies_tokenizer.json`; its vocabulary indices are unchanged. The second tokenizer file is byte-identical to upstream.

`models/toxtransformer/properties.json` contains only property index, stable identifier, available descriptive title, and original source name from the database used by upstream inference. Its indices cover the checkpoint vocabulary exactly. No raw assay metadata, measured activity rows, compound histories, deployment identifiers, or credentials are included. The catalog contains 1,670 missing titles, preserved as `null`.

The release adds an unpadded pretrained inference helper and numerical regression examples. This corrects the initial research example's padding gap to match upstream single-molecule inference. The configurable fixed-length representation in the separate from-scratch trainer remains unchanged.

## Historical reference files

Historical preprocessing, training, and evaluation files retain their original imports and operational assumptions. They are not imported by the installed package. They may require omitted legacy modules, Spark, tensor datasets, and a distributed GPU environment. Their purpose is to preserve scientific implementation evidence for reconstructing the original pipeline. The original commands are not presented as supported reproduction instructions.

The optional CUDA source and its build script are retained under `reference/cuda/`. The build script's source path was adjusted for that directory and a deployment-specific comment was generalized. The extension is not required by the research workflow and was not compiled during CPU validation.
