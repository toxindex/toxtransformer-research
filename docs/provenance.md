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

## Historical reference files

Historical preprocessing, training, and evaluation files retain their original imports and operational assumptions. They are not imported by the installed package. They may require omitted legacy modules, Spark, tensor datasets, and a distributed GPU environment. Their purpose is to preserve scientific implementation evidence for reconstructing the original pipeline. The original commands are not presented as supported reproduction instructions.

The optional CUDA source and its build script are retained under `reference/cuda/`. The build script's source path was adjusted for that directory and a deployment-specific comment was generalized. The extension is not required by the research workflow and was not compiled during CPU validation.
