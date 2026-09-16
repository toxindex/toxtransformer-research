# Reproduce the released model locally

## Download and verify

The pretrained inference bundle is under `models/toxtransformer/`. The 236,925,619-byte PyTorch checkpoint is tracked with Git LFS. Tokenizers and the property catalog are regular Git files. The tensor checkpoint is byte-for-byte identical to the inspected upstream artifact; only the tokenizer's stored filename was made portable.

```bash
git lfs install
git clone https://github.com/toxindex/toxtransformer-research.git
cd toxtransformer-research
git lfs pull
uv sync --locked --extra dev --python 3.11
uv run --locked python -m toxtransformer_research.reproduce
uv run --locked pytest -q
```

`reproduce` hashes all four artifact files, loads every model parameter strictly, checks the property mapping, and compares four predictions with `artifacts/reference-predictions.json`. The comparison uses CPU float32 and an absolute tolerance of `1e-5`. One example supplies artificial context labels solely to exercise that code path. None of these examples is an accuracy benchmark.

The manifest's weight SHA-256 is `8cae8b6490635e760a4dc00a66fb2626d8b3b8c1fba97fb7e0daee2935deeb1b`. `git lfs fsck` checks local LFS objects independently.

## Predict and identify the output

```bash
uv run --locked python -m toxtransformer_research.checkpoint \
  --manifest artifacts/upstream-checkpoint.json \
  --smiles CCO --property-index 0

uv run --locked python -m toxtransformer_research.checkpoint \
  --list-properties --search tox21
```

The first command emits JSON on stdout. Loading messages go to stderr. Prediction output includes the stable property identifier, original source, available title, probability of class 1, and context count. You can substitute `--property-id` for `--property-index` using an identifier returned by the catalog search.

The catalog has a unique entry for every index from 0 through 6,646. Titles are unavailable for 1,670 entries and remain `null`; they are not invented. Titles are descriptive labels inherited from upstream and are not a substitute for the original assay definition or binary-label threshold. The `source` field records the upstream source name.

## Use Python and observed context

Run this code in the installed environment:

```python
from toxtransformer_research.checkpoint import (
    load_checkpoint, predict_checkpoint, verify_manifest,
)

directory = "models/toxtransformer"
verify_manifest(directory, "artifacts/upstream-checkpoint.json")
model = load_checkpoint(directory)
result = predict_checkpoint(model, "CCO", property_index=0)
print(result)

# Optional: supply actual observed labels for other properties, in chosen order.
# These values are illustrative software inputs, not claims about ethanol.
result = predict_checkpoint(model, "CCO", property_index=0, context=[(1, 0), (2, 1)])
```

Context entries are `(property_index, binary_value)` pairs. The function rejects duplicate properties and the target itself in context. This interface uses exactly the supplied order and values. It does not retrieve measured activities, use the hosted service's cache, or select context through a mutual-information table.

## Molecular token positions matter

The pretrained command uses the upstream single-molecule inference convention: SELFIES start/end tokens, followed immediately by property/value tokens, with no right-padding gap. Changing that gap changes rotary-position relationships and can change the prediction. Earlier research examples used 120-position padding; the released inference command corrects that mismatch. The small training workflow retains its explicitly configured fixed token length and is a separate experiment.

The inference helper accepts up to 120 molecular tokens by default and rejects overlong or unknown-token inputs. Pass a larger `max_selfies` only for a deliberate experiment within the model's total sequence limit. The helper never silently truncates input.

## Common setup failures

| Symptom | Action |
| --- | --- |
| The `.pt` file is a short text file beginning with `version https://git-lfs.github.com/spec/v1` | Run `git lfs install` and `git lfs pull` from the cloned repository. |
| A ZIP download contains pointer files | Use the Git clone instructions; source archives are not the tested model distribution path. |
| Artifact checksum mismatch | Restore the model files from the selected Git revision and run `git lfs pull`; do not bypass the mismatch. |
| LFS reports a storage/bandwidth restriction | Report the failure to repository maintainers. The small pointer file is not usable model data. |
| CUDA is unavailable | The locked setup is CPU-only. Use the separate GPU environment described in `reproduction.md`; CPU is the numerical reference. |
| A molecule has a SELFIES symbol absent from the vocabulary | Use an in-vocabulary molecule or document a new vocabulary/training experiment. |

For a lightweight source-only checkout, set `GIT_LFS_SKIP_SMUDGE=1` when cloning. Run `git lfs pull` before model reproduction or the full test suite. GitHub Actions explicitly downloads LFS weights and runs the model checks.

## What this does and does not reproduce

The released bundle reproduces local inference from the stored checkpoint, including explicit observed property context. It also includes a tested small training workflow and historical source for study. The exact historical training corpus, transformations, training-run association, and benchmark outputs remain unverified as a single distributable release. See `reproduction.md` before making an exact-retraining or benchmark claim.
