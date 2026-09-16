# Historical scientific source

These files preserve implementation evidence from the upstream project. They are excluded from the installed package and from the supported test path.

| Directory | Contents | Status |
| --- | --- | --- |
| `upstream/` | ChemHarmony preprocessing, tensor/split construction, database building, mutual information, and evaluation | Copied from the selected upstream `main` snapshot; historical paths and dependencies remain |
| `historical_training/` | Full-data and bootstrap training scripts, trainers, schedulers, evaluators, and sampling classes | Recovered from a prior revision; not a complete runnable legacy environment |
| `cuda/` | Tensor-building CUDA extension | Optional historical acceleration; not built or used by the new workflow |

Consult `provenance.json` for each original path and revision. These scripts can write large datasets, replace output directories, and assume substantial memory or GPU resources. Read them before adapting them. The tested commands are in the root README, and the differences from the historical recipe are documented in `docs/reproduction.md`.
