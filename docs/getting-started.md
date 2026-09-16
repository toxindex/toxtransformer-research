# Start here: study and integrate ToxTransformer

## Before a first research meeting

Start with the [README](../README.md), [architecture](architecture.md), and [model card](../MODEL_CARD.md). Then follow [local reproduction](local-reproduction.md) to download the Git LFS weights, install the locked environment, and run the recorded prediction checks. Reading the entire repository is unnecessary preparation.

For a first code walkthrough, read [the checkpoint inference helper](../src/toxtransformer_research/checkpoint.py), then [the model](../src/cvae/multitask_encoder.py). Follow the tokenizer links in the README when tracing how SELFIES and property/value pairs become model inputs. Read [the data contract](data.md) and [training/reproduction notes](reproduction.md) before designing a new experiment. Historical source in `reference/` is supplementary material.

Bring an endpoint you want to study, its source label definition, and a proposed compound-level evaluation split. The bundled checks reproduce checkpoint inference and a small artificial training experiment. Exact historical retraining and scientific benchmark results remain unverified.

## Interpret a prediction

All 6,647 endpoints in this released checkpoint are binary. The reported probability is the model's probability of class 1. Class 1 means the positive label used to construct that endpoint; the original assay definition and label threshold determine whether that means active, toxic, or another property. Do not infer a universal hazard direction from the number alone. Calibration and an applicability domain have not been established by this release.

| Local checkpoint CLI JSON field | Meaning |
| --- | --- |
| `checkpoint_sha256` | Checkpoint identity |
| `prediction.smiles` | Query structure |
| `prediction.property_index` | Zero-based index in this checkpoint's catalog |
| `prediction.property` | Catalog entry with `index`, `property_id`, `source`, and nullable `title` |
| `prediction.probability` | Class-1 probability, between 0 and 1 |
| `prediction.n_context` | Number of supplied observed property labels |
| `prediction.selfies_tokens`, `prediction.selfies_padding` | Molecular token count and padding convention |

The Python `predict_checkpoint` function returns the inner prediction object without the catalog entry. Load `models/toxtransformer/properties.json` to join its `property_index` to the catalog's `index`. Use `--list-properties --search ...` to find endpoint identifiers. Missing titles remain null; no complete assay-definition dictionary is included.

Record the Git revision, manifest checksum, structure preprocessing, endpoint identifier, and exact observed-context pairs and order with each experiment. An endpoint's numeric index is specific to its model vocabulary. Match source definitions before comparing an endpoint with Pleiome or a hosted tool.

## Public projects and hosted services

Both research repositories are public on their `main` branches. No GitHub invitation or ToxIndex account is required to read the code or download the released weights. Earlier links to `toxindex/toxtransformer` and `toxindex/pleiome` refer to private development repositories and may return 404. Use the research URLs for public documentation and local experiments.

The released checkpoint is identified by its artifact manifest. A research checkout does not establish which revision, preprocessing, property context, or output adapter a hosted service is currently using. Compare those settings explicitly before expecting local and hosted predictions to agree.

| Resource | Purpose |
| --- | --- |
| [ToxTransformer research](https://github.com/toxindex/toxtransformer-research) | SELFIES-based model with 6,647 binary property endpoints |
| [Pleiome research](https://github.com/toxindex/pleiome-research) | Graph-based model with binary/numeric prediction and a SELFIES generation architecture |
| [ToxIndex](https://toxindex.com/) | Product overview and a public link for model directories |
| [Insilica](https://insilica.co/) | Company background |
| [ToxIndex platform](https://platform.toxindex.com/) | Hosted application |
| [Gateway documentation](https://gateway.toxindex.com/docs) and [OpenAPI](https://gateway.toxindex.com/openapi.json) | Hosted prediction interfaces; separate from these local Python interfaces |

Suggested directory description: “ToxIndex provides access to chemical and toxicological data, prediction tools, and scientific literature through AI-assisted search and workflows.” Link this description to https://toxindex.com/; use the platform URL for an application sign-in link.

## License and hosted access

The research code and released weights use the repository's MIT license and can be downloaded for local use without a hosted subscription. Original source datasets retain their own terms. Local execution requires your own compute resources.

Hosted access is governed separately. The [ToxIndex pricing page](https://toxindex.com/pricing) describes a free platform preview and paid plans quoted for the deployment and scope of work. It does not promise free ToxTransformer/Pleiome API calls or establish partner-specific terms. Consult the current page and the account's agreement for hosted access.

For an existing hosted ToxTransformer integration, create an API key at [account settings](https://platform.toxindex.com/settings/keys). Send `Authorization: Bearer $TOXINDEX_API_KEY` on both `POST https://gateway.toxindex.com/v1/runs/toxtransformer` with JSON `{"smiles":"CCO"}` and subsequent `GET /v1/runs/{run_id}` polls. The run statuses are `queued`, `running`, `completed`, and `failed`; read `result` on completion and `error` on failure. Keep the key outside the repository. Use the live gateway documentation for the hosted response schema and current access rules. This release does not specify every other model available through ToxIndex.
