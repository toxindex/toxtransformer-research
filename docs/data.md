# Data preparation and evaluation splits

## Supported CSV contract

Each row is one binary observation:

```csv
smiles,property_id,value,split
CCO,example_assay,1,train
CCN,example_assay,0,validation
CCCO,example_assay,1,test
```

This snippet illustrates the schema; the included `examples/toy.csv` provides a complete runnable input. `property_id` is a string identifier. Identifiers are sorted to create a deterministic zero-based mapping, saved as `properties.json`. Vocabulary and property mappings are fitted on the training partition only.

The reader canonicalizes SMILES with RDKit while preserving stereochemistry, then groups observations by canonical SMILES. Identical repeated observations are deduplicated. Conflicting labels for the same compound/property are rejected. The same canonical structure cannot occur in multiple partitions.

The workflow does not perform salt removal, tautomer normalization, endpoint harmonization, or activity threshold selection. Those decisions belong in a documented preprocessing step before this CSV. Binary values should carry a documented scientific meaning for each property.

## Choose the scientific split before training

Supply split assignments explicitly. For generalization to new compounds, use a compound-disjoint split. For a scaffold-generalization claim, construct and audit a scaffold-disjoint split before export. The reader checks exact canonical structure overlap; it does not prove scaffold separation or collapse alternate protonation and tautomer forms.

The supported workflow deliberately differs from the historical per-property observation splits. Historical observation-level splits can put different measurements of the same molecule into different partitions. Such a benchmark measures a different form of generalization.

Validation selects the checkpoint by structure-only binary cross-entropy. Test rows never enter the training loss or checkpoint selection. Unknown validation/test properties or SELFIES symbols cause an error rather than changing the training vocabulary. Molecular sequences and property lists exceeding the configured limits also cause an error; there is no silent truncation.

## Training and context

Training groups all supplied labels for a compound and shuffles their order using a seeded generator each epoch. Cross-entropy is averaged over valid observed property positions. Later queries can use earlier observed labels through teacher forcing. Molecules with more observations contribute more terms to the loss.

Default evaluation is structure-only. `--context N` evaluates every target using up to N other observed labels from the same held-out molecule, ordered by property index. The target itself is excluded. This reference context policy does not implement the deployed service's mutual-information ranking. Reports include the requested context count and the mean count actually supplied.

Per-property ROC AUC is undefined when a partition contains only one class for that property. The report records `null` and excludes that property from macro AUC, while reporting how many properties were excluded. Counts, class counts, log loss, Brier score, and accuracy accompany AUC.

## Historical preprocessing evidence

`reference/upstream/` retains the extracted preprocessing and evaluation source. The upstream BioBricks dependency file points to ChemHarmony commit `dd82369101d35d7e84601edc2f78d1f2b761e5d9`. This is a recorded dependency reference; its correspondence to the inspected checkpoint's actual training data has not been verified.

The source includes these decisions:

- `03_preprocess_activities.py` requires at least 100 positive and 100 negative observations per property. Its implemented imbalance threshold is `abs(n0 - n1) / (n0 + n1) <= 0.9`.
- `05_build_tensordataset.py` removes conflicting compound/property labels and excludes `ctdbase`. It constructs five observation-level splits using seeds 42 through 46 and per-class test targets.
- Later stages consume `activities_augmented.parquet` and `final_tensors`, whose complete production lineage is not established by the selected `main` snapshot.
- The original Spark tokenizer collected a distributed vocabulary without sorting. Exact reproduction requires the released tokenizer files rather than assuming a newly fitted vocabulary has the same indices.

The repository includes no measured toxicology dataset. Before publishing results, record the input source versions, terms of use, standardization rules, endpoint definitions, label thresholds, duplicate/conflict handling, split assignments, and all file checksums.
