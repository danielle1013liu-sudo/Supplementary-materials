# Numerical reproducibility

This model-only submission retains the approved thermal-harmonised v2 input bytes, exact split and CV definitions, fixed preprocessing and all six accepted fitted pipelines. It does not change scientific values or perform further model selection.

## Scientific locks

| Input | SHA-256 |
| --- | --- |
| TNF-alpha dataset | 289d85de602cf9277ba2cbce368439db2019df78b58bada7200fa9e855397429 |
| IL-10 dataset | 8aef2ec5bcae3b966559531c8ecbda9ec939a68686fa300d4a1b70293d03a018 |
| External cases | 8615aca1a5935ee98ee7feeca0f8fd585beef8a55fb547ce313443eb9ccb0406 |
| Ten-feature schema | 2a285420024a61d0aed54b51e06917daca6f56527599b8b8061d6b66f95e6720 |
| Archived legacy category codebook | b10ae08d8fb5c7a83d8b9099d1cebb23851f574abfbf93aa1b409fa6c483fe00 |

`data/checksums.json` locks all scientific inputs included in this repository, including the six fitted objects. The complete accepted parameter file is unchanged. Only the manifest's scope is reduced to the files distributed in this model-only package.

The 19 October 2026 publication-code revision changes only the letters and presentation order in `data/category_codebook.csv`; its current hash is recorded in `data/checksums.json`. The original file is retained byte-for-byte as `data/public_codebooks/legacy_codes_v1.csv`, and Pareto verification checks it against the unchanged historical scientific lock. Current presentation metadata and publication exports have their own checksum entries. Definitions, boundaries, caveats, counts, canonical datasets, scientific reference outputs and fitted objects remain unchanged.

## Executable verification

`python -m src.verify` performs the following without fitting models:

1. Verify input/model hashes and pinned software versions.
2. Reconstruct the 111/28 TNF and 106/27 IL-10 memberships from split seeds 1975 and 1376, requiring the exact saved order.
3. Reconstruct shuffled ten-fold assignments from CV seeds 8 and 24, requiring exact saved membership.
4. Compare every saved estimator/preprocessor parameter with its reconstructed fixed pipeline definition.
5. Generate Test and external-case predictions from each same saved Training-only object and require exact equality with accepted arrays.
6. Check stored pooled OOF predictions and recomputed Validation/Test metrics against the accepted reference and previous fixed-training records.
7. Check the complete archived 20-competitor Validation table at each final seed and confirm the accepted pipeline is the highest-ranked model/imputer combination.
8. Calculate mean R2 and sample SD across the same three seeds per target.

Exact means elementwise equality without numerical tolerance. Prediction outputs preserve raw values. The package was checked in the pinned Windows environment; cross-platform exactness is not assumed.

## Fixed training and optional grid

The previously completed fixed-seed training reproduction produced exact Training, pooled OOF Validation, Test and case predictions for all six pipelines. Its numerical evidence is retained in `outputs/validation/reproduction.json`, the two fixed-seed reproduction CSVs and OOF tables. `python -m src.train` can repeat this check using only the accepted Training IDs and six final seeds. It writes new fitted files separately and cannot replace the distributed accepted objects.

`src/grid_search.py` is an executable refactoring of the declared model-specific local comparison. It uses fold-fitted preprocessing, identical saved folds, pooled OOF ranking and only the target's three final estimator seeds. It is not executed by default. Historical comparison results, final all-20 certification tables and previously completed controlled-grid reproduction evidence are distributed without rerunning a grid for this upload. See `grid_search.md` for the distinction between historical reference-seed evidence and optional final-seed execution.

The upload preparation rechecked saved-model prediction, stored OOF and pipeline construction. It did not rerun training, grid comparison or seed selection. Machine-specific runtime provenance is generated locally in an ignored directory.

## Scope of inference

Validation, Test and case calculations use one accepted pipeline definition per target, with estimator seed as the only difference among its three final models. CV fits each preprocessing step within the fold; the final objects use the complete fixed 80% Training subset only. Test and case performance remains split-, Test-, and case-conditioned exploratory selection, not independent confirmation.
