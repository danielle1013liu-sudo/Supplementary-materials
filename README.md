# Hydrogel cytokine prediction and Pareto analysis

Research data and saved models for predicting TNF-alpha and IL-10 from ten hydrogel and experimental descriptors, with supplementary tables and Pareto-analysis replay.

## Current versions

| Component | Version / configuration |
| --- | --- |
| Canonical datasets | Thermal-harmonised v2; TNF-alpha: 139 records, IL-10: 133 |
| Publication category codes | `frequency_codes_20261019`, shared across both targets |
| TNF-alpha model | GradientBoostingRegressor; median imputation; seeds 3089 / 4280 / 5793 |
| IL-10 model | RandomForestRegressor; iterative Bayesian ridge imputation; seeds 85 / 115 / 158 |
| Verified environment | Windows; Python 3.13.9; scikit-learn 1.7.2; NumPy 2.3.5; pandas 2.3.3; SHAP 0.49.1 |

Use the exact dependencies in [requirements.txt](requirements.txt); the complete environment is recorded in [software_versions.json](data/software_versions.json).

## Use

From the repository root in the pinned Python environment:

```sh
python -m pip install -r requirements.txt
python -m src.public_predict data/tnf.csv --target tnf --output results/tnf_predictions.csv
python -m src.hydrogel_immune_ml.publication_audit
```

For IL-10, use `data/il10.csv` and `--target il10`. These commands load saved models without fitting. Canonical machine values are the default input; letter-coded model exports require `--codebook-version frequency_codes_20261019`.

## Data and supplementary material

- [Revised supplementary document](docs/Supplementary_frequency_ordered_20261019.docx)
- [Category codebook](data/category_codebook.csv), [old/new mapping](data/public_codebooks/category_code_changes_20261019.csv) and [publication CSVs](data/publication/)
- [Data dictionary](docs/data_dictionary.md), [replay instructions and verification details](docs/reproducibility.md), and [verification results](results/publication_codes_20261019/)

Publication letters are nominal labels ordered by pooled target-record frequency, with residual categories last; crosslink-level A/B/C means medium/high/low. Scientific assignments, canonical data and fitted models are unchanged. Supplementary CSVs retain display precision; use canonical data or `*_model_codes_20261019.csv` for prediction.

Results are retrospective, not independent confirmation. [Known cross-target descriptor differences](docs/cross_target_descriptor_notes.md) remain documented; P15 uses separate saved models. Source attribution is retained in [references](data/references.csv) and [record provenance](data/record_identity.csv). No repository licence is currently supplied.
