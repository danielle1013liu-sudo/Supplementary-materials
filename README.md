# Hydrogel cytokine prediction

Code, data and fitted models for *Supervised machine learning for predicting and explaining macrophage cytokine responses to hydrogels*.

| Component | Version |
| --- | --- |
| Data | `thermal_harmonisation_v2`; 139 TNF-alpha and 133 IL-10 records |
| Publication codes | `frequency_codes_20261019` |
| Models | `accepted_train80_release_v1`; TNF-alpha: GBR; IL-10: RF |
| Runtime | Python 3.13.9; scikit-learn 1.7.2; NumPy 2.3.5; pandas 2.3.3; SHAP 0.49.1 |

| Analysis | Files |
| --- | --- |
| GBR, XGB, RF and LGBM Validation results | [Validation table](results/model_comparison/validation.csv) |
| Primary-model Test results | [Test table](results/model_comparison/test.csv) |
| SHAP and t-SNE | [SHAP](outputs/shap), [coordinates](outputs/tsne), [figures](outputs/figures) |
| C4 and P15 case studies | [Case results](outputs/case_studies) |
| Pareto analysis | [Profiles and figures](results/pareto) |

The [supplementary document](docs/Supplementary_frequency_ordered_20261019.docx) and [publication CSVs](data/publication) share one codebook. Canonical data retain machine values; saved one-hot categories and column order are unchanged. Crosslink-level A/B/C mean medium/high/low, respectively.

Run from the repository root with Python 3.13.9:

```sh
python -m pip install -r requirements-test.txt -r requirements-plots.txt
python -m src.hydrogel_immune_ml.publication_audit
python -m pytest tests --replay-dir tmp/publication_codes_20261019/pareto
python -m src.plot_interpretation
python -m src.pareto.plotting --output tmp/pareto_figures
```

These commands verify saved objects and numerical references without fitting models or t-SNE. Parameters, splits and folds are in [data/model_parameters.json](data/model_parameters.json), [models/manifest.json](models/manifest.json) and [data/splits](data/splits). Validation uses archived out-of-fold results for four model families; Test uses the two primary families. C4 and P15 are retrospective case analyses; Test and case outcomes informed configuration selection. P15 uses separate whole-study-excluded models.

For prediction with canonical or publication-coded inputs:

```sh
python -m src.public_predict data/tnf.csv --target tnf --output tmp/tnf_predictions.csv
python -m src.public_predict data/publication/tnf_model_codes_20261019.csv --target tnf --codebook-version frequency_codes_20261019 --output tmp/tnf_coded_predictions.csv
```

Letter-coded inputs require an explicit version (`frequency_codes_20261019` or `legacy_codes_v1`). Empty cells, literal `none` and historical missing-category tokens remain distinct.
