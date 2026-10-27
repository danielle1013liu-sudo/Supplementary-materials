# Installation and reproduction

Use Python 3.13; reference checks used Python 3.13.9. Run commands from the repository directory in an activated virtual environment. Editable installation keeps the data and configuration paths available.

```sh
python -m pip install -e ".[test]"
python -m hydrogel validate-data
python -m pytest
```

Dependencies are specified in [pyproject.toml](../pyproject.toml) and [requirements.txt](../requirements.txt).

## Application and saved models

```sh
python -m streamlit run app.py
python -m streamlit run deploy/streamlit_app.py
python -m hydrogel predict --models deployment_assets/models/tnf --input data/cases/c4.csv --output runs/c4_tnf_predictions.csv
python -m hydrogel predict --models deployment_assets/models/il10 --input data/cases/c4.csv --output runs/c4_il10_predictions.csv
```

Either application entry point loads the saved Training-fitted models and diagnostics without fitting models. Results include mean ± SD, grouped SHAP contributions and applicability-domain status. The domain requires both a mean five-neighbour distance within its recorded threshold and categorical levels represented in Training.

Input CSVs use the ten columns and categorical codes in [data/feature_schema.json](../data/feature_schema.json). Empty numerical fields indicate missing values; unknown categories are rejected. Only load trusted model files.

## Reproduce the analyses

Settings are recorded in [config/tnf.json](../config/tnf.json) and [config/il10.json](../config/il10.json); partitions and fold assignments are in [data/splits](../data/splits).

```sh
python -m hydrogel compare --target all --output runs/four_models --check-reference
python -m hydrogel compare --target all --all-imputers --output runs/imputation_comparison --check-reference
python -m hydrogel train --target all --output runs/selected_models --check-reference
python -m hydrogel cases --target all --output runs/p32 --check-reference
python -m hydrogel verify --run runs/four_models
```

The comparison, training and case commands fit new models and write new outputs without replacing saved assets. Preprocessing is fitted within each CV training fold; Validation metrics use pooled out-of-fold Training predictions. Summaries report means and sample SDs of individual model metrics. Verification compares outputs with [reference](../reference).

Training, Test and C4 predictions use Training-fitted models. P32 remains in the main partition, but its separate case analysis excludes DOI `10.1186/s12951-024-02810-z`, leaving 148 TNF-α and 129 IL-10 observations. Memberships are recorded in `data/splits/*_p32.json`. The GelMA control's missing raw modulus is processed by the configured imputer.

Use new output paths. From another directory, place `--root /path/to/repository` before the subcommand.
