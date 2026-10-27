# Reproducing the recorded analyses

Run commands from the repository directory. Python 3.13.9 and the versions in `requirements.txt` were used for numerical verification. Install the project in editable mode so that its data and configuration directories remain available:

```sh
python -m pip install -e '.[test]'
python -m hydrogel validate-data
python -m pytest
```

## Model comparison

Each family uses the configuration in `config/tnf.json` or `config/il10.json`. The ordered Training and Test identities and Training fold assignments are stored in `data/splits/`. Preprocessing is fitted separately on each CV training fold. Validation metrics are calculated from pooled out-of-fold predictions for all Training observations.

```sh
python -m hydrogel compare --target all --output runs/four_models --check-reference
python -m hydrogel compare --target all --all-imputers --output runs/imputation_comparison --check-reference
```

The first command evaluates each model with its recorded imputation method. The second evaluates all five imputation methods with the same estimator parameters. Neither command changes the configuration files or partitions.

Each run writes `predictions.csv`, `metrics.csv` and `summary.csv`. The summary reports the arithmetic mean and sample standard deviation (`ddof=1`) of the three model-level metrics. These values differ from metrics calculated after averaging the three prediction vectors. Reference checking compares individual predictions and metrics with the numerical tables in `reference/`.

## Selected models and case studies

```sh
python -m hydrogel train --target all --output runs/selected_models --check-reference
python -m hydrogel cases --target all --output runs/p32 --check-reference
```

The Training, Test and C4 predictions use the same Training-fitted model in each realisation. C4 consists of HPA@SIN-GM and HPA@SIN-GM@DR. P32 observations remain eligible for the main Training/Test partition.

The P32 case analysis uses separate models fitted after excluding every observation from DOI `10.1186/s12951-024-02810-z`. The exact reference populations contain 148 TNF-alpha and 129 IL-10 observations. Their ordered identities are recorded in `data/splits/tnf_p32.json` and `data/splits/il10_p32.json`. The missing raw storage modulus for the GelMA control is processed by the configured imputer.

Both commands save three-model bundles under `models/<target>/<family>/<imputer>/` within their output directories. They create new outputs and do not replace the reference results.

## Prediction

Input CSV files must contain all ten descriptor columns and use the categorical codes in `data/feature_schema.json`. Numerical missing values may be represented by empty fields. Unknown categorical levels are rejected.

```sh
python -m hydrogel predict --models runs/selected_models/models/tnf/RF/median --input data/cases/c4.csv --output runs/c4_tnf_predictions.csv
python -m hydrogel verify --run runs/four_models
```

Prediction output contains the mean and sample standard deviation across the three saved models. When an input table includes a `target` column, only rows matching the saved model's target are used. Load only model bundles created by this project or another trusted source.

Use the global `--root` argument before the command to specify a repository directory from another working location. Output directories and prediction files must be new paths.

## Web application

Install `requirements-app.txt` and run `python -m streamlit run app.py` from the repository directory. The `deploy/streamlit_app.py` entry point starts the same application for Community Cloud. The tested runtime is Python 3.13.

The application loads the six saved Training-fitted models in `deployment_assets/`; it does not fit models when a user submits a formulation. Its categorical options come directly from the current feature schema. Results include the three individual predictions, their mean and sample standard deviation, grouped SHAP contributions and an applicability-domain assessment using the saved Training reference populations. The domain assessment requires both a mean five-neighbour distance within the recorded threshold and categorical levels observed in the target's Training population.

Model and diagnostic assets are checked against their recorded hashes. Numerical predictions are checked against the selected configurations in `reference/` when the resources are loaded. Input values and results can be downloaded directly from the interface.
