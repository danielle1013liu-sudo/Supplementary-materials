# Installation and reproduction

Use Python 3.13; reference checks used Python 3.13.9. Run commands from the repository directory in an activated virtual environment. Editable installation keeps the data and configuration paths available.

```sh
python -m pip install -e ".[test]"
python -m hydrogel validate-data
python -m pytest
node --test tests/frontend/*.test.js
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

Predictions and performance metrics are summarised as mean ± SD across repeated fits of each configuration (n = 3), with data partitions held fixed. Performance metrics are calculated separately for each fit before their means and sample SDs (`ddof=1`) are calculated. SD describes variation between fitted predictions or metrics, not experimental variability or a confidence interval.

Input CSVs use the ten columns and categorical codes in [data/feature_schema.json](../data/feature_schema.json). Empty numerical fields indicate missing values; unknown categories are rejected. Only load trusted model files.

Numerical descriptors accept finite, non-negative values or explicitly missing measurements; imputation remains in the fitted pipeline. The interface's downloadable descriptor guide contains current definitions, units and category mappings, without historical workflow metadata. Its prediction download summarises each cytokine; detailed results remain available from the inference API and reproduction commands.

## Design Screening

Open **Design Screening**, fix any experimental conditions and optionally restrict material categories or inclusive numerical ranges, then select **Screen candidate profiles**. Restrictions apply exactly to the existing grid; values are neither rounded to neighbouring grid points nor relaxed. Load a retained profile into **Model Prediction** to use its exact descriptors with the saved inference and SHAP functions.

The complete library is stored losslessly in [deployment_assets/screening](../deployment_assets/screening), with source and asset checksums, category mappings and individual predictions recorded in its manifest. Eligibility requires both target applicability domains, categorical support and non-negative individual predictions. Within the matching eligible subset, screening minimises TNF-α and maximises IL-10. It retains the mean-prediction front that is also non-dominated in at least six of the nine recorded target-model pairings. Exact objective ties are retained; no additional tolerance is introduced.

The figure may sample background candidates for display; all candidates are used for selection. Downloads contain every retained profile and its active constraints. Conditional profiles keep their candidate IDs and do not replace the canonical global profiles or P1–P3. With no constraints, the tests reproduce the original 13 retained IDs. Library loading and compact results are cached by the library version and restrictions.

## Reproduce the analyses

Settings are recorded in [config/tnf.json](../config/tnf.json) and [config/il10.json](../config/il10.json); partitions and fold assignments are in [data/splits](../data/splits).

```sh
python -m hydrogel compare --target all --output runs/four_models --check-reference
python -m hydrogel compare --target all --all-imputers --output runs/imputation_comparison --check-reference
python -m hydrogel train --target all --output runs/selected_models --check-reference
python -m hydrogel cases --target all --output runs/p32 --check-reference
python -m hydrogel verify --run runs/four_models
```

The comparison, training and case commands fit new models and write new outputs without replacing saved assets. Preprocessing is fitted within each CV training fold; Validation metrics use pooled out-of-fold Training predictions. Verification checks predictions and individual metrics against [reference](../reference), and requires `summary.csv` to contain all configuration/scope rows with the corresponding means and sample SDs.

Training, Test and C4 predictions use Training-fitted models. P32 remains in the main partition, but its separate case analysis excludes DOI `10.1186/s12951-024-02810-z`, leaving 148 TNF-α and 129 IL-10 observations. Memberships are recorded in `data/splits/*_p32.json`. The GelMA control's missing raw modulus is processed by the configured imputer.

Use new output paths. From another directory, place `--root /path/to/repository` before the subcommand.
