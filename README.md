# Hydrogel immunomodulation: supplementary data and modelling code

This repository accompanies the manuscript analysis of hydrogel-associated TNF-alpha and IL-10 concentrations. It contains the approved thermal-harmonised v2 datasets, six accepted Training-only fitted pipelines, executable modelling code and numerical reproducibility records. The default workflow loads the accepted models. Optional training and local-grid comparison are separate commands.

## Accepted models

| Target | Model and numerical imputation | Final estimator seeds | Split seed | CV seed | Training / Test |
| --- | --- | --- | --- | --- | --- |
| TNF-alpha | GradientBoostingRegressor + median | 3089, 4280, 5793 | 1975 | 8 | 111 / 28 |
| IL-10 | RandomForestRegressor + IterativeImputer(BayesianRidge) | 85, 115, 158 | 1376 | 24 | 106 / 27 |

TNF-alpha uses `n_estimators=1000`, `learning_rate=0.08`, `max_depth=3`, `min_samples_leaf=8` and `subsample=0.7`. IL-10 uses `n_estimators=250`, `max_depth=9`, `max_features=0.7`, `min_samples_leaf=1`, `bootstrap=False` and `n_jobs=1`. The IL-10 iterative imputer uses `random_state=42`, BayesianRidge and no posterior sampling.

All remaining estimator, imputer, encoder and column-transformer parameters are fixed in [data/model_parameters.json](data/model_parameters.json). The model hashes are recorded in [models/manifest.json](models/manifest.json). No historical model family is substituted for either accepted model.

For each seed, the **same fitted Training-only pipeline object predicts the held-out Test rows and both external formulations**. It is never refitted on Test, the external cases or the complete literature dataset. Validation uses clones of that same pipeline definition within the saved ten folds, fitting every preprocessing step only on the fold's Training observations.

## Data and fixed memberships

The approved datasets contain 139 TNF-alpha and 133 IL-10 observations. Both use the same ordered ten-feature schema and category codebook. Dataset, model, parameter and reference hashes are checked before execution.

| Input | SHA-256 |
| --- | --- |
| `data/tnf.csv` | `289d85de602cf9277ba2cbce368439db2019df78b58bada7200fa9e855397429` |
| `data/il10.csv` | `8aef2ec5bcae3b966559531c8ecbda9ec939a68686fa300d4a1b70293d03a018` |
| `data/external_cases.csv` | `8615aca1a5935ee98ee7feeca0f8fd585beef8a55fb547ce313443eb9ccb0406` |

The authoritative ordered Training/Test IDs, ten-fold assignments and membership hashes are in [data/splits](data/splits). `src.verify` independently reconstructs the row split and shuffled ten-fold assignment from the declared seeds and requires exact agreement with these saved definitions. These are row-based splits and folds, not study-grouped holdouts.

Missing descriptor values, all high IL-10 responses and P44 observations are retained. The external HPA@SIN-GM and HPA@SIN-GM@DR inputs retain RAW264_7, physical_self_assembly and the approved B-to-C bioactive-cue contrast. Feature definitions, units and case inputs are unchanged. See [docs/data_dictionary.md](docs/data_dictionary.md).

## Environment

The verified environment is Python 3.13.9 with the exact package versions in [requirements.txt](requirements.txt). The full accepted numerical environment is retained, including packages used by the original release; this repository contains no plotting code.

```text
conda env create -f environment.yml
conda activate hydrogel-supplement
```

Alternatively, in a Python 3.13.9 environment:

```text
python -m pip install -r requirements.txt
```

Run commands from the repository root. Do not use Python's `-O` option: integrity checks use assertions. The six supplied compressed pickle files are the accepted fitted objects; their SHA-256 hashes are verified before loading. No model attributes are patched for compatibility. Exact numerical reproduction has been checked on the pinned Windows environment; another platform must pass the same checks before its results are called exact.

## Default reproduction: no training or search

```text
python -m src.predict
python -m src.verify
```

Prediction exports retain individual-seed predictions, observed concentrations, fit membership, mean predictions and sample SD across the three fixed seeds. Verification checks the saved model and reconstructed pipeline parameters, exact split/fold reconstruction, stored pooled OOF predictions, metrics, final same-seed Validation evidence and freshly generated Test/case predictions against the accepted arrays.

Outputs are written to `outputs/predictions/`, `outputs/validation/panel_summary.csv` and `outputs/verification.json`. Detailed local runtime provenance is written to the ignored `outputs/runtime_provenance/` directory; machine-specific paths are not published.

## Optional fixed-seed training

```text
python -m src.train --target tnf
python -m src.train --target il10
```

Each command reconstructs the accepted model at only its three final seeds. It fits the exact saved Training IDs and reconstructs ten-fold pooled OOF Validation predictions. Test and both external cases are predicted together by the same Training-only fitted object. Elementwise equality with accepted Validation, Test and case arrays is required; Training predictions are also compared with the distributed fitted models. Reproduced objects are written to the ignored `outputs/retrained_models/` directory, leaving accepted model files unchanged.

The included `outputs/validation/reproduction.json` and per-target reproduction tables record the previously completed exact fixed-seed refit check. Default verification checks the stored OOF evidence without silently running these fits again.

## Optional model-specific local-grid comparison

```text
python -m pip install -r requirements-grid.txt
python -m src.grid_search --target tnf
python -m src.grid_search --target il10
```

The executable comparison crosses four model families with five approved numerical imputers and each family's declared local hyperparameter candidates. Every comparison uses the same Training rows and saved ten folds. Imputation and encoding are fitted within each fold. Candidates are ranked by unrounded pooled Validation R2, then RMSE and MAE. Test and case outcomes do not enter this comparison.

The optional command uses only the three accepted estimator seeds for its target. It is not called by prediction or verification and does not modify accepted parameters. There is no estimator-seed screening code. The historical grid results and complete all-20 Validation tables for the six final seeds are included in `data/validation_evidence/`. The distinction between historical reference-grid evidence and optional final-seed reruns is documented in [docs/grid_search.md](docs/grid_search.md).

## Reported performance and interpretation

| Target | Pooled Validation R2, mean ± sample SD | Test R2, mean ± sample SD |
| --- | --- | --- |
| TNF-alpha | 0.718182 ± 0.007108 | 0.730619 ± 0.011119 |
| IL-10 | 0.778003 ± 0.001186 | 0.967895 ± 0.000132 |

Each mean and sample SD uses the same three final seeds. Full-precision individual metrics and predictions are supplied. Validation R2 is calculated from pooled OOF predictions, not averaged fold R2 scores.

The reported Test and external-case performance reflects **split-, Test-, and case-conditioned exploratory selection**. It must not be described as untouched independent validation. The saved predictions and all observed values are retained without clipping or exclusions.

## Repository contents

```text
data/       Approved data, cases, codebook, fixed parameters, splits/folds and grid evidence
models/     Six accepted fitted Training-only pipelines and their hash manifest
src/        Loading, preprocessing, validation, optional grid, fixed training and prediction
outputs/    Accepted predictions, metrics, fixed-training evidence and verification results
docs/       Data dictionary, grid methods, reproducibility record and complete file manifest
```

[docs/files.txt](docs/files.txt) lists every file in the submitted repository. Plotting scripts, figures, exploratory seed-screening code, schedulers, monitors and task artefacts are excluded. The earlier repository contents remain available in Git history.
